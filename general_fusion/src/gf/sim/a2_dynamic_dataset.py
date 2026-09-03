"""A2-DYN 数据生成的共享确定性工具。

本模块是动态 profile 实例化、连续轨迹重采样和观测扰动的唯一实现入口。
pipeline 只负责读取配置、编排场景和写出产物。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from gf.sim.a2_dynamic_physics import (
    SENSOR_IDS,
    apply_observation_chain,
    generate_ar1_noise,
    generate_shared_noise,
    linear_sequence_drift,
    build_inlet_composition,
    evaluate_shared_physics,
    protocol_inlet_coefficient,
    simulate_dynamic_layers,
)
from gf.sim.a2_sensor_devices import (
    build_reference_waveform,
    estimate_ultrasonic_quality_series,
    estimate_ultrasonic_tof_series,
    simulate_ndir,
    simulate_tcd,
)


CANONICAL_NOISE_DT_S = 0.5
_LOW_DISCREPANCY_MULTIPLIER = 0.6180339887498949
_DYNAMIC_MANIFEST_SCHEMA_VERSION = "gf-a2-dynamic-manifest-1"
_DYNAMIC_RECORD_SCHEMA_VERSION = "gf-a2-dynamic-record-1"
_DYNAMIC_SPLITS = ("train", "val", "stress_val")
_TEST_SPLIT = "test"
_ALL_DYNAMIC_SPLITS = _DYNAMIC_SPLITS + (_TEST_SPLIT,)
_DYNAMIC_FAMILIES = (
    "D-IID",
    "D-KINETICS",
    "D-PROTOCOL",
    "D-NOISE-DRIFT",
    "D-ENV-CAL",
    "D-JOINT",
)
_DYNAMIC_REGIONS = ("interior", "near_boundary", "binary")
_PURE_REGION = "pure"
_ALL_DYNAMIC_REGIONS = _DYNAMIC_REGIONS + (_PURE_REGION,)
_DYNAMIC_TARGET_TOTAL = 100.0
_DYNAMIC_PURGE = np.asarray([100.0, 0.0, 0.0], dtype=np.float64)


def _serialize_composition_float32(values: np.ndarray) -> np.ndarray:
    """序列化组成 oracle，并在 float32 语义下显式闭合总和。"""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 3 or array.size == 0:
        raise ValueError("composition array must have a non-empty final dimension of three")
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > _DYNAMIC_TARGET_TOTAL):
        raise ValueError("composition array must be finite and lie in [0,100] mol%")
    serialized = array.astype(np.float32)
    flat = serialized.reshape(-1, 3)
    first_two_sum = np.sum(flat[:, :2], axis=1, dtype=np.float32)
    for _ in range(4):
        over = first_two_sum > np.float32(_DYNAMIC_TARGET_TOTAL)
        if not np.any(over):
            break
        rows = np.flatnonzero(over)
        correction_indices = np.argmax(flat[rows, :2], axis=-1)
        flat[rows, correction_indices] = np.nextafter(
            flat[rows, correction_indices],
            np.float32(-np.inf),
        )
        first_two_sum = np.sum(flat[:, :2], axis=1, dtype=np.float32)
    else:
        raise ValueError("float32 composition serialization could not make the first two components non-overflowing")
    flat[:, 2] = np.float32(np.float64(_DYNAMIC_TARGET_TOTAL) - first_two_sum)
    serialized = flat.reshape(serialized.shape)
    totals = np.sum(serialized, axis=-1, dtype=np.float32)
    if not np.isfinite(serialized).all() or np.any(serialized < 0.0) or np.any(
        serialized > _DYNAMIC_TARGET_TOTAL
    ):
        raise ValueError("float32 composition serialization produced an invalid component")
    if not np.all(np.abs(totals - np.float32(_DYNAMIC_TARGET_TOTAL)) <= np.float32(1.0e-6)):
        raise ValueError("float32 composition serialization failed closure")
    return serialized


@dataclass(frozen=True)
class ProtocolInstance:
    """一条 observation 的已解析进气协议。"""

    parameters: Mapping[str, Any]
    exposure_onset_s: float
    exposure_end_s: float
    phase_jitter_pct: tuple[float, float]


@dataclass(frozen=True)
class ObservationPerturbationAudit:
    """观测扰动的实际实例参数，不进入模型输入。"""

    ar1_rho: float
    drift_strength_pct_dynamic_range_per_min: float
    drift_slope_per_step: np.ndarray
    gain: np.ndarray
    offset: np.ndarray


def choose_profile_id(value: Any, index: int) -> str:
    if isinstance(value, list):
        if not value:
            raise ValueError("profile list must not be empty")
        return str(value[index % len(value)])
    result = str(value)
    if not result:
        raise ValueError("profile ID must be non-empty")
    return result


def sample_registered_range(
    values: Sequence[float],
    index: int,
    *,
    distribution: str = "uniform",
    salt: int = 0,
    zero_probability: float = 0.0,
    minimum_nonzero: float | None = None,
) -> float:
    """按确定性低差异坐标从注册范围抽样。"""

    if len(values) != 2:
        raise ValueError("registered range must contain two values")
    lower, upper = float(values[0]), float(values[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError("registered range must be finite and increasing")
    if isinstance(index, bool) or int(index) != index or index < 0:
        raise ValueError("index must be a non-negative integer")
    if not 0.0 <= zero_probability < 1.0:
        raise ValueError("zero_probability must lie in [0,1)")
    fraction = (_LOW_DISCREPANCY_MULTIPLIER * (int(index) + int(salt) + 1)) % 1.0
    if lower == upper:
        return lower
    if zero_probability > 0.0 and fraction < zero_probability:
        if lower != 0.0:
            raise ValueError("zero_probability requires a range whose lower bound is zero")
        return 0.0
    if zero_probability > 0.0:
        fraction = (fraction - zero_probability) / (1.0 - zero_probability)
        if minimum_nonzero is None:
            raise ValueError("minimum_nonzero is required for zero-inflated sampling")
        lower = float(minimum_nonzero)
    normalized = str(distribution).strip().lower().replace("-", "_")
    if normalized == "uniform":
        return lower + fraction * (upper - lower)
    if normalized == "log_uniform":
        if lower <= 0.0 or upper <= 0.0:
            raise ValueError("log-uniform ranges must be strictly positive")
        return math.exp(math.log(lower) + fraction * (math.log(upper) - math.log(lower)))
    raise ValueError(f"unsupported registered distribution {distribution!r}")


def resolve_protocol_instance(
    protocol: Mapping[str, Any],
    transport: Mapping[str, Any],
    *,
    index: int,
) -> ProtocolInstance:
    """把协议范围和 phase jitter 解析为单条 observation 的固定实例。"""

    kind = str(protocol["kind"])
    base_onset = float(protocol.get("onset_s", 30.0))
    if kind == "shifted_onset":
        base_onset = sample_registered_range(
            protocol["onset_range_s"], index, salt=3
        )
    jitter_range = transport["phase_duration_jitter_pct"]
    onset_jitter_magnitude = sample_registered_range(jitter_range, index, salt=5)
    exposure_jitter_magnitude = sample_registered_range(jitter_range, index, salt=11)
    onset_jitter = onset_jitter_magnitude * (-1.0 if index % 2 else 1.0)
    exposure_jitter = exposure_jitter_magnitude * (-1.0 if (index // 2) % 2 else 1.0)
    onset = base_onset * (1.0 + onset_jitter / 100.0)
    if onset < 0.0:
        raise ValueError("phase jitter produced a negative exposure onset")

    parameters: dict[str, Any] = {"kind": kind, "onset_s": onset}
    if kind in {"ramp", "smooth_ramp"}:
        parameters["ramp_duration_s"] = sample_registered_range(
            protocol["ramp_duration_range_s"], index, salt=17
        )

    if kind in {"short_pulse", "incomplete_recovery", "shifted_onset"}:
        base_duration = sample_registered_range(
            protocol["exposure_duration_range_s"], index, salt=23
        )
        exposure_duration = base_duration * (1.0 + exposure_jitter / 100.0)
        minimum_duration = float(protocol.get("minimum_effective_exposure_s", 0.0))
        exposure_duration = max(exposure_duration, minimum_duration)
        parameters["exposure_duration_s"] = exposure_duration
        exposure_end = onset + exposure_duration
    elif kind == "multi_pulse":
        count_range = protocol["pulse_count_range"]
        pulse_count = int(count_range[index % len(count_range)])
        pulse_width = sample_registered_range(protocol["pulse_width_range_s"], index, salt=29)
        pulse_period = sample_registered_range(protocol["pulse_period_range_s"], index, salt=31)
        parameters.update(
            {
                "pulse_count": pulse_count,
                "pulse_width_s": pulse_width,
                "pulse_period_s": pulse_period,
            }
        )
        exposure_end = onset + (pulse_count - 1) * pulse_period + pulse_width
    else:
        nominal_end = float(protocol["exposure_end_s"])
        base_duration = nominal_end - base_onset
        exposure_end = onset + base_duration * (1.0 + exposure_jitter / 100.0)

    if exposure_end <= onset:
        raise ValueError("protocol instance exposure_end_s must be after exposure_onset_s")
    if kind in {"step", "ramp", "smooth_ramp", "shifted_onset"}:
        parameters["exposure_end_s"] = exposure_end
    if kind == "incomplete_recovery":
        parameters["recovery_residual"] = sample_registered_range(
            protocol["recovery_residual_range"], index, salt=37
        )
    return ProtocolInstance(
        parameters=MappingProxyType(parameters),
        exposure_onset_s=float(onset),
        exposure_end_s=float(exposure_end),
        phase_jitter_pct=(float(onset_jitter), float(exposure_jitter)),
    )


def resample_continuous_series(
    source_time_s: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    target_time_s: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """把连续值序列线性插值到精确目标时间戳。"""

    source_time = np.asarray(source_time_s, dtype=np.float64)
    target_time = np.asarray(target_time_s, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if source_time.ndim != 1 or source_time.size < 2 or np.any(np.diff(source_time) <= 0.0):
        raise ValueError("source_time_s must be a strictly increasing one-dimensional array")
    if target_time.ndim != 1 or target_time.size == 0 or np.any(np.diff(target_time) <= 0.0):
        raise ValueError("target_time_s must be a strictly increasing one-dimensional array")
    if array.shape[0] != source_time.size or not np.isfinite(array).all():
        raise ValueError("values must be finite and aligned with source_time_s")
    if target_time[0] < source_time[0] or target_time[-1] > source_time[-1]:
        raise ValueError("target_time_s must lie inside source_time_s")
    flattened = array.reshape(array.shape[0], -1)
    result = np.column_stack(
        [np.interp(target_time, source_time, flattened[:, column]) for column in range(flattened.shape[1])]
    )
    return result.reshape((target_time.size,) + array.shape[1:])


def calibration_physical_scales(calibration: Mapping[str, Any]) -> tuple[float, float, float]:
    physical = calibration.get("physical_scales")
    if not isinstance(physical, Mapping):
        raise ValueError("calibration physical_scales must be an object")
    values = tuple(
        float(physical[key])
        for key in ("acoustic_path_length", "tcs_response", "ndir_absorbance")
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("calibration physical scales must be finite and positive")
    return values


def apply_dynamic_observation_profile(
    clean: np.ndarray,
    *,
    noise_profile: Mapping[str, Any],
    calibration_profile: Mapping[str, Any],
    noise_base: Sequence[float] | np.ndarray,
    quantization: Sequence[float] | np.ndarray,
    dt_s: float,
    index: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, ObservationPerturbationAudit]:
    """按注册顺序应用标定、漂移、AR(1)、共享噪声、白噪声和量化。"""

    signal = np.asarray(clean, dtype=np.float64)
    base = np.asarray(noise_base, dtype=np.float64)
    resolution = np.asarray(quantization, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != len(SENSOR_IDS):
        raise ValueError("clean must have shape (time, sensor)")
    if base.shape != (len(SENSOR_IDS),) or resolution.shape != base.shape:
        raise ValueError("noise_base and quantization must cover every sensor")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an explicit numpy.random.Generator")
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    scale = float(noise_profile["white_noise_scale"])
    canonical_rho = sample_registered_range(noise_profile["ar1_rho_range"], index, salt=41)
    rho = canonical_rho ** (dt / CANONICAL_NOISE_DT_S)
    individual_ar1 = np.column_stack(
        [
            generate_ar1_noise(
                signal.shape[0],
                rho=rho,
                innovation_std=float(base[channel]) * scale,
                rng=rng,
            )
            for channel in range(len(SENSOR_IDS))
        ]
    )
    white = rng.normal(0.0, base * scale, size=signal.shape)
    shared_load = sample_registered_range(
        noise_profile["shared_correlation_load_range"], index, salt=43
    )
    shared_vector = np.asarray(
        noise_profile.get("correlation_vector", [0.5, 0.25, 0.75]),
        dtype=np.float64,
    )
    shared_vector = shared_vector * base / max(float(np.max(base)), 1.0e-12)
    shared = generate_shared_noise(
        signal.shape[0],
        rho=rho,
        innovation_std=float(np.max(base) * scale * shared_load),
        channel_loadings=shared_vector,
        rng=rng,
    )
    correlated = individual_ar1 + shared

    drift_strength = sample_registered_range(
        noise_profile["drift_strength_range_pct_dynamic_range_per_min"],
        index,
        salt=47,
    )
    dynamic_range = np.maximum(np.ptp(signal, axis=0), base)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(SENSOR_IDS))
    slope_per_step = (
        signs * dynamic_range * (drift_strength / 100.0) * dt / 60.0
    )
    drift = linear_sequence_drift(
        signal.shape[0],
        intercept=rng.normal(0.0, base * scale * 0.2),
        slope_per_step=slope_per_step,
    )

    gains_raw = calibration_profile.get("sensor_gains")
    offsets_raw = calibration_profile.get("sensor_offsets")
    if not isinstance(gains_raw, Mapping) or not isinstance(offsets_raw, Mapping):
        raise ValueError("calibration profile must contain sensor_gains and sensor_offsets")
    gain = np.asarray([float(gains_raw[sensor_id]) for sensor_id in SENSOR_IDS])
    offset = np.asarray([float(offsets_raw[sensor_id]) for sensor_id in SENSOR_IDS])
    observed = apply_observation_chain(
        signal,
        gain=gain,
        offset=offset,
        drift=drift,
        correlated_noise=correlated,
        white_noise=white,
        quantization_resolution=resolution,
    )
    return observed, ObservationPerturbationAudit(
        ar1_rho=float(rho),
        drift_strength_pct_dynamic_range_per_min=float(drift_strength),
        drift_slope_per_step=slope_per_step,
        gain=gain,
        offset=offset,
    )


def unique_quantization_level_counts(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("values must have shape (time, channel)")
    return np.asarray(
        [np.unique(array[:, channel]).size for channel in range(array.shape[1])],
        dtype=np.int64,
    )


@dataclass(frozen=True)
class DynamicDataset:
    """A2-DYN 聚合包的开发视图。

    ``signals`` 保持 UnifiedSample 需要的 ``(N, sensor, time, feature)`` 形状；
    composition oracle 使用时间优先数组；equilibrium、clean device 和 device
    state 遵循 ``(N, sensor, time)`` 合同，且不会被公开 adapter 读取。
    """

    records: tuple[Mapping[str, Any], ...]
    signals: np.ndarray
    valid_mask: np.ndarray
    quality: np.ndarray
    time_s: np.ndarray
    target: np.ndarray
    phase_id: np.ndarray
    observation_index: np.ndarray
    inlet_composition: np.ndarray
    inlet_coefficient: np.ndarray
    chamber_composition: np.ndarray
    equilibrium_reference_signals: np.ndarray
    clean_device_signals: np.ndarray
    device_states: np.ndarray
    privileged_parameters: np.ndarray
    device_audit: Mapping[str, np.ndarray]
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        records = tuple(dict(record) for record in self.records)
        if not records:
            raise ValueError("dynamic dataset must contain at least one record")
        signals = np.asarray(self.signals, dtype=np.float32)
        if signals.ndim != 4 or signals.shape[1] != 3 or signals.shape[3] != 1:
            raise ValueError("signals must have shape (N, 3, T, 1)")
        count, _, timesteps, _ = signals.shape
        if len(records) != count:
            raise ValueError("records and signals must have the same row count")
        valid_mask = np.asarray(self.valid_mask, dtype=np.bool_)
        if valid_mask.shape != signals.shape:
            raise ValueError("valid_mask must have the same shape as signals")
        quality = np.asarray(self.quality, dtype=np.float32)
        if quality.shape != (count, 3, timesteps):
            raise ValueError("quality must have shape (N, 3, T)")
        time_s = np.asarray(self.time_s, dtype=np.float64)
        if time_s.shape != (timesteps,) or not np.isfinite(time_s).all() or (
            timesteps > 1 and np.any(np.diff(time_s) <= 0.0)
        ):
            raise ValueError("time_s must be a finite strictly increasing vector")
        target = np.asarray(self.target, dtype=np.float32)
        phase_id = np.asarray(self.phase_id, dtype=np.int8)
        observation_index = np.asarray(self.observation_index, dtype=np.int64)
        if target.shape != (count, 3):
            raise ValueError("target must have shape (N, 3)")
        if phase_id.shape != (count, timesteps):
            raise ValueError("phase_id must have shape (N, T)")
        if observation_index.shape != (count,) or not np.array_equal(
            observation_index, np.arange(count, dtype=np.int64)
        ):
            raise ValueError("observation_index must align with the row order")
        inlet_composition = np.asarray(self.inlet_composition, dtype=np.float32)
        chamber_composition = np.asarray(self.chamber_composition, dtype=np.float32)
        for name, value, shape in (
            ("inlet_composition", inlet_composition, (count, timesteps, 3)),
            ("inlet_coefficient", self.inlet_coefficient, (count, timesteps)),
            ("chamber_composition", chamber_composition, (count, timesteps, 3)),
            ("equilibrium_reference_signals", self.equilibrium_reference_signals, (count, 3, timesteps)),
            ("clean_device_signals", self.clean_device_signals, (count, 3, timesteps)),
            ("device_states", self.device_states, (count, 3, timesteps)),
            ("privileged_parameters", self.privileged_parameters, (count, 12)),
        ):
            array = np.asarray(value)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
        for name, composition in (
            ("inlet_composition", inlet_composition),
            ("chamber_composition", chamber_composition),
        ):
            if np.any(composition < 0.0) or np.any(composition > _DYNAMIC_TARGET_TOTAL):
                raise ValueError(f"{name} must lie in [0,100] mol%")
            totals = np.sum(composition, axis=2, dtype=np.float32)
            if not np.all(
                np.abs(totals - np.float32(_DYNAMIC_TARGET_TOTAL)) <= np.float32(1.0e-6)
            ):
                raise ValueError(f"{name} must close to 100 mol% after float32 serialization")
        if not np.isfinite(signals).all() or not np.isfinite(quality).all():
            raise ValueError("signals and quality must contain only finite values")
        if np.any((quality < 0.0) | (quality > 1.0)):
            raise ValueError("quality must lie in [0,1]")
        if not np.isfinite(target).all() or np.any(target < 0.0) or np.any(target > 100.0):
            raise ValueError("target must be finite and lie in [0,100]")
        if not np.allclose(target.sum(axis=1), 100.0, rtol=0.0, atol=1.0e-5):
            raise ValueError("target compositions must sum to 100 mol%")
        audit = {str(key): np.asarray(value) for key, value in self.device_audit.items()}
        required_audit = {
            "ultrasonic_peak_correlation",
            "ultrasonic_snr",
            "ultrasonic_estimated_tof_uncertainty_s",
            "ultrasonic_lock_status",
            "tcd_energy_balance_residual_w",
            "ndir_active_voltage_v",
            "ndir_reference_voltage_v",
            "ndir_saturation_mask",
            "ndir_quantization_platform_length",
        }
        missing = required_audit - set(audit)
        if missing:
            raise ValueError(f"device_audit is missing keys: {sorted(missing)}")
        for key in required_audit - {"ndir_quantization_platform_length"}:
            expected = (count, timesteps)
            if audit[key].shape != expected:
                raise ValueError(f"device_audit[{key!r}] must have shape {expected}")
        if audit["ndir_quantization_platform_length"].shape != (count,):
            raise ValueError("ndir_quantization_platform_length must have shape (N,)")
        if not np.isfinite(audit["ultrasonic_peak_correlation"]).all():
            raise ValueError("ultrasonic peak correlations must be finite")
        object.__setattr__(self, "records", records)
        for name, value in (
            ("signals", signals),
            ("valid_mask", valid_mask),
            ("quality", quality),
            ("time_s", time_s),
            ("target", target),
            ("phase_id", phase_id),
            ("observation_index", observation_index),
            ("inlet_composition", inlet_composition),
            ("inlet_coefficient", np.asarray(self.inlet_coefficient, dtype=np.float32)),
            ("chamber_composition", chamber_composition),
            ("equilibrium_reference_signals", np.asarray(self.equilibrium_reference_signals, dtype=np.float32)),
            ("clean_device_signals", np.asarray(self.clean_device_signals, dtype=np.float32)),
            ("device_states", np.asarray(self.device_states, dtype=np.float32)),
            ("privileged_parameters", np.asarray(self.privileged_parameters, dtype=np.float64)),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for value in audit.values():
            value.setflags(write=False)
        object.__setattr__(self, "device_audit", MappingProxyType(audit))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    @property
    def sample_count(self) -> int:
        return len(self.records)

    @property
    def timesteps(self) -> int:
        return int(self.signals.shape[2])

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(str(record["mixture_id"]) for record in self.records)

    def indices(
        self,
        *,
        family: str | None = None,
        split: str | None = None,
    ) -> np.ndarray:
        return np.asarray(
            [
                index
                for index, record in enumerate(self.records)
                if (family is None or record["family"] == family)
                and (split is None or record["split"] == split)
            ],
            dtype=np.int64,
        )


def _generate_observation_rows(
    *,
    data: Mapping[str, Any],
    experiment: Mapping[str, Any],
    calibrations: Mapping[str, Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    time_s: np.ndarray,
    dt_s: float,
    observation_id_start: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """对给定 assignments 生成观测行数组与 records（开发与 test 共享骨架）。

    assignment 决定 family / split / 区域 / 目标组成 / 身份；噪声与观测扰动由
    generation_seed + target_index + repeat_index 唯一决定，因此同一 assignment
    在任意调用路径都产生相同的观测行。observation_id 从
    ``observation_id_start + 1`` 起连续编号，保证开发与 test 行号不重叠。
    """

    expected_rows = sum(
        int(data["families"][assignment["family"]]["repeat_count"])
        for assignment in assignments
    )
    if expected_rows == 0:
        raise ValueError("assignments must contain at least one group")
    pilot = experiment["pilot"]
    timesteps = int(data["time_axis"]["timesteps"])
    sensor_count = len(SENSOR_IDS)
    signals = np.empty((expected_rows, sensor_count, timesteps, 1), dtype=np.float32)
    valid_mask = np.ones_like(signals, dtype=np.bool_)
    quality = np.ones((expected_rows, sensor_count, timesteps), dtype=np.float32)
    target = np.empty((expected_rows, 3), dtype=np.float32)
    phase_id = np.broadcast_to(
        _phase_id_for_time(time_s, data["phases"]), (expected_rows, timesteps)
    ).copy()
    observation_index = np.arange(expected_rows, dtype=np.int64)
    inlet = np.empty((expected_rows, timesteps, 3), dtype=np.float32)
    inlet_coefficient = np.empty((expected_rows, timesteps), dtype=np.float32)
    chamber = np.empty_like(inlet)
    equilibrium_reference = np.empty((expected_rows, sensor_count, timesteps), dtype=np.float32)
    clean_device = np.empty_like(equilibrium_reference)
    device_states = np.empty_like(equilibrium_reference)
    privileged = np.empty((expected_rows, 12), dtype=np.float64)
    device_audit = {
        "ultrasonic_peak_correlation": np.empty((expected_rows, timesteps), dtype=np.float32),
        "ultrasonic_snr": np.empty((expected_rows, timesteps), dtype=np.float32),
        "ultrasonic_estimated_tof_uncertainty_s": np.empty((expected_rows, timesteps), dtype=np.float32),
        "ultrasonic_lock_status": np.empty((expected_rows, timesteps), dtype=np.bool_),
        "tcd_energy_balance_residual_w": np.empty((expected_rows, timesteps), dtype=np.float32),
        "ndir_active_voltage_v": np.empty((expected_rows, timesteps), dtype=np.float32),
        "ndir_reference_voltage_v": np.empty((expected_rows, timesteps), dtype=np.float32),
        "ndir_saturation_mask": np.empty((expected_rows, timesteps), dtype=np.bool_),
        "ndir_quantization_platform_length": np.empty(expected_rows, dtype=np.int32),
    }
    records: list[dict[str, Any]] = []
    row = 0
    for assignment in assignments:
        family_config = data["families"][assignment["family"]]
        repeat_count = int(family_config["repeat_count"])
        trajectory = _build_clean_trajectory(
            data,
            assignment=assignment,
            target_composition=np.asarray(assignment["composition"], dtype=np.float64),
            calibrations=calibrations,
            time_s=time_s,
            dt_s=dt_s,
            ultrasonic_internal_noise_std=float(pilot["ultrasonic_internal_noise_std"]),
        )
        for repeat_index in range(repeat_count):
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(data["generation_seed"]), int(assignment["target_index"]), repeat_index]
                )
            )
            observed, perturbation = apply_dynamic_observation_profile(
                trajectory["clean_device_signals"],
                noise_profile=trajectory["noise_profile"],
                calibration_profile=trajectory["calibration_profile"],
                noise_base=np.asarray(pilot["observation_noise_std_by_sensor"], dtype=np.float64),
                quantization=np.asarray(pilot["observation_quantization_by_sensor"], dtype=np.float64),
                dt_s=dt_s,
                index=int(assignment["target_index"]),
                rng=rng,
            )
            signals[row, :, :, 0] = observed.T.astype(np.float32)
            target[row] = np.asarray(assignment["composition"], dtype=np.float32)
            inlet[row] = _serialize_composition_float32(trajectory["inlet_composition"])
            inlet_coefficient[row] = trajectory["protocol_coefficient"].astype(np.float32)
            chamber[row] = _serialize_composition_float32(trajectory["chamber_composition"])
            equilibrium_reference[row] = trajectory["equilibrium_reference_signals"].T.astype(np.float32)
            clean_device[row] = trajectory["clean_device_signals"].T.astype(np.float32)
            device_states[row] = trajectory["device_states"].T.astype(np.float32)
            privileged[row] = trajectory["privileged_parameters"]
            device_audit["ultrasonic_peak_correlation"][row] = trajectory[
                "ultrasonic_quality"
            ]["peak_correlation"].astype(np.float32)
            device_audit["ultrasonic_snr"][row] = trajectory["ultrasonic_quality"]["snr"].astype(
                np.float32
            )
            device_audit["ultrasonic_estimated_tof_uncertainty_s"][row] = trajectory[
                "ultrasonic_quality"
            ]["estimated_tof_uncertainty_s"].astype(np.float32)
            device_audit["ultrasonic_lock_status"][row] = trajectory["ultrasonic_quality"][
                "lock_status"
            ]
            device_audit["tcd_energy_balance_residual_w"][row] = trajectory[
                "tcd_energy_balance_residual_w"
            ].astype(np.float32)
            device_audit["ndir_active_voltage_v"][row] = trajectory["ndir_active_voltage_v"].astype(np.float32)
            device_audit["ndir_reference_voltage_v"][row] = trajectory[
                "ndir_reference_voltage_v"
            ].astype(np.float32)
            device_audit["ndir_saturation_mask"][row] = trajectory["ndir_saturation_mask"]
            device_audit["ndir_quantization_platform_length"][row] = _longest_equal_run(
                observed[:, 2]
            )
            records.append(
                _build_dynamic_record(
                    assignment,
                    trajectory,
                    observation_id=f"a2dyn-obs-{observation_id_start + row + 1:07d}",
                )
            )
            row += 1
    if row != expected_rows:
        raise RuntimeError(f"generated row count mismatch: expected {expected_rows}, got {row}")
    return {
        "signals": signals,
        "valid_mask": valid_mask,
        "quality": quality,
        "time_s": time_s,
        "target": target,
        "phase_id": phase_id,
        "observation_index": observation_index,
        "inlet": inlet,
        "inlet_coefficient": inlet_coefficient,
        "chamber": chamber,
        "equilibrium_reference": equilibrium_reference,
        "clean_device": clean_device,
        "device_states": device_states,
        "privileged": privileged,
        "device_audit": device_audit,
    }, records


def generate_a2_dynamic_development(
    output_dir: str | Path,
    *,
    data_config: Mapping[str, Any] | str | Path,
    experiment_config: Mapping[str, Any] | str | Path,
    a2h_config: Mapping[str, Any] | str | Path,
    eval_config: Mapping[str, Any] | str | Path | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> DynamicDataset:
    """生成 A2-DYN-3 的 train、val、stress_val 开发包。

    该函数只读取冻结配置，不接收临时 family、数量、采样率或阈值覆盖。
    所有 observation 的 clean 轨迹只生成一次，重复观测只重新执行观测扰动链。
    """

    data = _read_mapping_config(data_config, "data_config")
    experiment = _read_mapping_config(experiment_config, "experiment_config")
    a2h = _read_mapping_config(a2h_config, "a2h_config")
    evaluation = {} if eval_config is None else _read_mapping_config(eval_config, "eval_config")
    pilot = experiment.get("pilot")
    if not isinstance(pilot, Mapping) or pilot.get("status") != "PILOT_QUALIFIED":
        raise ValueError("A2-DYN-3 requires a qualified A2-DYN-2 pilot")
    selected_rate = float(pilot["selected_sample_rate_hz"])
    selected_duration = float(pilot["selected_duration_s"])
    if selected_rate != 5.0 or selected_duration != 240.0:
        raise ValueError("A2-DYN-3 requires the frozen 5 Hz / 240 s pilot selection")
    timesteps = int(data["time_axis"]["timesteps"])
    dt_s = float(data["time_axis"]["dt_s"])
    time_s = np.arange(timesteps, dtype=np.float64) * dt_s
    assignments = _build_development_assignments(data)
    calibrations = _calibration_profiles(data, a2h)
    arrays, records = _generate_observation_rows(
        data=data,
        experiment=experiment,
        calibrations=calibrations,
        assignments=assignments,
        time_s=time_s,
        dt_s=dt_s,
        observation_id_start=0,
    )
    output_path = Path(output_dir)
    manifest = _build_dynamic_manifest(
        data=data,
        evaluation=evaluation,
        experiment=experiment,
        records=records,
        source_hashes=source_hashes,
        **arrays,
    )
    _write_dynamic_dataset(
        output_path,
        data=data,
        evaluation=evaluation,
        experiment=experiment,
        a2h=a2h,
        records=records,
        manifest=manifest,
        **arrays,
    )
    return load_a2_dynamic_dataset(output_path)


def generate_a2_dynamic_dataset(
    output_dir: str | Path,
    *,
    data_config: Mapping[str, Any] | str | Path,
    experiment_config: Mapping[str, Any] | str | Path,
    a2h_config: Mapping[str, Any] | str | Path,
    eval_config: Mapping[str, Any] | str | Path | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> DynamicDataset:
    """A2-DYN 开发数据生成的稳定别名。"""

    return generate_a2_dynamic_development(
        output_dir,
        data_config=data_config,
        experiment_config=experiment_config,
        a2h_config=a2h_config,
        eval_config=eval_config,
        source_hashes=source_hashes,
    )


def _build_test_assignments(
    data: Mapping[str, Any],
    *,
    start_group_index: int,
    development_compositions: set[tuple[float, float, float]] | None = None,
) -> list[dict[str, Any]]:
    """构建 A2-DYN-4 的 test assignments（630 组，含 3 个规范 pure 顶点）。

    test 区域配额在配置中以“替换 3 个 binary 配额后的净值”冻结
    （interior 315 / near_boundary 189 / binary 123 / pure 3）。比例展开按
    pure 顶点占用的 3 个 binary 配额回补后的 315 / 189 / 126 基数进行，
    D-JOINT 的 3 个 binary 槽位替换为规范 pure 顶点，其余区域组成从
    ``development_compositions`` 之外的唯一低差异池中抽取。
    """

    distribution = data["composition_distribution"]
    quota_by_split = distribution["region_quota_by_split"]
    test_raw = quota_by_split[_TEST_SPLIT]
    vertices = list(distribution["pure_vertices"])
    pure_count = int(test_raw.get(_PURE_REGION, 0))
    if pure_count != len(vertices) or pure_count == 0:
        raise ValueError(
            f"test pure quota ({pure_count}) must match the registered "
            f"pure vertices ({len(vertices)})"
        )
    quotas = {region: int(test_raw[region]) for region in _DYNAMIC_REGIONS}
    if sum(quotas.values()) + pure_count != 630:
        raise ValueError(f"test region quotas must total 630 groups: {quotas}")
    # 纯顶点按设计替换 D-JOINT 的 binary 槽位：按回补后的 5:3:2 基数展开，
    # 保证每个 family 的常规区域分配可整除，替换后 binary 净值恰为配置值。
    base_quotas = dict(quotas)
    base_quotas["binary"] = base_quotas["binary"] + pure_count
    if int(quota_by_split.get("test", {}).get("pure", 0)) != pure_count:
        raise ValueError("test pure quota is not registered in region_quota_by_split")
    base_total = sum(base_quotas.values())
    if base_total != 630:
        raise ValueError(f"test base region quotas must total 630 groups: {base_quotas}")

    pool_used: set[tuple[float, float, float]] = set(development_compositions or ())
    pools: dict[str, list[tuple[float, float, float]]] = {}
    for region_index, region in enumerate(_DYNAMIC_REGIONS):
        pools[region] = _generate_region_compositions(
            quotas[region],
            region=region,
            seed_parts=(
                int(data["generation_seed"]),
                int(data["split_seed"]),
                3,
                region_index,
            ),
            used=pool_used,
        )
    families = data["families"]
    assignments: list[dict[str, Any]] = []
    pool_offsets = {region: 0 for region in _DYNAMIC_REGIONS}
    numeric_group_index = start_group_index
    target_index = start_group_index
    for family_index, family in enumerate(_DYNAMIC_FAMILIES):
        count = int(families[family]["groups_by_split"][_TEST_SPLIT])
        family_region_counts = _allocate_region_counts(count, base_quotas, base_total)
        labels = [
            region
            for region in _DYNAMIC_REGIONS
            for _ in range(family_region_counts[region])
        ]
        assignment_rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(data["split_seed"]), 3, family_index, 7003]
            )
        )
        labels = [labels[index] for index in assignment_rng.permutation(len(labels))]
        pure_vertices = list(vertices)
        if family == "D-JOINT":
            # D-JOINT 的 pure 槽固定在本 family 尾部；先移走 3 个 binary 标签，
            # 再追加规范 pure 顶点标签，避免纯气组混入数字编号流。
            removed = 0
            kept: list[str] = []
            for label in labels:
                if label == "binary" and removed < pure_count:
                    removed += 1
                    continue
                kept.append(label)
            if removed != pure_count:
                raise ValueError(
                    f"D-JOINT/test must carry {pure_count} pure vertices, "
                    f"removed {removed} binary slots"
                )
            labels = kept + [_PURE_REGION] * pure_count
        for local_index, region in enumerate(labels):
            if region == _PURE_REGION:
                if not pure_vertices:
                    raise ValueError("pure vertex queue exhausted before labels were filled")
                vertex = pure_vertices.pop(0)
                composition = tuple(
                    float(value) for value in vertex["composition_pct"]
                )
                mixture_id = str(vertex["mixture_id"])
            else:
                offset = pool_offsets[region]
                composition = pools[region][offset]
                pool_offsets[region] = offset + 1
                mixture_id = f"a2dyn-mix-{numeric_group_index + 1:07d}"
                numeric_group_index += 1
            assignments.append(
                {
                    "mixture_id": mixture_id,
                    "family": family,
                    "split": _TEST_SPLIT,
                    "family_index": local_index,
                    "target_index": target_index,
                    "composition_region": region,
                    "composition": composition,
                }
            )
            target_index += 1
        if family == "D-JOINT" and pure_vertices:
            raise ValueError("unassigned pure vertices remain after D-JOINT/test expansion")
    for region in _DYNAMIC_REGIONS:
        if pool_offsets[region] != len(pools[region]):
            raise RuntimeError(
                f"test composition pool for {region!r} was not fully consumed "
                f"({pool_offsets[region]} of {len(pools[region])})"
            )
    return assignments


def generate_a2_dynamic_test(
    output_dir: str | Path,
    *,
    data_config: Mapping[str, Any] | str | Path,
    experiment_config: Mapping[str, Any] | str | Path,
    a2h_config: Mapping[str, Any] | str | Path,
    eval_config: Mapping[str, Any] | str | Path | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> DynamicDataset:
    """生成 A2-DYN-4 的 test 观测并聚合为 6,300 观测的完整数据包。

    该函数只允许从 A2-DYN-3 的开发包（``development_only`` 且不含 test）聚合
    一次；重复调用会因数据包已含 test 而显式失败。开发行字节保持不变，
    完整包的 manifest、records 与全部数组在该函数内重写。
    """

    data = _read_mapping_config(data_config, "data_config")
    experiment = _read_mapping_config(experiment_config, "experiment_config")
    a2h = _read_mapping_config(a2h_config, "a2h_config")
    evaluation = {} if eval_config is None else _read_mapping_config(eval_config, "eval_config")
    pilot = experiment.get("pilot")
    if not isinstance(pilot, Mapping) or pilot.get("status") != "PILOT_QUALIFIED":
        raise ValueError("A2-DYN-4 requires a qualified A2-DYN-2 pilot")
    selected_rate = float(pilot["selected_sample_rate_hz"])
    selected_duration = float(pilot["selected_duration_s"])
    if selected_rate != 5.0 or selected_duration != 240.0:
        raise ValueError("A2-DYN-4 requires the frozen 5 Hz / 240 s pilot selection")
    output_path = Path(output_dir)
    development = load_a2_dynamic_dataset(output_path)
    development_manifest = development.manifest
    if development_manifest.get("development_only") is not True or (
        development_manifest.get("contains_test") is not False
    ):
        raise ValueError("A2-DYN-4 requires the frozen A2-DYN-3 development-only dataset")
    dev_splits = {str(record["split"]) for record in development.records}
    if not dev_splits or not dev_splits.issubset(set(_DYNAMIC_SPLITS)):
        raise ValueError("A2-DYN-4 requires a development-only dataset without test records")
    expected_dev_rows = sum(
        int(data["split_contract"]["totals"][split]["observations"])
        for split in _DYNAMIC_SPLITS
    )
    if development.sample_count != expected_dev_rows:
        raise ValueError(
            "A2-DYN-4 requires the frozen development row count "
            f"{expected_dev_rows}, got {development.sample_count}"
        )
    development_compositions = {
        (
            float(record["x_Ar_pct"]),
            float(record["x_He_pct"]),
            float(record["x_CO2_pct"]),
        )
        for record in development.records
    }
    assignments = _build_test_assignments(
        data,
        start_group_index=len(
            {str(record["mixture_id"]) for record in development.records}
        ),
        development_compositions=development_compositions,
    )
    timesteps = int(data["time_axis"]["timesteps"])
    dt_s = float(data["time_axis"]["dt_s"])
    time_s = np.arange(timesteps, dtype=np.float64) * dt_s
    calibrations = _calibration_profiles(data, a2h)
    test_arrays, test_records = _generate_observation_rows(
        data=data,
        experiment=experiment,
        calibrations=calibrations,
        assignments=assignments,
        time_s=time_s,
        dt_s=dt_s,
        observation_id_start=development.sample_count,
    )
    records = [dict(record) for record in development.records] + test_records
    combined = _combine_development_and_test_arrays(development, test_arrays)
    combined["time_s"] = np.asarray(time_s, dtype=np.float64)
    combined["observation_index"] = np.arange(len(records), dtype=np.int64)
    manifest = _build_dynamic_manifest(
        data=data,
        evaluation=evaluation,
        experiment=experiment,
        records=records,
        source_hashes=source_hashes,
        splits=_ALL_DYNAMIC_SPLITS,
        regions=_ALL_DYNAMIC_REGIONS,
        development_only=False,
        contains_test=True,
        manifest_status="TEST_GENERATED",
        **combined,
    )
    _write_dynamic_dataset(
        output_path,
        data=data,
        evaluation=evaluation,
        experiment=experiment,
        a2h=a2h,
        records=records,
        manifest=manifest,
        audit_stage="A2-DYN-4",
        **combined,
    )
    return load_a2_dynamic_dataset(output_path)


def _combine_development_and_test_arrays(
    development: DynamicDataset,
    test_arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """把开发包数组与 test 局部数组按行拼接（行序：开发在前）。"""

    # test 数组使用骨架键名（inlet / chamber / equilibrium_reference ...），
    # 开发包通过 DynamicDataset 属性名访问，二者需要显式映射。
    name_mapping = {
        "signals": "signals",
        "valid_mask": "valid_mask",
        "quality": "quality",
        "target": "target",
        "phase_id": "phase_id",
        "inlet": "inlet_composition",
        "inlet_coefficient": "inlet_coefficient",
        "chamber": "chamber_composition",
        "equilibrium_reference": "equilibrium_reference_signals",
        "clean_device": "clean_device_signals",
        "device_states": "device_states",
        "privileged": "privileged_parameters",
    }
    missing = set(name_mapping) - set(test_arrays)
    if missing:
        raise ValueError(f"test arrays are missing keys: {sorted(missing)}")
    combined = {
        test_key: np.concatenate(
            (getattr(development, dev_name), np.asarray(test_arrays[test_key])),
            axis=0,
        )
        for test_key, dev_name in name_mapping.items()
    }
    combined["device_audit"] = {
        key: np.concatenate(
            (
                np.asarray(development.device_audit[key]),
                np.asarray(test_arrays["device_audit"][key]),
            ),
            axis=0,
        )
        for key in test_arrays["device_audit"]
    }
    return combined


def load_a2_dynamic_dataset(dataset_dir: str | Path) -> DynamicDataset:
    """加载并校验 A2-DYN 聚合包，不默认读取任何外部 test。"""

    dataset_path = Path(dataset_dir)
    manifest = json.loads((dataset_path / "manifest.json").read_text(encoding="utf-8"))
    raw_records = (dataset_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    records = tuple(json.loads(line) for line in raw_records if line.strip())
    with np.load(dataset_path / "observations.npz", allow_pickle=False) as archive:
        signals = np.asarray(archive["signals"], dtype=np.float32).copy()
        valid_mask = np.asarray(archive["valid_mask"], dtype=np.bool_).copy()
        quality = np.asarray(archive["quality"], dtype=np.float32).copy()
        time_s = np.asarray(archive["time_s"], dtype=np.float64).copy()
        target = np.asarray(archive["target"], dtype=np.float32).copy()
        phase_id = np.asarray(archive["phase_id"], dtype=np.int8).copy()
        observation_index = np.asarray(archive["observation_index"], dtype=np.int64).copy()
    with np.load(dataset_path / "oracle.npz", allow_pickle=False) as archive:
        inlet = np.asarray(archive["inlet_composition"], dtype=np.float32).copy()
        chamber = np.asarray(archive["chamber_composition"], dtype=np.float32).copy()
        equilibrium_reference = np.asarray(archive["equilibrium_reference_signals"], dtype=np.float32).copy()
        clean_device = np.asarray(archive["clean_device_signals"], dtype=np.float32).copy()
        device_states = np.asarray(archive["device_states"], dtype=np.float32).copy()
        privileged = np.asarray(archive["privileged_parameters"], dtype=np.float64).copy()
        inlet_coefficient = np.asarray(archive["inlet_coefficient"], dtype=np.float32).copy()
    with np.load(dataset_path / "device_audit.npz", allow_pickle=False) as archive:
        device_audit = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if inlet_coefficient.shape != inlet.shape[:2]:
        raise ValueError("oracle inlet_coefficient is not aligned with inlet_composition")
    dataset = DynamicDataset(
        records=records,
        signals=signals,
        valid_mask=valid_mask,
        quality=quality,
        time_s=time_s,
        target=target,
        phase_id=phase_id,
        observation_index=observation_index,
        inlet_composition=inlet,
        inlet_coefficient=inlet_coefficient,
        chamber_composition=chamber,
        equilibrium_reference_signals=equilibrium_reference,
        clean_device_signals=clean_device,
        device_states=device_states,
        privileged_parameters=privileged,
        device_audit=device_audit,
        manifest=manifest,
    )
    expected_hash = manifest.get("content_sha256")
    if not isinstance(expected_hash, str) or expected_hash != dynamic_content_sha256(
        manifest,
        records,
        _dynamic_array_mapping(dataset, inlet_coefficient),
    ):
        raise ValueError("A2-DYN dataset content_sha256 mismatch")
    return dataset


def rebind_a2_dynamic_source_hashes(
    dataset: DynamicDataset,
    source_hashes: Mapping[str, str],
) -> DynamicDataset:
    """以当前代码的 source hash 重绑定完整数据包 manifest。

    数据包的 content identity 包含 source_hashes；任何源文件演进后重新执行
    冻结审计，都必须先用该函数重算 manifest，否则 freshness 校验会拒绝。
    数组与 records 不改变，只重算 manifest 绑定与内容 hash。
    """

    manifest = dict(dataset.manifest)
    manifest["source_hashes"] = {
        str(key): str(value) for key, value in source_hashes.items()
    }
    manifest["content_sha256"] = dynamic_content_sha256(
        manifest,
        dataset.records,
        _dynamic_array_mapping(dataset, dataset.inlet_coefficient),
    )
    return replace(dataset, manifest=manifest)


def dynamic_content_sha256(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
) -> str:
    """计算与文件包一致的内容 hash；阶段状态字段不属于内容身份。"""

    manifest_identity = dict(manifest)
    for key in ("content_sha256", "audit_sha256", "audit_status", "status"):
        manifest_identity.pop(key, None)
    record_identity = [
        {key: value for key, value in record.items() if key != "status"}
        for record in records
    ]
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"manifest": manifest_identity, "records": record_identity},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in (
        "signals",
        "valid_mask",
        "quality",
        "time_s",
        "target",
        "phase_id",
        "observation_index",
        "inlet_composition",
        "chamber_composition",
        "inlet_coefficient",
        "equilibrium_reference_signals",
        "clean_device_signals",
        "device_states",
        "privileged_parameters",
        "ultrasonic_peak_correlation",
        "ultrasonic_snr",
        "ultrasonic_estimated_tof_uncertainty_s",
        "ultrasonic_lock_status",
        "tcd_energy_balance_residual_w",
        "ndir_active_voltage_v",
        "ndir_reference_voltage_v",
        "ndir_saturation_mask",
        "ndir_quantization_platform_length",
    ):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_mapping_config(
    value: Mapping[str, Any] | str | Path,
    label: str,
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} at {path} must be a JSON object")
    return payload


def _dynamic_array_mapping(
    dataset: DynamicDataset,
    inlet_coefficient: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    return _dynamic_array_mapping_from_values(
        signals=dataset.signals,
        valid_mask=dataset.valid_mask,
        quality=dataset.quality,
        time_s=dataset.time_s,
        target=dataset.target,
        phase_id=dataset.phase_id,
        observation_index=dataset.observation_index,
        inlet=dataset.inlet_composition,
        inlet_coefficient=dataset.inlet_coefficient if inlet_coefficient is None else inlet_coefficient,
        chamber=dataset.chamber_composition,
        equilibrium_reference=dataset.equilibrium_reference_signals,
        clean_device=dataset.clean_device_signals,
        device_states=dataset.device_states,
        privileged=dataset.privileged_parameters,
        device_audit=dataset.device_audit,
    )


def _dynamic_array_mapping_from_values(
    *,
    signals: np.ndarray,
    valid_mask: np.ndarray,
    quality: np.ndarray,
    time_s: np.ndarray,
    target: np.ndarray,
    phase_id: np.ndarray,
    observation_index: np.ndarray,
    inlet: np.ndarray,
    inlet_coefficient: np.ndarray,
    chamber: np.ndarray,
    equilibrium_reference: np.ndarray,
    clean_device: np.ndarray,
    device_states: np.ndarray,
    privileged: np.ndarray,
    device_audit: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "signals": np.asarray(signals),
        "valid_mask": np.asarray(valid_mask),
        "quality": np.asarray(quality),
        "time_s": np.asarray(time_s),
        "target": np.asarray(target),
        "phase_id": np.asarray(phase_id),
        "observation_index": np.asarray(observation_index),
        "inlet_composition": np.asarray(inlet),
        "inlet_coefficient": np.asarray(inlet_coefficient),
        "chamber_composition": np.asarray(chamber),
        "equilibrium_reference_signals": np.asarray(equilibrium_reference),
        "clean_device_signals": np.asarray(clean_device),
        "device_states": np.asarray(device_states),
        "privileged_parameters": np.asarray(privileged),
        **{str(key): np.asarray(value) for key, value in device_audit.items()},
    }


def _write_dynamic_dataset(
    output_path: Path,
    *,
    data: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    experiment: Mapping[str, Any],
    a2h: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    signals: np.ndarray,
    valid_mask: np.ndarray,
    quality: np.ndarray,
    time_s: np.ndarray,
    target: np.ndarray,
    phase_id: np.ndarray,
    observation_index: np.ndarray,
    inlet: np.ndarray,
    inlet_coefficient: np.ndarray,
    chamber: np.ndarray,
    equilibrium_reference: np.ndarray,
    clean_device: np.ndarray,
    device_states: np.ndarray,
    privileged: np.ndarray,
    device_audit: Mapping[str, np.ndarray],
    audit_stage: str = "A2-DYN-3",
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path / "observations.npz",
        signals=np.asarray(signals, dtype=np.float32),
        valid_mask=np.asarray(valid_mask, dtype=np.bool_),
        quality=np.asarray(quality, dtype=np.float32),
        time_s=np.asarray(time_s, dtype=np.float64),
        target=np.asarray(target, dtype=np.float32),
        phase_id=np.asarray(phase_id, dtype=np.int8),
        observation_index=np.asarray(observation_index, dtype=np.int64),
    )
    np.savez_compressed(
        output_path / "oracle.npz",
        inlet_composition=np.asarray(inlet, dtype=np.float32),
        chamber_composition=np.asarray(chamber, dtype=np.float32),
        inlet_coefficient=np.asarray(inlet_coefficient, dtype=np.float32),
        equilibrium_reference_signals=np.asarray(equilibrium_reference, dtype=np.float32),
        clean_device_signals=np.asarray(clean_device, dtype=np.float32),
        device_states=np.asarray(device_states, dtype=np.float32),
        privileged_parameters=np.asarray(privileged, dtype=np.float64),
    )
    np.savez_compressed(
        output_path / "device_audit.npz",
        **{str(key): np.asarray(value) for key, value in device_audit.items()},
    )
    ultrasonic_candidates = data["hardware_profiles"]["ultrasonic"]["candidates"]
    selected_id = str(data["hardware_profiles"]["ultrasonic"]["selected_profile_id"])
    selected_profile = next(
        profile for profile in ultrasonic_candidates
        if str(profile["ultrasonic_profile_id"]) == selected_id
    )
    fixture_time, fixture_waveform = build_reference_waveform(selected_profile)
    np.savez_compressed(
        output_path / "waveform_fixtures.npz",
        reference_time_s=np.asarray(fixture_time, dtype=np.float64),
        reference_waveform=np.asarray(fixture_waveform, dtype=np.float64),
        ultrasonic_profile_id=np.asarray([selected_id]),
    )
    snapshot = {
        "data_config": dict(data),
        "evaluation_config": dict(evaluation),
        "experiment_config": dict(experiment),
        "a2h_config": dict(a2h),
    }
    (output_path / "config_snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "records.jsonl").write_text(
        "".join(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (output_path / "manifest.json").write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "audit.json").write_text(
        json.dumps(
            {
                "status": "NOT_RUN",
                "stage": audit_stage,
                "content_sha256": manifest["content_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_development_assignments(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    distribution = data["composition_distribution"]
    quota_by_split = distribution["region_quota_by_split"]
    pools: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
    used: set[tuple[float, float, float]] = set()
    for split_index, split in enumerate(_DYNAMIC_SPLITS):
        quotas = {region: int(quota_by_split[split][region]) for region in _DYNAMIC_REGIONS}
        if int(quota_by_split[split].get("pure", 0)) != 0:
            raise ValueError("A2-DYN-3 development view must not contain pure vertices")
        pools[split] = {}
        for region_index, region in enumerate(_DYNAMIC_REGIONS):
            pools[split][region] = _generate_region_compositions(
                quotas[region],
                region=region,
                seed_parts=(
                    int(data["generation_seed"]),
                    int(data["split_seed"]),
                    split_index,
                    region_index,
                ),
                used=used,
            )
    assignments: list[dict[str, Any]] = []
    pool_offsets = {split: {region: 0 for region in _DYNAMIC_REGIONS} for split in _DYNAMIC_SPLITS}
    group_index = 0
    families = data["families"]
    for split_index, split in enumerate(_DYNAMIC_SPLITS):
        total_groups = sum(int(families[family]["groups_by_split"][split]) for family in _DYNAMIC_FAMILIES)
        quotas = {region: int(quota_by_split[split][region]) for region in _DYNAMIC_REGIONS}
        if sum(quotas.values()) != total_groups:
            raise ValueError(f"composition quota does not match development groups for {split}")
        for family_index, family in enumerate(_DYNAMIC_FAMILIES):
            count = int(families[family]["groups_by_split"][split])
            family_quotas = _allocate_region_counts(count, quotas, total_groups)
            labels = [
                region
                for region in _DYNAMIC_REGIONS
                for _ in range(family_quotas[region])
            ]
            assignment_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(data["split_seed"]), split_index, family_index, 7001]
                )
            )
            labels = [labels[index] for index in assignment_rng.permutation(len(labels))]
            for local_index, region in enumerate(labels):
                offset = pool_offsets[split][region]
                composition = pools[split][region][offset]
                pool_offsets[split][region] = offset + 1
                assignments.append(
                    {
                        "mixture_id": f"a2dyn-mix-{group_index + 1:07d}",
                        "family": family,
                        "split": split,
                        "family_index": local_index,
                        "target_index": group_index,
                        "composition_region": region,
                        "composition": composition,
                    }
                )
                group_index += 1
    return assignments


def _allocate_region_counts(
    count: int,
    quotas: Mapping[str, int],
    total: int,
) -> dict[str, int]:
    if count < 0 or total <= 0 or sum(int(value) for value in quotas.values()) != total:
        raise ValueError("invalid region allocation inputs")
    raw = {region: count * int(quotas[region]) / total for region in _DYNAMIC_REGIONS}
    result = {region: int(math.floor(raw[region])) for region in _DYNAMIC_REGIONS}
    remainder = count - sum(result.values())
    order = sorted(
        _DYNAMIC_REGIONS,
        key=lambda region: (raw[region] - result[region], -_DYNAMIC_REGIONS.index(region)),
        reverse=True,
    )
    for region in order[:remainder]:
        result[region] += 1
    return result


def _generate_region_compositions(
    count: int,
    *,
    region: str,
    seed_parts: Sequence[int],
    used: set[tuple[float, float, float]],
) -> list[tuple[float, float, float]]:
    if region not in _DYNAMIC_REGIONS:
        raise ValueError(f"unsupported development composition region {region!r}")
    rng = np.random.default_rng(np.random.SeedSequence([int(value) for value in seed_parts]))
    result: list[tuple[float, float, float]] = []
    for index in range(count):
        for attempt in range(10000):
            if region == "interior":
                candidate = _quantize_dynamic_composition(5.0 + 85.0 * rng.dirichlet(np.ones(3)))
            elif region == "near_boundary":
                low_index = index % 3
                low = float(rng.uniform(0.1, 5.0))
                remaining = [item for item in range(3) if item != low_index]
                ratio = float(rng.uniform(0.15, 0.85))
                raw = np.zeros(3, dtype=np.float64)
                raw[low_index] = low
                raw[remaining[0]] = (100.0 - low) * ratio
                raw[remaining[1]] = 100.0 - low - raw[remaining[0]]
                candidate = _quantize_dynamic_composition(raw)
            else:
                zero_index = index % 3
                ratio = float(rng.uniform(0.05, 0.95))
                raw = np.zeros(3, dtype=np.float64)
                nonzero = [item for item in range(3) if item != zero_index]
                raw[nonzero[0]] = 100.0 * ratio
                raw[nonzero[1]] = 100.0 - raw[nonzero[0]]
                candidate = _quantize_dynamic_composition(raw)
            if _classify_dynamic_composition(candidate) != region:
                continue
            if candidate in used:
                continue
            used.add(candidate)
            result.append(candidate)
            break
        else:
            raise RuntimeError(f"could not generate a unique {region} composition at index {index}")
    return result


def _quantize_dynamic_composition(values: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("dynamic composition must contain three finite values")
    first = round(float(array[0]), 2)
    second = round(float(array[1]), 2)
    third = round(100.0 - first - second, 2)
    result = (first, second, third)
    if any(value < 0.0 or value > 100.0 for value in result) or not math.isclose(
        sum(result), 100.0, rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise RuntimeError(f"dynamic composition quantization failed: {result}")
    return result


def _classify_dynamic_composition(composition: Sequence[float]) -> str:
    values = tuple(float(value) for value in composition)
    if sum(value == 0.0 for value in values) == 1:
        return "binary"
    if min(values) >= 5.0:
        return "interior"
    if all(value > 0.0 for value in values) and min(values) <= 5.0:
        return "near_boundary"
    raise ValueError(f"composition does not match development regions: {values}")


def _calibration_profiles(
    data: Mapping[str, Any],
    a2h: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    source_profiles = {
        str(profile["calibration_profile_id"]): profile
        for profile in a2h["calibration_profiles"]
    }
    result: dict[str, Mapping[str, Any]] = {}
    for profile in data["calibration_profiles"]:
        profile_id = str(profile["calibration_profile_id"])
        source_id = str(profile.get("source_profile_id", profile_id))
        if source_id not in source_profiles:
            raise ValueError(f"unknown A2H calibration source profile {source_id!r}")
        source = dict(source_profiles[source_id])
        source["calibration_profile_id"] = profile_id
        result[profile_id] = source
    return result


def _build_clean_trajectory(
    data: Mapping[str, Any],
    *,
    assignment: Mapping[str, Any],
    target_composition: np.ndarray,
    calibrations: Mapping[str, Mapping[str, Any]],
    time_s: np.ndarray,
    dt_s: float,
    ultrasonic_internal_noise_std: float,
) -> dict[str, Any]:
    family = data["families"][assignment["family"]]
    split = str(assignment["split"])
    family_index = int(assignment["family_index"])
    target = np.asarray(target_composition, dtype=np.float64)
    protocol_id = choose_profile_id(family["protocol_by_split"][split], family_index)
    transport_id = choose_profile_id(family["transport_by_split"][split], family_index)
    environment_id = choose_profile_id(family["environment_by_split"][split], family_index)
    calibration_id = choose_profile_id(family["calibration_by_split"][split], family_index)
    noise_id = choose_profile_id(family["noise_by_split"][split], family_index)
    protocols = {str(item["protocol_profile_id"]): item for item in data["protocol_profiles"]}
    transports = {str(item["transport_profile_id"]): item for item in data["transport"]["profiles"]}
    environments = {str(item["environment_id"]): item for item in data["environment_profiles"]}
    noises = {str(item["noise_profile_id"]): item for item in data["noise_profiles"]}
    hardware = data["hardware_profiles"]
    ultrasonic_profiles = {
        str(item["ultrasonic_profile_id"]): item for item in hardware["ultrasonic"]["candidates"]
    }
    ultrasonic_profile_id = str(hardware["ultrasonic"]["selected_profile_id"])
    ultrasonic_profile = ultrasonic_profiles[ultrasonic_profile_id]
    multipath_profiles = {
        str(item["multipath_profile_id"]): item
        for item in hardware["ultrasonic"]["multipath_profiles"]
    }
    thermal_profile = hardware["thermal"]["profiles"][0]
    ndir_profile = hardware["ndir"]["profiles"][0]
    protocol = protocols[protocol_id]
    transport = transports[transport_id]
    environment = environments[environment_id]
    calibration = calibrations[calibration_id]
    noise = noises[noise_id]
    protocol_instance = resolve_protocol_instance(protocol, transport, index=int(assignment["target_index"]))
    coefficient = np.asarray(
        protocol_inlet_coefficient(time_s, **dict(protocol_instance.parameters)),
        dtype=np.float64,
    )
    inlet = build_inlet_composition(
        time_s,
        purge_composition_pct=data["inlet"]["purge_composition_pct"],
        target_composition_pct=target,
        coefficient=coefficient,
    )
    transport_contract = data["transport"]
    tau_transport = {
        "ultrasonic_tof": _sample_dynamic_transport(
            transport["tau_transport_ultrasonic_s"],
            int(assignment["target_index"]),
            53,
            transport_contract,
        ),
        "thermal_conductivity_voltage": _sample_dynamic_transport(
            transport["tau_transport_thermal_s"],
            int(assignment["target_index"]),
            59,
            transport_contract,
        ),
        "ndir_co2_voltage": _sample_dynamic_transport(
            transport["tau_transport_ndir_s"],
            int(assignment["target_index"]),
            61,
            transport_contract,
        ),
    }
    tau_mix = sample_registered_range(
        transport["tau_mix_s"],
        int(assignment["target_index"]),
        distribution=str(transport_contract["tau_mix_distribution"]),
        salt=67,
    )
    layers = simulate_dynamic_layers(
        inlet,
        dt_s=dt_s,
        tau_mix_s=tau_mix,
        tau_transport_s=tau_transport,
    )
    acoustic_scale, tcd_scale, ndir_scale = calibration_physical_scales(calibration)
    local = layers.local_composition_pct
    shared = evaluate_shared_physics(
        local["ultrasonic_tof"],
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        path_length_m=float(ultrasonic_profile["path_length_m"]) * acoustic_scale,
        sound_speed_model_id=data["physics_reference"]["eos"]["sound_speed_model_id"],
    )
    ultrasonic_clean = estimate_ultrasonic_tof_series(shared["tof_s"], ultrasonic_profile)
    ultrasonic_quality = estimate_ultrasonic_quality_series(
        local["ultrasonic_tof"],
        profile=ultrasonic_profile,
        internal_noise_std=ultrasonic_internal_noise_std,
        multipath_profile=multipath_profiles[
            str(ultrasonic_profile["multipath_profile_id"])
        ],
    )
    tcd = simulate_tcd(
        local["thermal_conductivity_voltage"],
        temperature_k=environment["temperature_k"],
        dt_s=dt_s,
        profile=thermal_profile,
        response_scale=tcd_scale,
    )
    ndir = simulate_ndir(
        local["ndir_co2_voltage"],
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        dt_s=dt_s,
        profile=ndir_profile,
        absorbance_scale=ndir_scale,
    )
    clean = np.column_stack((ultrasonic_clean, tcd.clean_voltage_v, ndir.clean_voltage_v))
    equilibrium_reference = np.column_stack(
        (
            shared["tof_s"],
            shared["thermal_voltage_v"] * tcd_scale,
            shared["ndir_voltage_v"] * ndir_scale,
        )
    )
    device_states = np.column_stack(
        (ultrasonic_clean, tcd.heater_temperature_k, ndir.active_reference_ratio)
    )
    privileged_parameters = np.asarray(
        [
            tau_mix,
            tau_transport["ultrasonic_tof"],
            tau_transport["thermal_conductivity_voltage"],
            tau_transport["ndir_co2_voltage"],
            float(environment["temperature_k"]),
            float(environment["pressure_pa"]),
            acoustic_scale,
            tcd_scale,
            ndir_scale,
            protocol_instance.exposure_onset_s,
            protocol_instance.exposure_end_s,
            float(assignment["target_index"]),
        ],
        dtype=np.float64,
    )
    return {
        "protocol_profile_id": protocol_id,
        "transport_profile_id": transport_id,
        "environment_id": environment_id,
        "calibration_profile_id": calibration_id,
        "noise_profile_id": noise_id,
        "ultrasonic_profile_id": ultrasonic_profile_id,
        "thermal_profile_id": str(thermal_profile["thermal_profile_id"]),
        "ndir_profile_id": str(ndir_profile["ndir_profile_id"]),
        "protocol_instance": protocol_instance,
        "protocol_coefficient": coefficient,
        "inlet_composition": inlet,
        "chamber_composition": layers.chamber_composition_pct,
        "equilibrium_reference_signals": equilibrium_reference,
        "clean_device_signals": clean,
        "device_states": device_states,
        "privileged_parameters": privileged_parameters,
        "ultrasonic_quality": ultrasonic_quality,
        "tcd_energy_balance_residual_w": tcd.energy_balance_residual_w,
        "ndir_active_voltage_v": ndir.active_voltage_v,
        "ndir_reference_voltage_v": ndir.reference_voltage_v,
        "ndir_saturation_mask": ndir.saturation_mask,
        "noise_profile": noise,
        "calibration_profile": calibration,
        "ultrasonic_adc_rate_hz": float(ultrasonic_profile["adc_rate_hz"]),
    }


def _sample_dynamic_transport(
    values: Sequence[float],
    index: int,
    salt: int,
    contract: Mapping[str, Any],
) -> float:
    if float(values[0]) == 0.0:
        return sample_registered_range(
            values,
            index,
            distribution=str(contract["nonzero_transport_distribution"]),
            salt=salt,
            zero_probability=float(contract["zero_transport_probability"]),
            minimum_nonzero=float(contract["minimum_nonzero_transport_s"]),
        )
    return sample_registered_range(
        values,
        index,
        distribution=str(contract["nonzero_transport_distribution"]),
        salt=salt,
    )


def _build_dynamic_record(
    assignment: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    observation_id: str,
) -> dict[str, Any]:
    target = tuple(float(value) for value in assignment["composition"])
    instance = trajectory["protocol_instance"]
    return {
        "schema_version": _DYNAMIC_RECORD_SCHEMA_VERSION,
        "observation_id": observation_id,
        "mixture_id": str(assignment["mixture_id"]),
        "split": str(assignment["split"]),
        "family": str(assignment["family"]),
        "composition_region": str(assignment["composition_region"]),
        "protocol_profile_id": str(trajectory["protocol_profile_id"]),
        "transport_profile_id": str(trajectory["transport_profile_id"]),
        "ultrasonic_profile_id": str(trajectory["ultrasonic_profile_id"]),
        "thermal_profile_id": str(trajectory["thermal_profile_id"]),
        "ndir_profile_id": str(trajectory["ndir_profile_id"]),
        "environment_id": str(trajectory["environment_id"]),
        "calibration_profile_id": str(trajectory["calibration_profile_id"]),
        "noise_profile_id": str(trajectory["noise_profile_id"]),
        "exposure_onset_s": float(instance.exposure_onset_s),
        "exposure_end_s": float(instance.exposure_end_s),
        "timesteps": 1200,
        "dt_s": 0.2,
        "status": "generated",
        "x_Ar_pct": target[0],
        "x_He_pct": target[1],
        "x_CO2_pct": target[2],
    }


def _phase_id_for_time(time_s: np.ndarray, phases: Sequence[Mapping[str, Any]]) -> np.ndarray:
    result = np.full(time_s.shape, -1, dtype=np.int8)
    for index, phase in enumerate(phases):
        mask = (time_s >= float(phase["start_s"])) & (time_s < float(phase["end_s_exclusive"]))
        result[mask] = index
    if np.any(result < 0):
        raise ValueError("frozen phases do not cover the complete dynamic time axis")
    return result


def _longest_equal_run(values: np.ndarray) -> int:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("platform values must be a non-empty vector")
    changes = np.flatnonzero(array[1:] != array[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [array.size]))
    return int(np.max(np.diff(boundaries)))


def _build_dynamic_manifest(
    *,
    data: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    experiment: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    signals: np.ndarray,
    valid_mask: np.ndarray,
    quality: np.ndarray,
    time_s: np.ndarray,
    target: np.ndarray,
    phase_id: np.ndarray,
    observation_index: np.ndarray,
    inlet: np.ndarray,
    inlet_coefficient: np.ndarray,
    chamber: np.ndarray,
    equilibrium_reference: np.ndarray,
    clean_device: np.ndarray,
    device_states: np.ndarray,
    privileged: np.ndarray,
    device_audit: Mapping[str, np.ndarray],
    source_hashes: Mapping[str, str] | None,
    splits: Sequence[str] | None = None,
    regions: Sequence[str] | None = None,
    development_only: bool = True,
    contains_test: bool = False,
    manifest_status: str = "DEVELOPMENT_GENERATED",
) -> dict[str, Any]:
    count = len(records)
    active_splits = tuple(splits) if splits is not None else _DYNAMIC_SPLITS
    active_regions = tuple(regions) if regions is not None else _DYNAMIC_REGIONS
    split_groups = {
        split: sorted(
            {
                str(record["mixture_id"])
                for record in records
                if record["split"] == split
            }
        )
        for split in active_splits
    }
    family_counts = {
        family: {
            "groups": {
                split: len(
                    {
                        str(record["mixture_id"])
                        for record in records
                        if record["family"] == family and record["split"] == split
                    }
                )
                for split in active_splits
            },
            "observations": {
                split: sum(
                    record["family"] == family and record["split"] == split
                    for record in records
                )
                for split in active_splits
            },
        }
        for family in _DYNAMIC_FAMILIES
    }
    region_counts = {
        split: {
            region: len(
                {
                    str(record["mixture_id"])
                    for record in records
                    if record["split"] == split and record["composition_region"] == region
                }
            )
            for region in active_regions
        }
        for split in active_splits
    }
    manifest: dict[str, Any] = {
        "schema_version": _DYNAMIC_MANIFEST_SCHEMA_VERSION,
        "dataset_schema_version": str(data["schema_version"]),
        "dataset_id": str(data["dataset_id"]),
        "data_version": str(data["data_version"]),
        "protocol_revision": str(data["protocol_revision"]),
        "status": manifest_status,
        "audit_status": "NOT_RUN",
        "development_only": development_only,
        "contains_test": contains_test,
        "generation_seed": int(data["generation_seed"]),
        "split_seed": int(data["split_seed"]),
        "observation_mode": str(data["observation_mode"]),
        "sensor_ids": list(data["sensor_ids"]),
        "target_names": list(data["target_names"]),
        "target_units": str(data["target_units"]),
        "timesteps": int(data["timesteps"]),
        "dt_s": float(data["dt_s"]),
        "sample_rate_hz": float(data["time_axis"]["sample_rate_hz"]),
        "duration_s": float(data["time_axis"]["duration_s"]),
        "sample_count": count,
        "mixture_count": len({str(record["mixture_id"]) for record in records}),
        "split_counts": {
            split: {
                "groups": len(split_groups[split]),
                "observations": sum(record["split"] == split for record in records),
            }
            for split in active_splits
        },
        "family_counts": family_counts,
        "region_counts": region_counts,
        "split_groups": split_groups,
        "split_hash": _canonical_sha256(split_groups),
        "records_sha256": _canonical_sha256(
            [{key: value for key, value in record.items() if key != "status"} for record in records]
        ),
        "array_shapes": {
            "signals": list(signals.shape),
            "valid_mask": list(valid_mask.shape),
            "quality": list(quality.shape),
            "time_s": list(time_s.shape),
            "target": list(target.shape),
            "phase_id": list(phase_id.shape),
            "observation_index": list(observation_index.shape),
            "inlet_composition": list(inlet.shape),
            "inlet_coefficient": list(inlet_coefficient.shape),
            "chamber_composition": list(chamber.shape),
            "equilibrium_reference_signals": list(equilibrium_reference.shape),
            "clean_device_signals": list(clean_device.shape),
            "device_states": list(device_states.shape),
            "privileged_parameters": list(privileged.shape),
        },
        "array_dtypes": {
            "signals": str(signals.dtype),
            "valid_mask": str(valid_mask.dtype),
            "quality": str(quality.dtype),
            "time_s": str(time_s.dtype),
            "target": str(target.dtype),
            "phase_id": str(phase_id.dtype),
            "observation_index": str(observation_index.dtype),
        },
        "oracle_arrays": [
            "inlet_composition",
            "chamber_composition",
            "inlet_coefficient",
            "equilibrium_reference_signals",
            "clean_device_signals",
            "device_states",
            "privileged_parameters",
        ],
        "device_audit_arrays": sorted(device_audit),
        "privileged_parameter_names": [
            "tau_mix_s",
            "tau_transport_ultrasonic_s",
            "tau_transport_thermal_s",
            "tau_transport_ndir_s",
            "temperature_k",
            "pressure_pa",
            "acoustic_path_scale",
            "tcs_response_scale",
            "ndir_absorbance_scale",
            "exposure_onset_s",
            "exposure_end_s",
            "group_index",
        ],
        "waveform_policy": "temporary_during_generation_and_registered_fixtures_only",
        "waveform_persisted": False,
        "waveform_fixture_count": 1,
        "ultrasonic_quality_source": "registered_profile_matched_filter_low_frequency_surrogate",
        "config_sha256": _canonical_sha256(data),
        "evaluation_config_sha256": _canonical_sha256(evaluation),
        "experiment_config_sha256": _canonical_sha256(experiment),
        "source_hashes": dict(source_hashes or {}),
    }
    manifest["content_sha256"] = dynamic_content_sha256(
        manifest,
        records,
        _dynamic_array_mapping_from_values(
            signals=signals,
            valid_mask=valid_mask,
            quality=quality,
            time_s=time_s,
            target=target,
            phase_id=phase_id,
            observation_index=observation_index,
            inlet=inlet,
            inlet_coefficient=inlet_coefficient,
            chamber=chamber,
            equilibrium_reference=equilibrium_reference,
            clean_device=clean_device,
            device_states=device_states,
            privileged=privileged,
            device_audit=device_audit,
        ),
    )
    return manifest


__all__ = [
    "CANONICAL_NOISE_DT_S",
    "DynamicDataset",
    "ObservationPerturbationAudit",
    "ProtocolInstance",
    "apply_dynamic_observation_profile",
    "calibration_physical_scales",
    "choose_profile_id",
    "dynamic_content_sha256",
    "generate_a2_dynamic_dataset",
    "generate_a2_dynamic_development",
    "generate_a2_dynamic_test",
    "load_a2_dynamic_dataset",
    "rebind_a2_dynamic_source_hashes",
    "resample_continuous_series",
    "resolve_protocol_instance",
    "sample_registered_range",
    "unique_quantization_level_counts",
]
