"""A2-DYN 分层时序物理与观测扰动。

本模块只表达共同气室、局部一阶输运、共享平衡物性和外层观测链。
历史 v1 动态路径的 WMS 热导率、A1 名义热导电压和 NDIR 名义电压
委托给 ``gf.sim.ar_he_co2``；正式声速经 ``gf.sim.a2dyn_sound_speed`` 路由，
v2 候选只在显式 pair 审计入口中委托给
``gf.sim.a2dyn_pair_virial``，这里不复制平衡物性公式。

组成统一使用 ``(Ar, He, CO2)`` 顺序的 mol%。解析状态更新不做 clip、
静默归一化或随机默认值。数组约定为时间优先，即 ``(time, channel)``。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import math
import os
from types import MappingProxyType
from typing import Any

import numpy as np

from gf.sim.ar_he_co2 import (
    A2DYN_OPERATIONAL_PRESSURE_RANGE_PA,
    A2DYN_TEMPERATURE_RANGE_K,
    SYSTEM_DELAY_S,
    ndir_co2_voltage,
    thermal_conductivity_voltage,
    wms_thermal_conductivity,
)
from gf.sim.a2dyn_sound_speed import (
    DIRECT_HEOS_SOUND_SPEED_MODEL_ID,
    a2dyn_sound_speed_for_model,
)
from gf.sim.a2dyn_pair_virial import (
    PAIR_SOUND_SPEED_MODEL_ID,
    pair_virial_sound_speed_for_model,
)


COMPOSITION_COMPONENTS = ("Ar", "He", "CO2")
SENSOR_IDS = (
    "ultrasonic_tof",
    "thermal_conductivity_voltage",
    "ndir_co2_voltage",
)
COMPOSITION_SUM_TOLERANCE_PCT = 1.0e-9
DEFAULT_QUANTIZATION_RESOLUTION = 0.01


def _sound_speed_for_audit(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    pressure_pa: float,
    *,
    model_id: str,
) -> float:
    if model_id == PAIR_SOUND_SPEED_MODEL_ID:
        return pair_virial_sound_speed_for_model(
            mole_fractions,
            temperature_k,
            pressure_pa,
            model_id=model_id,
        )
    return a2dyn_sound_speed_for_model(
        mole_fractions,
        temperature_k,
        pressure_pa,
        model_id=model_id,
    )


class DynamicPhysicsError(ValueError):
    """输入、状态或解析约束不满足时抛出的明确错误。"""


class PhysicsAuditError(DynamicPhysicsError):
    """注册物性参考核对未通过。"""

    def __init__(self, message: str, *, report: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report


def _finite_scalar(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise DynamicPhysicsError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise DynamicPhysicsError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise DynamicPhysicsError(f"{name} must be non-negative")
    return result


def _time_array(time_s: Any) -> tuple[np.ndarray, bool]:
    values = np.asarray(time_s, dtype=np.float64)
    scalar = values.ndim == 0
    if values.ndim > 1:
        raise DynamicPhysicsError("time_s must be a scalar or one-dimensional array")
    if not np.isfinite(values).all():
        raise DynamicPhysicsError("time_s must contain only finite values")
    if values.ndim == 1 and values.size > 1 and np.any(np.diff(values) <= 0.0):
        raise DynamicPhysicsError("time_s must be strictly increasing")
    return values, scalar


def _restore_scalar(values: np.ndarray, scalar: bool) -> float | np.ndarray:
    if scalar:
        return float(np.asarray(values).reshape(()))
    return values


def validate_composition_pct(
    composition_pct: Sequence[float] | np.ndarray,
    *,
    name: str = "composition_pct",
) -> np.ndarray:
    """校验一个 ``(Ar, He, CO2)`` mol% 组成并返回副本。"""

    values = np.asarray(composition_pct, dtype=np.float64)
    if values.ndim != 1 or values.shape != (3,):
        raise DynamicPhysicsError(f"{name} must have shape (3,) in Ar/He/CO2 order")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 100.0):
        raise DynamicPhysicsError(f"{name} must be finite and within [0,100] mol%")
    total = float(values.sum())
    if not math.isclose(total, 100.0, rel_tol=0.0, abs_tol=COMPOSITION_SUM_TOLERANCE_PCT):
        raise DynamicPhysicsError(f"{name} must sum to 100 mol%, got {total}")
    return values.copy()


def validate_composition_sequence_pct(
    compositions_pct: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str = "compositions_pct",
) -> np.ndarray:
    """校验时间序列组成，不修改任何输入值。"""

    values = np.asarray(compositions_pct, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, 3)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise DynamicPhysicsError(f"{name} must have shape (time, 3)")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 100.0):
        raise DynamicPhysicsError(f"{name} must be finite and within [0,100] mol%")
    totals = values.sum(axis=1)
    if not np.allclose(totals, 100.0, rtol=0.0, atol=COMPOSITION_SUM_TOLERANCE_PCT):
        index = int(np.flatnonzero(np.abs(totals - 100.0) > COMPOSITION_SUM_TOLERANCE_PCT)[0])
        raise DynamicPhysicsError(f"{name}[{index}] must sum to 100 mol%, got {totals[index]}")
    return values.copy()


def composition_pct_to_mole_fractions(
    composition_pct: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """把 mol% 显式转换为共享算子使用的摩尔分数。"""

    values = validate_composition_pct(composition_pct)
    return {
        component: float(values[index] / 100.0)
        for index, component in enumerate(COMPOSITION_COMPONENTS)
    }


def step_inlet_coefficient(time_s: Any, *, onset_s: float) -> float | np.ndarray:
    """阶跃进气系数：``0`` 在 onset 前，``1`` 在 onset 及之后。"""

    onset = _finite_scalar(onset_s, "onset_s", nonnegative=True)
    values, scalar = _time_array(time_s)
    coefficient = np.where(values < onset, 0.0, 1.0)
    return _restore_scalar(coefficient, scalar)


def linear_ramp_inlet_coefficient(
    time_s: Any,
    *,
    onset_s: float,
    duration_s: float,
) -> float | np.ndarray:
    """线性 ramp 进气系数，端点分别为 0 和 1。"""

    onset = _finite_scalar(onset_s, "onset_s", nonnegative=True)
    duration = _finite_scalar(duration_s, "duration_s", positive=True)
    values, scalar = _time_array(time_s)
    end = onset + duration
    coefficient = np.where(
        values <= onset,
        0.0,
        np.where(values >= end, 1.0, (values - onset) / duration),
    )
    return _restore_scalar(coefficient, scalar)


def smooth_ramp_inlet_coefficient(
    time_s: Any,
    *,
    onset_s: float,
    duration_s: float,
) -> float | np.ndarray:
    """smoothstep ramp 进气系数，内部为 ``3s²-2s³``。"""

    onset = _finite_scalar(onset_s, "onset_s", nonnegative=True)
    duration = _finite_scalar(duration_s, "duration_s", positive=True)
    values, scalar = _time_array(time_s)
    end = onset + duration
    progress = (values - onset) / duration
    shaped = 3.0 * progress**2 - 2.0 * progress**3
    coefficient = np.where(values <= onset, 0.0, np.where(values >= end, 1.0, shaped))
    return _restore_scalar(coefficient, scalar)


def protocol_inlet_coefficient(
    time_s: Any,
    *,
    kind: str,
    onset_s: float,
    exposure_end_s: float | None = None,
    ramp_duration_s: float | None = None,
    exposure_duration_s: float | None = None,
    pulse_count: int | None = None,
    pulse_width_s: float | None = None,
    pulse_period_s: float | None = None,
    recovery_residual: float | None = None,
) -> float | np.ndarray:
    """按注册的 protocol kind 生成进气系数。

    ``multi_pulse`` 要求显式给出 pulse 数、宽度和周期；不会猜测脉冲参数。
    step/ramp/shifted-onset 的 recovery 边界由 ``exposure_end_s`` 显式给出；
    ``incomplete_recovery`` 在暴露结束后保持显式给定的残留系数。
    """

    normalized_kind = str(kind).strip().lower().replace("-", "_")
    values, scalar = _time_array(time_s)
    onset = _finite_scalar(onset_s, "onset_s", nonnegative=True)

    if normalized_kind in {"step", "step_standard", "shifted_onset", "onset_shift"}:
        result = np.asarray(step_inlet_coefficient(values, onset_s=onset), dtype=np.float64)
    elif normalized_kind in {"ramp", "ramp_linear", "linear_ramp"}:
        if ramp_duration_s is None:
            raise DynamicPhysicsError("ramp_duration_s is required for a linear ramp")
        result = np.asarray(
            linear_ramp_inlet_coefficient(values, onset_s=onset, duration_s=ramp_duration_s),
            dtype=np.float64,
        )
    elif normalized_kind in {"smooth_ramp", "ramp_smooth"}:
        if ramp_duration_s is None:
            raise DynamicPhysicsError("ramp_duration_s is required for a smooth ramp")
        result = np.asarray(
            smooth_ramp_inlet_coefficient(values, onset_s=onset, duration_s=ramp_duration_s),
            dtype=np.float64,
        )
    elif normalized_kind in {"short_pulse", "pulse"}:
        if exposure_duration_s is None:
            raise DynamicPhysicsError("exposure_duration_s is required for a short pulse")
        duration = _finite_scalar(exposure_duration_s, "exposure_duration_s", positive=True)
        end = onset + duration
        result = np.where((values >= onset) & (values < end), 1.0, 0.0)
    elif normalized_kind in {"incomplete_recovery", "recovery"}:
        if exposure_duration_s is None:
            raise DynamicPhysicsError("exposure_duration_s is required for incomplete recovery")
        if recovery_residual is None:
            raise DynamicPhysicsError("recovery_residual is required for incomplete recovery")
        duration = _finite_scalar(exposure_duration_s, "exposure_duration_s", positive=True)
        residual = _finite_scalar(recovery_residual, "recovery_residual", nonnegative=True)
        if residual > 1.0:
            raise DynamicPhysicsError("recovery_residual must be within [0,1]")
        end = onset + duration
        result = np.where(values < onset, 0.0, np.where(values < end, 1.0, residual))
    elif normalized_kind in {"multi_pulse", "multipulse"}:
        if pulse_count is None or isinstance(pulse_count, bool) or int(pulse_count) != pulse_count:
            raise DynamicPhysicsError("pulse_count must be an explicit integer for multi_pulse")
        count = int(pulse_count)
        if count < 2:
            raise DynamicPhysicsError("pulse_count must be at least 2")
        if pulse_width_s is None or pulse_period_s is None:
            raise DynamicPhysicsError("pulse_width_s and pulse_period_s are required for multi_pulse")
        width = _finite_scalar(pulse_width_s, "pulse_width_s", positive=True)
        period = _finite_scalar(pulse_period_s, "pulse_period_s", positive=True)
        if width > period:
            raise DynamicPhysicsError("pulse_width_s must not exceed pulse_period_s")
        result = np.zeros_like(values, dtype=np.float64)
        for pulse_index in range(count):
            pulse_start = onset + pulse_index * period
            pulse_end = pulse_start + width
            result = np.where((values >= pulse_start) & (values < pulse_end), 1.0, result)
    else:
        raise DynamicPhysicsError(f"unsupported protocol kind {kind!r}")

    if normalized_kind in {
        "step",
        "step_standard",
        "shifted_onset",
        "onset_shift",
        "ramp",
        "ramp_linear",
        "linear_ramp",
        "smooth_ramp",
        "ramp_smooth",
    }:
        if exposure_end_s is not None:
            exposure_end = _finite_scalar(
                exposure_end_s,
                "exposure_end_s",
                nonnegative=True,
            )
            if exposure_end <= onset:
                raise DynamicPhysicsError("exposure_end_s must be after onset_s")
            result = np.where(values < exposure_end, result, 0.0)

    if not np.isfinite(result).all() or np.any(result < 0.0) or np.any(result > 1.0):
        raise DynamicPhysicsError(f"protocol {kind!r} produced coefficient outside [0,1]")
    return _restore_scalar(result, scalar)


def build_inlet_composition(
    time_s: Any,
    *,
    purge_composition_pct: Sequence[float],
    target_composition_pct: Sequence[float],
    coefficient: Any,
) -> np.ndarray:
    """按冻结公式 ``u=(1-b)·purge+b·target`` 构造进气组成。"""

    purge = validate_composition_pct(purge_composition_pct, name="purge_composition_pct")
    target = validate_composition_pct(target_composition_pct, name="target_composition_pct")
    times, scalar = _time_array(time_s)
    times_1d = np.atleast_1d(times)
    coefficient_array = np.asarray(coefficient, dtype=np.float64)
    if coefficient_array.ndim == 0:
        coefficient_array = np.full(times_1d.shape, float(coefficient_array), dtype=np.float64)
    elif scalar:
        coefficient_array = np.atleast_1d(coefficient_array)
    if coefficient_array.shape != times_1d.shape:
        raise DynamicPhysicsError("coefficient must have the same shape as time_s")
    if not np.isfinite(coefficient_array).all() or np.any(coefficient_array < 0.0) or np.any(coefficient_array > 1.0):
        raise DynamicPhysicsError("coefficient must be finite and within [0,1]")
    result = (1.0 - coefficient_array[:, None]) * purge + coefficient_array[:, None] * target
    sequence = validate_composition_sequence_pct(result, name="inlet_composition_pct")
    return sequence[0] if scalar else sequence


def analytic_exponential_update(
    previous_state: Any,
    forcing_state: Any,
    *,
    dt_s: float,
    tau_s: float,
) -> np.ndarray:
    """单步解析一阶更新：``x_next=u+(x_prev-u)exp(-dt/tau)``。"""

    dt = _finite_scalar(dt_s, "dt_s", positive=True)
    tau = _finite_scalar(tau_s, "tau_s", nonnegative=True)
    previous = np.asarray(previous_state, dtype=np.float64)
    forcing = np.asarray(forcing_state, dtype=np.float64)
    if previous.shape != forcing.shape:
        raise DynamicPhysicsError("previous_state and forcing_state must have the same shape")
    if not np.isfinite(previous).all() or not np.isfinite(forcing).all():
        raise DynamicPhysicsError("states must contain only finite values")
    if tau == 0.0:
        return forcing.copy()
    decay = math.exp(-dt / tau)
    return forcing + (previous - forcing) * decay


def simulate_first_order_series(
    forcing: Sequence[float] | np.ndarray,
    *,
    dt_s: float,
    tau_s: float,
    initial_state: Sequence[float] | float | None = None,
) -> np.ndarray:
    """对标量或多通道序列执行解析一阶更新。

    返回序列的第一个点是明确给定的初始状态；第 ``k`` 点由第 ``k`` 个
    forcing 作用一个 ``dt_s`` 后得到。这样阶段边界和初始条件均可审计。
    """

    values = np.asarray(forcing, dtype=np.float64)
    if values.ndim == 0 or values.shape[0] == 0:
        raise DynamicPhysicsError("forcing must contain a non-empty time axis")
    if not np.isfinite(values).all():
        raise DynamicPhysicsError("forcing must contain only finite values")
    _finite_scalar(dt_s, "dt_s", positive=True)
    _finite_scalar(tau_s, "tau_s", nonnegative=True)
    if initial_state is None:
        initial = np.asarray(values[0], dtype=np.float64).copy()
    else:
        initial = np.asarray(initial_state, dtype=np.float64)
        if initial.shape != values.shape[1:]:
            raise DynamicPhysicsError(
                f"initial_state shape {initial.shape} does not match forcing channel shape {values.shape[1:]}"
            )
        if not np.isfinite(initial).all():
            raise DynamicPhysicsError("initial_state must contain only finite values")
    output = np.empty_like(values, dtype=np.float64)
    output[0] = initial
    state = initial
    for index in range(1, values.shape[0]):
        state = analytic_exponential_update(state, values[index], dt_s=dt_s, tau_s=tau_s)
        output[index] = state
    return output


def simulate_well_mixed_chamber(
    inlet_composition_pct: Sequence[Sequence[float]] | np.ndarray,
    *,
    dt_s: float,
    tau_mix_s: float,
    initial_composition_pct: Sequence[float] | None = None,
) -> np.ndarray:
    """共同气室 CSTR 的解析更新。"""

    inlet = validate_composition_sequence_pct(inlet_composition_pct, name="inlet_composition_pct")
    initial = None if initial_composition_pct is None else validate_composition_pct(
        initial_composition_pct,
        name="initial_composition_pct",
    )
    state = simulate_first_order_series(
        inlet,
        dt_s=dt_s,
        tau_s=tau_mix_s,
        initial_state=initial,
    )
    return validate_composition_sequence_pct(state, name="chamber_composition_pct")


def simulate_local_transport(
    chamber_composition_pct: Sequence[Sequence[float]] | np.ndarray,
    *,
    dt_s: float,
    tau_transport_s: float,
    initial_composition_pct: Sequence[float] | None = None,
) -> np.ndarray:
    """单路局部气室 / 传输的一阶解析更新。"""

    chamber = validate_composition_sequence_pct(chamber_composition_pct, name="chamber_composition_pct")
    initial = None if initial_composition_pct is None else validate_composition_pct(
        initial_composition_pct,
        name="initial_composition_pct",
    )
    state = simulate_first_order_series(
        chamber,
        dt_s=dt_s,
        tau_s=tau_transport_s,
        initial_state=initial,
    )
    return validate_composition_sequence_pct(state, name="local_composition_pct")


@dataclass(frozen=True)
class DynamicTransportLayers:
    """共同气室和三路局部输运的可审计结果。"""

    chamber_composition_pct: np.ndarray
    local_composition_pct: Mapping[str, np.ndarray]


def simulate_dynamic_layers(
    inlet_composition_pct: Sequence[Sequence[float]] | np.ndarray,
    *,
    dt_s: float,
    tau_mix_s: float,
    tau_transport_s: Mapping[str, float],
    initial_composition_pct: Sequence[float] | None = None,
) -> DynamicTransportLayers:
    """按固定顺序执行共同气室，再执行三路局部输运。"""

    if set(tau_transport_s) != set(SENSOR_IDS):
        raise DynamicPhysicsError(f"tau_transport_s must cover exactly {list(SENSOR_IDS)}")
    chamber = simulate_well_mixed_chamber(
        inlet_composition_pct,
        dt_s=dt_s,
        tau_mix_s=tau_mix_s,
        initial_composition_pct=initial_composition_pct,
    )
    local = {
        sensor_id: simulate_local_transport(
            chamber,
            dt_s=dt_s,
            tau_transport_s=tau_transport_s[sensor_id],
            initial_composition_pct=initial_composition_pct,
        )
        for sensor_id in SENSOR_IDS
    }
    return DynamicTransportLayers(
        chamber_composition_pct=chamber,
        local_composition_pct=MappingProxyType(local),
    )


def evaluate_shared_physics(
    compositions_pct: Sequence[Sequence[float]] | np.ndarray,
    *,
    temperature_k: float | Sequence[float] | np.ndarray,
    pressure_pa: float | Sequence[float] | np.ndarray,
    path_length_m: float,
    system_delay_s: float = SYSTEM_DELAY_S,
    sound_speed_model_id: str = DIRECT_HEOS_SOUND_SPEED_MODEL_ID,
) -> dict[str, np.ndarray]:
    """调用注册的唯一平衡物性算子。

    返回键为 ``sound_speed_m_s``、``conductivity_w_m_k``、``tof_s``、
    ``thermal_voltage_v`` 和 ``ndir_voltage_v``。温度与压力可以是标量，
    也可以是与时间长度相同的一维数组。
    """

    compositions = validate_composition_sequence_pct(compositions_pct)
    path = _finite_scalar(path_length_m, "path_length_m", positive=True)
    delay = _finite_scalar(system_delay_s, "system_delay_s", nonnegative=True)
    count = compositions.shape[0]
    temperatures = _broadcast_environment(temperature_k, count, "temperature_k", positive=True)
    pressures = _broadcast_environment(pressure_pa, count, "pressure_pa", positive=True)
    sound_speed = np.empty(count, dtype=np.float64)
    conductivity = np.empty(count, dtype=np.float64)
    tof = np.empty(count, dtype=np.float64)
    thermal_voltage = np.empty(count, dtype=np.float64)
    ndir_voltage = np.empty(count, dtype=np.float64)
    for index, composition in enumerate(compositions):
        fractions = composition_pct_to_mole_fractions(composition)
        sound_speed[index] = a2dyn_sound_speed_for_model(
            fractions,
            float(temperatures[index]),
            float(pressures[index]),
            model_id=sound_speed_model_id,
        )
        conductivity[index] = wms_thermal_conductivity(fractions)
        tof[index] = path / sound_speed[index] + delay
        thermal_voltage[index] = thermal_conductivity_voltage(conductivity[index])
        ndir_voltage[index] = ndir_co2_voltage(
            float(composition[2]),
            float(pressures[index]),
            float(temperatures[index]),
        )
    return {
        "sound_speed_m_s": sound_speed,
        "conductivity_w_m_k": conductivity,
        "tof_s": tof,
        "thermal_voltage_v": thermal_voltage,
        "ndir_voltage_v": ndir_voltage,
    }


def audit_coolprop_sound_speed_grid(
    *,
    temperature_values_k: Sequence[float],
    pressure_values_pa: Sequence[float],
    simplex_step_pct: float = 1.0,
    max_relative_error: float = 0.005,
    raise_on_failure: bool = True,
    max_workers: int | None = None,
    sound_speed_model_id: str = DIRECT_HEOS_SOUND_SPEED_MODEL_ID,
    off_grid_count: int = 0,
    off_grid_seed: int = 20260831,
    check_pressure_direction: bool = False,
) -> dict[str, Any]:
    """在注册的组成单纯形和温压块上核对共享声速。

    CoolProp 是审计参考，不会替换 ``ar_he_co2.py`` 的生成时事实源。
    未安装参考库或任一点查询失败都会显式报错；误差超过门限时不会被
    噪声或校准项掩盖。离网审计和压力导数方向检查按显式参数启用。
    """

    step = _finite_scalar(simplex_step_pct, "simplex_step_pct", positive=True)
    units = int(round(100.0 / step))
    if not math.isclose(units * step, 100.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise PhysicsAuditError("simplex_step_pct must divide 100 exactly")
    gate = _finite_scalar(max_relative_error, "max_relative_error", nonnegative=True)
    temperatures = _validated_grid_values(temperature_values_k, "temperature_values_k")
    pressures = _validated_grid_values(pressure_values_pa, "pressure_values_pa")
    if not isinstance(off_grid_count, int) or isinstance(off_grid_count, bool) or off_grid_count < 0:
        raise PhysicsAuditError("off_grid_count must be a non-negative integer")
    if not isinstance(off_grid_seed, int) or isinstance(off_grid_seed, bool) or off_grid_seed < 0:
        raise PhysicsAuditError("off_grid_seed must be a non-negative integer")
    try:
        import CoolProp
        import CoolProp.CoolProp as coolprop
    except ImportError as exc:
        raise PhysicsAuditError("CoolProp is required for the registered EOS grid audit") from exc
    try:
        compositions = [
            (ar_units, he_units, units - ar_units - he_units)
            for ar_units in range(units + 1)
            for he_units in range(units - ar_units + 1)
        ]
        requested_workers = min(8, os.cpu_count() or 1) if max_workers is None else max_workers
        if requested_workers is None or isinstance(requested_workers, bool) or int(requested_workers) != requested_workers:
            raise PhysicsAuditError("max_workers must be a positive integer")
        worker_count = int(requested_workers)
        if worker_count <= 0:
            raise PhysicsAuditError("max_workers must be a positive integer")
        worker_count = min(worker_count, len(compositions))
        chunks = [compositions[index::worker_count] for index in range(worker_count)]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _coolprop_error_chunk,
                    chunk,
                    step,
                    temperatures,
                    pressures,
                    sound_speed_model_id,
                    check_pressure_direction,
                )
                for chunk in chunks
            ]
            worker_results = [future.result() for future in futures]
            error_entries = [entry for errors, _ in worker_results for entry in errors]
            direction_entries = [entry for _, directions in worker_results for entry in directions]
    except PhysicsAuditError:
        raise
    except Exception as exc:
        raise PhysicsAuditError("CoolProp EOS grid query failed") from exc
    signed_array = np.asarray([entry[0] for entry in error_entries], dtype=np.float64)
    error_array = np.abs(signed_array)
    worst_error, worst_composition, worst_temperature, worst_pressure, worst_model, worst_reference = max(
        error_entries,
        key=lambda entry: abs(entry[0]),
    )
    off_grid_report = _audit_coolprop_off_grid(
        sound_speed_model_id=sound_speed_model_id,
        count=off_grid_count,
        seed=off_grid_seed,
        max_relative_error=gate,
        check_pressure_direction=check_pressure_direction,
    )
    direction_passed = all(entry["match"] for entry in direction_entries)
    grid_passed = bool(float(error_array.max()) <= gate)
    off_grid_passed = off_grid_count == 0 or off_grid_report["status"] == "PASS"
    grid_status = "PASS" if grid_passed else "FAIL"
    status = "PASS" if grid_passed and off_grid_passed and direction_passed else "FAIL"
    report: dict[str, Any] = {
        "status": status,
        "grid_status": grid_status,
        "backend": "CoolProp",
        "phase_constraint": "gas",
        "package_version": str(getattr(CoolProp, "__version__", "unknown")),
        "sound_speed_model_id": sound_speed_model_id,
        "fluid_names": ["Argon", "Helium", "CarbonDioxide"],
        "simplex_step_pct": step,
        "composition_count": (units + 1) * (units + 2) // 2,
        "temperature_count": int(temperatures.size),
        "pressure_count": int(pressures.size),
        "query_count": int(error_array.size),
        "max_relative_error": float(error_array.max()),
        "max_error_case": {
            "composition_pct": [float(value * step) for value in worst_composition],
            "temperature_k": worst_temperature,
            "pressure_pa": worst_pressure,
            "relative_error": abs(worst_error),
            "signed_relative_error": worst_error,
            "model_speed_m_s": worst_model,
            "reference_speed_m_s": worst_reference,
        },
        "relative_error_percentiles": {
            "p50": float(np.percentile(error_array, 50.0)),
            "p95": float(np.percentile(error_array, 95.0)),
            "p100": float(np.percentile(error_array, 100.0)),
        },
        "absolute_error_percentiles": {
            "p50": float(np.percentile(np.abs(np.asarray([entry[4] - entry[5] for entry in error_entries])), 50.0)),
            "p95": float(np.percentile(np.abs(np.asarray([entry[4] - entry[5] for entry in error_entries])), 95.0)),
            "p100": float(np.percentile(np.abs(np.asarray([entry[4] - entry[5] for entry in error_entries])), 100.0)),
        },
        "signed_error_percentiles": {
            "p50": float(np.percentile(signed_array, 50.0)),
            "p95": float(np.percentile(signed_array, 95.0)),
            "p100": float(np.percentile(signed_array, 100.0)),
        },
        "max_relative_error_gate": gate,
        "off_grid": off_grid_report,
        "pressure_direction": {
            "checked": check_pressure_direction,
            "case_count": len(direction_entries),
            "mismatch_count": sum(not entry["match"] for entry in direction_entries),
            "status": "PASS" if direction_passed else "FAIL",
            "first_mismatch": next(
                (entry for entry in direction_entries if not entry["match"]),
                None,
            ),
        },
    }
    if report["status"] != "PASS" and raise_on_failure:
        raise PhysicsAuditError(
            f"CoolProp EOS audit failed: grid_max={report['max_relative_error']}, off_grid_max={off_grid_report['max_relative_error']}, direction={report['pressure_direction']['status']}",
            report=report,
        )
    return report


def _coolprop_error_chunk(
    compositions: Sequence[tuple[int, int, int]],
    step: float,
    temperatures: np.ndarray,
    pressures: np.ndarray,
    sound_speed_model_id: str,
    check_pressure_direction: bool,
) -> tuple[
    list[tuple[float, tuple[int, int, int], float, float, float, float]],
    list[dict[str, Any]],
]:
    import CoolProp.CoolProp as coolprop

    state = coolprop.AbstractState("HEOS", "Argon&Helium&CarbonDioxide")
    input_pair = coolprop.PT_INPUTS
    fluid_names = ("Argon", "Helium", "CarbonDioxide")
    errors: list[tuple[float, tuple[int, int, int], float, float, float, float]] = []
    directions: list[dict[str, Any]] = []
    for ar_units, he_units, co2_units in compositions:
        fractions = np.asarray([ar_units, he_units, co2_units], dtype=np.float64) / float(100.0 / step)
        active_state = state
        nonzero = np.flatnonzero(fractions > 0.0)
        if nonzero.size == 1:
            active_state = coolprop.AbstractState("HEOS", fluid_names[int(nonzero[0])])
        else:
            active_state.set_mole_fractions(
                [float(fractions[0]), float(fractions[1]), float(fractions[2])]
            )
        active_state.specify_phase(coolprop.iphase_gas)
        fraction_map = {
            "Ar": float(fractions[0]),
            "He": float(fractions[1]),
            "CO2": float(fractions[2]),
        }
        for temperature in temperatures:
            model_speeds: list[float] = []
            reference_speeds: list[float] = []
            for pressure in pressures:
                active_state.update(input_pair, float(pressure), float(temperature))
                reference_speed = float(active_state.speed_sound())
                if not math.isfinite(reference_speed) or reference_speed <= 0.0:
                    raise PhysicsAuditError("CoolProp returned an invalid speed of sound")
                model_speed = _sound_speed_for_audit(
                    fraction_map,
                    float(temperature),
                    float(pressure),
                    model_id=sound_speed_model_id,
                )
                model_speeds.append(model_speed)
                reference_speeds.append(reference_speed)
                errors.append(
                    (
                        (model_speed - reference_speed) / reference_speed,
                        (ar_units, he_units, co2_units),
                        float(temperature),
                        float(pressure),
                        model_speed,
                        reference_speed,
                    )
                )
            if check_pressure_direction and len(pressures) >= 2:
                model_delta = model_speeds[-1] - model_speeds[0]
                reference_delta = reference_speeds[-1] - reference_speeds[0]
                model_direction = 0 if math.isclose(model_delta, 0.0, abs_tol=1.0e-12) else int(math.copysign(1, model_delta))
                reference_direction = 0 if math.isclose(reference_delta, 0.0, abs_tol=1.0e-12) else int(math.copysign(1, reference_delta))
                directions.append(
                    {
                        "composition_pct": [ar_units * step, he_units * step, (100.0 / step - ar_units - he_units) * step],
                        "temperature_k": float(temperature),
                        "model_direction": model_direction,
                        "reference_direction": reference_direction,
                        "match": model_direction == reference_direction,
                    }
                )
    return errors, directions


def _audit_coolprop_off_grid(
    *,
    sound_speed_model_id: str,
    count: int,
    seed: int,
    max_relative_error: float,
    check_pressure_direction: bool,
) -> dict[str, Any]:
    if count == 0:
        return {
            "status": "NOT_RUN",
            "count": 0,
            "seed": seed,
            "sound_speed_model_id": sound_speed_model_id,
            "max_relative_error": None,
            "relative_error_percentiles": None,
            "absolute_error_percentiles_m_s": None,
            "max_error_case": None,
            "pressure_direction_checked": False,
        }
    try:
        import CoolProp.CoolProp as coolprop
    except ImportError as exc:
        raise PhysicsAuditError("CoolProp is required for the off-grid EOS audit") from exc

    state = coolprop.AbstractState("HEOS", "Argon&Helium&CarbonDioxide")
    temperature_lower, temperature_upper = A2DYN_TEMPERATURE_RANGE_K
    pressure_lower, pressure_upper = A2DYN_OPERATIONAL_PRESSURE_RANGE_PA
    signed_errors: list[float] = []
    absolute_errors_m_s: list[float] = []
    cases: list[dict[str, Any]] = []
    for offset in range(1, count + 1):
        index = seed + offset
        u1 = _radical_inverse(index, 2)
        u2 = _radical_inverse(index, 3)
        u3 = _radical_inverse(index, 5)
        u4 = _radical_inverse(index, 7)
        radius = math.sqrt(u1)
        composition = {
            "Ar": 1.0 - radius,
            "He": radius * (1.0 - u2),
            "CO2": radius * u2,
        }
        temperature = temperature_lower + (temperature_upper - temperature_lower) * u3
        pressure = pressure_lower + (pressure_upper - pressure_lower) * u4
        state.set_mole_fractions(
            [composition["Ar"], composition["He"], composition["CO2"]]
        )
        state.specify_phase(coolprop.iphase_gas)
        state.update(coolprop.PT_INPUTS, pressure, temperature)
        reference_speed = float(state.speed_sound())
        model_speed = _sound_speed_for_audit(
            composition,
            temperature,
            pressure,
            model_id=sound_speed_model_id,
        )
        if not math.isfinite(reference_speed) or reference_speed <= 0.0:
            raise PhysicsAuditError("CoolProp returned an invalid off-grid speed of sound")
        signed_error = (model_speed - reference_speed) / reference_speed
        signed_errors.append(signed_error)
        absolute_errors_m_s.append(abs(model_speed - reference_speed))
        cases.append(
            {
                "composition_pct": [
                    composition["Ar"] * 100.0,
                    composition["He"] * 100.0,
                    composition["CO2"] * 100.0,
                ],
                "temperature_k": temperature,
                "pressure_pa": pressure,
                "relative_error": abs(signed_error),
                "signed_relative_error": signed_error,
                "model_speed_m_s": model_speed,
                "reference_speed_m_s": reference_speed,
            }
        )
    error_array = np.abs(np.asarray(signed_errors, dtype=np.float64))
    absolute_array = np.asarray(absolute_errors_m_s, dtype=np.float64)
    worst_index = int(np.argmax(error_array))
    return {
        "status": "PASS" if float(error_array.max()) <= max_relative_error else "FAIL",
        "count": count,
        "seed": seed,
        "construction": "nested-radical-inverse-simplex",
        "sound_speed_model_id": sound_speed_model_id,
        "max_relative_error": float(error_array.max()),
        "relative_error_percentiles": {
            "p50": float(np.percentile(error_array, 50.0)),
            "p95": float(np.percentile(error_array, 95.0)),
            "p100": float(np.percentile(error_array, 100.0)),
        },
        "signed_error_percentiles": {
            "p50": float(np.percentile(np.asarray(signed_errors), 50.0)),
            "p95": float(np.percentile(np.asarray(signed_errors), 95.0)),
            "p100": float(np.percentile(np.asarray(signed_errors), 100.0)),
        },
        "absolute_error_percentiles_m_s": {
            "p50": float(np.percentile(absolute_array, 50.0)),
            "p95": float(np.percentile(absolute_array, 95.0)),
            "p100": float(np.percentile(absolute_array, 100.0)),
        },
        "max_error_case": cases[worst_index],
        "max_relative_error_gate": max_relative_error,
        "pressure_direction_checked": False,
    }


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    remaining = index
    while remaining:
        remaining, digit = divmod(remaining, base)
        value += digit * factor
        factor /= base
    return value


def _validated_grid_values(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all() or np.any(result <= 0.0):
        raise DynamicPhysicsError(f"{name} must be a non-empty finite positive one-dimensional sequence")
    return result


def _broadcast_environment(
    value: float | Sequence[float] | np.ndarray,
    count: int,
    name: str,
    *,
    positive: bool,
) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(count, float(values), dtype=np.float64)
    if values.shape != (count,):
        raise DynamicPhysicsError(f"{name} must be a scalar or shape ({count},)")
    if not np.isfinite(values).all() or (positive and np.any(values <= 0.0)):
        adjective = "finite and positive" if positive else "finite"
        raise DynamicPhysicsError(f"{name} must be {adjective}")
    return values


def generate_ar1_noise(
    length: int,
    *,
    rho: float,
    innovation_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成显式 AR(1) 过程，``innovation_std`` 是 epsilon 的边际尺度。"""

    if isinstance(length, bool) or int(length) != length or int(length) <= 0:
        raise DynamicPhysicsError("length must be a positive integer")
    count = int(length)
    correlation = _finite_scalar(rho, "rho")
    if correlation < 0.0 or correlation >= 1.0:
        raise DynamicPhysicsError("rho must satisfy 0 <= rho < 1")
    scale = _finite_scalar(innovation_std, "innovation_std", nonnegative=True)
    if not isinstance(rng, np.random.Generator):
        raise DynamicPhysicsError("rng must be an explicit numpy.random.Generator")
    innovations = rng.normal(loc=0.0, scale=scale, size=count)
    result = np.empty(count, dtype=np.float64)
    result[0] = innovations[0]
    coefficient = math.sqrt(1.0 - correlation**2)
    for index in range(1, count):
        result[index] = correlation * result[index - 1] + coefficient * innovations[index]
    return result


def generate_shared_noise(
    length: int,
    *,
    rho: float,
    innovation_std: float,
    channel_loadings: Sequence[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """公共 AR(1) 过程投影到固定通道载荷，避免复制成完全相同的噪声。"""

    loadings = np.asarray(channel_loadings, dtype=np.float64)
    if loadings.ndim != 1 or loadings.size == 0 or not np.isfinite(loadings).all():
        raise DynamicPhysicsError("channel_loadings must be a non-empty finite vector")
    common = generate_ar1_noise(
        length,
        rho=rho,
        innovation_std=innovation_std,
        rng=rng,
    )
    return common[:, None] * loadings[None, :]


def linear_sequence_drift(
    length: int,
    *,
    intercept: float | Sequence[float],
    slope_per_step: float | Sequence[float],
) -> np.ndarray:
    """生成序列级随机截距和低频斜率对应的确定性漂移。"""

    if isinstance(length, bool) or int(length) != length or int(length) <= 0:
        raise DynamicPhysicsError("length must be a positive integer")
    count = int(length)
    intercept_array = np.asarray(intercept, dtype=np.float64)
    slope_array = np.asarray(slope_per_step, dtype=np.float64)
    if intercept_array.shape != slope_array.shape:
        raise DynamicPhysicsError("intercept and slope_per_step must have the same shape")
    if intercept_array.ndim > 1 or not np.isfinite(intercept_array).all() or not np.isfinite(slope_array).all():
        raise DynamicPhysicsError("drift parameters must be finite scalar or one-dimensional arrays")
    steps = np.arange(count, dtype=np.float64)[:, None]
    return intercept_array.reshape(1, -1) + steps * slope_array.reshape(1, -1)


def quantize_signal(
    values: Sequence[float] | np.ndarray,
    resolution: float | Sequence[float] | np.ndarray = DEFAULT_QUANTIZATION_RESOLUTION,
) -> np.ndarray:
    """执行末端量化；不对越界值做 clip。"""

    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim == 0 or not np.isfinite(signal).all():
        raise DynamicPhysicsError("values must be a non-scalar finite array")
    step = np.asarray(resolution, dtype=np.float64)
    if not np.isfinite(step).all() or np.any(step <= 0.0):
        raise DynamicPhysicsError("resolution must be finite and positive")
    if step.ndim == 1 and signal.ndim >= 2:
        if step.shape != (signal.shape[-1],):
            raise DynamicPhysicsError("channel resolution must match the last signal dimension")
    elif step.ndim == 1 and signal.ndim == 1:
        if step.shape != (1,):
            raise DynamicPhysicsError("one-dimensional values require a scalar or one-channel resolution")
    elif step.ndim not in {0, 1}:
        raise DynamicPhysicsError("resolution must be scalar or one-dimensional")
    return np.round(signal / step) * step


def apply_observation_chain(
    clean_signal: Sequence[float] | np.ndarray,
    *,
    gain: float | Sequence[float] | np.ndarray = 1.0,
    offset: float | Sequence[float] | np.ndarray = 0.0,
    drift: Sequence[float] | np.ndarray | None = None,
    correlated_noise: Sequence[float] | np.ndarray | None = None,
    white_noise: Sequence[float] | np.ndarray | None = None,
    quantization_resolution: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """按 gain/offset → drift → correlated → white → quantization 固定顺序处理。"""

    clean = np.asarray(clean_signal, dtype=np.float64)
    if clean.ndim == 1:
        clean = clean[:, None]
    if clean.ndim != 2 or clean.shape[0] == 0 or not np.isfinite(clean).all():
        raise DynamicPhysicsError("clean_signal must be a non-empty finite (time, channel) array")
    channels = clean.shape[1]
    gain_array = _channel_parameter(gain, channels, "gain")
    offset_array = _channel_parameter(offset, channels, "offset")
    result = clean * gain_array[None, :] + offset_array[None, :]
    for name, contribution in (
        ("drift", drift),
        ("correlated_noise", correlated_noise),
        ("white_noise", white_noise),
    ):
        if contribution is None:
            continue
        noise = np.asarray(contribution, dtype=np.float64)
        if noise.ndim == 1 and channels == 1 and noise.shape == (clean.shape[0],):
            noise = noise[:, None]
        if noise.shape != clean.shape or not np.isfinite(noise).all():
            raise DynamicPhysicsError(f"{name} must have shape {clean.shape} and finite values")
        result = result + noise
    quantized = quantize_signal(result, quantization_resolution)
    if not np.isfinite(quantized).all():
        raise DynamicPhysicsError("observation chain produced non-finite values")
    return quantized


def _channel_parameter(value: Any, channels: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(channels, float(values), dtype=np.float64)
    if values.shape != (channels,) or not np.isfinite(values).all():
        raise DynamicPhysicsError(f"{name} must be a scalar or shape ({channels},) with finite values")
    return values


# 便于调用方按计划文档中的术语使用。
inlet_coefficient = protocol_inlet_coefficient
simulate_chamber = simulate_well_mixed_chamber
simulate_transport = simulate_local_transport
ar1_noise = generate_ar1_noise
shared_noise = generate_shared_noise
apply_observation_perturbations = apply_observation_chain


__all__ = [
    "COMPOSITION_COMPONENTS",
    "COMPOSITION_SUM_TOLERANCE_PCT",
    "DEFAULT_QUANTIZATION_RESOLUTION",
    "DynamicPhysicsError",
    "DynamicTransportLayers",
    "PhysicsAuditError",
    "SENSOR_IDS",
    "analytic_exponential_update",
    "audit_coolprop_sound_speed_grid",
    "apply_observation_chain",
    "apply_observation_perturbations",
    "ar1_noise",
    "build_inlet_composition",
    "composition_pct_to_mole_fractions",
    "evaluate_shared_physics",
    "generate_ar1_noise",
    "generate_shared_noise",
    "inlet_coefficient",
    "linear_ramp_inlet_coefficient",
    "linear_sequence_drift",
    "protocol_inlet_coefficient",
    "quantize_signal",
    "shared_noise",
    "simulate_chamber",
    "simulate_dynamic_layers",
    "simulate_first_order_series",
    "simulate_local_transport",
    "simulate_transport",
    "simulate_well_mixed_chamber",
    "smooth_ramp_inlet_coefficient",
    "step_inlet_coefficient",
    "validate_composition_pct",
    "validate_composition_sequence_pct",
]
