"""A2-DYN 三类设备采集模型。

设备层只把 ``a2_dynamic_physics`` 产生的局部组成转换为采集读数，
平衡声速调用正式 A2-DYN 路由；其余历史物性调用 ``ar_he_co2.py``。
超声内部波形只在一次 acquisition 调用内存活；默认结果不携带波形。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from scipy.signal import correlate

from gf.sim.a2_dynamic_physics import (
    composition_pct_to_mole_fractions,
    simulate_first_order_series,
    validate_composition_pct,
    validate_composition_sequence_pct,
)
from gf.sim.ar_he_co2 import (
    NDIR_BASELINE_V,
    SYSTEM_DELAY_S,
    TCS_REFERENCE_CONDUCTIVITY_W_M_K,
    TCS_RESPONSE_V_PER_W_M_K,
    thermal_conductivity_voltage,
    wms_thermal_conductivity,
)
from gf.sim.a2dyn_sound_speed import (
    DIRECT_HEOS_SOUND_SPEED_MODEL_ID,
    a2dyn_sound_speed_for_model,
)


ULTRASONIC_LOCK_CORRELATION_THRESHOLD = 0.35
ULTRASONIC_LOCK_SNR_THRESHOLD = 3.0
NDIR_REFERENCE_CHANNEL_V = 1.0
NDIR_ADC_MIN_V = 0.0
NDIR_ADC_MAX_V = 3.3
TCD_REFERENCE_GAS_CONDUCTANCE_W_PER_K = 0.020
TCD_BRIDGE_CALIBRATION_GAIN = 100.0
HITRAN_REFERENCE_CACHE_DIR = Path(__file__).resolve().parents[4] / "shared" / "hitran_cache"
_BOLTZMANN_J_PER_K = 1.380649e-23
_PA_PER_ATM = 101325.0
_HITRAN_CO2_GRID_STEP_PCT = 0.02
_HITRAN_GRID_CHUNK_SIZE = 256
_HITRAN_REGISTERED_TABLE_NAME = "CO2_2250p0000_2445p0000"
_HITRAN_RUNTIME_CACHE: tempfile.TemporaryDirectory[str] | None = None


class SensorDeviceError(ValueError):
    """设备参数或采集结果不满足冻结约束。"""


class UltrasonicLockError(SensorDeviceError):
    """互相关质量不足，且没有理论 ToF 回退。"""


def _finite(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise SensorDeviceError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise SensorDeviceError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise SensorDeviceError(f"{name} must be non-negative")
    return result


@dataclass(frozen=True)
class UltrasonicAcquisitionProfile:
    ultrasonic_profile_id: str
    excitation_type: str
    path_length_m: float
    center_frequency_hz: float
    fractional_bandwidth: float
    adc_rate_hz: float
    pulse_repetition_hz: float
    average_count: int
    tof_estimator: str
    reference_waveform_id: str
    window_duration_s: float
    attenuation_nepers_per_m_by_component: tuple[float, float, float]
    multipath_profile_id: str = "US-MP-NOMINAL"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "UltrasonicAcquisitionProfile":
        required = {
            "ultrasonic_profile_id",
            "excitation_type",
            "path_length_m",
            "center_frequency_hz",
            "fractional_bandwidth",
            "adc_rate_hz",
            "pulse_repetition_hz",
            "average_count",
            "tof_estimator",
            "reference_waveform_id",
            "window_duration_s",
            "attenuation_nepers_per_m_by_component",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise SensorDeviceError(f"ultrasonic profile is missing keys: {missing}")
        raw_average_count = raw["average_count"]
        if (
            isinstance(raw_average_count, bool)
            or not isinstance(raw_average_count, (int, float))
            or not math.isfinite(float(raw_average_count))
            or int(raw_average_count) != raw_average_count
            or int(raw_average_count) < 1
        ):
            raise SensorDeviceError("average_count must be a positive integer")
        attenuation_raw = raw["attenuation_nepers_per_m_by_component"]
        if not isinstance(attenuation_raw, Sequence) or isinstance(attenuation_raw, (str, bytes)) or len(attenuation_raw) != 3:
            raise SensorDeviceError("attenuation_nepers_per_m_by_component must contain Ar/He/CO2 values")
        attenuation = tuple(
            _finite(value, f"attenuation_nepers_per_m_by_component[{index}]", nonnegative=True)
            for index, value in enumerate(attenuation_raw)
        )
        profile = cls(
            ultrasonic_profile_id=str(raw["ultrasonic_profile_id"]),
            excitation_type=str(raw["excitation_type"]),
            path_length_m=_finite(raw["path_length_m"], "path_length_m", positive=True),
            center_frequency_hz=_finite(raw["center_frequency_hz"], "center_frequency_hz", positive=True),
            fractional_bandwidth=_finite(raw["fractional_bandwidth"], "fractional_bandwidth", positive=True),
            adc_rate_hz=_finite(raw["adc_rate_hz"], "adc_rate_hz", positive=True),
            pulse_repetition_hz=_finite(raw["pulse_repetition_hz"], "pulse_repetition_hz", positive=True),
            average_count=int(raw_average_count),
            tof_estimator=str(raw["tof_estimator"]),
            reference_waveform_id=str(raw["reference_waveform_id"]),
            window_duration_s=_finite(raw["window_duration_s"], "window_duration_s", positive=True),
            attenuation_nepers_per_m_by_component=attenuation,
            multipath_profile_id=str(raw.get("multipath_profile_id", "US-MP-NOMINAL")),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.ultrasonic_profile_id:
            raise SensorDeviceError("ultrasonic_profile_id must be non-empty")
        if self.excitation_type not in {"bandlimited_burst", "linear_chirp"}:
            raise SensorDeviceError(f"unsupported ultrasonic excitation_type {self.excitation_type!r}")
        if self.fractional_bandwidth >= 1.0:
            raise SensorDeviceError("fractional_bandwidth must be smaller than 1")
        if isinstance(self.average_count, bool) or self.average_count < 1:
            raise SensorDeviceError("average_count must be a positive integer")
        if self.tof_estimator not in {"reference_xcorr", "reference_xcorr_parabolic"}:
            raise SensorDeviceError(f"unsupported tof_estimator {self.tof_estimator!r}")
        if not self.reference_waveform_id or not self.multipath_profile_id:
            raise SensorDeviceError("waveform and multipath profile IDs must be non-empty")
        if self.center_frequency_hz >= self.adc_rate_hz / 2.0:
            raise SensorDeviceError("center_frequency_hz must be below the Nyquist frequency")


@dataclass(frozen=True)
class UltrasonicAcquisitionResult:
    tof_s: float | None
    peak_correlation: float
    snr: float
    estimated_tof_uncertainty_s: float
    lock_status: bool
    sample_lag: float | None
    waveform_samples: np.ndarray | None = None

    @property
    def quality(self) -> dict[str, float | bool]:
        return {
            "peak_correlation": self.peak_correlation,
            "snr": self.snr,
            "estimated_tof_uncertainty": self.estimated_tof_uncertainty_s,
            "lock_status": self.lock_status,
        }


def build_reference_waveform(
    profile: UltrasonicAcquisitionProfile | Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """按 acquisition profile 构造内存中的固定参考波包。"""

    acquisition = _coerce_ultrasonic_profile(profile)
    template_duration = min(
        acquisition.window_duration_s * 0.25,
        max(16.0 / acquisition.center_frequency_hz, 120.0e-6),
    )
    template_count = max(32, int(round(template_duration * acquisition.adc_rate_hz)))
    template_time = np.arange(template_count, dtype=np.float64) / acquisition.adc_rate_hz
    normalized_time = template_time / template_time[-1]
    envelope = np.sin(np.pi * normalized_time) ** 2
    if acquisition.excitation_type == "bandlimited_burst":
        carrier = np.sin(2.0 * np.pi * acquisition.center_frequency_hz * template_time)
    else:
        sweep = acquisition.center_frequency_hz * acquisition.fractional_bandwidth
        start_frequency = acquisition.center_frequency_hz - sweep / 2.0
        chirp_rate = sweep / template_duration
        phase = 2.0 * np.pi * (start_frequency * template_time + 0.5 * chirp_rate * template_time**2)
        carrier = np.sin(phase)
    waveform = envelope * carrier
    waveform = waveform - float(waveform.mean())
    norm = float(np.linalg.norm(waveform))
    if not math.isfinite(norm) or norm <= 0.0:
        raise SensorDeviceError("reference waveform has zero energy")
    waveform = waveform / norm
    return template_time, waveform


def acquire_ultrasonic_tof(
    composition_pct: Sequence[float],
    *,
    temperature_k: float,
    pressure_pa: float,
    profile: UltrasonicAcquisitionProfile | Mapping[str, Any],
    multipath_profile: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    internal_noise_std: float = 0.0,
    signal_amplitude: float = 1.0,
    rng: np.random.Generator | None = None,
    strict: bool = True,
    retain_waveform: bool = False,
    system_delay_s: float = SYSTEM_DELAY_S,
    sound_speed_model_id: str = DIRECT_HEOS_SOUND_SPEED_MODEL_ID,
) -> UltrasonicAcquisitionResult:
    """生成短波形、重复平均并互相关估计 ToF。

    ``strict=True`` 时失锁直接抛出 ``UltrasonicLockError``；``strict=False``
    只返回 ``tof_s=None`` 和 ``lock_status=False``，绝不返回理论 ``L/c``。
    """

    acquisition = _coerce_ultrasonic_profile(profile)
    composition = validate_composition_pct(composition_pct)
    temperature = _finite(temperature_k, "temperature_k", positive=True)
    pressure = _finite(pressure_pa, "pressure_pa", positive=True)
    noise_std = _finite(internal_noise_std, "internal_noise_std", nonnegative=True)
    amplitude = _finite(signal_amplitude, "signal_amplitude", positive=True)
    delay = _finite(system_delay_s, "system_delay_s", nonnegative=True)
    if noise_std > 0.0 and not isinstance(rng, np.random.Generator):
        raise SensorDeviceError("rng must be explicit when internal waveform noise is enabled")

    fractions = composition_pct_to_mole_fractions(composition)
    sound_speed = a2dyn_sound_speed_for_model(
        fractions,
        temperature,
        pressure,
        model_id=sound_speed_model_id,
    )
    true_tof = acquisition.path_length_m / sound_speed + delay
    if true_tof < 0.0 or true_tof >= acquisition.window_duration_s:
        raise SensorDeviceError(
            f"true ToF {true_tof} s does not fit acquisition window {acquisition.window_duration_s} s"
        )
    template_time, reference = build_reference_waveform(acquisition)
    sample_count = int(round(acquisition.window_duration_s * acquisition.adc_rate_hz))
    if sample_count <= len(reference) or sample_count < 4:
        raise SensorDeviceError("acquisition window must contain more samples than the reference waveform")
    receive_time = np.arange(sample_count, dtype=np.float64) / acquisition.adc_rate_hz
    components = _multipath_components(acquisition, multipath_profile)
    receive = np.zeros(sample_count, dtype=np.float64)
    for _ in range(acquisition.average_count):
        repetition = amplitude * np.interp(
            receive_time - true_tof,
            template_time,
            reference,
            left=0.0,
            right=0.0,
        )
        for component_amplitude, component_delay in components:
            repetition = repetition + amplitude * component_amplitude * np.interp(
                receive_time - true_tof - component_delay,
                template_time,
                reference,
                left=0.0,
                right=0.0,
            )
        if noise_std > 0.0:
            repetition = repetition + rng.normal(0.0, noise_std, size=sample_count)
        receive += repetition
    receive /= float(acquisition.average_count)
    filtered_reference = _frequency_bandpass(
        reference,
        acquisition.adc_rate_hz,
        acquisition.center_frequency_hz,
        acquisition.fractional_bandwidth,
    )
    filtered_receive = _frequency_bandpass(
        receive,
        acquisition.adc_rate_hz,
        acquisition.center_frequency_hz,
        acquisition.fractional_bandwidth,
    )
    denominator = float(np.linalg.norm(filtered_receive) * np.linalg.norm(filtered_reference))
    if not math.isfinite(denominator) or denominator <= 0.0:
        return _finish_ultrasonic_result(
            acquisition,
            peak_correlation=0.0,
            snr=0.0,
            lag=None,
            waveform=receive if retain_waveform else None,
            strict=strict,
        )
    correlation = correlate(filtered_receive, filtered_reference, mode="full", method="fft")
    correlation = np.asarray(correlation, dtype=np.float64) / denominator
    lags = np.arange(-len(filtered_reference) + 1, len(filtered_receive), dtype=np.float64)
    valid_indices = np.flatnonzero((lags >= 0.0) & (lags <= len(filtered_receive) - len(filtered_reference)))
    if valid_indices.size == 0:
        return _finish_ultrasonic_result(
            acquisition,
            peak_correlation=0.0,
            snr=0.0,
            lag=None,
            waveform=receive if retain_waveform else None,
            strict=strict,
        )
    peak_index = int(valid_indices[np.argmax(correlation[valid_indices])])
    peak = float(correlation[peak_index])
    exclusion = np.abs(lags - lags[peak_index]) <= max(len(filtered_reference), 3)
    off_peak = correlation[(lags >= 0.0) & ~exclusion]
    noise_floor = float(np.std(off_peak)) if off_peak.size else 0.0
    snr = abs(peak) / max(noise_floor, 1.0e-12)
    fractional_lag = 0.0
    if acquisition.tof_estimator == "reference_xcorr_parabolic" and 0 < peak_index < correlation.size - 1:
        left, center, right = correlation[peak_index - 1 : peak_index + 2]
        curvature = left - 2.0 * center + right
        if curvature != 0.0 and math.isfinite(float(curvature)):
            fractional_lag = 0.5 * float(left - right) / float(curvature)
            if fractional_lag < -0.5 or fractional_lag > 0.5:
                raise SensorDeviceError("parabolic refinement returned an invalid sub-sample lag")
    lag = float(lags[peak_index] + fractional_lag)
    return _finish_ultrasonic_result(
        acquisition,
        peak_correlation=peak,
        snr=snr,
        lag=lag,
        waveform=receive if retain_waveform else None,
        strict=strict,
    )


def ultrasonic_signal_amplitude(
    composition_pct: Sequence[float],
    profile: UltrasonicAcquisitionProfile | Mapping[str, Any],
) -> float:
    """按注册的组成相关衰减敏感性 profile 计算接收幅度。"""

    acquisition = _coerce_ultrasonic_profile(profile)
    fractions = validate_composition_pct(composition_pct) / 100.0
    attenuation = float(
        np.dot(fractions, np.asarray(acquisition.attenuation_nepers_per_m_by_component))
    )
    amplitude = math.exp(-attenuation * acquisition.path_length_m)
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        raise SensorDeviceError("ultrasonic attenuation produced an invalid amplitude")
    return amplitude


def estimate_ultrasonic_tof_series(
    theoretical_tof_s: Sequence[float] | np.ndarray,
    profile: UltrasonicAcquisitionProfile | Mapping[str, Any],
) -> np.ndarray:
    """应用经短波形探针验证的低频 ToF 估计器离散 surrogate。

    纯互相关保持一个 ADC sample 的分辨率；三点抛物线精化保持四分之一
    sample 的注册分辨率。该函数只用于批量低频序列，短波形候选仍由
    :func:`acquire_ultrasonic_tof` 独立验证。
    """

    acquisition = _coerce_ultrasonic_profile(profile)
    values = np.asarray(theoretical_tof_s, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise SensorDeviceError("theoretical_tof_s must be a non-empty finite non-negative vector")
    samples = values * acquisition.adc_rate_hz
    if acquisition.tof_estimator == "reference_xcorr":
        estimated_samples = np.round(samples)
    else:
        estimated_samples = np.round(samples * 4.0) / 4.0
    return estimated_samples / acquisition.adc_rate_hz


def estimate_ultrasonic_quality_series(
    compositions_pct: Sequence[Sequence[float]] | np.ndarray,
    *,
    profile: UltrasonicAcquisitionProfile | Mapping[str, Any],
    internal_noise_std: float,
    multipath_profile: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, np.ndarray]:
    """计算批量低频超声质量 surrogate。

    正式数据不逐时刻持久化 MHz 级波形，因此质量审计使用与
    :func:`acquire_ultrasonic_tof` 相同的注册 profile 做匹配滤波近似：
    组成相关衰减决定主波幅，重复平均决定等效噪声，多径决定有效信号能量。
    该函数不生成或伪造理论 ``L/c``，并按同一相关系数与 SNR 门计算锁定状态。
    """

    acquisition = _coerce_ultrasonic_profile(profile)
    compositions = validate_composition_sequence_pct(compositions_pct)
    noise_std = _finite(internal_noise_std, "internal_noise_std", nonnegative=True)
    if multipath_profile is None:
        multipath_components: tuple[tuple[float, float], ...] = ()
    else:
        multipath_components = _multipath_components(acquisition, multipath_profile)
    fractions = compositions / 100.0
    attenuation = np.asarray(acquisition.attenuation_nepers_per_m_by_component, dtype=np.float64)
    amplitude = np.exp(-
        np.sum(fractions * attenuation[None, :], axis=1) * acquisition.path_length_m
    )
    average_noise_std = noise_std / math.sqrt(float(acquisition.average_count))
    sample_count = int(round(acquisition.window_duration_s * acquisition.adc_rate_hz))
    multipath_energy = 1.0 + sum(component_amplitude**2 for component_amplitude, _ in multipath_components)
    noise_energy = average_noise_std * math.sqrt(float(sample_count))
    signal_energy = amplitude * math.sqrt(multipath_energy)
    peak_correlation = amplitude / np.sqrt(amplitude**2 + noise_energy**2 * multipath_energy)
    snr = amplitude / max(average_noise_std * math.sqrt(multipath_energy), 1.0e-12)
    if noise_std == 0.0:
        peak_correlation = np.ones_like(amplitude)
        snr = np.full_like(amplitude, np.finfo(np.float64).max)
    uncertainty = 1.0 / (
        acquisition.adc_rate_hz * np.maximum(snr, 1.0)
    )
    lock_status = (
        np.isfinite(peak_correlation)
        & np.isfinite(snr)
        & (peak_correlation >= ULTRASONIC_LOCK_CORRELATION_THRESHOLD)
        & (snr >= ULTRASONIC_LOCK_SNR_THRESHOLD)
    )
    if not np.isfinite(signal_energy).all() or not np.isfinite(peak_correlation).all():
        raise SensorDeviceError("ultrasonic quality surrogate produced non-finite values")
    return {
        "peak_correlation": np.asarray(peak_correlation, dtype=np.float64),
        "snr": np.asarray(snr, dtype=np.float64),
        "estimated_tof_uncertainty_s": np.asarray(uncertainty, dtype=np.float64),
        "lock_status": np.asarray(lock_status, dtype=np.bool_),
    }


def estimate_ndir_equilibrium_co2_series(
    clean_voltage_v: Sequence[float] | np.ndarray,
    *,
    temperature_k: float,
    pressure_pa: float,
    dt_s: float,
    profile: NDIRDeviceProfile | Mapping[str, Any],
    absorbance_scale: float = 1.0,
    domain_tolerance: float | None = None,
) -> np.ndarray:
    """从 clean NDIR 电压反演每个时刻的局部平衡 CO₂ 组成。

    ``simulate_ndir`` 的动态状态是 active/reference 比值的一阶更新，
    因而可用已观测 prefix 反解其平衡比值，再在注册 HITRAN 查找表上插值。
    大于输入误差预算的越界、非单调或 profile 不一致都会显式报错；
    只有在该明确预算内才把比值投影回注册端点，避免把通用 clip 当作兜底。

    ``domain_tolerance`` 是注册反演域（比值单位）的准入预算，默认 None 时
    使用 float32 持久化预算（16 eps），适用于 clean 输入；观测输入的反演
    （含标定、漂移与噪声）必须显式传入按该行注册扰动包络推导的预算。
    """

    ndir = _coerce_ndir_profile(profile)
    clean = np.asarray(clean_voltage_v, dtype=np.float64)
    if clean.ndim != 1 or clean.size == 0 or not np.isfinite(clean).all():
        raise SensorDeviceError("clean_voltage_v must be a non-empty finite vector")
    temperature = _finite(temperature_k, "temperature_k", positive=True)
    pressure = _finite(pressure_pa, "pressure_pa", positive=True)
    dt = _finite(dt_s, "dt_s", positive=True)
    absorption_scale = _finite(absorbance_scale, "absorbance_scale", positive=True)
    if domain_tolerance is not None:
        tolerance = _finite(domain_tolerance, "domain_tolerance", positive=True)
    ratio = clean / NDIR_BASELINE_V
    if ndir.tau_emitter_detector_s == 0.0:
        equilibrium_ratio = ratio.copy()
    else:
        decay = math.exp(-dt / ndir.tau_emitter_detector_s)
        equilibrium_ratio = np.empty_like(ratio)
        equilibrium_ratio[0] = ratio[0]
        equilibrium_ratio[1:] = (
            ratio[1:] - decay * ratio[:-1]
        ) / (1.0 - decay)

    co2_grid, active, reference = _hitran_band_transmission_grid(
        temperature,
        pressure,
        ndir.hitran_table_name,
        ndir.optical_path_m,
        ndir.active_center_wavenumber_cm1,
        ndir.active_fwhm_cm1,
        ndir.reference_center_wavenumber_cm1,
        ndir.reference_fwhm_cm1,
        ndir.wavenumber_range_cm1,
        ndir.wavenumber_step_cm1,
        absorption_scale,
    )
    ratio_grid = active / reference
    if np.any(np.diff(ratio_grid) >= 0.0):
        raise SensorDeviceError("registered NDIR ratio curve must be strictly decreasing in CO2")
    lower = float(ratio_grid[-1])
    upper = float(ratio_grid[0])
    if domain_tolerance is None:
        tolerance = max(abs(lower), abs(upper), 1.0) * (16.0 * np.finfo(np.float32).eps)
    if np.any(equilibrium_ratio < lower - tolerance) or np.any(equilibrium_ratio > upper + tolerance):
        raise SensorDeviceError("clean NDIR prefix contains a ratio outside the registered inversion range")
    equilibrium_ratio = np.clip(equilibrium_ratio, lower, upper)
    return np.interp(equilibrium_ratio, ratio_grid[::-1], co2_grid[::-1]).astype(np.float64)


def _finish_ultrasonic_result(
    profile: UltrasonicAcquisitionProfile,
    *,
    peak_correlation: float,
    snr: float,
    lag: float | None,
    waveform: np.ndarray | None,
    strict: bool,
) -> UltrasonicAcquisitionResult:
    locked = (
        lag is not None
        and math.isfinite(peak_correlation)
        and math.isfinite(snr)
        and peak_correlation >= ULTRASONIC_LOCK_CORRELATION_THRESHOLD
        and snr >= ULTRASONIC_LOCK_SNR_THRESHOLD
        and lag >= 0.0
    )
    tof = None if lag is None or not locked else lag / profile.adc_rate_hz
    uncertainty = 1.0 / (profile.adc_rate_hz * max(snr, 1.0)) if math.isfinite(snr) else math.inf
    result = UltrasonicAcquisitionResult(
        tof_s=tof,
        peak_correlation=float(peak_correlation),
        snr=float(snr),
        estimated_tof_uncertainty_s=float(uncertainty),
        lock_status=bool(locked),
        sample_lag=None if lag is None else float(lag),
        waveform_samples=waveform,
    )
    if strict and not result.lock_status:
        raise UltrasonicLockError(
            "ultrasonic acquisition lost lock: "
            f"peak_correlation={result.peak_correlation:.6g}, snr={result.snr:.6g}, "
            "no theoretical ToF fallback is permitted"
        )
    return result


def _frequency_bandpass(
    signal: np.ndarray,
    sample_rate_hz: float,
    center_frequency_hz: float,
    fractional_bandwidth: float,
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate_hz)
    half_bandwidth = center_frequency_hz * fractional_bandwidth / 2.0
    lower = max(0.0, center_frequency_hz - half_bandwidth)
    upper = min(sample_rate_hz / 2.0, center_frequency_hz + half_bandwidth)
    mask = (frequencies >= lower) & (frequencies <= upper)
    spectrum = np.fft.rfft(signal)
    return np.fft.irfft(spectrum * mask, n=signal.size)


def _multipath_components(
    profile: UltrasonicAcquisitionProfile,
    multipath_profile: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[tuple[float, float], ...]:
    raw_components: Any
    if multipath_profile is None:
        raw_components = []
    elif isinstance(multipath_profile, Mapping):
        raw_components = multipath_profile.get("components")
        if raw_components is None:
            raise SensorDeviceError("multipath_profile mapping must contain components")
    else:
        raw_components = multipath_profile
    if not isinstance(raw_components, Sequence) or isinstance(raw_components, (str, bytes)):
        raise SensorDeviceError("multipath components must be a sequence")
    result: list[tuple[float, float]] = []
    for index, raw in enumerate(raw_components):
        if not isinstance(raw, Mapping):
            raise SensorDeviceError(f"multipath components[{index}] must be an object")
        amplitude = _finite(raw.get("relative_amplitude"), f"multipath[{index}].relative_amplitude")
        delay = _finite(raw.get("delay_s"), f"multipath[{index}].delay_s", nonnegative=True)
        result.append((amplitude, delay))
    return tuple(result)


def _coerce_ultrasonic_profile(
    profile: UltrasonicAcquisitionProfile | Mapping[str, Any],
) -> UltrasonicAcquisitionProfile:
    if isinstance(profile, UltrasonicAcquisitionProfile):
        profile.validate()
        return profile
    if isinstance(profile, Mapping):
        return UltrasonicAcquisitionProfile.from_mapping(profile)
    raise SensorDeviceError("profile must be an UltrasonicAcquisitionProfile or mapping")


@dataclass(frozen=True)
class ThermalDeviceProfile:
    thermal_profile_id: str
    heater_heat_capacity_j_per_k: float
    gas_conductance_scale: float
    substrate_conductance_w_per_k: float
    heater_power_w: float
    tcr_per_k: float
    bridge_voltage_v: float
    flow_coupling: float
    heater_resistance_ohm: float
    reference_temperature_k: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ThermalDeviceProfile":
        required = {
            "thermal_profile_id",
            "heater_heat_capacity",
            "gas_conductance_scale",
            "substrate_conductance",
            "heater_power",
            "tcr",
            "bridge_voltage",
            "flow_coupling",
            "heater_resistance_ohm",
            "reference_temperature_k",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise SensorDeviceError(f"thermal profile is missing keys: {missing}")
        result = cls(
            thermal_profile_id=str(raw["thermal_profile_id"]),
            heater_heat_capacity_j_per_k=_finite(raw["heater_heat_capacity"], "heater_heat_capacity", positive=True),
            gas_conductance_scale=_finite(raw["gas_conductance_scale"], "gas_conductance_scale", positive=True),
            substrate_conductance_w_per_k=_finite(raw["substrate_conductance"], "substrate_conductance", positive=True),
            heater_power_w=_finite(raw["heater_power"], "heater_power", positive=True),
            tcr_per_k=_finite(raw["tcr"], "tcr", nonnegative=True),
            bridge_voltage_v=_finite(raw["bridge_voltage"], "bridge_voltage", positive=True),
            flow_coupling=_finite(raw["flow_coupling"], "flow_coupling", nonnegative=True),
            heater_resistance_ohm=_finite(raw["heater_resistance_ohm"], "heater_resistance_ohm", positive=True),
            reference_temperature_k=_finite(raw["reference_temperature_k"], "reference_temperature_k", positive=True),
        )
        if not result.thermal_profile_id:
            raise SensorDeviceError("thermal_profile_id must be non-empty")
        return result


@dataclass(frozen=True)
class TCDSimulationResult:
    heater_temperature_k: np.ndarray
    heater_resistance_ohm: np.ndarray
    bridge_output_v: np.ndarray
    clean_voltage_v: np.ndarray
    gas_conductance_w_per_k: np.ndarray
    energy_balance_residual_w: np.ndarray


def simulate_tcd(
    compositions_pct: Sequence[Sequence[float]] | np.ndarray,
    *,
    temperature_k: float | Sequence[float] | np.ndarray,
    dt_s: float,
    profile: ThermalDeviceProfile | Mapping[str, Any],
    substrate_temperature_k: float | Sequence[float] | np.ndarray | None = None,
    flow_fraction: float | Sequence[float] | np.ndarray = 0.0,
    initial_heater_temperature_k: float | None = None,
    response_scale: float = 1.0,
) -> TCDSimulationResult:
    """执行 TCD 集总能量平衡并输出一次固定校准映射后的 clean 电压。"""

    thermal = _coerce_thermal_profile(profile)
    compositions = validate_composition_sequence_pct(compositions_pct)
    dt = _finite(dt_s, "dt_s", positive=True)
    count = compositions.shape[0]
    gas_temperature = _broadcast(temperature_k, count, "temperature_k", positive=True)
    substrate = gas_temperature if substrate_temperature_k is None else _broadcast(
        substrate_temperature_k,
        count,
        "substrate_temperature_k",
        positive=True,
    )
    flow = _broadcast(flow_fraction, count, "flow_fraction", nonnegative=True)
    response = _finite(response_scale, "response_scale", positive=True)
    conductance = np.empty(count, dtype=np.float64)
    equilibrium_temperature = np.empty(count, dtype=np.float64)
    equilibrium_voltage = np.empty(count, dtype=np.float64)
    raw_equilibrium_bridge = np.empty(count, dtype=np.float64)
    for index, composition in enumerate(compositions):
        fractions = composition_pct_to_mole_fractions(composition)
        conductivity = wms_thermal_conductivity(fractions)
        conductance[index] = (
            TCD_REFERENCE_GAS_CONDUCTANCE_W_PER_K
            * thermal.gas_conductance_scale
            * conductivity
            / TCS_REFERENCE_CONDUCTIVITY_W_M_K
            * (1.0 + thermal.flow_coupling * flow[index])
        )
        total_conductance = conductance[index] + thermal.substrate_conductance_w_per_k
        equilibrium_temperature[index] = (
            thermal.heater_power_w
            + conductance[index] * gas_temperature[index]
            + thermal.substrate_conductance_w_per_k * substrate[index]
        ) / total_conductance
        equilibrium_voltage[index] = thermal_conductivity_voltage(
            conductivity,
            response_v_per_w_m_k=TCS_RESPONSE_V_PER_W_M_K * response,
        )
        raw_equilibrium_bridge[index] = _bridge_output(
            _heater_resistance(equilibrium_temperature[index], thermal),
            thermal,
        )
    if initial_heater_temperature_k is None:
        initial_temperature = float(equilibrium_temperature[0])
    else:
        initial_temperature = _finite(initial_heater_temperature_k, "initial_heater_temperature_k", positive=True)
    heater_temperature = np.empty(count, dtype=np.float64)
    heater_temperature[0] = initial_temperature
    residual = np.zeros(count, dtype=np.float64)
    for index in range(1, count):
        total_conductance = conductance[index] + thermal.substrate_conductance_w_per_k
        tau = thermal.heater_heat_capacity_j_per_k / total_conductance
        decay = math.exp(-dt / tau)
        previous = heater_temperature[index - 1]
        current_equilibrium = equilibrium_temperature[index]
        next_temperature = current_equilibrium + (previous - current_equilibrium) * decay
        average_temperature = current_equilibrium + (previous - current_equilibrium) * (
            (1.0 - decay) / (dt / tau)
        )
        flux = (
            thermal.heater_power_w
            - conductance[index] * (average_temperature - gas_temperature[index])
            - thermal.substrate_conductance_w_per_k * (average_temperature - substrate[index])
        )
        residual[index] = (
            thermal.heater_heat_capacity_j_per_k * (next_temperature - previous) / dt - flux
        )
        heater_temperature[index] = next_temperature
    heater_resistance = _heater_resistance(heater_temperature, thermal)
    bridge_output = _bridge_output(heater_resistance, thermal)
    clean_voltage = equilibrium_voltage + TCD_BRIDGE_CALIBRATION_GAIN * (
        bridge_output - raw_equilibrium_bridge
    )
    if not np.isfinite(clean_voltage).all() or not np.isfinite(residual).all():
        raise SensorDeviceError("TCD simulation produced non-finite values")
    return TCDSimulationResult(
        heater_temperature_k=heater_temperature,
        heater_resistance_ohm=heater_resistance,
        bridge_output_v=bridge_output,
        clean_voltage_v=clean_voltage,
        gas_conductance_w_per_k=conductance,
        energy_balance_residual_w=residual,
    )


def _heater_resistance(
    temperature_k: float | np.ndarray,
    profile: ThermalDeviceProfile,
) -> float | np.ndarray:
    resistance = profile.heater_resistance_ohm * (
        1.0 + profile.tcr_per_k * (np.asarray(temperature_k, dtype=np.float64) - profile.reference_temperature_k)
    )
    if not np.isfinite(resistance).all() or np.any(resistance <= 0.0):
        raise SensorDeviceError("TCD heater resistance became non-positive or non-finite")
    if np.asarray(temperature_k).ndim == 0:
        return float(resistance)
    return resistance


def _bridge_output(resistance: float | np.ndarray, profile: ThermalDeviceProfile) -> float | np.ndarray:
    reference = profile.heater_resistance_ohm
    output = profile.bridge_voltage_v * (np.asarray(resistance) / (np.asarray(resistance) + reference) - 0.5)
    if np.asarray(output).ndim == 0:
        return float(output)
    return output.astype(np.float64)


def _coerce_thermal_profile(profile: ThermalDeviceProfile | Mapping[str, Any]) -> ThermalDeviceProfile:
    if isinstance(profile, ThermalDeviceProfile):
        return profile
    if isinstance(profile, Mapping):
        return ThermalDeviceProfile.from_mapping(profile)
    raise SensorDeviceError("profile must be a ThermalDeviceProfile or mapping")


@dataclass(frozen=True)
class NDIRDeviceProfile:
    ndir_profile_id: str
    optical_path_m: float
    active_band_id: str
    reference_band_id: str
    source_spectrum_id: str
    detector_response_id: str
    effective_absorption_model_id: str
    tau_emitter_detector_s: float
    range_min_mol_pct: float
    range_max_mol_pct: float
    reference_asset_id: str
    hitran_table_name: str
    active_center_wavenumber_cm1: float
    active_fwhm_cm1: float
    reference_center_wavenumber_cm1: float
    reference_fwhm_cm1: float
    wavenumber_range_cm1: tuple[float, float]
    wavenumber_step_cm1: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NDIRDeviceProfile":
        required = {
            "ndir_profile_id",
            "optical_path_m",
            "active_band_id",
            "reference_band_id",
            "source_spectrum_id",
            "detector_response_id",
            "effective_absorption_model_id",
            "tau_emitter_detector_s",
            "range_min_mol_pct",
            "range_max_mol_pct",
            "reference_asset_id",
            "hitran_table_name",
            "active_center_wavenumber_cm1",
            "active_fwhm_cm1",
            "reference_center_wavenumber_cm1",
            "reference_fwhm_cm1",
            "wavenumber_range_cm1",
            "wavenumber_step_cm1",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise SensorDeviceError(f"NDIR profile is missing keys: {missing}")
        range_raw = raw["wavenumber_range_cm1"]
        if not isinstance(range_raw, Sequence) or isinstance(range_raw, (str, bytes)) or len(range_raw) != 2:
            raise SensorDeviceError("wavenumber_range_cm1 must contain [lower, upper]")
        wavenumber_range = (
            _finite(range_raw[0], "wavenumber_range_cm1[0]", positive=True),
            _finite(range_raw[1], "wavenumber_range_cm1[1]", positive=True),
        )
        result = cls(
            ndir_profile_id=str(raw["ndir_profile_id"]),
            optical_path_m=_finite(raw["optical_path_m"], "optical_path_m", positive=True),
            active_band_id=str(raw["active_band_id"]),
            reference_band_id=str(raw["reference_band_id"]),
            source_spectrum_id=str(raw["source_spectrum_id"]),
            detector_response_id=str(raw["detector_response_id"]),
            effective_absorption_model_id=str(raw["effective_absorption_model_id"]),
            tau_emitter_detector_s=_finite(raw["tau_emitter_detector_s"], "tau_emitter_detector_s", nonnegative=True),
            range_min_mol_pct=_finite(raw["range_min_mol_pct"], "range_min_mol_pct", nonnegative=True),
            range_max_mol_pct=_finite(raw["range_max_mol_pct"], "range_max_mol_pct", positive=True),
            reference_asset_id=str(raw["reference_asset_id"]),
            hitran_table_name=str(raw["hitran_table_name"]),
            active_center_wavenumber_cm1=_finite(raw["active_center_wavenumber_cm1"], "active_center_wavenumber_cm1", positive=True),
            active_fwhm_cm1=_finite(raw["active_fwhm_cm1"], "active_fwhm_cm1", positive=True),
            reference_center_wavenumber_cm1=_finite(raw["reference_center_wavenumber_cm1"], "reference_center_wavenumber_cm1", positive=True),
            reference_fwhm_cm1=_finite(raw["reference_fwhm_cm1"], "reference_fwhm_cm1", positive=True),
            wavenumber_range_cm1=wavenumber_range,
            wavenumber_step_cm1=_finite(raw["wavenumber_step_cm1"], "wavenumber_step_cm1", positive=True),
        )
        result.validate()
        return result

    def validate(self) -> None:
        identifiers = (
            self.ndir_profile_id,
            self.active_band_id,
            self.reference_band_id,
            self.source_spectrum_id,
            self.detector_response_id,
            self.effective_absorption_model_id,
            self.reference_asset_id,
            self.hitran_table_name,
        )
        if any(not value for value in identifiers):
            raise SensorDeviceError("NDIR profile identifiers must be non-empty")
        if self.range_min_mol_pct >= self.range_max_mol_pct:
            raise SensorDeviceError("NDIR range_min_mol_pct must be smaller than range_max_mol_pct")
        if self.range_min_mol_pct < 0.0 or self.range_max_mol_pct > 100.0:
            raise SensorDeviceError("NDIR registered range must lie within [0,100] mol%")
        lower, upper = self.wavenumber_range_cm1
        if lower >= upper:
            raise SensorDeviceError("NDIR wavenumber range must be increasing")
        for name, center in (
            ("active", self.active_center_wavenumber_cm1),
            ("reference", self.reference_center_wavenumber_cm1),
        ):
            if not lower <= center <= upper:
                raise SensorDeviceError(f"NDIR {name} band center must lie inside the registered range")
        if self.effective_absorption_model_id != "HITRAN2020-BANDINTEGRATED-GAUSSIAN-1":
            raise SensorDeviceError("unsupported NDIR absorption model")


@dataclass(frozen=True)
class NDIRSimulationResult:
    active_voltage_v: np.ndarray
    reference_voltage_v: np.ndarray
    active_reference_ratio: np.ndarray
    clean_voltage_v: np.ndarray
    saturation_mask: np.ndarray
    saturation_fraction: float


def simulate_ndir(
    compositions_pct: Sequence[Sequence[float]] | np.ndarray,
    *,
    temperature_k: float | Sequence[float] | np.ndarray,
    pressure_pa: float | Sequence[float] | np.ndarray,
    dt_s: float,
    profile: NDIRDeviceProfile | Mapping[str, Any],
    initial_ratio: float | None = None,
    absorbance_scale: float = 1.0,
) -> NDIRSimulationResult:
    """执行 active/reference 光学链和光源 / 探测器一阶响应。"""

    ndir = _coerce_ndir_profile(profile)
    compositions = validate_composition_sequence_pct(compositions_pct)
    dt = _finite(dt_s, "dt_s", positive=True)
    count = compositions.shape[0]
    temperatures = _broadcast(temperature_k, count, "temperature_k", positive=True)
    pressures = _broadcast(pressure_pa, count, "pressure_pa", positive=True)
    absorption_scale = _finite(absorbance_scale, "absorbance_scale", positive=True)
    co2 = compositions[:, 2]
    if np.any(co2 < ndir.range_min_mol_pct) or np.any(co2 > ndir.range_max_mol_pct):
        raise SensorDeviceError("CO2 composition is outside the registered NDIR range")
    equilibrium_active = np.empty(count, dtype=np.float64)
    equilibrium_reference = np.empty(count, dtype=np.float64)
    for temperature, pressure in sorted(set(zip(temperatures.tolist(), pressures.tolist()))):
        indices = np.flatnonzero((temperatures == temperature) & (pressures == pressure))
        active, reference = _hitran_band_transmission(
            co2[indices],
            temperature_k=float(temperature),
            pressure_pa=float(pressure),
            profile=ndir,
            absorbance_scale=absorption_scale,
        )
        equilibrium_active[indices] = active
        equilibrium_reference[indices] = reference
    equilibrium_ratio = equilibrium_active / equilibrium_reference
    if initial_ratio is None:
        initial = float(equilibrium_ratio[0])
    else:
        initial = _finite(initial_ratio, "initial_ratio", positive=True)
    ratio = simulate_first_order_series(
        equilibrium_ratio,
        dt_s=dt,
        tau_s=ndir.tau_emitter_detector_s,
        initial_state=initial,
    )
    reference = equilibrium_reference * NDIR_REFERENCE_CHANNEL_V
    active = reference * ratio
    clean = NDIR_BASELINE_V * (active / reference)
    saturation = (
        (active < NDIR_ADC_MIN_V)
        | (active > NDIR_ADC_MAX_V)
        | (reference < NDIR_ADC_MIN_V)
        | (reference > NDIR_ADC_MAX_V)
    )
    if not np.isfinite(active).all() or not np.isfinite(clean).all():
        raise SensorDeviceError("NDIR simulation produced non-finite values")
    return NDIRSimulationResult(
        active_voltage_v=active,
        reference_voltage_v=reference,
        active_reference_ratio=ratio,
        clean_voltage_v=clean,
        saturation_mask=saturation.astype(np.bool_),
        saturation_fraction=float(np.mean(saturation)),
    )


def _hitran_band_transmission(
    co2_pct: np.ndarray,
    *,
    temperature_k: float,
    pressure_pa: float,
    profile: NDIRDeviceProfile,
    absorbance_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    co2 = np.asarray(co2_pct, dtype=np.float64)
    if not np.isfinite(co2).all() or np.any(co2 < 0.0) or np.any(co2 > 100.0):
        raise SensorDeviceError("CO2 values for HITRAN integration must lie in [0,100] mol%")
    co2_grid, active_grid, reference_grid = _hitran_band_transmission_grid(
        float(temperature_k),
        float(pressure_pa),
        profile.hitran_table_name,
        profile.optical_path_m,
        profile.active_center_wavenumber_cm1,
        profile.active_fwhm_cm1,
        profile.reference_center_wavenumber_cm1,
        profile.reference_fwhm_cm1,
        profile.wavenumber_range_cm1,
        profile.wavenumber_step_cm1,
        float(absorbance_scale),
    )
    return np.interp(co2, co2_grid, active_grid), np.interp(co2, co2_grid, reference_grid)


@lru_cache(maxsize=32)
def _hitran_band_transmission_grid(
    temperature_k: float,
    pressure_pa: float,
    table_name: str,
    optical_path_m: float,
    active_center_wavenumber_cm1: float,
    active_fwhm_cm1: float,
    reference_center_wavenumber_cm1: float,
    reference_fwhm_cm1: float,
    wavenumber_range_cm1: tuple[float, float],
    wavenumber_step_cm1: float,
    absorbance_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """缓存直接带积分查找表；运行时只在 CO2 轴做注册精度插值。"""

    wavenumber, coefficient = _hitran_absorption_cross_section(
        table_name,
        float(temperature_k),
        float(pressure_pa),
        wavenumber_range_cm1,
        wavenumber_step_cm1,
    )
    active_weight = _gaussian_band_weight(
        wavenumber,
        active_center_wavenumber_cm1,
        active_fwhm_cm1,
    )
    reference_weight = _gaussian_band_weight(
        wavenumber,
        reference_center_wavenumber_cm1,
        reference_fwhm_cm1,
    )
    number_density_per_cm3 = pressure_pa / (_BOLTZMANN_J_PER_K * temperature_k) / 1.0e6
    unit_optical_depth = (
        coefficient[:, None]
        * number_density_per_cm3
        * (optical_path_m * 100.0)
        * absorbance_scale
    )
    co2_grid = np.arange(
        0.0,
        100.0 + 0.5 * _HITRAN_CO2_GRID_STEP_PCT,
        _HITRAN_CO2_GRID_STEP_PCT,
        dtype=np.float64,
    )
    active = np.empty_like(co2_grid)
    reference = np.empty_like(co2_grid)
    for start in range(0, co2_grid.size, _HITRAN_GRID_CHUNK_SIZE):
        stop = min(start + _HITRAN_GRID_CHUNK_SIZE, co2_grid.size)
        transmission = np.exp(
            -unit_optical_depth * (co2_grid[None, start:stop] / 100.0)
        )
        active[start:stop] = np.trapezoid(
            active_weight[:, None] * transmission,
            wavenumber,
            axis=0,
        )
        reference[start:stop] = np.trapezoid(
            reference_weight[:, None] * transmission,
            wavenumber,
            axis=0,
        )
    if (
        not np.isfinite(active).all()
        or not np.isfinite(reference).all()
        or np.any(active <= 0.0)
        or np.any(reference <= 0.0)
    ):
        raise SensorDeviceError("HITRAN NDIR integration produced invalid transmission")
    for values in (co2_grid, active, reference):
        values.setflags(write=False)
    return co2_grid, active, reference


@lru_cache(maxsize=1)
def _initialize_hitran_database() -> None:
    global _HITRAN_RUNTIME_CACHE

    try:
        import hapi
    except ImportError as exc:
        raise SensorDeviceError("NDIR HITRAN integration requires hitran-api") from exc
    if not HITRAN_REFERENCE_CACHE_DIR.is_dir():
        raise SensorDeviceError(
            f"registered HITRAN cache does not exist: {HITRAN_REFERENCE_CACHE_DIR}"
        )
    runtime_cache = tempfile.TemporaryDirectory(prefix="a2dyn-hitran-")
    for suffix in ("data", "header"):
        source = HITRAN_REFERENCE_CACHE_DIR / f"{_HITRAN_REGISTERED_TABLE_NAME}.{suffix}"
        if not source.is_file():
            runtime_cache.cleanup()
            raise SensorDeviceError(f"registered HITRAN asset does not exist: {source}")
        shutil.copy2(source, Path(runtime_cache.name) / source.name)
    hapi.db_begin(runtime_cache.name)
    _HITRAN_RUNTIME_CACHE = runtime_cache


@lru_cache(maxsize=32)
def _hitran_absorption_cross_section(
    table_name: str,
    temperature_k: float,
    pressure_pa: float,
    wavenumber_range_cm1: tuple[float, float],
    wavenumber_step_cm1: float,
) -> tuple[np.ndarray, np.ndarray]:
    _initialize_hitran_database()
    import hapi

    wavenumber, coefficient = hapi.absorptionCoefficient_Voigt(
        SourceTables=table_name,
        Environment={"T": temperature_k, "p": pressure_pa / _PA_PER_ATM},
        Diluent={"air": 1.0},
        OmegaRange=list(wavenumber_range_cm1),
        OmegaStep=wavenumber_step_cm1,
        HITRAN_units=True,
    )
    axis = np.asarray(wavenumber, dtype=np.float64)
    cross_section = np.asarray(coefficient, dtype=np.float64)
    if (
        axis.ndim != 1
        or cross_section.shape != axis.shape
        or axis.size < 2
        or not np.isfinite(cross_section).all()
        or np.any(cross_section < 0.0)
    ):
        raise SensorDeviceError("HITRAN returned an invalid absorption spectrum")
    axis.setflags(write=False)
    cross_section.setflags(write=False)
    return axis, cross_section


def _gaussian_band_weight(
    wavenumber_cm1: np.ndarray,
    center_cm1: float,
    fwhm_cm1: float,
) -> np.ndarray:
    weight = np.exp(
        -4.0 * math.log(2.0) * ((wavenumber_cm1 - center_cm1) / fwhm_cm1) ** 2
    )
    integral = float(np.trapezoid(weight, wavenumber_cm1))
    if not math.isfinite(integral) or integral <= 0.0:
        raise SensorDeviceError("NDIR Gaussian band has zero integral")
    return weight / integral


def quantization_plateau_lengths(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """返回每个通道最长连续相等平台的长度，用于 NDIR 量化审计。"""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[0] == 0 or not np.isfinite(array).all():
        raise SensorDeviceError("values must be a non-empty finite (time, channel) array")
    longest = np.ones(array.shape[1], dtype=np.int64)
    for channel in range(array.shape[1]):
        run = 1
        for index in range(1, array.shape[0]):
            if array[index, channel] == array[index - 1, channel]:
                run += 1
                longest[channel] = max(longest[channel], run)
            else:
                run = 1
    return longest


def _coerce_ndir_profile(profile: NDIRDeviceProfile | Mapping[str, Any]) -> NDIRDeviceProfile:
    if isinstance(profile, NDIRDeviceProfile):
        profile.validate()
        return profile
    if isinstance(profile, Mapping):
        return NDIRDeviceProfile.from_mapping(profile)
    raise SensorDeviceError("profile must be an NDIRDeviceProfile or mapping")


def _broadcast(
    value: float | Sequence[float] | np.ndarray,
    count: int,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(count, float(values), dtype=np.float64)
    if values.shape != (count,):
        raise SensorDeviceError(f"{name} must be a scalar or shape ({count},)")
    if not np.isfinite(values).all():
        raise SensorDeviceError(f"{name} must contain only finite values")
    if positive and np.any(values <= 0.0):
        raise SensorDeviceError(f"{name} must be positive")
    if nonnegative and np.any(values < 0.0):
        raise SensorDeviceError(f"{name} must be non-negative")
    return values


# 便于生成器按设备名调用。
simulate_ultrasonic = acquire_ultrasonic_tof
simulate_tcd_energy_balance = simulate_tcd
acquire_tcd = simulate_tcd
simulate_ndir_active_reference = simulate_ndir


__all__ = [
    "NDIR_ADC_MAX_V",
    "NDIR_ADC_MIN_V",
    "NDIRDeviceProfile",
    "NDIRSimulationResult",
    "SensorDeviceError",
    "TCDSimulationResult",
    "TCD_BRIDGE_CALIBRATION_GAIN",
    "ThermalDeviceProfile",
    "ULTRASONIC_LOCK_CORRELATION_THRESHOLD",
    "ULTRASONIC_LOCK_SNR_THRESHOLD",
    "UltrasonicAcquisitionProfile",
    "UltrasonicAcquisitionResult",
    "UltrasonicLockError",
    "acquire_tcd",
    "acquire_ultrasonic_tof",
    "build_reference_waveform",
    "estimate_ndir_equilibrium_co2_series",
    "estimate_ultrasonic_quality_series",
    "estimate_ultrasonic_tof_series",
    "quantization_plateau_lengths",
    "simulate_ndir",
    "simulate_ndir_active_reference",
    "simulate_tcd",
    "simulate_tcd_energy_balance",
    "simulate_ultrasonic",
    "ultrasonic_signal_amplitude",
]
