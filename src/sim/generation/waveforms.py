from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from sim.generation.acoustic_physics import hidden_attenuation_v2, hidden_sound_speed_v2


# 物理后端注入：syngas 场景需要 6 参数声速/衰减（多 x_co）。默认使用
# hydrogen_ng 5 参数实现，保持向后兼容。
SoundSpeedFn = Callable[..., float]
AttenuationFn = Callable[..., dict[str, float]]


CENTER_FREQUENCY_HZ = 200000.0
BURST_CYCLES = 8
SAMPLE_RATE_HZ = 1000000
ULTRASONIC_MEASUREMENT_WINDOW_S = 0.005
FIBER_MIC_MEASUREMENT_WINDOW_S = 0.010
DEFAULT_DAQ_BITS = 20
DEFAULT_WAVEFORM_DTYPE = "int32"
ADC_MAX_INT16 = 32767  # 旧常量保留供向后兼容引用；新代码用 WaveformSpec.adc_max
DEFAULT_NOISE_STD_V = 1e-3
CALIBRATION_STATUS = "pending"
ACOUSTIC_ATTENUATION_MODEL = "semi_empirical_multigas_relaxation_proxy_v2"
ULTRASONIC_MODEL_NAME = "tof_observed_transducer_proxy_v1"
ULTRASONIC_SYSTEM_DELAY_MODEL = "fixed_delay_plus_trigger_jitter_v1"
ULTRASONIC_TRANSDUCER_RESPONSE_MODEL = "second_order_resonant_bandpass_proxy_v1"
FIBER_MIC_MODEL_NAME = "fiber_interferometric_proxy_v1"
FIBER_MIC_ACOUSTIC_FIELD_MODEL = "probe_pressure_with_optional_reflections_v1"
FIBER_OPTICAL_DEMODULATION_MODEL = "linear_phase_demodulation_proxy_v1"


def adc_max_from_bits(daq_bits: int) -> int:
    """由 ADC 位深推导有符号整数的满量程最大值（2**(bits-1) - 1）。"""
    if daq_bits < 2:
        raise ValueError(f"daq_bits must be >= 2, got {daq_bits}")
    return (1 << (daq_bits - 1)) - 1


@dataclass(frozen=True, slots=True)
class WaveformSpec:
    model_name: str = ULTRASONIC_MODEL_NAME
    sample_rate_hz: int = SAMPLE_RATE_HZ
    center_frequency_hz: float = CENTER_FREQUENCY_HZ
    burst_cycles: int = BURST_CYCLES
    measurement_window_s: float = ULTRASONIC_MEASUREMENT_WINDOW_S
    daq_bits: int = DEFAULT_DAQ_BITS
    waveform_dtype: str = DEFAULT_WAVEFORM_DTYPE
    per_timestep_scale: bool = False
    daq_full_scale_v: float = 2.5  # NI-6453 量程，Phase 0 推荐分辨率最优档（超声峰值 ≤1.0V）
    noise_std_v: float = DEFAULT_NOISE_STD_V
    acoustic_attenuation_model: str = ACOUSTIC_ATTENUATION_MODEL
    system_delay_model: str = ULTRASONIC_SYSTEM_DELAY_MODEL
    system_delay_s: float = 8.0e-5
    cable_delay_s: float = 2.0e-6
    delay_correction_s: float = 8.2e-5
    trigger_jitter_std_s: float = 3.0e-6
    transducer_response_model: str = ULTRASONIC_TRANSDUCER_RESPONSE_MODEL
    transducer_bandwidth_hz: float = 20000.0  # PSC200K datasheet 未给出带宽，按典型压电陶瓷 ~10% 中心频率估计
    transducer_ringdown_cycles: float = 4.0
    calibration_status: str = CALIBRATION_STATUS

    @property
    def adc_max(self) -> int:
        """ADC 满量程最大整数值，由 ``daq_bits`` 推导。"""
        return adc_max_from_bits(self.daq_bits)

    @property
    def waveform_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.measurement_window_s))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["adc_max"] = self.adc_max
        payload["waveform_samples"] = self.waveform_samples
        return payload


@dataclass(frozen=True, slots=True)
class FiberProbeSpec:
    model_name: str = FIBER_MIC_MODEL_NAME
    probe_position_m: float = 0.5
    probe_path_length_factor: float = 0.5
    source_pressure_pa: float = 1.0
    pressure_sensitivity_rad_per_pa: float = 1.0
    displacement_sensitivity_m_per_pa: float | None = None
    interferometer_phase_bias_rad: float = math.pi / 2.0
    optical_wavelength_nm: float = 1550.0
    optical_link_loss_db: float = 0.0
    demod_gain_v_per_rad: float = 1.0
    photodetector_noise_std_v: float = 1.0e-4
    amplifier_noise_std_v: float = 1.0e-4
    voltage_saturation_v: float = 5.0
    daq_full_scale_v: float = 5.0
    daq_bits: int = 16
    calibration_status: str = CALIBRATION_STATUS


@dataclass(frozen=True, slots=True)
class FiberMicSpec:
    model_name: str = FIBER_MIC_MODEL_NAME
    sample_rate_hz: int = SAMPLE_RATE_HZ
    center_frequency_hz: float = CENTER_FREQUENCY_HZ
    burst_cycles: int = BURST_CYCLES
    measurement_window_s: float = FIBER_MIC_MEASUREMENT_WINDOW_S
    daq_bits: int = DEFAULT_DAQ_BITS
    waveform_dtype: str = DEFAULT_WAVEFORM_DTYPE
    per_timestep_scale: bool = False
    acoustic_attenuation_model: str = ACOUSTIC_ATTENUATION_MODEL
    acoustic_field_model: str = FIBER_MIC_ACOUSTIC_FIELD_MODEL
    fiber_optical_demodulation_model: str = FIBER_OPTICAL_DEMODULATION_MODEL
    probe: FiberProbeSpec = field(default_factory=FiberProbeSpec)
    acoustic_reflection_coef: float = 0.08
    max_reflections: int = 3
    calibration_status: str = CALIBRATION_STATUS

    @property
    def adc_max(self) -> int:
        """ADC 满量程最大整数值，由 ``daq_bits`` 推导。"""
        return adc_max_from_bits(self.daq_bits)

    @property
    def waveform_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.measurement_window_s))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["adc_max"] = self.adc_max
        payload["waveform_samples"] = self.waveform_samples
        return payload


def generate_burst_pulse(
    *,
    center_frequency_hz: float = CENTER_FREQUENCY_HZ,
    burst_cycles: int = BURST_CYCLES,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
    amplitude_v: float = 1.0,
) -> np.ndarray:
    sample_count = int(round(burst_cycles * sample_rate_hz / center_frequency_hz))
    t = np.arange(sample_count, dtype=np.float32) / float(sample_rate_hz)
    window = np.hanning(sample_count).astype(np.float32)
    pulse = amplitude_v * window * np.sin(2.0 * math.pi * center_frequency_hz * t)
    return pulse.astype(np.float32)


def transducer_response_pulse(spec: WaveformSpec) -> np.ndarray:
    pulse = generate_burst_pulse(
        center_frequency_hz=spec.center_frequency_hz,
        burst_cycles=spec.burst_cycles,
        sample_rate_hz=spec.sample_rate_hz,
        amplitude_v=1.0,
    )
    kernel = _resonant_kernel(
        center_frequency_hz=spec.center_frequency_hz,
        sample_rate_hz=spec.sample_rate_hz,
        bandwidth_hz=spec.transducer_bandwidth_hz,
        ringdown_cycles=spec.transducer_ringdown_cycles,
    )
    shaped = np.convolve(pulse, kernel, mode="full").astype(np.float32)
    peak = float(np.max(np.abs(shaped)))
    if peak <= 0.0:
        raise ValueError("transducer response pulse must have non-zero amplitude")
    return (shaped / peak).astype(np.float32)


def _compute_physics(
    x_h2: float,
    x_ch4: float,
    x_co2: float,
    x_n2: float,
    t_c: float,
    p_mpa: float,
    h_rh: float,
    *,
    f_hz: float,
    sound_speed_fn: SoundSpeedFn | None = None,
    attenuation_fn: AttenuationFn | None = None,
    extra_gas_kwargs: dict[str, float] | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute sound speed and attenuation via the injected or default backend.

    When ``extra_gas_kwargs`` is provided, positional gas arguments are
    forwarded together with the extra kwargs; otherwise the hydrogen_ng
    5-parameter positional call is used for backward compatibility.
    """
    extra = extra_gas_kwargs or {}
    speed_fn = sound_speed_fn or hidden_sound_speed_v2
    atten_fn = attenuation_fn or hidden_attenuation_v2
    if extra:
        c_sound = speed_fn(x_h2, x_ch4, x_co2, x_n2, t_c=t_c, **extra)
        attenuation = atten_fn(
            x_h2, x_ch4, x_co2, x_n2,
            **extra,
            t_c=t_c, p_mpa=p_mpa, h_rh=h_rh,
            c_mix=c_sound, f_hz=f_hz,
        )
    else:
        c_sound = speed_fn(x_h2, x_ch4, x_co2, x_n2, t_c)
        attenuation = atten_fn(
            x_h2, x_ch4, x_co2, x_n2, t_c, p_mpa, h_rh,
            c_mix=c_sound, f_hz=f_hz,
        )
    return c_sound, attenuation


def simulate_waveform_measurement(
    *,
    x_h2: float,
    x_ch4: float,
    x_co2: float,
    x_n2: float,
    t_c: float,
    p_mpa: float,
    h_rh: float,
    l_m: float,
    seed: int,
    spec: WaveformSpec,
    sound_speed_fn: SoundSpeedFn | None = None,
    attenuation_fn: AttenuationFn | None = None,
    extra_gas_kwargs: dict[str, float] | None = None,
) -> dict[str, object]:
    """Ultrasonic waveform simulation.

    To support alternative physics backends (e.g. syngas with x_co), callers
    may inject ``sound_speed_fn`` / ``attenuation_fn`` plus ``extra_gas_kwargs``
    that are forwarded to those functions. When omitted, the hydrogen_ng
    backend (5-component without x_co) is used.
    """
    if l_m <= 0.0:
        raise ValueError("l_m must be > 0")
    rng = random.Random(seed)
    c_sound, attenuation = _compute_physics(
        x_h2, x_ch4, x_co2, x_n2, t_c, p_mpa, h_rh,
        f_hz=spec.center_frequency_hz,
        sound_speed_fn=sound_speed_fn,
        attenuation_fn=attenuation_fn,
        extra_gas_kwargs=extra_gas_kwargs,
    )
    alpha_true_npm = float(attenuation["alpha_true_v2"])
    tof_true_s = float(l_m) / c_sound
    trigger_jitter_s = rng.gauss(0.0, spec.trigger_jitter_std_s)
    tof_observed_s = tof_true_s + spec.system_delay_s + spec.cable_delay_s + trigger_jitter_s
    # 1 MS/s 下生成脉冲，用 Lagrange 分数延迟实现亚样本 TOF 定位
    pulse = transducer_response_pulse(spec)
    clean_waveform = np.zeros(spec.waveform_samples, dtype=np.float32)
    delay_samples = tof_observed_s * float(spec.sample_rate_hz)
    peak_index = int(round(delay_samples))
    amp_scale = math.exp(-alpha_true_npm * float(l_m))
    _add_pulse_at_peak(clean_waveform, pulse * amp_scale, peak_index, 1.0)
    clean_waveform = _lagrange_fractional_shift(clean_waveform, delay_samples - peak_index)
    waveform = clean_waveform.copy()
    if spec.noise_std_v > 0.0:
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        waveform = waveform + noise_rng.normal(0.0, spec.noise_std_v, size=waveform.shape).astype(np.float32)
    signal_peak_abs_v = float(np.max(np.abs(clean_waveform))) if clean_waveform.size else 0.0
    clipped = float(np.max(np.abs(waveform))) >= spec.daq_full_scale_v
    corrected_tof_s = max(tof_observed_s - spec.delay_correction_s, 1.0 / float(spec.sample_rate_hz))
    sound_speed_estimated = float(l_m) / corrected_tof_s
    tof_quality = _tof_quality(signal_peak_abs_v, spec.noise_std_v, clipped)
    return _digitize_waveform(waveform, spec.adc_max, spec.daq_full_scale_v, spec.waveform_dtype, spec.per_timestep_scale) | {
        "tof_s": tof_true_s,
        "tof_true_s": tof_true_s,
        "tof_observed_s": tof_observed_s,
        "trigger_jitter_s": trigger_jitter_s,
        "peak_index": peak_index,
        "tof_peak_index": peak_index,
        "tof_quality": tof_quality,
        "tof_accepted": int(tof_quality >= 0.5 and not clipped),
        "alpha_true_npm": alpha_true_npm,
        "sound_speed_m_per_s": float(c_sound),
        "sound_speed_estimated_m_per_s": sound_speed_estimated,
    }


def simulate_fiber_mic_measurement(
    *,
    x_h2: float,
    x_ch4: float,
    x_co2: float,
    x_n2: float,
    t_c: float,
    p_mpa: float,
    h_rh: float,
    l_m: float,
    seed: int,
    spec: FiberMicSpec,
    sound_speed_fn: SoundSpeedFn | None = None,
    attenuation_fn: AttenuationFn | None = None,
    extra_gas_kwargs: dict[str, float] | None = None,
) -> dict[str, object]:
    if l_m <= 0.0:
        raise ValueError("l_m must be > 0")
    rng = random.Random(seed)
    c_sound, attenuation = _compute_physics(
        x_h2, x_ch4, x_co2, x_n2, t_c, p_mpa, h_rh,
        f_hz=spec.center_frequency_hz,
        sound_speed_fn=sound_speed_fn,
        attenuation_fn=attenuation_fn,
        extra_gas_kwargs=extra_gas_kwargs,
    )
    alpha_true_npm = float(attenuation["alpha_true_v2"])
    probe = spec.probe
    l_probe = float(l_m) * float(probe.probe_path_length_factor)
    tof_probe_s = l_probe / c_sound
    t_round_s = (2.0 * float(l_m)) / c_sound
    # 1 MS/s 下生成声压场，每个脉冲用 Lagrange 分数延迟定位后累加
    probe = spec.probe
    pulse = generate_burst_pulse(
        center_frequency_hz=spec.center_frequency_hz,
        burst_cycles=spec.burst_cycles,
        sample_rate_hz=spec.sample_rate_hz,
        amplitude_v=probe.source_pressure_pa,
    )
    probe_pressure = np.zeros(spec.waveform_samples, dtype=np.float32)
    fs = float(spec.sample_rate_hz)
    # 直接脉冲 + 反射脉冲，各自分数延迟后放置
    pulses = [(tof_probe_s, math.exp(-alpha_true_npm * l_probe))]
    for reflection_idx in range(1, int(spec.max_reflections) + 1):
        path_length = l_probe + (2.0 * reflection_idx * float(l_m))
        amplitude = (float(spec.acoustic_reflection_coef) ** reflection_idx) * math.exp(-alpha_true_npm * path_length)
        pulses.append((tof_probe_s + reflection_idx * t_round_s, amplitude))
    for delay_s, amp in pulses:
        delay_samples = delay_s * fs
        peak_idx = int(round(delay_samples))
        if peak_idx < 0 or peak_idx >= probe_pressure.shape[0]:
            continue
        pulse_frac = _lagrange_fractional_shift(pulse, delay_samples - peak_idx)
        _add_pulse_at_peak(probe_pressure, pulse_frac * amp, peak_idx, 1.0)
    optical_link_gain = 10.0 ** (-float(probe.optical_link_loss_db) / 20.0)
    if probe.displacement_sensitivity_m_per_pa is None:
        delta_phase = probe.pressure_sensitivity_rad_per_pa * probe_pressure
    else:
        wavelength_m = float(probe.optical_wavelength_nm) * 1.0e-9
        if wavelength_m <= 0.0:
            raise ValueError("optical_wavelength_nm must be > 0")
        delta_d = float(probe.displacement_sensitivity_m_per_pa) * probe_pressure
        delta_phase = (4.0 * math.pi / wavelength_m) * delta_d
    phase = float(probe.interferometer_phase_bias_rad) + delta_phase
    demod_voltage = float(probe.demod_gain_v_per_rad) * optical_link_gain * (phase - float(probe.interferometer_phase_bias_rad))
    phase_peak_rad = float(np.max(np.abs(delta_phase))) if delta_phase.size else 0.0
    probe_pressure_peak_pa = float(np.max(np.abs(probe_pressure))) if probe_pressure.size else 0.0
    demod_peak_v = float(np.max(np.abs(demod_voltage))) if demod_voltage.size else 0.0
    noise_std = math.hypot(float(probe.photodetector_noise_std_v), float(probe.amplifier_noise_std_v))
    if noise_std > 0.0:
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        demod_voltage = demod_voltage + noise_rng.normal(0.0, noise_std, size=demod_voltage.shape).astype(np.float32)
    waveform = np.clip(demod_voltage, -float(probe.voltage_saturation_v), float(probe.voltage_saturation_v)).astype(np.float32)
    return _digitize_waveform(waveform, spec.adc_max, probe.daq_full_scale_v, spec.waveform_dtype, spec.per_timestep_scale) | {
        "tof_direct_s": float(tof_probe_s),
        "probe_tof_s": float(tof_probe_s),
        "t_round_s": float(t_round_s),
        "alpha_true_npm": alpha_true_npm,
        "sound_speed_m_per_s": float(c_sound),
        "probe_pressure_peak_pa": probe_pressure_peak_pa,
        "phase_peak_rad": phase_peak_rad,
        "demod_peak_v": demod_peak_v,
    }


def _add_pulse(buffer: np.ndarray, pulse: np.ndarray, start: int, amplitude: float) -> None:
    if start < 0 or start >= buffer.shape[0]:
        return
    usable = min(pulse.shape[0], buffer.shape[0] - start)
    if usable > 0:
        buffer[start : start + usable] += pulse[:usable] * amplitude


def _add_pulse_at_peak(buffer: np.ndarray, pulse: np.ndarray, peak_index: int, amplitude: float) -> None:
    pulse_peak_offset = int(np.argmax(np.abs(pulse))) if pulse.size else 0
    _add_pulse(buffer, pulse, peak_index - pulse_peak_offset, amplitude)


def _zero_pad_shift(signal: np.ndarray, n_int: int) -> np.ndarray:
    """整数样本零填充延迟（右移 n_int），移出边界的部分丢弃，不环绕。

    替代 np.roll 的环绕移位：物理上信号延迟出数组边界应消失，而非卷到另一端。
    """
    out = np.zeros_like(signal)
    length = signal.shape[0]
    if n_int == 0:
        out[:] = signal
    elif 0 < n_int < length:
        out[n_int:] = signal[: length - n_int]
    elif -length < n_int < 0:
        out[: length + n_int] = signal[-n_int:]
    return out


def _lagrange_fractional_shift(signal: np.ndarray, delay_samples: float, order: int = 5) -> np.ndarray:
    """在输出采样率下将信号精确移动 delay_samples（含整数 + 小数部分）。

    使用 Lagrange 插值 FIR，5 阶在 0~0.8×Nyquist 内精度 < 0.002 采样。
    200kHz 载波在 1MS/s 下为 0.4×Nyquist，安全裕度充足。
    参考：Laakso & Välimäki 1996 "Splitting the Unit Delay" IEEE SPM。

    整数部分用零填充移位（不环绕），分数部分用 Lagrange FIR 卷积
    （mode="same" 边界自然补零）。延迟出窗口的能量应消失，而非卷到数组另一端。
    """
    n_int = int(math.floor(delay_samples))
    frac = delay_samples - n_int
    if abs(frac) < 1e-9:
        return _zero_pad_shift(signal, n_int).astype(signal.dtype)
    n = order
    half = n // 2
    h = np.ones(n + 1, dtype=np.float64)
    for k in range(n + 1):
        for j in range(n + 1):
            if j != k:
                h[k] *= (frac - (j - half)) / ((k - half) - (j - half))
    shifted = _zero_pad_shift(signal, n_int)
    return np.convolve(shifted, h, mode="same").astype(signal.dtype)


def _digitize_waveform(
    waveform: np.ndarray,
    adc_max: int,
    daq_full_scale_v: float,
    waveform_dtype: str,
    per_timestep_scale: bool = False,
) -> dict[str, object]:
    if daq_full_scale_v <= 0.0:
        raise ValueError("daq_full_scale_v must be > 0")
    peak_abs_v = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak_abs_v <= 0.0:
        raise ValueError("peak_abs_v must be > 0")
    np_dtype = np.dtype(waveform_dtype)
    if per_timestep_scale:
        # per-timestep 自适应 scale：按波形峰值定标，充分利用 dtype 动态范围
        # int16 下峰值→32767，量化步长 = peak_abs_v / 32767
        dtype_max = (1 << (np_dtype.itemsize * 8 - 1)) - 1
        scale_factor = peak_abs_v / dtype_max
        clip_max = dtype_max
    else:
        scale_factor = float(daq_full_scale_v) / float(adc_max)
        clip_max = adc_max
    waveform_int = np.clip(np.round(waveform / scale_factor), -clip_max, clip_max).astype(np_dtype)
    return {
        "waveform_float": waveform.astype(np.float32),
        "waveform_int": waveform_int,
        "scale_factor": float(scale_factor),
        "peak_abs_v": peak_abs_v,
    }


def _resonant_kernel(
    *,
    center_frequency_hz: float,
    sample_rate_hz: int,
    bandwidth_hz: float,
    ringdown_cycles: float,
) -> np.ndarray:
    if center_frequency_hz <= 0.0 or sample_rate_hz <= 0 or bandwidth_hz <= 0.0 or ringdown_cycles <= 0.0:
        raise ValueError("transducer response parameters must be > 0")
    duration_s = float(ringdown_cycles) / float(center_frequency_hz)
    sample_count = max(3, int(round(duration_s * float(sample_rate_hz))))
    t = np.arange(sample_count, dtype=np.float32) / float(sample_rate_hz)
    tau_s = 1.0 / (math.pi * float(bandwidth_hz))
    kernel = np.exp(-t / tau_s) * np.sin(2.0 * math.pi * float(center_frequency_hz) * t)
    peak = float(np.max(np.abs(kernel))) if kernel.size else 0.0
    if peak <= 0.0:
        raise ValueError("transducer response kernel must have non-zero amplitude")
    return (kernel / peak).astype(np.float32)


def _tof_quality(signal_peak_abs_v: float, noise_std_v: float, clipped: bool) -> float:
    if noise_std_v <= 0.0:
        snr = 1.0e6
    else:
        snr = max(0.0, float(signal_peak_abs_v) / float(noise_std_v))
    quality = snr / (snr + 10.0)
    if clipped:
        quality *= 0.5
    return float(max(0.0, min(1.0, quality)))
