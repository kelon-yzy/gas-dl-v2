"""RCDW 超声 + 光纤麦克风波形仿真。

完全等价 HG 主线 ``src/sim/generation/waveforms.py`` 的物理建模 + 数字化逻辑，
仅替换组分签名 (x_O2 / x_CO2 / x_N2, 删除 x_H2 / x_CH4) 和默认物理后端为
``rcdw_sound_speed`` / ``rcdw_attenuation``。

对应方案 §5.6。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from rcdw.sim.generation.acoustic_physics import rcdw_attenuation, rcdw_sound_speed


SoundSpeedFn = Callable[..., float]
AttenuationFn = Callable[..., dict[str, float]]


CENTER_FREQUENCY_HZ = 40000.0
BURST_CYCLES = 8
SAMPLE_RATE_HZ = 200000
ULTRASONIC_MEASUREMENT_WINDOW_S = 0.005
FIBER_MIC_MEASUREMENT_WINDOW_S = 0.010
ADC_MAX_INT16 = 32767
DEFAULT_NOISE_STD_V = 1e-3
CALIBRATION_STATUS = "pending"
ACOUSTIC_ATTENUATION_MODEL = "linear_mixing_v1"  # RCDW 声学模型版本（与 acoustic_physics 保持一致）
ULTRASONIC_MODEL_NAME = "tof_observed_transducer_proxy_v1"
ULTRASONIC_SYSTEM_DELAY_MODEL = "fixed_delay_plus_trigger_jitter_v1"
ULTRASONIC_TRANSDUCER_RESPONSE_MODEL = "second_order_resonant_bandpass_proxy_v1"
FIBER_MIC_MODEL_NAME = "fiber_interferometric_proxy_v1"
FIBER_MIC_ACOUSTIC_FIELD_MODEL = "probe_pressure_with_optional_reflections_v1"
FIBER_OPTICAL_DEMODULATION_MODEL = "linear_phase_demodulation_proxy_v1"


@dataclass(frozen=True, slots=True)
class WaveformSpec:
    model_name: str = ULTRASONIC_MODEL_NAME
    sample_rate_hz: int = SAMPLE_RATE_HZ
    center_frequency_hz: float = CENTER_FREQUENCY_HZ
    burst_cycles: int = BURST_CYCLES
    measurement_window_s: float = ULTRASONIC_MEASUREMENT_WINDOW_S
    adc_max_int16: int = ADC_MAX_INT16
    daq_full_scale_v: float = 5.0
    noise_std_v: float = DEFAULT_NOISE_STD_V
    acoustic_attenuation_model: str = ACOUSTIC_ATTENUATION_MODEL
    system_delay_model: str = ULTRASONIC_SYSTEM_DELAY_MODEL
    system_delay_s: float = 8.0e-5
    cable_delay_s: float = 2.0e-6
    delay_correction_s: float = 8.2e-5
    trigger_jitter_std_s: float = 3.0e-6
    transducer_response_model: str = ULTRASONIC_TRANSDUCER_RESPONSE_MODEL
    transducer_bandwidth_hz: float = 12000.0
    transducer_ringdown_cycles: float = 4.0
    calibration_status: str = CALIBRATION_STATUS

    @property
    def waveform_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.measurement_window_s))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
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
    adc_max_int16: int = ADC_MAX_INT16
    acoustic_attenuation_model: str = ACOUSTIC_ATTENUATION_MODEL
    acoustic_field_model: str = FIBER_MIC_ACOUSTIC_FIELD_MODEL
    fiber_optical_demodulation_model: str = FIBER_OPTICAL_DEMODULATION_MODEL
    probe: FiberProbeSpec = field(default_factory=FiberProbeSpec)
    acoustic_reflection_coef: float = 0.08
    max_reflections: int = 3
    calibration_status: str = CALIBRATION_STATUS

    @property
    def waveform_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.measurement_window_s))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
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
    x_o2: float,
    x_co2: float,
    x_n2: float,
    t_c: float,
    p_mpa: float,
    h_rh: float,
    *,
    f_hz: float,
    sound_speed_fn: SoundSpeedFn | None = None,
    attenuation_fn: AttenuationFn | None = None,
) -> tuple[float, dict[str, float]]:
    """计算声速与衰减；默认使用 RCDW 三组分后端。"""
    speed_fn = sound_speed_fn or rcdw_sound_speed
    atten_fn = attenuation_fn or rcdw_attenuation
    c_sound = speed_fn(x_o2, x_co2, x_n2, t_c)
    attenuation = atten_fn(
        x_o2, x_co2, x_n2, t_c, p_mpa, h_rh, c_mix=c_sound, f_hz=f_hz
    )
    return c_sound, attenuation


def simulate_waveform_measurement(
    *,
    x_o2: float,
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
) -> dict[str, object]:
    """RCDW 超声波形仿真。

    返回包含数字化波形、TOF 真值/观测值、声速估计值、质量分数等的字典。
    """
    if l_m <= 0.0:
        raise ValueError("l_m must be > 0")
    rng = random.Random(seed)
    c_sound, attenuation = _compute_physics(
        x_o2, x_co2, x_n2, t_c, p_mpa, h_rh,
        f_hz=spec.center_frequency_hz,
        sound_speed_fn=sound_speed_fn,
        attenuation_fn=attenuation_fn,
    )
    alpha_true_npm = float(attenuation["alpha_true_v2"])
    tof_true_s = float(l_m) / c_sound
    trigger_jitter_s = rng.gauss(0.0, spec.trigger_jitter_std_s)
    tof_observed_s = (
        tof_true_s + spec.system_delay_s + spec.cable_delay_s + trigger_jitter_s
    )
    pulse = transducer_response_pulse(spec)
    clean_waveform = np.zeros(spec.waveform_samples, dtype=np.float32)
    peak_index = int(round(tof_observed_s * spec.sample_rate_hz))
    amp_scale = math.exp(-alpha_true_npm * float(l_m))
    _add_pulse_at_peak(clean_waveform, pulse, peak_index, amp_scale)
    waveform = clean_waveform.copy()
    if spec.noise_std_v > 0.0:
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        waveform = waveform + noise_rng.normal(
            0.0, spec.noise_std_v, size=waveform.shape
        ).astype(np.float32)
    signal_peak_abs_v = (
        float(np.max(np.abs(clean_waveform))) if clean_waveform.size else 0.0
    )
    clipped = float(np.max(np.abs(waveform))) >= spec.daq_full_scale_v
    corrected_tof_s = max(
        tof_observed_s - spec.delay_correction_s,
        1.0 / float(spec.sample_rate_hz),
    )
    sound_speed_estimated = float(l_m) / corrected_tof_s
    tof_quality = _tof_quality(signal_peak_abs_v, spec.noise_std_v, clipped)
    return _digitize_waveform(waveform, spec.adc_max_int16, spec.daq_full_scale_v) | {
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
    x_o2: float,
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
) -> dict[str, object]:
    """光纤麦克风波形：与超声同步生成，但不进入训练（方案 §2.4）。"""
    if l_m <= 0.0:
        raise ValueError("l_m must be > 0")
    rng = random.Random(seed)
    c_sound, attenuation = _compute_physics(
        x_o2, x_co2, x_n2, t_c, p_mpa, h_rh,
        f_hz=spec.center_frequency_hz,
        sound_speed_fn=sound_speed_fn,
        attenuation_fn=attenuation_fn,
    )
    alpha_true_npm = float(attenuation["alpha_true_v2"])
    probe = spec.probe
    l_probe = float(l_m) * float(probe.probe_path_length_factor)
    tof_probe_s = l_probe / c_sound
    t_round_s = (2.0 * float(l_m)) / c_sound
    pulse = generate_burst_pulse(
        center_frequency_hz=spec.center_frequency_hz,
        burst_cycles=spec.burst_cycles,
        sample_rate_hz=spec.sample_rate_hz,
        amplitude_v=probe.source_pressure_pa,
    )
    probe_pressure = np.zeros(spec.waveform_samples, dtype=np.float32)
    _add_pulse_at_peak(
        probe_pressure,
        pulse,
        int(round(tof_probe_s * spec.sample_rate_hz)),
        math.exp(-alpha_true_npm * l_probe),
    )
    for reflection_idx in range(1, int(spec.max_reflections) + 1):
        path_length = l_probe + (2.0 * reflection_idx * float(l_m))
        amplitude = (
            float(spec.acoustic_reflection_coef) ** reflection_idx
        ) * math.exp(-alpha_true_npm * path_length)
        peak_index = int(
            round((tof_probe_s + reflection_idx * t_round_s) * spec.sample_rate_hz)
        )
        if peak_index >= probe_pressure.shape[0]:
            break
        _add_pulse_at_peak(probe_pressure, pulse, peak_index, amplitude)
    optical_link_gain = 10.0 ** (-float(probe.optical_link_loss_db) / 20.0)
    if probe.displacement_sensitivity_m_per_pa is None:
        delta_phase = probe.pressure_sensitivity_rad_per_pa * probe_pressure
    else:
        wavelength_m = float(probe.optical_wavelength_nm) * 1.0e-9
        if wavelength_m <= 0.0:
            raise ValueError("optical_wavelength_nm must be > 0")
        delta_d = (
            float(probe.displacement_sensitivity_m_per_pa) * probe_pressure
        )
        delta_phase = (4.0 * math.pi / wavelength_m) * delta_d
    phase = float(probe.interferometer_phase_bias_rad) + delta_phase
    demod_voltage = (
        float(probe.demod_gain_v_per_rad)
        * optical_link_gain
        * (phase - float(probe.interferometer_phase_bias_rad))
    )
    noise_std = math.hypot(
        float(probe.photodetector_noise_std_v),
        float(probe.amplifier_noise_std_v),
    )
    if noise_std > 0.0:
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        demod_voltage = demod_voltage + noise_rng.normal(
            0.0, noise_std, size=demod_voltage.shape
        ).astype(np.float32)
    waveform = np.clip(
        demod_voltage,
        -float(probe.voltage_saturation_v),
        float(probe.voltage_saturation_v),
    ).astype(np.float32)
    phase_peak_rad = float(np.max(np.abs(delta_phase))) if delta_phase.size else 0.0
    return _digitize_waveform(
        waveform, spec.adc_max_int16, probe.daq_full_scale_v
    ) | {
        "tof_direct_s": float(tof_probe_s),
        "probe_tof_s": float(tof_probe_s),
        "t_round_s": float(t_round_s),
        "alpha_true_npm": alpha_true_npm,
        "sound_speed_m_per_s": float(c_sound),
        "probe_pressure_peak_pa": float(np.max(np.abs(probe_pressure)))
        if probe_pressure.size
        else 0.0,
        "phase_peak_rad": phase_peak_rad,
        "demod_peak_v": float(np.max(np.abs(demod_voltage)))
        if demod_voltage.size
        else 0.0,
    }


def _add_pulse(
    buffer: np.ndarray, pulse: np.ndarray, start: int, amplitude: float
) -> None:
    if start < 0 or start >= buffer.shape[0]:
        return
    usable = min(pulse.shape[0], buffer.shape[0] - start)
    if usable > 0:
        buffer[start : start + usable] += pulse[:usable] * amplitude


def _add_pulse_at_peak(
    buffer: np.ndarray, pulse: np.ndarray, peak_index: int, amplitude: float
) -> None:
    pulse_peak_offset = int(np.argmax(np.abs(pulse))) if pulse.size else 0
    _add_pulse(buffer, pulse, peak_index - pulse_peak_offset, amplitude)


def _digitize_waveform(
    waveform: np.ndarray, adc_max_int16: int, daq_full_scale_v: float
) -> dict[str, object]:
    if daq_full_scale_v <= 0.0:
        raise ValueError("daq_full_scale_v must be > 0")
    peak_abs_v = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak_abs_v <= 0.0:
        raise ValueError("peak_abs_v must be > 0")
    scale_factor = float(daq_full_scale_v) / float(adc_max_int16)
    waveform_int16 = (
        np.clip(np.round(waveform / scale_factor), -adc_max_int16, adc_max_int16)
        .astype(np.int16)
    )
    return {
        "waveform_float": waveform.astype(np.float32),
        "waveform_int16": waveform_int16,
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
    if (
        center_frequency_hz <= 0.0
        or sample_rate_hz <= 0
        or bandwidth_hz <= 0.0
        or ringdown_cycles <= 0.0
    ):
        raise ValueError("transducer response parameters must be > 0")
    duration_s = float(ringdown_cycles) / float(center_frequency_hz)
    sample_count = max(3, int(round(duration_s * float(sample_rate_hz))))
    t = np.arange(sample_count, dtype=np.float32) / float(sample_rate_hz)
    tau_s = 1.0 / (math.pi * float(bandwidth_hz))
    kernel = np.exp(-t / tau_s) * np.sin(
        2.0 * math.pi * float(center_frequency_hz) * t
    )
    peak = float(np.max(np.abs(kernel))) if kernel.size else 0.0
    if peak <= 0.0:
        raise ValueError("transducer response kernel must have non-zero amplitude")
    return (kernel / peak).astype(np.float32)


def _tof_quality(
    signal_peak_abs_v: float, noise_std_v: float, clipped: bool
) -> float:
    if noise_std_v <= 0.0:
        snr = 1.0e6
    else:
        snr = max(0.0, float(signal_peak_abs_v) / float(noise_std_v))
    quality = snr / (snr + 10.0)
    if clipped:
        quality *= 0.5
    return float(max(0.0, min(1.0, quality)))
