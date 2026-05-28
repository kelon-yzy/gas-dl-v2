from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

import numpy as np

from sim.generation.acoustic_physics import hidden_attenuation_v2, hidden_sound_speed_v2


CENTER_FREQUENCY_HZ = 40000.0
BURST_CYCLES = 8
SAMPLE_RATE_HZ = 200000
ULTRASONIC_MEASUREMENT_WINDOW_S = 0.005
FIBER_MIC_MEASUREMENT_WINDOW_S = 0.010
ADC_MAX_INT16 = 32767
DEFAULT_NOISE_STD_V = 1e-3
CALIBRATION_STATUS = "pending"
ACOUSTIC_ATTENUATION_MODEL = "semi_empirical_relaxation_proxy_v1"
ULTRASONIC_MODEL_NAME = "simplified_tof_proxy_v1"
FIBER_MIC_MODEL_NAME = "acoustic_proxy_v1"
FIBER_MIC_ACOUSTIC_FIELD_MODEL = "direct_plus_wall_reflections_proxy_v1"
FIBER_OPTICAL_DEMODULATION_MODEL = "not_implemented"


@dataclass(frozen=True, slots=True)
class WaveformSpec:
    model_name: str = ULTRASONIC_MODEL_NAME
    sample_rate_hz: int = SAMPLE_RATE_HZ
    center_frequency_hz: float = CENTER_FREQUENCY_HZ
    burst_cycles: int = BURST_CYCLES
    measurement_window_s: float = ULTRASONIC_MEASUREMENT_WINDOW_S
    adc_max_int16: int = ADC_MAX_INT16
    noise_std_v: float = DEFAULT_NOISE_STD_V
    acoustic_attenuation_model: str = ACOUSTIC_ATTENUATION_MODEL
    system_delay_model: str = "not_implemented"
    transducer_response_model: str = "ideal_burst_no_transducer_response"
    calibration_status: str = CALIBRATION_STATUS

    @property
    def waveform_samples(self) -> int:
        return int(round(self.sample_rate_hz * self.measurement_window_s))

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["waveform_samples"] = self.waveform_samples
        return payload


@dataclass(frozen=True, slots=True)
class FiberMicSpec:
    model_name: str = FIBER_MIC_MODEL_NAME
    sample_rate_hz: int = SAMPLE_RATE_HZ
    center_frequency_hz: float = CENTER_FREQUENCY_HZ
    burst_cycles: int = BURST_CYCLES
    measurement_window_s: float = FIBER_MIC_MEASUREMENT_WINDOW_S
    adc_max_int16: int = ADC_MAX_INT16
    noise_std_v: float = DEFAULT_NOISE_STD_V
    acoustic_attenuation_model: str = ACOUSTIC_ATTENUATION_MODEL
    acoustic_field_model: str = FIBER_MIC_ACOUSTIC_FIELD_MODEL
    fiber_optical_demodulation_model: str = FIBER_OPTICAL_DEMODULATION_MODEL
    l_direct_factor: float = 0.5
    wall_reflection_coef: float = 0.5
    max_reflections: int = 15
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
) -> dict[str, object]:
    if l_m <= 0.0:
        raise ValueError("l_m must be > 0")
    rng = random.Random(seed)
    c_sound = hidden_sound_speed_v2(x_h2, x_ch4, x_co2, x_n2, t_c)
    attenuation = hidden_attenuation_v2(x_h2, x_ch4, x_co2, x_n2, t_c, p_mpa, h_rh, c_mix=c_sound, f_hz=spec.center_frequency_hz)
    alpha_true_npm = float(attenuation["alpha_true_v2"])
    tof_s = float(l_m) / c_sound
    pulse = generate_burst_pulse(
        center_frequency_hz=spec.center_frequency_hz,
        burst_cycles=spec.burst_cycles,
        sample_rate_hz=spec.sample_rate_hz,
        amplitude_v=1.0,
    )
    waveform = np.zeros(spec.waveform_samples, dtype=np.float32)
    peak_index = int(round(tof_s * spec.sample_rate_hz))
    start = max(0, min(peak_index, waveform.shape[0] - 1))
    usable = min(pulse.shape[0], waveform.shape[0] - start)
    amp_scale = math.exp(-alpha_true_npm * float(l_m))
    waveform[start : start + usable] = pulse[:usable] * amp_scale
    if spec.noise_std_v > 0.0:
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        waveform = waveform + noise_rng.normal(0.0, spec.noise_std_v, size=waveform.shape).astype(np.float32)
    return _digitize_waveform(waveform, spec.adc_max_int16) | {
        "tof_s": tof_s,
        "peak_index": peak_index,
        "alpha_true_npm": alpha_true_npm,
        "sound_speed_m_per_s": float(c_sound),
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
) -> dict[str, object]:
    if l_m <= 0.0:
        raise ValueError("l_m must be > 0")
    rng = random.Random(seed)
    c_sound = hidden_sound_speed_v2(x_h2, x_ch4, x_co2, x_n2, t_c)
    attenuation = hidden_attenuation_v2(x_h2, x_ch4, x_co2, x_n2, t_c, p_mpa, h_rh, c_mix=c_sound, f_hz=spec.center_frequency_hz)
    alpha_true_npm = float(attenuation["alpha_true_v2"])
    l_direct = float(l_m) * float(spec.l_direct_factor)
    tof_direct_s = l_direct / c_sound
    t_round_s = (2.0 * float(l_m)) / c_sound
    pulse = generate_burst_pulse(
        center_frequency_hz=spec.center_frequency_hz,
        burst_cycles=spec.burst_cycles,
        sample_rate_hz=spec.sample_rate_hz,
        amplitude_v=1.0,
    )
    waveform = np.zeros(spec.waveform_samples, dtype=np.float32)
    _add_pulse(waveform, pulse, int(round(tof_direct_s * spec.sample_rate_hz)), math.exp(-alpha_true_npm * l_direct))
    for reflection_idx in range(1, int(spec.max_reflections) + 1):
        path_length = l_direct + (2.0 * reflection_idx * float(l_m))
        amplitude = (float(spec.wall_reflection_coef) ** reflection_idx) * math.exp(-alpha_true_npm * path_length)
        start = int(round((tof_direct_s + reflection_idx * t_round_s) * spec.sample_rate_hz))
        if start >= waveform.shape[0]:
            break
        _add_pulse(waveform, pulse, start, amplitude)
    if spec.noise_std_v > 0.0:
        noise_rng = np.random.default_rng(rng.randrange(0, 2**32))
        waveform = waveform + noise_rng.normal(0.0, spec.noise_std_v, size=waveform.shape).astype(np.float32)
    return _digitize_waveform(waveform, spec.adc_max_int16) | {
        "tof_direct_s": float(tof_direct_s),
        "t_round_s": float(t_round_s),
        "alpha_true_npm": alpha_true_npm,
        "sound_speed_m_per_s": float(c_sound),
    }


def _add_pulse(buffer: np.ndarray, pulse: np.ndarray, start: int, amplitude: float) -> None:
    if start < 0 or start >= buffer.shape[0]:
        return
    usable = min(pulse.shape[0], buffer.shape[0] - start)
    if usable > 0:
        buffer[start : start + usable] += pulse[:usable] * amplitude


def _digitize_waveform(waveform: np.ndarray, adc_max_int16: int) -> dict[str, object]:
    peak_abs_v = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak_abs_v <= 0.0:
        raise ValueError("peak_abs_v must be > 0")
    scale_factor = peak_abs_v / adc_max_int16
    waveform_int16 = np.clip(np.round(waveform / scale_factor), -adc_max_int16, adc_max_int16).astype(np.int16)
    return {
        "waveform_float": waveform.astype(np.float32),
        "waveform_int16": waveform_int16,
        "scale_factor": float(scale_factor),
        "peak_abs_v": peak_abs_v,
    }
