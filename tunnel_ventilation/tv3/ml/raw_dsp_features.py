from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import hilbert
from scipy.stats import theilslopes

from tv3.sim.generation.tunnel_ventilation.acoustic_physics import hidden_sound_speed_v2
from tv3.sim.generation.waveforms import WaveformSpec, transducer_response_pulse


RAW_DSP_FRAME_SCHEMA_VERSION = "tv3-raw-dsp-frame-1"
FRESH_AIR_COMPOSITION = (0.04, 20.90, 79.06)
FORMAL_SLOW_CHANNELS = (
    "V_NDIR_CO2",
    "V_TCS",
    "T_C",
    "P_MPa",
    "H_RH",
    "L_m",
    "piston_position_m",
)
REQUIRED_SLOW_CHANNELS = ("T_C", "P_MPa", "H_RH", "L_m")


@dataclass(frozen=True, slots=True)
class RawDSPConfig:
    sample_rate_hz: float
    carrier_frequency_hz: float = 200000.0
    sound_speed_min_m_per_s: float = 250.0
    sound_speed_max_m_per_s: float = 400.0
    delay_min_s: float = 40.0e-6
    delay_max_s: float = 130.0e-6
    min_corr_peak: float = 0.50
    min_peak_to_sidelobe_ratio: float = 1.10
    min_snr_db: float = 10.0
    max_peak_width_samples: float = 12.0
    sidelobe_exclusion_samples: int = 8
    calibration_min_frames: int = 4
    snr_db_cap: float = 120.0
    peak_to_sidelobe_ratio_cap: float = 1.0e6


@dataclass(frozen=True, slots=True)
class RawDSPFrameResult:
    peak_index: float
    tof_observed_s: float
    corr_peak: float
    peak_to_sidelobe_ratio: float
    snr_db: float
    peak_width_samples: float
    peak_amplitude_v: float
    clipped: bool
    boundary_hit: bool
    quality: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class RawDSPSequenceResult:
    peak_index: np.ndarray
    tof_observed_s: np.ndarray
    delay_calibration_s: float
    tof_corrected_s: np.ndarray
    sound_speed_m_per_s: np.ndarray
    corr_peak: np.ndarray
    peak_to_sidelobe_ratio: np.ndarray
    snr_db: np.ndarray
    peak_width_samples: np.ndarray
    peak_amplitude_v: np.ndarray
    clipped: np.ndarray
    boundary_hit: np.ndarray
    quality: np.ndarray
    accepted: np.ndarray
    tof_l_m_intercept_s: float
    sound_speed_slope_m_per_s: float


def validate_raw_dsp_config(config: RawDSPConfig) -> None:
    if config.sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be > 0")
    if config.carrier_frequency_hz <= 0.0 or config.carrier_frequency_hz >= 0.5 * config.sample_rate_hz:
        raise ValueError("carrier_frequency_hz must be in (0, Nyquist)")
    if config.sound_speed_min_m_per_s <= 0.0:
        raise ValueError("sound_speed_min_m_per_s must be > 0")
    if config.sound_speed_max_m_per_s <= config.sound_speed_min_m_per_s:
        raise ValueError("sound_speed_max_m_per_s must exceed sound_speed_min_m_per_s")
    if config.delay_min_s < 0.0 or config.delay_max_s <= config.delay_min_s:
        raise ValueError("delay bounds must satisfy 0 <= delay_min_s < delay_max_s")
    if not 0.0 <= config.min_corr_peak <= 1.0:
        raise ValueError("min_corr_peak must be in [0, 1]")
    if config.min_peak_to_sidelobe_ratio <= 1.0:
        raise ValueError("min_peak_to_sidelobe_ratio must be > 1")
    if config.max_peak_width_samples <= 0.0:
        raise ValueError("max_peak_width_samples must be > 0")
    if config.sidelobe_exclusion_samples < 1:
        raise ValueError("sidelobe_exclusion_samples must be >= 1")
    if config.calibration_min_frames < 1:
        raise ValueError("calibration_min_frames must be >= 1")


def dequantize_waveforms(waveform_int: np.ndarray, scale: np.ndarray | float) -> np.ndarray:
    waveform_int = np.asarray(waveform_int)
    scale_array = np.asarray(scale, dtype=np.float32)
    if not np.issubdtype(waveform_int.dtype, np.integer):
        raise TypeError(f"waveform_int must have integer dtype, got {waveform_int.dtype}")
    if waveform_int.ndim < 1:
        raise ValueError("waveform_int must have at least one dimension")
    expected_scale_shape = waveform_int.shape[:-1]
    if scale_array.shape not in {(), expected_scale_shape}:
        raise ValueError(
            f"scale shape must be scalar or {expected_scale_shape}, got {scale_array.shape}"
        )
    if not np.isfinite(scale_array).all() or np.any(scale_array <= 0.0):
        raise ValueError("waveform scale must contain finite positive values")
    return waveform_int.astype(np.float32) * np.expand_dims(scale_array, axis=-1)


def exact_simulator_template(waveform_spec: Mapping[str, object]) -> np.ndarray:
    allowed = {field.name for field in fields(WaveformSpec)}
    kwargs = {key: waveform_spec[key] for key in allowed if key in waveform_spec}
    spec = WaveformSpec(**kwargs)
    template = transducer_response_pulse(spec)
    return np.asarray(template, dtype=np.float32).copy()


def template_digest(template: np.ndarray) -> str:
    values = _validate_template(template)
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes()).hexdigest()


def build_baseline_median_template(
    waveform_int_frames: np.ndarray,
    scales: np.ndarray,
    path_lengths_m: np.ndarray,
    *,
    config: RawDSPConfig,
    daq_full_scale_v: float,
    template_pre_samples: int,
    template_post_samples: int,
    min_template_snr_db: float,
    reference_peak_polarity: int = -1,
) -> np.ndarray:
    validate_raw_dsp_config(config)
    if template_pre_samples < 1 or template_post_samples < 1:
        raise ValueError("template_pre_samples and template_post_samples must be >= 1")
    if reference_peak_polarity not in {-1, 1}:
        raise ValueError("reference_peak_polarity must be -1 or 1")
    waveforms = dequantize_waveforms(waveform_int_frames, scales)
    path_lengths = np.asarray(path_lengths_m, dtype=np.float64)
    if waveforms.ndim != 2:
        raise ValueError(f"baseline waveform frames must be 2D, got {waveforms.shape}")
    if path_lengths.shape != (waveforms.shape[0],):
        raise ValueError(
            f"path_lengths_m shape must be {(waveforms.shape[0],)}, got {path_lengths.shape}"
        )
    if daq_full_scale_v <= 0.0:
        raise ValueError("daq_full_scale_v must be > 0")

    patches: list[np.ndarray] = []
    patch_length = template_pre_samples + template_post_samples + 1
    for waveform, path_length_m in zip(waveforms, path_lengths, strict=True):
        lower, upper = physical_peak_window_samples(
            path_length_m,
            waveform_samples=waveform.shape[0],
            config=config,
        )
        local = waveform[lower : upper + 1]
        coarse_peak = lower + _coarse_reference_peak(local, config, reference_peak_polarity)
        start = coarse_peak - template_pre_samples
        stop = coarse_peak + template_post_samples + 1
        if start < 0 or stop > waveform.shape[0]:
            continue
        noise = _noise_samples(waveform, lower, upper, template_pre_samples, template_post_samples)
        snr_db = _snr_db(float(np.max(np.abs(local))), noise, config.snr_db_cap)
        if snr_db < min_template_snr_db or float(np.max(np.abs(waveform))) >= daq_full_scale_v:
            continue
        patch = waveform[start:stop].astype(np.float64, copy=True)
        patch -= float(np.mean(patch))
        peak_abs = float(np.max(np.abs(patch)))
        if peak_abs <= 0.0:
            continue
        patches.append((patch / peak_abs).astype(np.float32))

    if not patches:
        raise ValueError("no baseline frames passed the train template quality criteria")
    stack = np.stack(patches, axis=0)
    if stack.shape[1] != patch_length:
        raise RuntimeError("baseline template patches have inconsistent lengths")
    template = np.median(stack, axis=0).astype(np.float32)
    peak_abs = float(np.max(np.abs(template)))
    if peak_abs <= 0.0:
        raise ValueError("median baseline template has zero amplitude")
    template = template / peak_abs
    if int(np.sign(template[template_pre_samples])) != reference_peak_polarity:
        template = -template
    return template.astype(np.float32)


def physical_peak_window_samples(
    path_length_m: float,
    *,
    waveform_samples: int,
    config: RawDSPConfig,
) -> tuple[int, int]:
    validate_raw_dsp_config(config)
    if not math.isfinite(path_length_m) or path_length_m <= 0.0:
        raise ValueError(f"path_length_m must be finite and > 0, got {path_length_m}")
    if waveform_samples < 3:
        raise ValueError("waveform_samples must be >= 3")
    lower_time_s = path_length_m / config.sound_speed_max_m_per_s + config.delay_min_s
    upper_time_s = path_length_m / config.sound_speed_min_m_per_s + config.delay_max_s
    lower = max(0, int(math.floor(lower_time_s * config.sample_rate_hz)))
    upper = min(waveform_samples - 1, int(math.ceil(upper_time_s * config.sample_rate_hz)))
    if lower >= upper:
        raise ValueError(
            f"physical peak window is empty after clipping: lower={lower}, upper={upper}, "
            f"waveform_samples={waveform_samples}"
        )
    return lower, upper


def parabolic_peak_offset(left: float, center: float, right: float) -> float:
    values = np.asarray((left, center, right), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("parabolic interpolation values must be finite")
    if center < left or center < right:
        raise ValueError("parabolic interpolation center must be a local maximum")
    denominator = left - (2.0 * center) + right
    if denominator == 0.0:
        return 0.0
    offset = 0.5 * (left - right) / denominator
    return float(np.clip(offset, -0.5, 0.5))


def extract_raw_dsp_frame(
    waveform_v: np.ndarray,
    template: np.ndarray,
    *,
    path_length_m: float,
    daq_full_scale_v: float,
    config: RawDSPConfig,
    template_peak_offset_samples: int | None = None,
) -> RawDSPFrameResult:
    validate_raw_dsp_config(config)
    waveform = np.asarray(waveform_v, dtype=np.float64)
    template_values = _validate_template(template).astype(np.float64)
    if waveform.ndim != 1:
        raise ValueError(f"waveform_v must be 1D, got {waveform.shape}")
    if not np.isfinite(waveform).all():
        raise ValueError("waveform_v contains non-finite values")
    if daq_full_scale_v <= 0.0:
        raise ValueError("daq_full_scale_v must be > 0")

    peak_lower, peak_upper = physical_peak_window_samples(
        path_length_m,
        waveform_samples=waveform.shape[0],
        config=config,
    )
    template_peak_offset = (
        int(np.argmax(np.abs(template_values)))
        if template_peak_offset_samples is None
        else int(template_peak_offset_samples)
    )
    if template_peak_offset < 0 or template_peak_offset >= template_values.size:
        raise ValueError(
            f"template_peak_offset_samples must be in [0, {template_values.size}), "
            f"got {template_peak_offset}"
        )
    start_lower = max(0, peak_lower - template_peak_offset)
    start_upper = min(waveform.shape[0] - template_values.size, peak_upper - template_peak_offset)
    if start_lower >= start_upper:
        raise ValueError(
            f"template does not fit inside the physical peak window: starts={start_lower}:{start_upper}"
        )

    segment = waveform[start_lower : start_upper + template_values.size]
    template_centered = template_values - float(np.mean(template_values))
    template_norm = float(np.linalg.norm(template_centered))
    if template_norm <= 0.0:
        raise ValueError("template has zero centered energy")
    numerator = np.correlate(segment, template_centered, mode="valid")
    window_length = template_values.size
    cumulative = np.concatenate(([0.0], np.cumsum(segment, dtype=np.float64)))
    cumulative_square = np.concatenate(([0.0], np.cumsum(segment * segment, dtype=np.float64)))
    local_sum = cumulative[window_length:] - cumulative[:-window_length]
    local_square_sum = cumulative_square[window_length:] - cumulative_square[:-window_length]
    local_energy = local_square_sum - (local_sum * local_sum / float(window_length))
    denominator = template_norm * np.sqrt(np.maximum(local_energy, 0.0))
    correlation = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, -np.inf, dtype=np.float64),
        where=denominator > 0.0,
    )
    if not np.isfinite(correlation).any():
        raise ValueError("physical search window contains no non-zero waveform energy")

    best = _phase_locked_peak_index(correlation, config)
    boundary_hit = best == 0 or best == correlation.size - 1
    fractional_offset = 0.0
    corr_peak = float(correlation[best])
    if not boundary_hit:
        left = float(correlation[best - 1])
        center = float(correlation[best])
        right = float(correlation[best + 1])
        fractional_offset = parabolic_peak_offset(left, center, right)
        corr_peak = center - 0.25 * (left - right) * fractional_offset
    corr_peak = float(np.clip(corr_peak, -1.0, 1.0))
    peak_index = start_lower + best + template_peak_offset + fractional_offset

    peak_to_sidelobe_ratio = _peak_to_sidelobe_ratio(
        correlation,
        best,
        exclusion=config.sidelobe_exclusion_samples,
        cap=config.peak_to_sidelobe_ratio_cap,
    )
    peak_width_samples = _peak_width_samples(correlation, best)
    signal_region = waveform[peak_lower : peak_upper + 1]
    peak_amplitude_v = float(np.max(np.abs(signal_region)))
    noise = _noise_samples(
        waveform,
        peak_lower,
        peak_upper,
        template_peak_offset,
        template_values.size - template_peak_offset - 1,
    )
    snr_db = _snr_db(peak_amplitude_v, noise, config.snr_db_cap)
    clipped = float(np.max(np.abs(waveform))) >= daq_full_scale_v
    accepted = bool(
        corr_peak >= config.min_corr_peak
        and peak_to_sidelobe_ratio >= config.min_peak_to_sidelobe_ratio
        and snr_db >= config.min_snr_db
        and peak_width_samples <= config.max_peak_width_samples
        and not clipped
        and not boundary_hit
    )
    quality = _quality_score(
        corr_peak=corr_peak,
        peak_to_sidelobe_ratio=peak_to_sidelobe_ratio,
        snr_db=snr_db,
        clipped=clipped,
        boundary_hit=boundary_hit,
        config=config,
    )
    return RawDSPFrameResult(
        peak_index=float(peak_index),
        tof_observed_s=float(peak_index / config.sample_rate_hz),
        corr_peak=corr_peak,
        peak_to_sidelobe_ratio=peak_to_sidelobe_ratio,
        snr_db=snr_db,
        peak_width_samples=peak_width_samples,
        peak_amplitude_v=peak_amplitude_v,
        clipped=clipped,
        boundary_hit=boundary_hit,
        quality=quality,
        accepted=accepted,
    )


def extract_raw_dsp_sequence(
    waveform_int: np.ndarray,
    scales: np.ndarray,
    slow: np.ndarray,
    slow_channel_names: Sequence[str],
    phase_ids: Sequence[str],
    template: np.ndarray,
    *,
    daq_full_scale_v: float,
    config: RawDSPConfig,
    template_peak_offset_samples: int | None = None,
) -> RawDSPSequenceResult:
    waveforms = dequantize_waveforms(waveform_int, scales)
    slow_values = np.asarray(slow, dtype=np.float64)
    if waveforms.ndim != 2:
        raise ValueError(f"sequence waveform must be 2D, got {waveforms.shape}")
    if slow_values.ndim != 2 or slow_values.shape[0] != waveforms.shape[0]:
        raise ValueError(
            f"slow sequence must have shape (timesteps, channels), got {slow_values.shape}"
        )
    if len(phase_ids) != waveforms.shape[0]:
        raise ValueError(f"phase length mismatch: {len(phase_ids)} != {waveforms.shape[0]}")
    channel_indices = _slow_channel_indices(slow_channel_names)
    path_lengths = slow_values[:, channel_indices["L_m"]]

    frames = [
        extract_raw_dsp_frame(
            waveform,
            template,
            path_length_m=float(path_length_m),
            daq_full_scale_v=daq_full_scale_v,
            config=config,
            template_peak_offset_samples=template_peak_offset_samples,
        )
        for waveform, path_length_m in zip(waveforms, path_lengths, strict=True)
    ]
    peak_index = np.asarray([frame.peak_index for frame in frames], dtype=np.float64)
    tof_observed_s = peak_index / config.sample_rate_hz
    corr_peak = np.asarray([frame.corr_peak for frame in frames], dtype=np.float64)
    peak_to_sidelobe_ratio = np.asarray(
        [frame.peak_to_sidelobe_ratio for frame in frames], dtype=np.float64
    )
    snr_db = np.asarray([frame.snr_db for frame in frames], dtype=np.float64)
    peak_width_samples = np.asarray([frame.peak_width_samples for frame in frames], dtype=np.float64)
    peak_amplitude_v = np.asarray([frame.peak_amplitude_v for frame in frames], dtype=np.float64)
    clipped = np.asarray([frame.clipped for frame in frames], dtype=bool)
    boundary_hit = np.asarray([frame.boundary_hit for frame in frames], dtype=bool)
    quality = np.asarray([frame.quality for frame in frames], dtype=np.float64)
    accepted = np.asarray([frame.accepted for frame in frames], dtype=bool)

    temperature_c = slow_values[:, channel_indices["T_C"]]
    pressure_mpa = slow_values[:, channel_indices["P_MPa"]]
    humidity_rh = slow_values[:, channel_indices["H_RH"]]
    delay_calibration_s = calibrate_sequence_delay_s(
        tof_observed_s,
        path_lengths,
        temperature_c,
        pressure_mpa,
        humidity_rh,
        phase_ids,
        accepted,
        min_frames=config.calibration_min_frames,
    )
    tof_corrected_s = tof_observed_s - delay_calibration_s
    if np.any(tof_corrected_s <= 0.0):
        raise ValueError("delay calibration produced non-positive corrected TOF")
    sound_speed = path_lengths / tof_corrected_s
    intercept_s, sound_speed_slope = fit_tof_vs_path_length(
        tof_observed_s,
        path_lengths,
        phase_ids,
        accepted,
    )
    return RawDSPSequenceResult(
        peak_index=peak_index.astype(np.float32),
        tof_observed_s=tof_observed_s.astype(np.float32),
        delay_calibration_s=float(delay_calibration_s),
        tof_corrected_s=tof_corrected_s.astype(np.float32),
        sound_speed_m_per_s=sound_speed.astype(np.float32),
        corr_peak=corr_peak.astype(np.float32),
        peak_to_sidelobe_ratio=peak_to_sidelobe_ratio.astype(np.float32),
        snr_db=snr_db.astype(np.float32),
        peak_width_samples=peak_width_samples.astype(np.float32),
        peak_amplitude_v=peak_amplitude_v.astype(np.float32),
        clipped=clipped,
        boundary_hit=boundary_hit,
        quality=quality.astype(np.float32),
        accepted=accepted,
        tof_l_m_intercept_s=float(intercept_s),
        sound_speed_slope_m_per_s=float(sound_speed_slope),
    )


def fresh_air_sound_speed_m_per_s(
    temperature_c: np.ndarray,
    pressure_mpa: np.ndarray,
    humidity_rh: np.ndarray,
) -> np.ndarray:
    temperature = np.asarray(temperature_c, dtype=np.float64)
    pressure = np.asarray(pressure_mpa, dtype=np.float64)
    humidity = np.asarray(humidity_rh, dtype=np.float64)
    if temperature.shape != pressure.shape or temperature.shape != humidity.shape:
        raise ValueError("temperature, pressure, and humidity must have identical shapes")
    if not np.isfinite(temperature).all() or not np.isfinite(pressure).all() or not np.isfinite(humidity).all():
        raise ValueError("fresh-air environment arrays must be finite")
    x_co2, x_o2, x_n2 = FRESH_AIR_COMPOSITION
    return np.asarray(
        [
            hidden_sound_speed_v2(
                0.0,
                0.0,
                x_co2,
                x_n2,
                t_c=float(current_temperature),
                x_o2=x_o2,
            )
            for current_temperature in temperature.flat
        ],
        dtype=np.float64,
    ).reshape(temperature.shape)


def calibrate_sequence_delay_s(
    tof_observed_s: np.ndarray,
    path_lengths_m: np.ndarray,
    temperature_c: np.ndarray,
    pressure_mpa: np.ndarray,
    humidity_rh: np.ndarray,
    phase_ids: Sequence[str],
    accepted: np.ndarray,
    *,
    min_frames: int,
) -> float:
    tof = np.asarray(tof_observed_s, dtype=np.float64)
    path_lengths = np.asarray(path_lengths_m, dtype=np.float64)
    accepted_mask = np.asarray(accepted, dtype=bool)
    expected_shape = tof.shape
    for name, values in (
        ("path_lengths_m", path_lengths),
        ("temperature_c", np.asarray(temperature_c)),
        ("pressure_mpa", np.asarray(pressure_mpa)),
        ("humidity_rh", np.asarray(humidity_rh)),
        ("accepted", accepted_mask),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{name} shape mismatch: {values.shape} != {expected_shape}")
    if len(phase_ids) != tof.size:
        raise ValueError(f"phase length mismatch: {len(phase_ids)} != {tof.size}")
    baseline = np.asarray([phase == "baseline" for phase in phase_ids], dtype=bool)
    calibration_mask = baseline & accepted_mask
    if int(np.sum(calibration_mask)) < min_frames:
        raise ValueError(
            f"baseline delay calibration requires at least {min_frames} accepted frames, "
            f"got {int(np.sum(calibration_mask))}"
        )
    fresh_speed = fresh_air_sound_speed_m_per_s(temperature_c, pressure_mpa, humidity_rh)
    candidates = tof[calibration_mask] - path_lengths[calibration_mask] / fresh_speed[calibration_mask]
    if not np.isfinite(candidates).all():
        raise ValueError("delay calibration candidates contain non-finite values")
    return float(np.median(candidates))


def fit_tof_vs_path_length(
    tof_observed_s: np.ndarray,
    path_lengths_m: np.ndarray,
    phase_ids: Sequence[str],
    accepted: np.ndarray,
) -> tuple[float, float]:
    tof = np.asarray(tof_observed_s, dtype=np.float64)
    path_lengths = np.asarray(path_lengths_m, dtype=np.float64)
    accepted_mask = np.asarray(accepted, dtype=bool)
    if tof.shape != path_lengths.shape or tof.shape != accepted_mask.shape:
        raise ValueError("TOF, path length, and accepted arrays must have identical shapes")
    if len(phase_ids) != tof.size:
        raise ValueError(f"phase length mismatch: {len(phase_ids)} != {tof.size}")
    steady = np.asarray([phase == "steady" for phase in phase_ids], dtype=bool) & accepted_mask
    if int(np.sum(steady)) < 2 or np.unique(path_lengths[steady]).size < 2:
        raise ValueError("tof vs L_m fit requires at least two accepted steady path lengths")
    slowness, intercept, _low_slope, _high_slope = theilslopes(tof[steady], path_lengths[steady])
    if not math.isfinite(slowness) or slowness <= 0.0:
        raise ValueError(f"tof vs L_m fit produced invalid slowness {slowness}")
    return float(intercept), float(1.0 / slowness)


def _validate_template(template: np.ndarray) -> np.ndarray:
    values = np.asarray(template, dtype=np.float32)
    if values.ndim != 1 or values.size < 3:
        raise ValueError(f"template must be 1D with at least 3 samples, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("template contains non-finite values")
    if float(np.max(np.abs(values))) <= 0.0:
        raise ValueError("template must have non-zero amplitude")
    return values


def _slow_channel_indices(slow_channel_names: Sequence[str]) -> dict[str, int]:
    lookup = {str(name): index for index, name in enumerate(slow_channel_names)}
    missing = [name for name in REQUIRED_SLOW_CHANNELS if name not in lookup]
    if missing:
        raise ValueError(f"missing required slow channels: {missing}; available={list(lookup)}")
    return {name: lookup[name] for name in REQUIRED_SLOW_CHANNELS}


def _noise_samples(
    waveform: np.ndarray,
    peak_lower: int,
    peak_upper: int,
    pre_samples: int,
    post_samples: int,
) -> np.ndarray:
    noise_stop_left = max(0, peak_lower - pre_samples)
    noise_start_right = min(waveform.size, peak_upper + post_samples + 1)
    return np.concatenate((waveform[:noise_stop_left], waveform[noise_start_right:]))


def _snr_db(signal_peak_abs: float, noise: np.ndarray, cap: float) -> float:
    if signal_peak_abs <= 0.0:
        return -cap
    if noise.size == 0:
        raise ValueError("SNR requires noise samples outside the physical search window")
    centered = noise - float(np.median(noise))
    noise_std = 1.4826 * float(np.median(np.abs(centered)))
    if noise_std == 0.0:
        return float(cap)
    return float(np.clip(20.0 * math.log10(signal_peak_abs / noise_std), -cap, cap))


def _peak_to_sidelobe_ratio(
    correlation: np.ndarray,
    peak_index: int,
    *,
    exclusion: int,
    cap: float,
) -> float:
    mask = np.ones(correlation.size, dtype=bool)
    mask[max(0, peak_index - exclusion) : min(correlation.size, peak_index + exclusion + 1)] = False
    finite_sidelobes = np.abs(correlation[mask & np.isfinite(correlation)])
    if finite_sidelobes.size == 0:
        return float(cap)
    sidelobe = float(np.max(finite_sidelobes))
    if sidelobe == 0.0:
        return float(cap)
    return float(min(abs(float(correlation[peak_index])) / sidelobe, cap))


def _phase_locked_peak_index(correlation: np.ndarray, config: RawDSPConfig) -> int:
    finite = np.isfinite(correlation)
    if not finite.any():
        raise ValueError("correlation contains no finite values")
    envelope_input = np.where(finite, correlation, 0.0)
    envelope = np.abs(hilbert(envelope_input))
    envelope_peak = int(np.argmax(envelope))
    half_cycle_samples = config.sample_rate_hz / (2.0 * config.carrier_frequency_hz)
    radius = max(1, int(math.floor(half_cycle_samples)))
    lower = max(0, envelope_peak - radius)
    upper = min(correlation.size, envelope_peak + radius + 1)
    return lower + int(np.argmax(correlation[lower:upper]))


def _coarse_reference_peak(
    waveform_window: np.ndarray,
    config: RawDSPConfig,
    reference_peak_polarity: int,
) -> int:
    envelope = np.abs(hilbert(waveform_window))
    envelope_peak = int(np.argmax(envelope))
    half_cycle_samples = config.sample_rate_hz / (2.0 * config.carrier_frequency_hz)
    radius = max(1, int(math.ceil(half_cycle_samples)))
    lower = max(0, envelope_peak - radius)
    upper = min(waveform_window.size, envelope_peak + radius + 1)
    local = waveform_window[lower:upper]
    if reference_peak_polarity == -1:
        return lower + int(np.argmin(local))
    return lower + int(np.argmax(local))


def _peak_width_samples(correlation: np.ndarray, peak_index: int) -> float:
    peak = float(correlation[peak_index])
    if peak <= 0.0:
        return float(correlation.size)
    threshold = 0.5 * peak
    left = peak_index
    while left > 0 and correlation[left - 1] >= threshold:
        left -= 1
    right = peak_index
    while right + 1 < correlation.size and correlation[right + 1] >= threshold:
        right += 1
    return float(right - left + 1)


def _quality_score(
    *,
    corr_peak: float,
    peak_to_sidelobe_ratio: float,
    snr_db: float,
    clipped: bool,
    boundary_hit: bool,
    config: RawDSPConfig,
) -> float:
    corr_score = float(np.clip(corr_peak, 0.0, 1.0))
    psr_score = float(np.clip(peak_to_sidelobe_ratio / config.min_peak_to_sidelobe_ratio, 0.0, 1.0))
    snr_linear = 10.0 ** (float(np.clip(snr_db, -120.0, 120.0)) / 20.0)
    snr_score = snr_linear / (snr_linear + 10.0)
    quality = corr_score * psr_score * snr_score
    if clipped:
        quality *= 0.5
    if boundary_hit:
        quality *= 0.5
    return float(np.clip(quality, 0.0, 1.0))
