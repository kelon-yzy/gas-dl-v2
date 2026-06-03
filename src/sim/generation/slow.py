from __future__ import annotations

import math
import random

import numpy as np

from sim.core.schema import SLOW_CHANNELS, SLOW_DYNAMIC_CHANNELS
from sim.generation.acoustic_physics import PROCESSING_PARAMS, main_sensor_features, thermal_conductivity_sensor_feature
from sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    HITRAN_ABSORPTION_BACKEND,
    VALID_OPTICAL_ABSORPTION_BACKENDS,
    compute_hitran_optical_absorption,
)
from sim.generation.phases import PhaseSchedule, phase_boundaries, resolve_phase_schedule
from sim.generation.spectral import HitranGridSpec, PreparedTabulatedSpectra
from sim.generation.waveforms import FiberMicSpec, WaveformSpec, simulate_fiber_mic_measurement, simulate_waveform_measurement


TAU_RISE_SYSTEM_S = {
    "V_NDIR_CH4": (8.0, 20.0),
    "V_NDIR_CO2": (6.0, 18.0),
    "V_TCS": (10.0, 35.0),
}
TAU_DECAY_SYSTEM_S = {
    "V_NDIR_CH4": (12.0, 30.0),
    "V_NDIR_CO2": (10.0, 28.0),
    "V_TCS": (20.0, 60.0),
}
NOISE_FRACTION = {"V_NDIR_CH4": 0.0025, "V_NDIR_CO2": 0.0025, "V_TCS": 0.003}


def build_sequence_arrays(
    conditions: list[dict[str, str]],
    *,
    timesteps: int,
    dt_s: float,
    seed: int,
    multi_path_phase: str,
    ultrasonic_spec: WaveformSpec,
    fiber_mic_spec: FiberMicSpec,
    path_lms: tuple[float, ...],
    phase_schedule: str | PhaseSchedule = "standard_exposure",
    stage_jitter: float = 0.0,
    optical_absorption_backend: str = EMPIRICAL_ABSORPTION_BACKEND,
    hitran_cache_root: str = "data/hitran_cache",
) -> dict[str, object]:
    if optical_absorption_backend not in VALID_OPTICAL_ABSORPTION_BACKENDS:
        raise ValueError(f"optical_absorption_backend must be one of {list(VALID_OPTICAL_ABSORPTION_BACKENDS)}, got {optical_absorption_backend!r}")
    sequence_count = len(conditions)
    slow = np.zeros((sequence_count, timesteps, len(SLOW_CHANNELS)), dtype=np.float32)
    ultrasonic = np.zeros((sequence_count, timesteps, ultrasonic_spec.waveform_samples), dtype=np.int16)
    ultrasonic_scale = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_s = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_observed_s = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_peak_index = np.zeros((sequence_count, timesteps), dtype=np.int32)
    ultrasonic_sound_speed = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_sound_speed_estimated = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_alpha = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_quality = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_accepted = np.zeros((sequence_count, timesteps), dtype=np.int8)
    fiber_mic = np.zeros((sequence_count, timesteps, fiber_mic_spec.waveform_samples), dtype=np.int16)
    fiber_mic_scale = np.zeros((sequence_count, timesteps), dtype=np.float32)
    slow_rows = []
    base_schedule = resolve_phase_schedule(phase_schedule)
    # 仅 empirical 后端在 standard_exposure + 无 jitter 时复现旧 wv4-smoke 单时间常数动力学；
    # HITRAN 后端的所有慢通道（含 V_TCS）始终走 equilibrium 多时间常数动力学。
    is_empirical = optical_absorption_backend == EMPIRICAL_ABSORPTION_BACKEND
    use_legacy_empirical_dynamics = is_empirical and base_schedule.name == "standard_exposure" and stage_jitter == 0.0
    is_baseline_scan = multi_path_phase == "baseline"
    is_steady_scan = multi_path_phase == "steady"
    spectra_cache: dict[tuple[str, HitranGridSpec], PreparedTabulatedSpectra] | None = (
        {} if optical_absorption_backend == HITRAN_ABSORPTION_BACKEND else None
    )

    root_rng = random.Random(seed)
    for seq_index, condition in enumerate(conditions):
        condition_rng = random.Random(root_rng.randrange(0, 2**32))
        sequence_rng = random.Random(root_rng.randrange(0, 2**32))
        baseline_condition = _main_feature_condition(condition, 0.0, 0.0, 0.0, 100.0, float(condition["L_m_base"]))
        target_condition = _main_feature_condition(
            condition,
            float(condition["x_H2"]),
            float(condition["x_CH4"]),
            float(condition["x_CO2"]),
            float(condition["x_N2"]),
            float(condition["L_m_base"]),
        )
        if optical_absorption_backend == EMPIRICAL_ABSORPTION_BACKEND:
            baseline_main = main_sensor_features(baseline_condition, condition_rng)
            target_main = main_sensor_features(target_condition, condition_rng)
        else:
            baseline_main = thermal_conductivity_sensor_feature(baseline_condition, condition_rng)
            target_main = thermal_conductivity_sensor_feature(target_condition, condition_rng)
        slow_params = _channel_dynamic_params(sequence_rng)
        slow_walk = {channel: 0.0 for channel in SLOW_DYNAMIC_CHANNELS}
        schedule = base_schedule.jittered(sequence_rng, stage_jitter)
        phase_intervals = _phase_intervals(schedule, timesteps)
        phase_ids, blends = schedule.resolve_timeline(timesteps)
        ndir_state: dict[str, float] = {}
        slow_state: dict[str, float] = {}
        for timestep in range(timesteps):
            phase_id = phase_ids[timestep]
            blend = blends[timestep]
            current_l_m = _path_l_m_for_schedule(
                float(condition["L_m_base"]),
                timestep,
                phase_intervals,
                is_baseline_scan,
                is_steady_scan,
                path_lms,
            )
            composition = _blend_composition(condition, blend)
            if is_empirical:
                if use_legacy_empirical_dynamics:
                    current = _dynamic_slow_features(baseline_main, target_main, timestep, timesteps, slow_params, slow_walk, sequence_rng)
                else:
                    current = _dynamic_features_from_equilibrium(
                        _blend_equilibrium_features(baseline_main, target_main, blend, channels=SLOW_DYNAMIC_CHANNELS),
                        slow_state,
                        timestep,
                        slow_params,
                        slow_walk,
                        sequence_rng,
                        channels=SLOW_DYNAMIC_CHANNELS,
                    )
            else:
                current = _dynamic_features_from_equilibrium(
                    _blend_equilibrium_features(baseline_main, target_main, blend, channels=("V_TCS",)),
                    slow_state,
                    timestep,
                    slow_params,
                    slow_walk,
                    sequence_rng,
                    channels=("V_TCS",),
                )
                ndir_equilibrium = _hitran_ndir_equilibrium(
                    condition,
                    composition=composition,
                    l_m=current_l_m,
                    hitran_cache_root=hitran_cache_root,
                    spectra_cache=spectra_cache,
                )
                current.update(
                    _dynamic_features_from_equilibrium(
                        ndir_equilibrium,
                        ndir_state,
                        timestep,
                        slow_params,
                        slow_walk,
                        sequence_rng,
                        channels=("V_NDIR_CH4", "V_NDIR_CO2"),
                    )
                )
            current["T_C"] = float(condition["T_C_base"])
            current["P_MPa"] = float(condition["P_MPa_base"])
            current["H_RH"] = float(condition["H_RH_base"])
            current["L_m"] = current_l_m
            current["piston_position_m"] = current_l_m
            slow_values = [float(current[channel]) for channel in SLOW_CHANNELS]
            slow[seq_index, timestep, :] = np.array(slow_values, dtype=np.float32)
            ultrasonic_result = simulate_waveform_measurement(
                **composition,
                t_c=float(current["T_C"]),
                p_mpa=float(current["P_MPa"]),
                h_rh=float(current["H_RH"]),
                l_m=float(current["L_m"]),
                seed=sequence_rng.randrange(0, 2**32),
                spec=ultrasonic_spec,
            )
            fiber_result = simulate_fiber_mic_measurement(
                **composition,
                t_c=float(current["T_C"]),
                p_mpa=float(current["P_MPa"]),
                h_rh=float(current["H_RH"]),
                l_m=float(current["L_m"]),
                seed=sequence_rng.randrange(0, 2**32),
                spec=fiber_mic_spec,
            )
            ultrasonic[seq_index, timestep, :] = ultrasonic_result["waveform_int16"]
            ultrasonic_scale[seq_index, timestep] = ultrasonic_result["scale_factor"]
            ultrasonic_tof_s[seq_index, timestep] = float(ultrasonic_result["tof_s"])
            ultrasonic_tof_observed_s[seq_index, timestep] = float(ultrasonic_result["tof_observed_s"])
            ultrasonic_peak_index[seq_index, timestep] = int(ultrasonic_result["peak_index"])
            ultrasonic_sound_speed[seq_index, timestep] = float(ultrasonic_result["sound_speed_m_per_s"])
            ultrasonic_sound_speed_estimated[seq_index, timestep] = float(ultrasonic_result["sound_speed_estimated_m_per_s"])
            ultrasonic_alpha[seq_index, timestep] = float(ultrasonic_result["alpha_true_npm"])
            ultrasonic_tof_quality[seq_index, timestep] = float(ultrasonic_result["tof_quality"])
            ultrasonic_tof_accepted[seq_index, timestep] = int(ultrasonic_result["tof_accepted"])
            fiber_mic[seq_index, timestep, :] = fiber_result["waveform_int16"]
            fiber_mic_scale[seq_index, timestep] = fiber_result["scale_factor"]
            slow_rows.append(_slow_row(condition["sequence_id"], timestep, dt_s, phase_id, current))
    return {
        "slow": slow,
        "ultrasonic": ultrasonic,
        "ultrasonic_scale": ultrasonic_scale,
        "ultrasonic_tof_s": ultrasonic_tof_s,
        "ultrasonic_tof_observed_s": ultrasonic_tof_observed_s,
        "ultrasonic_peak_index": ultrasonic_peak_index,
        "ultrasonic_sound_speed_m_per_s": ultrasonic_sound_speed,
        "ultrasonic_sound_speed_estimated_m_per_s": ultrasonic_sound_speed_estimated,
        "ultrasonic_alpha_true_npm": ultrasonic_alpha,
        "ultrasonic_tof_quality": ultrasonic_tof_quality,
        "ultrasonic_tof_accepted": ultrasonic_tof_accepted,
        "fiber_mic": fiber_mic,
        "fiber_mic_scale": fiber_mic_scale,
        "slow_rows": slow_rows,
    }


def _main_feature_condition(condition: dict[str, str], x_h2: float, x_ch4: float, x_co2: float, x_n2: float, l_m: float) -> dict[str, str]:
    return {
        "x_H2": _fmt(x_h2, 6),
        "x_CH4": _fmt(x_ch4, 6),
        "x_CO2": _fmt(x_co2, 6),
        "x_N2": _fmt(x_n2, 6),
        "T_C": condition["T_C_base"],
        "P_MPa": condition["P_MPa_base"],
        "H_RH": condition["H_RH_base"],
        "L_m": _fmt(l_m, 6),
    }


def _blend_composition(condition: dict[str, str], blend: float) -> dict[str, float]:
    return {
        "x_h2": float(condition["x_H2"]) * blend,
        "x_ch4": float(condition["x_CH4"]) * blend,
        "x_co2": float(condition["x_CO2"]) * blend,
        "x_n2": 100.0 + (float(condition["x_N2"]) - 100.0) * blend,
    }


def _blend_equilibrium_features(
    baseline_main: dict[str, float],
    target_main: dict[str, float],
    blend: float,
    *,
    channels: tuple[str, ...],
) -> dict[str, float]:
    return {
        channel: float(baseline_main[channel]) + (float(target_main[channel]) - float(baseline_main[channel])) * blend
        for channel in channels
    }


def _dynamic_slow_features(
    baseline_main: dict[str, float],
    target_main: dict[str, float],
    timestep: int,
    timesteps: int,
    slow_params: dict[str, dict[str, float]],
    slow_walk: dict[str, float],
    sequence_rng: random.Random,
    channels: tuple[str, ...] = SLOW_DYNAMIC_CHANNELS,
) -> dict[str, float]:
    current = {}
    for channel in channels:
        value = _channel_value(
            baseline=float(baseline_main[channel]),
            target=float(target_main[channel]),
            timestep=timestep,
            timesteps=timesteps,
            tau_rise_system_s=slow_params[channel]["tau_rise_system_s"],
            tau_decay_system_s=slow_params[channel]["tau_decay_system_s"],
        )
        slow_walk[channel] += sequence_rng.gauss(0.0, slow_params[channel]["random_walk_sigma"])
        value += slow_params[channel]["drift_slope"] * timestep
        value += slow_walk[channel]
        value += sequence_rng.gauss(0.0, slow_params[channel]["noise_sigma"])
        current[channel] = max(1e-9, value)
    return current


def _hitran_ndir_equilibrium(
    condition: dict[str, str],
    *,
    composition: dict[str, float],
    l_m: float,
    hitran_cache_root: str,
    spectra_cache: dict[tuple[str, HitranGridSpec], PreparedTabulatedSpectra] | None,
) -> dict[str, float]:
    optical = compute_hitran_optical_absorption(
        _main_feature_condition(
            condition,
            composition["x_h2"],
            composition["x_ch4"],
            composition["x_co2"],
            composition["x_n2"],
            l_m,
        ),
        cache_root=hitran_cache_root,
        spectra_cache=spectra_cache,
    )
    return {
        "V_NDIR_CH4": max(
            0.1,
            PROCESSING_PARAMS["optical_baseline_ch4_init"] * math.exp(-float(optical["absorption_ch4_observed"])),
        ),
        "V_NDIR_CO2": max(
            0.1,
            PROCESSING_PARAMS["optical_baseline_co2_init"] * math.exp(-float(optical["absorption_co2_observed"])),
        ),
    }


def _dynamic_features_from_equilibrium(
    equilibrium: dict[str, float],
    state: dict[str, float],
    timestep: int,
    slow_params: dict[str, dict[str, float]],
    slow_walk: dict[str, float],
    sequence_rng: random.Random,
    *,
    channels: tuple[str, ...],
) -> dict[str, float]:
    current = {}
    for channel in channels:
        target = float(equilibrium[channel])
        previous = state.get(channel, target)
        value = _multi_tau_channel_step(
            previous=previous,
            target=target,
            params=slow_params[channel],
        )
        slow_walk[channel] += sequence_rng.gauss(0.0, slow_params[channel]["random_walk_sigma"])
        value += slow_params[channel]["drift_slope"] * timestep
        value += slow_walk[channel]
        value += sequence_rng.gauss(0.0, slow_params[channel]["noise_sigma"])
        value = max(1e-9, value)
        state[channel] = value
        current[channel] = value
    return current


def _channel_dynamic_params(rng: random.Random) -> dict[str, dict[str, float]]:
    params = {}
    for channel in SLOW_DYNAMIC_CHANNELS:
        rise_min, rise_max = TAU_RISE_SYSTEM_S[channel]
        decay_min, decay_max = TAU_DECAY_SYSTEM_S[channel]
        base_scale = {"V_NDIR_CH4": 2.5, "V_NDIR_CO2": 2.5, "V_TCS": 1.5}[channel]
        params[channel] = {
            "tau_rise_system_s": rng.uniform(rise_min, rise_max),
            "tau_decay_system_s": rng.uniform(decay_min, decay_max),
            "fast_tau_fraction": rng.uniform(0.25, 0.45),
            "slow_tau_multiplier": rng.uniform(2.5, 4.5),
            "fast_response_weight": rng.uniform(0.55, 0.75),
            "recovery_floor_fraction": rng.uniform(0.02, 0.08),
            "noise_sigma": base_scale * NOISE_FRACTION[channel],
            "random_walk_sigma": base_scale * NOISE_FRACTION[channel] * 0.08,
            "drift_slope": rng.uniform(-1.0, 1.0) * base_scale * NOISE_FRACTION[channel] * 0.015,
        }
    return params


def _multi_tau_channel_step(previous: float, target: float, params: dict[str, float]) -> float:
    tau_key = "tau_rise_system_s" if target >= previous else "tau_decay_system_s"
    base_tau = params[tau_key]
    fast_tau = max(1e-6, base_tau * params["fast_tau_fraction"])
    slow_tau = max(1e-6, base_tau * params["slow_tau_multiplier"])
    fast_alpha = 1.0 - math.exp(-1.0 / fast_tau)
    slow_alpha = 1.0 - math.exp(-1.0 / slow_tau)
    weight = params["fast_response_weight"]
    alpha = weight * fast_alpha + (1.0 - weight) * slow_alpha
    if target < previous:
        target = target + (previous - target) * params["recovery_floor_fraction"]
    return previous + (target - previous) * alpha


def _channel_value(baseline: float, target: float, timestep: int, timesteps: int, tau_rise_system_s: float, tau_decay_system_s: float) -> float:
    q1, _, q3 = phase_boundaries(timesteps)
    if timestep < q1:
        return baseline
    if timestep < q3:
        progress = 1.0 - math.exp(-(timestep - q1 + 1) / tau_rise_system_s)
        return baseline + (target - baseline) * progress
    start_progress = 1.0 - math.exp(-(q3 - q1) / tau_rise_system_s)
    recovery_start = baseline + (target - baseline) * start_progress
    recovery_progress = math.exp(-(timestep - q3 + 1) / tau_decay_system_s)
    return baseline + (recovery_start - baseline) * recovery_progress


def _path_l_m_for_schedule(
    l_m_base: float,
    timestep: int,
    phase_intervals: tuple[tuple[str, int, int], ...],
    is_baseline_scan: bool,
    is_steady_scan: bool,
    path_lms: tuple[float, ...],
) -> float:
    for phase_id, start, end in phase_intervals:
        if start <= timestep < end:
            if is_baseline_scan and phase_id == "baseline":
                return float(path_lms[_scan_path_index(timestep - start, end - start, len(path_lms))])
            if is_steady_scan and phase_id == "steady":
                return float(path_lms[_scan_path_index(timestep - start, end - start, len(path_lms))])
            return float(l_m_base)
    raise ValueError(f"timestep {timestep} is outside phase schedule")


def _phase_intervals(schedule: PhaseSchedule, timesteps: int) -> tuple[tuple[str, int, int], ...]:
    bounds = schedule.boundaries(timesteps)
    starts = (0, *bounds)
    ends = (*bounds, timesteps)
    return tuple(
        (segment.name, start, end)
        for segment, start, end in zip(schedule.segments, starts, ends, strict=True)
    )


def _scan_path_index(local_timestep: int, span: int, option_count: int) -> int:
    if option_count <= 1 or span <= 1:
        return 0
    index = round(local_timestep * (option_count - 1) / (span - 1))
    return min(option_count - 1, max(0, int(index)))


def _slow_row(sequence_id: str, timestep: int, dt_s: float, phase_id: str, current: dict[str, float]) -> dict[str, str]:
    return {
        "sequence_id": sequence_id,
        "timestep": str(timestep),
        "timestamp_s": _fmt(timestep * dt_s, 1),
        "phase_id": phase_id,
        "V_NDIR_CH4": _fmt(float(current["V_NDIR_CH4"]), 6),
        "V_NDIR_CO2": _fmt(float(current["V_NDIR_CO2"]), 6),
        "V_TCS": _fmt(float(current["V_TCS"]), 6),
        "T_C": _fmt(float(current["T_C"]), 4),
        "P_MPa": _fmt(float(current["P_MPa"]), 5),
        "H_RH": _fmt(float(current["H_RH"]), 4),
        "L_m": _fmt(float(current["L_m"]), 5),
        "piston_position_m": _fmt(float(current["piston_position_m"]), 5),
    }


def _fmt(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"
