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
from sim.generation.phases import blend_for_timestep, phase_boundaries, phase_for_timestep
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
    optical_absorption_backend: str = EMPIRICAL_ABSORPTION_BACKEND,
    hitran_cache_root: str = "data/hitran_cache",
) -> dict[str, object]:
    if optical_absorption_backend not in VALID_OPTICAL_ABSORPTION_BACKENDS:
        raise ValueError(f"optical_absorption_backend must be one of {list(VALID_OPTICAL_ABSORPTION_BACKENDS)}, got {optical_absorption_backend!r}")
    sequence_count = len(conditions)
    slow = np.zeros((sequence_count, timesteps, len(SLOW_CHANNELS)), dtype=np.float32)
    ultrasonic = np.zeros((sequence_count, timesteps, ultrasonic_spec.waveform_samples), dtype=np.int16)
    ultrasonic_scale = np.zeros((sequence_count, timesteps), dtype=np.float32)
    fiber_mic = np.zeros((sequence_count, timesteps, fiber_mic_spec.waveform_samples), dtype=np.int16)
    fiber_mic_scale = np.zeros((sequence_count, timesteps), dtype=np.float32)
    slow_rows = []
    q1, q2, q3 = phase_boundaries(timesteps)
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
        ndir_state: dict[str, float] = {}
        for timestep in range(timesteps):
            phase_id = phase_for_timestep(timestep, timesteps)
            blend = blend_for_timestep(timestep, timesteps)
            current_l_m = _path_l_m_for_timestep(
                float(condition["L_m_base"]),
                timestep,
                q1,
                q2,
                q3,
                is_baseline_scan,
                is_steady_scan,
                path_lms,
            )
            composition = _blend_composition(condition, blend)
            if optical_absorption_backend == EMPIRICAL_ABSORPTION_BACKEND:
                current = _dynamic_slow_features(baseline_main, target_main, timestep, timesteps, slow_params, slow_walk, sequence_rng)
            else:
                current = _dynamic_slow_features(
                    baseline_main,
                    target_main,
                    timestep,
                    timesteps,
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
            fiber_mic[seq_index, timestep, :] = fiber_result["waveform_int16"]
            fiber_mic_scale[seq_index, timestep] = fiber_result["scale_factor"]
            slow_rows.append(_slow_row(condition["sequence_id"], timestep, dt_s, phase_id, current))
    return {
        "slow": slow,
        "ultrasonic": ultrasonic,
        "ultrasonic_scale": ultrasonic_scale,
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
        tau_key = "tau_rise_system_s" if target >= previous else "tau_decay_system_s"
        response = 1.0 - math.exp(-1.0 / slow_params[channel][tau_key])
        value = previous + (target - previous) * response
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
            "noise_sigma": base_scale * NOISE_FRACTION[channel],
            "random_walk_sigma": base_scale * NOISE_FRACTION[channel] * 0.08,
            "drift_slope": rng.uniform(-1.0, 1.0) * base_scale * NOISE_FRACTION[channel] * 0.015,
        }
    return params


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


def _path_l_m_for_timestep(
    l_m_base: float,
    timestep: int,
    q1: int,
    q2: int,
    q3: int,
    is_baseline_scan: bool,
    is_steady_scan: bool,
    path_lms: tuple[float, ...],
) -> float:
    if is_baseline_scan and timestep < q1:
        return float(path_lms[_scan_path_index(timestep, q1, len(path_lms))])
    if is_steady_scan and q2 <= timestep < q3:
        local = timestep - q2
        span = max(1, q3 - q2)
        return float(path_lms[_scan_path_index(local, span, len(path_lms))])
    return float(l_m_base)


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
