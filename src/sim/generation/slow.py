from __future__ import annotations

import math
import random

import numpy as np

from sim.core.schema import SLOW_CHANNELS, SLOW_DYNAMIC_CHANNELS
from sim.generation.acoustic_physics import main_sensor_features
from sim.generation.phases import blend_for_timestep, phase_boundaries, phase_for_timestep
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
) -> dict[str, object]:
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

    root_rng = random.Random(seed)
    for seq_index, condition in enumerate(conditions):
        condition_rng = random.Random(root_rng.randrange(0, 2**32))
        sequence_rng = random.Random(root_rng.randrange(0, 2**32))
        baseline_main = main_sensor_features(_main_feature_condition(condition, 0.0, 0.0, 0.0, 100.0, float(condition["L_m_base"])), condition_rng)
        target_main = main_sensor_features(
            _main_feature_condition(
                condition,
                float(condition["x_H2"]),
                float(condition["x_CH4"]),
                float(condition["x_CO2"]),
                float(condition["x_N2"]),
                float(condition["L_m_base"]),
            ),
            condition_rng,
        )
        slow_params = _channel_dynamic_params(sequence_rng)
        slow_walk = {channel: 0.0 for channel in SLOW_DYNAMIC_CHANNELS}
        for timestep in range(timesteps):
            phase_id = phase_for_timestep(timestep, timesteps)
            blend = blend_for_timestep(timestep, timesteps)
            current = _dynamic_slow_features(baseline_main, target_main, timestep, timesteps, slow_params, slow_walk, sequence_rng)
            current["T_C"] = float(condition["T_C_base"])
            current["P_MPa"] = float(condition["P_MPa_base"])
            current["H_RH"] = float(condition["H_RH_base"])
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
            current["L_m"] = current_l_m
            current["piston_position_m"] = current_l_m
            slow_values = [float(current[channel]) for channel in SLOW_CHANNELS]
            slow[seq_index, timestep, :] = np.array(slow_values, dtype=np.float32)
            composition = _blend_composition(condition, blend)
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
) -> dict[str, float]:
    current = {}
    for channel in SLOW_DYNAMIC_CHANNELS:
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
        return float(path_lms[min(len(path_lms) - 1, timestep // max(1, q1 // len(path_lms)))])
    if is_steady_scan and q2 <= timestep < q3:
        local = timestep - q2
        span = max(1, q3 - q2)
        return float(path_lms[min(len(path_lms) - 1, local // max(1, span // len(path_lms)))])
    return float(l_m_base)


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
