"""掘进通风场景的慢通道序列生成。

与 hydrogen_ng `sim.generation.slow` 和 syngas `sim.generation.syngas.slow` 并存。差异：
- SLOW_CHANNELS / SLOW_DYNAMIC_CHANNELS 来自 tunnel_ventilation_schema（7 通道）
- 物理函数走 sim.generation.tunnel_ventilation.acoustic_physics
- 波形仿真通过 sound_speed_fn / attenuation_fn 注入 tv3 物理，extra_gas_kwargs 透传 x_o2
- 仅支持 empirical_v1 后端（HITRAN 后端阶段 1 不实现，调用即拒绝）
- 无光学串扰矩阵（只有 CO2 一个红外活性组分）
- 无 V_NDIR_CH4 通道（场景无 CH₄）

baseline 语义：blend=0 时标准新鲜空气，blend=1 时实际 CO2/O2/N2 组分。

内存优化：当 temp_dir 传入时，大波形数组（ultrasonic / fiber_mic）改用 memmap 直接落盘，
避免大序列数场景下 np.zeros 申请几十 GiB RAM 导致 OOM / SIGBUS。
"""
from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import numpy as np

from sim.core.tunnel_ventilation_schema import SLOW_CHANNELS, SLOW_DYNAMIC_CHANNELS
from sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    VALID_OPTICAL_ABSORPTION_BACKENDS,
)
from sim.generation.phases import PhaseSchedule, resolve_phase_schedule
from sim.generation.tunnel_ventilation.acoustic_physics import (
    PROCESSING_PARAMS,
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
    main_sensor_features,
    thermal_conductivity_sensor_feature,
)
from sim.generation.waveforms import (
    FiberMicSpec,
    WaveformSpec,
    simulate_fiber_mic_measurement,
    simulate_waveform_measurement,
)


# 系统响应时间常数（与 hydrogen_ng 同量级）
TAU_RISE_SYSTEM_S = {
    "V_NDIR_CO2": (6.0, 18.0),
    "V_TCS": (10.0, 35.0),
}
TAU_DECAY_SYSTEM_S = {
    "V_NDIR_CO2": (10.0, 28.0),
    "V_TCS": (20.0, 60.0),
}
NOISE_FRACTION = {
    "V_NDIR_CO2": 0.0025,
    "V_TCS": 0.003,
}
_BASE_SCALE = {
    "V_NDIR_CO2": 2.5,
    "V_TCS": 1.5,
}
FRESH_AIR_COMPOSITION = {
    "x_co2": 0.04,
    "x_o2": 20.90,
    "x_n2": 79.06,
}

# tv3 阶段 1 仅支持 empirical 后端；HITRAN 后端待后续阶段实现
_TV3_VALID_BACKENDS = (EMPIRICAL_ABSORPTION_BACKEND,)


def build_sequence_arrays(
    conditions: list[dict[str, str]],
    *,
    timesteps: int,
    dt_s: float,
    seed: int,
    multi_path_phase: str,
    ultrasonic_spec: WaveformSpec,
    fiber_mic_spec: FiberMicSpec | None,
    path_lms: tuple[float, ...],
    phase_schedule: str | PhaseSchedule = "standard_exposure",
    stage_jitter: float = 0.0,
    optical_absorption_backend: str = EMPIRICAL_ABSORPTION_BACKEND,
    hitran_cache_root: str = "data/hitran_cache_tv3",  # noqa: ARG001 保留参数为兼容接口签名
    start_sequence_index: int = 0,
    temp_dir: Path | None = None,  # 大波形数组 memmap 落盘目录；None 时沿用 np.zeros
) -> dict[str, object]:
    if optical_absorption_backend not in VALID_OPTICAL_ABSORPTION_BACKENDS:
        raise ValueError(
            f"optical_absorption_backend must be one of {list(VALID_OPTICAL_ABSORPTION_BACKENDS)}, "
            f"got {optical_absorption_backend!r}"
        )
    if optical_absorption_backend not in _TV3_VALID_BACKENDS:
        raise ValueError(
            f"tunnel_ventilation 阶段 1 仅支持 empirical_v1 后端，"
            f"got {optical_absorption_backend!r}（HITRAN 后端待后续阶段实现）"
        )

    sequence_count = len(conditions)
    use_memmap = temp_dir is not None
    if use_memmap:
        temp_dir.mkdir(parents=True, exist_ok=True)

    slow = np.zeros((sequence_count, timesteps, len(SLOW_CHANNELS)), dtype=np.float32)
    # 大波形数组：6000×512×5000 int16 ≈ 28.6 GiB，必须 memmap 落盘避免 OOM
    if use_memmap:
        ultrasonic = np.lib.format.open_memmap(
            str(temp_dir / "ultrasonic.npy"), mode="w+",
            dtype=np.dtype(ultrasonic_spec.waveform_dtype),
            shape=(sequence_count, timesteps, ultrasonic_spec.waveform_samples),
        )
    else:
        ultrasonic = np.zeros(
            (sequence_count, timesteps, ultrasonic_spec.waveform_samples),
            dtype=np.dtype(ultrasonic_spec.waveform_dtype),
        )
    ultrasonic_scale = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_s = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_observed_s = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_peak_index = np.zeros((sequence_count, timesteps), dtype=np.int32)
    ultrasonic_sound_speed = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_sound_speed_estimated = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_alpha = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_quality = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_accepted = np.zeros((sequence_count, timesteps), dtype=np.int8)
    if fiber_mic_spec is not None:
        if use_memmap:
            fiber_mic = np.lib.format.open_memmap(
                str(temp_dir / "fiber_mic.npy"), mode="w+",
                dtype=np.dtype(fiber_mic_spec.waveform_dtype),
                shape=(sequence_count, timesteps, fiber_mic_spec.waveform_samples),
            )
        else:
            fiber_mic = np.zeros(
                (sequence_count, timesteps, fiber_mic_spec.waveform_samples),
                dtype=np.dtype(fiber_mic_spec.waveform_dtype),
            )
        fiber_mic_scale = np.zeros((sequence_count, timesteps), dtype=np.float32)
    slow_rows = []
    base_schedule = resolve_phase_schedule(phase_schedule)
    is_baseline_scan = multi_path_phase == "baseline"
    is_steady_scan = multi_path_phase == "steady"

    for seq_index, condition in enumerate(conditions):
        global_sequence_index = start_sequence_index + seq_index
        condition_rng = random.Random(_stable_uint32(seed, global_sequence_index, "condition"))
        sequence_rng = random.Random(_stable_uint32(seed, global_sequence_index, "sequence"))
        # baseline = 标准新鲜空气，target = 实际 CO2/O2/N2 组分
        baseline_condition = _main_feature_condition(
            condition,
            x_co2=FRESH_AIR_COMPOSITION["x_co2"],
            x_o2=FRESH_AIR_COMPOSITION["x_o2"],
            x_n2=FRESH_AIR_COMPOSITION["x_n2"],
            l_m=float(condition["L_m_base"]),
        )
        target_condition = _main_feature_condition(
            condition,
            x_co2=float(condition["x_CO2"]),
            x_o2=float(condition["x_O2"]),
            x_n2=float(condition["x_N2"]),
            l_m=float(condition["L_m_base"]),
        )
        baseline_main = main_sensor_features(baseline_condition, condition_rng, f_hz=ultrasonic_spec.center_frequency_hz)
        target_main = main_sensor_features(target_condition, condition_rng, f_hz=ultrasonic_spec.center_frequency_hz)
        slow_params = _channel_dynamic_params(sequence_rng)
        slow_walk = {channel: 0.0 for channel in SLOW_DYNAMIC_CHANNELS}
        schedule = base_schedule.jittered(sequence_rng, stage_jitter)
        phase_intervals = _phase_intervals(schedule, timesteps)
        phase_ids, blends = schedule.resolve_timeline(timesteps)
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
            current = _dynamic_features_from_equilibrium(
                _blend_equilibrium_features(baseline_main, target_main, blend, channels=SLOW_DYNAMIC_CHANNELS),
                slow_state,
                timestep,
                slow_params,
                slow_walk,
                sequence_rng,
                channels=SLOW_DYNAMIC_CHANNELS,
            )
            current["T_C"] = float(condition["T_C_base"])
            current["P_MPa"] = float(condition["P_MPa_base"])
            current["H_RH"] = float(condition["H_RH_base"])
            current["L_m"] = current_l_m
            current["piston_position_m"] = current_l_m
            slow_values = [float(current[channel]) for channel in SLOW_CHANNELS]
            slow[seq_index, timestep, :] = np.array(slow_values, dtype=np.float32)

            # 波形仿真：注入 tv3 物理。extra_gas_kwargs 透传 x_o2；
            # composition dict 含 x_co2/x_o2/x_n2。x_h2/x_ch4 始终为 0。
            extra_gas = {"x_o2": composition["x_o2"]}
            ultrasonic_result = simulate_waveform_measurement(
                x_h2=0.0,
                x_ch4=0.0,
                x_co2=composition["x_co2"],
                x_n2=composition["x_n2"],
                t_c=float(current["T_C"]),
                p_mpa=float(current["P_MPa"]),
                h_rh=float(current["H_RH"]),
                l_m=float(current["L_m"]),
                seed=sequence_rng.randrange(0, 2**32),
                spec=ultrasonic_spec,
                sound_speed_fn=hidden_sound_speed_v2,
                attenuation_fn=hidden_attenuation_v2,
                extra_gas_kwargs=extra_gas,
            )
            if fiber_mic_spec is not None:
                fiber_result = simulate_fiber_mic_measurement(
                    x_h2=0.0,
                    x_ch4=0.0,
                    x_co2=composition["x_co2"],
                    x_n2=composition["x_n2"],
                    t_c=float(current["T_C"]),
                    p_mpa=float(current["P_MPa"]),
                    h_rh=float(current["H_RH"]),
                    l_m=float(current["L_m"]),
                    seed=sequence_rng.randrange(0, 2**32),
                    spec=fiber_mic_spec,
                    sound_speed_fn=hidden_sound_speed_v2,
                    attenuation_fn=hidden_attenuation_v2,
                    extra_gas_kwargs=extra_gas,
                )
            ultrasonic[seq_index, timestep, :] = ultrasonic_result["waveform_int"]
            ultrasonic_scale[seq_index, timestep] = ultrasonic_result["scale_factor"]
            ultrasonic_tof_s[seq_index, timestep] = float(ultrasonic_result["tof_s"])
            ultrasonic_tof_observed_s[seq_index, timestep] = float(ultrasonic_result["tof_observed_s"])
            ultrasonic_peak_index[seq_index, timestep] = int(ultrasonic_result["peak_index"])
            ultrasonic_sound_speed[seq_index, timestep] = float(ultrasonic_result["sound_speed_m_per_s"])
            ultrasonic_sound_speed_estimated[seq_index, timestep] = float(ultrasonic_result["sound_speed_estimated_m_per_s"])
            ultrasonic_alpha[seq_index, timestep] = float(ultrasonic_result["alpha_true_npm"])
            ultrasonic_tof_quality[seq_index, timestep] = float(ultrasonic_result["tof_quality"])
            ultrasonic_tof_accepted[seq_index, timestep] = int(ultrasonic_result["tof_accepted"])
            if fiber_mic_spec is not None:
                fiber_mic[seq_index, timestep, :] = fiber_result["waveform_int"]
                fiber_mic_scale[seq_index, timestep] = fiber_result["scale_factor"]
            slow_rows.append(_slow_row(condition["sequence_id"], timestep, dt_s, phase_id, current))

    result = {
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
        "slow_rows": slow_rows,
    }
    if fiber_mic_spec is not None:
        result["fiber_mic"] = fiber_mic
        result["fiber_mic_scale"] = fiber_mic_scale
    return result


def _main_feature_condition(
    condition: dict[str, str],
    *,
    x_co2: float,
    x_o2: float,
    x_n2: float,
    l_m: float,
) -> dict[str, str]:
    return {
        "x_CO2": _fmt(x_co2, 6),
        "x_O2": _fmt(x_o2, 6),
        "x_N2": _fmt(x_n2, 6),
        "T_C": condition["T_C_base"],
        "P_MPa": condition["P_MPa_base"],
        "H_RH": condition["H_RH_base"],
        "L_m": _fmt(l_m, 6),
    }


def _blend_composition(condition: dict[str, str], blend: float) -> dict[str, float]:
    """从标准新鲜空气 baseline 到 target 的线性混合。

    baseline = 常规空气（CO2 0.04%, O2 20.90%, N2 79.06%），
    target = condition 给出的实际组分。
    blend ∈ [0, 1]：0 表示 baseline，1 表示 target。
    """
    base = FRESH_AIR_COMPOSITION
    return {
        "x_co2": base["x_co2"] + (float(condition["x_CO2"]) - base["x_co2"]) * blend,
        "x_o2": base["x_o2"] + (float(condition["x_O2"]) - base["x_o2"]) * blend,
        "x_n2": base["x_n2"] + (float(condition["x_N2"]) - base["x_n2"]) * blend,
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
        base_scale = _BASE_SCALE[channel]
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


def _stable_uint32(seed: int, sequence_index: int, stream_name: str) -> int:
    payload = f"{seed}:{sequence_index}:{stream_name}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**32)
