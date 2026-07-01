"""RCDW 慢通道动力学 + 序列数组装配（HITRAN/empirical 后端统一走 multi-tau）。

对应方案 §5.4。与 HG 主线 ``src/sim/generation/slow.py`` 的差异：
1. 删除 ``V_NDIR_CH4`` 整条线（RCDW 无 CH4 组分）。
2. 组分字段读取 ``x_O2 / x_CO2 / x_N2``；``_blend_composition`` baseline
   定义为 100% N2 纯背景气（与 HG 行为一致），blend=1 时恢复采样目标。
3. **仅保留 multi-tau equilibrium 路径**，不移植 legacy empirical 单指数 RC
   （方案 v1.1 §5.4 修正说明：legacy 路径触发条件 ``empirical`` 后端 + 无 jitter
   在 RCDW 中无实际意义，删除可避免死代码）。
4. waveforms 调用签名改为 ``x_o2 / x_co2 / x_n2``。
5. RNG seeding：复用 HG 的 blake2b 双流策略，独立重写。

multi-tau 动力学逐通道独立采样：``tau_rise / tau_decay / fast_tau_fraction /
slow_tau_multiplier / fast_response_weight / recovery_floor_fraction /
noise_sigma / random_walk_sigma / drift_slope``。
"""

from __future__ import annotations

import hashlib
import math
import random

import numpy as np

from rcdw.sim.core.schema import SLOW_CHANNELS, SLOW_DYNAMIC_CHANNELS
from rcdw.sim.generation.acoustic_physics import (
    PROCESSING_PARAMS,
    main_sensor_features,
    rcdw_thermal_conductivity_sensor_feature,
)
from rcdw.sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    HITRAN_ABSORPTION_BACKEND,
    VALID_OPTICAL_ABSORPTION_BACKENDS,
    compute_hitran_optical_absorption,
)
from rcdw.sim.generation.phases import PhaseSchedule, resolve_phase_schedule
from rcdw.sim.generation.spectral import HitranGridSpec, PreparedTabulatedSpectra
from rcdw.sim.generation.waveforms import (
    FiberMicSpec,
    WaveformSpec,
    simulate_fiber_mic_measurement,
    simulate_waveform_measurement,
)


# 动力学参数表（RCDW 版，仅 V_NDIR_CO2 与 V_TCS）。
TAU_RISE_SYSTEM_S = {
    "V_NDIR_CO2": (6.0, 18.0),
    "V_TCS": (10.0, 35.0),
}
TAU_DECAY_SYSTEM_S = {
    "V_NDIR_CO2": (10.0, 28.0),
    "V_TCS": (20.0, 60.0),
}
NOISE_FRACTION = {"V_NDIR_CO2": 0.0025, "V_TCS": 0.003}

# multi-tau 通道基础响应尺度（影响噪声/漂移幅度）。
_BASE_SCALE = {"V_NDIR_CO2": 2.5, "V_TCS": 1.5}


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
    optical_absorption_backend: str = HITRAN_ABSORPTION_BACKEND,
    hitran_cache_root: str = "data/hitran_cache",
    start_sequence_index: int = 0,
) -> dict[str, object]:
    """逐序列、逐时间步构造 slow + ultrasonic + fiber_mic 数组。

    返回字典 14 键：slow, ultrasonic, ultrasonic_scale,
    ultrasonic_tof_s, ultrasonic_tof_observed_s, ultrasonic_peak_index,
    ultrasonic_sound_speed_m_per_s, ultrasonic_sound_speed_estimated_m_per_s,
    ultrasonic_alpha_true_npm, ultrasonic_tof_quality, ultrasonic_tof_accepted,
    fiber_mic, fiber_mic_scale, slow_rows。

    Args:
        conditions: ``generate_condition_rows`` 输出。
        timesteps: 序列长度。
        dt_s: 采样间隔（秒），仅用于 slow_rows 中 timestamp_s。
        seed: 主 seed，配合 ``_stable_uint32`` 通过 blake2b 派生 condition/sequence 双流。
        multi_path_phase: 多光程扫描发生在哪个 phase（``"off" / "baseline" / "steady"``）。
        ultrasonic_spec / fiber_mic_spec: 波形仿真规格。
        path_lms: 多光程扫描时遍历的 L_m 取值序列。
        phase_schedule: 默认 ``"standard_exposure"``（v1.2 RCDW 唯一注册项）。
        stage_jitter: 段时长随机扰动比例 ∈ [0, 1)。
        optical_absorption_backend: ``hitran_hapi_v1``（默认）或 ``empirical_v1``。
        hitran_cache_root: HITRAN 谱线缓存目录（仅 hitran 后端使用）。
        start_sequence_index: 用于 chunk 并行时保持 RNG 在全局序号上稳定。

    Returns:
        14 键字典；详见函数体最后的 return。
    """
    if optical_absorption_backend not in VALID_OPTICAL_ABSORPTION_BACKENDS:
        raise ValueError(
            f"optical_absorption_backend must be one of "
            f"{list(VALID_OPTICAL_ABSORPTION_BACKENDS)}, got {optical_absorption_backend!r}"
        )
    sequence_count = len(conditions)
    n_slow_channels = len(SLOW_CHANNELS)
    slow = np.zeros((sequence_count, timesteps, n_slow_channels), dtype=np.float32)
    ultrasonic = np.zeros(
        (sequence_count, timesteps, ultrasonic_spec.waveform_samples), dtype=np.int16
    )
    ultrasonic_scale = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_s = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_observed_s = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_peak_index = np.zeros((sequence_count, timesteps), dtype=np.int32)
    ultrasonic_sound_speed = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_sound_speed_estimated = np.zeros(
        (sequence_count, timesteps), dtype=np.float32
    )
    ultrasonic_alpha = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_quality = np.zeros((sequence_count, timesteps), dtype=np.float32)
    ultrasonic_tof_accepted = np.zeros((sequence_count, timesteps), dtype=np.int8)
    fiber_mic = np.zeros(
        (sequence_count, timesteps, fiber_mic_spec.waveform_samples), dtype=np.int16
    )
    fiber_mic_scale = np.zeros((sequence_count, timesteps), dtype=np.float32)
    slow_rows: list[dict[str, str]] = []

    base_schedule = resolve_phase_schedule(phase_schedule)
    is_hitran = optical_absorption_backend == HITRAN_ABSORPTION_BACKEND
    is_baseline_scan = multi_path_phase == "baseline"
    is_steady_scan = multi_path_phase == "steady"
    spectra_cache: (
        dict[tuple[str, HitranGridSpec], PreparedTabulatedSpectra] | None
    ) = {} if is_hitran else None

    for seq_index, condition in enumerate(conditions):
        global_sequence_index = start_sequence_index + seq_index
        condition_rng = random.Random(
            _stable_uint32(seed, global_sequence_index, "condition")
        )
        sequence_rng = random.Random(
            _stable_uint32(seed, global_sequence_index, "sequence")
        )

        # baseline = 100% N2 纯背景气（O2=0, CO2=0, N2=100），exposure = 采样目标。
        baseline_condition = _main_feature_condition(
            condition,
            x_o2=0.0,
            x_co2=0.0,
            x_n2=100.0,
            l_m=float(condition["L_m_base"]),
        )
        target_condition = _main_feature_condition(
            condition,
            x_o2=float(condition["x_O2"]),
            x_co2=float(condition["x_CO2"]),
            x_n2=float(condition["x_N2"]),
            l_m=float(condition["L_m_base"]),
        )
        if is_hitran:
            # HITRAN 后端：V_TCS 走 main 模块的 TCS 计算（不依赖 NDIR/声学）；
            # V_NDIR_CO2 由 HITRAN 光学栈逐 timestep 计算 equilibrium。
            baseline_main = rcdw_thermal_conductivity_sensor_feature(
                baseline_condition, condition_rng
            )
            target_main = rcdw_thermal_conductivity_sensor_feature(
                target_condition, condition_rng
            )
        else:
            # empirical 后端：V_TCS 与 V_NDIR_CO2 都由 main_sensor_features 推 equilibrium。
            baseline_main = main_sensor_features(baseline_condition, condition_rng)
            target_main = main_sensor_features(target_condition, condition_rng)

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

            if is_hitran:
                # V_TCS multi-tau
                current = _dynamic_features_from_equilibrium(
                    _blend_equilibrium_features(
                        baseline_main, target_main, blend, channels=("V_TCS",)
                    ),
                    slow_state,
                    timestep,
                    slow_params,
                    slow_walk,
                    sequence_rng,
                    channels=("V_TCS",),
                )
                # V_NDIR_CO2 HITRAN equilibrium + multi-tau
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
                        channels=("V_NDIR_CO2",),
                    )
                )
            else:
                # empirical 后端：两个动态通道一起 multi-tau。
                current = _dynamic_features_from_equilibrium(
                    _blend_equilibrium_features(
                        baseline_main,
                        target_main,
                        blend,
                        channels=SLOW_DYNAMIC_CHANNELS,
                    ),
                    slow_state,
                    timestep,
                    slow_params,
                    slow_walk,
                    sequence_rng,
                    channels=SLOW_DYNAMIC_CHANNELS,
                )

            # 环境通道：基线值 + 传感器测量噪声（M2 修复）。
            # 噪声使 FeatureExtractor 的 delta_T/P/RH 不再恒为零，
            # 同时物理量级 << 基线值，不影响声学/光学计算。
            current["T_C"] = float(condition["T_C_base"]) + sequence_rng.gauss(0, 0.05)
            current["P_MPa"] = float(condition["P_MPa_base"]) + sequence_rng.gauss(0, 0.0005)
            current["H_RH"] = float(condition["H_RH_base"]) + sequence_rng.gauss(0, 0.1)
            current["L_m"] = current_l_m
            current["piston_position_m"] = current_l_m

            slow_values = [float(current[channel]) for channel in SLOW_CHANNELS]
            slow[seq_index, timestep, :] = np.array(slow_values, dtype=np.float32)

            ultrasonic_result = simulate_waveform_measurement(
                x_o2=composition["x_o2"],
                x_co2=composition["x_co2"],
                x_n2=composition["x_n2"],
                t_c=float(current["T_C"]),
                p_mpa=float(current["P_MPa"]),
                h_rh=float(current["H_RH"]),
                l_m=float(current["L_m"]),
                seed=sequence_rng.randrange(0, 2**32),
                spec=ultrasonic_spec,
            )
            fiber_result = simulate_fiber_mic_measurement(
                x_o2=composition["x_o2"],
                x_co2=composition["x_co2"],
                x_n2=composition["x_n2"],
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
            ultrasonic_tof_observed_s[seq_index, timestep] = float(
                ultrasonic_result["tof_observed_s"]
            )
            ultrasonic_peak_index[seq_index, timestep] = int(
                ultrasonic_result["peak_index"]
            )
            ultrasonic_sound_speed[seq_index, timestep] = float(
                ultrasonic_result["sound_speed_m_per_s"]
            )
            ultrasonic_sound_speed_estimated[seq_index, timestep] = float(
                ultrasonic_result["sound_speed_estimated_m_per_s"]
            )
            ultrasonic_alpha[seq_index, timestep] = float(
                ultrasonic_result["alpha_true_npm"]
            )
            ultrasonic_tof_quality[seq_index, timestep] = float(
                ultrasonic_result["tof_quality"]
            )
            ultrasonic_tof_accepted[seq_index, timestep] = int(
                ultrasonic_result["tof_accepted"]
            )
            fiber_mic[seq_index, timestep, :] = fiber_result["waveform_int16"]
            fiber_mic_scale[seq_index, timestep] = fiber_result["scale_factor"]
            slow_rows.append(
                _slow_row(condition["sequence_id"], timestep, dt_s, phase_id, current)
            )

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


def build_sequence_arrays_chunk(
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
    optical_absorption_backend: str = HITRAN_ABSORPTION_BACKEND,
    hitran_cache_root: str = "data/hitran_cache",
    start_sequence_index: int = 0,
) -> dict[str, object]:
    """Worker 入口，ProcessPoolExecutor 友好的同步包装。"""
    return build_sequence_arrays(
        conditions,
        timesteps=timesteps,
        dt_s=dt_s,
        seed=seed,
        multi_path_phase=multi_path_phase,
        ultrasonic_spec=ultrasonic_spec,
        fiber_mic_spec=fiber_mic_spec,
        path_lms=path_lms,
        phase_schedule=phase_schedule,
        stage_jitter=stage_jitter,
        optical_absorption_backend=optical_absorption_backend,
        hitran_cache_root=hitran_cache_root,
        start_sequence_index=start_sequence_index,
    )


# ---- 内部函数 ----


def _main_feature_condition(
    condition: dict[str, str],
    *,
    x_o2: float,
    x_co2: float,
    x_n2: float,
    l_m: float,
) -> dict[str, str]:
    """构造给 ``main_sensor_features`` / TCS / HITRAN 用的瞬时 condition。"""
    return {
        "x_O2": _fmt(x_o2, 6),
        "x_CO2": _fmt(x_co2, 6),
        "x_N2": _fmt(x_n2, 6),
        "T_C": condition["T_C_base"],
        "P_MPa": condition["P_MPa_base"],
        "H_RH": condition["H_RH_base"],
        "L_m": _fmt(l_m, 6),
    }


def _blend_composition(condition: dict[str, str], blend: float) -> dict[str, float]:
    """逐时间步组分插值。

    baseline (blend=0)：x_O2 = x_CO2 = 0，x_N2 = 100（纯 N2 背景气）
    target   (blend=1)：恢复 condition 中采样的目标浓度
    """
    return {
        "x_o2": float(condition["x_O2"]) * blend,
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
    """对指定通道的稳态电压做线性插值。"""
    return {
        channel: float(baseline_main[channel])
        + (float(target_main[channel]) - float(baseline_main[channel])) * blend
        for channel in channels
    }


def _hitran_ndir_equilibrium(
    condition: dict[str, str],
    *,
    composition: dict[str, float],
    l_m: float,
    hitran_cache_root: str,
    spectra_cache: dict[tuple[str, HitranGridSpec], PreparedTabulatedSpectra] | None,
) -> dict[str, float]:
    """逐 timestep 通过 HITRAN 计算 V_NDIR_CO2 的稳态电压。"""
    optical = compute_hitran_optical_absorption(
        _main_feature_condition(
            condition,
            x_o2=composition["x_o2"],
            x_co2=composition["x_co2"],
            x_n2=composition["x_n2"],
            l_m=l_m,
        ),
        cache_root=hitran_cache_root,
        spectra_cache=spectra_cache,
    )
    return {
        "V_NDIR_CO2": max(
            0.1,
            PROCESSING_PARAMS["optical_baseline_co2_init"]
            * math.exp(-float(optical["absorption_co2_observed"])),
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
    """multi-tau 步进 + drift + random walk + 高斯噪声。"""
    current: dict[str, float] = {}
    for channel in channels:
        target = float(equilibrium[channel])
        previous = state.get(channel, target)
        value = _multi_tau_channel_step(
            previous=previous,
            target=target,
            params=slow_params[channel],
        )
        slow_walk[channel] += sequence_rng.gauss(
            0.0, slow_params[channel]["random_walk_sigma"]
        )
        value += slow_params[channel]["drift_slope"] * timestep
        value += slow_walk[channel]
        value += sequence_rng.gauss(0.0, slow_params[channel]["noise_sigma"])
        value = max(1e-9, value)
        state[channel] = value
        current[channel] = value
    return current


def _channel_dynamic_params(rng: random.Random) -> dict[str, dict[str, float]]:
    """逐通道独立采样动力学参数（每个 sequence 一份）。"""
    params: dict[str, dict[str, float]] = {}
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
            "drift_slope": rng.uniform(-1.0, 1.0)
            * base_scale
            * NOISE_FRACTION[channel]
            * 0.015,
        }
    return params


def _multi_tau_channel_step(
    previous: float, target: float, params: dict[str, float]
) -> float:
    """双指数 RC 步进 + recovery floor。

    上升用 tau_rise，下降用 tau_decay；alpha = w*fast + (1-w)*slow，
    下降时 target 注入 recovery_floor 残余。

    注意：``tau_*_system_s`` 名义单位为秒，但 ``exp(-1/tau)`` 中
    隐含 dt=1（即 1 时间步 = 1 单位），实际单位为**时间步**。
    改变 ``dt_s`` 仅影响时间戳标注，不改变动力学响应。
    """
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
    """多光程扫描时给当前 timestep 选择 L_m。"""
    for phase_id, start, end in phase_intervals:
        if start <= timestep < end:
            if is_baseline_scan and phase_id == "baseline":
                return float(
                    path_lms[_scan_path_index(timestep - start, end - start, len(path_lms))]
                )
            if is_steady_scan and phase_id == "steady":
                return float(
                    path_lms[_scan_path_index(timestep - start, end - start, len(path_lms))]
                )
            return float(l_m_base)
    raise ValueError(f"timestep {timestep} is outside phase schedule")


def _phase_intervals(
    schedule: PhaseSchedule, timesteps: int
) -> tuple[tuple[str, int, int], ...]:
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


def _slow_row(
    sequence_id: str,
    timestep: int,
    dt_s: float,
    phase_id: str,
    current: dict[str, float],
) -> dict[str, str]:
    """构造 slow_sequence.csv 一行（字段顺序需与 SLOW_SEQUENCE_FIELDS 一致）。"""
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
    """blake2b 双流 RNG 派生函数（与 HG 主线一致，独立重写）。

    保证同一 (seed, sequence_index, stream_name) 组合无论 chunk 划分如何
    都产生相同 uint32，是多进程并行可复现性的核心。
    """
    payload = f"{seed}:{sequence_index}:{stream_name}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**32)
