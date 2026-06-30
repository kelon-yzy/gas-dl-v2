"""测试 RCDW slow.py 序列数组装配。

对应方案 §5.4 / §11.1。

策略：
- 不依赖 HITRAN 实际谱线（用合成 cache 注入）。
- 端到端调用 build_sequence_arrays，验证返回字典 14 键、shape、不变量。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from rcdw.sim.core.schema import SLOW_CHANNELS, SLOW_DYNAMIC_CHANNELS
from rcdw.sim.generation.conditions import generate_condition_rows
from rcdw.sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    HITRAN_ABSORPTION_BACKEND,
    build_hitran_grid_for_condition,
)
from rcdw.sim.generation.slow import (
    NOISE_FRACTION,
    TAU_DECAY_SYSTEM_S,
    TAU_RISE_SYSTEM_S,
    _blend_composition,
    _multi_tau_channel_step,
    _stable_uint32,
    build_sequence_arrays,
)
from rcdw.sim.generation.spectral import (
    DEFAULT_HITRAN_GAS_SPECS,
    hitran_cache_key,
    write_cached_spectrum,
)
from rcdw.sim.generation.waveforms import FiberMicSpec, WaveformSpec


# ---- blake2b RNG ----


def test_stable_uint32_deterministic():
    """同 (seed, idx, stream) 必产相同 uint32。"""
    a = _stable_uint32(42, 7, "condition")
    b = _stable_uint32(42, 7, "condition")
    assert a == b
    assert 0 <= a < 2**32


def test_stable_uint32_stream_isolation():
    """同 (seed, idx) 不同 stream 应得不同结果。"""
    cond = _stable_uint32(42, 7, "condition")
    seq = _stable_uint32(42, 7, "sequence")
    assert cond != seq


def test_stable_uint32_chunk_independence():
    """不同 sequence_index 应得不同 uint32（保证 chunk 拆分后结果稳定）。"""
    s = {_stable_uint32(42, i, "condition") for i in range(200)}
    assert len(s) == 200  # 200 个序号应几乎全不冲突


# ---- _blend_composition ----


def test_blend_composition_baseline_is_pure_n2():
    """blend=0 时 O2/CO2 归零, N2 = 100%。"""
    condition = {"x_O2": "15.0", "x_CO2": "8.0", "x_N2": "77.0"}
    result = _blend_composition(condition, 0.0)
    assert result == {"x_o2": 0.0, "x_co2": 0.0, "x_n2": 100.0}


def test_blend_composition_target_restored():
    """blend=1 时恢复采样目标。"""
    condition = {"x_O2": "15.0", "x_CO2": "8.0", "x_N2": "77.0"}
    result = _blend_composition(condition, 1.0)
    assert result["x_o2"] == 15.0
    assert result["x_co2"] == 8.0
    assert result["x_n2"] == 77.0


def test_blend_composition_midpoint_interpolation():
    """blend=0.5 应得到中点线性插值。"""
    condition = {"x_O2": "20.0", "x_CO2": "10.0", "x_N2": "70.0"}
    result = _blend_composition(condition, 0.5)
    assert result["x_o2"] == 10.0
    assert result["x_co2"] == 5.0
    assert result["x_n2"] == 85.0


# ---- multi-tau RC 步进 ----


def test_multi_tau_step_target_eq_previous_no_change():
    params = {
        "tau_rise_system_s": 10.0,
        "tau_decay_system_s": 20.0,
        "fast_tau_fraction": 0.3,
        "slow_tau_multiplier": 3.0,
        "fast_response_weight": 0.65,
        "recovery_floor_fraction": 0.05,
    }
    out = _multi_tau_channel_step(previous=2.5, target=2.5, params=params)
    assert math.isclose(out, 2.5, abs_tol=1e-9)


def test_multi_tau_step_rising_moves_toward_target():
    params = {
        "tau_rise_system_s": 5.0,
        "tau_decay_system_s": 20.0,
        "fast_tau_fraction": 0.3,
        "slow_tau_multiplier": 3.0,
        "fast_response_weight": 0.65,
        "recovery_floor_fraction": 0.05,
    }
    out = _multi_tau_channel_step(previous=0.5, target=2.0, params=params)
    assert 0.5 < out < 2.0


def test_multi_tau_step_decay_has_recovery_floor():
    """target < previous 时应注入 recovery floor 残余。"""
    params = {
        "tau_rise_system_s": 5.0,
        "tau_decay_system_s": 5.0,
        "fast_tau_fraction": 0.3,
        "slow_tau_multiplier": 3.0,
        "fast_response_weight": 0.65,
        "recovery_floor_fraction": 0.20,
    }
    out = _multi_tau_channel_step(previous=3.0, target=0.0, params=params)
    # 有 floor 时不应低于 target * (1 - large)
    assert out > 0.0


# ---- 动力学参数表 ----


def test_dynamic_params_no_ch4_field():
    """SLOW_DYNAMIC_CHANNELS 应仅含 V_NDIR_CO2 + V_TCS（无 CH4）。"""
    assert set(TAU_RISE_SYSTEM_S.keys()) == {"V_NDIR_CO2", "V_TCS"}
    assert set(TAU_DECAY_SYSTEM_S.keys()) == {"V_NDIR_CO2", "V_TCS"}
    assert set(NOISE_FRACTION.keys()) == {"V_NDIR_CO2", "V_TCS"}
    assert set(SLOW_DYNAMIC_CHANNELS) == {"V_NDIR_CO2", "V_TCS"}


# ---- 端到端 build_sequence_arrays (empirical backend) ----


def _build_smoke_conditions(n: int = 3, seed: int = 42) -> list[dict[str, str]]:
    return generate_condition_rows(n, seed=seed)


def test_build_sequence_arrays_returns_14_keys_empirical():
    conditions = _build_smoke_conditions(n=2)
    ultrasonic_spec = WaveformSpec()
    fiber_mic_spec = FiberMicSpec()
    result = build_sequence_arrays(
        conditions,
        timesteps=16,
        dt_s=0.5,
        seed=42,
        multi_path_phase="steady",
        ultrasonic_spec=ultrasonic_spec,
        fiber_mic_spec=fiber_mic_spec,
        path_lms=(0.25, 0.35),
        optical_absorption_backend=EMPIRICAL_ABSORPTION_BACKEND,
    )
    expected = {
        "slow",
        "ultrasonic",
        "ultrasonic_scale",
        "ultrasonic_tof_s",
        "ultrasonic_tof_observed_s",
        "ultrasonic_peak_index",
        "ultrasonic_sound_speed_m_per_s",
        "ultrasonic_sound_speed_estimated_m_per_s",
        "ultrasonic_alpha_true_npm",
        "ultrasonic_tof_quality",
        "ultrasonic_tof_accepted",
        "fiber_mic",
        "fiber_mic_scale",
        "slow_rows",
    }
    assert set(result.keys()) == expected


def test_build_sequence_arrays_slow_shape_empirical():
    conditions = _build_smoke_conditions(n=3)
    result = build_sequence_arrays(
        conditions,
        timesteps=16,
        dt_s=0.5,
        seed=42,
        multi_path_phase="off",
        ultrasonic_spec=WaveformSpec(),
        fiber_mic_spec=FiberMicSpec(),
        path_lms=(0.3,),
        optical_absorption_backend=EMPIRICAL_ABSORPTION_BACKEND,
    )
    slow: np.ndarray = result["slow"]  # type: ignore[assignment]
    assert slow.shape == (3, 16, len(SLOW_CHANNELS))
    assert slow.dtype == np.float32


def test_build_sequence_arrays_slow_rows_count():
    """slow_rows 长度 = N_seq * timesteps。"""
    conditions = _build_smoke_conditions(n=2)
    result = build_sequence_arrays(
        conditions,
        timesteps=16,
        dt_s=0.5,
        seed=42,
        multi_path_phase="steady",
        ultrasonic_spec=WaveformSpec(),
        fiber_mic_spec=FiberMicSpec(),
        path_lms=(0.3,),
        optical_absorption_backend=EMPIRICAL_ABSORPTION_BACKEND,
    )
    rows = result["slow_rows"]
    assert len(rows) == 2 * 16
    # 抽查字段
    first = rows[0]
    expected_fields = {
        "sequence_id",
        "timestep",
        "timestamp_s",
        "phase_id",
        "V_NDIR_CO2",
        "V_TCS",
        "T_C",
        "P_MPa",
        "H_RH",
        "L_m",
        "piston_position_m",
    }
    assert set(first.keys()) == expected_fields
    assert "V_NDIR_CH4" not in first


def test_build_sequence_arrays_ultrasonic_shapes():
    conditions = _build_smoke_conditions(n=2)
    spec = WaveformSpec()
    result = build_sequence_arrays(
        conditions,
        timesteps=16,
        dt_s=0.5,
        seed=42,
        multi_path_phase="off",
        ultrasonic_spec=spec,
        fiber_mic_spec=FiberMicSpec(),
        path_lms=(0.3,),
        optical_absorption_backend=EMPIRICAL_ABSORPTION_BACKEND,
    )
    assert result["ultrasonic"].shape == (2, 16, spec.waveform_samples)
    assert result["ultrasonic"].dtype == np.int16
    assert result["ultrasonic_tof_observed_s"].shape == (2, 16)
    assert result["ultrasonic_tof_accepted"].dtype == np.int8


def test_build_sequence_arrays_fiber_mic_shape():
    conditions = _build_smoke_conditions(n=2)
    fm_spec = FiberMicSpec()
    result = build_sequence_arrays(
        conditions,
        timesteps=16,
        dt_s=0.5,
        seed=42,
        multi_path_phase="off",
        ultrasonic_spec=WaveformSpec(),
        fiber_mic_spec=fm_spec,
        path_lms=(0.3,),
        optical_absorption_backend=EMPIRICAL_ABSORPTION_BACKEND,
    )
    assert result["fiber_mic"].shape == (2, 16, fm_spec.waveform_samples)


def test_build_sequence_arrays_invalid_backend():
    with pytest.raises(ValueError, match="optical_absorption_backend"):
        build_sequence_arrays(
            _build_smoke_conditions(n=1),
            timesteps=16,
            dt_s=0.5,
            seed=42,
            multi_path_phase="off",
            ultrasonic_spec=WaveformSpec(),
            fiber_mic_spec=FiberMicSpec(),
            path_lms=(0.3,),
            optical_absorption_backend="bogus_backend",
        )


def test_build_sequence_arrays_reproducibility():
    """同 seed + 同 conditions 应产相同结果（empirical 后端，单进程）。"""
    conds = _build_smoke_conditions(n=2)
    kwargs = dict(
        timesteps=16,
        dt_s=0.5,
        seed=42,
        multi_path_phase="off",
        ultrasonic_spec=WaveformSpec(),
        fiber_mic_spec=FiberMicSpec(),
        path_lms=(0.3,),
        optical_absorption_backend=EMPIRICAL_ABSORPTION_BACKEND,
    )
    a = build_sequence_arrays(conds, **kwargs)
    b = build_sequence_arrays(conds, **kwargs)
    np.testing.assert_array_equal(a["slow"], b["slow"])
    np.testing.assert_array_equal(a["ultrasonic_tof_observed_s"], b["ultrasonic_tof_observed_s"])


# ---- HITRAN 后端（合成 cache）端到端 ----


def _write_synthetic_cache_for_conditions(
    cache_root: Path, conditions: list[dict[str, str]]
) -> None:
    """为所有 condition * (CO2, H2O) 组合写入合成 cache。"""
    for cond in conditions:
        t_c = float(cond["T_C_base"])
        p_mpa = float(cond["P_MPa_base"])
        grid = build_hitran_grid_for_condition("co2", t_c=t_c, p_mpa=p_mpa)
        wn = np.arange(
            grid.wavenumber_min_cm1,
            grid.wavenumber_max_cm1 + grid.wavenumber_step_cm1 * 0.5,
            grid.wavenumber_step_cm1,
            dtype=np.float64,
        )
        co2_coeff = 1.0e-21 * np.exp(-((wn - 2347.0) / 40.0) ** 2)
        h2o_coeff = 1.0e-22 * np.exp(-((wn - 2330.0) / 60.0) ** 2)
        for gas_spec in DEFAULT_HITRAN_GAS_SPECS:
            data = co2_coeff if gas_spec.gas == "CO2" else h2o_coeff
            key = hitran_cache_key(gas_spec, grid)
            write_cached_spectrum(
                cache_root, key, wavenumber_cm1=wn, absorption_coeff_cm1=data
            )


def test_build_sequence_arrays_hitran_backend_with_synthetic_cache(tmp_path):
    conditions = _build_smoke_conditions(n=2)
    _write_synthetic_cache_for_conditions(tmp_path, conditions)
    result = build_sequence_arrays(
        conditions,
        timesteps=16,
        dt_s=0.5,
        seed=42,
        multi_path_phase="steady",
        ultrasonic_spec=WaveformSpec(),
        fiber_mic_spec=FiberMicSpec(),
        path_lms=(0.3,),
        optical_absorption_backend=HITRAN_ABSORPTION_BACKEND,
        hitran_cache_root=str(tmp_path),
    )
    slow = result["slow"]
    assert slow.shape == (2, 16, len(SLOW_CHANNELS))
    # V_NDIR_CO2 在所有 timestep 都应 > 0
    co2_idx = SLOW_CHANNELS.index("V_NDIR_CO2")
    assert (slow[:, :, co2_idx] > 0).all()


def test_baseline_phase_v_ndir_co2_high_hitran(tmp_path):
    """baseline 段 (blend=0, CO2=0) 时 V_NDIR_CO2 应接近 init 基线（吸收最弱）；
    steady 段 (blend=1, CO2>0) 应明显下降。"""
    # 构造一个 CO2 含量较高的 condition 以放大差异
    conditions = [
        {
            "sequence_id": "RCDW-Q000001",
            "mixture_id": "RCDW-M000001",
            "x_O2": "10.000000",
            "x_CO2": "15.000000",
            "x_N2": "75.000000",
            "T_C_base": "25.0000",
            "P_MPa_base": "0.1000",
            "H_RH_base": "40.0000",
            "L_m_base": "0.5000",
            "status": "synthetic_measurement",
        }
    ]
    _write_synthetic_cache_for_conditions(tmp_path, conditions)
    result = build_sequence_arrays(
        conditions,
        timesteps=64,
        dt_s=0.5,
        seed=42,
        multi_path_phase="off",
        ultrasonic_spec=WaveformSpec(),
        fiber_mic_spec=FiberMicSpec(),
        path_lms=(0.5,),
        optical_absorption_backend=HITRAN_ABSORPTION_BACKEND,
        hitran_cache_root=str(tmp_path),
    )
    co2_idx = SLOW_CHANNELS.index("V_NDIR_CO2")
    slow = result["slow"][0, :, co2_idx]
    # baseline 平均 vs steady 平均：baseline 段在前 ~15%(标准 STANDARD_EXPOSURE)
    baseline_mean = slow[:8].mean()
    steady_mean = slow[24:48].mean()  # steady 段约占中间 35%
    # CO2 吸收应使 steady 段 V_NDIR_CO2 < baseline 段
    assert steady_mean < baseline_mean


def test_l_m_scan_in_baseline_phase(tmp_path):
    """multi_path_phase='baseline' 时, baseline 段 L_m 应在 path_lms 中循环。"""
    conditions = _build_smoke_conditions(n=1)
    _write_synthetic_cache_for_conditions(tmp_path, conditions)
    result = build_sequence_arrays(
        conditions,
        timesteps=20,
        dt_s=0.5,
        seed=42,
        multi_path_phase="baseline",
        ultrasonic_spec=WaveformSpec(),
        fiber_mic_spec=FiberMicSpec(),
        path_lms=(0.2, 0.3, 0.4),
        optical_absorption_backend=HITRAN_ABSORPTION_BACKEND,
        hitran_cache_root=str(tmp_path),
    )
    l_m_idx = SLOW_CHANNELS.index("L_m")
    l_m_series = result["slow"][0, :, l_m_idx]
    # 至少应出现 path_lms 中的多个值（baseline 段做扫描）
    unique_vals = np.unique(np.round(l_m_series, 4))
    assert len(unique_vals) >= 2, f"L_m 应在 baseline 段扫描多个值, 实际 unique={unique_vals}"
