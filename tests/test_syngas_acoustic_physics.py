"""合成气声学物理 + 光学串扰测试。

验证：
- 声速混合公式正确加入 CO 项
- 衰减公式正确加入 CO V-T 项
- 热导混合正确加入 CO
- main_sensor_features 输出 V_NDIR_CO
- 3×3 串扰矩阵 Step 1 / Step 2 行为
- hydrogen_ng acoustic_physics 未被影响
"""
from __future__ import annotations

import random

import pytest

from sim.generation import acoustic_physics as hg_acoustic
from sim.generation.optical_crosstalk import apply_optical_crosstalk as hg_apply_crosstalk
from sim.generation.syngas.acoustic_physics import (
    _hidden_absorption_co,
    _hidden_lambda_mix,
    PROCESSING_PARAMS_V2,
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
    main_sensor_features,
    thermal_conductivity_sensor_feature,
)
from sim.generation.syngas.optical_crosstalk import (
    DEFAULT_SYNGAS_CROSSTALK_SPEC,
    apply_syngas_optical_crosstalk,
)


# ---------------------------------------------------------------------------
# 声速
# ---------------------------------------------------------------------------


def test_sound_speed_co_pure_close_to_352():
    """100% CO 在 25°C 应给出 ≈352 m/s。"""
    c = hidden_sound_speed_v2(x_h2=0, x_ch4=0, x_co2=0, x_n2=0, x_co=100.0, t_c=25.0)
    assert abs(c - 352.0) < 1.0


def test_sound_speed_co_and_n2_close():
    """同浓度的 CO 与 N2 给出的混合声速差 <1 m/s。"""
    c_co = hidden_sound_speed_v2(x_h2=20, x_ch4=10, x_co2=10, x_n2=0, x_co=60.0, t_c=25.0)
    c_n2 = hidden_sound_speed_v2(x_h2=20, x_ch4=10, x_co2=10, x_n2=60.0, x_co=0, t_c=25.0)
    assert abs(c_co - c_n2) < 1.0


def test_sound_speed_monotone_in_h2():
    """H2 增加显著提升声速。

    理想气混合 c=sqrt(γRT/M) 下，H2 5%→50% 提升约 115 m/s
    （旧线性加权代理夸大至 >200 m/s，新物理下阈值调整为 100）。
    """
    c_low = hidden_sound_speed_v2(x_h2=5, x_ch4=2, x_co2=5, x_n2=3, x_co=85.0, t_c=25.0)
    c_high = hidden_sound_speed_v2(x_h2=50, x_ch4=2, x_co2=5, x_n2=3, x_co=40.0, t_c=25.0)
    assert c_high > c_low + 100.0


def test_sound_speed_has_floor():
    """极端低 c_mix 输入应被限制到 ≥200 m/s。"""
    # 极端低温
    c = hidden_sound_speed_v2(x_h2=0, x_ch4=0, x_co2=100.0, x_n2=0, x_co=0, t_c=-300.0)
    assert c >= 200.0


# ---------------------------------------------------------------------------
# 衰减
# ---------------------------------------------------------------------------


def test_attenuation_has_alpha_co_field():
    res = hidden_attenuation_v2(
        x_h2=20, x_ch4=2, x_co2=10, x_n2=5, x_co=63.0, t_c=25.0, p_mpa=0.5, h_rh=50.0
    )
    assert "alpha_co_v2" in res
    assert "f_relax_co_eff" in res
    assert res["alpha_co_v2"] >= 0.0


def test_attenuation_co_contribution_increases_with_x_co():
    """x_CO 增大时 alpha_co_v2 应增加（其他条件不变）。"""
    low = hidden_attenuation_v2(
        x_h2=20, x_ch4=2, x_co2=10, x_n2=58.0, x_co=10.0, t_c=25.0, p_mpa=0.5, h_rh=50.0
    )
    high = hidden_attenuation_v2(
        x_h2=20, x_ch4=2, x_co2=10, x_n2=3.0, x_co=65.0, t_c=25.0, p_mpa=0.5, h_rh=50.0
    )
    assert high["alpha_co_v2"] > low["alpha_co_v2"]


def test_attenuation_co_h2o_acceleration():
    """提高 H_RH 应通过 k_h2o_to_f_relax_co 加速 CO 弛豫，改变 f_relax_co_eff。"""
    dry = hidden_attenuation_v2(
        x_h2=20, x_ch4=2, x_co2=10, x_n2=8, x_co=60.0, t_c=25.0, p_mpa=0.5, h_rh=10.0
    )
    wet = hidden_attenuation_v2(
        x_h2=20, x_ch4=2, x_co2=10, x_n2=8, x_co=60.0, t_c=25.0, p_mpa=0.5, h_rh=80.0
    )
    assert wet["f_relax_co_eff"] > dry["f_relax_co_eff"]


def test_attenuation_aggregate_is_sum_of_parts():
    """alpha_true_v2 应等于各组分项之和（不含负值截断）。"""
    res = hidden_attenuation_v2(
        x_h2=20, x_ch4=2, x_co2=10, x_n2=5, x_co=63.0, t_c=25.0, p_mpa=0.5, h_rh=50.0
    )
    parts = (
        res["alpha_classical_v2"]
        + res["alpha_co2_v2"]
        + res["alpha_ch4_v2"]
        + res["alpha_h2_diffusion_v2"]
        + res["alpha_n2_background_v2"]
        + res["alpha_co_v2"]
        + res["alpha_h2o_v2"]
    )
    assert abs(res["alpha_true_v2"] - parts) < 1e-9


# ---------------------------------------------------------------------------
# 吸收 + 热导
# ---------------------------------------------------------------------------


def test_hidden_absorption_co_nonnegative_in_range():
    for x_co in (0.0, 15.0, 35.0, 65.0):
        val = _hidden_absorption_co(x_co=x_co, h_rh=50.0, p_mpa=0.5, t_c=25.0)
        assert val >= 0.0


def test_hidden_absorption_co_monotone():
    a = _hidden_absorption_co(x_co=20.0, h_rh=50.0, p_mpa=0.5, t_c=25.0)
    b = _hidden_absorption_co(x_co=50.0, h_rh=50.0, p_mpa=0.5, t_c=25.0)
    assert b > a


def test_lambda_mix_co_term_effective():
    """WMS 下 CO 组分通过 φ_ij 影响热导率（CO 与 N2 的 λ/η 略异，故结果不同）。"""
    val_with_co = _hidden_lambda_mix(x_h2=20.0, x_ch4=0.0, x_co2=10.0, x_n2=0.0, x_co=60.0, t_c=25.0)
    val_co_as_n2 = _hidden_lambda_mix(x_h2=20.0, x_ch4=0.0, x_co2=10.0, x_n2=60.0, x_co=0.0, t_c=25.0)
    assert val_with_co != val_co_as_n2


# ---------------------------------------------------------------------------
# main_sensor_features
# ---------------------------------------------------------------------------


@pytest.fixture
def syngas_condition():
    return {
        "x_H2": "30.0",
        "x_CH4": "2.0",
        "x_CO2": "10.0",
        "x_N2": "3.0",
        "x_CO": "55.0",
        "T_C": "25.0",
        "P_MPa": "0.5",
        "H_RH": "50.0",
        "L_m": "1.0",
    }


def test_main_sensor_features_outputs_v_ndir_co(syngas_condition):
    rng = random.Random(123)
    res = main_sensor_features(syngas_condition, rng)
    assert "V_NDIR_CO" in res
    assert res["V_NDIR_CO"] >= 0.1
    assert "V_NDIR_CH4" in res
    assert "V_NDIR_CO2" in res
    assert "V_TCS" in res


def test_main_sensor_features_co_absorption_observed_equals_true_in_step1(syngas_condition):
    """Step 1（默认）下 CO 通道无串扰：observed_co == true_co。"""
    rng = random.Random(456)
    res = main_sensor_features(syngas_condition, rng)
    assert abs(res["absorption_co_observed"] - res["absorption_co_true"]) < 1e-9


def test_thermal_conductivity_sensor_feature_uses_x_co(syngas_condition):
    rng = random.Random(7)
    res = thermal_conductivity_sensor_feature(syngas_condition, rng)
    assert "V_TCS" in res
    assert res["V_TCS"] > 0


# ---------------------------------------------------------------------------
# 3×3 串扰
# ---------------------------------------------------------------------------


def test_crosstalk_step1_co_pure():
    """Step 1 模式：observed_co == true_co；CH4↔CO2 行为同 hydrogen_ng。"""
    res = apply_syngas_optical_crosstalk(
        absorption_ch4=1.0,
        absorption_co2=2.0,
        absorption_co=3.0,
        enable_co_crosstalk=False,
    )
    assert res["absorption_co_observed"] == 3.0
    assert res["absorption_co_cross_from_co2"] == 0.0
    assert res["absorption_co2_cross_from_co"] == 0.0
    # CH4↔CO2 行为：0.035 * co2 + ch4
    assert abs(res["absorption_ch4_observed"] - (1.0 + 0.035 * 2.0)) < 1e-9
    assert abs(res["absorption_co2_observed"] - (2.0 + 0.012 * 1.0)) < 1e-9


def test_crosstalk_step1_matches_hydrogen_ng_for_ch4_co2():
    """Step 1 的 CH4 / CO2 通道结果应与 hydrogen_ng 的 2×2 完全一致。"""
    sg = apply_syngas_optical_crosstalk(
        absorption_ch4=0.5, absorption_co2=0.3, absorption_co=2.0, enable_co_crosstalk=False
    )
    hg = hg_apply_crosstalk(absorption_ch4=0.5, absorption_co2=0.3)
    assert abs(sg["absorption_ch4_observed"] - hg["absorption_ch4_observed"]) < 1e-9
    assert abs(sg["absorption_co2_observed"] - hg["absorption_co2_observed"]) < 1e-9


def test_crosstalk_step2_co_receives_co2_leak():
    """Step 2：CO2 升高时，CO 通道 observed 会比 true 大。"""
    res = apply_syngas_optical_crosstalk(
        absorption_ch4=1.0,
        absorption_co2=2.0,
        absorption_co=3.0,
        enable_co_crosstalk=True,
    )
    expected_leak = DEFAULT_SYNGAS_CROSSTALK_SPEC.co_channel_co2_response * 2.0
    assert abs(res["absorption_co_cross_from_co2"] - expected_leak) < 1e-9
    assert res["absorption_co_observed"] > 3.0


def test_crosstalk_step2_co2_receives_co_leak():
    res = apply_syngas_optical_crosstalk(
        absorption_ch4=1.0,
        absorption_co2=2.0,
        absorption_co=3.0,
        enable_co_crosstalk=True,
    )
    expected_leak = DEFAULT_SYNGAS_CROSSTALK_SPEC.co2_channel_co_response * 3.0
    assert abs(res["absorption_co2_cross_from_co"] - expected_leak) < 1e-9


@pytest.mark.parametrize("a_ch4", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("a_co2", [0.0, 0.3, 1.5])
@pytest.mark.parametrize("a_co", [0.0, 0.5, 2.0])
def test_crosstalk_observed_ge_true_when_enabled(a_ch4, a_co2, a_co):
    """启用串扰时，observed >= true（系数非负前提下）。"""
    res = apply_syngas_optical_crosstalk(
        absorption_ch4=a_ch4,
        absorption_co2=a_co2,
        absorption_co=a_co,
        enable_co_crosstalk=True,
    )
    assert res["absorption_ch4_observed"] >= a_ch4 - 1e-12
    assert res["absorption_co2_observed"] >= a_co2 - 1e-12
    assert res["absorption_co_observed"] >= a_co - 1e-12


# ---------------------------------------------------------------------------
# 隔离性：hydrogen_ng 物理不变
# ---------------------------------------------------------------------------


def test_hydrogen_ng_acoustic_signature_unchanged():
    """hydrogen_ng 的 hidden_sound_speed_v2 签名仍是 5 参（无 x_co）。"""
    import inspect

    sig = inspect.signature(hg_acoustic.hidden_sound_speed_v2)
    assert list(sig.parameters) == ["x_h2", "x_ch4", "x_co2", "x_n2", "t_c"]


def test_hydrogen_ng_lambda_mix_signature_unchanged():
    """hydrogen_ng 的 _hidden_lambda_mix 签名不含 x_co（WMS 仍 5 参：x_h2/x_ch4/x_co2/x_n2/t_c）。"""
    import inspect

    sig = inspect.signature(hg_acoustic._hidden_lambda_mix)
    assert list(sig.parameters) == ["x_h2", "x_ch4", "x_co2", "x_n2", "t_c"]


def test_hydrogen_ng_params_no_co_keys():
    """hydrogen_ng PROCESSING_PARAMS_V2 不包含 CO 相关键，避免被误污染。"""
    keys = set(hg_acoustic.PROCESSING_PARAMS_V2)
    assert "alpha_lambda_max_co" not in keys
    assert "f_relax_co_per_atm" not in keys


def test_syngas_params_has_co_keys():
    keys = set(PROCESSING_PARAMS_V2)
    assert "alpha_lambda_max_co" in keys
    assert "f_relax_co_per_atm" in keys
    assert "k_h2o_to_f_relax_co" in keys
