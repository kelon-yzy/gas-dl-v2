"""掘进通风声学物理测试。

验证：
- 声速混合公式仅含 CO2/O2/N2 三组分
- 衰减公式含 alpha_o2 字段（200kHz 下 ≈ 0）
- 热导混合含 O2（WMS 规则）
- main_sensor_features 输出 V_NDIR_CO2/V_TCS，无 V_NDIR_CH4 / V_NDIR_CO
- 空气组成下 c_mix ≈ 346 m/s（与已知空气声速交叉验证）
- hydrogen_ng acoustic_physics 未被影响
"""
from __future__ import annotations

import math
import random

import pytest

from tv3.sim.generation import acoustic_physics as hg_acoustic
from tv3.sim.generation.tunnel_ventilation.acoustic_physics import (
    PROCESSING_PARAMS_V2,
    _hidden_absorption_co2,
    _hidden_lambda_mix,
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
    main_sensor_features,
    thermal_conductivity_sensor_feature,
)


# ---------------------------------------------------------------------------
# 声速
# ---------------------------------------------------------------------------


def test_sound_speed_co2_pure_close_to_270():
    """100% CO2 在 25°C 应给出 ≈270 m/s（理想气公式 c=sqrt(γRT/M)）。"""
    c = hidden_sound_speed_v2(x_h2=0, x_ch4=0, x_co2=100.0, x_n2=0, t_c=25.0, x_o2=0.0)
    assert abs(c - 270.0) < 2.0


def test_sound_speed_o2_pure_close_to_330():
    """100% O2 在 25°C 应给出 ≈330 m/s（NIST 理想气计算）。"""
    c = hidden_sound_speed_v2(x_h2=0, x_ch4=0, x_co2=0, x_n2=0, t_c=25.0, x_o2=100.0)
    assert abs(c - 330.0) < 2.0


def test_sound_speed_n2_pure_close_to_353():
    """100% N2 在 25°C 应给出 ≈353 m/s（NIST 理想气计算）。"""
    c = hidden_sound_speed_v2(x_h2=0, x_ch4=0, x_co2=0, x_n2=100.0, t_c=25.0, x_o2=0.0)
    assert abs(c - 353.0) < 2.0


def test_sound_speed_o2_n2_difference_about_6_percent():
    """O2 与 N2 声速差 ≈ 22 m/s（约 6.4%），是超声通道区分两者的物理基础。"""
    c_o2 = hidden_sound_speed_v2(x_h2=0, x_ch4=0, x_co2=0, x_n2=0, t_c=25.0, x_o2=100.0)
    c_n2 = hidden_sound_speed_v2(x_h2=0, x_ch4=0, x_co2=0, x_n2=100.0, t_c=25.0, x_o2=0.0)
    diff = c_n2 - c_o2
    assert 20.0 < diff < 25.0


def test_sound_speed_air_mixture_close_to_346():
    """空气组成（78% N2 + 21% O2 + 0.04% CO2）在 25°C 下 c_mix ≈ 346 m/s。

    与已知空气声速 (~346 m/s @25°C) 交叉验证，确认物理常数正确。
    """
    c = hidden_sound_speed_v2(
        x_h2=0, x_ch4=0, x_co2=0.04, x_n2=78.0, t_c=25.0, x_o2=21.0
    )
    assert abs(c - 346.0) < 3.0, f"air sound speed {c} not close to 346 m/s"


def test_sound_speed_has_floor():
    """极端低 c_mix 输入应被限制到 ≥200 m/s。"""
    c = hidden_sound_speed_v2(x_h2=0, x_ch4=0, x_co2=100.0, x_n2=0, t_c=-300.0, x_o2=0.0)
    assert c >= 200.0


def test_sound_speed_ignores_h2_ch4():
    """x_h2/x_ch4 在本场景始终为 0，传入非零值应被忽略。"""
    c_zero = hidden_sound_speed_v2(x_h2=0, x_ch4=0, x_co2=1.0, x_n2=78.0, t_c=25.0, x_o2=21.0)
    c_nonzero = hidden_sound_speed_v2(x_h2=50.0, x_ch4=30.0, x_co2=1.0, x_n2=78.0, t_c=25.0, x_o2=21.0)
    assert c_zero == c_nonzero


# ---------------------------------------------------------------------------
# 衰减
# ---------------------------------------------------------------------------


def test_attenuation_has_alpha_o2_field():
    res = hidden_attenuation_v2(
        x_h2=0, x_ch4=0, x_co2=1.0, x_n2=78.0, t_c=25.0, p_mpa=0.5, h_rh=50.0, x_o2=21.0
    )
    assert "alpha_o2_v2" in res
    assert "f_relax_o2_eff" in res
    assert res["alpha_o2_v2"] >= 0.0


def test_attenuation_o2_negligible_at_200khz():
    """200 kHz 下 alpha_o2 ≈ 0（dry air fr,O ≈ 24 Hz/atm，远低于载波）。"""
    res = hidden_attenuation_v2(
        x_h2=0, x_ch4=0, x_co2=1.0, x_n2=78.0, t_c=25.0, p_mpa=0.5, h_rh=50.0, x_o2=21.0
    )
    # alpha_o2 应远小于 alpha_co2 和 alpha_n2
    assert res["alpha_o2_v2"] < 1e-6
    assert res["alpha_o2_v2"] < res["alpha_co2_v2"]


def test_attenuation_no_alpha_ch4_or_h2_diffusion():
    """tv3 衰减不含 alpha_ch4 / alpha_h2_diffusion 字段。"""
    res = hidden_attenuation_v2(
        x_h2=0, x_ch4=0, x_co2=1.0, x_n2=78.0, t_c=25.0, p_mpa=0.5, h_rh=50.0, x_o2=21.0
    )
    assert "alpha_ch4_v2" not in res
    assert "alpha_h2_diffusion_v2" not in res


def test_attenuation_co2_contribution_increases_with_x_co2():
    """x_CO2 增大时 alpha_co2_v2 应增加（其他条件不变）。"""
    low = hidden_attenuation_v2(
        x_h2=0, x_ch4=0, x_co2=0.5, x_n2=78.0, t_c=25.0, p_mpa=0.5, h_rh=50.0, x_o2=21.0
    )
    high = hidden_attenuation_v2(
        x_h2=0, x_ch4=0, x_co2=5.0, x_n2=78.0, t_c=25.0, p_mpa=0.5, h_rh=50.0, x_o2=21.0
    )
    assert high["alpha_co2_v2"] > low["alpha_co2_v2"]


def test_attenuation_aggregate_is_sum_of_parts():
    """alpha_true_v2 应等于各组分项之和。"""
    res = hidden_attenuation_v2(
        x_h2=0, x_ch4=0, x_co2=1.0, x_n2=78.0, t_c=25.0, p_mpa=0.5, h_rh=50.0, x_o2=21.0
    )
    parts = (
        res["alpha_classical_v2"]
        + res["alpha_co2_v2"]
        + res["alpha_n2_background_v2"]
        + res["alpha_o2_v2"]
        + res["alpha_h2o_v2"]
    )
    assert abs(res["alpha_true_v2"] - parts) < 1e-9


# ---------------------------------------------------------------------------
# 吸收 + 热导
# ---------------------------------------------------------------------------


def test_hidden_absorption_co2_nonnegative_in_range():
    for x_co2 in (0.0, 1.0, 3.0, 5.0):
        val = _hidden_absorption_co2(x_co2=x_co2, h_rh=50.0, p_mpa=0.5, t_c=25.0)
        assert val >= 0.0


def test_hidden_absorption_co2_monotone():
    a = _hidden_absorption_co2(x_co2=1.0, h_rh=50.0, p_mpa=0.5, t_c=25.0)
    b = _hidden_absorption_co2(x_co2=5.0, h_rh=50.0, p_mpa=0.5, t_c=25.0)
    assert b > a


def test_lambda_mix_o2_term_effective():
    """WMS 下 O2 组分通过 φ_ij 影响热导率（O2 与 N2 的 λ/η 略异，故结果不同）。"""
    val_with_o2 = _hidden_lambda_mix(x_co2=1.0, x_o2=21.0, x_n2=78.0, t_c=25.0)
    val_o2_as_n2 = _hidden_lambda_mix(x_co2=1.0, x_o2=0.0, x_n2=99.0, t_c=25.0)
    assert val_with_o2 != val_o2_as_n2


def test_lambda_mix_o2_n2_difference_about_2_percent():
    """O2 与 N2 热导率差约 2.3%，是 TCS 通道区分两者的物理上限。"""
    lam_o2 = _hidden_lambda_mix(x_co2=0.0, x_o2=100.0, x_n2=0.0, t_c=25.0)
    lam_n2 = _hidden_lambda_mix(x_co2=0.0, x_o2=0.0, x_n2=100.0, t_c=25.0)
    relative_diff = abs(lam_o2 - lam_n2) / lam_n2
    assert 0.01 < relative_diff < 0.04, f"relative diff {relative_diff} not in [1%, 4%]"


# ---------------------------------------------------------------------------
# main_sensor_features
# ---------------------------------------------------------------------------


@pytest.fixture
def tv3_condition():
    return {
        "x_CO2": "1.0",
        "x_O2": "20.9",
        "x_N2": "78.1",
        "T_C": "25.0",
        "P_MPa": "0.5",
        "H_RH": "50.0",
        "L_m": "0.25",
    }


def test_main_sensor_features_outputs_v_ndir_channels(tv3_condition):
    rng = random.Random(123)
    res = main_sensor_features(tv3_condition, rng)
    # tv3 场景无 CH₄，不应有 V_NDIR_CH4
    assert "V_NDIR_CH4" not in res
    assert "V_NDIR_CO2" in res
    assert "V_TCS" in res
    # tv3 不含 V_NDIR_CO（syngas 才有）
    assert "V_NDIR_CO" not in res


def test_main_sensor_features_no_v_ndir_co(tv3_condition):
    """tv3 不输出 V_NDIR_CO 通道。"""
    rng = random.Random(123)
    res = main_sensor_features(tv3_condition, rng)
    assert "V_NDIR_CO" not in res


def test_main_sensor_features_no_crosstalk(tv3_condition):
    """tv3 无光学串扰：absorption_co2_observed == absorption_co2_true。"""
    rng = random.Random(789)
    res = main_sensor_features(tv3_condition, rng)
    assert abs(res["absorption_co2_observed"] - res["absorption_co2_true"]) < 1e-9


def test_thermal_conductivity_sensor_feature_uses_x_o2(tv3_condition):
    rng = random.Random(7)
    res = thermal_conductivity_sensor_feature(tv3_condition, rng)
    assert "V_TCS" in res
    assert res["V_TCS"] > 0


# ---------------------------------------------------------------------------
# 隔离性：hydrogen_ng 物理不变
# ---------------------------------------------------------------------------


def test_hydrogen_ng_acoustic_signature_unchanged():
    """hydrogen_ng 的 hidden_sound_speed_v2 签名仍是 5 参（无 x_o2）。"""
    import inspect

    sig = inspect.signature(hg_acoustic.hidden_sound_speed_v2)
    assert list(sig.parameters) == ["x_h2", "x_ch4", "x_co2", "x_n2", "t_c"]


def test_hydrogen_ng_params_no_o2_keys():
    """hydrogen_ng PROCESSING_PARAMS_V2 不包含 O2 相关键。"""
    keys = set(hg_acoustic.PROCESSING_PARAMS_V2)
    assert "alpha_lambda_max_o2" not in keys
    assert "f_relax_o2_per_atm" not in keys


def test_tv3_params_has_o2_keys():
    keys = set(PROCESSING_PARAMS_V2)
    assert "alpha_lambda_max_o2" in keys
    assert "f_relax_o2_per_atm" in keys


def test_tv3_params_no_ch4_h2_keys():
    """tv3 PROCESSING_PARAMS_V2 不含 CH4/H2 相关键。"""
    keys = set(PROCESSING_PARAMS_V2)
    assert "alpha_lambda_max_ch4" not in keys
    assert "f_relax_ch4_base_per_atm" not in keys
    assert "k_diffusion_h2" not in keys
