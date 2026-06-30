"""测试 RCDW 声学物理：三组分声速 + 衰减分项 + TCS 电压。

对应方案 §5.5 / §11.1。
"""
from __future__ import annotations

import math
import random

import pytest

from rcdw.sim.generation.acoustic_physics import (
    ACOUSTIC_MODEL_NAME,
    PROCESSING_PARAMS_V2,
    acoustic_model_metadata,
    main_sensor_features,
    rcdw_attenuation,
    rcdw_sound_speed,
    rcdw_thermal_conductivity_sensor_feature,
)


def test_pure_n2_sound_speed_close_to_353():
    """100% N2 @ 25°C 应接近 353 m/s。"""
    c = rcdw_sound_speed(0.0, 0.0, 100.0, 25.0)
    assert abs(c - 353.0) < 1.0


def test_pure_o2_sound_speed_close_to_329_5():
    c = rcdw_sound_speed(100.0, 0.0, 0.0, 25.0)
    assert abs(c - 329.5) < 1.0


def test_pure_co2_sound_speed_close_to_268():
    c = rcdw_sound_speed(0.0, 100.0, 0.0, 25.0)
    assert abs(c - 268.0) < 1.0


def test_sound_speed_temperature_coefficient():
    """T 提升 10°C 应使声速增加约 6 m/s（0.6 K⁻¹）。"""
    c_25 = rcdw_sound_speed(20.0, 10.0, 70.0, 25.0)
    c_35 = rcdw_sound_speed(20.0, 10.0, 70.0, 35.0)
    assert abs((c_35 - c_25) - 6.0) < 0.1


def test_sound_speed_clamp_lower_bound():
    """极端负温度（非物理）应被 clamp 到 200 m/s。"""
    c = rcdw_sound_speed(0.0, 100.0, 0.0, -1000.0)
    assert c == 200.0


def test_sound_speed_co2_monotonic_decrease():
    """CO2 增加（其他组分等比例减少）声速应下降（CO2 分子量大）。"""
    c_low = rcdw_sound_speed(10.0, 0.0, 90.0, 25.0)
    c_high = rcdw_sound_speed(10.0, 20.0, 70.0, 25.0)
    assert c_high < c_low


def test_attenuation_returns_all_terms():
    result = rcdw_attenuation(
        x_o2=20.0, x_co2=5.0, x_n2=75.0,
        t_c=25.0, p_mpa=0.1, h_rh=50.0,
    )
    required_keys = {
        "alpha_true_v2",
        "alpha_classical_v2",
        "alpha_co2_v2",
        "alpha_n2_background_v2",
        "alpha_o2_v2",
        "alpha_h2o_v2",
        "f_relax_co2_eff",
        "f_relax_n2_eff",
        "f_relax_o2_eff",
        "f_relax_h2o_eff",
        "h_w_pct_eff",
        "c_mix_used",
    }
    assert required_keys <= result.keys()


def test_attenuation_no_ch4_or_h2_terms():
    """RCDW 衰减不应含 CH4 弛豫或 H2 扩散字段。"""
    result = rcdw_attenuation(
        x_o2=10.0, x_co2=5.0, x_n2=85.0, t_c=25.0, p_mpa=0.1, h_rh=40.0
    )
    forbidden = {"alpha_ch4_v2", "alpha_h2_diffusion_v2", "f_relax_ch4_eff"}
    assert not (forbidden & result.keys())


def test_attenuation_co2_monotonic():
    """CO2 增加, alpha_co2 应单调增加。"""
    low = rcdw_attenuation(0.0, 1.0, 99.0, 25.0, 0.1, 40.0)
    high = rcdw_attenuation(0.0, 15.0, 85.0, 25.0, 0.1, 40.0)
    assert high["alpha_co2_v2"] > low["alpha_co2_v2"]


def test_attenuation_o2_monotonic():
    """O2 增加, alpha_o2 应单调增加。"""
    low = rcdw_attenuation(1.0, 5.0, 94.0, 25.0, 0.1, 40.0)
    high = rcdw_attenuation(20.0, 5.0, 75.0, 25.0, 0.1, 40.0)
    assert high["alpha_o2_v2"] > low["alpha_o2_v2"]


def test_attenuation_o2_relaxation_marked_tbd():
    """O2 弛豫参数标 TBD，但需有数值（方案 §13.3）。"""
    assert PROCESSING_PARAMS_V2["alpha_lambda_max_o2"] > 0.0
    assert PROCESSING_PARAMS_V2["f_relax_o2_per_atm"] > 0.0
    meta = acoustic_model_metadata()
    assert meta["o2_relaxation_source"] == "placeholder_v1"


def test_acoustic_model_metadata_version():
    meta = acoustic_model_metadata()
    assert meta["model"] == ACOUSTIC_MODEL_NAME == "linear_mixing_v1"
    assert meta["components"] == ["O2", "CO2", "N2"]
    speeds = meta["sound_speed_reference_m_per_s"]
    assert speeds == {"O2": 329.5, "CO2": 268.0, "N2": 353.0}
    assert "o2_vibrational_relaxation" in meta["attenuation_terms"]


def test_thermal_conductivity_uses_three_components():
    """TCS 电压计算应读取 x_O2 / x_CO2 / x_N2，不读 x_H2 / x_CH4。"""
    rng = random.Random(0)
    condition = {
        "x_O2": "10.0",
        "x_CO2": "5.0",
        "x_N2": "85.0",
        "T_C": "25.0",
        "P_MPa": "0.1",
    }
    feat = rcdw_thermal_conductivity_sensor_feature(condition, rng)
    assert "V_TCS" in feat
    assert "thermal_baseline_drift_observed" in feat
    assert math.isfinite(feat["V_TCS"])


def test_main_sensor_features_with_empirical_optical():
    """未指定 HITRAN 后端时, 走 empirical fallback 应能跑通。"""
    rng = random.Random(0)
    condition = {
        "x_O2": "10.0",
        "x_CO2": "5.0",
        "x_N2": "85.0",
        "T_C": "25.0",
        "P_MPa": "0.1",
        "H_RH": "40.0",
        "L_m": "0.5",
    }
    feat = main_sensor_features(condition, rng)
    required = {
        "TOF",
        "Amp",
        "V_NDIR_CO2",
        "V_TCS",
        "absorption_co2_observed",
        "thermal_baseline_drift_observed",
    }
    assert required <= feat.keys()
    # 不应出现 H2/CH4 相关字段
    assert "V_NDIR_CH4" not in feat
    assert "ndir_ch4_saturated" not in feat


def test_attenuation_h2o_via_rh():
    """RH 提升, H2O 弛豫吸收应增加。"""
    low = rcdw_attenuation(0.0, 5.0, 95.0, 25.0, 0.1, 10.0)
    high = rcdw_attenuation(0.0, 5.0, 95.0, 25.0, 0.1, 80.0)
    assert high["alpha_h2o_v2"] > low["alpha_h2o_v2"]
    assert high["h_w_pct_eff"] > low["h_w_pct_eff"]


def test_processing_params_no_ch4_or_h2_fields():
    """RCDW 衰减参数表不应含 CH4 / H2 相关字段。"""
    forbidden = {
        "alpha_lambda_max_ch4",
        "f_relax_ch4_base_per_atm",
        "f_relax_ch4_slope_per_atm",
        "k_diffusion_h2",
    }
    assert not (forbidden & PROCESSING_PARAMS_V2.keys())
