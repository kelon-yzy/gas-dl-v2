"""RCDW 声学物理：O2/CO2/N2 三组分声速 + 经典/弛豫衰减。

对应方案 §5.5。与 HG 主线 ``acoustic_physics.py`` 的差异：
- 声速公式从 H2/CH4/CO2/N2 四组分改为 O2/CO2/N2 三组分。
- 衰减公式新增 O2 振动弛豫项, 删除 H2 扩散项和 CH4 弛豫项。
- ``main_sensor_features`` / ``thermal_conductivity_sensor_feature``
  签名改为读取 ``x_O2 / x_CO2 / x_N2``, 删除 ``V_NDIR_CH4`` 相关计算。

声速模型版本: linear_mixing_v1（摩尔质量加权线性近似 + 0.6 K⁻¹ 温度修正）。
未来可替换为 c = sqrt(γRT/M_mix) + 弛豫修正；替换时同步更新
``acoustic_model_metadata.model`` 字段以便审计。
"""

from __future__ import annotations

import math
from typing import Any

from rcdw.sim.generation.gas_state import h2o_mole_percent_from_rh


# 通用处理参数（NDIR / TCS 响应曲线占位，与 HG 同源；CH4 相关字段已删除）。
PROCESSING_PARAMS = {
    "t_delay_s": 0.00008,
    "optical_baseline_co2_init": 2.5,
    "optical_saturation_absorbance": 4.0,
    "thermal_baseline_init": 1.1,
    "tcs_response_slope": 15.0,
    "tcs_lambda_offset": 0.026,
    "tcs_temperature_response": 0.004,
}

# 衰减模型物理参数（RCDW v1）。
# - O2 振动弛豫：参考 30-60 kHz 文献区间, 暂用 50000 Hz @ 1 atm,
#   alpha_lambda_max = 0.002（数量级与 N2 同量, 略小于 CO2/CH4）。
#   方案 §13.3 风险标注：实测校核后可能调整。
PROCESSING_PARAMS_V2 = {
    "alpha_classical_K_ref": 1.84e-11,
    "alpha_lambda_max_co2": 0.12,
    "f_relax_co2_per_atm": 28000.0,
    "k_h2o_to_f_relax_co2": 0.015,
    "alpha_lambda_max_n2": 0.004,
    "f_relax_n2_per_atm": 65000.0,
    "alpha_lambda_max_o2": 0.002,   # TBD: 实测校核
    "f_relax_o2_per_atm": 50000.0,  # TBD: 实测校核
    "alpha_lambda_max_h2o": 0.01,
    "f_relax_h2o_per_atm": 100000.0,
    "acoustic_excitation_frequency_hz": 40000.0,
}

ACOUSTIC_MODEL_NAME = "linear_mixing_v1"

_PRESSURE_MPA_TO_ATM = 1.0 / 0.101325
_T0_K = 293.15
_NP_TO_DB = 8.686

# 25°C 时各气体的声速参考值 (m/s)。
# - O2: 329.5 m/s (NIST Chemistry WebBook @ 25°C, 1 atm), 方案 §4.4 锁定。
# - CO2: 268 m/s (与 HG 主线一致)。
# - N2: 353 m/s (与 HG 主线一致)。
_SPEED_O2_MS = 329.5
_SPEED_CO2_MS = 268.0
_SPEED_N2_MS = 353.0


def acoustic_model_metadata() -> dict[str, Any]:
    """RCDW 声学模型元数据，落入 manifest 的 ``acoustic_model_metadata``。"""
    return {
        "model": ACOUSTIC_MODEL_NAME,
        "components": ["O2", "CO2", "N2"],
        "sound_speed_reference_m_per_s": {
            "O2": _SPEED_O2_MS,
            "CO2": _SPEED_CO2_MS,
            "N2": _SPEED_N2_MS,
        },
        "sound_speed_reference_source": {
            "O2": "NIST Chemistry WebBook @ 25C, 1 atm",
            "CO2": "shared with HG src/sim",
            "N2": "shared with HG src/sim",
        },
        "temperature_coefficient_per_k": 0.6,
        "attenuation_terms": [
            "classical",
            "co2_vibrational_relaxation",
            "n2_vibrational_relaxation",
            "o2_vibrational_relaxation",
            "h2o_vibrational_relaxation",
        ],
        "o2_relaxation_source": "placeholder_v1",  # 方案 §13.3 TBD
    }


def rcdw_sound_speed(
    x_o2: float, x_co2: float, x_n2: float, t_c: float
) -> float:
    """三组分声速线性混合 + 温度修正。

    c_mix = x_O2_frac * c_O2 + x_CO2_frac * c_CO2 + x_N2_frac * c_N2 + 0.6*(T - 25)

    Args:
        x_o2, x_co2, x_n2: 组分摩尔百分比 (%), 应满足 sum ≈ 100。
        t_c: 温度 (°C)。

    Returns:
        声速 (m/s), clamp 下限 200.0。
    """
    x_o2_frac = max(0.0, x_o2) / 100.0
    x_co2_frac = max(0.0, x_co2) / 100.0
    x_n2_frac = max(0.0, x_n2) / 100.0
    c_mix = (
        x_o2_frac * _SPEED_O2_MS
        + x_co2_frac * _SPEED_CO2_MS
        + x_n2_frac * _SPEED_N2_MS
    )
    c_mix += 0.6 * (t_c - 25.0)
    return max(c_mix, 200.0)


def rcdw_attenuation(
    x_o2: float,
    x_co2: float,
    x_n2: float,
    t_c: float,
    p_mpa: float,
    h_rh: float,
    *,
    c_mix: float | None = None,
    f_hz: float | None = None,
) -> dict[str, float]:
    """三组分声吸收：经典 + CO2/N2/O2/H2O 弛豫。

    与 HG 主线相比：删除 H2 扩散项 (RCDW 无 H2) 和 CH4 弛豫项, 新增 O2 弛豫。
    """
    params = PROCESSING_PARAMS_V2
    if f_hz is None:
        f_hz = params["acoustic_excitation_frequency_hz"]
    if c_mix is None:
        c_mix = rcdw_sound_speed(x_o2, x_co2, x_n2, t_c)
    c_mix = max(c_mix, 200.0)

    p_atm = max(p_mpa, 1e-4) * _PRESSURE_MPA_TO_ATM
    t_k = t_c + 273.15
    h_w_pct = h2o_mole_percent_from_rh(t_c, p_mpa, h_rh)

    # 经典 Stokes-Kirchhoff 吸收
    alpha_classical_db = (
        params["alpha_classical_K_ref"]
        * (f_hz ** 2)
        * (1.0 / max(p_atm, 1e-4))
        * math.sqrt(t_k / _T0_K)
    )
    alpha_classical_npm = alpha_classical_db / _NP_TO_DB

    # CO2 振动弛豫（受 H2O 加速）
    f_r_co2 = (
        params["f_relax_co2_per_atm"]
        * p_atm
        * (1.0 + params["k_h2o_to_f_relax_co2"] * h_w_pct)
    )
    alpha_lambda_co2 = (
        params["alpha_lambda_max_co2"]
        * (max(0.0, x_co2) / 100.0)
        * 2.0
        * f_hz
        * f_r_co2
        / (f_hz ** 2 + f_r_co2 ** 2)
    )
    alpha_co2_npm = alpha_lambda_co2 * f_hz / c_mix

    # N2 振动弛豫
    x_n2_frac = max(0.0, x_n2) / 100.0
    f_r_n2 = params["f_relax_n2_per_atm"] * p_atm
    alpha_lambda_n2 = (
        params["alpha_lambda_max_n2"]
        * x_n2_frac
        * 2.0
        * f_hz
        * f_r_n2
        / (f_hz ** 2 + f_r_n2 ** 2)
    )
    alpha_n2_npm = alpha_lambda_n2 * f_hz / c_mix

    # O2 振动弛豫（RCDW 新增）
    x_o2_frac = max(0.0, x_o2) / 100.0
    f_r_o2 = params["f_relax_o2_per_atm"] * p_atm
    alpha_lambda_o2 = (
        params["alpha_lambda_max_o2"]
        * x_o2_frac
        * 2.0
        * f_hz
        * f_r_o2
        / (f_hz ** 2 + f_r_o2 ** 2)
    )
    alpha_o2_npm = alpha_lambda_o2 * f_hz / c_mix

    # H2O 振动弛豫
    h_w_frac = max(0.0, h_w_pct) / 100.0
    f_r_h2o = params["f_relax_h2o_per_atm"] * p_atm
    alpha_lambda_h2o = (
        params["alpha_lambda_max_h2o"]
        * h_w_frac
        * 2.0
        * f_hz
        * f_r_h2o
        / (f_hz ** 2 + f_r_h2o ** 2)
    )
    alpha_h2o_npm = alpha_lambda_h2o * f_hz / c_mix

    alpha_true = max(
        0.0,
        alpha_classical_npm
        + alpha_co2_npm
        + alpha_n2_npm
        + alpha_o2_npm
        + alpha_h2o_npm,
    )

    return {
        "alpha_true_v2": alpha_true,
        "alpha_classical_v2": alpha_classical_npm,
        "alpha_co2_v2": alpha_co2_npm,
        "alpha_n2_background_v2": alpha_n2_npm,
        "alpha_o2_v2": alpha_o2_npm,
        "alpha_h2o_v2": alpha_h2o_npm,
        "f_relax_co2_eff": f_r_co2,
        "f_relax_n2_eff": f_r_n2,
        "f_relax_o2_eff": f_r_o2,
        "f_relax_h2o_eff": f_r_h2o,
        "h_w_pct_eff": h_w_pct,
        "c_mix_used": c_mix,
    }


def main_sensor_features(
    condition: dict[str, str],
    rng,
    *,
    optical_absorption: dict[str, object] | None = None,
) -> dict[str, float | bool | object]:
    """RCDW 主传感器特征（NDIR_CO2 + TCS + 超声）。

    `optical_absorption` 来自 HITRAN 后端（含 absorption_co2_observed 等）；
    若为 None 则使用 empirical fallback。
    """
    x_o2 = float(condition["x_O2"])
    x_co2 = float(condition["x_CO2"])
    x_n2 = float(condition["x_N2"])
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    h_rh = float(condition["H_RH"])
    l_m = float(condition["L_m"])

    sound_speed = rcdw_sound_speed(x_o2, x_co2, x_n2, t_c)
    attenuation = rcdw_attenuation(
        x_o2, x_co2, x_n2, t_c, p_mpa, h_rh, c_mix=sound_speed
    )
    tof = (
        l_m / sound_speed
        + PROCESSING_PARAMS["t_delay_s"]
        + rng.gauss(0.0, 0.000003)
    )
    amp = max(
        0.005,
        math.exp(-attenuation["alpha_true_v2"] * l_m) + rng.gauss(0.0, 0.0008),
    )
    # 中心频率漂移（CO2 拉低, 温度提升）
    f_peak = (
        40000.0 - 18.0 * x_co2 + 2.0 * (t_c - 20.0) + rng.gauss(0.0, 65.0)
    )
    a_fft_max = amp * (900.0 + rng.gauss(0.0, 20.0))

    if optical_absorption is None:
        absorption_co2 = _empirical_absorption_co2(x_co2, h_rh, p_mpa, t_c)
        absorption_h2o = _empirical_absorption_h2o(h_rh, p_mpa, t_c)
        from rcdw.sim.generation.optical_crosstalk import apply_optical_crosstalk

        optical_absorption_dict: dict[str, object] = dict(
            apply_optical_crosstalk(
                absorption_co2=absorption_co2, absorption_h2o=absorption_h2o
            )
        )
    else:
        optical_absorption_dict = optical_absorption

    lambda_true = _hidden_lambda_mix(x_o2, x_co2, x_n2, t_c)

    optical_drift_co2 = (
        0.0007 * (h_rh - 50.0) + 0.004 * (p_mpa - 1.0) + rng.gauss(0.0, 0.004)
    )
    thermal_drift = _thermal_baseline_drift(t_c, p_mpa, rng)

    optical_baseline_co2_now = (
        PROCESSING_PARAMS["optical_baseline_co2_init"]
        + optical_drift_co2
        + rng.gauss(0.0, 0.006)
    )

    absorption_co2_observed_value = float(
        optical_absorption_dict["absorption_co2_observed"]  # type: ignore[arg-type]
    )
    v_ndir_co2 = max(
        0.1,
        optical_baseline_co2_now * math.exp(-absorption_co2_observed_value)
        + rng.gauss(0.0, 0.008),
    )
    v_tcs = _tcs_voltage(lambda_true, t_c, thermal_drift, rng)

    return {
        "TOF": tof,
        "Amp": amp,
        "f_peak": f_peak,
        "A_fft_max": a_fft_max,
        "V_NDIR_CO2": v_ndir_co2,
        "V_TCS": v_tcs,
        "ndir_co2_saturated": absorption_co2_observed_value
        > PROCESSING_PARAMS["optical_saturation_absorbance"],
        **optical_absorption_dict,
        "optical_baseline_drift_co2_observed": optical_drift_co2,
        "thermal_baseline_drift_observed": thermal_drift,
    }


def rcdw_thermal_conductivity_sensor_feature(
    condition: dict[str, str], rng
) -> dict[str, float]:
    """RCDW 三组分 TCS 电压（不依赖 NDIR/声学）。"""
    x_o2 = float(condition["x_O2"])
    x_co2 = float(condition["x_CO2"])
    x_n2 = float(condition["x_N2"])
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    lambda_true = _hidden_lambda_mix(x_o2, x_co2, x_n2, t_c)
    thermal_drift = _thermal_baseline_drift(t_c, p_mpa, rng)
    return {
        "V_TCS": _tcs_voltage(lambda_true, t_c, thermal_drift, rng),
        "thermal_baseline_drift_observed": thermal_drift,
    }


# ---- 内部函数 ----


def _empirical_absorption_co2(x_co2: float, h_rh: float, p_mpa: float, t_c: float) -> float:
    """Empirical fallback：仅在未指定 HITRAN 后端时使用。"""
    return 0.045 * x_co2 + 0.0006 * h_rh + 0.012 * p_mpa + 0.00015 * (t_c - 25.0)


def _empirical_absorption_h2o(h_rh: float, p_mpa: float, t_c: float) -> float:
    return 0.003 * h_rh + 0.005 * p_mpa + 0.0001 * (t_c - 25.0)


def _hidden_lambda_mix(x_o2: float, x_co2: float, x_n2: float, t_c: float) -> float:
    """三组分热导率线性混合 (W/m·K @ 25°C 附近)。

    参考值: O2 ≈ 0.026, CO2 ≈ 0.017, N2 ≈ 0.026 W/m·K @ 300K。
    """
    x_o2_frac = max(0.0, x_o2) / 100.0
    x_co2_frac = max(0.0, x_co2) / 100.0
    x_n2_frac = max(0.0, x_n2) / 100.0
    lambda_mix = (
        0.026 * x_o2_frac + 0.017 * x_co2_frac + 0.026 * x_n2_frac
    )
    lambda_mix += 0.00002 * (t_c - 25.0)
    return lambda_mix


def _thermal_baseline_drift(t_c: float, p_mpa: float, rng) -> float:
    return 0.002 * (t_c - 25.0) + 0.004 * (p_mpa - 1.0) + rng.gauss(0.0, 0.003)


def _tcs_voltage(
    lambda_true: float, t_c: float, thermal_drift: float, rng
) -> float:
    thermal_baseline_now = PROCESSING_PARAMS["thermal_baseline_init"] + thermal_drift
    return (
        thermal_baseline_now
        + PROCESSING_PARAMS["tcs_response_slope"]
        * (lambda_true - PROCESSING_PARAMS["tcs_lambda_offset"])
        + PROCESSING_PARAMS["tcs_temperature_response"] * (t_c - 20.0)
        + rng.gauss(0.0, 0.006)
    )
