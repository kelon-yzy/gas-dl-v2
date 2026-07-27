from __future__ import annotations

import math

from sg.sim.generation.optical_crosstalk import apply_optical_crosstalk
from sg.sim.generation.gas_state import h2o_mole_percent_from_rh


PROCESSING_PARAMS = {
    "t_delay_s": 0.00008,
    "optical_baseline_ch4_init": 2.5,
    "optical_baseline_co2_init": 2.5,
    "optical_saturation_absorbance": 4.0,
    "thermal_baseline_init": 1.1,
    "tcs_response_slope": 15.0,
    "tcs_lambda_offset": 0.026,
    "tcs_temperature_response": 0.004,
}

PROCESSING_PARAMS_V2 = {
    "alpha_classical_K_ref": 1.84e-11,
    "alpha_lambda_max_co2": 0.12,
    "f_relax_co2_per_atm": 28000.0,
    "k_h2o_to_f_relax_co2": 0.015,
    "alpha_lambda_max_ch4": 0.034,
    "f_relax_ch4_base_per_atm": 30000.0,
    "f_relax_ch4_slope_per_atm": 120000.0,
    "k_diffusion_h2": 1.6e-3,
    "alpha_lambda_max_n2": 0.004,
    # Bass 1990 / ISO 9613-1 dry air: fr,N ≈ 9 Hz/atm（旧值 65000 错误，差 ~4 个数量级）
    "f_relax_n2_per_atm": 9.0,
    "alpha_lambda_max_h2o": 0.01,
    "f_relax_h2o_per_atm": 100000.0,
    "acoustic_excitation_frequency_hz": 200000.0,
}

_PRESSURE_MPA_TO_ATM = 1.0 / 0.101325
_T0_K = 293.15
_NP_TO_DB = 8.686
_R_GAS = 8.314  # J/(mol·K)
# 纯组分摩尔质量 (kg/mol) 与理想气体定压摩尔热容 (J/mol/K) @298.15K
# 来源：NIST WebBook (298.15K, 1 bar)。纯组分声速由 c=sqrt(γ·R·T/M) 在 298.15K
# 公式自洽计算（不查表）。旧 _SPEED_*_MS 查表常数已移除，其值约对应 293K，与本
# 公式（298.15K）不一致；详见 docs/物理模型严格化实施计划.md §2.2 审查修正。
_GAS_M = {"H2": 0.002016, "CH4": 0.01604, "CO2": 0.04401, "N2": 0.02801}
_GAS_CP = {"H2": 28.84, "CH4": 35.64, "CO2": 37.13, "N2": 29.12}


def hidden_sound_speed_v2(x_h2: float, x_ch4: float, x_co2: float, x_n2: float, t_c: float) -> float:
    """理想气体混合声速 c = sqrt(γ_mix · R · T / M_mix)。

    M_mix 与 cp_mix 按摩尔分数加权，γ_mix = cp_mix / (cp_mix - R)。
    替代旧的纯组分声速线性加权 + 温度修正工程代理。
    """
    fracs = {
        "H2": max(0.0, x_h2) / 100.0,
        "CH4": max(0.0, x_ch4) / 100.0,
        "CO2": max(0.0, x_co2) / 100.0,
        "N2": max(0.0, x_n2) / 100.0,
    }
    total = sum(fracs.values())
    if total <= 0.0:
        return 200.0
    fracs = {k: v / total for k, v in fracs.items()}
    m_mix = sum(fracs[k] * _GAS_M[k] for k in fracs)            # kg/mol
    cp_mix = sum(fracs[k] * _GAS_CP[k] for k in fracs)           # J/mol/K
    gamma_mix = cp_mix / max(cp_mix - _R_GAS, 1e-9)
    t_k = max(t_c + 273.15, 1.0)  # 防低于绝对零度（不物理输入），保 floor 行为
    c_mix = math.sqrt(gamma_mix * _R_GAS * t_k / max(m_mix, 1e-12))
    return max(c_mix, 200.0)


def hidden_attenuation_v2(
    x_h2: float,
    x_ch4: float,
    x_co2: float,
    x_n2: float,
    t_c: float,
    p_mpa: float,
    h_rh: float,
    *,
    c_mix: float | None = None,
    f_hz: float | None = None,
) -> dict[str, float]:
    params = PROCESSING_PARAMS_V2
    if f_hz is None:
        f_hz = params["acoustic_excitation_frequency_hz"]
    if c_mix is None:
        c_mix = hidden_sound_speed_v2(x_h2, x_ch4, x_co2, x_n2, t_c)
    c_mix = max(c_mix, 200.0)

    p_atm = max(p_mpa, 1e-4) * _PRESSURE_MPA_TO_ATM
    p_pa = max(p_mpa, 1e-4) * 1e6
    t_k = t_c + 273.15
    h_w_pct = _h2o_mole_pct(t_c, p_mpa, h_rh)

    alpha_classical_db = params["alpha_classical_K_ref"] * (f_hz**2) * (1.0 / max(p_atm, 1e-4)) * math.sqrt(t_k / _T0_K)
    alpha_classical_npm = alpha_classical_db / _NP_TO_DB

    f_r_co2 = params["f_relax_co2_per_atm"] * p_atm * (1.0 + params["k_h2o_to_f_relax_co2"] * h_w_pct)
    alpha_lambda_co2 = (
        params["alpha_lambda_max_co2"]
        * (max(0.0, x_co2) / 100.0)
        * 2.0
        * f_hz
        * f_r_co2
        / (f_hz**2 + f_r_co2**2)
    )
    alpha_co2_npm = alpha_lambda_co2 * f_hz / c_mix

    x_ch4_frac = max(0.0, x_ch4) / 100.0
    f_r_ch4 = (params["f_relax_ch4_base_per_atm"] + params["f_relax_ch4_slope_per_atm"] * x_ch4_frac) * p_atm
    alpha_lambda_ch4 = (
        params["alpha_lambda_max_ch4"]
        * x_ch4_frac
        * 2.0
        * f_hz
        * f_r_ch4
        / (f_hz**2 + f_r_ch4**2)
    )
    alpha_ch4_npm = alpha_lambda_ch4 * f_hz / c_mix

    x_h2_frac = max(0.0, x_h2) / 100.0
    alpha_diff_npm = (
        params["k_diffusion_h2"]
        * x_h2_frac
        * max(0.0, 1.0 - x_h2_frac)
        * (f_hz**2)
        / max(p_pa, 1e3)
        / max(c_mix**3, 1e6)
    )

    x_n2_frac = max(0.0, x_n2) / 100.0
    f_r_n2 = params["f_relax_n2_per_atm"] * p_atm
    alpha_lambda_n2 = (
        params["alpha_lambda_max_n2"]
        * x_n2_frac
        * 2.0
        * f_hz
        * f_r_n2
        / (f_hz**2 + f_r_n2**2)
    )
    alpha_n2_npm = alpha_lambda_n2 * f_hz / c_mix

    h_w_frac = max(0.0, h_w_pct) / 100.0
    f_r_h2o = params["f_relax_h2o_per_atm"] * p_atm
    alpha_lambda_h2o = (
        params["alpha_lambda_max_h2o"]
        * h_w_frac
        * 2.0
        * f_hz
        * f_r_h2o
        / (f_hz**2 + f_r_h2o**2)
    )
    alpha_h2o_npm = alpha_lambda_h2o * f_hz / c_mix
    alpha_true = max(0.0, alpha_classical_npm + alpha_co2_npm + alpha_ch4_npm + alpha_diff_npm + alpha_n2_npm + alpha_h2o_npm)

    return {
        "alpha_true_v2": alpha_true,
        "alpha_classical_v2": alpha_classical_npm,
        "alpha_co2_v2": alpha_co2_npm,
        "alpha_ch4_v2": alpha_ch4_npm,
        "alpha_h2_diffusion_v2": alpha_diff_npm,
        "alpha_n2_background_v2": alpha_n2_npm,
        "alpha_h2o_v2": alpha_h2o_npm,
        "f_relax_co2_eff": f_r_co2,
        "f_relax_ch4_eff": f_r_ch4,
        "f_relax_n2_eff": f_r_n2,
        "f_relax_h2o_eff": f_r_h2o,
        "h_w_pct_eff": h_w_pct,
        "c_mix_used": c_mix,
    }


def main_sensor_features(
    condition: dict[str, str],
    rng,
    *,
    optical_absorption: dict[str, object] | None = None,
    f_hz: float | None = None,
) -> dict[str, float | bool | object]:
    x_h2 = float(condition["x_H2"])
    x_ch4 = float(condition["x_CH4"])
    x_co2 = float(condition["x_CO2"])
    x_n2 = float(condition["x_N2"])
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    h_rh = float(condition["H_RH"])
    l_m = float(condition["L_m"])

    if f_hz is None:
        f_hz = PROCESSING_PARAMS_V2["acoustic_excitation_frequency_hz"]
    sound_speed = hidden_sound_speed_v2(x_h2, x_ch4, x_co2, x_n2, t_c)
    attenuation = hidden_attenuation_v2(x_h2, x_ch4, x_co2, x_n2, t_c, p_mpa, h_rh, c_mix=sound_speed, f_hz=f_hz)
    tof = l_m / sound_speed + PROCESSING_PARAMS["t_delay_s"] + rng.gauss(0.0, 0.000003)
    amp = max(0.005, math.exp(-attenuation["alpha_true_v2"] * l_m) + rng.gauss(0.0, 0.0008))
    f_peak = f_hz + 23.0 * x_h2 - 18.0 * x_co2 + 2.0 * (t_c - 20.0) + rng.gauss(0.0, 65.0)
    a_fft_max = amp * (900.0 + rng.gauss(0.0, 20.0))

    if optical_absorption is None:
        absorption_ch4 = _hidden_absorption_ch4(x_ch4, h_rh, p_mpa, t_c)
        absorption_co2 = _hidden_absorption_co2(x_co2, h_rh, p_mpa, t_c)
        optical_absorption = apply_optical_crosstalk(absorption_ch4=absorption_ch4, absorption_co2=absorption_co2)
    lambda_true = _hidden_lambda_mix(x_h2, x_ch4, x_co2, x_n2, t_c)

    optical_drift_ch4 = 0.0007 * (h_rh - 50.0) + 0.004 * (p_mpa - 1.0) + rng.gauss(0.0, 0.004)
    optical_drift_co2 = 0.0007 * (h_rh - 50.0) + 0.004 * (p_mpa - 1.0) + rng.gauss(0.0, 0.004)
    thermal_drift = _thermal_baseline_drift(t_c, p_mpa, rng)

    optical_baseline_ch4_now = PROCESSING_PARAMS["optical_baseline_ch4_init"] + optical_drift_ch4 + rng.gauss(0.0, 0.006)
    optical_baseline_co2_now = PROCESSING_PARAMS["optical_baseline_co2_init"] + optical_drift_co2 + rng.gauss(0.0, 0.006)

    v_ndir_ch4 = max(
        0.1,
        optical_baseline_ch4_now * math.exp(-optical_absorption["absorption_ch4_observed"]) + rng.gauss(0.0, 0.008),
    )
    v_ndir_co2 = max(
        0.1,
        optical_baseline_co2_now * math.exp(-optical_absorption["absorption_co2_observed"]) + rng.gauss(0.0, 0.008),
    )
    v_tcs = _tcs_voltage(lambda_true, t_c, thermal_drift, rng)

    return {
        "TOF": tof,
        "Amp": amp,
        "f_peak": f_peak,
        "A_fft_max": a_fft_max,
        "V_NDIR_CH4": v_ndir_ch4,
        "V_NDIR_CO2": v_ndir_co2,
        "V_TCS": v_tcs,
        "ndir_ch4_saturated": optical_absorption["absorption_ch4_observed"] > PROCESSING_PARAMS["optical_saturation_absorbance"],
        "ndir_co2_saturated": optical_absorption["absorption_co2_observed"] > PROCESSING_PARAMS["optical_saturation_absorbance"],
        **optical_absorption,
        "optical_baseline_drift_ch4_observed": optical_drift_ch4,
        "optical_baseline_drift_co2_observed": optical_drift_co2,
        "thermal_baseline_drift_observed": thermal_drift,
    }


def thermal_conductivity_sensor_feature(condition: dict[str, str], rng) -> dict[str, float]:
    """Compute the TCS equilibrium feature without NDIR or acoustic side products."""
    x_h2 = float(condition["x_H2"])
    x_ch4 = float(condition["x_CH4"])
    x_co2 = float(condition["x_CO2"])
    x_n2 = float(condition["x_N2"])
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    lambda_true = _hidden_lambda_mix(x_h2, x_ch4, x_co2, x_n2, t_c)
    thermal_drift = _thermal_baseline_drift(t_c, p_mpa, rng)
    return {
        "V_TCS": _tcs_voltage(lambda_true, t_c, thermal_drift, rng),
        "thermal_baseline_drift_observed": thermal_drift,
    }


def _h2o_mole_pct(t_c: float, p_mpa: float, h_rh: float) -> float:
    return h2o_mole_percent_from_rh(t_c, p_mpa, h_rh)


def _hidden_absorption_ch4(x_ch4: float, h_rh: float, p_mpa: float, t_c: float) -> float:
    return 0.008 * x_ch4 + 0.0008 * h_rh + 0.015 * p_mpa + 0.0002 * (t_c - 25.0)


def _hidden_absorption_co2(x_co2: float, h_rh: float, p_mpa: float, t_c: float) -> float:
    return 0.045 * x_co2 + 0.0006 * h_rh + 0.012 * p_mpa + 0.00015 * (t_c - 25.0)


# 纯组分热导率 λ (W/(m·K)) 与粘度 η (Pa·s) @298.15K，用于 WMS 混合规则
# 来源：NIST WebBook 298.15K（审查修正 2026-07-03）
_GAS_LAMBDA = {"H2": 0.1858, "CH4": 0.0340, "CO2": 0.0166, "N2": 0.0258}
_GAS_ETA = {"H2": 8.90e-6, "CH4": 1.108e-5, "CO2": 1.491e-5, "N2": 1.781e-5}
# 各组分热导率温度幂律指数（NIST 拟合，15~35°C 范围）
_LAMBDA_T_EXPONENT = {"H2": 0.78, "CH4": 0.82, "CO2": 0.87, "N2": 0.77}


def _wilke_phi(i: str, j: str) -> float:
    """Wilke 粘度混合函数 φ_ij（Mason-Saxena 热导混合规则共用）。"""
    eta_ratio = (_GAS_ETA[i] / _GAS_ETA[j]) ** 0.5
    m_ratio = (_GAS_M[j] / _GAS_M[i]) ** 0.25
    num = (1.0 + eta_ratio * m_ratio) ** 2
    den = math.sqrt(8.0 * (1.0 + _GAS_M[i] / _GAS_M[j]))
    return num / den


def _lambda_at_t(gas: str, t_c: float) -> float:
    """纯组分热导率的温度修正（逐组分幂律）。"""
    return _GAS_LAMBDA[gas] * ((t_c + 273.15) / 298.15) ** _LAMBDA_T_EXPONENT[gas]


def _hidden_lambda_mix(x_h2: float, x_ch4: float, x_co2: float, x_n2: float, t_c: float) -> float:
    """Wassiljewa-Mason-Saxena 混合热导率 (W/(m·K))。

    λ_m = Σ_i (y_i · λ_i(T) / Σ_j y_j · φ_ij)，φ_ij 用 Wilke 形式。
    逐组分幂律温度修正后再混合。替代旧线性经验公式。
    """
    fracs = {
        "H2": max(0.0, x_h2) / 100.0,
        "CH4": max(0.0, x_ch4) / 100.0,
        "CO2": max(0.0, x_co2) / 100.0,
        "N2": max(0.0, x_n2) / 100.0,
    }
    total = sum(fracs.values())
    if total <= 0.0:
        return 0.0258
    fracs = {k: v / total for k, v in fracs.items()}
    lam_mix = 0.0
    for i, yi in fracs.items():
        denom = sum(fracs[j] * _wilke_phi(i, j) for j in fracs)
        lam_mix += yi * _lambda_at_t(i, t_c) / max(denom, 1e-12)
    return lam_mix


def _thermal_baseline_drift(t_c: float, p_mpa: float, rng) -> float:
    return 0.002 * (t_c - 25.0) + 0.004 * (p_mpa - 1.0) + rng.gauss(0.0, 0.003)


def _tcs_voltage(lambda_true: float, t_c: float, thermal_drift: float, rng) -> float:
    thermal_baseline_now = PROCESSING_PARAMS["thermal_baseline_init"] + thermal_drift
    return (
        thermal_baseline_now
        + PROCESSING_PARAMS["tcs_response_slope"] * (lambda_true - PROCESSING_PARAMS["tcs_lambda_offset"])
        + PROCESSING_PARAMS["tcs_temperature_response"] * (t_c - 20.0)
        + rng.gauss(0.0, 0.006)
    )
