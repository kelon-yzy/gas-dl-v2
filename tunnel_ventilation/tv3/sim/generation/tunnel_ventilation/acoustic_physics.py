"""掘进通风场景的声学物理。

与 hydrogen_ng `sim.generation.acoustic_physics` 和 syngas `sim.generation.syngas.acoustic_physics`
并存。主要差异：
- 声速/热导混合公式仅含 CO2/O2/N2 三组分（移除 H2/CH4/CO）
- 衰减公式移除 alpha_ch4 + alpha_h2_diffusion，新增 alpha_o2（200kHz 下可忽略，取 0）
- 无光学串扰矩阵（只有 CO2 一个红外活性组分）
- 无 V_NDIR_CH4 通道（场景无 CH₄，无需 NDIR CH4 传感器）

物理常数来源详见 tunnel_ventilation/docs/foundation/physics_references.md。

签名约定：waveforms.py 的 `_compute_physics` 通过
``sound_speed_fn(x_h2, x_ch4, x_co2, x_n2, t_c=t_c, **extra)`` 调用，
extra 含 ``x_o2``。x_h2/x_ch4 在本场景始终为 0，函数内部忽略。
"""
from __future__ import annotations

import math
import random

from tv3.sim.generation.gas_state import h2o_mole_percent_from_rh


PROCESSING_PARAMS = {
    "t_delay_s": 0.00008,
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
    "alpha_lambda_max_n2": 0.004,
    "f_relax_n2_per_atm": 65000.0,
    "alpha_lambda_max_h2o": 0.01,
    "f_relax_h2o_per_atm": 100000.0,
    # O₂ V-T 弛豫在 200 kHz 下可忽略（dry air fr,O ≈ 24 Hz/atm，Bass 1990 JASA）。
    # 工程实现取 alpha_o2 ≈ 0：不引入弛豫项，仅经典吸收已覆盖。
    "alpha_lambda_max_o2": 0.0,
    "f_relax_o2_per_atm": 24.0,
    "acoustic_excitation_frequency_hz": 200000.0,
}

_PRESSURE_MPA_TO_ATM = 1.0 / 0.101325
_T0_K = 293.15
_NP_TO_DB = 8.686
_R_GAS = 8.314  # J/(mol·K)
# 纯组分摩尔质量 (kg/mol) 与理想气体定压摩尔热容 (J/mol/K) @298.15K
# 来源：NIST WebBook（CO2/N2 沿用主线代码；O2 由 NIST Shomate 手算确认 Chase 1998）
# O₂: cp=29.38, M=0.031998 → γ=29.38/(29.38-8.314)=1.413（双原子理论 1.40 一致）
_GAS_M = {"CO2": 0.04401, "O2": 0.031998, "N2": 0.02801}
_GAS_CP = {"CO2": 37.13, "O2": 29.38, "N2": 29.12}


def hidden_sound_speed_v2(
    x_h2: float,
    x_ch4: float,
    x_co2: float,
    x_n2: float,
    t_c: float,
    *,
    x_o2: float,
) -> float:
    """掘进通风声速混合公式，仅含 CO2/O2/N2 三组分。

    x_h2/x_ch4 在本场景始终为 0（保留参数为兼容 waveforms.py 调用约定），忽略。

    理想气体混合 c = sqrt(γ_mix · R · T / M_mix)，M_mix/cp_mix 摩尔分数加权。
    """
    del x_h2, x_ch4  # 场景无 H2/CH4，显式忽略
    fracs = {
        "CO2": max(0.0, x_co2) / 100.0,
        "O2": max(0.0, x_o2) / 100.0,
        "N2": max(0.0, x_n2) / 100.0,
    }
    total = sum(fracs.values())
    if total <= 0.0:
        return 200.0
    fracs = {k: v / total for k, v in fracs.items()}
    m_mix = sum(fracs[k] * _GAS_M[k] for k in fracs)
    cp_mix = sum(fracs[k] * _GAS_CP[k] for k in fracs)
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
    x_o2: float = 0.0,
) -> dict[str, float]:
    """掘进通风衰减公式，仅含 CO2/O2/N2 三组分类比 + H2O。

    移除 alpha_ch4 + alpha_h2_diffusion；alpha_o2 在 200kHz 下可忽略（取 0）。
    输出 dict 字段命名与 hg/syngas 对齐，便于下游统一处理。
    """
    del x_h2, x_ch4  # 场景无 H2/CH4，显式忽略
    params = PROCESSING_PARAMS_V2
    if f_hz is None:
        f_hz = params["acoustic_excitation_frequency_hz"]
    if c_mix is None:
        c_mix = hidden_sound_speed_v2(x_h2=0.0, x_ch4=0.0, x_co2=x_co2, x_n2=x_n2, t_c=t_c, x_o2=x_o2)
    c_mix = max(c_mix, 200.0)

    p_atm = max(p_mpa, 1e-4) * _PRESSURE_MPA_TO_ATM
    t_k = t_c + 273.15
    h_w_pct = h2o_mole_percent_from_rh(t_c, p_mpa, h_rh)

    # 经典吸收
    alpha_classical_db = (
        params["alpha_classical_K_ref"]
        * (f_hz**2)
        * (1.0 / max(p_atm, 1e-4))
        * math.sqrt(t_k / _T0_K)
    )
    alpha_classical_npm = alpha_classical_db / _NP_TO_DB

    # CO2 V-T 弛豫
    f_r_co2 = params["f_relax_co2_per_atm"] * p_atm * (1.0 + params["k_h2o_to_f_relax_co2"] * h_w_pct)
    alpha_co2_npm = _relaxation_alpha_npm(
        alpha_lambda_max=params["alpha_lambda_max_co2"],
        x_frac=max(0.0, x_co2) / 100.0,
        f_hz=f_hz,
        f_r=f_r_co2,
        c_mix=c_mix,
    )

    # N2 V-T 弛豫（很慢）
    x_n2_frac = max(0.0, x_n2) / 100.0
    f_r_n2 = params["f_relax_n2_per_atm"] * p_atm
    alpha_n2_npm = _relaxation_alpha_npm(
        alpha_lambda_max=params["alpha_lambda_max_n2"],
        x_frac=x_n2_frac,
        f_hz=f_hz,
        f_r=f_r_n2,
        c_mix=c_mix,
    )

    # O2 V-T 弛豫：200 kHz 下 fr,O ≈ 24 Hz/atm，远低于载波，工程取 0。
    # 保留字段以维持输出 schema 对齐，值为 0。
    f_r_o2 = params["f_relax_o2_per_atm"] * p_atm
    alpha_o2_npm = _relaxation_alpha_npm(
        alpha_lambda_max=params["alpha_lambda_max_o2"],
        x_frac=max(0.0, x_o2) / 100.0,
        f_hz=f_hz,
        f_r=f_r_o2,
        c_mix=c_mix,
    )

    # H2O V-T
    h_w_frac = max(0.0, h_w_pct) / 100.0
    f_r_h2o = params["f_relax_h2o_per_atm"] * p_atm
    alpha_h2o_npm = _relaxation_alpha_npm(
        alpha_lambda_max=params["alpha_lambda_max_h2o"],
        x_frac=h_w_frac,
        f_hz=f_hz,
        f_r=f_r_h2o,
        c_mix=c_mix,
    )
    alpha_true = max(
        0.0,
        alpha_classical_npm + alpha_co2_npm + alpha_n2_npm + alpha_o2_npm + alpha_h2o_npm,
    )

    return {
        "alpha_true_v2": alpha_true,
        "alpha_classical_v2": alpha_classical_npm,
        "alpha_co2_v2": alpha_co2_npm,
        "alpha_o2_v2": alpha_o2_npm,
        "alpha_n2_background_v2": alpha_n2_npm,
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
    rng: random.Random,
    *,
    f_hz: float | None = None,
) -> dict[str, float | bool | object]:
    """掘进通风主特征生成。

    与 hydrogen_ng 同名函数相比：
    - 多读取 x_O2
    - 声速/衰减签名多 x_o2（关键字参数）
    - 无光学串扰矩阵（只有 CO2 一个红外活性组分）
    - 不输出 V_NDIR_CH4（场景无 CH₄）
    - 输出不含 V_NDIR_CO（syngas 才有）
    """
    x_co2 = float(condition["x_CO2"])
    x_o2 = float(condition["x_O2"])
    x_n2 = float(condition["x_N2"])
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    h_rh = float(condition["H_RH"])
    l_m = float(condition["L_m"])

    if f_hz is None:
        f_hz = PROCESSING_PARAMS_V2["acoustic_excitation_frequency_hz"]
    sound_speed = hidden_sound_speed_v2(
        x_h2=0.0, x_ch4=0.0, x_co2=x_co2, x_n2=x_n2, t_c=t_c, x_o2=x_o2
    )
    attenuation = hidden_attenuation_v2(
        x_h2=0.0, x_ch4=0.0, x_co2=x_co2, x_n2=x_n2, t_c=t_c, p_mpa=p_mpa, h_rh=h_rh,
        c_mix=sound_speed, f_hz=f_hz, x_o2=x_o2,
    )
    tof = l_m / sound_speed + PROCESSING_PARAMS["t_delay_s"] + rng.gauss(0.0, 0.000003)
    amp = max(0.005, math.exp(-attenuation["alpha_true_v2"] * l_m) + rng.gauss(0.0, 0.0008))
    # f_peak: CO2 降低声速→升高 f_peak（与 hg 一致符号）；O2/N2 通过声速间接影响
    f_peak = f_hz - 18.0 * x_co2 + 2.0 * (t_c - 20.0) + rng.gauss(0.0, 65.0)
    a_fft_max = amp * (900.0 + rng.gauss(0.0, 20.0))

    # 光学吸收：仅 CO2 有红外活性
    absorption_co2 = _hidden_absorption_co2(x_co2, h_rh, p_mpa, t_c)
    lambda_true = _hidden_lambda_mix(x_co2, x_o2, x_n2, t_c)

    optical_drift_co2 = 0.0007 * (h_rh - 50.0) + 0.004 * (p_mpa - 1.0) + rng.gauss(0.0, 0.004)
    thermal_drift = _thermal_baseline_drift(t_c, p_mpa, rng)

    optical_baseline_co2_now = PROCESSING_PARAMS["optical_baseline_co2_init"] + optical_drift_co2 + rng.gauss(0.0, 0.006)

    v_ndir_co2 = max(
        0.1,
        optical_baseline_co2_now * math.exp(-absorption_co2) + rng.gauss(0.0, 0.008),
    )
    v_tcs = _tcs_voltage(lambda_true, t_c, thermal_drift, rng)

    return {
        "TOF": tof,
        "Amp": amp,
        "f_peak": f_peak,
        "A_fft_max": a_fft_max,
        "V_NDIR_CO2": v_ndir_co2,
        "V_TCS": v_tcs,
        "ndir_co2_saturated": absorption_co2 > PROCESSING_PARAMS["optical_saturation_absorbance"],
        "absorption_co2_true": absorption_co2,
        "absorption_co2_observed": absorption_co2,
        "optical_baseline_drift_co2_observed": optical_drift_co2,
        "thermal_baseline_drift_observed": thermal_drift,
    }


def thermal_conductivity_sensor_feature(condition: dict[str, str], rng: random.Random) -> dict[str, float]:
    """TCS 单通道。掘进通风场景需要 x_o2 入热导混合。"""
    x_co2 = float(condition["x_CO2"])
    x_o2 = float(condition["x_O2"])
    x_n2 = float(condition["x_N2"])
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    lambda_true = _hidden_lambda_mix(x_co2, x_o2, x_n2, t_c)
    thermal_drift = _thermal_baseline_drift(t_c, p_mpa, rng)
    return {
        "V_TCS": _tcs_voltage(lambda_true, t_c, thermal_drift, rng),
        "thermal_baseline_drift_observed": thermal_drift,
    }


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _relaxation_alpha_npm(
    *,
    alpha_lambda_max: float,
    x_frac: float,
    f_hz: float,
    f_r: float,
    c_mix: float,
) -> float:
    """V-T 弛豫单组分衰减（Np/m）。"""
    alpha_lambda = alpha_lambda_max * x_frac * 2.0 * f_hz * f_r / (f_hz**2 + f_r**2)
    return alpha_lambda * f_hz / c_mix


def _hidden_absorption_co2(x_co2: float, h_rh: float, p_mpa: float, t_c: float) -> float:
    """CO2 ν₃ 4.26 μm 经验吸收，复用 hg 系数。"""
    return 0.045 * x_co2 + 0.0006 * h_rh + 0.012 * p_mpa + 0.00015 * (t_c - 25.0)


# 纯组分热导率 λ (W/(m·K)) 与粘度 η (Pa·s) @298.15K，用于 WMS 混合规则
# 来源：NIST WebBook 298.15K（CO2/N2 沿用主线代码；O2 由 NIST/CRC/Engineering ToolBox 共识）
# O₂: λ₀=0.0264 W/m·K, n=0.80, η=2.058e-5 Pa·s（NBS TN.350 / NIST REFPROP）
_GAS_LAMBDA = {"CO2": 0.0166, "O2": 0.0264, "N2": 0.0258}
_GAS_ETA = {"CO2": 1.491e-5, "O2": 2.058e-5, "N2": 1.781e-5}
# 各组分热导率温度幂律指数（NIST 拟合，15~35°C 范围）
_LAMBDA_T_EXPONENT = {"CO2": 0.87, "O2": 0.80, "N2": 0.77}


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


def _hidden_lambda_mix(x_co2: float, x_o2: float, x_n2: float, t_c: float) -> float:
    """Wassiljewa-Mason-Saxena 混合热导率 (W/(m·K))，含 O2 组分。

    λ_m = Σ_i (y_i · λ_i(T) / Σ_j y_j · φ_ij)。逐组分幂律温度修正后再混合。
    φ_ij 由 Wilke 公式从 M 和 η 直接计算，无需查表。
    """
    fracs = {
        "CO2": max(0.0, x_co2) / 100.0,
        "O2": max(0.0, x_o2) / 100.0,
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


def _thermal_baseline_drift(t_c: float, p_mpa: float, rng: random.Random) -> float:
    return 0.002 * (t_c - 25.0) + 0.004 * (p_mpa - 1.0) + rng.gauss(0.0, 0.003)


def _tcs_voltage(lambda_true: float, t_c: float, thermal_drift: float, rng: random.Random) -> float:
    thermal_baseline_now = PROCESSING_PARAMS["thermal_baseline_init"] + thermal_drift
    return (
        thermal_baseline_now
        + PROCESSING_PARAMS["tcs_response_slope"] * (lambda_true - PROCESSING_PARAMS["tcs_lambda_offset"])
        + PROCESSING_PARAMS["tcs_temperature_response"] * (t_c - 20.0)
        + rng.gauss(0.0, 0.006)
    )
