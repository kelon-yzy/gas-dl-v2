from __future__ import annotations

import math

from sim.generation.optical_crosstalk import apply_optical_crosstalk
from sim.generation.gas_state import h2o_mole_percent_from_rh


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
    "f_relax_n2_per_atm": 65000.0,
    "alpha_lambda_max_h2o": 0.01,
    "f_relax_h2o_per_atm": 100000.0,
    "acoustic_excitation_frequency_hz": 40000.0,
}

_PRESSURE_MPA_TO_ATM = 1.0 / 0.101325
_T0_K = 293.15
_NP_TO_DB = 8.686
_SPEED_H2_MS = 1306.0
_SPEED_CH4_MS = 446.0
_SPEED_CO2_MS = 268.0
_SPEED_N2_MS = 353.0


def hidden_sound_speed_v2(x_h2: float, x_ch4: float, x_co2: float, x_n2: float, t_c: float) -> float:
    x_h2_frac = max(0.0, x_h2) / 100.0
    x_ch4_frac = max(0.0, x_ch4) / 100.0
    x_co2_frac = max(0.0, x_co2) / 100.0
    x_n2_frac = max(0.0, x_n2) / 100.0
    c_mix = (
        x_h2_frac * _SPEED_H2_MS
        + x_ch4_frac * _SPEED_CH4_MS
        + x_co2_frac * _SPEED_CO2_MS
        + x_n2_frac * _SPEED_N2_MS
    )
    c_mix += 0.6 * (t_c - 25.0)
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
) -> dict[str, float | bool | object]:
    x_h2 = float(condition["x_H2"])
    x_ch4 = float(condition["x_CH4"])
    x_co2 = float(condition["x_CO2"])
    x_n2 = float(condition["x_N2"])
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    h_rh = float(condition["H_RH"])
    l_m = float(condition["L_m"])

    sound_speed = hidden_sound_speed_v2(x_h2, x_ch4, x_co2, x_n2, t_c)
    attenuation = hidden_attenuation_v2(x_h2, x_ch4, x_co2, x_n2, t_c, p_mpa, h_rh, c_mix=sound_speed)
    tof = l_m / sound_speed + PROCESSING_PARAMS["t_delay_s"] + rng.gauss(0.0, 0.000003)
    amp = max(0.005, math.exp(-attenuation["alpha_true_v2"] * l_m) + rng.gauss(0.0, 0.0008))
    f_peak = 40000.0 + 23.0 * x_h2 - 18.0 * x_co2 + 2.0 * (t_c - 20.0) + rng.gauss(0.0, 65.0)
    a_fft_max = amp * (900.0 + rng.gauss(0.0, 20.0))

    if optical_absorption is None:
        absorption_ch4 = _hidden_absorption_ch4(x_ch4, h_rh, p_mpa, t_c)
        absorption_co2 = _hidden_absorption_co2(x_co2, h_rh, p_mpa, t_c)
        optical_absorption = apply_optical_crosstalk(absorption_ch4=absorption_ch4, absorption_co2=absorption_co2)
    lambda_true = _hidden_lambda_mix(x_h2, x_co2, t_c)

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
    x_co2 = float(condition["x_CO2"])
    t_c = float(condition["T_C"])
    p_mpa = float(condition["P_MPa"])
    lambda_true = _hidden_lambda_mix(x_h2, x_co2, t_c)
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


def _hidden_lambda_mix(x_h2: float, x_co2: float, t_c: float) -> float:
    return 0.034 + 0.00155 * x_h2 - 0.00011 * x_co2 + 0.00002 * (t_c - 25.0)


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
