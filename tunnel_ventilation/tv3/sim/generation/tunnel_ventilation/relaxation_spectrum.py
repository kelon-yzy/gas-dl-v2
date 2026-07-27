"""MRS-1 multifrequency relaxation dispersion forward model.

Pure functions only. Does not modify ``hidden_sound_speed_v2`` /
``hidden_attenuation_v2``. Constants match MRS-0
``configs/tv3_mrs/parameter_registry.json``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from tv3.sim.generation.gas_state import h2o_mole_percent_from_rh
from tv3.sim.generation.tunnel_ventilation.acoustic_physics import (
    PROCESSING_PARAMS_V2,
    _GAS_CP,
    _GAS_M,
    _NP_TO_DB,
    _PRESSURE_MPA_TO_ATM,
    _R_GAS,
    _T0_K,
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
)

# MRS-0 frozen vibrational constants
_THETA_VIB_K = {"O2": 2270.0, "N2": 3390.0, "CO2": 960.0}
# CO2 ν2 bending mode is doubly degenerate (literature); needed so derived
# alpha_lambda_max stays within factor 2 of legacy empirical 0.12.
_VIB_DEGENERACY = {"O2": 1.0, "N2": 1.0, "CO2": 2.0}

_BASS_O2_DRY_HZ = 24.0
_BASS_O2_HUM_SCALE = 40400.0
_BASS_O2_NUM_OFF = 0.02
_BASS_O2_DEN_OFF = 0.391
_BASS_N2_DRY_HZ = 9.0
_BASS_N2_HUM_SCALE = 280.0
_BASS_N2_TEMP_EXP = 4.17
_BASS_T0_K = 293.15


def c_vib_over_r(theta_k: float, t_k: float, *, degeneracy: float = 1.0) -> float:
    """Harmonic-oscillator vibrational heat capacity over R (per mole of species)."""
    if t_k <= 0.0 or theta_k <= 0.0:
        raise ValueError("theta_k and t_k must be > 0")
    x = theta_k / t_k
    ex = math.exp(x)
    return float(degeneracy) * (x * x * ex) / ((ex - 1.0) ** 2)


def bass_f_r_o2_hz(*, h_mole_percent: float, p_atm: float) -> float:
    """Bass 1990 / ISO 9613-1 oxygen vibrational relaxation frequency [Hz]."""
    h = max(0.0, float(h_mole_percent))
    p = max(float(p_atm), 1e-12)
    return p * (
        _BASS_O2_DRY_HZ
        + _BASS_O2_HUM_SCALE * h * (_BASS_O2_NUM_OFF + h) / (_BASS_O2_DEN_OFF + h)
    )


def bass_f_r_n2_hz(*, h_mole_percent: float, t_c: float, p_atm: float) -> float:
    """Bass 1990 / ISO 9613-1 nitrogen vibrational relaxation frequency [Hz]."""
    h = max(0.0, float(h_mole_percent))
    p = max(float(p_atm), 1e-12)
    t_k = float(t_c) + 273.15
    tr = t_k / _BASS_T0_K
    return p * (tr**-0.5) * (
        _BASS_N2_DRY_HZ
        + _BASS_N2_HUM_SCALE * h * math.exp(-_BASS_N2_TEMP_EXP * (tr ** (-1.0 / 3.0) - 1.0))
    )


def _normalized_fracs(x_co2: float, x_o2: float, x_n2: float) -> dict[str, float]:
    fracs = {
        "CO2": max(0.0, x_co2) / 100.0,
        "O2": max(0.0, x_o2) / 100.0,
        "N2": max(0.0, x_n2) / 100.0,
    }
    total = sum(fracs.values())
    if total <= 0.0:
        raise ValueError("composition fractions sum to 0")
    return {k: v / total for k, v in fracs.items()}


def _mix_gamma(fracs: dict[str, float]) -> float:
    cp_mix = sum(fracs[k] * _GAS_CP[k] for k in fracs)
    return cp_mix / max(cp_mix - _R_GAS, 1e-9)


def delta_c_over_c_from_cvib(*, gamma: float, x_frac: float, c_vib_r: float) -> float:
    """Small-C_vib relative dispersion step for one relaxation process."""
    return ((gamma - 1.0) ** 2 / (2.0 * gamma)) * x_frac * c_vib_r


def alpha_lambda_max_from_delta_c_over_c(delta_c_over_c: float) -> float:
    """Single-relaxation Kramers–Kronig: α_λmax = π · Δc/c."""
    return math.pi * delta_c_over_c


def _lorentzian_alpha_npm(
    *,
    alpha_lambda_max: float,
    f_hz: np.ndarray | float,
    f_r: float,
    c_eq: float,
) -> np.ndarray | float:
    """α [Np/m] from mixture-level α_λmax (x already folded into α_λmax)."""
    f = np.asarray(f_hz, dtype=np.float64)
    alpha_lambda = alpha_lambda_max * 2.0 * f * f_r / (f**2 + f_r**2)
    out = alpha_lambda * f / max(c_eq, 1e-12)
    if np.ndim(f_hz) == 0:
        return float(out)
    return out


def _process_block(
    *,
    name: str,
    f_r_hz: float,
    delta_c_m_per_s: float,
    alpha_lambda_max: float,
    c_vib_over_r_species: float,
    x_frac: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "f_r_hz": float(f_r_hz),
        "delta_c_m_per_s": float(delta_c_m_per_s),
        "alpha_lambda_max": float(alpha_lambda_max),
        "c_vib_over_r": float(c_vib_over_r_species),
        "x_frac": float(x_frac),
    }


def relaxation_spectrum(
    x_co2: float,
    x_o2: float,
    x_n2: float,
    t_c: float,
    p_mpa: float,
    h_rh: float,
    f_hz_array: np.ndarray | list[float] | tuple[float, ...],
) -> dict[str, Any]:
    """Multifrequency relaxation dispersion / absorption spectrum.

    Returns
    -------
    dict with:
      c_eq, c_f, alpha_f, alpha_classical_f,
      processes (o2/n2/co2/h2o), h_w_pct, p_atm, gamma_mix
    """
    f = np.asarray(f_hz_array, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError("f_hz_array must be 1-D")
    if np.any(f <= 0.0):
        raise ValueError("all frequencies must be > 0")

    fracs = _normalized_fracs(x_co2, x_o2, x_n2)
    gamma = _mix_gamma(fracs)
    t_k = max(float(t_c) + 273.15, 1.0)
    p_atm = max(float(p_mpa), 1e-4) * _PRESSURE_MPA_TO_ATM
    h_w_pct = h2o_mole_percent_from_rh(t_c, p_mpa, h_rh)
    h_w_frac = max(0.0, h_w_pct) / 100.0

    c_eq = hidden_sound_speed_v2(
        x_h2=0.0, x_ch4=0.0, x_co2=x_co2, x_n2=x_n2, t_c=t_c, x_o2=x_o2
    )

    # --- process frequencies ---
    f_r_o2 = bass_f_r_o2_hz(h_mole_percent=h_w_pct, p_atm=p_atm)
    f_r_n2 = bass_f_r_n2_hz(h_mole_percent=h_w_pct, t_c=t_c, p_atm=p_atm)
    params = PROCESSING_PARAMS_V2
    f_r_co2 = params["f_relax_co2_per_atm"] * p_atm * (
        1.0 + params["k_h2o_to_f_relax_co2"] * h_w_pct
    )
    f_r_h2o = params["f_relax_h2o_per_atm"] * p_atm

    # --- C_vib-derived strengths (O2/N2/CO2) ---
    cv_o2 = c_vib_over_r(_THETA_VIB_K["O2"], t_k, degeneracy=_VIB_DEGENERACY["O2"])
    cv_n2 = c_vib_over_r(_THETA_VIB_K["N2"], t_k, degeneracy=_VIB_DEGENERACY["N2"])
    cv_co2 = c_vib_over_r(_THETA_VIB_K["CO2"], t_k, degeneracy=_VIB_DEGENERACY["CO2"])

    dcc_o2 = delta_c_over_c_from_cvib(gamma=gamma, x_frac=fracs["O2"], c_vib_r=cv_o2)
    dcc_n2 = delta_c_over_c_from_cvib(gamma=gamma, x_frac=fracs["N2"], c_vib_r=cv_n2)
    dcc_co2 = delta_c_over_c_from_cvib(gamma=gamma, x_frac=fracs["CO2"], c_vib_r=cv_co2)

    # H2O: keep MRS-0 empirical intensity; convert via KK for dispersion consistency
    alpha_lmax_h2o_pure = float(params["alpha_lambda_max_h2o"])
    dcc_h2o = (alpha_lmax_h2o_pure * h_w_frac) / math.pi

    processes = {
        "o2": _process_block(
            name="o2",
            f_r_hz=f_r_o2,
            delta_c_m_per_s=dcc_o2 * c_eq,
            alpha_lambda_max=alpha_lambda_max_from_delta_c_over_c(dcc_o2),
            c_vib_over_r_species=cv_o2,
            x_frac=fracs["O2"],
        ),
        "n2": _process_block(
            name="n2",
            f_r_hz=f_r_n2,
            delta_c_m_per_s=dcc_n2 * c_eq,
            alpha_lambda_max=alpha_lambda_max_from_delta_c_over_c(dcc_n2),
            c_vib_over_r_species=cv_n2,
            x_frac=fracs["N2"],
        ),
        "co2": _process_block(
            name="co2",
            f_r_hz=f_r_co2,
            delta_c_m_per_s=dcc_co2 * c_eq,
            alpha_lambda_max=alpha_lambda_max_from_delta_c_over_c(dcc_co2),
            c_vib_over_r_species=cv_co2,
            x_frac=fracs["CO2"],
        ),
        "h2o": _process_block(
            name="h2o",
            f_r_hz=f_r_h2o,
            delta_c_m_per_s=dcc_h2o * c_eq,
            alpha_lambda_max=alpha_lmax_h2o_pure * h_w_frac,
            c_vib_over_r_species=float("nan"),
            x_frac=h_w_frac,
        ),
    }

    # Dispersion: c(f) = c_eq * (1 + Σ (Δc_i/c_eq) * f²/(f²+f_r²))
    disp = np.zeros_like(f)
    alpha_rel = np.zeros_like(f)
    for proc in processes.values():
        fr = float(proc["f_r_hz"])
        dcc = float(proc["delta_c_m_per_s"]) / c_eq
        disp += dcc * (f**2) / (f**2 + fr**2)
        alpha_rel = alpha_rel + _lorentzian_alpha_npm(
            alpha_lambda_max=float(proc["alpha_lambda_max"]),
            f_hz=f,
            f_r=fr,
            c_eq=c_eq,
        )

    alpha_classical = (
        PROCESSING_PARAMS_V2["alpha_classical_K_ref"]
        * (f**2)
        * (1.0 / max(p_atm, 1e-4))
        * math.sqrt(t_k / _T0_K)
        / _NP_TO_DB
    )
    c_f = c_eq * (1.0 + disp)
    alpha_f = np.maximum(0.0, alpha_classical + alpha_rel)

    return {
        "c_eq": float(c_eq),
        "c_frozen": float(c_eq + sum(p["delta_c_m_per_s"] for p in processes.values())),
        "c_f": c_f,
        "alpha_f": alpha_f,
        "alpha_classical_f": alpha_classical,
        "f_hz": f,
        "gamma_mix": float(gamma),
        "p_atm": float(p_atm),
        "h_w_pct": float(h_w_pct),
        "processes": processes,
        "m_mix": float(sum(fracs[k] * _GAS_M[k] for k in fracs)),
    }


def pure_gas_dispersion_step_m_per_s(gas: str, t_c: float = 26.85) -> float:
    """Pure-gas full dispersion step Δc [m/s] at temperature t_c (default ≈300 K)."""
    key = gas.upper()
    if key not in ("O2", "N2", "CO2"):
        raise ValueError(f"unsupported gas {gas!r}")
    x = {"CO2": 0.0, "O2": 0.0, "N2": 0.0}
    x[key] = 100.0
    # Dry, 1 atm — frequencies unused for Δc magnitude
    out = relaxation_spectrum(
        x_co2=x["CO2"],
        x_o2=x["O2"],
        x_n2=x["N2"],
        t_c=t_c,
        p_mpa=0.101325,
        h_rh=0.0,
        f_hz_array=np.array([1.0, 1.0e9]),
    )
    return float(out["processes"][key.lower()]["delta_c_m_per_s"])


def compare_alpha_at_200khz_vs_v2(
    x_co2: float,
    x_o2: float,
    x_n2: float,
    t_c: float,
    p_mpa: float,
    h_rh: float,
) -> dict[str, Any]:
    """Register MRS vs v2 α differences at 200 kHz (explicit, not silently absorbed)."""
    f0 = float(PROCESSING_PARAMS_V2["acoustic_excitation_frequency_hz"])
    mrs = relaxation_spectrum(x_co2, x_o2, x_n2, t_c, p_mpa, h_rh, np.array([f0]))
    v2 = hidden_attenuation_v2(
        x_h2=0.0,
        x_ch4=0.0,
        x_co2=x_co2,
        x_n2=x_n2,
        t_c=t_c,
        p_mpa=p_mpa,
        h_rh=h_rh,
        f_hz=f0,
        x_o2=x_o2,
    )
    alpha_mrs = float(mrs["alpha_f"][0])
    alpha_v2 = float(v2["alpha_true_v2"])
    # Component-level reconstruction for explanation
    procs = mrs["processes"]
    return {
        "f_hz": f0,
        "alpha_mrs_npm": alpha_mrs,
        "alpha_v2_npm": alpha_v2,
        "delta_npm": alpha_mrs - alpha_v2,
        "relative_delta": (alpha_mrs - alpha_v2) / max(abs(alpha_v2), 1e-18),
        "sources_of_difference": {
            "o2_enabled_in_mrs": {
                "mrs_alpha_lambda_max": procs["o2"]["alpha_lambda_max"],
                "v2_alpha_lambda_max": PROCESSING_PARAMS_V2["alpha_lambda_max_o2"],
                "mrs_f_r_hz": procs["o2"]["f_r_hz"],
                "v2_f_r_hz": v2["f_relax_o2_eff"],
                "note": "v2 sets alpha_lambda_max_o2=0; MRS derives O2 from C_vib + Bass f_r(h).",
            },
            "n2_intensity_rederived": {
                "mrs_alpha_lambda_max": procs["n2"]["alpha_lambda_max"],
                "v2_alpha_lambda_max_times_x": PROCESSING_PARAMS_V2["alpha_lambda_max_n2"]
                * max(0.0, x_n2)
                / 100.0,
                "mrs_f_r_hz": procs["n2"]["f_r_hz"],
                "v2_f_r_hz": v2["f_relax_n2_eff"],
                "note": (
                    "v2 uses empirical alpha_lambda_max_n2=0.004 (dry f_r only); "
                    "MRS uses C_vib-derived strength (~15× smaller) + Bass humidity catalysis."
                ),
            },
            "co2_intensity_rederived": {
                "mrs_alpha_lambda_max": procs["co2"]["alpha_lambda_max"],
                "v2_alpha_lambda_max_times_x": PROCESSING_PARAMS_V2["alpha_lambda_max_co2"]
                * max(0.0, x_co2)
                / 100.0,
                "note": (
                    "MRS CO2 strength from C_vib(θ=960 K, g=2); "
                    "legacy empirical alpha_lambda_max_co2=0.12 kept only in v2."
                ),
            },
            "classical_shared": {
                "note": "Classical α∝f² term uses the same K_ref as v2.",
            },
        },
        "claim": (
            "MRS-1 α(200 kHz) is allowed to differ from v2; differences are explained "
            "above and must not be silently absorbed. v2 behavior remains unchanged."
        ),
    }
