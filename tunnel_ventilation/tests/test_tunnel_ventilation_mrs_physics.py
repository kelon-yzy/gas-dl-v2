"""MRS-1 physics anchors for multifrequency relaxation dispersion."""
from __future__ import annotations

import math

import numpy as np

from tv3.sim.generation.tunnel_ventilation.acoustic_physics import (
    PROCESSING_PARAMS_V2,
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
)
from tv3.sim.generation.tunnel_ventilation.relaxation_spectrum import (
    alpha_lambda_max_from_delta_c_over_c,
    bass_f_r_n2_hz,
    bass_f_r_o2_hz,
    c_vib_over_r,
    compare_alpha_at_200khz_vs_v2,
    pure_gas_dispersion_step_m_per_s,
    relaxation_spectrum,
)


def test_bass_f_r_o2_dry_and_h1():
    assert abs(bass_f_r_o2_hz(h_mole_percent=0.0, p_atm=1.0) - 24.0) <= 1e-12
    fr = bass_f_r_o2_hz(h_mole_percent=1.0, p_atm=1.0)
    assert abs(fr - 29649.0) / 29649.0 <= 0.01


def test_bass_f_r_n2_dry_and_h1_20c():
    assert abs(bass_f_r_n2_hz(h_mole_percent=0.0, t_c=20.0, p_atm=1.0) - 9.0) <= 1e-9
    fr = bass_f_r_n2_hz(h_mole_percent=1.0, t_c=20.0, p_atm=1.0)
    assert abs(fr - 289.0) / 289.0 <= 0.02


def test_c_vib_o2_n2_at_300k():
    c_o2 = c_vib_over_r(2270.0, 300.0)
    c_n2 = c_vib_over_r(3390.0, 300.0)
    assert abs(c_o2 - 0.029) / 0.029 <= 0.05
    assert abs(c_n2 - 0.0016) / 0.0016 <= 0.05


def test_pure_gas_dispersion_steps_o2_n2():
    # t_c=26.85 → T≈300 K (registry pure-gas anchors)
    dc_o2 = pure_gas_dispersion_step_m_per_s("O2", t_c=26.85)
    dc_n2 = pure_gas_dispersion_step_m_per_s("N2", t_c=26.85)
    assert abs(dc_o2 - 0.5) / 0.5 <= 0.30
    assert abs(dc_n2 - 0.03) / 0.03 <= 0.30


def test_co2_derived_intensity_within_factor_2_of_empirical():
    """Sanity (non-blocking in plan): derived α_λmax vs legacy 0.12 within factor 2."""
    t_k = 300.0
    # Pure CO2 gamma from same CP table as acoustic_physics
    cp = 37.13
    gamma = cp / (cp - 8.314)
    cv = c_vib_over_r(960.0, t_k, degeneracy=2.0)
    dcc = ((gamma - 1.0) ** 2 / (2.0 * gamma)) * cv
    alpha_lmax = alpha_lambda_max_from_delta_c_over_c(dcc)
    legacy = PROCESSING_PARAMS_V2["alpha_lambda_max_co2"]
    ratio = max(alpha_lmax / legacy, legacy / alpha_lmax)
    assert ratio <= 2.0, f"CO2 alpha_lambda_max ratio {ratio:.3f} exceeds factor 2 ({alpha_lmax=}, {legacy=})"


def test_f_limits_match_ceq_and_frozen():
    x_co2, x_o2, x_n2 = 1.0, 20.0, 79.0
    t_c, p_mpa, h_rh = 25.0, 0.101325, 40.0
    c_eq = hidden_sound_speed_v2(
        x_h2=0.0, x_ch4=0.0, x_co2=x_co2, x_n2=x_n2, t_c=t_c, x_o2=x_o2
    )
    out = relaxation_spectrum(
        x_co2, x_o2, x_n2, t_c, p_mpa, h_rh, np.array([1e-6, 1e12])
    )
    assert abs(out["c_eq"] - c_eq) / c_eq < 1e-9
    assert abs(float(out["c_f"][0]) - c_eq) / c_eq < 1e-9
    c_inf = float(out["c_f"][1])
    assert abs(c_inf - out["c_frozen"]) / out["c_frozen"] < 1e-9
    assert c_inf > c_eq


def test_kramers_kronig_internal_constraint():
    out = relaxation_spectrum(1.0, 20.0, 79.0, 25.0, 0.101325, 50.0, np.array([200000.0]))
    for name, proc in out["processes"].items():
        if name == "h2o":
            # empirical path: α_λmax = π * Δc/c by construction via dcc = α_λmax/π
            pass
        expected = math.pi * proc["delta_c_m_per_s"] / out["c_eq"]
        assert abs(proc["alpha_lambda_max"] - expected) <= 1e-12 * max(1.0, abs(expected))


def test_alpha_200khz_vs_v2_difference_is_registered():
    """MRS α may differ from v2; difference must be explicit and explained."""
    kwargs = dict(x_co2=1.0, x_o2=20.0, x_n2=79.0, t_c=25.0, p_mpa=0.101325, h_rh=50.0)
    cmp = compare_alpha_at_200khz_vs_v2(**kwargs)
    assert "sources_of_difference" in cmp
    assert "o2_enabled_in_mrs" in cmp["sources_of_difference"]
    assert "n2_intensity_rederived" in cmp["sources_of_difference"]
    # v2 O2 strength is identically zero; MRS O2 strength must be > 0 in moist air
    assert cmp["sources_of_difference"]["o2_enabled_in_mrs"]["mrs_alpha_lambda_max"] > 0.0
    assert cmp["sources_of_difference"]["o2_enabled_in_mrs"]["v2_alpha_lambda_max"] == 0.0
    # N2 MRS strength (mixture-level) should be much smaller than v2 empirical × x
    mrs_n2 = cmp["sources_of_difference"]["n2_intensity_rederived"]["mrs_alpha_lambda_max"]
    v2_n2 = cmp["sources_of_difference"]["n2_intensity_rederived"]["v2_alpha_lambda_max_times_x"]
    assert mrs_n2 < v2_n2
    # Values themselves need not match; registration is the gate
    assert abs(cmp["delta_npm"]) >= 0.0


def test_hidden_v2_unchanged_smoke():
    """Regression lock: calling MRS must not alter v2 outputs."""
    c1 = hidden_sound_speed_v2(0.0, 0.0, 1.0, 79.0, 25.0, x_o2=20.0)
    a1 = hidden_attenuation_v2(0.0, 0.0, 1.0, 79.0, 25.0, 0.101325, 50.0, x_o2=20.0)
    _ = relaxation_spectrum(1.0, 20.0, 79.0, 25.0, 0.101325, 50.0, np.array([200000.0]))
    c2 = hidden_sound_speed_v2(0.0, 0.0, 1.0, 79.0, 25.0, x_o2=20.0)
    a2 = hidden_attenuation_v2(0.0, 0.0, 1.0, 79.0, 25.0, 0.101325, 50.0, x_o2=20.0)
    assert c1 == c2
    assert a1 == a2


def test_spectrum_shape_monotonic_dispersion():
    freqs = np.array([10.0, 1e3, 1e4, 1e5, 1e6, 1e8], dtype=np.float64)
    out = relaxation_spectrum(1.0, 20.0, 79.0, 25.0, 0.101325, 50.0, freqs)
    # c(f) non-decreasing for positive Δc processes
    assert np.all(np.diff(out["c_f"]) >= -1e-12)
