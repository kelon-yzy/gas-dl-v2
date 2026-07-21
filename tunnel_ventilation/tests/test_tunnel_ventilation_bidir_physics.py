"""F1 tests: bidirectional flow physics, waveform composition, packaging contract."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from tv3.sim.core.tunnel_ventilation_bidir_schema import (
    BIDIR_ORACLE_ARRAYS,
    SCHEMA_VERSION,
    SIM_REVISION_TAG,
)
from tv3.sim.generation.tunnel_ventilation.acoustic_physics import (
    hidden_attenuation_v2,
    hidden_sound_speed_v2,
)
from tv3.sim.generation.tunnel_ventilation.conditions import (
    generate_tunnel_ventilation_bidir_condition_rows,
)
from tv3.sim.generation.tunnel_ventilation.flow_physics import (
    bidirectional_transit_times_s,
    bidir_sim_revision,
    reciprocal_sum_path_velocity_m_per_s,
    reciprocal_sum_sound_speed_m_per_s,
    sample_v_path_m_per_s,
    sound_speed_bias_from_velocity_variance_m_per_s,
)
from tv3.sim.generation.tunnel_ventilation.slow import build_sequence_arrays
from tv3.sim.generation.waveforms import (
    WaveformSpec,
    simulate_bidirectional_waveform_measurement,
    simulate_waveform_measurement,
)
from tv3.sim.packaging.arrays import write_arrays
from tv3.sim.packaging.manifest import build_manifest


_TV3_KW = dict(
    x_h2=0.0,
    x_ch4=0.0,
    x_co2=1.0,
    x_n2=79.0,
    t_c=20.0,
    p_mpa=0.101325,
    h_rh=40.0,
)
_EXTRA = {"x_o2": 20.0}


def _quiet_spec(**overrides) -> WaveformSpec:
    base = dict(
        noise_std_v=0.0,
        trigger_jitter_std_s=0.0,
        waveform_dtype="int16",
        per_timestep_scale=True,
    )
    base.update(overrides)
    return WaveformSpec(**base)


def test_reciprocal_sum_exact_on_uniform_flow_grid():
    c = 342.9
    l_m = 0.25
    for v in (-4.0, -1.0, 0.0, 0.25, 1.0, 4.0):
        t_ab, t_ba = bidirectional_transit_times_s(l_m, c, v)
        c_hat = reciprocal_sum_sound_speed_m_per_s(l_m, t_ab, t_ba)
        v_hat = reciprocal_sum_path_velocity_m_per_s(l_m, t_ab, t_ba)
        assert abs(c_hat - c) <= 1e-9
        assert abs(v_hat - v) <= 1e-9


def test_var_v_second_order_residual_sign_and_magnitude():
    c = 342.9
    sigma_v = 0.4
    var_v = sigma_v**2
    bias = sound_speed_bias_from_velocity_variance_m_per_s(c, var_v)
    assert bias < 0.0
    assert abs(bias - (-var_v / c)) <= 1e-15
    # Prior table: residual O2 equivalent < 0.01 vol% at this sigma
    # δc/c per vol% O2 ≈ 7.1e-4 → |bias|/c / 7.1e-4 << 0.01
    o2_equiv = abs(bias) / c / 7.1e-4
    assert o2_equiv < 0.01


def test_zero_noise_zero_jitter_waveform_recovery_gate():
    spec = _quiet_spec()
    l_m = 0.25
    for v in (-3.0, -0.5, 0.0, 0.5, 2.5, 4.0):
        pair = simulate_bidirectional_waveform_measurement(
            **_TV3_KW,
            l_m=l_m,
            seed=11,
            spec=spec,
            v_path_m_per_s=v,
            sound_speed_fn=hidden_sound_speed_v2,
            attenuation_fn=hidden_attenuation_v2,
            extra_gas_kwargs=_EXTRA,
        )
        c_true = float(pair["sound_speed_m_per_s"])
        assert abs(pair["reciprocal_sum_sound_speed_m_per_s"] - c_true) <= 0.01
        assert abs(pair["reciprocal_sum_path_velocity_m_per_s"] - v) <= 0.01

        # Delay-corrected observed TOF must also recover (zero jitter, known τ)
        tau = spec.system_delay_s + spec.cable_delay_s
        t_ab = float(pair["tof_observed_ab_s"]) - tau
        t_ba = float(pair["tof_observed_ba_s"]) - tau
        assert abs(reciprocal_sum_sound_speed_m_per_s(l_m, t_ab, t_ba) - c_true) <= 0.01
        assert abs(reciprocal_sum_path_velocity_m_per_s(l_m, t_ab, t_ba) - v) <= 0.01


def test_zero_flow_ab_matches_unidirectional_pointwise():
    spec = _quiet_spec()
    controlled_seed = 4242
    ab_only = simulate_waveform_measurement(
        **_TV3_KW,
        l_m=0.22,
        seed=controlled_seed,
        spec=spec,
        sound_speed_fn=hidden_sound_speed_v2,
        attenuation_fn=hidden_attenuation_v2,
        extra_gas_kwargs=_EXTRA,
        path_velocity_m_per_s=0.0,
    )
    uni_match = simulate_waveform_measurement(
        **_TV3_KW,
        l_m=0.22,
        seed=controlled_seed,
        spec=spec,
        sound_speed_fn=hidden_sound_speed_v2,
        attenuation_fn=hidden_attenuation_v2,
        extra_gas_kwargs=_EXTRA,
    )
    for key in (
        "tof_true_s",
        "tof_observed_s",
        "peak_index",
        "tof_quality",
        "tof_accepted",
        "sound_speed_m_per_s",
        "alpha_true_npm",
        "trigger_jitter_s",
    ):
        assert ab_only[key] == uni_match[key]
    np.testing.assert_array_equal(ab_only["waveform_int"], uni_match["waveform_int"])

    pair = simulate_bidirectional_waveform_measurement(
        **_TV3_KW,
        l_m=0.22,
        seed=7,
        spec=spec,
        v_path_m_per_s=0.0,
        sound_speed_fn=hidden_sound_speed_v2,
        attenuation_fn=hidden_attenuation_v2,
        extra_gas_kwargs=_EXTRA,
    )
    assert abs(float(pair["tof_true_ab_s"]) - float(pair["tof_true_ba_s"])) <= 1e-15
    assert abs(float(pair["reciprocal_sum_path_velocity_m_per_s"])) <= 1e-12


def test_delay_asymmetry_enters_v_hat_not_c_hat():
    spec = _quiet_spec()
    l_m = 0.25
    v = 1.0
    delta_tau = 0.2e-6
    pair = simulate_bidirectional_waveform_measurement(
        **_TV3_KW,
        l_m=l_m,
        seed=3,
        spec=spec,
        v_path_m_per_s=v,
        delay_asymmetry_s=delta_tau,
        sound_speed_fn=hidden_sound_speed_v2,
        attenuation_fn=hidden_attenuation_v2,
        extra_gas_kwargs=_EXTRA,
    )
    c_true = float(pair["sound_speed_m_per_s"])
    # Per-direction true delay correction
    tau0 = spec.system_delay_s + spec.cable_delay_s
    t_ab = float(pair["tof_observed_ab_s"]) - (tau0 + 0.5 * delta_tau)
    t_ba = float(pair["tof_observed_ba_s"]) - (tau0 - 0.5 * delta_tau)
    assert abs(reciprocal_sum_sound_speed_m_per_s(l_m, t_ab, t_ba) - c_true) <= 0.01
    assert abs(reciprocal_sum_path_velocity_m_per_s(l_m, t_ab, t_ba) - v) <= 0.01

    # Common-mean delay correction leaves ĉ nearly exact but biases v̂
    t_ab_wrong = float(pair["tof_observed_ab_s"]) - tau0
    t_ba_wrong = float(pair["tof_observed_ba_s"]) - tau0
    c_wrong = reciprocal_sum_sound_speed_m_per_s(l_m, t_ab_wrong, t_ba_wrong)
    v_wrong = reciprocal_sum_path_velocity_m_per_s(l_m, t_ab_wrong, t_ba_wrong)
    assert abs(c_wrong - c_true) <= 0.05  # second-order in δτ
    expected_v_bias = -(c_true**2) * delta_tau / (2.0 * l_m)
    assert abs((v_wrong - v) - expected_v_bias) <= 0.02


def test_jitter_correlation_modes_distinguishable():
    spec = _quiet_spec(trigger_jitter_std_s=3.0e-6, noise_std_v=0.0)
    n = 80
    indep_corr = []
    shared_equal = 0
    for i in range(n):
        indep = simulate_bidirectional_waveform_measurement(
            **_TV3_KW,
            l_m=0.25,
            seed=1000 + i,
            spec=spec,
            v_path_m_per_s=1.0,
            jitter_correlation="independent",
            sound_speed_fn=hidden_sound_speed_v2,
            attenuation_fn=hidden_attenuation_v2,
            extra_gas_kwargs=_EXTRA,
        )
        shared = simulate_bidirectional_waveform_measurement(
            **_TV3_KW,
            l_m=0.25,
            seed=1000 + i,
            spec=spec,
            v_path_m_per_s=1.0,
            jitter_correlation="shared_trigger",
            sound_speed_fn=hidden_sound_speed_v2,
            attenuation_fn=hidden_attenuation_v2,
            extra_gas_kwargs=_EXTRA,
        )
        indep_corr.append(
            (indep["trigger_jitter_ab_s"], indep["trigger_jitter_ba_s"])
        )
        if shared["trigger_jitter_ab_s"] == shared["trigger_jitter_ba_s"]:
            shared_equal += 1

    ab = np.array([x[0] for x in indep_corr])
    ba = np.array([x[1] for x in indep_corr])
    corr = float(np.corrcoef(ab, ba)[0, 1])
    assert abs(corr) < 0.35
    assert shared_equal == n


def test_v_path_sampling_zero_anchor_fraction():
    rng = random.Random(0)
    values = sample_v_path_m_per_s(100, rng=rng, zero_anchor_fraction_min=0.10)
    assert values.shape == (100,)
    assert (np.abs(values) < 1e-15).sum() >= 10
    assert values.min() >= -4.0
    assert values.max() <= 4.0


def test_v_path_sampling_zero_fraction_means_no_forced_anchor():
    rng = random.Random(1)
    values = sample_v_path_m_per_s(20, rng=rng, zero_anchor_fraction_min=0.0)
    assert (np.abs(values) < 1e-15).sum() == 0


def test_v_path_sampling_lhs_one_per_stratum():
    """1D LHS places one sample in each equal-width stratum of the non-zero set."""
    rng = random.Random(2)
    n = 32
    values = sample_v_path_m_per_s(n, rng=rng, zero_anchor_fraction_min=0.0)
    lo, hi = -4.0, 4.0
    edges = np.linspace(lo, hi, n + 1)
    occupied = set()
    for v in values:
        bin_idx = int(np.searchsorted(edges, v, side="right") - 1)
        bin_idx = min(max(bin_idx, 0), n - 1)
        occupied.add(bin_idx)
    assert len(occupied) == n


def test_bidir_condition_rows_and_slow_path_smoke():
    rows = generate_tunnel_ventilation_bidir_condition_rows(4, seed=9)
    assert all("v_path_m_per_s" in row for row in rows)
    # Force one zero-flow row for AB/uni consistency at sequence level
    rows[0]["v_path_m_per_s"] = "0.000000"
    rows[0]["flow_scenario"] = "zero_anchor"
    spec = _quiet_spec()
    arrays = build_sequence_arrays(
        rows[:1],
        timesteps=4,
        dt_s=1.0,
        seed=1,
        multi_path_phase="steady",
        ultrasonic_spec=spec,
        fiber_mic_spec=None,
        path_lms=(0.20, 0.22, 0.25),
        bidirectional=True,
    )
    assert arrays["bidirectional"] is True
    assert arrays["ultrasonic_ab"].shape[0] == 1
    assert "ultrasonic" not in arrays
    assert abs(float(arrays["ultrasonic_v_path_true_m_per_s"][0, 0])) <= 1e-12


def test_bidir_packaging_and_manifest_contract(tmp_path: Path):
    rows = generate_tunnel_ventilation_bidir_condition_rows(2, seed=5)
    spec = _quiet_spec()
    arrays = build_sequence_arrays(
        rows,
        timesteps=4,
        dt_s=1.0,
        seed=2,
        multi_path_phase="steady",
        ultrasonic_spec=spec,
        fiber_mic_spec=None,
        path_lms=(0.20, 0.25),
        bidirectional=True,
    )
    labels = np.array([[float(r["x_CO2"]), float(r["x_O2"]), float(r["x_N2"])] for r in rows], dtype=np.float32)
    shapes = write_arrays(
        tmp_path,
        arrays,
        labels,
        [r["sequence_id"] for r in rows],
        ("V_NDIR_CO2", "V_TCS", "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"),
        ("x_CO2", "x_O2", "x_N2"),
        "npy",
        ultrasonic_dtype="int16",
    )
    assert (tmp_path / "sequences" / "ultrasonic_ab_int16.npy").is_file()
    assert (tmp_path / "sequences" / "ultrasonic_ba_int16.npy").is_file()
    for name in BIDIR_ORACLE_ARRAYS:
        assert (tmp_path / "sequences" / f"{name}.npy").is_file()
    assert (tmp_path / "sequences" / "ultrasonic_alpha_true_npm.npy").is_file()
    assert "ultrasonic_ab" in shapes
    assert "ultrasonic_alpha_true_npm" in shapes

    revision = bidir_sim_revision()
    assert revision["tag"] == SIM_REVISION_TAG
    manifest = build_manifest(
        dataset_slug="tv3-bidir-smoke",
        sequence_count=2,
        seed=5,
        timesteps=4,
        dt_s=1.0,
        storage="npy",
        multi_path_phase="steady",
        stage_profile="standard_exposure",
        stage_jitter=0.0,
        phase_schedule={"name": "standard_exposure"},
        sampling_strategy="lhs",
        path_lms=(0.20, 0.25),
        optical_absorption_backend="empirical_v1",
        shapes=shapes,
        slow_channels=("V_NDIR_CO2", "V_TCS", "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"),
        labels=("x_CO2", "x_O2", "x_N2"),
        sim_revision=revision,
        schema_version=SCHEMA_VERSION,
        composition_scheme="tunnel_ventilation_bidir",
    )
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["sim_revision"]["tag"] == SIM_REVISION_TAG


def test_unidirectional_path_unchanged_when_bidirectional_false():
    from tv3.sim.generation.tunnel_ventilation.conditions import (
        generate_tunnel_ventilation_condition_rows,
    )

    rows = generate_tunnel_ventilation_condition_rows(1, seed=3)
    spec = _quiet_spec()
    arrays = build_sequence_arrays(
        rows,
        timesteps=4,
        dt_s=1.0,
        seed=3,
        multi_path_phase="steady",
        ultrasonic_spec=spec,
        fiber_mic_spec=None,
        path_lms=(0.25,),
        bidirectional=False,
    )
    assert arrays["bidirectional"] is False
    assert "ultrasonic" in arrays
    assert "ultrasonic_ab" not in arrays
