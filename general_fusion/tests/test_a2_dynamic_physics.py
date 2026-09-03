from __future__ import annotations

import math

import numpy as np
import pytest

from gf.sim.ar_he_co2 import (
    A1_SOUND_SPEED_MODEL_ID,
    A2DYN_SOUND_SPEED_MODEL_ID,
    a2dyn_cp_t_virial_sound_speed,
    a2dyn_ideal_heat_capacity,
    a2dyn_mixture_virial,
    a2dyn_species_heat_capacity,
    a2dyn_thermodynamic_state,
    ideal_gas_sound_speed,
    sound_speed_for_model,
)
from gf.sim.a2_dynamic_physics import (
    DynamicPhysicsError,
    PhysicsAuditError,
    apply_observation_chain,
    audit_coolprop_sound_speed_grid,
    build_inlet_composition,
    evaluate_shared_physics,
    generate_ar1_noise,
    generate_shared_noise,
    linear_ramp_inlet_coefficient,
    linear_sequence_drift,
    protocol_inlet_coefficient,
    quantize_signal,
    simulate_local_transport,
    simulate_well_mixed_chamber,
    smooth_ramp_inlet_coefficient,
    step_inlet_coefficient,
    validate_composition_pct,
)


def test_step_ramp_and_smoothstep_have_registered_boundaries() -> None:
    time_s = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])

    assert np.array_equal(step_inlet_coefficient(time_s, onset_s=2.0), [0.0, 0.0, 1.0, 1.0, 1.0])
    assert np.allclose(
        linear_ramp_inlet_coefficient(time_s, onset_s=1.0, duration_s=2.0),
        [0.0, 0.0, 0.5, 1.0, 1.0],
    )
    assert np.allclose(
        smooth_ramp_inlet_coefficient(time_s, onset_s=1.0, duration_s=2.0),
        [0.0, 0.0, 0.5, 1.0, 1.0],
    )
    assert step_inlet_coefficient(2.0, onset_s=2.0) == 1.0


def test_protocol_recovery_and_inlet_composition_preserve_closure() -> None:
    time_s = np.arange(0.0, 7.0, 1.0)
    coefficient = protocol_inlet_coefficient(
        time_s,
        kind="incomplete_recovery",
        onset_s=2.0,
        exposure_duration_s=2.0,
        recovery_residual=0.25,
    )
    assert np.array_equal(coefficient, [0.0, 0.0, 1.0, 1.0, 0.25, 0.25, 0.25])

    inlet = build_inlet_composition(
        time_s,
        purge_composition_pct=[100.0, 0.0, 0.0],
        target_composition_pct=[20.0, 30.0, 50.0],
        coefficient=coefficient,
    )
    assert np.all(inlet >= 0.0)
    assert np.allclose(inlet.sum(axis=1), 100.0, atol=1.0e-12)
    assert np.allclose(build_inlet_composition(0.0, purge_composition_pct=[100, 0, 0], target_composition_pct=[0, 50, 50], coefficient=0.0), [100, 0, 0])

    standard_step = protocol_inlet_coefficient(
        time_s,
        kind="step",
        onset_s=2.0,
        exposure_end_s=5.0,
    )
    assert np.array_equal(standard_step, [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])


def test_cstr_and_local_transport_match_analytic_update_and_recovery() -> None:
    inlet = np.asarray(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ]
    )
    dt_s = 0.5
    tau_mix_s = 2.0
    chamber = simulate_well_mixed_chamber(
        inlet,
        dt_s=dt_s,
        tau_mix_s=tau_mix_s,
        initial_composition_pct=[100.0, 0.0, 0.0],
    )
    decay = math.exp(-dt_s / tau_mix_s)
    expected_first_transition = np.asarray([100.0 * decay, 100.0 * (1.0 - decay), 0.0])
    assert np.allclose(chamber[1], expected_first_transition)
    assert np.all(chamber >= 0.0)
    assert np.allclose(chamber.sum(axis=1), 100.0, atol=1.0e-12)

    local = simulate_local_transport(chamber, dt_s=dt_s, tau_transport_s=1.0)
    assert np.allclose(local.sum(axis=1), 100.0, atol=1.0e-12)
    assert np.allclose(
        simulate_local_transport(chamber, dt_s=dt_s, tau_transport_s=0.0),
        chamber,
    )
    assert local[1, 0] > chamber[1, 0]
    assert local[-1, 0] < local[0, 0]


def test_shared_physics_calls_registered_balance_operators() -> None:
    compositions = np.asarray([[100.0, 0.0, 0.0], [90.0, 5.0, 5.0]])
    result = evaluate_shared_physics(
        compositions,
        temperature_k=298.15,
        pressure_pa=101325.0,
        path_length_m=0.2,
    )
    assert set(result) == {
        "sound_speed_m_s",
        "conductivity_w_m_k",
        "tof_s",
        "thermal_voltage_v",
        "ndir_voltage_v",
    }
    assert all(np.isfinite(values).all() for values in result.values())
    assert result["tof_s"][0] > 0.0
    assert result["ndir_voltage_v"][1] < result["ndir_voltage_v"][0]


def test_a2dyn_cp_virial_model_uses_temperature_and_complete_pair_terms() -> None:
    fractions = {"Ar": 0.2, "He": 0.3, "CO2": 0.5}
    co2_low = a2dyn_species_heat_capacity("CO2", 278.15)
    co2_high = a2dyn_species_heat_capacity("CO2", 313.15)
    assert co2_high[0] > co2_low[0]
    assert np.isfinite(co2_low).all()
    assert np.isfinite(co2_high).all()

    heat_capacity = a2dyn_ideal_heat_capacity(fractions, 298.15)
    virial = a2dyn_mixture_virial(fractions, 298.15)
    state = a2dyn_thermodynamic_state(fractions, 298.15, 101325.0)
    assert heat_capacity["cp_molar_j_mol_k"] > heat_capacity["cv_molar_j_mol_k"] > 0.0
    assert np.isfinite(list(virial.values())).all()
    assert state["molar_density_mol_m3"] > 0.0
    assert state["cp_molar_j_mol_k"] > state["cv_molar_j_mol_k"] > 0.0
    assert state["sound_speed_m_s"] > 0.0

    for pure in (
        {"Ar": 1.0, "He": 0.0, "CO2": 0.0},
        {"Ar": 0.0, "He": 1.0, "CO2": 0.0},
        {"Ar": 0.0, "He": 0.0, "CO2": 1.0},
    ):
        assert np.isfinite(list(a2dyn_mixture_virial(pure, 298.15).values())).all()

    ideal = ideal_gas_sound_speed(fractions, 298.15)
    dynamic = a2dyn_cp_t_virial_sound_speed(fractions, 298.15, 101325.0)
    assert not math.isclose(dynamic, ideal, rel_tol=0.0, abs_tol=1.0e-3)
    assert math.isclose(
        a2dyn_cp_t_virial_sound_speed(fractions, 298.15, 0.0),
        a2dyn_thermodynamic_state(fractions, 298.15, 0.0)["sound_speed_m_s"],
        rel_tol=0.0,
        abs_tol=1.0e-12,
    )


def test_a2dyn_sound_speed_model_selection_has_no_implicit_fallback() -> None:
    fractions = {"Ar": 0.2, "He": 0.3, "CO2": 0.5}
    assert sound_speed_for_model(
        fractions,
        298.15,
        101325.0,
        model_id=A1_SOUND_SPEED_MODEL_ID,
    ) == ideal_gas_sound_speed(fractions, 298.15)
    with pytest.raises(ValueError, match="unsupported sound speed model"):
        sound_speed_for_model(fractions, 298.15, 101325.0, model_id="unknown")
    with pytest.raises(ValueError, match="A2DYN temperature"):
        sound_speed_for_model(fractions, 277.0, 101325.0, model_id=A2DYN_SOUND_SPEED_MODEL_ID)
    with pytest.raises(ValueError, match="A2DYN pressure"):
        sound_speed_for_model(fractions, 298.15, 112001.0, model_id=A2DYN_SOUND_SPEED_MODEL_ID)


def test_ar1_shared_noise_and_drift_are_explicit_and_reproducible() -> None:
    first = generate_ar1_noise(
        32,
        rho=0.7,
        innovation_std=0.2,
        rng=np.random.default_rng(123),
    )
    second = generate_ar1_noise(
        32,
        rho=0.7,
        innovation_std=0.2,
        rng=np.random.default_rng(123),
    )
    assert np.array_equal(first, second)
    shared = generate_shared_noise(
        32,
        rho=0.2,
        innovation_std=0.1,
        channel_loadings=[0.5, 0.25, 0.75],
        rng=np.random.default_rng(9),
    )
    assert shared.shape == (32, 3)
    assert np.allclose(shared[:, 1] / shared[:, 0], 0.5, atol=1.0e-12, equal_nan=False)
    assert np.allclose(linear_sequence_drift(4, intercept=[1.0, 2.0], slope_per_step=[0.1, -0.2]), [[1, 2], [1.1, 1.8], [1.2, 1.6], [1.3, 1.4]])


def test_observation_chain_quantizes_only_at_the_end() -> None:
    clean = np.asarray([[1.0], [1.0]])
    drift = np.asarray([[0.1], [0.2]])
    correlated = np.asarray([[0.03], [0.03]])
    white = np.asarray([[0.02], [0.02]])
    observed = apply_observation_chain(
        clean,
        gain=2.0,
        offset=0.05,
        drift=drift,
        correlated_noise=correlated,
        white_noise=white,
        quantization_resolution=0.1,
    )
    assert np.allclose(observed.reshape(-1), [2.2, 2.3], atol=1.0e-12)
    assert np.array_equal(quantize_signal([1.04, 1.06], 0.1), [1.0, 1.1])


def test_invalid_composition_and_ar_parameter_fail_loudly() -> None:
    with pytest.raises(DynamicPhysicsError):
        validate_composition_pct([100.0, 1.0, 0.0])
    with pytest.raises(DynamicPhysicsError):
        generate_ar1_noise(4, rho=1.0, innovation_std=1.0, rng=np.random.default_rng(1))


def test_small_registered_coolprop_grid_passes_and_failed_gate_is_visible() -> None:
    report = audit_coolprop_sound_speed_grid(
        temperature_values_k=[293.15, 298.15],
        pressure_values_pa=[98000.0, 101325.0],
        simplex_step_pct=10.0,
        max_relative_error=0.005,
        sound_speed_model_id=A1_SOUND_SPEED_MODEL_ID,
    )
    assert report["status"] == "PASS"
    assert report["grid_status"] == "PASS"
    assert report["package_version"] == "8.0.0"
    assert report["query_count"] == 66 * 2 * 2

    failed = audit_coolprop_sound_speed_grid(
        temperature_values_k=[298.15],
        pressure_values_pa=[101325.0],
        simplex_step_pct=10.0,
        max_relative_error=1.0e-12,
        raise_on_failure=False,
        sound_speed_model_id=A1_SOUND_SPEED_MODEL_ID,
    )
    assert failed["status"] == "FAIL"
    with pytest.raises(PhysicsAuditError):
        audit_coolprop_sound_speed_grid(
            temperature_values_k=[298.15],
            pressure_values_pa=[101325.0],
            simplex_step_pct=10.0,
            max_relative_error=1.0e-12,
            sound_speed_model_id=A1_SOUND_SPEED_MODEL_ID,
        )


def test_a2dyn_coolprop_numeric_gate_passes_and_pressure_direction_failure_stays_visible() -> None:
    report = audit_coolprop_sound_speed_grid(
        temperature_values_k=[293.15, 298.15],
        pressure_values_pa=[98000.0, 101325.0],
        simplex_step_pct=10.0,
        max_relative_error=0.005,
        max_workers=1,
        sound_speed_model_id=A2DYN_SOUND_SPEED_MODEL_ID,
        off_grid_count=32,
        off_grid_seed=20260831,
    )
    assert report["status"] == "PASS"
    assert report["grid_status"] == "PASS"
    assert report["off_grid"]["status"] == "PASS"
    assert report["off_grid"]["count"] == 32

    direction_report = audit_coolprop_sound_speed_grid(
        temperature_values_k=[293.15],
        pressure_values_pa=[90000.0, 112000.0],
        simplex_step_pct=10.0,
        max_relative_error=0.005,
        max_workers=1,
        sound_speed_model_id=A2DYN_SOUND_SPEED_MODEL_ID,
        raise_on_failure=False,
        check_pressure_direction=True,
    )
    assert direction_report["grid_status"] == "PASS"
    assert direction_report["status"] == "FAIL"
    assert direction_report["pressure_direction"]["mismatch_count"] > 0
