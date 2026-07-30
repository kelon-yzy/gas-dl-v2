from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import tv3.ml.mrs_varpro as mrs_varpro
from tv3.audit.mrs_ei_registry import (
    AUTHORIZATION_FIELDS,
    FORBIDDEN_AUTH_VALUE,
    verify_evidence_manifest,
)
from tv3.audit.mrs_ei_varpro import (
    B1_S1_FROZEN,
    B2_SOLVER_CORE_VERIFIED,
    run_mei3_b1_audit,
    run_mei3_b2_audit,
)
from tv3.ml.mrs_varpro import (
    S1Problem,
    augmented_residual,
    build_s1_parameterization,
    build_s1_settings,
    build_varpro_parameterization,
    evaluate_varpro,
    finite_difference_jacobian,
    pack_s1_parameters,
    predict_s1,
    run_b1_s1_numerical_audit,
    run_b2_solver_core_audit,
    solve_s1,
    solve_s2,
    solve_s3,
    varpro_projected_jacobian,
)
from tv3.sim.generation.tunnel_ventilation.mrs_observation import ideal_mrs_observation

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "configs" / "tv3_mrs_ei" / "mei3_solver_audit.json"
_STATUS_PATH = _ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
_PARENT_B0 = (
    _ROOT
    / "outputs"
    / "runs"
    / "tv3_mrs_ei"
    / "mei3_varpro_audit"
    / "freezes"
    / "20260729T023618492318Z_bdc968bc2f93"
)
_PARENT_B1 = (
    _ROOT
    / "outputs"
    / "runs"
    / "tv3_mrs_ei"
    / "mei3_varpro_audit"
    / "freezes"
    / "20260729T072036945412Z_646ad3f1c878"
)


def _load_runner():
    path = _ROOT / "scripts" / "run_tv3_mei3_b1_solver_audit.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei3_b1_solver_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_b2_runner():
    path = _ROOT / "scripts" / "run_tv3_mei3_solver_audit.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei3_solver_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _b0_stage_status():
    status = json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
    parent = json.loads((_PARENT_B0 / "mei3_verdict.json").read_text(encoding="utf-8"))
    status["allowed_next_stage"] = "MEI-3_varpro_audit"
    status["mei3"] = {
        "phase": parent["audit"]["phase"],
        "verdict": parent["audit"]["verdict"],
        "passed": parent["audit"]["passed"],
        "freeze_dir": _PARENT_B0.relative_to(_ROOT).as_posix(),
    }
    return status


def _b1_stage_status():
    status = json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
    parent = json.loads((_PARENT_B1 / "mei3_b1_verdict.json").read_text(encoding="utf-8"))
    status["allowed_next_stage"] = "MEI-3_varpro_audit"
    status["mei3"] = {
        "phase": parent["audit"]["phase"],
        "verdict": parent["audit"]["verdict"],
        "passed": parent["audit"]["passed"],
        "freeze_dir": _PARENT_B1.relative_to(_ROOT).as_posix(),
    }
    return status


def _fixture():
    config = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    b1 = config["b1_solver_audit"]
    spec = build_s1_parameterization(config)
    truth = b1["synthetic_truth"]
    parameters = pack_s1_parameters(
        truth["raw3_percent"],
        t_c=truth["t_c"],
        path_length_m=truth["path_length_m"],
        h_rh=truth["h_rh"],
        common_delay_s=truth["common_delay_s"],
        log_amplitude_gain=truth["log_amplitude_gain"],
        per_frequency_offsets=truth["per_frequency_offsets"],
    )
    ideal = ideal_mrs_observation(
        truth["raw3_percent"],
        t_c=truth["t_c"],
        p_mpa=b1["fixed_pressure_mpa"],
        h_rh=truth["h_rh"],
        path_length_m=truth["path_length_m"],
        frequencies_hz=b1["frequencies_hz"],
        phase_branch_cycles=b1["phase_branch_cycles"],
        observation_std=b1["observation_std"],
    )
    empty_problem = S1Problem(
        observation=ideal.vector,
        covariance=ideal.covariance,
        frequencies_hz=ideal.frequencies_hz,
        phase_branch_cycles=ideal.phase_branch_cycles,
        observation_std=b1["observation_std"],
        p_mpa=b1["fixed_pressure_mpa"],
    )
    observation = predict_s1(empty_problem, parameters, spec)
    problem = S1Problem(
        observation=observation,
        covariance=ideal.covariance,
        frequencies_hz=ideal.frequencies_hz,
        phase_branch_cycles=ideal.phase_branch_cycles,
        observation_std=b1["observation_std"],
        p_mpa=b1["fixed_pressure_mpa"],
    )
    return config, b1, spec, parameters, problem


def _initial_parameters(entry, n_frequencies):
    return pack_s1_parameters(
        entry["raw3_percent"],
        t_c=entry["t_c"],
        path_length_m=entry["path_length_m"],
        h_rh=entry["h_rh"],
        common_delay_s=0.0,
        log_amplitude_gain=0.0,
        per_frequency_offsets=np.zeros(n_frequencies),
    )


def test_b1_disposes_uninstantiated_s0_without_creating_a_runtime_method():
    config, b1, _spec, _truth, _problem = _fixture()
    assert b1["status"] == "ready_for_s1_freeze"
    s0 = config["method_matrix"]["S0"]
    assert s0["status"] == "historical_h1_not_instantiated"
    assert s0["execution_policy"] == "non_running_historical_note"
    assert s0["formal_pairing_eligible"] is False
    assert config["comparison_contract"]["running_methods"] == ["S1", "S2", "S3"]
    assert config["explicit_non_goals"][0] == "no_registered_sparse_simulation_generation"


def test_b1_numerical_audit_closes_after_s0_historical_disposition():
    config, _b1, _spec, _truth, _problem = _fixture()
    audit = run_b1_s1_numerical_audit(config)
    assert audit["s1_verified"] is True
    assert audit["s0_historical_disposition_accepted"] is True
    assert audit["b1_closed"] is True
    assert audit["blocking_issues"] == []
    assert audit["primary_comparison"] == ["S1", "S2"]
    assert audit["upper_bound_only"] == ["S3"]
    assert audit["formal_data_generated"] is False
    assert len(audit["multi_initialization_report"]) == 3


def test_b1_contract_audit_freezes_s1_and_preserves_authorizations():
    config, _b1, _spec, _truth, _problem = _fixture()
    status = _b0_stage_status()
    audit = run_mei3_b1_audit(
        project_root=_ROOT,
        config=config,
        parent_b0_freeze_dir=_PARENT_B0,
        current_stage_status=status,
    )
    assert audit["passed"] is True
    assert audit["verdict"] == B1_S1_FROZEN
    assert audit["s0_historical_disposition"]["status"] == (
        "historical_h1_not_instantiated"
    )
    assert audit["running_methods"] == ["S1", "S2", "S3"]
    assert audit["formal_solver_gate_ready"] is False
    assert audit["formal_solver_gate_blocker"] == "mei3_solver_core_not_verified"
    for field in AUTHORIZATION_FIELDS:
        assert audit["authorizations"][field] == FORBIDDEN_AUTH_VALUE


def test_b1_contract_audit_rejects_reintroduced_s0_runtime_method():
    config, _b1, _spec, _truth, _problem = _fixture()
    config["method_matrix"]["S0"]["formal_pairing_eligible"] = True
    status = _b0_stage_status()
    audit = run_mei3_b1_audit(
        project_root=_ROOT,
        config=config,
        parent_b0_freeze_dir=_PARENT_B0,
        current_stage_status=status,
    )
    assert audit["passed"] is False
    assert "S0 historical disposition contract is not closed" in audit["issues"]


def test_b1_runner_creates_verified_freeze_and_promotes_only_mei3(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    output_dir = tmp_path / "freeze"
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_b0_stage_status()), encoding="utf-8")
    before = json.loads(stage_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_b1_solver_audit.py",
            "--parent-b0-freeze-dir",
            str(_PARENT_B0),
            "--output-dir",
            str(output_dir),
            "--stage-status-path",
            str(stage_path),
        ],
    )
    assert runner.main() == 0
    after = json.loads(stage_path.read_text(encoding="utf-8"))
    assert after["mei0"] == before["mei0"]
    assert after["mei1"] == before["mei1"]
    assert after["mei3"]["verdict"] == B1_S1_FROZEN
    assert after["mei3"]["s0_status"] == "historical_h1_not_instantiated"
    assert after["mei3"]["s1_frozen"] is True
    assert after["mei3"]["primary_comparison"] == ["S1", "S2"]
    assert after["mei3"]["upper_bound_only"] == ["S3"]
    assert after["mei3"]["formal_solver_gate_ready"] is False
    assert verify_evidence_manifest(
        output_dir / "evidence_manifest.json", project_root=_ROOT
    ) == []
    assert (output_dir / "s0_historical_disposition.json").is_file()
    assert (output_dir / "s1_multi_initialization_report.json").is_file()


def test_b1_runner_refuses_existing_output(tmp_path, monkeypatch):
    runner = _load_runner()
    output_dir = tmp_path / "freeze"
    output_dir.mkdir()
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_b0_stage_status()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_b1_solver_audit.py",
            "--parent-b0-freeze-dir",
            str(_PARENT_B0),
            "--output-dir",
            str(output_dir),
            "--stage-status-path",
            str(stage_path),
        ],
    )
    assert runner.main() == 4


def test_s1_prediction_rejects_nonclosure_at_solver_boundary():
    _config, _b1, spec, truth, problem = _fixture()
    invalid = truth.copy()
    invalid[:2] = [80.0, 80.0]
    with pytest.raises(ValueError, match="physical domain"):
        predict_s1(problem, invalid, spec)


def test_s1_scaled_jacobian_matches_objective_directional_derivative():
    _config, _b1, spec, truth, problem = _fixture()
    values = truth.copy()
    values[0] += 0.2
    residual = augmented_residual(problem, values, spec)
    jacobian = finite_difference_jacobian(problem, values, spec)
    direction = np.linspace(-0.2, 0.2, values.size)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-5
    plus = values + epsilon * direction * spec.scales
    minus = values - epsilon * direction * spec.scales
    objective_plus = 0.5 * np.sum(augmented_residual(problem, plus, spec) ** 2)
    objective_minus = 0.5 * np.sum(augmented_residual(problem, minus, spec) ** 2)
    finite_difference = (objective_plus - objective_minus) / (2.0 * epsilon)
    analytic_from_jacobian = float((jacobian.T @ residual) @ direction)
    np.testing.assert_allclose(
        analytic_from_jacobian, finite_difference, rtol=2e-5, atol=1e-7
    )


def test_s1_frozen_multi_initializations_converge_to_same_physical_solution():
    config, b1, spec, _truth, problem = _fixture()
    settings = build_s1_settings(config)
    solutions = [
        solve_s1(
            problem,
            _initial_parameters(entry, len(b1["frequencies_hz"])),
            spec,
            settings,
        )
        for entry in b1["frozen_initializations"]
    ]
    assert all(solution.success for solution in solutions)
    raw3 = np.vstack([solution.raw3_percent for solution in solutions])
    assert np.max(np.ptp(raw3, axis=0)) < 1e-4
    objectives = np.asarray([solution.objective for solution in solutions])
    assert np.ptp(objectives) < 1e-10
    assert np.all(np.isfinite(objectives))
    assert all(solution.forward_calls > 0 for solution in solutions)


def test_s1_physical_solution_is_invariant_to_parameter_scale_units():
    config, b1, spec, _truth, problem = _fixture()
    initial = _initial_parameters(
        b1["frozen_initializations"][0], len(b1["frequencies_hz"])
    )
    reference = solve_s1(problem, initial, spec, build_s1_settings(config))
    changed_scales = spec.scales * np.asarray(
        [3.0, 0.5, 2.0, 4.0, 0.25, 5.0, 0.4, 2.0, 0.5, 3.0, 0.75]
    )
    reparameterized = build_s1_parameterization(config, scales=changed_scales)
    changed = solve_s1(
        problem, initial, reparameterized, build_s1_settings(config)
    )
    assert reference.success and changed.success
    np.testing.assert_allclose(
        changed.raw3_percent, reference.raw3_percent, rtol=0.0, atol=2e-4
    )
    np.testing.assert_allclose(
        changed.parameters[2:], reference.parameters[2:], rtol=0.0, atol=2e-5
    )


def test_b2_projected_jacobian_matches_independent_central_difference():
    config, _b1, spec, truth, problem = _fixture()
    varpro = build_varpro_parameterization(config, spec)
    beta = truth[varpro.nonlinear_indices].copy()
    beta += np.asarray([0.1, -0.1, 0.02, 1e-5, 0.1])
    actual = varpro_projected_jacobian(problem, beta, spec, varpro)
    expected = np.empty_like(actual)
    for column, full_index in enumerate(varpro.nonlinear_indices):
        step = spec.finite_difference_steps[full_index] * 0.25
        plus = beta.copy()
        minus = beta.copy()
        plus[column] += step
        minus[column] -= step
        expected[:, column] = (
            evaluate_varpro(problem, plus, spec, varpro).residual
            - evaluate_varpro(problem, minus, spec, varpro).residual
        ) / (2.0 * step)
    expected *= spec.scales[varpro.nonlinear_indices][np.newaxis, :]
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-7)


def test_s2_interior_jacobian_uses_two_predictions_per_nonlinear_parameter(monkeypatch):
    config, _b1, spec, truth, problem = _fixture()
    varpro = build_varpro_parameterization(config, spec)
    beta = truth[varpro.nonlinear_indices].copy()
    beta += np.asarray([0.1, -0.1, 0.02, 1e-5, 0.1])
    original = mrs_varpro._nonlinear_prediction
    calls = 0

    def counted_prediction(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mrs_varpro, "_nonlinear_prediction", counted_prediction)
    varpro_projected_jacobian(problem, beta, spec, varpro)
    assert calls == 2 * beta.size


def test_s1_forward_call_counter_matches_physical_residual_evaluations(monkeypatch):
    config, b1, spec, _truth, problem = _fixture()
    initial = _initial_parameters(b1["frozen_initializations"][0], 4)
    original = mrs_varpro.augmented_residual
    calls = 0

    def counted_residual(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mrs_varpro, "augmented_residual", counted_residual)
    solution = solve_s1(problem, initial, spec, build_s1_settings(config))
    assert solution.forward_calls == calls


def test_s2_forward_call_counter_matches_physical_predictions(monkeypatch):
    config, b1, spec, _truth, problem = _fixture()
    varpro = build_varpro_parameterization(config, spec)
    initial = _initial_parameters(b1["frozen_initializations"][0], 4)
    original = mrs_varpro._nonlinear_prediction
    calls = 0

    def counted_prediction(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mrs_varpro, "_nonlinear_prediction", counted_prediction)
    solution = solve_s2(
        problem,
        initial,
        spec,
        build_s1_settings(config),
        varpro,
        max_phase_branch_standardized_error=8.0,
    )
    assert solution.forward_calls == calls


def test_b2_s1_and_s2_reach_the_same_augmented_solution():
    config, b1, spec, _truth, problem = _fixture()
    varpro = build_varpro_parameterization(config, spec)
    initial = _initial_parameters(b1["frozen_initializations"][0], 4)
    settings = build_s1_settings(config)
    s1 = solve_s1(problem, initial, spec, settings)
    s2 = solve_s2(
        problem,
        initial,
        spec,
        settings,
        varpro,
        max_phase_branch_standardized_error=8.0,
    )
    assert s1.success and s2.success
    np.testing.assert_allclose(s2.objective, s1.objective, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(s2.parameters, s1.parameters, rtol=0.0, atol=2e-5)
    np.testing.assert_allclose(s2.raw3_percent, s1.raw3_percent, rtol=0.0, atol=2e-5)


def test_b2_s3_requires_explicit_truth_nuisance_parameters():
    config, b1, spec, _truth, problem = _fixture()
    varpro = build_varpro_parameterization(config, spec)
    initial = _initial_parameters(b1["frozen_initializations"][0], 4)
    with pytest.raises(ValueError, match="explicitly isolated truth"):
        solve_s3(
            problem,
            initial,
            spec,
            build_s1_settings(config),
            varpro,
            truth_linear_parameters=None,
            max_phase_branch_standardized_error=8.0,
        )


def test_b2_mechanism_audit_passes_all_required_negative_controls():
    config, _b1, _spec, _truth, _problem = _fixture()
    audit = run_b2_solver_core_audit(config)
    assert audit["solver_core_verified"] is True
    assert audit["formal_data_generated"] is False
    assert set(audit["negative_controls"]) == set(
        config["b2_solver_core"]["required_negative_controls"]
    )
    assert all(
        result["failed_as_required"] for result in audit["negative_controls"].values()
    )


def test_b2_contract_audit_verifies_parent_and_preserves_authorizations():
    config, _b1, _spec, _truth, _problem = _fixture()
    audit = run_mei3_b2_audit(
        project_root=_ROOT,
        config=config,
        parent_b1_freeze_dir=_PARENT_B1,
        current_stage_status=_b1_stage_status(),
    )
    assert audit["passed"] is True
    assert audit["verdict"] == B2_SOLVER_CORE_VERIFIED
    assert audit["formal_solver_gate_ready"] is False
    assert audit["formal_solver_gate_blocker"] == (
        "mei3_registered_data_authorization_ready_package_not_frozen"
    )
    for field in AUTHORIZATION_FIELDS:
        assert audit["authorizations"][field] == FORBIDDEN_AUTH_VALUE


def test_b2_runner_creates_append_only_freeze_and_updates_only_mei3(
    tmp_path, monkeypatch
):
    runner = _load_b2_runner()
    output_dir = tmp_path / "freeze"
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_b1_stage_status()), encoding="utf-8")
    before = json.loads(stage_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_solver_audit.py",
            "--parent-b1-freeze-dir",
            str(_PARENT_B1),
            "--output-dir",
            str(output_dir),
            "--stage-status-path",
            str(stage_path),
        ],
    )
    assert runner.main() == 0
    after = json.loads(stage_path.read_text(encoding="utf-8"))
    assert after["mei0"] == before["mei0"]
    assert after["mei1"] == before["mei1"]
    assert after["mei3"]["verdict"] == B2_SOLVER_CORE_VERIFIED
    assert after["mei3"]["s1_frozen"] is True
    assert after["mei3"]["s2_core_verified"] is True
    assert after["mei3"]["s3_upper_bound_verified"] is True
    assert after["mei3"]["formal_solver_gate_ready"] is False
    assert verify_evidence_manifest(
        output_dir / "evidence_manifest.json", project_root=_ROOT
    ) == []
    assert (output_dir / "projected_jacobian_report.json").is_file()
    assert (output_dir / "negative_controls_report.json").is_file()


def test_b2_runner_refuses_existing_output(tmp_path, monkeypatch):
    runner = _load_b2_runner()
    output_dir = tmp_path / "freeze"
    output_dir.mkdir()
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_b1_stage_status()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_solver_audit.py",
            "--parent-b1-freeze-dir",
            str(_PARENT_B1),
            "--output-dir",
            str(output_dir),
            "--stage-status-path",
            str(stage_path),
        ],
    )
    assert runner.main() == 4
