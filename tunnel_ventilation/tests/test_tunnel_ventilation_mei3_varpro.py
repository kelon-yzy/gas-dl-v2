from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tv3.audit.mrs_ei_registry import (
    AUTHORIZATION_FIELDS,
    FORBIDDEN_AUTH_VALUE,
    load_json,
    verify_evidence_manifest,
)
from tv3.audit.mrs_ei_varpro import (
    B0_REPRESENTATION_CLOSED,
    PHASE_A_SUPPORTED,
    assess_linear_candidates,
    run_b0_observation_operator_audit,
    run_b0_raw3_rank_audit,
    run_mei3_b0_audit,
    run_mei3_phase_a_audit,
    run_numerical_equivalence_audit,
    solve_conditionally_linear,
)
from tv3.sim.generation.tunnel_ventilation.mrs_observation import (
    RAW3_TANGENT_BASIS,
    ideal_mrs_observation,
    raw3_percent_from_tangent,
    raw3_tangent_coordinates,
    validate_raw3_percent,
)

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "configs" / "tv3_mrs_ei" / "mei3_varpro_audit.json"
_STATUS_PATH = _ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
_PARENT = (
    _ROOT
    / "outputs"
    / "runs"
    / "tv3_mrs_ei"
    / "mei1_forward_envelope"
    / "freezes"
    / "20260728T064100731550Z_1b55aa2e09cb"
)


def _load_runner():
    path = _ROOT / "scripts" / "run_tv3_mei3_varpro_audit.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei3_varpro_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mei1_stage_status():
    status = load_json(_STATUS_PATH)
    status["allowed_next_stage"] = "MEI-3_varpro_audit"
    return status


def test_conditionally_linear_solution_matches_closed_form():
    observation = np.array([1.2, 2.1, 3.3])
    baseline = np.array([0.2, 0.1, 0.3])
    matrix = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    covariance = np.diag([0.2, 0.3, 0.4]) ** 2
    prior_mean = np.array([0.0, 0.0])
    prior_std = np.array([2.0, 3.0])
    result = solve_conditionally_linear(
        observation=observation,
        nonlinear_prediction=baseline,
        design_matrix=matrix,
        covariance=covariance,
        prior_mean=prior_mean,
        prior_std=prior_std,
    )
    weight = np.linalg.inv(covariance)
    precision = np.diag(1.0 / prior_std**2)
    expected = np.linalg.solve(
        matrix.T @ weight @ matrix + precision,
        matrix.T @ weight @ (observation - baseline) + precision @ prior_mean,
    )
    assert np.allclose(result.parameters, expected, rtol=1e-12, atol=1e-12)
    assert result.augmented_rank == 2


def test_conditionally_linear_solution_rejects_non_positive_definite_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        solve_conditionally_linear(
            observation=[1.0, 2.0],
            nonlinear_prediction=[0.0, 0.0],
            design_matrix=np.ones((2, 1)),
            covariance=np.array([[1.0, 2.0], [2.0, 1.0]]),
            prior_mean=[0.0],
            prior_std=[1.0],
        )


def test_candidate_assessment_requires_unwrapped_phase():
    config = load_json(_CONFIG_PATH)
    supported = assess_linear_candidates(config)
    assert supported["common_delay"]["supported"] is True
    assert supported["log_amplitude_gain"]["supported"] is True
    assert supported["per_frequency_calibration_offset"]["supported"] is True
    assert supported["path_length_m"]["admitted_to_linear_block"] is False
    assert supported["complex_transfer_delay"]["supported"] is False

    changed = json.loads(json.dumps(config))
    changed["observation_representation"]["phase"] = "complex_transfer_real_imag"
    assert assess_linear_candidates(changed)["common_delay"]["supported"] is False


def test_per_frequency_offset_requires_shared_hierarchy_and_prior():
    config = load_json(_CONFIG_PATH)
    changed = json.loads(json.dumps(config))
    changed["conditionally_linear_blocks"]["per_frequency_calibration_offset"][
        "sharing"
    ] = ["sample_id", "frequency_hz"]
    assert (
        assess_linear_candidates(changed)["per_frequency_calibration_offset"][
            "supported"
        ]
        is False
    )


def test_common_delay_and_gain_require_device_view_sharing():
    config = load_json(_CONFIG_PATH)
    changed = json.loads(json.dumps(config))
    changed["conditionally_linear_blocks"]["common_delay"]["sharing"] = [
        "mixture_id",
        "view_id",
    ]
    assert assess_linear_candidates(changed)["common_delay"]["supported"] is False
    changed = json.loads(json.dumps(config))
    changed["conditionally_linear_blocks"]["log_amplitude_gain"]["sharing"] = [
        "mixture_id",
        "view_id",
    ]
    assert assess_linear_candidates(changed)["log_amplitude_gain"]["supported"] is False


def test_numerical_equivalence_fixture_passes_without_formal_data():
    result = run_numerical_equivalence_audit(load_json(_CONFIG_PATH))
    assert result["passed"] is True
    assert result["n_observations"] == 36
    assert result["n_linear_parameters"] == 6
    assert result["full_augmented_column_rank"] is True
    assert result["role"] == "in_memory_nonformal_numerical_equivalence_only"


def test_raw3_tangent_coordinates_preserve_all_three_outputs_and_closure():
    raw3 = np.array([2.515, 19.6, 77.885])
    coordinates = raw3_tangent_coordinates(raw3)
    reconstructed = raw3_percent_from_tangent(coordinates)
    np.testing.assert_allclose(reconstructed, raw3, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(RAW3_TANGENT_BASIS.sum(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(
        RAW3_TANGENT_BASIS.T @ RAW3_TANGENT_BASIS,
        np.eye(2),
        rtol=0.0,
        atol=1e-15,
    )


def test_observation_operator_rejects_nonclosure_instead_of_normalizing():
    config = load_json(_CONFIG_PATH)
    fixture = config["b0_representation_audit"]
    point = fixture["points"][0]
    raw3 = np.array([point["co2_percent"], point["o2_percent"], point["n2_percent"]])
    with pytest.raises(ValueError, match="sum=100"):
        validate_raw3_percent(raw3 * 1.03)
    with pytest.raises(ValueError, match="sum=100"):
        ideal_mrs_observation(
            raw3 * 1.03,
            t_c=point["t_c"],
            p_mpa=point["p_mpa"],
            h_rh=point["h_rh"],
            path_length_m=point["path_length_m"],
            frequencies_hz=fixture["frequencies_hz"],
            phase_branch_cycles=fixture["phase_branch_cycles"],
            observation_std=fixture["observation_std"],
        )


def test_observation_operator_requires_explicit_integer_phase_branches():
    config = load_json(_CONFIG_PATH)
    fixture = config["b0_representation_audit"]
    point = fixture["points"][0]
    raw3 = [point["co2_percent"], point["o2_percent"], point["n2_percent"]]
    with pytest.raises(ValueError, match="integer per frequency"):
        ideal_mrs_observation(
            raw3,
            t_c=point["t_c"],
            p_mpa=point["p_mpa"],
            h_rh=point["h_rh"],
            path_length_m=point["path_length_m"],
            frequencies_hz=fixture["frequencies_hz"],
            phase_branch_cycles=[0.0, 0.0, 0.0, 0.0],
            observation_std=fixture["observation_std"],
        )


def test_b0_observation_and_constrained_rank_audits_pass():
    config = load_json(_CONFIG_PATH)
    observation = run_b0_observation_operator_audit(config)
    rank = run_b0_raw3_rank_audit(config)
    assert observation["passed"] is True
    assert observation["scaled_nonclosure_input_rejected"] is True
    assert observation["silent_normalization"] is False
    assert rank["passed"] is True
    assert rank["raw_output_dimension"] == 3
    assert rank["effective_parameter_dimension"] == 2
    assert rank["n2_backfill"] is False
    for point in rank["points"]:
        assert set(point["rank_by_relative_tolerance"].values()) == {2}


def test_b0_audit_closes_representation_without_changing_authorizations():
    audit = run_mei3_b0_audit(
        project_root=_ROOT,
        config=load_json(_CONFIG_PATH),
        parent_mei1_freeze_dir=_PARENT,
        current_stage_status=_mei1_stage_status(),
    )
    assert audit["verdict"] == B0_REPRESENTATION_CLOSED
    assert audit["passed"] is True
    assert audit["composition_contract"]["posthoc_projection"] is False
    assert audit["composition_contract"]["n2_closure_backfill"] is False
    assert audit["formal_solver_gate_blocker"] == "mei3_solver_core_not_verified"
    for field in AUTHORIZATION_FIELDS:
        assert audit["authorizations"][field] == FORBIDDEN_AUTH_VALUE


def test_mei3_phase_a_audit_accepts_current_parent_and_preserves_authorizations():
    audit = run_mei3_phase_a_audit(
        project_root=_ROOT,
        config=load_json(_CONFIG_PATH),
        parent_mei1_freeze_dir=_PARENT,
        current_stage_status=_mei1_stage_status(),
    )
    assert audit["verdict"] == PHASE_A_SUPPORTED
    assert audit["passed"] is True
    assert audit["issues"] == []
    assert audit["allowed_next_stage"] == "MEI-3_varpro_audit"
    assert audit["formal_solver_gate_ready"] is False
    for field in AUTHORIZATION_FIELDS:
        assert audit["authorizations"][field] == FORBIDDEN_AUTH_VALUE


def test_mei3_phase_a_rejects_stale_parent_pointer():
    status = _mei1_stage_status()
    status["mei1"]["freeze_dir"] = "outputs/runs/tv3_mrs_ei/mei1/other"
    audit = run_mei3_phase_a_audit(
        project_root=_ROOT,
        config=load_json(_CONFIG_PATH),
        parent_mei1_freeze_dir=_PARENT,
        current_stage_status=status,
    )
    assert audit["passed"] is False
    assert any("same parent MEI-1 freeze" in issue for issue in audit["issues"])


def test_mei3_phase_a_does_not_pass_with_partial_linear_structure():
    config = load_json(_CONFIG_PATH)
    config["observation_representation"]["phase"] = "complex_transfer_real_imag"
    audit = run_mei3_phase_a_audit(
        project_root=_ROOT,
        config=config,
        parent_mei1_freeze_dir=_PARENT,
        current_stage_status=_mei1_stage_status(),
    )
    assert audit["passed"] is False
    assert audit["verdict"] == "mei3_varpro_not_applicable"
    assert "common_delay" not in audit["admitted_linear_blocks"]


def test_mei3_runner_creates_verified_freeze_and_updates_only_mei3_status(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    output_dir = tmp_path / "freeze"
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_mei1_stage_status()), encoding="utf-8")
    before = load_json(stage_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_varpro_audit.py",
            "--parent-mei1-freeze-dir",
            str(_PARENT),
            "--output-dir",
            str(output_dir),
            "--stage-status-path",
            str(stage_path),
        ],
    )
    assert runner.main() == 0
    after = load_json(stage_path)
    assert after["mei0"] == before["mei0"]
    assert after["mei1"] == before["mei1"]
    assert after["mei3"]["verdict"] == B0_REPRESENTATION_CLOSED
    assert after["mei3"]["phase"] == "b0_representation_audit"
    assert after["allowed_next_stage"] == "MEI-3_varpro_audit"
    assert (output_dir / "mei3_observation_operator_audit.json").is_file()
    assert (output_dir / "mei3_raw3_forward_rank_audit.json").is_file()
    assert verify_evidence_manifest(
        output_dir / "evidence_manifest.json", project_root=_ROOT
    ) == []


def test_mei3_runner_refuses_existing_output(tmp_path, monkeypatch):
    runner = _load_runner()
    output_dir = tmp_path / "freeze"
    output_dir.mkdir()
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_mei1_stage_status()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_varpro_audit.py",
            "--parent-mei1-freeze-dir",
            str(_PARENT),
            "--output-dir",
            str(output_dir),
            "--stage-status-path",
            str(stage_path),
        ],
    )
    assert runner.main() == 4
