"""MEI-3 Phase A audit for separable nuisance parameters."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from tv3.audit.mrs_ei_registry import (
    AUTHORIZATION_FIELDS,
    FORBIDDEN_AUTH_VALUE,
    REGISTRY_SCHEMA_VERSION,
    load_json,
    sha256_file,
    verify_evidence_manifest,
)
from tv3.sim.generation.tunnel_ventilation.mrs_observation import (
    OBSERVATION_FIELDS,
    OBSERVATION_UNITS,
    RAW3_TANGENT_BASIS,
    ideal_mrs_observation,
    raw3_percent_from_tangent,
    raw3_tangent_coordinates,
)
from tv3.ml.mrs_varpro import (
    ConditionalLinearSolution,
    run_b1_s1_numerical_audit,
    run_b2_solver_core_audit,
    solve_conditionally_linear,
)

STAGE = "MEI-3"
PHASE = "phase_a_structure_audit"
PHASE_A_SUPPORTED = "mei3_phase_a_structure_supported"
VARPRO_NOT_APPLICABLE = "mei3_varpro_not_applicable"
B0_PHASE = "b0_representation_audit"
B0_REPRESENTATION_CLOSED = "mei3_b0_representation_closed"
B0_REPRESENTATION_INVALID = "mei3_b0_representation_invalid"
B1_PHASE = "b1_s1_freeze"
B1_S1_FROZEN = "mei3_b1_s1_frozen"
B1_S1_INVALID = "mei3_b1_s1_invalid"
B2_PHASE = "b2_solver_core_audit"
B2_SOLVER_CORE_VERIFIED = "mei3_solver_core_verified"
B2_SOLVER_CORE_INVALID = "mei3_solver_core_invalid"


def _joint_normal_equation_reference(
    *,
    observation: np.ndarray,
    nonlinear_prediction: np.ndarray,
    design_matrix: np.ndarray,
    covariance: np.ndarray,
    prior_mean: np.ndarray,
    prior_std: np.ndarray,
) -> ConditionalLinearSolution:
    chol = np.linalg.cholesky(covariance)
    matrix = np.linalg.solve(chol, design_matrix)
    target = np.linalg.solve(chol, observation - nonlinear_prediction)
    prior_matrix = np.diag(1.0 / prior_std)
    normal = matrix.T @ matrix + prior_matrix.T @ prior_matrix
    rhs = matrix.T @ target + prior_matrix.T @ prior_matrix @ prior_mean
    parameters = np.linalg.solve(normal, rhs)
    singular_values = np.linalg.svd(np.vstack((matrix, prior_matrix)), compute_uv=False)
    return ConditionalLinearSolution(
        parameters=parameters,
        whitened_data_residual=target - matrix @ parameters,
        prior_residual=prior_matrix @ (prior_mean - parameters),
        augmented_rank=int(np.linalg.matrix_rank(np.vstack((matrix, prior_matrix)))),
        augmented_condition_number=float(singular_values[0] / singular_values[-1]),
    )


def assess_linear_candidates(config: dict[str, Any]) -> dict[str, Any]:
    representation = config["observation_representation"]
    blocks = config["conditionally_linear_blocks"]
    phase = representation.get("phase")
    delay = blocks["common_delay"]
    gain = blocks["log_amplitude_gain"]
    common_delay_supported = (
        representation.get("tof") == "raw_tof_s"
        and phase == "unwrapped_phase_rad"
        and bool(representation.get("phase_unwrapping_assumption"))
        and delay.get("sharing") == ["device_profile_id", "view_id"]
        and float(delay.get("prior_std", 0.0)) > 0.0
    )
    log_gain_supported = (
        representation.get("amplitude") == "log_amplitude"
        and gain.get("sharing") == ["device_profile_id", "view_id"]
        and float(gain.get("prior_std", 0.0)) > 0.0
    )
    offset = blocks["per_frequency_calibration_offset"]
    offset_supported = (
        offset.get("modality") == "log_amplitude"
        and offset.get("sharing") == ["device_profile_id", "frequency_hz"]
        and offset.get("independent_calibration_prior_required") is True
        and float(offset.get("prior_std", 0.0)) > 0.0
    )
    path_length_kept_nonlinear = (
        "path_length_m" in config.get("nonlinear_block", [])
        and config.get("path_length_policy")
        == "keep_in_beta_for_combined_observation_audit"
    )
    return {
        "common_delay": {
            "supported": common_delay_supported,
            "reason": "affine_in_raw_tof_and_fixed_branch_unwrapped_phase_with_device_view_sharing",
        },
        "log_amplitude_gain": {
            "supported": log_gain_supported,
            "reason": "additive_in_log_amplitude_with_device_view_sharing",
        },
        "per_frequency_calibration_offset": {
            "supported": offset_supported,
            "reason": "shared_across_samples_with_independent_calibration_prior",
        },
        "path_length_m": {
            "supported": False,
            "admitted_to_linear_block": False,
            "reason": "frozen_combined_observation_policy_keeps_path_length_in_beta",
            "policy_satisfied": path_length_kept_nonlinear,
        },
        "complex_transfer_delay": {
            "supported": False,
            "reason": "delay_enters_complex_real_imag_through_sine_and_cosine",
        },
    }


def _build_numerical_fixture(config: dict[str, Any]) -> dict[str, np.ndarray | list[str]]:
    fixture = config["synthetic_numerical_fixture"]
    frequencies = np.asarray(fixture["frequencies_hz"], dtype=np.float64)
    conditions = fixture["conditions"]
    n_freq = frequencies.size
    parameter_names = [
        "common_delay_s",
        "log_amplitude_gain",
        *[f"log_amplitude_offset_{int(f)}hz" for f in frequencies],
    ]
    n_params = len(parameter_names)
    n_rows = len(conditions) * n_freq * 3
    baseline = np.empty(n_rows, dtype=np.float64)
    matrix = np.zeros((n_rows, n_params), dtype=np.float64)
    std = np.empty(n_rows, dtype=np.float64)
    branches = np.zeros(n_freq, dtype=np.int64)

    row = 0
    for condition in conditions:
        co2 = float(condition["co2_percent"])
        o2 = float(condition["o2_percent"])
        n2 = 100.0 - co2 - o2
        ideal = ideal_mrs_observation(
            [co2, o2, n2],
            t_c=float(condition["t_c"]),
            p_mpa=float(condition["p_mpa"]),
            h_rh=float(condition["h_rh"]),
            path_length_m=float(condition["path_length_m"]),
            frequencies_hz=frequencies,
            phase_branch_cycles=branches,
            observation_std=fixture["observation_std"],
        )
        for freq_index, frequency in enumerate(frequencies):
            baseline[row] = float(ideal.raw_tof_s[freq_index])
            matrix[row, 0] = 1.0
            std[row] = float(fixture["observation_std"]["raw_tof_s"])
            row += 1

            baseline[row] = float(ideal.log_amplitude[freq_index])
            matrix[row, 1] = 1.0
            matrix[row, 2 + freq_index] = 1.0
            std[row] = float(fixture["observation_std"]["log_amplitude"])
            row += 1

            baseline[row] = float(ideal.unwrapped_phase_rad[freq_index])
            matrix[row, 0] = -2.0 * math.pi * frequency
            std[row] = float(fixture["observation_std"]["unwrapped_phase_rad"])
            row += 1

    injected = fixture["injected_linear_parameters"]
    truth = np.asarray(
        [
            injected["common_delay_s"],
            injected["log_amplitude_gain"],
            *injected["per_frequency_calibration_offset"],
        ],
        dtype=np.float64,
    )
    if truth.size != n_params:
        raise ValueError("injected linear parameter count does not match K4 fixture")
    prior_mean = np.zeros(n_params, dtype=np.float64)
    blocks = config["conditionally_linear_blocks"]
    prior_std = np.asarray(
        [
            blocks["common_delay"]["prior_std"],
            blocks["log_amplitude_gain"]["prior_std"],
            *([blocks["per_frequency_calibration_offset"]["prior_std"]] * n_freq),
        ],
        dtype=np.float64,
    )
    observation = baseline + matrix @ truth
    return {
        "observation": observation,
        "baseline": baseline,
        "design_matrix": matrix,
        "covariance": np.diag(std**2),
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "parameter_names": parameter_names,
    }


def run_numerical_equivalence_audit(config: dict[str, Any]) -> dict[str, Any]:
    fixture = _build_numerical_fixture(config)
    kwargs = {
        "observation": fixture["observation"],
        "nonlinear_prediction": fixture["baseline"],
        "design_matrix": fixture["design_matrix"],
        "covariance": fixture["covariance"],
        "prior_mean": fixture["prior_mean"],
        "prior_std": fixture["prior_std"],
    }
    projected = solve_conditionally_linear(**kwargs)
    joint = _joint_normal_equation_reference(**kwargs)
    parameter_difference = float(np.max(np.abs(projected.parameters - joint.parameters)))
    residual_difference = float(
        np.max(np.abs(projected.augmented_residual - joint.augmented_residual))
    )
    gates = config["numerical_gates"]
    full_rank = projected.augmented_rank == projected.parameters.size
    passed = (
        parameter_difference
        <= float(gates["max_parameter_difference_vs_joint_reference"])
        and residual_difference
        <= float(gates["max_projected_residual_difference_vs_joint_reference"])
        and (full_rank or not gates["require_full_augmented_column_rank"])
    )
    return {
        "passed": bool(passed),
        "n_observations": int(np.asarray(fixture["observation"]).size),
        "n_linear_parameters": int(projected.parameters.size),
        "parameter_names": fixture["parameter_names"],
        "max_parameter_difference_vs_joint_reference": parameter_difference,
        "max_projected_residual_difference_vs_joint_reference": residual_difference,
        "augmented_rank": projected.augmented_rank,
        "full_augmented_column_rank": full_rank,
        "augmented_condition_number": projected.augmented_condition_number,
        "projected_parameters": projected.parameters.tolist(),
        "role": "in_memory_nonformal_numerical_equivalence_only",
    }


def _b0_point_raw3(point: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [point["co2_percent"], point["o2_percent"], point["n2_percent"]],
        dtype=np.float64,
    )


def _b0_observation(
    config: dict[str, Any],
    point: dict[str, Any],
    raw3_percent: np.ndarray,
):
    fixture = config["b0_representation_audit"]
    frequencies = fixture["frequencies_hz"]
    return ideal_mrs_observation(
        raw3_percent,
        t_c=float(point["t_c"]),
        p_mpa=float(point["p_mpa"]),
        h_rh=float(point["h_rh"]),
        path_length_m=float(point["path_length_m"]),
        frequencies_hz=frequencies,
        phase_branch_cycles=fixture["phase_branch_cycles"],
        observation_std=fixture["observation_std"],
    )


def run_b0_observation_operator_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Audit the single legal observation path and its explicit unit contract."""
    contract = config["composition_parameterization"]
    fixture = config["b0_representation_audit"]
    point = fixture["points"][0]
    raw3 = _b0_point_raw3(point)
    observation = _b0_observation(config, point, raw3)
    covariance_eigenvalues = np.linalg.eigvalsh(observation.covariance)

    scaled_input_rejected = False
    try:
        _b0_observation(config, point, raw3 * 1.03)
    except ValueError as exc:
        scaled_input_rejected = "sum=100" in str(exc)

    expected_rows = len(fixture["frequencies_hz"]) * len(OBSERVATION_FIELDS)
    amendment = contract.get("scoped_contract_amendment") or {}
    contract_satisfied = (
        contract.get("scope") == "deterministic_mei3_s0_s1_s2_s3_only"
        and contract.get("input_representation") == "dry_raw3_percent"
        and contract.get("output_representation") == "dry_raw3_percent"
        and contract.get("output_dimension") == 3
        and contract.get("effective_dimension") == 2
        and contract.get("sum_percent") == 100.0
        and contract.get("nonnegative") is True
        and contract.get("enforcement") == "equality_constrained_tangent_space"
        and contract.get("posthoc_projection") is False
        and contract.get("n2_closure_backfill") is False
        and len(amendment.get("supersedes_for_this_scope") or []) == 2
        and set(amendment.get("does_not_change") or [])
        == {
            "training_labels",
            "learned_model_output_contracts",
            "primary_metric_units",
            "deployment_output_dimension",
        }
    )
    passed = (
        contract_satisfied
        and scaled_input_rejected
        and observation.vector.shape == (expected_rows,)
        and observation.covariance.shape == (expected_rows, expected_rows)
        and tuple(observation.row_fields) == OBSERVATION_FIELDS * len(fixture["frequencies_hz"])
        and tuple(observation.row_units) == OBSERVATION_UNITS * len(fixture["frequencies_hz"])
        and bool(np.all(covariance_eigenvalues > 0.0))
    )
    return {
        "passed": bool(passed),
        "operator": "ideal_mrs_observation",
        "row_order": "frequency_major_then_raw_tof_log_amplitude_unwrapped_phase",
        "row_fields": list(observation.row_fields),
        "row_units": list(observation.row_units),
        "n_observations": int(observation.vector.size),
        "covariance_positive_definite": bool(np.all(covariance_eigenvalues > 0.0)),
        "phase_branch_source": fixture["phase_branch_source"],
        "scaled_nonclosure_input_rejected": scaled_input_rejected,
        "silent_normalization": False,
        "posthoc_projection": False,
        "contract_satisfied": bool(contract_satisfied),
    }


def run_b0_raw3_rank_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Report Jacobian rank on the registered two-dimensional feasible tangent."""
    fixture = config["b0_representation_audit"]
    step = float(fixture["tangent_finite_difference_step_percent"])
    tolerances = tuple(float(value) for value in fixture["rank_relative_tolerances"])
    if step <= 0.0 or not tolerances or any(value <= 0.0 for value in tolerances):
        raise ValueError("B0 finite-difference step and rank tolerances must be positive")

    basis_gram = RAW3_TANGENT_BASIS.T @ RAW3_TANGENT_BASIS
    basis_sum = np.sum(RAW3_TANGENT_BASIS, axis=0)
    points: list[dict[str, Any]] = []
    for point in fixture["points"]:
        raw3 = _b0_point_raw3(point)
        coordinates = raw3_tangent_coordinates(raw3)
        baseline = _b0_observation(config, point, raw3)
        chol = np.linalg.cholesky(baseline.covariance)
        jacobian = np.empty((baseline.vector.size, 2), dtype=np.float64)
        for column in range(2):
            delta = np.zeros(2, dtype=np.float64)
            delta[column] = step
            plus = _b0_observation(
                config, point, raw3_percent_from_tangent(coordinates + delta)
            ).vector
            minus = _b0_observation(
                config, point, raw3_percent_from_tangent(coordinates - delta)
            ).vector
            jacobian[:, column] = np.linalg.solve(chol, (plus - minus) / (2.0 * step))

        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        ranks = [
            int(np.count_nonzero(singular_values > tolerance * singular_values[0]))
            for tolerance in tolerances
        ]
        points.append(
            {
                "raw3_percent": raw3.tolist(),
                "t_c": float(point["t_c"]),
                "p_mpa": float(point["p_mpa"]),
                "h_rh": float(point["h_rh"]),
                "path_length_m": float(point["path_length_m"]),
                "singular_values": singular_values.tolist(),
                "rank_by_relative_tolerance": {
                    f"{tolerance:.0e}": rank
                    for tolerance, rank in zip(tolerances, ranks, strict=True)
                },
                "full_effective_rank": all(rank == 2 for rank in ranks),
            }
        )

    basis_valid = np.allclose(basis_gram, np.eye(2), rtol=0.0, atol=1e-14) and np.allclose(
        basis_sum, 0.0, rtol=0.0, atol=1e-14
    )
    passed = basis_valid and all(point["full_effective_rank"] for point in points)
    return {
        "passed": bool(passed),
        "raw_output_dimension": 3,
        "effective_parameter_dimension": 2,
        "constraint": "x_CO2 + x_O2 + x_N2 = 100 percent",
        "constraint_enforcement": "during_optimization_in_orthonormal_sum_zero_tangent_space",
        "n2_backfill": False,
        "posthoc_projection": False,
        "tangent_basis": RAW3_TANGENT_BASIS.tolist(),
        "tangent_basis_orthonormal_and_sum_zero": bool(basis_valid),
        "finite_difference_step_percent": step,
        "rank_relative_tolerances": list(tolerances),
        "points": points,
        "excluded_direction": {
            "name": "unconstrained_total_scale",
            "status": "outside_registered_physical_domain",
            "information_claim": "not_observation_identified_and_not_an_estimand",
        },
    }


def _load_parent_mei1(parent_dir: Path, project_root: Path) -> dict[str, Any]:
    manifest_path = parent_dir / "evidence_manifest.json"
    verdict_path = parent_dir / "mei1_verdict.json"
    if not verdict_path.is_file():
        raise FileNotFoundError(f"parent MEI-1 verdict missing: {verdict_path}")
    verdict = load_json(verdict_path)
    manifest_sha = sha256_file(manifest_path)
    issues = verify_evidence_manifest(manifest_path, project_root=project_root)
    return {
        "verdict": verdict,
        "manifest_sha256": manifest_sha,
        "manifest_issues": issues,
        "metric": load_json(parent_dir / "metric_registry.json"),
    }


def run_mei3_phase_a_audit(
    *,
    project_root: Path,
    config: dict[str, Any],
    parent_mei1_freeze_dir: Path,
    current_stage_status: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    parent = _load_parent_mei1(parent_mei1_freeze_dir, project_root)
    issues.extend(parent["manifest_issues"])
    parent_audit = parent["verdict"].get("audit") or {}
    prerequisite = config["mei1_prerequisite"]

    for key, expected in (
        ("verdict", prerequisite["expected_verdict"]),
        ("allowed_next_stage", prerequisite["expected_allowed_next_stage"]),
        ("frozen_design", prerequisite["expected_frozen_design"]),
    ):
        if parent_audit.get(key) != expected:
            issues.append(f"parent MEI-1 {key} must be {expected!r}")
    if parent_audit.get("passed") is not True or parent_audit.get("blockers") != []:
        issues.append("parent MEI-1 must be passed with no blockers")
    if config.get("registry_schema_version") != REGISTRY_SCHEMA_VERSION:
        issues.append(f"MEI-3 config schema must be {REGISTRY_SCHEMA_VERSION}")
    current_mei1 = current_stage_status.get("mei1") or {}
    if parent_mei1_freeze_dir.name not in str(current_mei1.get("freeze_dir", "")):
        issues.append("current stage_status must point at the same parent MEI-1 freeze")
    if current_stage_status.get("allowed_next_stage") != "MEI-3_varpro_audit":
        issues.append("current allowed_next_stage must be MEI-3_varpro_audit")

    authorizations = parent_audit.get("authorizations") or {}
    for field in AUTHORIZATION_FIELDS:
        if authorizations.get(field) != FORBIDDEN_AUTH_VALUE:
            issues.append(f"parent authorization changed unexpectedly: {field}")

    metric = parent["metric"]
    contract = metric.get("varpro_observation_contract") or {}
    required_fields = set(contract.get("required_fields") or [])
    expected_fields = {
        "raw_tof_s",
        "log_amplitude_or_amplitude",
        "phase_rad_or_complex_transfer_real_imag",
        "observation_covariance",
        "frequency_hz",
        "device_profile_id",
        "view_id",
        "T_C",
        "P_MPa",
        "H_RH",
        "L_m",
    }
    if required_fields != expected_fields:
        issues.append("parent VarPro observation contract fields differ from MEI-3 v1")
    if contract.get("forbid_unconstrained_free_offsets_per_k4_sample") is not True:
        issues.append("parent contract must forbid unconstrained per-sample K4 offsets")

    candidates = assess_linear_candidates(config)
    numerical = run_numerical_equivalence_audit(config) if not issues else None
    registered_candidates = (
        "common_delay",
        "log_amplitude_gain",
        "per_frequency_calibration_offset",
    )
    admitted = [name for name in registered_candidates if candidates[name]["supported"]]
    structure_supported = all(candidates[name]["supported"] for name in registered_candidates)
    passed = not issues and structure_supported and bool(numerical and numerical["passed"])
    verdict = PHASE_A_SUPPORTED if passed else VARPRO_NOT_APPLICABLE
    transition = config["phase_a_transition"]
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "stage": STAGE,
        "phase": PHASE,
        "verdict": verdict,
        "passed": passed,
        "issues": issues,
        "admitted_linear_blocks": admitted,
        "candidate_assessment": candidates,
        "numerical_equivalence": numerical,
        "allowed_next_stage": (
            transition["allowed_next_stage_after_supported"] if passed else None
        ),
        "formal_solver_gate_ready": False,
        "formal_solver_gate_blocker": transition["formal_solver_gate_blocker"],
        "authorizations": {field: FORBIDDEN_AUTH_VALUE for field in AUTHORIZATION_FIELDS},
        "parent_mei1_freeze_dir": str(parent_mei1_freeze_dir.resolve()),
        "parent_mei1_manifest_sha256": parent["manifest_sha256"],
        "claim_scope": config["claim_scope"],
    }


def run_mei3_b0_audit(
    *,
    project_root: Path,
    config: dict[str, Any],
    parent_mei1_freeze_dir: Path,
    current_stage_status: dict[str, Any],
) -> dict[str, Any]:
    """Close B0 by auditing the observation operator and constrained raw3 rank."""
    phase_a = run_mei3_phase_a_audit(
        project_root=project_root,
        config=config,
        parent_mei1_freeze_dir=parent_mei1_freeze_dir,
        current_stage_status=current_stage_status,
    )
    observation = run_b0_observation_operator_audit(config) if phase_a["passed"] else None
    rank = run_b0_raw3_rank_audit(config) if phase_a["passed"] else None
    issues = list(phase_a["issues"])
    if observation is not None and not observation["passed"]:
        issues.append("B0 observation operator contract failed")
    if rank is not None and not rank["passed"]:
        issues.append("B0 constrained raw3 tangent rank audit failed")
    passed = bool(
        phase_a["passed"]
        and observation
        and observation["passed"]
        and rank
        and rank["passed"]
    )
    transition = config["b0_transition"]
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "stage": STAGE,
        "phase": B0_PHASE,
        "verdict": B0_REPRESENTATION_CLOSED if passed else B0_REPRESENTATION_INVALID,
        "passed": passed,
        "issues": issues,
        "phase_a_prerequisite": {
            "verdict": phase_a["verdict"],
            "passed": phase_a["passed"],
        },
        "admitted_linear_blocks": phase_a["admitted_linear_blocks"],
        "observation_operator_audit": observation,
        "raw3_forward_rank_audit": rank,
        "composition_contract": config["composition_parameterization"],
        "allowed_next_stage": transition["allowed_next_stage_after_closed"] if passed else None,
        "formal_solver_gate_ready": False,
        "formal_solver_gate_blocker": transition["formal_solver_gate_blocker"],
        "authorizations": phase_a["authorizations"],
        "parent_mei1_freeze_dir": phase_a["parent_mei1_freeze_dir"],
        "parent_mei1_manifest_sha256": phase_a["parent_mei1_manifest_sha256"],
        "claim_scope": config["claim_scope"],
    }


def run_mei3_b1_audit(
    *,
    project_root: Path,
    config: dict[str, Any],
    parent_b0_freeze_dir: Path,
    current_stage_status: dict[str, Any],
) -> dict[str, Any]:
    """Freeze the S0 historical disposition and the verified S1 contract."""
    issues: list[str] = []
    parent_manifest_path = parent_b0_freeze_dir / "evidence_manifest.json"
    parent_verdict_path = parent_b0_freeze_dir / "mei3_verdict.json"
    parent_manifest_sha = sha256_file(parent_manifest_path)
    issues.extend(verify_evidence_manifest(parent_manifest_path, project_root=project_root))
    parent = load_json(parent_verdict_path)
    parent_audit = parent.get("audit") or {}
    prerequisite = config["parent_b0"]
    if parent_audit.get("verdict") != prerequisite["expected_verdict"]:
        issues.append("parent B0 verdict mismatch")
    if parent_audit.get("phase") != prerequisite["expected_phase"]:
        issues.append("parent B0 phase mismatch")
    if parent_audit.get("passed") is not True:
        issues.append("parent B0 must be passed")
    if parent_manifest_sha != prerequisite["expected_manifest_sha256"]:
        issues.append("parent B0 manifest SHA256 mismatch")

    current_mei3 = current_stage_status.get("mei3") or {}
    if current_mei3.get("verdict") != prerequisite["expected_verdict"]:
        issues.append("stage_status must still point to the parent B0 verdict")
    if parent_b0_freeze_dir.name not in str(current_mei3.get("freeze_dir", "")):
        issues.append("stage_status must point to the same parent B0 freeze")
    if current_stage_status.get("allowed_next_stage") != "MEI-3_varpro_audit":
        issues.append("current allowed_next_stage must be MEI-3_varpro_audit")

    for field in AUTHORIZATION_FIELDS:
        if parent_audit.get("authorizations", {}).get(field) != FORBIDDEN_AUTH_VALUE:
            issues.append(f"parent B0 authorization changed unexpectedly: {field}")
        if config.get("authorizations", {}).get(field) != FORBIDDEN_AUTH_VALUE:
            issues.append(f"B1 authorization must remain forbidden: {field}")

    disposition = config["historical_h1_disposition"]
    mrs2_path = project_root / disposition["mrs2_verdict_path"]
    mrs6_path = project_root / disposition["mrs6_verdict_path"]
    legacy_plan_path = project_root / disposition["legacy_plan_path"]
    mrs2 = load_json(mrs2_path)
    mrs6 = load_json(mrs6_path)
    legacy_plan = legacy_plan_path.read_text(encoding="utf-8")
    if sha256_file(mrs2_path) != disposition["expected_mrs2_sha256"]:
        issues.append("historical MRS-2 verdict SHA256 changed")
    if (mrs2.get("decision") or {}).get("verdict") != disposition["expected_mrs2_verdict"]:
        issues.append("historical MRS-2 verdict changed")
    if sha256_file(mrs6_path) != disposition["expected_mrs6_sha256"]:
        issues.append("historical MRS-6 verdict SHA256 changed")
    if mrs6.get("verdict") != disposition["expected_mrs6_verdict"]:
        issues.append("historical MRS-6 verdict changed")
    if mrs6.get("mrs2_verdict_unchanged") != disposition["expected_mrs2_verdict"]:
        issues.append("MRS-6 no longer preserves the MRS-2 verdict")
    if "MRS-3 未进入" not in legacy_plan or "H1 解析反演基线" not in legacy_plan:
        issues.append("legacy plan no longer supports the H1 non-instantiation disposition")

    methods = config["method_matrix"]
    comparison = config["comparison_contract"]
    composition = config["composition_parameterization"]
    disposition_closed = bool(
        disposition["status"] == "historical_h1_not_instantiated"
        and disposition["historical_verdicts_mutable"] is False
        and methods["S0"]["status"] == "historical_h1_not_instantiated"
        and methods["S0"]["execution_policy"] == "non_running_historical_note"
        and methods["S0"]["formal_pairing_eligible"] is False
        and comparison["running_methods"] == ["S1", "S2", "S3"]
        and comparison["primary_comparison"] == ["S1", "S2"]
        and comparison["upper_bound_only"] == ["S3"]
        and comparison["freeze_s1_before_observing_s2_results"] is True
        and composition["scope"] == "deterministic_mei3_s1_s2_s3_only"
        and composition["posthoc_projection"] is False
        and composition["n2_closure_backfill"] is False
        and composition["silent_normalization"] is False
    )
    if not disposition_closed:
        issues.append("S0 historical disposition contract is not closed")

    s1_audit = run_b1_s1_numerical_audit(config) if not issues else None
    if s1_audit is not None and not s1_audit["s1_verified"]:
        issues.append("S1 numerical audit failed")
    passed = bool(not issues and s1_audit and s1_audit["b1_closed"])
    transition = config["b1_transition"]
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "stage": STAGE,
        "phase": B1_PHASE,
        "verdict": B1_S1_FROZEN if passed else B1_S1_INVALID,
        "passed": passed,
        "issues": issues,
        "s0_historical_disposition": {
            "status": disposition["status"],
            "execution_policy": methods["S0"]["execution_policy"],
            "formal_pairing_eligible": methods["S0"]["formal_pairing_eligible"],
            "future_reimplementation_policy": methods["S0"][
                "future_reimplementation_policy"
            ],
            "legacy_plan_path": disposition["legacy_plan_path"],
            "mrs2_verdict": disposition["expected_mrs2_verdict"],
            "mrs6_verdict": disposition["expected_mrs6_verdict"],
            "historical_verdicts_unchanged": disposition_closed,
        },
        "s1_numerical_audit": s1_audit,
        "running_methods": comparison["running_methods"],
        "primary_comparison": comparison["primary_comparison"],
        "upper_bound_only": comparison["upper_bound_only"],
        "allowed_next_stage": transition["allowed_next_stage_after_closed"]
        if passed
        else None,
        "formal_solver_gate_ready": False,
        "formal_solver_gate_blocker": transition["formal_solver_gate_blocker"],
        "authorizations": {field: FORBIDDEN_AUTH_VALUE for field in AUTHORIZATION_FIELDS},
        "parent_b0_freeze_dir": str(parent_b0_freeze_dir.resolve()),
        "parent_b0_manifest_sha256": parent_manifest_sha,
        "claim_scope": config["claim_scope"],
    }


def run_mei3_b2_audit(
    *,
    project_root: Path,
    config: dict[str, Any],
    parent_b1_freeze_dir: Path,
    current_stage_status: dict[str, Any],
) -> dict[str, Any]:
    """Verify the B1 parent and execute only the nonformal B2 mechanism audit."""
    issues: list[str] = []
    manifest_path = parent_b1_freeze_dir / "evidence_manifest.json"
    verdict_path = parent_b1_freeze_dir / "mei3_b1_verdict.json"
    manifest_sha = sha256_file(manifest_path)
    issues.extend(verify_evidence_manifest(manifest_path, project_root=project_root))
    parent_audit = (load_json(verdict_path).get("audit") or {})
    prerequisite = config["parent_b1"]
    if parent_audit.get("verdict") != prerequisite["expected_verdict"]:
        issues.append("parent B1 verdict mismatch")
    if parent_audit.get("phase") != prerequisite["expected_phase"]:
        issues.append("parent B1 phase mismatch")
    if parent_audit.get("passed") is not True:
        issues.append("parent B1 must be passed")
    if manifest_sha != prerequisite["expected_manifest_sha256"]:
        issues.append("parent B1 manifest SHA256 mismatch")

    current_mei3 = current_stage_status.get("mei3") or {}
    if current_mei3.get("verdict") != prerequisite["expected_verdict"]:
        issues.append("stage_status must still point to the parent B1 verdict")
    if parent_b1_freeze_dir.name not in str(current_mei3.get("freeze_dir", "")):
        issues.append("stage_status must point to the same parent B1 freeze")
    if current_stage_status.get("allowed_next_stage") != "MEI-3_varpro_audit":
        issues.append("current allowed_next_stage must be MEI-3_varpro_audit")
    for field in AUTHORIZATION_FIELDS:
        if parent_audit.get("authorizations", {}).get(field) != FORBIDDEN_AUTH_VALUE:
            issues.append(f"parent B1 authorization changed unexpectedly: {field}")
        if config.get("authorizations", {}).get(field) != FORBIDDEN_AUTH_VALUE:
            issues.append(f"B2 authorization must remain forbidden: {field}")

    numerical = run_b2_solver_core_audit(config) if not issues else None
    if numerical is not None and not numerical["solver_core_verified"]:
        issues.append("B2 solver core numerical audit failed")
    passed = bool(not issues and numerical and numerical["solver_core_verified"])
    transition = config["b2_transition"]
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "stage": STAGE,
        "phase": B2_PHASE,
        "verdict": B2_SOLVER_CORE_VERIFIED if passed else B2_SOLVER_CORE_INVALID,
        "passed": passed,
        "issues": issues,
        "solver_core_audit": numerical,
        "running_methods": config["comparison_contract"]["running_methods"],
        "primary_comparison": config["comparison_contract"]["primary_comparison"],
        "upper_bound_only": config["comparison_contract"]["upper_bound_only"],
        "allowed_next_stage": transition["allowed_next_stage_after_verified"]
        if passed
        else None,
        "formal_solver_gate_ready": transition["formal_solver_gate_ready"],
        "formal_solver_gate_blocker": transition["formal_solver_gate_blocker"],
        "authorizations": {field: FORBIDDEN_AUTH_VALUE for field in AUTHORIZATION_FIELDS},
        "parent_b1_freeze_dir": str(parent_b1_freeze_dir.resolve()),
        "parent_b1_manifest_sha256": manifest_sha,
        "claim_scope": config["claim_scope"],
    }


__all__ = [
    "B0_REPRESENTATION_CLOSED",
    "B0_REPRESENTATION_INVALID",
    "B1_S1_FROZEN",
    "B1_S1_INVALID",
    "B2_SOLVER_CORE_INVALID",
    "B2_SOLVER_CORE_VERIFIED",
    "ConditionalLinearSolution",
    "PHASE_A_SUPPORTED",
    "VARPRO_NOT_APPLICABLE",
    "assess_linear_candidates",
    "run_b0_observation_operator_audit",
    "run_b0_raw3_rank_audit",
    "run_mei3_b0_audit",
    "run_mei3_b1_audit",
    "run_mei3_b2_audit",
    "run_mei3_phase_a_audit",
    "run_numerical_equivalence_audit",
    "solve_conditionally_linear",
]
