"""MEI-3 B3 registered-data authorization-readiness audit."""
from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from tv3.audit.mrs_ei_registry import (
    AUTHORIZATION_FIELDS,
    FORBIDDEN_AUTH_VALUE,
    REGISTRY_SCHEMA_VERSION,
    load_json,
    sha256_file,
    verify_evidence_manifest,
)

B3_PHASE = "b3_registered_data_authorization_readiness"
B3_READY = "mei3_registered_data_authorization_ready"
B3_INCOMPLETE = "mei3_registered_data_authorization_incomplete"
B2_VERIFIED = "mei3_solver_core_verified"

_EXPECTED_TABLES = {
    "mixtures",
    "observation_rows",
    "covariance_blocks",
    "calibration_priors",
    "view_nuisance_calibration_priors",
    "s3_truth_nuisance",
}
_TRUTH_FIELDS = {"x_CO2_percent", "x_O2_percent", "x_N2_percent"}
_S3_ONLY_FIELDS = {"common_delay_s", "log_amplitude_gain", "log_amplitude_offsets"}
_VIEW_NUISANCE_PRIOR_FIELDS = {
    "common_delay_prior_mean",
    "common_delay_prior_std",
    "log_amplitude_gain_prior_mean",
    "log_amplitude_gain_prior_std",
}
_OFFSET_PRIOR_FIELDS = {
    "log_amplitude_offset_prior_mean",
    "log_amplitude_offset_prior_std",
}
_OBSERVATIONS = {"raw_tof_s", "log_amplitude", "unwrapped_phase_rad"}
_PROTOCOL_SCHEMA_VERSION = "tunnel-ventilation-mrs-ei-mei3-data-protocol-2"


def recompute_power_plan(config: dict[str, Any]) -> dict[str, Any]:
    plan = config["sample_size_and_power"]
    alpha = 1.0 - float(plan["ci_level"])
    target_power = float(plan["target_power"])
    margin = (
        float(plan["planning_alternative_improvement"])
        - float(plan["null_improvement"])
    )
    influence_std = float(plan["paired_p90_influence_std_upper_bound"])
    if not 0.0 < alpha < 1.0 or not 0.0 < target_power < 1.0:
        raise ValueError("power plan probabilities must be strictly between zero and one")
    if margin <= 0.0 or influence_std <= 0.0:
        raise ValueError("power plan requires positive effect margin and influence scale")
    normal = NormalDist()
    z_ci = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(target_power)
    raw_required = math.ceil(((z_ci + z_power) * influence_std / margin) ** 2)
    conditions = int(plan["registered_conditions_per_domain"])
    rounded_required = math.ceil(raw_required / conditions) * conditions
    achieved_power = normal.cdf(
        math.sqrt(rounded_required) * margin / influence_std - z_ci
    )
    return {
        "z_ci": z_ci,
        "z_power": z_power,
        "effect_margin": margin,
        "raw_required_mixture_ids_per_domain": raw_required,
        "rounded_mixture_ids_per_domain": rounded_required,
        "replicates_per_condition": rounded_required // conditions,
        "achieved_power_at_frozen_sample_size": achieved_power,
    }


def _audit_schema(config: dict[str, Any], issues: list[str]) -> None:
    tables = config.get("tables") or {}
    if set(tables) != _EXPECTED_TABLES:
        issues.append("B3 table inventory differs from the registered protocol")
        return
    mixtures = tables["mixtures"]
    mixture_fields = set(mixtures.get("fields") or {})
    if "mixture_id" not in mixture_fields or not _TRUTH_FIELDS <= mixture_fields:
        issues.append("mixtures table must own mixture_id and raw3 evaluation truth")
    invariants = mixtures.get("invariants") or {}
    if invariants.get("same_mixture_all_views_same_split") is not True:
        issues.append("all views of one mixture_id must remain in one split")
    if invariants.get("sequence_id_present") is not False:
        issues.append("mixtures table must not contain sequence_id")

    observations = tables["observation_rows"]
    observation_fields = observations.get("fields") or {}
    if not _OBSERVATIONS <= set(observation_fields):
        issues.append("observation rows must contain all three frozen observables")
    if observations.get("rows_per_view") != 4:
        issues.append("observation rows must contain fixed D0 K4")
    if observations.get("sequence_id_present") is not False:
        issues.append("observation rows must not contain sequence_id")
    expected_units = {"raw_tof_s": "s", "log_amplitude": "dimensionless", "unwrapped_phase_rad": "rad"}
    for field, unit in expected_units.items():
        if (observation_fields.get(field) or {}).get("unit") != unit:
            issues.append(f"{field} unit must be {unit}")

    covariance = tables["covariance_blocks"]
    covariance_field = (covariance.get("fields") or {}).get("observation_covariance") or {}
    if covariance_field.get("dtype") != "float64" or covariance_field.get("shape") != [12, 12]:
        issues.append("observation covariance must be float64 with shape [12, 12]")
    if covariance.get("matrix_row_order") != (
        "frequency_major_then_raw_tof_s_log_amplitude_unwrapped_phase_rad"
    ):
        issues.append("observation covariance row order is not frozen")
    if set(covariance.get("required_properties") or []) != {"symmetric", "positive_definite"}:
        issues.append("observation covariance must be symmetric positive definite")

    calibration = tables["calibration_priors"]
    if calibration.get("sharing") != ["device_profile_id", "frequency_hz"]:
        issues.append("calibration offsets must be shared by device profile and frequency")
    if calibration.get("shared_across_samples") is not True:
        issues.append("calibration offsets must be shared across samples")
    if calibration.get("independent_from_evaluation_mixtures") is not True:
        issues.append("calibration priors must be independent of evaluation mixtures")

    view_nuisance = tables["view_nuisance_calibration_priors"]
    view_fields = set((view_nuisance.get("fields") or {}))
    if view_nuisance.get("sharing") != ["device_profile_id", "view_id"]:
        issues.append("view nuisance priors must be shared by device profile and view_id")
    if view_nuisance.get("shared_across_samples") is not True:
        issues.append("view nuisance priors must be shared across samples")
    if view_nuisance.get("independent_from_evaluation_mixtures") is not True:
        issues.append("view nuisance priors must be independent of evaluation mixtures")
    if view_nuisance.get("evaluation_policy") != (
        "calibrate_on_calibration_split_join_posterior_as_prior_on_evaluation"
    ):
        issues.append("view nuisance evaluation policy must be calibrate-then-join-prior")
    if view_nuisance.get("forbids_unconstrained_per_sample_estimation") is not True:
        issues.append("view nuisance priors must forbid unconstrained per-sample estimation")
    if not _VIEW_NUISANCE_PRIOR_FIELDS <= view_fields:
        issues.append("view nuisance calibration table must expose delay and gain prior mean/std")

    if tables["s3_truth_nuisance"].get("physical_isolation") != (
        "separate_file_not_joinable_by_s1_s2_runtime"
    ):
        issues.append("S3 truth nuisance table is not physically isolated from S1/S2")

    view_protocol = config.get("view_protocol") or {}
    if view_protocol.get("views_per_mixture") != 1:
        issues.append("registered sparse protocol currently freezes views_per_mixture=1")
    if view_protocol.get("registered_view_ids") != ["view_0"]:
        issues.append("registered view_id inventory must be exactly [view_0]")
    if view_protocol.get("view_id_reuse_across_mixtures") is not True:
        issues.append("view_id must reuse across mixtures so device x view sharing has support")
    if view_protocol.get("replicates_are_distinct_mixture_ids_not_views") is not True:
        issues.append("replicates must be distinct mixture_id rows, not extra views")


def _audit_splits_and_statistics(config: dict[str, Any], issues: list[str]) -> dict[str, Any] | None:
    split = config.get("split_protocol") or {}
    if split.get("random_unit") != "mixture_id":
        issues.append("split random unit must be mixture_id")
    if split.get("forbid_sequence_id_grouping") is not True:
        issues.append("split protocol must explicitly forbid sequence_id grouping")
    if split.get("split_labels") != ["calibration", "test", "ood"]:
        issues.append("split labels must be frozen as [calibration, test, ood]")
    if split.get("calibration_policy") != (
        "independent_calibration_set_for_frequency_offsets_and_view_nuisance_priors"
    ):
        issues.append("calibration policy must cover frequency offsets and view nuisance priors")
    seeds = split.get("split_seeds") or []
    if len(seeds) != 3 or len(set(seeds)) != 3 or not all(isinstance(seed, int) for seed in seeds):
        issues.append("exactly three distinct integer split seeds must be frozen")
    if split.get("test_point_set") != "ambient_core_216":
        issues.append("test point set must be ambient_core_216")
    if split.get("ood_point_set") != "pressure_extension_low_rh_216":
        issues.append("OOD point set must be pressure_extension_low_rh_216")
    plan = config.get("sample_size_and_power") or {}
    if plan.get("evaluation_domains") != ["test", "ood"]:
        issues.append("evaluation domains must be frozen as [test, ood]")

    statistics = config.get("initialization_and_statistics") or {}
    if statistics.get("frozen_initialization_indices") != [0, 1, 2]:
        issues.append("B1 initialization indices must remain frozen as [0, 1, 2]")
    bootstrap = statistics.get("bootstrap") or {}
    if bootstrap.get("n_resamples") != 2000 or bootstrap.get("ci_level") != 0.95:
        issues.append("paired bootstrap must freeze 2000 resamples and 95% CI")
    if bootstrap.get("paired_on") != "mixture_id" or bootstrap.get("strata") != ["design_condition_id"]:
        issues.append("bootstrap must pair on mixture_id within design-condition strata")

    try:
        power = recompute_power_plan(config)
    except (KeyError, TypeError, ValueError) as exc:
        issues.append(f"power plan is not recomputable: {exc}")
        return None
    plan = config["sample_size_and_power"]
    expected = {
        "raw_required_mixture_ids_per_domain": power["raw_required_mixture_ids_per_domain"],
        "frozen_mixture_ids_per_domain": power["rounded_mixture_ids_per_domain"],
        "replicates_per_condition": power["replicates_per_condition"],
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            issues.append(f"sample_size_and_power.{key} must recompute to {value}")
    if power["achieved_power_at_frozen_sample_size"] < float(plan["target_power"]):
        issues.append("frozen sample size does not achieve target power")
    if plan.get("posthoc_sample_size_reduction_forbidden") is not True:
        issues.append("posthoc sample-size reduction must be forbidden")
    return power


def _audit_field_isolation(config: dict[str, Any], issues: list[str]) -> None:
    whitelists = config.get("runtime_field_whitelists") or {}
    if set(whitelists) != {"S1", "S2", "S3", "evaluation_only"}:
        issues.append("runtime field whitelist inventory is incomplete")
        return
    s1 = set(whitelists["S1"])
    s2 = set(whitelists["S2"])
    s3 = set(whitelists["S3"])
    evaluation = set(whitelists["evaluation_only"])
    if s1 != s2:
        issues.append("S1 and S2 runtime field whitelists must be identical")
    if (s1 | s2) & (_TRUTH_FIELDS | _S3_ONLY_FIELDS | evaluation):
        issues.append("S1/S2 runtime whitelist exposes truth-only fields")
    if not (_VIEW_NUISANCE_PRIOR_FIELDS | _OFFSET_PRIOR_FIELDS) <= s1:
        issues.append("S1/S2 must receive offset and view-nuisance calibration priors")
    if not _S3_ONLY_FIELDS <= s3 or _TRUTH_FIELDS & s3:
        issues.append("S3 must receive only isolated nuisance truth, not composition truth")
    if (_VIEW_NUISANCE_PRIOR_FIELDS | _OFFSET_PRIOR_FIELDS) & s3:
        issues.append("S3 must not receive calibration prior fields in place of nuisance truth")
    if not _TRUTH_FIELDS <= evaluation:
        issues.append("raw3 truth must remain evaluation-only")


def run_mei3_b3_readiness_audit(
    *,
    project_root: Path,
    config: dict[str, Any],
    parent_b2_freeze_dir: Path,
    current_stage_status: dict[str, Any],
) -> dict[str, Any]:
    issues: list[str] = []
    parent_manifest_path = parent_b2_freeze_dir / "evidence_manifest.json"
    parent_verdict_path = parent_b2_freeze_dir / "mei3_b2_verdict.json"
    parent_manifest_sha = sha256_file(parent_manifest_path)
    issues.extend(verify_evidence_manifest(parent_manifest_path, project_root=project_root))
    parent = load_json(parent_verdict_path).get("audit") or {}
    prerequisite = config.get("parent_b2") or {}
    if parent.get("verdict") != prerequisite.get("expected_verdict") or parent.get("passed") is not True:
        issues.append("parent B2 must be mei3_solver_core_verified")
    if parent.get("phase") != prerequisite.get("expected_phase"):
        issues.append("parent B2 phase does not match the B3 prerequisite")
    if parent_manifest_sha != prerequisite.get("expected_manifest_sha256"):
        issues.append("parent B2 manifest SHA256 does not match the frozen prerequisite")
    current = current_stage_status.get("mei3") or {}
    if current.get("verdict") != B2_VERIFIED or parent_b2_freeze_dir.name not in str(current.get("freeze_dir", "")):
        issues.append("stage_status must point to the same verified B2 freeze")
    if config.get("registry_schema_version") != REGISTRY_SCHEMA_VERSION:
        issues.append(f"B3 registry schema must be {REGISTRY_SCHEMA_VERSION}")
    if config.get("protocol_schema_version") != _PROTOCOL_SCHEMA_VERSION:
        issues.append(f"B3 protocol schema must be {_PROTOCOL_SCHEMA_VERSION}")
    if config.get("reserved_benchmark_schema_version") != "tunnel-ventilation-mrs-ei-1":
        issues.append("reserved benchmark schema changed unexpectedly")
    if config.get("phase") != B3_PHASE:
        issues.append(f"B3 phase must be {B3_PHASE}")

    _audit_schema(config, issues)
    power = _audit_splits_and_statistics(config, issues)
    _audit_field_isolation(config, issues)
    expected_hash_roles = {
        "schema", "protocol_config", "split_implementation", "generator", "solver",
        "metric_implementation", "b3_audit", "b3_runner", "b3_tests",
    }
    inventory = config.get("hash_inventory") or {}
    if set(inventory.get("required_source_roles") or []) != expected_hash_roles:
        issues.append("B3 SHA256 source-role inventory is incomplete")
    if inventory.get("hash_algorithm") != "SHA256":
        issues.append("B3 source hash algorithm must be SHA256")
    if "execution_plan" in set(inventory.get("required_source_roles") or []):
        issues.append("execution_plan must not enter the B3 input-contract hash inventory")

    authorizations = config.get("authorizations") or {}
    for field in AUTHORIZATION_FIELDS:
        if authorizations.get(field) != FORBIDDEN_AUTH_VALUE:
            issues.append(f"B3 must not change authorization: {field}")
    if "no_authorization_state_change" not in (config.get("explicit_non_goals") or []):
        issues.append("B3 must explicitly forbid authorization state changes")

    passed = not issues
    transition = config.get("b3_transition") or {}
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "stage": "MEI-3",
        "phase": B3_PHASE,
        "verdict": B3_READY if passed else B3_INCOMPLETE,
        "passed": passed,
        "issues": issues,
        "power_recomputation": power,
        "data_schema_frozen": passed,
        "split_protocol_frozen": passed,
        "runtime_field_isolation_frozen": passed,
        "hash_inventory_frozen": passed,
        "registered_sparse_simulation_generation_review_eligible": bool(
            passed and transition.get("registered_sparse_simulation_generation_review_eligible") is True
        ),
        "allowed_next_stage": None,
        "formal_solver_gate_ready": False,
        "formal_solver_gate_blocker": transition.get("formal_solver_gate_blocker"),
        "authorizations": {field: FORBIDDEN_AUTH_VALUE for field in AUTHORIZATION_FIELDS},
        "parent_b2_freeze_dir": str(parent_b2_freeze_dir.resolve()),
        "parent_b2_manifest_sha256": parent_manifest_sha,
        "claim_scope": config.get("claim_scope"),
        "formal_data_generated": False,
    }


PRE_B4_PHASE = "pre_b4_technical_readiness"
PRE_B4_READY = "mei3_pre_b4_technical_ready"
PRE_B4_FAILED = "mei3_pre_b4_technical_failed"
B4_PHASE = "b4_formal_solver_comparison"
B4_WAITING = "mei3_waiting_registered_data_authorization"
B4_AUTHORIZED_VALUE = "authorized"
B3_READY_VERDICT = B3_READY


def run_mei3_pre_b4_readiness_audit(
    *,
    project_root: Path,
    solver_config: dict[str, Any],
    protocol_config: dict[str, Any],
    parent_b3_freeze_dir: Path,
    current_stage_status: dict[str, Any],
    technical_report: dict[str, Any],
) -> dict[str, Any]:
    """Freeze technical readiness for B4 without authorizing formal data generation."""
    from tv3.ml.mrs_varpro import run_pre_b4_technical_audit

    issues: list[str] = []
    parent_manifest_path = parent_b3_freeze_dir / "evidence_manifest.json"
    parent_verdict_path = parent_b3_freeze_dir / "mei3_b3_verdict.json"
    parent_manifest_sha = sha256_file(parent_manifest_path)
    issues.extend(verify_evidence_manifest(parent_manifest_path, project_root=project_root))
    parent = load_json(parent_verdict_path).get("audit") or {}
    if parent.get("verdict") != B3_READY_VERDICT or parent.get("passed") is not True:
        issues.append("parent B3 must be mei3_registered_data_authorization_ready")
    current = current_stage_status.get("mei3") or {}
    if current.get("verdict") != B3_READY_VERDICT or parent_b3_freeze_dir.name not in str(
        current.get("freeze_dir", "")
    ):
        issues.append("stage_status must point to the same ready B3 freeze")
    if protocol_config.get("protocol_schema_version") != _PROTOCOL_SCHEMA_VERSION:
        issues.append(f"protocol schema must be {_PROTOCOL_SCHEMA_VERSION}")
    if "view_nuisance_calibration_priors" not in (protocol_config.get("tables") or {}):
        issues.append("Option A view-nuisance calibration priors must be registered")
    if technical_report.get("technical_ready") is not True:
        issues.append("pre-B4 technical audit did not pass")
    if technical_report.get("formal_data_generated") is not False:
        issues.append("pre-B4 must not generate formal data")
    authorizations = protocol_config.get("authorizations") or {}
    for field in AUTHORIZATION_FIELDS:
        if authorizations.get(field) != FORBIDDEN_AUTH_VALUE:
            issues.append(f"pre-B4 must not change authorization: {field}")

    # Recompute once more inside the gate for freeze reproducibility.
    recomputed = run_pre_b4_technical_audit(solver_config)
    if recomputed.get("technical_ready") is not True:
        issues.append("recomputed pre-B4 technical audit failed")

    passed = not issues and recomputed.get("technical_ready") is True
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "stage": "MEI-3",
        "phase": PRE_B4_PHASE,
        "verdict": PRE_B4_READY if passed else PRE_B4_FAILED,
        "passed": passed,
        "issues": issues,
        "technical_report": recomputed,
        "b4_technical_ready": passed,
        "allowed_next_stage": None,
        "formal_solver_gate_ready": False,
        "formal_solver_gate_blocker": (
            "registered_sparse_simulation_generation_forbidden_pending_independent_authorization"
            if passed
            else "pre_b4_technical_checks_failed"
        ),
        "authorizations": {field: FORBIDDEN_AUTH_VALUE for field in AUTHORIZATION_FIELDS},
        "parent_b3_freeze_dir": str(parent_b3_freeze_dir.resolve()),
        "parent_b3_manifest_sha256": parent_manifest_sha,
        "claim_scope": "registered_simulation_domain_only",
        "formal_data_generated": False,
    }


def assess_b4_execution_authorization(
    *,
    protocol_config: dict[str, Any],
    current_stage_status: dict[str, Any],
) -> dict[str, Any]:
    """Return whether B4 may generate formal paired comparisons."""
    issues: list[str] = []
    mei3 = current_stage_status.get("mei3") or {}
    if mei3.get("b4_technical_ready") is not True and mei3.get("verdict") != PRE_B4_READY:
        issues.append("B4 technical readiness has not been frozen")
    authorizations = {
        **(protocol_config.get("authorizations") or {}),
        **(mei3.get("authorizations") or {}),
    }
    generation = authorizations.get("registered_sparse_simulation_generation")
    if generation != B4_AUTHORIZED_VALUE:
        issues.append(
            "registered_sparse_simulation_generation remains forbidden_until_explicit_authorization"
        )
    authorized = not issues
    return {
        "phase": B4_PHASE,
        "verdict": (
            "mei3_b4_authorized_to_execute"
            if authorized
            else B4_WAITING
        ),
        "authorized": authorized,
        "issues": issues,
        "formal_data_generated": False,
        "formal_solver_gate_ready": authorized,
        "formal_solver_gate_blocker": None if authorized else issues[0] if issues else None,
    }


__all__ = [
    "B3_INCOMPLETE",
    "B3_PHASE",
    "B3_READY",
    "B4_AUTHORIZED_VALUE",
    "B4_PHASE",
    "B4_WAITING",
    "PRE_B4_FAILED",
    "PRE_B4_PHASE",
    "PRE_B4_READY",
    "assess_b4_execution_authorization",
    "recompute_power_plan",
    "run_mei3_b3_readiness_audit",
    "run_mei3_pre_b4_readiness_audit",
]
