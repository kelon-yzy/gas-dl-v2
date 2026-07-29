"""MEI-0 registry v2 unit and freeze contract tests."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from tv3.audit.mrs_ei_registry import (
    AUTHORIZATION_FIELDS,
    FORBIDDEN_AUTH_VALUE,
    FREEZE_MANIFEST_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    RESERVED_BENCHMARK_SCHEMA_VERSION,
    SCHEMA_VERSION,
    audit_mei0_registries,
    build_formal_mei1_points,
    build_named_point_set,
    build_narrow_points,
    combined_registry_contract_sha256,
    default_config_dir,
    dumps_stable,
    load_json,
    metric_with_delta_numerical,
    sha256_file,
    verify_evidence_manifest,
)

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "configs" / "tv3_mrs_ei"
_PARENT_MEI0 = (
    _ROOT
    / "outputs"
    / "runs"
    / "tv3_mrs_ei"
    / "mei0_registry"
    / "freezes"
    / "20260727T071921821957Z_f209e893a9e5"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_delta_result() -> dict:
    return {
        "delta_numerical_by_profile": {
            "low_cost_k4_primary": 1.0e-4,
            "registered_mrs2_stress": 2.0e-4,
        },
        "shared_upper_bound": 2.0e-4,
        "by_noise_profile": {
            "low_cost_k4_primary": {
                "noise_profile": "low_cost_k4_primary",
                "delta_numerical": 1.0e-4,
                "max_relative_change_fd": 5.0e-5,
                "max_relative_change_fresh_process_repeat": 0.0,
                "two_times_fd": 1.0e-4,
                "two_times_fresh_process_repeat": 0.0,
                "optimizer_relative_tolerance": 0.0,
                "nominal": {
                    "max_p90_o2_percent": 1.0,
                    "median_p90_o2_percent": 0.9,
                    "min_joint_rank": 3,
                    "max_joint_rank": 4,
                    "median_joint_rank": 3.0,
                    "rank_vector_sha256": "a" * 64,
                    "unstable_fd_count": 0,
                    "process_id": 1,
                },
                "fresh_process_repeats": [
                    {
                        "max_p90_o2_percent": 1.0,
                        "median_p90_o2_percent": 0.9,
                        "min_joint_rank": 3,
                        "max_joint_rank": 4,
                        "median_joint_rank": 3.0,
                        "rank_vector_sha256": "a" * 64,
                        "unstable_fd_count": 0,
                        "process_id": 2,
                    },
                    {
                        "max_p90_o2_percent": 1.0,
                        "median_p90_o2_percent": 0.9,
                        "min_joint_rank": 3,
                        "max_joint_rank": 4,
                        "median_joint_rank": 3.0,
                        "rank_vector_sha256": "a" * 64,
                        "unstable_fd_count": 0,
                        "process_id": 3,
                    },
                ],
                "repeat_execution": "fresh_process",
                "perturbations": [],
                "svd_rank_diagnostics": [],
                "svd_tolerance_policy": "rank_diagnostic_only_not_part_of_crb_p90_delta",
                "n_points": 216,
                "arm": "obs-cfreq",
                "frequencies_hz": [25000.0, 63000.0, 100000.0, 200000.0],
                "point_set": "ambient_core_216",
            },
            "registered_mrs2_stress": {
                "noise_profile": "registered_mrs2_stress",
                "delta_numerical": 2.0e-4,
                "max_relative_change_fd": 1.0e-4,
                "max_relative_change_fresh_process_repeat": 0.0,
                "two_times_fd": 2.0e-4,
                "two_times_fresh_process_repeat": 0.0,
                "optimizer_relative_tolerance": 0.0,
                "nominal": {
                    "max_p90_o2_percent": 7.0,
                    "median_p90_o2_percent": 6.0,
                    "min_joint_rank": 3,
                    "max_joint_rank": 4,
                    "median_joint_rank": 3.0,
                    "rank_vector_sha256": "b" * 64,
                    "unstable_fd_count": 0,
                    "process_id": 1,
                },
                "fresh_process_repeats": [
                    {
                        "max_p90_o2_percent": 7.0,
                        "median_p90_o2_percent": 6.0,
                        "min_joint_rank": 3,
                        "max_joint_rank": 4,
                        "median_joint_rank": 3.0,
                        "rank_vector_sha256": "b" * 64,
                        "unstable_fd_count": 0,
                        "process_id": 4,
                    },
                    {
                        "max_p90_o2_percent": 7.0,
                        "median_p90_o2_percent": 6.0,
                        "min_joint_rank": 3,
                        "max_joint_rank": 4,
                        "median_joint_rank": 3.0,
                        "rank_vector_sha256": "b" * 64,
                        "unstable_fd_count": 0,
                        "process_id": 5,
                    },
                ],
                "repeat_execution": "fresh_process",
                "perturbations": [],
                "svd_rank_diagnostics": [],
                "svd_tolerance_policy": "rank_diagnostic_only_not_part_of_crb_p90_delta",
                "n_points": 216,
                "arm": "obs-cfreq",
                "frequencies_hz": [25000.0, 63000.0, 100000.0, 200000.0],
                "point_set": "ambient_core_216",
            },
        },
        "formula": (
            "max(2*fd_relative_change, 2*fresh_process_relative_change, "
            "optimizer_relative_tolerance)"
        ),
        "optimizer_relative_tolerance": 0.0,
        "svd_tolerance_policy": "rank_diagnostic_only_not_part_of_crb_p90_delta",
        "n_points": 216,
        "point_set": "ambient_core_216",
        "arm": "obs-cfreq",
        "frequencies_hz": [25000.0, 63000.0, 100000.0, 200000.0],
        "noise_profiles": ["low_cost_k4_primary", "registered_mrs2_stress"],
    }


def test_default_config_dir_points_at_tv3_mrs_ei():
    assert default_config_dir().name == "tv3_mrs_ei"


def test_registry_v2_schema_is_distinct_from_benchmark_schema():
    for name in (
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
    ):
        data = load_json(_CFG / name)
        assert data["registry_schema_version"] == REGISTRY_SCHEMA_VERSION
        assert data["reserved_benchmark_schema_version"] == (
            RESERVED_BENCHMARK_SCHEMA_VERSION
        )
        assert data["registry_schema_version"] != data[
            "reserved_benchmark_schema_version"
        ]
    assert SCHEMA_VERSION == REGISTRY_SCHEMA_VERSION
    assert REGISTRY_SCHEMA_VERSION != RESERVED_BENCHMARK_SCHEMA_VERSION


def test_delta_numerical_has_no_practical_floor():
    metric = load_json(_CFG / "metric_registry.json")
    assert "delta_num" not in metric
    formula = metric["decision_thresholds"]["delta_numerical"]["formula"]
    assert "floor" not in formula.lower()
    assert "0.02" not in formula
    assert float(
        metric["decision_thresholds"]["delta_numerical"]["optimizer_relative_tolerance"]
    ) == 0.0


def test_delta_practical_is_separate_and_pre_registered():
    metric = load_json(_CFG / "metric_registry.json")
    practical = metric["decision_thresholds"]["delta_practical"]
    assert float(practical["value"]) == 0.02
    assert practical["source"] == "pre_registered_practical_equivalence_policy"
    assert practical["not_a_numerical_error"] is True


def test_low_cost_noise_profile_requires_all_fields():
    design = load_json(_CFG / "design_space.json")
    profile = copy.deepcopy(design["noise_profiles"]["low_cost_k4_primary"])
    del profile["relative_amp_std"]
    design["noise_profiles"]["low_cost_k4_primary"] = profile
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_numerical=False,
        registry_overrides={"design_space.json": design},
    )
    assert audit["passed"] is False
    assert any("low_cost_noise_profile_missing_traceable_fields" in item for item in audit["issues"])


def test_noise_profiles_do_not_inherit_from_each_other():
    design = load_json(_CFG / "design_space.json")
    low = design["noise_profiles"]["low_cost_k4_primary"]
    stress = design["noise_profiles"]["registered_mrs2_stress"]
    for field in (
        "jitter_std_s",
        "relative_amp_std",
        "covariance_model",
        "prior_std",
        "fixed_delay_s",
        "source",
        "refs",
    ):
        assert field in low
        assert field in stress
    assert low is not stress
    assert float(low["jitter_std_s"]) == 5.0e-7
    assert float(stress["jitter_std_s"]) == 3.0e-6
    assert float(low["prior_std"]["t_c"]) == 0.1
    assert float(stress["prior_std"]["t_c"]) == 1.0


def test_point_sets_are_216_216_and_432_unique():
    design = load_json(_CFG / "design_space.json")
    core = build_named_point_set(design, "ambient_core_216")
    pressure = build_named_point_set(design, "pressure_extension_low_rh_216")
    union = build_formal_mei1_points(design)
    assert len(core) == 216
    assert len(pressure) == 216
    assert len(union) == 432
    assert len({pid for pid, _ in core}) == 216
    assert all("ambient_core_216" in pid for pid, _ in core)
    assert all("pressure_extension_low_rh_216" in pid for pid, _ in pressure)


def test_high_pressure_points_are_in_formal_mei1_gate():
    design = load_json(_CFG / "design_space.json")
    union = build_formal_mei1_points(design)
    high = [pt for _, pt in union if float(pt.p_mpa) in {0.5, 0.709}]
    assert len(high) == 216


def test_drive_budget_is_not_labeled_acoustic_energy():
    design = load_json(_CFG / "design_space.json")
    terms = set(design["cost_function"]["terms"])
    assert "total_drive_budget" in terms
    assert "total_acoustic_energy" not in terms
    assert design["cost_function"]["actual_incident_acoustic_energy_status"] == (
        "unavailable_without_F5_calibration"
    )
    ledger = design["cost_function"]["cost_calculator"]["d0_ledger"]
    assert "total_drive_budget_relative_s" in ledger
    assert "total_acoustic_energy_relative_s" not in ledger


def test_finite_registry_information_gate_forbids_bootstrap():
    metric = load_json(_CFG / "metric_registry.json")
    finite = metric["statistics_protocols"]["finite_registry_information_audit"]
    assert finite["bootstrap"] == "forbidden"
    assert finite["random_unit"] == "none"


def test_raw3_contract_forbids_silent_normalization():
    metric = load_json(_CFG / "metric_registry.json")
    out = metric["component_reporting"]["output_contract"]
    assert out["point_estimate"]["mode"] == "raw3"
    assert out["point_estimate"]["out_dim"] == 3
    assert out["point_estimate"]["silent_normalization"] is False
    assert out["posterior"]["silent_normalization"] is False
    assert "silent_normalization" in out["forbidden"]


def test_stage_transition_selects_s1_when_varpro_not_applicable():
    metric = load_json(_CFG / "metric_registry.json")
    policy = metric["stage_transition_policy"]
    assert policy["mei3_varpro_not_applicable"]["mei4_baseline"] == "S1"
    assert policy["mei3_varpro_supported"]["mei4_baseline"] == "S2"


def test_fixed_k4_transition_skips_mei2_without_authorizing_data():
    metric = load_json(_CFG / "metric_registry.json")
    transition = metric["stage_transition_policy"]["mei1_fixed_k4_retained"]
    assert transition["allowed_next_stage"] == "MEI-3_varpro_audit"
    assert transition["skip_stage"] == "MEI-2_robust_design"
    assert transition["registered_sparse_simulation_generation_review_eligible"] is False
    assert metric["parked_nonblocking_policy"][
        "hardware_and_waveform_authorizations_remain_forbidden"
    ] is True


def test_parked_family_requires_explicit_revisit_trigger():
    model = load_json(_CFG / "model_family_registry.json")
    f2 = next(f for f in model["model_families"] if f["id"] == "F2_h2o_relaxation_params")
    del f2["parking_policy"]["revisit_trigger"]
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        registry_overrides={"model_family_registry.json": model},
    )
    assert audit["passed"] is False
    assert any("parking_policy.revisit_trigger" in issue for issue in audit["issues"])


def test_authorization_fields_are_independent():
    metric = load_json(_CFG / "metric_registry.json")
    auths = metric["authorizations"]
    assert set(AUTHORIZATION_FIELDS) == set(auths)
    for field in AUTHORIZATION_FIELDS:
        assert auths[field] == FORBIDDEN_AUTH_VALUE


def test_baseline_k4_frequencies():
    design = load_json(_CFG / "design_space.json")
    assert design["frequency_band"]["baseline_k4_hz"] == [
        25000.0,
        63000.0,
        100000.0,
        200000.0,
    ]


def test_narrow_points_alias_is_ambient_core():
    design = load_json(_CFG / "design_space.json")
    assert len(build_narrow_points(design)) == 216


def test_audit_rejects_live_delta_num():
    metric = load_json(_CFG / "metric_registry.json")
    metric["delta_num"] = {"floor": 0.02}
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_numerical=False,
        registry_overrides={"metric_registry.json": metric},
    )
    assert audit["passed"] is False
    assert any("must not contain delta_num" in item for item in audit["issues"])


def test_preflight_audit_passes_without_frozen_delta():
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_numerical=False,
    )
    assert audit["issues"] == []
    assert audit["passed"] is True


def test_audit_incomplete_before_delta_numerical_frozen():
    audit = audit_mei0_registries(_CFG, project_root=_ROOT)
    metric = load_json(_CFG / "metric_registry.json")
    if not metric["decision_thresholds"]["delta_numerical"]["by_noise_profile"]:
        assert audit["passed"] is False
        assert audit["verdict"] == "mei0_registry_incomplete"
        assert any("by_noise_profile" in msg for msg in audit["issues"])


def test_audit_rejects_tampered_upstream_verdict_hash():
    model = load_json(_CFG / "model_family_registry.json")
    model["lineage"]["mrs2_verdict"]["expected_sha256"] = "0" * 64
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_numerical=False,
        registry_overrides={"model_family_registry.json": model},
    )
    assert audit["passed"] is False
    assert any("mrs2 verdict sha256 mismatch" in item for item in audit["issues"])


def test_audit_rejects_quantitative_model_bound_without_refs():
    model = load_json(_CFG / "model_family_registry.json")
    f1 = next(item for item in model["model_families"] if item["id"] == "F1_humid_air_c_eq")
    f1["refs"] = []
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_numerical=False,
        registry_overrides={"model_family_registry.json": model},
    )
    assert audit["passed"] is False
    assert any("quantitative bound requires" in item for item in audit["issues"])


def test_traceable_model_family_requires_evidence_hash():
    model = load_json(_CFG / "model_family_registry.json")
    f1 = next(item for item in model["model_families"] if item["id"] == "F1_humid_air_c_eq")
    f1["evidence_sha256"] = None
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_numerical=False,
        registry_overrides={"model_family_registry.json": model},
    )
    assert audit["passed"] is False
    assert any("F1_humid_air_c_eq: traceable evidence requires evidence_sha256" in item for item in audit["issues"])


def test_traceable_f2_is_accepted_when_evidence_is_complete():
    model = load_json(_CFG / "model_family_registry.json")
    f2 = next(item for item in model["model_families"] if item["id"] == "F2_h2o_relaxation_params")
    evidence_path = "tv3/sim/generation/tunnel_ventilation/relaxation_spectrum.py"
    f2.update(
        {
            "source": "implemented_physics",
            "status": "represented_traceable",
            "implementation_or_holdout_path": evidence_path,
            "evidence_path": evidence_path,
            "evidence_sha256": sha256_file(_ROOT / evidence_path),
            "parameter_or_bias_bounds": [0.9, 1.1],
            "bound_semantics": "multiplicative_parameter_envelope",
            "can_clear_not_represented": True,
        }
    )
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_numerical=False,
        registry_overrides={"model_family_registry.json": model},
    )
    assert audit["passed"] is True


def test_solver_ci_lower_bound_must_exceed_delta_practical():
    metric = load_json(_CFG / "metric_registry.json")
    solver = metric["gates"]["solver"]
    solver.pop("require_bootstrap_ci_lb_gt_delta_practical")
    solver["require_bootstrap_ci_lb_positive"] = True
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_numerical=False,
        registry_overrides={"metric_registry.json": metric},
    )
    assert audit["passed"] is False
    assert any("CI lower bound must exceed delta_practical" in item for item in audit["issues"])


def test_audit_rejects_d4_equal_cost_eligibility():
    design = load_json(_CFG / "design_space.json")
    d4 = next(item for item in design["design_arms"] if item["id"] == "D4")
    d4["eligible_for_information_gate"] = True
    audit = audit_mei0_registries(
        _CFG,
        project_root=_ROOT,
        require_frozen_delta_numerical=False,
        registry_overrides={"design_space.json": design},
    )
    assert audit["passed"] is False
    assert any("D4 must be excluded" in item for item in audit["issues"])


def _load_freeze_module():
    import importlib.util

    path = _ROOT / "scripts" / "run_tv3_mei0_registry_freeze.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei0_registry_freeze", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mei0_v2_freeze_contains_parent_and_plan_hash(tmp_path, monkeypatch):
    freeze_mod = _load_freeze_module()
    monkeypatch.setattr(
        freeze_mod,
        "compute_delta_numerical",
        lambda design, metric: _fake_delta_result(),
    )

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in (
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
    ):
        (config_dir / name).write_text(
            (_CFG / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (config_dir / "stage_status.json").write_text(
        dumps_stable({"mei0": None, "mei1": {"stale": True}, "allowed_next_stage": None}),
        encoding="utf-8",
    )

    output_dir = tmp_path / "freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei0_registry_freeze.py",
            "--config-dir",
            str(config_dir),
            "--parent-mei0-freeze-dir",
            str(_PARENT_MEI0),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert freeze_mod.main() == 0

    manifest = load_json(output_dir / "evidence_manifest.json")
    assert manifest["schema_version"] == FREEZE_MANIFEST_SCHEMA_VERSION
    assert manifest["parent_freeze_id"] == _PARENT_MEI0.name
    assert manifest["parent_manifest_sha256"] == _sha256(
        _PARENT_MEI0 / "evidence_manifest.json"
    )
    plan = (
        _ROOT
        / "docs"
        / "active"
        / "tv3_mrs_information_efficient_inversion_experiment_plan.md"
    )
    assert manifest["plan_sha256"] == _sha256(plan)
    assert (output_dir / "experiment_plan_snapshot.md").is_file()
    assert (output_dir / "refreeze_execution_guide_snapshot.md").is_file()
    assert manifest["plan_path"].endswith("/experiment_plan_snapshot.md")
    assert "source_snapshots/" in manifest["source_sha256"]["freeze_script"]["path"]
    assert "input_contract_sha256" in manifest
    assert (output_dir / "domain_point_manifest.json").is_file()
    assert (output_dir / "registry_change_log.json").is_file()
    assert (output_dir / "numerical_stability_recompute.json").is_file()


def test_mei0_v2_freeze_uses_combined_contract_hash(tmp_path, monkeypatch):
    freeze_mod = _load_freeze_module()
    monkeypatch.setattr(
        freeze_mod,
        "compute_delta_numerical",
        lambda design, metric: _fake_delta_result(),
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    payloads = {
        name: load_json(_CFG / name)
        for name in (
            "model_family_registry.json",
            "design_space.json",
            "metric_registry.json",
        )
    }
    for name, payload in payloads.items():
        (config_dir / name).write_text(dumps_stable(payload), encoding="utf-8")
    (config_dir / "stage_status.json").write_text(
        dumps_stable({"mei1": {"keep": False}}),
        encoding="utf-8",
    )
    frozen_metric = metric_with_delta_numerical(
        payloads["metric_registry.json"], _fake_delta_result()
    )
    expected = combined_registry_contract_sha256(
        {
            "model_family_registry.json": payloads["model_family_registry.json"],
            "design_space.json": payloads["design_space.json"],
            "metric_registry.json": frozen_metric,
        }
    )
    output_dir = tmp_path / "freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei0_registry_freeze.py",
            "--config-dir",
            str(config_dir),
            "--parent-mei0-freeze-dir",
            str(_PARENT_MEI0),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert freeze_mod.main() == 0
    manifest = load_json(output_dir / "evidence_manifest.json")
    assert manifest["input_contract_sha256"] == expected


def test_mei0_v2_freeze_clears_stale_mei1_pointer(tmp_path, monkeypatch):
    freeze_mod = _load_freeze_module()
    monkeypatch.setattr(
        freeze_mod,
        "compute_delta_numerical",
        lambda design, metric: _fake_delta_result(),
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in (
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
    ):
        (config_dir / name).write_text(
            (_CFG / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (config_dir / "stage_status.json").write_text(
        dumps_stable(
            {
                "allowed_next_stage": None,
                "mei1": {"verdict": "mei1_inconclusive_forward_model", "stale": True},
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei0_registry_freeze.py",
            "--config-dir",
            str(config_dir),
            "--parent-mei0-freeze-dir",
            str(_PARENT_MEI0),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert freeze_mod.main() == 0
    promoted = load_json(config_dir / "stage_status.json")
    assert promoted["mei1"] is None
    assert promoted["allowed_next_stage"] == "MEI-1_forward_envelope"


def test_mei0_v2_freeze_keeps_all_authorizations_forbidden(tmp_path, monkeypatch):
    freeze_mod = _load_freeze_module()
    monkeypatch.setattr(
        freeze_mod,
        "compute_delta_numerical",
        lambda design, metric: _fake_delta_result(),
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in (
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
    ):
        (config_dir / name).write_text(
            (_CFG / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (config_dir / "stage_status.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei0_registry_freeze.py",
            "--config-dir",
            str(config_dir),
            "--parent-mei0-freeze-dir",
            str(_PARENT_MEI0),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert freeze_mod.main() == 0
    promoted = load_json(config_dir / "stage_status.json")
    for field in AUTHORIZATION_FIELDS:
        assert promoted["mei0"]["authorizations"][field] == FORBIDDEN_AUTH_VALUE
    verdict = load_json(output_dir / "mei0_verdict.json")
    for field in AUTHORIZATION_FIELDS:
        assert verdict["authorizations"][field] == FORBIDDEN_AUTH_VALUE


def test_mei0_v2_freeze_refuses_existing_output_dir(tmp_path, monkeypatch):
    freeze_mod = _load_freeze_module()
    monkeypatch.setattr(
        freeze_mod,
        "compute_delta_numerical",
        lambda design, metric: _fake_delta_result(),
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in (
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
    ):
        (config_dir / name).write_text(
            (_CFG / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (config_dir / "stage_status.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "freeze"
    output_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei0_registry_freeze.py",
            "--config-dir",
            str(config_dir),
            "--parent-mei0-freeze-dir",
            str(_PARENT_MEI0),
            "--output-dir",
            str(output_dir),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        freeze_mod.main()
    assert "refuse overwrite" in str(exc.value)


def test_mei0_v2_manifest_detects_tampered_source(tmp_path, monkeypatch):
    freeze_mod = _load_freeze_module()
    monkeypatch.setattr(
        freeze_mod,
        "compute_delta_numerical",
        lambda design, metric: _fake_delta_result(),
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name in (
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
    ):
        (config_dir / name).write_text(
            (_CFG / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (config_dir / "stage_status.json").write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei0_registry_freeze.py",
            "--config-dir",
            str(config_dir),
            "--parent-mei0-freeze-dir",
            str(_PARENT_MEI0),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert freeze_mod.main() == 0
    manifest_path = output_dir / "evidence_manifest.json"
    verdict = load_json(output_dir / "mei0_verdict.json")
    assert verify_evidence_manifest(
        manifest_path,
        project_root=_ROOT,
        expected_manifest_sha256=verdict["evidence_manifest"]["sha256"],
    ) == []
    metric_path = output_dir / "metric_registry.json"
    metric_path.write_text(metric_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    issues = verify_evidence_manifest(manifest_path, project_root=_ROOT)
    assert any("sha256 mismatch" in item for item in issues)
