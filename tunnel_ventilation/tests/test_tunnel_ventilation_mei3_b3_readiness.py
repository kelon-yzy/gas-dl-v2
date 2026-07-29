from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tv3.audit.mrs_ei_registry import (
    AUTHORIZATION_FIELDS,
    FORBIDDEN_AUTH_VALUE,
    verify_evidence_manifest,
)
from tv3.audit.mrs_ei_solver_gate import (
    B3_READY,
    recompute_power_plan,
    run_mei3_b3_readiness_audit,
)

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _ROOT / "configs" / "tv3_mrs_ei" / "mei3_solver_data_protocol.json"
_STATUS_PATH = _ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
_PARENT_B2 = (
    _ROOT
    / "outputs"
    / "runs"
    / "tv3_mrs_ei"
    / "mei3_varpro_audit"
    / "freezes"
    / "20260729T081421139186Z_c0ade3f5df14"
)


def _config():
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def _b2_stage_status():
    status = json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
    parent = json.loads((_PARENT_B2 / "mei3_b2_verdict.json").read_text(encoding="utf-8"))
    audit = parent["audit"]
    status["allowed_next_stage"] = "MEI-3_varpro_audit"
    status["mei3"] = {
        "phase": audit["phase"],
        "verdict": audit["verdict"],
        "passed": audit["passed"],
        "freeze_dir": _PARENT_B2.relative_to(_ROOT).as_posix(),
        "s0_status": "historical_h1_not_instantiated",
        "s1_frozen": True,
        "s2_core_verified": True,
        "s3_upper_bound_verified": True,
        "parent_b1_manifest_sha256": audit["parent_b1_manifest_sha256"],
        "parent_b0_manifest_sha256": "6e7823be7b056e0af86e3197c8b7096f3a6a330a570baeb2c2ad899c582ffbb2",
        "parent_mei1_manifest_sha256": "faf397f9457b8eadc8871c55e488da0d62671826bf724ac3fd66f9c03b029396",
    }
    return status


def _load_runner():
    path = _ROOT / "scripts" / "run_tv3_mei3_b3_data_readiness.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei3_b3_data_readiness", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b3_power_plan_recomputes_and_rounds_by_registered_condition():
    config = _config()
    report = recompute_power_plan(config)
    assert report["raw_required_mixture_ids_per_domain"] == 546
    assert report["rounded_mixture_ids_per_domain"] == 648
    assert report["replicates_per_condition"] == 3
    assert report["achieved_power_at_frozen_sample_size"] >= 0.8


def test_b3_contract_freezes_schema_splits_whitelists_and_authorization_boundary():
    audit = run_mei3_b3_readiness_audit(
        project_root=_ROOT,
        config=_config(),
        parent_b2_freeze_dir=_PARENT_B2,
        current_stage_status=_b2_stage_status(),
    )
    assert audit["passed"] is True
    assert audit["verdict"] == B3_READY
    assert audit["data_schema_frozen"] is True
    assert audit["split_protocol_frozen"] is True
    assert audit["runtime_field_isolation_frozen"] is True
    assert audit["registered_sparse_simulation_generation_review_eligible"] is True
    assert audit["formal_data_generated"] is False
    assert audit["formal_solver_gate_ready"] is False
    assert audit["allowed_next_stage"] is None
    for field in AUTHORIZATION_FIELDS:
        assert audit["authorizations"][field] == FORBIDDEN_AUTH_VALUE


def test_b3_schema_uses_mixture_id_and_has_no_sequence_id_fallback():
    config = _config()
    mixtures = config["tables"]["mixtures"]
    observations = config["tables"]["observation_rows"]
    assert mixtures["primary_key"] == ["mixture_id"]
    assert mixtures["invariants"]["sequence_id_present"] is False
    assert observations["sequence_id_present"] is False
    assert config["split_protocol"]["random_unit"] == "mixture_id"
    assert config["split_protocol"]["forbid_sequence_id_grouping"] is True


def test_b3_rejects_truth_exposure_to_s1_or_s2():
    config = _config()
    config["runtime_field_whitelists"]["S1"].append("x_O2_percent")
    audit = run_mei3_b3_readiness_audit(
        project_root=_ROOT,
        config=config,
        parent_b2_freeze_dir=_PARENT_B2,
        current_stage_status=_b2_stage_status(),
    )
    assert audit["passed"] is False
    assert "S1 and S2 runtime field whitelists must be identical" in audit["issues"]
    assert "S1/S2 runtime whitelist exposes truth-only fields" in audit["issues"]


def test_b3_requires_view_nuisance_calibration_priors_and_reuse():
    config = _config()
    assert config["protocol_schema_version"] == "tunnel-ventilation-mrs-ei-mei3-data-protocol-2"
    view_table = config["tables"]["view_nuisance_calibration_priors"]
    assert view_table["sharing"] == ["device_profile_id", "view_id"]
    assert view_table["evaluation_policy"] == (
        "calibrate_on_calibration_split_join_posterior_as_prior_on_evaluation"
    )
    assert config["view_protocol"]["view_id_reuse_across_mixtures"] is True
    assert "common_delay_prior_mean" in config["runtime_field_whitelists"]["S1"]
    assert "log_amplitude_gain_prior_std" in config["runtime_field_whitelists"]["S2"]
    assert "common_delay_s" not in config["runtime_field_whitelists"]["S1"]

    broken = _config()
    broken["tables"]["view_nuisance_calibration_priors"]["sharing"] = [
        "mixture_id",
        "view_id",
    ]
    audit = run_mei3_b3_readiness_audit(
        project_root=_ROOT,
        config=broken,
        parent_b2_freeze_dir=_PARENT_B2,
        current_stage_status=_b2_stage_status(),
    )
    assert audit["passed"] is False
    assert any("view nuisance priors must be shared" in issue for issue in audit["issues"])


def test_b3_rejects_posthoc_sample_size_reduction():
    config = _config()
    config["sample_size_and_power"]["frozen_mixture_ids_per_domain"] = 432
    audit = run_mei3_b3_readiness_audit(
        project_root=_ROOT,
        config=config,
        parent_b2_freeze_dir=_PARENT_B2,
        current_stage_status=_b2_stage_status(),
    )
    assert audit["passed"] is False
    assert any("frozen_mixture_ids_per_domain" in issue for issue in audit["issues"])


def test_b3_rejects_authorization_change_inside_readiness_package():
    config = _config()
    config["authorizations"]["registered_sparse_simulation_generation"] = "authorized"
    audit = run_mei3_b3_readiness_audit(
        project_root=_ROOT,
        config=config,
        parent_b2_freeze_dir=_PARENT_B2,
        current_stage_status=_b2_stage_status(),
    )
    assert audit["passed"] is False
    assert "B3 must not change authorization: registered_sparse_simulation_generation" in audit["issues"]


def test_b3_runner_creates_append_only_freeze_and_promotes_only_mei3(tmp_path, monkeypatch):
    runner = _load_runner()
    output_dir = tmp_path / "freeze"
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_b2_stage_status()), encoding="utf-8")
    before = json.loads(stage_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_b3_data_readiness.py",
            "--parent-b2-freeze-dir",
            str(_PARENT_B2),
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
    assert after["allowed_next_stage"] is None
    assert after["mei3"]["verdict"] == B3_READY
    assert after["mei3"]["b3_authorization_ready_package"] is True
    assert after["mei3"]["registered_sparse_simulation_generation_authorized"] is False
    assert after["mei3"]["formal_solver_gate_ready"] is False
    assert verify_evidence_manifest(output_dir / "evidence_manifest.json", project_root=_ROOT) == []
    assert (output_dir / "sample_size_power_report.json").is_file()
    assert (output_dir / "runtime_field_access_report.json").is_file()
    assert (output_dir / "source_hash_inventory.json").is_file()


def test_b3_runner_refuses_existing_output(tmp_path, monkeypatch):
    runner = _load_runner()
    output_dir = tmp_path / "freeze"
    output_dir.mkdir()
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_b2_stage_status()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_b3_data_readiness.py",
            "--parent-b2-freeze-dir",
            str(_PARENT_B2),
            "--output-dir",
            str(output_dir),
            "--stage-status-path",
            str(stage_path),
        ],
    )
    assert runner.main() == 4
