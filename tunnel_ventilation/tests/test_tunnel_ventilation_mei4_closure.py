from __future__ import annotations

import json
from pathlib import Path

import pytest

from tv3.audit.mei4_closure_checks import (
    STATUS_MATCH,
    STATUS_MISMATCH,
    STATUS_UNVERIFIABLE,
    CheckContext,
    register_check,
    run_checklist,
    summarize,
)

_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT = _ROOT / "configs" / "tv3_mrs_ei" / "mei4_closure_contract.json"
_STATUS = _ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _context() -> CheckContext:
    contract = _load(_CONTRACT)
    return CheckContext(
        project_root=_ROOT,
        c2_freeze_dir=_ROOT / contract["parent_c2"]["freeze_dir"],
    )


def test_duplicate_check_name_is_rejected():
    with pytest.raises(ValueError):
        register_check("json_field")(lambda *_args: None)


def test_unknown_check_name_fails_loudly():
    items = [{"id": "x", "claim": "c", "check": "no_such_check"}]
    with pytest.raises(KeyError):
        run_checklist(_context(), items, default_tolerance=1e-6)


def test_missing_artifact_is_unverifiable_not_silent_pass():
    items = [
        {
            "id": "missing",
            "claim": "reads an artifact that does not exist",
            "check": "json_field",
            "params": {"path": "outputs/definitely_absent_artifact.json", "pointer": ["a"]},
            "expected": 1,
        }
    ]
    results = run_checklist(_context(), items, default_tolerance=1e-6)
    assert results[0].status == STATUS_UNVERIFIABLE
    assert summarize(results)["n_unverifiable"] == 1


def test_tolerance_is_applied_per_item():
    base = {
        "id": "ratio",
        "claim": "Laplace/CRB median",
        "check": "c2_diagnostic",
        "params": {
            "method": "M1",
            "domain": "test",
            "field": "o2_laplace_to_crb_ratio",
            "stat": "median",
        },
        "expected": 1.00035,
    }
    context = _context()
    loose = run_checklist(context, [{**base, "tolerance": 1e-5}], default_tolerance=1e-9)
    tight = run_checklist(context, [{**base, "tolerance": 1e-12}], default_tolerance=1e-9)
    assert loose[0].status == STATUS_MATCH
    assert tight[0].status == STATUS_MISMATCH


def test_wrong_expectation_is_reported_as_mismatch():
    items = [
        {
            "id": "wrong_rejection_count",
            "claim": "deliberately wrong",
            "check": "c2_rejection",
            "params": {"method": "M1", "domain": "test"},
            "expected": {"n": 648, "rejected": 999, "reasons": {}},
        }
    ]
    results = run_checklist(_context(), items, default_tolerance=1e-6)
    assert results[0].status == STATUS_MISMATCH
    assert summarize(results)["mismatched_ids"] == ["wrong_rejection_count"]


def test_registered_closure_checklist_matches_frozen_evidence():
    contract = _load(_CONTRACT)
    results = run_checklist(
        _context(),
        contract["evidence_checklist"],
        default_tolerance=float(contract["default_tolerance"]),
    )
    report = summarize(results)
    assert report["n_mismatch"] == 0, report["mismatched_ids"]
    assert report["n_unverifiable"] == 0, report["unverifiable_ids"]
    assert report["n_checks"] == len(contract["evidence_checklist"])


def test_closure_contract_records_waiver_and_contradictions():
    contract = _load(_CONTRACT)
    waiver = contract["invariant_waivers"][0]
    assert waiver["waived"] is True
    assert set(waiver["missing_items"]) == {"SBC 秩直方图", "后验预测检验"}
    assert {item["id"] for item in contract["c0_contradiction_disposition"]} == {
        "contradiction_1_m2b_unreachable",
        "contradiction_2_t4_circular",
        "contradiction_3_prior_spec_mismatch",
    }


def test_closure_verdict_is_not_a_pass_and_grants_no_authorization():
    contract = _load(_CONTRACT)
    closure = contract["closure"]
    assert closure["verdict"] == "mei4_closed_on_c2_evidence"
    assert closure["verdict_semantics"] == "not_passed"
    assert closure["allowed_next_stage"] is None
    unchanged = closure["authorizations_unchanged"]
    assert unchanged["mei4_cc_sbi_training_draws"] == "forbidden_until_explicit_authorization"
    for field in ("formal_waveform_generation", "benchmark_packaging", "hardware_trial"):
        assert unchanged[field] == "forbidden_until_explicit_authorization"


def test_stage_status_reflects_the_closure():
    status = _load(_STATUS)["mei4"]
    assert status["status"] == "mei4_closed_on_c2_evidence"
    assert status["verdict_semantics"] == "not_passed"
    assert status["allowed_next_stage"] is None
    assert status["closure_path"] == "P-C"
    checks = status["closure_evidence_checks"]
    assert checks["n_mismatch"] == 0
    assert checks["n_unverifiable"] == 0


def test_corrected_claims_carry_their_superseded_record():
    contract = _load(_CONTRACT)
    corrected = {
        item["id"]: item
        for item in contract["evidence_checklist"]
        if "superseded_claim" in item
    }
    assert set(corrected) == {
        "F8_lower_min_test_all_levels",
        "F9_lower_min_ood_all_levels",
        "G7_d0_no_tof_equals_slow_only",
    }
    for item in corrected.values():
        assert item["superseded_claim"]["old_text"]
        assert item["superseded_claim"]["defect"]
