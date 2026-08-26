import json
from pathlib import Path

from gib.audit.s2_s3 import audit_summary


ROOT = Path(__file__).resolve().parents[1]


def _json_normalize(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def test_s2_has_four_quantified_complementarity_checks():
    summary = audit_summary()
    assert summary["s2"]["passed"]
    assert summary["s2"]["c4_pre_verdict"] == "eligible_for_P3_test"
    assert summary["s2"]["optical_primary"]["passed"]
    assert summary["s2"]["acoustic_or_thermal_primary"]["passed"]
    assert summary["s2"]["single_modality_near_degeneracy"]["passed"]
    assert summary["s2"]["cross_modal_disambiguation"]["passed"]
    assert summary["s2"]["low_concentration_target"]["passed"]


def test_s3_switches_are_independently_off_and_all_off_is_negative_control():
    summary = audit_summary()
    switches = summary["s3"]["switches"]
    assert summary["s3"]["passed"]
    for name, result in switches.items():
        assert result["passed"], name
        assert result["independent_off_pass"], name
        assert result["config_delta_fields"] == [name]
    assert summary["s3"]["all_off_negative_control"]["negative_control_only"]
    assert summary["s3"]["all_off_negative_control"]["passed"]


def test_frozen_s2_s3_evidence_matches_the_pure_forward_audit():
    evidence_path = ROOT / "configs" / "p2_s2_s3_frozen_evidence.json"
    with evidence_path.open(encoding="utf-8") as handle:
        actual = json.load(handle)
    assert _json_normalize(actual) == _json_normalize(audit_summary())
