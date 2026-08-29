from __future__ import annotations

from pathlib import Path

import pytest

from gf.sim.a2h_audit import A2HAuditError, run_difficulty_audit
from gf.sim.a2h_dataset import load_a2h_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "a2h_v2"


def test_a2h_difficulty_audit_qualifies_axes_on_development_view() -> None:
    dataset = load_a2h_dataset(DATA_DIR)
    result = run_difficulty_audit(dataset)

    assert result["status"] == "PASS"
    assert result["development_only"] is True
    assert len(result["eligible_axes"]) >= result["minimum_eligible_axes"]
    for axis in result["eligible_axes"]:
        eligibility = result["axes"][axis]["eligibility"]
        assert eligibility["status"] == "PASS"
        assert eligibility["checks"]["signals_within_registered_bounds"] is True


def test_a2h_difficulty_audit_rejects_full_view_with_hard_test() -> None:
    dataset = load_a2h_dataset(DATA_DIR, include_hard_test=True)
    with pytest.raises(A2HAuditError, match="hard_test"):
        run_difficulty_audit(dataset)
