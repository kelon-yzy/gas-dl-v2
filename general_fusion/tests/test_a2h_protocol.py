from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from gf.pipeline.a2h_benchmark import (
    A2HProtocolError,
    A2HTestUnlockError,
    _claim_hard_test_access,
    assert_hard_test_unlocked,
    run_a2h_protocol,
    validate_a2h_data_config,
    validate_a2h_eval_config,
    validate_a2h_experiment_config,
    validate_a2h_model_config,
    validate_a2h_train_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_a2h_configs_are_frozen_and_bound() -> None:
    validate_a2h_data_config(_load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2h_v2.json"))
    validate_a2h_eval_config(_load_json(CONFIG_ROOT / "eval" / "a2h_eval.json"))
    validate_a2h_train_config(_load_json(CONFIG_ROOT / "train" / "a2h_train.json"))
    validate_a2h_model_config(_load_json(CONFIG_ROOT / "model" / "a2h_candidate.json"))
    for filename in (
        "a2h_protocol.json",
        "a2h_difficulty_audit.json",
        "a2h_learning_noise.json",
        "a2h_ood.json",
        "a2h_algorithm.json",
        "a2h_formal.json",
    ):
        validate_a2h_experiment_config(_load_json(CONFIG_ROOT / "experiment" / filename))


def test_a2h_protocol_keeps_hard_test_locked() -> None:
    result = run_a2h_protocol(project_root=PROJECT_ROOT)
    assert result["stage"] == "A2H-0"
    assert result["status"] == "PASS"
    assert result["manifest"]["test_access"] == {
        "default": "locked",
        "unlocked": False,
        "evidence": None,
    }


def test_a2h_unlock_requires_all_registered_evidence() -> None:
    expected = "a" * 64
    expected_evidence = {
        "data_content_sha256": expected,
        "split_family_hash": expected,
        "eligible_axes_sha256": expected,
        "candidate_config_sha256": expected,
        "matched_baseline_config_sha256": expected,
        "selected_checkpoint_sha256": expected,
        "primary_chart_template_sha256": expected,
        "formal_run_status": "FROZEN",
    }
    actual_evidence = dict(expected_evidence)
    actual_evidence["selected_checkpoint_sha256"] = None
    with pytest.raises(A2HTestUnlockError, match="selected_checkpoint_sha256"):
        assert_hard_test_unlocked(
            actual=actual_evidence,
            expected=expected_evidence,
        )

    assert_hard_test_unlocked(
        actual=expected_evidence,
        expected=expected_evidence,
    )


def test_a2h_protocol_rejects_legacy_key() -> None:
    config = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2h_v2.json")
    invalid = deepcopy(config)
    invalid["sequence_id"] = "legacy"
    with pytest.raises(A2HProtocolError, match="forbidden legacy keys"):
        validate_a2h_data_config(invalid)


def test_a2h_hard_test_access_can_only_be_claimed_once(tmp_path: Path) -> None:
    path = tmp_path / "hard_test_access.json"
    evidence = {"selected_model": "B5", "data_content_sha256": "a" * 64}
    first = _claim_hard_test_access(path, evidence=evidence)
    assert first["status"] == "CLAIMED"
    with pytest.raises(A2HTestUnlockError, match="already claimed"):
        _claim_hard_test_access(path, evidence=evidence)
