from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from gf.pipeline.a2_benchmark import (
    A1_CONTENT_SHA256,
    A2ProtocolError,
    TestUnlockError,
    assert_test_unlocked,
    build_run_manifest,
    compute_split_hash,
    run_a2_protocol,
    validate_a2_eval_config,
    validate_a2_experiment_config,
    validate_a2_train_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_a2_configs_bind_to_frozen_a1_manifest() -> None:
    eval_config = _load_json(CONFIG_ROOT / "eval" / "a2_eval.json")
    train_config = _load_json(CONFIG_ROOT / "train" / "a2_train.json")
    validate_a2_eval_config(eval_config)
    validate_a2_train_config(train_config)

    manifest = _load_json(PROJECT_ROOT / "data" / "a1_formal" / "manifest.json")
    assert manifest["content_sha256"] == A1_CONTENT_SHA256
    assert compute_split_hash(manifest) == eval_config["data"]["split_hash"]

    for filename in ("a2_head_audit.json", "a2_deepsets.json", "a2_formal.json"):
        experiment = _load_json(CONFIG_ROOT / "experiment" / filename)
        validate_a2_experiment_config(experiment)


def test_a2_protocol_writes_locked_manifest() -> None:
    result = run_a2_protocol(project_root=PROJECT_ROOT)
    assert result["status"] == "PASS"
    manifest = result["manifest"]
    assert manifest["data_content_sha256"] == A1_CONTENT_SHA256
    assert manifest["exit_code"] == 0
    assert manifest["prediction_hash"] is None
    assert manifest["test_access"] == {
        "default": "locked",
        "unlocked": False,
        "evidence": None,
    }


def test_a2_test_unlock_requires_all_frozen_evidence() -> None:
    expected = "a" * 64
    with pytest.raises(TestUnlockError, match="selected_checkpoint_sha256"):
        assert_test_unlocked(
            candidate_config_hash=expected,
            selected_checkpoint_hash=None,
            primary_chart_template_hash=expected,
            formal_run_status="FROZEN",
            expected_candidate_config_hash=expected,
            expected_selected_checkpoint_hash=expected,
            expected_primary_chart_template_hash=expected,
        )

    assert_test_unlocked(
        candidate_config_hash=expected,
        selected_checkpoint_hash=expected,
        primary_chart_template_hash=expected,
        formal_run_status="FROZEN",
        expected_candidate_config_hash=expected,
        expected_selected_checkpoint_hash=expected,
        expected_primary_chart_template_hash=expected,
    )


def test_a2_manifest_records_provenance_and_rejects_forbidden_config_key() -> None:
    manifest = _load_json(PROJECT_ROOT / "data" / "a1_formal" / "manifest.json")
    run_manifest = build_run_manifest(
        project_root=PROJECT_ROOT,
        stage="A2-0",
        config_paths={"eval": CONFIG_ROOT / "eval" / "a2_eval.json"},
        data_manifest_path=PROJECT_ROOT / "data" / "a1_formal" / "manifest.json",
        exit_code=0,
        prediction_hash=None,
        status="PASS",
        test_unlocked=False,
        worktree_revision="test-revision",
        worktree_dirty=True,
    )
    assert run_manifest["worktree"] == {"revision": "test-revision", "dirty": True}
    assert run_manifest["data_content_sha256"] == manifest["content_sha256"]
    assert len(run_manifest["split_hash"]) == 64

    eval_config = _load_json(CONFIG_ROOT / "eval" / "a2_eval.json")
    invalid = deepcopy(eval_config)
    assert isinstance(invalid["data"], dict)
    invalid["data"]["sequence_id"] = "legacy"
    with pytest.raises(A2ProtocolError, match="forbidden legacy keys"):
        validate_a2_eval_config(invalid)
