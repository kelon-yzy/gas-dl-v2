from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from gf.pipeline.a2m_benchmark import (
    A2MProtocolError,
    A2MTestUnlockError,
    assert_formal_unlocked,
    run_a2m_protocol,
    validate_a2m_eval_config,
    validate_a2m_experiment_config,
    validate_a2m_train_config,
)
from gf.sim.a2m_dataset import (
    A2MTestLockError,
    generate_a2m_formal_holdout,
    load_a2m_dataset,
    validate_a2m_data_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_a2m_configs_are_frozen_and_formal_holdout_is_locked() -> None:
    data = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2m_v1.json")
    evaluation = _load_json(CONFIG_ROOT / "eval" / "a2m_eval.json")
    training = _load_json(CONFIG_ROOT / "train" / "a2m_train.json")
    validate_a2m_data_config(data)
    validate_a2m_eval_config(evaluation)
    validate_a2m_train_config(training)
    for filename in ("a2m_protocol.json", "a2m_dev.json", "a2m_formal.json"):
        validate_a2m_experiment_config(_load_json(CONFIG_ROOT / "experiment" / filename))

    result = run_a2m_protocol(project_root=PROJECT_ROOT)
    assert result["status"] == "PASS"
    assert result["manifest"]["formal_holdout"]["access"] == "locked"
    with pytest.raises(A2MTestLockError, match="formal holdout is locked"):
        load_a2m_dataset(PROJECT_ROOT / "data" / "a2m_v1")


def test_a2m_formal_holdout_has_new_groups_and_registered_hashes(tmp_path: Path) -> None:
    config = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2m_v1.json")
    dataset = generate_a2m_formal_holdout(
        tmp_path / "a2m_v1",
        config=config,
        project_root=PROJECT_ROOT,
    )
    assert len(dataset.observations) == 360
    assert all(observation.mixture_id.startswith("a2m-formal-") for observation in dataset.observations)
    assert set(observation.split_family for observation in dataset.observations) == {
        "iid",
        "calibration",
        "environment",
        "joint",
        "noise",
        "composition",
    }
    assert dataset.manifest["content_sha256"]
    assert dataset.manifest["split_hash"]
    assert dataset.manifest["profile_hash"]


def test_a2m_protocol_rejects_legacy_key_and_unlock_mismatch() -> None:
    config = _load_json(CONFIG_ROOT / "data" / "ar_he_co2_a2m_v1.json")
    invalid = deepcopy(config)
    invalid["sequence_id"] = "legacy"
    with pytest.raises(ValueError, match="forbidden legacy keys"):
        validate_a2m_data_config(invalid)

    expected = {key: "a" * 64 for key in (
        "data_content_sha256",
        "split_hash",
        "profile_hash",
        "protocol_config_sha256",
        "checkpoint_sha256",
        "primary_chart_template_sha256",
    )}
    expected["runtime_fingerprint"] = {"packages": {"pytorch": "test"}}
    expected["formal_run_status"] = "FROZEN"
    actual = dict(expected)
    actual["checkpoint_sha256"] = None
    with pytest.raises(A2MTestUnlockError, match="checkpoint_sha256"):
        assert_formal_unlocked(actual=actual, expected=expected)
    assert_formal_unlocked(actual=expected, expected=expected)
