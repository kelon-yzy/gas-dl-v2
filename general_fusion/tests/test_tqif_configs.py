from __future__ import annotations

import json
from pathlib import Path

from gf.dl.tqif import (
    build_tqif_matched_concat_model,
    build_tqif_model,
    validate_tqif_model_config,
)
from gf.dl.training import parameter_parity_report
from gf.dl.training import TorchTrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tqif_recipe_configs_validate_and_match_concat_capacity() -> None:
    for filename in ("tqif_token16_pair16.json", "tqif_token32_pair32.json"):
        config = _load_config(filename)
        validate_tqif_model_config(config)
        model = build_tqif_model(config)
        matched_config = _load_config(config["matched_concat"]["config"].split("/")[-1])
        matched = build_tqif_matched_concat_model(matched_config)
        report = parameter_parity_report(model, matched, tolerance=0.10)
        assert report["within_tolerance"] is True
        assert report["left_parameter_count"] > 0
        assert report["right_parameter_count"] > 0


def test_tqif_config_registers_generic_target_slots_and_no_formal_access() -> None:
    config = _load_config("tqif_token16_pair16.json")
    assert [entry["slot_id"] for entry in config["target_slot_registry"]] == [
        "slot_0",
        "slot_1",
        "slot_2",
    ]
    assert config["uses_target_name"] is False
    protocol = json.loads(
        (PROJECT_ROOT / "configs" / "experiment" / "a2_tqif_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["formal_holdout_access"] == "locked"
    assert protocol["test_access"]["old_test_access"] == "forbidden"
    assert protocol["allowed_read_splits"] == ["train", "inner_oof", "val"]

    train_config = _load_train_config()
    training = TorchTrainingConfig.from_mapping(train_config)
    assert training.optimizer_name == "LBFGS"
    assert training.target_scale == (100.0, 100.0, 100.0)
    assert train_config["loss"]["additional_terms"] == []


def _load_config(filename: str) -> dict[str, object]:
    value = json.loads((PROJECT_ROOT / "configs" / "model" / filename).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load_train_config() -> dict[str, object]:
    value = json.loads(
        (PROJECT_ROOT / "configs" / "train" / "a2_tqif_train.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value
