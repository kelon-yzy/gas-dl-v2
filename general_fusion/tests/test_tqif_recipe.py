from __future__ import annotations

import json
from pathlib import Path

import pytest
from gf.dl.tqif import (
    TQIF_RECIPE_SPECS,
    build_tqif_model,
    validate_tqif_model_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("filename", "field"),
    [
        ("tqif_token16_pair16.json", "token_dim"),
        ("tqif_token16_pair16.json", "pair_rank"),
        ("tqif_token32_pair32.json", "attention_heads"),
        ("tqif_token32_pair32.json", "query_ffn_layers"),
    ],
)
def test_tqif_recipe_rejects_fixed_spec_tampering(filename: str, field: str) -> None:
    config = _load(filename)
    config[field] = int(config[field]) + 1
    with pytest.raises(ValueError, match=field):
        validate_tqif_model_config(config)


def test_tqif_recipe_specs_are_complete_and_builder_uses_one_head_path() -> None:
    config = _load("tqif_token16_pair16.json")
    spec = TQIF_RECIPE_SPECS[config["recipe"]]
    for key, value in spec.items():
        if key == "head_id":
            assert config["head"]["id"] == value
        elif key in {"query_mode", "pair_evidence", "dropout"}:
            assert config[key] == value
        else:
            assert config[key] == value
    model = build_tqif_model(config)
    assert model.head.__class__.__name__ == "TargetSlotRegressionHead"

    shared_model = model.__class__(
        embedding_dim=16,
        token_dim=16,
        pair_hidden_dim=16,
        query_ffn_dim=32,
        attention_heads=2,
        output_dim=3,
        sensor_ids=config["sensor_ids"],
        sensor_types=config["sensor_types"],
        target_slot_registry=config["target_slot_registry"],
        query_mode="shared",
        use_pair=False,
        capacity_control_hidden_dim=96,
    )
    assert shared_model.head.__class__.__name__ == "SharedRegressionHead"


def _load(filename: str) -> dict[str, object]:
    value = json.loads(
        (PROJECT_ROOT / "configs" / "model" / filename).read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value
