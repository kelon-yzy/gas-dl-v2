from __future__ import annotations

import json
from pathlib import Path

import torch

from gf.dl.contracts import collate_samples
from gf.dl.training import (
    build_a2_model_from_config,
    parameter_parity_report,
    trainable_parameter_count,
)
from gf.sim.a1_dataset import generate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_concat_and_deepsets_share_the_registered_input_and_head_contract(tmp_path: Path) -> None:
    model_config = json.loads((PROJECT_ROOT / "configs" / "model" / "a2_deepsets.json").read_text(encoding="utf-8"))
    concat_config = json.loads((PROJECT_ROOT / "configs" / "model" / "a2_concat.json").read_text(encoding="utf-8"))
    train_config = json.loads((PROJECT_ROOT / "configs" / "train" / "a2_train.json").read_text(encoding="utf-8"))
    deepsets = build_a2_model_from_config(model_config, train_config, capacity_name="small")
    concat = build_a2_model_from_config(concat_config, train_config, capacity_name="small")
    assert deepsets.head_id == concat.head_id == "H0"
    assert deepsets.encoder.sensor_id_to_index == concat.encoder.sensor_id_to_index
    assert trainable_parameter_count(deepsets) > 0
    report = parameter_parity_report(deepsets, concat, tolerance=0.1)
    assert report["left_parameter_count"] > 0
    assert report["right_parameter_count"] > 0
    assert report["within_tolerance"] is True

    dataset = generate_dataset(
        tmp_path / "dataset",
        binary_per_pair=2,
        ternary_count=6,
        generation_seed=20260827,
        split_seed=20260827,
        data_version="a2-parity-smoke-r1",
    )
    batch = collate_samples(dataset.samples()[:2])
    with torch.no_grad():
        assert deepsets(batch).shape == concat(batch).shape == (2, 3)
