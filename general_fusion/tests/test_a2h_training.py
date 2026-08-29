from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gf.dl.training import A2FusionModel
from gf.pipeline.a2h_benchmark import (
    _build_torch_model,
    _fit_model,
    _parameter_parity_report,
)
from gf.sim.a2h_dataset import load_a2h_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "a2h_v2"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_a2h_m1_uses_real_sensor_token_deepsets_and_matched_capacity() -> None:
    train_config = _load_json(PROJECT_ROOT / "configs" / "train" / "a2h_train.json")
    model_config = _load_json(PROJECT_ROOT / "configs" / "model" / "a2h_candidate.json")
    model = _build_torch_model(
        model_id="M1",
        head_id="H0",
        train_config=train_config,
        model_config=model_config,
        include_context=False,
    )

    assert isinstance(model, A2FusionModel)
    assert model.representation == "sensor_token"
    assert model.fusion.pooling == "masked_mean"
    parity = _parameter_parity_report(train_config=train_config, model_config=model_config)
    assert parity["within_tolerance"] is True
    assert parity["C1_parameter_count"] != parity["M1_parameter_count"]


def test_a2h_context_arm_uses_train_only_nonzero_scaler_and_epoch_override() -> None:
    dataset = load_a2h_dataset(DATA_DIR)
    train_indices = dataset.indices(split_family="environment", split="train")
    val_indices = dataset.indices(split_family="environment", split="val")
    stress_indices = dataset.indices(split_family="environment", split="stress_val")
    train_environments = {
        (dataset.observations[int(index)].temperature_k, dataset.observations[int(index)].pressure_pa)
        for index in train_indices
    }
    assert len(train_environments) == 3

    train_config = _load_json(PROJECT_ROOT / "configs" / "train" / "a2h_train.json")
    model_config = _load_json(PROJECT_ROOT / "configs" / "model" / "a2h_candidate.json")
    fit = _fit_model(
        dataset,
        train_indices,
        val_indices,
        stress_indices,
        model_id="B5",
        seed=17,
        head_id="H0",
        train_config=train_config,
        model_config=model_config,
        include_context=True,
        max_epochs_override=2,
    )

    assert np.isfinite(fit.prediction).all()
    assert fit.resources["epochs_completed"] <= 2
    statistics = fit.resources["context_statistics"]
    assert statistics["temperature_k"]["std"] > 0.0
    assert statistics["pressure_pa"]["std"] > 0.0
