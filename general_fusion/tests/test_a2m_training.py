from __future__ import annotations

from pathlib import Path

import numpy as np

from gf.dl.mainstream_architectures import build_a2m_model
from gf.dl.training import TorchTrainingConfig, train_torch_model
from gf.sim.a1_dataset import generate_dataset


def test_a2m_adamw_recipe_trains_and_round_trips_checkpoint(tmp_path: Path) -> None:
    dataset = generate_dataset(
        tmp_path / "dataset",
        binary_per_pair=3,
        ternary_count=12,
        generation_seed=20260829,
        split_seed=20260829,
        data_version="a2m-training-smoke-r1",
    )
    samples = dataset.samples()
    train_samples = [sample for sample in samples if sample.metadata["split"] == "train"]
    validation_samples = [sample for sample in samples if sample.metadata["split"] == "val"]
    model = build_a2m_model("A2M-MLP", {"hidden_dim": 32})
    config = TorchTrainingConfig.from_mapping(
        {
            "max_epochs": 3,
            "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001},
            "loss": {"name": "mse", "target_scale": [100.0, 100.0, 100.0]},
            "early_stopping": {"patience": 2},
        }
    )
    checkpoint = tmp_path / "a2m.pt"
    result = train_torch_model(
        model,
        train_samples,
        validation_samples,
        config=config,
        seed=17,
        checkpoint_path=checkpoint,
    )
    assert result.best_epoch > 0
    assert result.validation_predictions.shape == (len(validation_samples), 3)
    assert np.isfinite(result.validation_predictions).all()
    payload = __import__("torch").load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == "gf-a2-checkpoint-1"
    model.load_state_dict(payload["state_dict"])
