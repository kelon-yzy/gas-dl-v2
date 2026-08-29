from __future__ import annotations

from pathlib import Path

import pytest

from gf.dl.training import (
    A2FusionModel,
    TorchTrainingConfig,
    prepare_a2_train_val_samples,
    train_torch_model,
)
from gf.sim.a1_dataset import generate_dataset


def test_a2_training_uses_only_train_and_val_and_writes_checkpoint(tmp_path: Path) -> None:
    dataset = generate_dataset(
        tmp_path / "dataset",
        binary_per_pair=3,
        ternary_count=30,
        generation_seed=20260827,
        split_seed=20260827,
        data_version="a2-training-smoke-r1",
    )
    train_samples, validation_samples, scaler = prepare_a2_train_val_samples(dataset.samples())
    assert scaler.fitted_group_ids == {sample.group_id for sample in train_samples}
    assert all(sample.metadata["split"] == "train" for sample in train_samples)
    assert all(sample.metadata["split"] == "val" for sample in validation_samples)

    model = A2FusionModel(
        representation="sensor_token",
        embedding_dim=8,
        fusion_hidden_dim=8,
        output_dim=3,
        sensor_ids=("ultrasonic_tof", "thermal_conductivity_voltage", "ndir_co2_voltage"),
        sensor_types=("acoustic_tof", "thermal_conductivity", "ndir"),
    )
    config = TorchTrainingConfig(
        max_epochs=5,
        patience=2,
        learning_rate=0.001,
        weight_decay=0.0,
        target_scale=(100.0, 100.0, 100.0),
    )
    checkpoint = tmp_path / "checkpoint.pt"
    result = train_torch_model(
        model,
        train_samples,
        validation_samples,
        config=config,
        seed=17,
        checkpoint_path=checkpoint,
    )
    assert result.best_epoch > 0
    assert result.epochs_completed <= config.max_epochs
    assert result.validation_predictions.shape == (len(validation_samples), 3)
    assert checkpoint.is_file()


def test_a2_training_rejects_test_samples_in_training_input(tmp_path: Path) -> None:
    dataset = generate_dataset(
        tmp_path / "dataset",
        binary_per_pair=2,
        ternary_count=6,
        generation_seed=20260827,
        split_seed=20260827,
        data_version="a2-training-lock-r1",
    )
    test_samples = [sample for sample in dataset.samples() if sample.metadata["split"] == "test"]
    validation_samples = [sample for sample in dataset.samples() if sample.metadata["split"] == "val"]
    model = A2FusionModel(
        representation="sensor_token",
        embedding_dim=4,
        fusion_hidden_dim=4,
        output_dim=3,
        sensor_ids=("ultrasonic_tof", "thermal_conductivity_voltage", "ndir_co2_voltage"),
        sensor_types=("acoustic_tof", "thermal_conductivity", "ndir"),
    )
    with pytest.raises(ValueError, match="test sample"):
        train_torch_model(
            model,
            test_samples,
            validation_samples,
            config=TorchTrainingConfig(2, 1, 0.001, 0.0, (100.0, 100.0, 100.0)),
            seed=17,
        )
