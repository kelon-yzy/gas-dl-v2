from __future__ import annotations

from pathlib import Path

from gf.dl.tqif import TQIFModel
from gf.dl.training import (
    TorchTrainingConfig,
    prepare_a2_train_val_samples,
    train_torch_model,
)
from gf.sim.a1_dataset import generate_dataset


def test_tqif_uses_the_frozen_train_val_trainer(tmp_path: Path) -> None:
    dataset = generate_dataset(
        tmp_path / "dataset",
        binary_per_pair=2,
        ternary_count=6,
        generation_seed=20260830,
        split_seed=20260830,
        data_version="a2-tqif-training-smoke-r1",
    )
    train_samples, validation_samples, scaler = prepare_a2_train_val_samples(dataset.samples())
    model = TQIFModel(
        embedding_dim=8,
        token_dim=8,
        pair_hidden_dim=8,
        query_ffn_dim=16,
        attention_heads=2,
        output_dim=3,
        sensor_ids=(
            "ultrasonic_tof",
            "thermal_conductivity_voltage",
            "ndir_co2_voltage",
        ),
        sensor_types=("acoustic_tof", "thermal_conductivity", "ndir"),
        target_slot_ids=("slot-0", "slot-1", "slot-2"),
    )
    result = train_torch_model(
        model,
        train_samples,
        validation_samples,
        config=TorchTrainingConfig(
            max_epochs=3,
            patience=2,
            learning_rate=1.0,
            weight_decay=0.0,
            target_scale=(100.0, 100.0, 100.0),
            optimizer_name="LBFGS",
            optimizer_max_iter=3,
            optimizer_history_size=10,
        ),
        seed=17,
        checkpoint_path=tmp_path / "tqif.pt",
    )
    assert scaler.fitted_group_ids == {sample.group_id for sample in train_samples}
    assert result.validation_predictions.shape == (len(validation_samples), 3)
    assert (tmp_path / "tqif.pt").is_file()
