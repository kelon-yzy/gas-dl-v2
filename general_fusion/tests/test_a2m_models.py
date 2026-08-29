from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gf.dl.contracts import collate_samples
from gf.dl.mainstream_architectures import (
    A2M_MODEL_IDS,
    FeatureTokenTransformer,
    build_a2m_model,
    validate_a2m_model_config,
)
from gf.dl.training import trainable_parameter_count
from gf.sim.a1_dataset import generate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config(filename: str) -> dict[str, object]:
    value = json.loads((PROJECT_ROOT / "configs" / "model" / filename).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_a2m_registry_models_support_forward_backward_and_parameter_count(tmp_path: Path) -> None:
    dataset = generate_dataset(
        tmp_path / "dataset",
        binary_per_pair=2,
        ternary_count=6,
        generation_seed=20260829,
        split_seed=20260829,
        data_version="a2m-model-smoke-r1",
    )
    batch = collate_samples(tuple(dataset.samples()[:3]))
    filenames = {
        "A2M-MLP": "a2m_mlp.json",
        "A2M-RESNET": "a2m_resnet.json",
        "A2M-FTT": "a2m_ftt.json",
    }
    for model_id in A2M_MODEL_IDS:
        config = _load_config(filenames[model_id])
        validate_a2m_model_config(config)
        for recipe in config["recipes"]:
            model = build_a2m_model(
                model_id,
                recipe,
                sensor_ids=config["sensor_ids"],
                sensor_types=config["sensor_types"],
            )
            prediction = model(batch)
            assert prediction.shape == (3, 3)
            assert torch.isfinite(prediction).all()
            prediction.square().mean().backward()
            assert trainable_parameter_count(model) > 0
            assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_a2m_ftt_uses_feature_tokens_without_time_position_encoding() -> None:
    config = _load_config("a2m_ftt.json")
    assert config["uses_time_position_encoding"] is False
    model = build_a2m_model(
        "A2M-FTT",
        config["recipes"][0],
        sensor_ids=config["sensor_ids"],
        sensor_types=config["sensor_types"],
    )
    assert isinstance(model, FeatureTokenTransformer)
    assert hasattr(model, "class_token")
    assert not hasattr(model, "positional_encoding")


def test_a2m_models_reject_pseudo_temporal_input(tmp_path: Path) -> None:
    dataset = generate_dataset(
        tmp_path / "dataset",
        binary_per_pair=2,
        ternary_count=6,
        generation_seed=20260829,
        split_seed=20260829,
        data_version="a2m-shape-smoke-r1",
    )
    sample = dataset.samples()[0]
    temporal_signals = tuple(
        torch.tensor([[float(signal[0, 0])], [float(signal[0, 0])]], dtype=torch.float32).numpy()
        for signal in sample.signals
    )
    from gf.dl.contracts import UnifiedSample

    temporal = UnifiedSample(
        signals=temporal_signals,
        sensor_id=sample.sensor_id,
        sensor_type=sample.sensor_type,
        valid_mask=tuple(torch.ones_like(torch.from_numpy(signal), dtype=torch.bool).numpy() for signal in temporal_signals),
        quality=tuple(torch.ones(2, dtype=torch.float32).numpy() for _ in temporal_signals),
        time=tuple(torch.tensor([0.0, 1.0], dtype=torch.float64).numpy() for _ in temporal_signals),
        target=sample.target,
        target_mask=sample.target_mask,
        group_id=sample.group_id,
        dataset_id=sample.dataset_id,
        metadata=sample.metadata,
    )
    batch = collate_samples((temporal,))
    model = build_a2m_model("A2M-MLP", {"hidden_dim": 32})
    with pytest.raises(ValueError, match="T=1,F=1"):
        model(batch)
