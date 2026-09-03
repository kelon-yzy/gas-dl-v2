from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gf.dl.contracts import UnifiedSample, collate_samples
from gf.dl.tqif import build_tqif_matched_concat_model
from gf.pipeline.a2_tqif_benchmark import build_tqif_variant


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_registered_tqif_variants_have_finite_gradients() -> None:
    sample = _sample()
    batch = collate_samples([sample])
    for recipe, hidden_dim in (("tqif_token16_pair16", 96), ("tqif_token32_pair32", 192)):
        config = _load(f"{recipe}.json")
        for model_id in ("TQIF-H0", "C1", "Q1", "I1", "TQIF-STR"):
            torch.manual_seed(17)
            model = build_tqif_variant(
                config,
                model_id,
                capacity_control_hidden_dim=hidden_dim if model_id in {"C1", "Q1"} else None,
            )
            _assert_finite_gradients(model, batch)

        matched_config = _load(
            f"tqif_matched_concat_{'token16' if 'token16' in recipe else 'token32'}.json"
        )
        torch.manual_seed(17)
        _assert_finite_gradients(build_tqif_matched_concat_model(matched_config), batch)


def _assert_finite_gradients(model: torch.nn.Module, batch: object) -> None:
    model.train()
    prediction = model(batch)
    assert prediction.shape == (1, 3)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable if parameter.grad is not None)


def _sample() -> UnifiedSample:
    signals = tuple(np.asarray([[value]], dtype=np.float32) for value in (1.0, 2.0, 3.0))
    return UnifiedSample(
        signals=signals,
        sensor_id=(
            "ultrasonic_tof",
            "thermal_conductivity_voltage",
            "ndir_co2_voltage",
        ),
        sensor_type=("acoustic_tof", "thermal_conductivity", "ndir"),
        valid_mask=tuple(np.ones_like(signal, dtype=np.bool_) for signal in signals),
        quality=tuple(np.ones(1, dtype=np.float32) for _ in signals),
        time=tuple(np.zeros(1, dtype=np.float64) for _ in signals),
        target=np.asarray([20.0, 30.0, 50.0], dtype=np.float32),
        target_mask=np.ones(3, dtype=np.bool_),
        group_id="gradient-group",
        dataset_id="gradient-dataset",
        metadata={"split": "train"},
    )


def _load(filename: str) -> dict[str, object]:
    value = json.loads(
        (PROJECT_ROOT / "configs" / "model" / filename).read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value
