from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from gf.dl.contracts import UnifiedSample, collate_samples
from gf.dl.tqif import (
    TQIFModel,
    load_tqif_checkpoint,
)
from gf.pipeline.a2_tqif_benchmark import _read_prediction_matrix
from gf.pipeline.tqif_common import TQIFArtifactError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_target_slot_swap_is_an_explainable_output_permutation() -> None:
    torch.manual_seed(17)
    first = _build_model(("slot-0", "slot-1", "slot-2"))
    torch.manual_seed(17)
    swapped = _build_model(("slot-1", "slot-0", "slot-2"))
    batch = collate_samples([_sample()])
    with torch.no_grad():
        first_output = first(batch)
        swapped_output = swapped(batch)
    assert torch.allclose(swapped_output, first_output[:, [1, 0, 2]], atol=1e-5, rtol=0.0)


@pytest.mark.parametrize(
    "slot_ids",
    [("slot-1", "slot-0", "slot-2"), ("slot-0", "slot-1")],
)
def test_checkpoint_rejects_slot_order_or_set_mismatch(tmp_path: Path, slot_ids: tuple[str, ...]) -> None:
    model = _build_model(("slot-0", "slot-1", "slot-2"))
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema_version": "tqif-checkpoint-1",
            "model_contract": model.checkpoint_contract(),
            "state_dict": model.state_dict(),
        },
        path,
    )
    changed = _build_model(slot_ids)
    with pytest.raises(ValueError, match="CHECKPOINT_CONTRACT_MISMATCH"):
        load_tqif_checkpoint(changed, str(path))


def test_sensor_id_type_mismatch_fails_at_model_boundary() -> None:
    model = _build_model(("slot-0", "slot-1", "slot-2"))
    sample = _sample()
    mismatched = type(sample)(
        signals=sample.signals,
        sensor_id=sample.sensor_id,
        sensor_type=("ndir", "thermal_conductivity", "acoustic_tof"),
        valid_mask=sample.valid_mask,
        quality=sample.quality,
        time=sample.time,
        target=sample.target,
        target_mask=sample.target_mask,
        group_id=sample.group_id,
        dataset_id=sample.dataset_id,
        metadata=sample.metadata,
    )
    with pytest.raises(ValueError, match="sensor_type mismatch"):
        model(collate_samples([mismatched]))


def test_ordinary_forward_does_not_request_attention_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _build_model(("slot-0", "slot-1", "slot-2"))
    batch = collate_samples([_sample()])
    requested: list[bool] = []
    original = model.fusion.sensor_attention.forward

    def wrapped(*args: object, **kwargs: object) -> object:
        requested.append(bool(kwargs["need_weights"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(model.fusion.sensor_attention, "forward", wrapped)
    model(batch)
    assert requested == [False]
    diagnostics = model.forward_with_diagnostics(batch)
    assert diagnostics.prediction.shape == (1, 3)
    assert diagnostics.fusion.sensor_attention.ndim == 4


def test_prediction_reader_reports_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(TQIFArtifactError) as error:
        _read_prediction_matrix(tmp_path / "predictions.csv", ("mix-1",))
    assert error.value.code == "INCOMPLETE_RUN_SET"


def _build_model(slot_ids: tuple[str, ...]) -> TQIFModel:
    return TQIFModel(
        embedding_dim=8,
        token_dim=8,
        pair_hidden_dim=8,
        query_ffn_dim=16,
        attention_heads=2,
        output_dim=len(slot_ids),
        sensor_ids=("sensor-a", "sensor-b", "sensor-c"),
        sensor_types=("type-a", "type-b", "type-c"),
        target_slot_ids=slot_ids,
    )


def _sample() -> UnifiedSample:
    signals = tuple(np.asarray([[value]], dtype=np.float32) for value in (1.0, 2.0, 3.0))
    return UnifiedSample(
        signals=signals,
        sensor_id=("sensor-a", "sensor-b", "sensor-c"),
        sensor_type=("type-a", "type-b", "type-c"),
        valid_mask=tuple(np.ones_like(signal, dtype=np.bool_) for signal in signals),
        quality=tuple(np.ones(1, dtype=np.float32) for _ in signals),
        time=tuple(np.zeros(1, dtype=np.float64) for _ in signals),
        target=np.asarray([20.0, 30.0, 50.0], dtype=np.float32),
        target_mask=np.ones(3, dtype=np.bool_),
        group_id="artifact-group",
        dataset_id="artifact-dataset",
        metadata={"split": "train"},
    )
