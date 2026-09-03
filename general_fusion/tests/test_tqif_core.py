from __future__ import annotations

import numpy as np
import torch

from gf.dl.contracts import UnifiedSample, collate_samples
from gf.dl.tqif import TQIFModel


SENSOR_IDS = ("sensor-a", "sensor-b", "sensor-c")
SENSOR_TYPES = ("type-a", "type-b", "type-c")
TARGET_SLOTS = ("slot-0", "slot-1", "slot-2")


def test_tqif_is_invariant_to_sensor_and_pair_permutation() -> None:
    torch.manual_seed(17)
    model = TQIFModel(
        embedding_dim=8,
        token_dim=8,
        pair_hidden_dim=8,
        query_ffn_dim=16,
        attention_heads=2,
        output_dim=3,
        sensor_ids=SENSOR_IDS,
        sensor_types=SENSOR_TYPES,
        target_slot_ids=TARGET_SLOTS,
    )
    model.eval()
    first = collate_samples([_sample(SENSOR_IDS, SENSOR_TYPES, [1.0, 2.0, 3.0], "g1")])
    permutation = (2, 0, 1)
    second = collate_samples(
        [
            _sample(
                tuple(SENSOR_IDS[index] for index in permutation),
                tuple(SENSOR_TYPES[index] for index in permutation),
                [3.0, 1.0, 2.0],
                "g2",
            )
        ]
    )
    with torch.no_grad():
        first_output = model(first)
        second_output = model(second)
    assert torch.allclose(first_output, second_output, atol=1e-5, rtol=0.0)

    first_diagnostics = model.forward_with_diagnostics(first).fusion
    second_diagnostics = model.forward_with_diagnostics(second).fusion
    assert torch.isfinite(first_diagnostics.pair_tokens).all()
    assert torch.isfinite(second_diagnostics.pair_tokens).all()
    assert first_diagnostics.pair_mask.sum().item() == second_diagnostics.pair_mask.sum().item() == 3


def test_tqif_retains_sensor_identity_in_sensor_tokens() -> None:
    torch.manual_seed(19)
    model = TQIFModel(
        embedding_dim=8,
        token_dim=8,
        pair_hidden_dim=8,
        query_ffn_dim=16,
        attention_heads=2,
        output_dim=3,
        sensor_ids=SENSOR_IDS,
        sensor_types=SENSOR_TYPES,
        target_slot_ids=TARGET_SLOTS,
    )
    first = collate_samples([_sample(SENSOR_IDS, SENSOR_TYPES, [1.0, 2.0, 3.0], "g1")])
    changed_identity = collate_samples(
        [_sample(("sensor-c", "sensor-b", "sensor-a"), ("type-c", "type-b", "type-a"), [1.0, 2.0, 3.0], "g2")]
    )
    first_encoding = model.encoder(first)
    changed_encoding = model.encoder(changed_identity)
    assert not torch.allclose(first_encoding.sensor_embeddings, changed_encoding.sensor_embeddings)


def test_tqif_masks_missing_sensor_and_explicitly_degenerates_for_one_sensor() -> None:
    torch.manual_seed(23)
    model = TQIFModel(
        embedding_dim=8,
        token_dim=8,
        pair_hidden_dim=8,
        query_ffn_dim=16,
        attention_heads=2,
        output_dim=3,
        sensor_ids=SENSOR_IDS,
        sensor_types=SENSOR_TYPES,
        target_slot_ids=TARGET_SLOTS,
    )
    model.eval()
    missing = collate_samples(
        [
            _sample(
                SENSOR_IDS,
                SENSOR_TYPES,
                [1.0, 2.0, 0.0],
                "missing",
                valid=[True, True, False],
            )
        ]
    )
    missing_diagnostics = model.forward_with_diagnostics(missing)
    assert torch.isfinite(missing_diagnostics.prediction).all()
    assert missing_diagnostics.fusion.pair_mask.sum().item() == 1
    assert torch.isfinite(missing_diagnostics.fusion.sensor_attention).all()
    assert torch.isfinite(missing_diagnostics.fusion.pair_attention).all()

    one_sensor = collate_samples([_sample(("sensor-a",), ("type-a",), [1.0], "one")])
    one_sensor_diagnostics = model.forward_with_diagnostics(one_sensor)
    assert one_sensor_diagnostics.fusion.pair_tokens.shape == (1, 0, 8)
    assert one_sensor_diagnostics.fusion.pair_attention.shape == (1, 2, 3, 0)
    assert torch.equal(
        one_sensor_diagnostics.fusion.gate,
        torch.zeros_like(one_sensor_diagnostics.fusion.gate),
    )


def test_tqif_forward_backward_is_finite() -> None:
    torch.manual_seed(29)
    model = TQIFModel(
        embedding_dim=8,
        token_dim=8,
        pair_hidden_dim=8,
        query_ffn_dim=16,
        attention_heads=2,
        output_dim=3,
        sensor_ids=SENSOR_IDS,
        sensor_types=SENSOR_TYPES,
        target_slot_ids=TARGET_SLOTS,
    )
    batch = collate_samples([_sample(SENSOR_IDS, SENSOR_TYPES, [1.0, 2.0, 3.0], "g1")])
    prediction = model(batch)
    assert prediction.shape == (1, 3)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable if parameter.grad is not None)


def test_tqif_checkpoint_round_trip_is_deterministic(tmp_path) -> None:
    torch.manual_seed(31)
    model = TQIFModel(
        embedding_dim=8,
        token_dim=8,
        pair_hidden_dim=8,
        query_ffn_dim=16,
        attention_heads=2,
        output_dim=3,
        sensor_ids=SENSOR_IDS,
        sensor_types=SENSOR_TYPES,
        target_slot_ids=TARGET_SLOTS,
    )
    model.eval()
    batch = collate_samples([_sample(SENSOR_IDS, SENSOR_TYPES, [1.0, 2.0, 3.0], "g1")])
    with torch.no_grad():
        expected = model(batch)
    checkpoint = tmp_path / "tqif-checkpoint.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)

    restored = TQIFModel(
        embedding_dim=8,
        token_dim=8,
        pair_hidden_dim=8,
        query_ffn_dim=16,
        attention_heads=2,
        output_dim=3,
        sensor_ids=SENSOR_IDS,
        sensor_types=SENSOR_TYPES,
        target_slot_ids=TARGET_SLOTS,
    )
    payload = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(payload["state_dict"])
    restored.eval()
    with torch.no_grad():
        actual = restored(batch)
    assert torch.equal(expected, actual)


def _sample(
    sensor_ids: tuple[str, ...],
    sensor_types: tuple[str, ...],
    values: list[float],
    group_id: str,
    *,
    valid: list[bool] | None = None,
) -> UnifiedSample:
    if valid is None:
        valid = [True] * len(values)
    signals = tuple(np.asarray([[value]], dtype=np.float32) for value in values)
    valid_masks = tuple(
        np.asarray([[is_valid]], dtype=np.bool_) for is_valid in valid
    )
    quality = tuple(
        np.asarray([1.0 if is_valid else 0.0], dtype=np.float32)
        for is_valid in valid
    )
    return UnifiedSample(
        signals=signals,
        sensor_id=sensor_ids,
        sensor_type=sensor_types,
        valid_mask=valid_masks,
        quality=quality,
        time=tuple(np.asarray([0.0], dtype=np.float64) for _ in values),
        target=np.asarray([20.0, 30.0, 50.0], dtype=np.float32),
        target_mask=np.ones(3, dtype=np.bool_),
        group_id=group_id,
        dataset_id="fixture",
        metadata={"split": "train"},
    )
