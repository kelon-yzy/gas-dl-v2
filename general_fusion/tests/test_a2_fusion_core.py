from __future__ import annotations

import numpy as np
import pytest
import torch

from gf.dl.contracts import UnifiedSample, collate_samples
from gf.dl.fusion_core import ConcatFusionCore, FusionCore
from gf.dl.sensor_encoders import A2ScalarTokenEncoder


SENSOR_IDS = ("sensor-a", "sensor-b", "sensor-c")
SENSOR_TYPES = ("type-a", "type-b", "type-c")


def test_a2_tokens_and_masked_mean_are_permutation_invariant() -> None:
    torch.manual_seed(17)
    encoder = A2ScalarTokenEncoder(
        embedding_dim=8,
        sensor_ids=SENSOR_IDS,
        sensor_types=SENSOR_TYPES,
    )
    core = FusionCore(embedding_dim=8, hidden_dim=6, pooling="masked_mean")
    first = collate_samples([_sample(SENSOR_IDS, SENSOR_TYPES, [1.0, 2.0, 3.0], qualities=[1.0, 0.1, 0.8])])
    permutation = (2, 0, 1)
    second = collate_samples(
        [
            _sample(
                tuple(SENSOR_IDS[index] for index in permutation),
                tuple(SENSOR_TYPES[index] for index in permutation),
                [3.0, 1.0, 2.0],
                qualities=[0.2, 0.9, 0.4],
            )
        ]
    )
    first_tokens, first_mask = encoder(first)
    second_tokens, second_mask = encoder(second)
    first_output = core(first_tokens, first_mask)
    second_output = core(second_tokens, second_mask)
    assert torch.allclose(first_output, second_output, atol=1e-6, rtol=0.0)


def test_a2_encoder_does_not_use_quality_or_time() -> None:
    torch.manual_seed(29)
    encoder = A2ScalarTokenEncoder(
        embedding_dim=8,
        sensor_ids=SENSOR_IDS,
        sensor_types=SENSOR_TYPES,
    )
    first = collate_samples([_sample(SENSOR_IDS, SENSOR_TYPES, [1.0, 2.0, 3.0], qualities=[1.0] * 3)])
    second = collate_samples([_sample(SENSOR_IDS, SENSOR_TYPES, [1.0, 2.0, 3.0], qualities=[0.0, 0.5, 0.2], times=[10.0, 20.0, 30.0])])
    first_tokens, _ = encoder(first)
    second_tokens, _ = encoder(second)
    assert torch.equal(first_tokens, second_tokens)


def test_masked_mean_ignores_padding_but_sum_does_not() -> None:
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, False]])
    mean_core = FusionCore(2, 2, pooling="masked_mean")
    sum_core = FusionCore(2, 2, pooling="sum")
    mean_core.eval()
    sum_core.eval()
    with torch.no_grad():
        mean_output = mean_core(embeddings, mask)
        sum_output = sum_core(embeddings, mask)
    assert mean_output.shape == sum_output.shape == (1, 2)
    assert torch.isfinite(mean_output).all()
    assert torch.isfinite(sum_output).all()


def test_concat_core_rejects_more_than_registered_sensors() -> None:
    core = ConcatFusionCore(4, 4, max_sensors=2)
    with pytest.raises(ValueError, match="more sensors"):
        core(torch.zeros(1, 3, 4), torch.ones(1, 3, dtype=torch.bool))


def _sample(
    sensor_ids: tuple[str, ...],
    sensor_types: tuple[str, ...],
    values: list[float],
    *,
    qualities: list[float],
    times: list[float] | None = None,
) -> UnifiedSample:
    sensor_signals = tuple(np.array([[value]], dtype=np.float32) for value in values)
    return UnifiedSample(
        signals=sensor_signals,
        sensor_id=sensor_ids,
        sensor_type=sensor_types,
        valid_mask=tuple(np.ones_like(signal, dtype=np.bool_) for signal in sensor_signals),
        quality=tuple(np.asarray([quality], dtype=np.float32) for quality in qualities),
        time=tuple(
            np.asarray([time], dtype=np.float64)
            for time in (times or [0.0] * len(sensor_ids))
        ),
        target=np.array([20.0, 30.0, 50.0], dtype=np.float32),
        target_mask=np.ones(3, dtype=np.bool_),
        group_id="group-1",
        dataset_id="fixture",
        metadata={"split": "train"},
    )
