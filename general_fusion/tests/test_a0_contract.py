from __future__ import annotations

import numpy as np
import pytest

from gf.dl.contracts import ContractError, UnifiedSample, collate_samples
from gf.dl.preprocessing import ScalerStateError, TrainGroupStandardScaler


def test_unified_sample_and_collate_preserve_masks_and_variable_shapes() -> None:
    first = _sample(
        group_id="train-group",
        sensor_id="sensor-a",
        values=np.array([[1.0], [2.0]], dtype=np.float32),
    )
    second = UnifiedSample(
        signals=(np.array([[3.0, 4.0], [5.0, 0.0], [7.0, 8.0]], dtype=np.float32),),
        sensor_id=("sensor-b",),
        sensor_type=("type-b",),
        valid_mask=(np.array([[True, True], [True, False], [True, True]]),),
        quality=(np.array([1.0, 0.5, 1.0], dtype=np.float32),),
        time=(np.array([0.0, 0.5, 2.0], dtype=np.float64),),
        target=np.array([2.0, 3.0, 4.0], dtype=np.float32),
        target_mask=np.array([True, True, True]),
        group_id="val-group",
        dataset_id="dataset-b",
        metadata={"source": "fixture"},
    )

    batch = collate_samples([first, second])

    assert batch.signals.shape == (2, 1, 3, 2)
    assert batch.valid_mask.shape == batch.signals.shape
    assert batch.feature_mask.tolist() == [[[True, False]], [[True, True]]]
    assert batch.delta_time[1, 0, :3].tolist() == pytest.approx([0.0, 0.5, 1.5])
    assert batch.signals[1, 0, 1, 1].item() == 0.0
    assert not batch.valid_mask[1, 0, 1, 1].item()
    with pytest.raises(ValueError):
        first.signals[0][0, 0] = 9.0


def test_contract_rejects_nonzero_invalid_values_and_nonincreasing_time() -> None:
    with pytest.raises(ContractError, match="must store 0"):
        UnifiedSample(
            signals=(np.array([[1.0], [2.0]], dtype=np.float32),),
            sensor_id=("sensor-a",),
            sensor_type=("type-a",),
            valid_mask=(np.array([[True], [False]]),),
            quality=(np.ones(2, dtype=np.float32),),
            time=(np.array([0.0, 1.0]),),
            target=np.ones(3, dtype=np.float32),
            target_mask=np.ones(3, dtype=np.bool_),
            group_id="group-a",
            dataset_id="dataset-a",
        )
    with pytest.raises(ContractError, match="strictly increasing"):
        UnifiedSample(
            signals=(np.array([[1.0], [2.0]], dtype=np.float32),),
            sensor_id=("sensor-a",),
            sensor_type=("type-a",),
            valid_mask=(np.ones((2, 1), dtype=np.bool_),),
            quality=(np.ones(2, dtype=np.float32),),
            time=(np.array([1.0, 1.0]),),
            target=np.ones(3, dtype=np.float32),
            target_mask=np.ones(3, dtype=np.bool_),
            group_id="group-a",
            dataset_id="dataset-a",
        )


def test_scaler_fits_only_explicit_training_groups_and_is_single_use() -> None:
    train = _sample(
        group_id="train-group",
        sensor_id="sensor-a",
        values=np.array([[1.0], [3.0]], dtype=np.float32),
    )
    validation = _sample(
        group_id="val-group",
        sensor_id="sensor-a",
        values=np.array([[100.0], [102.0]], dtype=np.float32),
    )
    scaler = TrainGroupStandardScaler()

    with pytest.raises(ScalerStateError, match="fitted"):
        scaler.transform(validation)
    scaler.fit([train, validation], {"train-group"})
    transformed = scaler.transform(validation)

    assert scaler.statistics["sensor-a"] == pytest.approx((2.0, 1.0))
    assert scaler.fitted_group_ids == frozenset({"train-group"})
    assert transformed.signals[0][:, 0].tolist() == pytest.approx([98.0, 100.0])
    with pytest.raises(ScalerStateError, match="only be fitted once"):
        scaler.fit([train, validation], {"train-group"})


def _sample(*, group_id: str, sensor_id: str, values: np.ndarray) -> UnifiedSample:
    time = np.arange(values.shape[0], dtype=np.float64)
    return UnifiedSample(
        signals=(values,),
        sensor_id=(sensor_id,),
        sensor_type=("type-a",),
        valid_mask=(np.ones_like(values, dtype=np.bool_),),
        quality=(np.ones(values.shape[0], dtype=np.float32),),
        time=(time,),
        target=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        target_mask=np.ones(3, dtype=np.bool_),
        group_id=group_id,
        dataset_id="dataset-a",
        metadata={"split": "train" if group_id.startswith("train") else "val"},
    )
