from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np

from gf.dl.contracts import UnifiedSample


class ScalerStateError(RuntimeError):
    """Raised when scaler fitting or transformation violates split ownership."""


class TrainGroupStandardScaler:
    def __init__(self) -> None:
        self._statistics: dict[str, tuple[float, float]] | None = None
        self._fitted_group_ids: frozenset[str] = frozenset()

    @property
    def statistics(self) -> Mapping[str, tuple[float, float]]:
        if self._statistics is None:
            raise ScalerStateError("scaler has not been fitted")
        return MappingProxyType(dict(self._statistics))

    @property
    def fitted_group_ids(self) -> frozenset[str]:
        return self._fitted_group_ids

    def fit(self, samples: list[UnifiedSample], train_group_ids: set[str] | frozenset[str]) -> None:
        if self._statistics is not None:
            raise ScalerStateError("scaler instances may only be fitted once")
        if not train_group_ids:
            raise ScalerStateError("train_group_ids must be non-empty")

        known_groups = {sample.group_id for sample in samples}
        unknown_groups = set(train_group_ids) - known_groups
        if unknown_groups:
            raise ScalerStateError(f"unknown training groups: {sorted(unknown_groups)}")

        selected = [sample for sample in samples if sample.group_id in train_group_ids]
        values_by_sensor: dict[str, list[np.ndarray]] = {}
        for sample in selected:
            for sensor_id, signal, mask in zip(sample.sensor_id, sample.signals, sample.valid_mask, strict=True):
                valid_values = signal[mask]
                if valid_values.size:
                    values_by_sensor.setdefault(sensor_id, []).append(valid_values.astype(np.float64, copy=False))

        statistics: dict[str, tuple[float, float]] = {}
        for sensor_id, parts in values_by_sensor.items():
            values = np.concatenate(parts)
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if not np.isfinite(mean) or not np.isfinite(std):
                raise ScalerStateError(f"non-finite scaler statistics for sensor {sensor_id}")
            if std <= 0.0:
                raise ScalerStateError(f"zero variance in training values for sensor {sensor_id}")
            statistics[sensor_id] = (mean, std)

        expected_sensors = {sensor_id for sample in selected for sensor_id in sample.sensor_id}
        missing_sensors = expected_sensors - set(statistics)
        if missing_sensors:
            raise ScalerStateError(f"no valid training values for sensors: {sorted(missing_sensors)}")

        self._statistics = statistics
        self._fitted_group_ids = frozenset(train_group_ids)

    def transform(self, sample: UnifiedSample) -> UnifiedSample:
        if self._statistics is None:
            raise ScalerStateError("scaler must be fitted before transform")

        transformed_signals: list[np.ndarray] = []
        for sensor_id, signal, mask in zip(sample.sensor_id, sample.signals, sample.valid_mask, strict=True):
            if sensor_id not in self._statistics:
                raise KeyError(f"sensor_id {sensor_id!r} was not present during scaler fitting")
            mean, std = self._statistics[sensor_id]
            transformed = np.zeros_like(signal, dtype=np.float32)
            transformed[mask] = ((signal[mask] - mean) / std).astype(np.float32)
            transformed_signals.append(transformed)

        return UnifiedSample(
            signals=tuple(transformed_signals),
            sensor_id=sample.sensor_id,
            sensor_type=sample.sensor_type,
            valid_mask=sample.valid_mask,
            quality=sample.quality,
            time=sample.time,
            target=sample.target,
            target_mask=sample.target_mask,
            group_id=sample.group_id,
            dataset_id=sample.dataset_id,
            metadata=sample.metadata,
        )
