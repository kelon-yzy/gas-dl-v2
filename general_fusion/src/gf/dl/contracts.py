from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, TypeAlias

import numpy as np
import torch


MetadataScalar: TypeAlias = str | int | float | bool


class ContractError(ValueError):
    """Raised when a sample or batch violates the frozen A0 contract."""


@dataclass(frozen=True)
class UnifiedSample:
    signals: tuple[np.ndarray, ...]
    sensor_id: tuple[str, ...]
    sensor_type: tuple[str, ...]
    valid_mask: tuple[np.ndarray, ...]
    quality: tuple[np.ndarray, ...]
    time: tuple[np.ndarray, ...]
    target: np.ndarray
    target_mask: np.ndarray
    group_id: str
    dataset_id: str
    metadata: Mapping[str, MetadataScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        signals = tuple(_readonly_array(value, np.float32) for value in self.signals)
        valid_mask = tuple(_readonly_array(value, np.bool_) for value in self.valid_mask)
        quality = tuple(_readonly_array(value, np.float32) for value in self.quality)
        time = tuple(_readonly_array(value, np.float64) for value in self.time)
        sensor_id = tuple(self.sensor_id)
        sensor_type = tuple(self.sensor_type)

        sensor_count = len(signals)
        if sensor_count == 0:
            raise ContractError("signals must contain at least one sensor")
        lengths = {
            "sensor_id": len(sensor_id),
            "sensor_type": len(sensor_type),
            "valid_mask": len(valid_mask),
            "quality": len(quality),
            "time": len(time),
        }
        mismatched = {name: length for name, length in lengths.items() if length != sensor_count}
        if mismatched:
            raise ContractError(f"sensor field lengths must equal {sensor_count}, got {mismatched}")
        if any(not value for value in sensor_id):
            raise ContractError("sensor_id values must be non-empty")
        if len(set(sensor_id)) != sensor_count:
            raise ContractError("sensor_id values must be unique within a sample")
        if any(not value for value in sensor_type):
            raise ContractError("sensor_type values must be non-empty")

        for index, (signal, mask, sensor_quality, sensor_time) in enumerate(
            zip(signals, valid_mask, quality, time, strict=True)
        ):
            name = sensor_id[index]
            if signal.ndim != 2:
                raise ContractError(f"signals[{name}] must have shape [T,F], got {signal.shape}")
            if signal.shape[0] == 0 or signal.shape[1] == 0:
                raise ContractError(f"signals[{name}] must have non-empty T and F axes")
            if mask.shape != signal.shape:
                raise ContractError(
                    f"valid_mask[{name}] shape {mask.shape} does not match signal shape {signal.shape}"
                )
            if sensor_quality.shape != (signal.shape[0],):
                raise ContractError(
                    f"quality[{name}] must have shape ({signal.shape[0]},), got {sensor_quality.shape}"
                )
            if sensor_time.shape != (signal.shape[0],):
                raise ContractError(f"time[{name}] must have shape ({signal.shape[0]},), got {sensor_time.shape}")
            if not np.isfinite(signal[mask]).all():
                raise ContractError(f"signals[{name}] contains non-finite values at valid positions")
            if np.any(signal[~mask] != 0.0):
                raise ContractError(f"signals[{name}] must store 0 at invalid positions")
            if not np.isfinite(sensor_quality).all() or np.any((sensor_quality < 0.0) | (sensor_quality > 1.0)):
                raise ContractError(f"quality[{name}] must be finite and within [0,1]")
            if not np.isfinite(sensor_time).all():
                raise ContractError(f"time[{name}] must be finite")
            if sensor_time.size > 1 and np.any(np.diff(sensor_time) <= 0.0):
                raise ContractError(f"time[{name}] must be strictly increasing")

        target = _readonly_array(self.target, np.float32)
        target_mask = _readonly_array(self.target_mask, np.bool_)
        if target.ndim != 1 or target.size == 0:
            raise ContractError(f"target must be a non-empty 1D array, got {target.shape}")
        if target_mask.shape != target.shape:
            raise ContractError(f"target_mask shape {target_mask.shape} does not match target shape {target.shape}")
        if not np.isfinite(target).all():
            raise ContractError("target must contain only finite values")
        if np.any(target[~target_mask] != 0.0):
            raise ContractError("target must store 0 where target_mask is false")
        if not self.group_id:
            raise ContractError("group_id must be non-empty")
        if not self.dataset_id:
            raise ContractError("dataset_id must be non-empty")

        metadata = dict(self.metadata)
        invalid_metadata = {key: type(value).__name__ for key, value in metadata.items() if not _is_metadata_scalar(value)}
        if invalid_metadata:
            raise ContractError(f"metadata values must be scalar, got {invalid_metadata}")

        object.__setattr__(self, "signals", signals)
        object.__setattr__(self, "sensor_id", sensor_id)
        object.__setattr__(self, "sensor_type", sensor_type)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "target_mask", target_mask)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class UnifiedBatch:
    signals: torch.Tensor
    valid_mask: torch.Tensor
    quality: torch.Tensor
    time: torch.Tensor
    delta_time: torch.Tensor
    sensor_mask: torch.Tensor
    feature_mask: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor
    sensor_id: tuple[tuple[str, ...], ...]
    sensor_type: tuple[tuple[str, ...], ...]
    group_id: tuple[str, ...]
    dataset_id: tuple[str, ...]
    metadata: tuple[Mapping[str, MetadataScalar], ...]


def collate_samples(samples: list[UnifiedSample] | tuple[UnifiedSample, ...]) -> UnifiedBatch:
    if not samples:
        raise ContractError("collate_samples requires at least one sample")

    batch_size = len(samples)
    max_sensors = max(len(sample.signals) for sample in samples)
    max_time = max(signal.shape[0] for sample in samples for signal in sample.signals)
    max_features = max(signal.shape[1] for sample in samples for signal in sample.signals)
    max_targets = max(sample.target.size for sample in samples)

    signals = torch.zeros((batch_size, max_sensors, max_time, max_features), dtype=torch.float32)
    valid_mask = torch.zeros_like(signals, dtype=torch.bool)
    quality = torch.zeros((batch_size, max_sensors, max_time), dtype=torch.float32)
    time = torch.zeros((batch_size, max_sensors, max_time), dtype=torch.float64)
    delta_time = torch.zeros((batch_size, max_sensors, max_time), dtype=torch.float32)
    sensor_mask = torch.zeros((batch_size, max_sensors), dtype=torch.bool)
    feature_mask = torch.zeros((batch_size, max_sensors, max_features), dtype=torch.bool)
    target = torch.zeros((batch_size, max_targets), dtype=torch.float32)
    target_mask = torch.zeros((batch_size, max_targets), dtype=torch.bool)
    sensor_ids: list[tuple[str, ...]] = []
    sensor_types: list[tuple[str, ...]] = []

    for batch_index, sample in enumerate(samples):
        padded_ids = list(sample.sensor_id) + [""] * (max_sensors - len(sample.sensor_id))
        padded_types = list(sample.sensor_type) + [""] * (max_sensors - len(sample.sensor_type))
        sensor_ids.append(tuple(padded_ids))
        sensor_types.append(tuple(padded_types))
        sensor_mask[batch_index, : len(sample.signals)] = True
        target_width = sample.target.size
        target[batch_index, :target_width] = _tensor_copy(sample.target)
        target_mask[batch_index, :target_width] = _tensor_copy(sample.target_mask)

        for sensor_index, (signal, mask, sensor_quality, sensor_time) in enumerate(
            zip(sample.signals, sample.valid_mask, sample.quality, sample.time, strict=True)
        ):
            time_length, feature_width = signal.shape
            signals[batch_index, sensor_index, :time_length, :feature_width] = _tensor_copy(signal)
            valid_mask[batch_index, sensor_index, :time_length, :feature_width] = _tensor_copy(mask)
            quality[batch_index, sensor_index, :time_length] = _tensor_copy(sensor_quality)
            time[batch_index, sensor_index, :time_length] = _tensor_copy(sensor_time)
            if time_length > 1:
                delta = np.zeros(time_length, dtype=np.float32)
                delta[1:] = np.diff(sensor_time).astype(np.float32)
                delta_time[batch_index, sensor_index, :time_length] = torch.from_numpy(delta)
            feature_mask[batch_index, sensor_index, :feature_width] = True

    return UnifiedBatch(
        signals=signals,
        valid_mask=valid_mask,
        quality=quality,
        time=time,
        delta_time=delta_time,
        sensor_mask=sensor_mask,
        feature_mask=feature_mask,
        target=target,
        target_mask=target_mask,
        sensor_id=tuple(sensor_ids),
        sensor_type=tuple(sensor_types),
        group_id=tuple(sample.group_id for sample in samples),
        dataset_id=tuple(sample.dataset_id for sample in samples),
        metadata=tuple(sample.metadata for sample in samples),
    )


def _readonly_array(value: np.ndarray, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _tensor_copy(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(value, copy=True))


def _is_metadata_scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool)) and not isinstance(value, complex)
