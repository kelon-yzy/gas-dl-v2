from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from gf.dl.contracts import ContractError, UnifiedBatch


class MaskedStatSensorEncoder(nn.Module):
    """Encode variable-length sensor observations without dataset-specific branches."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        sensor_ids: Sequence[str],
        sensor_types: Sequence[str],
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.sensor_id_to_index = _build_vocabulary(sensor_ids, "sensor_ids")
        self.sensor_type_to_index = _build_vocabulary(sensor_types, "sensor_types")
        self.signal_projection = nn.Linear(4, embedding_dim)
        self.sensor_id_embedding = nn.Embedding(len(self.sensor_id_to_index), embedding_dim)
        self.sensor_type_embedding = nn.Embedding(len(self.sensor_type_to_index), embedding_dim)
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, batch: UnifiedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        valid = batch.valid_mask
        counts = valid.sum(dim=(-1, -2)).to(batch.signals.dtype)
        safe_counts = counts.clamp_min(1.0)
        signal_sum = (batch.signals * valid).sum(dim=(-1, -2))
        mean = signal_sum / safe_counts
        centered = batch.signals - mean.unsqueeze(-1).unsqueeze(-1)
        variance = (centered.square() * valid).sum(dim=(-1, -2)) / safe_counts
        std = variance.sqrt()

        valid_time = valid.any(dim=-1)
        time_counts = valid_time.sum(dim=-1).to(batch.quality.dtype)
        safe_time_counts = time_counts.clamp_min(1.0)
        reliability = (batch.quality * valid_time).sum(dim=-1) / safe_time_counts
        mean_delta_time = (batch.delta_time * valid_time).sum(dim=-1) / safe_time_counts

        statistics = torch.stack((mean, std, reliability, mean_delta_time), dim=-1)
        sensor_id_index, sensor_type_index = self._metadata_indices(batch)
        embedding = (
            self.signal_projection(statistics)
            + self.sensor_id_embedding(sensor_id_index)
            + self.sensor_type_embedding(sensor_type_index)
        )
        embedding = self.normalization(embedding)
        embedding = embedding * batch.sensor_mask.unsqueeze(-1)
        reliability = reliability * batch.sensor_mask
        return embedding, reliability

    def _metadata_indices(self, batch: UnifiedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sensor_count = batch.sensor_mask.shape
        id_indices = torch.zeros((batch_size, sensor_count), dtype=torch.long, device=batch.signals.device)
        type_indices = torch.zeros_like(id_indices)
        for batch_index in range(batch_size):
            for sensor_index in range(sensor_count):
                if not bool(batch.sensor_mask[batch_index, sensor_index]):
                    continue
                sensor_id = batch.sensor_id[batch_index][sensor_index]
                sensor_type = batch.sensor_type[batch_index][sensor_index]
                if sensor_id not in self.sensor_id_to_index:
                    raise KeyError(f"unknown sensor_id {sensor_id!r}")
                if sensor_type not in self.sensor_type_to_index:
                    raise KeyError(f"unknown sensor_type {sensor_type!r}")
                id_indices[batch_index, sensor_index] = self.sensor_id_to_index[sensor_id]
                type_indices[batch_index, sensor_index] = self.sensor_type_to_index[sensor_type]
        if torch.any(batch.sensor_mask & (batch.valid_mask.sum(dim=(-1, -2)) == 0)):
            missing = batch.sensor_mask & (batch.valid_mask.sum(dim=(-1, -2)) == 0)
            if torch.any(batch.quality[missing] != 0.0):
                raise ContractError("fully missing sensors must have zero quality")
        return id_indices, type_indices


def _build_vocabulary(values: Sequence[str], name: str) -> dict[str, int]:
    normalized = tuple(values)
    if not normalized or any(not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return {value: index for index, value in enumerate(normalized)}


class A2ScalarTokenEncoder(nn.Module):
    """Encode one masked scalar observation per sensor with identity and type."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        sensor_ids: Sequence[str],
        sensor_types: Sequence[str],
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.sensor_id_to_index = _build_vocabulary(sensor_ids, "sensor_ids")
        self.sensor_type_to_index = _build_vocabulary(sensor_types, "sensor_types")
        self.observation_projection = nn.Linear(1, embedding_dim)
        self.sensor_id_embedding = nn.Embedding(len(self.sensor_id_to_index), embedding_dim)
        self.sensor_type_embedding = nn.Embedding(len(self.sensor_type_to_index), embedding_dim)
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, batch: UnifiedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        valid = batch.valid_mask
        counts = valid.sum(dim=(-1, -2))
        active = batch.sensor_mask
        if torch.any(active & (counts == 0)):
            raise ValueError("A2 scalar token encoder requires one valid observation per active sensor")
        safe_counts = counts.clamp_min(1).to(batch.signals.dtype)
        observation = (batch.signals * valid).sum(dim=(-1, -2)) / safe_counts
        id_indices, type_indices = self._metadata_indices(batch)
        tokens = (
            self.observation_projection(observation.unsqueeze(-1))
            + self.sensor_id_embedding(id_indices)
            + self.sensor_type_embedding(type_indices)
        )
        tokens = self.normalization(tokens)
        tokens = tokens * active.unsqueeze(-1)
        return tokens, active

    def _metadata_indices(self, batch: UnifiedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sensor_count = batch.sensor_mask.shape
        id_indices = torch.zeros((batch_size, sensor_count), dtype=torch.long, device=batch.signals.device)
        type_indices = torch.zeros_like(id_indices)
        for batch_index in range(batch_size):
            for sensor_index in range(sensor_count):
                if not bool(batch.sensor_mask[batch_index, sensor_index]):
                    continue
                sensor_id = batch.sensor_id[batch_index][sensor_index]
                sensor_type = batch.sensor_type[batch_index][sensor_index]
                if sensor_id not in self.sensor_id_to_index:
                    raise KeyError(f"unknown sensor_id {sensor_id!r}")
                if sensor_type not in self.sensor_type_to_index:
                    raise KeyError(f"unknown sensor_type {sensor_type!r}")
                id_indices[batch_index, sensor_index] = self.sensor_id_to_index[sensor_id]
                type_indices[batch_index, sensor_index] = self.sensor_type_to_index[sensor_type]
        return id_indices, type_indices
