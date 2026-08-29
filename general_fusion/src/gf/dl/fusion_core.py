from __future__ import annotations

import torch
from torch import nn


class FusionCore(nn.Module):
    """Set fusion over a variable number of sensors.

    ``quality_weighted_mean`` preserves the A0 behavior. A2 uses
    ``masked_mean`` as its registered Deep Sets aggregation and ``sum`` only
    as a single-factor ablation.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        *,
        pooling: str = "quality_weighted_mean",
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("embedding_dim and hidden_dim must be positive")
        if pooling not in {"quality_weighted_mean", "masked_mean", "sum"}:
            raise ValueError(
                "pooling must be quality_weighted_mean, masked_mean, or sum"
            )
        self.pooling = pooling
        self.output_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        sensor_embeddings: torch.Tensor,
        sensor_mask: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sensor_embeddings.ndim != 3:
            raise ValueError("sensor_embeddings must have shape [B,S,D]")
        if sensor_mask.shape != sensor_embeddings.shape[:2]:
            raise ValueError("sensor_mask shape must match [B,S]")
        if reliability is not None and reliability.shape != sensor_embeddings.shape[:2]:
            raise ValueError("reliability shape must match [B,S]")
        if not torch.isfinite(sensor_embeddings).all():
            raise ValueError("sensor embeddings and reliability must be finite")
        if sensor_mask.dtype != torch.bool:
            raise ValueError("sensor_mask must be boolean")
        if not torch.any(sensor_mask, dim=1).all():
            raise ValueError("each sample must contain at least one sensor")

        if self.pooling == "quality_weighted_mean":
            if reliability is None:
                raise ValueError("quality_weighted_mean requires reliability")
            if not torch.isfinite(reliability).all():
                raise ValueError("sensor embeddings and reliability must be finite")
            if torch.any((reliability < 0.0) | (reliability > 1.0)):
                raise ValueError("reliability must be within [0,1]")
            weights = reliability * sensor_mask.to(reliability.dtype)
            denominator = weights.sum(dim=1, keepdim=True)
            if torch.any(denominator <= 0.0):
                raise ValueError("each sample must contain at least one sensor with positive reliability")
            pooled = (sensor_embeddings * weights.unsqueeze(-1)).sum(dim=1) / denominator
        elif self.pooling == "masked_mean":
            weights = sensor_mask.to(sensor_embeddings.dtype)
            denominator = weights.sum(dim=1, keepdim=True)
            pooled = (sensor_embeddings * weights.unsqueeze(-1)).sum(dim=1) / denominator
        else:
            weights = sensor_mask.to(sensor_embeddings.dtype)
            pooled = (sensor_embeddings * weights.unsqueeze(-1)).sum(dim=1)
        return self.projection(pooled)


class ConcatFusionCore(nn.Module):
    """Ordered concatenation control for a registered maximum sensor count."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        *,
        max_sensors: int,
        concat_dim: int | None = None,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or hidden_dim <= 0 or max_sensors <= 0:
            raise ValueError("embedding_dim, hidden_dim, and max_sensors must be positive")
        if concat_dim is not None and concat_dim <= 0:
            raise ValueError("concat_dim must be positive when supplied")
        self.output_dim = hidden_dim
        self.max_sensors = max_sensors
        self.concat_dim = concat_dim
        self.token_projection = (
            nn.Linear(embedding_dim, concat_dim) if concat_dim is not None else nn.Identity()
        )
        flattened_dim = (concat_dim or embedding_dim) * max_sensors
        self.projection = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, sensor_embeddings: torch.Tensor, sensor_mask: torch.Tensor) -> torch.Tensor:
        if sensor_embeddings.ndim != 3:
            raise ValueError("sensor_embeddings must have shape [B,S,D]")
        if sensor_embeddings.shape[1] > self.max_sensors:
            raise ValueError("sensor_embeddings contains more sensors than max_sensors")
        if sensor_mask.shape != sensor_embeddings.shape[:2] or sensor_mask.dtype != torch.bool:
            raise ValueError("sensor_mask must be boolean with shape [B,S]")
        if not torch.isfinite(sensor_embeddings).all():
            raise ValueError("sensor_embeddings must be finite")
        if not torch.any(sensor_mask, dim=1).all():
            raise ValueError("each sample must contain at least one sensor")

        projected = self.token_projection(sensor_embeddings)
        projected = projected * sensor_mask.unsqueeze(-1)
        if projected.shape[1] < self.max_sensors:
            padding = torch.zeros(
                projected.shape[0],
                self.max_sensors - projected.shape[1],
                projected.shape[2],
                dtype=projected.dtype,
                device=projected.device,
            )
            projected = torch.cat((projected, padding), dim=1)
        return self.projection(projected.reshape(projected.shape[0], -1))
