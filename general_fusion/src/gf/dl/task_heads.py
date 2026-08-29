from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, fused_representation: torch.Tensor) -> torch.Tensor:
        return self.linear(fused_representation)


class SimplexProjectionHead(nn.Module):
    """Apply a deterministic Euclidean projection onto a fixed-total simplex."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        total: float = 100.0,
        base_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        _validate_head_dimensions(input_dim, output_dim)
        if total <= 0.0 or not torch.isfinite(torch.tensor(total)):
            raise ValueError("total must be finite and positive")
        self.total = float(total)
        self.base_head = base_head if base_head is not None else RegressionHead(input_dim, output_dim)

    def forward(self, fused_representation: torch.Tensor) -> torch.Tensor:
        return project_to_simplex(self.base_head(fused_representation), total=self.total)


class FixedTotalSoftmaxHead(nn.Module):
    """Learn logits and map them to a strictly positive fixed-total composition."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        total: float = 100.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        _validate_head_dimensions(input_dim, output_dim)
        if total <= 0.0 or not torch.isfinite(torch.tensor(total)):
            raise ValueError("total must be finite and positive")
        if temperature <= 0.0 or not torch.isfinite(torch.tensor(temperature)):
            raise ValueError("temperature must be finite and positive")
        self.total = float(total)
        self.temperature = float(temperature)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, fused_representation: torch.Tensor) -> torch.Tensor:
        logits = self.linear(fused_representation) / self.temperature
        return torch.softmax(logits, dim=-1) * self.total


class SparsemaxHead(nn.Module):
    """Learn logits and map them to a fixed-total composition with exact zeros."""

    def __init__(self, input_dim: int, output_dim: int, *, total: float = 100.0) -> None:
        super().__init__()
        _validate_head_dimensions(input_dim, output_dim)
        if total <= 0.0 or not torch.isfinite(torch.tensor(total)):
            raise ValueError("total must be finite and positive")
        self.total = float(total)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, fused_representation: torch.Tensor) -> torch.Tensor:
        return sparsemax(self.linear(fused_representation), dim=-1) * self.total


def project_to_simplex(values: torch.Tensor, *, total: float = 100.0) -> torch.Tensor:
    """Project the last dimension of ``values`` onto ``x >= 0, sum(x)=total``."""

    if values.ndim == 0:
        raise ValueError("values must have at least one dimension")
    if total <= 0.0 or not torch.isfinite(torch.tensor(total, device=values.device)):
        raise ValueError("total must be finite and positive")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    width = values.shape[-1]
    if width == 0:
        raise ValueError("values last dimension must be non-empty")

    flattened = values.reshape(-1, width)
    sorted_values, _ = torch.sort(flattened, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=-1) - total
    ranks = torch.arange(1, width + 1, device=values.device, dtype=values.dtype)
    support = sorted_values - cumulative / ranks > 0.0
    support_size = support.sum(dim=-1).clamp_min(1).to(torch.long)
    support_index = support_size - 1
    threshold = cumulative.gather(1, support_index.unsqueeze(1)).squeeze(1) / support_size.to(values.dtype)
    projected = (flattened - threshold.unsqueeze(1)).clamp_min(0.0)
    return projected.reshape_as(values)


def sparsemax(values: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Sparsemax projection onto the probability simplex."""

    if values.ndim == 0:
        raise ValueError("values must have at least one dimension")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    dimension = dim if dim >= 0 else values.ndim + dim
    if dimension < 0 or dimension >= values.ndim:
        raise ValueError("dim is out of range")
    if dimension != values.ndim - 1:
        moved = values.movedim(dimension, -1)
        projected = sparsemax(moved, dim=-1)
        return projected.movedim(-1, dimension)

    width = values.shape[-1]
    if width == 0:
        raise ValueError("values along dim must be non-empty")
    sorted_values, _ = torch.sort(values, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=-1)
    ranks = torch.arange(1, width + 1, device=values.device, dtype=values.dtype)
    support = 1.0 + ranks * sorted_values > cumulative
    support_size = support.sum(dim=-1).clamp_min(1).to(torch.long)
    support_index = support_size - 1
    cumulative_support = cumulative.gather(-1, support_index.unsqueeze(-1)).squeeze(-1)
    threshold = (cumulative_support - 1.0) / support_size.to(values.dtype)
    return (values - threshold.unsqueeze(-1)).clamp_min(0.0)


def build_task_head(
    config: Mapping[str, object],
    *,
    input_dim: int,
    output_dim: int,
    base_head: nn.Module | None = None,
) -> nn.Module:
    """Build one explicit H0-H3 head from a validated model entry."""

    head_id = config.get("id")
    if head_id == "H0":
        return RegressionHead(input_dim, output_dim)
    if head_id == "H1":
        return SimplexProjectionHead(input_dim, output_dim, base_head=base_head)
    if head_id == "H2":
        temperature = config.get("temperature", 1.0)
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise ValueError("H2 temperature must be numeric")
        return FixedTotalSoftmaxHead(input_dim, output_dim, temperature=float(temperature))
    if head_id == "H3":
        return SparsemaxHead(input_dim, output_dim)
    raise ValueError(f"unsupported A2 task head id: {head_id!r}")


def _validate_head_dimensions(input_dim: int, output_dim: int) -> None:
    if input_dim <= 0 or output_dim <= 0:
        raise ValueError("input_dim and output_dim must be positive")
