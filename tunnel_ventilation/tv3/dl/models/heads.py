from __future__ import annotations

import torch
from torch import nn


def build_regression_head(in_features: int, out_dim: int, dropout: float) -> nn.Sequential:
    """共享 MLP 回归 head：in_features→128→ReLU→Drop→64→ReLU→Drop→out_dim。

    CNN1D 与 TCN 编码器在 ``AdaptiveAvgPool1d`` 池化后接入完全相同的 head，
    抽取到此处避免逐行复制（见 KARPATHY_REVIEW 2.1）。两模型的 head 设计若需
    分化，应在此显式参数化，而不是各自维护一份副本。
    """
    return nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(64, out_dim),
    )


class TemporalPooling(nn.Module):
    """Time aggregation head for encoded NCT sequences."""

    def __init__(self, in_channels: int, mode: str = "mean"):
        super().__init__()
        if mode not in {"mean", "last", "attention"}:
            raise ValueError("pooling mode must be one of ['mean', 'last', 'attention']")
        self.mode = mode
        self.attention = nn.Conv1d(in_channels, 1, kernel_size=1) if mode == "attention" else None

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 3:
            raise ValueError(f"encoded must be NCT, got ndim={encoded.ndim}")
        if self.mode == "mean":
            return encoded.mean(dim=-1)
        if self.mode == "last":
            return encoded[:, :, -1]
        assert self.attention is not None
        weights = torch.softmax(self.attention(encoded), dim=-1)
        return torch.sum(encoded * weights, dim=-1)
