from __future__ import annotations

import torch
from torch import nn

from sg.dl.models.base import BaseRegressor
from sg.dl.models.heads import TemporalPooling, build_regression_head


class CNN1DRegressor(BaseRegressor):
    """1D 卷积回归器 — 慢变量基线模型。

    输入格式为 NCT（batch, channels, timesteps）。
    使用堆叠 Conv1d + AdaptiveAvgPool1d 编码，再经 MLP head 输出四组分预测。
    """

    input_format = "NCT"

    def __init__(
        self,
        in_channels: int = 8,
        out_dim: int = 4,
        hidden_channels: list[int] | None = None,
        kernel_size: int = 5,
        dropout: float = 0.1,
        pooling: str = "mean",
    ):
        super().__init__(out_dim=out_dim)
        hidden_channels = hidden_channels or [32, 64, 64]
        layers: list[nn.Module] = []
        current = in_channels
        for i, hidden in enumerate(hidden_channels):
            k = kernel_size if i < 2 else 3
            layers.extend([
                nn.Conv1d(current, hidden, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
            ])
            if i < 2:
                layers.append(nn.Dropout(dropout))
            current = hidden
        self.encoder = nn.Sequential(*layers)
        self.pool = TemporalPooling(current, mode=pooling)
        self.head = build_regression_head(current, out_dim, dropout)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        encoded = self.encoder(x)
        feats = self.pool(encoded)
        return self.head(feats)
