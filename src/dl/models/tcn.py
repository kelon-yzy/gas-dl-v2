from __future__ import annotations

import torch
from torch import nn

from dl.models.base import BaseRegressor


class CausalConv1d(nn.Module):
    """保持序列长度的因果 Conv1d。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, bias: bool = True):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding == 0:
            return out
        return out[:, :, :-self.padding]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if in_channels == out_channels:
            self.proj: nn.Module = nn.Identity()
        else:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.proj(x))


class TCNRegressor(BaseRegressor):
    """Temporal Convolutional Network 回归器。

    输入格式为 NCT（batch, channels, timesteps）。每个 block 含两层因果卷积，
    dilation 按 1, 2, 4... 递增，``receptive_field`` 记录最终时间感受野长度。
    """

    input_format = "NCT"

    def __init__(
        self,
        in_channels: int = 8,
        out_dim: int = 4,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(out_dim=out_dim)
        channels = channels or [32, 64, 64]
        self.kernel_size = kernel_size
        self.dilations = tuple(2**i for i in range(len(channels)))
        self.receptive_field = self._compute_receptive_field(kernel_size, self.dilations)

        layers: list[nn.Module] = []
        current = in_channels
        for hidden, dilation in zip(channels, self.dilations, strict=True):
            layers.append(TemporalBlock(current, hidden, kernel_size, dilation=dilation, dropout=dropout))
            current = hidden

        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(current, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        encoded = self.encoder(x)
        feats = self.pool(encoded).flatten(1)
        return self.head(feats)

    @staticmethod
    def _compute_receptive_field(kernel_size: int, dilations: tuple[int, ...]) -> int:
        return 1 + sum(2 * (kernel_size - 1) * dilation for dilation in dilations)
