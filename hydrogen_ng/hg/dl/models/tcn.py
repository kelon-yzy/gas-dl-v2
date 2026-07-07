from __future__ import annotations

import torch
from torch import nn

from hg.dl.models.base import BaseRegressor
from hg.dl.models.heads import TemporalPooling, build_regression_head


DEFAULT_TCN_CHANNELS = [32, 64, 64]


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
        target_timesteps: int | None = None,
        pooling: str = "mean",
    ):
        super().__init__(out_dim=out_dim)
        if target_timesteps is not None and target_timesteps < 1:
            raise ValueError("target_timesteps must be >= 1")
        if kernel_size < 2:
            raise ValueError("kernel_size must be >= 2")
        if channels is None:
            channels = tcn_channels_for_timesteps(target_timesteps, kernel_size=kernel_size)
        if len(channels) == 0:
            raise ValueError("channels must contain at least one block")
        self.kernel_size = kernel_size
        self.dilations = tuple(2**i for i in range(len(channels)))
        self.receptive_field = self._compute_receptive_field(kernel_size, self.dilations)
        if target_timesteps is not None and self.receptive_field < target_timesteps:
            raise ValueError(
                f"TCN receptive_field={self.receptive_field} is smaller than target_timesteps={target_timesteps}; "
                "increase channels/block count or omit target_timesteps for the legacy short-sequence model"
            )

        layers: list[nn.Module] = []
        current = in_channels
        for hidden, dilation in zip(channels, self.dilations, strict=True):
            layers.append(TemporalBlock(current, hidden, kernel_size, dilation=dilation, dropout=dropout))
            current = hidden

        self.encoder = nn.Sequential(*layers)
        self.pool = TemporalPooling(current, mode=pooling)
        self.head = build_regression_head(current, out_dim, dropout)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        encoded = self.encoder(x)
        feats = self.pool(encoded)
        return self.head(feats)

    @staticmethod
    def _compute_receptive_field(kernel_size: int, dilations: tuple[int, ...]) -> int:
        return 1 + sum(2 * (kernel_size - 1) * dilation for dilation in dilations)


def tcn_channels_for_timesteps(
    target_timesteps: int | None,
    *,
    kernel_size: int = 3,
    base_channels: tuple[int, ...] = (32, 64),
    max_channels: int = 64,
) -> list[int]:
    if target_timesteps is None:
        return list(DEFAULT_TCN_CHANNELS)
    if target_timesteps < 1:
        raise ValueError("target_timesteps must be >= 1")
    channels: list[int] = []
    dilations: list[int] = []
    while len(channels) == 0 or TCNRegressor._compute_receptive_field(kernel_size, tuple(dilations)) < target_timesteps:
        block_index = len(channels)
        channels.append(base_channels[block_index] if block_index < len(base_channels) else max_channels)
        dilations.append(2**block_index)
    return channels
