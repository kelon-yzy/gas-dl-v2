from __future__ import annotations

import torch
from torch import nn

from dl.models.base import BaseRegressor
from dl.models.heads import build_regression_head


class LSTMRegressor(BaseRegressor):
    """LSTM sequence regressor for NTC inputs."""

    input_format = "NTC"

    def __init__(
        self,
        in_channels: int = 8,
        out_dim: int = 4,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
        pooling: str = "last",
    ):
        super().__init__(out_dim=out_dim)
        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be one of ['last', 'mean']")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.pooling = pooling
        self.encoder = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        encoded_dim = hidden_size * (2 if bidirectional else 1)
        self.head = build_regression_head(encoded_dim, out_dim, dropout)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        encoded, _state = self.encoder(x)
        if self.pooling == "last":
            feats = encoded[:, -1, :]
        else:
            feats = encoded.mean(dim=1)
        return self.head(feats)
