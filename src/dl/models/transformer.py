from __future__ import annotations

import math

import torch
from torch import nn

from dl.models.base import BaseRegressor
from dl.models.heads import build_regression_head


class TransformerRegressor(BaseRegressor):
    """Transformer encoder regressor for NTC long sequences."""

    input_format = "NTC"

    def __init__(
        self,
        in_channels: int = 8,
        out_dim: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: str = "attention",
        max_timesteps: int = 4096,
    ):
        super().__init__(out_dim=out_dim)
        if pooling not in {"mean", "last", "attention"}:
            raise ValueError("pooling must be one of ['mean', 'last', 'attention']")
        self.pooling = pooling
        self.input_proj = nn.Linear(in_channels, d_model)
        self.register_buffer("positional_encoding", _sinusoidal_positions(max_timesteps, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.attention = nn.Linear(d_model, 1) if pooling == "attention" else None
        self.head = build_regression_head(d_model, out_dim, dropout)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        if x.shape[1] > self.positional_encoding.shape[1]:
            raise ValueError(f"input timesteps {x.shape[1]} exceeds max_timesteps {self.positional_encoding.shape[1]}")
        encoded = self.input_proj(x) + self.positional_encoding[:, : x.shape[1], :]
        encoded = self.encoder(encoded)
        if self.pooling == "last":
            feats = encoded[:, -1, :]
        elif self.pooling == "mean":
            feats = encoded.mean(dim=1)
        else:
            assert self.attention is not None
            weights = torch.softmax(self.attention(encoded), dim=1)
            feats = torch.sum(encoded * weights, dim=1)
        return self.head(feats)


def _sinusoidal_positions(max_timesteps: int, d_model: int) -> torch.Tensor:
    position = torch.arange(max_timesteps, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_timesteps, d_model, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 == 1:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)
