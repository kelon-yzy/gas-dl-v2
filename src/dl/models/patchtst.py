from __future__ import annotations

import torch
from torch import nn

from dl.models.base import BaseRegressor
from dl.models.heads import build_regression_head


class PatchTSTRegressor(BaseRegressor):
    """Patch-based transformer regressor for NTC time series."""

    input_format = "NTC"

    def __init__(
        self,
        in_channels: int = 8,
        out_dim: int = 4,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        pooling: str = "attention",
    ):
        super().__init__(out_dim=out_dim)
        if patch_len < 1:
            raise ValueError("patch_len must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if pooling not in {"mean", "last", "attention"}:
            raise ValueError("pooling must be one of ['mean', 'last', 'attention']")
        self.patch_len = patch_len
        self.stride = stride
        self.pooling = pooling
        self.patch_proj = nn.Linear(in_channels * patch_len, d_model)
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
        patches = self._patchify(x)
        encoded = self.encoder(self.patch_proj(patches))
        if self.pooling == "last":
            feats = encoded[:, -1, :]
        elif self.pooling == "mean":
            feats = encoded.mean(dim=1)
        else:
            assert self.attention is not None
            weights = torch.softmax(self.attention(encoded), dim=1)
            feats = torch.sum(encoded * weights, dim=1)
        return self.head(feats)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be NTC, got ndim={x.ndim}")
        if x.shape[1] < self.patch_len:
            pad = self.patch_len - x.shape[1]
            x = torch.nn.functional.pad(x, (0, 0, 0, pad), mode="replicate")
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.flatten(start_dim=2)
