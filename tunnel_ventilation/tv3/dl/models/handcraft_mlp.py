from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from tv3.dl.models.base import BaseRegressor
from tv3.dl.models.cnn1d_tcn_fusion import GasHeadNormalize


class HandcraftMLPRegressor(BaseRegressor):
    """Small gas-head MLP for ML hand-crafted feature matrices."""

    input_format = "FEATURES"

    def __init__(
        self,
        in_channels: int,
        out_dim: int = 4,
        hidden_dims: Sequence[int] = (128, 64),
        dropout: float = 0.25,
        output_prior: Sequence[float] = (9.288469, 75.755157, 4.994778, 9.961745),
    ):
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if out_dim != 4:
            raise ValueError("HandcraftMLPRegressor requires out_dim=4 for gas_head percentages")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if dropout < 0.0:
            raise ValueError("dropout must be >= 0")
        super().__init__(out_dim=out_dim)
        self.in_channels = in_channels

        layers: list[nn.Module] = []
        current = in_channels
        for hidden in hidden_dims:
            if int(hidden) < 1:
                raise ValueError("hidden_dims entries must be >= 1")
            layers.append(nn.Linear(current, int(hidden)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current = int(hidden)
        self.feature_mlp = nn.Sequential(*layers)
        self.gas_head = GasHeadNormalize(current, output_prior=output_prior)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must be shaped (B, features), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input features, got {x.shape[1]}")
        return self.gas_head(self.feature_mlp(x.float()))
