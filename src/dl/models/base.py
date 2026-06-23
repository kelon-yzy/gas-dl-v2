from __future__ import annotations

import torch
from torch import nn


class BaseRegressor(nn.Module):
    """v4 回归模型基类。

    所有正式模型继承此类，统一 input_format 和 out_dim 语义。
    子类只需实现 ``forward``。
    """

    input_format: str = "NCT"

    def __init__(self, out_dim: int = 4):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
