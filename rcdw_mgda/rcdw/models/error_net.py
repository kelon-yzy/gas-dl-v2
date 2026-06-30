"""误差预测器 ErrorNet。

每个气体一个独立 head，输入 13 维特征，输出预测误差（恒正）。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ErrorNet(nn.Module):
    """误差预测网络，3 个独立 head 分别预测 O₂/CO₂/N₂ 的模态误差。

    输入: (B, M=3, F=13) 扰动特征
    输出: (B, M=3, G=3)  预测误差，恒正（Softplus 保证）
    """

    def __init__(self, in_dim: int = 13, n_gas: int = 3, hidden: int = 32):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
                nn.Softplus(),
            )
            for _ in range(n_gas)
        ])

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, M=3, F=13)
        Returns:
            E_pred: (B, M=3, G=3)  恒正
        """
        outs = []
        for head in self.heads:
            out = head(feat).squeeze(-1)  # (B, M)
            outs.append(out)
        return torch.stack(outs, dim=-1)  # (B, M, G)
