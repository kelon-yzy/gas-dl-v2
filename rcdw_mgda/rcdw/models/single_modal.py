"""单模态浓度反演网络。

每个模态独立估计三气体浓度。
使用 clamp(min=0) + L1-normalize 保证 sum=1 且非负。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SingleModal(nn.Module):
    """基类：传感器信号(1) + 环境(3) → 三气体浓度(3)。"""

    def __init__(self, in_dim: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4) = [sensor_value, P, T, RH]
        Returns:
            (B, 3) = [C_O2, C_CO2, C_N2], sum≈1, 非负
        """
        raw = self.net(x).clamp(min=0.0)
        return raw / (raw.sum(dim=-1, keepdim=True) + 1e-6)


class NDIRNet(SingleModal):
    """NDIR 模态。输入: [S_ndir, P, T, RH]。"""
    pass


class TCDNet(SingleModal):
    """热导 TCD 模态。输入: [S_tc, P, T, RH]。"""
    pass


class USNet(SingleModal):
    """超声 US 模态。输入: [S_us, P, T, RH]。"""
    pass


# ---- 辅助函数 ----

# 通道索引常量（对应 x 的 dim=-1）
SENSOR_INDICES = {"ndir": 0, "tcd": 1, "us": 2}
ENV_INDICES = [3, 4, 5]  # P, T, RH


def extract_modal_input(x_last: torch.Tensor, modality: str) -> torch.Tensor:
    """从最后时刻 (B, 6) 提取特定模态的 (B, 4) 输入。

    Args:
        x_last: (B, 6) = [S_ndir, S_tc, S_us, P, T, RH]
        modality: "ndir" | "tcd" | "us"
    Returns:
        (B, 4) = [sensor_value, P, T, RH]
    """
    sensor_idx = SENSOR_INDICES[modality]
    sensor_val = x_last[:, sensor_idx : sensor_idx + 1]  # (B, 1)
    env = x_last[:, ENV_INDICES]  # (B, 3)
    return torch.cat([sensor_val, env], dim=-1)  # (B, 4)
