"""单模态浓度反演网络。

每个模态独立估计三气体浓度（O₂/CO₂/N₂）。
使用 clamp(min=0) + L1-normalize 保证 sum=1 且非负。

输入张量布局（v1.2 §8.2 12 维新布局）：
    (B, L, 12) = [
        V_NDIR_CO2,                     # idx 0
        V_TCS,                          # idx 1
        T_C,                            # idx 2 (环境)
        P_MPa,                          # idx 3 (环境)
        H_RH,                           # idx 4 (环境)
        L_m,                            # idx 5 (预留)
        piston_position_m,              # idx 6 (预留)
        ultrasonic_tof_observed_s,      # idx 7 (预留)
        ultrasonic_sound_speed_estimated_m_per_s,  # idx 8 (US 主信号)
        ultrasonic_peak_index,          # idx 9 (预留)
        ultrasonic_tof_quality,         # idx 10 (预留, ErrorNet 未来直读)
        ultrasonic_tof_accepted,        # idx 11 (预留, ErrorNet 未来直读)
    ]

每个单模态分支取 (sensor_value, T_C, P_MPa, H_RH) = 4 维输入。
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---- 通道索引常量（v1.2 §8.2）----

# Slow 通道（与 SLOW_CHANNELS 一一对应）
IDX_NDIR_CO2 = 0        # V_NDIR_CO2
IDX_TCS = 1             # V_TCS
IDX_T_C = 2             # T_C
IDX_P_MPa = 3           # P_MPa
IDX_H_RH = 4            # H_RH
IDX_L_m = 5             # L_m（预留）
IDX_PISTON = 6          # piston_position_m（预留）

# Ultrasonic 元数据
IDX_US_TOF = 7          # ultrasonic_tof_observed_s（预留）
IDX_US_SPEED = 8        # ultrasonic_sound_speed_estimated_m_per_s（US 主信号）
IDX_US_PEAK = 9         # ultrasonic_peak_index（预留）
IDX_US_QUALITY = 10     # ultrasonic_tof_quality（预留）
IDX_US_ACCEPTED = 11    # ultrasonic_tof_accepted（预留）

# 各模态主传感器索引（NDIR/TCD/USN 对应 W_base 行顺序）
SENSOR_INDICES = {"ndir": IDX_NDIR_CO2, "tcd": IDX_TCS, "usn": IDX_US_SPEED}

# 环境上下文索引（顺序: T_C, P_MPa, H_RH，与旧版 [P, T, RH] 不同）
ENV_INDICES = [IDX_T_C, IDX_P_MPa, IDX_H_RH]  # [2, 3, 4]

# 总通道维度
INPUT_CHANNELS = 12


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
            x: (B, 4) = [sensor_value, T_C, P_MPa, H_RH]
        Returns:
            (B, 3) = [C_O2, C_CO2, C_N2], sum≈1, 非负
        """
        raw = self.net(x).clamp(min=0.0)
        return raw / (raw.sum(dim=-1, keepdim=True) + 1e-6)


class NDIRNet(SingleModal):
    """NDIR 模态。输入: [V_NDIR_CO2, T_C, P_MPa, H_RH]。"""
    pass


class TCDNet(SingleModal):
    """热导 TCD 模态。输入: [V_TCS, T_C, P_MPa, H_RH]。"""
    pass


class USNet(SingleModal):
    """超声 US 模态。输入: [ultrasonic_sound_speed_estimated, T_C, P_MPa, H_RH]。"""
    pass


def extract_modal_input(x_last: torch.Tensor, modality: str) -> torch.Tensor:
    """从窗口最后时刻 (B, 12) 提取特定模态的 (B, 4) 输入。

    Args:
        x_last: (B, 12) 12 维通道布局（见模块 docstring）
        modality: "ndir" | "tcd" | "usn"
    Returns:
        (B, 4) = [sensor_value, T_C, P_MPa, H_RH]
    """
    if modality not in SENSOR_INDICES:
        raise ValueError(
            f"unknown modality: {modality!r}; valid: {list(SENSOR_INDICES)}"
        )
    sensor_idx = SENSOR_INDICES[modality]
    sensor_val = x_last[:, sensor_idx : sensor_idx + 1]  # (B, 1)
    env = x_last[:, ENV_INDICES]  # (B, 3) = [T_C, P_MPa, H_RH]
    return torch.cat([sensor_val, env], dim=-1)  # (B, 4)
