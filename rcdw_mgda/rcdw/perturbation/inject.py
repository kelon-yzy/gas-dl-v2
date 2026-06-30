"""扰动注入：五类传感器退化/环境突变模拟。"""
from __future__ import annotations

import torch

PERTURBATION_KINDS = [
    "optical_atten",
    "optical_scat",
    "thermal",
    "ultrasonic",
    "temperature",
]


def inject(
    x: torch.Tensor, kind: str, level: float
) -> torch.Tensor:
    """在输入张量上注入指定类型和强度的扰动。

    Args:
        x:     (B, L, 6) 或 (N, L, 6) 滑窗输入
        kind:  扰动类型
        level: 扰动强度，0=无扰动，0.11=最大
    Returns:
        扰动后的张量（原张量不变）
    """
    x = x.clone()

    if kind == "optical_atten":
        # NDIR 通道乘性衰减（光源老化 / 光路污染）
        x[..., 0] *= (1.0 - level)

    elif kind == "optical_scat":
        # NDIR 通道加性高斯噪声（散射干扰）
        x[..., 0] = x[..., 0] + level * torch.randn_like(x[..., 0])

    elif kind == "thermal":
        # 热导通道乘性扰动（温控漂移）
        x[..., 1] = x[..., 1] * (1.0 + level * torch.randn_like(x[..., 1]))

    elif kind == "ultrasonic":
        # 超声通道加性噪声（换能器老化 / 耦合不良）
        scale = x[..., 2].abs().mean()
        x[..., 2] = x[..., 2] + level * scale * torch.randn_like(x[..., 2])

    elif kind == "temperature":
        # 温度阶跃偏移（T 通道, 索引 4）
        x[..., 4] = x[..., 4] + level * 80.0

    else:
        raise ValueError(f"unknown perturbation kind: {kind}. "
                         f"valid: {PERTURBATION_KINDS}")
    return x
