"""组分归一化与湿/干基换算。"""
from __future__ import annotations

import torch


def rh_to_water_vol(RH: torch.Tensor, T: torch.Tensor | None = None) -> torch.Tensor:
    """将相对湿度（0–1 小数）近似转换为水汽体积分数。

    简化模型：假设 T=300K, P=1atm 下饱和水汽压 ~3.6 kPa。
    C_H2O ≈ RH * 3.6 / 101.325 ≈ RH * 0.0355

    ⚠️ 调用方必须传入 0–1 范围的 RH（非百分数）。管线中 ``H_RH``
    字段为百分数 20–80，需先 ``/100`` 再传入；否则结果溢出（如
    ``rh_to_water_vol(80) = 2.84``，导致 wet basis 总量为负）。

    ⚠️ T 参数当前不参与计算，硬编码 300K 假设。实际饱和水汽压
    随 T 强非线性变化（260K ~0.2 kPa → 340K ~27 kPa）。
    切换到 wet basis 前需补全 T 依赖（Magnus / Antoine 公式）。
    """
    if isinstance(RH, torch.Tensor) and (RH > 1.0).any():
        raise ValueError(
            f"RH 应为 0–1 小数，但检测到值 > 1（max={RH.max().item():.2f}）。"
            "若 RH 为百分数（如 H_RH），请先除以 100。"
        )
    return RH * 0.0355


def normalize_composition(
    C: torch.Tensor,
    RH: torch.Tensor | None = None,
    *,
    basis: str = "dry",
) -> torch.Tensor:
    """对 O₂/CO₂/N₂ 浓度归一化。

    Args:
        C:     (B, 3) 或 (N, 3) 浓度
        RH:    (B,) 相对湿度（仅 basis="wet" 时使用）
        basis: "dry" → sum=1.0; "wet" → sum=1.0-C_H2O
    Returns:
        归一化后的浓度，与 C 同 shape
    """
    eps = 1e-6
    if basis == "wet" and RH is not None:
        C_total = 1.0 - rh_to_water_vol(RH)
        if C_total.dim() == 1:
            C_total = C_total.unsqueeze(-1)  # (B, 1)
    else:
        C_total = 1.0
    return C / (C.sum(dim=-1, keepdim=True) + eps) * C_total
