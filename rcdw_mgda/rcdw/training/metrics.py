"""评价指标：MAE / RMSE / MRE / MaxRE。"""
from __future__ import annotations

import torch


def compute_metrics(
    pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8
) -> dict[str, float]:
    """计算回归指标。

    Args:
        pred: (N, G=3) 预测浓度
        ref:  (N, G=3) 真值浓度
    Returns:
        {"MAE": ..., "RMSE": ..., "MRE": ..., "MaxRE": ...}
    """
    e = (pred - ref).abs()
    re = e / (ref.abs() + eps)
    return {
        "MAE": e.mean().item(),
        "RMSE": ((pred - ref) ** 2).mean().sqrt().item(),
        "MRE": re.mean().item() * 100.0,
        "MaxRE": re.max().item() * 100.0,
    }


def compute_per_gas_metrics(
    pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8
) -> dict[str, dict[str, float]]:
    """按气体分别计算指标。

    Returns:
        {"O2": {...}, "CO2": {...}, "N2": {...}, "overall": {...}}
    """
    gas_names = ["O2", "CO2", "N2"]
    result = {}
    for g, name in enumerate(gas_names):
        result[name] = compute_metrics(pred[:, g : g + 1], ref[:, g : g + 1], eps)
    result["overall"] = compute_metrics(pred, ref, eps)
    return result
