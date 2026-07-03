"""评价指标：MAE / RMSE / MRE / MaxRE。"""
from __future__ import annotations

import torch

from rcdw.sim.core.schema import GAS_DISPLAY_NAMES


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
    pred: torch.Tensor,
    ref: torch.Tensor,
    eps: float = 1e-8,
    gas_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """按气体分别计算指标。

    Returns:
        {"O2": {...}, "CO2": {...}, "N2": {...}, "overall": {...}}
    """
    gas_names = list(gas_names or GAS_DISPLAY_NAMES)
    if pred.shape[1] != len(gas_names) or ref.shape[1] != len(gas_names):
        raise ValueError(
            f"gas_names length {len(gas_names)} does not match pred/ref shapes "
            f"{tuple(pred.shape)} / {tuple(ref.shape)}"
        )
    result = {}
    for g, name in enumerate(gas_names):
        result[name] = compute_metrics(pred[:, g : g + 1], ref[:, g : g + 1], eps)
    result["overall"] = compute_metrics(pred, ref, eps)
    return result
