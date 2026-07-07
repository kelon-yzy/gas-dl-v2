from __future__ import annotations

import torch

from sg.common.metrics import R2_ZERO_VARIANCE_EPSILON, RegressionMetrics
from sg.sim.core.schema import COMPONENT_FIELDS


def regression_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> RegressionMetrics:
    """计算整体 MAE、RMSE 与 pooled R2。

    R2 使用按输出通道去均值后的总平方和，不是逐组分 R2 的宏平均。
    """
    _validate_prediction_shapes(y_pred, y_true)
    pred = y_pred.detach().float()
    true = y_true.detach().float()
    err = pred - true
    mae = err.abs().mean()
    rmse = torch.sqrt(torch.mean(err.square()))

    ss_res = torch.sum(err.square())
    centered = true - torch.mean(true, dim=0, keepdim=True)
    ss_tot = torch.sum(centered.square())
    if ss_tot.item() < R2_ZERO_VARIANCE_EPSILON:
        r2 = torch.tensor(1.0 if ss_res.item() < R2_ZERO_VARIANCE_EPSILON else 0.0, dtype=true.dtype, device=true.device)
    else:
        r2 = 1.0 - ss_res / ss_tot
    return RegressionMetrics(mae=float(mae.item()), rmse=float(rmse.item()), r2=float(r2.item()))


def component_regression_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    component_names: tuple[str, ...] = COMPONENT_FIELDS,
) -> dict[str, RegressionMetrics]:
    """按组分分别计算回归指标。"""
    _validate_prediction_shapes(y_pred, y_true)
    if y_pred.ndim != 2:
        raise ValueError(f"Expected 2D tensors shaped (N, C), got {tuple(y_pred.shape)}")
    if y_pred.shape[1] != len(component_names):
        raise ValueError(
            f"component_names length {len(component_names)} does not match prediction channels {y_pred.shape[1]}"
        )
    return {
        name: regression_metrics(y_pred[:, idx:idx + 1], y_true[:, idx:idx + 1])
        for idx, name in enumerate(component_names)
    }


def _validate_prediction_shapes(y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Prediction and target shapes must match, got {tuple(y_pred.shape)} and {tuple(y_true.shape)}")
    if y_pred.numel() == 0:
        raise ValueError("Prediction and target tensors must not be empty")
