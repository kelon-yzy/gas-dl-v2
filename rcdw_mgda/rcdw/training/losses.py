"""损失函数。"""
from __future__ import annotations

import torch
import torch.nn as nn


class WeightedMSE(nn.Module):
    """按气体加权的 MSE。

    Args:
        weights: (G=3,) 各气体的 loss 权重
    """

    def __init__(self, weights: list[float] | None = None):
        super().__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3)
            target: (B, 3)
        """
        mse_per_gas = ((pred - target) ** 2).mean(dim=0)  # (3,)
        return (mse_per_gas * self.w).sum() / self.w.sum()


class StageBLoss(nn.Module):
    """阶段 B 联合损失。

    L = MSE(C_fused, C_ref)
      + lambda_e * MSE(E_pred, E_true)
      + lambda_s * |sum(C_fused) - 1|
    """

    def __init__(self, lambda_e: float = 1.0, lambda_s: float = 0.1):
        super().__init__()
        self.lambda_e = lambda_e
        self.lambda_s = lambda_s

    def forward(
        self,
        C_fused: torch.Tensor,
        C_ref: torch.Tensor,
        E_pred: torch.Tensor,
        Y_modal: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            C_fused: (B, G=3)   融合输出
            C_ref:   (B, G=3)   真值
            E_pred:  (B, M, G)  预测误差
            Y_modal: (B, M, G)  单模态候选（已 detach 或已冻结）
        """
        # 融合 MSE
        loss_fuse = ((C_fused - C_ref) ** 2).mean()

        # 误差监督: E_true = |Y_modal - C_ref|
        C_ref_expand = C_ref.unsqueeze(1).expand_as(Y_modal)  # (B, M, G)
        E_true = (Y_modal - C_ref_expand).abs()
        loss_error = ((E_pred - E_true) ** 2).mean()

        # 组分约束
        loss_sum = (C_fused.sum(dim=-1) - 1.0).abs().mean()

        total = loss_fuse + self.lambda_e * loss_error + self.lambda_s * loss_sum

        details = {
            "loss_total": total.item(),
            "loss_fuse": loss_fuse.item(),
            "loss_error": loss_error.item(),
            "loss_sum": loss_sum.item(),
        }
        return total, details
