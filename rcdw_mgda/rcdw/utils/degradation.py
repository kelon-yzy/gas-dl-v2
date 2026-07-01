"""退化模态硬抑制（eval-only）。

框架第十一节：当某模态的中位误差 > ratio × 最小中位误差时，
将其权重压到 cap，然后重归一化。
"""
from __future__ import annotations

import torch


def hard_suppress(
    W: torch.Tensor,
    E_pred: torch.Tensor,
    *,
    ratio: float = 4.0,
    cap: float = 0.04,
) -> tuple[torch.Tensor, torch.Tensor]:
    """退化硬抑制（⚠️ EVAL-ONLY，不可微）。

    逐样本判定：每个样本独立比较各模态误差，不依赖 batch 统计量，
    因此结果与 batch 大小无关（M3 修复）。

    Args:
        W:      (B, M=3, G=3) 融合权重
        E_pred: (B, M=3, G=3) 预测误差
        ratio:  退化判定倍率阈值
        cap:    退化模态最大权重
    Returns:
        W_suppressed: (B, M, G) 抑制后的权重
        degraded:     (B, M, G) bool，标记哪些样本-模态-气体对被抑制
    """
    assert not (W.requires_grad or E_pred.requires_grad), (
        "hard_suppress is eval-only (not differentiable through median/where). "
        "Inputs must not require grad; wrap call site in torch.no_grad() or "
        "detach the tensors before calling."
    )
    # 逐样本：每个气体的最小误差 → (B, 1, G)
    min_E = E_pred.min(dim=1, keepdim=True).values  # (B, 1, G)

    # 退化判定: E_pred[b,m,g] > ratio * min_E[b,g]
    degraded = E_pred > ratio * min_E  # (B, M, G) bool

    # 抑制: 退化模态权重压到 cap
    W_out = W.clone()
    W_out = torch.where(degraded, W_out.clamp(max=cap), W_out)

    # 重归一化 (dim=1 = 模态维)
    W_out = W_out / W_out.sum(dim=1, keepdim=True).clamp(min=1e-6)

    return W_out, degraded
