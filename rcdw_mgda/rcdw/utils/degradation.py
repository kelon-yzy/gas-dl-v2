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

    使用 torch.median / torch.where / bool 比较，整体不可微；
    必须在 torch.no_grad() 上下文中调用，禁止用于训练阶段。

    Args:
        W:      (B, M=3, G=3) 融合权重
        E_pred: (B, M=3, G=3) 预测误差
        ratio:  退化判定倍率阈值
        cap:    退化模态最大权重
    Returns:
        W_suppressed: (B, M, G) 抑制后的权重
        degraded:     (M, G)    bool，标记哪些模态-气体对被抑制
    """
    assert not (W.requires_grad or E_pred.requires_grad), (
        "hard_suppress is eval-only (not differentiable through median/where). "
        "Inputs must not require grad; wrap call site in torch.no_grad() or "
        "detach the tensors before calling."
    )
    # 中位误差: 对 batch 维求中位数 → (M, G)
    E_med = E_pred.median(dim=0).values  # (M, G)

    # 每种气体的最小中位误差 → (1, G)
    min_E = E_med.min(dim=0, keepdim=True).values  # (1, G)

    # 退化判定: median(E_m) > ratio * min(E_med)
    degraded = E_med > ratio * min_E  # (M, G) bool

    # 抑制: 退化模态权重压到 cap
    W_out = W.clone()
    degraded_3d = degraded.unsqueeze(0).expand_as(W)  # (B, M, G)
    W_out = torch.where(degraded_3d, W_out.clamp(max=cap), W_out)

    # 重归一化 (dim=1 = 模态维)
    W_out = W_out / W_out.sum(dim=1, keepdim=True).clamp(min=1e-6)

    return W_out, degraded
