"""数值对齐脚本：验证 RCDWFusion 的 PyTorch 实现与公式手算结果一致。

运行: cd rcdw_mgda && python -m scripts.numerical_check
通过条件: 所有维度 max abs diff < 1e-5
"""
from __future__ import annotations

import torch
import numpy as np
import sys


def hand_compute_rcdw(
    Y_modal: np.ndarray,
    E_pred: np.ndarray,
    W_base: np.ndarray,
    beta: float = 8.0,
    alpha_min: float = 0.1,
    alpha_max: float = 0.9,
    tau_a: float = 0.05,
    s_min: float = 0.05,
    s_max: float = 0.40,
    tau_s: float = 0.05,
) -> dict[str, np.ndarray]:
    """纯 NumPy 手算 RCDW 融合结果。"""
    eps = 1e-6
    B, M, G = Y_modal.shape

    # Wmix: softmax(-beta * E) over modality dim
    logits = -beta * E_pred  # (B, M, G)
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    Wmix = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # alpha and shift
    E_max = E_pred.max(axis=1)  # (B, G)
    E_min = E_pred.min(axis=1)  # (B, G)
    dE = E_max - E_min

    alpha = alpha_min + (alpha_max - alpha_min) * dE / (dE + tau_a)
    shift = s_min + (s_max - s_min) * dE / (dE + tau_s)

    alpha_3d = alpha[:, np.newaxis, :]  # (B, 1, G)
    shift_3d = shift[:, np.newaxis, :]

    # 基线锚定
    W = (1.0 - alpha_3d) * W_base + alpha_3d * Wmix

    # maxShift clamp
    W = np.clip(W, W_base - shift_3d, W_base + shift_3d)

    # 重归一
    W = W / (W.sum(axis=1, keepdims=True) + eps)

    # 融合
    C_fused = (W * Y_modal).sum(axis=1)

    return {"Wmix": Wmix, "alpha": alpha, "shift": shift, "W": W, "C_fused": C_fused}


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    B, M, G = 4, 3, 3

    # 固定输入
    Y_modal_np = np.random.rand(B, M, G).astype(np.float32)
    # 对每个模态归一化（模拟 sum=1 约束）
    Y_modal_np = Y_modal_np / Y_modal_np.sum(axis=2, keepdims=True)

    E_pred_np = np.abs(np.random.randn(B, M, G).astype(np.float32)) * 0.1

    W_base_np = np.array([
        [0.05, 0.70, 0.05],
        [0.50, 0.15, 0.45],
        [0.45, 0.15, 0.50],
    ], dtype=np.float32)

    # NumPy 手算
    expected = hand_compute_rcdw(Y_modal_np, E_pred_np, W_base_np)

    # PyTorch 计算
    from rcdw.models.rcdw import RCDWFusion

    W_base_t = torch.from_numpy(W_base_np)
    fusion = RCDWFusion(W_base_t)
    fusion.eval()

    Y_modal_t = torch.from_numpy(Y_modal_np)
    E_pred_t = torch.from_numpy(E_pred_np)

    with torch.no_grad():
        C_fused_t, W_t = fusion(Y_modal_t, E_pred_t)

    # 对比
    tol = 1e-5
    all_pass = True

    checks = [
        ("W", W_t.numpy(), expected["W"]),
        ("C_fused", C_fused_t.numpy(), expected["C_fused"]),
    ]

    for name, actual, exp in checks:
        diff = np.abs(actual - exp).max()
        status = "PASS" if diff < tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name}: max_abs_diff = {diff:.2e}  [{status}]")

    # 检查权重 sum=1（对每种气体）
    W_sum = W_t.sum(dim=1).numpy()
    sum_diff = np.abs(W_sum - 1.0).max()
    sum_status = "PASS" if sum_diff < tol else "FAIL"
    if sum_status == "FAIL":
        all_pass = False
    print(f"  W_sum=1: max_abs_diff = {sum_diff:.2e}  [{sum_status}]")

    if all_pass:
        print("\n=== ALL CHECKS PASSED ===")
    else:
        print("\n=== SOME CHECKS FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
