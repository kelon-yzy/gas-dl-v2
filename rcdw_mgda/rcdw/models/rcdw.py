"""RCDW 融合层 + 整体 RCDW_MGDA 模型。"""
from __future__ import annotations

import torch
import torch.nn as nn

from rcdw.models.single_modal import NDIRNet, TCDNet, USNet, extract_modal_input
from rcdw.models.feature import FeatureExtractor
from rcdw.models.error_net import ErrorNet


class RCDWFusion(nn.Module):
    """可靠性约束动态加权融合层（可微，无可学习参数）。

    维度约定:
      W_base[m, g]: 模态 m 对气体 g 的基线权重
      softmax 在 dim=1 (模态维) 归一化
      即对每种气体 g，各模态权重 sum=1
    """

    def __init__(
        self,
        W_base: torch.Tensor,
        *,
        beta: float = 8.0,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
        tau_a: float = 0.05,
        s_min: float = 0.05,
        s_max: float = 0.40,
        tau_s: float = 0.05,
    ):
        super().__init__()
        assert W_base.shape == (3, 3), f"W_base shape {W_base.shape} != (M=3, G=3)"
        self.register_buffer("W_base", W_base.clone())
        self.beta = beta
        self.a_min = alpha_min
        self.a_max = alpha_max
        self.tau_a = tau_a
        self.s_min = s_min
        self.s_max = s_max
        self.tau_s = tau_s

    def forward(
        self, Y_modal: torch.Tensor, E_pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Y_modal: (B, M=3, G=3)  单模态候选浓度
            E_pred:  (B, M=3, G=3)  预测误差（恒正）
        Returns:
            C_fused: (B, G=3)
            W_final: (B, M=3, G=3)
        """
        eps = 1e-6

        # Step 1: softmax Wmix — 误差小的模态权重大
        # dim=1 是模态维，对每种气体归一化
        Wmix = torch.softmax(-self.beta * E_pred, dim=1)  # (B, M, G)

        # Step 2: 自适应 alpha 和 shift
        # dE: 模态间误差差异 (B, G)
        E_max = E_pred.max(dim=1).values  # (B, G)
        E_min = E_pred.min(dim=1).values  # (B, G)
        dE = E_max - E_min  # (B, G)

        alpha = self.a_min + (self.a_max - self.a_min) * dE / (dE + self.tau_a)  # (B, G)
        shift = self.s_min + (self.s_max - self.s_min) * dE / (dE + self.tau_s)  # (B, G)

        # 扩展到 (B, 1, G) 以广播
        alpha = alpha.unsqueeze(1)  # (B, 1, G)
        shift = shift.unsqueeze(1)  # (B, 1, G)

        # Step 3: 基线锚定
        # W_base: (M, G) 广播到 (B, M, G)
        W = (1.0 - alpha) * self.W_base + alpha * Wmix  # (B, M, G)

        # Step 4: maxShift 约束
        W = torch.clamp(W, self.W_base - shift, self.W_base + shift)

        # Step 5: 重归一化（对每种气体，各模态权重 sum=1）
        W = W / (W.sum(dim=1, keepdim=True) + eps)

        # Step 6: 加权融合
        C_fused = (W * Y_modal).sum(dim=1)  # (B, G)

        return C_fused, W


class RCDW_MGDA(nn.Module):
    """完整的 RCDW-MGDA 模型。

    输入: (B, L=8, 12)  — v1.2 §8.2 12 维通道布局
    输出: dict with C, Y_modal, E_pred, W
    """

    def __init__(
        self,
        W_base: torch.Tensor,
        hidden: int = 32,
        window: int = 8,
        fusion_kwargs: dict | None = None,
    ):
        """构造 RCDW-MGDA 模型。

        Args:
            W_base: (M=3, G=3) 基线权重，每列 sum=1.0
            hidden: 单模态网络与 ErrorNet 的隐藏维
            window: 滑窗长度 L（FeatureExtractor 内部使用）
            fusion_kwargs: 透传给 RCDWFusion 的超参（beta/alpha_min/alpha_max/
                tau_a/s_min/s_max/tau_s）。None 时使用 RCDWFusion 默认值。
        """
        super().__init__()
        self.ndir = NDIRNet(in_dim=4, hidden=hidden)
        self.tcd = TCDNet(in_dim=4, hidden=hidden)
        self.usn = USNet(in_dim=4, hidden=hidden)
        self.feat = FeatureExtractor(window=window)
        self.err = ErrorNet(in_dim=13, n_gas=3, hidden=hidden)
        self.fuse = RCDWFusion(W_base, **(fusion_kwargs or {}))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, L=8, 12) 滑窗输入（v1.2 12 维布局）
        Returns:
            {"C": (B,3), "Y_modal": (B,3,3), "E_pred": (B,3,3), "W": (B,3,3)}
        """
        # 单模态反演：取窗口最后时刻 (B, 12)
        x_last = x[:, -1, :]

        y_nd = self.ndir(extract_modal_input(x_last, "ndir"))  # (B, 3)
        y_tc = self.tcd(extract_modal_input(x_last, "tcd"))    # (B, 3)
        y_us = self.usn(extract_modal_input(x_last, "usn"))    # (B, 3)
        Y_modal = torch.stack([y_nd, y_tc, y_us], dim=1)       # (B, M=3, G=3)

        # 特征提取：用完整滑窗
        feat = self.feat(x, Y_modal)  # (B, M=3, F=13)

        # 误差预测
        E_pred = self.err(feat)  # (B, M=3, G=3)

        # RCDW 融合
        C_fused, W = self.fuse(Y_modal, E_pred)

        return {"C": C_fused, "Y_modal": Y_modal, "E_pred": E_pred, "W": W}
