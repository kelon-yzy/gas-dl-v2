"""13 维扰动感知特征提取器。

输入始终为滑窗 (B, L=8, 6)。
输出 (B, M=3, F=13)。

特征列表:
  0  CV_m        滑窗变异系数 std/mean
  1  D_m         群体中位偏离 (跨气体平均)
  2  G_m         一阶差分能量 mean((S_k - S_{k-1})^2)
  3  Q_m         信号质量比 snr_m / sum(snr)
  4  B_m         群体偏差 (跨气体平均)
  5  delta_T     |T_k - T_{k-1}|
  6  delta_P     |P_k - P_{k-1}|
  7  delta_RH    |RH_k - RH_{k-1}|
  8  dev_max     |Y_m - mean(Y)| 跨气体 max
  9  dev_mean    |Y_m - mean(Y)| 跨气体 mean
  10 snr_proxy   |mu| / sigma
  11 drift       滑窗线性拟合斜率
  12 dt          固定采样间隔 = 1.0
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """扰动感知特征提取（纯计算，无可学习参数）。"""

    def __init__(self, window: int = 8):
        super().__init__()
        self.L = window
        # 预计算线性拟合用的时间轴
        t = torch.arange(window, dtype=torch.float32)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum()
        self.register_buffer("_t_centered", t - t_mean)  # (L,)
        self.register_buffer("_t_var", t_var.clone().detach())

    def forward(
        self, x: torch.Tensor, Y_modal: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, L, 6)   滑窗传感器数据
            Y_modal: (B, M=3, G=3) 单模态浓度候选
        Returns:
            (B, M=3, F=13) 扰动感知特征
        """
        B, L, _ = x.shape
        M, G = 3, 3
        eps = 1e-8
        device = x.device
        dtype = x.dtype

        # 传感器信号: (B, L, 3) = S_ndir, S_tc, S_us
        S = x[:, :, :3]
        # 环境: (B, L, 3) = P, T, RH
        env = x[:, :, 3:]

        feats = torch.zeros(B, M, 13, device=device, dtype=dtype)

        # --- 环境变化率（所有模态共享） ---
        # delta_T, delta_P, delta_RH: 最后两步之差
        delta_T = (env[:, -1, 1] - env[:, -2, 1]).abs()    # (B,)  T=env[:,1]
        delta_P = (env[:, -1, 0] - env[:, -2, 0]).abs()    # (B,)  P=env[:,0]
        delta_RH = (env[:, -1, 2] - env[:, -2, 2]).abs()   # (B,)  RH=env[:,2]

        # --- 跨模态统计 ---
        Y_median = Y_modal.median(dim=1).values  # (B, G)
        Y_mean = Y_modal.mean(dim=1)              # (B, G)

        # 信号质量比 SNR 分母（所有模态的 SNR 之和）
        snr_all = torch.zeros(B, M, device=device, dtype=dtype)
        for m in range(M):
            s_m = S[:, :, m]  # (B, L)
            mu_m = s_m.mean(dim=1)
            sigma_m = s_m.std(dim=1)
            snr_all[:, m] = mu_m.abs() / (sigma_m + eps)
        snr_total = snr_all.sum(dim=1, keepdim=True) + eps  # (B, 1)

        # --- 逐模态计算 ---
        for m in range(M):
            s_m = S[:, :, m]  # (B, L)
            Y_m = Y_modal[:, m, :]  # (B, G)

            # 0: CV_m = std / |mean|
            mu = s_m.mean(dim=1)         # (B,)
            sigma = s_m.std(dim=1)       # (B,)
            feats[:, m, 0] = sigma / (mu.abs() + eps)

            # 1: D_m = |Y_m - median(Y)|, 跨气体平均
            feats[:, m, 1] = (Y_m - Y_median).abs().mean(dim=-1)

            # 2: G_m = gradient energy
            diffs_sq = (s_m[:, 1:] - s_m[:, :-1]) ** 2  # (B, L-1)
            feats[:, m, 2] = diffs_sq.mean(dim=1)

            # 3: Q_m = snr_m / sum(snr)
            feats[:, m, 3] = snr_all[:, m] / snr_total.squeeze(1)

            # 4: B_m = group bias, 跨气体平均
            bias_sum = torch.zeros(B, device=device, dtype=dtype)
            for j in range(M):
                if j != m:
                    bias_sum += (Y_m - Y_modal[:, j, :]).abs().mean(dim=-1)
            feats[:, m, 4] = bias_sum / (M - 1)

            # 5,6,7: 环境变化率
            feats[:, m, 5] = delta_T
            feats[:, m, 6] = delta_P
            feats[:, m, 7] = delta_RH

            # 8: dev_max = |Y_m - mean(Y)| 跨气体 max
            feats[:, m, 8] = (Y_m - Y_mean).abs().max(dim=-1).values

            # 9: dev_mean = |Y_m - mean(Y)| 跨气体 mean
            feats[:, m, 9] = (Y_m - Y_mean).abs().mean(dim=-1)

            # 10: snr_proxy = |mu| / sigma
            feats[:, m, 10] = mu.abs() / (sigma + eps)

            # 11: drift = 线性拟合斜率
            s_centered = s_m - s_m.mean(dim=1, keepdim=True)  # (B, L)
            # slope = sum(t_centered * s_centered) / t_var
            slope = (self._t_centered.unsqueeze(0) * s_centered).sum(dim=1) / (self._t_var + eps)
            feats[:, m, 11] = slope

            # 12: dt = 固定常数
            feats[:, m, 12] = 1.0

        return feats
