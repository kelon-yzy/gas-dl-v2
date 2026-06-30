"""合成时序数据 + 滑窗切分。

生成平滑变化的三组分浓度时序，模拟真实标定过程中的连续采样。
输出滑窗张量 (N_windows, L, 6) 与标签 (N_windows, 3)。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def synth_timeseries(n_steps: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """生成 n_steps 个时间步的传感器读数与浓度真值。

    浓度沿时间轴平滑变化（线性插值 + 小噪声），保证滑窗内的
    CV / gradient / drift 等统计特征有非零值。

    Returns:
        X: (n_steps, 6)  [S_ndir, S_tc, S_us, P, T, RH]
        Y: (n_steps, 3)  [C_O2, C_CO2, C_N2], sum≈1
    """
    rng = np.random.default_rng(seed)

    # --- 平滑浓度轨迹 ---
    n_anchors = max(n_steps // 100, 5)
    anchors = rng.dirichlet([2.0, 1.0, 6.0], size=n_anchors + 1)
    seg_len = n_steps // n_anchors

    C = np.zeros((n_steps, 3), dtype=np.float64)
    for i in range(n_anchors):
        s = i * seg_len
        e = min((i + 1) * seg_len, n_steps)
        t = np.linspace(0.0, 1.0, e - s)
        for j in range(3):
            C[s:e, j] = (1.0 - t) * anchors[i, j] + t * anchors[i + 1, j]
    if n_anchors * seg_len < n_steps:
        C[n_anchors * seg_len :] = anchors[-1]

    C += 0.005 * rng.standard_normal(C.shape)
    C = np.clip(C, 0.01, None)
    C = C / C.sum(axis=1, keepdims=True)

    # --- 环境参数（缓慢漂移 + 小噪声） ---
    t_norm = np.linspace(0.0, 1.0, n_steps)
    T = 300.0 + 40.0 * np.sin(2.0 * np.pi * t_norm * 3.0) + 3.0 * rng.standard_normal(n_steps)
    P = 1.0 + 0.03 * np.sin(2.0 * np.pi * t_norm * 5.0) + 0.003 * rng.standard_normal(n_steps)
    RH = 0.025 + 0.015 * np.sin(2.0 * np.pi * t_norm * 2.0)
    RH = np.clip(RH + 0.002 * rng.standard_normal(n_steps), 0.0, 0.05)

    # --- 传感器信号（含物理近似） ---
    # NDIR: Beer-Lambert 对 CO2
    S_ndir = (1.0 - np.exp(-3.0 * C[:, 1])) + 0.01 * rng.standard_normal(n_steps)
    # 超声: v = sqrt(γRT / M_mix)
    M_mix = 32.0 * C[:, 0] + 44.0 * C[:, 1] + 28.0 * C[:, 2]
    S_us = np.sqrt(1.4 * 8.314 * T / (M_mix * 1e-3)) + 0.5 * rng.standard_normal(n_steps)
    # 热导: λ_mix ≈ Σ x_i λ_i
    S_tc = (0.026 * C[:, 0] + 0.017 * C[:, 1] + 0.026 * C[:, 2]
            + 1e-4 * rng.standard_normal(n_steps))

    X = np.stack([S_ndir, S_tc, S_us, P, T, RH], axis=1).astype(np.float32)
    Y = C.astype(np.float32)
    return X, Y


def make_windows(X: np.ndarray, Y: np.ndarray, L: int = 8
                 ) -> tuple[np.ndarray, np.ndarray]:
    """将 (N, 6) 时序切分为 (N-L+1, L, 6) 滑窗，标签取窗口最后时刻。"""
    N = len(X)
    assert N >= L, f"时序长度 {N} < 窗口 {L}"
    n_windows = N - L + 1
    X_w = np.zeros((n_windows, L, 6), dtype=np.float32)
    Y_w = np.zeros((n_windows, 3), dtype=np.float32)
    for i in range(n_windows):
        X_w[i] = X[i : i + L]
        Y_w[i] = Y[i + L - 1]
    return X_w, Y_w


def make_splits(
    n_train: int = 1400,
    n_val: int = 300,
    n_test: int = 300,
    L: int = 8,
    seed: int = 42,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """生成训练/验证/测试滑窗数据。

    Returns:
        {"train": (X_w, Y_w), "val": ..., "test": ...}
        X_w: (N, L, 6), Y_w: (N, 3)
    """
    n_total_windows = n_train + n_val + n_test
    n_raw = n_total_windows + L - 1
    X_raw, Y_raw = synth_timeseries(n_raw, seed=seed)
    X_w, Y_w = make_windows(X_raw, Y_raw, L=L)
    assert len(X_w) == n_total_windows

    s1 = n_train
    s2 = n_train + n_val
    return {
        "train": (X_w[:s1], Y_w[:s1]),
        "val": (X_w[s1:s2], Y_w[s1:s2]),
        "test": (X_w[s2:], Y_w[s2:]),
    }


class WindowedDataset(Dataset):
    """PyTorch Dataset，包装滑窗数据。"""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        assert X.ndim == 3 and X.shape[1:] == (8, 6), f"X shape {X.shape} != (N, 8, 6)"
        assert Y.ndim == 2 and Y.shape[1] == 3, f"Y shape {Y.shape} != (N, 3)"
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]
