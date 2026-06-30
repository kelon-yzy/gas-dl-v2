"""预处理占位模块。

合成数据已对齐，无需实际预处理。
真实数据接入时在此实现滤波/校准/补偿。
"""
from __future__ import annotations

import numpy as np


def sliding_mean_filter(x: np.ndarray, window: int = 8) -> np.ndarray:
    """滑动窗口均值滤波（按最后一个轴）。"""
    if x.shape[0] < window:
        return x
    kernel = np.ones(window) / window
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), 0, x)
    return out.astype(x.dtype)


def calibrate_linear(x: np.ndarray, a: float = 1.0, b: float = 0.0) -> np.ndarray:
    """线性零点+跨度校准: y = a * x + b"""
    return (a * x + b).astype(x.dtype)
