"""Z-score scaler 与异质通道归一化策略所需常量。"""

from __future__ import annotations


Z_SCORE_STD_EPSILON = 1e-12
"""标准差视为有效尺度的下限；低于此阈值的通道 std 用 1.0 替代以避免除零。"""
