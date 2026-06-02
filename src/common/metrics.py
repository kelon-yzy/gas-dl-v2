from __future__ import annotations

from dataclasses import dataclass


R2_ZERO_VARIANCE_EPSILON = 1e-12
"""Total sum-of-squares below this treats the target as constant for R2."""


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """四组分浓度回归的整体指标（pooled MAE / RMSE / R2）。

    ml（NumPy）与 dl（PyTorch）共用此数据结构与零方差阈值，计算层各自实现，
    避免指标 schema 双份维护（见 KARPATHY_REVIEW 2.2）。新增指标字段时只改这里。
    """

    mae: float
    rmse: float
    r2: float
