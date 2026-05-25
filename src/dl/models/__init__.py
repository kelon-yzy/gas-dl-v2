"""v4 DL 模型子模块 — 注册表与具体实现。"""

from dl.models.base import BaseRegressor
from dl.models.cnn1d import CNN1DRegressor
from dl.models.registry import MODEL_REGISTRY, build_model
from dl.models.tcn import CausalConv1d, TCNRegressor, TemporalBlock

__all__ = [
    "BaseRegressor",
    "CausalConv1d",
    "CNN1DRegressor",
    "MODEL_REGISTRY",
    "TCNRegressor",
    "TemporalBlock",
    "build_model",
]
