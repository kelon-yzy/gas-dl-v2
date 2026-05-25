"""v4 DL 模型子模块 — 注册表与具体实现。"""

from dl.models.base import BaseRegressor
from dl.models.cnn1d import CNN1DRegressor
from dl.models.registry import MODEL_REGISTRY, build_model

__all__ = [
    "BaseRegressor",
    "CNN1DRegressor",
    "MODEL_REGISTRY",
    "build_model",
]
