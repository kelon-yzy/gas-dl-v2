"""v4 DL 模型子模块 — 注册表与具体实现。"""

from dl.models.base import BaseRegressor
from dl.models.cnn1d import CNN1DRegressor
from dl.models.lstm import LSTMRegressor
from dl.models.patchtst import PatchTSTRegressor
from dl.models.registry import MODEL_REGISTRY, build_model
from dl.models.tcn import CausalConv1d, TCNRegressor, TemporalBlock
from dl.models.transformer import TransformerRegressor

__all__ = [
    "BaseRegressor",
    "CausalConv1d",
    "CNN1DRegressor",
    "LSTMRegressor",
    "MODEL_REGISTRY",
    "PatchTSTRegressor",
    "TCNRegressor",
    "TemporalBlock",
    "TransformerRegressor",
    "build_model",
]
