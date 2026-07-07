"""v4 DL 模型子模块 — 注册表与具体实现。"""

from tv3.dl.models.base import BaseRegressor
from tv3.dl.models.cnn1d import CNN1DRegressor
from tv3.dl.models.cnn1d_tcn_fusion import CNN1DTCNFusionRegressor, DeepAcousticEncoder1D, GasHeadNormalize, SlowFeatureEncoder
from tv3.dl.models.handcraft_mlp import HandcraftMLPRegressor
from tv3.dl.models.lstm import LSTMRegressor
from tv3.dl.models.patchtst import PatchTSTRegressor
from tv3.dl.models.phase_window_tcn import PhaseWindowTCNRegressor, WindowedFusionEncoder
from tv3.dl.models.registry import MODEL_REGISTRY, build_model
from tv3.dl.models.tcn import CausalConv1d, TCNRegressor, TemporalBlock
from tv3.dl.models.transformer import TransformerRegressor

__all__ = [
    "BaseRegressor",
    "CausalConv1d",
    "CNN1DRegressor",
    "CNN1DTCNFusionRegressor",
    "DeepAcousticEncoder1D",
    "GasHeadNormalize",
    "HandcraftMLPRegressor",
    "LSTMRegressor",
    "MODEL_REGISTRY",
    "PatchTSTRegressor",
    "PhaseWindowTCNRegressor",
    "SlowFeatureEncoder",
    "TCNRegressor",
    "TemporalBlock",
    "TransformerRegressor",
    "WindowedFusionEncoder",
    "build_model",
]
