"""v4 DL 模型子模块 — 注册表与具体实现。"""

from sg.dl.models.base import BaseRegressor
from sg.dl.models.cnn1d import CNN1DRegressor
from sg.dl.models.cnn1d_tcn_fusion import CNN1DTCNFusionRegressor, DeepAcousticEncoder1D, GasHeadNormalize, SlowFeatureEncoder
from sg.dl.models.handcraft_mlp import HandcraftMLPRegressor
from sg.dl.models.lstm import LSTMRegressor
from sg.dl.models.patchtst import PatchTSTRegressor
from sg.dl.models.phase_window_tcn import PhaseWindowTCNRegressor, WindowedFusionEncoder
from sg.dl.models.registry import MODEL_REGISTRY, build_model
from sg.dl.models.tcn import CausalConv1d, TCNRegressor, TemporalBlock
from sg.dl.models.transformer import TransformerRegressor

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
