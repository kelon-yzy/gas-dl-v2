"""v4 DL 模型子模块 — 注册表与具体实现。"""

from hg.dl.models.base import BaseRegressor
from hg.dl.models.cnn1d import CNN1DRegressor
from hg.dl.models.cnn1d_tcn_fusion import CNN1DTCNFusionRegressor, DeepAcousticEncoder1D, GasHeadNormalize, SlowFeatureEncoder
from hg.dl.models.handcraft_mlp import HandcraftMLPRegressor
from hg.dl.models.lstm import LSTMRegressor
from hg.dl.models.patchtst import PatchTSTRegressor
from hg.dl.models.phase_window_tcn import PhaseWindowTCNRegressor, WindowedFusionEncoder
from hg.dl.models.registry import MODEL_REGISTRY, build_model
from hg.dl.models.tcn import CausalConv1d, TCNRegressor, TemporalBlock
from hg.dl.models.transformer import TransformerRegressor

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
