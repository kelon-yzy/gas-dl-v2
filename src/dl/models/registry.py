from __future__ import annotations

from typing import Callable

from torch import nn

from dl.models.cnn1d import CNN1DRegressor
from dl.models.cnn1d_tcn_fusion import CNN1DTCNFusionRegressor
from dl.models.lstm import LSTMRegressor
from dl.models.patchtst import PatchTSTRegressor
from dl.models.tcn import TCNRegressor
from dl.models.transformer import TransformerRegressor

MODEL_REGISTRY: dict[str, type[nn.Module] | Callable[..., nn.Module]] = {
    "cnn1d": CNN1DRegressor,
    "cnn1d_tcn_fusion": CNN1DTCNFusionRegressor,
    "lstm": LSTMRegressor,
    "patchtst": PatchTSTRegressor,
    "tcn": TCNRegressor,
    "transformer": TransformerRegressor,
}


def build_model(config: dict[str, object]) -> nn.Module:
    """根据配置字典构造模型。

    ``config`` 必须包含 ``"name"`` 键，对应 ``MODEL_REGISTRY`` 中的注册名。
    其余键值对作为模型构造参数传入。
    """
    model_config = dict(config)
    name = model_config.pop("name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[name]
    if isinstance(entry, type) and issubclass(entry, nn.Module):
        return entry(**model_config)
    return entry(**model_config)
