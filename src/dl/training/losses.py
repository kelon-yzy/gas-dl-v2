from __future__ import annotations

from typing import Callable

from torch import nn


LOSS_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
}


def build_loss(config: str | dict[str, object]) -> nn.Module:
    """根据训练配置构造 PyTorch loss。

    ``config`` 可以直接传 loss 名称，也可以传含 ``name`` 的配置字典；
    其余键值对会作为构造参数传给对应 loss 类。
    """
    if isinstance(config, str):
        name = config
        kwargs: dict[str, object] = {}
    else:
        loss_config = dict(config)
        name = str(loss_config.pop("name"))
        kwargs = loss_config

    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss name: {name!r}. Available: {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)
