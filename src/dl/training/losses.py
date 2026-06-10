from __future__ import annotations

from typing import Callable

from torch import nn

from common.composition import ILR_N2_FIRST_TRANSFORM


COMPOSITIONAL_MSE_LOSS = "compositional_mse"
ILR_MSE_LOSS = "ilr_mse"
TRANSFORMED_TARGET_MSE_LOSSES = frozenset((COMPOSITIONAL_MSE_LOSS, ILR_MSE_LOSS))

LOSS_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "mse": nn.MSELoss,
    COMPOSITIONAL_MSE_LOSS: nn.MSELoss,
    ILR_MSE_LOSS: nn.MSELoss,
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


def validate_loss_target_transform(loss_name: str, target_transform_name: str | None) -> None:
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss name: {loss_name!r}. Available: {sorted(LOSS_REGISTRY)}")
    if loss_name == COMPOSITIONAL_MSE_LOSS and target_transform_name is None:
        raise ValueError("compositional_mse requires target_transform")
    if loss_name == ILR_MSE_LOSS and target_transform_name != ILR_N2_FIRST_TRANSFORM:
        raise ValueError("ilr_mse requires target_transform='ilr_n2_first'")
