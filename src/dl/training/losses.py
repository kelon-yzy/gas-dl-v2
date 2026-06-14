from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from common.composition import ILR_N2_FIRST_TRANSFORM
from sim.core.schema import COMPONENT_FIELDS


COMPOSITIONAL_MSE_LOSS = "compositional_mse"
ILR_MSE_LOSS = "ilr_mse"
FREE_COMPONENT_MSE_LOSS = "free_component_mse"
TRANSFORMED_TARGET_MSE_LOSSES = frozenset((COMPOSITIONAL_MSE_LOSS, ILR_MSE_LOSS))
DEFAULT_FREE_COMPONENT_COUNT = len(COMPONENT_FIELDS) - 1
IMPLICIT_GAS_HEAD_MODELS = frozenset(("cnn1d_tcn_fusion",))


class FreeComponentMSELoss(nn.Module):
    """MSE over free gas components [H2, CH4, CO2], leaving N2 as closure."""

    def __init__(self, free_components: int = DEFAULT_FREE_COMPONENT_COUNT):
        super().__init__()
        if free_components < 1:
            raise ValueError(f"free_components must be >= 1, got {free_components}")
        self.free_components = free_components
        self.loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 2 or target.ndim != 2:
            raise ValueError(
                "free_component_mse expects 2D tensors shaped "
                f"(batch, components), got pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        if pred.shape != target.shape:
            raise ValueError(
                "free_component_mse tensor shapes must match, "
                f"got pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        if pred.shape[1] < self.free_components:
            raise ValueError(
                f"free_component_mse requires at least {self.free_components} component columns, got {pred.shape[1]}"
            )
        return self.loss(pred[:, : self.free_components], target[:, : self.free_components])


LOSS_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "mse": nn.MSELoss,
    COMPOSITIONAL_MSE_LOSS: nn.MSELoss,
    ILR_MSE_LOSS: nn.MSELoss,
    FREE_COMPONENT_MSE_LOSS: FreeComponentMSELoss,
    "mae": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
}


def loss_config_name(config: str | dict[str, object]) -> str:
    if isinstance(config, str):
        return config
    if not isinstance(config, dict):
        raise ValueError("loss config must be a string or JSON object")
    if "name" not in config:
        raise ValueError("loss config object must contain a 'name' field")
    return str(config["name"])


def build_loss(config: str | dict[str, object]) -> nn.Module:
    """根据训练配置构造 PyTorch loss。

    ``config`` 可以直接传 loss 名称，也可以传含 ``name`` 的配置字典；
    其余键值对会作为构造参数传给对应 loss 类。
    """
    name = loss_config_name(config)
    if isinstance(config, str):
        kwargs: dict[str, object] = {}
    else:
        loss_config = dict(config)
        loss_config.pop("name")
        kwargs = loss_config

    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss name: {name!r}. Available: {sorted(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)


def validate_loss_target_transform(loss_config: str | dict[str, object], target_transform_name: str | None) -> None:
    loss_name = loss_config_name(loss_config)
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss name: {loss_name!r}. Available: {sorted(LOSS_REGISTRY)}")
    if loss_name == COMPOSITIONAL_MSE_LOSS and target_transform_name is None:
        raise ValueError("compositional_mse requires target_transform")
    if loss_name == ILR_MSE_LOSS and target_transform_name != ILR_N2_FIRST_TRANSFORM:
        raise ValueError("ilr_mse requires target_transform='ilr_n2_first'")
    if loss_name == FREE_COMPONENT_MSE_LOSS and target_transform_name is not None:
        raise ValueError("free_component_mse requires raw percentage targets without target_transform")


def validate_loss_model_output(
    loss_config: str | dict[str, object],
    *,
    model_name: str,
    model_kwargs: object,
) -> None:
    loss_name = loss_config_name(loss_config)
    if loss_name != FREE_COMPONENT_MSE_LOSS:
        return
    if not isinstance(model_kwargs, dict):
        raise ValueError("model_kwargs must be a JSON object when using free_component_mse")
    _validate_free_component_out_dim(model_kwargs)
    if model_name == "phase_window_tcn":
        if model_kwargs.get("output_mode") != "gas_head":
            raise ValueError("free_component_mse requires phase_window_tcn model_kwargs.output_mode='gas_head'")
        return
    if model_name in IMPLICIT_GAS_HEAD_MODELS:
        return
    raise ValueError("free_component_mse requires a gas-head DL model")


def _validate_free_component_out_dim(model_kwargs: dict[str, object]) -> None:
    if "out_dim" not in model_kwargs:
        return
    try:
        out_dim = int(model_kwargs["out_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError("free_component_mse requires model out_dim=4") from exc
    if out_dim != 4:
        raise ValueError("free_component_mse requires model out_dim=4")
