from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

import torch
from torch import nn

from tv3.common.composition import ILR_N2_FIRST_TRANSFORM
from tv3.sim.core.schema import COMPONENT_FIELDS


COMPOSITIONAL_MSE_LOSS = "compositional_mse"
ILR_MSE_LOSS = "ilr_mse"
FREE_COMPONENT_MSE_LOSS = "free_component_mse"
WEIGHTED_COMPONENT_MSE_LOSS = "weighted_component_mse"
WEIGHTED_FREE_COMPONENT_MSE_LOSS = "weighted_free_component_mse"
D2_TOF_PHASE_LOSS = "d2_tof_phase_loss"
TRANSFORMED_TARGET_MSE_LOSSES = frozenset((COMPOSITIONAL_MSE_LOSS, ILR_MSE_LOSS))
DEFAULT_FREE_COMPONENT_COUNT = len(COMPONENT_FIELDS) - 1
RAW_PERCENTAGE_LOSSES = frozenset(
    (FREE_COMPONENT_MSE_LOSS, WEIGHTED_COMPONENT_MSE_LOSS, WEIGHTED_FREE_COMPONENT_MSE_LOSS, D2_TOF_PHASE_LOSS)
)
# 三个 raw-percentage loss 都需要做 head 相关校验，因此共用这个总入口集合。
GAS_HEAD_PERCENTAGE_LOSSES = RAW_PERCENTAGE_LOSSES
# free-component 闭包类损失只监督前 3 列 (H2/CH4/CO2)，N2 靠 sum=100 闭包补全，
# 必须搭配 gas_head；只有监督全 4 列的 weighted_component_mse 不依赖闭包。
FREE_COMPONENT_CLOSURE_LOSSES = frozenset((FREE_COMPONENT_MSE_LOSS, WEIGHTED_FREE_COMPONENT_MSE_LOSS))
# 这些 loss 假设 4 列预测目标 sum=100%（hydrogen_ng 场景）。syngas 场景下
# 第 4 列是 x_CO 而非 x_N2，sum<100，闭包假设不成立，必须拒绝。
CLOSURE_DEPENDENT_LOSSES = frozenset(
    (
        COMPOSITIONAL_MSE_LOSS,
        ILR_MSE_LOSS,
        FREE_COMPONENT_MSE_LOSS,
        WEIGHTED_FREE_COMPONENT_MSE_LOSS,
    )
)
IMPLICIT_GAS_HEAD_MODELS = frozenset(("handcraft_mlp",))
# phase_window_tcn 上对 weighted_component_mse 放开的合法 head；
# free-component 闭包类损失仍只允许 gas_head。
RAW_PERCENTAGE_VALID_HEADS = frozenset(("raw4", "softmax100", "gas_head"))


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


class WeightedComponentMSELoss(nn.Module):
    """Component-wise MSE weighted by fixed per-component training statistics."""

    loss_name = WEIGHTED_COMPONENT_MSE_LOSS

    def __init__(
        self,
        component_weights: Sequence[float],
        component_count: int = len(COMPONENT_FIELDS),
    ):
        super().__init__()
        if component_count < 1:
            raise ValueError(f"component_count must be >= 1, got {component_count}")
        weights = torch.as_tensor(tuple(component_weights), dtype=torch.float32)
        if weights.ndim != 1:
            raise ValueError(f"{self.loss_name} component_weights must be a 1D sequence")
        if int(weights.numel()) != component_count:
            raise ValueError(
                f"{self.loss_name} requires {component_count} component weights, got {int(weights.numel())}"
            )
        if not bool(torch.isfinite(weights).all()):
            raise ValueError(f"{self.loss_name} component_weights must be finite")
        if bool(torch.any(weights <= 0.0)):
            raise ValueError(f"{self.loss_name} component_weights must be positive")
        self.component_count = component_count
        self.register_buffer("component_weights", weights)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 2 or target.ndim != 2:
            raise ValueError(
                f"{self.loss_name} expects 2D tensors shaped "
                f"(batch, components), got pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        if pred.shape != target.shape:
            raise ValueError(
                f"{self.loss_name} tensor shapes must match, "
                f"got pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )
        if pred.shape[1] < self.component_count:
            raise ValueError(
                f"{self.loss_name} requires at least {self.component_count} component columns, got {pred.shape[1]}"
            )
        error = pred[:, : self.component_count] - target[:, : self.component_count]
        weights = self.component_weights.to(device=error.device, dtype=error.dtype)
        return torch.mean(error * error * weights)


class WeightedFreeComponentMSELoss(WeightedComponentMSELoss):
    """Weighted MSE over free gas components [H2, CH4, CO2]."""

    loss_name = WEIGHTED_FREE_COMPONENT_MSE_LOSS

    def __init__(
        self,
        component_weights: Sequence[float],
        free_components: int = DEFAULT_FREE_COMPONENT_COUNT,
    ):
        super().__init__(component_weights=component_weights, component_count=free_components)


class D2TOFPhaseLoss(nn.Module):
    """D2 composite loss: 主三组分 loss + TOF / peak auxiliary loss.

    接收结构化 pred dict (含 ``prediction`` 与 ``aux``)、主 target 张量、
    以及 ``aux_targets`` dict（含 tof_true_s, tof_accepted 等），
    计算加权组合 loss。
    """

    accepts_aux_targets = True

    def __init__(
        self,
        component_loss: nn.Module,
        tof_loss_fn: nn.Module,
        peak_loss_fn: nn.Module,
        aux_weights: dict[str, float],
        waveform_length: int = 5000,
    ):
        super().__init__()
        self.component_loss = component_loss
        self.tof_loss_fn = tof_loss_fn
        self.peak_loss_fn = peak_loss_fn
        self.aux_weights = aux_weights
        self.waveform_length = waveform_length

    def forward(
        self,
        pred: dict[str, object] | torch.Tensor,
        target: torch.Tensor,
        aux_targets: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not isinstance(pred, dict):
            raise ValueError("D2TOFPhaseLoss requires structured model output (dict with 'prediction' and 'aux')")
        prediction = pred["prediction"]
        aux = pred.get("aux")
        if aux is None:
            raise ValueError("D2TOFPhaseLoss requires model output to contain 'aux' dict")

        main_loss = self.component_loss(prediction, target)

        tof_weight = self.aux_weights.get("tof_s", 0.0)
        peak_weight = self.aux_weights.get("peak_index", 0.0)

        if aux_targets is None and (tof_weight > 0.0 or peak_weight > 0.0):
            raise ValueError("D2TOFPhaseLoss requires aux_targets when aux_weights are non-zero")

        total_loss = main_loss

        if tof_weight > 0.0:
            tof_pred = aux["tof_s"]  # (B, T)
            tof_true = aux_targets["tof_true_s"]
            tof_accepted = aux_targets.get("tof_accepted")
            if tof_accepted is not None:
                mask = tof_accepted > 0.5
                if not mask.any():
                    raise ValueError("D2TOFPhaseLoss: tof_accepted mask is all-zero, cannot compute TOF loss")
                if mask.all():
                    tof_loss = self.tof_loss_fn(tof_pred, tof_true)
                else:
                    tof_loss = self.tof_loss_fn(tof_pred[mask], tof_true[mask])
            else:
                tof_loss = self.tof_loss_fn(tof_pred, tof_true)
            total_loss = total_loss + tof_weight * tof_loss

        if peak_weight > 0.0:
            peak_pred = aux["peak_index"]  # (B, T) — 模型输出需是归一化 index
            peak_true = aux_targets["peak_index"]  # (B, T) — 原始样本 index
            peak_true_norm = peak_true.float() / max(float(self.waveform_length) - 1.0, 1.0)
            peak_loss = self.peak_loss_fn(peak_pred, peak_true_norm)
            total_loss = total_loss + peak_weight * peak_loss

        return total_loss


LOSS_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "mse": nn.MSELoss,
    COMPOSITIONAL_MSE_LOSS: nn.MSELoss,
    ILR_MSE_LOSS: nn.MSELoss,
    FREE_COMPONENT_MSE_LOSS: FreeComponentMSELoss,
    WEIGHTED_COMPONENT_MSE_LOSS: WeightedComponentMSELoss,
    WEIGHTED_FREE_COMPONENT_MSE_LOSS: WeightedFreeComponentMSELoss,
    "mae": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "huber": nn.HuberLoss,
    D2_TOF_PHASE_LOSS: D2TOFPhaseLoss,  # sentinel; build_loss handles construction
}


def loss_config_name(config: str | dict[str, object]) -> str:
    if isinstance(config, str):
        return config
    if not isinstance(config, dict):
        raise ValueError("loss config must be a string or JSON object")
    if "name" not in config:
        raise ValueError("loss config object must contain a 'name' field")
    return str(config["name"])


def build_loss(config: str | dict[str, object], *, train_targets: object | None = None) -> nn.Module:
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

    if name == D2_TOF_PHASE_LOSS:
        return _build_d2_tof_phase_loss(kwargs, train_targets=train_targets)

    kwargs = _resolve_weighted_loss_kwargs(name, kwargs, train_targets=train_targets)
    return LOSS_REGISTRY[name](**kwargs)


def _build_d2_tof_phase_loss(
    kwargs: dict[str, object],
    *,
    train_targets: object | None,
) -> D2TOFPhaseLoss:
    component_loss_config = kwargs.get("component_loss")
    if component_loss_config is None:
        raise ValueError("d2_tof_phase_loss requires 'component_loss' config")
    component_loss = build_loss(component_loss_config, train_targets=train_targets)

    tof_loss_name = str(kwargs.get("tof_loss", "smooth_l1"))
    peak_loss_name = str(kwargs.get("peak_loss", "smooth_l1"))
    tof_loss_fn = build_loss(tof_loss_name)
    peak_loss_fn = build_loss(peak_loss_name)

    aux_weights_raw = kwargs.get("aux_weights")
    if not isinstance(aux_weights_raw, dict):
        raise ValueError("d2_tof_phase_loss requires 'aux_weights' dict")
    aux_weights: dict[str, float] = {}
    for k, v in aux_weights_raw.items():
        aux_weights[str(k)] = float(v)

    waveform_length = int(kwargs.get("waveform_length", 5000))
    return D2TOFPhaseLoss(
        component_loss=component_loss,
        tof_loss_fn=tof_loss_fn,
        peak_loss_fn=peak_loss_fn,
        aux_weights=aux_weights,
        waveform_length=waveform_length,
    )


def validate_loss_target_transform(loss_config: str | dict[str, object], target_transform_name: str | None) -> None:
    loss_name = loss_config_name(loss_config)
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss name: {loss_name!r}. Available: {sorted(LOSS_REGISTRY)}")
    if loss_name == COMPOSITIONAL_MSE_LOSS and target_transform_name is None:
        raise ValueError("compositional_mse requires target_transform")
    if loss_name == ILR_MSE_LOSS and target_transform_name != ILR_N2_FIRST_TRANSFORM:
        raise ValueError("ilr_mse requires target_transform='ilr_n2_first'")
    if loss_name in RAW_PERCENTAGE_LOSSES and target_transform_name is not None:
        raise ValueError(f"{loss_name} requires raw percentage targets without target_transform")


def validate_loss_composition_scheme(
    loss_config: str | dict[str, object],
    composition_scheme: str,
) -> None:
    """Reject closure-dependent losses on open-composition or non-closure datasets.

    syngas 场景下 4 列预测目标 sum<100（x_N2 是背景，不在 labels 里），
    闭包类 loss（compositional_mse / ilr_mse / free_component_mse /
    weighted_free_component_mse）的物理假设不成立，必须拒绝。

    tunnel_ventilation 场景数据层 sum=100% 闭包，但模型层不使用闭包残差头
    （直接输出 3 组分），闭包类 loss 依赖残差补全头，同样拒绝。
    """
    loss_name = loss_config_name(loss_config)
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss name: {loss_name!r}. Available: {sorted(LOSS_REGISTRY)}")
    if composition_scheme in ("syngas", "tunnel_ventilation") and loss_name in CLOSURE_DEPENDENT_LOSSES:
        raise ValueError(
            f"loss {loss_name!r} assumes sum=100 closure with residual head "
            f"(hydrogen_ng) and is incompatible with composition_scheme="
            f"{composition_scheme!r}. Use 'mse', 'weighted_component_mse', "
            f"'mae', 'smooth_l1', or 'huber' instead."
        )


def validate_loss_model_output(
    loss_config: str | dict[str, object],
    *,
    model_name: str,
    model_kwargs: object,
    composition_scheme: str = "tunnel_ventilation",
) -> None:
    """Check that the loss is compatible with the model's output head.

    composition_scheme:
        For "syngas", the open-composition target (4 columns sum<100) does not
        need gas-head closure, so weighted_component_mse / mse / etc. are
        allowed on any model. Closure-dependent losses are rejected earlier
        via ``validate_loss_composition_scheme``.

        For "tunnel_ventilation", fusion models must use direct raw outputs;
        closure residual heads and log-ratio coordinate heads are not part of
        the tv3 contract.
    """
    loss_name = loss_config_name(loss_config)
    if composition_scheme == "tunnel_ventilation":
        if not isinstance(model_kwargs, dict):
            raise ValueError(f"model_kwargs must be a JSON object for tunnel_ventilation {loss_name}")
        if model_name == "tof_phase_net":
            output_mode = model_kwargs.get("output_mode", "raw3")
            if output_mode != "raw3":
                raise ValueError(f"tunnel_ventilation {loss_name} requires tof_phase_net model_kwargs.output_mode='raw3'")
            return
        _validate_tunnel_ventilation_output_mode(model_name, model_kwargs, loss_name=loss_name)
        return
    if loss_name not in GAS_HEAD_PERCENTAGE_LOSSES:
        return
    if not isinstance(model_kwargs, dict):
        raise ValueError(f"model_kwargs must be a JSON object when using {loss_name}")
    if composition_scheme == "syngas":
        # 非 hg 场景不使用闭包残差头，weighted_component_mse 直接监督全列，
        # 不需要 gas-head / out_dim=4 校验。闭包类 loss 已由
        # validate_loss_composition_scheme 拒绝。
        return
    _validate_gas_head_out_dim(model_kwargs, loss_name=loss_name)
    if model_name == "phase_window_tcn":
        output_mode = model_kwargs.get("output_mode")
        if loss_name in FREE_COMPONENT_CLOSURE_LOSSES:
            # free-component 闭包类损失靠 sum=100 补 N2，只能用 gas_head。
            if output_mode != "gas_head":
                raise ValueError(
                    f"{loss_name} requires phase_window_tcn model_kwargs.output_mode='gas_head'"
                )
            return
        # weighted_component_mse 监督全 4 列，与 head 无关，
        # 允许 raw4 / softmax100 / gas_head；out_dim 已由上方 _validate_gas_head_out_dim 校验为 4。
        if output_mode not in RAW_PERCENTAGE_VALID_HEADS:
            raise ValueError(
                f"{loss_name} requires phase_window_tcn model_kwargs.output_mode "
                f"in {sorted(RAW_PERCENTAGE_VALID_HEADS)}, got {output_mode!r}"
            )
        return
    if model_name == "cnn1d_tcn_fusion":
        output_mode = model_kwargs.get("output_mode", "gas_head")
        if loss_name in FREE_COMPONENT_CLOSURE_LOSSES:
            # free-component 闭包类损失靠 sum=100 补 N2，只能用 gas_head。
            if output_mode != "gas_head":
                raise ValueError(
                    f"{loss_name} requires cnn1d_tcn_fusion model_kwargs.output_mode='gas_head'"
                )
            return
        if output_mode not in {"raw4", "gas_head"}:
            raise ValueError(
                f"{loss_name} requires cnn1d_tcn_fusion model_kwargs.output_mode "
                f"in ['gas_head', 'raw4'], got {output_mode!r}"
            )
        return
    if model_name in IMPLICIT_GAS_HEAD_MODELS:
        return
    raise ValueError(f"{loss_name} requires a gas-head DL model")


def _validate_gas_head_out_dim(model_kwargs: dict[str, object], *, loss_name: str) -> None:
    if "out_dim" not in model_kwargs:
        return
    try:
        out_dim = int(model_kwargs["out_dim"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{loss_name} requires model out_dim=4") from exc
    if out_dim != 4:
        raise ValueError(f"{loss_name} requires model out_dim=4")


def _validate_tunnel_ventilation_output_mode(
    model_name: str,
    model_kwargs: dict[str, object],
    *,
    loss_name: str,
) -> None:
    if model_name == "cnn1d_tcn_fusion":
        output_mode = model_kwargs.get("output_mode", "gas_head")
        if output_mode != "raw3":
            raise ValueError(
                f"tunnel_ventilation {loss_name} requires cnn1d_tcn_fusion "
                f"model_kwargs.output_mode='raw3'"
            )
        try:
            out_dim = int(model_kwargs.get("out_dim", 3))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"tunnel_ventilation {loss_name} requires model out_dim=3") from exc
        if out_dim != 3:
            raise ValueError(f"tunnel_ventilation {loss_name} requires model out_dim=3")
        return
    if model_name == "phase_window_tcn":
        output_mode = model_kwargs.get("output_mode", "raw4")
        if output_mode == "gas_head":
            raise ValueError(
                f"tunnel_ventilation {loss_name} requires direct raw outputs, "
                f"got phase_window_tcn model_kwargs.output_mode='gas_head'"
            )


def _resolve_weighted_loss_kwargs(
    loss_name: str,
    kwargs: dict[str, object],
    *,
    train_targets: object | None,
) -> dict[str, object]:
    if loss_name not in {WEIGHTED_COMPONENT_MSE_LOSS, WEIGHTED_FREE_COMPONENT_MSE_LOSS}:
        return kwargs
    resolved = dict(kwargs)
    weighting = resolved.pop("weighting", None)
    weight_normalization = str(resolved.pop("weight_normalization", "none"))
    if weight_normalization not in {"none", "mean_one"}:
        raise ValueError(f"{loss_name} weight_normalization must be one of ['none', 'mean_one']")
    if weighting is None:
        if "component_weights" not in resolved:
            raise ValueError(
                f"{loss_name} requires component_weights or weighting in ['inverse_train_var', 'fixed']"
            )
        resolved["component_weights"] = _normalize_component_weights(
            resolved["component_weights"],
            normalization=weight_normalization,
        )
        return resolved
    if weighting == "inverse_train_var":
        if "component_weights" in resolved:
            raise ValueError(f"{loss_name} cannot combine component_weights with weighting='inverse_train_var'")
        if train_targets is None:
            raise ValueError(f"{loss_name} weighting='inverse_train_var' requires train_targets")
        component_count = _weighted_loss_component_count(loss_name, resolved)
        weights = _inverse_train_variance_weights(train_targets, component_count)
        resolved["component_weights"] = _normalize_component_weights(weights, normalization=weight_normalization)
        return resolved
    if weighting == "fixed":
        if "loss_weights" not in resolved:
            raise ValueError(f"{loss_name} weighting='fixed' requires loss_weights")
        if "component_weights" in resolved:
            raise ValueError(f"{loss_name} cannot combine component_weights with weighting='fixed'")
        resolved["component_weights"] = _normalize_component_weights(
            resolved.pop("loss_weights"),
            normalization=weight_normalization,
        )
        return resolved
    raise ValueError(f"{loss_name} unknown weighting: {weighting!r}")


def _weighted_loss_component_count(loss_name: str, kwargs: dict[str, object]) -> int:
    if loss_name == WEIGHTED_COMPONENT_MSE_LOSS:
        return int(kwargs.get("component_count", len(COMPONENT_FIELDS)))
    return int(kwargs.get("free_components", DEFAULT_FREE_COMPONENT_COUNT))


def _inverse_train_variance_weights(train_targets: object, component_count: int) -> tuple[float, ...]:
    targets = torch.as_tensor(train_targets, dtype=torch.float32)
    if targets.ndim != 2:
        raise ValueError(f"train_targets must be shaped (samples, components), got {tuple(targets.shape)}")
    if targets.shape[1] < component_count:
        raise ValueError(f"train_targets requires at least {component_count} component columns, got {targets.shape[1]}")
    variances = torch.var(targets[:, :component_count], dim=0, unbiased=False)
    if not bool(torch.isfinite(variances).all()):
        raise ValueError("inverse_train_var requires finite train variances")
    if bool(torch.any(variances <= 0.0)):
        raise ValueError("inverse_train_var requires positive train variance for every selected component")
    weights = 1.0 / variances
    return tuple(float(value) for value in weights.tolist())


def _normalize_component_weights(weights: object, *, normalization: str) -> tuple[float, ...]:
    values = torch.as_tensor(tuple(weights), dtype=torch.float32)
    if normalization == "none":
        return tuple(float(value) for value in values.tolist())
    if normalization == "mean_one":
        if values.numel() < 1:
            raise ValueError("weight_normalization='mean_one' requires at least one weight")
        mean = torch.mean(values)
        if not bool(torch.isfinite(mean)) or float(mean.item()) <= 0.0:
            raise ValueError("weight_normalization='mean_one' requires positive finite mean weight")
        values = values / mean
        return tuple(float(value) for value in values.tolist())
    raise ValueError(f"unknown weight_normalization: {normalization!r}")
