from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from gf.dl.contracts import UnifiedBatch, UnifiedSample, collate_samples
from gf.dl.evaluation import evaluate_predictions
from gf.dl.fusion_core import ConcatFusionCore, FusionCore
from gf.dl.preprocessing import TrainGroupStandardScaler
from gf.dl.sensor_encoders import A2ScalarTokenEncoder
from gf.dl.task_heads import (
    FixedTotalSoftmaxHead,
    RegressionHead,
    SimplexProjectionHead,
    SparsemaxHead,
)


@dataclass(frozen=True)
class TorchTrainingConfig:
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float
    target_scale: tuple[float, ...]
    optimizer_name: str = "Adam"
    optimizer_max_iter: int = 20
    optimizer_history_size: int = 50

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "TorchTrainingConfig":
        optimizer = _required_mapping(config, "optimizer")
        loss = _required_mapping(config, "loss")
        early_stopping = _required_mapping(config, "early_stopping")
        max_epochs = config.get("max_epochs")
        patience = early_stopping.get("patience")
        if not isinstance(max_epochs, int) or isinstance(max_epochs, bool) or max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer")
        if not isinstance(patience, int) or isinstance(patience, bool) or patience <= 0:
            raise ValueError("early_stopping.patience must be a positive integer")
        learning_rate = optimizer.get("learning_rate")
        weight_decay = optimizer.get("weight_decay")
        optimizer_name = optimizer.get("name")
        if optimizer_name not in {"Adam", "AdamW", "LBFGS"}:
            raise ValueError("optimizer.name must be Adam, AdamW, or LBFGS")
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or learning_rate <= 0.0:
            raise ValueError("optimizer.learning_rate must be positive")
        if not isinstance(weight_decay, (int, float)) or isinstance(weight_decay, bool) or weight_decay < 0.0:
            raise ValueError("optimizer.weight_decay must be non-negative")
        optimizer_max_iter = optimizer.get("max_iter", 20)
        optimizer_history_size = optimizer.get("history_size", 50)
        if not isinstance(optimizer_max_iter, int) or isinstance(optimizer_max_iter, bool) or optimizer_max_iter <= 0:
            raise ValueError("optimizer.max_iter must be a positive integer")
        if not isinstance(optimizer_history_size, int) or isinstance(optimizer_history_size, bool) or optimizer_history_size <= 0:
            raise ValueError("optimizer.history_size must be a positive integer")
        target_scale = loss.get("target_scale")
        if (
            not isinstance(target_scale, list)
            or not target_scale
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or float(value) <= 0.0
                for value in target_scale
            )
        ):
            raise ValueError("loss.target_scale must be a non-empty list of positive numbers")
        return cls(
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            target_scale=tuple(float(value) for value in target_scale),
            optimizer_name=str(optimizer_name),
            optimizer_max_iter=optimizer_max_iter,
            optimizer_history_size=optimizer_history_size,
        )


class A2FusionModel(nn.Module):
    """A2 model with an explicit concat or Deep Sets representation."""

    def __init__(
        self,
        *,
        representation: str,
        embedding_dim: int,
        fusion_hidden_dim: int,
        output_dim: int,
        sensor_ids: Sequence[str],
        sensor_types: Sequence[str],
        head_id: str = "H0",
        pooling: str = "masked_mean",
        max_sensors: int | None = None,
        concat_dim: int | None = None,
        temperature: float = 1.0,
        context_keys: Sequence[str] = (),
    ) -> None:
        super().__init__()
        if representation not in {"torch_concat", "sensor_token", "deepsets"}:
            raise ValueError(f"unsupported A2 representation: {representation!r}")
        if max_sensors is None:
            max_sensors = len(sensor_ids)
        if max_sensors <= 0:
            raise ValueError("max_sensors must be positive")
        self.representation = representation
        self.context_keys = tuple(context_keys)
        if any(not key for key in self.context_keys) or len(set(self.context_keys)) != len(self.context_keys):
            raise ValueError("context_keys must contain unique non-empty names")
        self.encoder = A2ScalarTokenEncoder(
            embedding_dim=embedding_dim,
            sensor_ids=sensor_ids,
            sensor_types=sensor_types,
        )
        if representation == "torch_concat":
            self.fusion = ConcatFusionCore(
                embedding_dim=embedding_dim,
                hidden_dim=fusion_hidden_dim,
                max_sensors=max_sensors,
                concat_dim=concat_dim,
            )
        else:
            if pooling not in {"masked_mean", "sum"}:
                raise ValueError("A2 set representation requires masked_mean or sum pooling")
            self.fusion = FusionCore(
                embedding_dim=embedding_dim,
                hidden_dim=fusion_hidden_dim,
                pooling=pooling,
            )
        head_input_dim = fusion_hidden_dim + len(self.context_keys)
        if head_id == "H0":
            self.head = RegressionHead(head_input_dim, output_dim)
        elif head_id == "H1":
            self.head = SimplexProjectionHead(
                head_input_dim,
                output_dim,
                base_head=RegressionHead(head_input_dim, output_dim),
            )
        elif head_id == "H2":
            self.head = FixedTotalSoftmaxHead(
                head_input_dim,
                output_dim,
                temperature=temperature,
            )
        elif head_id == "H3":
            self.head = SparsemaxHead(head_input_dim, output_dim)
        else:
            raise ValueError(f"unsupported A2 head id: {head_id!r}")
        self.head_id = head_id

    def forward(self, batch: UnifiedBatch) -> torch.Tensor:
        tokens, token_mask = self.encoder(batch)
        if self.representation == "torch_concat":
            fused = self.fusion(tokens, token_mask)
        else:
            fused = self.fusion(tokens, token_mask)
        if self.context_keys:
            fused = torch.cat((fused, _context_tensor(batch, self.context_keys)), dim=1)
        return self.head(fused)


class TorchConcatMLP(nn.Module):
    """Direct ordered scalar concat control matching the A1 B5 input semantics."""

    def __init__(
        self,
        *,
        sensor_count: int,
        hidden_dim: int,
        output_dim: int,
        context_keys: Sequence[str] = (),
    ) -> None:
        super().__init__()
        if sensor_count <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("sensor_count, hidden_dim, and output_dim must be positive")
        self.sensor_count = sensor_count
        self.context_keys = tuple(context_keys)
        if any(not key for key in self.context_keys) or len(set(self.context_keys)) != len(self.context_keys):
            raise ValueError("context_keys must contain unique non-empty names")
        self.output_scale = 100.0
        self.optimization_loss_scale = 1.0
        self.backbone = nn.Sequential(
            nn.Linear(sensor_count + len(self.context_keys), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: UnifiedBatch) -> torch.Tensor:
        if batch.signals.shape[1] != self.sensor_count:
            raise ValueError("batch sensor count does not match TorchConcatMLP")
        valid = batch.valid_mask
        counts = valid.sum(dim=(-1, -2))
        if torch.any(counts <= 0):
            raise ValueError("TorchConcatMLP requires one valid observation per sensor")
        values = (batch.signals * valid).sum(dim=(-1, -2)) / counts.to(batch.signals.dtype)
        if self.context_keys:
            values = torch.cat((values, _context_tensor(batch, self.context_keys)), dim=1)
        return self.backbone(values) * self.output_scale


def build_a2_model_from_config(
    model_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    *,
    capacity_name: str,
    head_id: str | None = None,
) -> A2FusionModel:
    presets = train_config.get("capacity_presets")
    if not isinstance(presets, list):
        raise ValueError("train_config.capacity_presets must be a list")
    matching = [
        preset
        for preset in presets
        if isinstance(preset, Mapping) and preset.get("name") == capacity_name
    ]
    if len(matching) != 1:
        raise ValueError(f"capacity preset {capacity_name!r} must resolve to exactly one entry")
    preset = matching[0]
    model_head = head_id or str(model_config.get("head_id", "H0"))
    sensor_ids = model_config.get("sensor_ids")
    sensor_types = model_config.get("sensor_types")
    if not _is_string_sequence(sensor_ids) or not _is_string_sequence(sensor_types):
        raise ValueError("model_config sensor_ids and sensor_types must be non-empty string lists")
    concat_dim = None
    if model_config.get("representation") == "torch_concat":
        concat_dims = model_config.get("concat_dim_by_capacity")
        if concat_dims is not None:
            if not isinstance(concat_dims, Mapping):
                raise ValueError("concat_dim_by_capacity must be an object")
            value = concat_dims.get(capacity_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"concat_dim_by_capacity lacks a positive value for {capacity_name!r}")
            concat_dim = value
    return A2FusionModel(
        representation=str(model_config.get("representation", "")),
        embedding_dim=int(preset["encoder_hidden_dim"]),
        fusion_hidden_dim=int(preset["fusion_hidden_dim"]),
        output_dim=3,
        sensor_ids=sensor_ids,
        sensor_types=sensor_types,
        head_id=model_head,
        pooling=str(model_config.get("pool", "masked_mean")),
        max_sensors=len(sensor_ids),
        concat_dim=concat_dim,
        context_keys=tuple(str(key) for key in model_config.get("context_keys", ())),
    )


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_macro_RNMAE: float
    epochs_completed: int
    history: tuple[Mapping[str, float], ...]
    validation_predictions: np.ndarray


def train_torch_model(
    model: nn.Module,
    train_samples: Sequence[UnifiedSample],
    validation_samples: Sequence[UnifiedSample],
    *,
    config: TorchTrainingConfig | Mapping[str, Any],
    seed: int,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    if not train_samples or not validation_samples:
        raise ValueError("train_samples and validation_samples must both be non-empty")
    _validate_training_samples(train_samples, validation_samples)
    training_config = (
        config if isinstance(config, TorchTrainingConfig) else TorchTrainingConfig.from_mapping(config)
    )
    if len(training_config.target_scale) != validation_samples[0].target.size:
        raise ValueError("target_scale width does not match sample targets")
    torch.manual_seed(seed)
    model.train()
    if training_config.optimizer_name in {"Adam", "AdamW"}:
        optimizer_class = torch.optim.Adam if training_config.optimizer_name == "Adam" else torch.optim.AdamW
        optimizer = optimizer_class(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
    else:
        if training_config.weight_decay != 0.0:
            raise ValueError("LBFGS training requires weight_decay=0")
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=training_config.learning_rate,
            max_iter=training_config.optimizer_max_iter,
            history_size=training_config.optimizer_history_size,
            line_search_fn="strong_wolfe",
        )
    train_batch = collate_samples(tuple(train_samples))
    validation_batch = collate_samples(tuple(validation_samples))
    train_target = train_batch.target
    validation_target = validation_batch.target
    target_scale = torch.tensor(training_config.target_scale, dtype=torch.float32)
    best_metric = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[Mapping[str, float]] = []

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        if training_config.optimizer_name == "LBFGS":
            normalized_train_loss: torch.Tensor | None = None

            def closure() -> torch.Tensor:
                nonlocal normalized_train_loss
                optimizer.zero_grad(set_to_none=True)
                train_prediction = model(train_batch)
                normalized_train_loss = _normalized_masked_mse(
                    train_prediction,
                    train_target,
                    train_batch.target_mask,
                    target_scale,
                )
                optimization_loss = normalized_train_loss * getattr(
                    model,
                    "optimization_loss_scale",
                    target_scale.square().mean(),
                )
                if not torch.isfinite(optimization_loss):
                    raise RuntimeError(f"non-finite training loss at epoch {epoch}")
                optimization_loss.backward()
                return optimization_loss

            optimizer.step(closure)
            if normalized_train_loss is None:
                raise RuntimeError(f"LBFGS did not evaluate a loss at epoch {epoch}")
        else:
            optimizer.zero_grad(set_to_none=True)
            train_prediction = model(train_batch)
            normalized_train_loss = _normalized_masked_mse(
                train_prediction,
                train_target,
                train_batch.target_mask,
                target_scale,
            )
            optimization_loss = normalized_train_loss * getattr(
                model,
                "optimization_loss_scale",
                target_scale.square().mean(),
            )
            if not torch.isfinite(optimization_loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            optimization_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_batch)
        validation_values = validation_prediction.detach().cpu().numpy().astype(np.float64)
        validation_targets = validation_target.detach().cpu().numpy().astype(np.float64)
        validation_metric = evaluate_predictions(
            validation_targets,
            validation_values,
            validation_batch.group_id,
            np.arange(len(validation_samples), dtype=np.int64),
        )["macro_RNMAE"]
        metric_value = float(validation_metric)
        history.append({"epoch": float(epoch), "train_loss": float(normalized_train_loss.detach()), "val_macro_RNMAE": metric_value})
        if metric_value < best_metric:
            best_metric = metric_value
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path is not None:
                _write_checkpoint(
                    Path(checkpoint_path),
                    model=model,
                    seed=seed,
                    epoch=epoch,
                    validation_metric=metric_value,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training_config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_prediction = model(validation_batch).detach().cpu().numpy().astype(np.float64)
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_macro_RNMAE=best_metric,
        epochs_completed=len(history),
        history=tuple(history),
        validation_predictions=final_prediction,
    )


def prepare_a2_train_val_samples(
    samples: Sequence[UnifiedSample],
) -> tuple[list[UnifiedSample], list[UnifiedSample], TrainGroupStandardScaler]:
    """Fit one scaler on train groups and return only train and val samples."""

    train_samples = [sample for sample in samples if sample.metadata.get("split") == "train"]
    validation_samples = [sample for sample in samples if sample.metadata.get("split") == "val"]
    if not train_samples or not validation_samples:
        raise ValueError("A2 preparation requires non-empty train and val samples")
    train_groups = {sample.group_id for sample in train_samples}
    scaler = TrainGroupStandardScaler()
    scaler.fit(list(samples), train_groups)
    return (
        [scaler.transform(sample) for sample in train_samples],
        [scaler.transform(sample) for sample in validation_samples],
        scaler,
    )


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def parameter_parity_report(
    left: nn.Module,
    right: nn.Module,
    *,
    tolerance: float = 0.10,
) -> dict[str, Any]:
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    left_count = trainable_parameter_count(left)
    right_count = trainable_parameter_count(right)
    relative_difference = abs(left_count - right_count) / max(left_count, right_count)
    return {
        "left_parameter_count": left_count,
        "right_parameter_count": right_count,
        "relative_difference": float(relative_difference),
        "within_tolerance": bool(relative_difference <= tolerance),
    }


def _normalized_masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or target_mask.shape != target.shape:
        raise ValueError("prediction, target, and target_mask must have equal shape")
    scale = target_scale.to(device=prediction.device, dtype=prediction.dtype)
    mask = target_mask.to(device=prediction.device, dtype=prediction.dtype)
    error = (prediction / scale - target.to(prediction.device) / scale).square() * mask
    denominator = mask.sum()
    if denominator <= 0.0:
        raise ValueError("target_mask must contain at least one true value")
    return error.sum() / denominator


def _context_tensor(batch: UnifiedBatch, context_keys: Sequence[str]) -> torch.Tensor:
    rows: list[list[float]] = []
    for metadata in batch.metadata:
        row: list[float] = []
        for key in context_keys:
            value = metadata.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(float(value)):
                raise ValueError(f"batch metadata lacks finite context value {key!r}")
            row.append(float(value))
        rows.append(row)
    return torch.tensor(rows, dtype=batch.signals.dtype, device=batch.signals.device)


def _validate_training_samples(
    train_samples: Sequence[UnifiedSample],
    validation_samples: Sequence[UnifiedSample],
) -> None:
    train_groups = {sample.group_id for sample in train_samples}
    validation_groups = {sample.group_id for sample in validation_samples}
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(f"training and validation groups overlap: {sorted(overlap)}")
    for split_name, samples in (("train", train_samples), ("val", validation_samples)):
        for sample in samples:
            declared_split = sample.metadata.get("split")
            if declared_split == "test":
                raise ValueError(f"test sample passed to {split_name} training input")


def _write_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    seed: int,
    epoch: int,
    validation_metric: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "gf-a2-checkpoint-1",
            "seed": int(seed),
            "epoch": int(epoch),
            "validation_macro_RNMAE": float(validation_metric),
            "state_dict": model.state_dict(),
        },
        path,
    )


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _is_string_sequence(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, str) and bool(item) for item in value)


__all__ = [
    "A2FusionModel",
    "TorchConcatMLP",
    "TorchTrainingConfig",
    "TrainingResult",
    "build_a2_model_from_config",
    "parameter_parity_report",
    "prepare_a2_train_val_samples",
    "train_torch_model",
    "trainable_parameter_count",
]
