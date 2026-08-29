from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from gf.dl.contracts import UnifiedBatch


A2M_MODEL_IDS = ("A2M-MLP", "A2M-RESNET", "A2M-FTT")
A2M_MODEL_SCHEMA_VERSION = "gf-a2m-model-1"
EXPECTED_SENSOR_IDS = (
    "ultrasonic_tof",
    "thermal_conductivity_voltage",
    "ndir_co2_voltage",
)
EXPECTED_SENSOR_TYPES = ("acoustic_tof", "thermal_conductivity", "ndir")


class A2MMLP(nn.Module):
    """Ordered scalar-concat MLP used as the A2M primary control."""

    def __init__(
        self,
        *,
        sensor_count: int,
        hidden_dim: int,
        output_dim: int,
        sensor_ids: Sequence[str] = EXPECTED_SENSOR_IDS,
        sensor_types: Sequence[str] = EXPECTED_SENSOR_TYPES,
    ) -> None:
        super().__init__()
        _validate_dimensions(sensor_count, hidden_dim, output_dim)
        _validate_sensor_vocabulary(sensor_ids, sensor_types, sensor_count)
        self.sensor_count = sensor_count
        self.output_dim = output_dim
        self.sensor_ids = tuple(sensor_ids)
        self.sensor_types = tuple(sensor_types)
        self.output_scale = 100.0
        self.optimization_loss_scale = 1.0
        self.backbone = nn.Sequential(
            nn.Linear(sensor_count, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: UnifiedBatch) -> torch.Tensor:
        values = _steady_state_values(
            batch,
            sensor_ids=self.sensor_ids,
            sensor_types=self.sensor_types,
        )
        return self.backbone(values) * self.output_scale


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.activation = nn.ReLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.layers(values))


class TabularResNet(nn.Module):
    """Small residual network for a fixed-width tabular input."""

    def __init__(
        self,
        *,
        sensor_count: int,
        width: int,
        residual_blocks: int,
        output_dim: int,
        sensor_ids: Sequence[str] = EXPECTED_SENSOR_IDS,
        sensor_types: Sequence[str] = EXPECTED_SENSOR_TYPES,
    ) -> None:
        super().__init__()
        _validate_dimensions(sensor_count, width, output_dim)
        if residual_blocks <= 0:
            raise ValueError("residual_blocks must be positive")
        _validate_sensor_vocabulary(sensor_ids, sensor_types, sensor_count)
        self.sensor_count = sensor_count
        self.output_dim = output_dim
        self.sensor_ids = tuple(sensor_ids)
        self.sensor_types = tuple(sensor_types)
        self.output_scale = 100.0
        self.optimization_loss_scale = 1.0
        self.input_projection = nn.Sequential(
            nn.Linear(sensor_count, width),
            nn.ReLU(),
        )
        self.residual_blocks = nn.ModuleList(_ResidualBlock(width) for _ in range(residual_blocks))
        self.head = nn.Linear(width, output_dim)

    def forward(self, batch: UnifiedBatch) -> torch.Tensor:
        values = _steady_state_values(
            batch,
            sensor_ids=self.sensor_ids,
            sensor_types=self.sensor_types,
        )
        fused = self.input_projection(values)
        for block in self.residual_blocks:
            fused = block(fused)
        return self.head(fused) * self.output_scale


class FeatureTokenTransformer(nn.Module):
    """A light feature-token Transformer without time or positional encodings."""

    def __init__(
        self,
        *,
        sensor_count: int,
        token_dim: int,
        encoder_blocks: int,
        heads: int,
        output_dim: int,
        sensor_ids: Sequence[str] = EXPECTED_SENSOR_IDS,
        sensor_types: Sequence[str] = EXPECTED_SENSOR_TYPES,
    ) -> None:
        super().__init__()
        _validate_dimensions(sensor_count, token_dim, output_dim)
        if encoder_blocks <= 0:
            raise ValueError("encoder_blocks must be positive")
        if heads <= 0 or token_dim % heads != 0:
            raise ValueError("heads must be positive and divide token_dim")
        _validate_sensor_vocabulary(sensor_ids, sensor_types, sensor_count)
        self.sensor_count = sensor_count
        self.output_dim = output_dim
        self.sensor_ids = tuple(sensor_ids)
        self.sensor_types = tuple(sensor_types)
        self.sensor_id_to_index = {value: index for index, value in enumerate(self.sensor_ids)}
        self.sensor_type_to_index = {value: index for index, value in enumerate(self.sensor_types)}
        self.output_scale = 100.0
        self.optimization_loss_scale = 1.0
        self.scalar_projection = nn.Linear(1, token_dim)
        self.sensor_id_embedding = nn.Embedding(sensor_count, token_dim)
        self.sensor_type_embedding = nn.Embedding(len(self.sensor_type_to_index), token_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_blocks)
        self.normalization = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, output_dim)

    def forward(self, batch: UnifiedBatch) -> torch.Tensor:
        values = _steady_state_values(
            batch,
            sensor_ids=self.sensor_ids,
            sensor_types=self.sensor_types,
        )
        batch_size = values.shape[0]
        device = values.device
        sensor_indices = torch.arange(self.sensor_count, device=device).unsqueeze(0).expand(batch_size, -1)
        registered_type_indices = torch.tensor(
            [self.sensor_type_to_index[sensor_type] for sensor_type in self.sensor_types],
            dtype=torch.long,
            device=device,
        )
        type_indices = registered_type_indices.unsqueeze(0).expand(batch_size, -1)
        tokens = (
            self.scalar_projection(values.unsqueeze(-1))
            + self.sensor_id_embedding(sensor_indices)
            + self.sensor_type_embedding(type_indices)
        )
        class_token = self.class_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat((class_token, tokens), dim=1))
        return self.head(self.normalization(encoded[:, 0])) * self.output_scale


def build_a2m_model(
    model_id: str,
    recipe: Mapping[str, Any],
    *,
    sensor_ids: Sequence[str] = EXPECTED_SENSOR_IDS,
    sensor_types: Sequence[str] = EXPECTED_SENSOR_TYPES,
    output_dim: int = 3,
) -> nn.Module:
    """Build exactly one registered A2M architecture from one recipe."""

    if model_id not in A2M_MODEL_IDS:
        raise ValueError(f"unsupported A2M model id: {model_id!r}")
    if not isinstance(recipe, Mapping):
        raise ValueError("A2M recipe must be an object")
    sensor_count = len(sensor_ids)
    if len(sensor_types) != sensor_count:
        raise ValueError("sensor_ids and sensor_types must have equal length")
    if model_id == "A2M-MLP":
        hidden_dim = _positive_int(recipe, "hidden_dim")
        return A2MMLP(
            sensor_count=sensor_count,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            sensor_ids=sensor_ids,
            sensor_types=sensor_types,
        )
    if model_id == "A2M-RESNET":
        width = _positive_int(recipe, "width")
        residual_blocks = _positive_int(recipe, "residual_blocks")
        return TabularResNet(
            sensor_count=sensor_count,
            width=width,
            residual_blocks=residual_blocks,
            output_dim=output_dim,
            sensor_ids=sensor_ids,
            sensor_types=sensor_types,
        )
    token_dim = _positive_int(recipe, "token_dim")
    encoder_blocks = _positive_int(recipe, "encoder_blocks")
    heads = _positive_int(recipe, "heads")
    return FeatureTokenTransformer(
        sensor_count=sensor_count,
        token_dim=token_dim,
        encoder_blocks=encoder_blocks,
        heads=heads,
        output_dim=output_dim,
        sensor_ids=sensor_ids,
        sensor_types=sensor_types,
    )


def validate_a2m_model_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != A2M_MODEL_SCHEMA_VERSION:
        raise ValueError("A2M model schema_version is unsupported")
    model_id = config.get("model_id")
    if model_id not in A2M_MODEL_IDS:
        raise ValueError(f"A2M model_id must be one of {A2M_MODEL_IDS}")
    sensor_ids = config.get("sensor_ids")
    sensor_types = config.get("sensor_types")
    if sensor_ids != list(EXPECTED_SENSOR_IDS) or sensor_types != list(EXPECTED_SENSOR_TYPES):
        raise ValueError("A2M sensor vocabulary must match the frozen three-sensor order")
    input_shape = config.get("input_shape")
    if input_shape != {"timesteps": 1, "features": 1}:
        raise ValueError("A2M input_shape must be steady-state T=1,F=1")
    head = config.get("head")
    if not isinstance(head, Mapping) or head.get("id") != "H0" or head.get("output_dim") != 3:
        raise ValueError("A2M models must use the shared H0 three-target head")
    recipes = config.get("recipes")
    if not isinstance(recipes, list) or len(recipes) != 2:
        raise ValueError("each A2M model must define exactly two recipes")
    names: list[str] = []
    for recipe in recipes:
        if not isinstance(recipe, Mapping) or not isinstance(recipe.get("name"), str) or not recipe["name"]:
            raise ValueError("A2M recipe names must be non-empty strings")
        names.append(recipe["name"])
    if len(set(names)) != len(names):
        raise ValueError("A2M recipe names must be unique")
    if model_id == "A2M-MLP":
        if any(not isinstance(recipe.get("hidden_dim"), int) or recipe["hidden_dim"] <= 0 for recipe in recipes):
            raise ValueError("A2M-MLP recipes require positive hidden_dim")
    elif model_id == "A2M-RESNET":
        if any(
            not isinstance(recipe.get("width"), int)
            or recipe["width"] <= 0
            or recipe.get("residual_blocks") != 2
            for recipe in recipes
        ):
            raise ValueError("A2M-RESNET recipes require positive width and two residual blocks")
    elif any(
        not isinstance(recipe.get("token_dim"), int)
        or recipe["token_dim"] <= 0
        or recipe.get("encoder_blocks") != 1
        or recipe.get("heads") != 2
        for recipe in recipes
    ):
        raise ValueError("A2M-FTT recipes require token_dim, one encoder block, and two heads")


def _steady_state_values(
    batch: UnifiedBatch,
    *,
    sensor_ids: Sequence[str],
    sensor_types: Sequence[str],
) -> torch.Tensor:
    if batch.signals.ndim != 4 or batch.signals.shape[1] != len(sensor_ids):
        raise ValueError("A2M models require signals with shape [B,S,1,1]")
    if tuple(batch.signals.shape[2:]) != (1, 1):
        raise ValueError("A2M models require a real steady-state T=1,F=1 input")
    if batch.valid_mask.shape != batch.signals.shape:
        raise ValueError("valid_mask shape must match signals")
    if batch.sensor_mask.shape != batch.signals.shape[:2]:
        raise ValueError("sensor_mask shape must match the batch sensor axes")
    if not bool(torch.all(batch.sensor_mask)) or not bool(torch.all(batch.valid_mask)):
        raise ValueError("A2M primary models require all registered sensors and scalar values")
    if any(tuple(row[: len(sensor_ids)]) != tuple(sensor_ids) for row in batch.sensor_id):
        raise ValueError("batch sensor_id order does not match the registered A2M order")
    if any(tuple(row[: len(sensor_types)]) != tuple(sensor_types) for row in batch.sensor_type):
        raise ValueError("batch sensor_type order does not match the registered A2M order")
    values = batch.signals[:, :, 0, 0]
    if not bool(torch.isfinite(values).all()):
        raise ValueError("A2M input contains non-finite values")
    return values


def _validate_dimensions(sensor_count: int, width: int, output_dim: int) -> None:
    if sensor_count <= 0 or width <= 0 or output_dim <= 0:
        raise ValueError("sensor_count, width, and output_dim must be positive")


def _validate_sensor_vocabulary(
    sensor_ids: Sequence[str],
    sensor_types: Sequence[str],
    sensor_count: int,
) -> None:
    sensor_ids = tuple(sensor_ids)
    sensor_types = tuple(sensor_types)
    if len(sensor_ids) != sensor_count or len(sensor_types) != sensor_count:
        raise ValueError("sensor vocabulary length must equal sensor_count")
    if any(not value for value in sensor_ids + sensor_types):
        raise ValueError("sensor vocabulary values must be non-empty")
    if len(set(sensor_ids)) != sensor_count:
        raise ValueError("sensor_ids must be unique")


def _positive_int(recipe: Mapping[str, Any], key: str) -> int:
    value = recipe.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"A2M recipe field {key!r} must be a positive integer")
    return value


__all__ = [
    "A2M_MODEL_IDS",
    "A2M_MODEL_SCHEMA_VERSION",
    "A2MMLP",
    "EXPECTED_SENSOR_IDS",
    "EXPECTED_SENSOR_TYPES",
    "FeatureTokenTransformer",
    "TabularResNet",
    "build_a2m_model",
    "validate_a2m_model_config",
]
