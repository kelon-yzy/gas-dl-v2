from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from gf.dl.contracts import UnifiedBatch
from gf.dl.task_heads import build_tqif_task_head


TQIF_MODEL_SCHEMA_VERSION = "gf-tqif-model-1"
TQIF_RECIPE_NAMES = ("tqif_token16_pair16", "tqif_token32_pair32")
TQIF_QUERY_MODES = ("shared", "independent")
TQIF_HEAD_IDS = ("H0", "STR", "VAR_TOTAL")
TQIF_TARGET_SLOT_SCHEMA_VERSION = "gf-tqif-target-slot-1"
TQIF_SENSOR_REGISTRY_SCHEMA_VERSION = "gf-tqif-sensor-registry-1"
TQIF_RECIPE_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "tqif_token16_pair16": MappingProxyType(
            {
                "embedding_dim": 16,
                "token_dim": 16,
                "attention_heads": 2,
                "pair_hidden_dim": 16,
                "pair_rank": 16,
                "query_ffn_dim": 32,
                "sensor_projection_layers": 1,
                "pair_mlp_layers": 2,
                "query_ffn_layers": 2,
                "dropout": 0.0,
                "head_id": "H0",
                "query_mode": "independent",
                "pair_evidence": True,
            }
        ),
        "tqif_token32_pair32": MappingProxyType(
            {
                "embedding_dim": 32,
                "token_dim": 32,
                "attention_heads": 4,
                "pair_hidden_dim": 32,
                "pair_rank": 32,
                "query_ffn_dim": 64,
                "sensor_projection_layers": 1,
                "pair_mlp_layers": 2,
                "query_ffn_layers": 2,
                "dropout": 0.0,
                "head_id": "H0",
                "query_mode": "independent",
                "pair_evidence": True,
            }
        ),
    }
)


@dataclass(frozen=True)
class TQIFTargetSlot:
    """Immutable target-slot adapter entry; the fusion core only consumes slot IDs."""

    slot_id: str
    output_name: str
    value_range: tuple[float, float]
    loss_weight: float

    def __post_init__(self) -> None:
        if not self.slot_id or not self.output_name:
            raise ValueError("target slot IDs and output names must be non-empty")
        if len(self.value_range) != 2:
            raise ValueError("target slot value_range must contain [min, max]")
        lower, upper = (float(value) for value in self.value_range)
        if not torch.isfinite(torch.tensor([lower, upper])).all() or lower >= upper:
            raise ValueError("target slot value_range must be finite and increasing")
        if not torch.isfinite(torch.tensor(float(self.loss_weight))) or self.loss_weight <= 0.0:
            raise ValueError("target slot loss_weight must be finite and positive")
        object.__setattr__(self, "value_range", (lower, upper))
        object.__setattr__(self, "loss_weight", float(self.loss_weight))

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "output_name": self.output_name,
            "value_range": list(self.value_range),
            "loss_weight": self.loss_weight,
        }


@dataclass(frozen=True)
class TQIFTargetSlotRegistry:
    """Ordered, immutable target-slot registry used by the checkpoint contract."""

    slots: tuple[TQIFTargetSlot, ...]

    def __post_init__(self) -> None:
        slots = tuple(self.slots)
        if not slots:
            raise ValueError("target slot registry must not be empty")
        if len({slot.slot_id for slot in slots}) != len(slots):
            raise ValueError("target slot IDs must be unique")
        object.__setattr__(self, "slots", slots)

    @classmethod
    def from_mappings(cls, values: Sequence[Mapping[str, Any]]) -> "TQIFTargetSlotRegistry":
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError("target_slot_registry must be a list")
        if all(isinstance(value, TQIFTargetSlot) for value in values):
            return cls(tuple(values))
        slots: list[TQIFTargetSlot] = []
        for value in values:
            if not isinstance(value, Mapping):
                raise ValueError("each target slot must be an object")
            required = {"slot_id", "output_name", "value_range", "loss_weight"}
            if set(value) != required:
                raise ValueError(
                    "target slot entries must contain exactly "
                    "slot_id, output_name, value_range, and loss_weight"
                )
            raw_range = value["value_range"]
            if not isinstance(raw_range, Sequence) or isinstance(raw_range, (str, bytes)):
                raise ValueError("target slot value_range must be a list")
            if len(raw_range) != 2:
                raise ValueError("target slot value_range must contain [min, max]")
            slots.append(
                TQIFTargetSlot(
                    slot_id=str(value["slot_id"]),
                    output_name=str(value["output_name"]),
                    value_range=(float(raw_range[0]), float(raw_range[1])),
                    loss_weight=float(value["loss_weight"]),
                )
            )
        return cls(tuple(slots))

    @classmethod
    def from_ids(cls, values: Sequence[str]) -> "TQIFTargetSlotRegistry":
        ids = _validate_unique_strings(values, "target_slot_ids")
        return cls(
            tuple(
                TQIFTargetSlot(
                    slot_id=slot_id,
                    output_name=slot_id,
                    value_range=(0.0, 100.0),
                    loss_weight=1.0,
                )
                for slot_id in ids
            )
        )

    @property
    def slot_ids(self) -> tuple[str, ...]:
        return tuple(slot.slot_id for slot in self.slots)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(slot.output_name for slot in self.slots)

    def to_list(self) -> list[dict[str, Any]]:
        return [slot.to_dict() for slot in self.slots]


@dataclass(frozen=True)
class TQIFSensorSpec:
    sensor_id: str
    sensor_type: str

    def __post_init__(self) -> None:
        if not self.sensor_id or not self.sensor_type:
            raise ValueError("sensor_id and sensor_type must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"sensor_id": self.sensor_id, "sensor_type": self.sensor_type}


@dataclass(frozen=True)
class TQIFSensorEncoding:
    sensor_embeddings: torch.Tensor
    sensor_mask: torch.Tensor
    sensor_id_indices: torch.Tensor
    sensor_type_indices: torch.Tensor
    quality: torch.Tensor


@dataclass(frozen=True)
class TQIFDiagnostics:
    representation: torch.Tensor
    sensor_attention: torch.Tensor
    pair_attention: torch.Tensor
    gate: torch.Tensor
    pair_tokens: torch.Tensor
    pair_mask: torch.Tensor


@dataclass(frozen=True)
class TQIFModelDiagnostics:
    prediction: torch.Tensor
    fusion: TQIFDiagnostics


class TQIFScalarSensorEncoder(nn.Module):
    """Encode steady-state scalar observations with physical sensor metadata."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        sensor_ids: Sequence[str],
        sensor_types: Sequence[str],
        use_quality: bool = False,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.sensor_ids = _validate_unique_strings(sensor_ids, "sensor_ids")
        self.sensor_types = _validate_sensor_types(sensor_types, len(self.sensor_ids))
        self.sensor_registry = tuple(
            TQIFSensorSpec(sensor_id, sensor_type)
            for sensor_id, sensor_type in zip(self.sensor_ids, self.sensor_types, strict=True)
        )
        self.sensor_id_to_index = {
            value: index for index, value in enumerate(self.sensor_ids)
        }
        self.sensor_id_to_type = MappingProxyType(
            {
                spec.sensor_id: spec.sensor_type
                for spec in self.sensor_registry
            }
        )
        type_vocabulary = tuple(dict.fromkeys(self.sensor_types))
        self.sensor_type_vocabulary = type_vocabulary
        self.sensor_type_to_index = {
            value: index for index, value in enumerate(type_vocabulary)
        }
        self.embedding_dim = embedding_dim
        self.use_quality = bool(use_quality)
        self.observation_projection = nn.Linear(1, embedding_dim)
        self.sensor_id_embedding = nn.Embedding(len(self.sensor_ids), embedding_dim)
        self.sensor_type_embedding = nn.Embedding(len(type_vocabulary), embedding_dim)
        self.quality_projection = nn.Linear(1, embedding_dim) if use_quality else None
        self.normalization = nn.LayerNorm(embedding_dim)

    def forward(self, batch: UnifiedBatch) -> TQIFSensorEncoding:
        if batch.signals.ndim != 4:
            raise ValueError("TQIF scalar encoder requires signals with shape [B,S,T,F]")
        if batch.valid_mask.shape != batch.signals.shape:
            raise ValueError("valid_mask shape must match signals")
        if batch.quality.shape != batch.signals.shape[:3]:
            raise ValueError("quality shape must match [B,S,T]")
        if batch.sensor_mask.shape != batch.signals.shape[:2]:
            raise ValueError("sensor_mask shape must match [B,S]")
        if not torch.isfinite(batch.signals[batch.valid_mask]).all():
            raise ValueError("valid sensor observations must be finite")
        if not torch.isfinite(batch.quality).all():
            raise ValueError("sensor quality must be finite")

        valid = batch.valid_mask
        counts = valid.sum(dim=(-1, -2))
        active = batch.sensor_mask & (counts > 0)
        if not torch.any(active, dim=1).all():
            raise ValueError("each sample must contain at least one valid sensor")
        if torch.any((counts > 0) & ~batch.sensor_mask):
            raise ValueError("valid observations cannot be stored in a masked sensor slot")

        safe_counts = counts.clamp_min(1).to(batch.signals.dtype)
        observation = (batch.signals * valid).sum(dim=(-1, -2)) / safe_counts
        id_indices, type_indices = self._metadata_indices(batch)
        tokens = (
            self.observation_projection(observation.unsqueeze(-1))
            + self.sensor_id_embedding(id_indices)
            + self.sensor_type_embedding(type_indices)
        )

        valid_time = valid.any(dim=-1)
        time_counts = valid_time.sum(dim=-1).to(batch.quality.dtype)
        safe_time_counts = time_counts.clamp_min(1.0)
        quality = (batch.quality * valid_time).sum(dim=-1) / safe_time_counts
        if self.quality_projection is not None:
            tokens = tokens + self.quality_projection(quality.unsqueeze(-1))

        tokens = self.normalization(tokens) * active.unsqueeze(-1)
        quality = quality * active.to(quality.dtype)
        return TQIFSensorEncoding(
            sensor_embeddings=tokens,
            sensor_mask=active,
            sensor_id_indices=id_indices,
            sensor_type_indices=type_indices,
            quality=quality,
        )

    def _metadata_indices(self, batch: UnifiedBatch) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sensor_count = batch.sensor_mask.shape
        id_indices = torch.zeros(
            (batch_size, sensor_count), dtype=torch.long, device=batch.signals.device
        )
        type_indices = torch.zeros_like(id_indices)
        for batch_index in range(batch_size):
            for sensor_index in range(sensor_count):
                if not bool(batch.sensor_mask[batch_index, sensor_index]):
                    continue
                sensor_id = batch.sensor_id[batch_index][sensor_index]
                sensor_type = batch.sensor_type[batch_index][sensor_index]
                if sensor_id not in self.sensor_id_to_index:
                    raise KeyError(f"unknown sensor_id {sensor_id!r}")
                expected_type = self.sensor_id_to_type[sensor_id]
                if sensor_type != expected_type:
                    raise ValueError(
                        f"sensor_type mismatch for sensor_id {sensor_id!r}: "
                        f"expected {expected_type!r}, got {sensor_type!r}"
                    )
                id_indices[batch_index, sensor_index] = self.sensor_id_to_index[sensor_id]
                type_indices[batch_index, sensor_index] = self.sensor_type_to_index[sensor_type]
        return id_indices, type_indices


class SensorCapacityControl(nn.Module):
    """Residual width control used by no-pair ablation counterparts."""

    def __init__(self, token_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if token_dim <= 0 or hidden_dim <= 0:
            raise ValueError("capacity control dimensions must be positive")
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.normalization = nn.LayerNorm(token_dim)
        self.input_projection = nn.Linear(token_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, token_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.output_projection(
            torch.nn.functional.gelu(self.input_projection(self.normalization(values)))
        )


class TQIFFusionCore(nn.Module):
    """Target-query fusion over sensor tokens and shared unordered sensor pairs."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        token_dim: int,
        pair_hidden_dim: int,
        query_ffn_dim: int,
        attention_heads: int,
        sensor_type_count: int,
        target_count: int,
        query_mode: str = "independent",
        use_pair: bool = True,
        target_slot_ids: Sequence[str] | None = None,
        capacity_control_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_positive_dimensions(
            embedding_dim,
            token_dim,
            pair_hidden_dim,
            query_ffn_dim,
            attention_heads,
            sensor_type_count,
            target_count,
        )
        if token_dim % attention_heads != 0:
            raise ValueError("attention_heads must divide token_dim")
        if query_mode not in TQIF_QUERY_MODES:
            raise ValueError(f"query_mode must be one of {TQIF_QUERY_MODES}")
        if not isinstance(dropout, (int, float)) or isinstance(dropout, bool):
            raise ValueError("dropout must be numeric")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not use_pair and (
            not isinstance(capacity_control_hidden_dim, int)
            or isinstance(capacity_control_hidden_dim, bool)
            or capacity_control_hidden_dim <= 0
        ):
            raise ValueError(
                "no-pair fusion requires a positive capacity_control_hidden_dim"
            )
        if use_pair and capacity_control_hidden_dim is not None:
            raise ValueError("pair fusion must not register a no-pair capacity control")

        if target_slot_ids is None:
            target_slots = tuple(f"slot_{index}" for index in range(target_count))
        else:
            target_slots = _validate_unique_strings(target_slot_ids, "target_slot_ids")
            if len(target_slots) != target_count:
                raise ValueError("target_slot_ids length must equal target_count")

        self.embedding_dim = embedding_dim
        self.token_dim = token_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.query_ffn_dim = query_ffn_dim
        self.attention_heads = attention_heads
        self.sensor_type_count = sensor_type_count
        self.target_count = target_count
        self.query_mode = query_mode
        self.use_pair = bool(use_pair)
        self.target_slot_ids = target_slots
        self.target_slot_vocabulary = tuple(sorted(target_slots))
        target_slot_indices = torch.tensor(
            [self.target_slot_vocabulary.index(slot_id) for slot_id in target_slots],
            dtype=torch.long,
        )
        self.register_buffer("target_slot_indices", target_slot_indices, persistent=True)
        self.dropout = float(dropout)

        self.sensor_projection = nn.Linear(embedding_dim, token_dim)
        self.sensor_normalization = nn.LayerNorm(token_dim)
        if query_mode == "shared":
            self.shared_target_query = nn.Parameter(torch.empty(1, token_dim))
            nn.init.normal_(self.shared_target_query, mean=0.0, std=0.02)
            self.target_slot_embedding = None
        else:
            self.shared_target_query = None
            self.target_slot_embedding = nn.Embedding(
                len(self.target_slot_vocabulary), token_dim
            )
            nn.init.normal_(self.target_slot_embedding.weight, mean=0.0, std=0.02)
        self.sensor_attention = nn.MultiheadAttention(
            token_dim,
            attention_heads,
            dropout=self.dropout,
            batch_first=True,
        )

        if self.use_pair:
            pair_type_count = sensor_type_count * (sensor_type_count + 1) // 2
            self.pair_type_embedding = nn.Embedding(pair_type_count, token_dim)
            self.pair_projection = nn.Sequential(
                nn.Linear(token_dim * 4, pair_hidden_dim),
                nn.GELU(),
                nn.Linear(pair_hidden_dim, token_dim),
                nn.LayerNorm(token_dim),
            )
            self.pair_attention = nn.MultiheadAttention(
                token_dim,
                attention_heads,
                dropout=self.dropout,
                batch_first=True,
            )
            self.gate_projection = nn.Linear(token_dim * 3, token_dim)
            self.capacity_control = None
        else:
            self.pair_type_embedding = None
            self.pair_projection = None
            self.pair_attention = None
            self.gate_projection = None
            self.capacity_control = SensorCapacityControl(
                token_dim,
                int(capacity_control_hidden_dim),
            )

        self.query_ffn = nn.Sequential(
            nn.Linear(token_dim, query_ffn_dim),
            nn.GELU(),
            nn.Linear(query_ffn_dim, token_dim),
        )
        self.output_normalization = nn.LayerNorm(token_dim)

    def forward(
        self,
        sensor_embeddings: torch.Tensor,
        sensor_mask: torch.Tensor,
        sensor_type_indices: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | TQIFDiagnostics:
        return self._forward(
            sensor_embeddings,
            sensor_mask,
            sensor_type_indices,
            collect_diagnostics=return_diagnostics,
        )

    def forward_with_diagnostics(
        self,
        sensor_embeddings: torch.Tensor,
        sensor_mask: torch.Tensor,
        sensor_type_indices: torch.Tensor,
    ) -> TQIFDiagnostics:
        diagnostics = self._forward(
            sensor_embeddings,
            sensor_mask,
            sensor_type_indices,
            collect_diagnostics=True,
        )
        if not isinstance(diagnostics, TQIFDiagnostics):
            raise RuntimeError("diagnostic fusion path returned a prediction tensor")
        return diagnostics

    def _forward(
        self,
        sensor_embeddings: torch.Tensor,
        sensor_mask: torch.Tensor,
        sensor_type_indices: torch.Tensor,
        *,
        collect_diagnostics: bool,
    ) -> torch.Tensor | TQIFDiagnostics:
        self._validate_inputs(sensor_embeddings, sensor_mask, sensor_type_indices)
        tokens = self.sensor_normalization(self.sensor_projection(sensor_embeddings))
        if self.capacity_control is not None:
            tokens = self.capacity_control(tokens)
        tokens = tokens * sensor_mask.unsqueeze(-1)
        batch_size = tokens.shape[0]
        queries = self._target_queries()
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        sensor_evidence, sensor_attention = _masked_attention(
            self.sensor_attention,
            queries,
            tokens,
            sensor_mask,
            collect_weights=collect_diagnostics,
        )

        if self.use_pair:
            pair_tokens, pair_mask = self._build_pair_tokens(
                tokens,
                sensor_mask,
                sensor_type_indices,
            )
            pair_queries = queries + sensor_evidence
            pair_evidence, pair_attention = _masked_attention(
                self.pair_attention,
                pair_queries,
                pair_tokens,
                pair_mask,
                collect_weights=collect_diagnostics,
            )
            pair_available = pair_mask.any(dim=1).view(batch_size, 1, 1)
            gate = torch.sigmoid(
                self.gate_projection(
                    torch.cat((queries, sensor_evidence, pair_evidence), dim=-1)
                )
            ) * pair_available.to(pair_evidence.dtype)
        else:
            pair_tokens = tokens.new_zeros((batch_size, 0, self.token_dim))
            pair_mask = sensor_mask.new_zeros((batch_size, 0))
            pair_evidence = tokens.new_zeros(
                (batch_size, self.target_count, self.token_dim)
            )
            pair_attention = (
                tokens.new_zeros((batch_size, self.attention_heads, self.target_count, 0))
                if collect_diagnostics
                else None
            )
            gate = tokens.new_zeros(
                (batch_size, self.target_count, self.token_dim)
            )

        representation = self.output_normalization(
            queries + sensor_evidence + gate * pair_evidence
        )
        representation = self.output_normalization(
            representation + self.query_ffn(representation)
        )
        if not collect_diagnostics:
            return representation
        if sensor_attention is None or pair_attention is None:
            raise RuntimeError("diagnostic attention weights were not collected")
        return TQIFDiagnostics(
            representation=representation,
            sensor_attention=sensor_attention,
            pair_attention=pair_attention,
            gate=gate,
            pair_tokens=pair_tokens,
            pair_mask=pair_mask,
        )

    def _target_queries(self) -> torch.Tensor:
        if self.query_mode == "shared":
            if self.shared_target_query is None:
                raise RuntimeError("shared target query is unavailable")
            return self.shared_target_query.expand(self.target_count, -1)
        if self.target_slot_embedding is None:
            raise RuntimeError("target slot embedding is unavailable")
        return self.target_slot_embedding(self.target_slot_indices)

    def _build_pair_tokens(
        self,
        sensor_tokens: torch.Tensor,
        sensor_mask: torch.Tensor,
        sensor_type_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sensor_count, token_dim = sensor_tokens.shape
        if sensor_count < 2:
            return (
                sensor_tokens.new_zeros((batch_size, 0, token_dim)),
                sensor_mask.new_zeros((batch_size, 0)),
            )
        if self.pair_projection is None or self.pair_type_embedding is None:
            raise RuntimeError("pair modules are unavailable when use_pair is false")

        pair_indices = torch.triu_indices(
            sensor_count,
            sensor_count,
            offset=1,
            device=sensor_tokens.device,
        )
        left_index, right_index = pair_indices[0], pair_indices[1]
        left = sensor_tokens[:, left_index, :]
        right = sensor_tokens[:, right_index, :]
        left_type = sensor_type_indices[:, left_index]
        right_type = sensor_type_indices[:, right_index]
        pair_mask = sensor_mask[:, left_index] & sensor_mask[:, right_index]
        type_low = torch.minimum(left_type, right_type)
        type_high = torch.maximum(left_type, right_type)
        pair_type_index = _unordered_pair_type_index(
            type_low,
            type_high,
            self.sensor_type_count,
        )
        pair_type_embedding = self.pair_type_embedding(pair_type_index)
        pair_input = torch.cat(
            (left + right, (left - right).abs(), left * right, pair_type_embedding),
            dim=-1,
        )
        pair_tokens = self.pair_projection(pair_input)
        pair_tokens = pair_tokens * pair_mask.unsqueeze(-1)
        return pair_tokens, pair_mask

    def _validate_inputs(
        self,
        sensor_embeddings: torch.Tensor,
        sensor_mask: torch.Tensor,
        sensor_type_indices: torch.Tensor,
    ) -> None:
        if sensor_embeddings.ndim != 3:
            raise ValueError("sensor_embeddings must have shape [B,S,D]")
        if sensor_embeddings.shape[-1] != self.embedding_dim:
            raise ValueError("sensor_embeddings width does not match embedding_dim")
        if sensor_mask.shape != sensor_embeddings.shape[:2]:
            raise ValueError("sensor_mask shape must match [B,S]")
        if sensor_mask.dtype != torch.bool:
            raise ValueError("sensor_mask must be boolean")
        if sensor_type_indices.shape != sensor_embeddings.shape[:2]:
            raise ValueError("sensor_type_indices shape must match [B,S]")
        if sensor_type_indices.dtype != torch.long:
            raise ValueError("sensor_type_indices must be torch.long")
        if not torch.isfinite(sensor_embeddings).all():
            raise ValueError("sensor_embeddings must be finite")
        if not torch.any(sensor_mask, dim=1).all():
            raise ValueError("each sample must contain at least one valid sensor")
        valid_type = (sensor_type_indices >= 0) & (
            sensor_type_indices < self.sensor_type_count
        )
        if torch.any(sensor_mask & ~valid_type):
            raise ValueError("sensor_type_indices contain an unknown sensor type")


class TQIFModel(nn.Module):
    """A2 scalar-token adapter plus the dataset-agnostic TQIF fusion core."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        token_dim: int,
        pair_hidden_dim: int,
        query_ffn_dim: int,
        attention_heads: int,
        output_dim: int,
        sensor_ids: Sequence[str],
        sensor_types: Sequence[str],
        target_slot_ids: Sequence[str] | None = None,
        target_slot_registry: TQIFTargetSlotRegistry | Sequence[Mapping[str, Any]] | None = None,
        head_id: str = "H0",
        query_mode: str = "independent",
        use_pair: bool = True,
        use_quality: bool = False,
        total: float = 100.0,
        temperature: float = 1.0,
        total_hidden_dim: int | None = None,
        capacity_control_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        registry = _coerce_target_slot_registry(
            target_slot_registry,
            target_slot_ids,
            output_dim=output_dim,
        )
        target_slots = registry.slot_ids
        if head_id not in TQIF_HEAD_IDS:
            raise ValueError(f"head_id must be one of {TQIF_HEAD_IDS}")
        if query_mode not in TQIF_QUERY_MODES:
            raise ValueError(f"query_mode must be one of {TQIF_QUERY_MODES}")
        if head_id != "H0" and query_mode != "independent":
            raise ValueError("STR and VAR_TOTAL require independent target queries")

        self.output_dim = output_dim
        self.target_slot_ids = target_slots
        self.target_slot_registry = registry
        self.head_id = head_id
        self.query_mode = query_mode
        self.use_pair = bool(use_pair)
        self.optimization_loss_scale = 1.0
        self.encoder = TQIFScalarSensorEncoder(
            embedding_dim=embedding_dim,
            sensor_ids=sensor_ids,
            sensor_types=sensor_types,
            use_quality=use_quality,
        )
        self.fusion = TQIFFusionCore(
            embedding_dim=embedding_dim,
            token_dim=token_dim,
            pair_hidden_dim=pair_hidden_dim,
            query_ffn_dim=query_ffn_dim,
            attention_heads=attention_heads,
            sensor_type_count=len(self.encoder.sensor_type_vocabulary),
            target_count=output_dim,
            query_mode=query_mode,
            use_pair=use_pair,
            target_slot_ids=target_slots,
            capacity_control_hidden_dim=capacity_control_hidden_dim,
            dropout=dropout,
        )
        head_config: dict[str, object] = {
            "id": head_id,
            "total": total,
            "temperature": temperature,
            "total_hidden_dim": total_hidden_dim,
            "shared_query": query_mode == "shared",
        }
        self.head = build_tqif_task_head(
            head_config,
            input_dim=token_dim,
            target_count=output_dim,
        )

    def forward(self, batch: UnifiedBatch) -> torch.Tensor:
        encoding = self.encoder(batch)
        representation = self.fusion(
            encoding.sensor_embeddings,
            encoding.sensor_mask,
            encoding.sensor_type_indices,
        )
        return self._apply_head(representation)

    def forward_with_diagnostics(self, batch: UnifiedBatch) -> TQIFModelDiagnostics:
        encoding = self.encoder(batch)
        fusion = self.fusion.forward_with_diagnostics(
            encoding.sensor_embeddings,
            encoding.sensor_mask,
            encoding.sensor_type_indices,
        )
        return TQIFModelDiagnostics(
            prediction=self._apply_head(fusion.representation),
            fusion=fusion,
        )

    def _apply_head(self, representation: torch.Tensor) -> torch.Tensor:
        if self.query_mode == "shared":
            return self.head(representation[:, 0, :])
        return self.head(representation)

    def checkpoint_contract(self) -> dict[str, Any]:
        return {
            "schema_version": TQIF_MODEL_SCHEMA_VERSION,
            "target_slot_ids": list(self.target_slot_ids),
            "target_slot_hash": target_slot_registry_hash(self.target_slot_registry),
            "sensor_registry": [
                spec.to_dict() for spec in self.encoder.sensor_registry
            ],
            "sensor_registry_hash": sensor_registry_hash(self.encoder.sensor_registry),
        }


class MatchedConcatMLP(nn.Module):
    """Ordered concat control using the same scalar sensor encoder as TQIF."""

    def __init__(
        self,
        *,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        sensor_ids: Sequence[str],
        sensor_types: Sequence[str],
        use_quality: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or output_dim <= 0:
            raise ValueError("hidden_dim and output_dim must be positive")
        self.encoder = TQIFScalarSensorEncoder(
            embedding_dim=embedding_dim,
            sensor_ids=sensor_ids,
            sensor_types=sensor_types,
            use_quality=use_quality,
        )
        self.sensor_count = len(self.encoder.sensor_ids)
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.target_slot_registry = TQIFTargetSlotRegistry.from_ids(
            tuple(f"slot_{index}" for index in range(output_dim))
        )
        self.optimization_loss_scale = 1.0
        self.backbone = nn.Sequential(
            nn.Linear(self.sensor_count * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: UnifiedBatch) -> torch.Tensor:
        encoding = self.encoder(batch)
        if encoding.sensor_embeddings.shape[1] > self.sensor_count:
            raise ValueError("batch contains more sensors than the matched concat registry")
        values = encoding.sensor_embeddings * encoding.sensor_mask.unsqueeze(-1)
        if values.shape[1] < self.sensor_count:
            padding = values.new_zeros(
                values.shape[0],
                self.sensor_count - values.shape[1],
                values.shape[2],
            )
            values = torch.cat((values, padding), dim=1)
        return self.backbone(values.reshape(values.shape[0], -1))

    def checkpoint_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "gf-tqif-control-1",
            "target_slot_ids": list(self.target_slot_registry.slot_ids),
            "target_slot_hash": target_slot_registry_hash(self.target_slot_registry),
            "sensor_registry": [
                spec.to_dict() for spec in self.encoder.sensor_registry
            ],
            "sensor_registry_hash": sensor_registry_hash(self.encoder.sensor_registry),
        }


def build_tqif_model(config: Mapping[str, Any]) -> TQIFModel:
    validate_tqif_model_config(config)
    if config.get("model_id") != "TQIF":
        raise ValueError("build_tqif_model requires model_id='TQIF'")
    head = _required_mapping(config, "head")
    return TQIFModel(
        embedding_dim=int(config.get("embedding_dim", config["token_dim"])),
        token_dim=int(config["token_dim"]),
        pair_hidden_dim=int(config["pair_hidden_dim"]),
        query_ffn_dim=int(config["query_ffn_dim"]),
        attention_heads=int(config["attention_heads"]),
        output_dim=int(head["output_dim"]),
        sensor_ids=config["sensor_ids"],
        sensor_types=config["sensor_types"],
        target_slot_registry=TQIFTargetSlotRegistry.from_mappings(
            config["target_slot_registry"]
        ),
        head_id=str(head["id"]),
        query_mode=str(config["query_mode"]),
        use_pair=bool(config["pair_evidence"]),
        use_quality=bool(config.get("uses_quality", False)),
        total=float(head.get("total", 100.0)),
        temperature=float(head.get("temperature", 1.0)),
        total_hidden_dim=(
            int(head["total_hidden_dim"])
            if head.get("total_hidden_dim") is not None
            else None
        ),
        capacity_control_hidden_dim=(
            int(config["capacity_control_hidden_dim"])
            if config.get("capacity_control_hidden_dim") is not None
            else None
        ),
        dropout=float(config.get("dropout", 0.0)),
    )


def build_tqif_model_from_config(config: Mapping[str, Any]) -> TQIFModel:
    """Compatibility alias matching the existing A2 model builder naming."""

    return build_tqif_model(config)


def build_tqif_matched_concat_model(config: Mapping[str, Any]) -> MatchedConcatMLP:
    validate_tqif_model_config(config)
    if config.get("model_id") != "TQIF-MATCHED-CONCAT":
        raise ValueError("matched concat builder requires model_id='TQIF-MATCHED-CONCAT'")
    return MatchedConcatMLP(
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        output_dim=int(config["output_dim"]),
        sensor_ids=config["sensor_ids"],
        sensor_types=config["sensor_types"],
        use_quality=bool(config.get("uses_quality", False)),
    )


def build_tqif_matched_concat_model_from_config(
    config: Mapping[str, Any],
) -> MatchedConcatMLP:
    """Compatibility alias for the registered matched concat builder."""

    return build_tqif_matched_concat_model(config)


def validate_tqif_model_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != TQIF_MODEL_SCHEMA_VERSION:
        raise ValueError("TQIF model schema_version is unsupported")
    model_id = config.get("model_id")
    if model_id not in {"TQIF", "TQIF-MATCHED-CONCAT"}:
        raise ValueError("TQIF model_id must be TQIF or TQIF-MATCHED-CONCAT")
    sensor_ids = config.get("sensor_ids")
    sensor_types = config.get("sensor_types")
    if not isinstance(sensor_ids, list) or not isinstance(sensor_types, list):
        raise ValueError("sensor_ids and sensor_types must be lists")
    _validate_unique_strings(sensor_ids, "sensor_ids")
    _validate_sensor_types(sensor_types, len(sensor_ids))
    if config.get("input_shape") != {"timesteps": 1, "features": 1}:
        raise ValueError("TQIF input_shape must be steady-state T=1,F=1")
    for key in ("uses_dataset_id", "uses_target_name", "uses_time"):
        if config.get(key) is not False:
            raise ValueError(f"TQIF {key} must be false")
    if config.get("parameter_match_tolerance") != 0.10:
        raise ValueError("TQIF parameter_match_tolerance must be 0.10")

    if model_id == "TQIF-MATCHED-CONCAT":
        _require_positive_int(config, "embedding_dim")
        _require_positive_int(config, "hidden_dim")
        _require_positive_int(config, "output_dim")
        recipe = config.get("recipe")
        if recipe not in TQIF_RECIPE_NAMES:
            raise ValueError(f"matched concat recipe must be one of {TQIF_RECIPE_NAMES}")
        expected_embedding = int(TQIF_RECIPE_SPECS[str(recipe)]["embedding_dim"])
        if int(config["embedding_dim"]) != expected_embedding:
            raise ValueError("matched concat embedding_dim does not match its recipe")
        if config.get("matched_for") != recipe:
            raise ValueError("matched concat matched_for must equal recipe")
        return

    recipe = config.get("recipe")
    if recipe not in TQIF_RECIPE_NAMES:
        raise ValueError(f"TQIF recipe must be one of {TQIF_RECIPE_NAMES}")
    recipe_spec = TQIF_RECIPE_SPECS[str(recipe)]
    for key in (
        "embedding_dim",
        "token_dim",
        "attention_heads",
        "pair_hidden_dim",
        "pair_rank",
        "query_ffn_dim",
        "sensor_projection_layers",
        "pair_mlp_layers",
        "query_ffn_layers",
    ):
        _require_positive_int(config, key)
        if int(config[key]) != int(recipe_spec[key]):
            raise ValueError(f"{key} does not match frozen recipe {recipe!r}")
    dropout = config.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or float(dropout) != float(recipe_spec["dropout"]):
        raise ValueError(f"dropout does not match frozen recipe {recipe!r}")
    if config.get("query_mode") != recipe_spec["query_mode"]:
        raise ValueError(f"query_mode does not match frozen recipe {recipe!r}")
    if config.get("pair_evidence") != recipe_spec["pair_evidence"]:
        raise ValueError(f"pair_evidence does not match frozen recipe {recipe!r}")
    if int(config["token_dim"]) % int(config["attention_heads"]) != 0:
        raise ValueError("attention_heads must divide token_dim")
    if not isinstance(config.get("pair_evidence"), bool):
        raise ValueError("pair_evidence must be boolean")
    if config.get("query_mode") not in TQIF_QUERY_MODES:
        raise ValueError(f"query_mode must be one of {TQIF_QUERY_MODES}")
    head = config.get("head")
    if not isinstance(head, Mapping):
        raise ValueError("TQIF head must be an object")
    if head.get("id") != recipe_spec["head_id"]:
        raise ValueError(f"head.id does not match frozen recipe {recipe!r}")
    if head.get("id") not in TQIF_HEAD_IDS:
        raise ValueError(f"TQIF head id must be one of {TQIF_HEAD_IDS}")
    _require_positive_int(head, "output_dim")
    if set(head) - {"id", "kind", "constraint", "output_dim", "total", "temperature", "total_hidden_dim"}:
        raise ValueError("TQIF head contains unsupported fields")
    if int(head["output_dim"]) != 3:
        raise ValueError("A2 TQIF target head must expose exactly three slots")
    target_registry = config.get("target_slot_registry")
    if not isinstance(target_registry, list):
        raise ValueError("target_slot_registry must be a list")
    registry = TQIFTargetSlotRegistry.from_mappings(target_registry)
    if len(registry.slots) != int(head["output_dim"]):
        raise ValueError("target_slot_registry length must equal head.output_dim")
    if registry.slot_ids != ("slot_0", "slot_1", "slot_2"):
        raise ValueError("A2 TQIF target slot IDs are frozen to slot_0, slot_1, slot_2")
    if config.get("target_slot_ids") is not None:
        raise ValueError("target_slot_ids is deprecated; use target_slot_registry")
    if head.get("id") != "H0" and config.get("query_mode") != "independent":
        raise ValueError("STR and VAR_TOTAL require independent target queries")
    if head.get("id") == "STR":
        total = head.get("total", 100.0)
        temperature = head.get("temperature", 1.0)
        if not isinstance(total, (int, float)) or isinstance(total, bool) or total <= 0:
            raise ValueError("TQIF STR total must be positive")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature <= 0:
            raise ValueError("TQIF STR temperature must be positive")
    if head.get("id") == "VAR_TOTAL" and head.get("total_hidden_dim") is not None:
        _require_positive_int(head, "total_hidden_dim")
    matched = config.get("matched_concat")
    if not isinstance(matched, Mapping):
        raise ValueError("TQIF config must register a matched_concat mapping")
    _require_positive_int(matched, "embedding_dim")
    _require_positive_int(matched, "hidden_dim")
    if config.get("capacity_control_hidden_dim") is not None:
        raise ValueError("pair-enabled TQIF recipes must not register capacity control")


def target_slot_registry_hash(registry: TQIFTargetSlotRegistry) -> str:
    return _canonical_sha256(
        {
            "schema_version": TQIF_TARGET_SLOT_SCHEMA_VERSION,
            "slots": registry.to_list(),
        }
    )


def sensor_registry_hash(registry: Sequence[TQIFSensorSpec]) -> str:
    return _canonical_sha256(
        {
            "schema_version": TQIF_SENSOR_REGISTRY_SCHEMA_VERSION,
            "sensors": [spec.to_dict() for spec in registry],
        }
    )


def validate_tqif_checkpoint_payload(
    model: TQIFModel,
    payload: Mapping[str, Any],
) -> None:
    if payload.get("model_contract") != model.checkpoint_contract():
        raise ValueError("CHECKPOINT_CONTRACT_MISMATCH: TQIF slot or sensor registry changed")
    if not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("TQIF checkpoint state_dict must be a mapping")


def load_tqif_checkpoint(
    model: TQIFModel,
    path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("TQIF checkpoint root must be a mapping")
    validate_tqif_checkpoint_payload(model, payload)
    model.load_state_dict(payload["state_dict"])
    return payload


def _coerce_target_slot_registry(
    registry: TQIFTargetSlotRegistry | Sequence[Mapping[str, Any]] | None,
    target_slot_ids: Sequence[str] | None,
    *,
    output_dim: int,
) -> TQIFTargetSlotRegistry:
    if registry is not None and target_slot_ids is not None:
        raise ValueError("provide target_slot_registry or target_slot_ids, not both")
    if registry is None:
        if target_slot_ids is None:
            raise ValueError("target_slot_registry is required")
        result = TQIFTargetSlotRegistry.from_ids(target_slot_ids)
    elif isinstance(registry, TQIFTargetSlotRegistry):
        result = registry
    else:
        result = TQIFTargetSlotRegistry.from_mappings(registry)
    if len(result.slots) != output_dim:
        raise ValueError("target slot registry length must equal output_dim")
    return result


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _masked_attention(
    module: nn.MultiheadAttention,
    queries: torch.Tensor,
    keys: torch.Tensor,
    key_mask: torch.Tensor,
    *,
    collect_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size, query_count, token_dim = queries.shape
    key_count = keys.shape[1]
    if key_count == 0:
        return (
            queries.new_zeros((batch_size, query_count, token_dim)),
            (
                queries.new_zeros((batch_size, module.num_heads, query_count, 0))
                if collect_weights
                else None
            ),
        )
    available = key_mask.any(dim=1)
    context = queries.new_zeros((batch_size, query_count, token_dim))
    weights = (
        queries.new_zeros((batch_size, module.num_heads, query_count, key_count))
        if collect_weights
        else None
    )
    if not bool(available.any()):
        return context, weights
    selected = torch.nonzero(available, as_tuple=False).flatten()
    selected_context, selected_weights = module(
        queries.index_select(0, selected),
        keys.index_select(0, selected),
        keys.index_select(0, selected),
        key_padding_mask=~key_mask.index_select(0, selected),
        need_weights=collect_weights,
        average_attn_weights=False,
    )
    return (
        context.index_copy(0, selected, selected_context),
        weights.index_copy(0, selected, selected_weights)
        if collect_weights and selected_weights is not None and weights is not None
        else None,
    )


def _unordered_pair_type_index(
    type_low: torch.Tensor,
    type_high: torch.Tensor,
    type_count: int,
) -> torch.Tensor:
    if torch.any(type_low < 0) or torch.any(type_high >= type_count):
        raise ValueError("sensor type pair index is out of range")
    offset = type_low * type_count - type_low * (type_low - 1) // 2
    return offset + (type_high - type_low)


def _validate_positive_dimensions(*values: int) -> None:
    if any(value <= 0 for value in values):
        raise ValueError("TQIF dimensions must be positive")


def _validate_unique_strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if not normalized or any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _validate_sensor_types(values: Sequence[str], sensor_count: int) -> tuple[str, ...]:
    normalized = tuple(values)
    if len(normalized) != sensor_count:
        raise ValueError("sensor_types length must equal sensor_ids length")
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError("sensor_types must contain non-empty strings")
    return normalized


def _require_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


__all__ = [
    "MatchedConcatMLP",
    "SensorCapacityControl",
    "TQIFDiagnostics",
    "TQIF_HEAD_IDS",
    "TQIFModel",
    "TQIFModelDiagnostics",
    "TQIF_QUERY_MODES",
    "TQIF_RECIPE_NAMES",
    "TQIF_RECIPE_SPECS",
    "TQIF_MODEL_SCHEMA_VERSION",
    "TQIFSensorSpec",
    "TQIFScalarSensorEncoder",
    "TQIFFusionCore",
    "TQIFSensorEncoding",
    "TQIFTargetSlot",
    "TQIFTargetSlotRegistry",
    "build_tqif_matched_concat_model",
    "build_tqif_matched_concat_model_from_config",
    "build_tqif_model",
    "build_tqif_model_from_config",
    "load_tqif_checkpoint",
    "sensor_registry_hash",
    "target_slot_registry_hash",
    "validate_tqif_model_config",
    "validate_tqif_checkpoint_payload",
]
