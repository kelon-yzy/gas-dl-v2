"""Module C: frozen RawDSP physical group mapping and grouped bottleneck encoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


GROUP_SPEC_V1 = "raw_dsp_physics_groups_v1"
EXPECTED_FEATURE_COUNT = 1008
DEFAULT_GROUP_BOTTLENECK_DIM = 16
DEFAULT_GROUP_DROPOUT = 0.0
DEFAULT_ACTIVATION_DROPOUT = 0.1
PRE_REGISTERED_PERMUTATION_SEED = 20260712
EXPECTED_PARAMETER_COUNT = 28051

# Exhaustive, mutually exclusive channel-token → group mapping for frozen RawDSP 1008 cols.
PHYSICAL_GROUP_CHANNELS: dict[str, tuple[str, ...]] = {
    "G1_tof": ("ultrasonic_tof_observed_raw_dsp_s",),
    "G2_sound_speed": ("ultrasonic_sound_speed_raw_dsp_m_per_s",),
    "G3_peak_response": (
        "ultrasonic_peak_index_raw_dsp",
        "ultrasonic_corr_peak",
    ),
    "G4_signal_quality": (
        "ultrasonic_snr_db",
        "ultrasonic_raw_dsp_quality",
        "ultrasonic_raw_dsp_accepted",
    ),
    "G5_ndir_co2": ("V_NDIR_CO2",),
    "G6_tcs": ("V_TCS",),
    "G7_environment_geometry": (
        "T_C",
        "P_MPa",
        "H_RH",
        "L_m",
        "piston_position_m",
    ),
}

GROUP_ORDER: tuple[str, ...] = tuple(PHYSICAL_GROUP_CHANNELS)
EXPECTED_GROUP_COUNTS: dict[str, int] = {
    "G1_tof": 72,
    "G2_sound_speed": 72,
    "G3_peak_response": 144,
    "G4_signal_quality": 216,
    "G5_ndir_co2": 72,
    "G6_tcs": 72,
    "G7_environment_geometry": 360,
}

CHANNEL_TO_GROUP: dict[str, str] = {
    channel: group_id
    for group_id, channels in PHYSICAL_GROUP_CHANNELS.items()
    for channel in channels
}


@dataclass(frozen=True, slots=True)
class GroupedBottleneckConfig:
    group_spec: str = GROUP_SPEC_V1
    group_assignment: str = "physical"  # physical | permuted
    group_bottleneck_dim: int = DEFAULT_GROUP_BOTTLENECK_DIM
    group_dropout: float = DEFAULT_GROUP_DROPOUT
    activation_dropout: float = DEFAULT_ACTIVATION_DROPOUT
    permutation_seed: int | None = None

    def __post_init__(self) -> None:
        if self.group_spec != GROUP_SPEC_V1:
            raise ValueError(f"unsupported group_spec {self.group_spec!r}")
        if self.group_assignment not in {"physical", "permuted"}:
            raise ValueError(
                f"group_assignment must be 'physical' or 'permuted', got {self.group_assignment!r}"
            )
        if self.group_bottleneck_dim != DEFAULT_GROUP_BOTTLENECK_DIM:
            raise ValueError(
                f"P0 freezes group_bottleneck_dim={DEFAULT_GROUP_BOTTLENECK_DIM}, "
                f"got {self.group_bottleneck_dim}"
            )
        if self.group_dropout != DEFAULT_GROUP_DROPOUT:
            raise ValueError(
                f"P0 freezes group_dropout={DEFAULT_GROUP_DROPOUT}, got {self.group_dropout}"
            )
        if self.activation_dropout != DEFAULT_ACTIVATION_DROPOUT:
            raise ValueError(
                f"P0 freezes activation_dropout={DEFAULT_ACTIVATION_DROPOUT}, "
                f"got {self.activation_dropout}"
            )
        if self.group_assignment == "permuted":
            if self.permutation_seed is None:
                raise ValueError("permuted assignment requires permutation_seed")
            if int(self.permutation_seed) != PRE_REGISTERED_PERMUTATION_SEED:
                raise ValueError(
                    f"P0 freezes permutation_seed={PRE_REGISTERED_PERMUTATION_SEED}, "
                    f"got {self.permutation_seed}"
                )
        elif self.permutation_seed is not None:
            raise ValueError("physical assignment must leave permutation_seed as null")


@dataclass(frozen=True, slots=True)
class FeatureGroupMapping:
    group_spec: str
    group_assignment: str
    feature_names: tuple[str, ...]
    group_ids: tuple[str, ...]
    group_indices: tuple[np.ndarray, ...]
    group_counts: dict[str, int]
    feature_names_digest: str
    permutation_seed: int | None
    permutation_digest: str
    permutation_order: tuple[int, ...] | None

    @property
    def group_dims(self) -> tuple[int, ...]:
        return tuple(int(indices.size) for indices in self.group_indices)

    def as_diagnostics(self) -> dict[str, Any]:
        return {
            "group_spec": self.group_spec,
            "group_assignment": self.group_assignment,
            "group_counts": dict(self.group_counts),
            "feature_names_digest": self.feature_names_digest,
            "permutation_seed": self.permutation_seed,
            "permutation_digest": self.permutation_digest,
            "group_ids": list(self.group_ids),
            "group_index_digests": {
                group_id: _sha256_int_array(indices)
                for group_id, indices in zip(self.group_ids, self.group_indices, strict=True)
            },
        }


def channel_token_from_feature_name(feature_name: str) -> str:
    """Extract the unique channel/array token from a frozen RawDSP feature name.

    Names are shaped ``{window}|{source}:{channel}:{stat}``.
    """
    if "|" not in feature_name:
        raise ValueError(f"feature name missing window prefix separator: {feature_name!r}")
    _window, remainder = feature_name.split("|", 1)
    parts = remainder.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"feature name must be window|source:channel:stat, got {feature_name!r}"
        )
    # channel may not contain ':'; everything between source and final stat is channel.
    channel = ":".join(parts[1:-1])
    if not channel:
        raise ValueError(f"empty channel token in feature name: {feature_name!r}")
    return channel


def feature_names_digest(feature_names: Sequence[str]) -> str:
    payload = json.dumps(list(feature_names), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_physical_group_mapping(feature_names: Sequence[str]) -> FeatureGroupMapping:
    names = tuple(str(name) for name in feature_names)
    if len(names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_FEATURE_COUNT} RawDSP features, got {len(names)}"
        )
    if len(set(names)) != len(names):
        raise ValueError("feature names must be unique")

    assigned: dict[str, list[int]] = {group_id: [] for group_id in GROUP_ORDER}
    seen_indices: set[int] = set()
    for index, name in enumerate(names):
        channel = channel_token_from_feature_name(name)
        group_id = CHANNEL_TO_GROUP.get(channel)
        if group_id is None:
            raise ValueError(f"unknown channel token {channel!r} in feature {name!r}")
        if index in seen_indices:
            raise ValueError(f"duplicate feature index assignment: {index}")
        assigned[group_id].append(index)
        seen_indices.add(index)

    if len(seen_indices) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"group mapping coverage incomplete: assigned {len(seen_indices)} / {EXPECTED_FEATURE_COUNT}"
        )

    group_indices: list[np.ndarray] = []
    group_counts: dict[str, int] = {}
    for group_id in GROUP_ORDER:
        indices = np.asarray(assigned[group_id], dtype=np.int64)
        expected = EXPECTED_GROUP_COUNTS[group_id]
        if indices.size != expected:
            raise ValueError(
                f"{group_id} expected {expected} columns, got {indices.size}"
            )
        group_indices.append(indices)
        group_counts[group_id] = int(indices.size)

    if sum(group_counts.values()) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"group counts sum to {sum(group_counts.values())}, expected {EXPECTED_FEATURE_COUNT}"
        )

    return FeatureGroupMapping(
        group_spec=GROUP_SPEC_V1,
        group_assignment="physical",
        feature_names=names,
        group_ids=GROUP_ORDER,
        group_indices=tuple(group_indices),
        group_counts=group_counts,
        feature_names_digest=feature_names_digest(names),
        permutation_seed=None,
        permutation_digest="",
        permutation_order=None,
    )


def build_permuted_group_mapping(
    feature_names: Sequence[str],
    *,
    permutation_seed: int = PRE_REGISTERED_PERMUTATION_SEED,
) -> FeatureGroupMapping:
    """Same group sizes as physical mapping, but columns randomly reassigned once."""
    if int(permutation_seed) != PRE_REGISTERED_PERMUTATION_SEED:
        raise ValueError(
            f"P0 freezes permutation_seed={PRE_REGISTERED_PERMUTATION_SEED}, got {permutation_seed}"
        )
    physical = build_physical_group_mapping(feature_names)
    n_features = len(physical.feature_names)
    rng = np.random.default_rng(int(permutation_seed))
    order = rng.permutation(n_features).astype(np.int64)
    sizes = [EXPECTED_GROUP_COUNTS[group_id] for group_id in GROUP_ORDER]
    if sum(sizes) != n_features:
        raise RuntimeError("internal group size contract broken")

    group_indices: list[np.ndarray] = []
    cursor = 0
    for size in sizes:
        group_indices.append(order[cursor : cursor + size].copy())
        cursor += size
    if cursor != n_features:
        raise RuntimeError("permuted group slice did not cover all features")

    # Validate exhaustive unique partition.
    concat = np.concatenate(group_indices)
    if concat.size != n_features or np.unique(concat).size != n_features:
        raise RuntimeError("permuted groups must form a unique partition of all columns")

    perm_digest = _sha256_int_array(order)
    return FeatureGroupMapping(
        group_spec=GROUP_SPEC_V1,
        group_assignment="permuted",
        feature_names=physical.feature_names,
        group_ids=GROUP_ORDER,
        group_indices=tuple(group_indices),
        group_counts={group_id: EXPECTED_GROUP_COUNTS[group_id] for group_id in GROUP_ORDER},
        feature_names_digest=physical.feature_names_digest,
        permutation_seed=int(permutation_seed),
        permutation_digest=perm_digest,
        permutation_order=tuple(int(value) for value in order.tolist()),
    )


def build_group_mapping(
    feature_names: Sequence[str],
    *,
    config: GroupedBottleneckConfig,
) -> FeatureGroupMapping:
    if config.group_assignment == "physical":
        return build_physical_group_mapping(feature_names)
    return build_permuted_group_mapping(
        feature_names,
        permutation_seed=int(config.permutation_seed),
    )


def expected_parameter_count(
    *,
    group_dims: Sequence[int] = tuple(EXPECTED_GROUP_COUNTS[g] for g in GROUP_ORDER),
    bottleneck_dim: int = DEFAULT_GROUP_BOTTLENECK_DIM,
    hidden_dims: Sequence[int] = (64, 64),
    out_dim: int = 3,
) -> int:
    """Closed-form parameter count for the frozen Module C residual encoder."""
    total = 0
    for dim in group_dims:
        # Linear(dim -> b) + LayerNorm(b)
        total += int(dim) * bottleneck_dim + bottleneck_dim
        total += 2 * bottleneck_dim
    trunk_in = bottleneck_dim * len(tuple(group_dims))
    current = trunk_in
    for hidden in hidden_dims:
        total += current * int(hidden) + int(hidden)
        current = int(hidden)
    total += current * out_dim + out_dim
    return total


def build_grouped_bottleneck_module(
    *,
    group_dims: Sequence[int],
    bottleneck_dim: int = DEFAULT_GROUP_BOTTLENECK_DIM,
    hidden_dims: Sequence[int] = (64, 64),
    out_dim: int = 3,
    activation_dropout: float = DEFAULT_ACTIVATION_DROPOUT,
    group_dropout: float = DEFAULT_GROUP_DROPOUT,
    zero_init_output: bool = True,
):
    """Build the Module C residual network: per-group encoder + shared trunk."""
    from torch import nn

    if not group_dims:
        raise ValueError("group_dims must not be empty")
    if any(int(dim) < 1 for dim in group_dims):
        raise ValueError("all group dims must be >= 1")
    if bottleneck_dim < 1:
        raise ValueError("bottleneck_dim must be >= 1")
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer")
    if out_dim < 1:
        raise ValueError("out_dim must be >= 1")
    if activation_dropout < 0.0 or group_dropout < 0.0:
        raise ValueError("dropout rates must be >= 0")

    class _GroupedBottleneckResidualNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.group_dims = tuple(int(dim) for dim in group_dims)
            self.bottleneck_dim = int(bottleneck_dim)
            self.group_dropout_p = float(group_dropout)
            encoders: list[nn.Module] = []
            for dim in self.group_dims:
                encoders.append(
                    nn.Sequential(
                        nn.Linear(int(dim), self.bottleneck_dim),
                        nn.LayerNorm(self.bottleneck_dim),
                        nn.SiLU(),
                        nn.Dropout(float(activation_dropout)),
                    )
                )
            self.encoders = nn.ModuleList(encoders)
            trunk_layers: list[nn.Module] = []
            current = self.bottleneck_dim * len(self.group_dims)
            for hidden in hidden_dims:
                hidden_int = int(hidden)
                trunk_layers.append(nn.Linear(current, hidden_int))
                trunk_layers.append(nn.ReLU())
                if activation_dropout > 0.0:
                    trunk_layers.append(nn.Dropout(float(activation_dropout)))
                current = hidden_int
            output_layer = nn.Linear(current, int(out_dim))
            if zero_init_output:
                nn.init.zeros_(output_layer.weight)
                nn.init.zeros_(output_layer.bias)
            trunk_layers.append(output_layer)
            self.trunk = nn.Sequential(*trunk_layers)

        def forward(self, group_tensors: Sequence[Any]) -> Any:
            import torch

            if len(group_tensors) != len(self.encoders):
                raise ValueError(
                    f"expected {len(self.encoders)} group tensors, got {len(group_tensors)}"
                )
            encoded = []
            for encoder, group_x, expected_dim in zip(
                self.encoders, group_tensors, self.group_dims, strict=True
            ):
                if group_x.ndim != 2 or int(group_x.shape[1]) != expected_dim:
                    raise ValueError(
                        f"group tensor shape {tuple(group_x.shape)} does not match dim {expected_dim}"
                    )
                encoded.append(encoder(group_x))
            if self.group_dropout_p > 0.0 and self.training:
                keep = []
                for tensor in encoded:
                    if torch.rand((), device=tensor.device) < self.group_dropout_p:
                        keep.append(torch.zeros_like(tensor))
                    else:
                        keep.append(tensor)
                encoded = keep
            return self.trunk(torch.cat(encoded, dim=-1))

    return _GroupedBottleneckResidualNet()


def count_module_parameters(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def validate_group_mapping_against_config(
    mapping: FeatureGroupMapping,
    *,
    config: GroupedBottleneckConfig,
) -> None:
    if mapping.group_spec != config.group_spec:
        raise ValueError("group_spec mismatch between mapping and config")
    if mapping.group_assignment != config.group_assignment:
        raise ValueError("group_assignment mismatch between mapping and config")
    if config.group_assignment == "permuted":
        if mapping.permutation_seed != config.permutation_seed:
            raise ValueError("permutation_seed mismatch between mapping and config")
        if not mapping.permutation_digest:
            raise ValueError("permuted mapping missing permutation_digest")
    if mapping.group_counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(f"unexpected group_counts: {mapping.group_counts}")


def _sha256_int_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()
