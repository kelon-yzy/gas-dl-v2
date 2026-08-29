from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


OOF_SCHEMA_VERSION = "gf-a2-oof-1"
FOLD_SCHEMA_VERSION = "gf-a2-oof-fold-1"


@dataclass(frozen=True)
class OOFResult:
    predictions: np.ndarray
    fold_manifest: Mapping[str, Any]
    provenance: tuple[Mapping[str, Any], ...]


def build_grouped_fold_manifest(
    group_ids: Sequence[str],
    condition_families: Sequence[str],
    *,
    n_splits: int = 5,
    seed: int = 20260827,
) -> dict[str, Any]:
    groups = tuple(str(group) for group in group_ids)
    families = tuple(str(family) for family in condition_families)
    if len(groups) == 0 or len(groups) != len(families):
        raise ValueError("group_ids and condition_families must be non-empty and aligned")
    if len(set(groups)) != len(groups):
        raise ValueError("OOF fold construction requires one row per unique group")
    if n_splits < 2 or n_splits > len(groups):
        raise ValueError("n_splits must be at least 2 and no greater than group count")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    family_values = set(families)
    if not family_values <= {"binary", "ternary"}:
        raise ValueError("condition_families must contain only binary or ternary")

    assignments: list[dict[str, Any]] = []
    for family in sorted(family_values):
        family_groups = [group for group, value in zip(groups, families, strict=True) if value == family]
        if len(family_groups) < n_splits:
            raise ValueError(f"family {family!r} has fewer groups than n_splits")
        ordered = sorted(
            family_groups,
            key=lambda group: hashlib.sha256(f"{seed}:{family}:{group}".encode("utf-8")).hexdigest(),
        )
        assignments.extend(
            {
                "mixture_id": group,
                "condition_family": family,
                "fold": index % n_splits,
            }
            for index, group in enumerate(ordered)
        )
    assignments.sort(key=lambda item: item["mixture_id"])
    manifest = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "seed": int(seed),
        "n_splits": int(n_splits),
        "group_key": "mixture_id",
        "assignments": assignments,
    }
    validate_grouped_fold_manifest(manifest, groups, families)
    return manifest


def validate_grouped_fold_manifest(
    manifest: Mapping[str, Any],
    group_ids: Sequence[str],
    condition_families: Sequence[str],
) -> None:
    if manifest.get("schema_version") != FOLD_SCHEMA_VERSION:
        raise ValueError("unsupported OOF fold manifest schema")
    n_splits = manifest.get("n_splits")
    if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 2:
        raise ValueError("OOF fold manifest n_splits must be an integer >= 2")
    assignments = manifest.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("OOF fold manifest assignments must be a list")
    expected = {
        str(group): str(family)
        for group, family in zip(group_ids, condition_families, strict=True)
    }
    if len(set(expected)) != len(expected):
        raise ValueError("group_ids must be unique")
    seen: dict[str, int] = {}
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise ValueError("each OOF assignment must be an object")
        group = assignment.get("mixture_id")
        family = assignment.get("condition_family")
        fold = assignment.get("fold")
        if not isinstance(group, str) or group not in expected:
            raise ValueError("OOF assignment contains an unknown mixture_id")
        if family != expected[group]:
            raise ValueError(f"OOF family mismatch for {group!r}")
        if not isinstance(fold, int) or isinstance(fold, bool) or not 0 <= fold < n_splits:
            raise ValueError(f"OOF fold must be within [0,{n_splits})")
        if group in seen:
            raise ValueError(f"OOF group appears in more than one fold: {group}")
        seen[group] = fold
    if set(seen) != set(expected):
        raise ValueError("OOF fold manifest does not cover exactly the supplied groups")
    fold_families = {
        fold: {expected[group] for group, assigned_fold in seen.items() if assigned_fold == fold}
        for fold in range(n_splits)
    }
    if any(not values for values in fold_families.values()):
        raise ValueError("OOF fold manifest contains an empty fold")


def generate_grouped_oof_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    group_ids: Sequence[str],
    condition_families: Sequence[str],
    *,
    estimator_factory: Callable[[int], Any],
    n_splits: int = 5,
    seed: int = 20260827,
    model_config_hash: str,
    fold_manifest: Mapping[str, Any] | None = None,
    transformer_factory: Callable[[], Any] | None = None,
) -> OOFResult:
    feature_values = np.asarray(features, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    if feature_values.ndim != 2 or target_values.ndim != 2:
        raise ValueError("features and targets must be two-dimensional arrays")
    if len(feature_values) != len(target_values) or len(feature_values) != len(group_ids):
        raise ValueError("features, targets, and group_ids must have aligned rows")
    if not np.isfinite(feature_values).all() or not np.isfinite(target_values).all():
        raise ValueError("features and targets must be finite")
    _validate_hash(model_config_hash, "model_config_hash")
    if fold_manifest is None:
        fold_manifest = build_grouped_fold_manifest(
            group_ids,
            condition_families,
            n_splits=n_splits,
            seed=seed,
        )
    else:
        validate_grouped_fold_manifest(fold_manifest, group_ids, condition_families)
    assignments = {
        str(item["mixture_id"]): int(item["fold"])
        for item in fold_manifest["assignments"]
    }
    predictions = np.full_like(target_values, np.nan, dtype=np.float64)
    provenance: list[Mapping[str, Any]] = []
    groups = np.asarray([str(group) for group in group_ids], dtype=object)
    for fold in range(int(fold_manifest["n_splits"])):
        test_indices = np.asarray(
            [index for index, group in enumerate(groups) if assignments[str(group)] == fold],
            dtype=np.int64,
        )
        train_indices = np.asarray(
            [index for index, group in enumerate(groups) if assignments[str(group)] != fold],
            dtype=np.int64,
        )
        if len(test_indices) == 0 or len(train_indices) == 0:
            raise ValueError(f"OOF fold {fold} has an empty train or test partition")
        if set(groups[train_indices]) & set(groups[test_indices]):
            raise ValueError(f"OOF group leakage in fold {fold}")
        train_features = feature_values[train_indices]
        test_features = feature_values[test_indices]
        if transformer_factory is not None:
            transformer = transformer_factory()
            train_features = transformer.fit_transform(train_features)
            test_features = transformer.transform(test_features)
        estimator = estimator_factory(seed + fold)
        estimator.fit(train_features, target_values[train_indices])
        fold_predictions = np.asarray(estimator.predict(test_features), dtype=np.float64)
        if fold_predictions.shape != target_values[test_indices].shape:
            raise ValueError(f"OOF estimator returned shape {fold_predictions.shape}, expected {target_values[test_indices].shape}")
        if not np.isfinite(fold_predictions).all():
            raise ValueError(f"OOF estimator returned non-finite predictions in fold {fold}")
        predictions[test_indices] = fold_predictions
        train_group_hash = _canonical_hash(sorted(str(group) for group in groups[train_indices]))
        for row_index, row_prediction in zip(test_indices, fold_predictions, strict=True):
            provenance.append(
                {
                    "row_index": int(row_index),
                    "mixture_id": str(groups[row_index]),
                    "fold": fold,
                    "train_group_hash": train_group_hash,
                    "model_config_hash": model_config_hash,
                    "prediction_hash": _canonical_hash(row_prediction.tolist()),
                }
            )
    if np.isnan(predictions).any():
        raise ValueError("OOF predictions do not cover every row exactly once")
    provenance.sort(key=lambda item: int(item["row_index"]))
    return OOFResult(
        predictions=predictions,
        fold_manifest=dict(fold_manifest),
        provenance=tuple(provenance),
    )


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hash(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex string")


__all__ = [
    "FOLD_SCHEMA_VERSION",
    "OOF_SCHEMA_VERSION",
    "OOFResult",
    "build_grouped_fold_manifest",
    "generate_grouped_oof_predictions",
    "validate_grouped_fold_manifest",
]
