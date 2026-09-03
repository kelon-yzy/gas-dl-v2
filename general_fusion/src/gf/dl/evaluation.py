from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


TARGET_NAMES = ("x_Ar_pct", "x_He_pct", "x_CO2_pct")
TARGET_RANGES = np.full(len(TARGET_NAMES), 100.0, dtype=np.float64)


@dataclass(frozen=True)
class GroupAggregates:
    """Group-level targets and predictions used by every benchmark metric."""

    group_ids: tuple[str, ...]
    targets: np.ndarray
    predictions: np.ndarray


def aggregate_by_group(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: Sequence[str] | np.ndarray,
    indices: Sequence[int] | np.ndarray,
) -> GroupAggregates:
    target_values = np.asarray(targets, dtype=np.float64)
    prediction_values = np.asarray(predictions, dtype=np.float64)
    if target_values.ndim != 2 or prediction_values.shape != target_values.shape:
        raise ValueError("targets and predictions must be two-dimensional arrays with equal shape")
    group_values_all = np.asarray(groups, dtype=object)
    if group_values_all.ndim != 1 or len(group_values_all) != len(target_values):
        raise ValueError("groups must be a one-dimensional array aligned with targets")
    index_values = np.asarray(indices, dtype=np.int64)
    if index_values.ndim != 1 or len(index_values) == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence")
    if np.any(index_values < 0) or np.any(index_values >= len(target_values)):
        raise IndexError("indices contain an out-of-range row")

    selected_targets = target_values[index_values]
    selected_predictions = prediction_values[index_values]
    selected_groups = group_values_all[index_values]
    unique_groups = tuple(sorted({str(group) for group in selected_groups}))
    group_targets = np.vstack(
        [selected_targets[selected_groups == group].mean(axis=0) for group in unique_groups]
    )
    group_predictions = np.vstack(
        [selected_predictions[selected_groups == group].mean(axis=0) for group in unique_groups]
    )
    return GroupAggregates(
        group_ids=unique_groups,
        targets=group_targets,
        predictions=group_predictions,
    )


def evaluate_predictions(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: Sequence[str] | np.ndarray,
    indices: Sequence[int] | np.ndarray,
    *,
    target_ranges: Sequence[float] | np.ndarray = TARGET_RANGES,
) -> dict[str, Any]:
    grouped = aggregate_by_group(targets, predictions, groups, indices)
    ranges = np.asarray(target_ranges, dtype=np.float64)
    if ranges.ndim != 1 or ranges.shape[0] != grouped.targets.shape[1]:
        raise ValueError("target_ranges must have one positive value per target")
    if not np.isfinite(ranges).all() or np.any(ranges <= 0.0):
        raise ValueError("target_ranges must contain finite positive values")

    errors = grouped.targets - grouped.predictions
    absolute_error = np.abs(errors)
    squared_error = errors**2
    component_mae = absolute_error.mean(axis=0)
    component_rmse = np.sqrt(squared_error.mean(axis=0))
    component_rnmae = component_mae / ranges
    component_r2 = [
        r2_score(grouped.targets[:, index], grouped.predictions[:, index])
        for index in range(grouped.targets.shape[1])
    ]
    group_mae = absolute_error.mean(axis=1)
    return {
        "group_count": len(grouped.group_ids),
        "macro_RNMAE": float(component_rnmae.mean()),
        "component_RNMAE": [float(value) for value in component_rnmae],
        "component_MAE": [float(value) for value in component_mae],
        "component_RMSE": [float(value) for value in component_rmse],
        "component_R2": component_r2,
        "sum_absolute_error": float(np.abs(grouped.targets.sum(axis=1) - grouped.predictions.sum(axis=1)).mean()),
        "worst_group_MAE": float(group_mae.max()),
        "P90_group_MAE": float(np.percentile(group_mae, 90)),
    }


def group_bootstrap_comparison(
    method_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    targets: np.ndarray,
    groups: Sequence[str] | np.ndarray,
    *,
    seed: int,
    samples: int,
    indices: Sequence[int] | np.ndarray | None = None,
    target_ranges: Sequence[float] | np.ndarray = TARGET_RANGES,
) -> dict[str, Any]:
    method_values = np.asarray(method_predictions, dtype=np.float64)
    baseline_values = np.asarray(baseline_predictions, dtype=np.float64)
    target_values = np.asarray(targets, dtype=np.float64)
    if method_values.shape != baseline_values.shape or method_values.shape != target_values.shape:
        raise ValueError("method_predictions, baseline_predictions, and targets must have equal shape")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    if indices is None:
        index_values = np.arange(len(target_values), dtype=np.int64)
    else:
        index_values = np.asarray(indices, dtype=np.int64)
    ranges = np.asarray(target_ranges, dtype=np.float64)
    if ranges.ndim != 1 or ranges.shape[0] != target_values.shape[1]:
        raise ValueError("target_ranges must have one positive value per target")
    if not np.isfinite(ranges).all() or np.any(ranges <= 0.0):
        raise ValueError("target_ranges must contain finite positive values")
    group_values = np.asarray(groups, dtype=object)
    if group_values.ndim != 1 or len(group_values) != len(target_values):
        raise ValueError("groups must be a one-dimensional array aligned with targets")

    unique_groups = np.array(sorted({str(group_values[index]) for index in index_values}), dtype=object)
    if len(unique_groups) == 0:
        raise ValueError("bootstrap comparison requires at least one group")
    per_group_difference: list[float] = []
    for group in unique_groups:
        group_indices = np.asarray(
            [index for index in index_values if str(group_values[index]) == group],
            dtype=np.int64,
        )
        method_error = np.abs(
            target_values[group_indices].mean(axis=0) - method_values[group_indices].mean(axis=0)
        )
        baseline_error = np.abs(
            target_values[group_indices].mean(axis=0) - baseline_values[group_indices].mean(axis=0)
        )
        per_group_difference.append(float(np.mean((method_error - baseline_error) / ranges)))

    differences = np.asarray(per_group_difference, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw_indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrapped = differences[draw_indices].mean(axis=1)
    percentile_2_5 = float(np.percentile(bootstrapped, 2.5))
    percentile_97_5 = float(np.percentile(bootstrapped, 97.5))
    return {
        "seed": int(seed),
        "samples": int(samples),
        "mean": float(bootstrapped.mean()),
        "percentile_2_5": percentile_2_5,
        "percentile_97_5": percentile_97_5,
        "ci_excludes_zero": bool(percentile_97_5 < 0.0 or percentile_2_5 > 0.0),
    }


def r2_score(target: np.ndarray, prediction: np.ndarray) -> float | None:
    target_values = np.asarray(target, dtype=np.float64)
    prediction_values = np.asarray(prediction, dtype=np.float64)
    if target_values.shape != prediction_values.shape:
        raise ValueError("target and prediction must have equal shape")
    centered = target_values - target_values.mean()
    total = float(np.dot(centered, centered))
    if total <= 0.0:
        return None
    residual = target_values - prediction_values
    return float(1.0 - np.dot(residual, residual) / total)


def evaluate_output_constraints(
    predictions: np.ndarray,
    *,
    targets: np.ndarray | None = None,
    total: float = 100.0,
) -> dict[str, Any]:
    """Report explicit fixed-total and structural-zero diagnostics for a head."""

    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("predictions must have shape [N,C] with C>0")
    if not np.isfinite(values).all():
        raise ValueError("predictions must contain only finite values")
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("total must be finite and positive")
    diagnostics: dict[str, Any] = {
        "sample_count": int(values.shape[0]),
        "negative_rate": float(np.mean(values < 0.0)),
        "out_of_range_rate": float(np.mean((values < 0.0) | (values > total))),
        "strictly_positive_rate": float(np.mean(values > 0.0)),
        "composition_sum_bias": float(np.mean(values.sum(axis=1) - total)),
        "composition_sum_mae": float(np.mean(np.abs(values.sum(axis=1) - total))),
    }
    if targets is not None:
        target_values = np.asarray(targets, dtype=np.float64)
        if target_values.shape != values.shape:
            raise ValueError("targets must have the same shape as predictions")
        zero_mask = target_values == 0.0
        diagnostics["zero_component_count"] = int(zero_mask.sum())
        diagnostics["zero_component_residual_mae"] = (
            float(np.abs(values[zero_mask]).mean()) if np.any(zero_mask) else None
        )
        diagnostics["zero_component_prediction_max_abs"] = (
            float(np.abs(values[zero_mask]).max()) if np.any(zero_mask) else None
        )
    return diagnostics


__all__ = [
    "TARGET_RANGES",
    "GroupAggregates",
    "aggregate_by_group",
    "evaluate_predictions",
    "group_bootstrap_comparison",
    "evaluate_output_constraints",
    "r2_score",
]
