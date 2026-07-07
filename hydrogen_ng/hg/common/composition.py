from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from hg.sim.core.schema import COMPONENT_FIELDS


PERCENT_TOTAL = 100.0
DEFAULT_ZERO_REPLACEMENT_EPSILON = 1e-4
TRAIN_MIN_POSITIVE_HALF_EPSILON = "train_min_positive_half"
ZERO_REPLACEMENT_EPSILON_OPTIONS = (TRAIN_MIN_POSITIVE_HALF_EPSILON,)
ILR_BASIS_TOLERANCE = 1e-10
ALR_CH4_TRANSFORM = "alr_ch4"
ILR_N2_FIRST_TRANSFORM = "ilr_n2_first"
TARGET_TRANSFORM_OPTIONS = (ALR_CH4_TRANSFORM, ILR_N2_FIRST_TRANSFORM)
N2_FIRST_ILR_BASIS = np.array(
    [
        [-np.sqrt(3.0) / 6.0, -np.sqrt(3.0) / 6.0, -np.sqrt(3.0) / 6.0, np.sqrt(3.0) / 2.0],
        [-np.sqrt(2.0 / 3.0) / 2.0, np.sqrt(2.0 / 3.0), -np.sqrt(2.0 / 3.0) / 2.0, 0.0],
        [np.sqrt(0.5), 0.0, -np.sqrt(0.5), 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class ZeroReplacementAudit:
    """Split-level audit for explicit zero replacement before log-ratio transforms."""

    epsilon: float
    component_names: tuple[str, ...]
    replaced_counts: tuple[int, ...]
    max_abs_delta_percent: tuple[float, ...]
    total_rows: int


@dataclass(frozen=True, slots=True)
class TargetTransformSpec:
    """Resolved compositional target transform configuration."""

    name: str
    epsilon: float | str = DEFAULT_ZERO_REPLACEMENT_EPSILON
    reference_component: str = "x_CH4"


def resolve_target_transform_spec(config: str | dict[str, Any] | None) -> TargetTransformSpec | None:
    """Resolve a user-facing transform name or JSON object into a strict spec."""
    if config is None or config == "none":
        return None
    if isinstance(config, str):
        name = config
        epsilon: float | str = DEFAULT_ZERO_REPLACEMENT_EPSILON
        reference_component = "x_CH4"
    else:
        payload = dict(config)
        if "name" not in payload:
            raise ValueError("target_transform config requires a 'name' field")
        name = str(payload.pop("name"))
        raw_epsilon = payload.pop("epsilon", DEFAULT_ZERO_REPLACEMENT_EPSILON)
        epsilon = str(raw_epsilon) if isinstance(raw_epsilon, str) else float(raw_epsilon)
        reference_component = str(payload.pop("reference_component", "x_CH4"))
        if payload:
            raise ValueError(f"Unknown target_transform keys: {sorted(payload)}")

    if name not in TARGET_TRANSFORM_OPTIONS:
        raise ValueError(f"Unknown target_transform: {name!r}. Available: {TARGET_TRANSFORM_OPTIONS}")
    if isinstance(epsilon, str):
        if epsilon not in ZERO_REPLACEMENT_EPSILON_OPTIONS:
            raise ValueError(f"Unknown target_transform epsilon strategy: {epsilon!r}")
    elif epsilon <= 0.0:
        raise ValueError(f"target_transform epsilon must be > 0, got {epsilon}")
    if name == ALR_CH4_TRANSFORM and reference_component != "x_CH4":
        raise ValueError("alr_ch4 target_transform requires reference_component='x_CH4'")
    return TargetTransformSpec(name=name, epsilon=epsilon, reference_component=reference_component)


def resolve_zero_replacement_epsilon(
    epsilon: float | str,
    train_percent: np.ndarray,
) -> float:
    """Resolve fixed or data-driven zero-replacement epsilon from train labels."""
    if isinstance(epsilon, str):
        if epsilon != TRAIN_MIN_POSITIVE_HALF_EPSILON:
            raise ValueError(f"Unknown zero replacement epsilon strategy: {epsilon!r}")
        unit = close_to_unit_interval(train_percent)
        positives = unit[unit > 0.0]
        if positives.size == 0:
            raise ValueError("train_min_positive_half requires at least one positive train component")
        return float(np.min(positives) / 2.0)
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    return float(epsilon)


def resolve_target_transform_for_training(
    config: str | dict[str, Any] | TargetTransformSpec | None,
    train_percent: np.ndarray,
) -> TargetTransformSpec | None:
    """Resolve transform config and any data-driven epsilon against train labels."""
    spec = config if isinstance(config, TargetTransformSpec) else resolve_target_transform_spec(config)
    if spec is None:
        return None
    epsilon = resolve_zero_replacement_epsilon(spec.epsilon, train_percent)
    return TargetTransformSpec(
        name=spec.name,
        epsilon=epsilon,
        reference_component=spec.reference_component,
    )


def close_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """Close nonnegative composition rows to unit-sum proportions."""
    arr = _as_2d_float(values, name="values")
    _validate_nonnegative(arr, name="values")
    row_sums = arr.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("composition rows must have positive sum")
    return arr / row_sums


def close_to_100(values: np.ndarray) -> np.ndarray:
    """Close nonnegative composition rows to 100 percent."""
    return close_to_unit_interval(values) * PERCENT_TOTAL


def replace_zeros_multiplicative(
    unit_values: np.ndarray,
    *,
    epsilon: float = DEFAULT_ZERO_REPLACEMENT_EPSILON,
    component_names: tuple[str, ...] = COMPONENT_FIELDS,
) -> tuple[np.ndarray, ZeroReplacementAudit]:
    """Replace exact zeros in unit-sum compositions and preserve closure."""
    arr = close_to_unit_interval(unit_values)
    if arr.shape[1] != len(component_names):
        raise ValueError(f"component_names length {len(component_names)} does not match columns {arr.shape[1]}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    zero_mask = arr == 0.0
    replaced = arr.copy()
    for row_index in range(arr.shape[0]):
        row_zero_mask = zero_mask[row_index]
        zero_count = int(np.count_nonzero(row_zero_mask))
        if zero_count == 0:
            continue
        replacement_total = zero_count * epsilon
        if replacement_total >= 1.0:
            raise ValueError(
                f"zero replacement epsilon={epsilon} is too large for row with {zero_count} zero component(s)"
            )
        positive_sum = float(arr[row_index, ~row_zero_mask].sum())
        if positive_sum <= 0.0:
            raise ValueError("zero replacement requires at least one positive component per row")
        replaced[row_index, row_zero_mask] = epsilon
        replaced[row_index, ~row_zero_mask] *= (1.0 - replacement_total) / positive_sum

    deltas_percent = np.abs(replaced - arr) * PERCENT_TOTAL
    max_abs_delta_percent = (
        np.zeros(arr.shape[1], dtype=np.float64) if arr.shape[0] == 0 else deltas_percent.max(axis=0)
    )
    audit = ZeroReplacementAudit(
        epsilon=float(epsilon),
        component_names=component_names,
        replaced_counts=tuple(int(value) for value in zero_mask.sum(axis=0).tolist()),
        max_abs_delta_percent=tuple(float(value) for value in max_abs_delta_percent.tolist()),
        total_rows=int(arr.shape[0]),
    )
    return replaced, audit


def transform_composition_targets(
    y_percent: np.ndarray,
    spec: TargetTransformSpec,
    *,
    component_names: tuple[str, ...] = COMPONENT_FIELDS,
) -> tuple[np.ndarray, ZeroReplacementAudit]:
    """Apply explicit zero handling and transform percent targets to 3D coordinates."""
    if isinstance(spec.epsilon, str):
        raise ValueError("target_transform epsilon must be resolved before transforming targets")
    unit, audit = replace_zeros_multiplicative(
        close_to_unit_interval(y_percent),
        epsilon=spec.epsilon,
        component_names=component_names,
    )
    if spec.name == ALR_CH4_TRANSFORM:
        return alr_forward(unit, reference_index=_component_index(component_names, spec.reference_component)), audit
    if spec.name == ILR_N2_FIRST_TRANSFORM:
        return ilr_forward(unit), audit
    raise ValueError(f"Unknown target transform: {spec.name!r}")


def inverse_transform_composition_targets(
    transformed: np.ndarray,
    spec: TargetTransformSpec,
    *,
    component_names: tuple[str, ...] = COMPONENT_FIELDS,
) -> np.ndarray:
    """Inverse-transform 3D coordinates back to 100-percent component space."""
    if spec.name == ALR_CH4_TRANSFORM:
        unit = alr_inverse(transformed, reference_index=_component_index(component_names, spec.reference_component))
    elif spec.name == ILR_N2_FIRST_TRANSFORM:
        unit = ilr_inverse(transformed)
    else:
        raise ValueError(f"Unknown target transform: {spec.name!r}")
    return (unit * PERCENT_TOTAL).astype(np.float32, copy=False)


def alr_forward(unit_values: np.ndarray, *, reference_index: int) -> np.ndarray:
    """Additive log-ratio transform with output columns in non-reference component order."""
    arr = _positive_unit_values(unit_values, name="unit_values")
    _validate_reference_index(reference_index, arr.shape[1])
    keep = [index for index in range(arr.shape[1]) if index != reference_index]
    return np.log(arr[:, keep] / arr[:, reference_index : reference_index + 1]).astype(np.float32, copy=False)


def alr_inverse(transformed: np.ndarray, *, reference_index: int) -> np.ndarray:
    """Inverse additive log-ratio transform to unit-sum composition rows."""
    z = _as_2d_float(transformed, name="transformed")
    component_count = z.shape[1] + 1
    _validate_reference_index(reference_index, component_count)
    log_ratios = np.zeros((z.shape[0], component_count), dtype=np.float64)
    keep = [index for index in range(component_count) if index != reference_index]
    log_ratios[:, keep] = z
    exp_values = np.exp(log_ratios - log_ratios.max(axis=1, keepdims=True))
    return close_to_unit_interval(exp_values).astype(np.float32, copy=False)


def ilr_forward(unit_values: np.ndarray, *, basis: np.ndarray = N2_FIRST_ILR_BASIS) -> np.ndarray:
    """Isometric log-ratio transform using the fixed N2-first balance basis."""
    arr = _positive_unit_values(unit_values, name="unit_values")
    _validate_ilr_basis(basis, arr.shape[1])
    return (np.log(arr) @ basis.T).astype(np.float32, copy=False)


def ilr_inverse(transformed: np.ndarray, *, basis: np.ndarray = N2_FIRST_ILR_BASIS) -> np.ndarray:
    """Inverse ILR transform to unit-sum composition rows."""
    z = _as_2d_float(transformed, name="transformed")
    _validate_ilr_basis(basis, z.shape[1] + 1)
    log_values = z @ basis
    exp_values = np.exp(log_values - log_values.max(axis=1, keepdims=True))
    return close_to_unit_interval(exp_values).astype(np.float32, copy=False)


def aitchison_distance(y_pred_percent: np.ndarray, y_true_percent: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Row-wise Aitchison distance after the same explicit zero replacement policy."""
    pred_unit, _pred_audit = replace_zeros_multiplicative(close_to_unit_interval(y_pred_percent), epsilon=epsilon)
    true_unit, _true_audit = replace_zeros_multiplicative(close_to_unit_interval(y_true_percent), epsilon=epsilon)
    pred_ilr = ilr_forward(pred_unit)
    true_ilr = ilr_forward(true_unit)
    return np.linalg.norm(pred_ilr - true_ilr, axis=1)


def _positive_unit_values(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = close_to_unit_interval(values)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain strictly positive values for log-ratio transform")
    return arr


def _as_2d_float(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got ndim={arr.ndim}")
    if arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_nonnegative(values: np.ndarray, *, name: str) -> None:
    if np.any(values < 0.0):
        raise ValueError(f"{name} must not contain negative components")


def _component_index(component_names: tuple[str, ...], name: str) -> int:
    try:
        return component_names.index(name)
    except ValueError as exc:
        raise ValueError(f"Unknown component {name!r}. Available: {component_names}") from exc


def _validate_reference_index(reference_index: int, component_count: int) -> None:
    if reference_index < 0 or reference_index >= component_count:
        raise ValueError(f"reference_index must be in [0, {component_count}), got {reference_index}")


def _validate_ilr_basis(basis: np.ndarray, component_count: int) -> None:
    matrix = np.asarray(basis, dtype=np.float64)
    expected_shape = (component_count - 1, component_count)
    if matrix.shape != expected_shape:
        raise ValueError(f"ILR basis must be shaped {expected_shape}, got {matrix.shape}")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, np.zeros(component_count - 1), atol=ILR_BASIS_TOLERANCE):
        raise ValueError("ILR basis rows must sum to zero")
    gram = matrix @ matrix.T
    identity = np.eye(component_count - 1, dtype=np.float64)
    if not np.allclose(gram, identity, atol=ILR_BASIS_TOLERANCE):
        raise ValueError("ILR basis rows must be orthonormal")
