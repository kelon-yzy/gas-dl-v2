from __future__ import annotations

import numpy as np

from common.metrics import R2_ZERO_VARIANCE_EPSILON, RegressionMetrics
from sim.core.schema import COMPONENT_FIELDS


def regression_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> RegressionMetrics:
    """Compute pooled MAE, RMSE, and R2 for numpy predictions."""
    pred = _as_2d_array(y_pred, name="y_pred")
    true = _as_2d_array(y_true, name="y_true")
    if pred.shape != true.shape:
        raise ValueError(f"Prediction and target shapes must match, got {pred.shape} and {true.shape}")

    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    centered = true - np.mean(true, axis=0, keepdims=True)
    ss_tot = float(np.sum(centered * centered))
    if ss_tot < R2_ZERO_VARIANCE_EPSILON:
        r2 = 1.0 if ss_res < R2_ZERO_VARIANCE_EPSILON else 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    return RegressionMetrics(mae=mae, rmse=rmse, r2=float(r2))


def component_regression_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    component_names: tuple[str, ...] = COMPONENT_FIELDS,
) -> dict[str, RegressionMetrics]:
    """Compute per-component regression metrics."""
    pred = _as_2d_array(y_pred, name="y_pred")
    true = _as_2d_array(y_true, name="y_true")
    if pred.shape != true.shape:
        raise ValueError(f"Prediction and target shapes must match, got {pred.shape} and {true.shape}")
    if pred.shape[1] != len(component_names):
        raise ValueError(
            f"component_names length {len(component_names)} does not match prediction channels {pred.shape[1]}"
        )
    return {
        name: regression_metrics(pred[:, index : index + 1], true[:, index : index + 1])
        for index, name in enumerate(component_names)
    }


def _as_2d_array(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    return arr
