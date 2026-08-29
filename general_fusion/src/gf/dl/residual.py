from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor


@dataclass(frozen=True)
class ResidualFitResult:
    kind: str
    model: Any
    residual_targets: np.ndarray
    oof_explained_variance: tuple[float | None, ...]


def residual_targets(targets: np.ndarray, base_oof_predictions: np.ndarray) -> np.ndarray:
    target_values = np.asarray(targets, dtype=np.float64)
    base_values = np.asarray(base_oof_predictions, dtype=np.float64)
    if target_values.shape != base_values.shape or target_values.ndim != 2:
        raise ValueError("targets and base_oof_predictions must be equal two-dimensional arrays")
    if not np.isfinite(target_values).all() or not np.isfinite(base_values).all():
        raise ValueError("targets and base_oof_predictions must be finite")
    return target_values - base_values


def fit_residual_learner(
    kind: str,
    features: np.ndarray,
    targets: np.ndarray,
    base_oof_predictions: np.ndarray,
    *,
    ridge_alpha: float = 1.0,
    random_state: int = 20260827,
) -> ResidualFitResult:
    feature_values = np.asarray(features, dtype=np.float64)
    if feature_values.ndim != 2 or not np.isfinite(feature_values).all():
        raise ValueError("features must be a finite two-dimensional array")
    residual = residual_targets(targets, base_oof_predictions)
    if kind == "ridge_residual":
        if ridge_alpha <= 0.0:
            raise ValueError("ridge_alpha must be positive")
        model: Any = Ridge(alpha=ridge_alpha)
    elif kind == "shallow_gbdt_residual":
        model = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=random_state,
            )
        )
    else:
        raise ValueError(f"unsupported residual learner: {kind!r}")
    model.fit(feature_values, residual)
    fitted = np.asarray(model.predict(feature_values), dtype=np.float64)
    explained = tuple(
        _explained_variance(residual[:, index], fitted[:, index])
        for index in range(residual.shape[1])
    )
    return ResidualFitResult(
        kind=kind,
        model=model,
        residual_targets=residual,
        oof_explained_variance=explained,
    )


def apply_residual_learner(
    base_predictions: np.ndarray,
    residual_model: ResidualFitResult,
    features: np.ndarray,
) -> np.ndarray:
    base_values = np.asarray(base_predictions, dtype=np.float64)
    feature_values = np.asarray(features, dtype=np.float64)
    if base_values.ndim != 2 or feature_values.ndim != 2 or len(base_values) != len(feature_values):
        raise ValueError("base_predictions and features must be aligned two-dimensional arrays")
    residual_prediction = np.asarray(residual_model.model.predict(feature_values), dtype=np.float64)
    if residual_prediction.shape != base_values.shape:
        raise ValueError("residual model output shape does not match base predictions")
    if not np.isfinite(residual_prediction).all():
        raise ValueError("residual model returned non-finite predictions")
    return base_values + residual_prediction


def _explained_variance(target: np.ndarray, prediction: np.ndarray) -> float | None:
    centered = target - target.mean()
    total = float(np.dot(centered, centered))
    if total <= 0.0:
        return None
    residual = target - prediction
    return float(1.0 - np.dot(residual, residual) / total)


__all__ = [
    "ResidualFitResult",
    "apply_residual_learner",
    "fit_residual_learner",
    "residual_targets",
]
