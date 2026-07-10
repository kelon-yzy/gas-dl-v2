from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor


@dataclass(frozen=True, slots=True)
class ExtraTreesHeadConfig:
    """Configuration for a deployable raw3 ExtraTrees regression head."""

    n_estimators: int = 600
    max_features: float = 0.7
    min_samples_leaf: int = 2
    max_depth: int | None = None
    n_jobs: int = -1
    seed: int = 20260704
    out_dim: int = 3


class _ExtraTreesRaw3Regressor:
    """Observed-feature ExtraTrees model that directly predicts raw3 percentages."""

    def __init__(self, *, config: ExtraTreesHeadConfig | None = None):
        self.config = config or ExtraTreesHeadConfig()
        _validate_config(self.config)
        self.model = ExtraTreesRegressor(
            n_estimators=self.config.n_estimators,
            max_features=self.config.max_features,
            min_samples_leaf=self.config.min_samples_leaf,
            max_depth=self.config.max_depth,
            n_jobs=self.config.n_jobs,
            random_state=self.config.seed,
        )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: tuple[str, ...] | None = None,
    ) -> _ExtraTreesRaw3Regressor:
        del feature_names
        x_arr = _as_finite_2d(x, name="x")
        y_arr = _as_finite_2d(y, name="y", expected_cols=self.config.out_dim)
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"x/y row counts must match, got {x_arr.shape[0]} and {y_arr.shape[0]}")
        self.model.fit(x_arr, y_arr)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_arr = _as_finite_2d(x, name="x")
        predictions = np.asarray(self.model.predict(x_arr), dtype=np.float32)
        if predictions.ndim != 2 or predictions.shape[1] != self.config.out_dim:
            raise ValueError(
                f"extratrees predictions must have shape (N, {self.config.out_dim}), got {predictions.shape}"
            )
        if not np.isfinite(predictions).all():
            raise ValueError("extratrees predictions contain non-finite values")
        return predictions


def _validate_config(config: ExtraTreesHeadConfig) -> None:
    if config.n_estimators < 1:
        raise ValueError("n_estimators must be >= 1")
    if not 0.0 < config.max_features <= 1.0:
        raise ValueError("max_features must be in (0, 1]")
    if config.min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be >= 1")
    if config.max_depth is not None and config.max_depth < 1:
        raise ValueError("max_depth must be >= 1 when specified")
    if config.n_jobs == 0:
        raise ValueError("n_jobs must not be 0")
    if config.out_dim != 3:
        raise ValueError("tv3 extratrees head requires raw3 output")


def _as_finite_2d(values: np.ndarray, *, name: str, expected_cols: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise ValueError(f"{name} must have {expected_cols} columns, got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr
