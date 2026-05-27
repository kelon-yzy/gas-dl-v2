from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_SCALE_EPSILON = 1e-12


@dataclass(slots=True)
class MeanRegressor:
    """Multi-output mean baseline for v4 composition regression."""

    target_mean_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> MeanRegressor:
        x_arr = _as_2d_features(x)
        y_arr = _as_2d_targets(y)
        _validate_row_count(x_arr, y_arr)
        self.target_mean_ = y_arr.mean(axis=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_arr = _as_2d_features(x)
        if self.target_mean_ is None:
            raise ValueError("MeanRegressor must be fitted before predict")
        return np.repeat(self.target_mean_.reshape(1, -1), x_arr.shape[0], axis=0).astype(np.float32)


@dataclass(slots=True)
class RidgeRegressor:
    """Closed-form multi-output ridge regressor with optional feature scaling.

    This keeps the traditional ML path dependency-light by avoiding scikit-learn
    while still providing a real trainable baseline for generated benchmark runs.
    """

    alpha: float = 1.0
    fit_intercept: bool = True
    standardize: bool = True
    coef_: np.ndarray | None = None
    intercept_: np.ndarray | None = None
    x_mean_: np.ndarray | None = None
    x_scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeRegressor:
        if self.alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        x_arr = _as_2d_features(x)
        y_arr = _as_2d_targets(y)
        _validate_row_count(x_arr, y_arr)

        x_fit = self._fit_transform_x(x_arr)
        design = _design_matrix(x_fit, fit_intercept=self.fit_intercept)
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(self.alpha)
        if self.fit_intercept:
            penalty[0, 0] = 0.0
        lhs = design.T @ design + penalty
        rhs = design.T @ y_arr.astype(np.float64)
        try:
            weights = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(lhs) @ rhs

        if self.fit_intercept:
            self.intercept_ = weights[0].astype(np.float32)
            self.coef_ = weights[1:].astype(np.float32)
        else:
            self.intercept_ = np.zeros(y_arr.shape[1], dtype=np.float32)
            self.coef_ = weights.astype(np.float32)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("RidgeRegressor must be fitted before predict")
        x_arr = _as_2d_features(x)
        x_fit = self._transform_x(x_arr)
        return (x_fit @ self.coef_ + self.intercept_).astype(np.float32)

    def _fit_transform_x(self, x: np.ndarray) -> np.ndarray:
        if self.standardize:
            self.x_mean_ = x.mean(axis=0)
            scale = x.std(axis=0)
            self.x_scale_ = np.where(scale > _SCALE_EPSILON, scale, 1.0)
        else:
            self.x_mean_ = np.zeros(x.shape[1], dtype=np.float64)
            self.x_scale_ = np.ones(x.shape[1], dtype=np.float64)
        return self._transform_x(x)

    def _transform_x(self, x: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_scale_ is None:
            raise ValueError("RidgeRegressor feature transform is not fitted")
        if x.shape[1] != self.x_mean_.shape[0]:
            raise ValueError(f"feature count mismatch: {x.shape[1]} != {self.x_mean_.shape[0]}")
        return (x - self.x_mean_) / self.x_scale_


REGRESSOR_REGISTRY: dict[str, type[MeanRegressor] | type[RidgeRegressor]] = {
    "mean": MeanRegressor,
    "ridge": RidgeRegressor,
}


def build_regressor(config: str | dict[str, Any] | None = None) -> MeanRegressor | RidgeRegressor:
    """Build a traditional ML regressor from a name or config dict."""
    if config is None:
        name = "ridge"
        kwargs: dict[str, Any] = {}
    elif isinstance(config, str):
        name = config
        kwargs = {}
    else:
        model_config = dict(config)
        name = str(model_config.pop("name"))
        kwargs = model_config

    if name not in REGRESSOR_REGISTRY:
        raise ValueError(f"Unknown regressor name: {name!r}. Available: {sorted(REGRESSOR_REGISTRY)}")
    return REGRESSOR_REGISTRY[name](**kwargs)


def _as_2d_features(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"features must be a 2D array shaped (N, F), got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    return arr


def _as_2d_targets(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"targets must be a 1D or 2D array, got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError("targets must contain at least one row")
    return arr


def _validate_row_count(x: np.ndarray, y: np.ndarray) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"feature/target row count mismatch: {x.shape[0]} != {y.shape[0]}")


def _design_matrix(x: np.ndarray, *, fit_intercept: bool) -> np.ndarray:
    if not fit_intercept:
        return x
    ones = np.ones((x.shape[0], 1), dtype=x.dtype)
    return np.concatenate([ones, x], axis=1)
