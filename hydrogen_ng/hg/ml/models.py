from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


_SCALE_EPSILON = 1e-12
_MODALITIES = ("acoustic", "optical", "thermal")
_ACOUSTIC_PREFIXES = ("ultrasonic:", "fiber_mic:")
_ENVIRONMENT_CHANNELS = {"T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"}
_OPTICAL_CHANNEL_PREFIXES = ("V_NDIR",)
_THERMAL_CHANNELS = {"V_TCS"}


@dataclass(slots=True)
class MeanRegressor:
    """Multi-output mean baseline for v4 composition regression."""

    target_mean_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, *, feature_names: tuple[str, ...] | None = None) -> MeanRegressor:
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

    def fit(self, x: np.ndarray, y: np.ndarray, *, feature_names: tuple[str, ...] | None = None) -> RidgeRegressor:
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


@dataclass(slots=True)
class DynamicStackingSVRRegressor:
    """Patent-aligned multimodal stacking regressor.

    Layer 0 trains independent RBF-SVR models for acoustic, optical, and thermal
    feature views. Online inference estimates per-view drift MSE through
    Monte-Carlo noise in each view's z-scored feature space, converts inverse
    uncertainty into dynamic weights, and feeds weighted base predictions into
    a ridge meta learner.
    """

    svr_c: float = 10.0
    svr_epsilon: float = 0.01
    svr_gamma: str | float = "scale"
    ridge_alpha: float = 1.0
    mc_samples: int = 16
    mc_noise_std: float = 0.02
    baseline_error_constant: float = 1e-6
    random_seed: int = 123
    meta_standardize: bool = True
    n_jobs: int = 1
    groups_: dict[str, np.ndarray] | None = None
    scalers_: dict[str, StandardScaler] | None = None
    base_models_: dict[str, MultiOutputRegressor] | None = None
    meta_model_: RidgeRegressor | None = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: tuple[str, ...] | None = None,
    ) -> DynamicStackingSVRRegressor:
        if feature_names is None:
            raise ValueError("DynamicStackingSVRRegressor requires feature_names to split modality views")
        if self.mc_samples <= 0:
            raise ValueError(f"mc_samples must be > 0, got {self.mc_samples}")
        if self.mc_noise_std < 0.0:
            raise ValueError(f"mc_noise_std must be >= 0, got {self.mc_noise_std}")
        if self.baseline_error_constant <= 0.0:
            raise ValueError(f"baseline_error_constant must be > 0, got {self.baseline_error_constant}")
        if self.n_jobs != -1 and self.n_jobs <= 0:
            raise ValueError(f"n_jobs must be -1 or > 0, got {self.n_jobs}")

        x_arr = _as_2d_features(x)
        y_arr = _as_2d_targets(y)
        _validate_row_count(x_arr, y_arr)
        _validate_feature_name_count(x_arr, feature_names)
        groups = _split_modality_columns(feature_names)

        scalers: dict[str, StandardScaler] = {}
        base_models: dict[str, MultiOutputRegressor] = {}
        base_predictions: dict[str, np.ndarray] = {}
        for modality in _MODALITIES:
            view = x_arr[:, groups[modality]]
            scaler = StandardScaler()
            scaled = scaler.fit_transform(view)
            model = MultiOutputRegressor(
                SVR(kernel="rbf", C=self.svr_c, epsilon=self.svr_epsilon, gamma=self.svr_gamma),
                n_jobs=self.n_jobs,
            )
            model.fit(scaled, y_arr)
            scalers[modality] = scaler
            base_models[modality] = model
            base_predictions[modality] = model.predict(scaled)

        self.groups_ = groups
        self.scalers_ = scalers
        self.base_models_ = base_models
        weights = self._dynamic_weights(x_arr, base_predictions)
        meta_x = _weighted_meta_features(base_predictions, weights)
        self.meta_model_ = RidgeRegressor(alpha=self.ridge_alpha, standardize=self.meta_standardize).fit(meta_x, y_arr)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions, _weights = self.predict_with_diagnostics(x)
        return predictions

    def predict_with_diagnostics(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_arr = _as_2d_features(x)
        base_predictions = self._base_predictions(x_arr)
        weights = self._dynamic_weights(x_arr, base_predictions)
        if self.meta_model_ is None:
            raise ValueError("DynamicStackingSVRRegressor must be fitted before predict")
        predictions = self.meta_model_.predict(_weighted_meta_features(base_predictions, weights))
        return predictions.astype(np.float32, copy=False), weights.astype(np.float32, copy=False)

    def _base_predictions(self, x: np.ndarray) -> dict[str, np.ndarray]:
        if self.groups_ is None or self.scalers_ is None or self.base_models_ is None:
            raise ValueError("DynamicStackingSVRRegressor must be fitted before predict")
        predictions: dict[str, np.ndarray] = {}
        for modality in _MODALITIES:
            view = x[:, self.groups_[modality]]
            scaled = self.scalers_[modality].transform(view)
            predictions[modality] = self.base_models_[modality].predict(scaled)
        return predictions

    def _dynamic_weights(self, x: np.ndarray, base_predictions: dict[str, np.ndarray]) -> np.ndarray:
        if self.groups_ is None or self.scalers_ is None or self.base_models_ is None:
            raise ValueError("DynamicStackingSVRRegressor must be fitted before computing weights")
        rng = np.random.default_rng(self.random_seed)
        drift_columns: list[np.ndarray] = []
        for modality in _MODALITIES:
            view = x[:, self.groups_[modality]]
            scaled = self.scalers_[modality].transform(view)
            drift_sum = np.zeros(x.shape[0], dtype=np.float64)
            for _sample in range(self.mc_samples):
                noise = rng.normal(0.0, self.mc_noise_std, size=scaled.shape)
                noisy_prediction = self.base_models_[modality].predict(scaled + noise)
                delta = noisy_prediction - base_predictions[modality]
                drift_sum += np.mean(delta * delta, axis=1)
            drift_columns.append(drift_sum / float(self.mc_samples))
        drift = np.stack(drift_columns, axis=1)
        inverse_uncertainty = 1.0 / (drift + self.baseline_error_constant)
        return inverse_uncertainty / inverse_uncertainty.sum(axis=1, keepdims=True)


REGRESSOR_REGISTRY: dict[str, type[MeanRegressor] | type[RidgeRegressor] | type[DynamicStackingSVRRegressor]] = {
    "mean": MeanRegressor,
    "ridge": RidgeRegressor,
    "dynamic_stacking_svr": DynamicStackingSVRRegressor,
}


def build_regressor(config: str | dict[str, Any] | None = None) -> MeanRegressor | RidgeRegressor | DynamicStackingSVRRegressor:
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
    # float64 for numerical stability in closed-form ridge solve (ill-conditioned design matrices).
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"features must be a 2D array shaped (N, F), got ndim={arr.ndim}")
    if arr.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    return arr


def _as_2d_targets(y: np.ndarray) -> np.ndarray:
    # float64 to match feature dtype in closed-form solve.
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


def _split_modality_columns(feature_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {modality: [] for modality in _MODALITIES}
    for index, name in enumerate(feature_names):
        channel = _feature_channel(name)
        if name.startswith(_ACOUSTIC_PREFIXES):
            groups["acoustic"].append(index)
        elif channel.startswith(_OPTICAL_CHANNEL_PREFIXES):
            groups["optical"].append(index)
        elif channel in _THERMAL_CHANNELS:
            groups["thermal"].append(index)
        elif channel in _ENVIRONMENT_CHANNELS:
            for modality in _MODALITIES:
                groups[modality].append(index)

    missing = [modality for modality, indices in groups.items() if not indices]
    if missing:
        raise ValueError(f"dynamic_stacking_svr requires acoustic, optical, and thermal features; missing {missing}")
    return {modality: np.array(indices, dtype=np.int64) for modality, indices in groups.items()}


def _feature_channel(feature_name: str) -> str:
    parts = feature_name.split(":")
    if len(parts) < 3:
        return feature_name
    return parts[1]


def _validate_feature_name_count(x: np.ndarray, feature_names: tuple[str, ...]) -> None:
    if x.shape[1] != len(feature_names):
        raise ValueError(f"feature_names length {len(feature_names)} does not match feature count {x.shape[1]}")


def _weighted_meta_features(base_predictions: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    blocks = [
        base_predictions[modality] * weights[:, index : index + 1]
        for index, modality in enumerate(_MODALITIES)
    ]
    return np.concatenate(blocks, axis=1).astype(np.float64, copy=False)
