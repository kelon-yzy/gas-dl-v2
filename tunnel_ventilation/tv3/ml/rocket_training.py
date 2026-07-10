from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from tv3.common.metrics import conditional_metrics_to_payload
from tv3.ml.features import MLFeatureMatrix
from tv3.ml.minirocket_features import (
    MiniRocketFeatureConfig,
    build_minirocket_feature_cache,
)
from tv3.ml.models import RidgeRegressor
from tv3.ml.rocket_features import (
    RocketFeatureCache,
    RocketFeatureConfig,
    build_tv3_physics_feature_cache,
    default_cache_dir,
    load_cached_split_feature_matrix,
)
from tv3.ml.mlp_head import MlpHeadConfig, _ScaledMLPRegressor
from tv3.ml.training import SplitEvaluation, evaluate_regressor


DEFAULT_RIDGE_ALPHAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0)


@dataclass(frozen=True, slots=True)
class RocketTrainingResult:
    head: str
    dataset_dir: Path
    cache_dir: Path
    feature_cache: RocketFeatureCache
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    train_split: str
    evaluations: dict[str, SplitEvaluation]
    diagnostics: dict[str, Any]


class _ScaledRidgeCVRegressor:
    def __init__(self, *, alphas: tuple[float, ...]):
        self.scaler = StandardScaler()
        self.model = RidgeCV(alphas=np.asarray(alphas, dtype=np.float64))

    def fit(self, x: np.ndarray, y: np.ndarray, *, feature_names: tuple[str, ...] | None = None) -> _ScaledRidgeCVRegressor:
        x_scaled = self.scaler.fit_transform(np.asarray(x, dtype=np.float64))
        self.model.fit(x_scaled, np.asarray(y, dtype=np.float64))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(np.asarray(x, dtype=np.float64))
        return self.model.predict(x_scaled).astype(np.float32, copy=False)


class _ScaledClosedFormRidgeRegressor:
    def __init__(self, *, alpha: float):
        self.scaler = StandardScaler()
        self.model = RidgeRegressor(alpha=alpha, standardize=False)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: tuple[str, ...] | None = None,
    ) -> _ScaledClosedFormRidgeRegressor:
        x_scaled = self.scaler.fit_transform(np.asarray(x, dtype=np.float64))
        self.model.fit(x_scaled, np.asarray(y, dtype=np.float64), feature_names=feature_names)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(np.asarray(x, dtype=np.float64))
        return self.model.predict(x_scaled).astype(np.float32, copy=False)


class _TabPFNMultiRegressor:
    """TabPFN 多输出回归头。原生单输出，按标签列拆分 per-target 回归器。"""

    def __init__(self, *, device: str = "auto", n_estimators: int = 8, random_state: int = 0):
        from tabpfn import TabPFNRegressor

        self._make = lambda: TabPFNRegressor(
            device=device,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self._models: list = []

    def fit(self, x: np.ndarray, y: np.ndarray, *, feature_names=None) -> "_TabPFNMultiRegressor":
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        x_arr = np.asarray(x, dtype=np.float64)
        self._models = [self._make() for _ in range(y.shape[1])]
        for col, model in enumerate(self._models):
            model.fit(x_arr, y[:, col])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        return np.column_stack([m.predict(x_arr) for m in self._models]).astype(np.float32, copy=False)


def train_tv3_rocket_regressor(
    dataset_dir: Path | str,
    *,
    feature_config: RocketFeatureConfig | MiniRocketFeatureConfig | None = None,
    cache_dir: Path | str | None = None,
    head: str = "ridgecv",
    train_split: str = "train",
    eval_splits: tuple[str, ...] = ("val", "test", "extrapolation"),
    ridge_alphas: tuple[float, ...] = DEFAULT_RIDGE_ALPHAS,
    closed_form_alpha: float = 1.0,
    device: str = "auto",
    mlp_config: MlpHeadConfig | None = None,
) -> RocketTrainingResult:
    dataset_dir = Path(dataset_dir)
    if feature_config is None:
        feature_config = RocketFeatureConfig()
    feature_builder = feature_config.feature_builder
    cache_path = Path(cache_dir) if cache_dir is not None else default_cache_dir(dataset_dir, feature_builder)
    if isinstance(feature_config, MiniRocketFeatureConfig):
        feature_cache = build_minirocket_feature_cache(dataset_dir, cache_dir=cache_path, config=feature_config)
    else:
        feature_cache = build_tv3_physics_feature_cache(dataset_dir, cache_dir=cache_path, config=feature_config)
    train_matrix = load_cached_split_feature_matrix(dataset_dir, cache_path, split=train_split)
    # RocketFeatureCache 与 MiniRocketFeatureCache 字段兼容;统一进 RocketFeatureCache 供 payload 用
    if not isinstance(feature_cache, RocketFeatureCache):
        feature_cache = RocketFeatureCache(
            dataset_dir=feature_cache.dataset_dir,
            cache_dir=feature_cache.cache_dir,
            feature_config=feature_config,
            feature_names=feature_cache.feature_names,
            label_names=feature_cache.label_names,
            split_sequence_counts=feature_cache.split_sequence_counts,
        )
    resolved_mlp_config = mlp_config or MlpHeadConfig(device=device)
    model = _build_head(
        head,
        ridge_alphas=ridge_alphas,
        closed_form_alpha=closed_form_alpha,
        device=device,
        mlp_config=resolved_mlp_config,
    )
    fit_kwargs: dict[str, Any] = {"feature_names": train_matrix.feature_names}
    if head == "mlp":
        val_matrix = load_cached_split_feature_matrix(dataset_dir, cache_path, split="val")
        _validate_feature_contract(val_matrix, train_matrix)
        fit_kwargs["x_val"] = val_matrix.x
        fit_kwargs["y_val"] = val_matrix.y
        fit_kwargs["label_names"] = train_matrix.label_names
    model.fit(train_matrix.x, train_matrix.y, **fit_kwargs)

    evaluations: dict[str, SplitEvaluation] = {}
    for split_name in (train_split, *eval_splits):
        matrix = train_matrix if split_name == train_split else load_cached_split_feature_matrix(dataset_dir, cache_path, split=split_name)
        _validate_feature_contract(matrix, train_matrix)
        evaluations[split_name] = evaluate_regressor(
            model,
            matrix,
            split=split_name,
            composition_scheme="tunnel_ventilation",
        )

    diagnostics = _model_diagnostics(model, head=head, feature_names=train_matrix.feature_names, label_names=train_matrix.label_names)
    return RocketTrainingResult(
        head=head,
        dataset_dir=dataset_dir,
        cache_dir=cache_path,
        feature_cache=feature_cache,
        feature_names=train_matrix.feature_names,
        label_names=train_matrix.label_names,
        train_split=train_split,
        evaluations=evaluations,
        diagnostics=diagnostics,
    )


def rocket_training_payload(result: RocketTrainingResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset_dir": str(result.dataset_dir),
        "cache_dir": str(result.cache_dir),
        "head": result.head,
        "train_split": result.train_split,
        "feature_builder": result.feature_cache.feature_config.feature_builder,
        "feature_config": asdict(result.feature_cache.feature_config),
        "feature_count": len(result.feature_names),
        "label_names": list(result.label_names),
        "diagnostics": result.diagnostics,
        "evaluations": {},
    }
    for split_name, split_eval in result.evaluations.items():
        payload["evaluations"][split_name] = {
            "metrics": asdict(split_eval.metrics),
            "component_metrics": {name: asdict(metric) for name, metric in split_eval.component_metrics.items()},
            "conditional_metrics": conditional_metrics_to_payload(split_eval.conditional_metrics),
            "sum_abs_error": split_eval.sum_abs_error,
            "sequence_count": len(split_eval.sequence_ids),
        }
    return payload


def write_rocket_training_payload(result: RocketTrainingResult, output_path: Path | str) -> dict[str, Any]:
    payload = rocket_training_payload(result)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _build_head(
    head: str,
    *,
    ridge_alphas: tuple[float, ...],
    closed_form_alpha: float,
    device: str = "auto",
    mlp_config: MlpHeadConfig | None = None,
) -> Any:
    if head == "ridgecv":
        return _ScaledRidgeCVRegressor(alphas=ridge_alphas)
    if head == "ridge_closed_form":
        return _ScaledClosedFormRidgeRegressor(alpha=closed_form_alpha)
    if head == "tabpfn":
        return _TabPFNMultiRegressor(device=device)
    if head == "mlp":
        return _ScaledMLPRegressor(config=mlp_config or MlpHeadConfig(device=device))
    raise ValueError(f"unsupported rocket head {head!r}. available=('ridgecv', 'ridge_closed_form', 'tabpfn', 'mlp')")


def _validate_feature_contract(matrix: MLFeatureMatrix, reference: MLFeatureMatrix) -> None:
    if matrix.feature_names != reference.feature_names:
        raise ValueError("cached rocket feature names must match across splits")
    if matrix.label_names != reference.label_names:
        raise ValueError("cached rocket label names must match across splits")


def _model_diagnostics(
    model: Any,
    *,
    head: str,
    feature_names: tuple[str, ...],
    label_names: tuple[str, ...],
) -> dict[str, Any]:
    if head == "ridgecv":
        coef = np.asarray(model.model.coef_, dtype=np.float64)
        selected_alpha = float(model.model.alpha_)
    elif head == "ridge_closed_form":
        assert model.model.coef_ is not None
        coef = np.asarray(model.model.coef_.T, dtype=np.float64)
        selected_alpha = float(model.model.alpha)
    elif head == "tabpfn":
        return {"note": "TabPFN has no linear coefficients; diagnostics unavailable"}
    elif head == "mlp":
        return {
            "model_config": asdict(model.config),
            "hidden_dims": list(model.hidden_dims),
            "parameter_count": model.parameter_count,
            "best_epoch": model.best_epoch,
            "best_val_o2_r2": model.best_val_o2_r2,
            "standardize_targets": model.config.standardize_targets,
        }
    else:
        raise ValueError(f"unsupported diagnostics head {head!r}")
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return {
        "selected_alpha": selected_alpha,
        "coef_norms": {
            label_names[index]: float(np.linalg.norm(coef[index]))
            for index in range(coef.shape[0])
        },
        "top_feature_groups": {
            label_names[index]: _top_feature_groups(coef[index], feature_names)
            for index in range(coef.shape[0])
        },
    }


def _top_feature_groups(coefficients: np.ndarray, feature_names: tuple[str, ...], limit: int = 5) -> list[dict[str, Any]]:
    group_scores: dict[str, float] = {}
    for coefficient, feature_name in zip(coefficients, feature_names, strict=True):
        group_name = _feature_group_name(feature_name)
        group_scores[group_name] = group_scores.get(group_name, 0.0) + float(abs(coefficient))
    ordered = sorted(group_scores.items(), key=lambda item: item[1], reverse=True)
    return [{"group": group, "abs_coef_sum": score} for group, score in ordered[:limit]]


def _feature_group_name(feature_name: str) -> str:
    if "|" not in feature_name:
        return feature_name
    _window, remainder = feature_name.split("|", 1)
    parts = remainder.split(":")
    if len(parts) <= 2:
        return remainder
    return ":".join(parts[:-1])
