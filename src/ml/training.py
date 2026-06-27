from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.composition import (
    aitchison_distance,
    TargetTransformSpec,
    ZeroReplacementAudit,
    inverse_transform_composition_targets,
    resolve_target_transform_for_training,
    transform_composition_targets,
)
from common.metrics import CompositionalMetrics, conditional_component_metrics
from ml.features import MLFeatureConfig, MLFeatureMatrix, load_feature_matrix
from ml.metrics import RegressionMetrics, component_regression_metrics, regression_metrics
from ml.models import DynamicStackingSVRRegressor, MeanRegressor, RidgeRegressor, build_regressor


Regressor = MeanRegressor | RidgeRegressor | DynamicStackingSVRRegressor
"""Concrete regressor type alias. Use Protocol only when 5+ regressor types exist (KARPATHY_REVIEW 2.4)."""


def _default_bin_components(label_names: tuple[str, ...]) -> tuple[str, ...]:
    """Pick conditional-metric bin components based on which composition the
    labels describe. hydrogen_ng labels include ``x_N2`` (sum=100% closure);
    syngas labels do not (``x_N2`` is background, ``x_CO`` replaces it as the
    fourth target). Falls back to the last component plus ``x_CH4`` when the
    expected names are absent.
    """
    primary = (
        "x_N2"
        if "x_N2" in label_names
        else ("x_CO" if "x_CO" in label_names else label_names[-1])
    )
    bins = [primary]
    if "x_CH4" in label_names and "x_CH4" != primary:
        bins.append("x_CH4")
    return tuple(bins)


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    """Predictions and metrics for one evaluated split."""

    split: str
    metrics: RegressionMetrics
    component_metrics: dict[str, RegressionMetrics]
    compositional_metrics: CompositionalMetrics | None
    conditional_metrics: dict[str, dict[str, object]]
    sum_abs_error: float | None
    predictions: np.ndarray
    targets: np.ndarray
    sequence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MLTrainingResult:
    """Result bundle for a fitted traditional ML baseline."""

    model: Regressor
    feature_config: MLFeatureConfig
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    train_split: str
    evaluations: dict[str, SplitEvaluation]
    target_transform: TargetTransformSpec | None = None
    target_transform_audits: dict[str, ZeroReplacementAudit] | None = None

    @property
    def train_metrics(self) -> RegressionMetrics:
        return self.evaluations[self.train_split].metrics


def evaluate_regressor(
    model: Regressor,
    matrix: MLFeatureMatrix,
    *,
    split: str,
    target_transform: TargetTransformSpec | None = None,
) -> SplitEvaluation:
    """Evaluate a fitted regressor on one feature matrix."""
    predictions = model.predict(matrix.x)
    if target_transform is not None:
        predictions = inverse_transform_composition_targets(
            predictions,
            target_transform,
            component_names=matrix.label_names,
        )
    compositional_metrics = (
        _compositional_metrics(predictions, matrix.y, target_transform) if target_transform is not None else None
    )
    # sum=100% 闭包对 hg labels (含 x_N2) 才有意义；syngas labels 是 4 列 sum<100，
    # 强算会得到约 |sum - x_N2 - 100| 的无意义数值，置为 None 跳过。
    if "x_N2" in matrix.label_names:
        sum_abs_error: float | None = float(np.mean(np.abs(predictions.sum(axis=1) - 100.0)))
    else:
        sum_abs_error = None
    return SplitEvaluation(
        split=split,
        metrics=regression_metrics(predictions, matrix.y),
        component_metrics=component_regression_metrics(predictions, matrix.y, matrix.label_names),
        compositional_metrics=compositional_metrics,
        conditional_metrics=conditional_component_metrics(
            predictions,
            matrix.y,
            matrix.label_names,
            bin_components=_default_bin_components(matrix.label_names),
        ),
        sum_abs_error=sum_abs_error,
        predictions=predictions.astype(np.float32, copy=False),
        targets=matrix.y.astype(np.float32, copy=False),
        sequence_ids=matrix.sequence_ids,
    )


def train_regressor_on_dataset(
    dataset_dir: Path | str,
    *,
    model_config: str | dict[str, Any] | None = None,
    feature_config: MLFeatureConfig | None = None,
    train_split: str = "train",
    eval_splits: tuple[str, ...] = ("train", "val", "test", "extrapolation"),
    target_transform: str | dict[str, Any] | None = None,
) -> MLTrainingResult:
    """Fit and evaluate a dependency-light traditional ML regressor on a v4 benchmark.

    This is intentionally small: it provides a deterministic baseline path for
    generated benchmark runs without introducing a scikit-learn dependency.
    """
    feature_config = feature_config or MLFeatureConfig()
    train_matrix = load_feature_matrix(dataset_dir, split=train_split, config=feature_config)
    transform_spec = resolve_target_transform_for_training(target_transform, train_matrix.y)
    model = build_regressor(model_config)
    fit_targets, train_audit = _fit_targets(train_matrix, transform_spec)
    model.fit(train_matrix.x, fit_targets, feature_names=train_matrix.feature_names)

    evaluations: dict[str, SplitEvaluation] = {}
    audits: dict[str, ZeroReplacementAudit] | None = {} if transform_spec is not None else None
    if audits is not None and train_audit is not None:
        audits[train_split] = train_audit
    matrices: dict[str, MLFeatureMatrix] = {train_split: train_matrix}
    for split in eval_splits:
        matrix = matrices.get(split)
        if matrix is None:
            matrix = load_feature_matrix(dataset_dir, split=split, config=feature_config)
            matrices[split] = matrix
        _validate_feature_contract(matrix, train_matrix)
        if audits is not None:
            _unused_targets, audit = transform_composition_targets(
                matrix.y,
                transform_spec,
                component_names=matrix.label_names,
            )
            audits[split] = audit
        evaluations[split] = evaluate_regressor(model, matrix, split=split, target_transform=transform_spec)

    return MLTrainingResult(
        model=model,
        feature_config=feature_config,
        feature_names=train_matrix.feature_names,
        label_names=train_matrix.label_names,
        train_split=train_split,
        evaluations=evaluations,
        target_transform=transform_spec,
        target_transform_audits=audits,
    )


def _validate_feature_contract(matrix: MLFeatureMatrix, reference: MLFeatureMatrix) -> None:
    if matrix.feature_names != reference.feature_names:
        raise ValueError("feature names must match across train/eval splits")
    if matrix.label_names != reference.label_names:
        raise ValueError("label names must match across train/eval splits")


def _fit_targets(
    matrix: MLFeatureMatrix,
    transform_spec: TargetTransformSpec | None,
) -> tuple[np.ndarray, ZeroReplacementAudit | None]:
    if transform_spec is None:
        return matrix.y, None
    return transform_composition_targets(matrix.y, transform_spec, component_names=matrix.label_names)


def _compositional_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    transform_spec: TargetTransformSpec,
) -> CompositionalMetrics:
    distances = aitchison_distance(predictions, targets, epsilon=transform_spec.epsilon)
    return CompositionalMetrics(
        aitchison_mean=float(np.mean(distances)),
        aitchison_rmse=float(np.sqrt(np.mean(distances * distances))),
    )


__all__ = [
    "MeanRegressor",
    "RidgeRegressor",
    "DynamicStackingSVRRegressor",
    "Regressor",
    "SplitEvaluation",
    "MLTrainingResult",
    "evaluate_regressor",
    "train_regressor_on_dataset",
]
