from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ml.features import MLFeatureConfig, MLFeatureMatrix, load_feature_matrix
from ml.metrics import RegressionMetrics, component_regression_metrics, regression_metrics
from ml.models import MeanRegressor, RidgeRegressor, build_regressor


class RegressorProtocol(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> object:
        ...

    def predict(self, x: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    """Predictions and metrics for one evaluated split."""

    split: str
    metrics: RegressionMetrics
    component_metrics: dict[str, RegressionMetrics]
    predictions: np.ndarray
    targets: np.ndarray
    sequence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MLTrainingResult:
    """Result bundle for a fitted traditional ML baseline."""

    model: RegressorProtocol
    feature_config: MLFeatureConfig
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    train_split: str
    evaluations: dict[str, SplitEvaluation]

    @property
    def train_metrics(self) -> RegressionMetrics:
        return self.evaluations[self.train_split].metrics


def evaluate_regressor(model: RegressorProtocol, matrix: MLFeatureMatrix, *, split: str) -> SplitEvaluation:
    """Evaluate a fitted regressor on one feature matrix."""
    predictions = model.predict(matrix.x)
    return SplitEvaluation(
        split=split,
        metrics=regression_metrics(predictions, matrix.y),
        component_metrics=component_regression_metrics(predictions, matrix.y, matrix.label_names),
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
) -> MLTrainingResult:
    """Fit and evaluate a dependency-light traditional ML regressor on a v4 benchmark.

    This is intentionally small: it provides a deterministic baseline path for
    generated benchmark runs without introducing a scikit-learn dependency.
    """
    feature_config = feature_config or MLFeatureConfig()
    train_matrix = load_feature_matrix(dataset_dir, split=train_split, config=feature_config)
    model = build_regressor(model_config)
    model.fit(train_matrix.x, train_matrix.y)

    evaluations: dict[str, SplitEvaluation] = {}
    matrices: dict[str, MLFeatureMatrix] = {train_split: train_matrix}
    for split in eval_splits:
        matrix = matrices.get(split)
        if matrix is None:
            matrix = load_feature_matrix(dataset_dir, split=split, config=feature_config)
            matrices[split] = matrix
        _validate_feature_contract(matrix, train_matrix)
        evaluations[split] = evaluate_regressor(model, matrix, split=split)

    return MLTrainingResult(
        model=model,
        feature_config=feature_config,
        feature_names=train_matrix.feature_names,
        label_names=train_matrix.label_names,
        train_split=train_split,
        evaluations=evaluations,
    )


def _validate_feature_contract(matrix: MLFeatureMatrix, reference: MLFeatureMatrix) -> None:
    if matrix.feature_names != reference.feature_names:
        raise ValueError("feature names must match across train/eval splits")
    if matrix.label_names != reference.label_names:
        raise ValueError("label names must match across train/eval splits")


__all__ = [
    "MeanRegressor",
    "RidgeRegressor",
    "RegressorProtocol",
    "SplitEvaluation",
    "MLTrainingResult",
    "evaluate_regressor",
    "train_regressor_on_dataset",
]
