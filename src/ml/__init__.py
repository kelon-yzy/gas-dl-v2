"""Traditional ML baselines for v4 benchmark regression."""

from ml.features import (
    DEFAULT_SEQUENCE_STATISTICS,
    DEFAULT_WAVEFORM_FRAME_FEATURES,
    MLFeatureConfig,
    MLFeatureMatrix,
    load_feature_matrix,
    sequence_stat_features,
    waveform_stat_features,
)
from ml.metrics import RegressionMetrics, component_regression_metrics, regression_metrics
from ml.models import MeanRegressor, RidgeRegressor, REGRESSOR_REGISTRY, build_regressor
from ml.training import MLTrainingResult, SplitEvaluation, evaluate_regressor, train_regressor_on_dataset

__all__ = [
    "DEFAULT_SEQUENCE_STATISTICS",
    "DEFAULT_WAVEFORM_FRAME_FEATURES",
    "MLFeatureConfig",
    "MLFeatureMatrix",
    "load_feature_matrix",
    "sequence_stat_features",
    "waveform_stat_features",
    "MeanRegressor",
    "RidgeRegressor",
    "REGRESSOR_REGISTRY",
    "build_regressor",
    "RegressionMetrics",
    "regression_metrics",
    "component_regression_metrics",
    "SplitEvaluation",
    "MLTrainingResult",
    "evaluate_regressor",
    "train_regressor_on_dataset",
]
