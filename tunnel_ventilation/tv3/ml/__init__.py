"""Traditional ML baselines for v4 benchmark regression."""

from tv3.ml.features import (
    DEFAULT_SEQUENCE_STATISTICS,
    DEFAULT_WAVEFORM_FRAME_FEATURES,
    MLFeatureConfig,
    MLFeatureMatrix,
    load_feature_matrix,
    sequence_stat_features,
    waveform_stat_features,
)
from tv3.ml.evaluation_protocol import BaselineProtocolResult, run_baseline_protocol
from tv3.ml.metrics import RegressionMetrics, component_regression_metrics, regression_metrics
from tv3.ml.models import DynamicStackingSVRRegressor, MeanRegressor, RidgeRegressor, REGRESSOR_REGISTRY, build_regressor
from tv3.ml.training import MLTrainingResult, Regressor, SplitEvaluation, evaluate_regressor, train_regressor_on_dataset

__all__ = [
    "DEFAULT_SEQUENCE_STATISTICS",
    "DEFAULT_WAVEFORM_FRAME_FEATURES",
    "MLFeatureConfig",
    "MLFeatureMatrix",
    "load_feature_matrix",
    "sequence_stat_features",
    "waveform_stat_features",
    "BaselineProtocolResult",
    "run_baseline_protocol",
    "MeanRegressor",
    "RidgeRegressor",
    "DynamicStackingSVRRegressor",
    "REGRESSOR_REGISTRY",
    "build_regressor",
    "RegressionMetrics",
    "regression_metrics",
    "component_regression_metrics",
    "Regressor",
    "SplitEvaluation",
    "MLTrainingResult",
    "evaluate_regressor",
    "train_regressor_on_dataset",
]
