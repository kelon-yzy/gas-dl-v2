"""训练基础组件：loss、metrics、trainer 编排入口。"""

from tv3.dl.training.losses import LOSS_REGISTRY, build_loss
from tv3.dl.training.metrics import RegressionMetrics, component_regression_metrics, regression_metrics
from tv3.dl.training.trainer import (
    OPTIMIZER_REGISTRY,
    EpochMetrics,
    TrainHistory,
    Trainer,
    build_optimizer,
)

__all__ = [
    "EpochMetrics",
    "LOSS_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "TrainHistory",
    "Trainer",
    "RegressionMetrics",
    "build_loss",
    "build_optimizer",
    "component_regression_metrics",
    "regression_metrics",
]
