"""训练基础组件：loss、metrics 与后续 trainer 编排入口。"""

from dl.training.losses import LOSS_REGISTRY, build_loss
from dl.training.metrics import RegressionMetrics, component_regression_metrics, regression_metrics

__all__ = [
    "LOSS_REGISTRY",
    "build_loss",
    "RegressionMetrics",
    "regression_metrics",
    "component_regression_metrics",
]
