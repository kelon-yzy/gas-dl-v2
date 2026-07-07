"""深度学习模块 — 数据读取、模型注册与训练编排。"""

from hg.dl.data.scalers import apply_scaler, load_scaler
from hg.dl.data.splits import SPLIT_NAMES, load_splits, resolve_split_indices, split_sequence_ids
from hg.dl.data.augmentation import TimeSeriesAugmentConfig, augment_sequence
from hg.dl.data.dataset import MODALITY_OPTIONS, V4BenchmarkDataset
from hg.dl.models.registry import MODEL_REGISTRY, build_model
from hg.dl.training.losses import LOSS_REGISTRY, build_loss
from hg.dl.training.metrics import RegressionMetrics, component_regression_metrics, regression_metrics
from hg.dl.training.trainer import OPTIMIZER_REGISTRY, Trainer, build_optimizer

__all__ = [
    "V4BenchmarkDataset",
    "TimeSeriesAugmentConfig",
    "MODALITY_OPTIONS",
    "augment_sequence",
    "load_splits",
    "SPLIT_NAMES",
    "resolve_split_indices",
    "split_sequence_ids",
    "load_scaler",
    "apply_scaler",
    "MODEL_REGISTRY",
    "build_model",
    "LOSS_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "Trainer",
    "build_loss",
    "build_optimizer",
    "RegressionMetrics",
    "regression_metrics",
    "component_regression_metrics",
]
