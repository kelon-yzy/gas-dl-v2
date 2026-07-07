"""深度学习模块 — 数据读取、模型注册与训练编排。"""

from sg.dl.data.scalers import apply_scaler, load_scaler
from sg.dl.data.splits import SPLIT_NAMES, load_splits, resolve_split_indices, split_sequence_ids
from sg.dl.data.augmentation import TimeSeriesAugmentConfig, augment_sequence
from sg.dl.data.dataset import MODALITY_OPTIONS, V4BenchmarkDataset
from sg.dl.models.registry import MODEL_REGISTRY, build_model
from sg.dl.training.losses import LOSS_REGISTRY, build_loss
from sg.dl.training.metrics import RegressionMetrics, component_regression_metrics, regression_metrics
from sg.dl.training.trainer import OPTIMIZER_REGISTRY, Trainer, build_optimizer

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
