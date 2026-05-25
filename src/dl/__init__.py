"""深度学习模块 — 数据读取、模型注册与训练编排。"""

from dl.data.scalers import apply_scaler, load_scaler
from dl.data.splits import SPLIT_NAMES, load_splits, resolve_split_indices, split_sequence_ids
from dl.data.dataset import MODALITY_OPTIONS, V4BenchmarkDataset
from dl.models.registry import MODEL_REGISTRY, build_model

__all__ = [
    "V4BenchmarkDataset",
    "MODALITY_OPTIONS",
    "load_splits",
    "SPLIT_NAMES",
    "resolve_split_indices",
    "split_sequence_ids",
    "load_scaler",
    "apply_scaler",
    "MODEL_REGISTRY",
    "build_model",
]
