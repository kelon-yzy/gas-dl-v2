"""数据适配器、融合核心、任务头、训练与评估。

公开符号按需加载，轻量评估工具不会再隐式导入完整 Torch 训练栈。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_MODULE_EXPORTS = {
    "gf.dl.contracts": ("ContractError", "UnifiedBatch", "UnifiedSample", "collate_samples"),
    "gf.dl.evaluation": (
        "TARGET_RANGES",
        "GroupAggregates",
        "aggregate_by_group",
        "evaluate_predictions",
        "evaluate_output_constraints",
        "group_bootstrap_comparison",
    ),
    "gf.dl.fusion_core": ("ConcatFusionCore", "FusionCore"),
    "gf.dl.mainstream_architectures": (
        "A2M_MODEL_IDS",
        "A2M_MODEL_SCHEMA_VERSION",
        "A2MMLP",
        "EXPECTED_SENSOR_IDS",
        "EXPECTED_SENSOR_TYPES",
        "FeatureTokenTransformer",
        "TabularResNet",
        "build_a2m_model",
        "validate_a2m_model_config",
    ),
    "gf.dl.sensor_encoders": ("A2ScalarTokenEncoder", "MaskedStatSensorEncoder"),
    "gf.dl.task_heads": (
        "FixedTotalSoftmaxSlotHead",
        "FixedTotalSoftmaxHead",
        "FixedTotalTargetHead",
        "RegressionHead",
        "SharedRegressionHead",
        "SimplexProjectionHead",
        "SparsemaxHead",
        "TargetSlotRegressionHead",
        "VariableTotalCompositionHead",
        "VariableTotalTargetHead",
        "build_task_head",
        "build_tqif_task_head",
        "project_to_simplex",
        "sparsemax",
    ),
    "gf.dl.tqif": (
        "MatchedConcatMLP",
        "SensorCapacityControl",
        "TQIFDiagnostics",
        "TQIF_HEAD_IDS",
        "TQIFModel",
        "TQIFModelDiagnostics",
        "TQIF_QUERY_MODES",
        "TQIF_RECIPE_NAMES",
        "TQIF_MODEL_SCHEMA_VERSION",
        "TQIF_RECIPE_SPECS",
        "TQIFSensorSpec",
        "TQIFScalarSensorEncoder",
        "TQIFFusionCore",
        "TQIFSensorEncoding",
        "TQIFTargetSlot",
        "TQIFTargetSlotRegistry",
        "build_tqif_matched_concat_model",
        "build_tqif_matched_concat_model_from_config",
        "build_tqif_model",
        "build_tqif_model_from_config",
        "load_tqif_checkpoint",
        "sensor_registry_hash",
        "target_slot_registry_hash",
        "validate_tqif_checkpoint_payload",
        "validate_tqif_model_config",
    ),
    "gf.dl.training": (
        "A2FusionModel",
        "TorchTrainingConfig",
        "TrainingResult",
        "TorchConcatMLP",
        "build_a2_model_from_config",
        "parameter_parity_report",
        "prepare_a2_train_val_samples",
        "train_torch_model",
        "trainable_parameter_count",
    ),
    "gf.dl.residual": (
        "ResidualFitResult",
        "apply_residual_learner",
        "fit_residual_learner",
        "residual_targets",
    ),
    "gf.dl.temporal_baselines": (
        "PILOT_PROBE_IDS",
        "fit_pilot_linear_probes",
        "pilot_probe_feature_vector",
    ),
}
_EXPORT_MODULE = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}
__all__ = list(_EXPORT_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
