"""CLI 编排、运行状态、汇总、图表和报告。

公开入口保持兼容，但按需加载，单个 benchmark 不再导入其余训练流水线。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_MODULE_EXPORTS = {
    "gf.pipeline.a0_smoke": ("A0SmokeSummary", "run_a0_smoke"),
    "gf.pipeline.a1_benchmark": ("run_a1",),
    "gf.pipeline.a2_benchmark": (
        "A2ProtocolError",
        "TestUnlockError",
        "assert_test_unlocked",
        "build_run_manifest",
        "compute_split_hash",
        "run_a2",
        "run_a2_protocol",
        "run_a2_smoke",
        "run_a2_torch_concat_validation",
        "run_a2_oof_diagnostic",
        "run_a2_validation",
        "validate_a2_eval_config",
        "validate_a2_experiment_config",
        "validate_a2_train_config",
        "verify_test_unlock_evidence",
    ),
    "gf.pipeline.a2h_benchmark": (
        "A2HProtocolError",
        "A2HTestUnlockError",
        "assert_hard_test_unlocked",
        "build_a2h_run_manifest",
        "compute_a2h_split_hash",
        "run_a2h",
        "run_a2h_algorithm_comparison",
        "run_a2h_difficulty_audit",
        "run_a2h_formal",
        "run_a2h_generation",
        "run_a2h_learning_noise",
        "run_a2h_ood",
        "run_a2h_protocol",
        "run_a2h_smoke",
        "write_a2h_failure_cases",
        "validate_a2h_data_config",
        "validate_a2h_eval_config",
        "validate_a2h_experiment_config",
        "validate_a2h_train_config",
        "verify_hard_test_unlock_evidence",
    ),
    "gf.pipeline.a2m_benchmark": (
        "A2MProtocolError",
        "A2MTestUnlockError",
        "assert_formal_unlocked",
        "build_a2m_run_manifest",
        "run_a2m",
        "run_a2m_a1_reproduction",
        "run_a2m_development",
        "run_a2m_formal",
        "run_a2m_generation",
        "run_a2m_protocol",
        "run_a2m_smoke",
        "validate_a2m_eval_config",
        "validate_a2m_experiment_config",
        "validate_a2m_train_config",
    ),
    "gf.pipeline.a2_dynamic_benchmark": (
        "PLANNED_STAGES",
        "run_a2_dynamic_benchmark",
        "run_a2_dynamic_development_generation",
        "run_a2_dynamic_difficulty_audit",
        "run_a2_dynamic_physics_smoke",
        "run_a2_dynamic_test_generation",
    ),
    "gf.pipeline.a2_dynamic_pilot": ("run_a2_dynamic_pilot",),
    "gf.pipeline.a2_dynamic_protocol": (
        "A2DynamicProtocolError",
        "run_a2_dynamic_protocol",
        "validate_a2_dynamic_configs",
        "validate_a2_dynamic_data_config",
        "validate_a2_dynamic_eval_config",
        "validate_a2_dynamic_experiment_config",
        "validate_a2_dynamic_pilot_config",
        "validate_a2_dynamic_records",
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
