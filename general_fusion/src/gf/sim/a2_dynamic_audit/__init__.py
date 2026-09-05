"""A2-DYN 审计包：难度资格审计（A2-DYN-3）与完整包冻结审计（A2-DYN-4）。

审计按类别拆分子模块（schema / physics / dynamic / baselines / HEOS
interpolation / jacobian / freeze），本文件保留公开入口 ``run_a2_dynamic_difficulty_audit`` 与
``run_a2_dynamic_freeze_audit`` 及全量符号重导出，调用方 import 路径不变。

审计只读取 ``DynamicDataset`` 和冻结配置。模型可见信号与隐藏 oracle
始终分开：B-LAST 只使用观测序列端点；O-EQ 使用 clean 平衡参照信号；
O-KIN 使用 clean 设备信号反演前向模型可逆性上界；O-KIN-OBS 使用与
O-KIN 完全相同的反演算子但输入最终观测信号，作为噪声受限 headroom
参照（§11.4 门 2 与 §11.6 均以 O-KIN-OBS 为准）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from gf.pipeline.a2_dynamic_protocol import validate_a2_dynamic_records
from gf.sim.a2_dynamic_dataset import DynamicDataset, dynamic_content_sha256
from gf.sim.a2_dynamic_audit._shared import (
    AUDIT_HORIZONS,
    AUDIT_SCHEMA_VERSION,
    DEVELOPMENT_SPLITS,
    EARLY_HORIZONS,
    FAMILIES,
    HEOS_INTERPOLATION_COMPOSITIONS,
    HEOS_INTERPOLATION_GRID_SIZE,
    HEOS_INTERPOLATION_STEP_PCT,
    HEOS_INTERPOLATION_TOF_TOLERANCE_S,
    MIN_UNIQUE_QUANTIZED_LEVELS,
    OBSERVED_ADMISSION_DRIFT_MINUTES,
    OBSERVED_ADMISSION_SIGMA_FACTOR,
    PURGE_COMPOSITION,
    TARGET_TANGENT_DIRECTIONS,
    TARGET_TOTAL,
    _dataset_arrays,
    _horizon_indices,
)
from gf.sim.a2_dynamic_audit._schema import (
    _audit_schema,
    _audit_complete_schema,
    _quantized_record_composition,
)
from gf.sim.a2_dynamic_audit._physics import _audit_physics
from gf.sim.a2_dynamic_audit._dynamic import (
    _audit_dynamic_non_degenerate,
    _phase_index,
    _t50_index,
    _t50_separation_fraction,
    _transition_variance_ratio,
)
from gf.sim.a2_dynamic_audit._baselines import (
    _audit_baselines,
    _endpoint_features,
    _equilibrium_features,
    _fit_small_mlp,
    _kinetic_oracle_predictions,
    _metrics,
    _observed_admission_budgets,
    _paired_late_reference_evidence,
)
from gf.sim.a2_dynamic_audit._heos_interpolation import (
    _registered_heos_interpolated_tof,
)
from gf.sim.a2_dynamic_audit._jacobian import (
    _audit_jacobian,
    _condition_number,
    _equilibrium_jacobian_block,
    _equilibrium_observation,
    _jacobian_sample,
    _per_horizon_jacobian_summary,
    _sensor_responses_at_endpoint,
    _stacked_equilibrium_jacobians,
)
from gf.sim.a2_dynamic_audit._freeze import (
    _audit_pure_boundary,
    run_a2_dynamic_freeze_audit,
)


def run_a2_dynamic_difficulty_audit(
    dataset: DynamicDataset,
    *,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any] | None = None,
    physics_audit: Mapping[str, Any] | None = None,
    a2h_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """运行 A2-DYN-3 全部开发数据审计并返回可序列化证据。"""

    validate_a2_dynamic_records(
        dataset.records,
        data_config,
        require_frozen_counts=False,
    )
    if not isinstance(a2h_config, Mapping):
        raise ValueError(
            "difficulty audit requires the A2H calibration registry (a2h_config) "
            "to build the O-KIN-OBS admission budgets"
        )
    schema = _audit_schema(dataset, data_config)
    physics = _audit_physics(dataset, data_config, eval_config, physics_audit)
    dynamic = _audit_dynamic_non_degenerate(dataset, data_config, experiment_config)
    baselines = _audit_baselines(
        dataset,
        data_config,
        eval_config,
        experiment_config,
        a2h_config,
    )
    jacobian = _audit_jacobian(dataset, data_config, eval_config)

    difficulty_gate = eval_config["qualification_gates"]["dynamic_difficulty"]
    late_reference_horizon = str(difficulty_gate["late_reference_horizon"])
    secondary_late_reference_horizon = str(
        difficulty_gate["secondary_late_reference_horizon"]
    )
    min_paired_rows = int(difficulty_gate["min_paired_rows"])
    qualified_families: list[str] = []
    family_gate: dict[str, Any] = {}
    for family in FAMILIES:
        result = baselines["families"][family]
        early = [
            result["difficulty"][horizon]
            for horizon in EARLY_HORIZONS
            if result["difficulty"].get(horizon) is not None
        ]
        paired_counts = {
            horizon: result["difficulty"][horizon]["paired_row_count"]
            for horizon in EARLY_HORIZONS
            if result["difficulty"].get(horizon) is not None
        }
        # 配对行数不足时显式 FAIL，不允许在残存子集上照常出数（F1）。
        paired_rows_ok = (
            len(paired_counts) == len(EARLY_HORIZONS)
            and all(count >= min_paired_rows for count in paired_counts.values())
        )
        degradation_pass_count = sum(
            item["relative_degradation"] is not None
            and item["relative_degradation"] >= float(difficulty_gate["min_relative_degradation"])
            for item in early
        )
        headroom_pass_count = sum(
            item["oracle_headroom_vs_last"] is not None
            and item["oracle_headroom_vs_last"] >= float(difficulty_gate["min_oracle_headroom_vs_last"])
            for item in early
        )
        family_passed = (
            paired_rows_ok
            and degradation_pass_count >= int(difficulty_gate["min_horizons_passing"])
            and headroom_pass_count >= int(difficulty_gate["min_horizons_passing"])
            and result["baseline_fit_status"] == "PASS"
            and result["oracle_fit_status"] == "PASS"
        )
        family_gate[family] = {
            "status": "QUALIFIED" if family_passed else "FAILED",
            "relative_degradation_pass_count": degradation_pass_count,
            "oracle_headroom_pass_count": headroom_pass_count,
            "minimum_horizons_passing": int(difficulty_gate["min_horizons_passing"]),
            "baseline_fit_status": result["baseline_fit_status"],
            "oracle_fit_status": result["oracle_fit_status"],
            "late_reference_horizon": late_reference_horizon,
            "secondary_late_reference_horizon": secondary_late_reference_horizon,
            "pairing": str(difficulty_gate["pairing"]),
            "min_paired_rows": min_paired_rows,
            "paired_row_counts": paired_counts,
            "paired_rows_ok": paired_rows_ok,
            "max_early_oracle_inversion_failure_fraction": max(
                (
                    item["O-KIN-OBS_val_inversion_failure_fraction"]
                    for item in early
                    if item["O-KIN-OBS_val_inversion_failure_fraction"] is not None
                ),
                default=0.0,
            ),
        }
        if family_passed:
            qualified_families.append(family)

    iid_passed = family_gate["D-IID"]["status"] == "QUALIFIED"
    pressure_axes = [family for family in qualified_families if family != "D-IID"]
    global_physics_ok = bool(schema["status"] == "PASS" and physics["status"] == "PASS")
    global_dynamic_ok = bool(dynamic["status"] == "PASS")
    jacobian_ok = bool(jacobian["status"] == "PASS")
    qualification_passed = bool(
        global_physics_ok
        and global_dynamic_ok
        and jacobian_ok
        and iid_passed
        and len(pressure_axes) >= 2
    )
    failed_requirements = []
    if not global_physics_ok:
        failed_requirements.append("schema_or_physics")
    if not global_dynamic_ok:
        failed_requirements.append("dynamic_non_degenerate")
    if not jacobian_ok:
        failed_requirements.append("jacobian")
    if not iid_passed:
        failed_requirements.append("D-IID")
    if len(pressure_axes) < 2:
        failed_requirements.append("two_independent_pressure_axes")

    result = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage": "A2-DYN-3",
        "status": "DIFFICULTY_QUALIFIED" if qualification_passed else "DIFFICULTY_QUALIFICATION_FAILED",
        "development_only": True,
        "contains_test": False,
        "sample_count": dataset.sample_count,
        "mixture_count": len(set(dataset.group_ids)),
        "schema": schema,
        "physics": physics,
        "dynamic_non_degenerate": dynamic,
        "baselines": baselines,
        "jacobian": jacobian,
        "family_gate": family_gate,
        "qualified_families": qualified_families,
        "eligible_dynamic_axes": pressure_axes,
        "failed_requirements": failed_requirements,
        "content_sha256": dynamic_content_sha256(
            dataset.manifest,
            dataset.records,
            _dataset_arrays(dataset),
        ),
    }
    return result


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "run_a2_dynamic_difficulty_audit",
    "run_a2_dynamic_freeze_audit",
]
