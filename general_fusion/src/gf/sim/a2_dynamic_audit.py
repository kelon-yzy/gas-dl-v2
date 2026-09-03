"""A2-DYN-3 开发数据的独立资格审计。

审计只读取 ``DynamicDataset`` 和冻结配置。模型可见信号与隐藏 oracle
始终分开：B-LAST 只使用观测序列端点，O-EQ / O-KIN 只用于资格上界审计。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from gf.dl.evaluation import evaluate_output_constraints, evaluate_predictions
from gf.pipeline.a2_dynamic_protocol import validate_a2_dynamic_records
from gf.sim.ar_he_co2 import SYSTEM_DELAY_S
from gf.sim.a2_dynamic_dataset import DynamicDataset, dynamic_content_sha256
from gf.sim.a2_dynamic_physics import evaluate_shared_physics, simulate_first_order_series
from gf.sim.a2_sensor_devices import (
    NDIRDeviceProfile,
    estimate_ndir_equilibrium_co2_series,
    estimate_ultrasonic_tof_series,
)


AUDIT_SCHEMA_VERSION = "gf-a2-dynamic-audit-1"
DEVELOPMENT_SPLITS = ("train", "val", "stress_val")
FAMILIES = (
    "D-IID",
    "D-KINETICS",
    "D-PROTOCOL",
    "D-NOISE-DRIFT",
    "D-ENV-CAL",
    "D-JOINT",
)
EARLY_HORIZONS = ("P015", "P030", "P060")
AUDIT_HORIZONS = ("P005", "P015", "P030", "P060", "P120", "P150")
MIN_UNIQUE_QUANTIZED_LEVELS = 10
TARGET_TOTAL = 100.0
PURGE_COMPOSITION = np.asarray([100.0, 0.0, 0.0], dtype=np.float64)
HEOS_INTERPOLATION_STEP_PCT = 1.0
HEOS_INTERPOLATION_GRID_SIZE = int(round(TARGET_TOTAL / HEOS_INTERPOLATION_STEP_PCT))
HEOS_INTERPOLATION_TOF_TOLERANCE_S = 1.0e-7
TARGET_TANGENT_DIRECTIONS = np.asarray(
    [[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]],
    dtype=np.float64,
)
HEOS_INTERPOLATION_COMPOSITIONS = np.asarray(
    [
        [
            HEOS_INTERPOLATION_STEP_PCT * float(ar_index),
            HEOS_INTERPOLATION_STEP_PCT * float(he_index),
            TARGET_TOTAL
            - HEOS_INTERPOLATION_STEP_PCT * float(ar_index)
            - HEOS_INTERPOLATION_STEP_PCT * float(he_index),
        ]
        for ar_index in range(HEOS_INTERPOLATION_GRID_SIZE + 1)
        for he_index in range(HEOS_INTERPOLATION_GRID_SIZE + 1 - ar_index)
    ],
    dtype=np.float64,
)


def run_a2_dynamic_difficulty_audit(
    dataset: DynamicDataset,
    *,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any] | None = None,
    physics_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """运行 A2-DYN-3 全部开发数据审计并返回可序列化证据。"""

    validate_a2_dynamic_records(
        dataset.records,
        data_config,
        require_frozen_counts=False,
    )
    schema = _audit_schema(dataset, data_config)
    physics = _audit_physics(dataset, data_config, eval_config, physics_audit)
    dynamic = _audit_dynamic_non_degenerate(dataset, data_config, experiment_config)
    baselines = _audit_baselines(dataset, data_config, eval_config)
    jacobian = _audit_jacobian(dataset, data_config, eval_config)

    difficulty_gate = eval_config["qualification_gates"]["dynamic_difficulty"]
    qualified_families: list[str] = []
    family_gate: dict[str, Any] = {}
    for family in FAMILIES:
        result = baselines["families"][family]
        early = [
            result["difficulty"][horizon]
            for horizon in EARLY_HORIZONS
            if result["difficulty"].get(horizon) is not None
        ]
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
        late_valid = result["difficulty"].get("P150") is not None
        family_passed = (
            late_valid
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
            "late_reference_horizon": "P150",
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


def _audit_schema(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_group_counts = {
        split: sum(
            int(data_config["families"][family]["groups_by_split"][split])
            for family in FAMILIES
        )
        for split in DEVELOPMENT_SPLITS
    }
    expected_observation_counts = {
        split: sum(
            int(data_config["families"][family]["observation_rows_by_split"][split])
            for family in FAMILIES
        )
        for split in DEVELOPMENT_SPLITS
    }
    actual_groups = {
        split: len(
            {
                str(record["mixture_id"])
                for record in dataset.records
                if record["split"] == split
            }
        )
        for split in DEVELOPMENT_SPLITS
    }
    actual_observations = {
        split: sum(record["split"] == split for record in dataset.records)
        for split in DEVELOPMENT_SPLITS
    }
    expected_phase_counts = np.asarray(
        [int(phase["timesteps"]) for phase in data_config["phases"]],
        dtype=np.int64,
    )
    actual_phase_counts = np.asarray(
        [
            np.bincount(
                row.astype(np.int64),
                minlength=len(data_config["phases"]),
            )
            
            for row in dataset.phase_id
        ],
        dtype=np.int64,
    )
    checks = {
        "development_only": bool(dataset.manifest.get("development_only") is True),
        "contains_test_false": bool(dataset.manifest.get("contains_test") is False),
        "record_count": actual_observations == expected_observation_counts,
        "group_count": actual_groups == expected_group_counts,
        "no_test_records": all(record["split"] in DEVELOPMENT_SPLITS for record in dataset.records),
        "observation_index_order": bool(
            np.array_equal(dataset.observation_index, np.arange(dataset.sample_count, dtype=np.int64))
        ),
        "quality_fixed_one": bool(np.array_equal(dataset.quality, np.ones_like(dataset.quality))),
        "valid_mask_all_true": bool(np.all(dataset.valid_mask)),
        "phase_coverage": bool(np.all(actual_phase_counts == expected_phase_counts[None, :])),
    }
    expected_hash = dataset.manifest.get("content_sha256")
    actual_hash = dynamic_content_sha256(
        dataset.manifest,
        dataset.records,
        _dataset_arrays(dataset),
    )
    checks["content_hash"] = expected_hash == actual_hash
    checks["unique_observation_ids"] = len(
        {str(record["observation_id"]) for record in dataset.records}
    ) == dataset.sample_count
    composition_by_mixture: dict[str, tuple[float, float, float]] = {}
    for record in dataset.records:
        mixture_id = str(record["mixture_id"])
        composition = (
            float(record["x_Ar_pct"]),
            float(record["x_He_pct"]),
            float(record["x_CO2_pct"]),
        )
        previous = composition_by_mixture.setdefault(mixture_id, composition)
        if previous != composition:
            checks["unique_mixture_compositions"] = False
            break
    else:
        checks["unique_mixture_compositions"] = len(composition_by_mixture) == len(
            {str(record["mixture_id"]) for record in dataset.records}
        )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected_group_counts": expected_group_counts,
        "actual_group_counts": actual_groups,
        "expected_observation_counts": expected_observation_counts,
        "actual_observation_counts": actual_observations,
    }


def _quantized_record_composition(
    record: Mapping[str, Any],
    target_names: Sequence[str],
) -> tuple[float, float, float]:
    """按 0.01 mol% 网格化记录组成（与 records 校验器同一网格）。"""

    return tuple(
        round(float(record[name]) / 0.01) * 0.01 for name in target_names
    )


def _audit_complete_schema(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    """完整包（train / val / stress_val / test，含 pure 顶点）的 schema 审计。

    A2-DYN-3 的 ``_audit_schema`` 只允许开发包视角；本函数以 test 已加入的
    冻结包为对象：4 个 split 计数、区域配额（含 pure 3）、规范 pure 顶点、
    开发与 test 的组成与 group 零交集，以及数组级不变量。
    """

    all_splits = ("train", "val", "stress_val", "test")
    regions_all = ("interior", "near_boundary", "binary", "pure")
    target_names = tuple(str(name) for name in data_config["target_names"])
    if len(target_names) != 3:
        raise ValueError("complete schema audit requires exactly three target names")
    families = data_config["families"]
    expected_group_counts = {
        split: sum(
            int(families[family]["groups_by_split"][split]) for family in FAMILIES
        )
        for split in all_splits
    }
    expected_observation_counts = {
        split: sum(
            int(families[family]["observation_rows_by_split"][split])
            for family in FAMILIES
        )
        for split in all_splits
    }
    actual_groups = {
        split: len(
            {
                str(record["mixture_id"])
                for record in dataset.records
                if record["split"] == split
            }
        )
        for split in all_splits
    }
    actual_observations = {
        split: sum(record["split"] == split for record in dataset.records)
        for split in all_splits
    }
    region_quota = data_config["composition_distribution"]["region_quota_by_split"]
    expected_region_counts = {
        split: {
            region: int(region_quota[split].get(region, 0)) for region in regions_all
        }
        for split in all_splits
    }
    actual_region_counts: dict[str, dict[str, int]] = {
        split: {region: 0 for region in regions_all} for split in all_splits
    }
    for record in dataset.records:
        split = str(record["split"])
        region = str(record["composition_region"])
        if split not in actual_region_counts or region not in actual_region_counts[split]:
            raise ValueError(f"record has an unsupported split or region: {split!r} / {region!r}")
        actual_region_counts[split][region] += 1
    region_groups = {
        split: {region: 0 for region in regions_all} for split in all_splits
    }
    seen_groups: set[str] = set()
    for record in dataset.records:
        mixture_id = str(record["mixture_id"])
        split = str(record["split"])
        region = str(record["composition_region"])
        if mixture_id not in seen_groups:
            region_groups[split][region] += 1
            seen_groups.add(mixture_id)
    canonical_vertices = {
        str(item["mixture_id"]): tuple(
            float(value) for value in item["composition_pct"]
        )
        for item in data_config["composition_distribution"]["pure_vertices"]
    }
    if len(canonical_vertices) != 3:
        raise ValueError("complete schema audit requires exactly three canonical pure vertices")
    pure_records = [
        record
        for record in dataset.records
        if str(record["composition_region"]) == "pure"
    ]
    pure_by_mixture = {
        str(record["mixture_id"]): tuple(
            float(record[name]) for name in target_names
        )
        for record in pure_records
    }
    development_compositions: set[tuple[float, float, float]] = set()
    test_compositions: set[tuple[float, float, float]] = set()
    for record in dataset.records:
        if str(record["composition_region"]) == "pure":
            continue
        quantized = _quantized_record_composition(record, target_names)
        if str(record["split"]) == "test":
            test_compositions.add(quantized)
        else:
            development_compositions.add(quantized)
    expected_phase_counts = np.asarray(
        [int(phase["timesteps"]) for phase in data_config["phases"]],
        dtype=np.int64,
    )
    actual_phase_counts = np.asarray(
        [
            np.bincount(
                row.astype(np.int64),
                minlength=len(data_config["phases"]),
            )
            for row in dataset.phase_id
        ],
        dtype=np.int64,
    )
    checks = {
        "development_only_false": bool(dataset.manifest.get("development_only") is False),
        "contains_test_true": bool(dataset.manifest.get("contains_test") is True),
        "record_count": actual_observations == expected_observation_counts,
        "group_count": actual_groups == expected_group_counts,
        "all_splits_present": all(
            actual_observations[split] > 0 and actual_groups[split] > 0
            for split in all_splits
        ),
        "region_group_counts_match": region_groups == expected_region_counts,
        "pure_vertices_registered": pure_by_mixture == canonical_vertices,
        "pure_only_in_test": all(
            str(record["split"]) == "test" and str(record["family"]) == "D-JOINT"
            for record in pure_records
        ),
        "development_test_compositions_disjoint": not (
            development_compositions & test_compositions
        ),
        "observation_index_order": bool(
            np.array_equal(
                dataset.observation_index,
                np.arange(dataset.sample_count, dtype=np.int64),
            )
        ),
        "quality_fixed_one": bool(np.array_equal(dataset.quality, np.ones_like(dataset.quality))),
        "valid_mask_all_true": bool(np.all(dataset.valid_mask)),
        "phase_coverage": bool(np.all(actual_phase_counts == expected_phase_counts[None, :])),
        "unique_observation_ids": len(
            {str(record["observation_id"]) for record in dataset.records}
        ) == dataset.sample_count,
        "unique_mixture_compositions": len(
            {
                str(record["mixture_id"])
                for record in dataset.records
                if str(record["composition_region"]) != "pure"
            }
        ) == len(
            {
                _quantized_record_composition(record, target_names)
                for record in dataset.records
                if str(record["composition_region"]) != "pure"
            }
        ),
    }
    expected_hash = dataset.manifest.get("content_sha256")
    actual_hash = dynamic_content_sha256(
        dataset.manifest,
        dataset.records,
        _dataset_arrays(dataset),
    )
    checks["content_hash"] = expected_hash == actual_hash
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "expected_group_counts": expected_group_counts,
        "actual_group_counts": actual_groups,
        "expected_observation_counts": expected_observation_counts,
        "actual_observation_counts": actual_observations,
        "expected_region_counts": expected_region_counts,
        "actual_region_group_counts": region_groups,
        "pure_vertex_mixture_ids": sorted(canonical_vertices),
        "development_non_pure_composition_count": len(development_compositions),
        "test_non_pure_composition_count": len(test_compositions),
    }


def _audit_pure_boundary(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
) -> dict[str, Any]:
    """pure 顶点序列的边界审计（目标等于 purge，按 §10.5 单列）。"""

    target_names = tuple(str(name) for name in data_config["target_names"])
    pure_indices = [
        index
        for index, record in enumerate(dataset.records)
        if str(record["composition_region"]) == "pure"
    ]
    signal_bounds = data_config["signal_bounds"]
    observed = np.transpose(dataset.signals[:, :, :, 0], (0, 2, 1))
    observed = observed[pure_indices]
    inlet = np.asarray(dataset.inlet_composition)[pure_indices]
    chamber = np.asarray(dataset.chamber_composition)[pure_indices]
    audit = {
        key: np.asarray(value)[pure_indices] for key, value in dataset.device_audit.items()
    }
    bound_checks: dict[str, bool] = {}
    for channel, sensor_id in enumerate(data_config["sensor_ids"]):
        lower, upper = (float(value) for value in signal_bounds[sensor_id])
        outside = (observed[:, :, channel] < lower) | (observed[:, :, channel] > upper)
        bound_checks[str(sensor_id)] = bool(not np.any(outside))
    composition_by_mixture: dict[str, tuple[float, float, float]] = {}
    for record in dataset.records:
        if str(record["composition_region"]) == "pure":
            composition_by_mixture[str(record["mixture_id"])] = tuple(
                float(record[name]) for name in target_names
            )
    canonical_compositions = {
        tuple(float(value) for value in item["composition_pct"])
        for item in data_config["composition_distribution"]["pure_vertices"]
    }
    # 只有 pure-Ar 的目标等于 purge；pure-He / pure-CO2 顶点目标不同，是
    # 强动态激励（不能与 purge 恒等混为一谈）。这里只核对组成与规范顶点一致，
    # 动力学特征在下方单独报告，不入完整性 gate。
    checks = {
        "finite": bool(
            np.isfinite(observed).all() and np.isfinite(inlet).all() and np.isfinite(chamber).all()
        ),
        "closure": bool(
            np.max(np.abs(inlet.sum(axis=2) - TARGET_TOTAL)) <= 1.0e-5
            and np.max(np.abs(chamber.sum(axis=2) - TARGET_TOTAL)) <= 1.0e-5
        ),
        "nonnegative": bool(np.all(inlet >= 0.0) and np.all(chamber >= 0.0)),
        "signal_bounds": all(bound_checks.values()),
        "ultrasonic_lock": float(np.mean(audit["ultrasonic_lock_status"])) >= 0.95,
        "composition_matches_canonical_vertex": (
            set(composition_by_mixture.values()) == canonical_compositions
            and len(composition_by_mixture) == len(canonical_compositions)
        ),
    }
    clean_all = np.transpose(
        np.asarray(dataset.clean_device_signals)[pure_indices], (0, 2, 1)
    )
    static_rows: list[int] = []
    dynamic_rows: list[int] = []
    for row_index, index in enumerate(pure_indices):
        record = dataset.records[index]
        composition = np.asarray(
            [float(record[name]) for name in target_names], dtype=np.float64
        )
        if np.allclose(composition, PURGE_COMPOSITION, rtol=0.0, atol=1.0e-6):
            static_rows.append(row_index)
        else:
            dynamic_rows.append(row_index)
    if static_rows:
        checks["purge_target_clean_static"] = bool(
            np.max(np.ptp(clean_all[static_rows], axis=1)) <= 1.0e-9
        )
    else:
        checks["purge_target_clean_static"] = False
    if dynamic_rows:
        checks["other_vertex_clean_dynamic"] = bool(
            np.any(np.ptp(clean_all[dynamic_rows], axis=1) > 1.0e-6)
        )
    else:
        checks["other_vertex_clean_dynamic"] = False
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "pure_row_count": len(pure_indices),
        "pure_group_count": len(composition_by_mixture),
        "ndir_saturation_fraction": float(np.mean(audit["ndir_saturation_mask"])),
        "ultrasonic_lock_rate": float(np.mean(audit["ultrasonic_lock_status"])),
        "max_tcd_energy_balance_residual_w": float(
            np.max(np.abs(audit["tcd_energy_balance_residual_w"]))
        ),
    }


def run_a2_dynamic_freeze_audit(
    dataset: DynamicDataset,
    *,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    physics_audit: Mapping[str, Any] | None,
    development_content_sha256: str | None = None,
) -> dict[str, Any]:
    """A2-DYN-4 完整数据包的冻结审计，通过后返回 ``DATA_FROZEN``。

    记录级契约（profile 引用、pure 顶点规范、组成去重、冻结总数）由
    ``validate_a2_dynamic_records`` 全量把关；此处补充数组级 schema、
    非 pure 行的守恒 / 设备 / 动态非退化，以及 pure 顶点的单列边界审计。
    """

    validate_a2_dynamic_records(
        dataset.records,
        data_config,
        require_frozen_counts=True,
    )
    schema = _audit_complete_schema(dataset, data_config)
    non_pure_indices = np.asarray(
        [
            index
            for index, record in enumerate(dataset.records)
            if str(record["composition_region"]) != "pure"
        ],
        dtype=np.int64,
    )
    physics = _audit_physics(
        dataset,
        data_config,
        eval_config,
        physics_audit,
        subset_indices=non_pure_indices,
    )
    dynamic = _audit_dynamic_non_degenerate(
        dataset,
        data_config,
        experiment_config,
        subset_split="test",
        exclude_pure=True,
    )
    pure = _audit_pure_boundary(dataset, data_config)
    schema_ok = schema["status"] == "PASS"
    physics_ok = physics["status"] == "PASS"
    dynamic_ok = dynamic["status"] == "PASS"
    failed_requirements: list[str] = []
    if not schema_ok:
        failed_requirements.append("complete_schema")
    if not physics_ok:
        failed_requirements.append("physics")
    if not dynamic_ok:
        failed_requirements.append("test_dynamic_non_degenerate")
    frozen = schema_ok and physics_ok and dynamic_ok
    content_sha256 = dynamic_content_sha256(
        dataset.manifest,
        dataset.records,
        _dataset_arrays(dataset),
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "stage": "A2-DYN-4",
        "status": "DATA_FROZEN" if frozen else "DATA_FREEZE_FAILED",
        "development_only": False,
        "contains_test": True,
        "sample_count": dataset.sample_count,
        "mixture_count": len(set(dataset.group_ids)),
        "schema": schema,
        "physics": physics,
        "dynamic_non_degenerate_test": dynamic,
        "pure_boundary": pure,
        "failed_requirements": failed_requirements,
        "content_sha256": content_sha256,
        "development_content_sha256": development_content_sha256,
        "test_split_summary": {
            "groups": len(
                {
                    str(record["mixture_id"])
                    for record in dataset.records
                    if record["split"] == "test"
                }
            ),
            "observations": sum(
                record["split"] == "test" for record in dataset.records
            ),
        },
    }


def _audit_physics(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    physics_audit: Mapping[str, Any] | None,
    *,
    subset_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """全包（或指定子集）的守恒、边界与设备审计。

    默认对全部观测统计；``subset_indices`` 只用于 A2-DYN-4 冻结审计剔除
    目标等于 purge 的 pure 顶点序列（它们按 §10.5 单列为边界审计）。
    """

    signal_bounds = data_config["signal_bounds"]
    observed = np.transpose(dataset.signals[:, :, :, 0], (0, 2, 1))
    inlet_composition = np.asarray(dataset.inlet_composition)
    chamber_composition = np.asarray(dataset.chamber_composition)
    device_audit = {key: np.asarray(value) for key, value in dataset.device_audit.items()}
    if subset_indices is not None:
        indices = np.asarray(subset_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= dataset.sample_count):
            raise ValueError("physics audit subset_indices must be valid row indices")
        observed = observed[indices]
        inlet_composition = inlet_composition[indices]
        chamber_composition = chamber_composition[indices]
        device_audit = {key: value[indices] for key, value in device_audit.items()}
    bound_checks: dict[str, bool] = {}
    outside_fraction: dict[str, float] = {}
    for channel, sensor_id in enumerate(data_config["sensor_ids"]):
        lower, upper = (float(value) for value in signal_bounds[sensor_id])
        outside = (observed[:, :, channel] < lower) | (observed[:, :, channel] > upper)
        outside_fraction[str(sensor_id)] = float(np.mean(outside))
        bound_checks[str(sensor_id)] = bool(not np.any(outside))
    inlet_sum_error = np.abs(inlet_composition.sum(axis=2) - TARGET_TOTAL)
    chamber_sum_error = np.abs(chamber_composition.sum(axis=2) - TARGET_TOTAL)
    pilot_gate = data_config.get("pilot_dynamic_gate", {})
    physics_gate = eval_config["qualification_gates"]["physics_and_schema"]
    inlet_sum_tolerance = float(physics_gate["max_inlet_sum_error_pct"])
    chamber_sum_tolerance = float(physics_gate["max_chamber_sum_error_pct"])
    tcd_residual = float(np.max(np.abs(device_audit["tcd_energy_balance_residual_w"])))
    tcd_limit = float(pilot_gate.get("maximum_tcd_energy_residual_w", 1.0e-10))
    ndir_saturation_fraction = float(np.mean(device_audit["ndir_saturation_mask"]))
    ultrasonic_lock_rate = float(np.mean(device_audit["ultrasonic_lock_status"]))
    ultrasonic_peak = np.asarray(device_audit["ultrasonic_peak_correlation"], dtype=np.float64)
    ultrasonic_snr = np.asarray(device_audit["ultrasonic_snr"], dtype=np.float64)
    ultrasonic_uncertainty = np.asarray(
        device_audit["ultrasonic_estimated_tof_uncertainty_s"],
        dtype=np.float64,
    )
    external_checks = {
        "provided": physics_audit is not None,
        "status_pass": physics_audit is not None and physics_audit.get("status") == "PASS",
        "physics_verified": physics_audit is not None and physics_audit.get("physics_status") == "PHYSICS_VERIFIED",
        "heos_grid_consistency": physics_audit is not None and physics_audit.get("checks", {}).get("heos_generator_grid_consistency") is True,
        "heos_off_grid_consistency": physics_audit is not None and physics_audit.get("checks", {}).get("heos_generator_off_grid_consistency") is True,
        "heos_pressure_direction": physics_audit is not None and physics_audit.get("checks", {}).get("heos_pressure_direction") is True,
        "ndir_zero_and_sensitivity": physics_audit is not None and physics_audit.get("checks", {}).get("ndir_low_co2_sensitivity") is True,
        "thermal_parity": physics_audit is not None and physics_audit.get("checks", {}).get("steady_thermal_parity") is True,
        "old_speed_migration": physics_audit is not None and "ultrasonic_tof_new_minus_legacy_s" in physics_audit.get("parity", {}),
    }
    checks = {
        "finite_arrays": bool(
            np.isfinite(observed).all()
            and np.isfinite(inlet_composition).all()
            and np.isfinite(chamber_composition).all()
        ),
        "inlet_sum": bool(np.max(inlet_sum_error) <= inlet_sum_tolerance),
        "chamber_sum": bool(np.max(chamber_sum_error) <= chamber_sum_tolerance),
        "inlet_nonnegative": bool(np.all(inlet_composition >= 0.0)),
        "chamber_nonnegative": bool(np.all(chamber_composition >= 0.0)),
        "signal_bounds": all(bound_checks.values()),
        "ultrasonic_lock": ultrasonic_lock_rate >= 0.95,
        "ultrasonic_quality_finite": bool(
            np.isfinite(ultrasonic_peak).all()
            and np.isfinite(ultrasonic_snr).all()
            and np.isfinite(ultrasonic_uncertainty).all()
            and np.all(ultrasonic_peak >= 0.0)
            and np.all(ultrasonic_snr > 0.0)
            and np.all(ultrasonic_uncertainty > 0.0)
        ),
        "ultrasonic_quality_data_dependent": bool(
            np.ptp(ultrasonic_peak) > 0.0
            and np.ptp(ultrasonic_snr) > 0.0
            and np.ptp(ultrasonic_uncertainty) > 0.0
        ),
        "tcd_energy_balance": tcd_residual <= tcd_limit,
        "ndir_unsaturated": ndir_saturation_fraction == 0.0,
        "external_physics_audit": all(external_checks.values()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "signal_outside_fraction": outside_fraction,
        "max_inlet_sum_error_pct": float(np.max(inlet_sum_error)),
        "max_chamber_sum_error_pct": float(np.max(chamber_sum_error)),
        "configured_inlet_sum_tolerance_pct": inlet_sum_tolerance,
        "configured_chamber_sum_tolerance_pct": chamber_sum_tolerance,
        "closure_tolerance_basis": "configured qualification gates applied to serialized float32 oracle arrays",
        "ultrasonic_lock_rate": ultrasonic_lock_rate,
        "ultrasonic_peak_correlation_range": [float(np.min(ultrasonic_peak)), float(np.max(ultrasonic_peak))],
        "ultrasonic_snr_range": [float(np.min(ultrasonic_snr)), float(np.max(ultrasonic_snr))],
        "ultrasonic_uncertainty_range_s": [
            float(np.min(ultrasonic_uncertainty)),
            float(np.max(ultrasonic_uncertainty)),
        ],
        "tcd_max_energy_balance_residual_w": tcd_residual,
        "ndir_saturation_fraction": ndir_saturation_fraction,
        "external_physics_checks": external_checks,
        "audited_row_count": int(observed.shape[0]),
    }


def _audit_dynamic_non_degenerate(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any] | None,
    *,
    subset_split: str | None = None,
    exclude_pure: bool = False,
) -> dict[str, Any]:
    """动态非退化审计（默认全部观测，冻结审计时限定 test 并剔除 pure）。

    pure 顶点目标等于 purge（§10.5），没有可激励的动力学，必须单列为边界
    审计，不能进入幅值 / t50 统计分母。
    """
    if not isinstance(experiment_config, Mapping):
        raise ValueError("dynamic non-degeneracy audit requires the frozen experiment_config")
    experiment_pilot = experiment_config.get("pilot")
    if not isinstance(experiment_pilot, Mapping):
        raise ValueError("dynamic non-degeneracy audit requires experiment_config.pilot")
    noise_base = np.asarray(
        experiment_pilot["observation_noise_std_by_sensor"],
        dtype=np.float64,
    )
    if noise_base.shape != (len(data_config["sensor_ids"]),) or not np.isfinite(noise_base).all():
        raise ValueError("experiment pilot observation_noise_std_by_sensor is invalid")
    if np.any(noise_base < 0.0):
        raise ValueError("experiment pilot observation noise must be non-negative")
    thresholds = experiment_pilot["dynamic_gate"]
    if not isinstance(thresholds, Mapping):
        raise ValueError("dynamic non-degeneracy audit requires pilot.dynamic_gate")
    minimum_t50_separation = int(thresholds["minimum_t50_separation_samples"])
    if minimum_t50_separation < 0:
        raise ValueError("minimum_t50_separation_samples must be non-negative")
    clean = np.transpose(dataset.clean_device_signals, (0, 2, 1))
    observed = np.transpose(dataset.signals[:, :, :, 0], (0, 2, 1))
    active_fraction: dict[str, float] = {}
    quantization_fraction: dict[str, float] = {}
    family_degenerate_fraction: dict[str, float] = {}
    t50_separation_fraction: dict[str, float] = {}
    family_row_counts: dict[str, int] = {}
    for family in FAMILIES:
        indices = dataset.indices(family=family, split=subset_split)
        if exclude_pure:
            indices = np.asarray(
                [
                    index
                    for index in indices
                    if str(dataset.records[int(index)]["composition_region"]) != "pure"
                ],
                dtype=np.int64,
            )
        family_row_counts[family] = int(indices.size)
        if indices.size == 0:
            raise ValueError(f"dynamic audit family {family!r} has no auditable observations")
        clean_family = clean[indices]
        p2p = np.ptp(clean_family, axis=1)
        active = p2p > (5.0 * noise_base)[None, :]
        active_count = np.sum(active, axis=1)
        active_fraction[family] = float(np.mean(active_count >= 2))
        quantized = np.asarray(
            [
                np.min(
                    [
                        np.unique(observed[index, :, channel]).size
                        for channel in range(observed.shape[2])
                    ]
                )
                >= MIN_UNIQUE_QUANTIZED_LEVELS
                for index in indices
            ],
            dtype=bool,
        )
        quantization_fraction[family] = float(np.mean(quantized))
        degenerate = (active_count < 2) | (~quantized)
        family_degenerate_fraction[family] = float(np.mean(degenerate))
        t50_separation_fraction[family] = _t50_separation_fraction(
            clean_family,
            min_separation_samples=minimum_t50_separation,
        )
    minimum_active = float(thresholds["minimum_active_channel_fraction"])
    minimum_quantized = float(thresholds["minimum_quantized_level_fraction"])
    minimum_t50 = float(thresholds["minimum_t50_pair_fraction"])
    maximum_degenerate = float(thresholds["maximum_family_degenerate_fraction"])
    total_rows = sum(family_row_counts.values())
    global_active_fraction = sum(
        active_fraction[family] * family_row_counts[family]
        for family in FAMILIES
    ) / total_rows
    global_quantized_fraction = sum(
        quantization_fraction[family] * family_row_counts[family]
        for family in FAMILIES
    ) / total_rows
    global_t50_fraction = sum(
        t50_separation_fraction[family] * family_row_counts[family]
        for family in FAMILIES
    ) / total_rows
    checks = {
        "active_channel_fraction": global_active_fraction >= minimum_active,
        "quantized_level_fraction": global_quantized_fraction >= minimum_quantized,
        "t50_pair_fraction": global_t50_fraction >= minimum_t50,
        "family_degenerate_fraction": max(family_degenerate_fraction.values()) <= maximum_degenerate,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "active_channel_fraction": active_fraction,
        "global_active_channel_fraction": global_active_fraction,
        "quantized_level_fraction": quantization_fraction,
        "global_quantized_level_fraction": global_quantized_fraction,
        "t50_pair_fraction": t50_separation_fraction,
        "global_t50_pair_fraction": global_t50_fraction,
        "configured_minimum_t50_separation_samples": minimum_t50_separation,
        "family_degenerate_fraction": family_degenerate_fraction,
    }


def _t50_separation_fraction(clean: np.ndarray, *, min_separation_samples: int) -> float:
    if clean.ndim != 3 or clean.shape[0] == 0:
        raise ValueError("clean family signals must have shape (N,T,3)")
    fractions: list[bool] = []
    for sequence in clean:
        t50 = [_t50_index(sequence[:, channel]) for channel in (1, 2)]
        fractions.append(
            t50[0] is not None
            and t50[1] is not None
            and abs(t50[0] - t50[1]) >= min_separation_samples
        )
    return float(np.mean(fractions))


def _t50_index(values: np.ndarray) -> int | None:
    baseline = float(values[0])
    excursion = values - baseline
    endpoint = float(values[int(np.argmax(np.abs(excursion)))])
    delta = endpoint - baseline
    if abs(delta) <= 0.0:
        return None
    threshold = baseline + 0.5 * delta
    if delta > 0.0:
        candidates = np.flatnonzero(values >= threshold)
    else:
        candidates = np.flatnonzero(values <= threshold)
    return int(candidates[0]) if candidates.size else None


def _audit_baselines(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    horizon_indices = _horizon_indices(dataset.time_s, dataset.records, data_config)
    target_ranges = np.asarray(
        [float(eval_config["target_ranges"][name]) for name in data_config["target_names"]],
        dtype=np.float64,
    )
    family_results: dict[str, Any] = {}
    kinetic_cache: dict[int, dict[str, np.ndarray]] = {}
    heos_interpolation_cache: dict[tuple[float, float], np.ndarray] = {}
    for family in FAMILIES:
        family_result: dict[str, Any] = {
            "baseline_fit_status": "NOT_RUN",
            "oracle_fit_status": "NOT_RUN",
            "horizons": {},
            "difficulty": {},
            "fit_diagnostics": {},
        }
        baseline_fit_ok = True
        oracle_fit_ok = True
        for horizon in AUDIT_HORIZONS:
            raw_indices_by_split = {
                split: dataset.indices(family=family, split=split)
                for split in DEVELOPMENT_SPLITS
            }
            indices_by_split = {
                split: raw_indices[
                    horizon_indices[horizon][raw_indices] >= 0
                ]
                for split, raw_indices in raw_indices_by_split.items()
            }
            endpoints = {
                split: _endpoint_features(dataset.signals, indices, horizon_indices[horizon][indices])
                for split, indices in indices_by_split.items()
            }
            equilibrium = {
                split: _equilibrium_features(
                    dataset.equilibrium_reference_signals,
                    indices,
                    horizon_indices[horizon][indices],
                )
                for split, indices in indices_by_split.items()
            }
            b_last_predictions, b_last_fit = _fit_small_mlp(
                endpoints["train"],
                dataset.target[indices_by_split["train"]],
                [endpoints[split] for split in DEVELOPMENT_SPLITS],
            )
            o_eq_predictions = _fit_ridge(
                equilibrium["train"],
                dataset.target[indices_by_split["train"]],
                [equilibrium[split] for split in DEVELOPMENT_SPLITS],
            )
            o_kin_predictions = {
                split: _kinetic_oracle_predictions(
                    dataset,
                    indices,
                    horizon_indices[horizon][indices],
                    data_config=data_config,
                    kinetic_cache=kinetic_cache,
                    heos_interpolation_cache=heos_interpolation_cache,
                )
                for split, indices in indices_by_split.items()
            }
            o_eq_finite = all(
                np.isfinite(prediction).all()
                for prediction in o_eq_predictions
            )
            o_kin_finite = all(
                np.isfinite(prediction).all()
                for prediction in o_kin_predictions.values()
            )
            baseline_fit_ok = baseline_fit_ok and b_last_fit["status"] == "PASS" and o_eq_finite
            oracle_fit_ok = oracle_fit_ok and o_kin_finite
            family_result["fit_diagnostics"][horizon] = {
                "B-LAST": b_last_fit,
                "O-EQ": {
                    "status": "PASS" if o_eq_finite else "FAIL",
                    "finite_predictions": o_eq_finite,
                },
                "O-KIN": {
                    "status": "PASS" if o_kin_finite else "FAIL",
                    "finite_predictions": o_kin_finite,
                },
            }
            split_metrics: dict[str, Any] = {}
            for split_index, split in enumerate(DEVELOPMENT_SPLITS):
                indices = indices_by_split[split]
                if indices.size == 0:
                    split_metrics[split] = None
                    continue
                targets = dataset.target[indices]
                split_metrics[split] = {
                    "B-LAST": _metrics(
                        targets,
                        b_last_predictions[split_index],
                        dataset.group_ids,
                        indices,
                        target_ranges,
                    ),
                    "O-EQ": _metrics(
                        targets,
                        o_eq_predictions[split_index],
                        dataset.group_ids,
                        indices,
                        target_ranges,
                    ),
                    "O-KIN": _metrics(
                        targets,
                        o_kin_predictions[split],
                        dataset.group_ids,
                        indices,
                        target_ranges,
                    ),
                }
            family_result["horizons"][horizon] = split_metrics
            val_last = split_metrics["val"]["B-LAST"]["macro_RNMAE"]
            val_kin = split_metrics["val"]["O-KIN"]["macro_RNMAE"]
            family_result["difficulty"][horizon] = {
                "B-LAST_val_macro_RNMAE": val_last,
                "O-KIN_val_macro_RNMAE": val_kin,
                "relative_degradation": None,
                "oracle_headroom_vs_last": None,
            }
        late_error = family_result["difficulty"]["P150"]["B-LAST_val_macro_RNMAE"]
        for horizon in EARLY_HORIZONS:
            item = family_result["difficulty"][horizon]
            if late_error > 0.0:
                item["relative_degradation"] = float(
                    item["B-LAST_val_macro_RNMAE"] / late_error - 1.0
                )
            if item["B-LAST_val_macro_RNMAE"] > 0.0:
                item["oracle_headroom_vs_last"] = float(
                    1.0
                    - item["O-KIN_val_macro_RNMAE"]
                    / item["B-LAST_val_macro_RNMAE"]
                )
        family_result["baseline_fit_status"] = "PASS" if baseline_fit_ok else "FAIL"
        family_result["oracle_fit_status"] = "PASS" if oracle_fit_ok else "FAIL"
        family_results[family] = family_result
    return {
        "baseline_registry": ["B-LAST", "O-EQ", "O-KIN"],
        "horizon_order": list(AUDIT_HORIZONS),
        "families": family_results,
    }


def _horizon_indices(
    time_s: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    data_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for horizon in data_config["prefix_horizons"]:
        horizon_id = str(horizon["horizon_id"])
        exposure_after = float(horizon["exposure_after_s"])
        indices = np.empty(len(records), dtype=np.int64)
        for row, record in enumerate(records):
            if horizon_id == "FULL":
                cutoff = float(time_s[-1])
            else:
                cutoff = (
                    float(record["exposure_onset_s"])
                    + exposure_after
                    - float(data_config["dt_s"])
                )
            position = int(np.searchsorted(time_s, cutoff, side="right") - 1)
            if (
                position < 0
                or position >= len(time_s)
                or (
                    horizon_id != "FULL"
                    and cutoff >= float(records[row]["exposure_end_s"])
                )
            ):
                indices[row] = -1
            else:
                indices[row] = position
        result[horizon_id] = indices
    return result


def _endpoint_features(
    signals: np.ndarray,
    indices: np.ndarray,
    endpoints: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(indices, dtype=np.int64)
    if rows.size == 0:
        return np.empty((0, signals.shape[1]), dtype=np.float64)
    return np.asarray(
        [signals[row, :, int(endpoints[position]), 0] for position, row in enumerate(rows)],
        dtype=np.float64,
    )


def _equilibrium_features(
    equilibrium: np.ndarray,
    indices: np.ndarray,
    endpoints: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(indices, dtype=np.int64)
    if rows.size == 0:
        return np.empty((0, equilibrium.shape[1]), dtype=np.float64)
    return np.asarray(
        [equilibrium[row, :, int(endpoints[position])] for position, row in enumerate(rows)],
        dtype=np.float64,
    )


def _fit_small_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if np.asarray(train_x).ndim != 2 or np.asarray(train_x).shape[0] == 0:
        raise ValueError("B-LAST requires a non-empty two-dimensional training feature matrix")
    mean = np.asarray(train_x, dtype=np.float64).mean(axis=0)
    scale = np.asarray(train_x, dtype=np.float64).std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    scaled_train = (train_x - mean) / scale
    model = MLPRegressor(
        hidden_layer_sizes=(24,),
        activation="tanh",
        solver="lbfgs",
        alpha=1.0e-6,
        max_iter=2000,
        tol=1.0e-3,
        random_state=17,
    )
    model.fit(scaled_train, np.asarray(train_y, dtype=np.float64) / 100.0)
    predictions: list[np.ndarray] = []
    for values in eval_x:
        values_array = np.asarray(values, dtype=np.float64)
        predictions.append(
            np.empty((0, train_y.shape[1]), dtype=np.float64)
            if values_array.shape[0] == 0
            else model.predict((values_array - mean) / scale) * 100.0
        )
    n_iter = int(getattr(model, "n_iter_", 0))
    max_iter = int(model.max_iter)
    loss = float(getattr(model, "loss_", math.nan))
    finite_predictions = all(np.isfinite(prediction).all() for prediction in predictions)
    diagnostics = {
        "status": "PASS"
        if n_iter < max_iter and math.isfinite(loss) and finite_predictions
        else "FAIL",
        "solver": str(model.solver),
        "n_iter": n_iter,
        "max_iter": max_iter,
        "loss": loss,
        "finite_predictions": finite_predictions,
    }
    return predictions, diagnostics


def _fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: Sequence[np.ndarray],
) -> list[np.ndarray]:
    mean = np.asarray(train_x, dtype=np.float64).mean(axis=0)
    scale = np.asarray(train_x, dtype=np.float64).std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    model = Ridge(alpha=1.0e-8)
    model.fit((train_x - mean) / scale, np.asarray(train_y, dtype=np.float64))
    predictions: list[np.ndarray] = []
    for values in eval_x:
        values_array = np.asarray(values, dtype=np.float64)
        predictions.append(
            np.empty((0, train_y.shape[1]), dtype=np.float64)
            if values_array.shape[0] == 0
            else model.predict((values_array - mean) / scale)
        )
    return predictions


def _kinetic_oracle_predictions(
    dataset: DynamicDataset,
    indices: np.ndarray,
    endpoints: np.ndarray,
    *,
    data_config: Mapping[str, Any],
    kinetic_cache: dict[int, dict[str, np.ndarray]],
    heos_interpolation_cache: dict[tuple[float, float], np.ndarray],
) -> np.ndarray:
    """用 clean 设备序列和注册动力学参数反演目标组成。"""

    ultrasonic_profile = next(
        item
        for item in data_config["hardware_profiles"]["ultrasonic"]["candidates"]
        if str(item["ultrasonic_profile_id"])
        == str(data_config["hardware_profiles"]["ultrasonic"]["selected_profile_id"])
    )
    ndir_profile = NDIRDeviceProfile.from_mapping(data_config["hardware_profiles"]["ndir"]["profiles"][0])
    model_id = str(data_config["physics_reference"]["eos"]["sound_speed_model_id"])
    result = np.empty((len(indices), 3), dtype=np.float64)
    dt_s = float(dataset.manifest["dt_s"])
    for position, row in enumerate(np.asarray(indices, dtype=np.int64)):
        parameters = dataset.privileged_parameters[row]
        endpoint = int(endpoints[position])
        if endpoint < 0 or endpoint >= dataset.timesteps:
            raise ValueError(f"O-KIN endpoint is invalid at row {row}: {endpoint}")
        cache_entry = kinetic_cache.get(int(row))
        if cache_entry is None:
            mix_response = simulate_first_order_series(
                dataset.inlet_coefficient[row],
                dt_s=dt_s,
                tau_s=float(parameters[0]),
                initial_state=0.0,
            )
            local_response_series = np.column_stack(
                [
                    simulate_first_order_series(
                        mix_response,
                        dt_s=dt_s,
                        tau_s=float(parameters[index]),
                        initial_state=0.0,
                    )
                    for index in (1, 2, 3)
                ]
            )
            local_co2_series = estimate_ndir_equilibrium_co2_series(
                dataset.clean_device_signals[row, 2],
                temperature_k=float(parameters[4]),
                pressure_pa=float(parameters[5]),
                dt_s=dt_s,
                profile=ndir_profile,
                absorbance_scale=float(parameters[8]),
            )
            cache_entry = {
                "local_responses": local_response_series,
                "local_co2": local_co2_series,
            }
            kinetic_cache[int(row)] = cache_entry
        local_responses = np.asarray(cache_entry["local_responses"][endpoint], dtype=np.float64)
        if np.any(local_responses <= 1.0e-12):
            raise ValueError(f"O-KIN response is not identifiable at row {row}, endpoint {endpoint}")
        target_co2 = float(cache_entry["local_co2"][endpoint] / local_responses[2])
        if not math.isfinite(target_co2) or not 0.0 <= target_co2 <= TARGET_TOTAL:
            raise ValueError(f"O-KIN CO2 inversion is outside [0,100] at row {row}")
        acoustic_response = float(local_responses[0])
        observed_tof = float(dataset.clean_device_signals[row, 0, endpoint])
        maximum_he = TARGET_TOTAL - target_co2

        def endpoint_tof(helium_pct: float) -> float:
            target = np.asarray(
                [TARGET_TOTAL - helium_pct - target_co2, helium_pct, target_co2],
                dtype=np.float64,
            )
            local = PURGE_COMPOSITION + acoustic_response * (target - PURGE_COMPOSITION)
            return _registered_heos_interpolated_tof(
                local,
                temperature_k=float(parameters[4]),
                pressure_pa=float(parameters[5]),
                path_length_m=float(ultrasonic_profile["path_length_m"]) * float(parameters[6]),
                sound_speed_model_id=model_id,
                cache=heos_interpolation_cache,
            )

        tof_at_zero_he = endpoint_tof(0.0)
        tof_at_max_he = endpoint_tof(maximum_he)
        if maximum_he <= 1.0e-12:
            if abs(observed_tof - tof_at_zero_he) > HEOS_INTERPOLATION_TOF_TOLERANCE_S:
                raise ValueError(
                    f"O-KIN ultrasonic inversion is not compatible with the zero-He boundary at row {row}"
                )
            helium = 0.0
        else:
            tof_step_s = (
                0.25 if "parabolic" in str(ultrasonic_profile["tof_estimator"]) else 1.0
            ) / float(ultrasonic_profile["adc_rate_hz"])
            boundary_tolerance_s = 0.5 * tof_step_s + HEOS_INTERPOLATION_TOF_TOLERANCE_S
            tof_lower = min(tof_at_zero_he, tof_at_max_he)
            tof_upper = max(tof_at_zero_he, tof_at_max_he)
            if observed_tof < tof_lower - boundary_tolerance_s or observed_tof > tof_upper + boundary_tolerance_s:
                raise ValueError(
                    f"O-KIN ultrasonic inversion exceeds the registered quantization boundary at row {row}: "
                    f"observed_tof={observed_tof:.9g}, range=[{tof_lower:.9g},{tof_upper:.9g}], "
                    f"tolerance_s={boundary_tolerance_s:.3g}"
                )
            observed_tof = min(tof_upper, max(tof_lower, observed_tof))
            direction = 1.0 if tof_at_max_he >= tof_at_zero_he else -1.0
            lower_he = 0.0
            upper_he = maximum_he
            for _ in range(48):
                middle_he = 0.5 * (lower_he + upper_he)
                middle_tof = endpoint_tof(middle_he)
                if direction * middle_tof < direction * observed_tof:
                    lower_he = middle_he
                else:
                    upper_he = middle_he
            helium = 0.5 * (lower_he + upper_he)
        result[position] = np.asarray(
            [TARGET_TOTAL - helium - target_co2, helium, target_co2],
            dtype=np.float64,
        )
        if not np.isfinite(result[position]).all() or not np.allclose(
            result[position].sum(), TARGET_TOTAL, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError(f"O-KIN produced an invalid composition at row {row}")
    return result


def _registered_heos_interpolated_tof(
    composition: np.ndarray,
    *,
    temperature_k: float,
    pressure_pa: float,
    path_length_m: float,
    sound_speed_model_id: str,
    cache: dict[tuple[float, float], np.ndarray],
) -> float:
    cache_key = (float(temperature_k), float(pressure_pa))
    grid = cache.get(cache_key)
    if grid is None:
        reference = evaluate_shared_physics(
            HEOS_INTERPOLATION_COMPOSITIONS,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            path_length_m=1.0,
            sound_speed_model_id=sound_speed_model_id,
        )
        inverse_speed = 1.0 / np.asarray(reference["sound_speed_m_s"], dtype=np.float64)
        grid = np.full(
            (HEOS_INTERPOLATION_GRID_SIZE + 1, HEOS_INTERPOLATION_GRID_SIZE + 1),
            np.nan,
            dtype=np.float64,
        )
        for sample, value in zip(HEOS_INTERPOLATION_COMPOSITIONS, inverse_speed):
            ar_index = int(round(float(sample[0]) / HEOS_INTERPOLATION_STEP_PCT))
            he_index = int(round(float(sample[1]) / HEOS_INTERPOLATION_STEP_PCT))
            grid[ar_index, he_index] = float(value)
        if not np.isfinite(grid[~np.isnan(grid)]).all():
            raise ValueError("registered HEOS interpolation table is not finite")
        cache[cache_key] = grid
    values = np.asarray(composition, dtype=np.float64)
    if (
        values.shape != (3,)
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > TARGET_TOTAL)
    ):
        raise ValueError("registered HEOS interpolation expects a finite composition in [0,100]")
    if not math.isclose(float(values.sum()), TARGET_TOTAL, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("registered HEOS interpolation composition must sum to 100 mol%")
    helium_units = float(values[1] / HEOS_INTERPOLATION_STEP_PCT)
    carbon_dioxide_units = float(values[2] / HEOS_INTERPOLATION_STEP_PCT)
    helium_index = int(math.floor(helium_units))
    carbon_dioxide_index = int(math.floor(carbon_dioxide_units))
    helium_fraction = helium_units - helium_index
    carbon_dioxide_fraction = carbon_dioxide_units - carbon_dioxide_index
    if helium_index + carbon_dioxide_index == HEOS_INTERPOLATION_GRID_SIZE:
        if not math.isclose(helium_fraction, 0.0, abs_tol=1.0e-10) or not math.isclose(
            carbon_dioxide_fraction, 0.0, abs_tol=1.0e-10
        ):
            raise ValueError("registered HEOS interpolation reached an invalid simplex boundary")
        inverse_speed_at_composition = float(grid[0, helium_index])
    elif helium_index + carbon_dioxide_index < HEOS_INTERPOLATION_GRID_SIZE:
        if helium_fraction + carbon_dioxide_fraction <= 1.0:
            vertices = (
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index,
                    ]
                ),
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - 1
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index + 1,
                    ]
                ),
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - 1
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index,
                    ]
                ),
            )
            inverse_speed_at_composition = (
                vertices[0]
                + helium_fraction * (vertices[1] - vertices[0])
                + carbon_dioxide_fraction * (vertices[2] - vertices[0])
            )
        else:
            vertices = (
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - 1
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index + 1,
                    ]
                ),
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - 1
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index,
                    ]
                ),
                float(
                    grid[
                        HEOS_INTERPOLATION_GRID_SIZE
                        - 2
                        - helium_index
                        - carbon_dioxide_index,
                        helium_index + 1,
                    ]
                ),
            )
            inverse_speed_at_composition = (
                (1.0 - carbon_dioxide_fraction) * vertices[0]
                + (1.0 - helium_fraction) * vertices[1]
                + (helium_fraction + carbon_dioxide_fraction - 1.0) * vertices[2]
            )
    else:
        raise ValueError("registered HEOS interpolation composition is outside the simplex")
    inverse_speed_at_composition = float(inverse_speed_at_composition)
    if not math.isfinite(inverse_speed_at_composition) or inverse_speed_at_composition <= 0.0:
        raise ValueError("registered HEOS interpolation produced an invalid inverse speed")
    tof = float(path_length_m) * inverse_speed_at_composition + SYSTEM_DELAY_S
    if not math.isfinite(tof) or tof <= 0.0:
        raise ValueError("registered HEOS interpolation produced an invalid ToF")
    return tof


def _metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    group_ids: Sequence[str],
    indices: np.ndarray,
    target_ranges: np.ndarray,
) -> dict[str, Any]:
    selected_groups = np.asarray(group_ids, dtype=object)[indices]
    metrics = evaluate_predictions(
        targets,
        predictions,
        selected_groups,
        np.arange(len(indices), dtype=np.int64),
        target_ranges=target_ranges,
    )
    metrics["constraints"] = evaluate_output_constraints(
        predictions,
        targets=targets,
        total=TARGET_TOTAL,
    )
    metrics["row_count"] = int(len(indices))
    return metrics


def _audit_jacobian(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    early_endpoints = _horizon_indices(dataset.time_s, dataset.records, data_config)
    samples: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_indices = dataset.indices(family=family, split="stress_val")
        selected = family_indices[:: max(1, len(family_indices) // 12)]
        for row in selected[:12]:
            endpoints = [int(early_endpoints[horizon][row]) for horizon in EARLY_HORIZONS]
            fixed, joint = _stacked_equilibrium_jacobians(
                dataset,
                data_config,
                row=int(row),
                endpoints=endpoints,
            )
            target_columns = joint[:, : len(TARGET_TANGENT_DIRECTIONS)]
            nuisance_columns = joint[:, len(TARGET_TANGENT_DIRECTIONS) :]
            nuisance_rank = int(np.linalg.matrix_rank(nuisance_columns))
            nuisance_projection = nuisance_columns @ np.linalg.lstsq(
                nuisance_columns,
                target_columns,
                rcond=None,
            )[0]
            projected_target = target_columns - nuisance_projection
            samples.append(
                {
                    "family": family,
                    "row": int(row),
                    "horizons": list(EARLY_HORIZONS),
                    "fixed_rank": int(np.linalg.matrix_rank(fixed)),
                    "fixed_expected_rank": len(TARGET_TANGENT_DIRECTIONS),
                    "fixed_condition_number": _condition_number(fixed),
                    "joint_parameter_rank": int(np.linalg.matrix_rank(joint)),
                    "joint_parameter_expected_rank": int(joint.shape[1]),
                    "joint_target_rank": int(np.linalg.matrix_rank(projected_target)),
                    "joint_target_expected_rank": len(TARGET_TANGENT_DIRECTIONS),
                    "joint_nuisance_rank": nuisance_rank,
                    "joint_target_condition_number": _condition_number(projected_target),
                }
            )
    fixed_ranks = np.asarray(
        [item["fixed_rank"] for item in samples], dtype=np.float64
    )
    fixed_conditions = np.asarray(
        [item["fixed_condition_number"] for item in samples], dtype=np.float64
    )
    joint_parameter_ranks = np.asarray(
        [item["joint_parameter_rank"] for item in samples], dtype=np.float64
    )
    joint_target_ranks = np.asarray(
        [item["joint_target_rank"] for item in samples], dtype=np.float64
    )
    joint_target_conditions = np.asarray(
        [item["joint_target_condition_number"] for item in samples], dtype=np.float64
    )
    physics_gate = eval_config["qualification_gates"]["physics_and_schema"]
    rank_fraction = float(physics_gate["min_jacobian_full_rank_fraction"])
    max_condition = float(physics_gate["max_jacobian_p95_condition_number"])
    target_rank = len(TARGET_TANGENT_DIRECTIONS)
    joint_parameter_rank = int(samples[0]["joint_parameter_expected_rank"])
    fixed_rank_fraction = float(np.mean(fixed_ranks >= target_rank))
    joint_parameter_rank_fraction = float(
        np.mean(joint_parameter_ranks >= joint_parameter_rank)
    )
    joint_target_rank_fraction = float(np.mean(joint_target_ranks >= target_rank))
    fixed_p95 = float(np.percentile(fixed_conditions, 95))
    joint_target_p95 = float(np.percentile(joint_target_conditions, 95))
    checks = {
        "fixed_full_rank_fraction": fixed_rank_fraction >= rank_fraction,
        "joint_parameter_full_rank_fraction": joint_parameter_rank_fraction >= rank_fraction,
        "joint_target_full_rank_fraction": joint_target_rank_fraction >= rank_fraction,
        "fixed_p95_condition_number": fixed_p95 <= max_condition,
        "joint_target_p95_condition_number": joint_target_p95 <= max_condition,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "sample_count": len(samples),
        "fixed_full_rank_fraction": fixed_rank_fraction,
        "joint_parameter_full_rank_fraction": joint_parameter_rank_fraction,
        "joint_target_full_rank_fraction": joint_target_rank_fraction,
        "fixed_condition_number_p50": float(np.percentile(fixed_conditions, 50)),
        "fixed_condition_number_p95": fixed_p95,
        "joint_target_condition_number_p50": float(np.percentile(joint_target_conditions, 50)),
        "joint_target_condition_number_p95": joint_target_p95,
        "gate_minimum_full_rank_fraction": rank_fraction,
        "gate_maximum_p95_condition_number": max_condition,
        "basis": "stacked_P015_P030_P060_finite_difference_with_target_projection_off_shared_nuisance_span",
        "parameter_columns": [
            "target_tangent_Ar_minus_He",
            "target_tangent_Ar_minus_CO2",
            "tau_mix_s",
            "acoustic_path_scale",
            "tcs_response_scale",
            "ndir_absorbance_scale",
        ],
        "samples": samples,
    }


def _sensor_responses_at_endpoint(
    coefficient: np.ndarray,
    endpoint: int,
    *,
    dt_s: float,
    tau_mix_s: float,
    tau_transport_s: Sequence[float],
) -> np.ndarray:
    mix_response = simulate_first_order_series(
        coefficient,
        dt_s=dt_s,
        tau_s=tau_mix_s,
        initial_state=0.0,
    )
    responses = np.asarray(
        [
            simulate_first_order_series(
                mix_response,
                dt_s=dt_s,
                tau_s=float(tau),
                initial_state=0.0,
            )[endpoint]
            for tau in tau_transport_s
        ],
        dtype=np.float64,
    )
    if np.any(responses <= 1.0e-12):
        raise ValueError(f"Jacobian response is not positive at endpoint {endpoint}")
    return responses


def _equilibrium_observation(
    data_config: Mapping[str, Any],
    record: Mapping[str, Any],
    target: np.ndarray,
    responses: np.ndarray,
    *,
    temperature_k: float,
    pressure_pa: float,
    acoustic_path_scale: float,
    tcd_response_scale: float,
    ndir_absorbance_scale: float,
) -> np.ndarray:
    ultrasonic = next(
        item for item in data_config["hardware_profiles"]["ultrasonic"]["candidates"]
        if str(item["ultrasonic_profile_id"]) == str(record["ultrasonic_profile_id"])
    )
    model_id = str(data_config["physics_reference"]["eos"]["sound_speed_model_id"])
    composition = np.asarray(target, dtype=np.float64)
    response_array = np.asarray(responses, dtype=np.float64)
    if composition.shape != (3,) or response_array.shape != (3,):
        raise ValueError("equilibrium observation expects one target and three sensor responses")
    local = np.asarray(
        [PURGE_COMPOSITION + response * (composition - PURGE_COMPOSITION) for response in response_array],
        dtype=np.float64,
    )
    shared = evaluate_shared_physics(
        local,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        path_length_m=float(ultrasonic["path_length_m"]) * acoustic_path_scale,
        sound_speed_model_id=model_id,
    )
    return np.asarray(
        [
            shared["tof_s"][0],
            shared["thermal_voltage_v"][1] * tcd_response_scale,
            shared["ndir_voltage_v"][2] * ndir_absorbance_scale,
        ],
        dtype=np.float64,
    )


def _stacked_equilibrium_jacobians(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    *,
    row: int,
    endpoints: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    if len(endpoints) != len(EARLY_HORIZONS):
        raise ValueError("Jacobian stacking requires the registered early horizons")
    parameters = dataset.privileged_parameters[row]
    dt_s = float(dataset.manifest["dt_s"])
    base_target = np.asarray([60.0, 30.0, 10.0], dtype=np.float64)
    tau_mix = float(parameters[0])
    tau_transport = np.asarray(parameters[1:4], dtype=np.float64)
    temperature_k = float(parameters[4])
    pressure_pa = float(parameters[5])
    acoustic_scale = float(parameters[6])
    tcd_scale = float(parameters[7])
    ndir_scale = float(parameters[8])
    target_step = 1.0e-3
    nuisance_step = 1.0e-3
    tau_step = max(1.0e-3, abs(tau_mix) * 1.0e-3)
    fixed_blocks: list[np.ndarray] = []
    joint_blocks: list[np.ndarray] = []
    output_scale = np.asarray([1.0e-6, 1.0, 1.0], dtype=np.float64)
    for endpoint in endpoints:
        if endpoint < 0 or endpoint >= dataset.timesteps:
            raise ValueError(f"Jacobian endpoint is invalid at row {row}: {endpoint}")
        responses = _sensor_responses_at_endpoint(
            dataset.inlet_coefficient[row],
            endpoint,
            dt_s=dt_s,
            tau_mix_s=tau_mix,
            tau_transport_s=tau_transport,
        )

        def observation(
            target: np.ndarray,
            response_values: np.ndarray = responses,
            *,
            path_multiplier: float = 1.0,
            tcd_multiplier: float = 1.0,
            ndir_multiplier: float = 1.0,
        ) -> np.ndarray:
            return _equilibrium_observation(
                data_config,
                dataset.records[row],
                target,
                response_values,
                temperature_k=temperature_k,
                pressure_pa=pressure_pa,
                acoustic_path_scale=acoustic_scale * path_multiplier,
                tcd_response_scale=tcd_scale * tcd_multiplier,
                ndir_absorbance_scale=ndir_scale * ndir_multiplier,
            )

        target_columns: list[np.ndarray] = []
        for direction in TARGET_TANGENT_DIRECTIONS:
            plus = base_target + target_step * direction
            minus = base_target - target_step * direction
            target_columns.append(
                (observation(plus) - observation(minus)) / (2.0 * target_step)
            )
        tau_plus = _sensor_responses_at_endpoint(
            dataset.inlet_coefficient[row],
            endpoint,
            dt_s=dt_s,
            tau_mix_s=tau_mix + tau_step,
            tau_transport_s=tau_transport,
        )
        tau_minus = _sensor_responses_at_endpoint(
            dataset.inlet_coefficient[row],
            endpoint,
            dt_s=dt_s,
            tau_mix_s=max(0.0, tau_mix - tau_step),
            tau_transport_s=tau_transport,
        )
        nuisance_columns = [
            (observation(base_target, tau_plus) - observation(base_target, tau_minus))
            / (2.0 * tau_step),
            (observation(base_target, path_multiplier=1.0 + nuisance_step)
             - observation(base_target, path_multiplier=1.0 - nuisance_step))
            / (2.0 * nuisance_step),
            (observation(base_target, tcd_multiplier=1.0 + nuisance_step)
             - observation(base_target, tcd_multiplier=1.0 - nuisance_step))
            / (2.0 * nuisance_step),
            (observation(base_target, ndir_multiplier=1.0 + nuisance_step)
             - observation(base_target, ndir_multiplier=1.0 - nuisance_step))
            / (2.0 * nuisance_step),
        ]
        fixed_blocks.append(np.column_stack(target_columns) / output_scale[:, None])
        joint_blocks.append(
            np.column_stack(target_columns + nuisance_columns) / output_scale[:, None]
        )
    return np.vstack(fixed_blocks), np.vstack(joint_blocks)


def _condition_number(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= 1.0e-15:
        return math.inf
    return float(singular_values[0] / singular_values[-1])


def _dataset_arrays(dataset: DynamicDataset) -> dict[str, np.ndarray]:
    return {
        "signals": dataset.signals,
        "valid_mask": dataset.valid_mask,
        "quality": dataset.quality,
        "time_s": dataset.time_s,
        "target": dataset.target,
        "phase_id": dataset.phase_id,
        "observation_index": dataset.observation_index,
        "inlet_composition": dataset.inlet_composition,
        "inlet_coefficient": dataset.inlet_coefficient,
        "chamber_composition": dataset.chamber_composition,
        "equilibrium_reference_signals": dataset.equilibrium_reference_signals,
        "clean_device_signals": dataset.clean_device_signals,
        "device_states": dataset.device_states,
        "privileged_parameters": dataset.privileged_parameters,
        **dict(dataset.device_audit),
    }


__all__ = ["AUDIT_SCHEMA_VERSION", "run_a2_dynamic_difficulty_audit"]
