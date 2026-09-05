"""完整包冻结审计（A2-DYN-4）与 pure 顶点边界审计。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from gf.sim.a2_dynamic_dataset import DynamicDataset, dynamic_content_sha256
from gf.sim.a2_dynamic_audit._shared import (
    AUDIT_SCHEMA_VERSION,
    PURGE_COMPOSITION,
    TARGET_TOTAL,
    _dataset_arrays,
)
from gf.sim.a2_dynamic_audit._schema import _audit_complete_schema
from gf.sim.a2_dynamic_audit._physics import _audit_physics
from gf.sim.a2_dynamic_audit._dynamic import _audit_dynamic_non_degenerate
from gf.pipeline.a2_dynamic_protocol import validate_a2_dynamic_records


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
