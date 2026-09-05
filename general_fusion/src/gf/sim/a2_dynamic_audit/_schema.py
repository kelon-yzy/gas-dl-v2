"""Schema 审计：开发包视角（A2-DYN-3）与完整包视角（A2-DYN-4）。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from gf.sim.a2_dynamic_dataset import DynamicDataset, dynamic_content_sha256
from gf.sim.a2_dynamic_audit._shared import (
    DEVELOPMENT_SPLITS,
    FAMILIES,
    _dataset_arrays,
)


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
