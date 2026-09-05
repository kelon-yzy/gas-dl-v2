"""A2-DYN 编排入口。

profile 抽样、动态物理和数据打包由 ``gf.sim`` 实现；本模块只负责阶段前置
条件、文件路径、状态和运行证据的编排。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from gf.sim.ar_he_co2 import A1_SOUND_SPEED_MODEL_ID
from gf.sim.a2_dynamic_physics import (
    audit_coolprop_sound_speed_grid,
    build_inlet_composition,
    evaluate_shared_physics,
    protocol_inlet_coefficient,
    simulate_dynamic_layers,
)
from gf.sim.a2_sensor_devices import (
    acquire_ultrasonic_tof,
    estimate_ultrasonic_tof_series,
    simulate_ndir,
    simulate_tcd,
    ultrasonic_signal_amplitude,
)
from gf.sim.a2dyn_sound_speed import coolprop_runtime_identity
from gf.sim.a2_dynamic_dataset import (
    generate_a2_dynamic_development,
    generate_a2_dynamic_test,
    load_a2_dynamic_dataset,
    rebind_a2_dynamic_source_hashes,
)
from gf.pipeline.a2_dynamic_pilot import run_a2_dynamic_pilot
from gf.pipeline.a2_dynamic_protocol import run_a2_dynamic_protocol
from gf.pipeline.a2_dynamic_protocol import validate_a2_dynamic_configs
from gf.pipeline.a2_dynamic_baselines import (
    run_a2_dynamic_baselines,
    run_a2_dynamic_handoff,
    run_a2_dynamic_replay_smoke,
    run_a2_dynamic_report,
)


PROTOCOL_STAGE = "protocol"
PLANNED_STAGES = (
    "protocol",
    "physics-smoke",
    "pilot",
    "generate-development",
    "audit",
    "generate-test",
    "baselines",
    "replay-smoke",
    "report",
    "handoff",
)


def run_a2_dynamic_benchmark(stage: str, *, project_root: str = ".") -> dict[str, Any]:
    if stage == "physics-smoke":
        return run_a2_dynamic_physics_smoke(project_root=project_root)
    if stage == "pilot":
        return run_a2_dynamic_pilot(project_root=project_root)
    if stage == "generate-development":
        return run_a2_dynamic_development_generation(project_root=project_root)
    if stage == "generate-test":
        return run_a2_dynamic_test_generation(project_root=project_root)
    if stage == "audit":
        return run_a2_dynamic_difficulty_audit(project_root=project_root)
    if stage == "baselines":
        return run_a2_dynamic_baselines(project_root=project_root)
    if stage == "replay-smoke":
        return run_a2_dynamic_replay_smoke(project_root=project_root)
    if stage == "report":
        return run_a2_dynamic_report(project_root=project_root)
    if stage == "handoff":
        return run_a2_dynamic_handoff(project_root=project_root)
    if stage != PROTOCOL_STAGE:
        if stage not in PLANNED_STAGES:
            raise ValueError(f"unknown A2-DYN stage {stage!r}; expected one of {PLANNED_STAGES}")
        raise NotImplementedError(
            f"A2-DYN stage {stage!r} is not implemented; complete the corresponding work package before running it"
        )
    result = run_a2_dynamic_protocol(project_root, verify_reference_assets=True)
    return {"status": result["status"], "stage": stage, "result": result}


def run_a2_dynamic_development_generation(
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """执行 A2-DYN-3 的开发数据生成，不读取或生成 test。"""

    root = _resolve_project_root(project_root)
    config_paths = _dynamic_config_paths(root)
    data_config = _read_json_object(config_paths["data"])
    eval_config = _read_json_object(config_paths["eval"])
    experiment_config = _read_json_object(config_paths["experiment"])
    protocol = run_a2_dynamic_protocol(root, verify_reference_assets=True)
    if protocol["status"] != "PASS":
        raise ValueError(f"A2-DYN-0 prerequisite did not pass: {protocol['status']}")
    pilot = experiment_config.get("pilot")
    if not isinstance(pilot, Mapping) or pilot.get("status") != "PILOT_QUALIFIED":
        raise ValueError("A2-DYN-3 requires the registered PILOT_QUALIFIED result")
    a2h_config = _read_json_object(root / "configs" / "data" / "ar_he_co2_a2h_v2.json")
    source_hashes = _dynamic_dependency_hashes(root)
    data_dir = root / str(data_config["storage"]["data_dir"])
    dataset = generate_a2_dynamic_development(
        data_dir,
        data_config=data_config,
        experiment_config=experiment_config,
        a2h_config=a2h_config,
        eval_config=eval_config,
        source_hashes=source_hashes,
    )
    run_dir = root / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-3-development"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "DEVELOPMENT_GENERATED",
        "stage": "A2-DYN-3",
        "operation": "generate-development",
        "dataset_dir": str(data_dir.relative_to(root)).replace("\\", "/"),
        "sample_count": dataset.sample_count,
        "mixture_count": len(set(dataset.group_ids)),
        "split_counts": dataset.manifest["split_counts"],
        "content_sha256": dataset.manifest["content_sha256"],
        "source_hashes": source_hashes,
        "protocol_status": protocol["status"],
        "pilot_status": pilot["status"],
    }
    _write_json(run_dir / "manifest.json", result)
    _write_json(
        run_dir / "resolved_config.json",
        {
            "data_config": data_config,
            "evaluation_config": eval_config,
            "experiment_config": experiment_config,
            "source_hashes": source_hashes,
        },
    )
    return result


def run_a2_dynamic_difficulty_audit(
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """对冻结完整数据包重跑 A2-DYN-3R2 难度审计与 A2-DYN-4R2 冻结审计。

    A2-DYN-3R2 修复（F1–F6）改变了审计口径但不重新生成数据：本函数从
    冻结的完整包重建开发子集（使用 A2-DYN-4 备份的开发 manifest，内容
    hash 必须逐位复现），在开发子集上运行修正后的难度审计，并在完整包
    上原样重跑冻结审计（不重绑定 source hash，content hash 必须不变）。
    """

    from gf.sim.a2_dynamic_audit import (
        run_a2_dynamic_difficulty_audit as audit_dynamic_dataset,
    )
    from gf.sim.a2_dynamic_audit import run_a2_dynamic_freeze_audit

    root = _resolve_project_root(project_root)
    config_paths = _dynamic_config_paths(root)
    data_config = _read_json_object(config_paths["data"])
    eval_config = _read_json_object(config_paths["eval"])
    experiment_config = _read_json_object(config_paths["experiment"])
    a2h_config = _read_json_object(
        root / str(data_config["source_registry"]["a2h"]["config_path"])
    )
    dataset_dir = root / str(data_config["storage"]["data_dir"])
    dataset = load_a2_dynamic_dataset(dataset_dir)
    if dataset.manifest.get("contains_test") is not True or (
        dataset.manifest.get("development_only") is not False
    ):
        raise ValueError(
            "A2-DYN-3R2 re-audit requires the frozen full package "
            "(contains_test=true, development_only=false); "
            "development-only packages are historical A2-DYN-3 artifacts"
        )
    development = _development_subset_of_frozen_package(dataset, root)
    freshness = _frozen_reaudit_freshness(
        root,
        dataset.manifest,
        data_config=data_config,
        eval_config=eval_config,
        experiment_config=experiment_config,
    )
    physics_audit = run_a2_dynamic_physics_smoke(root)
    if physics_audit["status"] != "PASS":
        raise ValueError(
            "A2-DYN-1 physics smoke must pass before the re-audit: "
            f"{physics_audit['status']}"
        )
    difficulty_audit = audit_dynamic_dataset(
        development,
        data_config=data_config,
        eval_config=eval_config,
        experiment_config=experiment_config,
        physics_audit=physics_audit,
        a2h_config=a2h_config,
    )
    difficulty_audit["stage"] = "A2-DYN-3R2"
    difficulty_audit["dataset_freshness"] = freshness
    difficulty_audit["physics_audit_refresh"] = {
        "status": "PASS",
        "source": "run_a2_dynamic_physics_smoke",
        "dependency_hashes": physics_audit["dependency_hashes"],
    }
    difficulty_audit["reaudit_scope"] = {
        "work_package": "A2-DYN-3R2",
        "criteria_revision": "F1-F6 remediation (docs/algorithm/archive/15)",
        "data_regenerated": False,
        "development_content_source": (
            "outputs/runs/a2_dynamic_v1/a2-dyn-4-test/development_subset_backup"
        ),
        "full_package_content_sha256": dataset.manifest["content_sha256"],
    }
    difficulty_sha256 = _canonical_json_sha256(difficulty_audit)
    difficulty_audit["audit_sha256"] = difficulty_sha256

    freeze_audit = run_a2_dynamic_freeze_audit(
        dataset,
        data_config=data_config,
        eval_config=eval_config,
        experiment_config=experiment_config,
        physics_audit=physics_audit,
        development_content_sha256=str(development.manifest["content_sha256"]),
    )
    freeze_audit["stage"] = "A2-DYN-4R2"
    freeze_audit["dataset_freshness"] = freshness
    freeze_audit["physics_audit_refresh"] = {
        "status": "PASS",
        "source": "run_a2_dynamic_physics_smoke",
        "dependency_hashes": physics_audit["dependency_hashes"],
    }
    freeze_audit["reaudit_scope"] = {
        "work_package": "A2-DYN-4R2",
        "criteria_revision": "F4/F6 remediation (docs/algorithm/archive/15)",
        "data_regenerated": False,
        "source_hashes_rebound": False,
    }
    freeze_sha256 = _canonical_json_sha256(freeze_audit)
    freeze_audit["audit_sha256"] = freeze_sha256

    summary_dir = root / "outputs" / "summary" / "a2_dynamic_v1"
    run_dir = root / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-3r2-reaudit"
    archive_dir = root / "outputs" / "archive" / "a2_dynamic_v1"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_dir / "a2_dyn_3r2_audit.json", difficulty_audit)
    _write_json(summary_dir / "a2_dyn_4r2_freeze_audit.json", freeze_audit)
    eligible_path = summary_dir / "eligible_dynamic_axes.json"
    if eligible_path.exists():
        _write_json(
            archive_dir / "eligible_dynamic_axes_a2dyn3_r1.json",
            _read_json_object(eligible_path),
        )
    _write_json(
        eligible_path,
        {
            "schema_version": difficulty_audit["schema_version"],
            "stage": difficulty_audit["stage"],
            "status": difficulty_audit["status"],
            "development_only": difficulty_audit["development_only"],
            "contains_test": difficulty_audit["contains_test"],
            "content_sha256": difficulty_audit["content_sha256"],
            "audit_sha256": difficulty_sha256,
            "qualified_families": difficulty_audit["qualified_families"],
            "eligible_dynamic_axes": difficulty_audit["eligible_dynamic_axes"],
            "failed_requirements": difficulty_audit["failed_requirements"],
        },
    )
    _write_json(run_dir / "audit_manifest.json", difficulty_audit)
    _write_json(run_dir / "freeze_audit_manifest.json", freeze_audit)
    _write_json(
        run_dir / "resolved_config.json",
        {
            "data_config": data_config,
            "evaluation_config": eval_config,
            "experiment_config": experiment_config,
            "a2h_config": a2h_config,
        },
    )
    _write_json(dataset_dir / "audit.json", freeze_audit)
    data_manifest = dict(dataset.manifest)
    data_manifest["audit_status"] = freeze_audit["status"]
    data_manifest["audit_sha256"] = freeze_sha256
    data_manifest["status"] = freeze_audit["status"]
    _write_json(dataset_dir / "manifest.json", data_manifest)
    freeze_passed = freeze_audit["status"] == "DATA_FROZEN"
    overall_status = (
        difficulty_audit["status"]
        if (difficulty_audit["status"] != "DIFFICULTY_QUALIFIED" or freeze_passed)
        else "DATA_FREEZE_FAILED"
    )
    return {
        "status": overall_status,
        "stage": "A2-DYN-3R2",
        "operation": "reaudit",
        "difficulty_status": difficulty_audit["status"],
        "freeze_status": freeze_audit["status"],
        "dataset_dir": str(dataset_dir.relative_to(root)).replace("\\", "/"),
        "development_content_sha256": difficulty_audit["content_sha256"],
        "full_package_content_sha256": dataset.manifest["content_sha256"],
        "difficulty_audit_sha256": difficulty_sha256,
        "freeze_audit_sha256": freeze_sha256,
        "eligible_dynamic_axes": difficulty_audit["eligible_dynamic_axes"],
        "qualified_families": difficulty_audit["qualified_families"],
        "failed_requirements": difficulty_audit["failed_requirements"],
        "freeze_failed_requirements": freeze_audit["failed_requirements"],
        "summary_path": "outputs/summary/a2_dynamic_v1/a2_dyn_3r2_audit.json",
    }


def _development_subset_of_frozen_package(
    dataset: Any,
    root: Path,
) -> Any:
    """从冻结完整包重建 A2-DYN-3 开发子集（开发行在前、test 行在后）。

    开发子集的 manifest 身份来自 A2-DYN-4 备份的原始开发 manifest；重建
    后必须逐位复现其 ``content_sha256``，否则说明数据内容与冻结时不一致。
    """

    from dataclasses import replace

    from gf.sim.a2_dynamic_dataset import (
        _dynamic_array_mapping,
        dynamic_content_sha256,
    )

    records = dataset.records
    development_rows = np.asarray(
        [index for index, record in enumerate(records) if str(record["split"]) != "test"],
        dtype=np.int64,
    )
    test_rows = np.asarray(
        [index for index, record in enumerate(records) if str(record["split"]) == "test"],
        dtype=np.int64,
    )
    count = int(development_rows.size)
    if count == 0 or test_rows.size == 0:
        raise ValueError("frozen package must contain both development and test rows")
    if not np.array_equal(development_rows, np.arange(count, dtype=np.int64)) or (
        not np.array_equal(test_rows, np.arange(count, len(records), dtype=np.int64))
    ):
        raise ValueError(
            "frozen package rows must be ordered development-first then test; "
            "cannot reconstruct the A2-DYN-3 development subset"
        )
    backup_manifest_path = (
        root
        / "outputs"
        / "runs"
        / "a2_dynamic_v1"
        / "a2-dyn-4-test"
        / "development_subset_backup"
        / "manifest.json"
    )
    if not backup_manifest_path.is_file():
        raise ValueError(
            "development subset backup manifest is missing: "
            f"{backup_manifest_path.relative_to(root)}"
        )
    development_manifest = _read_json_object(backup_manifest_path)
    subset = replace(
        dataset,
        records=records[:count],
        signals=dataset.signals[:count],
        valid_mask=dataset.valid_mask[:count],
        quality=dataset.quality[:count],
        target=dataset.target[:count],
        phase_id=dataset.phase_id[:count],
        observation_index=np.arange(count, dtype=np.int64),
        inlet_composition=dataset.inlet_composition[:count],
        inlet_coefficient=dataset.inlet_coefficient[:count],
        chamber_composition=dataset.chamber_composition[:count],
        equilibrium_reference_signals=dataset.equilibrium_reference_signals[:count],
        clean_device_signals=dataset.clean_device_signals[:count],
        device_states=dataset.device_states[:count],
        privileged_parameters=dataset.privileged_parameters[:count],
        device_audit={
            key: np.asarray(value)[:count] for key, value in dataset.device_audit.items()
        },
        manifest=development_manifest,
    )
    expected_hash = development_manifest.get("content_sha256")
    actual_hash = dynamic_content_sha256(
        development_manifest,
        subset.records,
        _dynamic_array_mapping(subset),
    )
    if expected_hash != actual_hash:
        raise ValueError(
            "reconstructed development subset does not reproduce the frozen "
            f"content hash: expected {expected_hash}, got {actual_hash}"
        )
    return subset


_REAUDIT_ALLOWED_CHANGED_DEPENDENCIES = frozenset(
    {
        "configs/eval/a2_dynamic_eval.json",
        "configs/experiment/a2_dynamic_protocol.json",
        "src/gf/pipeline/a2_dynamic_benchmark.py",
        "src/gf/pipeline/a2_dynamic_protocol.py",
        "src/gf/sim/a2_dynamic_audit.py",
        "src/gf/sim/a2_dynamic_audit/__init__.py",
        "src/gf/sim/a2_dynamic_audit/_baselines.py",
        "src/gf/sim/a2_dynamic_audit/_dynamic.py",
        "src/gf/sim/a2_dynamic_audit/_freeze.py",
        "src/gf/sim/a2_dynamic_audit/_heos_interpolation.py",
        "src/gf/sim/a2_dynamic_audit/_jacobian.py",
        "src/gf/sim/a2_dynamic_audit/_physics.py",
        "src/gf/sim/a2_dynamic_audit/_schema.py",
        "src/gf/sim/a2_dynamic_audit/_shared.py",
        "src/gf/sim/a2_sensor_devices.py",
    }
)


def _frozen_reaudit_freshness(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> dict[str, Any]:
    """冻结重审计的新鲜度检查：数据内容不变，口径文件白名单内演进。

    常规 freshness（当前配置/源 hash 必须与 manifest 一致）在重审计下必然
    失败——F1–F6 有意修改了审计口径文件。此处改为：数据配置 hash 必须仍与
    冻结值一致；eval / experiment / 源文件只允许审计口径白名单内的差异，
    白名单外的任何源文件变化都说明生成语义可能改变，重审计必须拒绝。
    """

    data_config_match = (
        manifest.get("config_sha256") == _canonical_json_sha256(data_config)
    )
    if not data_config_match:
        raise ValueError(
            "A2-DYN-3R2 re-audit requires the frozen data config to be unchanged"
        )
    stored_hashes = dict(manifest.get("source_hashes", {}))
    current_hashes = _dynamic_dependency_hashes(root)
    changed = sorted(
        key
        for key in set(stored_hashes) | set(current_hashes)
        if stored_hashes.get(key) != current_hashes.get(key)
    )
    unexpected = [key for key in changed if key not in _REAUDIT_ALLOWED_CHANGED_DEPENDENCIES]
    if unexpected:
        raise ValueError(
            "A2-DYN-3R2 re-audit refuses to run: dependencies outside the "
            f"audit-criteria allowlist changed: {unexpected}"
        )
    return {
        "status": "PASS",
        "mode": "frozen_data_reaudit",
        "data_config_sha256_match": True,
        "evaluation_config_sha256_match": (
            manifest.get("evaluation_config_sha256")
            == _canonical_json_sha256(eval_config)
        ),
        "experiment_config_sha256_match": (
            manifest.get("experiment_config_sha256")
            == _canonical_json_sha256(experiment_config)
        ),
        "changed_dependencies": changed,
        "changed_dependency_allowlist": sorted(_REAUDIT_ALLOWED_CHANGED_DEPENDENCIES),
        "note": (
            "A2-DYN-3R2 re-audit: frozen data audited under revised criteria; "
            "content hashes verified unchanged; only audit-criteria dependencies "
            "differ from the frozen manifest binding"
        ),
    }


def run_a2_dynamic_test_generation(
    project_root: str | Path = ".",
) -> dict[str, Any]:
    """执行 A2-DYN-4 的 test 生成、聚合与冻结审计（DATA_FROZEN）。"""

    root = _resolve_project_root(project_root)
    config_paths = _dynamic_config_paths(root)
    data_config = _read_json_object(config_paths["data"])
    eval_config = _read_json_object(config_paths["eval"])
    experiment_config = _read_json_object(config_paths["experiment"])
    protocol = run_a2_dynamic_protocol(root, verify_reference_assets=True)
    if protocol["status"] != "PASS":
        raise ValueError(f"A2-DYN-0 prerequisite did not pass: {protocol['status']}")
    pilot = experiment_config.get("pilot")
    if not isinstance(pilot, Mapping) or pilot.get("status") != "PILOT_QUALIFIED":
        raise ValueError("A2-DYN-4 requires the registered PILOT_QUALIFIED result")
    dataset_dir = root / str(data_config["storage"]["data_dir"])
    development = load_a2_dynamic_dataset(dataset_dir)
    if development.manifest.get("development_only") is not True or (
        development.manifest.get("contains_test") is not False
    ):
        raise ValueError("A2-DYN-4 requires the frozen A2-DYN-3 development-only dataset")
    development_manifest = _read_json_object(dataset_dir / "manifest.json")
    development_content_sha256 = str(development_manifest["content_sha256"])
    run_dir = root / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-4-test"
    backup_dir = run_dir / "development_subset_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    _write_json(backup_dir / "manifest.json", development_manifest)
    development_audit_path = dataset_dir / "audit.json"
    if development_audit_path.exists():
        _write_json(backup_dir / "audit.json", _read_json_object(development_audit_path))
    physics_audit = run_a2_dynamic_physics_smoke(root)
    if physics_audit["status"] != "PASS":
        raise ValueError(
            "A2-DYN-1 physics smoke must pass before A2-DYN-4 test generation: "
            f"{physics_audit['status']}"
        )
    source_hashes = _dynamic_dependency_hashes(root)
    generate_a2_dynamic_test(
        dataset_dir,
        data_config=config_paths["data"],
        experiment_config=config_paths["experiment"],
        a2h_config=root / "configs" / "data" / "ar_he_co2_a2h_v2.json",
        eval_config=config_paths["eval"],
        source_hashes=source_hashes,
    )
    result = _finalize_a2_dynamic_freeze(
        root,
        dataset_dir,
        data_config=data_config,
        eval_config=eval_config,
        experiment_config=experiment_config,
        physics_audit=physics_audit,
        development_content_sha256=development_content_sha256,
        source_hashes=source_hashes,
    )
    result["operation"] = "generate-test"
    result["protocol_status"] = protocol["status"]
    result["pilot_status"] = pilot["status"]
    return result


def _finalize_a2_dynamic_freeze(
    root: Path,
    dataset_dir: Path,
    *,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    physics_audit: Mapping[str, Any],
    development_content_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """对完整数据包运行冻结审计并写出全部产物（可独立重放，不重新生成）。"""

    from gf.sim.a2_dynamic_audit import run_a2_dynamic_freeze_audit

    dataset = rebind_a2_dynamic_source_hashes(
        load_a2_dynamic_dataset(dataset_dir),
        source_hashes,
    )
    audit = run_a2_dynamic_freeze_audit(
        dataset,
        data_config=data_config,
        eval_config=eval_config,
        experiment_config=experiment_config,
        physics_audit=physics_audit,
        development_content_sha256=development_content_sha256,
    )
    freshness = _validate_dynamic_dataset_freshness(
        root,
        dataset.manifest,
        data_config=data_config,
        eval_config=eval_config,
        experiment_config=experiment_config,
    )
    audit["dataset_freshness"] = freshness
    audit["physics_audit_refresh"] = {
        "status": "PASS",
        "source": "run_a2_dynamic_physics_smoke",
        "dependency_hashes": physics_audit["dependency_hashes"],
    }
    audit_sha256 = _canonical_json_sha256(audit)
    audit["audit_sha256"] = audit_sha256
    summary_dir = root / "outputs" / "summary" / "a2_dynamic_v1"
    run_dir = root / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-4-test"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_dir / "a2_dyn_4_freeze_audit.json", audit)
    _write_json(run_dir / "audit_manifest.json", audit)
    _write_json(dataset_dir / "audit.json", audit)
    data_manifest = dict(dataset.manifest)
    data_manifest["status"] = audit["status"]
    data_manifest["audit_status"] = audit["status"]
    data_manifest["audit_sha256"] = audit_sha256
    _write_json(dataset_dir / "manifest.json", data_manifest)
    result = {
        "status": audit["status"],
        "stage": "A2-DYN-4",
        "operation": "finalize-freeze",
        "dataset_dir": str(dataset_dir.relative_to(root)).replace("\\", "/"),
        "sample_count": dataset.sample_count,
        "mixture_count": len(set(dataset.group_ids)),
        "split_counts": data_manifest["split_counts"],
        "content_sha256": audit["content_sha256"],
        "audit_sha256": audit_sha256,
        "development_content_sha256": development_content_sha256,
        "source_hashes": source_hashes,
        "physics_status": physics_audit["physics_status"],
        "summary_path": "outputs/summary/a2_dynamic_v1/a2_dyn_4_freeze_audit.json",
    }
    _write_json(run_dir / "manifest.json", result)
    _write_json(
        run_dir / "resolved_config.json",
        {
            "data_config": data_config,
            "evaluation_config": eval_config,
            "experiment_config": experiment_config,
            "source_hashes": source_hashes,
            "development_content_sha256": development_content_sha256,
        },
    )
    return result


def run_a2_dynamic_physics_smoke(project_root: str | Path = ".") -> dict[str, Any]:
    """执行 A2-DYN-1 解析物理、设备和 HEOS 生成器一致性阶段门。"""

    root = _resolve_project_root(project_root)
    config_path = root / "configs" / "data" / "ar_he_co2_a2_dynamic_v1.json"
    eval_path = root / "configs" / "eval" / "a2_dynamic_eval.json"
    experiment_path = root / "configs" / "experiment" / "a2_dynamic_protocol.json"
    data_config = json.loads(config_path.read_text(encoding="utf-8"))
    eval_config = json.loads(eval_path.read_text(encoding="utf-8"))
    experiment_config = json.loads(experiment_path.read_text(encoding="utf-8"))
    if not isinstance(data_config, dict):
        raise ValueError("A2-DYN data config must be a JSON object")
    protocol_summary = validate_a2_dynamic_configs(root, verify_reference_assets=True)
    sound_speed_model = data_config["physics_reference"]["eos"]
    sound_speed_model_id = sound_speed_model["sound_speed_model_id"]
    time_axis = data_config["time_axis"]
    timesteps = int(time_axis["timesteps"])
    dt_s = float(time_axis["dt_s"])
    time_s = np.arange(timesteps, dtype=np.float64) * dt_s
    target = np.asarray([20.0, 30.0, 50.0], dtype=np.float64)
    coefficient = protocol_inlet_coefficient(
        time_s,
        kind="step",
        onset_s=30.0,
        exposure_end_s=180.0,
    )
    inlet = build_inlet_composition(
        time_s,
        purge_composition_pct=[100.0, 0.0, 0.0],
        target_composition_pct=target,
        coefficient=coefficient,
    )
    transport = data_config["transport"]
    train_transport = next(
        item for item in transport["profiles"] if item["transport_profile_id"] == "KIN-TRAIN"
    )
    tau_transport = {
        "ultrasonic_tof": float(np.mean(train_transport["tau_transport_ultrasonic_s"])),
        "thermal_conductivity_voltage": float(np.mean(train_transport["tau_transport_thermal_s"])),
        "ndir_co2_voltage": float(np.mean(train_transport["tau_transport_ndir_s"])),
    }
    layers = simulate_dynamic_layers(
        inlet,
        dt_s=dt_s,
        tau_mix_s=float(np.mean(train_transport["tau_mix_s"])),
        tau_transport_s=tau_transport,
    )
    environment = next(
        item for item in data_config["environment_profiles"] if item["environment_id"] == "ENV-NOMINAL"
    )
    ultrasonic_profiles = data_config["hardware_profiles"]["ultrasonic"]["candidates"]
    selected_profile_id = data_config["hardware_profiles"]["ultrasonic"]["selected_profile_id"]
    selected_ultrasonic = next(
        item for item in ultrasonic_profiles if item["ultrasonic_profile_id"] == selected_profile_id
    )
    acoustic_path_length_m = float(selected_ultrasonic["path_length_m"])
    shared = evaluate_shared_physics(
        layers.local_composition_pct["ultrasonic_tof"],
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        path_length_m=acoustic_path_length_m,
        sound_speed_model_id=sound_speed_model_id,
    )
    multipath_profiles = {
        item["multipath_profile_id"]: item
        for item in data_config["hardware_profiles"]["ultrasonic"]["multipath_profiles"]
    }
    ultrasonic_results = [
        (
            profile["ultrasonic_profile_id"],
            multipath_id,
            acquire_ultrasonic_tof(
                target,
                temperature_k=environment["temperature_k"],
                pressure_pa=environment["pressure_pa"],
                profile=profile,
                multipath_profile=multipath_profiles[multipath_id],
                signal_amplitude=ultrasonic_signal_amplitude(target, profile),
                sound_speed_model_id=sound_speed_model_id,
            ),
        )
        for profile in ultrasonic_profiles
        for multipath_id in ("US-MP-NOMINAL", "US-MP-OOD")
    ]
    low_frequency_ultrasonic = estimate_ultrasonic_tof_series(shared["tof_s"], selected_ultrasonic)
    thermal_profile = data_config["hardware_profiles"]["thermal"]["profiles"][0]
    tcd = simulate_tcd(
        layers.local_composition_pct["thermal_conductivity_voltage"],
        temperature_k=environment["temperature_k"],
        dt_s=dt_s,
        profile=thermal_profile,
    )
    ndir_profile = data_config["hardware_profiles"]["ndir"]["profiles"][0]
    ndir = simulate_ndir(
        layers.local_composition_pct["ndir_co2_voltage"],
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        dt_s=dt_s,
        profile=ndir_profile,
    )
    steady_compositions = np.asarray([target] * 8)
    steady = evaluate_shared_physics(
        steady_compositions,
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        path_length_m=acoustic_path_length_m,
        sound_speed_model_id=sound_speed_model_id,
    )
    steady_legacy = evaluate_shared_physics(
        steady_compositions,
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        path_length_m=acoustic_path_length_m,
        sound_speed_model_id=A1_SOUND_SPEED_MODEL_ID,
    )
    steady_tcd = simulate_tcd(
        steady_compositions,
        temperature_k=environment["temperature_k"],
        dt_s=dt_s,
        profile=thermal_profile,
    )
    steady_ndir = simulate_ndir(
        steady_compositions,
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        dt_s=dt_s,
        profile=ndir_profile,
    )
    repeated_steady_ndir = simulate_ndir(
        steady_compositions,
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        dt_s=dt_s,
        profile=ndir_profile,
    )
    zero_co2_ndir = simulate_ndir(
        np.repeat(np.asarray([[100.0, 0.0, 0.0]]), 4, axis=0),
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        dt_s=dt_s,
        profile=ndir_profile,
    )
    low_co2_ndir = simulate_ndir(
        np.repeat(np.asarray([[99.5, 0.0, 0.5]]), 4, axis=0),
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        dt_s=dt_s,
        profile=ndir_profile,
    )
    low_co2_delta_v = abs(float(low_co2_ndir.clean_voltage_v[-1] - zero_co2_ndir.clean_voltage_v[-1]))
    eos_reference = data_config["physics_reference"]["eos"]
    eos_grid = eos_reference["query_grid"]
    eos_gate = eos_reference["error_gate"]["max_relative_error"]
    eos_report = audit_coolprop_sound_speed_grid(
        temperature_values_k=eos_grid["temperature_values_k"],
        pressure_values_pa=eos_grid["pressure_values_pa"],
        simplex_step_pct=eos_grid["simplex_step_pct"],
        max_relative_error=eos_gate,
        raise_on_failure=False,
        sound_speed_model_id=sound_speed_model_id,
        off_grid_count=int(eos_reference["off_grid_audit"]["count"]),
        off_grid_seed=int(eos_reference["off_grid_audit"]["seed"]),
        check_pressure_direction=True,
    )
    parity = {
        "thermal_max_absolute_difference": float(
            np.max(np.abs(steady_tcd.clean_voltage_v - steady["thermal_voltage_v"]))
        ),
        "ndir_repeat_max_absolute_difference": float(
            np.max(np.abs(steady_ndir.clean_voltage_v - repeated_steady_ndir.clean_voltage_v))
        ),
        "ultrasonic_tof_new_minus_legacy_s": {
            "max_absolute_difference": float(
                np.max(np.abs(steady["tof_s"] - steady_legacy["tof_s"]))
            ),
            "mean_difference": float(np.mean(steady["tof_s"] - steady_legacy["tof_s"])),
        },
    }
    parity_gate = float(
        eval_config["qualification_gates"]["physics_and_schema"]["max_nominal_parity_absolute_error"]
    )
    checks = {
        "composition_nonnegative": bool(np.all(layers.chamber_composition_pct >= 0.0)),
        "composition_closure": bool(np.allclose(layers.chamber_composition_pct.sum(axis=1), 100.0, atol=1.0e-8)),
        "step_recovery_returns_to_purge": bool(np.all(coefficient[time_s >= 180.0] == 0.0)),
        "ultrasonic_lock": all(item.lock_status for _, _, item in ultrasonic_results),
        "ultrasonic_no_theoretical_fallback": all(item.tof_s is not None for _, _, item in ultrasonic_results),
        "ultrasonic_nominal_and_ood_multipath": {multipath_id for _, multipath_id, _ in ultrasonic_results} == {"US-MP-NOMINAL", "US-MP-OOD"},
        "ultrasonic_low_frequency_estimator": bool(
            np.all(np.isfinite(low_frequency_ultrasonic))
            and np.any(low_frequency_ultrasonic != shared["tof_s"])
        ),
        "tcd_energy_balance": bool(np.max(np.abs(tcd.energy_balance_residual_w)) < 1.0e-10),
        "ndir_no_saturation": bool(ndir.saturation_fraction == 0.0),
        "ndir_low_co2_sensitivity": low_co2_delta_v >= float(
            experiment_config["pilot"]["dynamic_gate"]["minimum_ndir_low_co2_delta_v"]
        ),
        "steady_thermal_parity": bool(parity["thermal_max_absolute_difference"] <= parity_gate),
        "steady_ndir_repeat_parity": bool(parity["ndir_repeat_max_absolute_difference"] <= parity_gate),
        "heos_generator_grid_consistency": eos_report["grid_status"] == "PASS",
        "heos_generator_off_grid_consistency": eos_report["off_grid"]["status"] == "PASS",
        "heos_pressure_direction": eos_report["pressure_direction"]["status"] == "PASS",
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    stage_status = "PASS" if not failed_checks else "PHYSICS_INVALID"
    result = {
        "status": stage_status,
        "stage": "A2-DYN-1R4",
        "physics_status": "PHYSICS_VERIFIED" if stage_status == "PASS" else "PHYSICS_INVALID",
        "verification_scope": sound_speed_model["verification_scope"],
        "independent_physics_validation": sound_speed_model["independent_physics_validation"],
        "protocol_status": protocol_summary["status"],
        "checks": checks,
        "failed_checks": failed_checks,
        "parity": parity,
        "coolprop_eos_grid": eos_report,
        "ultrasonic_quality": {
            f"{profile_id}/{multipath_id}": item.quality
            for profile_id, multipath_id, item in ultrasonic_results
        },
        "selected_ultrasonic_profile_id": selected_profile_id,
        "selected_tof_estimator": selected_ultrasonic["tof_estimator"],
        "ndir_model_id": ndir_profile["effective_absorption_model_id"],
        "ndir_0p5_minus_0molpct_absolute_delta_v": low_co2_delta_v,
        "ndir_saturation_fraction": ndir.saturation_fraction,
        "max_tcd_energy_balance_residual_w": float(np.max(np.abs(tcd.energy_balance_residual_w))),
        "sound_speed_model_id": sound_speed_model_id,
        "runtime_identity": coolprop_runtime_identity(),
        "model_asset_hash": sound_speed_model["model_asset"]["sha256"],
        "dependency_hashes": _physics_dependency_hashes(root, data_config),
    }
    summary_dir = root / "outputs" / "summary" / "a2_dynamic_v1"
    run_dir = root / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-1r4-physics-smoke"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (summary_dir / "physics_audit_r4.json").write_text(payload, encoding="utf-8")
    (run_dir / "manifest.json").write_text(payload, encoding="utf-8")
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "data": data_config,
                "evaluation": eval_config,
                "experiment": experiment_config,
                "dependency_hashes": result["dependency_hashes"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _physics_dependency_hashes(root: Path, data_config: Mapping[str, Any]) -> dict[str, str]:
    relative_paths = [
        "configs/data/ar_he_co2_a2_dynamic_v1.json",
        "configs/eval/a2_dynamic_eval.json",
        "configs/experiment/a2_dynamic_protocol.json",
        "src/gf/__init__.py",
        "src/gf/pipeline/__init__.py",
        "src/gf/pipeline/a2_dynamic_benchmark.py",
        "src/gf/pipeline/a2_dynamic_protocol.py",
        "src/gf/sim/a2_dynamic_physics.py",
        "src/gf/sim/a2_sensor_devices.py",
        "src/gf/sim/__init__.py",
        "src/gf/sim/a2dyn_sound_speed.py",
        "src/gf/sim/ar_he_co2.py",
        str(data_config["physics_reference"]["eos"]["model_asset"]["path"]),
    ]
    for source_id in ("a1", "a2h"):
        source = data_config["source_registry"][source_id]
        relative_paths.extend((str(source["config_path"]), str(source["manifest_path"])))
    relative_paths.extend(
        str(item["path"])
        for item in data_config["physics_reference"]["ndir_reference"]["asset_files"]
    )
    return {
        relative.replace("\\", "/"): _sha256_file((root / relative).resolve())
        for relative in relative_paths
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dynamic_config_paths(root: Path) -> dict[str, Path]:
    return {
        "data": root / "configs" / "data" / "ar_he_co2_a2_dynamic_v1.json",
        "eval": root / "configs" / "eval" / "a2_dynamic_eval.json",
        "experiment": root / "configs" / "experiment" / "a2_dynamic_protocol.json",
    }


def _validate_dynamic_dataset_freshness(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_config_hashes = {
        "config_sha256": _canonical_json_sha256(data_config),
        "evaluation_config_sha256": _canonical_json_sha256(eval_config),
        "experiment_config_sha256": _canonical_json_sha256(experiment_config),
    }
    actual_config_hashes = {
        key: manifest.get(key)
        for key in expected_config_hashes
    }
    if actual_config_hashes != expected_config_hashes:
        raise ValueError(
            "A2-DYN dataset is stale: configuration hashes do not match the current frozen configs"
        )
    expected_source_hashes = _dynamic_dependency_hashes(root)
    actual_source_hashes = manifest.get("source_hashes")
    if actual_source_hashes != expected_source_hashes:
        raise ValueError(
            "A2-DYN dataset is stale: source dependency hashes do not match the current code/assets"
        )
    return {
        "status": "PASS",
        "config_hashes_match": True,
        "source_hashes_match": True,
        "source_hash_count": len(expected_source_hashes),
    }


def _dynamic_dependency_hashes(root: Path) -> dict[str, str]:
    relative_paths = [
        "configs/data/ar_he_co2_a1_v1.json",
        "configs/data/ar_he_co2_a2_dynamic_v1.json",
        "configs/data/ar_he_co2_a2h_v2.json",
        "configs/data/a2dyn_direct_heos_v1.json",
        "configs/eval/a2_dynamic_eval.json",
        "configs/experiment/a2_dynamic_protocol.json",
        "data/a1_formal/manifest.json",
        "data/a2h_v2/manifest.json",
        "outputs/runs/a2_dynamic_v1/a2-dyn-2r4-pilot/manifest.json",
        "src/gf/pipeline/a2_dynamic_benchmark.py",
        "src/gf/pipeline/a2_dynamic_protocol.py",
        "src/gf/sim/a2_dynamic_audit/__init__.py",
        "src/gf/sim/a2_dynamic_audit/_baselines.py",
        "src/gf/sim/a2_dynamic_audit/_dynamic.py",
        "src/gf/sim/a2_dynamic_audit/_freeze.py",
        "src/gf/sim/a2_dynamic_audit/_heos_interpolation.py",
        "src/gf/sim/a2_dynamic_audit/_jacobian.py",
        "src/gf/sim/a2_dynamic_audit/_physics.py",
        "src/gf/sim/a2_dynamic_audit/_schema.py",
        "src/gf/sim/a2_dynamic_audit/_shared.py",
        "src/gf/sim/a2_dynamic_dataset.py",
        "src/gf/sim/a2_dynamic_physics.py",
        "src/gf/sim/a2_sensor_devices.py",
        "src/gf/sim/a2dyn_sound_speed.py",
        "src/gf/sim/ar_he_co2.py",
        "../shared/hitran_cache/CO2_2250p0000_2445p0000.data",
        "../shared/hitran_cache/CO2_2250p0000_2445p0000.header",
    ]
    return {
        relative_path: _sha256_file((root / relative_path).resolve())
        for relative_path in relative_paths
    }


def _resolve_project_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    if (root / "configs" / "data").is_dir():
        return root
    nested = root / "general_fusion"
    if (nested / "configs" / "data").is_dir():
        return nested
    raise ValueError(f"cannot locate general_fusion project root from {root}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a registered A2-DYN benchmark stage.")
    parser.add_argument("--stage", required=True, choices=PLANNED_STAGES)
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args(argv)
    result = run_a2_dynamic_benchmark(args.stage, project_root=args.project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] in {
        "PASS",
        "PILOT_QUALIFIED",
        "DEVELOPMENT_GENERATED",
        "DIFFICULTY_QUALIFIED",
        "DATA_FROZEN",
        "DEVELOPMENT_BASELINES_COMPLETE",
        "REPLAY_SMOKE_COMPLETE",
        "REPORT_GENERATED",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PLANNED_STAGES",
    "main",
    "run_a2_dynamic_benchmark",
    "run_a2_dynamic_baselines",
    "run_a2_dynamic_development_generation",
    "run_a2_dynamic_difficulty_audit",
    "run_a2_dynamic_handoff",
    "run_a2_dynamic_physics_smoke",
    "run_a2_dynamic_pilot",
    "run_a2_dynamic_replay_smoke",
    "run_a2_dynamic_report",
    "run_a2_dynamic_test_generation",
]
