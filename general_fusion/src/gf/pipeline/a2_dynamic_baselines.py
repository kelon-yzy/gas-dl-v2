"""A2-DYN-5 开发侧正式基线、指标和因果回放编排。

本模块只读取 train / val / stress_val。完整包当前为 ``DATA_FREEZE_FAILED``，
因此任何 test 读取都明确拒绝；开发证据不会伪装成 ``DYNAMIC_QUALIFIED``。
模型实现位于 :mod:`gf.dl.temporal_baselines`，这里负责冻结数据门、逐
horizon 训练、group-level 指标、预测物化和阶段产物。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.linear_model import Ridge

from gf.dl.evaluation import (
    evaluate_output_constraints,
    evaluate_predictions,
    group_bootstrap_comparison,
)
from gf.dl.temporal_baselines import (
    FORMAL_BASELINE_IDS,
    FORMAL_HORIZON_IDS,
    TEMPORAL_SEQUENCE_LENGTH,
    causal_sequence_matrix,
    fit_causal_neural_model,
    fit_formal_classical_model,
    formal_feature_matrix,
    select_ewma_alpha,
)
from gf.pipeline.a2_dynamic_protocol import (
    load_a2_dynamic_configs,
    run_a2_dynamic_protocol,
)
from gf.sim.a2_dynamic_audit import (
    _kinetic_oracle_predictions,
    _observed_admission_budgets,
)
from gf.sim.a2_dynamic_audit._shared import DEVELOPMENT_SPLITS, _horizon_indices
from gf.sim.a2_dynamic_dataset import (
    _calibration_profiles,
    load_a2_dynamic_dataset,
)


BASELINE_SCHEMA_VERSION = "gf-a2-dynamic-baselines-1"
REPLAY_SCHEMA_VERSION = "gf-a2-dynamic-replay-1"
REPORT_SCHEMA_VERSION = "gf-a2-dynamic-report-1"
HANDOFF_SCHEMA_VERSION = "gf-a2-dynamic-handoff-1"
SEED_ORDER = (17, 29, 43, 71, 101)
PRIMARY_SPLITS = ("train", "val", "stress_val")
PREDICTION_COLUMNS = (
    "run_id",
    "model_id",
    "seed",
    "observation_id",
    "mixture_id",
    "split",
    "family",
    "horizon_id",
    "cutoff_s",
    "horizon_valid",
    "prediction_valid",
    "y_true_Ar_pct",
    "y_true_He_pct",
    "y_true_CO2_pct",
    "y_pred_Ar_pct",
    "y_pred_He_pct",
    "y_pred_CO2_pct",
    "latency_ms",
)


def run_a2_dynamic_baselines(project_root: str | Path = ".") -> dict[str, Any]:
    """在开发 split 上执行 A2-DYN-5 正式基线矩阵。"""

    root, data_config, eval_config, experiment_config = load_a2_dynamic_configs(project_root)
    protocol = run_a2_dynamic_protocol(root, verify_reference_assets=True)
    if protocol["status"] != "PASS":
        raise ValueError(f"A2-DYN-0 prerequisite did not pass: {protocol['status']}")
    dataset_dir = root / str(data_config["storage"]["data_dir"])
    dataset = load_a2_dynamic_dataset(dataset_dir)
    allowed_splits = tuple(eval_config["development_splits"])
    _assert_development_only_access(dataset, allowed_splits)
    if tuple(allowed_splits) != PRIMARY_SPLITS:
        raise ValueError("A2-DYN-5 development split contract is not frozen")

    source_hashes = _baseline_source_hashes(root)
    config_hashes = {
        "data": _canonical_hash(data_config),
        "evaluation": _canonical_hash(eval_config),
        "experiment": _canonical_hash(experiment_config),
    }
    input_hash = _canonical_hash(
        {
            "data_content_sha256": dataset.manifest["content_sha256"],
            "config_hashes": config_hashes,
            "source_hashes": source_hashes,
            "seeds": list(SEED_ORDER),
            "sequence_length": TEMPORAL_SEQUENCE_LENGTH,
        }
    )
    summary_dir = root / "outputs" / "summary" / "a2_dynamic_v1"
    summary_path = summary_dir / "a2_dyn_5_baselines.json"
    prediction_path = root / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-5-baselines" / "predictions.csv"
    if summary_path.is_file() and prediction_path.is_file():
        existing = _read_json(summary_path)
        if existing.get("input_hash") == input_hash and existing.get("status") == "DEVELOPMENT_BASELINES_COMPLETE":
            return existing

    horizon_indices = _horizon_indices(dataset.time_s, dataset.records, data_config)
    target_ranges = np.asarray(
        [float(eval_config["target_ranges"][name]) for name in data_config["target_names"]],
        dtype=np.float64,
    )
    rows_by_split = {
        split: dataset.indices(split=split)
        for split in PRIMARY_SPLITS
    }
    valid_rows = {
        horizon: {
            split: rows_by_split[split][horizon_indices[horizon][rows_by_split[split]] >= 0]
            for split in PRIMARY_SPLITS
        }
        for horizon in FORMAL_HORIZON_IDS
    }
    feature_cache: dict[tuple[str, str, str], np.ndarray] = {}
    sequence_cache: dict[tuple[str, str], np.ndarray] = {}

    run_dir = prediction_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    prediction_count = 0
    prediction_store: dict[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    model_results: dict[str, Any] = {}
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()

        for model_id in ("B-LAST", "B-DELTA", "B-EWMA", "B-STAT", "B-TCN", "B-GRU", "B-STEADY"):
            model_results[model_id] = {
                "kind": "diagnostic" if model_id == "B-STEADY" else "deployable",
                "seed_records": [],
            }
            for seed in SEED_ORDER:
                seed_record = {"seed": int(seed), "horizons": {}}
                for horizon in FORMAL_HORIZON_IDS:
                    input_horizon = "P150" if model_id == "B-STEADY" else horizon
                    if model_id == "B-STEADY" and horizon not in {"P150", "FULL"}:
                        seed_record["horizons"][horizon] = {split: None for split in PRIMARY_SPLITS}
                        continue
                    train_rows = valid_rows[input_horizon]["train"]
                    eval_rows = {
                        split: valid_rows[input_horizon][split]
                        for split in PRIMARY_SPLITS
                    }
                    if model_id == "B-STEADY":
                        eval_rows = {
                            split: np.intersect1d(eval_rows[split], valid_rows[horizon][split])
                            for split in PRIMARY_SPLITS
                        }
                    if train_rows.size == 0:
                        raise ValueError(f"{model_id} {horizon} has no valid train rows")
                    if model_id in {"B-TCN", "B-GRU"}:
                        train_sequence = _cached_sequences(
                            sequence_cache,
                            dataset.signals,
                            input_horizon,
                            "train",
                            train_rows,
                            horizon_indices[input_horizon],
                        )
                        eval_sequences = [
                            _cached_sequences(
                                sequence_cache,
                                dataset.signals,
                                input_horizon,
                                split,
                                eval_rows[split],
                                horizon_indices[input_horizon],
                            )
                            for split in PRIMARY_SPLITS
                        ]
                        predictions, diagnostics, model = fit_causal_neural_model(
                            model_id,
                            train_sequence,
                            dataset.target[train_rows],
                            [train_sequence, *eval_sequences],
                            seed=int(seed),
                        )
                        train_prediction = predictions[0]
                        split_predictions = predictions[1:]
                        checkpoint_path = run_dir / "checkpoints" / f"{model_id}__{horizon}__seed{seed}.pt"
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "schema_version": "gf-a2-dynamic-baseline-checkpoint-1",
                                "model_id": model_id,
                                "horizon_id": horizon,
                                "seed": int(seed),
                                "sequence_length": int(train_sequence.shape[1]),
                                "state_dict": model.state_dict(),
                            },
                            checkpoint_path,
                        )
                        latency_fn = _temporal_latency_fn(model)
                        feature_inputs = eval_sequences
                    else:
                        alpha = float(experiment_config["pilot"]["ewma_alpha"])
                        alpha_selection = None
                        if model_id == "B-EWMA":
                            alpha, alpha_selection = select_ewma_alpha(
                                dataset.signals,
                                train_rows,
                                horizon_indices[input_horizon][train_rows],
                                dataset.target[train_rows],
                                seed=int(seed),
                            )
                        feature_model = "B-LAST" if model_id == "B-STEADY" else model_id
                        train_features = _cached_features(
                            feature_cache,
                            feature_model,
                            input_horizon,
                            "train",
                            dataset.signals,
                            train_rows,
                            horizon_indices[input_horizon],
                            ewma_alpha=alpha,
                        )
                        eval_features = [
                            _cached_features(
                                feature_cache,
                                feature_model,
                                input_horizon,
                                split,
                                dataset.signals,
                                eval_rows[split],
                                horizon_indices[input_horizon],
                                ewma_alpha=alpha,
                            )
                            for split in PRIMARY_SPLITS
                        ]
                        predictions, diagnostics, fitted = fit_formal_classical_model(
                            model_id if model_id != "B-STEADY" else "B-STEADY",
                            train_features,
                            dataset.target[train_rows],
                            [train_features, *eval_features],
                            seed=int(seed),
                        )
                        train_prediction = predictions[0]
                        split_predictions = predictions[1:]
                        diagnostics["ewma_selection"] = alpha_selection
                        latency_fn = _classical_latency_fn(fitted)
                        feature_inputs = eval_features
                    horizon_metrics: dict[str, Any] = {}
                    train_metric = _metric_bundle(
                        dataset,
                        train_rows,
                        train_prediction,
                        target_ranges,
                        latency_ms=_measure_latency(latency_fn, train_features if model_id not in {"B-TCN", "B-GRU"} else train_sequence),
                    )
                    horizon_metrics["train"] = train_metric
                    for split, split_rows, split_prediction, split_input in zip(
                        PRIMARY_SPLITS, (eval_rows[split] for split in PRIMARY_SPLITS), split_predictions, feature_inputs
                    ):
                        latency = _measure_latency(latency_fn, split_input)
                        metric = _metric_bundle(
                            dataset,
                            split_rows,
                            split_prediction,
                            target_ranges,
                            latency_ms=latency,
                        )
                        horizon_metrics[split] = metric
                        prediction_store[(model_id, int(seed), horizon, split)] = (
                            np.asarray(split_rows, dtype=np.int64),
                            np.asarray(split_prediction, dtype=np.float64),
                        )
                        prediction_count += _write_predictions(
                            writer,
                            dataset,
                            run_id="a2-dyn-5-baselines",
                            model_id=model_id,
                            seed=seed,
                            horizon=horizon,
                            rows=split_rows,
                            predictions=split_prediction,
                            time_s=dataset.time_s,
                            horizon_indices=horizon_indices[horizon],
                            latency_ms=latency,
                        )
                    seed_record["horizons"][horizon] = horizon_metrics
                    seed_record["horizons"][horizon]["fit"] = diagnostics
                model_results[model_id]["seed_records"].append(seed_record)

        oracle_results, oracle_count = _run_oracles(
            dataset,
            root=root,
            data_config=data_config,
            eval_config=eval_config,
            experiment_config=experiment_config,
            horizon_indices=horizon_indices,
            target_ranges=target_ranges,
            rows_by_split=rows_by_split,
            valid_rows=valid_rows,
            writer=writer,
            prediction_store=prediction_store,
        )
        prediction_count += oracle_count
        model_results.update(oracle_results)

    b_ref = _select_b_ref(model_results, prediction_store, dataset, eval_config)
    _attach_early_gain(model_results, prediction_store, b_ref, dataset)
    dynamic_metrics = _attach_dynamic_metrics(model_results, prediction_store, dataset, eval_config)
    temporal_gate = _evaluate_temporal_gate(
        prediction_store,
        b_ref,
        dataset,
        eval_config,
    )
    headroom = _evaluate_headroom(prediction_store, dataset, eval_config)
    formal_status = (
        "FORMAL_BLOCKED_DATA_FREEZE_FAILED"
        if dataset.manifest.get("status") != "DATA_FROZEN"
        else "FORMAL_TEST_NOT_RUN"
    )
    summary = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "stage": "A2-DYN-5",
        "status": "DEVELOPMENT_BASELINES_COMPLETE",
        "formal_status": formal_status,
        "data_manifest_status": dataset.manifest.get("status"),
        "data_content_sha256": dataset.manifest["content_sha256"],
        "development_only": True,
        "test_access": {
            "allowed_read_splits": list(PRIMARY_SPLITS),
            "test_rows_read": 0,
            "unlock_required": "DATA_FROZEN",
            "blocked_reason": "A2-DYN-4R2 freeze audit is DATA_FREEZE_FAILED",
        },
        "config_hashes": config_hashes,
        "source_hashes": source_hashes,
        "input_hash": input_hash,
        "model_ids": list(model_results),
        "formal_training_seeds": list(SEED_ORDER),
        "prediction_path": _relative(root, prediction_path),
        "prediction_scope": list(PRIMARY_SPLITS),
        "train_predictions_materialized": True,
        "prediction_count": int(prediction_count),
        "models": model_results,
        "b_ref_selection": b_ref,
        "dynamic_metrics": dynamic_metrics,
        "temporal_information_gate": temporal_gate,
        "new_algorithm_headroom": headroom,
        "terminal_status": "FORMAL_BLOCKED_DATA_FREEZE_FAILED",
        "new_algorithm_handoff_allowed": False,
    }
    _write_json(summary_path, summary)
    _write_json(
        run_dir / "run_manifest.json",
        {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "run_id": "a2-dyn-5-baselines",
            "stage": "A2-DYN-5",
            "status": summary["status"],
            "input_hash": input_hash,
            "data_content_sha256": dataset.manifest["content_sha256"],
            "allowed_read_splits": list(PRIMARY_SPLITS),
            "test_rows_read": 0,
            "prediction_count": int(prediction_count),
            "artifact_paths": {
                "predictions": _relative(root, prediction_path),
                "summary": _relative(root, summary_path),
            },
        },
    )
    return summary


def run_a2_dynamic_replay_smoke(project_root: str | Path = ".") -> dict[str, Any]:
    """执行小规模 virtual-clock 与 wall-clock 因果回放。"""

    root, data_config, eval_config, experiment_config = load_a2_dynamic_configs(project_root)
    del eval_config
    dataset = load_a2_dynamic_dataset(root / str(data_config["storage"]["data_dir"]))
    _assert_development_only_access(dataset, PRIMARY_SPLITS)
    baseline_summary = _read_json(root / "outputs" / "summary" / "a2_dynamic_v1" / "a2_dyn_5_baselines.json")
    if baseline_summary.get("status") != "DEVELOPMENT_BASELINES_COMPLETE":
        raise ValueError("replay smoke requires a completed A2-DYN-5 development baseline")
    h_indices = _horizon_indices(dataset.time_s, dataset.records, data_config)
    train_rows = dataset.indices(split="train")
    p015_rows = train_rows[h_indices["P015"][train_rows] >= 0]
    if p015_rows.size == 0:
        raise ValueError("replay smoke requires valid P015 train rows")
    endpoints = h_indices["P015"][p015_rows]
    train_features = formal_feature_matrix("B-LAST", dataset.signals, p015_rows, endpoints)
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    model = Ridge(alpha=1.0e-6).fit((train_features - mean) / scale, dataset.target[p015_rows] / 100.0)
    replay_rows = dataset.indices(split="val")[: min(12, dataset.indices(split="val").size)]
    virtual_updates = 0
    virtual_violations: list[str] = []
    wall_times: list[float] = []
    for row in replay_rows:
        state = None
        last_timestamp = -math.inf
        for timestep, timestamp in enumerate(dataset.time_s):
            if timestamp <= last_timestamp:
                virtual_violations.append("timestamps_not_increasing")
            last_timestamp = float(timestamp)
            prefix = dataset.signals[int(row), :, : timestep + 1, 0].T
            current = prefix[-1:]
            feature = current
            started = time.perf_counter()
            prediction = model.predict((feature - mean) / scale)[0] * 100.0
            wall_times.append((time.perf_counter() - started) * 1000.0)
            if state is not None and not np.isfinite(prediction).all():
                virtual_violations.append("non_finite_prediction")
            state = prediction
            virtual_updates += 1
        if state is None:
            virtual_violations.append("state_not_reset")
    p95 = float(np.percentile(np.asarray(wall_times, dtype=np.float64), 95)) if wall_times else None
    result = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "stage": "A2-DYN-5",
        "operation": "replay-smoke",
        "status": "REPLAY_SMOKE_COMPLETE" if not virtual_violations else "REPLAY_SMOKE_FAILED",
        "clock": "virtual_clock_full_and_wall_clock_smoke",
        "sample_count": int(replay_rows.size),
        "virtual_updates": int(virtual_updates),
        "future_padding": False,
        "state_reset_between_observations": True,
        "causal_prefix_only": True,
        "virtual_clock_violations": virtual_violations,
        "wall_clock_latency_p95_ms": p95,
        "minimum_update_period_s": float(experiment_config["pilot"]["selected_sample_rate_hz"] ** -1),
        "wall_clock_within_update_period": p95 is not None and p95 < float(experiment_config["pilot"]["selected_sample_rate_hz"] ** -1) * 1000.0,
        "baseline_input_hash": baseline_summary["input_hash"],
    }
    summary_path = root / "outputs" / "summary" / "a2_dynamic_v1" / "a2_dyn_5_replay_smoke.json"
    run_dir = root / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-5-replay-smoke"
    _write_json(summary_path, result)
    _write_json(run_dir / "run_manifest.json", result)
    return result


def run_a2_dynamic_report(project_root: str | Path = ".") -> dict[str, Any]:
    """将开发基线、回放和冻结门状态汇总为可审查报告。"""

    root, _, _, _ = load_a2_dynamic_configs(project_root)
    baseline_path = root / "outputs" / "summary" / "a2_dynamic_v1" / "a2_dyn_5_baselines.json"
    replay_path = root / "outputs" / "summary" / "a2_dynamic_v1" / "a2_dyn_5_replay_smoke.json"
    if not baseline_path.is_file() or not replay_path.is_file():
        raise ValueError("report requires completed baselines and replay-smoke artifacts")
    baseline = _read_json(baseline_path)
    replay = _read_json(replay_path)
    report_path = root / "outputs" / "reports" / "a2_dynamic_v1" / "a2_dyn_5_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    gate = baseline["temporal_information_gate"]
    lines = [
        "# A2-DYN-5 开发侧基线与因果回放报告",
        "",
        f"- 数据内容 hash：`{baseline['data_content_sha256']}`",
        f"- 数据冻结状态：`{baseline['data_manifest_status']}`",
        f"- 开发基线：`{baseline['status']}`，预测行数 `{baseline['prediction_count']}`",
        f"- 正式 test：阻塞（`{baseline['formal_status']}`），读取行数 `{baseline['test_access']['test_rows_read']}`",
        f"- 开发侧时间门候选结论：`{gate['development_gate_status']}`",
        f"- 因果回放：`{replay['status']}`，wall-clock p95 `{replay['wall_clock_latency_p95_ms']}` ms",
        "",
        "## 门控边界",
        "",
        "本报告只使用 train / val / stress_val。由于 A2-DYN-4R2 冻结审计为 DATA_FREEZE_FAILED，开发侧数值不升级为正式 DYNAMIC_QUALIFIED，也不生成新算法 handoff。",
        "",
        "## B-REF",
        "",
        "```json",
        json.dumps(baseline["b_ref_selection"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 回放证据",
        "",
        "```json",
        json.dumps(replay, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "stage": "A2-DYN-5",
        "operation": "report",
        "status": "REPORT_GENERATED",
        "baseline_summary": _relative(root, baseline_path),
        "replay_summary": _relative(root, replay_path),
        "report_path": _relative(root, report_path),
        "formal_status": baseline["formal_status"],
        "new_algorithm_handoff_allowed": False,
    }
    _write_json(root / "outputs" / "summary" / "a2_dynamic_v1" / "a2_dyn_5_report.json", result)
    _write_json(root / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-5-report" / "run_manifest.json", result)
    return result


def run_a2_dynamic_handoff(project_root: str | Path = ".") -> dict[str, Any]:
    """执行 A2-DYN-6：仅在正式门通过时生成 handoff，否则写出阻断事实。"""

    root, _, _, _ = load_a2_dynamic_configs(project_root)
    protocol = run_a2_dynamic_protocol(root, verify_reference_assets=True)
    if protocol["status"] != "PASS":
        raise ValueError(f"A2-DYN-0 prerequisite did not pass: {protocol['status']}")

    summary_dir = root / "outputs" / "summary" / "a2_dynamic_v1"
    artifact_paths = {
        "baseline": summary_dir / "a2_dyn_5_baselines.json",
        "replay": summary_dir / "a2_dyn_5_replay_smoke.json",
        "report": summary_dir / "a2_dyn_5_report.json",
        "difficulty_audit": summary_dir / "a2_dyn_3r2_audit.json",
        "freeze_audit": summary_dir / "a2_dyn_4r2_freeze_audit.json",
    }
    missing = [name for name, path in artifact_paths.items() if not path.is_file()]
    if missing:
        raise ValueError(
            "A2-DYN-6 requires completed A2-DYN-5 and audit artifacts; "
            f"missing: {', '.join(missing)}"
        )

    baseline = _read_json(artifact_paths["baseline"])
    replay = _read_json(artifact_paths["replay"])
    report = _read_json(artifact_paths["report"])
    difficulty = _read_json(artifact_paths["difficulty_audit"])
    freeze = _read_json(artifact_paths["freeze_audit"])
    gate = baseline.get("temporal_information_gate", {})
    data_status = baseline.get("data_manifest_status")
    freeze_status = freeze.get("status")
    formal_status = baseline.get("formal_status")

    if data_status != "DATA_FROZEN" or freeze_status != "DATA_FROZEN":
        status = (
            "A2_DYN_6_BLOCKED_DATA_FREEZE_FAILED"
            if "DATA_FREEZE_FAILED" in {data_status, freeze_status}
            else "A2_DYN_6_BLOCKED_DATA_NOT_FROZEN"
        )
        closure_decision = "WAIT_FOR_DATA_V2_OR_SUCCESSFUL_DATA_FREEZE"
        reason = "A2-DYN-4R2 freeze audit is not DATA_FROZEN; formal test and handoff remain locked"
    elif formal_status != "FORMAL_TEST_COMPLETE":
        status = "A2_DYN_6_BLOCKED_FORMAL_BASELINE_INCOMPLETE"
        closure_decision = "RUN_FORMAL_A2_DYN_5_BEFORE_HANDOFF"
        reason = "DATA_FROZEN is present but A2-DYN-5 formal test evidence is incomplete"
    elif gate.get("formal_gate_status") == "DYNAMIC_QUALIFIED" and baseline.get(
        "new_algorithm_handoff_allowed"
    ) is True:
        status = "A2_DYN_6_HANDOFF_READY"
        closure_decision = "GENERATE_DYNAMIC_HANDOFF"
        reason = "formal temporal information gate and handoff flag both passed"
    else:
        status = "A2_DYN_6_CLOSED_FORMAL_NEGATIVE"
        closure_decision = "CLOSE_WITHOUT_NEW_ALGORITHM"
        reason = "formal evidence does not satisfy the registered dynamic qualification gate"

    result = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "stage": "A2-DYN-6",
        "operation": "handoff",
        "status": status,
        "handoff_generated": status == "A2_DYN_6_HANDOFF_READY",
        "new_algorithm_handoff_allowed": status == "A2_DYN_6_HANDOFF_READY",
        "algorithm_search_allowed": False,
        "closure_decision": closure_decision,
        "reason": reason,
        "data_manifest_status": data_status,
        "freeze_audit_status": freeze_status,
        "formal_status": formal_status,
        "development_gate_status": gate.get("development_gate_status"),
        "formal_gate_status": gate.get("formal_gate_status"),
        "data_content_sha256": baseline.get("data_content_sha256"),
        "baseline_input_hash": baseline.get("input_hash"),
        "audit_artifacts": {
            name: _relative(root, path) for name, path in artifact_paths.items()
        },
        "upstream_statuses": {
            "baseline": baseline.get("status"),
            "replay": replay.get("status"),
            "report": report.get("status"),
            "difficulty": difficulty.get("status"),
            "freeze": freeze.get("status"),
        },
    }
    output_path = summary_dir / "a2_dyn_6_closure.json"
    result["artifact_paths"] = {"closure": _relative(root, output_path)}
    _write_json(output_path, result)
    run_dir = root / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-6-handoff"
    _write_json(run_dir / "run_manifest.json", result)
    if result["handoff_generated"]:
        _write_json(summary_dir / "dynamic_handoff.json", result)
    return result


def assert_a2_dynamic_test_unlocked(manifest: Mapping[str, Any]) -> None:
    """共享 test 硬门：只有完整包 ``DATA_FROZEN`` 才能读取 test。"""

    if manifest.get("status") != "DATA_FROZEN":
        raise ValueError(
            "A2-DYN test access is locked until manifest.status == 'DATA_FROZEN'; "
            f"got {manifest.get('status')!r}"
        )


def _run_oracles(
    dataset: Any,
    *,
    root: Path,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    horizon_indices: Mapping[str, np.ndarray],
    target_ranges: np.ndarray,
    rows_by_split: Mapping[str, np.ndarray],
    valid_rows: Mapping[str, Mapping[str, np.ndarray]],
    writer: csv.DictWriter,
    prediction_store: dict[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], int]:
    results: dict[str, Any] = {}
    written = 0
    equilibrium_signals = np.asarray(dataset.equilibrium_reference_signals, dtype=np.float32)[:, :, :, None]
    for model_id in ("O-EQ",):
        results[model_id] = {"kind": "oracle", "seed_records": []}
        for seed in SEED_ORDER:
            record = {"seed": int(seed), "deterministic_oracle": False, "horizons": {}}
            for horizon in FORMAL_HORIZON_IDS:
                train_rows = valid_rows[horizon]["train"]
                eval_rows = [valid_rows[horizon][split] for split in PRIMARY_SPLITS]
                train_features = formal_feature_matrix(
                    "O-EQ", equilibrium_signals, train_rows, horizon_indices[horizon][train_rows]
                )
                eval_features = [
                    formal_feature_matrix(
                        "O-EQ", equilibrium_signals, rows, horizon_indices[horizon][rows]
                    )
                    for rows in eval_rows
                ]
                predictions, diagnostics, _ = fit_formal_classical_model(
                    "O-EQ", train_features, dataset.target[train_rows], [train_features, *eval_features], seed=int(seed)
                )
                split_predictions = predictions[1:]
                horizon_metrics = {
                    "train": _metric_bundle(dataset, train_rows, predictions[0], target_ranges, latency_ms=None)
                }
                for split, rows, prediction in zip(PRIMARY_SPLITS, eval_rows, split_predictions):
                    metric = _metric_bundle(dataset, rows, prediction, target_ranges, latency_ms=None)
                    horizon_metrics[split] = metric
                    prediction_store[(model_id, int(seed), horizon, split)] = (rows, prediction)
                    written += _write_predictions(
                        writer, dataset, run_id="a2-dyn-5-baselines", model_id=model_id, seed=seed,
                        horizon=horizon, rows=rows, predictions=prediction, time_s=dataset.time_s,
                        horizon_indices=horizon_indices[horizon], latency_ms=None,
                    )
                horizon_metrics["fit"] = diagnostics
                record["horizons"][horizon] = horizon_metrics
            results[model_id]["seed_records"].append(record)

    calibrations = _calibration_profiles(
        data_config,
        _read_json(root / str(data_config["source_registry"]["a2h"]["config_path"])),
    )
    noise_base = np.asarray(experiment_config["pilot"]["observation_noise_std_by_sensor"], dtype=np.float64)
    admission_budgets = _observed_admission_budgets(
        dataset,
        np.concatenate(tuple(rows_by_split.values())),
        data_config=data_config,
        calibrations=calibrations,
        noise_base=noise_base,
    )
    kinetic_cache: dict[int, dict[str, np.ndarray]] = {}
    heos_interpolation_cache: dict[tuple[float, float], np.ndarray] = {}
    for model_id, mode in (("O-KIN", "clean"), ("O-KIN-OBS", "observed")):
        results[model_id] = {
            "kind": "oracle",
            "seed_semantics": "deterministic_repeat_for_seed_alignment",
            "seed_records": [],
        }
        base_record: dict[str, Any] = {"seed": 0, "deterministic_oracle": True, "horizons": {}}
        for horizon in FORMAL_HORIZON_IDS:
            horizon_metrics: dict[str, Any] = {}
            for split in PRIMARY_SPLITS:
                rows = valid_rows[horizon][split]
                failures: list[dict[str, Any]] = []
                prediction = _kinetic_oracle_predictions(
                    dataset,
                    rows,
                    horizon_indices[horizon][rows],
                    data_config=data_config,
                    kinetic_cache=kinetic_cache,
                    heos_interpolation_cache=heos_interpolation_cache,
                    input_mode=mode,
                    admission_budgets=admission_budgets if mode == "observed" else None,
                    inversion_failures=failures if mode == "observed" else None,
                )
                metric = _metric_bundle(dataset, rows, prediction, target_ranges, latency_ms=None)
                if mode == "observed":
                    metric["inversion_failure_count"] = int(len(failures))
                    metric["inversion_failure_fraction"] = float(len(failures) / max(len(rows), 1))
                horizon_metrics[split] = metric
                prediction_store[(model_id, 0, horizon, split)] = (rows, prediction)
                written += _write_predictions(
                    writer, dataset, run_id="a2-dyn-5-baselines", model_id=model_id, seed=0,
                    horizon=horizon, rows=rows, predictions=prediction, time_s=dataset.time_s,
                    horizon_indices=horizon_indices[horizon], latency_ms=None,
                )
            base_record["horizons"][horizon] = horizon_metrics
        results[model_id]["seed_records"].append(base_record)
        results[model_id]["seed_count"] = 1
    return results, written


def _assert_development_only_access(dataset: Any, allowed_splits: Sequence[str]) -> None:
    actual = {str(record["split"]) for record in dataset.records}
    if not actual.intersection(set(allowed_splits)):
        raise ValueError("dataset has no registered development rows")
    forbidden = actual - set(allowed_splits)
    if forbidden and forbidden != {"test"}:
        raise ValueError(f"dataset contains unknown split labels: {sorted(forbidden)}")


def _cached_features(
    cache: dict[tuple[str, str, str], np.ndarray],
    model_id: str,
    horizon: str,
    split: str,
    signals: np.ndarray,
    rows: np.ndarray,
    horizon_index: np.ndarray,
    *,
    ewma_alpha: float,
) -> np.ndarray:
    key = (model_id, horizon, split, str(float(ewma_alpha)))
    if key not in cache:
        cache[key] = formal_feature_matrix(
            model_id,
            signals,
            rows,
            horizon_index[rows],
            ewma_alpha=ewma_alpha,
        )
    return cache[key]


def _cached_sequences(
    cache: dict[tuple[str, str], np.ndarray],
    signals: np.ndarray,
    horizon: str,
    split: str,
    rows: np.ndarray,
    horizon_index: np.ndarray,
) -> np.ndarray:
    key = (horizon, split)
    if key not in cache:
        cache[key] = causal_sequence_matrix(
            signals,
            rows,
            horizon_index[rows],
            sequence_length=TEMPORAL_SEQUENCE_LENGTH,
        )
    return cache[key]


def _metric_bundle(
    dataset: Any,
    rows: np.ndarray,
    predictions: np.ndarray,
    target_ranges: np.ndarray,
    *,
    latency_ms: float | None,
) -> dict[str, Any] | None:
    row_values = np.asarray(rows, dtype=np.int64)
    prediction_values = np.asarray(predictions, dtype=np.float64)
    if row_values.size == 0:
        return None
    if prediction_values.shape != (row_values.size, 3):
        raise ValueError("prediction rows do not align with dataset rows")
    finite_mask = np.isfinite(prediction_values).all(axis=1)
    valid_rows = row_values[finite_mask]
    valid_predictions = prediction_values[finite_mask]
    if valid_rows.size == 0:
        return {
            "status": "NO_VALID_PREDICTIONS",
            "row_count": int(row_values.size),
            "prediction_invalid_count": int((~finite_mask).sum()),
            "LatencyP95": latency_ms,
        }
    result = evaluate_predictions(
        dataset.target[valid_rows],
        valid_predictions,
        np.asarray(dataset.group_ids, dtype=object)[valid_rows],
        np.arange(valid_rows.size, dtype=np.int64),
        target_ranges=target_ranges,
    )
    constraints = evaluate_output_constraints(
        valid_predictions,
        targets=dataset.target[valid_rows],
    )
    result.update(constraints)
    result["worst_slice_MAE"] = result["worst_group_MAE"]
    result["row_count"] = int(row_values.size)
    result["prediction_valid_count"] = int(valid_rows.size)
    result["prediction_invalid_count"] = int((~finite_mask).sum())
    result["LatencyP95"] = latency_ms
    result["family"] = {}
    for family in sorted({str(dataset.records[int(row)]["family"]) for row in valid_rows}):
        family_rows = np.asarray(
            [int(row) for row in valid_rows if str(dataset.records[int(row)]["family"]) == family],
            dtype=np.int64,
        )
        positions = np.asarray([int(np.flatnonzero(valid_rows == row)[0]) for row in family_rows], dtype=np.int64)
        family_metric = evaluate_predictions(
            dataset.target[family_rows],
            valid_predictions[positions],
            np.asarray(dataset.group_ids, dtype=object)[family_rows],
            np.arange(family_rows.size, dtype=np.int64),
            target_ranges=target_ranges,
        )
        family_metric["row_count"] = int(family_rows.size)
        result["family"][family] = family_metric
    return result


def _write_predictions(
    writer: csv.DictWriter,
    dataset: Any,
    *,
    run_id: str,
    model_id: str,
    seed: int,
    horizon: str,
    rows: np.ndarray,
    predictions: np.ndarray,
    time_s: np.ndarray,
    horizon_indices: np.ndarray,
    latency_ms: float | None,
) -> int:
    row_values = np.asarray(rows, dtype=np.int64)
    prediction_values = np.asarray(predictions, dtype=np.float64)
    if row_values.size == 0:
        return 0
    if prediction_values.shape != (row_values.size, 3):
        raise ValueError("predictions must align with rows for CSV materialization")
    for row, prediction in zip(row_values, prediction_values):
        record = dataset.records[int(row)]
        endpoint = int(horizon_indices[int(row)])
        writer.writerow(
            {
                "run_id": run_id,
                "model_id": model_id,
                "seed": int(seed),
                "observation_id": record["observation_id"],
                "mixture_id": record["mixture_id"],
                "split": record["split"],
                "family": record["family"],
                "horizon_id": horizon,
                "cutoff_s": "" if endpoint < 0 else float(time_s[endpoint]),
                "horizon_valid": endpoint >= 0,
                "prediction_valid": bool(np.isfinite(prediction).all()),
                "y_true_Ar_pct": float(record["x_Ar_pct"]),
                "y_true_He_pct": float(record["x_He_pct"]),
                "y_true_CO2_pct": float(record["x_CO2_pct"]),
                "y_pred_Ar_pct": float(prediction[0]) if np.isfinite(prediction[0]) else "nan",
                "y_pred_He_pct": float(prediction[1]) if np.isfinite(prediction[1]) else "nan",
                "y_pred_CO2_pct": float(prediction[2]) if np.isfinite(prediction[2]) else "nan",
                "latency_ms": "" if latency_ms is None else float(latency_ms),
            }
        )
    return int(row_values.size)


def _select_b_ref(
    model_results: Mapping[str, Any],
    prediction_store: Mapping[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    dataset: Any,
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    del model_results
    selected: dict[str, Any] = {}
    for horizon in FORMAL_HORIZON_IDS:
        candidates: dict[str, float] = {}
        for model_id in ("B-LAST", "B-DELTA", "B-EWMA"):
            values = []
            rows = dataset.indices(family="D-IID", split="val")
            rows = rows[_horizon_indices_from_store(prediction_store, model_id, 17, horizon, rows)]
            for seed in SEED_ORDER:
                stored = prediction_store.get((model_id, int(seed), horizon, "val"))
                if stored is None:
                    continue
                stored_rows, prediction = stored
                mask = np.isin(stored_rows, rows)
                if np.any(mask):
                    metric = evaluate_predictions(
                        dataset.target[stored_rows[mask]],
                        prediction[mask],
                        np.asarray(dataset.group_ids, dtype=object)[stored_rows[mask]],
                        np.arange(int(mask.sum()), dtype=np.int64),
                        target_ranges=np.full(3, 100.0),
                    )
                    values.append(float(metric["macro_RNMAE"]))
            if values:
                candidates[model_id] = float(np.mean(values))
        selected[horizon] = {
            "candidates_val_D-IID_mean_macro_RNMAE": candidates,
            "selected_model": min(candidates, key=candidates.get) if candidates else None,
        }
    return selected


def _horizon_indices_from_store(
    prediction_store: Mapping[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    model_id: str,
    seed: int,
    horizon: str,
    rows: np.ndarray,
) -> np.ndarray:
    stored = prediction_store.get((model_id, int(seed), horizon, "val"))
    if stored is None:
        return np.zeros(rows.shape, dtype=bool)
    return np.isin(rows, stored[0])


def _attach_early_gain(
    model_results: dict[str, Any],
    prediction_store: Mapping[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    b_ref: Mapping[str, Any],
    dataset: Any,
) -> None:
    for model_id, result in model_results.items():
        for seed_record in result.get("seed_records", []):
            seed = int(seed_record["seed"])
            for horizon, horizon_metrics in seed_record.get("horizons", {}).items():
                reference_id = b_ref.get(horizon, {}).get("selected_model")
                if reference_id is None:
                    continue
                for split in ("val", "stress_val"):
                    metric = horizon_metrics.get(split)
                    reference = prediction_store.get((reference_id, seed, horizon, split))
                    candidate = prediction_store.get((model_id, seed, horizon, split))
                    if metric is None or reference is None or candidate is None:
                        continue
                    common = np.intersect1d(reference[0], candidate[0])
                    if common.size == 0:
                        continue
                    ref_map = {int(row): prediction for row, prediction in zip(*reference)}
                    cand_map = {int(row): prediction for row, prediction in zip(*candidate)}
                    ref_prediction = np.asarray([ref_map[int(row)] for row in common], dtype=np.float64)
                    cand_prediction = np.asarray([cand_map[int(row)] for row in common], dtype=np.float64)
                    target = dataset.target[common]
                    ref_error = float(np.mean(np.abs(target - ref_prediction)))
                    cand_error = float(np.mean(np.abs(target - cand_prediction)))
                    metric["EarlyGain"] = None if ref_error <= 0.0 else float((ref_error - cand_error) / ref_error)


def _attach_dynamic_metrics(
    model_results: Mapping[str, Any],
    prediction_store: Mapping[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    dataset: Any,
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    del eval_config
    result: dict[str, Any] = {}
    horizon_seconds = {"P005": 5.0, "P015": 15.0, "P030": 30.0, "P060": 60.0, "P120": 120.0, "P150": 150.0, "FULL": 240.0}
    for model_id, model_result in model_results.items():
        result[model_id] = {}
        for seed_record in model_result.get("seed_records", []):
            seed = int(seed_record["seed"])
            result[model_id][str(seed)] = {}
            for split in ("val", "stress_val"):
                horizon_values: dict[str, float | None] = {}
                for horizon in ("P015", "P030", "P060", "P120"):
                    stored = prediction_store.get((model_id, seed, horizon, split))
                    if stored is None:
                        horizon_values[horizon] = None
                        continue
                    rows, prediction = stored
                    valid = np.isfinite(prediction).all(axis=1)
                    horizon_values[horizon] = (
                        float(np.mean(np.abs(dataset.target[rows[valid]] - prediction[valid])) / 100.0)
                        if np.any(valid)
                        else None
                    )
                available = [value for value in horizon_values.values() if value is not None]
                auec = None
                if len(available) == 4:
                    times = np.asarray([15.0, 30.0, 60.0, 120.0], dtype=np.float64)
                    auec = float(np.trapezoid(np.asarray([horizon_values[h] for h in ("P015", "P030", "P060", "P120")]), times) / (times[-1] - times[0]))
                result[model_id][str(seed)][split] = {
                    "Error": horizon_values,
                    "AUEC": auec,
                    "TTA@5mol%": _tta_metric(model_id, seed, split, prediction_store, dataset, horizon_seconds, 5.0),
                    "TTA@2mol%": _tta_metric(model_id, seed, split, prediction_store, dataset, horizon_seconds, 2.0),
                }
    return result


def _tta_metric(
    model_id: str,
    seed: int,
    split: str,
    prediction_store: Mapping[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    dataset: Any,
    horizon_seconds: Mapping[str, float],
    threshold: float,
) -> dict[str, Any]:
    horizon_order = ("P005", "P015", "P030", "P060", "P120", "P150")
    maps = {}
    rows_union: set[int] = set()
    for horizon in horizon_order:
        stored = prediction_store.get((model_id, seed, horizon, split))
        if stored is None:
            continue
        rows, prediction = stored
        maps[horizon] = {int(row): prediction[index] for index, row in enumerate(rows)}
        rows_union.update(int(row) for row in rows)
    reached: list[float] = []
    censored = 0
    for row in sorted(rows_union):
        reached_at = None
        for horizon in horizon_order:
            prediction = maps.get(horizon, {}).get(row)
            if prediction is None or not np.isfinite(prediction).all():
                continue
            if float(np.mean(np.abs(dataset.target[row] - prediction))) <= threshold:
                reached_at = horizon_seconds[horizon]
                break
        if reached_at is None:
            censored += 1
        else:
            reached.append(float(reached_at))
    return {
        "threshold_mol_pct": float(threshold),
        "reached_count": int(len(reached)),
        "censored_count": int(censored),
        "reached_fraction": float(len(reached) / max(len(reached) + censored, 1)),
        "mean_reached_s": float(np.mean(reached)) if reached else None,
        "censoring": "right",
    }


def _evaluate_temporal_gate(
    prediction_store: Mapping[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    b_ref: Mapping[str, Any],
    dataset: Any,
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    gate = eval_config["qualification_gates"]["temporal_information"]
    families = ["D-IID"] + sorted(
        {
            str(record["family"])
            for record in dataset.records
            if str(record["split"]) == "val" and str(record["family"]) != "D-IID"
        }
    )
    family_results: dict[str, Any] = {}
    for candidate_id in ("B-STAT", "B-TCN"):
        family_results[candidate_id] = {}
        for family in families:
            family_results[candidate_id][family] = {}
            for horizon in gate["required_horizons"]:
                family_results[candidate_id][family][horizon] = _compare_gate_cell(
                    candidate_id,
                    family,
                    horizon,
                    b_ref,
                    prediction_store,
                    dataset,
                    eval_config,
                    gate,
                )
    horizon_results = {
        candidate: dict(family_results[candidate].get("D-IID", {}))
        for candidate in family_results
    }
    required = tuple(gate["required_horizons"])
    iid_pass = {
        candidate: sum(family_results[candidate]["D-IID"].get(h, {}).get("status") == "PASS" for h in required)
        >= int(gate["min_horizons_passing"])
        for candidate in family_results
    }
    pressure_pass = {
        candidate: [
            family
            for family in families
            if family != "D-IID"
            and sum(family_results[candidate][family].get(h, {}).get("status") == "PASS" for h in required)
            >= int(gate["min_horizons_passing"])
        ]
        for candidate in family_results
    }
    candidate_pass = {
        candidate: bool(iid_pass[candidate] and pressure_pass[candidate])
        for candidate in family_results
    }
    return {
        "development_gate_status": "DYNAMIC_QUALIFIED" if any(candidate_pass.values()) else "TEMPORAL_REDUNDANT",
        "formal_gate_status": "BLOCKED_DATA_FREEZE_FAILED",
        "candidate_horizon_results": horizon_results,
        "family_results": family_results,
        "iid_pass": iid_pass,
        "pressure_pass_families": pressure_pass,
        "candidate_pass": candidate_pass,
        "test_rows_used": 0,
        "selection_frozen_before_test": True,
    }


def _compare_gate_cell(
    candidate_id: str,
    family: str,
    horizon: str,
    b_ref: Mapping[str, Any],
    prediction_store: Mapping[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    dataset: Any,
    eval_config: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    ref_id = b_ref.get(horizon, {}).get("selected_model")
    if ref_id is None:
        return {"status": "NO_REFERENCE", "family": family, "horizon": horizon}
    seed_values: list[float] = []
    direction_count = 0
    component_max_degradation = 0.0
    bootstrap = None
    for seed in SEED_ORDER:
        ref = prediction_store.get((ref_id, int(seed), horizon, "val"))
        cand = prediction_store.get((candidate_id, int(seed), horizon, "val"))
        if ref is None or cand is None:
            continue
        common = np.intersect1d(ref[0], cand[0])
        common = np.asarray(
            [row for row in common if str(dataset.records[int(row)]["family"]) == family],
            dtype=np.int64,
        )
        if common.size == 0:
            continue
        ref_map = {int(row): prediction for row, prediction in zip(*ref)}
        cand_map = {int(row): prediction for row, prediction in zip(*cand)}
        ref_prediction = np.asarray([ref_map[int(row)] for row in common], dtype=np.float64)
        cand_prediction = np.asarray([cand_map[int(row)] for row in common], dtype=np.float64)
        targets = dataset.target[common]
        ref_metric = evaluate_predictions(
            targets,
            ref_prediction,
            np.asarray(dataset.group_ids, dtype=object)[common],
            np.arange(common.size),
            target_ranges=np.full(3, 100.0),
        )
        cand_metric = evaluate_predictions(
            targets,
            cand_prediction,
            np.asarray(dataset.group_ids, dtype=object)[common],
            np.arange(common.size),
            target_ranges=np.full(3, 100.0),
        )
        reference = float(ref_metric["macro_RNMAE"])
        candidate = float(cand_metric["macro_RNMAE"])
        improvement = (reference - candidate) / reference if reference > 0.0 else None
        if improvement is None:
            continue
        seed_values.append(float(improvement))
        direction_count += int(improvement > 0.0)
        component_max_degradation = max(
            component_max_degradation,
            max(
                float(cand_value - ref_value)
                for cand_value, ref_value in zip(
                    cand_metric["component_RNMAE"], ref_metric["component_RNMAE"]
                )
            ),
        )
        if bootstrap is None:
            bootstrap = group_bootstrap_comparison(
                cand_prediction,
                ref_prediction,
                targets,
                np.asarray(dataset.group_ids, dtype=object)[common],
                seed=int(eval_config["bootstrap_seed"]),
                samples=int(eval_config["bootstrap_samples"]),
                target_ranges=np.full(3, 100.0),
            )
    mean_improvement = float(np.mean(seed_values)) if seed_values else None
    status = bool(
        mean_improvement is not None
        and mean_improvement >= float(gate["min_mean_relative_improvement"])
        and direction_count >= int(gate["min_seeds_same_direction"])
        and component_max_degradation <= float(gate["max_component_rnmae_degradation"])
        and bootstrap is not None
        and bootstrap["percentile_97_5"] < float(gate["paired_group_bootstrap_ci_upper_max"])
    )
    return {
        "status": "PASS" if status else "FAIL",
        "family": family,
        "horizon": horizon,
        "reference_model": ref_id,
        "seed_improvements": seed_values,
        "mean_relative_improvement": mean_improvement,
        "seeds_same_direction": int(direction_count),
        "component_max_rnmae_degradation": float(component_max_degradation),
        "paired_group_bootstrap": bootstrap,
    }


def _evaluate_headroom(
    prediction_store: Mapping[tuple[str, int, str, str], tuple[np.ndarray, np.ndarray]],
    dataset: Any,
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    del eval_config
    rows = dataset.indices(split="stress_val")
    values: dict[str, Any] = {}
    for candidate_id in ("B-EWMA", "B-STAT", "B-TCN"):
        candidate = prediction_store.get((candidate_id, 17, "P060", "stress_val"))
        oracle = prediction_store.get(("O-KIN-OBS", 0, "P060", "stress_val"))
        if candidate is None or oracle is None:
            values[candidate_id] = None
            continue
        common = np.intersect1d(candidate[0], oracle[0])
        if common.size == 0:
            values[candidate_id] = None
            continue
        candidate_map = {int(row): prediction for row, prediction in zip(*candidate)}
        oracle_map = {int(row): prediction for row, prediction in zip(*oracle)}
        oracle_values = np.asarray([oracle_map[int(row)] for row in common])
        finite = np.isfinite(oracle_values).all(axis=1)
        if not np.any(finite):
            values[candidate_id] = {"oracle_valid_count": 0}
            continue
        comparable_rows = common[finite]
        target = dataset.target[comparable_rows]
        candidate_values = np.asarray([candidate_map[int(row)] for row in comparable_rows])
        candidate_error = float(np.mean(np.abs(target - candidate_values)))
        oracle_error = float(np.mean(np.abs(target - oracle_values[finite])))
        values[candidate_id] = {
            "oracle_valid_count": int(finite.sum()),
            "candidate_valid_count": int(comparable_rows.size),
            "oracle_invalid_count": int((~finite).sum()),
            "oracle_headroom_fraction": None if candidate_error <= 0.0 else float((candidate_error - oracle_error) / candidate_error),
            "candidate_mae_mol_pct": candidate_error,
            "oracle_mae_mol_pct": oracle_error,
        }
    values["qualified_pressure_axes_observed"] = int(sum(str(dataset.records[int(row)]["family"]) != "D-IID" for row in rows))
    values["new_algorithm_handoff_allowed"] = False
    return values


def _classical_latency_fn(fitted: Any):
    model, mean, scale = fitted

    def predict(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).reshape(1, -1)
        return np.asarray(model.predict((array - mean) / scale), dtype=np.float64) * 100.0

    return predict


def _temporal_latency_fn(model: Any):
    model_id = str(getattr(model, "_model_id", "B-TCN"))
    mean = np.asarray(getattr(model, "_channel_mean"), dtype=np.float32)
    scale = np.asarray(getattr(model, "_channel_scale"), dtype=np.float32)

    def predict(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 2:
            array = array[None, ...]
        tensor = torch.from_numpy(((array - mean) / scale).astype(np.float32))
        if model_id == "B-TCN":
            tensor = tensor.transpose(1, 2)
        with torch.no_grad():
            return model(tensor).cpu().numpy() * 100.0

    return predict


def _measure_latency(predict_fn: Any, samples: np.ndarray) -> float | None:
    values = np.asarray(samples)
    if values.ndim == 0 or values.shape[0] == 0:
        return None
    limit = min(int(values.shape[0]), 64)
    for index in range(min(3, limit)):
        predict_fn(values[index])
    measurements = []
    for index in range(limit):
        started = time.perf_counter()
        prediction = predict_fn(values[index])
        elapsed = (time.perf_counter() - started) * 1000.0
        if not np.isfinite(prediction).all():
            raise ValueError("latency probe produced a non-finite prediction")
        measurements.append(elapsed)
    return float(np.percentile(np.asarray(measurements, dtype=np.float64), 95))


def _baseline_source_hashes(root: Path) -> dict[str, str]:
    paths = (
        "src/gf/pipeline/a2_dynamic_baselines.py",
        "src/gf/dl/temporal_baselines.py",
        "src/gf/dl/evaluation.py",
        "src/gf/sim/a2_dynamic_dataset.py",
        "src/gf/sim/a2_dynamic_audit/_baselines.py",
        "src/gf/sim/a2_dynamic_audit/_heos_interpolation.py",
    )
    return {path: _sha256_file(root / path) for path in paths}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


__all__ = [
    "assert_a2_dynamic_test_unlocked",
    "run_a2_dynamic_baselines",
    "run_a2_dynamic_replay_smoke",
    "run_a2_dynamic_report",
]
