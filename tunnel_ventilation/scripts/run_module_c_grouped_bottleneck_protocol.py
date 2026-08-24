"""Module C grouped bottleneck 单变量对照协议编排。

依据：tunnel_ventilation/docs/archive/completed/module_c_grouped_bottleneck_implementation_plan.md

只编排既有入口，不复制训练逻辑：
1. 复审 B7 protocol 的 12 个派生 split + RawDSP / fidelity / B7 provenance
2. 对每个 split 以单一冻结 training seed 跑 C1 physical 与 C2 permuted
3. 与 C0 B7 做同 split / 同 seed paired 汇总，输出 bottleneck verdict
4. 无论 --stage 为 train 或 all，均先执行前置审计

矩阵规模（为控制成本，不做 multi-seed）：
    4 protocols × 3 split seeds × 1 training seed × 2 variants = 24 条训练

用法（在 tunnel_ventilation 根目录）：

    python scripts/run_module_c_grouped_bottleneck_protocol.py --dry-run
    python scripts/run_module_c_grouped_bottleneck_protocol.py --stage all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tv3.ml.grouped_bottleneck import (
    EXPECTED_GROUP_COUNTS,
    EXPECTED_PARAMETER_COUNT,
    GROUP_SPEC_V1,
    PRE_REGISTERED_PERMUTATION_SEED,
    feature_names_digest,
)
from tv3.pipeline.multiseed_utils import load_json, run_command

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_ROOT = PROJECT_ROOT / "data" / "tv3-formal-6000-splits"
DEFAULT_B7_PROTOCOL_ROOT = PROJECT_ROOT / "outputs" / "tv3_b7_protocol"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tv3_module_c_grouped_bottleneck"
DEFAULT_RUNS_ROOT = DEFAULT_OUTPUT_ROOT
DEFAULT_SUMMARY_ROOT = DEFAULT_OUTPUT_ROOT
DEFAULT_REPORTS_ROOT = DEFAULT_OUTPUT_ROOT

C1_CONFIG = PROJECT_ROOT / "configs" / "tv3_module_c_grouped_bottleneck_physical.json"
C2_CONFIG = PROJECT_ROOT / "configs" / "tv3_module_c_grouped_bottleneck_permuted.json"

SPLIT_SEEDS: tuple[int, ...] = (20260704, 20260712, 20260720)
# 单 training seed：与 B7 C0 的 seed=42 配对；不做 42/123/456 multi-seed。
TRAINING_SEEDS: tuple[int, ...] = (42,)
PROTOCOL_IDS = ("R", "L", "S-Y", "S-L")
EXPECTED_ROWS_PER_VARIANT = len(PROTOCOL_IDS) * len(SPLIT_SEEDS) * len(TRAINING_SEEDS)  # 12
SUCCESS_STATUSES = frozenset({"ok", "revalidated_exists"})
MODULE_C_HEAD = "grouped_oof_ridge_residual_mlp"
DELTA_TOLERANCE = -0.01
PHYSICAL_VS_PERMUTED_OOD_GAIN = 0.01

FROZEN_MODULE_C_MLP = {
    "hidden_dims": [64, 64],
    "dropout": 0.1,
    "weight_decay": 0.0001,
    "lr": 0.001,
    "batch_size": 256,
    "max_epochs": 200,
    "patience": 20,
    "loss_weights": [1.0, 2.0, 1.0],
    "standardize_targets": True,
    "device": "cuda",
    "out_dim": 3,
    "zero_init_output": True,
}


def _load_b7_protocol_module():
    module_name = "module_c_b7_protocol_helper"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        PROJECT_ROOT / "scripts" / "run_b7_repeated_split_ood_protocol.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_protocol_matrix():
    return _load_b7_protocol_module().build_protocol_matrix()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_success_status(status: Any) -> bool:
    return status in SUCCESS_STATUSES


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _mean_std(values: list[float]) -> dict[str, float]:
    mean = _mean(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": round(mean, 6), "std": round(math.sqrt(var), 6), "n": len(values)}


def _validate_b7_protocol_root(b7_protocol_root: Path) -> None:
    split_metrics_path = b7_protocol_root / "split_metrics.json"
    if not split_metrics_path.is_file():
        raise FileNotFoundError(f"missing B7 split metrics: {split_metrics_path}")
    payload = load_json(split_metrics_path)
    verdict = payload.get("verdict")
    if not isinstance(verdict, dict):
        raise ValueError("B7 split metrics are missing verdict")
    if verdict.get("protocol_pass") is not True:
        raise ValueError("B7 split metrics verdict.protocol_pass must be true")
    if verdict.get("matrix_complete") is not True:
        raise ValueError("B7 split metrics verdict.matrix_complete must be true")
    if verdict.get("unexpected_row_count") != 0:
        raise ValueError("B7 split metrics verdict.unexpected_row_count must be 0")

    b7 = _load_b7_protocol_module()
    expected_keys = {
        (spec.protocol_id, spec.split_seed, training_seed)
        for spec in b7.build_protocol_matrix()
        for training_seed in b7.TRAINING_SEEDS
    }
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("B7 split metrics are missing rows")
    actual_keys = {
        (
            str(row.get("protocol_id")),
            int(row.get("split_seed")),
            int(row.get("training_seed")),
        )
        for row in rows
        if isinstance(row, dict)
    }
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError(
            "B7 split metrics rows do not match the complete frozen protocol matrix"
        )


def load_c0_b7_matrix(b7_protocol_root: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Load frozen B7 protocol rows as C0 anchors for the Module C training-seed set.

    B7 formal matrix may contain multi-seed rows; Module C only keeps TRAINING_SEEDS
    (currently the single seed 42) for paired comparison.
    """
    _validate_b7_protocol_root(b7_protocol_root)
    matrix_path = b7_protocol_root / "result_matrix.csv"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"missing B7 protocol result matrix: {matrix_path}")
    rows_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            training_seed = int(raw["training_seed"])
            if training_seed not in TRAINING_SEEDS:
                continue
            key = (
                str(raw["protocol_id"]),
                int(raw["split_seed"]),
                training_seed,
            )
            if key in rows_by_key:
                raise ValueError(f"duplicate C0 B7 row for {key}")
            if not _is_success_status(raw.get("b7_status")):
                raise ValueError(f"C0 B7 row not successful for {key}: {raw.get('b7_status')!r}")
            b7_metrics_path = b7_protocol_root / str(raw["dataset_name"]) / "b7_s42" / "metrics.json"
            b7_metrics = load_json(b7_metrics_path)
            b7_feature_names = b7_metrics.get("feature_names")
            if not isinstance(b7_feature_names, list):
                raise ValueError(f"C0 B7 metrics are missing feature_names: {b7_metrics_path}")
            rows_by_key[key] = {
                "protocol_id": key[0],
                "dataset_name": raw["dataset_name"],
                "split_seed": key[1],
                "training_seed": key[2],
                "is_ood_evidence": str(raw.get("is_ood_evidence")).lower() in {"true", "1"},
                "c0_status": raw.get("b7_status"),
                "c0_test_o2_r2": _optional_float(raw.get("b7_test_o2_r2")),
                "c0_extrapolation_o2_r2": _optional_float(raw.get("b7_extrapolation_o2_r2")),
                "c0_val_o2_r2": _optional_float(raw.get("b7_val_o2_r2")),
                "c0_feature_names_digest": feature_names_digest(b7_feature_names),
            }
    expected = {
        (protocol_id, split_seed, training_seed)
        for protocol_id in PROTOCOL_IDS
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    }
    missing = sorted(expected - set(rows_by_key))
    if missing:
        raise ValueError(f"C0 B7 matrix incomplete, missing {len(missing)} rows, e.g. {missing[:3]}")
    return rows_by_key


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _audit_prerequisites(
    spec,
    *,
    splits_root: Path,
    b7_protocol_root: Path,
    source_dir: Path | None = None,
    raw_dsp_bootstrap: Path | None = None,
) -> list[str]:
    b7 = _load_b7_protocol_module()
    dataset_dir = splits_root / spec.dataset_name
    resolved_source = source_dir or b7.DEFAULT_SOURCE_DIR
    resolved_bootstrap = raw_dsp_bootstrap or b7.DEFAULT_RAW_DSP_BOOTSTRAP
    errors: list[str] = []
    _summary, split_errors = b7._audit_derived_split(
        spec,
        source_dir=resolved_source,
        splits_root=splits_root,
        raw_dsp_bootstrap=resolved_bootstrap,
    )
    errors.extend(split_errors)
    cache_dir = dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1"
    errors.extend(b7._audit_raw_dsp_cache(dataset_dir, cache_dir))
    fidelity_path = b7_protocol_root / spec.dataset_name / "raw_dsp_fidelity" / "metrics.json"
    if not fidelity_path.is_file():
        errors.append(f"missing fidelity metrics: {fidelity_path}")
    else:
        errors.extend(b7._audit_fidelity_metrics(fidelity_path, dataset_dir))
    b7_metrics = b7_protocol_root / spec.dataset_name / "b7_s42" / "metrics.json"
    if not b7_metrics.is_file():
        errors.append(f"missing B7 provenance metrics: {b7_metrics}")
    b1_metrics = b7_protocol_root / spec.dataset_name / "b1" / "metrics.json"
    if not b1_metrics.is_file():
        errors.append(f"missing B1 reference metrics: {b1_metrics}")
    return errors


def run_module_c_seed(
    spec,
    *,
    variant: str,
    training_seed: int,
    splits_root: Path,
    runs_root: Path,
    b7_protocol_root: Path,
    c0_feature_names_digest: str,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    if variant not in {"physical", "permuted"}:
        raise ValueError(f"unknown variant {variant!r}")
    config_path = C1_CONFIG if variant == "physical" else C2_CONFIG
    dataset_dir = splits_root / spec.dataset_name
    run_dir = runs_root / variant / spec.protocol_id / f"split_{spec.split_seed}" / f"seed_{training_seed}"
    metrics_path = run_dir / "metrics.json"
    fidelity_path = b7_protocol_root / spec.dataset_name / "raw_dsp_fidelity" / "metrics.json"
    b1_path = b7_protocol_root / spec.dataset_name / "b1" / "metrics.json"
    record: dict[str, Any] = {
        "stage": "module_c",
        "variant": variant,
        "protocol_id": spec.protocol_id,
        "dataset_name": spec.dataset_name,
        "split_seed": spec.split_seed,
        "training_seed": training_seed,
        "model": f"c_{variant}_grouped_bottleneck",
        "run_dir": str(run_dir),
        "config_sha256": _file_sha256(config_path),
    }
    if metrics_path.is_file() and not overwrite and not dry_run:
        payload = load_json(metrics_path)
        errors = _audit_module_c_payload(
            payload,
            variant=variant,
            training_seed=training_seed,
            dataset_dir=dataset_dir,
            expected_feature_names_digest=c0_feature_names_digest,
        )
        record["status"] = "revalidated_exists" if not errors else "audit_fail"
        record["audit_errors"] = errors
        record["metrics"] = _extract_run_metrics(payload)
        return record

    cmd = [
        sys.executable,
        "-m",
        "tv3.pipeline.run_tv3_rocket_baseline",
        "--config",
        str(config_path),
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(training_seed),
        "--raw-dsp-fidelity-metrics-path",
        str(fidelity_path),
        "--raw-dsp-reference-metrics-path",
        str(b1_path),
        "--device",
        "cuda",
    ]
    if overwrite:
        cmd.append("--overwrite")
    record["command"] = cmd
    started = time.perf_counter()
    proc = run_command(cmd, cwd=PROJECT_ROOT, dry_run=dry_run)
    record["elapsed_s"] = round(time.perf_counter() - started, 2)
    if dry_run or proc is None:
        record["status"] = "dry_run"
        return record
    if proc.returncode != 0 or not metrics_path.is_file():
        record["status"] = "fail"
        record["returncode"] = proc.returncode
        return record
    payload = load_json(metrics_path)
    errors = _audit_module_c_payload(
        payload,
        variant=variant,
        training_seed=training_seed,
        dataset_dir=dataset_dir,
        expected_feature_names_digest=c0_feature_names_digest,
    )
    record["status"] = "ok" if not errors else "audit_fail"
    record["audit_errors"] = errors
    record["metrics"] = _extract_run_metrics(payload)
    return record


def _audit_module_c_payload(
    payload: dict[str, Any],
    *,
    variant: str,
    training_seed: int,
    dataset_dir: Path,
    expected_feature_names_digest: str,
) -> list[str]:
    errors: list[str] = []
    if payload.get("head") != MODULE_C_HEAD:
        errors.append(f"head={payload.get('head')!r}")
    if payload.get("feature_builder") != "d0_raw_dsp_physics_stats_v1":
        errors.append(f"feature_builder={payload.get('feature_builder')!r}")
    if payload.get("feature_count") != 1008:
        errors.append(f"feature_count={payload.get('feature_count')!r}")
    if Path(str(payload.get("dataset_dir", ""))).resolve() != dataset_dir.resolve():
        errors.append("dataset_dir mismatch")

    grouped = payload.get("grouped_bottleneck")
    if not isinstance(grouped, dict):
        errors.append("grouped_bottleneck missing")
        grouped = {}
    if grouped.get("group_spec") != GROUP_SPEC_V1:
        errors.append(f"group_spec={grouped.get('group_spec')!r}")
    expected_assignment = "physical" if variant == "physical" else "permuted"
    if grouped.get("group_assignment") != expected_assignment:
        errors.append(f"group_assignment={grouped.get('group_assignment')!r}")
    if grouped.get("group_counts") != EXPECTED_GROUP_COUNTS:
        errors.append(f"group_counts={grouped.get('group_counts')!r}")
    if grouped.get("group_bottleneck_dim") != 16:
        errors.append(f"group_bottleneck_dim={grouped.get('group_bottleneck_dim')!r}")
    if float(grouped.get("group_dropout", -1)) != 0.0:
        errors.append(f"group_dropout={grouped.get('group_dropout')!r}")
    if grouped.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        errors.append(f"parameter_count={grouped.get('parameter_count')!r}")
    if grouped.get("feature_names_digest") != expected_feature_names_digest:
        errors.append("feature_names_digest does not match the paired C0 B7 metrics")
    if variant == "permuted":
        if grouped.get("permutation_seed") != PRE_REGISTERED_PERMUTATION_SEED:
            errors.append(f"permutation_seed={grouped.get('permutation_seed')!r}")
        if not grouped.get("permutation_digest"):
            errors.append("missing permutation_digest")
    else:
        if grouped.get("permutation_seed") is not None:
            errors.append("physical assignment must have null permutation_seed")
        if grouped.get("permutation_digest") not in {"", None}:
            errors.append("physical assignment must have empty permutation_digest")

    early = payload.get("early_stopping")
    if not isinstance(early, dict) or early.get("monitor") != "val_o2_r2":
        errors.append("early_stopping.monitor must be val_o2_r2")
    elif early.get("uses_combined_ridge_prediction") is not True:
        errors.append("early_stopping.uses_combined_ridge_prediction must be true")

    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return [*errors, "diagnostics missing"]
    residual = diagnostics.get("residual_mlp")
    if not isinstance(residual, dict):
        errors.append("diagnostics.residual_mlp missing")
        residual = {}
    model_config = residual.get("model_config", {})
    if not isinstance(model_config, dict):
        errors.append("residual model_config missing")
        model_config = {}
    if model_config.get("seed") != training_seed:
        errors.append(f"training seed mismatch: {model_config.get('seed')} != {training_seed}")
    for field, expected in FROZEN_MODULE_C_MLP.items():
        if model_config.get(field) != expected:
            errors.append(f"model_config.{field}={model_config.get(field)!r}, expected {expected!r}")
    if residual.get("standardize_targets") is not True:
        errors.append("standardize_targets must be true")
    if residual.get("zero_init_output") is not True:
        errors.append("zero_init_output must be true")
    oof = diagnostics.get("oof")
    if not isinstance(oof, dict):
        errors.append("diagnostics.oof missing")
        oof = {}
    if oof.get("fold_count") != 5:
        errors.append(f"oof fold_count={oof.get('fold_count')!r}")
    if oof.get("fold_seed") != 20260711:
        errors.append(f"oof fold_seed={oof.get('fold_seed')!r}")
    if oof.get("coverage_complete") is not True:
        errors.append("oof coverage incomplete")
    leakage = diagnostics.get("leakage_audit")
    if not isinstance(leakage, dict):
        errors.append("leakage_audit missing")
        leakage = {}
    for field in (
        "oof_used_for_residual_targets",
        "full_ridge_fit_on_train_only",
        "val_residual_from_full_ridge",
        "oof_coverage_complete",
    ):
        if leakage.get(field) is not True:
            errors.append(f"leakage_audit.{field}={leakage.get(field)!r}")

    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, dict):
        errors.append("evaluations missing")
        evaluations = {}
    for split_name in ("train", "val", "test", "extrapolation"):
        split_payload = evaluations.get(split_name)
        if not isinstance(split_payload, dict):
            errors.append(f"missing evaluations.{split_name}")
            continue
        comps = split_payload.get("component_metrics")
        if not isinstance(comps, dict):
            errors.append(f"missing component_metrics for {split_name}")
            continue
        for gas in ("x_CO2", "x_O2", "x_N2"):
            if gas not in comps or "r2" not in comps[gas]:
                errors.append(f"missing {split_name}.{gas}.r2")
        if "sum_abs_error" not in split_payload:
            errors.append(f"missing sum_abs_error for {split_name}")
    return errors


def _extract_run_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    evaluations = payload.get("evaluations", {})
    out: dict[str, Any] = {
        "components": {},
        "sum_abs_error": {},
        "parameter_count": (payload.get("grouped_bottleneck") or {}).get("parameter_count"),
        "grouped_bottleneck": payload.get("grouped_bottleneck"),
    }
    for split_name, split_payload in evaluations.items():
        if not isinstance(split_payload, dict):
            continue
        comp = split_payload.get("component_metrics", {})
        out["components"][split_name] = {
            gas: {
                "r2": float(metrics.get("r2")) if "r2" in metrics else None,
                "mae": float(metrics.get("mae")) if "mae" in metrics else None,
                "rmse": float(metrics.get("rmse")) if "rmse" in metrics else None,
            }
            for gas, metrics in comp.items()
            if isinstance(metrics, dict)
        }
        if "sum_abs_error" in split_payload:
            out["sum_abs_error"][split_name] = float(split_payload["sum_abs_error"])
    return out


def _matrix_row(
    *,
    spec,
    variant: str,
    c0: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    metrics = run.get("metrics") or {}
    row: dict[str, Any] = {
        "variant": variant,
        "protocol_id": spec.protocol_id,
        "dataset_name": spec.dataset_name,
        "split_seed": spec.split_seed,
        "training_seed": run.get("training_seed"),
        "is_ood_evidence": spec.is_ood_evidence,
        "status": run.get("status"),
        "c0_status": c0.get("c0_status"),
        "parameter_count": metrics.get("parameter_count"),
    }
    for split_name in ("val", "test", "extrapolation"):
        cand = metrics.get("components", {}).get(split_name, {}).get("x_O2", {}).get("r2")
        c0_key = f"c0_{split_name}_o2_r2"
        c0_val = c0.get(c0_key)
        row[f"c1c2_{split_name}_o2_r2"] = cand
        row[f"c0_{split_name}_o2_r2"] = c0_val
        if cand is not None and c0_val is not None:
            row[f"delta_vs_c0_{split_name}"] = float(cand) - float(c0_val)
        else:
            row[f"delta_vs_c0_{split_name}"] = None
        for gas in ("x_CO2", "x_N2"):
            row[f"{split_name}_{gas}_r2"] = (
                metrics.get("components", {}).get(split_name, {}).get(gas, {}).get("r2")
            )
        row[f"{split_name}_sum_abs_error"] = metrics.get("sum_abs_error", {}).get(split_name)
    return row


def evaluate_module_c_verdict(
    physical_rows: list[dict[str, Any]],
    permuted_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pre-registered Module C verdict gates (§5.2 / §5.3)."""
    completeness = _matrix_completeness(physical_rows, permuted_rows)
    if not completeness["complete"]:
        return {
            "verdict": "audit_failed",
            "reason": "matrix incomplete or unpaired with C0",
            "completeness": completeness,
        }

    physical_by_key = _index_rows(physical_rows)
    permuted_by_key = _index_rows(permuted_rows)

    # Gate: parameter counts
    for rows in (physical_rows, permuted_rows):
        for row in rows:
            if row.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
                return {
                    "verdict": "audit_failed",
                    "reason": f"parameter_count != {EXPECTED_PARAMETER_COUNT}",
                    "completeness": completeness,
                }

    checks: dict[str, Any] = {}
    c1_vs_c0_ok = True
    c1_vs_c2_ok = True
    ood_gain_hit = False

    for protocol_id in PROTOCOL_IDS:
        protocol_keys = [
            (protocol_id, split_seed, training_seed)
            for split_seed in SPLIT_SEEDS
            for training_seed in TRAINING_SEEDS
        ]
        phys = [physical_by_key[key] for key in protocol_keys]
        perm = [permuted_by_key[key] for key in protocol_keys]
        test_c1_c0 = [float(row["delta_vs_c0_test"]) for row in phys]
        test_mean = _mean(test_c1_c0)
        test_non_inferior = test_mean >= DELTA_TOLERANCE
        protocol_check: dict[str, Any] = {
            "c1_vs_c0_test": _mean_std(test_c1_c0),
            "c1_vs_c0_test_non_inferior": test_non_inferior,
        }
        if not test_non_inferior:
            c1_vs_c0_ok = False

        if protocol_id in {"S-Y", "S-L"}:
            ood_c1_c0 = [float(row["delta_vs_c0_extrapolation"]) for row in phys]
            ood_mean = _mean(ood_c1_c0)
            ood_non_inferior = ood_mean >= DELTA_TOLERANCE
            # 单 training seed 时不做 multi-seed cluster 门；仅看 split 间 paired mean。
            worst_split_vs_c0 = min(ood_c1_c0)
            protocol_check["c1_vs_c0_extrapolation"] = _mean_std(ood_c1_c0)
            protocol_check["c1_vs_c0_extrapolation_non_inferior"] = ood_non_inferior
            protocol_check["worst_split_delta_vs_c0_extrapolation"] = worst_split_vs_c0
            if not ood_non_inferior:
                c1_vs_c0_ok = False

            test_c1_c2 = [
                float(p["c1c2_test_o2_r2"]) - float(q["c1c2_test_o2_r2"])
                for p, q in zip(phys, perm, strict=True)
            ]
            ood_c1_c2 = [
                float(p["c1c2_extrapolation_o2_r2"]) - float(q["c1c2_extrapolation_o2_r2"])
                for p, q in zip(phys, perm, strict=True)
            ]
            test_mean_c2 = _mean(test_c1_c2)
            ood_mean_c2 = _mean(ood_c1_c2)
            protocol_check["c1_vs_c2_test"] = _mean_std(test_c1_c2)
            protocol_check["c1_vs_c2_extrapolation"] = _mean_std(ood_c1_c2)
            protocol_check["c1_vs_c2_non_negative"] = test_mean_c2 >= 0.0 and ood_mean_c2 >= 0.0
            protocol_check["worst_split_delta_vs_c2_extrapolation"] = min(ood_c1_c2)
            if test_mean_c2 < 0.0 or ood_mean_c2 < 0.0:
                c1_vs_c2_ok = False
            if ood_mean_c2 >= PHYSICAL_VS_PERMUTED_OOD_GAIN:
                ood_gain_hit = True
        checks[protocol_id] = protocol_check

    if not c1_vs_c0_ok:
        verdict = "grouped_failed"
    elif not c1_vs_c2_ok or not ood_gain_hit:
        verdict = "compression_only"
    else:
        verdict = "bottleneck_pass"

    return {
        "verdict": verdict,
        "c1_vs_c0_non_inferior": c1_vs_c0_ok,
        "c1_vs_c2_directional": c1_vs_c2_ok,
        "ood_gain_hit": ood_gain_hit,
        "training_seeds": list(TRAINING_SEEDS),
        "expected_rows_per_variant": EXPECTED_ROWS_PER_VARIANT,
        "checks": checks,
        "completeness": completeness,
        "note": (
            "Single training seed (42) only; multi-seed replication is intentionally skipped. "
            "No p-values. S-Y and S-L must be reported separately."
        ),
    }


def _index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["protocol_id"]), int(row["split_seed"]), int(row["training_seed"]))
        indexed[key] = row
    return indexed


def _matrix_completeness(
    physical_rows: list[dict[str, Any]],
    permuted_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = {
        (protocol_id, split_seed, training_seed)
        for protocol_id in PROTOCOL_IDS
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    }
    physical_keys = set(_index_rows(physical_rows))
    permuted_keys = set(_index_rows(permuted_rows))
    physical_ok = all(
        _is_success_status(row.get("status"))
        and _is_success_status(row.get("c0_status"))
        and row.get("delta_vs_c0_test") is not None
        and row.get("delta_vs_c0_extrapolation") is not None
        for row in physical_rows
    )
    permuted_ok = all(
        _is_success_status(row.get("status"))
        and _is_success_status(row.get("c0_status"))
        and row.get("delta_vs_c0_test") is not None
        and row.get("delta_vs_c0_extrapolation") is not None
        for row in permuted_rows
    )
    complete = (
        physical_keys == expected
        and permuted_keys == expected
        and physical_ok
        and permuted_ok
        and len(physical_rows) == EXPECTED_ROWS_PER_VARIANT
        and len(permuted_rows) == EXPECTED_ROWS_PER_VARIANT
    )
    return {
        "complete": complete,
        "physical_count": len(physical_rows),
        "permuted_count": len(permuted_rows),
        "missing_physical": sorted(expected - physical_keys),
        "missing_permuted": sorted(expected - permuted_keys),
    }


def write_outputs(
    *,
    physical_rows: list[dict[str, Any]],
    permuted_rows: list[dict[str, Any]],
    verdict: dict[str, Any],
    summary_root: Path,
    reports_root: Path,
    manifest: dict[str, Any],
) -> None:
    all_rows = physical_rows + permuted_rows
    summary_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    _write_json(summary_root / "protocol_manifest.json", manifest)
    _write_json(
        summary_root / "split_metrics.json",
        {"physical_rows": physical_rows, "permuted_rows": permuted_rows, "verdict": verdict},
    )
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with (summary_root / "result_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    md_lines = [
        "# Module C Grouped Bottleneck Result Matrix",
        "",
        f"Verdict: **{verdict.get('verdict')}**",
        "",
        "| variant | protocol | split_seed | train_seed | test O2 R2 | Δtest vs C0 | OOD O2 R2 | ΔOOD vs C0 | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in all_rows:
        md_lines.append(
            "| {variant} | {protocol} | {split} | {seed} | {test} | {dtest} | {ood} | {dood} | {status} |".format(
                variant=row.get("variant"),
                protocol=row.get("protocol_id"),
                split=row.get("split_seed"),
                seed=row.get("training_seed"),
                test=_fmt(row.get("c1c2_test_o2_r2")),
                dtest=_fmt(row.get("delta_vs_c0_test")),
                ood=_fmt(row.get("c1c2_extrapolation_o2_r2")),
                dood=_fmt(row.get("delta_vs_c0_extrapolation")),
                status=row.get("status"),
            )
        )
    (reports_root / "result_matrix.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    verdict_md = [
        "# Module C Verdict",
        "",
        f"- verdict: `{verdict.get('verdict')}`",
        f"- c1_vs_c0_non_inferior: {verdict.get('c1_vs_c0_non_inferior')}",
        f"- c1_vs_c2_directional: {verdict.get('c1_vs_c2_directional')}",
        f"- ood_gain_hit: {verdict.get('ood_gain_hit')}",
        "",
        "## Checks",
        "",
        "```json",
        json.dumps(verdict.get("checks", {}), indent=2, ensure_ascii=False),
        "```",
        "",
        verdict.get("note", ""),
        "",
    ]
    (reports_root / "verdict.md").write_text("\n".join(verdict_md), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Module C grouped bottleneck protocol.")
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--b7-protocol-root", type=Path, default=DEFAULT_B7_PROTOCOL_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--summary-root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--stage", choices=("audit", "train", "all"), default="all")
    parser.add_argument("--protocol", default="all", help="Comma-separated R,L,S-Y,S-L or all.")
    parser.add_argument("--split-seeds", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_int_list(value: str, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if value == "all":
        return default
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    protocol_filter = (
        list(PROTOCOL_IDS)
        if args.protocol == "all"
        else [item.strip() for item in args.protocol.split(",") if item.strip()]
    )
    split_seeds = _parse_int_list(args.split_seeds, default=SPLIT_SEEDS)
    training_seeds = TRAINING_SEEDS

    all_specs = build_protocol_matrix()
    specs = [
        spec
        for spec in all_specs
        if spec.protocol_id in protocol_filter and spec.split_seed in split_seeds
    ]
    if not specs:
        raise ValueError("no protocol specs selected")

    c0_matrix = load_c0_b7_matrix(args.b7_protocol_root)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits_root": str(args.splits_root),
        "b7_protocol_root": str(args.b7_protocol_root),
        "runs_root": str(args.runs_root),
        "c1_config_sha256": _file_sha256(C1_CONFIG),
        "c2_config_sha256": _file_sha256(C2_CONFIG),
        "split_seeds": list(split_seeds),
        "training_seeds": list(training_seeds),
        "protocol_ids": protocol_filter,
        "variants": ["physical", "permuted"],
        "is_complete_formal_matrix": (
            set(protocol_filter) == set(PROTOCOL_IDS)
            and set(split_seeds) == set(SPLIT_SEEDS)
            and set(training_seeds) == set(TRAINING_SEEDS)
        ),
        "matrix": [asdict(spec) | {"dataset_name": spec.dataset_name} for spec in specs],
        "invariants": {
            "feature_builder": "d0_raw_dsp_physics_stats_v1",
            "feature_count": 1008,
            "head": MODULE_C_HEAD,
            "group_spec": GROUP_SPEC_V1,
            "group_bottleneck_dim": 16,
            "group_dropout": 0.0,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "permutation_seed": PRE_REGISTERED_PERMUTATION_SEED,
            "oof_folds": 5,
            "oof_seed": 20260711,
            "mlp_hidden_dims": [64, 64],
            "early_stopping": "val_o2_r2_combined",
        },
    }
    args.summary_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.summary_root / "protocol_manifest.json", manifest)
    runs_path = args.summary_root / "runs.jsonl"

    if args.dry_run:
        for spec in specs:
            print(
                f"[DRY-RUN] audit+train {spec.protocol_id} {spec.dataset_name} "
                f"split_seed={spec.split_seed}"
            )
            for seed in training_seeds:
                print(f"  - C1 physical seed={seed}; C2 permuted seed={seed}")
        print("\n[DRY-RUN] wrote protocol_manifest.json only beyond dry-run prints")
        return 0

    physical_rows: list[dict[str, Any]] = []
    permuted_rows: list[dict[str, Any]] = []
    failures = 0

    for spec in specs:
        audit_errors = _audit_prerequisites(
            spec,
            splits_root=args.splits_root,
            b7_protocol_root=args.b7_protocol_root,
        )
        audit_record = {
            "stage": "audit",
            "protocol_id": spec.protocol_id,
            "dataset_name": spec.dataset_name,
            "status": "ok" if not audit_errors else "audit_fail",
            "audit_errors": audit_errors,
        }
        _append_jsonl(runs_path, audit_record)
        if audit_errors:
            failures += 1
            continue

        if args.stage in {"train", "all"}:
            for training_seed in training_seeds:
                c0 = c0_matrix[(spec.protocol_id, spec.split_seed, training_seed)]
                for variant, sink in (("physical", physical_rows), ("permuted", permuted_rows)):
                    run = run_module_c_seed(
                        spec,
                        variant=variant,
                        training_seed=training_seed,
                        splits_root=args.splits_root,
                        runs_root=args.runs_root,
                        b7_protocol_root=args.b7_protocol_root,
                        c0_feature_names_digest=str(c0["c0_feature_names_digest"]),
                        dry_run=False,
                        overwrite=args.overwrite,
                    )
                    _append_jsonl(runs_path, {k: v for k, v in run.items() if k != "command"})
                    if not _is_success_status(run.get("status")):
                        failures += 1
                    sink.append(_matrix_row(spec=spec, variant=variant, c0=c0, run=run))

    if physical_rows or permuted_rows:
        verdict = evaluate_module_c_verdict(physical_rows, permuted_rows)
        write_outputs(
            physical_rows=physical_rows,
            permuted_rows=permuted_rows,
            verdict=verdict,
            summary_root=args.summary_root,
            reports_root=args.reports_root,
            manifest=manifest,
        )
        print(f"\nverdict: {verdict['verdict']}")
        print(json.dumps(verdict.get("checks", {}), indent=2, ensure_ascii=False))

    print(f"\nwritten under: {args.runs_root}")
    print(f"summary: {args.summary_root}")
    print(f"reports: {args.reports_root}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
