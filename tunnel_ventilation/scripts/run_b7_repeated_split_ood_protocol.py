"""B7 冻结后的重复 split × 独立 OOD 协议编排。

依据：tunnel_ventilation/docs/active/b7_repeated_split_ood_protocol_implementation_plan.md

只负责编排既有入口，不复制训练逻辑、不重定义 B7 超参数：
1. 派生 random / LHS / SPXY-observed splits（跳过 source RawDSP hardlink）
2. 按派生 split 的 train 重建 RawDSP cache + fidelity 审计
3. 每 split 跑一次 B1 RawDSP Ridge，再跑 B7 三 training seed
4. 汇总 result_matrix / protocol_manifest

用法（在 tunnel_ventilation 根目录）：

    python scripts/run_b7_repeated_split_ood_protocol.py --dry-run
    python scripts/run_b7_repeated_split_ood_protocol.py --stage derive
    python scripts/run_b7_repeated_split_ood_protocol.py --stage all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tv3.pipeline.multiseed_utils import load_json, run_command, verify_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "tv3-formal-6000"
DEFAULT_SPLITS_ROOT = PROJECT_ROOT / "data" / "tv3-formal-6000-splits"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tv3_b7_protocol"
DEFAULT_RAW_DSP_BOOTSTRAP = (
    PROJECT_ROOT / "data" / "tv3-formal-6000" / "features" / "raw_dsp" / "raw_dsp_frame_v1"
)
B1_CONFIG = PROJECT_ROOT / "configs" / "tv3_d2b_raw_dsp_ridge.json"
B7_CONFIG = PROJECT_ROOT / "configs" / "tv3_d2b_oof_ridge_residual_mlp.json"
RAW_DSP_FEATURES_CONFIG = PROJECT_ROOT / "configs" / "tv3_d2b_raw_dsp_features.json"

SPLIT_SEEDS: tuple[int, ...] = (20260704, 20260712, 20260720)
TRAINING_SEEDS: tuple[int, ...] = (42, 123, 456)
PROTOCOL_IDS = ("R", "L", "S-Y", "S-L")


@dataclass(frozen=True)
class ProtocolSplitSpec:
    protocol_id: str
    split_seed: int
    split_strategy: str
    extrapolation_strategy: str
    spxy_alpha: float | None = None
    spxy_x_profile: str | None = None
    is_ood_evidence: bool = False

    @property
    def dataset_name(self) -> str:
        if self.protocol_id == "R":
            return f"random_s{self.split_seed}"
        if self.protocol_id == "L":
            return f"lhs_s{self.split_seed}"
        if self.protocol_id == "S-Y":
            return f"spxy_observed_a05_ymargin_s{self.split_seed}"
        if self.protocol_id == "S-L":
            return f"spxy_observed_a05_lhsboundary_s{self.split_seed}"
        raise ValueError(f"unknown protocol_id={self.protocol_id!r}")


def build_protocol_matrix() -> list[ProtocolSplitSpec]:
    specs: list[ProtocolSplitSpec] = []
    for seed in SPLIT_SEEDS:
        specs.append(
            ProtocolSplitSpec(
                protocol_id="R",
                split_seed=seed,
                split_strategy="random",
                extrapolation_strategy="none",
                is_ood_evidence=False,
            )
        )
        specs.append(
            ProtocolSplitSpec(
                protocol_id="L",
                split_seed=seed,
                split_strategy="lhs_stratified_split_v1",
                extrapolation_strategy="none",
                is_ood_evidence=False,
            )
        )
        specs.append(
            ProtocolSplitSpec(
                protocol_id="S-Y",
                split_seed=seed,
                split_strategy="spxy_v1",
                extrapolation_strategy="y_margin_ood",
                spxy_alpha=0.5,
                spxy_x_profile="observed_v1",
                is_ood_evidence=True,
            )
        )
        specs.append(
            ProtocolSplitSpec(
                protocol_id="S-L",
                split_seed=seed,
                split_strategy="spxy_v1",
                extrapolation_strategy="lhs_boundary",
                spxy_alpha=0.5,
                spxy_x_profile="observed_v1",
                is_ood_evidence=True,
            )
        )
    return specs


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def derive_split(
    spec: ProtocolSplitSpec,
    *,
    source_dir: Path,
    splits_root: Path,
    raw_dsp_bootstrap: Path,
    dry_run: bool,
) -> dict[str, Any]:
    output_dir = splits_root / spec.dataset_name
    summary_path = output_dir / "splits" / "split_summary.json"
    record: dict[str, Any] = {
        "stage": "derive",
        "protocol_id": spec.protocol_id,
        "dataset_name": spec.dataset_name,
        "output_dir": str(output_dir),
        "split_seed": spec.split_seed,
    }
    if summary_path.is_file() and not dry_run:
        summary = load_json(summary_path)
        record["status"] = "skipped_exists"
        record["split_hash"] = summary.get("split_hash")
        record["x_feature_profile"] = summary.get("x_feature_profile")
        record["ood_set_hash"] = summary.get("ood_set_hash")
        return record

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "recompute_tv3_split.py"),
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(output_dir),
        "--split-strategy",
        spec.split_strategy,
        "--seed",
        str(spec.split_seed),
        "--extrapolation-strategy",
        spec.extrapolation_strategy,
    ]
    if spec.spxy_alpha is not None:
        cmd.extend(["--spxy-alpha", str(spec.spxy_alpha)])
    if spec.spxy_x_profile is not None:
        cmd.extend(["--spxy-x-profile", spec.spxy_x_profile])
        cmd.extend(["--raw-dsp-cache-dir", str(raw_dsp_bootstrap)])

    started = time.perf_counter()
    proc = run_command(cmd, cwd=PROJECT_ROOT, dry_run=dry_run)
    record["elapsed_s"] = round(time.perf_counter() - started, 2)
    record["command"] = cmd
    if dry_run or proc is None:
        record["status"] = "dry_run"
        return record
    if proc.returncode != 0:
        record["status"] = "fail"
        record["returncode"] = proc.returncode
        return record
    summary = load_json(summary_path)
    if spec.spxy_x_profile == "observed_v1":
        if summary.get("x_feature_profile") != "spxy_observed_stats_v1":
            record["status"] = "audit_fail"
            record["reason"] = (
                f"expected x_feature_profile=spxy_observed_stats_v1, got {summary.get('x_feature_profile')!r}"
            )
            return record
        if (output_dir / "features" / "raw_dsp").exists():
            record["status"] = "audit_fail"
            record["reason"] = "source RawDSP cache was hard-linked into derived split"
            return record
    record["status"] = "ok"
    record["split_hash"] = summary.get("split_hash")
    record["x_feature_profile"] = summary.get("x_feature_profile")
    record["ood_set_hash"] = summary.get("ood_set_hash")
    return record


def build_raw_dsp_for_split(
    spec: ProtocolSplitSpec,
    *,
    splits_root: Path,
    dry_run: bool,
    workers: int | None,
) -> dict[str, Any]:
    dataset_dir = splits_root / spec.dataset_name
    cache_dir = dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1"
    manifest_path = cache_dir / "manifest.json"
    record: dict[str, Any] = {
        "stage": "raw_dsp",
        "protocol_id": spec.protocol_id,
        "dataset_name": spec.dataset_name,
        "cache_dir": str(cache_dir),
    }
    if manifest_path.is_file() and not dry_run:
        manifest = load_json(manifest_path)
        record["status"] = "skipped_exists"
        record["build_signature"] = manifest.get("build_signature")
        record["split_hash"] = manifest.get("split_hash")
        record["template_source_split"] = manifest.get("template_source_split")
        return record

    cmd = [
        sys.executable,
        "-m",
        "tv3.pipeline.build_tv3_raw_dsp_features",
        "--config",
        str(RAW_DSP_FEATURES_CONFIG),
        "--dataset-dir",
        str(dataset_dir),
        "--cache-dir",
        str(cache_dir),
    ]
    if workers is not None:
        cmd.extend(["--workers", str(workers)])
    started = time.perf_counter()
    proc = run_command(cmd, cwd=PROJECT_ROOT, dry_run=dry_run)
    record["elapsed_s"] = round(time.perf_counter() - started, 2)
    record["command"] = cmd
    if dry_run or proc is None:
        record["status"] = "dry_run"
        return record
    if proc.returncode != 0:
        record["status"] = "fail"
        record["returncode"] = proc.returncode
        return record
    manifest = load_json(manifest_path)
    split_summary = load_json(dataset_dir / "splits" / "split_summary.json")
    errors: list[str] = []
    if manifest.get("template_source_split") != "train":
        errors.append(f"template_source_split={manifest.get('template_source_split')!r}")
    if manifest.get("split_hash") != split_summary.get("split_hash"):
        errors.append("RawDSP manifest split_hash does not match split_summary")
    if manifest.get("diagnostic_only") is True:
        errors.append("diagnostic_only cache is not allowed for protocol")
    if errors:
        record["status"] = "audit_fail"
        record["audit_errors"] = errors
        return record
    record["status"] = "ok"
    record["build_signature"] = manifest.get("build_signature")
    record["split_hash"] = manifest.get("split_hash")
    record["template_digest"] = manifest.get("template_digest")
    return record


def audit_raw_dsp_fidelity(
    spec: ProtocolSplitSpec,
    *,
    splits_root: Path,
    output_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    dataset_dir = splits_root / spec.dataset_name
    audit_dir = output_root / spec.dataset_name / "raw_dsp_fidelity"
    metrics_path = audit_dir / "metrics.json"
    record: dict[str, Any] = {
        "stage": "fidelity",
        "protocol_id": spec.protocol_id,
        "dataset_name": spec.dataset_name,
        "metrics_path": str(metrics_path),
    }
    if metrics_path.is_file() and not dry_run:
        payload = load_json(metrics_path)
        record["status"] = "skipped_exists" if payload.get("status") == "passed" else "audit_fail"
        record["fidelity_status"] = payload.get("status")
        return record

    cmd = [
        sys.executable,
        "-m",
        "tv3.pipeline.audit_d2b_frame_fidelity",
        "--dataset-dir",
        str(dataset_dir),
        "--cache-dir",
        str(dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1"),
        "--output-dir",
        str(audit_dir),
    ]
    started = time.perf_counter()
    proc = run_command(cmd, cwd=PROJECT_ROOT, dry_run=dry_run)
    record["elapsed_s"] = round(time.perf_counter() - started, 2)
    record["command"] = cmd
    if dry_run or proc is None:
        record["status"] = "dry_run"
        return record
    if not metrics_path.is_file():
        record["status"] = "fail"
        record["reason"] = "fidelity metrics.json missing"
        return record
    payload = load_json(metrics_path)
    record["fidelity_status"] = payload.get("status")
    record["status"] = "ok" if payload.get("status") == "passed" else "audit_fail"
    return record


def _write_run_config(
    *,
    base_config_path: Path,
    output_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    seed: int | None,
    fidelity_metrics_path: Path | None,
    b1_metrics_path: Path | None,
) -> Path:
    config = load_json(base_config_path)
    config["dataset_dir"] = str(dataset_dir)
    config["output_dir"] = str(output_dir)
    config["overwrite"] = False
    if seed is not None:
        config["seed"] = int(seed)
    if fidelity_metrics_path is not None:
        config["raw_dsp_fidelity_metrics_path"] = str(fidelity_metrics_path)
    if b1_metrics_path is not None:
        config["raw_dsp_reference_metrics_path"] = str(b1_metrics_path)
    # B6 报告仅作历史锚点（plan §3 对照）；路径保留，不作为新 split 通过线
    if "b6_multiseed_report_path" in config and config["b6_multiseed_report_path"]:
        config["b6_multiseed_report_path"] = str(
            (PROJECT_ROOT / config["b6_multiseed_report_path"]).resolve()
            if not Path(config["b6_multiseed_report_path"]).is_absolute()
            else config["b6_multiseed_report_path"]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def run_b1(
    spec: ProtocolSplitSpec,
    *,
    splits_root: Path,
    output_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    dataset_dir = splits_root / spec.dataset_name
    run_dir = output_root / spec.dataset_name / "b1"
    metrics_path = run_dir / "metrics.json"
    record: dict[str, Any] = {
        "stage": "b1",
        "protocol_id": spec.protocol_id,
        "dataset_name": spec.dataset_name,
        "model": "b1_raw_dsp_ridge",
        "training_seed": None,
        "metrics_path": str(metrics_path),
    }
    if metrics_path.is_file() and not dry_run:
        payload = load_json(metrics_path)
        record["status"] = "skipped_exists"
        record["metrics"] = _extract_run_metrics(payload)
        return record

    config_path = run_dir / "run_config.json"
    if not dry_run:
        _write_run_config(
            base_config_path=B1_CONFIG,
            output_path=config_path,
            dataset_dir=dataset_dir,
            output_dir=run_dir,
            seed=None,
            fidelity_metrics_path=None,
            b1_metrics_path=None,
        )
    cmd = [
        sys.executable,
        "-m",
        "tv3.pipeline.run_tv3_rocket_baseline",
        "--config",
        str(config_path),
    ]
    started = time.perf_counter()
    proc = run_command(cmd, cwd=PROJECT_ROOT, dry_run=dry_run)
    record["elapsed_s"] = round(time.perf_counter() - started, 2)
    record["command"] = cmd
    if dry_run or proc is None:
        record["status"] = "dry_run"
        return record
    if proc.returncode != 0 or not metrics_path.is_file():
        record["status"] = "fail"
        record["returncode"] = None if proc is None else proc.returncode
        return record
    payload = load_json(metrics_path)
    record["status"] = "ok"
    record["metrics"] = _extract_run_metrics(payload)
    record["config_sha256"] = _file_sha256(config_path)
    return record


def run_b7_seed(
    spec: ProtocolSplitSpec,
    *,
    training_seed: int,
    splits_root: Path,
    output_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    dataset_dir = splits_root / spec.dataset_name
    run_dir = output_root / spec.dataset_name / f"b7_s{training_seed}"
    metrics_path = run_dir / "metrics.json"
    record: dict[str, Any] = {
        "stage": "b7",
        "protocol_id": spec.protocol_id,
        "dataset_name": spec.dataset_name,
        "model": "b7_oof_ridge_residual_mlp",
        "training_seed": training_seed,
        "metrics_path": str(metrics_path),
        "b7_config_sha256": _file_sha256(B7_CONFIG),
    }
    if metrics_path.is_file() and not dry_run:
        payload = load_json(metrics_path)
        record["status"] = "skipped_exists"
        record["metrics"] = _extract_run_metrics(payload)
        return record

    config_path = run_dir / "run_config.json"
    fidelity_path = output_root / spec.dataset_name / "raw_dsp_fidelity" / "metrics.json"
    b1_metrics = output_root / spec.dataset_name / "b1" / "metrics.json"
    if not dry_run:
        _write_run_config(
            base_config_path=B7_CONFIG,
            output_path=config_path,
            dataset_dir=dataset_dir,
            output_dir=run_dir,
            seed=training_seed,
            fidelity_metrics_path=fidelity_path,
            b1_metrics_path=b1_metrics,
        )
    cmd = [
        sys.executable,
        "-m",
        "tv3.pipeline.run_tv3_rocket_baseline",
        "--config",
        str(config_path),
        "--seed",
        str(training_seed),
    ]
    started = time.perf_counter()
    proc = run_command(cmd, cwd=PROJECT_ROOT, dry_run=dry_run)
    record["elapsed_s"] = round(time.perf_counter() - started, 2)
    record["command"] = cmd
    if dry_run or proc is None:
        record["status"] = "dry_run"
        return record
    if proc.returncode != 0 or not metrics_path.is_file():
        record["status"] = "fail"
        record["returncode"] = None if proc is None else proc.returncode
        return record
    payload = load_json(metrics_path)
    errors = _audit_b7_frozen(payload, training_seed=training_seed)
    record["status"] = "ok" if not errors else "audit_fail"
    record["audit_errors"] = errors
    record["metrics"] = _extract_run_metrics(payload)
    record["config_sha256"] = _file_sha256(config_path)
    return record


def _audit_b7_frozen(payload: dict[str, Any], *, training_seed: int) -> list[str]:
    errors: list[str] = []
    if payload.get("head") != "oof_ridge_residual_mlp":
        errors.append(f"head={payload.get('head')!r}")
    if payload.get("feature_builder") != "d0_raw_dsp_physics_stats_v1":
        errors.append(f"feature_builder={payload.get('feature_builder')!r}")
    if payload.get("feature_count") != 1008:
        errors.append(f"feature_count={payload.get('feature_count')!r}")
    residual = payload.get("diagnostics", {}).get("residual_mlp", {})
    model_config = residual.get("model_config", {})
    if model_config.get("seed") != training_seed:
        errors.append(f"training seed mismatch: {model_config.get('seed')} != {training_seed}")
    if tuple(model_config.get("hidden_dims", residual.get("hidden_dims", []))) not in {(64, 64), ()}:
        hidden = model_config.get("hidden_dims", residual.get("hidden_dims"))
        if hidden != [64, 64] and hidden != (64, 64):
            errors.append(f"hidden_dims not frozen (64,64): {hidden!r}")
    oof = payload.get("diagnostics", {}).get("oof", {})
    if oof.get("fold_count") != 5:
        errors.append(f"oof fold_count={oof.get('fold_count')!r}")
    if oof.get("fold_seed") != 20260711:
        errors.append(f"oof fold_seed={oof.get('fold_seed')!r}")
    # test / extrapolation 永不参与早停：早停字段只允许引用 val
    early = residual.get("early_stopping") or residual.get("selection") or {}
    monitor = str(early.get("monitor", early.get("metric", "val_o2_r2"))).lower()
    if "test" in monitor or "extrap" in monitor:
        errors.append(f"early stopping monitor must be val-only, got {monitor!r}")
    return errors


def _extract_run_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    evaluations = payload.get("evaluations", {})
    out: dict[str, Any] = {"components": {}, "sum_abs_error": {}, "o2_bins": {}}
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
        elif "composition_metrics" in split_payload:
            composition = split_payload["composition_metrics"]
            if isinstance(composition, dict) and "sum_abs_error" in composition:
                out["sum_abs_error"][split_name] = float(composition["sum_abs_error"])
        bins = split_payload.get("o2_bins") or split_payload.get("bin_metrics") or {}
        if isinstance(bins, dict):
            out["o2_bins"][split_name] = bins
    train_o2 = out["components"].get("train", {}).get("x_O2", {}).get("r2")
    val_o2 = out["components"].get("val", {}).get("x_O2", {}).get("r2")
    if train_o2 is not None and val_o2 is not None:
        out["train_val_gap_o2_r2"] = float(train_o2) - float(val_o2)
    return out


def _matrix_row_from_records(
    *,
    spec: ProtocolSplitSpec,
    b1: dict[str, Any] | None,
    b7: dict[str, Any],
) -> dict[str, Any]:
    b7_metrics = b7.get("metrics") or {}
    b1_metrics = (b1 or {}).get("metrics") or {}
    row: dict[str, Any] = {
        "protocol_id": spec.protocol_id,
        "dataset_name": spec.dataset_name,
        "split_seed": spec.split_seed,
        "training_seed": b7.get("training_seed"),
        "is_ood_evidence": spec.is_ood_evidence,
        "b7_status": b7.get("status"),
        "b1_status": None if b1 is None else b1.get("status"),
    }
    for split_name in ("val", "test", "extrapolation"):
        b7_o2 = b7_metrics.get("components", {}).get(split_name, {}).get("x_O2", {})
        b1_o2 = b1_metrics.get("components", {}).get(split_name, {}).get("x_O2", {})
        row[f"b7_{split_name}_o2_r2"] = b7_o2.get("r2")
        row[f"b1_{split_name}_o2_r2"] = b1_o2.get("r2")
        if b7_o2.get("r2") is not None and b1_o2.get("r2") is not None:
            row[f"delta_o2_r2_{split_name}"] = float(b7_o2["r2"]) - float(b1_o2["r2"])
        else:
            row[f"delta_o2_r2_{split_name}"] = None
        for gas in ("x_CO2", "x_N2"):
            row[f"b7_{split_name}_{gas}_r2"] = (
                b7_metrics.get("components", {}).get(split_name, {}).get(gas, {}).get("r2")
            )
        row[f"b7_{split_name}_sum_abs_error"] = b7_metrics.get("sum_abs_error", {}).get(split_name)
    row["train_val_gap_o2_r2"] = b7_metrics.get("train_val_gap_o2_r2")
    return row


def _mean_std(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": round(mean, 6), "std": round(math.sqrt(var), 6), "n": len(values)}


def evaluate_protocol_pass(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """按实施计划 §6.1 判定 protocol_pass（不报告 p 值）。"""
    by_protocol: dict[str, list[dict[str, Any]]] = {pid: [] for pid in PROTOCOL_IDS}
    for row in rows:
        if row.get("b7_status") not in {"ok", "skipped_exists"}:
            continue
        if row.get("delta_o2_r2_test") is None:
            continue
        by_protocol[str(row["protocol_id"])].append(row)

    checks: dict[str, Any] = {}
    all_ok = True
    for protocol_id in PROTOCOL_IDS:
        protocol_rows = by_protocol[protocol_id]
        if not protocol_rows:
            checks[protocol_id] = {"passed": False, "reason": "missing paired rows"}
            all_ok = False
            continue
        test_deltas = [float(row["delta_o2_r2_test"]) for row in protocol_rows]
        test_mean = sum(test_deltas) / len(test_deltas)
        ood_deltas = [float(row["delta_o2_r2_extrapolation"]) for row in protocol_rows]
        ood_mean = sum(ood_deltas) / len(ood_deltas)
        # 同一 split seed 下三 training seed 全负
        by_split_seed: dict[int, list[float]] = {}
        for row in protocol_rows:
            by_split_seed.setdefault(int(row["split_seed"]), []).append(
                float(row["delta_o2_r2_extrapolation"])
            )
        all_negative_split = any(
            len(vals) >= 3 and all(v < 0 for v in vals) for vals in by_split_seed.values()
        )
        gain_on_test_and_ood = test_mean > 0 and ood_mean > 0
        # 非 OOD 协议（R/L）不要求 OOD 证据，但仍报告 extrapolation Δ
        if protocol_id in {"S-Y", "S-L"}:
            passed = test_mean > 0 and ood_mean > 0 and not all_negative_split
        else:
            passed = test_mean > 0
        checks[protocol_id] = {
            "passed": passed,
            "test_delta_o2_r2": _mean_std(test_deltas),
            "extrap_delta_o2_r2": _mean_std(ood_deltas),
            "gain_on_test_and_ood": gain_on_test_and_ood,
            "all_negative_training_seeds_on_some_split": all_negative_split,
        }
        all_ok = all_ok and passed

    return {
        "protocol_pass": all_ok and all(checks[pid]["passed"] for pid in PROTOCOL_IDS),
        "checks": checks,
        "note": (
            "3 training seeds are insufficient for significance claims; "
            "no p-values are reported. Judgment is directional consistency only."
        ),
    }


def write_result_matrix(rows: list[dict[str, Any]], output_root: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    csv_path = output_root / "result_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# B7 Repeated Split × OOD Result Matrix",
        "",
        "| protocol | dataset | split_seed | train_seed | "
        "B7 test O2 R2 | B1 test O2 R2 | Δtest | "
        "B7 OOD O2 R2 | B1 OOD O2 R2 | ΔOOD | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        md_lines.append(
            "| {protocol_id} | {dataset_name} | {split_seed} | {training_seed} | "
            "{b7_test:.4f} | {b1_test:.4f} | {d_test} | "
            "{b7_ood:.4f} | {b1_ood:.4f} | {d_ood} | {status} |".format(
                protocol_id=row["protocol_id"],
                dataset_name=row["dataset_name"],
                split_seed=row["split_seed"],
                training_seed=row["training_seed"],
                b7_test=_fmt(row.get("b7_test_o2_r2")),
                b1_test=_fmt(row.get("b1_test_o2_r2")),
                d_test=_fmt_delta(row.get("delta_o2_r2_test")),
                b7_ood=_fmt(row.get("b7_extrapolation_o2_r2")),
                b1_ood=_fmt(row.get("b1_extrapolation_o2_r2")),
                d_ood=_fmt_delta(row.get("delta_o2_r2_extrapolation")),
                status=row.get("b7_status"),
            )
        )
    (output_root / "result_matrix.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _fmt_delta(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run B7 repeated-split / OOD protocol.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--raw-dsp-bootstrap",
        type=Path,
        default=DEFAULT_RAW_DSP_BOOTSTRAP,
        help="observed_v1 SPXY 仅用于划分的 RawDSP cache；不会硬链接进派生目录。",
    )
    parser.add_argument(
        "--stage",
        choices=("derive", "raw_dsp", "train", "all"),
        default="all",
    )
    parser.add_argument("--protocol-ids", type=str, default=None, help="comma-separated: R,L,S-Y,S-L")
    parser.add_argument("--split-seeds", type=str, default=None, help="comma-separated subset")
    parser.add_argument("--training-seeds", type=str, default=None, help="comma-separated subset")
    parser.add_argument("--workers", type=int, default=None, help="RawDSP builder workers override")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_filter = (
        tuple(item.strip() for item in args.protocol_ids.split(",") if item.strip())
        if args.protocol_ids
        else PROTOCOL_IDS
    )
    split_seeds = (
        tuple(int(item.strip()) for item in args.split_seeds.split(",") if item.strip())
        if args.split_seeds
        else SPLIT_SEEDS
    )
    training_seeds = (
        tuple(int(item.strip()) for item in args.training_seeds.split(",") if item.strip())
        if args.training_seeds
        else TRAINING_SEEDS
    )
    specs = [
        spec
        for spec in build_protocol_matrix()
        if spec.protocol_id in protocol_filter and spec.split_seed in split_seeds
    ]

    print("==== B7 repeated-split / OOD protocol ====")
    print(f"source_dir:   {args.source_dir}")
    print(f"splits_root:  {args.splits_root}")
    print(f"output_root:  {args.output_root}")
    print(f"stage:        {args.stage}")
    print(f"splits:       {len(specs)}")
    print(f"train seeds:  {list(training_seeds)}")
    print(f"dry_run:      {args.dry_run}")

    if not args.dry_run:
        verify_dataset(args.source_dir)
        if any(spec.spxy_x_profile == "observed_v1" for spec in specs):
            if not (args.raw_dsp_bootstrap / "manifest.json").is_file():
                raise FileNotFoundError(
                    f"observed_v1 bootstrap RawDSP missing: {args.raw_dsp_bootstrap}"
                )

    args.output_root.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_root / "runs.jsonl"
    protocol_manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.source_dir),
        "splits_root": str(args.splits_root),
        "output_root": str(args.output_root),
        "b1_config_sha256": _file_sha256(B1_CONFIG),
        "b7_config_sha256": _file_sha256(B7_CONFIG),
        "split_seeds": list(split_seeds),
        "training_seeds": list(training_seeds),
        "protocol_ids": list(protocol_filter),
        "matrix": [asdict(spec) | {"dataset_name": spec.dataset_name} for spec in specs],
        "invariants": {
            "feature_builder": "d0_raw_dsp_physics_stats_v1",
            "feature_count": 1008,
            "head": "oof_ridge_residual_mlp",
            "oof_folds": 5,
            "oof_seed": 20260711,
            "mlp_hidden_dims": [64, 64],
            "early_stopping": "val_only",
            "spxy_x_profile_for_ood": "spxy_observed_stats_v1",
            "skip_source_raw_dsp_hardlink": True,
        },
    }
    _write_json(args.output_root / "protocol_manifest.json", protocol_manifest)

    if args.dry_run:
        for spec in specs:
            print(
                f"[DRY-RUN] {spec.protocol_id} {spec.dataset_name} "
                f"strategy={spec.split_strategy} ood={spec.extrapolation_strategy} "
                f"profile={spec.spxy_x_profile}"
            )
            for seed in training_seeds:
                print(f"  - B1 once; B7 training_seed={seed}")
        print("\n[DRY-RUN] no files written beyond protocol_manifest.json")
        return 0

    b1_by_dataset: dict[str, dict[str, Any]] = {}
    matrix_rows: list[dict[str, Any]] = []
    failures = 0

    for spec in specs:
        if args.stage in {"derive", "all"}:
            derive_record = derive_split(
                spec,
                source_dir=args.source_dir,
                splits_root=args.splits_root,
                raw_dsp_bootstrap=args.raw_dsp_bootstrap,
                dry_run=False,
            )
            _append_jsonl(runs_path, {k: v for k, v in derive_record.items() if k != "command"})
            if derive_record.get("status") not in {"ok", "skipped_exists"}:
                failures += 1
                continue

        if args.stage in {"raw_dsp", "all"}:
            raw_record = build_raw_dsp_for_split(
                spec,
                splits_root=args.splits_root,
                dry_run=False,
                workers=args.workers,
            )
            _append_jsonl(runs_path, {k: v for k, v in raw_record.items() if k != "command"})
            if raw_record.get("status") not in {"ok", "skipped_exists"}:
                failures += 1
                continue
            fidelity_record = audit_raw_dsp_fidelity(
                spec,
                splits_root=args.splits_root,
                output_root=args.output_root,
                dry_run=False,
            )
            _append_jsonl(runs_path, {k: v for k, v in fidelity_record.items() if k != "command"})
            if fidelity_record.get("status") not in {"ok", "skipped_exists"}:
                failures += 1
                continue

        if args.stage in {"train", "all"}:
            b1_record = run_b1(
                spec,
                splits_root=args.splits_root,
                output_root=args.output_root,
                dry_run=False,
            )
            _append_jsonl(runs_path, {k: v for k, v in b1_record.items() if k != "command"})
            b1_by_dataset[spec.dataset_name] = b1_record
            if b1_record.get("status") not in {"ok", "skipped_exists"}:
                failures += 1
                continue
            for training_seed in training_seeds:
                b7_record = run_b7_seed(
                    spec,
                    training_seed=training_seed,
                    splits_root=args.splits_root,
                    output_root=args.output_root,
                    dry_run=False,
                )
                _append_jsonl(runs_path, {k: v for k, v in b7_record.items() if k != "command"})
                if b7_record.get("status") not in {"ok", "skipped_exists"}:
                    failures += 1
                matrix_rows.append(
                    _matrix_row_from_records(spec=spec, b1=b1_record, b7=b7_record)
                )

    if matrix_rows:
        write_result_matrix(matrix_rows, args.output_root)
        verdict = evaluate_protocol_pass(matrix_rows)
        _write_json(args.output_root / "split_metrics.json", {"rows": matrix_rows, "verdict": verdict})
        print(f"\nprotocol_pass: {verdict['protocol_pass']}")
        for protocol_id, check in verdict["checks"].items():
            print(f"  {protocol_id}: passed={check['passed']} detail={check}")

    print(f"\nwritten under: {args.output_root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
