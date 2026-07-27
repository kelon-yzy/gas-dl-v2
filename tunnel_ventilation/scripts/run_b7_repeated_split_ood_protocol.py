"""B7 冻结后的重复 split × 独立 OOD 协议编排。

依据：tunnel_ventilation/docs/archive/completed/b7_repeated_split_ood_protocol_implementation_plan.md

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

from tv3.ml.rocket_training import load_raw_dsp_fidelity, load_raw_dsp_provenance
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
SUCCESS_STATUSES = frozenset({"ok", "revalidated_exists"})
EVALUATION_SPLITS = ("val", "test", "extrapolation")
FROZEN_B7_MLP_CONFIG = {
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


def _is_success_status(status: Any) -> bool:
    return status in SUCCESS_STATUSES


def _same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).resolve() == expected.resolve()


def _source_hashes(source_dir: Path) -> dict[str, str]:
    return {
        "manifest_sha256": _file_sha256(source_dir / "manifest.json"),
        "labels_sha256": _file_sha256(source_dir / "labels" / "y.npy"),
        "condition_grid_sha256": _file_sha256(source_dir / "condition_grid_sequence.csv"),
    }


def _expected_split_policy(spec: ProtocolSplitSpec) -> str:
    if spec.protocol_id == "R":
        return "random_mixture_id_split_v4"
    if spec.protocol_id == "L":
        return "lhs_stratified_split_v1"
    return f"spxy_v1:{spec.extrapolation_strategy}"


def _audit_derived_split(
    spec: ProtocolSplitSpec,
    *,
    source_dir: Path,
    splits_root: Path,
    raw_dsp_bootstrap: Path,
) -> tuple[dict[str, Any], list[str]]:
    dataset_dir = splits_root / spec.dataset_name
    summary_path = dataset_dir / "splits" / "split_summary.json"
    if not summary_path.is_file():
        return {}, [f"split summary missing: {summary_path}"]
    summary = load_json(summary_path)
    errors: list[str] = []
    if summary.get("split_policy") != _expected_split_policy(spec):
        errors.append(f"split_policy={summary.get('split_policy')!r}")
    if summary.get("split_seed") != spec.split_seed:
        errors.append(f"split_seed={summary.get('split_seed')!r}")
    if summary.get("source_hashes") != _source_hashes(source_dir):
        errors.append("source_hashes do not match the current source dataset")
    if "features" not in summary.get("skipped_hardlink_toplevel", []):
        errors.append("derived split did not record features/ as skipped hardlink content")
    for field in ("split_hash", "ood_set_hash"):
        if not isinstance(summary.get(field), str) or not summary[field]:
            errors.append(f"split summary is missing {field}")
    if spec.spxy_x_profile is not None:
        expected_profile = "spxy_observed_stats_v1"
        if summary.get("spxy_x_profile_cli") != spec.spxy_x_profile:
            errors.append(f"spxy_x_profile_cli={summary.get('spxy_x_profile_cli')!r}")
        if summary.get("x_feature_profile") != expected_profile:
            errors.append(f"x_feature_profile={summary.get('x_feature_profile')!r}")
        if summary.get("x_feature_profile_role") != "protocol_default":
            errors.append(f"x_feature_profile_role={summary.get('x_feature_profile_role')!r}")
        if summary.get("spxy_alpha") != spec.spxy_alpha:
            errors.append(f"spxy_alpha={summary.get('spxy_alpha')!r}")
        if summary.get("extrapolation_strategy") != spec.extrapolation_strategy:
            errors.append(f"extrapolation_strategy={summary.get('extrapolation_strategy')!r}")
        bootstrap = summary.get("raw_dsp_bootstrap")
        if not isinstance(bootstrap, dict):
            errors.append("observed SPXY split is missing raw_dsp_bootstrap")
        else:
            if bootstrap.get("role") != "split_selection_bootstrap_only":
                errors.append(f"raw_dsp_bootstrap.role={bootstrap.get('role')!r}")
            if not _same_path(bootstrap.get("cache_dir"), raw_dsp_bootstrap):
                errors.append("raw_dsp_bootstrap.cache_dir does not match the requested bootstrap")
            manifest_path = raw_dsp_bootstrap / "manifest.json"
            if not manifest_path.is_file() or bootstrap.get("manifest_sha256") != _file_sha256(manifest_path):
                errors.append("raw_dsp_bootstrap manifest hash does not match")
    return summary, errors


def _audit_raw_dsp_cache(dataset_dir: Path, cache_dir: Path) -> list[str]:
    manifest_path = cache_dir / "manifest.json"
    summary_path = dataset_dir / "splits" / "split_summary.json"
    if not manifest_path.is_file():
        return [f"RawDSP manifest missing: {manifest_path}"]
    if not summary_path.is_file():
        return [f"split summary missing: {summary_path}"]
    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    errors: list[str] = []
    if manifest.get("template_source_split") != "train":
        errors.append(f"template_source_split={manifest.get('template_source_split')!r}")
    if manifest.get("split_hash") != summary.get("split_hash"):
        errors.append("RawDSP manifest split_hash does not match split_summary")
    if manifest.get("split_policy") != summary.get("split_policy"):
        errors.append("RawDSP manifest split_policy does not match split_summary")
    if manifest.get("split_seed") != summary.get("split_seed"):
        errors.append("RawDSP manifest split_seed does not match split_summary")
    if manifest.get("diagnostic_only") is True:
        errors.append("diagnostic_only cache is not allowed for protocol")
    for field in ("build_signature", "template_digest", "template_source_sequence_ids_digest"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            errors.append(f"RawDSP manifest is missing {field}")
    if manifest.get("complete_dataset") is not True:
        errors.append(f"complete_dataset={manifest.get('complete_dataset')!r}")
    return errors


def _audit_fidelity_metrics(metrics_path: Path, dataset_dir: Path) -> list[str]:
    try:
        provenance = load_raw_dsp_provenance(dataset_dir)
        load_raw_dsp_fidelity(metrics_path, dataset_dir=dataset_dir, provenance=provenance)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return [str(exc)]
    return []


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
        summary, errors = _audit_derived_split(
            spec,
            source_dir=source_dir,
            splits_root=splits_root,
            raw_dsp_bootstrap=raw_dsp_bootstrap,
        )
        record["status"] = "revalidated_exists" if not errors else "audit_fail"
        record["audit_errors"] = errors
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
    summary, errors = _audit_derived_split(
        spec,
        source_dir=source_dir,
        splits_root=splits_root,
        raw_dsp_bootstrap=raw_dsp_bootstrap,
    )
    record["status"] = "ok" if not errors else "audit_fail"
    record["audit_errors"] = errors
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
    cache_existed = manifest_path.is_file()

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
    errors = _audit_raw_dsp_cache(dataset_dir, cache_dir)
    record["status"] = "revalidated_exists" if cache_existed and not errors else "ok"
    if errors:
        record["status"] = "audit_fail"
    record["audit_errors"] = errors
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
        errors = _audit_fidelity_metrics(metrics_path, dataset_dir)
        record["status"] = "revalidated_exists" if not errors else "audit_fail"
        record["audit_errors"] = errors
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
    if proc.returncode != 0 or not metrics_path.is_file():
        record["status"] = "fail"
        record["returncode"] = proc.returncode
        record["reason"] = "fidelity metrics.json missing"
        return record
    payload = load_json(metrics_path)
    record["fidelity_status"] = payload.get("status")
    errors = _audit_fidelity_metrics(metrics_path, dataset_dir)
    record["status"] = "ok" if not errors else "audit_fail"
    record["audit_errors"] = errors
    return record


def _build_run_config_payload(
    *,
    base_config_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    seed: int | None,
    fidelity_metrics_path: Path | None,
    b1_metrics_path: Path | None,
) -> dict[str, Any]:
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
    return config


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
    config = _build_run_config_payload(
        base_config_path=base_config_path,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        seed=seed,
        fidelity_metrics_path=fidelity_metrics_path,
        b1_metrics_path=b1_metrics_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _audit_run_config(config_path: Path, expected_config: dict[str, Any]) -> list[str]:
    if not config_path.is_file():
        return [f"run config missing: {config_path}"]
    actual_config = load_json(config_path)
    if actual_config != expected_config:
        return ["run config does not match the frozen protocol config"]
    return []


def _audit_raw_dsp_run_payload(
    payload: dict[str, Any],
    *,
    dataset_dir: Path,
    expected_head: str,
) -> list[str]:
    errors: list[str] = []
    if payload.get("head") != expected_head:
        errors.append(f"head={payload.get('head')!r}")
    if payload.get("feature_builder") != "d0_raw_dsp_physics_stats_v1":
        errors.append(f"feature_builder={payload.get('feature_builder')!r}")
    if payload.get("feature_count") != 1008:
        errors.append(f"feature_count={payload.get('feature_count')!r}")
    if not _same_path(payload.get("dataset_dir"), dataset_dir):
        errors.append("metrics dataset_dir does not match the derived split")
    try:
        expected_provenance = load_raw_dsp_provenance(dataset_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        errors.append(str(exc))
        return errors
    provenance = payload.get("raw_dsp_provenance")
    if not isinstance(provenance, dict):
        errors.append("metrics are missing raw_dsp_provenance")
    else:
        for field in ("build_signature", "template_digest", "template_source_split"):
            if provenance.get(field) != expected_provenance.get(field):
                errors.append(f"metrics raw_dsp_provenance.{field} does not match the current cache")
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, dict):
        errors.append("metrics are missing evaluations")
        return errors
    for split_name in EVALUATION_SPLITS:
        split_payload = evaluations.get(split_name)
        if not isinstance(split_payload, dict):
            errors.append(f"metrics are missing {split_name} evaluation")
            continue
        components = split_payload.get("component_metrics")
        if not isinstance(components, dict):
            errors.append(f"metrics are missing {split_name}.component_metrics")
            continue
        for component in ("x_CO2", "x_O2", "x_N2"):
            try:
                value = float(components[component]["r2"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"metrics are missing {split_name}.{component}.r2")
                continue
            if not math.isfinite(value):
                errors.append(f"metrics {split_name}.{component}.r2 is not finite")
    return errors


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
    config_path = run_dir / "run_config.json"
    expected_config = _build_run_config_payload(
        base_config_path=B1_CONFIG,
        dataset_dir=dataset_dir,
        output_dir=run_dir,
        seed=None,
        fidelity_metrics_path=None,
        b1_metrics_path=None,
    )
    if metrics_path.is_file() and not dry_run:
        payload = load_json(metrics_path)
        errors = _audit_run_config(config_path, expected_config)
        errors.extend(_audit_raw_dsp_run_payload(payload, dataset_dir=dataset_dir, expected_head="ridgecv"))
        record["status"] = "revalidated_exists" if not errors else "audit_fail"
        record["audit_errors"] = errors
        record["metrics"] = _extract_run_metrics(payload)
        return record

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
    errors = _audit_run_config(config_path, expected_config)
    errors.extend(_audit_raw_dsp_run_payload(payload, dataset_dir=dataset_dir, expected_head="ridgecv"))
    record["status"] = "ok" if not errors else "audit_fail"
    record["audit_errors"] = errors
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
    config_path = run_dir / "run_config.json"
    fidelity_path = output_root / spec.dataset_name / "raw_dsp_fidelity" / "metrics.json"
    b1_metrics = output_root / spec.dataset_name / "b1" / "metrics.json"
    expected_config = _build_run_config_payload(
        base_config_path=B7_CONFIG,
        dataset_dir=dataset_dir,
        output_dir=run_dir,
        seed=training_seed,
        fidelity_metrics_path=fidelity_path,
        b1_metrics_path=b1_metrics,
    )
    if metrics_path.is_file() and not dry_run:
        payload = load_json(metrics_path)
        errors = _audit_run_config(config_path, expected_config)
        errors.extend(_audit_raw_dsp_run_payload(payload, dataset_dir=dataset_dir, expected_head="oof_ridge_residual_mlp"))
        errors.extend(_audit_b7_frozen(payload, training_seed=training_seed))
        record["status"] = "revalidated_exists" if not errors else "audit_fail"
        record["audit_errors"] = errors
        record["metrics"] = _extract_run_metrics(payload)
        return record

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
    errors = _audit_run_config(config_path, expected_config)
    errors.extend(_audit_raw_dsp_run_payload(payload, dataset_dir=dataset_dir, expected_head="oof_ridge_residual_mlp"))
    errors.extend(_audit_b7_frozen(payload, training_seed=training_seed))
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
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return [*errors, "diagnostics is missing or invalid"]
    residual = diagnostics.get("residual_mlp")
    if not isinstance(residual, dict):
        return [*errors, "diagnostics.residual_mlp is missing or invalid"]
    model_config = residual.get("model_config", {})
    if not isinstance(model_config, dict):
        return [*errors, "residual_mlp.model_config is missing or invalid"]
    if model_config.get("seed") != training_seed:
        errors.append(f"training seed mismatch: {model_config.get('seed')} != {training_seed}")
    for field, expected in FROZEN_B7_MLP_CONFIG.items():
        if model_config.get(field) != expected:
            errors.append(f"model_config.{field}={model_config.get(field)!r}, expected {expected!r}")
    if residual.get("standardize_targets") is not True:
        errors.append(f"residual_mlp.standardize_targets={residual.get('standardize_targets')!r}")
    if residual.get("zero_init_output") is not True:
        errors.append(f"residual_mlp.zero_init_output={residual.get('zero_init_output')!r}")
    oof = diagnostics.get("oof")
    if not isinstance(oof, dict):
        errors.append("diagnostics.oof is missing or invalid")
        oof = {}
    if oof.get("fold_count") != 5:
        errors.append(f"oof fold_count={oof.get('fold_count')!r}")
    if oof.get("fold_seed") != 20260711:
        errors.append(f"oof fold_seed={oof.get('fold_seed')!r}")
    if oof.get("coverage_complete") is not True:
        errors.append(f"oof coverage_complete={oof.get('coverage_complete')!r}")
    leakage = diagnostics.get("leakage_audit")
    if not isinstance(leakage, dict):
        errors.append("diagnostics.leakage_audit is missing or invalid")
        leakage = {}
    for field in ("oof_used_for_residual_targets", "full_ridge_fit_on_train_only", "val_residual_from_full_ridge", "oof_coverage_complete"):
        if leakage.get(field) is not True:
            errors.append(f"leakage_audit.{field}={leakage.get(field)!r}")
    early = residual.get("early_stopping")
    if not isinstance(early, dict) or early.get("monitor") != "val_o2_r2":
        monitor = None if not isinstance(early, dict) else early.get("monitor")
        errors.append(f"early stopping monitor must be 'val_o2_r2', got {monitor!r}")
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
    expected_keys = {
        (protocol_id, split_seed, training_seed)
        for protocol_id in PROTOCOL_IDS
        for split_seed in SPLIT_SEEDS
        for training_seed in TRAINING_SEEDS
    }
    rows_by_key: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        try:
            key = (str(row["protocol_id"]), int(row["split_seed"]), int(row["training_seed"]))
        except (KeyError, TypeError, ValueError):
            continue
        rows_by_key.setdefault(key, []).append(row)
    unexpected_keys = set(rows_by_key) - expected_keys

    checks: dict[str, Any] = {}
    all_ok = True
    for protocol_id in PROTOCOL_IDS:
        protocol_expected = {key for key in expected_keys if key[0] == protocol_id}
        missing_keys = sorted(protocol_expected - set(rows_by_key))
        duplicate_keys = sorted(key for key in protocol_expected if len(rows_by_key.get(key, [])) > 1)
        protocol_rows = [rows_by_key[key][0] for key in sorted(protocol_expected) if len(rows_by_key.get(key, [])) == 1]
        failed_rows = []
        for row in protocol_rows:
            if not _is_success_status(row.get("b7_status")) or not _is_success_status(row.get("b1_status")):
                failed_rows.append(row)
                continue
            for field in ("delta_o2_r2_test", "delta_o2_r2_extrapolation"):
                value = row.get(field)
                try:
                    valid_value = value is not None and math.isfinite(float(value))
                except (TypeError, ValueError):
                    valid_value = False
                if not valid_value:
                    failed_rows.append(row)
                    break
        if missing_keys or duplicate_keys or failed_rows:
            checks[protocol_id] = {
                "passed": False,
                "reason": "incomplete, duplicate, failed, or unpaired protocol rows",
                "missing_count": len(missing_keys),
                "duplicate_count": len(duplicate_keys),
                "invalid_count": len(failed_rows),
            }
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
        "protocol_pass": (
            not unexpected_keys
            and all_ok
            and all(checks[pid]["passed"] for pid in PROTOCOL_IDS)
        ),
        "matrix_complete": all(
            len(rows_by_key.get(key, [])) == 1 for key in expected_keys
        ) and not unexpected_keys,
        "unexpected_row_count": sum(len(rows_by_key[key]) for key in unexpected_keys),
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
            "{b7_test} | {b1_test} | {d_test} | "
            "{b7_ood} | {b1_ood} | {d_ood} | {status} |".format(
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
    if not protocol_filter or len(set(protocol_filter)) != len(protocol_filter) or set(protocol_filter) - set(PROTOCOL_IDS):
        raise ValueError(f"protocol_ids must be a non-empty subset of {PROTOCOL_IDS}")
    if not split_seeds or len(set(split_seeds)) != len(split_seeds) or set(split_seeds) - set(SPLIT_SEEDS):
        raise ValueError(f"split_seeds must be a non-empty subset of {SPLIT_SEEDS}")
    if not training_seeds or len(set(training_seeds)) != len(training_seeds) or set(training_seeds) - set(TRAINING_SEEDS):
        raise ValueError(f"training_seeds must be a non-empty subset of {TRAINING_SEEDS}")
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
        "is_complete_formal_matrix": (
            set(protocol_filter) == set(PROTOCOL_IDS)
            and set(split_seeds) == set(SPLIT_SEEDS)
            and set(training_seeds) == set(TRAINING_SEEDS)
        ),
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
            if not _is_success_status(derive_record.get("status")):
                failures += 1
                continue
        else:
            summary_path = args.splits_root / spec.dataset_name / "splits" / "split_summary.json"
            if not summary_path.is_file():
                _append_jsonl(
                    runs_path,
                    {
                        "stage": "derive_prerequisite",
                        "protocol_id": spec.protocol_id,
                        "dataset_name": spec.dataset_name,
                        "status": "audit_fail",
                        "reason": f"split summary missing: {summary_path}",
                    },
                )
                failures += 1
                continue
            derive_record = derive_split(
                spec,
                source_dir=args.source_dir,
                splits_root=args.splits_root,
                raw_dsp_bootstrap=args.raw_dsp_bootstrap,
                dry_run=False,
            )
            _append_jsonl(runs_path, {k: v for k, v in derive_record.items() if k != "command"})
            if not _is_success_status(derive_record.get("status")):
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
            if not _is_success_status(raw_record.get("status")):
                failures += 1
                continue
            fidelity_record = audit_raw_dsp_fidelity(
                spec,
                splits_root=args.splits_root,
                output_root=args.output_root,
                dry_run=False,
            )
            _append_jsonl(runs_path, {k: v for k, v in fidelity_record.items() if k != "command"})
            if not _is_success_status(fidelity_record.get("status")):
                failures += 1
                continue
        elif args.stage == "train":
            dataset_dir = args.splits_root / spec.dataset_name
            cache_manifest = dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1" / "manifest.json"
            if not cache_manifest.is_file():
                _append_jsonl(
                    runs_path,
                    {
                        "stage": "raw_dsp_prerequisite",
                        "protocol_id": spec.protocol_id,
                        "dataset_name": spec.dataset_name,
                        "status": "audit_fail",
                        "reason": f"RawDSP manifest missing: {cache_manifest}",
                    },
                )
                failures += 1
                continue
            raw_record = build_raw_dsp_for_split(
                spec,
                splits_root=args.splits_root,
                dry_run=False,
                workers=args.workers,
            )
            _append_jsonl(runs_path, {k: v for k, v in raw_record.items() if k != "command"})
            if not _is_success_status(raw_record.get("status")):
                failures += 1
                continue
            metrics_path = args.output_root / spec.dataset_name / "raw_dsp_fidelity" / "metrics.json"
            if not metrics_path.is_file():
                _append_jsonl(
                    runs_path,
                    {
                        "stage": "fidelity_prerequisite",
                        "protocol_id": spec.protocol_id,
                        "dataset_name": spec.dataset_name,
                        "status": "audit_fail",
                        "reason": f"fidelity metrics missing: {metrics_path}",
                    },
                )
                failures += 1
                continue
            fidelity_record = audit_raw_dsp_fidelity(
                spec,
                splits_root=args.splits_root,
                output_root=args.output_root,
                dry_run=False,
            )
            _append_jsonl(runs_path, {k: v for k, v in fidelity_record.items() if k != "command"})
            if not _is_success_status(fidelity_record.get("status")):
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
            if not _is_success_status(b1_record.get("status")):
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
                if not _is_success_status(b7_record.get("status")):
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
