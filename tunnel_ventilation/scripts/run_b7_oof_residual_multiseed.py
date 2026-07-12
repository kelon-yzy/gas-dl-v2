"""B7 OOF Ridge residual MLP 三 seed 正式复核编排。

在服务器 tunnel_ventilation 根目录执行：

    python scripts/run_b7_oof_residual_multiseed.py

前置条件：
  - data/tv3-formal-6000 已存在
  - outputs/tv3_d2b/raw_dsp_frame_fidelity/metrics.json 已通过
  - outputs/tv3_d2b/raw_dsp_ridge_provenance/metrics.json 存在
  - outputs/tv3_r5t_b6_multiseed/replication_report.json 为 B6 stable_pass

产出（写入 outputs/tv3_d2b/b7_oof_ridge_residual_mlp/）：
  - s{seed}/metrics.json
  - summary.json
  - replication_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from tv3.pipeline.multiseed_utils import (
    evaluate_o2_single_seed,
    extract_o2_r2,
    load_json,
    run_command,
    verify_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tv3_d2b" / "b7_oof_ridge_residual_mlp"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "tv3-formal-6000"
RAW_DSP_FIDELITY_PATH = PROJECT_ROOT / "outputs" / "tv3_d2b" / "raw_dsp_frame_fidelity" / "metrics.json"
B6_REPORT_PATH = PROJECT_ROOT / "outputs" / "tv3_r5t_b6_multiseed" / "replication_report.json"
B7_CONFIG = CONFIG_DIR / "tv3_d2b_oof_ridge_residual_mlp.json"

SEEDS: tuple[int, ...] = (42, 123, 456)
B1_THRESHOLDS = {
    "val_o2_r2": 0.4780,
    "test_o2_r2": 0.4786,
    "extrap_o2_r2": 0.3695,
    "test_strict": True,
    "extrap_strict": True,
}
PREFLIGHT_TESTS = (
    "tests/test_tv3_b7_oof_residual.py",
    "tests/test_tv3_r5_mlp.py",
    "tests/test_tv3_raw_dsp_pipeline.py",
    "tests/test_d2b_frame_fidelity_audit.py",
)


@dataclass(frozen=True)
class B7ExperimentSpec:
    seed: int
    config_path: Path = B7_CONFIG
    feature_builder: str = "d0_raw_dsp_physics_stats_v1"
    feature_count: int = 1008

    @property
    def run_name(self) -> str:
        return f"s{self.seed}"


def run_preflight(*, cwd: Path, dry_run: bool, skip: bool) -> None:
    if skip:
        print("[SKIP] preflight checks")
        return

    print("==== preflight ====")
    pytest_cmd = [sys.executable, "-m", "pytest", *PREFLIGHT_TESTS, "-q"]
    proc = run_command(pytest_cmd, cwd=cwd, dry_run=dry_run)
    if proc is not None and proc.returncode != 0:
        raise RuntimeError(f"preflight pytest failed with exit code {proc.returncode}")

    if not RAW_DSP_FIDELITY_PATH.is_file():
        raise FileNotFoundError(f"RawDSP fidelity metrics missing: {RAW_DSP_FIDELITY_PATH}")
    fidelity = load_json(RAW_DSP_FIDELITY_PATH)
    if fidelity.get("status") != "passed":
        raise RuntimeError(
            f"RawDSP fidelity status is {fidelity.get('status')!r}, expected 'passed'"
        )
    if not B6_REPORT_PATH.is_file():
        raise FileNotFoundError(f"B6 multiseed report missing: {B6_REPORT_PATH}")
    b6_report = load_json(B6_REPORT_PATH)
    verdict = b6_report.get("groups", {}).get("b6", {}).get("verdict")
    if verdict != "stable_pass":
        raise RuntimeError(f"B6 multiseed verdict is {verdict!r}, expected 'stable_pass'")
    print(f"[OK] RawDSP fidelity passed; B6 report verdict={verdict!r}")


def audit_b7_metrics(payload: dict[str, Any], spec: B7ExperimentSpec) -> list[str]:
    errors: list[str] = []
    diagnostics = payload.get("diagnostics", {})
    residual = diagnostics.get("residual_mlp", {})
    model_config = residual.get("model_config", {})

    if model_config.get("seed") != spec.seed:
        errors.append(f"seed mismatch: {model_config.get('seed')} != {spec.seed}")
    if residual.get("standardize_targets") is not True:
        errors.append("residual standardize_targets is not true")
    if residual.get("zero_init_output") is not True:
        errors.append("residual zero_init_output is not true")
    if payload.get("feature_builder") != spec.feature_builder:
        errors.append(
            f"feature_builder mismatch: {payload.get('feature_builder')} != {spec.feature_builder}"
        )
    if payload.get("feature_count") != spec.feature_count:
        errors.append(
            f"feature_count mismatch: {payload.get('feature_count')} != {spec.feature_count}"
        )
    if payload.get("head") != "oof_ridge_residual_mlp":
        errors.append(f"head mismatch: {payload.get('head')}")

    oof = diagnostics.get("oof", {})
    if oof.get("fold_count") != 5:
        errors.append(f"oof fold_count is not 5: {oof.get('fold_count')}")
    if oof.get("fold_seed") != 20260711:
        errors.append(f"oof fold_seed mismatch: {oof.get('fold_seed')}")
    if oof.get("coverage_complete") is not True:
        errors.append("oof coverage_complete is not true")

    for split in ("train", "val", "test", "extrapolation"):
        if split not in payload.get("evaluations", {}):
            errors.append(f"missing evaluations[{split}]")
            continue
        comp_metrics = payload["evaluations"][split].get("component_metrics", {})
        for comp in ("x_CO2", "x_O2", "x_N2"):
            metrics = comp_metrics.get(comp, {})
            for key in ("r2", "mae", "rmse"):
                if key not in metrics:
                    errors.append(f"missing {split}/{comp}/{key}")

    fidelity = payload.get("raw_dsp_fidelity", {})
    if fidelity.get("status") != "passed":
        errors.append(f"raw_dsp_fidelity.status is not passed: {fidelity.get('status')}")
    if not payload.get("raw_dsp_provenance"):
        errors.append("raw_dsp_provenance missing")
    if not diagnostics.get("b1_reference", {}).get("metrics_sha256"):
        errors.append("b1_reference metrics_sha256 missing")
    if not diagnostics.get("b6_reference", {}).get("report_sha256"):
        errors.append("b6_reference report_sha256 missing")
    return errors


def evaluate_b7_seed(o2_r2: dict[str, float]) -> dict[str, Any]:
    return evaluate_o2_single_seed(o2_r2, thresholds=B1_THRESHOLDS)


def load_b6_seed_baseline(report_path: Path) -> dict[str, Any]:
    report = load_json(report_path)
    b6 = report.get("groups", {}).get("b6")
    if not isinstance(b6, dict) or b6.get("verdict") != "stable_pass":
        raise ValueError("B6 multiseed report must contain groups.b6 verdict='stable_pass'")
    rows = b6.get("per_seed")
    if not isinstance(rows, list):
        raise ValueError("B6 multiseed report is missing groups.b6.per_seed")

    by_seed: dict[int, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("B6 multiseed per_seed entries must be objects")
        seed = int(row["seed"])
        o2_r2 = row.get("o2_r2")
        if seed in by_seed or not isinstance(o2_r2, dict):
            raise ValueError("B6 multiseed report has duplicate or malformed per_seed entries")
        by_seed[seed] = {
            split: float(o2_r2[split])
            for split in ("val", "test", "extrapolation")
        }
    if set(by_seed) != set(SEEDS):
        raise ValueError(f"B6 multiseed seeds must be {list(SEEDS)}, got {sorted(by_seed)}")

    stats = b6.get("o2_r2_stats")
    if not isinstance(stats, dict):
        raise ValueError("B6 multiseed report is missing groups.b6.o2_r2_stats")
    return {
        "report_path": str(report_path),
        "report_sha256": _file_sha256(report_path),
        "per_seed": by_seed,
        "std": {
            split: float(stats[split]["std"])
            for split in ("test", "extrapolation")
        },
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_verdict(
    per_seed: list[dict[str, Any]],
    stats: dict[str, Any],
    *,
    b6_std: dict[str, float],
) -> str:
    if {row["seed"] for row in per_seed} != set(SEEDS):
        return "partial"
    if any(not row["seed_evaluation"]["passed"] for row in per_seed):
        return "failed"

    mean_deltas = {
        split: sum(row["b6_paired_delta"][split] for row in per_seed) / len(per_seed)
        for split in ("test", "extrapolation")
    }
    if any(delta < -0.01 for delta in mean_deltas.values()):
        return "failed"

    non_negative = all(delta >= 0.0 for delta in mean_deltas.values())
    uplift = any(delta >= 0.01 for delta in mean_deltas.values())
    both_higher_std = all(stats[split]["std"] > b6_std[split] for split in ("test", "extrapolation"))
    if non_negative and uplift and not both_higher_std:
        return "residual_pass"
    return "noninferior_only"


def run_experiment(
    spec: B7ExperimentSpec,
    *,
    output_root: Path,
    dataset_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    output_dir = output_root / spec.run_name
    metrics_path = output_dir / "metrics.json"
    cmd = [
        sys.executable,
        "-m",
        "tv3.pipeline.run_tv3_rocket_baseline",
        "--config",
        str(spec.config_path),
        "--dataset-dir",
        str(dataset_dir),
        "--seed",
        str(spec.seed),
        "--output-dir",
        str(output_dir),
    ]
    started = time.perf_counter()
    print(f"\n==== [B7 OOF Ridge Residual] seed={spec.seed} -> {output_dir} ====")
    proc = run_command(cmd, cwd=PROJECT_ROOT, dry_run=dry_run)
    elapsed_s = time.perf_counter() - started
    record: dict[str, Any] = {
        "group": "b7",
        "label": "B7 OOF Ridge Residual MLP",
        "seed": spec.seed,
        "run_name": spec.run_name,
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "elapsed_s": round(elapsed_s, 2),
        "command": cmd,
    }
    if dry_run or proc is None:
        record["status"] = "dry_run"
        return record
    if proc.returncode != 0:
        record["status"] = "fail"
        record["returncode"] = proc.returncode
        record["reason"] = "non-zero exit code"
        return record
    if not metrics_path.is_file():
        record["status"] = "fail"
        record["reason"] = "metrics.json missing after successful exit"
        return record

    payload = load_json(metrics_path)
    audit_errors = audit_b7_metrics(payload, spec)
    record["status"] = "ok" if not audit_errors else "audit_fail"
    record["audit_errors"] = audit_errors
    record["o2_r2"] = extract_o2_r2(payload)
    return record


def build_replication_report(
    records: list[dict[str, Any]],
    *,
    b6_baseline: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "ok":
            continue
        o2_r2 = record["o2_r2"]
        seed = int(record["seed"])
        b6_o2_r2 = b6_baseline["per_seed"].get(seed)
        if b6_o2_r2 is None:
            raise ValueError(f"B6 multiseed baseline is missing seed {seed}")
        seed_eval = evaluate_b7_seed(o2_r2)
        rows.append(
            {
                "seed": seed,
                "run_name": record["run_name"],
                "metrics_path": record["metrics_path"],
                "elapsed_s": record["elapsed_s"],
                "o2_r2": o2_r2,
                "b6_paired_delta": {
                    "test": o2_r2["test"] - b6_o2_r2["test"],
                    "extrapolation": o2_r2["extrapolation"] - b6_o2_r2["extrapolation"],
                },
                "seed_evaluation": seed_eval,
            }
        )
    rows = sorted(rows, key=lambda item: item["seed"])
    stats: dict[str, Any] = {}
    for split in ("val", "test", "extrapolation"):
        values = [row["o2_r2"][split] for row in rows]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stats[split] = {
            "mean": round(mean, 4),
            "std": round(math.sqrt(variance), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    verdict = _paired_verdict(rows, stats, b6_std=b6_baseline["std"]) if rows else "partial"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "b1_thresholds": B1_THRESHOLDS,
        "expected_seeds": list(SEEDS),
        "completed_seeds": [row["seed"] for row in rows],
        "b6_reference": {
            "report_path": b6_baseline["report_path"],
            "report_sha256": b6_baseline["report_sha256"],
        },
        "per_seed": rows,
        "o2_r2_stats": stats,
        "pass_count": sum(1 for row in rows if row["seed_evaluation"]["passed"]),
        "verdict": verdict,
    }


def print_summary_table(report: dict[str, Any]) -> None:
    print("\n==== B7 seed × split (O2 R2) ====")
    header = (
        f"{'seed':>4}  {'val':>8}  {'test':>8}  {'extrap':>8}  "
        f"{'Δtest_B6':>10}  {'Δextrap_B6':>11}  {'seed_pass':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in report.get("per_seed", []):
        o2 = row["o2_r2"]
        delta = row["b6_paired_delta"]
        passed = "yes" if row["seed_evaluation"]["passed"] else "no"
        print(
            f"{row['seed']:>4}  "
            f"{o2['val']:8.4f}  {o2['test']:8.4f}  {o2['extrapolation']:8.4f}  "
            f"{delta['test']:+10.4f}  {delta['extrapolation']:+11.4f}  "
            f"{passed:>9}"
        )
    stats = report.get("o2_r2_stats", {})
    if stats:
        print(
            f"{'mean':>4}  "
            f"{stats['val']['mean']:8.4f}  {stats['test']['mean']:8.4f}  "
            f"{stats['extrapolation']['mean']:8.4f}  "
            f"{'':>10}  {'':>11}  {report['verdict']:>9}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run B7 OOF Ridge residual MLP multiseed replication.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated subset of seeds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root: Path = args.output_root
    dataset_dir: Path = args.dataset_dir
    if args.seeds:
        seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    else:
        seeds = SEEDS
    specs = [B7ExperimentSpec(seed=seed) for seed in seeds]

    print("==== B7 OOF Ridge residual multiseed ====")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"dataset_dir:  {dataset_dir}")
    print(f"output_root:  {output_root}")
    print(f"runs:         {len(specs)}")

    if not args.dry_run:
        verify_dataset(dataset_dir)
    run_preflight(cwd=PROJECT_ROOT, dry_run=args.dry_run, skip=args.skip_preflight)
    b6_baseline = load_b6_seed_baseline(B6_REPORT_PATH)

    if output_root.exists() and any(output_root.iterdir()) and not args.dry_run:
        # Formal B7 root must not silently overwrite prior seed metrics.
        existing_metrics = list(output_root.glob("s*/metrics.json"))
        if existing_metrics:
            raise FileExistsError(
                f"refusing to overwrite existing B7 outputs under {output_root}; "
                "choose a new output-root or remove completed seed dirs explicitly"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for spec in specs:
        records.append(
            run_experiment(spec, output_root=output_root, dataset_dir=dataset_dir, dry_run=args.dry_run)
        )

    if args.dry_run:
        print("\n[DRY-RUN] no files written")
        return 0

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "records": [
            {key: value for key, value in record.items() if key != "command"}
            for record in records
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = build_replication_report(records, b6_baseline=b6_baseline)
    (output_root / "replication_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_summary_table(report)

    failed = [record for record in records if record.get("status") != "ok"]
    print(f"\nwritten: {output_root}")
    print("  summary.json")
    print("  replication_report.json")
    if failed:
        print(f"\n{len(failed)} run(s) failed or failed audit:")
        for record in failed:
            print(
                f"  - {record['run_name']}: {record.get('status')} "
                f"({record.get('reason', record.get('audit_errors'))})"
            )
        return 1
    print(f"\nverdict: {report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
