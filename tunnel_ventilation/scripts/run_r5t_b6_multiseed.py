"""R5-T 与 B6 三 seed 稳定性复核 — 6 组 MLP 一键编排。

在服务器 tunnel_ventilation 根目录执行：

    python scripts/run_r5t_b6_multiseed.py

前置条件：
  - data/tv3-formal-6000 已存在（数据集在服务器，本脚本不生成数据）
  - outputs/tv3_d2b/raw_dsp_frame_fidelity/metrics.json 已通过（B6 前置）

产出（统一写入 outputs/tv3_r5t_b6_multiseed/）：
  - r5t_s{seed}/metrics.json
  - b6_s{seed}/metrics.json
  - runs.jsonl
  - summary.json
  - replication_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tv3_r5t_b6_multiseed"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "tv3-formal-6000"
RAW_DSP_FIDELITY_PATH = PROJECT_ROOT / "outputs" / "tv3_d2b" / "raw_dsp_frame_fidelity" / "metrics.json"

SEEDS: tuple[int, ...] = (42, 123, 456)

R5T_CONFIG = CONFIG_DIR / "tv3_r5_mlp_target_scaled.json"
B6_CONFIG = CONFIG_DIR / "tv3_d2b_raw_dsp_mlp_target_scaled.json"

RIDGE_BASELINES = {
    "r5t": {
        "label": "D0-observed Ridge",
        "val_o2_r2": 0.4226,
        "test_o2_r2": 0.4571,
        "extrap_o2_r2": 0.3708,
    },
    "b6": {
        "label": "B1 RawDSP Ridge",
        "val_o2_r2": 0.4280,
        "test_o2_r2": 0.4786,
        "extrap_o2_r2": 0.3695,
    },
}

SINGLE_SEED_THRESHOLDS = {
    "r5t": {
        "val_o2_r2": 0.4726,
        "test_o2_r2": 0.5071,
        "extrap_o2_r2": 0.4208,
        "test_strict": False,
        "extrap_strict": False,
    },
    "b6": {
        "val_o2_r2": 0.4780,
        "test_o2_r2": 0.4786,
        "extrap_o2_r2": 0.3695,
        "test_strict": True,
        "extrap_strict": True,
    },
}

PREFLIGHT_TESTS = (
    "tests/test_tv3_r5_mlp.py",
    "tests/test_tv3_raw_dsp_pipeline.py",
    "tests/test_d2b_frame_fidelity_audit.py",
)


@dataclass(frozen=True)
class ExperimentSpec:
    group: str
    label: str
    config_path: Path
    feature_builder: str
    feature_count: int
    seed: int

    @property
    def run_name(self) -> str:
        return f"{self.group}_s{self.seed}"


def build_experiments(output_root: Path) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for seed in SEEDS:
        specs.append(
            ExperimentSpec(
                group="r5t",
                label="R5-T observed MLP",
                config_path=R5T_CONFIG,
                feature_builder="d0_observed_physics_stats_v1",
                feature_count=864,
                seed=seed,
            )
        )
        specs.append(
            ExperimentSpec(
                group="b6",
                label="B6 RawDSP MLP",
                config_path=B6_CONFIG,
                feature_builder="d0_raw_dsp_physics_stats_v1",
                feature_count=1008,
                seed=seed,
            )
        )
    return specs


def _experiment_output_dir(output_root: Path, spec: ExperimentSpec) -> Path:
    return output_root / spec.run_name


def _run_command(cmd: Sequence[str], *, cwd: Path, dry_run: bool) -> subprocess.CompletedProcess[str] | None:
    printable = " ".join(cmd)
    print(f"\n[{_now_hms()}] {printable}", flush=True)
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8")


def _now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_preflight(*, cwd: Path, dry_run: bool, skip: bool) -> None:
    if skip:
        print("[SKIP] preflight checks")
        return

    print("==== preflight ====")
    pytest_cmd = [sys.executable, "-m", "pytest", *PREFLIGHT_TESTS, "-q"]
    proc = _run_command(pytest_cmd, cwd=cwd, dry_run=dry_run)
    if proc is not None and proc.returncode != 0:
        raise RuntimeError(f"preflight pytest failed with exit code {proc.returncode}")

    if not RAW_DSP_FIDELITY_PATH.is_file():
        raise FileNotFoundError(f"RawDSP fidelity metrics missing: {RAW_DSP_FIDELITY_PATH}")

    fidelity = _load_json(RAW_DSP_FIDELITY_PATH)
    if fidelity.get("status") != "passed":
        raise RuntimeError(
            f"RawDSP fidelity status is {fidelity.get('status')!r}, expected 'passed'"
        )
    signature = fidelity.get("source", {}).get("cache_build_signature")
    print(f"[OK] RawDSP fidelity passed; cache_build_signature={signature!r}")


def verify_dataset(dataset_dir: Path) -> None:
    manifest = dataset_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"dataset not found: {dataset_dir}\n"
            "服务器上应已有 data/tv3-formal-6000；可用 DATASET_DIR 覆盖路径。"
        )
    payload = _load_json(manifest)
    sequence_count = payload.get("sequence_count")
    print(f"[OK] dataset {dataset_dir} sequence_count={sequence_count}")


def run_experiment(
    spec: ExperimentSpec,
    *,
    output_root: Path,
    dataset_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    output_dir = _experiment_output_dir(output_root, spec)
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
    print(f"\n==== [{spec.label}] seed={spec.seed} -> {output_dir} ====")
    proc = _run_command(cmd, cwd=PROJECT_ROOT, dry_run=dry_run)
    elapsed_s = time.perf_counter() - started

    record: dict[str, Any] = {
        "group": spec.group,
        "label": spec.label,
        "seed": spec.seed,
        "run_name": spec.run_name,
        "output_dir": str(output_dir),
        "metrics_path": str(metrics_path),
        "elapsed_s": round(elapsed_s, 2),
        "command": cmd,
    }

    if dry_run:
        record["status"] = "dry_run"
        return record

    if proc is None:
        record["status"] = "dry_run"
        return record

    if proc.returncode != 0:
        record["status"] = "fail"
        record["returncode"] = proc.returncode
        record["reason"] = "non-zero exit code"
        if metrics_path.is_file():
            record["metrics_present"] = True
        return record

    if not metrics_path.is_file():
        record["status"] = "fail"
        record["reason"] = "metrics.json missing after successful exit"
        return record

    payload = _load_json(metrics_path)
    audit_errors = audit_metrics(payload, spec)
    record["status"] = "ok" if not audit_errors else "audit_fail"
    record["audit_errors"] = audit_errors
    record["o2_r2"] = _extract_o2_r2(payload)
    record["payload_summary"] = _summarize_payload(payload, spec)
    return record


def _extract_o2_r2(payload: dict[str, Any]) -> dict[str, float]:
    evaluations = payload["evaluations"]
    return {
        split: float(evaluations[split]["component_metrics"]["x_O2"]["r2"])
        for split in ("val", "test", "extrapolation")
    }


def _summarize_payload(payload: dict[str, Any], spec: ExperimentSpec) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics", {})
    model_config = diagnostics.get("model_config", {})
    return {
        "feature_builder": payload.get("feature_builder"),
        "feature_count": payload.get("feature_count"),
        "standardize_targets": model_config.get("standardize_targets"),
        "out_dim": model_config.get("out_dim"),
        "seed": model_config.get("seed"),
        "best_epoch": diagnostics.get("best_epoch"),
        "parameter_count": diagnostics.get("parameter_count"),
        "expected_feature_builder": spec.feature_builder,
        "expected_feature_count": spec.feature_count,
    }


def audit_metrics(payload: dict[str, Any], spec: ExperimentSpec) -> list[str]:
    errors: list[str] = []

    diagnostics = payload.get("diagnostics", {})
    model_config = diagnostics.get("model_config", {})
    if model_config.get("seed") != spec.seed:
        errors.append(f"seed mismatch: {model_config.get('seed')} != {spec.seed}")
    if model_config.get("standardize_targets") is not True:
        errors.append("standardize_targets is not true")
    if model_config.get("out_dim") != 3:
        errors.append(f"out_dim is not 3: {model_config.get('out_dim')}")

    if payload.get("feature_builder") != spec.feature_builder:
        errors.append(
            f"feature_builder mismatch: {payload.get('feature_builder')} != {spec.feature_builder}"
        )
    if payload.get("feature_count") != spec.feature_count:
        errors.append(
            f"feature_count mismatch: {payload.get('feature_count')} != {spec.feature_count}"
        )

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

    if spec.group == "b6":
        fidelity = payload.get("raw_dsp_fidelity", {})
        if fidelity.get("status") != "passed":
            errors.append(f"raw_dsp_fidelity.status is not passed: {fidelity.get('status')}")
        if not payload.get("raw_dsp_provenance"):
            errors.append("raw_dsp_provenance missing")

    return errors


def _compare_threshold(value: float, threshold: float, *, strict: bool) -> bool:
    return value > threshold if strict else value >= threshold


def evaluate_single_seed(group: str, o2_r2: dict[str, float]) -> dict[str, Any]:
    thresholds = SINGLE_SEED_THRESHOLDS[group]
    checks = {
        "val": _compare_threshold(o2_r2["val"], thresholds["val_o2_r2"], strict=False),
        "test": _compare_threshold(
            o2_r2["test"],
            thresholds["test_o2_r2"],
            strict=thresholds["test_strict"],
        ),
        "extrapolation": _compare_threshold(
            o2_r2["extrapolation"],
            thresholds["extrap_o2_r2"],
            strict=thresholds["extrap_strict"],
        ),
    }
    return {
        "thresholds": thresholds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _group_verdict(pass_count: int) -> str:
    if pass_count == 3:
        return "stable_pass"
    if pass_count == 2:
        return "insufficient_evidence"
    return "not_passed"


def build_replication_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[dict[str, Any]]] = {"r5t": [], "b6": []}
    for record in records:
        if record.get("status") != "ok":
            continue
        group = record["group"]
        o2_r2 = record["o2_r2"]
        ridge = RIDGE_BASELINES[group]
        seed_eval = evaluate_single_seed(group, o2_r2)
        by_group[group].append(
            {
                "seed": record["seed"],
                "run_name": record["run_name"],
                "metrics_path": record["metrics_path"],
                "elapsed_s": record["elapsed_s"],
                "o2_r2": o2_r2,
                "ridge_delta": {
                    "val": o2_r2["val"] - ridge["val_o2_r2"],
                    "test": o2_r2["test"] - ridge["test_o2_r2"],
                    "extrapolation": o2_r2["extrapolation"] - ridge["extrap_o2_r2"],
                },
                "seed_evaluation": seed_eval,
            }
        )

    group_summary: dict[str, Any] = {}
    for group, rows in by_group.items():
        rows = sorted(rows, key=lambda item: item["seed"])
        pass_count = sum(1 for row in rows if row["seed_evaluation"]["passed"])
        stats: dict[str, Any] = {}
        for split in ("val", "test", "extrapolation"):
            values = [row["o2_r2"][split] for row in rows]
            if values:
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                stats[split] = {
                    "mean": round(mean, 4),
                    "std": round(math.sqrt(variance), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                }
        group_summary[group] = {
            "label": RIDGE_BASELINES[group]["label"],
            "ridge_baseline": RIDGE_BASELINES[group],
            "completed_seeds": [row["seed"] for row in rows],
            "per_seed": rows,
            "o2_r2_stats": stats,
            "pass_count": pass_count,
            "verdict": _group_verdict(pass_count),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seeds": list(SEEDS),
        "groups": group_summary,
    }


def print_summary_table(report: dict[str, Any]) -> None:
    print("\n==== model × seed × split (O2 R2) ====")
    header = (
        f"{'group':<6} {'seed':>4}  "
        f"{'val':>8}  {'test':>8}  {'extrap':>8}  "
        f"{'Δval':>8}  {'Δtest':>8}  {'Δextrap':>8}  "
        f"{'seed_pass':>9}"
    )
    print(header)
    print("-" * len(header))

    for group in ("r5t", "b6"):
        group_data = report["groups"].get(group, {})
        for row in group_data.get("per_seed", []):
            o2 = row["o2_r2"]
            delta = row["ridge_delta"]
            passed = "yes" if row["seed_evaluation"]["passed"] else "no"
            print(
                f"{group:<6} {row['seed']:>4}  "
                f"{o2['val']:8.4f}  {o2['test']:8.4f}  {o2['extrapolation']:8.4f}  "
                f"{delta['val']:+8.4f}  {delta['test']:+8.4f}  {delta['extrapolation']:+8.4f}  "
                f"{passed:>9}"
            )

        stats = group_data.get("o2_r2_stats", {})
        if stats:
            print(
                f"{group:<6} {'mean':>4}  "
                f"{stats['val']['mean']:8.4f}  {stats['test']['mean']:8.4f}  "
                f"{stats['extrapolation']['mean']:8.4f}  "
                f"{'':>8}  {'':>8}  {'':>8}  "
                f"{group_data['verdict']:>9}"
            )
            print(
                f"{group:<6} {'std':>4}  "
                f"{stats['val']['std']:8.4f}  {stats['test']['std']:8.4f}  "
                f"{stats['extrapolation']['std']:8.4f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run R5-T and B6 multiseed MLP replication (6 runs).")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="tv3 dataset root (default: data/tv3-formal-6000)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="unified output directory (default: outputs/tv3_r5t_b6_multiseed)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing training")
    parser.add_argument("--skip-preflight", action="store_true", help="skip pytest and RawDSP fidelity checks")
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="comma-separated subset: r5t,b6 (default: both)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="comma-separated subset of seeds (default: 42,123,456)",
    )
    return parser


def _parse_subset(value: str | None, allowed: set[str], label: str) -> set[str] | None:
    if value is None:
        return None
    selected = {item.strip() for item in value.split(",") if item.strip()}
    unknown = selected - allowed
    if unknown:
        raise ValueError(f"unknown {label}: {sorted(unknown)}")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root: Path = args.output_root
    dataset_dir: Path = args.dataset_dir

    selected_groups = _parse_subset(args.groups, {"r5t", "b6"}, "groups")
    if args.seeds:
        seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    else:
        seeds = SEEDS

    specs = [
        spec
        for spec in build_experiments(output_root)
        if spec.seed in seeds and (selected_groups is None or spec.group in selected_groups)
    ]

    print("==== R5-T / B6 multiseed replication ====")
    print(f"project_root: {PROJECT_ROOT}")
    print(f"dataset_dir:  {dataset_dir}")
    print(f"output_root:  {output_root}")
    print(f"runs:         {len(specs)}")

    if not args.dry_run:
        verify_dataset(dataset_dir)
    run_preflight(cwd=PROJECT_ROOT, dry_run=args.dry_run, skip=args.skip_preflight)

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for spec in specs:
        record = run_experiment(spec, output_root=output_root, dataset_dir=dataset_dir, dry_run=args.dry_run)
        records.append(record)

    if args.dry_run:
        print("\n[DRY-RUN] no files written")
        return 0

    runs_path = output_root / "runs.jsonl"
    runs_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )

    ok_records = [record for record in records if record.get("status") == "ok"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "records": [
            {
                key: value
                for key, value in record.items()
                if key not in {"command", "payload_summary"}
            }
            for record in records
        ],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = build_replication_report(ok_records)
    (output_root / "replication_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print_summary_table(report)

    failed = [record for record in records if record.get("status") != "ok"]
    print(f"\nwritten: {output_root}")
    print(f"  runs.jsonl")
    print(f"  summary.json")
    print(f"  replication_report.json")
    if failed:
        print(f"\n{len(failed)} run(s) failed or failed audit:")
        for record in failed:
            print(f"  - {record['run_name']}: {record.get('status')} ({record.get('reason', record.get('audit_errors'))})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
