"""阶段 Ⅰ-4：tv3-formal 基线训练编排。

5 配置 × 3 seeds = 15 runs。每个 run 写到 outputs/tv3_baseline/{model}/seed{seed}/。
最终在 outputs/tv3_baseline/summary.json 汇总 per-component metrics。

用法：
    python scripts/run_tv3_baseline.py [--epochs N] [--dry-run]

前置依赖：data/tv3-formal 已生成（见 experiment_roadmap.md Ⅰ-2）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "tv3"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tv3_baseline"

DL_MODELS = ("cnn1d", "tcn", "lstm", "patchtst")
ML_MODELS = ("ridge",)
SEEDS = (42, 123, 456)


def _config_path(model: str) -> Path:
    name = {
        "cnn1d": "tv3_baseline.json",
        "tcn": "tv3_tcn.json",
        "lstm": "tv3_lstm.json",
        "patchtst": "tv3_patchtst.json",
        "ridge": "tv3_ridge.json",
    }[model]
    return CONFIG_DIR / name


def _run_dl(model: str, seed: int, epochs: int | None, dry_run: bool) -> dict:
    output_dir = OUTPUT_ROOT / model / f"seed{seed}"
    cmd = [
        sys.executable, "-m", "dl.cli",
        "--config", str(_config_path(model)),
        "--output-dir", str(output_dir),
        "--seed", str(seed),
    ]
    if epochs is not None:
        cmd.extend(["--epochs", str(epochs)])
    print(f"\n[{datetime.now():%H:%M:%S}] running {model} seed={seed}\n  {' '.join(cmd)}", flush=True)
    if dry_run:
        return {"model": model, "seed": seed, "skipped": True}
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env={**_env(), "PYTHONPATH": str(PROJECT_ROOT / "src")})
    metrics_path = output_dir / "metrics.json"
    if proc.returncode != 0:
        result = {
            "model": model,
            "seed": seed,
            "status": "fail",
            "reason": "non-zero exit code",
            "returncode": proc.returncode,
        }
        if metrics_path.is_file():
            print(
                f"  [error] {model} seed={seed}: non-zero exit code {proc.returncode}; metrics.json kept for diagnostics",
                flush=True,
            )
            result["metrics_path"] = str(metrics_path)
            result["payload"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        return result
    if not metrics_path.is_file():
        return {"model": model, "seed": seed, "status": "fail", "reason": "no metrics.json"}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {"model": model, "seed": seed, "status": "ok", "metrics_path": str(metrics_path), "payload": payload}


def _run_ml(model: str, seed: int, dry_run: bool) -> dict:
    """Ridge 不依赖 seed（closed-form），但仍为每个 seed 写一份记录便于对齐结构。"""
    output_dir = OUTPUT_ROOT / model / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics.json"
    cmd = [
        sys.executable, "-m", "ml.cli",
        "--config", str(_config_path(model)),
        "--json",
    ]
    print(f"\n[{datetime.now():%H:%M:%S}] running {model} seed={seed} (closed-form, seed only used for record)\n  {' '.join(cmd)}", flush=True)
    if dry_run:
        return {"model": model, "seed": seed, "skipped": True}
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                          env={**_env(), "PYTHONPATH": str(PROJECT_ROOT / "src")})
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return {"model": model, "seed": seed, "status": "fail", "returncode": proc.returncode}
    out_path.write_text(proc.stdout, encoding="utf-8")
    payload = json.loads(proc.stdout)
    return {"model": model, "seed": seed, "status": "ok", "metrics_path": str(out_path), "payload": payload}


def _env() -> dict:
    import os
    return {k: v for k, v in os.environ.items()}


def _summarize(records: list[dict]) -> dict:
    """Pivot per-run records into a flat per-model / per-component summary."""
    summary: dict[str, dict] = {}
    for rec in records:
        if rec.get("status") != "ok":
            continue
        model = rec["model"]
        seed = rec["seed"]
        payload = rec["payload"]
        evaluations = payload.get("evaluations") or payload.get("splits") or {}
        if isinstance(evaluations, list):
            split_map = {item["split"]: item for item in evaluations}
        else:
            split_map = evaluations
        summary.setdefault(model, {})[f"seed{seed}"] = {
            split: {
                "metrics": data.get("metrics"),
                "component_metrics": data.get("component_metrics"),
            }
            for split, data in split_map.items()
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run tv3 baseline matrix.")
    parser.add_argument("--epochs", type=int, default=None, help="Override DL epochs (default: config)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated subset of models to run.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated subset of seeds.")
    args = parser.parse_args()

    if args.models:
        wanted = tuple(m.strip() for m in args.models.split(",") if m.strip())
    else:
        wanted = DL_MODELS + ML_MODELS
    if args.seeds:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    else:
        seeds = SEEDS

    records: list[dict] = []
    for seed in seeds:
        for model in wanted:
            if model in ML_MODELS:
                records.append(_run_ml(model, seed, dry_run=args.dry_run))
            else:
                records.append(_run_dl(model, seed, args.epochs, dry_run=args.dry_run))

    if args.dry_run:
        return 0
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Partial reruns 应保留其他模型已有记录，不覆盖 summary.json / runs.jsonl
    runs_path = OUTPUT_ROOT / "runs.jsonl"
    existing_records: dict[tuple[str, int], dict] = {}
    if runs_path.is_file():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            existing_records[(rec.get("model"), rec.get("seed"))] = rec
    for rec in records:
        existing_records[(rec["model"], rec["seed"])] = rec
    merged = list(existing_records.values())
    runs_path.write_text(
        "\n".join(
            json.dumps({k: v for k, v in rec.items() if k != "payload"}, ensure_ascii=False)
            for rec in merged
        ),
        encoding="utf-8",
    )
    # summary 只汇总 runs.jsonl 中 status=ok 的记录，避免失败 run 的旧 metrics.json 混入。
    summary = _summarize_from_disk(merged)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUTPUT_ROOT / 'summary.json'}")
    fail = [r for r in merged if r.get("status") == "fail"]
    if fail:
        print(f"{len(fail)} runs failed:")
        for r in fail:
            print(f"  - {r['model']} seed={r['seed']} -> {r}")
        return 1
    return 0


def _summarize_from_disk(records: list[dict]) -> dict:
    """Rebuild summary from successful run records and their metrics.json files."""
    summary: dict[str, dict] = {}
    for rec in records:
        if rec.get("status") != "ok":
            continue
        model = rec["model"]
        seed = rec["seed"]
        metrics_value = rec.get("metrics_path")
        metrics_path = Path(metrics_value) if metrics_value else OUTPUT_ROOT / model / f"seed{seed}" / "metrics.json"
        if not metrics_path.is_file():
            continue
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data.get("evaluations"), dict):
            split_map = {
                k: {"metrics": v.get("metrics"), "component_metrics": v.get("component_metrics")}
                for k, v in data["evaluations"].items()
            }
        else:
            splits = data.get("splits") or []
            split_map = {
                s["split"]: {"metrics": s.get("metrics"), "component_metrics": s.get("component_metrics")}
                for s in splits if isinstance(s, dict)
            }
        summary.setdefault(model, {})[f"seed{seed}"] = split_map
    return summary

if __name__ == "__main__":
    raise SystemExit(main())
