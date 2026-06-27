"""阶段 Ⅰ-3：sg4-formal 基线训练编排。

5 配置 × 3 seeds = 15 runs。每个 run 写到 outputs/sg4_baseline/{model}/seed{seed}/。
最终在 outputs/sg4_baseline/summary.json 汇总 per-component metrics。

用法：
    python scripts/run_sg4_baseline.py [--epochs N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "sg4"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sg4_baseline"

DL_MODELS = ("cnn1d", "tcn", "lstm", "patchtst")
ML_MODELS = ("ridge",)
SEEDS = (42, 123, 2026)


def _config_path(model: str) -> Path:
    name = {
        "cnn1d": "sg4_baseline.json",
        "tcn": "sg4_tcn.json",
        "lstm": "sg4_lstm.json",
        "patchtst": "sg4_patchtst.json",
        "ridge": "sg4_ridge.json",
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
    # Windows: PyTorch LSTM 退出阶段 cuDNN 资源回收会触发 STATUS_STACK_BUFFER_OVERRUN (0xC0000409 = 3221226505)，
    # 训练已完成、metrics.json 已写出，仅退出码异常。优先按 metrics.json 是否存在判定成功，避免误报。
    if proc.returncode != 0:
        if metrics_path.is_file():
            warning = f"non-zero exit code {proc.returncode} but metrics.json present (likely Windows cuDNN teardown)"
            print(f"  [warning] {model} seed={seed}: {warning}", flush=True)
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            return {
                "model": model, "seed": seed,
                "status": "ok", "warning": warning,
                "returncode": proc.returncode,
                "metrics_path": str(metrics_path), "payload": payload,
            }
        return {"model": model, "seed": seed, "status": "fail", "returncode": proc.returncode}
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
        # dl payload structure: {"evaluations": [{"split":"val", "metrics":{...}, "component_metrics":{...}}, ...]}
        # ml payload structure: {"splits": [{"split":"val", ...}]}
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
    parser = argparse.ArgumentParser(description="Run sg4 baseline matrix.")
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
    # Partial reruns（如 --models patchtst）应保留其他模型已有记录，不覆盖 summary.json / runs.jsonl
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
    # summary 同样合并：直接重读所有 metrics.json，避免增量误差
    summary = _summarize_from_disk()
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


def _summarize_from_disk() -> dict:
    """重新从磁盘扫描所有 metrics.json 重建 summary，确保部分重跑后数据完整。"""
    summary: dict[str, dict] = {}
    for model in DL_MODELS + ML_MODELS:
        for seed in SEEDS:
            metrics_path = OUTPUT_ROOT / model / f"seed{seed}" / "metrics.json"
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
