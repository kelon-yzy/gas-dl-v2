"""阶段 Ⅱ ablation 训练编排。

三组 ablation：
- co_channel: B 组(去 V_NDIR_CO) / C 组(仅 CO 光学+环境) × (TCN + Ridge) × 3 seeds。
              A 组(全通道)复用 outputs/sg4_baseline/{tcn,ridge}。
- loss:       TCN × {mse, mae, huber, smooth_l1} × 3 seeds。
              weighted_component_mse 复用 outputs/sg4_baseline/tcn。
- crosstalk:  TCN × 3 seeds，数据集 data/sg4-formal-crosstalk（需先用
              --enable-co-crosstalk 生成）。对比基线 outputs/sg4_baseline/tcn。

产物：outputs/sg4_ablation/{experiment}/{tag}/seed{seed}/metrics.json
      outputs/sg4_ablation/summary.json（扫描全部已存在 metrics.json，部分重跑不丢历史）

用法：
    python scripts/run_sg4_ablation.py --experiment all
    python scripts/run_sg4_ablation.py --experiment co_channel --epochs 3 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "sg4" / "ablation"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "sg4_ablation"
SEEDS = (42, 123, 2026)

# experiment -> ((tag, kind, config_relpath), ...)；kind ∈ {"dl", "ml"}
EXPERIMENTS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "co_channel": (
        ("tcn_dropco", "dl", "co_channel/tcn_dropco.json"),
        ("tcn_coonly", "dl", "co_channel/tcn_coonly.json"),
        ("ridge_dropco", "ml", "co_channel/ridge_dropco.json"),
        ("ridge_coonly", "ml", "co_channel/ridge_coonly.json"),
    ),
    "loss": (
        ("tcn_mse", "dl", "loss/tcn_mse.json"),
        ("tcn_mae", "dl", "loss/tcn_mae.json"),
        ("tcn_huber", "dl", "loss/tcn_huber.json"),
        ("tcn_smoothl1", "dl", "loss/tcn_smoothl1.json"),
    ),
    "crosstalk": (
        ("tcn_crosstalk", "dl", "crosstalk/tcn_crosstalk.json"),
    ),
}


def _env() -> dict:
    return {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}


def _run_dl(config_path: Path, output_dir: Path, seed: int, epochs: int | None, dry_run: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "dl.cli",
        "--config", str(config_path),
        "--output-dir", str(output_dir),
        "--seed", str(seed),
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    print(f"\n[{datetime.now():%H:%M:%S}] DL {output_dir.parent.name}/seed{seed}\n  {' '.join(cmd)}", flush=True)
    if dry_run:
        return {"kind": "dl", "skipped": True}
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=_env())
    metrics_path = output_dir / "metrics.json"
    # Windows: PyTorch 退出阶段 cuDNN 资源回收偶发 STATUS_STACK_BUFFER_OVERRUN
    # (0xC0000409=3221226505)，训练已完成、metrics.json 已写出，仅退出码异常。
    if proc.returncode != 0:
        if metrics_path.is_file():
            warning = f"non-zero exit {proc.returncode} but metrics.json present (likely Windows cuDNN teardown)"
            print(f"  [warning] {warning}", flush=True)
            return {"kind": "dl", "status": "ok", "warning": warning, "returncode": proc.returncode,
                    "metrics_path": str(metrics_path)}
        return {"kind": "dl", "status": "fail", "returncode": proc.returncode}
    if not metrics_path.is_file():
        return {"kind": "dl", "status": "fail", "reason": "no metrics.json"}
    return {"kind": "dl", "status": "ok", "metrics_path": str(metrics_path)}


def _run_ml(config_path: Path, output_dir: Path, dry_run: bool) -> dict:
    """Ridge closed-form，不依赖 seed；为对齐结构每个 seed 写一份相同结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics.json"
    cmd = [sys.executable, "-m", "ml.cli", "--config", str(config_path), "--json"]
    print(f"\n[{datetime.now():%H:%M:%S}] ML {output_dir.parent.name}/{output_dir.name} (closed-form)\n  {' '.join(cmd)}", flush=True)
    if dry_run:
        return {"kind": "ml", "skipped": True}
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", env=_env())
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return {"kind": "ml", "status": "fail", "returncode": proc.returncode}
    out_path.write_text(proc.stdout, encoding="utf-8")
    return {"kind": "ml", "status": "ok", "metrics_path": str(out_path)}


def _extract_metrics(metrics_path: Path) -> dict:
    """提取 val/test split 的 metrics + component_metrics（dl 与 ml 结构一致）。"""
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    evaluations = data.get("evaluations") or {}
    if isinstance(evaluations, list):
        evaluations = {item["split"]: item for item in evaluations}
    out: dict = {}
    for split in ("val", "test"):
        ev = evaluations.get(split)
        if not isinstance(ev, dict):
            continue
        out[split] = {
            "metrics": ev.get("metrics"),
            "component_metrics": ev.get("component_metrics"),
        }
    return out


def _summarize_from_disk() -> dict:
    """扫描全部 EXPERIMENTS 的已存在 metrics.json，部分重跑也保持 summary 完整。"""
    summary: dict = {}
    for exp, runs in EXPERIMENTS.items():
        for tag, _kind, _rel in runs:
            for seed in SEEDS:
                metrics_path = OUTPUT_ROOT / exp / tag / f"seed{seed}" / "metrics.json"
                if not metrics_path.is_file():
                    continue
                try:
                    summary.setdefault(exp, {}).setdefault(tag, {})[f"seed{seed}"] = _extract_metrics(metrics_path)
                except json.JSONDecodeError:
                    continue
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sg4 ablation experiments.")
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "all"), default="all")
    parser.add_argument("--epochs", type=int, default=None, help="Override DL epochs (default: config).")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds subset.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip()) if args.seeds else SEEDS
    selected = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]

    records: list[dict] = []
    for exp in selected:
        for tag, kind, rel in EXPERIMENTS[exp]:
            config_path = CONFIG_DIR / rel
            for seed in seeds:
                output_dir = OUTPUT_ROOT / exp / tag / f"seed{seed}"
                if kind == "dl":
                    rec = _run_dl(config_path, output_dir, seed, args.epochs, args.dry_run)
                else:
                    rec = _run_ml(config_path, output_dir, args.dry_run)
                rec.update(experiment=exp, tag=tag, seed=seed)
                records.append(rec)

    if args.dry_run:
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = _summarize_from_disk()
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {OUTPUT_ROOT / 'summary.json'}")

    fails = [r for r in records if r.get("status") == "fail"]
    if fails:
        print(f"{len(fails)} runs failed:")
        for r in fails:
            print(f"  - {r['experiment']}/{r['tag']} seed={r['seed']} -> {r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
