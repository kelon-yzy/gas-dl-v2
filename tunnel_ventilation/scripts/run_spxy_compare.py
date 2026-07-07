"""SPXY 对比矩阵：多模态 DL 训练编排。

在 tv3-formal-6000 的多个 split 变体上训练 cnn1d_tcn_fusion（slow+ultrasonic），
对比 random / lhs_stratified / spxy_v1 等划分策略对 ID test 与 extrapolation 指标的影响。

前置依赖：
- data/tv3-formal-6000 及其 split 变体已生成（见 scripts/recompute_tv3_split.py）
- 数据集跳过了 fiber_mic，DL 侧固定 --modalities slow,ultrasonic

用法：
    python scripts/run_spxy_compare.py                         # 默认 A1-A4, seed=42
    python scripts/run_spxy_compare.py --seeds 42,123,456      # 多 seed 量化方差
    python scripts/run_spxy_compare.py --datasets A1,A3 --epochs 10   # 快速验证
    python scripts/run_spxy_compare.py --ridge                 # 额外跑 Ridge 线性对照
    python scripts/run_spxy_compare.py --dry-run               # 只打印命令

输出：
- outputs/spxy_compare/{experiment}_s{seed}/metrics.json  （DL 每个 run）
- outputs/spxy_compare/{experiment}_ridge/metrics.json    （Ridge 每实验一次）
- outputs/spxy_compare/runs.jsonl                          （所有 run 状态，支持 partial rerun）
- outputs/spxy_compare/summary.json                        （per-model per-experiment 指标汇总）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment" / "tv3" / "tv3_tcn_multimodal.json"
RIDGE_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment" / "tv3" / "tv3_ridge.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "spxy_compare"

# 实验名 -> 数据集目录
DATASETS: dict[str, str] = {
    "A1_random": "data/tv3-formal-6000",
    "A2_lhs": "data/tv3-formal-6000-lhs",
    "A3_spxy": "data/tv3-formal-6000-spxy",
    "A4_spxy_a03": "data/tv3-formal-6000-spxy-a03",
    "A5_spxy_lhsbound": "data/tv3-formal-6000-spxy-lhsbound",  # 需先用 lhs_boundary 重算生成
}
DEFAULT_DATASETS = ("A1_random", "A2_lhs", "A3_spxy", "A4_spxy_a03")
DEFAULT_MODALITIES = "slow,ultrasonic"  # 数据集跳过了 fiber_mic，必须排除
DEFAULT_EVAL_SPLITS = "val,test,extrapolation"  # extrapolation 只记录外推指标，不参与早停


def _env() -> dict:
    import os
    return {k: v for k, v in os.environ.items()}


def _run_dl(name: str, dataset_dir: Path, seed: int, args: argparse.Namespace) -> dict:
    """运行一个 DL run（cnn1d_tcn_fusion 多模态）。"""
    output_dir = OUTPUT_ROOT / f"{name}_s{seed}"
    cmd = [
        sys.executable, "-m", "dl.cli",
        "--config", str(CONFIG_PATH),
        "--dataset-dir", str(dataset_dir),
        "--modalities", args.modalities,
        "--batch-size", str(args.batch_size),
        "--eval-splits", args.eval_splits,
        "--seed", str(seed),
        "--output-dir", str(output_dir),
    ]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    print(f"\n[{datetime.now():%H:%M:%S}] running {name} seed={seed}\n  {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return {"experiment": name, "seed": seed, "model": "tcn_multimodal", "skipped": True}

    proc = subprocess.run(
        cmd, cwd=PROJECT_ROOT, env={**_env(), "PYTHONPATH": str(PROJECT_ROOT / "src")}
    )
    metrics_path = output_dir / "metrics.json"
    if proc.returncode != 0:
        result: dict = {
            "experiment": name, "seed": seed, "model": "tcn_multimodal",
            "status": "fail", "reason": "non-zero exit code", "returncode": proc.returncode,
        }
        if metrics_path.is_file():
            print(f"  [error] {name} seed={seed}: exit {proc.returncode}; metrics.json kept", flush=True)
            result["metrics_path"] = str(metrics_path)
            result["payload"] = json.loads(metrics_path.read_text(encoding="utf-8"))
        return result
    if not metrics_path.is_file():
        return {"experiment": name, "seed": seed, "model": "tcn_multimodal",
                "status": "fail", "reason": "no metrics.json"}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {"experiment": name, "seed": seed, "model": "tcn_multimodal",
            "status": "ok", "metrics_path": str(metrics_path), "payload": payload}


def _run_ridge(name: str, dataset_dir: Path, args: argparse.Namespace) -> dict:
    """运行 Ridge 线性对照（closed-form，不依赖 seed，每实验一次）。"""
    output_dir = OUTPUT_ROOT / f"{name}_ridge"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "metrics.json"
    cmd = [
        sys.executable, "-m", "ml.cli",
        "--config", str(RIDGE_CONFIG_PATH),
        "--dataset-dir", str(dataset_dir),
        "--json",
    ]
    print(f"\n[{datetime.now():%H:%M:%S}] running {name} ridge\n  {' '.join(cmd)}", flush=True)
    if args.dry_run:
        return {"experiment": name, "model": "ridge", "skipped": True}

    proc = subprocess.run(
        cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
        env={**_env(), "PYTHONPATH": str(PROJECT_ROOT / "src")},
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return {"experiment": name, "model": "ridge", "status": "fail", "returncode": proc.returncode}
    out_path.write_text(proc.stdout, encoding="utf-8")
    payload = json.loads(proc.stdout)
    return {"experiment": name, "model": "ridge",
            "status": "ok", "metrics_path": str(out_path), "payload": payload}


def _extract_split_metrics(payload: dict) -> dict:
    """从 dl.cli/ml.cli 的 metrics.json 提取 per-split 指标摘要。"""
    evaluations = payload.get("evaluations") or payload.get("splits") or {}
    if isinstance(evaluations, list):
        evaluations = {item["split"]: item for item in evaluations}
    out: dict[str, object] = {}
    for split, data in evaluations.items():
        if not isinstance(data, dict):
            continue
        out[split] = {
            "metrics": data.get("metrics"),
            "component_metrics": data.get("component_metrics"),
        }
    return out


def _summarize(records: list[dict]) -> dict:
    """按 model -> experiment 汇总成功 run 的 per-split 指标。"""
    summary: dict[str, dict] = {}
    for rec in records:
        if rec.get("status") != "ok":
            continue
        model = rec.get("model", "?")
        exp = rec.get("experiment", "?")
        seed = rec.get("seed")
        exp_key = exp if seed is None else f"{exp}_s{seed}"
        summary.setdefault(model, {})[exp_key] = _extract_split_metrics(rec.get("payload", {}))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SPXY comparison matrix on tv3-formal-6000 splits.")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS),
                        help=f"逗号分隔的实验名，可选: {list(DATASETS.keys())}")
    parser.add_argument("--seeds", type=str, default="42", help="逗号分隔的 seed 列表")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 DL epochs（默认 config 50）")
    parser.add_argument("--batch-size", type=int, default=16, help="DL batch size（默认 16）")
    parser.add_argument("--modalities", type=str, default=DEFAULT_MODALITIES,
                        help=f"DL 模态组合（默认 {DEFAULT_MODALITIES}，数据集无 fiber_mic）")
    parser.add_argument("--eval-splits", type=str, default=DEFAULT_EVAL_SPLITS,
                        help=f"评估 split（默认 {DEFAULT_EVAL_SPLITS}）")
    parser.add_argument("--ridge", action="store_true", help="额外跑 Ridge 线性对照（每数据集 1 次）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = parser.parse_args()

    wanted = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    for name in wanted:
        if name not in DATASETS:
            parser.error(f"未知实验名 {name!r}，可选: {list(DATASETS.keys())}")
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    records: list[dict] = []
    for seed in seeds:
        for name in wanted:
            dataset_dir = PROJECT_ROOT / DATASETS[name]
            if not args.dry_run and not dataset_dir.is_dir():
                print(f"[warn] 跳过 {name}：数据集目录 {dataset_dir} 不存在", flush=True)
                records.append({"experiment": name, "seed": seed, "model": "tcn_multimodal",
                                "status": "skip", "reason": f"dataset dir missing: {dataset_dir}"})
                continue
            records.append(_run_dl(name, dataset_dir, seed, args))

    if args.ridge:
        for name in wanted:
            dataset_dir = PROJECT_ROOT / DATASETS[name]
            if not args.dry_run and not dataset_dir.is_dir():
                continue
            records.append(_run_ridge(name, dataset_dir, args))

    if args.dry_run:
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # partial rerun：合并已有 runs.jsonl 记录
    runs_path = OUTPUT_ROOT / "runs.jsonl"
    existing: dict[tuple, dict] = {}
    if runs_path.is_file():
        for line in runs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("experiment"), rec.get("seed"), rec.get("model"))
            existing[key] = rec
    for rec in records:
        key = (rec.get("experiment"), rec.get("seed"), rec.get("model"))
        existing[key] = rec
    merged = list(existing.values())
    runs_path.write_text(
        "\n".join(
            json.dumps({k: v for k, v in rec.items() if k != "payload"}, ensure_ascii=False)
            for rec in merged
        ),
        encoding="utf-8",
    )
    summary = _summarize(merged)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUTPUT_ROOT / 'summary.json'}")

    fail = [r for r in merged if r.get("status") == "fail"]
    if fail:
        print(f"{len(fail)} runs failed:")
        for r in fail:
            print(f"  - {r.get('experiment')} seed={r.get('seed')} model={r.get('model')} -> {r}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
