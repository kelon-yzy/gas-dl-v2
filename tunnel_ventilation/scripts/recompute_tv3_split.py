"""从已有 tv3 数据集重算 split，复用物理数据（硬链接大文件）。

避免为每种 split 策略重新跑物理仿真。source 数据集的 slow/ultrasonic/labels
等全部硬链接到 output 目录（同 inode，不占额外磁盘），仅 splits/ 用新策略重算。

用法：
    python scripts/recompute_tv3_split.py \
        --source-dir data/tv3-formal-6000 \
        --output-dir data/tv3-formal-6000-spxy \
        --split-strategy spxy_v1 --spxy-alpha 0.5 \
        --extrapolation-strategy y_margin_ood
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tv3.sim.core.tunnel_ventilation_schema import SPLIT_FIELDS, SPLIT_NAMES
from tv3.sim.packaging.io import write_csv, write_json
from tv3.sim.packaging.splits import build_default_split_rows
from tv3.sim.packaging.spxy_split import (
    SpxySplitError,
    build_lhs_stratified_split_with_summary,
    build_spxy_split_with_summary,
)

_VALID_SPLIT_STRATEGIES = ("random", "spxy_v1", "lhs_stratified_split_v1")
_VALID_EXTRAPOLATION_STRATEGIES = ("none", "y_margin_ood", "lhs_boundary", "kmeans_boundary")
# SPXY 特征构建（方案 A）只需这 4 个数组，不加载 29 GB 的超声波形
_SPXY_ARRAY_KEYS = (
    "slow",
    "ultrasonic_tof_s",
    "ultrasonic_sound_speed_m_per_s",
    "ultrasonic_alpha_true_npm",
)


def _load_conditions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_spxy_arrays(sequences_dir: Path) -> dict[str, np.ndarray]:
    return {key: np.load(sequences_dir / f"{key}.npy", mmap_mode="r") for key in _SPXY_ARRAY_KEYS}


def _link_tree(src: Path, dst: Path, *, skip: frozenset[str] = frozenset()) -> None:
    """递归硬链接 src 到 dst；skip 中的顶层条目跳过。要求同文件系统。"""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in skip:
            continue
        target = dst / item.name
        if item.is_dir():
            _link_tree(item, target)
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        os.link(item, target)


def recompute_split(
    source_dir: Path,
    output_dir: Path,
    *,
    split_strategy: str,
    spxy_alpha: float = 0.5,
    extrapolation_strategy: str = "none",
    seed: int = 20260704,
) -> dict[str, object]:
    """从 source 数据集重算 split，硬链接物理数据到 output，写新 splits/。"""
    if split_strategy not in _VALID_SPLIT_STRATEGIES:
        raise ValueError(f"split_strategy must be one of {_VALID_SPLIT_STRATEGIES}")
    if split_strategy == "spxy_v1" and extrapolation_strategy == "none":
        raise ValueError("split_strategy='spxy_v1' 要求 extrapolation_strategy 非 none")
    if split_strategy != "spxy_v1" and extrapolation_strategy != "none":
        raise ValueError(f"extrapolation_strategy={extrapolation_strategy!r} 仅在 spxy_v1 下有效")

    conditions = _load_conditions(source_dir / "condition_grid_sequence.csv")
    labels = np.load(source_dir / "labels" / "y.npy")

    if split_strategy == "spxy_v1":
        arrays = _load_spxy_arrays(source_dir / "sequences")
        rows, summary = build_spxy_split_with_summary(
            conditions,
            arrays,
            labels,
            seed=seed,
            alpha=spxy_alpha,
            extrapolation_strategy=extrapolation_strategy,
        )
    elif split_strategy == "lhs_stratified_split_v1":
        rows, summary = build_lhs_stratified_split_with_summary(conditions, labels, seed=seed)
    else:
        rows = build_default_split_rows(conditions, seed=seed)
        summary = {"split_policy": "random_mixture_id_split_v4", "group_field": "mixture_id"}

    output_dir.mkdir(parents=True, exist_ok=True)
    _link_tree(source_dir, output_dir, skip=frozenset({"splits"}))
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for name in SPLIT_NAMES:
        write_csv(splits_dir / f"{name}.csv", SPLIT_FIELDS, rows[name])
    # 统一补 splits 计数字段，与 benchmark.py _split_summary 对齐
    summary["splits"] = {
        name: {
            "sequence_count": len(rows[name]),
            "mixture_count": len({r["mixture_id"] for r in rows[name]}),
        }
        for name in SPLIT_NAMES
    }
    summary.setdefault("group_field", "mixture_id")
    write_json(splits_dir / "split_summary.json", summary)
    return {"split_policy": summary["split_policy"], "splits": {n: len(rows[n]) for n in SPLIT_NAMES}}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute tv3 dataset split without re-running physics simulation."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-strategy", choices=_VALID_SPLIT_STRATEGIES, required=True)
    parser.add_argument("--spxy-alpha", type=float, default=0.5)
    parser.add_argument("--extrapolation-strategy", choices=_VALID_EXTRAPOLATION_STRATEGIES, default="none")
    parser.add_argument("--seed", type=int, default=20260704)
    args = parser.parse_args()

    if not args.source_dir.is_dir() or not (args.source_dir / "condition_grid_sequence.csv").is_file():
        parser.error(f"--source-dir 必须是已生成的 tv3 benchmark 目录: {args.source_dir}")

    try:
        info = recompute_split(
            args.source_dir,
            args.output_dir,
            split_strategy=args.split_strategy,
            spxy_alpha=args.spxy_alpha,
            extrapolation_strategy=args.extrapolation_strategy,
            seed=args.seed,
        )
    except (SpxySplitError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    print(f"[recompute-split] {info['split_policy']} -> {args.output_dir}")
    for name in SPLIT_NAMES:
        print(f"  {name}: {info['splits'][name]} sequences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
