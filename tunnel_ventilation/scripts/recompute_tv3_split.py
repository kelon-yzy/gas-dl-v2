"""从已有 tv3 数据集重算 split，复用物理数据（硬链接大文件）。

避免为每种 split 策略重新跑物理仿真。source 数据集的 slow/ultrasonic/labels
等全部硬链接到 output 目录（同 inode，不占额外磁盘），仅 splits/ 用新策略重算。

B7 协议约束：
- 默认跳过 source 的 `features/`（尤其 `features/raw_dsp/`），不得把旧 random-train
  RawDSP cache 硬链接进派生目录。
- `--spxy-x-profile observed_v1` 为 B7 正式 OOD 协议默认 profile；需显式提供
  RawDSP cache 路径（仅用于 SPXY X 距离，不作模型特征）。
- `--spxy-x-profile oracle_v1` 保留旧含 true alpha 行为，summary 标记为
  `oracle_split_sensitivity`，不得作为 B7 正式 OOD 结论依据。

用法：
    python scripts/recompute_tv3_split.py \\
        --source-dir data/tv3-formal-6000 \\
        --output-dir data/tv3-formal-6000-splits/spxy_observed_a05_ymargin_s20260704 \\
        --split-strategy spxy_v1 --spxy-alpha 0.5 \\
        --extrapolation-strategy y_margin_ood \\
        --spxy-x-profile observed_v1 \\
        --raw-dsp-cache-dir data/tv3-formal-6000/features/raw_dsp/raw_dsp_frame_v1 \\
        --seed 20260704
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tv3.sim.core.tunnel_ventilation_schema import SPLIT_FIELDS, SPLIT_NAMES
from tv3.sim.packaging.io import write_csv, write_json
from tv3.sim.packaging.splits import build_default_split_rows
from tv3.sim.packaging.spxy_split import (
    SPXY_X_PROFILE_OBSERVED_V1,
    SPXY_X_PROFILE_ORACLE_V1,
    VALID_SPXY_X_PROFILES,
    SpxySplitError,
    build_lhs_stratified_split_with_summary,
    build_spxy_split_with_summary,
    hash_sequence_id_set,
)

_VALID_SPLIT_STRATEGIES = ("random", "spxy_v1", "lhs_stratified_split_v1")
_VALID_EXTRAPOLATION_STRATEGIES = ("none", "y_margin_ood", "lhs_boundary", "kmeans_boundary")

# oracle_v1：SPXY 特征只需这 4 个仿真数组，不加载超声波形
_ORACLE_SPXY_ARRAY_KEYS = (
    "slow",
    "ultrasonic_tof_s",
    "ultrasonic_sound_speed_m_per_s",
    "ultrasonic_alpha_true_npm",
)

# observed_v1：slow + RawDSP frame 输出（不含 true/oracle 物理量）
_OBSERVED_RAW_DSP_KEYS = (
    "ultrasonic_tof_observed_raw_dsp_s",
    "ultrasonic_peak_index_raw_dsp",
    "ultrasonic_sound_speed_raw_dsp_m_per_s",
    "ultrasonic_corr_peak",
    "ultrasonic_snr_db",
    "ultrasonic_raw_dsp_quality",
    "ultrasonic_raw_dsp_accepted",
)

# 派生目录默认跳过：split 依赖的 RawDSP cache 必须重建，不得硬链接
_DEFAULT_SKIP_TOPLEVEL = frozenset({"splits", "features"})


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_conditions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_oracle_spxy_arrays(sequences_dir: Path) -> dict[str, np.ndarray]:
    return {key: np.load(sequences_dir / f"{key}.npy", mmap_mode="r") for key in _ORACLE_SPXY_ARRAY_KEYS}


def _load_observed_spxy_arrays(
    sequences_dir: Path,
    raw_dsp_cache_dir: Path,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "slow": np.load(sequences_dir / "slow.npy", mmap_mode="r"),
    }
    for key in _OBSERVED_RAW_DSP_KEYS:
        path = raw_dsp_cache_dir / f"{key}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"observed_v1 需要 RawDSP 数组: {path}；"
                "请传入 --raw-dsp-cache-dir，或先构建 source 的 RawDSP cache 仅作 SPXY bootstrap"
            )
        arrays[key] = np.load(path, mmap_mode="r")
    return arrays


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
        try:
            os.link(item, target)
        except OSError as exc:
            raise RuntimeError(
                f"硬链接失败: {item} -> {target} ({exc})；"
                "派生 split 要求与 source 同文件系统"
            ) from exc


def _condition_field_stats(
    conditions: list[dict[str, str]],
    rows: list[dict[str, str]],
    field: str,
) -> dict[str, float] | None:
    by_seq = {str(row["sequence_id"]): row for row in conditions}
    values = []
    for row in rows:
        cond = by_seq[row["sequence_id"]]
        if field not in cond or cond[field] == "":
            return None
        values.append(float(cond[field]))
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _enrich_summary_ranges(
    summary: dict[str, Any],
    *,
    conditions: list[dict[str, str]],
    labels: np.ndarray,
    rows: dict[str, list[dict[str, str]]],
) -> None:
    seq_to_idx = {str(c["sequence_id"]): i for i, c in enumerate(conditions)}
    label_fields = ("x_CO2", "x_O2", "x_N2")
    condition_fields = ("L_m", "T_C", "H_RH")
    ranges: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        split_rows = rows[split_name]
        entry: dict[str, Any] = {}
        if split_rows:
            indices = [seq_to_idx[r["sequence_id"]] for r in split_rows]
            for col, name in enumerate(label_fields):
                if labels.shape[1] <= col:
                    continue
                vals = np.asarray(labels[indices, col], dtype=np.float64)
                entry[name] = {
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "mean": float(vals.mean()),
                    "p10": float(np.quantile(vals, 0.10)),
                    "p50": float(np.quantile(vals, 0.50)),
                    "p90": float(np.quantile(vals, 0.90)),
                }
            for field in condition_fields:
                stats = _condition_field_stats(conditions, split_rows, field)
                if stats is not None:
                    entry[field] = stats
        ranges[split_name] = entry
    summary["component_condition_ranges"] = ranges


def _compute_split_hash(rows: dict[str, list[dict[str, str]]]) -> str:
    parts: list[str] = []
    for name in SPLIT_NAMES:
        ids = ",".join(r["sequence_id"] for r in rows[name])
        parts.append(f"{name}:{ids}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _assert_split_invariants(rows: dict[str, list[dict[str, str]]], *, expected_n: int) -> None:
    all_ids = [r["sequence_id"] for name in SPLIT_NAMES for r in rows[name]]
    if len(all_ids) != expected_n:
        raise SpxySplitError(f"split 总数不守恒: {len(all_ids)} != {expected_n}")
    if len(set(all_ids)) != expected_n:
        raise SpxySplitError("split 集合存在 ID overlap")


def recompute_split(
    source_dir: Path,
    output_dir: Path,
    *,
    split_strategy: str,
    spxy_alpha: float = 0.5,
    extrapolation_strategy: str = "none",
    seed: int = 20260704,
    spxy_x_profile: str = SPXY_X_PROFILE_ORACLE_V1,
    raw_dsp_cache_dir: Path | None = None,
    skip_toplevel: frozenset[str] = _DEFAULT_SKIP_TOPLEVEL,
) -> dict[str, object]:
    """从 source 数据集重算 split，硬链接物理数据到 output，写新 splits/。"""
    if split_strategy not in _VALID_SPLIT_STRATEGIES:
        raise ValueError(f"split_strategy must be one of {_VALID_SPLIT_STRATEGIES}")
    if split_strategy == "spxy_v1" and extrapolation_strategy == "none":
        raise ValueError("split_strategy='spxy_v1' 要求 extrapolation_strategy 非 none")
    if split_strategy != "spxy_v1" and extrapolation_strategy != "none":
        raise ValueError(f"extrapolation_strategy={extrapolation_strategy!r} 仅在 spxy_v1 下有效")
    if spxy_x_profile not in VALID_SPXY_X_PROFILES:
        raise ValueError(f"spxy_x_profile must be one of {list(VALID_SPXY_X_PROFILES)}")
    if split_strategy != "spxy_v1" and spxy_x_profile != SPXY_X_PROFILE_ORACLE_V1:
        raise ValueError("--spxy-x-profile 仅在 split_strategy=spxy_v1 下有效")

    conditions = _load_conditions(source_dir / "condition_grid_sequence.csv")
    labels = np.load(source_dir / "labels" / "y.npy")
    source_hashes = {
        "manifest_sha256": _file_sha256(source_dir / "manifest.json"),
        "labels_sha256": _file_sha256(source_dir / "labels" / "y.npy"),
        "condition_grid_sha256": _file_sha256(source_dir / "condition_grid_sequence.csv"),
    }

    raw_dsp_bootstrap_meta: dict[str, Any] | None = None
    if split_strategy == "spxy_v1":
        if spxy_x_profile == SPXY_X_PROFILE_OBSERVED_V1:
            if raw_dsp_cache_dir is None:
                raise ValueError(
                    "spxy_x_profile=observed_v1 要求 --raw-dsp-cache-dir；"
                    "该 cache 仅用于 SPXY X 划分，派生目录仍会跳过 features/ 并重建 RawDSP"
                )
            raw_dsp_cache_dir = raw_dsp_cache_dir.resolve()
            arrays = _load_observed_spxy_arrays(source_dir / "sequences", raw_dsp_cache_dir)
            manifest_path = raw_dsp_cache_dir / "manifest.json"
            raw_dsp_bootstrap_meta = {
                "role": "split_selection_bootstrap_only",
                "cache_dir": str(raw_dsp_cache_dir),
                "manifest_sha256": _file_sha256(manifest_path) if manifest_path.is_file() else None,
                "note": (
                    "observed RawDSP 数组仅参与 SPXY X 距离；"
                    "模型特征必须使用派生 split 重建的 train-calibrated cache"
                ),
            }
        else:
            arrays = _load_oracle_spxy_arrays(source_dir / "sequences")
        rows, summary = build_spxy_split_with_summary(
            conditions,
            arrays,
            labels,
            seed=seed,
            alpha=spxy_alpha,
            extrapolation_strategy=extrapolation_strategy,
            x_profile=spxy_x_profile,
        )
    elif split_strategy == "lhs_stratified_split_v1":
        rows, summary = build_lhs_stratified_split_with_summary(conditions, labels, seed=seed)
        summary["ood_set_hash"] = hash_sequence_id_set(
            [r["sequence_id"] for r in rows["extrapolation"]]
        )
    else:
        rows = build_default_split_rows(conditions, seed=seed)
        summary = {
            "split_policy": "random_mixture_id_split_v4",
            "group_field": "mixture_id",
            "ood_set_hash": hash_sequence_id_set(
                [r["sequence_id"] for r in rows["extrapolation"]]
            ),
            "extrapolation_note": "random remainder; not physical OOD",
        }

    _assert_split_invariants(rows, expected_n=len(conditions))
    summary["split_seed"] = int(seed)
    summary["source_dataset"] = str(source_dir.resolve())
    summary["source_hashes"] = source_hashes
    summary["split_hash"] = _compute_split_hash(rows)
    summary["skipped_hardlink_toplevel"] = sorted(skip_toplevel)
    if raw_dsp_bootstrap_meta is not None:
        summary["raw_dsp_bootstrap"] = raw_dsp_bootstrap_meta
    _enrich_summary_ranges(summary, conditions=conditions, labels=labels, rows=rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _link_tree(source_dir, output_dir, skip=skip_toplevel)
    if "features" in skip_toplevel:
        # 显式占位，防止下游误以为 source RawDSP 已可用
        features_note = output_dir / "features" / "RAW_DSP_MUST_REBUILD.txt"
        features_note.parent.mkdir(parents=True, exist_ok=True)
        features_note.write_text(
            "Derived split must rebuild train-calibrated RawDSP cache; "
            "source features/raw_dsp was intentionally not hard-linked.\n",
            encoding="utf-8",
        )

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    for name in SPLIT_NAMES:
        write_csv(splits_dir / f"{name}.csv", SPLIT_FIELDS, rows[name])
    summary["splits"] = {
        name: {
            "sequence_count": len(rows[name]),
            "mixture_count": len({r["mixture_id"] for r in rows[name]}),
        }
        for name in SPLIT_NAMES
    }
    summary.setdefault("group_field", "mixture_id")
    write_json(splits_dir / "split_summary.json", summary)
    return {
        "split_policy": summary["split_policy"],
        "splits": {n: len(rows[n]) for n in SPLIT_NAMES},
        "split_hash": summary["split_hash"],
        "x_feature_profile": summary.get("x_feature_profile"),
        "ood_set_hash": summary.get("ood_set_hash"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute tv3 dataset split without re-running physics simulation."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-strategy", choices=_VALID_SPLIT_STRATEGIES, required=True)
    parser.add_argument("--spxy-alpha", type=float, default=0.5)
    parser.add_argument(
        "--extrapolation-strategy",
        choices=_VALID_EXTRAPOLATION_STRATEGIES,
        default="none",
    )
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument(
        "--spxy-x-profile",
        choices=VALID_SPXY_X_PROFILES,
        default=SPXY_X_PROFILE_ORACLE_V1,
        help=(
            "SPXY X 特征 profile。B7 协议必须用 observed_v1；"
            "oracle_v1 仅作 oracle_split_sensitivity。"
        ),
    )
    parser.add_argument(
        "--raw-dsp-cache-dir",
        type=Path,
        default=None,
        help="observed_v1 所需 RawDSP cache 目录（仅用于 SPXY X，不硬链接到派生目录）。",
    )
    parser.add_argument(
        "--also-link-features",
        action="store_true",
        help="危险：同时硬链接 source features/。B7 协议禁止使用。",
    )
    args = parser.parse_args()

    if not args.source_dir.is_dir() or not (args.source_dir / "condition_grid_sequence.csv").is_file():
        parser.error(f"--source-dir 必须是已生成的 tv3 benchmark 目录: {args.source_dir}")

    skip = frozenset({"splits"}) if args.also_link_features else _DEFAULT_SKIP_TOPLEVEL
    try:
        info = recompute_split(
            args.source_dir,
            args.output_dir,
            split_strategy=args.split_strategy,
            spxy_alpha=args.spxy_alpha,
            extrapolation_strategy=args.extrapolation_strategy,
            seed=args.seed,
            spxy_x_profile=args.spxy_x_profile,
            raw_dsp_cache_dir=args.raw_dsp_cache_dir,
            skip_toplevel=skip,
        )
    except (SpxySplitError, ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))
        return 2

    print(f"[recompute-split] {info['split_policy']} -> {args.output_dir}")
    if info.get("x_feature_profile"):
        print(f"  x_feature_profile: {info['x_feature_profile']}")
    print(f"  split_hash: {info['split_hash']}")
    print(f"  ood_set_hash: {info.get('ood_set_hash')}")
    for name in SPLIT_NAMES:
        print(f"  {name}: {info['splits'][name]} sequences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
