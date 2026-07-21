"""S-Flow selector for F5: mixture-level |v_path| holdout.

Train/val/test are drawn only from mixtures whose median |v_path| ≤ train_max.
Mixtures with median |v_path| in (train_max, ood_max] become the pure extrapolation OOD set.
Preserves mixture_id grouping (no sequence-level leakage across splits).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

from tv3.sim.core.tunnel_ventilation_bidir_schema import SPLIT_FIELDS, SPLIT_NAMES
from tv3.sim.packaging.io import write_csv, write_json
from tv3.sim.packaging.splits import build_split_rows_from_group_sets
from tv3.sim.packaging.spxy_split import hash_sequence_id_set

DEFAULT_TRAIN_ABS_V_MAX = 2.5
DEFAULT_OOD_ABS_V_MAX = 4.0
S_FLOW_SPLIT_POLICY = "s_flow_abs_v_path_mixture_median_v1"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_conditions(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mixture_median_abs_v_path(conditions: list[dict[str, str]]) -> dict[str, float]:
    """Return mixture_id → median |v_path_m_per_s|."""
    buckets: dict[str, list[float]] = {}
    for row in conditions:
        mixture_id = str(row["mixture_id"])
        abs_v = abs(float(row["v_path_m_per_s"]))
        buckets.setdefault(mixture_id, []).append(abs_v)
    out: dict[str, float] = {}
    for mixture_id, values in buckets.items():
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2 == 1:
            out[mixture_id] = float(ordered[mid])
        else:
            out[mixture_id] = 0.5 * (float(ordered[mid - 1]) + float(ordered[mid]))
    return out


def classify_mixtures_by_abs_v(
    median_abs_v: dict[str, float],
    *,
    train_abs_v_max: float = DEFAULT_TRAIN_ABS_V_MAX,
    ood_abs_v_max: float = DEFAULT_OOD_ABS_V_MAX,
) -> dict[str, str]:
    """Map mixture_id → {'in_domain' | 'ood' | 'excluded'}."""
    if train_abs_v_max <= 0.0:
        raise ValueError("train_abs_v_max must be > 0")
    if ood_abs_v_max <= train_abs_v_max:
        raise ValueError("ood_abs_v_max must be > train_abs_v_max")
    labels: dict[str, str] = {}
    for mixture_id, abs_v in median_abs_v.items():
        if abs_v <= train_abs_v_max:
            labels[mixture_id] = "in_domain"
        elif abs_v <= ood_abs_v_max:
            labels[mixture_id] = "ood"
        else:
            labels[mixture_id] = "excluded"
    return labels


def zero_anchor_sequence_ids(conditions: list[dict[str, str]], *, atol: float = 1e-12) -> tuple[str, ...]:
    return tuple(
        str(row["sequence_id"])
        for row in conditions
        if abs(float(row["v_path_m_per_s"])) <= atol
    )


def _split_in_domain_train_val_test(
    conditions: list[dict[str, str]],
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, list[dict[str, str]]]:
    """Assign every in-domain mixture to train/val/test (ratios sum to 1; no remainder)."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-12:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    group_ids = sorted({str(row["mixture_id"]) for row in conditions})
    rng = random.Random(seed)
    rng.shuffle(group_ids)
    n_groups = len(group_ids)
    if n_groups < 1:
        raise ValueError("in-domain mixture set is empty")
    n_train = max(1, int(round(n_groups * train_ratio)))
    n_val = max(0, int(round(n_groups * val_ratio)))
    if n_groups >= 3:
        n_val = max(1, n_val)
    n_test = n_groups - n_train - n_val
    if n_groups >= 3 and n_test < 1:
        # Rebalance so each of train/val/test gets at least one mixture when possible.
        n_train = max(1, n_groups - 2)
        n_val = 1
        n_test = n_groups - n_train - n_val
    if n_train + n_val + n_test != n_groups:
        raise RuntimeError("in-domain split counts do not cover all mixtures")
    train_end = n_train
    val_end = train_end + n_val
    split_groups = {
        "train": set(group_ids[:train_end]),
        "val": set(group_ids[train_end:val_end]),
        "test": set(group_ids[val_end:]),
    }
    return build_split_rows_from_group_sets(conditions, split_groups)


def _link_tree(src: Path, dst: Path, *, skip: frozenset[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in skip:
            continue
        target = dst / item.name
        if item.is_dir():
            _link_tree(item, target, skip=frozenset())
            continue
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(item, target)
        except OSError as exc:
            raise RuntimeError(
                f"hardlink failed: {item} -> {target} ({exc}); "
                "derived split must share a filesystem with the source"
            ) from exc


def derive_s_flow_split(
    source_dir: Path | str,
    output_dir: Path | str,
    *,
    seed: int = 20260721,
    train_abs_v_max: float = DEFAULT_TRAIN_ABS_V_MAX,
    ood_abs_v_max: float = DEFAULT_OOD_ABS_V_MAX,
    skip_toplevel: frozenset[str] = frozenset({"splits", "features"}),
) -> dict[str, Any]:
    """Hard-link source dataset and write S-Flow splits/."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    conditions = _load_conditions(source_dir / "condition_grid_sequence.csv")
    if not conditions:
        raise ValueError(f"empty condition grid: {source_dir}")
    if "v_path_m_per_s" not in conditions[0]:
        raise ValueError("S-Flow requires v_path_m_per_s in condition_grid_sequence.csv")

    median_abs_v = mixture_median_abs_v_path(conditions)
    mixture_labels = classify_mixtures_by_abs_v(
        median_abs_v,
        train_abs_v_max=train_abs_v_max,
        ood_abs_v_max=ood_abs_v_max,
    )
    in_domain = [row for row in conditions if mixture_labels[str(row["mixture_id"])] == "in_domain"]
    ood = [row for row in conditions if mixture_labels[str(row["mixture_id"])] == "ood"]
    excluded = [row for row in conditions if mixture_labels[str(row["mixture_id"])] == "excluded"]
    if not in_domain:
        raise ValueError("S-Flow in-domain mixture set is empty")
    if not ood:
        raise ValueError("S-Flow OOD mixture set is empty")

    # Random mixture split among in-domain only; extrapolation is pure OOD.
    id_rows = _split_in_domain_train_val_test(in_domain, seed=seed)
    rows: dict[str, list[dict[str, str]]] = {
        "train": id_rows["train"],
        "val": id_rows["val"],
        "test": id_rows["test"],
        "extrapolation": [
            {"sequence_id": row["sequence_id"], "mixture_id": row["mixture_id"]} for row in ood
        ],
    }
    for row in rows["extrapolation"]:
        if median_abs_v[str(row["mixture_id"])] <= train_abs_v_max:
            raise ValueError(
                f"S-Flow extrapolation contaminated by in-domain mixture {row['mixture_id']}"
            )
    all_ids = [r["sequence_id"] for name in SPLIT_NAMES for r in rows[name]]
    expected_ids = {str(row["sequence_id"]) for row in in_domain} | {
        str(row["sequence_id"]) for row in ood
    }
    if set(all_ids) != expected_ids:
        raise ValueError("S-Flow split ID set does not match in-domain∪OOD")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("S-Flow split has overlapping sequence_ids")

    source_hashes = {
        "manifest_sha256": _file_sha256(source_dir / "manifest.json"),
        "labels_sha256": _file_sha256(source_dir / "labels" / "y.npy"),
        "condition_grid_sha256": _file_sha256(source_dir / "condition_grid_sequence.csv"),
    }
    zero_ids = zero_anchor_sequence_ids(conditions)
    summary: dict[str, Any] = {
        "split_policy": S_FLOW_SPLIT_POLICY,
        "group_field": "mixture_id",
        "split_seed": int(seed),
        "train_abs_v_max": float(train_abs_v_max),
        "ood_abs_v_max": float(ood_abs_v_max),
        "aggregation": "mixture_median_abs_v_path",
        "source_dataset": str(source_dir.resolve()),
        "source_hashes": source_hashes,
        "skipped_hardlink_toplevel": sorted(skip_toplevel),
        "mixture_counts": {
            "in_domain": len({str(r["mixture_id"]) for r in in_domain}),
            "ood": len({str(r["mixture_id"]) for r in ood}),
            "excluded": len({str(r["mixture_id"]) for r in excluded}),
        },
        "sequence_counts": {
            "in_domain": len(in_domain),
            "ood": len(ood),
            "excluded": len(excluded),
        },
        "ood_set_hash": hash_sequence_id_set([r["sequence_id"] for r in rows["extrapolation"]]),
        "zero_anchor_sequence_ids_hash": hash_sequence_id_set(zero_ids),
        "zero_anchor_sequence_count": len(zero_ids),
        "extrapolation_note": (
            f"pure OOD: mixtures with median |v_path| in ({train_abs_v_max}, {ood_abs_v_max}]; "
            "in-domain mixtures are assigned only to train/val/test"
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _link_tree(source_dir, output_dir, skip=skip_toplevel)
    if "features" in skip_toplevel:
        note = output_dir / "features" / "BIDIR_FEATURES_MUST_REBUILD.txt"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(
            "Derived S-Flow split must rebuild train-calibrated bidir feature cache; "
            "source features/ was intentionally not hard-linked.\n",
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
    summary["split_hash"] = hashlib.sha256(
        json.dumps(
            {name: [r["sequence_id"] for r in rows[name]] for name in SPLIT_NAMES},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    write_json(splits_dir / "split_summary.json", summary)
    return {
        "split_policy": S_FLOW_SPLIT_POLICY,
        "splits": {name: len(rows[name]) for name in SPLIT_NAMES},
        "split_hash": summary["split_hash"],
        "ood_set_hash": summary["ood_set_hash"],
        "zero_anchor_sequence_count": len(zero_ids),
        "mixture_counts": summary["mixture_counts"],
    }


__all__ = [
    "DEFAULT_OOD_ABS_V_MAX",
    "DEFAULT_TRAIN_ABS_V_MAX",
    "S_FLOW_SPLIT_POLICY",
    "classify_mixtures_by_abs_v",
    "derive_s_flow_split",
    "mixture_median_abs_v_path",
    "zero_anchor_sequence_ids",
]
