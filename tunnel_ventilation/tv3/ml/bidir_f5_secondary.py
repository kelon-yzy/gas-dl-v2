"""F5-S: bidir secondary selectors (S-Y / S-L) for criterion (d).

Frozen contract: ``docs/archive/parked/tv3_bidirectional_ultrasound_implementation_plan.md`` §F5-S.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from tv3.sim.packaging.spxy_split import (
    BIDIR_SPXY_OBSERVED_AB_EXPECTED_DIM,
    SPXY_X_PROFILE_BIDIR_OBSERVED_AB_V1,
    spxy_x_feature_names,
)

# Frozen F5-S split seeds (plan §F5-S.3).
F5S_SPLIT_SEEDS: tuple[int, ...] = (20260704, 20260712, 20260720)
F5S_SPXY_ALPHA = 0.5
F5S_X_PROFILE = SPXY_X_PROFILE_BIDIR_OBSERVED_AB_V1
F5S_GATE_SPLITS: tuple[str, ...] = ("test", "extrapolation")
F5S_PRIMARY_HEAD = "b1_ridge"
F5S_GATE_ARMS: tuple[str, ...] = ("A1", "A3")

SELECTOR_SPECS: tuple[dict[str, str], ...] = (
    {
        "selector_id": "s_y",
        "extrapolation_strategy": "y_margin_ood",
        "dir_prefix": "s_y",
        "slug_template": "spxy_ab_a05_ymargin_s{seed}",
    },
    {
        "selector_id": "s_l",
        "extrapolation_strategy": "lhs_boundary",
        "dir_prefix": "s_l",
        "slug_template": "spxy_ab_a05_lhsboundary_s{seed}",
    },
)


def f5s_split_dirname(*, selector_id: str, seed: int) -> str:
    for spec in SELECTOR_SPECS:
        if spec["selector_id"] == selector_id:
            return spec["slug_template"].format(seed=int(seed))
    raise KeyError(f"unknown selector_id={selector_id!r}")


def f5s_split_relpath(*, selector_id: str, seed: int) -> Path:
    for spec in SELECTOR_SPECS:
        if spec["selector_id"] == selector_id:
            return Path(spec["dir_prefix"]) / f5s_split_dirname(selector_id=selector_id, seed=seed)
    raise KeyError(f"unknown selector_id={selector_id!r}")


def expected_x_feature_digest(*, slow_channels: int = 7) -> dict[str, Any]:
    names = spxy_x_feature_names(F5S_X_PROFILE, slow_channels=slow_channels)
    if len(names) != BIDIR_SPXY_OBSERVED_AB_EXPECTED_DIM:
        raise RuntimeError(
            f"F5-S X dim drift: expected {BIDIR_SPXY_OBSERVED_AB_EXPECTED_DIM}, got {len(names)}"
        )
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    return {
        "x_feature_profile_cli": F5S_X_PROFILE,
        "x_feature_count": len(names),
        "x_feature_names": names,
        "x_feature_names_digest": digest,
    }


def evaluate_criterion_d(
    cell_metrics: dict[str, dict[str, Any]],
    *,
    threshold: float = -0.01,
    head: str = F5S_PRIMARY_HEAD,
    seeds: tuple[int, ...] = F5S_SPLIT_SEEDS,
) -> dict[str, Any]:
    """Evaluate the 12 paired ΔR² cells for criterion (d).

    ``cell_metrics`` keys: ``{selector_id}:{seed}:{arm}:{head}`` with nested
    ``evaluations[split].component_metrics.x_O2.r2``.
    """
    cells: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in SELECTOR_SPECS:
        selector_id = spec["selector_id"]
        for seed in seeds:
            for split_name in F5S_GATE_SPLITS:
                key_a1 = f"{selector_id}:{seed}:A1:{head}"
                key_a3 = f"{selector_id}:{seed}:A3:{head}"
                if key_a1 not in cell_metrics or key_a3 not in cell_metrics:
                    missing.append(f"{selector_id}:{seed}:{split_name}")
                    cells.append(
                        {
                            "selector_id": selector_id,
                            "seed": int(seed),
                            "split": split_name,
                            "head": head,
                            "status": "missing",
                            "passed": False,
                        }
                    )
                    continue
                try:
                    r2_a1 = float(
                        cell_metrics[key_a1]["evaluations"][split_name]["component_metrics"]["x_O2"]["r2"]
                    )
                    r2_a3 = float(
                        cell_metrics[key_a3]["evaluations"][split_name]["component_metrics"]["x_O2"]["r2"]
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    missing.append(f"{selector_id}:{seed}:{split_name}")
                    cells.append(
                        {
                            "selector_id": selector_id,
                            "seed": int(seed),
                            "split": split_name,
                            "head": head,
                            "status": "incomplete_metrics",
                            "error": str(exc),
                            "passed": False,
                        }
                    )
                    continue
                delta = r2_a3 - r2_a1
                passed = bool(delta >= threshold)
                cells.append(
                    {
                        "selector_id": selector_id,
                        "seed": int(seed),
                        "split": split_name,
                        "head": head,
                        "a1_o2_r2": r2_a1,
                        "a3_o2_r2": r2_a3,
                        "delta_r2_o2": delta,
                        "threshold_min": threshold,
                        "status": "ok",
                        "passed": passed,
                    }
                )

    expected_n = len(SELECTOR_SPECS) * len(seeds) * len(F5S_GATE_SPLITS)
    complete = len(missing) == 0 and len(cells) == expected_n
    all_pass = complete and all(cell["passed"] for cell in cells)
    if not complete:
        status = "incomplete"
        passed = False
    elif all_pass:
        status = "passed"
        passed = True
    else:
        status = "failed"
        passed = False
    return {
        "status": status,
        "passed": passed,
        "threshold_min": threshold,
        "head": head,
        "expected_cells": expected_n,
        "n_cells": len(cells),
        "missing": missing,
        "cells": cells,
        "note": (
            "Criterion d uses only paired A3−A1 O₂ R² on the same selector/seed/split; "
            "no cross-selector or mean aggregation."
        ),
    }


def audit_derived_split_summary(
    summary: dict[str, Any],
    *,
    require_split_hash: bool = True,
) -> list[str]:
    """Return issue strings if a derived split_summary fails F5-S provenance gates."""
    issues: list[str] = []
    if summary.get("spxy_x_profile_cli") != F5S_X_PROFILE:
        issues.append(
            f"spxy_x_profile_cli={summary.get('spxy_x_profile_cli')!r} != {F5S_X_PROFILE!r}"
        )
    if summary.get("x_feature_profile") != "bidir_spxy_observed_ab_stats_v1":
        issues.append(f"x_feature_profile={summary.get('x_feature_profile')!r}")
    if int(summary.get("x_feature_count") or 0) != BIDIR_SPXY_OBSERVED_AB_EXPECTED_DIM:
        issues.append(f"x_feature_count={summary.get('x_feature_count')}")
    # After recompute, require split_hash.
    if require_split_hash and not summary.get("split_hash"):
        issues.append("missing split_hash")
    if not summary.get("ood_set_hash"):
        issues.append("missing ood_set_hash")
    if not summary.get("x_feature_matrix_hash"):
        issues.append("missing x_feature_matrix_hash")
    bootstrap = summary.get("raw_dsp_bootstrap") or {}
    if bootstrap.get("role") != "split_selection_bootstrap_only":
        issues.append("raw_dsp_bootstrap.role must be split_selection_bootstrap_only")
    names = summary.get("x_feature_names") or []
    if any("_ba_" in str(n) or str(n).endswith("_ba") for n in names):
        issues.append("x_feature_names contains BA columns")
    return issues


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = [
    "F5S_GATE_ARMS",
    "F5S_GATE_SPLITS",
    "F5S_PRIMARY_HEAD",
    "F5S_SPLIT_SEEDS",
    "F5S_SPXY_ALPHA",
    "F5S_X_PROFILE",
    "SELECTOR_SPECS",
    "audit_derived_split_summary",
    "evaluate_criterion_d",
    "expected_x_feature_digest",
    "f5s_split_dirname",
    "f5s_split_relpath",
    "write_json",
]
