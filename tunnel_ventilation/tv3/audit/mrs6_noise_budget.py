"""MRS-6 noise-budget sensitivity scan over precomputed MRS-2 Jacobians.

Spec derivation only: this module does NOT rerun or alter the frozen MRS-2
verdict (`mrs2_rank_upgraded_p90_fail`) and does not touch the business gates.
Jacobians are independent of the noise model, so each point/arm Jacobian is
computed once and every noise budget is re-evaluated with cheap 5x5 algebra.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tv3.audit.identifiability_v3_mrs import (
    ARM_IDS,
    MrsPoint,
    fisher_rank_crb,
    local_mrs_jacobian,
    observation_noise_std,
    single_nuisance_equivalent_o2,
)

REQUIRED_PRIOR_KEYS = ("t_c", "path_length_m", "h_rh")


@dataclass(frozen=True)
class NoiseBudget:
    """One noise/prior scenario evaluated against precomputed Jacobians."""

    budget_id: str
    jitter_std_s: float
    relative_amp_std: float
    prior_std: Mapping[str, float]
    phase_std_s_at_anchor: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(self.jitter_std_s) or self.jitter_std_s <= 0.0:
            raise ValueError("jitter_std_s must be finite and > 0")
        if not math.isfinite(self.relative_amp_std) or self.relative_amp_std < 0.0:
            raise ValueError("relative_amp_std must be finite and >= 0")
        for key in REQUIRED_PRIOR_KEYS:
            if key not in self.prior_std:
                raise ValueError(f"prior_std missing required key {key!r}")


@dataclass(frozen=True)
class ArmJacobians:
    """Noise-independent Jacobians for one arm over a point set."""

    arm: str
    f_hz: tuple[float, ...]
    entries: tuple[dict[str, Any], ...] = field(repr=False)


def precompute_arm_jacobians(
    points: Sequence[MrsPoint],
    *,
    arm: str,
    f_hz: Sequence[float],
    parameter_steps: Mapping[str, float],
    parameter_bounds: Mapping[str, Sequence[float]],
    fixed_delay_s: float,
    rh_delta: float,
    p_scan_mpa: Sequence[float],
    max_relative_step_disagreement: float = 0.01,
) -> ArmJacobians:
    if arm not in ARM_IDS:
        raise ValueError(f"unknown arm {arm!r}")
    if not points:
        raise ValueError("points must be non-empty")
    entries: list[dict[str, Any]] = []
    for point in points:
        loc = local_mrs_jacobian(
            point,
            arm=arm,
            f_hz=f_hz,
            parameter_steps=parameter_steps,
            parameter_bounds=parameter_bounds,
            fixed_delay_s=fixed_delay_s,
            rh_delta=rh_delta,
            p_scan_mpa=p_scan_mpa,
            max_relative_step_disagreement=max_relative_step_disagreement,
        )
        entries.append(
            {
                "point": point,
                "jacobian": loc["jacobian"],
                "labels": loc["labels"],
                "all_stable": loc["all_stable"],
            }
        )
    return ArmJacobians(arm=arm, f_hz=tuple(float(f) for f in f_hz), entries=tuple(entries))


def evaluate_budget(
    jacobians: ArmJacobians,
    *,
    budget: NoiseBudget,
    parameter_steps: Mapping[str, float],
    window_width_percent: float,
) -> dict[str, Any]:
    """Re-evaluate CRB/P90/nuisance for every point under one noise budget."""
    budget.validate()
    p90s: list[float] = []
    ranks: list[int] = []
    nuisance_fracs: list[float] = []
    invertible = True
    for entry in jacobians.entries:
        sigmas = observation_noise_std(
            entry["labels"],
            point=entry["point"],
            jitter_std_s=budget.jitter_std_s,
            relative_amp_std=budget.relative_amp_std,
            phase_std_s_at_anchor=budget.phase_std_s_at_anchor,
        )
        fish = fisher_rank_crb(
            entry["jacobian"],
            row_sigmas=sigmas,
            parameter_steps=parameter_steps,
            prior_std=budget.prior_std,
        )
        nuis = single_nuisance_equivalent_o2(
            entry["jacobian"],
            row_sigmas=sigmas,
            parameter_steps=parameter_steps,
            prior_std=budget.prior_std,
            window_width_percent=window_width_percent,
        )
        invertible = invertible and bool(fish["fisher_aug_invertible"])
        ranks.append(int(fish["joint_rank"]))
        p90 = float(fish["p90_o2_percent"])
        p90s.append(p90 if math.isfinite(p90) else float("inf"))
        nuisance_fracs.append(float(nuis["worst_fraction_of_window"]))
    finite_p90s = sorted(p90s)
    return {
        "arm": jacobians.arm,
        "budget_id": budget.budget_id,
        "jitter_std_s": budget.jitter_std_s,
        "relative_amp_std": budget.relative_amp_std,
        "prior_std": dict(budget.prior_std),
        "n_points": len(jacobians.entries),
        "min_joint_rank": min(ranks),
        "max_p90_o2_percent": max(p90s),
        "median_p90_o2_percent": finite_p90s[len(finite_p90s) // 2],
        "max_nuisance_fraction": max(nuisance_fracs),
        "all_crlb_invertible": invertible,
    }


def budget_passes(
    row: Mapping[str, Any],
    *,
    target_p90: float,
    max_nuisance_fraction: float,
    statistic: str = "max",
) -> bool:
    """Spec-target check (same frozen numbers as MRS-2 gates; not a verdict)."""
    if statistic not in ("max", "median"):
        raise ValueError("statistic must be 'max' or 'median'")
    key = "max_p90_o2_percent" if statistic == "max" else "median_p90_o2_percent"
    return (
        bool(row["all_crlb_invertible"])
        and int(row["min_joint_rank"]) >= 2
        and float(row[key]) <= target_p90
        and float(row["max_nuisance_fraction"]) <= max_nuisance_fraction
    )


def required_budget_from_scan(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_p90: float,
    max_nuisance_fraction: float,
    statistic: str = "max",
) -> Mapping[str, Any] | None:
    """First (loosest) passing row from a scan ordered loose→tight, else None."""
    for row in rows:
        if budget_passes(
            row,
            target_p90=target_p90,
            max_nuisance_fraction=max_nuisance_fraction,
            statistic=statistic,
        ):
            return row
    return None


def pareto_passing_combos(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_p90: float,
    max_nuisance_fraction: float,
    statistic: str = "max",
) -> list[Mapping[str, Any]]:
    """Passing combos not dominated by a looser passing combo.

    A combo dominates another if both its jitter and its T prior are looser
    (>=) with at least one strictly looser. Loose combos are cheaper hardware.
    """
    passing = [
        r
        for r in rows
        if budget_passes(
            r,
            target_p90=target_p90,
            max_nuisance_fraction=max_nuisance_fraction,
            statistic=statistic,
        )
    ]

    def _key(r: Mapping[str, Any]) -> tuple[float, float]:
        return (float(r["jitter_std_s"]), float(r["prior_std"]["t_c"]))

    pareto: list[Mapping[str, Any]] = []
    for row in passing:
        jr, tr = _key(row)
        dominated = any(
            (jo >= jr and to >= tr and (jo > jr or to > tr))
            for jo, to in (_key(other) for other in passing)
        )
        if not dominated:
            pareto.append(row)
    return sorted(pareto, key=_key, reverse=True)


__all__ = [
    "ArmJacobians",
    "NoiseBudget",
    "budget_passes",
    "evaluate_budget",
    "pareto_passing_combos",
    "precompute_arm_jacobians",
    "required_budget_from_scan",
]
