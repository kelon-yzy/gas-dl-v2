"""MRS-6 noise-budget scan unit tests (spec derivation; MRS-2 verdict untouched)."""
from __future__ import annotations

import pytest

from tv3.audit.identifiability_v3_mrs import MrsPoint
from tv3.audit.mrs6_noise_budget import (
    NoiseBudget,
    budget_passes,
    evaluate_budget,
    pareto_passing_combos,
    precompute_arm_jacobians,
    required_budget_from_scan,
)

_STEPS = {
    "o2_percent": 0.01,
    "co2_percent": 0.01,
    "t_c": 0.1,
    "path_length_m": 0.001,
    "h_rh": 0.5,
}
_BOUNDS = {
    "co2_percent": [0.03, 5.0],
    "o2_percent": [18.0, 21.2],
    "t_c": [15.0, 35.0],
    "path_length_m": [0.2, 0.3],
    "h_rh": [20.0, 80.0],
}
_F_HZ = [25000.0, 63000.0, 100000.0, 200000.0]
_PRIOR = {"t_c": 1.0, "path_length_m": 0.0001, "h_rh": 2.0, "co2_percent": 0.05}


@pytest.fixture(scope="module")
def cfreq_jacobians():
    points = [
        MrsPoint(1.0, 20.0, 25.0, 0.25, 50.0, 0.101325),
        MrsPoint(2.515, 19.2, 25.0, 0.25, 50.0, 0.101325),
    ]
    return precompute_arm_jacobians(
        points,
        arm="obs-cfreq",
        f_hz=_F_HZ,
        parameter_steps=_STEPS,
        parameter_bounds=_BOUNDS,
        fixed_delay_s=8.2e-5,
        rh_delta=20.0,
        p_scan_mpa=[0.10, 0.50],
    )


def _budget(jitter_s: float, t_prior: float = 1.0) -> NoiseBudget:
    prior = dict(_PRIOR)
    prior["t_c"] = t_prior
    return NoiseBudget(
        budget_id=f"jit{jitter_s:g}_T{t_prior:g}",
        jitter_std_s=jitter_s,
        relative_amp_std=0.02,
        prior_std=prior,
    )


def test_p90_improves_with_smaller_jitter(cfreq_jacobians):
    loose = evaluate_budget(
        cfreq_jacobians, budget=_budget(3e-6), parameter_steps=_STEPS, window_width_percent=0.8
    )
    tight = evaluate_budget(
        cfreq_jacobians, budget=_budget(3e-8), parameter_steps=_STEPS, window_width_percent=0.8
    )
    assert tight["max_p90_o2_percent"] < loose["max_p90_o2_percent"]
    assert tight["min_joint_rank"] >= loose["min_joint_rank"]


def test_p90_improves_with_tighter_t_prior(cfreq_jacobians):
    loose = evaluate_budget(
        cfreq_jacobians, budget=_budget(3e-6, 1.0), parameter_steps=_STEPS, window_width_percent=0.8
    )
    tight = evaluate_budget(
        cfreq_jacobians, budget=_budget(3e-6, 0.1), parameter_steps=_STEPS, window_width_percent=0.8
    )
    assert tight["max_p90_o2_percent"] < loose["max_p90_o2_percent"]


def test_registered_budget_reproduces_mrs2_scale(cfreq_jacobians):
    """At the frozen MRS-2 noise (3 us, T=1K) P90 stays far above the 0.4 target."""
    row = evaluate_budget(
        cfreq_jacobians, budget=_budget(3e-6), parameter_steps=_STEPS, window_width_percent=0.8
    )
    assert row["max_p90_o2_percent"] > 2.0
    assert not budget_passes(row, target_p90=0.4, max_nuisance_fraction=0.5)


def test_required_budget_picks_first_passing():
    rows = [
        {
            "jitter_std_s": 3e-6,
            "min_joint_rank": 4,
            "max_p90_o2_percent": 5.0,
            "median_p90_o2_percent": 4.0,
            "max_nuisance_fraction": 3.0,
            "all_crlb_invertible": True,
            "prior_std": {"t_c": 1.0},
        },
        {
            "jitter_std_s": 1e-7,
            "min_joint_rank": 4,
            "max_p90_o2_percent": 0.3,
            "median_p90_o2_percent": 0.2,
            "max_nuisance_fraction": 0.4,
            "all_crlb_invertible": True,
            "prior_std": {"t_c": 1.0},
        },
    ]
    hit = required_budget_from_scan(rows, target_p90=0.4, max_nuisance_fraction=0.5)
    assert hit is not None and hit["jitter_std_s"] == 1e-7
    assert required_budget_from_scan(rows[:1], target_p90=0.4, max_nuisance_fraction=0.5) is None


def test_pareto_drops_dominated_combos():
    def _row(jit: float, t: float) -> dict:
        return {
            "jitter_std_s": jit,
            "min_joint_rank": 4,
            "max_p90_o2_percent": 0.3,
            "median_p90_o2_percent": 0.2,
            "max_nuisance_fraction": 0.4,
            "all_crlb_invertible": True,
            "prior_std": {"t_c": t},
        }

    rows = [_row(1e-7, 1.0), _row(1e-7, 0.1), _row(5e-8, 1.0)]
    pareto = pareto_passing_combos(rows, target_p90=0.4, max_nuisance_fraction=0.5)
    keys = {(r["jitter_std_s"], r["prior_std"]["t_c"]) for r in pareto}
    assert keys == {(1e-7, 1.0)}
