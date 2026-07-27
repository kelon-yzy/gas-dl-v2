"""MEI-1 forward-envelope unit tests."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from tv3.audit.identifiability_v3_mrs import MrsPoint
from tv3.audit.mrs_ei_forward_envelope import (
    EnvelopeSpec,
    apply_envelope,
    baseline_spectrum,
    build_aligned_delta_tof,
    decide_mei1_verdict,
    design_id,
    diffraction_seed_delta_tof,
    enumerate_k4_designs,
    make_spectrum_fn,
    principal_angle_deg,
    rank_designs,
    select_audit_points,
    spearman_rank_corr,
)
from tv3.audit.mrs_ei_registry import load_json

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "configs" / "tv3_mrs_ei"


def test_principal_angle_identical_is_zero():
    v = np.array([1.0, 2.0, 3.0])
    assert principal_angle_deg(v, v) == 0.0
    assert principal_angle_deg(v, -v) == 0.0


def test_spearman_perfect():
    assert abs(spearman_rank_corr([0, 1, 2, 3], [0, 1, 2, 3]) - 1.0) < 1e-12


def test_enumerate_includes_baseline_first():
    pool = [25000, 40000, 63000, 100000, 160000, 200000]
    base = [25000, 63000, 100000, 200000]
    designs = enumerate_k4_designs(pool, base)
    assert designs[0] == tuple(sorted(base))
    assert len(designs) == 15
    assert design_id(base) == "K4[25,63,100,200k]"


def test_f1_zero_correction_matches_baseline():
    pt = MrsPoint(1.0, 20.0, 25.0, 0.25, 50.0, 0.101325)
    f = np.array([25000.0, 100000.0, 200000.0])
    base = baseline_spectrum(pt, f)
    same = apply_envelope(base, pt, EnvelopeSpec("F0", "baseline"))
    assert np.allclose(same["c_f"], base["c_f"])
    scaled = apply_envelope(
        baseline_spectrum(pt, f),
        pt,
        EnvelopeSpec("F1", "envelope", c_eq_relative_correction=0.01),
    )
    assert math.isclose(float(scaled["c_eq"]), float(base["c_eq"]) * 1.01, rel_tol=1e-12)
    assert np.allclose(scaled["c_f"], np.asarray(base["c_f"]) * 1.01)


def test_f5_changes_alpha_not_c():
    pt = MrsPoint(1.0, 20.0, 25.0, 0.25, 50.0, 0.101325)
    f = np.array([25000.0, 63000.0, 200000.0])
    base = baseline_spectrum(pt, f)
    out = make_spectrum_fn(
        EnvelopeSpec("F5", "proxy", alpha_ripple_amp=0.05)
    )(pt, f)
    assert np.allclose(out["c_f"], base["c_f"])
    assert not np.allclose(out["alpha_f"], base["alpha_f"])


def test_select_full_narrow_grid_is_216():
    design = load_json(_CFG / "design_space.json")
    pts = select_audit_points(design, mode="full_narrow_grid", stride=1)
    assert len(pts) == 216


def test_rank_designs_uses_delta_num_ties():
    summaries = {
        "A": {"max_p90_o2_percent": 1.000},
        "B": {"max_p90_o2_percent": 1.005},  # 0.5% above A
        "C": {"max_p90_o2_percent": 1.030},  # 3% above A
    }
    rows = rank_designs(summaries, metric="max_p90_o2_percent", delta_num=0.02)
    by_id = {r["design_id"]: r for r in rows}
    assert by_id["A"]["rank"] == by_id["B"]["rank"] == 0
    assert by_id["C"]["rank"] == 1
    assert rows[0]["ranking_resolvable"] is True
    assert rows[0]["distinguishable_rank_levels"] == 2


def test_rank_designs_unresolvable_when_span_within_delta_num():
    summaries = {
        "A": {"max_p90_o2_percent": 7.53},
        "B": {"max_p90_o2_percent": 7.55},
        "C": {"max_p90_o2_percent": 7.58},
    }
    rows = rank_designs(summaries, metric="max_p90_o2_percent", delta_num=0.02)
    assert all(r["rank"] == 0 for r in rows)
    assert rows[0]["ranking_resolvable"] is False


def test_aligned_delta_tof_parallel_and_orthogonal():
    j = np.array([1.0, 2.0, 0.5, -0.2], dtype=np.float64)
    y0 = np.array([1e-3, 1.1e-3, 0.9e-3, 1.2e-3], dtype=np.float64)
    seed = np.array([0.3, -0.1, 0.4, 0.2], dtype=np.float64)
    d_par, meta_par = build_aligned_delta_tof(
        direction="parallel",
        j_o2=j,
        y0=y0,
        relative_rms=0.01,
        seed_delta_tof=seed,
    )
    d_orth, meta_orth = build_aligned_delta_tof(
        direction="orthogonal",
        j_o2=j,
        y0=y0,
        relative_rms=0.01,
        seed_delta_tof=seed,
    )
    assert meta_par["angle_to_o2_jacobian_deg"] < 1e-6
    assert abs(meta_orth["angle_to_o2_jacobian_deg"] - 90.0) < 1e-6
    assert abs(float(np.dot(d_orth, j))) < 1e-12 * float(np.linalg.norm(d_orth) * np.linalg.norm(j) + 1.0)


def test_decide_verdict_blocks_unrepresented_and_unresolvable():
    out = decide_mei1_verdict(
        issues=[],
        flip_events=[],
        unrepresented_blocking=["F2_h2o_relaxation_params"],
        ranking_resolvable=True,
    )
    assert out["passed"] is False
    assert out["verdict"] == "mei1_inconclusive_forward_model"
    assert any("unrepresented" in b for b in out["blockers"])

    out2 = decide_mei1_verdict(
        issues=[],
        flip_events=[],
        unrepresented_blocking=[],
        ranking_resolvable=False,
    )
    assert out2["passed"] is False
    assert "design_ranking_not_resolvable_within_delta_num" in out2["blockers"]


def test_diffraction_seed_nonzero():
    pt = MrsPoint(1.0, 20.0, 25.0, 0.25, 50.0, 0.101325)
    f = np.array([25000.0, 63000.0, 100000.0, 200000.0])
    c = np.full_like(f, 340.0)
    seed = diffraction_seed_delta_tof(
        pt, f_hz=f, c_f=c, amp=0.002, frequency_floor_hz=15000.0
    )
    assert float(np.linalg.norm(seed)) > 0.0
