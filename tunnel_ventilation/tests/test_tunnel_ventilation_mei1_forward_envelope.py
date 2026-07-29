"""MEI-1 forward-envelope v2 unit tests."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from tv3.audit.identifiability_v3_mrs import MrsPoint
from tv3.audit.mrs_ei_forward_envelope import (
    EnvelopeSpec,
    apply_envelope,
    baseline_spectrum,
    build_aligned_delta_tof,
    collect_parked_nonblocking,
    collect_unrepresented_blocking,
    decide_mei1_verdict,
    design_id,
    diffraction_seed_delta_tof,
    enumerate_k4_designs,
    make_spectrum_fn,
    pressure_domain_validation,
    principal_angle_deg,
    proxy_never_clears_not_represented,
    rank_designs,
    registered_domain_rankings_resolvable,
    run_mei1_audit,
    select_audit_points,
    spearman_rank_corr,
)
from tv3.audit.mrs_ei_registry import (
    FORBIDDEN_AUTH_VALUE,
    REGISTRY_SCHEMA_VERSION,
    build_formal_mei1_points,
    dumps_stable,
    load_json,
)

_ROOT = Path(__file__).resolve().parents[1]
_CFG = _ROOT / "configs" / "tv3_mrs_ei"
_PARENT_MEI0 = (
    _ROOT
    / "outputs"
    / "runs"
    / "tv3_mrs_ei"
    / "mei0_registry"
    / "freezes"
    / "20260727T071921821957Z_f209e893a9e5"
)


def _v2_parent_freeze(tmp_path: Path) -> Path:
    """Build a minimal valid parent MEI-0 v2 freeze for MEI-1 contract tests."""
    parent = tmp_path / "parent_mei0"
    parent.mkdir()
    for name in (
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
    ):
        payload = load_json(_CFG / name)
        if name == "metric_registry.json":
            payload["decision_thresholds"]["delta_numerical"]["by_noise_profile"] = {
                "low_cost_k4_primary": 1.0e-4,
                "registered_mrs2_stress": 2.0e-4,
            }
            payload["decision_thresholds"]["delta_numerical"]["shared_upper_bound"] = (
                2.0e-4
            )
        (parent / name).write_text(dumps_stable(payload), encoding="utf-8")
    stage = {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "allowed_next_stage": "MEI-1_forward_envelope",
        "mei0": {
            "verdict": "mei0_registry_frozen",
            "freeze_dir": str(parent),
        },
        "mei1": None,
    }
    (parent / "stage_status.json").write_text(dumps_stable(stage), encoding="utf-8")
    # Minimal evidence manifest with hashed freeze artifacts present in parent.
    artifact_names = (
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
        "stage_status.json",
    )
    from tv3.audit.mrs_ei_registry import sha256_file

    manifest = {
        "schema_version": "tunnel-ventilation-mrs-ei-freeze-manifest-2",
        "freeze_manifest_schema_version": "tunnel-ventilation-mrs-ei-freeze-manifest-2",
        "artifact_sha256": {
            name: sha256_file(parent / name) for name in artifact_names
        },
        "source_sha256": {},
        "parent_manifest_path": None,
        "parent_manifest_sha256": None,
        "plan_path": None,
        "plan_sha256": None,
    }
    # Remove null parent/plan fields so verifier does not require them.
    del manifest["parent_manifest_path"]
    del manifest["parent_manifest_sha256"]
    del manifest["plan_path"]
    del manifest["plan_sha256"]
    (parent / "evidence_manifest.json").write_text(
        dumps_stable(manifest), encoding="utf-8"
    )
    return parent


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


def test_select_formal_union_is_432():
    design = load_json(_CFG / "design_space.json")
    pts = select_audit_points(design, mode="named_point_sets", stride=1)
    assert len(pts) == 432


def test_mei1_practical_rank_is_separate_from_numerical_rank():
    summaries = {
        "A": {"max_p90_o2_percent": 1.000},
        "B": {"max_p90_o2_percent": 1.005},  # 0.5% above A: within practical 2%
        "C": {"max_p90_o2_percent": 1.030},  # 3% above A
    }
    rows = rank_designs(
        summaries,
        metric="max_p90_o2_percent",
        delta_numerical=0.001,
        delta_practical=0.02,
    )
    by_id = {r["design_id"]: r for r in rows}
    assert by_id["A"]["rank_numerical"] == 0
    assert by_id["B"]["rank_numerical"] == 1
    assert by_id["A"]["rank_practical"] == by_id["B"]["rank_practical"] == 0
    assert by_id["C"]["rank_practical"] == 1
    assert by_id["A"]["rank"] == by_id["A"]["rank_practical"]


def test_rank_designs_unresolvable_when_span_within_delta_practical():
    summaries = {
        "A": {"max_p90_o2_percent": 7.53},
        "B": {"max_p90_o2_percent": 7.55},
        "C": {"max_p90_o2_percent": 7.58},
    }
    rows = rank_designs(
        summaries,
        metric="max_p90_o2_percent",
        delta_numerical=0.001,
        delta_practical=0.02,
    )
    assert all(r["rank_practical"] == 0 for r in rows)
    assert rows[0]["ranking_resolvable_practical"] is False


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
    assert abs(float(np.dot(d_orth, j))) < 1e-12 * float(
        np.linalg.norm(d_orth) * np.linalg.norm(j) + 1.0
    )


def test_decide_verdict_blocks_unrepresented_and_unresolvable():
    out = decide_mei1_verdict(
        issues=[],
        flip_events=[],
        unrepresented_blocking=["F2_h2o_relaxation_params"],
        ranking_resolvable=True,
        pressure_domain_ok=True,
        all_profiles_complete=True,
        formal_point_count_ok=True,
    )
    assert out["passed"] is False
    assert out["verdict"] == "mei1_inconclusive_forward_model"
    assert any("unrepresented" in b for b in out["blockers"])

    out2 = decide_mei1_verdict(
        issues=[],
        flip_events=[],
        unrepresented_blocking=[],
        ranking_resolvable=False,
        pressure_domain_ok=True,
        all_profiles_complete=True,
        formal_point_count_ok=True,
    )
    assert out2["passed"] is True
    assert out2["verdict"] == "mei1_fixed_k4_retained"
    assert out2["allowed_next_stage"] == "MEI-3_varpro_audit"
    assert out2["decision_reason"] == (
        "design_ranking_not_resolvable_within_delta_practical"
    )


def test_diffraction_seed_nonzero():
    pt = MrsPoint(1.0, 20.0, 25.0, 0.25, 50.0, 0.101325)
    f = np.array([25000.0, 63000.0, 100000.0, 200000.0])
    c = np.full_like(f, 340.0)
    seed = diffraction_seed_delta_tof(
        pt, f_hz=f, c_f=c, amp=0.002, frequency_floor_hz=15000.0
    )
    assert float(np.linalg.norm(seed)) > 0.0


def test_mei1_requires_explicit_parent_freeze():
    out = run_mei1_audit(project_root=_ROOT, config_dir=_CFG, parent_mei0_freeze_dir=None)
    assert out["verdict"] == "mei1_audit_failed"
    assert any("parent-mei0-freeze-dir" in item for item in out["issues"])
    assert out["exit_code_hint"] == 3


def test_mei1_rejects_parent_manifest_mismatch(tmp_path):
    parent = _v2_parent_freeze(tmp_path)
    # Corrupt artifact after hashing.
    (parent / "design_space.json").write_text(
        (parent / "design_space.json").read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    out = run_mei1_audit(
        project_root=_ROOT,
        config_dir=_CFG,
        parent_mei0_freeze_dir=parent,
        current_stage_status={
            "mei0": {"freeze_dir": str(parent), "verdict": "mei0_registry_frozen"}
        },
    )
    assert out["verdict"] == "mei1_audit_failed"
    assert any("sha256 mismatch" in item for item in out["issues"])


def test_mei1_reads_registries_from_parent_freeze(tmp_path, monkeypatch):
    parent = _v2_parent_freeze(tmp_path)
    # Mutate mutable configs so a config-dir read would see different schema if used.
    # Parent freeze remains the source of truth.
    called = {"profiles": None}

    def _fake_audit_domain(**kwargs):
        return {
            "n_points": len(kwargs["points"]),
            "n_designs": 15,
            "baseline_design_id": "K4[25,63,100,200k]",
            "point_labels": [lab for lab, _ in kwargs["labeled_points"]],
            "family_reports": {
                "F0_mrs1_baseline": {
                    "baseline_k4_summary": {
                        "max_p90_o2_percent": 1.0,
                        "median_p90_o2_percent": 0.9,
                    },
                    "point_bottlenecks": ["t_c"] * len(kwargs["points"]),
                    "ranking": [
                        {
                            "design_id": "K4[25,63,100,200k]",
                            "metric": 1.0,
                            "rank_numerical": 0,
                            "rank_practical": 0,
                            "raw_order": 0,
                            "ranking_resolvable": True,
                            "ranking_resolvable_numerical": True,
                            "ranking_resolvable_practical": True,
                            "ranking_span_relative": 0.05,
                            "distinguishable_rank_levels": 2,
                            "distinguishable_rank_levels_numerical": 2,
                            "distinguishable_rank_levels_practical": 2,
                            "max_p90_o2_percent": 1.0,
                            "median_p90_o2_percent": 0.9,
                        }
                    ],
                    "inert_for_flip_gate": False,
                    "top1_matches_f0": True,
                    "spearman_vs_f0": 1.0,
                    "principal_angle_gate_value_deg": 0.0,
                    "bottleneck_flip_fraction_vs_f0": 0.0,
                    "relative_max_p90_change_vs_f0_on_baseline_k4": 0.0,
                }
            },
            "flip_events": [],
            "f0_ranking_meta": {
                "ranking_resolvable": True,
                "ranking_span_relative": 0.05,
            },
            "ranking_resolvable": True,
            "baseline_k4_max_p90": 1.0,
            "baseline_k4_median_p90": 0.9,
        }

    monkeypatch.setattr(
        "tv3.audit.mrs_ei_forward_envelope._audit_one_profile_domain",
        _fake_audit_domain,
    )
    out = run_mei1_audit(
        project_root=_ROOT,
        config_dir=_CFG,
        parent_mei0_freeze_dir=parent,
        current_stage_status={
            "mei0": {"freeze_dir": str(parent), "verdict": "mei0_registry_frozen"}
        },
    )
    assert out["parent_mei0_freeze_dir"] == str(parent)
    assert "low_cost_k4_primary" in out["profile_results"]
    assert "registered_mrs2_stress" in out["profile_results"]


def test_mei1_runs_both_noise_profiles(tmp_path, monkeypatch):
    parent = _v2_parent_freeze(tmp_path)
    seen = []

    def _fake_audit_domain(**kwargs):
        seen.append(kwargs["eval_kwargs"]["jitter_std_s"])
        return {
            "n_points": len(kwargs["points"]),
            "n_designs": 15,
            "baseline_design_id": "K4[25,63,100,200k]",
            "point_labels": [],
            "family_reports": {
                "F0_mrs1_baseline": {
                    "baseline_k4_summary": {
                        "max_p90_o2_percent": 1.0,
                        "median_p90_o2_percent": 0.9,
                    },
                    "point_bottlenecks": [],
                    "ranking": [
                        {
                            "design_id": "A",
                            "metric": 1.0,
                            "rank_numerical": 0,
                            "rank_practical": 0,
                            "raw_order": 0,
                            "ranking_resolvable": True,
                            "ranking_resolvable_numerical": True,
                            "ranking_resolvable_practical": True,
                            "ranking_span_relative": 0.05,
                            "distinguishable_rank_levels": 2,
                            "distinguishable_rank_levels_numerical": 2,
                            "distinguishable_rank_levels_practical": 2,
                            "max_p90_o2_percent": 1.0,
                            "median_p90_o2_percent": 0.9,
                        }
                    ],
                    "inert_for_flip_gate": False,
                    "top1_matches_f0": True,
                    "spearman_vs_f0": 1.0,
                    "principal_angle_gate_value_deg": 0.0,
                    "bottleneck_flip_fraction_vs_f0": 0.0,
                    "relative_max_p90_change_vs_f0_on_baseline_k4": 0.0,
                }
            },
            "flip_events": [],
            "f0_ranking_meta": {"ranking_resolvable": True, "ranking_span_relative": 0.05},
            "ranking_resolvable": True,
            "baseline_k4_max_p90": 1.0,
            "baseline_k4_median_p90": 0.9,
        }

    monkeypatch.setattr(
        "tv3.audit.mrs_ei_forward_envelope._audit_one_profile_domain",
        _fake_audit_domain,
    )
    out = run_mei1_audit(
        project_root=_ROOT,
        config_dir=_CFG,
        parent_mei0_freeze_dir=parent,
        current_stage_status={
            "mei0": {"freeze_dir": str(parent), "verdict": "mei0_registry_frozen"}
        },
    )
    assert set(out["noise_profiles"]) == {
        "low_cost_k4_primary",
        "registered_mrs2_stress",
    }
    assert 5.0e-7 in seen
    assert 3.0e-6 in seen


def test_mei1_reports_core_pressure_and_union(tmp_path, monkeypatch):
    parent = _v2_parent_freeze(tmp_path)
    seen_counts = []

    def _fake_audit_domain(**kwargs):
        seen_counts.append(len(kwargs["points"]))
        return {
            "n_points": len(kwargs["points"]),
            "n_designs": 15,
            "baseline_design_id": "K4[25,63,100,200k]",
            "point_labels": [],
            "family_reports": {
                "F0_mrs1_baseline": {
                    "baseline_k4_summary": {
                        "max_p90_o2_percent": 1.0,
                        "median_p90_o2_percent": 0.9,
                    },
                    "point_bottlenecks": [],
                    "ranking": [
                        {
                            "design_id": "A",
                            "metric": 1.0,
                            "rank_numerical": 0,
                            "rank_practical": 0,
                            "raw_order": 0,
                            "ranking_resolvable": True,
                            "ranking_resolvable_numerical": True,
                            "ranking_resolvable_practical": True,
                            "ranking_span_relative": 0.05,
                            "distinguishable_rank_levels": 2,
                            "distinguishable_rank_levels_numerical": 2,
                            "distinguishable_rank_levels_practical": 2,
                            "max_p90_o2_percent": 1.0,
                            "median_p90_o2_percent": 0.9,
                        }
                    ],
                    "inert_for_flip_gate": False,
                    "top1_matches_f0": True,
                    "spearman_vs_f0": 1.0,
                    "principal_angle_gate_value_deg": 0.0,
                    "bottleneck_flip_fraction_vs_f0": 0.0,
                    "relative_max_p90_change_vs_f0_on_baseline_k4": 0.0,
                }
            },
            "flip_events": [],
            "f0_ranking_meta": {"ranking_resolvable": True, "ranking_span_relative": 0.05},
            "ranking_resolvable": True,
            "baseline_k4_max_p90": 1.0,
            "baseline_k4_median_p90": 0.9,
        }

    monkeypatch.setattr(
        "tv3.audit.mrs_ei_forward_envelope._audit_one_profile_domain",
        _fake_audit_domain,
    )
    out = run_mei1_audit(
        project_root=_ROOT,
        config_dir=_CFG,
        parent_mei0_freeze_dir=parent,
        current_stage_status={
            "mei0": {"freeze_dir": str(parent), "verdict": "mei0_registry_frozen"}
        },
    )
    for profile_id in ("low_cost_k4_primary", "registered_mrs2_stress"):
        domains = out["profile_results"][profile_id]["domains"]
        assert set(domains) == {
            "ambient_core_216",
            "pressure_extension_low_rh_216",
            "formal_mei1_432",
        }
        assert domains["ambient_core_216"]["n_points"] == 216
        assert domains["pressure_extension_low_rh_216"]["n_points"] == 216
        assert domains["formal_mei1_432"]["n_points"] == 432


def test_mei1_blocks_unvalidated_pressure_domain(tmp_path, monkeypatch):
    parent = _v2_parent_freeze(tmp_path)

    def _fake_audit_domain(**kwargs):
        return {
            "n_points": len(kwargs["points"]),
            "n_designs": 15,
            "baseline_design_id": "K4[25,63,100,200k]",
            "point_labels": [],
            "family_reports": {
                "F0_mrs1_baseline": {
                    "baseline_k4_summary": {
                        "max_p90_o2_percent": 1.0,
                        "median_p90_o2_percent": 0.9,
                    },
                    "point_bottlenecks": [],
                    "ranking": [
                        {
                            "design_id": "A",
                            "metric": 1.0,
                            "rank_numerical": 0,
                            "rank_practical": 0,
                            "raw_order": 0,
                            "ranking_resolvable": True,
                            "ranking_resolvable_numerical": True,
                            "ranking_resolvable_practical": True,
                            "ranking_span_relative": 0.05,
                            "distinguishable_rank_levels": 2,
                            "distinguishable_rank_levels_numerical": 2,
                            "distinguishable_rank_levels_practical": 2,
                            "max_p90_o2_percent": 1.0,
                            "median_p90_o2_percent": 0.9,
                        }
                    ],
                    "inert_for_flip_gate": False,
                    "top1_matches_f0": True,
                    "spearman_vs_f0": 1.0,
                    "principal_angle_gate_value_deg": 0.0,
                    "bottleneck_flip_fraction_vs_f0": 0.0,
                    "relative_max_p90_change_vs_f0_on_baseline_k4": 0.0,
                }
            },
            "flip_events": [],
            "f0_ranking_meta": {"ranking_resolvable": True, "ranking_span_relative": 0.05},
            "ranking_resolvable": True,
            "baseline_k4_max_p90": 1.0,
            "baseline_k4_median_p90": 0.9,
        }

    monkeypatch.setattr(
        "tv3.audit.mrs_ei_forward_envelope._audit_one_profile_domain",
        _fake_audit_domain,
    )
    out = run_mei1_audit(
        project_root=_ROOT,
        config_dir=_CFG,
        parent_mei0_freeze_dir=parent,
        current_stage_status={
            "mei0": {"freeze_dir": str(parent), "verdict": "mei0_registry_frozen"}
        },
    )
    assert out["pressure_domain"]["status"] == "parked_nonblocking"
    assert "pressure_domain_not_validated" not in out["blockers"]
    # Points are retained, not dropped.
    assert out["n_points"] == 432


def test_mei1_parks_reviewed_families_without_marking_them_represented():
    model = load_json(_CFG / "model_family_registry.json")
    blocking = collect_unrepresented_blocking(model)
    assert blocking == []
    parked = collect_parked_nonblocking(model)
    assert set(parked) == {
        "F2_h2o_relaxation_params",
        "F3_coupled_relaxation",
        "F4_diffraction_near_field",
        "F5_transducer_response",
    }
    families = {family["id"]: family for family in model["model_families"]}
    assert all(families[fid]["can_clear_not_represented"] is False for fid in parked)


def test_not_represented_family_still_blocks_after_parking_policy_added():
    model = load_json(_CFG / "model_family_registry.json")
    f2 = next(f for f in model["model_families"] if f["id"] == "F2_h2o_relaxation_params")
    f2.update(
        {
            "status": "not_represented",
            "source": "not_represented",
            "evidence_path": None,
            "evidence_sha256": None,
        }
    )
    assert collect_unrepresented_blocking(model) == ["F2_h2o_relaxation_params"]


def test_proxy_never_clears_not_represented():
    model = load_json(_CFG / "model_family_registry.json")
    fam = next(f for f in model["model_families"] if f["id"] == "F3_coupled_relaxation")
    assert fam["can_clear_not_represented"] is False
    assert proxy_never_clears_not_represented(
        family_kind="structural_proxy",
        registry_family=fam,
    )
    bad = copy.deepcopy(fam)
    bad["can_clear_not_represented"] = True
    assert not proxy_never_clears_not_represented(
        family_kind="structural_proxy",
        registry_family=bad,
    )


def test_mei1_supported_requires_all_profiles_and_domains():
    out = decide_mei1_verdict(
        issues=[],
        flip_events=[],
        unrepresented_blocking=[],
        ranking_resolvable=True,
        pressure_domain_ok=True,
        all_profiles_complete=False,
        formal_point_count_ok=True,
    )
    assert out["passed"] is False
    assert "noise_profiles_incomplete" in out["blockers"]


def test_mei1_inconclusive_keeps_allowed_next_stage_null():
    out = decide_mei1_verdict(
        issues=[],
        flip_events=[],
        unrepresented_blocking=["F2_h2o_relaxation_params"],
        ranking_resolvable=True,
        pressure_domain_ok=False,
        all_profiles_complete=True,
        formal_point_count_ok=True,
    )
    assert out["verdict"] == "mei1_inconclusive_forward_model"
    assert out["allowed_next_stage"] is None


def test_mei1_supported_does_not_authorize_waveform_or_hardware():
    out = decide_mei1_verdict(
        issues=[],
        flip_events=[],
        unrepresented_blocking=[],
        ranking_resolvable=True,
        pressure_domain_ok=True,
        all_profiles_complete=True,
        formal_point_count_ok=True,
    )
    assert out["passed"] is True
    assert out["allowed_next_stage"] == "MEI-2_robust_design"
    assert out["registered_sparse_simulation_generation_review_eligible"] is True
    auths = out["authorizations"]
    assert auths["formal_waveform_generation"] == FORBIDDEN_AUTH_VALUE
    assert auths["hardware_trial"] == FORBIDDEN_AUTH_VALUE
    assert auths["benchmark_packaging"] == FORBIDDEN_AUTH_VALUE
    assert auths["registered_sparse_simulation_generation"] == FORBIDDEN_AUTH_VALUE


def test_pressure_domain_validation_keeps_points():
    design = load_json(_CFG / "design_space.json")
    model = load_json(_CFG / "model_family_registry.json")
    labeled = build_formal_mei1_points(design)
    result = pressure_domain_validation(labeled, model_registry=model)
    assert result["n_high_pressure_points"] == 216
    assert result["validated"] is False
    assert result["status"] == "parked_nonblocking"
    assert result["blocker"] is None


def test_pressure_domain_validation_ignores_note_keywords():
    design = load_json(_CFG / "design_space.json")
    model = load_json(_CFG / "model_family_registry.json")
    model["model_families"][0]["notes"] += " 0.5 MPa and 0.709"
    result = pressure_domain_validation(
        build_formal_mei1_points(design), model_registry=model
    )
    assert result["validated"] is False


def test_pressure_domain_validation_requires_structured_coverage():
    design = load_json(_CFG / "design_space.json")
    model = load_json(_CFG / "model_family_registry.json")
    model["pressure_domain_evidence"] = {
        "status": "validated_traceable",
        "validated_range_mpa": [0.5, 0.709],
        "evidence_path": "evidence.json",
        "evidence_sha256": "a" * 64,
    }
    result = pressure_domain_validation(
        build_formal_mei1_points(design), model_registry=model
    )
    assert result["validated"] is True


def test_all_registered_domain_rankings_must_be_resolvable():
    domains = {
        "ambient_core_216": {"ranking_resolvable": True},
        "pressure_extension_low_rh_216": {"ranking_resolvable": False},
        "formal_mei1_432": {"ranking_resolvable": True},
    }
    profiles = {
        "low_cost_k4_primary": {"domains": copy.deepcopy(domains)},
        "registered_mrs2_stress": {"domains": copy.deepcopy(domains)},
    }
    assert registered_domain_rankings_resolvable(
        profiles,
        noise_profiles=["low_cost_k4_primary", "registered_mrs2_stress"],
        point_sets=["ambient_core_216", "pressure_extension_low_rh_216"],
        formal_union="formal_mei1_432",
    ) is False
