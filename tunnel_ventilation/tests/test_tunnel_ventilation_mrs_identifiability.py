"""MRS-2 forward identifiability unit tests."""
from __future__ import annotations

import math

import numpy as np

from tv3.audit.identifiability_v2 import DEFAULT_RANK_RELATIVE_TOL, _relative_svd_rank
from tv3.audit.identifiability_v3_mrs import (
    MrsPoint,
    choose_mrs2_verdict,
    evaluate_point_arm,
    fisher_rank_crb,
    local_mrs_jacobian,
    observation_vector,
    tof_std_s_for_frequency,
)


_BOUNDS = {
    "co2_percent": (0.03, 5.0),
    "o2_percent": (18.0, 21.2),
    "t_c": (15.0, 35.0),
    "path_length_m": (0.2, 0.3),
    "h_rh": (20.0, 80.0),
}
_STEPS = {
    "o2_percent": 0.01,
    "co2_percent": 0.01,
    "t_c": 0.1,
    "path_length_m": 0.001,
    "h_rh": 0.5,
}
_PRIOR = {"t_c": 1.0, "path_length_m": 1e-4, "h_rh": 2.0, "co2_percent": 0.05}
_POINT = MrsPoint(1.0, 20.0, 25.0, 0.25, 50.0, 0.101325)
_F8 = (10000.0, 16000.0, 25000.0, 40000.0, 63000.0, 100000.0, 160000.0, 200000.0)


def _eval(arm: str, f_hz=None):
    return evaluate_point_arm(
        _POINT,
        arm=arm,
        f_hz=f_hz if f_hz is not None else _F8,
        parameter_steps=_STEPS,
        parameter_bounds=_BOUNDS,
        fixed_delay_s=82e-6,
        rh_delta=20.0,
        p_scan_mpa=(0.10, 0.50),
        jitter_std_s=3e-6,
        relative_amp_std=0.02,
        prior_std=_PRIOR,
        window_width_percent=0.8,
    )


def test_relative_svd_rank_imported_convention():
    # 1×n nonzero → rank 1
    j = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float64)
    assert _relative_svd_rank(j, relative_tol=DEFAULT_RANK_RELATIVE_TOL) == 1
    # two independent rows → rank 2
    j2 = np.array([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0]], dtype=np.float64)
    assert _relative_svd_rank(j2) == 2


def test_obs_single_200k_rank_is_one():
    """Negative control: single-frequency acoustic obs must have relative SVD rank 1."""
    out = _eval("obs-single-200k", f_hz=(200000.0,))
    assert out["n_obs"] == 1
    assert out["joint_rank"] == 1
    assert out["rank_upgraded"] is False


def test_obs_cfreq_can_exceed_rank_one():
    out = _eval("obs-cfreq")
    assert out["n_obs"] == 8
    # Multifreq dispersion should lift acoustic rank above the single-TOF baseline.
    assert out["joint_rank"] >= 2


def test_tof_noise_jitter_is_frequency_independent_by_default():
    s200 = tof_std_s_for_frequency(200000.0, jitter_std_s=3e-6)
    s10 = tof_std_s_for_frequency(10000.0, jitter_std_s=3e-6)
    assert abs(s200 - 3e-6) <= 1e-15
    assert abs(s10 - 3e-6) <= 1e-15
    s10_phase = tof_std_s_for_frequency(
        10000.0, jitter_std_s=3e-6, phase_std_s_at_anchor=1e-7
    )
    assert s10_phase > s10



def test_rh_diff_stacks_two_humidity_operating_points():
    y_c, lab_c = observation_vector(
        _POINT,
        arm="obs-calpha",
        f_hz=_F8,
        fixed_delay_s=82e-6,
        rh_delta=20.0,
        p_scan_mpa=(0.1, 0.5),
    )
    y_r, lab_r = observation_vector(
        _POINT,
        arm="obs-rh-diff",
        f_hz=_F8,
        fixed_delay_s=82e-6,
        rh_delta=20.0,
        p_scan_mpa=(0.1, 0.5),
    )
    assert y_c.size * 2 == y_r.size
    assert any(x.startswith("rh1:") for x in lab_r)


def test_fd_stability_flag_present():
    loc = local_mrs_jacobian(
        _POINT,
        arm="obs-cfreq",
        f_hz=_F8,
        parameter_steps=_STEPS,
        parameter_bounds=_BOUNDS,
        fixed_delay_s=82e-6,
        rh_delta=20.0,
        p_scan_mpa=(0.1, 0.5),
    )
    assert "all_stable" in loc
    assert set(loc["parameter_meta"]) == set(_STEPS)


def test_fisher_aug_invertible_with_priors_multifreq():
    out = _eval("obs-cfreq")
    assert out["fisher_aug_invertible"] is True
    assert math.isfinite(out["p90_o2_percent"])
    assert out["p90_o2_percent"] > 0.0


def test_choose_verdict_negative_control_failure():
    summaries = {
        "obs-single-200k": {"min_joint_rank": 2},
        "obs-cfreq": {"min_joint_rank": 2, "max_p90_o2_percent": 0.1, "max_nuisance_fraction": 0.1, "all_crlb_invertible": True},
        "obs-calpha": {"min_joint_rank": 2, "max_p90_o2_percent": 0.1, "max_nuisance_fraction": 0.1, "all_crlb_invertible": True},
        "obs-rh-diff": {"min_joint_rank": 2, "max_p90_o2_percent": 0.1, "max_nuisance_fraction": 0.1, "all_crlb_invertible": True},
        "obs-p-scan": {"min_joint_rank": 2, "max_p90_o2_percent": 0.1, "max_nuisance_fraction": 0.1, "all_crlb_invertible": True},
    }
    d = choose_mrs2_verdict(
        single_200k_min_rank=2,
        arm_summaries=summaries,
        target_p90=0.4,
        max_nuisance_fraction=0.5,
        max_rejection_rate=0.05,
        rejection_rate=0.0,
    )
    assert d["verdict"] == "audit_failed"


def test_choose_verdict_rank_still_deficient():
    summaries = {
        "obs-single-200k": {"min_joint_rank": 1},
        "obs-cfreq": {"min_joint_rank": 1, "max_p90_o2_percent": 9.0, "max_nuisance_fraction": 2.0, "all_crlb_invertible": False},
        "obs-calpha": {"min_joint_rank": 1, "max_p90_o2_percent": 9.0, "max_nuisance_fraction": 2.0, "all_crlb_invertible": False},
        "obs-rh-diff": {"min_joint_rank": 1, "max_p90_o2_percent": 9.0, "max_nuisance_fraction": 2.0, "all_crlb_invertible": False},
        "obs-p-scan": {"min_joint_rank": 1, "max_p90_o2_percent": 9.0, "max_nuisance_fraction": 2.0, "all_crlb_invertible": False},
    }
    d = choose_mrs2_verdict(
        single_200k_min_rank=1,
        arm_summaries=summaries,
        target_p90=0.4,
        max_nuisance_fraction=0.5,
        max_rejection_rate=0.05,
        rejection_rate=0.0,
    )
    assert d["verdict"] == "mrs2_rank_still_deficient"
    assert d["allow_mrs3"] is False


def test_choose_verdict_rank_upgraded_p90_pass():
    summaries = {
        "obs-single-200k": {"min_joint_rank": 1},
        "obs-cfreq": {"min_joint_rank": 3, "max_p90_o2_percent": 0.2, "max_nuisance_fraction": 0.2, "all_crlb_invertible": True},
        "obs-calpha": {"min_joint_rank": 3, "max_p90_o2_percent": 0.2, "max_nuisance_fraction": 0.2, "all_crlb_invertible": True},
        "obs-rh-diff": {"min_joint_rank": 3, "max_p90_o2_percent": 0.2, "max_nuisance_fraction": 0.2, "all_crlb_invertible": True},
        "obs-p-scan": {"min_joint_rank": 3, "max_p90_o2_percent": 0.2, "max_nuisance_fraction": 0.2, "all_crlb_invertible": True},
    }
    d = choose_mrs2_verdict(
        single_200k_min_rank=1,
        arm_summaries=summaries,
        target_p90=0.4,
        max_nuisance_fraction=0.5,
        max_rejection_rate=0.05,
        rejection_rate=0.0,
    )
    assert d["verdict"] == "mrs2_rank_upgraded_p90_pass"
    assert d["allow_mrs3"] is True
