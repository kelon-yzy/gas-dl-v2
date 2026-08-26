import numpy as np
import json
from pathlib import Path

from gib.audit.forward import (
    AuditConfig,
    CANDIDATES,
    ETA_DEFAULT,
    analyze_candidate,
    forward_observation,
    g3_1_forward_audit,
    negative_controls,
    screen_candidate,
    subspace_minimum_angle_deg,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_g3_1_config():
    return json.loads((ROOT / "configs" / "p3_g3_1_forward.json").read_text(encoding="utf-8"))


def test_primary_candidate_is_selected_by_pure_forward_audit():
    screened = screen_candidate()
    assert screened["candidate_verdict"] == "candidate_selected"
    assert all(item["passed"] for item in screened["negative_controls"].values())


def test_schur_complement_differs_from_unmarginalized_target_fisher():
    result = analyze_candidate()
    assert np.all(np.diag(result.effective_fisher) < np.diag(result.target_fisher))
    assert np.all(np.diag(result.crb) > 0)


def test_joint_rank_is_consistent_across_required_tolerances():
    result = analyze_candidate()
    assert len(set(result.ranks_by_tolerance.values())) == 1
    assert result.joint_rank == result.whitened_j_theta.shape[1] + result.whitened_j_eta.shape[1]


def test_modality_metrics_cover_all_physical_components_and_active_blocks():
    result = analyze_candidate()
    assert set(result.modality_blocks) == {"ndir", "acoustic_raw", "thermal", "slow", "calibration"}
    assert all(len(values) == 3 for values in result.modality_sensitivity.values())
    assert all(len(values) == 3 for values in result.modality_effective_information_share.values())
    assert len(result.component_pair_similarity) == 6


def test_slow_observation_matches_the_frozen_measured_channel_contract():
    profile = CANDIDATES["GIB-C4-LR"]
    observation = forward_observation(
        profile.baseline_composition[:-1],
        ETA_DEFAULT,
        AuditConfig(modalities=("slow",)),
    )
    assert observation.labels == (
        "slow_T_K",
        "slow_P_kPa",
        "slow_RH_frac",
        "slow_q_flow",
    )
    assert observation.values[-1] == ETA_DEFAULT[-1]


def test_calibration_observations_are_separate_from_slow_channels():
    profile = CANDIDATES["GIB-C4-LR"]
    observation = forward_observation(
        profile.baseline_composition[:-1],
        ETA_DEFAULT,
        AuditConfig(modalities=("calibration",)),
    )
    assert observation.labels == (
        "calibration_L_m",
        "calibration_gain",
        "calibration_baseline",
        "calibration_delay_s",
        "calibration_crosstalk",
    )


def test_noise_monotonicity_and_modality_off_negative_controls():
    controls = negative_controls()
    assert controls["noise_monotonicity"]["passed"]
    assert controls["modality_off"]["passed"]


def test_raw_and_dsp_are_alternate_not_concurrent_views():
    try:
        analyze_candidate(AuditConfig(modalities=("ndir", "acoustic_raw", "acoustic_dsp", "thermal")))
    except ValueError as error:
        assert "alternate views" in str(error)
    else:
        raise AssertionError("raw and DSP views must not be counted concurrently")


def test_subspace_angle_is_finite():
    result = analyze_candidate()
    angle = subspace_minimum_angle_deg(result.whitened_j_theta, result.whitened_j_eta)
    assert np.isfinite(angle)
    assert 0.0 <= angle <= 90.0


def test_g3_1_forward_audit_passes_every_required_positive_and_negative_control():
    report = g3_1_forward_audit(_load_g3_1_config())
    assert report["gate_verdict"] == "pass"
    assert report["next_allowed_task"] == "P3-02"
    assert all(check["passed"] for check in report["checks"].values())
    assert set(report["checks"]["component_perturbations"]["components"]) == {"N2", "CO2", "O2", "Ar"}
    assert report["checks"]["all_off_negative_control"]["negative_control_only"]
    assert not report["checks"]["all_off_negative_control"]["target_profile_eligible"]


def test_g3_1_forward_audit_is_hash_stable():
    first = g3_1_forward_audit(_load_g3_1_config())
    second = g3_1_forward_audit(_load_g3_1_config())
    assert first == second


def test_g3_1_forward_audit_fails_an_incorrect_preregistered_sign():
    config = _load_g3_1_config()
    config["primary_response"]["CO2"]["sign"] = 1
    report = g3_1_forward_audit(config)
    assert report["gate_verdict"] == "fail"
    assert not report["checks"]["component_perturbations"]["components"]["CO2"]["response_sign_pass"]
