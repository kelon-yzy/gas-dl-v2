"""F4 tests: bidirectional identifiability v2 Fisher and deploy hygiene."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tv3.audit.identifiability_v2 import (
    BidirAcousticPoint,
    fisher_information_bidir,
    local_bidir_tof_sensitivity,
    midpair_tof_std_s,
    observed_bidir_tof_s,
)


BOUNDS = {
    "co2_percent": (0.03, 5.0),
    "o2_percent": (18.0, 21.2),
    "t_c": (15.0, 35.0),
    "path_length_m": (0.2, 0.3),
    "v_path_m_per_s": (-4.0, 4.0),
}
STEPS = {
    "co2_percent": 0.01,
    "o2_percent": 0.01,
    "t_c": 0.1,
    "path_length_m": 0.001,
    "v_path_m_per_s": 0.01,
}


def _derivatives(point: BidirAcousticPoint):
    return local_bidir_tof_sensitivity(
        point,
        parameter_steps=STEPS,
        parameter_bounds=BOUNDS,
        fixed_delay_s=82e-6,
        max_relative_step_disagreement=0.01,
    )


def _load_runner():
    project_root = Path(__file__).resolve().parents[1]
    name = "test_tv3_identifiability_v2_runner"
    spec = importlib.util.spec_from_file_location(
        name, project_root / "scripts" / "run_tv3_identifiability_v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_bidir_tof_depends_on_signed_flow():
    point = BidirAcousticPoint(1.0, 20.0, 20.0, 0.25, v_path_m_per_s=2.0)
    t_ab, t_ba = observed_bidir_tof_s(point, fixed_delay_s=82e-6)
    assert t_ab < t_ba
    zero = BidirAcousticPoint(1.0, 20.0, 20.0, 0.25, v_path_m_per_s=0.0)
    z_ab, z_ba = observed_bidir_tof_s(zero, fixed_delay_s=82e-6)
    assert z_ab == pytest.approx(z_ba, abs=1e-15)


def test_bidir_fisher_acoustic_subsystem_full_rank():
    derivatives = _derivatives(BidirAcousticPoint(1.0, 20.0, 25.0, 0.25, 1.0))
    result = fisher_information_bidir(derivatives, tof_std_s=3e-6, temperature_std_c=1.0)
    assert result["joint_rank"] >= 2
    assert result["acoustic_subsystem_full_rank"] is True
    assert result["conditional_o2_information"] > 0.0


def test_v_path_sensitivity_is_differential_mode():
    derivatives = _derivatives(BidirAcousticPoint(1.0, 20.0, 20.0, 0.25, 0.0))
    d_ab = float(derivatives["v_path_m_per_s"]["derivative_tof_ab_s_per_unit"])
    d_ba = float(derivatives["v_path_m_per_s"]["derivative_tof_ba_s_per_unit"])
    d_mid = float(derivatives["v_path_m_per_s"]["derivative_tof_mid_s_per_unit"])
    assert d_ab * d_ba < 0.0
    assert abs(d_mid) < abs(d_ab) * 1e-6


def test_midpair_std_is_half_variance():
    assert midpair_tof_std_s(3e-6) == pytest.approx(3e-6 / (2**0.5))


def test_runner_produces_coarse_or_continuous_without_flow_block(tmp_path):
    runner = _load_runner()
    project = tmp_path
    registry = {
        "schema_version": "tunnel-ventilation-bidir-1",
        "trigger_jitter_scenarios": {"scenarios": []},
    }
    registry_path = project / "configs" / "tv3_bidir" / "parameter_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    import hashlib

    sha = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    f3_dir = project / "outputs" / "tv3_bidir" / "dsp_fidelity"
    f3_dir.mkdir(parents=True)
    (f3_dir / "f3_verdict.json").write_text(
        json.dumps(
            {
                "verdict": "f3_dsp_passed",
                "feature_builder": "raw_dsp_bidirectional_v1",
                "delay_calibration_digest": "abc",
            }
        ),
        encoding="utf-8",
    )
    config = {
        "schema_version": "tv3-identifiability-bidir-2",
        "output_dir": "outputs/tv3_bidir/identifiability_v2",
        "f0_registry": {"path": "configs/tv3_bidir/parameter_registry.json", "expected_sha256": sha},
        "f3_prerequisite": {
            "verdict_path": "outputs/tv3_bidir/dsp_fidelity/f3_verdict.json",
            "expected_verdict": "f3_dsp_passed",
            "feature_builder": "raw_dsp_bidirectional_v1",
        },
        "parameter_bounds": {k: list(v) for k, v in BOUNDS.items()},
        "global_grid": {
            "co2_percent": [1.0],
            "o2_percent": [20.0],
            "t_c": [20.0],
            "path_length_m": [0.25],
            "v_path_m_per_s": [0.0, 1.0],
        },
        "narrow_windows": [{"id": "o2_20_0", "center_percent": 20.0, "width_percent": 0.8}],
        "narrow_context_grid": {
            "co2_percent": [1.0],
            "t_c": [20.0],
            "path_length_m": [0.25],
            "v_path_m_per_s": [0.0],
        },
        "finite_difference": {"steps": STEPS, "max_relative_step_disagreement": 0.01},
        "observation": {
            "fixed_delay_s": 82e-6,
            "source": "test",
            "jitter_correlation": "independent",
            "temperature_std_c_for_fisher": 1.0,
            "sequence_frames_for_prior_crosscheck": 64,
        },
        "jitter_scenarios": [
            {"id": "conservative_v1", "std_s": 3e-6, "source": "test"},
            {"id": "nominal_daq_half_sample", "std_s": 5e-7, "source": "test"},
        ],
        "shared_uncertainty_scenarios": [
            {"id": "temperature_1K_scenario", "parameter": "t_c", "std": 1.0, "source": "test"}
        ],
        "uncertainty_model": {"type": "diagonal", "source": "test"},
        "representation_audit": [
            {
                "name": "flow_projection",
                "unit": "m/s",
                "representation": "implemented_physics",
                "source": "test",
                "distribution": "test",
                "correlation_group": "flow",
                "deployable_observable": True,
                "v1_representation": "not_represented",
                "blocks_go_verdict": False,
            }
        ],
        "business_thresholds": {
            "target_p90_o2_error_percent": 0.4,
            "max_nuisance_fraction_of_signal": 0.5,
            "max_rejection_rate": 0.05,
        },
        "rejection_policy": {
            "id": "f4_blocking_nuisance_reject_all",
            "source": "test",
            "reject_if_unrepresented_blocking_nuisance": True,
        },
        "prior_crosscheck": {
            "reference_point": {
                "co2_percent": 1.0,
                "o2_percent": 20.0,
                "t_c": 20.0,
                "path_length_m": 0.25,
                "v_path_m_per_s": 0.0,
            },
            "expected_single_frame_o2_vol_percent": {
                "jitter_3us": 5.8,
                "jitter_0p5us": 0.97,
                "temperature_1K": 2.4,
            },
            "relative_tolerance": 0.5,
            "notes": "test",
        },
        "f5_amplitude_gate_preregistration": {
            "a3_minus_a1_o2_mae_min_vol_percent": 0.5,
            "a3_minus_a2_o2_mae_max_vol_percent": 0.25,
            "v_path_zero_anchor_delta_mae_max_vol_percent": 0.05,
            "selector_r2_noninferior_delta": -0.01,
            "notes": "test",
        },
    }
    config_path = project / "configs" / "tv3_bidir_identifiability_v2.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    out = runner.run_identifiability_v2(config_path, project_root=project)
    verdict = json.loads((out / "f4_verdict.json").read_text(encoding="utf-8"))
    assert verdict["passed"] is True
    assert verdict["verdict"] in {
        "coarse_monitoring_only",
        "continuous_regression_supported",
    }
    assert "flow_projection" not in json.dumps(verdict.get("blocking_nuisances", []))
    assert (out / "conservative_v1" / "fisher_information.csv").is_file()
    assert (out / "nominal_daq_half_sample" / "fisher_information.csv").is_file()
    # v1 path must not be written by this runner
    assert not (project / "outputs" / "tv3_identifiability").exists()
