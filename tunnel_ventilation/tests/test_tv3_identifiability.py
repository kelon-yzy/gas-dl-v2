"""tv3 单向 TOF 可辨识性审计测试。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tv3.audit.error_budget import combined_p90_o2_error_percent, equivalent_o2_std_percent
from tv3.audit.identifiability import AcousticPoint, fisher_information, local_tof_sensitivity


BOUNDS = {
    "co2_percent": (0.03, 5.0),
    "o2_percent": (18.0, 21.2),
    "t_c": (15.0, 35.0),
    "path_length_m": (0.2, 0.3),
}
STEPS = {"co2_percent": 0.01, "o2_percent": 0.01, "t_c": 0.1, "path_length_m": 0.001}


def _load_runner_module():
    project_root = Path(__file__).resolve().parents[1]
    module_name = "test_tv3_identifiability_runner_module"
    spec = importlib.util.spec_from_file_location(module_name, project_root / "scripts" / "run_tv3_identifiability.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _derivatives(point: AcousticPoint):
    return local_tof_sensitivity(
        point,
        parameter_steps=STEPS,
        parameter_bounds=BOUNDS,
        fixed_delay_s=82e-6,
        max_relative_step_disagreement=0.01,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(project_root: Path, manifest_sha256: str) -> dict:
    return {
        "schema_version": "tv3-identifiability-1",
        "baseline": {
            "dir": "outputs/tv3_baseline_freeze",
            "manifest_sha256": manifest_sha256,
            "expected_contract": {
                "component_fields": ["x_CO2", "x_O2", "x_N2"],
                "output": "raw3",
                "feature_builder": "d0_raw_dsp_physics_stats_v1",
                "feature_count": 1008,
            },
        },
        "output_dir": "outputs/tv3_identifiability",
        "parameter_bounds": {key: list(value) for key, value in BOUNDS.items()},
        "global_grid": {"co2_percent": [0.03], "o2_percent": [18.0], "t_c": [25.0], "path_length_m": [0.25]},
        "narrow_windows": [{"id": "o2_20", "center_percent": 20.0, "width_percent": 0.8}],
        "narrow_context_grid": {"co2_percent": [0.03], "t_c": [25.0], "path_length_m": [0.25]},
        "finite_difference": {"steps": STEPS, "max_relative_step_disagreement": 0.01},
        "observation": {"fixed_delay_s": 82e-6, "tof_std_s": 3e-6},
        "uncertainty_scenarios": [
            {"id": "temperature", "parameter": "t_c", "std": 1.0, "source": "test"},
            {"id": "jitter", "parameter": "tof_s", "std": 3e-6, "source": "test"},
        ],
        "uncertainty_model": {"type": "diagonal", "source": "test"},
        "representation_audit": [
            {
                "name": "flow_projection",
                "unit": "m/s",
                "representation": "not_represented",
                "source": "test",
                "distribution": "not_available",
                "correlation_group": "flow",
                "deployable_observable": False,
                "v1_representation": "not_represented",
                "blocks_go_verdict": True,
            }
        ],
        "business_thresholds": {
            "target_p90_o2_error_percent": None,
            "max_nuisance_fraction_of_signal": None,
            "max_rejection_rate": None,
        },
    }


def test_boundary_difference_preserves_closure_and_is_marked_forward():
    point = AcousticPoint(co2_percent=1.0, o2_percent=18.0, t_c=25.0, path_length_m=0.25)
    derivatives = _derivatives(point)

    assert point.n2_percent == pytest.approx(81.0)
    assert derivatives["o2_percent"]["scheme"] == "forward"
    assert derivatives["o2_percent"]["stable"] is True


def test_single_tof_fisher_does_not_claim_nuisance_marginalization():
    result = fisher_information(_derivatives(AcousticPoint(1.0, 20.0, 25.0, 0.25)), tof_std_s=3e-6)

    assert result["conditional_o2_information"] > 0.0
    assert result["joint_rank"] == 1
    assert result["nuisance_marginalized_status"] == "unavailable_rank_deficient"


def test_error_budget_converts_and_combines_equivalent_o2_error():
    equivalent = equivalent_o2_std_percent(
        tof_per_o2_s_per_percent=2e-7,
        tof_per_nuisance_s_per_unit=1e-6,
        nuisance_std=0.2,
    )

    assert equivalent == pytest.approx(1.0)
    assert combined_p90_o2_error_percent([equivalent, equivalent]) > 1.0


def test_runner_writes_inconclusive_verdict_without_business_threshold(tmp_path):
    runner = _load_runner_module()
    baseline_dir = tmp_path / "outputs" / "tv3_baseline_freeze"
    manifest_path = baseline_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "contract": {
                "component_fields": ["x_CO2", "x_O2", "x_N2"],
                "output": "raw3",
                "feature_builder": "d0_raw_dsp_physics_stats_v1",
                "feature_count": 1008,
            }
        },
    )
    _write_json(baseline_dir / "verdict.json", {"status": "frozen"})
    config_path = tmp_path / "configs" / "tv3_identifiability.json"
    _write_json(config_path, _config(tmp_path, _sha256(manifest_path)))

    output_dir = runner.run_identifiability(config_path, project_root=tmp_path)

    verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "inconclusive_missing_business_threshold"
    assert audit["status"] == "passed"
    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_identifiability(config_path, project_root=tmp_path)


def test_runner_reports_configured_accuracy_and_nuisance_gate_failures(tmp_path):
    runner = _load_runner_module()
    baseline_dir = tmp_path / "outputs" / "tv3_baseline_freeze"
    manifest_path = baseline_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "contract": {
                "component_fields": ["x_CO2", "x_O2", "x_N2"],
                "output": "raw3",
                "feature_builder": "d0_raw_dsp_physics_stats_v1",
                "feature_count": 1008,
            }
        },
    )
    _write_json(baseline_dir / "verdict.json", {"status": "frozen"})
    config = _config(tmp_path, _sha256(manifest_path))
    config["business_thresholds"]["target_p90_o2_error_percent"] = 0.4
    config["business_thresholds"]["max_nuisance_fraction_of_signal"] = 0.5
    config_path = tmp_path / "configs" / "tv3_identifiability.json"
    _write_json(config_path, config)

    output_dir = runner.run_identifiability(config_path, project_root=tmp_path)

    verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert verdict["status"] == "inconclusive_missing_business_threshold"
    assert verdict["blocking_fields"] == ["max_rejection_rate"]
    assert verdict["business_gate_assessment"]["target_p90_o2_error_percent"]["status"] == "failed"
    assert verdict["business_gate_assessment"]["max_nuisance_fraction_of_signal"]["status"] == "failed"
    assert metrics["business_gate_assessment"] == verdict["business_gate_assessment"]
    assert (output_dir / "nuisance_fraction_summary.csv").is_file()


def test_runner_rejects_all_points_for_unrepresented_blocking_nuisance(tmp_path):
    runner = _load_runner_module()
    baseline_dir = tmp_path / "outputs" / "tv3_baseline_freeze"
    manifest_path = baseline_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "contract": {
                "component_fields": ["x_CO2", "x_O2", "x_N2"],
                "output": "raw3",
                "feature_builder": "d0_raw_dsp_physics_stats_v1",
                "feature_count": 1008,
            }
        },
    )
    _write_json(baseline_dir / "verdict.json", {"status": "frozen"})
    config = _config(tmp_path, _sha256(manifest_path))
    config["business_thresholds"] = {
        "target_p90_o2_error_percent": 0.4,
        "max_nuisance_fraction_of_signal": 0.5,
        "max_rejection_rate": 0.05,
    }
    config["rejection_policy"] = {
        "id": "reject_blocking_nuisance",
        "source": "test",
        "reject_if_unrepresented_blocking_nuisance": True,
    }
    config_path = tmp_path / "configs" / "tv3_identifiability.json"
    _write_json(config_path, config)

    output_dir = runner.run_identifiability(config_path, project_root=tmp_path)

    verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    rejection_gate = verdict["business_gate_assessment"]["max_rejection_rate"]
    assert verdict["status"] == "information_source_upgrade_required"
    assert verdict["blocking_nuisances"] == ["flow_projection"]
    assert rejection_gate["evaluated_point_count"] == 2
    assert rejection_gate["rejected_point_count"] == 2
    assert rejection_gate["observed_rejection_rate"] == pytest.approx(1.0)
    assert rejection_gate["status"] == "failed"


def test_runner_rejects_incomplete_representation_audit(tmp_path):
    runner = _load_runner_module()
    config = _config(tmp_path, "not-used")
    del config["representation_audit"][0]["distribution"]

    with pytest.raises(ValueError, match="representation audit is missing"):
        runner._validate_config(config)


def test_runner_rejects_nuisance_threshold_below_one_percent(tmp_path):
    runner = _load_runner_module()
    config = _config(tmp_path, "not-used")
    config["business_thresholds"]["max_nuisance_fraction_of_signal"] = 0.009

    with pytest.raises(ValueError, match="within \\[0.01, 1.0\\]"):
        runner._validate_config(config)


def test_runner_rejects_rejection_rate_without_policy(tmp_path):
    runner = _load_runner_module()
    config = _config(tmp_path, "not-used")
    config["business_thresholds"]["max_rejection_rate"] = 0.05

    with pytest.raises(ValueError, match="rejection_policy must contain exactly"):
        runner._validate_config(config)
