from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tv3.audit.mrs_ei_posterior_gate import (
    coverage_with_rejections,
    crps_from_samples,
    run_posterior_core_audit,
)
from tv3.audit.mrs_ei_registry import verify_evidence_manifest
from tv3.ml.mrs_posterior import (
    PosteriorConstructionError,
    laplace_from_jacobian,
    raw3_from_tangent,
    require_fixed_method_settings,
    require_method_payload,
    sample_nonnegative_tangent_gaussian,
    standard_normal_quantiles,
    tangent_from_raw3,
)

_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs" / "tv3_mrs_ei" / "mei4_posterior_audit.json"
_STATUS = _ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runner():
    path = _ROOT / "scripts" / "run_tv3_mei4_c1_posterior_audit.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei4_c1_posterior_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_c0_runner():
    path = _ROOT / "scripts" / "run_tv3_mei4_c0_contract_freeze.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei4_c0_contract_freeze", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tangent_map_schur_and_nonnegative_truncation_are_consistent():
    raw3 = np.asarray([20.0, 30.0, 50.0])
    assert np.allclose(raw3_from_tangent(tangent_from_raw3(raw3)), raw3)
    posterior = laplace_from_jacobian(np.eye(2), mean_z=tangent_from_raw3(raw3))
    samples = sample_nonnegative_tangent_gaussian(
        posterior, candidates=4096, minimum_accepted=100, seed=13
    )
    assert samples.mass_estimate == 1.0
    assert np.all(samples.raw3_percent >= 0.0)
    schur_jacobian = np.random.default_rng(23).normal(size=(12, 5))
    schur_posterior = laplace_from_jacobian(
        schur_jacobian,
        mean_z=(0.0, 0.0),
        composition_scales=(0.5, 2.0),
    )
    expected = np.linalg.inv(schur_jacobian.T @ schur_jacobian)[:2, :2]
    expected *= np.outer([0.5, 2.0], [0.5, 2.0])
    assert schur_posterior.covariance_standardized.shape == (5, 5)
    assert np.allclose(schur_posterior.covariance_z, expected, atol=1e-12)
    with pytest.raises(PosteriorConstructionError, match="curvature_not_positive_definite"):
        laplace_from_jacobian(np.asarray([[1.0, 0.0], [0.0, 0.0]]), mean_z=(0.0, 0.0))


def test_metrics_count_rejections_as_uncovered():
    report = coverage_with_rejections(
        [0.0, 0.0, 0.0], [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]], [False, True, False]
    )
    assert report == {"n": 3, "covered": 2, "rejected": 1, "coverage": 2.0 / 3.0}
    assert crps_from_samples([-1.0, 0.0, 1.0], 0.0) >= 0.0


def test_sobol_endpoint_quantiles_remain_finite():
    normal = standard_normal_quantiles(np.asarray([[0.0, 1.0], [0.5, 0.25]]))
    assert np.all(np.isfinite(normal))


def test_c1_core_audit_passes_and_negative_controls_fail_explicitly():
    report = run_posterior_core_audit(_load(_CONFIG))
    assert report["passed"] is True
    assert report["linear_gaussian"]["schur_covariance_max_abs_error"] <= _load(_CONFIG)[
        "linear_gaussian"
    ]["schur_covariance_atol"]
    assert report["metrics"]["nll"] == pytest.approx(
        report["metrics"]["nll_analytic"], abs=1e-12
    )
    assert report["metrics"]["crps"] == pytest.approx(
        report["metrics"]["crps_analytic"], abs=0.015
    )
    assert report["metrics"]["rejection_coverage"] == {
        "n": 3,
        "covered": 2,
        "rejected": 1,
        "coverage": 2.0 / 3.0,
    }
    assert report["negative_controls"] == {
        "truth_field": "explicit_failure",
        "nonpositive_curvature": "explicit_failure",
        "phase_branch": "explicit_failure",
        "recalibration": "explicit_failure",
    }
    with pytest.raises(PosteriorConstructionError):
        require_method_payload({"truth_raw3_percent": [1.0, 2.0, 97.0]})
    with pytest.raises(PosteriorConstructionError):
        require_fixed_method_settings({"temperature_scaling": 1.0})


def test_c1_runner_freezes_verified_synthetic_audit(tmp_path, monkeypatch):
    runner = _load_runner()
    c0_runner = _load_c0_runner()
    stage_path = tmp_path / "stage_status.json"
    status = _load(_STATUS)
    status.pop("mei4", None)
    stage_path.write_text(json.dumps(status), encoding="utf-8")
    c0_output_dir = tmp_path / "c0_freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei4_c0_contract_freeze.py",
            "--stage-status-path",
            str(stage_path),
            "--output-dir",
            str(c0_output_dir),
        ],
    )
    assert c0_runner.main() == 0

    first_output_dir = tmp_path / "c1_freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei4_c1_posterior_audit.py",
            "--stage-status-path",
            str(stage_path),
            "--output-dir",
            str(first_output_dir),
        ],
    )

    assert runner.main() == 0
    assert verify_evidence_manifest(first_output_dir / "evidence_manifest.json", project_root=_ROOT) == []
    report = _load(first_output_dir / "mei4_posterior_core_report.json")
    assert report["status"] == "mei4_posterior_core_verified"
    assert _load(stage_path)["mei4"]["phase"] == "c1_posterior_core_audit"

    appended_output_dir = tmp_path / "c1_appended_freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei4_c1_posterior_audit.py",
            "--stage-status-path",
            str(stage_path),
            "--output-dir",
            str(appended_output_dir),
        ],
    )
    assert runner.main() == 0
    assert verify_evidence_manifest(appended_output_dir / "evidence_manifest.json", project_root=_ROOT) == []
    manifest = _load(appended_output_dir / "evidence_manifest.json")
    assert Path(manifest["parent_manifest_path"]).resolve() == (
        first_output_dir / "evidence_manifest.json"
    ).resolve()
    assert _load(stage_path)["mei4"]["freeze_dir"] == appended_output_dir.resolve().as_posix()
