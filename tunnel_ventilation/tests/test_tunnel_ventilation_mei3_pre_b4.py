from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tv3.audit.mrs_ei_registry import verify_evidence_manifest
from tv3.audit.mrs_ei_solver_gate import (
    B4_WAITING,
    PRE_B4_READY,
    assess_b4_execution_authorization,
    run_mei3_pre_b4_readiness_audit,
)
from tv3.ml.mrs_varpro import (
    S1Parameterization,
    build_b1_synthetic_problem,
    build_s1_settings,
    build_varpro_parameterization,
    pack_s1_parameters,
    run_pre_b4_technical_audit,
    solve_s1,
    solve_s2,
)

_ROOT = Path(__file__).resolve().parents[1]
_SOLVER_CONFIG = _ROOT / "configs" / "tv3_mrs_ei" / "mei3_solver_audit.json"
_PROTOCOL = _ROOT / "configs" / "tv3_mrs_ei" / "mei3_solver_data_protocol.json"
_STATUS = _ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
_PARENT_B3 = (
    _ROOT
    / "outputs"
    / "runs"
    / "tv3_mrs_ei"
    / "mei3_varpro_audit"
    / "freezes"
    / "20260729T104111344740Z_b435a5b57d0f"
)


def _solver_config():
    return json.loads(_SOLVER_CONFIG.read_text(encoding="utf-8"))


def _protocol():
    return json.loads(_PROTOCOL.read_text(encoding="utf-8"))


def _b3_stage_status():
    status = json.loads(_STATUS.read_text(encoding="utf-8"))
    parent = json.loads((_PARENT_B3 / "mei3_b3_verdict.json").read_text(encoding="utf-8"))
    audit = parent["audit"]
    status["allowed_next_stage"] = None
    status["mei3"] = {
        "phase": audit["phase"],
        "verdict": audit["verdict"],
        "passed": audit["passed"],
        "freeze_dir": _PARENT_B3.relative_to(_ROOT).as_posix(),
        "b3_authorization_ready_package": True,
        "authorizations": audit["authorizations"],
        "parent_b2_manifest_sha256": audit["parent_b2_manifest_sha256"],
    }
    return status


def _load_pre_b4_runner():
    path = _ROOT / "scripts" / "run_tv3_mei3_pre_b4_technical_audit.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei3_pre_b4_technical_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_b4_runner():
    path = _ROOT / "scripts" / "run_tv3_mei3_b4_formal_comparison.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei3_b4_formal_comparison", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_option_a_truth_recovery_and_crb_pass_pre_b4_gate():
    report = run_pre_b4_technical_audit(_solver_config())
    assert report["technical_ready"] is True
    assert abs(report["truth_recovery"]["wide_prior_o2_error_pp"]["S1"]) > 1.0
    assert abs(report["truth_recovery"]["option_a_o2_error_pp"]["S1"]) <= 0.5
    assert abs(report["truth_recovery"]["option_a_o2_error_pp"]["S3"]) < 1e-6
    assert report["relative_crb"]["crb_o2_std_percent"] > 0.0
    assert report["bound_failure_recording"]["raised_exception"] is False
    assert report["bound_failure_recording"]["s1_bound_hit"] is True
    assert report["bound_failure_recording"]["s2_bound_hit"] is True


def test_bound_hit_is_recordable_failure_not_exception():
    config = _solver_config()
    problem, spec, _truth = build_b1_synthetic_problem(config)
    settings = build_s1_settings(config)
    varpro = build_varpro_parameterization(config, spec)
    initial = pack_s1_parameters(
        config["b1_solver_audit"]["frozen_initializations"][0]["raw3_percent"],
        t_c=24.8,
        path_length_m=0.2498,
        h_rh=48.0,
        common_delay_s=0.0,
        log_amplitude_gain=0.0,
        per_frequency_offsets=[0.0, 0.0, 0.0, 0.0],
    )
    lower = spec.lower_bounds.copy()
    upper = spec.upper_bounds.copy()
    gain_index = list(spec.names).index("log_amplitude_gain")
    lower[gain_index] = -0.005
    upper[gain_index] = 0.005
    bound_spec = S1Parameterization(
        names=spec.names,
        scales=spec.scales.copy(),
        lower_bounds=lower,
        upper_bounds=upper,
        finite_difference_steps=spec.finite_difference_steps.copy(),
        prior_indices=spec.prior_indices.copy(),
        prior_mean=spec.prior_mean.copy(),
        prior_std=spec.prior_std.copy(),
    )
    s1 = solve_s1(problem, initial, bound_spec, settings)
    s2 = solve_s2(
        problem,
        initial,
        bound_spec,
        settings,
        varpro,
        max_phase_branch_standardized_error=8.0,
    )
    assert s1.bound_hit is True
    assert s2.success is False
    assert s2.bound_hit is True


def test_pre_b4_contract_audit_requires_b3_and_option_a_tables():
    technical = run_pre_b4_technical_audit(_solver_config())
    audit = run_mei3_pre_b4_readiness_audit(
        project_root=_ROOT,
        solver_config=_solver_config(),
        protocol_config=_protocol(),
        parent_b3_freeze_dir=_PARENT_B3,
        current_stage_status=_b3_stage_status(),
        technical_report=technical,
    )
    assert audit["passed"] is True
    assert audit["verdict"] == PRE_B4_READY
    assert audit["b4_technical_ready"] is True
    assert audit["formal_solver_gate_ready"] is False
    assert "authorization" in audit["formal_solver_gate_blocker"]


def test_b4_authorization_gate_blocks_until_explicit_authorization():
    status = _b3_stage_status()
    status["mei3"]["verdict"] = PRE_B4_READY
    status["mei3"]["b4_technical_ready"] = True
    decision = assess_b4_execution_authorization(
        protocol_config=_protocol(),
        current_stage_status=status,
    )
    assert decision["authorized"] is False
    assert decision["verdict"] == B4_WAITING

    status["mei3"]["authorizations"] = {
        **status["mei3"]["authorizations"],
        "registered_sparse_simulation_generation": "authorized",
    }
    authorized = assess_b4_execution_authorization(
        protocol_config=_protocol(),
        current_stage_status=status,
    )
    assert authorized["authorized"] is True


def test_pre_b4_runner_freezes_and_b4_runner_refuses_without_auth(tmp_path, monkeypatch):
    runner = _load_pre_b4_runner()
    output_dir = tmp_path / "freeze"
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_b3_stage_status()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_pre_b4_technical_audit.py",
            "--parent-b3-freeze-dir",
            str(_PARENT_B3),
            "--output-dir",
            str(output_dir),
            "--stage-status-path",
            str(stage_path),
        ],
    )
    assert runner.main() == 0
    after = json.loads(stage_path.read_text(encoding="utf-8"))
    assert after["mei3"]["verdict"] == PRE_B4_READY
    assert after["mei3"]["b4_technical_ready"] is True
    assert after["mei3"]["formal_solver_gate_ready"] is False
    assert verify_evidence_manifest(output_dir / "evidence_manifest.json", project_root=_ROOT) == []

    b4 = _load_b4_runner()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_b4_formal_comparison.py",
            "--stage-status-path",
            str(stage_path),
        ],
    )
    assert b4.main() == 5
