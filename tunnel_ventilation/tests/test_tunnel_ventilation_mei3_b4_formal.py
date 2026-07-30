from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from tv3.audit.mrs_ei_b4_formal import (
    calibrate_option_a_priors,
    evaluate_solver_gate,
    generate_registered_sparse_dataset,
    run_b4_formal_comparison,
)
from tv3.audit.mrs_ei_registry import verify_evidence_manifest
from tv3.audit.mrs_ei_solver_gate import B4_WAITING, assess_b4_execution_authorization

_ROOT = Path(__file__).resolve().parents[1]
_GATE = _ROOT / "configs" / "tv3_mrs_ei" / "mei3_b4_formal_gate.json"
_PROTOCOL = _ROOT / "configs" / "tv3_mrs_ei" / "mei3_solver_data_protocol.json"
_SOLVER = _ROOT / "configs" / "tv3_mrs_ei" / "mei3_solver_audit.json"
_DESIGN = _ROOT / "configs" / "tv3_mrs_ei" / "design_space.json"
_STATUS = _ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _authorized_status():
    status = _load(_STATUS)
    mei3 = status["mei3"]
    mei3["b4_technical_ready"] = True
    mei3["verdict"] = "mei3_pre_b4_technical_ready"
    mei3["authorizations"] = {
        **mei3["authorizations"],
        "registered_sparse_simulation_generation": "authorized",
    }
    mei3["registered_sparse_simulation_generation_authorized"] = True
    return status


def _unauthorized_status():
    status = _authorized_status()
    status["mei3"]["authorizations"]["registered_sparse_simulation_generation"] = (
        "forbidden_until_explicit_authorization"
    )
    status["mei3"]["registered_sparse_simulation_generation_authorized"] = False
    return status


def _load_b4_runner():
    path = _ROOT / "scripts" / "run_tv3_mei3_b4_formal_comparison.py"
    spec = importlib.util.spec_from_file_location("run_tv3_mei3_b4_formal_comparison", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_b4_gate_requires_independent_authorization():
    decision = assess_b4_execution_authorization(
        protocol_config=_load(_PROTOCOL),
        current_stage_status=_unauthorized_status(),
    )
    assert decision["authorized"] is False
    assert decision["verdict"] == B4_WAITING


def test_generate_registered_sparse_respects_view_and_split_contracts():
    dataset = generate_registered_sparse_dataset(
        protocol=_load(_PROTOCOL),
        gate=_load(_GATE),
        design_space=_load(_DESIGN),
        mixture_limit_per_split=2,
    )
    assert dataset["meta"]["view_id"] == "view_0"
    assert set(dataset["meta"]["split_counts"]) == {"calibration", "test", "ood"}
    assert all(row["view_id"] == "view_0" for row in dataset["observation_rows"])
    assert "sequence_id" not in dataset["mixtures"][0]
    mixture_ids = {row["mixture_id"] for row in dataset["mixtures"]}
    assert len(mixture_ids) == len(dataset["mixtures"])
    # 2 conditions × 3 replicates per split
    assert dataset["meta"]["split_counts"]["test"] == 6
    assert dataset["meta"]["split_counts"]["ood"] == 6


def test_option_a_calibration_estimates_shared_priors():
    dataset = generate_registered_sparse_dataset(
        protocol=_load(_PROTOCOL),
        gate=_load(_GATE),
        design_space=_load(_DESIGN),
        mixture_limit_per_split=3,
    )
    calibration = calibrate_option_a_priors(
        records=dataset["records"],
        solver_config=_load(_SOLVER),
        gate=_load(_GATE),
    )
    truth = dataset["meta"]["shared_truth_nuisance"]
    view = calibration["view_nuisance_calibration_priors"][0]
    assert abs(view["common_delay_prior_mean"] - truth["common_delay_s"]) < 5e-6
    assert abs(view["log_amplitude_gain_prior_mean"] - truth["log_amplitude_gain"]) < 0.05
    assert len(calibration["calibration_priors"]) == 4


def test_b4_smoke_paired_comparison_runs_when_authorized():
    result = run_b4_formal_comparison(
        project_root=_ROOT,
        protocol=_load(_PROTOCOL),
        gate=_load(_GATE),
        solver_config=_load(_SOLVER),
        design_space=_load(_DESIGN),
        stage_status=_authorized_status(),
        mixture_limit_per_split=1,
        bootstrap_n_resamples_override=20,
    )
    assert result["authorized"] is True
    assert result["formal_data_generated"] is True
    assert result["smoke_mode"] is True
    assert result["gate_report"]["verdict"] in {
        "mei3_varpro_supported",
        "mei3_full_parameter_baseline_retained",
    }
    assert {"test", "ood"} <= set(result["gate_report"]["domain_reports"])
    assert all(
        set(row) >= {"S1", "S2", "S3", "mixture_id", "split"}
        for row in result["paired_rows"]
    )
    # Known T/L/RH conditions must be joined; S3 with truth nuisances should recover O2.
    s3_errors = [row["S3"]["abs_err_o2"] for row in result["paired_rows"]]
    assert max(s3_errors) < 2.0
    s1_errors = [row["S1"]["abs_err_o2"] for row in result["paired_rows"]]
    assert max(s1_errors) < 5.0


def test_b4_runner_refuses_without_auth_and_freezes_smoke(tmp_path, monkeypatch):
    runner = _load_b4_runner()
    stage_path = tmp_path / "stage_status.json"
    stage_path.write_text(json.dumps(_unauthorized_status()), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_b4_formal_comparison.py",
            "--stage-status-path",
            str(stage_path),
        ],
    )
    assert runner.main() == 5

    stage_path.write_text(json.dumps(_authorized_status()), encoding="utf-8")
    output_dir = tmp_path / "freeze"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tv3_mei3_b4_formal_comparison.py",
            "--stage-status-path",
            str(stage_path),
            "--output-dir",
            str(output_dir),
            "--mixture-limit-per-split",
            "1",
            "--bootstrap-n-resamples",
            "16",
            "--allow-incomplete-smoke",
        ],
    )
    assert runner.main() == 0
    assert (output_dir / "solver_comparison.csv").is_file()
    assert (output_dir / "bootstrap_report.json").is_file()
    assert (output_dir / "s3_truth_nuisance.json").is_file()
    assert verify_evidence_manifest(
        output_dir / "evidence_manifest.json", project_root=_ROOT
    ) == []
    # Smoke mode must not rewrite formal stage status verdict.
    after = json.loads(stage_path.read_text(encoding="utf-8"))
    assert after["mei3"]["verdict"] == "mei3_pre_b4_technical_ready"


def test_b4_status_promotion_preserves_pre_b4_parent_on_rerun(tmp_path):
    runner = _load_b4_runner()
    stage_path = tmp_path / "stage_status.json"
    status = _authorized_status()
    pre_b4_manifest = "a" * 64
    status["mei3"].pop("parent_pre_b4_manifest_sha256", None)
    status["mei3"]["phase"] = "b3_registered_data_authorization"
    status["mei3"]["evidence_manifest_sha256"] = pre_b4_manifest
    stage_path.write_text(json.dumps(status), encoding="utf-8")
    result = {
        "smoke_mode": False,
        "gate_report": {
            "verdict": "mei3_full_parameter_baseline_retained",
            "passed_solver_gate": False,
        },
    }

    runner._promote_stage_status(
        stage_path,
        result=result,
        freeze_dir="outputs/runs/first",
        manifest_sha256="b" * 64,
        created_at_utc="2026-07-30T00:00:00+00:00",
    )
    runner._promote_stage_status(
        stage_path,
        result=result,
        freeze_dir="outputs/runs/second",
        manifest_sha256="c" * 64,
        created_at_utc="2026-07-30T00:00:01+00:00",
    )

    promoted = _load(stage_path)["mei3"]
    assert promoted["evidence_manifest_sha256"] == "c" * 64
    assert promoted["parent_pre_b4_manifest_sha256"] == pre_b4_manifest


def test_non_degeneration_uses_configured_absolute_thresholds():
    gate = _load(_GATE)
    gate["solver_gate"]["delta_practical"] = 0.01
    protocol = _load(_PROTOCOL)
    protocol["initialization_and_statistics"]["bootstrap"]["n_resamples"] = 8
    paired_rows = []
    for split in ("test", "ood"):
        for index, s1_o2 in enumerate((1.0, 1.1, 1.2)):
            s1 = {
                "abs_err_o2": s1_o2,
                "abs_err_co2": 0.1,
                "abs_err_n2": 0.1,
                "success": True,
                "bound_hit": False,
                "iterations": 1,
                "forward_calls": 1,
                "relative_crb_efficiency_o2": 1.0,
            }
            s2 = {**s1, "abs_err_o2": s1_o2 * 0.9, "abs_err_co2": 0.121}
            paired_rows.append(
                {
                    "split": split,
                    "design_condition_id": f"{split}_{index}",
                    "S1": s1,
                    "S2": s2,
                    "S3": s2,
                }
            )

    report = evaluate_solver_gate(paired_rows, protocol=protocol, gate=gate)

    assert report["passed_solver_gate"] is False
    assert any("mae_co2 absolute increase" in issue for issue in report["issues"])
    non_degeneration = report["domain_reports"]["test"]["non_degeneration"]
    assert non_degeneration["semantics"] == "maximum_absolute_increase"
    assert non_degeneration["thresholds"]["mae_co2"] == 0.02
    assert abs(non_degeneration["observed_increases"]["mae_co2"] - 0.021) < 1e-12
