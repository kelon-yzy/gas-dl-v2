from __future__ import annotations

import json
import sys
from pathlib import Path

from hg.pipeline import run_n2_improvement_workflow
from hg.pipeline.experiment_config import ExperimentConfig
from hg.pipeline.run_n2_improvement_workflow import format_markdown_plan, planned_steps, run_workflow


def _config(dataset_dir: Path, output_root: Path, *, experiment_name: str = "formal_full") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=experiment_name,
        dataset_dir=dataset_dir,
        output_root=output_root,
        seed=123,
        device="cpu",
        eval_splits=("val", "test"),
        training={},
        ml_runs=(),
        dl_runs=(),
    )


def test_planned_steps_match_formal_n2_workflow(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    output_root = tmp_path / "outputs"
    config_path = tmp_path / "formal_full.json"

    plan = planned_steps(
        config=_config(dataset_dir, output_root),
        config_path=config_path,
    )

    assert [step.name for step in plan.steps] == [
        "inspect_composition_labels",
        "run_experiment_dry_run",
        "run_experiment",
        "analyze_n2_improvement",
        "analyze_phase_aware_n2",
    ]
    assert plan.artifacts == {
        "run_root": str(output_root / "runs" / "formal_full"),
        "composition_label_report": str(output_root / "reports" / "formal_full_composition_labels.md"),
        "composition_label_json": str(output_root / "reports" / "formal_full_composition_labels.json"),
        "n2_improvement_report": str(output_root / "reports" / "formal_full_n2_improvement.md"),
        "n2_improvement_json": str(output_root / "reports" / "formal_full_n2_improvement.json"),
        "phase_aware_n2_report": str(output_root / "reports" / "formal_full_phase_aware_n2.md"),
        "phase_aware_n2_json": str(output_root / "reports" / "formal_full_phase_aware_n2.json"),
    }
    assert plan.steps[0].command == (
        sys.executable,
        "-m",
        "pipeline.inspect_composition_labels",
        "--dataset-dir",
        str(dataset_dir),
        "--output-path",
        str(output_root / "reports" / "formal_full_composition_labels.md"),
        "--json-output-path",
        str(output_root / "reports" / "formal_full_composition_labels.json"),
    )
    assert "--dry-run" in plan.steps[1].command
    assert "--dry-run" not in plan.steps[2].command
    assert "--output-root" in plan.steps[2].command
    assert "--device" in plan.steps[2].command
    assert plan.steps[3].command[-6:] == (
        "--run-root",
        str(output_root / "runs" / "formal_full"),
        "--output-path",
        str(output_root / "reports" / "formal_full_n2_improvement.md"),
        "--json-output-path",
        str(output_root / "reports" / "formal_full_n2_improvement.json"),
    )
    assert plan.steps[4].command[-7:] == (
        "--phase-aware",
        "--run-root",
        str(output_root / "runs" / "formal_full"),
        "--output-path",
        str(output_root / "reports" / "formal_full_phase_aware_n2.md"),
        "--json-output-path",
        str(output_root / "reports" / "formal_full_phase_aware_n2.json"),
    )


def test_planned_steps_use_config_experiment_name_for_default_outputs(tmp_path: Path):
    output_root = tmp_path / "outputs"

    plan = planned_steps(
        config=_config(tmp_path / "dataset", output_root, experiment_name="server_suite"),
        config_path=tmp_path / "server_config.json",
    )

    assert plan.artifacts["run_root"] == str(output_root / "runs" / "server_suite")
    assert plan.artifacts["composition_label_json"] == str(output_root / "reports" / "server_suite_composition_labels.json")
    assert plan.artifacts["phase_aware_n2_json"] == str(output_root / "reports" / "server_suite_phase_aware_n2.json")
    assert plan.steps[3].command[-6:] == (
        "--run-root",
        str(output_root / "runs" / "server_suite"),
        "--output-path",
        str(output_root / "reports" / "server_suite_n2_improvement.md"),
        "--json-output-path",
        str(output_root / "reports" / "server_suite_n2_improvement.json"),
    )
    assert plan.steps[4].command[-1].endswith("server_suite_phase_aware_n2.json")


def test_workflow_dry_run_does_not_execute_commands(tmp_path: Path):
    plan = planned_steps(
        config=_config(tmp_path / "dataset", tmp_path / "outputs"),
        config_path=tmp_path / "config.json",
    )

    payload = run_workflow(plan=plan, execute=False)
    report = format_markdown_plan(payload)

    assert payload["execute"] is False
    assert payload["mode"] == "validate_only"
    assert payload["artifacts"]["n2_improvement_json"].endswith("formal_full_n2_improvement.json")
    assert payload["artifacts"]["phase_aware_n2_json"].endswith("formal_full_phase_aware_n2.json")
    assert "completed_steps" not in payload
    assert "## Artifacts" in report
    assert "inspect_composition_labels" in report
    assert "analyze_n2_improvement" in report
    assert "analyze_phase_aware_n2" in report


def test_workflow_validate_only_cli_matches_default_mode(tmp_path: Path, capsys, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "server_suite",
                "dataset_dir": str(tmp_path / "dataset"),
                "output_root": str(tmp_path / "outputs"),
                "seed": 123,
                "device": "cpu",
                "eval_splits": ["val"],
                "training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "num_workers": 0,
                    "optimizer": "adamw",
                    "lr": 0.001,
                    "weight_decay": 0.0,
                    "loss": "mse",
                    "early_stopping": {"enabled": False},
                    "scheduler": {"name": "none"},
                },
                "ml_runs": [],
                "dl_runs": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_n2_improvement_workflow.sys, "executable", "python")

    exit_code = run_n2_improvement_workflow.main(["--config", str(config_path), "--validate-only", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["mode"] == "validate_only"
    assert payload["execute"] is False
    assert payload["artifacts"]["composition_label_json"].endswith("server_suite_composition_labels.json")
    assert payload["steps"][3]["command"][-3].endswith("server_suite_n2_improvement.md")
    assert payload["steps"][3]["command"][-1].endswith("server_suite_n2_improvement.json")
    assert payload["steps"][4]["command"][-3].endswith("server_suite_phase_aware_n2.md")
    assert payload["steps"][4]["command"][-1].endswith("server_suite_phase_aware_n2.json")


def test_workflow_execute_json_keeps_stdout_parseable(tmp_path: Path, capsys, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment_name": "server_suite",
                "dataset_dir": str(tmp_path / "dataset"),
                "output_root": str(tmp_path / "outputs"),
                "seed": 123,
                "device": "cpu",
                "eval_splits": ["val"],
                "training": {
                    "epochs": 1,
                    "batch_size": 4,
                    "num_workers": 0,
                    "optimizer": "adamw",
                    "lr": 0.001,
                    "weight_decay": 0.0,
                    "loss": "mse",
                    "early_stopping": {"enabled": False},
                    "scheduler": {"name": "none"},
                },
                "ml_runs": [],
                "dl_runs": [],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def fake_run(command, check, **kwargs):
        calls.append((command, check, kwargs))
        stream = kwargs.get("stdout")
        if stream is not None:
            print("child output", file=stream)

    monkeypatch.setattr(run_n2_improvement_workflow.sys, "executable", "python")
    monkeypatch.setattr(run_n2_improvement_workflow.subprocess, "run", fake_run)

    exit_code = run_n2_improvement_workflow.main(["--config", str(config_path), "--execute", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["mode"] == "execute"
    assert payload["completed_steps"] == [
        "inspect_composition_labels",
        "run_experiment_dry_run",
        "run_experiment",
        "analyze_n2_improvement",
        "analyze_phase_aware_n2",
    ]
    assert "[workflow start] inspect_composition_labels" in captured.err
    assert "child output" in captured.err
    assert len(calls) == 5
    assert all(kwargs["stdout"] is kwargs["stderr"] for _command, _check, kwargs in calls)
