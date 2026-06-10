from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pipeline.experiment_config import ExperimentConfig, load_experiment_config


DEFAULT_CONFIG_PATH = Path("configs/experiment/formal_full.json")


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    name: str
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkflowPlan:
    steps: tuple[WorkflowStep, ...]
    artifacts: dict[str, str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the formal N2 ALR/ILR improvement workflow.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Optional dataset_dir override.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Experiment JSON config.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional output root override for run_experiment.")
    parser.add_argument("--run-root", type=Path, default=None, help="Run root consumed by analyze_n2_improvement.")
    parser.add_argument("--report-path", type=Path, default=None, help="Markdown report path for N2 analysis.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override for run_experiment.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", default=False, help="Validate config and print planned commands without executing them.")
    mode.add_argument("--execute", action="store_true", default=False, help="Execute the workflow instead of printing it.")
    parser.add_argument("--json", action="store_true", default=False, help="Print JSON instead of Markdown.")
    return parser


def planned_steps(
    *,
    config: ExperimentConfig,
    config_path: Path,
    run_root: Path | None = None,
    report_path: Path | None = None,
) -> WorkflowPlan:
    label_report_path = config.output_root / "reports" / f"{config.experiment_name}_composition_labels.md"
    label_json_report_path = label_report_path.with_suffix(".json")
    effective_run_root = run_root or config.output_root / "runs" / config.experiment_name
    effective_report_path = report_path or config.output_root / "reports" / f"{config.experiment_name}_n2_improvement.md"
    effective_json_report_path = effective_report_path.with_suffix(".json")

    run_experiment_base = [
        sys.executable,
        "-m",
        "pipeline.run_experiment",
        "--config",
        str(config_path),
        "--dataset-dir",
        str(config.dataset_dir),
        "--output-root",
        str(config.output_root),
        "--device",
        config.device,
    ]

    return WorkflowPlan(
        steps=(
            WorkflowStep(
                name="inspect_composition_labels",
                command=(
                    sys.executable,
                    "-m",
                    "pipeline.inspect_composition_labels",
                    "--dataset-dir",
                    str(config.dataset_dir),
                    "--output-path",
                    str(label_report_path),
                    "--json-output-path",
                    str(label_json_report_path),
                ),
            ),
            WorkflowStep(
                name="run_experiment_dry_run",
                command=tuple((*run_experiment_base, "--dry-run")),
            ),
            WorkflowStep(
                name="run_experiment",
                command=tuple(run_experiment_base),
            ),
            WorkflowStep(
                name="analyze_n2_improvement",
                command=(
                    sys.executable,
                    "-m",
                    "pipeline.analyze_n2_improvement",
                    "--run-root",
                    str(effective_run_root),
                    "--output-path",
                    str(effective_report_path),
                    "--json-output-path",
                    str(effective_json_report_path),
                ),
            ),
        ),
        artifacts={
            "run_root": str(effective_run_root),
            "composition_label_report": str(label_report_path),
            "composition_label_json": str(label_json_report_path),
            "n2_improvement_report": str(effective_report_path),
            "n2_improvement_json": str(effective_json_report_path),
        },
    )


def run_workflow(
    *,
    plan: WorkflowPlan,
    execute: bool,
    log_stream: Any | None = None,
    subprocess_output_stream: Any | None = None,
) -> dict[str, Any]:
    log_stream = sys.stdout if log_stream is None else log_stream
    payload: dict[str, Any] = {
        "execute": execute,
        "mode": "execute" if execute else "validate_only",
        "artifacts": plan.artifacts,
        "steps": [_step_payload(step) for step in plan.steps],
    }
    if not execute:
        return payload

    completed = []
    for step in plan.steps:
        print(f"[workflow start] {step.name}", file=log_stream, flush=True)
        subprocess_kwargs = {}
        if subprocess_output_stream is not None:
            subprocess_kwargs = {"stdout": subprocess_output_stream, "stderr": subprocess_output_stream}
        subprocess.run(step.command, check=True, **subprocess_kwargs)
        print(f"[workflow done] {step.name}", file=log_stream, flush=True)
        completed.append(step.name)
    payload["completed_steps"] = completed
    return payload


def format_markdown_plan(payload: dict[str, Any]) -> str:
    lines = [
        "# N2 Improvement Workflow",
        "",
        f"- execute: `{str(payload['execute']).lower()}`",
        "",
        "## Artifacts",
        "",
        "| artifact | path |",
        "|---|---|",
    ]
    for name, path in payload["artifacts"].items():
        lines.append(f"| {name} | `{path}` |")
    lines.extend(
        [
            "",
            "## Steps",
            "",
            "| step | command |",
            "|---|---|",
        ]
    )
    for step in payload["steps"]:
        lines.append(f"| {step['name']} | `{_format_command(step['command'])}` |")
    if "completed_steps" in payload:
        lines.extend(["", "## Completed", ""])
        lines.extend(f"- {name}" for name in payload["completed_steps"])
    return "\n".join(lines).rstrip() + "\n"


def _step_payload(step: WorkflowStep) -> dict[str, Any]:
    return {
        "name": step.name,
        "command": list(step.command),
    }


def _format_command(command: Sequence[str]) -> str:
    return " ".join(command).replace("\\", "\\\\")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_experiment_config(
        args.config,
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        device=args.device,
    )
    plan = planned_steps(
        config=config,
        config_path=args.config,
        run_root=args.run_root,
        report_path=args.report_path,
    )
    log_stream = sys.stderr if args.json else sys.stdout
    subprocess_output_stream = sys.stderr if args.json and args.execute else None
    payload = run_workflow(
        plan=plan,
        execute=args.execute,
        log_stream=log_stream,
        subprocess_output_stream=subprocess_output_stream,
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_markdown_plan(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
