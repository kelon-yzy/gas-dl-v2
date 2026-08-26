"""Command-line entry point for the GIB contract utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .audit.forward import g3_1_forward_audit
from .audit.grid import g3_2_grid_audit
from .common.io import atomic_write_json, sha256_file as common_sha256_file
from .freeze import freeze_attempt, verify_evidence_manifest
from .pipeline.baseline import run_baselines
from .pipeline.candidate_review import review_candidates
from .pipeline.data_efficiency import run_data_efficiency
from .pipeline.adaptive_sampling import run_c5a_from_pilot
from .pipeline.conditional_solver import run_figs, run_ic_rdu_vp
from .pipeline.multiview import run_multiview
from .pipeline.solver_preflight import execute_solver_preflight
from .pipeline.teacher_preflight import run_teacher_preflight
from .sim.pilot import build_pilot_dataset


def _input_binding(value: str) -> tuple[str, Path]:
    role, separator, path = value.partition("=")
    if not separator or not role or not path:
        raise argparse.ArgumentTypeError("input must use ROLE=PATH")
    return role, Path(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze", help="promote one complete attempt")
    freeze_parser.add_argument("--workspace-root", type=Path, required=True)
    freeze_parser.add_argument("--attempt-dir", type=Path, required=True)
    freeze_parser.add_argument("--freeze-root", type=Path, required=True)
    freeze_parser.add_argument("--freeze-id", required=True)
    freeze_parser.add_argument("--input", action="append", type=_input_binding, required=True)
    freeze_parser.add_argument("--source-snapshot", action="append", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify-freeze", help="recompute a freeze manifest")
    verify_parser.add_argument("freeze_dir", type=Path)

    forward_parser = subparsers.add_parser("audit-forward", help="run the P3 G3-1 forward audit")
    forward_parser.add_argument("--config", type=Path, required=True)
    forward_parser.add_argument("--attempt-dir", type=Path, required=True)

    grid_parser = subparsers.add_parser("audit-grid", help="run the P3 G3-2 grid audit")
    grid_parser.add_argument("--config", type=Path, required=True)
    grid_parser.add_argument("--g3-1-freeze", type=Path, required=True)
    grid_parser.add_argument("--attempt-dir", type=Path, required=True)

    pilot_parser = subparsers.add_parser("pilot-generate", help="generate a P3 pilot or technical dry-run")
    pilot_parser.add_argument("--config", type=Path, required=True)
    pilot_parser.add_argument("--g3-1-freeze", type=Path, required=True)
    pilot_parser.add_argument("--g3-2-freeze", type=Path, required=True)
    pilot_parser.add_argument("--attempt-dir", type=Path, required=True)
    pilot_parser.add_argument("--dry-run", action="store_true")

    baseline_parser = subparsers.add_parser("run-baselines", help="run frozen P3 baselines and G3-3")
    baseline_parser.add_argument("--config", type=Path, required=True)
    baseline_parser.add_argument("--pilot-freeze", type=Path, required=True)
    baseline_parser.add_argument("--attempt-dir", type=Path, required=True)

    multiview_parser = subparsers.add_parser("run-multiview", help="run frozen P3 C4 multiview evaluation")
    multiview_parser.add_argument("--config", type=Path, required=True)
    multiview_parser.add_argument("--pilot-freeze", type=Path, required=True)
    multiview_parser.add_argument("--attempt-dir", type=Path, required=True)

    solver_parser = subparsers.add_parser("run-solver-preflight", help="run frozen P3 C2 solver preflight")
    solver_parser.add_argument("--config", type=Path, required=True)
    solver_parser.add_argument("--pilot-freeze", type=Path, required=True)
    solver_parser.add_argument("--execution-registry", type=Path, required=True)
    solver_parser.add_argument("--git-commit", required=True)
    solver_parser.add_argument("--attempt-dir", type=Path, required=True)

    efficiency_parser = subparsers.add_parser("run-data-efficiency", help="run frozen P3 C5-B evaluation")
    efficiency_parser.add_argument("--config", type=Path, required=True)
    efficiency_parser.add_argument("--pilot-freeze", type=Path, required=True)
    efficiency_parser.add_argument("--execution-registry", type=Path, required=True)
    efficiency_parser.add_argument("--git-commit", required=True)
    efficiency_parser.add_argument("--attempt-dir", type=Path, required=True)
    efficiency_parser.add_argument("--resume", action="store_true")

    sampling_parser = subparsers.add_parser("run-adaptive-sampling", help="run frozen P3 C5-A evaluation")
    sampling_parser.add_argument("--config", type=Path, required=True)
    sampling_parser.add_argument("--pilot-freeze", type=Path, required=True)
    sampling_parser.add_argument("--baseline-freeze", type=Path, required=True)
    sampling_parser.add_argument("--baseline-plan", type=Path, required=True)
    sampling_parser.add_argument("--attempt-dir", type=Path, required=True)

    ic_rdu_parser = subparsers.add_parser("run-ic-rdu-vp", help="run frozen P3 C2 conditional solver evaluation")
    ic_rdu_parser.add_argument("--config", type=Path, required=True)
    ic_rdu_parser.add_argument("--solver-plan", type=Path, required=True)
    ic_rdu_parser.add_argument("--activation-freeze", type=Path, required=True)
    ic_rdu_parser.add_argument("--pilot-freeze", type=Path, required=True)
    ic_rdu_parser.add_argument("--execution-registry", type=Path, required=True)
    ic_rdu_parser.add_argument("--git-commit", required=True)
    ic_rdu_parser.add_argument("--attempt-dir", type=Path, required=True)

    figs_parser = subparsers.add_parser("run-figs", help="run frozen P3 C5-D physical solver routing")
    figs_parser.add_argument("--config", type=Path, required=True)
    figs_parser.add_argument("--solver-plan", type=Path, required=True)
    figs_parser.add_argument("--activation-freeze", type=Path, required=True)
    figs_parser.add_argument("--pilot-freeze", type=Path, required=True)
    figs_parser.add_argument("--execution-registry", type=Path, required=True)
    figs_parser.add_argument("--git-commit", required=True)
    figs_parser.add_argument("--attempt-dir", type=Path, required=True)

    teacher_parser = subparsers.add_parser("run-teacher-preflight", help="run frozen P3 C5-C raw-teacher activation gate")
    teacher_parser.add_argument("--config", type=Path, required=True)
    teacher_parser.add_argument("--baseline-plan", type=Path, required=True)
    teacher_parser.add_argument("--pilot-freeze", type=Path, required=True)
    teacher_parser.add_argument("--baseline-freeze", type=Path, required=True)
    teacher_parser.add_argument("--attempt-dir", type=Path, required=True)

    review_parser = subparsers.add_parser("review-candidates", help="derive P3 G3-4 from verified candidate freezes")
    review_parser.add_argument("--candidate-freeze", action="append", type=_input_binding, required=True)
    review_parser.add_argument("--attempt-dir", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _audit_forward(config_path: Path, attempt_dir: Path) -> int:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    if attempt_dir.exists():
        raise FileExistsError(f"attempt directory already exists: {attempt_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = g3_1_forward_audit(config)
    bound_paths = {
        "config": config_path,
        "s2_s3_config": config_path.parent / str(config["s2_s3_config"]),
        "source_registry": config_path.parent / str(config["source_registry"]),
        "execution_registry": config_path.parent / str(config["execution_registry"]),
        "p2_freeze_manifest": project_root / str(config["p2_freeze_manifest"]),
        "forward_code": Path(__file__).parent / "audit" / "forward.py",
    }
    report["provenance"] = {
        name: {"path": path.relative_to(project_root).as_posix(), "sha256": _sha256_file(path)}
        for name, path in bound_paths.items()
    }
    attempt_dir.mkdir(parents=True)
    report_path = attempt_dir / "forward_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempt_manifest = {
        "schema_version": "gib-benchmark-1",
        "attempt_id": attempt_dir.name,
        "task_id": "P3-01",
        "status": "complete",
        "gate_verdict": report["gate_verdict"],
        "claim_scope": report["claim_scope"],
        "evidence_files": [
            {"path": report_path.name, "sha256": _sha256_file(report_path)},
        ],
        "next_allowed_task": report["next_allowed_task"],
    }
    (attempt_dir / "attempt_manifest.json").write_text(
        json.dumps(attempt_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"attempt_dir": str(attempt_dir), "gate_verdict": report["gate_verdict"]}, sort_keys=True))
    return 0 if report["gate_verdict"] == "pass" else 1


def _audit_grid(config_path: Path, g3_1_freeze: Path, attempt_dir: Path) -> int:
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    workspace_root = project_root.parent
    if attempt_dir.exists():
        raise FileExistsError(f"attempt directory already exists: {attempt_dir}")
    frozen_grid = json.loads(config_path.read_text(encoding="utf-8"))
    report = g3_2_grid_audit(frozen_grid)
    g3_1_verification = verify_evidence_manifest(g3_1_freeze)

    table_path = workspace_root / "docs" / "p2" / "generated" / "s1_grid_table.md"
    renderer_path = workspace_root / "docs" / "p2" / "tools" / "render_s1_grid_table.py"
    expected_table = runpy.run_path(str(renderer_path))["render"]()
    actual_table = table_path.read_text(encoding="utf-8")
    table_match = expected_table == actual_table
    report["checks"]["generated_markdown_exact_match"] = {
        "expected_sha256": hashlib.sha256(expected_table.encode("utf-8")).hexdigest().upper(),
        "actual_sha256": _sha256_file(table_path),
        "passed": table_match,
    }
    if not table_match:
        report["gate_verdict"] = "fail"
        report["next_allowed_task"] = "P2-06"
    report["g3_1_input"] = g3_1_verification
    bound_paths = {
        "grid_config": config_path,
        "grid_code": Path(__file__).parent / "audit" / "grid.py",
        "forward_code": Path(__file__).parent / "audit" / "forward.py",
        "generated_table": table_path,
        "table_renderer": renderer_path,
        "g3_1_evidence_manifest": g3_1_freeze.resolve() / "evidence_manifest.json",
    }
    report["provenance"] = {
        name: {"path": path.relative_to(workspace_root).as_posix(), "sha256": _sha256_file(path)}
        for name, path in bound_paths.items()
    }
    attempt_dir.mkdir(parents=True)
    report_path = attempt_dir / "grid_audit.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempt_manifest = {
        "schema_version": "gib-benchmark-1",
        "attempt_id": attempt_dir.name,
        "task_id": "P3-02",
        "status": "complete",
        "gate_verdict": report["gate_verdict"],
        "claim_scope": report["claim_scope"],
        "evidence_files": [{"path": report_path.name, "sha256": _sha256_file(report_path)}],
        "next_allowed_task": report["next_allowed_task"],
    }
    (attempt_dir / "attempt_manifest.json").write_text(
        json.dumps(attempt_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"attempt_dir": str(attempt_dir), "gate_verdict": report["gate_verdict"]}, sort_keys=True))
    return 0 if report["gate_verdict"] == "pass" else 1


def _pilot_generate(
    config_path: Path,
    g3_1_freeze: Path,
    g3_2_freeze: Path,
    attempt_dir: Path,
    *,
    dry_run: bool,
) -> int:
    config_path = config_path.resolve()
    plan = json.loads(config_path.read_text(encoding="utf-8"))
    g3_1 = verify_evidence_manifest(g3_1_freeze)
    g3_2 = verify_evidence_manifest(g3_2_freeze)
    raw_dsp_path = Path(__file__).parent / "pipeline" / "raw_dsp.py"
    summary = build_pilot_dataset(
        plan,
        config_root=config_path.parent,
        output_dir=attempt_dir,
        dry_run=dry_run,
        raw_dsp_code_sha256=common_sha256_file(raw_dsp_path),
    )
    task_id = "P3-03" if dry_run else "P3-04"
    attempt_manifest = {
        "schema_version": "gib-benchmark-1",
        "attempt_id": attempt_dir.name,
        "task_id": task_id,
        "status": "complete",
        "task_status": "completed",
        "pilot_plan_status": "frozen",
        "pilot_integrity": summary["pilot_integrity"],
        "gate_verdict": "pass" if not dry_run and summary["pilot_integrity"] == "pass" else "not_applicable",
        "claim_scope": summary["claim_scope"],
        "input_freezes": [g3_1, g3_2],
        "config_sha256": common_sha256_file(config_path),
        "generation_summary_ref": "generation_summary.json",
        "next_allowed_task": "P3-04" if dry_run else "P3-05",
    }
    atomic_write_json(attempt_dir / "attempt_manifest.json", attempt_manifest)
    print(
        json.dumps(
            {
                "attempt_dir": str(attempt_dir),
                "task_id": task_id,
                "task_status": "completed",
                "sequence_count": summary["sequence_count"],
            },
            sort_keys=True,
        )
    )
    return 0


def _run_baselines(config_path: Path, pilot_freeze: Path, attempt_dir: Path) -> int:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run_baselines(
        config,
        config_sha256=common_sha256_file(config_path),
        pilot_freeze=pilot_freeze,
        output_dir=attempt_dir,
    )
    print(
        json.dumps(
            {
                "attempt_dir": str(attempt_dir),
                "task_id": "P3-05",
                "gate_verdict": result["gate_verdict"],
                "baseline_sufficient": result["baseline_sufficient"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["gate_verdict"] == "pass" else 1


def _run_multiview(config_path: Path, pilot_freeze: Path, attempt_dir: Path) -> int:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    result = run_multiview(config, pilot_freeze=pilot_freeze, output_dir=attempt_dir)
    print(json.dumps({"attempt_dir": str(attempt_dir), "candidate_verdict": result["candidate_verdict"]}, sort_keys=True))
    return 0


def _run_solver_preflight(
    config_path: Path,
    pilot_freeze: Path,
    execution_registry_path: Path,
    git_commit: str,
    attempt_dir: Path,
) -> int:
    config = json.loads(config_path.resolve().read_text(encoding="utf-8"))
    registry = json.loads(execution_registry_path.resolve().read_text(encoding="utf-8"))
    result = execute_solver_preflight(
        config,
        pilot_freeze=pilot_freeze,
        execution_registry=registry,
        git_commit=git_commit,
        output_dir=attempt_dir,
    )
    print(json.dumps({"attempt_dir": str(attempt_dir), "c2_preflight": result["c2_preflight"]}, sort_keys=True))
    return 0


def _run_data_efficiency(
    config_path: Path,
    pilot_freeze: Path,
    execution_registry_path: Path,
    git_commit: str,
    attempt_dir: Path,
    *,
    resume: bool,
) -> int:
    config_path = config_path.resolve()
    execution_registry_path = execution_registry_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    execution_registry = json.loads(execution_registry_path.read_text(encoding="utf-8"))
    result = run_data_efficiency(
        config,
        config_sha256=common_sha256_file(config_path),
        pilot_freeze=pilot_freeze,
        execution_registry=execution_registry,
        execution_registry_sha256=common_sha256_file(execution_registry_path),
        git_commit=git_commit,
        output_dir=attempt_dir,
        resume=resume,
    )
    print(json.dumps({"attempt_dir": str(attempt_dir), "candidate_verdict": result["candidate_verdict"]}, sort_keys=True))
    return 0


def _run_adaptive_sampling(
    config_path: Path,
    pilot_freeze: Path,
    baseline_freeze: Path,
    baseline_plan_path: Path,
    attempt_dir: Path,
) -> int:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_plan_path"] = str(config_path)
    result = run_c5a_from_pilot(
        config,
        pilot_freeze=pilot_freeze,
        baseline_freeze=baseline_freeze,
        baseline_plan_path=baseline_plan_path.resolve(),
        output_dir=attempt_dir,
    )
    print(json.dumps({"attempt_dir": str(attempt_dir), "candidate_verdict": result["candidate_verdict"]}, sort_keys=True))
    return 0


def _run_ic_rdu_vp(
    config_path: Path,
    solver_plan_path: Path,
    activation_freeze: Path,
    pilot_freeze: Path,
    execution_registry_path: Path,
    git_commit: str,
    attempt_dir: Path,
) -> int:
    config_path = config_path.resolve()
    solver_plan_path = solver_plan_path.resolve()
    execution_registry_path = execution_registry_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    solver_plan = json.loads(solver_plan_path.read_text(encoding="utf-8"))
    execution_registry = json.loads(execution_registry_path.read_text(encoding="utf-8"))
    config["_plan_path"] = str(config_path)
    solver_plan["_plan_path"] = str(solver_plan_path)
    result = run_ic_rdu_vp(
        config,
        solver_plan,
        activation_freeze=activation_freeze,
        pilot_freeze=pilot_freeze,
        execution_registry=execution_registry,
        git_commit=git_commit,
        output_dir=attempt_dir,
    )
    print(json.dumps({"attempt_dir": str(attempt_dir), "candidate_verdict": result["candidate_verdict"]}, sort_keys=True))
    return 0


def _run_figs(
    config_path: Path,
    solver_plan_path: Path,
    activation_freeze: Path,
    pilot_freeze: Path,
    execution_registry_path: Path,
    git_commit: str,
    attempt_dir: Path,
) -> int:
    config_path = config_path.resolve()
    solver_plan_path = solver_plan_path.resolve()
    execution_registry_path = execution_registry_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    solver_plan = json.loads(solver_plan_path.read_text(encoding="utf-8"))
    execution_registry = json.loads(execution_registry_path.read_text(encoding="utf-8"))
    config["_plan_path"] = str(config_path)
    solver_plan["_plan_path"] = str(solver_plan_path)
    result = run_figs(
        config,
        solver_plan,
        activation_freeze=activation_freeze,
        pilot_freeze=pilot_freeze,
        execution_registry=execution_registry,
        git_commit=git_commit,
        output_dir=attempt_dir,
    )
    print(json.dumps({"attempt_dir": str(attempt_dir), "candidate_verdict": result["candidate_verdict"]}, sort_keys=True))
    return 0


def _run_teacher_preflight(
    config_path: Path,
    baseline_plan_path: Path,
    pilot_freeze: Path,
    baseline_freeze: Path,
    attempt_dir: Path,
) -> int:
    config_path = config_path.resolve()
    baseline_plan_path = baseline_plan_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    baseline_plan = json.loads(baseline_plan_path.read_text(encoding="utf-8"))
    config["_plan_path"] = str(config_path)
    result = run_teacher_preflight(
        config,
        baseline_plan,
        pilot_freeze=pilot_freeze,
        baseline_freeze=baseline_freeze,
        output_dir=attempt_dir,
    )
    print(json.dumps({"attempt_dir": str(attempt_dir), "candidate_verdict": result["candidate_verdict"]}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify-freeze":
        print(json.dumps(verify_evidence_manifest(args.freeze_dir), sort_keys=True))
        return 0
    if args.command == "audit-forward":
        return _audit_forward(args.config, args.attempt_dir)
    if args.command == "audit-grid":
        return _audit_grid(args.config, args.g3_1_freeze, args.attempt_dir)
    if args.command == "pilot-generate":
        return _pilot_generate(
            args.config,
            args.g3_1_freeze,
            args.g3_2_freeze,
            args.attempt_dir,
            dry_run=args.dry_run,
        )
    if args.command == "run-baselines":
        return _run_baselines(args.config, args.pilot_freeze, args.attempt_dir)
    if args.command == "run-multiview":
        return _run_multiview(args.config, args.pilot_freeze, args.attempt_dir)
    if args.command == "run-solver-preflight":
        return _run_solver_preflight(
            args.config,
            args.pilot_freeze,
            args.execution_registry,
            args.git_commit,
            args.attempt_dir,
        )
    if args.command == "run-data-efficiency":
        return _run_data_efficiency(
            args.config,
            args.pilot_freeze,
            args.execution_registry,
            args.git_commit,
            args.attempt_dir,
            resume=args.resume,
        )
    if args.command == "run-adaptive-sampling":
        return _run_adaptive_sampling(
            args.config,
            args.pilot_freeze,
            args.baseline_freeze,
            args.baseline_plan,
            args.attempt_dir,
        )
    if args.command == "run-ic-rdu-vp":
        return _run_ic_rdu_vp(
            args.config,
            args.solver_plan,
            args.activation_freeze,
            args.pilot_freeze,
            args.execution_registry,
            args.git_commit,
            args.attempt_dir,
        )
    if args.command == "run-figs":
        return _run_figs(
            args.config,
            args.solver_plan,
            args.activation_freeze,
            args.pilot_freeze,
            args.execution_registry,
            args.git_commit,
            args.attempt_dir,
        )
    if args.command == "run-teacher-preflight":
        return _run_teacher_preflight(
            args.config,
            args.baseline_plan,
            args.pilot_freeze,
            args.baseline_freeze,
            args.attempt_dir,
        )
    if args.command == "review-candidates":
        candidates = {candidate_id: path for candidate_id, path in args.candidate_freeze}
        result = review_candidates(candidates, args.attempt_dir)
        print(json.dumps({"attempt_dir": str(args.attempt_dir), "gate_verdict": result["gate_verdict"]}, sort_keys=True))
        return 0

    bindings: dict[str, list[Path]] = defaultdict(list)
    for role, path in args.input:
        bindings[role].append(path)
    target = freeze_attempt(
        workspace_root=args.workspace_root,
        attempt_dir=args.attempt_dir,
        freeze_root=args.freeze_root,
        freeze_id=args.freeze_id,
        input_files=bindings,
        source_snapshots=args.source_snapshot,
    )
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
