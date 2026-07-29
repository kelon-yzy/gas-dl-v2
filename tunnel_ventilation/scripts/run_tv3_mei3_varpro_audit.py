#!/usr/bin/env python3
"""Run and freeze the MEI-3 B0 representation and VarPro structure audit."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.mrs_ei_registry import (  # noqa: E402
    FREEZE_MANIFEST_SCHEMA_VERSION,
    dumps_stable,
    load_json,
    sha256_bytes,
    sha256_file,
    verify_evidence_manifest,
)
from tv3.audit.mrs_ei_varpro import run_mei3_b0_audit  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-mei1-freeze-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stage-status-path", type=Path, default=None)
    return parser.parse_args()


def _relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_TV3_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_TV3_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _git_relevant_paths_dirty() -> bool:
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "configs/tv3_mrs_ei",
            "tv3/audit/mrs_ei_varpro.py",
            "scripts/run_tv3_mei3_varpro_audit.py",
            "tests/test_tunnel_ventilation_mei3_varpro.py",
        ],
        cwd=_TV3_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _promote_stage_status(
    path: Path,
    *,
    audit: dict[str, object],
    freeze_dir: str,
    manifest_sha256: str,
    created_at_utc: str,
    config_sha256: str,
) -> None:
    status = load_json(path)
    status["allowed_next_stage"] = audit["allowed_next_stage"]
    status["mei3"] = {
        "phase": audit["phase"],
        "verdict": audit["verdict"],
        "passed": audit["passed"],
        "freeze_dir": freeze_dir,
        "verdict_path": f"{freeze_dir}/mei3_verdict.json",
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha256,
        "completed_at_utc": created_at_utc,
        "admitted_linear_blocks": audit["admitted_linear_blocks"],
        "formal_solver_gate_ready": audit["formal_solver_gate_ready"],
        "formal_solver_gate_blocker": audit["formal_solver_gate_blocker"],
        "authorizations": audit["authorizations"],
        "parent_mei1_manifest_sha256": audit["parent_mei1_manifest_sha256"],
        "config_sha256": config_sha256,
    }
    temp = path.with_name(f".{path.name}.tmp")
    if temp.exists():
        raise FileExistsError(f"stage status staging path exists: {temp}")
    temp.write_bytes(dumps_stable(status).encode("utf-8"))
    temp.replace(path)


def main() -> int:
    args = _parse_args()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei3_varpro_audit.json"
    )
    stage_status_path = (
        args.stage_status_path.resolve()
        if args.stage_status_path is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    )
    parent_dir = args.parent_mei1_freeze_dir.resolve()
    if not parent_dir.is_dir():
        print(f"parent MEI-1 freeze missing: {parent_dir}", file=sys.stderr)
        return 3

    config = load_json(config_path)
    current_status = load_json(stage_status_path)
    audit = run_mei3_b0_audit(
        project_root=_TV3_ROOT,
        config=config,
        parent_mei1_freeze_dir=parent_dir,
        current_stage_status=current_status,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["issues"]:
        return 3
    if not audit["passed"]:
        return 2

    created_at = datetime.now(timezone.utc)
    audit_code_path = _TV3_ROOT / "tv3" / "audit" / "mrs_ei_varpro.py"
    forward_path = (
        _TV3_ROOT
        / "tv3"
        / "sim"
        / "generation"
        / "tunnel_ventilation"
        / "relaxation_spectrum.py"
    )
    observation_path = (
        _TV3_ROOT
        / "tv3"
        / "sim"
        / "generation"
        / "tunnel_ventilation"
        / "mrs_observation.py"
    )
    tests_path = _TV3_ROOT / "tests" / "test_tunnel_ventilation_mei3_varpro.py"
    contract = {
        "parent_mei1_manifest_sha256": audit["parent_mei1_manifest_sha256"],
        "config_sha256": sha256_file(config_path),
        "audit_code_sha256": sha256_file(audit_code_path),
        "runner_sha256": sha256_file(Path(__file__)),
        "mrs1_forward_sha256": sha256_file(forward_path),
        "observation_operator_sha256": sha256_file(observation_path),
        "tests_sha256": sha256_file(tests_path),
    }
    contract_sha = sha256_bytes(dumps_stable(contract).encode("utf-8"))
    stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _TV3_ROOT
        / "outputs"
        / "runs"
        / "tv3_mrs_ei"
        / "mei3_varpro_audit"
        / "freezes"
        / f"{stamp}_{contract_sha[:12]}"
    )
    if output_dir.exists():
        print(f"refuse overwrite of existing freeze directory: {output_dir}", file=sys.stderr)
        return 4
    staging = output_dir.with_name(f".{output_dir.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging exists: {staging}")
    staging.mkdir(parents=True)

    freeze_relative = _relative_to_root(output_dir)
    (staging / "mei3_run_config.json").write_bytes(dumps_stable(config).encode("utf-8"))
    (staging / "mei3_structure_audit.json").write_bytes(
        dumps_stable(audit).encode("utf-8")
    )
    (staging / "mei3_observation_operator_audit.json").write_bytes(
        dumps_stable(audit["observation_operator_audit"]).encode("utf-8")
    )
    (staging / "mei3_raw3_forward_rank_audit.json").write_bytes(
        dumps_stable(audit["raw3_forward_rank_audit"]).encode("utf-8")
    )
    verdict = {
        "created_at_utc": created_at.isoformat(),
        "output_dir": freeze_relative,
        "audit": audit,
    }
    (staging / "mei3_verdict.json").write_bytes(dumps_stable(verdict).encode("utf-8"))
    summary = [
        "# tv3 MEI-3 VarPro B0",
        "",
        f"- verdict: `{audit['verdict']}`",
        f"- passed: `{audit['passed']}`",
        f"- admitted_linear_blocks: `{audit['admitted_linear_blocks']}`",
        f"- observation_operator_passed: `{audit['observation_operator_audit']['passed']}`",
        f"- constrained_raw3_rank_passed: `{audit['raw3_forward_rank_audit']['passed']}`",
        f"- formal_solver_gate_ready: `{audit['formal_solver_gate_ready']}`",
        f"- formal_solver_gate_blocker: `{audit['formal_solver_gate_blocker']}`",
        f"- allowed_next_stage: `{audit['allowed_next_stage']}`",
        "",
        "This phase contains only in-memory structure, representation, and rank audits.",
    ]
    (staging / "mei3_summary.md").write_text("\n".join(summary), encoding="utf-8")
    shutil.copy2(parent_dir / "evidence_manifest.json", staging / "parent_mei1_manifest.json")

    snapshots = staging / "source_snapshots"
    snapshots.mkdir()
    source_paths = {
        "audit_code": audit_code_path,
        "runner": Path(__file__).resolve(),
        "mrs1_forward": forward_path,
        "observation_operator": observation_path,
        "tests": tests_path,
    }
    artifacts = [
        "mei3_run_config.json",
        "mei3_structure_audit.json",
        "mei3_observation_operator_audit.json",
        "mei3_raw3_forward_rank_audit.json",
        "mei3_verdict.json",
        "mei3_summary.md",
        "parent_mei1_manifest.json",
    ]
    source_sha256: dict[str, dict[str, str]] = {
        "mei3_run_config": {
            "path": f"{freeze_relative}/mei3_run_config.json",
            "sha256": sha256_file(staging / "mei3_run_config.json"),
        }
    }
    for name, source in source_paths.items():
        relative = f"source_snapshots/{name}{source.suffix}"
        shutil.copy2(source, staging / relative)
        artifacts.append(relative)
        source_sha256[name] = {
            "path": f"{freeze_relative}/{relative}",
            "sha256": sha256_file(staging / relative),
        }

    manifest = {
        "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "freeze_manifest_schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": created_at.isoformat(),
        "freeze_dir": freeze_relative,
        "parent_freeze_id": parent_dir.name,
        "parent_manifest_path": _relative_to_root(parent_dir / "evidence_manifest.json"),
        "parent_manifest_sha256": audit["parent_mei1_manifest_sha256"],
        "input_contract_sha256": contract_sha,
        "git_commit": _git_commit(),
        "git_relevant_paths_dirty": _git_relevant_paths_dirty(),
        "artifact_sha256": {name: sha256_file(staging / name) for name in artifacts},
        "source_sha256": source_sha256,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    (staging / "evidence_manifest.json").write_bytes(
        dumps_stable(manifest).encode("utf-8")
    )
    staging.rename(output_dir)
    manifest_path = output_dir / "evidence_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest_issues = verify_evidence_manifest(
        manifest_path,
        project_root=_TV3_ROOT,
        expected_manifest_sha256=manifest_sha,
    )
    if manifest_issues:
        raise RuntimeError(f"MEI-3 manifest verification failed: {manifest_issues}")

    _promote_stage_status(
        stage_status_path,
        audit=audit,
        freeze_dir=freeze_relative,
        manifest_sha256=manifest_sha,
        created_at_utc=created_at.isoformat(),
        config_sha256=sha256_file(config_path),
    )
    print(f"MEI-3 B0 freeze: {output_dir}")
    print(f"evidence manifest SHA256: {manifest_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
