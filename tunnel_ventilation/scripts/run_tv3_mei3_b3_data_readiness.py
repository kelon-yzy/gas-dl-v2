#!/usr/bin/env python3
"""Freeze the MEI-3 B3 registered-data authorization-readiness package."""
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
from tv3.audit.mrs_ei_solver_gate import run_mei3_b3_readiness_audit  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-b2-freeze-dir", type=Path, required=True)
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


def _git_relevant_paths_dirty(paths: list[Path]) -> bool:
    relative_paths = [_relative_to_root(path) for path in paths]
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative_paths],
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
    previous = status.get("mei3") or {}
    status["allowed_next_stage"] = audit["allowed_next_stage"]
    status["mei3"] = {
        "phase": audit["phase"],
        "verdict": audit["verdict"],
        "passed": audit["passed"],
        "freeze_dir": freeze_dir,
        "verdict_path": f"{freeze_dir}/mei3_b3_verdict.json",
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha256,
        "completed_at_utc": created_at_utc,
        "s0_status": previous.get("s0_status"),
        "s1_frozen": previous.get("s1_frozen"),
        "s2_core_verified": previous.get("s2_core_verified"),
        "s3_upper_bound_verified": previous.get("s3_upper_bound_verified"),
        "b3_authorization_ready_package": audit["passed"],
        "registered_sparse_simulation_generation_review_eligible": audit[
            "registered_sparse_simulation_generation_review_eligible"
        ],
        "registered_sparse_simulation_generation_authorized": False,
        "formal_solver_gate_ready": audit["formal_solver_gate_ready"],
        "formal_solver_gate_blocker": audit["formal_solver_gate_blocker"],
        "authorizations": audit["authorizations"],
        "parent_b2_manifest_sha256": audit["parent_b2_manifest_sha256"],
        "parent_b1_manifest_sha256": previous.get("parent_b1_manifest_sha256"),
        "parent_b0_manifest_sha256": previous.get("parent_b0_manifest_sha256"),
        "parent_mei1_manifest_sha256": previous.get("parent_mei1_manifest_sha256"),
        "config_sha256": config_sha256,
    }
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stage status staging path exists: {staging}")
    staging.write_bytes(dumps_stable(status).encode("utf-8"))
    staging.replace(path)


def main() -> int:
    args = _parse_args()
    config_path = (
        args.config.resolve()
        if args.config is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei3_solver_data_protocol.json"
    )
    stage_status_path = (
        args.stage_status_path.resolve()
        if args.stage_status_path is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    )
    parent_dir = args.parent_b2_freeze_dir.resolve()
    if not parent_dir.is_dir():
        print(f"parent B2 freeze missing: {parent_dir}", file=sys.stderr)
        return 3
    config = load_json(config_path)
    audit = run_mei3_b3_readiness_audit(
        project_root=_TV3_ROOT,
        config=config,
        parent_b2_freeze_dir=parent_dir,
        current_stage_status=load_json(stage_status_path),
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if audit["issues"]:
        return 3
    if not audit["passed"]:
        return 2

    source_paths = {
        "schema": _TV3_ROOT / "tv3" / "sim" / "generation" / "tunnel_ventilation" / "mrs_observation.py",
        "protocol_config": config_path,
        "split_implementation": _TV3_ROOT / "tv3" / "sim" / "packaging" / "splits.py",
        "generator": _TV3_ROOT / "tv3" / "sim" / "generation" / "tunnel_ventilation" / "relaxation_spectrum.py",
        "solver": _TV3_ROOT / "tv3" / "ml" / "mrs_varpro.py",
        "metric_implementation": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_registry.py",
        "b3_audit": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_solver_gate.py",
        "b3_runner": Path(__file__).resolve(),
        "b3_tests": _TV3_ROOT / "tests" / "test_tunnel_ventilation_mei3_b3_readiness.py",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"B3 source inventory missing: {missing}")
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    contract = {
        "parent_b2_manifest_sha256": audit["parent_b2_manifest_sha256"],
        **{f"{name}_sha256": value for name, value in source_hashes.items()},
    }
    contract_sha = sha256_bytes(dumps_stable(contract).encode("utf-8"))
    created_at = datetime.now(timezone.utc)
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
    payloads = {
        "mei3_solver_data_protocol.json": config,
        "sample_size_power_report.json": audit["power_recomputation"],
        "runtime_field_access_report.json": {
            "whitelists": config["runtime_field_whitelists"],
            "s3_truth_isolation": config["tables"]["s3_truth_nuisance"]["physical_isolation"],
            "passed": audit["runtime_field_isolation_frozen"],
        },
        "source_hash_inventory.json": source_hashes,
        "authorization_readiness_report.json": {
            "review_eligible": audit["registered_sparse_simulation_generation_review_eligible"],
            "authorizations": audit["authorizations"],
            "formal_data_generated": audit["formal_data_generated"],
            "formal_solver_gate_ready": audit["formal_solver_gate_ready"],
            "blocker": audit["formal_solver_gate_blocker"],
        },
    }
    for name, payload in payloads.items():
        (staging / name).write_bytes(dumps_stable(payload).encode("utf-8"))
    verdict = {
        "created_at_utc": created_at.isoformat(),
        "output_dir": freeze_relative,
        "audit": audit,
    }
    (staging / "mei3_b3_verdict.json").write_bytes(dumps_stable(verdict).encode("utf-8"))
    summary = [
        "# tv3 MEI-3 B3 data authorization readiness",
        "",
        f"- verdict: `{audit['verdict']}`",
        f"- passed: `{audit['passed']}`",
        f"- review_eligible: `{audit['registered_sparse_simulation_generation_review_eligible']}`",
        f"- formal_solver_gate_ready: `{audit['formal_solver_gate_ready']}`",
        f"- blocker: `{audit['formal_solver_gate_blocker']}`",
        "",
        "This freeze contains contracts and readiness evidence only; no formal data were generated.",
    ]
    (staging / "mei3_b3_summary.md").write_text("\n".join(summary), encoding="utf-8")
    shutil.copy2(parent_dir / "evidence_manifest.json", staging / "parent_b2_manifest.json")

    snapshots = staging / "source_snapshots"
    snapshots.mkdir()
    artifacts = list(payloads) + [
        "mei3_b3_verdict.json",
        "mei3_b3_summary.md",
        "parent_b2_manifest.json",
    ]
    source_sha256: dict[str, dict[str, str]] = {}
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
        "parent_manifest_sha256": audit["parent_b2_manifest_sha256"],
        "input_contract_sha256": contract_sha,
        "git_commit": _git_commit(),
        "git_relevant_paths_dirty": _git_relevant_paths_dirty(list(source_paths.values())),
        "artifact_sha256": {name: sha256_file(staging / name) for name in artifacts},
        "source_sha256": source_sha256,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    (staging / "evidence_manifest.json").write_bytes(dumps_stable(manifest).encode("utf-8"))
    staging.rename(output_dir)
    manifest_path = output_dir / "evidence_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest_issues = verify_evidence_manifest(
        manifest_path,
        project_root=_TV3_ROOT,
        expected_manifest_sha256=manifest_sha,
    )
    if manifest_issues:
        raise RuntimeError(f"MEI-3 B3 manifest verification failed: {manifest_issues}")
    _promote_stage_status(
        stage_status_path,
        audit=audit,
        freeze_dir=freeze_relative,
        manifest_sha256=manifest_sha,
        created_at_utc=created_at.isoformat(),
        config_sha256=sha256_file(config_path),
    )
    print(f"MEI-3 B3 freeze: {output_dir}")
    print(f"evidence manifest SHA256: {manifest_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
