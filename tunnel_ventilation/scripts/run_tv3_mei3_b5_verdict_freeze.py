#!/usr/bin/env python3
"""Freeze the MEI-3 B5 baseline verdict from an already verified B4 freeze."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=None)
    parser.add_argument("--stage-status-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_TV3_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_from_root(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _TV3_ROOT / candidate


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


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{field} must be a SHA256 hex digest")
    return value


def _load_parent_b4(
    stage_status: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[Path, Path, Path, str]:
    previous = stage_status.get("mei3") or {}
    parent_policy = contract["parent_b4"]
    if previous.get("phase") != parent_policy["required_phase"]:
        raise RuntimeError("B5 requires stage_status to point to the completed B4 freeze")
    if bool(previous.get("smoke_mode")) and parent_policy["require_non_smoke_freeze"]:
        raise RuntimeError("B5 cannot freeze a smoke-mode B4 result")
    parent_freeze = _resolve_from_root(str(previous.get("freeze_dir") or ""))
    manifest_path = parent_freeze / "evidence_manifest.json"
    verdict_path = parent_freeze / "mei3_verdict.json"
    expected_manifest_sha = _require_sha256(
        previous.get("evidence_manifest_sha256"), field="B4 evidence_manifest_sha256"
    )
    if parent_policy["require_manifest_verification"]:
        issues = verify_evidence_manifest(
            manifest_path,
            project_root=_TV3_ROOT,
            expected_manifest_sha256=expected_manifest_sha,
        )
        if issues:
            raise RuntimeError(f"B4 manifest verification failed: {issues}")
    parent_verdict = load_json(verdict_path)
    if Path(str(parent_verdict.get("output_dir") or "")).as_posix() != _relative_to_root(parent_freeze):
        raise RuntimeError("B4 verdict output_dir does not match stage_status freeze_dir")
    return parent_freeze, manifest_path, verdict_path, expected_manifest_sha


def _resolve_verdict(
    parent_verdict: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    if bool(parent_verdict.get("smoke_mode")):
        raise RuntimeError("B5 cannot promote a smoke-mode B4 verdict")
    gate_report = parent_verdict.get("gate_report") or {}
    verdict = gate_report.get("verdict")
    policy = (contract.get("verdict_policy") or {}).get(verdict)
    if not isinstance(policy, dict):
        raise RuntimeError(f"B4 verdict is not eligible for B5 closure: {verdict!r}")
    passed_solver_gate = bool(gate_report.get("passed_solver_gate"))
    if passed_solver_gate != bool(policy["required_b4_passed_solver_gate"]):
        raise RuntimeError("B4 passed_solver_gate is inconsistent with B5 verdict policy")
    return {
        "verdict": verdict,
        "b4_passed_solver_gate": passed_solver_gate,
        "mei4_baseline": policy["mei4_baseline"],
    }


def _promote_stage_status(
    path: Path,
    *,
    contract: dict[str, Any],
    decision: dict[str, Any],
    parent_freeze_dir: str,
    parent_manifest_sha256: str,
    parent_verdict_sha256: str,
    freeze_dir: str,
    manifest_sha256: str,
    created_at_utc: str,
) -> None:
    status = load_json(path)
    previous = status.get("mei3") or {}
    if previous.get("phase") != contract["parent_b4"]["required_phase"]:
        raise RuntimeError("stage_status changed after B4 evidence was verified")
    if previous.get("evidence_manifest_sha256") != parent_manifest_sha256:
        raise RuntimeError("stage_status B4 manifest changed after verification")
    status["allowed_next_stage"] = contract["closure"]["allowed_next_stage"]
    status["mei3"] = {
        **previous,
        "phase": contract["phase"],
        "verdict": decision["verdict"],
        "freeze_dir": freeze_dir,
        "verdict_path": f"{freeze_dir}/mei3_b5_verdict.json",
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha256,
        "completed_at_utc": created_at_utc,
        "b5_contract_frozen": True,
        "b5_data_generated": contract["closure"]["b5_data_generated"],
        "mei4_baseline": decision["mei4_baseline"],
        "parent_b4_freeze_dir": parent_freeze_dir,
        "parent_b4_manifest_sha256": parent_manifest_sha256,
        "parent_b4_verdict_sha256": parent_verdict_sha256,
    }
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stage status staging path exists: {staging}")
    staging.write_bytes(dumps_stable(status).encode("utf-8"))
    staging.replace(path)


def main() -> int:
    args = _parse_args()
    contract_path = (
        args.contract_config.resolve()
        if args.contract_config is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei3_b5_verdict_contract.json"
    )
    stage_path = (
        args.stage_status_path.resolve()
        if args.stage_status_path is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    )
    contract = load_json(contract_path)
    stage_status = load_json(stage_path)
    parent_freeze, parent_manifest_path, parent_verdict_path, parent_manifest_sha = (
        _load_parent_b4(stage_status, contract)
    )
    parent_verdict = load_json(parent_verdict_path)
    decision = _resolve_verdict(parent_verdict, contract)
    parent_freeze_relative = _relative_to_root(parent_freeze)
    parent_verdict_sha = sha256_file(parent_verdict_path)
    created_at = datetime.now(timezone.utc)
    input_contract = {
        "b5_contract_sha256": sha256_file(contract_path),
        "parent_b4_manifest_sha256": parent_manifest_sha,
        "parent_b4_verdict_sha256": parent_verdict_sha,
        "verdict": decision["verdict"],
        "mei4_baseline": decision["mei4_baseline"],
    }
    input_contract_sha = sha256_bytes(dumps_stable(input_contract).encode("utf-8"))
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
        / f"{stamp}_{input_contract_sha[:12]}"
    )
    if output_dir.exists():
        print(f"refuse overwrite of existing freeze directory: {output_dir}", file=sys.stderr)
        return 4
    staging = output_dir.with_name(f".{output_dir.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging exists: {staging}")
    staging.mkdir(parents=True)

    freeze_relative = _relative_to_root(output_dir)
    verdict = {
        "created_at_utc": created_at.isoformat(),
        "phase": contract["phase"],
        "claim_scope": contract["claim_scope"],
        "verdict": decision["verdict"],
        "mei4_baseline": decision["mei4_baseline"],
        "b4_passed_solver_gate": decision["b4_passed_solver_gate"],
        "b5_data_generated": contract["closure"]["b5_data_generated"],
        "parent_b4_freeze_dir": parent_freeze_relative,
        "parent_b4_manifest_sha256": parent_manifest_sha,
        "parent_b4_verdict_sha256": parent_verdict_sha,
        "interpretation_constraints": contract["interpretation_constraints"],
        "closure": contract["closure"],
    }
    payloads = {
        "mei3_b5_verdict_contract.json": contract,
        "mei3_b5_verdict.json": verdict,
        "parent_b4_stage_status.json": stage_status,
    }
    for name, payload in payloads.items():
        (staging / name).write_bytes(dumps_stable(payload).encode("utf-8"))
    shutil.copy2(parent_manifest_path, staging / "parent_b4_manifest.json")
    shutil.copy2(parent_verdict_path, staging / "parent_b4_verdict.json")
    summary = [
        "# tv3 MEI-3 B5 verdict freeze",
        "",
        f"- verdict: `{decision['verdict']}`",
        f"- MEI-4 baseline: `{decision['mei4_baseline']}`",
        f"- parent B4 freeze: `{parent_freeze_relative}`",
        f"- parent B4 manifest SHA256: `{parent_manifest_sha}`",
        "- B5 data generated: `False`",
        "- B4 was neither rerun nor rewritten.",
    ]
    (staging / "mei3_b5_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    source_paths = {
        "b5_contract": contract_path,
        "b5_runner": Path(__file__).resolve(),
    }
    snapshots = staging / "source_snapshots"
    snapshots.mkdir()
    artifacts = list(payloads) + [
        "parent_b4_manifest.json",
        "parent_b4_verdict.json",
        "mei3_b5_summary.md",
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
        "input_contract_sha256": input_contract_sha,
        "parent_manifest_path": _relative_to_root(parent_manifest_path),
        "parent_manifest_sha256": parent_manifest_sha,
        "git_commit": _git_commit(),
        "git_relevant_paths_dirty": _git_relevant_paths_dirty(list(source_paths.values())),
        "artifact_sha256": {name: sha256_file(staging / name) for name in artifacts},
        "source_sha256": source_sha256,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (staging / "evidence_manifest.json").write_bytes(dumps_stable(manifest).encode("utf-8"))
    staging.rename(output_dir)
    manifest_path = output_dir / "evidence_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    issues = verify_evidence_manifest(
        manifest_path,
        project_root=_TV3_ROOT,
        expected_manifest_sha256=manifest_sha,
    )
    if issues:
        raise RuntimeError(f"MEI-3 B5 manifest verification failed: {issues}")
    _promote_stage_status(
        stage_path,
        contract=contract,
        decision=decision,
        parent_freeze_dir=parent_freeze_relative,
        parent_manifest_sha256=parent_manifest_sha,
        parent_verdict_sha256=parent_verdict_sha,
        freeze_dir=freeze_relative,
        manifest_sha256=manifest_sha,
        created_at_utc=created_at.isoformat(),
    )
    print(
        json.dumps(
            {
                "freeze_dir": freeze_relative,
                "manifest_sha256": manifest_sha,
                "verdict": decision["verdict"],
                "mei4_baseline": decision["mei4_baseline"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
