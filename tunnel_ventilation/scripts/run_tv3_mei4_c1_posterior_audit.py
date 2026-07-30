#!/usr/bin/env python3
"""Freeze MEI-4 C1 synthetic posterior-mechanism audit evidence."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.mrs_ei_posterior_gate import run_posterior_core_audit  # noqa: E402
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
    parser.add_argument("--audit-config", type=Path, default=None)
    parser.add_argument("--stage-status-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_TV3_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _TV3_ROOT / path


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


def _verify_manifest(path: Path, *, expected_sha256: str, name: str) -> None:
    issues = verify_evidence_manifest(
        path,
        project_root=_TV3_ROOT,
        expected_manifest_sha256=expected_sha256,
    )
    if issues:
        raise RuntimeError(f"{name} manifest verification failed: {issues}")


def _find_c0_manifest(*, start_manifest: Path, contract_path: Path) -> Path:
    manifest_path = start_manifest
    for _ in range(64):
        candidate_contract = manifest_path.parent / "mei4_execution_contract.json"
        if candidate_contract.resolve() == contract_path.resolve():
            return manifest_path
        manifest = load_json(manifest_path)
        parent_path = manifest.get("parent_manifest_path")
        parent_sha256 = manifest.get("parent_manifest_sha256")
        if not isinstance(parent_path, str) or not isinstance(parent_sha256, str):
            break
        manifest_path = _resolve_from_root(parent_path)
        _verify_manifest(
            manifest_path,
            expected_sha256=parent_sha256,
            name="C1 parent",
        )
    raise RuntimeError("C1 evidence chain does not lead to the declared C0 contract")


def _verify_c0(
    stage_status: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path, Path]:
    mei4 = stage_status.get("mei4")
    if not isinstance(mei4, dict):
        raise RuntimeError("C1 requires the C0 execution-contract freeze")
    phase = mei4.get("phase")
    if phase == "c0_execution_contract_freeze":
        if mei4.get("status") != "mei4_contract_frozen":
            raise RuntimeError("C1 requires a completed C0 execution-contract freeze")
    elif phase == "c1_posterior_core_audit":
        if mei4.get("status") != "mei4_posterior_core_verified":
            raise RuntimeError("C1 append requires a verified prior C1 freeze")
    else:
        raise RuntimeError("C1 requires C0 or a verified prior C1 freeze")
    if mei4.get("baseline_solver") != "S1":
        raise RuntimeError("C1 requires C0 to retain the S1 baseline")
    freeze_dir = _resolve_from_root(str(mei4.get("freeze_dir") or ""))
    parent_manifest_path = _resolve_from_root(
        str(mei4.get("evidence_manifest_path") or freeze_dir / "evidence_manifest.json")
    )
    _verify_manifest(
        parent_manifest_path,
        expected_sha256=str(mei4.get("evidence_manifest_sha256") or ""),
        name="C0" if phase == "c0_execution_contract_freeze" else "C1",
    )
    contract_path = _resolve_from_root(str(mei4.get("execution_contract_path") or ""))
    contract = load_json(contract_path)
    if contract.get("phase") != "c0_execution_contract_freeze":
        raise RuntimeError("C0 execution contract has an invalid phase")
    c0_manifest_path = _find_c0_manifest(
        start_manifest=parent_manifest_path,
        contract_path=contract_path,
    )
    return mei4, parent_manifest_path, c0_manifest_path, contract_path


def _promote_status(
    path: Path,
    *,
    prior: Mapping[str, Any],
    parent_manifest_sha256: str,
    c0_manifest_sha256: str,
    freeze_dir: str,
    manifest_sha256: str,
    result: Mapping[str, Any],
    created_at_utc: str,
) -> None:
    status = load_json(path)
    current = status.get("mei4")
    if not isinstance(current, dict) or current.get("phase") != prior.get("phase"):
        raise RuntimeError("stage_status changed after parent verification")
    if current.get("evidence_manifest_sha256") != prior.get("evidence_manifest_sha256"):
        raise RuntimeError("parent manifest changed after verification")
    status["allowed_next_stage"] = None
    status["mei4"] = {
        **current,
        "phase": "c1_posterior_core_audit",
        "status": result["status"],
        "freeze_dir": freeze_dir,
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha256,
        "c1_report_path": f"{freeze_dir}/mei4_posterior_core_report.json",
        "parent_c0_manifest_sha256": c0_manifest_sha256,
        "parent_c1_manifest_sha256": (
            parent_manifest_sha256
            if prior.get("phase") == "c1_posterior_core_audit"
            else None
        ),
        "created_at_utc": created_at_utc,
    }
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stage status staging path exists: {staging}")
    staging.write_bytes(dumps_stable(status).encode("utf-8"))
    staging.replace(path)


def main() -> int:
    args = _parse_args()
    audit_path = (
        args.audit_config.resolve()
        if args.audit_config is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei4_posterior_audit.json"
    )
    stage_path = (
        args.stage_status_path.resolve()
        if args.stage_status_path is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    )
    audit_config = load_json(audit_path)
    if audit_config.get("phase") != "c1_posterior_core_audit":
        raise RuntimeError("unsupported MEI-4 C1 audit configuration")
    stage_status = load_json(stage_path)
    prior, parent_manifest_path, c0_manifest_path, c0_contract_path = _verify_c0(
        stage_status
    )
    result = run_posterior_core_audit(audit_config)
    created_at = datetime.now(timezone.utc)
    input_contract = {
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "c0_manifest_sha256": sha256_file(c0_manifest_path),
        "c0_contract_sha256": sha256_file(c0_contract_path),
        "c1_audit_config_sha256": sha256_file(audit_path),
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
        / "mei4_posterior_calibration"
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
    negative_controls = {
        "all_explicitly_failed": all(
            value == "explicit_failure" for value in result["negative_controls"].values()
        ),
        "controls": result["negative_controls"],
    }
    payloads = {
        "mei4_posterior_audit.json": audit_config,
        "mei4_posterior_core_report.json": result,
        "mei4_negative_controls_report.json": negative_controls,
        "parent_c0_manifest.json": load_json(c0_manifest_path),
    }
    if prior.get("phase") == "c1_posterior_core_audit":
        payloads["parent_c1_manifest.json"] = load_json(parent_manifest_path)
    for name, payload in payloads.items():
        (staging / name).write_bytes(dumps_stable(payload).encode("utf-8"))
    summary = [
        "# tv3 MEI-4 C1 posterior-core audit freeze",
        "",
        f"- status: `{result['status']}`",
        f"- parent manifest SHA256: `{sha256_file(parent_manifest_path)}`",
        f"- parent C0 manifest SHA256: `{sha256_file(c0_manifest_path)}`",
        "- audit data: in-memory synthetic fixtures only",
        "- B4 observations were not read.",
    ]
    (staging / "mei4_c1_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    source_paths = {
        "c1_audit_config": audit_path,
        "c1_runner": Path(__file__).resolve(),
        "posterior": _TV3_ROOT / "tv3" / "ml" / "mrs_posterior.py",
        "posterior_gate": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_posterior_gate.py",
    }
    snapshots = staging / "source_snapshots"
    snapshots.mkdir()
    artifacts = [*payloads, "mei4_c1_summary.md"]
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
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "git_commit": _git_commit(),
        "git_relevant_paths_dirty": _git_relevant_paths_dirty(list(source_paths.values())),
        "artifact_sha256": {name: sha256_file(staging / name) for name in artifacts},
        "source_sha256": source_sha256,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
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
        raise RuntimeError(f"MEI-4 C1 manifest verification failed: {issues}")
    _promote_status(
        stage_path,
        prior=prior,
        parent_manifest_sha256=sha256_file(parent_manifest_path),
        c0_manifest_sha256=sha256_file(c0_manifest_path),
        freeze_dir=freeze_relative,
        manifest_sha256=manifest_sha,
        result=result,
        created_at_utc=created_at.isoformat(),
    )
    print(json.dumps({"freeze_dir": freeze_relative, "manifest_sha256": manifest_sha, "status": result["status"]}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
