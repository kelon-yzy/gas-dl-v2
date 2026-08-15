#!/usr/bin/env python3
"""Freeze the MEI-4 C4 CC-SBI trigger audit and authorization stop."""
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

from tv3.audit.mrs_ei_mei4_c4 import build_cc_sbi_trigger_audit  # noqa: E402
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
    parser.add_argument("--stage-status-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(_TV3_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _TV3_ROOT / path


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_TV3_ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _git_relevant_paths_dirty(paths: list[Path]) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *[_relative_to_root(path) for path in paths]],
        cwd=_TV3_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _verify_manifest(path: Path, expected_sha256: str, name: str) -> None:
    issues = verify_evidence_manifest(path, project_root=_TV3_ROOT, expected_manifest_sha256=expected_sha256)
    if issues:
        raise RuntimeError(f"{name} manifest verification failed: {issues}")


def _find_c2_manifest(start: Path) -> Path:
    manifest = start
    for _ in range(64):
        if (manifest.parent / "coverage_report.json").is_file() and (manifest.parent / "laplace_diagnostics.json").is_file():
            return manifest
        payload = load_json(manifest)
        parent_path = payload.get("parent_manifest_path")
        parent_sha = payload.get("parent_manifest_sha256")
        if not isinstance(parent_path, str) or not isinstance(parent_sha, str):
            break
        manifest = _resolve_from_root(parent_path)
        _verify_manifest(manifest, parent_sha, "C4 parent chain")
    raise RuntimeError("C4 evidence chain does not lead to the C2 deterministic evaluation")


def _verify_inputs(status: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Path, Path]:
    mei4 = status.get("mei4")
    if not isinstance(mei4, dict) or mei4.get("phase") != "c3_mc_calibration":
        raise RuntimeError("C4 requires the completed C3 Monte Carlo calibration state")
    if mei4.get("status") != "mei4_c3_mc_calibration_complete":
        raise RuntimeError("C4 requires C3 Monte Carlo calibration completion")
    c3_manifest = _resolve_from_root(str(mei4.get("evidence_manifest_path") or ""))
    _verify_manifest(c3_manifest, str(mei4.get("evidence_manifest_sha256") or ""), "C3")
    report_paths = mei4.get("c3_report_paths")
    if not isinstance(report_paths, dict):
        raise RuntimeError("C4 requires C3 report paths")
    for name in ("sbc", "ppc", "m2b"):
        path = _resolve_from_root(str(report_paths.get(name) or ""))
        if not path.is_file():
            raise RuntimeError(f"C4 requires the C3 {name} report")
    contract_path = _resolve_from_root(str(mei4.get("execution_contract_path") or ""))
    contract = load_json(contract_path)
    if contract.get("phase") != "c0_execution_contract_freeze":
        raise RuntimeError("C4 requires the frozen C0 execution contract")
    if not mei4.get("m2b_triggered"):
        raise RuntimeError("C4 requires the PSIS-triggered M2b C3 branch")
    return mei4, contract, c3_manifest, _find_c2_manifest(c3_manifest)


def _write_freeze(
    *,
    parent_manifest: Path,
    input_contract: Mapping[str, Any],
    payloads: Mapping[str, Any],
    summary: str,
    source_paths: Mapping[str, Path],
    output_dir: Path,
) -> tuple[str, str]:
    if output_dir.exists():
        raise FileExistsError(f"refuse overwrite of existing freeze directory: {output_dir}")
    staging = output_dir.with_name(f".{output_dir.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    freeze_dir = _relative_to_root(output_dir)
    for name, payload in payloads.items():
        (staging / name).write_bytes(dumps_stable(payload).encode("utf-8"))
    (staging / "mei4_c4_summary.md").write_text(summary, encoding="utf-8")
    artifacts = [*payloads, "mei4_c4_summary.md"]
    snapshots = staging / "source_snapshots"
    snapshots.mkdir()
    source_sha256: dict[str, dict[str, str]] = {}
    for name, source in source_paths.items():
        relative = f"source_snapshots/{name}{source.suffix}"
        shutil.copy2(source, staging / relative)
        artifacts.append(relative)
        source_sha256[name] = {"path": f"{freeze_dir}/{relative}", "sha256": sha256_file(staging / relative)}
    manifest = {
        "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "freeze_manifest_schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_dir": freeze_dir,
        "input_contract_sha256": sha256_bytes(dumps_stable(input_contract).encode("utf-8")),
        "parent_manifest_path": _relative_to_root(parent_manifest),
        "parent_manifest_sha256": sha256_file(parent_manifest),
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
    _verify_manifest(manifest_path, manifest_sha, "C4")
    return freeze_dir, manifest_sha


def _promote_status(
    *,
    path: Path,
    prior: Mapping[str, Any],
    freeze_dir: str,
    manifest_sha256: str,
    audit: Mapping[str, Any],
) -> None:
    status = load_json(path)
    current = status.get("mei4")
    if not isinstance(current, dict) or current.get("evidence_manifest_sha256") != prior.get("evidence_manifest_sha256"):
        raise RuntimeError("stage_status changed after C4 input verification")
    triggered = bool(audit["cc_sbi_triggered"])
    authorizations = dict(current.get("authorizations") or {})
    authorizations["mei4_cc_sbi_training_draws"] = "forbidden_until_explicit_mei4_authorization"
    next_status = "mei4_waiting_cc_sbi_training_authorization" if triggered else "mei4_cc_sbi_skipped_not_triggered"
    status["allowed_next_stage"] = None
    status["mei4"] = {
        **current,
        "phase": "c4_cc_sbi_authorization_stop" if triggered else "c4_cc_sbi_trigger_audit",
        "status": next_status,
        "freeze_dir": freeze_dir,
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha256,
        "c4_trigger_audit_path": f"{freeze_dir}/cc_sbi_trigger_audit.json",
        "cc_sbi_contract_path": f"{freeze_dir}/mei4_cc_sbi_contract.json",
        "cc_sbi_triggered": triggered,
        "cc_sbi_triggered_by": list(audit["triggered_by"]),
        "authorizations": authorizations,
        "c5_review_eligible": not triggered,
        "allowed_next_stage": None,
    }
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stage status staging path exists: {staging}")
    staging.write_bytes(dumps_stable(status).encode("utf-8"))
    staging.replace(path)


def main() -> int:
    args = _parse_args()
    stage_path = args.stage_status_path.resolve() if args.stage_status_path else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    prior, execution_contract, c3_manifest, c2_manifest = _verify_inputs(load_json(stage_path))
    c3_reports = prior["c3_report_paths"]
    coverage = load_json(c2_manifest.parent / "coverage_report.json")
    diagnostics = load_json(c2_manifest.parent / "laplace_diagnostics.json")
    audit = build_cc_sbi_trigger_audit(
        execution_contract=execution_contract,
        coverage_report=coverage,
        laplace_diagnostics=diagnostics,
        sbc_report=load_json(_resolve_from_root(str(c3_reports["sbc"]))),
        ppc_report=load_json(_resolve_from_root(str(c3_reports["ppc"]))),
        m2b_report=load_json(_resolve_from_root(str(c3_reports["m2b"]))),
    )
    c4_contract_source = _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei4_cc_sbi_contract.json"
    c4_contract = load_json(c4_contract_source)
    if c4_contract["entry_conditions"]["requires_independent_authorization"] != audit["authorization_requirement"]:
        raise RuntimeError("C4 contract authorization requirement differs from the frozen C0 policy")
    created_at = datetime.now(timezone.utc)
    input_contract = {
        "c3_manifest_sha256": sha256_file(c3_manifest),
        "c2_manifest_sha256": sha256_file(c2_manifest),
        "c0_execution_contract_sha256": sha256_file(_resolve_from_root(str(prior["execution_contract_path"]))),
        "c4_contract_source_sha256": sha256_file(c4_contract_source),
        "trigger_audit_sha256": sha256_bytes(dumps_stable(audit).encode("utf-8")),
    }
    suffix = sha256_bytes(dumps_stable(input_contract).encode("utf-8"))[:12]
    output_dir = args.output_dir.resolve() if args.output_dir else _TV3_ROOT / "outputs" / "runs" / "tv3_mrs_ei" / "mei4_posterior_calibration" / "freezes" / f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{suffix}"
    status = "mei4_waiting_cc_sbi_training_authorization" if audit["cc_sbi_triggered"] else "mei4_cc_sbi_skipped_not_triggered"
    summary = "\n".join((
        "# tv3 MEI-4 C4 CC-SBI trigger audit",
        "",
        f"- status: `{status}`",
        f"- triggered by: `{', '.join(audit['triggered_by']) or 'none'}`",
        "- no CC-SBI training draws were generated and no CC-SBI training occurred.",
        "- the independent `mei4_cc_sbi_training_draws` authorization remains required.",
        "",
    ))
    freeze_dir, manifest_sha = _write_freeze(
        parent_manifest=c3_manifest,
        input_contract=input_contract,
        payloads={
            "cc_sbi_trigger_audit.json": audit,
            "mei4_cc_sbi_contract.json": c4_contract,
            "parent_c2_manifest.json": load_json(c2_manifest),
            "parent_c3_manifest.json": load_json(c3_manifest),
        },
        summary=summary,
        source_paths={
            "c4_runner": Path(__file__).resolve(),
            "c4_auditor": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_mei4_c4.py",
            "c4_contract": c4_contract_source,
        },
        output_dir=output_dir,
    )
    _promote_status(path=stage_path, prior=prior, freeze_dir=freeze_dir, manifest_sha256=manifest_sha, audit=audit)
    print(json.dumps({"status": status, "freeze_dir": freeze_dir, "manifest_sha256": manifest_sha, "triggered_by": audit["triggered_by"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
