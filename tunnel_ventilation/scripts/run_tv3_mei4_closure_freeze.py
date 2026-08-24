#!/usr/bin/env python3
"""Freeze the tv3 MRS-EI line closure on frozen MEI-4 C2 evidence (path P-C).

This runner performs no new physics and no observation-space sampling. It
verifies the C2 freeze, recomputes every claim the methodology paper cites from
frozen artifacts, and records the closure decision together with the C0 contract
contradictions and the evidence-chain gaps.

Refuses to write a freeze when any registered claim disagrees with the evidence.
"""
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

from tv3.audit.mei4_closure_checks import (  # noqa: E402
    STATUS_MATCH,
    CheckContext,
    run_checklist,
    summarize,
)
from tv3.audit.mrs_ei_registry import (  # noqa: E402
    FREEZE_MANIFEST_SCHEMA_VERSION,
    dumps_stable,
    load_json,
    sha256_bytes,
    sha256_file,
    verify_evidence_manifest,
)

_STAGE = "mei4_posterior_calibration"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=None)
    parser.add_argument("--stage-status-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run the evidence checklist and print the report without freezing.",
    )
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
    relative = [_relative_to_root(path) for path in paths]
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative],
        cwd=_TV3_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _verify_parent_state(stage_status: dict[str, Any], contract: dict[str, Any]) -> Path:
    """Confirm stage_status still describes the state this closure supersedes."""
    previous = stage_status.get("mei4") or {}
    expected_state = contract["decision"]["supersedes_state"]
    if previous.get("status") != expected_state:
        raise RuntimeError(
            f"MEI-4 stage status is {previous.get('status')!r}, expected {expected_state!r}"
        )
    if previous.get("allowed_next_stage") is not None:
        raise RuntimeError("MEI-4 allowed_next_stage must be null before closure")
    parent = contract["parent_c2"]
    freeze_dir = _TV3_ROOT / parent["freeze_dir"]
    recorded = (previous.get("c2_report_paths") or {}).get("coverage")
    if not recorded or not recorded.startswith(parent["freeze_dir"]):
        raise RuntimeError("stage_status C2 report paths do not match the closure contract")
    manifest_path = freeze_dir / "evidence_manifest.json"
    if parent.get("require_manifest_verification", True):
        issues = verify_evidence_manifest(
            manifest_path,
            project_root=_TV3_ROOT,
            expected_manifest_sha256=parent["manifest_sha256"],
        )
        if issues:
            raise RuntimeError(f"C2 manifest verification failed: {issues}")
    return freeze_dir


def _render_summary(
    contract: dict[str, Any], report: dict[str, Any], freeze_relative: str
) -> str:
    closure = contract["closure"]
    waiver = contract["invariant_waivers"][0]
    check_summary = report["summary"]
    lines = [
        "# tv3 MRS-EI 收尾冻结（MEI-4，路径 P-C）",
        "",
        f"- verdict: `{closure['verdict']}`（语义 `{closure['verdict_semantics']}`）",
        f"- line status: `{closure['line_status']}`",
        f"- allowed_next_stage: `{closure['allowed_next_stage']}`",
        f"- freeze: `{freeze_relative}`",
        f"- parent C2: `{contract['parent_c2']['freeze_dir']}`",
        "",
        "## 证据核对",
        "",
        f"- 登记claim数: {check_summary['n_checks']}",
        f"- 一致: {check_summary['n_match']}",
        f"- 不一致: {check_summary['n_mismatch']}",
        f"- 无法验证: {check_summary['n_unverifiable']}",
        "",
        "## 六件套完整性豁免",
        "",
        f"- 已有: {'、'.join(waiver['present_items'])}",
        f"- 缺失: {'、'.join(waiver['missing_items'])}",
        f"- 后果: {waiver['consequence']}",
        "",
        "## C0 契约矛盾处置",
        "",
    ]
    for item in contract["c0_contradiction_disposition"]:
        lines.append(f"- `{item['id']}` → `{item['disposition']}`：{item['consequence']}")
    lines += ["", "## 证据链缺口", ""]
    for gap in contract["evidence_gaps"]:
        lines.append(f"- {gap['item']}：`{gap['status']}`")
    lines += ["", "## 收尾后禁止", ""]
    lines += [f"- {rule}" for rule in closure["forbidden_after_closure"]]
    lines += [
        "",
        "本次收尾未运行任何新计算，未改写任何既有 freeze，未解除任何授权。",
    ]
    return "\n".join(lines) + "\n"


def _promote_stage_status(
    path: Path,
    *,
    contract: dict[str, Any],
    freeze_dir: str,
    manifest_sha256: str,
    created_at_utc: str,
    check_summary: dict[str, Any],
) -> None:
    status = load_json(path)
    previous = status.get("mei4") or {}
    if previous.get("status") != contract["decision"]["supersedes_state"]:
        raise RuntimeError("stage_status changed after C2 evidence was verified")
    closure = contract["closure"]
    status["allowed_next_stage"] = closure["allowed_next_stage"]
    status["mei4"] = {
        **previous,
        "phase": contract["phase"],
        "status": closure["verdict"],
        "verdict_semantics": closure["verdict_semantics"],
        "line_status": closure["line_status"],
        "closure_freeze_dir": freeze_dir,
        "closure_evidence_manifest_sha256": manifest_sha256,
        "closure_completed_at_utc": created_at_utc,
        "closure_path": contract["decision"]["path"],
        "closure_evidence_checks": check_summary,
        "mc_review_eligible": False,
        "allowed_next_stage": closure["allowed_next_stage"],
        "authorizations": closure["authorizations_unchanged"],
        "reopen_requires": closure["reopen_requires"],
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
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei4_closure_contract.json"
    )
    stage_path = (
        args.stage_status_path.resolve()
        if args.stage_status_path is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    )
    contract = load_json(contract_path)
    stage_status = load_json(stage_path)
    c2_freeze_dir = _verify_parent_state(stage_status, contract)

    context = CheckContext(project_root=_TV3_ROOT, c2_freeze_dir=c2_freeze_dir)
    results = run_checklist(
        context,
        contract["evidence_checklist"],
        default_tolerance=float(contract["default_tolerance"]),
    )
    check_summary = summarize(results)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": sha256_file(contract_path),
        "parent_c2_freeze_dir": _relative_to_root(c2_freeze_dir),
        "summary": check_summary,
        "results": [result.as_dict() for result in results],
    }

    if args.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if check_summary["n_mismatch"] == 0 else 5

    if check_summary["n_mismatch"]:
        print(
            "refuse to freeze: registered claims disagree with frozen evidence: "
            f"{check_summary['mismatched_ids']}",
            file=sys.stderr,
        )
        return 5
    if check_summary["n_unverifiable"]:
        print(
            "refuse to freeze: unverifiable claims: "
            f"{check_summary['unverifiable_ids']}",
            file=sys.stderr,
        )
        return 6

    created_at = datetime.now(timezone.utc)
    input_contract = {
        "closure_contract_sha256": report["contract_sha256"],
        "parent_c2_manifest_sha256": contract["parent_c2"]["manifest_sha256"],
        "verdict": contract["closure"]["verdict"],
        "path": contract["decision"]["path"],
        "n_checks": check_summary["n_checks"],
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
        / _STAGE
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
        "verdict": contract["closure"]["verdict"],
        "verdict_semantics": contract["closure"]["verdict_semantics"],
        "decision": contract["decision"],
        "parent_c2": contract["parent_c2"],
        "invariant_waivers": contract["invariant_waivers"],
        "c0_contradiction_disposition": contract["c0_contradiction_disposition"],
        "evidence_gaps": contract["evidence_gaps"],
        "evidence_check_summary": check_summary,
        "closure": contract["closure"],
        "new_computation_performed": False,
        "observation_space_sampling_performed": False,
    }
    payloads = {
        "mei4_closure_contract.json": contract,
        "mei4_closure_verdict.json": verdict,
        "evidence_checklist_report.json": report,
        "parent_stage_status.json": stage_status,
    }
    for name, payload in payloads.items():
        (staging / name).write_bytes(dumps_stable(payload).encode("utf-8"))
    shutil.copy2(
        c2_freeze_dir / "evidence_manifest.json", staging / "parent_c2_manifest.json"
    )
    (staging / "mei4_closure_summary.md").write_text(
        _render_summary(contract, report, freeze_relative), encoding="utf-8"
    )

    source_paths = {
        "closure_contract": contract_path,
        "closure_runner": Path(__file__).resolve(),
        "closure_checks": _TV3_ROOT / "tv3" / "audit" / "mei4_closure_checks.py",
    }
    (staging / "source_snapshots").mkdir()
    artifacts = list(payloads) + ["parent_c2_manifest.json", "mei4_closure_summary.md"]
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
        "parent_manifest_path": _relative_to_root(c2_freeze_dir / "evidence_manifest.json"),
        "parent_manifest_sha256": contract["parent_c2"]["manifest_sha256"],
        "git_commit": _git_commit(),
        "git_relevant_paths_dirty": _git_relevant_paths_dirty(list(source_paths.values())),
        "artifact_sha256": {name: sha256_file(staging / name) for name in artifacts},
        "source_sha256": source_sha256,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (staging / "evidence_manifest.json").write_bytes(
        dumps_stable(manifest).encode("utf-8")
    )
    staging.rename(output_dir)

    manifest_path = output_dir / "evidence_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    issues = verify_evidence_manifest(
        manifest_path,
        project_root=_TV3_ROOT,
        expected_manifest_sha256=manifest_sha,
    )
    if issues:
        raise RuntimeError(f"closure manifest verification failed: {issues}")
    _promote_stage_status(
        stage_path,
        contract=contract,
        freeze_dir=freeze_relative,
        manifest_sha256=manifest_sha,
        created_at_utc=created_at.isoformat(),
        check_summary=check_summary,
    )
    print(
        json.dumps(
            {
                "freeze_dir": freeze_relative,
                "manifest_sha256": manifest_sha,
                "verdict": contract["closure"]["verdict"],
                "evidence_checks": check_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
