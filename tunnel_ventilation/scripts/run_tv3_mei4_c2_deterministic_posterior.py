#!/usr/bin/env python3
"""Freeze MEI-4 C2 deterministic posterior evaluation from read-only B4 data."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.mrs_ei_mei4_formal import run_c2_deterministic_posterior  # noqa: E402
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
    result = subprocess.run(["git", "status", "--porcelain", "--", *[_relative_to_root(path) for path in paths]], cwd=_TV3_ROOT, capture_output=True, text=True, check=False)
    return result.returncode != 0 or bool(result.stdout.strip())


def _verify_manifest(path: Path, expected_sha256: str, name: str) -> None:
    issues = verify_evidence_manifest(path, project_root=_TV3_ROOT, expected_manifest_sha256=expected_sha256)
    if issues:
        raise RuntimeError(f"{name} manifest verification failed: {issues}")


def _find_c0_manifest(start: Path, contract_path: Path) -> Path:
    manifest = start
    for _ in range(64):
        if (manifest.parent / "mei4_execution_contract.json").resolve() == contract_path.resolve():
            return manifest
        payload = load_json(manifest)
        parent_path = payload.get("parent_manifest_path")
        parent_sha = payload.get("parent_manifest_sha256")
        if not isinstance(parent_path, str) or not isinstance(parent_sha, str):
            break
        manifest = _resolve_from_root(parent_path)
        _verify_manifest(manifest, parent_sha, "C2 parent chain")
    raise RuntimeError("C2 evidence chain does not lead to the declared C0 contract")


def _find_c1_manifest(start: Path) -> Path:
    manifest = start
    for _ in range(64):
        if (manifest.parent / "mei4_posterior_core_report.json").is_file():
            return manifest
        payload = load_json(manifest)
        parent_path = payload.get("parent_manifest_path")
        parent_sha = payload.get("parent_manifest_sha256")
        if not isinstance(parent_path, str) or not isinstance(parent_sha, str):
            break
        manifest = _resolve_from_root(parent_path)
        _verify_manifest(manifest, parent_sha, "C2 parent chain")
    raise RuntimeError("C2 evidence chain does not lead to a verified C1 freeze")


def _verify_inputs(status: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Path, Path, Path]:
    mei4 = status.get("mei4")
    if not isinstance(mei4, dict) or mei4.get("baseline_solver") != "S1":
        raise RuntimeError("C2 requires a frozen S1 MEI-4 state")
    current_manifest = _resolve_from_root(str(mei4.get("evidence_manifest_path") or ""))
    _verify_manifest(current_manifest, str(mei4.get("evidence_manifest_sha256") or ""), "C2 input")
    if mei4.get("phase") == "c1_posterior_core_audit":
        if mei4.get("status") != "mei4_posterior_core_verified":
            raise RuntimeError("C2 requires C1 posterior-core verification")
        c1_manifest = current_manifest
    elif mei4.get("phase") == "c3_mc_authorization_stop":
        if mei4.get("status") != "mei4_waiting_mc_authorization":
            raise RuntimeError("C2 recomputation requires the C3 authorization stop state")
        c1_manifest = _find_c1_manifest(current_manifest)
    else:
        raise RuntimeError("C2 requires C1 or the C3 authorization stop state")
    contract_path = _resolve_from_root(str(mei4.get("execution_contract_path") or ""))
    contract = load_json(contract_path)
    if contract.get("phase") != "c0_execution_contract_freeze":
        raise RuntimeError("C2 requires the frozen C0 execution contract")
    c0_manifest = _find_c0_manifest(c1_manifest, contract_path)
    b4 = contract["parent_freezes"]["b4"]
    b5 = contract["parent_freezes"]["b5"]
    b4_dir = _resolve_from_root(str(b4["freeze_dir"]))
    _verify_manifest(b4_dir / "evidence_manifest.json", str(b4["evidence_manifest_sha256"]), "B4")
    _verify_manifest(_resolve_from_root(str(b5["freeze_dir"])) / "evidence_manifest.json", str(b5["evidence_manifest_sha256"]), "B5")
    return mei4, contract, c1_manifest, c0_manifest, b4_dir


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _promote_status(path: Path, prior: Mapping[str, Any], freeze_dir: str, manifest_sha256: str, c0_sha: str, c1_sha: str, created_at: str) -> None:
    status = load_json(path)
    current = status.get("mei4")
    if not isinstance(current, dict) or current.get("evidence_manifest_sha256") != prior.get("evidence_manifest_sha256"):
        raise RuntimeError("stage_status changed after C2 input verification")
    status["allowed_next_stage"] = None
    retained = {
        key: value
        for key, value in current.items()
        if key not in {"mc_protocol_path"}
    }
    status["mei4"] = {
        **retained,
        "phase": "c2_deterministic_posterior_evaluation",
        "status": "mei4_c2_deterministic_evaluation_complete",
        "freeze_dir": freeze_dir,
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha256,
        "parent_c0_manifest_sha256": c0_sha,
        "parent_c1_manifest_sha256": c1_sha,
        "c2_report_paths": {
            "coverage": f"{freeze_dir}/coverage_report.json",
            "nll_crps": f"{freeze_dir}/nll_crps_report.json",
            "group_coverage": f"{freeze_dir}/group_coverage_report.json",
            "diagnostics": f"{freeze_dir}/laplace_diagnostics.json",
        },
        "mc_review_eligible": False,
        "allowed_next_stage": None,
        "created_at_utc": created_at,
    }
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stage status staging path exists: {staging}")
    staging.write_bytes(dumps_stable(status).encode("utf-8"))
    staging.replace(path)


def main() -> int:
    args = _parse_args()
    stage_path = args.stage_status_path.resolve() if args.stage_status_path else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    status = load_json(stage_path)
    prior, contract, c1_manifest, c0_manifest, b4_dir = _verify_inputs(status)

    def progress(done: int, total: int, mixture_id: str) -> None:
        if done == 1 or done == total or done % 12 == 0:
            print(f"[C2] {done}/{total} {mixture_id}", flush=True)

    result = run_c2_deterministic_posterior(b4_dir=b4_dir, contract=contract, progress_callback=progress)
    created_at = datetime.now(timezone.utc)
    input_contract = {
        "c0_contract_sha256": sha256_file(_resolve_from_root(str(prior["execution_contract_path"]))),
        "c0_manifest_sha256": sha256_file(c0_manifest),
        "c1_manifest_sha256": sha256_file(c1_manifest),
        "b4_manifest_sha256": sha256_file(b4_dir / "evidence_manifest.json"),
        "b5_manifest_sha256": contract["parent_freezes"]["b5"]["evidence_manifest_sha256"],
    }
    contract_sha = sha256_bytes(dumps_stable(input_contract).encode("utf-8"))
    stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir.resolve() if args.output_dir else _TV3_ROOT / "outputs" / "runs" / "tv3_mrs_ei" / "mei4_posterior_calibration" / "freezes" / f"{stamp}_{contract_sha[:12]}"
    if output_dir.exists():
        print(f"refuse overwrite of existing freeze directory: {output_dir}", file=sys.stderr)
        return 4
    staging = output_dir.with_name(f".{output_dir.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    freeze_dir = _relative_to_root(output_dir)
    payloads = {
        "coverage_report.json": result["coverage_report"],
        "nll_crps_report.json": result["nll_crps_report"],
        "group_coverage_report.json": result["group_coverage_report"],
        "laplace_diagnostics.json": result["laplace_diagnostics"],
        "parent_c0_manifest.json": load_json(c0_manifest),
        "parent_c1_manifest.json": load_json(c1_manifest),
    }
    for name, payload in payloads.items():
        (staging / name).write_bytes(dumps_stable(payload).encode("utf-8"))
    _write_csv(staging / "posterior_intervals_test.csv", result["posterior_intervals_test"])
    _write_csv(staging / "posterior_intervals_ood.csv", result["posterior_intervals_ood"])
    summary = "\n".join((
        "# tv3 MEI-4 C2 deterministic posterior evaluation",
        "",
        "- status: `mei4_c2_deterministic_evaluation_complete`",
        "- methods: `M1`, `M1b` (control only), `M2`",
        "- B4/B5 freeze assets were read-only.",
        "- No observation-space sampling, SBC, PPC, M2b, or CC-SBI was performed.",
        "- This is not a passing verdict; C3 authorization is required for the remaining calibration evidence.",
        "",
    ))
    (staging / "mei4_c2_summary.md").write_text(summary, encoding="utf-8")
    source_paths = {
        "c2_runner": Path(__file__).resolve(),
        "c2_formal": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_mei4_formal.py",
        "posterior": _TV3_ROOT / "tv3" / "ml" / "mrs_posterior.py",
        "posterior_gate": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_posterior_gate.py",
        "solver": _TV3_ROOT / "tv3" / "ml" / "mrs_varpro.py",
        "b4_formal": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_b4_formal.py",
    }
    snapshots = staging / "source_snapshots"
    snapshots.mkdir()
    artifacts = [*payloads, "posterior_intervals_test.csv", "posterior_intervals_ood.csv", "mei4_c2_summary.md"]
    source_sha256 = {}
    for name, source in source_paths.items():
        relative = f"source_snapshots/{name}{source.suffix}"
        shutil.copy2(source, staging / relative)
        artifacts.append(relative)
        source_sha256[name] = {"path": f"{freeze_dir}/{relative}", "sha256": sha256_file(staging / relative)}
    manifest = {
        "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "freeze_manifest_schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": created_at.isoformat(),
        "freeze_dir": freeze_dir,
        "input_contract_sha256": contract_sha,
        "parent_manifest_path": _relative_to_root(c1_manifest),
        "parent_manifest_sha256": sha256_file(c1_manifest),
        "parent_c0_manifest_sha256": sha256_file(c0_manifest),
        "parent_b4_manifest_sha256": sha256_file(b4_dir / "evidence_manifest.json"),
        "parent_b5_manifest_sha256": contract["parent_freezes"]["b5"]["evidence_manifest_sha256"],
        "git_commit": _git_commit(),
        "git_relevant_paths_dirty": _git_relevant_paths_dirty(list(source_paths.values())),
        "artifact_sha256": {name: sha256_file(staging / name) for name in artifacts},
        "source_sha256": source_sha256,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
    }
    (staging / "evidence_manifest.json").write_bytes(dumps_stable(manifest).encode("utf-8"))
    staging.rename(output_dir)
    manifest_path = output_dir / "evidence_manifest.json"
    manifest_sha = sha256_file(manifest_path)
    issues = verify_evidence_manifest(manifest_path, project_root=_TV3_ROOT, expected_manifest_sha256=manifest_sha)
    if issues:
        raise RuntimeError(f"C2 manifest verification failed: {issues}")
    _promote_status(stage_path, prior, freeze_dir, manifest_sha, sha256_file(c0_manifest), sha256_file(c1_manifest), created_at.isoformat())
    print(json.dumps({"freeze_dir": freeze_dir, "manifest_sha256": manifest_sha, "status": result["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
