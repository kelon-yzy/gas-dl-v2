#!/usr/bin/env python3
"""Freeze the MEI-4 C3 observation-space Monte Carlo readiness package."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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
from tv3.audit.mrs_ei_mei4_parallel import load_runtime_config, run_c3_resumable  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-status-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--runtime-config-path", type=Path, default=None)
    parser.add_argument("--attempt-dir", type=Path, default=None, help="Create a fresh C3 compute attempt here.")
    parser.add_argument("--resume-attempt", type=Path, default=None, help="Resume an existing validated C3 attempt.")
    parser.add_argument("--workers", type=int, default=None, help="Override only the runtime worker count.")
    parser.add_argument(
        "--authorize-registered-sparse-simulation-generation",
        action="store_true",
        help="Record the explicit MEI-4-scoped user authorization before any MC work.",
    )
    parser.add_argument(
        "--run-authorized-mc",
        action="store_true",
        help="Run the registered SBC, PPC, and triggered M2b calculations after authorization.",
    )
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


def _verify_inputs(status: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    mei4 = status.get("mei4")
    if not isinstance(mei4, dict) or mei4.get("phase") != "c2_deterministic_posterior_evaluation":
        raise RuntimeError("C3 requires a completed C2 deterministic posterior freeze")
    if mei4.get("status") != "mei4_c2_deterministic_evaluation_complete":
        raise RuntimeError("C3 requires C2 deterministic evaluation completion")
    manifest_path = _resolve_from_root(str(mei4.get("evidence_manifest_path") or ""))
    issues = verify_evidence_manifest(manifest_path, project_root=_TV3_ROOT, expected_manifest_sha256=str(mei4.get("evidence_manifest_sha256") or ""))
    if issues:
        raise RuntimeError(f"C2 manifest verification failed: {issues}")
    coverage = load_json(manifest_path.parent / "coverage_report.json")
    if coverage.get("status") != "mei4_c2_intermediate_no_pass_verdict":
        raise RuntimeError("C3 refuses a C2 report that claims a calibration verdict")
    contract = load_json(_resolve_from_root(str(mei4.get("execution_contract_path") or "")))
    if contract.get("authorizations", {}).get("registered_sparse_simulation_generation") == "authorized":
        raise RuntimeError("C3 contract must not inherit the MEI-3 generation authorization")
    return mei4, contract, manifest_path


def _promote_status(path: Path, prior: Mapping[str, Any], freeze_dir: str, manifest_sha: str, created_at: str) -> None:
    status = load_json(path)
    current = status.get("mei4")
    if not isinstance(current, dict) or current.get("evidence_manifest_sha256") != prior.get("evidence_manifest_sha256"):
        raise RuntimeError("stage_status changed after C3 input verification")
    status["allowed_next_stage"] = None
    status["mei4"] = {
        **current,
        "phase": "c3_mc_authorization_stop",
        "status": "mei4_waiting_mc_authorization",
        "freeze_dir": freeze_dir,
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha,
        "mc_protocol_path": f"{freeze_dir}/mei4_mc_protocol.json",
        "mc_review_eligible": True,
        "allowed_next_stage": None,
        "created_at_utc": created_at,
    }
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stage status staging path exists: {staging}")
    staging.write_bytes(dumps_stable(status).encode("utf-8"))
    staging.replace(path)


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
    (staging / "mei4_c3_summary.md").write_text(summary, encoding="utf-8")
    artifacts = [*payloads, "mei4_c3_summary.md"]
    snapshots = staging / "source_snapshots"
    snapshots.mkdir()
    source_sha256: dict[str, dict[str, str]] = {}
    for name, source in source_paths.items():
        relative = f"source_snapshots/{name}{source.suffix}"
        shutil.copy2(source, staging / relative)
        artifacts.append(relative)
        source_sha256[name] = {
            "path": f"{freeze_dir}/{relative}",
            "sha256": sha256_file(staging / relative),
        }
    created_at = datetime.now(timezone.utc)
    manifest = {
        "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "freeze_manifest_schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": created_at.isoformat(),
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
    issues = verify_evidence_manifest(manifest_path, project_root=_TV3_ROOT, expected_manifest_sha256=manifest_sha)
    if issues:
        raise RuntimeError(f"C3 manifest verification failed: {issues}")
    return freeze_dir, manifest_sha


def _verify_waiting_authorization(status: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    mei4 = status.get("mei4")
    if not isinstance(mei4, dict) or mei4.get("phase") != "c3_mc_authorization_stop":
        raise RuntimeError("MEI-4 C3 authorization requires the frozen C3 stop state")
    if mei4.get("status") != "mei4_waiting_mc_authorization":
        raise RuntimeError("MEI-4 C3 is not waiting for a new authorization")
    parent_manifest = _resolve_from_root(str(mei4.get("evidence_manifest_path") or ""))
    issues = verify_evidence_manifest(parent_manifest, project_root=_TV3_ROOT, expected_manifest_sha256=str(mei4.get("evidence_manifest_sha256") or ""))
    if issues:
        raise RuntimeError(f"C3 readiness manifest verification failed: {issues}")
    contract = load_json(_resolve_from_root(str(mei4.get("execution_contract_path") or "")))
    if contract.get("phase") != "c0_execution_contract_freeze":
        raise RuntimeError("MEI-4 C3 requires the frozen C0 execution contract")
    if mei4.get("authorizations", {}).get("registered_sparse_simulation_generation") != "forbidden_until_explicit_mei4_authorization":
        raise RuntimeError("C3 authorization state is not the expected MEI-4 waiting value")
    return mei4, contract, parent_manifest


def _promote_authorized_status(
    *, path: Path, prior: Mapping[str, Any], freeze_dir: str, manifest_sha: str, authorization: Mapping[str, Any]
) -> None:
    status = load_json(path)
    current = status.get("mei4")
    if not isinstance(current, dict) or current.get("evidence_manifest_sha256") != prior.get("evidence_manifest_sha256"):
        raise RuntimeError("stage_status changed after C3 authorization verification")
    authorizations = dict(current.get("authorizations") or {})
    authorizations["registered_sparse_simulation_generation"] = "authorized"
    status["allowed_next_stage"] = None
    status["mei4"] = {
        **current,
        "phase": "c3_mc_calibration",
        "status": "mei4_mc_authorized_pending_execution",
        "freeze_dir": freeze_dir,
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha,
        "authorization_freeze_dir": freeze_dir,
        "authorization_record": authorization,
        "authorizations": authorizations,
        "mc_review_eligible": False,
        "allowed_next_stage": None,
    }
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stage status staging path exists: {staging}")
    staging.write_bytes(dumps_stable(status).encode("utf-8"))
    staging.replace(path)


def _authorize_mc(stage_path: Path, output_dir: Path | None) -> dict[str, str]:
    prior, contract, parent_manifest = _verify_waiting_authorization(load_json(stage_path))
    created_at = datetime.now(timezone.utc)
    authorization = {
        "field": "registered_sparse_simulation_generation",
        "value": "authorized",
        "authorized_at_utc": created_at.isoformat(),
        "authority": "explicit_user_decision",
        "scope": "mei4_registered_sparse_simulation_generation_only",
        "permitted_activities": ["SBC", "PPC", "M2b_only_when_registered_PSIS_triggered"],
        "unchanged_authorizations": ["formal_waveform_generation", "benchmark_packaging", "hardware_trial", "mei4_cc_sbi_training_draws"],
        "note": "user explicitly authorized MEI-4 registered_sparse_simulation_generation and requested C3 execution",
    }
    input_contract = {
        "readiness_manifest_sha256": sha256_file(parent_manifest),
        "execution_contract_sha256": sha256_file(_resolve_from_root(str(prior["execution_contract_path"]))),
        "authorization": authorization,
    }
    suffix = sha256_bytes(dumps_stable(input_contract).encode("utf-8"))[:12]
    target = output_dir or (_TV3_ROOT / "outputs" / "runs" / "tv3_mrs_ei" / "mei4_posterior_calibration" / "freezes" / f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{suffix}")
    freeze_dir, manifest_sha = _write_freeze(
        parent_manifest=parent_manifest,
        input_contract=input_contract,
        payloads={"mei4_mc_authorization.json": authorization, "mei4_mc_protocol.json": contract["mc_protocol"]},
        summary="\n".join((
            "# tv3 MEI-4 C3 authorization freeze",
            "",
            "- status: `mei4_mc_authorized_pending_execution`",
            "- authorization: `registered_sparse_simulation_generation` scoped to MEI-4",
            "- permitted: SBC, PPC, and conditionally triggered M2b only",
            "- CC-SBI training remains unauthorized.",
            "",
        )),
        source_paths={"c3_runner": Path(__file__).resolve()},
        output_dir=target,
    )
    _promote_authorized_status(path=stage_path, prior=prior, freeze_dir=freeze_dir, manifest_sha=manifest_sha, authorization=authorization)
    return {"freeze_dir": freeze_dir, "manifest_sha256": manifest_sha}


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
        issues = verify_evidence_manifest(manifest, project_root=_TV3_ROOT, expected_manifest_sha256=parent_sha)
        if issues:
            raise RuntimeError(f"C3 parent chain verification failed: {issues}")
    raise RuntimeError("C3 evidence chain does not lead to the C2 deterministic evaluation")


def _m2b_triggered(c2_manifest: Path, contract: Mapping[str, Any]) -> tuple[bool, dict[str, float]]:
    diagnostics = load_json(c2_manifest.parent / "laplace_diagnostics.json")
    policy = contract["method_policy"]["M2"]
    rates: dict[str, float] = {}
    for domain in ("test", "ood"):
        row = diagnostics["methods"]["M2"][domain]
        exceeded = int(row["rejection_reasons"].get("psis_k_hat_exceeded_for_M2", 0))
        rates[domain] = exceeded / int(row["n"])
    return any(rate > float(policy["maximum_over_threshold_rate"]) for rate in rates.values()), rates


def _promote_mc_complete(
    *, path: Path, prior: Mapping[str, Any], freeze_dir: str, manifest_sha: str, m2b_rates: Mapping[str, float]
) -> None:
    status = load_json(path)
    current = status.get("mei4")
    if not isinstance(current, dict) or current.get("evidence_manifest_sha256") != prior.get("evidence_manifest_sha256"):
        raise RuntimeError("stage_status changed after C3 MC input verification")
    status["allowed_next_stage"] = None
    status["mei4"] = {
        **current,
        "phase": "c3_mc_calibration",
        "status": "mei4_c3_mc_calibration_complete",
        "freeze_dir": freeze_dir,
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha,
        "c3_report_paths": {
            "sbc": f"{freeze_dir}/sbc_rank_histograms.json",
            "ppc": f"{freeze_dir}/ppc_report.json",
            "m2b": f"{freeze_dir}/bootstrap_posterior_report.json",
        },
        "m2b_triggered": True,
        "m2_psis_over_threshold_rates": dict(m2b_rates),
        "c4_review_eligible": True,
        "allowed_next_stage": None,
    }
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stage status staging path exists: {staging}")
    staging.write_bytes(dumps_stable(status).encode("utf-8"))
    staging.replace(path)


def _run_authorized_mc(
    stage_path: Path,
    output_dir: Path | None,
    *,
    runtime_config_path: Path,
    attempt_dir: Path | None,
    resume_attempt: Path | None,
    workers: int | None,
) -> dict[str, str]:
    prior = load_json(stage_path).get("mei4")
    if not isinstance(prior, dict) or prior.get("phase") != "c3_mc_calibration" or prior.get("status") != "mei4_mc_authorized_pending_execution":
        raise RuntimeError("C3 MC execution requires the MEI-4 authorized-pending-execution state")
    if prior.get("authorizations", {}).get("registered_sparse_simulation_generation") != "authorized":
        raise RuntimeError("C3 MC execution requires the MEI-4-scoped generation authorization")
    parent_manifest = _resolve_from_root(str(prior.get("evidence_manifest_path") or ""))
    issues = verify_evidence_manifest(parent_manifest, project_root=_TV3_ROOT, expected_manifest_sha256=str(prior.get("evidence_manifest_sha256") or ""))
    if issues:
        raise RuntimeError(f"C3 authorization manifest verification failed: {issues}")
    contract = load_json(_resolve_from_root(str(prior.get("execution_contract_path") or "")))
    c2_manifest = _find_c2_manifest(parent_manifest)
    triggered, rates = _m2b_triggered(c2_manifest, contract)
    if not triggered:
        raise RuntimeError("C3 refuses M2b because the registered PSIS trigger is absent")
    b4_dir = _resolve_from_root(str(contract["parent_freezes"]["b4"]["freeze_dir"]))
    b4_manifest = b4_dir / "evidence_manifest.json"
    issues = verify_evidence_manifest(b4_manifest, project_root=_TV3_ROOT, expected_manifest_sha256=str(contract["parent_freezes"]["b4"]["evidence_manifest_sha256"]))
    if issues:
        raise RuntimeError(f"B4 manifest verification failed: {issues}")

    source_paths = {
        "c3_runner": Path(__file__).resolve(),
        "mc_engine": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_mei4_mc.py",
        "mc_parallel": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_mei4_parallel.py",
        "posterior_formal": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_mei4_formal.py",
        "posterior_gate": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_posterior_gate.py",
        "registry": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_registry.py",
        "posterior_core": _TV3_ROOT / "tv3" / "ml" / "mrs_posterior.py",
        "solver": _TV3_ROOT / "tv3" / "ml" / "mrs_varpro.py",
        "relaxation_spectrum": _TV3_ROOT
        / "tv3"
        / "sim"
        / "generation"
        / "tunnel_ventilation"
        / "relaxation_spectrum.py",
        "runtime_config": runtime_config_path,
    }
    runtime = load_runtime_config(runtime_config_path, workers_override=workers)
    binding = {
        "authorization_manifest_sha256": sha256_file(parent_manifest),
        "c2_manifest_sha256": sha256_file(c2_manifest),
        "b4_manifest_sha256": sha256_file(b4_manifest),
        "execution_contract_sha256": sha256_file(_resolve_from_root(str(prior["execution_contract_path"]))),
        "m2_psis_over_threshold_rates": rates,
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
    }
    if attempt_dir is not None and resume_attempt is not None:
        raise RuntimeError("--attempt-dir and --resume-attempt are mutually exclusive")
    resume = resume_attempt is not None
    if resume_attempt is not None:
        target_attempt = resume_attempt.resolve()
    elif attempt_dir is not None:
        target_attempt = attempt_dir.resolve()
    else:
        attempt_suffix = sha256_bytes(dumps_stable({"binding": binding, "runtime": asdict(runtime)}).encode("utf-8"))[:12]
        attempt_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target_attempt = (
            _TV3_ROOT
            / "outputs"
            / "runs"
            / "tv3_mrs_ei"
            / "mei4_posterior_calibration"
            / "attempts"
            / f"{attempt_stamp}_{attempt_suffix}"
        )

    progress_state: dict[str, dict[str, float]] = {}

    def progress(label: str, done: int, total: int, item: str) -> None:
        now = time.monotonic()
        state = progress_state.setdefault(
            label,
            {"started": now, "baseline": float(done), "last_printed": float(done)},
        )
        should_print = done == total or done == int(state["baseline"]) or done - state["last_printed"] >= 25
        if not should_print:
            return
        elapsed = now - state["started"]
        completed_now = done - int(state["baseline"])
        rate = completed_now / elapsed if completed_now > 0 and elapsed > 0.0 else 0.0
        eta = (total - done) / rate if rate > 0.0 else float("nan")
        print(
            f"[{label}] {done}/{total} rate={rate:.3f}/s eta_s={eta:.1f} {item}",
            flush=True,
        )
        state["last_printed"] = float(done)

    execution = run_c3_resumable(
        b4_dir=b4_dir,
        contract=contract,
        binding=binding,
        runtime=runtime,
        attempt_dir=target_attempt,
        resume=resume,
        m2b_triggered=True,
        progress_callback=progress,
    )
    result = execution["reports"]
    attempt = execution["attempt"]
    attempt_path = Path(str(attempt["attempt_dir"]))
    created_at = datetime.now(timezone.utc)
    input_contract = {
        **binding,
        "attempt_id": attempt["attempt_id"],
        "attempt_manifest_sha256": attempt["attempt_manifest_sha256"],
        "attempt_result_sha256": attempt["result_sha256"],
    }
    suffix = sha256_bytes(dumps_stable(input_contract).encode("utf-8"))[:12]
    target = output_dir or (_TV3_ROOT / "outputs" / "runs" / "tv3_mrs_ei" / "mei4_posterior_calibration" / "freezes" / f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{suffix}")
    freeze_dir, manifest_sha = _write_freeze(
        parent_manifest=parent_manifest,
        input_contract=input_contract,
        payloads={
            "sbc_rank_histograms.json": result["sbc_rank_histograms"],
            "ppc_report.json": result["ppc_report"],
            "bootstrap_posterior_report.json": result["bootstrap_posterior_report"],
            "c3_attempt_manifest.json": load_json(attempt_path / "attempt_manifest.json"),
            "parent_c2_manifest.json": load_json(c2_manifest),
        },
        summary="\n".join((
            "# tv3 MEI-4 C3 authorized Monte Carlo calibration",
            "",
            "- status: `mei4_c3_mc_calibration_complete`",
            "- executed: SBC, PPC, and PSIS-triggered M2b only",
            f"- resumable attempt: `{attempt['attempt_id']}`",
            "- no formal waveform generation, benchmark packaging, hardware work, or CC-SBI training occurred.",
            "",
        )),
        source_paths=source_paths,
        output_dir=target,
    )
    _promote_mc_complete(path=stage_path, prior=prior, freeze_dir=freeze_dir, manifest_sha=manifest_sha, m2b_rates=rates)
    return {
        "freeze_dir": freeze_dir,
        "manifest_sha256": manifest_sha,
        "attempt_id": str(attempt["attempt_id"]),
    }


def main() -> int:
    args = _parse_args()
    stage_path = args.stage_status_path.resolve() if args.stage_status_path else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    runtime_config_path = (
        args.runtime_config_path.resolve()
        if args.runtime_config_path
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei4_c3_runtime.json"
    )
    if (args.attempt_dir is not None or args.resume_attempt is not None or args.workers is not None) and not args.run_authorized_mc:
        raise RuntimeError("attempt and worker options require --run-authorized-mc")
    if args.output_dir is not None and args.authorize_registered_sparse_simulation_generation and args.run_authorized_mc:
        raise RuntimeError("--output-dir may target only one C3 freeze per invocation")
    if args.authorize_registered_sparse_simulation_generation:
        authorization = _authorize_mc(stage_path, args.output_dir.resolve() if args.output_dir else None)
        print(json.dumps({"status": "mei4_mc_authorized_pending_execution", **authorization}, ensure_ascii=False, indent=2))
        if not args.run_authorized_mc:
            return 0
    if args.run_authorized_mc:
        result = _run_authorized_mc(
            stage_path,
            args.output_dir.resolve() if args.output_dir else None,
            runtime_config_path=runtime_config_path,
            attempt_dir=args.attempt_dir,
            resume_attempt=args.resume_attempt,
            workers=args.workers,
        )
        print(json.dumps({"status": "mei4_c3_mc_calibration_complete", **result}, ensure_ascii=False, indent=2))
        return 0
    prior, contract, c2_manifest = _verify_inputs(load_json(stage_path))
    created_at = datetime.now(timezone.utc)
    protocol = {
        "schema_version": "tunnel-ventilation-mrs-ei-mei4-mc-protocol-1",
        "phase": "c3_mc_authorization_stop",
        "source_contract": {"execution_contract_path": prior["execution_contract_path"], "mc_protocol": contract["mc_protocol"]},
        "authorization_stop": {
            "required_record": "MEI-4 scoped registered_sparse_simulation_generation authorization",
            "before_authorization_status": "mei4_waiting_mc_authorization",
            "permitted_after_authorization": ["SBC", "PPC", "M2b_only_when_registered_PSIS_triggered"],
            "forbidden": contract["mc_protocol"]["forbidden_uses"],
        },
    }
    input_contract = {"c2_manifest_sha256": sha256_file(c2_manifest), "c0_contract_sha256": sha256_file(_resolve_from_root(str(prior["execution_contract_path"]))), "mc_protocol_sha256": sha256_bytes(dumps_stable(protocol).encode("utf-8"))}
    input_sha = sha256_bytes(dumps_stable(input_contract).encode("utf-8"))
    stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir.resolve() if args.output_dir else _TV3_ROOT / "outputs" / "runs" / "tv3_mrs_ei" / "mei4_posterior_calibration" / "freezes" / f"{stamp}_{input_sha[:12]}"
    if output_dir.exists():
        print(f"refuse overwrite of existing freeze directory: {output_dir}", file=sys.stderr)
        return 4
    staging = output_dir.with_name(f".{output_dir.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging exists: {staging}")
    staging.mkdir(parents=True)
    freeze_dir = _relative_to_root(output_dir)
    payloads = {"mei4_mc_protocol.json": protocol, "parent_c2_manifest.json": load_json(c2_manifest)}
    for name, payload in payloads.items():
        (staging / name).write_bytes(dumps_stable(payload).encode("utf-8"))
    summary = "\n".join((
        "# tv3 MEI-4 C3 Monte Carlo readiness freeze",
        "",
        "- status: `mei4_waiting_mc_authorization`",
        "- C3 does not generate observations or run SBC/PPC.",
        "- A new MEI-4 scoped authorization is required before the registered observation-space work.",
        "",
    ))
    (staging / "mei4_c3_summary.md").write_text(summary, encoding="utf-8")
    source_paths = {"c3_runner": Path(__file__).resolve()}
    snapshots = staging / "source_snapshots"
    snapshots.mkdir()
    artifacts = [*payloads, "mei4_c3_summary.md"]
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
        "input_contract_sha256": input_sha,
        "parent_manifest_path": _relative_to_root(c2_manifest),
        "parent_manifest_sha256": sha256_file(c2_manifest),
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
    issues = verify_evidence_manifest(manifest_path, project_root=_TV3_ROOT, expected_manifest_sha256=manifest_sha)
    if issues:
        raise RuntimeError(f"C3 manifest verification failed: {issues}")
    _promote_status(stage_path, prior, freeze_dir, manifest_sha, created_at.isoformat())
    print(json.dumps({"freeze_dir": freeze_dir, "manifest_sha256": manifest_sha, "status": "mei4_waiting_mc_authorization"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
