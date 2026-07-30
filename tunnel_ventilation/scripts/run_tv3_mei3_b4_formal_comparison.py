#!/usr/bin/env python3
"""B4 formal S1--S3 paired comparison (authorization gated)."""
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

import numpy as np

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.mrs_ei_b4_formal import run_b4_formal_comparison  # noqa: E402
from tv3.audit.mrs_ei_registry import (  # noqa: E402
    FREEZE_MANIFEST_SCHEMA_VERSION,
    dumps_stable,
    load_json,
    sha256_bytes,
    sha256_file,
    verify_evidence_manifest,
)
from tv3.audit.mrs_ei_solver_gate import B4_WAITING  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-config", type=Path, default=None)
    parser.add_argument("--protocol-config", type=Path, default=None)
    parser.add_argument("--solver-config", type=Path, default=None)
    parser.add_argument("--design-space", type=Path, default=None)
    parser.add_argument("--stage-status-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--mixture-limit-per-split",
        type=int,
        default=None,
        help="Smoke-only limit on design conditions per split; forbidden for formal freeze.",
    )
    parser.add_argument(
        "--bootstrap-n-resamples",
        type=int,
        default=None,
        help="Override bootstrap resamples (tests/smoke only).",
    )
    parser.add_argument(
        "--allow-incomplete-smoke",
        action="store_true",
        help="Permit mixture/bootstrap reductions; writes smoke_mode=true and non-final verdict.",
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
    relative_paths = [_relative_to_root(path) for path in paths]
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *relative_paths],
        cwd=_TV3_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_solver_comparison(paired_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    flat: list[dict[str, object]] = []
    for row in paired_rows:
        base = {
            "mixture_id": row["mixture_id"],
            "split": row["split"],
            "design_condition_id": row["design_condition_id"],
            "data_split_seed_index": row["data_split_seed_index"],
            "truth_co2": row["truth_raw3_percent"][0],
            "truth_o2": row["truth_raw3_percent"][1],
            "truth_n2": row["truth_raw3_percent"][2],
            "crb_o2_std_percent": row["crb_o2_std_percent"],
        }
        for method in ("S1", "S2", "S3"):
            payload = row[method]
            flat.append(
                {
                    **base,
                    "method": method,
                    "frozen_initialization_index": payload["frozen_initialization_index"],
                    "success": payload["success"],
                    "stop_reason": payload["stop_reason"],
                    "objective": payload["objective"],
                    "iterations": payload["iterations"],
                    "forward_calls": payload["forward_calls"],
                    "bound_hit": payload["bound_hit"],
                    "pred_co2": payload["raw3_percent"][0],
                    "pred_o2": payload["raw3_percent"][1],
                    "pred_n2": payload["raw3_percent"][2],
                    "abs_err_co2": payload["abs_err_co2"],
                    "abs_err_o2": payload["abs_err_o2"],
                    "abs_err_n2": payload["abs_err_n2"],
                    "relative_crb_efficiency_o2": payload["relative_crb_efficiency_o2"],
                }
            )
    return flat


def _resolve_parent_pre_b4_manifest_sha256(previous: dict[str, object]) -> str:
    recorded_parent = previous.get("parent_pre_b4_manifest_sha256")
    if recorded_parent is not None:
        if not isinstance(recorded_parent, str) or len(recorded_parent) != 64:
            raise RuntimeError("parent_pre_b4_manifest_sha256 must be a SHA256 hex digest")
        return recorded_parent
    if previous.get("phase") == "b4_formal_solver_comparison":
        raise RuntimeError(
            "B4 rerun requires the immutable parent_pre_b4_manifest_sha256 record"
        )
    initial_parent = previous.get("evidence_manifest_sha256")
    if not isinstance(initial_parent, str) or len(initial_parent) != 64:
        raise RuntimeError("pre-B4 stage status must provide evidence_manifest_sha256")
    return initial_parent


def _promote_stage_status(
    path: Path,
    *,
    result: dict[str, object],
    freeze_dir: str,
    manifest_sha256: str,
    created_at_utc: str,
) -> None:
    status = load_json(path)
    previous = status.get("mei3") or {}
    parent_pre_b4_manifest_sha256 = _resolve_parent_pre_b4_manifest_sha256(previous)
    gate_report = result["gate_report"]
    smoke = bool(result.get("smoke_mode"))
    # B5 owns baseline promotion; B4 only records the comparison verdict.
    status["allowed_next_stage"] = None
    status["mei3"] = {
        **previous,
        "phase": "b4_formal_solver_comparison",
        "verdict": gate_report["verdict"] if not smoke else "mei3_b4_smoke_completed",
        "passed": bool(gate_report["passed_solver_gate"]) and not smoke,
        "freeze_dir": freeze_dir,
        "verdict_path": f"{freeze_dir}/mei3_verdict.json",
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha256,
        "completed_at_utc": created_at_utc,
        "registered_sparse_simulation_generation_authorized": True,
        "formal_solver_gate_ready": True,
        "formal_solver_gate_blocker": None,
        "formal_data_generated": True,
        "smoke_mode": smoke,
        "b4_passed_solver_gate": bool(gate_report["passed_solver_gate"]),
        "authorizations": previous.get("authorizations"),
        "authorization_record": previous.get("authorization_record"),
        "parent_pre_b4_manifest_sha256": parent_pre_b4_manifest_sha256,
    }
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"stage status staging path exists: {staging}")
    staging.write_bytes(dumps_stable(status).encode("utf-8"))
    staging.replace(path)


def main() -> int:
    args = _parse_args()
    gate_path = (
        args.gate_config.resolve()
        if args.gate_config is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei3_b4_formal_gate.json"
    )
    protocol_path = (
        args.protocol_config.resolve()
        if args.protocol_config is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei3_solver_data_protocol.json"
    )
    solver_path = (
        args.solver_config.resolve()
        if args.solver_config is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei3_solver_audit.json"
    )
    design_path = (
        args.design_space.resolve()
        if args.design_space is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "design_space.json"
    )
    stage_path = (
        args.stage_status_path.resolve()
        if args.stage_status_path is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    )

    smoke_requested = (
        args.mixture_limit_per_split is not None or args.bootstrap_n_resamples is not None
    )
    if smoke_requested and not args.allow_incomplete_smoke:
        print(
            "Refuse reduced B4 run without --allow-incomplete-smoke "
            "(formal sample size and bootstrap are frozen).",
            file=sys.stderr,
        )
        return 4

    gate = load_json(gate_path)
    protocol = load_json(protocol_path)
    solver_config = load_json(solver_path)
    design_space = load_json(design_path)
    stage_status = load_json(stage_path)

    def _progress(method: str, done: int, total: int, mixture_id: str) -> None:
        if done == 1 or done == total or done % 50 == 0:
            print(f"[{method}] {done}/{total} {mixture_id}", flush=True)

    result = run_b4_formal_comparison(
        project_root=_TV3_ROOT,
        protocol=protocol,
        gate=gate,
        solver_config=solver_config,
        design_space=design_space,
        stage_status=stage_status,
        mixture_limit_per_split=args.mixture_limit_per_split,
        bootstrap_n_resamples_override=args.bootstrap_n_resamples,
        progress_callback=_progress,
    )
    print(json.dumps(result.get("decision"), indent=2, ensure_ascii=False))
    if not result["authorized"]:
        print(
            "B4 refused: independent registered_sparse_simulation_generation authorization required.",
            file=sys.stderr,
        )
        assert result["decision"]["verdict"] == B4_WAITING
        return 5

    created_at = datetime.now(timezone.utc)
    contract = {
        "gate_sha256": sha256_file(gate_path),
        "protocol_sha256": sha256_file(protocol_path),
        "solver_config_sha256": sha256_file(solver_path),
        "design_space_sha256": sha256_file(design_path),
        "b4_module_sha256": sha256_file(
            _TV3_ROOT / "tv3" / "audit" / "mrs_ei_b4_formal.py"
        ),
        "solver_module_sha256": sha256_file(_TV3_ROOT / "tv3" / "ml" / "mrs_varpro.py"),
        "observation_sha256": sha256_file(
            _TV3_ROOT
            / "tv3"
            / "sim"
            / "generation"
            / "tunnel_ventilation"
            / "mrs_observation.py"
        ),
        "authorization_record": (stage_status.get("mei3") or {}).get("authorization_record"),
        "smoke_mode": bool(result.get("smoke_mode")),
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
    gate_report = result["gate_report"]
    payloads = {
        "mei3_solver_run_config.json": {
            "gate": gate,
            "protocol": protocol,
            "solver_config": solver_config,
            "dataset_meta": result["dataset_meta"],
            "authorization_record": (stage_status.get("mei3") or {}).get(
                "authorization_record"
            ),
        },
        "mei3_b4_formal_gate.json": gate,
        "mei3_solver_data_protocol.json": protocol,
        "calibration_report.json": result["calibration"],
        "bootstrap_report.json": gate_report["bootstrap_report"],
        "convergence_report.json": {
            domain: {
                method: {
                    "convergence_failure_rate": report[method]["convergence_failure_rate"],
                    "bound_hit_rate": report[method]["bound_hit_rate"],
                    "mean_iterations": report[method]["mean_iterations"],
                    "mean_forward_calls": report[method]["mean_forward_calls"],
                }
                for method in ("S1", "S2", "S3")
            }
            for domain, report in gate_report["domain_reports"].items()
        },
        "registered_observations.json": {
            "mixtures": result["mixtures"],
            "observation_rows": result["observation_rows"],
            "covariance_blocks": result["covariance_blocks"],
        },
        "s3_truth_nuisance.json": result["s3_truth_nuisance"],
        "view_nuisance_calibration_priors.json": result["calibration"][
            "view_nuisance_calibration_priors"
        ],
        "calibration_priors.json": result["calibration"]["calibration_priors"],
        "paired_solutions.json": result["paired_rows"],
        "mei3_verdict.json": {
            "created_at_utc": created_at.isoformat(),
            "output_dir": freeze_relative,
            "smoke_mode": bool(result.get("smoke_mode")),
            "gate_report": gate_report,
            "dataset_meta": result["dataset_meta"],
        },
    }
    for name, payload in payloads.items():
        (staging / name).write_bytes(dumps_stable(payload).encode("utf-8"))

    comparison_rows = _flatten_solver_comparison(result["paired_rows"])
    _write_csv(staging / "solver_comparison.csv", comparison_rows)
    crb_rows = [
        {
            "mixture_id": row["mixture_id"],
            "split": row["split"],
            "crb_o2_std_percent": row["crb_o2_std_percent"],
            "s1_abs_err_o2": row["S1"]["abs_err_o2"],
            "s2_abs_err_o2": row["S2"]["abs_err_o2"],
            "s3_abs_err_o2": row["S3"]["abs_err_o2"],
            "s1_relative_crb_efficiency_o2": row["S1"]["relative_crb_efficiency_o2"],
            "s2_relative_crb_efficiency_o2": row["S2"]["relative_crb_efficiency_o2"],
            "s3_relative_crb_efficiency_o2": row["S3"]["relative_crb_efficiency_o2"],
        }
        for row in result["paired_rows"]
    ]
    _write_csv(staging / "crb_efficiency.csv", crb_rows)

    summary = [
        "# tv3 MEI-3 B4 formal S1--S3 comparison",
        "",
        f"- verdict: `{gate_report['verdict']}`",
        f"- passed_solver_gate: `{gate_report['passed_solver_gate']}`",
        f"- smoke_mode: `{bool(result.get('smoke_mode'))}`",
        f"- formal_data_generated: `{True}`",
        "",
        "S3 is upper-bound audit only. Primary comparison is S1 vs S2.",
    ]
    for domain, report in gate_report["domain_reports"].items():
        summary.append(
            f"- {domain} O2 P90 relative improvement: "
            f"`{report['paired_relative_improvement_o2_p90']}`"
        )
    (staging / "mei3_b4_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    source_paths = {
        "b4_formal": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_b4_formal.py",
        "solver": _TV3_ROOT / "tv3" / "ml" / "mrs_varpro.py",
        "observation_operator": (
            _TV3_ROOT / "tv3" / "sim" / "generation" / "tunnel_ventilation" / "mrs_observation.py"
        ),
        "gate_config": gate_path,
        "protocol_config": protocol_path,
        "solver_config": solver_path,
        "design_space": design_path,
        "b4_runner": Path(__file__).resolve(),
    }
    snapshots = staging / "source_snapshots"
    snapshots.mkdir()
    artifacts = list(payloads) + [
        "solver_comparison.csv",
        "crb_efficiency.csv",
        "mei3_b4_summary.md",
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
        "smoke_mode": bool(result.get("smoke_mode")),
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
        raise RuntimeError(f"MEI-3 B4 manifest verification failed: {issues}")

    if not result.get("smoke_mode"):
        _promote_stage_status(
            stage_path,
            result=result,
            freeze_dir=freeze_relative,
            manifest_sha256=manifest_sha,
            created_at_utc=created_at.isoformat(),
        )
    print(f"MEI-3 B4 freeze: {output_dir}")
    print(f"evidence manifest SHA256: {manifest_sha}")
    print(f"verdict: {gate_report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
