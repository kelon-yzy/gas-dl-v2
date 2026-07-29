#!/usr/bin/env python3
"""Run MEI-1 forward-envelope audit from an immutable parent MEI-0 freeze.

Prerequisite: MEI-0 ``mei0_registry_frozen`` with registry schema v2.
Does not generate waveforms. Reads registries from --parent-mei0-freeze-dir.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.mrs_ei_forward_envelope import run_mei1_audit  # noqa: E402
from tv3.audit.mrs_ei_registry import (  # noqa: E402
    FORBIDDEN_AUTH_VALUE,
    FREEZE_MANIFEST_SCHEMA_VERSION,
    dumps_stable,
    load_json,
    sha256_bytes,
    sha256_file,
    verify_evidence_manifest,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing mei1_forward_envelope.json run config only",
    )
    p.add_argument(
        "--parent-mei0-freeze-dir",
        type=Path,
        required=True,
        help="Immutable MEI-0 freeze directory providing registries",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_TV3_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _contract_sha(
    *,
    parent_manifest_sha256: str,
    mei1_run_config_sha256: str,
    audit_code_sha256: str,
    mrs1_forward_sha256: str,
) -> str:
    payload = {
        "parent_mei0_manifest_sha256": parent_manifest_sha256,
        "mei1_run_config_sha256": mei1_run_config_sha256,
        "mei1_audit_code_sha256": audit_code_sha256,
        "mrs1_forward_sha256": mrs1_forward_sha256,
    }
    return sha256_bytes(dumps_stable(payload).encode("utf-8"))


def _promote_staging(staging: Path, output_dir: Path, *, attempts: int = 8) -> None:
    """Atomically promote staging dir; retry transient Windows locks (WinError 5)."""
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            staging.rename(output_dir)
            return
        except OSError as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            time.sleep(0.25 * (attempt + 1))
    assert last_error is not None
    raise last_error


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
            "tv3/audit/mrs_ei_registry.py",
            "tv3/audit/mrs_ei_forward_envelope.py",
            "scripts/run_tv3_mei0_registry_freeze.py",
            "scripts/run_tv3_mei1_forward_envelope.py",
            "tests/test_tunnel_ventilation_mei0_registry.py",
            "tests/test_tunnel_ventilation_mei1_forward_envelope.py",
        ],
        cwd=_TV3_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def main() -> int:
    args = _parse_args()
    config_dir = (
        args.config_dir.resolve()
        if args.config_dir is not None
        else (_TV3_ROOT / "configs" / "tv3_mrs_ei").resolve()
    )
    parent_dir = args.parent_mei0_freeze_dir.resolve()
    created_at = datetime.now(timezone.utc)

    if not parent_dir.is_dir():
        print(f"parent MEI-0 freeze missing: {parent_dir}", file=sys.stderr)
        return 3

    print("MEI-1: running forward-envelope audit from parent freeze...")
    try:
        audit = run_mei1_audit(
            project_root=_TV3_ROOT,
            config_dir=config_dir,
            parent_mei0_freeze_dir=parent_dir,
        )
    except Exception as exc:  # noqa: BLE001 - map technical failures to exit 4
        print(f"MEI-1 numerical/artifact failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4

    print(
        json.dumps(
            {
                "verdict": audit["verdict"],
                "passed": audit["passed"],
                "n_points": audit.get("n_points"),
                "n_designs": audit.get("n_designs"),
                "blockers": audit.get("blockers"),
                "issues": audit.get("issues"),
                "noise_profiles": audit.get("noise_profiles"),
                "pressure_domain": audit.get("pressure_domain"),
                "parked_nonblocking_families": audit.get(
                    "parked_nonblocking_families"
                ),
                "decision_reason": audit.get("decision_reason"),
                "frozen_design": audit.get("frozen_design"),
                "authorizations": audit.get("authorizations"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if audit.get("issues"):
        # Preflight/contract failure: do not write a scientific freeze.
        return int(audit.get("exit_code_hint") or 3)

    mei1_run_config = load_json(config_dir / "mei1_forward_envelope.json")
    run_config_sha = sha256_file(config_dir / "mei1_forward_envelope.json")
    audit_code_sha = sha256_file(
        _TV3_ROOT / "tv3" / "audit" / "mrs_ei_forward_envelope.py"
    )
    mrs1_sha = sha256_file(
        _TV3_ROOT
        / "tv3"
        / "sim"
        / "generation"
        / "tunnel_ventilation"
        / "relaxation_spectrum.py"
    )
    contract_sha = _contract_sha(
        parent_manifest_sha256=str(audit["parent_mei0_manifest_sha256"]),
        mei1_run_config_sha256=run_config_sha,
        audit_code_sha256=audit_code_sha,
        mrs1_forward_sha256=mrs1_sha,
    )

    stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    default_out = (
        _TV3_ROOT
        / "outputs"
        / "runs"
        / "tv3_mrs_ei"
        / "mei1_forward_envelope"
        / "freezes"
        / f"{stamp}_{contract_sha[:12]}"
    )
    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_out
    if output_dir.exists():
        print(f"refuse overwrite of existing freeze directory: {output_dir}", file=sys.stderr)
        return 4

    staging = output_dir.with_name(f".{output_dir.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging exists: {staging}")
    staging.mkdir(parents=True)

    # Config copy (run config) vs results file must use distinct names.
    (staging / "mei1_run_config.json").write_bytes(
        dumps_stable(mei1_run_config).encode("utf-8")
    )
    for name in (
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
        "stage_status.json",
        "evidence_manifest.json",
    ):
        src = parent_dir / name
        if src.is_file():
            shutil.copy2(src, staging / name)
    shutil.copy2(
        parent_dir / "evidence_manifest.json",
        staging / "parent_mei0_manifest.json",
    )

    slim_profiles: dict[str, Any] = {}
    for profile_id, rep in (audit.get("profile_results") or {}).items():
        slim_domains = {}
        for domain_id, domain in (rep.get("domains") or {}).items():
            slim = dict(domain)
            fams = {}
            for fid, fam in (domain.get("family_reports") or {}).items():
                fam_slim = dict(fam)
                fam_slim.pop("point_bottlenecks", None)
                fams[fid] = fam_slim
            slim["family_reports"] = fams
            slim_domains[domain_id] = slim
        slim_profiles[profile_id] = {**rep, "domains": slim_domains}

    results_payload = {
        "created_at_utc": created_at.isoformat(),
        "parent_mei0_freeze_dir": _relative_to_root(parent_dir),
        "parent_mei0_manifest_sha256": audit["parent_mei0_manifest_sha256"],
        "noise_profiles": audit.get("noise_profiles"),
        "point_sets": audit.get("point_sets"),
        "formal_point_union": audit.get("formal_point_union"),
        "profile_results": slim_profiles,
        "pressure_domain": audit.get("pressure_domain"),
        "unrepresented_registry_families": audit.get("unrepresented_registry_families"),
        "parked_nonblocking_families": audit.get("parked_nonblocking_families"),
        "flip_events": audit.get("flip_events"),
        "f0_ranking_meta": audit.get("f0_ranking_meta"),
        "delta_numerical_shared_upper_bound": audit.get(
            "delta_numerical_shared_upper_bound"
        ),
        "delta_practical": audit.get("delta_practical"),
    }
    (staging / "mei1_forward_envelope.json").write_bytes(
        dumps_stable(results_payload).encode("utf-8")
    )
    (staging / "family_envelope_report.json").write_bytes(
        dumps_stable(
            {
                "profile_results": slim_profiles,
                "unrepresented_registry_families": audit.get(
                    "unrepresented_registry_families"
                ),
                "parked_nonblocking_families": audit.get(
                    "parked_nonblocking_families"
                ),
            }
        ).encode("utf-8")
    )
    (staging / "noise_profile_comparison.json").write_bytes(
        dumps_stable(
            {
                profile_id: {
                    "baseline_k4_max_p90": rep.get("baseline_k4_max_p90"),
                    "baseline_k4_median_p90": rep.get("baseline_k4_median_p90"),
                    "f0_ranking_meta": rep.get("f0_ranking_meta"),
                    "n_points_formal": rep.get("n_points_formal"),
                }
                for profile_id, rep in slim_profiles.items()
            }
        ).encode("utf-8")
    )
    domain_manifest = {
        "formal_mei1_432": {"n_points": audit.get("n_points")},
        "point_sets": audit.get("point_sets"),
        "noise_profiles": audit.get("noise_profiles"),
    }
    (staging / "domain_point_manifest.json").write_bytes(
        dumps_stable(domain_manifest).encode("utf-8")
    )

    ranking_csv_lines = [
        "noise_profile,point_set,family_id,design_id,rank_numerical,rank_practical,"
        "raw_order,max_p90_o2_percent,median_p90_o2_percent,"
        "ranking_resolvable_practical,ranking_span_relative"
    ]
    for profile_id, rep in slim_profiles.items():
        for domain_id, domain in (rep.get("domains") or {}).items():
            for fid, fam in (domain.get("family_reports") or {}).items():
                for row in fam.get("ranking") or []:
                    ranking_csv_lines.append(
                        f"{profile_id},{domain_id},{fid},{row['design_id']},"
                        f"{row.get('rank_numerical')},{row.get('rank_practical')},"
                        f"{row['raw_order']},"
                        f"{row['max_p90_o2_percent']:.8g},"
                        f"{row['median_p90_o2_percent']:.8g},"
                        f"{row.get('ranking_resolvable_practical')},"
                        f"{row['ranking_span_relative']:.8g}"
                    )
    (staging / "design_ranking.csv").write_text(
        "\n".join(ranking_csv_lines) + "\n", encoding="utf-8"
    )

    output_relative = _relative_to_root(output_dir)
    verdict = {
        "created_at_utc": created_at.isoformat(),
        "config_dir": _relative_to_root(config_dir),
        "parent_mei0_freeze_dir": _relative_to_root(parent_dir),
        "output_dir": output_relative,
        "audit": {
            key: audit[key]
            for key in (
                "verdict",
                "passed",
                "allowed_next_stage",
                "blockers",
                "decision_reason",
                "frozen_design",
                "issues",
                "n_points",
                "n_designs",
                "noise_profiles",
                "point_sets",
                "formal_point_union",
                "pressure_domain",
                "unrepresented_registry_families",
                "parked_nonblocking_families",
                "f0_ranking_meta",
                "authorizations",
                "registered_sparse_simulation_generation_review_eligible",
                "delta_numerical_shared_upper_bound",
                "delta_practical",
                "parent_mei0_manifest_sha256",
                "comsol_holdout_status",
            )
            if key in audit
        },
    }
    (staging / "mei1_verdict.json").write_bytes(dumps_stable(verdict).encode("utf-8"))

    summary_lines = [
        "# tv3 MEI-1 forward envelope (v2)",
        "",
        f"- verdict: `{audit['verdict']}`",
        f"- passed: `{audit['passed']}`",
        f"- allowed_next_stage: `{audit['allowed_next_stage']}`",
        f"- n_points: `{audit.get('n_points')}`",
        f"- n_designs: `{audit.get('n_designs')}`",
        f"- noise_profiles: `{audit.get('noise_profiles')}`",
        f"- formal_waveform_generation: `{FORBIDDEN_AUTH_VALUE}`",
        f"- review_eligible: `{audit.get('registered_sparse_simulation_generation_review_eligible')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = audit.get("blockers") or []
    if blockers:
        for b in blockers:
            summary_lines.append(f"- `{b}`")
    else:
        summary_lines.append("- none")
    summary_lines.append("")
    (staging / "mei1_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    stage_snapshot = {
        "registry_schema_version": audit.get("registry_schema_version"),
        "reserved_benchmark_schema_version": audit.get(
            "reserved_benchmark_schema_version"
        ),
        "allowed_next_stage": audit["allowed_next_stage"],
        "mei0": {
            "verdict": "mei0_registry_frozen",
            "freeze_dir": _relative_to_root(parent_dir),
            "evidence_manifest_sha256": audit["parent_mei0_manifest_sha256"],
        },
        "mei1": {
            "verdict": audit["verdict"],
            "passed": audit["passed"],
            "freeze_dir": output_relative,
            "verdict_path": f"{output_relative}/mei1_verdict.json",
            "evidence_manifest_path": f"{output_relative}/evidence_manifest.json",
            "passed_at_utc": created_at.isoformat(),
            "formal_waveform_generation": FORBIDDEN_AUTH_VALUE,
            "authorizations": audit.get("authorizations"),
            "registered_sparse_simulation_generation_review_eligible": audit.get(
                "registered_sparse_simulation_generation_review_eligible"
            ),
            "delta_practical": audit.get("delta_practical"),
            "delta_numerical_shared_upper_bound": audit.get(
                "delta_numerical_shared_upper_bound"
            ),
            "n_points": audit.get("n_points"),
            "n_designs": audit.get("n_designs"),
            "comsol_holdout_status": audit.get("comsol_holdout_status"),
            "blockers": audit.get("blockers"),
            "decision_reason": audit.get("decision_reason"),
            "frozen_design": audit.get("frozen_design"),
            "unrepresented_registry_families": audit.get(
                "unrepresented_registry_families"
            ),
            "parked_nonblocking_families": audit.get(
                "parked_nonblocking_families"
            ),
            "f0_ranking_meta": audit.get("f0_ranking_meta"),
            "config_sha256": audit.get("registry_sha256"),
        },
    }
    (staging / "stage_status.json").write_bytes(
        dumps_stable(stage_snapshot).encode("utf-8")
    )

    immutable = [
        "mei1_run_config.json",
        "mei1_forward_envelope.json",
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
        "parent_mei0_manifest.json",
        "domain_point_manifest.json",
        "noise_profile_comparison.json",
        "design_ranking.csv",
        "family_envelope_report.json",
        "mei1_verdict.json",
        "mei1_summary.md",
        "stage_status.json",
    ]
    source_paths = {
        "audit_code": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_forward_envelope.py",
        "freeze_script": Path(__file__).resolve(),
        "mrs1_forward": (
            _TV3_ROOT
            / "tv3"
            / "sim"
            / "generation"
            / "tunnel_ventilation"
            / "relaxation_spectrum.py"
        ),
    }
    source_snapshot_dir = staging / "source_snapshots"
    source_snapshot_dir.mkdir()
    source_sha256: dict[str, dict[str, str]] = {
        "mei1_run_config": {
            "path": f"{output_relative}/mei1_run_config.json",
            "sha256": sha256_file(staging / "mei1_run_config.json"),
        }
    }
    for name, path in sorted(source_paths.items()):
        relative_name = f"source_snapshots/{name}{path.suffix or '.bin'}"
        snapshot_path = staging / relative_name
        shutil.copy2(path, snapshot_path)
        immutable.append(relative_name)
        source_sha256[name] = {
            "path": f"{output_relative}/{relative_name}",
            "sha256": sha256_file(snapshot_path),
        }
    manifest = {
        "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "freeze_manifest_schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": created_at.isoformat(),
        "freeze_dir": output_relative,
        "parent_freeze_id": parent_dir.name,
        "parent_manifest_path": _relative_to_root(parent_dir / "evidence_manifest.json"),
        "parent_manifest_sha256": audit["parent_mei0_manifest_sha256"],
        "input_contract_sha256": contract_sha,
        "git_commit": _git_commit(),
        "git_relevant_paths_dirty": _git_relevant_paths_dirty(),
        "artifact_sha256": {name: sha256_file(staging / name) for name in immutable},
        "source_sha256": source_sha256,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    (staging / "evidence_manifest.json").write_bytes(dumps_stable(manifest).encode("utf-8"))
    manifest_sha256 = sha256_file(staging / "evidence_manifest.json")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _promote_staging(staging, output_dir)

    manifest_issues = verify_evidence_manifest(
        output_dir / "evidence_manifest.json",
        project_root=_TV3_ROOT,
        expected_manifest_sha256=manifest_sha256,
    )
    if manifest_issues:
        print(f"frozen evidence verification failed: {manifest_issues}", file=sys.stderr)
        return 4

    # Promote mutable stage_status only after successful freeze write.
    promoted = load_json(config_dir / "stage_status.json")
    promoted["mei1"] = dict(stage_snapshot["mei1"])
    promoted["mei1"]["evidence_manifest_sha256"] = manifest_sha256
    promoted["allowed_next_stage"] = audit["allowed_next_stage"]
    (config_dir / "stage_status.json").write_bytes(dumps_stable(promoted).encode("utf-8"))

    print(f"wrote {output_relative}")
    print(f"evidence_manifest_sha256={manifest_sha256}")
    if audit["passed"]:
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
