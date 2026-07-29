#!/usr/bin/env python3
"""Freeze the MEI-0 MRS-EI registry into a new immutable evidence directory."""
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

from tv3.audit.mrs_ei_registry import (  # noqa: E402
    AUTHORIZATION_FIELDS,
    FORBIDDEN_AUTH_VALUE,
    FREEZE_MANIFEST_SCHEMA_VERSION,
    REGISTRY_FILES,
    REGISTRY_SCHEMA_VERSION,
    RESERVED_BENCHMARK_SCHEMA_VERSION,
    audit_mei0_registries,
    build_formal_mei1_points,
    build_named_point_set,
    combined_registry_contract_sha256,
    compute_delta_numerical,
    default_config_dir,
    dumps_stable,
    load_json,
    metric_with_delta_numerical,
    sha256_bytes,
    sha256_file,
    verify_evidence_manifest,
)

_PLAN_PATH = (
    _TV3_ROOT
    / "docs"
    / "active"
    / "tv3_mrs_information_efficient_inversion_experiment_plan.md"
)
_GUIDE_PATH = (
    _TV3_ROOT
    / "docs"
    / "active"
    / "tv3_mrs_ei_versioned_refreeze_execution_guide.md"
)
_RELEVANT_PATHS = (
    "configs/tv3_mrs_ei",
    "tv3/audit/mrs_ei_registry.py",
    "tv3/audit/mrs_ei_forward_envelope.py",
    "scripts/run_tv3_mei0_registry_freeze.py",
    "scripts/run_tv3_mei1_forward_envelope.py",
    "tests/test_tunnel_ventilation_mei0_registry.py",
    "tests/test_tunnel_ventilation_mei1_forward_envelope.py",
    "docs/active/tv3_mrs_information_efficient_inversion_experiment_plan.md",
    "docs/active/tv3_mrs_ei_versioned_refreeze_execution_guide.md",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory with MEI-0 registries (default: configs/tv3_mrs_ei)",
    )
    parser.add_argument(
        "--parent-mei0-freeze-dir",
        type=Path,
        required=True,
        help="Immutable parent MEI-0 freeze directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact new freeze directory. It must not exist. By default a versioned "
            "directory is created under outputs/runs/tv3_mrs_ei/mei0_registry/freezes."
        ),
    )
    return parser.parse_args()


def _relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_TV3_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _write_atomic(path: Path, text: str) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    if temp_path.exists():
        raise FileExistsError(f"atomic write staging path already exists: {temp_path}")
    temp_path.write_bytes(text.encode("utf-8"))
    temp_path.replace(path)


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
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_TV3_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_relevant_paths_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--", *_RELEVANT_PATHS],
            cwd=_TV3_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return True
    if result.returncode != 0:
        return True
    return bool(result.stdout.strip())


def _resolve_output_dir(
    requested: Path | None,
    *,
    created_at: datetime,
    contract_sha256: str,
) -> Path:
    if requested is not None:
        output_dir = requested.resolve()
    else:
        freeze_id = (
            created_at.strftime("%Y%m%dT%H%M%S%fZ")
            + "_"
            + contract_sha256[:12]
        )
        output_dir = (
            _TV3_ROOT
            / "outputs"
            / "runs"
            / "tv3_mrs_ei"
            / "mei0_registry"
            / "freezes"
            / freeze_id
        ).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refuse overwrite of existing freeze directory: {output_dir}")
    return output_dir


def _source_evidence_paths(model: dict[str, object]) -> dict[str, Path]:
    lineage = model["lineage"]
    assert isinstance(lineage, dict)
    mrs1 = lineage["mrs1_forward"]
    mrs2 = lineage["mrs2_verdict"]
    mrs6 = lineage["mrs6_verdict"]
    mrs0 = lineage["mrs0_registry"]
    assert isinstance(mrs0, dict)
    assert isinstance(mrs1, dict)
    assert isinstance(mrs2, dict)
    assert isinstance(mrs6, dict)
    paths = {
        "freeze_script": Path(__file__).resolve(),
        "registry_audit": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_registry.py",
        "fisher_crb": _TV3_ROOT / "tv3" / "audit" / "identifiability_v3_mrs.py",
        "mrs0_registry": _TV3_ROOT / str(mrs0["path"]),
        "mrs1_forward": _TV3_ROOT / str(mrs1["module"]),
        "mrs1_verdict": _TV3_ROOT / str(mrs1["verdict_path"]),
        "mrs2_verdict": _TV3_ROOT / str(mrs2["path"]),
        "mrs6_verdict": _TV3_ROOT / str(mrs6["path"]),
    }
    families = model.get("model_families")
    if isinstance(families, list):
        for family in families:
            if not isinstance(family, dict):
                continue
            evidence_path = family.get("evidence_path")
            if isinstance(evidence_path, str) and evidence_path:
                path = Path(evidence_path)
                if not path.is_absolute():
                    path = _TV3_ROOT / path
                paths[f"family_evidence_{family.get('id')}"] = path
    pressure = model.get("pressure_domain_evidence")
    if isinstance(pressure, dict):
        evidence_path = pressure.get("evidence_path")
        if isinstance(evidence_path, str) and evidence_path:
            path = Path(evidence_path)
            if not path.is_absolute():
                path = _TV3_ROOT / path
            paths["pressure_domain_evidence"] = path
    return paths


def _domain_point_manifest(design: dict[str, Any]) -> dict[str, Any]:
    core = build_named_point_set(design, "ambient_core_216")
    pressure = build_named_point_set(design, "pressure_extension_low_rh_216")
    union = build_formal_mei1_points(design)
    return {
        "ambient_core_216": {
            "n_points": len(core),
            "point_ids": [pid for pid, _ in core],
        },
        "pressure_extension_low_rh_216": {
            "n_points": len(pressure),
            "point_ids": [pid for pid, _ in pressure],
        },
        "formal_mei1_432": {
            "n_points": len(union),
            "point_ids": [pid for pid, _ in union],
        },
    }


def _registry_change_log(
    *,
    parent_dir: Path,
    model: dict[str, Any],
    design: dict[str, Any],
    metric: dict[str, Any],
) -> dict[str, Any]:
    changes = [
        "registry_schema_version -> tunnel-ventilation-mrs-ei-registry-2",
        "reserved_benchmark_schema_version kept as tunnel-ventilation-mrs-ei-1",
        "delta_num removed; decision_thresholds.delta_numerical + delta_practical",
        "noise_profiles: low_cost_k4_primary + registered_mrs2_stress (full independent fields)",
        "point_sets: ambient_core_216, pressure_extension_low_rh_216, formal_mei1_432",
        "cost terms renamed to drive budget semantics",
        "statistics_protocols split into three protocols",
        "varpro_observation_contract + raw3 output semantics + stage_transition_policy",
        "authorizations all forbidden_until_explicit_authorization",
    ]
    parent_metric = None
    if (parent_dir / "metric_registry.json").is_file():
        parent_metric = load_json(parent_dir / "metric_registry.json")
    return {
        "parent_freeze_dir": _relative_to_root(parent_dir),
        "parent_had_delta_num": bool(
            isinstance(parent_metric, dict) and "delta_num" in parent_metric
        ),
        "child_has_delta_num": "delta_num" in metric,
        "noise_profile_ids": sorted((design.get("noise_profiles") or {}).keys()),
        "point_set_ids": sorted((design.get("point_sets") or {}).keys()),
        "model_family_ids": [
            f.get("id") for f in (model.get("model_families") or []) if isinstance(f, dict)
        ],
        "changes": changes,
    }


def main() -> int:
    args = _parse_args()
    config_dir = (args.config_dir or default_config_dir()).resolve()
    parent_dir = args.parent_mei0_freeze_dir.resolve()
    created_at = datetime.now(timezone.utc)
    if not parent_dir.is_dir():
        raise SystemExit(f"parent MEI-0 freeze missing: {parent_dir}")
    if args.output_dir is not None and args.output_dir.resolve().exists():
        raise SystemExit(
            f"refuse overwrite of existing freeze directory: {args.output_dir.resolve()}"
        )

    model = load_json(config_dir / "model_family_registry.json")
    design = load_json(config_dir / "design_space.json")
    metric_path = config_dir / "metric_registry.json"
    metric = load_json(metric_path)
    if "delta_num" in metric:
        raise SystemExit("refuse freeze: live metric registry still contains delta_num")

    print(
        "MEI-0: recomputing delta_numerical on ambient_core_216 "
        "for both noise profiles..."
    )
    delta_result = compute_delta_numerical(design, metric)
    frozen_metric = metric_with_delta_numerical(metric, delta_result)
    registries = {
        "model_family_registry.json": model,
        "design_space.json": design,
        "metric_registry.json": frozen_metric,
    }
    contract_sha256 = combined_registry_contract_sha256(registries)
    audit = audit_mei0_registries(
        config_dir,
        project_root=_TV3_ROOT,
        registry_overrides={"metric_registry.json": frozen_metric},
    )
    if not audit["passed"]:
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return 2

    output_dir = _resolve_output_dir(
        args.output_dir,
        created_at=created_at,
        contract_sha256=contract_sha256,
    )
    staging_dir = output_dir.with_name(f".{output_dir.name}.tmp")
    if staging_dir.exists():
        raise FileExistsError(f"freeze staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)

    for name in ("model_family_registry.json", "design_space.json"):
        shutil.copy2(config_dir / name, staging_dir / name)
    (staging_dir / "metric_registry.json").write_bytes(
        dumps_stable(frozen_metric).encode("utf-8")
    )
    for name in REGISTRY_FILES:
        if sha256_file(staging_dir / name) != audit["registry_sha256"][name]:
            raise RuntimeError(f"sha256 mismatch after staging registry {name}")

    delta_path = staging_dir / "numerical_stability_recompute.json"
    delta_path.write_bytes(dumps_stable(delta_result).encode("utf-8"))
    domain_manifest = _domain_point_manifest(design)
    (staging_dir / "domain_point_manifest.json").write_bytes(
        dumps_stable(domain_manifest).encode("utf-8")
    )
    change_log = _registry_change_log(
        parent_dir=parent_dir,
        model=model,
        design=design,
        metric=frozen_metric,
    )
    (staging_dir / "registry_change_log.json").write_bytes(
        dumps_stable(change_log).encode("utf-8")
    )

    shutil.copy2(_PLAN_PATH, staging_dir / "experiment_plan_snapshot.md")
    shutil.copy2(_GUIDE_PATH, staging_dir / "refreeze_execution_guide_snapshot.md")

    output_relative = _relative_to_root(output_dir)
    parent_manifest_path = parent_dir / "evidence_manifest.json"
    parent_manifest_sha256 = (
        sha256_file(parent_manifest_path) if parent_manifest_path.is_file() else None
    )
    if parent_manifest_sha256 is None:
        raise SystemExit(f"parent MEI-0 evidence manifest missing: {parent_manifest_path}")
    plan_snapshot_path = staging_dir / "experiment_plan_snapshot.md"
    plan_sha256 = sha256_file(plan_snapshot_path)
    git_commit = _git_commit()
    git_dirty = _git_relevant_paths_dirty()

    stage = {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "reserved_benchmark_schema_version": RESERVED_BENCHMARK_SCHEMA_VERSION,
        "notes": (
            "MEI-0 freeze directories are append-only. Registry changes require a new "
            "versioned freeze; later-stage statuses are not carried across re-freezes."
        ),
        "registry_files": list(REGISTRY_FILES),
        "allowed_next_stage": "MEI-1_forward_envelope",
        "mei0": {
            "verdict": audit["verdict"],
            "registry_sha256": audit["registry_sha256"],
            "input_contract_sha256": contract_sha256,
            "freeze_dir": output_relative,
            "verdict_path": f"{output_relative}/mei0_verdict.json",
            "evidence_manifest_path": f"{output_relative}/evidence_manifest.json",
            "passed_at_utc": created_at.isoformat(),
            "delta_numerical_by_profile": delta_result["delta_numerical_by_profile"],
            "delta_numerical_shared_upper_bound": delta_result["shared_upper_bound"],
            "delta_practical": 0.02,
            "authorizations": {
                field: FORBIDDEN_AUTH_VALUE for field in AUTHORIZATION_FIELDS
            },
        },
        # Clear stale MEI-1 current pointer on MEI-0 re-freeze.
        "mei1": None,
    }
    stage_snapshot_path = staging_dir / "stage_status.json"
    stage_snapshot_path.write_bytes(dumps_stable(stage).encode("utf-8"))

    artifact_names = (
        *REGISTRY_FILES,
        "numerical_stability_recompute.json",
        "domain_point_manifest.json",
        "registry_change_log.json",
        "stage_status.json",
        "experiment_plan_snapshot.md",
        "refreeze_execution_guide_snapshot.md",
    )
    artifact_paths = {name: staging_dir / name for name in artifact_names}
    source_paths = _source_evidence_paths(model)
    source_snapshot_dir = staging_dir / "source_snapshots"
    source_snapshot_dir.mkdir()
    source_sha256: dict[str, dict[str, str]] = {}
    for name, path in sorted(source_paths.items()):
        suffix = path.suffix or ".bin"
        relative_name = f"source_snapshots/{name}{suffix}"
        snapshot_path = staging_dir / relative_name
        shutil.copy2(path, snapshot_path)
        artifact_paths[relative_name] = snapshot_path
        source_sha256[name] = {
            "path": f"{output_relative}/{relative_name}",
            "sha256": sha256_file(snapshot_path),
        }
    artifact_sha256 = {
        name: sha256_file(path) for name, path in sorted(artifact_paths.items())
    }
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    manifest = {
        "schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "freeze_manifest_schema_version": FREEZE_MANIFEST_SCHEMA_VERSION,
        "created_at_utc": created_at.isoformat(),
        "freeze_dir": output_relative,
        "parent_freeze_id": parent_dir.name,
        "parent_manifest_path": _relative_to_root(parent_manifest_path),
        "parent_manifest_sha256": parent_manifest_sha256,
        "plan_path": f"{output_relative}/experiment_plan_snapshot.md",
        "plan_sha256": plan_sha256,
        "git_commit": git_commit,
        "git_relevant_paths_dirty": git_dirty,
        "input_contract_sha256": contract_sha256,
        "artifact_sha256": artifact_sha256,
        "source_sha256": source_sha256,
        "environment": environment,
    }
    # Include hashes of the hashed payload itself for convenience in readers.
    manifest["artifact_sha256"] = artifact_sha256
    manifest_path = staging_dir / "evidence_manifest.json"
    manifest_path.write_bytes(dumps_stable(manifest).encode("utf-8"))
    manifest_sha256 = sha256_file(manifest_path)
    promoted_stage = json.loads(json.dumps(stage))
    promoted_stage["mei0"]["evidence_manifest_sha256"] = manifest_sha256

    verdict = {
        "created_at_utc": created_at.isoformat(),
        "config_dir": _relative_to_root(config_dir),
        "output_dir": output_relative,
        "audit": audit,
        "delta_numerical_by_profile": delta_result["delta_numerical_by_profile"],
        "delta_numerical_shared_upper_bound": delta_result["shared_upper_bound"],
        "delta_practical": 0.02,
        "numerical_stability_recompute_summary": {
            "shared_upper_bound": delta_result["shared_upper_bound"],
            "by_noise_profile": {
                pid: {
                    "delta_numerical": row["delta_numerical"],
                    "max_relative_change_fd": row["max_relative_change_fd"],
                    "max_relative_change_fresh_process_repeat": row[
                        "max_relative_change_fresh_process_repeat"
                    ],
                    "nominal": row["nominal"],
                    "fresh_process_repeats": row["fresh_process_repeats"],
                    "n_points": row["n_points"],
                }
                for pid, row in delta_result["by_noise_profile"].items()
            },
            "n_points": delta_result["n_points"],
        },
        "evidence_manifest": {
            "path": f"{output_relative}/evidence_manifest.json",
            "sha256": manifest_sha256,
        },
        "authorizations": {
            field: FORBIDDEN_AUTH_VALUE for field in AUTHORIZATION_FIELDS
        },
    }
    (staging_dir / "mei0_verdict.json").write_bytes(
        dumps_stable(verdict).encode("utf-8")
    )
    summary_lines = [
        "# tv3 MEI-0 registry freeze (v2)",
        "",
        f"- verdict: `{audit['verdict']}`",
        f"- allowed_next_stage: `{audit['allowed_next_stage']}`",
        f"- delta_numerical_shared_upper_bound: `{delta_result['shared_upper_bound']}`",
        f"- delta_practical: `0.02`",
        f"- input_contract_sha256: `{contract_sha256}`",
        f"- evidence_manifest_sha256: `{manifest_sha256}`",
        f"- formal_waveform_generation: `{FORBIDDEN_AUTH_VALUE}`",
        f"- point_count_formal: `{domain_manifest['formal_mei1_432']['n_points']}`",
        "",
    ]
    (staging_dir / "mei0_summary.md").write_bytes(
        "\n".join(summary_lines).encode("utf-8")
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _promote_staging(staging_dir, output_dir)
    manifest_issues = verify_evidence_manifest(
        output_dir / "evidence_manifest.json",
        project_root=_TV3_ROOT,
        expected_manifest_sha256=manifest_sha256,
    )
    if manifest_issues:
        raise RuntimeError(f"frozen evidence verification failed: {manifest_issues}")
    _write_atomic(metric_path, dumps_stable(frozen_metric))
    _write_atomic(config_dir / "stage_status.json", dumps_stable(promoted_stage))

    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
