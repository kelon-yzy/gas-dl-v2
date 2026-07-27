#!/usr/bin/env python3
"""Freeze the MEI-0 MRS-EI registry into a new immutable evidence directory."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.mrs_ei_registry import (  # noqa: E402
    REGISTRY_FILES,
    audit_mei0_registries,
    compute_delta_num,
    default_config_dir,
    dumps_stable,
    load_json,
    metric_with_delta_num,
    sha256_bytes,
    sha256_file,
    verify_evidence_manifest,
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


def _resolve_output_dir(
    requested: Path | None,
    *,
    created_at: datetime,
    metric_sha256: str,
) -> Path:
    if requested is not None:
        output_dir = requested.resolve()
    else:
        freeze_id = (
            created_at.strftime("%Y%m%dT%H%M%S%fZ")
            + "_"
            + metric_sha256[:12]
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
    return {
        "freeze_script": Path(__file__).resolve(),
        "registry_audit": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_registry.py",
        "fisher_crb": _TV3_ROOT / "tv3" / "audit" / "identifiability_v3_mrs.py",
        "mrs0_registry": _TV3_ROOT / str(mrs0["path"]),
        "mrs1_forward": _TV3_ROOT / str(mrs1["module"]),
        "mrs1_verdict": _TV3_ROOT / str(mrs1["verdict_path"]),
        "mrs2_verdict": _TV3_ROOT / str(mrs2["path"]),
        "mrs6_verdict": _TV3_ROOT / str(mrs6["path"]),
    }


def main() -> int:
    args = _parse_args()
    config_dir = (args.config_dir or default_config_dir()).resolve()
    created_at = datetime.now(timezone.utc)
    if args.output_dir is not None and args.output_dir.resolve().exists():
        raise SystemExit(
            f"refuse overwrite of existing freeze directory: {args.output_dir.resolve()}"
        )

    model = load_json(config_dir / "model_family_registry.json")
    design = load_json(config_dir / "design_space.json")
    metric_path = config_dir / "metric_registry.json"
    metric = load_json(metric_path)

    print("MEI-0: recomputing delta_num on 216 narrow points (obs-cfreq K4)...")
    delta_result = compute_delta_num(design, metric)
    frozen_metric = metric_with_delta_num(metric, delta_result)
    metric_sha256 = sha256_bytes(dumps_stable(frozen_metric).encode("utf-8"))
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
        metric_sha256=metric_sha256,
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

    delta_path = staging_dir / "delta_num_recompute.json"
    delta_path.write_bytes(dumps_stable(delta_result).encode("utf-8"))

    output_relative = _relative_to_root(output_dir)
    stage = {
        "schema_version": "tunnel-ventilation-mrs-ei-1",
        "notes": (
            "MEI-0 freeze directories are append-only. Registry changes require a new "
            "versioned freeze; later-stage statuses are not carried across re-freezes."
        ),
        "registry_files": list(REGISTRY_FILES),
        "allowed_next_stage": audit["allowed_next_stage"],
        "mei0": {
            "verdict": audit["verdict"],
            "registry_sha256": audit["registry_sha256"],
            "freeze_dir": output_relative,
            "verdict_path": f"{output_relative}/mei0_verdict.json",
            "evidence_manifest_path": f"{output_relative}/evidence_manifest.json",
            "passed_at_utc": created_at.isoformat(),
            "formal_waveform_generation": "forbidden_until_authorized",
            "delta_num": frozen_metric["delta_num"]["frozen_value"],
        },
    }
    stage_snapshot_path = staging_dir / "stage_status.json"
    stage_snapshot_path.write_bytes(dumps_stable(stage).encode("utf-8"))

    artifact_paths = {
        name: staging_dir / name
        for name in (
            *REGISTRY_FILES,
            "delta_num_recompute.json",
            "stage_status.json",
        )
    }
    source_paths = _source_evidence_paths(model)
    manifest = {
        "schema_version": "tunnel-ventilation-mrs-ei-freeze-manifest-1",
        "created_at_utc": created_at.isoformat(),
        "freeze_dir": output_relative,
        "artifact_sha256": {
            name: sha256_file(path) for name, path in sorted(artifact_paths.items())
        },
        "source_sha256": {
            name: {
                "path": _relative_to_root(path),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(source_paths.items())
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
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
        "delta_num": frozen_metric["delta_num"]["frozen_value"],
        "delta_num_recompute_summary": {
            "max_relative_change_fd": delta_result["max_relative_change_fd"],
            "max_relative_change_fresh_process_repeat": delta_result[
                "max_relative_change_fresh_process_repeat"
            ],
            "nominal": delta_result["nominal"],
            "fresh_process_repeats": delta_result["fresh_process_repeats"],
            "svd_rank_diagnostics": delta_result["svd_rank_diagnostics"],
            "n_points": delta_result["n_points"],
        },
        "evidence_manifest": {
            "path": f"{output_relative}/evidence_manifest.json",
            "sha256": manifest_sha256,
        },
    }
    (staging_dir / "mei0_verdict.json").write_bytes(
        dumps_stable(verdict).encode("utf-8")
    )
    summary_lines = [
        "# tv3 MEI-0 registry freeze",
        "",
        f"- verdict: `{audit['verdict']}`",
        f"- allowed_next_stage: `{audit['allowed_next_stage']}`",
        f"- delta_num: `{verdict['delta_num']}`",
        f"- evidence_manifest_sha256: `{manifest_sha256}`",
        "- formal_waveform_generation: `forbidden_until_authorized`",
        "",
    ]
    (staging_dir / "mei0_summary.md").write_bytes(
        "\n".join(summary_lines).encode("utf-8")
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.rename(output_dir)
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
