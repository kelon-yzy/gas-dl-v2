#!/usr/bin/env python3
"""Run MEI-1 forward-envelope audit and write append-only verdict artifacts.

Prerequisite: MEI-0 ``mei0_registry_frozen``. Does not generate waveforms.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.mrs_ei_forward_envelope import run_mei1_audit  # noqa: E402
from tv3.audit.mrs_ei_registry import dumps_stable, load_json, sha256_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_TV3_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _source_evidence(stage: dict[str, Any]) -> dict[str, dict[str, str]]:
    paths = {
        "audit_code": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_forward_envelope.py",
        "freeze_script": _TV3_ROOT / "scripts" / "run_tv3_mei1_forward_envelope.py",
        "mrs1_forward": (
            _TV3_ROOT
            / "tv3"
            / "sim"
            / "generation"
            / "tunnel_ventilation"
            / "relaxation_spectrum.py"
        ),
        "fisher_crb": _TV3_ROOT / "tv3" / "audit" / "identifiability_v3_mrs.py",
    }
    out: dict[str, dict[str, str]] = {
        name: {"path": _relative_to_root(path), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }
    mei0 = stage.get("mei0") or {}
    mei0_manifest = mei0.get("evidence_manifest_path")
    mei0_sha = mei0.get("evidence_manifest_sha256")
    if not mei0_manifest or not mei0_sha:
        raise SystemExit("MEI-0 evidence_manifest_path/sha256 missing from stage_status")
    mei0_path = _TV3_ROOT / str(mei0_manifest)
    if not mei0_path.is_file():
        raise SystemExit(f"MEI-0 evidence manifest missing: {mei0_path}")
    actual = sha256_file(mei0_path)
    if actual != str(mei0_sha):
        raise SystemExit(
            "MEI-0 evidence manifest sha256 mismatch: "
            f"stage_status={mei0_sha} file={actual}"
        )
    out["mei0_evidence_manifest"] = {
        "path": _relative_to_root(mei0_path),
        "sha256": actual,
    }
    return out


def main() -> int:
    args = _parse_args()
    config_dir = (
        args.config_dir.resolve()
        if args.config_dir is not None
        else (_TV3_ROOT / "configs" / "tv3_mrs_ei").resolve()
    )
    created_at = datetime.now(timezone.utc)
    existing_stage = load_json(config_dir / "stage_status.json")

    print("MEI-1: running forward-envelope audit...")
    audit = run_mei1_audit(project_root=_TV3_ROOT, config_dir=config_dir)
    print(
        json.dumps(
            {
                "verdict": audit["verdict"],
                "passed": audit["passed"],
                "n_points": audit["n_points"],
                "n_designs": audit["n_designs"],
                "blockers": audit.get("blockers"),
                "flip_events": audit["flip_events"],
                "unrepresented_registry_families": audit["unrepresented_registry_families"],
                "f0_ranking_meta": audit.get("f0_ranking_meta"),
                "comsol_holdout_status": audit["comsol_holdout_status"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    stamp = created_at.strftime("%Y%m%dT%H%M%S%fZ")
    short = audit["registry_sha256"]["mei1_forward_envelope.json"][:12]
    default_out = (
        _TV3_ROOT
        / "outputs"
        / "runs"
        / "tv3_mrs_ei"
        / "mei1_forward_envelope"
        / "freezes"
        / f"{stamp}_{short}"
    )
    output_dir = args.output_dir.resolve() if args.output_dir is not None else default_out
    if output_dir.exists():
        raise SystemExit(f"refuse overwrite of existing freeze directory: {output_dir}")

    staging = output_dir.with_name(f".{output_dir.name}.tmp")
    if staging.exists():
        raise FileExistsError(f"staging exists: {staging}")
    staging.mkdir(parents=True)

    slim_families = {}
    for fid, rep in audit["family_reports"].items():
        slim = dict(rep)
        slim.pop("point_bottlenecks", None)
        slim_families[fid] = slim
    audit_slim = dict(audit)
    audit_slim["family_reports"] = slim_families

    ranking_csv_lines = [
        "family_id,design_id,rank,raw_order,max_p90_o2_percent,median_p90_o2_percent,"
        "ranking_resolvable,ranking_span_relative"
    ]
    for fid, rep in slim_families.items():
        for row in rep["ranking"]:
            ranking_csv_lines.append(
                f"{fid},{row['design_id']},{row['rank']},{row['raw_order']},"
                f"{row['max_p90_o2_percent']:.8g},{row['median_p90_o2_percent']:.8g},"
                f"{row['ranking_resolvable']},{row['ranking_span_relative']:.8g}"
            )
    (staging / "design_ranking.csv").write_text(
        "\n".join(ranking_csv_lines) + "\n", encoding="utf-8"
    )
    shutil.copy2(config_dir / "mei1_forward_envelope.json", staging / "mei1_forward_envelope.json")
    for name in (
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
    ):
        shutil.copy2(config_dir / name, staging / name)

    output_relative = _relative_to_root(output_dir)
    verdict = {
        "created_at_utc": created_at.isoformat(),
        "config_dir": _relative_to_root(config_dir),
        "output_dir": output_relative,
        "audit": audit_slim,
    }
    (staging / "mei1_verdict.json").write_bytes(dumps_stable(verdict).encode("utf-8"))

    summary_lines = [
        "# tv3 MEI-1 forward envelope",
        "",
        f"- verdict: `{audit['verdict']}`",
        f"- passed: `{audit['passed']}`",
        f"- allowed_next_stage: `{audit['allowed_next_stage']}`",
        f"- n_points: `{audit['n_points']}`",
        f"- n_designs: `{audit['n_designs']}`",
        f"- baseline_design: `{audit['baseline_design_id']}`",
        f"- comsol_holdout: `{audit['comsol_holdout_status']}`",
        f"- formal_waveform_generation: `{audit['formal_waveform_generation']}`",
        f"- ranking_resolvable: `{audit.get('f0_ranking_meta', {}).get('ranking_resolvable')}`",
        f"- ranking_span_relative: `{audit.get('f0_ranking_meta', {}).get('ranking_span_relative')}`",
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
    summary_lines.append("## Unrepresented registry families")
    summary_lines.append("")
    for fid in audit["unrepresented_registry_families"]:
        summary_lines.append(f"- `{fid}`")
    if not audit["unrepresented_registry_families"]:
        summary_lines.append("- none")
    summary_lines.append("")
    summary_lines.append("## Flip events")
    summary_lines.append("")
    if audit["flip_events"]:
        for ev in audit["flip_events"]:
            summary_lines.append(f"- `{ev['family_id']}`: {', '.join(ev['reasons'])}")
    else:
        summary_lines.append("- none")
    summary_lines.append("")
    summary_lines.append("## Family top class vs F0")
    summary_lines.append("")
    for fid, rep in slim_families.items():
        ang = rep.get("principal_angle_gate_value_deg")
        summary_lines.append(
            f"- `{fid}`: top1=`{rep['top1_design_id']}`, "
            f"top_class={rep.get('top_equivalence_class')}, "
            f"spearman={rep['spearman_vs_f0']}, "
            f"angle_gate_deg={ang}, "
            f"bot_flip={rep['bottleneck_flip_fraction_vs_f0']}"
        )
    summary_lines.append("")
    (staging / "mei1_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    # Freeze snapshot hashed before embedding evidence_manifest_sha256 (MEI-0 style).
    stage_snapshot = {
        "schema_version": "tunnel-ventilation-mrs-ei-1",
        "notes": existing_stage.get("notes"),
        "registry_files": existing_stage.get("registry_files"),
        "allowed_next_stage": audit["allowed_next_stage"],
        "mei0": existing_stage.get("mei0"),
        "mei1": {
            "verdict": audit["verdict"],
            "passed": audit["passed"],
            "freeze_dir": output_relative,
            "verdict_path": f"{output_relative}/mei1_verdict.json",
            "evidence_manifest_path": f"{output_relative}/evidence_manifest.json",
            "passed_at_utc": created_at.isoformat(),
            "formal_waveform_generation": "forbidden_until_authorized",
            "delta_num": audit["delta_num"],
            "n_points": audit["n_points"],
            "n_designs": audit["n_designs"],
            "comsol_holdout_status": audit["comsol_holdout_status"],
            "blockers": audit.get("blockers"),
            "unrepresented_registry_families": audit["unrepresented_registry_families"],
            "f0_ranking_meta": audit.get("f0_ranking_meta"),
            "config_sha256": audit["registry_sha256"],
        },
    }
    (staging / "stage_status.json").write_bytes(dumps_stable(stage_snapshot).encode("utf-8"))

    immutable = [
        "mei1_forward_envelope.json",
        "model_family_registry.json",
        "design_space.json",
        "metric_registry.json",
        "design_ranking.csv",
        "mei1_verdict.json",
        "mei1_summary.md",
        "stage_status.json",
    ]
    manifest = {
        "schema_version": "tunnel-ventilation-mrs-ei-freeze-manifest-1",
        "created_at_utc": created_at.isoformat(),
        "freeze_dir": output_relative,
        "artifact_sha256": {name: sha256_file(staging / name) for name in immutable},
        "source_sha256": _source_evidence(existing_stage),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "notes": (
            "No mutable copies of verdict/summary/stage_status are written outside this "
            "freeze directory. evidence_manifest_sha256 is recorded on promoted "
            "configs/tv3_mrs_ei/stage_status.json after freeze, matching MEI-0."
        ),
    }
    (staging / "evidence_manifest.json").write_bytes(dumps_stable(manifest).encode("utf-8"))
    manifest_sha256 = sha256_file(staging / "evidence_manifest.json")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(output_dir)

    promoted = load_json(config_dir / "stage_status.json")
    promoted["mei1"] = dict(stage_snapshot["mei1"])
    promoted["mei1"]["evidence_manifest_sha256"] = manifest_sha256
    promoted["allowed_next_stage"] = audit["allowed_next_stage"]
    (config_dir / "stage_status.json").write_bytes(dumps_stable(promoted).encode("utf-8"))

    print(f"wrote {output_relative}")
    print(f"evidence_manifest_sha256={manifest_sha256}")
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
