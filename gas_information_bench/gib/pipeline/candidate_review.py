"""Deterministic P3 G3-4 review derived only from verified candidate freezes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping
from uuid import uuid4

from ..common.io import atomic_promote_directory, atomic_write_json, remove_owned_staging, sha256_file
from ..freeze import verify_evidence_manifest


def review_candidates(candidate_freezes: Mapping[str, Path], output_dir: Path) -> dict[str, object]:
    if not candidate_freezes:
        raise ValueError("G3-4 requires candidate freezes")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"attempt directory already exists: {target}")
    candidates = []
    for candidate_id, freeze_dir in candidate_freezes.items():
        verification = verify_evidence_manifest(freeze_dir)
        manifest_path = Path(freeze_dir) / "attempt" / "attempt_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verdict = manifest.get("candidate_verdict")
        if verdict is None and manifest.get("task_id") == "P3-06":
            verdict = "not_applicable"
        if verdict not in {"enter_P4", "reject", "not_activated", "inconclusive", "not_applicable"}:
            raise ValueError(f"candidate freeze has no terminal verdict: {candidate_id}")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "task_id": manifest["task_id"],
                "candidate_verdict": verdict,
                "freeze_id": verification["freeze_id"],
                "evidence_manifest_sha256": sha256_file(Path(freeze_dir) / "evidence_manifest.json"),
            }
        )
    enter_count = sum(item["candidate_verdict"] == "enter_P4" for item in candidates)
    result: dict[str, object] = {
        "schema_version": "gib-benchmark-1",
        "task_id": "P3-13",
        "task_status": "completed",
        "gate_verdict": "pass" if enter_count else "fail",
        "p3_verdict": "pass" if enter_count else "return_to_P1_P2",
        "enter_P4_count": enter_count,
        "candidates": candidates,
        "next_allowed_task": "P3-14" if enter_count else "return_to_P1_P2",
    }
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        atomic_write_json(staging / "g3_4_summary.json", result)
        atomic_write_json(
            staging / "attempt_manifest.json",
            {
                "schema_version": "gib-benchmark-1",
                "attempt_id": target.name,
                "task_id": "P3-13",
                "status": "complete",
                "task_status": "completed",
                "gate_verdict": result["gate_verdict"],
                "p3_verdict": result["p3_verdict"],
                "next_allowed_task": result["next_allowed_task"],
            },
        )
        atomic_promote_directory(staging, target)
    except Exception:
        remove_owned_staging(staging)
        raise
    return result


__all__ = ["review_candidates"]
