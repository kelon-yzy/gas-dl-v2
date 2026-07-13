"""冻结 tv3 当前 B1 / B7 正式基线，不重跑训练。"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tv3_baseline_freeze"
SUCCESS_STATUSES = frozenset({"ok", "revalidated_exists"})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required baseline evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _git_output(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _require_clean_worktree(project_root: Path) -> str:
    dirty_paths = _git_output(project_root, "status", "--porcelain=v1")
    if dirty_paths:
        raise RuntimeError(
            "formal baseline freeze requires a clean worktree; commit or otherwise resolve these paths first:\n"
            f"{dirty_paths}"
        )
    return _git_output(project_root, "rev-parse", "HEAD")


def _artifact_paths(project_root: Path) -> dict[str, Path]:
    return {
        "b1_config": project_root / "configs" / "tv3_d2b_raw_dsp_ridge.json",
        "b7_config": project_root / "configs" / "tv3_d2b_oof_ridge_residual_mlp.json",
        "fidelity_metrics": project_root / "outputs" / "tv3_d2b" / "raw_dsp_frame_fidelity" / "metrics.json",
        "b1_metrics": project_root / "outputs" / "tv3_d2b" / "raw_dsp_ridge_provenance" / "metrics.json",
        "protocol_manifest": project_root / "outputs" / "tv3_b7_protocol" / "protocol_manifest.json",
        "protocol_metrics": project_root / "outputs" / "tv3_b7_protocol" / "split_metrics.json",
        "protocol_runs": project_root / "outputs" / "tv3_b7_protocol" / "runs.jsonl",
    }


def _collect_protocol_records(
    runs_path: Path,
    protocol_matrix: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    expected_seeds = {
        (row["protocol_id"], row["dataset_name"]): row["split_seed"] for row in protocol_matrix
    }
    expected = {(*identity, split_seed) for identity, split_seed in expected_seeds.items()}
    derived: dict[tuple[str, str, int], dict[str, Any]] = {}
    raw_dsp: dict[tuple[str, str, int], dict[str, Any]] = {}
    errors: list[str] = []

    for line_number, line in enumerate(runs_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        stage = record.get("stage")
        if stage not in {"derive", "raw_dsp"}:
            continue
        if record.get("status") not in SUCCESS_STATUSES:
            errors.append(f"runs.jsonl:{line_number} {stage} status={record.get('status')!r}")
            continue
        identity = (record.get("protocol_id"), record.get("dataset_name"))
        expected_seed = expected_seeds.get(identity)
        if expected_seed is None:
            errors.append(f"runs.jsonl:{line_number} has an unknown protocol row {identity!r}")
            continue
        record_seed = record.get("split_seed")
        if stage == "derive" and record_seed != expected_seed:
            errors.append(f"runs.jsonl:{line_number} derived split_seed={record_seed!r}")
            continue
        if stage == "raw_dsp" and record_seed is not None and record_seed != expected_seed:
            errors.append(f"runs.jsonl:{line_number} RawDSP split_seed={record_seed!r}")
            continue
        key = (*identity, expected_seed)
        if not isinstance(record.get("split_hash"), str) or not record["split_hash"]:
            errors.append(f"runs.jsonl:{line_number} is missing split_hash")
            continue
        if stage == "derive":
            payload = {
                "protocol_id": key[0],
                "dataset_name": key[1],
                "split_seed": key[2],
                "split_hash": record["split_hash"],
                "ood_set_hash": record.get("ood_set_hash"),
                "x_feature_profile": record.get("x_feature_profile"),
            }
            previous = derived.get(key)
            if previous is not None and previous != payload:
                errors.append(f"runs.jsonl has inconsistent derived split evidence for {key!r}")
            derived[key] = payload
            continue

        payload = {
            "protocol_id": key[0],
            "dataset_name": key[1],
            "split_seed": key[2],
            "split_hash": record["split_hash"],
            "build_signature": record.get("build_signature"),
            "template_digest": record.get("template_digest"),
        }
        if not isinstance(payload["build_signature"], str) or not payload["build_signature"]:
            errors.append(f"runs.jsonl:{line_number} is missing RawDSP build_signature")
        if not isinstance(payload["template_digest"], str) or not payload["template_digest"]:
            errors.append(f"runs.jsonl:{line_number} is missing RawDSP template_digest")
        previous = raw_dsp.get(key)
        if previous is not None and previous != payload:
            errors.append(f"runs.jsonl has inconsistent RawDSP evidence for {key!r}")
        raw_dsp[key] = payload

    if set(derived) != expected:
        errors.append("runs.jsonl derived rows do not cover the frozen protocol matrix")
    if set(raw_dsp) != expected:
        errors.append("runs.jsonl RawDSP rows do not cover the frozen protocol matrix")
    for key in expected & set(derived) & set(raw_dsp):
        if derived[key]["split_hash"] != raw_dsp[key]["split_hash"]:
            errors.append(f"RawDSP split_hash does not match derived split for {key!r}")

    return (
        [derived[key] for key in sorted(derived)],
        [raw_dsp[key] for key in sorted(raw_dsp)],
        errors,
    )


def _audit_evidence(project_root: Path) -> tuple[dict[str, Any], list[str]]:
    paths = _artifact_paths(project_root)
    b1_config = _read_json(paths["b1_config"])
    b7_config = _read_json(paths["b7_config"])
    fidelity = _read_json(paths["fidelity_metrics"])
    b1_metrics = _read_json(paths["b1_metrics"])
    protocol_manifest = _read_json(paths["protocol_manifest"])
    protocol_metrics = _read_json(paths["protocol_metrics"])
    errors: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    invariants = protocol_manifest.get("invariants", {})
    verdict = protocol_metrics.get("verdict", {})
    expected_rows = len(protocol_manifest.get("matrix", [])) * len(protocol_manifest.get("training_seeds", []))
    expect(b1_config.get("dataset_dir") == "data/tv3-formal-6000", "B1 config dataset_dir is not tv3-formal-6000")
    expect(b7_config.get("dataset_dir") == "data/tv3-formal-6000", "B7 config dataset_dir is not tv3-formal-6000")
    expect(b1_config.get("feature_builder") == "d0_raw_dsp_physics_stats_v1", "B1 feature builder changed")
    expect(b7_config.get("feature_builder") == "d0_raw_dsp_physics_stats_v1", "B7 feature builder changed")
    expect(b7_config.get("head") == "oof_ridge_residual_mlp", "B7 head changed")
    expect(protocol_manifest.get("is_complete_formal_matrix") is True, "B7 protocol matrix is not formal and complete")
    expect(invariants.get("feature_builder") == "d0_raw_dsp_physics_stats_v1", "protocol feature builder changed")
    expect(invariants.get("feature_count") == 1008, "protocol feature count changed")
    expect(invariants.get("head") == "oof_ridge_residual_mlp", "protocol head changed")
    expect(invariants.get("early_stopping") == "val_only", "protocol early stopping is not val_only")
    expect(invariants.get("skip_source_raw_dsp_hardlink") is True, "protocol may reuse split-dependent RawDSP cache")
    expect(protocol_manifest.get("b1_config_sha256") == _sha256(paths["b1_config"]), "B1 config hash differs from protocol evidence")
    expect(protocol_manifest.get("b7_config_sha256") == _sha256(paths["b7_config"]), "B7 config hash differs from protocol evidence")
    expect(verdict.get("protocol_pass") is True, "B7 protocol verdict is not protocol_pass")
    expect(verdict.get("matrix_complete") is True, "B7 metric matrix is incomplete")
    expect(verdict.get("unexpected_row_count") == 0, "B7 metric matrix has unexpected rows")
    expect(len(protocol_metrics.get("rows", [])) == expected_rows == 36, "B7 result row count is not 36")
    expect(fidelity.get("status") == "passed", "RawDSP frame fidelity is not passed")
    expect(fidelity.get("source", {}).get("template_mode") == "train_baseline_median", "RawDSP fidelity template mode changed")
    expect(fidelity.get("source", {}).get("template_source_split") == "train", "RawDSP fidelity template is not train-only")
    provenance = b1_metrics.get("raw_dsp_provenance", {})
    expect(b1_metrics.get("feature_builder") == "d0_raw_dsp_physics_stats_v1", "B1 metrics feature builder changed")
    expect(b1_metrics.get("feature_count") == 1008, "B1 metrics feature count changed")
    expect(provenance.get("diagnostic_only") is False, "B1 RawDSP provenance is diagnostic-only")
    expect(provenance.get("complete_dataset") is True, "B1 RawDSP provenance is incomplete")
    expect(
        provenance.get("build_signature") == fidelity.get("source", {}).get("cache_build_signature"),
        "B1 provenance and fidelity cache build signatures differ",
    )
    expect(
        provenance.get("template_digest") == fidelity.get("source", {}).get("template_digest"),
        "B1 provenance and fidelity template digests differ",
    )
    split_hashes, raw_dsp_caches, run_errors = _collect_protocol_records(
        paths["protocol_runs"], protocol_manifest.get("matrix", [])
    )
    errors.extend(run_errors)

    return {
        "paths": paths,
        "b1_metrics": b1_metrics,
        "fidelity": fidelity,
        "protocol_manifest": protocol_manifest,
        "protocol_metrics": protocol_metrics,
        "split_hashes": split_hashes,
        "raw_dsp_caches": raw_dsp_caches,
    }, errors


def _environment() -> dict[str, Any]:
    packages = ("numpy", "scikit-learn", "torch")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }


def freeze_baseline(project_root: Path, output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"baseline freeze output already exists: {output_dir}")
    commit = _require_clean_worktree(project_root)
    evidence, errors = _audit_evidence(project_root)
    if errors:
        detail = "\n".join(f"- {error}" for error in errors)
        raise RuntimeError(f"formal baseline freeze audit failed:\n{detail}")

    paths: dict[str, Path] = evidence["paths"]
    protocol_manifest = evidence["protocol_manifest"]
    protocol_metrics = evidence["protocol_metrics"]
    b1_metrics = evidence["b1_metrics"]
    fidelity = evidence["fidelity"]
    output_dir.mkdir(parents=True)
    artifact_hashes = {
        name: {"path": str(path.relative_to(project_root)), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    frozen_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "tv3-baseline-freeze-1",
            "frozen_at": frozen_at,
            "git": {"commit": commit, "worktree_clean": True},
            "dataset": {"name": "tv3-formal-6000", "schema_version": "tunnel-ventilation-1"},
            "contract": {
                "component_fields": ["x_CO2", "x_O2", "x_N2"],
                "output": "raw3",
                "feature_builder": protocol_manifest["invariants"]["feature_builder"],
                "feature_count": protocol_manifest["invariants"]["feature_count"],
            },
            "models": {"b1_head": "ridgecv", "b7_head": protocol_manifest["invariants"]["head"]},
            "artifacts": artifact_hashes,
        },
    )
    _write_json(
        output_dir / "metrics.json",
        {
            "b1_evaluations": b1_metrics["evaluations"],
            "b7_protocol_verdict": protocol_metrics["verdict"],
            "frame_fidelity_status": fidelity["status"],
        },
    )
    _write_json(
        output_dir / "split_hashes.json",
        {"derived_splits": evidence["split_hashes"], "raw_dsp_caches": evidence["raw_dsp_caches"]},
    )
    _write_json(output_dir / "environment.json", _environment())
    _write_json(
        output_dir / "audit.json",
        {
            "status": "passed",
            "checks": [
                "clean_git_worktree",
                "B1_B7_config_hashes_match_protocol",
                "RawDSP_fidelity_and_B1_provenance_match",
                "B7_36_row_protocol_pass",
                "derived_split_and_RawDSP_hashes_complete",
            ],
        },
    )
    _write_json(
        output_dir / "verdict.json",
        {
            "status": "frozen",
            "reason": "B1/B7 evidence, split provenance, RawDSP fidelity, code revision and environment are consistent.",
        },
    )
    (output_dir / "README.md").write_text(
        "# tv3 B1/B7 正式基线冻结\n\n"
        "此目录由 `scripts/freeze_tv3_baseline.py` 生成，记录后续可辨识性审计的唯一比较基线。"
        "产物不可覆盖；基线更新必须使用新的目录和新的正式协议。\n",
        encoding="utf-8",
    )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freeze the formal tv3 B1/B7 baseline")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    output_dir = freeze_baseline(PROJECT_ROOT, args.output_dir)
    print(f"frozen tv3 B1/B7 baseline: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
