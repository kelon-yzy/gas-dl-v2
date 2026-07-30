#!/usr/bin/env python3
"""Freeze the independently versioned MEI-4 C0 execution contract."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import binom

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-config", type=Path, default=None)
    parser.add_argument("--stage-status-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()

def _relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_TV3_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()

def _resolve_from_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _TV3_ROOT / path

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

def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{field} must be a SHA256 digest")
    return value

def _verify_parent(
    *,
    name: str,
    specification: Mapping[str, Any],
) -> tuple[Path, Path]:
    freeze_dir = _resolve_from_root(str(specification["freeze_dir"]))
    manifest_path = freeze_dir / "evidence_manifest.json"
    expected_digest = _require_digest(
        specification.get("evidence_manifest_sha256"),
        field=f"parent_freezes.{name}.evidence_manifest_sha256",
    )
    issues = verify_evidence_manifest(
        manifest_path,
        project_root=_TV3_ROOT,
        expected_manifest_sha256=expected_digest,
    )
    if issues:
        raise RuntimeError(f"{name} manifest verification failed: {issues}")
    return freeze_dir, manifest_path

def _verify_b5_stage_status(
    *, contract: Mapping[str, Any], stage_status: Mapping[str, Any]
) -> None:
    mei3 = stage_status.get("mei3")
    if not isinstance(mei3, Mapping) or mei3.get("phase") != "b5_verdict_freeze":
        raise RuntimeError("C0 requires stage_status to retain the completed MEI-3 B5 closure")
    b5 = contract["parent_freezes"]["b5"]
    if mei3.get("freeze_dir") != b5["freeze_dir"]:
        raise RuntimeError("stage_status B5 freeze_dir does not match the C0 contract")
    if mei3.get("evidence_manifest_sha256") != b5["evidence_manifest_sha256"]:
        raise RuntimeError("stage_status B5 manifest digest does not match the C0 contract")
    if mei3.get("verdict") != b5["verdict"] or mei3.get("mei4_baseline") != "S1":
        raise RuntimeError("C0 requires the B5 S1 baseline verdict")

def _verify_b5_verdict(b5_dir: Path, contract: Mapping[str, Any]) -> None:
    verdict = load_json(b5_dir / "mei3_b5_verdict.json")
    b5 = contract["parent_freezes"]["b5"]
    closure = verdict.get("closure") or {}
    if verdict.get("verdict") != b5["verdict"]:
        raise RuntimeError("B5 verdict does not match the C0 contract")
    if verdict.get("mei4_baseline") != b5["mei4_baseline"]:
        raise RuntimeError("B5 did not retain S1 as the MEI-4 baseline")
    if closure.get("no_mei4_transition") is not True:
        raise RuntimeError("C0 requires B5 to preserve its explicit no-transition closure")

def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)

def _nested_fields(row: Mapping[str, Any]) -> dict[str, list[str]]:
    return {
        name: sorted(value)
        for name, value in row.items()
        if isinstance(value, Mapping)
    }

def _inventory_record(
    *,
    b4_dir: Path,
    b4_manifest: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    relative_path = str(record["path"])
    path = b4_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"B4 asset missing: {path}")
    expected_digest = (b4_manifest.get("artifact_sha256") or {}).get(relative_path)
    if expected_digest != sha256_file(path):
        raise RuntimeError(f"B4 manifest asset binding failed: {relative_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    common = {
        "asset_id": str(record["asset_id"]),
        "path": _relative_to_root(path),
        "sha256": sha256_file(path),
        "line_count": _line_count(path),
        "kind": str(record["kind"]),
    }
    if record["kind"] == "tables":
        if not isinstance(data, Mapping):
            raise RuntimeError(f"{relative_path} must be a table object")
        expected_rows = record["expected_rows"]
        if set(data) != set(expected_rows):
            raise RuntimeError(f"{relative_path} table names do not match the C0 contract")
        tables = {}
        for name, expected_count in expected_rows.items():
            rows = data[name]
            if not isinstance(rows, list) or len(rows) != int(expected_count) or not rows:
                raise RuntimeError(f"{relative_path}.{name} row count does not match C0")
            if not isinstance(rows[0], Mapping):
                raise RuntimeError(f"{relative_path}.{name} rows must be objects")
            tables[name] = {
                "row_count": len(rows),
                "fields": sorted(rows[0]),
                "nested_fields": _nested_fields(rows[0]),
            }
        return {**common, "tables": tables}
    if record["kind"] == "rows":
        expected_count = int(record["expected_rows"])
        if not isinstance(data, list) or len(data) != expected_count or not data:
            raise RuntimeError(f"{relative_path} row count does not match C0")
        if not isinstance(data[0], Mapping):
            raise RuntimeError(f"{relative_path} rows must be objects")
        return {
            **common,
            "row_count": len(data),
            "fields": sorted(data[0]),
            "nested_fields": _nested_fields(data[0]),
        }
    if record["kind"] == "object":
        if not isinstance(data, Mapping):
            raise RuntimeError(f"{relative_path} must be an object")
        return {**common, "object_field_count": len(data), "fields": sorted(data)}
    raise RuntimeError(f"unsupported C0 asset kind: {record['kind']!r}")

def _build_asset_inventory(
    *, b4_dir: Path, b4_manifest: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    records = contract["asset_inventory"]["records"]
    inventory = [
        _inventory_record(b4_dir=b4_dir, b4_manifest=b4_manifest, record=record)
        for record in records
    ]
    if len({row["asset_id"] for row in inventory}) != len(inventory):
        raise RuntimeError("C0 asset IDs must be unique")
    return {
        "parent_b4_freeze_dir": _relative_to_root(b4_dir),
        "parent_b4_manifest_sha256": sha256_file(b4_dir / "evidence_manifest.json"),
        "assets": inventory,
        "method_read_whitelist": contract["method_input_policy"]["allowed_assets"],
        "audit_only_fields": contract["method_input_policy"]["audit_only_fields"],
        "forbidden_method_fields": contract["method_input_policy"]["forbidden_method_fields"],
    }

def _validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != "tunnel-ventilation-mrs-ei-mei4-contract-1":
        raise RuntimeError("unsupported MEI-4 C0 contract schema")
    if contract.get("phase") != "c0_execution_contract_freeze":
        raise RuntimeError("C0 contract phase is invalid")
    if contract["frozen_design"]["baseline_solver"] != "S1":
        raise RuntimeError("MEI-4 must retain S1 as its deterministic baseline")
    if contract["frozen_design"]["frequencies_hz"] != [25000.0, 63000.0, 100000.0, 200000.0]:
        raise RuntimeError("MEI-4 must retain the frozen D0 K4 frequencies")
    gate = contract["calibration_gate"]
    if int(gate["samples_per_domain"]) != 648 or int(gate["family_size"]) != 24:
        raise RuntimeError("C0 calibration gate must cover 24 bands of 648 samples")
    if not np.isclose(float(gate["sidak_alpha_each"]), 1.0 - 0.95 ** (1.0 / 24.0)):
        raise RuntimeError("C0 Sidak alpha does not match the declared family")
    expected_levels = {"0.5", "0.8", "0.9", "0.95"}
    if set(gate["exact_binomial_acceptance_counts"]) != expected_levels:
        raise RuntimeError("C0 must freeze every nominal coverage acceptance interval")
    for level_text, frozen in gate["exact_binomial_acceptance_counts"].items():
        level = float(level_text)
        expected = {
            "lower_inclusive": int(binom.ppf(float(gate["sidak_alpha_each"]) / 2.0, 648, level)),
            "upper_inclusive": int(binom.isf(float(gate["sidak_alpha_each"]) / 2.0, 648, level)),
        }
        if frozen != expected:
            raise RuntimeError(f"C0 exact binomial interval mismatch for {level_text}")
    if contract["authorizations"]["registered_sparse_simulation_generation"] == "authorized":
        raise RuntimeError("C0 cannot inherit the MEI-3 generation authorization")

def _promote_stage_status(
    path: Path,
    *,
    contract: Mapping[str, Any],
    b4_manifest_sha256: str,
    b5_manifest_sha256: str,
    freeze_dir: str,
    manifest_sha256: str,
    created_at_utc: str,
) -> None:
    status = load_json(path)
    _verify_b5_stage_status(contract=contract, stage_status=status)
    status["allowed_next_stage"] = None
    status["mei4"] = {
        "phase": contract["phase"],
        "status": "mei4_contract_frozen",
        "freeze_dir": freeze_dir,
        "execution_contract_path": f"{freeze_dir}/mei4_execution_contract.json",
        "evidence_manifest_path": f"{freeze_dir}/evidence_manifest.json",
        "evidence_manifest_sha256": manifest_sha256,
        "created_at_utc": created_at_utc,
        "baseline_solver": contract["frozen_design"]["baseline_solver"],
        "parent_b4_manifest_sha256": b4_manifest_sha256,
        "parent_b5_manifest_sha256": b5_manifest_sha256,
        "authorizations": contract["authorizations"],
        "mc_review_eligible": False,
        "allowed_next_stage": None,
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
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei4_execution_contract.json"
    )
    stage_path = (
        args.stage_status_path.resolve()
        if args.stage_status_path is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    )
    contract = load_json(contract_path)
    stage_status = load_json(stage_path)
    _validate_contract(contract)
    _verify_b5_stage_status(contract=contract, stage_status=stage_status)
    b4_dir, b4_manifest_path = _verify_parent(
        name="b4", specification=contract["parent_freezes"]["b4"]
    )
    b5_dir, b5_manifest_path = _verify_parent(
        name="b5", specification=contract["parent_freezes"]["b5"]
    )
    _verify_b5_verdict(b5_dir, contract)
    b4_manifest = load_json(b4_manifest_path)
    inventory = _build_asset_inventory(
        b4_dir=b4_dir, b4_manifest=b4_manifest, contract=contract
    )
    created_at = datetime.now(timezone.utc)
    input_contract = {
        "mei4_execution_contract_sha256": sha256_file(contract_path),
        "parent_b4_manifest_sha256": sha256_file(b4_manifest_path),
        "parent_b5_manifest_sha256": sha256_file(b5_manifest_path),
        "b4_asset_inventory_sha256": sha256_bytes(dumps_stable(inventory).encode("utf-8")),
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
        / "mei4_posterior_calibration"
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
    payloads = {
        "mei4_execution_contract.json": contract,
        "b4_asset_inventory.json": inventory,
        "parent_b4_manifest.json": b4_manifest,
        "parent_b5_manifest.json": load_json(b5_manifest_path),
    }
    for name, payload in payloads.items():
        (staging / name).write_bytes(dumps_stable(payload).encode("utf-8"))
    summary = [
        "# tv3 MEI-4 C0 execution contract freeze",
        "",
        "- status: `mei4_contract_frozen`",
        "- deterministic baseline: `S1`",
        f"- parent B4 manifest SHA256: `{sha256_file(b4_manifest_path)}`",
        f"- parent B5 manifest SHA256: `{sha256_file(b5_manifest_path)}`",
        "- B4 and B5 freezes were verified read-only.",
        "- No observation-space sampling was performed.",
    ]
    (staging / "mei4_c0_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    source_paths = {
        "mei4_contract": contract_path,
        "c0_runner": Path(__file__).resolve(),
        "registry_helpers": _TV3_ROOT / "tv3" / "audit" / "mrs_ei_registry.py",
    }
    snapshots = staging / "source_snapshots"
    snapshots.mkdir()
    artifacts = [*payloads, "mei4_c0_summary.md"]
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
        "parent_manifest_path": _relative_to_root(b5_manifest_path),
        "parent_manifest_sha256": sha256_file(b5_manifest_path),
        "parent_b4_manifest_path": _relative_to_root(b4_manifest_path),
        "parent_b4_manifest_sha256": sha256_file(b4_manifest_path),
        "git_commit": _git_commit(),
        "git_relevant_paths_dirty": _git_relevant_paths_dirty(list(source_paths.values())),
        "artifact_sha256": {name: sha256_file(staging / name) for name in artifacts},
        "source_sha256": source_sha256,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
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
        raise RuntimeError(f"MEI-4 C0 manifest verification failed: {issues}")
    _promote_stage_status(
        stage_path,
        contract=contract,
        b4_manifest_sha256=sha256_file(b4_manifest_path),
        b5_manifest_sha256=sha256_file(b5_manifest_path),
        freeze_dir=freeze_relative,
        manifest_sha256=manifest_sha,
        created_at_utc=created_at.isoformat(),
    )
    print(
        json.dumps(
            {
                "freeze_dir": freeze_relative,
                "manifest_sha256": manifest_sha,
                "status": "mei4_contract_frozen",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
