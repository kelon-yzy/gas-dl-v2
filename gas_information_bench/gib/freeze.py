"""Append-only evidence freeze contract owned by :mod:`gib`."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4


class FreezeContractError(ValueError):
    """Raised when an attempt cannot be promoted to an immutable freeze."""


REQUIRED_INPUT_ROLES = frozenset(
    {
        "config",
        "schema",
        "gate",
        "code",
        "source_registry",
    }
)
_FREEZE_ID_PATTERN = re.compile(r"^GIB-FREEZE-[A-Z0-9][A-Z0-9._-]{2,63}$")
_ATTEMPT_MANIFEST = "attempt_manifest.json"
_EVIDENCE_MANIFEST = "evidence_manifest.json"
_EVIDENCE_MANIFEST_FIELDS = {
    "schema_version",
    "freeze_id",
    "hash_algorithm",
    "attempt_status",
    "required_input_roles",
    "inputs",
    "source_snapshots",
    "evidence_files",
}
_INPUT_RECORD_FIELDS = {"role", "logical_path", "sha256", "snapshot_path"}
_SNAPSHOT_RECORD_FIELDS = {"logical_path", "sha256", "snapshot_path"}
_EVIDENCE_RECORD_FIELDS = {"path", "sha256"}
_SHA256_PATTERN = re.compile(r"^[A-F0-9]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeContractError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise FreezeContractError(f"{label} must be a JSON object: {path}")
    return value


def _workspace_file(path: Path, workspace_root: Path) -> tuple[Path, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FreezeContractError(f"freeze input is not a file: {path}")
    try:
        logical_path = resolved.relative_to(workspace_root).as_posix()
    except ValueError as exc:
        raise FreezeContractError(f"freeze input is outside workspace_root: {path}") from exc
    return resolved, logical_path


def _validate_separate_roots(attempt_dir: Path, freeze_root: Path) -> None:
    attempt = attempt_dir.resolve()
    freezes = freeze_root.resolve()
    if attempt == freezes or attempt.is_relative_to(freezes) or freezes.is_relative_to(attempt):
        raise FreezeContractError("attempts and freezes must use physically separate directory trees")


def _attempt_files(attempt_dir: Path) -> list[Path]:
    if not attempt_dir.is_dir():
        raise FreezeContractError(f"attempt directory does not exist: {attempt_dir}")
    if any(path.is_symlink() for path in attempt_dir.rglob("*")):
        raise FreezeContractError("attempt directory must not contain symbolic links")
    files = sorted(path for path in attempt_dir.rglob("*") if path.is_file())
    if not files:
        raise FreezeContractError("attempt directory contains no evidence files")
    manifest_path = attempt_dir / _ATTEMPT_MANIFEST
    if manifest_path not in files:
        raise FreezeContractError(f"attempt is missing {_ATTEMPT_MANIFEST}")
    manifest = _read_json_object(manifest_path, "attempt manifest")
    if manifest.get("status") != "complete":
        raise FreezeContractError("only an attempt with status=complete can be frozen")
    return files


def _normalize_inputs(
    input_files: Mapping[str, Sequence[Path]],
    workspace_root: Path,
) -> list[dict[str, str]]:
    missing_roles = sorted(REQUIRED_INPUT_ROLES - set(input_files))
    if missing_roles:
        raise FreezeContractError(f"freeze inputs are missing required roles: {missing_roles}")
    records: list[dict[str, str]] = []
    for role in sorted(input_files):
        paths = input_files[role]
        if not paths:
            raise FreezeContractError(f"freeze input role has no files: {role}")
        for path in paths:
            resolved, logical_path = _workspace_file(Path(path), workspace_root)
            records.append(
                {
                    "role": role,
                    "logical_path": logical_path,
                    "sha256": _sha256_file(resolved),
                }
            )
    logical_paths = [record["logical_path"] for record in records]
    if len(logical_paths) != len(set(logical_paths)):
        raise FreezeContractError("each freeze input file must have exactly one owner role")
    return records


def _snapshot_records(paths: Sequence[Path], workspace_root: Path) -> list[dict[str, str]]:
    if not paths:
        raise FreezeContractError("at least one source snapshot is required")
    records = []
    for path in paths:
        resolved, logical_path = _workspace_file(Path(path), workspace_root)
        records.append({"logical_path": logical_path, "sha256": _sha256_file(resolved)})
    logical_paths = [record["logical_path"] for record in records]
    if len(logical_paths) != len(set(logical_paths)):
        raise FreezeContractError("source snapshot paths must be unique")
    return sorted(records, key=lambda record: record["logical_path"])


def _copy_snapshot(logical_path: str, workspace_root: Path, snapshot_root: Path) -> str:
    logical = Path(logical_path)
    if logical.name == _EVIDENCE_MANIFEST and "outputs/archive/freezes" in logical.as_posix():
        short_hash = hashlib.sha256(logical_path.encode("utf-8")).hexdigest()[:12]
        destination = snapshot_root / "upstream_evidence" / f"{logical.parent.name}-{short_hash}.json"
    else:
        destination = snapshot_root / logical
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workspace_root / Path(logical_path), destination)
    return destination.relative_to(snapshot_root.parent).as_posix()


def _freeze_member(root: Path, value: object, label: str) -> tuple[Path, str]:
    relative = Path(str(value))
    if not str(value) or relative.is_absolute() or ".." in relative.parts:
        raise FreezeContractError(f"{label} path must stay inside the freeze: {value}")
    path = root / relative
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise FreezeContractError(f"{label} path escapes the freeze: {value}") from exc
    if not path.is_file():
        raise FreezeContractError(f"{label} file is missing: {path}")
    return path, relative.as_posix()


def _require_exact_fields(record: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(record)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise FreezeContractError(f"{label} fields mismatch: missing={missing}, extra={extra}")


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise FreezeContractError(f"{label} must be an uppercase SHA256 hex string")
    return value


def _require_relative_logical_path(value: object, label: str) -> str:
    relative = Path(str(value))
    if not isinstance(value, str) or not value or relative.is_absolute() or ".." in relative.parts:
        raise FreezeContractError(f"{label} must be a relative workspace path")
    return relative.as_posix()


def freeze_attempt(
    *,
    workspace_root: Path,
    attempt_dir: Path,
    freeze_root: Path,
    freeze_id: str,
    input_files: Mapping[str, Sequence[Path]],
    source_snapshots: Sequence[Path],
) -> Path:
    """Promote one complete attempt into a new immutable evidence directory."""

    workspace = workspace_root.resolve()
    attempt = attempt_dir.resolve()
    freezes = freeze_root.resolve()
    if not workspace.is_dir():
        raise FreezeContractError(f"workspace_root does not exist: {workspace_root}")
    if _FREEZE_ID_PATTERN.fullmatch(freeze_id) is None:
        raise FreezeContractError(f"invalid freeze_id: {freeze_id}")
    _validate_separate_roots(attempt, freezes)
    attempt_files = _attempt_files(attempt)
    inputs = _normalize_inputs(input_files, workspace)
    snapshots = _snapshot_records(source_snapshots, workspace)

    freezes.mkdir(parents=True, exist_ok=True)
    target = freezes / freeze_id
    if target.exists():
        raise FreezeContractError(f"freeze already exists and cannot be overwritten: {target}")
    staging = freezes / f".{freeze_id}.staging-{uuid4().hex}"
    staging.mkdir()
    try:
        attempt_snapshot = staging / "attempt"
        shutil.copytree(attempt, attempt_snapshot)
        evidence_files = [
            {
                "path": (Path("attempt") / path.relative_to(attempt)).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in attempt_files
        ]

        snapshot_root = staging / "source_snapshots"
        copied_paths: dict[str, str] = {}
        for record in [*inputs, *snapshots]:
            logical_path = record["logical_path"]
            if logical_path not in copied_paths:
                copied_paths[logical_path] = _copy_snapshot(logical_path, workspace, snapshot_root)
        for record in inputs:
            record["snapshot_path"] = copied_paths[record["logical_path"]]
        for record in snapshots:
            record["snapshot_path"] = copied_paths[record["logical_path"]]

        manifest = {
            "schema_version": "gib-benchmark-1",
            "freeze_id": freeze_id,
            "hash_algorithm": "SHA256",
            "attempt_status": "complete",
            "required_input_roles": sorted(REQUIRED_INPUT_ROLES),
            "inputs": inputs,
            "source_snapshots": snapshots,
            "evidence_files": evidence_files,
        }
        (staging / _EVIDENCE_MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            raise FreezeContractError(f"freeze target appeared during promotion: {target}")
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def verify_evidence_manifest(freeze_dir: Path) -> dict[str, int | str]:
    """Recompute every hash recorded by a GIB evidence manifest."""

    root = freeze_dir.resolve()
    manifest = _read_json_object(root / _EVIDENCE_MANIFEST, "evidence manifest")
    _require_exact_fields(manifest, _EVIDENCE_MANIFEST_FIELDS, "evidence manifest")
    if manifest.get("schema_version") != "gib-benchmark-1":
        raise FreezeContractError("unexpected freeze schema_version")
    freeze_id = manifest.get("freeze_id")
    if not isinstance(freeze_id, str) or _FREEZE_ID_PATTERN.fullmatch(freeze_id) is None:
        raise FreezeContractError("invalid freeze_id in evidence manifest")
    if freeze_id != root.name:
        raise FreezeContractError("freeze directory name does not match freeze_id")
    if manifest.get("hash_algorithm") != "SHA256":
        raise FreezeContractError("freeze hash_algorithm must be SHA256")
    if manifest.get("attempt_status") != "complete":
        raise FreezeContractError("freeze attempt_status must be complete")
    if set(manifest.get("required_input_roles", ())) != REQUIRED_INPUT_ROLES:
        raise FreezeContractError("freeze required_input_roles do not match the contract")
    inputs = manifest.get("inputs")
    snapshots = manifest.get("source_snapshots")
    evidence = manifest.get("evidence_files")
    if not isinstance(inputs, list) or not inputs:
        raise FreezeContractError("evidence manifest inputs must be a non-empty list")
    if not isinstance(snapshots, list) or not snapshots:
        raise FreezeContractError("evidence manifest source_snapshots must be a non-empty list")
    if not isinstance(evidence, list) or not evidence:
        raise FreezeContractError("evidence manifest file lists are invalid")
    if not all(isinstance(item, dict) for item in inputs):
        raise FreezeContractError("input record must be an object")
    actual_roles = {item.get("role") for item in inputs}
    if not REQUIRED_INPUT_ROLES.issubset(actual_roles):
        missing = sorted(REQUIRED_INPUT_ROLES - actual_roles)
        raise FreezeContractError(f"evidence manifest input roles mismatch: missing={missing}")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise FreezeContractError("freeze directory must not contain symbolic links")
    registered_paths = {_EVIDENCE_MANIFEST}
    for label, records in (("input", inputs), ("source snapshot", snapshots)):
        seen_logical_paths: set[str] = set()
        seen_snapshot_paths: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise FreezeContractError(f"{label} record must be an object")
            expected_fields = _INPUT_RECORD_FIELDS if label == "input" else _SNAPSHOT_RECORD_FIELDS
            _require_exact_fields(record, expected_fields, f"{label} record")
            logical_path = _require_relative_logical_path(record.get("logical_path"), f"{label} logical_path")
            if logical_path in seen_logical_paths:
                raise FreezeContractError(f"duplicate {label} logical_path: {logical_path}")
            seen_logical_paths.add(logical_path)
            _require_sha256(record.get("sha256"), f"{label} sha256")
            path, relative = _freeze_member(root, record.get("snapshot_path", ""), label)
            if relative in seen_snapshot_paths:
                raise FreezeContractError(f"duplicate {label} snapshot_path: {relative}")
            seen_snapshot_paths.add(relative)
            registered_paths.add(relative)
            if _sha256_file(path) != record.get("sha256"):
                raise FreezeContractError(f"{label} hash mismatch: {path}")
    seen_evidence_paths: set[str] = set()
    for record in evidence:
        if not isinstance(record, dict):
            raise FreezeContractError("evidence file record must be an object")
        _require_exact_fields(record, _EVIDENCE_RECORD_FIELDS, "evidence file record")
        _require_sha256(record.get("sha256"), "evidence file sha256")
        path, relative = _freeze_member(root, record.get("path", ""), "evidence file")
        if relative in seen_evidence_paths:
            raise FreezeContractError(f"duplicate evidence file path: {relative}")
        seen_evidence_paths.add(relative)
        registered_paths.add(relative)
        if _sha256_file(path) != record.get("sha256"):
            raise FreezeContractError(f"evidence file hash mismatch: {path}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_paths != registered_paths:
        added = sorted(actual_paths - registered_paths)
        missing = sorted(registered_paths - actual_paths)
        raise FreezeContractError(
            f"freeze file set does not match the manifest: added={added}, missing={missing}"
        )
    return {
        "freeze_id": str(manifest["freeze_id"]),
        "input_count": len(inputs),
        "source_snapshot_count": len(snapshots),
        "evidence_file_count": len(evidence),
    }


__all__ = [
    "FreezeContractError",
    "REQUIRED_INPUT_ROLES",
    "freeze_attempt",
    "verify_evidence_manifest",
]
