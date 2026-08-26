"""Atomic file I/O and hashing for GIB artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4


class AtomicWriteError(FileExistsError):
    """Raised when append-only artifact promotion would overwrite a file."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    target = Path(path)
    if target.exists():
        raise AtomicWriteError(f"artifact already exists and cannot be overwritten: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    try:
        with staging.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise AtomicWriteError(f"artifact target appeared during promotion: {target}")
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            staging.unlink()
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        for row in rows
    )
    atomic_write_bytes(path, payload)


def atomic_promote_directory(staging: Path, target: Path) -> None:
    source = Path(staging)
    destination = Path(target)
    if not source.is_dir():
        raise FileNotFoundError(f"staging directory does not exist: {source}")
    if destination.exists():
        raise AtomicWriteError(f"artifact directory already exists and cannot be overwritten: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def remove_owned_staging(path: Path) -> None:
    staging = Path(path)
    if staging.exists():
        if ".staging-" not in staging.name:
            raise ValueError(f"refusing to remove a non-staging path: {staging}")
        shutil.rmtree(staging)


__all__ = [
    "AtomicWriteError",
    "atomic_write_bytes",
    "atomic_promote_directory",
    "atomic_write_json",
    "atomic_write_jsonl",
    "canonical_json_bytes",
    "sha256_bytes",
    "sha256_file",
    "remove_owned_staging",
]
