from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
import subprocess
from typing import Any


class TQIFArtifactError(ValueError):
    """Raised when a TQIF artifact cannot satisfy its immutable contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalized_text_hash(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def binary_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TQIFArtifactError("INVALID_ARTIFACT", f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if hasattr(value, "item") and callable(value.item):
        return json_safe(value.item())
    if hasattr(value, "tolist") and callable(value.tolist):
        return json_safe(value.tolist())
    return value


def resolve_project_file(project_root: Path, configured: str | Path) -> Path:
    candidate = Path(configured)
    resolved = candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()
    if not resolved.is_relative_to(project_root):
        raise TQIFArtifactError(
            "PROTOCOL_ACCESS_VIOLATION",
            f"configured path escapes project root: {configured}",
        )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def relative_path(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


def git_snapshot(project_root: Path) -> dict[str, Any]:
    git_root = Path(_git(project_root, "rev-parse", "--show-toplevel").strip()).resolve()
    revision = _git(project_root, "rev-parse", "HEAD").strip()
    if not revision:
        raise TQIFArtifactError("FORMAL_NOT_FROZEN", "git revision is empty")
    status = _git(project_root, "status", "--porcelain", "--untracked-files=all")
    dirty = bool(status.strip())
    diff_hash: str | None = None
    if dirty:
        tracked_diff = _git(project_root, "diff", "HEAD", "--binary")
        untracked_paths = _git(
            project_root,
            "ls-files",
            "--others",
            "--exclude-standard",
        ).splitlines()
        digest = hashlib.sha256()
        digest.update(status.replace("\r\n", "\n").encode("utf-8"))
        digest.update(tracked_diff.replace("\r\n", "\n").encode("utf-8"))
        for relative in sorted(path for path in untracked_paths if path):
            candidate = (git_root / relative).resolve()
            if candidate.is_file() and candidate.is_relative_to(git_root):
                digest.update(relative.replace("\\", "/").encode("utf-8"))
                digest.update(b"\0")
                digest.update(candidate.read_bytes())
        diff_hash = digest.hexdigest()
    return {
        "git_commit": revision,
        "git_dirty": dirty,
        "source_diff_hash": diff_hash,
    }


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8", newline="\n")
    except FileExistsError as exc:
        raise TQIFArtifactError("RUN_IN_PROGRESS", f"lock already exists: {path}") from exc
    try:
        handle.write("locked\n")
        handle.close()
        yield
    finally:
        if path.exists():
            path.unlink()


def canonical_run_input_hash(
    *,
    protocol_hash: str,
    dataset_manifest_hash: str,
    split_manifest_hash: str,
    model_config_hash: str,
    train_config_hash: str,
    eval_config_hash: str,
    seed: int,
    model_id: str,
    recipe_id: str,
    source_snapshot: Mapping[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
            "protocol_hash": protocol_hash,
            "dataset_manifest_hash": dataset_manifest_hash,
            "split_manifest_hash": split_manifest_hash,
            "model_config_hash": model_config_hash,
            "train_config_hash": train_config_hash,
            "eval_config_hash": eval_config_hash,
            "seed": int(seed),
            "model_id": model_id,
            "recipe_id": recipe_id,
    }
    if source_snapshot is not None:
        payload["source_snapshot"] = source_snapshot
    return canonical_hash(payload)


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


__all__ = [
    "TQIFArtifactError",
    "binary_sha256",
    "canonical_hash",
    "canonical_json_bytes",
    "canonical_run_input_hash",
    "exclusive_lock",
    "git_snapshot",
    "normalized_text_hash",
    "read_json_object",
    "relative_path",
    "resolve_project_file",
    "write_json",
]
