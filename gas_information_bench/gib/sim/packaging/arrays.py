"""Stable per-sequence NumPy artifact storage."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

from ...common.io import atomic_write_bytes, sha256_file


def array_npy_bytes(array: np.ndarray) -> bytes:
    value = np.asarray(array)
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def write_array_artifact(path: Path, array: np.ndarray) -> dict[str, object]:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object arrays are not allowed")
    if value.dtype.names is None and not np.all(np.isfinite(value)):
        raise ValueError("numeric array contains non-finite values")
    atomic_write_bytes(path, array_npy_bytes(value))
    return {
        "sha256": sha256_file(path),
        "dtype": value.dtype.str if value.dtype.names is None else str(value.dtype.descr),
        "shape": list(value.shape),
    }


def read_array_artifact(path: Path) -> np.ndarray:
    with Path(path).open("rb") as handle:
        value = np.load(handle, allow_pickle=False)
    return np.asarray(value)


__all__ = ["array_npy_bytes", "read_array_artifact", "write_array_artifact"]
