"""Packaging publish: relocate on-disk memmap instead of 2x copy."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tv3.sim.packaging.arrays import _write_npy


def test_write_npy_relocate_same_filesystem(tmp_path: Path) -> None:
    src = tmp_path / "merged_ultrasonic_ab.npy"
    dst = tmp_path / "sequences" / "ultrasonic_ab_int16.npy"
    dst.parent.mkdir(parents=True, exist_ok=True)

    payload = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    mm = np.lib.format.open_memmap(src, mode="w+", dtype=payload.dtype, shape=payload.shape)
    mm[:] = payload
    mm.flush()

    _write_npy(dst, mm, use_memmap=True, relocate=True)

    assert not src.exists()
    assert dst.is_file()
    loaded = np.load(dst)
    np.testing.assert_array_equal(loaded, payload)


def test_write_npy_copy_when_relocate_disabled(tmp_path: Path) -> None:
    src = tmp_path / "merged.npy"
    dst = tmp_path / "out.npy"
    payload = np.arange(12, dtype=np.float32).reshape(3, 4)
    mm = np.lib.format.open_memmap(src, mode="w+", dtype=payload.dtype, shape=payload.shape)
    mm[:] = payload
    mm.flush()

    _write_npy(dst, mm, use_memmap=True, relocate=False)

    assert src.exists()
    assert dst.is_file()
    np.testing.assert_array_equal(np.load(dst), payload)
