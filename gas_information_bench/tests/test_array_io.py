from pathlib import Path

import numpy as np
import pytest

from gib.common import io
from gib.common.io import AtomicWriteError, atomic_write_bytes, sha256_bytes, sha256_file
from gib.sim.packaging.arrays import read_array_artifact, write_array_artifact


def test_array_artifact_round_trip_is_byte_stable(tmp_path: Path):
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    first_record = write_array_artifact(first, array)
    second_record = write_array_artifact(second, array)
    assert first.read_bytes() == second.read_bytes()
    assert first_record["sha256"] == second_record["sha256"] == sha256_file(first)
    assert np.array_equal(read_array_artifact(first), array)


def test_atomic_write_never_overwrites_an_existing_target(tmp_path: Path):
    target = tmp_path / "artifact.bin"
    atomic_write_bytes(target, b"original")
    with pytest.raises(AtomicWriteError):
        atomic_write_bytes(target, b"replacement")
    assert target.read_bytes() == b"original"


def test_atomic_write_exposes_replace_failure_and_cleans_its_staging(tmp_path: Path, monkeypatch):
    target = tmp_path / "artifact.bin"

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        atomic_write_bytes(target, b"payload")
    assert not target.exists()
    assert not list(tmp_path.glob(".*.staging-*"))
    assert sha256_bytes(b"payload") == "239F59ED55E737C77147CF55AD0C1B030B6D7EE748A7426952F9B852D5A935E5"
