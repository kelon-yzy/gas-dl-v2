from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
import platform
import sys
from typing import Any, Callable

import numpy as np
import scipy
import sklearn
import torch


def runtime_fingerprint(*, device: str = "cpu", dtype: str = "float32") -> dict[str, Any]:
    """Return the numerical runtime facts that can affect a benchmark run."""

    if not device:
        raise ValueError("device must be non-empty")
    if not dtype:
        raise ValueError("dtype must be non-empty")
    return {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pytorch": torch.__version__,
        },
        "blas": {
            "numpy_config": _capture_config(np.__config__.show),
            "pytorch_config": torch.__config__.show(),
        },
        "device": device,
        "dtype": dtype,
        "determinism": {
            "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _capture_config(function: Callable[[], Any]) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        function()
    return buffer.getvalue().strip()


__all__ = ["canonical_sha256", "runtime_fingerprint", "sha256_file"]
