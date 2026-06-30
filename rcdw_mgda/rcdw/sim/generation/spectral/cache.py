"""Spectral cache 读写：HITRAN 系数 .npz 文件 + metadata roundtrip 校验。"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class SpectralCacheKey:
    backend: str
    gas: str
    source_version: str
    wavenumber_min_cm1: float
    wavenumber_max_cm1: float
    wavenumber_step_cm1: float
    temperature_k: float
    pressure_atm: float


def cache_path(cache_root: Path | str, key: SpectralCacheKey) -> Path:
    return Path(cache_root) / (
        f"{_safe_token(key.backend)}__{_safe_token(key.gas)}__"
        f"{_safe_token(key.source_version)}__"
        f"{key.wavenumber_min_cm1:.4f}_{key.wavenumber_max_cm1:.4f}_{key.wavenumber_step_cm1:.4f}__"
        f"T{key.temperature_k:.3f}__P{key.pressure_atm:.6f}.npz"
    )


def read_cached_spectrum(
    cache_root: Path | str, key: SpectralCacheKey
) -> tuple[np.ndarray, np.ndarray] | None:
    path = cache_path(cache_root, key)
    if not path.is_file():
        return None
    with np.load(path) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        if metadata != asdict(key):
            raise ValueError("cached spectrum metadata does not match requested key")
        return (
            payload["wavenumber_cm1"].astype(np.float64),
            payload["absorption_coeff_cm1"].astype(np.float64),
        )


def write_cached_spectrum(
    cache_root: Path | str,
    key: SpectralCacheKey,
    *,
    wavenumber_cm1: np.ndarray,
    absorption_coeff_cm1: np.ndarray,
) -> Path:
    if wavenumber_cm1.shape != absorption_coeff_cm1.shape:
        raise ValueError(
            "wavenumber_cm1 and absorption_coeff_cm1 must have the same shape"
        )
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    path = cache_path(root, key)
    tmp_path = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                wavenumber_cm1=wavenumber_cm1.astype(np.float64),
                absorption_coeff_cm1=absorption_coeff_cm1.astype(np.float64),
                metadata=json.dumps(asdict(key), ensure_ascii=True, sort_keys=True),
            )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def _safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
