from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sg.sim.packaging.constants import Z_SCORE_STD_EPSILON


def load_scaler(scaler_path: Path | str) -> dict[str, object]:
    """Load a v4 z-score scaler from a JSON file.

    The expected JSON format matches the output of
    ``sim.packaging.scalers.fit_z_score_scalers``.

    Returns:
        A dict with keys ``"method"``, ``"channel_names"``, ``"mean"``, ``"std"``,
        and optionally ``"modal_groups"`` / ``"modal_stats"`` for modal scalers.
    """
    path = Path(scaler_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_scaler_payload(payload)
    return payload


def apply_scaler(x: np.ndarray, scaler: dict[str, object]) -> np.ndarray:
    """Apply z-score normalization using a loaded v4 scaler dict."""
    if x.ndim not in {2, 3}:
        raise ValueError(f"apply_scaler expects a 2D or 3D array, got ndim={x.ndim}")
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    std = np.where(std > Z_SCORE_STD_EPSILON, std, 1.0)
    if x.shape[-1] != mean.shape[0]:
        raise ValueError(f"last dimension must match scaler channels: {x.shape[-1]} != {mean.shape[0]}")
    if x.ndim == 3:
        mean = mean.reshape(1, 1, -1)
        std = std.reshape(1, 1, -1)
    else:
        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
    return (x - mean) / std


def _validate_scaler_payload(payload: dict[str, object]) -> None:
    required = {"method", "channel_names", "mean", "std"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Scaler payload missing keys: {sorted(missing)}")
    if payload["method"] != "z_score":
        raise ValueError(f"Unsupported scaler method: {payload['method']}")
