from __future__ import annotations

import numpy as np

from sim.packaging.constants import Z_SCORE_STD_EPSILON


def fit_z_score_scalers(
    matrix: np.ndarray,
    train_indexes: list[int],
    channel_names: tuple[str, ...],
    modal_groups: dict[str, tuple[str, ...]],
    transform_target: str = "slow",
) -> tuple[dict[str, object], dict[str, object]]:
    if not train_indexes:
        raise ValueError("Cannot fit scalers without train sequences.")
    if matrix.ndim != 3:
        raise ValueError(f"matrix must be a 3D array shaped (sequence, timestep, channel), got ndim={matrix.ndim}")
    if matrix.shape[-1] != len(channel_names):
        raise ValueError(f"matrix channel count must match channel_names: {matrix.shape[-1]} != {len(channel_names)}")
    train_x = matrix[train_indexes]
    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1))
    std = np.where(std > Z_SCORE_STD_EPSILON, std, 1.0)
    sequence_scaler = {
        "method": "z_score",
        "fit_scope": "train_split_only",
        "transform_target": transform_target,
        "channel_axis": -1,
        "channel_names": list(channel_names),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }
    modal_scaler = {
        "method": "z_score",
        "fit_scope": "train_split_only",
        "transform_target": transform_target,
        "channel_axis": -1,
        "modal_groups": {name: list(channels) for name, channels in modal_groups.items()},
        "modal_stats": {},
    }
    channel_index = {channel: index for index, channel in enumerate(channel_names)}
    for modal_name, channels in modal_groups.items():
        modal_scaler["modal_stats"][modal_name] = {
            "channel_names": list(channels),
            "mean": [float(mean[channel_index[channel]]) for channel in channels],
            "std": [float(std[channel_index[channel]]) for channel in channels],
        }
    return sequence_scaler, modal_scaler
