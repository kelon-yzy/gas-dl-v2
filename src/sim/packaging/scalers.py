from __future__ import annotations

import numpy as np


def fit_z_score_scalers(matrix, train_indexes, channel_names, modal_groups, transform_target="slow", channel_axis=2):
    if not train_indexes:
        raise ValueError("Cannot fit scalers without train sequences.")
    train_x = matrix[train_indexes]
    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1))
    std = np.where(std > 1e-15, std, 1.0)
    sequence_scaler = {
        "method": "z_score",
        "fit_scope": "train_split_only",
        "transform_target": transform_target,
        "channel_axis": channel_axis,
        "channel_names": list(channel_names),
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
    }
    modal_scaler = {
        "method": "z_score",
        "fit_scope": "train_split_only",
        "transform_target": transform_target,
        "channel_axis": channel_axis,
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
