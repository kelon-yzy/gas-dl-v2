"""Train-only Z-score scaler + 异质通道 passthrough 策略。

对应方案 §6.5 (v1.2 异质通道归一化策略表)。

策略：
- 连续物理量通道（V_NDIR_CO2, V_TCS, T_C, P_MPa, H_RH, L_m, piston_position_m）
  走 train-only Z-score 拟合 (mean, std)，应用阶段 x' = (x - mean) / std。
- 离散/有界量通道（``ultrasonic_peak_index``, ``ultrasonic_tof_quality``,
  ``ultrasonic_tof_accepted``）走 passthrough，保留原始物理语义。
  这些通道由调用方通过 ``skip_channels`` 显式声明；scaler 输出
  字典对这些通道记录 ``{"strategy": "passthrough"}`` 占位。

注意 RCDW slow 矩阵当前仅 7 个通道（SLOW_CHANNELS），跳过通道实际仅在
ultrasonic 元数据进入 12 维拼接张量时才生效（Phase 4 Dataset 层）。
本 scaler 仅对 slow 矩阵拟合，跳过通道列表保留作为接口契约和未来扩展点。
"""

from __future__ import annotations

import numpy as np

from rcdw.sim.packaging.constants import Z_SCORE_STD_EPSILON


DEFAULT_PASSTHROUGH_CHANNELS: tuple[str, ...] = (
    "ultrasonic_peak_index",
    "ultrasonic_tof_quality",
    "ultrasonic_tof_accepted",
)


def fit_z_score_scalers(
    matrix: np.ndarray,
    train_indexes: list[int],
    channel_names: tuple[str, ...],
    modal_groups: dict[str, tuple[str, ...]],
    transform_target: str = "slow",
    skip_channels: tuple[str, ...] = DEFAULT_PASSTHROUGH_CHANNELS,
) -> tuple[dict[str, object], dict[str, object]]:
    """逐通道 train-only Z-score 拟合，跳过通道走 passthrough。

    Args:
        matrix: (N_seq, T, C)，按 channel_names 顺序的最后一维。
        train_indexes: train split 的序列索引列表。
        channel_names: 通道名顺序（必须与 matrix 最后一维长度一致）。
        modal_groups: 模态分组（``optical / thermal / environment`` 等）。
        transform_target: 元数据用，标识 scaler 目标（``"slow"``）。
        skip_channels: 跳过 Z-score 的通道名集合（默认三个 ultrasonic 元数据）。

    Returns:
        (sequence_scaler, modal_scaler)：两个 JSON 友好字典。
    """
    if not train_indexes:
        raise ValueError("Cannot fit scalers without train sequences.")
    if matrix.ndim != 3:
        raise ValueError(
            f"matrix must be a 3D array shaped (sequence, timestep, channel), "
            f"got ndim={matrix.ndim}"
        )
    if matrix.shape[-1] != len(channel_names):
        raise ValueError(
            f"matrix channel count must match channel_names: "
            f"{matrix.shape[-1]} != {len(channel_names)}"
        )

    train_x = matrix[train_indexes]
    mean = train_x.mean(axis=(0, 1))
    std = train_x.std(axis=(0, 1))
    std = np.where(std > Z_SCORE_STD_EPSILON, std, 1.0)

    skip_set = set(skip_channels)

    # sequence_scaler: 逐通道 mean/std；跳过通道用 strategy=passthrough。
    channel_entries: list[dict[str, object]] = []
    mean_list: list[float | None] = []
    std_list: list[float | None] = []
    for index, channel in enumerate(channel_names):
        if channel in skip_set:
            channel_entries.append({"channel": channel, "strategy": "passthrough"})
            mean_list.append(None)
            std_list.append(None)
        else:
            channel_entries.append(
                {
                    "channel": channel,
                    "strategy": "z_score",
                    "mean": float(mean[index]),
                    "std": float(std[index]),
                }
            )
            mean_list.append(float(mean[index]))
            std_list.append(float(std[index]))

    sequence_scaler: dict[str, object] = {
        "method": "z_score",
        "fit_scope": "train_split_only",
        "transform_target": transform_target,
        "channel_axis": -1,
        "channel_names": list(channel_names),
        "channel_entries": channel_entries,
        "mean": mean_list,
        "std": std_list,
        "skip_channels": list(skip_channels),
    }

    # modal_scaler: 按 modal_groups 分组聚合统计量（跳过通道仍登记 passthrough）。
    channel_index = {channel: index for index, channel in enumerate(channel_names)}
    modal_stats: dict[str, dict[str, object]] = {}
    for modal_name, channels in modal_groups.items():
        modal_means: list[float | None] = []
        modal_stds: list[float | None] = []
        modal_strategies: list[str] = []
        for channel in channels:
            if channel not in channel_index:
                raise ValueError(
                    f"modal group {modal_name!r} channel {channel!r} not in channel_names"
                )
            idx = channel_index[channel]
            if channel in skip_set:
                modal_means.append(None)
                modal_stds.append(None)
                modal_strategies.append("passthrough")
            else:
                modal_means.append(float(mean[idx]))
                modal_stds.append(float(std[idx]))
                modal_strategies.append("z_score")
        modal_stats[modal_name] = {
            "channel_names": list(channels),
            "mean": modal_means,
            "std": modal_stds,
            "strategies": modal_strategies,
        }

    modal_scaler: dict[str, object] = {
        "method": "z_score",
        "fit_scope": "train_split_only",
        "transform_target": transform_target,
        "channel_axis": -1,
        "modal_groups": {name: list(channels) for name, channels in modal_groups.items()},
        "modal_stats": modal_stats,
        "skip_channels": list(skip_channels),
    }
    return sequence_scaler, modal_scaler
