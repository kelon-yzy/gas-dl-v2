"""RCDW benchmark 不变量校验。

对应方案 §7.1。检查项：
1. 无 LEGACY 字段（base_condition_id / noise_seed_index / noise_seed）
2. sequence_id 唯一
3. mixture_id 唯一
4. 组分和 = 100% ± 1e-5
5. split 覆盖完整（每个 sequence 恰好出现在一个 split 中）
6. slow 数组形状 (N, T, 7)
7. labels 形状 (N, 3)
8. ultrasonic / fiber_mic 序列轴 = N
9. ultrasonic 元数组形状 = (N, T)
10. scaler 输出对 ``passthrough`` 通道显式标记（方案 §6.5 v1.2）

与 HG 主线差异：``SPLIT_NAMES`` 不含 extrapolation（方案 §2.6）。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from rcdw.sim.core.schema import (
    COMPONENT_FIELDS as DEFAULT_COMPONENT_FIELDS,
    LEGACY_CONDITION_FIELDS,
    SLOW_CHANNELS as DEFAULT_SLOW_CHANNELS,
    SPLIT_NAMES,
)


def validate_benchmark_assets(
    conditions: list[dict[str, str]],
    split_rows: dict[str, list[dict[str, str]]],
    arrays: dict[str, Any] | None = None,
    labels: np.ndarray | None = None,
    *,
    component_fields: tuple[str, ...] = DEFAULT_COMPONENT_FIELDS,
    slow_channels: tuple[str, ...] = DEFAULT_SLOW_CHANNELS,
    background_fields: tuple[str, ...] = (),
    require_sum_100: bool = True,
    scaler: dict[str, Any] | None = None,
    expected_passthrough_channels: tuple[str, ...] = (),
) -> dict[str, object]:
    """逐项校验 benchmark assets，全部通过返回 summary。任一失败直接 raise。

    Args:
        conditions: ``generate_condition_rows`` 输出。
        split_rows: ``build_default_split_rows`` 输出。
        arrays: ``build_sequence_arrays`` 返回的字典；为 None 时跳过 shape 校验。
        labels: (N, 3) 标签数组；为 None 时跳过 shape 校验。
        component_fields: 组分字段顺序。
        slow_channels: SLOW_CHANNELS 顺序。
        background_fields: 背景气字段（RCDW 三组分闭包为空）。
        require_sum_100: 是否要求组分 + 背景之和 = 100%。
        scaler: 可选 scaler 字典；若提供则校验 passthrough 标记。
        expected_passthrough_channels: 期望被标记为 passthrough 的通道。

    Returns:
        summary dict（status="pass"、sequence_count、mixture_count、split_counts）。
    """
    _validate_no_legacy_fields(conditions)
    _validate_unique_ids(conditions, "sequence_id")
    _validate_unique_ids(conditions, "mixture_id")
    if require_sum_100:
        _validate_component_sums(
            conditions,
            component_fields=component_fields,
            background_fields=background_fields,
        )
    _validate_split_coverage(conditions, split_rows)
    if arrays is not None and labels is not None:
        _validate_array_shapes(
            conditions,
            arrays,
            labels,
            component_fields=component_fields,
            slow_channels=slow_channels,
        )
    if scaler is not None and expected_passthrough_channels:
        _validate_scaler_passthrough(scaler, expected_passthrough_channels)

    return {
        "status": "pass",
        "sequence_count": len(conditions),
        "mixture_count": len({row["mixture_id"] for row in conditions}),
        "split_counts": {
            name: len(split_rows.get(name, [])) for name in SPLIT_NAMES
        },
    }


# ---- 单项校验函数 ----


def _validate_no_legacy_fields(conditions: list[dict[str, str]]) -> None:
    for row in conditions:
        present = [field for field in LEGACY_CONDITION_FIELDS if field in row]
        if present:
            raise ValueError(
                f"legacy condition fields are not allowed: {present}; "
                f"see schema.py LEGACY_CONDITION_FIELDS for the black list."
            )


def _validate_unique_ids(conditions: list[dict[str, str]], field: str) -> None:
    counts = Counter(row[field] for row in conditions)
    duplicates = [value for value, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"{field} must be unique; duplicates: {duplicates[:5]}")


def _validate_component_sums(
    conditions: list[dict[str, str]],
    *,
    component_fields: tuple[str, ...],
    background_fields: tuple[str, ...],
) -> None:
    all_fields = (*component_fields, *background_fields)
    for row in conditions:
        total = sum(float(row[name]) for name in all_fields)
        if abs(total - 100.0) > 1e-5:
            raise ValueError(
                f"component+background total must equal 100, got {total} "
                f"for fields {all_fields} (sequence_id={row.get('sequence_id', '?')})"
            )


def _validate_split_coverage(
    conditions: list[dict[str, str]],
    split_rows: dict[str, list[dict[str, str]]],
) -> None:
    expected = {row["sequence_id"] for row in conditions}
    assigned = [
        row["sequence_id"]
        for name in SPLIT_NAMES
        for row in split_rows.get(name, [])
    ]
    if set(assigned) != expected or len(assigned) != len(expected):
        raise ValueError(
            "split rows must cover every sequence exactly once; "
            f"expected {len(expected)} unique, got {len(assigned)} assigned "
            f"({len(set(assigned))} unique)."
        )


def _validate_array_shapes(
    conditions: list[dict[str, str]],
    arrays: dict[str, Any],
    labels: np.ndarray,
    *,
    component_fields: tuple[str, ...],
    slow_channels: tuple[str, ...],
) -> None:
    sequence_count = len(conditions)
    slow = arrays["slow"]
    if slow.shape[0] != sequence_count:
        raise ValueError(
            f"slow array sequence axis must match condition rows: "
            f"got {slow.shape[0]}, expected {sequence_count}"
        )
    if slow.shape[2] != len(slow_channels):
        raise ValueError(
            f"slow channel axis must match slow channel schema: "
            f"got {slow.shape[2]}, expected {len(slow_channels)}"
        )
    if labels.shape != (sequence_count, len(component_fields)):
        raise ValueError(
            f"label array shape must match condition rows and labels: "
            f"got {labels.shape}, expected ({sequence_count}, {len(component_fields)})"
        )
    for name in ("ultrasonic", "fiber_mic", "ultrasonic_scale", "fiber_mic_scale"):
        if arrays[name].shape[0] != sequence_count:
            raise ValueError(
                f"{name} sequence axis must match condition rows: "
                f"got {arrays[name].shape[0]}, expected {sequence_count}"
            )
    for name in (
        "ultrasonic_tof_s",
        "ultrasonic_tof_observed_s",
        "ultrasonic_peak_index",
        "ultrasonic_sound_speed_m_per_s",
        "ultrasonic_sound_speed_estimated_m_per_s",
        "ultrasonic_alpha_true_npm",
        "ultrasonic_tof_quality",
        "ultrasonic_tof_accepted",
    ):
        if arrays[name].shape != slow.shape[:2]:
            raise ValueError(
                f"{name} shape must match slow sequence and timestep axes: "
                f"got {arrays[name].shape}, expected {slow.shape[:2]}"
            )


def _validate_scaler_passthrough(
    scaler: dict[str, Any],
    expected_passthrough_channels: tuple[str, ...],
) -> None:
    """v1.2 §6.5 不变量：跳过归一化的通道必须在 scaler 输出中显式标记。"""
    entries = scaler.get("channel_entries", [])
    if not entries:
        return  # 老版 scaler 无 channel_entries，向后兼容
    passthrough_in_scaler = {
        entry["channel"]
        for entry in entries
        if isinstance(entry, dict) and entry.get("strategy") == "passthrough"
    }
    for channel in expected_passthrough_channels:
        # 仅在通道确实存在于 scaler 输出中时才要求 passthrough 标记
        present = any(
            isinstance(entry, dict) and entry.get("channel") == channel
            for entry in entries
        )
        if present and channel not in passthrough_in_scaler:
            raise ValueError(
                f"channel {channel!r} must be marked strategy='passthrough' "
                f"in scaler output (v1.2 §6.5)"
            )
