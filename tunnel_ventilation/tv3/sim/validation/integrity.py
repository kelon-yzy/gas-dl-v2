from __future__ import annotations

from collections import Counter

import numpy as np

from tv3.sim.core.schema import (
    COMPONENT_FIELDS as DEFAULT_COMPONENT_FIELDS,
    LEGACY_CONDITION_FIELDS,
    SLOW_CHANNELS as DEFAULT_SLOW_CHANNELS,
    SPLIT_NAMES,
)


def validate_benchmark_assets(
    conditions: list[dict[str, str]],
    split_rows: dict[str, list[dict[str, str]]],
    arrays: dict[str, object] | None = None,
    labels: np.ndarray | None = None,
    *,
    component_fields: tuple[str, ...] = DEFAULT_COMPONENT_FIELDS,
    slow_channels: tuple[str, ...] = DEFAULT_SLOW_CHANNELS,
    background_fields: tuple[str, ...] = (),
    require_sum_100: bool = True,
) -> dict[str, object]:
    """Validate benchmark assets.

    component_fields / slow_channels / background_fields:
        Override defaults to support alternative composition schemes
        (e.g. syngas where COMPONENT_FIELDS = (x_H2, x_CH4, x_CO2, x_CO) and
        BACKGROUND_FIELDS = (x_N2,)).
    require_sum_100:
        When True (default), COMPONENT_FIELDS + BACKGROUND_FIELDS must sum
        to 100%. For hydrogen_ng, BACKGROUND_FIELDS is empty so this checks
        COMPONENT_FIELDS = 100% (closed composition). For syngas, the total
        sums to 100% only when x_N2 is included (background).
    """
    _validate_no_legacy_fields(conditions)
    _validate_unique_ids(conditions, "sequence_id")
    _validate_unique_ids(conditions, "mixture_id")
    if require_sum_100:
        _validate_component_sums(conditions, component_fields=component_fields, background_fields=background_fields)
    _validate_split_coverage(conditions, split_rows)
    if arrays is not None and labels is not None:
        _validate_array_shapes(
            conditions,
            arrays,
            labels,
            component_fields=component_fields,
            slow_channels=slow_channels,
        )
    return {
        "status": "pass",
        "sequence_count": len(conditions),
        "mixture_count": len({row["mixture_id"] for row in conditions}),
        "split_counts": {name: len(split_rows.get(name, [])) for name in SPLIT_NAMES},
    }


def _validate_no_legacy_fields(conditions: list[dict[str, str]]) -> None:
    for row in conditions:
        present = [field for field in LEGACY_CONDITION_FIELDS if field in row]
        if present:
            raise ValueError(f"legacy condition fields are not allowed: {present}")


def _validate_unique_ids(conditions: list[dict[str, str]], field: str) -> None:
    counts = Counter(row[field] for row in conditions)
    duplicates = [value for value, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"{field} must be unique")


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
            raise ValueError(f"component+background total must equal 100, got {total} for fields {all_fields}")


def _validate_split_coverage(
    conditions: list[dict[str, str]],
    split_rows: dict[str, list[dict[str, str]]],
) -> None:
    expected = {row["sequence_id"] for row in conditions}
    assigned = [row["sequence_id"] for name in SPLIT_NAMES for row in split_rows.get(name, [])]
    if set(assigned) != expected or len(assigned) != len(expected):
        raise ValueError("split rows must cover every sequence exactly once")


def _validate_array_shapes(
    conditions: list[dict[str, str]],
    arrays: dict[str, object],
    labels: np.ndarray,
    *,
    component_fields: tuple[str, ...],
    slow_channels: tuple[str, ...],
) -> None:
    sequence_count = len(conditions)
    slow = arrays["slow"]
    if slow.shape[0] != sequence_count:
        raise ValueError("slow array sequence axis must match condition rows")
    if slow.shape[2] != len(slow_channels):
        raise ValueError("slow channel axis must match slow channel schema")
    if labels.shape != (sequence_count, len(component_fields)):
        raise ValueError("label array shape must match condition rows and labels")
    if arrays.get("bidirectional"):
        _validate_bidir_array_shapes(arrays, sequence_count=sequence_count, slow=slow)
        return
    for name in ("ultrasonic", "ultrasonic_scale"):
        if arrays[name].shape[0] != sequence_count:
            raise ValueError(f"{name} sequence axis must match condition rows")
    for name in ("fiber_mic", "fiber_mic_scale"):
        if name in arrays and arrays[name].shape[0] != sequence_count:
            raise ValueError(f"{name} sequence axis must match condition rows")
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
            raise ValueError(f"{name} shape must match slow sequence and timestep axes")


def _validate_bidir_array_shapes(
    arrays: dict[str, object],
    *,
    sequence_count: int,
    slow: np.ndarray,
) -> None:
    for name in ("ultrasonic_ab", "ultrasonic_ba", "ultrasonic_ab_scale", "ultrasonic_ba_scale"):
        if name not in arrays:
            raise ValueError(f"bidirectional arrays missing {name}")
        if arrays[name].shape[0] != sequence_count:
            raise ValueError(f"{name} sequence axis must match condition rows")
    for name in ("fiber_mic", "fiber_mic_scale"):
        if name in arrays and arrays[name].shape[0] != sequence_count:
            raise ValueError(f"{name} sequence axis must match condition rows")
    for name in (
        "ultrasonic_tof_true_ab_s",
        "ultrasonic_tof_true_ba_s",
        "ultrasonic_tof_observed_ab_s",
        "ultrasonic_tof_observed_ba_s",
        "ultrasonic_peak_index_ab",
        "ultrasonic_peak_index_ba",
        "ultrasonic_tof_quality_ab",
        "ultrasonic_tof_quality_ba",
        "ultrasonic_tof_accepted_ab",
        "ultrasonic_tof_accepted_ba",
        "ultrasonic_v_path_true_m_per_s",
        "ultrasonic_sound_speed_m_per_s",
        "ultrasonic_alpha_true_npm",
    ):
        if name not in arrays:
            raise ValueError(f"bidirectional arrays missing {name}")
        if arrays[name].shape != slow.shape[:2]:
            raise ValueError(f"{name} shape must match slow sequence and timestep axes")
    forbidden_uni = ("ultrasonic", "ultrasonic_tof_s", "ultrasonic_tof_observed_s")
    present_uni = [name for name in forbidden_uni if name in arrays]
    if present_uni:
        raise ValueError(f"bidirectional arrays must not include unidirectional keys: {present_uni}")
