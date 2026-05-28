from __future__ import annotations

from collections import Counter

import numpy as np

from sim.core.schema import COMPONENT_FIELDS, LEGACY_CONDITION_FIELDS, SLOW_CHANNELS, SPLIT_NAMES


def validate_benchmark_assets(
    conditions: list[dict[str, str]],
    split_rows: dict[str, list[dict[str, str]]],
    arrays: dict[str, object] | None = None,
    labels: np.ndarray | None = None,
) -> dict[str, object]:
    _validate_no_legacy_fields(conditions)
    _validate_unique_ids(conditions, "sequence_id")
    _validate_unique_ids(conditions, "mixture_id")
    _validate_component_sums(conditions)
    _validate_split_coverage(conditions, split_rows)
    if arrays is not None and labels is not None:
        _validate_array_shapes(conditions, arrays, labels)
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


def _validate_component_sums(conditions: list[dict[str, str]]) -> None:
    for row in conditions:
        total = sum(float(row[name]) for name in COMPONENT_FIELDS)
        if abs(total - 100.0) > 1e-5:
            raise ValueError("component total must equal 100")


def _validate_split_coverage(
    conditions: list[dict[str, str]],
    split_rows: dict[str, list[dict[str, str]]],
) -> None:
    expected = {row["sequence_id"] for row in conditions}
    assigned = [row["sequence_id"] for name in SPLIT_NAMES for row in split_rows.get(name, [])]
    if set(assigned) != expected or len(assigned) != len(expected):
        raise ValueError("split rows must cover every sequence exactly once")


def _validate_array_shapes(conditions: list[dict[str, str]], arrays: dict[str, object], labels: np.ndarray) -> None:
    sequence_count = len(conditions)
    slow = arrays["slow"]
    if slow.shape[0] != sequence_count:
        raise ValueError("slow array sequence axis must match condition rows")
    if slow.shape[2] != len(SLOW_CHANNELS):
        raise ValueError("slow channel axis must match slow channel schema")
    if labels.shape != (sequence_count, len(COMPONENT_FIELDS)):
        raise ValueError("label array shape must match condition rows and labels")
    for name in ("ultrasonic", "fiber_mic", "ultrasonic_scale", "fiber_mic_scale"):
        if arrays[name].shape[0] != sequence_count:
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
