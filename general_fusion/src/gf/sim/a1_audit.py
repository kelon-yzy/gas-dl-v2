from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from gf.sim.a1_dataset import (
    A1Dataset,
    A1PhysicsConfig,
    DEFAULT_A1_PHYSICS,
    SENSOR_IDS,
    deterministic_signal_vector,
)


AUDIT_SCHEMA_VERSION = "gf-a1-audit-1"
FULL_RANK_REQUIRED = 2
MAX_P95_CONDITION_NUMBER = 1_000.0
MIN_DYNAMIC_RANGE_TO_RESOLUTION = 10.0
MAX_DIRECTION_COSINE = 0.999
MIN_NON_SATURATED_MODALITIES = 2


def run_information_audit(
    dataset: A1Dataset,
    physics: A1PhysicsConfig = DEFAULT_A1_PHYSICS,
) -> dict[str, Any]:
    physics.validate()
    deterministic_values = np.vstack(
        [deterministic_signal_vector(condition.composition, physics) for condition in dataset.conditions]
    )
    reference_compositions = ((100.0, 0.0, 0.0), (0.0, 100.0, 0.0), (0.0, 0.0, 100.0))
    reference_values = np.vstack(
        [deterministic_signal_vector(composition, physics) for composition in reference_compositions]
    )
    all_values = np.vstack([deterministic_values, reference_values])

    jacobians, condition_numbers, degenerate_directions = _finite_difference_audit(
        dataset, physics
    )
    normalized_jacobians = jacobians / _resolution_scales(physics)[None, :, None]
    ranks = np.array(
        [np.linalg.matrix_rank(jacobian, tol=1.0e-10) for jacobian in normalized_jacobians],
        dtype=np.int64,
    )
    direction_cosines = _direction_cosines(normalized_jacobians)
    dynamic_range = _dynamic_range(all_values, physics)
    saturation = _saturation_summary(deterministic_values, physics)
    non_saturated_modalities = np.sum(
        np.logical_not(
            np.column_stack(
                [
                    np.asarray(saturation[sensor_id]["sample_saturated"], dtype=bool)
                    for sensor_id in SENSOR_IDS
                ]
            )
        ),
        axis=1,
    )

    p95_condition = float(np.percentile(condition_numbers, 95))
    checks = {
        "jacobian_full_rank_fraction": bool(np.mean(ranks == FULL_RANK_REQUIRED) >= 1.0),
        "jacobian_p95_condition_number": p95_condition < MAX_P95_CONDITION_NUMBER,
        "dynamic_range_resolution": all(
            item["range_to_resolution"] >= MIN_DYNAMIC_RANGE_TO_RESOLUTION
            for item in dynamic_range.values()
        ),
        "non_saturated_modalities": bool(
            np.all(non_saturated_modalities >= MIN_NON_SATURATED_MODALITIES)
        ),
        "modality_direction_separation": bool(
            direction_cosines["ndir_max_abs_cosine"] < MAX_DIRECTION_COSINE
        ),
    }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "sample_count": len(dataset.conditions),
        "reference_point_count": len(reference_compositions),
        "jacobian": {
            "step_pct": physics.jacobian_step_pct,
            "interior_count": len(jacobians),
            "rank_min": int(ranks.min()),
            "rank_max": int(ranks.max()),
            "full_rank_fraction": float(np.mean(ranks == FULL_RANK_REQUIRED)),
            "condition_number_median": float(np.median(condition_numbers)),
            "condition_number_p95": p95_condition,
            "condition_number_max": float(np.max(condition_numbers)),
            "worst_degenerate_direction": degenerate_directions[
                int(np.argmax(condition_numbers))
            ],
        },
        "dynamic_range": dynamic_range,
        "saturation": saturation,
        "degeneration_directions": direction_cosines,
        "gate": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "thresholds": {
                "max_p95_condition_number": MAX_P95_CONDITION_NUMBER,
                "min_dynamic_range_to_resolution": MIN_DYNAMIC_RANGE_TO_RESOLUTION,
                "max_direction_cosine": MAX_DIRECTION_COSINE,
                "min_non_saturated_modalities": MIN_NON_SATURATED_MODALITIES,
            },
        },
    }


def _finite_difference_audit(
    dataset: A1Dataset,
    physics: A1PhysicsConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    jacobians: list[np.ndarray] = []
    condition_numbers: list[float] = []
    directions: list[dict[str, float]] = []
    step = physics.jacobian_step_pct
    for condition in dataset.conditions:
        ar, he, co2 = condition.composition
        if min(ar, he, co2) <= step:
            continue
        base = np.array([ar, he, co2], dtype=np.float64)
        plus_ar = base + np.array([step, 0.0, -step])
        minus_ar = base - np.array([step, 0.0, -step])
        plus_he = base + np.array([0.0, step, -step])
        minus_he = base - np.array([0.0, step, -step])
        jacobian = np.column_stack(
            [
                (
                    deterministic_signal_vector(plus_ar, physics)
                    - deterministic_signal_vector(minus_ar, physics)
                )
                / (2.0 * step),
                (
                    deterministic_signal_vector(plus_he, physics)
                    - deterministic_signal_vector(minus_he, physics)
                )
                / (2.0 * step),
            ]
        )
        normalized = jacobian / _resolution_scales(physics)[:, None]
        singular_values = np.linalg.svd(normalized, compute_uv=False)
        if singular_values[-1] <= 0.0:
            condition_number = float("inf")
        else:
            condition_number = float(singular_values[0] / singular_values[-1])
        right_singular_vectors = np.linalg.svd(normalized, full_matrices=False)[2]
        worst_direction = right_singular_vectors[-1]
        jacobians.append(jacobian)
        condition_numbers.append(condition_number)
        directions.append(
            {
                "d_x_Ar_pct": float(worst_direction[0]),
                "d_x_He_pct": float(worst_direction[1]),
            }
        )
    if not jacobians:
        raise ValueError("information audit requires at least one interior composition")
    return np.asarray(jacobians), np.asarray(condition_numbers), directions


def _resolution_scales(physics: A1PhysicsConfig) -> np.ndarray:
    return np.array(
        [
            physics.tof_resolution_s,
            physics.thermal_voltage_resolution_v,
            physics.ndir_voltage_noise_std_v,
        ],
        dtype=np.float64,
    )


def _dynamic_range(values: np.ndarray, physics: A1PhysicsConfig) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    resolutions = _resolution_scales(physics)
    for sensor_index, sensor_id in enumerate(SENSOR_IDS):
        sensor_values = values[:, sensor_index]
        lower, upper = physics.signal_bounds[sensor_id]
        value_range = float(sensor_values.max() - sensor_values.min())
        result[sensor_id] = {
            "minimum": float(sensor_values.min()),
            "maximum": float(sensor_values.max()),
            "range": value_range,
            "resolution": float(resolutions[sensor_index]),
            "range_to_resolution": float(value_range / resolutions[sensor_index]),
            "bound_min": lower,
            "bound_max": upper,
        }
    return result


def _saturation_summary(
    values: np.ndarray,
    physics: A1PhysicsConfig,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for sensor_index, sensor_id in enumerate(SENSOR_IDS):
        lower, upper = physics.signal_bounds[sensor_id]
        sensor_values = values[:, sensor_index]
        saturated = (sensor_values <= lower) | (sensor_values >= upper)
        result[sensor_id] = {
            "sample_saturated": saturated.tolist(),
            "sample_saturated_count": int(saturated.sum()),
            "sample_saturated_fraction": float(saturated.mean()),
            "outside_bound_count": int(((sensor_values < lower) | (sensor_values > upper)).sum()),
            "reference_endpoint_note": (
                "reference pure-gas points are audit-only and may lie on a response endpoint"
            ),
        }
    return result


def _direction_cosines(normalized_jacobians: np.ndarray) -> dict[str, Any]:
    pair_values: dict[str, list[float]] = {}
    for left, right in combinations(range(len(SENSOR_IDS)), 2):
        values: list[float] = []
        for jacobian in normalized_jacobians:
            left_vector = jacobian[left]
            right_vector = jacobian[right]
            denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
            if denominator <= 0.0:
                values.append(1.0)
            else:
                values.append(float(np.dot(left_vector, right_vector) / denominator))
        pair_values[f"{SENSOR_IDS[left]}__{SENSOR_IDS[right]}"] = values
    all_values = np.asarray([abs(value) for values in pair_values.values() for value in values])
    ndir_values = np.asarray(
        [
            abs(value)
            for pair, values in pair_values.items()
            if pair.endswith("__ndir_co2_voltage")
            for value in values
        ]
    )
    return {
        "pair_abs_cosine_max": {
            pair: float(np.max(np.abs(values))) for pair, values in pair_values.items()
        },
        "pair_abs_cosine_median": {
            pair: float(np.median(np.abs(values))) for pair, values in pair_values.items()
        },
        "max_abs_cosine": float(np.max(all_values)),
        "ndir_max_abs_cosine": float(np.max(ndir_values)),
    }
