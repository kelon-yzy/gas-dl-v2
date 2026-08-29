from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import math
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from sklearn.neural_network import MLPRegressor

from gf.dl.evaluation import evaluate_predictions
from gf.sim.a2h_dataset import (
    A2HDataset,
    A2HObservation,
    A2HPhysicsConfig,
    CalibrationProfile,
    NoiseProfile,
    SENSOR_IDS,
    composition_region,
    deterministic_a2h_signal_vector,
    nominal_calibration_profile,
)


AUDIT_SCHEMA_VERSION = "gf-a2h-audit-1"
FULL_RANK_REQUIRED = 2
TARGET_TOTAL = 100.0
DEFAULT_THRESHOLDS = {
    "min_jacobian_full_rank_fraction": 0.99,
    "max_jacobian_p95_condition_number": 1000.0,
    "min_concat_relative_iid_degradation": 0.25,
    "min_oracle_relative_concat_headroom": 0.20,
    "max_outside_signal_bound_fraction": 0.0,
}


class A2HAuditError(ValueError):
    """Raised when an A2H audit cannot provide a traceable result."""


def run_difficulty_audit(
    dataset: A2HDataset,
    *,
    eval_config: Mapping[str, Any] | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    """Run the pre-registered difficulty qualification on development splits only."""

    if dataset.hard_test_indices.size:
        raise A2HAuditError(
            "difficulty audit received hard_test observations; load the locked development view"
        )
    if seed < 0:
        raise ValueError("audit seed must be non-negative")
    thresholds = dict(DEFAULT_THRESHOLDS)
    configured = (eval_config or {}).get("eligibility", {})
    if isinstance(configured, Mapping):
        for key in thresholds:
            if key in configured:
                thresholds[key] = float(configured[key])

    profile_data = _profile_data(dataset)
    base_physics = A2HPhysicsConfig.from_mapping(dataset.manifest["physics"])
    iid_train = dataset.indices(split_family="iid", split="train")
    iid_val = dataset.indices(split_family="iid", split="val")
    if len(iid_train) == 0 or len(iid_val) == 0:
        raise A2HAuditError("A2H difficulty audit requires non-empty iid train and val groups")

    concat_model, feature_mean, feature_scale = _fit_concat_model(
        dataset.signals[iid_train],
        _targets(dataset, iid_train),
        seed=seed,
    )
    iid_predictions = _predict_concat(
        concat_model,
        dataset.signals[iid_val],
        feature_mean,
        feature_scale,
    )
    iid_metrics = evaluate_predictions(
        _targets(dataset, iid_val),
        iid_predictions,
        _groups(dataset, iid_val),
        np.arange(len(iid_val), dtype=np.int64),
    )
    axis_results: dict[str, dict[str, Any]] = {}
    for split_family in _audit_families(dataset):
        train_indices = dataset.indices(split_family=split_family, split="train")
        stress_indices = dataset.indices(split_family=split_family, split="stress_val")
        val_indices = dataset.indices(split_family=split_family, split="val")
        if len(train_indices) == 0 or len(val_indices) == 0 or len(stress_indices) == 0:
            continue
        axis_model, axis_feature_mean, axis_feature_scale = _fit_concat_model(
            dataset.signals[train_indices],
            _targets(dataset, train_indices),
            seed=seed,
        )
        stress_targets = _targets(dataset, stress_indices)
        stress_predictions = _predict_concat(
            axis_model,
            dataset.signals[stress_indices],
            axis_feature_mean,
            axis_feature_scale,
        )
        stress_metrics = evaluate_predictions(
            stress_targets,
            stress_predictions,
            _groups(dataset, stress_indices),
            np.arange(len(stress_indices), dtype=np.int64),
        )
        val_predictions = _predict_concat(
            axis_model,
            dataset.signals[val_indices],
            axis_feature_mean,
            axis_feature_scale,
        )
        val_metrics = evaluate_predictions(
            _targets(dataset, val_indices),
            val_predictions,
            _groups(dataset, val_indices),
            np.arange(len(val_indices), dtype=np.int64),
        )

        observations = _one_observation_per_group(dataset, stress_indices)
        jacobian = _jacobian_summary(observations, base_physics, profile_data)
        true_oracle_predictions = _oracle_predictions(
            dataset,
            stress_indices,
            base_physics=base_physics,
            profile_data=profile_data,
            use_true_context=True,
        )
        nominal_oracle_predictions = _oracle_predictions(
            dataset,
            stress_indices,
            base_physics=base_physics,
            profile_data=profile_data,
            use_true_context=False,
        )
        true_oracle_metrics = evaluate_predictions(
            stress_targets,
            true_oracle_predictions,
            _groups(dataset, stress_indices),
            np.arange(len(stress_indices), dtype=np.int64),
        )
        nominal_oracle_metrics = evaluate_predictions(
            stress_targets,
            nominal_oracle_predictions,
            _groups(dataset, stress_indices),
            np.arange(len(stress_indices), dtype=np.int64),
        )
        nearest_neighbor = nearest_neighbor_coverage(
            _unique_group_targets(dataset, train_indices),
            stress_targets,
        )
        physical = _physical_signal_summary(
            dataset,
            np.concatenate((train_indices, stress_indices)),
            base_physics,
        )
        eligibility = evaluate_difficulty_eligibility(
            jacobian=jacobian,
            concat_iid_macro=float(val_metrics["macro_RNMAE"]),
            concat_stress_macro=float(stress_metrics["macro_RNMAE"]),
            oracle_stress_macro=float(true_oracle_metrics["macro_RNMAE"]),
            outside_bound_fraction=float(physical["outside_bound_fraction"]),
            thresholds=thresholds,
        )
        axis_results[split_family] = {
            "split_family": split_family,
            "split_counts": _axis_split_counts(dataset, split_family),
            "concat_baseline": {
                "iid_val": iid_metrics,
                "family_val": val_metrics,
                "stress_val": stress_metrics,
                "relative_iid_degradation": _relative_degradation(
                    float(val_metrics["macro_RNMAE"]),
                    float(stress_metrics["macro_RNMAE"]),
                ),
            },
            "oracle": {
                "true_context_and_profile": true_oracle_metrics,
                "nominal_context_and_profile": nominal_oracle_metrics,
                "true_profile_headroom": _relative_improvement(
                    float(stress_metrics["macro_RNMAE"]),
                    float(true_oracle_metrics["macro_RNMAE"]),
                ),
                "oracle_gap_to_concat": _relative_improvement(
                    float(stress_metrics["macro_RNMAE"]),
                    float(true_oracle_metrics["macro_RNMAE"]),
                ),
            },
            "jacobian": jacobian,
            "physical_signal": physical,
            "nearest_neighbor": nearest_neighbor,
            "stratification": _stratified_counts(dataset, stress_indices),
            "eligibility": eligibility,
        }

    eligible_axes = [
        name
        for name, result in axis_results.items()
        if result["eligibility"]["status"] == "PASS"
    ]
    minimum_axes = int((eval_config or {}).get("eligibility", {}).get("min_eligible_axes", 2))
    if len(eligible_axes) >= minimum_axes:
        status = "PASS"
    else:
        status = "STOPPED_INSUFFICIENT_ELIGIBLE_AXES"
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": status,
        "audit_seed": int(seed),
        "development_only": True,
        "thresholds": thresholds,
        "iid_reference": {
            "train_group_count": int(len(iid_train)),
            "val_group_count": int(len(iid_val)),
            "concat_baseline": iid_metrics,
        },
        "axes": axis_results,
        "eligible_axes": eligible_axes,
        "minimum_eligible_axes": minimum_axes,
        "decision": (
            "enter_A2H_algorithm_comparison"
            if status == "PASS"
            else "stop_before_algorithm_comparison"
        ),
    }


def evaluate_difficulty_eligibility(
    *,
    jacobian: Mapping[str, Any],
    concat_iid_macro: float,
    concat_stress_macro: float,
    oracle_stress_macro: float,
    outside_bound_fraction: float,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    limits = dict(DEFAULT_THRESHOLDS)
    if thresholds is not None:
        limits.update({key: float(value) for key, value in thresholds.items()})
    if concat_iid_macro <= 0.0 or concat_stress_macro < 0.0 or oracle_stress_macro < 0.0:
        raise ValueError("difficulty metrics must be non-negative and iid concat must be positive")
    degradation = _relative_degradation(concat_iid_macro, concat_stress_macro)
    headroom = _relative_improvement(concat_stress_macro, oracle_stress_macro)
    checks = {
        "jacobian_full_rank_fraction": float(jacobian["full_rank_fraction"]) >= limits["min_jacobian_full_rank_fraction"],
        "jacobian_p95_condition_number": float(jacobian["condition_number_p95"]) < limits["max_jacobian_p95_condition_number"],
        "concat_relative_iid_degradation": degradation >= limits["min_concat_relative_iid_degradation"],
        "oracle_relative_concat_headroom": headroom >= limits["min_oracle_relative_concat_headroom"],
        "signals_within_registered_bounds": outside_bound_fraction <= limits["max_outside_signal_bound_fraction"],
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failure_reasons": failures,
        "concat_relative_iid_degradation": degradation,
        "oracle_relative_concat_headroom": headroom,
    }


def finite_difference_jacobian(
    composition: Sequence[float],
    *,
    physics: A2HPhysicsConfig,
    calibration: CalibrationProfile,
    step_pct: float | None = None,
) -> np.ndarray:
    values = np.asarray(composition, dtype=np.float64)
    if values.shape != (3,) or not np.allclose(values.sum(), TARGET_TOTAL, atol=1.0e-8):
        raise ValueError("composition must have shape (3,) and sum to 100")
    step = physics.jacobian_step_pct if step_pct is None else float(step_pct)
    if step <= 0.0 or min(values) <= step:
        raise ValueError("finite_difference_jacobian requires an interior composition")
    columns: list[np.ndarray] = []
    for component_index in (0, 1):
        direction = np.zeros(3, dtype=np.float64)
        direction[component_index] = step
        direction[2] = -step
        plus = deterministic_a2h_signal_vector(values + direction, physics=physics, calibration=calibration)
        minus = deterministic_a2h_signal_vector(values - direction, physics=physics, calibration=calibration)
        columns.append((plus - minus) / (2.0 * step))
    return np.column_stack(columns)


def invert_observation(
    observed_signal: Sequence[float],
    *,
    physics: A2HPhysicsConfig,
    calibration: CalibrationProfile,
    noise: NoiseProfile,
) -> dict[str, Any]:
    """Invert one observation with registered context and no target-derived initialization."""

    observed = np.asarray(observed_signal, dtype=np.float64)
    if observed.shape != (len(SENSOR_IDS),) or not np.isfinite(observed).all():
        raise ValueError("observed_signal must contain one finite value per sensor")
    scales = _noise_scales(physics) * max(noise.white_scale, 1.0)
    scales = np.maximum(scales, np.finfo(np.float64).tiny)

    def composition_from_logits(logits: np.ndarray) -> np.ndarray:
        extended = np.asarray([logits[0], logits[1], 0.0], dtype=np.float64)
        extended -= extended.max()
        probabilities = np.exp(extended)
        probabilities /= probabilities.sum()
        return probabilities * TARGET_TOTAL

    def residuals(logits: np.ndarray) -> np.ndarray:
        composition = composition_from_logits(logits)
        prediction = deterministic_a2h_signal_vector(
            composition,
            physics=physics,
            calibration=calibration,
        )
        return (prediction - observed) / scales

    starts = np.asarray(
        [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [-4.0, -4.0]],
        dtype=np.float64,
    )
    results = [
        least_squares(
            residuals,
            start,
            max_nfev=500,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
        )
        for start in starts
    ]
    successful = [result for result in results if result.success and np.isfinite(result.cost)]
    if not successful:
        messages = [str(result.message) for result in results]
        raise A2HAuditError(f"oracle inversion failed for all registered starts: {messages}")
    best = min(successful, key=lambda result: float(result.cost))
    composition = composition_from_logits(best.x)
    return {
        "composition": composition,
        "objective": float(2.0 * best.cost),
        "iterations": int(getattr(best, "nfev", 0)),
        "starts_attempted": len(starts),
    }


def nearest_neighbor_coverage(
    train_compositions: np.ndarray,
    query_compositions: np.ndarray,
) -> dict[str, Any]:
    train_values = np.asarray(train_compositions, dtype=np.float64)
    query_values = np.asarray(query_compositions, dtype=np.float64)
    if train_values.ndim != 2 or query_values.ndim != 2 or train_values.shape[1] != 3 or query_values.shape[1] != 3:
        raise ValueError("composition arrays must have shape [N,3]")
    if len(train_values) == 0 or len(query_values) == 0:
        raise ValueError("nearest neighbor coverage requires non-empty arrays")
    distances = np.sqrt(
        ((query_values[:, None, :] - train_values[None, :, :]) ** 2).sum(axis=2)
    ).min(axis=1)
    return {
        "train_group_count": int(len(train_values)),
        "query_group_count": int(len(query_values)),
        "distance_metric": "euclidean_mol_percent",
        "mean_distance": float(distances.mean()),
        "median_distance": float(np.median(distances)),
        "P90_distance": float(np.percentile(distances, 90)),
        "max_distance": float(distances.max()),
        "distances": [float(value) for value in distances],
    }


def _jacobian_summary(
    observations: Sequence[A2HObservation],
    base_physics: A2HPhysicsConfig,
    profile_data: Mapping[str, Any],
) -> dict[str, Any]:
    ranks: list[int] = []
    condition_numbers: list[float] = []
    directions: list[dict[str, float]] = []
    for observation in observations:
        if observation.condition_family == "pure" or observation.condition_family == "binary":
            continue
        physics, calibration = _observation_context(observation, base_physics, profile_data, use_true_context=True)
        jacobian = finite_difference_jacobian(
            observation.composition,
            physics=physics,
            calibration=calibration,
        )
        normalized = jacobian / _resolution_scales(physics)[:, None]
        _, singular_values, right_vectors = np.linalg.svd(normalized, full_matrices=False)
        rank = int(np.linalg.matrix_rank(normalized, tol=1.0e-10))
        condition = float("inf") if singular_values[-1] <= 0.0 else float(singular_values[0] / singular_values[-1])
        ranks.append(rank)
        condition_numbers.append(condition)
        direction = right_vectors[-1]
        directions.append({"d_x_Ar_pct": float(direction[0]), "d_x_He_pct": float(direction[1])})
    if not ranks:
        return {
            "interior_count": 0,
            "full_rank_fraction": 0.0,
            "rank_min": 0,
            "rank_max": 0,
            "condition_number_median": float("inf"),
            "condition_number_p95": float("inf"),
            "condition_number_max": float("inf"),
            "worst_degenerate_direction": None,
        }
    condition_array = np.asarray(condition_numbers, dtype=np.float64)
    worst_index = int(np.argmax(condition_array))
    return {
        "interior_count": len(ranks),
        "full_rank_fraction": float(np.mean(np.asarray(ranks) == FULL_RANK_REQUIRED)),
        "rank_min": int(min(ranks)),
        "rank_max": int(max(ranks)),
        "condition_number_median": float(np.median(condition_array)),
        "condition_number_p95": float(np.percentile(condition_array, 95)),
        "condition_number_max": float(condition_array.max()),
        "worst_degenerate_direction": directions[worst_index],
    }


def _oracle_predictions(
    dataset: A2HDataset,
    indices: np.ndarray,
    *,
    base_physics: A2HPhysicsConfig,
    profile_data: Mapping[str, Any],
    use_true_context: bool,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for index in indices:
        observation = dataset.observations[int(index)]
        physics, calibration = _observation_context(
            observation,
            base_physics,
            profile_data,
            use_true_context=use_true_context,
        )
        noise = profile_data["noises"][observation.noise_profile_id]
        result = invert_observation(
            dataset.signals[int(index)],
            physics=physics,
            calibration=calibration,
            noise=noise,
        )
        predictions.append(result["composition"])
    return np.vstack(predictions)


def _observation_context(
    observation: A2HObservation,
    base_physics: A2HPhysicsConfig,
    profile_data: Mapping[str, Any],
    *,
    use_true_context: bool,
) -> tuple[A2HPhysicsConfig, CalibrationProfile]:
    if use_true_context:
        environment = profile_data["environments"][observation.environment_id]
        calibration = profile_data["calibrations"][observation.calibration_profile_id]
    else:
        environment = profile_data["environments"][profile_data["nominal_environment_id"]]
        calibration = nominal_calibration_profile()
    physics = replace(
        base_physics,
        temperature_k=float(environment["temperature_k"]),
        pressure_pa=float(environment["pressure_pa"]),
    )
    return physics, calibration


def _physical_signal_summary(
    dataset: A2HDataset,
    indices: np.ndarray,
    physics: A2HPhysicsConfig,
) -> dict[str, Any]:
    values = dataset.signals[indices]
    outside_by_sensor: dict[str, int] = {}
    saturated_by_sensor: dict[str, int] = {}
    for sensor_index, sensor_id in enumerate(SENSOR_IDS):
        lower, upper = physics.signal_bounds[sensor_id]
        sensor_values = values[:, sensor_index]
        outside_by_sensor[sensor_id] = int(np.sum((sensor_values < lower) | (sensor_values > upper)))
        saturated_by_sensor[sensor_id] = int(np.sum((sensor_values <= lower) | (sensor_values >= upper)))
    outside_count = sum(outside_by_sensor.values())
    return {
        "sample_count": int(len(indices)),
        "signal_bounds": {sensor_id: list(physics.signal_bounds[sensor_id]) for sensor_id in SENSOR_IDS},
        "outside_bound_fraction": float(outside_count / max(values.size, 1)),
        "outside_bound_by_sensor": outside_by_sensor,
        "saturated_by_sensor": saturated_by_sensor,
        "dynamic_range": {
            sensor_id: {
                "minimum": float(values[:, index].min()),
                "maximum": float(values[:, index].max()),
                "range": float(values[:, index].max() - values[:, index].min()),
                "resolution": float(_resolution_scales(physics)[index]),
                "range_to_resolution": float(
                    (values[:, index].max() - values[:, index].min()) / _resolution_scales(physics)[index]
                ),
            }
            for index, sensor_id in enumerate(SENSOR_IDS)
        },
    }


def _fit_concat_model(
    signals: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
) -> tuple[MLPRegressor, np.ndarray, np.ndarray]:
    features = np.asarray(signals, dtype=np.float64)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    if np.any(feature_scale <= 0.0) or not np.isfinite(feature_scale).all():
        raise A2HAuditError("concat baseline cannot fit a zero-variance sensor on iid train")
    model = MLPRegressor(
        hidden_layer_sizes=(32,),
        solver="lbfgs",
        alpha=1.0e-4,
        max_iter=1000,
        random_state=seed,
    )
    model.fit((features - feature_mean) / feature_scale, np.asarray(targets, dtype=np.float64) / TARGET_TOTAL)
    return model, feature_mean, feature_scale


def _predict_concat(
    model: MLPRegressor,
    signals: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
) -> np.ndarray:
    return np.asarray(model.predict((signals - feature_mean) / feature_scale) * TARGET_TOTAL, dtype=np.float64)


def _profile_data(dataset: A2HDataset) -> dict[str, Any]:
    environments = {
        str(item["environment_id"]): {
            "temperature_k": float(item["temperature_k"]),
            "pressure_pa": float(item["pressure_pa"]),
        }
        for item in dataset.manifest["environment_blocks"]
    }
    calibrations = {
        profile.calibration_profile_id: profile
        for profile in (CalibrationProfile.from_mapping(item) for item in dataset.manifest["calibration_profiles"])
    }
    noises = {
        profile.noise_profile_id: profile
        for profile in (NoiseProfile.from_mapping(item) for item in dataset.manifest["noise_profiles"])
    }
    nominal_ids = [environment_id for environment_id in environments if "NOMINAL" in environment_id]
    if len(nominal_ids) != 1:
        raise A2HAuditError("A2H manifest must contain exactly one nominal environment")
    return {
        "environments": environments,
        "calibrations": calibrations,
        "noises": noises,
        "nominal_environment_id": nominal_ids[0],
    }


def _resolution_scales(physics: A2HPhysicsConfig) -> np.ndarray:
    return np.asarray(
        [
            physics.tof_resolution_s,
            physics.thermal_voltage_resolution_v,
            physics.ndir_voltage_noise_std_v,
        ],
        dtype=np.float64,
    )


def _targets(dataset: A2HDataset, indices: Sequence[int]) -> np.ndarray:
    return np.asarray([dataset.observations[int(index)].composition for index in indices], dtype=np.float64)


def _groups(dataset: A2HDataset, indices: Sequence[int]) -> tuple[str, ...]:
    return tuple(dataset.observations[int(index)].mixture_id for index in indices)


def _unique_group_targets(dataset: A2HDataset, indices: Sequence[int]) -> np.ndarray:
    values: dict[str, tuple[float, float, float]] = {}
    for index in indices:
        observation = dataset.observations[int(index)]
        values.setdefault(observation.mixture_id, observation.composition)
    return np.asarray([values[key] for key in sorted(values)], dtype=np.float64)


def _one_observation_per_group(dataset: A2HDataset, indices: Sequence[int]) -> list[A2HObservation]:
    selected: dict[str, A2HObservation] = {}
    for index in indices:
        observation = dataset.observations[int(index)]
        selected.setdefault(observation.mixture_id, observation)
    return [selected[key] for key in sorted(selected)]


def _audit_families(dataset: A2HDataset) -> tuple[str, ...]:
    families = sorted({observation.split_family for observation in dataset.observations})
    return tuple(family for family in families if family != "iid")


def _axis_split_counts(dataset: A2HDataset, split_family: str) -> dict[str, int]:
    return {
        split: int(len(dataset.indices(split_family=split_family, split=split)))
        for split in ("train", "val", "stress_val")
    }


def _stratified_counts(dataset: A2HDataset, indices: Sequence[int]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    regions = dataset.manifest.get("composition_regions", {})
    for index in indices:
        observation = dataset.observations[int(index)]
        values = {
            "condition_family": observation.condition_family,
            "environment_id": observation.environment_id,
            "calibration_profile_id": observation.calibration_profile_id,
            "noise_profile_id": observation.noise_profile_id,
            "composition_region": composition_region(observation.composition, regions=regions),
        }
        for field, value in values.items():
            field_counts = result.setdefault(field, {})
            field_counts[value] = field_counts.get(value, 0) + 1
    return result


def _relative_degradation(reference: float, value: float) -> float:
    if reference <= 0.0:
        raise ValueError("relative degradation reference must be positive")
    return float((value - reference) / reference)


def _relative_improvement(reference: float, value: float) -> float:
    if reference <= 0.0:
        raise ValueError("relative improvement reference must be positive")
    return float((reference - value) / reference)


def _noise_scales(physics: A2HPhysicsConfig) -> np.ndarray:
    return np.asarray(
        [
            physics.tof_resolution_s / math.sqrt(12.0),
            physics.thermal_voltage_resolution_v / math.sqrt(12.0),
            physics.ndir_voltage_noise_std_v,
        ],
        dtype=np.float64,
    )


__all__ = [
    "A2HAuditError",
    "AUDIT_SCHEMA_VERSION",
    "evaluate_difficulty_eligibility",
    "finite_difference_jacobian",
    "invert_observation",
    "nearest_neighbor_coverage",
    "run_difficulty_audit",
]
