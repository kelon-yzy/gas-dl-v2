from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

from gf.dl.evaluation import (
    TARGET_RANGES,
    evaluate_predictions,
    group_bootstrap_comparison,
)
from gf.dl.preprocessing import TrainGroupStandardScaler
from gf.sim.a1_dataset import A1Dataset, SENSOR_IDS, TARGET_NAMES


RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0)
GBDT_CONFIGS = (
    {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 200, "learning_rate": 0.03, "max_depth": 2},
)


@dataclass(frozen=True)
class BaselineSuiteResult:
    summary: Mapping[str, Any]
    predictions: Mapping[str, np.ndarray]
    targets: np.ndarray
    group_ids: tuple[str, ...]


def fit_full_regression_baselines(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    *,
    seed: int = 17,
) -> dict[str, dict[str, Any]]:
    """Fit the registered full-input B3 and B4 implementations once."""
    groups = np.arange(len(targets)).astype(str)
    result: dict[str, dict[str, Any]] = {}
    for baseline_id, fitter in (
        ("B3", _fit_full_ridge),
        ("B4", _fit_full_gbdt),
    ):
        fitted = fitter(features, targets, train_indices, validation_indices, seed)
        prediction = np.asarray(fitted["prediction"], dtype=np.float64)
        result[baseline_id] = {
            "prediction": prediction,
            "validation": evaluate_predictions(
                targets,
                prediction,
                groups,
                validation_indices,
            ),
            "parameters": dict(fitted["parameters"]),
            "resources": dict(fitted["resources"]),
        }
    return result


def run_baseline_suite(
    dataset: A1Dataset,
    *,
    training_seed: int = 20260827,
    include_mlp: bool = False,
    mlp_seeds: Sequence[int] = (17, 29, 43, 71, 101),
    bootstrap_seed: int = 20260827,
    bootstrap_samples: int = 0,
) -> BaselineSuiteResult:
    samples = dataset.samples()
    train_indices, val_indices, test_indices = _split_indices(dataset)
    train_group_ids = {samples[index].group_id for index in train_indices}
    scaler = TrainGroupStandardScaler()
    scaler.fit(samples, train_group_ids)
    scaled_samples = [scaler.transform(sample) for sample in samples]
    features = _feature_matrix(scaled_samples)
    targets = np.vstack([sample.target for sample in samples]).astype(np.float64)
    groups = np.array([sample.group_id for sample in samples], dtype=object)

    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    single_results: dict[tuple[str, str], dict[str, Any]] = {}

    mean_prediction = np.tile(targets[train_indices].mean(axis=0), (len(targets), 1))
    predictions["B0"] = mean_prediction
    rows.append(
        _make_row(
            baseline_id="B0",
            variant="mean",
            description="train-target mean",
            prediction=mean_prediction,
            targets=targets,
            groups=groups,
            val_indices=val_indices,
            test_indices=test_indices,
            parameters={},
            resources={"training_time_s": 0.0, "inference_time_s": 0.0, "parameter_count": 0},
        )
    )

    for baseline_id, algorithm in (("B1", "ridge"), ("B2", "gbdt")):
        for sensor_index, sensor_id in enumerate(SENSOR_IDS):
            variant = f"{sensor_id}"
            candidate = _fit_single_sensor(
                algorithm=algorithm,
                features=features[:, [sensor_index]],
                targets=targets,
                train_indices=train_indices,
                val_indices=val_indices,
                seed=training_seed,
            )
            prediction = candidate["prediction"]
            key = f"{baseline_id}__{variant}"
            predictions[key] = prediction
            row = _make_row(
                baseline_id=baseline_id,
                variant=variant,
                description=f"single-sensor {algorithm}",
                prediction=prediction,
                targets=targets,
                groups=groups,
                val_indices=val_indices,
                test_indices=test_indices,
                parameters=candidate["parameters"],
                resources=candidate["resources"],
            )
            rows.append(row)
            single_results[(baseline_id, sensor_id)] = {
                "key": key,
                "row": row,
                "prediction": prediction,
            }

    ridge_candidate = _fit_full_ridge(
        features,
        targets,
        train_indices,
        val_indices,
        training_seed,
    )
    predictions["B3"] = ridge_candidate["prediction"]
    rows.append(
        _make_row(
            baseline_id="B3",
            variant="full",
            description="full-input Ridge",
            prediction=predictions["B3"],
            targets=targets,
            groups=groups,
            val_indices=val_indices,
            test_indices=test_indices,
            parameters=ridge_candidate["parameters"],
            resources=ridge_candidate["resources"],
        )
    )

    gbdt_candidate = _fit_full_gbdt(
        features,
        targets,
        train_indices,
        val_indices,
        training_seed,
    )
    predictions["B4"] = gbdt_candidate["prediction"]
    rows.append(
        _make_row(
            baseline_id="B4",
            variant="full",
            description="full-input GBDT",
            prediction=predictions["B4"],
            targets=targets,
            groups=groups,
            val_indices=val_indices,
            test_indices=test_indices,
            parameters=gbdt_candidate["parameters"],
            resources=gbdt_candidate["resources"],
        )
    )

    selected_single = {
        sensor_id: min(
            (
                single_results[("B1", sensor_id)],
                single_results[("B2", sensor_id)],
            ),
            key=lambda item: item["row"]["validation"]["macro_RNMAE"],
        )
        for sensor_id in SENSOR_IDS
    }
    late_predictions = np.mean(
        [selected_single[sensor_id]["prediction"] for sensor_id in SENSOR_IDS],
        axis=0,
    )
    predictions["B6"] = late_predictions
    rows.append(
        _make_row(
            baseline_id="B6",
            variant="equal_weight_late_fusion",
            description="equal-weight late fusion of selected single-sensor models",
            prediction=late_predictions,
            targets=targets,
            groups=groups,
            val_indices=val_indices,
            test_indices=test_indices,
            parameters={
                "selected_single_models": {
                    sensor_id: selected_single[sensor_id]["key"] for sensor_id in SENSOR_IDS
                },
                "weights": {sensor_id: 1.0 / len(SENSOR_IDS) for sensor_id in SENSOR_IDS},
            },
            resources=_sum_resources(
                [selected_single[sensor_id]["row"]["resources"] for sensor_id in SENSOR_IDS]
            ),
        )
    )

    validation_errors = np.array(
        [
            selected_single[sensor_id]["row"]["validation"]["macro_RNMAE"]
            for sensor_id in SENSOR_IDS
        ],
        dtype=np.float64,
    )
    if np.any(validation_errors <= 0.0):
        raise ValueError("late-fusion validation weights require strictly positive validation errors")
    inverse_errors = 1.0 / validation_errors
    fixed_weights = inverse_errors / inverse_errors.sum()
    fixed_predictions = sum(
        weight * selected_single[sensor_id]["prediction"]
        for weight, sensor_id in zip(fixed_weights, SENSOR_IDS, strict=True)
    )
    predictions["B7"] = fixed_predictions
    rows.append(
        _make_row(
            baseline_id="B7",
            variant="validation_inverse_error",
            description="validation-only inverse-error late fusion",
            prediction=fixed_predictions,
            targets=targets,
            groups=groups,
            val_indices=val_indices,
            test_indices=test_indices,
            parameters={
                "selected_single_models": {
                    sensor_id: selected_single[sensor_id]["key"] for sensor_id in SENSOR_IDS
                },
                "weights": {
                    sensor_id: float(weight)
                    for sensor_id, weight in zip(SENSOR_IDS, fixed_weights, strict=True)
                },
            },
            resources=_sum_resources(
                [selected_single[sensor_id]["row"]["resources"] for sensor_id in SENSOR_IDS]
            ),
        )
    )

    mlp_predictions: list[np.ndarray] = []
    mlp_resources: list[Mapping[str, Any]] = []
    if include_mlp:
        for seed in mlp_seeds:
            mlp_result = _fit_mlp(
                features,
                targets,
                train_indices,
                seed=seed,
            )
            mlp_prediction = mlp_result["prediction"]
            mlp_resources.append(mlp_result["resources"])
            mlp_predictions.append(mlp_prediction)
            key = f"B5__seed_{seed}"
            predictions[key] = mlp_prediction
            rows.append(
                _make_row(
                    baseline_id="B5",
                    variant=f"seed_{seed}",
                    description="full-input concatenation MLP",
                    prediction=mlp_prediction,
                    targets=targets,
                    groups=groups,
                    val_indices=val_indices,
                    test_indices=test_indices,
                    parameters={
                        "seed": int(seed),
                        "hidden_layer_sizes": [32],
                        "solver": "lbfgs",
                        "max_iter": 1000,
                    },
                    resources=mlp_result["resources"],
                )
            )
        mean_prediction = np.mean(mlp_predictions, axis=0)
        predictions["B5__mean"] = mean_prediction
        rows.append(
            _make_row(
                baseline_id="B5",
                variant="mean",
                description="full-input concatenation MLP mean across frozen seeds",
                prediction=mean_prediction,
                targets=targets,
                groups=groups,
                val_indices=val_indices,
                test_indices=test_indices,
                parameters={
                    "seeds": [int(seed) for seed in mlp_seeds],
                    "aggregation": "arithmetic_mean_of_predictions",
                },
                resources=_sum_resources(mlp_resources),
            )
        )

    best_single = min(
        (
            row
            for row in rows
            if row["baseline_id"] in {"B1", "B2"}
        ),
        key=lambda row: row["validation"]["macro_RNMAE"],
    )
    full_candidates = [
        row for row in rows if row["baseline_id"] in {"B3", "B4", "B6", "B7"}
    ]
    best_full = min(full_candidates, key=lambda row: row["validation"]["macro_RNMAE"])
    overall_full_candidates = [
        row
        for row in rows
        if row["baseline_id"] in {"B3", "B4", "B6", "B7"}
        or (row["baseline_id"] == "B5" and row["variant"] == "mean")
    ]
    best_overall_full = min(
        overall_full_candidates,
        key=lambda row: row["validation"]["macro_RNMAE"],
    )
    gate = _pilot_gate(best_single, full_candidates, targets, rows)
    bootstrap = {}
    if bootstrap_samples > 0:
        bootstrap = group_bootstrap_comparison(
            predictions[best_overall_full["key"]],
            predictions[best_single["key"]],
            targets,
            groups,
            seed=bootstrap_seed,
            samples=bootstrap_samples,
        )

    rows.sort(key=lambda row: (int(row["baseline_id"][1:]), row["variant"]))
    summary = {
        "schema_version": "gf-a1-baselines-1",
        "sample_count": len(samples),
        "split_counts": {
            split: int(np.sum([condition.split == split for condition in dataset.conditions]))
            for split in ("train", "val", "test")
        },
        "training_seed": int(training_seed),
        "model_selection_split": "val",
        "fit_group_count": len(train_group_ids),
        "models": rows,
        "best_single": {
            "key": best_single["key"],
            "validation_macro_RNMAE": best_single["validation"]["macro_RNMAE"],
        },
        "best_full_fusion": {
            "key": best_full["key"],
            "validation_macro_RNMAE": best_full["validation"]["macro_RNMAE"],
        },
        "best_overall_full_input": {
            "key": best_overall_full["key"],
            "validation_macro_RNMAE": best_overall_full["validation"]["macro_RNMAE"],
            "test_macro_RNMAE": best_overall_full["test"]["macro_RNMAE"],
        },
        "gate": gate,
        "bootstrap": bootstrap,
        "not_applicable": {
            "B8": "T=1 steady-state observations have no sequence dimension for temporal modeling"
        },
    }
    return BaselineSuiteResult(
        summary=summary,
        predictions=predictions,
        targets=targets,
        group_ids=tuple(str(group) for group in groups),
    )


def _feature_matrix(samples: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [
            [sample.signals[index][0, 0] for index in range(len(SENSOR_IDS))]
            for sample in samples
        ],
        dtype=np.float64,
    )


def _split_indices(dataset: A1Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    splits = [
        np.asarray(
            [index for index, condition in enumerate(dataset.conditions) if condition.split == split],
            dtype=np.int64,
        )
        for split in ("train", "val", "test")
    ]
    if any(len(indices) == 0 for indices in splits):
        raise ValueError("A1 baseline requires non-empty train, val, and test splits")
    return splits[0], splits[1], splits[2]


def _fit_single_sensor(
    *,
    algorithm: str,
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    candidates: list[tuple[dict[str, Any], Any, float]] = []
    if algorithm == "ridge":
        for alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=alpha)
            start = time.perf_counter()
            model.fit(features[train_indices], targets[train_indices])
            candidates.append(({"alpha": alpha}, model, time.perf_counter() - start))
    elif algorithm == "gbdt":
        for config in GBDT_CONFIGS:
            estimator = GradientBoostingRegressor(random_state=seed, **config)
            model = MultiOutputRegressor(estimator)
            start = time.perf_counter()
            model.fit(features[train_indices], targets[train_indices])
            candidates.append((dict(config), model, time.perf_counter() - start))
    else:
        raise ValueError(f"unsupported single-sensor algorithm: {algorithm}")
    selected_parameters, selected_model, training_time = min(
        candidates,
        key=lambda item: float(
            evaluate_predictions(
                targets,
                item[1].predict(features),
                np.arange(len(targets)).astype(str),
                val_indices,
            )["macro_RNMAE"]
        ),
    )
    start = time.perf_counter()
    prediction = np.asarray(selected_model.predict(features), dtype=np.float64)
    return {
        "parameters": selected_parameters,
        "prediction": prediction,
        "resources": {
            "training_time_s": float(training_time),
            "inference_time_s": float(time.perf_counter() - start),
            "parameter_count": _model_parameter_count(selected_model),
        },
    }


def _fit_full_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    del seed
    candidates = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha)
        start = time.perf_counter()
        model.fit(features[train_indices], targets[train_indices])
        prediction = np.asarray(model.predict(features), dtype=np.float64)
        candidates.append((alpha, prediction, model, time.perf_counter() - start))
    alpha, _, selected_model, training_time = min(
        candidates,
        key=lambda item: evaluate_predictions(
            targets,
            item[1],
            np.arange(len(targets)).astype(str),
            val_indices,
        )["macro_RNMAE"],
    )
    start = time.perf_counter()
    prediction = np.asarray(selected_model.predict(features), dtype=np.float64)
    return {
        "parameters": {"alpha": alpha},
        "prediction": prediction,
        "resources": {
            "training_time_s": float(training_time),
            "inference_time_s": float(time.perf_counter() - start),
            "parameter_count": _model_parameter_count(selected_model),
        },
    }


def _fit_full_gbdt(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    candidates = []
    for config in GBDT_CONFIGS:
        model = MultiOutputRegressor(
            GradientBoostingRegressor(random_state=seed, **config)
        )
        start = time.perf_counter()
        model.fit(features[train_indices], targets[train_indices])
        prediction = np.asarray(model.predict(features), dtype=np.float64)
        candidates.append((dict(config), prediction, model, time.perf_counter() - start))
    config, _, selected_model, training_time = min(
        candidates,
        key=lambda item: evaluate_predictions(
            targets,
            item[1],
            np.arange(len(targets)).astype(str),
            val_indices,
        )["macro_RNMAE"],
    )
    start = time.perf_counter()
    prediction = np.asarray(selected_model.predict(features), dtype=np.float64)
    return {
        "parameters": config,
        "prediction": prediction,
        "resources": {
            "training_time_s": float(training_time),
            "inference_time_s": float(time.perf_counter() - start),
            "parameter_count": _model_parameter_count(selected_model),
        },
    }


def _fit_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    model = MLPRegressor(
        hidden_layer_sizes=(32,),
        solver="lbfgs",
        alpha=1.0e-4,
        max_iter=1000,
        random_state=seed,
    )
    start = time.perf_counter()
    model.fit(features[train_indices], targets[train_indices] / TARGET_RANGES)
    training_time = time.perf_counter() - start
    start = time.perf_counter()
    prediction = np.asarray(model.predict(features) * TARGET_RANGES, dtype=np.float64)
    return {
        "prediction": prediction,
        "resources": {
            "training_time_s": float(training_time),
            "inference_time_s": float(time.perf_counter() - start),
            "parameter_count": _model_parameter_count(model),
        },
    }


def _make_row(
    *,
    baseline_id: str,
    variant: str,
    description: str,
    prediction: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    parameters: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> dict[str, Any]:
    if baseline_id == "B0" or variant in {"full", "equal_weight_late_fusion", "validation_inverse_error"}:
        key = baseline_id
    else:
        key = f"{baseline_id}__{variant}"
    return {
        "key": key,
        "baseline_id": baseline_id,
        "variant": variant,
        "description": description,
        "parameters": dict(parameters),
        "resources": dict(resources),
        "validation": evaluate_predictions(targets, prediction, groups, val_indices),
        "test": evaluate_predictions(targets, prediction, groups, test_indices),
    }


def _pilot_gate(
    best_single: Mapping[str, Any],
    full_candidates: Sequence[Mapping[str, Any]],
    targets: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    del targets
    best_full = min(full_candidates, key=lambda row: row["validation"]["macro_RNMAE"])
    single_value = float(best_single["validation"]["macro_RNMAE"])
    full_value = float(best_full["validation"]["macro_RNMAE"])
    relative_improvement = (single_value - full_value) / single_value
    component_improvements = {
        row["key"]: [
            float(best_single["validation"]["component_RNMAE"][index]
                  - row["validation"]["component_RNMAE"][index])
            for index in range(3)
        ]
        for row in (best_full,)
    }
    best_single_components = [
        row for row in rows if row["key"] == best_single["key"]
    ]
    checks = {
        "full_beats_best_single_by_5_percent": relative_improvement >= 0.05,
        "improvement_in_at_least_two_components": sum(
            value > 0.0 for value in component_improvements[best_full["key"]]
        ) >= 2,
        "best_single_row_present": len(best_single_components) == 1,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "best_single_key": best_single["key"],
        "best_full_key": best_full["key"],
        "best_single_validation_macro_RNMAE": single_value,
        "best_full_validation_macro_RNMAE": full_value,
        "relative_improvement": relative_improvement,
        "component_improvements": component_improvements,
        "checks": checks,
    }


def _model_parameter_count(model: Any) -> int:
    if isinstance(model, MultiOutputRegressor):
        return int(
            sum(
                tree.tree_.node_count
                for estimator in model.estimators_
                for tree in np.asarray(estimator.estimators_).ravel()
            )
        )
    if isinstance(model, Ridge):
        return int(model.coef_.size + model.intercept_.size)
    if isinstance(model, MLPRegressor):
        return int(
            sum(coef.size for coef in model.coefs_)
            + sum(intercept.size for intercept in model.intercepts_)
        )
    raise TypeError(f"cannot count parameters for model type {type(model).__name__}")


def _sum_resources(resources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not resources:
        raise ValueError("resource aggregation requires at least one model")
    return {
        "training_time_s": float(sum(float(item["training_time_s"]) for item in resources)),
        "inference_time_s": float(sum(float(item["inference_time_s"]) for item in resources)),
        "parameter_count": int(sum(int(item["parameter_count"]) for item in resources)),
    }
