"""Frozen C4 shared/private multiview preflight on the P3 pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..common.io import atomic_promote_directory, atomic_write_json, remove_owned_staging
from ..contract import ContractError
from .baseline import PilotMetadata, _labels, _partition_indices, deployment_features, load_pilot_metadata


VIEW_NAMES = ("ndir", "acoustic", "thermal", "context")


def _view_indices() -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {name: [] for name in VIEW_NAMES}
    channel_groups = {
        "ndir": (0, 1),
        "acoustic": (2, 3, 4, 5),
        "thermal": (6, 7),
    }
    for frame in range(7):
        frame_start = frame * 24
        for name, channels in channel_groups.items():
            for statistic_start in (0, 8, 16):
                groups[name].extend(frame_start + statistic_start + channel for channel in channels)
    groups["context"].extend(range(168, 177))
    return {name: np.asarray(indices, dtype=int) for name, indices in groups.items()}


def _orthogonal_projection(input_width: int, output_width: int, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"C4|{seed}|{input_width}|{output_width}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    matrix = rng.normal(size=(input_width, output_width))
    matrix /= np.maximum(np.linalg.norm(matrix, axis=0, keepdims=True), np.finfo(float).eps)
    return matrix


class FrozenEncoder:
    def __init__(self, method_id: str, config: dict[str, Any], seed: int):
        self.method_id = method_id
        self.config = config
        self.seed = seed
        self.scaler = StandardScaler()
        self.groups = _view_indices()
        self.projections: dict[str, np.ndarray] = {}

    def fit(self, features: np.ndarray) -> "FrozenEncoder":
        scaled = self.scaler.fit_transform(features)
        width = int(self.config["output_width"])
        if self.method_id == "early_fusion":
            self.projections["all"] = _orthogonal_projection(scaled.shape[1], width, self.seed)
        else:
            groups = self.groups
            if self.method_id == "random_grouping":
                rng = np.random.default_rng(self.seed)
                columns = rng.permutation(scaled.shape[1])
                sizes = [len(groups[name]) for name in VIEW_NAMES]
                offsets = np.cumsum([0, *sizes])
                groups = {name: columns[offsets[i] : offsets[i + 1]] for i, name in enumerate(VIEW_NAMES)}
                self.groups = groups
            shared_width = int(self.config["shared_width"])
            private_width = int(self.config["private_width_per_view"])
            for index, name in enumerate(VIEW_NAMES):
                input_width = len(groups[name])
                self.projections[f"shared:{name}"] = _orthogonal_projection(input_width, shared_width, self.seed + index)
                self.projections[f"private:{name}"] = _orthogonal_projection(input_width, private_width, self.seed + 100 + index)
        return self

    def transform_parts(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled = self.scaler.transform(features)
        if self.method_id == "early_fusion":
            encoded = np.tanh(scaled @ self.projections["all"])
            return encoded, np.empty((scaled.shape[0], 0))
        shared = np.mean(
            [np.tanh(scaled[:, self.groups[name]] @ self.projections[f"shared:{name}"]) for name in VIEW_NAMES],
            axis=0,
        )
        private = np.concatenate(
            [np.tanh(scaled[:, self.groups[name]] @ self.projections[f"private:{name}"]) for name in VIEW_NAMES],
            axis=1,
        )
        return shared, private

    def transform(self, features: np.ndarray) -> np.ndarray:
        shared, private = self.transform_parts(features)
        return np.concatenate([shared, private], axis=1)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    denominator = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
    return None if denominator == 0.0 else 1.0 - float(np.sum((y_pred - y_true) ** 2)) / denominator


def _paired_bootstrap_upper(values: dict[str, list[float]], config: dict[str, Any]) -> float:
    group_values = np.asarray([np.mean(item) for item in values.values()], dtype=float)
    if group_values.size < 2:
        raise ContractError("paired bootstrap requires at least two mixture groups")
    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    draws = rng.integers(0, group_values.size, size=(int(config["bootstrap_resamples"]), group_values.size))
    estimates = np.mean(group_values[draws], axis=1)
    return float(np.quantile(estimates, 0.5 + float(config["confidence"]) / 2.0))


def _masked_features(features: np.ndarray, train: np.ndarray, view: str) -> np.ndarray:
    masked = features.copy()
    columns = _view_indices()[view]
    masked[:, columns] = np.mean(features[train][:, columns], axis=0)
    return masked


def _nuisance_ood(features: np.ndarray, train: np.ndarray, test: np.ndarray, quantile: float) -> np.ndarray:
    context = _view_indices()["context"]
    mean = np.mean(features[train][:, context], axis=0)
    std = np.std(features[train][:, context], axis=0)
    if np.any(std == 0.0):
        raise ContractError("nuisance OOD train context has zero variance")
    distances = np.linalg.norm((features[test][:, context] - mean) / std, axis=1)
    threshold = float(np.quantile(distances, quantile))
    selected = np.flatnonzero(distances >= threshold)
    if selected.size < 2:
        raise ContractError("nuisance OOD selection is too small")
    return selected


def run_multiview(config: dict[str, Any], *, pilot_freeze: Path, output_dir: Path) -> dict[str, Any]:
    if config.get("plan_status") != "frozen_before_fit" or config.get("eligibility") != "eligible_for_P3_test":
        raise ContractError("C4 plan is not frozen and eligible")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"attempt directory already exists: {target}")
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        metadata = load_pilot_metadata(pilot_freeze)
        if metadata.generation_summary["plan_id"] != config["pilot_plan_id"]:
            raise ContractError("pilot plan mismatch")
        features, _ = deployment_features(metadata)
        rows: list[dict[str, Any]] = []
        paired: dict[str, dict[str, dict[str, list[float]]]] = {"masked": {}, "nuisance": {}}
        probe_r2: list[float] = []
        wrong_pairing_deltas: list[float] = []
        modality_contributions: list[dict[str, Any]] = []
        methods = ("early_fusion", "shared_private", "random_grouping")
        full_r2: dict[str, list[float]] = {name: [] for name in methods}
        for split_id in config["split_ids"]:
            partitions = _partition_indices(metadata, split_id)
            train, val, test = partitions["train"], partitions["val"], partitions["test"]
            y_train = _labels(metadata, train)
            y_val = _labels(metadata, val)
            for seed in config["seeds"]:
                fitted: dict[str, tuple[FrozenEncoder, Ridge]] = {}
                for method_id in methods:
                    encoder = FrozenEncoder(method_id, config["encoder"], int(seed)).fit(features[train])
                    model = Ridge(alpha=float(config["encoder"]["ridge_alpha"])).fit(encoder.transform(features[train]), y_train)
                    fitted[method_id] = (encoder, model)
                    rows.append(
                        {
                            "split_id": split_id,
                            "seed": int(seed),
                            "method_id": method_id,
                            "partition": "validation",
                            "mae": float(np.mean(np.abs(model.predict(encoder.transform(features[val])) - y_val))),
                        }
                    )
                # Test labels open only after all three fixed-capacity models are fit.
                y_test = _labels(metadata, test)
                masked = _masked_features(features, train, str(config["ood"]["masked_view"]))
                nuisance_local = _nuisance_ood(
                    features,
                    train,
                    test,
                    float(config["ood"]["nuisance_distance_quantile"]),
                )
                predictions: dict[str, dict[str, np.ndarray]] = {}
                for method_id, (encoder, model) in fitted.items():
                    prediction = model.predict(encoder.transform(features[test]))
                    masked_prediction = model.predict(encoder.transform(masked[test]))
                    predictions[method_id] = {"full": prediction, "masked": masked_prediction}
                    full_r2[method_id].append(_r2(y_test, prediction))
                    for ood_id, local, pred in (
                        ("full", np.arange(test.size), prediction),
                        ("masked", np.arange(test.size), masked_prediction),
                        ("nuisance", nuisance_local, prediction[nuisance_local]),
                    ):
                        truth = y_test[local]
                        for cell_id in sorted(set(metadata.cell_ids[test[local]])):
                            cell_local = np.flatnonzero(metadata.cell_ids[test[local]] == cell_id)
                            for component_index, component in enumerate(config["components"]):
                                component_truth = truth[cell_local, component_index : component_index + 1]
                                component_prediction = pred[cell_local, component_index : component_index + 1]
                                rows.append(
                                    {
                                        "split_id": split_id,
                                        "seed": int(seed),
                                        "method_id": method_id,
                                        "partition": "test",
                                        "ood_id": ood_id,
                                        "grid_cell_id": str(cell_id),
                                        "component": component,
                                        "mae": float(np.mean(np.abs(component_prediction - component_truth))),
                                        "r2": _r2(component_truth, component_prediction),
                                        "sample_count": int(cell_local.size),
                                    }
                                )
                shared_encoder, _ = fitted["shared_private"]
                shared_train, _ = shared_encoder.transform_parts(features[train])
                shared_test, _ = shared_encoder.transform_parts(features[test])
                probe = Ridge(alpha=1.0).fit(shared_train, y_train)
                probe_r2.append(_r2(y_test, probe.predict(shared_test)))
                rng = np.random.default_rng(int(seed))
                wrong = features[test].copy()
                for columns in _view_indices().values():
                    wrong[:, columns] = wrong[rng.permutation(test.size)][:, columns]
                candidate_encoder, candidate_model = fitted["shared_private"]
                normal_mae = np.mean(np.abs(predictions["shared_private"]["full"] - y_test), axis=1)
                wrong_mae = np.mean(np.abs(candidate_model.predict(candidate_encoder.transform(wrong)) - y_test), axis=1)
                wrong_pairing_deltas.append(float(np.mean(wrong_mae - normal_mae)))
                for view in VIEW_NAMES:
                    ablated = _masked_features(features, train, view)
                    ablated_mae = float(
                        np.mean(np.abs(candidate_model.predict(candidate_encoder.transform(ablated[test])) - y_test))
                    )
                    modality_contributions.append(
                        {"split_id": split_id, "seed": int(seed), "view": view, "mae_increase": ablated_mae - float(np.mean(normal_mae))}
                    )
                for ood_id, local in (("masked", np.arange(test.size)), ("nuisance", nuisance_local)):
                    baseline_errors = np.mean(np.abs(predictions["early_fusion"][ood_id if ood_id == "masked" else "full"][local] - y_test[local]), axis=1)
                    candidate_errors = np.mean(np.abs(predictions["shared_private"][ood_id if ood_id == "masked" else "full"][local] - y_test[local]), axis=1)
                    for offset, global_index in enumerate(test[local]):
                        mixture_id = str(metadata.mixture_ids[global_index])
                        cell_id = str(metadata.cell_ids[global_index])
                        relative = (candidate_errors[offset] - baseline_errors[offset]) / baseline_errors[offset]
                        paired[ood_id].setdefault(cell_id, {}).setdefault(mixture_id, []).append(float(relative))

        gate = config["gate"]
        ood_upper = {
            ood_id: {cell_id: _paired_bootstrap_upper(values, gate) for cell_id, values in cells.items()}
            for ood_id, cells in paired.items()
        }
        ood_pass = all(
            value <= -float(gate["minimum_ood_mae_reduction"])
            for cells in ood_upper.values()
            for value in cells.values()
        )
        r2_regression = float(np.mean(full_r2["early_fusion"]) - np.mean(full_r2["shared_private"]))
        r2_pass = r2_regression <= float(gate["maximum_full_modality_r2_regression"])
        probe_pass = float(np.mean(probe_r2)) >= float(gate["minimum_shared_probe_r2"])
        random_grouping_not_better = float(np.mean(full_r2["shared_private"])) > float(np.mean(full_r2["random_grouping"]))
        wrong_pairing_pass = float(np.mean(wrong_pairing_deltas)) > 0.0
        passed = ood_pass and r2_pass and probe_pass and random_grouping_not_better and wrong_pairing_pass
        result = {
            "schema_version": "gib-benchmark-1",
            "task_id": "P3-09",
            "task_status": "completed",
            "candidate_id": config["candidate_id"],
            "candidate_verdict": "enter_P4" if passed else "reject",
            "gate_checks": {
                "ood_relative_mae_ci_upper": ood_upper,
                "ood_pass": ood_pass,
                "full_modality_r2_regression": r2_regression,
                "full_modality_r2_pass": r2_pass,
                "mean_shared_probe_r2": float(np.mean(probe_r2)),
                "shared_probe_pass": probe_pass,
                "random_grouping_control_pass": random_grouping_not_better,
                "wrong_pairing_mean_mae_increase": float(np.mean(wrong_pairing_deltas)),
                "wrong_pairing_control_pass": wrong_pairing_pass,
            },
            "encoder_contract": {
                "output_width": int(config["encoder"]["output_width"]),
                "trainable_encoder_parameters": 0,
                "same_input_information": True,
            },
            "metric_rows": rows,
            "modality_contributions": modality_contributions,
            "claim_scope": config["claim_scope"],
            "next_allowed_task": "P3-13" if passed else "candidate_terminal_reject",
        }
        atomic_write_json(staging / "multiview_results.json", result)
        atomic_write_json(
            staging / "attempt_manifest.json",
            {
                "schema_version": "gib-benchmark-1",
                "attempt_id": target.name,
                "task_id": "P3-09",
                "status": "complete",
                "task_status": "completed",
                "candidate_verdict": result["candidate_verdict"],
                "claim_scope": result["claim_scope"],
                "next_allowed_task": result["next_allowed_task"],
            },
        )
        atomic_promote_directory(staging, target)
        return result
    except Exception:
        remove_owned_staging(staging)
        raise


__all__ = ["FrozenEncoder", "run_multiview"]
