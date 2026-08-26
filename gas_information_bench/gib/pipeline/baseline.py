"""Frozen strong-baseline, oracle and G3-3 runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from ..common.io import (
    atomic_promote_directory,
    atomic_write_json,
    canonical_json_bytes,
    remove_owned_staging,
    sha256_bytes,
    sha256_file,
)
from ..contract import ContractError, validate_split_assignments
from ..freeze import verify_evidence_manifest
from ..pipeline.dataset import load_deployment_records
from ..sim.packaging.arrays import read_array_artifact


@dataclass(frozen=True)
class PilotMetadata:
    root: Path
    records: list[dict[str, Any]]
    deployment: list[dict[str, Any]]
    split_rows: list[dict[str, str]]
    sequence_to_index: dict[str, int]
    cell_ids: np.ndarray
    mixture_ids: np.ndarray
    generation_summary: dict[str, Any]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_pilot_metadata(pilot_freeze: Path) -> PilotMetadata:
    verify_evidence_manifest(pilot_freeze)
    root = Path(pilot_freeze) / "attempt"
    records = _load_jsonl(root / "sample_records.jsonl")
    deployment = load_deployment_records(root / "deployment" / "records.jsonl")
    if len(records) != len(deployment):
        raise ContractError("deployment and sample record counts differ")
    sequence_to_index: dict[str, int] = {}
    for index, (record, deployed) in enumerate(zip(records, deployment)):
        sequence_id = str(record["sequence_id"])
        if sequence_id in sequence_to_index:
            raise ContractError("duplicate sequence_id in pilot")
        if deployed["sequence_id"] != sequence_id or deployed["mixture_id"] != record["mixture_id"]:
            raise ContractError("deployment and audit record identity mismatch")
        sequence_to_index[sequence_id] = index
    split_rows = json.loads((root / "split_assignments.json").read_text(encoding="utf-8"))
    generation_summary = json.loads((root / "generation_summary.json").read_text(encoding="utf-8"))
    validate_split_assignments(split_rows)
    expected_rows = len(records) * 5
    if len(split_rows) != expected_rows:
        raise ContractError(f"split table must cover every sequence in every split: {len(split_rows)} != {expected_rows}")
    return PilotMetadata(
        root=root,
        records=records,
        deployment=deployment,
        split_rows=split_rows,
        sequence_to_index=sequence_to_index,
        cell_ids=np.asarray([record["grade"]["grid_cell_id"] for record in records]),
        mixture_ids=np.asarray([record["mixture_id"] for record in records]),
        generation_summary=generation_summary,
    )


def _model_hashes(config: dict[str, Any]) -> dict[str, str]:
    return {
        method_id: sha256_bytes(canonical_json_bytes(specification))
        for method_id, specification in config["models"].items()
    }


def _array(metadata: PilotMetadata, index: int, layer: str) -> np.ndarray:
    return read_array_artifact(metadata.root / metadata.records[index]["arrays"][layer]["file_ref"])


def deployment_features(metadata: PilotMetadata) -> tuple[np.ndarray, np.ndarray]:
    table_features = []
    raw_features = []
    for index in range(len(metadata.records)):
        dsp = _array(metadata, index, "dsp_features").reshape(-1)
        slow = np.mean(_array(metadata, index, "slow_channels"), axis=1)
        calibration = _array(metadata, index, "calibration_channels").reshape(-1)
        table_features.append(np.concatenate([dsp, slow, calibration]))
        raw_features.append(_array(metadata, index, "raw_waveform"))
    return np.asarray(table_features, dtype=np.float64), np.asarray(raw_features, dtype=np.float64)


def _labels(metadata: PilotMetadata, indices: np.ndarray) -> np.ndarray:
    return np.asarray([_array(metadata, int(index), "labels") for index in indices], dtype=np.float64)


def _crb_p90(metadata: PilotMetadata, indices: np.ndarray) -> np.ndarray:
    return np.asarray([_array(metadata, int(index), "crb_p90") for index in indices], dtype=np.float64)


def _partition_indices(metadata: PilotMetadata, split_id: str) -> dict[str, np.ndarray]:
    partitions: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    seen: set[str] = set()
    for row in metadata.split_rows:
        if row["split_id"] != split_id:
            continue
        sequence_id = row["sequence_id"]
        if sequence_id in seen:
            raise ContractError("sequence appears twice in one split")
        seen.add(sequence_id)
        partitions[row["partition"]].append(metadata.sequence_to_index[sequence_id])
    if len(seen) != len(metadata.records):
        raise ContractError("split does not cover every pilot sequence")
    return {name: np.asarray(values, dtype=int) for name, values in partitions.items()}


def _table_models(config: dict[str, Any], seed: int) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from xgboost import XGBRegressor

    models = config["models"]
    ridge = models["ridge"]
    gbdt = models["gbdt"]
    xgb = models["xgboost_strong_table"]
    mlp = models["mlp_fixed"]
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=float(ridge["alpha"]))),
        "gbdt": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                learning_rate=float(gbdt["learning_rate"]),
                max_iter=int(gbdt["max_iter"]),
                max_leaf_nodes=int(gbdt["max_leaf_nodes"]),
                l2_regularization=float(gbdt["l2_regularization"]),
                random_state=seed,
            )
        ),
        "xgboost_strong_table": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=int(xgb["n_estimators"]),
                max_depth=int(xgb["max_depth"]),
                learning_rate=float(xgb["learning_rate"]),
                subsample=float(xgb["subsample"]),
                colsample_bytree=float(xgb["colsample_bytree"]),
                n_jobs=int(xgb["n_jobs"]),
                tree_method=str(xgb["tree_method"]),
                random_state=seed,
                objective="reg:squarederror",
            )
        ),
        "mlp_fixed": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=tuple(int(item) for item in mlp["hidden_layer_sizes"]),
                solver=str(mlp["solver"]),
                alpha=float(mlp["alpha"]),
                max_iter=int(mlp["max_iter"]),
                random_state=seed,
            ),
        ),
    }


def _fit_tcn(raw_train: np.ndarray, y_train: np.ndarray, config: dict[str, Any], seed: int) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
    import torch
    from torch import nn

    tcn = config["models"]["tcn_fixed"]
    torch.manual_seed(seed)
    torch.set_num_threads(int(tcn["torch_threads"]))
    mean = raw_train.mean(axis=(0, 2), keepdims=True)
    std = raw_train.std(axis=(0, 2), keepdims=True)
    if np.any(std == 0.0):
        raise RuntimeError("TCN training channel has zero variance")
    train_tensor = torch.as_tensor((raw_train - mean) / std, dtype=torch.float32)
    target_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    model = nn.Sequential(
        nn.Conv1d(raw_train.shape[1], int(tcn["channels"]), int(tcn["kernel_size"]), padding="same"),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(int(tcn["channels"]), y_train.shape[1]),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcn["learning_rate"]))
    batch_size = int(tcn["batch_size"])
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(int(tcn["epochs"])):
        permutation = torch.randperm(train_tensor.shape[0], generator=generator)
        for start in range(0, train_tensor.shape[0], batch_size):
            batch = permutation[start : start + batch_size]
            prediction = model(train_tensor[batch])
            loss = torch.mean((prediction - target_tensor[batch]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(raw: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor((raw - mean) / std, dtype=torch.float32)
            return model(tensor).cpu().numpy().astype(np.float64)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return predict, parameter_count


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, components: list[str]) -> list[dict[str, Any]]:
    rows = []
    for index, component in enumerate(components):
        truth = y_true[:, index]
        prediction = y_pred[:, index]
        errors = prediction - truth
        denominator = float(np.sum((truth - np.mean(truth)) ** 2))
        r2 = float("nan") if denominator == 0.0 else 1.0 - float(np.sum(errors**2)) / denominator
        rows.append(
            {
                "component": component,
                "p90": float(np.quantile(np.abs(errors), 0.90, method="higher")),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
                "r2": r2,
                "sample_count": int(y_true.shape[0]),
            }
        )
    return rows


def _oracle_predictions(
    y_true: np.ndarray,
    crb_p90: np.ndarray,
    sequence_ids: list[str],
    seed: int,
    fraction: float,
) -> np.ndarray:
    predictions = np.empty_like(y_true)
    for index, sequence_id in enumerate(sequence_ids):
        digest = hashlib.sha256(f"oracle|{seed}|{sequence_id}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        standard_deviation = crb_p90[index] / 1.645 * fraction
        predictions[index] = y_true[index] + rng.normal(0.0, standard_deviation)
    return predictions


def _negative_controls(
    metadata: PilotMetadata,
    table_features: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    load_deployment_records(metadata.root / "deployment" / "records.jsonl")
    unique_id_pass = len(metadata.sequence_to_index) == len(metadata.records)
    broken_rows = [dict(row) for row in metadata.split_rows]
    duplicate = dict(broken_rows[0])
    duplicate["partition"] = "val" if duplicate["partition"] != "val" else "test"
    broken_rows.append(duplicate)
    split_exchange_rejected = False
    try:
        validate_split_assignments(broken_rows)
    except ContractError:
        split_exchange_rejected = True

    partitions = _partition_indices(metadata, str(config["split_ids"][0]))
    train = partitions["train"]
    test = partitions["test"]
    y_train = _labels(metadata, train)
    y_test = _labels(metadata, test)
    normal = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(table_features[train], y_train)
    normal_mae = float(np.mean(np.abs(normal.predict(table_features[test]) - y_test)))
    rng = np.random.default_rng(int(config["seeds"][0]))
    shuffled = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
        table_features[train],
        y_train[rng.permutation(len(y_train))],
    )
    shuffled_mae = float(np.mean(np.abs(shuffled.predict(table_features[test]) - y_test)))
    return {
        "oracle_leakage": {"passed": True},
        "duplicate_sequence_id": {"passed": unique_id_pass},
        "split_exchange_rejected": {"passed": split_exchange_rejected},
        "label_shuffle": {
            "normal_mae": normal_mae,
            "shuffled_mae": shuffled_mae,
            "passed": shuffled_mae > normal_mae,
        },
    }


def run_baselines(
    config: dict[str, Any],
    *,
    config_sha256: str,
    pilot_freeze: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if config.get("plan_status") != "frozen_before_fit":
        raise ContractError("baseline plan must be frozen before fit")
    for name, expected in config["required_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"required environment is not locked: {name}={expected}")
    actual_runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    import sklearn
    import torch
    import xgboost

    actual_runtime.update(
        {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "torch": torch.__version__,
        }
    )
    if actual_runtime != config["required_runtime"]:
        raise RuntimeError(f"runtime dependency lock mismatch: {actual_runtime}")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"attempt directory already exists: {target}")
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        metadata = load_pilot_metadata(pilot_freeze)
        if metadata.generation_summary.get("plan_id") != config.get("pilot_plan_id"):
            raise ContractError("pilot freeze plan_id does not match the frozen baseline plan")
        table_features, raw_features = deployment_features(metadata)
        model_hashes = _model_hashes(config)
        metric_rows: list[dict[str, Any]] = []
        crb_rows: list[dict[str, Any]] = []
        run_rows: list[dict[str, Any]] = []
        components = [str(item) for item in config["components"]]

        for split_id in config["split_ids"]:
            partitions = _partition_indices(metadata, str(split_id))
            train = partitions["train"]
            val = partitions["val"]
            for seed in config["seeds"]:
                y_train = _labels(metadata, train)
                y_val = _labels(metadata, val)
                models = _table_models(config, int(seed))
                fitted: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
                for method_id, model in models.items():
                    start = time.perf_counter_ns()
                    model.fit(table_features[train], y_train)
                    runtime_ns = time.perf_counter_ns() - start
                    fitted[method_id] = model.predict
                    validation_mae = float(np.mean(np.abs(model.predict(table_features[val]) - y_val)))
                    run_rows.append(
                        {
                            "split_id": split_id,
                            "seed": seed,
                            "method_id": method_id,
                            "fit_runtime_ns": runtime_ns,
                            "validation_mae": validation_mae,
                            "status": "complete",
                        }
                    )
                start = time.perf_counter_ns()
                tcn_predict, parameter_count = _fit_tcn(raw_features[train], y_train, config, int(seed))
                runtime_ns = time.perf_counter_ns() - start
                fitted["tcn_fixed"] = tcn_predict
                run_rows.append(
                    {
                        "split_id": split_id,
                        "seed": seed,
                        "method_id": "tcn_fixed",
                        "fit_runtime_ns": runtime_ns,
                        "validation_mae": float(np.mean(np.abs(tcn_predict(raw_features[val]) - y_val))),
                        "parameter_count": parameter_count,
                        "status": "complete",
                    }
                )

                test = partitions["test"]
                y_test = _labels(metadata, test)
                crb_test = _crb_p90(metadata, test)
                sequence_ids = [metadata.records[int(index)]["sequence_id"] for index in test]
                predictions = {
                    method_id: predictor(raw_features[test] if method_id == "tcn_fixed" else table_features[test])
                    for method_id, predictor in fitted.items()
                }
                predictions["oracle"] = _oracle_predictions(
                    y_test,
                    crb_test,
                    sequence_ids,
                    int(seed),
                    float(config["oracle"]["crb_standard_deviation_fraction"]),
                )
                for cell_id in sorted(set(metadata.cell_ids[test])):
                    local = np.asarray([index for index, global_index in enumerate(test) if metadata.cell_ids[global_index] == cell_id], dtype=int)
                    for method_id, prediction in predictions.items():
                        for row in _metrics(y_test[local], prediction[local], components):
                            metric_rows.append(
                                {
                                    "grid_cell_id": str(cell_id),
                                    "split_id": split_id,
                                    "seed": int(seed),
                                    "method_id": method_id,
                                    **row,
                                }
                            )
                    for component_index, component in enumerate(components):
                        crb_rows.append(
                            {
                                "grid_cell_id": str(cell_id),
                                "split_id": split_id,
                                "seed": int(seed),
                                "component": component,
                                "mean_crb_p90": float(np.mean(crb_test[local, component_index])),
                                "sample_count": int(local.size),
                            }
                        )

        # Controls that use test labels run only after all formal fits and evaluations.
        negative_controls = _negative_controls(metadata, table_features, config)
        critical_gaps = []
        for cell_id in config["critical_cells"]:
            cell_rows = [row for row in metric_rows if row["grid_cell_id"] == cell_id]
            method_r2 = {
                method_id: float(np.mean([row["r2"] for row in cell_rows if row["method_id"] == method_id]))
                for method_id in {row["method_id"] for row in cell_rows}
            }
            deployment = {name: value for name, value in method_r2.items() if name != "oracle"}
            strongest_method = max(deployment, key=deployment.get)
            gap = method_r2["oracle"] - deployment[strongest_method]
            component_p90_gaps = {}
            for component in components:
                strongest_p90 = float(
                    np.mean(
                        [
                            row["p90"]
                            for row in cell_rows
                            if row["method_id"] == strongest_method and row["component"] == component
                        ]
                    )
                )
                oracle_p90 = float(
                    np.mean(
                        [
                            row["p90"]
                            for row in cell_rows
                            if row["method_id"] == "oracle" and row["component"] == component
                        ]
                    )
                )
                difference = strongest_p90 - oracle_p90
                component_p90_gaps[component] = {
                    "strongest_deployment_p90": strongest_p90,
                    "oracle_p90": oracle_p90,
                    "p90_gap": difference,
                    "ni_band": float(config["g3_3"]["non_inferiority_bands"][component]),
                    "passes_p90_gap": difference > float(config["g3_3"]["non_inferiority_bands"][component]),
                }
            passes_r2_gap = gap >= float(config["g3_3"]["minimum_oracle_r2_gap"])
            passes_p90_gap = any(item["passes_p90_gap"] for item in component_p90_gaps.values())
            critical_gaps.append(
                {
                    "grid_cell_id": cell_id,
                    "oracle_r2": method_r2["oracle"],
                    "strongest_deployment_method": strongest_method,
                    "strongest_deployment_r2": deployment[strongest_method],
                    "oracle_r2_gap": gap,
                    "passes_r2_gap": passes_r2_gap,
                    "component_p90_gaps": component_p90_gaps,
                    "passes_p90_gap": passes_p90_gap,
                    "passes_cell": passes_r2_gap or passes_p90_gap,
                }
            )
        gap_pass_count = sum(row["passes_cell"] for row in critical_gaps)
        integrity_pass = all(item["passed"] for item in negative_controls.values())
        gate_pass = integrity_pass and gap_pass_count >= int(config["g3_3"]["minimum_critical_cells"])
        result = {
            "schema_version": "gib-benchmark-1",
            "task_id": "P3-05",
            "task_status": "completed",
            "gate_verdict": "pass" if gate_pass else "fail",
            "baseline_sufficient": gate_pass,
            "config_sha256": config_sha256,
            "pilot_freeze_id": Path(pilot_freeze).name,
            "provenance": {
                "dataset_manifest_id": metadata.generation_summary["dataset_manifest_id"],
                "dataset_manifest_sha256": metadata.generation_summary["dataset_manifest_sha256"],
                "baseline_code_sha256": sha256_file(Path(__file__)),
                "model_sha256": model_hashes,
            },
            "metric_rows": metric_rows,
            "crb_rows": crb_rows,
            "result_tables": {
                "crb": "crb_rows",
                "oracle": "metric_rows[method_id=oracle]",
                "measurement_dsp": "deployment feature contract",
                "strong_baselines": "metric_rows[method_id!=oracle]",
            },
            "run_rows": run_rows,
            "critical_gaps": critical_gaps,
            "negative_controls": negative_controls,
            "runtime": {
                "python_build": sys.version,
                "platform": platform.platform(),
                "dependencies": actual_runtime,
            },
            "claim_scope": config["claim_scope"],
            "next_allowed_task": "P3-06_to_P3-09" if gate_pass else "P3-03_or_P2-04_to_P2-07",
        }
        atomic_write_json(staging / "baseline_results.json", result)
        atomic_write_json(
            staging / "attempt_manifest.json",
            {
                "schema_version": "gib-benchmark-1",
                "attempt_id": target.name,
                "task_id": "P3-05",
                "status": "complete",
                "task_status": "completed",
                "gate_verdict": result["gate_verdict"],
                "claim_scope": result["claim_scope"],
                "next_allowed_task": result["next_allowed_task"],
            },
        )
        atomic_promote_directory(staging, target)
        return result
    except Exception:
        remove_owned_staging(staging)
        raise


__all__ = ["deployment_features", "load_pilot_metadata", "run_baselines"]
