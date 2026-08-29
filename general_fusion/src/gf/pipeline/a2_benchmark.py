from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from pathlib import Path
import re
import subprocess
from typing import Any

from gf.dl.evaluation import evaluate_output_constraints, evaluate_predictions
from gf.dl.evaluation import group_bootstrap_comparison
from gf.dl.training import (
    TorchTrainingConfig,
    TorchConcatMLP,
    build_a2_model_from_config,
    parameter_parity_report,
    prepare_a2_train_val_samples,
    train_torch_model,
)
from gf.sim.a1_dataset import load_dataset
from gf.dl.residual import apply_residual_learner, fit_residual_learner, residual_targets
from gf.ml.oof import build_grouped_fold_manifest, generate_grouped_oof_predictions


A1_DATA_SCHEMA_VERSION = "gf-a1-data-1"
A1_DATA_VERSION = "gf-a1-v1-20260827-r1"
A1_CONTENT_SHA256 = "310f9471f99c704fdeb68b846c5d8b351d701bbd7f6cef172a08ceccc77110c4"
A1_DATASET_ID = "ar_he_co2"
A1_SPLIT_SEED = 20260827
A1_SAMPLE_COUNT = 1200
A1_SPLIT_COUNTS = {"train": 840, "val": 180, "test": 180}
A2_EVAL_SCHEMA_VERSION = "gf-a2-eval-1"
A2_TRAIN_SCHEMA_VERSION = "gf-a2-train-1"
A2_EXPERIMENT_SCHEMA_VERSION = "gf-a2-experiment-1"
A2_RUN_MANIFEST_SCHEMA_VERSION = "gf-a2-run-manifest-1"
A2_PROTOCOL_SCHEMA_VERSION = "gf-a2-protocol-1"
A2_FORMAL_STATUS = "FROZEN"
A2_ALLOWED_SELECTION_SPLITS = ("train", "inner_oof", "val")
A2_TRAINING_SEEDS = (17, 29, 43, 71, 101)
TARGET_NAMES = ("x_Ar_pct", "x_He_pct", "x_CO2_pct")
FORBIDDEN_KEYS = frozenset(
    {"base_condition_id", "noise_seed_index", "noise_seed", "sequence_id"}
)
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class A2ProtocolError(ValueError):
    """Raised when an A2 contract or provenance invariant is violated."""


class TestUnlockError(A2ProtocolError):
    """Raised when formal test evidence is incomplete or inconsistent."""

    __test__ = False


def run_a2(
    *,
    project_root: str | Path | None = None,
    mode: str = "protocol",
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    experiment_config_paths: Sequence[str | Path] | None = None,
    unlock_test: bool = False,
    selection_record_path: str | Path | None = None,
    selected_checkpoint_path: str | Path | None = None,
    formal_run_status: str | None = None,
    max_epochs_override: int | None = None,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    """Run one explicit A2 protocol, validation, or diagnostic entry point."""

    if mode == "smoke":
        return run_a2_smoke(
            project_root=project_root,
            max_epochs=3 if max_epochs_override is None else max_epochs_override,
        )
    if mode == "torch-concat":
        return run_a2_torch_concat_validation(
            project_root=project_root,
            max_epochs_override=max_epochs_override,
        )
    if mode == "head-audit":
        return run_a2_validation(
            project_root=project_root,
            stage="head_audit",
            max_epochs_override=max_epochs_override,
        )
    if mode == "deepsets":
        return run_a2_validation(
            project_root=project_root,
            stage="deepsets",
            max_epochs_override=max_epochs_override,
        )
    if mode == "oof":
        return run_a2_oof_diagnostic(
            project_root=project_root,
            bootstrap_samples=bootstrap_samples,
        )
    if mode != "protocol":
        raise NotImplementedError(
            "supported A2 modes are 'protocol', 'smoke', 'torch-concat', 'head-audit', 'deepsets', and 'oof'"
        )
    return run_a2_protocol(
        project_root=project_root,
        eval_config_path=eval_config_path,
        train_config_path=train_config_path,
        experiment_config_paths=experiment_config_paths,
        unlock_test=unlock_test,
        selection_record_path=selection_record_path,
        selected_checkpoint_path=selected_checkpoint_path,
        formal_run_status=formal_run_status,
    )


def run_a2_smoke(
    *,
    project_root: str | Path | None = None,
    max_epochs: int = 3,
) -> dict[str, Any]:
    """Run one train/val-only smoke pass for the matched A2 model interfaces."""

    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    root = (Path(project_root) if project_root is not None else _default_project_root()).resolve()
    protocol = run_a2_protocol(project_root=root)
    eval_path = _resolve_file(root, root / "configs" / "eval" / "a2_eval.json")
    train_path = _resolve_file(root, root / "configs" / "train" / "a2_train.json")
    deepsets_path = _resolve_file(root, root / "configs" / "model" / "a2_deepsets.json")
    concat_path = _resolve_file(root, root / "configs" / "model" / "a2_concat.json")
    train_config = _read_json_object(train_path)
    deepsets_config = _read_json_object(deepsets_path)
    concat_config = _read_json_object(concat_path)
    dataset_dir = root / "data" / "a1_formal"
    dataset = load_dataset(dataset_dir)
    train_samples, validation_samples, scaler = prepare_a2_train_val_samples(dataset.samples())
    training_config = TorchTrainingConfig(
        max_epochs=max_epochs,
        patience=min(2, max_epochs),
        learning_rate=float(train_config["optimizer"]["learning_rate"]),
        weight_decay=float(train_config["optimizer"]["weight_decay"]),
        target_scale=tuple(float(value) for value in train_config["loss"]["target_scale"]),
    )
    model_results: dict[str, Any] = {}
    prediction_parts: list[np.ndarray] = []
    for model_name, model_config in (("C1", concat_config), ("M1", deepsets_config)):
        torch_model = build_a2_model_from_config(
            model_config,
            train_config,
            capacity_name="small",
            head_id="H0",
        )
        fit = train_torch_model(
            torch_model,
            train_samples,
            validation_samples,
            config=training_config,
            seed=17,
        )
        groups = tuple(sample.group_id for sample in validation_samples)
        targets = np.vstack([sample.target for sample in validation_samples]).astype(np.float64)
        metrics = evaluate_predictions(
            targets,
            fit.validation_predictions,
            groups,
            np.arange(len(validation_samples), dtype=np.int64),
        )
        model_results[model_name] = {
            "best_epoch": fit.best_epoch,
            "validation": metrics,
            "parameter_count": int(sum(parameter.numel() for parameter in torch_model.parameters())),
        }
        prediction_parts.append(fit.validation_predictions)

    digest = hashlib.sha256()
    for prediction in prediction_parts:
        array = np.asarray(prediction, dtype=np.float64, order="C")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    config_paths = {
        "eval_config": eval_path,
        "train_config": train_path,
        "model.C1": concat_path,
        "model.M1": deepsets_path,
    }
    run_dir = root / "outputs" / "runs" / "a2" / "a2-3-smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = build_run_manifest(
        project_root=root,
        stage="A2-3-smoke",
        config_paths=config_paths,
        data_manifest_path=root / "data" / "a1_formal" / "manifest.json",
        exit_code=0,
        prediction_hash=digest.hexdigest(),
        status="PASS",
        test_unlocked=False,
    )
    run_manifest["scaler_fitted_group_count"] = len(scaler.fitted_group_ids)
    run_manifest["models"] = model_results
    _write_json(run_dir / "manifest.json", run_manifest)
    return {
        "stage": "A2-3-smoke",
        "status": "PASS",
        "protocol": protocol,
        "models": model_results,
        "manifest": run_manifest,
    }


def run_a2_validation(
    *,
    project_root: str | Path | None = None,
    stage: str,
    capacity_name: str = "small",
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    """Run a train/val-only five-seed A2 validation stage.

    ``max_epochs_override`` is intended only for a smoke test and is recorded in
    the result. The default uses the frozen training configuration.
    """

    if stage not in {"head_audit", "deepsets"}:
        raise ValueError("stage must be head_audit or deepsets")
    if max_epochs_override is not None and max_epochs_override <= 0:
        raise ValueError("max_epochs_override must be positive")
    root = (Path(project_root) if project_root is not None else _default_project_root()).resolve()
    run_a2_protocol(project_root=root)
    eval_path = _resolve_file(root, root / "configs" / "eval" / "a2_eval.json")
    train_path = _resolve_file(root, root / "configs" / "train" / "a2_train.json")
    experiment_filename = "a2_head_audit.json" if stage == "head_audit" else "a2_deepsets.json"
    experiment_path = _resolve_file(root, root / "configs" / "experiment" / experiment_filename)
    experiment = _read_json_object(experiment_path)
    validate_a2_experiment_config(experiment)
    train_config = _read_json_object(train_path)
    if stage == "head_audit":
        model_paths = {
            "H0": _resolve_file(root, root / "configs" / "model" / "a2_concat.json"),
            "H1": _resolve_file(root, root / "configs" / "model" / "a2_concat.json"),
            "H2": _resolve_file(root, root / "configs" / "model" / "a2_concat.json"),
        }
        model_configs = {head: _read_json_object(path) for head, path in model_paths.items()}
        candidate_names = ("H0", "H1", "H2")
    else:
        model_paths = {
            "C1": _resolve_file(root, root / "configs" / "model" / "a2_concat.json"),
            "M1": _resolve_file(root, root / "configs" / "model" / "a2_deepsets.json"),
        }
        model_configs = {name: _read_json_object(path) for name, path in model_paths.items()}
        candidate_names = ("C1", "M1")

    dataset = load_dataset(root / "data" / "a1_formal")
    train_samples, validation_samples, scaler = prepare_a2_train_val_samples(dataset.samples())
    if max_epochs_override is None:
        training_config = TorchTrainingConfig.from_mapping(train_config)
        execution_mode = "frozen"
        run_name = stage
    else:
        training_config = TorchTrainingConfig(
            max_epochs=max_epochs_override,
            patience=min(2, max_epochs_override),
            learning_rate=float(train_config["optimizer"]["learning_rate"]),
            weight_decay=float(train_config["optimizer"]["weight_decay"]),
            target_scale=tuple(float(value) for value in train_config["loss"]["target_scale"]),
        )
        execution_mode = "smoke_override"
        run_name = f"{stage}-smoke"

    targets = np.vstack([sample.target for sample in validation_samples]).astype(np.float64)
    groups = tuple(sample.group_id for sample in validation_samples)
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in candidate_names}
    all_predictions: list[np.ndarray] = []
    for name in candidate_names:
        for seed in A2_TRAINING_SEEDS:
            torch.manual_seed(seed)
            model = build_a2_model_from_config(
                model_configs[name],
                train_config,
                capacity_name=capacity_name,
                head_id=(
                    name
                    if stage == "head_audit"
                    else str(model_configs[name].get("head_id", "H0"))
                ),
            )
            checkpoint_path = (
                root
                / "outputs"
                / "runs"
                / "a2"
                / run_name
                / f"{name.lower()}_seed_{seed}.pt"
            )
            fit = train_torch_model(
                model,
                train_samples,
                validation_samples,
                config=training_config,
                seed=seed,
                checkpoint_path=checkpoint_path,
            )
            metrics = evaluate_predictions(
                targets,
                fit.validation_predictions,
                groups,
                np.arange(len(validation_samples), dtype=np.int64),
            )
            diagnostics = evaluate_output_constraints(
                fit.validation_predictions,
                targets=targets,
            )
            records[name].append(
                {
                    "seed": int(seed),
                    "best_epoch": fit.best_epoch,
                    "epochs_completed": fit.epochs_completed,
                    "validation": metrics,
                    "output_constraints": diagnostics,
                    "checkpoint": _relative_path(root, checkpoint_path),
                }
            )
            all_predictions.append(fit.validation_predictions)

    summaries = {
        name: _summarize_seed_records(values)
        for name, values in records.items()
    }
    if stage == "head_audit":
        gate = _head_audit_gate(summaries)
        parity = None
    else:
        parity = _model_parity_for_capacity(
            model_configs["C1"],
            model_configs["M1"],
            train_config,
            capacity_name=capacity_name,
        )
        gate = _deepsets_gate(summaries, parity)

    prediction_hash = _prediction_array_hash(all_predictions)
    config_paths = {
        "eval_config": eval_path,
        "train_config": train_path,
        "experiment_config": experiment_path,
        **{f"model.{name}": path for name, path in model_paths.items()},
    }
    run_dir = root / "outputs" / "runs" / "a2" / run_name
    summary_dir = root / "outputs" / "summary" / "a2"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = build_run_manifest(
        project_root=root,
        stage=f"A2-{experiment['stage'].split('-')[-1]}-{stage}",
        config_paths=config_paths,
        data_manifest_path=root / "data" / "a1_formal" / "manifest.json",
        exit_code=0,
        prediction_hash=prediction_hash,
        status="EXECUTED",
        test_unlocked=False,
    )
    run_manifest["execution_mode"] = execution_mode
    run_manifest["scaler_fitted_group_count"] = len(scaler.fitted_group_ids)
    _write_json(run_dir / "manifest.json", run_manifest)
    result = {
        "stage": stage,
        "status": "EXECUTED",
        "execution_mode": execution_mode,
        "capacity_name": capacity_name,
        "seed_order": list(A2_TRAINING_SEEDS),
        "models": summaries,
        "gate": gate,
        "parameter_parity": parity,
        "manifest": run_manifest,
    }
    summary_name = f"a2_{stage}.json" if execution_mode == "frozen" else f"a2_{stage}_smoke.json"
    _write_json(summary_dir / summary_name, result)
    return result


def run_a2_torch_concat_validation(
    *,
    project_root: str | Path | None = None,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    """Reproduce the A1 B5 input with the shared Torch train/val contract."""

    if max_epochs_override is not None and max_epochs_override <= 0:
        raise ValueError("max_epochs_override must be positive")
    root = (Path(project_root) if project_root is not None else _default_project_root()).resolve()
    run_a2_protocol(project_root=root)
    eval_path = _resolve_file(root, root / "configs" / "eval" / "a2_eval.json")
    train_path = _resolve_file(root, root / "configs" / "train" / "a2_train.json")
    model_path = _resolve_file(root, root / "configs" / "model" / "a2_concat.json")
    eval_config = _read_json_object(eval_path)
    train_config = _read_json_object(train_path)
    model_config = _read_json_object(model_path)
    dataset = load_dataset(root / "data" / "a1_formal")
    train_samples, validation_samples, scaler = prepare_a2_train_val_samples(dataset.samples())
    if max_epochs_override is None:
        training_config = TorchTrainingConfig.from_mapping(train_config)
        execution_mode = "frozen"
        run_name = "torch-concat"
    else:
        training_config = TorchTrainingConfig(
            max_epochs=max_epochs_override,
            patience=min(2, max_epochs_override),
            learning_rate=float(train_config["optimizer"]["learning_rate"]),
            weight_decay=float(train_config["optimizer"]["weight_decay"]),
            target_scale=tuple(float(value) for value in train_config["loss"]["target_scale"]),
        )
        execution_mode = "smoke_override"
        run_name = "torch-concat-smoke"
    strong_baseline = model_config.get("strong_torch_baseline")
    if not isinstance(strong_baseline, Mapping):
        raise ValueError("a2_concat strong_torch_baseline is required")
    hidden_dim = strong_baseline.get("hidden_dim")
    if not isinstance(hidden_dim, int) or isinstance(hidden_dim, bool) or hidden_dim <= 0:
        raise ValueError("strong_torch_baseline.hidden_dim must be a positive integer")

    targets = np.vstack([sample.target for sample in validation_samples]).astype(np.float64)
    groups = tuple(sample.group_id for sample in validation_samples)
    seed_records: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    for seed in A2_TRAINING_SEEDS:
        torch.manual_seed(seed)
        model = TorchConcatMLP(
            sensor_count=len(model_config["sensor_ids"]),
            hidden_dim=hidden_dim,
            output_dim=len(targets[0]),
        )
        checkpoint_path = root / "outputs" / "runs" / "a2" / run_name / f"seed_{seed}.pt"
        fit = train_torch_model(
            model,
            train_samples,
            validation_samples,
            config=training_config,
            seed=seed,
            checkpoint_path=checkpoint_path,
        )
        metrics = evaluate_predictions(
            targets,
            fit.validation_predictions,
            groups,
            np.arange(len(validation_samples), dtype=np.int64),
        )
        seed_records.append(
            {
                "seed": int(seed),
                "best_epoch": fit.best_epoch,
                "epochs_completed": fit.epochs_completed,
                "validation": metrics,
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "checkpoint": _relative_path(root, checkpoint_path),
            }
        )
        predictions.append(fit.validation_predictions)

    summary = _summarize_seed_records(seed_records)
    reference = float(eval_config["baseline"]["a1_validation_macro_RNMAE"])
    allowed_gap = float(eval_config["baseline"]["same_order_relative_gap"])
    relative_gap = (summary["mean_validation_macro_RNMAE"] - reference) / reference
    gate = {
        "status": "PASS" if relative_gap <= allowed_gap else "FAIL",
        "reference": "A1 B5 independent-seed validation mean",
        "reference_validation_macro_RNMAE": reference,
        "torch_concat_mean_validation_macro_RNMAE": summary["mean_validation_macro_RNMAE"],
        "relative_gap": relative_gap,
        "allowed_relative_gap": allowed_gap,
        "same_order": relative_gap <= allowed_gap,
    }
    config_paths = {
        "eval_config": eval_path,
        "train_config": train_path,
        "model_config": model_path,
    }
    run_dir = root / "outputs" / "runs" / "a2" / run_name
    summary_dir = root / "outputs" / "summary" / "a2"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = build_run_manifest(
        project_root=root,
        stage="A2-1-torch-concat",
        config_paths=config_paths,
        data_manifest_path=root / "data" / "a1_formal" / "manifest.json",
        exit_code=0,
        prediction_hash=_prediction_array_hash(predictions),
        status="EXECUTED",
        test_unlocked=False,
    )
    run_manifest["execution_mode"] = execution_mode
    run_manifest["scaler_fitted_group_count"] = len(scaler.fitted_group_ids)
    result = {
        "stage": "A2-1-torch-concat",
        "status": "EXECUTED",
        "execution_mode": execution_mode,
        "seed_order": list(A2_TRAINING_SEEDS),
        "model": summary,
        "gate": gate,
        "manifest": run_manifest,
    }
    _write_json(run_dir / "manifest.json", run_manifest)
    summary_name = "a2_torch_concat.json" if execution_mode == "frozen" else "a2_torch_concat_smoke.json"
    _write_json(summary_dir / summary_name, result)
    return result


def run_a2_oof_diagnostic(
    *,
    project_root: str | Path | None = None,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    """Diagnose train-only B5/GBDT residual structure with grouped OOF."""

    root = (Path(project_root) if project_root is not None else _default_project_root()).resolve()
    run_a2_protocol(project_root=root)
    eval_path = _resolve_file(root, root / "configs" / "eval" / "a2_eval.json")
    train_path = _resolve_file(root, root / "configs" / "train" / "a2_train.json")
    oof_path = _resolve_file(root, root / "configs" / "model" / "a2_oof_residual.json")
    concat_path = _resolve_file(root, root / "configs" / "model" / "a2_concat.json")
    eval_config = _read_json_object(eval_path)
    dataset = load_dataset(root / "data" / "a1_formal")
    train_samples, validation_samples, scaler = prepare_a2_train_val_samples(dataset.samples())
    train_features = _scalar_feature_matrix(train_samples)
    validation_features = _scalar_feature_matrix(validation_samples)
    train_targets = np.vstack([sample.target for sample in train_samples]).astype(np.float64)
    validation_targets = np.vstack([sample.target for sample in validation_samples]).astype(np.float64)
    train_groups = tuple(sample.group_id for sample in train_samples)
    train_families = tuple(str(sample.metadata["condition_family"]) for sample in train_samples)
    validation_groups = tuple(sample.group_id for sample in validation_samples)
    fold_manifest = build_grouped_fold_manifest(
        train_groups,
        train_families,
        n_splits=5,
        seed=int(eval_config["bootstrap_seed"]),
    )
    oof_config_hash = sha256_file(oof_path)
    base_factories = {
        "B5": lambda seed: _TargetScaledMLP(seed),
        "B4": lambda seed: MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=2,
                random_state=seed,
            )
        ),
    }
    residual_results: dict[str, Any] = {}
    validation_predictions: dict[str, dict[str, np.ndarray]] = {}
    all_prediction_arrays: list[np.ndarray] = []
    sample_count = int(eval_config["bootstrap_samples"] if bootstrap_samples is None else bootstrap_samples)
    if sample_count <= 0:
        raise ValueError("bootstrap_samples must be positive")
    execution_mode = "frozen" if bootstrap_samples is None else "smoke_override"
    run_name = "oof" if execution_mode == "frozen" else "oof-smoke"
    for base_name, factory in base_factories.items():
        base_oof = generate_grouped_oof_predictions(
            train_features,
            train_targets,
            train_groups,
            train_families,
            estimator_factory=factory,
            n_splits=5,
            seed=int(eval_config["bootstrap_seed"]),
            model_config_hash=sha256_file(concat_path),
            fold_manifest=fold_manifest,
        )
        residual = residual_targets(train_targets, base_oof.predictions)
        residual_oof = generate_grouped_oof_predictions(
            train_features,
            residual,
            train_groups,
            train_families,
            estimator_factory=lambda seed: Ridge(alpha=1.0),
            n_splits=5,
            seed=int(eval_config["bootstrap_seed"]),
            model_config_hash=oof_config_hash,
            fold_manifest=fold_manifest,
        )
        oof_explained_variance = _explained_variance_by_component(residual, residual_oof.predictions)
        base_model = factory(int(eval_config["bootstrap_seed"]))
        base_model.fit(train_features, train_targets)
        base_validation = np.asarray(base_model.predict(validation_features), dtype=np.float64)
        residual_model = fit_residual_learner(
            "ridge_residual",
            train_features,
            train_targets,
            base_oof.predictions,
            ridge_alpha=1.0,
        )
        corrected_validation = apply_residual_learner(
            base_validation,
            residual_model,
            validation_features,
        )
        base_metrics = evaluate_predictions(
            validation_targets,
            base_validation,
            validation_groups,
            np.arange(len(validation_samples), dtype=np.int64),
        )
        corrected_metrics = evaluate_predictions(
            validation_targets,
            corrected_validation,
            validation_groups,
            np.arange(len(validation_samples), dtype=np.int64),
        )
        bootstrap = group_bootstrap_comparison(
            corrected_validation,
            base_validation,
            validation_targets,
            validation_groups,
            seed=int(eval_config["bootstrap_seed"]),
            samples=sample_count,
        )
        relative_improvement = (
            float(base_metrics["macro_RNMAE"] - corrected_metrics["macro_RNMAE"])
            / float(base_metrics["macro_RNMAE"])
        )
        component_delta = [
            float(corrected - base)
            for corrected, base in zip(
                corrected_metrics["component_RNMAE"],
                base_metrics["component_RNMAE"],
                strict=True,
            )
        ]
        residual_results[base_name] = {
            "base_validation": base_metrics,
            "corrected_validation": corrected_metrics,
            "relative_improvement": relative_improvement,
            "component_delta_corrected_minus_base": component_delta,
            "oof_explained_variance": oof_explained_variance,
            "bootstrap": bootstrap,
            "provenance_rows": len(base_oof.provenance) + len(residual_oof.provenance),
            "fold_manifest": base_oof.fold_manifest,
        }
        validation_predictions[base_name] = {
            "base": base_validation,
            "corrected": corrected_validation,
        }
        all_prediction_arrays.extend([base_oof.predictions, residual_oof.predictions, corrected_validation])

    reference_metrics = residual_results["B5"]["base_validation"]
    reference_prediction = validation_predictions["B5"]["base"]
    activation: dict[str, dict[str, Any]] = {}
    for base_name, payload in residual_results.items():
        candidate_prediction = validation_predictions[base_name]["corrected"]
        candidate_metrics = evaluate_predictions(
            validation_targets,
            candidate_prediction,
            validation_groups,
            np.arange(len(validation_samples), dtype=np.int64),
        )
        candidate_bootstrap = group_bootstrap_comparison(
            candidate_prediction,
            reference_prediction,
            validation_targets,
            validation_groups,
            seed=int(eval_config["bootstrap_seed"]),
            samples=sample_count,
        )
        activation[base_name] = _residual_activation_gate(
            payload,
            reference_metrics=reference_metrics,
            candidate_metrics=candidate_metrics,
            bootstrap=candidate_bootstrap,
        )
    config_paths = {
        "eval_config": eval_path,
        "train_config": train_path,
        "oof_config": oof_path,
        "concat_config": concat_path,
    }
    run_dir = root / "outputs" / "runs" / "a2" / run_name
    summary_dir = root / "outputs" / "summary" / "a2"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "fold_manifest.json", fold_manifest)
    _write_json(
        run_dir / "provenance.json",
        {
            base_name: {
                "fold_manifest": payload["fold_manifest"],
                "provenance_rows": payload["provenance_rows"],
            }
            for base_name, payload in residual_results.items()
        },
    )
    run_manifest = build_run_manifest(
        project_root=root,
        stage="A2-4-oof",
        config_paths=config_paths,
        data_manifest_path=root / "data" / "a1_formal" / "manifest.json",
        exit_code=0,
        prediction_hash=_prediction_array_hash(all_prediction_arrays),
        status="EXECUTED",
        test_unlocked=False,
    )
    run_manifest["execution_mode"] = execution_mode
    run_manifest["scaler_fitted_group_count"] = len(scaler.fitted_group_ids)
    result = {
        "stage": "A2-4-oof",
        "status": "EXECUTED",
        "execution_mode": execution_mode,
        "bootstrap_samples": sample_count,
        "models": residual_results,
        "activation": activation,
        "manifest": run_manifest,
    }
    _write_json(run_dir / "manifest.json", run_manifest)
    summary_name = "a2_oof_diagnostic.json" if execution_mode == "frozen" else "a2_oof_diagnostic_smoke.json"
    _write_json(summary_dir / summary_name, result)
    return result


def _summarize_seed_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("seed records must be non-empty")
    metrics = [record["validation"] for record in records]
    macro_values = np.asarray([float(metric["macro_RNMAE"]) for metric in metrics], dtype=np.float64)
    component_values = np.asarray(
        [metric["component_RNMAE"] for metric in metrics],
        dtype=np.float64,
    )
    return {
        "seed_records": list(records),
        "mean_validation_macro_RNMAE": float(macro_values.mean()),
        "std_validation_macro_RNMAE": float(macro_values.std(ddof=0)),
        "mean_component_RNMAE": [float(value) for value in component_values.mean(axis=0)],
        "worst_seed_validation_macro_RNMAE": float(macro_values.max()),
    }


def _head_candidate_gate(
    h0: Mapping[str, Any],
    candidate_name: str,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    h0_values = [float(record["validation"]["macro_RNMAE"]) for record in h0["seed_records"]]
    candidate_values = [
        float(record["validation"]["macro_RNMAE"])
        for record in candidate["seed_records"]
    ]
    relative_improvement = (
        float(h0["mean_validation_macro_RNMAE"] - candidate["mean_validation_macro_RNMAE"])
        / float(h0["mean_validation_macro_RNMAE"])
    )
    component_delta = [
        float(candidate_value - h0_value)
        for candidate_value, h0_value in zip(
            candidate["mean_component_RNMAE"],
            h0["mean_component_RNMAE"],
            strict=True,
        )
    ]
    same_direction = sum(
        candidate_value < h0_value
        for h0_value, candidate_value in zip(h0_values, candidate_values, strict=True)
    )
    checks = {
        "relative_improvement_at_least_5_percent": relative_improvement >= 0.05,
        "at_least_four_seeds_improve": same_direction >= 4,
        "component_degradation_within_0_005": max(component_delta) <= 0.005,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "comparison": f"{candidate_name}_minus_H0",
        "relative_improvement": relative_improvement,
        "same_direction_seed_count": same_direction,
        f"component_delta_{candidate_name}_minus_H0": component_delta,
        "checks": checks,
    }


def _head_audit_gate(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    h0 = summaries["H0"]
    candidates = {
        candidate_name: _head_candidate_gate(h0, candidate_name, summaries[candidate_name])
        for candidate_name in ("H1", "H2")
    }
    selected_head = next(
        (candidate_name for candidate_name in ("H1", "H2") if candidates[candidate_name]["status"] == "PASS"),
        "H0",
    )
    return {
        "status": "PASS" if selected_head != "H0" else "FAIL",
        "selected_head": selected_head,
        "selection_rule": "first passing candidate in registered order H1, H2; otherwise H0",
        "candidates": candidates,
    }


def _deepsets_gate(
    summaries: Mapping[str, Mapping[str, Any]],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    concat = summaries["C1"]
    deepsets = summaries["M1"]
    concat_values = [float(record["validation"]["macro_RNMAE"]) for record in concat["seed_records"]]
    deepsets_values = [float(record["validation"]["macro_RNMAE"]) for record in deepsets["seed_records"]]
    relative_improvement = (
        float(concat["mean_validation_macro_RNMAE"] - deepsets["mean_validation_macro_RNMAE"])
        / float(concat["mean_validation_macro_RNMAE"])
    )
    component_delta = [
        float(deepsets_value - concat_value)
        for deepsets_value, concat_value in zip(
            deepsets["mean_component_RNMAE"],
            concat["mean_component_RNMAE"],
            strict=True,
        )
    ]
    same_direction = sum(
        deepsets_value < concat_value
        for concat_value, deepsets_value in zip(concat_values, deepsets_values, strict=True)
    )
    checks = {
        "relative_improvement_at_least_5_percent": relative_improvement >= 0.05,
        "at_least_four_seeds_improve": same_direction >= 4,
        "component_degradation_within_0_005": max(component_delta) <= 0.005,
        "parameter_match_within_0_10": bool(parity["within_tolerance"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "comparison": "M1_minus_C1",
        "relative_improvement": relative_improvement,
        "same_direction_seed_count": same_direction,
        "component_delta_M1_minus_C1": component_delta,
        "checks": checks,
    }


def _model_parity_for_capacity(
    concat_config: Mapping[str, Any],
    deepsets_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    *,
    capacity_name: str,
) -> dict[str, Any]:
    torch.manual_seed(17)
    concat = build_a2_model_from_config(
        concat_config,
        train_config,
        capacity_name=capacity_name,
        head_id="H0",
    )
    torch.manual_seed(17)
    deepsets = build_a2_model_from_config(
        deepsets_config,
        train_config,
        capacity_name=capacity_name,
        head_id="H0",
    )
    return parameter_parity_report(concat, deepsets, tolerance=0.10)


def _prediction_array_hash(predictions: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for prediction in predictions:
        array = np.asarray(prediction, dtype=np.float64, order="C")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _scalar_feature_matrix(samples: Sequence[Any]) -> np.ndarray:
    return np.asarray(
        [
            [sample.signals[index][0, 0] for index in range(len(sample.signals))]
            for sample in samples
        ],
        dtype=np.float64,
    )


class _TargetScaledMLP:
    def __init__(self, seed: int) -> None:
        self._model = MLPRegressor(
            hidden_layer_sizes=(32,),
            solver="lbfgs",
            alpha=1.0e-4,
            max_iter=1000,
            random_state=seed,
        )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "_TargetScaledMLP":
        self._model.fit(features, targets / 100.0)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(features) * 100.0, dtype=np.float64)


def _explained_variance_by_component(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> list[float | None]:
    result: list[float | None] = []
    for index in range(targets.shape[1]):
        target = targets[:, index]
        prediction = predictions[:, index]
        centered = target - target.mean()
        total = float(np.dot(centered, centered))
        if total <= 0.0:
            result.append(None)
            continue
        residual = target - prediction
        result.append(float(1.0 - np.dot(residual, residual) / total))
    return result


def _residual_activation_gate(
    payload: Mapping[str, Any],
    *,
    reference_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    positive_oof_explained_variance = all(
        value is not None and float(value) > 0.0
        for value in payload["oof_explained_variance"]
    )
    component_delta = [
        float(candidate - reference)
        for candidate, reference in zip(
            candidate_metrics["component_RNMAE"],
            reference_metrics["component_RNMAE"],
            strict=True,
        )
    ]
    relative_improvement = (
        float(reference_metrics["macro_RNMAE"] - candidate_metrics["macro_RNMAE"])
        / float(reference_metrics["macro_RNMAE"])
    )
    component_ok = max(component_delta) <= 0.005
    improvement_ok = relative_improvement >= 0.05
    bootstrap_ok = bool(bootstrap["ci_excludes_zero"])
    checks = {
        "positive_oof_explained_variance": positive_oof_explained_variance,
        "val_improvement_at_least_5_percent": improvement_ok,
        "bootstrap_ci_excludes_zero": bootstrap_ok,
        "component_degradation_within_0_005": component_ok,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "reference": "B5_base_validation",
        "relative_improvement_vs_B5": relative_improvement,
        "component_delta_vs_B5": component_delta,
        "checks": checks,
    }


def run_a2_protocol(
    *,
    project_root: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    experiment_config_paths: Sequence[str | Path] | None = None,
    unlock_test: bool = False,
    selection_record_path: str | Path | None = None,
    selected_checkpoint_path: str | Path | None = None,
    formal_run_status: str | None = None,
) -> dict[str, Any]:
    root = (Path(project_root) if project_root is not None else _default_project_root()).resolve()
    config_dir = root / "configs"
    eval_path = _resolve_file(
        root,
        Path(eval_config_path) if eval_config_path is not None else config_dir / "eval" / "a2_eval.json",
    )
    train_path = _resolve_file(
        root,
        Path(train_config_path) if train_config_path is not None else config_dir / "train" / "a2_train.json",
    )
    default_experiments = (
        config_dir / "experiment" / "a2_head_audit.json",
        config_dir / "experiment" / "a2_deepsets.json",
        config_dir / "experiment" / "a2_formal.json",
    )
    experiment_paths = tuple(
        _resolve_file(root, Path(path))
        for path in (experiment_config_paths or default_experiments)
    )
    if not experiment_paths:
        raise A2ProtocolError("at least one A2 experiment config is required")

    eval_config = _read_json_object(eval_path)
    train_config = _read_json_object(train_path)
    validate_a2_eval_config(eval_config)
    validate_a2_train_config(train_config)
    manifest_path = _resolve_file(root, Path(str(eval_config["data"]["manifest_path"])))
    a1_manifest = _read_and_validate_a1_manifest(manifest_path, eval_config)

    experiment_configs: list[tuple[Path, dict[str, Any]]] = []
    for experiment_path in experiment_paths:
        experiment = _read_json_object(experiment_path)
        validate_a2_experiment_config(experiment)
        _validate_experiment_bindings(
            experiment,
            eval_config=eval_config,
            eval_config_path=eval_path,
            train_config_path=train_path,
            root=root,
        )
        experiment_configs.append((experiment_path, experiment))

    unlock_evidence: dict[str, Any] | None = None
    if unlock_test:
        if selection_record_path is None or selected_checkpoint_path is None:
            raise TestUnlockError(
                "--unlock-test requires both a selection record and a selected checkpoint"
            )
        formal_path, formal_config = _find_formal_experiment(experiment_configs)
        candidate_path = _resolve_file(root, Path(str(formal_config["candidate_config"])))
        chart_path = _resolve_file(root, Path(str(formal_config["primary_chart_template"])))
        unlock_evidence = verify_test_unlock_evidence(
            project_root=root,
            candidate_config_path=candidate_path,
            selected_checkpoint_path=selected_checkpoint_path,
            primary_chart_template_path=chart_path,
            selection_record_path=selection_record_path,
            formal_run_status=formal_run_status,
        )
        unlock_evidence["formal_experiment_config"] = _relative_path(root, formal_path)
    elif formal_run_status is not None:
        raise TestUnlockError("formal_run_status is only valid together with unlock_test=True")

    config_paths = {
        "eval_config": eval_path,
        "train_config": train_path,
    }
    for experiment_path, experiment in experiment_configs:
        prefix = str(experiment["experiment_id"])
        config_paths[prefix] = experiment_path
        for key in (
            "data_config",
            "eval_config",
            "train_config",
            "candidate_config",
            "baseline_config",
            "primary_chart_template",
        ):
            value = experiment.get(key)
            if isinstance(value, str):
                referenced_path = _resolve_file(root, Path(value))
                config_paths[f"{prefix}.{key}"] = referenced_path

    run_dir = root / "outputs" / "runs" / "a2" / "a2-0-protocol"
    summary_dir = root / "outputs" / "summary" / "a2"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = build_run_manifest(
        project_root=root,
        stage="A2-0",
        config_paths=config_paths,
        data_manifest_path=manifest_path,
        exit_code=0,
        prediction_hash=None,
        status="PASS",
        test_unlocked=unlock_test,
        unlock_evidence=unlock_evidence,
    )
    _write_json(run_dir / "manifest.json", run_manifest)
    protocol = {
        "schema_version": A2_PROTOCOL_SCHEMA_VERSION,
        "stage": "A2-0",
        "status": "PASS",
        "data": {
            "data_version": a1_manifest["data_version"],
            "content_sha256": a1_manifest["content_sha256"],
            "split_hash": run_manifest["split_hash"],
        },
        "config_hashes": run_manifest["config_hashes"],
        "test_access": run_manifest["test_access"],
        "run_manifest": _relative_path(root, run_dir / "manifest.json"),
    }
    _write_json(summary_dir / "a2_protocol.json", protocol)
    return {
        "stage": "A2-0",
        "status": "PASS",
        "run_dir": str(run_dir),
        "summary_path": str(summary_dir / "a2_protocol.json"),
        "manifest": run_manifest,
    }


def validate_a2_eval_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    if config.get("schema_version") != A2_EVAL_SCHEMA_VERSION:
        raise A2ProtocolError("A2 evaluation config has unsupported schema_version")
    if config.get("parent_schema_version") != "gf-eval-1":
        raise A2ProtocolError("A2 evaluation config must declare parent_schema_version=gf-eval-1")
    if config.get("metric") != "macro_RNMAE":
        raise A2ProtocolError("A2 evaluation metric must be macro_RNMAE")
    target_ranges = config.get("target_ranges")
    if not isinstance(target_ranges, Mapping) or set(target_ranges) != set(TARGET_NAMES):
        raise A2ProtocolError("target_ranges must cover exactly the three A1 targets")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0 for value in target_ranges.values()):
        raise A2ProtocolError("target_ranges must contain positive numeric values")

    data = _required_mapping(config, "data")
    if data.get("dataset_id") != A1_DATASET_ID:
        raise A2ProtocolError("A2 data.dataset_id must be ar_he_co2")
    if data.get("data_version") != A1_DATA_VERSION:
        raise A2ProtocolError("A2 must use the frozen A1 data_version")
    _require_hash(data, "content_sha256")
    if data["content_sha256"] != A1_CONTENT_SHA256:
        raise A2ProtocolError("A2 data content_sha256 does not match frozen A1 formal v1")
    if data.get("split_seed") != A1_SPLIT_SEED:
        raise A2ProtocolError("A2 split_seed does not match frozen A1 split")
    _require_hash(data, "split_hash")
    manifest_path = data.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path:
        raise A2ProtocolError("A2 data.manifest_path must be a non-empty string")

    seeds = _required_int_list(config, "formal_training_seeds")
    if tuple(seeds) != A2_TRAINING_SEEDS:
        raise A2ProtocolError(f"formal_training_seeds must be {list(A2_TRAINING_SEEDS)}")
    if config.get("split_counts") != A1_SPLIT_COUNTS:
        raise A2ProtocolError(f"split_counts must be {A1_SPLIT_COUNTS}")
    if config.get("split_seed") != A1_SPLIT_SEED:
        raise A2ProtocolError("top-level split_seed does not match frozen A1 split")
    if config.get("bootstrap_seed") != A1_SPLIT_SEED:
        raise A2ProtocolError("bootstrap_seed does not match the registered A1 seed")
    if config.get("bootstrap_samples") != 2000:
        raise A2ProtocolError("A2 bootstrap_samples must be 2000")
    _require_float(config, "confidence_level", expected=0.95)

    promotion = _required_mapping(config, "promotion")
    _require_float(promotion, "relative_improvement", expected=0.05)
    if promotion.get("min_seeds_same_direction") != 4:
        raise A2ProtocolError("A2 requires at least four seeds in the same direction")
    _require_float(promotion, "max_component_absolute_degradation", expected=0.005)
    if promotion.get("primary_statistic") != "mean_of_independent_seed_validation_macro_RNMAE":
        raise A2ProtocolError("A2 primary statistic must be the independent-seed mean")

    baseline = _required_mapping(config, "baseline")
    if baseline.get("primary_id") != "B5":
        raise A2ProtocolError("A2 primary strong baseline must be B5")
    if baseline.get("comparison") != "five_independent_seeds":
        raise A2ProtocolError("A2 B5 comparison must use five independent seeds")
    if baseline.get("secondary_ensemble_key") != "B5__mean":
        raise A2ProtocolError("A2 secondary ensemble must be B5__mean")
    _require_float(baseline, "a1_validation_macro_RNMAE", expected=0.00581589)
    _require_float(baseline, "same_order_relative_gap", expected=0.2)

    selection_splits = config.get("model_selection_splits")
    if tuple(selection_splits or ()) != A2_ALLOWED_SELECTION_SPLITS:
        raise A2ProtocolError(
            f"model_selection_splits must be {list(A2_ALLOWED_SELECTION_SPLITS)}"
        )
    test_access = _required_mapping(config, "test_access")
    if test_access.get("default") != "locked":
        raise A2ProtocolError("A2 test access must default to locked")
    if test_access.get("unlock_flag") != "--unlock-test":
        raise A2ProtocolError("A2 test unlock flag must be --unlock-test")
    required_evidence = test_access.get("required_evidence")
    expected_evidence = (
        "candidate_config_sha256",
        "selected_checkpoint_sha256",
        "primary_chart_template_sha256",
        "formal_run_status",
    )
    if tuple(required_evidence or ()) != expected_evidence:
        raise A2ProtocolError("A2 test lock evidence fields are incomplete or reordered")
    if test_access.get("required_formal_run_status") != A2_FORMAL_STATUS:
        raise A2ProtocolError("A2 formal test unlock requires status FROZEN")


def validate_a2_train_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    if config.get("schema_version") != A2_TRAIN_SCHEMA_VERSION:
        raise A2ProtocolError("A2 training config has unsupported schema_version")
    seeds = _required_int_list(config, "seeds")
    if tuple(seeds) != A2_TRAINING_SEEDS:
        raise A2ProtocolError(f"training seeds must be {list(A2_TRAINING_SEEDS)}")
    optimizer = _required_mapping(config, "optimizer")
    if optimizer.get("name") not in {"Adam", "LBFGS"}:
        raise A2ProtocolError("A2 optimizer must be Adam or LBFGS")
    _require_positive_number(optimizer, "learning_rate")
    _require_nonnegative_number(optimizer, "weight_decay")
    if optimizer.get("name") == "LBFGS":
        if optimizer.get("weight_decay") != 0.0:
            raise A2ProtocolError("LBFGS A2 training must use weight_decay=0")
        for key in ("max_iter", "history_size"):
            value = optimizer.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise A2ProtocolError(f"LBFGS {key} must be a positive integer")
    loss = _required_mapping(config, "loss")
    if loss.get("name") != "mse":
        raise A2ProtocolError("A2 first-pass loss must be mse for all heads")
    target_scale = loss.get("target_scale")
    if target_scale != [100.0, 100.0, 100.0]:
        raise A2ProtocolError("A2 target_scale must be [100, 100, 100]")
    max_epochs = config.get("max_epochs")
    if not isinstance(max_epochs, int) or isinstance(max_epochs, bool) or max_epochs <= 0:
        raise A2ProtocolError("max_epochs must be a positive integer")
    early_stopping = _required_mapping(config, "early_stopping")
    if early_stopping.get("enabled") is not True:
        raise A2ProtocolError("A2 early stopping must be enabled")
    if early_stopping.get("monitor") != "val_macro_RNMAE":
        raise A2ProtocolError("A2 early stopping must monitor val_macro_RNMAE")
    if early_stopping.get("selection_split") != "val":
        raise A2ProtocolError("A2 checkpoint selection must use val")
    if early_stopping.get("test_access") != "forbidden":
        raise A2ProtocolError("A2 early stopping must forbid test access")
    search = _required_mapping(config, "search")
    trials = search.get("trials_per_candidate")
    if not isinstance(trials, int) or isinstance(trials, bool) or trials <= 0:
        raise A2ProtocolError("search.trials_per_candidate must be a positive integer")
    if search.get("selection_split") != "val":
        raise A2ProtocolError("A2 search selection_split must be val")
    if tuple(search.get("allowed_splits") or ()) != A2_ALLOWED_SELECTION_SPLITS:
        raise A2ProtocolError("A2 search allowed_splits must exclude test")
    presets = config.get("capacity_presets")
    if not isinstance(presets, list) or len(presets) != 2:
        raise A2ProtocolError("A2 must pre-register exactly two capacity presets")
    names: list[str] = []
    for preset in presets:
        if not isinstance(preset, Mapping):
            raise A2ProtocolError("each capacity preset must be an object")
        name = preset.get("name")
        if not isinstance(name, str) or not name:
            raise A2ProtocolError("capacity preset name must be non-empty")
        names.append(name)
        for key in ("encoder_hidden_dim", "fusion_hidden_dim", "head_hidden_dim"):
            value = preset.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise A2ProtocolError(f"capacity preset {key} must be a positive integer")
    if len(set(names)) != len(names):
        raise A2ProtocolError("capacity preset names must be unique")
    _require_float(config, "parameter_match_tolerance", expected=0.10)


def validate_a2_experiment_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    if config.get("schema_version") != A2_EXPERIMENT_SCHEMA_VERSION:
        raise A2ProtocolError("A2 experiment config has unsupported schema_version")
    stage = config.get("stage")
    if stage not in {"A2-2", "A2-3", "A2-5"}:
        raise A2ProtocolError("A2 experiment stage must be A2-2, A2-3, or A2-5")
    experiment_id = config.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise A2ProtocolError("experiment_id must be a non-empty string")
    for key in ("data_config", "data_manifest", "eval_config", "train_config", "candidate_config", "output_dir"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise A2ProtocolError(f"{key} must be a non-empty path string")
    allowed_splits = config.get("allowed_read_splits")
    if tuple(allowed_splits or ()) != A2_ALLOWED_SELECTION_SPLITS:
        raise A2ProtocolError("A2 experiment allowed_read_splits must exclude test")
    test_access = _required_mapping(config, "test_access")
    if test_access.get("mode") != "locked" or test_access.get("unlock_flag") != "--unlock-test":
        raise A2ProtocolError("every A2 experiment must keep test locked")
    if stage == "A2-5":
        chart_template = config.get("primary_chart_template")
        if not isinstance(chart_template, str) or not chart_template:
            raise A2ProtocolError("formal A2 experiment requires primary_chart_template")
        selection_record = config.get("selection_record")
        if not isinstance(selection_record, str) or not selection_record:
            raise A2ProtocolError("formal A2 experiment requires selection_record")


def compute_split_hash(manifest: Mapping[str, Any]) -> str:
    conditions = manifest.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise A2ProtocolError("A1 manifest conditions must be a non-empty list")
    assignments: list[dict[str, str]] = []
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise A2ProtocolError("A1 manifest condition must be an object")
        mixture_id = condition.get("mixture_id")
        split = condition.get("split")
        if not isinstance(mixture_id, str) or not mixture_id:
            raise A2ProtocolError("A1 condition mixture_id must be non-empty")
        if split not in {"train", "val", "test"}:
            raise A2ProtocolError(f"unsupported A1 split {split!r}")
        assignments.append({"mixture_id": mixture_id, "split": str(split)})
    assignments.sort(key=lambda item: item["mixture_id"])
    return _canonical_sha256(
        {
            "schema_version": "gf-a2-split-1",
            "split_seed": manifest.get("split_seed"),
            "assignments": assignments,
        }
    )


def build_run_manifest(
    *,
    project_root: str | Path,
    stage: str,
    config_paths: Mapping[str, str | Path],
    data_manifest_path: str | Path,
    exit_code: int,
    prediction_hash: str | None,
    status: str,
    test_unlocked: bool,
    unlock_evidence: Mapping[str, Any] | None = None,
    worktree_revision: str | None = None,
    worktree_dirty: bool | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if not stage:
        raise A2ProtocolError("stage must be non-empty")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise A2ProtocolError("exit_code must be an integer")
    if prediction_hash is not None:
        _validate_hash_value(prediction_hash, "prediction_hash")
    if not config_paths:
        raise A2ProtocolError("config_paths must contain at least one config")
    manifest_path = _resolve_file(root, Path(data_manifest_path))
    data_manifest = _read_json_object(manifest_path)
    split_hash = _validate_a1_manifest_for_run(data_manifest)
    config_hashes: dict[str, str] = {}
    for name, path in sorted(config_paths.items()):
        if not isinstance(name, str) or not name:
            raise A2ProtocolError("config hash names must be non-empty strings")
        resolved = _resolve_file(root, Path(path))
        config_hashes[name] = sha256_file(resolved)

    if (worktree_revision is None) != (worktree_dirty is None):
        raise A2ProtocolError("worktree_revision and worktree_dirty must be supplied together")
    if worktree_revision is None:
        worktree_revision, worktree_dirty = _git_state(root)
    if not isinstance(worktree_revision, str) or not worktree_revision:
        raise A2ProtocolError("worktree_revision must be a non-empty string")
    if not isinstance(worktree_dirty, bool):
        raise A2ProtocolError("worktree_dirty must be boolean")
    if test_unlocked and unlock_evidence is None:
        raise TestUnlockError("unlocked run manifest requires unlock evidence")
    if not test_unlocked and unlock_evidence is not None:
        raise TestUnlockError("locked run manifest cannot contain unlock evidence")

    return {
        "schema_version": A2_RUN_MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        "worktree": {
            "revision": worktree_revision,
            "dirty": worktree_dirty,
        },
        "config_hashes": config_hashes,
        "data_version": data_manifest["data_version"],
        "data_content_sha256": data_manifest["content_sha256"],
        "split_hash": split_hash,
        "exit_code": exit_code,
        "prediction_hash": prediction_hash,
        "test_access": {
            "default": "locked",
            "unlocked": test_unlocked,
            "evidence": dict(unlock_evidence) if unlock_evidence is not None else None,
        },
    }


def assert_test_unlocked(
    *,
    candidate_config_hash: str | None,
    selected_checkpoint_hash: str | None,
    primary_chart_template_hash: str | None,
    formal_run_status: str | None,
    expected_candidate_config_hash: str | None,
    expected_selected_checkpoint_hash: str | None,
    expected_primary_chart_template_hash: str | None,
    required_formal_run_status: str = A2_FORMAL_STATUS,
) -> None:
    values = {
        "candidate_config_sha256": candidate_config_hash,
        "selected_checkpoint_sha256": selected_checkpoint_hash,
        "primary_chart_template_sha256": primary_chart_template_hash,
    }
    expected = {
        "candidate_config_sha256": expected_candidate_config_hash,
        "selected_checkpoint_sha256": expected_selected_checkpoint_hash,
        "primary_chart_template_sha256": expected_primary_chart_template_hash,
    }
    for name, value in values.items():
        if value is None:
            raise TestUnlockError(f"test unlock evidence is missing {name}")
        _validate_hash_value(value, name)
        expected_value = expected[name]
        if expected_value is None:
            raise TestUnlockError(f"selection record is missing {name}")
        _validate_hash_value(expected_value, f"expected_{name}")
        if value != expected_value:
            raise TestUnlockError(f"test unlock evidence mismatch for {name}")
    if formal_run_status != required_formal_run_status:
        raise TestUnlockError(
            f"formal_run_status must be {required_formal_run_status}, got {formal_run_status!r}"
        )


def verify_test_unlock_evidence(
    *,
    project_root: str | Path,
    candidate_config_path: str | Path,
    selected_checkpoint_path: str | Path,
    primary_chart_template_path: str | Path,
    selection_record_path: str | Path,
    formal_run_status: str | None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    candidate_path = _resolve_file(root, Path(candidate_config_path))
    checkpoint_path = _resolve_file(root, Path(selected_checkpoint_path))
    chart_path = _resolve_file(root, Path(primary_chart_template_path))
    record_path = _resolve_file(root, Path(selection_record_path))
    record = _read_json_object(record_path)
    candidate_hash = sha256_file(candidate_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    chart_hash = sha256_file(chart_path)
    assert_test_unlocked(
        candidate_config_hash=candidate_hash,
        selected_checkpoint_hash=checkpoint_hash,
        primary_chart_template_hash=chart_hash,
        formal_run_status=formal_run_status,
        expected_candidate_config_hash=_record_hash(record, "candidate_config_sha256"),
        expected_selected_checkpoint_hash=_record_hash(record, "selected_checkpoint_sha256"),
        expected_primary_chart_template_hash=_record_hash(record, "primary_chart_template_sha256"),
    )
    return {
        "candidate_config_sha256": candidate_hash,
        "selected_checkpoint_sha256": checkpoint_hash,
        "primary_chart_template_sha256": chart_hash,
        "formal_run_status": formal_run_status,
    }


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_and_validate_a1_manifest(
    path: Path,
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _read_json_object(path)
    _validate_a1_manifest_for_run(manifest)
    data = _required_mapping(eval_config, "data")
    if manifest["content_sha256"] != data["content_sha256"]:
        raise A2ProtocolError("A1 manifest content_sha256 does not match A2 evaluation config")
    if manifest["data_version"] != data["data_version"]:
        raise A2ProtocolError("A1 manifest data_version does not match A2 evaluation config")
    if compute_split_hash(manifest) != data["split_hash"]:
        raise A2ProtocolError("A1 manifest split hash does not match A2 evaluation config")
    return manifest


def _validate_a1_manifest_for_run(manifest: Mapping[str, Any]) -> str:
    _validate_no_forbidden_keys(manifest)
    if manifest.get("schema_version") != A1_DATA_SCHEMA_VERSION:
        raise A2ProtocolError("A2 requires an A1 data manifest with schema gf-a1-data-1")
    if manifest.get("dataset_id") != A1_DATASET_ID:
        raise A2ProtocolError("A2 requires the ar_he_co2 A1 manifest")
    if manifest.get("data_version") != A1_DATA_VERSION:
        raise A2ProtocolError("A2 requires the frozen A1 formal data version")
    if manifest.get("content_sha256") != A1_CONTENT_SHA256:
        raise A2ProtocolError("A1 manifest content_sha256 does not match the frozen formal hash")
    if manifest.get("sample_count") != A1_SAMPLE_COUNT:
        raise A2ProtocolError("A2 requires 1200 formal A1 samples")
    if manifest.get("split_seed") != A1_SPLIT_SEED:
        raise A2ProtocolError("A1 manifest split_seed does not match the frozen split")
    conditions = manifest.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != A1_SAMPLE_COUNT:
        raise A2ProtocolError("A1 manifest conditions do not match formal sample count")
    mixture_ids = [
        condition.get("mixture_id")
        for condition in conditions
        if isinstance(condition, Mapping)
    ]
    if len(mixture_ids) != A1_SAMPLE_COUNT or len(set(mixture_ids)) != A1_SAMPLE_COUNT:
        raise A2ProtocolError("A1 manifest mixture_id values must be unique")
    split_counts = {
        split: sum(
            isinstance(condition, Mapping) and condition.get("split") == split
            for condition in conditions
        )
        for split in ("train", "val", "test")
    }
    if split_counts != A1_SPLIT_COUNTS:
        raise A2ProtocolError(f"A1 manifest split counts must be {A1_SPLIT_COUNTS}, got {split_counts}")
    return compute_split_hash(manifest)


def _validate_experiment_bindings(
    experiment: Mapping[str, Any],
    *,
    eval_config: Mapping[str, Any],
    eval_config_path: Path,
    train_config_path: Path,
    root: Path,
) -> None:
    if experiment["data_manifest"] != eval_config["data"]["manifest_path"]:
        raise A2ProtocolError(
            f"{experiment['experiment_id']} must use the A2 evaluation manifest path"
        )
    referenced_eval = _resolve_file(root, Path(str(experiment["eval_config"])))
    if referenced_eval != eval_config_path:
        referenced_eval_config = _read_json_object(referenced_eval)
        validate_a2_eval_config(referenced_eval_config)
        if referenced_eval_config != eval_config:
            raise A2ProtocolError(f"{experiment['experiment_id']} eval_config differs from selected A2 eval config")
    referenced_train = _resolve_file(root, Path(str(experiment["train_config"])))
    if referenced_train != train_config_path:
        referenced_train_config = _read_json_object(referenced_train)
        validate_a2_train_config(referenced_train_config)
        selected_train_config = _read_json_object(train_config_path)
        if referenced_train_config != selected_train_config:
            raise A2ProtocolError(f"{experiment['experiment_id']} train_config differs from selected A2 train config")
    data_config = _read_json_object(_resolve_file(root, Path(str(experiment["data_config"]))))
    if data_config.get("schema_version") != A1_DATA_SCHEMA_VERSION:
        raise A2ProtocolError(f"{experiment['experiment_id']} data_config is not A1 schema")
    if data_config.get("data_version") != A1_DATA_VERSION:
        raise A2ProtocolError(f"{experiment['experiment_id']} data_config changes A1 data_version")
    if data_config.get("split_seed") != A1_SPLIT_SEED:
        raise A2ProtocolError(f"{experiment['experiment_id']} data_config changes A1 split_seed")


def _find_formal_experiment(
    experiments: Sequence[tuple[Path, Mapping[str, Any]]],
) -> tuple[Path, Mapping[str, Any]]:
    formal = [item for item in experiments if item[1].get("stage") == "A2-5"]
    if len(formal) != 1:
        raise TestUnlockError("exactly one A2-5 formal experiment is required for test unlock")
    return formal[0]


def _record_hash(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TestUnlockError(f"selection record is missing {key}")
    return value


def _git_state(root: Path) -> tuple[str, bool]:
    revision_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = revision_result.stdout.strip()
    if not revision:
        raise A2ProtocolError("git rev-parse returned an empty worktree revision")
    return revision, bool(status_result.stdout.strip())


def _validate_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_KEYS & set(value)
        if forbidden:
            raise A2ProtocolError(f"forbidden legacy keys: {sorted(forbidden)}")
        for child in value.values():
            _validate_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _validate_no_forbidden_keys(child)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_hash_value(value: str, name: str) -> None:
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise A2ProtocolError(f"{name} must be a lowercase SHA256 hex string")


def _require_hash(config: Mapping[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, str):
        raise A2ProtocolError(f"{key} must be a SHA256 string")
    _validate_hash_value(value, key)


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise A2ProtocolError(f"{key} must be an object")
    return value


def _required_int_list(config: Mapping[str, Any], key: str) -> list[int]:
    value = config.get(key)
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise A2ProtocolError(f"{key} must be a list of integers")
    if len(set(value)) != len(value):
        raise A2ProtocolError(f"{key} must not contain duplicates")
    return value


def _require_float(config: Mapping[str, Any], key: str, *, expected: float) -> None:
    value = config.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) != expected:
        raise A2ProtocolError(f"{key} must be exactly {expected}")


def _require_positive_number(config: Mapping[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0.0:
        raise A2ProtocolError(f"{key} must be a positive number")


def _require_nonnegative_number(config: Mapping[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0.0:
        raise A2ProtocolError(f"{key} must be a non-negative number")


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise A2ProtocolError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_file(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise A2ProtocolError(f"configured path escapes project root: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one general_fusion A2 protocol stage.")
    parser.add_argument(
        "--mode",
        choices=("protocol", "smoke", "torch-concat", "head-audit", "deepsets", "oof"),
        default="protocol",
    )
    parser.add_argument("--project-root", type=Path, default=_default_project_root())
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--experiment-config", type=Path, action="append")
    parser.add_argument("--unlock-test", action="store_true")
    parser.add_argument("--selection-record", type=Path)
    parser.add_argument("--selected-checkpoint", type=Path)
    parser.add_argument("--formal-run-status")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_a2(
        project_root=args.project_root,
        mode=args.mode,
        eval_config_path=args.eval_config,
        train_config_path=args.train_config,
        experiment_config_paths=args.experiment_config,
        unlock_test=args.unlock_test,
        selection_record_path=args.selection_record,
        selected_checkpoint_path=args.selected_checkpoint,
        formal_run_status=args.formal_run_status,
        max_epochs_override=args.max_epochs,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps({"stage": result["stage"], "status": result["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
