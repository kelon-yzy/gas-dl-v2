from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

from gf.dl.contracts import UnifiedSample, collate_samples
from gf.dl.evaluation import evaluate_output_constraints, evaluate_predictions, group_bootstrap_comparison
from gf.dl.preprocessing import TrainGroupStandardScaler
from gf.dl.training import (
    A2FusionModel,
    TorchConcatMLP,
    TorchTrainingConfig,
    parameter_parity_report,
    train_torch_model,
    trainable_parameter_count,
)
from gf.pipeline.tqif_common import canonical_hash as _canonical_sha256
from gf.sim.a1_dataset import A1PhysicsConfig, DEFAULT_A1_PHYSICS, deterministic_signal_vector
from gf.sim.a2h_audit import run_difficulty_audit
from gf.sim.a2h_dataset import (
    A2HDataset,
    A2H_DATA_VERSION_PREFIX,
    A2H_DEVELOPMENT_SPLITS,
    A2H_SCHEMA_VERSION,
    A2H_SPLITS,
    FORBIDDEN_KEYS,
    SENSOR_IDS,
    TARGET_NAMES,
    composition_region,
    compute_split_family_hash,
    generate_a2h_dataset,
    load_a2h_dataset,
    nominal_signal_parity,
)


A2H_EVAL_SCHEMA_VERSION = "gf-a2h-eval-2"
A2H_TRAIN_SCHEMA_VERSION = "gf-a2h-train-2"
A2H_MODEL_SCHEMA_VERSION = "gf-a2h-model-2"
A2H_EXPERIMENT_SCHEMA_VERSION = "gf-a2h-experiment-2"
A2H_RUN_MANIFEST_SCHEMA_VERSION = "gf-a2h-run-manifest-2"
A2H_PROTOCOL_SCHEMA_VERSION = "gf-a2h-protocol-2"
A2H_FORMAL_STATUS = "FROZEN"
A2H_TRAINING_SEEDS = (17, 29, 43, 71, 101)
A2H_SPLIT_FAMILIES = ("iid", "noise", "environment", "calibration", "composition", "joint")
A2H_ALLOWED_READ_SPLITS = ("train", "inner_oof", "val", "stress_val")
A2H_REQUIRED_HARD_EVIDENCE = (
    "data_content_sha256",
    "split_family_hash",
    "eligible_axes_sha256",
    "candidate_config_sha256",
    "matched_baseline_config_sha256",
    "selected_checkpoint_sha256",
    "primary_chart_template_sha256",
    "formal_run_status",
)
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
A2H_DATA_RELATIVE_DIR = Path("data/a2h_v2")
A2H_OUTPUT_NAMESPACE = "a2h_v2"
A2H_DEFAULT_DATA_CONFIG = Path("configs/data/ar_he_co2_a2h_v2.json")


class A2HProtocolError(ValueError):
    """Raised when an A2H contract, provenance, or stage gate is violated."""


class A2HTestUnlockError(A2HProtocolError):
    """Raised when hard-test evidence is incomplete or mismatched."""

    __test__ = False


@dataclass(frozen=True)
class _FitResult:
    model_id: str
    seed: int
    head_id: str
    feature_map: str
    include_context: bool
    prediction: np.ndarray
    resources: Mapping[str, Any]


@dataclass(frozen=True)
class _TorchArtifact:
    model: torch.nn.Module
    scaler: TrainGroupStandardScaler
    context_statistics: Mapping[str, tuple[float, float]]
    resources: Mapping[str, Any]


_TORCH_FIT_CACHE: dict[tuple[Any, ...], _TorchArtifact] = {}


def run_a2h(
    *,
    project_root: str | Path | None = None,
    mode: str = "protocol",
    data_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    max_epochs_override: int | None = None,
    bootstrap_samples: int | None = None,
    unlock_hard_test: bool = False,
    selection_record_path: str | Path | None = None,
    selected_checkpoint_path: str | Path | None = None,
    formal_run_status: str | None = None,
) -> dict[str, Any]:
    """Run one explicit A2H protocol stage."""

    root = _project_root(project_root)
    if mode == "protocol":
        return run_a2h_protocol(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
        )
    if mode == "generate":
        return run_a2h_generation(project_root=root, data_config_path=data_config_path)
    if mode == "audit":
        return run_a2h_difficulty_audit(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
        )
    if mode == "learning-noise":
        return run_a2h_learning_noise(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
            max_epochs_override=max_epochs_override,
        )
    if mode == "ood":
        return run_a2h_ood(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
            max_epochs_override=max_epochs_override,
        )
    if mode == "compare":
        return run_a2h_algorithm_comparison(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
            max_epochs_override=max_epochs_override,
        )
    if mode == "formal":
        return run_a2h_formal(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
            unlock_hard_test=unlock_hard_test,
            selection_record_path=selection_record_path,
            selected_checkpoint_path=selected_checkpoint_path,
            formal_run_status=formal_run_status,
            bootstrap_samples=bootstrap_samples,
        )
    if mode == "smoke":
        return run_a2h_smoke(project_root=root)
    if mode == "all":
        stages: dict[str, Any] = {
            "protocol": run_a2h_protocol(
                project_root=root,
                data_config_path=data_config_path,
                eval_config_path=eval_config_path,
                train_config_path=train_config_path,
            ),
            "generation": run_a2h_generation(project_root=root, data_config_path=data_config_path),
            "audit": run_a2h_difficulty_audit(
                project_root=root,
                data_config_path=data_config_path,
                eval_config_path=eval_config_path,
            ),
        }
        stages["learning_noise"] = run_a2h_learning_noise(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
            max_epochs_override=max_epochs_override,
        )
        stages["ood"] = run_a2h_ood(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
            max_epochs_override=max_epochs_override,
        )
        stages["comparison"] = run_a2h_algorithm_comparison(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
            max_epochs_override=max_epochs_override,
        )
        if unlock_hard_test:
            stages["formal"] = run_a2h_formal(
                project_root=root,
                data_config_path=data_config_path,
                eval_config_path=eval_config_path,
                train_config_path=train_config_path,
                unlock_hard_test=True,
                selection_record_path=selection_record_path,
                selected_checkpoint_path=selected_checkpoint_path,
                formal_run_status=formal_run_status,
                bootstrap_samples=bootstrap_samples,
            )
        terminal_stage = stages.get("formal", stages["comparison"])
        return {"stage": "A2H-all", "status": terminal_stage["status"], "stages": stages}
    raise NotImplementedError(
        "supported A2H modes are protocol, generate, audit, learning-noise, ood, compare, formal, smoke, and all"
    )


def run_a2h_protocol(
    *,
    project_root: str | Path | None = None,
    data_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    experiment_config_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    paths = _default_config_paths(root, data_config_path, eval_config_path, train_config_path)
    data_config = _read_json_object(paths["data_config"])
    eval_config = _read_json_object(paths["eval_config"])
    train_config = _read_json_object(paths["train_config"])
    validate_a2h_data_config(data_config)
    validate_a2h_eval_config(eval_config)
    validate_a2h_train_config(train_config)
    model_path = _resolve_file(root, root / "configs" / "model" / "a2h_candidate.json")
    model_config = _read_json_object(model_path)
    validate_a2h_model_config(model_config)
    if not _parameter_parity_report(train_config=train_config, model_config=model_config)["within_tolerance"]:
        raise A2HProtocolError("A2H C1 and M1 exceed the registered parameter-match tolerance")
    a1_reference = _validate_a1_reference(root, data_config)
    default_experiments = (
        root / "configs" / "experiment" / "a2h_protocol.json",
        root / "configs" / "experiment" / "a2h_difficulty_audit.json",
        root / "configs" / "experiment" / "a2h_learning_noise.json",
        root / "configs" / "experiment" / "a2h_ood.json",
        root / "configs" / "experiment" / "a2h_algorithm.json",
        root / "configs" / "experiment" / "a2h_formal.json",
    )
    experiment_paths = tuple(
        _resolve_file(root, Path(path)) for path in (experiment_config_paths or default_experiments)
    )
    if not experiment_paths:
        raise A2HProtocolError("at least one A2H experiment config is required")
    config_paths: dict[str, Path] = {
        "data_config": paths["data_config"],
        "eval_config": paths["eval_config"],
        "train_config": paths["train_config"],
        "model_config": model_path,
    }
    for experiment_path in experiment_paths:
        experiment = _read_json_object(experiment_path)
        validate_a2h_experiment_config(experiment)
        _validate_experiment_bindings(
            experiment,
            root=root,
            data_config_path=paths["data_config"],
            eval_config_path=paths["eval_config"],
            train_config_path=paths["train_config"],
        )
        experiment_id = str(experiment["experiment_id"])
        config_paths[experiment_id] = experiment_path
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
                config_paths[f"{experiment_id}.{key}"] = _resolve_file(root, Path(value))

    data_manifest_path = _a2h_data_dir(root) / "manifest.json"
    run_dir = _a2h_run_root(root) / "a2h-0-protocol"
    summary_dir = _a2h_summary_dir(root)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_a2h_run_manifest(
        project_root=root,
        stage="A2H-0",
        config_paths=config_paths,
        data_manifest_path=data_manifest_path if data_manifest_path.is_file() else None,
        status="PASS",
        test_unlocked=False,
    )
    manifest["a1_reference"] = a1_reference
    _write_json(run_dir / "manifest.json", manifest)
    protocol = {
        "schema_version": A2H_PROTOCOL_SCHEMA_VERSION,
        "stage": "A2H-0",
        "status": "PASS",
        "data_contract": {
            "schema_version": data_config["schema_version"],
            "data_version": data_config["data_version"],
            "dataset_id": data_config["dataset_id"],
            "split_families": sorted(data_config["families"]),
            "hard_test_default": "locked",
        },
        "a1_reference": a1_reference,
        "config_hashes": manifest["config_hashes"],
        "run_manifest": _relative_path(root, run_dir / "manifest.json"),
        "status": "FROZEN_PROTOCOL",
    }
    _write_json(summary_dir / "a2h_protocol.json", protocol)
    return {
        "stage": "A2H-0",
        "status": "PASS",
        "run_dir": str(run_dir),
        "summary_path": str(summary_dir / "a2h_protocol.json"),
        "manifest": manifest,
    }


def run_a2h_generation(
    *,
    project_root: str | Path | None = None,
    data_config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    protocol = run_a2h_protocol(project_root=root, data_config_path=data_config_path)
    data_path = _resolve_file(
        root,
        Path(data_config_path) if data_config_path is not None else root / A2H_DEFAULT_DATA_CONFIG,
    )
    data_config = _read_json_object(data_path)
    output_dir = _a2h_data_dir(root)
    dataset = generate_a2h_dataset(output_dir, config=data_config)
    a1_config_path = _resolve_file(root, root / "configs" / "data" / "ar_he_co2_a1_v1.json")
    a1_config = _read_json_object(a1_config_path)
    a1_manifest_path = _resolve_file(root, root / "data" / "a1_formal" / "manifest.json")
    a1_manifest = _read_json_object(a1_manifest_path)
    parity = nominal_signal_parity(
        [
            (condition["x_Ar_pct"], condition["x_He_pct"], condition["x_CO2_pct"])
            for condition in a1_manifest["conditions"]
        ],
        a1_physics=A1PhysicsConfig.from_mapping(a1_config["physics"]),
    )
    tolerance = float(data_config["nominal_parity"]["absolute_tolerance"])
    if parity["max_absolute_difference"] > tolerance:
        raise A2HProtocolError(
            f"A2H nominal deterministic signal parity failed: {parity['max_absolute_difference']} > {tolerance}"
        )
    manifest = _read_json_object(output_dir / "manifest.json")
    summary_dir = _a2h_summary_dir(root)
    run_dir = _a2h_run_root(root) / "a2h-1-generation"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    parity_payload = {
        "schema_version": "gf-a2h-nominal-parity-1",
        "status": "PASS",
        "reference_data_version": a1_manifest["data_version"],
        "a2h_data_version": manifest["data_version"],
        "tolerance": tolerance,
        **parity,
    }
    _write_json(summary_dir / "nominal_parity.json", parity_payload)
    run_manifest = build_a2h_run_manifest(
        project_root=root,
        stage="A2H-1",
        config_paths={"data_config": data_path},
        data_manifest_path=output_dir / "manifest.json",
        status="PASS",
        test_unlocked=False,
    )
    run_manifest["nominal_parity"] = parity_payload
    _write_json(run_dir / "manifest.json", run_manifest)
    return {
        "stage": "A2H-1",
        "status": "PASS",
        "protocol": protocol,
        "data_dir": str(output_dir),
        "manifest": manifest,
        "nominal_parity": parity_payload,
        "run_manifest": run_manifest,
    }


def run_a2h_difficulty_audit(
    *,
    project_root: str | Path | None = None,
    data_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    _ensure_data_current(root, data_config_path)
    eval_path = _resolve_file(
        root,
        Path(eval_config_path) if eval_config_path is not None else root / "configs" / "eval" / "a2h_eval.json",
    )
    eval_config = _read_json_object(eval_path)
    validate_a2h_eval_config(eval_config)
    dataset = load_a2h_dataset(_a2h_data_dir(root))
    audit = run_difficulty_audit(dataset, eval_config=eval_config)
    summary_dir = _a2h_summary_dir(root)
    run_dir = _a2h_run_root(root) / "a2h-2-difficulty-audit"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_dir / "a2h_difficulty_audit.json", audit)
    eligible_payload = {
        "schema_version": "gf-a2h-eligible-axes-1",
        "eligible_axes": audit["eligible_axes"],
        "minimum_eligible_axes": audit["minimum_eligible_axes"],
        "audit_status": audit["status"],
    }
    _write_json(summary_dir / "eligible_axes.json", eligible_payload)
    run_manifest = build_a2h_run_manifest(
        project_root=root,
        stage="A2H-2",
        config_paths={"eval_config": eval_path},
        data_manifest_path=_a2h_data_dir(root) / "manifest.json",
        status=audit["status"],
        test_unlocked=False,
    )
    run_manifest["eligible_axes"] = audit["eligible_axes"]
    run_manifest["eligible_axes_sha256"] = sha256_file(summary_dir / "eligible_axes.json")
    _write_json(run_dir / "manifest.json", run_manifest)
    return {
        "stage": "A2H-2",
        "status": audit["status"],
        "audit": audit,
        "eligible_axes_path": str(summary_dir / "eligible_axes.json"),
        "manifest": run_manifest,
    }


def run_a2h_learning_noise(
    *,
    project_root: str | Path | None = None,
    data_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    _ensure_data_current(root, data_config_path)
    eval_path = _resolve_file(
        root,
        Path(eval_config_path) if eval_config_path is not None else root / "configs" / "eval" / "a2h_eval.json",
    )
    eval_config = _read_json_object(eval_path)
    validate_a2h_eval_config(eval_config)
    train_path, train_config, model_path, model_config = _load_training_bundle(
        root,
        train_config_path=train_config_path,
    )
    dataset = load_a2h_dataset(_a2h_data_dir(root))
    fractions = tuple(float(value) for value in eval_config["learning_curve"]["fractions"])
    iid_train = dataset.indices(split_family="iid", split="train")
    iid_val = dataset.indices(split_family="iid", split="val")
    nested_groups = _nested_group_order(dataset, iid_train, seed=int(eval_config["bootstrap_seed"]))
    learning_rows: list[dict[str, Any]] = []
    for fraction in fractions:
        group_count = max(1, int(round(len(nested_groups) * fraction)))
        selected = set(nested_groups[:group_count])
        train_indices = np.asarray(
            [index for index in iid_train if dataset.observations[int(index)].mixture_id in selected],
            dtype=np.int64,
        )
        method_rows: dict[str, Any] = {}
        for model_id, seeds in (("B3", (17,)), ("B4", (17,)), ("B5", A2H_TRAINING_SEEDS)):
            records: list[dict[str, Any]] = []
            for seed in seeds:
                fit = _fit_model(
                    dataset,
                    train_indices,
                    iid_val,
                    iid_val,
                    model_id=model_id,
                    seed=seed,
                    head_id="H0",
                    train_config=train_config,
                    model_config=model_config,
                    max_epochs_override=max_epochs_override,
                )
                records.append(
                    {
                        "seed": int(seed),
                        "validation": _score_prediction(dataset, iid_val, fit.prediction),
                        "resources": dict(fit.resources),
                    }
                )
            method_rows[model_id] = _summarize_records(records)
        learning_rows.append(
            {
                "fraction": fraction,
                "train_group_count": group_count,
                "models": method_rows,
            }
        )

    noise_family = "noise"
    noise_indices_by_split = {
        split: dataset.indices(split_family=noise_family, split=split)
        for split in A2H_DEVELOPMENT_SPLITS
    }
    noise_train = dataset.indices(split_family=noise_family, split="train")
    noise_val = dataset.indices(split_family=noise_family, split="val")
    noise_response: dict[str, Any] = {}
    for split, indices in noise_indices_by_split.items():
        if len(indices) == 0:
            continue
        profile_ids = sorted({dataset.observations[int(index)].noise_profile_id for index in indices})
        profile_rows: dict[str, Any] = {}
        for profile_id in profile_ids:
            profile_indices = np.asarray(
                [index for index in indices if dataset.observations[int(index)].noise_profile_id == profile_id],
                dtype=np.int64,
            )
            seed_rows = []
            for seed in A2H_TRAINING_SEEDS:
                fit = _fit_model(
                    dataset,
                    noise_train,
                    noise_val,
                    profile_indices,
                    model_id="B5",
                    seed=seed,
                    head_id="H0",
                    train_config=train_config,
                    model_config=model_config,
                    max_epochs_override=max_epochs_override,
                )
                seed_rows.append(
                    {
                        "seed": int(seed),
                        "single_observation": _score_prediction(dataset, profile_indices, fit.prediction, semantics="single_observation"),
                        "same_mixture_repeat_mean": _score_prediction(dataset, profile_indices, fit.prediction, semantics="same_mixture_repeat_mean"),
                        "no_average": _score_prediction(dataset, profile_indices, fit.prediction, semantics="no_average"),
                    }
                )
            profile_rows[profile_id] = _summarize_noise_records(seed_rows)
        noise_response[split] = profile_rows
    result = {
        "schema_version": "gf-a2h-learning-noise-1",
        "stage": "A2H-3",
        "status": "PASS",
        "learning_curve": learning_rows,
        "noise_response": noise_response,
        "hard_test_read": False,
        "selection_rule": "iid validation only; no hard_test feedback",
    }
    summary_dir = _a2h_summary_dir(root)
    run_dir = _a2h_run_root(root) / "a2h-3-learning-noise"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_dir / "a2h_learning_noise.json", result)
    manifest = build_a2h_run_manifest(
        project_root=root,
        stage="A2H-3",
        config_paths={"eval_config": eval_path, "train_config": train_path, "model_config": model_path},
        data_manifest_path=_a2h_data_dir(root) / "manifest.json",
        status="PASS",
        test_unlocked=False,
    )
    _write_json(run_dir / "manifest.json", manifest)
    result["manifest"] = manifest
    return result


def run_a2h_ood(
    *,
    project_root: str | Path | None = None,
    data_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    _ensure_data_current(root, data_config_path)
    eval_path = _resolve_file(
        root,
        Path(eval_config_path) if eval_config_path is not None else root / "configs" / "eval" / "a2h_eval.json",
    )
    eval_config = _read_json_object(eval_path)
    validate_a2h_eval_config(eval_config)
    train_path, train_config, model_path, model_config = _load_training_bundle(
        root,
        train_config_path=train_config_path,
    )
    dataset = load_a2h_dataset(_a2h_data_dir(root))
    train_indices = dataset.indices(split_family="iid", split="train")
    val_indices = dataset.indices(split_family="iid", split="val")
    axis_names = ("environment", "calibration", "composition", "joint")
    axes: dict[str, Any] = {}
    for axis_name in axis_names:
        axis_train_indices = dataset.indices(split_family=axis_name, split="train")
        stress_indices = dataset.indices(split_family=axis_name, split="stress_val")
        family_val_indices = dataset.indices(split_family=axis_name, split="val")
        if len(stress_indices) == 0:
            continue
        model_results: dict[str, Any] = {}
        for model_id, seeds in (("B3", (17,)), ("B4", (17,)), ("B5", A2H_TRAINING_SEEDS)):
            records = []
            for seed in seeds:
                combined = np.concatenate((val_indices, family_val_indices, stress_indices))
                fit = _fit_model(
                    dataset,
                    axis_train_indices,
                    family_val_indices,
                    combined,
                    model_id=model_id,
                    seed=seed,
                    head_id="H0",
                    train_config=train_config,
                    model_config=model_config,
                    max_epochs_override=max_epochs_override,
                )
                first = len(val_indices)
                second = first + len(family_val_indices)
                records.append(
                    {
                        "seed": int(seed),
                        "iid_val": _score_prediction(dataset, val_indices, fit.prediction[:first]),
                        "family_val": _score_prediction(dataset, family_val_indices, fit.prediction[first:second]),
                        "stress_val": _score_prediction(dataset, stress_indices, fit.prediction[second:]),
                        "stress_stratification": _stratified_metrics(dataset, stress_indices, fit.prediction[second:]),
                        "resources": dict(fit.resources),
                    }
                )
            model_results[model_id] = _summarize_records(records, metric_key="stress_val")
        axes[axis_name] = {
            "models": model_results,
            "stress_stratification": _stratification_counts(dataset, stress_indices),
            "geometric_distance": _geometric_distance_summary(dataset, axis_train_indices, stress_indices),
        }

    environment_indices = dataset.indices(split_family="environment", split="stress_val")
    if len(environment_indices):
        context_results: dict[str, Any] = {}
        context_train = dataset.indices(split_family="environment", split="train")
        context_val = dataset.indices(split_family="environment", split="val")
        combined = np.concatenate((val_indices, environment_indices))
        for model_id in ("B5", "C1", "M1"):
            model_arms: dict[str, Any] = {}
            for arm, include_context in (("ENV-C0", False), ("ENV-C1", True)):
                rows = []
                for seed in A2H_TRAINING_SEEDS:
                    fit = _fit_model(
                        dataset,
                        context_train,
                        context_val,
                        combined,
                        model_id=model_id,
                        seed=seed,
                        head_id="H0",
                        train_config=train_config,
                        model_config=model_config,
                        include_context=include_context,
                        max_epochs_override=max_epochs_override,
                    )
                    rows.append(
                        {
                            "seed": int(seed),
                            "iid_val": _score_prediction(dataset, val_indices, fit.prediction[: len(val_indices)]),
                            "stress_val": _score_prediction(dataset, environment_indices, fit.prediction[len(val_indices) :]),
                            "resources": dict(fit.resources),
                        }
                    )
                model_arms[arm] = _summarize_records(rows, metric_key="stress_val")
            context_results[model_id] = model_arms
        axes["environment"]["matched_context_arms"] = context_results

    if "composition" in axes:
        composition_indices = dataset.indices(split_family="composition", split="stress_val")
        composition_train = dataset.indices(split_family="composition", split="train")
        composition_val = dataset.indices(split_family="composition", split="val")
        head_results: dict[str, Any] = {}
        for head_id in ("H0", "H1"):
            rows = []
            for seed in A2H_TRAINING_SEEDS:
                fit = _fit_model(
                    dataset,
                    composition_train,
                    composition_val,
                    composition_indices,
                    model_id="B5",
                    seed=seed,
                    head_id=head_id,
                    train_config=train_config,
                    model_config=model_config,
                    max_epochs_override=max_epochs_override,
                )
                rows.append(
                    {
                        "seed": int(seed),
                        "stress_val": _score_prediction(dataset, composition_indices, fit.prediction),
                        "stratified": _stratified_metrics(dataset, composition_indices, fit.prediction),
                    }
                )
            head_results[head_id] = _summarize_records(rows, metric_key="stress_val")
        axes["composition"]["output_heads"] = head_results

    result = {
        "schema_version": "gf-a2h-ood-1",
        "stage": "A2H-4-5",
        "status": "PASS",
        "axes": axes,
        "hard_test_read": False,
        "context_interpretation": "ENV-C1 minus ENV-C0 is context information gain only",
        "selection_rule": "stress_val reports are diagnostic; hard_test remains locked",
    }
    summary_dir = _a2h_summary_dir(root)
    run_dir = _a2h_run_root(root) / "a2h-4-5-ood"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_dir / "a2h_ood.json", result)
    manifest = build_a2h_run_manifest(
        project_root=root,
        stage="A2H-4-5",
        config_paths={"eval_config": eval_path, "train_config": train_path, "model_config": model_path},
        data_manifest_path=_a2h_data_dir(root) / "manifest.json",
        status="PASS",
        test_unlocked=False,
    )
    _write_json(run_dir / "manifest.json", manifest)
    result["manifest"] = manifest
    return result


def run_a2h_algorithm_comparison(
    *,
    project_root: str | Path | None = None,
    data_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    _ensure_data_current(root, data_config_path)
    audit_result = _load_or_run_audit(root, eval_config_path=eval_config_path)
    audit = audit_result["audit"] if "audit" in audit_result else audit_result
    if len(audit["eligible_axes"]) < int(audit["minimum_eligible_axes"]):
        return {
            "stage": "A2H-6",
            "status": "STOPPED_INSUFFICIENT_ELIGIBLE_AXES",
            "eligible_axes": audit["eligible_axes"],
            "hard_test_read": False,
        }
    eval_path = _resolve_file(
        root,
        Path(eval_config_path) if eval_config_path is not None else root / "configs" / "eval" / "a2h_eval.json",
    )
    train_path, train_config, candidate_path, model_config = _load_training_bundle(
        root,
        train_config_path=train_config_path,
    )
    eval_config = _read_json_object(eval_path)
    validate_a2h_eval_config(eval_config)
    dataset = load_a2h_dataset(_a2h_data_dir(root))
    model_matrix: dict[str, Any] = {}
    for axis_name in audit["eligible_axes"]:
        train_indices = dataset.indices(split_family=axis_name, split="train")
        val_indices = dataset.indices(split_family=axis_name, split="val")
        stress_indices = dataset.indices(split_family=axis_name, split="stress_val")
        if len(stress_indices) == 0:
            continue
        axis_models: dict[str, Any] = {}
        for model_id in ("B3", "B4", "B5", "C1", "M1"):
            axis_models[model_id] = _run_axis_model_records(
                dataset,
                train_indices,
                val_indices,
                stress_indices,
                model_id=model_id,
                seeds=(17,) if model_id in {"B3", "B4"} else A2H_TRAINING_SEEDS,
                train_config=train_config,
                model_config=model_config,
                max_epochs_override=max_epochs_override,
            )
        if axis_name == "composition":
            for head_id in ("H0", "H1"):
                axis_models[head_id] = _run_axis_model_records(
                    dataset,
                    train_indices,
                    val_indices,
                    stress_indices,
                    model_id="B5",
                    seeds=A2H_TRAINING_SEEDS,
                    head_id=head_id,
                    train_config=train_config,
                    model_config=model_config,
                    max_epochs_override=max_epochs_override,
                )
        model_matrix[axis_name] = axis_models

    candidate_gates = {
        candidate: _promotion_gate(
            model_matrix,
            eligible_axes=audit["eligible_axes"],
            candidate=candidate,
            baseline="B5",
            thresholds=eval_config["promotion"],
        )
        for candidate in ("C1", "M1")
    }
    passing_candidates = [candidate for candidate, gate in candidate_gates.items() if gate["status"] == "PASS"]
    selected_model = passing_candidates[0] if passing_candidates else "B5"
    comparison_status = "POSITIVE_RESULT" if passing_candidates else "NEGATIVE_RESULT"
    comparison = {
        "schema_version": "gf-a2h-algorithm-comparison-1",
        "stage": "A2H-6",
        "status": comparison_status,
        "eligible_axes": audit["eligible_axes"],
        "models": model_matrix,
        "parameter_parity": _parameter_parity_report(
            train_config=train_config,
            model_config=model_config,
        ),
        "promotion": {
            "candidates": candidate_gates,
            "selected_model": selected_model,
            "selection_rule": "first passing registered candidate C1, M1; otherwise frozen B5",
        },
        "hard_test_read": False,
    }
    summary_dir = _a2h_summary_dir(root)
    run_dir = _a2h_run_root(root) / "a2h-6-algorithm-comparison"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_dir / "a2h_algorithm_comparison.json", comparison)
    baseline_path = root / "configs" / "model" / "a2h_matched_baselines.json"
    chart_path = root / "configs" / "experiment" / "a2h_primary_chart_template.json"
    checkpoint_path = _a2h_run_root(root) / "selected_checkpoint.json"
    checkpoint_payload = {
        "schema_version": "gf-a2h-checkpoint-descriptor-1",
        "artifact_kind": "frozen_reproducible_selection_descriptor",
        "selected_model": selected_model,
        "status": comparison_status,
        "eligible_axes": audit["eligible_axes"],
        "training_seeds": list(A2H_TRAINING_SEEDS),
        "stress_val_selection_complete": True,
        "hard_test_read": False,
    }
    _write_json(checkpoint_path, checkpoint_payload)
    data_manifest = _read_json_object(_a2h_data_dir(root) / "manifest.json")
    selection_payload = {
        "schema_version": "gf-a2h-selection-1",
        "selected_model": selected_model,
        "comparison_status": comparison_status,
        "data_content_sha256": data_manifest["content_sha256"],
        "split_family_hash": data_manifest["split_family_hash"],
        "eligible_axes_sha256": sha256_file(summary_dir / "eligible_axes.json"),
        "candidate_config_sha256": sha256_file(candidate_path),
        "matched_baseline_config_sha256": sha256_file(baseline_path),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "primary_chart_template_sha256": sha256_file(chart_path),
        "formal_run_status": A2H_FORMAL_STATUS,
    }
    _write_json(_a2h_run_root(root) / "selection.json", selection_payload)
    manifest = build_a2h_run_manifest(
        project_root=root,
        stage="A2H-6",
        config_paths={
            "eval_config": eval_path,
            "train_config": train_path,
            "candidate_config": candidate_path,
            "baseline_config": baseline_path,
        },
        data_manifest_path=_a2h_data_dir(root) / "manifest.json",
        status=comparison_status,
        test_unlocked=False,
    )
    _write_json(run_dir / "manifest.json", manifest)
    comparison["manifest"] = manifest
    comparison["selection_record"] = selection_payload
    return comparison


def run_a2h_formal(
    *,
    project_root: str | Path | None = None,
    data_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    unlock_hard_test: bool = False,
    selection_record_path: str | Path | None = None,
    selected_checkpoint_path: str | Path | None = None,
    formal_run_status: str | None = None,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    if not unlock_hard_test:
        return {
            "stage": "A2H-7",
            "status": "LOCKED",
            "hard_test_read": False,
            "required_evidence": list(A2H_REQUIRED_HARD_EVIDENCE),
        }
    audit = _load_or_run_audit(root, eval_config_path=eval_config_path)
    audit_payload = audit["audit"] if "audit" in audit else audit
    if len(audit_payload["eligible_axes"]) < int(audit_payload["minimum_eligible_axes"]):
        raise A2HTestUnlockError("hard_test unlock is forbidden when fewer than two axes are eligible")
    comparison_path = _a2h_summary_dir(root) / "a2h_algorithm_comparison.json"
    if not comparison_path.is_file():
        run_a2h_algorithm_comparison(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
        )
    data_manifest_path = _a2h_data_dir(root) / "manifest.json"
    data_manifest = _read_json_object(data_manifest_path)
    summary_dir = _a2h_summary_dir(root)
    selection_path = _resolve_file(
        root,
        Path(selection_record_path)
        if selection_record_path is not None
        else _a2h_run_root(root) / "selection.json",
    )
    checkpoint_path = _resolve_file(
        root,
        Path(selected_checkpoint_path)
        if selected_checkpoint_path is not None
        else _a2h_run_root(root) / "selected_checkpoint.json",
    )
    train_path, train_config, candidate_path, model_config = _load_training_bundle(
        root,
        train_config_path=train_config_path,
    )
    baseline_path = _resolve_file(root, root / "configs" / "model" / "a2h_matched_baselines.json")
    chart_path = _resolve_file(root, root / "configs" / "experiment" / "a2h_primary_chart_template.json")
    evidence = verify_hard_test_unlock_evidence(
        project_root=root,
        data_manifest_path=data_manifest_path,
        eligible_axes_path=summary_dir / "eligible_axes.json",
        candidate_config_path=candidate_path,
        matched_baseline_config_path=baseline_path,
        selected_checkpoint_path=checkpoint_path,
        primary_chart_template_path=chart_path,
        selection_record_path=selection_path,
        formal_run_status=formal_run_status,
    )
    eval_path = _resolve_file(
        root,
        Path(eval_config_path) if eval_config_path is not None else root / "configs" / "eval" / "a2h_eval.json",
    )
    eval_config = _read_json_object(eval_path)
    validate_a2h_eval_config(eval_config)
    selected_model = str(evidence["selected_model"])
    development_dataset = load_a2h_dataset(_a2h_data_dir(root))
    for axis in A2H_SPLIT_FAMILIES:
        train_indices = development_dataset.indices(split_family=axis, split="train")
        val_indices = development_dataset.indices(split_family=axis, split="val")
        for seed in A2H_TRAINING_SEEDS:
            _fit_model(
                development_dataset,
                train_indices,
                val_indices,
                val_indices,
                model_id=selected_model,
                seed=seed,
                head_id="H0",
                train_config=train_config,
                model_config=model_config,
            )
            if selected_model != "B5":
                _fit_model(
                    development_dataset,
                    train_indices,
                    val_indices,
                    val_indices,
                    model_id="B5",
                    seed=seed,
                    head_id="H0",
                    train_config=train_config,
                    model_config=model_config,
                )
    access_ledger_path = _a2h_run_root(root) / "hard_test_access.json"
    access_claim = _claim_hard_test_access(access_ledger_path, evidence=evidence)
    dataset = load_a2h_dataset(_a2h_data_dir(root), include_hard_test=True)
    selected_seed_predictions: dict[str, list[np.ndarray]] = {axis: [] for axis in A2H_SPLIT_FAMILIES}
    baseline_seed_predictions: dict[str, list[np.ndarray]] = {axis: [] for axis in A2H_SPLIT_FAMILIES}
    for axis in A2H_SPLIT_FAMILIES:
        train_indices = dataset.indices(split_family=axis, split="train")
        val_indices = dataset.indices(split_family=axis, split="val")
        hard_indices = dataset.indices(split_family=axis, split="hard_test")
        if len(hard_indices) == 0:
            continue
        for seed in A2H_TRAINING_SEEDS:
            selected_fit = _fit_model(
                dataset,
                train_indices,
                val_indices,
                hard_indices,
                model_id=selected_model,
                seed=seed,
                head_id="H0",
                train_config=train_config,
                model_config=model_config,
            )
            baseline_fit = selected_fit if selected_model == "B5" else _fit_model(
                dataset,
                train_indices,
                val_indices,
                hard_indices,
                model_id="B5",
                seed=seed,
                head_id="H0",
                train_config=train_config,
                model_config=model_config,
            )
            selected_seed_predictions[axis].append(selected_fit.prediction)
            baseline_seed_predictions[axis].append(baseline_fit.prediction)

    hard_axes: dict[str, Any] = {}
    bootstrap_count = int(eval_config["bootstrap_samples"] if bootstrap_samples is None else bootstrap_samples)
    for axis in A2H_SPLIT_FAMILIES:
        hard_indices = dataset.indices(split_family=axis, split="hard_test")
        if len(hard_indices) == 0:
            continue
        selected_prediction = np.mean(selected_seed_predictions[axis], axis=0)
        baseline_prediction = np.mean(baseline_seed_predictions[axis], axis=0)
        targets = _targets(dataset, hard_indices)
        groups = _groups(dataset, hard_indices)
        hard_axes[axis] = {
            "selected_model": _score_prediction(dataset, hard_indices, selected_prediction),
            "matched_B5_baseline": _score_prediction(dataset, hard_indices, baseline_prediction),
            "selected_output_constraints": evaluate_output_constraints(selected_prediction, targets=targets),
            "matched_B5_output_constraints": evaluate_output_constraints(baseline_prediction, targets=targets),
            "stratification": _stratified_metrics(dataset, hard_indices, selected_prediction),
            "bootstrap": group_bootstrap_comparison(
                selected_prediction,
                baseline_prediction,
                targets,
                groups,
                seed=int(eval_config["bootstrap_seed"]),
                samples=bootstrap_count,
            ),
        }

    hard_gate = _hard_test_gate(
        hard_axes,
        selected_model=selected_model,
        thresholds=eval_config["promotion"],
        eligible_axes=audit_payload["eligible_axes"],
    )
    comparison = _read_json_object(comparison_path)
    if selected_model != "B5" and comparison.get("status") == "POSITIVE_RESULT" and hard_gate["status"] == "PASS":
        final_status = "POSITIVE_RESULT"
    elif _benchmark_saturated(audit_payload):
        final_status = "BENCHMARK_SATURATED"
    else:
        final_status = "NEGATIVE_RESULT"
    result = {
        "schema_version": "gf-a2h-formal-1",
        "stage": "A2H-7",
        "status": final_status,
        "formal_run_status": formal_run_status,
        "selected_model": selected_model,
        "eligible_axes": audit_payload["eligible_axes"],
        "hard_test_read": True,
        "hard_test_axes": hard_axes,
        "hard_test_gate": hard_gate,
        "evidence": evidence,
        "hard_test_access_claim": access_claim,
        "no_tuning_after_unlock": True,
        "bootstrap_samples": bootstrap_count,
    }
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _a2h_run_root(root) / "a2h-7-formal"
    report_dir = _a2h_report_dir(root)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(summary_dir / "a2h_formal.json", result)
    manifest = build_a2h_run_manifest(
        project_root=root,
        stage="A2H-7",
        config_paths={
            "eval_config": eval_path,
            "train_config": train_path,
            "candidate_config": candidate_path,
            "baseline_config": baseline_path,
            "primary_chart_template": chart_path,
            "selection_record": selection_path,
            "selected_checkpoint": checkpoint_path,
        },
        data_manifest_path=data_manifest_path,
        status=final_status,
        test_unlocked=True,
        unlock_evidence=evidence,
    )
    _write_json(run_dir / "manifest.json", manifest)
    result["manifest"] = manifest
    _write_formal_report(report_dir / "A2H正式报告.md", result, data_manifest)
    _write_formal_review(report_dir / "A2H评审记录.md", result)
    return result


def write_a2h_failure_cases(*, project_root: str | Path | None = None) -> dict[str, Any]:
    """Export failure cases from completed development and formal summaries only."""

    root = _project_root(project_root)
    summary_dir = _a2h_summary_dir(root)
    comparison = _read_json_object(summary_dir / "a2h_algorithm_comparison.json")
    formal = _read_json_object(summary_dir / "a2h_formal.json")
    candidate_failures: list[dict[str, Any]] = []
    promotion = comparison.get("promotion", {})
    candidates = promotion.get("candidates", {})
    if isinstance(candidates, Mapping):
        for candidate, gate in candidates.items():
            if not isinstance(gate, Mapping) or gate.get("status") == "PASS":
                continue
            axis_failures: list[dict[str, Any]] = []
            axis_results = gate.get("axis_results", {})
            if isinstance(axis_results, Mapping):
                for axis, row in axis_results.items():
                    if not isinstance(row, Mapping):
                        continue
                    reasons: list[str] = []
                    if float(row.get("relative_improvement", 0.0)) < 0.05:
                        reasons.append("relative_improvement_below_0.05")
                    if int(row.get("same_direction_seed_count", 0)) < 4:
                        reasons.append("fewer_than_four_same_direction_seeds")
                    if max(float(value) for value in row.get("component_delta", ())) > 0.005:
                        reasons.append("component_degradation_above_0.005")
                    if reasons:
                        axis_failures.append(
                            {
                                "axis": axis,
                                "reasons": reasons,
                                "relative_improvement": row.get("relative_improvement"),
                                "same_direction_seed_count": row.get("same_direction_seed_count"),
                                "component_delta": row.get("component_delta"),
                            }
                        )
            candidate_failures.append(
                {
                    "candidate": candidate,
                    "baseline": gate.get("baseline"),
                    "status": gate.get("status"),
                    "checks": gate.get("checks"),
                    "axis_failures": axis_failures,
                }
            )

    hard_axis_rows: list[dict[str, Any]] = []
    hard_axes = formal.get("hard_test_axes", {})
    if isinstance(hard_axes, Mapping):
        for axis, payload in hard_axes.items():
            if not isinstance(payload, Mapping):
                continue
            selected = payload.get("selected_model", {})
            baseline = payload.get("matched_B5_baseline", {})
            hard_axis_rows.append(
                {
                    "axis": axis,
                    "selected_model": formal.get("selected_model"),
                    "selected_macro_RNMAE": selected.get("macro_RNMAE"),
                    "matched_B5_macro_RNMAE": baseline.get("macro_RNMAE"),
                    "selected_worst_group_MAE": selected.get("worst_group_MAE"),
                    "matched_B5_worst_group_MAE": baseline.get("worst_group_MAE"),
                    "output_constraints": payload.get("selected_output_constraints"),
                }
            )
    if not hard_axis_rows:
        raise A2HProtocolError("formal summary has no hard-test axis for failure-case export")
    worst_axis = max(
        hard_axis_rows,
        key=lambda row: float(row["selected_macro_RNMAE"]),
    )
    result = {
        "schema_version": "gf-a2h-failure-cases-1",
        "stage": "A2H-7",
        "formal_status": formal.get("status"),
        "hard_test_read": formal.get("hard_test_read"),
        "candidate_failures": candidate_failures,
        "hard_test_axes": hard_axis_rows,
        "worst_hard_test_axis": worst_axis,
        "source_summaries": {
            "algorithm_comparison": f"outputs/summary/{A2H_OUTPUT_NAMESPACE}/a2h_algorithm_comparison.json",
            "formal": f"outputs/summary/{A2H_OUTPUT_NAMESPACE}/a2h_formal.json",
        },
    }
    _write_json(summary_dir / "a2h_failure_cases.json", result)
    return result


def run_a2h_smoke(*, project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    generation = run_a2h_generation(project_root=root)
    audit = run_a2h_difficulty_audit(project_root=root)
    return {
        "stage": "A2H-smoke",
        "status": "PASS" if audit["status"] == "PASS" else audit["status"],
        "generation": {"status": generation["status"], "data_version": generation["manifest"]["data_version"]},
        "audit": {"status": audit["status"], "eligible_axes": audit["audit"]["eligible_axes"]},
        "hard_test_read": False,
    }


def validate_a2h_data_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    if config.get("schema_version") != A2H_SCHEMA_VERSION:
        raise A2HProtocolError("A2H data config has unsupported schema_version")
    if config.get("dataset_id") != "ar_he_co2":
        raise A2HProtocolError("A2H data config must use dataset_id=ar_he_co2")
    if not str(config.get("data_version", "")).startswith(A2H_DATA_VERSION_PREFIX):
        raise A2HProtocolError("A2H data_version must use the gf-a2h-v2 namespace")
    if config.get("timesteps") != 1 or config.get("observation_mode") != "steady_state_repeated_observation":
        raise A2HProtocolError("A2H must use T=1 steady-state repeated observations")
    if config.get("sensor_ids") != list(SENSOR_IDS) or config.get("target_names") != list(TARGET_NAMES):
        raise A2HProtocolError("A2H sensor and target order must match the frozen contract")
    if float(config.get("composition_total_pct", 0.0)) != 100.0:
        raise A2HProtocolError("A2H composition_total_pct must be 100")
    A2HPhysicsConfig = _load_a2h_physics_type()
    A2HPhysicsConfig.from_mapping(config.get("physics", {}))
    reference = _required_mapping(config, "a1_reference")
    for key in ("manifest_path", "data_version", "content_sha256", "split_hash"):
        if not isinstance(reference.get(key), str) or not reference[key]:
            raise A2HProtocolError(f"a1_reference.{key} must be a non-empty string")
    _require_hash(reference, "content_sha256")
    _require_hash(reference, "split_hash")
    environments = config.get("environment_blocks")
    if not isinstance(environments, list) or not environments:
        raise A2HProtocolError("environment_blocks must be a non-empty list")
    environment_ids = [item.get("environment_id") for item in environments if isinstance(item, Mapping)]
    if len(environment_ids) != len(set(environment_ids)) or any(not value for value in environment_ids):
        raise A2HProtocolError("environment_id values must be unique and non-empty")
    noise_profiles = config.get("noise_profiles")
    if not isinstance(noise_profiles, list) or not noise_profiles:
        raise A2HProtocolError("noise_profiles must be a non-empty list")
    noise_ids = [item.get("noise_profile_id") for item in noise_profiles if isinstance(item, Mapping)]
    if len(noise_ids) != len(set(noise_ids)) or any(not value for value in noise_ids):
        raise A2HProtocolError("noise_profile_id values must be unique and non-empty")
    calibration_profiles = config.get("calibration_profiles")
    if not isinstance(calibration_profiles, list) or not calibration_profiles:
        raise A2HProtocolError("calibration_profiles must be a non-empty list")
    calibration_ids = [item.get("calibration_profile_id") for item in calibration_profiles if isinstance(item, Mapping)]
    if len(calibration_ids) != len(set(calibration_ids)) or any(not value for value in calibration_ids):
        raise A2HProtocolError("calibration_profile_id values must be unique and non-empty")
    families = config.get("families")
    expected_families = {"iid", "noise", "environment", "calibration", "composition", "joint"}
    if not isinstance(families, Mapping) or set(families) != expected_families:
        raise A2HProtocolError(f"families must be exactly {sorted(expected_families)}")
    valid_modes = {"mixed", "interior", "binary", "pure", "near_boundary", "concentration_band", "simplex_sector", "simplex_sector_and_pure"}
    for family_name, family in families.items():
        if not isinstance(family, Mapping):
            raise A2HProtocolError(f"families[{family_name!r}] must be an object")
        splits = family.get("splits")
        if not isinstance(splits, Mapping) or set(splits) != set(A2H_SPLITS):
            raise A2HProtocolError(f"families[{family_name!r}].splits must cover {list(A2H_SPLITS)}")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in splits.values()):
            raise A2HProtocolError(f"families[{family_name!r}].splits must contain positive integers")
        for mapping_key in ("composition_mode_by_split", "environment_by_split", "calibration_by_split", "noise_by_split"):
            mapping = family.get(mapping_key)
            if not isinstance(mapping, Mapping) or set(mapping) != set(A2H_SPLITS):
                raise A2HProtocolError(f"families[{family_name!r}].{mapping_key} must cover all A2H splits")
            for split, value in mapping.items():
                if isinstance(value, str) and value:
                    continue
                if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
                    continue
                raise A2HProtocolError(
                    f"families[{family_name!r}].{mapping_key}[{split!r}] must be a profile id or list"
                )
        modes = family["composition_mode_by_split"].values()
        if any(str(mode) not in valid_modes for mode in modes):
            raise A2HProtocolError(f"families[{family_name!r}] contains an unsupported composition mode")
        if isinstance(family.get("repeat_count"), bool) or not isinstance(family.get("repeat_count"), int) or family["repeat_count"] <= 0:
            raise A2HProtocolError(f"families[{family_name!r}].repeat_count must be positive")
    evidence = config.get("required_hard_test_evidence")
    if tuple(evidence or ()) != A2H_REQUIRED_HARD_EVIDENCE:
        raise A2HProtocolError("required_hard_test_evidence does not match the frozen hard-test lock")


def validate_a2h_eval_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    if config.get("schema_version") != A2H_EVAL_SCHEMA_VERSION or config.get("parent_schema_version") != "gf-eval-1":
        raise A2HProtocolError("A2H evaluation schema or parent schema is unsupported")
    if config.get("metric") != "macro_RNMAE":
        raise A2HProtocolError("A2H metric must be macro_RNMAE")
    ranges = config.get("target_ranges")
    if not isinstance(ranges, Mapping) or set(ranges) != set(TARGET_NAMES) or any(float(value) <= 0.0 for value in ranges.values()):
        raise A2HProtocolError("target_ranges must cover the three positive target ranges")
    if tuple(config.get("formal_training_seeds") or ()) != A2H_TRAINING_SEEDS:
        raise A2HProtocolError(f"formal_training_seeds must be {list(A2H_TRAINING_SEEDS)}")
    if config.get("bootstrap_samples") != 2000 or config.get("confidence_level") != 0.95:
        raise A2HProtocolError("A2H bootstrap_samples and confidence_level are frozen at 2000 and 0.95")
    if tuple(config.get("split_families") or ()) != ("iid", "noise", "environment", "calibration", "composition", "joint"):
        raise A2HProtocolError("split_families order is frozen")
    if tuple(config.get("development_splits") or ()) != A2H_DEVELOPMENT_SPLITS:
        raise A2HProtocolError("development_splits must exclude hard_test")
    if config.get("hard_test_split") != "hard_test":
        raise A2HProtocolError("hard_test_split must be hard_test")
    eligibility = _required_mapping(config, "eligibility")
    required_eligibility = {
        "min_jacobian_full_rank_fraction",
        "max_jacobian_p95_condition_number",
        "min_concat_relative_iid_degradation",
        "min_oracle_relative_concat_headroom",
        "max_outside_signal_bound_fraction",
        "min_eligible_axes",
    }
    if set(eligibility) != required_eligibility:
        raise A2HProtocolError("eligibility thresholds are incomplete or contain unregistered fields")
    if float(eligibility["min_jacobian_full_rank_fraction"]) != 0.99 or float(eligibility["max_jacobian_p95_condition_number"]) != 1000.0 or float(eligibility["min_concat_relative_iid_degradation"]) != 0.25 or float(eligibility["min_oracle_relative_concat_headroom"]) != 0.20 or int(eligibility["min_eligible_axes"]) != 2:
        raise A2HProtocolError("A2H eligibility thresholds do not match the registered gates")
    promotion = _required_mapping(config, "promotion")
    if float(promotion["relative_improvement"]) != 0.05 or int(promotion["min_seeds_same_direction"]) != 4 or float(promotion["max_component_absolute_degradation"]) != 0.005 or int(promotion["min_eligible_axes_same_direction"]) != 2:
        raise A2HProtocolError("A2H promotion thresholds do not match the registered gates")
    if promotion.get("primary_statistic") != "mean_of_independent_seed_stress_val_macro_RNMAE":
        raise A2HProtocolError("A2H primary statistic must be the independent-seed stress-val mean")
    test_access = _required_mapping(config, "test_access")
    if test_access.get("default") != "locked" or test_access.get("unlock_flag") != "--unlock-hard-test" or test_access.get("required_formal_run_status") != A2H_FORMAL_STATUS:
        raise A2HProtocolError("A2H hard-test access lock is not frozen")
    if tuple(test_access.get("required_evidence") or ()) != A2H_REQUIRED_HARD_EVIDENCE:
        raise A2HProtocolError("A2H hard-test evidence order is not frozen")


def validate_a2h_train_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    if config.get("schema_version") != A2H_TRAIN_SCHEMA_VERSION:
        raise A2HProtocolError("A2H training schema is unsupported")
    if tuple(config.get("seeds") or ()) != A2H_TRAINING_SEEDS:
        raise A2HProtocolError("A2H training seeds are not the frozen five seeds")
    optimizer = _required_mapping(config, "optimizer")
    if optimizer.get("name") not in {"Adam", "LBFGS"} or float(optimizer.get("learning_rate", 0.0)) <= 0.0 or float(optimizer.get("weight_decay", -1.0)) < 0.0:
        raise A2HProtocolError("A2H optimizer configuration is invalid")
    if optimizer.get("name") == "LBFGS" and float(optimizer.get("weight_decay")) != 0.0:
        raise A2HProtocolError("LBFGS A2H training must use weight_decay=0")
    loss = _required_mapping(config, "loss")
    if loss.get("name") != "mse" or tuple(float(value) for value in loss.get("target_scale", ())) != (100.0, 100.0, 100.0):
        raise A2HProtocolError("A2H loss must be MSE with target scale [100,100,100]")
    if isinstance(config.get("max_epochs"), bool) or not isinstance(config.get("max_epochs"), int) or config["max_epochs"] <= 0:
        raise A2HProtocolError("A2H max_epochs must be a positive integer")
    early_stopping = _required_mapping(config, "early_stopping")
    if early_stopping.get("test_access") != "forbidden" or early_stopping.get("selection_split") != "family.val":
        raise A2HProtocolError("A2H early stopping must be train/validation only")
    if config.get("hard_test_access") != "forbidden":
        raise A2HProtocolError("A2H training must forbid hard_test access")


def validate_a2h_model_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    if config.get("schema_version") != A2H_MODEL_SCHEMA_VERSION:
        raise A2HProtocolError("A2H model schema is unsupported")
    if config.get("sensor_ids") != list(SENSOR_IDS):
        raise A2HProtocolError("A2H model sensor_ids must match the frozen sensor order")
    sensor_types = config.get("sensor_types")
    if not isinstance(sensor_types, list) or len(sensor_types) != len(SENSOR_IDS) or any(not value for value in sensor_types):
        raise A2HProtocolError("A2H model sensor_types must cover every sensor")
    models = _required_mapping(config, "models")
    if set(models) != {"B5", "C1", "M1"}:
        raise A2HProtocolError("A2H model config must define exactly B5, C1 and M1")
    b5 = _required_mapping(models, "B5")
    if b5.get("kind") != "raw_scalar_mlp" or not isinstance(b5.get("hidden_dim"), int) or b5["hidden_dim"] <= 0:
        raise A2HProtocolError("A2H B5 definition is invalid")
    c1 = _required_mapping(models, "C1")
    if c1.get("kind") != "ordered_concat_common_head" or c1.get("representation") != "torch_concat":
        raise A2HProtocolError("A2H C1 must use the ordered torch concat representation")
    m1 = _required_mapping(models, "M1")
    if m1.get("kind") != "token_deepsets_common_head" or m1.get("representation") != "sensor_token" or m1.get("pool") != "masked_mean":
        raise A2HProtocolError("A2H M1 must use sensor tokens and masked-mean Deep Sets")
    for model_id, definition in (("C1", c1), ("M1", m1)):
        if not isinstance(definition.get("capacity_name"), str) or not definition["capacity_name"]:
            raise A2HProtocolError(f"A2H {model_id} capacity_name is required")


def validate_a2h_experiment_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    if config.get("schema_version") != A2H_EXPERIMENT_SCHEMA_VERSION:
        raise A2HProtocolError("A2H experiment schema is unsupported")
    for key in ("stage", "experiment_id", "kind", "data_config", "eval_config", "train_config", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise A2HProtocolError(f"A2H experiment field {key} must be a non-empty string")
    allowed = config.get("allowed_read_splits")
    if not isinstance(allowed, list) or any(split not in A2H_ALLOWED_READ_SPLITS for split in allowed):
        raise A2HProtocolError("A2H experiment allowed_read_splits contains an unregistered split")
    if config.get("hard_test_access") not in {"locked", "locked_until_evidence"}:
        raise A2HProtocolError("A2H experiment hard_test access must remain locked")
    if config.get("kind") in {"matched_algorithm_comparison", "formal_hard_test"}:
        for key in ("candidate_config", "baseline_config"):
            if not isinstance(config.get(key), str) or not config[key]:
                raise A2HProtocolError(f"{config['kind']} requires {key}")
    if config.get("kind") == "formal_hard_test" and not isinstance(config.get("primary_chart_template"), str):
        raise A2HProtocolError("formal_hard_test requires primary_chart_template")


def build_a2h_run_manifest(
    *,
    project_root: str | Path,
    stage: str,
    config_paths: Mapping[str, str | Path],
    data_manifest_path: str | Path | None,
    status: str,
    test_unlocked: bool,
    unlock_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if test_unlocked != (unlock_evidence is not None):
        raise A2HTestUnlockError("unlocked run manifest must contain unlock evidence and locked one must not")
    config_hashes = {
        str(name): sha256_file(_resolve_file(root, Path(path)))
        for name, path in sorted(config_paths.items())
    }
    data_version = None
    data_content_sha256 = None
    split_family_hash = None
    if data_manifest_path is not None:
        manifest_path = _resolve_file(root, Path(data_manifest_path))
        data_manifest = _read_json_object(manifest_path)
        data_version = data_manifest.get("data_version")
        data_content_sha256 = data_manifest.get("content_sha256")
        split_family_hash = data_manifest.get("split_family_hash")
        _require_hash(data_manifest, "content_sha256")
        _require_hash(data_manifest, "split_family_hash")
    revision, dirty = _git_state(root)
    return {
        "schema_version": A2H_RUN_MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        "worktree": {"revision": revision, "dirty": dirty},
        "config_hashes": config_hashes,
        "data_version": data_version,
        "data_content_sha256": data_content_sha256,
        "split_family_hash": split_family_hash,
        "test_access": {
            "default": "locked",
            "unlocked": test_unlocked,
            "evidence": dict(unlock_evidence) if unlock_evidence is not None else None,
        },
    }


def assert_hard_test_unlocked(
    *,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    required_formal_run_status: str = A2H_FORMAL_STATUS,
) -> None:
    for key in A2H_REQUIRED_HARD_EVIDENCE:
        if key not in actual or actual[key] in (None, ""):
            raise A2HTestUnlockError(f"hard-test evidence is missing {key}")
        if key != "formal_run_status":
            _validate_hash_value(str(actual[key]), key)
        if key not in expected or expected[key] in (None, ""):
            raise A2HTestUnlockError(f"selection record is missing {key}")
        if actual[key] != expected[key]:
            raise A2HTestUnlockError(f"hard-test evidence mismatch for {key}")
    if actual["formal_run_status"] != required_formal_run_status:
        raise A2HTestUnlockError(
            f"formal_run_status must be {required_formal_run_status}, got {actual['formal_run_status']!r}"
        )


def verify_hard_test_unlock_evidence(
    *,
    project_root: str | Path,
    data_manifest_path: str | Path,
    eligible_axes_path: str | Path,
    candidate_config_path: str | Path,
    matched_baseline_config_path: str | Path,
    selected_checkpoint_path: str | Path,
    primary_chart_template_path: str | Path,
    selection_record_path: str | Path,
    formal_run_status: str | None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    data_manifest = _read_json_object(_resolve_file(root, Path(data_manifest_path)))
    eligible_path = _resolve_file(root, Path(eligible_axes_path))
    candidate_path = _resolve_file(root, Path(candidate_config_path))
    baseline_path = _resolve_file(root, Path(matched_baseline_config_path))
    checkpoint_path = _resolve_file(root, Path(selected_checkpoint_path))
    chart_path = _resolve_file(root, Path(primary_chart_template_path))
    record = _read_json_object(_resolve_file(root, Path(selection_record_path)))
    actual = {
        "data_content_sha256": data_manifest.get("content_sha256"),
        "split_family_hash": data_manifest.get("split_family_hash"),
        "eligible_axes_sha256": sha256_file(eligible_path),
        "candidate_config_sha256": sha256_file(candidate_path),
        "matched_baseline_config_sha256": sha256_file(baseline_path),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "primary_chart_template_sha256": sha256_file(chart_path),
        "formal_run_status": formal_run_status,
    }
    assert_hard_test_unlocked(actual=actual, expected=record)
    if not isinstance(record.get("selected_model"), str) or not record["selected_model"]:
        raise A2HTestUnlockError("selection record must contain selected_model")
    actual["selected_model"] = record["selected_model"]
    return actual


def _claim_hard_test_access(
    path: Path,
    *,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "gf-a2h-hard-test-access-1",
        "status": "CLAIMED",
        "evidence_sha256": _canonical_sha256(evidence),
        "evidence": dict(evidence),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise A2HTestUnlockError(
            f"hard-test access was already claimed for this protocol: {path.as_posix()}"
        ) from exc
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_a2h_split_hash(value: A2HDataset | Mapping[str, Any]) -> str:
    return compute_split_family_hash(value)


def _fit_model(
    dataset: A2HDataset,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    eval_indices: Sequence[int],
    *,
    model_id: str,
    seed: int,
    head_id: str,
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    include_context: bool = False,
    max_epochs_override: int | None = None,
) -> _FitResult:
    if model_id not in {"B3", "B4", "B5", "C1", "M1"}:
        raise ValueError(f"unsupported A2H comparison model {model_id!r}")
    if head_id not in {"H0", "H1"}:
        raise ValueError(f"unsupported A2H output head {head_id!r}")
    validate_a2h_train_config(train_config)
    validate_a2h_model_config(model_config)
    feature_map = "deepsets" if model_id == "M1" else "raw"
    if model_id == "B3":
        train_features, feature_names = _feature_matrix(dataset, train_indices, feature_map="raw", include_context=include_context)
        eval_features, _ = _feature_matrix(dataset, eval_indices, feature_map="raw", include_context=include_context)
        feature_mean, feature_scale = _fit_feature_scaler(train_features, feature_names)
        estimator: Any = Ridge(alpha=1.0)
    elif model_id == "B4":
        train_features, feature_names = _feature_matrix(dataset, train_indices, feature_map="raw", include_context=include_context)
        eval_features, _ = _feature_matrix(dataset, eval_indices, feature_map="raw", include_context=include_context)
        feature_mean, feature_scale = _fit_feature_scaler(train_features, feature_names)
        estimator = MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=seed,
            )
        )
    else:
        return _fit_torch_model(
            dataset,
            train_indices,
            validation_indices,
            eval_indices,
            model_id=model_id,
            seed=seed,
            head_id=head_id,
            train_config=train_config,
            model_config=model_config,
            include_context=include_context,
            max_epochs_override=max_epochs_override,
        )
    train_scaled = (train_features - feature_mean) / feature_scale
    eval_scaled = (eval_features - feature_mean) / feature_scale
    started = time.perf_counter()
    estimator.fit(train_scaled, _targets(dataset, train_indices))
    training_time = time.perf_counter() - started
    started = time.perf_counter()
    prediction = np.asarray(estimator.predict(eval_scaled), dtype=np.float64)
    if head_id == "H1":
        prediction = _project_rows_to_simplex(prediction, total=100.0)
    inference_time = time.perf_counter() - started
    return _FitResult(
        model_id=model_id,
        seed=seed,
        head_id=head_id,
        feature_map=feature_map,
        include_context=include_context,
        prediction=prediction,
        resources={
            "training_time_s": float(training_time),
            "inference_time_s": float(inference_time),
            "parameter_count": _parameter_count(estimator),
            "feature_count": int(train_features.shape[1]),
            "constant_context_features": [],
        },
    )


def _fit_torch_model(
    dataset: A2HDataset,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    eval_indices: Sequence[int],
    *,
    model_id: str,
    seed: int,
    head_id: str,
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    include_context: bool,
    max_epochs_override: int | None,
) -> _FitResult:
    cache_key = (
        str(dataset.manifest["content_sha256"]),
        model_id,
        int(seed),
        head_id,
        bool(include_context),
        _index_group_hash(dataset, train_indices),
        _index_group_hash(dataset, validation_indices),
        _canonical_sha256(train_config),
        _canonical_sha256(model_config),
        max_epochs_override,
    )
    artifact = _TORCH_FIT_CACHE.get(cache_key)
    if artifact is None:
        train_samples = dataset.samples(train_indices)
        validation_samples = dataset.samples(validation_indices)
        context_statistics: dict[str, tuple[float, float]] = {}
        if include_context:
            context_statistics = _fit_context_scaler(train_samples)
            train_samples = [_with_scaled_context(sample, context_statistics) for sample in train_samples]
            validation_samples = [_with_scaled_context(sample, context_statistics) for sample in validation_samples]
        scaler = TrainGroupStandardScaler()
        scaler.fit(train_samples + validation_samples, {sample.group_id for sample in train_samples})
        train_scaled = [scaler.transform(sample) for sample in train_samples]
        validation_scaled = [scaler.transform(sample) for sample in validation_samples]
        model = _build_torch_model(
            model_id=model_id,
            head_id=head_id,
            train_config=train_config,
            model_config=model_config,
            include_context=include_context,
        )
        training = TorchTrainingConfig.from_mapping(train_config)
        if max_epochs_override is not None:
            if max_epochs_override <= 0:
                raise ValueError("max_epochs_override must be positive")
            training = replace(training, max_epochs=int(max_epochs_override))
        started = time.perf_counter()
        result = train_torch_model(
            model,
            train_scaled,
            validation_scaled,
            config=training,
            seed=seed,
        )
        training_time = time.perf_counter() - started
        artifact = _TorchArtifact(
            model=model,
            scaler=scaler,
            context_statistics=context_statistics,
            resources={
                "training_time_s": float(training_time),
                "parameter_count": trainable_parameter_count(model),
                "feature_count": len(train_scaled[0].sensor_id) + len(context_statistics),
                "best_epoch": int(result.best_epoch),
                "epochs_completed": int(result.epochs_completed),
                "context_statistics": {
                    key: {"mean": mean, "std": std}
                    for key, (mean, std) in context_statistics.items()
                },
            },
        )
        _TORCH_FIT_CACHE[cache_key] = artifact

    eval_samples = dataset.samples(eval_indices)
    if include_context:
        eval_samples = [_with_scaled_context(sample, artifact.context_statistics) for sample in eval_samples]
    eval_scaled = [artifact.scaler.transform(sample) for sample in eval_samples]
    started = time.perf_counter()
    with torch.no_grad():
        prediction = artifact.model(collate_samples(tuple(eval_scaled))).detach().cpu().numpy().astype(np.float64)
    if model_id == "B5" and head_id == "H1":
        prediction = _project_rows_to_simplex(prediction, total=100.0)
    inference_time = time.perf_counter() - started
    return _FitResult(
        model_id=model_id,
        seed=seed,
        head_id=head_id,
        feature_map="deepsets" if model_id == "M1" else "raw",
        include_context=include_context,
        prediction=prediction,
        resources={
            **artifact.resources,
            "inference_time_s": float(inference_time),
        },
    )


def _index_group_hash(dataset: A2HDataset, indices: Sequence[int]) -> str:
    return _canonical_sha256(
        [dataset.observations[int(index)].mixture_id for index in np.asarray(indices, dtype=np.int64)]
    )


def _build_torch_model(
    *,
    model_id: str,
    head_id: str,
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    include_context: bool,
) -> torch.nn.Module:
    models = _required_mapping(model_config, "models")
    definition = _required_mapping(models, model_id)
    context_keys = ("temperature_k_scaled", "pressure_pa_scaled") if include_context else ()
    if model_id == "B5":
        return TorchConcatMLP(
            sensor_count=len(SENSOR_IDS),
            hidden_dim=int(definition["hidden_dim"]),
            output_dim=len(TARGET_NAMES),
            context_keys=context_keys,
        )
    capacity_name = str(definition["capacity_name"])
    presets = train_config.get("capacity_presets")
    if not isinstance(presets, list):
        raise A2HProtocolError("A2H capacity_presets must be a list")
    matching = [preset for preset in presets if isinstance(preset, Mapping) and preset.get("name") == capacity_name]
    if len(matching) != 1:
        raise A2HProtocolError(f"capacity preset {capacity_name!r} must resolve exactly once")
    preset = matching[0]
    return A2FusionModel(
        representation=str(definition["representation"]),
        embedding_dim=int(preset["encoder_hidden_dim"]),
        fusion_hidden_dim=int(preset["fusion_hidden_dim"]),
        output_dim=len(TARGET_NAMES),
        sensor_ids=tuple(model_config["sensor_ids"]),
        sensor_types=tuple(model_config["sensor_types"]),
        head_id=head_id,
        pooling=str(definition.get("pool", "masked_mean")),
        max_sensors=len(SENSOR_IDS),
        concat_dim=int(definition["concat_dim"]) if definition.get("concat_dim") is not None else None,
        context_keys=context_keys,
    )


def _fit_feature_scaler(features: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    constant = [name for name, value in zip(names, scale, strict=True) if value <= 0.0]
    if constant:
        raise A2HProtocolError(f"A2H train scaler found zero-variance features: {constant}")
    return mean, scale


def _fit_context_scaler(samples: Sequence[UnifiedSample]) -> dict[str, tuple[float, float]]:
    raw_keys = ("temperature_k", "pressure_pa")
    result: dict[str, tuple[float, float]] = {}
    for key in raw_keys:
        values = np.asarray([float(sample.metadata[key]) for sample in samples], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0.0:
            raise A2HProtocolError(f"ENV-C1 requires non-zero train-only variance for {key}")
        result[key] = (mean, std)
    return result


def _with_scaled_context(
    sample: UnifiedSample,
    statistics: Mapping[str, tuple[float, float]],
) -> UnifiedSample:
    metadata = dict(sample.metadata)
    for key, (mean, std) in statistics.items():
        metadata[f"{key}_scaled"] = (float(metadata[key]) - mean) / std
    return UnifiedSample(
        signals=sample.signals,
        sensor_id=sample.sensor_id,
        sensor_type=sample.sensor_type,
        valid_mask=sample.valid_mask,
        quality=sample.quality,
        time=sample.time,
        target=sample.target,
        target_mask=sample.target_mask,
        group_id=sample.group_id,
        dataset_id=sample.dataset_id,
        metadata=metadata,
    )


def _run_axis_model_records(
    dataset: A2HDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    stress_indices: np.ndarray,
    *,
    model_id: str,
    seeds: Sequence[int],
    head_id: str = "H0",
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    combined = np.concatenate((val_indices, stress_indices))
    for seed in seeds:
        fit = _fit_model(
            dataset,
            train_indices,
            val_indices,
            combined,
            model_id=model_id,
            seed=int(seed),
            head_id=head_id,
            train_config=train_config,
            model_config=model_config,
            max_epochs_override=max_epochs_override,
        )
        val_prediction = fit.prediction[: len(val_indices)]
        stress_prediction = fit.prediction[len(val_indices) :]
        rows.append(
            {
                "seed": int(seed),
                "validation": _score_prediction(dataset, val_indices, val_prediction),
                "stress_val": _score_prediction(dataset, stress_indices, stress_prediction),
                "stress_stratification": _stratified_metrics(dataset, stress_indices, stress_prediction),
                "resources": dict(fit.resources),
            }
        )
    return _summarize_records(rows, metric_key="stress_val")


def _promotion_gate(
    model_matrix: Mapping[str, Any],
    *,
    eligible_axes: Sequence[str],
    candidate: str,
    baseline: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    axis_rows: dict[str, Any] = {}
    for axis in eligible_axes:
        candidate_summary = model_matrix[axis][candidate]
        baseline_summary = model_matrix[axis][baseline]
        candidate_records = candidate_summary["seed_records"]
        baseline_records = baseline_summary["seed_records"]
        if len(candidate_records) != len(baseline_records):
            raise A2HProtocolError(f"{candidate} and {baseline} seed counts differ on {axis}")
        candidate_values = np.asarray([record["stress_val"]["macro_RNMAE"] for record in candidate_records], dtype=np.float64)
        baseline_values = np.asarray([record["stress_val"]["macro_RNMAE"] for record in baseline_records], dtype=np.float64)
        candidate_components = np.asarray([record["stress_val"]["component_RNMAE"] for record in candidate_records], dtype=np.float64)
        baseline_components = np.asarray([record["stress_val"]["component_RNMAE"] for record in baseline_records], dtype=np.float64)
        mean_candidate = float(candidate_values.mean())
        mean_baseline = float(baseline_values.mean())
        axis_rows[axis] = {
            "relative_improvement": float((mean_baseline - mean_candidate) / mean_baseline),
            "same_direction_seed_count": int(np.sum(candidate_values < baseline_values)),
            "component_delta": [float(value) for value in (candidate_components.mean(axis=0) - baseline_components.mean(axis=0))],
            "candidate_mean_stress_macro_RNMAE": mean_candidate,
            "baseline_mean_stress_macro_RNMAE": mean_baseline,
        }
    passing_axes = [
        axis
        for axis, row in axis_rows.items()
        if row["relative_improvement"] >= float(thresholds["relative_improvement"])
        and row["same_direction_seed_count"] >= int(thresholds["min_seeds_same_direction"])
        and max(row["component_delta"]) <= float(thresholds["max_component_absolute_degradation"])
    ]
    checks = {
        "mean_improvement_on_all_eligible_axes": len(passing_axes) == len(axis_rows),
        "at_least_two_eligible_axes_improve": len(passing_axes) >= int(thresholds["min_eligible_axes_same_direction"]),
        "no_component_exceeds_registered_degradation": all(
            max(row["component_delta"]) <= float(thresholds["max_component_absolute_degradation"])
            for row in axis_rows.values()
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate": candidate,
        "baseline": baseline,
        "axis_results": axis_rows,
        "passing_axes": passing_axes,
        "checks": checks,
    }


def _hard_test_gate(
    hard_axes: Mapping[str, Any],
    *,
    selected_model: str,
    thresholds: Mapping[str, Any],
    eligible_axes: Sequence[str],
) -> dict[str, Any]:
    if selected_model == "B5":
        return {"status": "NOT_APPLICABLE", "reason": "selected model is the frozen primary baseline"}
    rows: dict[str, Any] = {}
    for axis in eligible_axes:
        if axis not in hard_axes:
            continue
        selected = hard_axes[axis]["selected_model"]
        baseline = hard_axes[axis]["matched_B5_baseline"]
        delta = [
            float(candidate - reference)
            for candidate, reference in zip(
                selected["component_RNMAE"], baseline["component_RNMAE"], strict=True
            )
        ]
        reference_value = float(baseline["macro_RNMAE"])
        candidate_value = float(selected["macro_RNMAE"])
        rows[axis] = {
            "relative_improvement": float((reference_value - candidate_value) / reference_value),
            "component_delta": delta,
            "bootstrap_ci_excludes_zero": bool(hard_axes[axis]["bootstrap"]["ci_excludes_zero"]),
        }
    checks = {
        "all_eligible_axes_meet_improvement": all(
            row["relative_improvement"] >= float(thresholds["relative_improvement"])
            for row in rows.values()
        ),
        "all_eligible_axes_component_safe": all(
            max(row["component_delta"]) <= float(thresholds["max_component_absolute_degradation"])
            for row in rows.values()
        ),
    }
    return {"status": "PASS" if rows and all(checks.values()) else "FAIL", "axes": rows, "checks": checks}


def _benchmark_saturated(audit: Mapping[str, Any]) -> bool:
    values = [
        float(axis["eligibility"]["oracle_relative_concat_headroom"])
        for axis in audit.get("axes", {}).values()
    ]
    return bool(values) and all(value < 0.05 for value in values)


def _parameter_parity_report(
    *,
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> dict[str, Any]:
    concat = _build_torch_model(
        model_id="C1",
        head_id="H0",
        train_config=train_config,
        model_config=model_config,
        include_context=False,
    )
    deepsets = _build_torch_model(
        model_id="M1",
        head_id="H0",
        train_config=train_config,
        model_config=model_config,
        include_context=False,
    )
    tolerance = float(train_config["parameter_match_tolerance"])
    parity = parameter_parity_report(concat, deepsets, tolerance=tolerance)
    return {
        "C1_parameter_count": parity["left_parameter_count"],
        "M1_parameter_count": parity["right_parameter_count"],
        "relative_difference": parity["relative_difference"],
        "within_tolerance": parity["within_tolerance"],
    }


def _score_prediction(
    dataset: A2HDataset,
    indices: Sequence[int],
    prediction: np.ndarray,
    *,
    semantics: str = "same_mixture_repeat_mean",
) -> dict[str, Any]:
    index_values = np.asarray(indices, dtype=np.int64)
    if len(index_values) != len(prediction):
        raise ValueError("prediction length does not match indices")
    if semantics == "single_observation":
        selected_positions = [
            position
            for position, index in enumerate(index_values)
            if dataset.observations[int(index)].repeat_index == 0
        ]
        if not selected_positions:
            raise ValueError("single_observation semantics require repeat_index=0 rows")
        selected_positions_array = np.asarray(selected_positions, dtype=np.int64)
        targets = _targets(dataset, index_values[selected_positions_array])
        groups = _groups(dataset, index_values[selected_positions_array])
        predictions = np.asarray(prediction)[selected_positions_array]
    elif semantics == "same_mixture_repeat_mean":
        targets = _targets(dataset, index_values)
        groups = _groups(dataset, index_values)
        predictions = np.asarray(prediction)
    elif semantics == "no_average":
        targets = _targets(dataset, index_values)
        groups = tuple(dataset.observations[int(index)].observation_id for index in index_values)
        predictions = np.asarray(prediction)
    else:
        raise ValueError(f"unsupported A2H repeat semantics: {semantics!r}")
    metric = evaluate_predictions(
        targets,
        predictions,
        groups,
        np.arange(len(targets), dtype=np.int64),
    )
    metric["repeat_semantics"] = semantics
    metric["output_constraints"] = evaluate_output_constraints(predictions, targets=targets)
    return metric


def _feature_matrix(
    dataset: A2HDataset,
    indices: Sequence[int],
    *,
    feature_map: str,
    include_context: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    index_values = np.asarray(indices, dtype=np.int64)
    signals = np.asarray(dataset.signals[index_values], dtype=np.float64)
    if feature_map != "raw":
        raise ValueError(f"unsupported A2H feature map: {feature_map!r}")
    features = signals
    names = list(SENSOR_IDS)
    if include_context:
        contexts = np.asarray(
            [
                [
                    dataset.observations[int(index)].temperature_k,
                    dataset.observations[int(index)].pressure_pa,
                ]
                for index in index_values
            ],
            dtype=np.float64,
        )
        features = np.column_stack((features, contexts))
        names.extend(("temperature_k", "pressure_pa"))
    return features, tuple(names)


def _summarize_records(records: Sequence[Mapping[str, Any]], *, metric_key: str = "validation") -> dict[str, Any]:
    if not records:
        raise ValueError("records must be non-empty")
    macro = np.asarray([float(record[metric_key]["macro_RNMAE"]) for record in records], dtype=np.float64)
    components = np.asarray([record[metric_key]["component_RNMAE"] for record in records], dtype=np.float64)
    return {
        "seed_records": [dict(record) for record in records],
        "mean_stress_macro_RNMAE" if metric_key == "stress_val" else "mean_validation_macro_RNMAE": float(macro.mean()),
        "std_stress_macro_RNMAE" if metric_key == "stress_val" else "std_validation_macro_RNMAE": float(macro.std(ddof=0)),
        "mean_component_RNMAE": [float(value) for value in components.mean(axis=0)],
        "worst_seed_macro_RNMAE": float(macro.max()),
    }


def _summarize_noise_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        semantics: _summarize_records(
            [{"seed": record["seed"], "validation": record[semantics]} for record in records]
        )
        for semantics in ("single_observation", "same_mixture_repeat_mean", "no_average")
    }


def _stratified_metrics(dataset: A2HDataset, indices: Sequence[int], prediction: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    regions = dataset.manifest.get("composition_regions", {})
    labels = [
        {
            "condition_family": observation.condition_family,
            "environment_id": observation.environment_id,
            "calibration_profile_id": observation.calibration_profile_id,
            "noise_profile_id": observation.noise_profile_id,
            "composition_region": composition_region(observation.composition, regions=regions),
        }
        for observation in (dataset.observations[int(index)] for index in indices)
    ]
    for field in labels[0] if labels else ():
        values = sorted({label[field] for label in labels})
        field_result = {}
        for value in values:
            positions = np.asarray([position for position, label in enumerate(labels) if label[field] == value], dtype=np.int64)
            field_result[value] = _score_prediction(
                dataset,
                np.asarray(indices, dtype=np.int64)[positions],
                np.asarray(prediction)[positions],
            )
        result[field] = field_result
    return result


def _stratification_counts(dataset: A2HDataset, indices: Sequence[int]) -> dict[str, dict[str, int]]:
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
            counts = result.setdefault(field, {})
            counts[value] = counts.get(value, 0) + 1
    return result


def _geometric_distance_summary(dataset: A2HDataset, train_indices: Sequence[int], query_indices: Sequence[int]) -> dict[str, Any]:
    train = _unique_group_targets(dataset, train_indices)
    query = _targets(dataset, query_indices)
    distances = np.sqrt(((query[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return {
        "metric": "euclidean_mol_percent",
        "mean": float(distances.mean()),
        "P90": float(np.percentile(distances, 90)),
        "max": float(distances.max()),
    }


def _nested_group_order(dataset: A2HDataset, indices: Sequence[int], *, seed: int) -> list[str]:
    groups: dict[str, list[str]] = {}
    for index in indices:
        observation = dataset.observations[int(index)]
        groups.setdefault(observation.condition_family, []).append(observation.mixture_id)
    rng = np.random.default_rng(seed)
    for values in groups.values():
        values[:] = list(dict.fromkeys(values))
        rng.shuffle(values)
    ordered: list[str] = []
    while any(groups.values()):
        for family in sorted(groups):
            if groups[family]:
                ordered.append(groups[family].pop())
    return ordered


def _unique_group_targets(dataset: A2HDataset, indices: Sequence[int]) -> np.ndarray:
    values: dict[str, tuple[float, float, float]] = {}
    for index in indices:
        observation = dataset.observations[int(index)]
        values.setdefault(observation.mixture_id, observation.composition)
    return np.asarray([values[key] for key in sorted(values)], dtype=np.float64)


def _targets(dataset: A2HDataset, indices: Sequence[int]) -> np.ndarray:
    return np.asarray([dataset.observations[int(index)].composition for index in indices], dtype=np.float64)


def _groups(dataset: A2HDataset, indices: Sequence[int]) -> tuple[str, ...]:
    return tuple(dataset.observations[int(index)].mixture_id for index in indices)


def _project_rows_to_simplex(values: np.ndarray, *, total: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("simplex projection requires a two-dimensional matrix")
    projected: list[np.ndarray] = []
    for row in matrix:
        sorted_values = np.sort(row)[::-1]
        cumulative = np.cumsum(sorted_values) - total
        ranks = np.arange(1, len(row) + 1, dtype=np.float64)
        support = sorted_values - cumulative / ranks > 0.0
        support_size = max(int(support.sum()), 1)
        threshold = cumulative[support_size - 1] / support_size
        projected.append(np.maximum(row - threshold, 0.0))
    return np.asarray(projected, dtype=np.float64)


def _parameter_count(model: Any) -> int:
    if isinstance(model, Ridge):
        return int(model.coef_.size + model.intercept_.size)
    if isinstance(model, MultiOutputRegressor):
        return int(
            sum(
                tree.tree_.node_count
                for estimator in model.estimators_
                for tree in np.asarray(estimator.estimators_, dtype=object).reshape(-1)
            )
        )
    raise TypeError(f"cannot count parameters for {type(model).__name__}")


def _ensure_data_current(root: Path, data_config_path: str | Path | None) -> None:
    data_config_file = _resolve_file(
        root,
        Path(data_config_path) if data_config_path is not None else root / A2H_DEFAULT_DATA_CONFIG,
    )
    config = _read_json_object(data_config_file)
    validate_a2h_data_config(config)
    manifest_path = _a2h_data_dir(root) / "manifest.json"
    expected_hash = _canonical_sha256(config)
    if not manifest_path.is_file() or _read_json_object(manifest_path).get("generator_config_sha256") != expected_hash:
        run_a2h_generation(project_root=root, data_config_path=data_config_file)


def _load_or_run_audit(root: Path, *, eval_config_path: str | Path | None) -> dict[str, Any]:
    audit_path = _a2h_summary_dir(root) / "a2h_difficulty_audit.json"
    if audit_path.is_file():
        return {"audit": _read_json_object(audit_path)}
    return run_a2h_difficulty_audit(project_root=root, eval_config_path=eval_config_path)


def _default_config_paths(
    root: Path,
    data_config_path: str | Path | None,
    eval_config_path: str | Path | None,
    train_config_path: str | Path | None,
) -> dict[str, Path]:
    return {
        "data_config": _resolve_file(root, Path(data_config_path) if data_config_path is not None else root / A2H_DEFAULT_DATA_CONFIG),
        "eval_config": _resolve_file(root, Path(eval_config_path) if eval_config_path is not None else root / "configs" / "eval" / "a2h_eval.json"),
        "train_config": _resolve_file(root, Path(train_config_path) if train_config_path is not None else root / "configs" / "train" / "a2h_train.json"),
    }


def _load_training_bundle(
    root: Path,
    *,
    train_config_path: str | Path | None,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    train_path = _resolve_file(
        root,
        Path(train_config_path) if train_config_path is not None else root / "configs" / "train" / "a2h_train.json",
    )
    model_path = _resolve_file(root, root / "configs" / "model" / "a2h_candidate.json")
    train_config = _read_json_object(train_path)
    model_config = _read_json_object(model_path)
    validate_a2h_train_config(train_config)
    validate_a2h_model_config(model_config)
    parity = _parameter_parity_report(train_config=train_config, model_config=model_config)
    if not parity["within_tolerance"]:
        raise A2HProtocolError("A2H C1 and M1 exceed the registered parameter-match tolerance")
    return train_path, train_config, model_path, model_config


def _validate_a1_reference(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    reference = config["a1_reference"]
    manifest_path = _resolve_file(root, Path(str(reference["manifest_path"])))
    manifest = _read_json_object(manifest_path)
    if manifest.get("data_version") != reference["data_version"] or manifest.get("content_sha256") != reference["content_sha256"]:
        raise A2HProtocolError("A1 reference manifest has changed; A2H cannot proceed")
    split_hash = _a1_split_hash(manifest)
    if split_hash != reference["split_hash"]:
        raise A2HProtocolError("A1 reference split hash has changed; A2H cannot proceed")
    return {
        "manifest_path": _relative_path(root, manifest_path),
        "data_version": manifest["data_version"],
        "content_sha256": manifest["content_sha256"],
        "split_hash": split_hash,
    }


def _a1_split_hash(manifest: Mapping[str, Any]) -> str:
    assignments = sorted(
        {
            (str(condition["mixture_id"]), str(condition["split"]))
            for condition in manifest["conditions"]
        }
    )
    return _canonical_sha256(
        {
            "schema_version": "gf-a2-split-1",
            "split_seed": manifest.get("split_seed"),
            "assignments": [{"mixture_id": mixture_id, "split": split} for mixture_id, split in assignments],
        }
    )


def _validate_experiment_bindings(
    experiment: Mapping[str, Any],
    *,
    root: Path,
    data_config_path: Path,
    eval_config_path: Path,
    train_config_path: Path,
) -> None:
    for key, expected in (("data_config", data_config_path), ("eval_config", eval_config_path), ("train_config", train_config_path)):
        actual = _resolve_file(root, Path(str(experiment[key])))
        if actual != expected:
            if sha256_file(actual) != sha256_file(expected):
                raise A2HProtocolError(f"{experiment['experiment_id']} {key} differs from selected protocol config")


def _profile_metric_key(metric_key: str) -> str:
    return metric_key


def _git_state(root: Path) -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not revision:
        raise A2HProtocolError("git rev-parse returned an empty revision")
    return revision, bool(status)


def _a2h_data_dir(root: Path) -> Path:
    return root / A2H_DATA_RELATIVE_DIR


def _a2h_run_root(root: Path) -> Path:
    return root / "outputs" / "runs" / A2H_OUTPUT_NAMESPACE


def _a2h_summary_dir(root: Path) -> Path:
    return root / "outputs" / "summary" / A2H_OUTPUT_NAMESPACE


def _a2h_report_dir(root: Path) -> Path:
    return root / "outputs" / "reports" / A2H_OUTPUT_NAMESPACE


def _project_root(project_root: str | Path | None) -> Path:
    return (Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]).resolve()


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise A2HProtocolError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _resolve_file(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise A2HProtocolError(f"configured path escapes project root: {path}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _validate_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_KEYS & set(value)
        if forbidden:
            raise A2HProtocolError(f"forbidden legacy keys: {sorted(forbidden)}")
        for child in value.values():
            _validate_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _validate_no_forbidden_keys(child)


def _validate_hash_value(value: str, name: str) -> None:
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise A2HProtocolError(f"{name} must be a lowercase SHA256 hex string")


def _require_hash(config: Mapping[str, Any], key: str) -> None:
    value = config.get(key)
    if not isinstance(value, str):
        raise A2HProtocolError(f"{key} must be a SHA256 string")
    _validate_hash_value(value, key)


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise A2HProtocolError(f"{key} must be an object")
    return value


def _load_a2h_physics_type() -> Any:
    from gf.sim.a2h_dataset import A2HPhysicsConfig

    return A2HPhysicsConfig


def _validate_hash_value_or_none(value: Any, name: str) -> None:
    if value is not None:
        _validate_hash_value(str(value), name)


def _format_float(value: float) -> str:
    return f"{value:.8f}"


def _write_formal_report(path: Path, result: Mapping[str, Any], data_manifest: Mapping[str, Any]) -> None:
    lines = [
        "# A2H 正式报告",
        "",
        f"- formal run status: `{result['formal_run_status']}`",
        f"- final status: `{result['status']}`",
        f"- selected model: `{result['selected_model']}`",
        f"- data version: `{data_manifest['data_version']}`",
        f"- hard-test access: `{result['hard_test_read']}`",
        f"- failure cases: `outputs/summary/{A2H_OUTPUT_NAMESPACE}/a2h_failure_cases.json`",
        "",
        "## Hard-test axes",
        "",
    ]
    for axis, payload in result["hard_test_axes"].items():
        selected = payload["selected_model"]
        baseline = payload["matched_B5_baseline"]
        lines.append(
            f"- `{axis}`: selected macro_RNMAE={_format_float(float(selected['macro_RNMAE']))}; "
            f"B5 macro_RNMAE={_format_float(float(baseline['macro_RNMAE']))}; "
            f"bootstrap CI excludes zero={payload['bootstrap']['ci_excludes_zero']}"
        )
    lines.extend(
        [
            "",
            "Hard-test was unlocked once after the data, split-family, eligible-axis, candidate, matched-baseline, checkpoint and chart hashes were verified. No hard-test result was used for tuning.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_formal_review(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# A2H 评审记录",
        "",
        f"- 终态：`{result['status']}`",
        f"- 合格困难轴：{', '.join(result['eligible_axes'])}",
        f"- 主臂：`{result['selected_model']}`",
        f"- hard test：`{'已解锁一次' if result['hard_test_read'] else '保持锁定'}`",
        "- 失败案例：[A2H failure cases](../../summary/a2h_v2/a2h_failure_cases.json)",
        "",
        "## 判定",
        "",
        "A2H 只在 stress-val 资格审计完成后进行 matched 比较；hard test 结果不回流到训练规模、扰动范围或候选选择。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one general_fusion A2H protocol stage.")
    parser.add_argument("--mode", choices=("protocol", "generate", "audit", "learning-noise", "ood", "compare", "formal", "smoke", "all"), default="protocol")
    parser.add_argument("--project-root", type=Path, default=_project_root(None))
    parser.add_argument("--data-config", type=Path)
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--unlock-hard-test", action="store_true")
    parser.add_argument("--selection-record", type=Path)
    parser.add_argument("--selected-checkpoint", type=Path)
    parser.add_argument("--formal-run-status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_a2h(
        project_root=args.project_root,
        mode=args.mode,
        data_config_path=args.data_config,
        eval_config_path=args.eval_config,
        train_config_path=args.train_config,
        max_epochs_override=args.max_epochs,
        bootstrap_samples=args.bootstrap_samples,
        unlock_hard_test=args.unlock_hard_test,
        selection_record_path=args.selection_record,
        selected_checkpoint_path=args.selected_checkpoint,
        formal_run_status=args.formal_run_status,
    )
    print(json.dumps({"stage": result.get("stage"), "status": result.get("status")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
