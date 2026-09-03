from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from gf.dl.contracts import UnifiedSample, collate_samples
from gf.dl.evaluation import evaluate_predictions, group_bootstrap_comparison
from gf.dl.mainstream_architectures import (
    A2M_MODEL_IDS,
    build_a2m_model,
    validate_a2m_model_config,
)
from gf.dl.preprocessing import TrainGroupStandardScaler
from gf.dl.training import TorchTrainingConfig, train_torch_model, trainable_parameter_count
from gf.ml.baselines import fit_full_regression_baselines
from gf.pipeline.runtime import runtime_fingerprint, sha256_file
from gf.pipeline.tqif_common import canonical_hash as canonical_sha256
from gf.sim.a2h_dataset import load_a2h_dataset
from gf.sim.a2m_dataset import (
    A2M_AXES,
    A2M_DATA_VERSION_PREFIX,
    A2M_PRIMARY_AXES,
    A2M_SCHEMA_VERSION,
    A2MTestLockError,
    compute_a2m_split_hash,
    generate_a2m_formal_holdout,
    load_a2m_dataset,
    validate_a2m_data_config,
)


A2M_EVAL_SCHEMA_VERSION = "gf-a2m-eval-1"
A2M_TRAIN_SCHEMA_VERSION = "gf-a2m-train-1"
A2M_EXPERIMENT_SCHEMA_VERSION = "gf-a2m-experiment-1"
A2M_PROTOCOL_SCHEMA_VERSION = "gf-a2m-protocol-1"
A2M_RUN_MANIFEST_SCHEMA_VERSION = "gf-a2m-run-manifest-1"
A2M_FORMAL_STATUS = "FROZEN"
A2M_TRAINING_SEEDS = (17, 29, 43, 71, 101)
A2M_ALLOWED_DEV_SPLITS = ("train", "val", "stress_val")
A2M_FORMAL_EVIDENCE = (
    "data_content_sha256",
    "split_hash",
    "profile_hash",
    "protocol_config_sha256",
    "checkpoint_sha256",
    "runtime_fingerprint",
    "primary_chart_template_sha256",
    "formal_run_status",
)
A2M_DEFAULT_DATA_CONFIG = Path("configs/data/ar_he_co2_a2m_v1.json")
A2M_DEFAULT_EVAL_CONFIG = Path("configs/eval/a2m_eval.json")
A2M_DEFAULT_TRAIN_CONFIG = Path("configs/train/a2m_train.json")
A2M_DEFAULT_MODEL_CONFIGS = (
    Path("configs/model/a2m_mlp.json"),
    Path("configs/model/a2m_resnet.json"),
    Path("configs/model/a2m_ftt.json"),
)
A2M_DEFAULT_CHART_CONFIG = Path("configs/experiment/a2m_primary_chart_template.json")
A2M_DEFAULT_HOLDOUT_DIR = Path("data/a2m_v1")


class A2MProtocolError(ValueError):
    """Raised when an A2M protocol, provenance, or access lock is violated."""


class A2MTestUnlockError(A2MProtocolError):
    """Raised when formal evidence is incomplete or formal access is repeated."""

    __test__ = False


@dataclass(frozen=True)
class A2MFitResult:
    model_id: str
    recipe_name: str
    seed: int
    split_family: str
    validation: Mapping[str, Any]
    validation_by_axis: Mapping[str, Mapping[str, Any]]
    stress: Mapping[str, Any]
    validation_prediction: np.ndarray
    stress_prediction: np.ndarray
    stress_targets: np.ndarray
    stress_groups: tuple[str, ...]
    stress_by_axis: Mapping[str, Mapping[str, Any]]
    stress_predictions_by_axis: Mapping[str, np.ndarray]
    stress_targets_by_axis: Mapping[str, np.ndarray]
    stress_groups_by_axis: Mapping[str, tuple[str, ...]]
    resources: Mapping[str, Any]


def run_a2m(
    *,
    project_root: str | Path | None = None,
    mode: str = "protocol",
    data_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    max_epochs_override: int | None = None,
    bootstrap_samples: int | None = None,
    unlock_formal: bool = False,
    formal_run_status: str | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    if mode == "reproduce":
        return run_a2m_a1_reproduction(project_root=root)
    if mode == "protocol":
        return run_a2m_protocol(project_root=root)
    if mode == "generate":
        return run_a2m_generation(project_root=root, data_config_path=data_config_path)
    if mode == "smoke":
        return run_a2m_smoke(project_root=root, max_epochs_override=max_epochs_override)
    if mode == "dev":
        return run_a2m_development(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
            max_epochs_override=max_epochs_override,
            bootstrap_samples=bootstrap_samples,
        )
    if mode == "formal":
        return run_a2m_formal(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
            unlock_formal=unlock_formal,
            formal_run_status=formal_run_status,
            bootstrap_samples=bootstrap_samples,
        )
    if mode == "all":
        result: dict[str, Any] = {
            "reproduction": run_a2m_a1_reproduction(project_root=root),
            "protocol": run_a2m_protocol(project_root=root),
            "generation": run_a2m_generation(project_root=root, data_config_path=data_config_path),
            "smoke": run_a2m_smoke(project_root=root, max_epochs_override=max_epochs_override),
        }
        result["development"] = run_a2m_development(
            project_root=root,
            data_config_path=data_config_path,
            eval_config_path=eval_config_path,
            train_config_path=train_config_path,
            max_epochs_override=max_epochs_override,
            bootstrap_samples=bootstrap_samples,
        )
        if unlock_formal:
            result["formal"] = run_a2m_formal(
                project_root=root,
                data_config_path=data_config_path,
                eval_config_path=eval_config_path,
                train_config_path=train_config_path,
                unlock_formal=True,
                formal_run_status=formal_run_status,
                bootstrap_samples=bootstrap_samples,
            )
        return result
    raise ValueError(f"unsupported A2M mode: {mode!r}")


def run_a2m_a1_reproduction(*, project_root: str | Path) -> dict[str, Any]:
    """Audit frozen A1 B5-SK without rewriting any A1 artifact."""

    from gf.ml.baselines import run_baseline_suite
    from gf.sim.a1_dataset import load_dataset

    root = _project_root(project_root)
    dataset = load_dataset(root / "data" / "a1_formal")
    summary_path = root / "outputs" / "summary" / "a1_formal" / "baseline_summary.json"
    predictions_path = root / "outputs" / "summary" / "a1_formal" / "predictions.csv"
    expected = _read_json(summary_path)
    saved_prediction = _read_a1_b5_mean_predictions(predictions_path, len(dataset.conditions))
    targets = np.vstack([condition.composition for condition in dataset.conditions]).astype(np.float64)
    groups = np.asarray(dataset.group_ids, dtype=object)
    val_indices = np.asarray(
        [index for index, condition in enumerate(dataset.conditions) if condition.split == "val"],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, condition in enumerate(dataset.conditions) if condition.split == "test"],
        dtype=np.int64,
    )
    saved_validation = evaluate_predictions(targets, saved_prediction, groups, val_indices)
    saved_test = evaluate_predictions(targets, saved_prediction, groups, test_indices)
    current = run_baseline_suite(
        dataset,
        training_seed=20260827,
        include_mlp=True,
        mlp_seeds=A2M_TRAINING_SEEDS,
    )
    current_prediction = current.predictions["B5__mean"]
    report = {
        "schema_version": "gf-a2m-a1-reproduction-1",
        "status": "PASS_WITH_NEW_REFERENCE",
        "historical_identity": "B5-SK",
        "current_pyTorch_identity": "A2M-MLP",
        "a1_artifacts": {
            "baseline_summary_sha256": sha256_file(str(summary_path)),
            "predictions_sha256": sha256_file(str(predictions_path)),
            "data_manifest_sha256": sha256_file(str(root / "data" / "a1_formal" / "manifest.json")),
        },
        "saved_prediction_recomputed": {
            "validation": saved_validation,
            "test": saved_test,
            "summary_validation_macro_RNMAE": expected["best_overall_full_input"]["validation_macro_RNMAE"],
            "summary_test_macro_RNMAE": expected["best_overall_full_input"]["test_macro_RNMAE"],
        },
        "current_fit": {
            "validation_macro_RNMAE": current.summary["best_overall_full_input"]["validation_macro_RNMAE"],
            "test_macro_RNMAE": current.summary["best_overall_full_input"]["test_macro_RNMAE"],
            "max_absolute_prediction_delta_vs_saved_csv": float(np.max(np.abs(current_prediction - saved_prediction))),
            "mean_absolute_prediction_delta_vs_saved_csv": float(np.mean(np.abs(current_prediction - saved_prediction))),
        },
        "root_cause": {
            "classification": "DEPENDENCY_RUNTIME_DRIFT_NO_HISTORICAL_FINGERPRINT",
            "evidence": "A1 summary has no historical runtime fingerprint; current isolated runtime produces a different sklearn LBFGS fit.",
            "frozen_a1_action": "read_only",
            "a2m_action": "freeze_current_runtime_and_use_A2M_MLP_as_new_reference",
        },
        "runtime_fingerprint": runtime_fingerprint(device="cpu", dtype="float32"),
    }
    output_dir = root / "outputs" / "summary" / "a2m"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "a2m_a1_reproduction.json", report)
    _write_json(
        root / "outputs" / "runs" / "a2m" / "a2m-0-reproduction" / "manifest.json",
        {
            "schema_version": A2M_RUN_MANIFEST_SCHEMA_VERSION,
            "stage": "A2M-0",
            "status": report["status"],
            "report_path": "outputs/summary/a2m/a2m_a1_reproduction.json",
            "runtime_fingerprint": report["runtime_fingerprint"],
        },
    )
    return report


def run_a2m_protocol(*, project_root: str | Path) -> dict[str, Any]:
    root = _project_root(project_root)
    configs = _load_a2m_configs(root)
    _validate_a2m_config_set(configs, root=root)
    source = _verify_a2h_source(root, configs["data"])
    config_hashes = {
        name: sha256_file(str(path))
        for name, path in configs["paths"].items()
        if name != "chart"
    }
    config_hashes["chart"] = sha256_file(str(configs["paths"]["chart"]))
    manifest = {
        "schema_version": A2M_PROTOCOL_SCHEMA_VERSION,
        "stage": "A2M-1",
        "status": "PASS",
        "config_hashes": config_hashes,
        "a2h_development_source": source,
        "allowed_read_splits": list(A2M_ALLOWED_DEV_SPLITS),
        "formal_holdout": {
            "data_dir": "data/a2m_v1",
            "split": "formal",
            "access": "locked",
            "formal_run_status": None,
        },
        "model_ids": list(A2M_MODEL_IDS),
        "training_seeds": list(A2M_TRAINING_SEEDS),
        "runtime_fingerprint": runtime_fingerprint(device="cpu", dtype="float32"),
    }
    path = root / "outputs" / "runs" / "a2m" / "a2m-1-protocol" / "manifest.json"
    _write_json(path, manifest)
    _write_json(root / "outputs" / "summary" / "a2m" / "a2m_protocol.json", manifest)
    return {"stage": "A2M-1", "status": "PASS", "manifest": manifest}


def run_a2m_generation(
    *,
    project_root: str | Path,
    data_config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    config_path = _resolve(root, data_config_path or A2M_DEFAULT_DATA_CONFIG)
    config = _read_json(config_path)
    try:
        validate_a2m_data_config(config)
    except ValueError as exc:
        raise A2MProtocolError(str(exc)) from exc
    source = _verify_a2h_source(root, config)
    data_dir = root / A2M_DEFAULT_HOLDOUT_DIR
    existing_manifest_path = data_dir / "manifest.json"
    if existing_manifest_path.is_file():
        manifest = _read_json(existing_manifest_path)
        if (
            manifest.get("data_version") != config["data_version"]
            or manifest.get("generator_config_sha256") != canonical_sha256(config)
            or manifest.get("source_config_sha256") != config["a2h_development_source"]["config_sha256"]
        ):
            raise A2MProtocolError("existing A2M holdout does not match the frozen data configuration")
        status = "REUSED"
    else:
        dataset = generate_a2m_formal_holdout(
            data_dir,
            config=config,
            project_root=root,
        )
        manifest = dict(dataset.manifest)
        status = "GENERATED"
    result = {
        "schema_version": "gf-a2m-generation-1",
        "stage": "A2M-1",
        "status": status,
        "data_version": manifest["data_version"],
        "data_dir": _relative(root, data_dir),
        "content_sha256": manifest["content_sha256"],
        "split_hash": manifest["split_hash"],
        "profile_hash": manifest["profile_hash"],
        "source": source,
        "formal_access": "locked",
    }
    _write_json(root / "outputs" / "summary" / "a2m" / "a2m_generation.json", result)
    _write_json(root / "outputs" / "runs" / "a2m" / "a2m-1-generation" / "manifest.json", result)
    return result


def run_a2m_smoke(
    *,
    project_root: str | Path,
    max_epochs_override: int | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    started_at = _utc_timestamp()
    configs = _load_a2m_configs(root)
    _validate_a2m_config_set(configs, root=root)
    dataset = load_a2h_dataset(root / "data" / "a2h_v2")
    records: list[dict[str, Any]] = []
    for model_id in A2M_MODEL_IDS:
        for recipe in configs["model_by_id"][model_id]["recipes"]:
            fit = _fit_a2m_on_a2h(
                dataset,
                axis="iid",
                model_id=model_id,
                recipe_name=str(recipe["name"]),
                seed=17,
                model_config=configs["model_by_id"][model_id],
                train_config=configs["train"],
                root=root,
                output_dir=root / "outputs" / "runs" / "a2m" / "a2m-3-smoke",
                max_epochs_override=2 if max_epochs_override is None else max_epochs_override,
            )
            records.append(_fit_to_record(fit, include_predictions=False))
    finished_at = _utc_timestamp()
    result = {
        "schema_version": "gf-a2m-smoke-1",
        "stage": "A2M-3",
        "status": "PASS",
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_status": "SUCCESS",
        "formal_holdout_access": "locked",
        "recipe_count": len(records),
        "expected_recipe_count": 6,
        "runs": records,
        "runtime_fingerprint": runtime_fingerprint(device="cpu", dtype="float32"),
    }
    _write_json(root / "outputs" / "summary" / "a2m" / "a2m_smoke.json", result)
    _write_json(root / "outputs" / "runs" / "a2m" / "a2m-3-smoke" / "manifest.json", result)
    return result


def validate_a2m_eval_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != A2M_EVAL_SCHEMA_VERSION or config.get("parent_schema_version") != "gf-eval-1":
        raise A2MProtocolError("A2M evaluation schema is unsupported")
    if config.get("metric") != "macro_RNMAE":
        raise A2MProtocolError("A2M metric must be macro_RNMAE")
    if config.get("target_ranges") != {"x_Ar_pct": 100.0, "x_He_pct": 100.0, "x_CO2_pct": 100.0}:
        raise A2MProtocolError("A2M target ranges are not frozen")
    if tuple(config.get("training_seeds") or ()) != A2M_TRAINING_SEEDS:
        raise A2MProtocolError("A2M training seeds are not the frozen five seeds")
    if config.get("bootstrap_samples") != 2000 or config.get("confidence_level") != 0.95:
        raise A2MProtocolError("A2M bootstrap is frozen at 2000 samples and 0.95 confidence")
    if tuple(config.get("axes") or ()) != A2M_AXES or tuple(config.get("primary_axes") or ()) != A2M_PRIMARY_AXES:
        raise A2MProtocolError("A2M axes are not frozen")
    if config.get("composition_axis_role") != "diagnostic_only" or config.get("formal_split") != "formal":
        raise A2MProtocolError("A2M composition or formal split semantics changed")
    if config.get("equivalence_band") != {"relative_macro_RNMAE": 0.05}:
        raise A2MProtocolError("A2M equivalence band is not frozen")
    promotion = _required_mapping(config, "promotion")
    expected_promotion = {
        "max_iid_relative_degradation": 0.05,
        "min_stress_axes": 2,
        "min_relative_improvement": 0.05,
        "min_seeds_same_direction": 4,
        "max_component_absolute_degradation": 0.005,
        "bootstrap_ci_upper_strictly_below_zero": True,
        "matched_optimizer_audit_required": True,
    }
    if dict(promotion) != expected_promotion:
        raise A2MProtocolError("A2M promotion gates are not frozen")
    access = _required_mapping(config, "test_access")
    if access.get("default") != "locked" or access.get("unlock_flag") != "--unlock-formal" or access.get("required_formal_run_status") != A2M_FORMAL_STATUS:
        raise A2MProtocolError("A2M formal access lock is not frozen")
    if tuple(access.get("required_evidence") or ()) != A2M_FORMAL_EVIDENCE:
        raise A2MProtocolError("A2M formal evidence order is not frozen")


def validate_a2m_train_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != A2M_TRAIN_SCHEMA_VERSION:
        raise A2MProtocolError("A2M training schema is unsupported")
    if tuple(config.get("seeds") or ()) != A2M_TRAINING_SEEDS:
        raise A2MProtocolError("A2M training seeds are not frozen")
    loss = _required_mapping(config, "loss")
    if loss.get("name") != "mse" or tuple(loss.get("target_scale") or ()) != (100.0, 100.0, 100.0):
        raise A2MProtocolError("A2M loss must be normalized MSE with target scale [100,100,100]")
    max_epochs = config.get("max_epochs")
    if not isinstance(max_epochs, int) or isinstance(max_epochs, bool) or max_epochs <= 0:
        raise A2MProtocolError("A2M max_epochs must be a positive integer")
    early = _required_mapping(config, "early_stopping")
    if early.get("enabled") is not True or early.get("monitor") != "val_macro_RNMAE" or early.get("selection_split") != "a2h.family.val" or early.get("test_access") != "forbidden":
        raise A2MProtocolError("A2M early stopping must use only A2H train and val")
    if config.get("formal_holdout_access") != "forbidden":
        raise A2MProtocolError("A2M training must forbid formal holdout access")
    recipes = config.get("recipes")
    if not isinstance(recipes, Mapping) or set(recipes) != set(A2M_MODEL_IDS):
        raise A2MProtocolError("A2M train recipes must cover exactly the three model IDs")
    for model_id in A2M_MODEL_IDS:
        model_recipes = recipes[model_id]
        if not isinstance(model_recipes, Mapping) or len(model_recipes) != 2:
            raise A2MProtocolError(f"{model_id} must have exactly two training recipes")
        for recipe_name, recipe in model_recipes.items():
            if not isinstance(recipe_name, str) or not recipe_name or not isinstance(recipe, Mapping):
                raise A2MProtocolError(f"{model_id} recipe definitions must be named objects")
            optimizer = _required_mapping(recipe, "optimizer")
            name = optimizer.get("name")
            if model_id == "A2M-MLP" and name not in {"LBFGS", "AdamW"}:
                raise A2MProtocolError("A2M-MLP recipes must use LBFGS and AdamW")
            if model_id != "A2M-MLP" and name != "AdamW":
                raise A2MProtocolError(f"{model_id} recipes must use frozen AdamW")
            if name == "LBFGS" and float(optimizer.get("weight_decay", -1.0)) != 0.0:
                raise A2MProtocolError("LBFGS recipe must use weight_decay=0")
            TorchTrainingConfig.from_mapping(
                {
                    "max_epochs": max_epochs,
                    "optimizer": optimizer,
                    "loss": loss,
                    "early_stopping": {"patience": early["patience"]},
                }
            )
    tolerance = config.get("parameter_match_tolerance")
    if float(tolerance) != 0.10:
        raise A2MProtocolError("A2M parameter match tolerance must be 10%")


def validate_a2m_experiment_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != A2M_EXPERIMENT_SCHEMA_VERSION:
        raise A2MProtocolError("A2M experiment schema is unsupported")
    for key in ("stage", "experiment_id", "kind", "data_config", "eval_config", "train_config", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise A2MProtocolError(f"A2M experiment field {key} must be non-empty")
    model_configs = config.get("model_configs")
    if model_configs != [path.as_posix() for path in A2M_DEFAULT_MODEL_CONFIGS]:
        raise A2MProtocolError("A2M experiment model registry is not frozen")
    allowed = config.get("allowed_read_splits")
    if config.get("kind") == "formal_comparison":
        if allowed != ["formal"] or config.get("formal_holdout_access") != "locked_until_frozen" or config.get("unlock_flag") != "--unlock-formal":
            raise A2MProtocolError("A2M formal experiment must remain locked")
    else:
        if allowed != list(A2M_ALLOWED_DEV_SPLITS) or config.get("formal_holdout_access") != "locked":
            raise A2MProtocolError("A2M development experiment may read only train, val, and stress_val")


def build_a2m_run_manifest(
    *,
    project_root: str | Path,
    stage: str,
    config_paths: Mapping[str, str | Path],
    data_manifest_path: str | Path | None,
    status: str,
    formal_run_status: str | None,
    test_unlocked: bool,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    if test_unlocked != (evidence is not None):
        raise A2MTestUnlockError("formal unlocked manifest must contain evidence and locked one must not")
    if formal_run_status is not None and formal_run_status != A2M_FORMAL_STATUS:
        raise A2MTestUnlockError("formal_run_status must be FROZEN")
    result: dict[str, Any] = {
        "schema_version": A2M_RUN_MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        "config_hashes": {
            name: sha256_file(str(_resolve(root, path)))
            for name, path in sorted(config_paths.items())
        },
        "formal_run_status": formal_run_status,
        "test_unlocked": bool(test_unlocked),
    }
    if data_manifest_path is not None:
        manifest = _read_json(_resolve(root, data_manifest_path))
        result["data_version"] = manifest.get("data_version")
        result["data_content_sha256"] = manifest.get("content_sha256")
        result["split_hash"] = manifest.get("split_hash")
        result["profile_hash"] = manifest.get("profile_hash")
    if evidence is not None:
        result["evidence"] = dict(evidence)
    return result


def assert_formal_unlocked(*, actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for key in A2M_FORMAL_EVIDENCE:
        if key not in expected or actual.get(key) != expected.get(key):
            raise A2MTestUnlockError(f"formal evidence mismatch for {key}")
    if actual.get("formal_run_status") != A2M_FORMAL_STATUS:
        raise A2MTestUnlockError("formal_run_status must be FROZEN")


def _validate_a2m_config_set(configs: Mapping[str, Any], *, root: Path) -> None:
    try:
        validate_a2m_data_config(configs["data"])
        validate_a2m_eval_config(configs["eval"])
        validate_a2m_train_config(configs["train"])
        for model_config in configs["models"]:
            validate_a2m_model_config(model_config)
        for experiment_config in configs["experiments"]:
            validate_a2m_experiment_config(experiment_config)
    except (TypeError, ValueError) as exc:
        raise A2MProtocolError(str(exc)) from exc
    model_ids = [config["model_id"] for config in configs["models"]]
    if tuple(model_ids) != A2M_MODEL_IDS:
        raise A2MProtocolError("A2M model registry order is not frozen")
    train_recipe_names = {
        model_id: set(configs["train"]["recipes"][model_id])
        for model_id in A2M_MODEL_IDS
    }
    for model_config in configs["models"]:
        names = {recipe["name"] for recipe in model_config["recipes"]}
        if names != train_recipe_names[model_config["model_id"]]:
            raise A2MProtocolError(f"recipe registry mismatch for {model_config['model_id']}")
    chart = configs["chart"]
    if chart.get("schema_version") != "gf-a2m-chart-template-1" or chart.get("template_id") != "a2m-primary-v1":
        raise A2MProtocolError("A2M primary chart template is not frozen")
    del root


def _load_a2m_configs(root: Path) -> dict[str, Any]:
    paths: dict[str, Path] = {
        "data": root / A2M_DEFAULT_DATA_CONFIG,
        "eval": root / A2M_DEFAULT_EVAL_CONFIG,
        "train": root / A2M_DEFAULT_TRAIN_CONFIG,
        "chart": root / A2M_DEFAULT_CHART_CONFIG,
        "protocol": root / "configs/experiment/a2m_protocol.json",
        "dev": root / "configs/experiment/a2m_dev.json",
        "formal": root / "configs/experiment/a2m_formal.json",
    }
    model_paths = [root / path for path in A2M_DEFAULT_MODEL_CONFIGS]
    models = [_read_json(path) for path in model_paths]
    return {
        "paths": {**paths, "model_0": model_paths[0], "model_1": model_paths[1], "model_2": model_paths[2]},
        "data": _read_json(paths["data"]),
        "eval": _read_json(paths["eval"]),
        "train": _read_json(paths["train"]),
        "models": models,
        "model_by_id": {config["model_id"]: config for config in models},
        "chart": _read_json(paths["chart"]),
        "experiments": [_read_json(paths[key]) for key in ("protocol", "dev", "formal")],
    }


def _verify_a2h_source(root: Path, a2m_data_config: Mapping[str, Any]) -> dict[str, Any]:
    source = a2m_data_config["a2h_development_source"]
    source_config_path = _resolve(root, source["config_path"])
    source_manifest_path = _resolve(root, source["manifest_path"])
    if sha256_file(str(source_config_path)) != source["config_sha256"]:
        raise A2MProtocolError("A2H development config hash mismatch")
    manifest = _read_json(source_manifest_path)
    for key in ("schema_version", "data_version", "content_sha256", "split_family_hash"):
        if manifest.get(key) != source[key]:
            raise A2MProtocolError(f"A2H development source {key} mismatch")
    return {
        "config_path": _relative(root, source_config_path),
        "config_sha256": source["config_sha256"],
        "manifest_path": _relative(root, source_manifest_path),
        "data_version": source["data_version"],
        "content_sha256": source["content_sha256"],
        "split_family_hash": source["split_family_hash"],
        "allowed_splits": list(A2M_ALLOWED_DEV_SPLITS),
    }


def _fit_a2m_on_a2h(
    dataset: Any,
    *,
    axis: str,
    model_id: str,
    recipe_name: str,
    seed: int,
    model_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    root: Path,
    output_dir: Path,
    max_epochs_override: int | None,
) -> A2MFitResult:
    train_indices = dataset.indices(split_family=axis, split="train")
    validation_indices = dataset.indices(split_family=axis, split="val")
    stress_indices = dataset.indices(split_family=axis, split="stress_val")
    return _fit_a2m_samples(
        dataset.samples(train_indices),
        dataset.samples(validation_indices),
        dataset.samples(stress_indices),
        axis=axis,
        model_id=model_id,
        recipe_name=recipe_name,
        seed=seed,
        model_config=model_config,
        train_config=train_config,
        root=root,
        output_dir=output_dir,
        max_epochs_override=max_epochs_override,
    )


def _fit_a2m_pooled_on_a2h(
    dataset: Any,
    *,
    model_id: str,
    recipe_name: str,
    seed: int,
    model_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    root: Path,
    output_dir: Path,
    max_epochs_override: int | None,
) -> A2MFitResult:
    train_samples: list[UnifiedSample] = []
    validation_samples: list[UnifiedSample] = []
    validation_by_axis: dict[str, list[UnifiedSample]] = {}
    stress_by_axis: dict[str, list[UnifiedSample]] = {}
    for axis in A2M_AXES:
        train_samples.extend(dataset.samples(dataset.indices(split_family=axis, split="train")))
        validation_by_axis[axis] = dataset.samples(dataset.indices(split_family=axis, split="val"))
        validation_samples.extend(validation_by_axis[axis])
        if axis in A2M_PRIMARY_AXES:
            stress_by_axis[axis] = dataset.samples(dataset.indices(split_family=axis, split="stress_val"))
    stress_samples = [sample for axis in A2M_PRIMARY_AXES for sample in stress_by_axis[axis]]
    return _fit_a2m_samples(
        train_samples,
        validation_samples,
        stress_samples,
        axis="pooled",
        validation_by_axis=validation_by_axis,
        stress_by_axis=stress_by_axis,
        model_id=model_id,
        recipe_name=recipe_name,
        seed=seed,
        model_config=model_config,
        train_config=train_config,
        root=root,
        output_dir=output_dir,
        max_epochs_override=max_epochs_override,
    )


def _fit_a2m_samples(
    train_samples: Sequence[UnifiedSample],
    validation_samples: Sequence[UnifiedSample],
    stress_samples: Sequence[UnifiedSample],
    *,
    axis: str,
    validation_by_axis: Mapping[str, Sequence[UnifiedSample]] | None = None,
    stress_by_axis: Mapping[str, Sequence[UnifiedSample]] | None = None,
    model_id: str,
    recipe_name: str,
    seed: int,
    model_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    root: Path,
    output_dir: Path,
    max_epochs_override: int | None,
) -> A2MFitResult:
    del root
    if not train_samples or not validation_samples or not stress_samples:
        raise A2MProtocolError(f"A2M {axis} fit requires non-empty train, val, and stress_val")
    recipe = _find_model_recipe(model_config, recipe_name)
    train_recipe = train_config["recipes"][model_id][recipe_name]
    scaler = TrainGroupStandardScaler()
    scaler.fit(list(train_samples) + list(validation_samples), {sample.group_id for sample in train_samples})
    train_scaled = [scaler.transform(sample) for sample in train_samples]
    validation_scaled = [scaler.transform(sample) for sample in validation_samples]
    torch.manual_seed(seed)
    model = build_a2m_model(
        model_id,
        recipe,
        sensor_ids=model_config["sensor_ids"],
        sensor_types=model_config["sensor_types"],
        output_dim=model_config["head"]["output_dim"],
    )
    training_mapping = {
        "max_epochs": train_config["max_epochs"] if max_epochs_override is None else max_epochs_override,
        "optimizer": train_recipe["optimizer"],
        "loss": train_config["loss"],
        "early_stopping": {"patience": train_config["early_stopping"]["patience"]},
    }
    training = TorchTrainingConfig.from_mapping(training_mapping)
    checkpoint_path = output_dir / axis / model_id / recipe_name / f"seed_{seed}.pt"
    started = time.perf_counter()
    result = train_torch_model(
        model,
        train_scaled,
        validation_scaled,
        config=training,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )
    training_time = time.perf_counter() - started
    if result.best_epoch <= 0 or not checkpoint_path.is_file():
        raise A2MProtocolError(f"A2M {model_id}/{recipe_name}/seed_{seed} did not produce a checkpoint")
    validation_batch = collate_samples(tuple(validation_scaled))
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        validation_prediction = model(validation_batch).detach().cpu().numpy().astype(np.float64)
        validation_by_axis_metrics: dict[str, Mapping[str, Any]] = {}
        validation_samples_by_axis = validation_by_axis or {axis: validation_samples}
        for axis_name, axis_values in validation_samples_by_axis.items():
            scaled_axis_values = [scaler.transform(sample) for sample in axis_values]
            axis_batch = collate_samples(tuple(scaled_axis_values))
            axis_prediction = model(axis_batch).detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(axis_prediction).all():
                raise A2MProtocolError(f"A2M {model_id}/{recipe_name}/seed_{seed} produced non-finite output")
            axis_targets = np.vstack([sample.target for sample in axis_values]).astype(np.float64)
            axis_groups = tuple(sample.group_id for sample in axis_values)
            validation_by_axis_metrics[axis_name] = evaluate_predictions(
                axis_targets,
                axis_prediction,
                axis_groups,
                np.arange(len(axis_values), dtype=np.int64),
            )
        stress_predictions_by_axis: dict[str, np.ndarray] = {}
        stress_by_axis_metrics: dict[str, Mapping[str, Any]] = {}
        stress_targets_by_axis: dict[str, np.ndarray] = {}
        stress_groups_by_axis: dict[str, tuple[str, ...]] = {}
        evaluation_count = 0
        axis_samples = stress_by_axis or {axis: stress_samples}
        for axis_name, axis_values in axis_samples.items():
            scaled_axis_values = [scaler.transform(sample) for sample in axis_values]
            axis_batch = collate_samples(tuple(scaled_axis_values))
            axis_prediction = model(axis_batch).detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(axis_prediction).all():
                raise A2MProtocolError(f"A2M {model_id}/{recipe_name}/seed_{seed} produced non-finite output")
            axis_targets = np.vstack([sample.target for sample in axis_values]).astype(np.float64)
            axis_groups = tuple(sample.group_id for sample in axis_values)
            stress_predictions_by_axis[axis_name] = axis_prediction
            stress_targets_by_axis[axis_name] = axis_targets
            stress_groups_by_axis[axis_name] = axis_groups
            stress_by_axis_metrics[axis_name] = evaluate_predictions(
                axis_targets,
                axis_prediction,
                axis_groups,
                np.arange(len(axis_values), dtype=np.int64),
            )
            evaluation_count += len(axis_values)
        stress_prediction = np.vstack([stress_predictions_by_axis[name] for name in axis_samples])
    inference_time = time.perf_counter() - started
    if not np.isfinite(validation_prediction).all() or not np.isfinite(stress_prediction).all():
        raise A2MProtocolError(f"A2M {model_id}/{recipe_name}/seed_{seed} produced non-finite output")
    validation_targets = np.vstack([sample.target for sample in validation_samples]).astype(np.float64)
    stress_targets = np.vstack([sample.target for sample in stress_samples]).astype(np.float64)
    validation = evaluate_predictions(
        validation_targets,
        validation_prediction,
        [sample.group_id for sample in validation_samples],
        np.arange(len(validation_samples), dtype=np.int64),
    )
    stress = evaluate_predictions(
        stress_targets,
        stress_prediction,
        [sample.group_id for sample in stress_samples],
        np.arange(len(stress_samples), dtype=np.int64),
    )
    device_name = str(next(model.parameters()).device.type)
    peak_memory = int(torch.cuda.max_memory_allocated()) if device_name == "cuda" else 0
    resources = {
        "parameter_count": trainable_parameter_count(model),
        "training_time_s": float(training_time),
        "inference_time_s": float(inference_time / max(evaluation_count, 1)),
        "peak_memory_bytes": peak_memory,
        "best_epoch": int(result.best_epoch),
        "epochs_completed": int(result.epochs_completed),
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256_file(str(checkpoint_path)),
        "runtime_fingerprint": runtime_fingerprint(device=device_name, dtype="float32"),
        "failed": False,
    }
    return A2MFitResult(
        model_id=model_id,
        recipe_name=recipe_name,
        seed=seed,
        split_family=axis,
        validation=validation,
        validation_by_axis=validation_by_axis_metrics,
        stress=stress,
        validation_prediction=validation_prediction,
        stress_prediction=stress_prediction,
        stress_targets=stress_targets,
        stress_groups=tuple(sample.group_id for sample in stress_samples),
        stress_by_axis=stress_by_axis_metrics,
        stress_predictions_by_axis=stress_predictions_by_axis,
        stress_targets_by_axis=stress_targets_by_axis,
        stress_groups_by_axis=stress_groups_by_axis,
        resources=resources,
    )


def _fit_classical_baselines(dataset: Any, *, axis: str) -> dict[str, Any]:
    train_indices = dataset.indices(split_family=axis, split="train")
    val_indices = dataset.indices(split_family=axis, split="val")
    stress_indices = dataset.indices(split_family=axis, split="stress_val")
    samples = dataset.samples()
    scaler = TrainGroupStandardScaler()
    train_samples = [samples[int(index)] for index in train_indices]
    val_samples = [samples[int(index)] for index in val_indices]
    stress_samples = [samples[int(index)] for index in stress_indices]
    scaler.fit(train_samples + val_samples, {sample.group_id for sample in train_samples})
    all_scaled = [scaler.transform(sample) for sample in samples]
    features = np.asarray(
        [[float(sample.signals[sensor][0, 0]) for sensor in range(len(sample.signals))] for sample in all_scaled],
        dtype=np.float64,
    )
    targets = np.vstack([sample.target for sample in samples]).astype(np.float64)
    fitted = fit_full_regression_baselines(
        features,
        targets,
        train_indices,
        val_indices,
        seed=17,
    )
    result: dict[str, Any] = {}
    for model_id, model_result in fitted.items():
        prediction = model_result["prediction"]
        result[model_id] = {
            "validation": model_result["validation"],
            "stress": evaluate_predictions(
                targets[stress_indices],
                prediction[stress_indices],
                [samples[int(index)].group_id for index in stress_indices],
                np.arange(len(stress_indices)),
            ),
            "parameter_count": model_result["resources"]["parameter_count"],
        }
    return result


def run_a2m_development(
    *,
    project_root: str | Path,
    data_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    max_epochs_override: int | None = None,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    started_at = _utc_timestamp()
    configs = _load_a2m_configs(root)
    if data_config_path is not None:
        configs["data"] = _read_json(_resolve(root, data_config_path))
    if eval_config_path is not None:
        configs["eval"] = _read_json(_resolve(root, eval_config_path))
    if train_config_path is not None:
        configs["train"] = _read_json(_resolve(root, train_config_path))
    _validate_a2m_config_set(configs, root=root)
    _verify_a2h_source(root, configs["data"])
    dataset = load_a2h_dataset(root / "data" / "a2h_v2")
    run_records: list[A2MFitResult] = []
    classical: dict[str, Any] = {}
    for axis in A2M_PRIMARY_AXES:
        classical[axis] = _fit_classical_baselines(dataset, axis=axis)
    for model_id in A2M_MODEL_IDS:
        for recipe in configs["model_by_id"][model_id]["recipes"]:
            for seed in A2M_TRAINING_SEEDS:
                run_records.append(
                    _fit_a2m_pooled_on_a2h(
                        dataset,
                        model_id=model_id,
                        recipe_name=str(recipe["name"]),
                        seed=seed,
                        model_config=configs["model_by_id"][model_id],
                        train_config=configs["train"],
                        root=root,
                        output_dir=root / "outputs" / "runs" / "a2m" / "a2m-4-development",
                        max_epochs_override=max_epochs_override,
                    )
                )
    selected = _select_development_recipes(run_records, configs["eval"])
    promotion = _development_promotion(run_records, selected, configs["eval"], bootstrap_samples=bootstrap_samples)
    selected_checkpoint_paths = {
        model_id: {
            str(seed): next(
                record.resources["checkpoint_path"]
                for record in run_records
                if record.model_id == model_id
                and record.recipe_name == selected[model_id]["recipe_name"]
                and record.split_family == "pooled"
                and record.seed == seed
            )
            for seed in A2M_TRAINING_SEEDS
        }
        for model_id in A2M_MODEL_IDS
    }
    selected_checkpoint_hashes = {
        model_id: {
            seed: sha256_file(path)
            for seed, path in seed_paths.items()
        }
        for model_id, seed_paths in selected_checkpoint_paths.items()
    }
    selection = {
        "schema_version": "gf-a2m-selection-1",
        "status": promotion["status"],
        "selected_recipes": {model_id: selected[model_id]["recipe_name"] for model_id in A2M_MODEL_IDS},
        "selected_checkpoints": selected_checkpoint_paths,
        "selected_checkpoint_hashes": selected_checkpoint_hashes,
        "config_hashes": _config_hashes(configs),
        "formal_holdout_access": "locked",
    }
    finished_at = _utc_timestamp()
    summary = {
        "schema_version": "gf-a2m-dev-1",
        "stage": "A2M-4",
        "status": "PASS",
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_status": "SUCCESS",
        "primary_axes": list(A2M_PRIMARY_AXES),
        "run_count": len(run_records),
        "expected_run_count": 30,
        "runs": [_fit_to_record(record, include_predictions=False) for record in run_records],
        "models": _summarize_development_records(run_records),
        "classical_baselines": classical,
        "selection": selection,
        "promotion": promotion,
        "formal_holdout_access": "locked",
        "runtime_fingerprint": runtime_fingerprint(device="cpu", dtype="float32"),
    }
    _write_json(root / "outputs" / "summary" / "a2m" / "a2m_dev.json", summary)
    _write_json(root / "outputs" / "runs" / "a2m" / "selection.json", selection)
    _write_json(
        root / "outputs" / "runs" / "a2m" / "a2m-4-development" / "manifest.json",
        {
            "schema_version": A2M_RUN_MANIFEST_SCHEMA_VERSION,
            "stage": "A2M-4",
            "status": "PASS",
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_status": "SUCCESS",
            "run_count": len(run_records),
            "resources": _summarize_resource_mappings([record.resources for record in run_records]),
            "formal_holdout_access": "locked",
            "config_hashes": _config_hashes(configs),
        },
    )
    return summary


def _record_stress_metrics(record: A2MFitResult, axis: str) -> Mapping[str, Any]:
    try:
        return record.stress_by_axis[axis]
    except KeyError as exc:
        raise A2MProtocolError(f"A2M record {record.model_id}/{record.recipe_name}/seed_{record.seed} lacks stress axis {axis}") from exc


def _record_validation_metrics(record: A2MFitResult, axis: str) -> Mapping[str, Any]:
    try:
        return record.validation_by_axis[axis]
    except KeyError as exc:
        raise A2MProtocolError(f"A2M record {record.model_id}/{record.recipe_name}/seed_{record.seed} lacks validation axis {axis}") from exc


def _record_stress_prediction(record: A2MFitResult, axis: str) -> np.ndarray:
    try:
        return record.stress_predictions_by_axis[axis]
    except KeyError as exc:
        raise A2MProtocolError(f"A2M record {record.model_id}/{record.recipe_name}/seed_{record.seed} lacks stress predictions for {axis}") from exc


def _record_stress_targets(record: A2MFitResult, axis: str) -> np.ndarray:
    try:
        return record.stress_targets_by_axis[axis]
    except KeyError as exc:
        raise A2MProtocolError(f"A2M record {record.model_id}/{record.recipe_name}/seed_{record.seed} lacks stress targets for {axis}") from exc


def _record_stress_groups(record: A2MFitResult, axis: str) -> tuple[str, ...]:
    try:
        return record.stress_groups_by_axis[axis]
    except KeyError as exc:
        raise A2MProtocolError(f"A2M record {record.model_id}/{record.recipe_name}/seed_{record.seed} lacks stress groups for {axis}") from exc


def _select_development_recipes(records: Sequence[A2MFitResult], eval_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    del eval_config
    selected: dict[str, dict[str, Any]] = {}
    for model_id in A2M_MODEL_IDS:
        recipe_names = sorted({record.recipe_name for record in records if record.model_id == model_id})
        candidates = []
        for recipe_name in recipe_names:
            values = [
                float(
                    np.mean(
                        [
                            _record_stress_metrics(record, axis)["macro_RNMAE"]
                            for axis in A2M_PRIMARY_AXES
                        ]
                    )
                )
                for record in records
                if record.model_id == model_id and record.recipe_name == recipe_name
            ]
            candidates.append(
                {
                    "model_id": model_id,
                    "recipe_name": recipe_name,
                    "mean_primary_stress_macro_RNMAE": float(np.mean(values)),
                    "seed_count": len(values),
                }
            )
        selected[model_id] = min(candidates, key=lambda value: value["mean_primary_stress_macro_RNMAE"])
    return selected


def _development_promotion(
    records: Sequence[A2MFitResult],
    selected: Mapping[str, Mapping[str, Any]],
    eval_config: Mapping[str, Any],
    *,
    bootstrap_samples: int | None,
) -> dict[str, Any]:
    promotion = eval_config["promotion"]
    baseline_recipe = selected["A2M-MLP"]["recipe_name"]
    candidate_rows: dict[str, Any] = {}
    for model_id in ("A2M-RESNET", "A2M-FTT"):
        candidate_recipe = selected[model_id]["recipe_name"]
        axis_rows: dict[str, Any] = {}
        for axis in A2M_PRIMARY_AXES:
            baseline = [record for record in records if record.model_id == "A2M-MLP" and record.recipe_name == baseline_recipe and record.split_family == "pooled"]
            candidate = [record for record in records if record.model_id == model_id and record.recipe_name == candidate_recipe and record.split_family == "pooled"]
            baseline_by_seed = {record.seed: record for record in baseline}
            candidate_by_seed = {record.seed: record for record in candidate}
            baseline_mean = float(np.mean([_record_stress_metrics(record, axis)["macro_RNMAE"] for record in baseline]))
            candidate_mean = float(np.mean([_record_stress_metrics(record, axis)["macro_RNMAE"] for record in candidate]))
            direction_count = sum(
                _record_stress_metrics(candidate_by_seed[seed], axis)["macro_RNMAE"]
                < _record_stress_metrics(baseline_by_seed[seed], axis)["macro_RNMAE"]
                for seed in A2M_TRAINING_SEEDS
            )
            baseline_prediction = np.mean([_record_stress_prediction(record, axis) for record in baseline], axis=0)
            candidate_prediction = np.mean([_record_stress_prediction(record, axis) for record in candidate], axis=0)
            bootstrap = _bootstrap_from_records(
                candidate,
                baseline,
                axis=axis,
                samples=2000 if bootstrap_samples is None else bootstrap_samples,
                seed=int(eval_config["bootstrap_seed"]),
            )
            baseline_components = np.mean([_record_stress_metrics(record, axis)["component_RNMAE"] for record in baseline], axis=0)
            candidate_components = np.mean([_record_stress_metrics(record, axis)["component_RNMAE"] for record in candidate], axis=0)
            relative_improvement = (baseline_mean - candidate_mean) / baseline_mean
            axis_rows[axis] = {
                "baseline_mean_macro_RNMAE": baseline_mean,
                "candidate_mean_macro_RNMAE": candidate_mean,
                "relative_improvement": float(relative_improvement),
                "seed_same_direction": int(direction_count),
                "component_delta": [float(value) for value in candidate_components - baseline_components],
                "bootstrap": bootstrap,
                "mean_prediction_finite": bool(np.isfinite(baseline_prediction).all() and np.isfinite(candidate_prediction).all()),
                "passes_axis": bool(
                    relative_improvement >= float(promotion["min_relative_improvement"])
                    and direction_count >= int(promotion["min_seeds_same_direction"])
                    and bootstrap["percentile_97_5"] < 0.0
                    and np.max(candidate_components - baseline_components) <= float(promotion["max_component_absolute_degradation"])
                ),
            }
        iid = axis_rows["iid"]
        qualifying = [axis for axis in A2M_PRIMARY_AXES if axis != "iid" and axis_rows[axis]["passes_axis"]]
        iid_relative_degradation = (
            iid["candidate_mean_macro_RNMAE"] - iid["baseline_mean_macro_RNMAE"]
        ) / iid["baseline_mean_macro_RNMAE"]
        candidate_rows[model_id] = {
            "recipe_name": candidate_recipe,
            "iid_relative_degradation": float(iid_relative_degradation),
            "qualifying_axes": qualifying,
            "axis_rows": axis_rows,
            "matched_optimizer_audit": {"status": "REQUIRED_IF_PROMOTED"},
            "passes_development_gate": bool(
                iid_relative_degradation <= float(promotion["max_iid_relative_degradation"])
                and len(qualifying) >= int(promotion["min_stress_axes"])
            ),
        }
    passing = [model_id for model_id, row in candidate_rows.items() if row["passes_development_gate"]]
    return {
        "status": "PASS_CANDIDATE" if passing else "MLP_RETAINED",
        "baseline": {"model_id": "A2M-MLP", "recipe_name": baseline_recipe},
        "candidates": candidate_rows,
        "passing_candidates": passing,
        "matched_audit_required": bool(passing),
    }


def _bootstrap_from_records(
    candidate: Sequence[A2MFitResult],
    baseline: Sequence[A2MFitResult],
    *,
    axis: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if len(candidate) != len(baseline) or not candidate:
        raise A2MProtocolError("paired bootstrap requires equal non-empty seed records")
    candidate_by_seed = {record.seed: record for record in candidate}
    baseline_by_seed = {record.seed: record for record in baseline}
    if set(candidate_by_seed) != set(baseline_by_seed):
        raise A2MProtocolError("paired bootstrap seed sets do not match")
    first_candidate = candidate_by_seed[min(candidate_by_seed)]
    first_baseline = baseline_by_seed[min(baseline_by_seed)]
    candidate_groups = _record_stress_groups(first_candidate, axis)
    baseline_groups = _record_stress_groups(first_baseline, axis)
    if candidate_groups != baseline_groups:
        raise A2MProtocolError("paired bootstrap mixture_id groups do not match")
    ordered_seeds = sorted(candidate_by_seed)
    candidate_prediction = np.mean([_record_stress_prediction(candidate_by_seed[record_seed], axis) for record_seed in ordered_seeds], axis=0)
    baseline_prediction = np.mean([_record_stress_prediction(baseline_by_seed[record_seed], axis) for record_seed in ordered_seeds], axis=0)
    targets = _record_stress_targets(first_candidate, axis)
    comparison = group_bootstrap_comparison(
        candidate_prediction,
        baseline_prediction,
        targets,
        candidate_groups,
        seed=seed,
        samples=samples,
        indices=np.arange(len(targets), dtype=np.int64),
    )
    comparison["unit"] = "paired_mixture_id"
    comparison["seed_metric_differences"] = {
        str(record_seed): float(
            _record_stress_metrics(candidate_by_seed[record_seed], axis)["macro_RNMAE"]
            - _record_stress_metrics(baseline_by_seed[record_seed], axis)["macro_RNMAE"]
        )
        for record_seed in sorted(candidate_by_seed)
    }
    return comparison


def _summarize_development_records(records: Sequence[A2MFitResult]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_id in A2M_MODEL_IDS:
        result[model_id] = {}
        for recipe_name in sorted({record.recipe_name for record in records if record.model_id == model_id}):
            selected = [record for record in records if record.model_id == model_id and record.recipe_name == recipe_name]
            result[model_id][recipe_name] = {
                "seed_count": len({record.seed for record in selected}),
                "axis_count": len(A2M_PRIMARY_AXES),
                "axes": {
                    axis: {
                        "validation_macro_RNMAE_mean": float(np.mean([_record_validation_metrics(record, axis)["macro_RNMAE"] for record in selected])),
                        "stress_macro_RNMAE_mean": float(np.mean([_record_stress_metrics(record, axis)["macro_RNMAE"] for record in selected])),
                        "stress_macro_RNMAE_std": float(np.std([_record_stress_metrics(record, axis)["macro_RNMAE"] for record in selected], ddof=0)),
                    }
                    for axis in A2M_PRIMARY_AXES
                },
                "parameter_counts": sorted({int(record.resources["parameter_count"]) for record in selected}),
                "checkpoint_hashes": sorted({str(record.resources["checkpoint_sha256"]) for record in selected}),
            }
    return result


def run_a2m_formal(
    *,
    project_root: str | Path,
    data_config_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    unlock_formal: bool,
    formal_run_status: str | None,
    bootstrap_samples: int | None = None,
) -> dict[str, Any]:
    started_at = _utc_timestamp()
    if not unlock_formal or formal_run_status != A2M_FORMAL_STATUS:
        raise A2MTestUnlockError("A2M formal requires --unlock-formal and formal_run_status=FROZEN")
    root = _project_root(project_root)
    configs = _load_a2m_configs(root)
    if data_config_path is not None:
        configs["data"] = _read_json(_resolve(root, data_config_path))
    if eval_config_path is not None:
        configs["eval"] = _read_json(_resolve(root, eval_config_path))
    if train_config_path is not None:
        configs["train"] = _read_json(_resolve(root, train_config_path))
    _validate_a2m_config_set(configs, root=root)
    generation = run_a2m_generation(project_root=root, data_config_path=data_config_path)
    selection_path = root / "outputs" / "runs" / "a2m" / "selection.json"
    if not selection_path.is_file():
        raise A2MProtocolError("A2M formal requires a completed development selection")
    selection = _read_json(selection_path)
    expected_hashes = _config_hashes(configs)
    if selection.get("config_hashes") != expected_hashes:
        raise A2MProtocolError("A2M selection config hashes do not match formal config")
    data_manifest_path = root / A2M_DEFAULT_HOLDOUT_DIR / "manifest.json"
    data_manifest = _read_json(data_manifest_path)
    selected_checkpoint_hashes = _required_mapping(selection, "selected_checkpoint_hashes")
    for model_id in A2M_MODEL_IDS:
        model_hashes = _required_mapping(selected_checkpoint_hashes, model_id)
        model_paths = _required_mapping(selection["selected_checkpoints"], model_id)
        for seed in A2M_TRAINING_SEEDS:
            seed_key = str(seed)
            checkpoint_path = Path(str(model_paths[seed_key]))
            expected_checkpoint_hash = str(model_hashes[seed_key])
            if sha256_file(str(checkpoint_path)) != expected_checkpoint_hash:
                raise A2MProtocolError(f"A2M selected checkpoint hash mismatch: {model_id}/seed_{seed}")
    evidence = {
        "data_content_sha256": data_manifest["content_sha256"],
        "split_hash": data_manifest["split_hash"],
        "profile_hash": data_manifest["profile_hash"],
        "protocol_config_sha256": expected_hashes["protocol"],
        "checkpoint_sha256": canonical_sha256(selected_checkpoint_hashes),
        "runtime_fingerprint": runtime_fingerprint(device="cpu", dtype="float32"),
        "primary_chart_template_sha256": expected_hashes["chart"],
        "formal_run_status": A2M_FORMAL_STATUS,
    }
    access_path = root / "outputs" / "runs" / "a2m" / "formal_access.json"
    _claim_formal_access(access_path, evidence=evidence)
    # Formal labels are first loaded after the evidence and one-time access claim.
    formal_dataset = load_a2m_dataset(root / A2M_DEFAULT_HOLDOUT_DIR, include_formal=True)
    development_dataset = load_a2h_dataset(root / "data" / "a2h_v2")
    formal_records: list[dict[str, Any]] = []
    for model_id in A2M_MODEL_IDS:
        recipe_name = selection["selected_recipes"][model_id]
        for seed in A2M_TRAINING_SEEDS:
            train_indices = development_dataset.indices(split_family="iid", split="train")
            val_indices = development_dataset.indices(split_family="iid", split="val")
            train_samples = development_dataset.samples(train_indices)
            val_samples = development_dataset.samples(val_indices)
            formal_samples = formal_dataset.samples()
            fit = _fit_a2m_formal_samples(
                train_samples,
                val_samples,
                formal_samples,
                model_id=model_id,
                recipe_name=recipe_name,
                seed=seed,
                model_config=configs["model_by_id"][model_id],
                train_config=configs["train"],
                output_dir=root / "outputs" / "runs" / "a2m" / "a2m-5-formal",
            )
            formal_records.append(fit)
    summary = _summarize_formal_records(formal_records, formal_dataset, configs["eval"], selection, evidence, bootstrap_samples)
    finished_at = _utc_timestamp()
    summary["started_at"] = started_at
    summary["finished_at"] = finished_at
    summary["exit_status"] = "SUCCESS"
    summary["manifest"].update(
        {
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_status": "SUCCESS",
            "resources": _summarize_resource_mappings([record["resources"] for record in formal_records]),
        }
    )
    summary["generation"] = generation
    _write_json(root / "outputs" / "summary" / "a2m" / "a2m_formal.json", summary)
    _write_json(root / "outputs" / "runs" / "a2m" / "a2m-5-formal" / "manifest.json", summary["manifest"])
    return summary


def _fit_a2m_formal_samples(
    train_samples: Sequence[UnifiedSample],
    validation_samples: Sequence[UnifiedSample],
    formal_samples: Sequence[UnifiedSample],
    *,
    model_id: str,
    recipe_name: str,
    seed: int,
    model_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    scaler = TrainGroupStandardScaler()
    scaler.fit(list(train_samples) + list(validation_samples), {sample.group_id for sample in train_samples})
    train_scaled = [scaler.transform(sample) for sample in train_samples]
    validation_scaled = [scaler.transform(sample) for sample in validation_samples]
    formal_scaled = [scaler.transform(sample) for sample in formal_samples]
    recipe = _find_model_recipe(model_config, recipe_name)
    train_recipe = train_config["recipes"][model_id][recipe_name]
    torch.manual_seed(seed)
    model = build_a2m_model(
        model_id,
        recipe,
        sensor_ids=model_config["sensor_ids"],
        sensor_types=model_config["sensor_types"],
        output_dim=model_config["head"]["output_dim"],
    )
    training = TorchTrainingConfig.from_mapping(
        {
            "max_epochs": train_config["max_epochs"],
            "optimizer": train_recipe["optimizer"],
            "loss": train_config["loss"],
            "early_stopping": {"patience": train_config["early_stopping"]["patience"]},
        }
    )
    checkpoint_path = output_dir / model_id / recipe_name / f"seed_{seed}.pt"
    started = time.perf_counter()
    result = train_torch_model(
        model,
        train_scaled,
        validation_scaled,
        config=training,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )
    training_time = time.perf_counter() - started
    if result.best_epoch <= 0 or not checkpoint_path.is_file():
        raise A2MProtocolError(f"formal {model_id}/{recipe_name}/seed_{seed} has no checkpoint")
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        prediction = model(collate_samples(tuple(formal_scaled))).detach().cpu().numpy().astype(np.float64)
    inference_time = time.perf_counter() - started
    if not np.isfinite(prediction).all():
        raise A2MProtocolError(f"formal {model_id}/{recipe_name}/seed_{seed} produced non-finite output")
    return {
        "model_id": model_id,
        "recipe_name": recipe_name,
        "seed": seed,
        "prediction": prediction,
        "validation_macro_RNMAE": result.best_validation_macro_RNMAE,
        "resources": {
            "parameter_count": trainable_parameter_count(model),
            "training_time_s": float(training_time),
            "inference_time_s": float(inference_time / max(len(formal_samples), 1)),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if str(next(model.parameters()).device.type) == "cuda" else 0,
            "best_epoch": int(result.best_epoch),
            "epochs_completed": int(result.epochs_completed),
            "checkpoint_sha256": sha256_file(str(checkpoint_path)),
            "checkpoint_path": checkpoint_path.as_posix(),
            "runtime_fingerprint": runtime_fingerprint(device=str(next(model.parameters()).device.type), dtype="float32"),
            "failed": False,
        },
    }


def _summarize_formal_records(
    records: Sequence[Mapping[str, Any]],
    dataset: Any,
    eval_config: Mapping[str, Any],
    selection: Mapping[str, Any],
    evidence: Mapping[str, Any],
    bootstrap_samples: int | None,
) -> dict[str, Any]:
    targets = np.vstack([observation.composition for observation in dataset.observations]).astype(np.float64)
    groups = np.asarray(dataset.group_ids, dtype=object)
    rows: dict[str, Any] = {}
    for model_id in A2M_MODEL_IDS:
        model_records = [record for record in records if record["model_id"] == model_id]
        predictions = np.asarray([record["prediction"] for record in model_records], dtype=np.float64)
        mean_prediction = predictions.mean(axis=0)
        axis_rows: dict[str, Any] = {}
        for axis in A2M_AXES:
            indices = dataset.indices(axis=axis)
            metrics = evaluate_predictions(targets, mean_prediction, groups, indices)
            axis_rows[axis] = {
                "ensemble_mean": metrics,
                "seed_records": [
                    {
                        "seed": record["seed"],
                        "metrics": evaluate_predictions(targets, record["prediction"], groups, indices),
                        "resources": record["resources"],
                    }
                    for record in model_records
                ],
            }
        rows[model_id] = {
            "recipe_name": model_records[0]["recipe_name"],
            "axes": axis_rows,
            "parameter_counts": sorted({record["resources"]["parameter_count"] for record in model_records}),
            "checkpoint_hashes": sorted({record["resources"]["checkpoint_sha256"] for record in model_records}),
        }
    baseline = rows["A2M-MLP"]
    for model_id in ("A2M-RESNET", "A2M-FTT"):
        rows[model_id]["formal_comparison"] = _formal_comparison(
            rows[model_id],
            baseline,
            dataset,
            eval_config,
            candidate_model_id=model_id,
            baseline_model_id="A2M-MLP",
            raw_records=records,
            bootstrap_samples=bootstrap_samples,
        )
    passing = [
        model_id
        for model_id in ("A2M-RESNET", "A2M-FTT")
        if rows[model_id]["formal_comparison"]["passes_formal_gate"]
    ]
    final_state = "POSITIVE_RESULT" if passing else "MLP_RETAINED"
    manifest = {
        "schema_version": A2M_RUN_MANIFEST_SCHEMA_VERSION,
        "stage": "A2M-5",
        "status": "PASS",
        "run_count": len(records),
        "expected_run_count": len(A2M_MODEL_IDS) * len(A2M_TRAINING_SEEDS),
        "formal_run_status": A2M_FORMAL_STATUS,
        "test_unlocked": True,
        "evidence": dict(evidence),
        "selection": dict(selection),
        "runtime_fingerprint": evidence["runtime_fingerprint"],
    }
    return {
        "schema_version": "gf-a2m-formal-1",
        "stage": "A2M-5",
        "status": "PASS",
        "run_count": len(records),
        "expected_run_count": len(A2M_MODEL_IDS) * len(A2M_TRAINING_SEEDS),
        "final_state": final_state,
        "passing_candidates": passing,
        "models": rows,
        "formal_run_status": A2M_FORMAL_STATUS,
        "manifest": manifest,
    }


def _formal_comparison(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    dataset: Any,
    eval_config: Mapping[str, Any],
    *,
    candidate_model_id: str,
    baseline_model_id: str,
    raw_records: Sequence[Mapping[str, Any]],
    bootstrap_samples: int | None,
) -> dict[str, Any]:
    promotion = eval_config["promotion"]
    axis_rows: dict[str, Any] = {}
    qualifying: list[str] = []
    for axis in A2M_PRIMARY_AXES:
        candidate_mean = candidate["axes"][axis]["ensemble_mean"]
        baseline_mean = baseline["axes"][axis]["ensemble_mean"]
        candidate_seeds = {row["seed"]: row["metrics"] for row in candidate["axes"][axis]["seed_records"]}
        baseline_seeds = {row["seed"]: row["metrics"] for row in baseline["axes"][axis]["seed_records"]}
        direction_count = sum(candidate_seeds[seed]["macro_RNMAE"] < baseline_seeds[seed]["macro_RNMAE"] for seed in A2M_TRAINING_SEEDS)
        indices = dataset.indices(axis=axis)
        candidate_model_records = [record for record in raw_records if record["model_id"] == candidate_model_id]
        baseline_model_records = [record for record in raw_records if record["model_id"] == baseline_model_id]
        candidate_model_records = [
            record for record in candidate_model_records if record["recipe_name"] == candidate["recipe_name"]
        ]
        baseline_model_records = [
            record for record in baseline_model_records if record["recipe_name"] == baseline["recipe_name"]
        ]
        candidate_prediction = np.mean(
            [record["prediction"] for record in candidate_model_records],
            axis=0,
        )
        baseline_prediction = np.mean(
            [record["prediction"] for record in baseline_model_records],
            axis=0,
        )
        targets = np.vstack([observation.composition for observation in dataset.observations]).astype(np.float64)
        groups = np.asarray(dataset.group_ids, dtype=object)
        bootstrap = group_bootstrap_comparison(
            candidate_prediction,
            baseline_prediction,
            targets,
            groups,
            seed=int(eval_config["bootstrap_seed"]),
            samples=2000 if bootstrap_samples is None else bootstrap_samples,
            indices=indices,
        )
        bootstrap["unit"] = "paired_mixture_id"
        relative_improvement = (baseline_mean["macro_RNMAE"] - candidate_mean["macro_RNMAE"]) / baseline_mean["macro_RNMAE"]
        component_delta = np.asarray(candidate_mean["component_RNMAE"]) - np.asarray(baseline_mean["component_RNMAE"])
        pass_axis = bool(
            relative_improvement >= float(promotion["min_relative_improvement"])
            and direction_count >= int(promotion["min_seeds_same_direction"])
            and np.max(component_delta) <= float(promotion["max_component_absolute_degradation"])
            and bootstrap["percentile_97_5"] < 0.0
        )
        if pass_axis and axis != "iid":
            qualifying.append(axis)
        axis_rows[axis] = {
            "relative_improvement": float(relative_improvement),
            "seed_same_direction": int(direction_count),
            "component_delta": [float(value) for value in component_delta],
            "bootstrap": bootstrap,
            "passes_axis": pass_axis,
        }
    iid_degradation = (
        candidate["axes"]["iid"]["ensemble_mean"]["macro_RNMAE"]
        - baseline["axes"]["iid"]["ensemble_mean"]["macro_RNMAE"]
    ) / baseline["axes"]["iid"]["ensemble_mean"]["macro_RNMAE"]
    return {
        "iid_relative_degradation": float(iid_degradation),
        "qualifying_axes": qualifying,
        "axis_rows": axis_rows,
        "passes_formal_gate": bool(
            iid_degradation <= float(promotion["max_iid_relative_degradation"])
            and len(qualifying) >= int(promotion["min_stress_axes"])
            and all(axis_rows[axis]["bootstrap"]["ci_excludes_zero"] for axis in qualifying)
        ),
    }


def _claim_formal_access(path: Path, *, evidence: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": "gf-a2m-formal-access-1",
        "status": "CLAIMED",
        "evidence_sha256": canonical_sha256(evidence),
        "evidence": dict(evidence),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise A2MTestUnlockError(f"A2M formal access was already claimed: {path.as_posix()}") from exc
    return payload


def _config_hashes(configs: Mapping[str, Any]) -> dict[str, str]:
    return {
        "data": sha256_file(str(configs["paths"]["data"])),
        "eval": sha256_file(str(configs["paths"]["eval"])),
        "train": sha256_file(str(configs["paths"]["train"])),
        "model_A2M-MLP": sha256_file(str(configs["paths"]["model_0"])),
        "model_A2M-RESNET": sha256_file(str(configs["paths"]["model_1"])),
        "model_A2M-FTT": sha256_file(str(configs["paths"]["model_2"])),
        "protocol": sha256_file(str(configs["paths"]["protocol"])),
        "dev": sha256_file(str(configs["paths"]["dev"])),
        "formal": sha256_file(str(configs["paths"]["formal"])),
        "chart": sha256_file(str(configs["paths"]["chart"])),
    }


def _fit_to_record(result: A2MFitResult, *, include_predictions: bool) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model_id": result.model_id,
        "recipe_name": result.recipe_name,
        "seed": result.seed,
        "split_family": result.split_family,
        "validation": dict(result.validation),
        "validation_by_axis": {axis: dict(metrics) for axis, metrics in result.validation_by_axis.items()},
        "stress": dict(result.stress),
        "stress_by_axis": {axis: dict(metrics) for axis, metrics in result.stress_by_axis.items()},
        "resources": dict(result.resources),
    }
    if include_predictions:
        record["validation_prediction"] = result.validation_prediction.tolist()
        record["stress_prediction"] = result.stress_prediction.tolist()
    return record


def _find_model_recipe(model_config: Mapping[str, Any], recipe_name: str) -> Mapping[str, Any]:
    matches = [recipe for recipe in model_config["recipes"] if recipe.get("name") == recipe_name]
    if len(matches) != 1:
        raise A2MProtocolError(f"recipe {recipe_name!r} must resolve exactly once")
    return matches[0]


def _read_a1_b5_mean_predictions(path: Path, expected_count: int) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise A2MProtocolError("A1 predictions row count does not match the frozen manifest")
    return np.asarray(
        [
            [float(row[f"B5__mean__{name}"]) for name in ("x_Ar_pct", "x_He_pct", "x_CO2_pct")]
            for row in rows
        ],
        dtype=np.float64,
    )


def _summarize_resource_mappings(resources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not resources:
        raise A2MProtocolError("resource summary requires at least one completed run")
    return {
        "run_count": len(resources),
        "failed_count": sum(bool(resource["failed"]) for resource in resources),
        "parameter_counts": sorted({int(resource["parameter_count"]) for resource in resources}),
        "training_time_s_total": float(sum(float(resource["training_time_s"]) for resource in resources)),
        "inference_time_s_total": float(sum(float(resource["inference_time_s"]) for resource in resources)),
        "peak_memory_bytes_max": max(int(resource["peak_memory_bytes"]) for resource in resources),
    }


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _required_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise A2MProtocolError(f"{key} must be an object")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise A2MProtocolError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _project_root(project_root: str | Path | None) -> Path:
    return Path(project_root or Path(__file__).resolve().parents[3]).resolve()


__all__ = [
    "A2MFitResult",
    "A2MProtocolError",
    "A2MTestUnlockError",
    "A2M_FORMAL_EVIDENCE",
    "A2M_FORMAL_STATUS",
    "A2M_TRAINING_SEEDS",
    "assert_formal_unlocked",
    "build_a2m_run_manifest",
    "run_a2m",
    "run_a2m_a1_reproduction",
    "run_a2m_development",
    "run_a2m_formal",
    "run_a2m_generation",
    "run_a2m_protocol",
    "run_a2m_smoke",
    "validate_a2m_eval_config",
    "validate_a2m_experiment_config",
    "validate_a2m_train_config",
]
