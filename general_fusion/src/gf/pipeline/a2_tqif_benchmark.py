from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import tracemalloc
import sys
from typing import Any, Callable

import numpy as np
import torch

from gf.dl.contracts import UnifiedBatch, collate_samples
from gf.dl.evaluation import evaluate_predictions, group_bootstrap_comparison
from gf.dl.mainstream_architectures import build_a2m_model, validate_a2m_model_config
from gf.dl.tqif import (
    TQIFModel,
    TQIFTargetSlotRegistry,
    build_tqif_matched_concat_model,
    build_tqif_model,
    load_tqif_checkpoint,
    sensor_registry_hash,
    target_slot_registry_hash,
)
from gf.dl.training import (
    TorchTrainingConfig,
    prepare_a2_train_val_samples,
    train_torch_model,
    trainable_parameter_count,
)
from gf.pipeline.tqif_common import (
    TQIFArtifactError,
    binary_sha256,
    canonical_hash,
    canonical_run_input_hash,
    exclusive_lock,
    git_snapshot,
    normalized_text_hash,
    read_json_object,
    relative_path,
    resolve_project_file,
    write_json,
)
from gf.sim.a1_dataset import load_dataset_splits


TQIF_PROTOCOL_SCHEMA_VERSION = "tqif-protocol-1"
TQIF_RUN_SCHEMA_VERSION = "tqif-run-1"
TQIF_PREDICTION_SCHEMA_VERSION = "tqif-prediction-1"
TQIF_METRICS_SCHEMA_VERSION = "tqif-metrics-1"
TQIF_PARAMETER_PROFILE_SCHEMA_VERSION = "tqif-parameter-profile-1"
TQIF_SELECTION_SCHEMA_VERSION = "tqif-selection-1"
TQIF_PREDICTION_FIELDS = (
    "run_id",
    "phase",
    "split",
    "seed",
    "sample_id",
    "mixture_id",
    "y_true_x_Ar_pct",
    "y_true_x_He_pct",
    "y_true_x_CO2_pct",
    "pred_x_Ar_pct",
    "pred_x_He_pct",
    "pred_x_CO2_pct",
    "model_id",
    "recipe_id",
)
TQIF_ALLOWED_READ_SPLITS = ("train", "inner_oof", "val")
TQIF_SEEDS = (17, 29, 43, 71, 101)
TQIF_FORBIDDEN_KEYS = frozenset(
    {"base_condition_id", "noise_seed_index", "noise_seed", "sequence_id"}
)
TQIF_COMPARISONS = (
    ("TQIF-H0", "C0"),
    ("Q1", "C1"),
    ("TQIF-H0", "I1"),
    ("I1", "C1"),
    ("TQIF-H0", "Q1"),
    ("TQIF-STR", "TQIF-H0"),
)
TQIF_CAPACITY_CANDIDATES = (
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
    160,
    192,
    224,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
)
TQIF_PARAMETER_PROFILE_RUNTIME_FIELDS = frozenset(
    {"peak_memory_bytes", "single_batch_forward_ms"}
)


class TQIFProtocolError(TQIFArtifactError):
    """Raised for a reproducible TQIF protocol or artifact failure."""


def run_tqif(
    *,
    project_root: str | Path | None = None,
    stage: str = "protocol",
    protocol_config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    if stage == "protocol":
        return run_tqif_protocol(
            project_root=root,
            protocol_config_path=protocol_config_path,
        )
    if stage == "smoke":
        return run_tqif_smoke(project_root=root)
    if stage == "select":
        _require_protocol_passed(root)
        return run_tqif_select(project_root=root)
    if stage == "baseline":
        _require_protocol_passed(root)
        return run_tqif_baseline(project_root=root)
    if stage == "development":
        return run_tqif_development(project_root=root)
    if stage == "ablation":
        return run_tqif_ablation(project_root=root)
    if stage == "all":
        protocol = run_tqif_protocol(
            project_root=root,
            protocol_config_path=protocol_config_path,
        )
        smoke = run_tqif_smoke(project_root=root)
        baseline = run_tqif_baseline(project_root=root)
        development = run_tqif_development(project_root=root)
        ablation = (
            run_tqif_ablation(project_root=root)
            if development["status"] == "PASS"
            else None
        )
        selection = run_tqif_select(project_root=root)
        return {
            "stage": "all",
            "status": selection["selection_status"],
            "protocol": protocol,
            "smoke": smoke,
            "baseline": baseline,
            "development": development,
            "ablation": ablation,
            "selection": selection,
        }
    raise ValueError(
        "supported TQIF stages are protocol, smoke, baseline, development, ablation, select, and all"
    )


def run_tqif_protocol(
    *,
    project_root: str | Path | None = None,
    protocol_config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = _project_root(project_root)
    started_at = _utc_now()
    configured_protocol = (
        Path(protocol_config_path)
        if protocol_config_path is not None
        else root / "configs" / "experiment" / "a2_tqif_protocol.json"
    )
    protocol_path = resolve_project_file(root, configured_protocol)
    protocol_config = read_json_object(protocol_path)
    _validate_protocol_definition(protocol_config)

    data_config_path = _resolve_configured_path(root, protocol_config, "data_config")
    data_manifest_path = _resolve_configured_path(root, protocol_config, "data_manifest")
    eval_config_path = _resolve_configured_path(root, protocol_config, "eval_config")
    train_config_path = _resolve_configured_path(root, protocol_config, "train_config")
    ablation_config_path = _resolve_configured_path(root, protocol_config, "ablation_config")
    model_paths = [
        _resolve_configured_path(root, protocol_config, "model_configs", index=index)
        for index in range(len(protocol_config["model_configs"]))
    ]
    baseline_model_path = _resolve_configured_path(
        root,
        protocol_config,
        "baseline_model_config",
    )
    data_config = read_json_object(data_config_path)
    data_manifest = read_json_object(data_manifest_path)
    eval_config = read_json_object(eval_config_path)
    train_config = read_json_object(train_config_path)
    ablation_config = read_json_object(ablation_config_path)
    model_configs = [read_json_object(path) for path in model_paths]
    baseline_model_config = read_json_object(baseline_model_path)

    _validate_no_forbidden_keys(data_config)
    _validate_no_forbidden_keys(data_manifest)
    _validate_no_forbidden_keys(train_config)
    _validate_no_forbidden_keys(eval_config)
    _validate_no_forbidden_keys(ablation_config)
    for config in model_configs:
        _validate_no_forbidden_keys(config)
    _validate_no_forbidden_keys(baseline_model_config)
    _validate_data_manifest(data_manifest)
    _validate_tqif_train_config(train_config)
    _validate_tqif_eval_config(eval_config)
    _validate_tqif_ablation_config(ablation_config)
    _validate_model_bundle(model_configs)
    validate_a2m_model_config(baseline_model_config)
    if baseline_model_config.get("model_id") != "A2M-MLP":
        raise TQIFProtocolError("INVALID_ARTIFACT", "baseline model must be A2M-MLP")
    if protocol_config.get("baseline_recipe") != "mlp_lbfgs_width32":
        raise TQIFProtocolError("INVALID_ARTIFACT", "baseline recipe must be mlp_lbfgs_width32")
    if protocol_config.get("baseline_reference_mode") != "rebuild_current_a2_v1":
        raise TQIFProtocolError("INVALID_ARTIFACT", "baseline reference mode is not frozen")
    _validate_protocol_bindings(protocol_config, root, data_config_path, eval_config_path, train_config_path, model_paths)

    parameter_profiles = build_parameter_profiles(model_configs)
    parameter_profile_hash = canonical_hash(_parameter_profile_identity(parameter_profiles))
    source = git_snapshot(root)
    protocol_payload = {
        "phase": "A2",
        "stage": protocol_config["stage"],
        "protocol": protocol_config,
        "data_config": data_config,
        "train_config": train_config,
        "eval_config": eval_config,
        "ablation_config": ablation_config,
        "model_configs": model_configs,
        "baseline_model_config": baseline_model_config,
        "parameter_profile_hash": parameter_profile_hash,
        "source": source,
    }
    protocol_hash = canonical_hash(protocol_payload)
    target_registry = _target_registry(model_configs[0])
    sensor_registry = _sensor_registry(model_configs[0])
    manifest: dict[str, Any] = {
        "schema_version": TQIF_PROTOCOL_SCHEMA_VERSION,
        "status": "PASS",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "phase": "A2",
        "stage": protocol_config["stage"],
        "protocol_hash": protocol_hash,
        "parameter_profile_hash": parameter_profile_hash,
        "dataset_manifest_hash": canonical_hash(data_manifest),
        "split_manifest_hash": _split_manifest_hash(data_manifest),
        "model_config_hashes": {
            relative_path(root, path): canonical_hash(config)
            for path, config in zip(model_paths, model_configs, strict=True)
        },
        "baseline_model_config_hash": canonical_hash(baseline_model_config),
        "train_config_hash": canonical_hash(train_config),
        "eval_config_hash": canonical_hash(eval_config),
        "ablation_config_hash": canonical_hash(ablation_config),
        "target_slot_ids": list(target_registry.slot_ids),
        "target_slot_hash": target_slot_registry_hash(target_registry),
        "sensor_registry": [spec.to_dict() for spec in sensor_registry],
        "sensor_registry_hash": sensor_registry_hash(sensor_registry),
        "allowed_read_splits": list(TQIF_ALLOWED_READ_SPLITS),
        "forbidden_training_keys": sorted(TQIF_FORBIDDEN_KEYS - {"sequence_id"}),
        "parameter_profiles": parameter_profiles,
        "source": source,
        "artifact_paths": {
            "protocol_config": relative_path(root, protocol_path),
            "data_config": relative_path(root, data_config_path),
            "data_manifest": relative_path(root, data_manifest_path),
            "train_config": relative_path(root, train_config_path),
            "eval_config": relative_path(root, eval_config_path),
            "ablation_config": relative_path(root, ablation_config_path),
            "baseline_model_config": relative_path(root, baseline_model_path),
        },
    }
    manifest_path = root / "outputs" / "runs" / "a2" / "tqif" / "protocol_manifest.json"
    if manifest_path.is_file():
        existing = read_json_object(manifest_path)
        if existing.get("protocol_hash") == protocol_hash:
            return existing
        raise TQIFProtocolError(
            "RUN_ID_HASH_CONFLICT",
            "protocol_manifest.json already records different frozen inputs",
        )
    write_json(manifest_path, manifest)
    profiles_path = root / "outputs" / "runs" / "a2" / "tqif" / "parameter_profiles.json"
    write_json(profiles_path, parameter_profiles)
    return manifest


def build_parameter_profiles(
    model_configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    full_configs = {
        str(config["recipe"]): config
        for config in model_configs
        if config.get("model_id") == "TQIF"
    }
    matched_configs = {
        str(config["recipe"]): config
        for config in model_configs
        if config.get("model_id") == "TQIF-MATCHED-CONCAT"
    }
    if set(full_configs) != set(matched_configs):
        raise TQIFProtocolError(
            "PARAMETER_PARITY_FAILED",
            "each frozen TQIF recipe must have one matched concat control",
        )

    recipe_profiles: dict[str, Any] = {}
    for recipe in sorted(full_configs):
        full_config = full_configs[recipe]
        matched_config = matched_configs[recipe]
        full_model = build_tqif_model(full_config)
        matched_model = build_tqif_matched_concat_model(matched_config)
        fixed_models = {
            "TQIF-H0": full_model,
            "C0": matched_model,
            "I1": _build_variant(full_config, "I1"),
            "TQIF-STR": _build_variant(full_config, "TQIF-STR"),
        }
        selected_hidden = _search_no_pair_capacity(
            full_config,
            fixed_models,
        )
        models = {
            **fixed_models,
            "C1": _build_variant(
                full_config,
                "C1",
                capacity_control_hidden_dim=selected_hidden["C1"],
            ),
            "Q1": _build_variant(
                full_config,
                "Q1",
                capacity_control_hidden_dim=selected_hidden["Q1"],
            ),
        }
        counts = {
            model_id: trainable_parameter_count(model)
            for model_id, model in models.items()
        }
        comparisons = _comparison_report(counts)
        if not all(entry["within_tolerance"] for entry in comparisons.values()):
            raise TQIFProtocolError(
                "PARAMETER_PARITY_FAILED",
                f"{recipe} has a main comparison outside the 10% tolerance",
            )
        batch = _profile_batch(full_config)
        measured = {
            model_id: _measure_model(model, batch)
            for model_id, model in models.items()
        }
        recipe_profiles[recipe] = {
            "schema_version": TQIF_PARAMETER_PROFILE_SCHEMA_VERSION,
            "recipe_id": recipe,
            "candidate_hidden_dims": list(TQIF_CAPACITY_CANDIDATES),
            "selected_capacity_control_hidden_dim": selected_hidden,
            "models": {
                model_id: {
                    "trainable_parameter_count": counts[model_id],
                    **measured[model_id],
                }
                for model_id in sorted(models)
            },
            "comparisons": comparisons,
        }
    return {
        "schema_version": TQIF_PARAMETER_PROFILE_SCHEMA_VERSION,
        "parameter_match_tolerance": 0.10,
        "recipes": recipe_profiles,
    }


def _parameter_profile_identity(value: Any) -> Any:
    """Remove machine-noise measurements before hashing frozen profile inputs."""

    if isinstance(value, Mapping):
        return {
            key: _parameter_profile_identity(child)
            for key, child in value.items()
            if key not in TQIF_PARAMETER_PROFILE_RUNTIME_FIELDS
        }
    if isinstance(value, list):
        return [_parameter_profile_identity(child) for child in value]
    return value


def run_tqif_smoke(*, project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    protocol = _require_protocol_passed(root)
    full_configs, matched_configs = _load_model_configs_for_protocol(root, protocol)
    run_root = root / "outputs" / "runs" / "a2" / "tqif"
    results: list[dict[str, Any]] = []
    for recipe in sorted(full_configs):
        full_config = full_configs[recipe]
        profile = protocol["parameter_profiles"]["recipes"][recipe]
        selected = profile["selected_capacity_control_hidden_dim"]
        torch.manual_seed(17)
        model_entries = {
            "TQIF-H0": _build_variant(full_config, "TQIF-H0"),
            "C0": build_tqif_matched_concat_model(matched_configs[recipe]),
            "C1": _build_variant(full_config, "C1", capacity_control_hidden_dim=selected["C1"]),
            "Q1": _build_variant(full_config, "Q1", capacity_control_hidden_dim=selected["Q1"]),
            "I1": _build_variant(full_config, "I1"),
            "TQIF-STR": _build_variant(full_config, "TQIF-STR"),
        }
        data_path = root / str(protocol["artifact_paths"]["data_manifest"])
        dataset = load_dataset_splits(
            data_path.parent,
            allowed_splits=("train",),
        )
        smoke_samples = dataset.samples()[:2]
        if not smoke_samples:
            raise TQIFProtocolError("INVALID_ARTIFACT", "train split has no samples for smoke")
        batch = collate_samples(smoke_samples)
        for model_id, model in model_entries.items():
            restore_factory = _smoke_restore_factory(
                full_config,
                matched_configs[recipe],
                model_id,
                selected,
            )
            results.append(
                _run_smoke_model(
                    root,
                    protocol,
                    recipe,
                    model_id,
                    model,
                    batch,
                    full_config,
                    restore_factory,
                )
            )
    return {"stage": "A2-TQIF-1", "status": "PASS", "runs": results}


def run_tqif_select(*, project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    protocol = _require_protocol_passed(root)
    capacity = _require_summary(
        root / "outputs" / "summary" / "a2" / "tqif" / "capacity_summary.json",
        stage="A2-TQIF-3",
        allowed_statuses={"PASS", "NEGATIVE_RESULT"},
    )
    if capacity["status"] == "NEGATIVE_RESULT":
        selection = {
            "schema_version": TQIF_SELECTION_SCHEMA_VERSION,
            "phase": "A2",
            "stage": "A2-TQIF-5",
            "status": "PASS",
            "selection_status": "NEGATIVE_RESULT",
            "selected_model_id": None,
            "selected_recipe_id": None,
            "selected_checkpoint_policy": None,
            "reason": "NO_CAPACITY_RECIPE_PASSED",
            "capacity_summary_hash": canonical_hash(capacity),
            "parameter_profile_hash": protocol["parameter_profile_hash"],
            "protocol_hash": protocol["protocol_hash"],
            "allowed_next_phase": None,
        }
        return _write_selection(root, selection)

    ablation = _require_summary(
        root / "outputs" / "summary" / "a2" / "tqif" / "ablation_summary.json",
        stage="A2-TQIF-4",
        allowed_statuses={"PASS"},
    )
    comparisons = ablation["comparisons"]
    query_pass = any(
        comparisons[key]["status"] == "PASS"
        for key in ("Q1__vs__C1", "TQIF-H0__vs__I1")
    )
    pair_pass = any(
        comparisons[key]["status"] == "PASS"
        for key in ("I1__vs__C1", "TQIF-H0__vs__Q1")
    )
    overall_pass = comparisons["TQIF-H0__vs__C0"]["status"] == "PASS"
    head_pass = comparisons["TQIF-STR__vs__TQIF-H0"]["status"] == "PASS"
    query_only_pass = comparisons["Q1__vs__C0"]["status"] == "PASS"
    pair_only_pass = comparisons["I1__vs__C0"]["status"] == "PASS"
    if query_pass and pair_pass and overall_pass:
        selection_status = "A2_POSITIVE_CANDIDATE"
        selected_model_id = "TQIF-STR" if head_pass else "TQIF-H0"
    elif query_pass and not pair_pass and query_only_pass:
        selection_status = "QUERY_ONLY"
        selected_model_id = None
    elif pair_pass and not query_pass and pair_only_pass:
        selection_status = "PAIR_ONLY"
        selected_model_id = None
    elif head_pass and not query_pass and not pair_pass and not overall_pass:
        selection_status = "HEAD_ONLY"
        selected_model_id = None
    else:
        selection_status = "NEGATIVE_RESULT"
        selected_model_id = None
    selected_recipe = str(capacity["selected_recipe_id"])
    selected_checkpoints = None
    if selected_model_id is not None:
        model_summary = ablation["models"][f"{selected_model_id}::{selected_recipe}"]
        selected_checkpoints = {
            str(record["seed"]): record["checkpoint_path"]
            for record in model_summary["seed_records"]
        }
    selection = {
        "schema_version": TQIF_SELECTION_SCHEMA_VERSION,
        "phase": "A2",
        "stage": "A2-TQIF-5",
        "status": "PASS",
        "selection_status": selection_status,
        "selected_model_id": selected_model_id,
        "selected_recipe_id": selected_recipe if selected_model_id is not None else None,
        "selected_checkpoint_policy": "per_seed_best_val" if selected_model_id else None,
        "selected_checkpoints": selected_checkpoints,
        "mechanism_gates": {
            "query": query_pass,
            "pair": pair_pass,
            "overall": overall_pass,
            "head": head_pass,
        },
        "evidence_comparison_hash": canonical_hash(comparisons),
        "capacity_summary_hash": canonical_hash(capacity),
        "parameter_profile_hash": protocol["parameter_profile_hash"],
        "protocol_hash": protocol["protocol_hash"],
        "allowed_next_phase": "A2H" if selected_model_id is not None else None,
    }
    return _write_selection(root, selection)


def run_tqif_baseline(*, project_root: str | Path | None = None) -> dict[str, Any]:
    """Rebuild the A2M-MLP anchor and matched controls on the current A2 data."""
    root = _project_root(project_root)
    protocol = _require_protocol_passed(root)
    context = _training_context(root, protocol)
    full_configs, matched_configs = _load_model_configs_for_protocol(root, protocol)
    del full_configs
    baseline_config = _load_baseline_model_config(root, protocol)
    baseline_recipe = _find_named_recipe(baseline_config, "mlp_lbfgs_width32")
    records: list[dict[str, Any]] = []
    for seed in TQIF_SEEDS:
        records.append(
            _run_training_model(
                root=root,
                protocol=protocol,
                context=context,
                stage="A2-TQIF-2",
                model_id="A2M-MLP",
                recipe_id="mlp_lbfgs_width32",
                seed=seed,
                model_config_identity={"config": baseline_config, "recipe": baseline_recipe},
                model_factory=lambda recipe=baseline_recipe, config=baseline_config: build_a2m_model(
                    "A2M-MLP",
                    recipe,
                    sensor_ids=config["sensor_ids"],
                    sensor_types=config["sensor_types"],
                    output_dim=int(config["head"]["output_dim"]),
                ),
            )
        )
    for recipe in sorted(matched_configs):
        matched_config = matched_configs[recipe]
        for seed in TQIF_SEEDS:
            records.append(
                _run_training_model(
                    root=root,
                    protocol=protocol,
                    context=context,
                    stage="A2-TQIF-2",
                    model_id="C0",
                    recipe_id=recipe,
                    seed=seed,
                    model_config_identity=matched_config,
                    model_factory=lambda config=matched_config: build_tqif_matched_concat_model(config),
                )
            )
    if len(records) != 15:
        raise TQIFProtocolError("INCOMPLETE_RUN_SET", "A2-TQIF-2 requires exactly 15 runs")
    models = _summarize_records(records, context["eval_config"])
    history_summary_path = root / "outputs" / "summary" / "a2m" / "a2m_formal.json"
    history_identity = None
    if history_summary_path.is_file():
        history = read_json_object(history_summary_path)
        history_identity = history.get("manifest", {}).get("evidence", {}).get("data_content_sha256")
    result: dict[str, Any] = {
        "schema_version": "tqif-baseline-1",
        "stage": "A2-TQIF-2",
        "status": "PASS",
        "protocol_hash": protocol["protocol_hash"],
        "dataset_manifest_hash": protocol["dataset_manifest_hash"],
        "baseline_reference_mode": "rebuild_current_a2_v1",
        "model_runs": [record["run_id"] for record in records],
        "run_count": len(records),
        "expected_run_count": 15,
        "missing_runs": [],
        "abnormal_runs": [],
        "history": {
            "summary_path": relative_path(root, history_summary_path),
            "data_identity": history_identity,
            "current_data_identity": context["data_content_sha256"],
            "used_for_metric_gate": False,
            "use": "recipe_and_model_identity_only",
        },
        "models": models,
        "new_anchor": {
            "model_key": "A2M-MLP::mlp_lbfgs_width32",
            "mean_validation_macro_RNMAE": models[
                "A2M-MLP::mlp_lbfgs_width32"
            ]["mean_validation_macro_RNMAE"],
            "data_content_sha256": context["data_content_sha256"],
        },
        "block_reasons": [],
        "candidate_training_started": False,
    }
    baseline_path = root / "outputs" / "summary" / "a2" / "tqif" / "baseline_summary.json"
    write_json(baseline_path, result)
    return result


def run_tqif_development(*, project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    protocol = _require_protocol_passed(root)
    baseline = _require_summary(
        root / "outputs" / "summary" / "a2" / "tqif" / "baseline_summary.json",
        stage="A2-TQIF-2",
        allowed_statuses={"PASS"},
    )
    context = _training_context(root, protocol)
    full_configs, _ = _load_model_configs_for_protocol(root, protocol)
    records: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for recipe in sorted(full_configs):
        full_config = full_configs[recipe]
        full_records = [
            _run_training_model(
                root=root,
                protocol=protocol,
                context=context,
                stage="A2-TQIF-3",
                model_id="TQIF-H0",
                recipe_id=recipe,
                seed=seed,
                model_config_identity=full_config,
                model_factory=lambda config=full_config: _build_variant(config, "TQIF-H0"),
            )
            for seed in TQIF_SEEDS
        ]
        c0_records = list(baseline["models"][f"C0::{recipe}"]["seed_records"])
        records.extend(full_records)
        records.extend(c0_records)
        recipe_models = _summarize_records(full_records + c0_records, context["eval_config"])
        models.update(recipe_models)
        comparisons[f"TQIF-H0__vs__C0::{recipe}"] = _comparison_gate(
            full_records,
            c0_records,
            context=context,
            eval_config=context["eval_config"],
        )
    selected_recipe = _select_capacity_recipe(comparisons, models)
    status = "PASS" if selected_recipe is not None else "NEGATIVE_RESULT"
    summary = {
        "schema_version": "tqif-capacity-1",
        "stage": "A2-TQIF-3",
        "status": status,
        "protocol_hash": protocol["protocol_hash"],
        "logical_run_count": 20,
        "physical_new_run_count": 10,
        "reused_run_count": 10,
        "models": models,
        "comparisons": comparisons,
        "selected_recipe_id": selected_recipe,
        "stop_reason": None if selected_recipe is not None else "NO_RECIPE_PASSED_CAPACITY_GATE",
    }
    write_json(root / "outputs" / "summary" / "a2" / "tqif" / "capacity_summary.json", summary)
    return summary


def run_tqif_ablation(*, project_root: str | Path | None = None) -> dict[str, Any]:
    root = _project_root(project_root)
    protocol = _require_protocol_passed(root)
    capacity = _require_summary(
        root / "outputs" / "summary" / "a2" / "tqif" / "capacity_summary.json",
        stage="A2-TQIF-3",
        allowed_statuses={"PASS"},
    )
    recipe = str(capacity["selected_recipe_id"])
    context = _training_context(root, protocol)
    full_configs, _ = _load_model_configs_for_protocol(root, protocol)
    full_config = full_configs[recipe]
    selected_capacity = protocol["parameter_profiles"]["recipes"][recipe][
        "selected_capacity_control_hidden_dim"
    ]
    existing_models = capacity["models"]
    records_by_model: dict[str, list[dict[str, Any]]] = {
        "C0": list(existing_models[f"C0::{recipe}"]["seed_records"]),
        "TQIF-H0": list(existing_models[f"TQIF-H0::{recipe}"]["seed_records"]),
    }
    for model_id in ("C1", "Q1", "I1", "TQIF-STR"):
        capacity_hidden = selected_capacity.get(model_id)
        records_by_model[model_id] = [
            _run_training_model(
                root=root,
                protocol=protocol,
                context=context,
                stage="A2-TQIF-4",
                model_id=model_id,
                recipe_id=recipe,
                seed=seed,
                model_config_identity={
                    "base_config": full_config,
                    "variant": model_id,
                    "capacity_control_hidden_dim": capacity_hidden,
                },
                model_factory=lambda config=full_config, variant=model_id, hidden=capacity_hidden: _build_variant(
                    config,
                    variant,
                    capacity_control_hidden_dim=hidden,
                ),
            )
            for seed in TQIF_SEEDS
        ]
    all_records = [record for values in records_by_model.values() for record in values]
    models = _summarize_records(all_records, context["eval_config"])
    comparison_pairs = (*TQIF_COMPARISONS, ("Q1", "C0"), ("I1", "C0"))
    comparisons = {
        f"{candidate}__vs__{baseline}": _comparison_gate(
            records_by_model[candidate],
            records_by_model[baseline],
            context=context,
            eval_config=context["eval_config"],
        )
        for candidate, baseline in comparison_pairs
    }
    summary = {
        "schema_version": "tqif-ablation-1",
        "stage": "A2-TQIF-4",
        "status": "PASS",
        "protocol_hash": protocol["protocol_hash"],
        "selected_recipe_id": recipe,
        "logical_run_count": 30,
        "physical_new_run_count": 20,
        "reused_run_count": 10,
        "models": models,
        "comparisons": comparisons,
    }
    summary_root = root / "outputs" / "summary" / "a2" / "tqif"
    write_json(summary_root / "ablation_comparison.json", {"comparisons": comparisons})
    write_json(summary_root / "ablation_summary.json", summary)
    return summary


def _training_context(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    data_manifest_path = root / str(protocol["artifact_paths"]["data_manifest"])
    dataset = load_dataset_splits(
        data_manifest_path.parent,
        allowed_splits=("train", "val"),
    )
    train_samples, validation_samples, _ = prepare_a2_train_val_samples(dataset.samples())
    train_splits = {sample.metadata.get("split") for sample in train_samples}
    validation_splits = {sample.metadata.get("split") for sample in validation_samples}
    if train_splits != {"train"} or validation_splits != {"val"}:
        raise TQIFProtocolError(
            "PROTOCOL_ACCESS_VIOLATION",
            f"training split audit failed: train={train_splits}, val={validation_splits}",
        )
    train_config = read_json_object(root / str(protocol["artifact_paths"]["train_config"]))
    eval_config = read_json_object(root / str(protocol["artifact_paths"]["eval_config"]))
    if canonical_hash(train_config) != protocol["train_config_hash"]:
        raise TQIFProtocolError("RUN_ID_HASH_CONFLICT", "train config changed after protocol")
    if canonical_hash(eval_config) != protocol["eval_config_hash"]:
        raise TQIFProtocolError("RUN_ID_HASH_CONFLICT", "eval config changed after protocol")
    validation_batch = collate_samples(tuple(validation_samples))
    return {
        "root": root,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "validation_batch": validation_batch,
        "targets": validation_batch.target.detach().cpu().numpy().astype(np.float64),
        "groups": tuple(validation_batch.group_id),
        "train_config": train_config,
        "training_config": TorchTrainingConfig.from_mapping(train_config),
        "eval_config": eval_config,
        "data_content_sha256": dataset.manifest["content_sha256"],
    }


def _load_baseline_model_config(
    root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / str(protocol["artifact_paths"]["baseline_model_config"])
    config = read_json_object(path)
    if canonical_hash(config) != protocol["baseline_model_config_hash"]:
        raise TQIFProtocolError(
            "RUN_ID_HASH_CONFLICT",
            "baseline model config changed after protocol",
        )
    validate_a2m_model_config(config)
    return config


def _find_named_recipe(config: Mapping[str, Any], recipe_name: str) -> Mapping[str, Any]:
    recipes = [
        recipe
        for recipe in config.get("recipes", [])
        if isinstance(recipe, Mapping) and recipe.get("name") == recipe_name
    ]
    if len(recipes) != 1:
        raise TQIFProtocolError(
            "INVALID_ARTIFACT",
            f"recipe {recipe_name!r} must resolve exactly once",
        )
    return recipes[0]


def _run_training_model(
    *,
    root: Path,
    protocol: Mapping[str, Any],
    context: Mapping[str, Any],
    stage: str,
    model_id: str,
    recipe_id: str,
    seed: int,
    model_config_identity: Mapping[str, Any],
    model_factory: Callable[[], torch.nn.Module],
) -> dict[str, Any]:
    source = dict(protocol["source"])
    model_config_hash = canonical_hash(model_config_identity)
    input_hash = canonical_run_input_hash(
        protocol_hash=str(protocol["protocol_hash"]),
        dataset_manifest_hash=str(protocol["dataset_manifest_hash"]),
        split_manifest_hash=str(protocol["split_manifest_hash"]),
        model_config_hash=model_config_hash,
        train_config_hash=str(protocol["train_config_hash"]),
        eval_config_hash=str(protocol["eval_config_hash"]),
        seed=seed,
        model_id=model_id,
        recipe_id=recipe_id,
        source_snapshot=source,
    )
    run_id = f"{stage}__{recipe_id}__{model_id}__seed{seed}"
    run_dir = root / "outputs" / "runs" / "a2" / "tqif" / run_id
    manifest_path = run_dir / "run_manifest.json"
    metrics_path = run_dir / "metrics.json"
    prediction_path = run_dir / "predictions.csv"
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if manifest_path.is_file():
        existing = read_json_object(manifest_path)
        if existing.get("status") == "PASS" and existing.get("input_hash") == input_hash:
            for required in (metrics_path, prediction_path, checkpoint_path):
                if not required.is_file():
                    raise TQIFProtocolError(
                        "INCOMPLETE_RUN_SET",
                        f"completed run {run_id} lacks {required.name}",
                    )
            return read_json_object(metrics_path)
        raise TQIFProtocolError(
            "RUN_ID_HASH_CONFLICT",
            f"run {run_id} already exists with different or incomplete inputs",
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(run_dir / ".lock"):
        started_at = _utc_now()
        write_json(
            run_dir / "resolved_config.json",
            {
                "model": model_config_identity,
                "train": context["train_config"],
                "stage": stage,
                "model_id": model_id,
                "recipe_id": recipe_id,
                "seed": seed,
            },
        )
        manifest = {
            "schema_version": TQIF_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "phase": "A2",
            "stage": stage,
            "status": "IN_PROGRESS",
            "model_id": model_id,
            "recipe_id": recipe_id,
            "seed": seed,
            "dataset_manifest_hash": protocol["dataset_manifest_hash"],
            "split_manifest_hash": protocol["split_manifest_hash"],
            "model_config_hash": model_config_hash,
            "train_config_hash": protocol["train_config_hash"],
            "eval_config_hash": protocol["eval_config_hash"],
            "protocol_hash": protocol["protocol_hash"],
            "target_slot_ids": protocol["target_slot_ids"],
            "target_slot_hash": protocol["target_slot_hash"],
            "sensor_registry_hash": protocol["sensor_registry_hash"],
            "input_hash": input_hash,
            "split_audit": {"train": ["train"], "selection": ["val"]},
            "started_at": started_at,
            "finished_at": None,
            **source,
            "artifact_paths": {},
        }
        write_json(manifest_path, manifest)
        (run_dir / "train.log").write_text(
            f"stage={stage} model={model_id} recipe={recipe_id} seed={seed} status=IN_PROGRESS\n",
            encoding="utf-8",
            newline="\n",
        )
        model = _build_seeded_model(model_factory, seed)
        tracemalloc.start()
        training_started = time.perf_counter()
        try:
            fit = train_torch_model(
                model,
                context["train_samples"],
                context["validation_samples"],
                config=context["training_config"],
                seed=seed,
                checkpoint_path=checkpoint_path,
            )
        finally:
            _, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        training_time = time.perf_counter() - training_started
        if not checkpoint_path.is_file() or fit.best_epoch <= 0:
            raise TQIFProtocolError(
                "INCOMPLETE_RUN_SET",
                f"run {run_id} did not produce a selected checkpoint",
            )
        restored = _build_seeded_model(model_factory, seed)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
            raise TQIFProtocolError("INVALID_ARTIFACT", f"invalid checkpoint payload for {run_id}")
        checkpoint_contract = getattr(restored, "checkpoint_contract", None)
        if checkpoint_contract is not None:
            if not callable(checkpoint_contract) or payload.get("model_contract") != checkpoint_contract():
                raise TQIFProtocolError(
                    "CHECKPOINT_CONTRACT_MISMATCH",
                    f"checkpoint contract changed for {run_id}",
                )
        restored.load_state_dict(payload["state_dict"])
        restored.eval()
        inference_started = time.perf_counter()
        with torch.no_grad():
            prediction = (
                restored(context["validation_batch"])
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        inference_time = time.perf_counter() - inference_started
        if not np.isfinite(prediction).all():
            raise TQIFProtocolError("NON_FINITE_METRIC", f"non-finite prediction for {run_id}")
        validation = evaluate_predictions(
            context["targets"],
            prediction,
            context["groups"],
            np.arange(len(context["validation_samples"]), dtype=np.int64),
        )
        resources = {
            "parameter_count": trainable_parameter_count(restored),
            "training_time_s": float(training_time),
            "inference_time_s_per_sample": float(
                inference_time / max(len(context["validation_samples"]), 1)
            ),
            "peak_memory_bytes": int(peak_memory),
            "best_epoch": int(fit.best_epoch),
            "epochs_completed": int(fit.epochs_completed),
            "checkpoint_sha256": binary_sha256(checkpoint_path),
        }
        _write_prediction_csv(
            prediction_path,
            run_id=run_id,
            model_id=model_id,
            recipe_id=recipe_id,
            seed=seed,
            batch=context["validation_batch"],
            prediction=prediction,
            split="val",
        )
        record = {
            "schema_version": TQIF_METRICS_SCHEMA_VERSION,
            "run_id": run_id,
            "stage": stage,
            "status": "PASS",
            "model_id": model_id,
            "recipe_id": recipe_id,
            "seed": int(seed),
            "validation": validation,
            "resources": resources,
            "checkpoint_path": relative_path(root, checkpoint_path),
            "prediction_path": relative_path(root, prediction_path),
        }
        write_json(metrics_path, record)
        manifest.update(
            {
                "status": "PASS",
                "finished_at": _utc_now(),
                "artifact_paths": {
                    "checkpoint": relative_path(root, checkpoint_path),
                    "metrics": relative_path(root, metrics_path),
                    "predictions": relative_path(root, prediction_path),
                },
            }
        )
        write_json(manifest_path, manifest)
        with (run_dir / "train.log").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                f"status=PASS best_epoch={fit.best_epoch} val_macro_RNMAE={validation['macro_RNMAE']:.12g}\n"
            )
        return record


def _build_seeded_model(
    model_factory: Callable[[], torch.nn.Module],
    seed: int,
) -> torch.nn.Module:
    """Construct a model from a run-local seed, independent of prior stage RNG use."""
    torch.manual_seed(seed)
    return model_factory()


def _summarize_records(
    records: Sequence[Mapping[str, Any]],
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        key = f"{record['model_id']}::{record['recipe_id']}"
        grouped.setdefault(key, []).append(record)
    summaries: dict[str, Any] = {}
    bootstrap = eval_config["bootstrap"]
    for key, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda record: int(record["seed"]))
        seeds = [int(record["seed"]) for record in ordered]
        if seeds != list(TQIF_SEEDS):
            raise TQIFProtocolError(
                "INCOMPLETE_RUN_SET",
                f"{key} has seeds {seeds}, expected {list(TQIF_SEEDS)}",
            )
        metrics = [record["validation"] for record in ordered]
        macro = np.asarray([metric["macro_RNMAE"] for metric in metrics], dtype=np.float64)
        component = np.asarray([metric["component_RNMAE"] for metric in metrics], dtype=np.float64)
        component_mae = np.asarray([metric["component_MAE"] for metric in metrics], dtype=np.float64)
        component_rmse = np.asarray([metric["component_RMSE"] for metric in metrics], dtype=np.float64)
        rng = np.random.default_rng(int(bootstrap["seed"]))
        draws = rng.integers(0, len(macro), size=(int(bootstrap["repeats"]), len(macro)))
        boot_means = macro[draws].mean(axis=1)
        summaries[key] = {
            "model_id": ordered[0]["model_id"],
            "recipe_id": ordered[0]["recipe_id"],
            "seed_records": ordered,
            "mean_validation_macro_RNMAE": float(macro.mean()),
            "std_validation_macro_RNMAE": float(macro.std(ddof=0)),
            "median_validation_macro_RNMAE": float(np.median(macro)),
            "worst_seed_validation_macro_RNMAE": float(macro.max()),
            "bootstrap_95_ci": [
                float(np.percentile(boot_means, 2.5)),
                float(np.percentile(boot_means, 97.5)),
            ],
            "mean_component_RNMAE": [float(value) for value in component.mean(axis=0)],
            "mean_component_MAE": [float(value) for value in component_mae.mean(axis=0)],
            "mean_component_RMSE": [float(value) for value in component_rmse.mean(axis=0)],
            "parameter_count": int(ordered[0]["resources"]["parameter_count"]),
            "mean_training_time_s": float(
                np.mean([record["resources"]["training_time_s"] for record in ordered])
            ),
        }
    return summaries


def _comparison_gate(
    candidate_records: Sequence[Mapping[str, Any]],
    baseline_records: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any],
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = {int(record["seed"]): record for record in candidate_records}
    baseline = {int(record["seed"]): record for record in baseline_records}
    if set(candidate) != set(TQIF_SEEDS) or set(baseline) != set(TQIF_SEEDS):
        raise TQIFProtocolError("INCOMPLETE_RUN_SET", "comparison lacks one or more frozen seeds")
    candidate_values = np.asarray(
        [candidate[seed]["validation"]["macro_RNMAE"] for seed in TQIF_SEEDS],
        dtype=np.float64,
    )
    baseline_values = np.asarray(
        [baseline[seed]["validation"]["macro_RNMAE"] for seed in TQIF_SEEDS],
        dtype=np.float64,
    )
    candidate_component = np.asarray(
        [candidate[seed]["validation"]["component_RNMAE"] for seed in TQIF_SEEDS],
        dtype=np.float64,
    ).mean(axis=0)
    baseline_component = np.asarray(
        [baseline[seed]["validation"]["component_RNMAE"] for seed in TQIF_SEEDS],
        dtype=np.float64,
    ).mean(axis=0)
    candidate_predictions = np.mean(
        [_read_prediction_matrix(Path(context["root"]) / candidate[seed]["prediction_path"], context["groups"]) for seed in TQIF_SEEDS],
        axis=0,
    )
    baseline_predictions = np.mean(
        [_read_prediction_matrix(Path(context["root"]) / baseline[seed]["prediction_path"], context["groups"]) for seed in TQIF_SEEDS],
        axis=0,
    )
    bootstrap_config = eval_config["bootstrap"]
    paired = group_bootstrap_comparison(
        candidate_predictions,
        baseline_predictions,
        context["targets"],
        context["groups"],
        seed=int(bootstrap_config["seed"]),
        samples=int(bootstrap_config["repeats"]),
    )
    relative_improvement = float(
        (baseline_values.mean() - candidate_values.mean()) / baseline_values.mean()
    )
    component_delta = candidate_component - baseline_component
    direction_count = int(np.sum(candidate_values < baseline_values))
    candidate_parameters = int(candidate_records[0]["resources"]["parameter_count"])
    baseline_parameters = int(baseline_records[0]["resources"]["parameter_count"])
    parameter_delta = abs(candidate_parameters - baseline_parameters) / max(
        candidate_parameters,
        baseline_parameters,
    )
    promotion = eval_config["promotion"]
    checks = {
        "relative_improvement": relative_improvement
        >= float(promotion["min_macro_relative_gain"]),
        "seed_direction": direction_count >= int(promotion["min_seed_same_direction"]),
        "paired_ci_upper_below_zero": paired["percentile_97_5"] < 0.0,
        "component_degradation": float(component_delta.max())
        <= float(promotion["max_component_absolute_rnmae_degradation"]),
        "parameter_match": parameter_delta <= 0.10,
    }
    seed_differences = candidate_values - baseline_values
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate_model_id": candidate_records[0]["model_id"],
        "baseline_model_id": baseline_records[0]["model_id"],
        "relative_improvement": relative_improvement,
        "same_direction_seed_count": direction_count,
        "seed_macro_RNMAE_differences": [float(value) for value in seed_differences],
        "median_seed_improvement": float(np.median(baseline_values - candidate_values)),
        "component_RNMAE_delta": [float(value) for value in component_delta],
        "parameter_relative_difference": float(parameter_delta),
        "paired_group_bootstrap": paired,
        "checks": checks,
    }


def _read_prediction_matrix(path: Path, expected_groups: Sequence[str]) -> np.ndarray:
    if not path.is_file():
        raise TQIFProtocolError("INCOMPLETE_RUN_SET", f"prediction file is missing: {path}")
    rows: list[list[float]] = []
    groups: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "val":
                raise TQIFProtocolError(
                    "PROTOCOL_ACCESS_VIOLATION",
                    f"non-val prediction entered A2 comparison: {path}",
                )
            groups.append(str(row["mixture_id"]))
            rows.append(
                [
                    float(row["pred_x_Ar_pct"]),
                    float(row["pred_x_He_pct"]),
                    float(row["pred_x_CO2_pct"]),
                ]
            )
    if tuple(groups) != tuple(expected_groups):
        raise TQIFProtocolError("INVALID_ARTIFACT", f"prediction group order changed: {path}")
    values = np.asarray(rows, dtype=np.float64)
    if values.shape != (len(expected_groups), 3) or not np.isfinite(values).all():
        raise TQIFProtocolError("INVALID_ARTIFACT", f"prediction values are invalid: {path}")
    return values


def _select_capacity_recipe(
    comparisons: Mapping[str, Mapping[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
) -> str | None:
    passing: list[str] = []
    for key, comparison in comparisons.items():
        if comparison["status"] == "PASS":
            passing.append(key.split("::", 1)[1])
    if not passing:
        return None
    return max(
        passing,
        key=lambda recipe: (
            float(comparisons[f"TQIF-H0__vs__C0::{recipe}"]["relative_improvement"]),
            float(comparisons[f"TQIF-H0__vs__C0::{recipe}"]["median_seed_improvement"]),
            -float(models[f"TQIF-H0::{recipe}"]["worst_seed_validation_macro_RNMAE"]),
            -int(models[f"TQIF-H0::{recipe}"]["parameter_count"]),
            recipe == "tqif_token16_pair16",
        ),
    )


def _require_summary(
    path: Path,
    *,
    stage: str,
    allowed_statuses: set[str],
) -> dict[str, Any]:
    if not path.is_file():
        raise TQIFProtocolError("PREREQUISITE_NOT_PASSED", f"missing prerequisite: {path}")
    summary = read_json_object(path)
    if summary.get("stage") != stage or summary.get("status") not in allowed_statuses:
        raise TQIFProtocolError(
            "PREREQUISITE_NOT_PASSED",
            f"{stage} prerequisite has status {summary.get('status')!r}",
        )
    return summary


def _write_selection(root: Path, selection: Mapping[str, Any]) -> dict[str, Any]:
    run_path = root / "outputs" / "runs" / "a2" / "tqif" / "selection_manifest.json"
    summary_path = root / "outputs" / "summary" / "a2" / "tqif" / "selection_manifest.json"
    if run_path.is_file():
        existing = read_json_object(run_path)
        if canonical_hash(existing) != canonical_hash(selection):
            raise TQIFProtocolError("RUN_ID_HASH_CONFLICT", "selection manifest changed")
        return existing
    write_json(run_path, selection)
    write_json(summary_path, selection)
    return dict(selection)


def _run_smoke_model(
    root: Path,
    protocol: Mapping[str, Any],
    recipe: str,
    model_id: str,
    model: torch.nn.Module,
    batch: UnifiedBatch,
    full_config: Mapping[str, Any],
    restore_factory: Callable[[], torch.nn.Module],
) -> dict[str, Any]:
    seed = 17
    source = git_snapshot(root)
    model_config_hash = canonical_hash(
        {
            "base_config": full_config,
            "variant": model_id,
            "capacity_control_hidden_dim": getattr(
                getattr(model, "fusion", None),
                "capacity_control",
                None,
            ).hidden_dim
            if getattr(getattr(model, "fusion", None), "capacity_control", None) is not None
            else None,
        }
    )
    input_hash = canonical_run_input_hash(
        protocol_hash=str(protocol["protocol_hash"]),
        dataset_manifest_hash=str(protocol["dataset_manifest_hash"]),
        split_manifest_hash=str(protocol["split_manifest_hash"]),
        model_config_hash=model_config_hash,
        train_config_hash=str(protocol["train_config_hash"]),
        eval_config_hash=str(protocol["eval_config_hash"]),
        seed=seed,
        model_id=model_id,
        recipe_id=recipe,
        source_snapshot=source,
    )
    run_id = f"A2-TQIF-1__{recipe}__{model_id}__smoke"
    run_dir = root / "outputs" / "runs" / "a2" / "tqif" / run_id
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.is_file():
        existing = read_json_object(manifest_path)
        if existing.get("status") == "PASS" and existing.get("input_hash") == input_hash:
            return existing
        raise TQIFProtocolError(
            "RUN_ID_HASH_CONFLICT",
            f"completed smoke run {run_id} has different inputs",
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(run_dir / ".lock"):
        started_at = _utc_now()
        write_json(
            run_dir / "resolved_config.json",
            {
                "base_config": full_config,
                "variant_id": model_id,
                "recipe_id": recipe,
            },
        )
        (run_dir / "train.log").write_text(
            f"stage=A2-TQIF-1 model={model_id} recipe={recipe} seed={seed}\n",
            encoding="utf-8",
            newline="\n",
        )
        torch.manual_seed(seed)
        model.train()
        prediction = model(batch)
        loss = prediction.square().mean()
        if not torch.isfinite(loss):
            raise TQIFProtocolError("NON_FINITE_METRIC", f"non-finite smoke loss for {run_id}")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not gradients or any(gradient is None for gradient in gradients):
            raise TQIFProtocolError("INVALID_ARTIFACT", f"missing smoke gradient for {run_id}")
        if any(not torch.isfinite(gradient).all() for gradient in gradients if gradient is not None):
            raise TQIFProtocolError("NON_FINITE_METRIC", f"non-finite smoke gradient for {run_id}")
        model.eval()
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "schema_version": "tqif-checkpoint-1",
            "seed": seed,
            "state_dict": model.state_dict(),
        }
        checkpoint_contract = getattr(model, "checkpoint_contract", None)
        if checkpoint_contract is not None:
            checkpoint_payload["model_contract"] = checkpoint_contract()
        torch.save(checkpoint_payload, checkpoint_path)
        if checkpoint_contract is not None:
            restored = restore_factory()
            load_tqif_checkpoint(restored, str(checkpoint_path))
        metrics = {
            "schema_version": TQIF_METRICS_SCHEMA_VERSION,
            "run_id": run_id,
            "status": "PASS",
            "smoke_loss": float(loss.detach()),
            "prediction_shape": list(prediction.shape),
            "gradient_count": len(gradients),
        }
        write_json(run_dir / "metrics.json", metrics)
        _write_prediction_csv(
            run_dir / "predictions.csv",
            run_id=run_id,
            model_id=model_id,
            recipe_id=recipe,
            seed=seed,
            batch=batch,
            prediction=prediction.detach().cpu().numpy(),
        )
        run_manifest = {
            "schema_version": TQIF_RUN_SCHEMA_VERSION,
            "run_id": run_id,
            "phase": "A2",
            "stage": "A2-TQIF-1",
            "status": "PASS",
            "model_id": model_id,
            "recipe_id": recipe,
            "seed": seed,
            "dataset_manifest_hash": protocol["dataset_manifest_hash"],
            "split_manifest_hash": protocol["split_manifest_hash"],
            "model_config_hash": model_config_hash,
            "train_config_hash": protocol["train_config_hash"],
            "eval_config_hash": protocol["eval_config_hash"],
            "protocol_hash": protocol["protocol_hash"],
            "target_slot_ids": protocol["target_slot_ids"],
            "target_slot_hash": protocol["target_slot_hash"],
            "sensor_registry_hash": protocol["sensor_registry_hash"],
            "input_hash": input_hash,
            "started_at": started_at,
            "finished_at": _utc_now(),
            **source,
            "artifact_paths": {
                "checkpoint": relative_path(root, checkpoint_path),
                "metrics": relative_path(root, run_dir / "metrics.json"),
            },
        }
        write_json(manifest_path, run_manifest)
    return run_manifest


def _smoke_restore_factory(
    full_config: Mapping[str, Any],
    matched_config: Mapping[str, Any],
    model_id: str,
    selected_capacity: Mapping[str, int],
) -> Callable[[], torch.nn.Module]:
    if model_id == "C0":
        return lambda: build_tqif_matched_concat_model(matched_config)
    capacity = (
        selected_capacity[model_id]
        if model_id in {"C1", "Q1"}
        else None
    )
    return lambda: _build_variant(
        full_config,
        model_id,
        capacity_control_hidden_dim=capacity,
    )


def build_tqif_variant(
    base_config: Mapping[str, Any],
    variant_id: str,
    *,
    capacity_control_hidden_dim: int | None = None,
) -> torch.nn.Module:
    return _build_variant(
        base_config,
        variant_id,
        capacity_control_hidden_dim=capacity_control_hidden_dim,
    )


def _build_variant(
    base_config: Mapping[str, Any],
    variant_id: str,
    *,
    capacity_control_hidden_dim: int | None = None,
) -> torch.nn.Module:
    if variant_id == "TQIF-H0":
        return build_tqif_model(base_config)
    if variant_id == "C0":
        raise ValueError("C0 must be built from its matched concat config")
    head_id = "STR" if variant_id == "TQIF-STR" else "H0"
    query_mode = "shared" if variant_id in {"C1", "I1"} else "independent"
    use_pair = variant_id in {"I1", "TQIF-STR"}
    return TQIFModel(
        embedding_dim=int(base_config["embedding_dim"]),
        token_dim=int(base_config["token_dim"]),
        pair_hidden_dim=int(base_config["pair_hidden_dim"]),
        query_ffn_dim=int(base_config["query_ffn_dim"]),
        attention_heads=int(base_config["attention_heads"]),
        output_dim=int(base_config["head"]["output_dim"]),
        sensor_ids=base_config["sensor_ids"],
        sensor_types=base_config["sensor_types"],
        target_slot_registry=TQIFTargetSlotRegistry.from_mappings(
            base_config["target_slot_registry"]
        ),
        head_id=head_id,
        query_mode=query_mode,
        use_pair=use_pair,
        use_quality=bool(base_config.get("uses_quality", False)),
        total=float(base_config["head"].get("total", 100.0)),
        temperature=float(base_config["head"].get("temperature", 1.0)),
        capacity_control_hidden_dim=capacity_control_hidden_dim,
        dropout=float(base_config["dropout"]),
    )


def _search_no_pair_capacity(
    base_config: Mapping[str, Any],
    fixed_models: Mapping[str, torch.nn.Module],
) -> dict[str, int]:
    fixed_counts = {
        model_id: trainable_parameter_count(model)
        for model_id, model in fixed_models.items()
    }
    c1_counts: dict[int, int] = {}
    q1_counts: dict[int, int] = {}
    for hidden_dim in TQIF_CAPACITY_CANDIDATES:
        c1_counts[hidden_dim] = trainable_parameter_count(
            _build_variant(
                base_config,
                "C1",
                capacity_control_hidden_dim=hidden_dim,
            )
        )
        q1_counts[hidden_dim] = trainable_parameter_count(
            _build_variant(
                base_config,
                "Q1",
                capacity_control_hidden_dim=hidden_dim,
            )
        )
    best: tuple[float, int, int] | None = None
    for c1_hidden, c1_count in c1_counts.items():
        for q1_hidden, q1_count in q1_counts.items():
            counts = {
                **fixed_counts,
                "C1": c1_count,
                "Q1": q1_count,
            }
            comparisons = _comparison_report(counts)
            max_delta = max(item["relative_difference"] for item in comparisons.values())
            score = (max_delta, c1_hidden, q1_hidden)
            if best is None or score < best:
                best = score
    if best is None or best[0] > 0.10:
        raise TQIFProtocolError(
            "PARAMETER_PARITY_FAILED",
            f"no no-pair capacity profile satisfies the 10% tolerance: {best}",
        )
    return {"C1": best[1], "Q1": best[2]}


def _comparison_report(counts: Mapping[str, int]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for left, right in TQIF_COMPARISONS:
        left_count = int(counts[left])
        right_count = int(counts[right])
        relative = abs(left_count - right_count) / max(left_count, right_count)
        result[f"{left}__vs__{right}"] = {
            "left": left,
            "right": right,
            "left_parameter_count": left_count,
            "right_parameter_count": right_count,
            "relative_difference": float(relative),
            "within_tolerance": bool(relative <= 0.10),
        }
    return result


def _measure_model(model: torch.nn.Module, batch: UnifiedBatch) -> dict[str, int | float]:
    model.eval()
    tracemalloc.start()
    with torch.no_grad():
        model(batch)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    with torch.no_grad():
        model(batch)
        start = time.perf_counter()
        for _ in range(3):
            model(batch)
        elapsed = time.perf_counter() - start
    return {
        "peak_memory_bytes": int(peak),
        "single_batch_forward_ms": float(elapsed * 1000.0 / 3.0),
    }


def _write_prediction_csv(
    path: Path,
    *,
    run_id: str,
    model_id: str,
    recipe_id: str,
    seed: int,
    batch: UnifiedBatch,
    prediction: np.ndarray,
    split: str = "smoke",
) -> None:
    targets = batch.target.detach().cpu().numpy()
    if prediction.shape != targets.shape or prediction.shape[1] != 3:
        raise TQIFProtocolError("INVALID_ARTIFACT", "smoke prediction shape does not match target slots")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(TQIF_PREDICTION_FIELDS), lineterminator="\n")
        writer.writeheader()
        for index, (group_id, truth, predicted) in enumerate(
            zip(batch.group_id, targets, prediction, strict=True)
        ):
            writer.writerow(
                {
                    "run_id": run_id,
                    "phase": "A2",
                    "split": split,
                    "seed": seed,
                    "sample_id": group_id if split != "smoke" else f"smoke-{index:04d}",
                    "mixture_id": group_id,
                    "y_true_x_Ar_pct": float(truth[0]),
                    "y_true_x_He_pct": float(truth[1]),
                    "y_true_x_CO2_pct": float(truth[2]),
                    "pred_x_Ar_pct": float(predicted[0]),
                    "pred_x_He_pct": float(predicted[1]),
                    "pred_x_CO2_pct": float(predicted[2]),
                    "model_id": model_id,
                    "recipe_id": recipe_id,
                }
            )


def _profile_batch(config: Mapping[str, Any]) -> UnifiedBatch:
    sensor_ids = tuple(str(value) for value in config["sensor_ids"])
    sensor_types = tuple(str(value) for value in config["sensor_types"])
    sensor_count = len(sensor_ids)
    return UnifiedBatch(
        signals=torch.ones((1, sensor_count, 1, 1), dtype=torch.float32),
        valid_mask=torch.ones((1, sensor_count, 1, 1), dtype=torch.bool),
        quality=torch.ones((1, sensor_count, 1), dtype=torch.float32),
        time=torch.zeros((1, sensor_count, 1), dtype=torch.float64),
        delta_time=torch.zeros((1, sensor_count, 1), dtype=torch.float32),
        sensor_mask=torch.ones((1, sensor_count), dtype=torch.bool),
        feature_mask=torch.ones((1, sensor_count, 1), dtype=torch.bool),
        target=torch.zeros((1, 3), dtype=torch.float32),
        target_mask=torch.ones((1, 3), dtype=torch.bool),
        sensor_id=(sensor_ids,),
        sensor_type=(sensor_types,),
        group_id=("profile-group",),
        dataset_id=("profile-dataset",),
        metadata=({},),
    )


def _validate_protocol_definition(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "gf-a2-tqif-experiment-1":
        raise TQIFProtocolError("INVALID_ARTIFACT", "unsupported TQIF experiment schema")
    if config.get("stage") != "A2-TQIF-0":
        raise TQIFProtocolError("GATE_NOT_PASSED", "protocol stage must be A2-TQIF-0")
    if config.get("kind") != "protocol":
        raise TQIFProtocolError("INVALID_ARTIFACT", "TQIF protocol kind must be protocol")
    if tuple(config.get("allowed_read_splits", ())) != TQIF_ALLOWED_READ_SPLITS:
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "read split allowlist is not frozen")
    if config.get("formal_holdout_access") != "locked":
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "formal holdout must remain locked")
    test_access = config.get("test_access")
    if not isinstance(test_access, Mapping) or test_access.get("mode") != "locked":
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "test access must remain locked")
    if test_access.get("old_test_access") != "forbidden":
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "old test access must be forbidden")
    if tuple(config.get("seeds", ())) != TQIF_SEEDS:
        raise TQIFProtocolError("INVALID_ARTIFACT", "protocol seeds are not frozen")
    for key in (
        "data_config",
        "data_manifest",
        "eval_config",
        "train_config",
        "ablation_config",
        "baseline_model_config",
        "output_dir",
    ):
        _reject_forbidden_path(config.get(key), key)
    if config.get("baseline_recipe") != "mlp_lbfgs_width32":
        raise TQIFProtocolError("INVALID_ARTIFACT", "baseline_recipe is not frozen")
    if config.get("baseline_reference_mode") != "rebuild_current_a2_v1":
        raise TQIFProtocolError("INVALID_ARTIFACT", "baseline_reference_mode is not frozen")
    model_configs = config.get("model_configs")
    if not isinstance(model_configs, list) or not model_configs:
        raise TQIFProtocolError("INVALID_ARTIFACT", "model_configs must be a non-empty list")
    for index, value in enumerate(model_configs):
        _reject_forbidden_path(value, f"model_configs[{index}]")


def _validate_protocol_bindings(
    protocol: Mapping[str, Any],
    root: Path,
    data_config_path: Path,
    eval_config_path: Path,
    train_config_path: Path,
    model_paths: Sequence[Path],
) -> None:
    expected = {
        "data_config": data_config_path,
        "eval_config": eval_config_path,
        "train_config": train_config_path,
    }
    for key, path in expected.items():
        configured = root / str(protocol[key])
        if configured.resolve() != path.resolve():
            raise TQIFProtocolError("RUN_ID_HASH_CONFLICT", f"protocol binding changed for {key}")
    if len(model_paths) != len(protocol["model_configs"]):
        raise TQIFProtocolError("RUN_ID_HASH_CONFLICT", "protocol model binding count changed")


def _validate_model_bundle(configs: Sequence[Mapping[str, Any]]) -> None:
    for config in configs:
        if config.get("model_id") == "TQIF":
            from gf.dl.tqif import validate_tqif_model_config

            validate_tqif_model_config(config)
        elif config.get("model_id") == "TQIF-MATCHED-CONCAT":
            from gf.dl.tqif import validate_tqif_model_config

            validate_tqif_model_config(config)
        else:
            raise TQIFProtocolError("INVALID_ARTIFACT", "protocol contains an unsupported TQIF model")
    model_ids = [str(config["model_id"]) for config in configs]
    if model_ids.count("TQIF") != model_ids.count("TQIF-MATCHED-CONCAT"):
        raise TQIFProtocolError("PARAMETER_PARITY_FAILED", "TQIF and matched controls are unbalanced")
    first = configs[0]
    first_sensor_ids = tuple(first["sensor_ids"])
    first_sensor_types = tuple(first["sensor_types"])
    first_registry = _target_registry(first) if first.get("model_id") == "TQIF" else None
    for config in configs:
        if tuple(config["sensor_ids"]) != first_sensor_ids or tuple(config["sensor_types"]) != first_sensor_types:
            raise TQIFProtocolError("INVALID_ARTIFACT", "model sensor registries are inconsistent")
        if first_registry is not None and config.get("model_id") == "TQIF":
            if _target_registry(config).slot_ids != first_registry.slot_ids:
                raise TQIFProtocolError("CHECKPOINT_CONTRACT_MISMATCH", "model target slot registries differ")


def _validate_tqif_train_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "gf-a2-tqif-train-1":
        raise TQIFProtocolError("INVALID_ARTIFACT", "unsupported TQIF train schema")
    if tuple(config.get("seeds", ())) != TQIF_SEEDS:
        raise TQIFProtocolError("INVALID_ARTIFACT", "TQIF seeds are not frozen")
    if config.get("formal_holdout_access") != "forbidden":
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "training holdout access must be forbidden")
    early = config.get("early_stopping")
    if not isinstance(early, Mapping) or early.get("selection_split") != "a2.val":
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "training selection must use a2.val")
    if config.get("parameter_match_tolerance") != 0.1:
        raise TQIFProtocolError("PARAMETER_PARITY_FAILED", "train tolerance must be 0.10")
    data_access = config.get("data_access")
    if not isinstance(data_access, Mapping):
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "train data_access is missing")
    if tuple(data_access.get("allowed_splits", ())) != TQIF_ALLOWED_READ_SPLITS:
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "train split allowlist changed")
    if tuple(data_access.get("forbidden_keys", ())) != tuple(
        sorted(TQIF_FORBIDDEN_KEYS)
    ):
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "train forbidden key list changed")
    if data_access.get("test_access") != "forbidden":
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "train test access must be forbidden")


def _validate_tqif_eval_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "gf-a2-tqif-eval-1":
        raise TQIFProtocolError("INVALID_ARTIFACT", "unsupported TQIF eval schema")
    if tuple(config.get("allowed_read_splits", ())) != TQIF_ALLOWED_READ_SPLITS:
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "evaluation split allowlist changed")
    if config.get("formal_holdout_access") != "locked":
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "evaluation holdout must remain locked")
    test_access = config.get("test_access")
    if not isinstance(test_access, Mapping) or test_access.get("default") != "locked":
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "evaluation test access must remain locked")
    bootstrap = config.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or bootstrap.get("repeats") != 2000 or bootstrap.get("seed") != 20260830:
        raise TQIFProtocolError("INVALID_ARTIFACT", "bootstrap repeats and seed are not frozen")


def _validate_tqif_ablation_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "gf-a2-tqif-experiment-1":
        raise TQIFProtocolError("INVALID_ARTIFACT", "unsupported TQIF ablation schema")
    if config.get("stage") != "A2-TQIF-4" or config.get("kind") != "mechanism_ablation":
        raise TQIFProtocolError("INVALID_ARTIFACT", "invalid TQIF ablation stage")
    if tuple(config.get("recipes", ())) != ("tqif_token16_pair16", "tqif_token32_pair32"):
        raise TQIFProtocolError("INVALID_ARTIFACT", "ablation recipes are not frozen")
    if tuple(config.get("seeds", ())) != TQIF_SEEDS:
        raise TQIFProtocolError("INVALID_ARTIFACT", "ablation seeds are not frozen")
    if config.get("allowed_read_splits") != list(TQIF_ALLOWED_READ_SPLITS):
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "ablation split allowlist changed")
    if config.get("formal_holdout_access") != "locked" or config.get("test_access") != "locked":
        raise TQIFProtocolError("PROTOCOL_ACCESS_VIOLATION", "ablation holdout access must remain locked")
    matrix = config.get("matrix")
    if not isinstance(matrix, list) or {entry.get("id") for entry in matrix if isinstance(entry, Mapping)} != {
        "C0",
        "C1",
        "Q1",
        "I1",
        "TQIF-H0",
        "TQIF-STR",
    }:
        raise TQIFProtocolError("INVALID_ARTIFACT", "ablation matrix is incomplete")


def _validate_data_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != "gf-a1-data-1":
        raise TQIFProtocolError("INVALID_ARTIFACT", "unsupported A1 data manifest schema")
    conditions = manifest.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise TQIFProtocolError("INVALID_ARTIFACT", "data manifest conditions are missing")
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise TQIFProtocolError("INVALID_ARTIFACT", "data manifest condition is not an object")
        split = condition.get("split")
        if split not in {"train", "val", "test"}:
            raise TQIFProtocolError("INVALID_ARTIFACT", f"unsupported data split: {split!r}")
        if not isinstance(condition.get("mixture_id"), str) or not condition["mixture_id"]:
            raise TQIFProtocolError("INVALID_ARTIFACT", "data condition mixture_id is missing")


def _validate_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = TQIF_FORBIDDEN_KEYS & set(value)
        if forbidden:
            raise TQIFProtocolError(
                "PROTOCOL_ACCESS_VIOLATION",
                f"forbidden legacy keys: {sorted(forbidden)}",
            )
        for child in value.values():
            _validate_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _validate_no_forbidden_keys(child)


def _split_manifest_hash(manifest: Mapping[str, Any]) -> str:
    assignments = sorted(
        {
            (str(condition["mixture_id"]), str(condition["split"]))
            for condition in manifest["conditions"]
        }
    )
    return canonical_hash(
        {
            "schema_version": "tqif-split-1",
            "assignments": [
                {"mixture_id": mixture_id, "split": split}
                for mixture_id, split in assignments
            ],
        }
    )


def _resolve_configured_path(
    root: Path,
    config: Mapping[str, Any],
    key: str,
    *,
    index: int | None = None,
) -> Path:
    value = config[key] if index is None else config[key][index]
    _reject_forbidden_path(value, key)
    return resolve_project_file(root, value)


def _reject_forbidden_path(value: Any, name: str) -> None:
    if not isinstance(value, (str, Path)):
        return
    parts = {part.lower() for part in Path(value).parts}
    if any(part in {"test", "tests", "test_data", "a1_test", "a2_test"} or "test" in part for part in parts):
        raise TQIFProtocolError(
            "PROTOCOL_ACCESS_VIOLATION",
            f"{name} points at a forbidden test path: {value}",
        )


def _target_registry(config: Mapping[str, Any]) -> TQIFTargetSlotRegistry:
    return TQIFTargetSlotRegistry.from_mappings(config["target_slot_registry"])


def _sensor_registry(config: Mapping[str, Any]) -> tuple[Any, ...]:
    from gf.dl.tqif import TQIFSensorSpec

    return tuple(
        TQIFSensorSpec(sensor_id, sensor_type)
        for sensor_id, sensor_type in zip(
            config["sensor_ids"], config["sensor_types"], strict=True
        )
    )


def _require_protocol_passed(root: Path) -> dict[str, Any]:
    path = root / "outputs" / "runs" / "a2" / "tqif" / "protocol_manifest.json"
    if not path.is_file():
        raise TQIFProtocolError(
            "PREREQUISITE_NOT_PASSED",
            "A2-TQIF-0 protocol_manifest.json does not exist",
        )
    manifest = read_json_object(path)
    if manifest.get("status") != "PASS":
        raise TQIFProtocolError(
            "PREREQUISITE_NOT_PASSED",
            "A2-TQIF-0 protocol manifest is not PASS",
        )
    if manifest.get("source") != git_snapshot(root):
        raise TQIFProtocolError(
            "RUN_ID_HASH_CONFLICT",
            "source snapshot changed after protocol; rerun protocol before continuing",
        )
    return manifest


def _load_model_configs_for_protocol(
    root: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    full: dict[str, Mapping[str, Any]] = {}
    matched: dict[str, Mapping[str, Any]] = {}
    for relative, expected_hash in protocol["model_config_hashes"].items():
        config = read_json_object(resolve_project_file(root, relative))
        if canonical_hash(config) != expected_hash:
            raise TQIFProtocolError(
                "RUN_ID_HASH_CONFLICT",
                f"model config changed after protocol: {relative}",
            )
        if config["model_id"] == "TQIF":
            full[str(config["recipe"])] = config
        else:
            matched[str(config["recipe"])] = config
    return full, matched


def _project_root(project_root: str | Path | None) -> Path:
    return (Path(project_root) if project_root is not None else Path(__file__).resolve().parents[3]).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one TQIF A2 protocol stage.")
    parser.add_argument(
        "--stage",
        choices=("protocol", "smoke", "baseline", "development", "ablation", "select", "all"),
        default="protocol",
    )
    parser.add_argument("--project-root", type=Path, default=_project_root(None))
    parser.add_argument("--protocol-config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_tqif(
            project_root=args.project_root,
            stage=args.stage,
            protocol_config_path=args.protocol_config,
        )
    except TQIFArtifactError as exc:
        failure_path = (
            args.project_root.resolve()
            / "outputs"
            / "runs"
            / "a2"
            / "tqif"
            / "failures.log"
        )
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "stage": args.stage,
                        "error_code": exc.code,
                        "message": exc.message,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        print(json.dumps({"status": "BLOCKED", "error_code": exc.code, "message": exc.message}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"stage": result.get("stage"), "status": result.get("status")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TQIFProtocolError",
    "build_parameter_profiles",
    "build_tqif_variant",
    "main",
    "run_tqif",
    "run_tqif_ablation",
    "run_tqif_baseline",
    "run_tqif_development",
    "run_tqif_protocol",
    "run_tqif_select",
    "run_tqif_smoke",
]
