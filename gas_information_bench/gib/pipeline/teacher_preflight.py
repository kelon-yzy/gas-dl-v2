"""P3-11 raw-teacher activation gate and conditional CR-PKD runner."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from ..common.io import atomic_promote_directory, atomic_write_json, remove_owned_staging, sha256_file
from ..contract import ContractError
from ..freeze import verify_evidence_manifest
from .baseline import _array, _labels, _partition_indices, deployment_features, load_pilot_metadata


COMPONENTS = ("N2", "CO2", "O2", "Ar")


class TeacherPreflightError(ValueError):
    """Raised when the C5-C frozen teacher gate cannot be evaluated."""


def _raw_features(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1 or not np.all(np.isfinite(array)):
        raise TeacherPreflightError("raw teacher input must be finite channel-by-time data")
    return np.concatenate([array.mean(axis=1), array.std(axis=1), array.min(axis=1), array.max(axis=1)])


def _xgb_model(specification: Mapping[str, Any], seed: int) -> Any:
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor

    return MultiOutputRegressor(
        XGBRegressor(
            n_estimators=int(specification["n_estimators"]),
            max_depth=int(specification["max_depth"]),
            learning_rate=float(specification["learning_rate"]),
            subsample=float(specification["subsample"]),
            colsample_bytree=float(specification["colsample_bytree"]),
            n_jobs=int(specification["n_jobs"]),
            tree_method=str(specification["tree_method"]),
            random_state=seed,
            objective="reg:squarederror",
        )
    )


def _group_bootstrap(rows: list[dict[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise TeacherPreflightError("teacher gate requires paired rows")
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((str(row["split_id"]), str(row["mixture_id"])), []).append(index)
    group_rows = list(groups.values())
    resamples = int(config["statistics"]["bootstrap_resamples"])
    rng = np.random.default_rng(int(config["statistics"]["bootstrap_seed"]))
    candidate = np.asarray([row["teacher_errors"] for row in rows], dtype=np.float64)
    reference = np.asarray([row["dsp_errors"] for row in rows], dtype=np.float64)
    absolute = np.empty((resamples, 4), dtype=np.float64)
    relative = np.empty((resamples, 4), dtype=np.float64)
    for repeat in range(resamples):
        selected = np.concatenate([group_rows[int(index)] for index in rng.integers(0, len(group_rows), size=len(group_rows))])
        candidate_p90 = np.quantile(candidate[selected], 0.90, axis=0, method="higher")
        reference_p90 = np.quantile(reference[selected], 0.90, axis=0, method="higher")
        absolute[repeat] = candidate_p90 - reference_p90
        relative[repeat] = 1.0 - np.divide(candidate_p90, reference_p90, out=np.ones(4), where=reference_p90 > 0.0)
    alpha = (1.0 - float(config["statistics"]["confidence_level"])) / 2.0
    result = {
        "absolute_non_inferiority": [],
        "relative_improvement": [],
    }
    bands = config["gates"]["student_non_inferiority_bands"]
    for index, component in enumerate(COMPONENTS):
        result["absolute_non_inferiority"].append(
            {
                "component": component,
                "point": float(np.quantile(candidate[:, index], 0.90, method="higher") - np.quantile(reference[:, index], 0.90, method="higher")),
                "ci_upper": float(np.quantile(absolute[:, index], 1.0 - alpha)),
                "band": float(bands[component]),
            }
        )
        result["relative_improvement"].append(
            {
                "component": component,
                "point": float(np.mean(relative[:, index])),
                "ci_lower": float(np.quantile(relative[:, index], alpha)),
                "minimum": float(config["gates"]["teacher_minimum_relative_improvement"]),
            }
        )
    result["ni_pass"] = all(item["ci_upper"] <= item["band"] for item in result["absolute_non_inferiority"])
    result["teacher_improvement_pass"] = any(item["ci_lower"] >= item["minimum"] for item in result["relative_improvement"])
    result["teacher_gate_pass"] = bool(result["ni_pass"] and result["teacher_improvement_pass"])
    return result


def _write_attempt(target: Path, result: Mapping[str, Any], candidate_verdict: str) -> None:
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        atomic_write_json(staging / "teacher_preflight_results.json", result)
        atomic_write_json(
            staging / "attempt_manifest.json",
            {
                "schema_version": "gib-benchmark-1",
                "attempt_id": target.name,
                "task_id": "P3-11",
                "status": "complete",
                "task_status": "completed",
                "candidate_verdict": candidate_verdict,
                "claim_scope": result["claim_scope"],
                "next_allowed_task": "P3-13",
            },
        )
        atomic_promote_directory(staging, target)
    except Exception:
        remove_owned_staging(staging)
        raise


def run_teacher_preflight(
    config: dict[str, Any],
    baseline_plan: Mapping[str, Any],
    *,
    pilot_freeze: Path,
    baseline_freeze: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if config.get("schema_version") != "gib-benchmark-1" or config.get("task_id") != "P3-11":
        raise TeacherPreflightError("C5-C plan identity mismatch")
    if config.get("plan_status") != "frozen_before_teacher_fit":
        raise TeacherPreflightError("C5-C teacher plan must be frozen before fit")
    verify_evidence_manifest(pilot_freeze)
    verify_evidence_manifest(baseline_freeze)
    baseline_snapshot = Path(baseline_freeze) / "source_snapshots" / "gas_information_bench" / "configs" / "p3_baseline_plan.json"
    if json.loads(baseline_snapshot.read_text(encoding="utf-8")) != baseline_plan:
        raise TeacherPreflightError("working baseline plan differs from the frozen G3-3 input")
    metadata = load_pilot_metadata(pilot_freeze)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"attempt directory already exists: {target}")
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        raw_features = np.asarray([_raw_features(_array(metadata, index, "raw_waveform")) for index in range(len(metadata.records))], dtype=np.float64)
        dsp_features, _ = deployment_features(metadata)
        labels = _labels(metadata, np.arange(len(metadata.records), dtype=int))
        model_spec = baseline_plan["models"]["xgboost_strong_table"]
        rows: list[dict[str, Any]] = []
        for split_id in baseline_plan["split_ids"]:
            partitions = _partition_indices(metadata, str(split_id))
            train = partitions["train"]
            test = partitions["test"]
            for seed in baseline_plan["seeds"]:
                raw_model = _xgb_model(model_spec, int(seed))
                dsp_model = _xgb_model(model_spec, int(seed))
                raw_model.fit(raw_features[train], labels[train])
                dsp_model.fit(dsp_features[train], labels[train])
                raw_prediction = np.asarray(raw_model.predict(raw_features[test]), dtype=np.float64)
                dsp_prediction = np.asarray(dsp_model.predict(dsp_features[test]), dtype=np.float64)
                truth = labels[test]
                for local, index in enumerate(test):
                    record = metadata.records[int(index)]
                    rows.append(
                        {
                            "mixture_id": str(record["mixture_id"]),
                            "sequence_id": str(record["sequence_id"]),
                            "grid_cell_id": str(record["grade"]["grid_cell_id"]),
                            "split_id": str(split_id),
                            "seed": int(seed),
                            "teacher_errors": np.abs(raw_prediction[local] - truth[local]).tolist(),
                            "dsp_errors": np.abs(dsp_prediction[local] - truth[local]).tolist(),
                            "raw_teacher_input": "raw_waveform_summary_only",
                            "student_input": "not_activated",
                        }
                    )
        gate = _group_bootstrap(rows, config)
        if gate["teacher_gate_pass"]:
            raise TeacherPreflightError("teacher gate passed; the CR-PKD student stage must run before this task can complete")
        candidate_verdict = "not_activated"
        result = {
            "schema_version": "gib-benchmark-1",
            "task_id": "P3-11",
            "task_status": "completed",
            "candidate_verdict": candidate_verdict,
            "teacher_gate": gate,
            "student": {"status": "not_run"},
            "coverage": {"row_count": len(rows), "grid_cells": sorted({row["grid_cell_id"] for row in rows}), "split_count": len({row["split_id"] for row in rows}), "seed_count": len({row["seed"] for row in rows})},
            "rows": rows,
            "claim_scope": config["claim_scope"],
            "next_allowed_task": "P3-13",
            "provenance": {"pilot_freeze_id": Path(pilot_freeze).name, "baseline_freeze_id": Path(baseline_freeze).name, "teacher_code_sha256": sha256_file(Path(__file__)), "dataset_manifest_id": metadata.generation_summary["dataset_manifest_id"]},
        }
        _write_attempt(target, result, candidate_verdict)
        remove_owned_staging(staging)
        return result
    except Exception:
        remove_owned_staging(staging)
        raise


__all__ = ["TeacherPreflightError", "run_teacher_preflight"]
