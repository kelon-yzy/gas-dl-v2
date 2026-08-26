"""Pre-registered C5-A adaptive-sampling evaluation.

The module owns policy selection and paired evaluation only.  It deliberately
does not manufacture per-mixture predictions from aggregate baseline metrics.
Stage A receives a simulation callback; stage B receives a deployment runner
that is called with the physically truncated waveform.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from ..common.io import atomic_promote_directory, atomic_write_json, remove_owned_staging, sha256_file
from ..freeze import verify_evidence_manifest
from .baseline import _array, _labels, _partition_indices, load_pilot_metadata


COMPONENTS = ("N2", "CO2", "O2", "Ar")
PRIMARY_COSTS = (
    "measurement_time_ms",
    "sample_points",
    "active_modality_count",
    "flops",
)
MAIN_METHODS = (
    "full_sampling",
    "fixed_short_window",
    "uncertainty_early_exit",
    "crb_early_exit",
    "crb_dynamic_modality",
)
NEGATIVE_CONTROLS = ("random_stop", "equal_length_fixed", "crb_rank_shuffle")
RAW_MODALITY_CHANNELS = {
    "ndir": slice(0, 4),
    "acoustic_raw": slice(4, 6),
    "thermal": slice(6, 8),
}
ORACLE_POLICY_FIELDS = frozenset(
    {"oracle_features", "truth_nuisance", "truth_derived_features", "oracle_results", "labels"}
)


class AdaptiveSamplingError(ValueError):
    """Raised when C5-A inputs violate the frozen evaluation contract."""


@dataclass(frozen=True)
class SamplingCheckpoint:
    """One deployable stopping point in increasing acquisition order."""

    sample_points: int
    measurement_time_ms: float
    active_modalities: tuple[str, ...]
    estimated_crb_p90: tuple[float, float, float, float]
    uncertainty: tuple[float, float, float, float]
    delta_i: float
    delta_i_per_cost: float
    cumulative_information_fraction: float
    flops: float


@dataclass(frozen=True)
class SamplingObservation:
    """A paired evaluation unit; policy inputs remain separate from truth."""

    mixture_id: str
    sequence_id: str
    grid_cell_id: str
    information_band: str
    split_id: str
    seed: int
    truth: tuple[float, float, float, float]
    reference_prediction: tuple[float, float, float, float]
    raw_waveform: np.ndarray
    checkpoints: tuple[SamplingCheckpoint, ...]
    policy_inputs: Mapping[str, Any]
    slow_channels: np.ndarray | None = None
    calibration_channels: np.ndarray | None = None


StageAPredictor = Callable[[SamplingObservation, SamplingCheckpoint, str], Sequence[float]]
StageBRunner = Callable[[SamplingObservation, SamplingCheckpoint, np.ndarray], Sequence[float]]


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def validate_sampling_plan(plan: Mapping[str, Any]) -> None:
    """Validate the values that P2 and P3 freeze before Stage A."""

    if plan.get("schema_version") != "gib-benchmark-1":
        raise AdaptiveSamplingError("sampling plan schema_version mismatch")
    if plan.get("plan_status") != "frozen_before_stage_a":
        raise AdaptiveSamplingError("sampling plan must be frozen before stage A")
    if tuple(plan.get("components", ())) != COMPONENTS:
        raise AdaptiveSamplingError("component order must remain N2/CO2/O2/Ar")
    expected_splits = tuple(f"GIB-SPLIT-{index:02d}" for index in range(1, 6))
    if tuple(plan.get("split_ids", ())) != expected_splits or tuple(plan.get("seeds", ())) != (101, 202, 303):
        raise AdaptiveSamplingError("split IDs and seeds do not match the frozen paired design")
    cells = plan.get("grid_cells")
    if not isinstance(cells, list) or len(cells) != 9 or len(set(cells)) != 9:
        raise AdaptiveSamplingError("sampling plan must contain nine unique grid cells")
    if tuple(plan.get("stage_a", {}).get("methods", ())) != MAIN_METHODS:
        raise AdaptiveSamplingError("stage A methods do not match P3-07")
    if tuple(plan.get("negative_controls", {}).get("methods", ())) != NEGATIVE_CONTROLS:
        raise AdaptiveSamplingError("negative controls do not match P3-07")
    if plan.get("candidate_method") != "crb_dynamic_modality":
        raise AdaptiveSamplingError("candidate method must be pre-registered")
    if plan.get("threshold_selection") != "pre_registered_no_result_tuning":
        raise AdaptiveSamplingError("result-dependent threshold selection is forbidden")
    statistics = plan.get("statistics", {})
    if statistics != {
        "confidence_level": 0.95,
        "bootstrap_type": "paired_group_bootstrap",
        "resampling_unit": "mixture_id",
        "resamples": 10000,
        "seed": 20260824,
        "stratify_by": ["grid_cell_id", "split_id", "seed"],
    }:
        raise AdaptiveSamplingError("paired bootstrap contract drift")
    expected_ni = {"N2": 0.008, "CO2": 0.003, "O2": 0.010, "Ar": 0.005}
    gates = plan.get("gates", {})
    if gates.get("ni_bands") != expected_ni:
        raise AdaptiveSamplingError("NI bands do not match the authorized P2 values")
    if float(gates.get("minimum_raw_cost_reduction", -1.0)) != 0.20:
        raise AdaptiveSamplingError("raw-cost reduction gate must remain 20 percent")
    if float(gates.get("maximum_other_cost_regression", -1.0)) != 0.05:
        raise AdaptiveSamplingError("NR5 gate must remain five percent")
    policy = plan.get("policies", {}).get("crb_dynamic_modality", {})
    if set(policy) != {"maximum_relative_crb", "minimum_information_fraction"}:
        raise AdaptiveSamplingError("candidate policy thresholds must be explicit")
    if float(policy["maximum_relative_crb"]) < 1.0:
        raise AdaptiveSamplingError("relative CRB threshold cannot beat the full checkpoint")
    fraction = float(policy["minimum_information_fraction"])
    if not 0.0 < fraction <= 1.0:
        raise AdaptiveSamplingError("information fraction must be in (0, 1]")


def _vector(value: Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4,) or not np.all(np.isfinite(array)):
        raise AdaptiveSamplingError(f"{label} must contain four finite component values")
    return array


def _validate_observation(observation: SamplingObservation, plan: Mapping[str, Any]) -> None:
    if not observation.mixture_id.startswith("GIB-M-") or not observation.sequence_id.startswith("GIB-Q-"):
        raise AdaptiveSamplingError("mixture_id and sequence_id namespaces are invalid")
    if observation.grid_cell_id not in plan["grid_cells"]:
        raise AdaptiveSamplingError("observation uses an unknown grid cell")
    if observation.split_id not in plan["split_ids"] or observation.seed not in plan["seeds"]:
        raise AdaptiveSamplingError("observation uses an unknown split or seed")
    if observation.information_band not in {"sufficient", "critical", "insufficient"}:
        raise AdaptiveSamplingError("unknown information band")
    _vector(observation.truth, "truth")
    _vector(observation.reference_prediction, "reference_prediction")
    raw = np.asarray(observation.raw_waveform)
    if raw.ndim < 1 or raw.shape[-1] < 1 or not np.all(np.isfinite(raw)):
        raise AdaptiveSamplingError("raw waveform must be finite and have a time axis")
    forbidden = ORACLE_POLICY_FIELDS & set(observation.policy_inputs)
    if forbidden:
        raise AdaptiveSamplingError(f"oracle fields are forbidden in policy inputs: {sorted(forbidden)}")
    if not observation.checkpoints:
        raise AdaptiveSamplingError("at least one checkpoint is required")
    previous_points = 0
    previous_time = -math.inf
    previous_information = -math.inf
    for checkpoint in observation.checkpoints:
        if checkpoint.sample_points <= previous_points or checkpoint.sample_points > raw.shape[-1]:
            raise AdaptiveSamplingError("checkpoint sample points must increase within the raw waveform")
        if checkpoint.measurement_time_ms <= previous_time or checkpoint.measurement_time_ms <= 0.0:
            raise AdaptiveSamplingError("checkpoint measurement time must strictly increase")
        if checkpoint.cumulative_information_fraction < previous_information:
            raise AdaptiveSamplingError("checkpoint information fraction must be monotone")
        if not 0.0 < checkpoint.cumulative_information_fraction <= 1.0:
            raise AdaptiveSamplingError("checkpoint information fraction must be in (0, 1]")
        if not checkpoint.active_modalities or len(set(checkpoint.active_modalities)) != len(checkpoint.active_modalities):
            raise AdaptiveSamplingError("active modalities must be a non-empty unique tuple")
        _vector(checkpoint.estimated_crb_p90, "estimated_crb_p90")
        _vector(checkpoint.uncertainty, "uncertainty")
        for value in (checkpoint.delta_i, checkpoint.delta_i_per_cost, checkpoint.flops):
            if not math.isfinite(value) or value < 0.0:
                raise AdaptiveSamplingError("information and FLOP values must be finite and non-negative")
        previous_points = checkpoint.sample_points
        previous_time = checkpoint.measurement_time_ms
        previous_information = checkpoint.cumulative_information_fraction
    if observation.checkpoints[-1].sample_points != raw.shape[-1]:
        raise AdaptiveSamplingError("last checkpoint must represent full sampling")


def _first(checkpoints: Sequence[SamplingCheckpoint], predicate: Callable[[SamplingCheckpoint], bool]) -> SamplingCheckpoint:
    for checkpoint in checkpoints:
        if predicate(checkpoint):
            return checkpoint
    return checkpoints[-1]


def select_checkpoint(
    observation: SamplingObservation,
    method: str,
    plan: Mapping[str, Any],
    *,
    candidate_checkpoint: SamplingCheckpoint | None = None,
) -> SamplingCheckpoint:
    """Apply one frozen policy without reading truth or reference predictions."""

    checkpoints = observation.checkpoints
    full = checkpoints[-1]
    policies = plan["policies"]
    if method == "full_sampling":
        return full
    if method == "fixed_short_window":
        minimum = math.ceil(full.sample_points * float(policies[method]["sample_fraction"]))
        return _first(checkpoints, lambda item: item.sample_points >= minimum)
    if method == "uncertainty_early_exit":
        threshold = _vector(policies[method]["component_thresholds"], "uncertainty threshold")
        return _first(checkpoints, lambda item: bool(np.all(_vector(item.uncertainty, "uncertainty") <= threshold)))
    if method in {"crb_early_exit", "crb_dynamic_modality"}:
        specification = policies[method]
        maximum_relative = float(specification["maximum_relative_crb"])
        full_crb = _vector(full.estimated_crb_p90, "full CRB")

        def clears(item: SamplingCheckpoint) -> bool:
            relative = np.divide(
                _vector(item.estimated_crb_p90, "checkpoint CRB"),
                full_crb,
                out=np.full(4, np.inf),
                where=full_crb > 0.0,
            )
            if not bool(np.all(relative <= maximum_relative)):
                return False
            if method == "crb_dynamic_modality":
                return item.cumulative_information_fraction >= float(specification["minimum_information_fraction"])
            return True

        return _first(checkpoints, clears)
    if method == "random_stop":
        digest = hashlib.sha256(
            f"{plan['negative_controls']['random_seed']}|{observation.sequence_id}".encode("utf-8")
        ).digest()
        return checkpoints[int.from_bytes(digest[:8], "big") % len(checkpoints)]
    if method == "equal_length_fixed":
        if candidate_checkpoint is None:
            raise AdaptiveSamplingError("equal-length control requires the candidate checkpoint")
        return _first(checkpoints, lambda item: item.sample_points >= candidate_checkpoint.sample_points)
    if method == "crb_rank_shuffle":
        digest = hashlib.sha256(
            f"{plan['negative_controls']['rank_shuffle_seed']}|{observation.sequence_id}".encode("utf-8")
        ).digest()
        order = np.random.default_rng(int.from_bytes(digest[:8], "big")).permutation(len(checkpoints))
        candidate = select_checkpoint(observation, "crb_dynamic_modality", plan)
        return checkpoints[int(order[checkpoints.index(candidate)])]
    raise AdaptiveSamplingError(f"unknown sampling method: {method}")


def _costs(checkpoint: SamplingCheckpoint) -> dict[str, float]:
    return {
        "measurement_time_ms": checkpoint.measurement_time_ms,
        "sample_points": float(checkpoint.sample_points),
        "active_modality_count": float(len(checkpoint.active_modalities)),
        "flops": checkpoint.flops,
    }


def _method_rows(
    observations: Sequence[SamplingObservation],
    plan: Mapping[str, Any],
    method: str,
    predictor: StageAPredictor,
) -> list[dict[str, Any]]:
    rows = []
    for observation in observations:
        candidate_checkpoint = select_checkpoint(observation, "crb_dynamic_modality", plan)
        checkpoint = select_checkpoint(
            observation,
            method,
            plan,
            candidate_checkpoint=candidate_checkpoint,
        )
        prediction = (
            _vector(observation.reference_prediction, "reference_prediction")
            if method == "full_sampling"
            else _vector(predictor(observation, checkpoint, method), "stage A prediction")
        )
        rows.append(_row(observation, method, checkpoint, prediction))
    return rows


def _row(
    observation: SamplingObservation,
    method: str,
    checkpoint: SamplingCheckpoint,
    prediction: np.ndarray,
) -> dict[str, Any]:
    return {
        "mixture_id": observation.mixture_id,
        "sequence_id": observation.sequence_id,
        "grid_cell_id": observation.grid_cell_id,
        "information_band": observation.information_band,
        "split_id": observation.split_id,
        "seed": observation.seed,
        "method_id": method,
        "absolute_error": np.abs(prediction - _vector(observation.truth, "truth")).tolist(),
        "reference_absolute_error": np.abs(
            _vector(observation.reference_prediction, "reference_prediction")
            - _vector(observation.truth, "truth")
        ).tolist(),
        "costs": _costs(checkpoint),
        "reference_costs": _costs(observation.checkpoints[-1]),
        "stop_sample_points": checkpoint.sample_points,
        "active_modalities": list(checkpoint.active_modalities),
        "delta_i": checkpoint.delta_i,
        "delta_i_per_cost": checkpoint.delta_i_per_cost,
        "cumulative_information_fraction": checkpoint.cumulative_information_fraction,
    }


def _bootstrap_gate(rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> dict[str, Any]:
    if not rows:
        raise AdaptiveSamplingError("paired evaluation requires non-empty rows")
    strata: dict[tuple[str, str, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        key = (str(row["grid_cell_id"]), str(row["split_id"]), int(row["seed"]))
        strata[key][str(row["mixture_id"])].append(index)
    errors = np.asarray([row["absolute_error"] for row in rows], dtype=np.float64)
    reference_errors = np.asarray([row["reference_absolute_error"] for row in rows], dtype=np.float64)
    cost = {name: np.asarray([row["costs"][name] for row in rows], dtype=np.float64) for name in PRIMARY_COSTS}
    reference_cost = {
        name: np.asarray([row["reference_costs"][name] for row in rows], dtype=np.float64)
        for name in PRIMARY_COSTS
    }
    if any(np.any(reference_cost[name] <= 0.0) for name in PRIMARY_COSTS):
        raise AdaptiveSamplingError("reference costs must be positive")
    repeats = int(plan["statistics"]["resamples"])
    ni_samples = np.empty((repeats, 4), dtype=np.float64)
    reduction_samples = {name: np.empty(repeats, dtype=np.float64) for name in PRIMARY_COSTS}
    grouped = [list(groups.values()) for groups in strata.values()]
    deterministic = all(len(groups) == 1 for groups in grouped)
    rng = np.random.default_rng(int(plan["statistics"]["seed"]))
    for repeat in range(1 if deterministic else repeats):
        if deterministic:
            selected = [index for groups in grouped for index in groups[0]]
        else:
            selected = []
            for groups in grouped:
                draw = rng.integers(0, len(groups), size=len(groups))
                for group_index in draw:
                    selected.extend(groups[int(group_index)])
        indices = np.asarray(selected, dtype=int)
        ni_samples[repeat] = (
            np.quantile(errors[indices], 0.90, axis=0, method="higher")
            - np.quantile(reference_errors[indices], 0.90, axis=0, method="higher")
        )
        for name in PRIMARY_COSTS:
            reduction_samples[name][repeat] = 1.0 - float(np.mean(cost[name][indices])) / float(
                np.mean(reference_cost[name][indices])
            )
    if deterministic:
        ni_samples[1:] = ni_samples[0]
        for samples in reduction_samples.values():
            samples[1:] = samples[0]
    alpha = (1.0 - float(plan["statistics"]["confidence_level"])) / 2.0
    ni_upper = np.quantile(ni_samples, 1.0 - alpha, axis=0)
    bands = plan["gates"]["ni_bands"]
    ni = {
        component: {
            "point": float(
                np.quantile(errors[:, index], 0.90, method="higher")
                - np.quantile(reference_errors[:, index], 0.90, method="higher")
            ),
            "ci_upper": float(ni_upper[index]),
            "band": float(bands[component]),
            "passed": bool(ni_upper[index] <= float(bands[component])),
        }
        for index, component in enumerate(COMPONENTS)
    }
    costs = {}
    for name in PRIMARY_COSTS:
        samples = reduction_samples[name]
        costs[name] = {
            "point_reduction": 1.0 - float(np.mean(cost[name])) / float(np.mean(reference_cost[name])),
            "ci_lower_reduction": float(np.quantile(samples, alpha)),
            "ci_upper_regression": float(np.quantile(-samples, 1.0 - alpha)),
        }
    evidence_cost = str(plan["gates"]["pre_registered_evidence_cost"])
    if evidence_cost not in PRIMARY_COSTS:
        raise AdaptiveSamplingError("unknown pre-registered evidence cost")
    raw_cost_pass = costs[evidence_cost]["ci_lower_reduction"] >= float(
        plan["gates"]["minimum_raw_cost_reduction"]
    )
    nr5_pass = all(
        costs[name]["ci_upper_regression"] <= float(plan["gates"]["maximum_other_cost_regression"])
        for name in PRIMARY_COSTS
        if name != evidence_cost
    )
    return {
        "ni": ni,
        "costs": costs,
        "evidence_cost": evidence_cost,
        "ni_pass": all(item["passed"] for item in ni.values()),
        "raw_cost_pass": bool(raw_cost_pass),
        "nr5_pass": bool(nr5_pass),
        "joint_gate_pass": bool(all(item["passed"] for item in ni.values()) and raw_cost_pass and nr5_pass),
    }


def _stratum_summaries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["grid_cell_id"]), str(row["split_id"]), int(row["seed"]))].append(row)
    summaries = []
    for (cell, split_id, seed), items in sorted(grouped.items()):
        errors = np.asarray([item["absolute_error"] for item in items], dtype=np.float64)
        reference = np.asarray([item["reference_absolute_error"] for item in items], dtype=np.float64)
        summaries.append(
            {
                "grid_cell_id": cell,
                "split_id": split_id,
                "seed": seed,
                "mixture_group_count": len({str(item["mixture_id"]) for item in items}),
                "sequence_count": len(items),
                "p90_difference": {
                    component: float(
                        np.quantile(errors[:, index], 0.90, method="higher")
                        - np.quantile(reference[:, index], 0.90, method="higher")
                    )
                    for index, component in enumerate(COMPONENTS)
                },
                "cost_reduction": {
                    name: 1.0
                    - float(np.mean([item["costs"][name] for item in items]))
                    / float(np.mean([item["reference_costs"][name] for item in items]))
                    for name in PRIMARY_COSTS
                },
                "mean_delta_i": float(np.mean([item["delta_i"] for item in items])),
                "mean_delta_i_per_cost": float(np.mean([item["delta_i_per_cost"] for item in items])),
            }
        )
    return summaries


def _coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells = sorted({str(row["grid_cell_id"]) for row in rows})
    bands = sorted({str(row["information_band"]) for row in rows})
    combinations = sorted({(str(row["grid_cell_id"]), str(row["split_id"]), int(row["seed"])) for row in rows})
    return {"grid_cells": cells, "information_bands": bands, "grid_split_seed_count": len(combinations)}


def simulate_stage_a(
    observations: Sequence[SamplingObservation],
    plan: Mapping[str, Any],
    predictor: StageAPredictor,
) -> dict[str, Any]:
    """Run the inexpensive comparison and all registered negative controls."""

    validate_sampling_plan(plan)
    if not observations:
        raise AdaptiveSamplingError("stage A requires observations")
    for observation in observations:
        _validate_observation(observation, plan)
    coverage = _coverage([_row(obs, "full_sampling", obs.checkpoints[-1], _vector(obs.reference_prediction, "reference")) for obs in observations])
    if set(coverage["grid_cells"]) != set(plan["grid_cells"]):
        raise AdaptiveSamplingError("stage A must cover all nine grid cells")
    if not {"critical", "insufficient"}.issubset(coverage["information_bands"]):
        raise AdaptiveSamplingError("stage A must cover critical and insufficient bands")
    methods = {}
    method_rows = {}
    for method in MAIN_METHODS:
        rows = _method_rows(observations, plan, method, predictor)
        method_rows[method] = rows
        methods[method] = {**_bootstrap_gate(rows, plan), "strata": _stratum_summaries(rows)}
    controls = {}
    for method in NEGATIVE_CONTROLS:
        rows = _method_rows(observations, plan, method, predictor)
        method_rows[method] = rows
        controls[method] = {**_bootstrap_gate(rows, plan), "strata": _stratum_summaries(rows)}
    candidate_gate = methods[plan["candidate_method"]]["joint_gate_pass"]
    controls_pass = not bool(plan["negative_controls"]["must_fail_joint_gate"]) or not any(
        result["joint_gate_pass"] for result in controls.values()
    )
    return {
        "stage": "A",
        "coverage": coverage,
        "methods": methods,
        "negative_controls": controls,
        "stage_a_gate_pass": bool(candidate_gate and controls_pass),
        "candidate_gate_pass": bool(candidate_gate),
        "negative_controls_pass": bool(controls_pass),
        "rows": method_rows,
    }


def freeze_strategy(stage_a: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Bind Stage B to pre-registered thresholds; Stage A cannot alter them."""

    validate_sampling_plan(plan)
    if not stage_a.get("stage_a_gate_pass"):
        raise AdaptiveSamplingError("stage A did not authorize stage B")
    policy = dict(plan["policies"][plan["candidate_method"]])
    frozen_payload = {
        "strategy_id": plan["strategy_id"],
        "method_id": plan["candidate_method"],
        "policy": policy,
        "threshold_source": "p3_c5a_sampling_plan.json",
        "threshold_selection": plan["threshold_selection"],
    }
    return {**frozen_payload, "strategy_sha256": _canonical_sha256(frozen_payload)}


def run_stage_b(
    observations: Sequence[SamplingObservation],
    plan: Mapping[str, Any],
    frozen_strategy: Mapping[str, Any],
    runner: StageBRunner,
) -> dict[str, Any]:
    """Physically truncate inputs, rerun the fixed method, and evaluate 9x5x3."""

    validate_sampling_plan(plan)
    expected = freeze_strategy({"stage_a_gate_pass": True}, plan)
    if dict(frozen_strategy) != expected:
        raise AdaptiveSamplingError("stage B strategy differs from the frozen Stage A policy")
    rows = []
    for observation in observations:
        _validate_observation(observation, plan)
        checkpoint = select_checkpoint(observation, plan["candidate_method"], plan)
        truncated = np.asarray(observation.raw_waveform)[..., : checkpoint.sample_points].copy()
        if truncated.shape[-1] != checkpoint.sample_points:
            raise AdaptiveSamplingError("stage B input was not physically truncated")
        prediction = _vector(runner(observation, checkpoint, truncated), "stage B prediction")
        rows.append(_row(observation, str(plan["candidate_method"]), checkpoint, prediction))
    coverage = _coverage(rows)
    expected_count = len(plan["grid_cells"]) * len(plan["split_ids"]) * len(plan["seeds"])
    if coverage["grid_split_seed_count"] != expected_count:
        raise AdaptiveSamplingError("stage B must cover every grid/split/seed combination")
    gate = _bootstrap_gate(rows, plan)
    return {
        "stage": "B",
        "coverage": coverage,
        "gate": gate,
        "strata": _stratum_summaries(rows),
        "rows": rows,
    }


def run_c5a(
    observations: Sequence[SamplingObservation],
    plan: Mapping[str, Any],
    stage_a_predictor: StageAPredictor,
    stage_b_runner: StageBRunner,
) -> dict[str, Any]:
    """Execute the P3-07 state machine and return its terminal candidate verdict."""

    stage_a = simulate_stage_a(observations, plan, stage_a_predictor)
    if not stage_a["stage_a_gate_pass"]:
        return {
            "schema_version": "gib-benchmark-1",
            "task_id": "P3-07",
            "task_status": "completed",
            "candidate_verdict": "reject",
            "stage_a": stage_a,
            "stage_b": {"status": "not_run", "reason": "stage_a_gate_failed"},
        }
    strategy = freeze_strategy(stage_a, plan)
    stage_b = run_stage_b(observations, plan, strategy, stage_b_runner)
    return {
        "schema_version": "gib-benchmark-1",
        "task_id": "P3-07",
        "task_status": "completed",
        "candidate_verdict": "enter_P4" if stage_b["gate"]["joint_gate_pass"] else "reject",
        "frozen_strategy": strategy,
        "stage_a": stage_a,
        "stage_b": stage_b,
    }


def _raw_summary_features(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1 or not np.all(np.isfinite(array)):
        raise AdaptiveSamplingError("raw waveform must be a finite channel-by-time matrix")
    return np.concatenate(
        [
            np.mean(array, axis=1),
            np.std(array, axis=1),
            np.min(array, axis=1),
            np.max(array, axis=1),
        ]
    )


def _sampling_features(
    raw_waveform: np.ndarray,
    slow_channels: np.ndarray,
    calibration_channels: np.ndarray,
    active_modalities: Sequence[str],
    sample_points: int,
) -> np.ndarray:
    """Build fixed-width deployable features from only acquired modalities."""

    raw = np.asarray(raw_waveform, dtype=np.float64)
    slow = np.asarray(slow_channels, dtype=np.float64)
    calibration = np.asarray(calibration_channels, dtype=np.float64).reshape(-1)
    if raw.ndim != 2 or raw.shape[0] != 8 or slow.ndim != 2:
        raise AdaptiveSamplingError("sampling features require 8 Raw channels and 2-D slow channels")
    if sample_points < 1 or sample_points > raw.shape[1]:
        raise AdaptiveSamplingError("sampling checkpoint exceeds the Raw time axis")
    active = set(active_modalities)
    unknown = active - {*RAW_MODALITY_CHANNELS, "slow", "calibration"}
    if unknown:
        raise AdaptiveSamplingError(f"unknown sampling modalities: {sorted(unknown)}")
    blocks: list[np.ndarray] = []
    for modality, channel_slice in RAW_MODALITY_CHANNELS.items():
        width = channel_slice.stop - channel_slice.start
        if modality in active:
            block = _raw_summary_features(raw[channel_slice, :sample_points])
        else:
            block = np.zeros(width * 4, dtype=np.float64)
        blocks.append(block)
    blocks.append(np.mean(slow, axis=1) if "slow" in active else np.zeros(slow.shape[0], dtype=np.float64))
    blocks.append(calibration if "calibration" in active else np.zeros_like(calibration))
    return np.concatenate(blocks)


def _sampling_checkpoints(metadata: Any, index: int) -> tuple[SamplingCheckpoint, ...]:
    raw = _array(metadata, index, "raw_waveform")
    incremental = _array(metadata, index, "incremental_information")
    full_crb = np.maximum(_vector(_array(metadata, index, "crb_p90"), "full CRB"), np.finfo(float).eps)
    increments = len(incremental)
    if increments < 1:
        raise AdaptiveSamplingError("incremental information must contain at least one modality")
    traces = np.maximum(np.asarray(incremental["delta_I_trace"], dtype=np.float64), 0.0)
    total_trace = float(np.sum(traces))
    if total_trace > 0.0:
        fractions = np.cumsum(traces) / total_trace
    else:
        fractions = np.arange(1, increments + 1, dtype=np.float64) / increments
    fractions[-1] = 1.0
    fractions = np.maximum.accumulate(fractions)
    costs = np.maximum(np.asarray(incremental["delta_cost"], dtype=np.float64), 0.0)
    checkpoints: list[SamplingCheckpoint] = []
    previous_points = 0
    previous_time = 0.0
    active: list[str] = []
    for position in range(increments):
        modality = str(incremental["increment_type"][position])
        if modality in active:
            raise AdaptiveSamplingError(f"incremental modality is repeated: {modality}")
        active.append(modality)
        target_points = int(math.ceil(raw.shape[-1] * (position + 1) / increments))
        sample_points = max(previous_points + 1, min(target_points, raw.shape[-1]))
        cumulative_time = float(np.sum(costs[: position + 1])) + (position + 1) * 1.0e-6
        measurement_time = max(previous_time + 1.0e-6, cumulative_time)
        fraction = float(fractions[position])
        estimated_crb = full_crb / math.sqrt(max(fraction, np.finfo(float).eps))
        checkpoints.append(
            SamplingCheckpoint(
                sample_points=sample_points,
                measurement_time_ms=measurement_time,
                active_modalities=tuple(active),
                estimated_crb_p90=tuple(float(item) for item in estimated_crb),
                uncertainty=tuple(float(item) for item in estimated_crb),
                delta_i=float(traces[position]),
                delta_i_per_cost=float(incremental["delta_I_per_delta_cost"][position]),
                cumulative_information_fraction=fraction,
                flops=float(sample_points * len(active)),
            )
        )
        previous_points = sample_points
        previous_time = measurement_time
    return tuple(checkpoints)


def _fit_sampling_reference_models(metadata: Any, baseline_plan: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[tuple[str, int], Any]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    alpha = float(baseline_plan["models"]["ridge"]["alpha"])
    raw_features = np.asarray([
        _sampling_features(
            _array(metadata, index, "raw_waveform"),
            _array(metadata, index, "slow_channels"),
            _array(metadata, index, "calibration_channels"),
            ("ndir", "acoustic_raw", "thermal", "slow", "calibration"),
            _array(metadata, index, "raw_waveform").shape[1],
        )
        for index in range(len(metadata.records))
    ], dtype=np.float64)
    labels = _labels(metadata, np.arange(len(metadata.records), dtype=int))
    models: dict[tuple[str, int], Any] = {}
    for split_id in plan["split_ids"]:
        train = _partition_indices(metadata, str(split_id))["train"]
        for seed in plan["seeds"]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(raw_features[train], labels[train])
            models[(str(split_id), int(seed))] = model
    return models


def _pilot_sampling_observations(metadata: Any, plan: Mapping[str, Any], models: Mapping[tuple[str, int], Any]) -> list[SamplingObservation]:
    observations: list[SamplingObservation] = []
    for split_id in plan["split_ids"]:
        test = _partition_indices(metadata, str(split_id))["test"]
        for seed in plan["seeds"]:
            model = models[(str(split_id), int(seed))]
            for index in test:
                position = int(index)
                record = metadata.records[position]
                raw = np.asarray(_array(metadata, position, "raw_waveform"), dtype=np.float64)
                slow = np.asarray(_array(metadata, position, "slow_channels"), dtype=np.float64)
                calibration = np.asarray(_array(metadata, position, "calibration_channels"), dtype=np.float64)
                checkpoints = _sampling_checkpoints(metadata, position)
                full = checkpoints[-1]
                reference = model.predict(
                    _sampling_features(raw, slow, calibration, full.active_modalities, full.sample_points)[None, :]
                )[0]
                observations.append(
                    SamplingObservation(
                        mixture_id=str(record["mixture_id"]),
                        sequence_id=str(record["sequence_id"]),
                        grid_cell_id=str(record["grade"]["grid_cell_id"]),
                        information_band=str(record["grade"]["information_band"]),
                        split_id=str(split_id),
                        seed=int(seed),
                        truth=tuple(float(item) for item in _array(metadata, position, "labels")),
                        reference_prediction=tuple(float(item) for item in reference),
                        raw_waveform=raw,
                        checkpoints=checkpoints,
                        policy_inputs={
                            "estimated_crb_p90": list(full.estimated_crb_p90),
                            "incremental_information": [
                                {
                                    "active_modality": checkpoint.active_modalities[-1],
                                    "delta_i": checkpoint.delta_i,
                                    "delta_i_per_cost": checkpoint.delta_i_per_cost,
                                }
                                for checkpoint in checkpoints
                            ],
                            "native_measurement_cost": full.measurement_time_ms,
                        },
                        slow_channels=slow,
                        calibration_channels=calibration,
                    )
                )
    return observations


def run_c5a_from_pilot(
    plan: dict[str, Any],
    *,
    pilot_freeze: Path,
    baseline_freeze: Path,
    baseline_plan_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the frozen C5-A state machine against pilot Raw arrays."""

    validate_sampling_plan(plan)
    verify_evidence_manifest(pilot_freeze)
    verify_evidence_manifest(baseline_freeze)
    baseline_plan = json.loads(Path(baseline_plan_path).read_text(encoding="utf-8"))
    metadata = load_pilot_metadata(pilot_freeze)
    if metadata.generation_summary.get("plan_id") != "GIB-P3-PILOT-v2":
        raise AdaptiveSamplingError("pilot freeze is not the frozen P3 pilot plan")
    baseline_snapshot = Path(baseline_freeze) / "source_snapshots" / "gas_information_bench" / "configs" / "p3_baseline_plan.json"
    if not baseline_snapshot.is_file():
        raise AdaptiveSamplingError("baseline freeze is missing p3_baseline_plan.json snapshot")
    frozen_baseline_plan = json.loads(baseline_snapshot.read_text(encoding="utf-8"))
    if frozen_baseline_plan != baseline_plan:
        raise AdaptiveSamplingError("working baseline plan differs from the frozen G3-3 baseline input")
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"attempt directory already exists: {target}")
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        models = _fit_sampling_reference_models(metadata, baseline_plan, plan)
        observations = _pilot_sampling_observations(metadata, plan, models)

        def predictor(observation: SamplingObservation, checkpoint: SamplingCheckpoint, method: str) -> Sequence[float]:
            del method
            model = models[(observation.split_id, int(observation.seed))]
            if observation.slow_channels is None or observation.calibration_channels is None:
                raise AdaptiveSamplingError("pilot sampling observation is missing deployable modality arrays")
            features = _sampling_features(
                observation.raw_waveform,
                observation.slow_channels,
                observation.calibration_channels,
                checkpoint.active_modalities,
                checkpoint.sample_points,
            )
            return model.predict(features[None, :])[0]

        result = run_c5a(observations, plan, predictor, predictor)
        result["provenance"] = {
            "pilot_freeze_id": Path(pilot_freeze).name,
            "pilot_evidence_manifest_sha256": sha256_file(Path(pilot_freeze) / "evidence_manifest.json"),
            "baseline_freeze_id": Path(baseline_freeze).name,
            "baseline_evidence_manifest_sha256": sha256_file(Path(baseline_freeze) / "evidence_manifest.json"),
            "sampling_code_sha256": sha256_file(Path(__file__)),
            "sampling_plan_sha256": sha256_file(Path(plan["_plan_path"])),
            "baseline_plan_sha256": sha256_file(Path(baseline_plan_path)),
            "dataset_manifest_id": metadata.generation_summary["dataset_manifest_id"],
        }
        result["observation_count"] = len(observations)
        result["next_allowed_task"] = "P3-13"
        atomic_write_json(staging / "sampling_results.json", result)
        atomic_write_json(
            staging / "attempt_manifest.json",
            {
                "schema_version": "gib-benchmark-1",
                "attempt_id": target.name,
                "task_id": "P3-07",
                "status": "complete",
                "task_status": "completed",
                "candidate_verdict": result["candidate_verdict"],
                "claim_scope": plan["claim_scope"],
                "observation_count": len(observations),
                "next_allowed_task": "P3-13",
            },
        )
        atomic_promote_directory(staging, target)
        return result
    except Exception:
        remove_owned_staging(staging)
        raise


__all__ = [
    "AdaptiveSamplingError",
    "SamplingCheckpoint",
    "SamplingObservation",
    "freeze_strategy",
    "run_c5a",
    "run_c5a_from_pilot",
    "run_stage_b",
    "select_checkpoint",
    "simulate_stage_a",
    "validate_sampling_plan",
]
