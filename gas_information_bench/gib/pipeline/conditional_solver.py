"""P3-10 and P3-12 conditional solver evaluations.

This module orchestrates conditional comparisons but delegates every numerical
solve and paired bootstrap to the existing ``gib.audit.solver`` owner.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from ..audit.solver import (
    SolverOptions,
    _failure_row,
    _options_from_config,
    _result_row,
    paired_group_bootstrap,
    solve_classical_vp,
    solve_tsvd_ridge_vp,
    solve_vplr,
)
from ..common.io import atomic_promote_directory, atomic_write_json, remove_owned_staging, sha256_file
from ..contract import validate_solver_row
from ..freeze import verify_evidence_manifest
from .solver_preflight import _runtime_metadata, build_solver_trials


COMPONENTS = ("N2", "CO2", "O2", "Ar")
SOLVER_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "classical_vp": solve_classical_vp,
    "vplr": solve_vplr,
    "tsvd_ridge_vp": solve_tsvd_ridge_vp,
}


class ConditionalSolverError(ValueError):
    """Raised when a conditional P3 contract is invalid."""


def _load_activation(activation_freeze: Path, task_id: str, expected_freeze_id: str) -> dict[str, Any]:
    verify_evidence_manifest(activation_freeze)
    if Path(activation_freeze).name != expected_freeze_id:
        raise ConditionalSolverError("activation freeze differs from the frozen conditional plan")
    result_path = Path(activation_freeze) / "attempt" / "solver_preflight_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    activated = bool(result.get("activated_tasks", {}).get(task_id))
    return {
        "freeze_id": Path(activation_freeze).name,
        "task_id": task_id,
        "activated": activated,
        "c2_preflight": result.get("c2_preflight"),
        "method_verdicts": result.get("method_verdicts", []),
    }


def _validate_common_plan(config: Mapping[str, Any], task_id: str) -> None:
    if config.get("schema_version") != "gib-benchmark-1":
        raise ConditionalSolverError("conditional plan schema_version mismatch")
    if config.get("task_id") != task_id or config.get("plan_status") != "frozen_before_fit":
        raise ConditionalSolverError("conditional plan is not frozen for this task")
    if config.get("statistics", {}).get("resampling_unit") != "mixture_id":
        raise ConditionalSolverError("conditional bootstrap must resample mixture_id")
    statistics = config.get("statistics", {})
    if statistics.get("bootstrap_resamples") != 10000 or statistics.get("bootstrap_seed") != 20260824:
        raise ConditionalSolverError("conditional bootstrap profile drift")
    if config.get("reference_method") != "vplr":
        raise ConditionalSolverError("conditional reference solver must remain VPLR")


def _angle_band(grid_cell_id: str) -> str:
    value = str(grid_cell_id).rsplit("-", 1)[-1]
    if value not in {"HIG", "MED", "LOW"}:
        raise ConditionalSolverError(f"unknown angle band in grid cell: {grid_cell_id}")
    return value


def _conditioned_options(
    base: SolverOptions,
    config: Mapping[str, Any],
    information_band: str,
    angle_band: str,
    *,
    use_information: bool,
    fixed_regularization: bool = False,
) -> SolverOptions:
    if not use_information:
        return base
    bands = config["conditioning"]["information_band"]
    if information_band not in bands:
        raise ConditionalSolverError(f"unknown information band: {information_band}")
    specification = bands[information_band]
    maximum = int(specification["max_iterations"]) + int(config["conditioning"]["angle_penalty_iterations"][angle_band])
    maximum = min(maximum, int(config["conditioning"]["maximum_iterations"]))
    ridge_alpha = base.ridge_alpha if fixed_regularization else float(specification["ridge_alpha"])
    return replace(base, max_iterations=maximum, ridge_alpha=ridge_alpha)


def _row_for_result(
    trial: Any,
    result: Any,
    method_id: str,
    runtime_metadata: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    row = _result_row(trial, result, COMPONENTS, 0, runtime_metadata)
    row["method_id"] = method_id
    row["execution_mode"] = mode
    validate_solver_row(row)
    return row


def _failed_row(trial: Any, method_id: str, error: Exception, runtime_metadata: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    row = _failure_row(trial, method_id, 0, error, 0, runtime_metadata)
    row["execution_mode"] = mode
    return row


def _gate(summary: Mapping[str, Any], gates: Mapping[str, Any], *, require_wall_clock: bool) -> dict[str, Any]:
    bands = gates["non_inferiority_bands"]
    ni = all(
        float(summary["precision_p90_difference"][index]["ci"][1]) <= float(bands[component])
        for index, component in enumerate(COMPONENTS)
    )
    costs = summary["cost_relative_reduction"]
    e30 = any(float(costs[name]["ci"][0]) >= float(gates.get("e30_minimum_reduction", 1.0)) for name in ("iterations", "forward_calls"))
    e20 = float(costs["solver_wall_clock"]["ci"][0]) >= float(gates["e20_minimum_reduction"])
    nr5_floor = -float(gates["nr5_maximum_regression"])
    nr5 = all(float(costs[name]["ci"][0]) >= nr5_floor for name in costs)
    nonconvergence = float(summary["nonconvergence_rate_difference"]["ci"][1]) <= float(gates["nr5_maximum_regression"])
    efficiency = e20 if require_wall_clock else (e30 or e20)
    return {"ni": ni, "e30": e30, "e20": e20, "nr5": nr5 and nonconvergence, "passes": ni and efficiency and nr5 and nonconvergence}


def _paired_summaries(rows: Sequence[Mapping[str, Any]], methods: Sequence[str], reference: str, config: Mapping[str, Any]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for method in methods:
        if method == reference:
            continue
        summary = paired_group_bootstrap(
            rows,
            method,
            component_count=len(COMPONENTS),
            resamples=int(config["statistics"]["bootstrap_resamples"]),
            seed=int(config["statistics"]["bootstrap_seed"]),
            confidence=float(config["statistics"]["confidence_level"]),
            reference_method=reference,
        )
        summaries[method] = {"bootstrap": summary, "gates": _gate(summary, config["gates"], require_wall_clock=False)}
    return summaries


def _cell_gate(rows: Sequence[Mapping[str, Any]], method: str, reference: str, config: Mapping[str, Any]) -> dict[str, Any]:
    cells = {}
    for cell in sorted({str(row["grid_cell_id"]) for row in rows}):
        local = [row for row in rows if str(row["grid_cell_id"]) == cell]
        summary = paired_group_bootstrap(
            local,
            method,
            component_count=len(COMPONENTS),
            resamples=int(config["statistics"]["bootstrap_resamples"]),
            seed=int(config["statistics"]["bootstrap_seed"]),
            confidence=float(config["statistics"]["confidence_level"]),
            reference_method=reference,
        )
        cells[cell] = {"bootstrap": summary, "gates": _gate(summary, config["gates"], require_wall_clock=False)}
    return cells


def _equivalent(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> bool:
    identity = ("grid_cell_id", "split_id", "seed", "mixture_id", "sequence_id", "repeat_index")
    left_map = {tuple(row[field] for field in identity): row for row in rows if row["method_id"] == left}
    right_map = {tuple(row[field] for field in identity): row for row in rows if row["method_id"] == right}
    if set(left_map) != set(right_map) or not left_map:
        return False
    fields = ("component_abs_errors", "iterations", "forward_calls", "convergence")
    return all(left_map[key].get(field) == right_map[key].get(field) for key in left_map for field in fields)


def _learn_update_scales(
    training_trials: Sequence[Any],
    base_options: SolverOptions,
    config: Mapping[str, Any],
) -> dict[tuple[str, int], float]:
    """Select one small update-scale parameter on each split's train partition."""

    candidates = [float(value) for value in config["learned_update"]["scale_candidates"]]
    if not candidates or any(value <= 0.0 for value in candidates):
        raise ConditionalSolverError("learned update scales must be positive")
    selected: dict[tuple[str, int], float] = {}
    identities = sorted({(str(trial.split_id), int(trial.seed)) for trial in training_trials})
    for identity in identities:
        local = [trial for trial in training_trials if (str(trial.split_id), int(trial.seed)) == identity]
        scores: list[tuple[float, float, float]] = []
        for scale in candidates:
            errors: list[float] = []
            iterations: list[int] = []
            for trial in local:
                options = _conditioned_options(
                    base_options,
                    config,
                    trial.information_band,
                    _angle_band(trial.grid_cell_id),
                    use_information=True,
                )
                result = solve_vplr(trial.problem, replace(options, update_scale=scale))
                errors.append(float(np.mean(np.abs(result.linear_parameters - trial.truth_linear))))
                iterations.append(int(result.iterations))
            scores.append((float(np.mean(errors)), float(np.mean(iterations)), scale))
        selected[identity] = min(scores)[2]
    return selected


def _write_attempt(target: Path, result: Mapping[str, Any], task_id: str, candidate_verdict: str) -> None:
    staging = target.parent / f".{target.name}.staging-{uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        atomic_write_json(staging / "conditional_solver_results.json", result)
        atomic_write_json(
            staging / "attempt_manifest.json",
            {
                "schema_version": "gib-benchmark-1",
                "attempt_id": target.name,
                "task_id": task_id,
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


def run_ic_rdu_vp(
    config: dict[str, Any],
    solver_plan: Mapping[str, Any],
    *,
    activation_freeze: Path,
    pilot_freeze: Path,
    execution_registry: Mapping[str, Any],
    git_commit: str,
    output_dir: Path,
) -> dict[str, Any]:
    _validate_common_plan(config, "P3-10")
    activation = _load_activation(
        activation_freeze,
        "P3-10",
        str(config["activation_input"]["freeze_id"]),
    )
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"attempt directory already exists: {target}")
    if not activation["activated"]:
        result = {
            "schema_version": "gib-benchmark-1",
            "task_id": "P3-10",
            "task_status": "completed",
            "candidate_verdict": "not_activated",
            "activation": activation,
            "claim_scope": config["claim_scope"],
            "next_allowed_task": "P3-13",
        }
        _write_attempt(target, result, "P3-10", "not_activated")
        return result
    verify_evidence_manifest(pilot_freeze)
    base_options = _options_from_config(solver_plan["solver_budget"])
    trials = build_solver_trials(pilot_freeze, solver_plan)
    training_trials = build_solver_trials(
        pilot_freeze,
        solver_plan,
        partition="train",
        mixtures_per_cell=int(config["learned_update"]["training_mixtures_per_cell"]),
    )
    learned_update_scales = _learn_update_scales(training_trials, base_options, config)
    runtime_metadata = _runtime_metadata(execution_registry, git_commit)
    rows: list[dict[str, Any]] = []
    for trial in trials:
        angle = _angle_band(trial.grid_cell_id)
        conditioned = _conditioned_options(base_options, config, trial.information_band, angle, use_information=True)
        learned_scale = learned_update_scales[(str(trial.split_id), int(trial.seed))]
        option_map = {
            "classical_vp": base_options,
            "vplr": base_options,
            "tsvd_ridge_vp": base_options,
            "fixed_depth_vp": replace(base_options, max_iterations=int(config["conditioning"]["maximum_iterations"])),
            "ic_rdu_vp": replace(conditioned, update_scale=learned_scale),
            "ablation_without_information_conditioning": base_options,
            "ablation_fixed_regularization": _conditioned_options(base_options, config, trial.information_band, angle, use_information=True, fixed_regularization=True),
            "ablation_fixed_update": replace(conditioned, update_scale=1.0),
        }
        functions = {
            "classical_vp": solve_classical_vp,
            "vplr": solve_vplr,
            "tsvd_ridge_vp": solve_tsvd_ridge_vp,
            "fixed_depth_vp": solve_vplr,
            "ic_rdu_vp": solve_vplr,
            "ablation_without_information_conditioning": solve_vplr,
            "ablation_fixed_regularization": solve_vplr,
            "ablation_fixed_update": solve_vplr,
        }
        for method_id in config["comparison_methods"] + ["ablation_without_information_conditioning", "ablation_fixed_regularization", "ablation_fixed_update"]:
            try:
                result = functions[method_id](trial.problem, option_map[method_id])
                rows.append(_row_for_result(trial, result, method_id, runtime_metadata, mode=method_id))
            except Exception as exc:
                rows.append(_failed_row(trial, method_id, exc, runtime_metadata, mode=method_id))

    methods = config["comparison_methods"] + ["ablation_without_information_conditioning", "ablation_fixed_regularization", "ablation_fixed_update"]
    failed = [row for row in rows if row.get("status") != "complete"]
    summaries = {} if failed else _paired_summaries(rows, methods, config["reference_method"], config)
    cell_summaries = {} if failed else _cell_gate(rows, "ic_rdu_vp", config["reference_method"], config)
    candidate_gate = summaries.get("ic_rdu_vp", {}).get("gates", {"passes": False, "reason": "failed rows retained"})
    equivalent_ablations = [
        method for method in ("ablation_without_information_conditioning", "ablation_fixed_regularization", "ablation_fixed_update", "fixed_depth_vp")
        if not failed and _equivalent(rows, "ic_rdu_vp", method)
    ]
    shuffled_labels = np.random.default_rng(int(config["negative_control_seed"])).permutation(len(trials))
    shuffled_rows: list[dict[str, Any]] = []
    for trial_index, trial in enumerate(trials):
        shuffled_trial = trials[int(shuffled_labels[trial_index])]
        angle = _angle_band(shuffled_trial.grid_cell_id)
        options = _conditioned_options(base_options, config, shuffled_trial.information_band, angle, use_information=True)
        options = replace(options, update_scale=learned_update_scales[(str(trial.split_id), int(trial.seed))])
        try:
            result = solve_vplr(trial.problem, options)
            shuffled_rows.append(_row_for_result(trial, result, "negative_control_cell_label_shuffle", runtime_metadata, mode="cell_label_shuffle"))
        except Exception as exc:
            shuffled_rows.append(_failed_row(trial, "negative_control_cell_label_shuffle", exc, runtime_metadata, mode="cell_label_shuffle"))
    rows.extend(shuffled_rows)
    controls = {} if failed else _paired_summaries(rows, ["negative_control_cell_label_shuffle"], config["reference_method"], config)
    control_pass = controls.get("negative_control_cell_label_shuffle", {}).get("gates", {}).get("passes", False)
    cell_pass_count = sum(item["gates"]["passes"] for item in cell_summaries.values())
    candidate_verdict = "enter_P4" if candidate_gate.get("passes") and cell_pass_count >= 2 and not equivalent_ablations and not control_pass else "reject"
    result = {
        "schema_version": "gib-benchmark-1",
        "task_id": "P3-10",
        "task_status": "completed",
        "candidate_verdict": candidate_verdict,
        "activation": activation,
        "coverage": {"trial_count": len(trials), "grid_split_seed_count": len({(t.grid_cell_id, t.split_id, t.seed) for t in trials})},
        "reference_method": config["reference_method"],
        "method_summaries": summaries,
        "cell_summaries": cell_summaries,
        "negative_controls": controls,
        "equivalent_core_ablations": equivalent_ablations,
        "learned_update_scales": [
            {"split_id": split_id, "seed": seed, "update_scale": scale}
            for (split_id, seed), scale in sorted(learned_update_scales.items())
        ],
        "failed_rows": failed,
        "solver_rows": rows,
        "claim_scope": config["claim_scope"],
        "next_allowed_task": "P3-13",
        "provenance": {"activation_freeze_id": Path(activation_freeze).name, "pilot_freeze_id": Path(pilot_freeze).name, "solver_plan_sha256": sha256_file(Path(solver_plan["_plan_path"])), "conditional_plan_sha256": sha256_file(Path(config["_plan_path"]))},
    }
    _write_attempt(target, result, "P3-10", candidate_verdict)
    return result


def _rule_for_trial(trial: Any, config: Mapping[str, Any], *, shuffled_label: tuple[str, str] | None = None) -> dict[str, Any]:
    information = trial.information_band if shuffled_label is None else shuffled_label[0]
    angle = _angle_band(trial.grid_cell_id) if shuffled_label is None else shuffled_label[1]
    matches = [
        rule for rule in config["route_rules"]
        if information in rule["information_bands"] and angle in rule["angle_bands"]
    ]
    if len(matches) != 1:
        raise ConditionalSolverError(f"physical route rule is not unique for {information}/{angle}")
    return dict(matches[0])


def run_figs(
    config: dict[str, Any],
    solver_plan: Mapping[str, Any],
    *,
    activation_freeze: Path,
    pilot_freeze: Path,
    execution_registry: Mapping[str, Any],
    git_commit: str,
    output_dir: Path,
) -> dict[str, Any]:
    _validate_common_plan(config, "P3-12")
    activation = _load_activation(
        activation_freeze,
        "P3-12",
        str(config["activation_input"]["freeze_id"]),
    )
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"attempt directory already exists: {target}")
    if not activation["activated"]:
        result = {"schema_version": "gib-benchmark-1", "task_id": "P3-12", "task_status": "completed", "candidate_verdict": "not_activated", "activation": activation, "claim_scope": config["claim_scope"], "next_allowed_task": "P3-13"}
        _write_attempt(target, result, "P3-12", "not_activated")
        return result
    verify_evidence_manifest(pilot_freeze)
    options = _options_from_config(solver_plan["solver_budget"])
    trials = build_solver_trials(pilot_freeze, solver_plan)
    runtime_metadata = _runtime_metadata(execution_registry, git_commit)
    base_rows: list[dict[str, Any]] = []
    for trial in trials:
        for method_id, function in SOLVER_FUNCTIONS.items():
            try:
                result = function(trial.problem, options)
                base_rows.append(_row_for_result(trial, result, method_id, runtime_metadata, mode=method_id))
            except Exception as exc:
                base_rows.append(_failed_row(trial, method_id, exc, runtime_metadata, mode=method_id))
    failed = [row for row in base_rows if row.get("status") != "complete"]
    by_identity = {
        (row["grid_cell_id"], row["split_id"], row["seed"], row["mixture_id"], row["sequence_id"], row["repeat_index"], row["method_id"]): row
        for row in base_rows
    }
    route_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for trial in trials:
        identity = (trial.grid_cell_id, trial.split_id, trial.seed, trial.mixture_id, trial.sequence_id, 0)
        rule = _rule_for_trial(trial, config)
        selected = by_identity[(*identity, rule["method_id"])]
        routed = dict(selected)
        routed["method_id"] = "physical_rule_route"
        routed["route_rule_id"] = rule["rule_id"]
        routed["route_method_id"] = rule["method_id"]
        validate_solver_row(routed)
        route_rows.append(routed)
        candidates = [by_identity[(*identity, method)] for method in SOLVER_FUNCTIONS]
        oracle = min(candidates, key=lambda row: sum(row.get("component_abs_errors", [float("inf")])) )
        oracle_row = dict(oracle)
        oracle_row["method_id"] = "oracle_region_route"
        oracle_row["route_method_id"] = oracle["method_id"]
        validate_solver_row(oracle_row)
        oracle_rows.append(oracle_row)
    shuffled = np.random.default_rng(int(config["negative_control_seed"])).permutation(len(trials))
    shuffled_rows: list[dict[str, Any]] = []
    for index, trial in enumerate(trials):
        label_trial = trials[int(shuffled[index])]
        rule = _rule_for_trial(trial, config, shuffled_label=(label_trial.information_band, _angle_band(label_trial.grid_cell_id)))
        identity = (trial.grid_cell_id, trial.split_id, trial.seed, trial.mixture_id, trial.sequence_id, 0)
        row = dict(by_identity[(*identity, rule["method_id"])])
        row["method_id"] = "negative_control_region_label_shuffle"
        row["route_rule_id"] = rule["rule_id"]
        row["route_method_id"] = rule["method_id"]
        validate_solver_row(row)
        shuffled_rows.append(row)
    fixed_rows = []
    for trial in trials:
        identity = (trial.grid_cell_id, trial.split_id, trial.seed, trial.mixture_id, trial.sequence_id, 0)
        row = dict(by_identity[(*identity, config["reference_method"])])
        row["method_id"] = "fixed_strongest_solver_control"
        validate_solver_row(row)
        fixed_rows.append(row)
    all_rows = base_rows + route_rows + oracle_rows + shuffled_rows + fixed_rows
    methods = ["physical_rule_route", "oracle_region_route", "negative_control_region_label_shuffle", "fixed_strongest_solver_control"]
    summaries = {} if failed else _paired_summaries(all_rows, methods, config["reference_method"], config)
    route_gate = summaries.get("physical_rule_route", {}).get("gates", {"passes": False})
    region_coverage: dict[str, Any] = {}
    for rule in config["route_rules"]:
        local = [row for row in route_rows if row.get("route_rule_id") == rule["rule_id"]]
        region_coverage[rule["rule_id"]] = {"rows": len(local), "splits": sorted({row["split_id"] for row in local}), "seeds": sorted({row["seed"] for row in local}), "sufficient": len({row["split_id"] for row in local}) >= int(config["minimum_region_splits"]) and set(row["seed"] for row in local) >= set(config["required_seeds"])}
    coverage_ok = all(item["sufficient"] for item in region_coverage.values())
    shuffle_pass = summaries.get("negative_control_region_label_shuffle", {}).get("gates", {}).get("passes", False)
    if not coverage_ok:
        candidate_verdict = "inconclusive"
    else:
        candidate_verdict = "enter_P4" if route_gate.get("passes") and not shuffle_pass else "reject"
    result = {
        "schema_version": "gib-benchmark-1",
        "task_id": "P3-12",
        "task_status": "completed",
        "candidate_verdict": candidate_verdict,
        "activation": activation,
        "coverage": {"trial_count": len(trials), "grid_split_seed_count": len({(t.grid_cell_id, t.split_id, t.seed) for t in trials})},
        "reference_method": config["reference_method"],
        "method_summaries": summaries,
        "region_coverage": region_coverage,
        "failed_rows": failed,
        "solver_rows": all_rows,
        "claim_scope": config["claim_scope"],
        "next_allowed_task": "P3-13",
        "provenance": {"activation_freeze_id": Path(activation_freeze).name, "pilot_freeze_id": Path(pilot_freeze).name, "solver_plan_sha256": sha256_file(Path(solver_plan["_plan_path"])), "conditional_plan_sha256": sha256_file(Path(config["_plan_path"]))},
    }
    _write_attempt(target, result, "P3-12", candidate_verdict)
    return result


__all__ = ["ConditionalSolverError", "run_figs", "run_ic_rdu_vp"]
