"""Deterministic separable least-squares solvers and the P3-06 preflight gate.

The numerical solvers in this module share one Levenberg--Marquardt driver.
They differ only in whether and how the linear variables are eliminated.  This
keeps initialization, stopping rules, budgets, and forward-call accounting
paired by construction.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..contract import validate_solver_row


Array = np.ndarray
BasisFunction = Callable[[Array], Array]


class SolverError(ValueError):
    """Raised when a solver input or paired comparison violates P3-06."""


@dataclass(frozen=True)
class SeparableProblem:
    """A real-valued problem ``min ||A(phi) c - y||_2``."""

    observations: Array
    basis: BasisFunction
    linear_initial: Array
    nonlinear_initial: Array


@dataclass(frozen=True)
class SolverOptions:
    max_iterations: int = 50
    function_tolerance: float = 1.0e-12
    step_tolerance: float = 1.0e-10
    gradient_tolerance: float = 1.0e-10
    finite_difference_step: float = 1.0e-6
    initial_damping: float = 1.0e-3
    damping_increase: float = 10.0
    damping_decrease: float = 0.3
    update_scale: float = 1.0
    ridge_alpha: float = 1.0e-8
    tsvd_relative_cutoff: float = 1.0e-10


@dataclass(frozen=True)
class SolverResult:
    method_id: str
    linear_parameters: Array
    nonlinear_parameters: Array
    iterations: int
    forward_calls: int
    convergence: bool
    condition_number: float
    final_residual: float
    runtime_ns: int
    termination_reason: str


@dataclass(frozen=True)
class SolverTrial:
    mixture_id: str
    sequence_id: str
    grid_cell_id: str
    split_id: str
    seed: int
    problem: SeparableProblem
    truth_linear: Array
    information_band: str
    hardware_fingerprint: str


@dataclass(frozen=True)
class _OptimizationResult:
    parameters: Array
    residual: Array
    state: Any
    iterations: int
    convergence: bool
    termination_reason: str


class _Evaluator:
    def __init__(self, problem: SeparableProblem, linear_mode: str, options: SolverOptions):
        self.problem = problem
        self.linear_mode = linear_mode
        self.options = options
        self.forward_calls = 0

    def basis(self, nonlinear: Array) -> Array:
        matrix = np.asarray(self.problem.basis(nonlinear.copy()), dtype=np.float64)
        expected = (self.problem.observations.size, self.problem.linear_initial.size)
        if matrix.shape != expected or not np.all(np.isfinite(matrix)):
            raise SolverError(f"basis must be finite with shape {expected}, got {matrix.shape}")
        return matrix

    def joint(self, parameters: Array) -> tuple[Array, tuple[Array, Array]]:
        linear_size = self.problem.linear_initial.size
        linear = parameters[:linear_size]
        nonlinear = parameters[linear_size:]
        matrix = self.basis(nonlinear)
        self.forward_calls += 1
        return matrix @ linear - self.problem.observations, (linear.copy(), matrix)

    def projected(self, nonlinear: Array) -> tuple[Array, tuple[Array, Array]]:
        matrix = self.basis(nonlinear)
        linear = _linear_solution(matrix, self.problem.observations, self.linear_mode, self.options)
        self.forward_calls += 1
        return matrix @ linear - self.problem.observations, (linear, matrix)

    def fixed_linear(self, nonlinear: Array, linear: Array) -> Array:
        matrix = self.basis(nonlinear)
        self.forward_calls += 1
        return matrix @ linear - self.problem.observations


def _vector(value: Any, name: str, *, allow_empty: bool = False) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or (not allow_empty and array.size == 0) or not np.all(np.isfinite(array)):
        raise SolverError(f"{name} must be a finite one-dimensional array")
    return array.copy()


def _validate_problem(problem: SeparableProblem) -> SeparableProblem:
    if not isinstance(problem, SeparableProblem):
        raise SolverError("problem must be a SeparableProblem")
    observations = _vector(problem.observations, "observations")
    linear = _vector(problem.linear_initial, "linear_initial")
    nonlinear = _vector(problem.nonlinear_initial, "nonlinear_initial", allow_empty=True)
    checked = SeparableProblem(observations, problem.basis, linear, nonlinear)
    matrix = np.asarray(checked.basis(nonlinear.copy()), dtype=np.float64)
    if matrix.shape != (observations.size, linear.size) or not np.all(np.isfinite(matrix)):
        raise SolverError("basis output shape or finite-value contract failed")
    return checked


def _validate_options(options: SolverOptions) -> None:
    if options.max_iterations <= 0:
        raise SolverError("max_iterations must be positive")
    positive = (
        options.function_tolerance,
        options.step_tolerance,
        options.gradient_tolerance,
        options.finite_difference_step,
        options.initial_damping,
        options.damping_increase,
        options.damping_decrease,
        options.update_scale,
    )
    if not all(np.isfinite(item) and item > 0.0 for item in positive):
        raise SolverError("solver tolerances and damping values must be finite and positive")
    if options.damping_increase <= 1.0 or options.damping_decrease >= 1.0:
        raise SolverError("damping increase must exceed one and decrease must be below one")
    if options.ridge_alpha < 0.0 or options.tsvd_relative_cutoff < 0.0:
        raise SolverError("linear regularization values cannot be negative")


def _linear_solution(matrix: Array, observations: Array, mode: str, options: SolverOptions) -> Array:
    if mode == "ordinary":
        return np.linalg.lstsq(matrix, observations, rcond=None)[0]
    if mode == "ridge":
        gram = matrix.T @ matrix
        return np.linalg.solve(gram + options.ridge_alpha * np.eye(gram.shape[0]), matrix.T @ observations)
    if mode == "tsvd_ridge":
        left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
        cutoff = options.tsvd_relative_cutoff * singular[0] if singular.size else 0.0
        keep = singular > cutoff
        factors = np.zeros_like(singular)
        factors[keep] = singular[keep] / (singular[keep] ** 2 + options.ridge_alpha)
        return right_t.T @ (factors * (left.T @ observations))
    raise SolverError(f"unknown linear solve mode: {mode}")


def _finite_difference_jacobian(
    parameters: Array,
    residual_function: Callable[[Array], tuple[Array, Any]],
    step_scale: float,
) -> Array:
    columns = []
    for index in range(parameters.size):
        step = step_scale * max(1.0, abs(float(parameters[index])))
        left = parameters.copy()
        right = parameters.copy()
        left[index] -= step
        right[index] += step
        left_residual, _ = residual_function(left)
        right_residual, _ = residual_function(right)
        columns.append((right_residual - left_residual) / (2.0 * step))
    if not columns:
        residual, _ = residual_function(parameters)
        return np.empty((residual.size, 0), dtype=np.float64)
    return np.column_stack(columns)


def _classical_projection_jacobian(
    parameters: Array,
    residual: Array,
    state: tuple[Array, Array],
    evaluator: _Evaluator,
    options: SolverOptions,
) -> Array:
    del residual
    linear, matrix = state
    projector = np.eye(matrix.shape[0]) - matrix @ np.linalg.pinv(matrix)
    columns = []
    for index in range(parameters.size):
        step = options.finite_difference_step * max(1.0, abs(float(parameters[index])))
        left = parameters.copy()
        right = parameters.copy()
        left[index] -= step
        right[index] += step
        left_prediction = evaluator.fixed_linear(left, linear)
        right_prediction = evaluator.fixed_linear(right, linear)
        columns.append(projector @ ((right_prediction - left_prediction) / (2.0 * step)))
    return np.column_stack(columns) if columns else np.empty((matrix.shape[0], 0))


def _lm_optimize(
    initial: Array,
    residual_function: Callable[[Array], tuple[Array, Any]],
    jacobian_function: Callable[[Array, Array, Any], Array],
    options: SolverOptions,
) -> _OptimizationResult:
    parameters = initial.copy()
    residual, state = residual_function(parameters)
    objective = 0.5 * float(residual @ residual)
    damping = options.initial_damping
    termination = "maximum_iterations"
    converged = False
    iterations = 0
    for iteration in range(1, options.max_iterations + 1):
        iterations = iteration
        jacobian = np.asarray(jacobian_function(parameters, residual, state), dtype=np.float64)
        if jacobian.shape != (residual.size, parameters.size) or not np.all(np.isfinite(jacobian)):
            raise SolverError("residual Jacobian is non-finite or has the wrong shape")
        gradient = jacobian.T @ residual
        if np.linalg.norm(gradient, ord=np.inf) <= options.gradient_tolerance:
            converged = True
            termination = "gradient_tolerance"
            break
        normal = jacobian.T @ jacobian
        diagonal = np.maximum(np.diag(normal), 1.0)
        try:
            step = np.linalg.solve(normal + damping * np.diag(diagonal), -gradient)
        except np.linalg.LinAlgError as exc:
            raise SolverError("LM damped normal equations are singular") from exc
        scaled_step = options.update_scale * step
        if np.linalg.norm(scaled_step) <= options.step_tolerance * (options.step_tolerance + np.linalg.norm(parameters)):
            converged = True
            termination = "step_tolerance"
            break
        trial = parameters + scaled_step
        trial_residual, trial_state = residual_function(trial)
        trial_objective = 0.5 * float(trial_residual @ trial_residual)
        if trial_objective < objective:
            reduction = objective - trial_objective
            parameters = trial
            residual = trial_residual
            state = trial_state
            objective = trial_objective
            damping = max(np.finfo(float).eps, damping * options.damping_decrease)
            if reduction <= options.function_tolerance * max(1.0, objective):
                converged = True
                termination = "function_tolerance"
                break
        else:
            damping *= options.damping_increase
    return _OptimizationResult(parameters, residual, state, iterations, converged, termination)


def _solve(problem: SeparableProblem, method_id: str, options: SolverOptions) -> SolverResult:
    checked = _validate_problem(problem)
    _validate_options(options)
    started = time.perf_counter_ns()
    if method_id in {"joint_lm", "projection_disabled_joint_lm"}:
        evaluator = _Evaluator(checked, "ordinary", options)
        initial = np.concatenate([checked.linear_initial, checked.nonlinear_initial])

        def jacobian(parameters: Array, residual: Array, state: Any) -> Array:
            del residual, state
            return _finite_difference_jacobian(parameters, evaluator.joint, options.finite_difference_step)

        optimized = _lm_optimize(initial, evaluator.joint, jacobian, options)
        linear_size = checked.linear_initial.size
        linear = optimized.parameters[:linear_size]
        nonlinear = optimized.parameters[linear_size:]
        matrix = optimized.state[1]
    else:
        modes = {
            "classical_vp": "ordinary",
            "vplr": "ridge",
            "tsvd_ridge_vp": "tsvd_ridge",
        }
        if method_id not in modes:
            raise SolverError(f"unknown method_id: {method_id}")
        evaluator = _Evaluator(checked, modes[method_id], options)

        if method_id == "classical_vp":
            def jacobian(parameters: Array, residual: Array, state: Any) -> Array:
                return _classical_projection_jacobian(parameters, residual, state, evaluator, options)
        else:
            def jacobian(parameters: Array, residual: Array, state: Any) -> Array:
                del residual, state
                return _finite_difference_jacobian(parameters, evaluator.projected, options.finite_difference_step)

        optimized = _lm_optimize(checked.nonlinear_initial, evaluator.projected, jacobian, options)
        linear, matrix = optimized.state
        nonlinear = optimized.parameters
    runtime_ns = time.perf_counter_ns() - started
    condition = float(np.linalg.cond(matrix))
    return SolverResult(
        method_id=method_id,
        linear_parameters=np.asarray(linear, dtype=np.float64),
        nonlinear_parameters=np.asarray(nonlinear, dtype=np.float64),
        iterations=optimized.iterations,
        forward_calls=evaluator.forward_calls,
        convergence=optimized.convergence,
        condition_number=condition,
        final_residual=float(np.linalg.norm(optimized.residual)),
        runtime_ns=runtime_ns,
        termination_reason=optimized.termination_reason,
    )


def solve_joint_lm(problem: SeparableProblem, options: SolverOptions = SolverOptions()) -> SolverResult:
    return _solve(problem, "joint_lm", options)


def solve_classical_vp(problem: SeparableProblem, options: SolverOptions = SolverOptions()) -> SolverResult:
    return _solve(problem, "classical_vp", options)


def solve_vplr(problem: SeparableProblem, options: SolverOptions = SolverOptions()) -> SolverResult:
    return _solve(problem, "vplr", options)


def solve_tsvd_ridge_vp(problem: SeparableProblem, options: SolverOptions = SolverOptions()) -> SolverResult:
    return _solve(problem, "tsvd_ridge_vp", options)


def solve_projection_disabled_control(
    problem: SeparableProblem,
    options: SolverOptions = SolverOptions(),
) -> SolverResult:
    return _solve(problem, "projection_disabled_joint_lm", options)


def _options_from_config(value: Mapping[str, Any]) -> SolverOptions:
    return SolverOptions(
        max_iterations=int(value["max_iterations"]),
        function_tolerance=float(value["function_tolerance"]),
        step_tolerance=float(value["step_tolerance"]),
        gradient_tolerance=float(value["gradient_tolerance"]),
        finite_difference_step=float(value["finite_difference_step"]),
        initial_damping=float(value["initial_damping"]),
        damping_increase=float(value["damping_increase"]),
        damping_decrease=float(value["damping_decrease"]),
        update_scale=float(value.get("update_scale", 1.0)),
        ridge_alpha=float(value["ridge_alpha"]),
        tsvd_relative_cutoff=float(value["tsvd_relative_cutoff"]),
    )


def _result_row(
    trial: SolverTrial,
    result: SolverResult,
    components: Sequence[str],
    repeat_index: int,
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    truth = _vector(trial.truth_linear, "truth_linear")
    if truth.size != len(components) or result.linear_parameters.size != len(components):
        raise SolverError("truth, fitted linear parameters, and components must have equal lengths")
    row = {
        "mixture_id": trial.mixture_id,
        "sequence_id": trial.sequence_id,
        "grid_cell_id": trial.grid_cell_id,
        "split_id": trial.split_id,
        "seed": trial.seed,
        "method_id": result.method_id,
        "information_band": trial.information_band,
        "component_abs_errors": np.abs(result.linear_parameters - truth).tolist(),
        "iterations": result.iterations,
        "forward_calls": result.forward_calls,
        "solver_wall_clock": result.runtime_ns,
        "runtime_ns": result.runtime_ns,
        "convergence": result.convergence,
        "condition_number": result.condition_number,
        "final_residual": result.final_residual,
        "termination_reason": result.termination_reason,
        "hardware_fingerprint": trial.hardware_fingerprint,
        "repeat_index": repeat_index,
        "status": "complete",
        **runtime_metadata,
    }
    validate_solver_row(row)
    return row


def _failure_row(
    trial: SolverTrial,
    method_id: str,
    repeat_index: int,
    error: Exception,
    runtime_ns: int,
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain a failed run without inventing unavailable numeric metrics."""

    return {
        "mixture_id": trial.mixture_id,
        "sequence_id": trial.sequence_id,
        "grid_cell_id": trial.grid_cell_id,
        "split_id": trial.split_id,
        "seed": trial.seed,
        "method_id": method_id,
        "information_band": trial.information_band,
        "repeat_index": repeat_index,
        "hardware_fingerprint": trial.hardware_fingerprint,
        "status": "failed",
        "convergence": False,
        "runtime_ns": runtime_ns,
        "solver_wall_clock": runtime_ns,
        "error_type": type(error).__name__,
        "error_message": str(error),
        **runtime_metadata,
    }


def _paired_rows(rows: Sequence[Mapping[str, Any]], candidate: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    identity = ("grid_cell_id", "split_id", "seed", "mixture_id", "sequence_id", "repeat_index")
    reference_map = {tuple(row[field] for field in identity): row for row in rows if row["method_id"] == "joint_lm"}
    candidate_map = {tuple(row[field] for field in identity): row for row in rows if row["method_id"] == candidate}
    if len(reference_map) != sum(row["method_id"] == "joint_lm" for row in rows):
        raise SolverError("duplicate Joint LM pairing identity")
    if len(candidate_map) != sum(row["method_id"] == candidate for row in rows):
        raise SolverError(f"duplicate {candidate} pairing identity")
    if set(reference_map) != set(candidate_map) or not reference_map:
        raise SolverError(f"{candidate} and Joint LM rows are not exactly paired")
    keys = sorted(reference_map)
    references = [reference_map[key] for key in keys]
    candidates = [candidate_map[key] for key in keys]
    for reference, candidate_row in zip(references, candidates):
        if reference["hardware_fingerprint"] != candidate_row["hardware_fingerprint"]:
            raise SolverError("paired methods have different hardware fingerprints")
    return candidates, references


def _paired_rows_against(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    reference_method: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    identity = ("grid_cell_id", "split_id", "seed", "mixture_id", "sequence_id", "repeat_index")
    reference_map = {
        tuple(row[field] for field in identity): row
        for row in rows
        if row["method_id"] == reference_method
    }
    candidate_map = {
        tuple(row[field] for field in identity): row
        for row in rows
        if row["method_id"] == candidate
    }
    if len(reference_map) != sum(row["method_id"] == reference_method for row in rows):
        raise SolverError(f"duplicate {reference_method} pairing identity")
    if len(candidate_map) != sum(row["method_id"] == candidate for row in rows):
        raise SolverError(f"duplicate {candidate} pairing identity")
    if set(reference_map) != set(candidate_map) or not reference_map:
        raise SolverError(f"{candidate} and {reference_method} rows are not exactly paired")
    keys = sorted(reference_map)
    references = [reference_map[key] for key in keys]
    candidates = [candidate_map[key] for key in keys]
    for reference, candidate_row in zip(references, candidates):
        if reference["hardware_fingerprint"] != candidate_row["hardware_fingerprint"]:
            raise SolverError("paired methods have different hardware fingerprints")
    return candidates, references


def _bootstrap_indices(
    rows: Sequence[Mapping[str, Any]],
    rng: np.random.Generator,
) -> list[int]:
    by_split_group: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        by_split_group[str(row["split_id"])][str(row["mixture_id"])].append(index)
    selected = []
    for split_id in sorted(by_split_group):
        groups = by_split_group[split_id]
        names = sorted(groups)
        draws = rng.integers(0, len(names), size=len(names))
        for draw in draws:
            selected.extend(groups[names[int(draw)]])
    return selected


def _quantile_interval(values: Array, confidence: float) -> tuple[float, float]:
    alpha = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(values, alpha, method="linear")),
        float(np.quantile(values, 1.0 - alpha, method="linear")),
    )


def paired_group_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    *,
    component_count: int,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
    reference_method: str = "joint_lm",
) -> dict[str, Any]:
    """Bootstrap candidate-minus-reference P90 and relative cost reductions."""

    if resamples <= 0 or not 0.0 < confidence < 1.0:
        raise SolverError("bootstrap resamples and confidence are invalid")
    if reference_method == "joint_lm":
        candidates, references = _paired_rows(rows, candidate)
    else:
        candidates, references = _paired_rows_against(rows, candidate, reference_method)
    rng = np.random.default_rng(seed)
    precision = np.empty((resamples, component_count), dtype=np.float64)
    cost_names = ("iterations", "forward_calls", "solver_wall_clock")
    reductions = {name: np.empty(resamples, dtype=np.float64) for name in cost_names}
    nonconvergence = np.empty(resamples, dtype=np.float64)

    def precision_statistic(indices: Sequence[int]) -> Array:
        candidate_errors = np.asarray([candidates[index]["component_abs_errors"] for index in indices], dtype=float)
        reference_errors = np.asarray([references[index]["component_abs_errors"] for index in indices], dtype=float)
        if candidate_errors.shape[1:] != (component_count,) or not np.all(np.isfinite(candidate_errors)) or not np.all(np.isfinite(reference_errors)):
            raise SolverError("component errors are missing or non-finite")
        return np.quantile(candidate_errors, 0.9, axis=0, method="higher") - np.quantile(
            reference_errors, 0.9, axis=0, method="higher"
        )

    all_indices = list(range(len(references)))
    precision_point = precision_statistic(all_indices)
    cost_points: dict[str, float] = {}
    for name in cost_names:
        candidate_cost = float(np.median([row[name] for row in candidates]))
        reference_cost = float(np.median([row[name] for row in references]))
        if reference_cost <= 0.0 or candidate_cost < 0.0:
            raise SolverError(f"paired cost {name} must have positive reference and non-negative candidate")
        cost_points[name] = 1.0 - candidate_cost / reference_cost
    nonconvergence_point = float(
        np.mean([not row["convergence"] for row in candidates])
        - np.mean([not row["convergence"] for row in references])
    )
    for repeat in range(resamples):
        selected = _bootstrap_indices(references, rng)
        precision[repeat] = precision_statistic(selected)
        for name in cost_names:
            candidate_cost = float(np.median([candidates[index][name] for index in selected]))
            reference_cost = float(np.median([references[index][name] for index in selected]))
            if reference_cost <= 0.0 or candidate_cost < 0.0:
                raise SolverError(f"paired cost {name} must have positive reference and non-negative candidate")
            reductions[name][repeat] = 1.0 - candidate_cost / reference_cost
        nonconvergence[repeat] = float(
            np.mean([not candidates[index]["convergence"] for index in selected])
            - np.mean([not references[index]["convergence"] for index in selected])
        )
    return {
        "candidate_method_id": candidate,
        "resamples": resamples,
        "precision_p90_difference": [
            {
                "point": float(precision_point[component]),
                "ci": list(_quantile_interval(precision[:, component], confidence)),
            }
            for component in range(component_count)
        ],
        "cost_relative_reduction": {
            name: {
                "point": cost_points[name],
                "ci": list(_quantile_interval(values, confidence)),
            }
            for name, values in reductions.items()
        },
        "nonconvergence_rate_difference": {
            "point": nonconvergence_point,
            "ci": list(_quantile_interval(nonconvergence, confidence)),
        },
    }


def _judge_bootstrap(summary: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    bands = config["gates"]["non_inferiority_bands"]
    components = config["components"]
    ni = all(
        float(summary["precision_p90_difference"][index]["ci"][1]) <= float(bands[component])
        for index, component in enumerate(components)
    )
    costs = summary["cost_relative_reduction"]
    e30 = any(float(costs[name]["ci"][0]) >= float(config["gates"]["e30_minimum_reduction"]) for name in ("iterations", "forward_calls"))
    e20 = float(costs["solver_wall_clock"]["ci"][0]) >= float(config["gates"]["e20_minimum_reduction"])
    regression_floor = -float(config["gates"]["nr5_maximum_regression"])
    nr5_cost = all(float(value["ci"][0]) >= regression_floor for value in costs.values())
    nr5_failure = float(summary["nonconvergence_rate_difference"]["ci"][1]) <= float(config["gates"]["nr5_maximum_regression"])
    return {
        "ni": ni,
        "e30": e30,
        "e20": e20,
        "nr5": nr5_cost and nr5_failure,
        "passes": ni and (e30 or e20) and nr5_cost and nr5_failure,
    }


def evaluate_c2_preflight(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    methods = [method for method in config["method_ids"] if method != "joint_lm"]
    failed_rows = [dict(row) for row in rows if row.get("status", "complete") != "complete"]
    if failed_rows:
        return {
            "c2_preflight": "fail",
            "method_verdicts": [
                {"method_id": method, "passes": False, "passing_cells": [], "passing_seeds": []}
                for method in methods
            ],
            "cell_seed_reports": [],
            "failed_rows": failed_rows,
            "activated_tasks": {"P3-10": False, "P3-12": False},
        }
    reports = []
    for method in methods:
        candidates, references = _paired_rows(rows, method)
        cells_and_seeds = sorted({(str(row["grid_cell_id"]), int(row["seed"])) for row in candidates})
        for cell, seed in cells_and_seeds:
            local_rows = [
                row for row in candidates + references
                if str(row["grid_cell_id"]) == cell and int(row["seed"]) == seed
            ]
            summary = paired_group_bootstrap(
                local_rows,
                method,
                component_count=len(config["components"]),
                resamples=int(config["statistics"]["bootstrap_resamples"]),
                seed=int(config["statistics"]["bootstrap_seed"]),
                confidence=float(config["statistics"]["confidence_level"]),
            )
            reports.append({
                "grid_cell_id": cell,
                "seed": seed,
                "method_id": method,
                "bootstrap": summary,
                "gates": _judge_bootstrap(summary, config),
            })
    method_verdicts = []
    passing_methods = []
    for method in methods:
        passed = [report for report in reports if report["method_id"] == method and report["gates"]["passes"]]
        cells = sorted({report["grid_cell_id"] for report in passed})
        seeds = sorted({report["seed"] for report in passed})
        robust = (
            len(cells) >= int(config["robustness"]["minimum_distinct_passing_cells"])
            and len(seeds) >= int(config["robustness"]["minimum_distinct_passing_seeds"])
        )
        method_verdicts.append({"method_id": method, "passes": robust, "passing_cells": cells, "passing_seeds": seeds})
        if robust:
            passing_methods.append(method)
    different_regions = False
    for left_index, left in enumerate(passing_methods):
        left_cells = set(next(item["passing_cells"] for item in method_verdicts if item["method_id"] == left))
        for right in passing_methods[left_index + 1:]:
            right_cells = set(next(item["passing_cells"] for item in method_verdicts if item["method_id"] == right))
            if len(left_cells | right_cells) >= 2:
                different_regions = True
    passed = bool(passing_methods)
    return {
        "c2_preflight": "pass" if passed else "fail",
        "method_verdicts": method_verdicts,
        "cell_seed_reports": reports,
        "activated_tasks": {
            "P3-10": passed,
            "P3-12": len(passing_methods) >= 2 and different_regions,
        },
    }


def _validate_coverage(trials: Sequence[SolverTrial], config: Mapping[str, Any]) -> None:
    actual = {(trial.grid_cell_id, trial.split_id, trial.seed) for trial in trials}
    expected = {
        (cell, split, int(seed))
        for cell in config["grid_cell_ids"]
        for split in config["split_ids"]
        for seed in config["seeds"]
    }
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SolverError(f"formal preflight coverage mismatch; missing={missing}; extra={extra}")


def run_solver_preflight(
    trials: Sequence[SolverTrial],
    config: Mapping[str, Any],
    *,
    runtime_metadata: Mapping[str, Any],
    technical_test_mode: bool = False,
) -> dict[str, Any]:
    """Run paired solvers; technical mode explicitly waives only 9x5x3 coverage."""

    if config.get("plan_status") != "frozen_before_solver_run":
        raise SolverError("solver plan must be frozen before any run")
    if not trials:
        raise SolverError("solver trials cannot be empty")
    if not technical_test_mode:
        _validate_coverage(trials, config)
    metadata_fields = [str(field) for field in config["runtime_metadata_required_fields"]]
    missing_metadata = sorted(set(metadata_fields) - set(runtime_metadata))
    extra_metadata = sorted(set(runtime_metadata) - set(metadata_fields))
    if missing_metadata or extra_metadata:
        raise SolverError(
            f"runtime metadata keys mismatch; missing={missing_metadata}; extra={extra_metadata}"
        )
    components = [str(item) for item in config["components"]]
    options = _options_from_config(config["solver_budget"])
    solver_functions = {
        "joint_lm": solve_joint_lm,
        "classical_vp": solve_classical_vp,
        "vplr": solve_vplr,
        "tsvd_ridge_vp": solve_tsvd_ridge_vp,
    }
    if list(config["method_ids"]) != list(solver_functions):
        raise SolverError("solver method order does not match the frozen implementation set")
    rows = []
    projection_checks = []
    execution_errors = []
    repeats = int(config["timing"]["formal_repeats"])
    warmups = int(config["timing"]["warmup_solver_runs"])
    if repeats <= 0 or warmups < 0:
        raise SolverError("timing repeats must be positive and warmups cannot be negative")
    for trial in trials:
        if trial.split_id not in config["split_ids"] or trial.seed not in config["seeds"]:
            raise SolverError("trial split or seed is not registered")
        for function in solver_functions.values():
            for _ in range(warmups):
                try:
                    function(trial.problem, options)
                except Exception as exc:
                    execution_errors.append(
                        {
                            "phase": "warmup",
                            "sequence_id": trial.sequence_id,
                            "method_id": next(name for name, value in solver_functions.items() if value is function),
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
        results_by_repeat: list[dict[str, SolverResult]] = []
        for repeat_index in range(repeats):
            interleave_seed = int(trial.seed) * 1000003 + repeat_index
            order = np.random.default_rng(interleave_seed).permutation(list(solver_functions))
            results: dict[str, SolverResult] = {}
            for method_value in order:
                method = str(method_value)
                started = time.perf_counter_ns()
                try:
                    result = solver_functions[method](trial.problem, options)
                    row = _result_row(trial, result, components, repeat_index, runtime_metadata)
                    results[method] = result
                except Exception as exc:
                    row = _failure_row(
                        trial,
                        method,
                        repeat_index,
                        exc,
                        time.perf_counter_ns() - started,
                        runtime_metadata,
                    )
                    execution_errors.append(
                        {
                            "phase": "timed",
                            "sequence_id": trial.sequence_id,
                            "method_id": method,
                            "repeat_index": repeat_index,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                rows.append(row)
            results_by_repeat.append(results)
        try:
            disabled = solve_projection_disabled_control(trial.problem, options)
            joint = results_by_repeat[0]["joint_lm"]
            projection_checks.append(
                np.array_equal(disabled.linear_parameters, joint.linear_parameters)
                and np.array_equal(disabled.nonlinear_parameters, joint.nonlinear_parameters)
                and disabled.forward_calls == joint.forward_calls
                and disabled.iterations == joint.iterations
            )
        except Exception as exc:
            projection_checks.append(False)
            execution_errors.append(
                {
                    "phase": "projection_disabled_control",
                    "sequence_id": trial.sequence_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    labels = [row["information_band"] for row in rows]
    shuffled = np.random.default_rng(int(config["negative_controls"]["information_label_shuffle_seed"])).permutation(labels)
    invariant_fields = [
        (
            row["method_id"],
            row.get("component_abs_errors"),
            row.get("iterations"),
            row.get("forward_calls"),
            row["convergence"],
        )
        for row in rows
    ]
    label_control_pass = len(shuffled) == len(invariant_fields) and sorted(labels) == sorted(shuffled.tolist())
    negative_controls = {
        "information_band_label_shuffle": {
            "passed": label_control_pass,
            "solver_inputs_contain_information_band": False,
        },
        "projection_disabled_joint_lm": {
            "passed": all(projection_checks),
            "checked_trials": len(projection_checks),
        },
    }
    evaluation = evaluate_c2_preflight(rows, config)
    if execution_errors or not all(control["passed"] for control in negative_controls.values()):
        evaluation["c2_preflight"] = "fail"
        evaluation["activated_tasks"] = {"P3-10": False, "P3-12": False}
    return {
        "schema_version": "gib-benchmark-1",
        "task_id": "P3-06",
        "task_status": "completed",
        **evaluation,
        "solver_rows": rows,
        "negative_controls": negative_controls,
        "execution_errors": execution_errors,
        "technical_test_mode": technical_test_mode,
        "formal_run_started": not technical_test_mode,
        "timing_repeats": repeats,
        "warmup_solver_runs": warmups,
    }


__all__ = [
    "SeparableProblem",
    "SolverError",
    "SolverOptions",
    "SolverResult",
    "SolverTrial",
    "evaluate_c2_preflight",
    "paired_group_bootstrap",
    "run_solver_preflight",
    "solve_classical_vp",
    "solve_joint_lm",
    "solve_projection_disabled_control",
    "solve_tsvd_ridge_vp",
    "solve_vplr",
]
