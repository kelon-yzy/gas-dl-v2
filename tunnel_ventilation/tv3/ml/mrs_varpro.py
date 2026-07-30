"""Deterministic MEI-3 S1/S2/S3 solvers on the registered dry-basis domain."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from tv3.sim.generation.tunnel_ventilation.mrs_observation import (
    RAW3_TANGENT_BASIS,
    ideal_mrs_observation,
    raw3_percent_from_tangent,
    raw3_tangent_coordinates,
)


class SolverDomainError(ValueError):
    """Recoverable bound or finite-difference domain failure during a solve."""


@dataclass(frozen=True)
class S1Problem:
    observation: np.ndarray
    covariance: np.ndarray
    frequencies_hz: np.ndarray
    phase_branch_cycles: np.ndarray
    observation_std: Mapping[str, float]
    p_mpa: float


@dataclass(frozen=True)
class S1Parameterization:
    names: tuple[str, ...]
    scales: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    finite_difference_steps: np.ndarray
    prior_indices: np.ndarray
    prior_mean: np.ndarray
    prior_std: np.ndarray


@dataclass(frozen=True)
class S1SolverSettings:
    max_iterations: int
    initial_damping: float
    damping_increase: float
    damping_decrease: float
    max_line_search_steps: int
    line_search_contraction: float
    gradient_tolerance: float
    step_tolerance: float
    objective_tolerance: float


@dataclass(frozen=True)
class S1Iteration:
    iteration: int
    objective: float
    gradient_inf_norm: float
    scaled_step_norm: float
    step_length: float
    damping: float
    forward_calls: int


@dataclass(frozen=True)
class S1Solution:
    parameters: np.ndarray
    raw3_percent: np.ndarray
    success: bool
    stop_reason: str
    iterations: tuple[S1Iteration, ...]
    objective: float
    forward_calls: int
    bound_hit: bool
    bound_parameters: tuple[str, ...]


@dataclass(frozen=True)
class ConditionalLinearSolution:
    parameters: np.ndarray
    whitened_data_residual: np.ndarray
    prior_residual: np.ndarray
    augmented_rank: int
    augmented_condition_number: float

    @property
    def augmented_residual(self) -> np.ndarray:
        return np.concatenate((self.whitened_data_residual, self.prior_residual))


@dataclass(frozen=True)
class VarProParameterization:
    nonlinear_indices: np.ndarray
    linear_indices: np.ndarray
    nonlinear_names: tuple[str, ...]
    linear_names: tuple[str, ...]


@dataclass(frozen=True)
class VarProEvaluation:
    residual: np.ndarray
    linear_parameters: np.ndarray
    augmented_rank: int
    augmented_condition_number: float


@dataclass(frozen=True)
class VarProSolution:
    method: str
    parameters: np.ndarray
    raw3_percent: np.ndarray
    linear_parameters: np.ndarray
    success: bool
    stop_reason: str
    iterations: tuple[S1Iteration, ...]
    objective: float
    forward_calls: int
    bound_hit: bool
    bound_parameters: tuple[str, ...]


def _bound_hit_info(
    parameters: np.ndarray, spec: S1Parameterization
) -> tuple[bool, tuple[str, ...]]:
    hits: list[str] = []
    for index, name in enumerate(spec.names):
        step = float(spec.finite_difference_steps[index])
        lower = float(spec.lower_bounds[index])
        upper = float(spec.upper_bounds[index])
        value = float(parameters[index])
        if abs(value - lower) <= step or abs(value - upper) <= step:
            hits.append(name)
    return bool(hits), tuple(hits)


def _as_vector(name: str, value: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 1-D array")
    return array


def solve_conditionally_linear(
    *,
    observation: np.ndarray | Sequence[float],
    nonlinear_prediction: np.ndarray | Sequence[float],
    design_matrix: np.ndarray,
    covariance: np.ndarray,
    prior_mean: np.ndarray | Sequence[float],
    prior_std: np.ndarray | Sequence[float],
) -> ConditionalLinearSolution:
    """Solve the Phase A linear nuisance block after whitening data and priors."""
    y = _as_vector("observation", observation)
    b = _as_vector("nonlinear_prediction", nonlinear_prediction)
    a0 = _as_vector("prior_mean", prior_mean)
    sigma_a = _as_vector("prior_std", prior_std)
    matrix = np.asarray(design_matrix, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if y.shape != b.shape:
        raise ValueError("observation and nonlinear_prediction shapes differ")
    if matrix.ndim != 2 or matrix.shape != (y.size, a0.size):
        raise ValueError("design_matrix shape must be (n_observations, n_parameters)")
    if cov.shape != (y.size, y.size):
        raise ValueError("covariance shape must be (n_observations, n_observations)")
    if sigma_a.shape != a0.shape or np.any(sigma_a <= 0.0):
        raise ValueError("prior_std must be positive and match prior_mean")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(cov)):
        raise ValueError("design_matrix and covariance must be finite")
    if not np.allclose(cov, cov.T, rtol=0.0, atol=1e-14):
        raise ValueError("covariance must be symmetric")
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    whitened_matrix = np.linalg.solve(chol, matrix)
    whitened_target = np.linalg.solve(chol, y - b)
    prior_matrix = np.diag(1.0 / sigma_a)
    augmented_matrix = np.vstack((whitened_matrix, prior_matrix))
    augmented_target = np.concatenate((whitened_target, prior_matrix @ a0))
    parameters, _residuals, rank, singular_values = np.linalg.lstsq(
        augmented_matrix, augmented_target, rcond=None
    )
    return ConditionalLinearSolution(
        parameters=parameters,
        whitened_data_residual=whitened_target - whitened_matrix @ parameters,
        prior_residual=prior_matrix @ (a0 - parameters),
        augmented_rank=int(rank),
        augmented_condition_number=float(singular_values[0] / singular_values[-1]),
    )


def build_s1_parameterization(
    config: Mapping[str, Any], *, scales: Sequence[float] | None = None
) -> S1Parameterization:
    b1 = config["b1_solver_audit"]
    frequencies = tuple(float(value) for value in b1["frequencies_hz"])
    names = (
        "composition_tangent_0_percent",
        "composition_tangent_1_percent",
        "t_c",
        "path_length_m",
        "h_rh",
        "common_delay_s",
        "log_amplitude_gain",
        *(f"log_amplitude_offset_{int(value)}hz" for value in frequencies),
    )
    registry = b1["parameters"]
    scale_values = np.asarray(
        [registry[name]["scale"] for name in names] if scales is None else scales,
        dtype=np.float64,
    )
    lower = np.asarray([registry[name]["lower"] for name in names], dtype=np.float64)
    upper = np.asarray([registry[name]["upper"] for name in names], dtype=np.float64)
    steps = np.asarray(
        [registry[name]["finite_difference_step"] for name in names], dtype=np.float64
    )
    if scale_values.shape != (len(names),) or np.any(scale_values <= 0.0):
        raise ValueError("S1 parameter scales must be positive and match parameter names")
    if np.any(lower >= upper) or np.any(steps <= 0.0):
        raise ValueError("S1 parameter bounds and finite-difference steps are invalid")
    prior_names = tuple(b1["priors"])
    prior_indices = np.asarray([names.index(name) for name in prior_names], dtype=np.int64)
    prior_mean = np.asarray([b1["priors"][name]["mean"] for name in prior_names])
    prior_std = np.asarray([b1["priors"][name]["std"] for name in prior_names])
    if np.any(prior_std <= 0.0):
        raise ValueError("S1 prior standard deviations must be positive")
    return S1Parameterization(
        names=names,
        scales=scale_values,
        lower_bounds=lower,
        upper_bounds=upper,
        finite_difference_steps=steps,
        prior_indices=prior_indices,
        prior_mean=prior_mean.astype(np.float64),
        prior_std=prior_std.astype(np.float64),
    )


def build_s1_settings(config: Mapping[str, Any]) -> S1SolverSettings:
    values = config["b1_solver_audit"]["solver"]
    return S1SolverSettings(**{name: values[name] for name in S1SolverSettings.__annotations__})


def build_varpro_parameterization(
    config: Mapping[str, Any], spec: S1Parameterization
) -> VarProParameterization:
    contract = config["b2_solver_core"]
    nonlinear_names = tuple(contract["nonlinear_parameter_names"])
    linear_names = tuple(contract["linear_parameter_names"])
    if "path_length_m" in linear_names:
        raise ValueError("path_length_m must remain in the nonlinear block")
    if nonlinear_names + linear_names != spec.names:
        raise ValueError("B2 nonlinear and linear blocks must exactly partition S1 parameters")
    if contract["per_frequency_offset_scope"] != (
        "device_profile_id_x_frequency_hz_shared_across_samples"
    ):
        raise ValueError("per-frequency offsets must be shared by device profile and frequency")
    prior_names = {spec.names[index] for index in spec.prior_indices}
    missing_priors = sorted(set(linear_names) - prior_names)
    if missing_priors:
        raise ValueError(f"linear parameters require independent priors: {missing_priors}")
    return VarProParameterization(
        nonlinear_indices=np.asarray([spec.names.index(name) for name in nonlinear_names]),
        linear_indices=np.asarray([spec.names.index(name) for name in linear_names]),
        nonlinear_names=nonlinear_names,
        linear_names=linear_names,
    )


def _linear_design(problem: S1Problem, varpro: VarProParameterization) -> np.ndarray:
    matrix = np.zeros((problem.observation.size, len(varpro.linear_names)))
    for column, name in enumerate(varpro.linear_names):
        if name == "common_delay_s":
            matrix[0::3, column] = 1.0
            matrix[2::3, column] = -2.0 * np.pi * problem.frequencies_hz
        elif name == "log_amplitude_gain":
            matrix[1::3, column] = 1.0
        elif name.startswith("log_amplitude_offset_"):
            frequency = float(name.removeprefix("log_amplitude_offset_").removesuffix("hz"))
            matches = np.flatnonzero(problem.frequencies_hz == frequency)
            if matches.size != 1:
                raise ValueError(f"linear offset frequency does not match K4: {frequency}")
            matrix[3 * int(matches[0]) + 1, column] = 1.0
        else:
            raise ValueError(f"unsupported B2 linear parameter: {name}")
    return matrix


def _assemble_parameters(
    nonlinear: np.ndarray,
    linear: np.ndarray,
    spec: S1Parameterization,
    varpro: VarProParameterization,
) -> np.ndarray:
    values = np.empty(len(spec.names), dtype=np.float64)
    values[varpro.nonlinear_indices] = nonlinear
    values[varpro.linear_indices] = linear
    return values


def _nonlinear_prediction(
    problem: S1Problem,
    nonlinear: np.ndarray,
    spec: S1Parameterization,
    varpro: VarProParameterization,
) -> np.ndarray:
    linear_zero = np.zeros(varpro.linear_indices.size)
    return predict_s1(problem, _assemble_parameters(nonlinear, linear_zero, spec, varpro), spec)


def _linear_prior(
    spec: S1Parameterization, varpro: VarProParameterization
) -> tuple[np.ndarray, np.ndarray]:
    prior_by_index = {
        int(index): (float(mean), float(std))
        for index, mean, std in zip(
            spec.prior_indices, spec.prior_mean, spec.prior_std, strict=True
        )
    }
    mean, std = zip(*(prior_by_index[int(index)] for index in varpro.linear_indices), strict=True)
    return np.asarray(mean), np.asarray(std)


def validate_phase_branch_consistency(
    problem: S1Problem, *, max_standardized_error: float
) -> None:
    _validate_problem(problem)
    frequencies = np.asarray(problem.frequencies_hz, dtype=np.float64)
    tof_rows = np.arange(0, problem.observation.size, 3)
    phase_rows = tof_rows + 2
    coefficients = 2.0 * np.pi * frequencies
    discrepancy = (
        problem.observation[phase_rows]
        + coefficients * problem.observation[tof_rows]
        - 2.0 * np.pi * problem.phase_branch_cycles
    )
    variances = (
        problem.covariance[phase_rows, phase_rows]
        + coefficients**2 * problem.covariance[tof_rows, tof_rows]
        + 2.0 * coefficients * problem.covariance[phase_rows, tof_rows]
    )
    if np.any(variances <= 0.0):
        raise ValueError("phase-branch consistency variance must be positive")
    if float(np.max(np.abs(discrepancy) / np.sqrt(variances))) > max_standardized_error:
        raise ValueError("phase_branch_cycles are inconsistent with the observed TOF and phase")


def evaluate_varpro(
    problem: S1Problem,
    nonlinear_parameters: Sequence[float],
    spec: S1Parameterization,
    varpro: VarProParameterization,
) -> VarProEvaluation:
    nonlinear = np.asarray(nonlinear_parameters, dtype=np.float64)
    if nonlinear.shape != varpro.nonlinear_indices.shape:
        raise ValueError("S2 nonlinear parameter shape is invalid")
    probe = _assemble_parameters(
        nonlinear, np.zeros(varpro.linear_indices.size), spec, varpro
    )
    if not _is_feasible(probe, spec):
        raise SolverDomainError("S2 nonlinear parameters are outside the registered physical domain")
    prior_mean, prior_std = _linear_prior(spec, varpro)
    solution = solve_conditionally_linear(
        observation=problem.observation,
        nonlinear_prediction=_nonlinear_prediction(problem, nonlinear, spec, varpro),
        design_matrix=_linear_design(problem, varpro),
        covariance=problem.covariance,
        prior_mean=prior_mean,
        prior_std=prior_std,
    )
    values = _assemble_parameters(nonlinear, solution.parameters, spec, varpro)
    if not _is_feasible(values, spec):
        raise SolverDomainError("S2 conditional linear solution violates the frozen S1 bounds")
    # Phase A uses target-minus-prediction residuals; negate them to preserve S1's
    # prediction-minus-target residual convention before appending nonlinear priors.
    data_residual = -solution.whitened_data_residual
    prior_by_index = {
        int(index): (float(mean), float(std))
        for index, mean, std in zip(
            spec.prior_indices, spec.prior_mean, spec.prior_std, strict=True
        )
    }
    prior_residual = np.asarray(
        [
            (values[int(index)] - prior_by_index[int(index)][0])
            / prior_by_index[int(index)][1]
            for index in spec.prior_indices
        ]
    )
    return VarProEvaluation(
        residual=np.concatenate((data_residual, prior_residual)),
        linear_parameters=solution.parameters,
        augmented_rank=solution.augmented_rank,
        augmented_condition_number=solution.augmented_condition_number,
    )


def _varpro_projected_jacobian_with_forward_calls(
    problem: S1Problem,
    nonlinear_parameters: Sequence[float],
    spec: S1Parameterization,
    varpro: VarProParameterization,
) -> tuple[np.ndarray, int]:
    """Differentiate the augmented base residual and count physical predictions."""
    nonlinear = np.asarray(nonlinear_parameters, dtype=np.float64)
    chol = _validate_problem(problem)
    design = _linear_design(problem, varpro)
    whitened_design = np.linalg.solve(chol, design)
    prior_rows = spec.prior_indices.size
    augmented_design = np.zeros(
        (problem.observation.size + prior_rows, varpro.linear_indices.size)
    )
    augmented_design[: problem.observation.size] = whitened_design
    linear_positions = {int(index): column for column, index in enumerate(varpro.linear_indices)}
    for row, (index, std) in enumerate(zip(spec.prior_indices, spec.prior_std, strict=True)):
        if int(index) in linear_positions:
            augmented_design[problem.observation.size + row, linear_positions[int(index)]] = 1.0 / std
    q, _r = np.linalg.qr(augmented_design, mode="reduced")
    projector = np.eye(augmented_design.shape[0]) - q @ q.T
    base_jacobian = np.zeros((augmented_design.shape[0], nonlinear.size))
    nonlinear_positions = {int(index): column for column, index in enumerate(varpro.nonlinear_indices)}
    forward_calls = 0
    center_prediction: np.ndarray | None = None
    for column, full_index in enumerate(varpro.nonlinear_indices):
        step = spec.finite_difference_steps[int(full_index)]
        plus = nonlinear.copy()
        minus = nonlinear.copy()
        plus[column] += step
        minus[column] -= step
        plus_probe = _assemble_parameters(
            plus, np.zeros(varpro.linear_indices.size), spec, varpro
        )
        minus_probe = _assemble_parameters(
            minus, np.zeros(varpro.linear_indices.size), spec, varpro
        )
        plus_ok = _is_feasible(plus_probe, spec)
        minus_ok = _is_feasible(minus_probe, spec)
        if plus_ok and minus_ok:
            plus_prediction = _nonlinear_prediction(problem, plus, spec, varpro)
            minus_prediction = _nonlinear_prediction(problem, minus, spec, varpro)
            forward_calls += 2
            base_jacobian[: problem.observation.size, column] = np.linalg.solve(
                chol, (plus_prediction - minus_prediction) / (2.0 * step)
            )
        elif plus_ok:
            if center_prediction is None:
                center_prediction = _nonlinear_prediction(problem, nonlinear, spec, varpro)
                forward_calls += 1
            plus_prediction = _nonlinear_prediction(problem, plus, spec, varpro)
            forward_calls += 1
            base_jacobian[: problem.observation.size, column] = np.linalg.solve(
                chol, (plus_prediction - center_prediction) / step
            )
        elif minus_ok:
            if center_prediction is None:
                center_prediction = _nonlinear_prediction(problem, nonlinear, spec, varpro)
                forward_calls += 1
            minus_prediction = _nonlinear_prediction(problem, minus, spec, varpro)
            forward_calls += 1
            base_jacobian[: problem.observation.size, column] = np.linalg.solve(
                chol, (center_prediction - minus_prediction) / step
            )
        else:
            raise SolverDomainError(
                "projected jacobian finite difference left the registered domain"
            )
    for row, (index, std) in enumerate(zip(spec.prior_indices, spec.prior_std, strict=True)):
        column = nonlinear_positions.get(int(index))
        if column is not None:
            base_jacobian[problem.observation.size + row, column] = 1.0 / std
    nonlinear_scales = spec.scales[varpro.nonlinear_indices]
    return projector @ base_jacobian * nonlinear_scales[np.newaxis, :], forward_calls


def varpro_projected_jacobian(
    problem: S1Problem,
    nonlinear_parameters: Sequence[float],
    spec: S1Parameterization,
    varpro: VarProParameterization,
) -> np.ndarray:
    """Differentiate the augmented base residual, then apply the exact projector."""
    jacobian, _forward_calls = _varpro_projected_jacobian_with_forward_calls(
        problem, nonlinear_parameters, spec, varpro
    )
    return jacobian


def pack_s1_parameters(
    raw3_percent: Sequence[float],
    *,
    t_c: float,
    path_length_m: float,
    h_rh: float,
    common_delay_s: float,
    log_amplitude_gain: float,
    per_frequency_offsets: Sequence[float],
) -> np.ndarray:
    return np.concatenate(
        (
            raw3_tangent_coordinates(raw3_percent),
            np.asarray(
                [t_c, path_length_m, h_rh, common_delay_s, log_amplitude_gain],
                dtype=np.float64,
            ),
            np.asarray(per_frequency_offsets, dtype=np.float64),
        )
    )


def _validate_problem(problem: S1Problem) -> np.ndarray:
    observation = np.asarray(problem.observation, dtype=np.float64)
    covariance = np.asarray(problem.covariance, dtype=np.float64)
    frequencies = np.asarray(problem.frequencies_hz, dtype=np.float64)
    branches = np.asarray(problem.phase_branch_cycles)
    expected_rows = frequencies.size * 3
    if observation.shape != (expected_rows,) or not np.all(np.isfinite(observation)):
        raise ValueError("S1 observation must be a finite frequency-major vector")
    if covariance.shape != (expected_rows, expected_rows):
        raise ValueError("S1 covariance shape does not match observation")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-14):
        raise ValueError("S1 covariance must be symmetric")
    if branches.shape != frequencies.shape or not np.issubdtype(branches.dtype, np.integer):
        raise ValueError("S1 phase branches must be explicit integers per frequency")
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("S1 covariance must be positive definite") from exc


def _is_feasible(parameters: np.ndarray, spec: S1Parameterization) -> bool:
    if parameters.shape != spec.scales.shape or not np.all(np.isfinite(parameters)):
        return False
    if np.any(parameters < spec.lower_bounds) or np.any(parameters > spec.upper_bounds):
        return False
    try:
        raw3_percent_from_tangent(parameters[:2])
    except ValueError:
        return False
    return True


def predict_s1(
    problem: S1Problem, parameters: Sequence[float], spec: S1Parameterization
) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    if not _is_feasible(values, spec):
        raise ValueError("S1 parameters are outside the registered physical domain")
    raw3 = raw3_percent_from_tangent(values[:2])
    ideal = ideal_mrs_observation(
        raw3,
        t_c=float(values[2]),
        p_mpa=float(problem.p_mpa),
        h_rh=float(values[4]),
        path_length_m=float(values[3]),
        frequencies_hz=problem.frequencies_hz,
        phase_branch_cycles=problem.phase_branch_cycles,
        observation_std=problem.observation_std,
    ).vector
    prediction = ideal.copy()
    delay = float(values[5])
    gain = float(values[6])
    offsets = values[7:]
    if offsets.size != problem.frequencies_hz.size:
        raise ValueError("S1 per-frequency offset count does not match frequencies")
    prediction[0::3] += delay
    prediction[1::3] += gain + offsets
    prediction[2::3] -= 2.0 * np.pi * problem.frequencies_hz * delay
    return prediction


def augmented_residual(
    problem: S1Problem,
    parameters: Sequence[float],
    spec: S1Parameterization,
    *,
    covariance_cholesky: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64)
    chol = _validate_problem(problem) if covariance_cholesky is None else covariance_cholesky
    data = np.linalg.solve(chol, predict_s1(problem, values, spec) - problem.observation)
    prior = (values[spec.prior_indices] - spec.prior_mean) / spec.prior_std
    return np.concatenate((data, prior))


def _finite_difference_jacobian_with_forward_calls(
    problem: S1Problem,
    parameters: Sequence[float],
    spec: S1Parameterization,
    *,
    covariance_cholesky: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    values = np.asarray(parameters, dtype=np.float64)
    chol = _validate_problem(problem) if covariance_cholesky is None else covariance_cholesky
    center = augmented_residual(problem, values, spec, covariance_cholesky=chol)
    forward_calls = 1
    jacobian = np.empty(
        (problem.observation.size + spec.prior_indices.size, values.size), dtype=np.float64
    )
    for column, physical_step in enumerate(spec.finite_difference_steps):
        delta = np.zeros_like(values)
        delta[column] = physical_step
        plus_ok = _is_feasible(values + delta, spec)
        minus_ok = _is_feasible(values - delta, spec)
        if plus_ok and minus_ok:
            plus = augmented_residual(
                problem, values + delta, spec, covariance_cholesky=chol
            )
            minus = augmented_residual(
                problem, values - delta, spec, covariance_cholesky=chol
            )
            forward_calls += 2
            jacobian[:, column] = (plus - minus) / (2.0 * physical_step)
        elif plus_ok:
            plus = augmented_residual(
                problem, values + delta, spec, covariance_cholesky=chol
            )
            forward_calls += 1
            jacobian[:, column] = (plus - center) / physical_step
        elif minus_ok:
            minus = augmented_residual(
                problem, values - delta, spec, covariance_cholesky=chol
            )
            forward_calls += 1
            jacobian[:, column] = (center - minus) / physical_step
        else:
            raise SolverDomainError(
                "finite difference left the registered S1 domain on both sides"
            )
    return jacobian * spec.scales[np.newaxis, :], forward_calls


def finite_difference_jacobian(
    problem: S1Problem,
    parameters: Sequence[float],
    spec: S1Parameterization,
    *,
    covariance_cholesky: np.ndarray | None = None,
) -> np.ndarray:
    """Finite-difference Jacobian in standardized solver coordinates."""
    jacobian, _forward_calls = _finite_difference_jacobian_with_forward_calls(
        problem, parameters, spec, covariance_cholesky=covariance_cholesky
    )
    return jacobian


def solve_s1(
    problem: S1Problem,
    initial_parameters: Sequence[float],
    spec: S1Parameterization,
    settings: S1SolverSettings,
) -> S1Solution:
    """Solve S1 with scaled damped Gauss-Newton and explicit backtracking."""
    chol = _validate_problem(problem)
    parameters = np.asarray(initial_parameters, dtype=np.float64).copy()
    if not _is_feasible(parameters, spec):
        raise ValueError("S1 initial parameters are outside the registered physical domain")
    damping = float(settings.initial_damping)
    trace: list[S1Iteration] = []
    forward_calls = 0
    residual = augmented_residual(problem, parameters, spec, covariance_cholesky=chol)
    forward_calls += 1
    objective = 0.5 * float(residual @ residual)

    for iteration in range(settings.max_iterations):
        try:
            jacobian, jacobian_forward_calls = _finite_difference_jacobian_with_forward_calls(
                problem, parameters, spec, covariance_cholesky=chol
            )
        except SolverDomainError:
            return _solution(
                parameters,
                False,
                "boundary_finite_difference",
                trace,
                objective,
                forward_calls,
                spec,
            )
        forward_calls += jacobian_forward_calls
        gradient = jacobian.T @ residual
        gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
        if gradient_norm <= settings.gradient_tolerance:
            return _solution(
                parameters, True, "gradient_tolerance", trace, objective, forward_calls, spec
            )
        normal = jacobian.T @ jacobian + damping * np.eye(parameters.size)
        scaled_step = np.linalg.solve(normal, -gradient)
        step_norm = float(np.linalg.norm(scaled_step))
        if step_norm <= settings.step_tolerance:
            return _solution(
                parameters, True, "step_tolerance", trace, objective, forward_calls, spec
            )

        accepted = False
        step_length = 1.0
        previous_objective = objective
        for _ in range(settings.max_line_search_steps):
            candidate = parameters + step_length * scaled_step * spec.scales
            if _is_feasible(candidate, spec):
                candidate_residual = augmented_residual(
                    problem, candidate, spec, covariance_cholesky=chol
                )
                forward_calls += 1
                candidate_objective = 0.5 * float(candidate_residual @ candidate_residual)
                if candidate_objective < objective:
                    parameters = candidate
                    residual = candidate_residual
                    objective = candidate_objective
                    accepted = True
                    break
            step_length *= settings.line_search_contraction

        trace.append(
            S1Iteration(
                iteration=iteration,
                objective=objective,
                gradient_inf_norm=gradient_norm,
                scaled_step_norm=step_norm,
                step_length=step_length if accepted else 0.0,
                damping=damping,
                forward_calls=forward_calls,
            )
        )
        if not accepted:
            damping *= settings.damping_increase
            continue
        damping = max(damping * settings.damping_decrease, np.finfo(float).eps)
        relative_change = (previous_objective - objective) / max(previous_objective, 1.0)
        if relative_change <= settings.objective_tolerance:
            return _solution(
                parameters, True, "objective_tolerance", trace, objective, forward_calls, spec
            )

    return _solution(
        parameters, False, "max_iterations", trace, objective, forward_calls, spec
    )


def solve_s2(
    problem: S1Problem,
    initial_parameters: Sequence[float],
    spec: S1Parameterization,
    settings: S1SolverSettings,
    varpro: VarProParameterization,
    *,
    max_phase_branch_standardized_error: float,
) -> VarProSolution:
    """Solve S2 by eliminating the admitted linear nuisance block at every iterate."""
    validate_phase_branch_consistency(
        problem, max_standardized_error=max_phase_branch_standardized_error
    )
    initial = np.asarray(initial_parameters, dtype=np.float64)
    if initial.shape == spec.scales.shape:
        nonlinear = initial[varpro.nonlinear_indices]
    elif initial.shape == varpro.nonlinear_indices.shape:
        nonlinear = initial.copy()
    else:
        raise ValueError("S2 initial parameter shape is invalid")

    def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = evaluate_varpro(problem, values, spec, varpro)
        return result.residual, result.linear_parameters

    def jacobian(values: np.ndarray) -> tuple[np.ndarray, int]:
        return _varpro_projected_jacobian_with_forward_calls(problem, values, spec, varpro)

    return _solve_reduced(
        method="S2",
        problem=problem,
        initial_nonlinear=nonlinear,
        spec=spec,
        settings=settings,
        varpro=varpro,
        evaluate=evaluate,
        jacobian=jacobian,
    )


def solve_s3(
    problem: S1Problem,
    initial_parameters: Sequence[float],
    spec: S1Parameterization,
    settings: S1SolverSettings,
    varpro: VarProParameterization,
    *,
    truth_linear_parameters: Sequence[float] | None,
    max_phase_branch_standardized_error: float,
) -> VarProSolution:
    """Solve the S3 upper bound with explicitly supplied truth nuisance values."""
    if truth_linear_parameters is None:
        raise ValueError("S3 requires explicitly isolated truth linear parameters")
    linear = np.asarray(truth_linear_parameters, dtype=np.float64)
    if linear.shape != varpro.linear_indices.shape or not np.all(np.isfinite(linear)):
        raise ValueError("S3 truth linear parameter shape is invalid")
    validate_phase_branch_consistency(
        problem, max_standardized_error=max_phase_branch_standardized_error
    )
    initial = np.asarray(initial_parameters, dtype=np.float64)
    if initial.shape == spec.scales.shape:
        nonlinear = initial[varpro.nonlinear_indices]
    elif initial.shape == varpro.nonlinear_indices.shape:
        nonlinear = initial.copy()
    else:
        raise ValueError("S3 initial parameter shape is invalid")
    chol = _validate_problem(problem)

    def evaluate(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        full = _assemble_parameters(values, linear, spec, varpro)
        if not _is_feasible(full, spec):
            raise SolverDomainError("S3 parameters are outside the frozen S1 bounds")
        return augmented_residual(
            problem, full, spec, covariance_cholesky=chol
        ), linear

    def jacobian(values: np.ndarray) -> tuple[np.ndarray, int]:
        matrix = np.empty(
            (problem.observation.size + spec.prior_indices.size, values.size)
        )
        forward_calls = 0
        center: np.ndarray | None = None
        for column, full_index in enumerate(varpro.nonlinear_indices):
            step = spec.finite_difference_steps[int(full_index)]
            plus = values.copy()
            minus = values.copy()
            plus[column] += step
            minus[column] -= step
            plus_probe = _assemble_parameters(
                plus, linear, spec, varpro
            )
            minus_probe = _assemble_parameters(
                minus, linear, spec, varpro
            )
            plus_ok = _is_feasible(plus_probe, spec)
            minus_ok = _is_feasible(minus_probe, spec)
            if plus_ok and minus_ok:
                plus_residual = evaluate(plus)[0]
                minus_residual = evaluate(minus)[0]
                forward_calls += 2
                matrix[:, column] = (plus_residual - minus_residual) / (2.0 * step)
            elif plus_ok:
                if center is None:
                    center = evaluate(values)[0]
                    forward_calls += 1
                plus_residual = evaluate(plus)[0]
                forward_calls += 1
                matrix[:, column] = (plus_residual - center) / step
            elif minus_ok:
                if center is None:
                    center = evaluate(values)[0]
                    forward_calls += 1
                minus_residual = evaluate(minus)[0]
                forward_calls += 1
                matrix[:, column] = (center - minus_residual) / step
            else:
                raise SolverDomainError(
                    "S3 finite difference left the registered domain on both sides"
                )
        return matrix * spec.scales[varpro.nonlinear_indices][np.newaxis, :], forward_calls

    return _solve_reduced(
        method="S3",
        problem=problem,
        initial_nonlinear=nonlinear,
        spec=spec,
        settings=settings,
        varpro=varpro,
        evaluate=evaluate,
        jacobian=jacobian,
    )


def _solve_reduced(
    *,
    method: str,
    problem: S1Problem,
    initial_nonlinear: np.ndarray,
    spec: S1Parameterization,
    settings: S1SolverSettings,
    varpro: VarProParameterization,
    evaluate: Any,
    jacobian: Any,
) -> VarProSolution:
    nonlinear = initial_nonlinear.copy()
    scales = spec.scales[varpro.nonlinear_indices]
    damping = float(settings.initial_damping)
    trace: list[S1Iteration] = []
    try:
        residual, linear = evaluate(nonlinear)
    except SolverDomainError:
        zero_linear = np.zeros(varpro.linear_indices.size, dtype=np.float64)
        return _varpro_solution(
            method,
            nonlinear,
            zero_linear,
            spec,
            varpro,
            False,
            "boundary_hit",
            trace,
            float("inf"),
            0,
        )
    forward_calls = 1
    objective = 0.5 * float(residual @ residual)
    for iteration in range(settings.max_iterations):
        try:
            matrix, jacobian_forward_calls = jacobian(nonlinear)
        except SolverDomainError:
            return _varpro_solution(
                method,
                nonlinear,
                linear,
                spec,
                varpro,
                False,
                "boundary_finite_difference",
                trace,
                objective,
                forward_calls,
            )
        forward_calls += jacobian_forward_calls
        gradient = matrix.T @ residual
        gradient_norm = float(np.linalg.norm(gradient, ord=np.inf))
        if gradient_norm <= settings.gradient_tolerance:
            return _varpro_solution(
                method, nonlinear, linear, spec, varpro, True,
                "gradient_tolerance", trace, objective, forward_calls
            )
        normal = matrix.T @ matrix + damping * np.eye(nonlinear.size)
        scaled_step = np.linalg.solve(normal, -gradient)
        step_norm = float(np.linalg.norm(scaled_step))
        if step_norm <= settings.step_tolerance:
            return _varpro_solution(
                method, nonlinear, linear, spec, varpro, True,
                "step_tolerance", trace, objective, forward_calls
            )
        accepted = False
        step_length = 1.0
        previous_objective = objective
        for _ in range(settings.max_line_search_steps):
            candidate = nonlinear + step_length * scaled_step * scales
            full_probe = _assemble_parameters(
                candidate, np.zeros(varpro.linear_indices.size), spec, varpro
            )
            if _is_feasible(full_probe, spec):
                try:
                    candidate_residual, candidate_linear = evaluate(candidate)
                except SolverDomainError:
                    step_length *= settings.line_search_contraction
                    continue
                forward_calls += 1
                candidate_objective = 0.5 * float(candidate_residual @ candidate_residual)
                if candidate_objective < objective:
                    nonlinear = candidate
                    linear = candidate_linear
                    residual = candidate_residual
                    objective = candidate_objective
                    accepted = True
                    break
            step_length *= settings.line_search_contraction
        trace.append(
            S1Iteration(
                iteration=iteration,
                objective=objective,
                gradient_inf_norm=gradient_norm,
                scaled_step_norm=step_norm,
                step_length=step_length if accepted else 0.0,
                damping=damping,
                forward_calls=forward_calls,
            )
        )
        if not accepted:
            damping *= settings.damping_increase
            continue
        damping = max(damping * settings.damping_decrease, np.finfo(float).eps)
        relative_change = (previous_objective - objective) / max(previous_objective, 1.0)
        if relative_change <= settings.objective_tolerance:
            return _varpro_solution(
                method, nonlinear, linear, spec, varpro, True,
                "objective_tolerance", trace, objective, forward_calls
            )
    return _varpro_solution(
        method, nonlinear, linear, spec, varpro, False,
        "max_iterations", trace, objective, forward_calls
    )


def _varpro_solution(
    method: str,
    nonlinear: np.ndarray,
    linear: np.ndarray,
    spec: S1Parameterization,
    varpro: VarProParameterization,
    success: bool,
    stop_reason: str,
    trace: list[S1Iteration],
    objective: float,
    forward_calls: int,
) -> VarProSolution:
    parameters = _assemble_parameters(nonlinear, linear, spec, varpro)
    bound_hit, bound_parameters = _bound_hit_info(parameters, spec)
    if not success and stop_reason.startswith("boundary"):
        bound_hit = True
    try:
        raw3 = raw3_percent_from_tangent(parameters[:2])
    except ValueError:
        raw3 = np.full(3, np.nan)
        success = False
        stop_reason = "boundary_hit"
        bound_hit = True
    return VarProSolution(
        method=method,
        parameters=parameters,
        raw3_percent=raw3,
        linear_parameters=linear.copy(),
        success=success,
        stop_reason=stop_reason,
        iterations=tuple(trace),
        objective=objective,
        forward_calls=forward_calls,
        bound_hit=bound_hit,
        bound_parameters=bound_parameters,
    )


def _solution(
    parameters: np.ndarray,
    success: bool,
    stop_reason: str,
    trace: list[S1Iteration],
    objective: float,
    forward_calls: int,
    spec: S1Parameterization,
) -> S1Solution:
    bound_hit, bound_parameters = _bound_hit_info(parameters, spec)
    if not success and stop_reason.startswith("boundary"):
        bound_hit = True
    try:
        raw3 = raw3_percent_from_tangent(parameters[:2])
    except ValueError:
        raw3 = np.full(3, np.nan)
        success = False
        stop_reason = "boundary_hit"
        bound_hit = True
    return S1Solution(
        parameters=parameters.copy(),
        raw3_percent=raw3,
        success=success,
        stop_reason=stop_reason,
        iterations=tuple(trace),
        objective=objective,
        forward_calls=forward_calls,
        bound_hit=bound_hit,
        bound_parameters=bound_parameters,
    )


def build_b1_synthetic_problem(
    config: Mapping[str, Any],
) -> tuple[S1Problem, S1Parameterization, np.ndarray]:
    """Build the registered in-memory B1 fixture; this is not formal data."""
    b1 = config["b1_solver_audit"]
    spec = build_s1_parameterization(config)
    truth = b1["synthetic_truth"]
    truth_parameters = pack_s1_parameters(
        truth["raw3_percent"],
        t_c=truth["t_c"],
        path_length_m=truth["path_length_m"],
        h_rh=truth["h_rh"],
        common_delay_s=truth["common_delay_s"],
        log_amplitude_gain=truth["log_amplitude_gain"],
        per_frequency_offsets=truth["per_frequency_offsets"],
    )
    ideal = ideal_mrs_observation(
        truth["raw3_percent"],
        t_c=truth["t_c"],
        p_mpa=b1["fixed_pressure_mpa"],
        h_rh=truth["h_rh"],
        path_length_m=truth["path_length_m"],
        frequencies_hz=b1["frequencies_hz"],
        phase_branch_cycles=b1["phase_branch_cycles"],
        observation_std=b1["observation_std"],
    )
    baseline_problem = S1Problem(
        observation=ideal.vector,
        covariance=ideal.covariance,
        frequencies_hz=ideal.frequencies_hz,
        phase_branch_cycles=ideal.phase_branch_cycles,
        observation_std=b1["observation_std"],
        p_mpa=b1["fixed_pressure_mpa"],
    )
    return (
        S1Problem(
            observation=predict_s1(baseline_problem, truth_parameters, spec),
            covariance=ideal.covariance,
            frequencies_hz=ideal.frequencies_hz,
            phase_branch_cycles=ideal.phase_branch_cycles,
            observation_std=b1["observation_std"],
            p_mpa=b1["fixed_pressure_mpa"],
        ),
        spec,
        truth_parameters,
    )


def run_b1_s1_numerical_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run B1 numerical checks while preserving the unresolved S0 blocker."""
    b1 = config["b1_solver_audit"]
    problem, spec, truth = build_b1_synthetic_problem(config)
    settings = build_s1_settings(config)
    probe = truth.copy()
    probe[0] += 0.2
    residual = augmented_residual(problem, probe, spec)
    jacobian = finite_difference_jacobian(problem, probe, spec)
    direction = np.linspace(-0.2, 0.2, probe.size)
    direction /= np.linalg.norm(direction)
    epsilon = float(b1["numerical_gates"]["directional_derivative_step"])
    objective_plus = 0.5 * np.sum(
        augmented_residual(problem, probe + epsilon * direction * spec.scales, spec)
        ** 2
    )
    objective_minus = 0.5 * np.sum(
        augmented_residual(problem, probe - epsilon * direction * spec.scales, spec)
        ** 2
    )
    derivative_fd = float((objective_plus - objective_minus) / (2.0 * epsilon))
    derivative_jacobian = float((jacobian.T @ residual) @ direction)
    derivative_error = abs(derivative_fd - derivative_jacobian) / max(
        abs(derivative_fd), 1.0
    )

    initializations = []
    for index, entry in enumerate(b1["frozen_initializations"]):
        initial = pack_s1_parameters(
            entry["raw3_percent"],
            t_c=entry["t_c"],
            path_length_m=entry["path_length_m"],
            h_rh=entry["h_rh"],
            common_delay_s=0.0,
            log_amplitude_gain=0.0,
            per_frequency_offsets=np.zeros(problem.frequencies_hz.size),
        )
        solution = solve_s1(problem, initial, spec, settings)
        initializations.append(
            {
                "initialization_index": index,
                "success": solution.success,
                "stop_reason": solution.stop_reason,
                "objective": solution.objective,
                "raw3_percent": solution.raw3_percent.tolist(),
                "iterations": len(solution.iterations),
                "forward_calls": solution.forward_calls,
            }
        )
    raw3_solutions = np.asarray([entry["raw3_percent"] for entry in initializations])
    objectives = np.asarray([entry["objective"] for entry in initializations])

    factors = np.asarray(b1["scale_invariance_factors"], dtype=np.float64)
    changed_spec = build_s1_parameterization(config, scales=spec.scales * factors)
    reference_initial = pack_s1_parameters(
        b1["frozen_initializations"][0]["raw3_percent"],
        t_c=b1["frozen_initializations"][0]["t_c"],
        path_length_m=b1["frozen_initializations"][0]["path_length_m"],
        h_rh=b1["frozen_initializations"][0]["h_rh"],
        common_delay_s=0.0,
        log_amplitude_gain=0.0,
        per_frequency_offsets=np.zeros(problem.frequencies_hz.size),
    )
    reference = solve_s1(problem, reference_initial, spec, settings)
    changed = solve_s1(problem, reference_initial, changed_spec, settings)
    scale_parameter_difference = float(
        np.max(np.abs(reference.parameters - changed.parameters))
    )
    gates = b1["numerical_gates"]
    s1_verified = bool(
        derivative_error <= float(gates["max_directional_derivative_relative_error"])
        and all(entry["success"] for entry in initializations)
        and float(np.max(np.ptp(raw3_solutions, axis=0)))
        <= float(gates["max_multi_start_raw3_spread_percent"])
        and float(np.ptp(objectives)) <= float(gates["max_multi_start_objective_spread"])
        and reference.success
        and changed.success
        and scale_parameter_difference
        <= float(gates["max_scale_invariance_parameter_difference"])
    )
    methods = config["method_matrix"]
    comparison = config["comparison_contract"]
    s0_disposed = bool(
        methods["S0"]["status"] == "historical_h1_not_instantiated"
        and methods["S0"]["execution_policy"] == "non_running_historical_note"
        and methods["S0"]["formal_pairing_eligible"] is False
        and comparison["running_methods"] == ["S1", "S2", "S3"]
        and comparison["primary_comparison"] == ["S1", "S2"]
        and comparison["upper_bound_only"] == ["S3"]
        and comparison["historical_non_running_notes"] == ["S0"]
    )
    return {
        "role": b1["role"],
        "s1_verified": s1_verified,
        "s0_historical_disposition_accepted": s0_disposed,
        "b1_closed": bool(s1_verified and s0_disposed),
        "blocking_issues": []
        if s0_disposed
        else ["s0_historical_disposition_not_closed"],
        "running_methods": comparison["running_methods"],
        "primary_comparison": comparison["primary_comparison"],
        "upper_bound_only": comparison["upper_bound_only"],
        "parameter_scale_table": {
            name: {
                "scale": float(spec.scales[index]),
                "lower": float(spec.lower_bounds[index]),
                "upper": float(spec.upper_bounds[index]),
                "finite_difference_step": float(spec.finite_difference_steps[index]),
            }
            for index, name in enumerate(spec.names)
        },
        "gradient_comparison": {
            "directional_derivative_jacobian": derivative_jacobian,
            "directional_derivative_finite_difference": derivative_fd,
            "relative_error": derivative_error,
        },
        "multi_initialization_report": initializations,
        "max_multi_start_raw3_spread_percent": float(
            np.max(np.ptp(raw3_solutions, axis=0))
        ),
        "multi_start_objective_spread": float(np.ptp(objectives)),
        "scale_invariance_max_parameter_difference": scale_parameter_difference,
        "formal_data_generated": False,
    }


def run_b2_solver_core_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the nonformal B2 mechanism checks without generating registered data."""
    problem, spec, truth = build_b1_synthetic_problem(config)
    settings = build_s1_settings(config)
    varpro = build_varpro_parameterization(config, spec)
    b1 = config["b1_solver_audit"]
    b2 = config["b2_solver_core"]
    initial_entry = b1["frozen_initializations"][0]
    initial = pack_s1_parameters(
        initial_entry["raw3_percent"],
        t_c=initial_entry["t_c"],
        path_length_m=initial_entry["path_length_m"],
        h_rh=initial_entry["h_rh"],
        common_delay_s=0.0,
        log_amplitude_gain=0.0,
        per_frequency_offsets=np.zeros(problem.frequencies_hz.size),
    )
    phase_limit = float(b2["max_phase_branch_standardized_error"])
    s1 = solve_s1(problem, initial, spec, settings)
    s2 = solve_s2(
        problem,
        initial,
        spec,
        settings,
        varpro,
        max_phase_branch_standardized_error=phase_limit,
    )
    truth_linear = truth[varpro.linear_indices]
    s3 = solve_s3(
        problem,
        initial,
        spec,
        settings,
        varpro,
        truth_linear_parameters=truth_linear,
        max_phase_branch_standardized_error=phase_limit,
    )

    probe = truth[varpro.nonlinear_indices].copy()
    probe += np.asarray([0.1, -0.1, 0.02, 1e-5, 0.1])
    projected = varpro_projected_jacobian(problem, probe, spec, varpro)
    reference = np.empty_like(projected)
    step_factor = float(b2["jacobian_check_step_factor"])
    for column, full_index in enumerate(varpro.nonlinear_indices):
        step = spec.finite_difference_steps[int(full_index)] * step_factor
        plus = probe.copy()
        minus = probe.copy()
        plus[column] += step
        minus[column] -= step
        reference[:, column] = (
            evaluate_varpro(problem, plus, spec, varpro).residual
            - evaluate_varpro(problem, minus, spec, varpro).residual
        ) / (2.0 * step)
    reference *= spec.scales[varpro.nonlinear_indices][np.newaxis, :]
    jacobian_relative_error = float(
        np.max(np.abs(projected - reference)) / max(float(np.max(np.abs(reference))), 1.0)
    )

    negative_controls: dict[str, dict[str, Any]] = {}

    def record_failure(name: str, action: Any) -> None:
        try:
            action()
        except (ValueError, np.linalg.LinAlgError) as exc:
            negative_controls[name] = {"failed_as_required": True, "error": str(exc)}
        else:
            negative_controls[name] = {"failed_as_required": False, "error": None}

    wrong_branches = problem.phase_branch_cycles.copy()
    # Stress the highest-frequency branch, which has the smallest margin to the 8σ gate.
    wrong_branches[-1] += 1
    record_failure(
        "wrong_phase_branch",
        lambda: solve_s2(
            S1Problem(
                observation=problem.observation,
                covariance=problem.covariance,
                frequencies_hz=problem.frequencies_hz,
                phase_branch_cycles=wrong_branches,
                observation_std=problem.observation_std,
                p_mpa=problem.p_mpa,
            ),
            initial,
            spec,
            settings,
            varpro,
            max_phase_branch_standardized_error=phase_limit,
        ),
    )
    unshared_config = copy.deepcopy(config)
    unshared_config["b2_solver_core"]["per_frequency_offset_scope"] = (
        "per_sample_unconstrained"
    )
    record_failure(
        "unconstrained_per_sample_k4_offsets",
        lambda: build_varpro_parameterization(unshared_config, spec),
    )
    invalid_covariance = problem.covariance.copy()
    invalid_covariance[0, 0] = -1.0
    record_failure(
        "non_positive_definite_covariance",
        lambda: solve_s2(
            S1Problem(
                observation=problem.observation,
                covariance=invalid_covariance,
                frequencies_hz=problem.frequencies_hz,
                phase_branch_cycles=problem.phase_branch_cycles,
                observation_std=problem.observation_std,
                p_mpa=problem.p_mpa,
            ),
            initial,
            spec,
            settings,
            varpro,
            max_phase_branch_standardized_error=phase_limit,
        ),
    )
    path_linear_config = copy.deepcopy(config)
    nonlinear_names = path_linear_config["b2_solver_core"]["nonlinear_parameter_names"]
    linear_names = path_linear_config["b2_solver_core"]["linear_parameter_names"]
    nonlinear_names[nonlinear_names.index("path_length_m")] = "common_delay_s"
    linear_names[linear_names.index("common_delay_s")] = "path_length_m"
    record_failure(
        "path_length_forced_into_linear_block",
        lambda: build_varpro_parameterization(path_linear_config, spec),
    )

    objective_difference = abs(s1.objective - s2.objective)
    parameter_difference = float(np.max(np.abs(s1.parameters - s2.parameters)))
    raw3_difference = float(np.max(np.abs(s1.raw3_percent - s2.raw3_percent)))
    gates = b2["numerical_gates"]
    required_controls = set(b2["required_negative_controls"])
    core_verified = bool(
        s1.success
        and s2.success
        and s3.success
        and jacobian_relative_error
        <= float(gates["max_projected_jacobian_relative_error"])
        and objective_difference <= float(gates["max_s1_s2_objective_difference"])
        and parameter_difference <= float(gates["max_s1_s2_parameter_difference"])
        and raw3_difference <= float(gates["max_s1_s2_raw3_difference_percent"])
        and set(negative_controls) == required_controls
        and all(entry["failed_as_required"] for entry in negative_controls.values())
    )
    return {
        "role": b2["role"],
        "solver_core_verified": core_verified,
        "projected_jacobian": {
            "max_relative_error": jacobian_relative_error,
            "shape": list(projected.shape),
        },
        "s1_s2_equivalence": {
            "s1_success": s1.success,
            "s2_success": s2.success,
            "objective_difference": objective_difference,
            "max_parameter_difference": parameter_difference,
            "max_raw3_difference_percent": raw3_difference,
            "s1_objective": s1.objective,
            "s2_objective": s2.objective,
        },
        "s3_upper_bound": {
            "success": s3.success,
            "objective": s3.objective,
            "truth_linear_parameters_explicit": True,
            "formal_pairing_eligible": False,
        },
        "negative_controls": negative_controls,
        "formal_data_generated": False,
    }


def _with_option_a_priors(
    spec: S1Parameterization, truth: np.ndarray
) -> S1Parameterization:
    """Join tight calibration posteriors for delay/gain as if Option A had run."""
    names = list(spec.names)
    delay_index = names.index("common_delay_s")
    gain_index = names.index("log_amplitude_gain")
    prior_indices = list(spec.prior_indices)
    prior_mean = spec.prior_mean.copy()
    prior_std = spec.prior_std.copy()
    index_map = {int(index): row for row, index in enumerate(prior_indices)}
    prior_mean[index_map[delay_index]] = float(truth[delay_index])
    prior_std[index_map[delay_index]] = 1e-7
    prior_mean[index_map[gain_index]] = float(truth[gain_index])
    prior_std[index_map[gain_index]] = 1e-3
    return S1Parameterization(
        names=spec.names,
        scales=spec.scales.copy(),
        lower_bounds=spec.lower_bounds.copy(),
        upper_bounds=spec.upper_bounds.copy(),
        finite_difference_steps=spec.finite_difference_steps.copy(),
        prior_indices=spec.prior_indices.copy(),
        prior_mean=prior_mean,
        prior_std=prior_std,
    )


def composition_crb_o2_std(
    problem: S1Problem,
    parameters: Sequence[float],
    spec: S1Parameterization,
) -> float:
    """Return the O2 CRB standard deviation from the whitened augmented Jacobian."""
    values = np.asarray(parameters, dtype=np.float64)
    jacobian_scaled = finite_difference_jacobian(problem, values, spec)
    jacobian_physical = jacobian_scaled / spec.scales[np.newaxis, :]
    fisher = jacobian_physical.T @ jacobian_physical
    covariance = np.linalg.inv(0.5 * (fisher + fisher.T))
    tangent_cov = covariance[:2, :2]
    o2_gradient = RAW3_TANGENT_BASIS[1, :]
    variance = float(o2_gradient @ tangent_cov @ o2_gradient)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("composition CRB for O2 is not positive")
    return float(np.sqrt(variance))


def run_pre_b4_technical_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Nonformal pre-B4 checks: Option A recovery, CRB, and recordable bound failures."""
    problem, spec, truth = build_b1_synthetic_problem(config)
    settings = build_s1_settings(config)
    varpro = build_varpro_parameterization(config, spec)
    b1 = config["b1_solver_audit"]
    b2 = config["b2_solver_core"]
    initial_entry = b1["frozen_initializations"][0]
    initial = pack_s1_parameters(
        initial_entry["raw3_percent"],
        t_c=initial_entry["t_c"],
        path_length_m=initial_entry["path_length_m"],
        h_rh=initial_entry["h_rh"],
        common_delay_s=0.0,
        log_amplitude_gain=0.0,
        per_frequency_offsets=np.zeros(problem.frequencies_hz.size),
    )
    phase_limit = float(b2["max_phase_branch_standardized_error"])
    truth_raw3 = raw3_percent_from_tangent(truth[:2])
    truth_o2 = float(truth_raw3[1])

    wide_s1 = solve_s1(problem, initial, spec, settings)
    wide_s2 = solve_s2(
        problem,
        initial,
        spec,
        settings,
        varpro,
        max_phase_branch_standardized_error=phase_limit,
    )
    option_a_spec = _with_option_a_priors(spec, truth)
    option_a_s1 = solve_s1(problem, initial, option_a_spec, settings)
    option_a_s2 = solve_s2(
        problem,
        initial,
        option_a_spec,
        settings,
        varpro,
        max_phase_branch_standardized_error=phase_limit,
    )
    s3 = solve_s3(
        problem,
        initial,
        option_a_spec,
        settings,
        varpro,
        truth_linear_parameters=truth[varpro.linear_indices],
        max_phase_branch_standardized_error=phase_limit,
    )
    crb_o2 = composition_crb_o2_std(problem, truth, option_a_spec)

    def o2_error(solution: S1Solution | VarProSolution) -> float:
        return float(solution.raw3_percent[1] - truth_o2)

    wide_s1_error = o2_error(wide_s1)
    wide_s2_error = o2_error(wide_s2)
    option_a_s1_error = o2_error(option_a_s1)
    option_a_s2_error = o2_error(option_a_s2)
    s3_error = o2_error(s3)

    # Bound-failure recording: shrink gain bounds so truth gain is outside them.
    tight_bounds = spec.upper_bounds.copy()
    lower_bounds = spec.lower_bounds.copy()
    gain_index = list(spec.names).index("log_amplitude_gain")
    lower_bounds[gain_index] = -0.005
    tight_bounds[gain_index] = 0.005
    bound_spec = S1Parameterization(
        names=spec.names,
        scales=spec.scales.copy(),
        lower_bounds=lower_bounds,
        upper_bounds=tight_bounds,
        finite_difference_steps=spec.finite_difference_steps.copy(),
        prior_indices=spec.prior_indices.copy(),
        prior_mean=spec.prior_mean.copy(),
        prior_std=spec.prior_std.copy(),
    )
    bound_s1 = solve_s1(problem, initial, bound_spec, settings)
    bound_s2 = solve_s2(
        problem,
        initial,
        bound_spec,
        settings,
        varpro,
        max_phase_branch_standardized_error=phase_limit,
    )

    recovery_gate = 0.5
    bound_recording_ok = (
        bound_s1.bound_hit
        and (bound_s2.bound_hit or bound_s2.success is False)
        and bound_s1.stop_reason is not None
        and bound_s2.stop_reason is not None
    )
    technical_ready = bool(
        wide_s1.success
        and wide_s2.success
        and option_a_s1.success
        and option_a_s2.success
        and s3.success
        and abs(option_a_s1_error) <= recovery_gate
        and abs(option_a_s2_error) <= recovery_gate
        and abs(s3_error) <= 1e-6
        and abs(option_a_s1_error) < abs(wide_s1_error)
        and abs(option_a_s2_error) < abs(wide_s2_error)
        and bound_recording_ok
        and crb_o2 > 0.0
    )
    return {
        "role": "in_memory_nonformal_pre_b4_technical_readiness_only",
        "technical_ready": technical_ready,
        "truth_recovery": {
            "wide_prior_o2_error_pp": {
                "S1": wide_s1_error,
                "S2": wide_s2_error,
            },
            "option_a_o2_error_pp": {
                "S1": option_a_s1_error,
                "S2": option_a_s2_error,
                "S3": s3_error,
            },
            "recovery_gate_pp": recovery_gate,
            "option_a_policy": "calibrate_on_calibration_split_join_posterior_as_prior_on_evaluation",
        },
        "relative_crb": {
            "crb_o2_std_percent": crb_o2,
            "option_a_s1_abs_error_over_crb": abs(option_a_s1_error) / crb_o2,
            "option_a_s2_abs_error_over_crb": abs(option_a_s2_error) / crb_o2,
            "option_a_s1_relative_efficiency": (crb_o2**2)
            / max(option_a_s1_error**2, 1e-30),
            "option_a_s2_relative_efficiency": (crb_o2**2)
            / max(option_a_s2_error**2, 1e-30),
        },
        "bound_failure_recording": {
            "s1_success": bound_s1.success,
            "s2_success": bound_s2.success,
            "s1_stop_reason": bound_s1.stop_reason,
            "s2_stop_reason": bound_s2.stop_reason,
            "s1_bound_hit": bound_s1.bound_hit,
            "s2_bound_hit": bound_s2.bound_hit,
            "raised_exception": False,
        },
        "formal_data_generated": False,
        "formal_solver_gate_ready": False,
        "formal_solver_gate_blocker": (
            "registered_sparse_simulation_generation_forbidden_pending_independent_authorization"
            if technical_ready
            else "pre_b4_technical_checks_failed"
        ),
    }


__all__ = [
    "ConditionalLinearSolution",
    "S1Iteration",
    "S1Parameterization",
    "S1Problem",
    "S1Solution",
    "S1SolverSettings",
    "SolverDomainError",
    "VarProEvaluation",
    "VarProParameterization",
    "VarProSolution",
    "augmented_residual",
    "build_b1_synthetic_problem",
    "build_s1_parameterization",
    "build_s1_settings",
    "build_varpro_parameterization",
    "composition_crb_o2_std",
    "evaluate_varpro",
    "finite_difference_jacobian",
    "pack_s1_parameters",
    "predict_s1",
    "run_b1_s1_numerical_audit",
    "run_b2_solver_core_audit",
    "run_pre_b4_technical_audit",
    "solve_conditionally_linear",
    "solve_s1",
    "solve_s2",
    "solve_s3",
    "validate_phase_branch_consistency",
    "varpro_projected_jacobian",
]
