"""MEI-4 parameter-space posterior primitives for the frozen MRS S1 model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.special import ndtri
from scipy.stats import genpareto, qmc

from tv3.sim.generation.tunnel_ventilation.mrs_observation import (
    RAW3_SIMPLEX_CENTER,
    RAW3_TANGENT_BASIS,
    raw3_tangent_coordinates,
)


class PosteriorConstructionError(ValueError):
    """Raised when a registered posterior construction invariant is violated."""


@dataclass(frozen=True)
class GaussianTangentPosterior:
    mean_z: np.ndarray
    covariance_z: np.ndarray
    curvature: np.ndarray
    covariance_standardized: np.ndarray
    condition_number: float


@dataclass(frozen=True)
class TruncatedTangentSamples:
    z: np.ndarray
    raw3_percent: np.ndarray
    accepted: int
    candidates: int

    @property
    def mass_estimate(self) -> float:
        return float(self.accepted / self.candidates)


@dataclass(frozen=True)
class PsisWeights:
    normalized: np.ndarray
    k_hat: float
    tail_size: int


_AUDIT_ONLY_FIELDS = frozenset(
    {
        "x_CO2_percent",
        "x_O2_percent",
        "x_N2_percent",
        "truth_raw3_percent",
        "crb_o2_std_percent",
        "s3_truth_nuisance",
    }
)


def raw3_from_tangent(z: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(z, dtype=np.float64)
    if values.shape == (2,):
        return RAW3_SIMPLEX_CENTER + RAW3_TANGENT_BASIS @ values
    if values.ndim == 2 and values.shape[1] == 2:
        return RAW3_SIMPLEX_CENTER + values @ RAW3_TANGENT_BASIS.T
    raise PosteriorConstructionError("tangent coordinates must have shape (2,) or (n, 2)")


def tangent_from_raw3(raw3_percent: Sequence[float]) -> np.ndarray:
    return raw3_tangent_coordinates(raw3_percent)


def standard_normal_quantiles(unit_hypercube: np.ndarray) -> np.ndarray:
    """Map Sobol points to finite standard-normal quantiles."""
    values = np.asarray(unit_hypercube, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise PosteriorConstructionError("Sobol points must lie in the closed unit hypercube")
    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    return ndtri(np.clip(values, lower, upper))


def _validate_curvature(
    curvature: np.ndarray,
    *,
    minimum_eigenvalue: float,
    maximum_condition_number: float,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(curvature, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise PosteriorConstructionError("curvature must be a square matrix")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise PosteriorConstructionError("curvature must be finite and symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(eigenvalues[0]) <= minimum_eigenvalue:
        raise PosteriorConstructionError("curvature_not_positive_definite")
    condition_number = float(eigenvalues[-1] / eigenvalues[0])
    if condition_number > maximum_condition_number:
        raise PosteriorConstructionError("curvature_condition_number_exceeded")
    return eigenvalues, condition_number


def schur_marginal_covariance(
    curvature: np.ndarray,
    *,
    composition_scales: Sequence[float],
    minimum_eigenvalue: float = 1e-12,
    maximum_condition_number: float = 1e12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return physical-z covariance and standardized full covariance from J.T @ J."""
    matrix = np.asarray(curvature, dtype=np.float64)
    _eigenvalues, condition_number = _validate_curvature(
        matrix,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_condition_number=maximum_condition_number,
    )
    if matrix.shape[0] < 2:
        raise PosteriorConstructionError("curvature must include two composition coordinates")
    covariance_standardized = np.linalg.inv(matrix)
    if matrix.shape[0] == 2:
        marginal_standardized = covariance_standardized
    else:
        h_zz = matrix[:2, :2]
        h_zn = matrix[:2, 2:]
        h_nn = matrix[2:, 2:]
        marginal_standardized = np.linalg.inv(h_zz - h_zn @ np.linalg.solve(h_nn, h_zn.T))
    # The inverse-vs-Schur comparison has O(eps * condition_number) rounding error.
    schur_rtol = max(1e-9, 32.0 * np.finfo(np.float64).eps * condition_number)
    if not np.allclose(
        marginal_standardized,
        covariance_standardized[:2, :2],
        rtol=schur_rtol,
        atol=1e-12,
    ):
        raise PosteriorConstructionError("Schur marginalization disagrees with joint covariance")
    scales = np.asarray(composition_scales, dtype=np.float64)
    if scales.shape != (2,) or np.any(scales <= 0.0):
        raise PosteriorConstructionError("composition scales must be two positive values")
    covariance_z = marginal_standardized * np.outer(scales, scales)
    return covariance_z, covariance_standardized, condition_number


def laplace_from_jacobian(
    jacobian: np.ndarray,
    *,
    mean_z: Sequence[float],
    composition_scales: Sequence[float] = (1.0, 1.0),
    minimum_eigenvalue: float = 1e-12,
    maximum_condition_number: float = 1e12,
) -> GaussianTangentPosterior:
    values = np.asarray(jacobian, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise PosteriorConstructionError("Jacobian must be finite with at least two columns")
    covariance_z, covariance_standardized, condition_number = schur_marginal_covariance(
        values.T @ values,
        composition_scales=composition_scales,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_condition_number=maximum_condition_number,
    )
    z = np.asarray(mean_z, dtype=np.float64)
    if z.shape != (2,) or not np.all(np.isfinite(z)):
        raise PosteriorConstructionError("Laplace mean_z must contain two finite values")
    return GaussianTangentPosterior(
        mean_z=z,
        covariance_z=covariance_z,
        curvature=values.T @ values,
        covariance_standardized=covariance_standardized,
        condition_number=condition_number,
    )


def sample_nonnegative_tangent_gaussian(
    posterior: GaussianTangentPosterior,
    *,
    candidates: int,
    minimum_accepted: int,
    seed: int,
) -> TruncatedTangentSamples:
    if candidates < 2 or candidates & (candidates - 1):
        raise PosteriorConstructionError("Sobol candidate count must be a power of two")
    if minimum_accepted < 1:
        raise PosteriorConstructionError("minimum accepted sample count must be positive")
    generator = qmc.Sobol(d=2, scramble=True, seed=int(seed))
    standard_normal = standard_normal_quantiles(
        generator.random_base2(int(np.log2(candidates)))
    )
    transform = np.linalg.cholesky(posterior.covariance_z)
    z = posterior.mean_z + standard_normal @ transform.T
    raw3 = raw3_from_tangent(z)
    accepted = np.all(raw3 >= 0.0, axis=1)
    accepted_z = z[accepted]
    accepted_raw3 = raw3[accepted]
    if accepted_z.shape[0] < minimum_accepted:
        raise PosteriorConstructionError("truncation_interval_numerical_failure")
    return TruncatedTangentSamples(
        z=accepted_z,
        raw3_percent=accepted_raw3,
        accepted=int(accepted_z.shape[0]),
        candidates=int(candidates),
    )


def equal_tailed_intervals(
    samples: TruncatedTangentSamples, *, levels: Sequence[float]
) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = {}
    for level in levels:
        if not 0.0 < float(level) < 1.0:
            raise PosteriorConstructionError("interval level must lie in (0, 1)")
        tail = (1.0 - float(level)) / 2.0
        result[str(float(level))] = np.quantile(
            samples.raw3_percent, [tail, 1.0 - tail], axis=0, method="linear"
        ).T.tolist()
    return result


def weighted_equal_tailed_intervals(
    raw3_percent: np.ndarray,
    weights: Sequence[float],
    *,
    levels: Sequence[float],
) -> dict[str, list[list[float]]]:
    """Construct equal-tailed component intervals from weighted posterior draws."""
    values = np.asarray(raw3_percent, dtype=np.float64)
    probabilities = np.asarray(weights, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or probabilities.shape != (values.shape[0],)
        or values.shape[0] < 2
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
    ):
        raise PosteriorConstructionError("weighted interval inputs are invalid")
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise PosteriorConstructionError("weighted interval weights must have positive mass")
    normalized = probabilities / total
    result: dict[str, list[list[float]]] = {}
    for level in levels:
        if not 0.0 < float(level) < 1.0:
            raise PosteriorConstructionError("interval level must lie in (0, 1)")
        tail = (1.0 - float(level)) / 2.0
        component_intervals: list[list[float]] = []
        for component in range(3):
            order = np.argsort(values[:, component])
            sorted_values = values[order, component]
            cumulative = np.cumsum(normalized[order])
            component_intervals.append(
                [
                    float(np.interp(tail, cumulative, sorted_values)),
                    float(np.interp(1.0 - tail, cumulative, sorted_values)),
                ]
            )
        result[str(float(level))] = component_intervals
    return result


def psis_weights(log_weights: Sequence[float]) -> PsisWeights:
    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 1 or values.size < 50 or not np.all(np.isfinite(values)):
        raise PosteriorConstructionError("PSIS requires at least 50 finite log weights")
    raw = np.exp(values - float(np.max(values)))
    order = np.argsort(raw)
    sorted_weights = raw[order].copy()
    tail_size = min(int(0.2 * raw.size), max(20, int(3.0 * np.sqrt(raw.size))))
    threshold_index = raw.size - tail_size - 1
    threshold = float(sorted_weights[threshold_index])
    excess = sorted_weights[-tail_size:] - threshold
    k_hat, _location, scale = genpareto.fit(excess, floc=0.0)
    if not np.isfinite(k_hat) or not np.isfinite(scale) or scale <= 0.0:
        raise PosteriorConstructionError("PSIS generalized Pareto fit failed")
    probabilities = (np.arange(tail_size, dtype=np.float64) + 0.5) / tail_size
    sorted_weights[-tail_size:] = threshold + genpareto.ppf(
        probabilities, k_hat, loc=0.0, scale=scale
    )
    sorted_weights = np.minimum(sorted_weights, np.mean(sorted_weights) * raw.size**0.75)
    smoothed = np.empty_like(sorted_weights)
    smoothed[order] = sorted_weights
    normalized = smoothed / np.sum(smoothed)
    return PsisWeights(normalized=normalized, k_hat=float(k_hat), tail_size=tail_size)


def require_method_payload(payload: Mapping[str, object]) -> None:
    forbidden = sorted(_AUDIT_ONLY_FIELDS.intersection(payload))
    if forbidden:
        raise PosteriorConstructionError(
            f"posterior method payload contains audit-only fields: {forbidden}"
        )


def require_fixed_method_settings(settings: Mapping[str, object]) -> None:
    banned = {"temperature_scaling", "prior_narrowing", "conformal_calibration"}
    present = sorted(banned.intersection(settings))
    if present:
        raise PosteriorConstructionError(
            f"unregistered posterior recalibration settings: {present}"
        )


__all__ = [
    "GaussianTangentPosterior",
    "PosteriorConstructionError",
    "PsisWeights",
    "TruncatedTangentSamples",
    "equal_tailed_intervals",
    "laplace_from_jacobian",
    "psis_weights",
    "raw3_from_tangent",
    "require_fixed_method_settings",
    "require_method_payload",
    "sample_nonnegative_tangent_gaussian",
    "schur_marginal_covariance",
    "tangent_from_raw3",
    "standard_normal_quantiles",
    "weighted_equal_tailed_intervals",
]
