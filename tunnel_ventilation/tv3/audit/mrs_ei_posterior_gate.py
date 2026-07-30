"""MEI-4 posterior calibration estimators and synthetic C1 audit."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binom, kstest, norm

from tv3.ml.mrs_posterior import (
    PosteriorConstructionError,
    equal_tailed_intervals,
    laplace_from_jacobian,
    psis_weights,
    require_fixed_method_settings,
    require_method_payload,
    sample_nonnegative_tangent_gaussian,
    tangent_from_raw3,
)
from tv3.ml.mrs_varpro import S1Problem, validate_phase_branch_consistency
from tv3.sim.generation.tunnel_ventilation.mrs_observation import RAW3_TANGENT_BASIS


def exact_binomial_acceptance_band(
    *, n: int, nominal_coverage: float, alpha: float
) -> dict[str, int]:
    if n < 1 or not 0.0 < nominal_coverage < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("binomial band arguments are invalid")
    return {
        "lower_inclusive": int(binom.ppf(alpha / 2.0, n, nominal_coverage)),
        "upper_inclusive": int(binom.isf(alpha / 2.0, n, nominal_coverage)),
    }


def coverage_with_rejections(
    truth: Sequence[float],
    intervals: Sequence[Sequence[float]],
    rejected: Sequence[bool],
) -> dict[str, Any]:
    values = np.asarray(truth, dtype=np.float64)
    bounds = np.asarray(intervals, dtype=np.float64)
    rejection = np.asarray(rejected, dtype=bool)
    if bounds.shape != (values.size, 2) or rejection.shape != values.shape:
        raise ValueError("coverage inputs have incompatible shapes")
    covered = (~rejection) & (values >= bounds[:, 0]) & (values <= bounds[:, 1])
    return {
        "n": int(values.size),
        "covered": int(np.sum(covered)),
        "rejected": int(np.sum(rejection)),
        "coverage": float(np.mean(covered)),
    }


def gaussian_nll(value: float, *, mean: float, std: float) -> float:
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError("Gaussian standard deviation must be positive")
    return float(-norm.logpdf(float(value), loc=float(mean), scale=float(std)))


def crps_from_samples(samples: Sequence[float], truth: float) -> float:
    values = np.sort(np.asarray(samples, dtype=np.float64))
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("CRPS requires at least two finite samples")
    n = values.size
    pairwise_mean = float(np.sum((2 * np.arange(n) - n + 1) * values) / (n * n))
    return float(np.mean(np.abs(values - float(truth))) - pairwise_mean)


def sbc_uniformity(ranks: Sequence[int], *, posterior_draws: int) -> dict[str, float | bool]:
    values = np.asarray(ranks, dtype=np.int64)
    if values.ndim != 1 or values.size == 0 or np.any(values < 0) or np.any(values > posterior_draws):
        raise ValueError("SBC ranks are outside the registered range")
    scaled = (values + 0.5) / (posterior_draws + 1.0)
    statistic, pvalue = kstest(scaled, "uniform")
    return {"statistic": float(statistic), "pvalue": float(pvalue)}


def _invalid_phase_problem() -> S1Problem:
    frequencies = np.asarray([25000.0, 63000.0, 100000.0, 200000.0])
    observation = np.zeros(12, dtype=np.float64)
    observation[0::3] = 1e-3
    covariance = np.eye(12, dtype=np.float64)
    covariance[0::3, 0::3] = np.eye(4) * 1e-12
    return S1Problem(
        observation=observation,
        covariance=covariance,
        frequencies_hz=frequencies,
        phase_branch_cycles=np.zeros(4, dtype=np.int64),
        observation_std={"raw_tof_s": 1.0, "log_amplitude": 1.0, "unwrapped_phase_rad": 1.0},
        p_mpa=0.101325,
    )


def _must_fail(action) -> str:
    try:
        action()
    except (PosteriorConstructionError, ValueError):
        return "explicit_failure"
    raise AssertionError("negative control unexpectedly succeeded")


def run_posterior_core_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run C1 entirely from deterministic in-memory fixtures."""
    rng = np.random.default_rng(int(config["fixture_seed"]))
    linear = config["linear_gaussian"]
    jacobian = np.asarray([[1.0, 0.2], [0.1, 1.2], [0.7, -0.3]], dtype=np.float64)
    posterior = laplace_from_jacobian(jacobian, mean_z=(0.0, 0.0))
    analytic_covariance = np.linalg.inv(jacobian.T @ jacobian)
    covariance_error = float(np.max(np.abs(posterior.covariance_z - analytic_covariance)))
    schur_jacobian = rng.normal(size=(12, 5))
    schur_posterior = laplace_from_jacobian(
        schur_jacobian,
        mean_z=(0.0, 0.0),
        composition_scales=(0.5, 2.0),
    )
    schur_direct = np.linalg.inv(schur_jacobian.T @ schur_jacobian)[:2, :2]
    schur_expected = schur_direct * np.outer([0.5, 2.0], [0.5, 2.0])
    schur_error = float(np.max(np.abs(schur_posterior.covariance_z - schur_expected)))
    count = int(linear["simulation_count"])
    level = float(linear["nominal_coverage"])
    latent = rng.normal(size=count)
    observation = latent + rng.normal(size=count)
    z_value = float(norm.ppf((1.0 + level) / 2.0))
    covered = np.abs(latent - observation) <= z_value
    coverage_count = int(np.sum(covered))
    band = exact_binomial_acceptance_band(
        n=count, nominal_coverage=level, alpha=float(linear["binomial_alpha"])
    )
    truncation = config["truncation"]
    interior_mean = tangent_from_raw3(truncation["interior_mean_raw3_percent"])
    interior = laplace_from_jacobian(
        np.linalg.cholesky(np.linalg.inv(np.asarray(truncation["interior_covariance_z"]))).T,
        mean_z=interior_mean,
    )
    interior_samples = sample_nonnegative_tangent_gaussian(
        interior,
        candidates=int(truncation["candidates"]),
        minimum_accepted=int(truncation["minimum_accepted_candidates"]),
        seed=int(config["fixture_seed"]) + 1,
    )
    interval_levels = (0.5, 0.8, 0.9, 0.95)
    interior_intervals = equal_tailed_intervals(interior_samples, levels=interval_levels)
    analytic_raw_covariance = (
        RAW3_TANGENT_BASIS
        @ np.asarray(truncation["interior_covariance_z"])
        @ RAW3_TANGENT_BASIS.T
    )
    interval_error = 0.0
    for interval_level in interval_levels:
        z_interval = float(norm.ppf((1.0 + interval_level) / 2.0))
        expected_quantiles = np.column_stack(
            (
                np.asarray(truncation["interior_mean_raw3_percent"]) - z_interval * np.sqrt(np.diag(analytic_raw_covariance)),
                np.asarray(truncation["interior_mean_raw3_percent"]) + z_interval * np.sqrt(np.diag(analytic_raw_covariance)),
            )
        )
        interval_error = max(
            interval_error,
            float(np.max(np.abs(np.asarray(interior_intervals[str(interval_level)]) - expected_quantiles))),
        )
    boundary = laplace_from_jacobian(
        np.linalg.cholesky(np.linalg.inv(np.asarray(truncation["boundary_covariance_z"]))).T,
        mean_z=tangent_from_raw3(truncation["boundary_mean_raw3_percent"]),
    )
    boundary_samples = sample_nonnegative_tangent_gaussian(
        boundary,
        candidates=int(truncation["candidates"]),
        minimum_accepted=int(truncation["minimum_accepted_candidates"]),
        seed=int(config["fixture_seed"]) + 2,
    )
    ranks = rng.integers(0, int(config["sbc"]["posterior_draws"]) + 1, size=int(config["sbc"]["replicates"]))
    sbc = sbc_uniformity(ranks, posterior_draws=int(config["sbc"]["posterior_draws"]))
    heavy_tail = np.log1p(rng.pareto(1.0 / 0.9, size=int(config["psis"]["tail_size"])))
    psis = psis_weights(heavy_tail)
    negative_controls = {
        "truth_field": _must_fail(lambda: require_method_payload({"truth_raw3_percent": [1, 2, 97]})),
        "nonpositive_curvature": _must_fail(lambda: laplace_from_jacobian(np.asarray([[1.0, 0.0], [0.0, 0.0]]), mean_z=(0.0, 0.0))),
        "phase_branch": _must_fail(lambda: validate_phase_branch_consistency(_invalid_phase_problem(), max_standardized_error=float(config["phase_branch_max_standardized_error"]))),
        "recalibration": _must_fail(lambda: require_fixed_method_settings({"temperature_scaling": 1.0})),
    }
    rejection_coverage = coverage_with_rejections(
        [0.0, 0.0, 0.0],
        [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
        [False, True, False],
    )
    nll = gaussian_nll(0.0, mean=0.0, std=1.0)
    crps = crps_from_samples(rng.normal(size=4096), 0.0)
    metrics = {
        "nll": nll,
        "nll_analytic": 0.5 * float(np.log(2.0 * np.pi)),
        "crps": crps,
        "crps_analytic": float((np.sqrt(2.0) - 1.0) / np.sqrt(np.pi)),
        "rejection_coverage": rejection_coverage,
        "sbc": sbc,
    }
    passed = (
        covariance_error <= float(linear["covariance_atol"])
        and schur_error <= float(linear["schur_covariance_atol"])
        and band["lower_inclusive"] <= coverage_count <= band["upper_inclusive"]
        and interval_error <= float(truncation["interior_quantile_atol_percent"])
        and 1.0 - boundary_samples.mass_estimate >= float(truncation["minimum_boundary_mass_loss"])
        and abs(nll - metrics["nll_analytic"]) <= float(config["estimator_checks"]["nll_atol"])
        and abs(crps - metrics["crps_analytic"]) <= float(config["estimator_checks"]["crps_monte_carlo_atol"])
        and rejection_coverage["coverage"] == float(config["estimator_checks"]["rejection_coverage_expected"])
        and sbc["pvalue"] >= float(config["sbc"]["alpha"])
        and psis.k_hat >= float(config["psis"]["required_minimum_k_hat"])
        and all(value == "explicit_failure" for value in negative_controls.values())
    )
    return {
        "status": "mei4_posterior_core_verified" if passed else "mei4_posterior_core_invalid",
        "passed": passed,
        "linear_gaussian": {"covariance_max_abs_error": covariance_error, "schur_covariance_max_abs_error": schur_error, "coverage_count": coverage_count, "acceptance_band": band},
        "truncation": {"interior_mass": interior_samples.mass_estimate, "boundary_mass": boundary_samples.mass_estimate, "interior_max_quantile_error_percent": interval_error, "interior_intervals": interior_intervals},
        "metrics": metrics,
        "psis": {"k_hat": psis.k_hat, "tail_size": psis.tail_size},
        "negative_controls": negative_controls,
    }


__all__ = [
    "coverage_with_rejections",
    "crps_from_samples",
    "exact_binomial_acceptance_band",
    "gaussian_nll",
    "run_posterior_core_audit",
    "sbc_uniformity",
]
