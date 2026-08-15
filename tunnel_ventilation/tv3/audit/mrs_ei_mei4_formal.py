"""Read-only C2 evaluation of frozen MEI-4 deterministic posteriors."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import gaussian_kde, qmc

from tv3.audit.mrs_ei_b4_formal import (
    apply_known_condition_priors,
    apply_option_a_priors_to_spec,
)
from tv3.audit.mrs_ei_posterior_gate import (
    coverage_with_rejections,
    crps_from_samples,
    weighted_crps_from_samples,
)
from tv3.ml.mrs_posterior import (
    GaussianTangentPosterior,
    PosteriorConstructionError,
    equal_tailed_intervals,
    laplace_from_jacobian,
    psis_weights,
    raw3_from_tangent,
    sample_nonnegative_tangent_gaussian,
    standard_normal_quantiles,
    tangent_from_raw3,
    weighted_equal_tailed_intervals,
)
from tv3.ml.mrs_varpro import (
    S1Problem,
    augmented_residual,
    build_s1_parameterization,
    build_s1_settings,
    build_varpro_parameterization,
    finite_difference_jacobian,
    pack_s1_parameters,
    solve_s1,
    varpro_projected_jacobian,
)
from tv3.sim.generation.tunnel_ventilation.mrs_observation import RAW3_TANGENT_BASIS

COMPONENTS = ("CO2", "O2", "N2")
_METHOD_FIELDS = ("mixture_id", "split", "design_condition_id", "S1", "S2")
_FORBIDDEN_FIELDS = frozenset(
    {"sequence_id", "base_condition_id", "noise_seed_index", "noise_seed", "target_transform"}
)


@dataclass(frozen=True)
class PosteriorRecord:
    mixture_id: str
    split: str
    design_condition_id: str
    t_c: float
    p_mpa: float
    h_rh: float
    path_length_m: float
    frequencies_hz: np.ndarray
    observation: np.ndarray
    covariance: np.ndarray
    phase_branch_cycles: np.ndarray
    observation_std: Mapping[str, float]
    s1: Mapping[str, Any]
    s2: Mapping[str, Any]


@dataclass(frozen=True)
class MethodPosterior:
    method: str
    intervals: Mapping[str, list[list[float]]] | None
    raw3_samples: np.ndarray | None
    z_samples: np.ndarray | None
    weights: np.ndarray | None
    rejected: bool
    rejection_reason: str | None
    diagnostics: Mapping[str, Any]
    laplace: GaussianTangentPosterior | None
    parameter_samples: np.ndarray | None = None


def _load_value(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed(method: str, mixture_id: str, label: str) -> int:
    digest = hashlib.sha256(f"{method}|{mixture_id}|{label}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _array_summary(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9, method="linear")),
        "mean": float(np.mean(finite)),
    }


def _method_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = sorted(_FORBIDDEN_FIELDS.intersection(row))
    if forbidden:
        raise RuntimeError(f"method payload contains forbidden fields: {forbidden}")
    return {field: row[field] for field in _METHOD_FIELDS}


def load_frozen_c2_inputs(b4_dir: Path) -> tuple[list[PosteriorRecord], Mapping[str, Any], Mapping[str, Any]]:
    """Load only registered method fields; retain audit tables separately."""
    observations = _load_value(b4_dir / "registered_observations.json")
    paired_rows = _load_value(b4_dir / "paired_solutions.json")
    run_config = _load_value(b4_dir / "mei3_solver_run_config.json")
    calibration = {
        "calibration_priors": _load_value(b4_dir / "calibration_priors.json"),
        "view_nuisance_calibration_priors": _load_value(
            b4_dir / "view_nuisance_calibration_priors.json"
        ),
    }
    if not isinstance(observations, dict) or not isinstance(paired_rows, list):
        raise RuntimeError("B4 frozen observations or paired solutions have an invalid schema")
    method_rows = {
        str(row["mixture_id"]): _method_payload(row)
        for row in paired_rows
        if row.get("split") in {"test", "ood"}
    }
    mixture_rows = {str(row["mixture_id"]): row for row in observations["mixtures"]}
    covariance_rows = {
        str(row["covariance_block_id"]): row for row in observations["covariance_blocks"]
    }
    rows_by_mixture: dict[str, list[Mapping[str, Any]]] = {}
    for row in observations["observation_rows"]:
        rows_by_mixture.setdefault(str(row["mixture_id"]), []).append(row)
    observation_std = run_config["dataset_meta"]["observation_std"]
    records: list[PosteriorRecord] = []
    for mixture_id, solution in sorted(method_rows.items()):
        mixture = mixture_rows.get(mixture_id)
        rows = rows_by_mixture.get(mixture_id)
        if mixture is None or rows is None or len(rows) != 4:
            raise RuntimeError(f"B4 table join failed for mixture_id={mixture_id}")
        rows = sorted(rows, key=lambda item: float(item["frequency_hz"]))
        covariance_id = str(rows[0]["covariance_block_id"])
        if any(str(row["covariance_block_id"]) != covariance_id for row in rows):
            raise RuntimeError(f"mixture_id={mixture_id} has inconsistent covariance blocks")
        covariance_row = covariance_rows.get(covariance_id)
        if covariance_row is None:
            raise RuntimeError(f"mixture_id={mixture_id} covariance block is missing")
        records.append(
            PosteriorRecord(
                mixture_id=mixture_id,
                split=str(solution["split"]),
                design_condition_id=str(solution["design_condition_id"]),
                t_c=float(rows[0]["T_C"]),
                p_mpa=float(rows[0]["P_MPa"]),
                h_rh=float(rows[0]["H_RH"]),
                path_length_m=float(rows[0]["L_m"]),
                frequencies_hz=np.asarray([row["frequency_hz"] for row in rows], dtype=np.float64),
                observation=np.asarray(
                    [item for row in rows for item in (row["raw_tof_s"], row["log_amplitude"], row["unwrapped_phase_rad"])],
                    dtype=np.float64,
                ),
                covariance=np.asarray(covariance_row["observation_covariance"], dtype=np.float64),
                phase_branch_cycles=np.asarray(
                    [row["phase_branch_cycles"] for row in rows], dtype=np.int64
                ),
                observation_std={key: float(value) for key, value in observation_std.items()},
                s1=solution["S1"],
                s2=solution["S2"],
            )
        )
    counts = {split: sum(record.split == split for record in records) for split in ("test", "ood")}
    if counts != {"test": 648, "ood": 648}:
        raise RuntimeError(f"B4 evaluation split count mismatch: {counts}")
    return records, run_config["solver_config"], {"mixtures": mixture_rows, "paired": paired_rows, "calibration": calibration}


def _problem(record: PosteriorRecord) -> S1Problem:
    return S1Problem(
        observation=record.observation,
        covariance=record.covariance,
        frequencies_hz=record.frequencies_hz,
        phase_branch_cycles=record.phase_branch_cycles,
        observation_std=record.observation_std,
        p_mpa=record.p_mpa,
    )


def _spec_for_record(base_spec: Any, calibration: Mapping[str, Any], record: PosteriorRecord) -> Any:
    return apply_known_condition_priors(
        apply_option_a_priors_to_spec(base_spec, calibration),
        t_c=record.t_c,
        path_length_m=record.path_length_m,
        h_rh=record.h_rh,
    )


def _frozen_initialization(solver_config: Mapping[str, Any], record: PosteriorRecord, index: int) -> np.ndarray:
    initial = solver_config["b1_solver_audit"]["frozen_initializations"][index]
    return pack_s1_parameters(
        initial["raw3_percent"],
        t_c=record.t_c,
        path_length_m=record.path_length_m,
        h_rh=record.h_rh,
        common_delay_s=0.0,
        log_amplitude_gain=0.0,
        per_frequency_offsets=np.zeros(record.frequencies_hz.size),
    )


def run_s1_replay_check(
    records: Sequence[PosteriorRecord],
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    replay = contract["s1_replay_check"]
    base_spec = build_s1_parameterization(solver_config)
    settings = build_s1_settings(solver_config)
    probe_rows: list[dict[str, Any]] = []
    for split in ("test", "ood"):
        eligible = sorted(
            (record for record in records if record.split == split),
            key=lambda record: record.mixture_id,
        )
        indices = np.linspace(0, len(eligible) - 1, int(replay["mixtures_per_domain"]), dtype=int)
        for record in (eligible[int(index)] for index in indices):
            spec = _spec_for_record(base_spec, calibration, record)
            candidates = [
                solve_s1(_problem(record), _frozen_initialization(solver_config, record, index), spec, settings)
                for index in replay["initializations"]
            ]
            successful = [solution for solution in candidates if solution.success]
            selected = min(successful or candidates, key=lambda solution: solution.objective)
            frozen = record.s1
            raw3_matches = np.allclose(
                selected.raw3_percent,
                np.asarray(frozen["raw3_percent"], dtype=np.float64),
                rtol=0.0,
                atol=float(replay["raw3_percent_atol"]),
            )
            objective_matches = np.isclose(
                selected.objective,
                float(frozen["objective"]),
                rtol=float(replay["objective_rtol"]),
                atol=float(replay["objective_atol"]),
            )
            probe_rows.append(
                {
                    "mixture_id": record.mixture_id,
                    "split": split,
                    "raw3_matches": bool(raw3_matches),
                    "objective_matches": bool(objective_matches),
                    "replayed_objective": float(selected.objective),
                    "frozen_objective": float(frozen["objective"]),
                }
            )
    passed = all(row["raw3_matches"] and row["objective_matches"] for row in probe_rows)
    return {"passed": passed, "probes": probe_rows, "n_probes": len(probe_rows)}


def _truncated_laplace(
    *, method: str, mixture_id: str, posterior: GaussianTangentPosterior, contract: Mapping[str, Any]
) -> tuple[Mapping[str, list[list[float]]], np.ndarray, np.ndarray, dict[str, float]]:
    rules = contract["posterior_parameterization"]["marginal_intervals"]
    levels = contract["calibration_gate"]["nominal_levels"]
    initial = sample_nonnegative_tangent_gaussian(
        posterior,
        candidates=int(rules["initial_candidates"]),
        minimum_accepted=int(rules["minimum_accepted_candidates"]),
        # A doubled-resolution QMC check must compare nested prefixes of one scramble.
        seed=_seed(method, mixture_id, "truncation"),
    )
    verified = sample_nonnegative_tangent_gaussian(
        posterior,
        candidates=int(rules["verification_candidates"]),
        minimum_accepted=int(rules["minimum_accepted_candidates"]),
        seed=_seed(method, mixture_id, "truncation"),
    )
    initial_intervals = equal_tailed_intervals(initial, levels=levels)
    intervals = equal_tailed_intervals(verified, levels=levels)
    max_difference = float(
        np.max(np.abs(np.asarray(initial_intervals["0.95"]) - np.asarray(intervals["0.95"])))
    )
    if max_difference > float(rules["quantile_atol_percent"]):
        raise PosteriorConstructionError("truncation_interval_numerical_failure")
    return intervals, verified.raw3_percent, verified.z, {
        "truncation_mass_loss": float(1.0 - verified.mass_estimate),
        "interval_resolution_difference_percent": max_difference,
    }


def _rejected(method: str, reason: str, diagnostics: Mapping[str, Any]) -> MethodPosterior:
    return MethodPosterior(method, None, None, None, None, True, reason, diagnostics, None)


def _laplace_method(
    *, method: str, record: PosteriorRecord, spec: Any, solver_config: Mapping[str, Any], contract: Mapping[str, Any]
) -> MethodPosterior:
    try:
        if method == "M1":
            parameters = np.asarray(record.s1["parameters"], dtype=np.float64)
            jacobian = finite_difference_jacobian(_problem(record), parameters, spec)
            scales = spec.scales[:2]
        elif method == "M1b":
            parameters = np.asarray(record.s2["parameters"], dtype=np.float64)
            varpro = build_varpro_parameterization(solver_config, spec)
            if tuple(varpro.nonlinear_indices[:2]) != (0, 1):
                raise RuntimeError("S2 composition coordinates must lead the nonlinear block")
            jacobian = varpro_projected_jacobian(
                _problem(record), parameters[varpro.nonlinear_indices], spec, varpro
            )
            scales = spec.scales[varpro.nonlinear_indices[:2]]
        else:
            raise ValueError(f"unsupported Laplace method: {method}")
        curvature = contract["curvature_and_marginalization"]
        posterior = laplace_from_jacobian(
            jacobian,
            mean_z=parameters[:2],
            composition_scales=scales,
            minimum_eigenvalue=float(curvature["positive_definite_rule"]["minimum_eigenvalue"]),
            maximum_condition_number=float(curvature["positive_definite_rule"]["maximum_condition_number"]),
        )
        intervals, raw3, z, diagnostics = _truncated_laplace(
            method=method, mixture_id=record.mixture_id, posterior=posterior, contract=contract
        )
    except PosteriorConstructionError as exc:
        return _rejected(method, str(exc), {"construction_error": str(exc)})
    raw_covariance = RAW3_TANGENT_BASIS @ posterior.covariance_z @ RAW3_TANGENT_BASIS.T
    return MethodPosterior(
        method,
        intervals,
        raw3,
        z,
        None,
        False,
        None,
        {**diagnostics, "condition_number": posterior.condition_number, "laplace_std_percent": np.sqrt(np.diag(raw_covariance)).tolist()},
        posterior,
    )


def _importance_method(
    record: PosteriorRecord, spec: Any, contract: Mapping[str, Any], proposal: MethodPosterior
) -> MethodPosterior:
    if proposal.laplace is None:
        return _rejected("M2", "m1_proposal_unavailable", {"m1_rejection_reason": proposal.rejection_reason})
    policy = contract["method_policy"]["M2"]
    count = int(policy["samples_per_mixture"])
    if count < 2 or count & (count - 1):
        raise RuntimeError("M2 samples_per_mixture must be a Sobol power of two")
    mean = np.asarray(record.s1["parameters"], dtype=np.float64)
    covariance = proposal.laplace.covariance_standardized
    normal = standard_normal_quantiles(
        qmc.Sobol(d=mean.size, scramble=True, seed=_seed("M2", record.mixture_id, "proposal")).random_base2(
            int(np.log2(count))
        )
    )
    standardized = normal @ np.linalg.cholesky(covariance).T
    parameters = mean + standardized * spec.scales
    raw3 = raw3_from_tangent(parameters[:, :2])
    valid = (
        np.all(parameters >= spec.lower_bounds, axis=1)
        & np.all(parameters <= spec.upper_bounds, axis=1)
        & np.all(raw3 >= 0.0, axis=1)
    )
    valid_parameters = parameters[valid]
    valid_standardized = standardized[valid]
    valid_raw3 = raw3[valid]
    if valid_parameters.shape[0] < 50:
        return _rejected("M2", "importance_proposal_insufficient_domain_samples", {"accepted_proposal_draws": int(valid_parameters.shape[0])})
    log_target = np.asarray(
        [-0.5 * float(residual @ residual) for residual in (augmented_residual(_problem(record), value, spec) for value in valid_parameters)],
        dtype=np.float64,
    )
    log_proposal = -0.5 * np.einsum(
        "ij,jk,ik->i", valid_standardized, proposal.laplace.curvature, valid_standardized
    )
    try:
        psis = psis_weights(log_target - log_proposal)
    except PosteriorConstructionError as exc:
        return _rejected("M2", str(exc), {"construction_error": str(exc)})
    diagnostics = {
        "psis_k_hat": psis.k_hat,
        "psis_tail_size": psis.tail_size,
        "accepted_proposal_draws": int(valid_parameters.shape[0]),
        "proposal_domain_loss": float(1.0 - valid_parameters.shape[0] / count),
    }
    if psis.k_hat > float(policy["psis_k_hat_threshold"]):
        return _rejected("M2", "psis_k_hat_exceeded_for_M2", diagnostics)
    intervals = weighted_equal_tailed_intervals(
        valid_raw3, psis.normalized, levels=contract["calibration_gate"]["nominal_levels"]
    )
    return MethodPosterior(
        "M2",
        intervals,
        valid_raw3,
        valid_parameters[:, :2],
        psis.normalized,
        False,
        None,
        diagnostics,
        None,
        valid_parameters,
    )


def _flat_row(record: PosteriorRecord, posterior: MethodPosterior) -> dict[str, Any]:
    row: dict[str, Any] = {
        "mixture_id": record.mixture_id,
        "split": record.split,
        "design_condition_id": record.design_condition_id,
        "P_MPa": record.p_mpa,
        "H_RH": record.h_rh,
        "method": posterior.method,
        "rejected": posterior.rejected,
        "rejection_reason": posterior.rejection_reason or "",
    }
    for key, value in posterior.diagnostics.items():
        if isinstance(value, (int, float, str, bool)):
            row[key] = value
    for level, bounds in (posterior.intervals or {}).items():
        for component, interval in zip(COMPONENTS, bounds, strict=True):
            row[f"{component}_lower_{level}"] = interval[0]
            row[f"{component}_upper_{level}"] = interval[1]
    return row


def _density_nll(samples: np.ndarray, truth: np.ndarray, weights: np.ndarray | None) -> list[float]:
    component_nll = []
    for component in range(3):
        density = float(gaussian_kde(samples[:, component], weights=weights)([truth[component]])[0])
        if density <= 0.0 or not np.isfinite(density):
            raise RuntimeError("posterior marginal KDE density is non-positive")
        component_nll.append(float(-np.log(density)))
    return component_nll


def _joint_nll(samples: np.ndarray, truth: np.ndarray, weights: np.ndarray | None) -> float:
    density = float(gaussian_kde(samples.T, weights=weights)(truth[:, np.newaxis])[0])
    if density <= 0.0 or not np.isfinite(density):
        raise RuntimeError("posterior joint KDE density is non-positive")
    return float(-np.log(density))


def _audit_method(
    *, record: PosteriorRecord, posterior: MethodPosterior, truth: np.ndarray, crb_o2: float,
    contract: Mapping[str, Any], coverage_events: list[dict[str, Any]], metric_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]],
) -> None:
    for level in contract["calibration_gate"]["nominal_levels"]:
        bounds = (posterior.intervals or {}).get(str(float(level)))
        for component, truth_value, interval in zip(COMPONENTS, truth, bounds or [[0.0, 0.0]] * 3, strict=True):
            coverage = coverage_with_rejections([truth_value], [interval], [posterior.rejected])
            coverage_events.append(
                {
                    "method": posterior.method,
                    "split": record.split,
                    "component": component,
                    "level": str(float(level)),
                    "covered": coverage["covered"],
                    "rejected": coverage["rejected"],
                    "design_condition_id": record.design_condition_id,
                    "P_MPa": f"{record.p_mpa:g}",
                    "H_RH": f"{record.h_rh:g}",
                }
            )
    if posterior.rejected:
        diagnostic_rows.append({"method": posterior.method, "split": record.split, "rejected": True, "rejection_reason": posterior.rejection_reason, **posterior.diagnostics})
        return
    assert posterior.raw3_samples is not None and posterior.z_samples is not None
    nll = _density_nll(posterior.raw3_samples, truth, posterior.weights)
    crps = [
        (weighted_crps_from_samples if posterior.weights is not None else crps_from_samples)(
            posterior.raw3_samples[:, component], *([posterior.weights] if posterior.weights is not None else []), truth[component]
        )
        for component in range(3)
    ]
    metric_rows.append(
        {
            "method": posterior.method,
            "split": record.split,
            "nll": nll,
            "crps": crps,
            "joint_z_nll": _joint_nll(posterior.z_samples, tangent_from_raw3(truth), posterior.weights),
        }
    )
    diagnostic_rows.append(
        {
            "method": posterior.method,
            "split": record.split,
            "rejected": False,
            **posterior.diagnostics,
            "o2_laplace_to_crb_ratio": (
                float(posterior.diagnostics["laplace_std_percent"][1]) / crb_o2
                if "laplace_std_percent" in posterior.diagnostics and crb_o2 > 0.0
                else None
            ),
        }
    )


def _audit_values(audit_tables: Mapping[str, Any], mixture_id: str) -> tuple[np.ndarray, float]:
    """Read truth and CRB only after the method's posterior is constructed."""
    mixture = audit_tables["mixtures"].get(mixture_id)
    paired = next(
        (row for row in audit_tables["paired"] if row.get("mixture_id") == mixture_id),
        None,
    )
    if mixture is None or paired is None:
        raise RuntimeError(f"audit join failed for mixture_id={mixture_id}")
    return (
        np.asarray(
            [mixture["x_CO2_percent"], mixture["x_O2_percent"], mixture["x_N2_percent"]],
            dtype=np.float64,
        ),
        float(paired["crb_o2_std_percent"]),
    )


def _coverage_report(events: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    bands = contract["calibration_gate"]["exact_binomial_acceptance_counts"]
    rows: list[dict[str, Any]] = []
    for method in ("M1", "M1b", "M2"):
        for split in ("test", "ood"):
            for component in COMPONENTS:
                for level in contract["calibration_gate"]["nominal_levels"]:
                    selected = [event for event in events if event["method"] == method and event["split"] == split and event["component"] == component and event["level"] == str(float(level))]
                    covered = sum(int(event["covered"]) for event in selected)
                    rejected = sum(int(event["rejected"]) for event in selected)
                    band = bands[str(float(level))]
                    rows.append({"method": method, "domain": split, "component": component, "nominal_level": level, "n": len(selected), "covered": covered, "rejected": rejected, "coverage": covered / len(selected), "acceptance_band": band, "within_acceptance_band": band["lower_inclusive"] <= covered <= band["upper_inclusive"]})
    return {"status": "mei4_c2_intermediate_no_pass_verdict", "primary_bands": rows, "sbc_and_ppc": "not_run_requires_mei4_observation_space_authorization"}


def _group_coverage_report(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reports: dict[str, list[dict[str, Any]]] = {}
    for group in ("design_condition_id", "P_MPa", "H_RH"):
        grouped: dict[tuple[str, str, str, str, str], list[Mapping[str, Any]]] = {}
        for event in events:
            key = (str(event["method"]), str(event["split"]), str(event["component"]), str(event["level"]), str(event[group]))
            grouped.setdefault(key, []).append(event)
        reports[group] = [
            {"method": key[0], "domain": key[1], "component": key[2], "nominal_level": float(key[3]), "group": key[4], "n": len(values), "covered": sum(int(value["covered"]) for value in values), "rejected": sum(int(value["rejected"]) for value in values), "coverage": sum(int(value["covered"]) for value in values) / len(values)}
            for key, values in sorted(grouped.items())
        ]
    return {"group_reports": reports, "role": "diagnostic_only_not_a_calibration_gate"}


def _nll_crps_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"methods": {}}
    for method in ("M1", "M1b", "M2"):
        report["methods"][method] = {}
        for split in ("test", "ood"):
            selected = [row for row in rows if row["method"] == method and row["split"] == split]
            component_rows = {component: _array_summary([row["nll"][index] for row in selected]) | {"crps": _array_summary([row["crps"][index] for row in selected])} for index, component in enumerate(COMPONENTS)}
            report["methods"][method][split] = {"n_evaluable": len(selected), "components": component_rows, "joint_z_nll": _array_summary([row["joint_z_nll"] for row in selected])}
    return report


def _full_hessian_probe(record: PosteriorRecord, spec: Any) -> dict[str, Any]:
    parameters = np.asarray(record.s1["parameters"], dtype=np.float64)
    try:
        base = finite_difference_jacobian(_problem(record), parameters, spec)
        gauss_newton = base.T @ base

        def gradient(values: np.ndarray) -> np.ndarray:
            residual = augmented_residual(_problem(record), values, spec)
            return finite_difference_jacobian(_problem(record), values, spec).T @ residual

        center = gradient(parameters)
        hessian = np.empty_like(gauss_newton)
        for index, step in enumerate(spec.finite_difference_steps):
            delta = np.zeros_like(parameters)
            delta[index] = step
            plus_ok = np.all(parameters + delta <= spec.upper_bounds)
            minus_ok = np.all(parameters - delta >= spec.lower_bounds)
            if plus_ok and minus_ok:
                column = (gradient(parameters + delta) - gradient(parameters - delta)) / (2.0 * step / spec.scales[index])
            elif plus_ok:
                column = (gradient(parameters + delta) - center) / (step / spec.scales[index])
            elif minus_ok:
                column = (center - gradient(parameters - delta)) / (step / spec.scales[index])
            else:
                raise RuntimeError("complete Hessian finite difference left the frozen domain")
            hessian[:, index] = column
        hessian = 0.5 * (hessian + hessian.T)
        return {"mixture_id": record.mixture_id, "split": record.split, "status": "computed", "relative_frobenius_difference": float(np.linalg.norm(hessian - gauss_newton, ord="fro") / np.linalg.norm(hessian, ord="fro"))}
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        return {"mixture_id": record.mixture_id, "split": record.split, "status": "unavailable", "error": str(exc)}


def run_c2_deterministic_posterior(
    *, b4_dir: Path, contract: Mapping[str, Any], progress_callback: Callable[[int, int, str], None] | None = None
) -> dict[str, Any]:
    records, solver_config, audit_tables = load_frozen_c2_inputs(b4_dir)
    calibration = audit_tables["calibration"]
    replay = run_s1_replay_check(records, solver_config, calibration, contract)
    if not replay["passed"]:
        raise RuntimeError("S1 replay consistency check failed; C2 must stop before posterior evaluation")
    base_spec = build_s1_parameterization(solver_config)
    flat_by_domain = {"test": [], "ood": []}
    coverage_events: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        spec = _spec_for_record(base_spec, calibration, record)
        m1 = _laplace_method(method="M1", record=record, spec=spec, solver_config=solver_config, contract=contract)
        m1b = _laplace_method(method="M1b", record=record, spec=spec, solver_config=solver_config, contract=contract)
        m2 = _importance_method(record, spec, contract, m1)
        truth, crb = _audit_values(audit_tables, record.mixture_id)
        for posterior in (m1, m1b, m2):
            flat_by_domain[record.split].append(_flat_row(record, posterior))
            _audit_method(record=record, posterior=posterior, truth=truth, crb_o2=crb, contract=contract, coverage_events=coverage_events, metric_rows=metric_rows, diagnostic_rows=diagnostic_rows)
        if progress_callback is not None:
            progress_callback(index, len(records), record.mixture_id)
    hessian_rows = []
    count = int(contract["curvature_and_marginalization"]["complete_hessian_check"]["mixtures_per_domain"])
    for split in ("test", "ood"):
        eligible = sorted(
            (record for record in records if record.split == split),
            key=lambda record: record.mixture_id,
        )
        for position in np.linspace(0, len(eligible) - 1, count, dtype=int):
            record = eligible[int(position)]
            hessian_rows.append(_full_hessian_probe(record, _spec_for_record(base_spec, calibration, record)))
    diagnostics: dict[str, Any] = {"s1_replay": replay, "complete_hessian": hessian_rows, "methods": {}}
    for method in ("M1", "M1b", "M2"):
        diagnostics["methods"][method] = {}
        for split in ("test", "ood"):
            selected = [row for row in diagnostic_rows if row["method"] == method and row["split"] == split]
            diagnostics["methods"][method][split] = {
                "n": len(selected),
                "rejected": sum(bool(row["rejected"]) for row in selected),
                "condition_number": _array_summary([row.get("condition_number", np.nan) for row in selected]),
                "truncation_mass_loss": _array_summary([row.get("truncation_mass_loss", np.nan) for row in selected]),
                "o2_laplace_to_crb_ratio": _array_summary([row.get("o2_laplace_to_crb_ratio", np.nan) for row in selected]),
                "psis_k_hat": _array_summary([row.get("psis_k_hat", np.nan) for row in selected]),
                "rejection_reasons": {
                    reason: sum(row.get("rejection_reason") == reason for row in selected)
                    for reason in sorted(
                        {
                            str(row["rejection_reason"])
                            for row in selected
                            if row.get("rejection_reason")
                        }
                    )
                },
            }
    return {
        "status": "mei4_c2_deterministic_evaluation_complete",
        "posterior_intervals_test": flat_by_domain["test"],
        "posterior_intervals_ood": flat_by_domain["ood"],
        "coverage_report": _coverage_report(coverage_events, contract),
        "nll_crps_report": _nll_crps_report(metric_rows),
        "group_coverage_report": _group_coverage_report(coverage_events),
        "laplace_diagnostics": diagnostics,
    }


__all__ = ["load_frozen_c2_inputs", "run_c2_deterministic_posterior", "run_s1_replay_check"]
