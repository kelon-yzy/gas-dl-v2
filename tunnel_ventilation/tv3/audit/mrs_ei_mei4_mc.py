"""Authorized MEI-4 C3 observation-space calibration calculations.

This module deliberately returns aggregate evidence only.  Synthetic observations
and their generating truth stay in memory and are never promoted to an evaluation
table or benchmark artifact.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import gaussian_kde, kstest

from tv3.audit.mrs_ei_mei4_formal import (
    COMPONENTS,
    MethodPosterior,
    PosteriorRecord,
    _frozen_initialization,
    _importance_method,
    _laplace_method,
    _problem,
    _spec_for_record,
    load_frozen_c2_inputs,
)
from tv3.audit.mrs_ei_posterior_gate import coverage_with_rejections, crps_from_samples, sbc_uniformity
from tv3.ml.mrs_posterior import PosteriorConstructionError, raw3_from_tangent, tangent_from_raw3
from tv3.ml.mrs_varpro import (
    build_s1_parameterization,
    build_s1_settings,
    build_varpro_parameterization,
    pack_s1_parameters,
    predict_s1,
    solve_s1,
    solve_s2,
)


MC_METHODS = ("M1", "M1b", "M2")


@dataclass(frozen=True)
class _Template:
    record: PosteriorRecord
    truth_raw3: np.ndarray
    generation_parameters: np.ndarray


@dataclass(frozen=True)
class C3Task:
    phase: str
    domain: str
    order: int
    item_id: str
    record: PosteriorRecord
    truth_raw3: tuple[float, float, float] | None = None

    @property
    def task_id(self) -> str:
        return f"{self.phase}/{self.domain}/{self.item_id}"


def _task_result(task: C3Task, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "phase": task.phase,
        "domain": task.domain,
        "order": task.order,
        "item_id": task.item_id,
        "payload": dict(payload),
    }


def _ordered_phase_results(
    results: Sequence[Mapping[str, Any]], *, phase: str, domain: str
) -> list[Mapping[str, Any]]:
    selected = [row for row in results if row.get("phase") == phase and row.get("domain") == domain]
    ordered = sorted(selected, key=lambda row: int(row["order"]))
    orders = [int(row["order"]) for row in ordered]
    if orders != list(range(len(orders))):
        raise RuntimeError(f"{phase} {domain} task order is incomplete or duplicated: {orders}")
    return ordered


def _seed(*parts: str | int) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _solution_payload(solution: Any, method: str) -> dict[str, Any]:
    payload = {
        "method": method,
        "success": bool(solution.success),
        "stop_reason": str(solution.stop_reason),
        "objective": float(solution.objective),
        "forward_calls": int(solution.forward_calls),
        "iterations": len(solution.iterations),
        "bound_hit": bool(solution.bound_hit),
        "bound_parameters": list(solution.bound_parameters),
        "raw3_percent": np.asarray(solution.raw3_percent, dtype=np.float64).tolist(),
        "parameters": np.asarray(solution.parameters, dtype=np.float64).tolist(),
    }
    if method == "S2":
        payload["linear_parameters"] = np.asarray(solution.linear_parameters, dtype=np.float64).tolist()
    return payload


def _select_solution(candidates: Sequence[Any]) -> Any:
    successful = [candidate for candidate in candidates if candidate.success]
    return min(successful or list(candidates), key=lambda candidate: float(candidate.objective))


def _fit_s1(record: PosteriorRecord, solver_config: Mapping[str, Any], calibration: Mapping[str, Any]) -> Mapping[str, Any]:
    spec = _spec_for_record(build_s1_parameterization(solver_config), calibration, record)
    settings = build_s1_settings(solver_config)
    candidates = [
        solve_s1(_problem(record), _frozen_initialization(solver_config, record, index), spec, settings)
        for index in range(len(solver_config["b1_solver_audit"]["frozen_initializations"]))
    ]
    payload = _solution_payload(_select_solution(candidates), "S1")
    payload["all_initializations_forward_calls"] = int(sum(candidate.forward_calls for candidate in candidates))
    return payload


def _fit_s1_s2(
    record: PosteriorRecord, solver_config: Mapping[str, Any], calibration: Mapping[str, Any]
) -> PosteriorRecord:
    spec = _spec_for_record(build_s1_parameterization(solver_config), calibration, record)
    settings = build_s1_settings(solver_config)
    varpro = build_varpro_parameterization(solver_config, spec)
    phase_limit = float(solver_config["b2_solver_core"]["max_phase_branch_standardized_error"])
    initials = [
        _frozen_initialization(solver_config, record, index)
        for index in range(len(solver_config["b1_solver_audit"]["frozen_initializations"]))
    ]
    s1 = _solution_payload(
        _select_solution([solve_s1(_problem(record), initial, spec, settings) for initial in initials]), "S1"
    )
    try:
        s2 = _solution_payload(
            _select_solution(
                [
                    solve_s2(
                        _problem(record),
                        initial,
                        spec,
                        settings,
                        varpro,
                        max_phase_branch_standardized_error=phase_limit,
                    )
                    for initial in initials
                ]
            ),
            "S2",
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        # This is an explicit failed S2 construction, not a substituted posterior.
        s2 = {
            "method": "S2",
            "success": False,
            "stop_reason": f"s2_solver_error:{exc}",
            "objective": float("inf"),
            "forward_calls": 0,
            "iterations": 0,
            "bound_hit": True,
            "bound_parameters": [],
            "raw3_percent": [float("nan")] * 3,
            "parameters": [float("nan")] * len(spec.names),
            "linear_parameters": [float("nan")] * len(varpro.linear_indices),
        }
    return replace(record, s1=s1, s2=s2)


def _posterior_or_rejected(
    method: str,
    record: PosteriorRecord,
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> MethodPosterior:
    spec = _spec_for_record(build_s1_parameterization(solver_config), calibration, record)
    try:
        if method == "M1":
            return _laplace_method(method="M1", record=record, spec=spec, solver_config=solver_config, contract=contract)
        if method == "M1b":
            return _laplace_method(method="M1b", record=record, spec=spec, solver_config=solver_config, contract=contract)
        if method == "M2":
            m1 = _laplace_method(method="M1", record=record, spec=spec, solver_config=solver_config, contract=contract)
            return _importance_method(record, spec, contract, m1)
    except (PosteriorConstructionError, ValueError, np.linalg.LinAlgError) as exc:
        return MethodPosterior(method, None, None, None, None, True, f"posterior_construction_error:{exc}", {}, None)
    raise ValueError(f"unsupported MC method: {method}")


def _all_posteriors(
    record: PosteriorRecord,
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> Mapping[str, MethodPosterior]:
    spec = _spec_for_record(build_s1_parameterization(solver_config), calibration, record)
    try:
        m1 = _laplace_method(method="M1", record=record, spec=spec, solver_config=solver_config, contract=contract)
    except (PosteriorConstructionError, ValueError, np.linalg.LinAlgError) as exc:
        m1 = MethodPosterior("M1", None, None, None, None, True, f"posterior_construction_error:{exc}", {}, None)
    try:
        m1b = _laplace_method(method="M1b", record=record, spec=spec, solver_config=solver_config, contract=contract)
    except (PosteriorConstructionError, ValueError, np.linalg.LinAlgError) as exc:
        m1b = MethodPosterior("M1b", None, None, None, None, True, f"posterior_construction_error:{exc}", {}, None)
    try:
        m2 = _importance_method(record, spec, contract, m1)
    except (PosteriorConstructionError, ValueError, np.linalg.LinAlgError) as exc:
        m2 = MethodPosterior("M2", None, None, None, None, True, f"posterior_construction_error:{exc}", {}, None)
    return {"M1": m1, "M1b": m1b, "M2": m2}


def _load_context(
    b4_dir: Path,
) -> tuple[dict[str, list[_Template]], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], list[PosteriorRecord]]:
    records, solver_config, audit_tables = load_frozen_c2_inputs(b4_dir)
    nuisance_rows = json.loads((b4_dir / "s3_truth_nuisance.json").read_text(encoding="utf-8"))
    nuisance_by_id = {str(row["mixture_id"]): row for row in nuisance_rows}
    templates: dict[str, list[_Template]] = {"test": [], "ood": []}
    calibration = audit_tables["calibration"]
    for record in records:
        mixture = audit_tables["mixtures"][record.mixture_id]
        nuisance = nuisance_by_id.get(record.mixture_id)
        if nuisance is None:
            raise RuntimeError(f"missing isolated generator nuisance for {record.mixture_id}")
        truth = np.asarray(
            [mixture["x_CO2_percent"], mixture["x_O2_percent"], mixture["x_N2_percent"]], dtype=np.float64
        )
        parameters = pack_s1_parameters(
            truth,
            t_c=record.t_c,
            path_length_m=record.path_length_m,
            h_rh=record.h_rh,
            common_delay_s=float(nuisance["common_delay_s"]),
            log_amplitude_gain=float(nuisance["log_amplitude_gain"]),
            per_frequency_offsets=nuisance["log_amplitude_offsets"],
        )
        # Verify that generator-only truth is legal before it is isolated from inference.
        predict_s1(_problem(record), parameters, _spec_for_record(build_s1_parameterization(solver_config), calibration, record))
        templates[record.split].append(_Template(record, truth, parameters))
    if {name: len(values) for name, values in templates.items()} != {"test": 648, "ood": 648}:
        raise RuntimeError("C3 templates must retain the frozen 648-mixture domains")
    return templates, solver_config, calibration, audit_tables, records


def _sbc_record(template: _Template, *, index: int, rng: np.random.Generator, solver_config: Mapping[str, Any], calibration: Mapping[str, Any]) -> PosteriorRecord:
    record = template.record
    spec = _spec_for_record(build_s1_parameterization(solver_config), calibration, record)
    mean = predict_s1(_problem(record), template.generation_parameters, spec)
    observation = mean + rng.multivariate_normal(np.zeros(mean.size), record.covariance)
    return replace(record, mixture_id=f"SBC-{record.split}-{index:04d}", observation=observation)


def _posterior_draws(posterior: MethodPosterior, *, draws: int, rng: np.random.Generator) -> np.ndarray:
    if posterior.rejected or posterior.raw3_samples is None:
        raise PosteriorConstructionError("posterior_rejected_before_sbc_rank")
    probabilities = posterior.weights
    if probabilities is None:
        indices = rng.integers(0, posterior.raw3_samples.shape[0], size=draws)
    else:
        indices = rng.choice(posterior.raw3_samples.shape[0], size=draws, replace=True, p=probabilities)
    return posterior.raw3_samples[indices]


def build_sbc_tasks(
    *,
    templates: Mapping[str, Sequence[_Template]],
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[C3Task]:
    replicates = int(contract["mc_protocol"]["sbc"]["replicates_per_domain_per_method"])
    tasks: list[C3Task] = []
    for domain in ("test", "ood"):
        generator = np.random.default_rng(int(contract["mc_protocol"]["sbc"]["seeds"][domain]))
        for order in range(replicates):
            template = templates[domain][int(generator.integers(0, len(templates[domain])))]
            record = _sbc_record(
                template,
                index=order + 1,
                rng=generator,
                solver_config=solver_config,
                calibration=calibration,
            )
            tasks.append(
                C3Task(
                    phase="sbc",
                    domain=domain,
                    order=order,
                    item_id=f"{order + 1:04d}",
                    record=record,
                    truth_raw3=tuple(float(value) for value in template.truth_raw3),
                )
            )
    return tasks


def run_sbc_task(
    task: C3Task,
    *,
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if task.phase != "sbc" or task.truth_raw3 is None:
        raise ValueError(f"invalid SBC task: {task.task_id}")
    fitted = _fit_s1_s2(task.record, solver_config, calibration)
    posteriors = _all_posteriors(fitted, solver_config, calibration, contract)
    draws = int(contract["mc_protocol"]["sbc"]["posterior_draws_per_replicate"])
    truth = np.asarray(task.truth_raw3, dtype=np.float64)
    methods: dict[str, Any] = {}
    for method in MC_METHODS:
        posterior = posteriors[method]
        if posterior.rejected:
            methods[method] = {"rejected": True, "ranks": None}
            continue
        samples = _posterior_draws(
            posterior,
            draws=draws,
            rng=np.random.default_rng(_seed("sbc", task.domain, method, task.order)),
        )
        methods[method] = {
            "rejected": False,
            "ranks": {
                component: int(np.sum(samples[:, component_index] < truth[component_index]))
                for component_index, component in enumerate(COMPONENTS)
            },
        }
    return _task_result(task, methods)


def aggregate_sbc_results(results: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    protocol = contract["mc_protocol"]["sbc"]
    replicates = int(protocol["replicates_per_domain_per_method"])
    draws = int(protocol["posterior_draws_per_replicate"])
    alpha = float(protocol["alpha"])
    report: dict[str, Any] = {"schema_version": "tunnel-ventilation-mrs-ei-mei4-sbc-1", "methods": {}}
    for domain in ("test", "ood"):
        domain_results = _ordered_phase_results(results, phase="sbc", domain=domain)
        if len(domain_results) != replicates:
            raise RuntimeError(f"SBC {domain} task count mismatch: {len(domain_results)} != {replicates}")
        for method in MC_METHODS:
            ranks = {component: [] for component in COMPONENTS}
            rejected = 0
            for result in domain_results:
                method_result = result["payload"][method]
                if bool(method_result["rejected"]):
                    rejected += 1
                    continue
                for component in COMPONENTS:
                    ranks[component].append(int(method_result["ranks"][component]))
            components: dict[str, Any] = {}
            for component in COMPONENTS:
                values = ranks[component]
                uniformity = (
                    sbc_uniformity(values, posterior_draws=draws)
                    if values
                    else {"statistic": float("nan"), "pvalue": 0.0}
                )
                components[component] = {
                    "n_ranked": len(values),
                    "histogram": np.bincount(values, minlength=draws + 1).tolist(),
                    "uniformity": uniformity,
                    "passed": len(values) == replicates and float(uniformity["pvalue"]) >= alpha,
                }
            report["methods"].setdefault(method, {})[domain] = {
                "n_replicates": replicates,
                "rejected": rejected,
                "components": components,
                "passed": rejected == 0 and all(row["passed"] for row in components.values()),
            }
    return report


def run_sbc(
    *,
    templates: Mapping[str, Sequence[_Template]],
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    tasks = build_sbc_tasks(
        templates=templates,
        solver_config=solver_config,
        calibration=calibration,
        contract=contract,
    )
    results = []
    total = int(contract["mc_protocol"]["sbc"]["replicates_per_domain_per_method"])
    for task in tasks:
        results.append(
            run_sbc_task(task, solver_config=solver_config, calibration=calibration, contract=contract)
        )
        if progress_callback is not None:
            progress_callback("SBC", task.order + 1, total, f"{task.domain}:{task.order + 1}")
    return aggregate_sbc_results(results, contract)


def _valid_parameters(values: np.ndarray, spec: Any) -> np.ndarray:
    raw3 = raw3_from_tangent(values[:, :2])
    return (
        np.all(values >= spec.lower_bounds, axis=1)
        & np.all(values <= spec.upper_bounds, axis=1)
        & np.all(raw3 >= 0.0, axis=1)
    )


def _predictive_parameters(
    method: str,
    posterior: MethodPosterior,
    record: PosteriorRecord,
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if posterior.rejected:
        raise PosteriorConstructionError("posterior_rejected_before_ppc")
    spec = _spec_for_record(build_s1_parameterization(solver_config), calibration, record)
    if method == "M2":
        if posterior.parameter_samples is None or posterior.weights is None:
            raise PosteriorConstructionError("M2 parameter samples are unavailable for PPC")
        indices = rng.choice(posterior.parameter_samples.shape[0], size=draws, replace=True, p=posterior.weights)
        return posterior.parameter_samples[indices]
    if posterior.laplace is None:
        raise PosteriorConstructionError("Laplace covariance is unavailable for PPC")
    if method == "M1":
        base = np.asarray(record.s1["parameters"], dtype=np.float64)
        positions = np.arange(base.size)
    elif method == "M1b":
        base = np.asarray(record.s2["parameters"], dtype=np.float64)
        positions = build_varpro_parameterization(solver_config, spec).nonlinear_indices
    else:
        raise ValueError(f"unsupported PPC method: {method}")
    covariance = posterior.laplace.covariance_standardized
    if covariance.shape != (positions.size, positions.size):
        raise PosteriorConstructionError("Laplace covariance dimension does not match predictive coordinates")
    normal = rng.multivariate_normal(np.zeros(positions.size), covariance, size=max(4096, draws * 8))
    candidates = np.tile(base, (normal.shape[0], 1))
    candidates[:, positions] += normal * spec.scales[positions]
    valid = candidates[_valid_parameters(candidates, spec)]
    if valid.shape[0] < draws:
        raise PosteriorConstructionError("insufficient_physical_posterior_draws_for_ppc")
    return valid[:draws]


def _uniform_report(values: Sequence[float], alpha: float) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or not np.all(np.isfinite(data)):
        return {"n": int(data.size), "statistic": float("nan"), "pvalue": 0.0, "passed": False}
    statistic, pvalue = kstest(data, "uniform")
    return {
        "n": int(data.size),
        "mean": float(np.mean(data)),
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "passed": bool(pvalue >= alpha),
    }


def build_ppc_tasks(records: Sequence[PosteriorRecord]) -> list[C3Task]:
    domain_order = {"test": 0, "ood": 0}
    tasks: list[C3Task] = []
    for record in records:
        if record.split not in domain_order:
            continue
        order = domain_order[record.split]
        domain_order[record.split] += 1
        tasks.append(C3Task("ppc", record.split, order, record.mixture_id, record))
    return tasks


def run_ppc_task(
    task: C3Task,
    *,
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if task.phase != "ppc":
        raise ValueError(f"invalid PPC task: {task.task_id}")
    draws = int(contract["mc_protocol"]["ppc"]["y_rep_per_frozen_mixture"])
    record = task.record
    spec = _spec_for_record(build_s1_parameterization(solver_config), calibration, record)
    chol = np.linalg.cholesky(record.covariance)
    posteriors = _all_posteriors(record, solver_config, calibration, contract)
    methods: dict[str, Any] = {}
    for method in MC_METHODS:
        posterior = posteriors[method]
        try:
            parameters = _predictive_parameters(
                method,
                posterior,
                record,
                solver_config,
                calibration,
                draws,
                np.random.default_rng(_seed("ppc", task.domain, method, record.mixture_id)),
            )
        except PosteriorConstructionError:
            methods[method] = {"rejected": True}
            continue
        predictions = np.asarray([predict_s1(_problem(record), item, spec) for item in parameters])
        observed_norm = np.asarray(
            [np.linalg.norm(np.linalg.solve(chol, prediction - record.observation)) for prediction in predictions]
        )
        noise = np.random.default_rng(
            _seed("ppc-noise", task.domain, method, record.mixture_id)
        ).multivariate_normal(np.zeros(record.observation.size), record.covariance, size=draws)
        replicated = predictions + noise
        replicated_norm = np.asarray(
            [
                np.linalg.norm(np.linalg.solve(chol, prediction - item))
                for prediction, item in zip(predictions, replicated, strict=True)
            ]
        )
        methods[method] = {
            "rejected": False,
            "norm": float(np.mean(replicated_norm >= observed_norm)),
            "channels": np.mean(replicated <= record.observation[np.newaxis, :], axis=0).tolist(),
        }
    return _task_result(task, methods)


def aggregate_ppc_results(results: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    alpha = float(contract["mc_protocol"]["ppc"]["systematic_deviation_tail_probability_threshold"])
    payload: dict[str, Any] = {"schema_version": "tunnel-ventilation-mrs-ei-mei4-ppc-1", "methods": {}}
    for domain in ("test", "ood"):
        domain_results = _ordered_phase_results(results, phase="ppc", domain=domain)
        values = {
            method: {"norm": [], "channels": [[] for _ in range(12)], "rejected": 0}
            for method in MC_METHODS
        }
        for result in domain_results:
            for method in MC_METHODS:
                method_result = result["payload"][method]
                if bool(method_result["rejected"]):
                    values[method]["rejected"] += 1
                    continue
                values[method]["norm"].append(float(method_result["norm"]))
                for channel, value in enumerate(method_result["channels"]):
                    values[method]["channels"][channel].append(float(value))
        for method in MC_METHODS:
            norm_report = _uniform_report(values[method]["norm"], alpha)
            channel_reports = [_uniform_report(channel, alpha) for channel in values[method]["channels"]]
            payload["methods"].setdefault(method, {})[domain] = {
                "n_frozen_mixtures": len(domain_results),
                "rejected": values[method]["rejected"],
                "whitened_residual_norm_tail_probability": norm_report,
                "per_channel_empirical_quantile": channel_reports,
                "passed": values[method]["rejected"] == 0
                and norm_report["passed"]
                and all(row["passed"] for row in channel_reports),
            }
    return payload


def run_ppc(
    *,
    records: Sequence[PosteriorRecord],
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    tasks = build_ppc_tasks(records)
    totals = {domain: sum(task.domain == domain for task in tasks) for domain in ("test", "ood")}
    results = []
    for task in tasks:
        results.append(
            run_ppc_task(task, solver_config=solver_config, calibration=calibration, contract=contract)
        )
        if progress_callback is not None:
            progress_callback("PPC", task.order + 1, totals[task.domain], f"{task.domain}:{task.item_id}")
    return aggregate_ppc_results(results, contract)


def _intervals(samples: np.ndarray, levels: Sequence[float]) -> dict[str, list[list[float]]]:
    result: dict[str, list[list[float]]] = {}
    for level in levels:
        tail = (1.0 - float(level)) / 2.0
        result[str(float(level))] = np.quantile(samples, [tail, 1.0 - tail], axis=0, method="linear").T.tolist()
    return result


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return {"count": 0}
    return {"count": int(finite.size), "median": float(np.median(finite)), "p90": float(np.quantile(finite, 0.9)), "mean": float(np.mean(finite))}


def build_m2b_tasks(records: Sequence[PosteriorRecord], audit_tables: Mapping[str, Any]) -> list[C3Task]:
    domain_order = {"test": 0, "ood": 0}
    tasks: list[C3Task] = []
    for record in records:
        if record.split not in domain_order:
            continue
        mixture = audit_tables["mixtures"][record.mixture_id]
        truth = (
            float(mixture["x_CO2_percent"]),
            float(mixture["x_O2_percent"]),
            float(mixture["x_N2_percent"]),
        )
        order = domain_order[record.split]
        domain_order[record.split] += 1
        tasks.append(C3Task("m2b", record.split, order, record.mixture_id, record, truth))
    return tasks


def run_m2b_task(
    task: C3Task,
    *,
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if task.phase != "m2b" or task.truth_raw3 is None:
        raise ValueError(f"invalid M2b task: {task.task_id}")
    protocol = contract["mc_protocol"]["m2b"]
    draws = int(protocol["resamples_per_mixture"])
    levels = contract["calibration_gate"]["nominal_levels"]
    record = task.record
    spec = _spec_for_record(build_s1_parameterization(solver_config), calibration, record)
    truth = np.asarray(task.truth_raw3, dtype=np.float64)
    rejected = False
    reason = ""
    forward_calls = 0
    samples: list[np.ndarray] = []
    try:
        center = predict_s1(_problem(record), np.asarray(record.s1["parameters"], dtype=np.float64), spec)
        rng = np.random.default_rng(
            _seed("m2b", record.split, record.mixture_id, int(protocol["seeds"][record.split]))
        )
        for draw in range(draws):
            boot = replace(
                record,
                observation=center + rng.multivariate_normal(np.zeros(center.size), record.covariance),
            )
            solved = _fit_s1(boot, solver_config, calibration)
            forward_calls += int(solved["all_initializations_forward_calls"])
            raw3 = np.asarray(solved["raw3_percent"], dtype=np.float64)
            if raw3.shape != (3,) or not np.all(np.isfinite(raw3)) or np.any(raw3 < 0.0):
                raise PosteriorConstructionError(f"bootstrap_nonfinite_draw_{draw}")
            samples.append(raw3)
    except (PosteriorConstructionError, ValueError, np.linalg.LinAlgError) as exc:
        rejected = True
        reason = str(exc)
    intervals = _intervals(np.asarray(samples), levels) if not rejected else None
    row = {
        "mixture_id": record.mixture_id,
        "split": record.split,
        "design_condition_id": record.design_condition_id,
        "method": "M2b",
        "rejected": rejected,
        "rejection_reason": reason,
        "bootstrap_resamples": draws,
        "forward_calls": forward_calls,
        "intervals": intervals,
    }
    events: list[dict[str, Any]] = []
    for level in levels:
        bounds = intervals[str(float(level))] if intervals else [[0.0, 0.0]] * 3
        for component, target, interval in zip(COMPONENTS, truth, bounds, strict=True):
            coverage = coverage_with_rejections([target], [interval], [rejected])
            events.append(
                {
                    "split": record.split,
                    "component": component,
                    "level": str(float(level)),
                    "covered": coverage["covered"],
                    "rejected": coverage["rejected"],
                }
            )
    scores = {"nll": [], "crps": []}
    if not rejected:
        values = np.asarray(samples)
        for component in range(3):
            density = float(gaussian_kde(values[:, component])([truth[component]])[0])
            if not np.isfinite(density) or density <= 0.0:
                raise RuntimeError("M2b marginal density is non-positive")
            scores["nll"].append(float(-np.log(density)))
            scores["crps"].append(crps_from_samples(values[:, component], truth[component]))
    return _task_result(task, {"row": row, "events": events, "scores": scores})


def aggregate_m2b_results(results: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> dict[str, Any]:
    draws = int(contract["mc_protocol"]["m2b"]["resamples_per_mixture"])
    levels = contract["calibration_gate"]["nominal_levels"]
    ordered: list[Mapping[str, Any]] = []
    for domain in ("test", "ood"):
        ordered.extend(_ordered_phase_results(results, phase="m2b", domain=domain))
    events = [event for result in ordered for event in result["payload"]["events"]]
    rows = [result["payload"]["row"] for result in ordered]
    scores: dict[str, dict[str, list[float]]] = {
        domain: {"nll": [], "crps": []} for domain in ("test", "ood")
    }
    for result in ordered:
        domain = str(result["domain"])
        scores[domain]["nll"].extend(float(value) for value in result["payload"]["scores"]["nll"])
        scores[domain]["crps"].extend(float(value) for value in result["payload"]["scores"]["crps"])
    bands = contract["calibration_gate"]["exact_binomial_acceptance_counts"]
    coverage_rows = []
    for domain in ("test", "ood"):
        for component in COMPONENTS:
            for level in levels:
                selected = [
                    row
                    for row in events
                    if row["split"] == domain
                    and row["component"] == component
                    and row["level"] == str(float(level))
                ]
                if not selected:
                    raise RuntimeError(f"M2b has no coverage events for {domain}/{component}/{level}")
                covered = sum(int(row["covered"]) for row in selected)
                rejected = sum(int(row["rejected"]) for row in selected)
                band = bands[str(float(level))]
                coverage_rows.append(
                    {
                        "method": "M2b",
                        "domain": domain,
                        "component": component,
                        "nominal_level": level,
                        "n": len(selected),
                        "covered": covered,
                        "rejected": rejected,
                        "coverage": covered / len(selected),
                        "acceptance_band": band,
                        "within_acceptance_band": band["lower_inclusive"] <= covered <= band["upper_inclusive"],
                    }
                )
    return {
        "schema_version": "tunnel-ventilation-mrs-ei-mei4-m2b-1",
        "status": "mei4_m2b_bootstrap_complete",
        "resamples_per_mixture": draws,
        "coverage_report": {
            "primary_bands": coverage_rows,
            "passed": all(row["within_acceptance_band"] for row in coverage_rows),
        },
        "score_report": {
            domain: {name: _summary(values) for name, values in rows_by_name.items()}
            for domain, rows_by_name in scores.items()
        },
        "cost": {
            "forward_calls": int(sum(int(row["forward_calls"]) for row in rows)),
            "per_mixture_forward_calls": _summary([float(row["forward_calls"]) for row in rows]),
        },
        "rejected": {
            domain: sum(bool(row["rejected"]) for row in rows if row["split"] == domain)
            for domain in ("test", "ood")
        },
        "per_mixture": rows,
    }


def _m2b_report(
    *,
    records: Sequence[PosteriorRecord],
    audit_tables: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    tasks = build_m2b_tasks(records, audit_tables)
    results = []
    for index, task in enumerate(tasks, start=1):
        results.append(
            run_m2b_task(task, solver_config=solver_config, calibration=calibration, contract=contract)
        )
        if progress_callback is not None:
            progress_callback("M2b", index, len(tasks), task.item_id)
    return aggregate_m2b_results(results, contract)


def build_c3_tasks(
    *,
    templates: Mapping[str, Sequence[_Template]],
    records: Sequence[PosteriorRecord],
    audit_tables: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, list[C3Task]]:
    return {
        "sbc": build_sbc_tasks(
            templates=templates,
            solver_config=solver_config,
            calibration=calibration,
            contract=contract,
        ),
        "ppc": build_ppc_tasks(records),
        "m2b": build_m2b_tasks(records, audit_tables),
    }


def execute_c3_task(
    task: C3Task,
    *,
    solver_config: Mapping[str, Any],
    calibration: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if task.phase == "sbc":
        return run_sbc_task(task, solver_config=solver_config, calibration=calibration, contract=contract)
    if task.phase == "ppc":
        return run_ppc_task(task, solver_config=solver_config, calibration=calibration, contract=contract)
    if task.phase == "m2b":
        return run_m2b_task(task, solver_config=solver_config, calibration=calibration, contract=contract)
    raise ValueError(f"unsupported C3 task phase: {task.phase}")


def aggregate_c3_results(
    results: Sequence[Mapping[str, Any]], *, contract: Mapping[str, Any], m2b_triggered: bool
) -> dict[str, Any]:
    if not m2b_triggered:
        raise RuntimeError("M2b must not run unless the registered PSIS trigger is present")
    return {
        "sbc_rank_histograms": aggregate_sbc_results(results, contract),
        "ppc_report": aggregate_ppc_results(results, contract),
        "bootstrap_posterior_report": aggregate_m2b_results(results, contract),
    }


def run_c3_mc_calibration(
    *,
    b4_dir: Path,
    contract: Mapping[str, Any],
    m2b_triggered: bool,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    if not m2b_triggered:
        raise RuntimeError("M2b must not run unless the registered PSIS trigger is present")
    templates, solver_config, calibration, audit_tables, records = _load_context(b4_dir)
    return {
        "sbc_rank_histograms": run_sbc(templates=templates, solver_config=solver_config, calibration=calibration, contract=contract, progress_callback=progress_callback),
        "ppc_report": run_ppc(records=records, solver_config=solver_config, calibration=calibration, contract=contract, progress_callback=progress_callback),
        "bootstrap_posterior_report": _m2b_report(records=records, audit_tables=audit_tables, solver_config=solver_config, calibration=calibration, contract=contract, progress_callback=progress_callback),
    }


__all__ = [
    "C3Task",
    "aggregate_c3_results",
    "build_c3_tasks",
    "execute_c3_task",
    "run_c3_mc_calibration",
]
