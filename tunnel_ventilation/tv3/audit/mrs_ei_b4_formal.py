"""MEI-3 B4 registered sparse generation and S1/S2/S3 paired comparison."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from tv3.audit.mrs_ei_registry import build_named_point_set
from tv3.audit.mrs_ei_solver_gate import (
    B4_AUTHORIZED_VALUE,
    B4_PHASE,
    assess_b4_execution_authorization,
)
from tv3.ml.mrs_varpro import (
    S1Parameterization,
    S1Problem,
    build_s1_parameterization,
    build_s1_settings,
    build_varpro_parameterization,
    composition_crb_o2_std,
    pack_s1_parameters,
    solve_conditionally_linear,
    solve_s1,
    solve_s2,
    solve_s3,
)
from tv3.sim.generation.tunnel_ventilation.mrs_observation import (
    ideal_mrs_observation,
    validate_raw3_percent,
)

NON_DEGENERATION_METRICS = (
    "mae_o2",
    "mae_co2",
    "mae_n2",
    "convergence_failure_rate",
    "worst_group_error_o2",
    "bound_hit_rate",
)


@dataclass(frozen=True)
class MixtureRecord:
    mixture_id: str
    split: str
    design_condition_id: str
    raw3_percent: np.ndarray
    t_c: float
    p_mpa: float
    h_rh: float
    path_length_m: float
    device_profile_id: str
    view_id: str
    observation: np.ndarray
    covariance: np.ndarray
    frequencies_hz: np.ndarray
    phase_branch_cycles: np.ndarray
    observation_std: Mapping[str, float]


def _as_protocol_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _linear_design(frequencies_hz: np.ndarray) -> np.ndarray:
    n_freq = int(frequencies_hz.size)
    matrix = np.zeros((n_freq * 3, 2 + n_freq), dtype=np.float64)
    matrix[0::3, 0] = 1.0
    matrix[2::3, 0] = -2.0 * np.pi * frequencies_hz
    matrix[1::3, 1] = 1.0
    for index, frequency in enumerate(frequencies_hz):
        matrix[3 * index + 1, 2 + index] = 1.0
    return matrix


def _apply_linear_nuisance(
    baseline: np.ndarray,
    *,
    frequencies_hz: np.ndarray,
    common_delay_s: float,
    log_amplitude_gain: float,
    log_amplitude_offsets: Sequence[float],
) -> np.ndarray:
    prediction = np.asarray(baseline, dtype=np.float64).copy()
    offsets = np.asarray(log_amplitude_offsets, dtype=np.float64)
    if offsets.shape != frequencies_hz.shape:
        raise ValueError("log_amplitude_offsets must match frequencies")
    prediction[0::3] += float(common_delay_s)
    prediction[1::3] += float(log_amplitude_gain) + offsets
    prediction[2::3] -= 2.0 * np.pi * frequencies_hz * float(common_delay_s)
    return prediction


def _observation_std_from_noise_profile(
    design_space: Mapping[str, Any], noise_profile_id: str
) -> dict[str, float]:
    profile = design_space["noise_profiles"][noise_profile_id]
    return {
        "raw_tof_s": float(profile["jitter_std_s"]),
        "log_amplitude": float(profile["relative_amp_std"]),
        "unwrapped_phase_rad": 0.1,
    }


def _raw3_from_point(point: Any) -> np.ndarray:
    return validate_raw3_percent(
        [point.co2_percent, point.o2_percent, point.n2_percent]
    )


def generate_registered_sparse_dataset(
    *,
    protocol: Mapping[str, Any],
    gate: Mapping[str, Any],
    design_space: Mapping[str, Any],
    mixture_limit_per_split: int | None = None,
) -> dict[str, Any]:
    """Generate one frozen registered sparse observation package."""
    frozen = protocol["frozen_design"]
    frequencies = np.asarray(frozen["frequencies_hz"], dtype=np.float64)
    view_ids = list(protocol["view_protocol"]["registered_view_ids"])
    if protocol["view_protocol"]["views_per_mixture"] != 1 or view_ids != ["view_0"]:
        raise ValueError("B4 currently freezes views_per_mixture=1 with view_0 only")
    view_id = view_ids[0]
    device_profile_id = str(gate["generation"]["device_profile_id"])
    observation_std = _observation_std_from_noise_profile(
        design_space, str(frozen["noise_profile_id"])
    )
    branches = np.asarray(gate["generation"]["phase_branch_cycles"], dtype=np.int64)
    if branches.shape != frequencies.shape:
        raise ValueError("phase_branch_cycles must match D0 K4")
    truth_nuisance = gate["generation"]["shared_truth_nuisance"]
    delay = float(truth_nuisance["common_delay_s"])
    gain = float(truth_nuisance["log_amplitude_gain"])
    offsets = np.asarray(truth_nuisance["log_amplitude_offsets"], dtype=np.float64)
    if offsets.shape != frequencies.shape:
        raise ValueError("shared truth offsets must match D0 K4")

    split_seeds = list(protocol["split_protocol"]["split_seeds"])
    if len(split_seeds) != 3:
        raise ValueError("exactly three split seeds are required")
    replicates = int(protocol["sample_size_and_power"]["replicates_per_condition"])
    calib_replicates = int(gate["generation"]["calibration_replicates_per_condition"])
    split_specs = [
        (
            "calibration",
            str(gate["generation"]["calibration_point_set"]),
            calib_replicates,
            int(split_seeds[0]),
        ),
        (
            "test",
            str(protocol["split_protocol"]["test_point_set"]),
            replicates,
            int(split_seeds[1]),
        ),
        (
            "ood",
            str(protocol["split_protocol"]["ood_point_set"]),
            replicates,
            int(split_seeds[2]),
        ),
    ]

    mixtures: list[MixtureRecord] = []
    mixture_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    covariance_blocks: list[dict[str, Any]] = []
    s3_truth_rows: list[dict[str, Any]] = []
    mixture_counter = 0

    for split, point_set_id, n_replicates, seed in split_specs:
        points = build_named_point_set(design_space, point_set_id)
        if mixture_limit_per_split is not None:
            points = points[: int(mixture_limit_per_split)]
        rng = _as_protocol_rng(seed)
        for condition_index, (condition_id, point) in enumerate(points):
            for replicate in range(n_replicates):
                mixture_counter += 1
                mixture_id = f"M{mixture_counter:06d}"
                raw3 = _raw3_from_point(point)
                ideal = ideal_mrs_observation(
                    raw3,
                    t_c=float(point.t_c),
                    p_mpa=float(point.p_mpa),
                    h_rh=float(point.h_rh),
                    path_length_m=float(point.path_length_m),
                    frequencies_hz=frequencies,
                    phase_branch_cycles=branches,
                    observation_std=observation_std,
                )
                clean = _apply_linear_nuisance(
                    ideal.vector,
                    frequencies_hz=frequencies,
                    common_delay_s=delay,
                    log_amplitude_gain=gain,
                    log_amplitude_offsets=offsets,
                )
                noise = rng.multivariate_normal(
                    mean=np.zeros(clean.size),
                    cov=ideal.covariance,
                )
                observation = clean + noise
                cov_id = f"{mixture_id}|{view_id}|cov"
                record = MixtureRecord(
                    mixture_id=mixture_id,
                    split=split,
                    design_condition_id=condition_id,
                    raw3_percent=raw3,
                    t_c=float(point.t_c),
                    p_mpa=float(point.p_mpa),
                    h_rh=float(point.h_rh),
                    path_length_m=float(point.path_length_m),
                    device_profile_id=device_profile_id,
                    view_id=view_id,
                    observation=observation,
                    covariance=ideal.covariance.copy(),
                    frequencies_hz=frequencies.copy(),
                    phase_branch_cycles=branches.copy(),
                    observation_std=observation_std,
                )
                mixtures.append(record)
                mixture_rows.append(
                    {
                        "mixture_id": mixture_id,
                        "x_CO2_percent": float(raw3[0]),
                        "x_O2_percent": float(raw3[1]),
                        "x_N2_percent": float(raw3[2]),
                        "split": split,
                        "design_condition_id": condition_id,
                        "replicate_index": replicate,
                        "condition_index": condition_index,
                    }
                )
                covariance_blocks.append(
                    {
                        "covariance_block_id": cov_id,
                        "mixture_id": mixture_id,
                        "view_id": view_id,
                        "observation_covariance": ideal.covariance.tolist(),
                    }
                )
                s3_truth_rows.append(
                    {
                        "mixture_id": mixture_id,
                        "view_id": view_id,
                        "common_delay_s": delay,
                        "log_amplitude_gain": gain,
                        "log_amplitude_offsets": offsets.tolist(),
                    }
                )
                for freq_index, frequency in enumerate(frequencies):
                    observation_rows.append(
                        {
                            "observation_row_id": (
                                f"{mixture_id}|{view_id}|{int(frequency)}hz"
                            ),
                            "mixture_id": mixture_id,
                            "device_profile_id": device_profile_id,
                            "view_id": view_id,
                            "frequency_hz": float(frequency),
                            "raw_tof_s": float(observation[3 * freq_index]),
                            "log_amplitude": float(observation[3 * freq_index + 1]),
                            "unwrapped_phase_rad": float(observation[3 * freq_index + 2]),
                            "phase_branch_cycles": int(branches[freq_index]),
                            "covariance_block_id": cov_id,
                            "T_C": float(point.t_c),
                            "P_MPa": float(point.p_mpa),
                            "H_RH": float(point.h_rh),
                            "L_m": float(point.path_length_m),
                        }
                    )

    counts = {
        split: sum(1 for row in mixture_rows if row["split"] == split)
        for split in ("calibration", "test", "ood")
    }
    if mixture_limit_per_split is None:
        expected = int(protocol["sample_size_and_power"]["frozen_mixture_ids_per_domain"])
        if counts["test"] != expected or counts["ood"] != expected:
            raise ValueError(
                f"frozen evaluation sample size mismatch: test={counts['test']} "
                f"ood={counts['ood']} expected={expected}"
            )
    return {
        "mixtures": mixture_rows,
        "observation_rows": observation_rows,
        "covariance_blocks": covariance_blocks,
        "s3_truth_nuisance": s3_truth_rows,
        "records": mixtures,
        "meta": {
            "device_profile_id": device_profile_id,
            "view_id": view_id,
            "frequencies_hz": frequencies.tolist(),
            "observation_std": dict(observation_std),
            "split_counts": counts,
            "shared_truth_nuisance": {
                "common_delay_s": delay,
                "log_amplitude_gain": gain,
                "log_amplitude_offsets": offsets.tolist(),
            },
            "mixture_limit_per_split": mixture_limit_per_split,
        },
    }


def calibrate_option_a_priors(
    *,
    records: Sequence[MixtureRecord],
    solver_config: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate shared device×view / device×frequency priors on calibration only."""
    calibration = [row for row in records if row.split == "calibration"]
    if not calibration:
        raise ValueError("calibration split is empty")
    frequencies = calibration[0].frequencies_hz
    design = _linear_design(frequencies)
    stacked_design = []
    stacked_target = []
    stacked_cov = []
    for row in calibration:
        ideal = ideal_mrs_observation(
            row.raw3_percent,
            t_c=row.t_c,
            p_mpa=row.p_mpa,
            h_rh=row.h_rh,
            path_length_m=row.path_length_m,
            frequencies_hz=row.frequencies_hz,
            phase_branch_cycles=row.phase_branch_cycles,
            observation_std=row.observation_std,
        )
        stacked_design.append(design)
        stacked_target.append(row.observation - ideal.vector)
        stacked_cov.append(row.covariance)
    matrix = np.vstack(stacked_design)
    target = np.concatenate(stacked_target)
    covariance = np.zeros((target.size, target.size), dtype=np.float64)
    offset = 0
    for block in stacked_cov:
        size = block.shape[0]
        covariance[offset : offset + size, offset : offset + size] = block
        offset += size
    base_spec = build_s1_parameterization(solver_config)
    prior_names = (
        "common_delay_s",
        "log_amplitude_gain",
        *(
            f"log_amplitude_offset_{int(frequency)}hz"
            for frequency in frequencies
        ),
    )
    prior_mean = []
    prior_std = []
    for name in prior_names:
        index = list(base_spec.names).index(name)
        prior_row = int(np.where(base_spec.prior_indices == index)[0][0])
        prior_mean.append(float(base_spec.prior_mean[prior_row]))
        prior_std.append(float(base_spec.prior_std[prior_row]))
    solution = solve_conditionally_linear(
        observation=target,
        nonlinear_prediction=np.zeros(target.size),
        design_matrix=matrix,
        covariance=covariance,
        prior_mean=prior_mean,
        prior_std=prior_std,
    )
    chol = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(chol, matrix)
    fisher = whitened.T @ whitened + np.diag(1.0 / np.asarray(prior_std) ** 2)
    cov_a = np.linalg.inv(0.5 * (fisher + fisher.T))
    std_a = np.sqrt(np.clip(np.diag(cov_a), 0.0, None))
    floors = gate["generation"]["posterior_std_floor"]
    std_a[0] = max(float(std_a[0]), float(floors["common_delay_s"]))
    std_a[1] = max(float(std_a[1]), float(floors["log_amplitude_gain"]))
    std_a[2:] = np.maximum(std_a[2:], float(floors["log_amplitude_offset"]))
    device_profile_id = calibration[0].device_profile_id
    view_id = calibration[0].view_id
    view_priors = [
        {
            "device_profile_id": device_profile_id,
            "view_id": view_id,
            "common_delay_prior_mean": float(solution.parameters[0]),
            "common_delay_prior_std": float(std_a[0]),
            "log_amplitude_gain_prior_mean": float(solution.parameters[1]),
            "log_amplitude_gain_prior_std": float(std_a[1]),
            "calibration_set_id": "calibration",
        }
    ]
    offset_priors = []
    for index, frequency in enumerate(frequencies):
        offset_priors.append(
            {
                "device_profile_id": device_profile_id,
                "frequency_hz": float(frequency),
                "log_amplitude_offset_prior_mean": float(solution.parameters[2 + index]),
                "log_amplitude_offset_prior_std": float(std_a[2 + index]),
                "calibration_set_id": "calibration",
            }
        )
    return {
        "view_nuisance_calibration_priors": view_priors,
        "calibration_priors": offset_priors,
        "linear_parameter_names": list(prior_names),
        "posterior_mean": solution.parameters.tolist(),
        "posterior_std": std_a.tolist(),
        "n_calibration_mixtures": len(calibration),
        "augmented_rank": solution.augmented_rank,
        "augmented_condition_number": solution.augmented_condition_number,
    }


def apply_option_a_priors_to_spec(
    spec: S1Parameterization, calibration: Mapping[str, Any]
) -> S1Parameterization:
    names = list(spec.names)
    prior_mean = spec.prior_mean.copy()
    prior_std = spec.prior_std.copy()
    index_map = {int(index): row for row, index in enumerate(spec.prior_indices)}
    view = calibration["view_nuisance_calibration_priors"][0]
    delay_index = names.index("common_delay_s")
    gain_index = names.index("log_amplitude_gain")
    prior_mean[index_map[delay_index]] = float(view["common_delay_prior_mean"])
    prior_std[index_map[delay_index]] = float(view["common_delay_prior_std"])
    prior_mean[index_map[gain_index]] = float(view["log_amplitude_gain_prior_mean"])
    prior_std[index_map[gain_index]] = float(view["log_amplitude_gain_prior_std"])
    for row in calibration["calibration_priors"]:
        name = f"log_amplitude_offset_{int(row['frequency_hz'])}hz"
        index = names.index(name)
        prior_mean[index_map[index]] = float(row["log_amplitude_offset_prior_mean"])
        prior_std[index_map[index]] = float(row["log_amplitude_offset_prior_std"])
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


def apply_known_condition_priors(
    spec: S1Parameterization,
    *,
    t_c: float,
    path_length_m: float,
    h_rh: float,
) -> S1Parameterization:
    """Join measured T/L/RH as solver-allowed conditions into the registered priors."""
    names = list(spec.names)
    prior_mean = spec.prior_mean.copy()
    index_map = {int(index): row for row, index in enumerate(spec.prior_indices)}
    prior_mean[index_map[names.index("t_c")]] = float(t_c)
    prior_mean[index_map[names.index("path_length_m")]] = float(path_length_m)
    prior_mean[index_map[names.index("h_rh")]] = float(h_rh)
    return S1Parameterization(
        names=spec.names,
        scales=spec.scales.copy(),
        lower_bounds=spec.lower_bounds.copy(),
        upper_bounds=spec.upper_bounds.copy(),
        finite_difference_steps=spec.finite_difference_steps.copy(),
        prior_indices=spec.prior_indices.copy(),
        prior_mean=prior_mean,
        prior_std=spec.prior_std.copy(),
    )


def _problem_from_record(record: MixtureRecord) -> S1Problem:
    return S1Problem(
        observation=record.observation.copy(),
        covariance=record.covariance.copy(),
        frequencies_hz=record.frequencies_hz.copy(),
        phase_branch_cycles=record.phase_branch_cycles.copy(),
        observation_std=dict(record.observation_std),
        p_mpa=record.p_mpa,
    )


def _initial_parameters(
    *,
    solver_config: Mapping[str, Any],
    init_index: int,
    record: MixtureRecord,
) -> np.ndarray:
    entry = solver_config["b1_solver_audit"]["frozen_initializations"][int(init_index)]
    # Composition starts from the frozen multi-init set; measured conditions are known.
    return pack_s1_parameters(
        entry["raw3_percent"],
        t_c=float(record.t_c),
        path_length_m=float(record.path_length_m),
        h_rh=float(record.h_rh),
        common_delay_s=0.0,
        log_amplitude_gain=0.0,
        per_frequency_offsets=np.zeros(record.frequencies_hz.size),
    )


def _select_best(solutions: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in solutions if row["success"]]
    pool = successful or solutions
    return min(pool, key=lambda row: float(row["objective"]))


def _component_errors(
    *,
    truth: np.ndarray,
    estimate: np.ndarray,
    success: bool,
    failure_abs_error_percent: float,
) -> dict[str, float]:
    if (
        success
        and estimate.shape == (3,)
        and np.all(np.isfinite(estimate))
    ):
        abs_err = np.abs(estimate - truth)
    else:
        if estimate.shape == (3,) and np.all(np.isfinite(estimate)):
            abs_err = np.abs(estimate - truth)
        else:
            abs_err = np.full(3, float(failure_abs_error_percent), dtype=np.float64)
    return {
        "abs_err_co2": float(abs_err[0]),
        "abs_err_o2": float(abs_err[1]),
        "abs_err_n2": float(abs_err[2]),
    }


def solve_paired_methods(
    *,
    records: Sequence[MixtureRecord],
    solver_config: Mapping[str, Any],
    gate: Mapping[str, Any],
    calibration: Mapping[str, Any],
    s3_truth_by_mixture: Mapping[str, Mapping[str, Any]],
    progress_callback: Any | None = None,
) -> list[dict[str, Any]]:
    """Run S1 then S2 then S3 with exact pairing; failures remain in metrics."""
    settings = build_s1_settings(solver_config)
    base_spec = build_s1_parameterization(solver_config)
    option_a_spec = apply_option_a_priors_to_spec(base_spec, calibration)
    varpro = build_varpro_parameterization(solver_config, option_a_spec)
    phase_limit = float(solver_config["b2_solver_core"]["max_phase_branch_standardized_error"])
    init_indices = list(
        range(len(solver_config["b1_solver_audit"]["frozen_initializations"]))
    )
    failure_penalty = float(gate["generation"]["failure_abs_error_percent"])
    evaluation = [row for row in records if row.split in {"test", "ood"}]
    rows: list[dict[str, Any]] = []

    # Freeze S1 completely before observing S2.
    s1_by_mixture: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(evaluation):
        problem = _problem_from_record(record)
        spec = apply_known_condition_priors(
            option_a_spec,
            t_c=record.t_c,
            path_length_m=record.path_length_m,
            h_rh=record.h_rh,
        )
        candidates = []
        for init_index in init_indices:
            initial = _initial_parameters(
                solver_config=solver_config,
                init_index=init_index,
                record=record,
            )
            solution = solve_s1(problem, initial, spec, settings)
            candidates.append(
                {
                    "method": "S1",
                    "frozen_initialization_index": int(init_index),
                    "success": bool(solution.success),
                    "stop_reason": solution.stop_reason,
                    "objective": float(solution.objective),
                    "forward_calls": int(solution.forward_calls),
                    "iterations": len(solution.iterations),
                    "bound_hit": bool(solution.bound_hit),
                    "bound_parameters": list(solution.bound_parameters),
                    "raw3_percent": solution.raw3_percent.tolist(),
                    "parameters": solution.parameters.tolist(),
                }
            )
        best = _select_best(candidates)
        s1_by_mixture[record.mixture_id] = best
        if progress_callback is not None:
            progress_callback("S1", index + 1, len(evaluation), record.mixture_id)

    s2_by_mixture: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(evaluation):
        problem = _problem_from_record(record)
        spec = apply_known_condition_priors(
            option_a_spec,
            t_c=record.t_c,
            path_length_m=record.path_length_m,
            h_rh=record.h_rh,
        )
        candidates = []
        for init_index in init_indices:
            initial = _initial_parameters(
                solver_config=solver_config,
                init_index=init_index,
                record=record,
            )
            solution = solve_s2(
                problem,
                initial,
                spec,
                settings,
                varpro,
                max_phase_branch_standardized_error=phase_limit,
            )
            candidates.append(
                {
                    "method": "S2",
                    "frozen_initialization_index": int(init_index),
                    "success": bool(solution.success),
                    "stop_reason": solution.stop_reason,
                    "objective": float(solution.objective),
                    "forward_calls": int(solution.forward_calls),
                    "iterations": len(solution.iterations),
                    "bound_hit": bool(solution.bound_hit),
                    "bound_parameters": list(solution.bound_parameters),
                    "raw3_percent": solution.raw3_percent.tolist(),
                    "parameters": solution.parameters.tolist(),
                    "linear_parameters": solution.linear_parameters.tolist(),
                }
            )
        best = _select_best(candidates)
        s2_by_mixture[record.mixture_id] = best
        if progress_callback is not None:
            progress_callback("S2", index + 1, len(evaluation), record.mixture_id)

    for index, record in enumerate(evaluation):
        problem = _problem_from_record(record)
        spec = apply_known_condition_priors(
            option_a_spec,
            t_c=record.t_c,
            path_length_m=record.path_length_m,
            h_rh=record.h_rh,
        )
        truth_nuisance = s3_truth_by_mixture[record.mixture_id]
        truth_linear = np.concatenate(
            (
                [float(truth_nuisance["common_delay_s"]), float(truth_nuisance["log_amplitude_gain"])],
                np.asarray(truth_nuisance["log_amplitude_offsets"], dtype=np.float64),
            )
        )
        candidates = []
        for init_index in init_indices:
            initial = _initial_parameters(
                solver_config=solver_config,
                init_index=init_index,
                record=record,
            )
            solution = solve_s3(
                problem,
                initial,
                spec,
                settings,
                varpro,
                truth_linear_parameters=truth_linear,
                max_phase_branch_standardized_error=phase_limit,
            )
            candidates.append(
                {
                    "method": "S3",
                    "frozen_initialization_index": int(init_index),
                    "success": bool(solution.success),
                    "stop_reason": solution.stop_reason,
                    "objective": float(solution.objective),
                    "forward_calls": int(solution.forward_calls),
                    "iterations": len(solution.iterations),
                    "bound_hit": bool(solution.bound_hit),
                    "bound_parameters": list(solution.bound_parameters),
                    "raw3_percent": solution.raw3_percent.tolist(),
                    "parameters": solution.parameters.tolist(),
                    "linear_parameters": solution.linear_parameters.tolist(),
                }
            )
        best_s3 = _select_best(candidates)
        if progress_callback is not None:
            progress_callback("S3", index + 1, len(evaluation), record.mixture_id)

        truth = record.raw3_percent
        crb = composition_crb_o2_std(
            problem,
            pack_s1_parameters(
                truth,
                t_c=record.t_c,
                path_length_m=record.path_length_m,
                h_rh=record.h_rh,
                common_delay_s=float(truth_nuisance["common_delay_s"]),
                log_amplitude_gain=float(truth_nuisance["log_amplitude_gain"]),
                per_frequency_offsets=truth_nuisance["log_amplitude_offsets"],
            ),
            spec,
        )
        paired = {
            "mixture_id": record.mixture_id,
            "split": record.split,
            "design_condition_id": record.design_condition_id,
            "truth_raw3_percent": truth.tolist(),
            "crb_o2_std_percent": float(crb),
            "data_split_seed_index": {"test": 1, "ood": 2}[record.split],
        }
        for method, best in (
            ("S1", s1_by_mixture[record.mixture_id]),
            ("S2", s2_by_mixture[record.mixture_id]),
            ("S3", best_s3),
        ):
            estimate = np.asarray(best["raw3_percent"], dtype=np.float64)
            errors = _component_errors(
                truth=truth,
                estimate=estimate,
                success=bool(best["success"]),
                failure_abs_error_percent=failure_penalty,
            )
            paired[method] = {
                **best,
                **errors,
                "relative_crb_efficiency_o2": (
                    (crb**2) / max(errors["abs_err_o2"] ** 2, 1e-30)
                ),
            }
        rows.append(paired)
    return rows


def _p90(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("p90 requires a non-empty sample")
    return float(np.quantile(array, 0.9, method="linear"))


def _domain_metrics(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, float]:
    abs_o2 = [float(row[method]["abs_err_o2"]) for row in rows]
    abs_co2 = [float(row[method]["abs_err_co2"]) for row in rows]
    abs_n2 = [float(row[method]["abs_err_n2"]) for row in rows]
    failures = [0.0 if row[method]["success"] else 1.0 for row in rows]
    bound_hits = [1.0 if row[method]["bound_hit"] else 0.0 for row in rows]
    by_condition: dict[str, list[float]] = {}
    for row in rows:
        by_condition.setdefault(str(row["design_condition_id"]), []).append(
            float(row[method]["abs_err_o2"])
        )
    worst_group = max(float(np.mean(values)) for values in by_condition.values())
    return {
        "n": float(len(rows)),
        "p90_abs_err_o2": _p90(abs_o2),
        "mae_o2": float(np.mean(abs_o2)),
        "mae_co2": float(np.mean(abs_co2)),
        "mae_n2": float(np.mean(abs_n2)),
        "convergence_failure_rate": float(np.mean(failures)),
        "bound_hit_rate": float(np.mean(bound_hits)),
        "worst_group_error_o2": worst_group,
        "mean_iterations": float(np.mean([row[method]["iterations"] for row in rows])),
        "mean_forward_calls": float(
            np.mean([row[method]["forward_calls"] for row in rows])
        ),
        "mean_relative_crb_efficiency_o2": float(
            np.mean([row[method]["relative_crb_efficiency_o2"] for row in rows])
        ),
    }


def _relative_improvement(baseline: float, candidate: float) -> float:
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("baseline P90 must be positive and finite")
    return float((baseline - candidate) / baseline)


def _non_degeneration_thresholds(gate: Mapping[str, Any]) -> dict[str, float]:
    policy = gate["solver_gate"]["non_degeneration"]
    if policy["semantics"] != "maximum_absolute_increase":
        raise ValueError("non-degeneration semantics must be maximum_absolute_increase")
    thresholds = {name: float(value) for name, value in policy["thresholds"].items()}
    if set(thresholds) != set(NON_DEGENERATION_METRICS):
        raise ValueError("non-degeneration thresholds must cover every registered metric")
    if any(not math.isfinite(value) or value < 0.0 for value in thresholds.values()):
        raise ValueError("non-degeneration thresholds must be finite and non-negative")
    return thresholds


def stratified_paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_resamples: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the paired O2 P90 relative improvement within design strata."""
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must lie in (0, 1)")
    strata: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        strata.setdefault(str(row["design_condition_id"]), []).append(row)
    rng = _as_protocol_rng(seed)
    statistics = []
    stratum_items = list(strata.items())
    for _ in range(int(n_resamples)):
        sample: list[Mapping[str, Any]] = []
        for _condition_id, members in stratum_items:
            choices = rng.integers(0, len(members), size=len(members))
            sample.extend(members[int(index)] for index in choices)
        s1 = _domain_metrics(sample, "S1")["p90_abs_err_o2"]
        s2 = _domain_metrics(sample, "S2")["p90_abs_err_o2"]
        statistics.append(_relative_improvement(s1, s2))
    array = np.asarray(statistics, dtype=np.float64)
    alpha = 1.0 - float(ci_level)
    return {
        "n_resamples": int(n_resamples),
        "ci_level": float(ci_level),
        "point_estimate": _relative_improvement(
            _domain_metrics(rows, "S1")["p90_abs_err_o2"],
            _domain_metrics(rows, "S2")["p90_abs_err_o2"],
        ),
        "ci_lower": float(np.quantile(array, alpha / 2.0, method="linear")),
        "ci_upper": float(np.quantile(array, 1.0 - alpha / 2.0, method="linear")),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
    }


def evaluate_solver_gate(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    protocol: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    delta = float(gate["solver_gate"]["delta_practical"])
    non_degeneration_thresholds = _non_degeneration_thresholds(gate)
    bootstrap_cfg = protocol["initialization_and_statistics"]["bootstrap"]
    domain_reports = {}
    bootstrap_reports = {}
    issues: list[str] = []
    for domain in gate["solver_gate"]["domains"]:
        rows = [row for row in paired_rows if row["split"] == domain]
        if not rows:
            issues.append(f"missing evaluation rows for domain {domain}")
            continue
        s1 = _domain_metrics(rows, "S1")
        s2 = _domain_metrics(rows, "S2")
        s3 = _domain_metrics(rows, "S3")
        improvement = _relative_improvement(s1["p90_abs_err_o2"], s2["p90_abs_err_o2"])
        bootstrap = stratified_paired_bootstrap(
            rows,
            n_resamples=int(bootstrap_cfg["n_resamples"]),
            ci_level=float(bootstrap_cfg["ci_level"]),
            seed=int(protocol["split_protocol"]["split_seeds"][{"test": 1, "ood": 2}[domain]]),
        )
        degradations = {
            "mae_o2": s2["mae_o2"] - s1["mae_o2"],
            "mae_co2": s2["mae_co2"] - s1["mae_co2"],
            "mae_n2": s2["mae_n2"] - s1["mae_n2"],
            "convergence_failure_rate": (
                s2["convergence_failure_rate"] - s1["convergence_failure_rate"]
            ),
            "worst_group_error_o2": (
                s2["worst_group_error_o2"] - s1["worst_group_error_o2"]
            ),
            "bound_hit_rate": s2["bound_hit_rate"] - s1["bound_hit_rate"],
        }
        domain_ok = (
            improvement > delta
            and bootstrap["ci_lower"] > delta
            and all(
                degradations[name] <= non_degeneration_thresholds[name]
                for name in NON_DEGENERATION_METRICS
            )
        )
        if improvement <= delta:
            issues.append(f"{domain}: O2 P90 relative improvement {improvement} <= {delta}")
        if bootstrap["ci_lower"] <= delta:
            issues.append(
                f"{domain}: bootstrap CI lower bound {bootstrap['ci_lower']} <= {delta}"
            )
        for name, value in degradations.items():
            threshold = non_degeneration_thresholds[name]
            if value > threshold:
                issues.append(f"{domain}: {name} absolute increase {value} > {threshold}")
        domain_reports[domain] = {
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "paired_relative_improvement_o2_p90": improvement,
            "non_degeneration": {
                "semantics": "maximum_absolute_increase",
                "thresholds": non_degeneration_thresholds,
                "observed_increases": degradations,
            },
            "passed": domain_ok,
        }
        bootstrap_reports[domain] = bootstrap

    passed = not issues
    if passed:
        verdict = "mei3_varpro_supported"
    else:
        # Structure already supported by Phase A; retain full-parameter baseline.
        verdict = "mei3_full_parameter_baseline_retained"
    return {
        "phase": B4_PHASE,
        "verdict": verdict,
        "passed_solver_gate": passed,
        "delta_practical": delta,
        "issues": issues,
        "domain_reports": domain_reports,
        "bootstrap_report": bootstrap_reports,
        "primary_comparison": ["S1", "S2"],
        "upper_bound_only": ["S3"],
        "claim_scope": "registered_simulation_domain_only",
    }


def run_b4_formal_comparison(
    *,
    project_root,
    protocol: Mapping[str, Any],
    gate: Mapping[str, Any],
    solver_config: Mapping[str, Any],
    design_space: Mapping[str, Any],
    stage_status: Mapping[str, Any],
    mixture_limit_per_split: int | None = None,
    bootstrap_n_resamples_override: int | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Authorization-gated formal generation and paired comparison."""
    decision = assess_b4_execution_authorization(
        protocol_config={
            **protocol,
            "authorizations": {
                **gate.get("authorizations", {}),
                **protocol.get("authorizations", {}),
                **((stage_status.get("mei3") or {}).get("authorizations") or {}),
            },
        },
        current_stage_status=stage_status,
    )
    if not decision["authorized"]:
        return {
            "authorized": False,
            "decision": decision,
            "formal_data_generated": False,
        }
    auth = (stage_status.get("mei3") or {}).get("authorizations") or {}
    if auth.get("registered_sparse_simulation_generation") != B4_AUTHORIZED_VALUE:
        raise RuntimeError("B4 authorized decision inconsistent with stage_status")

    dataset = generate_registered_sparse_dataset(
        protocol=protocol,
        gate=gate,
        design_space=design_space,
        mixture_limit_per_split=mixture_limit_per_split,
    )
    calibration = calibrate_option_a_priors(
        records=dataset["records"],
        solver_config=solver_config,
        gate=gate,
    )
    s3_truth = {
        row["mixture_id"]: row for row in dataset["s3_truth_nuisance"]
    }
    paired_rows = solve_paired_methods(
        records=dataset["records"],
        solver_config=solver_config,
        gate=gate,
        calibration=calibration,
        s3_truth_by_mixture=s3_truth,
        progress_callback=progress_callback,
    )
    protocol_for_gate = dict(protocol)
    if bootstrap_n_resamples_override is not None:
        protocol_for_gate = {
            **protocol,
            "initialization_and_statistics": {
                **protocol["initialization_and_statistics"],
                "bootstrap": {
                    **protocol["initialization_and_statistics"]["bootstrap"],
                    "n_resamples": int(bootstrap_n_resamples_override),
                },
            },
        }
    gate_report = evaluate_solver_gate(
        paired_rows,
        protocol=protocol_for_gate,
        gate=gate,
    )
    return {
        "authorized": True,
        "decision": decision,
        "formal_data_generated": True,
        "dataset_meta": dataset["meta"],
        "mixtures": dataset["mixtures"],
        "observation_rows": dataset["observation_rows"],
        "covariance_blocks": dataset["covariance_blocks"],
        "s3_truth_nuisance": dataset["s3_truth_nuisance"],
        "calibration": calibration,
        "paired_rows": paired_rows,
        "gate_report": gate_report,
        "smoke_mode": mixture_limit_per_split is not None,
    }


__all__ = [
    "MixtureRecord",
    "apply_known_condition_priors",
    "apply_option_a_priors_to_spec",
    "calibrate_option_a_priors",
    "evaluate_solver_gate",
    "generate_registered_sparse_dataset",
    "run_b4_formal_comparison",
    "solve_paired_methods",
    "stratified_paired_bootstrap",
]
