"""Jacobian 可辨识性审计（§10.7，采样声明口径）。

joint（目标+nuisance 同估）的 nuisance 投影只对堆叠口径有意义：单 horizon
只有 3 个输出观测而 nuisance 已有 4 列，行满秩时 nuisance 列空间吸收整个
输出空间，投影残差恒为零（rank/条件数退化）。逐 horizon 的可辨识性由
fixed 切向矩阵承载，joint 可辨识性依赖 horizon 间堆叠，二者口径不同。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from gf.sim.a2_dynamic_dataset import DynamicDataset
from gf.sim.a2_dynamic_physics import evaluate_shared_physics, simulate_first_order_series
from gf.sim.a2_dynamic_audit._shared import (
    DEVELOPMENT_SPLITS,
    EARLY_HORIZONS,
    FAMILIES,
    PURGE_COMPOSITION,
    TARGET_TANGENT_DIRECTIONS,
    _horizon_indices,
)


def _audit_jacobian(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
) -> dict[str, Any]:
    """有限差分 Jacobian 审计（采样声明口径，F5 扩面）。

    采样覆盖 train / val / stress_val 三个 split、六 family，每 family 每
    split 确定性等距抽取 12 行（不足 12 行时全取）；每个样本计算堆叠
    （P015/P030/P060）与逐 horizon 两类矩阵。所有结论是采样结论，不是
    全量结论；``sampled_rows / total_rows`` 显式声明覆盖规模。
    """

    early_endpoints = _horizon_indices(dataset.time_s, dataset.records, data_config)
    samples: list[dict[str, Any]] = []
    sampled_row_counts: dict[str, int] = {}
    total_row_counts: dict[str, int] = {}
    for family in FAMILIES:
        for split in DEVELOPMENT_SPLITS:
            family_indices = dataset.indices(family=family, split=split)
            total_row_counts[f"{family}/{split}"] = int(family_indices.size)
            selected = family_indices[:: max(1, len(family_indices) // 12)]
            selected = selected[:12]
            if selected.size == 0:
                raise ValueError(
                    f"jacobian audit has no auditable rows for family {family!r} split {split!r}"
                )
            sampled_row_counts[f"{family}/{split}"] = int(selected.size)
            for row in selected:
                endpoints = [
                    int(early_endpoints[horizon][row]) for horizon in EARLY_HORIZONS
                ]
                sample = _jacobian_sample(
                    dataset,
                    data_config,
                    row=int(row),
                    endpoints=endpoints,
                    family=family,
                    split=split,
                )
                samples.append(sample)
    fixed_ranks = np.asarray(
        [item["fixed_rank"] for item in samples], dtype=np.float64
    )
    fixed_conditions = np.asarray(
        [item["fixed_condition_number"] for item in samples], dtype=np.float64
    )
    joint_parameter_ranks = np.asarray(
        [item["joint_parameter_rank"] for item in samples], dtype=np.float64
    )
    joint_target_ranks = np.asarray(
        [item["joint_target_rank"] for item in samples], dtype=np.float64
    )
    joint_target_conditions = np.asarray(
        [item["joint_target_condition_number"] for item in samples], dtype=np.float64
    )
    physics_gate = eval_config["qualification_gates"]["physics_and_schema"]
    rank_fraction = float(physics_gate["min_jacobian_full_rank_fraction"])
    max_condition = float(physics_gate["max_jacobian_p95_condition_number"])
    target_rank = len(TARGET_TANGENT_DIRECTIONS)
    joint_parameter_rank = int(samples[0]["joint_parameter_expected_rank"])
    fixed_rank_fraction = float(np.mean(fixed_ranks >= target_rank))
    joint_parameter_rank_fraction = float(
        np.mean(joint_parameter_ranks >= joint_parameter_rank)
    )
    joint_target_rank_fraction = float(np.mean(joint_target_ranks >= target_rank))
    fixed_p95 = float(np.percentile(fixed_conditions, 95))
    joint_target_p95 = float(np.percentile(joint_target_conditions, 95))
    checks = {
        "fixed_full_rank_fraction": fixed_rank_fraction >= rank_fraction,
        "joint_parameter_full_rank_fraction": joint_parameter_rank_fraction >= rank_fraction,
        "joint_target_full_rank_fraction": joint_target_rank_fraction >= rank_fraction,
        "fixed_p95_condition_number": fixed_p95 <= max_condition,
        "joint_target_p95_condition_number": joint_target_p95 <= max_condition,
    }
    per_horizon = _per_horizon_jacobian_summary(samples)
    for horizon, summary in per_horizon.items():
        checks[f"fixed_full_rank_fraction_{horizon}"] = (
            summary["fixed_full_rank_fraction"] >= rank_fraction
        )
        checks[f"fixed_p95_condition_number_{horizon}"] = (
            summary["fixed_condition_number_p95"] <= max_condition
        )
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "sample_count": len(samples),
        "sampled_row_counts": sampled_row_counts,
        "total_row_counts": total_row_counts,
        "sampling_strategy": (
            "deterministic_equidistant_12_rows_per_family_per_split_train_val_stress_val"
        ),
        "sampled_rows": len(samples),
        "total_rows": int(sum(total_row_counts.values())),
        "scope_note": (
            "rank fractions and condition numbers are computed on the declared "
            "sample set, not on all development rows"
        ),
        "fixed_full_rank_fraction": fixed_rank_fraction,
        "joint_parameter_full_rank_fraction": joint_parameter_rank_fraction,
        "joint_target_full_rank_fraction": joint_target_rank_fraction,
        "fixed_condition_number_p50": float(np.percentile(fixed_conditions, 50)),
        "fixed_condition_number_p95": fixed_p95,
        "joint_target_condition_number_p50": float(np.percentile(joint_target_conditions, 50)),
        "joint_target_condition_number_p95": joint_target_p95,
        "gate_minimum_full_rank_fraction": rank_fraction,
        "gate_maximum_p95_condition_number": max_condition,
        "basis": "per_horizon_fixed_target_tangent_and_stacked_nuisance_projection_finite_difference",
        "per_horizon": per_horizon,
        "parameter_columns": [
            "target_tangent_Ar_minus_He",
            "target_tangent_Ar_minus_CO2",
            "tau_mix_s",
            "acoustic_path_scale",
            "tcs_response_scale",
            "ndir_absorbance_scale",
        ],
        "samples": samples,
    }


def _jacobian_sample(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    *,
    row: int,
    endpoints: Sequence[int],
    family: str,
    split: str,
) -> dict[str, Any]:
    fixed_by_horizon: dict[str, np.ndarray] = {}
    joint_by_horizon: dict[str, np.ndarray] = {}
    for horizon, endpoint in zip(EARLY_HORIZONS, endpoints):
        fixed, joint = _equilibrium_jacobian_block(
            dataset,
            data_config,
            row=row,
            endpoint=int(endpoint),
        )
        fixed_by_horizon[horizon] = fixed
        joint_by_horizon[horizon] = joint
    stacked_fixed = np.vstack([fixed_by_horizon[h] for h in EARLY_HORIZONS])
    stacked_joint = np.vstack([joint_by_horizon[h] for h in EARLY_HORIZONS])
    sample: dict[str, Any] = {
        "family": family,
        "split": split,
        "row": row,
        "horizons": list(EARLY_HORIZONS),
        "fixed_rank": int(np.linalg.matrix_rank(stacked_fixed)),
        "fixed_expected_rank": len(TARGET_TANGENT_DIRECTIONS),
        "fixed_condition_number": _condition_number(stacked_fixed),
        "joint_parameter_rank": int(np.linalg.matrix_rank(stacked_joint)),
        "joint_parameter_expected_rank": int(stacked_joint.shape[1]),
    }
    for horizon in EARLY_HORIZONS:
        fixed_h = fixed_by_horizon[horizon]
        sample[f"fixed_rank_{horizon}"] = int(np.linalg.matrix_rank(fixed_h))
        sample[f"fixed_condition_number_{horizon}"] = _condition_number(fixed_h)
    # joint 的 nuisance 投影只在堆叠口径下有意义（见模块 docstring）；
    # 逐 horizon 的 joint_target 统计不产生，防止把欠定系统的数值假象
    # 当成可辨识性证据。
    target_columns = stacked_joint[:, : len(TARGET_TANGENT_DIRECTIONS)]
    nuisance_columns = stacked_joint[:, len(TARGET_TANGENT_DIRECTIONS) :]
    nuisance_projection = nuisance_columns @ np.linalg.lstsq(
        nuisance_columns,
        target_columns,
        rcond=None,
    )[0]
    projected_target = target_columns - nuisance_projection
    sample["joint_nuisance_rank"] = int(np.linalg.matrix_rank(nuisance_columns))
    sample["joint_target_rank"] = int(np.linalg.matrix_rank(projected_target))
    sample["joint_target_expected_rank"] = len(TARGET_TANGENT_DIRECTIONS)
    sample["joint_target_condition_number"] = _condition_number(projected_target)
    return sample


def _per_horizon_jacobian_summary(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for horizon in EARLY_HORIZONS:
        fixed_ranks = np.asarray(
            [item[f"fixed_rank_{horizon}"] for item in samples], dtype=np.float64
        )
        fixed_conditions = np.asarray(
            [item[f"fixed_condition_number_{horizon}"] for item in samples],
            dtype=np.float64,
        )
        target_rank = len(TARGET_TANGENT_DIRECTIONS)
        summary[horizon] = {
            "sample_count": int(len(samples)),
            "fixed_full_rank_fraction": float(np.mean(fixed_ranks >= target_rank)),
            "fixed_condition_number_p50": float(np.percentile(fixed_conditions, 50)),
            "fixed_condition_number_p95": float(np.percentile(fixed_conditions, 95)),
        }
    return summary


def _sensor_responses_at_endpoint(
    coefficient: np.ndarray,
    endpoint: int,
    *,
    dt_s: float,
    tau_mix_s: float,
    tau_transport_s: Sequence[float],
) -> np.ndarray:
    mix_response = simulate_first_order_series(
        coefficient,
        dt_s=dt_s,
        tau_s=tau_mix_s,
        initial_state=0.0,
    )
    responses = np.asarray(
        [
            simulate_first_order_series(
                mix_response,
                dt_s=dt_s,
                tau_s=float(tau),
                initial_state=0.0,
            )[endpoint]
            for tau in tau_transport_s
        ],
        dtype=np.float64,
    )
    if np.any(responses <= 1.0e-12):
        raise ValueError(f"Jacobian response is not positive at endpoint {endpoint}")
    return responses


def _equilibrium_observation(
    data_config: Mapping[str, Any],
    record: Mapping[str, Any],
    target: np.ndarray,
    responses: np.ndarray,
    *,
    temperature_k: float,
    pressure_pa: float,
    acoustic_path_scale: float,
    tcd_response_scale: float,
    ndir_absorbance_scale: float,
) -> np.ndarray:
    ultrasonic = next(
        item for item in data_config["hardware_profiles"]["ultrasonic"]["candidates"]
        if str(item["ultrasonic_profile_id"]) == str(record["ultrasonic_profile_id"])
    )
    model_id = str(data_config["physics_reference"]["eos"]["sound_speed_model_id"])
    composition = np.asarray(target, dtype=np.float64)
    response_array = np.asarray(responses, dtype=np.float64)
    if composition.shape != (3,) or response_array.shape != (3,):
        raise ValueError("equilibrium observation expects one target and three sensor responses")
    local = np.asarray(
        [PURGE_COMPOSITION + response * (composition - PURGE_COMPOSITION) for response in response_array],
        dtype=np.float64,
    )
    shared = evaluate_shared_physics(
        local,
        temperature_k=temperature_k,
        pressure_pa=pressure_pa,
        path_length_m=float(ultrasonic["path_length_m"]) * acoustic_path_scale,
        sound_speed_model_id=model_id,
    )
    return np.asarray(
        [
            shared["tof_s"][0],
            shared["thermal_voltage_v"][1] * tcd_response_scale,
            shared["ndir_voltage_v"][2] * ndir_absorbance_scale,
        ],
        dtype=np.float64,
    )


def _equilibrium_jacobian_block(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    *,
    row: int,
    endpoint: int,
) -> tuple[np.ndarray, np.ndarray]:
    """单 endpoint 的目标-响应 Jacobian。

    fixed 块含两个目标切向列（Ar-He / Ar-CO₂）；joint 块额外含四个共享
    nuisance 列（tau_mix / 声程 / TCS 标度 / NDIR 标度），供堆叠口径的
    nuisance 投影审计使用。
    """
    if endpoint < 0 or endpoint >= dataset.timesteps:
        raise ValueError(f"Jacobian endpoint is invalid at row {row}: {endpoint}")
    parameters = dataset.privileged_parameters[row]
    dt_s = float(dataset.manifest["dt_s"])
    base_target = np.asarray([60.0, 30.0, 10.0], dtype=np.float64)
    tau_mix = float(parameters[0])
    tau_transport = np.asarray(parameters[1:4], dtype=np.float64)
    temperature_k = float(parameters[4])
    pressure_pa = float(parameters[5])
    acoustic_scale = float(parameters[6])
    tcd_scale = float(parameters[7])
    ndir_scale = float(parameters[8])
    target_step = 1.0e-3
    nuisance_step = 1.0e-3
    tau_step = max(1.0e-3, abs(tau_mix) * 1.0e-3)
    output_scale = np.asarray([1.0e-6, 1.0, 1.0], dtype=np.float64)
    responses = _sensor_responses_at_endpoint(
        dataset.inlet_coefficient[row],
        endpoint,
        dt_s=dt_s,
        tau_mix_s=tau_mix,
        tau_transport_s=tau_transport,
    )

    def observation(
        target: np.ndarray,
        response_values: np.ndarray = responses,
        *,
        path_multiplier: float = 1.0,
        tcd_multiplier: float = 1.0,
        ndir_multiplier: float = 1.0,
    ) -> np.ndarray:
        return _equilibrium_observation(
            data_config,
            dataset.records[row],
            target,
            response_values,
            temperature_k=temperature_k,
            pressure_pa=pressure_pa,
            acoustic_path_scale=acoustic_scale * path_multiplier,
            tcd_response_scale=tcd_scale * tcd_multiplier,
            ndir_absorbance_scale=ndir_scale * ndir_multiplier,
        )

    target_columns: list[np.ndarray] = []
    for direction in TARGET_TANGENT_DIRECTIONS:
        plus = base_target + target_step * direction
        minus = base_target - target_step * direction
        target_columns.append(
            (observation(plus) - observation(minus)) / (2.0 * target_step)
        )
    tau_plus = _sensor_responses_at_endpoint(
        dataset.inlet_coefficient[row],
        endpoint,
        dt_s=dt_s,
        tau_mix_s=tau_mix + tau_step,
        tau_transport_s=tau_transport,
    )
    tau_minus = _sensor_responses_at_endpoint(
        dataset.inlet_coefficient[row],
        endpoint,
        dt_s=dt_s,
        tau_mix_s=max(0.0, tau_mix - tau_step),
        tau_transport_s=tau_transport,
    )
    nuisance_columns = [
        (observation(base_target, tau_plus) - observation(base_target, tau_minus))
        / (2.0 * tau_step),
        (observation(base_target, path_multiplier=1.0 + nuisance_step)
         - observation(base_target, path_multiplier=1.0 - nuisance_step))
        / (2.0 * nuisance_step),
        (observation(base_target, tcd_multiplier=1.0 + nuisance_step)
         - observation(base_target, tcd_multiplier=1.0 - nuisance_step))
        / (2.0 * nuisance_step),
        (observation(base_target, ndir_multiplier=1.0 + nuisance_step)
         - observation(base_target, ndir_multiplier=1.0 - nuisance_step))
        / (2.0 * nuisance_step),
    ]
    fixed = np.column_stack(target_columns) / output_scale[:, None]
    joint = (
        np.column_stack(target_columns + nuisance_columns) / output_scale[:, None]
    )
    return fixed, joint


def _stacked_equilibrium_jacobians(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    *,
    row: int,
    endpoints: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """把注册的三个早期 horizon 的 Jacobian 块堆叠成一个矩阵。"""
    if len(endpoints) != len(EARLY_HORIZONS):
        raise ValueError("Jacobian stacking requires the registered early horizons")
    fixed_blocks: list[np.ndarray] = []
    joint_blocks: list[np.ndarray] = []
    for endpoint in endpoints:
        fixed, joint = _equilibrium_jacobian_block(
            dataset,
            data_config,
            row=row,
            endpoint=int(endpoint),
        )
        fixed_blocks.append(fixed)
        joint_blocks.append(joint)
    return np.vstack(fixed_blocks), np.vstack(joint_blocks)


def _condition_number(matrix: np.ndarray) -> float:
    singular_values = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if singular_values.size == 0 or singular_values[-1] <= 1.0e-15:
        return math.inf
    return float(singular_values[0] / singular_values[-1])
