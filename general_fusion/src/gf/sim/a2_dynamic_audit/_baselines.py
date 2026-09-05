"""基线族与 oracle 审计（B-LAST / O-EQ / O-KIN / O-KIN-OBS，难度门证据）。"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.neural_network import MLPRegressor

from gf.dl.evaluation import evaluate_output_constraints, evaluate_predictions
from gf.sim.a2_dynamic_dataset import DynamicDataset, _calibration_profiles
from gf.sim.a2_dynamic_physics import simulate_first_order_series
from gf.sim.a2_sensor_devices import (
    NDIRDeviceProfile,
    NDIR_BASELINE_V,
    SensorDeviceError,
    estimate_ndir_equilibrium_co2_series,
)
from gf.sim.a2_dynamic_audit._heos_interpolation import (
    _registered_heos_interpolated_tof,
)
from gf.sim.a2_dynamic_audit._shared import (
    AUDIT_HORIZONS,
    DEVELOPMENT_SPLITS,
    EARLY_HORIZONS,
    FAMILIES,
    HEOS_INTERPOLATION_TOF_TOLERANCE_S,
    OBSERVED_ADMISSION_DRIFT_MINUTES,
    OBSERVED_ADMISSION_SIGMA_FACTOR,
    PURGE_COMPOSITION,
    TARGET_TOTAL,
    _horizon_indices,
)


def _audit_baselines(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any],
    a2h_config: Mapping[str, Any],
) -> dict[str, Any]:
    difficulty_gate = eval_config["qualification_gates"]["dynamic_difficulty"]
    late_reference_horizon = str(difficulty_gate["late_reference_horizon"])
    secondary_late_reference_horizon = str(
        difficulty_gate["secondary_late_reference_horizon"]
    )
    if str(difficulty_gate["pairing"]) != "valid_at_both_horizons":
        raise ValueError(
            "dynamic difficulty gate must freeze the valid_at_both_horizons pairing rule"
        )
    if late_reference_horizon not in AUDIT_HORIZONS or (
        secondary_late_reference_horizon not in AUDIT_HORIZONS
    ):
        raise ValueError("late reference horizons must be registered audit horizons")
    experiment_pilot = experiment_config.get("pilot")
    if not isinstance(experiment_pilot, Mapping):
        raise ValueError("baseline audit requires experiment_config.pilot")
    noise_base = np.asarray(
        experiment_pilot["observation_noise_std_by_sensor"],
        dtype=np.float64,
    )
    if noise_base.shape != (len(data_config["sensor_ids"]),) or not np.isfinite(noise_base).all():
        raise ValueError("experiment pilot observation_noise_std_by_sensor is invalid")
    calibrations = _calibration_profiles(data_config, a2h_config)
    horizon_indices = _horizon_indices(dataset.time_s, dataset.records, data_config)
    target_ranges = np.asarray(
        [float(eval_config["target_ranges"][name]) for name in data_config["target_names"]],
        dtype=np.float64,
    )
    admission_budgets = _observed_admission_budgets(
        dataset,
        np.arange(dataset.sample_count, dtype=np.int64),
        data_config=data_config,
        calibrations=calibrations,
        noise_base=noise_base,
    )
    family_results: dict[str, Any] = {}
    for family in FAMILIES:
        family_result: dict[str, Any] = {
            "baseline_fit_status": "NOT_RUN",
            "oracle_fit_status": "NOT_RUN",
            "horizons": {},
            "difficulty": {},
            "fit_diagnostics": {},
        }
        baseline_fit_ok = True
        oracle_fit_ok = True
        b_last_val_store: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        o_kin_obs_val_store: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        kinetic_cache: dict[int, dict[str, np.ndarray]] = {}
        heos_interpolation_cache: dict[tuple[float, float], np.ndarray] = {}
        for horizon in AUDIT_HORIZONS:
            raw_indices_by_split = {
                split: dataset.indices(family=family, split=split)
                for split in DEVELOPMENT_SPLITS
            }
            indices_by_split = {
                split: raw_indices[
                    horizon_indices[horizon][raw_indices] >= 0
                ]
                for split, raw_indices in raw_indices_by_split.items()
            }
            endpoints = {
                split: _endpoint_features(dataset.signals, indices, horizon_indices[horizon][indices])
                for split, indices in indices_by_split.items()
            }
            equilibrium = {
                split: _equilibrium_features(
                    dataset.equilibrium_reference_signals,
                    indices,
                    horizon_indices[horizon][indices],
                )
                for split, indices in indices_by_split.items()
            }
            b_last_predictions, b_last_fit = _fit_small_mlp(
                endpoints["train"],
                dataset.target[indices_by_split["train"]],
                [endpoints[split] for split in DEVELOPMENT_SPLITS],
            )
            # O-EQ 与 B-LAST 使用完全相同的 _fit_small_mlp（同结构、同 seed、同标准化），
            # 只有输入特征不同：clean 平衡信号 vs 带噪端点（F3 方案 A）。
            o_eq_predictions, o_eq_fit = _fit_small_mlp(
                equilibrium["train"],
                dataset.target[indices_by_split["train"]],
                [equilibrium[split] for split in DEVELOPMENT_SPLITS],
            )
            inversion_failures: list[dict[str, Any]] = []
            o_kin_predictions = {
                split: _kinetic_oracle_predictions(
                    dataset,
                    indices,
                    horizon_indices[horizon][indices],
                    data_config=data_config,
                    kinetic_cache=kinetic_cache,
                    heos_interpolation_cache=heos_interpolation_cache,
                )
                for split, indices in indices_by_split.items()
            }
            o_kin_obs_predictions = {
                split: _kinetic_oracle_predictions(
                    dataset,
                    indices,
                    horizon_indices[horizon][indices],
                    data_config=data_config,
                    kinetic_cache=kinetic_cache,
                    heos_interpolation_cache=heos_interpolation_cache,
                    input_mode="observed",
                    admission_budgets=admission_budgets,
                    inversion_failures=inversion_failures,
                )
                for split, indices in indices_by_split.items()
            }
            failure_fraction_by_split = {
                split: (
                    float(
                        np.mean(
                            ~np.isfinite(o_kin_obs_predictions[split]).all(axis=1)
                        )
                    )
                    if indices_by_split[split].size
                    else None
                )
                for split in DEVELOPMENT_SPLITS
            }
            o_eq_finite = all(
                np.isfinite(prediction).all()
                for prediction in o_eq_predictions
            )
            o_kin_finite = all(
                np.isfinite(prediction).all()
                for prediction in o_kin_predictions.values()
            )
            # 门控 oracle 是 O-KIN-OBS：val 无任何成功反演行的 horizon 判 FAIL，
            # 失败率本身按行显式报告，不用放宽反演容差救回（F2）。
            o_kin_obs_usable = failure_fraction_by_split["val"] is None or (
                failure_fraction_by_split["val"] < 1.0
            )
            baseline_fit_ok = (
                baseline_fit_ok
                and b_last_fit["status"] == "PASS"
                and o_eq_fit["status"] == "PASS"
                and o_eq_finite
            )
            oracle_fit_ok = oracle_fit_ok and o_kin_finite and o_kin_obs_usable
            family_result["fit_diagnostics"][horizon] = {
                "B-LAST": b_last_fit,
                "O-EQ": o_eq_fit,
                "O-KIN": {
                    "status": "PASS" if o_kin_finite else "FAIL",
                    "finite_predictions": o_kin_finite,
                },
                "O-KIN-OBS": {
                    "status": "PASS" if o_kin_obs_usable else "FAIL",
                    "inversion_failure_fraction_by_split": failure_fraction_by_split,
                    "inversion_failure_count": len(inversion_failures),
                },
            }
            split_metrics: dict[str, Any] = {}
            for split_index, split in enumerate(DEVELOPMENT_SPLITS):
                indices = indices_by_split[split]
                if indices.size == 0:
                    split_metrics[split] = None
                    continue
                targets = dataset.target[indices]
                observed_ok = np.isfinite(
                    o_kin_obs_predictions[split]
                ).all(axis=1)
                split_metrics[split] = {
                    "B-LAST": _metrics(
                        targets,
                        b_last_predictions[split_index],
                        dataset.group_ids,
                        indices,
                        target_ranges,
                    ),
                    "O-EQ": _metrics(
                        targets,
                        o_eq_predictions[split_index],
                        dataset.group_ids,
                        indices,
                        target_ranges,
                    ),
                    "O-KIN": _metrics(
                        targets,
                        o_kin_predictions[split],
                        dataset.group_ids,
                        indices,
                        target_ranges,
                    ),
                    "O-KIN-OBS": _metrics(
                        targets[observed_ok],
                        o_kin_obs_predictions[split][observed_ok],
                        dataset.group_ids,
                        indices[observed_ok],
                        target_ranges,
                    )
                    if bool(np.any(observed_ok))
                    else None,
                }
            family_result["horizons"][horizon] = split_metrics
            b_last_val_store[horizon] = (
                indices_by_split["val"],
                b_last_predictions[DEVELOPMENT_SPLITS.index("val")],
            )
            o_kin_obs_val_store[horizon] = (
                indices_by_split["val"],
                o_kin_obs_predictions["val"],
            )
            val_metrics = split_metrics["val"]
            val_last = val_metrics["B-LAST"]["macro_RNMAE"]
            val_kin = val_metrics["O-KIN"]["macro_RNMAE"]
            val_kin_obs = (
                val_metrics["O-KIN-OBS"]["macro_RNMAE"]
                if val_metrics["O-KIN-OBS"] is not None
                else None
            )
            family_result["difficulty"][horizon] = {
                "B-LAST_val_macro_RNMAE": val_last,
                "O-EQ_val_macro_RNMAE": val_metrics["O-EQ"]["macro_RNMAE"],
                "O-KIN_val_macro_RNMAE": val_kin,
                "O-KIN-OBS_val_macro_RNMAE": val_kin_obs,
                "O-KIN-OBS_val_row_count": (
                    int(val_metrics["O-KIN-OBS"]["row_count"])
                    if val_metrics["O-KIN-OBS"] is not None
                    else 0
                ),
                "O-KIN-OBS_val_inversion_failure_fraction": failure_fraction_by_split["val"],
                "relative_degradation": None,
                "oracle_headroom_vs_last": None,
                "oeq_upper_bound_of_blast_holds": bool(
                    val_metrics["O-EQ"]["macro_RNMAE"] is not None
                    and val_last is not None
                    and val_metrics["O-EQ"]["macro_RNMAE"] <= val_last
                ),
                "early_row_count": None,
                "late_row_count": None,
                "paired_row_count": None,
                "paired_population": None,
                "secondary_late_reference": None,
            }
        for horizon in EARLY_HORIZONS:
            item = family_result["difficulty"][horizon]
            for field, reference in (
                ("paired_population", late_reference_horizon),
                ("secondary_late_reference", secondary_late_reference_horizon),
            ):
                item[field] = _paired_late_reference_evidence(
                    dataset,
                    target_ranges,
                    early_indices=b_last_val_store[horizon][0],
                    early_predictions=b_last_val_store[horizon][1],
                    late_indices=b_last_val_store[reference][0],
                    late_predictions=b_last_val_store[reference][1],
                    late_reference_horizon=reference,
                )
            item["early_row_count"] = item["paired_population"]["early_row_count"]
            item["late_row_count"] = item["paired_population"]["late_row_count"]
            item["paired_row_count"] = item["paired_population"]["paired_row_count"]
            item["relative_degradation"] = item["paired_population"][
                "relative_degradation"
            ]
            # headroom 与 B-LAST 在 O-KIN-OBS 反演成功的同一批 val 行上配对计算。
            val_indices, observed_predictions = o_kin_obs_val_store[horizon]
            _, b_last_val_predictions = b_last_val_store[horizon]
            observed_ok = np.isfinite(observed_predictions).all(axis=1)
            paired_indices = val_indices[observed_ok]
            if paired_indices.size:
                observed_metrics = _metrics(
                    dataset.target[paired_indices],
                    observed_predictions[observed_ok],
                    dataset.group_ids,
                    paired_indices,
                    target_ranges,
                )
                last_metrics = _metrics(
                    dataset.target[paired_indices],
                    b_last_val_predictions[observed_ok],
                    dataset.group_ids,
                    paired_indices,
                    target_ranges,
                )
                if last_metrics["macro_RNMAE"] is not None and (
                    last_metrics["macro_RNMAE"] > 0.0
                ):
                    item["oracle_headroom_vs_last"] = float(
                        1.0
                        - observed_metrics["macro_RNMAE"]
                        / last_metrics["macro_RNMAE"]
                    )
        family_result["baseline_fit_status"] = "PASS" if baseline_fit_ok else "FAIL"
        family_result["oracle_fit_status"] = "PASS" if oracle_fit_ok else "FAIL"
        family_results[family] = family_result
    return {
        "baseline_registry": ["B-LAST", "O-EQ", "O-KIN", "O-KIN-OBS"],
        "horizon_order": list(AUDIT_HORIZONS),
        "late_reference_horizon": late_reference_horizon,
        "secondary_late_reference_horizon": secondary_late_reference_horizon,
        "pairing": str(difficulty_gate["pairing"]),
        "families": family_results,
    }


def _paired_late_reference_evidence(
    dataset: DynamicDataset,
    target_ranges: np.ndarray,
    *,
    early_indices: np.ndarray,
    early_predictions: np.ndarray,
    late_indices: np.ndarray,
    late_predictions: np.ndarray,
    late_reference_horizon: str,
) -> dict[str, Any]:
    """在早期与晚期参照同时有效的同一批 val 行上计算 B-LAST 配对比值（F1）。

    分子分母必须来自同一行集合：晚期参照 horizon 有效、早期 horizon 无效
    （或反之）的行不得进入任何一侧。
    """

    paired = np.intersect1d(np.asarray(early_indices), np.asarray(late_indices))
    early_positions = np.searchsorted(np.asarray(early_indices), paired)
    late_positions = np.searchsorted(np.asarray(late_indices), paired)
    evidence: dict[str, Any] = {
        "late_reference_horizon": late_reference_horizon,
        "early_row_count": int(np.asarray(early_indices).size),
        "late_row_count": int(np.asarray(late_indices).size),
        "paired_row_count": int(paired.size),
        "B-LAST_val_macro_RNMAE_early_paired": None,
        "B-LAST_val_macro_RNMAE_late_paired": None,
        "relative_degradation": None,
    }
    if paired.size:
        early_metrics = _metrics(
            dataset.target[paired],
            early_predictions[early_positions],
            dataset.group_ids,
            paired,
            target_ranges,
        )
        late_metrics = _metrics(
            dataset.target[paired],
            late_predictions[late_positions],
            dataset.group_ids,
            paired,
            target_ranges,
        )
        evidence["B-LAST_val_macro_RNMAE_early_paired"] = early_metrics["macro_RNMAE"]
        evidence["B-LAST_val_macro_RNMAE_late_paired"] = late_metrics["macro_RNMAE"]
        if late_metrics["macro_RNMAE"] is not None and late_metrics["macro_RNMAE"] > 0.0:
            evidence["relative_degradation"] = float(
                early_metrics["macro_RNMAE"] / late_metrics["macro_RNMAE"] - 1.0
            )
    return evidence


def _observed_admission_budgets(
    dataset: DynamicDataset,
    indices: np.ndarray,
    *,
    data_config: Mapping[str, Any],
    calibrations: Mapping[str, Mapping[str, Any]],
    noise_base: np.ndarray,
) -> dict[int, tuple[float, float]]:
    """O-KIN-OBS 的观测域准入预算（NDIR 比值单位 / 超声秒单位）。

    预算是该行注册 noise 与 calibration profile 能产生的最大观测扰动包络：
    ``|gain-1|·|信号上界| + |offset| + 漂移上界 + 5×白噪 σ``。NDIR 一侧的
    白噪项按发射/探测器一阶去卷积的噪声增益放大（该增益是注册 profile
    的确定函数）。预算只用于判定观测是否仍落在物理可达域的扰动邻域内并
    按行记录失败；反演算子本身与 O-KIN 完全一致，预算内的越界样本投影
    回物理域端点，与 clean 口径下 float32 预算的处理方式相同。
    """

    noise_by_id = {
        str(profile["noise_profile_id"]): profile
        for profile in data_config["noise_profiles"]
    }
    ndir_tau_s = float(
        data_config["hardware_profiles"]["ndir"]["profiles"][0]["tau_emitter_detector_s"]
    )
    dt_s = float(dataset.manifest["dt_s"])
    if ndir_tau_s > 0.0:
        decay = math.exp(-dt_s / ndir_tau_s)
        deconvolution_gain = math.sqrt(1.0 + decay * decay) / (1.0 - decay)
    else:
        deconvolution_gain = 1.0
    duration_min = float(dataset.manifest["duration_s"]) / 60.0
    budgets: dict[int, tuple[float, float]] = {}
    for row in np.asarray(indices, dtype=np.int64):
        record = dataset.records[int(row)]
        noise = noise_by_id[str(record["noise_profile_id"])]
        calibration = calibrations[str(record["calibration_profile_id"])]
        gains = calibration["sensor_gains"]
        offsets = calibration["sensor_offsets"]
        scale = float(noise["white_noise_scale"])
        drift_pct = float(
            noise["drift_strength_range_pct_dynamic_range_per_min"][1]
        )
        ndir_clean = np.asarray(dataset.clean_device_signals[int(row), 2], dtype=np.float64)
        tof_clean = np.asarray(dataset.clean_device_signals[int(row), 0], dtype=np.float64)
        ndir_band = (
            abs(float(gains["ndir_co2_voltage"]) - 1.0)
            + abs(float(offsets["ndir_co2_voltage"])) / NDIR_BASELINE_V
            + float(np.ptp(ndir_clean))
            / NDIR_BASELINE_V
            * drift_pct
            / 100.0
            * duration_min
            + OBSERVED_ADMISSION_SIGMA_FACTOR
            * deconvolution_gain
            * float(noise_base[2])
            * scale
            / NDIR_BASELINE_V
        )
        tof_band = (
            abs(float(gains["ultrasonic_tof"]) - 1.0) * float(np.max(np.abs(tof_clean)))
            + abs(float(offsets["ultrasonic_tof"]))
            + float(np.ptp(tof_clean)) * drift_pct / 100.0 * duration_min
            + OBSERVED_ADMISSION_SIGMA_FACTOR * float(noise_base[0]) * scale
        )
        budgets[int(row)] = (float(ndir_band), float(tof_band))
    return budgets


def _endpoint_features(
    signals: np.ndarray,
    indices: np.ndarray,
    endpoints: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(indices, dtype=np.int64)
    if rows.size == 0:
        return np.empty((0, signals.shape[1]), dtype=np.float64)
    return np.asarray(
        [signals[row, :, int(endpoints[position]), 0] for position, row in enumerate(rows)],
        dtype=np.float64,
    )


def _equilibrium_features(
    equilibrium: np.ndarray,
    indices: np.ndarray,
    endpoints: np.ndarray,
) -> np.ndarray:
    rows = np.asarray(indices, dtype=np.int64)
    if rows.size == 0:
        return np.empty((0, equilibrium.shape[1]), dtype=np.float64)
    return np.asarray(
        [equilibrium[row, :, int(endpoints[position])] for position, row in enumerate(rows)],
        dtype=np.float64,
    )


def _fit_small_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    eval_x: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if np.asarray(train_x).ndim != 2 or np.asarray(train_x).shape[0] == 0:
        raise ValueError("B-LAST requires a non-empty two-dimensional training feature matrix")
    mean = np.asarray(train_x, dtype=np.float64).mean(axis=0)
    scale = np.asarray(train_x, dtype=np.float64).std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    scaled_train = (train_x - mean) / scale
    model = MLPRegressor(
        hidden_layer_sizes=(24,),
        activation="tanh",
        solver="lbfgs",
        alpha=1.0e-6,
        max_iter=2000,
        tol=1.0e-3,
        random_state=17,
    )
    model.fit(scaled_train, np.asarray(train_y, dtype=np.float64) / 100.0)
    predictions: list[np.ndarray] = []
    for values in eval_x:
        values_array = np.asarray(values, dtype=np.float64)
        predictions.append(
            np.empty((0, train_y.shape[1]), dtype=np.float64)
            if values_array.shape[0] == 0
            else model.predict((values_array - mean) / scale) * 100.0
        )
    n_iter = int(getattr(model, "n_iter_", 0))
    max_iter = int(model.max_iter)
    loss = float(getattr(model, "loss_", math.nan))
    finite_predictions = all(np.isfinite(prediction).all() for prediction in predictions)
    diagnostics = {
        "status": "PASS"
        if n_iter < max_iter and math.isfinite(loss) and finite_predictions
        else "FAIL",
        "solver": str(model.solver),
        "n_iter": n_iter,
        "max_iter": max_iter,
        "loss": loss,
        "finite_predictions": finite_predictions,
    }
    return predictions, diagnostics


def _kinetic_oracle_predictions(
    dataset: DynamicDataset,
    indices: np.ndarray,
    endpoints: np.ndarray,
    *,
    data_config: Mapping[str, Any],
    kinetic_cache: dict[int, dict[str, np.ndarray]],
    heos_interpolation_cache: dict[tuple[float, float], np.ndarray],
    input_mode: str = "clean",
    admission_budgets: Mapping[int, tuple[float, float]] | None = None,
    inversion_failures: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    """用设备序列和注册动力学参数反演目标组成。

    ``input_mode="clean"``（O-KIN）：输入 clean 设备信号，任何越界或不可
    辨识都显式 ``raise``，作为前向模型可逆性上界审计。
    ``input_mode="observed"``（O-KIN-OBS）：反演算子与特权动力学参数完全
    相同，输入换成最终观测信号（含标定、漂移、AR(1)、白噪与量化）；观测
    超出该行注册扰动包络（``admission_budgets``）的行按行记入
    ``inversion_failures`` 并以 NaN 占位，不静默丢弃、不放宽反演容差。
    """

    if input_mode not in ("clean", "observed"):
        raise ValueError(f"unsupported kinetic oracle input_mode {input_mode!r}")
    observed_mode = input_mode == "observed"
    if observed_mode and (admission_budgets is None or inversion_failures is None):
        raise ValueError(
            "observed-mode kinetic inversion requires admission budgets and a failure sink"
        )

    def record_failure(row: int, endpoint: int, stage: str, reason: str) -> None:
        inversion_failures.append(
            {
                "row": int(row),
                "endpoint": int(endpoint),
                "stage": stage,
                "reason": reason,
            }
        )

    ultrasonic_profile = next(
        item
        for item in data_config["hardware_profiles"]["ultrasonic"]["candidates"]
        if str(item["ultrasonic_profile_id"])
        == str(data_config["hardware_profiles"]["ultrasonic"]["selected_profile_id"])
    )
    ndir_profile = NDIRDeviceProfile.from_mapping(data_config["hardware_profiles"]["ndir"]["profiles"][0])
    model_id = str(data_config["physics_reference"]["eos"]["sound_speed_model_id"])
    result = np.empty((len(indices), 3), dtype=np.float64)
    dt_s = float(dataset.manifest["dt_s"])
    for position, row in enumerate(np.asarray(indices, dtype=np.int64)):
        parameters = dataset.privileged_parameters[row]
        endpoint = int(endpoints[position])
        if endpoint < 0 or endpoint >= dataset.timesteps:
            raise ValueError(f"O-KIN endpoint is invalid at row {row}: {endpoint}")
        cache_entry = kinetic_cache.get(int(row))
        if cache_entry is None:
            mix_response = simulate_first_order_series(
                dataset.inlet_coefficient[row],
                dt_s=dt_s,
                tau_s=float(parameters[0]),
                initial_state=0.0,
            )
            local_response_series = np.column_stack(
                [
                    simulate_first_order_series(
                        mix_response,
                        dt_s=dt_s,
                        tau_s=float(parameters[index]),
                        initial_state=0.0,
                    )
                    for index in (1, 2, 3)
                ]
            )
            cache_entry = {
                "local_responses": local_response_series,
            }
            kinetic_cache[int(row)] = cache_entry
        co2_cache_key = "local_co2_observed" if observed_mode else "local_co2_clean"
        if co2_cache_key not in cache_entry:
            if observed_mode:
                ndir_series = np.asarray(
                    dataset.signals[int(row), 2, :, 0], dtype=np.float64
                )
                ndir_tolerance = float(admission_budgets[int(row)][0])
            else:
                ndir_series = np.asarray(
                    dataset.clean_device_signals[int(row), 2], dtype=np.float64
                )
                ndir_tolerance = None
            try:
                local_co2_series = estimate_ndir_equilibrium_co2_series(
                    ndir_series,
                    temperature_k=float(parameters[4]),
                    pressure_pa=float(parameters[5]),
                    dt_s=dt_s,
                    profile=ndir_profile,
                    absorbance_scale=float(parameters[8]),
                    domain_tolerance=ndir_tolerance,
                )
            except SensorDeviceError as error:
                if observed_mode:
                    record_failure(int(row), endpoint, "ndir_ratio_domain", str(error))
                    result[position] = np.full(3, np.nan, dtype=np.float64)
                    continue
                raise
            cache_entry[co2_cache_key] = local_co2_series
        local_responses = np.asarray(cache_entry["local_responses"][endpoint], dtype=np.float64)
        if np.any(local_responses <= 1.0e-12):
            # 响应来自特权动力学参数，与输入模式无关：非正即数据缺陷。
            raise ValueError(f"O-KIN response is not identifiable at row {row}, endpoint {endpoint}")
        target_co2 = float(cache_entry[co2_cache_key][endpoint] / local_responses[2])
        if not math.isfinite(target_co2) or not 0.0 <= target_co2 <= TARGET_TOTAL:
            if observed_mode:
                record_failure(
                    int(row),
                    endpoint,
                    "co2_inversion_range",
                    f"CO2 inversion outside [0,100] at row {row}: {target_co2!r}",
                )
                result[position] = np.full(3, np.nan, dtype=np.float64)
                continue
            raise ValueError(f"O-KIN CO2 inversion is outside [0,100] at row {row}")
        acoustic_response = float(local_responses[0])
        if observed_mode:
            observed_tof = float(dataset.signals[int(row), 0, endpoint, 0])
        else:
            observed_tof = float(dataset.clean_device_signals[int(row), 0, endpoint])
        maximum_he = TARGET_TOTAL - target_co2

        def endpoint_tof(helium_pct: float) -> float:
            target = np.asarray(
                [TARGET_TOTAL - helium_pct - target_co2, helium_pct, target_co2],
                dtype=np.float64,
            )
            local = PURGE_COMPOSITION + acoustic_response * (target - PURGE_COMPOSITION)
            return _registered_heos_interpolated_tof(
                local,
                temperature_k=float(parameters[4]),
                pressure_pa=float(parameters[5]),
                path_length_m=float(ultrasonic_profile["path_length_m"]) * float(parameters[6]),
                sound_speed_model_id=model_id,
                cache=heos_interpolation_cache,
            )

        tof_at_zero_he = endpoint_tof(0.0)
        tof_at_max_he = endpoint_tof(maximum_he)
        if maximum_he <= 1.0e-12:
            zero_he_tolerance_s = HEOS_INTERPOLATION_TOF_TOLERANCE_S
            if observed_mode:
                zero_he_tolerance_s += float(admission_budgets[int(row)][1])
            if abs(observed_tof - tof_at_zero_he) > zero_he_tolerance_s:
                if observed_mode:
                    record_failure(
                        int(row),
                        endpoint,
                        "tof_zero_he_boundary",
                        (
                            f"observed ToF is incompatible with the zero-He boundary at row {row}: "
                            f"observed_tof={observed_tof:.9g}, expected={tof_at_zero_he:.9g}, "
                            f"tolerance_s={zero_he_tolerance_s:.3g}"
                        ),
                    )
                    result[position] = np.full(3, np.nan, dtype=np.float64)
                    continue
                raise ValueError(
                    f"O-KIN ultrasonic inversion is not compatible with the zero-He boundary at row {row}"
                )
            helium = 0.0
        else:
            tof_step_s = (
                0.25 if "parabolic" in str(ultrasonic_profile["tof_estimator"]) else 1.0
            ) / float(ultrasonic_profile["adc_rate_hz"])
            boundary_tolerance_s = 0.5 * tof_step_s + HEOS_INTERPOLATION_TOF_TOLERANCE_S
            if observed_mode:
                boundary_tolerance_s += float(admission_budgets[int(row)][1])
            tof_lower = min(tof_at_zero_he, tof_at_max_he)
            tof_upper = max(tof_at_zero_he, tof_at_max_he)
            if observed_tof < tof_lower - boundary_tolerance_s or observed_tof > tof_upper + boundary_tolerance_s:
                if observed_mode:
                    record_failure(
                        int(row),
                        endpoint,
                        "tof_quantization_boundary",
                        (
                            f"observed ToF exceeds the registered quantization boundary at row {row}: "
                            f"observed_tof={observed_tof:.9g}, range=[{tof_lower:.9g},{tof_upper:.9g}], "
                            f"tolerance_s={boundary_tolerance_s:.3g}"
                        ),
                    )
                    result[position] = np.full(3, np.nan, dtype=np.float64)
                    continue
                raise ValueError(
                    f"O-KIN ultrasonic inversion exceeds the registered quantization boundary at row {row}: "
                    f"observed_tof={observed_tof:.9g}, range=[{tof_lower:.9g},{tof_upper:.9g}], "
                    f"tolerance_s={boundary_tolerance_s:.3g}"
                )
            observed_tof = min(tof_upper, max(tof_lower, observed_tof))
            direction = 1.0 if tof_at_max_he >= tof_at_zero_he else -1.0
            lower_he = 0.0
            upper_he = maximum_he
            for _ in range(48):
                middle_he = 0.5 * (lower_he + upper_he)
                middle_tof = endpoint_tof(middle_he)
                if direction * middle_tof < direction * observed_tof:
                    lower_he = middle_he
                else:
                    upper_he = middle_he
            helium = 0.5 * (lower_he + upper_he)
        result[position] = np.asarray(
            [TARGET_TOTAL - helium - target_co2, helium, target_co2],
            dtype=np.float64,
        )
        if not np.isfinite(result[position]).all() or not np.allclose(
            result[position].sum(), TARGET_TOTAL, rtol=0.0, atol=1.0e-9
        ):
            if observed_mode:
                record_failure(
                    int(row),
                    endpoint,
                    "composition_invalid",
                    f"O-KIN-OBS produced an invalid composition at row {row}",
                )
                result[position] = np.full(3, np.nan, dtype=np.float64)
                continue
            raise ValueError(f"O-KIN produced an invalid composition at row {row}")
    return result


def _metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    group_ids: Sequence[str],
    indices: np.ndarray,
    target_ranges: np.ndarray,
) -> dict[str, Any]:
    selected_groups = np.asarray(group_ids, dtype=object)[indices]
    metrics = evaluate_predictions(
        targets,
        predictions,
        selected_groups,
        np.arange(len(indices), dtype=np.int64),
        target_ranges=target_ranges,
    )
    metrics["constraints"] = evaluate_output_constraints(
        predictions,
        targets=targets,
        total=TARGET_TOTAL,
    )
    metrics["row_count"] = int(len(indices))
    return metrics
