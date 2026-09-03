"""A2-DYN-2 pilot：比较时间轴、设备候选与动态非退化资格。

pilot 只持有低频序列和设备摘要，不写正式 observation 包，也不持久化
超声高频波形。数据生成语义由 ``gf.sim.a2_dynamic_dataset`` 唯一实现。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from gf.dl.temporal_baselines import (
    PILOT_PROBE_IDS,
    fit_pilot_linear_probes,
    pilot_probe_feature_vector,
)
from gf.pipeline.a2_dynamic_protocol import (
    validate_a2_dynamic_configs,
    validate_a2_dynamic_pilot_config,
)
from gf.sim.a2_dynamic_dataset import (
    apply_dynamic_observation_profile,
    calibration_physical_scales,
    choose_profile_id,
    resample_continuous_series,
    resolve_protocol_instance,
    sample_registered_range,
    unique_quantization_level_counts,
)
from gf.sim.a2_dynamic_physics import (
    DynamicTransportLayers,
    build_inlet_composition,
    evaluate_shared_physics,
    protocol_inlet_coefficient,
    simulate_dynamic_layers,
)
from gf.sim.a2dyn_sound_speed import coolprop_runtime_identity
from gf.sim.a2_sensor_devices import (
    acquire_ultrasonic_tof,
    estimate_ultrasonic_tof_series,
    simulate_ndir,
    simulate_tcd,
    ultrasonic_signal_amplitude,
)


_FAMILIES = (
    "D-IID",
    "D-KINETICS",
    "D-PROTOCOL",
    "D-NOISE-DRIFT",
    "D-ENV-CAL",
    "D-JOINT",
)
_SPLITS = ("train", "val", "stress_val")
_SENSOR_IDS = ("ultrasonic_tof", "thermal_conductivity_voltage", "ndir_co2_voltage")
_HORIZON_IDS = ("P005", "P015", "P030", "P060", "P120", "P150")
_MIN_UNIQUE_QUANTIZED_LEVELS = 10


def run_a2_dynamic_pilot(project_root: str | Path = ".") -> dict[str, Any]:
    """运行冻结的 A2-DYN-2 pilot 并写出可复算 manifest。"""

    root = _resolve_project_root(project_root)
    data_path = root / "configs" / "data" / "ar_he_co2_a2_dynamic_v1.json"
    eval_path = root / "configs" / "eval" / "a2_dynamic_eval.json"
    experiment_path = root / "configs" / "experiment" / "a2_dynamic_protocol.json"
    data_config = _read_json(data_path)
    eval_config = _read_json(eval_path)
    experiment_config = _read_json(experiment_path)
    a2h_relative = str(data_config["source_registry"]["a2h"]["config_path"])
    a2h_path = (root / a2h_relative).resolve()
    a2h_config = _read_json(a2h_path)
    protocol_summary = validate_a2_dynamic_configs(root, verify_reference_assets=True)
    pilot = _validate_pilot_spec(experiment_config.get("pilot"))
    assignments = _build_assignments(pilot)
    compositions = _build_pilot_compositions(int(pilot["mixture_count"]), int(pilot["seed"]))

    started = time.perf_counter()
    base_cache = _build_base_cache(data_config, a2h_config, pilot, assignments, compositions)
    scenarios: dict[str, dict[str, Any]] = {}
    for sample_rate_hz in pilot["sample_rates_hz"]:
        for duration_s in pilot["durations_s"]:
            key = _scenario_key(sample_rate_hz, duration_s)
            scenarios[key] = _run_scenario(
                data_config,
                eval_config,
                pilot,
                assignments,
                compositions,
                sample_rate_hz=float(sample_rate_hz),
                duration_s=float(duration_s),
                base_cache=base_cache,
            )
    ultrasonic = _evaluate_ultrasonic_candidates(
        data_config,
        pilot,
        compositions[: int(pilot["ultrasonic_quality_sample_count"])],
    )
    selection = _select_pilot_axes(pilot, eval_config, scenarios, ultrasonic)
    peak_bytes = _peak_process_working_set_bytes()
    elapsed_s = time.perf_counter() - started

    audit_key = _selected_or_reference_key(pilot, selection, scenarios)
    audit_scenario = scenarios[audit_key]
    probe_checks = _probe_checks(audit_scenario["pilot_probe_metrics"], pilot)
    dynamic_checks = audit_scenario["dynamic_audit"]["checks"]
    device_checks = audit_scenario["device_audit"]["checks"]
    resource = _resource_summary(pilot, peak_bytes, audit_scenario, base_cache)
    checks = {
        "pilot_count": len(assignments) == int(pilot["mixture_count"]),
        "family_counts": all(
            sum(item["family"] == family for item in assignments) == int(pilot["groups_per_family"])
            for family in _FAMILIES
        ),
        "split_counts": _split_counts(assignments) == _pilot_split_totals(pilot),
        "sample_rate_selected": selection["sample_rate_hz"] is not None,
        "duration_selected": selection["duration_s"] is not None,
        "ultrasonic_selected": selection["ultrasonic_profile_id"] is not None,
        "tcd_ndir_device_audit": all(device_checks.values()),
        "dynamic_non_degenerate": all(dynamic_checks.values()),
        "stress_tier_identifiable": probe_checks["stress_identifiable"],
        "resource_within_limit": resource["within_limit"],
        "no_high_frequency_waveform_persisted": resource["waveform_persisted_bytes"] == 0,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    status = "PILOT_QUALIFIED" if not failed_checks else "PILOT_INVALID"
    dependency_hashes = _dependency_hashes(root, data_config, a2h_relative)
    runtime_identity = coolprop_runtime_identity()
    result: dict[str, Any] = {
        "status": status,
        "stage": "A2-DYN-2R4",
        "protocol_status": protocol_summary["protocol_status"],
        "protocol_revision": data_config["protocol_revision"],
        "pilot_spec": pilot,
        "mixture_count": len(assignments),
        "composition_hash": _sha256_json(compositions.tolist()),
        "family_counts": {
            family: sum(item["family"] == family for item in assignments)
            for family in _FAMILIES
        },
        "split_counts": _split_counts(assignments),
        "scenarios": scenarios,
        "ultrasonic_candidates": ultrasonic,
        "selection": selection,
        "audit_scenario_key": audit_key,
        "device_audit": audit_scenario["device_audit"],
        "dynamic_audit": audit_scenario["dynamic_audit"],
        "pilot_probe_checks": probe_checks,
        "resource": resource,
        "runtime_identity": runtime_identity,
        "checks": checks,
        "failed_checks": failed_checks,
        "elapsed_s": elapsed_s,
        "dependency_hashes": dependency_hashes,
    }
    summary_dir = root / "outputs" / "summary" / "a2_dynamic_v1"
    run_dir = root / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-2r4-pilot"
    summary_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (summary_dir / "pilot_audit_r4.json").write_text(payload, encoding="utf-8")
    (run_dir / "manifest.json").write_text(payload, encoding="utf-8")
    resolved_config = {
        "data": data_config,
        "evaluation": eval_config,
        "experiment": experiment_config,
        "a2h_calibration_profiles": a2h_config["calibration_profiles"],
        "runtime_identity": runtime_identity,
        "dependency_hashes": dependency_hashes,
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _resolve_project_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    if (root / "configs" / "data").is_dir():
        return root
    nested = root / "general_fusion"
    if (nested / "configs" / "data").is_dir():
        return nested
    raise ValueError(f"cannot locate general_fusion project root from {root}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON config {path} must contain an object")
    return value


def _validate_pilot_spec(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("experiment.pilot must be a mapping")
    result = dict(raw)
    validate_a2_dynamic_pilot_config(result)
    return result


def _build_assignments(pilot: Mapping[str, Any]) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    split_counts_per_family = {
        split: int(pilot["split_counts_per_family"][split])
        for split in _SPLITS
    }
    if sum(split_counts_per_family.values()) != int(pilot["groups_per_family"]):
        raise ValueError("pilot split counts per family must sum to groups_per_family")
    index = 0
    for family in _FAMILIES:
        for family_index in range(int(pilot["groups_per_family"])):
            cumulative = 0
            split = None
            for candidate in _SPLITS:
                cumulative += split_counts_per_family[candidate]
                if family_index < cumulative:
                    split = candidate
                    break
            if split is None:
                raise ValueError(f"pilot family index {family_index} has no registered split")
            assignments.append(
                {
                    "mixture_id": f"a2dyn-mix-pilot-{index:04d}",
                    "family": family,
                    "family_index": family_index,
                    "split": split,
                    "target_index": index,
                }
            )
            index += 1
    return assignments


def _build_pilot_compositions(count: int, seed: int) -> np.ndarray:
    values: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()
    for index in range(count):
        u = (0.6180339887498949 * (index + 1) + seed * 1.0e-7) % 1.0
        v = (0.4142135623730950 * (index + 1) + seed * 3.0e-7) % 1.0
        if u + v > 1.0:
            u, v = 1.0 - u, 1.0 - v
        if index % 4 == 1:
            low = 0.5 + 4.0 * u
            raw = np.asarray([low, (100.0 - low) * (0.1 + 0.8 * v), 0.0])
            raw[2] = 100.0 - raw[0] - raw[1]
        elif index % 4 == 2:
            low = 0.5 + 4.0 * v
            first = (100.0 - low) * (0.1 + 0.8 * u)
            raw = np.asarray([first, 100.0 - first - low, low])
        else:
            raw = 5.0 + 85.0 * np.asarray([u, v, 1.0 - u - v])
        composition = np.round(raw, 2)
        composition[2] = round(100.0 - float(composition[0]) - float(composition[1]), 2)
        key = tuple(float(item) for item in composition)
        if np.any(composition < 0.0) or not math.isclose(float(composition.sum()), 100.0, abs_tol=1.0e-9):
            raise ValueError(f"pilot composition {index} is outside the closed simplex")
        if key in seen:
            raise ValueError(f"pilot composition is not unique at index {index}: {key}")
        seen.add(key)
        values.append(composition)
    return np.asarray(values, dtype=np.float64)


def _build_base_cache(
    data_config: Mapping[str, Any],
    a2h_config: Mapping[str, Any],
    pilot: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    compositions: np.ndarray,
) -> list[dict[str, Any]]:
    # The source grid must cover every candidate. Derive it from the frozen
    # candidate lists so a new candidate cannot silently be truncated.
    reference_sample_rate_hz = max(float(rate) for rate in pilot["sample_rates_hz"])
    reference_duration_s = max(float(duration) for duration in pilot["durations_s"])
    dt_s = 1.0 / reference_sample_rate_hz
    timesteps = int(round(reference_duration_s * reference_sample_rate_hz))
    time_s = np.arange(timesteps, dtype=np.float64) * dt_s
    hardware = data_config["hardware_profiles"]
    ultrasonic_by_id = {
        item["ultrasonic_profile_id"]: item for item in hardware["ultrasonic"]["candidates"]
    }
    ultrasonic_profile = ultrasonic_by_id[hardware["ultrasonic"]["selected_profile_id"]]
    thermal_profile = hardware["thermal"]["profiles"][0]
    ndir_profile = hardware["ndir"]["profiles"][0]
    transport_by_id = {item["transport_profile_id"]: item for item in data_config["transport"]["profiles"]}
    environment_by_id = {item["environment_id"]: item for item in data_config["environment_profiles"]}
    protocol_by_id = {item["protocol_profile_id"]: item for item in data_config["protocol_profiles"]}
    noise_by_id = {item["noise_profile_id"]: item for item in data_config["noise_profiles"]}
    calibration_by_id = {
        item["calibration_profile_id"]: item for item in a2h_config["calibration_profiles"]
    }
    transport_contract = data_config["transport"]
    model_id = data_config["physics_reference"]["eos"]["sound_speed_model_id"]
    cache: list[dict[str, Any]] = []
    for assignment in assignments:
        index = int(assignment["target_index"])
        family_index = int(assignment["family_index"])
        family = data_config["families"][assignment["family"]]
        split = assignment["split"]
        protocol_id = choose_profile_id(family["protocol_by_split"][split], family_index)
        transport_id = choose_profile_id(family["transport_by_split"][split], family_index)
        environment_id = choose_profile_id(family["environment_by_split"][split], family_index)
        calibration_id = choose_profile_id(family["calibration_by_split"][split], family_index)
        noise_id = choose_profile_id(family["noise_by_split"][split], family_index)
        protocol = protocol_by_id[protocol_id]
        transport = transport_by_id[transport_id]
        environment = environment_by_id[environment_id]
        calibration = calibration_by_id[calibration_id]
        protocol_instance = resolve_protocol_instance(protocol, transport, index=index)
        coefficient = protocol_inlet_coefficient(time_s, **protocol_instance.parameters)
        inlet = build_inlet_composition(
            time_s,
            purge_composition_pct=data_config["inlet"]["purge_composition_pct"],
            target_composition_pct=compositions[index],
            coefficient=coefficient,
        )
        tau_transport = {
            "ultrasonic_tof": _sample_transport_range(transport["tau_transport_ultrasonic_s"], index, 53, transport_contract),
            "thermal_conductivity_voltage": _sample_transport_range(transport["tau_transport_thermal_s"], index, 59, transport_contract),
            "ndir_co2_voltage": _sample_transport_range(transport["tau_transport_ndir_s"], index, 61, transport_contract),
        }
        layers = simulate_dynamic_layers(
            inlet,
            dt_s=dt_s,
            tau_mix_s=sample_registered_range(
                transport["tau_mix_s"],
                index,
                distribution=str(transport_contract["tau_mix_distribution"]),
                salt=67,
            ),
            tau_transport_s=tau_transport,
        )
        acoustic_scale, tcd_scale, ndir_scale = calibration_physical_scales(calibration)
        local = layers.local_composition_pct
        shared = evaluate_shared_physics(
            local["ultrasonic_tof"],
            temperature_k=environment["temperature_k"],
            pressure_pa=environment["pressure_pa"],
            path_length_m=float(ultrasonic_profile["path_length_m"]) * acoustic_scale,
            sound_speed_model_id=model_id,
        )
        ultrasonic_clean = estimate_ultrasonic_tof_series(shared["tof_s"], ultrasonic_profile)
        tcd = simulate_tcd(
            local["thermal_conductivity_voltage"],
            temperature_k=environment["temperature_k"],
            dt_s=dt_s,
            profile=thermal_profile,
            response_scale=tcd_scale,
        )
        ndir = simulate_ndir(
            local["ndir_co2_voltage"],
            temperature_k=environment["temperature_k"],
            pressure_pa=environment["pressure_pa"],
            dt_s=dt_s,
            profile=ndir_profile,
            absorbance_scale=ndir_scale,
        )
        cache.append(
            {
                "source_time_s": time_s,
                "clean": np.column_stack((ultrasonic_clean, tcd.clean_voltage_v, ndir.clean_voltage_v)),
                "layers": layers,
                "tau_transport": tau_transport,
                "tcd_residual": tcd.energy_balance_residual_w,
                "ndir_saturation_fraction": ndir.saturation_fraction,
                "noise_profile": noise_by_id[noise_id],
                "calibration_profile": calibration,
                "protocol_instance": protocol_instance,
                "family": assignment["family"],
            }
        )
    return cache


def _sample_transport_range(
    values: Sequence[float],
    index: int,
    salt: int,
    transport_contract: Mapping[str, Any],
) -> float:
    if float(values[0]) == 0.0:
        return sample_registered_range(
            values,
            index,
            distribution=str(transport_contract["nonzero_transport_distribution"]),
            salt=salt,
            zero_probability=float(transport_contract["zero_transport_probability"]),
            minimum_nonzero=float(transport_contract["minimum_nonzero_transport_s"]),
        )
    return sample_registered_range(
        values,
        index,
        distribution=str(transport_contract["nonzero_transport_distribution"]),
        salt=salt,
    )


def _run_scenario(
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    pilot: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    compositions: np.ndarray,
    *,
    sample_rate_hz: float,
    duration_s: float,
    base_cache: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dt_s = 1.0 / sample_rate_hz
    timesteps = int(round(duration_s * sample_rate_hz))
    time_s = np.arange(timesteps, dtype=np.float64) * dt_s
    noise_base = np.asarray(pilot["observation_noise_std_by_sensor"], dtype=np.float64)
    quantization = np.asarray(pilot["observation_quantization_by_sensor"], dtype=np.float64)
    signal_bounds = tuple(
        tuple(float(value) for value in data_config["signal_bounds"][sensor_id])
        for sensor_id in _SENSOR_IDS
    )
    feature_store = {
        probe_id: {horizon: {split: [] for split in _SPLITS} for horizon in _HORIZON_IDS}
        for probe_id in pilot["pilot_probe_ids"]
    }
    target_store = {horizon: {split: [] for split in _SPLITS} for horizon in _HORIZON_IDS}
    group_store = {horizon: {split: [] for split in _SPLITS} for horizon in _HORIZON_IDS}
    sequence_audits: list[dict[str, Any]] = []
    max_sequence_bytes = 0
    for assignment in assignments:
        index = int(assignment["target_index"])
        base = base_cache[index]
        source_time = base["source_time_s"]
        clean = resample_continuous_series(source_time, base["clean"], time_s)
        base_layers = base["layers"]
        layers = DynamicTransportLayers(
            chamber_composition_pct=resample_continuous_series(
                source_time, base_layers.chamber_composition_pct, time_s
            ),
            local_composition_pct={
                sensor_id: resample_continuous_series(
                    source_time, base_layers.local_composition_pct[sensor_id], time_s
                )
                for sensor_id in _SENSOR_IDS
            },
        )
        tcd_residual = resample_continuous_series(
            source_time, np.asarray(base["tcd_residual"])[:, None], time_s
        )[:, 0]
        rng = np.random.default_rng(
            int(pilot["seed"])
            + 1009 * index
            + 100_000 * int(round(sample_rate_hz * 10.0))
            + 10 * int(round(duration_s))
        )
        observed, perturbation = apply_dynamic_observation_profile(
            clean,
            noise_profile=base["noise_profile"],
            calibration_profile=base["calibration_profile"],
            noise_base=noise_base,
            quantization=quantization,
            dt_s=dt_s,
            index=index,
            rng=rng,
        )
        max_sequence_bytes = max(
            max_sequence_bytes,
            clean.nbytes
            + observed.nbytes
            + layers.chamber_composition_pct.nbytes
            + sum(value.nbytes for value in layers.local_composition_pct.values()),
        )
        protocol_instance = base["protocol_instance"]
        sequence_audits.append(
            _sequence_audit(
                clean,
                observed,
                time_s,
                noise_base * float(base["noise_profile"]["white_noise_scale"]),
                tcd_residual,
                float(base["ndir_saturation_fraction"]),
                float(protocol_instance.exposure_onset_s),
                float(protocol_instance.exposure_end_s),
                assignment["family"],
                perturbation,
                pilot,
                signal_bounds,
            )
        )
        for horizon, exposure_after_s in zip(_HORIZON_IDS, pilot["horizons_after_onset_s"]):
            cutoff_s = float(protocol_instance.exposure_onset_s) + float(exposure_after_s) - dt_s
            if cutoff_s > time_s[-1] + 1.0e-9 or cutoff_s >= float(protocol_instance.exposure_end_s):
                continue
            end_index = int(np.searchsorted(time_s, cutoff_s, side="right") - 1)
            if end_index < 0:
                continue
            prefix_observed = observed[: end_index + 1]
            prefix_clean = clean[: end_index + 1]
            split = assignment["split"]
            target_store[horizon][split].append(compositions[index].copy())
            group_store[horizon][split].append(str(assignment["mixture_id"]))
            for probe_id in pilot["pilot_probe_ids"]:
                feature_store[probe_id][horizon][split].append(
                    pilot_probe_feature_vector(
                        probe_id,
                        prefix_observed,
                        prefix_clean,
                        layers,
                        base["tau_transport"],
                        ewma_alpha=float(pilot["ewma_alpha"]),
                    )
                )
    target_ranges = [float(eval_config["target_ranges"][name]) for name in data_config["target_names"]]
    probe_metrics = fit_pilot_linear_probes(
        feature_store,
        target_store,
        group_store,
        target_ranges=target_ranges,
    )
    low_co2_delta = _ndir_low_co2_delta(data_config, dt_s)
    return {
        "sample_rate_hz": sample_rate_hz,
        "duration_s": duration_s,
        "timesteps": timesteps,
        "timestamp_alignment_max_error_s": float(
            np.max(np.abs(time_s - np.arange(timesteps, dtype=np.float64) / sample_rate_hz))
        ),
        "dynamic_audit": _aggregate_dynamic_audit(sequence_audits, sample_rate_hz, duration_s, pilot),
        "device_audit": _aggregate_device_audit(sequence_audits, low_co2_delta, pilot),
        "pilot_probe_metrics": probe_metrics,
        "horizon_availability": {
            horizon: sum(len(target_store[horizon][split]) for split in _SPLITS)
            for horizon in _HORIZON_IDS
        },
        "max_sequence_bytes": int(max_sequence_bytes),
    }


def _sequence_audit(
    clean: np.ndarray,
    observed: np.ndarray,
    time_s: np.ndarray,
    noise_std: np.ndarray,
    tcd_residual: np.ndarray,
    ndir_saturation_fraction: float,
    exposure_onset_s: float,
    exposure_end_s: float,
    family: str,
    perturbation: Any,
    pilot: Mapping[str, Any],
    signal_bounds: Sequence[Sequence[float]],
) -> dict[str, Any]:
    p2p = np.ptp(clean, axis=0)
    active_channels = int(np.sum(p2p > 5.0 * noise_std))
    quantized_levels = unique_quantization_level_counts(observed)
    t50_s = _t50_seconds(clean, time_s, exposure_onset_s, exposure_end_s)
    finite_t50 = t50_s[np.isfinite(t50_s)]
    minimum_separation_s = float(pilot["dynamic_gate"]["minimum_t50_separation_samples"]) * (
        time_s[1] - time_s[0]
    )
    pair_separation = bool(
        finite_t50.size >= 2 and np.max(finite_t50) - np.min(finite_t50) >= minimum_separation_s
    )
    transition_end_s = min(exposure_end_s, exposure_onset_s + 60.0)
    transition = clean[(time_s >= exposure_onset_s) & (time_s < transition_end_s)]
    transition_variance = float(np.var(np.diff(transition, axis=0))) if transition.shape[0] > 1 else 0.0
    phase_counts = {
        "baseline": int(np.sum(time_s < exposure_onset_s)),
        "transition": int(np.sum((time_s >= exposure_onset_s) & (time_s < transition_end_s))),
        "steady": int(np.sum((time_s >= transition_end_s) & (time_s < exposure_end_s))),
        "recovery": int(np.sum(time_s >= exposure_end_s)),
    }
    bound_violations = [
        int(np.sum((observed[:, channel] < lower) | (observed[:, channel] > upper)))
        for channel, (lower, upper) in enumerate(signal_bounds)
    ]
    return {
        "family": family,
        "active_channels": active_channels,
        "unique_quantized_levels_by_sensor": [int(value) for value in quantized_levels],
        "quantized_levels_min": int(np.min(quantized_levels)),
        "t50_s": [float(value) if math.isfinite(float(value)) else None for value in t50_s],
        "t50_pair_separation": pair_separation,
        "transition_variance": transition_variance,
        "phase_counts": phase_counts,
        "exposure_onset_s": exposure_onset_s,
        "exposure_end_s": exposure_end_s,
        "tcd_energy_residual_max_w": float(np.max(np.abs(tcd_residual))),
        "ndir_saturation_fraction": ndir_saturation_fraction,
        "bound_violations_by_sensor": bound_violations,
        "bound_violations": int(sum(bound_violations)),
        "observed_min_by_sensor": [float(value) for value in np.min(observed, axis=0)],
        "observed_max_by_sensor": [float(value) for value in np.max(observed, axis=0)],
        "observation_profile": {
            "ar1_rho": float(perturbation.ar1_rho),
            "drift_strength_pct_dynamic_range_per_min": float(
                perturbation.drift_strength_pct_dynamic_range_per_min
            ),
        },
    }


def _t50_seconds(
    values: np.ndarray,
    time_s: np.ndarray,
    exposure_onset_s: float,
    exposure_end_s: float,
) -> np.ndarray:
    baseline_mask = (time_s < exposure_onset_s) & (time_s >= max(0.0, exposure_onset_s - 20.0))
    response_end_s = min(exposure_end_s, exposure_onset_s + 120.0, time_s[-1] + (time_s[1] - time_s[0]))
    response_mask = (time_s >= exposure_onset_s) & (time_s < response_end_s)
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    if not np.any(baseline_mask) or not np.any(response_mask):
        return result
    baseline_values = values[baseline_mask]
    response_values = values[response_mask]
    response_times = time_s[response_mask]
    tail_count = max(1, int(round(min(10.0, response_end_s - exposure_onset_s) / (time_s[1] - time_s[0]))))
    for channel in range(values.shape[1]):
        baseline = float(np.median(baseline_values[:, channel]))
        final = float(np.median(response_values[-tail_count:, channel]))
        delta = final - baseline
        if abs(delta) <= 1.0e-12:
            continue
        threshold = baseline + 0.5 * delta
        series = response_values[:, channel]
        indices = np.flatnonzero(series >= threshold) if delta > 0.0 else np.flatnonzero(series <= threshold)
        if indices.size:
            result[channel] = float(response_times[int(indices[0])] - exposure_onset_s)
    return result


def _aggregate_dynamic_audit(
    sequence_audits: Sequence[Mapping[str, Any]],
    sample_rate_hz: float,
    duration_s: float,
    pilot: Mapping[str, Any],
) -> dict[str, Any]:
    gate = pilot["dynamic_gate"]
    active_fraction = float(np.mean([item["active_channels"] >= 2 for item in sequence_audits]))
    quantized_fraction = float(
        np.mean(
            [item["quantized_levels_min"] >= _MIN_UNIQUE_QUANTIZED_LEVELS for item in sequence_audits]
        )
    )
    t50_fraction = float(np.mean([item["t50_pair_separation"] for item in sequence_audits]))
    family_degenerate: dict[str, float] = {}
    for family in _FAMILIES:
        items = [item for item in sequence_audits if item["family"] == family]
        family_degenerate[family] = float(
            np.mean(
                [
                    item["active_channels"] < 2
                    or item["quantized_levels_min"] < _MIN_UNIQUE_QUANTIZED_LEVELS
                    for item in items
                ]
            )
        )
    phase_counts = {
        phase: int(sum(item["phase_counts"][phase] for item in sequence_audits))
        for phase in ("baseline", "transition", "steady", "recovery")
    }
    phase_nonempty = {phase: count > 0 for phase, count in phase_counts.items()}
    recovery_sequence_fraction = float(
        np.mean([item["phase_counts"]["recovery"] > 0 for item in sequence_audits])
    )
    checks = {
        "nonpure_active_channel_fraction": active_fraction >= float(gate["minimum_active_channel_fraction"]),
        "quantized_level_fraction": quantized_fraction >= float(gate["minimum_quantized_level_fraction"]),
        "low_frequency_t50_pair_fraction": t50_fraction >= float(gate["minimum_t50_pair_fraction"]),
        "all_phases_nonempty": all(phase_nonempty.values()),
        "transition_variance_nonwhite": any(float(item["transition_variance"]) > 1.0e-16 for item in sequence_audits),
        "family_degenerate_fraction": all(
            value <= float(gate["maximum_family_degenerate_fraction"])
            for value in family_degenerate.values()
        ),
    }
    return {
        "checks": checks,
        "active_channel_fraction": active_fraction,
        "quantized_level_fraction": quantized_fraction,
        "low_frequency_t50_pair_fraction": t50_fraction,
        "family_degenerate_fraction": family_degenerate,
        "phase_counts": phase_counts,
        "phase_nonempty": phase_nonempty,
        "recovery_sequence_fraction": recovery_sequence_fraction,
        "sample_rate_hz": sample_rate_hz,
        "duration_s": duration_s,
    }


def _aggregate_device_audit(
    sequence_audits: Sequence[Mapping[str, Any]],
    ndir_low_co2_delta_v: float,
    pilot: Mapping[str, Any],
) -> dict[str, Any]:
    gate = pilot["dynamic_gate"]
    max_residual = float(max(item["tcd_energy_residual_max_w"] for item in sequence_audits))
    saturation = float(max(item["ndir_saturation_fraction"] for item in sequence_audits))
    bound_by_sensor = [
        int(sum(item["bound_violations_by_sensor"][channel] for item in sequence_audits))
        for channel in range(3)
    ]
    checks = {
        "tcd_energy_balance": max_residual <= float(gate["maximum_tcd_energy_residual_w"]),
        "ndir_no_saturation": saturation <= float(gate["maximum_ndir_saturation_fraction"]),
        "ndir_low_co2_sensitivity": ndir_low_co2_delta_v >= float(gate["minimum_ndir_low_co2_delta_v"]),
        "signal_bounds": sum(bound_by_sensor) == 0,
    }
    return {
        "checks": checks,
        "max_tcd_energy_residual_w": max_residual,
        "max_ndir_saturation_fraction": saturation,
        "ndir_0p5_minus_0molpct_absolute_delta_v": ndir_low_co2_delta_v,
        "bound_violations_by_sensor": bound_by_sensor,
        "bound_violations": int(sum(bound_by_sensor)),
        "observed_min_by_sensor": [
            float(min(item["observed_min_by_sensor"][channel] for item in sequence_audits))
            for channel in range(3)
        ],
        "observed_max_by_sensor": [
            float(max(item["observed_max_by_sensor"][channel] for item in sequence_audits))
            for channel in range(3)
        ],
    }


def _ndir_low_co2_delta(data_config: Mapping[str, Any], dt_s: float) -> float:
    profile = data_config["hardware_profiles"]["ndir"]["profiles"][0]
    environment = next(
        item for item in data_config["environment_profiles"] if item["environment_id"] == "ENV-NOMINAL"
    )
    zero = simulate_ndir(
        np.repeat(np.asarray([[100.0, 0.0, 0.0]]), 4, axis=0),
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        dt_s=dt_s,
        profile=profile,
    )
    low = simulate_ndir(
        np.repeat(np.asarray([[99.5, 0.0, 0.5]]), 4, axis=0),
        temperature_k=environment["temperature_k"],
        pressure_pa=environment["pressure_pa"],
        dt_s=dt_s,
        profile=profile,
    )
    return abs(float(low.clean_voltage_v[-1] - zero.clean_voltage_v[-1]))


def _probe_checks(metrics: Mapping[str, Any], pilot: Mapping[str, Any]) -> dict[str, Any]:
    privileged = metrics.get("P-O-KIN-LS", {}).get("P060", {}).get("stress_val")
    endpoint = metrics.get("P-B-LAST-LS", {}).get("P060", {}).get("stress_val")
    improvement = _relative_probe_improvement(privileged, endpoint)
    p150 = metrics.get("P-B-LAST-LS", {}).get("P150", {}).get("stress_val")
    p015 = metrics.get("P-B-LAST-LS", {}).get("P015", {}).get("stress_val")
    early_degradation = None
    if p150 and p015 and float(p150["macro_RNMAE"]) > 0.0:
        early_degradation = float(p015["macro_RNMAE"]) / float(p150["macro_RNMAE"]) - 1.0
    threshold = float(pilot["dynamic_gate"]["minimum_stress_privileged_probe_improvement_fraction"])
    return {
        "stress_identifiable": improvement is not None and improvement >= threshold,
        "stress_privileged_probe_improvement_fraction": improvement,
        "stress_p_o_kin_ls_p060_macro_RNMAE": None if not privileged else float(privileged["macro_RNMAE"]),
        "stress_p_b_last_ls_p060_macro_RNMAE": None if not endpoint else float(endpoint["macro_RNMAE"]),
        "p_b_last_ls_p015_over_p150_degradation": early_degradation,
        "metric_definition": "group-level target-range RNMAE from gf.dl.evaluation.evaluate_predictions",
    }


def _relative_probe_improvement(
    privileged: Mapping[str, Any] | None,
    endpoint: Mapping[str, Any] | None,
) -> float | None:
    if not privileged or not endpoint:
        return None
    baseline = float(endpoint["macro_RNMAE"])
    candidate = float(privileged["macro_RNMAE"])
    if not math.isfinite(baseline) or not math.isfinite(candidate) or baseline <= 0.0:
        return None
    return 1.0 - candidate / baseline


def _evaluate_ultrasonic_candidates(
    data_config: Mapping[str, Any],
    pilot: Mapping[str, Any],
    probe_compositions: Sequence[np.ndarray],
) -> dict[str, Any]:
    if len(probe_compositions) != int(pilot["ultrasonic_quality_sample_count"]):
        raise ValueError("pilot ultrasonic probe count does not match the registered sample count")
    hardware = data_config["hardware_profiles"]["ultrasonic"]
    profiles = {item["ultrasonic_profile_id"]: item for item in hardware["candidates"]}
    multipath_profiles = {
        item["multipath_profile_id"]: item for item in hardware["multipath_profiles"]
    }
    environment = next(
        item for item in data_config["environment_profiles"] if item["environment_id"] == "ENV-NOMINAL"
    )
    model_id = data_config["physics_reference"]["eos"]["sound_speed_model_id"]
    results: dict[str, Any] = {}
    for profile_offset, profile_id in enumerate(pilot["ultrasonic_candidate_ids"]):
        profile = profiles[profile_id]
        errors: list[float] = []
        snr_values: list[float] = []
        lock_count = 0
        elapsed = 0.0
        by_multipath: dict[str, dict[str, Any]] = {}
        for multipath_offset, multipath_id in enumerate(("US-MP-NOMINAL", "US-MP-OOD")):
            start_index = len(errors)
            start_locks = lock_count
            for index, composition in enumerate(probe_compositions):
                truth = evaluate_shared_physics(
                    np.asarray([composition]),
                    temperature_k=environment["temperature_k"],
                    pressure_pa=environment["pressure_pa"],
                    path_length_m=float(profile["path_length_m"]),
                    sound_speed_model_id=model_id,
                )["tof_s"][0]
                started = time.perf_counter()
                acquisition = acquire_ultrasonic_tof(
                    composition,
                    temperature_k=environment["temperature_k"],
                    pressure_pa=environment["pressure_pa"],
                    profile=profile,
                    multipath_profile=multipath_profiles[multipath_id],
                    signal_amplitude=ultrasonic_signal_amplitude(composition, profile),
                    internal_noise_std=float(pilot["ultrasonic_internal_noise_std"]),
                    rng=np.random.default_rng(
                        int(pilot["seed"])
                        + 10_000 * profile_offset
                        + 1_000 * multipath_offset
                        + index
                    ),
                    strict=False,
                    retain_waveform=False,
                    sound_speed_model_id=model_id,
                )
                elapsed += time.perf_counter() - started
                lock_count += int(acquisition.lock_status)
                snr_values.append(float(acquisition.snr))
                errors.append(
                    math.inf if acquisition.tof_s is None else abs(float(acquisition.tof_s) - float(truth))
                )
            subset = np.asarray(errors[start_index:], dtype=np.float64)
            by_multipath[multipath_id] = {
                "samples": int(subset.size),
                "lock_rate": (lock_count - start_locks) / int(subset.size),
                "p95_absolute_tof_error_s": math.inf if not np.isfinite(subset).any() else float(np.percentile(subset[np.isfinite(subset)], 95.0)),
            }
        finite_errors = np.asarray([value for value in errors if math.isfinite(value)], dtype=np.float64)
        p95_error = math.inf if finite_errors.size == 0 else float(np.percentile(finite_errors, 95.0))
        total = len(errors)
        lock_rate = lock_count / total
        results[profile_id] = {
            "tof_estimator": profile["tof_estimator"],
            "excitation_type": profile["excitation_type"],
            "composition_samples": len(probe_compositions),
            "acquisition_samples": total,
            "multipath_audit": by_multipath,
            "p95_absolute_tof_error_s": p95_error,
            "median_snr": float(np.median(snr_values)),
            "lock_rate": lock_rate,
            "mean_latency_ms": elapsed / total * 1000.0,
            "waveform_persisted": False,
            "checks": {
                "precision": p95_error <= float(pilot["selection_rule"]["p95_tof_error_gate_s"]),
                "lock_rate": lock_rate >= float(pilot["selection_rule"]["lock_rate_gate"]),
                "nominal_and_ood_multipath": set(by_multipath) == {"US-MP-NOMINAL", "US-MP-OOD"},
                "no_theoretical_fallback": all(math.isfinite(value) for value in errors),
            },
        }
    return results


def _select_pilot_axes(
    pilot: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    scenarios: Mapping[str, Mapping[str, Any]],
    ultrasonic: Mapping[str, Any],
) -> dict[str, Any]:
    rule = pilot["selection_rule"]
    comparison_duration = float(rule["sample_rate_comparison_duration_s"])
    minimum_rate = 1.0 / float(eval_config["realtime"]["minimum_update_period_s"])
    rate_candidates: dict[str, Any] = {}
    qualified_scores: list[tuple[float, float]] = []
    for raw_rate in pilot["sample_rates_hz"]:
        rate = float(raw_rate)
        scenario = scenarios[_scenario_key(rate, comparison_duration)]
        checks = _probe_checks(scenario["pilot_probe_metrics"], pilot)
        score = checks["stress_privileged_probe_improvement_fraction"]
        dynamic_and_device = all(scenario["dynamic_audit"]["checks"].values()) and all(
            scenario["device_audit"]["checks"].values()
        )
        qualifies = bool(rate >= minimum_rate and dynamic_and_device and score is not None)
        rate_candidates[f"{rate:g}Hz"] = {
            "meets_realtime_update_period": rate >= minimum_rate,
            "dynamic_and_device_qualified": dynamic_and_device,
            "information_score": score,
            "qualified": qualifies,
        }
        if qualifies:
            qualified_scores.append((rate, float(score)))
    selected_rate: float | None = None
    if qualified_scores:
        best_score = max(score for _, score in qualified_scores)
        threshold = best_score - float(rule["max_information_score_gap"])
        selected_rate = min(rate for rate, score in qualified_scores if score >= threshold)

    duration_candidates: dict[str, Any] = {}
    selected_duration: float | None = None
    if selected_rate is not None:
        for raw_duration in sorted(float(item) for item in pilot["durations_s"]):
            scenario = scenarios[_scenario_key(selected_rate, raw_duration)]
            eligible = _duration_eligible(scenario)
            duration_candidates[f"{raw_duration:g}s"] = {
                "p150_count": int(scenario["horizon_availability"]["P150"]),
                "recovery_sequence_fraction": float(scenario["dynamic_audit"]["recovery_sequence_fraction"]),
                "eligible": eligible,
            }
            if selected_duration is None and eligible:
                selected_duration = raw_duration

    candidates = [
        (profile_id, item)
        for profile_id, item in ultrasonic.items()
        if all(item["checks"].values())
    ]
    candidates.sort(
        key=lambda pair: (
            pair[1]["p95_absolute_tof_error_s"],
            -pair[1]["lock_rate"],
            -pair[1]["median_snr"],
            pair[1]["mean_latency_ms"],
            pair[0],
        )
    )
    selected_ultrasonic = None if not candidates else candidates[0][0]
    selected_item = None if not candidates else candidates[0][1]
    return {
        "sample_rate_hz": selected_rate,
        "duration_s": selected_duration,
        "ultrasonic_profile_id": selected_ultrasonic,
        "tof_estimator": None if selected_item is None else selected_item["tof_estimator"],
        "excitation_type": None if selected_item is None else selected_item["excitation_type"],
        "sample_rate_candidates": rate_candidates,
        "duration_candidates": duration_candidates,
        "rule": rule,
    }


def _duration_eligible(scenario: Mapping[str, Any]) -> bool:
    return bool(
        scenario["horizon_availability"].get("P150", 0) > 0
        and scenario["dynamic_audit"]["recovery_sequence_fraction"] > 0.0
        and all(scenario["dynamic_audit"]["checks"].values())
        and all(scenario["device_audit"]["checks"].values())
    )


def _selected_or_reference_key(
    pilot: Mapping[str, Any],
    selection: Mapping[str, Any],
    scenarios: Mapping[str, Mapping[str, Any]],
) -> str:
    if selection["sample_rate_hz"] is not None and selection["duration_s"] is not None:
        return _scenario_key(selection["sample_rate_hz"], selection["duration_s"])
    reference = _scenario_key(
        pilot["registered_reference_sample_rate_hz"],
        pilot["registered_reference_duration_s"],
    )
    if reference not in scenarios:
        raise ValueError("registered reference scenario is absent")
    return reference


def _resource_summary(
    pilot: Mapping[str, Any],
    peak_bytes: int,
    audit_scenario: Mapping[str, Any],
    base_cache: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gate = pilot["resource_gate"]
    rows = int(gate["formal_rows"])
    timesteps = int(gate["formal_timesteps"])
    channels = int(gate["signal_channels"])
    float32_bytes = int(gate["float32_bytes"])
    signal_bytes = rows * channels * timesteps * float32_bytes
    breakdown = {
        "signals_float32": signal_bytes,
        "valid_mask_bool": rows * channels * timesteps,
        "quality_float32": signal_bytes,
        "time_s_float64": timesteps * 8,
        "target_float32": rows * channels * float32_bytes,
        "phase_id_int8": rows * timesteps,
        "observation_index_int64": rows * 8,
        "oracle_float32_arrays": signal_bytes * int(gate["formal_oracle_float32_channel_arrays"]),
    }
    formal_core_bytes = int(sum(breakdown.values()))
    pilot_ok = int(peak_bytes) <= int(gate["maximum_pilot_peak_bytes"])
    formal_ok = formal_core_bytes <= int(gate["maximum_formal_core_array_bytes"])
    return {
        "process_peak_working_set_bytes": int(peak_bytes),
        "base_cache_unique_array_bytes": _base_cache_unique_array_bytes(base_cache),
        "max_sequence_working_bytes": int(audit_scenario["max_sequence_bytes"]),
        "formal_array_breakdown_bytes": breakdown,
        "formal_core_array_bytes": formal_core_bytes,
        "waveform_persistence": gate["waveform_persistence"],
        "waveform_persisted_bytes": 0,
        "pilot_peak_within_limit": pilot_ok,
        "formal_core_arrays_within_limit": formal_ok,
        "within_limit": pilot_ok and formal_ok,
    }


def _base_cache_unique_array_bytes(base_cache: Sequence[Mapping[str, Any]]) -> int:
    arrays: list[np.ndarray] = []
    for item in base_cache:
        arrays.extend((item["source_time_s"], item["clean"], item["tcd_residual"]))
        layers = item["layers"]
        arrays.append(layers.chamber_composition_pct)
        arrays.extend(layers.local_composition_pct.values())
    unique = {id(array): array for array in arrays}
    return int(sum(array.nbytes for array in unique.values()))


def _dependency_hashes(
    root: Path,
    data_config: Mapping[str, Any],
    a2h_relative: str,
) -> dict[str, str]:
    relative_paths = [
        "configs/data/ar_he_co2_a2_dynamic_v1.json",
        str(data_config["physics_reference"]["eos"]["model_asset"]["path"]),
        "configs/eval/a2_dynamic_eval.json",
        "configs/experiment/a2_dynamic_protocol.json",
        a2h_relative,
        "src/gf/__init__.py",
        "src/gf/pipeline/__init__.py",
        "src/gf/pipeline/a2_dynamic_pilot.py",
        "src/gf/pipeline/a2_dynamic_protocol.py",
        "src/gf/sim/__init__.py",
        "src/gf/sim/a2_dynamic_dataset.py",
        "src/gf/sim/a2_dynamic_physics.py",
        "src/gf/sim/a2_sensor_devices.py",
        "src/gf/sim/a2dyn_sound_speed.py",
        "src/gf/sim/ar_he_co2.py",
        "src/gf/dl/temporal_baselines.py",
        "src/gf/dl/evaluation.py",
        "src/gf/dl/__init__.py",
    ]
    for source_id in ("a1", "a2h"):
        source = data_config["source_registry"][source_id]
        relative_paths.extend((str(source["config_path"]), str(source["manifest_path"])))
    relative_paths.extend(
        str(item["path"])
        for item in data_config["physics_reference"]["ndir_reference"]["asset_files"]
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = (root / relative).resolve()
        result[relative.replace("\\", "/")] = _sha256_file(path)
    return result


def _peak_process_working_set_bytes() -> int:
    """返回当前进程生命周期峰值 RSS，不启用高开销分配追踪。"""

    if os.name == "nt":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = kernel32.GetCurrentProcess()
        succeeded = psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not succeeded:
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource

    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum_rss if os.uname().sysname == "Darwin" else maximum_rss * 1024


def _split_counts(assignments: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {split: sum(item["split"] == split for item in assignments) for split in _SPLITS}


def _pilot_split_totals(pilot: Mapping[str, Any]) -> dict[str, int]:
    per_family = pilot["split_counts_per_family"]
    return {
        split: int(per_family[split]) * len(_FAMILIES)
        for split in _SPLITS
    }


def _scenario_key(sample_rate_hz: float, duration_s: float) -> str:
    return f"{float(sample_rate_hz):g}Hz_{float(duration_s):g}s"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["run_a2_dynamic_pilot"]
