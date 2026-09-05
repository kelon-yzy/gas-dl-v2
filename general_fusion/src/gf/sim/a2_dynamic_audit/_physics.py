"""守恒、边界与设备级审计（物理真实性）。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from gf.sim.a2_dynamic_dataset import DynamicDataset
from gf.sim.a2_dynamic_audit._shared import TARGET_TOTAL


def _audit_physics(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    physics_audit: Mapping[str, Any] | None,
    *,
    subset_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """全包（或指定子集）的守恒、边界与设备审计。

    默认对全部观测统计；``subset_indices`` 只用于 A2-DYN-4 冻结审计剔除
    目标等于 purge 的 pure 顶点序列（它们按 §10.5 单列为边界审计）。
    """

    signal_bounds = data_config["signal_bounds"]
    observed = np.transpose(dataset.signals[:, :, :, 0], (0, 2, 1))
    inlet_composition = np.asarray(dataset.inlet_composition)
    chamber_composition = np.asarray(dataset.chamber_composition)
    device_audit = {key: np.asarray(value) for key, value in dataset.device_audit.items()}
    if subset_indices is not None:
        indices = np.asarray(subset_indices, dtype=np.int64)
        if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= dataset.sample_count):
            raise ValueError("physics audit subset_indices must be valid row indices")
        observed = observed[indices]
        inlet_composition = inlet_composition[indices]
        chamber_composition = chamber_composition[indices]
        device_audit = {key: value[indices] for key, value in device_audit.items()}
    bound_checks: dict[str, bool] = {}
    outside_fraction: dict[str, float] = {}
    for channel, sensor_id in enumerate(data_config["sensor_ids"]):
        lower, upper = (float(value) for value in signal_bounds[sensor_id])
        outside = (observed[:, :, channel] < lower) | (observed[:, :, channel] > upper)
        outside_fraction[str(sensor_id)] = float(np.mean(outside))
        bound_checks[str(sensor_id)] = bool(not np.any(outside))
    inlet_sum_error = np.abs(inlet_composition.sum(axis=2) - TARGET_TOTAL)
    chamber_sum_error = np.abs(chamber_composition.sum(axis=2) - TARGET_TOTAL)
    pilot_gate = data_config.get("pilot_dynamic_gate", {})
    physics_gate = eval_config["qualification_gates"]["physics_and_schema"]
    inlet_sum_tolerance = float(physics_gate["max_inlet_sum_error_pct"])
    chamber_sum_tolerance = float(physics_gate["max_chamber_sum_error_pct"])
    tcd_residual = float(np.max(np.abs(device_audit["tcd_energy_balance_residual_w"])))
    tcd_limit = float(pilot_gate.get("maximum_tcd_energy_residual_w", 1.0e-10))
    ndir_saturation_fraction = float(np.mean(device_audit["ndir_saturation_mask"]))
    ultrasonic_lock_rate = float(np.mean(device_audit["ultrasonic_lock_status"]))
    ultrasonic_peak = np.asarray(device_audit["ultrasonic_peak_correlation"], dtype=np.float64)
    ultrasonic_snr = np.asarray(device_audit["ultrasonic_snr"], dtype=np.float64)
    ultrasonic_uncertainty = np.asarray(
        device_audit["ultrasonic_estimated_tof_uncertainty_s"],
        dtype=np.float64,
    )
    external_checks = {
        "provided": physics_audit is not None,
        "status_pass": physics_audit is not None and physics_audit.get("status") == "PASS",
        "physics_verified": physics_audit is not None and physics_audit.get("physics_status") == "PHYSICS_VERIFIED",
        "heos_grid_consistency": physics_audit is not None and physics_audit.get("checks", {}).get("heos_generator_grid_consistency") is True,
        "heos_off_grid_consistency": physics_audit is not None and physics_audit.get("checks", {}).get("heos_generator_off_grid_consistency") is True,
        "heos_pressure_direction": physics_audit is not None and physics_audit.get("checks", {}).get("heos_pressure_direction") is True,
        "ndir_zero_and_sensitivity": physics_audit is not None and physics_audit.get("checks", {}).get("ndir_low_co2_sensitivity") is True,
        "thermal_parity": physics_audit is not None and physics_audit.get("checks", {}).get("steady_thermal_parity") is True,
        "old_speed_migration": physics_audit is not None and "ultrasonic_tof_new_minus_legacy_s" in physics_audit.get("parity", {}),
    }
    checks = {
        "finite_arrays": bool(
            np.isfinite(observed).all()
            and np.isfinite(inlet_composition).all()
            and np.isfinite(chamber_composition).all()
        ),
        "inlet_sum": bool(np.max(inlet_sum_error) <= inlet_sum_tolerance),
        "chamber_sum": bool(np.max(chamber_sum_error) <= chamber_sum_tolerance),
        "inlet_nonnegative": bool(np.all(inlet_composition >= 0.0)),
        "chamber_nonnegative": bool(np.all(chamber_composition >= 0.0)),
        "signal_bounds": all(bound_checks.values()),
        "ultrasonic_lock": ultrasonic_lock_rate >= 0.95,
        "ultrasonic_quality_finite": bool(
            np.isfinite(ultrasonic_peak).all()
            and np.isfinite(ultrasonic_snr).all()
            and np.isfinite(ultrasonic_uncertainty).all()
            and np.all(ultrasonic_peak >= 0.0)
            and np.all(ultrasonic_snr > 0.0)
            and np.all(ultrasonic_uncertainty > 0.0)
        ),
        "ultrasonic_quality_data_dependent": bool(
            np.ptp(ultrasonic_peak) > 0.0
            and np.ptp(ultrasonic_snr) > 0.0
            and np.ptp(ultrasonic_uncertainty) > 0.0
        ),
        "tcd_energy_balance": tcd_residual <= tcd_limit,
        "ndir_unsaturated": ndir_saturation_fraction == 0.0,
        "external_physics_audit": all(external_checks.values()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "signal_outside_fraction": outside_fraction,
        "max_inlet_sum_error_pct": float(np.max(inlet_sum_error)),
        "max_chamber_sum_error_pct": float(np.max(chamber_sum_error)),
        "configured_inlet_sum_tolerance_pct": inlet_sum_tolerance,
        "configured_chamber_sum_tolerance_pct": chamber_sum_tolerance,
        "closure_tolerance_basis": "configured qualification gates applied to serialized float32 oracle arrays",
        "ultrasonic_lock_rate": ultrasonic_lock_rate,
        "ultrasonic_peak_correlation_range": [float(np.min(ultrasonic_peak)), float(np.max(ultrasonic_peak))],
        "ultrasonic_snr_range": [float(np.min(ultrasonic_snr)), float(np.max(ultrasonic_snr))],
        "ultrasonic_uncertainty_range_s": [
            float(np.min(ultrasonic_uncertainty)),
            float(np.max(ultrasonic_uncertainty)),
        ],
        "tcd_max_energy_balance_residual_w": tcd_residual,
        "ndir_saturation_fraction": ndir_saturation_fraction,
        "external_physics_checks": external_checks,
        "audited_row_count": int(observed.shape[0]),
    }
