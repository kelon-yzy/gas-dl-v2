from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from gf.sim.a2_dynamic_dataset import (
    _build_dynamic_manifest,
    _build_test_assignments,
    _serialize_composition_float32,
    apply_dynamic_observation_profile,
    resample_continuous_series,
    resolve_protocol_instance,
    sample_registered_range,
    unique_quantization_level_counts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dynamic_data_config() -> dict[str, object]:
    path = PROJECT_ROOT / "configs" / "data" / "ar_he_co2_a2_dynamic_v1.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_float32_composition_serialization_closes_each_time_step() -> None:
    values = np.asarray(
        [
            [[20.123456789, 30.987654321, 48.88888889]],
            [[0.123456789, 4.987654321, 94.88888889]],
        ],
        dtype=np.float64,
    )

    serialized = _serialize_composition_float32(values)

    assert serialized.dtype == np.float32
    assert np.allclose(
        np.sum(serialized, axis=-1, dtype=np.float32),
        np.float32(100.0),
        rtol=0.0,
        atol=1.0e-6,
    )


def test_registered_log_uniform_sampling_and_zero_mass_are_explicit() -> None:
    sampled = sample_registered_range([1.0, 100.0], 0, distribution="log_uniform")
    expected = math.exp(0.6180339887498949 * math.log(100.0))
    assert math.isclose(sampled, expected, rel_tol=0.0, abs_tol=1.0e-12)

    values = [
        sample_registered_range(
            [0.0, 2.0],
            index,
            distribution="log_uniform",
            zero_probability=0.2,
            minimum_nonzero=0.05,
        )
        for index in range(20)
    ]
    assert 0.0 in values
    assert all(value == 0.0 or 0.05 <= value <= 2.0 for value in values)


def test_continuous_resampling_uses_exact_target_timestamps() -> None:
    source_time = np.arange(0.0, 4.0, 0.2)
    target_time = np.arange(0.0, 4.0, 0.5)
    values = np.column_stack((source_time, source_time**2))
    resampled = resample_continuous_series(source_time, values, target_time)

    assert np.array_equal(resampled[:, 0], target_time)
    assert np.max(np.abs(resampled[:, 1] - target_time**2)) <= 0.0100000001


def test_protocol_instance_preserves_true_onset_end_and_multi_pulse_ranges() -> None:
    no_jitter = {"phase_duration_jitter_pct": [0.0, 0.0]}
    step = resolve_protocol_instance(
        {"kind": "step", "onset_s": 30.0, "exposure_end_s": 180.0},
        no_jitter,
        index=0,
    )
    assert step.exposure_onset_s == 30.0
    assert step.exposure_end_s == 180.0
    assert step.parameters["exposure_end_s"] == 180.0

    multi = resolve_protocol_instance(
        {
            "kind": "multi_pulse",
            "onset_s": 30.0,
            "pulse_count_range": [2, 3],
            "pulse_width_range_s": [15.0, 30.0],
            "pulse_period_range_s": [50.0, 70.0],
        },
        no_jitter,
        index=1,
    )
    assert multi.parameters["pulse_count"] in {2, 3}
    assert 15.0 <= multi.parameters["pulse_width_s"] <= 30.0
    assert 50.0 <= multi.parameters["pulse_period_s"] <= 70.0
    assert multi.exposure_end_s > multi.exposure_onset_s


def test_observation_profile_applies_calibration_and_counts_actual_levels() -> None:
    clean = np.asarray([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]])
    noise = {
        "white_noise_scale": 1.0,
        "ar1_rho_range": [0.0, 0.0],
        "shared_correlation_load_range": [0.0, 0.0],
        "drift_strength_range_pct_dynamic_range_per_min": [0.0, 0.0],
    }
    calibration = {
        "sensor_gains": {
            "ultrasonic_tof": 2.0,
            "thermal_conductivity_voltage": 3.0,
            "ndir_co2_voltage": 4.0,
        },
        "sensor_offsets": {
            "ultrasonic_tof": 0.1,
            "thermal_conductivity_voltage": 0.2,
            "ndir_co2_voltage": 0.3,
        },
    }
    observed, audit = apply_dynamic_observation_profile(
        clean,
        noise_profile=noise,
        calibration_profile=calibration,
        noise_base=[0.0, 0.0, 0.0],
        quantization=[1.0e-9, 1.0e-9, 1.0e-9],
        dt_s=0.5,
        index=0,
        rng=np.random.default_rng(7),
    )

    assert np.allclose(observed, clean * [2.0, 3.0, 4.0] + [0.1, 0.2, 0.3])
    assert np.array_equal(unique_quantization_level_counts(observed), [3, 3, 3])
    assert audit.drift_strength_pct_dynamic_range_per_min == 0.0


def test_test_assignments_follow_frozen_test_quota_and_pure_policy() -> None:
    data = _dynamic_data_config()

    assignments = _build_test_assignments(data, start_group_index=3780)

    assert len(assignments) == 630
    assert {assignment["split"] for assignment in assignments} == {"test"}
    regions = [assignment["composition_region"] for assignment in assignments]
    assert regions.count("interior") == 315
    assert regions.count("near_boundary") == 189
    assert regions.count("binary") == 123
    assert regions.count("pure") == 3
    assert [assignment["family"] for assignment in assignments].count("D-JOINT") == 90
    pure = [assignment for assignment in assignments if assignment["composition_region"] == "pure"]
    assert [assignment["mixture_id"] for assignment in pure] == [
        "a2dyn-mix-pure-Ar",
        "a2dyn-mix-pure-He",
        "a2dyn-mix-pure-CO2",
    ]
    assert all(assignment["family"] == "D-JOINT" for assignment in pure)
    non_pure = [assignment for assignment in assignments if assignment["composition_region"] != "pure"]
    assert len(non_pure) == 627
    numeric_ids = [int(assignment["mixture_id"].rsplit("-", 1)[1]) for assignment in non_pure]
    assert numeric_ids == list(range(3781, 3781 + 627))
    assert [assignment["target_index"] for assignment in assignments] == list(
        range(3780, 3780 + 630)
    )
    compositions = [tuple(assignment["composition"]) for assignment in non_pure]
    assert len(set(compositions)) == len(compositions)


def test_test_assignments_avoid_development_compositions() -> None:
    data = _dynamic_data_config()
    base = _build_test_assignments(data, start_group_index=3780)
    blocked = {
        tuple(assignment["composition"])
        for assignment in base
        if assignment["composition_region"] != "pure"
    }
    blocked = set(list(blocked)[:50])

    assignments = _build_test_assignments(
        data,
        start_group_index=3780,
        development_compositions=blocked,
    )

    got = {
        tuple(assignment["composition"])
        for assignment in assignments
        if assignment["composition_region"] != "pure"
    }
    assert not (got & blocked)
    assert len(got) == 627


def _mini_manifest_arrays(record_count: int, timesteps: int = 1200) -> dict[str, object]:
    sensor_count = 3
    return {
        "signals": np.zeros((record_count, sensor_count, timesteps, 1), dtype=np.float32),
        "valid_mask": np.ones((record_count, sensor_count, timesteps, 1), dtype=np.bool_),
        "quality": np.ones((record_count, sensor_count, timesteps), dtype=np.float32),
        "time_s": np.arange(timesteps, dtype=np.float64) * 0.2,
        "target": np.zeros((record_count, 3), dtype=np.float32),
        "phase_id": np.zeros((record_count, timesteps), dtype=np.int8),
        "observation_index": np.arange(record_count, dtype=np.int64),
        "inlet": np.zeros((record_count, timesteps, 3), dtype=np.float32),
        "inlet_coefficient": np.zeros((record_count, timesteps), dtype=np.float32),
        "chamber": np.zeros((record_count, timesteps, 3), dtype=np.float32),
        "equilibrium_reference": np.zeros((record_count, sensor_count, timesteps), dtype=np.float32),
        "clean_device": np.zeros((record_count, sensor_count, timesteps), dtype=np.float32),
        "device_states": np.zeros((record_count, sensor_count, timesteps), dtype=np.float32),
        "privileged": np.zeros((record_count, 12), dtype=np.float64),
        "device_audit": {
            "ultrasonic_peak_correlation": np.zeros((record_count, timesteps), dtype=np.float32),
            "ultrasonic_snr": np.zeros((record_count, timesteps), dtype=np.float32),
            "ultrasonic_estimated_tof_uncertainty_s": np.zeros((record_count, timesteps), dtype=np.float32),
            "ultrasonic_lock_status": np.ones((record_count, timesteps), dtype=np.bool_),
            "tcd_energy_balance_residual_w": np.zeros((record_count, timesteps), dtype=np.float32),
            "ndir_active_voltage_v": np.zeros((record_count, timesteps), dtype=np.float32),
            "ndir_reference_voltage_v": np.zeros((record_count, timesteps), dtype=np.float32),
            "ndir_saturation_mask": np.zeros((record_count, timesteps), dtype=np.bool_),
            "ndir_quantization_platform_length": np.zeros(record_count, dtype=np.int32),
        },
    }


def test_complete_manifest_build_covers_test_split_and_pure_region() -> None:
    data = _dynamic_data_config()
    records = [
        {"mixture_id": "a2dyn-mix-000001", "family": "D-IID", "split": "train", "composition_region": "interior"},
        {"mixture_id": "a2dyn-mix-000002", "family": "D-IID", "split": "val", "composition_region": "near_boundary"},
        {"mixture_id": "a2dyn-mix-000003", "family": "D-IID", "split": "stress_val", "composition_region": "binary"},
        {"mixture_id": "a2dyn-mix-000004", "family": "D-IID", "split": "test", "composition_region": "interior"},
        {"mixture_id": "a2dyn-mix-pure-Ar", "family": "D-JOINT", "split": "test", "composition_region": "pure"},
    ]
    arrays = _mini_manifest_arrays(len(records))

    manifest = _build_dynamic_manifest(
        data=data,
        evaluation={},
        experiment={},
        records=records,
        source_hashes={"configs/data/ar_he_co2_a2_dynamic_v1.json": "test"},
        splits=("train", "val", "stress_val", "test"),
        regions=("interior", "near_boundary", "binary", "pure"),
        development_only=False,
        contains_test=True,
        manifest_status="TEST_GENERATED",
        **arrays,
    )

    assert manifest["development_only"] is False
    assert manifest["contains_test"] is True
    assert manifest["status"] == "TEST_GENERATED"
    assert set(manifest["split_groups"]) == {"train", "val", "stress_val", "test"}
    assert manifest["region_counts"]["test"]["pure"] == 1
    assert manifest["region_counts"]["test"]["interior"] == 1
    assert isinstance(manifest["content_sha256"], str)
    assert isinstance(manifest["split_hash"], str)
    assert len(manifest["split_groups"]["test"]) == 2
