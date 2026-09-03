"""A2-DYN-0 机器协议与三份配置的轻量校验。

此模块只负责冻结阶段的契约校验：schema、范围、单位、profile 引用、
split 配额和已登记来源 hash。它不生成数据、不执行设备仿真，也不选择
任何时序算法。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


DATA_CONFIG_RELATIVE_PATH = Path("configs/data/ar_he_co2_a2_dynamic_v1.json")
EVAL_CONFIG_RELATIVE_PATH = Path("configs/eval/a2_dynamic_eval.json")
EXPERIMENT_CONFIG_RELATIVE_PATH = Path("configs/experiment/a2_dynamic_protocol.json")

DATA_SCHEMA_VERSION = "gf-a2-dynamic-data-1"
EVAL_SCHEMA_VERSION = "gf-a2-dynamic-eval-1"
EXPERIMENT_SCHEMA_VERSION = "gf-a2-dynamic-experiment-1"

SENSOR_IDS = (
    "ultrasonic_tof",
    "thermal_conductivity_voltage",
    "ndir_co2_voltage",
)
TARGET_NAMES = ("x_Ar_pct", "x_He_pct", "x_CO2_pct")
SPLITS = ("train", "val", "stress_val", "test")
FAMILIES = (
    "D-IID",
    "D-KINETICS",
    "D-PROTOCOL",
    "D-NOISE-DRIFT",
    "D-ENV-CAL",
    "D-JOINT",
)
SOURCE_LEVELS = (
    "shared_physics",
    "literature_structure",
    "literature_anchor",
    "existing_a0_proxy",
    "a2h_registered_range",
    "application_range",
    "sensitivity_tier_1",
    "sensitivity_tier_2",
    "hardware_calibrated",
)
FORBIDDEN_KEYS = {
    "sequence_id",
    "base_condition_id",
    "noise_seed_index",
    "noise_seed",
}
LEGACY_DYNAMIC_KEYS = {
    "tau_s",
    "sensor_multiplier",
    "tof_time_constant_s",
    "thermal_time_constant_s",
    "ndir_time_constant_s",
}

EXPECTED_PHASES = (
    ("baseline", 0.0, 30.0, 150),
    ("transition", 30.0, 90.0, 300),
    ("steady", 90.0, 180.0, 450),
    ("recovery", 180.0, 240.0, 300),
)
EXPECTED_HORIZONS = (
    ("P005", 5.0, 34.8, 175, True),
    ("P015", 15.0, 44.8, 225, True),
    ("P030", 30.0, 59.8, 300, True),
    ("P060", 60.0, 89.8, 450, True),
    ("P120", 120.0, 149.8, 750, True),
    ("P150", 150.0, 179.8, 900, True),
    ("FULL", 240.0, 239.8, 1200, False),
)
EXPECTED_TRANSPORT = {
    "KIN-TRAIN": {
        "tau_mix_s": [6.0, 18.0],
        "tau_transport_ultrasonic_s": [0.0, 1.0],
        "tau_transport_thermal_s": [1.0, 6.0],
        "tau_transport_ndir_s": [2.0, 10.0],
        "phase_duration_jitter_pct": [0.0, 5.0],
    },
    "KIN-VAL": {
        "tau_mix_s": [8.0, 22.0],
        "tau_transport_ultrasonic_s": [0.0, 2.0],
        "tau_transport_thermal_s": [2.0, 8.0],
        "tau_transport_ndir_s": [4.0, 14.0],
        "phase_duration_jitter_pct": [0.0, 8.0],
    },
    "KIN-STRESS": {
        "tau_mix_s": [24.0, 45.0],
        "tau_transport_ultrasonic_s": [1.0, 4.0],
        "tau_transport_thermal_s": [8.0, 18.0],
        "tau_transport_ndir_s": [12.0, 28.0],
        "phase_duration_jitter_pct": [8.0, 20.0],
    },
    "KIN-TEST": {
        "tau_mix_s": [45.0, 75.0],
        "tau_transport_ultrasonic_s": [2.0, 6.0],
        "tau_transport_thermal_s": [15.0, 30.0],
        "tau_transport_ndir_s": [24.0, 45.0],
        "phase_duration_jitter_pct": [15.0, 30.0],
    },
}
EXPECTED_FAMILY_GROUPS = {
    "D-IID": {"train": 720, "val": 180, "stress_val": 180, "test": 180},
    "D-KINETICS": {"train": 360, "val": 90, "stress_val": 90, "test": 90},
    "D-PROTOCOL": {"train": 360, "val": 90, "stress_val": 90, "test": 90},
    "D-NOISE-DRIFT": {"train": 360, "val": 90, "stress_val": 90, "test": 90},
    "D-ENV-CAL": {"train": 360, "val": 90, "stress_val": 90, "test": 90},
    "D-JOINT": {"train": 360, "val": 90, "stress_val": 90, "test": 90},
}
EXPECTED_FAMILY_REPEATS = {
    "D-IID": 1,
    "D-KINETICS": 1,
    "D-PROTOCOL": 1,
    "D-NOISE-DRIFT": 3,
    "D-ENV-CAL": 1,
    "D-JOINT": 2,
}
EXPECTED_REGION_QUOTAS = {
    "train": {"interior": 1260, "near_boundary": 756, "binary": 504, "pure": 0},
    "val": {"interior": 315, "near_boundary": 189, "binary": 126, "pure": 0},
    "stress_val": {"interior": 315, "near_boundary": 189, "binary": 126, "pure": 0},
    "test": {"interior": 315, "near_boundary": 189, "binary": 123, "pure": 3},
}
EXPECTED_SPLIT_TOTALS = {
    "train": {"groups": 2520, "observations": 3600},
    "val": {"groups": 630, "observations": 900},
    "stress_val": {"groups": 630, "observations": 900},
    "test": {"groups": 630, "observations": 900},
}
EXPECTED_NOISE = {
    "NOISE-1X": (1.0, [0.00, 0.40], [0.00, 0.10], [0.00, 0.10]),
    "NOISE-2X": (2.0, [0.20, 0.60], [0.10, 0.20], [0.10, 0.25]),
    "NOISE-5X": (5.0, [0.65, 0.85], [0.20, 0.35], [0.25, 0.75]),
    "NOISE-10X": (10.0, [0.85, 0.97], [0.35, 0.50], [0.75, 1.50]),
    "NOISE-CORR-5X": (5.0, [0.65, 0.85], [0.20, 0.35], [0.25, 0.75]),
}
EXPECTED_ENVIRONMENTS = {
    "ENV-TRAIN-LOW": (293.15, 98000.0),
    "ENV-NOMINAL": (298.15, 101325.0),
    "ENV-TRAIN-HIGH": (303.15, 105000.0),
    "ENV-NEAR": (308.15, 108000.0),
    "ENV-MID": (313.15, 112000.0),
    "ENV-FAR": (278.15, 90000.0),
}
EXPECTED_SIGNAL_BOUNDS = {
    "ultrasonic_tof": [0.0, 0.01],
    "thermal_conductivity_voltage": [-2.0, 4.0],
    "ndir_co2_voltage": [-1.0, 3.5],
}
REQUIRED_ULTRASONIC_FIELDS = {
    "ultrasonic_profile_id",
    "excitation_type",
    "path_length_m",
    "center_frequency_hz",
    "fractional_bandwidth",
    "adc_rate_hz",
    "pulse_repetition_hz",
    "average_count",
    "tof_estimator",
    "reference_waveform_id",
    "window_duration_s",
    "attenuation_nepers_per_m_by_component",
    "multipath_profile_id",
}
REQUIRED_THERMAL_FIELDS = {
    "thermal_profile_id",
    "heater_heat_capacity",
    "gas_conductance_scale",
    "substrate_conductance",
    "heater_power",
    "tcr",
    "bridge_voltage",
    "flow_coupling",
    "heater_resistance_ohm",
    "reference_temperature_k",
}
REQUIRED_NDIR_FIELDS = {
    "ndir_profile_id",
    "optical_path_m",
    "active_band_id",
    "reference_band_id",
    "source_spectrum_id",
    "detector_response_id",
    "effective_absorption_model_id",
    "tau_emitter_detector_s",
    "range_min_mol_pct",
    "range_max_mol_pct",
    "reference_asset_id",
    "hitran_table_name",
    "active_center_wavenumber_cm1",
    "active_fwhm_cm1",
    "reference_center_wavenumber_cm1",
    "reference_fwhm_cm1",
    "wavenumber_range_cm1",
    "wavenumber_step_cm1",
}


class A2DynamicProtocolError(ValueError):
    """A2-DYN-0 配置契约错误。"""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise A2DynamicProtocolError(f"{name} must be an object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise A2DynamicProtocolError(f"{name} must be an array")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise A2DynamicProtocolError(f"{name} must be a finite number")
    return float(value)


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise A2DynamicProtocolError(f"{name} must be positive")
    return result


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise A2DynamicProtocolError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise A2DynamicProtocolError(f"{name} must be >= {minimum}")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise A2DynamicProtocolError(f"{name} must be a non-empty string")
    return value


def _close(actual: Any, expected: float, name: str, *, tolerance: float = 1.0e-9) -> None:
    if not math.isclose(_finite(actual, name), expected, rel_tol=0.0, abs_tol=tolerance):
        raise A2DynamicProtocolError(f"{name} must be {expected}, got {actual!r}")


def _exact_list(actual: Any, expected: Sequence[Any], name: str) -> None:
    values = _sequence(actual, name)
    if list(values) != list(expected):
        raise A2DynamicProtocolError(f"{name} must be {list(expected)!r}, got {actual!r}")


def _range(value: Any, name: str, *, lower: float = 0.0, upper: float | None = None) -> list[float]:
    values = _sequence(value, name)
    if len(values) != 2:
        raise A2DynamicProtocolError(f"{name} must contain [minimum, maximum]")
    minimum = _finite(values[0], f"{name}[0]")
    maximum = _finite(values[1], f"{name}[1]")
    if minimum < lower or minimum > maximum or (upper is not None and maximum > upper):
        raise A2DynamicProtocolError(f"{name} has invalid range {list(values)!r}")
    return [minimum, maximum]


def _hash(value: Any, name: str) -> str:
    result = _nonempty_string(value, name).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise A2DynamicProtocolError(f"{name} must be a lowercase SHA-256 hex digest")
    return result


def _validate_no_forbidden_keys(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in FORBIDDEN_KEYS:
                raise A2DynamicProtocolError(f"forbidden legacy key at {path}.{key}")
            if key in LEGACY_DYNAMIC_KEYS:
                raise A2DynamicProtocolError(f"legacy fixed-response key at {path}.{key}")
            _validate_no_forbidden_keys(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_no_forbidden_keys(nested, f"{path}[{index}]")


def _validate_source_level(value: Any, name: str) -> None:
    if value not in SOURCE_LEVELS:
        raise A2DynamicProtocolError(f"{name} has unregistered source_level {value!r}")


def _validate_source_levels(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "source_level":
                _validate_source_level(nested, f"{path}.{key}")
            _validate_source_levels(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_source_levels(nested, f"{path}[{index}]")


def _validate_range_map(mapping: Mapping[str, Any], name: str, expected: Mapping[str, Sequence[float]]) -> None:
    missing = set(expected) - set(mapping)
    if missing:
        raise A2DynamicProtocolError(f"{name} is missing range fields {sorted(missing)}")
    for key, values in expected.items():
        actual = _range(mapping[key], f"{name}.{key}")
        if actual != [float(value) for value in values]:
            raise A2DynamicProtocolError(f"{name}.{key} must be {list(values)!r}, got {actual!r}")


def _validate_profile_metadata(profile: Mapping[str, Any], required_fields: set[str], name: str) -> None:
    missing = required_fields - set(profile)
    if missing:
        raise A2DynamicProtocolError(f"{name} is missing profile fields {sorted(missing)}")
    _validate_source_level(profile.get("source_level"), f"{name}.source_level")
    units = _mapping(profile.get("field_units"), f"{name}.field_units")
    sources = _mapping(profile.get("field_sources"), f"{name}.field_sources")
    missing_units = required_fields - set(units)
    if missing_units:
        raise A2DynamicProtocolError(f"{name}.field_units is missing {sorted(missing_units)}")
    missing_sources = required_fields - set(sources)
    if missing_sources:
        raise A2DynamicProtocolError(f"{name}.field_sources is missing {sorted(missing_sources)}")
    for field, source_level in sources.items():
        _validate_source_level(source_level, f"{name}.field_sources.{field}")


def validate_a2_dynamic_data_config(config: Mapping[str, Any]) -> None:
    """校验动态数据配置，不访问项目文件系统。"""

    _validate_no_forbidden_keys(config)
    _validate_source_levels(config)
    if config.get("schema_version") != DATA_SCHEMA_VERSION:
        raise A2DynamicProtocolError("A2-DYN data schema_version is unsupported")
    if config.get("dataset_id") != "ar_he_co2":
        raise A2DynamicProtocolError("A2-DYN dataset_id must be ar_he_co2")
    data_version = str(config.get("data_version", ""))
    if not data_version.startswith("gf-a2-dynamic-v1-") or not data_version.endswith("-r4"):
        raise A2DynamicProtocolError("A2-DYN data_version must use the frozen r4 namespace")
    if config.get("protocol_revision") != "a2-dyn-0-r4":
        raise A2DynamicProtocolError("A2-DYN protocol_revision is not frozen")
    if config.get("status") != "PROTOCOL_FROZEN":
        raise A2DynamicProtocolError("A2-DYN data config status must be PROTOCOL_FROZEN")
    _integer(config.get("generation_seed"), "generation_seed", minimum=0)
    _integer(config.get("split_seed"), "split_seed", minimum=0)
    if config.get("observation_mode") != "dynamic_exposure_sequence":
        raise A2DynamicProtocolError("A2-DYN observation_mode must be dynamic_exposure_sequence")
    _exact_list(config.get("sensor_ids"), SENSOR_IDS, "sensor_ids")
    _exact_list(config.get("target_names"), TARGET_NAMES, "target_names")
    if config.get("target_units") != "mol%":
        raise A2DynamicProtocolError("target_units must be mol%")
    _close(config.get("composition_total_pct"), 100.0, "composition_total_pct")
    _close(config.get("composition_quantization_pct"), 0.01, "composition_quantization_pct")

    axis = _mapping(config.get("time_axis"), "time_axis")
    _close(axis.get("duration_s"), 240.0, "time_axis.duration_s")
    _close(axis.get("sample_rate_hz"), 5.0, "time_axis.sample_rate_hz")
    _close(axis.get("dt_s"), 0.2, "time_axis.dt_s")
    _integer(axis.get("timesteps"), "time_axis.timesteps", minimum=1)
    if axis.get("timesteps") != 1200:
        raise A2DynamicProtocolError("time_axis.timesteps must be 1200")
    _close(axis.get("time_start_s"), 0.0, "time_axis.time_start_s")
    _close(axis.get("time_end_s"), 239.8, "time_axis.time_end_s")
    if axis.get("synchronization") != "shared_outer_axis":
        raise A2DynamicProtocolError("time_axis.synchronization must be shared_outer_axis")
    _integer(config.get("timesteps"), "timesteps", minimum=1)
    if config.get("timesteps") != 1200:
        raise A2DynamicProtocolError("timesteps must be 1200")
    _close(config.get("dt_s"), 0.2, "dt_s")

    phases = _sequence(config.get("phases"), "phases")
    if len(phases) != len(EXPECTED_PHASES):
        raise A2DynamicProtocolError("phases must contain baseline, transition, steady, and recovery")
    for raw, (phase_id, start, end, timesteps) in zip(phases, EXPECTED_PHASES):
        phase = _mapping(raw, f"phases[{phase_id}]")
        if phase.get("phase_id") != phase_id:
            raise A2DynamicProtocolError(f"phase order or id mismatch at {phase_id}")
        _close(phase.get("start_s"), start, f"phases[{phase_id}].start_s")
        _close(phase.get("end_s_exclusive"), end, f"phases[{phase_id}].end_s_exclusive")
        _integer(phase.get("timesteps"), f"phases[{phase_id}].timesteps", minimum=1)
        if phase.get("timesteps") != timesteps:
            raise A2DynamicProtocolError(f"phases[{phase_id}].timesteps must be {timesteps}")

    horizons = _sequence(config.get("prefix_horizons"), "prefix_horizons")
    if len(horizons) != len(EXPECTED_HORIZONS):
        raise A2DynamicProtocolError("prefix_horizons must contain P005 through FULL")
    for raw, expected in zip(horizons, EXPECTED_HORIZONS):
        horizon = _mapping(raw, f"prefix_horizons[{expected[0]}]")
        horizon_id, exposure_after, cutoff, timesteps, realtime = expected
        if horizon.get("horizon_id") != horizon_id:
            raise A2DynamicProtocolError(f"prefix horizon order or id mismatch at {horizon_id}")
        _close(horizon.get("exposure_after_s"), exposure_after, f"{horizon_id}.exposure_after_s")
        _close(horizon.get("cutoff_s"), cutoff, f"{horizon_id}.cutoff_s")
        _integer(horizon.get("timesteps"), f"{horizon_id}.timesteps", minimum=1)
        if horizon.get("timesteps") != timesteps or horizon.get("realtime") is not realtime:
            raise A2DynamicProtocolError(f"{horizon_id} timing fields are not frozen")

    inlet = _mapping(config.get("inlet"), "inlet")
    _exact_list(inlet.get("purge_composition_pct"), [100.0, 0.0, 0.0], "inlet.purge_composition_pct")
    _exact_list(inlet.get("composition_order"), ["Ar", "He", "CO2"], "inlet.composition_order")
    _range(inlet.get("coefficient_range"), "inlet.coefficient_range", lower=0.0, upper=1.0)
    if inlet.get("composition_formula") != "u(t)=(1-b(t))*x_purge+b(t)*x_target":
        raise A2DynamicProtocolError("inlet composition formula is not the frozen contract")
    if inlet.get("target_is_fixed_per_observation") is not True:
        raise A2DynamicProtocolError("target_is_fixed_per_observation must be true")

    protocols = _sequence(config.get("protocol_profiles"), "protocol_profiles")
    protocol_map: dict[str, Mapping[str, Any]] = {}
    for raw in protocols:
        profile = _mapping(raw, "protocol_profiles[]")
        profile_id = _nonempty_string(profile.get("protocol_profile_id"), "protocol_profile_id")
        if profile_id in protocol_map:
            raise A2DynamicProtocolError(f"duplicate protocol_profile_id {profile_id!r}")
        protocol_map[profile_id] = profile
    expected_protocols = {
        "STEP_STANDARD",
        "RAMP_LINEAR",
        "RAMP_SMOOTH",
        "ONSET_SHIFT",
        "SHORT_PULSE",
        "MULTI_PULSE",
        "INCOMPLETE_RECOVERY",
    }
    if set(protocol_map) != expected_protocols:
        raise A2DynamicProtocolError("protocol_profiles do not cover the registered seven profiles")
    for profile_id, profile in protocol_map.items():
        _validate_source_level(profile.get("source_level"), f"protocol_profiles[{profile_id}].source_level")
        kind = _nonempty_string(profile.get("kind"), f"protocol_profiles[{profile_id}].kind")
        if kind in {"ramp", "smooth_ramp"}:
            _range(profile.get("ramp_duration_range_s"), f"protocol_profiles[{profile_id}].ramp_duration_range_s", lower=0.0)
        if kind == "shifted_onset":
            _range(profile.get("onset_range_s"), f"protocol_profiles[{profile_id}].onset_range_s", lower=0.0)
        if kind in {"short_pulse", "incomplete_recovery"}:
            duration = _range(profile.get("exposure_duration_range_s"), f"protocol_profiles[{profile_id}].exposure_duration_range_s", lower=0.0)
            if kind == "short_pulse" and duration[0] < 60.0:
                raise A2DynamicProtocolError("SHORT_PULSE effective exposure must be at least 60 s")
        if kind == "multi_pulse":
            pulse_range = _range(profile.get("pulse_count_range"), f"protocol_profiles[{profile_id}].pulse_count_range", lower=1.0)
            if pulse_range[0] < 2.0 or pulse_range[1] > 3.0:
                raise A2DynamicProtocolError("MULTI_PULSE pulse_count_range must be [2,3]")
            width = _range(profile.get("pulse_width_range_s"), f"protocol_profiles[{profile_id}].pulse_width_range_s", lower=0.0)
            period = _range(profile.get("pulse_period_range_s"), f"protocol_profiles[{profile_id}].pulse_period_range_s", lower=0.0)
            if width[1] > period[0]:
                raise A2DynamicProtocolError("MULTI_PULSE pulse width must not exceed the minimum period")
            if profile.get("target_switches_within_observation") != 0:
                raise A2DynamicProtocolError("MULTI_PULSE may not switch target composition")

    transport = _mapping(config.get("transport"), "transport")
    if transport.get("state_units") != "mol%" or transport.get("update_rule") != "analytic_exponential":
        raise A2DynamicProtocolError("transport must use mol% analytic_exponential states")
    if transport.get("shared_chamber_model") != "well_mixed_cstr" or transport.get("local_sensor_model") != "first_order_transport_only":
        raise A2DynamicProtocolError("transport model layering is not frozen")
    if transport.get("tau_mix_distribution") != "log_uniform" or transport.get("nonzero_transport_distribution") != "log_uniform":
        raise A2DynamicProtocolError("transport ranges must use the registered log-uniform distributions")
    zero_probability = _finite(transport.get("zero_transport_probability"), "transport.zero_transport_probability")
    if not 0.0 < zero_probability < 1.0:
        raise A2DynamicProtocolError("transport.zero_transport_probability must lie in (0,1)")
    _positive(transport.get("minimum_nonzero_transport_s"), "transport.minimum_nonzero_transport_s")
    profiles = _sequence(transport.get("profiles"), "transport.profiles")
    transport_map: dict[str, Mapping[str, Any]] = {}
    for raw in profiles:
        profile = _mapping(raw, "transport.profiles[]")
        profile_id = _nonempty_string(profile.get("transport_profile_id"), "transport_profile_id")
        if profile_id in transport_map:
            raise A2DynamicProtocolError(f"duplicate transport_profile_id {profile_id!r}")
        transport_map[profile_id] = profile
    if set(transport_map) != set(EXPECTED_TRANSPORT):
        raise A2DynamicProtocolError("transport profiles do not cover train, val, stress_val, and test ranges")
    for profile_id, expected in EXPECTED_TRANSPORT.items():
        profile = transport_map[profile_id]
        _validate_source_level(profile.get("source_level"), f"transport.profiles[{profile_id}].source_level")
        _validate_range_map(profile, f"transport.profiles[{profile_id}]", expected)

    hardware = _mapping(config.get("hardware_profiles"), "hardware_profiles")
    ultrasonic = _mapping(hardware.get("ultrasonic"), "hardware_profiles.ultrasonic")
    if ultrasonic.get("geometry") != "transverse_single_path":
        raise A2DynamicProtocolError("ultrasonic geometry must be transverse_single_path")
    if ultrasonic.get("flow_coupling") != "excluded_from_v1":
        raise A2DynamicProtocolError("v1 ultrasonic flow coupling must be excluded")
    candidates = _sequence(ultrasonic.get("candidates"), "ultrasonic.candidates")
    candidate_ids: set[str] = set()
    expected_excitation = {"bandlimited_burst", "linear_chirp"}
    expected_estimators = {"reference_xcorr", "reference_xcorr_parabolic"}
    multipath_ids = {
        _nonempty_string(_mapping(raw, "ultrasonic.multipath_profiles[]").get("multipath_profile_id"), "multipath_profile_id")
        for raw in _sequence(ultrasonic.get("multipath_profiles"), "ultrasonic.multipath_profiles")
    }
    if len(candidates) != 2:
        raise A2DynamicProtocolError("ultrasonic must freeze exactly two pilot candidates")
    for raw in candidates:
        candidate = _mapping(raw, "ultrasonic.candidates[]")
        profile_id = _nonempty_string(candidate.get("ultrasonic_profile_id"), "ultrasonic_profile_id")
        if profile_id in candidate_ids:
            raise A2DynamicProtocolError(f"duplicate ultrasonic_profile_id {profile_id!r}")
        candidate_ids.add(profile_id)
        _validate_profile_metadata(candidate, REQUIRED_ULTRASONIC_FIELDS, f"ultrasonic.{profile_id}")
        if candidate.get("excitation_type") not in expected_excitation:
            raise A2DynamicProtocolError(f"ultrasonic.{profile_id} excitation_type is not registered")
        if candidate.get("tof_estimator") not in expected_estimators:
            raise A2DynamicProtocolError(f"ultrasonic.{profile_id} tof_estimator is not registered")
        _positive(candidate.get("path_length_m"), f"ultrasonic.{profile_id}.path_length_m")
        _positive(candidate.get("center_frequency_hz"), f"ultrasonic.{profile_id}.center_frequency_hz")
        _positive(candidate.get("adc_rate_hz"), f"ultrasonic.{profile_id}.adc_rate_hz")
        if _finite(candidate.get("adc_rate_hz"), f"ultrasonic.{profile_id}.adc_rate_hz") <= 2.0 * _finite(candidate.get("center_frequency_hz"), f"ultrasonic.{profile_id}.center_frequency_hz"):
            raise A2DynamicProtocolError(f"ultrasonic.{profile_id}.adc_rate_hz must resolve the carrier")
        fractional_bandwidth = _finite(candidate.get("fractional_bandwidth"), f"ultrasonic.{profile_id}.fractional_bandwidth")
        if not 0.0 < fractional_bandwidth < 1.0:
            raise A2DynamicProtocolError(f"ultrasonic.{profile_id}.fractional_bandwidth must be in (0,1)")
        _integer(candidate.get("average_count"), f"ultrasonic.{profile_id}.average_count", minimum=1)
        _positive(candidate.get("pulse_repetition_hz"), f"ultrasonic.{profile_id}.pulse_repetition_hz")
        _positive(candidate.get("window_duration_s"), f"ultrasonic.{profile_id}.window_duration_s")
        attenuation = _sequence(candidate.get("attenuation_nepers_per_m_by_component"), f"ultrasonic.{profile_id}.attenuation_nepers_per_m_by_component")
        if len(attenuation) != 3 or any(_finite(value, f"ultrasonic.{profile_id}.attenuation[{index}]") < 0.0 for index, value in enumerate(attenuation)):
            raise A2DynamicProtocolError(f"ultrasonic.{profile_id} attenuation must contain three non-negative values")
        if candidate.get("multipath_profile_id") not in multipath_ids:
            raise A2DynamicProtocolError(f"ultrasonic.{profile_id} references an unknown multipath profile")
    if candidate_ids != {"US-BURST-XCORR-1", "US-CHIRP-XCORR-PARABOLIC-1"}:
        raise A2DynamicProtocolError("ultrasonic candidate IDs are not frozen")
    selected_profile_id = ultrasonic.get("selected_profile_id")
    if selected_profile_id not in candidate_ids:
        raise A2DynamicProtocolError("ultrasonic selected_profile_id must reference a registered candidate")
    selected_candidate = next(item for item in candidates if item.get("ultrasonic_profile_id") == selected_profile_id)
    if ultrasonic.get("selection_status") != "PILOT_QUALIFIED":
        raise A2DynamicProtocolError("ultrasonic selection_status must be PILOT_QUALIFIED")
    if ultrasonic.get("selected_tof_estimator") != selected_candidate.get("tof_estimator") or ultrasonic.get("selected_excitation_type") != selected_candidate.get("excitation_type"):
        raise A2DynamicProtocolError("ultrasonic selected estimator or excitation does not match selected_profile_id")
    _nonempty_string(ultrasonic.get("selection_manifest"), "ultrasonic.selection_manifest")
    multipath_profiles = _sequence(ultrasonic.get("multipath_profiles"), "ultrasonic.multipath_profiles")
    if len(multipath_profiles) != len(multipath_ids) or multipath_ids != {"US-MP-NOMINAL", "US-MP-OOD"}:
        raise A2DynamicProtocolError("ultrasonic multipath profiles are incomplete")
    for raw in multipath_profiles:
        profile = _mapping(raw, "ultrasonic.multipath_profiles[]")
        profile_id = profile["multipath_profile_id"]
        _validate_source_level(profile.get("source_level"), f"ultrasonic.multipath_profiles[{profile_id}].source_level")
        for component in _sequence(profile.get("components"), f"ultrasonic.multipath_profiles[{profile_id}].components"):
            component_map = _mapping(component, "ultrasonic multipath component")
            amplitude = _finite(component_map.get("relative_amplitude"), "multipath.relative_amplitude")
            delay = _finite(component_map.get("delay_s"), "multipath.delay_s")
            if amplitude < 0.0 or delay <= 0.0:
                raise A2DynamicProtocolError("multipath amplitude and delay must be non-negative and positive respectively")

    thermal = _mapping(hardware.get("thermal"), "hardware_profiles.thermal")
    thermal_profiles = _sequence(thermal.get("profiles"), "thermal.profiles")
    if len(thermal_profiles) != 1:
        raise A2DynamicProtocolError("thermal hardware must contain exactly one frozen profile")
    thermal_profile = _mapping(thermal_profiles[0], "thermal.profiles[0]")
    if thermal_profile.get("thermal_profile_id") != "TCD-LUMPED-SYNTH-1":
        raise A2DynamicProtocolError("thermal profile ID is not frozen")
    _validate_profile_metadata(thermal_profile, REQUIRED_THERMAL_FIELDS, "thermal.TCD-LUMPED-SYNTH-1")
    for field in ("heater_heat_capacity", "gas_conductance_scale", "substrate_conductance", "heater_power", "tcr", "bridge_voltage", "heater_resistance_ohm", "reference_temperature_k"):
        _positive(thermal_profile.get(field), f"thermal.TCD-LUMPED-SYNTH-1.{field}")
    if _finite(thermal_profile.get("flow_coupling"), "thermal.flow_coupling") < 0.0:
        raise A2DynamicProtocolError("thermal flow_coupling must be non-negative")
    if thermal.get("formula_id") != "tcd_lumped_energy_balance_v1":
        raise A2DynamicProtocolError("thermal formula_id is not frozen")

    ndir = _mapping(hardware.get("ndir"), "hardware_profiles.ndir")
    ndir_profiles = _sequence(ndir.get("profiles"), "ndir.profiles")
    if len(ndir_profiles) != 1:
        raise A2DynamicProtocolError("NDIR hardware must contain exactly one frozen profile")
    ndir_profile = _mapping(ndir_profiles[0], "ndir.profiles[0]")
    if ndir_profile.get("ndir_profile_id") != "NDIR-HIGHRANGE-SHORTPATH-1":
        raise A2DynamicProtocolError("NDIR profile ID is not frozen")
    _validate_profile_metadata(ndir_profile, REQUIRED_NDIR_FIELDS, "ndir.NDIR-HIGHRANGE-SHORTPATH-1")
    _positive(ndir_profile.get("optical_path_m"), "ndir.optical_path_m")
    _positive(ndir_profile.get("tau_emitter_detector_s"), "ndir.tau_emitter_detector_s")
    _close(ndir_profile.get("range_min_mol_pct"), 0.0, "ndir.range_min_mol_pct")
    _close(ndir_profile.get("range_max_mol_pct"), 100.0, "ndir.range_max_mol_pct")
    if ndir_profile.get("effective_absorption_model_id") != "HITRAN2020-BANDINTEGRATED-GAUSSIAN-1":
        raise A2DynamicProtocolError("NDIR must use the registered HITRAN band-integrated model")
    if ndir_profile.get("hitran_table_name") != "CO2_2250p0000_2445p0000":
        raise A2DynamicProtocolError("NDIR HITRAN table is not registered")
    wavenumber_range = _range(ndir_profile.get("wavenumber_range_cm1"), "ndir.wavenumber_range_cm1", lower=0.0)
    for field in ("active_center_wavenumber_cm1", "reference_center_wavenumber_cm1"):
        center = _finite(ndir_profile.get(field), f"ndir.{field}")
        if not wavenumber_range[0] <= center <= wavenumber_range[1]:
            raise A2DynamicProtocolError(f"ndir.{field} must lie inside wavenumber_range_cm1")
    for field in ("active_fwhm_cm1", "reference_fwhm_cm1", "wavenumber_step_cm1"):
        _positive(ndir_profile.get(field), f"ndir.{field}")
    if ndir.get("range_switching") != "forbidden_in_v1" or ndir.get("active_reference_chain") != "active_reference_ratio":
        raise A2DynamicProtocolError("NDIR range and active/reference semantics are not frozen")

    environments = _sequence(config.get("environment_profiles"), "environment_profiles")
    environment_ids: set[str] = set()
    for raw in environments:
        profile = _mapping(raw, "environment_profiles[]")
        profile_id = _nonempty_string(profile.get("environment_id"), "environment_id")
        if profile_id in environment_ids:
            raise A2DynamicProtocolError(f"duplicate environment_id {profile_id!r}")
        environment_ids.add(profile_id)
        if profile_id not in EXPECTED_ENVIRONMENTS:
            raise A2DynamicProtocolError(f"unknown environment_id {profile_id!r}")
        temperature, pressure = EXPECTED_ENVIRONMENTS[profile_id]
        _close(profile.get("temperature_k"), temperature, f"environment_profiles[{profile_id}].temperature_k")
        _close(profile.get("pressure_pa"), pressure, f"environment_profiles[{profile_id}].pressure_pa")
    if environment_ids != set(EXPECTED_ENVIRONMENTS):
        raise A2DynamicProtocolError("environment_profiles are incomplete")

    calibration_profiles = _sequence(config.get("calibration_profiles"), "calibration_profiles")
    calibration_ids = set()
    for raw in calibration_profiles:
        profile = _mapping(raw, "calibration_profiles[]")
        profile_id = _nonempty_string(profile.get("calibration_profile_id"), "calibration_profile_id")
        if profile_id in calibration_ids:
            raise A2DynamicProtocolError(f"duplicate calibration_profile_id {profile_id!r}")
        calibration_ids.add(profile_id)
        if profile.get("source_profile_id") != profile_id or profile.get("source_registry") != "a2h":
            raise A2DynamicProtocolError(f"calibration profile {profile_id!r} must reference the A2H profile with the same ID")
        if profile.get("dynamic_addition") != "sequence_drift_is_selected_by_noise_profile":
            raise A2DynamicProtocolError(f"calibration profile {profile_id!r} has an unregistered dynamic addition")
    if calibration_ids != {"CAL-NOMINAL", "CAL-LIGHT", "CAL-SHARED-DRIFT", "CAL-CONFLICT"}:
        raise A2DynamicProtocolError("calibration_profiles are incomplete")

    noise_profiles = _sequence(config.get("noise_profiles"), "noise_profiles")
    noise_ids: set[str] = set()
    for raw in noise_profiles:
        profile = _mapping(raw, "noise_profiles[]")
        profile_id = _nonempty_string(profile.get("noise_profile_id"), "noise_profile_id")
        if profile_id in noise_ids:
            raise A2DynamicProtocolError(f"duplicate noise_profile_id {profile_id!r}")
        noise_ids.add(profile_id)
        if profile_id not in EXPECTED_NOISE:
            raise A2DynamicProtocolError(f"unknown noise_profile_id {profile_id!r}")
        white, rho, correlated, drift = EXPECTED_NOISE[profile_id]
        _close(profile.get("white_noise_scale"), white, f"noise_profiles[{profile_id}].white_noise_scale")
        for field, expected in (("ar1_rho_range", rho), ("shared_correlation_load_range", correlated), ("drift_strength_range_pct_dynamic_range_per_min", drift)):
            actual = _range(profile.get(field), f"noise_profiles[{profile_id}].{field}", lower=0.0)
            if actual != expected:
                raise A2DynamicProtocolError(f"noise_profiles[{profile_id}].{field} must be {expected!r}")
        _integer(profile.get("independent_repeat_count"), f"noise_profiles[{profile_id}].independent_repeat_count", minimum=1)
        if profile.get("independent_repeat_count") != 3:
            raise A2DynamicProtocolError("all registered noise profiles must reserve three independent repeats")
        if profile_id == "NOISE-CORR-5X":
            _exact_list(profile.get("correlation_vector"), [0.50, 0.25, 0.75], "NOISE-CORR-5X.correlation_vector")
    if noise_ids != set(EXPECTED_NOISE):
        raise A2DynamicProtocolError("noise_profiles are incomplete")

    composition = _mapping(config.get("composition_distribution"), "composition_distribution")
    if composition.get("sampling_strategy") != "stratified_low_discrepancy_simplex":
        raise A2DynamicProtocolError("composition sampling strategy is not frozen")
    _exact_list(composition.get("coordinate_order"), list(TARGET_NAMES), "composition.coordinate_order")
    _close(composition.get("non_pure_quantization_pct"), 0.01, "composition.non_pure_quantization_pct")
    if composition.get("non_pure_coordinate_must_be_unique") is not True:
        raise A2DynamicProtocolError("non_pure_coordinate_must_be_unique must be true")
    regions = _mapping(composition.get("regions"), "composition.regions")
    if set(regions) != {"interior", "near_boundary", "binary", "pure"}:
        raise A2DynamicProtocolError("composition regions are incomplete")
    for region_id, expected_proportion in (("interior", 0.50), ("near_boundary", 0.30), ("binary", 0.20)):
        region = _mapping(regions[region_id], f"composition.regions.{region_id}")
        _close(region.get("proportion"), expected_proportion, f"composition.regions.{region_id}.proportion")
    pure_region = _mapping(regions["pure"], "composition.regions.pure")
    if pure_region.get("proportion") is not None or pure_region.get("reserved_for") != "D-JOINT/test" or pure_region.get("canonical_group_count") != 3:
        raise A2DynamicProtocolError("pure composition policy is not frozen")
    pure_vertices = _sequence(composition.get("pure_vertices"), "composition.pure_vertices")
    if len(pure_vertices) != 3:
        raise A2DynamicProtocolError("exactly three canonical pure vertices are required")
    pure_ids: set[str] = set()
    for raw in pure_vertices:
        vertex = _mapping(raw, "composition.pure_vertices[]")
        vertex_id = _nonempty_string(vertex.get("mixture_id"), "composition.pure_vertices[].mixture_id")
        if vertex_id in pure_ids:
            raise A2DynamicProtocolError(f"duplicate pure vertex mixture_id {vertex_id!r}")
        pure_ids.add(vertex_id)
        values = [_finite(value, f"{vertex_id}.composition_pct") for value in _sequence(vertex.get("composition_pct"), f"{vertex_id}.composition_pct")]
        if len(values) != 3 or sum(values) != 100.0 or sum(value > 0.0 for value in values) != 1:
            raise A2DynamicProtocolError(f"pure vertex {vertex_id!r} is not a canonical simplex vertex")
    quotas = _mapping(composition.get("region_quota_by_split"), "composition.region_quota_by_split")
    if set(quotas) != set(SPLITS):
        raise A2DynamicProtocolError("composition quotas must cover all four splits")
    for split in SPLITS:
        quota = _mapping(quotas[split], f"composition.region_quota_by_split.{split}")
        if dict(quota) != EXPECTED_REGION_QUOTAS[split]:
            raise A2DynamicProtocolError(f"composition quota for {split} is not frozen")

    families = _mapping(config.get("families"), "families")
    if set(families) != set(FAMILIES):
        raise A2DynamicProtocolError(f"families must be exactly {list(FAMILIES)!r}")
    protocol_ids = set(protocol_map)
    transport_ids = set(transport_map)
    for family_id in FAMILIES:
        family = _mapping(families[family_id], f"families.{family_id}")
        groups = _mapping(family.get("groups_by_split"), f"families.{family_id}.groups_by_split")
        repeats = EXPECTED_FAMILY_REPEATS[family_id]
        if dict(groups) != EXPECTED_FAMILY_GROUPS[family_id]:
            raise A2DynamicProtocolError(f"groups_by_split for {family_id} is not frozen")
        if family.get("repeat_count") != repeats:
            raise A2DynamicProtocolError(f"repeat_count for {family_id} must be {repeats}")
        observations = _mapping(family.get("observation_rows_by_split"), f"families.{family_id}.observation_rows_by_split")
        expected_observations = {split: EXPECTED_FAMILY_GROUPS[family_id][split] * repeats for split in SPLITS}
        if dict(observations) != expected_observations:
            raise A2DynamicProtocolError(f"observation_rows_by_split for {family_id} is inconsistent with group counts and repeat_count")
        for mapping_name, ids, allow_lists in (
            ("protocol_by_split", protocol_ids, True),
            ("transport_by_split", transport_ids, False),
            ("environment_by_split", environment_ids, True),
            ("calibration_by_split", calibration_ids, False),
            ("noise_by_split", noise_ids, False),
            ("composition_mode_by_split", {"matched_global", "D-JOINT_test_includes_pure_vertices"}, False),
        ):
            mapping = _mapping(family.get(mapping_name), f"families.{family_id}.{mapping_name}")
            if set(mapping) != set(SPLITS):
                raise A2DynamicProtocolError(f"families.{family_id}.{mapping_name} must cover all splits")
            for split, value in mapping.items():
                values = list(value) if allow_lists and isinstance(value, list) else [value]
                if not values or any(item not in ids for item in values):
                    raise A2DynamicProtocolError(f"families.{family_id}.{mapping_name}.{split} references an unknown profile")
        _validate_source_level(family.get("source_level"), f"families.{family_id}.source_level")
    _validate_family_totals(families)

    split_contract = _mapping(config.get("split_contract"), "split_contract")
    _exact_list(split_contract.get("split_order"), SPLITS, "split_contract.split_order")
    for key, expected in (("group_key", "mixture_id"), ("observation_key", "observation_id"), ("bootstrap_unit", "mixture_id"), ("mixture_id_namespace", "a2dyn-mix-"), ("observation_id_namespace", "a2dyn-obs-"), ("test_access", "locked_until_data_frozen")):
        if split_contract.get(key) != expected:
            raise A2DynamicProtocolError(f"split_contract.{key} is not frozen")
    if split_contract.get("group_must_be_single_split") is not True or split_contract.get("observation_id_global_unique") is not True:
        raise A2DynamicProtocolError("split group and observation uniqueness rules must be true")
    totals = _mapping(split_contract.get("totals"), "split_contract.totals")
    if {split: dict(totals.get(split, {})) for split in SPLITS} != EXPECTED_SPLIT_TOTALS:
        raise A2DynamicProtocolError("split_contract.totals are not frozen")
    if dict(_mapping(split_contract.get("overall"), "split_contract.overall")) != {"groups": 4410, "observations": 6300}:
        raise A2DynamicProtocolError("split_contract.overall must be 4410 groups and 6300 observations")

    record_schema = _mapping(config.get("record_schema"), "record_schema")
    if record_schema.get("schema_version") != "gf-a2-dynamic-record-1":
        raise A2DynamicProtocolError("record schema_version is not frozen")
    required_record_fields = set(_sequence(record_schema.get("required_fields"), "record_schema.required_fields"))
    if {"mixture_id", "observation_id", "split", "family", "protocol_profile_id", "transport_profile_id", "ultrasonic_profile_id", "thermal_profile_id", "ndir_profile_id"} - required_record_fields:
        raise A2DynamicProtocolError("record_schema.required_fields misses identity or profile fields")
    _exact_list(record_schema.get("forbidden_fields"), ["sequence_id", "base_condition_id", "noise_seed_index", "noise_seed"], "record_schema.forbidden_fields")
    if set(record_schema.get("status_values", [])) != {"generated", "audited", "rejected"}:
        raise A2DynamicProtocolError("record_schema.status_values are not frozen")
    model_forbidden = set(_sequence(record_schema.get("model_input_forbidden_fields"), "record_schema.model_input_forbidden_fields"))
    if {"target", "phase_id", "clean_device_signals", "chamber_composition", "privileged_parameters", "device_states"} - model_forbidden:
        raise A2DynamicProtocolError("model input must exclude target and privileged dynamics")

    storage = _mapping(config.get("storage"), "storage")
    expected_storage = {
        "data_dir": "data/a2_dynamic_v1",
        "config_snapshot": "data/a2_dynamic_v1/config_snapshot.json",
        "manifest": "data/a2_dynamic_v1/manifest.json",
        "records": "data/a2_dynamic_v1/records.jsonl",
        "observations": "data/a2_dynamic_v1/observations.npz",
        "oracle": "data/a2_dynamic_v1/oracle.npz",
        "device_audit": "data/a2_dynamic_v1/device_audit.npz",
        "waveform_fixtures": "data/a2_dynamic_v1/waveform_fixtures.npz",
        "audit": "data/a2_dynamic_v1/audit.json",
    }
    for key, expected in expected_storage.items():
        if storage.get(key) != expected:
            raise A2DynamicProtocolError(f"storage.{key} must be {expected}")

    array_contract = _mapping(config.get("array_contract"), "array_contract")
    expected_arrays = {
        "signals": ("float32", ["N", 3, 1200, 1]),
        "valid_mask": ("bool", ["N", 3, 1200, 1]),
        "quality": ("float32", ["N", 3, 1200]),
        "time_s": ("float64", [1200]),
        "target": ("float32", ["N", 3]),
        "phase_id": ("int8", ["N", 1200]),
        "observation_index": ("int64", ["N"]),
    }
    if set(array_contract) != set(expected_arrays):
        raise A2DynamicProtocolError("array_contract must cover exactly the registered arrays")
    for name, (dtype, shape) in expected_arrays.items():
        definition = _mapping(array_contract[name], f"array_contract.{name}")
        if definition.get("dtype") != dtype or list(definition.get("shape", [])) != shape:
            raise A2DynamicProtocolError(f"array_contract.{name} dtype or shape is not frozen")
    if _mapping(array_contract["quality"], "array_contract.quality").get("fixed_value") != 1.0:
        raise A2DynamicProtocolError("quality must be fixed at 1.0 in v1")

    physics_reference = _mapping(config.get("physics_reference"), "physics_reference")
    eos = _mapping(physics_reference.get("eos"), "physics_reference.eos")
    if eos.get("backend") != "HEOS" or eos.get("package_name") != "CoolProp" or eos.get("package_version") != "8.0.0":
        raise A2DynamicProtocolError("CoolProp HEOS generator backend/version is not frozen")
    if dict(_mapping(eos.get("fluid_name_map"), "physics_reference.eos.fluid_name_map")) != {"Ar": "Argon", "He": "Helium", "CO2": "CarbonDioxide"}:
        raise A2DynamicProtocolError("CoolProp fluid name map is not frozen")
    if eos.get("sound_speed_model_id") != "a2dyn_direct_multifluid_eos_v1":
        raise A2DynamicProtocolError("A2-DYN sound speed model is not registered")
    if eos.get("definition") != "CoolProp.AbstractState(HEOS).speed_sound":
        raise A2DynamicProtocolError("A2-DYN sound speed generator definition is not frozen")
    if eos.get("generator_role") != "primary_definition" or eos.get("verification_scope") != "generator_consistency":
        raise A2DynamicProtocolError("A2-DYN HEOS generator role or verification scope is not frozen")
    if eos.get("independent_physics_validation") != "NOT_CLAIMED":
        raise A2DynamicProtocolError("A2-DYN must not claim independent validation against its own generator")
    model_asset = _mapping(eos.get("model_asset"), "physics_reference.eos.model_asset")
    _nonempty_string(model_asset.get("path"), "physics_reference.eos.model_asset.path")
    _hash(model_asset.get("sha256"), "physics_reference.eos.model_asset.sha256")
    _exact_list(eos.get("registered_temperature_range_k"), [278.15, 313.15], "physics_reference.eos.registered_temperature_range_k")
    _exact_list(eos.get("registered_pressure_range_pa"), [90000.0, 112000.0], "physics_reference.eos.registered_pressure_range_pa")
    if eos.get("shared_physics_query_path") != "src/gf/sim/a2dyn_sound_speed.py":
        raise A2DynamicProtocolError("A2-DYN shared sound-speed query path is not frozen")
    _hash(eos.get("shared_physics_query_sha256"), "physics_reference.eos.shared_physics_query_sha256")
    eos_grid = _mapping(eos.get("query_grid"), "physics_reference.eos.query_grid")
    _close(eos_grid.get("simplex_step_pct"), 1.0, "physics_reference.eos.query_grid.simplex_step_pct")
    if eos_grid.get("include_pure_vertices") is not True:
        raise A2DynamicProtocolError("EOS query grid must include pure vertices")
    _exact_list(eos_grid.get("temperature_values_k"), [278.15, 293.15, 298.15, 303.15, 308.15, 313.15], "physics_reference.eos.query_grid.temperature_values_k")
    _exact_list(eos_grid.get("pressure_values_pa"), [90000.0, 98000.0, 101325.0, 105000.0, 108000.0, 112000.0], "physics_reference.eos.query_grid.pressure_values_pa")
    _close(eos_grid.get("composition_sum_tolerance_pct"), 1.0e-9, "physics_reference.eos.query_grid.composition_sum_tolerance_pct")
    eos_gate = _mapping(eos.get("error_gate"), "physics_reference.eos.error_gate")
    _close(eos_gate.get("max_relative_error"), 0.0, "physics_reference.eos.error_gate.max_relative_error")
    _exact_list(eos_gate.get("report_percentiles"), [50.0, 95.0, 100.0], "physics_reference.eos.error_gate.report_percentiles")
    if eos_gate.get("failure_action") != "reject_generator_or_runtime":
        raise A2DynamicProtocolError("A2-DYN generator consistency failure action is not frozen")
    off_grid = _mapping(eos.get("off_grid_audit"), "physics_reference.eos.off_grid_audit")
    _integer(off_grid.get("count"), "physics_reference.eos.off_grid_audit.count", minimum=10000)
    if off_grid.get("count") != 10000 or off_grid.get("seed") != 20260831:
        raise A2DynamicProtocolError("A2-DYN off-grid audit count or seed is not frozen")
    if off_grid.get("construction") != "nested-radical-inverse-simplex":
        raise A2DynamicProtocolError("A2-DYN off-grid construction is not frozen")

    ndir_reference = _mapping(physics_reference.get("ndir_reference"), "physics_reference.ndir_reference")
    if ndir_reference.get("database") != "HITRAN2020" or ndir_reference.get("backend") != "hitran_hapi_v1" or ndir_reference.get("package_name") != "hitran-api" or ndir_reference.get("package_version") != "1.3.0.0":
        raise A2DynamicProtocolError("HITRAN2020 reference backend/version is not frozen")
    if ndir_reference.get("reference_asset_id") != ndir_profile.get("reference_asset_id"):
        raise A2DynamicProtocolError("NDIR profile and HITRAN asset IDs do not match")
    band_grid = _mapping(ndir_reference.get("bandpass_grid_cm1"), "physics_reference.ndir_reference.bandpass_grid_cm1")
    for key, expected in (("minimum", 2250.0), ("maximum", 2445.0), ("step", 0.1)):
        _close(band_grid.get(key), expected, f"physics_reference.ndir_reference.bandpass_grid_cm1.{key}")
    asset_files = _sequence(ndir_reference.get("asset_files"), "physics_reference.ndir_reference.asset_files")
    if len(asset_files) != 2:
        raise A2DynamicProtocolError("HITRAN2020 asset registration must include data and header hashes")
    for index, raw in enumerate(asset_files):
        asset = _mapping(raw, f"physics_reference.ndir_reference.asset_files[{index}]")
        _nonempty_string(asset.get("path"), f"HITRAN asset path {index}")
        _hash(asset.get("sha256"), f"HITRAN asset hash {index}")
    ndir_gate = _mapping(ndir_reference.get("error_gate"), "physics_reference.ndir_reference.error_gate")
    _close(ndir_gate.get("max_relative_error"), 0.01, "physics_reference.ndir_reference.error_gate.max_relative_error")
    _exact_list(ndir_gate.get("report_percentiles"), [50.0, 95.0, 100.0], "physics_reference.ndir_reference.error_gate.report_percentiles")

    signal_bounds = _mapping(config.get("signal_bounds"), "signal_bounds")
    for sensor_id, expected in EXPECTED_SIGNAL_BOUNDS.items():
        actual = _range(signal_bounds.get(sensor_id), f"signal_bounds.{sensor_id}", lower=-math.inf)
        if actual != expected:
            raise A2DynamicProtocolError(f"signal_bounds.{sensor_id} must be {expected!r}")
    if signal_bounds.get("source_registry") != "A2H_frozen_signal_bounds":
        raise A2DynamicProtocolError("signal_bounds source registry is not frozen")

    source_registry = _mapping(config.get("source_registry"), "source_registry")
    for source_id in ("a1", "a2h"):
        source = _mapping(source_registry.get(source_id), f"source_registry.{source_id}")
        for key in ("config_path", "manifest_path", "data_version", "content_sha256"):
            _nonempty_string(source.get(key), f"source_registry.{source_id}.{key}")
        _hash(source.get("config_sha256"), f"source_registry.{source_id}.config_sha256")
        _hash(source.get("content_sha256"), f"source_registry.{source_id}.content_sha256")
        if source_id == "a1":
            _hash(source.get("split_hash"), "source_registry.a1.split_hash")
        else:
            _hash(source.get("profile_hash"), "source_registry.a2h.profile_hash")
    shared_physics = _mapping(source_registry.get("shared_physics"), "source_registry.shared_physics")
    _nonempty_string(shared_physics.get("path"), "source_registry.shared_physics.path")
    _hash(shared_physics.get("sha256"), "source_registry.shared_physics.sha256")
    if shared_physics.get("sha256") != eos.get("shared_physics_query_sha256"):
        raise A2DynamicProtocolError("shared physics hash is duplicated inconsistently")
    if list(config.get("allowed_source_levels", [])) != list(SOURCE_LEVELS):
        raise A2DynamicProtocolError("allowed_source_levels is not the registered taxonomy")


def validate_a2_dynamic_records(
    records: Sequence[Mapping[str, Any]],
    data_config: Mapping[str, Any],
    *,
    require_frozen_counts: bool = False,
) -> None:
    """校验 records 的身份、单位、分组互斥和 profile 引用。

    该校验不读取信号数组；它只负责生成阶段可以确定的轻量数据契约。
    """

    validate_a2_dynamic_data_config(data_config)
    if not isinstance(records, (list, tuple)):
        raise A2DynamicProtocolError("records must be an array of objects")
    record_schema = _mapping(data_config["record_schema"], "record_schema")
    required_fields = set(_sequence(record_schema["required_fields"], "record_schema.required_fields"))
    allowed_statuses = set(_sequence(record_schema["status_values"], "record_schema.status_values"))
    protocol_ids = {
        _mapping(item, "protocol_profiles[]")["protocol_profile_id"]
        for item in _sequence(data_config["protocol_profiles"], "protocol_profiles")
    }
    transport_ids = {
        _mapping(item, "transport.profiles[]")["transport_profile_id"]
        for item in _sequence(_mapping(data_config["transport"], "transport")["profiles"], "transport.profiles")
    }
    environment_ids = {
        _mapping(item, "environment_profiles[]")["environment_id"]
        for item in _sequence(data_config["environment_profiles"], "environment_profiles")
    }
    calibration_ids = {
        _mapping(item, "calibration_profiles[]")["calibration_profile_id"]
        for item in _sequence(data_config["calibration_profiles"], "calibration_profiles")
    }
    noise_ids = {
        _mapping(item, "noise_profiles[]")["noise_profile_id"]
        for item in _sequence(data_config["noise_profiles"], "noise_profiles")
    }
    hardware = _mapping(data_config["hardware_profiles"], "hardware_profiles")
    ultrasonic_ids = {
        _mapping(item, "ultrasonic.candidates[]")["ultrasonic_profile_id"]
        for item in _sequence(_mapping(hardware["ultrasonic"], "hardware_profiles.ultrasonic")["candidates"], "ultrasonic.candidates")
    }
    thermal_ids = {
        _mapping(item, "thermal.profiles[]")["thermal_profile_id"]
        for item in _sequence(_mapping(hardware["thermal"], "hardware_profiles.thermal")["profiles"], "thermal.profiles")
    }
    ndir_ids = {
        _mapping(item, "ndir.profiles[]")["ndir_profile_id"]
        for item in _sequence(_mapping(hardware["ndir"], "hardware_profiles.ndir")["profiles"], "ndir.profiles")
    }
    families = _mapping(data_config["families"], "families")
    seen_observations: set[str] = set()
    group_state: dict[str, tuple[str, tuple[float, float, float]]] = {}
    composition_groups: dict[tuple[float, float, float], str] = {}
    pure_vertices: dict[tuple[float, float, float], str] = {}
    composition_config = _mapping(data_config["composition_distribution"], "composition_distribution")
    for raw_vertex in _sequence(composition_config["pure_vertices"], "pure_vertices"):
        vertex = _mapping(raw_vertex, "pure_vertices[]")
        values = tuple(float(value) for value in _sequence(vertex["composition_pct"], "pure_vertices[].composition_pct"))
        if len(values) != 3:
            raise A2DynamicProtocolError("pure vertex composition must contain three components")
        pure_vertices[values] = _nonempty_string(vertex["mixture_id"], "pure_vertices[].mixture_id")
    for index, raw in enumerate(records):
        record = _mapping(raw, f"records[{index}]")
        _validate_no_forbidden_keys(record, f"records[{index}]")
        missing = required_fields - set(record)
        if missing:
            raise A2DynamicProtocolError(f"records[{index}] is missing fields {sorted(missing)}")
        if record.get("schema_version") != record_schema["schema_version"]:
            raise A2DynamicProtocolError(f"records[{index}].schema_version is unsupported")
        observation_id = _nonempty_string(record.get("observation_id"), f"records[{index}].observation_id")
        mixture_id = _nonempty_string(record.get("mixture_id"), f"records[{index}].mixture_id")
        if not observation_id.startswith(str(data_config["split_contract"]["observation_id_namespace"])):
            raise A2DynamicProtocolError(f"records[{index}].observation_id uses an unregistered namespace")
        if not mixture_id.startswith(str(data_config["split_contract"]["mixture_id_namespace"])):
            raise A2DynamicProtocolError(f"records[{index}].mixture_id uses an unregistered namespace")
        if observation_id in seen_observations:
            raise A2DynamicProtocolError(f"duplicate observation_id {observation_id!r}")
        seen_observations.add(observation_id)
        split = record.get("split")
        family_id = record.get("family")
        if split not in SPLITS or family_id not in FAMILIES:
            raise A2DynamicProtocolError(f"records[{index}] has an unknown split or family")
        composition_region = record.get("composition_region")
        if composition_region not in {"interior", "near_boundary", "binary", "pure"}:
            raise A2DynamicProtocolError(f"records[{index}].composition_region is unsupported")
        composition = tuple(_finite(record.get(name), f"records[{index}].{name}") for name in TARGET_NAMES)
        if any(value < 0.0 or value > 100.0 for value in composition) or not math.isclose(sum(composition), 100.0, rel_tol=0.0, abs_tol=1.0e-6):
            raise A2DynamicProtocolError(f"records[{index}] target composition must be non-negative and sum to 100 mol%")
        previous_group = group_state.get(mixture_id)
        if previous_group is not None and (previous_group[0] != split or previous_group[1] != composition):
            raise A2DynamicProtocolError(f"mixture_id {mixture_id!r} crosses split or changes target composition")
        group_state[mixture_id] = (str(split), composition)
        quantized = tuple(round(value / 0.01) * 0.01 for value in composition)
        previous_composition_group = composition_groups.get(quantized)
        expected_pure_group = pure_vertices.get(quantized)
        if composition_region == "pure":
            if expected_pure_group != mixture_id:
                raise A2DynamicProtocolError(f"pure composition {quantized!r} must use its canonical mixture_id")
        elif previous_composition_group is not None and previous_composition_group != mixture_id:
            raise A2DynamicProtocolError(f"duplicate non-pure composition across mixture_id: {quantized!r}")
        composition_groups[quantized] = mixture_id
        _integer(record.get("timesteps"), f"records[{index}].timesteps", minimum=1)
        if record.get("timesteps") != 1200:
            raise A2DynamicProtocolError(f"records[{index}].timesteps must be 1200")
        _close(record.get("dt_s"), 0.2, f"records[{index}].dt_s")
        _nonempty_string(record.get("status"), f"records[{index}].status")
        if record.get("status") not in allowed_statuses:
            raise A2DynamicProtocolError(f"records[{index}].status is unsupported")
        onset = _finite(record.get("exposure_onset_s"), f"records[{index}].exposure_onset_s")
        exposure_end = _finite(record.get("exposure_end_s"), f"records[{index}].exposure_end_s")
        if onset < 0.0 or exposure_end <= onset or exposure_end > 240.0:
            raise A2DynamicProtocolError(f"records[{index}] exposure interval is outside the 240 s protocol")
        family = _mapping(families[family_id], f"families.{family_id}")
        _validate_record_profile_reference(record, family, split, "protocol_profile_id", "protocol_by_split", protocol_ids, index)
        _validate_record_profile_reference(record, family, split, "transport_profile_id", "transport_by_split", transport_ids, index)
        _validate_record_profile_reference(record, family, split, "environment_id", "environment_by_split", environment_ids, index)
        _validate_record_profile_reference(record, family, split, "calibration_profile_id", "calibration_by_split", calibration_ids, index)
        _validate_record_profile_reference(record, family, split, "noise_profile_id", "noise_by_split", noise_ids, index)
        if record.get("ultrasonic_profile_id") not in ultrasonic_ids:
            raise A2DynamicProtocolError(f"records[{index}] references an unknown ultrasonic profile")
        if record.get("thermal_profile_id") not in thermal_ids:
            raise A2DynamicProtocolError(f"records[{index}] references an unknown thermal profile")
        if record.get("ndir_profile_id") not in ndir_ids:
            raise A2DynamicProtocolError(f"records[{index}] references an unknown NDIR profile")
    if require_frozen_counts:
        actual_totals = {
            split: {
                "groups": len({mixture_id for mixture_id, state in group_state.items() if state[0] == split}),
                "observations": sum(1 for raw in records if _mapping(raw, "record").get("split") == split),
            }
            for split in SPLITS
        }
        if actual_totals != EXPECTED_SPLIT_TOTALS:
            raise A2DynamicProtocolError(f"frozen record totals do not match {EXPECTED_SPLIT_TOTALS!r}, got {actual_totals!r}")


def _validate_record_profile_reference(
    record: Mapping[str, Any],
    family: Mapping[str, Any],
    split: str,
    record_field: str,
    family_field: str,
    known_ids: set[str],
    index: int,
) -> None:
    value = record.get(record_field)
    if value not in known_ids:
        raise A2DynamicProtocolError(f"records[{index}].{record_field} references an unknown profile")
    allowed_raw = _mapping(family[family_field], f"family.{family_field}")[split]
    allowed = allowed_raw if isinstance(allowed_raw, list) else [allowed_raw]
    if value not in allowed:
        raise A2DynamicProtocolError(f"records[{index}].{record_field} is incompatible with its family and split")


def _validate_family_totals(families: Mapping[str, Any]) -> None:
    for split in SPLITS:
        groups = sum(int(_mapping(families[family_id], f"families.{family_id}")["groups_by_split"][split]) for family_id in FAMILIES)
        observations = sum(int(_mapping(families[family_id], f"families.{family_id}")["observation_rows_by_split"][split]) for family_id in FAMILIES)
        expected = EXPECTED_SPLIT_TOTALS[split]
        if groups != expected["groups"] or observations != expected["observations"]:
            raise A2DynamicProtocolError(f"family totals for {split} do not match the frozen split totals")


def validate_a2_dynamic_eval_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    _validate_source_levels(config)
    if config.get("schema_version") != EVAL_SCHEMA_VERSION or config.get("parent_schema_version") != "gf-eval-1":
        raise A2DynamicProtocolError("A2-DYN evaluation schema or parent schema is unsupported")
    if config.get("dataset_schema_version") != DATA_SCHEMA_VERSION or config.get("metric") != "macro_RNMAE":
        raise A2DynamicProtocolError("A2-DYN evaluation dataset schema or primary metric is not frozen")
    target_ranges = _mapping(config.get("target_ranges"), "target_ranges")
    if set(target_ranges) != set(TARGET_NAMES) or any(_finite(value, f"target_ranges.{key}") <= 0.0 for key, value in target_ranges.items()):
        raise A2DynamicProtocolError("target_ranges must cover the three positive mol% ranges")
    _exact_list(config.get("horizon_order"), [item[0] for item in EXPECTED_HORIZONS], "horizon_order")
    _exact_list(config.get("primary_realtime_horizons"), ["P015", "P030", "P060", "P120"], "primary_realtime_horizons")
    _exact_list(config.get("early_gate_horizons"), ["P015", "P030", "P060"], "early_gate_horizons")
    required_metrics = set(_sequence(config.get("required_metrics"), "required_metrics"))
    if {"macro_RNMAE", "component_MAE", "component_RMSE", "AUEC", "EarlyGain", "LatencyP95"} - required_metrics:
        raise A2DynamicProtocolError("required_metrics misses a registered metric")
    _exact_list(config.get("formal_training_seeds"), [17, 29, 43, 71, 101], "formal_training_seeds")
    if config.get("bootstrap_seed") != 20260831 or config.get("bootstrap_samples") != 2000 or config.get("confidence_level") != 0.95:
        raise A2DynamicProtocolError("bootstrap seed, count, or confidence level is not frozen")
    if config.get("bootstrap_unit") != "mixture_id":
        raise A2DynamicProtocolError("bootstrap_unit must be mixture_id")
    _exact_list(config.get("split_families"), FAMILIES, "split_families")
    _exact_list(config.get("development_splits"), ["train", "val", "stress_val"], "development_splits")
    if config.get("test_split") != "test":
        raise A2DynamicProtocolError("test_split must be test")
    test_access = _mapping(config.get("test_access"), "test_access")
    if test_access.get("default") != "locked" or test_access.get("unlock_flag") != "--unlock-test-after-freeze" or test_access.get("unlock_status") != "DATA_FROZEN":
        raise A2DynamicProtocolError("test access must remain locked until DATA_FROZEN")
    if test_access.get("result_reuse_requires_revision") is not True:
        raise A2DynamicProtocolError("test result reuse must require a protocol revision")

    baselines = _sequence(config.get("baseline_registry"), "baseline_registry")
    baseline_map: dict[str, Mapping[str, Any]] = {}
    for raw in baselines:
        baseline = _mapping(raw, "baseline_registry[]")
        model_id = _nonempty_string(baseline.get("model_id"), "baseline.model_id")
        if model_id in baseline_map:
            raise A2DynamicProtocolError(f"duplicate baseline model_id {model_id!r}")
        baseline_map[model_id] = baseline
    expected_models = {"B-LAST", "B-DELTA", "B-EWMA", "B-STAT", "B-TCN", "B-GRU", "B-STEADY", "O-EQ", "O-KIN"}
    if set(baseline_map) != expected_models:
        raise A2DynamicProtocolError("baseline_registry does not cover the frozen deployable, diagnostic, and oracle baselines")
    for model_id, baseline in baseline_map.items():
        _nonempty_string(baseline.get("kind"), f"baseline_registry.{model_id}.kind")
        _nonempty_string(baseline.get("input"), f"baseline_registry.{model_id}.input")
        _nonempty_string(baseline.get("model"), f"baseline_registry.{model_id}.model")
        if not isinstance(baseline.get("causal"), bool):
            raise A2DynamicProtocolError(f"baseline_registry.{model_id}.causal must be boolean")
    if baseline_map["B-EWMA"].get("selection_data") != "train_only":
        raise A2DynamicProtocolError("B-EWMA selection must use train only")
    if baseline_map["B-STEADY"].get("causal") is not False or baseline_map["B-STEADY"].get("allowed_horizons") != ["P150", "FULL"]:
        raise A2DynamicProtocolError("B-STEADY may only be a late diagnostic baseline")
    if any(baseline_map[model_id].get("kind") != "oracle" for model_id in ("O-EQ", "O-KIN")):
        raise A2DynamicProtocolError("O-EQ and O-KIN must remain oracle-only")

    gates = _mapping(config.get("qualification_gates"), "qualification_gates")
    difficulty = _mapping(gates.get("dynamic_difficulty"), "qualification_gates.dynamic_difficulty")
    if difficulty.get("baseline_model") != "B-LAST" or difficulty.get("late_reference_horizon") != "P150" or difficulty.get("early_horizons") != ["P015", "P030", "P060"] or difficulty.get("min_relative_degradation") != 0.25 or difficulty.get("min_horizons_passing") != 2 or difficulty.get("oracle_model") != "O-KIN" or difficulty.get("min_oracle_headroom_vs_last") != 0.20:
        raise A2DynamicProtocolError("dynamic difficulty gates are not frozen")
    temporal = _mapping(gates.get("temporal_information"), "qualification_gates.temporal_information")
    for key, expected in {
        "reference_candidates": ["B-LAST", "B-DELTA", "B-EWMA"],
        "reference_selection_split": "val",
        "reference_selection_family": "D-IID",
        "reference_freeze_before_test": True,
        "candidate_models": ["B-STAT", "B-TCN"],
        "required_horizons": ["P015", "P030", "P060"],
        "min_mean_relative_improvement": 0.10,
        "min_horizons_passing": 2,
        "min_seeds_same_direction": 4,
        "paired_group_bootstrap_ci_upper_max": 0.0,
        "max_component_rnmae_degradation": 0.005,
        "required_families": ["D-IID", "one_qualified_pressure_family"],
        "late_static_reference": "B-STEADY",
        "late_horizon": "P150",
        "max_late_relative_degradation": 0.05,
    }.items():
        if temporal.get(key) != expected:
            raise A2DynamicProtocolError(f"qualification_gates.temporal_information.{key} is not frozen")
    headroom = _mapping(gates.get("new_algorithm_headroom"), "qualification_gates.new_algorithm_headroom")
    if headroom.get("simple_model_equivalence_fraction") != 0.05 or headroom.get("min_qualified_pressure_axes") != 1 or headroom.get("min_oracle_headroom") != 0.10 or headroom.get("tcn_must_beat_statistical_baseline") is not True:
        raise A2DynamicProtocolError("new algorithm headroom gates are not frozen")
    physics_gates = _mapping(gates.get("physics_and_schema"), "qualification_gates.physics_and_schema")
    for key, expected in {
        "max_outside_signal_bound_fraction": 0.0,
        "max_inlet_sum_error_pct": 1.0e-6,
        "max_chamber_sum_error_pct": 1.0e-5,
        "min_jacobian_full_rank_fraction": 0.99,
        "max_jacobian_p95_condition_number": 1000.0,
        "max_nominal_parity_absolute_error": 1.0e-6,
    }.items():
        if physics_gates.get(key) != expected:
            raise A2DynamicProtocolError(f"physics_and_schema gate {key} is not frozen")
    statuses = _sequence(config.get("terminal_statuses"), "terminal_statuses")
    _exact_list(statuses, ["DYNAMIC_QUALIFIED", "TEMPORAL_REDUNDANT", "DYNAMIC_UNIDENTIFIABLE", "PHYSICS_INVALID", "BASELINE_SATURATED", "INVALID_PROTOCOL"], "terminal_statuses")


def validate_a2_dynamic_experiment_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    _validate_source_levels(config)
    if config.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise A2DynamicProtocolError("A2-DYN experiment schema is unsupported")
    expected_strings = {
        "stage": "A2-DYN-0",
        "experiment_id": "a2-dyn-protocol",
        "kind": "protocol",
        "status": "PROTOCOL_FROZEN",
        "data_config": str(DATA_CONFIG_RELATIVE_PATH).replace("\\", "/"),
        "eval_config": str(EVAL_CONFIG_RELATIVE_PATH).replace("\\", "/"),
        "test_access": "locked",
        "test_unlock_requires": "DATA_FROZEN",
        "logical_split_filter_field": "split",
    }
    for key, expected in expected_strings.items():
        if config.get(key) != expected:
            raise A2DynamicProtocolError(f"experiment.{key} must be {expected!r}")
    _exact_list(config.get("allowed_read_splits"), ["train", "val", "stress_val"], "allowed_read_splits")
    _exact_list(config.get("model_selection_splits"), ["train", "val", "stress_val"], "model_selection_splits")
    if config.get("algorithm_search_allowed") is not False or config.get("complex_access_control") is not False:
        raise A2DynamicProtocolError("A2-DYN-0 may not start algorithm search or complex access control")
    _exact_list(config.get("work_packages"), ["A2-DYN-0", "A2-DYN-1", "A2-DYN-2", "A2-DYN-3", "A2-DYN-4", "A2-DYN-5", "A2-DYN-6"], "work_packages")
    stage_gates = _mapping(config.get("stage_gates"), "stage_gates")
    expected_stage_gates = {
        "A2-DYN-0": "PROTOCOL_FROZEN",
        "A2-DYN-1": "PHYSICS_VERIFIED",
        "A2-DYN-2": "PILOT_QUALIFIED",
        "A2-DYN-3": "DIFFICULTY_QUALIFIED",
        "A2-DYN-4": "DATA_FROZEN",
        "A2-DYN-5": "DYNAMIC_QUALIFIED_OR_TERMINAL_STATUS",
        "A2-DYN-6": "HANDOFF_OR_CLOSED",
    }
    if dict(stage_gates) != expected_stage_gates:
        raise A2DynamicProtocolError("stage_gates are not frozen")
    output_dirs = _mapping(config.get("output_dirs"), "output_dirs")
    expected_dirs = {
        "runs": "outputs/runs/a2_dynamic_v1",
        "summary": "outputs/summary/a2_dynamic_v1",
        "reports": "outputs/reports/a2_dynamic_v1",
        "archive": "outputs/archive/a2_dynamic_v1",
    }
    if dict(output_dirs) != expected_dirs:
        raise A2DynamicProtocolError("output_dirs are not frozen")
    source_policy = _mapping(config.get("source_hash_policy"), "source_hash_policy")
    for key in ("config_hashes_are_required", "a1_and_a2h_are_read_only", "reference_asset_hashes_are_registered", "test_result_reuse_requires_protocol_revision"):
        if source_policy.get(key) is not True:
            raise A2DynamicProtocolError(f"source_hash_policy.{key} must be true")
    cli_policy = _mapping(config.get("cli_policy"), "cli_policy")
    if cli_policy.get("entrypoint") != "gf.pipeline.a2_dynamic_benchmark" or cli_policy.get("allowed_override_arguments") != ["--stage", "--project-root"] or cli_policy.get("unimplemented_stage_behavior") != "fail_explicitly":
        raise A2DynamicProtocolError("CLI override policy is not frozen")
    frozen_arguments = set(_sequence(cli_policy.get("frozen_arguments"), "cli_policy.frozen_arguments"))
    if {"family", "split", "sample_rate", "duration", "threshold", "seed", "model_matrix"} - frozen_arguments:
        raise A2DynamicProtocolError("CLI policy leaves a frozen argument overrideable")


def validate_a2_dynamic_pilot_config(config: Mapping[str, Any]) -> None:
    """校验实验配置中 pilot 的冻结规模、候选和资格门。"""

    _validate_no_forbidden_keys(config, "experiment.pilot")
    if config.get("stage") != "A2-DYN-2" or config.get("status") not in {"READY", "PILOT_QUALIFIED"}:
        raise A2DynamicProtocolError("experiment.pilot stage/status is not frozen")
    for key in ("seed", "mixture_count", "groups_per_family", "ultrasonic_quality_sample_count"):
        _integer(config.get(key), f"experiment.pilot.{key}", minimum=1)
    if config.get("mixture_count") != 240 or config.get("groups_per_family") != 40:
        raise A2DynamicProtocolError("experiment.pilot size must be 240 groups and 40 groups per family")
    if dict(_mapping(config.get("split_counts_per_family"), "experiment.pilot.split_counts_per_family")) != {
        "train": 20,
        "val": 10,
        "stress_val": 10,
    }:
        raise A2DynamicProtocolError("experiment.pilot split_counts_per_family are not frozen")
    _exact_list(config.get("splits"), ["train", "val", "stress_val"], "experiment.pilot.splits")
    _exact_list(config.get("sample_rates_hz"), [1.0, 2.0, 5.0], "experiment.pilot.sample_rates_hz")
    _exact_list(config.get("durations_s"), [120.0, 240.0, 360.0], "experiment.pilot.durations_s")
    _close(config.get("registered_reference_sample_rate_hz"), 5.0, "experiment.pilot.registered_reference_sample_rate_hz")
    _close(config.get("registered_reference_duration_s"), 240.0, "experiment.pilot.registered_reference_duration_s")
    if config.get("selected_sample_rate_hz") != 5.0 or config.get("selected_duration_s") != 240.0:
        if config.get("status") == "PILOT_QUALIFIED":
            raise A2DynamicProtocolError("qualified pilot selected time axis is not frozen")
    if config.get("selected_ultrasonic_profile_id") != "US-CHIRP-XCORR-PARABOLIC-1":
        if config.get("status") == "PILOT_QUALIFIED":
            raise A2DynamicProtocolError("qualified pilot ultrasonic profile is not frozen")
    if config.get("selected_tof_estimator") != "reference_xcorr_parabolic":
        if config.get("status") == "PILOT_QUALIFIED":
            raise A2DynamicProtocolError("qualified pilot ToF estimator is not frozen")
    if config.get("selected_excitation_type") != "linear_chirp":
        if config.get("status") == "PILOT_QUALIFIED":
            raise A2DynamicProtocolError("qualified pilot excitation is not frozen")
    if config.get("result_manifest") != "outputs/runs/a2_dynamic_v1/a2-dyn-2r4-pilot/manifest.json":
        raise A2DynamicProtocolError("experiment.pilot result_manifest is not frozen")
    _exact_list(
        config.get("ultrasonic_candidate_ids"),
        ["US-BURST-XCORR-1", "US-CHIRP-XCORR-PARABOLIC-1"],
        "experiment.pilot.ultrasonic_candidate_ids",
    )
    _exact_list(
        config.get("pilot_probe_ids"),
        ["P-B-LAST-LS", "P-B-EWMA-LS", "P-B-STAT-LS", "P-O-KIN-LS"],
        "experiment.pilot.pilot_probe_ids",
    )
    _close(config.get("ewma_alpha"), 0.2, "experiment.pilot.ewma_alpha")
    _exact_list(
        config.get("horizons_after_onset_s"),
        [5.0, 15.0, 30.0, 60.0, 120.0, 150.0],
        "experiment.pilot.horizons_after_onset_s",
    )
    _exact_list(
        config.get("observation_noise_std_by_sensor"),
        [1.0e-6, 0.002, 0.002],
        "experiment.pilot.observation_noise_std_by_sensor",
    )
    _exact_list(
        config.get("observation_quantization_by_sensor"),
        [1.0e-8, 0.001, 0.001],
        "experiment.pilot.observation_quantization_by_sensor",
    )
    _close(config.get("ultrasonic_internal_noise_std"), 0.01, "experiment.pilot.ultrasonic_internal_noise_std")
    dynamic_gate = {
        "minimum_active_channel_fraction": 0.95,
        "minimum_quantized_level_fraction": 1.0,
        "minimum_t50_pair_fraction": 0.70,
        "minimum_t50_separation_samples": 2,
        "minimum_stress_privileged_probe_improvement_fraction": 0.05,
        "maximum_family_degenerate_fraction": 0.05,
        "maximum_ndir_saturation_fraction": 0.0,
        "maximum_tcd_energy_residual_w": 1.0e-10,
        "minimum_ndir_low_co2_delta_v": 1.0e-5,
    }
    if dict(_mapping(config.get("dynamic_gate"), "experiment.pilot.dynamic_gate")) != dynamic_gate:
        raise A2DynamicProtocolError("experiment.pilot.dynamic_gate is not frozen")
    resource_gate = {
        "formal_rows": 6300,
        "formal_timesteps": 1200,
        "signal_channels": 3,
        "float32_bytes": 4,
        "formal_oracle_float32_channel_arrays": 5,
        "maximum_formal_core_array_bytes": 805306368,
        "maximum_pilot_peak_bytes": 268435456,
        "waveform_persistence": "temporary_only",
    }
    if dict(_mapping(config.get("resource_gate"), "experiment.pilot.resource_gate")) != resource_gate:
        raise A2DynamicProtocolError("experiment.pilot.resource_gate is not frozen")
    selection_rule = {
        "sample_rate": "choose_lowest_qualified_rate_within_information_score_gap_of_best_at_registered_duration",
        "sample_rate_comparison_duration_s": 240.0,
        "max_information_score_gap": 0.02,
        "duration": "choose_shortest_qualified_duration_with_P150_and_observed_recovery_at_selected_rate",
        "ultrasonic": "minimize_p95_absolute_tof_error_then_maximize_lock_rate_then_maximize_snr_then_minimize_latency",
        "p95_tof_error_gate_s": 1.0e-6,
        "lock_rate_gate": 0.95,
    }
    if dict(_mapping(config.get("selection_rule"), "experiment.pilot.selection_rule")) != selection_rule:
        raise A2DynamicProtocolError("experiment.pilot.selection_rule is not frozen")


def _resolve_general_fusion_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    if (root / "configs" / "data").is_dir():
        return root
    nested = root / "general_fusion"
    if (nested / "configs" / "data").is_dir():
        return nested
    raise A2DynamicProtocolError(f"cannot locate general_fusion project root from {root}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise A2DynamicProtocolError(f"cannot read JSON config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise A2DynamicProtocolError(f"invalid JSON config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise A2DynamicProtocolError(f"JSON config {path} must contain an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise A2DynamicProtocolError(f"cannot hash registered source {path}: {exc}") from exc
    return digest.hexdigest()


def _resolve_registered_asset(general_fusion_root: Path, relative_path: str) -> Path:
    candidate = (general_fusion_root / relative_path).resolve()
    if candidate.exists():
        return candidate
    raise A2DynamicProtocolError(f"registered source asset does not exist: {candidate}")


def _validate_registered_sources(config: Mapping[str, Any], general_fusion_root: Path) -> None:
    registry = _mapping(config["source_registry"], "source_registry")
    for source_id in ("a1", "a2h"):
        source = _mapping(registry[source_id], f"source_registry.{source_id}")
        config_path = _resolve_registered_asset(general_fusion_root, str(source["config_path"]))
        manifest_path = _resolve_registered_asset(general_fusion_root, str(source["manifest_path"]))
        if _sha256_file(config_path) != source["config_sha256"]:
            raise A2DynamicProtocolError(f"source_registry.{source_id}.config_sha256 does not match {config_path}")
        manifest = _read_json(manifest_path)
        if manifest.get("data_version") != source["data_version"] or manifest.get("content_sha256") != source["content_sha256"]:
            raise A2DynamicProtocolError(f"source_registry.{source_id} manifest identity does not match the frozen registry")
        if source_id == "a1" and manifest.get("split_hash") is not None and manifest.get("split_hash") != source["split_hash"]:
            raise A2DynamicProtocolError("source_registry.a1 split_hash does not match the frozen A1 manifest")
        if source_id == "a2h" and manifest.get("profile_hash") != source["profile_hash"]:
            raise A2DynamicProtocolError("source_registry.a2h profile_hash does not match the frozen A2H manifest")
    shared = _mapping(registry["shared_physics"], "source_registry.shared_physics")
    shared_path = _resolve_registered_asset(general_fusion_root, str(shared["path"]))
    if _sha256_file(shared_path) != shared["sha256"]:
        raise A2DynamicProtocolError(f"shared physics hash does not match {shared_path}")
    eos = _mapping(_mapping(config["physics_reference"], "physics_reference")["eos"], "physics_reference.eos")
    model_asset = _mapping(eos["model_asset"], "physics_reference.eos.model_asset")
    model_path = _resolve_registered_asset(general_fusion_root, str(model_asset["path"]))
    if _sha256_file(model_path) != model_asset["sha256"]:
        raise A2DynamicProtocolError(f"A2-DYN direct HEOS asset hash does not match {model_path}")
    model_payload = _read_json(model_path)
    _validate_a2dyn_direct_heos_asset(model_payload)
    runtime_evaluator = _mapping(model_payload["runtime_evaluator"], "A2-DYN direct HEOS runtime_evaluator")
    if runtime_evaluator.get("path") != shared.get("path") or runtime_evaluator.get("sha256") != shared.get("sha256"):
        raise A2DynamicProtocolError("A2-DYN direct HEOS asset and shared runtime identity do not match")


def _validate_a2dyn_direct_heos_asset(asset: Mapping[str, Any]) -> None:
    if asset.get("schema_version") != "gf-a2dyn-direct-heos-1":
        raise A2DynamicProtocolError("A2-DYN direct HEOS asset schema is unsupported")
    if asset.get("model_id") != "a2dyn_direct_multifluid_eos_v1":
        raise A2DynamicProtocolError("A2-DYN direct HEOS asset model_id is not registered")
    if asset.get("status") != "FORMAL_SYNTHETIC_GENERATOR":
        raise A2DynamicProtocolError("A2-DYN direct HEOS asset status is not frozen")
    if asset.get("definition") != "CoolProp.AbstractState(HEOS).speed_sound" or asset.get("backend") != "HEOS":
        raise A2DynamicProtocolError("A2-DYN direct HEOS definition is not frozen")
    package = _mapping(asset.get("package"), "A2-DYN direct HEOS package")
    if (
        package.get("name") != "CoolProp"
        or package.get("version") != "8.0.0"
        or package.get("source_revision") != "61b616edfbb49f32633b21d1f901bdba1002340a"
        or package.get("binary_module") != "CoolProp.CoolProp"
    ):
        raise A2DynamicProtocolError("A2-DYN direct HEOS package identity is not frozen")
    _hash(package.get("binary_sha256"), "A2-DYN direct HEOS package binary_sha256")
    if asset.get("phase_constraint") != "gas":
        raise A2DynamicProtocolError("A2-DYN direct HEOS phase constraint is not frozen")
    _exact_list(asset.get("component_order"), ["Ar", "He", "CO2"], "A2-DYN direct HEOS component_order")
    if dict(_mapping(asset.get("fluid_name_map"), "A2-DYN direct HEOS fluid_name_map")) != {"Ar": "Argon", "He": "Helium", "CO2": "CarbonDioxide"}:
        raise A2DynamicProtocolError("A2-DYN direct HEOS fluid map is not frozen")
    _exact_list(asset.get("temperature_range_k"), [278.15, 313.15], "A2-DYN direct HEOS temperature_range_k")
    _exact_list(asset.get("pressure_range_pa"), [90000.0, 112000.0], "A2-DYN direct HEOS pressure_range_pa")
    composition = _mapping(asset.get("composition_contract"), "A2-DYN direct HEOS composition_contract")
    if composition.get("basis") != "mole_fraction":
        raise A2DynamicProtocolError("A2-DYN direct HEOS composition basis is not frozen")
    for field, expected in (("lower_bound", 0.0), ("upper_bound", 1.0), ("sum", 1.0), ("sum_tolerance", 1.0e-12)):
        _close(composition.get(field), expected, f"A2-DYN direct HEOS composition_contract.{field}")
    runtime_evaluator = _mapping(asset.get("runtime_evaluator"), "A2-DYN direct HEOS runtime_evaluator")
    if runtime_evaluator.get("path") != "src/gf/sim/a2dyn_sound_speed.py":
        raise A2DynamicProtocolError("A2-DYN direct HEOS runtime path is not frozen")
    _hash(runtime_evaluator.get("sha256"), "A2-DYN direct HEOS runtime sha256")
    if asset.get("verification_scope") != "generator_consistency" or asset.get("independent_physics_validation") != "NOT_CLAIMED":
        raise A2DynamicProtocolError("A2-DYN direct HEOS validation scope is not frozen")


def _validate_registered_reference_assets(config: Mapping[str, Any], general_fusion_root: Path) -> None:
    reference = _mapping(_mapping(config["physics_reference"], "physics_reference")["ndir_reference"], "physics_reference.ndir_reference")
    for index, raw in enumerate(_sequence(reference["asset_files"], "physics_reference.ndir_reference.asset_files")):
        asset = _mapping(raw, f"asset_files[{index}]")
        path = _resolve_registered_asset(general_fusion_root, str(asset["path"]))
        actual = _sha256_file(path)
        if actual != asset["sha256"]:
            raise A2DynamicProtocolError(f"HITRAN reference asset hash mismatch for {path}")


def load_a2_dynamic_configs(project_root: str | Path = ".") -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = _resolve_general_fusion_root(project_root)
    data = _read_json(root / DATA_CONFIG_RELATIVE_PATH)
    evaluation = _read_json(root / EVAL_CONFIG_RELATIVE_PATH)
    experiment = _read_json(root / EXPERIMENT_CONFIG_RELATIVE_PATH)
    return root, data, evaluation, experiment


def validate_a2_dynamic_configs(project_root: str | Path = ".", *, verify_reference_assets: bool = False) -> dict[str, Any]:
    root, data, evaluation, experiment = load_a2_dynamic_configs(project_root)
    validate_a2_dynamic_data_config(data)
    validate_a2_dynamic_eval_config(evaluation)
    validate_a2_dynamic_experiment_config(experiment)
    if experiment["data_config"] != str(DATA_CONFIG_RELATIVE_PATH).replace("\\", "/") or experiment["eval_config"] != str(EVAL_CONFIG_RELATIVE_PATH).replace("\\", "/"):
        raise A2DynamicProtocolError("experiment config references do not point to the frozen A2-DYN configs")
    if evaluation["dataset_schema_version"] != data["schema_version"]:
        raise A2DynamicProtocolError("evaluation and data schema versions do not match")
    pilot = _mapping(experiment.get("pilot"), "experiment.pilot")
    validate_a2_dynamic_pilot_config(pilot)
    if pilot.get("status") == "PILOT_QUALIFIED":
        axis = _mapping(data["time_axis"], "data.time_axis")
        ultrasonic = _mapping(_mapping(data["hardware_profiles"], "hardware_profiles")["ultrasonic"], "hardware_profiles.ultrasonic")
        expected_pairs = (
            ("selected_sample_rate_hz", axis["sample_rate_hz"]),
            ("selected_duration_s", axis["duration_s"]),
            ("selected_ultrasonic_profile_id", ultrasonic["selected_profile_id"]),
            ("selected_tof_estimator", ultrasonic["selected_tof_estimator"]),
            ("selected_excitation_type", ultrasonic["selected_excitation_type"]),
        )
        for field, expected in expected_pairs:
            if pilot.get(field) != expected:
                raise A2DynamicProtocolError(f"qualified pilot {field} does not match the formal data contract")
        resource_gate = _mapping(pilot.get("resource_gate"), "experiment.pilot.resource_gate")
        if resource_gate.get("formal_timesteps") != axis["timesteps"]:
            raise A2DynamicProtocolError("pilot resource formal_timesteps does not match the formal time axis")
        if pilot.get("result_manifest") != ultrasonic.get("selection_manifest"):
            raise A2DynamicProtocolError("pilot result_manifest and ultrasonic selection_manifest must match")
    _validate_registered_sources(data, root)
    if verify_reference_assets:
        _validate_registered_reference_assets(data, root)
    return {
        "status": "PASS",
        "protocol_status": data["status"],
        "data_version": data["data_version"],
        "schema_version": data["schema_version"],
        "families": list(FAMILIES),
        "split_totals": data["split_contract"]["totals"],
        "source_hashes_verified": True,
        "reference_assets_registered": True,
        "reference_assets_verified": verify_reference_assets,
    }


def run_a2_dynamic_protocol(project_root: str | Path = ".", *, verify_reference_assets: bool = False) -> dict[str, Any]:
    """执行 A2-DYN-0 protocol stage，返回可序列化的校验摘要。"""

    summary = validate_a2_dynamic_configs(project_root, verify_reference_assets=verify_reference_assets)
    return {"status": "PASS", "stage": "A2-DYN-0", "summary": summary}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen A2-DYN-0 machine protocol.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--verify-reference-assets", action="store_true")
    args = parser.parse_args(argv)
    result = run_a2_dynamic_protocol(args.project_root, verify_reference_assets=args.verify_reference_assets)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "A2DynamicProtocolError",
    "DATA_CONFIG_RELATIVE_PATH",
    "EVAL_CONFIG_RELATIVE_PATH",
    "EXPERIMENT_CONFIG_RELATIVE_PATH",
    "load_a2_dynamic_configs",
    "main",
    "run_a2_dynamic_protocol",
    "validate_a2_dynamic_pilot_config",
    "validate_a2_dynamic_configs",
    "validate_a2_dynamic_data_config",
    "validate_a2_dynamic_eval_config",
    "validate_a2_dynamic_experiment_config",
    "validate_a2_dynamic_records",
]
