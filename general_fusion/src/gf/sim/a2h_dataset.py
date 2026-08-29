from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from gf.dl.contracts import UnifiedSample
from gf.sim.a1_dataset import (
    A1PhysicsConfig,
    DEFAULT_A1_PHYSICS,
    SENSOR_IDS,
    TARGET_NAMES,
    deterministic_signal_vector,
)
from gf.sim.ar_he_co2 import SENSOR_TYPES


A2H_SCHEMA_VERSION = "gf-a2h-data-2"
A2H_DATASET_ID = "ar_he_co2"
A2H_DATA_VERSION_PREFIX = "gf-a2h-v2-"
A2H_SPLITS = ("train", "val", "stress_val", "hard_test")
A2H_DEVELOPMENT_SPLITS = ("train", "val", "stress_val")
FORBIDDEN_KEYS = frozenset(
    {"base_condition_id", "noise_seed_index", "noise_seed", "sequence_id"}
)


@dataclass(frozen=True)
class A2HPhysicsConfig:
    """可变温压的 A2H 物理参数容器。

    方程仍由 ``a1_dataset.deterministic_signal_vector`` 调用；本类只允许
    A2H 的环境参数变化，并把参数转换成 A1 物理函数需要的结构。
    """

    temperature_k: float
    pressure_pa: float
    acoustic_path_length_m: float
    ndir_optical_path_length_m: float
    timesteps: int
    dt_s: float
    tof_delay_s: float
    ndir_baseline_v: float
    ndir_effective_absorbance_per_co2_percent: float
    tcs_baseline_v: float
    tcs_response_v_per_w_m_k: float
    tcs_reference_conductivity_w_m_k: float
    tof_resolution_s: float
    thermal_voltage_resolution_v: float
    ndir_voltage_noise_std_v: float
    jacobian_step_pct: float
    signal_bounds: Mapping[str, tuple[float, float]]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "A2HPhysicsConfig":
        required = {
            "temperature_k",
            "pressure_pa",
            "acoustic_path_length_m",
            "ndir_optical_path_length_m",
            "timesteps",
            "dt_s",
            "tof_delay_s",
            "ndir_baseline_v",
            "ndir_effective_absorbance_per_co2_percent",
            "tcs_baseline_v",
            "tcs_response_v_per_w_m_k",
            "tcs_reference_conductivity_w_m_k",
            "tof_resolution_s",
            "thermal_voltage_resolution_v",
            "ndir_voltage_noise_std_v",
            "jacobian_step_pct",
            "signal_bounds",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"A2H physics config is missing keys: {missing}")
        bounds_raw = raw["signal_bounds"]
        if not isinstance(bounds_raw, Mapping):
            raise ValueError("signal_bounds must be a mapping")
        bounds: dict[str, tuple[float, float]] = {}
        for sensor_id in SENSOR_IDS:
            value = bounds_raw.get(sensor_id)
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != 2
            ):
                raise ValueError(f"signal_bounds[{sensor_id!r}] must contain [min, max]")
            lower, upper = float(value[0]), float(value[1])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"signal_bounds[{sensor_id!r}] must be finite and increasing")
            bounds[sensor_id] = (lower, upper)
        config = cls(
            temperature_k=float(raw["temperature_k"]),
            pressure_pa=float(raw["pressure_pa"]),
            acoustic_path_length_m=float(raw["acoustic_path_length_m"]),
            ndir_optical_path_length_m=float(raw["ndir_optical_path_length_m"]),
            timesteps=int(raw["timesteps"]),
            dt_s=float(raw["dt_s"]),
            tof_delay_s=float(raw["tof_delay_s"]),
            ndir_baseline_v=float(raw["ndir_baseline_v"]),
            ndir_effective_absorbance_per_co2_percent=float(
                raw["ndir_effective_absorbance_per_co2_percent"]
            ),
            tcs_baseline_v=float(raw["tcs_baseline_v"]),
            tcs_response_v_per_w_m_k=float(raw["tcs_response_v_per_w_m_k"]),
            tcs_reference_conductivity_w_m_k=float(raw["tcs_reference_conductivity_w_m_k"]),
            tof_resolution_s=float(raw["tof_resolution_s"]),
            thermal_voltage_resolution_v=float(raw["thermal_voltage_resolution_v"]),
            ndir_voltage_noise_std_v=float(raw["ndir_voltage_noise_std_v"]),
            jacobian_step_pct=float(raw["jacobian_step_pct"]),
            signal_bounds=MappingProxyType(bounds),
        )
        config.validate()
        return config

    @classmethod
    def from_a1(cls, physics: A1PhysicsConfig = DEFAULT_A1_PHYSICS) -> "A2HPhysicsConfig":
        return cls.from_mapping(physics.to_dict())

    def validate(self) -> None:
        positive = {
            "temperature_k": self.temperature_k,
            "pressure_pa": self.pressure_pa,
            "acoustic_path_length_m": self.acoustic_path_length_m,
            "ndir_optical_path_length_m": self.ndir_optical_path_length_m,
            "dt_s": self.dt_s,
            "tof_delay_s": self.tof_delay_s,
            "ndir_baseline_v": self.ndir_baseline_v,
            "ndir_effective_absorbance_per_co2_percent": self.ndir_effective_absorbance_per_co2_percent,
            "tcs_response_v_per_w_m_k": self.tcs_response_v_per_w_m_k,
            "tcs_reference_conductivity_w_m_k": self.tcs_reference_conductivity_w_m_k,
            "tof_resolution_s": self.tof_resolution_s,
            "thermal_voltage_resolution_v": self.thermal_voltage_resolution_v,
            "ndir_voltage_noise_std_v": self.ndir_voltage_noise_std_v,
            "jacobian_step_pct": self.jacobian_step_pct,
        }
        invalid = [
            name for name, value in positive.items() if not math.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(f"A2H physics parameters must be finite and positive: {invalid}")
        if not math.isfinite(self.tcs_baseline_v):
            raise ValueError("tcs_baseline_v must be finite")
        if self.timesteps != 1:
            raise ValueError("A2H uses steady-state observations with timesteps=1")
        if self.jacobian_step_pct >= 0.25:
            raise ValueError("jacobian_step_pct must be smaller than the interior margin")
        if set(self.signal_bounds) != set(SENSOR_IDS):
            raise ValueError(f"signal_bounds must cover exactly {list(SENSOR_IDS)}")

    def to_a1_physics(self) -> A1PhysicsConfig:
        """Return the structural parameter object consumed by the shared A1 equations."""

        return A1PhysicsConfig(
            temperature_k=self.temperature_k,
            pressure_pa=self.pressure_pa,
            acoustic_path_length_m=self.acoustic_path_length_m,
            ndir_optical_path_length_m=self.ndir_optical_path_length_m,
            timesteps=self.timesteps,
            dt_s=self.dt_s,
            tof_delay_s=self.tof_delay_s,
            ndir_baseline_v=self.ndir_baseline_v,
            ndir_effective_absorbance_per_co2_percent=self.ndir_effective_absorbance_per_co2_percent,
            tcs_baseline_v=self.tcs_baseline_v,
            tcs_response_v_per_w_m_k=self.tcs_response_v_per_w_m_k,
            tcs_reference_conductivity_w_m_k=self.tcs_reference_conductivity_w_m_k,
            tof_resolution_s=self.tof_resolution_s,
            thermal_voltage_resolution_v=self.thermal_voltage_resolution_v,
            ndir_voltage_noise_std_v=self.ndir_voltage_noise_std_v,
            jacobian_step_pct=self.jacobian_step_pct,
            signal_bounds=MappingProxyType(dict(self.signal_bounds)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_k": self.temperature_k,
            "pressure_pa": self.pressure_pa,
            "acoustic_path_length_m": self.acoustic_path_length_m,
            "ndir_optical_path_length_m": self.ndir_optical_path_length_m,
            "timesteps": self.timesteps,
            "dt_s": self.dt_s,
            "tof_delay_s": self.tof_delay_s,
            "ndir_baseline_v": self.ndir_baseline_v,
            "ndir_effective_absorbance_per_co2_percent": self.ndir_effective_absorbance_per_co2_percent,
            "tcs_baseline_v": self.tcs_baseline_v,
            "tcs_response_v_per_w_m_k": self.tcs_response_v_per_w_m_k,
            "tcs_reference_conductivity_w_m_k": self.tcs_reference_conductivity_w_m_k,
            "tof_resolution_s": self.tof_resolution_s,
            "thermal_voltage_resolution_v": self.thermal_voltage_resolution_v,
            "ndir_voltage_noise_std_v": self.ndir_voltage_noise_std_v,
            "jacobian_step_pct": self.jacobian_step_pct,
            "signal_bounds": {
                sensor_id: list(self.signal_bounds[sensor_id]) for sensor_id in SENSOR_IDS
            },
        }


@dataclass(frozen=True)
class CalibrationProfile:
    calibration_profile_id: str
    sensor_offsets: Mapping[str, float]
    sensor_gains: Mapping[str, float]
    acoustic_path_scale: float
    tcs_response_scale: float
    ndir_absorbance_scale: float
    source_level: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CalibrationProfile":
        profile_id = str(raw.get("calibration_profile_id", ""))
        if not profile_id:
            raise ValueError("calibration_profile_id must be non-empty")
        offsets = _sensor_float_mapping(raw.get("sensor_offsets"), "sensor_offsets")
        gains = _sensor_float_mapping(raw.get("sensor_gains"), "sensor_gains")
        physical_scales = raw.get("physical_scales")
        if not isinstance(physical_scales, Mapping):
            raise ValueError("physical_scales must be an object")
        required = {"acoustic_path_length", "tcs_response", "ndir_absorbance"}
        if set(physical_scales) != required:
            raise ValueError(f"physical_scales must contain exactly {sorted(required)}")
        values = {
            "acoustic_path_scale": float(physical_scales["acoustic_path_length"]),
            "tcs_response_scale": float(physical_scales["tcs_response"]),
            "ndir_absorbance_scale": float(physical_scales["ndir_absorbance"]),
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError(f"physical calibration scales must be finite and positive: {values}")
        if any(not math.isfinite(value) for value in offsets.values()):
            raise ValueError("sensor_offsets must contain finite values")
        if any(not math.isfinite(value) or value <= 0.0 for value in gains.values()):
            raise ValueError("sensor_gains must contain finite positive values")
        return cls(
            calibration_profile_id=profile_id,
            sensor_offsets=MappingProxyType(offsets),
            sensor_gains=MappingProxyType(gains),
            source_level=str(raw.get("source_level", "unspecified")),
            **values,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_profile_id": self.calibration_profile_id,
            "sensor_offsets": dict(self.sensor_offsets),
            "sensor_gains": dict(self.sensor_gains),
            "physical_scales": {
                "acoustic_path_length": self.acoustic_path_scale,
                "tcs_response": self.tcs_response_scale,
                "ndir_absorbance": self.ndir_absorbance_scale,
            },
            "source_level": self.source_level,
        }


@dataclass(frozen=True)
class NoiseProfile:
    noise_profile_id: str
    white_scale: float
    correlated_scale: float
    batch_scale: float
    correlation_vector: tuple[float, float, float]
    source_level: str

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NoiseProfile":
        profile_id = str(raw.get("noise_profile_id", ""))
        if not profile_id:
            raise ValueError("noise_profile_id must be non-empty")
        values = {
            "white_scale": float(raw.get("white_scale", 0.0)),
            "correlated_scale": float(raw.get("correlated_scale", 0.0)),
            "batch_scale": float(raw.get("batch_scale", 0.0)),
        }
        if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
            raise ValueError(f"noise scales must be finite and non-negative: {values}")
        vector_raw = raw.get("correlation_vector", [1.0, 1.0, 1.0])
        if (
            not isinstance(vector_raw, Sequence)
            or isinstance(vector_raw, (str, bytes))
            or len(vector_raw) != len(SENSOR_IDS)
        ):
            raise ValueError("correlation_vector must contain one value per sensor")
        vector = tuple(float(value) for value in vector_raw)
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("correlation_vector must contain finite values")
        return cls(
            noise_profile_id=profile_id,
            correlation_vector=vector,
            source_level=str(raw.get("source_level", "unspecified")),
            **values,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "noise_profile_id": self.noise_profile_id,
            "white_scale": self.white_scale,
            "correlated_scale": self.correlated_scale,
            "batch_scale": self.batch_scale,
            "correlation_vector": list(self.correlation_vector),
            "source_level": self.source_level,
        }


@dataclass(frozen=True)
class A2HCondition:
    mixture_id: str
    x_ar_pct: float
    x_he_pct: float
    x_co2_pct: float
    split_family: str
    split: str
    condition_family: str
    binary_pair: str | None = None

    def __post_init__(self) -> None:
        _validate_composition(
            (self.x_ar_pct, self.x_he_pct, self.x_co2_pct),
            condition_family=self.condition_family,
            binary_pair=self.binary_pair,
        )
        if not self.mixture_id:
            raise ValueError("mixture_id must be non-empty")
        if not self.split_family:
            raise ValueError("split_family must be non-empty")
        if self.split not in A2H_SPLITS:
            raise ValueError(f"split must be one of {A2H_SPLITS}, got {self.split!r}")

    @property
    def composition(self) -> tuple[float, float, float]:
        return self.x_ar_pct, self.x_he_pct, self.x_co2_pct

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mixture_id": self.mixture_id,
            "x_Ar_pct": self.x_ar_pct,
            "x_He_pct": self.x_he_pct,
            "x_CO2_pct": self.x_co2_pct,
            "split_family": self.split_family,
            "split": self.split,
            "condition_family": self.condition_family,
        }
        if self.binary_pair is not None:
            result["binary_pair"] = self.binary_pair
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A2HCondition":
        return cls(
            mixture_id=str(raw["mixture_id"]),
            x_ar_pct=float(raw["x_Ar_pct"]),
            x_he_pct=float(raw["x_He_pct"]),
            x_co2_pct=float(raw["x_CO2_pct"]),
            split_family=str(raw["split_family"]),
            split=str(raw["split"]),
            condition_family=str(raw["condition_family"]),
            binary_pair=str(raw["binary_pair"]) if raw.get("binary_pair") is not None else None,
        )


@dataclass(frozen=True)
class A2HObservation:
    observation_id: str
    mixture_id: str
    x_ar_pct: float
    x_he_pct: float
    x_co2_pct: float
    split_family: str
    split: str
    condition_family: str
    environment_id: str
    calibration_profile_id: str
    noise_profile_id: str
    repeat_index: int
    temperature_k: float
    pressure_pa: float
    binary_pair: str | None = None

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must be non-empty")
        if not self.mixture_id:
            raise ValueError("mixture_id must be non-empty")
        if not self.environment_id or not self.calibration_profile_id or not self.noise_profile_id:
            raise ValueError("environment, calibration, and noise profile IDs must be non-empty")
        if not isinstance(self.repeat_index, int) or isinstance(self.repeat_index, bool) or self.repeat_index < 0:
            raise ValueError("repeat_index must be a non-negative integer")
        if not math.isfinite(self.temperature_k) or self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be finite and positive")
        if not math.isfinite(self.pressure_pa) or self.pressure_pa <= 0.0:
            raise ValueError("pressure_pa must be finite and positive")
        _validate_composition(
            (self.x_ar_pct, self.x_he_pct, self.x_co2_pct),
            condition_family=self.condition_family,
            binary_pair=self.binary_pair,
        )
        if not self.split_family:
            raise ValueError("split_family must be non-empty")
        if self.split not in A2H_SPLITS:
            raise ValueError(f"split must be one of {A2H_SPLITS}, got {self.split!r}")

    @property
    def composition(self) -> tuple[float, float, float]:
        return self.x_ar_pct, self.x_he_pct, self.x_co2_pct

    @property
    def condition(self) -> A2HCondition:
        return A2HCondition(
            mixture_id=self.mixture_id,
            x_ar_pct=self.x_ar_pct,
            x_he_pct=self.x_he_pct,
            x_co2_pct=self.x_co2_pct,
            split_family=self.split_family,
            split=self.split,
            condition_family=self.condition_family,
            binary_pair=self.binary_pair,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "observation_id": self.observation_id,
            "mixture_id": self.mixture_id,
            "x_Ar_pct": self.x_ar_pct,
            "x_He_pct": self.x_he_pct,
            "x_CO2_pct": self.x_co2_pct,
            "split_family": self.split_family,
            "split": self.split,
            "condition_family": self.condition_family,
            "environment_id": self.environment_id,
            "calibration_profile_id": self.calibration_profile_id,
            "noise_profile_id": self.noise_profile_id,
            "repeat_index": self.repeat_index,
            "temperature_k": self.temperature_k,
            "pressure_pa": self.pressure_pa,
        }
        if self.binary_pair is not None:
            result["binary_pair"] = self.binary_pair
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A2HObservation":
        return cls(
            observation_id=str(raw["observation_id"]),
            mixture_id=str(raw["mixture_id"]),
            x_ar_pct=float(raw["x_Ar_pct"]),
            x_he_pct=float(raw["x_He_pct"]),
            x_co2_pct=float(raw["x_CO2_pct"]),
            split_family=str(raw["split_family"]),
            split=str(raw["split"]),
            condition_family=str(raw["condition_family"]),
            environment_id=str(raw["environment_id"]),
            calibration_profile_id=str(raw["calibration_profile_id"]),
            noise_profile_id=str(raw["noise_profile_id"]),
            repeat_index=int(raw["repeat_index"]),
            temperature_k=float(raw["temperature_k"]),
            pressure_pa=float(raw["pressure_pa"]),
            binary_pair=str(raw["binary_pair"]) if raw.get("binary_pair") is not None else None,
        )


@dataclass(frozen=True)
class A2HDataset:
    observations: tuple[A2HObservation, ...]
    signals: np.ndarray
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.signals, dtype=np.float32)
        if values.shape != (len(self.observations), len(SENSOR_IDS)):
            raise ValueError(
                f"signals must have shape ({len(self.observations)}, {len(SENSOR_IDS)}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("signals must contain only finite values")
        observation_ids = [observation.observation_id for observation in self.observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("observation_id values must be unique")
        compositions: dict[str, tuple[float, float, float]] = {}
        for observation in self.observations:
            previous = compositions.setdefault(observation.mixture_id, observation.composition)
            if not np.allclose(previous, observation.composition, rtol=0.0, atol=1.0e-8):
                raise ValueError(f"all observations in a mixture group must share a target: {observation.mixture_id}")
        values.setflags(write=False)
        object.__setattr__(self, "signals", values)
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    @property
    def conditions(self) -> tuple[A2HCondition, ...]:
        by_id: dict[str, A2HCondition] = {}
        for observation in self.observations:
            by_id.setdefault(observation.mixture_id, observation.condition)
        return tuple(by_id[key] for key in sorted(by_id))

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(observation.mixture_id for observation in self.observations)

    @property
    def hard_test_indices(self) -> np.ndarray:
        return np.asarray(
            [index for index, observation in enumerate(self.observations) if observation.split == "hard_test"],
            dtype=np.int64,
        )

    def indices(
        self,
        *,
        split_family: str | None = None,
        split: str | None = None,
    ) -> np.ndarray:
        if split is not None and split not in A2H_SPLITS:
            raise ValueError(f"split must be one of {A2H_SPLITS}")
        return np.asarray(
            [
                index
                for index, observation in enumerate(self.observations)
                if (split_family is None or observation.split_family == split_family)
                and (split is None or observation.split == split)
            ],
            dtype=np.int64,
        )

    def samples(self, indices: Sequence[int] | np.ndarray | None = None) -> list[UnifiedSample]:
        index_values = (
            np.arange(len(self.observations), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        samples: list[UnifiedSample] = []
        for index in index_values:
            observation = self.observations[int(index)]
            signal_values = self.signals[int(index)]
            signals = tuple(
                np.array([[signal_values[sensor_index]]], dtype=np.float32)
                for sensor_index in range(len(SENSOR_IDS))
            )
            metadata: dict[str, str | int | float] = {
                "mixture_id": observation.mixture_id,
                "observation_id": observation.observation_id,
                "split_family": observation.split_family,
                "split": observation.split,
                "condition_family": observation.condition_family,
                "environment_id": observation.environment_id,
                "calibration_profile_id": observation.calibration_profile_id,
                "noise_profile_id": observation.noise_profile_id,
                "repeat_index": observation.repeat_index,
                "temperature_k": observation.temperature_k,
                "pressure_pa": observation.pressure_pa,
                "x_Ar_pct": observation.x_ar_pct,
                "x_He_pct": observation.x_he_pct,
                "x_CO2_pct": observation.x_co2_pct,
            }
            if observation.binary_pair is not None:
                metadata["binary_pair"] = observation.binary_pair
            samples.append(
                UnifiedSample(
                    signals=signals,
                    sensor_id=SENSOR_IDS,
                    sensor_type=tuple(SENSOR_TYPES[sensor_id] for sensor_id in SENSOR_IDS),
                    valid_mask=tuple(np.ones_like(signal, dtype=np.bool_) for signal in signals),
                    quality=tuple(np.ones(1, dtype=np.float32) for _ in SENSOR_IDS),
                    time=tuple(np.array([0.0], dtype=np.float64) for _ in SENSOR_IDS),
                    target=np.asarray(observation.composition, dtype=np.float32),
                    target_mask=np.ones(len(TARGET_NAMES), dtype=np.bool_),
                    group_id=observation.mixture_id,
                    dataset_id="ar_he_co2_a2h",
                    metadata=metadata,
                )
            )
        return samples


def deterministic_a2h_signal_vector(
    composition: Sequence[float],
    *,
    physics: A2HPhysicsConfig,
    calibration: CalibrationProfile | None = None,
) -> np.ndarray:
    """Evaluate the shared physics, then apply only the registered profile transform."""

    physics.validate()
    profile = calibration or nominal_calibration_profile()
    scaled_physics = replace(
        physics,
        acoustic_path_length_m=physics.acoustic_path_length_m * profile.acoustic_path_scale,
        tcs_response_v_per_w_m_k=physics.tcs_response_v_per_w_m_k * profile.tcs_response_scale,
        ndir_effective_absorbance_per_co2_percent=(
            physics.ndir_effective_absorbance_per_co2_percent * profile.ndir_absorbance_scale
        ),
    )
    values = deterministic_signal_vector(composition, scaled_physics.to_a1_physics())
    transformed = np.asarray(
        [
            values[index] * profile.sensor_gains[sensor_id] + profile.sensor_offsets[sensor_id]
            for index, sensor_id in enumerate(SENSOR_IDS)
        ],
        dtype=np.float64,
    )
    if not np.isfinite(transformed).all():
        raise ValueError(f"A2H deterministic signal contains non-finite values: {transformed}")
    return transformed


def generate_a2h_dataset(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | str | Path,
    data_version: str | None = None,
    generation_seed: int | None = None,
    split_seed: int | None = None,
) -> A2HDataset:
    raw_config = _read_config(config)
    _validate_config_for_generation(raw_config)
    version = str(data_version or raw_config["data_version"])
    generation_state = int(
        raw_config["generation_seed"] if generation_seed is None else generation_seed
    )
    split_state = int(raw_config["split_seed"] if split_seed is None else split_seed)
    if generation_state < 0 or split_state < 0:
        raise ValueError("generation_seed and split_seed must be non-negative")

    base_physics = A2HPhysicsConfig.from_mapping(raw_config["physics"])
    environments = _parse_environments(raw_config["environment_blocks"])
    calibrations = {
        profile.calibration_profile_id: profile
        for profile in (
            CalibrationProfile.from_mapping(value)
            for value in raw_config["calibration_profiles"]
        )
    }
    noises = {
        profile.noise_profile_id: profile
        for profile in (NoiseProfile.from_mapping(value) for value in raw_config["noise_profiles"])
    }
    records: list[A2HObservation] = []
    signal_values: list[np.ndarray] = []
    used_mixture_ids: set[str] = set()
    used_observation_ids: set[str] = set()
    batch_effects: dict[tuple[str, str, str, str, str], np.ndarray] = {}
    family_items = list(raw_config["families"].items())
    regions = raw_config["composition_regions"]

    for family_index, (family_name, family_raw) in enumerate(family_items):
        if not isinstance(family_raw, Mapping):
            raise ValueError(f"families[{family_name!r}] must be an object")
        split_counts = family_raw["splits"]
        for split_index, split in enumerate(A2H_SPLITS):
            count = int(split_counts[split])
            family_rng = np.random.default_rng(
                np.random.SeedSequence([generation_state, split_state, family_index, split_index])
            )
            compositions = _generate_compositions(
                count,
                mode=str(family_raw["composition_mode_by_split"][split]),
                rng=family_rng,
                regions=regions,
            )
            repeat_count = int(family_raw["repeat_count"])
            if repeat_count <= 0:
                raise ValueError(f"families[{family_name!r}].repeat_count must be positive")
            for mixture_index, composition in enumerate(compositions, start=1):
                mixture_id = f"a2h-{family_name}-{split}-m{mixture_index:04d}"
                if mixture_id in used_mixture_ids:
                    raise ValueError(f"duplicate generated mixture_id: {mixture_id}")
                used_mixture_ids.add(mixture_id)
                condition_family, binary_pair = _classify_composition(composition)
                environment_id = _select_profile_id(
                    family_raw["environment_by_split"][split],
                    mixture_index=mixture_index - 1,
                    field="environment_by_split",
                )
                calibration_id = _select_profile_id(
                    family_raw["calibration_by_split"][split],
                    mixture_index=mixture_index - 1,
                    field="calibration_by_split",
                )
                noise_id = _select_profile_id(
                    family_raw["noise_by_split"][split],
                    mixture_index=mixture_index - 1,
                    field="noise_by_split",
                )
                if environment_id not in environments:
                    raise ValueError(f"unknown environment_id {environment_id!r}")
                if calibration_id not in calibrations:
                    raise ValueError(f"unknown calibration_profile_id {calibration_id!r}")
                if noise_id not in noises:
                    raise ValueError(f"unknown noise_profile_id {noise_id!r}")
                environment = environments[environment_id]
                calibration = calibrations[calibration_id]
                noise_profile = noises[noise_id]
                effective_physics = replace(
                    base_physics,
                    temperature_k=environment["temperature_k"],
                    pressure_pa=environment["pressure_pa"],
                )
                base_signal = deterministic_a2h_signal_vector(
                    composition,
                    physics=effective_physics,
                    calibration=calibration,
                )
                batch_key = (family_name, split, environment_id, calibration_id, noise_id)
                if batch_key not in batch_effects:
                    batch_effects[batch_key] = (
                        family_rng.normal(size=len(SENSOR_IDS))
                        * _noise_scales(effective_physics)
                        * noise_profile.batch_scale
                    )
                batch_effect = batch_effects[batch_key]
                for repeat_index in range(repeat_count):
                    observation_id = f"{mixture_id}-o{repeat_index + 1:02d}"
                    if observation_id in used_observation_ids:
                        raise ValueError(f"duplicate generated observation_id: {observation_id}")
                    used_observation_ids.add(observation_id)
                    correlated = (
                        family_rng.normal()
                        * _noise_scales(effective_physics)
                        * noise_profile.correlated_scale
                        * np.asarray(noise_profile.correlation_vector, dtype=np.float64)
                    )
                    white = family_rng.normal(size=len(SENSOR_IDS)) * _noise_scales(effective_physics) * noise_profile.white_scale
                    signal = base_signal + batch_effect + correlated + white
                    _validate_signal_bounds(signal, effective_physics, observation_id)
                    records.append(
                        A2HObservation(
                            observation_id=observation_id,
                            mixture_id=mixture_id,
                            x_ar_pct=composition[0],
                            x_he_pct=composition[1],
                            x_co2_pct=composition[2],
                            split_family=family_name,
                            split=split,
                            condition_family=condition_family,
                            binary_pair=binary_pair,
                            environment_id=environment_id,
                            calibration_profile_id=calibration_id,
                            noise_profile_id=noise_id,
                            repeat_index=repeat_index,
                            temperature_k=environment["temperature_k"],
                            pressure_pa=environment["pressure_pa"],
                        )
                    )
                    signal_values.append(signal.astype(np.float32))

    observations = tuple(records)
    signals = np.vstack(signal_values).astype(np.float32)
    manifest = _build_manifest(
        raw_config=raw_config,
        version=version,
        generation_seed=generation_state,
        split_seed=split_state,
        base_physics=base_physics,
        observations=observations,
        signals=signals,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path / "observations.npz", signals=signals)
    (output_path / "config_snapshot.json").write_text(
        json.dumps(raw_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return load_a2h_dataset(output_path, include_hard_test=False)


def load_a2h_dataset(
    dataset_dir: str | Path,
    *,
    include_hard_test: bool = False,
) -> A2HDataset:
    dataset_path = Path(dataset_dir)
    manifest = json.loads((dataset_path / "manifest.json").read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    with np.load(dataset_path / "observations.npz", allow_pickle=False) as archive:
        full_signals = np.asarray(archive["signals"], dtype=np.float32).copy()
    full_observations = tuple(
        A2HObservation.from_dict(value) for value in manifest["observations"]
    )
    if full_signals.shape != (len(full_observations), len(SENSOR_IDS)):
        raise ValueError(
            f"A2H dataset shape mismatch: observations={len(full_observations)}, signals={full_signals.shape}"
        )
    expected_hash = manifest.get("content_sha256")
    if not isinstance(expected_hash, str) or expected_hash != _content_sha256(manifest, full_signals):
        raise ValueError("A2H dataset content_sha256 mismatch")
    full_dataset = A2HDataset(
        observations=full_observations,
        signals=full_signals,
        manifest=manifest,
    )
    _validate_dataset_structure(full_dataset)
    if include_hard_test:
        return full_dataset
    visible = np.asarray(
        [index for index, observation in enumerate(full_observations) if observation.split != "hard_test"],
        dtype=np.int64,
    )
    return A2HDataset(
        observations=tuple(full_observations[int(index)] for index in visible),
        signals=full_signals[visible],
        manifest=manifest,
    )


def load_dataset(dataset_dir: str | Path, *, include_hard_test: bool = False) -> A2HDataset:
    """Short alias retained for callers that use the sim package convention."""

    return load_a2h_dataset(dataset_dir, include_hard_test=include_hard_test)


def nominal_calibration_profile() -> CalibrationProfile:
    return CalibrationProfile(
        calibration_profile_id="CAL-NOMINAL",
        sensor_offsets=MappingProxyType({sensor_id: 0.0 for sensor_id in SENSOR_IDS}),
        sensor_gains=MappingProxyType({sensor_id: 1.0 for sensor_id in SENSOR_IDS}),
        acoustic_path_scale=1.0,
        tcs_response_scale=1.0,
        ndir_absorbance_scale=1.0,
        source_level="A1_frozen_reference",
    )


def nominal_signal_parity(
    compositions: Sequence[Sequence[float]],
    *,
    a1_physics: A1PhysicsConfig = DEFAULT_A1_PHYSICS,
) -> dict[str, Any]:
    a2h_physics = A2HPhysicsConfig.from_a1(a1_physics)
    differences = np.asarray(
        [
            deterministic_a2h_signal_vector(
                composition,
                physics=a2h_physics,
                calibration=nominal_calibration_profile(),
            )
            - deterministic_signal_vector(composition, a1_physics)
            for composition in compositions
        ],
        dtype=np.float64,
    )
    absolute = np.abs(differences)
    return {
        "sample_count": int(len(compositions)),
        "max_absolute_difference": float(absolute.max()) if absolute.size else 0.0,
        "mean_absolute_difference": float(absolute.mean()) if absolute.size else 0.0,
        "per_sensor_max_absolute_difference": {
            sensor_id: float(absolute[:, index].max()) if absolute.size else 0.0
            for index, sensor_id in enumerate(SENSOR_IDS)
        },
    }


def compute_split_family_hash(value: A2HDataset | Mapping[str, Any] | Sequence[A2HObservation]) -> str:
    if isinstance(value, A2HDataset):
        observations = value.observations
    elif isinstance(value, Mapping):
        raw_observations = value.get("observations")
        if not isinstance(raw_observations, list):
            raise ValueError("manifest observations must be a list")
        observations = tuple(A2HObservation.from_dict(item) for item in raw_observations)
    else:
        observations = tuple(value)
    assignments = sorted(
        {
            (
                observation.split_family,
                observation.mixture_id,
                observation.split,
            )
            for observation in observations
        }
    )
    return _canonical_sha256(
        {
            "schema_version": "gf-a2h-split-1",
            "assignments": [
                {
                    "split_family": family,
                    "mixture_id": mixture_id,
                    "split": split,
                }
                for family, mixture_id, split in assignments
            ],
        }
    )


def composition_region(
    composition: Sequence[float],
    *,
    regions: Mapping[str, Any] | None = None,
) -> str:
    values = tuple(float(value) for value in composition)
    family, _ = _classify_composition(values)
    if family == "pure":
        return "pure_gas"
    if family == "binary":
        return "binary_boundary"
    near_boundary = float((regions or {}).get("near_boundary_max_pct", 2.0))
    if min(values) <= near_boundary:
        return "near_boundary"
    configured_band = (regions or {}).get("concentration_band", {})
    if isinstance(configured_band, Mapping):
        component = str(configured_band.get("component", "CO2"))
        component_index = {"Ar": 0, "He": 1, "CO2": 2}.get(component)
        if component_index is not None:
            lower = float(configured_band.get("minimum_pct", -math.inf))
            upper = float(configured_band.get("maximum_pct", math.inf))
            if lower <= values[component_index] <= upper:
                return "concentration_band"
    sector = (regions or {}).get("simplex_sector", {})
    if isinstance(sector, Mapping) and values[0] >= float(sector.get("x_Ar_min_pct", math.inf)) and values[1] <= float(sector.get("x_He_max_pct", -math.inf)):
        return "simplex_sector"
    return "center"


def _build_manifest(
    *,
    raw_config: Mapping[str, Any],
    version: str,
    generation_seed: int,
    split_seed: int,
    base_physics: A2HPhysicsConfig,
    observations: Sequence[A2HObservation],
    signals: np.ndarray,
) -> dict[str, Any]:
    conditions_by_id: dict[str, A2HCondition] = {}
    for observation in observations:
        conditions_by_id.setdefault(observation.mixture_id, observation.condition)
    profiles = {
        "environment_blocks": raw_config["environment_blocks"],
        "noise_profiles": raw_config["noise_profiles"],
        "calibration_profiles": raw_config["calibration_profiles"],
    }
    manifest: dict[str, Any] = {
        "schema_version": A2H_SCHEMA_VERSION,
        "dataset_id": A2H_DATASET_ID,
        "data_version": version,
        "generation_seed": generation_seed,
        "split_seed": split_seed,
        "observation_mode": raw_config["observation_mode"],
        "timesteps": 1,
        "sensor_ids": list(SENSOR_IDS),
        "target_names": list(TARGET_NAMES),
        "target_units": raw_config["target_units"],
        "composition_total_pct": float(raw_config["composition_total_pct"]),
        "a1_reference": raw_config["a1_reference"],
        "physics": base_physics.to_dict(),
        "environment_blocks": raw_config["environment_blocks"],
        "noise_profiles": raw_config["noise_profiles"],
        "calibration_profiles": raw_config["calibration_profiles"],
        "composition_regions": raw_config["composition_regions"],
        "families": raw_config["families"],
        "conditions": [
            conditions_by_id[key].to_dict() for key in sorted(conditions_by_id)
        ],
        "observations": [observation.to_dict() for observation in observations],
        "sample_count": len(observations),
        "mixture_count": len(conditions_by_id),
        "split_counts": _split_counts(observations),
        "profile_hash": _canonical_sha256(profiles),
        "generator_config_sha256": _canonical_sha256(raw_config),
        "split_family_hash": compute_split_family_hash(observations),
        "hard_test_locked_by_default": True,
        "random_state_policy": "registered_generation_and_split_seed_only",
    }
    manifest["content_sha256"] = _content_sha256(manifest, signals)
    return manifest


def _validate_config_for_generation(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != A2H_SCHEMA_VERSION:
        raise ValueError("unsupported A2H data schema_version")
    if config.get("dataset_id") != A2H_DATASET_ID:
        raise ValueError("A2H data_config must use dataset_id=ar_he_co2")
    if not str(config.get("data_version", "")).startswith(A2H_DATA_VERSION_PREFIX):
        raise ValueError("A2H data_version must use the gf-a2h-v2 namespace")
    if config.get("timesteps") != 1 or config.get("observation_mode") != "steady_state_repeated_observation":
        raise ValueError("A2H observations must be steady-state with timesteps=1")
    if config.get("sensor_ids") != list(SENSOR_IDS) or config.get("target_names") != list(TARGET_NAMES):
        raise ValueError("A2H sensor_ids and target_names must match the frozen three-sensor contract")
    _validate_no_forbidden_keys(config)
    A2HPhysicsConfig.from_mapping(config["physics"])
    for key in ("environment_blocks", "noise_profiles", "calibration_profiles", "families"):
        if not isinstance(config.get(key), (list, dict)):
            raise ValueError(f"A2H config field {key!r} must be a list or object")


def _parse_environments(raw: Any) -> dict[str, dict[str, float]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("environment_blocks must be a non-empty list")
    result: dict[str, dict[str, float]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("each environment block must be an object")
        environment_id = str(item.get("environment_id", ""))
        if not environment_id or environment_id in result:
            raise ValueError(f"environment_id must be unique and non-empty: {environment_id!r}")
        temperature_k = float(item["temperature_k"])
        pressure_pa = float(item["pressure_pa"])
        if not math.isfinite(temperature_k) or temperature_k <= 0.0 or not math.isfinite(pressure_pa) or pressure_pa <= 0.0:
            raise ValueError(f"invalid environment values for {environment_id!r}")
        result[environment_id] = {
            "temperature_k": temperature_k,
            "pressure_pa": pressure_pa,
        }
    return result


def _select_profile_id(value: Any, *, mixture_index: int, field: str) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return value[mixture_index % len(value)]
    raise ValueError(f"{field} entries must be a non-empty profile id or list of profile ids")


def _generate_compositions(
    count: int,
    *,
    mode: str,
    rng: np.random.Generator,
    regions: Mapping[str, Any],
) -> list[tuple[float, float, float]]:
    if count < 0:
        raise ValueError("composition count must be non-negative")
    if count == 0:
        return []
    compositions: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for index in range(count):
        if mode == "mixed":
            if index < 3:
                raw = np.zeros(3, dtype=np.float64)
                raw[index] = 100.0
                candidate = tuple(float(value) for value in raw)
            elif index < 9:
                pair = ((0, 1), (0, 2), (1, 2))[(index - 3) % 3]
                ratio = (index - 2) / 8.0
                raw = np.zeros(3, dtype=np.float64)
                raw[pair[0]] = 100.0 * ratio
                raw[pair[1]] = 100.0 * (1.0 - ratio)
                candidate = _quantize_composition(raw)
            else:
                candidate = _random_interior(rng)
        elif mode == "interior":
            candidate = _random_interior(rng)
        elif mode == "binary":
            pair = ((0, 1), (0, 2), (1, 2))[index % 3]
            ratio = (index + 1) / (count + 1)
            raw = np.zeros(3, dtype=np.float64)
            raw[pair[0]] = 100.0 * ratio
            raw[pair[1]] = 100.0 * (1.0 - ratio)
            candidate = _quantize_composition(raw)
        elif mode == "pure":
            if count > 3:
                raise ValueError("pure mode cannot generate more than three unique mixtures")
            raw = np.zeros(3, dtype=np.float64)
            raw[index] = 100.0
            candidate = tuple(float(value) for value in raw)
        elif mode == "near_boundary":
            candidate = _random_near_boundary(rng, float(regions["near_boundary_max_pct"]))
        elif mode == "concentration_band":
            band = regions["concentration_band"]
            candidate = _random_concentration_band(
                rng,
                float(band["minimum_pct"]),
                float(band["maximum_pct"]),
            )
        elif mode == "simplex_sector":
            sector = regions["simplex_sector"]
            candidate = _random_simplex_sector(
                rng,
                float(sector["x_Ar_min_pct"]),
                float(sector["x_He_max_pct"]),
            )
        elif mode == "simplex_sector_and_pure":
            if index < 3:
                raw = np.zeros(3, dtype=np.float64)
                raw[index] = 100.0
                candidate = tuple(float(value) for value in raw)
            else:
                sector = regions["simplex_sector"]
                candidate = _random_simplex_sector(
                    rng,
                    float(sector["x_Ar_min_pct"]),
                    float(sector["x_He_max_pct"]),
                )
        else:
            raise ValueError(f"unsupported composition generation mode: {mode!r}")
        if candidate in seen:
            raise RuntimeError(f"composition generator produced a duplicate: {candidate}")
        seen.add(candidate)
        compositions.append(candidate)
    return compositions


def _random_interior(rng: np.random.Generator) -> tuple[float, float, float]:
    draw = rng.dirichlet(np.ones(3))
    return _quantize_composition(0.25 + 99.25 * draw)


def _random_near_boundary(rng: np.random.Generator, maximum: float) -> tuple[float, float, float]:
    if maximum <= 0.0 or maximum >= 33.0:
        raise ValueError("near_boundary_max_pct must be within (0,33)")
    zero_index = int(rng.integers(0, 3))
    raw = np.zeros(3, dtype=np.float64)
    raw[zero_index] = float(rng.uniform(0.05, maximum))
    remaining = [index for index in range(3) if index != zero_index]
    ratio = float(rng.uniform(0.15, 0.85))
    raw[remaining[0]] = (100.0 - raw[zero_index]) * ratio
    raw[remaining[1]] = 100.0 - raw[zero_index] - raw[remaining[0]]
    return _quantize_composition(raw)


def _random_concentration_band(
    rng: np.random.Generator,
    minimum: float,
    maximum: float,
) -> tuple[float, float, float]:
    if not 0.0 < minimum < maximum < 100.0:
        raise ValueError("concentration band must be strictly inside (0,100)")
    co2 = float(rng.uniform(minimum, maximum))
    ar = float(rng.uniform(0.05, 100.0 - co2 - 0.05))
    return _quantize_composition((ar, 100.0 - co2 - ar, co2))


def _random_simplex_sector(
    rng: np.random.Generator,
    ar_min: float,
    he_max: float,
) -> tuple[float, float, float]:
    if ar_min <= 0.0 or he_max <= 0.0 or ar_min + he_max >= 100.0:
        raise ValueError("simplex sector leaves no valid interior")
    ar = float(rng.uniform(ar_min, 100.0 - 0.25))
    he_upper = min(he_max, 100.0 - ar - 0.25)
    if he_upper <= 0.05:
        ar = ar_min
        he_upper = min(he_max, 100.0 - ar - 0.25)
    he = float(rng.uniform(0.05, he_upper))
    return _quantize_composition((ar, he, 100.0 - ar - he))


def _quantize_composition(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("composition must contain three values")
    first, second = round(float(values[0]), 2), round(float(values[1]), 2)
    third = round(100.0 - first - second, 2)
    result = (first, second, third)
    if any(value < 0.0 or value > 100.0 for value in result) or not math.isclose(sum(result), 100.0, abs_tol=1.0e-8):
        raise RuntimeError(f"composition quantization failed: {result}")
    return result


def _classify_composition(composition: Sequence[float]) -> tuple[str, str | None]:
    nonzero = sum(float(value) > 0.0 for value in composition)
    if nonzero == 1:
        return "pure", None
    if nonzero == 2:
        indices = [index for index, value in enumerate(composition) if float(value) > 0.0]
        names = ("Ar", "He", "CO2")
        return "binary", f"{names[indices[0]]}-{names[indices[1]]}"
    return "ternary", None


def _validate_composition(
    composition: Sequence[float],
    *,
    condition_family: str,
    binary_pair: str | None,
) -> None:
    values = tuple(float(value) for value in composition)
    if len(values) != 3 or any(not math.isfinite(value) or value < 0.0 or value > 100.0 for value in values):
        raise ValueError(f"composition must contain three finite values in [0,100], got {values}")
    if not math.isclose(sum(values), 100.0, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(f"composition must sum to 100 mol%, got {sum(values)}")
    inferred, inferred_pair = _classify_composition(values)
    if condition_family != inferred:
        raise ValueError(f"condition_family={condition_family!r} does not match composition={values}")
    if condition_family == "binary" and binary_pair != inferred_pair:
        raise ValueError(f"binary_pair={binary_pair!r} does not match composition={values}")
    if condition_family != "binary" and binary_pair is not None:
        raise ValueError("only binary conditions may specify binary_pair")


def _sensor_float_mapping(value: Any, name: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(SENSOR_IDS):
        raise ValueError(f"{name} must cover exactly {list(SENSOR_IDS)}")
    return {sensor_id: float(value[sensor_id]) for sensor_id in SENSOR_IDS}


def _noise_scales(physics: A2HPhysicsConfig) -> np.ndarray:
    return np.asarray(
        [
            physics.tof_resolution_s / math.sqrt(12.0),
            physics.thermal_voltage_resolution_v / math.sqrt(12.0),
            physics.ndir_voltage_noise_std_v,
        ],
        dtype=np.float64,
    )


def _validate_signal_bounds(signal: Sequence[float], physics: A2HPhysicsConfig, observation_id: str) -> None:
    for index, sensor_id in enumerate(SENSOR_IDS):
        lower, upper = physics.signal_bounds[sensor_id]
        value = float(signal[index])
        if value < lower or value > upper:
            raise ValueError(
                f"signal for {observation_id} is outside registered bound for {sensor_id}: {value} not in [{lower}, {upper}]"
            )


def _split_counts(observations: Sequence[A2HObservation]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for observation in observations:
        family_counts = result.setdefault(observation.split_family, {split: 0 for split in A2H_SPLITS})
        family_counts[observation.split] += 1
    return result


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(manifest)
    if manifest.get("schema_version") != A2H_SCHEMA_VERSION:
        raise ValueError("unsupported A2H manifest schema_version")
    if manifest.get("dataset_id") != A2H_DATASET_ID:
        raise ValueError("unsupported A2H manifest dataset_id")
    if not str(manifest.get("data_version", "")).startswith(A2H_DATA_VERSION_PREFIX):
        raise ValueError("A2H manifest has invalid data_version namespace")
    if manifest.get("sensor_ids") != list(SENSOR_IDS) or manifest.get("target_names") != list(TARGET_NAMES):
        raise ValueError("A2H manifest sensor or target contract mismatch")
    if manifest.get("timesteps") != 1:
        raise ValueError("A2H manifest timesteps must be 1")
    A2HPhysicsConfig.from_mapping(manifest["physics"])
    observations = manifest.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("A2H manifest observations must be a non-empty list")
    if int(manifest.get("sample_count", -1)) != len(observations):
        raise ValueError("A2H manifest sample_count does not match observations")
    parsed = tuple(A2HObservation.from_dict(value) for value in observations)
    if compute_split_family_hash(parsed) != manifest.get("split_family_hash"):
        raise ValueError("A2H manifest split_family_hash mismatch")
    expected_counts = _split_counts(parsed)
    if expected_counts != manifest.get("split_counts"):
        raise ValueError("A2H manifest split_counts mismatch")
    _validate_group_exclusivity(parsed)


def _validate_dataset_structure(dataset: A2HDataset) -> None:
    observations = dataset.observations
    expected_mixture_count = len({observation.mixture_id for observation in observations})
    if int(dataset.manifest.get("mixture_count", -1)) != expected_mixture_count:
        raise ValueError("A2H manifest mixture_count mismatch")
    physics = A2HPhysicsConfig.from_mapping(dataset.manifest["physics"])
    for index, signal in enumerate(dataset.signals):
        _validate_signal_bounds(signal, physics, observations[index].observation_id)


def _validate_group_exclusivity(observations: Sequence[A2HObservation]) -> None:
    by_family: dict[str, dict[str, set[str]]] = {}
    for observation in observations:
        family = by_family.setdefault(observation.split_family, {split: set() for split in A2H_SPLITS})
        family[observation.split].add(observation.mixture_id)
    for family_name, split_groups in by_family.items():
        for left_index, left_split in enumerate(A2H_SPLITS):
            for right_split in A2H_SPLITS[left_index + 1 :]:
                overlap = split_groups[left_split] & split_groups[right_split]
                if overlap:
                    raise ValueError(
                        f"mixture_id groups overlap within split family {family_name!r}: {sorted(overlap)}"
                    )


def _read_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return json.loads(json.dumps(config, ensure_ascii=False))
    return json.loads(Path(config).read_text(encoding="utf-8"))


def _validate_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_KEYS & set(value)
        if forbidden:
            raise ValueError(f"forbidden legacy keys: {sorted(forbidden)}")
        for child in value.values():
            _validate_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _validate_no_forbidden_keys(child)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _content_sha256(manifest: Mapping[str, Any], signals: np.ndarray) -> str:
    manifest_without_hashes = dict(manifest)
    for key in ("content_sha256",):
        manifest_without_hashes.pop(key, None)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            manifest_without_hashes,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    array = np.asarray(signals, dtype=np.float32, order="C")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "A2HCondition",
    "A2HDataset",
    "A2HObservation",
    "A2HPhysicsConfig",
    "A2H_DATASET_ID",
    "A2H_DATA_VERSION_PREFIX",
    "A2H_DEVELOPMENT_SPLITS",
    "A2H_SCHEMA_VERSION",
    "A2H_SPLITS",
    "CalibrationProfile",
    "FORBIDDEN_KEYS",
    "NoiseProfile",
    "composition_region",
    "compute_split_family_hash",
    "deterministic_a2h_signal_vector",
    "generate_a2h_dataset",
    "load_a2h_dataset",
    "load_dataset",
    "nominal_calibration_profile",
    "nominal_signal_parity",
]
