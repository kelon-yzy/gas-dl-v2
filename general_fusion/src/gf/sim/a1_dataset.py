from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from gf.dl.contracts import UnifiedSample
from gf.sim.ar_he_co2 import (
    REFERENCE_PRESSURE_PA,
    REFERENCE_TEMPERATURE_K,
    SENSOR_TYPES,
    SYSTEM_DELAY_S,
    ideal_gas_sound_speed,
    ndir_co2_voltage,
    thermal_conductivity_voltage,
    wms_thermal_conductivity,
)


A1_SCHEMA_VERSION = "gf-a1-data-1"
TARGET_NAMES = ("x_Ar_pct", "x_He_pct", "x_CO2_pct")
SENSOR_IDS = tuple(SENSOR_TYPES)
FORBIDDEN_MANIFEST_KEYS = frozenset(
    {"base_condition_id", "noise_seed", "noise_seed_index", "sequence_id"}
)


@dataclass(frozen=True)
class A1PhysicsConfig:
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
    def from_mapping(cls, raw: Mapping[str, Any]) -> "A1PhysicsConfig":
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
            raise ValueError(f"A1 physics config is missing keys: {missing}")
        bounds_raw = raw["signal_bounds"]
        if not isinstance(bounds_raw, Mapping):
            raise ValueError("signal_bounds must be a mapping")
        bounds: dict[str, tuple[float, float]] = {}
        for sensor_id in SENSOR_IDS:
            value = bounds_raw.get(sensor_id)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
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
            raise ValueError(f"A1 physics parameters must be finite and positive: {invalid}")
        if not math.isfinite(self.tcs_baseline_v):
            raise ValueError("tcs_baseline_v must be finite")
        if self.timesteps != 1:
            raise ValueError("A1 formal benchmark uses steady-state observations with timesteps=1")
        if not math.isclose(self.temperature_k, REFERENCE_TEMPERATURE_K, abs_tol=1e-9):
            raise ValueError("A1 temperature is frozen at 298.15 K")
        if not math.isclose(self.pressure_pa, REFERENCE_PRESSURE_PA, abs_tol=1e-6):
            raise ValueError("A1 pressure is frozen at 101325 Pa")
        if self.jacobian_step_pct >= 0.25:
            raise ValueError("jacobian_step_pct must be smaller than the interior composition margin")
        if set(self.signal_bounds) != set(SENSOR_IDS):
            raise ValueError(f"signal_bounds must cover exactly {list(SENSOR_IDS)}")

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


DEFAULT_A1_PHYSICS = A1PhysicsConfig.from_mapping(
    {
        "temperature_k": REFERENCE_TEMPERATURE_K,
        "pressure_pa": REFERENCE_PRESSURE_PA,
        "acoustic_path_length_m": 0.2,
        "ndir_optical_path_length_m": 0.01,
        "timesteps": 1,
        "dt_s": 1.0,
        "tof_delay_s": SYSTEM_DELAY_S,
        "ndir_baseline_v": 2.5,
        "ndir_effective_absorbance_per_co2_percent": 0.04269004521460337,
        "tcs_baseline_v": 1.1,
        "tcs_response_v_per_w_m_k": 15.0,
        "tcs_reference_conductivity_w_m_k": 0.026,
        "tof_resolution_s": 1.0e-8,
        "thermal_voltage_resolution_v": 5.0 / 65_536.0,
        "ndir_voltage_noise_std_v": 0.016,
        "jacobian_step_pct": 0.01,
        "signal_bounds": {
            "ultrasonic_tof": [0.0, 0.01],
            "thermal_conductivity_voltage": [-2.0, 4.0],
            "ndir_co2_voltage": [0.0, 2.5],
        },
    }
)


@dataclass(frozen=True)
class A1Condition:
    mixture_id: str
    x_ar_pct: float
    x_he_pct: float
    x_co2_pct: float
    split: str
    condition_family: str
    binary_pair: str | None = None

    def __post_init__(self) -> None:
        if not self.mixture_id:
            raise ValueError("mixture_id must be non-empty")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test, got {self.split!r}")
        if self.condition_family not in {"binary", "ternary"}:
            raise ValueError(f"unsupported condition_family {self.condition_family!r}")
        values = (self.x_ar_pct, self.x_he_pct, self.x_co2_pct)
        if any(not math.isfinite(value) or value < 0.0 or value > 100.0 for value in values):
            raise ValueError(f"composition must be finite and within [0,100], got {values}")
        if not math.isclose(sum(values), 100.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"composition must sum to 100 mol%, got {sum(values)}")
        if self.condition_family == "binary":
            if self.binary_pair is None:
                raise ValueError("binary conditions must specify binary_pair")
            if sum(value > 0.0 for value in values) != 2:
                raise ValueError("binary conditions must contain exactly two non-zero components")
        elif self.binary_pair is not None:
            raise ValueError("ternary conditions cannot specify binary_pair")

    @property
    def composition(self) -> tuple[float, float, float]:
        return self.x_ar_pct, self.x_he_pct, self.x_co2_pct

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "mixture_id": self.mixture_id,
            "x_Ar_pct": self.x_ar_pct,
            "x_He_pct": self.x_he_pct,
            "x_CO2_pct": self.x_co2_pct,
            "split": self.split,
            "condition_family": self.condition_family,
        }
        if self.binary_pair is not None:
            result["binary_pair"] = self.binary_pair
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "A1Condition":
        return cls(
            mixture_id=str(raw["mixture_id"]),
            x_ar_pct=float(raw["x_Ar_pct"]),
            x_he_pct=float(raw["x_He_pct"]),
            x_co2_pct=float(raw["x_CO2_pct"]),
            split=str(raw["split"]),
            condition_family=str(raw["condition_family"]),
            binary_pair=str(raw["binary_pair"]) if raw.get("binary_pair") is not None else None,
        )


@dataclass(frozen=True)
class A1Dataset:
    conditions: tuple[A1Condition, ...]
    signals: np.ndarray
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.signals, dtype=np.float32)
        if values.shape != (len(self.conditions), len(SENSOR_IDS)):
            raise ValueError(
                f"signals must have shape ({len(self.conditions)}, {len(SENSOR_IDS)}), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("signals must contain only finite values")
        if len({condition.mixture_id for condition in self.conditions}) != len(self.conditions):
            raise ValueError("conditions must have unique mixture_id values")
        values.setflags(write=False)
        object.__setattr__(self, "signals", values)
        object.__setattr__(self, "conditions", tuple(self.conditions))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(condition.mixture_id for condition in self.conditions)

    def samples(self) -> list[UnifiedSample]:
        samples: list[UnifiedSample] = []
        for index, condition in enumerate(self.conditions):
            signal_values = self.signals[index]
            signals = tuple(
                np.array([[signal_values[sensor_index]]], dtype=np.float32)
                for sensor_index in range(len(SENSOR_IDS))
            )
            target = np.array(condition.composition, dtype=np.float32)
            metadata: dict[str, str | float] = {
                "mixture_id": condition.mixture_id,
                "split": condition.split,
                "condition_family": condition.condition_family,
                "x_Ar_pct": condition.x_ar_pct,
                "x_He_pct": condition.x_he_pct,
                "x_CO2_pct": condition.x_co2_pct,
            }
            if condition.binary_pair is not None:
                metadata["binary_pair"] = condition.binary_pair
            samples.append(
                UnifiedSample(
                    signals=signals,
                    sensor_id=SENSOR_IDS,
                    sensor_type=tuple(SENSOR_TYPES[sensor_id] for sensor_id in SENSOR_IDS),
                    valid_mask=tuple(np.ones_like(signal, dtype=np.bool_) for signal in signals),
                    quality=tuple(np.ones(1, dtype=np.float32) for _ in SENSOR_IDS),
                    time=tuple(np.array([0.0], dtype=np.float64) for _ in SENSOR_IDS),
                    target=target,
                    target_mask=np.ones(len(TARGET_NAMES), dtype=np.bool_),
                    group_id=condition.mixture_id,
                    dataset_id="ar_he_co2",
                    metadata=metadata,
                )
            )
        return samples


def generate_a1_conditions(
    *,
    binary_per_pair: int,
    ternary_count: int,
    generation_seed: int,
    id_prefix: str = "a1",
) -> tuple[A1Condition, ...]:
    if binary_per_pair <= 0:
        raise ValueError("binary_per_pair must be positive")
    if ternary_count <= 0:
        raise ValueError("ternary_count must be positive")
    if generation_seed < 0:
        raise ValueError("generation_seed must be non-negative")

    conditions: list[A1Condition] = []
    seen_compositions: set[tuple[float, float, float]] = set()
    pairs = (("Ar", "He"), ("Ar", "CO2"), ("He", "CO2"))
    component_names = ("Ar", "He", "CO2")

    for pair in pairs:
        for sample_index in range(binary_per_pair):
            ratio = (sample_index + 1) / (binary_per_pair + 1)
            composition = {name: 0.0 for name in component_names}
            composition[pair[0]] = 100.0 * ratio
            composition[pair[1]] = 100.0 * (1.0 - ratio)
            values = _quantize_composition(
                (composition["Ar"], composition["He"], composition["CO2"])
            )
            if values in seen_compositions:
                raise RuntimeError(f"duplicate binary composition generated: {values}")
            seen_compositions.add(values)
            conditions.append(
                A1Condition(
                    mixture_id=f"{id_prefix}-binary-{pair[0].lower()}-{pair[1].lower()}-{sample_index + 1:04d}",
                    x_ar_pct=values[0],
                    x_he_pct=values[1],
                    x_co2_pct=values[2],
                    split="train",
                    condition_family="binary",
                    binary_pair=f"{pair[0]}-{pair[1]}",
                )
            )

    rng = np.random.default_rng(generation_seed)
    while len(conditions) < 3 * binary_per_pair + ternary_count:
        draw = rng.dirichlet(np.ones(3))
        values = _quantize_composition(tuple(0.25 + 99.25 * draw))
        if min(values) <= 0.0 or values in seen_compositions:
            continue
        seen_compositions.add(values)
        ternary_index = len(conditions) - 3 * binary_per_pair + 1
        conditions.append(
            A1Condition(
                mixture_id=f"{id_prefix}-ternary-{ternary_index:04d}",
                x_ar_pct=values[0],
                x_he_pct=values[1],
                x_co2_pct=values[2],
                split="train",
                condition_family="ternary",
            )
        )
    return tuple(conditions)


def assign_a1_splits(
    conditions: Sequence[A1Condition],
    *,
    split_seed: int,
) -> tuple[A1Condition, ...]:
    if not conditions:
        raise ValueError("conditions must be non-empty")
    ordered = sorted(conditions, key=lambda condition: condition.mixture_id)
    group_ids = np.array([condition.mixture_id for condition in ordered], dtype=object)
    first_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=split_seed)
    all_indices = np.arange(len(ordered))
    remaining_indices, test_indices = next(first_splitter.split(all_indices, groups=group_ids))
    remaining_ids = group_ids[remaining_indices]
    second_splitter = GroupShuffleSplit(n_splits=1, test_size=3.0 / 17.0, random_state=split_seed)
    train_local, val_local = next(second_splitter.split(remaining_indices, groups=remaining_ids))
    split_by_id = {str(group_ids[index]): "test" for index in test_indices}
    split_by_id.update({str(remaining_ids[index]): "val" for index in val_local})
    split_by_id.update({str(remaining_ids[index]): "train" for index in train_local})
    return tuple(
        A1Condition(
            mixture_id=condition.mixture_id,
            x_ar_pct=condition.x_ar_pct,
            x_he_pct=condition.x_he_pct,
            x_co2_pct=condition.x_co2_pct,
            split=split_by_id[condition.mixture_id],
            condition_family=condition.condition_family,
            binary_pair=condition.binary_pair,
        )
        for condition in ordered
    )


def deterministic_signal_vector(
    composition: Sequence[float],
    physics: A1PhysicsConfig = DEFAULT_A1_PHYSICS,
) -> np.ndarray:
    if len(composition) != 3:
        raise ValueError("composition must contain Ar, He, and CO2 percentages")
    x_ar_pct, x_he_pct, x_co2_pct = (float(value) for value in composition)
    fractions = {
        "Ar": x_ar_pct / 100.0,
        "He": x_he_pct / 100.0,
        "CO2": x_co2_pct / 100.0,
    }
    speed = ideal_gas_sound_speed(fractions, physics.temperature_k)
    tof = physics.acoustic_path_length_m / speed + physics.tof_delay_s
    conductivity = wms_thermal_conductivity(fractions)
    thermal_voltage = thermal_conductivity_voltage(
        conductivity,
        baseline_v=physics.tcs_baseline_v,
        response_v_per_w_m_k=physics.tcs_response_v_per_w_m_k,
        reference_conductivity_w_m_k=physics.tcs_reference_conductivity_w_m_k,
    )
    ndir_voltage = ndir_co2_voltage(
        x_co2_pct,
        physics.pressure_pa,
        physics.temperature_k,
        effective_absorbance_per_co2_percent=physics.ndir_effective_absorbance_per_co2_percent,
        baseline_v=physics.ndir_baseline_v,
    )
    values = np.array([tof, thermal_voltage, ndir_voltage], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"deterministic signal contains non-finite values: {values}")
    return values


def generate_dataset(
    output_dir: str | Path,
    *,
    binary_per_pair: int,
    ternary_count: int,
    generation_seed: int,
    split_seed: int,
    data_version: str,
    physics: A1PhysicsConfig = DEFAULT_A1_PHYSICS,
) -> A1Dataset:
    if not data_version:
        raise ValueError("data_version must be non-empty")
    physics.validate()
    conditions = assign_a1_splits(
        generate_a1_conditions(
            binary_per_pair=binary_per_pair,
            ternary_count=ternary_count,
            generation_seed=generation_seed,
        ),
        split_seed=split_seed,
    )
    rng = np.random.default_rng(generation_seed)
    noise_scale = np.array(
        [
            physics.tof_resolution_s / math.sqrt(12.0),
            physics.thermal_voltage_resolution_v / math.sqrt(12.0),
            physics.ndir_voltage_noise_std_v,
        ],
        dtype=np.float64,
    )
    signals = np.vstack(
        [
            deterministic_signal_vector(condition.composition, physics)
            + rng.normal(loc=0.0, scale=noise_scale)
            for condition in conditions
        ]
    ).astype(np.float32)
    manifest = {
        "schema_version": A1_SCHEMA_VERSION,
        "dataset_id": "ar_he_co2",
        "data_version": data_version,
        "generation_seed": int(generation_seed),
        "split_seed": int(split_seed),
        "sample_count": len(conditions),
        "binary_count": 3 * binary_per_pair,
        "ternary_count": ternary_count,
        "observation_mode": "steady_state_single_observation",
        "sensor_ids": list(SENSOR_IDS),
        "target_names": list(TARGET_NAMES),
        "physics": physics.to_dict(),
        "conditions": [condition.to_dict() for condition in conditions],
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path / "observations.npz", signals=signals)
    manifest["content_sha256"] = _content_sha256(manifest, signals)
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return load_dataset(output_path)


def load_dataset(dataset_dir: str | Path) -> A1Dataset:
    dataset_path = Path(dataset_dir)
    manifest_path = dataset_path / "manifest.json"
    observations_path = dataset_path / "observations.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_keys(manifest)
    if manifest.get("schema_version") != A1_SCHEMA_VERSION:
        raise ValueError(f"unsupported A1 schema version: {manifest.get('schema_version')!r}")
    if manifest.get("dataset_id") != "ar_he_co2":
        raise ValueError(f"unsupported A1 dataset_id: {manifest.get('dataset_id')!r}")
    if manifest.get("sensor_ids") != list(SENSOR_IDS):
        raise ValueError("A1 sensor_ids do not match the frozen contract")
    if manifest.get("target_names") != list(TARGET_NAMES):
        raise ValueError("A1 target_names do not match the frozen contract")
    conditions = tuple(A1Condition.from_dict(raw) for raw in manifest["conditions"])
    with np.load(observations_path, allow_pickle=False) as archive:
        signals = np.array(archive["signals"], dtype=np.float32, copy=True)
    expected_count = int(manifest["sample_count"])
    if len(conditions) != expected_count or signals.shape != (expected_count, len(SENSOR_IDS)):
        raise ValueError(
            f"dataset shape mismatch: conditions={len(conditions)}, signals={signals.shape}, expected={expected_count}"
        )
    expected_hash = manifest.get("content_sha256")
    if not isinstance(expected_hash, str) or expected_hash != _content_sha256(manifest, signals):
        raise ValueError("A1 dataset content_sha256 mismatch")
    dataset = A1Dataset(conditions=conditions, signals=signals, manifest=manifest)
    _validate_split_counts(dataset)
    return dataset


def load_dataset_splits(
    dataset_dir: str | Path,
    *,
    allowed_splits: Sequence[str],
) -> A1Dataset:
    """Load only the manifest rows in an explicit split allowlist."""

    requested = tuple(dict.fromkeys(str(value) for value in allowed_splits))
    if not requested or any(value not in {"train", "val", "test"} for value in requested):
        raise ValueError("allowed_splits must contain only train, val, and test")
    dataset_path = Path(dataset_dir)
    manifest_path = dataset_path / "manifest.json"
    observations_path = dataset_path / "observations.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_manifest_keys(manifest)
    if manifest.get("schema_version") != A1_SCHEMA_VERSION:
        raise ValueError(f"unsupported A1 schema version: {manifest.get('schema_version')!r}")
    if manifest.get("dataset_id") != "ar_he_co2":
        raise ValueError(f"unsupported A1 dataset_id: {manifest.get('dataset_id')!r}")
    if manifest.get("sensor_ids") != list(SENSOR_IDS):
        raise ValueError("A1 sensor_ids do not match the frozen contract")
    if manifest.get("target_names") != list(TARGET_NAMES):
        raise ValueError("A1 target_names do not match the frozen contract")
    conditions = tuple(A1Condition.from_dict(raw) for raw in manifest["conditions"])
    with np.load(observations_path, allow_pickle=False) as archive:
        signals = np.array(archive["signals"], dtype=np.float32, copy=True)
    expected_count = int(manifest["sample_count"])
    if len(conditions) != expected_count or signals.shape != (expected_count, len(SENSOR_IDS)):
        raise ValueError(
            f"dataset shape mismatch: conditions={len(conditions)}, signals={signals.shape}, expected={expected_count}"
        )
    expected_hash = manifest.get("content_sha256")
    if not isinstance(expected_hash, str) or expected_hash != _content_sha256(manifest, signals):
        raise ValueError("A1 dataset content hash does not match manifest")
    selected_indices = [
        index for index, condition in enumerate(conditions) if condition.split in requested
    ]
    if not selected_indices:
        raise ValueError(f"allowed_splits have no rows: {requested}")
    return A1Dataset(
        conditions=tuple(conditions[index] for index in selected_indices),
        signals=signals[selected_indices],
        manifest=manifest,
    )


def _quantize_composition(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError("composition must contain three values")
    first, second = round(float(values[0]), 2), round(float(values[1]), 2)
    third = round(100.0 - first - second, 2)
    result = (first, second, third)
    if not math.isclose(sum(result), 100.0, rel_tol=0.0, abs_tol=1e-8):
        raise RuntimeError(f"composition quantization failed: {result}")
    return result


def _content_sha256(manifest: Mapping[str, Any], signals: np.ndarray) -> str:
    manifest_without_hash = dict(manifest)
    manifest_without_hash.pop("content_sha256", None)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            manifest_without_hash,
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


def _validate_manifest_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = FORBIDDEN_MANIFEST_KEYS & set(value)
        if forbidden:
            raise ValueError(f"forbidden A1 manifest keys: {sorted(forbidden)}")
        for child in value.values():
            _validate_manifest_keys(child)
    elif isinstance(value, list):
        for child in value:
            _validate_manifest_keys(child)


def _validate_split_counts(dataset: A1Dataset) -> None:
    counts = {
        split: sum(condition.split == split for condition in dataset.conditions)
        for split in ("train", "val", "test")
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"A1 split contains no samples: {counts}")
    expected = {"train", "val", "test"}
    actual = {condition.split for condition in dataset.conditions}
    if actual != expected:
        raise ValueError(f"A1 splits must be exactly {sorted(expected)}, got {sorted(actual)}")
