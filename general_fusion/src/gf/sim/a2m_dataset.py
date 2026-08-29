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

from gf.dl.contracts import UnifiedSample
from gf.sim.a1_dataset import SENSOR_IDS, TARGET_NAMES
from gf.sim.a2h_dataset import (
    A2HPhysicsConfig,
    CalibrationProfile,
    NoiseProfile,
    _classify_composition,
    _generate_compositions,
    _noise_scales,
    _parse_environments,
    _validate_signal_bounds,
    deterministic_a2h_signal_vector,
)
from gf.sim.ar_he_co2 import SENSOR_TYPES


A2M_SCHEMA_VERSION = "gf-a2m-data-1"
A2M_DATA_VERSION_PREFIX = "gf-a2m-v1-"
A2M_SPLIT = "formal"
A2M_AXES = ("iid", "calibration", "environment", "joint", "noise", "composition")
A2M_PRIMARY_AXES = ("iid", "calibration", "environment", "joint", "noise")
A2M_FORBIDDEN_KEYS = frozenset(
    {"base_condition_id", "noise_seed_index", "noise_seed", "sequence_id"}
)


class A2MTestLockError(ValueError):
    """Raised when the formal holdout is requested before the formal unlock."""

    __test__ = False


@dataclass(frozen=True)
class A2MObservation:
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
        if not self.observation_id or not self.mixture_id:
            raise ValueError("observation_id and mixture_id must be non-empty")
        if self.split != A2M_SPLIT or self.split_family not in A2M_AXES:
            raise ValueError(f"A2M observation must use split=formal and axis in {A2M_AXES}")
        if not self.environment_id or not self.calibration_profile_id or not self.noise_profile_id:
            raise ValueError("A2M profile IDs must be non-empty")
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

    @property
    def composition(self) -> tuple[float, float, float]:
        return self.x_ar_pct, self.x_he_pct, self.x_co2_pct

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
    def from_dict(cls, raw: Mapping[str, Any]) -> "A2MObservation":
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
class A2MDataset:
    observations: tuple[A2MObservation, ...]
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
        mixture_ids = [observation.mixture_id for observation in self.observations]
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("A2M observation_id values must be unique")
        if len(set(mixture_ids)) != len(mixture_ids):
            raise ValueError("A2M formal holdout must use one observation per mixture_id")
        values.setflags(write=False)
        object.__setattr__(self, "signals", values)
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "manifest", MappingProxyType(dict(self.manifest)))

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(observation.mixture_id for observation in self.observations)

    def indices(self, *, axis: str | None = None) -> np.ndarray:
        if axis is not None and axis not in A2M_AXES:
            raise ValueError(f"axis must be one of {A2M_AXES}")
        return np.asarray(
            [
                index
                for index, observation in enumerate(self.observations)
                if axis is None or observation.split_family == axis
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
                    dataset_id="ar_he_co2_a2m",
                    metadata=metadata,
                )
            )
        return samples


def validate_a2m_data_config(config: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(config)
    if config.get("schema_version") != A2M_SCHEMA_VERSION:
        raise ValueError("A2M data schema_version is unsupported")
    if config.get("dataset_id") != "ar_he_co2":
        raise ValueError("A2M data_config must use dataset_id=ar_he_co2")
    if not str(config.get("data_version", "")).startswith(A2M_DATA_VERSION_PREFIX):
        raise ValueError("A2M data_version must use the gf-a2m-v1 namespace")
    if config.get("timesteps") != 1 or config.get("observation_mode") != "steady_state_single_observation":
        raise ValueError("A2M must use steady-state T=1 observations")
    if config.get("sensor_ids") != list(SENSOR_IDS) or config.get("target_names") != list(TARGET_NAMES):
        raise ValueError("A2M sensor and target order must match the frozen contract")
    if float(config.get("composition_total_pct", 0.0)) != 100.0:
        raise ValueError("A2M composition_total_pct must be 100")
    source = config.get("a2h_development_source")
    if not isinstance(source, Mapping):
        raise ValueError("a2h_development_source must be an object")
    for key in (
        "config_path",
        "config_sha256",
        "manifest_path",
        "schema_version",
        "data_version",
        "content_sha256",
        "split_family_hash",
    ):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ValueError(f"a2h_development_source.{key} must be non-empty")
    for key in ("config_sha256", "content_sha256", "split_family_hash"):
        _validate_hash(str(source[key]), f"a2h_development_source.{key}")
    if source.get("schema_version") != "gf-a2h-data-2" or not str(source["data_version"]).startswith("gf-a2h-v2-"):
        raise ValueError("A2M development source must be the frozen A2H v2 dataset")
    if source.get("allowed_splits") != ["train", "val", "stress_val"]:
        raise ValueError("A2M development source may expose only train, val, and stress_val")
    holdout = config.get("formal_holdout")
    if not isinstance(holdout, Mapping):
        raise ValueError("formal_holdout must be an object")
    if holdout.get("split") != A2M_SPLIT:
        raise ValueError("A2M formal holdout split is frozen as formal")
    prefix = holdout.get("mixture_id_prefix")
    if not isinstance(prefix, str) or not prefix.startswith("a2m-formal-"):
        raise ValueError("A2M formal holdout mixture IDs must use the a2m-formal namespace")
    if holdout.get("axes") != list(A2M_AXES) or holdout.get("primary_axes") != list(A2M_PRIMARY_AXES):
        raise ValueError("A2M holdout axes are not frozen")
    repeat_count = holdout.get("repeat_count")
    if not isinstance(repeat_count, int) or isinstance(repeat_count, bool) or repeat_count != 1:
        raise ValueError("A2M formal holdout repeat_count must be one")
    families = holdout.get("families")
    if not isinstance(families, Mapping) or set(families) != set(A2M_AXES):
        raise ValueError("A2M formal holdout families must cover all registered axes")
    for axis in A2M_AXES:
        family = families[axis]
        if not isinstance(family, Mapping):
            raise ValueError(f"A2M formal family {axis!r} must be an object")
        count = family.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"A2M formal family {axis!r} count must be positive")
        for key in ("composition_mode", "environment_id", "calibration_profile_id", "noise_profile_id"):
            if not isinstance(family.get(key), str) or not family[key]:
                raise ValueError(f"A2M formal family {axis!r}.{key} must be non-empty")


def generate_a2m_formal_holdout(
    output_dir: str | Path,
    *,
    config: Mapping[str, Any] | str | Path,
    project_root: str | Path | None = None,
) -> A2MDataset:
    raw_config = _read_config(config)
    validate_a2m_data_config(raw_config)
    generator_config_hash = _canonical_sha256(raw_config)
    root = Path(project_root or ".").resolve()
    source_config_path = root / str(raw_config["a2h_development_source"]["config_path"])
    source_config = _read_config(source_config_path)
    source_hash = _sha256_file(source_config_path)
    if source_hash != raw_config["a2h_development_source"]["config_sha256"]:
        raise ValueError("A2H source config hash does not match the A2M data contract")
    base_physics = A2HPhysicsConfig.from_mapping(source_config["physics"])
    environments = _parse_environments(source_config["environment_blocks"])
    calibrations = {
        profile.calibration_profile_id: profile
        for profile in (CalibrationProfile.from_mapping(value) for value in source_config["calibration_profiles"])
    }
    noises = {
        profile.noise_profile_id: profile
        for profile in (NoiseProfile.from_mapping(value) for value in source_config["noise_profiles"])
    }
    regions = source_config["composition_regions"]
    families = raw_config["formal_holdout"]["families"]
    records: list[A2MObservation] = []
    signal_values: list[np.ndarray] = []
    used_mixture_ids: set[str] = set()
    used_observation_ids: set[str] = set()
    batch_effects: dict[tuple[str, str, str, str], np.ndarray] = {}

    for axis_index, axis in enumerate(A2M_AXES):
        family = families[axis]
        family_rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(raw_config["generation_seed"]), int(raw_config["split_seed"]), axis_index]
            )
        )
        compositions = _generate_compositions(
            int(family["count"]),
            mode=str(family["composition_mode"]),
            rng=family_rng,
            regions=regions,
        )
        environment_id = str(family["environment_id"])
        calibration_id = str(family["calibration_profile_id"])
        noise_id = str(family["noise_profile_id"])
        if environment_id not in environments:
            raise ValueError(f"unknown A2M environment_id {environment_id!r}")
        if calibration_id not in calibrations:
            raise ValueError(f"unknown A2M calibration_profile_id {calibration_id!r}")
        if noise_id not in noises:
            raise ValueError(f"unknown A2M noise_profile_id {noise_id!r}")
        environment = environments[environment_id]
        calibration = calibrations[calibration_id]
        noise_profile = noises[noise_id]
        effective_physics = base_physics.__class__.from_mapping(
            {
                **base_physics.to_dict(),
                "temperature_k": environment["temperature_k"],
                "pressure_pa": environment["pressure_pa"],
            }
        )
        batch_key = (axis, environment_id, calibration_id, noise_id)
        batch_effects[batch_key] = (
            family_rng.normal(size=len(SENSOR_IDS))
            * _noise_scales(effective_physics)
            * noise_profile.batch_scale
        )
        for mixture_index, composition in enumerate(compositions, start=1):
            mixture_id = f"{raw_config['formal_holdout']['mixture_id_prefix']}{axis}-m{mixture_index:04d}"
            if mixture_id in used_mixture_ids:
                raise ValueError(f"duplicate A2M mixture_id: {mixture_id}")
            used_mixture_ids.add(mixture_id)
            condition_family, binary_pair = _classify_composition(composition)
            base_signal = deterministic_a2h_signal_vector(
                composition,
                physics=effective_physics,
                calibration=calibration,
            )
            for repeat_index in range(int(raw_config["formal_holdout"]["repeat_count"])):
                observation_id = f"{mixture_id}-o{repeat_index + 1:02d}"
                if observation_id in used_observation_ids:
                    raise ValueError(f"duplicate A2M observation_id: {observation_id}")
                used_observation_ids.add(observation_id)
                correlated = (
                    family_rng.normal()
                    * _noise_scales(effective_physics)
                    * noise_profile.correlated_scale
                    * np.asarray(noise_profile.correlation_vector, dtype=np.float64)
                )
                white = (
                    family_rng.normal(size=len(SENSOR_IDS))
                    * _noise_scales(effective_physics)
                    * noise_profile.white_scale
                )
                signal = base_signal + batch_effects[batch_key] + correlated + white
                _validate_signal_bounds(signal, effective_physics, observation_id)
                records.append(
                    A2MObservation(
                        observation_id=observation_id,
                        mixture_id=mixture_id,
                        x_ar_pct=composition[0],
                        x_he_pct=composition[1],
                        x_co2_pct=composition[2],
                        split_family=axis,
                        split=A2M_SPLIT,
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
    profiles = {
        "environment_blocks": source_config["environment_blocks"],
        "calibration_profiles": source_config["calibration_profiles"],
        "noise_profiles": source_config["noise_profiles"],
    }
    manifest: dict[str, Any] = {
        "schema_version": A2M_SCHEMA_VERSION,
        "dataset_id": "ar_he_co2",
        "data_version": raw_config["data_version"],
        "generation_seed": int(raw_config["generation_seed"]),
        "split_seed": int(raw_config["split_seed"]),
        "observation_mode": raw_config["observation_mode"],
        "timesteps": 1,
        "sensor_ids": list(SENSOR_IDS),
        "target_names": list(TARGET_NAMES),
        "target_units": raw_config["target_units"],
        "composition_total_pct": float(raw_config["composition_total_pct"]),
        "a2h_development_source": raw_config["a2h_development_source"],
        "formal_holdout": raw_config["formal_holdout"],
        "generator_config_sha256": generator_config_hash,
        "source_config_sha256": source_hash,
        "physics": base_physics.to_dict(),
        "profile_hash": _canonical_sha256(profiles),
        "split_hash": compute_a2m_split_hash(observations),
        "observations": [observation.to_dict() for observation in observations],
        "sample_count": len(observations),
        "mixture_count": len(used_mixture_ids),
        "split_counts": {axis: sum(observation.split_family == axis for observation in observations) for axis in A2M_AXES},
        "random_state_policy": "registered_generation_and_split_seed_only",
    }
    manifest["content_sha256"] = _content_sha256(manifest, signals)
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
    return load_a2m_dataset(output_path, include_formal=True)


def load_a2m_dataset(
    dataset_dir: str | Path,
    *,
    include_formal: bool = False,
) -> A2MDataset:
    if not include_formal:
        raise A2MTestLockError("A2M formal holdout is locked; pass include_formal=True only after formal unlock")
    dataset_path = Path(dataset_dir)
    manifest = json.loads((dataset_path / "manifest.json").read_text(encoding="utf-8"))
    _validate_manifest(manifest)
    with np.load(dataset_path / "observations.npz", allow_pickle=False) as archive:
        signals = np.asarray(archive["signals"], dtype=np.float32).copy()
    observations = tuple(A2MObservation.from_dict(value) for value in manifest["observations"])
    if signals.shape != (len(observations), len(SENSOR_IDS)):
        raise ValueError("A2M dataset shape does not match the manifest")
    if manifest.get("content_sha256") != _content_sha256(manifest, signals):
        raise ValueError("A2M dataset content_sha256 mismatch")
    dataset = A2MDataset(observations=observations, signals=signals, manifest=manifest)
    _validate_signal_bounds_from_manifest(dataset)
    return dataset


def compute_a2m_split_hash(
    value: A2MDataset | Mapping[str, Any] | Sequence[A2MObservation],
) -> str:
    if isinstance(value, A2MDataset):
        observations = value.observations
    elif isinstance(value, Mapping):
        raw_observations = value.get("observations")
        if not isinstance(raw_observations, list):
            raise ValueError("A2M manifest observations must be a list")
        observations = tuple(A2MObservation.from_dict(item) for item in raw_observations)
    else:
        observations = tuple(value)
    assignments = sorted(
        {
            (observation.split_family, observation.mixture_id, observation.split)
            for observation in observations
        }
    )
    return _canonical_sha256(
        {
            "schema_version": "gf-a2m-split-1",
            "assignments": [
                {"axis": axis, "mixture_id": mixture_id, "split": split}
                for axis, mixture_id, split in assignments
            ],
        }
    )


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    _validate_no_forbidden_keys(manifest)
    if manifest.get("schema_version") != A2M_SCHEMA_VERSION:
        raise ValueError("unsupported A2M manifest schema_version")
    if manifest.get("dataset_id") != "ar_he_co2":
        raise ValueError("unsupported A2M manifest dataset_id")
    if not str(manifest.get("data_version", "")).startswith(A2M_DATA_VERSION_PREFIX):
        raise ValueError("A2M manifest has invalid data_version namespace")
    if manifest.get("sensor_ids") != list(SENSOR_IDS) or manifest.get("target_names") != list(TARGET_NAMES):
        raise ValueError("A2M manifest sensor or target contract mismatch")
    if manifest.get("timesteps") != 1 or manifest.get("formal_holdout", {}).get("split") != A2M_SPLIT:
        raise ValueError("A2M manifest must describe the formal T=1 holdout")
    observations = manifest.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("A2M manifest observations must be a non-empty list")
    parsed = tuple(A2MObservation.from_dict(value) for value in observations)
    if compute_a2m_split_hash(parsed) != manifest.get("split_hash"):
        raise ValueError("A2M manifest split_hash mismatch")
    expected_counts = {axis: sum(observation.split_family == axis for observation in parsed) for axis in A2M_AXES}
    if expected_counts != manifest.get("split_counts"):
        raise ValueError("A2M manifest split_counts mismatch")
    if int(manifest.get("sample_count", -1)) != len(parsed):
        raise ValueError("A2M manifest sample_count does not match observations")
    if int(manifest.get("mixture_count", -1)) != len({observation.mixture_id for observation in parsed}):
        raise ValueError("A2M manifest mixture_count does not match observations")
    _validate_hash(str(manifest.get("content_sha256", "")), "content_sha256")
    _validate_hash(str(manifest.get("profile_hash", "")), "profile_hash")
    _validate_hash(str(manifest.get("generator_config_sha256", "")), "generator_config_sha256")
    _validate_hash(str(manifest.get("source_config_sha256", "")), "source_config_sha256")


def _validate_signal_bounds_from_manifest(dataset: A2MDataset) -> None:
    physics = A2HPhysicsConfig.from_mapping(dataset.manifest["physics"])
    for signal, observation in zip(dataset.signals, dataset.observations, strict=True):
        _validate_signal_bounds(signal, physics, observation.observation_id)


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


def _read_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return json.loads(json.dumps(config, ensure_ascii=False))
    return json.loads(Path(config).read_text(encoding="utf-8"))


def _validate_no_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = A2M_FORBIDDEN_KEYS & set(value)
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _validate_hash(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


__all__ = [
    "A2M_AXES",
    "A2M_DATA_VERSION_PREFIX",
    "A2MObservation",
    "A2MDataset",
    "A2M_FORBIDDEN_KEYS",
    "A2M_PRIMARY_AXES",
    "A2M_SCHEMA_VERSION",
    "A2MTestLockError",
    "compute_a2m_split_hash",
    "generate_a2m_formal_holdout",
    "load_a2m_dataset",
    "validate_a2m_data_config",
]
