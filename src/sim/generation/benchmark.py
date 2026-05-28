from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim.core.ids import BenchmarkDatasetId
from sim.core.schema import (
    COMPONENT_FIELDS,
    CONDITION_GRID_FIELDS,
    MULTI_PATH_PHASES,
    SEQUENCE_INDEX_FIELDS,
    SEQUENCE_LABEL_FIELDS,
    SLOW_CHANNELS,
    SLOW_MODAL_GROUPS,
    SLOW_SEQUENCE_FIELDS,
    SPLIT_FIELDS,
    SPLIT_NAMES,
    VALID_STORAGE_FORMATS,
)
from sim.generation.conditions import build_label_rows, generate_condition_rows
from sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    HITRAN_ABSORPTION_BACKEND,
    VALID_OPTICAL_ABSORPTION_BACKENDS,
    hitran_manifest_metadata,
    validate_hitran_benchmark_cache,
)
from sim.generation.slow import build_sequence_arrays
from sim.generation.waveforms import FiberMicSpec, WaveformSpec
from sim.packaging.arrays import write_arrays
from sim.packaging.index import build_sequence_index_rows
from sim.packaging.io import write_csv, write_json
from sim.packaging.manifest import build_manifest
from sim.packaging.scalers import fit_z_score_scalers
from sim.packaging.splits import build_default_split_rows
from sim.validation.integrity import validate_benchmark_assets


DEFAULT_WAVEFORM_PATH_LMS = (0.20, 0.25, 0.30, 0.35, 0.40)
DEFAULT_HITRAN_CACHE_ROOT = "data/hitran_cache"


@dataclass(frozen=True, slots=True)
class BenchmarkGenerationSpec:
    dataset_slug: str
    sequence_count: int
    seed: int
    timesteps: int = 128
    dt_s: float = 0.5
    storage: str = "memmap"
    multi_path_phase: str = "steady"
    stage_profile: str = "standard_exposure"
    sampling_strategy: str = "lhs"
    path_lms: tuple[float, ...] = DEFAULT_WAVEFORM_PATH_LMS
    optical_absorption_backend: str = HITRAN_ABSORPTION_BACKEND
    hitran_cache_root: str = DEFAULT_HITRAN_CACHE_ROOT


def generate_benchmark_dataset(output_root: Path | str, spec: BenchmarkGenerationSpec) -> dict[str, object]:
    _validate_spec(spec)
    dataset_id = BenchmarkDatasetId(spec.dataset_slug)
    output_dir = Path(output_root) / str(dataset_id)

    conditions = generate_condition_rows(spec.sequence_count, seed=spec.seed, sampling_strategy=spec.sampling_strategy)
    optical_metadata = _optical_absorption_metadata(spec)
    if spec.optical_absorption_backend == HITRAN_ABSORPTION_BACKEND:
        validate_hitran_benchmark_cache(conditions, cache_root=spec.hitran_cache_root)
    split_rows = build_default_split_rows(conditions, seed=spec.seed)
    labels = _label_array(conditions)
    ultrasonic_spec = WaveformSpec()
    fiber_mic_spec = FiberMicSpec()
    acoustic_metadata = _acoustic_model_metadata(ultrasonic_spec, fiber_mic_spec)
    arrays = build_sequence_arrays(
        conditions,
        timesteps=spec.timesteps,
        dt_s=spec.dt_s,
        seed=spec.seed,
        multi_path_phase=spec.multi_path_phase,
        ultrasonic_spec=ultrasonic_spec,
        fiber_mic_spec=fiber_mic_spec,
        path_lms=spec.path_lms,
        optical_absorption_backend=spec.optical_absorption_backend,
        hitran_cache_root=spec.hitran_cache_root,
    )
    validation_summary = validate_benchmark_assets(conditions, split_rows, arrays, labels)
    sequence_ids = [row["sequence_id"] for row in conditions]
    shapes = write_arrays(output_dir, arrays, labels, sequence_ids, SLOW_CHANNELS, COMPONENT_FIELDS, spec.storage)
    manifest = build_manifest(
        dataset_slug=str(dataset_id),
        sequence_count=spec.sequence_count,
        seed=spec.seed,
        timesteps=spec.timesteps,
        dt_s=spec.dt_s,
        storage=spec.storage,
        multi_path_phase=spec.multi_path_phase,
        sampling_strategy=spec.sampling_strategy,
        path_lms=spec.path_lms,
        optical_absorption_backend=spec.optical_absorption_backend,
        shapes=shapes,
        slow_channels=SLOW_CHANNELS,
        labels=COMPONENT_FIELDS,
        optical_absorption_metadata=optical_metadata,
        acoustic_model_metadata=acoustic_metadata,
    )

    write_csv(output_dir / "condition_grid_sequence.csv", CONDITION_GRID_FIELDS, conditions)
    write_csv(
        output_dir / "sequence_index.csv",
        SEQUENCE_INDEX_FIELDS,
        build_sequence_index_rows(
            conditions,
            stage_profile=spec.stage_profile,
            timesteps=spec.timesteps,
            dt_s=spec.dt_s,
        ),
    )
    write_csv(output_dir / "sequence_labels.csv", SEQUENCE_LABEL_FIELDS, build_label_rows(conditions))
    write_csv(output_dir / "sequences" / "slow_sequence_long.csv", SLOW_SEQUENCE_FIELDS, arrays["slow_rows"])
    for split_name in SPLIT_NAMES:
        write_csv(output_dir / "splits" / f"{split_name}.csv", SPLIT_FIELDS, split_rows[split_name])
    write_json(output_dir / "splits" / "split_summary.json", _split_summary(split_rows))
    train_sequence_ids = {row["sequence_id"] for row in split_rows["train"]}
    train_indexes = [index for index, sequence_id in enumerate(sequence_ids) if sequence_id in train_sequence_ids]
    slow_scaler, slow_modal_scaler = fit_z_score_scalers(
        arrays["slow"],
        train_indexes,
        channel_names=SLOW_CHANNELS,
        modal_groups=SLOW_MODAL_GROUPS,
    )
    write_json(output_dir / "scalers" / "scaler_slow_sequence.json", slow_scaler)
    write_json(output_dir / "scalers" / "scaler_slow_sequence_modal.json", slow_modal_scaler)
    write_json(
        output_dir / "metadata" / "waveform_spec.json",
        {
            "ultrasonic": ultrasonic_spec.to_dict(),
            "fiber_mic": fiber_mic_spec.to_dict(),
            "slow_channels": list(SLOW_CHANNELS),
            "labels": list(COMPONENT_FIELDS),
            "timesteps": spec.timesteps,
            "dt_s": spec.dt_s,
            "path_lms": [float(path_l_m) for path_l_m in spec.path_lms],
            "optical_absorption_backend": spec.optical_absorption_backend,
            **acoustic_metadata,
            **optical_metadata,
        },
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "quality" / "validation_summary.json", validation_summary)

    return {
        "dataset_slug": str(dataset_id),
        "sequence_count": len(conditions),
        "output_dir": str(output_dir),
        "optical_absorption_backend": spec.optical_absorption_backend,
        "validation": validation_summary,
    }


def _validate_spec(spec: BenchmarkGenerationSpec) -> None:
    if spec.timesteps < 4:
        raise ValueError("timesteps must be >= 4")
    if spec.storage not in VALID_STORAGE_FORMATS:
        raise ValueError(f"storage must be one of {list(VALID_STORAGE_FORMATS)}, got {spec.storage}")
    if spec.multi_path_phase not in MULTI_PATH_PHASES:
        raise ValueError(f"multi_path_phase must be one of {list(MULTI_PATH_PHASES)}, got {spec.multi_path_phase}")
    if len(spec.path_lms) == 0:
        raise ValueError("path_lms must contain at least one value")
    if any(path_l_m <= 0.0 for path_l_m in spec.path_lms):
        raise ValueError("path_lms values must be > 0")
    if spec.optical_absorption_backend not in VALID_OPTICAL_ABSORPTION_BACKENDS:
        raise ValueError(
            f"optical_absorption_backend must be one of {list(VALID_OPTICAL_ABSORPTION_BACKENDS)}, got {spec.optical_absorption_backend!r}"
        )
    if spec.optical_absorption_backend == HITRAN_ABSORPTION_BACKEND and not str(spec.hitran_cache_root).strip():
        raise ValueError("hitran_cache_root must be non-empty when optical_absorption_backend is hitran_hapi_v1")


def _optical_absorption_metadata(spec: BenchmarkGenerationSpec) -> dict[str, object]:
    if spec.optical_absorption_backend == HITRAN_ABSORPTION_BACKEND:
        return hitran_manifest_metadata(spec.hitran_cache_root)
    if spec.optical_absorption_backend == EMPIRICAL_ABSORPTION_BACKEND:
        return {
            "optical_crosstalk_policy": "empirical_matrix_v1",
        }
    raise ValueError(f"unsupported optical_absorption_backend: {spec.optical_absorption_backend!r}")


def _acoustic_model_metadata(ultrasonic_spec: WaveformSpec, fiber_mic_spec: FiberMicSpec) -> dict[str, object]:
    return {
        "ultrasonic_model": ultrasonic_spec.model_name,
        "ultrasonic_system_delay_model": ultrasonic_spec.system_delay_model,
        "ultrasonic_system_delay_s": ultrasonic_spec.system_delay_s,
        "ultrasonic_cable_delay_s": ultrasonic_spec.cable_delay_s,
        "ultrasonic_delay_correction_s": ultrasonic_spec.delay_correction_s,
        "ultrasonic_trigger_jitter_std_s": ultrasonic_spec.trigger_jitter_std_s,
        "ultrasonic_transducer_response_model": ultrasonic_spec.transducer_response_model,
        "ultrasonic_transducer_bandwidth_hz": ultrasonic_spec.transducer_bandwidth_hz,
        "fiber_mic_model": fiber_mic_spec.model_name,
        "fiber_mic_acoustic_field_model": fiber_mic_spec.acoustic_field_model,
        "fiber_optical_demodulation_model": fiber_mic_spec.fiber_optical_demodulation_model,
        "acoustic_attenuation_model": ultrasonic_spec.acoustic_attenuation_model,
    }


def _label_array(conditions: list[dict[str, str]]) -> np.ndarray:
    return np.array([[float(row[name]) for name in COMPONENT_FIELDS] for row in conditions], dtype=np.float32)


def _split_summary(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    return {
        "split_policy": "stratified_mixture_id_group_split_v4",
        "group_field": "mixture_id",
        "splits": {
            name: {
                "sequence_count": len(rows),
                "mixture_count": len({row["mixture_id"] for row in rows}),
            }
            for name, rows in split_rows.items()
        },
    }
