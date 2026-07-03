from __future__ import annotations

import math
import os
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from sim.generation.phases import PHASE_SCHEDULES, resolve_phase_schedule
from sim.generation.slow import build_sequence_arrays, build_sequence_arrays_chunk
from sim.generation.waveforms import FiberMicSpec, WaveformSpec
from sim.packaging.arrays import write_arrays
from sim.packaging.index import build_sequence_index_rows
from sim.packaging.io import write_csv, write_json
from sim.packaging.manifest import build_manifest
from sim.packaging.scalers import fit_z_score_scalers
from sim.packaging.splits import build_default_split_rows
from sim.validation.integrity import validate_benchmark_assets


DEFAULT_WAVEFORM_PATH_LMS = (0.18, 0.20, 0.22, 0.25, 0.28)  # 200kHz 声程上限 0.3m 约束下的 5 档多光程扫描
DEFAULT_HITRAN_CACHE_ROOT = "data/hitran_cache"
DEFAULT_MAX_WORKERS = 24
ARRAY_KEYS = (
    "slow",
    "ultrasonic",
    "ultrasonic_scale",
    "ultrasonic_tof_s",
    "ultrasonic_tof_observed_s",
    "ultrasonic_peak_index",
    "ultrasonic_sound_speed_m_per_s",
    "ultrasonic_sound_speed_estimated_m_per_s",
    "ultrasonic_alpha_true_npm",
    "ultrasonic_tof_quality",
    "ultrasonic_tof_accepted",
    "fiber_mic",
    "fiber_mic_scale",
)


@dataclass(frozen=True, slots=True)
class TimeAxisPreset:
    name: str
    timesteps: int
    dt_s: float


TIME_AXIS_PRESETS = {
    "short": TimeAxisPreset("short", 128, 0.5),
    "standard": TimeAxisPreset("standard", 512, 0.5),
    "long": TimeAxisPreset("long", 1024, 0.5),
    "xlong": TimeAxisPreset("xlong", 2048, 0.5),
}


def resolve_time_axis_preset(name: str) -> TimeAxisPreset:
    try:
        return TIME_AXIS_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"time_axis_preset must be one of {sorted(TIME_AXIS_PRESETS)}, got {name!r}") from exc


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
    stage_jitter: float = 0.0
    sampling_strategy: str = "lhs"
    path_lms: tuple[float, ...] = DEFAULT_WAVEFORM_PATH_LMS
    optical_absorption_backend: str = HITRAN_ABSORPTION_BACKEND
    hitran_cache_root: str = DEFAULT_HITRAN_CACHE_ROOT
    workers: int = 1
    chunk_size: int | None = None
    temp_dir: str | None = None
    keep_chunks: bool = False


def generate_benchmark_dataset(output_root: Path | str, spec: BenchmarkGenerationSpec) -> dict[str, object]:
    _validate_spec(spec)
    dataset_id = BenchmarkDatasetId(spec.dataset_slug)
    output_root = Path(output_root)
    output_dir = output_root / str(dataset_id)
    staging_dir = output_root / f"{dataset_id}.tmp-{uuid.uuid4().hex[:12]}"
    phase_schedule = resolve_phase_schedule(spec.stage_profile)
    phase_schedule_metadata = phase_schedule.to_dict()

    try:
        conditions = generate_condition_rows(spec.sequence_count, seed=spec.seed, sampling_strategy=spec.sampling_strategy)
        optical_metadata = _optical_absorption_metadata(spec)
        if spec.optical_absorption_backend == HITRAN_ABSORPTION_BACKEND:
            validate_hitran_benchmark_cache(conditions, cache_root=spec.hitran_cache_root)
        split_rows = build_default_split_rows(conditions, seed=spec.seed)
        labels = _label_array(conditions)
        ultrasonic_spec = WaveformSpec()
        fiber_mic_spec = FiberMicSpec()
        acoustic_metadata = _acoustic_model_metadata(ultrasonic_spec, fiber_mic_spec)
        arrays = _build_sequence_arrays_for_spec(
            conditions=conditions,
            spec=spec,
            phase_schedule=phase_schedule,
            ultrasonic_spec=ultrasonic_spec,
            fiber_mic_spec=fiber_mic_spec,
            staging_dir=staging_dir,
        )
        validation_summary = validate_benchmark_assets(conditions, split_rows, arrays, labels)
        sequence_ids = [row["sequence_id"] for row in conditions]
        shapes = write_arrays(staging_dir, arrays, labels, sequence_ids, SLOW_CHANNELS, COMPONENT_FIELDS, spec.storage, ultrasonic_dtype=ultrasonic_spec.waveform_dtype, fiber_dtype=fiber_mic_spec.waveform_dtype)
        sim_revision = {
            "ultrasonic_center_frequency_hz": float(ultrasonic_spec.center_frequency_hz),
            "sample_rate_hz": int(ultrasonic_spec.sample_rate_hz),
            "daq_bits": int(ultrasonic_spec.daq_bits),
            "waveform_dtype": str(ultrasonic_spec.waveform_dtype),
            "l_m_range": [0.2, 0.3],
            "physics_backend": "ideal_gas_wms_oversample",
            "tag": "v6-phys-strict",
        }
        manifest = build_manifest(
            dataset_slug=str(dataset_id),
            sequence_count=spec.sequence_count,
            seed=spec.seed,
            timesteps=spec.timesteps,
            dt_s=spec.dt_s,
            storage=spec.storage,
            multi_path_phase=spec.multi_path_phase,
            stage_profile=spec.stage_profile,
            stage_jitter=spec.stage_jitter,
            phase_schedule=phase_schedule_metadata,
            sampling_strategy=spec.sampling_strategy,
            path_lms=spec.path_lms,
            optical_absorption_backend=spec.optical_absorption_backend,
            shapes=shapes,
            slow_channels=SLOW_CHANNELS,
            labels=COMPONENT_FIELDS,
            optical_absorption_metadata=optical_metadata,
            acoustic_model_metadata=acoustic_metadata,
            sim_revision=sim_revision,
        )

        write_csv(staging_dir / "condition_grid_sequence.csv", CONDITION_GRID_FIELDS, conditions)
        write_csv(
            staging_dir / "sequence_index.csv",
            SEQUENCE_INDEX_FIELDS,
            build_sequence_index_rows(
                conditions,
                stage_profile=spec.stage_profile,
                timesteps=spec.timesteps,
                dt_s=spec.dt_s,
            ),
        )
        write_csv(staging_dir / "sequence_labels.csv", SEQUENCE_LABEL_FIELDS, build_label_rows(conditions))
        write_csv(staging_dir / "sequences" / "slow_sequence_long.csv", SLOW_SEQUENCE_FIELDS, arrays["slow_rows"])
        for split_name in SPLIT_NAMES:
            write_csv(staging_dir / "splits" / f"{split_name}.csv", SPLIT_FIELDS, split_rows[split_name])
        write_json(staging_dir / "splits" / "split_summary.json", _split_summary(split_rows, conditions))
        train_sequence_ids = {row["sequence_id"] for row in split_rows["train"]}
        train_indexes = [index for index, sequence_id in enumerate(sequence_ids) if sequence_id in train_sequence_ids]
        slow_scaler, slow_modal_scaler = fit_z_score_scalers(
            arrays["slow"],
            train_indexes,
            channel_names=SLOW_CHANNELS,
            modal_groups=SLOW_MODAL_GROUPS,
        )
        write_json(staging_dir / "scalers" / "scaler_slow_sequence.json", slow_scaler)
        write_json(staging_dir / "scalers" / "scaler_slow_sequence_modal.json", slow_modal_scaler)
        write_json(
            staging_dir / "metadata" / "waveform_spec.json",
            {
                "ultrasonic": ultrasonic_spec.to_dict(),
                "fiber_mic": fiber_mic_spec.to_dict(),
                "slow_channels": list(SLOW_CHANNELS),
                "labels": list(COMPONENT_FIELDS),
                "timesteps": spec.timesteps,
                "dt_s": spec.dt_s,
                "stage_profile": spec.stage_profile,
                "stage_jitter": spec.stage_jitter,
                "phase_schedule": phase_schedule_metadata,
                "path_lms": [float(path_l_m) for path_l_m in spec.path_lms],
                "optical_absorption_backend": spec.optical_absorption_backend,
                **acoustic_metadata,
                **optical_metadata,
            },
        )
        write_json(staging_dir / "manifest.json", manifest)
        write_json(staging_dir / "quality" / "validation_summary.json", validation_summary)
        _cleanup_parallel_temp_arrays(arrays)
        _publish_staging_dir(staging_dir, output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

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
    if spec.stage_profile not in PHASE_SCHEDULES:
        raise ValueError(f"stage_profile must be one of {sorted(PHASE_SCHEDULES)}, got {spec.stage_profile!r}")
    if spec.stage_jitter < 0.0 or spec.stage_jitter >= 1.0:
        raise ValueError("stage_jitter must be in [0, 1)")
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
    if spec.workers <= 0:
        raise ValueError("workers must be positive")
    if spec.chunk_size is not None and spec.chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided")


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


def _split_summary(
    split_rows: dict[str, list[dict[str, str]]],
    conditions: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    """Build split summary with per-split component statistics and distribution checks.

    When ``conditions`` is provided, each split entry includes ``components``
    with per-field mean / std / min / p25 / p50 / p75 / max, and a top-level
    ``distribution_checks`` block runs KS tests between train and every other
    split.  When ``conditions`` is *None* the legacy minimal summary is returned
    (backward-compatible path).
    """
    summary: dict[str, object] = {
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
    if conditions is None:
        return summary

    # Build sequence_id → component-values lookup.
    seq_to_components: dict[str, list[float]] = {
        row["sequence_id"]: [float(row[name]) for name in COMPONENT_FIELDS]
        for row in conditions
    }

    # Per-split component statistics.
    splits_dict: dict[str, dict[str, object]] = {}
    for split_name, rows in split_rows.items():
        entry: dict[str, object] = {
            "sequence_count": len(rows),
            "mixture_count": len({row["mixture_id"] for row in rows}),
        }
        # Gather component values for every sequence in this split.
        component_cols: list[list[float]] = []
        for row in rows:
            sid = row["sequence_id"]
            if sid in seq_to_components:
                component_cols.append(seq_to_components[sid])
        if component_cols:
            arr = np.array(component_cols, dtype=np.float64)
            comp_stats: dict[str, dict[str, float]] = {}
            for col_idx, field in enumerate(COMPONENT_FIELDS):
                col = arr[:, col_idx]
                comp_stats[field] = {
                    "mean": float(np.mean(col)),
                    "std": float(np.std(col, ddof=0)),
                    "min": float(np.min(col)),
                    "p25": float(np.percentile(col, 25)),
                    "p50": float(np.median(col)),
                    "p75": float(np.percentile(col, 75)),
                    "max": float(np.max(col)),
                }
            entry["components"] = comp_stats
        splits_dict[split_name] = entry

    summary["splits"] = splits_dict  # type: ignore[assignment]

    # KS tests: train vs each non-train split.
    train_rows = split_rows.get("train", [])
    train_values: list[list[float]] = [
        seq_to_components[r["sequence_id"]]
        for r in train_rows
        if r["sequence_id"] in seq_to_components
    ]
    if not train_values:
        return summary

    train_arr = np.array(train_values, dtype=np.float64)
    from scipy.stats import ks_2samp

    dist_checks: dict[str, dict[str, object]] = {}
    for other in SPLIT_NAMES:
        if other == "train":
            continue
        other_rows = split_rows.get(other, [])
        other_values = [
            seq_to_components[r["sequence_id"]]
            for r in other_rows
            if r["sequence_id"] in seq_to_components
        ]
        if not other_values:
            continue
        other_arr = np.array(other_values, dtype=np.float64)
        ks_results: dict[str, dict[str, float]] = {}
        has_warning = False
        for col_idx, field in enumerate(COMPONENT_FIELDS):
            stat, pval = ks_2samp(train_arr[:, col_idx], other_arr[:, col_idx])
            ks_results[field] = {"statistic": float(stat), "p_value": float(pval)}
            if pval < 0.05:
                has_warning = True
        dist_checks[f"{other}_vs_train"] = {
            "status": "warn" if has_warning else "pass",
            "ks_tests": ks_results,
        }

    if dist_checks:
        summary["distribution_checks"] = dist_checks

    return summary


def default_worker_count(sequence_count: int | None = None) -> int:
    cpu_count = os.cpu_count() or 1
    workers = max(1, min(DEFAULT_MAX_WORKERS, cpu_count - 2))
    if sequence_count is not None:
        workers = min(workers, max(1, int(sequence_count)))
    return workers


def default_chunk_size(sequence_count: int, workers: int) -> int:
    return max(1, int(math.ceil(float(sequence_count) / float(max(1, workers)))))


def _build_sequence_arrays_for_spec(
    *,
    conditions: list[dict[str, str]],
    spec: BenchmarkGenerationSpec,
    phase_schedule,
    ultrasonic_spec: WaveformSpec,
    fiber_mic_spec: FiberMicSpec,
    staging_dir: Path,
) -> dict[str, object]:
    if spec.workers == 1 or len(conditions) <= 1:
        return build_sequence_arrays(
            conditions,
            timesteps=spec.timesteps,
            dt_s=spec.dt_s,
            seed=spec.seed,
            multi_path_phase=spec.multi_path_phase,
            ultrasonic_spec=ultrasonic_spec,
            fiber_mic_spec=fiber_mic_spec,
            path_lms=spec.path_lms,
            phase_schedule=phase_schedule,
            stage_jitter=spec.stage_jitter,
            optical_absorption_backend=spec.optical_absorption_backend,
            hitran_cache_root=spec.hitran_cache_root,
        )
    chunk_size = spec.chunk_size or default_chunk_size(len(conditions), spec.workers)
    temp_dir = Path(spec.temp_dir) if spec.temp_dir is not None else staging_dir / ".chunks"
    chunk_specs = _condition_chunks(conditions, chunk_size)
    temp_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=min(spec.workers, len(chunk_specs))) as executor:
        futures = [
            executor.submit(
                _generate_chunk_file,
                chunk_index,
                chunk_conditions,
                start_index,
                temp_dir,
                spec,
                phase_schedule,
                ultrasonic_spec,
                fiber_mic_spec,
            )
            for chunk_index, start_index, chunk_conditions in chunk_specs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    arrays = _merge_chunk_files(results, sequence_count=len(conditions), temp_dir=temp_dir)
    if not spec.keep_chunks and spec.temp_dir is None:
        arrays["_temp_dir_to_cleanup"] = str(temp_dir)
    return arrays


def _condition_chunks(conditions: list[dict[str, str]], chunk_size: int) -> list[tuple[int, int, list[dict[str, str]]]]:
    chunks = []
    for chunk_index, start in enumerate(range(0, len(conditions), chunk_size)):
        chunks.append((chunk_index, start, conditions[start : start + chunk_size]))
    return chunks


def _generate_chunk_file(
    chunk_index: int,
    conditions: list[dict[str, str]],
    start_sequence_index: int,
    temp_dir: Path,
    spec: BenchmarkGenerationSpec,
    phase_schedule,
    ultrasonic_spec: WaveformSpec,
    fiber_mic_spec: FiberMicSpec,
) -> dict[str, object]:
    arrays = build_sequence_arrays_chunk(
        conditions,
        timesteps=spec.timesteps,
        dt_s=spec.dt_s,
        seed=spec.seed,
        multi_path_phase=spec.multi_path_phase,
        ultrasonic_spec=ultrasonic_spec,
        fiber_mic_spec=fiber_mic_spec,
        path_lms=spec.path_lms,
        phase_schedule=phase_schedule,
        stage_jitter=spec.stage_jitter,
        optical_absorption_backend=spec.optical_absorption_backend,
        hitran_cache_root=spec.hitran_cache_root,
        start_sequence_index=start_sequence_index,
    )
    chunk_path = temp_dir / f"chunk-{chunk_index:05d}.npz"
    np.savez(
        chunk_path,
        **{key: arrays[key] for key in ARRAY_KEYS},
        slow_rows=np.array(arrays["slow_rows"], dtype=object),
    )
    return {
        "chunk_index": chunk_index,
        "start_sequence_index": start_sequence_index,
        "sequence_count": len(conditions),
        "path": str(chunk_path),
    }


def _merge_chunk_files(results: list[dict[str, object]], *, sequence_count: int, temp_dir: Path) -> dict[str, object]:
    ordered = sorted(results, key=lambda item: int(item["chunk_index"]))
    if not ordered:
        raise ValueError("no chunk files were generated")
    arrays: dict[str, object] = {}
    with np.load(str(ordered[0]["path"]), allow_pickle=True) as first_payload:
        for key in ARRAY_KEYS:
            sample = first_payload[key]
            target = np.lib.format.open_memmap(
                temp_dir / f"merged_{key}.npy",
                mode="w+",
                dtype=sample.dtype,
                shape=(sequence_count, *sample.shape[1:]),
            )
            arrays[key] = target

    slow_rows: list[dict[str, str]] = []
    for result in ordered:
        start = int(result["start_sequence_index"])
        count = int(result["sequence_count"])
        with np.load(str(result["path"]), allow_pickle=True) as payload:
            end = start + count
            for key in ARRAY_KEYS:
                arrays[key][start:end] = payload[key]
            slow_rows.extend(dict(row) for row in payload["slow_rows"].tolist())
    for key in ARRAY_KEYS:
        arrays[key].flush()
    arrays["slow_rows"] = slow_rows
    return arrays


def _publish_staging_dir(staging_dir: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if not output_dir.exists():
        shutil.move(str(staging_dir), str(output_dir))
        return

    backup_dir = output_dir.parent / f"{output_dir.name}.bak-{uuid.uuid4().hex[:12]}"
    output_dir.rename(backup_dir)
    try:
        shutil.move(str(staging_dir), str(output_dir))
    except Exception:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        backup_dir.rename(output_dir)
        raise
    shutil.rmtree(backup_dir)


def _cleanup_parallel_temp_arrays(arrays: dict[str, object]) -> None:
    temp_dir_value = arrays.pop("_temp_dir_to_cleanup", None)
    if temp_dir_value is None:
        return
    for key in ARRAY_KEYS:
        array = arrays.get(key)
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    temp_dir = Path(str(temp_dir_value))
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
