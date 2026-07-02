"""合成气场景的 benchmark 生成。

与 hydrogen_ng `sim.generation.benchmark` 并存。差异：
- 使用 syngas_schema 的字段（COMPONENT_FIELDS 含 x_CO，BACKGROUND_FIELDS 含 x_N2）
- 使用 syngas.slow.build_sequence_arrays（含 V_NDIR_CO）
- 使用 syngas.conditions.generate_syngas_condition_rows（方案 B + 条件顺序）
- manifest 标记 composition_scheme="syngas"
- validation 走 sum<100 模式（COMPONENT_FIELDS 不强制 100%，但 component+background 等于 100%）

复用：packaging.*（与 scheme 无关）、splits、scalers、phases。
"""
from __future__ import annotations

import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim.core.ids import BenchmarkDatasetId
from sim.core.schema import MULTI_PATH_PHASES, SPLIT_NAMES, VALID_STORAGE_FORMATS
from sim.core.syngas_schema import (
    BACKGROUND_FIELDS,
    COMPONENT_FIELDS,
    COMPOSITION_SCHEME,
    CONDITION_GRID_FIELDS,
    SCHEMA_VERSION,
    SEQUENCE_INDEX_FIELDS,
    SEQUENCE_LABEL_FIELDS,
    SLOW_CHANNELS,
    SLOW_MODAL_GROUPS,
    SLOW_SEQUENCE_FIELDS,
    SPLIT_FIELDS,
)
from sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    HITRAN_ABSORPTION_BACKEND,
    VALID_OPTICAL_ABSORPTION_BACKENDS,
)
from sim.generation.phases import PHASE_SCHEDULES, PhaseSchedule, resolve_phase_schedule
from sim.generation.syngas.conditions import (
    build_syngas_label_rows as build_label_rows,
    generate_syngas_condition_rows as generate_condition_rows,
)
from sim.generation.syngas.optical_backend import (
    hitran_syngas_manifest_metadata,
    validate_hitran_syngas_benchmark_cache,
)
from sim.generation.syngas.slow import build_sequence_arrays
from sim.generation.waveforms import FiberMicSpec, WaveformSpec
from sim.packaging.arrays import write_arrays
from sim.packaging.index import build_sequence_index_rows
from sim.packaging.io import write_csv, write_json
from sim.packaging.manifest import build_manifest
from sim.packaging.scalers import fit_z_score_scalers
from sim.packaging.splits import build_default_split_rows
from sim.validation.integrity import validate_benchmark_assets


DEFAULT_WAVEFORM_PATH_LMS = (0.18, 0.20, 0.22, 0.25, 0.28)  # 200kHz 声程上限 0.3m 约束下的 5 档多光程扫描
DEFAULT_HITRAN_CACHE_ROOT = "data/hitran_cache_syngas"
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
class SyngasBenchmarkGenerationSpec:
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
    optical_absorption_backend: str = EMPIRICAL_ABSORPTION_BACKEND
    hitran_cache_root: str = DEFAULT_HITRAN_CACHE_ROOT
    workers: int = 1
    chunk_size: int | None = None
    temp_dir: str | None = None
    keep_chunks: bool = False
    enable_co_crosstalk: bool = False


def generate_syngas_benchmark_dataset(
    output_root: Path | str,
    spec: SyngasBenchmarkGenerationSpec,
) -> dict[str, object]:
    _validate_spec(spec)
    dataset_id = BenchmarkDatasetId(spec.dataset_slug)
    output_root = Path(output_root)
    output_dir = output_root / str(dataset_id)
    staging_dir = output_root / f"{dataset_id}.tmp-{uuid.uuid4().hex[:12]}"
    phase_schedule = resolve_phase_schedule(spec.stage_profile)
    phase_schedule_metadata = phase_schedule.to_dict()

    try:
        conditions = generate_condition_rows(
            spec.sequence_count,
            seed=spec.seed,
            sampling_strategy=spec.sampling_strategy,
        )
        # HITRAN 后端：cache_only_prechecked 策略，生成前先校验所需 cache 存在；
        # 否则生成中途遇到 MissingHitranCacheError，浪费已生成的数据。
        if spec.optical_absorption_backend == HITRAN_ABSORPTION_BACKEND:
            validate_hitran_syngas_benchmark_cache(conditions, cache_root=spec.hitran_cache_root)
        optical_metadata = _optical_absorption_metadata(spec)
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
        # syngas: 4 列预测目标 sum<100，但 component + background = 100%
        validation_summary = validate_benchmark_assets(
            conditions,
            split_rows,
            arrays,
            labels,
            component_fields=COMPONENT_FIELDS,
            slow_channels=SLOW_CHANNELS,
            background_fields=BACKGROUND_FIELDS,
            require_sum_100=True,
        )
        sequence_ids = [row["sequence_id"] for row in conditions]
        shapes = write_arrays(staging_dir, arrays, labels, sequence_ids, SLOW_CHANNELS, COMPONENT_FIELDS, spec.storage, ultrasonic_dtype=ultrasonic_spec.waveform_dtype, fiber_dtype=fiber_mic_spec.waveform_dtype)
        sim_revision = {
            "ultrasonic_center_frequency_hz": float(ultrasonic_spec.center_frequency_hz),
            "sample_rate_hz": int(ultrasonic_spec.sample_rate_hz),
            "daq_bits": int(ultrasonic_spec.daq_bits),
            "waveform_dtype": str(ultrasonic_spec.waveform_dtype),
            "l_m_range": [0.2, 0.3],
            "tag": "v5-200khz-20bit-L03",
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
            schema_version=SCHEMA_VERSION,
            composition_scheme=COMPOSITION_SCHEME,
            background_fields=BACKGROUND_FIELDS,
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
        write_json(staging_dir / "splits" / "split_summary.json", _split_summary(split_rows))
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
                "background_fields": list(BACKGROUND_FIELDS),
                "composition_scheme": COMPOSITION_SCHEME,
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
        "composition_scheme": COMPOSITION_SCHEME,
        "sequence_count": len(conditions),
        "output_dir": str(output_dir),
        "optical_absorption_backend": spec.optical_absorption_backend,
        "validation": validation_summary,
    }


def _validate_spec(spec: SyngasBenchmarkGenerationSpec) -> None:
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
            f"optical_absorption_backend must be one of {list(VALID_OPTICAL_ABSORPTION_BACKENDS)}, "
            f"got {spec.optical_absorption_backend!r}"
        )
    if spec.optical_absorption_backend == HITRAN_ABSORPTION_BACKEND:
        if not str(spec.hitran_cache_root).strip():
            raise ValueError("hitran_cache_root must be non-empty when optical_absorption_backend == hitran_hapi_v1")
        if spec.enable_co_crosstalk:
            raise ValueError(
                "enable_co_crosstalk is mutually exclusive with hitran_hapi_v1 backend: HITRAN already "
                "models multi-gas crosstalk via spectral integration. Set enable_co_crosstalk=False."
            )
    if spec.workers <= 0:
        raise ValueError("workers must be positive")
    if spec.chunk_size is not None and spec.chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided")


def _optical_absorption_metadata(spec: SyngasBenchmarkGenerationSpec) -> dict[str, object]:
    if spec.optical_absorption_backend == EMPIRICAL_ABSORPTION_BACKEND:
        policy = (
            "syngas_empirical_3x3_step2_co2_co_crosstalk"
            if spec.enable_co_crosstalk
            else "syngas_empirical_3x3_step1_co_pure"
        )
        return {
            "optical_crosstalk_policy": policy,
            "enable_co_crosstalk": spec.enable_co_crosstalk,
        }
    if spec.optical_absorption_backend == HITRAN_ABSORPTION_BACKEND:
        return hitran_syngas_manifest_metadata(spec.hitran_cache_root)
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
    """4 列预测目标（不含 x_N2）。"""
    return np.array(
        [[float(row[name]) for name in COMPONENT_FIELDS] for row in conditions],
        dtype=np.float32,
    )


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
    spec: SyngasBenchmarkGenerationSpec,
    phase_schedule: PhaseSchedule,
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
            enable_co_crosstalk=spec.enable_co_crosstalk,
        )
    from sim.generation.syngas._parallel import build_arrays_parallel

    return build_arrays_parallel(
        conditions=conditions,
        spec=spec,
        phase_schedule=phase_schedule,
        ultrasonic_spec=ultrasonic_spec,
        fiber_mic_spec=fiber_mic_spec,
        staging_dir=staging_dir,
        array_keys=ARRAY_KEYS,
    )


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
    from sim.generation.syngas._parallel import cleanup_parallel_temp_arrays

    cleanup_parallel_temp_arrays(arrays, ARRAY_KEYS)
