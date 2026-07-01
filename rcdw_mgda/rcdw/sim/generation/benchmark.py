"""RCDW benchmark 全流程编排：conditions → arrays → packaging → validation。

对应方案 §5.9。

与 HG 主线差异：
- 单进程版本（v1.x 简化，未来如需要 ProcessPoolExecutor 并行可参考 HG 主线移植）。
- ``composition_scheme="rcdw_o2_co2_n2"``、``SPLIT_NAMES=("train","val","test")``。
- ``train_modalities=["slow", "ultrasonic"]``（光纤麦克风落盘但不进训练）。
- HITRAN 后端默认开启；缓存目录默认 ``rcdw_mgda/data/hitran_cache/``。
"""

from __future__ import annotations

import math
import shutil
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rcdw.sim.core.ids import BenchmarkDatasetId
from rcdw.sim.core.schema import (
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
from rcdw.sim.generation.acoustic_physics import (
    acoustic_model_metadata as rcdw_acoustic_model_metadata,
)
from rcdw.sim.generation.conditions import build_label_rows, generate_condition_rows
from rcdw.sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    HITRAN_ABSORPTION_BACKEND,
    VALID_OPTICAL_ABSORPTION_BACKENDS,
    hitran_manifest_metadata,
    validate_hitran_benchmark_cache,
)
from rcdw.sim.generation.phases import PHASE_SCHEDULES, resolve_phase_schedule
from rcdw.sim.generation.slow import build_sequence_arrays, build_sequence_arrays_chunk
from rcdw.sim.generation.waveforms import FiberMicSpec, WaveformSpec
from rcdw.sim.packaging.arrays import write_arrays
from rcdw.sim.packaging.index import build_sequence_index_rows
from rcdw.sim.packaging.io import write_csv, write_json
from rcdw.sim.packaging.manifest import build_manifest
from rcdw.sim.packaging.scalers import (
    DEFAULT_PASSTHROUGH_CHANNELS,
    INPUT_CHANNEL_ORDER,
    INPUT_PASSTHROUGH_CHANNELS,
    INPUT_SCALER_VERSION,
    fit_input_channel_scaler,
    fit_z_score_scalers,
)
from rcdw.sim.packaging.splits import build_default_split_rows
from rcdw.sim.validation.integrity import validate_benchmark_assets


DEFAULT_WAVEFORM_PATH_LMS = (0.20, 0.25, 0.30, 0.35, 0.40)
DEFAULT_HITRAN_CACHE_ROOT = "data/hitran_cache"


@dataclass(frozen=True, slots=True)
class BenchmarkGenerationSpec:
    """RCDW benchmark 生成规格。所有运行时参数集中在此 dataclass。"""

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
    train_modalities: tuple[str, ...] = ("slow", "ultrasonic")
    num_workers: int = 1
    chunk_size: int = 0


def generate_benchmark_dataset(
    output_root: Path | str, spec: BenchmarkGenerationSpec
) -> dict[str, object]:
    """端到端生成 RCDW benchmark：写盘到 ``{output_root}/{dataset_slug}/``。

    流程：
        1. 校验 spec
        2. generate_condition_rows
        3. validate HITRAN cache（hitran 后端）
        4. build_default_split_rows
        5. build_sequence_arrays
        6. validate_benchmark_assets
        7. write_arrays + write_csv（condition_grid / sequence_index /
           sequence_labels / slow_sequence_long / splits/*.csv）
        8. fit_z_score_scalers（仅 train + 异质通道 passthrough）
        9. build_manifest + write_json
        10. 写 validation_summary 与 waveform_spec.json

    Returns:
        ``{"dataset_slug", "sequence_count", "output_dir",
        "optical_absorption_backend", "validation"}``
    """
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
        optical_metadata = _optical_absorption_metadata(spec)
        if spec.optical_absorption_backend == HITRAN_ABSORPTION_BACKEND:
            validate_hitran_benchmark_cache(
                conditions, cache_root=spec.hitran_cache_root
            )
        split_rows = build_default_split_rows(conditions, seed=spec.seed)
        labels = _label_array(conditions)
        ultrasonic_spec = WaveformSpec()
        fiber_mic_spec = FiberMicSpec()
        acoustic_metadata = _acoustic_model_metadata_for_manifest(
            ultrasonic_spec, fiber_mic_spec
        )
        arrays = _build_sequence_arrays_for_spec(
            conditions,
            spec=spec,
            ultrasonic_spec=ultrasonic_spec,
            fiber_mic_spec=fiber_mic_spec,
        )
        validate_benchmark_assets(conditions, split_rows, arrays, labels)

        sequence_ids = [row["sequence_id"] for row in conditions]
        shapes = write_arrays(
            staging_dir,
            arrays,
            labels,
            sequence_ids,
            SLOW_CHANNELS,
            COMPONENT_FIELDS,
            spec.storage,
        )

        # CSV 落盘
        write_csv(
            staging_dir / "condition_grid_sequence.csv",
            CONDITION_GRID_FIELDS,
            conditions,
        )
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
        write_csv(
            staging_dir / "sequence_labels.csv",
            SEQUENCE_LABEL_FIELDS,
            build_label_rows(conditions),
        )
        write_csv(
            staging_dir / "sequences" / "slow_sequence_long.csv",
            SLOW_SEQUENCE_FIELDS,
            arrays["slow_rows"],
        )
        for split_name in SPLIT_NAMES:
            write_csv(
                staging_dir / "splits" / f"{split_name}.csv",
                SPLIT_FIELDS,
                split_rows[split_name],
            )
        write_json(
            staging_dir / "splits" / "split_summary.json",
            _split_summary(split_rows),
        )

        # Scaler 拟合（仅 train + 异质通道 passthrough）
        train_sequence_ids = {row["sequence_id"] for row in split_rows["train"]}
        train_indexes = [
            index
            for index, sequence_id in enumerate(sequence_ids)
            if sequence_id in train_sequence_ids
        ]
        slow_scaler, slow_modal_scaler = fit_z_score_scalers(
            arrays["slow"],
            train_indexes,
            channel_names=SLOW_CHANNELS,
            modal_groups=SLOW_MODAL_GROUPS,
            skip_channels=DEFAULT_PASSTHROUGH_CHANNELS,
        )
        validation_summary = validate_benchmark_assets(
            conditions,
            split_rows,
            arrays,
            labels,
            scaler=slow_scaler,
            expected_passthrough_channels=DEFAULT_PASSTHROUGH_CHANNELS,
        )
        write_json(
            staging_dir / "scalers" / "scaler_slow_sequence.json", slow_scaler
        )
        write_json(
            staging_dir / "scalers" / "scaler_slow_sequence_modal.json",
            slow_modal_scaler,
        )

        # 12 维 input scaler（H1 修复）：覆盖模型实际消费的全部 12 通道，
        # train-only 拟合，是唯一真正作用到模型输入的标准化产物。
        input_scaler = fit_input_channel_scaler(
            _collect_input_train_values(arrays, train_indexes)
        )
        write_json(staging_dir / "scalers" / "input_scaler.json", input_scaler)

        # Manifest
        scaler_metadata = {
            "passthrough_channels": list(DEFAULT_PASSTHROUGH_CHANNELS),
            "peak_index_strategy": "skip",
            "transform_target": "slow",
        }
        input_normalization = {
            "applied": True,
            "coverage": "input_12ch",
            "artifact": "scalers/input_scaler.json",
            "method": "z_score",
            "fit_scope": "train_split_only",
            "version": INPUT_SCALER_VERSION,
            "channel_order": list(INPUT_CHANNEL_ORDER),
            "passthrough_channels": list(INPUT_PASSTHROUGH_CHANNELS),
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
            scaler_metadata=scaler_metadata,
            input_normalization=input_normalization,
            train_modalities=spec.train_modalities,
        )

        # waveform_spec.json + manifest.json + validation_summary.json
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
                "path_lms": [float(p) for p in spec.path_lms],
                "optical_absorption_backend": spec.optical_absorption_backend,
                **acoustic_metadata,
                **optical_metadata,
            },
        )
        write_json(staging_dir / "manifest.json", manifest)
        write_json(
            staging_dir / "quality" / "validation_summary.json", validation_summary
        )

        _publish_staging_dir(staging_dir, output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {
        "dataset_slug": str(dataset_id),
        "sequence_count": len(conditions),
        "output_dir": str(output_dir),
        "optical_absorption_backend": spec.optical_absorption_backend,
        "validation": validation_summary,
    }


# ---- 内部函数 ----


def _validate_spec(spec: BenchmarkGenerationSpec) -> None:
    if spec.sequence_count <= 0:
        raise ValueError("sequence_count must be > 0")
    if spec.timesteps < 4:
        raise ValueError("timesteps must be >= 4")
    if spec.storage not in VALID_STORAGE_FORMATS:
        raise ValueError(
            f"storage must be one of {list(VALID_STORAGE_FORMATS)}, got {spec.storage}"
        )
    if spec.multi_path_phase not in MULTI_PATH_PHASES:
        raise ValueError(
            f"multi_path_phase must be one of {list(MULTI_PATH_PHASES)}, "
            f"got {spec.multi_path_phase}"
        )
    if spec.stage_profile not in PHASE_SCHEDULES:
        raise ValueError(
            f"stage_profile must be one of {sorted(PHASE_SCHEDULES)}, "
            f"got {spec.stage_profile!r}"
        )
    if spec.stage_jitter < 0.0 or spec.stage_jitter >= 1.0:
        raise ValueError("stage_jitter must be in [0, 1)")
    if len(spec.path_lms) == 0:
        raise ValueError("path_lms must contain at least one value")
    if any(path_l_m <= 0.0 for path_l_m in spec.path_lms):
        raise ValueError("path_lms values must be > 0")
    if spec.optical_absorption_backend not in VALID_OPTICAL_ABSORPTION_BACKENDS:
        raise ValueError(
            f"optical_absorption_backend must be one of "
            f"{list(VALID_OPTICAL_ABSORPTION_BACKENDS)}, "
            f"got {spec.optical_absorption_backend!r}"
        )
    if spec.num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if spec.chunk_size < 0:
        raise ValueError("chunk_size must be >= 0")
    if spec.chunk_size != 0 and spec.chunk_size <= 0:
        raise ValueError("chunk_size must be 0 for auto or > 0")
    if (
        spec.optical_absorption_backend == HITRAN_ABSORPTION_BACKEND
        and not str(spec.hitran_cache_root).strip()
    ):
        raise ValueError(
            "hitran_cache_root must be non-empty when optical_absorption_backend is hitran_hapi_v1"
        )


def _build_sequence_arrays_for_spec(
    conditions: list[dict[str, str]],
    *,
    spec: BenchmarkGenerationSpec,
    ultrasonic_spec: WaveformSpec,
    fiber_mic_spec: FiberMicSpec,
) -> dict[str, object]:
    if spec.num_workers <= 1 or len(conditions) <= 1:
        return build_sequence_arrays(
            conditions,
            timesteps=spec.timesteps,
            dt_s=spec.dt_s,
            seed=spec.seed,
            multi_path_phase=spec.multi_path_phase,
            ultrasonic_spec=ultrasonic_spec,
            fiber_mic_spec=fiber_mic_spec,
            path_lms=spec.path_lms,
            phase_schedule=resolve_phase_schedule(spec.stage_profile),
            stage_jitter=spec.stage_jitter,
            optical_absorption_backend=spec.optical_absorption_backend,
            hitran_cache_root=spec.hitran_cache_root,
        )

    chunk_size = spec.chunk_size or max(1, math.ceil(len(conditions) / spec.num_workers))
    chunks = [
        (start, conditions[start : start + chunk_size])
        for start in range(0, len(conditions), chunk_size)
    ]
    with ProcessPoolExecutor(max_workers=spec.num_workers) as executor:
        futures = [
            executor.submit(
                build_sequence_arrays_chunk,
                chunk_conditions,
                timesteps=spec.timesteps,
                dt_s=spec.dt_s,
                seed=spec.seed,
                multi_path_phase=spec.multi_path_phase,
                ultrasonic_spec=ultrasonic_spec,
                fiber_mic_spec=fiber_mic_spec,
                path_lms=spec.path_lms,
                phase_schedule=spec.stage_profile,
                stage_jitter=spec.stage_jitter,
                optical_absorption_backend=spec.optical_absorption_backend,
                hitran_cache_root=spec.hitran_cache_root,
                start_sequence_index=start,
            )
            for start, chunk_conditions in chunks
        ]
        chunk_arrays = [future.result() for future in futures]
    return _merge_sequence_array_chunks(chunk_arrays)


def _merge_sequence_array_chunks(chunks: list[dict[str, object]]) -> dict[str, object]:
    if not chunks:
        raise ValueError("chunks must not be empty")
    merged: dict[str, object] = {}
    for key, first_value in chunks[0].items():
        if key == "slow_rows":
            rows: list[dict[str, str]] = []
            for chunk in chunks:
                rows.extend(chunk[key])  # type: ignore[arg-type]
            merged[key] = rows
        elif isinstance(first_value, np.ndarray):
            merged[key] = np.concatenate(
                [np.asarray(chunk[key]) for chunk in chunks], axis=0
            )
        else:
            raise TypeError(f"unsupported sequence array chunk key {key!r}")
    return merged


def _optical_absorption_metadata(
    spec: BenchmarkGenerationSpec,
) -> dict[str, object]:
    if spec.optical_absorption_backend == HITRAN_ABSORPTION_BACKEND:
        return hitran_manifest_metadata(spec.hitran_cache_root)
    if spec.optical_absorption_backend == EMPIRICAL_ABSORPTION_BACKEND:
        return {"optical_crosstalk_policy": "empirical_matrix_v1"}
    raise ValueError(
        f"unsupported optical_absorption_backend: {spec.optical_absorption_backend!r}"
    )


def _acoustic_model_metadata_for_manifest(
    ultrasonic_spec: WaveformSpec, fiber_mic_spec: FiberMicSpec
) -> dict[str, object]:
    base = dict(rcdw_acoustic_model_metadata())
    base.update(
        {
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
    )
    return base


def _label_array(conditions: list[dict[str, str]]) -> np.ndarray:
    return np.array(
        [[float(row[name]) for name in COMPONENT_FIELDS] for row in conditions],
        dtype=np.float32,
    )


# Input scaler 收集顺序：ultrasonic 元数据紧跟 SLOW_CHANNELS 之后，
# 与 INPUT_CHANNEL_ORDER[7:] 及 rcdw.data.dataset 的拼接顺序一致。
_INPUT_ULTRASONIC_CHANNELS = (
    "ultrasonic_tof_observed_s",
    "ultrasonic_sound_speed_estimated_m_per_s",
    "ultrasonic_peak_index",
    "ultrasonic_tof_quality",
    "ultrasonic_tof_accepted",
)


def _collect_input_train_values(
    arrays: dict[str, object], train_indexes: list[int]
) -> dict[str, np.ndarray]:
    """按 train_indexes 收集 12 维消费布局各通道的 train-only 展平取值。

    slow(7) 取自 ``arrays["slow"][train, :, i]``；ultrasonic(5) 取自各自的
    ``(N, T)`` 数组。返回 dict 供 ``fit_input_channel_scaler`` 逐通道拟合。
    """
    slow = np.asarray(arrays["slow"])  # (N, T, 7)
    train_slow = slow[train_indexes]   # (n, T, 7)
    values: dict[str, np.ndarray] = {}
    for index, channel in enumerate(SLOW_CHANNELS):
        values[channel] = train_slow[:, :, index].astype(np.float64).ravel()
    for channel in _INPUT_ULTRASONIC_CHANNELS:
        arr = np.asarray(arrays[channel])  # (N, T)
        values[channel] = arr[train_indexes].astype(np.float64).ravel()
    return values


def _split_summary(
    split_rows: dict[str, list[dict[str, str]]],
) -> dict[str, object]:
    return {
        "split_policy": "stratified_mixture_id_group_split_rcdw_v1",
        "group_field": "mixture_id",
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "splits": {
            name: {
                "sequence_count": len(rows),
                "mixture_count": len({row["mixture_id"] for row in rows}),
            }
            for name, rows in split_rows.items()
        },
    }


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
            shutil.rmtree(output_dir, ignore_errors=True)
        backup_dir.rename(output_dir)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)
