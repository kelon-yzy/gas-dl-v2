"""掘进通风场景的 benchmark 生成。

与 hydrogen_ng `sim.generation.benchmark` 和 syngas `sim.generation.syngas.benchmark` 并存。
差异：
- 使用 tunnel_ventilation_schema 的字段（COMPONENT_FIELDS = (x_CO2, x_O2, x_N2)，BACKGROUND_FIELDS = ()）
- 使用 tunnel_ventilation.slow.build_sequence_arrays（8 通道，无 V_NDIR_CO）
- 使用 tunnel_ventilation.conditions.generate_tunnel_ventilation_condition_rows（2D LHS）
- manifest 标记 composition_scheme="tunnel_ventilation"
- validation 走 sum=100 模式（3 列预测目标严格闭包）
- 阶段 1 仅支持 empirical_v1 后端（HITRAN 待后续阶段实现）

复用：packaging.*（与 scheme 无关）、splits、scalers、phases。
"""
from __future__ import annotations

import math
import os
import shutil
import sys
import time as _time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tv3.sim.core.ids import BenchmarkDatasetId
from tv3.sim.core.schema import MULTI_PATH_PHASES, SPLIT_NAMES, VALID_STORAGE_FORMATS
from tv3.sim.core.tunnel_ventilation_schema import (
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
from tv3.sim.core import tunnel_ventilation_bidir_schema as bidir_schema
from tv3.sim.generation.optical_backend import (
    EMPIRICAL_ABSORPTION_BACKEND,
    VALID_OPTICAL_ABSORPTION_BACKENDS,
)
from tv3.sim.generation.phases import PHASE_SCHEDULES, PhaseSchedule, resolve_phase_schedule
from tv3.sim.generation.tunnel_ventilation.conditions import (
    COMPOSITION_DOMAIN_NARROW,
    COMPOSITION_DOMAIN_WIDE,
    L_M_BASE_RANGE,
    VALID_COMPOSITION_DOMAINS,
    build_tunnel_ventilation_label_rows as build_label_rows,
    generate_tunnel_ventilation_bidir_condition_rows,
    generate_tunnel_ventilation_condition_rows as generate_condition_rows,
    resolve_composition_ranges,
)
from tv3.sim.generation.tunnel_ventilation.flow_physics import bidir_sim_revision
from tv3.sim.generation.tunnel_ventilation.slow import build_sequence_arrays
from tv3.sim.generation.waveforms import FiberMicSpec, WaveformSpec
from tv3.sim.packaging.arrays import write_arrays
from tv3.sim.packaging.index import build_sequence_index_rows
from tv3.sim.packaging.io import write_csv, write_json
from tv3.sim.packaging.manifest import build_manifest
from tv3.sim.packaging.scalers import fit_z_score_scalers
from tv3.sim.packaging.splits import build_default_split_rows
from tv3.sim.validation.integrity import validate_benchmark_assets


DEFAULT_WAVEFORM_PATH_LMS = (0.18, 0.20, 0.22, 0.25, 0.28)  # 200kHz 声程上限 0.3m 约束下的 5 档多光程扫描
DEFAULT_MAX_WORKERS = 24
# tv3 阶段 1 仅支持 empirical 后端
_TV3_VALID_BACKENDS = (EMPIRICAL_ABSORPTION_BACKEND,)


def _array_keys(skip_fiber_mic: bool, *, bidirectional: bool = False) -> tuple[str, ...]:
    """数据集数组 key 列表；skip_fiber_mic=True 时排除 fiber_mic 相关 key。"""
    if bidirectional:
        keys = (
            "slow",
            "ultrasonic_ab",
            "ultrasonic_ba",
            "ultrasonic_ab_scale",
            "ultrasonic_ba_scale",
            "ultrasonic_tof_true_ab_s",
            "ultrasonic_tof_true_ba_s",
            "ultrasonic_tof_observed_ab_s",
            "ultrasonic_tof_observed_ba_s",
            "ultrasonic_peak_index_ab",
            "ultrasonic_peak_index_ba",
            "ultrasonic_tof_quality_ab",
            "ultrasonic_tof_quality_ba",
            "ultrasonic_tof_accepted_ab",
            "ultrasonic_tof_accepted_ba",
            "ultrasonic_v_path_true_m_per_s",
            "ultrasonic_sound_speed_m_per_s",
            "ultrasonic_alpha_true_npm",
        )
    else:
        keys = (
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
        )
    if not skip_fiber_mic:
        keys += ("fiber_mic", "fiber_mic_scale")
    return keys


ARRAY_KEYS = _array_keys(skip_fiber_mic=False)


@dataclass(frozen=True, slots=True)
class TunnelVentilationBenchmarkGenerationSpec:
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
    hitran_cache_root: str = "data/hitran_cache_tv3"  # 阶段 1 不使用，保留为接口兼容
    workers: int = 1
    chunk_size: int | None = None
    temp_dir: str | None = None
    keep_chunks: bool = False
    skip_fiber_mic: bool = False
    bidirectional: bool = False
    # None → WaveformSpec default (3 μs conservative). F3 nominal uses 0.5 μs.
    trigger_jitter_std_s: float | None = None
    # 数据集划分策略（tunnel_ventilation/docs/active/spxy_split_implementation_plan.md）
    # random: 现有 mixture_id shuffle 划分（build_default_split_rows）
    # spxy_v1: ID pool 内 SPXY 选 train + 独立 OOD selector 选 extrapolation + Y 分箱分层 val/test
    # lhs_stratified_split_v1: 全量 Y 分箱分层随机四分类（SPXY 简单对照）
    split_strategy: str = "random"
    spxy_alpha: float = 0.5  # SPXY X/Y 距离权重，仅 spxy_v1 用；1.0=KS, 0.5=标准, 0.0=纯 Y
    extrapolation_strategy: str = "none"  # 仅 spxy_v1 用；none/y_margin_ood/lhs_boundary/kmeans_boundary
    # A1：仅 F 线可选 wide；默认 narrow，单向与旧 bidir 命令行为不变
    composition_domain: str = COMPOSITION_DOMAIN_NARROW


def _log(msg: str) -> None:
    print(f"[tv3-gen] {msg}", file=sys.stderr, flush=True)


def generate_tunnel_ventilation_benchmark_dataset(
    output_root: Path | str,
    spec: TunnelVentilationBenchmarkGenerationSpec,
) -> dict[str, object]:
    _validate_spec(spec)
    dataset_id = BenchmarkDatasetId(spec.dataset_slug)
    output_root = Path(output_root)
    output_dir = output_root / str(dataset_id)
    staging_dir = output_root / f"{dataset_id}.tmp-{uuid.uuid4().hex[:12]}"
    phase_schedule = resolve_phase_schedule(spec.stage_profile)
    phase_schedule_metadata = phase_schedule.to_dict()
    arrays: dict[str, object] = {}

    try:
        _log(f"generating {spec.sequence_count} condition rows ...")
        if spec.bidirectional:
            ranges = resolve_composition_ranges(spec.composition_domain)
            conditions = generate_tunnel_ventilation_bidir_condition_rows(
                spec.sequence_count,
                seed=spec.seed,
                sampling_strategy=spec.sampling_strategy,
                ranges=ranges,
            )
            condition_grid_fields = bidir_schema.CONDITION_GRID_FIELDS
            schema_version = bidir_schema.SCHEMA_VERSION
            composition_scheme = bidir_schema.COMPOSITION_SCHEME
        else:
            if spec.composition_domain != COMPOSITION_DOMAIN_NARROW:
                raise ValueError(
                    "composition_domain='wide' is F-line only (A1); "
                    "unidirectional generation must use composition_domain='narrow'"
                )
            conditions = generate_condition_rows(
                spec.sequence_count,
                seed=spec.seed,
                sampling_strategy=spec.sampling_strategy,
            )
            condition_grid_fields = CONDITION_GRID_FIELDS
            schema_version = SCHEMA_VERSION
            composition_scheme = COMPOSITION_SCHEME
        optical_metadata = _optical_absorption_metadata(spec)
        labels = _label_array(conditions)
        # tv3 采用 int16 + per-timestep 自适应 scale 存储波形（方案 B）
        # 物理 ADC 仍为 20-bit（daq_bits=20），存储时按每 timestep 峰值定标压缩为 int16
        # 实测峰值占满量程 ~22%，per-timestep scale 比固定 scale 量化步长小 ~4.6×
        ultrasonic_kwargs: dict[str, object] = {
            "per_timestep_scale": True,
            "waveform_dtype": "int16",
        }
        if spec.trigger_jitter_std_s is not None:
            if float(spec.trigger_jitter_std_s) < 0.0:
                raise ValueError("trigger_jitter_std_s must be >= 0")
            ultrasonic_kwargs["trigger_jitter_std_s"] = float(spec.trigger_jitter_std_s)
        ultrasonic_spec = WaveformSpec(**ultrasonic_kwargs)
        fiber_mic_spec = FiberMicSpec(per_timestep_scale=True, waveform_dtype="int16") if not spec.skip_fiber_mic else None
        acoustic_metadata = _acoustic_model_metadata(ultrasonic_spec, fiber_mic_spec)
        _log(f"conditions done ({len(conditions)} rows), building waveforms (workers={spec.workers}) ...")
        t_wave = _time.perf_counter()
        array_keys = _array_keys(spec.skip_fiber_mic, bidirectional=spec.bidirectional)
        arrays = _build_sequence_arrays_for_spec(
            conditions=conditions,
            spec=spec,
            phase_schedule=phase_schedule,
            ultrasonic_spec=ultrasonic_spec,
            fiber_mic_spec=fiber_mic_spec,
            staging_dir=staging_dir,
            array_keys=array_keys,
        )
        # split 必须在 arrays 生成之后：spxy_v1 需要 arrays["slow"]/ultrasonic_* 构建 X 特征
        if spec.bidirectional and spec.split_strategy == "spxy_v1":
            raise ValueError(
                "bidirectional + split_strategy='spxy_v1' is not supported in F2; use random or lhs_stratified_split_v1"
            )
        split_rows, split_summary_extra = _build_split_rows_for_spec(spec, conditions, arrays, labels)
        _log(f"waveforms done ({_time.perf_counter() - t_wave:.1f}s), validating ...")
        # tv3: 3 列预测目标 sum=100%（严格闭包），BACKGROUND_FIELDS 为空
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
        _log("validation passed, writing arrays ...")
        fiber_dtype = fiber_mic_spec.waveform_dtype if fiber_mic_spec is not None else "int16"
        shapes = write_arrays(staging_dir, arrays, labels, sequence_ids, SLOW_CHANNELS, COMPONENT_FIELDS, spec.storage, ultrasonic_dtype=ultrasonic_spec.waveform_dtype, fiber_dtype=fiber_dtype)
        if spec.bidirectional:
            ranges = resolve_composition_ranges(spec.composition_domain)
            sim_revision = {
                **bidir_sim_revision(),
                "ultrasonic_center_frequency_hz": float(ultrasonic_spec.center_frequency_hz),
                "sample_rate_hz": int(ultrasonic_spec.sample_rate_hz),
                "daq_bits": int(ultrasonic_spec.daq_bits),
                "waveform_dtype": str(ultrasonic_spec.waveform_dtype),
                "l_m_range": [float(min(spec.path_lms)), float(max(spec.path_lms))],
                "l_m_base_range": [float(L_M_BASE_RANGE[0]), float(L_M_BASE_RANGE[1])],
                "bidirectional": True,
                "skip_fiber_mic": bool(spec.skip_fiber_mic),
                "composition_domain": spec.composition_domain,
                "composition_ranges": {
                    "x_CO2": [float(ranges.co2[0]), float(ranges.co2[1])],
                    "x_O2": [float(ranges.o2[0]), float(ranges.o2[1])],
                    "x_N2": [float(ranges.n2_min), float(ranges.n2_max)],
                },
            }
            if spec.composition_domain == COMPOSITION_DOMAIN_WIDE:
                from tv3.sim.generation.tunnel_ventilation.bidir_registry import (
                    load_f0_registry_wide,
                )

                wide_reg = load_f0_registry_wide()
                sim_revision["composition_domain_tag"] = "wide_hazard_v1"
                sim_revision["f0_registry_file"] = "parameter_registry_wide.json"
                sim_revision["f0_registry_sha256"] = wide_reg["sha256"]
        else:
            sim_revision = {
                "ultrasonic_center_frequency_hz": float(ultrasonic_spec.center_frequency_hz),
                "sample_rate_hz": int(ultrasonic_spec.sample_rate_hz),
                "daq_bits": int(ultrasonic_spec.daq_bits),
                "waveform_dtype": str(ultrasonic_spec.waveform_dtype),
                # l_m_range：多光程扫描离散档位 path_lms 的 min/max
                # l_m_base_range：每条序列基准光程 L_m_base 的采样范围（非扫描阶段使用）
                "l_m_range": [float(min(spec.path_lms)), float(max(spec.path_lms))],
                "l_m_base_range": [float(L_M_BASE_RANGE[0]), float(L_M_BASE_RANGE[1])],
                "physics_backend": "ideal_gas_wms_fracdelay",
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
            schema_version=schema_version,
            composition_scheme=composition_scheme,
            background_fields=BACKGROUND_FIELDS,
        )

        write_csv(staging_dir / "condition_grid_sequence.csv", condition_grid_fields, conditions)
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
        write_json(staging_dir / "splits" / "split_summary.json", _split_summary(split_rows, split_summary_extra))
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
                "fiber_mic": fiber_mic_spec.to_dict() if fiber_mic_spec is not None else None,
                "skip_fiber_mic": spec.skip_fiber_mic,
                "slow_channels": list(SLOW_CHANNELS),
                "labels": list(COMPONENT_FIELDS),
                "background_fields": list(BACKGROUND_FIELDS),
                "composition_scheme": composition_scheme,
                "timesteps": spec.timesteps,
                "dt_s": spec.dt_s,
                "stage_profile": spec.stage_profile,
                "stage_jitter": spec.stage_jitter,
                "phase_schedule": phase_schedule_metadata,
                "path_lms": [float(path_l_m) for path_l_m in spec.path_lms],
                "optical_absorption_backend": spec.optical_absorption_backend,
                "bidirectional": bool(spec.bidirectional),
                **acoustic_metadata,
                **optical_metadata,
            },
        )
        write_json(staging_dir / "manifest.json", manifest)
        write_json(staging_dir / "quality" / "validation_summary.json", validation_summary)
        # 关闭大波形数组的 memmap 句柄：数据已写入 staging_dir/sequences/，
        # 必须释放句柄才能安全清理临时目录（Windows 要求无打开句柄才能删除）
        _close_waveform_memmap(arrays)
        _cleanup_parallel_temp_arrays(arrays, array_keys)
        # 串行路径的 .waveform_temp 目录也需要清理，避免随 staging 迁入最终输出
        _cleanup_serial_waveform_temp(staging_dir)
        _log("writing metadata / CSVs / scalers ...")
        _publish_staging_dir(staging_dir, output_dir)
    except Exception:
        if staging_dir.exists():
            try:
                _close_waveform_memmap(arrays)
            except Exception:
                pass
            shutil.rmtree(staging_dir)
        raise

    return {
        "dataset_slug": str(dataset_id),
        "composition_scheme": composition_scheme,
        "composition_domain": spec.composition_domain,
        "schema_version": schema_version,
        "bidirectional": bool(spec.bidirectional),
        "sequence_count": len(conditions),
        "output_dir": str(output_dir),
        "optical_absorption_backend": spec.optical_absorption_backend,
        "validation": validation_summary,
    }


def _validate_spec(spec: TunnelVentilationBenchmarkGenerationSpec) -> None:
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
    if spec.optical_absorption_backend not in _TV3_VALID_BACKENDS:
        raise ValueError(
            f"tunnel_ventilation 阶段 1 仅支持 empirical_v1 后端，"
            f"got {spec.optical_absorption_backend!r}（HITRAN 后端待后续阶段实现）"
        )
    if spec.workers <= 0:
        raise ValueError("workers must be positive")
    if spec.chunk_size is not None and spec.chunk_size <= 0:
        raise ValueError("chunk_size must be positive when provided")
    if spec.split_strategy not in _VALID_SPLIT_STRATEGIES:
        raise ValueError(
            f"split_strategy must be one of {list(_VALID_SPLIT_STRATEGIES)}, got {spec.split_strategy!r}"
        )
    if not (0.0 <= spec.spxy_alpha <= 1.0):
        raise ValueError(f"spxy_alpha must be in [0, 1], got {spec.spxy_alpha}")
    if spec.extrapolation_strategy not in _VALID_EXTRAPOLATION_STRATEGIES:
        raise ValueError(
            f"extrapolation_strategy must be one of {list(_VALID_EXTRAPOLATION_STRATEGIES)}, "
            f"got {spec.extrapolation_strategy!r}"
        )
    # spxy_v1 必须配独立 OOD selector；其他策略不允许带 OOD selector
    if spec.split_strategy == "spxy_v1" and spec.extrapolation_strategy == "none":
        raise ValueError(
            "split_strategy='spxy_v1' 要求 extrapolation_strategy 为 "
            "y_margin_ood/lhs_boundary/kmeans_boundary 之一，不能为 none"
        )
    if spec.split_strategy != "spxy_v1" and spec.extrapolation_strategy != "none":
        raise ValueError(
            f"extrapolation_strategy={spec.extrapolation_strategy!r} 仅在 split_strategy='spxy_v1' 下有效"
        )
    if spec.composition_domain not in VALID_COMPOSITION_DOMAINS:
        raise ValueError(
            f"composition_domain must be one of {list(VALID_COMPOSITION_DOMAINS)}, "
            f"got {spec.composition_domain!r}"
        )
    if spec.composition_domain == COMPOSITION_DOMAIN_WIDE and not spec.bidirectional:
        raise ValueError(
            "composition_domain='wide' requires bidirectional=True (A1: F-line only)"
        )
    # Write-once isolation: wide must not publish into a narrow slug (and vice versa).
    slug = str(spec.dataset_slug)
    ends_wide = slug.endswith("-wide")
    if spec.composition_domain == COMPOSITION_DOMAIN_WIDE and not ends_wide:
        raise ValueError(
            "composition_domain='wide' requires dataset_slug ending with '-wide' "
            "(write-once isolation; refuse overwriting frozen narrow datasets)"
        )
    if spec.composition_domain == COMPOSITION_DOMAIN_NARROW and ends_wide:
        raise ValueError(
            "dataset_slug ends with '-wide' but composition_domain='narrow'; "
            "use composition_domain='wide' or rename the slug"
        )


def _optical_absorption_metadata(spec: TunnelVentilationBenchmarkGenerationSpec) -> dict[str, object]:
    """tv3 阶段 1 只有 CO2 一个红外活性组分，无串扰矩阵。"""
    return {
        "optical_crosstalk_policy": "tv3_empirical_co2_only_no_crosstalk",
        "ndir_active_gases": ["CO2"],
        "ndir_inactive_gases": ["O2", "N2"],  # 同核双原子，无红外活性
    }


def _acoustic_model_metadata(ultrasonic_spec: WaveformSpec, fiber_mic_spec: FiberMicSpec | None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "ultrasonic_model": ultrasonic_spec.model_name,
        "ultrasonic_system_delay_model": ultrasonic_spec.system_delay_model,
        "ultrasonic_system_delay_s": ultrasonic_spec.system_delay_s,
        "ultrasonic_cable_delay_s": ultrasonic_spec.cable_delay_s,
        "ultrasonic_delay_correction_s": ultrasonic_spec.delay_correction_s,
        "ultrasonic_trigger_jitter_std_s": ultrasonic_spec.trigger_jitter_std_s,
        "ultrasonic_transducer_response_model": ultrasonic_spec.transducer_response_model,
        "ultrasonic_transducer_bandwidth_hz": ultrasonic_spec.transducer_bandwidth_hz,
        "acoustic_attenuation_model": ultrasonic_spec.acoustic_attenuation_model,
    }
    if fiber_mic_spec is not None:
        metadata["fiber_mic_model"] = fiber_mic_spec.model_name
        metadata["fiber_mic_acoustic_field_model"] = fiber_mic_spec.acoustic_field_model
        metadata["fiber_optical_demodulation_model"] = fiber_mic_spec.fiber_optical_demodulation_model
    else:
        metadata["fiber_mic_model"] = None
        metadata["fiber_mic_acoustic_field_model"] = None
        metadata["fiber_optical_demodulation_model"] = None
    return metadata


def _label_array(conditions: list[dict[str, str]]) -> np.ndarray:
    """3 列预测目标（含 x_N2）。"""
    return np.array(
        [[float(row[name]) for name in COMPONENT_FIELDS] for row in conditions],
        dtype=np.float32,
    )


def _split_summary(
    split_rows: dict[str, list[dict[str, str]]],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    summary: dict[str, object] = {
        "group_field": "mixture_id",
        "splits": {
            name: {
                "sequence_count": len(rows),
                "mixture_count": len({row["mixture_id"] for row in rows}),
            }
            for name, rows in split_rows.items()
        },
    }
    if extra:
        summary.update(extra)
    else:
        summary["split_policy"] = "random_mixture_id_split_v4"
    return summary


# 支持的划分策略与 OOD selector（tunnel_ventilation/docs/active/spxy_split_implementation_plan.md）
_VALID_SPLIT_STRATEGIES = ("random", "spxy_v1", "lhs_stratified_split_v1")
_VALID_EXTRAPOLATION_STRATEGIES = ("none", "y_margin_ood", "lhs_boundary", "kmeans_boundary")


def _build_split_rows_for_spec(
    spec: TunnelVentilationBenchmarkGenerationSpec,
    conditions: list[dict[str, str]],
    arrays: dict[str, object],
    labels: np.ndarray,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    """根据 spec.split_strategy 分派划分，返回 (split_rows, summary_extra)。

    random: 现有 build_default_split_rows（mixture_id shuffle）。
    spxy_v1: SPXY 选 train + 独立 OOD selector + Y 分箱分层 val/test。
    lhs_stratified_split_v1: 全量 Y 分箱分层随机四分类。
    """
    if spec.split_strategy == "spxy_v1":
        from tv3.sim.packaging.spxy_split import build_spxy_split_with_summary

        rows, extra = build_spxy_split_with_summary(
            conditions,
            arrays,
            labels,
            seed=spec.seed,
            alpha=spec.spxy_alpha,
            extrapolation_strategy=spec.extrapolation_strategy,
        )
        return rows, extra
    if spec.split_strategy == "lhs_stratified_split_v1":
        from tv3.sim.packaging.spxy_split import build_lhs_stratified_split_with_summary

        rows, extra = build_lhs_stratified_split_with_summary(conditions, labels, seed=spec.seed)
        return rows, extra
    rows = build_default_split_rows(conditions, seed=spec.seed)
    return rows, {"split_policy": "random_mixture_id_split_v4"}


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
    spec: TunnelVentilationBenchmarkGenerationSpec,
    phase_schedule: PhaseSchedule,
    ultrasonic_spec: WaveformSpec,
    fiber_mic_spec: FiberMicSpec | None,
    staging_dir: Path,
    array_keys: tuple[str, ...],
) -> dict[str, object]:
    if spec.workers == 1 or len(conditions) <= 1:
        waveform_temp = staging_dir / ".waveform_temp"
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
            temp_dir=waveform_temp,
            bidirectional=spec.bidirectional,
        )
    from tv3.sim.generation.tunnel_ventilation._parallel import build_arrays_parallel

    arrays = build_arrays_parallel(
        conditions=conditions,
        spec=spec,
        phase_schedule=phase_schedule,
        ultrasonic_spec=ultrasonic_spec,
        fiber_mic_spec=fiber_mic_spec,
        staging_dir=staging_dir,
        array_keys=array_keys,
    )
    if spec.bidirectional:
        arrays["bidirectional"] = True
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


def _cleanup_parallel_temp_arrays(arrays: dict[str, object], array_keys: tuple[str, ...]) -> None:
    from tv3.sim.generation.tunnel_ventilation._parallel import cleanup_parallel_temp_arrays

    cleanup_parallel_temp_arrays(arrays, array_keys)


def _close_waveform_memmap(arrays: dict[str, object]) -> None:
    """关闭大波形数组的 memmap 句柄。

    在 write_arrays 发布到最终目录后调用。若 storage=memmap 已对临时
    merged_*.npy 做 rename/replace，句柄可能已关闭；重复 close 仍安全。
    """
    for key in ("ultrasonic", "ultrasonic_ab", "ultrasonic_ba", "fiber_mic"):
        arr = arrays.get(key)
        if arr is not None:
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                try:
                    mmap.close()
                except ValueError:
                    # Already closed after relocate publish.
                    pass


def _cleanup_serial_waveform_temp(staging_dir: Path) -> None:
    """清理串行路径（workers=1）产生的 .waveform_temp 临时目录。"""
    waveform_temp = staging_dir / ".waveform_temp"
    if waveform_temp.exists():
        import shutil
        shutil.rmtree(waveform_temp)
