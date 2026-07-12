from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tv3.common.splits import load_splits, resolve_split_indices
from tv3.common.waveform import waveform_array_filename
from tv3.ml.raw_dsp_features import (
    FORMAL_SLOW_CHANNELS,
    RAW_DSP_FRAME_SCHEMA_VERSION,
    RawDSPConfig,
    RawDSPSequenceResult,
    build_baseline_median_template,
    exact_simulator_template,
    extract_raw_dsp_sequence,
    template_digest,
    validate_raw_dsp_config,
)


logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path("features") / "raw_dsp" / "raw_dsp_frame_v1"
TEMPLATE_MODES = ("exact_simulator_debug", "train_baseline_median")

FRAME_OUTPUTS: dict[str, tuple[str, np.dtype[Any]]] = {
    "ultrasonic_peak_index_raw_dsp.npy": ("peak_index", np.dtype(np.float32)),
    "ultrasonic_tof_observed_raw_dsp_s.npy": ("tof_observed_s", np.dtype(np.float32)),
    "ultrasonic_tof_corrected_raw_dsp_s.npy": ("tof_corrected_s", np.dtype(np.float32)),
    "ultrasonic_sound_speed_raw_dsp_m_per_s.npy": ("sound_speed_m_per_s", np.dtype(np.float32)),
    "ultrasonic_corr_peak.npy": ("corr_peak", np.dtype(np.float32)),
    "ultrasonic_peak_to_sidelobe_ratio.npy": ("peak_to_sidelobe_ratio", np.dtype(np.float32)),
    "ultrasonic_snr_db.npy": ("snr_db", np.dtype(np.float32)),
    "ultrasonic_peak_width_samples.npy": ("peak_width_samples", np.dtype(np.float32)),
    "ultrasonic_peak_amplitude_raw_dsp_v.npy": ("peak_amplitude_v", np.dtype(np.float32)),
    "ultrasonic_raw_dsp_quality.npy": ("quality", np.dtype(np.float32)),
    "ultrasonic_raw_dsp_accepted.npy": ("accepted", np.dtype(np.bool_)),
    "ultrasonic_raw_dsp_clipped.npy": ("clipped", np.dtype(np.bool_)),
    "ultrasonic_raw_dsp_boundary_hit.npy": ("boundary_hit", np.dtype(np.bool_)),
}
SEQUENCE_OUTPUTS: dict[str, tuple[str, np.dtype[Any]]] = {
    "ultrasonic_delay_calibration_s.npy": ("delay_calibration_s", np.dtype(np.float32)),
    "ultrasonic_tof_l_m_intercept_s.npy": ("tof_l_m_intercept_s", np.dtype(np.float32)),
    "ultrasonic_sound_speed_slope_raw_dsp_m_per_s.npy": (
        "sound_speed_slope_m_per_s",
        np.dtype(np.float32),
    ),
}

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_dir": None,
    "cache_dir": None,
    "template_mode": "train_baseline_median",
    "template_source_split": "train",
    "template_max_frames": 512,
    "template_pre_samples": 25,
    "template_post_samples": 33,
    "template_min_snr_db": 20.0,
    "template_reference_peak_polarity": -1,
    "chunk_size_sequences": 4,
    "workers": 1,
    "max_sequences": None,
    "raw_dsp": {},
}


@dataclass(frozen=True, slots=True)
class RawDSPPreflight:
    dataset_dir: Path
    dataset_slug: str
    sequence_count: int
    timesteps: int
    waveform_samples: int
    waveform_path: Path
    waveform_dtype: str
    waveform_scale_path: Path
    sample_rate_hz: float
    daq_full_scale_v: float
    slow_path: Path
    slow_channel_names: tuple[str, ...]
    extra_slow_channels: tuple[str, ...]
    phase_csv_path: Path
    sequence_ids: tuple[str, ...]
    split_indices: dict[str, list[int]]
    waveform_spec: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RawDSPFeatureCache:
    dataset_dir: Path
    cache_dir: Path
    template_mode: str
    template_digest: str
    sequence_count: int
    timesteps: int
    reused: bool


@dataclass(frozen=True, slots=True)
class _SequenceBatchTask:
    dataset_dir: str
    waveform_filename: str
    sequence_indices: tuple[int, ...]
    phase_ids: tuple[tuple[str, ...], ...]
    slow_channel_names: tuple[str, ...]
    template: np.ndarray
    template_peak_offset_samples: int
    daq_full_scale_v: float
    config: RawDSPConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build tv3 RawDSP frame feature cache.")
    parser.add_argument("--config", type=Path, default=None, help="JSON config file; CLI values override it.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="tv3 dataset root.")
    parser.add_argument("--cache-dir", type=Path, default=None, help="RawDSP cache directory.")
    parser.add_argument("--template-mode", choices=TEMPLATE_MODES, default=None)
    parser.add_argument("--workers", type=int, default=None, help="CPU worker process count.")
    parser.add_argument("--max-sequences", type=int, default=None, help="Diagnostic prefix sequence count.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate inputs and print preflight JSON only.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _resolve_args(build_parser().parse_args(argv))
    if args.dataset_dir is None:
        raise ValueError("dataset_dir is required")
    preflight = preflight_tv3_raw_dsp_dataset(args.dataset_dir)
    if args.preflight_only:
        print(json.dumps(_preflight_payload(preflight), indent=2, ensure_ascii=False))
        return 0
    cache = build_tv3_raw_dsp_feature_cache(
        preflight,
        cache_dir=args.cache_dir,
        template_mode=args.template_mode,
        template_source_split=args.template_source_split,
        template_max_frames=args.template_max_frames,
        template_pre_samples=args.template_pre_samples,
        template_post_samples=args.template_post_samples,
        template_min_snr_db=args.template_min_snr_db,
        template_reference_peak_polarity=args.template_reference_peak_polarity,
        chunk_size_sequences=args.chunk_size_sequences,
        workers=args.workers,
        max_sequences=args.max_sequences,
        raw_dsp_overrides=args.raw_dsp,
    )
    print(
        json.dumps(
            {
                "dataset_dir": str(cache.dataset_dir),
                "cache_dir": str(cache.cache_dir),
                "template_mode": cache.template_mode,
                "template_digest": cache.template_digest,
                "sequence_count": cache.sequence_count,
                "timesteps": cache.timesteps,
                "reused": cache.reused,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def preflight_tv3_raw_dsp_dataset(dataset_dir: Path | str) -> RawDSPPreflight:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    waveform_spec_path = dataset_dir / "metadata" / "waveform_spec.json"
    slow_names_path = dataset_dir / "metadata" / "slow_channel_names.npy"
    sequence_ids_path = dataset_dir / "metadata" / "sequence_ids.npy"
    slow_path = dataset_dir / "sequences" / "slow.npy"
    scale_path = dataset_dir / "sequences" / "ultrasonic_scale.npy"
    phase_csv_path = dataset_dir / "sequences" / "slow_sequence_long.csv"
    required_paths = (
        manifest_path,
        waveform_spec_path,
        slow_names_path,
        sequence_ids_path,
        slow_path,
        scale_path,
        phase_csv_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing RawDSP input: {path}")

    manifest = _read_json_object(manifest_path)
    if manifest.get("composition_scheme") != "tunnel_ventilation":
        raise ValueError(
            f"RawDSP only supports tunnel_ventilation, got {manifest.get('composition_scheme')!r}"
        )
    waveform_spec_payload = _read_json_object(waveform_spec_path)
    ultrasonic_spec = waveform_spec_payload.get("ultrasonic")
    if not isinstance(ultrasonic_spec, dict):
        raise ValueError("waveform_spec.json must contain an ultrasonic object")
    required_spec_fields = (
        "sample_rate_hz",
        "waveform_dtype",
        "daq_full_scale_v",
        "waveform_samples",
    )
    missing_spec = [field for field in required_spec_fields if field not in ultrasonic_spec]
    if missing_spec:
        raise ValueError(f"ultrasonic waveform metadata missing fields: {missing_spec}")
    waveform_dtype = str(ultrasonic_spec["waveform_dtype"])
    waveform_path = dataset_dir / "sequences" / waveform_array_filename("ultrasonic", waveform_dtype)
    if not waveform_path.is_file():
        raise FileNotFoundError(f"missing ultrasonic waveform array: {waveform_path}")

    waveform = np.load(waveform_path, mmap_mode="r")
    scale = np.load(scale_path, mmap_mode="r")
    slow = np.load(slow_path, mmap_mode="r")
    if waveform.ndim != 3:
        raise ValueError(f"ultrasonic waveform array must be 3D, got {waveform.shape}")
    if not np.issubdtype(waveform.dtype, np.integer):
        raise TypeError(f"ultrasonic waveform dtype must be integer, got {waveform.dtype}")
    if str(waveform.dtype) != waveform_dtype:
        raise ValueError(f"waveform dtype metadata mismatch: {waveform_dtype!r} != {waveform.dtype}")
    if scale.shape != waveform.shape[:-1]:
        raise ValueError(f"ultrasonic scale shape mismatch: {scale.shape} != {waveform.shape[:-1]}")
    if slow.ndim != 3 or slow.shape[:2] != waveform.shape[:2]:
        raise ValueError(f"slow and waveform shapes are not aligned: {slow.shape} vs {waveform.shape}")
    if int(ultrasonic_spec["waveform_samples"]) != waveform.shape[2]:
        raise ValueError(
            f"waveform sample metadata mismatch: {ultrasonic_spec['waveform_samples']} != {waveform.shape[2]}"
        )

    slow_channel_names = tuple(_load_str_array(slow_names_path))
    if len(slow_channel_names) != slow.shape[2]:
        raise ValueError(f"slow channel metadata mismatch: {len(slow_channel_names)} != {slow.shape[2]}")
    missing_slow = [name for name in FORMAL_SLOW_CHANNELS if name not in slow_channel_names]
    if missing_slow:
        raise ValueError(f"missing required formal slow channels: {missing_slow}")
    extra_slow = tuple(name for name in slow_channel_names if name not in FORMAL_SLOW_CHANNELS)

    sequence_ids = tuple(_load_str_array(sequence_ids_path))
    if len(sequence_ids) != waveform.shape[0]:
        raise ValueError(f"sequence id count mismatch: {len(sequence_ids)} != {waveform.shape[0]}")
    splits = load_splits(dataset_dir / "splits")
    split_indices = resolve_split_indices(splits, list(sequence_ids))
    flattened = [index for indices in split_indices.values() for index in indices]
    if len(flattened) != len(set(flattened)):
        raise ValueError("split files assign at least one sequence more than once")
    if set(flattened) != set(range(len(sequence_ids))):
        raise ValueError("split files must cover every sequence exactly once")

    phase_lookup = _load_phase_lookup(phase_csv_path)
    for sequence_id in sequence_ids:
        phases = phase_lookup.get(sequence_id)
        if phases is None:
            raise ValueError(f"phase CSV missing sequence_id={sequence_id!r}")
        if len(phases) != waveform.shape[1]:
            raise ValueError(
                f"phase row count mismatch for {sequence_id}: {len(phases)} != {waveform.shape[1]}"
            )

    return RawDSPPreflight(
        dataset_dir=dataset_dir,
        dataset_slug=str(manifest.get("dataset_slug", dataset_dir.name)),
        sequence_count=int(waveform.shape[0]),
        timesteps=int(waveform.shape[1]),
        waveform_samples=int(waveform.shape[2]),
        waveform_path=waveform_path,
        waveform_dtype=waveform_dtype,
        waveform_scale_path=scale_path,
        sample_rate_hz=float(ultrasonic_spec["sample_rate_hz"]),
        daq_full_scale_v=float(ultrasonic_spec["daq_full_scale_v"]),
        slow_path=slow_path,
        slow_channel_names=slow_channel_names,
        extra_slow_channels=extra_slow,
        phase_csv_path=phase_csv_path,
        sequence_ids=sequence_ids,
        split_indices=split_indices,
        waveform_spec=dict(ultrasonic_spec),
    )


def build_tv3_raw_dsp_feature_cache(
    preflight: RawDSPPreflight,
    *,
    cache_dir: Path | str | None = None,
    template_mode: str = "train_baseline_median",
    template_source_split: str = "train",
    template_max_frames: int = 512,
    template_pre_samples: int = 25,
    template_post_samples: int = 33,
    template_min_snr_db: float = 20.0,
    template_reference_peak_polarity: int = -1,
    chunk_size_sequences: int = 4,
    workers: int = 1,
    max_sequences: int | None = None,
    raw_dsp_overrides: dict[str, Any] | None = None,
) -> RawDSPFeatureCache:
    if template_mode not in TEMPLATE_MODES:
        raise ValueError(f"unsupported template_mode {template_mode!r}")
    if template_source_split != "train":
        raise ValueError("train_baseline_median template_source_split must be 'train'")
    if template_max_frames < 1:
        raise ValueError("template_max_frames must be >= 1")
    if template_reference_peak_polarity not in {-1, 1}:
        raise ValueError("template_reference_peak_polarity must be -1 or 1")
    if chunk_size_sequences < 1:
        raise ValueError("chunk_size_sequences must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if max_sequences is not None and max_sequences < 1:
        raise ValueError("max_sequences must be >= 1")
    if preflight.extra_slow_channels:
        logger.warning(
            "dataset has non-formal slow channels %s; RawDSP uses only named required channels",
            list(preflight.extra_slow_channels),
        )

    config_payload = dict(raw_dsp_overrides or {})
    fixed_metadata_fields = {"sample_rate_hz", "carrier_frequency_hz"}
    if fixed_metadata_fields.intersection(config_payload):
        raise ValueError("raw_dsp sample rate and carrier frequency are fixed by waveform metadata")
    unknown_raw_dsp = set(config_payload) - set(RawDSPConfig.__dataclass_fields__)  # type: ignore[attr-defined]
    if unknown_raw_dsp:
        raise ValueError(f"unknown raw_dsp config keys: {sorted(unknown_raw_dsp)}")
    carrier_frequency_hz = preflight.waveform_spec.get("center_frequency_hz")
    if carrier_frequency_hz is None:
        raise ValueError("ultrasonic waveform metadata missing center_frequency_hz")
    config = RawDSPConfig(
        sample_rate_hz=preflight.sample_rate_hz,
        carrier_frequency_hz=float(carrier_frequency_hz),
        **config_payload,
    )
    validate_raw_dsp_config(config)

    selected_count = preflight.sequence_count if max_sequences is None else min(max_sequences, preflight.sequence_count)
    selected_indices = tuple(range(selected_count))
    selected_ids = tuple(preflight.sequence_ids[index] for index in selected_indices)
    cache_dir = Path(cache_dir) if cache_dir is not None else preflight.dataset_dir / DEFAULT_CACHE_ROOT
    input_summaries = _input_file_summaries(preflight)
    split_binding = _split_binding_payload(preflight)
    train_source_ids = tuple(
        preflight.sequence_ids[index] for index in preflight.split_indices.get(template_source_split, [])
    )
    build_contract = {
        "schema_version": RAW_DSP_FRAME_SCHEMA_VERSION,
        "dataset_slug": preflight.dataset_slug,
        "selected_sequence_ids_digest": _string_digest(selected_ids),
        "selected_sequence_count": selected_count,
        "complete_dataset": selected_count == preflight.sequence_count,
        "template_mode": template_mode,
        "template_source_split": template_source_split,
        "template_source_sequence_ids_digest": _string_digest(train_source_ids),
        "template_max_frames": template_max_frames,
        "template_pre_samples": template_pre_samples,
        "template_post_samples": template_post_samples,
        "template_min_snr_db": template_min_snr_db,
        "template_reference_peak_polarity": template_reference_peak_polarity,
        "split_hash": split_binding.get("split_hash"),
        "split_policy": split_binding.get("split_policy"),
        "split_seed": split_binding.get("split_seed"),
        "raw_dsp": asdict(config),
        "input_files": input_summaries,
        "code_files": _code_file_summaries(),
    }
    build_signature = hashlib.sha256(
        json.dumps(build_contract, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if cache_dir.exists():
        return _load_existing_cache(
            preflight,
            cache_dir,
            build_signature=build_signature,
            template_mode=template_mode,
            selected_count=selected_count,
        )

    staging_dir = cache_dir.with_name(f"{cache_dir.name}.staging")
    if staging_dir.exists():
        raise FileExistsError(
            f"RawDSP staging directory already exists after an earlier incomplete build: {staging_dir}"
        )
    staging_dir.mkdir(parents=True)
    phase_lookup = _load_phase_lookup(preflight.phase_csv_path)
    template, template_source_frame_count, template_peak_offset_samples = _build_template(
        preflight,
        selected_indices=selected_indices,
        phase_lookup=phase_lookup,
        template_mode=template_mode,
        template_source_split=template_source_split,
        template_max_frames=template_max_frames,
        template_pre_samples=template_pre_samples,
        template_post_samples=template_post_samples,
        template_min_snr_db=template_min_snr_db,
        template_reference_peak_polarity=template_reference_peak_polarity,
        config=config,
    )
    np.save(staging_dir / "template.npy", template.astype(np.float32, copy=False))

    frame_outputs = {
        filename: np.lib.format.open_memmap(
            staging_dir / filename,
            mode="w+",
            dtype=dtype,
            shape=(selected_count, preflight.timesteps),
        )
        for filename, (_field, dtype) in FRAME_OUTPUTS.items()
    }
    sequence_outputs = {
        filename: np.lib.format.open_memmap(
            staging_dir / filename,
            mode="w+",
            dtype=dtype,
            shape=(selected_count,),
        )
        for filename, (_field, dtype) in SEQUENCE_OUTPUTS.items()
    }

    for output_start in range(0, selected_count, chunk_size_sequences):
        chunk_indices = selected_indices[output_start : output_start + chunk_size_sequences]
        tasks = _build_worker_tasks(
            preflight,
            chunk_indices,
            phase_lookup,
            template,
            template_peak_offset_samples,
            config,
            workers,
        )
        results = _run_worker_tasks(tasks, workers)
        result_by_index = {sequence_index: result for sequence_index, result in results}
        for local_offset, sequence_index in enumerate(chunk_indices):
            result = result_by_index[sequence_index]
            output_index = output_start + local_offset
            for filename, (field_name, _dtype) in FRAME_OUTPUTS.items():
                frame_outputs[filename][output_index] = getattr(result, field_name)
            for filename, (field_name, _dtype) in SEQUENCE_OUTPUTS.items():
                sequence_outputs[filename][output_index] = getattr(result, field_name)
        for values in (*frame_outputs.values(), *sequence_outputs.values()):
            values.flush()
        logger.info("RawDSP processed %d/%d sequences", min(output_start + len(chunk_indices), selected_count), selected_count)

    np.save(staging_dir / "sequence_ids.npy", np.asarray(selected_ids, dtype=object), allow_pickle=True)
    output_shapes = {
        filename: list(values.shape)
        for filename, values in {**frame_outputs, **sequence_outputs}.items()
    }
    _close_memmaps((*frame_outputs.values(), *sequence_outputs.values()))
    manifest = build_contract | {
        "build_signature": build_signature,
        "source_dataset": str(preflight.dataset_dir),
        "source_waveform_dtype": preflight.waveform_dtype,
        "sample_rate_hz": preflight.sample_rate_hz,
        "carrier_frequency_hz": preflight.waveform_spec.get("center_frequency_hz"),
        "search_window": {
            "sound_speed_min_m_per_s": config.sound_speed_min_m_per_s,
            "sound_speed_max_m_per_s": config.sound_speed_max_m_per_s,
            "delay_min_s": config.delay_min_s,
            "delay_max_s": config.delay_max_s,
        },
        "peak_interpolation_method": "three_point_parabolic",
        "delay_calibration_method": "per_sequence_baseline_fresh_air_median",
        "tof_path_fit_method": "steady_phase_theil_sen",
        "template_digest": template_digest(template),
        "template_peak_offset_samples": template_peak_offset_samples,
        "template_reference_peak_polarity": int(np.sign(template[template_peak_offset_samples])),
        "template_source_frame_count": template_source_frame_count,
        "diagnostic_only": template_mode == "exact_simulator_debug",
        "quality_thresholds": {
            "min_corr_peak": config.min_corr_peak,
            "min_peak_to_sidelobe_ratio": config.min_peak_to_sidelobe_ratio,
            "min_snr_db": config.min_snr_db,
            "max_peak_width_samples": config.max_peak_width_samples,
            "reject_clipped": True,
            "reject_boundary_hit": True,
        },
        "output_shapes": output_shapes,
        "workers": workers,
        "chunk_size_sequences": chunk_size_sequences,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": version("tunnel-ventilation"),
    }
    (staging_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _validate_output_cache(staging_dir, selected_count, preflight.timesteps)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.replace(cache_dir)
    return RawDSPFeatureCache(
        dataset_dir=preflight.dataset_dir,
        cache_dir=cache_dir,
        template_mode=template_mode,
        template_digest=str(manifest["template_digest"]),
        sequence_count=selected_count,
        timesteps=preflight.timesteps,
        reused=False,
    )


def _build_template(
    preflight: RawDSPPreflight,
    *,
    selected_indices: tuple[int, ...],
    phase_lookup: dict[str, tuple[str, ...]],
    template_mode: str,
    template_source_split: str,
    template_max_frames: int,
    template_pre_samples: int,
    template_post_samples: int,
    template_min_snr_db: float,
    template_reference_peak_polarity: int,
    config: RawDSPConfig,
) -> tuple[np.ndarray, int, int]:
    if template_mode == "exact_simulator_debug":
        required = (
            "center_frequency_hz",
            "burst_cycles",
            "transducer_bandwidth_hz",
            "transducer_ringdown_cycles",
        )
        missing = [field for field in required if field not in preflight.waveform_spec]
        if missing:
            raise ValueError(f"exact simulator template metadata missing fields: {missing}")
        template = exact_simulator_template(preflight.waveform_spec)
        return template, 0, int(np.argmax(np.abs(template)))

    allowed_indices = set(selected_indices)
    train_indices = [index for index in preflight.split_indices[template_source_split] if index in allowed_indices]
    if not train_indices:
        raise ValueError("selected sequences contain no train rows for train_baseline_median template")
    waveform = np.load(preflight.waveform_path, mmap_mode="r")
    scale = np.load(preflight.waveform_scale_path, mmap_mode="r")
    slow = np.load(preflight.slow_path, mmap_mode="r")
    path_index = preflight.slow_channel_names.index("L_m")
    frames: list[np.ndarray] = []
    scales: list[float] = []
    path_lengths: list[float] = []
    for sequence_index in train_indices:
        sequence_id = preflight.sequence_ids[sequence_index]
        for timestep, phase in enumerate(phase_lookup[sequence_id]):
            if phase != "baseline":
                continue
            frames.append(np.asarray(waveform[sequence_index, timestep]))
            scales.append(float(scale[sequence_index, timestep]))
            path_lengths.append(float(slow[sequence_index, timestep, path_index]))
            if len(frames) >= template_max_frames:
                break
        if len(frames) >= template_max_frames:
            break
    if not frames:
        raise ValueError("train split contains no baseline waveform frames")
    template = build_baseline_median_template(
        np.stack(frames),
        np.asarray(scales, dtype=np.float32),
        np.asarray(path_lengths, dtype=np.float64),
        config=config,
        daq_full_scale_v=preflight.daq_full_scale_v,
        template_pre_samples=template_pre_samples,
        template_post_samples=template_post_samples,
        min_template_snr_db=template_min_snr_db,
        reference_peak_polarity=template_reference_peak_polarity,
    )
    return template, len(frames), template_pre_samples


def _build_worker_tasks(
    preflight: RawDSPPreflight,
    sequence_indices: tuple[int, ...],
    phase_lookup: dict[str, tuple[str, ...]],
    template: np.ndarray,
    template_peak_offset_samples: int,
    config: RawDSPConfig,
    workers: int,
) -> tuple[_SequenceBatchTask, ...]:
    batch_count = min(workers, len(sequence_indices))
    batches = [sequence_indices[offset::batch_count] for offset in range(batch_count)]
    return tuple(
        _SequenceBatchTask(
            dataset_dir=str(preflight.dataset_dir),
            waveform_filename=preflight.waveform_path.name,
            sequence_indices=tuple(batch),
            phase_ids=tuple(phase_lookup[preflight.sequence_ids[index]] for index in batch),
            slow_channel_names=preflight.slow_channel_names,
            template=template,
            template_peak_offset_samples=template_peak_offset_samples,
            daq_full_scale_v=preflight.daq_full_scale_v,
            config=config,
        )
        for batch in batches
        if batch
    )


def _run_worker_tasks(
    tasks: tuple[_SequenceBatchTask, ...],
    workers: int,
) -> list[tuple[int, RawDSPSequenceResult]]:
    if workers == 1:
        return _process_sequence_batch(tasks[0])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        batches = list(executor.map(_process_sequence_batch, tasks))
    return [item for batch in batches for item in batch]


def _process_sequence_batch(task: _SequenceBatchTask) -> list[tuple[int, RawDSPSequenceResult]]:
    dataset_dir = Path(task.dataset_dir)
    waveform = np.load(dataset_dir / "sequences" / task.waveform_filename, mmap_mode="r")
    scale = np.load(dataset_dir / "sequences" / "ultrasonic_scale.npy", mmap_mode="r")
    slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")
    results: list[tuple[int, RawDSPSequenceResult]] = []
    for sequence_index, phase_ids in zip(task.sequence_indices, task.phase_ids, strict=True):
        result = extract_raw_dsp_sequence(
            waveform[sequence_index],
            scale[sequence_index],
            slow[sequence_index],
            task.slow_channel_names,
            phase_ids,
            task.template,
            daq_full_scale_v=task.daq_full_scale_v,
            config=task.config,
            template_peak_offset_samples=task.template_peak_offset_samples,
        )
        results.append((sequence_index, result))
    return results


def _load_existing_cache(
    preflight: RawDSPPreflight,
    cache_dir: Path,
    *,
    build_signature: str,
    template_mode: str,
    selected_count: int,
) -> RawDSPFeatureCache:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileExistsError(f"RawDSP cache exists without manifest: {cache_dir}")
    manifest = _read_json_object(manifest_path)
    if manifest.get("build_signature") != build_signature:
        raise ValueError(
            f"RawDSP cache manifest mismatch for {cache_dir}; remove or choose a different cache_dir"
        )
    _validate_output_cache(cache_dir, selected_count, preflight.timesteps)
    logger.info("Reusing explicit matching RawDSP cache: %s", cache_dir)
    return RawDSPFeatureCache(
        dataset_dir=preflight.dataset_dir,
        cache_dir=cache_dir,
        template_mode=template_mode,
        template_digest=str(manifest["template_digest"]),
        sequence_count=selected_count,
        timesteps=preflight.timesteps,
        reused=True,
    )


def _validate_output_cache(cache_dir: Path, sequence_count: int, timesteps: int) -> None:
    template = np.load(cache_dir / "template.npy")
    if template.ndim != 1 or template.size < 3 or not np.isfinite(template).all():
        raise ValueError("cached RawDSP template is invalid")
    for filename, (_field, dtype) in FRAME_OUTPUTS.items():
        values = np.load(cache_dir / filename, mmap_mode="r")
        if values.shape != (sequence_count, timesteps):
            raise ValueError(f"RawDSP output shape mismatch for {filename}: {values.shape}")
        if values.dtype != dtype:
            raise ValueError(f"RawDSP output dtype mismatch for {filename}: {values.dtype} != {dtype}")
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
            raise ValueError(f"RawDSP output contains non-finite values: {filename}")
        _close_memmaps((values,))
    for filename, (_field, dtype) in SEQUENCE_OUTPUTS.items():
        values = np.load(cache_dir / filename, mmap_mode="r")
        if values.shape != (sequence_count,):
            raise ValueError(f"RawDSP output shape mismatch for {filename}: {values.shape}")
        if values.dtype != dtype:
            raise ValueError(f"RawDSP output dtype mismatch for {filename}: {values.dtype} != {dtype}")
        if not np.isfinite(values).all():
            raise ValueError(f"RawDSP output contains non-finite values: {filename}")
        _close_memmaps((values,))


def _close_memmaps(values: Sequence[np.ndarray]) -> None:
    for value in values:
        value.flush()
        mmap_handle = getattr(value, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()


def _load_phase_lookup(path: Path) -> dict[str, tuple[str, ...]]:
    rows: dict[str, list[tuple[int, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence_id", "timestep", "phase_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"phase CSV missing required columns {sorted(required)}")
        for row in reader:
            rows.setdefault(row["sequence_id"], []).append((int(row["timestep"]), row["phase_id"]))
    return {
        sequence_id: tuple(phase for _timestep, phase in sorted(items, key=lambda item: item[0]))
        for sequence_id, items in rows.items()
    }


def _split_binding_payload(preflight: RawDSPPreflight) -> dict[str, Any]:
    """从派生数据集的 split_summary.json 读取 hash / policy，写入 RawDSP manifest。"""
    summary_path = preflight.dataset_dir / "splits" / "split_summary.json"
    if not summary_path.is_file():
        return {}
    payload = _read_json_object(summary_path)
    return {
        "split_hash": payload.get("split_hash"),
        "split_policy": payload.get("split_policy"),
        "split_seed": payload.get("split_seed"),
        "x_feature_profile": payload.get("x_feature_profile"),
        "ood_set_hash": payload.get("ood_set_hash"),
    }


def _input_file_summaries(preflight: RawDSPPreflight) -> dict[str, dict[str, Any]]:
    paths = {
        "manifest": preflight.dataset_dir / "manifest.json",
        "waveform_spec": preflight.dataset_dir / "metadata" / "waveform_spec.json",
        "sequence_ids": preflight.dataset_dir / "metadata" / "sequence_ids.npy",
        "slow_channel_names": preflight.dataset_dir / "metadata" / "slow_channel_names.npy",
        "slow": preflight.slow_path,
        "waveform": preflight.waveform_path,
        "waveform_scale": preflight.waveform_scale_path,
        "phase_csv": preflight.phase_csv_path,
    }
    summary_path = preflight.dataset_dir / "splits" / "split_summary.json"
    if summary_path.is_file():
        paths["split_summary"] = summary_path
    for split_name in preflight.split_indices:
        paths[f"split_{split_name}"] = preflight.dataset_dir / "splits" / f"{split_name}.csv"
    return {name: _file_summary(path) for name, path in paths.items()}


def _code_file_summaries() -> dict[str, dict[str, Any]]:
    return {
        "pipeline": _file_summary(Path(__file__)),
        "raw_dsp_features": _file_summary(Path(__file__).parents[1] / "ml" / "raw_dsp_features.py"),
    }


def _file_summary(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _preflight_payload(preflight: RawDSPPreflight) -> dict[str, Any]:
    return {
        "dataset_dir": str(preflight.dataset_dir),
        "dataset_slug": preflight.dataset_slug,
        "sequence_count": preflight.sequence_count,
        "timesteps": preflight.timesteps,
        "waveform_samples": preflight.waveform_samples,
        "waveform_path": str(preflight.waveform_path),
        "waveform_dtype": preflight.waveform_dtype,
        "waveform_scale_path": str(preflight.waveform_scale_path),
        "sample_rate_hz": preflight.sample_rate_hz,
        "slow_channel_names": list(preflight.slow_channel_names),
        "extra_slow_channels": list(preflight.extra_slow_channels),
        "phase_csv_path": str(preflight.phase_csv_path),
        "split_sequence_counts": {
            split_name: len(indices) for split_name, indices in preflight.split_indices.items()
        },
    }


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config = dict(DEFAULT_CONFIG)
    if args.config is not None:
        config.update(_load_config(args.config))
    for key in ("dataset_dir", "cache_dir", "template_mode", "workers", "max_sequences"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config["dataset_dir"] = Path(config["dataset_dir"]) if config["dataset_dir"] is not None else None
    config["cache_dir"] = Path(config["cache_dir"]) if config["cache_dir"] is not None else None
    config["preflight_only"] = bool(args.preflight_only)
    return argparse.Namespace(**config)


def _load_config(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    unknown = set(payload) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown RawDSP config keys: {sorted(unknown)}")
    raw_dsp = payload.get("raw_dsp", {})
    if not isinstance(raw_dsp, dict):
        raise ValueError("raw_dsp config must be a JSON object")
    return payload


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _load_str_array(path: Path) -> list[str]:
    values = np.load(path, allow_pickle=True)
    return [str(value) for value in values.tolist()]


def _string_digest(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
