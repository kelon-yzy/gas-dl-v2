from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tv3.common.splits import load_splits, resolve_split_indices
from tv3.ml.features import MLFeatureMatrix, sequence_stat_features
from tv3.ml.raw_dsp_features import FORMAL_SLOW_CHANNELS, RAW_DSP_FRAME_SCHEMA_VERSION


DEFAULT_ROCKET_SEQUENCE_STATISTICS = ("mean", "std", "min", "max", "range", "first", "last", "delta", "slope")
DEFAULT_PHASE_WINDOWS = ("baseline", "exposure", "steady", "recovery")
DEFAULT_EARLY_FRACTIONS = (0.25, 0.5, 0.75)
DEFAULT_PHYSICS_ARRAYS = (
    "ultrasonic_tof_s",
    "ultrasonic_tof_observed_s",
    "ultrasonic_peak_index",
    "ultrasonic_sound_speed_m_per_s",
    "ultrasonic_sound_speed_estimated_m_per_s",
    "ultrasonic_alpha_true_npm",
    "ultrasonic_tof_quality",
    "ultrasonic_tof_accepted",
)
DEFAULT_FEATURE_BUILDER = "physics_stats_v1"
DEFAULT_FEATURE_CACHE_ROOT = Path("features") / "rocket"
D0_OBSERVED_FEATURE_BUILDER = "d0_observed_physics_stats_v1"
D0_OBSERVED_PHYSICS_ARRAYS = (
    "ultrasonic_tof_observed_s",
    "ultrasonic_peak_index",
    "ultrasonic_sound_speed_estimated_m_per_s",
    "ultrasonic_tof_quality",
    "ultrasonic_tof_accepted",
)
RAW_DSP_FEATURE_BUILDER = "d0_raw_dsp_physics_stats_v1"
RAW_DSP_FRAME_CACHE_ROOT = Path("features") / "raw_dsp" / "raw_dsp_frame_v1"
RAW_DSP_PHYSICS_ARRAYS = (
    "ultrasonic_tof_observed_raw_dsp_s",
    "ultrasonic_peak_index_raw_dsp",
    "ultrasonic_sound_speed_raw_dsp_m_per_s",
    "ultrasonic_corr_peak",
    "ultrasonic_snr_db",
    "ultrasonic_raw_dsp_quality",
    "ultrasonic_raw_dsp_accepted",
)
RAW_DSP_FORBIDDEN_SIMULATOR_ARRAYS = (
    "ultrasonic_tof_s",
    "ultrasonic_tof_observed_s",
    "ultrasonic_peak_index",
    "ultrasonic_sound_speed_m_per_s",
    "ultrasonic_sound_speed_estimated_m_per_s",
    "ultrasonic_alpha_true_npm",
)


@dataclass(frozen=True, slots=True)
class RocketFeatureConfig:
    feature_builder: str = DEFAULT_FEATURE_BUILDER
    include_slow: bool = True
    slow_channels: tuple[str, ...] | None = None
    physics_arrays: tuple[str, ...] = DEFAULT_PHYSICS_ARRAYS
    sequence_statistics: tuple[str, ...] = DEFAULT_ROCKET_SEQUENCE_STATISTICS
    phase_windows: tuple[str, ...] = DEFAULT_PHASE_WINDOWS
    early_fractions: tuple[float, ...] = DEFAULT_EARLY_FRACTIONS


def d0_observed_feature_config() -> RocketFeatureConfig:
    return RocketFeatureConfig(
        feature_builder=D0_OBSERVED_FEATURE_BUILDER,
        include_slow=True,
        slow_channels=None,
        physics_arrays=D0_OBSERVED_PHYSICS_ARRAYS,
        sequence_statistics=DEFAULT_ROCKET_SEQUENCE_STATISTICS,
        phase_windows=DEFAULT_PHASE_WINDOWS,
        early_fractions=DEFAULT_EARLY_FRACTIONS,
    )


def validate_d0_observed_feature_config(config: RocketFeatureConfig) -> None:
    expected = d0_observed_feature_config()
    if config != expected:
        raise ValueError(
            "R7 requires the frozen D0-observed feature contract; "
            f"expected={expected!r}, got={config!r}"
        )


def d0_raw_dsp_feature_config() -> RocketFeatureConfig:
    return RocketFeatureConfig(
        feature_builder=RAW_DSP_FEATURE_BUILDER,
        include_slow=True,
        slow_channels=FORMAL_SLOW_CHANNELS,
        physics_arrays=RAW_DSP_PHYSICS_ARRAYS,
        sequence_statistics=DEFAULT_ROCKET_SEQUENCE_STATISTICS,
        phase_windows=DEFAULT_PHASE_WINDOWS,
        early_fractions=DEFAULT_EARLY_FRACTIONS,
    )


def validate_d0_raw_dsp_feature_config(config: RocketFeatureConfig) -> None:
    expected = d0_raw_dsp_feature_config()
    if config != expected:
        raise ValueError(
            "D2b B1 requires the frozen D0-RawDSP feature contract; "
            f"expected={expected!r}, got={config!r}"
        )


@dataclass(frozen=True, slots=True)
class RocketFeatureCache:
    dataset_dir: Path
    cache_dir: Path
    feature_config: RocketFeatureConfig
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    split_sequence_counts: dict[str, int]


def default_cache_dir(dataset_dir: Path | str, feature_builder: str = DEFAULT_FEATURE_BUILDER) -> Path:
    return Path(dataset_dir) / DEFAULT_FEATURE_CACHE_ROOT / feature_builder


def build_tv3_physics_feature_cache(
    dataset_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    config: RocketFeatureConfig | None = None,
) -> RocketFeatureCache:
    dataset_dir = Path(dataset_dir)
    config = config or RocketFeatureConfig()
    _validate_feature_config(config)
    _validate_tunnel_ventilation_dataset(dataset_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir(dataset_dir, config.feature_builder)
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(dataset_dir / "splits")
    master_sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    _validate_physics_array_source(dataset_dir, config, master_sequence_ids)
    split_indices = resolve_split_indices(splits, master_sequence_ids)
    labels = np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)
    label_names = tuple(_load_str_array(dataset_dir / "metadata" / "label_names.npy"))
    phase_lookup = _load_phase_lookup(dataset_dir / "sequences" / "slow_sequence_long.csv")
    slow_names = tuple(_load_str_array(dataset_dir / "metadata" / "slow_channel_names.npy"))

    feature_names: tuple[str, ...] | None = None
    split_sequence_counts: dict[str, int] = {}
    for split_name, indices in split_indices.items():
        sequence_ids = tuple(master_sequence_ids[index] for index in indices)
        x, current_feature_names = _build_split_features(
            dataset_dir,
            split_indices=indices,
            sequence_ids=sequence_ids,
            slow_channel_names=slow_names,
            phase_lookup=phase_lookup,
            config=config,
        )
        if feature_names is None:
            feature_names = current_feature_names
        elif current_feature_names != feature_names:
            raise ValueError(f"feature names drifted across splits: {split_name}")
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite features detected in split {split_name}")
        np.save(cache_dir / f"feature_matrix_{split_name}.npy", x.astype(np.float32, copy=False))
        split_labels = labels[indices]
        if split_labels.shape[0] != x.shape[0]:
            raise ValueError(f"label row mismatch for split {split_name}: {split_labels.shape[0]} != {x.shape[0]}")
        split_sequence_counts[split_name] = len(sequence_ids)

    assert feature_names is not None
    (cache_dir / "feature_names.json").write_text(
        json.dumps(list(feature_names), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_manifest(
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        config=config,
        feature_names=feature_names,
        split_sequence_counts=split_sequence_counts,
        label_names=label_names,
        slow_channel_names=slow_names,
    )
    _validate_cache_shapes(dataset_dir, cache_dir, feature_names)
    return RocketFeatureCache(
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        feature_config=config,
        feature_names=feature_names,
        label_names=label_names,
        split_sequence_counts=split_sequence_counts,
    )


def load_cached_split_feature_matrix(
    dataset_dir: Path | str,
    cache_dir: Path | str,
    *,
    split: str,
) -> MLFeatureMatrix:
    dataset_dir = Path(dataset_dir)
    cache_dir = Path(cache_dir)
    splits = load_splits(dataset_dir / "splits")
    master_sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    split_indices = resolve_split_indices(splits, master_sequence_ids)[split]
    sequence_ids = tuple(master_sequence_ids[index] for index in split_indices)
    x = np.load(cache_dir / f"feature_matrix_{split}.npy").astype(np.float32, copy=False)
    y = np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)[split_indices]
    feature_names = tuple(json.loads((cache_dir / "feature_names.json").read_text(encoding="utf-8")))
    label_names = tuple(_load_str_array(dataset_dir / "metadata" / "label_names.npy"))
    if x.shape[0] != len(sequence_ids):
        raise ValueError(f"cached split row mismatch for {split}: {x.shape[0]} != {len(sequence_ids)}")
    return MLFeatureMatrix(
        x=x,
        y=y,
        feature_names=feature_names,
        label_names=label_names,
        sequence_ids=sequence_ids,
    )


def _build_split_features(
    dataset_dir: Path,
    *,
    split_indices: list[int],
    sequence_ids: tuple[str, ...],
    slow_channel_names: tuple[str, ...],
    phase_lookup: dict[str, tuple[str, ...]],
    config: RocketFeatureConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    if config.include_slow:
        slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")[split_indices].astype(np.float32)
        channel_names = slow_channel_names
        if config.slow_channels is not None:
            slow, channel_names = _select_slow_channels(slow, slow_channel_names, config.slow_channels)
        block, names = _windowed_sequence_features(
            slow,
            sequence_ids=sequence_ids,
            channel_names=channel_names,
            phase_lookup=phase_lookup,
            statistics=config.sequence_statistics,
            source_prefix="slow",
            phase_windows=config.phase_windows,
            early_fractions=config.early_fractions,
        )
        blocks.append(block)
        feature_names.extend(names)

    for array_name in config.physics_arrays:
        values = np.load(_physics_array_path(dataset_dir, config, array_name), mmap_mode="r")[split_indices].astype(
            np.float32
        )
        block, names = _windowed_sequence_features(
            values[..., np.newaxis],
            sequence_ids=sequence_ids,
            channel_names=(array_name,),
            phase_lookup=phase_lookup,
            statistics=config.sequence_statistics,
            source_prefix="physics",
            phase_windows=config.phase_windows,
            early_fractions=config.early_fractions,
        )
        blocks.append(block)
        feature_names.extend(names)

    if not blocks:
        raise ValueError("physics feature cache requires at least one enabled modality")
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False), tuple(feature_names)


def _windowed_sequence_features(
    values: np.ndarray,
    *,
    sequence_ids: tuple[str, ...],
    channel_names: tuple[str, ...],
    phase_lookup: dict[str, tuple[str, ...]],
    statistics: tuple[str, ...],
    source_prefix: str,
    phase_windows: tuple[str, ...],
    early_fractions: tuple[float, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    full_block, full_names = sequence_stat_features(
        values,
        channel_names=channel_names,
        statistics=statistics,
        prefix=f"full|{source_prefix}",
    )
    feature_blocks.append(full_block)
    feature_names.extend(full_names)

    for phase_name in phase_windows:
        masks = _build_phase_masks(sequence_ids, phase_lookup, phase_name)
        block, names = sequence_stat_features(
            values,
            channel_names=channel_names,
            statistics=statistics,
            prefix=f"ph_{phase_name}|{source_prefix}",
            masks=masks,
        )
        feature_blocks.append(block)
        feature_names.extend(names)

    for fraction in early_fractions:
        masks = _build_early_masks(sequence_ids, phase_lookup, fraction)
        block, names = sequence_stat_features(
            values,
            channel_names=channel_names,
            statistics=statistics,
            prefix=f"early_{fraction:.2f}|{source_prefix}",
            masks=masks,
        )
        feature_blocks.append(block)
        feature_names.extend(names)

    return np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False), tuple(feature_names)


def _build_phase_masks(
    sequence_ids: tuple[str, ...],
    phase_lookup: dict[str, tuple[str, ...]],
    phase_name: str,
) -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for sequence_id in sequence_ids:
        phases = phase_lookup[sequence_id]
        mask = np.array([current == phase_name for current in phases], dtype=bool)
        if not mask.any():
            raise ValueError(f"phase {phase_name!r} selected no rows for sequence_id={sequence_id!r}")
        masks.append(mask)
    return tuple(masks)


def _build_early_masks(
    sequence_ids: tuple[str, ...],
    phase_lookup: dict[str, tuple[str, ...]],
    fraction: float,
) -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for sequence_id in sequence_ids:
        phases = phase_lookup[sequence_id]
        cutoff = max(1, int(np.ceil(len(phases) * fraction)))
        mask = np.zeros(len(phases), dtype=bool)
        mask[:cutoff] = True
        masks.append(mask)
    return tuple(masks)


def _validate_feature_config(config: RocketFeatureConfig) -> None:
    if not config.include_slow and not config.physics_arrays:
        raise ValueError("at least one of slow or physics_arrays must be enabled")
    if not config.sequence_statistics:
        raise ValueError("sequence_statistics must not be empty")
    if len(set(config.sequence_statistics)) != len(config.sequence_statistics):
        raise ValueError("sequence_statistics must not contain duplicates")
    if len(set(config.phase_windows)) != len(config.phase_windows):
        raise ValueError("phase_windows must not contain duplicates")
    if len(set(config.physics_arrays)) != len(config.physics_arrays):
        raise ValueError("physics_arrays must not contain duplicates")
    for fraction in config.early_fractions:
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"early fraction must be in (0, 1], got {fraction}")
    if config.feature_builder == RAW_DSP_FEATURE_BUILDER:
        validate_d0_raw_dsp_feature_config(config)
        forbidden = set(config.physics_arrays).intersection(RAW_DSP_FORBIDDEN_SIMULATOR_ARRAYS)
        if forbidden:
            raise ValueError(f"RawDSP feature contract cannot read simulator-derived arrays: {sorted(forbidden)}")


def _validate_tunnel_ventilation_dataset(dataset_dir: Path) -> None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    composition_scheme = str(manifest.get("composition_scheme", ""))
    if composition_scheme != "tunnel_ventilation":
        raise ValueError(f"physics_stats cache only supports tunnel_ventilation, got {composition_scheme!r}")


def _select_slow_channels(
    slow: np.ndarray,
    channel_names: tuple[str, ...],
    keep: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    name_to_index = {name: index for index, name in enumerate(channel_names)}
    indices: list[int] = []
    for channel in keep:
        if channel not in name_to_index:
            raise ValueError(f"unknown slow channel {channel!r}. available={list(channel_names)}")
        indices.append(name_to_index[channel])
    return slow[:, :, indices], tuple(channel_names[index] for index in indices)


def _physics_array_path(dataset_dir: Path, config: RocketFeatureConfig, array_name: str) -> Path:
    if config.feature_builder == RAW_DSP_FEATURE_BUILDER:
        return dataset_dir / RAW_DSP_FRAME_CACHE_ROOT / f"{array_name}.npy"
    return dataset_dir / "sequences" / f"{array_name}.npy"


def _validate_physics_array_source(
    dataset_dir: Path,
    config: RocketFeatureConfig,
    master_sequence_ids: list[str],
) -> None:
    if config.feature_builder != RAW_DSP_FEATURE_BUILDER:
        return
    raw_dsp_dir = dataset_dir / RAW_DSP_FRAME_CACHE_ROOT
    manifest_path = raw_dsp_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing RawDSP frame cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != RAW_DSP_FRAME_SCHEMA_VERSION:
        raise ValueError(
            f"RawDSP frame schema mismatch: {manifest.get('schema_version')!r} != {RAW_DSP_FRAME_SCHEMA_VERSION!r}"
        )
    if manifest.get("diagnostic_only") is not False:
        raise ValueError("D2b B1 requires train_baseline_median RawDSP cache, not diagnostic exact template")
    if manifest.get("complete_dataset") is not True:
        raise ValueError("D2b B1 requires a complete-dataset RawDSP cache")
    cached_sequence_ids = _load_str_array(raw_dsp_dir / "sequence_ids.npy")
    if cached_sequence_ids != master_sequence_ids:
        raise ValueError("RawDSP cache sequence_ids do not match dataset metadata ordering")
    for array_name in config.physics_arrays:
        path = raw_dsp_dir / f"{array_name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing RawDSP feature array: {path}")


def _load_phase_lookup(path: Path) -> dict[str, tuple[str, ...]]:
    rows: dict[str, list[tuple[int, str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
        index_sequence = header.index("sequence_id")
        index_timestep = header.index("timestep")
        index_phase = header.index("phase_id")
        for line in handle:
            parts = line.rstrip("\n").split(",")
            rows.setdefault(parts[index_sequence], []).append((int(parts[index_timestep]), parts[index_phase]))
    return {
        sequence_id: tuple(phase for _timestep, phase in sorted(items, key=lambda item: item[0]))
        for sequence_id, items in rows.items()
    }


def _load_str_array(path: Path) -> list[str]:
    values = np.load(path, allow_pickle=True)
    return [str(value) for value in values.tolist()]


def _write_manifest(
    *,
    dataset_dir: Path,
    cache_dir: Path,
    config: RocketFeatureConfig,
    feature_names: tuple[str, ...],
    split_sequence_counts: dict[str, int],
    label_names: tuple[str, ...],
    slow_channel_names: tuple[str, ...],
) -> None:
    split_summary_path = dataset_dir / "splits" / "split_summary.json"
    split_policy = None
    if split_summary_path.is_file():
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
        split_policy = split_summary.get("split_policy")
    manifest = {
        "dataset_slug": dataset_dir.name,
        "schema_version": "tv3-rocket-feature-1",
        "sequence_count": int(sum(split_sequence_counts.values())),
        "split_sequence_counts": split_sequence_counts,
        "split_policy": split_policy,
        "feature_builder": config.feature_builder,
        "kernel_seed": None,
        "kernel_count": None,
        "kernel_lengths": [],
        "dilations": [],
        "pooling_stats": list(config.sequence_statistics),
        "modalities": _modalities_payload(config),
        "slow_channels": list(config.slow_channels or slow_channel_names),
        "source_arrays": list(config.physics_arrays),
        "source_array_root": str(
            RAW_DSP_FRAME_CACHE_ROOT if config.feature_builder == RAW_DSP_FEATURE_BUILDER else Path("sequences")
        ),
        "phase_windows": list(config.phase_windows),
        "early_fractions": list(config.early_fractions),
        "label_names": list(label_names),
        "feature_count": len(feature_names),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (cache_dir / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _modalities_payload(config: RocketFeatureConfig) -> list[str]:
    modalities: list[str] = []
    if config.include_slow:
        modalities.append("slow")
    if config.physics_arrays:
        modalities.append("physics_arrays")
    return modalities


def _validate_cache_shapes(dataset_dir: Path, cache_dir: Path, feature_names: tuple[str, ...]) -> None:
    splits = load_splits(dataset_dir / "splits")
    feature_name_count = len(feature_names)
    for split_name, rows in splits.items():
        matrix = np.load(cache_dir / f"feature_matrix_{split_name}.npy", mmap_mode="r")
        if matrix.shape[0] != len(rows):
            raise ValueError(f"cached row count mismatch for {split_name}: {matrix.shape[0]} != {len(rows)}")
        if matrix.shape[1] != feature_name_count:
            raise ValueError(
                f"cached column count mismatch for {split_name}: {matrix.shape[1]} != {feature_name_count}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError(f"cached matrix contains non-finite values for split {split_name}")
