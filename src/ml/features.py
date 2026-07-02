from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import csv

import numpy as np

from common.scalers import apply_scaler, load_scaler
from common.splits import load_splits, resolve_split_indices
from common.waveform import waveform_array_path
from common.windows import WINDOW_KIND_EARLY, WINDOW_KIND_PHASE, WindowConfig


DEFAULT_SEQUENCE_STATISTICS = ("mean", "std", "min", "max", "last", "delta", "slope")
DEFAULT_WAVEFORM_FRAME_FEATURES = ("mean", "std", "mean_abs", "max_abs", "energy", "peak_index")
MODALITY_OPTIONS = ("slow", "ultrasonic", "fiber_mic")


@dataclass(frozen=True, slots=True)
class MLFeatureConfig:
    """Feature extraction settings for tabular v4 benchmark baselines."""

    modalities: tuple[str, ...] = ("slow",)
    sequence_statistics: tuple[str, ...] = DEFAULT_SEQUENCE_STATISTICS
    waveform_frame_features: tuple[str, ...] = DEFAULT_WAVEFORM_FRAME_FEATURES
    slow_scaler_path: Path | str | None = None
    slow_channels: tuple[str, ...] | None = None
    phase_filter: str | None = None
    early_fraction: float | None = None
    feature_windows: tuple[WindowConfig | None, ...] | None = None


@dataclass(frozen=True, slots=True)
class MLFeatureMatrix:
    """A split-specific tabular feature matrix and aligned labels."""

    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    sequence_ids: tuple[str, ...]


def load_feature_matrix(
    dataset_dir: Path | str,
    *,
    split: str,
    config: MLFeatureConfig | None = None,
) -> MLFeatureMatrix:
    """Load one v4 benchmark split as tabular ML features.

    The returned rows follow the split CSV order and are aligned to ``labels/y.npy``.
    Slow channels use sequence-level statistics. Waveform modalities are first
    reduced to frame-level descriptors and then summarized over timesteps.
    """
    dataset_dir = Path(dataset_dir)
    config = config or MLFeatureConfig()
    _validate_modalities(config.modalities)
    if config.slow_channels is not None and "slow" not in config.modalities:
        raise ValueError("slow_channels requires 'slow' in modalities")
    _validate_window_config(config)
    if config.feature_windows is not None:
        return _load_multiwindow_feature_matrix(dataset_dir, split=split, config=config)

    splits = load_splits(dataset_dir / "splits")
    master_sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    split_indices = resolve_split_indices(splits, master_sequence_ids)[split]
    split_sequence_ids = tuple(master_sequence_ids[index] for index in split_indices)
    masks = _load_timestep_masks(dataset_dir, split_sequence_ids, config)
    labels = np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)[split_indices]
    label_names = tuple(_load_str_array(dataset_dir / "metadata" / "label_names.npy"))

    parts: list[np.ndarray] = []
    names: list[str] = []
    if "slow" in config.modalities:
        slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")[split_indices].astype(np.float32)
        if config.slow_scaler_path is not None:
            slow = apply_scaler(slow, load_scaler(config.slow_scaler_path)).astype(np.float32)
        slow_channel_names = tuple(_load_str_array(dataset_dir / "metadata" / "slow_channel_names.npy"))
        if config.slow_channels is not None:
            slow, slow_channel_names = _select_slow_channels(slow, slow_channel_names, config.slow_channels)
        slow_features, slow_names = sequence_stat_features(
            slow,
            channel_names=slow_channel_names,
            statistics=config.sequence_statistics,
            prefix="slow",
            masks=masks,
        )
        parts.append(slow_features)
        names.extend(slow_names)
    if "ultrasonic" in config.modalities:
        ultrasonic_features, ultrasonic_names = waveform_stat_features(
            dataset_dir,
            split_indices,
            modality="ultrasonic",
            frame_features=config.waveform_frame_features,
            sequence_statistics=config.sequence_statistics,
            masks=masks,
        )
        parts.append(ultrasonic_features)
        names.extend(ultrasonic_names)
    if "fiber_mic" in config.modalities:
        fiber_features, fiber_names = waveform_stat_features(
            dataset_dir,
            split_indices,
            modality="fiber_mic",
            frame_features=config.waveform_frame_features,
            sequence_statistics=config.sequence_statistics,
            masks=masks,
        )
        parts.append(fiber_features)
        names.extend(fiber_names)

    if not parts:
        raise ValueError("At least one modality must be selected")
    x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    return MLFeatureMatrix(x=x, y=labels, feature_names=tuple(names), label_names=label_names, sequence_ids=split_sequence_ids)


def _load_multiwindow_feature_matrix(
    dataset_dir: Path,
    *,
    split: str,
    config: MLFeatureConfig,
) -> MLFeatureMatrix:
    matrices: list[MLFeatureMatrix] = []
    names: list[str] = []
    for window in config.feature_windows or ():
        window_config = _single_window_config(config, window)
        matrix = load_feature_matrix(dataset_dir, split=split, config=window_config)
        matrices.append(matrix)
        tag = _window_tag(window)
        names.extend(f"{tag}|{name}" for name in matrix.feature_names)

    reference = matrices[0]
    for matrix in matrices[1:]:
        if matrix.label_names != reference.label_names:
            raise ValueError("multi-window feature matrices must have matching label names")
        if matrix.sequence_ids != reference.sequence_ids:
            raise ValueError("multi-window feature matrices must have matching sequence ids")
        if not np.array_equal(matrix.y, reference.y):
            raise ValueError("multi-window feature matrices must have matching labels")

    x = np.concatenate([matrix.x for matrix in matrices], axis=1).astype(np.float32, copy=False)
    return MLFeatureMatrix(
        x=x,
        y=reference.y,
        feature_names=tuple(names),
        label_names=reference.label_names,
        sequence_ids=reference.sequence_ids,
    )


def sequence_stat_features(
    values: np.ndarray,
    *,
    channel_names: tuple[str, ...],
    statistics: tuple[str, ...] = DEFAULT_SEQUENCE_STATISTICS,
    prefix: str,
    masks: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Summarize an ``(N, T, C)`` sequence tensor into tabular statistics."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"values must be shaped (N, T, C), got ndim={values.ndim}")
    if values.shape[-1] != len(channel_names):
        raise ValueError(f"channel_names length {len(channel_names)} does not match channel count {values.shape[-1]}")

    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []
    for stat in statistics:
        block = _masked_sequence_stat(values, masks, stat) if masks is not None else _sequence_stat(values, stat)
        feature_blocks.append(block)
        feature_names.extend(f"{prefix}:{channel}:{stat}" for channel in channel_names)
    return np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False), tuple(feature_names)


def waveform_stat_features(
    dataset_dir: Path | str,
    split_indices: list[int],
    *,
    modality: str,
    frame_features: tuple[str, ...] = DEFAULT_WAVEFORM_FRAME_FEATURES,
    sequence_statistics: tuple[str, ...] = DEFAULT_SEQUENCE_STATISTICS,
    masks: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract compact waveform descriptors for one waveform modality."""
    if modality not in {"ultrasonic", "fiber_mic"}:
        raise ValueError(f"Unsupported waveform modality: {modality!r}")
    dataset_dir = Path(dataset_dir)
    waveform = np.load(waveform_array_path(dataset_dir, modality), mmap_mode="r")[split_indices]
    scale = np.load(dataset_dir / "sequences" / f"{modality}_scale.npy", mmap_mode="r")[split_indices]
    frames, frame_names = _waveform_frame_descriptors(waveform, scale, frame_features, prefix=modality)
    return sequence_stat_features(frames, channel_names=frame_names, statistics=sequence_statistics, prefix=modality, masks=masks)


def _sequence_stat(values: np.ndarray, stat: str) -> np.ndarray:
    if stat == "mean":
        return values.mean(axis=1)
    if stat == "std":
        return values.std(axis=1)
    if stat == "min":
        return values.min(axis=1)
    if stat == "max":
        return values.max(axis=1)
    if stat == "first":
        return values[:, 0, :]
    if stat == "last":
        return values[:, -1, :]
    if stat == "delta":
        return values[:, -1, :] - values[:, 0, :]
    if stat == "range":
        return values.max(axis=1) - values.min(axis=1)
    if stat == "slope":
        return _least_squares_slope(values)
    raise ValueError(f"Unsupported sequence statistic: {stat!r}")


def _least_squares_slope(values: np.ndarray) -> np.ndarray:
    timesteps = values.shape[1]
    if timesteps <= 1:
        return np.zeros((values.shape[0], values.shape[2]), dtype=np.float32)
    t = np.arange(timesteps, dtype=np.float32)
    t = t - t.mean()
    denom = float(np.sum(t * t))
    centered = values - values.mean(axis=1, keepdims=True)
    return np.sum(centered * t.reshape(1, -1, 1), axis=1) / denom


def _masked_sequence_stat(values: np.ndarray, masks: tuple[np.ndarray, ...], stat: str) -> np.ndarray:
    rows = []
    for row, mask in zip(values, masks, strict=True):
        selected = row[mask]
        if selected.shape[0] == 0:
            raise ValueError("timestep window selected no frames")
        rows.append(_sequence_stat(selected[np.newaxis, :, :], stat)[0])
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def _waveform_frame_descriptors(
    waveform_int16: np.ndarray,
    scale: np.ndarray,
    frame_features: tuple[str, ...],
    *,
    prefix: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    waveform = waveform_int16.astype(np.float32) * scale.astype(np.float32)[..., np.newaxis]
    abs_waveform = np.abs(waveform)
    sample_count = waveform.shape[-1]
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for feature in frame_features:
        if feature == "mean":
            block = waveform.mean(axis=-1)
        elif feature == "std":
            block = waveform.std(axis=-1)
        elif feature == "mean_abs":
            block = abs_waveform.mean(axis=-1)
        elif feature == "max_abs":
            block = abs_waveform.max(axis=-1)
        elif feature == "energy":
            block = np.mean(waveform * waveform, axis=-1)
        elif feature == "peak_index":
            block = np.argmax(abs_waveform, axis=-1).astype(np.float32) / max(1, sample_count - 1)
        else:
            raise ValueError(f"Unsupported waveform frame feature: {feature!r}")
        blocks.append(block[..., np.newaxis])
        names.append(f"{prefix}_{feature}")
    return np.concatenate(blocks, axis=-1).astype(np.float32, copy=False), tuple(names)


def _validate_modalities(modalities: tuple[str, ...]) -> None:
    if not modalities:
        raise ValueError("modalities must not be empty")
    for modality in modalities:
        if modality not in MODALITY_OPTIONS:
            raise ValueError(f"Unknown modality: {modality!r}. Available: {MODALITY_OPTIONS}")


def _select_slow_channels(
    slow: np.ndarray,
    channel_names: tuple[str, ...],
    keep: tuple[str, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """按保留通道名过滤 slow 张量的通道维（channel ablation）。

    slow 形状 (N, T, C)；返回过滤后的张量与对应通道名。
    """
    if not keep:
        raise ValueError("slow_channels must not be empty")
    name_to_index = {name: index for index, name in enumerate(channel_names)}
    indices: list[int] = []
    for channel in keep:
        if channel not in name_to_index:
            raise ValueError(f"Unknown slow channel: {channel!r}. Available: {list(channel_names)}")
        indices.append(name_to_index[channel])
    return slow[:, :, indices], tuple(channel_names[index] for index in indices)


def _validate_window_config(config: MLFeatureConfig) -> None:
    if config.early_fraction is not None and (config.early_fraction <= 0.0 or config.early_fraction > 1.0):
        raise ValueError("early_fraction must be in (0, 1]")
    if config.feature_windows is None:
        return
    if config.phase_filter is not None or config.early_fraction is not None:
        raise ValueError("feature_windows cannot be combined with phase_filter or early_fraction")
    if not config.feature_windows:
        raise ValueError("feature_windows must not be empty")
    for window in config.feature_windows:
        if window is None:
            continue
        if not isinstance(window, WindowConfig):
            raise ValueError("feature_windows entries must be WindowConfig or None")
        if window.kind == WINDOW_KIND_PHASE:
            if not str(window.value):
                raise ValueError("phase feature window value must be a non-empty string")
        elif window.kind == WINDOW_KIND_EARLY:
            fraction = float(window.value)
            if fraction <= 0.0 or fraction > 1.0:
                raise ValueError("early feature window value must be in (0, 1]")
        else:
            raise ValueError(f"Unknown feature window kind: {window.kind!r}")


def _single_window_config(config: MLFeatureConfig, window: WindowConfig | None) -> MLFeatureConfig:
    if window is None:
        return replace(config, feature_windows=None, phase_filter=None, early_fraction=None)
    if window.kind == WINDOW_KIND_PHASE:
        return replace(config, feature_windows=None, phase_filter=str(window.value), early_fraction=None)
    if window.kind == WINDOW_KIND_EARLY:
        return replace(config, feature_windows=None, phase_filter=None, early_fraction=float(window.value))
    raise ValueError(f"Unknown feature window kind: {window.kind!r}")


def _window_tag(window: WindowConfig | None) -> str:
    if window is None:
        return "full"
    if window.kind == WINDOW_KIND_PHASE:
        return f"ph_{window.value}"
    if window.kind == WINDOW_KIND_EARLY:
        return f"early_{float(window.value):.2f}"
    raise ValueError(f"Unknown feature window kind: {window.kind!r}")


def _load_timestep_masks(dataset_dir: Path, sequence_ids: tuple[str, ...], config: MLFeatureConfig) -> tuple[np.ndarray, ...] | None:
    if config.phase_filter is None and config.early_fraction is None:
        return None
    phase_rows = _load_phase_rows(dataset_dir / "sequences" / "slow_sequence_long.csv")
    masks = []
    for sequence_id in sequence_ids:
        rows = phase_rows[sequence_id]
        timesteps = len(rows)
        mask = np.ones(timesteps, dtype=bool)
        if config.phase_filter is not None:
            mask &= np.array([phase_id == config.phase_filter for _timestep, phase_id in rows], dtype=bool)
        if config.early_fraction is not None:
            cutoff = max(1, int(np.ceil(timesteps * config.early_fraction)))
            early_mask = np.zeros(timesteps, dtype=bool)
            early_mask[:cutoff] = True
            mask &= early_mask
        if not mask.any():
            raise ValueError(f"empty timestep window for sequence_id={sequence_id!r}")
        masks.append(mask)
    return tuple(masks)


@lru_cache(maxsize=8)
def _load_phase_rows(path: Path) -> dict[str, tuple[tuple[int, str], ...]]:
    # 按文件路径缓存：同一数据集在 protocol 多次评估中复用，避免重复读取长 CSV。
    # 假设数据集生成后该 CSV 内容不再变化（生产 CLI 为独立进程，天然满足）。
    rows: dict[str, list[tuple[int, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["sequence_id"], []).append((int(row["timestep"]), row["phase_id"]))
    return {sequence_id: tuple(sorted(items, key=lambda item: item[0])) for sequence_id, items in rows.items()}


def _load_str_array(path: Path) -> list[str]:
    values = np.load(path, allow_pickle=True)
    return [str(value) for value in values.tolist()]
