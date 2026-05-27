from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SEQUENCE_STATISTICS = ("mean", "std", "min", "max", "last", "delta", "slope")
DEFAULT_WAVEFORM_FRAME_FEATURES = ("mean", "std", "mean_abs", "max_abs", "energy", "peak_index")
MODALITY_OPTIONS = ("slow", "ultrasonic", "fiber_mic")
_Z_SCORE_STD_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class MLFeatureConfig:
    """Feature extraction settings for tabular v4 benchmark baselines."""

    modalities: tuple[str, ...] = ("slow",)
    sequence_statistics: tuple[str, ...] = DEFAULT_SEQUENCE_STATISTICS
    waveform_frame_features: tuple[str, ...] = DEFAULT_WAVEFORM_FRAME_FEATURES
    slow_scaler_path: Path | str | None = None


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

    splits = _load_splits(dataset_dir / "splits")
    master_sequence_ids = _load_str_array(dataset_dir / "metadata" / "sequence_ids.npy")
    split_indices = _resolve_split_indices(splits, master_sequence_ids)[split]
    labels = np.load(dataset_dir / "labels" / "y.npy").astype(np.float32)[split_indices]
    label_names = tuple(_load_str_array(dataset_dir / "metadata" / "label_names.npy"))

    parts: list[np.ndarray] = []
    names: list[str] = []
    if "slow" in config.modalities:
        slow = np.load(dataset_dir / "sequences" / "slow.npy", mmap_mode="r")[split_indices].astype(np.float32)
        if config.slow_scaler_path is not None:
            slow = _apply_scaler(slow, _load_scaler(config.slow_scaler_path)).astype(np.float32)
        slow_channel_names = tuple(_load_str_array(dataset_dir / "metadata" / "slow_channel_names.npy"))
        slow_features, slow_names = sequence_stat_features(
            slow,
            channel_names=slow_channel_names,
            statistics=config.sequence_statistics,
            prefix="slow",
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
        )
        parts.append(fiber_features)
        names.extend(fiber_names)

    if not parts:
        raise ValueError("At least one modality must be selected")
    x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    sequence_ids = tuple(master_sequence_ids[index] for index in split_indices)
    return MLFeatureMatrix(x=x, y=labels, feature_names=tuple(names), label_names=label_names, sequence_ids=sequence_ids)


def sequence_stat_features(
    values: np.ndarray,
    *,
    channel_names: tuple[str, ...],
    statistics: tuple[str, ...] = DEFAULT_SEQUENCE_STATISTICS,
    prefix: str,
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
        block = _sequence_stat(values, stat)
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
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract compact waveform descriptors for one waveform modality."""
    if modality not in {"ultrasonic", "fiber_mic"}:
        raise ValueError(f"Unsupported waveform modality: {modality!r}")
    dataset_dir = Path(dataset_dir)
    waveform = np.load(dataset_dir / "sequences" / f"{modality}_int16.npy", mmap_mode="r")[split_indices]
    scale = np.load(dataset_dir / "sequences" / f"{modality}_scale.npy", mmap_mode="r")[split_indices]
    frames, frame_names = _waveform_frame_descriptors(waveform, scale, frame_features, prefix=modality)
    return sequence_stat_features(frames, channel_names=frame_names, statistics=sequence_statistics, prefix=modality)


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


def _load_str_array(path: Path) -> list[str]:
    values = np.load(path, allow_pickle=True)
    return [str(value) for value in values.tolist()]


def _load_splits(split_dir: Path) -> dict[str, list[dict[str, str]]]:
    split_names = ("train", "val", "test", "extrapolation")
    splits: dict[str, list[dict[str, str]]] = {}
    for name in split_names:
        path = split_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        _validate_split_rows(rows, name)
        splits[name] = rows
    return splits


def _resolve_split_indices(
    splits: dict[str, list[dict[str, str]]],
    sequence_ids: list[str],
) -> dict[str, list[int]]:
    lookup = {sid: idx for idx, sid in enumerate(sequence_ids)}
    indices: dict[str, list[int]] = {}
    for name, rows in splits.items():
        indices[name] = []
        for row in rows:
            sid = row["sequence_id"]
            if sid not in lookup:
                raise KeyError(f"sequence_id {sid} (split={name}) not found in master id list")
            indices[name].append(lookup[sid])
    return indices


def _validate_split_rows(rows: list[dict[str, str]], split_name: str) -> None:
    if not rows:
        return
    missing = {"sequence_id", "mixture_id"}.difference(rows[0])
    if missing:
        raise ValueError(f"Split {split_name} missing required columns: {sorted(missing)}")


def _load_scaler(scaler_path: Path | str) -> dict[str, object]:
    payload = json.loads(Path(scaler_path).read_text(encoding="utf-8"))
    missing = {"method", "channel_names", "mean", "std"}.difference(payload)
    if missing:
        raise ValueError(f"Scaler payload missing keys: {sorted(missing)}")
    if payload["method"] != "z_score":
        raise ValueError(f"Unsupported scaler method: {payload['method']}")
    return payload


def _apply_scaler(x: np.ndarray, scaler: dict[str, object]) -> np.ndarray:
    if x.ndim not in {2, 3}:
        raise ValueError(f"apply_scaler expects a 2D or 3D array, got ndim={x.ndim}")
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    std = np.where(std > _Z_SCORE_STD_EPSILON, std, 1.0)
    if x.shape[-1] != mean.shape[0]:
        raise ValueError(f"last dimension must match scaler channels: {x.shape[-1]} != {mean.shape[0]}")
    if x.ndim == 3:
        mean = mean.reshape(1, 1, -1)
        std = std.reshape(1, 1, -1)
    else:
        mean = mean.reshape(1, -1)
        std = std.reshape(1, -1)
    return (x - mean) / std
