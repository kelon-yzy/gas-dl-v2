from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TimeSeriesAugmentConfig:
    jitter_std: float = 0.0
    window_fraction: float = 1.0


def augment_sequence(values: np.ndarray, config: TimeSeriesAugmentConfig, rng: np.random.Generator) -> np.ndarray:
    if config.window_fraction <= 0.0 or config.window_fraction > 1.0:
        raise ValueError("window_fraction must be in (0, 1]")
    if config.jitter_std < 0.0:
        raise ValueError("jitter_std must be >= 0")
    augmented = np.asarray(values, dtype=np.float32)
    if config.window_fraction < 1.0:
        augmented = _window_slice_resample(augmented, config.window_fraction, rng)
    if config.jitter_std > 0.0:
        scale = np.std(augmented, axis=0, keepdims=True)
        augmented = augmented + rng.normal(0.0, config.jitter_std, size=augmented.shape).astype(np.float32) * np.maximum(scale, 1e-6)
    return augmented.astype(np.float32, copy=False)


def _window_slice_resample(values: np.ndarray, window_fraction: float, rng: np.random.Generator) -> np.ndarray:
    timesteps = values.shape[0]
    window = max(1, int(round(timesteps * window_fraction)))
    if window >= timesteps:
        return values
    start = int(rng.integers(0, timesteps - window + 1))
    sliced = values[start : start + window]
    source_x = np.linspace(0.0, 1.0, num=window, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, num=timesteps, dtype=np.float32)
    channels = [np.interp(target_x, source_x, sliced[:, channel]) for channel in range(values.shape[1])]
    return np.stack(channels, axis=1).astype(np.float32)
