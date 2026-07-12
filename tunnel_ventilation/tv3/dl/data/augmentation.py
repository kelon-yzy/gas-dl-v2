from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class TimeSeriesAugmentConfig:
    """时间序列数据增强配置。

    Notes
    -----
    - 所有变换在 NTC 张量上进行（shape=(T, C)）。
    - ``apply_prob`` 是整体概率门控：每个样本按该概率决定是否应用本次配置启用的全部变换。
      取值 1.0 时永远应用；取值 0.0 时永远跳过。
    - ``window_fraction`` 与 ``jitter_std`` 保留向后兼容字段；P3 新引入
      ``max_shift`` / ``amplitude_scale_range`` / ``gaussian_noise_std`` 三种策略。
    - ``amplitude_apply_from_channel`` 控制幅度缩放只作用于波形通道。dataclass 默认 8
      保留 legacy 行为；CLI 会按实际 slow 通道数推断该值。
    """

    jitter_std: float = 0.0
    window_fraction: float = 1.0
    max_shift: int = 0
    amplitude_scale_range: Optional[tuple[float, float]] = None
    amplitude_apply_from_channel: int = 8
    gaussian_noise_std: float = 0.0
    apply_prob: float = 1.0


def augment_sequence(values: np.ndarray, config: TimeSeriesAugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """对单个 ``(T, C)`` 时间序列样本应用增强。"""
    _validate_config(config)

    augmented = np.asarray(values, dtype=np.float32)

    # apply_prob = 1.0 时无需采样（保持向后兼容的确定性输出）。
    if config.apply_prob < 1.0:
        if config.apply_prob <= 0.0 or rng.random() >= config.apply_prob:
            return augmented.astype(np.float32, copy=False)

    if config.window_fraction < 1.0:
        augmented = _window_slice_resample(augmented, config.window_fraction, rng)

    if config.max_shift > 0:
        augmented = _time_shift(augmented, config.max_shift, rng)

    if config.amplitude_scale_range is not None:
        augmented = _amplitude_scale(
            augmented,
            config.amplitude_scale_range,
            config.amplitude_apply_from_channel,
            rng,
        )

    if config.jitter_std > 0.0:
        scale = np.std(augmented, axis=0, keepdims=True)
        augmented = augmented + rng.normal(0.0, config.jitter_std, size=augmented.shape).astype(np.float32) * np.maximum(scale, 1e-6)

    if config.gaussian_noise_std > 0.0:
        augmented = augmented + rng.normal(0.0, config.gaussian_noise_std, size=augmented.shape).astype(np.float32)

    return augmented.astype(np.float32, copy=False)


def _validate_config(config: TimeSeriesAugmentConfig) -> None:
    if config.window_fraction <= 0.0 or config.window_fraction > 1.0:
        raise ValueError("window_fraction must be in (0, 1]")
    if config.jitter_std < 0.0:
        raise ValueError("jitter_std must be >= 0")
    if config.max_shift < 0:
        raise ValueError("max_shift must be >= 0")
    if config.gaussian_noise_std < 0.0:
        raise ValueError("gaussian_noise_std must be >= 0")
    if not 0.0 <= config.apply_prob <= 1.0:
        raise ValueError("apply_prob must be in [0, 1]")
    if config.amplitude_apply_from_channel < 0:
        raise ValueError("amplitude_apply_from_channel must be >= 0")
    if config.amplitude_scale_range is not None:
        lo, hi = config.amplitude_scale_range
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError("amplitude_scale_range bounds must be > 0")
        if lo > hi:
            raise ValueError("amplitude_scale_range lower bound must be <= upper bound")


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


def _time_shift(values: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """随机平移时间轴起点；越界部分通过端点重复填充以保持形状。

    与 improvement_plan.md 中的示例不同，这里采用端点重复（edge padding）
    而不是片段拼接，避免引入跨越非物理边界的伪影。
    """
    timesteps = values.shape[0]
    if max_shift <= 0 or timesteps <= 1:
        return values
    effective_shift = int(min(max_shift, timesteps - 1))
    shift = int(rng.integers(-effective_shift, effective_shift + 1))
    if shift == 0:
        return values
    if shift > 0:
        head = np.repeat(values[shift : shift + 1], shift, axis=0)
        return np.concatenate([head, values[:-shift]], axis=0).astype(np.float32)
    abs_shift = -shift
    tail = np.repeat(values[-abs_shift - 1 : -abs_shift], abs_shift, axis=0)
    return np.concatenate([values[abs_shift:], tail], axis=0).astype(np.float32)


def _amplitude_scale(
    values: np.ndarray,
    scale_range: tuple[float, float],
    apply_from_channel: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """对指定起始列之后的通道施加随机幅度缩放。"""
    if apply_from_channel >= values.shape[1]:
        return values
    lo, hi = scale_range
    scale = float(rng.uniform(lo, hi))
    if scale == 1.0:
        return values
    out = values.copy()
    out[:, apply_from_channel:] = out[:, apply_from_channel:] * scale
    return out.astype(np.float32, copy=False)
