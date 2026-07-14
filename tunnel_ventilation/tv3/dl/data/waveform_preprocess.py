"""波形 dequant / normalize / stats 与设备侧组装。

CPU 路径与 GPU 路径共用同一套数值约定：
- dequant: ``int_or_float * scale[:, None]``
- per-timestep z-score: population std（``ddof=0`` / ``correction=0``），下限 ``1e-6``
- stats 在 normalize 之前、对 dequant 电压计算
- 通道顺序：slow → 各波形 stats（ultrasonic 再 fiber_mic）→ 各波形样本
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

WAVEFORM_PREPROCESS_OPTIONS = ("cpu", "gpu")
WAVEFORM_MODALITY_ORDER = ("ultrasonic", "fiber_mic")
NORMALIZE_STD_FLOOR = 1e-6
MODEL_INPUT_KWARG_KEYS = frozenset({"phase_stats"})


def dequantize_waveform_numpy(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``values (T, L)`` × ``scale (T,)`` → float32 电压。"""
    if values.ndim != 2:
        raise ValueError(f"waveform values must be 2D (T, L), got {values.shape}")
    if scale.ndim != 1 or scale.shape[0] != values.shape[0]:
        raise ValueError(f"scale must be shaped (T,), got {scale.shape} for values {values.shape}")
    return values.astype(np.float32, copy=False) * scale.astype(np.float32, copy=False)[:, np.newaxis]


def normalize_waveform_numpy(values: np.ndarray, *, eps: float = NORMALIZE_STD_FLOOR) -> np.ndarray:
    """逐帧 z-score（population std）。"""
    mean = values.mean(axis=-1, keepdims=True)
    std = np.maximum(values.std(axis=-1, keepdims=True), eps)
    return (values - mean) / std


def waveform_stats_numpy(values: np.ndarray, features: Sequence[str]) -> np.ndarray:
    """对 dequant 后电压计算逐帧 stats，shape ``(T, F)``。"""
    if not features:
        raise ValueError("features must not be empty")
    blocks: list[np.ndarray] = []
    abs_values = np.abs(values)
    for feature in features:
        if feature == "log_std":
            block = np.log1p(values.std(axis=-1, keepdims=True))
        elif feature == "log_max_abs":
            block = np.log1p(abs_values.max(axis=-1, keepdims=True))
        else:
            raise ValueError(f"Unsupported waveform stats feature: {feature!r}")
        blocks.append(block.astype(np.float32, copy=False))
    return np.concatenate(blocks, axis=-1).astype(np.float32, copy=False)


def dequantize_waveform_torch(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Batch-aware dequant：``values (B, T, L)`` × ``scale (B, T)``。"""
    if values.ndim != 3:
        raise ValueError(f"waveform values must be shaped (B, T, L), got {tuple(values.shape)}")
    if scale.shape != values.shape[:2]:
        raise ValueError(
            f"scale must be shaped {tuple(values.shape[:2])}, got {tuple(scale.shape)}"
        )
    return values.float() * scale.unsqueeze(-1)


def normalize_waveform_torch(
    values: torch.Tensor,
    *,
    eps: float = NORMALIZE_STD_FLOOR,
) -> torch.Tensor:
    """逐帧 z-score；``correction=0`` 对齐 NumPy population std。"""
    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, keepdim=True, correction=0).clamp_min(eps)
    return (values - mean) / std


def waveform_stats_torch(values: torch.Tensor, features: Sequence[str]) -> torch.Tensor:
    """对 dequant 后电压计算逐帧 stats，shape ``(B, T, F)``。"""
    if not features:
        raise ValueError("features must not be empty")
    blocks: list[torch.Tensor] = []
    abs_values = values.abs()
    for feature in features:
        if feature == "log_std":
            block = torch.log1p(values.std(dim=-1, keepdim=True, correction=0))
        elif feature == "log_max_abs":
            block = torch.log1p(abs_values.amax(dim=-1, keepdim=True))
        else:
            raise ValueError(f"Unsupported waveform stats feature: {feature!r}")
        blocks.append(block)
    return torch.cat(blocks, dim=-1)


def numpy_to_tensor(array: np.ndarray, *, dtype: np.dtype | None = None) -> torch.Tensor:
    """单次 contiguous 物化后 ``from_numpy``；memmap 只读视图会复制一次。"""
    if dtype is None:
        arr = np.ascontiguousarray(array)
    else:
        arr = np.ascontiguousarray(array, dtype=dtype)
    if not arr.flags.writeable:
        arr = np.array(arr, copy=True)
    return torch.from_numpy(arr)


class WaveformDevicePreprocessor(nn.Module):
    """将 DataLoader 的 raw 波形 batch 组装为模型 NTC/NCT 输入。

    期望 batch 键（按模态出现）：
    - ``slow``: float ``(B, T, C_slow)``
    - ``ultrasonic`` + ``ultrasonic_scale``
    - ``fiber_mic`` + ``fiber_mic_scale``
    """

    def __init__(
        self,
        *,
        modalities: Sequence[str],
        waveform_stats_features: Sequence[str] = (),
        normalize_waveforms: bool = False,
        input_format: str = "NTC",
    ):
        super().__init__()
        self.modalities = tuple(modalities)
        self.waveform_stats_features = tuple(waveform_stats_features)
        self.normalize_waveforms = bool(normalize_waveforms)
        self.input_format = str(input_format).upper()
        if self.input_format not in {"NTC", "NCT"}:
            raise ValueError(f"input_format must be NTC or NCT, got {self.input_format!r}")
        self.waveform_modalities = tuple(
            modality for modality in WAVEFORM_MODALITY_ORDER if modality in self.modalities
        )
        if not self.waveform_modalities:
            raise ValueError("WaveformDevicePreprocessor requires ultrasonic and/or fiber_mic")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        waveform_stat_parts: list[torch.Tensor] = []
        waveform_parts: list[torch.Tensor] = []

        if "slow" in self.modalities:
            if "slow" not in batch:
                raise KeyError("batch missing required key 'slow'")
            parts.append(batch["slow"].float())

        for modality in self.waveform_modalities:
            key = modality
            scale_key = f"{modality}_scale"
            if key not in batch or scale_key not in batch:
                raise KeyError(f"batch missing required keys {key!r} / {scale_key!r}")
            dequantized = dequantize_waveform_torch(batch[key], batch[scale_key])
            if self.waveform_stats_features:
                waveform_stat_parts.append(
                    waveform_stats_torch(dequantized, self.waveform_stats_features)
                )
            if self.normalize_waveforms:
                dequantized = normalize_waveform_torch(dequantized)
            waveform_parts.append(dequantized)

        parts.extend(waveform_stat_parts)
        parts.extend(waveform_parts)
        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        if self.input_format == "NCT":
            x = x.transpose(1, 2)
        return x


def move_raw_batch_to_device(
    batch: dict[str, object],
    *,
    device: torch.device,
    non_blocking: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """把 raw dict batch 搬到 device，拆出 aux_targets。"""
    aux_targets: dict[str, torch.Tensor] | None = None
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "aux_targets":
            if not isinstance(value, dict):
                raise TypeError("aux_targets must be a dict of tensors")
            aux_targets = {
                aux_key: aux_value.to(device, non_blocking=non_blocking)
                for aux_key, aux_value in value.items()
            }
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"batch[{key!r}] must be a Tensor, got {type(value).__name__}")
        moved[key] = value.to(device, non_blocking=non_blocking)
    return moved, aux_targets


def model_kwargs_from_raw_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: batch[key] for key in MODEL_INPUT_KWARG_KEYS if key in batch}
