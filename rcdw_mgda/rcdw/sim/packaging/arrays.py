"""序列数组落盘：13 个 npy + labels/y.npy + 元数据。

对应方案 §6.1。落盘清单：
- sequences/slow.npy (N, T, 7) float32
- sequences/ultrasonic_int16.npy (N, T, W_us) int16
- sequences/ultrasonic_scale.npy (N, T) float32
- sequences/ultrasonic_tof_s.npy (N, T) float32
- sequences/ultrasonic_tof_observed_s.npy (N, T) float32
- sequences/ultrasonic_peak_index.npy (N, T) int32
- sequences/ultrasonic_sound_speed_m_per_s.npy (N, T) float32
- sequences/ultrasonic_sound_speed_estimated_m_per_s.npy (N, T) float32
- sequences/ultrasonic_alpha_true_npm.npy (N, T) float32
- sequences/ultrasonic_tof_quality.npy (N, T) float32
- sequences/ultrasonic_tof_accepted.npy (N, T) int8
- sequences/fiber_mic_int16.npy (N, T, W_fm) int16
- sequences/fiber_mic_scale.npy (N, T) float32
- labels/y.npy (N, 3) float32
- metadata/sequence_ids.npy, slow_channel_names.npy, label_names.npy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


_NPY_ENTRIES: tuple[tuple[str, str], ...] = (
    # (sequences/ 文件名前缀, arrays dict key)
    ("slow", "slow"),
    ("ultrasonic_int16", "ultrasonic"),
    ("ultrasonic_scale", "ultrasonic_scale"),
    ("ultrasonic_tof_s", "ultrasonic_tof_s"),
    ("ultrasonic_tof_observed_s", "ultrasonic_tof_observed_s"),
    ("ultrasonic_peak_index", "ultrasonic_peak_index"),
    ("ultrasonic_sound_speed_m_per_s", "ultrasonic_sound_speed_m_per_s"),
    (
        "ultrasonic_sound_speed_estimated_m_per_s",
        "ultrasonic_sound_speed_estimated_m_per_s",
    ),
    ("ultrasonic_alpha_true_npm", "ultrasonic_alpha_true_npm"),
    ("ultrasonic_tof_quality", "ultrasonic_tof_quality"),
    ("ultrasonic_tof_accepted", "ultrasonic_tof_accepted"),
    ("fiber_mic_int16", "fiber_mic"),
    ("fiber_mic_scale", "fiber_mic_scale"),
)


def write_arrays(
    output_dir: Path,
    arrays: dict[str, Any],
    labels: np.ndarray,
    sequence_ids: list[str],
    slow_channel_names: tuple[str, ...],
    label_names: tuple[str, ...],
    storage: str,
) -> dict[str, list[int]]:
    """落盘 13 个 sequence 数组 + labels + 元数据，返回各数组 shape 字典。"""
    sequences_dir = output_dir / "sequences"
    labels_dir = output_dir / "labels"
    metadata_dir = output_dir / "metadata"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    use_memmap = storage in {"memmap", "both"}
    for file_stem, array_key in _NPY_ENTRIES:
        _write_npy(
            sequences_dir / f"{file_stem}.npy",
            arrays[array_key],
            use_memmap=use_memmap,
        )

    np.save(labels_dir / "y.npy", labels)
    np.save(metadata_dir / "sequence_ids.npy", np.array(sequence_ids))
    np.save(metadata_dir / "slow_channel_names.npy", np.array(slow_channel_names))
    np.save(metadata_dir / "label_names.npy", np.array(label_names))

    if storage in {"npz", "both"}:
        npz_kwargs = {key: arrays[key] for _, key in _NPY_ENTRIES}
        np.savez_compressed(
            sequences_dir / "waveform_sequence.npz",
            **npz_kwargs,
            y=labels,
            sequence_ids=np.array(sequence_ids),
            slow_channel_names=np.array(slow_channel_names),
            label_names=np.array(label_names),
        )

    shapes = {key: list(arrays[key].shape) for _, key in _NPY_ENTRIES}
    shapes["y"] = list(labels.shape)
    return shapes


def _write_npy(path: Path, array: np.ndarray, *, use_memmap: bool) -> None:
    if use_memmap:
        target = np.lib.format.open_memmap(
            path, mode="w+", dtype=array.dtype, shape=array.shape
        )
        target[:] = array
        target.flush()
        return
    np.save(path, array)
