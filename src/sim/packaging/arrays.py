from __future__ import annotations

from pathlib import Path

import numpy as np


def write_arrays(output_dir: Path, arrays: dict[str, object], labels: np.ndarray, sequence_ids: list[str], slow_channel_names: tuple[str, ...], label_names: tuple[str, ...], storage: str) -> dict[str, list[int]]:
    sequences_dir = output_dir / "sequences"
    labels_dir = output_dir / "labels"
    metadata_dir = output_dir / "metadata"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    slow = arrays["slow"]
    ultrasonic = arrays["ultrasonic"]
    ultrasonic_scale = arrays["ultrasonic_scale"]
    ultrasonic_tof_s = arrays["ultrasonic_tof_s"]
    ultrasonic_peak_index = arrays["ultrasonic_peak_index"]
    ultrasonic_sound_speed = arrays["ultrasonic_sound_speed_m_per_s"]
    ultrasonic_alpha = arrays["ultrasonic_alpha_true_npm"]
    fiber_mic = arrays["fiber_mic"]
    fiber_mic_scale = arrays["fiber_mic_scale"]

    _write_npy(sequences_dir / "slow.npy", slow, use_memmap=storage in {"memmap", "both"})
    _write_npy(sequences_dir / "ultrasonic_int16.npy", ultrasonic, use_memmap=storage in {"memmap", "both"})
    _write_npy(sequences_dir / "ultrasonic_scale.npy", ultrasonic_scale, use_memmap=storage in {"memmap", "both"})
    _write_npy(sequences_dir / "ultrasonic_tof_s.npy", ultrasonic_tof_s, use_memmap=storage in {"memmap", "both"})
    _write_npy(sequences_dir / "ultrasonic_peak_index.npy", ultrasonic_peak_index, use_memmap=storage in {"memmap", "both"})
    _write_npy(sequences_dir / "ultrasonic_sound_speed_m_per_s.npy", ultrasonic_sound_speed, use_memmap=storage in {"memmap", "both"})
    _write_npy(sequences_dir / "ultrasonic_alpha_true_npm.npy", ultrasonic_alpha, use_memmap=storage in {"memmap", "both"})
    _write_npy(sequences_dir / "fiber_mic_int16.npy", fiber_mic, use_memmap=storage in {"memmap", "both"})
    _write_npy(sequences_dir / "fiber_mic_scale.npy", fiber_mic_scale, use_memmap=storage in {"memmap", "both"})
    np.save(labels_dir / "y.npy", labels)
    np.save(metadata_dir / "sequence_ids.npy", np.array(sequence_ids))
    np.save(metadata_dir / "slow_channel_names.npy", np.array(slow_channel_names))
    np.save(metadata_dir / "label_names.npy", np.array(label_names))

    if storage in {"npz", "both"}:
        np.savez_compressed(
            sequences_dir / "waveform_sequence.npz",
            ultrasonic=ultrasonic,
            ultrasonic_scale=ultrasonic_scale,
            ultrasonic_tof_s=ultrasonic_tof_s,
            ultrasonic_peak_index=ultrasonic_peak_index,
            ultrasonic_sound_speed_m_per_s=ultrasonic_sound_speed,
            ultrasonic_alpha_true_npm=ultrasonic_alpha,
            fiber_mic=fiber_mic,
            fiber_mic_scale=fiber_mic_scale,
            slow=slow,
            y=labels,
            sequence_ids=np.array(sequence_ids),
            slow_channel_names=np.array(slow_channel_names),
            label_names=np.array(label_names),
        )

    return {
        "slow": list(slow.shape),
        "ultrasonic": list(ultrasonic.shape),
        "ultrasonic_scale": list(ultrasonic_scale.shape),
        "ultrasonic_tof_s": list(ultrasonic_tof_s.shape),
        "ultrasonic_peak_index": list(ultrasonic_peak_index.shape),
        "ultrasonic_sound_speed_m_per_s": list(ultrasonic_sound_speed.shape),
        "ultrasonic_alpha_true_npm": list(ultrasonic_alpha.shape),
        "fiber_mic": list(fiber_mic.shape),
        "fiber_mic_scale": list(fiber_mic_scale.shape),
        "y": list(labels.shape),
    }


def _write_npy(path: Path, array, *, use_memmap: bool) -> None:
    if use_memmap:
        target = np.lib.format.open_memmap(path, mode="w+", dtype=array.dtype, shape=array.shape)
        target[:] = array
        target.flush()
        return
    np.save(path, array)
