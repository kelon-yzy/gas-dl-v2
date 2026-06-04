from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


ARRAY_FILES = {
    "ultrasonic": "sequences/ultrasonic_int16.npy",
    "ultrasonic_scale": "sequences/ultrasonic_scale.npy",
    "ultrasonic_tof_s": "sequences/ultrasonic_tof_s.npy",
    "ultrasonic_tof_observed_s": "sequences/ultrasonic_tof_observed_s.npy",
    "ultrasonic_peak_index": "sequences/ultrasonic_peak_index.npy",
    "ultrasonic_sound_speed_m_per_s": "sequences/ultrasonic_sound_speed_m_per_s.npy",
    "ultrasonic_sound_speed_estimated_m_per_s": "sequences/ultrasonic_sound_speed_estimated_m_per_s.npy",
    "ultrasonic_alpha_true_npm": "sequences/ultrasonic_alpha_true_npm.npy",
    "ultrasonic_tof_quality": "sequences/ultrasonic_tof_quality.npy",
    "ultrasonic_tof_accepted": "sequences/ultrasonic_tof_accepted.npy",
    "fiber_mic": "sequences/fiber_mic_int16.npy",
    "fiber_mic_scale": "sequences/fiber_mic_scale.npy",
    "slow": "sequences/slow.npy",
    "y": "labels/y.npy",
    "sequence_ids": "metadata/sequence_ids.npy",
    "slow_channel_names": "metadata/slow_channel_names.npy",
    "label_names": "metadata/label_names.npy",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bundle generated waveform arrays into sequences/waveform_sequence.npz.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser


def bundle_waveform_sequence(dataset_dir: Path | str, *, overwrite: bool = False) -> dict[str, object]:
    dataset_dir = Path(dataset_dir)
    target = dataset_dir / "sequences" / "waveform_sequence.npz"
    if target.exists() and not overwrite:
        raise FileExistsError(f"waveform bundle already exists: {target}")
    missing = [relative for relative in ARRAY_FILES.values() if not (dataset_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"dataset is missing required array files: {missing}")
    arrays = {name: np.load(dataset_dir / relative, mmap_mode="r") for name, relative in ARRAY_FILES.items()}
    np.savez_compressed(target, **arrays)
    return {
        "dataset_dir": str(dataset_dir),
        "bundle_path": str(target),
        "arrays": sorted(arrays),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = bundle_waveform_sequence(args.dataset_dir, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
