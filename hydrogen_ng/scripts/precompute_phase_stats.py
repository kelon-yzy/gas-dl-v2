"""
Precompute phase-window statistics for P3-A phase-stat branch.

For each sequence, computes 3 windows (full, exposure, recovery) × (slow stats +
acoustic frame stats) = 420-d feature vector, saved as
`<dataset>/features/phase_stats.npy` (shape: [N_sequences, 420]).

Usage (on server where data exists):
    PYTHONPATH=src python scripts/precompute_phase_stats.py \
        --dataset-dir data/wv4-formal-hitran-standard-6000

The output matches the same statistic set used by Ridge multiwindow ML features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hg.common.splits import load_splits, resolve_split_indices
from hg.common.windows import WINDOW_KIND_PHASE, WindowConfig
from hg.ml.features import (
    DEFAULT_SEQUENCE_STATISTICS,
    DEFAULT_WAVEFORM_FRAME_FEATURES,
    MLFeatureConfig,
    load_feature_matrix,
)

PHASE_STAT_WINDOWS = (
    None,
    WindowConfig(kind=WINDOW_KIND_PHASE, value="exposure"),
    WindowConfig(kind=WINDOW_KIND_PHASE, value="recovery"),
)


def _build_sequence_ids(dataset_dir: Path) -> list[str]:
    """Read sequence_ids from metadata/sequence_ids.npy."""
    ids_path = dataset_dir / "metadata" / "sequence_ids.npy"
    if not ids_path.is_file():
        raise FileNotFoundError(f"sequence_ids metadata not found: {ids_path}")
    return [str(x) for x in np.load(ids_path, allow_pickle=True).tolist()]


def precompute(dataset_dir: Path, *, force: bool = False) -> Path:
    """Main entry point."""
    csv_path = dataset_dir / "sequences" / "slow_sequence_long.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"slow_sequence_long.csv not found: {csv_path}")

    out_dir = dataset_dir / "features"
    out_path = out_dir / "phase_stats.npy"
    if out_path.is_file() and not force:
        print(f"Phase stats already exist: {out_path}")
        return out_path

    sequence_ids = _build_sequence_ids(dataset_dir)
    n_sequences = len(sequence_ids)
    print(f"Dataset: {dataset_dir}")
    print(f"Sequences: {n_sequences}")

    config = MLFeatureConfig(
        modalities=("slow", "ultrasonic", "fiber_mic"),
        sequence_statistics=DEFAULT_SEQUENCE_STATISTICS,
        waveform_frame_features=DEFAULT_WAVEFORM_FRAME_FEATURES,
        feature_windows=PHASE_STAT_WINDOWS,
    )
    stats = _build_all_sequence_feature_matrix(dataset_dir, sequence_ids, config)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_path, stats)
    
    # Save Z-score normalizer computed on train split only
    scaler_path = out_dir / "phase_stats_scaler.json"
    _save_scaler(stats, sequence_ids, dataset_dir, scaler_path)
    
    print(f"\nSaved: {out_path}")
    print(f"Shape: {stats.shape}  dtype: {stats.dtype}")
    print(f"Min: {stats.min():.6f}  Max: {stats.max():.6f}  Mean: {stats.mean():.6f}  Std: {stats.std():.6f}")
    return out_path


def _save_scaler(
    stats: NDArray[np.float32],
    sequence_ids: list[str],
    dataset_dir: Path,
    scaler_path: Path,
) -> None:
    """Compute and save Z-score normalizer from train split only."""
    splits = load_splits(dataset_dir / "splits")
    train_indices = resolve_split_indices(splits, sequence_ids)["train"]
    train_stats = stats[train_indices]
    
    mean = train_stats.mean(axis=0).tolist()
    std = train_stats.std(axis=0).tolist()
    
    # Clip tiny std to avoid division by near-zero
    eps = 1e-8
    std = [max(s, eps) for s in std]
    
    scaler_path.write_text(json.dumps({"mean": mean, "std": std}, indent=2), encoding="utf-8")
    print(f"Saved scaler: {scaler_path}")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Mean range: [{np.min(mean):.4f}, {np.max(mean):.4f}]")
    print(f"  Std range: [{np.min(std):.6f}, {np.max(std):.4f}]")


def _build_all_sequence_feature_matrix(
    dataset_dir: Path,
    sequence_ids: list[str],
    config: MLFeatureConfig,
) -> NDArray[np.float32]:
    splits = load_splits(dataset_dir / "splits")
    id_to_index = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}
    seen = np.zeros(len(sequence_ids), dtype=bool)
    stats: NDArray[np.float32] | None = None

    for split in splits:
        matrix = load_feature_matrix(dataset_dir, split=split, config=config)
        if stats is None:
            stats = np.empty((len(sequence_ids), matrix.x.shape[1]), dtype=np.float32)
        elif matrix.x.shape[1] != stats.shape[1]:
            raise ValueError("phase-stat feature dimension changed across splits")
        for row_index, sequence_id in enumerate(matrix.sequence_ids):
            if sequence_id not in id_to_index:
                raise ValueError(f"split {split!r} contains unknown sequence_id={sequence_id!r}")
            src_idx = id_to_index[sequence_id]
            if seen[src_idx]:
                raise ValueError(f"sequence_id={sequence_id!r} appears in more than one split")
            stats[src_idx] = matrix.x[row_index]
            seen[src_idx] = True
        print(f"  Loaded {split}: {len(matrix.sequence_ids)} sequences")

    if stats is None:
        raise ValueError("no split rows found for phase-stat precompute")
    if not bool(seen.all()):
        missing = [sequence_ids[index] for index, value in enumerate(seen) if not value]
        raise ValueError(f"split files do not cover all sequence_ids: {missing[:5]}")
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute phase-window statistics for P3-A.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Path to v4 benchmark dataset root.")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Overwrite existing phase_stats.npy.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    precompute(args.dataset_dir, force=args.force)


if __name__ == "__main__":
    main()
