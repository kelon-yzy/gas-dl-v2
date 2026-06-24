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
import csv
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

WINDOW_PHASES = ("full", "exposure", "recovery")
SLOW_CHANNEL_NAMES: tuple[str, ...] = (
    "V_NDIR_CH4", "V_NDIR_CO2", "V_TCS", "T_C",
    "P_MPa", "H_RH", "L_m", "piston_position_m",
)
FRAME_STAT_NAMES: tuple[str, ...] = (
    "mean", "std", "mean_abs", "max_abs", "energy", "peak_index",
)
CROSS_TIMESTEP_STAT_NAMES: tuple[str, ...] = (
    "mean", "std", "min", "max", "last", "delta", "slope",
)


def _compute_frame_features(waveform_2d: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute frame-level statistics for one timestep's waveform.

    Args:
        waveform_2d: shape (timesteps, waveform_length_in_samples), raw int16.
    Returns:
        shape (timesteps, 6): [mean, std, mean_abs, max_abs, energy, peak_index].
    """
    wf = waveform_2d.astype(np.float32)
    t = wf.shape[0]
    out = np.empty((t, 6), dtype=np.float32)
    out[:, 0] = wf.mean(axis=1)                # mean
    out[:, 1] = wf.std(axis=1)                  # std
    out[:, 2] = np.abs(wf).mean(axis=1)         # mean_abs
    out[:, 3] = np.abs(wf).max(axis=1)          # max_abs
    out[:, 4] = (wf * wf).mean(axis=1)          # energy
    out[:, 5] = np.argmax(np.abs(wf), axis=1).astype(np.float32)  # peak_index
    return out


def _cross_timestep_stats(data_2d: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute 7 cross-timestep statistics for each column of a (timesteps, features) array.

    Returns:
        shape (features * 7,): [col0_mean, col0_std, ..., col0_slope,
                                 col1_mean, ..., colF_slope].
    """
    t = data_2d.shape[0]
    if t == 0:
        return np.zeros(data_2d.shape[1] * 7, dtype=np.float32)
    f = data_2d.shape[1]
    mean_v = data_2d.mean(axis=0)
    std_v = data_2d.std(axis=0)
    min_v = data_2d.min(axis=0)
    max_v = data_2d.max(axis=0)
    last_v = data_2d[-1] if t > 0 else np.zeros(f)
    delta_v = data_2d[-1] - data_2d[0] if t > 0 else np.zeros(f)

    # slope: linear fit slope for each feature
    slope_v = np.zeros(f, dtype=np.float32)
    if t >= 2:
        x = np.arange(t, dtype=np.float32)
        x_mean = x.mean()
        x_demean = x - x_mean
        x_var = (x_demean * x_demean).sum()
        if x_var > 0:
            for i in range(f):
                y = data_2d[:, i]
                slope_v[i] = float(((x_demean * (y - y.mean())).sum() / x_var))

    result = np.empty(f * 7, dtype=np.float32)
    for i in range(f):
        base = i * 7
        result[base + 0] = mean_v[i]
        result[base + 1] = std_v[i]
        result[base + 2] = min_v[i]
        result[base + 3] = max_v[i]
        result[base + 4] = last_v[i]
        result[base + 5] = delta_v[i]
        result[base + 6] = slope_v[i]
    return result


def _load_phase_rows(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    """Parse slow_sequence_long.csv, group rows by sequence_id."""
    rows_by_seq: dict[str, list[dict[str, str]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_by_seq.setdefault(str(row["sequence_id"]), []).append(row)
    return rows_by_seq


def _build_sequence_ids(dataset_dir: Path) -> list[str]:
    """Read sequence_ids from metadata/sequence_ids.npy, fallback to CSV."""
    ids_path = dataset_dir / "metadata" / "sequence_ids.npy"
    if ids_path.is_file():
        return [str(x) for x in np.load(ids_path)]
    # fallback: parse slow_sequence_long.csv
    csv_path = dataset_dir / "sequences" / "slow_sequence_long.csv"
    phase_rows = _load_phase_rows(csv_path)
    return sorted(phase_rows, key=lambda sid: sid)


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

    # Load phase rows
    phase_rows = _load_phase_rows(csv_path)
    print(f"Phase CSV rows loaded for {len(phase_rows)} sequences")

    # Open mmap arrays
    seq_dir = dataset_dir / "sequences"
    slow = np.load(seq_dir / "slow.npy", mmap_mode="r")          # (N, T, 8)
    ultrasonic = np.load(seq_dir / "ultrasonic_int16.npy", mmap_mode="r")  # (N, T, 1000)
    fiber_mic = np.load(seq_dir / "fiber_mic_int16.npy", mmap_mode="r")    # (N, T, 2000)

    n_slow = SLOW_CHANNEL_NAMES.__len__()        # 8
    n_frame = FRAME_STAT_NAMES.__len__()          # 6
    n_cross = CROSS_TIMESTEP_STAT_NAMES.__len__()  # 7
    n_windows = WINDOW_PHASES.__len__()           # 3

    # slow: 8 ch × 7 stats = 56
    # ultrasonic: 6 frame feat × 7 stats = 42
    # fiber_mic: same = 42
    # per window = 56 + 42 + 42 = 140
    # total = 140 × 3 = 420
    per_window = n_slow * n_cross + n_frame * n_cross + n_frame * n_cross  # 140
    total_dim = per_window * n_windows  # 420

    stats = np.empty((n_sequences, total_dim), dtype=np.float32)

    for src_idx, sequence_id in enumerate(sequence_ids):
        rows = phase_rows.get(sequence_id)
        if not rows:
            raise ValueError(f"No phase rows for sequence_id={sequence_id!r}")

        # Build per-phase masks
        phase_ids = [r["phase_id"] for r in rows]
        full_mask = np.ones(len(phase_ids), dtype=bool)
        exposure_mask = np.array([p == "exposure" for p in phase_ids], dtype=bool)
        recovery_mask = np.array([p == "recovery" for p in phase_ids], dtype=bool)
        window_masks = {"full": full_mask, "exposure": exposure_mask, "recovery": recovery_mask}

        # Precompute frame features for the whole sequence once per modality
        us_wf = ultrasonic[src_idx].astype(np.float32)     # (T, 1000)
        fm_wf = fiber_mic[src_idx].astype(np.float32)      # (T, 2000)
        us_frame = _compute_frame_features(us_wf)            # (T, 6)
        fm_frame = _compute_frame_features(fm_wf)            # (T, 6)
        slow_data = slow[src_idx].astype(np.float32)         # (T, 8)

        offset = 0
        for window_name in WINDOW_PHASES:
            m = window_masks[window_name]
            t_count = int(m.sum())

            # slow stats
            if t_count > 0:
                slow_stats = _cross_timestep_stats(slow_data[m])  # (56,)
            else:
                slow_stats = np.zeros(n_slow * n_cross, dtype=np.float32)
            stats[src_idx, offset:offset + n_slow * n_cross] = slow_stats
            offset += n_slow * n_cross

            # ultrasonic frame stats
            if t_count > 0:
                us_stats = _cross_timestep_stats(us_frame[m])  # (42,)
            else:
                us_stats = np.zeros(n_frame * n_cross, dtype=np.float32)
            stats[src_idx, offset:offset + n_frame * n_cross] = us_stats
            offset += n_frame * n_cross

            # fiber_mic frame stats
            if t_count > 0:
                fm_stats = _cross_timestep_stats(fm_frame[m])  # (42,)
            else:
                fm_stats = np.zeros(n_frame * n_cross, dtype=np.float32)
            stats[src_idx, offset:offset + n_frame * n_cross] = fm_stats
            offset += n_frame * n_cross

        if (src_idx + 1) % 500 == 0:
            print(f"  Processed {src_idx + 1}/{n_sequences} sequences")

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
    import json
    
    from src.dl.data.splits import load_splits, resolve_split_indices
    
    splits = load_splits(dataset_dir / "splits")
    train_indices = resolve_split_indices(splits, sequence_ids)["train"]
    train_stats = stats[train_indices]
    
    mean = train_stats.mean(axis=0).tolist()
    std = train_stats.std(axis=0).tolist()
    
    # Clip tiny std to avoid division by near-zero
    eps = 1e-8
    std = [max(s, eps) for s in std]
    
    scaler_path.write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    print(f"Saved scaler: {scaler_path}")
    print(f"  Train samples: {len(train_indices)}")
    print(f"  Mean range: [{np.min(mean):.4f}, {np.max(mean):.4f}]")
    print(f"  Std range: [{np.min(std):.6f}, {np.max(std):.4f}]")


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