from __future__ import annotations

from pathlib import Path

import numpy as np

from common.windows import WINDOW_KIND_PHASE, WindowConfig
from ml.features import MLFeatureConfig, load_feature_matrix
from scripts.precompute_phase_stats import precompute
from sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset


def test_precomputed_phase_stats_match_ml_multiwindow_features(tmp_path: Path):
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug="phase-stats-smoke",
            sequence_count=16,
            seed=9,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
        ),
    )
    dataset_dir = tmp_path / "phase-stats-smoke"

    out_path = precompute(dataset_dir, force=True)

    stats = np.load(out_path)
    master_ids = [str(value) for value in np.load(dataset_dir / "metadata" / "sequence_ids.npy", allow_pickle=True)]
    id_to_index = {sequence_id: index for index, sequence_id in enumerate(master_ids)}
    config = MLFeatureConfig(
        modalities=("slow", "ultrasonic", "fiber_mic"),
        feature_windows=(
            None,
            WindowConfig(kind=WINDOW_KIND_PHASE, value="exposure"),
            WindowConfig(kind=WINDOW_KIND_PHASE, value="recovery"),
        ),
    )
    train_matrix = load_feature_matrix(dataset_dir, split="train", config=config)
    train_indices = [id_to_index[sequence_id] for sequence_id in train_matrix.sequence_ids]

    np.testing.assert_allclose(stats[train_indices], train_matrix.x, rtol=1e-6, atol=1e-6)
    assert stats.shape[1] == 420
    assert (dataset_dir / "features" / "phase_stats_scaler.json").is_file()
