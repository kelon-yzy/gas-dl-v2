from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ml.rocket_features import (
    RocketFeatureConfig,
    build_tv3_physics_feature_cache,
    load_cached_split_feature_matrix,
)


def _make_tv3_smoke_dataset(tmp_path: Path, slug: str = "tv3-rocket-smoke", sequences: int = 16) -> Path:
    from sim.generation.tunnel_ventilation import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    generate_tunnel_ventilation_benchmark_dataset(
        tmp_path,
        TunnelVentilationBenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=20260706,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    return tmp_path / slug


def test_build_tv3_physics_feature_cache_writes_manifest_and_split_matrices(tmp_path: Path):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, sequences=16)
    cache = build_tv3_physics_feature_cache(dataset_dir)

    manifest = json.loads((cache.cache_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    feature_names = json.loads((cache.cache_dir / "feature_names.json").read_text(encoding="utf-8"))

    assert manifest["dataset_slug"] == dataset_dir.name
    assert manifest["feature_builder"] == "physics_stats_v1"
    assert manifest["source_arrays"][0] == "ultrasonic_tof_s"
    assert manifest["feature_count"] == len(feature_names) == len(cache.feature_names)
    assert set(manifest["split_sequence_counts"]) == {"train", "val", "test", "extrapolation"}

    train = np.load(cache.cache_dir / "feature_matrix_train.npy")
    val = np.load(cache.cache_dir / "feature_matrix_val.npy")
    assert train.shape[1] == val.shape[1] == len(feature_names)
    assert train.shape[0] == manifest["split_sequence_counts"]["train"]
    assert np.isfinite(train).all()
    assert np.isfinite(val).all()


def test_build_tv3_physics_feature_cache_is_deterministic(tmp_path: Path):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, sequences=12)
    cache_dir = dataset_dir / "features" / "rocket" / "physics_stats_v1"

    build_tv3_physics_feature_cache(dataset_dir, cache_dir=cache_dir)
    first = np.load(cache_dir / "feature_matrix_train.npy")
    build_tv3_physics_feature_cache(dataset_dir, cache_dir=cache_dir)
    second = np.load(cache_dir / "feature_matrix_train.npy")

    np.testing.assert_allclose(first, second)


def test_load_cached_split_feature_matrix_aligns_labels_and_sequence_ids(tmp_path: Path):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, sequences=16)
    cache = build_tv3_physics_feature_cache(
        dataset_dir,
        config=RocketFeatureConfig(slow_channels=("V_NDIR_CO2", "V_TCS")),
    )

    matrix = load_cached_split_feature_matrix(dataset_dir, cache.cache_dir, split="val")

    assert matrix.x.shape[0] == matrix.y.shape[0] == len(matrix.sequence_ids)
    assert matrix.label_names == ("x_CO2", "x_O2", "x_N2")
    assert any(name.startswith("full|slow:V_NDIR_CO2:") for name in matrix.feature_names)
    assert any(name.startswith("ph_exposure|physics:ultrasonic_tof_s:") for name in matrix.feature_names)
    assert np.isfinite(matrix.x).all()

