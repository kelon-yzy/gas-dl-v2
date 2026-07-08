from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tv3.ml.minirocket_features import (
    MINIROCKET_RAW_BUILDER,
    MINIROCKET_SCALAR_BUILDER,
    MiniRocketFeatureConfig,
    build_minirocket_feature_cache,
    generate_fixed_kernels,
)
from tv3.ml.rocket_features import load_cached_split_feature_matrix


def _make_tv3_smoke_dataset(tmp_path: Path, slug: str = "tv3-minirocket-smoke", sequences: int = 16) -> Path:
    from tv3.sim.generation.tunnel_ventilation import (
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


def test_generate_fixed_kernels_is_deterministic():
    first = generate_fixed_kernels(64, (7, 9, 11), seed=42)
    second = generate_fixed_kernels(64, (7, 9, 11), seed=42)
    assert first.lengths == second.lengths
    for w1, w2 in zip(first.weights, second.weights, strict=True):
        np.testing.assert_allclose(w1, w2)
    assert first.biases == second.biases


def test_generate_fixed_kernels_zero_mean():
    kernels = generate_fixed_kernels(32, (7, 9), seed=42)
    for weight in kernels.weights:
        assert abs(float(weight.mean())) < 1e-6, "fixed kernels must be zero-mean"


def test_build_minirocket_scalar_cache_writes_manifest_and_split_matrices(tmp_path: Path):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, sequences=16)
    config = MiniRocketFeatureConfig(
        feature_builder=MINIROCKET_SCALAR_BUILDER,
        physics_arrays=("ultrasonic_tof_s", "ultrasonic_sound_speed_m_per_s"),
        num_kernels=32,
        kernel_lengths=(7, 9),
        kernel_seed=42,
    )
    cache = build_minirocket_feature_cache(dataset_dir, config=config)

    manifest = json.loads((cache.cache_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    feature_names = json.loads((cache.cache_dir / "feature_names.json").read_text(encoding="utf-8"))

    assert manifest["feature_builder"] == MINIROCKET_SCALAR_BUILDER
    assert manifest["kernel_count"] == 32
    assert manifest["kernel_seed"] == 42
    assert manifest["feature_count"] == len(feature_names) == len(cache.feature_names)
    assert set(manifest["split_sequence_counts"]) == {"train", "val", "test", "extrapolation"}
    assert "minirocket_scalar" in manifest["modalities"]

    train = np.load(cache.cache_dir / "feature_matrix_train.npy")
    assert train.shape[1] == len(feature_names)
    assert train.shape[0] == manifest["split_sequence_counts"]["train"]
    assert np.isfinite(train).all()
    # R1a:每 array 出 num_kernels*2 维(PPV+max),2 array + slow(8 通道 * 5 统计)
    assert any(name.startswith("minirocket_scalar|ultrasonic_tof_s:kernel0:ppv") for name in feature_names)


def test_build_minirocket_raw_cache_writes_manifest_and_split_matrices(tmp_path: Path):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-minirocket-raw-smoke", sequences=12)
    config = MiniRocketFeatureConfig(
        feature_builder=MINIROCKET_RAW_BUILDER,
        num_kernels=16,
        kernel_lengths=(7, 9),
        kernel_seed=42,
        raw_zscore=True,
        include_slow=True,
    )
    cache = build_minirocket_feature_cache(dataset_dir, config=config)

    manifest = json.loads((cache.cache_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    feature_names = json.loads((cache.cache_dir / "feature_names.json").read_text(encoding="utf-8"))

    assert manifest["feature_builder"] == MINIROCKET_RAW_BUILDER
    assert manifest["kernel_count"] == 16
    assert manifest["raw_zscore"] is True
    assert manifest["source_arrays"] == ["ultrasonic_raw"]
    assert "minirocket_raw" in manifest["modalities"]
    assert manifest["feature_count"] == len(feature_names)

    val = np.load(cache.cache_dir / "feature_matrix_val.npy")
    assert val.shape[1] == len(feature_names)
    assert np.isfinite(val).all()
    # R1b:每核每统计前缀 minirocket_raw
    assert any(name.startswith("minirocket_raw:kernel0:ppv:") for name in feature_names)


def test_minirocket_cache_is_deterministic(tmp_path: Path):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-minirocket-det", sequences=12)
    cache_dir = dataset_dir / "features" / "rocket" / MINIROCKET_SCALAR_BUILDER
    config = MiniRocketFeatureConfig(
        feature_builder=MINIROCKET_SCALAR_BUILDER,
        physics_arrays=("ultrasonic_tof_s",),
        num_kernels=16,
        kernel_seed=42,
    )
    build_minirocket_feature_cache(dataset_dir, cache_dir=cache_dir, config=config)
    first = np.load(cache_dir / "feature_matrix_train.npy")
    build_minirocket_feature_cache(dataset_dir, cache_dir=cache_dir, config=config)
    second = np.load(cache_dir / "feature_matrix_train.npy")
    np.testing.assert_allclose(first, second)


def test_minirocket_cache_feature_names_stable_across_splits(tmp_path: Path):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, sequences=16)
    config = MiniRocketFeatureConfig(
        feature_builder=MINIROCKET_SCALAR_BUILDER,
        physics_arrays=("ultrasonic_tof_s",),
        num_kernels=8,
        kernel_seed=42,
    )
    cache = build_minirocket_feature_cache(dataset_dir, config=config)
    train = load_cached_split_feature_matrix(dataset_dir, cache.cache_dir, split="train")
    val = load_cached_split_feature_matrix(dataset_dir, cache.cache_dir, split="val")
    assert train.feature_names == val.feature_names == cache.feature_names
    assert train.label_names == val.label_names == ("x_CO2", "x_O2", "x_N2")
