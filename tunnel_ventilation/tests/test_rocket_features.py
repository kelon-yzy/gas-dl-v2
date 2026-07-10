from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.ml.rocket_features import (
    RAW_DSP_FORBIDDEN_SIMULATOR_ARRAYS,
    RocketFeatureConfig,
    build_tv3_physics_feature_cache,
    d0_raw_dsp_feature_config,
    load_cached_split_feature_matrix,
    validate_d0_raw_dsp_feature_config,
)
from tv3.pipeline.build_tv3_raw_dsp_features import (
    build_tv3_raw_dsp_feature_cache,
    preflight_tv3_raw_dsp_dataset,
)


def _make_tv3_smoke_dataset(tmp_path: Path, slug: str = "tv3-rocket-smoke", sequences: int = 16) -> Path:
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


def test_d0_raw_dsp_feature_builder_reads_derived_cache_with_frozen_d0_statistics(tmp_path: Path):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, sequences=16)
    preflight = preflight_tv3_raw_dsp_dataset(dataset_dir)
    raw_dsp_cache = build_tv3_raw_dsp_feature_cache(
        preflight,
        template_mode="train_baseline_median",
        template_max_frames=32,
        workers=1,
    )
    config = d0_raw_dsp_feature_config()

    cache = build_tv3_physics_feature_cache(dataset_dir, config=config)
    manifest = json.loads((cache.cache_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    val = load_cached_split_feature_matrix(dataset_dir, cache.cache_dir, split="val")

    assert raw_dsp_cache.cache_dir == dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1"
    assert manifest["feature_builder"] == "d0_raw_dsp_physics_stats_v1"
    assert Path(manifest["source_array_root"]) == Path("features") / "raw_dsp" / "raw_dsp_frame_v1"
    assert manifest["slow_channels"] == list(config.slow_channels)
    assert set(manifest["source_arrays"]).isdisjoint(RAW_DSP_FORBIDDEN_SIMULATOR_ARRAYS)
    assert any(
        name.startswith("ph_steady|physics:ultrasonic_sound_speed_raw_dsp_m_per_s:")
        for name in val.feature_names
    )
    assert np.isfinite(val.x).all()


def test_d0_raw_dsp_feature_contract_rejects_simulator_observed_array():
    config = d0_raw_dsp_feature_config()
    invalid = RocketFeatureConfig(
        feature_builder=config.feature_builder,
        include_slow=config.include_slow,
        slow_channels=config.slow_channels,
        physics_arrays=config.physics_arrays + ("ultrasonic_tof_observed_s",),
        sequence_statistics=config.sequence_statistics,
        phase_windows=config.phase_windows,
        early_fractions=config.early_fractions,
    )

    with pytest.raises(ValueError, match="frozen D0-RawDSP feature contract"):
        validate_d0_raw_dsp_feature_config(invalid)


def test_d2b_ridge_config_matches_frozen_raw_dsp_contract():
    project_root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (project_root / "configs" / "tv3_d2b_raw_dsp_ridge.json").read_text(encoding="utf-8")
    )
    config = d0_raw_dsp_feature_config()

    assert payload["feature_builder"] == config.feature_builder
    assert payload["slow_channels"] == list(config.slow_channels)
    assert payload["physics_arrays"] == list(config.physics_arrays)
    assert payload["sequence_statistics"] == list(config.sequence_statistics)
    assert payload["phase_windows"] == list(config.phase_windows)
    assert payload["early_fractions"] == list(config.early_fractions)
