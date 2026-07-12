from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from tv3.ml.extratrees_head import ExtraTreesHeadConfig
from tv3.ml.extratrees_training import train_tv3_extratrees_regressor
from tv3.ml.rocket_features import (
    RocketFeatureConfig,
    d0_observed_feature_config,
)
from tv3.pipeline.run_tv3_extratrees_baseline import _build_feature_config, _load_config, main


def _make_tv3_smoke_dataset(tmp_path: Path, slug: str = "tv3-extratrees-smoke", sequences: int = 32) -> Path:
    from tv3.sim.generation.tunnel_ventilation import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    generate_tunnel_ventilation_benchmark_dataset(
        tmp_path,
        TunnelVentilationBenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=20260710,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    return tmp_path / slug


def _d0_observed_feature_config() -> RocketFeatureConfig:
    return d0_observed_feature_config()


def test_extratrees_raw3_regressor_predicts_finite_raw3(tmp_path: Path):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path)
    result = train_tv3_extratrees_regressor(
        dataset_dir,
        feature_config=_d0_observed_feature_config(),
        extratrees_config=ExtraTreesHeadConfig(
            n_estimators=8,
            max_features=0.7,
            min_samples_leaf=1,
            n_jobs=1,
            seed=42,
        ),
    )

    train_eval = result.evaluations["train"]
    assert train_eval.predictions.shape == (len(train_eval.sequence_ids), 3)
    assert np.isfinite(train_eval.predictions).all()
    assert len(result.feature_names) == 864
    assert result.diagnostics["model_config"]["n_estimators"] == 8
    assert result.diagnostics["model_config"]["seed"] == 42
    assert len(result.diagnostics["feature_importances"]) == len(result.feature_names)


def test_extratrees_rejects_oracle_feature_config_before_loading_dataset(tmp_path: Path):
    observed = d0_observed_feature_config()
    oracle = replace(
        observed,
        physics_arrays=("ultrasonic_tof_s", *observed.physics_arrays[1:]),
    )

    with pytest.raises(ValueError, match="frozen D0-observed feature contract"):
        train_tv3_extratrees_regressor(tmp_path / "missing", feature_config=oracle)


def test_r7_formal_config_matches_frozen_d0_observed_contract():
    project_root = Path(__file__).resolve().parents[1]
    payload = _load_config(project_root / "configs" / "tv3_r7_extratrees_observed.json")

    assert _build_feature_config(payload) == d0_observed_feature_config()


def test_extratrees_runner_writes_metrics_json(tmp_path: Path, capsys):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-extratrees-cli-smoke")
    output_dir = tmp_path / "outputs" / "tv3_extratrees_smoke"
    config_path = tmp_path / "tv3_extratrees_smoke.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "output_dir": str(output_dir),
                "feature_builder": "d0_observed_physics_stats_v1",
                "include_slow": True,
                "physics_arrays": list(_d0_observed_feature_config().physics_arrays),
                "sequence_statistics": list(_d0_observed_feature_config().sequence_statistics),
                "phase_windows": list(_d0_observed_feature_config().phase_windows),
                "early_fractions": list(_d0_observed_feature_config().early_fractions),
                "eval_splits": ["val", "test", "extrapolation"],
                "extratrees_n_estimators": 8,
                "extratrees_max_features": 0.7,
                "extratrees_min_samples_leaf": 1,
                "extratrees_max_depth": None,
                "extratrees_n_jobs": 1,
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["head"] == "extratrees"
    assert payload["feature_builder"] == "d0_observed_physics_stats_v1"
    assert payload["feature_count"] == 864
    assert payload["feature_config"]["physics_arrays"] == list(_d0_observed_feature_config().physics_arrays)
    assert payload["diagnostics"]["model_config"]["seed"] == 42
    assert set(payload["evaluations"]) == {"train", "val", "test", "extrapolation"}
    assert (output_dir / "metrics.json").is_file()
