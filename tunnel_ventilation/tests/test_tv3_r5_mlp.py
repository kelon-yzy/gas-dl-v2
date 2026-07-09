from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.ml.mlp_head import MlpHeadConfig, _ScaledMLPRegressor
from tv3.ml.rocket_training import _build_head, train_tv3_rocket_regressor
from tv3.ml.rocket_features import RocketFeatureConfig
from tv3.pipeline.run_tv3_rocket_baseline import main


def _make_tv3_smoke_dataset(tmp_path: Path, slug: str = "tv3-mlp-smoke", sequences: int = 32) -> Path:
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


def _d0_observed_feature_config() -> RocketFeatureConfig:
    return RocketFeatureConfig(
        feature_builder="d0_observed_physics_stats_v1",
        physics_arrays=(
            "ultrasonic_tof_observed_s",
            "ultrasonic_peak_index",
            "ultrasonic_sound_speed_estimated_m_per_s",
            "ultrasonic_tof_quality",
            "ultrasonic_tof_accepted",
        ),
        sequence_statistics=("mean", "std", "delta"),
        phase_windows=("baseline", "steady"),
        early_fractions=(0.5,),
    )


def test_scaled_mlp_regressor_predict_shape_and_finite(tmp_path: Path):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path)
    result = train_tv3_rocket_regressor(
        dataset_dir,
        feature_config=_d0_observed_feature_config(),
        head="mlp",
        mlp_config=MlpHeadConfig(
            hidden_dims=(32, 16),
            batch_size=8,
            max_epochs=3,
            patience=2,
            device="cpu",
            seed=42,
        ),
    )
    train_eval = result.evaluations["train"]
    predictions = train_eval.predictions

    assert predictions.shape[1] == 3
    assert predictions.shape[0] == len(train_eval.sequence_ids)
    assert np.isfinite(predictions).all()
    assert result.diagnostics["parameter_count"] > 0
    assert 1 <= result.diagnostics["best_epoch"] <= 3


def test_scaled_mlp_regressor_early_stop_rolls_back_best_checkpoint():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(64, 16))
    y_train = rng.normal(size=(64, 3))
    x_val = rng.normal(size=(32, 16))
    y_val = rng.normal(size=(32, 3))
    label_names = ("x_CO2", "x_O2", "x_N2")

    model = _ScaledMLPRegressor(
        config=MlpHeadConfig(
            hidden_dims=(32,),
            batch_size=16,
            max_epochs=20,
            patience=2,
            device="cpu",
            seed=0,
        )
    )
    model.fit(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        label_names=label_names,
    )

    assert model.best_epoch >= 1
    assert model.best_epoch < 20
    predictions = model.predict(x_val)
    assert predictions.shape == (32, 3)
    assert np.isfinite(predictions).all()


def test_build_head_mlp_is_available():
    model = _build_head(
        "mlp",
        ridge_alphas=(1.0,),
        closed_form_alpha=1.0,
        mlp_config=MlpHeadConfig(device="cpu"),
    )
    assert isinstance(model, _ScaledMLPRegressor)


def test_run_tv3_rocket_baseline_mlp_head_writes_metrics_json(tmp_path: Path, capsys):
    dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-mlp-cli-smoke")
    output_dir = tmp_path / "outputs" / "tv3_mlp_smoke"
    config_path = tmp_path / "tv3_mlp_smoke.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "output_dir": str(output_dir),
                "feature_set": "physics_stats",
                "head": "mlp",
                "feature_builder": "d0_observed_physics_stats_v1",
                "physics_arrays": list(_d0_observed_feature_config().physics_arrays),
                "sequence_statistics": list(_d0_observed_feature_config().sequence_statistics),
                "phase_windows": list(_d0_observed_feature_config().phase_windows),
                "early_fractions": list(_d0_observed_feature_config().early_fractions),
                "mlp_hidden_dims": [32, 16],
                "mlp_batch_size": 8,
                "mlp_max_epochs": 3,
                "mlp_patience": 2,
                "device": "cpu",
                "seed": 42,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["--config", str(config_path)])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["head"] == "mlp"
    assert payload["feature_builder"] == "d0_observed_physics_stats_v1"
    assert payload["diagnostics"]["parameter_count"] > 0
    assert set(payload["evaluations"]) == {"train", "val", "test", "extrapolation"}


def test_scaled_mlp_regressor_requires_val_split_for_fit():
    model = _ScaledMLPRegressor(config=MlpHeadConfig(device="cpu", max_epochs=1))
    with pytest.raises(ValueError, match="requires x_val and y_val"):
        model.fit(np.ones((4, 3)), np.ones((4, 3)))


def test_scaled_mlp_regressor_rejects_mismatched_train_row_counts():
    model = _ScaledMLPRegressor(config=MlpHeadConfig(device="cpu", max_epochs=1))
    with pytest.raises(ValueError, match="x/y row counts must match"):
        model.fit(
            np.ones((4, 3)),
            np.ones((3, 3)),
            x_val=np.ones((2, 3)),
            y_val=np.ones((2, 3)),
            label_names=("x_CO2", "x_O2", "x_N2"),
        )


def test_scaled_mlp_regressor_rejects_mismatched_val_row_counts():
    model = _ScaledMLPRegressor(config=MlpHeadConfig(device="cpu", max_epochs=1))
    with pytest.raises(ValueError, match="x_val/y_val row counts must match"):
        model.fit(
            np.ones((4, 3)),
            np.ones((4, 3)),
            x_val=np.ones((2, 3)),
            y_val=np.ones((3, 3)),
            label_names=("x_CO2", "x_O2", "x_N2"),
        )
