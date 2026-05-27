from __future__ import annotations

from pathlib import Path

import numpy as np

from ml import (
    MLFeatureConfig,
    MeanRegressor,
    RegressionMetrics,
    RidgeRegressor,
    build_regressor,
    component_regression_metrics,
    load_feature_matrix,
    regression_metrics,
    sequence_stat_features,
    train_regressor_on_dataset,
)
from sim.core.schema import COMPONENT_FIELDS
from sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset


def _make_smoke_dataset(tmp_path: Path, slug: str = "ml-smoke", sequences: int = 16) -> Path:
    generate_benchmark_dataset(
        tmp_path,
        BenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=123,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
        ),
    )
    return tmp_path / slug


class TestMLFeatures:
    def test_sequence_stat_features_include_expected_names_and_values(self):
        values = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]],
                [[2.0, 1.0], [2.0, 3.0], [2.0, 5.0]],
            ],
            dtype=np.float32,
        )

        x, names = sequence_stat_features(
            values,
            channel_names=("a", "b"),
            statistics=("mean", "last", "delta", "slope"),
            prefix="slow",
        )

        assert names == (
            "slow:a:mean", "slow:b:mean",
            "slow:a:last", "slow:b:last",
            "slow:a:delta", "slow:b:delta",
            "slow:a:slope", "slow:b:slope",
        )
        assert x.shape == (2, 8)
        np.testing.assert_allclose(x[0], [3.0, 14.0 / 3.0, 5.0, 8.0, 4.0, 6.0, 2.0, 3.0])
        np.testing.assert_allclose(x[1], [2.0, 3.0, 2.0, 5.0, 0.0, 4.0, 0.0, 2.0])

    def test_load_feature_matrix_slow_only_matches_split_rows(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-feature-slow")
        matrix = load_feature_matrix(
            dataset_dir,
            split="train",
            config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean", "last")),
        )

        assert matrix.x.ndim == 2
        assert matrix.x.shape[0] == matrix.y.shape[0] == len(matrix.sequence_ids)
        assert matrix.x.shape[1] == 8 * 2
        assert matrix.label_names == COMPONENT_FIELDS
        assert all(name.startswith("slow:") for name in matrix.feature_names)

    def test_load_feature_matrix_can_include_waveform_descriptors(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-feature-wave", sequences=8)
        matrix = load_feature_matrix(
            dataset_dir,
            split="train",
            config=MLFeatureConfig(
                modalities=("slow", "ultrasonic"),
                sequence_statistics=("mean",),
                waveform_frame_features=("max_abs", "peak_index"),
            ),
        )

        assert matrix.x.shape[0] == len(matrix.sequence_ids)
        assert matrix.x.shape[1] == 8 + 2
        assert "ultrasonic:ultrasonic_max_abs:mean" in matrix.feature_names
        assert "ultrasonic:ultrasonic_peak_index:mean" in matrix.feature_names
        assert np.isfinite(matrix.x).all()


class TestMLModels:
    def test_mean_regressor_predicts_training_target_mean(self):
        x = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        y = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]], dtype=np.float32)

        model = MeanRegressor().fit(x, y)
        pred = model.predict(np.array([[10.0], [20.0]], dtype=np.float32))

        np.testing.assert_allclose(pred, [[2.0, 4.0], [2.0, 4.0]])

    def test_ridge_regressor_fits_multioutput_linear_signal(self):
        x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y = np.concatenate([2.0 * x + 1.0, -1.0 * x + 5.0], axis=1)

        model = RidgeRegressor(alpha=0.0, standardize=False).fit(x, y)
        pred = model.predict(np.array([[4.0], [5.0]], dtype=np.float32))

        np.testing.assert_allclose(pred, [[9.0, 1.0], [11.0, 0.0]], atol=1e-5)

    def test_build_regressor_from_name_or_config(self):
        assert isinstance(build_regressor("mean"), MeanRegressor)
        model = build_regressor({"name": "ridge", "alpha": 0.25})
        assert isinstance(model, RidgeRegressor)
        assert model.alpha == 0.25


class TestMLMetrics:
    def test_regression_metrics_for_perfect_prediction(self):
        y_true = np.array([[0.1, 0.6, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1]], dtype=np.float32)
        metrics = regression_metrics(y_true, y_true)
        assert metrics == RegressionMetrics(mae=0.0, rmse=0.0, r2=1.0)

    def test_component_metrics_use_v4_component_fields(self):
        y_true = np.array([[0.1, 0.6, 0.2, 0.1], [0.2, 0.5, 0.2, 0.1]], dtype=np.float32)
        metrics = component_regression_metrics(y_true + 0.1, y_true)
        assert tuple(metrics) == COMPONENT_FIELDS
        for value in metrics.values():
            assert isinstance(value, RegressionMetrics)
            assert round(value.mae, 6) == 0.1


class TestMLTraining:
    def test_train_regressor_on_dataset_returns_split_evaluations(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-train", sequences=16)
        result = train_regressor_on_dataset(
            dataset_dir,
            model_config={"name": "ridge", "alpha": 1.0},
            feature_config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean", "last", "slope")),
            eval_splits=("train", "val"),
        )

        assert set(result.evaluations) == {"train", "val"}
        assert result.label_names == COMPONENT_FIELDS
        assert result.train_metrics == result.evaluations["train"].metrics
        for split_eval in result.evaluations.values():
            assert split_eval.predictions.shape == split_eval.targets.shape
            assert split_eval.predictions.shape[1] == 4
            assert np.isfinite(split_eval.predictions).all()
            assert isinstance(split_eval.metrics, RegressionMetrics)
