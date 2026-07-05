from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common.composition import TRAIN_MIN_POSITIVE_HALF_EPSILON
from common.windows import WindowConfig
from ml.cli import build_parser as build_ml_cli_parser, run as run_ml_cli
from ml import (
    DynamicStackingSVRRegressor,
    MLFeatureConfig,
    MeanRegressor,
    RegressionMetrics,
    RidgeRegressor,
    build_regressor,
    component_regression_metrics,
    load_feature_matrix,
    regression_metrics,
    run_baseline_protocol,
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


def _make_tv3_smoke_dataset(tmp_path: Path, slug: str = "tv3-ml-smoke", sequences: int = 16) -> Path:
    from sim.generation.tunnel_ventilation import (
        TunnelVentilationBenchmarkGenerationSpec,
        generate_tunnel_ventilation_benchmark_dataset,
    )

    generate_tunnel_ventilation_benchmark_dataset(
        tmp_path,
        TunnelVentilationBenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=20260704,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
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

    def test_load_feature_matrix_can_filter_phase_and_early_window(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-feature-window", sequences=8)
        full = load_feature_matrix(
            dataset_dir,
            split="train",
            config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean",)),
        )
        exposure = load_feature_matrix(
            dataset_dir,
            split="train",
            config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean",), phase_filter="exposure"),
        )
        early = load_feature_matrix(
            dataset_dir,
            split="train",
            config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean",), early_fraction=0.5),
        )

        assert full.x.shape == exposure.x.shape == early.x.shape
        assert not np.allclose(full.x, exposure.x)
        assert not np.allclose(full.x, early.x)

    def test_load_feature_matrix_can_concatenate_multiple_windows(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-feature-multiwindow", sequences=8)
        config = MLFeatureConfig(
            modalities=("slow",),
            sequence_statistics=("mean",),
            feature_windows=(
                None,
                WindowConfig(kind="phase", value="exposure"),
                WindowConfig(kind="phase", value="recovery"),
            ),
        )

        multi = load_feature_matrix(dataset_dir, split="train", config=config)
        full = load_feature_matrix(
            dataset_dir,
            split="train",
            config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean",)),
        )
        exposure = load_feature_matrix(
            dataset_dir,
            split="train",
            config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean",), phase_filter="exposure"),
        )
        recovery = load_feature_matrix(
            dataset_dir,
            split="train",
            config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean",), phase_filter="recovery"),
        )

        np.testing.assert_allclose(multi.x, np.concatenate([full.x, exposure.x, recovery.x], axis=1))
        np.testing.assert_allclose(multi.y, full.y)
        assert multi.sequence_ids == full.sequence_ids
        assert multi.label_names == full.label_names
        assert multi.feature_names[: len(full.feature_names)] == tuple(f"full|{name}" for name in full.feature_names)
        assert "ph_exposure|slow:T_C:mean" in multi.feature_names
        assert "ph_recovery|slow:T_C:mean" in multi.feature_names


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
        dynamic = build_regressor({"name": "dynamic_stacking_svr", "mc_samples": 2, "n_jobs": 2})
        assert isinstance(dynamic, DynamicStackingSVRRegressor)
        assert dynamic.mc_samples == 2
        assert dynamic.n_jobs == 2

    def test_dynamic_stacking_svr_uses_modality_views_and_weights(self):
        x = np.array(
            [
                [0.0, 0.1, 0.9, 20.0],
                [0.2, 0.2, 0.8, 21.0],
                [0.4, 0.3, 0.7, 22.0],
                [0.6, 0.4, 0.6, 23.0],
                [0.8, 0.5, 0.5, 24.0],
                [1.0, 0.6, 0.4, 25.0],
            ],
            dtype=np.float32,
        )
        y = np.stack(
            [
                0.5 * x[:, 0] + x[:, 1],
                x[:, 2],
            ],
            axis=1,
        ).astype(np.float32)
        feature_names = (
            "ultrasonic:ultrasonic_max_abs:mean",
            "slow:V_NDIR_CH4:mean",
            "slow:V_TCS:mean",
            "slow:T_C:mean",
        )

        model = DynamicStackingSVRRegressor(mc_samples=2, random_seed=7, n_jobs=2).fit(x, y, feature_names=feature_names)
        predictions, weights = model.predict_with_diagnostics(x)

        assert predictions.shape == y.shape
        assert weights.shape == (x.shape[0], 3)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(x.shape[0]), atol=1e-6)
        assert np.isfinite(predictions).all()

    def test_dynamic_stacking_svr_rejects_invalid_n_jobs(self):
        x = np.ones((4, 4), dtype=np.float32)
        y = np.ones((4, 2), dtype=np.float32)
        feature_names = (
            "ultrasonic:ultrasonic_max_abs:mean",
            "slow:V_NDIR_CH4:mean",
            "slow:V_TCS:mean",
            "slow:T_C:mean",
        )
        model = DynamicStackingSVRRegressor(mc_samples=2, n_jobs=0)

        try:
            model.fit(x, y, feature_names=feature_names)
        except ValueError as exc:
            assert "n_jobs" in str(exc)
        else:
            raise AssertionError("expected n_jobs contract error")

    def test_dynamic_stacking_svr_requires_feature_names(self):
        x = np.ones((4, 3), dtype=np.float32)
        y = np.ones((4, 1), dtype=np.float32)
        model = DynamicStackingSVRRegressor()

        try:
            model.fit(x, y)
        except ValueError as exc:
            assert "feature_names" in str(exc)
        else:
            raise AssertionError("expected feature_names contract error")


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
            assert split_eval.sum_abs_error >= 0.0

    def test_train_regressor_on_tunnel_ventilation_uses_o2_co2_bins(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-ml-train", sequences=16)
        result = train_regressor_on_dataset(
            dataset_dir,
            model_config={"name": "ridge", "alpha": 1.0},
            feature_config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean", "last")),
            eval_splits=("train", "val"),
        )

        assert result.label_names == ("x_CO2", "x_O2", "x_N2")
        for split_eval in result.evaluations.values():
            assert set(split_eval.conditional_metrics) == {"o2_bins", "co2_bins"}
            assert split_eval.sum_abs_error is not None

    def test_train_regressor_rejects_target_transform_on_tunnel_ventilation(self, tmp_path: Path):
        dataset_dir = _make_tv3_smoke_dataset(tmp_path, slug="tv3-ml-transform-reject", sequences=8)

        try:
            train_regressor_on_dataset(
                dataset_dir,
                model_config={"name": "ridge", "alpha": 1.0},
                feature_config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean",)),
                eval_splits=("train",),
                target_transform="ilr_n2_first",
            )
        except ValueError as exc:
            assert "tunnel_ventilation" in str(exc)
            assert "target_transform" in str(exc)
        else:
            raise AssertionError("tv3 ML target_transform should be rejected")

    def test_train_regressor_can_use_alr_ch4_target_transform(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-train-alr", sequences=16)
        result = train_regressor_on_dataset(
            dataset_dir,
            model_config={"name": "ridge", "alpha": 1.0},
            feature_config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean", "last")),
            eval_splits=("train", "val"),
            target_transform={"name": "alr_ch4", "epsilon": TRAIN_MIN_POSITIVE_HALF_EPSILON},
        )

        assert result.target_transform is not None
        assert result.target_transform.name == "alr_ch4"
        assert isinstance(result.target_transform.epsilon, float)
        assert result.target_transform_audits is not None
        assert set(result.target_transform_audits) == {"train", "val"}
        assert result.target_transform_audits["train"].epsilon == result.target_transform.epsilon
        for split_eval in result.evaluations.values():
            assert split_eval.predictions.shape == split_eval.targets.shape
            assert split_eval.predictions.shape[1] == 4
            assert split_eval.compositional_metrics is not None
            assert split_eval.compositional_metrics.aitchison_mean >= 0.0
            assert set(split_eval.conditional_metrics) == {"n2_bins", "ch4_bins"}
            np.testing.assert_allclose(split_eval.predictions.sum(axis=1), np.full(split_eval.predictions.shape[0], 100.0), atol=1e-4)

    def test_run_baseline_protocol_passes_target_transform_to_windows(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-protocol-ilr", sequences=16)
        result = run_baseline_protocol(
            dataset_dir,
            model_config={"name": "ridge", "alpha": 1.0},
            feature_config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean",)),
            phases=("baseline",),
            early_fractions=(0.5,),
            eval_splits=("train", "val"),
            target_transform="ilr_n2_first",
        )

        assert result.full.target_transform is not None
        assert result.full.target_transform.name == "ilr_n2_first"
        assert result.per_phase["baseline"].target_transform.name == "ilr_n2_first"
        assert result.early[0.5].target_transform.name == "ilr_n2_first"

    def test_run_baseline_protocol_returns_phase_and_early_results(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-protocol", sequences=16)
        result = run_baseline_protocol(
            dataset_dir,
            model_config="mean",
            feature_config=MLFeatureConfig(modalities=("slow",), sequence_statistics=("mean",)),
            phases=("baseline", "exposure"),
            early_fractions=(0.5, 1.0),
            eval_splits=("train", "val"),
        )

        assert set(result.per_phase) == {"baseline", "exposure"}
        assert set(result.early) == {0.5, 1.0}
        assert set(result.full.evaluations) == {"train", "val"}

    def test_ml_cli_protocol_outputs_json(self, tmp_path: Path, capsys):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-cli-protocol-json", sequences=16)
        parser = build_ml_cli_parser()
        args = parser.parse_args(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--model",
                "mean",
                "--protocol",
                "--phases",
                "baseline,exposure",
                "--early-fractions",
                "0.5,1.0",
                "--json",
            ]
        )

        run_ml_cli(args)

        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == {"full", "per_phase", "early"}
        assert set(payload["per_phase"]) == {"baseline", "exposure"}
        assert set(payload["early"]) == {"0.5", "1.0"}

    def test_ml_cli_protocol_writes_markdown_report(self, tmp_path: Path, capsys):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-cli-protocol-report", sequences=16)
        report_path = tmp_path / "reports" / "baseline_protocol.md"
        parser = build_ml_cli_parser()
        args = parser.parse_args(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--model",
                "mean",
                "--protocol",
                "--phases",
                "baseline",
                "--early-fractions",
                "0.5",
                "--report-path",
                str(report_path),
            ]
        )

        run_ml_cli(args)

        assert "wrote protocol report" in capsys.readouterr().out
        report = report_path.read_text(encoding="utf-8")
        assert "# Baseline Evaluation Protocol" in report
        assert "## Per Phase" in report
        assert "### baseline" in report
        assert "## Early Windows" in report

    def test_ml_cli_reads_json_config(self, tmp_path: Path, capsys):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-cli-config", sequences=16)
        config_path = tmp_path / "ml_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_dir": str(dataset_dir),
                    "model": "mean",
                    "modalities": ["slow"],
                    "sequence_statistics": ["mean"],
                    "protocol": True,
                    "phases": ["baseline"],
                    "early_fractions": [0.5],
                    "json": True,
                }
            ),
            encoding="utf-8",
        )
        parser = build_ml_cli_parser()
        args = parser.parse_args(["--config", str(config_path)])

        run_ml_cli(args)

        payload = json.loads(capsys.readouterr().out)
        assert set(payload["per_phase"]) == {"baseline"}
        assert set(payload["early"]) == {"0.5"}

    def test_ml_cli_json_config_preserves_target_transform_metadata(self, tmp_path: Path, capsys):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="ml-cli-config-alr", sequences=16)
        config_path = tmp_path / "ml_config_alr.json"
        config_path.write_text(
            json.dumps(
                {
                    "dataset_dir": str(dataset_dir),
                    "model": "ridge",
                    "modalities": ["slow"],
                    "sequence_statistics": ["mean", "last"],
                    "target_transform": {"name": "alr_ch4", "epsilon": TRAIN_MIN_POSITIVE_HALF_EPSILON},
                    "json": True,
                }
            ),
            encoding="utf-8",
        )
        parser = build_ml_cli_parser()
        args = parser.parse_args(["--config", str(config_path)])

        run_ml_cli(args)

        payload = json.loads(capsys.readouterr().out)
        assert payload["target_transform"]["name"] == "alr_ch4"
        assert isinstance(payload["target_transform"]["epsilon"], float)
        assert set(payload["target_transform_audits"]) == {"train", "val", "test", "extrapolation"}
        assert payload["target_transform_audits"]["train"]["epsilon"] == payload["target_transform"]["epsilon"]
        assert payload["evaluations"]["val"]["compositional_metrics"]["aitchison_mean"] >= 0.0
        assert set(payload["evaluations"]["val"]["conditional_metrics"]) == {"n2_bins", "ch4_bins"}
