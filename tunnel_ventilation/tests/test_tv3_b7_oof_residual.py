from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.ml.mlp_head import MlpHeadConfig, build_raw3_mlp
from tv3.ml.ridge_residual_head import OofRidgeResidualMlpRegressor
from tv3.ml.rocket_training import (
    B7_HEAD,
    DEFAULT_RIDGE_ALPHAS,
    _build_head,
    load_b6_multiseed_report,
)
from tv3.pipeline.run_tv3_rocket_baseline import main


LABEL_NAMES = ("x_CO2", "x_O2", "x_N2")


def _synthetic_regression(*, n_train: int = 80, n_val: int = 24, n_features: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(n_train, n_features))
    coef = rng.normal(size=(n_features, 3))
    y_train = x_train @ coef + 0.05 * rng.normal(size=(n_train, 3))
    x_val = rng.normal(size=(n_val, n_features))
    y_val = x_val @ coef + 0.05 * rng.normal(size=(n_val, 3))
    return x_train, y_train, x_val, y_val


def _write_b6_report(path: Path, *, verdict: str = "stable_pass") -> Path:
    payload = {
        "seeds": [42, 123, 456],
        "groups": {
            "b6": {
                "verdict": verdict,
                "pass_count": 3,
                "completed_seeds": [42, 123, 456],
                "o2_r2_stats": {
                    "val": {"mean": 0.5581, "std": 0.0096},
                    "test": {"mean": 0.5356, "std": 0.0170},
                    "extrapolation": {"mean": 0.4835, "std": 0.0036},
                },
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_zero_init_output_layer_predicts_all_zeros():
    import torch

    module = build_raw3_mlp(
        in_dim=8,
        hidden_dims=(4, 4),
        out_dim=3,
        dropout=0.0,
        zero_init_output=True,
    )
    module.eval()
    with torch.no_grad():
        predictions = module(torch.randn(16, 8)).numpy()
    assert np.allclose(predictions, 0.0, atol=1e-7)


def test_oof_ridge_residual_coverage_and_fold_disjoint():
    x_train, y_train, x_val, y_val = _synthetic_regression()
    model = OofRidgeResidualMlpRegressor(
        ridge_alphas=DEFAULT_RIDGE_ALPHAS,
        mlp_config=MlpHeadConfig(
            hidden_dims=(16, 16),
            batch_size=16,
            max_epochs=3,
            patience=2,
            device="cpu",
            seed=11,
        ),
        oof_folds=5,
        oof_seed=20260711,
    )
    model.fit(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        label_names=LABEL_NAMES,
    )

    oof = model.diagnostics["oof"]
    assert oof["fold_count"] == 5
    assert oof["fold_seed"] == 20260711
    assert oof["coverage_complete"] is True
    assert oof["train_row_count"] == x_train.shape[0]
    assert sum(fold["holdout_row_count"] for fold in oof["folds"]) == x_train.shape[0]
    assert all(fold["fit_holdout_disjoint"] for fold in oof["folds"])
    assert model.diagnostics["leakage_audit"]["oof_used_for_residual_targets"] is True
    assert model.diagnostics["ridge"]["full_selected_alpha"] > 0.0


def test_oof_ridge_residual_combined_predict_is_raw3_and_finite():
    x_train, y_train, x_val, y_val = _synthetic_regression(seed=3)
    model = OofRidgeResidualMlpRegressor(
        ridge_alphas=(0.1, 1.0, 10.0),
        mlp_config=MlpHeadConfig(
            hidden_dims=(8,),
            batch_size=16,
            max_epochs=2,
            patience=1,
            device="cpu",
            seed=3,
        ),
        oof_folds=4,
        oof_seed=20260711,
    )
    model.fit(
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        label_names=LABEL_NAMES,
    )
    predictions = model.predict(x_val)
    assert predictions.shape == y_val.shape
    assert np.isfinite(predictions).all()
    assert model.residual_mlp is not None
    assert model.residual_mlp.target_scaler is not None
    assert model.diagnostics["residual_mlp"]["standardize_targets"] is True
    assert model.diagnostics["residual_mlp"]["zero_init_output"] is True


def test_oof_seed_is_independent_of_mlp_training_seed():
    x_train, y_train, x_val, y_val = _synthetic_regression(seed=5)
    shared_kwargs = {
        "ridge_alphas": (0.1, 1.0, 10.0),
        "oof_folds": 4,
        "oof_seed": 20260711,
    }
    model_a = OofRidgeResidualMlpRegressor(
        **shared_kwargs,
        mlp_config=MlpHeadConfig(hidden_dims=(8,), batch_size=16, max_epochs=1, patience=1, device="cpu", seed=1),
    )
    model_b = OofRidgeResidualMlpRegressor(
        **shared_kwargs,
        mlp_config=MlpHeadConfig(hidden_dims=(8,), batch_size=16, max_epochs=1, patience=1, device="cpu", seed=2),
    )
    model_a.fit(x_train, y_train, x_val=x_val, y_val=y_val, label_names=LABEL_NAMES)
    model_b.fit(x_train, y_train, x_val=x_val, y_val=y_val, label_names=LABEL_NAMES)

    folds_a = [(fold["holdout_row_count"], fold["selected_alpha"]) for fold in model_a.diagnostics["oof"]["folds"]]
    folds_b = [(fold["holdout_row_count"], fold["selected_alpha"]) for fold in model_b.diagnostics["oof"]["folds"]]
    assert folds_a == folds_b
    assert model_a.diagnostics["ridge"]["full_selected_alpha"] == model_b.diagnostics["ridge"]["full_selected_alpha"]


def test_build_head_registers_oof_ridge_residual_mlp():
    model = _build_head(
        B7_HEAD,
        ridge_alphas=DEFAULT_RIDGE_ALPHAS,
        closed_form_alpha=1.0,
        mlp_config=MlpHeadConfig(device="cpu", hidden_dims=(8,)),
        oof_folds=5,
        oof_seed=20260711,
    )
    assert isinstance(model, OofRidgeResidualMlpRegressor)


def test_load_b6_multiseed_report_accepts_frozen_means(tmp_path: Path):
    report_path = _write_b6_report(tmp_path / "replication_report.json")
    payload = load_b6_multiseed_report(report_path)
    assert payload["verdict"] == "stable_pass"
    assert payload["o2_r2_means"]["test"] == pytest.approx(0.5356)
    assert payload["o2_r2_means"]["extrapolation"] == pytest.approx(0.4835)
    assert payload["report_sha256"]


def test_load_b6_multiseed_report_rejects_failed_verdict(tmp_path: Path):
    report_path = _write_b6_report(tmp_path / "bad_report.json", verdict="not_passed")
    with pytest.raises(ValueError, match="stable_pass"):
        load_b6_multiseed_report(report_path)


def test_b7_formal_config_matches_b6_feature_contract_with_residual_overrides():
    project_root = Path(__file__).resolve().parents[1]
    b6 = json.loads(
        (project_root / "configs" / "tv3_d2b_raw_dsp_mlp_target_scaled.json").read_text(encoding="utf-8")
    )
    b7 = json.loads(
        (project_root / "configs" / "tv3_d2b_oof_ridge_residual_mlp.json").read_text(encoding="utf-8")
    )
    feature_keys = (
        "feature_set",
        "feature_builder",
        "include_slow",
        "slow_channels",
        "physics_arrays",
        "sequence_statistics",
        "phase_windows",
        "early_fractions",
        "eval_splits",
        "raw_dsp_fidelity_metrics_path",
        "raw_dsp_reference_metrics_path",
    )
    for key in feature_keys:
        assert b7[key] == b6[key], key
    assert b7["dataset_dir"] == b6["dataset_dir"]
    assert b7["head"] == "oof_ridge_residual_mlp"
    assert b7["output_dir"] == "outputs/tv3_d2b/b7_oof_ridge_residual_mlp"
    assert b7["mlp_hidden_dims"] == [64, 64]
    assert b7["oof_folds"] == 5
    assert b7["oof_seed"] == 20260711
    assert b7["b6_multiseed_report_path"] == "outputs/tv3_r5t_b6_multiseed/replication_report.json"
    assert b7["mlp_standardize_targets"] is True
    assert b7["device"] == "cuda"


def test_run_tv3_rocket_baseline_rejects_unknown_b7_config_keys(tmp_path: Path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(tmp_path / "unused"),
                "output_dir": str(tmp_path / "out"),
                "unknown_b7_key": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown rocket config keys"):
        main(["--config", str(config_path)])
