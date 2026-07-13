"""Module C grouped bottleneck unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.ml.grouped_bottleneck import (
    EXPECTED_FEATURE_COUNT,
    EXPECTED_GROUP_COUNTS,
    EXPECTED_PARAMETER_COUNT,
    GROUP_ORDER,
    PRE_REGISTERED_PERMUTATION_SEED,
    GroupedBottleneckConfig,
    build_group_mapping,
    build_grouped_bottleneck_module,
    build_permuted_group_mapping,
    build_physical_group_mapping,
    channel_token_from_feature_name,
    count_module_parameters,
    expected_parameter_count,
    feature_names_digest,
)
from tv3.ml.grouped_ridge_residual_head import GroupedOofRidgeResidualMlpRegressor
from tv3.ml.mlp_head import MlpHeadConfig
from tv3.ml.rocket_training import MODULE_C_HEAD, DEFAULT_RIDGE_ALPHAS, _build_head
from tv3.pipeline.run_tv3_rocket_baseline import main


LABEL_NAMES = ("x_CO2", "x_O2", "x_N2")


def _synthetic_feature_names() -> tuple[str, ...]:
    """Build a 1008-name contract matching frozen RawDSP naming."""
    windows = (
        "full",
        "ph_baseline",
        "ph_exposure",
        "ph_steady",
        "ph_recovery",
        "early_0.25",
        "early_0.50",
        "early_0.75",
    )
    stats = ("mean", "std", "min", "max", "range", "first", "last", "delta", "slope")
    slow_channels = (
        "V_NDIR_CO2",
        "V_TCS",
        "T_C",
        "P_MPa",
        "H_RH",
        "L_m",
        "piston_position_m",
    )
    physics_channels = (
        "ultrasonic_tof_observed_raw_dsp_s",
        "ultrasonic_peak_index_raw_dsp",
        "ultrasonic_sound_speed_raw_dsp_m_per_s",
        "ultrasonic_corr_peak",
        "ultrasonic_snr_db",
        "ultrasonic_raw_dsp_quality",
        "ultrasonic_raw_dsp_accepted",
    )
    names: list[str] = []
    for window in windows:
        for stat in stats:
            for channel in slow_channels:
                names.append(f"{window}|slow:{channel}:{stat}")
        for stat in stats:
            for channel in physics_channels:
                names.append(f"{window}|physics:{channel}:{stat}")
    assert len(names) == EXPECTED_FEATURE_COUNT
    return tuple(names)


def _synthetic_regression(
    *,
    n_train: int = 80,
    n_val: int = 24,
    n_features: int = EXPECTED_FEATURE_COUNT,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(n_train, n_features))
    coef = rng.normal(size=(n_features, 3))
    y_train = x_train @ coef + 0.05 * rng.normal(size=(n_train, 3))
    x_val = rng.normal(size=(n_val, n_features))
    y_val = x_val @ coef + 0.05 * rng.normal(size=(n_val, 3))
    return x_train, y_train, x_val, y_val


def test_channel_token_parsing_and_physical_mapping_is_strict_partition():
    names = _synthetic_feature_names()
    assert channel_token_from_feature_name(names[0]) == "V_NDIR_CO2"
    mapping = build_physical_group_mapping(names)
    assert mapping.group_counts == EXPECTED_GROUP_COUNTS
    assert mapping.group_assignment == "physical"
    assert sum(mapping.group_counts.values()) == EXPECTED_FEATURE_COUNT
    concat = np.concatenate(mapping.group_indices)
    assert concat.size == EXPECTED_FEATURE_COUNT
    assert np.unique(concat).size == EXPECTED_FEATURE_COUNT
    assert mapping.feature_names_digest == feature_names_digest(names)


def test_unknown_or_duplicate_group_mapping_raises():
    names = list(_synthetic_feature_names())
    bad = names.copy()
    bad[0] = "full|slow:UNKNOWN_CHANNEL:mean"
    with pytest.raises(ValueError, match="unknown channel"):
        build_physical_group_mapping(bad)
    short = names[:-1]
    with pytest.raises(ValueError, match="expected 1008"):
        build_physical_group_mapping(short)


def test_permuted_mapping_preserves_sizes_and_uses_frozen_seed():
    names = _synthetic_feature_names()
    physical = build_physical_group_mapping(names)
    permuted = build_permuted_group_mapping(names, permutation_seed=PRE_REGISTERED_PERMUTATION_SEED)
    assert permuted.group_counts == physical.group_counts
    assert permuted.permutation_seed == PRE_REGISTERED_PERMUTATION_SEED
    assert permuted.permutation_digest
    # Not identical to physical for at least one group.
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(physical.group_indices, permuted.group_indices, strict=True)
    )
    with pytest.raises(ValueError, match="permutation_seed"):
        build_permuted_group_mapping(names, permutation_seed=1)


def test_parameter_count_is_pre_registered_28051():
    assert expected_parameter_count() == EXPECTED_PARAMETER_COUNT
    module = build_grouped_bottleneck_module(
        group_dims=tuple(EXPECTED_GROUP_COUNTS[g] for g in GROUP_ORDER),
        bottleneck_dim=16,
        hidden_dims=(64, 64),
        out_dim=3,
        activation_dropout=0.1,
        group_dropout=0.0,
        zero_init_output=True,
    )
    assert count_module_parameters(module) == EXPECTED_PARAMETER_COUNT


def test_zero_init_grouped_encoder_predicts_zeros():
    import torch

    module = build_grouped_bottleneck_module(
        group_dims=(8, 8, 16),
        bottleneck_dim=4,
        hidden_dims=(8, 8),
        out_dim=3,
        activation_dropout=0.0,
        group_dropout=0.0,
        zero_init_output=True,
    )
    module.eval()
    groups = [torch.randn(5, dim) for dim in (8, 8, 16)]
    with torch.no_grad():
        out = module(groups).numpy()
    assert np.allclose(out, 0.0, atol=1e-7)


def test_nonzero_group_dropout_is_rejected():
    with pytest.raises(NotImplementedError, match="group-level dropout is not implemented"):
        build_grouped_bottleneck_module(
            group_dims=(8, 8, 16),
            bottleneck_dim=4,
            hidden_dims=(8, 8),
            out_dim=3,
            activation_dropout=0.0,
            group_dropout=0.1,
            zero_init_output=True,
        )


def test_grouped_oof_residual_coverage_raw3_and_diagnostics():
    names = _synthetic_feature_names()
    x_train, y_train, x_val, y_val = _synthetic_regression(seed=11)
    model = GroupedOofRidgeResidualMlpRegressor(
        ridge_alphas=(0.1, 1.0, 10.0),
        mlp_config=MlpHeadConfig(
            hidden_dims=(64, 64),
            batch_size=16,
            max_epochs=2,
            patience=1,
            device="cpu",
            seed=11,
            dropout=0.1,
        ),
        oof_folds=4,
        oof_seed=20260711,
        grouped_config=GroupedBottleneckConfig(group_assignment="physical"),
    )
    model.fit(
        x_train,
        y_train,
        feature_names=names,
        x_val=x_val,
        y_val=y_val,
        label_names=LABEL_NAMES,
    )
    predictions = model.predict(x_val)
    assert predictions.shape == y_val.shape
    assert np.isfinite(predictions).all()
    oof = model.diagnostics["oof"]
    assert oof["coverage_complete"] is True
    assert oof["fold_seed"] == 20260711
    assert model.diagnostics["residual_mlp"]["zero_init_output"] is True
    assert model.diagnostics["residual_mlp"]["standardize_targets"] is True
    assert model.diagnostics["residual_mlp"]["early_stopping"] == {
        "monitor": "val_o2_r2",
        "uses_combined_ridge_prediction": True,
    }
    grouped = model.diagnostics["grouped_bottleneck"]
    assert grouped["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert grouped["group_assignment"] == "physical"
    assert grouped["group_counts"] == EXPECTED_GROUP_COUNTS


def test_permuted_and_physical_share_parameter_count_and_differ_in_assignment():
    names = _synthetic_feature_names()
    x_train, y_train, x_val, y_val = _synthetic_regression(seed=3)
    shared = {
        "ridge_alphas": (1.0, 10.0),
        "mlp_config": MlpHeadConfig(
            hidden_dims=(64, 64),
            batch_size=32,
            max_epochs=1,
            patience=1,
            device="cpu",
            seed=3,
        ),
        "oof_folds": 4,
        "oof_seed": 20260711,
    }
    physical = GroupedOofRidgeResidualMlpRegressor(
        **shared,
        grouped_config=GroupedBottleneckConfig(group_assignment="physical"),
    )
    permuted = GroupedOofRidgeResidualMlpRegressor(
        **shared,
        grouped_config=GroupedBottleneckConfig(
            group_assignment="permuted",
            permutation_seed=PRE_REGISTERED_PERMUTATION_SEED,
        ),
    )
    physical.fit(x_train, y_train, feature_names=names, x_val=x_val, y_val=y_val, label_names=LABEL_NAMES)
    permuted.fit(x_train, y_train, feature_names=names, x_val=x_val, y_val=y_val, label_names=LABEL_NAMES)
    assert (
        physical.diagnostics["grouped_bottleneck"]["parameter_count"]
        == permuted.diagnostics["grouped_bottleneck"]["parameter_count"]
        == EXPECTED_PARAMETER_COUNT
    )
    assert physical.diagnostics["grouped_bottleneck"]["group_assignment"] == "physical"
    assert permuted.diagnostics["grouped_bottleneck"]["group_assignment"] == "permuted"
    assert permuted.diagnostics["grouped_bottleneck"]["permutation_digest"]


def test_build_head_registers_module_c():
    model = _build_head(
        MODULE_C_HEAD,
        ridge_alphas=DEFAULT_RIDGE_ALPHAS,
        closed_form_alpha=1.0,
        mlp_config=MlpHeadConfig(device="cpu", hidden_dims=(64, 64)),
        oof_folds=5,
        oof_seed=20260711,
        grouped_bottleneck_config=GroupedBottleneckConfig(group_assignment="physical"),
    )
    assert isinstance(model, GroupedOofRidgeResidualMlpRegressor)


def test_module_c_configs_match_except_assignment_fields():
    project_root = Path(__file__).resolve().parents[1]
    c1 = json.loads(
        (project_root / "configs" / "tv3_module_c_grouped_bottleneck_physical.json").read_text(
            encoding="utf-8"
        )
    )
    c2 = json.loads(
        (project_root / "configs" / "tv3_module_c_grouped_bottleneck_permuted.json").read_text(
            encoding="utf-8"
        )
    )
    b7 = json.loads(
        (project_root / "configs" / "tv3_d2b_oof_ridge_residual_mlp.json").read_text(encoding="utf-8")
    )
    shared_keys = (
        "feature_set",
        "feature_builder",
        "include_slow",
        "slow_channels",
        "physics_arrays",
        "sequence_statistics",
        "phase_windows",
        "early_fractions",
        "eval_splits",
        "mlp_hidden_dims",
        "mlp_dropout",
        "mlp_weight_decay",
        "mlp_lr",
        "mlp_batch_size",
        "mlp_max_epochs",
        "mlp_patience",
        "mlp_loss_weights",
        "mlp_standardize_targets",
        "oof_folds",
        "oof_seed",
        "device",
    )
    for key in shared_keys:
        assert c1[key] == c2[key] == b7[key], key
    assert c1["head"] == c2["head"] == "grouped_oof_ridge_residual_mlp"
    assert c1["group_assignment"] == "physical"
    assert c2["group_assignment"] == "permuted"
    assert c1["permutation_seed"] is None
    assert c2["permutation_seed"] == PRE_REGISTERED_PERMUTATION_SEED
    assert c1["group_bottleneck_dim"] == c2["group_bottleneck_dim"] == 16
    assert c1["group_dropout"] == c2["group_dropout"] == 0.0


def test_run_tv3_rocket_baseline_rejects_unknown_module_c_keys(tmp_path: Path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(tmp_path / "unused"),
                "output_dir": str(tmp_path / "out"),
                "unknown_module_c_key": 1,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown rocket config keys"):
        main(["--config", str(config_path)])


def test_group_mapping_builder_respects_config_assignment():
    names = _synthetic_feature_names()
    physical = build_group_mapping(
        names, config=GroupedBottleneckConfig(group_assignment="physical")
    )
    permuted = build_group_mapping(
        names,
        config=GroupedBottleneckConfig(
            group_assignment="permuted",
            permutation_seed=PRE_REGISTERED_PERMUTATION_SEED,
        ),
    )
    assert physical.group_assignment == "physical"
    assert permuted.group_assignment == "permuted"
