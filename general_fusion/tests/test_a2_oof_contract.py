from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from gf.dl.residual import apply_residual_learner, fit_residual_learner, residual_targets
from gf.ml.oof import build_grouped_fold_manifest, generate_grouped_oof_predictions


def test_grouped_oof_is_disjoint_and_has_row_provenance() -> None:
    group_ids = [f"binary-{index}" for index in range(5)] + [f"ternary-{index}" for index in range(5)]
    families = ["binary"] * 5 + ["ternary"] * 5
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    targets = np.column_stack((features[:, 0] * 0.5, features[:, 1] * -0.25))
    fold_manifest = build_grouped_fold_manifest(group_ids, families, n_splits=5, seed=20260827)
    result = generate_grouped_oof_predictions(
        features,
        targets,
        group_ids,
        families,
        estimator_factory=lambda seed: Ridge(alpha=1.0),
        n_splits=5,
        seed=20260827,
        model_config_hash="a" * 64,
        fold_manifest=fold_manifest,
    )
    assert result.predictions.shape == targets.shape
    assert np.isfinite(result.predictions).all()
    assert len(result.provenance) == len(group_ids)
    assert [row["row_index"] for row in result.provenance] == list(range(len(group_ids)))
    assert {row["mixture_id"] for row in result.provenance} == set(group_ids)
    assert {row["fold"] for row in result.provenance} == set(range(5))
    assert all(len(row["train_group_hash"]) == 64 for row in result.provenance)


def test_oof_fold_builder_rejects_family_without_fold_coverage() -> None:
    with pytest.raises(ValueError, match="fewer groups"):
        build_grouped_fold_manifest(
            ["binary-1", "binary-2", "ternary-1", "ternary-2", "ternary-3", "ternary-4", "ternary-5"],
            ["binary", "binary", "ternary", "ternary", "ternary", "ternary", "ternary"],
            n_splits=5,
        )


def test_residual_label_is_target_minus_oof_base_and_never_in_sample_base() -> None:
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    targets = np.column_stack((features[:, 0] * 0.5, features[:, 1] * -0.25))
    base = np.full_like(targets, 1.5)
    residual = residual_targets(targets, base)
    np.testing.assert_allclose(residual, targets - base)
    fit = fit_residual_learner(
        "ridge_residual",
        features,
        targets,
        base,
        ridge_alpha=1.0,
    )
    prediction = apply_residual_learner(base, fit, features)
    assert prediction.shape == targets.shape
    assert np.isfinite(prediction).all()
