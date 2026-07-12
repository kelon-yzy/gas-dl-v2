from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from tv3.ml.mlp_head import MlpHeadConfig, _ScaledMLPRegressor
from tv3.ml.ridge_head import ScaledRidgeCVRegressor


DEFAULT_OOF_FOLDS = 5
DEFAULT_OOF_SEED = 20260711


class OofRidgeResidualMlpRegressor:
    """Public prediction = full-train Ridge + residual MLP trained on OOF Ridge residuals."""

    def __init__(
        self,
        *,
        ridge_alphas: tuple[float, ...],
        mlp_config: MlpHeadConfig | None = None,
        oof_folds: int = DEFAULT_OOF_FOLDS,
        oof_seed: int = DEFAULT_OOF_SEED,
    ):
        if oof_folds < 2:
            raise ValueError(f"oof_folds must be >= 2, got {oof_folds}")
        if not ridge_alphas:
            raise ValueError("ridge_alphas must not be empty")
        self.ridge_alphas = tuple(float(alpha) for alpha in ridge_alphas)
        self.mlp_config = mlp_config or MlpHeadConfig()
        self.oof_folds = int(oof_folds)
        self.oof_seed = int(oof_seed)
        self.ridge_full: ScaledRidgeCVRegressor | None = None
        self.residual_mlp: _ScaledMLPRegressor | None = None
        self.diagnostics: dict[str, Any] = {}

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: tuple[str, ...] | None = None,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        label_names: tuple[str, ...] | None = None,
    ) -> OofRidgeResidualMlpRegressor:
        del feature_names  # accepted for rocket head API compatibility
        x_arr = _as_finite_2d(x, name="x")
        y_arr = _as_finite_2d(y, name="y", expected_cols=self.mlp_config.out_dim)
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"x/y row counts must match, got {x_arr.shape[0]} and {y_arr.shape[0]}")
        if x_val is None or y_val is None:
            raise ValueError("oof ridge residual mlp requires x_val and y_val for early stopping")
        if label_names is None:
            raise ValueError("oof ridge residual mlp requires label_names to monitor val O2 R2")
        x_val_arr = _as_finite_2d(x_val, name="x_val")
        y_val_arr = _as_finite_2d(y_val, name="y_val", expected_cols=self.mlp_config.out_dim)
        if x_val_arr.shape[0] != y_val_arr.shape[0]:
            raise ValueError(
                f"x_val/y_val row counts must match, got {x_val_arr.shape[0]} and {y_val_arr.shape[0]}"
            )
        if x_val_arr.shape[1] != x_arr.shape[1]:
            raise ValueError("x_val feature dimension must match x")
        if x_arr.shape[0] < self.oof_folds:
            raise ValueError(
                f"train rows ({x_arr.shape[0]}) must be >= oof_folds ({self.oof_folds})"
            )

        y_ridge_oof, oof_diagnostics = self._build_oof_ridge_predictions(x_arr, y_arr)
        r_train = y_arr - y_ridge_oof

        ridge_full = ScaledRidgeCVRegressor(alphas=self.ridge_alphas)
        ridge_full.fit(x_arr, y_arr)
        y_ridge_val = ridge_full.predict(x_val_arr)
        r_val = y_val_arr - y_ridge_val

        residual_config = replace(
            self.mlp_config,
            zero_init_output=True,
            standardize_targets=True,
        )
        _assert_zero_init_combined_equals_ridge(
            x=x_arr,
            ridge_predictions=ridge_full.predict(x_arr),
            mlp_config=residual_config,
        )
        residual_mlp = _ScaledMLPRegressor(config=residual_config)
        residual_mlp.fit(
            x_arr,
            r_train,
            x_val=x_val_arr,
            y_val=y_val_arr,
            label_names=label_names,
            early_stop_combine_base=y_ridge_val,
        )

        self.ridge_full = ridge_full
        self.residual_mlp = residual_mlp
        self.diagnostics = {
            "oof": oof_diagnostics,
            "ridge": {
                "full_selected_alpha": ridge_full.selected_alpha,
                "alphas": list(self.ridge_alphas),
            },
            "residual_mlp": {
                "model_config": asdict(residual_config),
                "parameter_count": residual_mlp.parameter_count,
                "best_epoch": residual_mlp.best_epoch,
                "best_val_o2_r2": residual_mlp.best_val_o2_r2,
                "standardize_targets": True,
                "zero_init_output": True,
                "early_stopping": {"monitor": "val_o2_r2"},
                "val_residual_rmse": float(np.sqrt(np.mean(np.square(r_val)))),
            },
            "leakage_audit": {
                "oof_used_for_residual_targets": True,
                "full_ridge_fit_on_train_only": True,
                "val_residual_from_full_ridge": True,
                "oof_coverage_complete": oof_diagnostics["coverage_complete"],
            },
        }
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.ridge_full is None or self.residual_mlp is None:
            raise RuntimeError("oof ridge residual mlp must be fit before predict")
        ridge_pred = self.ridge_full.predict(x)
        residual_pred = self.residual_mlp.predict(x)
        combined = ridge_pred + residual_pred
        if not np.isfinite(combined).all():
            raise ValueError("combined oof ridge residual predictions contain non-finite values")
        return combined.astype(np.float32, copy=False)

    def _build_oof_ridge_predictions(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        n_rows = x.shape[0]
        y_ridge_oof = np.zeros_like(y, dtype=np.float64)
        coverage = np.zeros(n_rows, dtype=np.int32)
        fold_payloads: list[dict[str, Any]] = []
        splitter = KFold(n_splits=self.oof_folds, shuffle=True, random_state=self.oof_seed)

        for fold_index, (fit_idx, hold_idx) in enumerate(splitter.split(x)):
            fit_idx = np.asarray(fit_idx, dtype=np.int64)
            hold_idx = np.asarray(hold_idx, dtype=np.int64)
            if np.intersect1d(fit_idx, hold_idx).size != 0:
                raise RuntimeError(f"OOF fold {fold_index} has overlapping fit/holdout indices")
            fold_ridge = ScaledRidgeCVRegressor(alphas=self.ridge_alphas)
            fold_ridge.fit(x[fit_idx], y[fit_idx])
            hold_pred = fold_ridge.predict(x[hold_idx])
            y_ridge_oof[hold_idx] = hold_pred
            coverage[hold_idx] += 1
            fold_payloads.append(
                {
                    "fold_index": fold_index,
                    "fit_row_count": int(fit_idx.size),
                    "holdout_row_count": int(hold_idx.size),
                    "selected_alpha": fold_ridge.selected_alpha,
                    "fit_holdout_disjoint": True,
                }
            )

        if not np.array_equal(coverage, np.ones(n_rows, dtype=np.int32)):
            raise RuntimeError(
                "OOF coverage incomplete: each train row must be written by exactly one holdout fold"
            )
        if y_ridge_oof.shape[0] != n_rows:
            raise RuntimeError("OOF prediction row count must equal train row count")

        return y_ridge_oof, {
            "fold_count": self.oof_folds,
            "fold_seed": self.oof_seed,
            "coverage_complete": True,
            "train_row_count": int(n_rows),
            "folds": fold_payloads,
        }


def _assert_zero_init_combined_equals_ridge(
    *,
    x: np.ndarray,
    ridge_predictions: np.ndarray,
    mlp_config: MlpHeadConfig,
) -> None:
    import torch

    from tv3.ml.mlp_head import _set_seed, build_raw3_mlp

    _set_seed(mlp_config.seed)
    device = _resolve_device(mlp_config.device)
    module = build_raw3_mlp(
        in_dim=x.shape[1],
        hidden_dims=mlp_config.hidden_dims,
        out_dim=mlp_config.out_dim,
        dropout=mlp_config.dropout,
        zero_init_output=True,
    ).to(device)
    module.eval()
    with torch.no_grad():
        residual = module(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    if not np.allclose(residual, 0.0, atol=1e-7):
        raise RuntimeError("zero-initialized residual MLP must predict all zeros before training")
    combined = np.asarray(ridge_predictions, dtype=np.float64) + residual
    if not np.allclose(combined, ridge_predictions, atol=1e-6, rtol=0.0):
        raise RuntimeError("combined prediction must equal ridge_full before residual training")


def _as_finite_2d(values: np.ndarray, *, name: str, expected_cols: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if expected_cols is not None and arr.shape[1] != expected_cols:
        raise ValueError(f"{name} must have {expected_cols} columns, got {arr.shape[1]}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
