"""Module C grouped OOF Ridge residual head.

Public prediction:
    Y_hat(X) = Ridge_full(X) + Residual_grouped(X)

Ridge / OOF residual targets / raw3 / target scaling / val O2 R2 early stopping /
zero-init output all reuse the B7 contract. Only residual encoding is grouped.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from tv3.ml.grouped_bottleneck import (
    DEFAULT_ACTIVATION_DROPOUT,
    DEFAULT_GROUP_BOTTLENECK_DIM,
    DEFAULT_GROUP_DROPOUT,
    EXPECTED_PARAMETER_COUNT,
    GroupedBottleneckConfig,
    build_group_mapping,
    build_grouped_bottleneck_module,
    count_module_parameters,
    expected_parameter_count,
    validate_group_mapping_against_config,
)
from tv3.ml.mlp_head import MlpHeadConfig, _as_finite_2d, _o2_r2, _resolve_device, _set_seed
from tv3.ml.ridge_head import ScaledRidgeCVRegressor
from tv3.ml.ridge_residual_head import (
    DEFAULT_OOF_FOLDS,
    DEFAULT_OOF_SEED,
    build_oof_ridge_predictions,
)


class GroupedOofRidgeResidualMlpRegressor:
    """B7 residual recipe with a grouped bottleneck residual encoder."""

    def __init__(
        self,
        *,
        ridge_alphas: tuple[float, ...],
        mlp_config: MlpHeadConfig | None = None,
        oof_folds: int = DEFAULT_OOF_FOLDS,
        oof_seed: int = DEFAULT_OOF_SEED,
        grouped_config: GroupedBottleneckConfig | None = None,
    ):
        if oof_folds < 2:
            raise ValueError(f"oof_folds must be >= 2, got {oof_folds}")
        if not ridge_alphas:
            raise ValueError("ridge_alphas must not be empty")
        self.ridge_alphas = tuple(float(alpha) for alpha in ridge_alphas)
        self.mlp_config = mlp_config or MlpHeadConfig()
        self.oof_folds = int(oof_folds)
        self.oof_seed = int(oof_seed)
        self.grouped_config = grouped_config or GroupedBottleneckConfig()
        self.ridge_full: ScaledRidgeCVRegressor | None = None
        self.residual_model: _ScaledGroupedBottleneckRegressor | None = None
        self.group_mapping = None
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
    ) -> GroupedOofRidgeResidualMlpRegressor:
        if feature_names is None:
            raise ValueError("grouped residual head requires feature_names for group mapping")
        x_arr = _as_finite_2d(x, name="x")
        y_arr = _as_finite_2d(y, name="y", expected_cols=self.mlp_config.out_dim)
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"x/y row counts must match, got {x_arr.shape[0]} and {y_arr.shape[0]}")
        if x_val is None or y_val is None:
            raise ValueError("grouped residual head requires x_val and y_val for early stopping")
        if label_names is None:
            raise ValueError("grouped residual head requires label_names to monitor val O2 R2")
        x_val_arr = _as_finite_2d(x_val, name="x_val")
        y_val_arr = _as_finite_2d(y_val, name="y_val", expected_cols=self.mlp_config.out_dim)
        if x_val_arr.shape[0] != y_val_arr.shape[0]:
            raise ValueError(
                f"x_val/y_val row counts must match, got {x_val_arr.shape[0]} and {y_val_arr.shape[0]}"
            )
        if x_val_arr.shape[1] != x_arr.shape[1]:
            raise ValueError("x_val feature dimension must match x")
        if x_arr.shape[1] != len(feature_names):
            raise ValueError(
                f"feature_names length {len(feature_names)} must match x columns {x_arr.shape[1]}"
            )
        if x_arr.shape[0] < self.oof_folds:
            raise ValueError(
                f"train rows ({x_arr.shape[0]}) must be >= oof_folds ({self.oof_folds})"
            )

        group_mapping = build_group_mapping(feature_names, config=self.grouped_config)
        validate_group_mapping_against_config(group_mapping, config=self.grouped_config)

        y_ridge_oof, oof_diagnostics = build_oof_ridge_predictions(
            x_arr,
            y_arr,
            ridge_alphas=self.ridge_alphas,
            oof_folds=self.oof_folds,
            oof_seed=self.oof_seed,
        )
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
        residual_model = _ScaledGroupedBottleneckRegressor(
            config=residual_config,
            group_mapping=group_mapping,
            grouped_config=self.grouped_config,
        )
        residual_model.assert_zero_init_equals_ridge(
            x=x_arr,
            ridge_predictions=ridge_full.predict(x_arr),
        )
        residual_model.fit(
            x_arr,
            r_train,
            x_val=x_val_arr,
            y_val=y_val_arr,
            label_names=label_names,
            early_stop_combine_base=y_ridge_val,
        )
        if residual_model.parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                f"grouped residual parameter_count={residual_model.parameter_count} "
                f"!= pre-registered {EXPECTED_PARAMETER_COUNT}"
            )

        self.ridge_full = ridge_full
        self.residual_model = residual_model
        self.group_mapping = group_mapping
        self.diagnostics = {
            "oof": oof_diagnostics,
            "ridge": {
                "full_selected_alpha": ridge_full.selected_alpha,
                "alphas": list(self.ridge_alphas),
            },
            "residual_mlp": {
                "model_config": asdict(residual_config),
                "parameter_count": residual_model.parameter_count,
                "best_epoch": residual_model.best_epoch,
                "best_val_o2_r2": residual_model.best_val_o2_r2,
                "standardize_targets": True,
                "zero_init_output": True,
                "early_stopping": {
                    "monitor": "val_o2_r2",
                    "uses_combined_ridge_prediction": True,
                },
                "val_residual_rmse": float(np.sqrt(np.mean(np.square(r_val)))),
            },
            "grouped_bottleneck": {
                **group_mapping.as_diagnostics(),
                "group_bottleneck_dim": self.grouped_config.group_bottleneck_dim,
                "group_dropout": self.grouped_config.group_dropout,
                "activation_dropout": self.grouped_config.activation_dropout,
                "parameter_count": residual_model.parameter_count,
                "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
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
        if self.ridge_full is None or self.residual_model is None:
            raise RuntimeError("grouped residual head must be fit before predict")
        ridge_pred = self.ridge_full.predict(x)
        residual_pred = self.residual_model.predict(x)
        combined = ridge_pred + residual_pred
        if not np.isfinite(combined).all():
            raise ValueError("combined grouped residual predictions contain non-finite values")
        return combined.astype(np.float32, copy=False)


class _ScaledGroupedBottleneckRegressor:
    """Per-group StandardScaler → grouped bottleneck residual → raw3 residual units."""

    def __init__(
        self,
        *,
        config: MlpHeadConfig,
        group_mapping,
        grouped_config: GroupedBottleneckConfig,
    ):
        self.config = config
        self.group_mapping = group_mapping
        self.grouped_config = grouped_config
        self.group_scalers: list[StandardScaler] = []
        self.target_scaler: StandardScaler | None = None
        self._module: Any = None
        self._device: Any = None
        self.best_epoch: int = 0
        self.best_val_o2_r2: float | None = None
        self.parameter_count: int = 0

    def assert_zero_init_equals_ridge(
        self,
        *,
        x: np.ndarray,
        ridge_predictions: np.ndarray,
    ) -> None:
        import torch

        _set_seed(self.config.seed)
        device = _resolve_device(self.config.device)
        module = build_grouped_bottleneck_module(
            group_dims=self.group_mapping.group_dims,
            bottleneck_dim=self.grouped_config.group_bottleneck_dim,
            hidden_dims=self.config.hidden_dims,
            out_dim=self.config.out_dim,
            activation_dropout=self.grouped_config.activation_dropout,
            group_dropout=self.grouped_config.group_dropout,
            zero_init_output=True,
        ).to(device)
        expected = expected_parameter_count(
            group_dims=self.group_mapping.group_dims,
            bottleneck_dim=self.grouped_config.group_bottleneck_dim,
            hidden_dims=self.config.hidden_dims,
            out_dim=self.config.out_dim,
        )
        actual = count_module_parameters(module)
        if actual != expected:
            raise RuntimeError(
                f"grouped bottleneck parameter_count {actual} != closed-form {expected}"
            )
        # Use raw unscaled groups only for the zero-output structural check.
        group_tensors = [
            torch.tensor(x[:, indices], dtype=torch.float32, device=device)
            for indices in self.group_mapping.group_indices
        ]
        module.eval()
        with torch.no_grad():
            residual = module(group_tensors).cpu().numpy()
        if not np.allclose(residual, 0.0, atol=1e-7):
            raise RuntimeError("zero-initialized grouped residual must predict all zeros")
        combined = np.asarray(ridge_predictions, dtype=np.float64) + residual
        if not np.allclose(combined, ridge_predictions, atol=1e-6, rtol=0.0):
            raise RuntimeError("combined prediction must equal ridge_full before residual training")

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_val: np.ndarray,
        y_val: np.ndarray,
        label_names: tuple[str, ...],
        early_stop_combine_base: np.ndarray,
    ) -> _ScaledGroupedBottleneckRegressor:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        x_arr = _as_finite_2d(x, name="x")
        y_arr = _as_finite_2d(y, name="y", expected_cols=self.config.out_dim)
        x_val_arr = _as_finite_2d(x_val, name="x_val")
        y_val_arr = _as_finite_2d(y_val, name="y_val", expected_cols=self.config.out_dim)
        combine_base = _as_finite_2d(
            early_stop_combine_base,
            name="early_stop_combine_base",
            expected_cols=self.config.out_dim,
        )
        if combine_base.shape[0] != y_val_arr.shape[0]:
            raise ValueError("early_stop_combine_base row count must match y_val")

        _set_seed(self.config.seed)
        self._device = _resolve_device(self.config.device)

        self.group_scalers = []
        train_groups: list[np.ndarray] = []
        val_groups: list[np.ndarray] = []
        for indices in self.group_mapping.group_indices:
            scaler = StandardScaler()
            train_block = scaler.fit_transform(x_arr[:, indices])
            val_block = scaler.transform(x_val_arr[:, indices])
            self.group_scalers.append(scaler)
            train_groups.append(train_block.astype(np.float32, copy=False))
            val_groups.append(val_block.astype(np.float32, copy=False))

        if self.config.standardize_targets:
            self.target_scaler = StandardScaler().fit(y_arr)
            y_train_for_loss = self.target_scaler.transform(y_arr)
        else:
            self.target_scaler = None
            y_train_for_loss = y_arr

        module = build_grouped_bottleneck_module(
            group_dims=self.group_mapping.group_dims,
            bottleneck_dim=self.grouped_config.group_bottleneck_dim,
            hidden_dims=self.config.hidden_dims,
            out_dim=self.config.out_dim,
            activation_dropout=self.grouped_config.activation_dropout,
            group_dropout=self.grouped_config.group_dropout,
            zero_init_output=True,
        ).to(self._device)
        self.parameter_count = count_module_parameters(module)
        if self.parameter_count != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                f"grouped residual parameter_count={self.parameter_count} "
                f"!= pre-registered {EXPECTED_PARAMETER_COUNT}"
            )

        loss_weights = torch.tensor(self.config.loss_weights, dtype=torch.float32, device=self._device)
        if loss_weights.numel() != self.config.out_dim:
            raise ValueError(
                f"loss_weights length {loss_weights.numel()} must match out_dim {self.config.out_dim}"
            )
        optimizer = torch.optim.AdamW(
            module.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Pack groups as a single concatenated tensor for DataLoader simplicity; split in loop.
        train_concat = np.concatenate(train_groups, axis=1)
        val_concat = np.concatenate(val_groups, axis=1)
        group_dims = self.group_mapping.group_dims
        x_tensor = torch.tensor(train_concat, dtype=torch.float32)
        y_tensor = torch.tensor(y_train_for_loss, dtype=torch.float32)
        batch_size = min(self.config.batch_size, max(1, x_tensor.shape[0]))
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        x_val_tensor = torch.tensor(val_concat, dtype=torch.float32, device=self._device)

        best_state: dict[str, Any] | None = None
        best_val_o2_r2 = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, self.config.max_epochs + 1):
            module.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                optimizer.zero_grad(set_to_none=True)
                predictions = module(_split_concat_groups(batch_x, group_dims))
                per_dim = (predictions - batch_y).square()
                loss = (per_dim * loss_weights).mean()
                loss.backward()
                optimizer.step()

            module.eval()
            with torch.no_grad():
                residual_predictions = self._decode_predictions(
                    module(_split_concat_groups(x_val_tensor, group_dims)).cpu().numpy()
                )
            val_predictions = combine_base + residual_predictions
            val_o2_r2 = _o2_r2(val_predictions, y_val_arr, label_names)
            if val_o2_r2 > best_val_o2_r2:
                best_val_o2_r2 = val_o2_r2
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in module.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

        if best_state is None:
            raise RuntimeError("grouped residual training did not produce a checkpoint")
        module.load_state_dict(best_state)
        module.eval()
        self._module = module
        self.best_epoch = best_epoch
        self.best_val_o2_r2 = float(best_val_o2_r2)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self._module is None or not self.group_scalers:
            raise RuntimeError("grouped residual regressor must be fit before predict")
        x_arr = _as_finite_2d(x, name="x")
        groups = []
        for scaler, indices in zip(self.group_scalers, self.group_mapping.group_indices, strict=True):
            groups.append(scaler.transform(x_arr[:, indices]).astype(np.float32, copy=False))
        concat = np.concatenate(groups, axis=1)
        x_tensor = torch.tensor(concat, dtype=torch.float32, device=self._device)
        self._module.eval()
        with torch.no_grad():
            predictions = self._module(
                _split_concat_groups(x_tensor, self.group_mapping.group_dims)
            ).cpu().numpy()
        predictions = self._decode_predictions(predictions)
        if not np.isfinite(predictions).all():
            raise ValueError("grouped residual predictions contain non-finite values")
        return predictions.astype(np.float32, copy=False)

    def _decode_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            return predictions
        return self.target_scaler.inverse_transform(predictions)


def _split_concat_groups(concat_x, group_dims: tuple[int, ...]):
    import torch

    pieces = []
    cursor = 0
    for dim in group_dims:
        pieces.append(concat_x[:, cursor : cursor + dim])
        cursor += dim
    if cursor != concat_x.shape[1]:
        raise RuntimeError(
            f"group dims sum {cursor} does not match concat feature dim {concat_x.shape[1]}"
        )
    return pieces
