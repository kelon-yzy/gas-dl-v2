from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from tv3.ml.metrics import component_regression_metrics


@dataclass(frozen=True, slots=True)
class MlpHeadConfig:
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.1
    weight_decay: float = 1e-4
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 20
    loss_weights: tuple[float, ...] = (1.0, 2.0, 1.0)
    standardize_targets: bool = False
    device: str = "auto"
    seed: int = 20260704
    out_dim: int = 3


class _ScaledMLPRegressor:
    """StandardScaler(x) → shared-trunk MLP → raw3 percentage outputs."""

    def __init__(self, *, config: MlpHeadConfig | None = None):
        self.config = config or MlpHeadConfig()
        self.scaler = StandardScaler()
        self.target_scaler: StandardScaler | None = None
        self._module: Any = None
        self._device: Any = None
        self.best_epoch: int = 0
        self.best_val_o2_r2: float | None = None
        self.parameter_count: int = 0

    @property
    def hidden_dims(self) -> tuple[int, ...]:
        return self.config.hidden_dims

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: tuple[str, ...] | None = None,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        label_names: tuple[str, ...] | None = None,
    ) -> _ScaledMLPRegressor:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        x_arr = _as_finite_2d(x, name="x")
        y_arr = _as_finite_2d(y, name="y", expected_cols=self.config.out_dim)
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"x/y row counts must match, got {x_arr.shape[0]} and {y_arr.shape[0]}")
        if x_val is None or y_val is None:
            raise ValueError("mlp head requires x_val and y_val for early stopping")
        x_val_arr = _as_finite_2d(x_val, name="x_val")
        y_val_arr = _as_finite_2d(y_val, name="y_val", expected_cols=self.config.out_dim)
        if x_val_arr.shape[0] != y_val_arr.shape[0]:
            raise ValueError(
                f"x_val/y_val row counts must match, got {x_val_arr.shape[0]} and {y_val_arr.shape[0]}"
            )
        if x_val_arr.shape[1] != x_arr.shape[1]:
            raise ValueError("x_val feature dimension must match x")
        if label_names is None:
            raise ValueError("mlp head requires label_names to monitor val O2 R2")

        _set_seed(self.config.seed)
        self._device = _resolve_device(self.config.device)

        x_scaled = self.scaler.fit_transform(x_arr)
        x_val_scaled = self.scaler.transform(x_val_arr)
        if self.config.standardize_targets:
            self.target_scaler = StandardScaler().fit(y_arr)
            y_train_for_loss = self.target_scaler.transform(y_arr)
        else:
            self.target_scaler = None
            y_train_for_loss = y_arr

        module = _build_raw3_mlp(
            in_dim=x_scaled.shape[1],
            hidden_dims=self.config.hidden_dims,
            out_dim=self.config.out_dim,
            dropout=self.config.dropout,
        ).to(self._device)
        self.parameter_count = sum(parameter.numel() for parameter in module.parameters())

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

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
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

        x_val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32, device=self._device)

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
                predictions = module(batch_x)
                per_dim = (predictions - batch_y).square()
                loss = (per_dim * loss_weights).mean()
                loss.backward()
                optimizer.step()

            module.eval()
            with torch.no_grad():
                val_predictions = self._decode_predictions(module(x_val_tensor).cpu().numpy())
            val_o2_r2 = _o2_r2(val_predictions, y_val_arr, label_names)
            if val_o2_r2 > best_val_o2_r2:
                best_val_o2_r2 = val_o2_r2
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.patience:
                    break

        if best_state is None:
            raise RuntimeError("mlp training did not produce a checkpoint")
        module.load_state_dict(best_state)
        module.eval()
        self._module = module
        self.best_epoch = best_epoch
        self.best_val_o2_r2 = float(best_val_o2_r2)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self._module is None:
            raise RuntimeError("mlp regressor must be fit before predict")
        x_arr = _as_finite_2d(x, name="x")
        x_scaled = self.scaler.transform(x_arr)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self._device)
        self._module.eval()
        with torch.no_grad():
            predictions = self._module(x_tensor).cpu().numpy()
        predictions = self._decode_predictions(predictions)
        if not np.isfinite(predictions).all():
            raise ValueError("mlp predictions contain non-finite values")
        return predictions.astype(np.float32, copy=False)

    def _decode_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            return predictions
        # This conditions optimization only; public predictions remain raw3 percentages.
        return self.target_scaler.inverse_transform(predictions)


def _build_raw3_mlp(
    *,
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
    dropout: float,
):
    from torch import nn

    if in_dim < 1:
        raise ValueError("in_dim must be >= 1")
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer")
    if out_dim < 1:
        raise ValueError("out_dim must be >= 1")
    if dropout < 0.0:
        raise ValueError("dropout must be >= 0")

    layers: list[nn.Module] = []
    current = in_dim
    for hidden in hidden_dims:
        hidden_int = int(hidden)
        if hidden_int < 1:
            raise ValueError("hidden_dims entries must be >= 1")
        layers.append(nn.Linear(current, hidden_int))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current = hidden_int
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


def _resolve_device(device: str):
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _o2_r2(predictions: np.ndarray, targets: np.ndarray, label_names: tuple[str, ...]) -> float:
    if "x_O2" not in label_names:
        raise ValueError("label_names must include x_O2 for mlp early stopping")
    metrics = component_regression_metrics(predictions, targets, label_names)
    return float(metrics["x_O2"].r2)
