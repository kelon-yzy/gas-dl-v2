from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common.composition import (
    N2_FIRST_ILR_BASIS,
    ALR_CH4_TRANSFORM,
    ILR_N2_FIRST_TRANSFORM,
    TargetTransformSpec,
    aitchison_distance,
    inverse_transform_composition_targets,
    resolve_target_transform_spec,
)
from common.metrics import CompositionalMetrics, conditional_component_metrics
from dl.training.metrics import RegressionMetrics, component_regression_metrics, regression_metrics
from sim.core.schema import COMPONENT_FIELDS as DEFAULT_COMPONENT_FIELDS

HYDROGEN_NG_SCHEME = "hydrogen_ng"
SYNGAS_SCHEME = "syngas"
_VALID_SCHEMES = (HYDROGEN_NG_SCHEME, SYNGAS_SCHEME)

OPTIMIZER_REGISTRY: dict[str, type[optim.Optimizer]] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}


@dataclass(frozen=True, slots=True)
class EarlyStoppingConfig:
    enabled: bool = False
    monitor: str = "val_loss"
    patience: int = 20
    min_delta: float = 0.0
    mode: str = "min"


@dataclass(frozen=True, slots=True)
class AmpConfig:
    enabled: bool = False
    dtype: str = "float16"


def build_optimizer(model: nn.Module, config: str | dict[str, object]) -> optim.Optimizer:
    if isinstance(config, str):
        name = config
        kwargs: dict[str, object] = {}
    else:
        opt_config = dict(config)
        name = str(opt_config.pop("name"))
        kwargs = opt_config

    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name!r}. Available: {sorted(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[name](model.parameters(), **kwargs)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    learning_rate: float
    val_loss: float | None = None
    epoch_seconds: float | None = None
    train_seconds: float | None = None
    val_seconds: float | None = None
    train_samples_per_second: float | None = None
    gpu_memory_allocated_mb: float | None = None
    gpu_memory_reserved_mb: float | None = None
    train_metrics: RegressionMetrics | None = None
    val_metrics: RegressionMetrics | None = None
    val_component_metrics: dict[str, RegressionMetrics] | None = None
    val_compositional_metrics: CompositionalMetrics | None = None
    val_sum_abs_error: float | None = None


@dataclass
class TrainHistory:
    epochs: list[EpochMetrics] = field(default_factory=list)
    stopped_early: bool = False
    stop_reason: str | None = None
    best_checkpoint_path: str | None = None

    @property
    def best_epoch(self) -> EpochMetrics | None:
        if not self.epochs:
            return None
        candidates = [e for e in self.epochs if e.val_loss is not None]
        if not candidates:
            candidates = self.epochs
        return min(candidates, key=lambda e: e.train_loss if e.val_loss is None else e.val_loss)


class Trainer:
    """DL 回归模型轻量训练器。

    提供 fit → evaluate → checkpoint 的最小闭环，配合
    V4BenchmarkDataset + build_model + build_loss 使用（见 KARPATHY_REVIEW 4.2）。
    不支持自动超参搜索或分布式训练。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device | str = "cpu",
        target_transform: str | dict[str, Any] | TargetTransformSpec | None = None,
        *,
        component_names: tuple[str, ...] = DEFAULT_COMPONENT_FIELDS,
        composition_scheme: str = HYDROGEN_NG_SCHEME,
    ):
        if composition_scheme not in _VALID_SCHEMES:
            raise ValueError(
                f"composition_scheme must be one of {_VALID_SCHEMES}, got {composition_scheme!r}"
            )
        if composition_scheme == SYNGAS_SCHEME and target_transform is not None:
            raise ValueError(
                "syngas composition_scheme is incompatible with target_transform "
                "(ILR/ALR assume sum=100 closure)"
            )
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.target_transform = (
            target_transform
            if isinstance(target_transform, TargetTransformSpec)
            else resolve_target_transform_spec(target_transform)
        )
        if self.target_transform is not None and isinstance(self.target_transform.epsilon, str):
            raise ValueError("Trainer target_transform epsilon must be resolved before training")
        self.component_names = tuple(component_names)
        self.composition_scheme = composition_scheme
        self.history = TrainHistory()

    @staticmethod
    def _unpack_batch(
        batch: tuple[object, torch.Tensor],
        *,
        device: torch.device,
        non_blocking: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        """Unpack a DataLoader batch, handling both tensor and dict inputs.

        Moves all tensors to *device* and returns (x, y, extra_kwargs) where
        *extra_kwargs* is a dict of additional inputs for model forward kwargs.
        """
        xb, yb = batch
        if isinstance(xb, dict):
            x = xb["x"].to(device, non_blocking=non_blocking)
            kwargs = {
                k: v.to(device, non_blocking=non_blocking)
                for k, v in xb.items()
                if k != "x"
            }
            return x, yb.to(device, non_blocking=non_blocking), kwargs
        return xb.to(device, non_blocking=non_blocking), yb.to(device, non_blocking=non_blocking), {}

    def fit(
        self,
        train_loader: DataLoader,
        *,
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
        epochs: int = 50,
        early_stopping: EarlyStoppingConfig | None = None,
        scheduler: optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau | None = None,
        best_checkpoint_path: Path | str | None = None,
        epoch_callback: Callable[[EpochMetrics, TrainHistory, int], None] | None = None,
        amp: AmpConfig | None = None,
        non_blocking: bool = False,
        grad_clip_norm: float = 0.0,
    ) -> TrainHistory:
        early_stopping = early_stopping or EarlyStoppingConfig(enabled=False)
        amp = amp or AmpConfig(enabled=False)
        _validate_early_stopping(early_stopping, val_loader=val_loader)
        amp_dtype = _validate_amp(amp, device=self.device)
        scaler = torch.amp.GradScaler("cuda", enabled=amp.enabled)
        best_score = _best_monitored_score(self.history, early_stopping) if early_stopping.enabled else None
        unimproved_epochs = (
            _unimproved_epochs_since_best(self.history, early_stopping)
            if early_stopping.enabled and best_score is not None
            else 0
        )
        best_path = Path(best_checkpoint_path) if best_checkpoint_path is not None else None
        if best_path is not None:
            self.history.best_checkpoint_path = str(best_path)

        self.model.train()
        for epoch in range(len(self.history.epochs) + 1, epochs + 1):
            _cuda_synchronize(self.device)
            epoch_start = time.perf_counter()
            train_start = epoch_start
            # 在设备上累积 loss，epoch 末再取标量，避免 per-batch .item() 强制 CUDA 同步。
            loss_accum = torch.zeros((), dtype=torch.float64, device=self.device)
            n_batches = 0
            n_samples = 0
            for batch in train_loader:
                xb, yb, model_kwargs = self._unpack_batch(batch, device=self.device, non_blocking=non_blocking)
                self.optimizer.zero_grad(set_to_none=True)
                if amp.enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        pred = self.model(xb, **model_kwargs)
                        loss = self.loss_fn(pred, self._loss_targets(yb))
                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0.0:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    pred = self.model(xb, **model_kwargs)
                    loss = self.loss_fn(pred, self._loss_targets(yb))
                    loss.backward()
                    if grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                    self.optimizer.step()
                loss_accum += loss.detach().double()
                n_batches += 1
                n_samples += int(xb.shape[0])

            _cuda_synchronize(self.device)
            train_seconds = time.perf_counter() - train_start
            avg_train_loss = float(loss_accum.item()) / max(n_batches, 1)

            entry = EpochMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                learning_rate=_current_learning_rate(self.optimizer),
                train_seconds=train_seconds,
                train_samples_per_second=n_samples / train_seconds if train_seconds > 0.0 else None,
            )
            if val_loader is not None:
                _cuda_synchronize(self.device)
                val_start = time.perf_counter()
                val_result = self.evaluate(val_loader, non_blocking=non_blocking, amp=amp)
                _cuda_synchronize(self.device)
                entry.val_seconds = time.perf_counter() - val_start
                entry.val_loss = val_result["loss"]
                entry.val_metrics = val_result["metrics"]
                entry.val_component_metrics = val_result["component_metrics"]
                entry.val_compositional_metrics = val_result["compositional_metrics"]
                entry.val_sum_abs_error = val_result["sum_abs_error"]

            self.history.epochs.append(entry)
            _step_scheduler(scheduler, entry)
            entry.learning_rate = _current_learning_rate(self.optimizer)
            if best_path is not None and _is_best_epoch(entry, self.history.best_epoch):
                self.save_checkpoint(best_path)
            if early_stopping.enabled:
                current_score = _monitored_value(entry, early_stopping.monitor)
                if _improved(current_score, best_score, early_stopping):
                    best_score = current_score
                    unimproved_epochs = 0
                else:
                    unimproved_epochs += 1
                    if unimproved_epochs >= early_stopping.patience:
                        self.history.stopped_early = True
                        self.history.stop_reason = (
                            f"{early_stopping.monitor} did not improve for "
                            f"{early_stopping.patience} epoch(s)"
                        )
            _cuda_synchronize(self.device)
            entry.epoch_seconds = time.perf_counter() - epoch_start
            memory = _gpu_memory_mb(self.device)
            entry.gpu_memory_allocated_mb = memory["allocated"]
            entry.gpu_memory_reserved_mb = memory["reserved"]
            if epoch_callback is not None:
                epoch_callback(entry, self.history, epochs)
            if self.history.stopped_early:
                break

        return self.history

    @torch.inference_mode()
    def evaluate(
        self,
        data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        *,
        non_blocking: bool = False,
        amp: AmpConfig | None = None,
    ) -> dict[str, Any]:
        amp = amp or AmpConfig(enabled=False)
        amp_dtype = _validate_amp(amp, device=self.device)
        self.model.eval()
        loss_accum = torch.zeros((), dtype=torch.float64, device=self.device)
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        n_batches = 0

        for batch in data_loader:
            xb, yb, model_kwargs = self._unpack_batch(batch, device=self.device, non_blocking=non_blocking)
            if amp.enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = self.model(xb, **model_kwargs)
                    loss = self.loss_fn(pred, self._loss_targets(yb))
            else:
                pred = self.model(xb, **model_kwargs)
                loss = self.loss_fn(pred, self._loss_targets(yb))
            loss_accum += loss.detach().double()
            all_preds.append(pred.cpu())
            all_targets.append(yb.cpu())
            n_batches += 1

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)
        y_pred_raw = self._metric_predictions(y_pred)
        metrics = regression_metrics(y_pred_raw, y_true)
        comp_metrics = component_regression_metrics(
            y_pred_raw, y_true, component_names=self.component_names
        )
        compositional_metrics = self._compositional_metrics(y_pred_raw, y_true)
        bin_components = self._conditional_bin_components()
        # 转 float32 后再 → numpy：autocast 下 AMP bfloat16 输出不支持直接 .numpy()。
        conditional_metrics = conditional_component_metrics(
            y_pred_raw.detach().float().cpu().numpy(),
            y_true.detach().float().cpu().numpy(),
            component_names=self.component_names,
            bin_components=bin_components,
        )
        # sum_abs_error 只在 sum=100% 闭包场景有意义；syngas 下 4 列 sum<100，
        # 强行计算会得到约 |sum - x_N2 - 100| 的无意义数值，置为 None 跳过。
        if self.composition_scheme == HYDROGEN_NG_SCHEME:
            sum_abs_error = float(torch.mean(torch.abs(y_pred_raw.sum(dim=1) - 100.0)).item())
        else:
            sum_abs_error = None

        self.model.train()
        return {
            "loss": float(loss_accum.item()) / max(n_batches, 1),
            "metrics": metrics,
            "component_metrics": comp_metrics,
            "compositional_metrics": compositional_metrics,
            "conditional_metrics": conditional_metrics,
            "sum_abs_error": sum_abs_error,
        }

    @torch.inference_mode()
    def predict(
        self,
        data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        *,
        non_blocking: bool = False,
        amp: AmpConfig | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        amp = amp or AmpConfig(enabled=False)
        amp_dtype = _validate_amp(amp, device=self.device)
        self.model.eval()
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for batch in data_loader:
            xb, yb, model_kwargs = self._unpack_batch(batch, device=self.device, non_blocking=non_blocking)
            if amp.enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = self.model(xb, **model_kwargs)
            else:
                pred = self.model(xb, **model_kwargs)
            preds.append(self._metric_predictions(pred).cpu())
            targets.append(yb.cpu())
        self.model.train()
        return torch.cat(preds), torch.cat(targets)

    def save_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # torch.compile 会把模型包成 OptimizedModule，state_dict 带 `_orig_mod.` 前缀；
        # 取原始模块保存，保证与未 compile 的已归档 checkpoint 双向兼容。
        model = getattr(self.model, "_orig_mod", self.model)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: Path | str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model = getattr(self.model, "_orig_mod", self.model)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", TrainHistory())

    def _conditional_bin_components(self) -> tuple[str, ...]:
        """Pick conditional-metric bin components per composition scheme.

        hydrogen_ng → ("x_N2", "x_CH4") to match the historical N2-improvement
        analysis tooling. syngas → ("x_CO", "x_CH4") since x_N2 is not in
        labels (it is background, computed as 100 - sum). Falls back to the
        last component plus x_CH4 when names are unrecognised.
        """
        if self.composition_scheme == SYNGAS_SCHEME:
            primary = "x_CO" if "x_CO" in self.component_names else self.component_names[-1]
        else:
            primary = "x_N2" if "x_N2" in self.component_names else self.component_names[-1]
        bins = [primary]
        if "x_CH4" in self.component_names and "x_CH4" != primary:
            bins.append("x_CH4")
        return tuple(bins)

    def _loss_targets(self, y_raw: torch.Tensor) -> torch.Tensor:
        if self.target_transform is None:
            return y_raw
        unit = _replace_zeros_multiplicative_tensor(
            _close_to_unit_tensor(y_raw),
            epsilon=self.target_transform.epsilon,
        )
        if self.target_transform.name == ALR_CH4_TRANSFORM:
            return torch.log(unit[:, (0, 2, 3)] / unit[:, 1:2])
        if self.target_transform.name == ILR_N2_FIRST_TRANSFORM:
            basis = torch.as_tensor(N2_FIRST_ILR_BASIS, dtype=unit.dtype, device=unit.device)
            return torch.log(unit) @ basis.T
        raise ValueError(f"Unknown target transform: {self.target_transform.name!r}")

    def _metric_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        if self.target_transform is None:
            return predictions
        # 转 float32 后再 → numpy：AMP bfloat16/float16 输出不支持直接 .numpy()。
        raw = inverse_transform_composition_targets(
            predictions.detach().float().cpu().numpy(),
            self.target_transform,
        )
        return torch.from_numpy(raw).to(predictions.device)

    def _compositional_metrics(
        self,
        y_pred_raw: torch.Tensor,
        y_true_raw: torch.Tensor,
    ) -> CompositionalMetrics | None:
        if self.target_transform is None:
            return None
        distances = aitchison_distance(
            y_pred_raw.detach().float().cpu().numpy(),
            y_true_raw.detach().float().cpu().numpy(),
            epsilon=self.target_transform.epsilon,
        )
        return CompositionalMetrics(
            aitchison_mean=float(np.mean(distances)),
            aitchison_rmse=float(np.sqrt(np.mean(distances * distances))),
        )


def _validate_early_stopping(
    config: EarlyStoppingConfig,
    *,
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None,
) -> None:
    if not config.enabled:
        return
    if config.monitor != "val_loss":
        raise ValueError("early stopping monitor must be 'val_loss'")
    if config.mode != "min":
        raise ValueError("early stopping mode must be 'min'")
    if config.patience < 1:
        raise ValueError("early stopping patience must be >= 1")
    if config.min_delta < 0.0:
        raise ValueError("early stopping min_delta must be >= 0")
    if val_loader is None:
        raise ValueError("early stopping requires a non-empty val split")


def _validate_amp(config: AmpConfig, *, device: torch.device) -> torch.dtype:
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    if config.dtype not in dtypes:
        raise ValueError(f"amp.dtype must be one of {sorted(dtypes)}, got {config.dtype!r}")
    if config.enabled and device.type != "cuda":
        raise ValueError("amp.enabled=true requires a CUDA device")
    return dtypes[config.dtype]


def _best_monitored_score(history: TrainHistory, config: EarlyStoppingConfig) -> float | None:
    best_score: float | None = None
    for entry in history.epochs:
        score = _monitored_value(entry, config.monitor)
        if _improved(score, best_score, config):
            best_score = score
    return best_score


def _unimproved_epochs_since_best(history: TrainHistory, config: EarlyStoppingConfig) -> int:
    best_index = -1
    best_score: float | None = None
    for index, entry in enumerate(history.epochs):
        score = _monitored_value(entry, config.monitor)
        if _improved(score, best_score, config):
            best_score = score
            best_index = index
    if best_index < 0:
        return 0
    return len(history.epochs) - best_index - 1


def _close_to_unit_tensor(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"target tensor must be shaped (N, C), got {tuple(values.shape)}")
    if torch.any(values < 0.0):
        raise ValueError("target tensor must not contain negative components")
    row_sums = values.sum(dim=1, keepdim=True)
    if torch.any(row_sums <= 0.0):
        raise ValueError("target composition rows must have positive sum")
    return values.float() / row_sums.float()


def _replace_zeros_multiplicative_tensor(unit_values: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    zero_mask = unit_values == 0.0
    zero_count = zero_mask.sum(dim=1, keepdim=True).to(unit_values.dtype)
    replacement_total = zero_count * float(epsilon)
    if torch.any(replacement_total >= 1.0):
        raise ValueError("zero replacement epsilon is too large for target rows")
    positive_sum = unit_values.masked_fill(zero_mask, 0.0).sum(dim=1, keepdim=True)
    if torch.any(positive_sum <= 0.0):
        raise ValueError("zero replacement requires at least one positive component per row")
    positive_scale = (1.0 - replacement_total) / positive_sum
    return torch.where(zero_mask, torch.full_like(unit_values, float(epsilon)), unit_values * positive_scale)


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _gpu_memory_mb(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"allocated": None, "reserved": None}
    scale = 1024.0 * 1024.0
    return {
        "allocated": torch.cuda.memory_allocated(device) / scale,
        "reserved": torch.cuda.memory_reserved(device) / scale,
    }


def _monitored_value(epoch: EpochMetrics, monitor: str) -> float:
    if monitor == "val_loss" and epoch.val_loss is not None:
        return float(epoch.val_loss)
    raise ValueError(f"Cannot monitor {monitor!r} for epoch without validation metrics")


def _improved(current: float, best: float | None, config: EarlyStoppingConfig) -> bool:
    if best is None:
        return True
    return current < best - config.min_delta


def _current_learning_rate(optimizer: optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _step_scheduler(
    scheduler: optim.lr_scheduler.LRScheduler | optim.lr_scheduler.ReduceLROnPlateau | None,
    epoch: EpochMetrics,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if epoch.val_loss is None:
            raise ValueError("ReduceLROnPlateau requires val_loss")
        scheduler.step(epoch.val_loss)
        return
    scheduler.step()


def _is_best_epoch(current: EpochMetrics, best: EpochMetrics | None) -> bool:
    if best is None:
        return True
    current_score = current.val_loss if current.val_loss is not None else current.train_loss
    best_score = best.val_loss if best.val_loss is not None else best.train_loss
    return current_score <= best_score
