from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dl.training.metrics import RegressionMetrics, component_regression_metrics, regression_metrics

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
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)
        self.history = TrainHistory()

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
    ) -> TrainHistory:
        early_stopping = early_stopping or EarlyStoppingConfig(enabled=False)
        amp = amp or AmpConfig(enabled=False)
        _validate_early_stopping(early_stopping, val_loader=val_loader)
        amp_dtype = _validate_amp(amp, device=self.device)
        scaler = torch.amp.GradScaler("cuda", enabled=amp.enabled)
        best_score: float | None = None
        unimproved_epochs = 0
        best_path = Path(best_checkpoint_path) if best_checkpoint_path is not None else None
        if best_path is not None:
            self.history.best_checkpoint_path = str(best_path)

        self.model.train()
        for epoch in range(1, epochs + 1):
            _cuda_synchronize(self.device)
            epoch_start = time.perf_counter()
            train_start = epoch_start
            total_loss = 0.0
            n_batches = 0
            n_samples = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=non_blocking)
                yb = yb.to(self.device, non_blocking=non_blocking)
                self.optimizer.zero_grad()
                if amp.enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        pred = self.model(xb)
                        loss = self.loss_fn(pred, yb)
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1
                n_samples += int(xb.shape[0])

            _cuda_synchronize(self.device)
            train_seconds = time.perf_counter() - train_start
            avg_train_loss = total_loss / max(n_batches, 1)

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

    @torch.no_grad()
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
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        n_batches = 0

        for xb, yb in data_loader:
            xb = xb.to(self.device, non_blocking=non_blocking)
            yb = yb.to(self.device, non_blocking=non_blocking)
            if amp.enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = self.model(xb)
                    loss = self.loss_fn(pred, yb)
            else:
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(yb.cpu())
            n_batches += 1

        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)
        metrics = regression_metrics(y_pred, y_true)
        comp_metrics = component_regression_metrics(y_pred, y_true)

        self.model.train()
        return {
            "loss": total_loss / max(n_batches, 1),
            "metrics": metrics,
            "component_metrics": comp_metrics,
        }

    @torch.no_grad()
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
        for xb, yb in data_loader:
            xb = xb.to(self.device, non_blocking=non_blocking)
            if amp.enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = self.model(xb)
            else:
                pred = self.model(xb)
            preds.append(pred.cpu())
            targets.append(yb.cpu())
        self.model.train()
        return torch.cat(preds), torch.cat(targets)

    def save_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
            },
            path,
        )

    def load_checkpoint(self, path: Path | str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint.get("history", TrainHistory())


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
