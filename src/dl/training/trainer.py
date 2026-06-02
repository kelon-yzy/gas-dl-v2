from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dl.training.metrics import RegressionMetrics, component_regression_metrics, regression_metrics

OPTIMIZER_REGISTRY: dict[str, type[optim.Optimizer]] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": optim.SGD,
}


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
    val_loss: float | None = None
    train_metrics: RegressionMetrics | None = None
    val_metrics: RegressionMetrics | None = None
    val_component_metrics: dict[str, RegressionMetrics] | None = None


@dataclass
class TrainHistory:
    epochs: list[EpochMetrics] = field(default_factory=list)

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
    ) -> TrainHistory:
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_loss / max(n_batches, 1)

            entry = EpochMetrics(epoch=epoch, train_loss=avg_train_loss)
            if val_loader is not None:
                val_result = self.evaluate(val_loader)
                entry.val_loss = val_result["loss"]
                entry.val_metrics = val_result["metrics"]
                entry.val_component_metrics = val_result["component_metrics"]

            self.history.epochs.append(entry)

        return self.history

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        n_batches = 0

        for xb, yb in data_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            pred = self.model(xb)
            total_loss += self.loss_fn(pred, yb).item()
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
    def predict(self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        preds: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for xb, yb in data_loader:
            xb = xb.to(self.device)
            preds.append(self.model(xb).cpu())
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
