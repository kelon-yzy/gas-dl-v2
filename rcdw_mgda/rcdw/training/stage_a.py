"""阶段 A：单模态独立预训练。

依次训练 NDIRNet / TCDNet / USNet，各自保存 checkpoint。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from rcdw.models.single_modal import NDIRNet, TCDNet, USNet, extract_modal_input
from rcdw.training.losses import WeightedMSE
from rcdw.training.metrics import compute_metrics


def train_single_modal(
    model: nn.Module,
    modality: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    loss_weights: list[float],
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    save_path: Path | str = "runs/stage_a",
    device: str = "cpu",
) -> nn.Module:
    """训练单个模态网络。"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    criterion = WeightedMSE(loss_weights).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- 训练 ---
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x_w, y in train_loader:
            x_w, y = x_w.to(device), y.to(device)
            x_last = x_w[:, -1, :]  # (B, 12)
            inp = extract_modal_input(x_last, modality)  # (B, 4)
            pred = model(inp)  # (B, 3)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(y)
            train_count += len(y)
        scheduler.step()

        # --- 验证 ---
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for x_w, y in val_loader:
                x_w, y = x_w.to(device), y.to(device)
                x_last = x_w[:, -1, :]
                inp = extract_modal_input(x_last, modality)
                pred = model(inp)
                loss = criterion(pred, y)
                val_loss_sum += loss.item() * len(y)
                val_count += len(y)
        val_loss = val_loss_sum / val_count

        if epoch % 20 == 0 or epoch == 1:
            print(f"  [{modality}] epoch {epoch:3d}  "
                  f"train={train_loss_sum / train_count:.6f}  val={val_loss:.6f}")

        # --- 早停 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path / f"{modality}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{modality}] early stop at epoch {epoch}")
                break

    # 加载最佳权重
    model.load_state_dict(torch.load(save_path / f"{modality}.pt", weights_only=True))
    return model


def run_stage_a(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: str = "cpu",
    save_dir: str = "runs/stage_a",
) -> dict[str, nn.Module]:
    """运行完整的阶段 A：训练 3 个单模态网络。"""
    sa = cfg["training"]["stage_a"]
    hidden = cfg["model"]["single_modal"]["hidden"]

    models = {
        "ndir": NDIRNet(in_dim=4, hidden=hidden),
        "tcd": TCDNet(in_dim=4, hidden=hidden),
        "usn": USNet(in_dim=4, hidden=hidden),
    }

    loss_weights_map = {
        "ndir": sa["ndir_loss_weights"],   # [0.1, 1.0, 0.1]
        "tcd": [1.0, 1.0, 1.0],
        "usn": [1.0, 1.0, 1.0],
    }

    trained = {}
    for name, model in models.items():
        print(f"\n=== Stage A: training {name.upper()} ===")
        trained[name] = train_single_modal(
            model,
            modality=name,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_weights=loss_weights_map[name],
            epochs=sa["epochs"],
            lr=sa["lr"],
            weight_decay=sa["weight_decay"],
            patience=sa["patience"],
            save_path=save_dir,
            device=device,
        )

    return trained
