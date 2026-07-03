"""阶段 B：联合训练 ErrorNet + RCDW（冻结单模态网络）。"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from rcdw.models.rcdw import RCDW_MGDA
from rcdw.training.losses import StageBLoss
from rcdw.training.metrics import compute_metrics


def run_stage_b(
    model: RCDW_MGDA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: str = "cpu",
    save_dir: str = "runs/stage_b",
) -> RCDW_MGDA:
    """运行阶段 B 联合训练。"""
    sb = cfg["training"]["stage_b"]
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    # 冻结单模态网络
    if sb.get("freeze_single_modal", True):
        for name in ["ndir", "tcd", "usn"]:
            sub = getattr(model, name)
            for param in sub.parameters():
                param.requires_grad = False
        print("  [stage_b] single modal networks frozen")

    # 只优化 ErrorNet 参数（FeatureExtractor 和 RCDWFusion 无可学习参数）
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  [stage_b] trainable params: {sum(p.numel() for p in trainable)}")

    criterion = StageBLoss(
        lambda_e=sb["lambda_error"],
        lambda_s=sb["lambda_sum"],
    ).to(device)
    optimizer = torch.optim.AdamW(trainable, lr=sb["lr"], weight_decay=sb["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sb["epochs"])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, sb["epochs"] + 1):
        # --- 训练 ---
        model.train()
        # FeatureExtractor 和 RCDWFusion 无 dropout/BN，train mode 不影响
        train_loss_sum = 0.0
        train_count = 0
        for x_w, y in train_loader:
            x_w, y = x_w.to(device), y.to(device)
            out = model(x_w)
            y_modal = out["Y_modal"].detach()
            loss, _ = criterion(out["C"], y, out["E_pred"], y_modal)
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
        all_pred = []
        all_ref = []
        with torch.no_grad():
            for x_w, y in val_loader:
                x_w, y = x_w.to(device), y.to(device)
                out = model(x_w)
                y_modal = out["Y_modal"].detach()
                loss, _ = criterion(out["C"], y, out["E_pred"], y_modal)
                val_loss_sum += loss.item() * len(y)
                val_count += len(y)
                all_pred.append(out["C"].cpu())
                all_ref.append(y.cpu())
        val_loss = val_loss_sum / val_count

        if epoch % 20 == 0 or epoch == 1:
            pred_cat = torch.cat(all_pred)
            ref_cat = torch.cat(all_ref)
            m = compute_metrics(pred_cat, ref_cat)
            print(f"  [stage_b] epoch {epoch:3d}  "
                  f"train={train_loss_sum / train_count:.6f}  val={val_loss:.6f}  "
                  f"MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}")

        # --- 早停 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path / "rcdw.pt")
        else:
            patience_counter += 1
            if patience_counter >= sb["patience"]:
                print(f"  [stage_b] early stop at epoch {epoch}")
                break

    model.load_state_dict(
        torch.load(save_path / "rcdw.pt", weights_only=True)
    )
    return model
