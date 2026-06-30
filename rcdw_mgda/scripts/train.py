"""一键两阶段训练入口。

用法: cd rcdw_mgda && python -m scripts.train --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from rcdw.data.synth import make_splits, WindowedDataset
from rcdw.training.stage_a import run_stage_a
from rcdw.training.stage_b import run_stage_b
from rcdw.models.rcdw import RCDW_MGDA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stage", type=str, default="both",
                        choices=["a", "b", "both"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- 数据 ---
    dc = cfg["data"]
    splits = make_splits(
        n_train=dc["n_train"], n_val=dc["n_val"], n_test=dc["n_test"],
        L=dc["window"], seed=dc["seed"],
    )
    bs = cfg["training"]["stage_a"]["batch_size"]
    train_loader = DataLoader(WindowedDataset(*splits["train"]), batch_size=bs, shuffle=True)
    val_loader = DataLoader(WindowedDataset(*splits["val"]), batch_size=bs)

    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
        print(f"Auto-selected device: {device}")

    # --- Stage A ---
    if args.stage in ("a", "both"):
        print("\n" + "=" * 60)
        print("STAGE A: Single-modal pretraining")
        print("=" * 60)
        trained_models = run_stage_a(train_loader, val_loader, cfg, device=device)

    # --- Stage B ---
    if args.stage in ("b", "both"):
        print("\n" + "=" * 60)
        print("STAGE B: Joint training ErrorNet + RCDW")
        print("=" * 60)

        W_base = torch.tensor(cfg["model"]["W_base"], dtype=torch.float32)
        hidden = cfg["model"]["single_modal"]["hidden"]
        model = RCDW_MGDA(W_base, hidden=hidden)

        # 加载 Stage A checkpoint
        for name, attr_name in [("ndir", "ndir"), ("tcd", "tcd"), ("us", "usn")]:
            ckpt_path = Path(f"runs/stage_a/{name}.pt")
            if ckpt_path.exists():
                getattr(model, attr_name).load_state_dict(
                    torch.load(ckpt_path, weights_only=True)
                )
                print(f"  Loaded {ckpt_path}")
            else:
                print(f"  WARNING: {ckpt_path} not found, using random init")

        model = run_stage_b(model, train_loader, val_loader, cfg, device=device)

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
