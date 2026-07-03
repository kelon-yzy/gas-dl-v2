"""一键两阶段训练入口（v1.2 Phase 4：从 benchmark 数据集加载）。

用法:
    cd rcdw_mgda
    # 1) 先生成 benchmark (Phase 3)
    python -m scripts.generate_benchmark --config configs/smoke.yaml
    # 2) 再训练
    python -m scripts.train --config configs/smoke.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from rcdw.data.dataset import BenchmarkDataset
from rcdw.models.rcdw import RCDW_MGDA
from rcdw.training.stage_a import run_stage_a
from rcdw.training.stage_b import run_stage_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--stage", type=str, default="both", choices=["a", "b", "both"]
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- 可复现性（M4 修复）---
    seed = int(cfg["data"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- 数据 ---
    dc = cfg["data"]
    data_root = dc.get("dataset_root")
    if not data_root:
        raise ValueError(
            "configs/*.yaml 必须含 data.dataset_root 字段(指向已生成的 benchmark 目录),"
            " 例如 'data/rcdw-smoke'。"
        )
    window = int(dc.get("window", 8))
    modalities = tuple(dc.get("train_modalities", ("slow", "ultrasonic")))
    apply_input_scaler = dc.get("apply_input_scaler")

    train_ds = BenchmarkDataset(
        data_root,
        split="train",
        window=window,
        modalities=modalities,
        apply_input_scaler=apply_input_scaler,
    )
    val_ds = BenchmarkDataset(
        data_root,
        split="val",
        window=window,
        modalities=modalities,
        apply_input_scaler=apply_input_scaler,
    )
    sa_save_dir = cfg["training"]["stage_a"].get("save_dir", "runs/stage_a")
    sb_save_dir = cfg["training"]["stage_b"].get("save_dir", "runs/stage_b")
    print(f"[train] data_root={data_root}")
    print(f"[train] apply_input_scaler={apply_input_scaler}")
    print(f"[train] stage_a_save_dir={sa_save_dir}")
    print(f"[train] stage_b_save_dir={sb_save_dir}")
    print(f"[train] train windows={len(train_ds)}, val windows={len(val_ds)}")

    bs = cfg["training"]["stage_a"]["batch_size"]
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=bs)

    device = args.device
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"
        print(f"Auto-selected device: {device}")

    # --- Stage A ---
    if args.stage in ("a", "both"):
        print("\n" + "=" * 60)
        print("STAGE A: Single-modal pretraining")
        print("=" * 60)
        run_stage_a(train_loader, val_loader, cfg, device=device, save_dir=sa_save_dir)

    # --- Stage B ---
    if args.stage in ("b", "both"):
        print("\n" + "=" * 60)
        print("STAGE B: Joint training ErrorNet + RCDW")
        print("=" * 60)

        W_base = torch.tensor(cfg["model"]["W_base"], dtype=torch.float32)
        hidden = cfg["model"]["single_modal"]["hidden"]
        err_hidden = cfg["model"].get("error_net", {}).get("hidden", hidden)
        fusion_kwargs = cfg["model"].get("fusion", {}) or {}
        model = RCDW_MGDA(
            W_base,
            hidden=hidden,
            err_hidden=err_hidden,
            window=window,
            fusion_kwargs=fusion_kwargs,
        )

        # 加载 Stage A checkpoint（磁盘文件名 == 子模块名）
        for name in ["ndir", "tcd", "usn"]:
            ckpt_path = Path(sa_save_dir) / f"{name}.pt"
            if ckpt_path.exists():
                getattr(model, name).load_state_dict(
                    torch.load(ckpt_path, weights_only=True)
                )
                print(f"  Loaded {ckpt_path}")
            else:
                print(f"  WARNING: {ckpt_path} not found, using random init")

        bs_b = cfg["training"]["stage_b"].get("batch_size", bs)
        g_b = torch.Generator().manual_seed(seed)
        train_loader_b = DataLoader(
            train_ds, batch_size=bs_b, shuffle=True, generator=g_b
        )
        val_loader_b = DataLoader(val_ds, batch_size=bs_b)
        run_stage_b(
            model,
            train_loader_b,
            val_loader_b,
            cfg,
            device=device,
            save_dir=sb_save_dir,
        )

    print("\n=== Training complete ===")


if __name__ == "__main__":
    main()
