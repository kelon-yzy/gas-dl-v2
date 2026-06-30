"""扰动实验脚本：五类扰动 × 7 强度，输出指标曲线 + 权重曲线。

用法: cd rcdw_mgda && python -m scripts.perturb --ckpt runs/stage_b/rcdw.pt
"""
from __future__ import annotations

import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rcdw.data.synth import make_splits, WindowedDataset
from rcdw.models.rcdw import RCDW_MGDA
from rcdw.perturbation.inject import inject, PERTURBATION_KINDS
from rcdw.training.metrics import compute_per_gas_metrics
from rcdw.utils.degradation import hard_suppress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/stage_b/rcdw.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="runs/perturb")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    dc = cfg["data"]
    splits = make_splits(
        n_train=dc["n_train"], n_val=dc["n_val"], n_test=dc["n_test"],
        L=dc["window"], seed=dc["seed"],
    )
    X_test = torch.from_numpy(splits["test"][0])
    Y_test = torch.from_numpy(splits["test"][1])

    # 模型
    W_base = torch.tensor(cfg["model"]["W_base"], dtype=torch.float32)
    model = RCDW_MGDA(W_base, hidden=cfg["model"]["single_modal"]["hidden"])
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    model.eval()

    kinds = cfg["perturbation"]["kinds"]
    levels = cfg["perturbation"]["levels"]
    deg_cfg = cfg["degradation"]
    gas_names = ["O2", "CO2", "N2"]

    for kind in kinds:
        print(f"\n=== Perturbation: {kind} ===")
        results_by_level = []
        weights_by_level = []

        for level in levels:
            X_perturbed = inject(X_test, kind, level)
            with torch.no_grad():
                out = model(X_perturbed)
                # 退化硬抑制
                W_final, degraded = hard_suppress(
                    out["W"], out["E_pred"],
                    ratio=deg_cfg["ratio"], cap=deg_cfg["cap"],
                )
                C_fused = (W_final * out["Y_modal"]).sum(dim=1)

            metrics = compute_per_gas_metrics(C_fused, Y_test)
            results_by_level.append(metrics)

            # 平均权重: (M=3, G=3)
            W_avg = W_final.mean(dim=0).numpy()
            weights_by_level.append(W_avg)

            print(f"  level={level:.2f}  "
                  f"MAE={metrics['overall']['MAE']:.4f}  "
                  f"RMSE={metrics['overall']['RMSE']:.4f}  "
                  f"degraded={degraded.any().item()}")

        # --- 绘制指标曲线 ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for g, gas in enumerate(gas_names):
            ax = axes[g]
            for metric_name in ["MAE", "RMSE", "MRE"]:
                vals = [r[gas][metric_name] for r in results_by_level]
                ax.plot(levels, vals, marker="o", label=metric_name)
            ax.set_xlabel("Perturbation level")
            ax.set_ylabel("Error")
            ax.set_title(f"{kind} → {gas}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{kind}_metrics.png", dpi=150)
        plt.close()

        # --- 绘制权重曲线（以 CO2 为例）---
        fig, ax = plt.subplots(figsize=(8, 5))
        modal_names = ["NDIR", "TCD", "US"]
        for m, modal in enumerate(modal_names):
            vals = [w[m, 1] for w in weights_by_level]  # gas=1 (CO2)
            ax.plot(levels, vals, marker="s", label=modal)
        ax.set_xlabel("Perturbation level")
        ax.set_ylabel("Weight for CO₂")
        ax.set_title(f"{kind} → CO₂ modality weights")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(out_dir / f"{kind}_weights_CO2.png", dpi=150)
        plt.close()

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
