"""扰动实验脚本：5 类扰动 × 7 强度，输出指标曲线 + 权重曲线。

v1.2 Phase 4+5：通道索引重映射到 12 维新布局（方案 §9.1）。

用法:
    cd rcdw_mgda && python -m scripts.perturb \\
        --ckpt runs/stage_b/rcdw.pt \\
        --config configs/smoke.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from rcdw.data.dataset import BenchmarkDataset
from rcdw.models.rcdw import RCDW_MGDA
from rcdw.perturbation.inject import PERTURBATION_KINDS, inject
from rcdw.training.metrics import compute_per_gas_metrics
from rcdw.utils.degradation import hard_suppress


def _collect_split_tensor(ds: BenchmarkDataset) -> tuple[torch.Tensor, torch.Tensor]:
    """把 BenchmarkDataset 全部窗口拼成 (N, L, 12) + (N, 3)。"""
    xs, ys = [], []
    for idx in range(len(ds)):
        x_w, y = ds[idx]
        xs.append(x_w)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/stage_b/rcdw.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="runs/perturb")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数据
    dc = cfg["data"]
    data_root = dc.get("dataset_root")
    if not data_root:
        raise ValueError("configs/*.yaml 必须含 data.dataset_root 字段。")
    window = int(dc.get("window", 8))
    modalities = tuple(dc.get("train_modalities", ("slow", "ultrasonic")))
    test_ds = BenchmarkDataset(
        data_root, split="test", window=window, modalities=modalities
    )
    X_test, Y_test = _collect_split_tensor(test_ds)
    print(f"[perturb] test windows={len(test_ds)} shape={tuple(X_test.shape)}")

    # 模型
    W_base = torch.tensor(cfg["model"]["W_base"], dtype=torch.float32)
    fusion_kwargs = cfg["model"].get("fusion", {}) or {}
    model = RCDW_MGDA(
        W_base,
        hidden=cfg["model"]["single_modal"]["hidden"],
        window=window,
        fusion_kwargs=fusion_kwargs,
    )
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    model.eval()

    kinds = cfg["perturbation"]["kinds"]
    levels = cfg["perturbation"]["levels"]
    deg_cfg = cfg["degradation"]
    gas_names = ["O2", "CO2", "N2"]

    for kind in kinds:
        if kind not in PERTURBATION_KINDS:
            raise ValueError(f"unknown perturbation kind: {kind}")
        print(f"\n=== Perturbation: {kind} ===")
        results_by_level = []
        weights_by_level = []

        for level in levels:
            X_perturbed = inject(X_test, kind, level)
            with torch.no_grad():
                out = model(X_perturbed)
                W_final, degraded = hard_suppress(
                    out["W"], out["E_pred"],
                    ratio=deg_cfg["ratio"], cap=deg_cfg["cap"],
                )
                C_fused = (W_final * out["Y_modal"]).sum(dim=1)

            metrics = compute_per_gas_metrics(C_fused, Y_test)
            results_by_level.append(metrics)

            W_avg = W_final.mean(dim=0).numpy()
            weights_by_level.append(W_avg)

            print(
                f"  level={level:.2f}  "
                f"MAE={metrics['overall']['MAE']:.4f}  "
                f"RMSE={metrics['overall']['RMSE']:.4f}  "
                f"degraded={degraded.any().item()}"
            )

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
            vals = [w[m, 1] for w in weights_by_level]
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
