"""扰动实验脚本：按配置中的扰动类型 × 多强度输出指标曲线 + 权重曲线。

v1.2 Phase 4+5：通道索引重映射到 12 维新布局（方案 §9.1）。
Phase 6E 起配置可加入 ``pressure_drift``。
Phase R1 起可通过 ``--report-json`` 输出 raw/HS 双指标与 ErrorNet 校准报告。

用法:
    cd rcdw_mgda && python -m scripts.perturb \\
        --ckpt runs/stage_b/rcdw.pt \\
        --config configs/smoke.yaml
"""
from __future__ import annotations

import argparse
import json
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


def _metric_payload(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """把 metric dict 转成 JSON 友好的纯 float。"""
    return {
        gas: {name: float(value) for name, value in values.items()}
        for gas, values in metrics.items()
    }


def _matrix_payload(
    values: torch.Tensor,
    *,
    row_names: list[str],
    col_names: list[str],
) -> dict[str, dict[str, float]]:
    """把 (M, G) 张量转成 {modal: {gas: value}}。"""
    return {
        row: {
            col: float(values[row_idx, col_idx].item())
            for col_idx, col in enumerate(col_names)
        }
        for row_idx, row in enumerate(row_names)
    }


def _safe_pearson(x: torch.Tensor, y: torch.Tensor) -> float | None:
    """计算 Pearson 相关；任一向量零方差时返回 None。"""
    x = x.float()
    y = y.float()
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered.square().sum()) * (y_centered.square().sum()))
    if denom <= 1e-12:
        return None
    return float((x_centered * y_centered).sum().item() / denom.item())


def _thresholded_relative_error_payload(
    pred: torch.Tensor,
    ref: torch.Tensor,
    *,
    gas_names: list[str],
    threshold: float = 0.01,
    eps: float = 1e-8,
) -> dict[str, object]:
    """计算低真值过滤后的 MRE/MaxRE，避免极小 ref 主导报告。"""
    abs_error = (pred - ref).abs()
    rel_error = abs_error / (ref.abs() + eps)
    by_gas: dict[str, dict[str, float | int | None]] = {}
    for gas_idx, gas in enumerate(gas_names):
        mask = ref[:, gas_idx].abs() >= threshold
        count = int(mask.sum().item())
        if count == 0:
            by_gas[gas] = {"count": 0, "MRE": None, "MaxRE": None}
            continue
        vals = rel_error[:, gas_idx][mask]
        by_gas[gas] = {
            "count": count,
            "MRE": float(vals.mean().item() * 100.0),
            "MaxRE": float(vals.max().item() * 100.0),
        }

    overall_mask = ref.abs() >= threshold
    overall_count = int(overall_mask.sum().item())
    if overall_count == 0:
        overall = {"count": 0, "MRE": None, "MaxRE": None}
    else:
        overall_vals = rel_error[overall_mask]
        overall = {
            "count": overall_count,
            "MRE": float(overall_vals.mean().item() * 100.0),
            "MaxRE": float(overall_vals.max().item() * 100.0),
        }
    return {
        "threshold": float(threshold),
        "by_gas": by_gas,
        "overall": overall,
    }


def _error_calibration_payload(
    y_modal: torch.Tensor,
    e_pred: torch.Tensor,
    y_ref: torch.Tensor,
    *,
    modal_names: list[str],
    gas_names: list[str],
) -> dict[str, object]:
    """汇总 ErrorNet 预测误差与真实单模态误差的校准关系。"""
    actual = (y_modal - y_ref.unsqueeze(1)).abs()
    by_pair: dict[str, dict[str, object]] = {}
    for modal_idx, modal in enumerate(modal_names):
        by_pair[modal] = {}
        for gas_idx, gas in enumerate(gas_names):
            actual_vec = actual[:, modal_idx, gas_idx]
            pred_vec = e_pred[:, modal_idx, gas_idx]
            by_pair[modal][gas] = {
                "actual_mae_mean": float(actual_vec.mean().item()),
                "pred_error_mean": float(pred_vec.mean().item()),
                "pearson": _safe_pearson(actual_vec, pred_vec),
            }

    # 逐气体判断 ErrorNet 是否选中真实误差最小的模态。
    best_actual = actual.argmin(dim=1)  # (B, G)
    best_pred = e_pred.argmin(dim=1)    # (B, G)
    best_modality_accuracy = {
        gas: float((best_actual[:, gas_idx] == best_pred[:, gas_idx]).float().mean().item())
        for gas_idx, gas in enumerate(gas_names)
    }
    best_modality_accuracy["overall"] = float(
        (best_actual == best_pred).float().mean().item()
    )
    return {
        "by_modality_gas": by_pair,
        "best_modality_accuracy": best_modality_accuracy,
    }


def _summarize_model_outputs(
    out: dict[str, torch.Tensor],
    y_ref: torch.Tensor,
    *,
    deg_cfg: dict,
    modal_names: list[str],
    gas_names: list[str],
) -> tuple[dict[str, object], torch.Tensor]:
    """计算 raw fusion、hard-suppress fusion、权重与退化统计。"""
    raw_metrics = compute_per_gas_metrics(out["C"], y_ref)
    W_final, degraded = hard_suppress(
        out["W"],
        out["E_pred"],
        ratio=deg_cfg["ratio"],
        cap=deg_cfg["cap"],
    )
    C_hs = (W_final * out["Y_modal"]).sum(dim=1)
    hs_metrics = compute_per_gas_metrics(C_hs, y_ref)

    degraded_rate = degraded.float().mean(dim=0)
    payload: dict[str, object] = {
        "raw": _metric_payload(raw_metrics),
        "raw_thresholded_relative_error": _thresholded_relative_error_payload(
            out["C"], y_ref, gas_names=gas_names
        ),
        "hard_suppress": _metric_payload(hs_metrics),
        "hard_suppress_thresholded_relative_error": (
            _thresholded_relative_error_payload(C_hs, y_ref, gas_names=gas_names)
        ),
        "weights_raw_mean": _matrix_payload(
            out["W"].mean(dim=0),
            row_names=modal_names,
            col_names=gas_names,
        ),
        "weights_hard_suppress_mean": _matrix_payload(
            W_final.mean(dim=0),
            row_names=modal_names,
            col_names=gas_names,
        ),
        "degraded_rate": _matrix_payload(
            degraded_rate,
            row_names=modal_names,
            col_names=gas_names,
        ),
        "degraded_any_rate": float(degraded.any(dim=(1, 2)).float().mean().item()),
    }
    return payload, W_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="runs/stage_b/rcdw.pt")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output-dir", type=str, default="runs/perturb")
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="可选：输出结构化诊断报告 JSON（raw/HS 双指标、degraded rate、ErrorNet 校准）。",
    )
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
    apply_input_scaler = dc.get("apply_input_scaler")
    test_ds = BenchmarkDataset(
        data_root,
        split="test",
        window=window,
        modalities=modalities,
        apply_input_scaler=apply_input_scaler,
    )
    X_test, Y_test = _collect_split_tensor(test_ds)
    print(f"[perturb] apply_input_scaler={apply_input_scaler}")
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
    modal_names = ["NDIR", "TCD", "US"]

    report: dict[str, object] | None = None
    if args.report_json:
        report = {
            "schema_version": "rcdw-perturb-report-1",
            "config": str(args.config),
            "ckpt": str(args.ckpt),
            "output_dir": str(out_dir),
            "dataset_root": str(data_root),
            "split": "test",
            "window": window,
            "test_windows": len(test_ds),
            "input_shape": list(X_test.shape),
            "apply_input_scaler": apply_input_scaler,
            "perturbation": {
                "space": cfg.get("perturbation", {}).get("space", "standardized"),
                "kinds": list(kinds),
                "levels": [float(level) for level in levels],
            },
            "degradation": {
                "ratio": float(deg_cfg["ratio"]),
                "cap": float(deg_cfg["cap"]),
            },
            "baseline": {},
            "perturbations": {},
        }

        with torch.no_grad():
            baseline_out = model(X_test)
        baseline_payload, _baseline_w = _summarize_model_outputs(
            baseline_out,
            Y_test,
            deg_cfg=deg_cfg,
            modal_names=modal_names,
            gas_names=gas_names,
        )
        baseline_payload["error_calibration"] = _error_calibration_payload(
            baseline_out["Y_modal"],
            baseline_out["E_pred"],
            Y_test,
            modal_names=modal_names,
            gas_names=gas_names,
        )
        report["baseline"] = baseline_payload

    for kind in kinds:
        if kind not in PERTURBATION_KINDS:
            raise ValueError(f"unknown perturbation kind: {kind}")
        print(f"\n=== Perturbation: {kind} ===")
        results_by_level = []
        weights_by_level = []
        kind_report: list[dict[str, object]] = []

        for level in levels:
            X_perturbed = inject(X_test, kind, level)
            with torch.no_grad():
                out = model(X_perturbed)
                level_payload, W_final = _summarize_model_outputs(
                    out,
                    Y_test,
                    deg_cfg=deg_cfg,
                    modal_names=modal_names,
                    gas_names=gas_names,
                )

            metrics = level_payload["hard_suppress"]
            results_by_level.append(metrics)

            W_avg = W_final.mean(dim=0).numpy()
            weights_by_level.append(W_avg)
            if report is not None:
                kind_report.append(
                    {
                        "level": float(level),
                        **level_payload,
                    }
                )

            print(
                f"  level={level:.2f}  "
                f"raw_MAE={level_payload['raw']['overall']['MAE']:.4f}  "
                f"hs_MAE={level_payload['hard_suppress']['overall']['MAE']:.4f}  "
                f"hs_RMSE={level_payload['hard_suppress']['overall']['RMSE']:.4f}  "
                f"degraded_rate={level_payload['degraded_any_rate']:.3f}"
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

        if report is not None:
            report["perturbations"][kind] = kind_report

    if report is not None:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Report saved to {report_path}")

    print(f"\nPlots saved to {out_dir}/")


if __name__ == "__main__":
    main()
