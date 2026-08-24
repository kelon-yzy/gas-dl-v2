"""分析 tv3 D2 TOF-PhaseNet 训练过程与实验结果并生成可视化图表.

读 outputs/tv3_d2/tof_phasenet_s20260704/ 下的 metrics.json 与 metrics_live.jsonl,
生成训练曲线、三组分 R2 对比、o2_bins 分箱、auxiliary TOF 对比四张图.
"""
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（Windows 优先 Microsoft YaHei）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---- 配置 ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUN_DIR = PROJECT_ROOT / "outputs" / "tv3_d2" / "tof_phasenet_s20260704"
METRICS_PATH = RUN_DIR / "metrics.json"
LIVE_PATH = RUN_DIR / "metrics_live.jsonl"
OUTPUT_DIR = RUN_DIR  # 图直接放 run 目录

# dataviz skill 验证过的分类色板（与 analyze_d0_results.py 对齐）
COMPONENT_COLORS = {
    "x_CO2": "#2a78d6",
    "x_O2": "#1baf7a",
    "x_N2": "#eda100",
}
COMP_LABELS = {"x_CO2": "CO2", "x_O2": "O2", "x_N2": "N2"}
COMPONENTS = ["x_CO2", "x_O2", "x_N2"]

# D0-observed 基线（clean 6000，见 docs/掘进通风实验日志.md §2.2，作为 D2 对比锚点）
D0_OBSERVED = {
    "val": {"x_CO2": 0.9878, "x_O2": 0.4226, "x_N2": 0.8799},
    "test": {"x_CO2": 0.9863, "x_O2": 0.4571, "x_N2": 0.8719},
    "extrapolation": {"x_CO2": 0.9863, "x_O2": 0.3708, "x_N2": 0.8719},
}
D0_OBSERVED_AUX = {
    # observed TOF 估计器对 true TOF 的 MAE，等于 D2 metrics 里的 observed_tof_mae_s
    "tof_mae_s": 8.199898729799315e-05,
}
# 验收线（dl_training_plan）
THRESHOLDS = {"x_CO2": 0.95, "x_O2": 0.70, "x_N2": 0.80}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#c3c2b7"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)


def plot_training_curves(metrics: dict[str, Any], out_path: Path) -> None:
    """图1：训练曲线 train/val loss + LR + best epoch 标记."""
    history = metrics["history"]
    epochs = np.array([h["epoch"] for h in history])
    train_loss = np.array([h["train_loss"] for h in history])
    val_loss = np.array([h["val_loss"] for h in history])
    lr = np.array([h["learning_rate"] for h in history])
    best_epoch = metrics["best_epoch"]["epoch"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, facecolor=SURFACE)
    fig.suptitle("D2 TOF-PhaseNet 训练曲线（tv3-formal-6000, seed=20260704）", fontsize=14, color=INK, y=0.97)

    ax1.plot(epochs, train_loss, color=COMPONENT_COLORS["x_N2"], linewidth=2, label="train_loss")
    ax1.plot(epochs, val_loss, color=COMPONENT_COLORS["x_CO2"], linewidth=2, label="val_loss", marker="o", markersize=4)
    ax1.axvline(best_epoch, color="#c62828", linestyle="--", linewidth=1.2, alpha=0.8)
    ax1.scatter([best_epoch], [val_loss[best_epoch - 1]], color="#c62828", zorder=5, s=60)
    ax1.annotate(f"best ep{best_epoch}\nval={val_loss[best_epoch-1]:.3f}",
                 xy=(best_epoch, val_loss[best_epoch - 1]),
                 xytext=(best_epoch + 1.5, val_loss.min() + 0.15),
                 fontsize=8, color="#c62828",
                 arrowprops=dict(arrowstyle="->", color="#c62828", lw=1))
    ax1.set_ylabel("loss (d2_tof_phase_loss)", color=INK)
    ax1.legend(loc="upper right", frameon=False, fontsize=9)
    ax1.set_ylim(0.8, 2.5)
    _style_ax(ax1)
    ax1.grid(axis="y", color=GRID, alpha=0.3, linewidth=0.6)

    ax2.step(epochs, lr, where="mid", color=COMPONENT_COLORS["x_O2"], linewidth=2)
    ax2.scatter([best_epoch], [lr[best_epoch - 1]], color="#c62828", zorder=5, s=50)
    ax2.set_ylabel("learning rate", color=INK)
    ax2.set_xlabel("epoch", color=INK)
    ax2.set_yscale("log")
    _style_ax(ax2)
    ax2.grid(axis="y", color=GRID, alpha=0.3, linewidth=0.6)

    # early stop 标注
    n_ran = metrics.get("stopped_early", False)
    stop_reason = metrics.get("stop_reason", "")
    if n_ran:
        ax1.text(0.98, 0.97, f"early stop @ ep{len(epochs)}\n{stop_reason}",
                 transform=ax1.transAxes, ha="right", va="top", fontsize=8,
                 color=MUTED, bbox=dict(facecolor=SURFACE, edgecolor=GRID, boxstyle="round,pad=0.3"))

    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.10, hspace=0.12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_val_r2_by_epoch(metrics: dict[str, Any], out_path: Path) -> None:
    """图2：val 三组分 R2 随 epoch 演化."""
    history = metrics["history"]
    epochs = np.array([h["epoch"] for h in history])
    best_epoch = metrics["best_epoch"]["epoch"]

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=SURFACE)
    fig.suptitle("D2 val 三组分 R2 随训练演化", fontsize=14, color=INK, y=0.97)

    for comp in COMPONENTS:
        vals = np.array([h["val_component_metrics"][comp]["r2"] for h in history])
        ax.plot(epochs, vals, color=COMPONENT_COLORS[comp], linewidth=2,
                label=f"{COMP_LABELS[comp]} (best={vals.max():.3f})", marker="o", markersize=3)
        # D0-observed 基线
        ax.axhline(D0_OBSERVED["val"][comp], color=COMPONENT_COLORS[comp],
                   linestyle=":", linewidth=1.2, alpha=0.6)

    ax.axvline(best_epoch, color="#c62828", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("val R2", color=INK)
    ax.set_ylim(-0.1, 1.05)
    _style_ax(ax)
    ax.grid(axis="y", color=GRID, alpha=0.3, linewidth=0.6)

    # 图例：模型线 + 基线说明
    from matplotlib.lines import Line2D
    custom = [
        Line2D([0], [0], color=COMPONENT_COLORS[c], lw=2, marker="o", markersize=4,
               label=f"{COMP_LABELS[c]} R2")
        for c in COMPONENTS
    ] + [
        Line2D([0], [0], color=MUTED, lw=1.2, ls=":", label="D0-observed 基线"),
        Line2D([0], [0], color="#c62828", lw=1.2, ls="--", label=f"best epoch {best_epoch}"),
    ]
    ax.legend(handles=custom, loc="center right", frameon=False, fontsize=9)

    fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_split_comparison(metrics: dict[str, Any], out_path: Path) -> None:
    """图3：best checkpoint 三组分 R2 在 val/test/extrap 三 split 对比 D0-observed."""
    splits = ["val", "test", "extrapolation"]
    evals = metrics["evaluations"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor=SURFACE, sharey=True)
    fig.suptitle("D2 best checkpoint 三组分 R2 vs D0-observed 基线", fontsize=14, color=INK, y=1.0)

    width = 0.35
    for i, split in enumerate(splits):
        ax = axes[i]
        d2_vals = [evals[split]["component_metrics"][c]["r2"] for c in COMPONENTS]
        d0_vals = [D0_OBSERVED[split][c] for c in COMPONENTS]
        x = np.arange(len(COMPONENTS))
        bars1 = ax.bar(x - width / 2, d2_vals, width, color=COMPONENT_COLORS.values(),
                       edgecolor=SURFACE, linewidth=1.5, label="D2 (this run)")
        bars2 = ax.bar(x + width / 2, d0_vals, width,
                       color=[COMPONENT_COLORS[c] for c in COMPONENTS],
                       edgecolor=SURFACE, linewidth=1.5, alpha=0.35, hatch="//", label="D0-observed")
        ax.set_xticks(x)
        ax.set_xticklabels([COMP_LABELS[c] for c in COMPONENTS], color=INK)
        ax.set_title(split, loc="left", fontsize=11, color=INK)
        if i == 0:
            ax.set_ylabel("R2", color=INK)
        ax.set_ylim(-0.15, 1.1)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style_ax(ax)
        ax.grid(axis="y", color=GRID, alpha=0.3, linewidth=0.6)
        # 数值标注
        for bar, val in zip(bars1, d2_vals):
            yoff = 0.02 if val >= 0 else -0.04
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + yoff, f"{val:.3f}",
                    ha="center", va=va, color=INK, fontsize=8)
        for bar, val in zip(bars2, d0_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                    ha="center", va="bottom", color=MUTED, fontsize=7)
        if i == 2:
            ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.12, wspace=0.12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_o2_bins(metrics: dict[str, Any], out_path: Path) -> None:
    """图4：best checkpoint O2 窄分箱 R2（val/test/extrap），对数刻度展示极端负值."""
    splits = ["val", "test", "extrapolation"]
    evals = metrics["evaluations"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=SURFACE, sharey=True)
    fig.suptitle("D2 best checkpoint O2 窄区间分箱 R2（o2_bins）", fontsize=14, color=INK, y=1.0)

    for i, split in enumerate(splits):
        ax = axes[i]
        bins = evals[split]["conditional_metrics"]["o2_bins"]["bins"]
        bin_keys = list(bins.keys())
        # 分箱标签取区间中点
        labels = [f"{bins[k]['range'][0]:.1f}\n~{bins[k]['range'][1]:.1f}" for k in bin_keys]
        o2_r2 = [bins[k]["component_metrics"]["x_O2"]["r2"] for k in bin_keys]
        counts = [bins[k]["count"] for k in bin_keys]
        # 用 O2 色，正负不同深浅
        colors = [COMPONENT_COLORS["x_O2"] if v >= 0 else "#c62828" for v in o2_r2]

        x = np.arange(len(bin_keys))
        bars = ax.bar(x, o2_r2, color=colors, edgecolor=SURFACE, linewidth=1.5, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, color=MUTED)
        ax.set_title(split, loc="left", fontsize=11, color=INK)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        _style_ax(ax)
        ax.grid(axis="y", color=GRID, alpha=0.3, linewidth=0.6)
        # 数值标注
        for bar, val, cnt in zip(bars, o2_r2, counts):
            yoff = 1.0 if val >= 0 else -1.0
            va = "bottom" if val >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2, val + yoff, f"{val:.1f}",
                    ha="center", va=va, color=INK, fontsize=8)
            ax.text(bar.get_x() + bar.get_width() / 2, 0.5, f"n={cnt}",
                    ha="center", va="bottom", color=MUTED, fontsize=7)
        ax.set_ylim(-35, 5)

    fig.text(0.5, 0.01, "分箱区间（O2 %）  |  红色=负 R2，绿色=非负 R2  |  D0-observed 同窗口亦全负（见实验日志 §2.2）",
             ha="center", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.16, wspace=0.1)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_auxiliary_tof(metrics: dict[str, Any], out_path: Path) -> None:
    """图5：auxiliary TOF 指标对比 D2 softargmax vs observed 估计器."""
    splits = ["val", "test", "extrapolation"]
    evals = metrics["evaluations"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)
    fig.suptitle("D2 auxiliary TOF 恢复质量（d2_minus_observed_tof_mae_s > 0 = 不如 observed 估计器）",
                 fontsize=13, color=INK, y=0.99)

    # 子图1：TOF MAE 对比
    ax = axes[0]
    d2_mae = [evals[s]["auxiliary_metrics"]["tof_mae_s"] * 1e6 for s in splits]  # 转 μs
    obs_mae = [evals[s]["auxiliary_metrics"]["observed_tof_mae_s"] * 1e6 for s in splits]
    x = np.arange(len(splits))
    width = 0.35
    bars1 = ax.bar(x - width / 2, d2_mae, width, color=COMPONENT_COLORS["x_O2"],
                   edgecolor=SURFACE, linewidth=1.5, label="D2 softargmax")
    bars2 = ax.bar(x + width / 2, obs_mae, width, color="#7e7e7e",
                   edgecolor=SURFACE, linewidth=1.5, label="observed 估计器（基线）")
    ax.set_xticks(x)
    ax.set_xticklabels(splits, color=INK)
    ax.set_ylabel("TOF MAE (μs)", color=INK)
    ax.set_title("TOF 恢复 MAE", loc="left", fontsize=11, color=INK)
    for bar, val in zip(bars1, d2_mae):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val:.2f}",
                ha="center", va="bottom", color=INK, fontsize=8)
    for bar, val in zip(bars2, obs_mae):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val:.2f}",
                ha="center", va="bottom", color=MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_ax(ax)
    ax.grid(axis="y", color=GRID, alpha=0.3, linewidth=0.6)
    ax.set_ylim(0, max(d2_mae + obs_mae) * 1.25)

    # 子图2：d2_minus_observed 差值（>0 即不如基线）
    ax = axes[1]
    diffs = [evals[s]["auxiliary_metrics"]["d2_minus_observed_tof_mae_s"] * 1e6 for s in splits]
    corrs = [evals[s]["auxiliary_metrics"]["tof_corr"] for s in splits]
    colors = ["#c62828" if d > 0 else COMPONENT_COLORS["x_O2"] for d in diffs]
    bars = ax.bar(x, diffs, color=colors, edgecolor=SURFACE, linewidth=1.5, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, color=INK)
    ax.set_ylabel("ΔMAE = D2 − observed (μs)", color=INK)
    ax.set_title("ΔMAE（正值=D2 更差，停止条件2触发）", loc="left", fontsize=11, color=INK)
    ax.axhline(0, color=MUTED, linewidth=1)
    for bar, val, corr in zip(bars, diffs, corrs):
        yoff = 0.005 if val >= 0 else -0.005
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, val + yoff,
                f"Δ={val:.3f}\ncorr={corr:.4f}",
                ha="center", va=va, color=INK, fontsize=8)
    _style_ax(ax)
    ax.grid(axis="y", color=GRID, alpha=0.3, linewidth=0.6)
    ymax = max(diffs) * 1.4 if max(diffs) > 0 else 0.01
    ymin = min(diffs) * 1.4 if min(diffs) < 0 else 0
    ax.set_ylim(ymin, ymax)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.10, wspace=0.22)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary(metrics: dict[str, Any], live: list[dict[str, Any]]) -> None:
    """打印关键指标摘要到 stdout."""
    evals = metrics["evaluations"]
    best = metrics["best_epoch"]
    print("\n" + "=" * 70)
    print("D2 TOF-PhaseNet 训练结果摘要")
    print("=" * 70)
    print(f"实际训练 epoch：{len(metrics['history'])} / 配置 {metrics.get('epochs', '?')}")
    print(f"best epoch：{best['epoch']}（val_loss={best['val_loss']:.4f}）")
    print(f"early stop：{metrics.get('stopped_early', False)}（{metrics.get('stop_reason', '')}）")
    print()
    print("best checkpoint 三组分 R2：")
    print(f"{'split':<14}{'CO2':>10}{'O2':>10}{'N2':>10}")
    for split in ["val", "test", "extrapolation"]:
        cm = evals[split]["component_metrics"]
        print(f"{split:<14}{cm['x_CO2']['r2']:>10.4f}{cm['x_O2']['r2']:>10.4f}{cm['x_N2']['r2']:>10.4f}")
    print()
    print("O2 vs D0-observed 基线（目标 +0.05）：")
    for split in ["val", "test", "extrapolation"]:
        d2 = evals[split]["component_metrics"]["x_O2"]["r2"]
        d0 = D0_OBSERVED[split]["x_O2"]
        print(f"  {split:<14} D2={d2:.4f}  D0-obs={d0:.4f}  Δ={d2 - d0:+.4f}")
    print()
    print("auxiliary TOF（停止条件2: d2_minus_observed_tof_mae_s < 0 才算 D2 优于 observed）：")
    for split in ["val", "test", "extrapolation"]:
        aux = evals[split]["auxiliary_metrics"]
        print(f"  {split:<14} D2_MAE={aux['tof_mae_s']:.3e}s  obs_MAE={aux['observed_tof_mae_s']:.3e}s  "
              f"Δ={aux['d2_minus_observed_tof_mae_s']:+.3e}s  corr={aux['tof_corr']:.5f}")
    print("=" * 70)


def main() -> None:
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    live = load_jsonl(LIVE_PATH)

    print_summary(metrics, live)
    plot_training_curves(metrics, OUTPUT_DIR / "d2_training_curves.png")
    plot_val_r2_by_epoch(metrics, OUTPUT_DIR / "d2_val_r2_by_epoch.png")
    plot_split_comparison(metrics, OUTPUT_DIR / "d2_split_comparison.png")
    plot_o2_bins(metrics, OUTPUT_DIR / "d2_o2_bins.png")
    plot_auxiliary_tof(metrics, OUTPUT_DIR / "d2_auxiliary_tof.png")
    print("\n全部图表已生成于：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
