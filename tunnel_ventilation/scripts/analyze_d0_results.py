"""分析 tv3 D0 Ridge 特征拆分实验结果并生成可视化图表."""
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（Windows 优先 Microsoft YaHei）
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ---- 配置 ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tv3_d0_local"
CONFIGS = {
    "oracle": OUTPUT_DIR / "oracle_ridge" / "metrics.json",
    "observed": OUTPUT_DIR / "observed_ridge" / "metrics.json",
    "tof_only": OUTPUT_DIR / "tof_only_ridge" / "metrics.json",
    "slow_only": OUTPUT_DIR / "slow_only_ridge" / "metrics.json",
}
SPLITS = ["train", "val", "test", "extrapolation"]
COMPONENTS = ["x_CO2", "x_O2", "x_N2"]
CONFIG_ORDER = ["oracle", "observed", "tof_only", "slow_only"]

# dataviz skill 验证过的分类色板
COMPONENT_COLORS = {
    "x_CO2": "#2a78d6",
    "x_O2": "#1baf7a",
    "x_N2": "#eda100",
}
CONFIG_COLORS = {
    "oracle": "#2a78d6",
    "observed": "#1baf7a",
    "tof_only": "#eda100",
    "slow_only": "#008300",
}
COMP_LABELS = {"x_CO2": "CO2", "x_O2": "O2", "x_N2": "N2"}
THRESHOLDS = {"x_CO2": 0.95, "x_O2": 0.70, "x_N2": 0.80}


def load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_split_r2(metrics: dict[str, Any]) -> dict[str, dict[str, float]]:
    """提取每个 split 下三组分的 R2."""
    return {
        split: {
            comp: metrics["evaluations"][split]["component_metrics"][comp]["r2"]
            for comp in COMPONENTS
        }
        for split in SPLITS
    }


def extract_val_o2_r2(metrics: dict[str, Any]) -> float:
    return metrics["evaluations"]["val"]["component_metrics"]["x_O2"]["r2"]


def extract_top_features(metrics: dict[str, Any]) -> dict[str, list[tuple[str, float]]]:
    """提取每个组分的 top-5 特征及归一化权重."""
    top = {}
    for comp in COMPONENTS:
        groups = metrics["diagnostics"]["top_feature_groups"][comp]
        names = [g["group"].replace("physics:", "").replace("slow:", "") for g in groups]
        weights = np.array([g["abs_coef_sum"] for g in groups], dtype=float)
        total = weights.sum()
        top[comp] = [(name, w / total if total > 0 else 0.0) for name, w in zip(names, weights)]
    return top


def main() -> None:
    results = {cfg: load_metrics(path) for cfg, path in CONFIGS.items()}

    # 聚合数据
    split_r2 = {cfg: extract_split_r2(m) for cfg, m in results.items()}
    val_o2_r2 = {cfg: extract_val_o2_r2(m) for cfg, m in results.items()}
    top_features = {cfg: extract_top_features(m) for cfg, m in results.items()}

    # 创建画布：使用 gridspec 布局
    fig = plt.figure(figsize=(12, 14), facecolor="#fcfcfb")
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.25)
    fig.suptitle("tv3 D0 Ridge 特征拆分实验结果", fontsize=16, color="#0b0b0b", y=0.98)

    # ---- 图 1：各 split 下三组分 R2 对比（small multiples，每组分一个子图）----
    for comp_idx, comp in enumerate(COMPONENTS):
        ax = fig.add_subplot(gs[0, comp_idx])
        x = np.arange(len(SPLITS))
        width = 0.18
        for j, cfg in enumerate(CONFIG_ORDER):
            vals = [split_r2[cfg][split][comp] for split in SPLITS]
            offset = (j - 1.5) * width
            ax.bar(x + offset, vals, width, label=cfg, color=CONFIG_COLORS[cfg], edgecolor="#fcfcfb", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(SPLITS, fontsize=8)
        ax.set_title(f"{COMP_LABELS[comp]} R2", loc="left", fontsize=11, color="#0b0b0b")
        if comp_idx == 0:
            ax.set_ylabel("R2")
        if comp_idx == 2:
            ax.legend(title="配置", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
        ax.set_ylim(-0.3, 1.1)
        ax.axhline(THRESHOLDS[comp], color="#c3c2b7", linestyle="--", linewidth=1)
        ax.set_facecolor("#fcfcfb")
        ax.tick_params(colors="#52514e")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ---- 图 2：val 集 O2 R2 横向对比条形图 ----
    ax2 = fig.add_subplot(gs[1, :])
    cfg_labels = ["oracle", "observed", "tof_only", "slow_only"]
    o2_vals = [val_o2_r2[cfg] for cfg in cfg_labels]
    y_pos = np.arange(len(cfg_labels))
    bars = ax2.barh(y_pos, o2_vals, color=[CONFIG_COLORS[cfg] for cfg in cfg_labels], edgecolor="#fcfcfb", linewidth=2, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cfg_labels)
    ax2.set_xlabel("val R2")
    ax2.set_title("val 集 O2 R2 横向对比", loc="left", fontsize=12, color="#0b0b0b")
    ax2.set_xlim(-0.3, 0.7)
    ax2.axvline(0.7, color="#c3c2b7", linestyle="--", linewidth=1)
    ax2.text(0.72, 3.3, "验收线 0.70", color="#52514e", fontsize=8, va="top")
    for bar, val in zip(bars, o2_vals):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", ha="left", color="#0b0b0b", fontsize=9)
    ax2.set_facecolor("#fcfcfb")
    ax2.tick_params(colors="#52514e")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ---- 图 3：observed 配置 top-5 特征重要性归一化堆叠条形图 ----
    ax3 = fig.add_subplot(gs[2, :])
    cfg = "observed"
    comp_names = ["CO2", "O2", "N2"]
    y_positions = []
    left = np.zeros(len(comp_names))
    feature_names_set: set[str] = set()
    for comp_idx, comp in enumerate(COMPONENTS):
        for feat_name, _weight in top_features[cfg][comp]:
            feature_names_set.add(feat_name)
    sorted_features = sorted(feature_names_set)
    n_features = len(sorted_features)
    cmap = plt.get_cmap("Blues")
    feat_colors = {feat: cmap(0.25 + 0.65 * i / max(1, n_features - 1)) for i, feat in enumerate(sorted_features)}

    bar_height = 0.5
    for comp_idx, (comp, label) in enumerate(zip(COMPONENTS, comp_names)):
        y_positions.append(comp_idx)
        for feat_name, weight in top_features[cfg][comp]:
            ax3.barh(comp_idx, weight, left=left[comp_idx], color=feat_colors[feat_name], edgecolor="#fcfcfb", linewidth=1, height=bar_height)
            if weight > 0.11:
                ax3.text(left[comp_idx] + weight / 2, comp_idx, feat_name, ha="center", va="center", color="#ffffff", fontsize=7)
            left[comp_idx] += weight

    ax3.set_yticks(y_positions)
    ax3.set_yticklabels(comp_names)
    ax3.set_xlabel("归一化特征重要性")
    ax3.set_title(f"{cfg} 配置 top-5 特征重要性（归一化堆叠）", loc="left", fontsize=12, color="#0b0b0b")
    ax3.set_xlim(0, 1.05)
    handles = [plt.Rectangle((0, 0), 1, 1, color=feat_colors[f]) for f in sorted_features]
    ax3.legend(handles, sorted_features, title="特征", loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8, ncol=3)
    ax3.set_facecolor("#fcfcfb")
    ax3.tick_params(colors="#52514e")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.08, right=0.88, top=0.94, bottom=0.12, hspace=0.35, wspace=0.25)
    out_path = OUTPUT_DIR / "d0_analysis.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
