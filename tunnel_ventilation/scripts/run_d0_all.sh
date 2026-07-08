#!/usr/bin/env bash
# ============================================================================
# D0 实验一键运行脚本（服务器端 — 完整 6 组）
#
# 配置对齐:
#   - 数据集生成命令与 server_training_guide.md §6.1 一致
#   - D0-oracle 配置与 tv3_rocket_ridge.json (R0) 除 feature_builder/output_dir 外完全相同
#   - 其余 D0 配置仅按方案文档变化 physics_arrays / slow_channels
#
# 硬件说明:
#   - D0 使用 sklearn RidgeCV，纯 CPU 计算，不使用 GPU
#   - RTX 5880 在 D1-D5 的 PyTorch 训练中才会用到
#   - 数据集生成使用 CPU 多进程，WORKERS 控制并行度
#   - 内存瓶颈在数据集生成阶段 (~15 GB peak @ workers=4)
#
# 用法:
#   cd gas-dl-v2/tunnel_ventilation
#   source .venv/bin/activate
#
#   # 默认 workers=4
#   bash scripts/run_d0_all.sh
#
#   # 自定义 workers（CPU 核数充裕时）
#   WORKERS=8 bash scripts/run_d0_all.sh
#
# 产出:
#   data/tv3-formal-6000/                        # 数据集 (~29 GB, memmap)
#   data/tv3-formal-6000/features/rocket/d0_*/   # 特征缓存 (每个实验独立)
#   outputs/tv3_d0/<exp_name>/metrics.json       # 结果
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---- 可覆盖参数 ----------------------------------------------------------------
DATASET="${DATASET:-tv3-formal-6000}"
DATA_DIR="data/${DATASET}"
SEED="${SEED:-20260704}"
TIMESTEPS="${TIMESTEPS:-512}"
DT_S="${DT_S:-0.5}"
WORKERS="${WORKERS:-4}"        # 数据集生成并行度；内存不足时降为 2
SEQUENCES="${SEQUENCES:-6000}"

# ---- 环境检查 ------------------------------------------------------------------
echo "==== 环境信息 ===="
echo "Python: $(python --version 2>&1)"
echo "工作目录: $(pwd)"
echo "数据集: ${DATA_DIR}"
echo "workers: ${WORKERS}"
echo ""

# ---- 配置矩阵 ------------------------------------------------------------------
# 格式: "exp_name|config_file"
EXPERIMENTS=(
  "oracle_ridge|configs/tv3_d0_oracle_ridge.json"
  "observed_ridge|configs/tv3_d0_observed_ridge.json"
  "tof_only_ridge|configs/tv3_d0_tof_only_ridge.json"
  "slow_only_ridge|configs/tv3_d0_slow_only_ridge.json"
  "no_tof_ridge|configs/tv3_d0_no_tof_ridge.json"
  "no_tcs_ridge|configs/tv3_d0_no_tcs_ridge.json"
)

# ---- 1. 生成数据集（如不存在）---------------------------------------------------
if [ -f "${DATA_DIR}/manifest.json" ]; then
  echo "[SKIP] 数据集 ${DATA_DIR} 已存在，跳过生成。"
  SEQUENCE_COUNT=$(python -c "import json; print(json.load(open('${DATA_DIR}/manifest.json'))['sequence_count'])")
  echo "       现有序列数: ${SEQUENCE_COUNT}"
else
  echo "[GEN] 生成数据集 ${DATASET} (${SEQUENCES} 序列 × ${TIMESTEPS} 时步, workers=${WORKERS})..."
  echo "      命令对齐 server_training_guide.md §6.1"
  python -m tv3.pipeline.generate_tunnel_ventilation_benchmark \
    --output-root data \
    --dataset "${DATASET}" \
    --sequences "${SEQUENCES}" \
    --seed "${SEED}" \
    --timesteps "${TIMESTEPS}" \
    --dt-s "${DT_S}" \
    --optical-absorption-backend empirical_v1 \
    --storage memmap \
    --workers "${WORKERS}" \
    --skip-fiber-mic
  echo "[OK] 数据集生成完成。"
fi

# ---- 2. 运行 D0 实验 -----------------------------------------------------------
echo ""
echo "=============================================="
echo "  D0 实验矩阵 (6 组 RidgeCV, CPU only)"
echo "  alpha 范围: 0.0001 ~ 100.0 (13 点，与 R0 一致)"
echo "=============================================="

RUN=0
TOTAL=${#EXPERIMENTS[@]}
for entry in "${EXPERIMENTS[@]}"; do
  RUN=$((RUN + 1))
  EXP_NAME="${entry%%|*}"
  CONFIG="${entry##*|}"

  echo ""
  echo "---- [${RUN}/${TOTAL}] ${EXP_NAME} ----"
  START_TS=$(date +%s)
  python -m tv3.pipeline.run_tv3_rocket_baseline --config "${CONFIG}"
  ELAPSED=$(( $(date +%s) - START_TS ))
  echo "[OK] ${EXP_NAME} (${ELAPSED}s) -> outputs/tv3_d0/${EXP_NAME}/metrics.json"
done

# ---- 3. 汇总表 -----------------------------------------------------------------
echo ""
echo "=============================================="
echo "  D0 汇总表"
echo "=============================================="
echo ""

python - << 'PYEOF'
import json
from pathlib import Path

EXPS = [
    ("D0-oracle",     "oracle_ridge",    "R0 上限复现"),
    ("D0-observed",   "observed_ridge",  "可部署基线"),
    ("D0-tof-only",   "tof_only_ridge",  "TOF 支撑度"),
    ("D0-slow-only",  "slow_only_ridge", "慢通道对照"),
    ("D0-no-tof",     "no_tof_ridge",    "TOF 必要性"),
    ("D0-no-tcs",     "no_tcs_ridge",    "TCS 边际贡献"),
]

HEADER_FMT = (
    f"{'实验':<16s}"
    f"{'val CO2 R2':>11s}  {'val O2 R2':>10s}  {'val N2 R2':>10s}"
    f"  {'test O2 R2':>10s}  {'extrap O2 R2':>12s}"
    f"  {'features':>9s}  {'alpha':>8s}"
    f"  {'目的':<16s}"
)
SEP = "-" * len(HEADER_FMT)

print(HEADER_FMT)
print(SEP)

oracle_o2 = None
observed_o2 = None

for label, slug, purpose in EXPS:
    metrics_path = Path(f"outputs/tv3_d0/{slug}/metrics.json")
    if not metrics_path.is_file():
        print(f"{label:<16s}  {'-- MISSING --':>60s}  {purpose:<16s}")
        continue
    m = json.loads(metrics_path.read_text(encoding="utf-8"))

    def r2(split, comp):
        return m["evaluations"][split]["component_metrics"][comp]["r2"]

    val_co2   = r2("val", "x_CO2")
    val_o2    = r2("val", "x_O2")
    val_n2    = r2("val", "x_N2")
    test_o2   = r2("test", "x_O2")
    extrap_o2 = r2("extrapolation", "x_O2")
    n_feat    = m["feature_count"]
    alpha     = m["diagnostics"]["selected_alpha"]

    print(
        f"{label:<16s}"
        f"{val_co2:11.4f}  {val_o2:10.4f}  {val_n2:10.4f}"
        f"  {test_o2:10.4f}  {extrap_o2:12.4f}"
        f"  {n_feat:9d}  {alpha:8.1f}"
        f"  {purpose:<16s}"
    )

    if label == "D0-oracle":
        oracle_o2 = val_o2
    elif label == "D0-observed":
        observed_o2 = val_o2

print(SEP)
print()

# ---- 判读规则（对齐方案文档 §D0 判读规则）----
if oracle_o2 is not None and observed_o2 is not None:
    gap = oracle_o2 - observed_o2
    print(f"D0-oracle  val O2 R2  = {oracle_o2:.4f}")
    print(f"D0-observed val O2 R2 = {observed_o2:.4f}")
    print(f"Oracle 膨胀 (gap)      = {gap:.4f}")
    print()

    if gap <= 0.03:
        print(">>> 判读: R0 几乎无 oracle 膨胀。D1/D4 可直接以 D0-observed 为强基线推进。")
    elif gap <= 0.10:
        print(">>> 判读: oracle 有边际影响。D1 仍可推进，但论文必须用 D0-observed 为真实基线。")
    else:
        print(">>> 判读: R0 明显受 oracle 特征抬升 (>0.10)。优先推进 D2 TOF/phase 可微估计。")

    # alpha bound warning
    if alpha_min := min(
        m["diagnostics"]["selected_alpha"]
        for label, slug, _p in EXPS
        if (Path(f"outputs/tv3_d0/{slug}/metrics.json").is_file()
            and (m := json.loads(Path(f"outputs/tv3_d0/{slug}/metrics.json").read_text(encoding="utf-8"))))
    ):
        # read all selected alphas
        alphas = []
        for label, slug, _p in EXPS:
            mp = Path(f"outputs/tv3_d0/{slug}/metrics.json")
            if mp.is_file():
                alphas.append(json.loads(mp.read_text(encoding="utf-8"))["diagnostics"]["selected_alpha"])
        if any(a <= 0.0001 for a in alphas):
            print()
            print(">>> 注意: 部分实验 selected_alpha 触底 0.0001。")
            print("    若 D0 结论关键，可扩展 alpha 下限至 1e-5 / 1e-6 重跑确认。")
PYEOF

echo ""
echo "[DONE] 全部 D0 实验完成。"
echo "输出: outputs/tv3_d0/"
echo "特征缓存: data/tv3-formal-6000/features/rocket/d0_*/"
