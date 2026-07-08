#!/usr/bin/env bash
# ============================================================================
# D0 最小实验集（必做 3 组: oracle / observed / tof-only）
#
# 配置对齐:
#   - 数据集生成命令与 server_training_guide.md §6.1 一致
#   - D0-oracle 配置与 tv3_rocket_ridge.json (R0) 除 feature_builder/output_dir 外完全相同
#
# 硬件说明:
#   - D0 使用 sklearn RidgeCV，纯 CPU 计算，不使用 GPU
#   - RTX 5880 在 D1-D5 的 PyTorch 训练中才会用到
#
# 用法:
#   cd gas-dl-v2/tunnel_ventilation
#   source .venv/bin/activate
#   bash scripts/run_d0_minimal.sh
#
#   # 自定义 workers
#   WORKERS=8 bash scripts/run_d0_minimal.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---- 可覆盖参数 ----------------------------------------------------------------
DATASET="${DATASET:-tv3-formal-6000}"
DATA_DIR="data/${DATASET}"
SEED="${SEED:-20260704}"
WORKERS="${WORKERS:-4}"
SEQUENCES="${SEQUENCES:-6000}"
TIMESTEPS="${TIMESTEPS:-512}"
DT_S="${DT_S:-0.5}"

# ---- 环境检查 ------------------------------------------------------------------
echo "==== D0 最小实验集 ===="
echo "Python: $(python --version 2>&1)"
echo "数据集: ${DATA_DIR}"
echo "workers: ${WORKERS}"

# ---- 1. 生成数据集（如不存在）---------------------------------------------------
if [ -f "${DATA_DIR}/manifest.json" ]; then
  echo "[SKIP] 数据集已存在。"
else
  echo "[GEN] 生成 ${DATASET} (${SEQUENCES} 序列, workers=${WORKERS})..."
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

# ---- 2. 运行 3 组必做实验 (CPU only, RidgeCV) ----------------------------------
for label_config in \
  "D0-oracle|configs/tv3_d0_oracle_ridge.json" \
  "D0-observed|configs/tv3_d0_observed_ridge.json" \
  "D0-tof-only|configs/tv3_d0_tof_only_ridge.json"
do
  LABEL="${label_config%%|*}"
  CONFIG="${label_config##*|}"
  echo ""
  echo "---- ${LABEL} ----"
  python -m tv3.pipeline.run_tv3_rocket_baseline --config "${CONFIG}"
  echo "[OK] ${LABEL}"
done

# ---- 3. 快速汇总 + 判读 --------------------------------------------------------
echo ""
echo "=============================================="
echo "  D0 汇总 (必做 3 组)"
echo "=============================================="

python - << 'PYEOF'
import json

for label, slug in [
    ("D0-oracle",   "oracle_ridge"),
    ("D0-observed", "observed_ridge"),
    ("D0-tof-only", "tof_only_ridge"),
]:
    m = json.loads(open(f"outputs/tv3_d0/{slug}/metrics.json"))
    print(f"\n{label}  (features={m['feature_count']}, alpha={m['diagnostics']['selected_alpha']})")
    for split in ["val", "test", "extrapolation"]:
        cm = m["evaluations"][split]["component_metrics"]
        print(f"  {split:14s}  CO2={cm['x_CO2']['r2']:.4f}  O2={cm['x_O2']['r2']:.4f}  N2={cm['x_N2']['r2']:.4f}")

oracle_o2   = json.loads(open("outputs/tv3_d0/oracle_ridge/metrics.json"))["evaluations"]["val"]["component_metrics"]["x_O2"]["r2"]
observed_o2 = json.loads(open("outputs/tv3_d0/observed_ridge/metrics.json"))["evaluations"]["val"]["component_metrics"]["x_O2"]["r2"]
gap = oracle_o2 - observed_o2
print(f"\n-----")
print(f"oracle val O2 R2  = {oracle_o2:.4f}")
print(f"observed val O2 R2 = {observed_o2:.4f}")
print(f"oracle 膨胀        = {gap:.4f}")
print()
if gap <= 0.03:
    print(">>> 判读: 几乎无 oracle 膨胀。D1/D4 可直接以 D0-observed 推进。")
elif gap <= 0.10:
    print(">>> 判读: oracle 有边际影响。以 D0-observed 为真实基线推进 D1。")
else:
    print(">>> 判读: oracle 膨胀 >0.10。优先推进 D2 TOF/phase 可微估计。")

# alpha bound check
for label, slug in [("D0-oracle", "oracle_ridge"), ("D0-observed", "observed_ridge"), ("D0-tof-only", "tof_only_ridge")]:
    a = json.loads(open(f"outputs/tv3_d0/{slug}/metrics.json"))["diagnostics"]["selected_alpha"]
    if a <= 0.0001:
        print(f">>> 注意: {label} selected_alpha 触底 0.0001，可能需要扩展 alpha 下限。")
        break
PYEOF

echo ""
echo "[DONE] 下一步: 根据 oracle 膨胀大小决定优先推进 D1 (物理序列 DL) 还是 D2 (TOF/phase 可微估计)。"
