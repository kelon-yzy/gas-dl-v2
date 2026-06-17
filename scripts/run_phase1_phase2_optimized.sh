#!/bin/bash
# PhaseWindowTCN Phase 1 诊断实验批量运行脚本
# 只运行 Phase 1；Phase 2 必须在 G1 判读后手动触发。
# 当前 OOM 修正版：batch_size=16, num_workers=2, epochs=80, patience=10

set -e  # 遇到错误立即退出

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

DATASET_DIR="data/wv4-formal-hitran-standard-6000"
OUTPUT_ROOT="outputs"
EXTRA_ARGS=("$@")

echo "========================================"
echo "PhaseWindowTCN Phase 1 诊断实验"
echo "========================================"
echo "入口: python -m pipeline.run_experiment"
echo "配置: batch=16, workers=2, epochs=80"
echo "说明: Phase 2 不会自动运行，需完成 G1 判读后手动触发"
echo "========================================"
echo ""

# ==========================================
# Phase 1: 诊断批次 (4 个实验)
# ==========================================
echo "[Phase 1] 诊断批次 - 损失函数对比"
echo "实验数: 4"
echo "预计时长: 1-2 小时"
echo "----------------------------------------"
echo ""

echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python -m pipeline.run_experiment \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation.json \
  --dataset-dir "$DATASET_DIR" \
  --output-root "$OUTPUT_ROOT" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "[Phase 1] 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "========================================"
echo "Phase 1 已完成，当前停在 G1 决策门"
echo ""
echo "结果位置: $OUTPUT_ROOT/runs/"
echo "  - $OUTPUT_ROOT/runs/phase_window_tcn_ablation/"
echo "汇总: $OUTPUT_ROOT/summary/phase_window_tcn_ablation_summary.csv"
echo "报告: $OUTPUT_ROOT/reports/phase_window_tcn_ablation.md"
echo ""
echo "请按 docs/PhaseWindowTCN实验执行与验收流程.md 的 G1 规则判读。"
echo "仅当 G1 判定进入 Phase 2 时，再手动运行："
echo "  bash scripts/run_phase1_phase2_commands.sh phase2-structure"
echo "  bash scripts/run_phase1_phase2_commands.sh phase2-followup"
echo ""
echo "========================================"
