#!/bin/bash
# PhaseWindowTCN 诊断与消融实验批量运行脚本
# 优化配置：batch_size=32, num_workers=2, epochs=80, patience=10
# 预计总时长：2.5-3.5 小时（优化前 14 小时）

set -e  # 遇到错误立即退出

DATASET_DIR="data/wv4-formal-hitran-standard-6000"
OUTPUT_ROOT="outputs"

echo "========================================"
echo "PhaseWindowTCN 诊断与消融实验"
echo "========================================"
echo "优化配置: batch=32, workers=2, epochs=80"
echo "预计总时长: 2.5-3.5 小时"
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

python src/pipeline/run_experiment.py \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation.json \
  --dataset-dir "$DATASET_DIR" \
  --output-root "$OUTPUT_ROOT"

echo ""
echo "[Phase 1] 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "========================================"
echo ""

# ==========================================
# Phase 2: 结构消融 (3 个实验)
# ==========================================
echo "[Phase 2] 结构消融实验"
echo "实验数: 3"
echo "预计时长: 1-1.5 小时"
echo "----------------------------------------"
echo ""

# Phase 2.1: 分离编码器 vs 深层 TCN
echo "[Phase 2.1] 分离编码器 vs 深层 TCN (2 runs)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python src/pipeline/run_experiment.py \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_structure.json \
  --dataset-dir "$DATASET_DIR" \
  --output-root "$OUTPUT_ROOT"

echo ""
echo "[Phase 2.1] 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Phase 2.2: 组合消融（分离 + 深层）
echo "[Phase 2.2] 组合消融 - 分离编码器 + 深层 TCN (1 run)"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python src/pipeline/run_experiment.py \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_followup.json \
  --dataset-dir "$DATASET_DIR" \
  --output-root "$OUTPUT_ROOT"

echo ""
echo "[Phase 2.2] 完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ==========================================
# 完成总结
# ==========================================
echo "========================================"
echo "所有实验完成"
echo "========================================"
echo "总完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "结果位置: $OUTPUT_ROOT/runs/"
echo ""
echo "Phase 1 输出目录:"
echo "  - $OUTPUT_ROOT/runs/phase_window_tcn_ablation/"
echo ""
echo "Phase 2 输出目录:"
echo "  - $OUTPUT_ROOT/runs/phase_window_tcn_ablation_structure/"
echo "  - $OUTPUT_ROOT/runs/phase_window_tcn_ablation_followup/"
echo ""
echo "========================================"
