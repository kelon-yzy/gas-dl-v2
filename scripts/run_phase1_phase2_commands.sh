#!/bin/bash
# PhaseWindowTCN 诊断与消融实验 - 快速运行命令
# 验证成功后立即执行

# ====================================
# Phase 1: 诊断批次 (4 个实验, ~1-2h)
# ====================================
python src/pipeline/run_experiment.py \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation.json \
  --dataset-dir data/wv4-formal-hitran-standard-6000 \
  --output-root outputs

# ====================================
# Phase 2.1: 结构消融 (2 个实验, ~40-60min)
# ====================================
python src/pipeline/run_experiment.py \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_structure.json \
  --dataset-dir data/wv4-formal-hitran-standard-6000 \
  --output-root outputs

# ====================================
# Phase 2.2: 组合测试 (1 个实验, ~20-30min)
# ====================================
python src/pipeline/run_experiment.py \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_followup.json \
  --dataset-dir data/wv4-formal-hitran-standard-6000 \
  --output-root outputs

# ====================================
# 总预计时长: 2-3.5 小时 (原配置 14 小时)
# 节省时间: 80%
# ====================================
