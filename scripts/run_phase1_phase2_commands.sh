#!/bin/bash
# PhaseWindowTCN 诊断与条件结构消融命令入口
# 用法：
#   bash scripts/run_phase1_phase2_commands.sh phase1
#   bash scripts/run_phase1_phase2_commands.sh phase2-structure
#   bash scripts/run_phase1_phase2_commands.sh phase2-followup
#
# Phase 2 只能在 G1 判读确认需要结构消融后手动运行。

set -e

export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}src"

DATASET_DIR="${DATASET_DIR:-data/wv4-formal-hitran-standard-6000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

usage() {
  echo "Usage: bash scripts/run_phase1_phase2_commands.sh <phase> [extra run_experiment args]"
  echo ""
  echo "Phases:"
  echo "  phase1            运行 Phase 1 诊断批次"
  echo "  phase2-structure  运行 Phase 2.1 结构消融（仅 G1 允许后）"
  echo "  phase2-followup   运行 Phase 2.2 followup（仅 G2 允许后）"
}

run_experiment() {
  local config_path="$1"
  shift
  python -m pipeline.run_experiment \
    --config "$config_path" \
    --dataset-dir "$DATASET_DIR" \
    --output-root "$OUTPUT_ROOT" \
    "$@"
}

phase="${1:-}"
if [ -z "$phase" ]; then
  usage
  exit 2
fi
shift

case "$phase" in
  phase1)
    run_experiment configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation.json "$@"
    ;;
  phase2-structure)
    echo "仅当 G1 判定进入 Phase 2 时运行。"
    run_experiment configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_structure.json "$@"
    ;;
  phase2-followup)
    echo "仅当 Phase 2 出现正信号且 G2 判定需要 followup 时运行。"
    run_experiment configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_followup.json "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
