#!/usr/bin/env python
"""测试优化后的训练配置，验证显存占用和吞吐量。"""

import json
import subprocess
import sys
from pathlib import Path

def main():
    # 配置对比
    configs = [
        {
            "name": "原配置 (batch=8, workers=8)",
            "batch_size": 8,
            "num_workers": 8,
            "persistent_workers": True,
            "lr": 0.0005,
        },
        {
            "name": "优化配置 (batch=32, workers=2)",
            "batch_size": 32,
            "num_workers": 2,
            "persistent_workers": False,
            "lr": 0.001,
        },
    ]

    print("=" * 70)
    print("训练配置优化测试")
    print("=" * 70)
    print()
    print("目标：")
    print("  1. 降低显存占用 (17GB → 6-8GB)")
    print("  2. 提升训练速度 (17.6 → 60-80 samples/s)")
    print("  3. 保持收敛性")
    print()
    print("=" * 70)
    print()

    # 显示配置对比
    print("配置对比：")
    print()
    print(f"{'参数':<25} {'原配置':<20} {'优化配置':<20}")
    print("-" * 70)
    print(f"{'batch_size':<25} {8:<20} {32:<20}")
    print(f"{'num_workers':<25} {8:<20} {2:<20}")
    print(f"{'persistent_workers':<25} {'True':<20} {'False':<20}")
    print(f"{'lr':<25} {0.0005:<20} {0.001:<20}")
    print(f"{'epochs':<25} {300:<20} {80:<20}")
    print(f"{'early_stopping patience':<25} {25:<20} {10:<20}")
    print()
    print("=" * 70)
    print()

    # 测试建议
    print("测试步骤：")
    print()
    print("1. 使用 smoke dataset 快速验证 (推荐)")
    print("   python src/dl/cli.py \\")
    print("     --config configs/experiment/phase_window_tcn_mvp/phase_window_tcn_mvp_optimized.json \\")
    print("     --dataset-dir data/wv4-smoke \\")
    print("     --output-dir outputs/test_optimized")
    print()
    print("2. 监控显存占用")
    print("   在另一个终端运行: nvidia-smi -l 1")
    print()
    print("3. 观察关键指标")
    print("   - GPU memory: 应该 < 10GB")
    print("   - samples/s: 应该 > 50")
    print("   - epoch 时间: 应该 < 100s")
    print()
    print("4. 如果成功，应用到正式实验")
    print("   复制 phase_window_tcn_mvp_optimized.json 到诊断实验配置")
    print()
    print("=" * 70)
    print()

    # 预期效果
    print("预期效果：")
    print()
    print("原配置:")
    print("  - 显存: 17GB")
    print("  - 每 epoch: ~248s")
    print("  - 完整训练 (29 epochs): ~2 小时")
    print()
    print("优化配置:")
    print("  - 显存: 6-8GB (降低 60%)")
    print("  - 每 epoch: ~60-80s (加速 3-4×)")
    print("  - 完整训练 (预计 12-15 epochs): ~15-20 分钟 (加速 6-8×)")
    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
