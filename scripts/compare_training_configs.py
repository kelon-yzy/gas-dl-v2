"""对比原配置和优化配置的关键参数与预期效果。"""

import json
from pathlib import Path


def load_config(path: Path) -> dict:
    """加载配置文件。"""
    with open(path) as f:
        return json.load(f)


def compare_configs():
    """对比两个配置。"""
    base_dir = Path("configs/experiment/phase_window_tcn_mvp")
    original = load_config(base_dir / "phase_window_tcn_mvp.json")
    optimized = load_config(base_dir / "phase_window_tcn_mvp_optimized.json")

    print("=" * 80)
    print("训练配置对比：原配置 vs 优化配置")
    print("=" * 80)
    print()

    # 训练参数对比
    params = [
        ("实验名称", "experiment_name"),
        ("最大 Epochs", "training.epochs"),
        ("Batch Size", "training.batch_size"),
        ("Num Workers", "training.num_workers"),
        ("Pin Memory", "training.pin_memory"),
        ("Persistent Workers", "training.persistent_workers"),
        ("Prefetch Factor", "training.prefetch_factor"),
        ("学习率", "training.lr"),
        ("Weight Decay", "training.weight_decay"),
        ("Early Stop Patience", "training.early_stopping.patience"),
    ]

    print(f"{'参数':<30} {'原配置':<25} {'优化配置':<25}")
    print("-" * 80)

    for name, path in params:
        orig_val = original
        opt_val = optimized
        for key in path.split("."):
            orig_val = orig_val.get(key, "N/A")
            opt_val = opt_val.get(key, "N/A")

        # 高亮变化
        marker = " *" if orig_val != opt_val else ""
        print(f"{name:<30} {str(orig_val):<25} {str(opt_val):<25}{marker}")

    print()
    print("=" * 80)
    print()

    # 预期效果
    print("预期效果分析：")
    print()

    # 计算理论加速比
    orig_batch = original["training"]["batch_size"]
    opt_batch = optimized["training"]["batch_size"]
    batch_speedup = opt_batch / orig_batch

    orig_epochs = original["training"]["epochs"]
    opt_epochs = optimized["training"]["epochs"]

    orig_patience = original["training"]["early_stopping"]["patience"]
    opt_patience = optimized["training"]["early_stopping"]["patience"]

    print(f"1. GPU 利用率")
    print(f"   Batch size 提升: {orig_batch} → {opt_batch} ({batch_speedup:.1f}×)")
    print(f"   理论吞吐量: 17.6 samples/s → {17.6 * batch_speedup * 0.7:.1f} samples/s (估算)")
    print()

    print(f"2. 显存占用")
    print(f"   Num workers: {original['training']['num_workers']} → {optimized['training']['num_workers']}")
    print(f"   Persistent workers: {original['training']['persistent_workers']} → {optimized['training']['persistent_workers']}")
    print(f"   预期显存: 17GB → 6-8GB (降低 ~60%)")
    print()

    print(f"3. 训练时长")
    print(f"   每 epoch 预期: 248s → 60-80s (加速 3-4×)")
    print(f"   Early stop patience: {orig_patience} → {opt_patience}")
    print(f"   MVP best_epoch=4, 预计停止: ~12-15 epochs")
    print(f"   完整训练: ~2 小时 → ~15-25 分钟 (加速 5-8×)")
    print()

    print(f"4. Phase 1 诊断批次 (3 个实验)")
    print(f"   原配置: ~6 小时")
    print(f"   优化后: ~1-1.5 小时")
    print()

    print("=" * 80)
    print()

    # 测试命令
    print("快速测试命令 (使用 smoke dataset):")
    print()
    print("python src/dl/cli.py \\")
    print("  --config configs/experiment/phase_window_tcn_mvp/phase_window_tcn_mvp_optimized.json \\")
    print("  --dataset-dir data/wv4-smoke \\")
    print("  --output-dir outputs/test_optimized")
    print()
    print("监控命令 (另一个终端):")
    print("nvidia-smi -l 1")
    print()
    print("=" * 80)


if __name__ == "__main__":
    compare_configs()
