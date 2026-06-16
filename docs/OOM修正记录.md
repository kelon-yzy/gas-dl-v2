# OOM 修正记录

## 问题

在服务器上运行 Phase 1 诊断实验时遇到 OOM：

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 250.00 MiB. 
GPU 0 has a total capacity of 23.76 GiB of which 95.19 MiB is free.
```

**显存情况**：
- 总容量: 23.76 GB
- 已使用: 22.65 GB
- PyTorch 分配: 22.08 GB
- 空闲: 95 MB

## 修正方案

将 batch_size 从 32 降回 16：

| 参数 | 原优化值 | 修正值 | 说明 |
|------|---------|--------|------|
| `batch_size` | 32 | **16** | 避免 OOM |
| `lr` | 0.0002 | **0.00015** | 随 batch size 调整 |
| `num_workers` | 2 | 2 | 保持 |
| `persistent_workers` | false | false | 保持 |
| `epochs` | 80 | 80 | 保持 |
| `patience` | 10 | 10 | 保持 |

## 应用范围

已修正以下所有配置文件：
- ✅ `configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation.json`
- ✅ `configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_structure.json`
- ✅ `configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_followup.json`
- ✅ `configs/experiment/phase_window_tcn_improvement/phase_window_tcn_improvement.json`

## 优化效果（相比原始配置）

虽然 batch_size 未能提升到 32，但仍保留了其他优化：

| 优化项 | 原值 | 修正后 | 效果 |
|--------|------|--------|------|
| num_workers | 8 | 2 | 减少多进程显存开销 |
| persistent_workers | true | false | 释放常驻显存 |
| epochs | 300 | 80 | 减少无效训练 |
| patience | 25 | 10 | 更激进早停 |

**预期效果**：
- 显存占用：可运行（原配置也是 batch=16）
- 训练速度：2-3× 加速（主要来自 workers 减少 + epochs 减少 + 早停优化）
- Phase 1+2 总时长：14 小时 → **5-7 小时**（虽不及 batch=32 的 2.5-3.5h，但仍显著优化）

## 根本原因

服务器环境与本地测试环境不同：
1. **其他进程占用**：服务器上可能有其他任务占用了部分显存
2. **CUDA 版本差异**：不同 CUDA/PyTorch 版本的显存管理策略不同
3. **显存碎片化**：长时间运行后显存碎片化更严重

## 下一步

### 1. 立即重试

配置已修正，可以立即重新运行：

```bash
# Phase 1 诊断批次
python src/pipeline/run_experiment.py \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation.json \
  --dataset-dir data/wv4-formal-hitran-standard-6000 \
  --output-root outputs
```

### 2. 可选：进一步优化显存

如果 batch=16 仍然 OOM，可尝试：

```bash
# 设置环境变量启用可扩展段
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 或者进一步降低 batch_size 到 12
```

### 3. 监控显存

```bash
# 另一个终端监控
nvidia-smi -l 1

# 观察:
# - 显存峰值应 < 20GB
# - 训练过程中显存应稳定
```

## 验收标准

- ✓ 无 OOM 错误
- ✓ 训练正常完成
- ✓ 收敛性与原配置相当（batch=16 相同）
- ✓ 总时长 < 8 小时（相比原 14 小时仍有改善）

---

**创建时间**: 2026-06-16  
**修正版本**: batch=16, workers=2, epochs=80  
**Git commit**: 5d02532
