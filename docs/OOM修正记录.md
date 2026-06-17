# OOM 修正记录

## 状态

本文档记录旧 24 GiB 环境的 OOM 事实和回退处理。当前 48 GiB 服务器上的训练加速方案见 `docs/训练配置优化方案.md`。

## 当前目标服务器

| 资源 | 配置 |
|---|---|
| GPU | NVIDIA RTX5880, 48 GiB 显存 |
| CPU | AMD vCPU, 32 核 |
| 内存 | 64 GB |

## 旧环境 OOM 事实

旧服务器运行 Phase 1 诊断实验时遇到：

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 250.00 MiB.
GPU 0 has a total capacity of 23.76 GiB of which 95.19 MiB is free.
```

当时显存状态：

- 总容量：23.76 GiB
- 已使用：22.65 GiB
- PyTorch 分配：22.08 GiB
- 空闲：95 MiB

该事实只说明 `batch_size=32` 不适合旧 24 GiB 环境，不能直接外推到当前 48 GiB 服务器。

## 已完成回退

当时将 PhaseWindowTCN 相关正式诊断配置回退为：

| 参数 | 回退值 |
|---|---:|
| `batch_size` | 16 |
| `lr` | 0.00015 |
| `num_workers` | 2 |
| `persistent_workers` | false |
| `epochs` | 80 |
| `early_stopping.patience` | 10 |

已覆盖：

- `configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation.json`
- `configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_structure.json`
- `configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_followup.json`
- `configs/experiment/phase_window_tcn_improvement/phase_window_tcn_improvement.json`

## 新服务器处理方式

新服务器是 NVIDIA RTX5880 48 GiB，可以重新测试更大 batch，但必须按候选方案验收：

- 先跑 `batch_size=16` 基线。
- 再按 `20 -> 24 -> 32` 上探。
- 每一步记录 `gpu_memory_allocated_mb`、`gpu_memory_reserved_mb`、`train_samples_per_second`、`val_loss`、`test/extrapolation x_N2 R2`。
- 只有速度提升且收敛不变差，才更新正式配置。

## OOM 时的临时措施

如果新服务器仍 OOM：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

PowerShell：

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

该环境变量只作为碎片化/OOM 缓解，不计入加速收益。

---

**更新日期**: 2026-06-16
