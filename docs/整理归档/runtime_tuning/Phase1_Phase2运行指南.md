# Phase 1 & Phase 2 诊断实验运行指南

## 快速开始

## 目标服务器

| 资源 | 配置 |
|---|---|
| GPU | NVIDIA RTX5880, 48 GiB 显存 |
| CPU | AMD vCPU, 32 核 |
| 内存 | 64 GB |

### 方式 1：Phase 1 批量运行脚本（推荐）

```bash
# Linux/Mac
bash scripts/run_phase1_phase2_optimized.sh

# Windows Git Bash
bash scripts/run_phase1_phase2_optimized.sh

# Windows PowerShell (需要手动运行各命令，见下文)
```

该脚本只运行 Phase 1，并在完成后停在 G1 决策门；Phase 2 必须根据 G1 判读结果手动触发。
**监控**: 在另一个终端运行 `nvidia-smi -l 1` 观察显存和 GPU 利用率

---

## 方式 2：分步运行

### Phase 1: 诊断批次（4 个实验）

**目的**: 对比不同损失函数对 N2 预测的影响  
**预计时长**: 1-2 小时  
**实验配置**: free_component_mse, weighted_component_mse, weighted_free_component_mse, handcraft_mlp

```bash
PYTHONPATH=src python -m pipeline.run_experiment \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation.json \
  --dataset-dir data/wv4-formal-hitran-standard-6000 \
  --output-root outputs
```

**输出位置**: `outputs/runs/phase_window_tcn_ablation/`

**包含的实验**:

1. `phase_window_tcn_gas_free` - 自由组分 MSE 损失
2. `phase_window_tcn_gas_varweight` - 方差加权 MSE 损失
3. `phase_window_tcn_gas_free_varweight` - 方差加权自由组分 MSE
4. `phase_window_tcn_handcraft_mlp` - 手工特征 MLP 基线

---

### Phase 2.1: 结构消融 - 编码器共享 vs 深层 TCN（2 个实验）

**进入条件**: 仅在 G1 判定需要进入 Phase 2 后运行。
**目的**: 对比编码器共享策略和 TCN 深度的影响  
**预计时长**: 40-60 分钟

```bash
PYTHONPATH=src python -m pipeline.run_experiment \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_structure.json \
  --dataset-dir data/wv4-formal-hitran-standard-6000 \
  --output-root outputs
```

**输出位置**: `outputs/runs/phase_window_tcn_ablation_structure/`

**包含的实验**:

1. `phase_window_tcn_gas_free_split` - 分离 window 编码器（share_window_encoder=false）
2. `phase_window_tcn_gas_free_deep` - 深层 TCN（5 blocks vs 3 blocks）

---

### Phase 2.2: 结构消融 - 组合测试（1 个实验）

**进入条件**: 仅在 Phase 2.1 出现正信号且 G2 判定需要 followup 后运行。
**目的**: 测试分离编码器 + 深层 TCN 的组合效果  
**预计时长**: 20-30 分钟

```bash
PYTHONPATH=src python -m pipeline.run_experiment \
  --config configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_followup.json \
  --dataset-dir data/wv4-formal-hitran-standard-6000 \
  --output-root outputs
```

**输出位置**: `outputs/runs/phase_window_tcn_ablation_followup/`

**包含的实验**:

1. `phase_window_tcn_gas_free_split_deep` - 分离编码器 + 深层 TCN

---

## 优化配置总结

当前正式诊断批使用以下基线配置；更激进的 batch、worker、`torch.compile` 只按 `docs/训练配置优化方案.md` 做候选实测。

```json
{
  "epochs": 80,
  "batch_size": 16,
  "num_workers": 2,
  "pin_memory": true,
  "persistent_workers": false,
  "prefetch_factor": 2,
  "lr": 0.00015,
  "amp": {
    "enabled": true,
    "dtype": "float16"
  },
  "performance": {
    "cudnn_benchmark": true,
    "tf32": true,
    "compile": false
  },
  "early_stopping": {
    "patience": 10
  }
}
```

**对比原配置**:

- batch_size: 当前保持 16；48 GiB 服务器可候选测试 20/24/32
- num_workers: 8 → 2 (-75%)
- persistent_workers: true → false
- epochs: 300 → 80 (-73%)
- patience: 25 → 10 (-60%)
- lr: 0.0001 → 0.00015
- TF32/cuDNN benchmark: 已启用

---

## 监控与验证

### 实时监控

```bash
# GPU 显存和利用率
nvidia-smi -l 1

# 训练日志（另一个终端）
tail -f outputs/runs/phase_window_tcn_ablation/*/metrics_live.jsonl
```

### 预期指标

- **显存占用**: 以 `metrics_live.jsonl` 和 `nvidia-smi` 实测为准，不再使用旧的 6-8GB 估算
- **吞吐量**: 先记录当前基线，再按主方案比较 worker/compile/batch 候选
- **每 epoch**: 以 `epoch_seconds`、`train_seconds`、`val_seconds` 为准
- **收敛**: 看 `best_epoch`、`val_loss`、`test/extrapolation x_N2 R2`

### 关键验收

每个实验完成后检查：

1. `outputs/runs/<experiment_name>/<run_name>/metrics_live.jsonl` - 训练曲线
2. `outputs/runs/<experiment_name>/<run_name>/best_checkpoint.pt` - 最佳模型
3. `outputs/summary/phase_window_tcn_ablation_summary.csv` - 汇总结果（Phase 1 完成后）

---

## 故障排查

### 问题 1: 显存 OOM

**症状**: CUDA out of memory  
**解决**:

```bash
# 先回到正式基线
# 修改配置文件中的 "batch_size": 16

# 若疑似碎片化，再设置：
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### 问题 2: 训练过慢

**症状**: samples/s < 40  
**原因**: 可能数据加载成为瓶颈  
**解决**:

```bash
# 按 docs/训练配置优化方案.md 做 worker sweep:
# num_workers=0/2/4/8
```

### 问题 3: 收敛变差

**症状**: val_loss 比原配置高 > 10%  
**解决**:

```bash
# 若测试 batch=24/32 后收敛变差，回到基线：
# batch_size=16, lr=0.00015
```

### 问题 4: 过早停止

**症状**: best_epoch < 8  
**解决**:

```bash
# 增加 patience
# 修改配置文件中的 "patience": 10 → 15
```

---

## 结果分析

### Phase 1 完成后

查看损失函数对比：

```bash
python scripts/analyze_phase1_results.py \
  --summary outputs/summary/phase_window_tcn_ablation_summary.csv
```

关注指标：

- N2 R² (test, extrapolation)
- CH4, CO2, H2 MAE
- 训练时长

### Phase 2 完成后

对比结构变体：

```bash
python scripts/analyze_phase2_results.py \
  --structure outputs/summary/phase_window_tcn_ablation_structure_summary.csv \
  --followup outputs/summary/phase_window_tcn_ablation_followup_summary.csv
```

关注：

- 分离编码器是否提升性能
- 深层 TCN 是否改善收敛
- 组合方案是否有协同效应

---

## 时间线估算

时间线以当前 48 GiB 服务器实测为准。先记录 Phase 1 基线的 `epoch_seconds` 和 `train_samples_per_second`，再决定是否做候选加速。

| 阶段 | 实验数 | 执行条件 | 时间口径 |
|---|---:|---|---|
| Phase 1 诊断批次 | 4 | 默认执行 | 先实测 |
| Phase 2.1 结构消融 | 2 | 仅 G1 允许后 | 先实测 |
| Phase 2.2 组合测试 | 1 | 仅 G2 允许后 | 先实测 |

不要再使用旧的 `2-3.5 小时` 总时长估算作为验收标准。

---

## 完成后操作

### 1. 归档结果

```bash
# 打包实验输出
tar -czf phase1_phase2_results_$(date +%Y%m%d).tar.gz \
  outputs/runs/phase_window_tcn_ablation* \
  outputs/summary/phase_window_tcn_ablation*_summary.csv
```

### 2. 备份检查点

```bash
# 只保留最佳检查点，节省空间
find outputs/runs -name "checkpoint.pt" -delete
```

### 3. 生成报告

```bash
# 综合分析报告
python scripts/generate_ablation_report.py \
  --output docs/PhaseWindowTCN诊断与消融实验报告.md
```

---

**创建时间**: 2026-06-16  
**当前基线**: batch=16, workers=2, epochs=80
**加速候选**: 见 `docs/训练配置优化方案.md`
