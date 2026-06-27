# P1: TCN 容量扩张实验计划

**日期**: 2026-06-24
**优先级**: P1
**预期收益**: +0.05 R²
**实验成本**: 1天
**风险等级**: 低

---

## 实验目标

验证当前模型是否受限于参数容量不足，通过增加 TCN 通道数提升表达能力。

## 基线配置 (P0-B)

```json
{
  "tcn_channels": [64, 64, 64],
  "tcn_kernel_size": 3,
  "tcn_dropout": 0.25,
  "loss": {
    "name": "weighted_component_mse",
    "component_weights": [0.009596, 0.012900, 0.162064, 0.029976]
  }
}
```

**基线性能**: test overall R²=+0.4844
- H₂: +0.7758
- CH₄: +0.4285
- CO₂: +0.2653
- N₂: -0.0124

**参数量**: ~73K

---

## P1 配置变更

```json
{
  "tcn_channels": [128, 128, 128],
  "tcn_kernel_size": 3,
  "tcn_dropout": 0.30,
  "loss": {
    "name": "weighted_component_mse",
    "component_weights": [0.009596, 0.012900, 0.162064, 0.029976]
  }
}
```

**参数量**: ~290K (4倍于 baseline)
**变更说明**:
- TCN 通道数从 64 → 128 (每层)
- Dropout 从 0.25 → 0.30 (防止过拟合)
- 保持 P0-B 的温和权重配置

---

## 实施步骤

1. ✅ **创建配置文件**: `configs/experiment/dl_p1_tcn_capacity.json`
2. ⏳ **训练**: seed=20260623 × 80 epochs
   ```bash
   python src/pipeline/run_experiment.py \
       --config configs/experiment/dl_p1_tcn_capacity.json
   ```
3. ⏳ **评估**: 对比 P0-B baseline 的 R² 提升

---

## 验收标准

- ✅ **成功**: test overall R² ≥ 0.53 (至少 +0.05 相对于 P0-B 的 0.4844)
- ⚠️ **部分成功**: R² 在 0.50-0.53 之间
- ❌ **失败**: R² < 0.50 或出现严重过拟合 (train-val gap > 0.15)

---

## 风险与缓解

| 风险 | 概率 | 缓解措施 | 状态 |
|------|------|---------|------|
| 过拟合 | 中 | 增加 dropout 到 0.30 | ✅ 已应用 |
| 训练时间过长 | 低 | 参数量增加 4× 但 TCN 并行，实际增幅约 1.5× | - |
| 显存不足 | 低 | 当前 batch_size=16，若 OOM 可降至 12 或 8 | - |

---

## 预期结果

### 成功场景
- test overall R² ≥ 0.53
- CO₂ R² 进一步提升 (期望 > 0.30)
- N₂ R² 转正或接近 0
- train-val gap 保持在合理范围 (< 0.10)

### 失败场景
- R² 无提升或下降 → 说明瓶颈不在 TCN 容量
- 严重过拟合 → 需要更强的正则化或数据增强
- 训练不稳定 → 需要降低学习率或调整优化器

---

## 后续决策

### 若 P1 成功 (R² ≥ 0.53)
- 接受 P1 配置为新 baseline
- 继续 P2: TCN 感受野扩展
- 考虑 P3: 数据增强叠加

### 若 P1 部分成功 (0.50 ≤ R² < 0.53)
- Multi-seed 验证 (3 seeds)
- 若稳定则接受，否则回退 P0-B
- 尝试中等容量配置 (tcn_channels=[96, 96, 96])

### 若 P1 失败 (R² < 0.50)
- 放弃纯容量扩张路线
- 转向 P4: Multi-scale CNN (架构改进)
- 或承认当前数据量下 DL 无法超越 Ridge

---

## 实验记录

### Run 1: seed=20260623
- **状态**: ⏳ 待运行
- **配置文件**: `configs/experiment/dl_p1_tcn_capacity.json`
- **输出目录**: `outputs/runs/dl_p1_tcn_capacity/`
- **预计训练时间**: ~30-40 分钟 (80 epochs, batch_size=16)
- **注意**: 需要先 `git pull` 确保服务器代码是最新版本（包含 `output_mode` 参数支持，commit 1e2c27f）

### 性能指标
- **overall R²**: _待填写_
- **H₂ R²**: _待填写_
- **CH₄ R²**: _待填写_
- **CO₂ R²**: _待填写_
- **N₂ R²**: _待填写_
- **train-val gap**: _待填写_
- **参数量**: ~290K
- **训练时间**: _待填写_

---

## 参考

- 基线配置: `configs/experiment/dl_p0_mild_weights.json` (run: `cnn1d_tcn_fusion_raw4_p0b_ch4co2_mild`)
- 改进计划: `docs/improvement_plan.md` (P1 章节)
- Bai et al. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
