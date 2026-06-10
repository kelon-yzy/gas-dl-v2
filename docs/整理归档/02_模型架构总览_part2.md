> 复核说明：本页按当前代码与结果文件修正。深度学习融合策略里，只有 Early/Middle 有明确 YAML 入口；Late 目前主要体现在传统 ML 的二阶段融合。

## 5. 多模态融合策略详解

### 5.1 Early Fusion（早期融合）

**原理**：在特征提取层就将多模态信息融合

```
ultrasonic → CNN_encoder_1 ──┐
                              ├─→ Concat → Temporal_Head → Output
fiber_mic  → CNN_encoder_2 ──┤
                              │
slow       → MLP_encoder ─────┘
```

**特点**：
- ✅ 充分交互，模态间可以早期互补
- ✅ 参数效率高
- ❌ 不同模态特征空间差异大时可能冲突

**当前实现入口**：`configs/deep/slow_only_early_fusion_film_multimodal_formal.yaml`

### 5.2 Middle Fusion（中期融合）

**原理**：各模态独立编码到中间层，再通过专用 fusion layer 汇合

```
ultrasonic → CNN_encoder_1 → intermediate_1 ──┐
                                               │
fiber_mic  → CNN_encoder_2 → intermediate_2 ──┼─→ Fusion_Layer → Output
                                               │
slow       → MLP_encoder   → intermediate_3 ──┘
```

**特点**：
- ✅ 各模态保持独立语义空间
- ✅ fusion layer 可学习模态间交互
- ⚖️ 参数量适中

**当前实现入口**：`configs/deep/fusion_formal.yaml` 与 `configs/deep/slow_only_*_multimodal_formal.yaml`

### 5.3 Late Fusion（晚期融合）

**原理**：各模态完全独立预测，最后加权融合

```
ultrasonic → CNN_encoder_1 → TCN_1 → Output_1 ──┐
                                                 │
fiber_mic  → CNN_encoder_2 → TCN_2 → Output_2 ──┼─→ Weighted_Sum → Final_Output
                                                 │
slow       → MLP_encoder   → TCN_3 → Output_3 ──┘
```

**特点**：
- ✅ 模态完全解耦，可独立调试
- ✅ 可解释性强（知道每个模态的贡献）
- ❌ 参数量最大
- ❌ 无法捕获模态间底层交互

**当前实现状态**：仓库中没有独立的深度学习 `fusion_late.yaml`；若按“晚期融合”理解，当前最接近的是传统 ML 的 `TraditionalFusionModel` 将三模态预测再做 Ridge 融合

### 5.4 融合策略对比

| 维度 | Early | Middle | Late |
|------|-------|--------|------|
| 融合位置 | 编码器后 | 中间层 | 输出层 |
| 参数量 | 小 | 中 | 大 |
| 模态交互 | 强 | 中 | 弱 |
| 可解释性 | 弱 | 中 | 强 |
| 训练难度 | 低 | 中 | 高 |
| 适用场景 | 模态特征空间相似 | 通用 | 需要独立调试 |

## 6. 输出层设计

### 6.1 Bounded Simplex 参数化

**问题**：四组分浓度需要满足：
1. 非负约束：`x_i ≥ 0`
2. 和约束：`x_H2 + x_CH4 + x_CO2 + x_N2 = 100`

**解决方案**：bounded_simplex 参数化

```python
# 只显式预测 3 个比例 + 1 个总量
logits[3]  # H2, CH4, CO2 的 logits
total      # 前三项总量（0-100）

# 计算输出
p = softmax(logits)  # 归一化到 [0,1] 且和为 1
x_H2  = p[0] * total
x_CH4 = p[1] * total
x_CO2 = p[2] * total
x_N2  = 100 - total  # 残差推导
```

**优势**：
- ✅ 自动满足非负约束
- ✅ 自动满足和约束
- ✅ 只需预测 4 个参数（3 logits + 1 total）
- ✅ N2 作为"剩余气体"，符合物理直觉

### 6.2 3-Task Loss（当前仓库实际口径）

当前仓库里的 `3-task loss` 不是额外叠加多个辅助项，而是通过 `loss_columns: 3` 只对 `H2/CH4/CO2` 三个自由头计算主损失，`N2` 由 `100 - sum(first_three)` 推导，不再单独回传梯度。

**原因**：
- `derive_last: true` 下，`N2` 与前三项强耦合
- 继续对 `N2` 单独计算 loss，容易把 `CH4` 与 `N2` 的误差绑死在一起

**效果**：
- 去掉 `N2` 单独监督后，当前可见 `bounded_simplex_3task_loss_slow_branch` 比同系列 4-task 等权 MSE 更好
- 当前仓库中该配置的可见结果约为 `macro_RMSE=2.0185`

### 6.3 不确定性加权损失（UW Loss）

**原理**：为每个组分学习一个不确定性权重

```python
# 模型预测
pred[4], log_sigma[4] = model(input)

# 损失函数
Loss = Σ (1 / (2 * sigma_i^2)) * (pred_i - target_i)^2 + log(sigma_i)
```

**优势**：
- ✅ 自动平衡各组分的学习难度
- ✅ 低浓度组分（CO2, N2）可以有更大的容忍度
- ✅ 不需要手动调权重

**当前配置文件**：`configs/deep/bounded_simplex_3task_uw_slow_branch.yaml`

**当前观察**：
- 4-task UW 历史上出现过 `CH4/N2` 权重塌缩
- 当前仓库更推荐和 `loss_columns: 3` 一起使用，而不是直接做 4-task UW

## 7. 模型训练配置

### 7.1 通用超参数

| 参数 | 当前常见值 | 说明 |
|------|------------|------|
| epochs | 200 上限 | 实际常因早停提前结束 |
| seed | 42 | 主线随机种子 |
| device | cuda | GPU 训练 |
| num_workers | 0-4 | 以具体 YAML 为准，当前不少 run 为 0 |

### 7.2 优化器配置

| 参数 | 当前推荐值 | 说明 |
|------|--------------|------|
| optimizer | AdamW | 带权重衰减的 Adam |
| lr | 0.0002 | bounded_simplex 系列当前推荐学习率 |
| weight_decay | 1e-5 | 权重衰减系数 |
| betas | (0.9, 0.999) | Adam 动量参数 |

### 7.3 学习率调度

| 策略 | 参数 | 说明 |
|------|------|------|
| warmup | 15 epochs | 当前推荐值 |
| scheduler | cosine / plateau 两类并存 | 以具体 YAML 为准 |
| eta_min | 1e-5 | cosine 常见最小学习率 |
| sigma warmup freeze | 启用 | UW 情况下 warmup 期间冻结 sigma，避免权重膨胀 |

### 7.4 早停策略

| 参数 | 当前常见值 | 说明 |
|------|--------------|------|
| patience | 25-30 epochs | 等权常用 30，UW 常用 25 |
| metric | `val_macro_RMSE` | 监控指标 |
| mode | `min` | 越小越好 |
| min_delta | 0.0001 | 最小改善幅度 |

### 7.5 数据增强

| 方法 | 参数 | 说明 |
|------|------|------|
| 噪声注入 | std=0.01 | 训练时给 slow 加噪声 |
| 时间裁剪 | 无 | 保持完整 120 timesteps |
| Mixup | 可选 | 需在配置中启用 |

## 8. 模型评估指标

### 8.1 主要指标

| 指标 | 定义 | 说明 |
|------|------|------|
| **macro_RMSE** | `mean(RMSE_i)` | 四组分 RMSE 均值，**主指标** |
| **macro_MAE** | `mean(MAE_i)` | 四组分 MAE 均值 |
| **macro_MRE** | `mean(MRE_H2, MRE_CH4)` | 仅 H2/CH4 的 MRE 均值 |
| **per_component R²** | `R²_i` for i ∈ {H2, CH4, CO2, N2} | 各组分决定系数 |

### 8.2 辅助指标

| 指标 | 定义 | 说明 |
|------|------|------|
| **sum_error_mean** | `mean(|sum(pred) - 100|)` | 总和偏差均值 |
| **sum_error_max** | `max(|sum(pred) - 100|)` | 总和偏差最大值 |
| **macro_SMAPE** | 对称 MAPE | 低浓度组分的相对误差 |

### 8.3 训练监控指标

| 指标 | 说明 |
|------|------|
| train_loss | 训练集损失 |
| val_loss | 验证集损失 |
| val_macro_RMSE | 验证集 macro RMSE（早停监控）|
| val_macro_MAE | 验证集 macro MAE |
| learning_rate | 当前学习率 |

## 9. 模型性能对比（按当前仓库可见结果）

### 9.1 传统 ML vs 深度学习

| 结果来源 | 最佳模型 | macro_RMSE | 备注 |
|----------|----------|------------|------|
| **历史 `waveform_v3` 传统 ML** | `svr_ridge` / `v3_raw_tph` / `fused` | **0.6122** | 来自 `outputs/summary/results.tsv` |
| **`seedpath_formal` 30k 传统 ML** | `svr_ridge` / `v3_raw_tph` / `fused` | **2.4292** | 来自 `four_component_formal_seedpath_grid_summary.csv` |
| **`exp02_deep_e2e` 当前最佳可见** | `v3_tcn_multimodal_seed42` | **0.9792** | 当前 YAML 仍指向 `data/waveform_v3/` |
| **`exp03_full_training` 当前最佳可见** | `v3_waveform_only_seed42` | **0.6411** | 目录内 `summary.json`，未纳入 `results.tsv` |

**注意**：
- 当前仓库里的传统 ML 与深度学习结果并不都处在同一数据源口径上
- `results.tsv` 已刷新，但 `exp03_full_training` 结果目前仍需直接看各 run 目录
- 做正式结论时必须先区分 `waveform_v3` 与 `waveform_v3_seedpath_formal`

### 9.2 慢变量专用模型对比

| 模型 | 输入 | 当前最佳可见 macro_RMSE | 说明 |
|------|------|--------------------------|------|
| GRU | slow[8] | 暂无稳定结果入表 | 当前 `exp02` 未见顶层 `gru_slow` 结果 |
| LSTM | slow[8] | 0.7428（`exp03_full_training`） | `exp02` 历史旧结果较差 |
| TCN | slow[8] | 0.6861（`exp03_full_training`） | 当前 slow-only 中表现最好 |
| Transformer | slow[8] | 暂无稳定结果入表 | 配置存在，当前未见对应输出 |

## 10. 模型文件组织

### 10.1 代码结构

```
src/dl/models/
├── __init__.py
├── registry.py                          # 模型注册表
├── config_utils.py                      # 配置解析
├── acoustic_waveform_encoder.py         # 通道 1 编码器
├── multimodal_fusion_v3.py              # 双波形基础融合
├── multimodal_wrapper.py                # 通用融合包装器（6 个变体）
├── cnn1d_tcn_fusion.py                  # 原始 CNN1D-TCN
├── cnn1d_tcn_fusion_slow_branch.py      # 当前重点结构性消融版本
├── lstm.py                              # LSTM backbone
├── gru.py                               # GRU backbone
├── tcn.py                               # TCN backbone
├── cnn1d.py                             # CNN1D backbone
├── cnn_lstm.py                          # CNN-LSTM hybrid
├── transformer_encoder.py               # Transformer backbone
└── branch_fusion.py                     # BranchFusion 架构
```

### 10.2 配置文件组织

```
configs/deep/
├── slow_only_gru_formal.yaml                    # 慢变量专用
├── slow_only_lstm_formal.yaml
├── slow_only_tcn_formal.yaml
├── slow_only_transformer_formal.yaml
├── slow_only_branch_fusion_formal.yaml
├── slow_only_gru_multimodal_formal.yaml         # 全融合（6 个）
├── slow_only_lstm_multimodal_formal.yaml
├── slow_only_tcn_multimodal_formal.yaml
├── slow_only_transformer_multimodal_formal.yaml
├── slow_only_cnn1d_multimodal_formal.yaml
├── slow_only_cnn_lstm_multimodal_formal.yaml
├── fusion_formal.yaml                           # 纯双波形
├── bounded_simplex_3task_loss_slow_branch.yaml  # ✅ 主力配置
└── bounded_simplex_3task_uw_slow_branch.yaml    # UW Loss 版本
```

## 11. 参考文档

| 文档 | 内容 |
|------|------|
| `docs/CNN1D-TCN当前模型说明.md` | CNN1D-TCN 详细说明 |
| `docs/cnn1d融合模型架构.md` | CNN1D 融合模型架构 |
| `docs/早期融合_Early_Fusion_完整实验方案.md` | Early Fusion 方案 |
| `docs/中期融合（Intermediate Fusion）完整方案.md` | Middle Fusion 方案 |
| `docs/晚期融合（Late Fusion）完整方案.md` | Late Fusion 方案 |
| `docs/新项目目标架构说明.md` | 项目架构总体说明 |
