# 掘进通风 DL 训练适配方案

> 本文档定义掘进通风场景（CO₂/O₂/N₂）的深度学习训练策略，涵盖通道可辨识性分析、模型选型、Loss 选择、配置模板、实验矩阵与验收标准。
>
> 前置依赖：仿真链路（阶段 1–3）完成，`tv3-formal` 数据集可用。
> 仿真适配方案见 [adaptation_plan.md](adaptation_plan.md)，实验编排见 [experiment_roadmap.md](experiment_roadmap.md)。

## 1. 结论先行

推荐训练管线：

```
tv3-formal (600 seq / 512 steps; 受当前内存/磁盘约束从原 6000 调整)
    → TCN (primary) + CNN1D (secondary)
    → weighted_component_mse, weights=[1.0, 2.0, 1.0]
    → 5 模型 × 3 seeds 基线
    → 通道消融 + O₂ 可辨识性消融 + Loss 消融
    → 汇总 → 决定是否引入 O₂ 专用通道
```

四条核心判断：

1. **TCN 为主力模型**：syngas 基线中 TCN 与 Ridge 并列最优（R² ≈ 0.96），时序建模能力适合多模态信号融合。
2. **O₂ 是最弱组分**：无直接光学通道，仅声学（声速差 ~6.4%）和热导（差异 ~2%）间接推断，需要重点关注。
3. **Loss 加权 O₂**：O₂ 可观测信号弱且动态范围中等（18–21.2%），给予 2× 权重以补偿学习难度。
4. **不引入新模型结构**：先用现有 5 种模型/算法建立基线，评估四模态通道的极限能力。

## 2. 通道可辨识性分析

### 2.1 通道-组分可观测性矩阵

| 通道 | CO₂ 贡献 | O₂ 贡献 | N₂ 贡献 | 物理机制 |
|------|:--------:|:-------:|:-------:|----------|
| V_NDIR_CO2 | ★★★ | — | — | CO₂ ν₃ 4.26 μm 红外吸收 |
| V_NDIR_CH4 | — | — | — | 场景无 CH₄，通道仅含噪声 |
| V_TCS | ★ | ★★ | ★★ | 混合气热导率 λ_mix |
| T_C | ★ | ★ | ★ | 温度修正：声速、热导 |
| P_MPa | ★ | ★ | ★ | 压力修正：密度、声速 |
| H_RH | ★ | ★ | ★ | 湿度影响声学和热导 |
| L_m | ★ | ★ | ★ | 声程影响 TOF 测量精度 |
| piston_position_m | — | — | — | Phase schedule 状态标记 |
| 超声波形 | ★★ | ★★ | ★★ | c_mix = f(γ_mix, M_mix, T) + 衰减 α |
| 光纤麦克风波形 | ★ | ★ | ★ | 声学反射 / 相位 |

图例：★★★ 直接可观测 | ★★ 间接可观测 | ★ 修正/微弱贡献 | — 无贡献

### 2.2 O₂/N₂ 辨识性风险分析

O₂ 和 N₂ 的物理差异：

| 物理量 | O₂ | N₂ | 差异 | 可辨识性 |
|--------|------|------|------|----------|
| 摩尔质量 M (g/mol) | 32.00 | 28.01 | 14.3% | 影响声速 |
| 比热比 γ | 1.400 | 1.400 | 0% | 无区分能力 |
| 声速 c₀ (m/s, 300K) | ~330.6 | ~353.1 | 6.4% | 可测（超声） |
| 热导率 λ (mW/m·K, 300K) | ~26.3 | ~25.8 | ~2% | 边际（TCS） |

分析：

- **声速**是区分 O₂/N₂ 的主要物理通道。M_O₂ = 32 vs M_N₂ = 28，声速差约 22 m/s（6.4%），在 200 kHz 超声和 0.2–0.3 m 声程下可产生可测的 TOF 差异。
- **热导率**差异极小（~2%），TCS 通道提供的辨识力有限，但作为独立信息源可能对多通道融合有增量贡献。
- **γ 相同**意味着声速差异完全来自摩尔质量差异，没有额外的比热比信号。
- 总体评估：O₂/N₂ 辨识是物理上可行但边际困难的任务，模型需要从多通道弱信号中提取相关性。

### 2.3 CO₂ 优势分析

CO₂ 通过 V_NDIR_CO2 通道被直接观测，信噪比高。类比 hydrogen_ng 中 CH₄ 通过 V_NDIR_CH4 直接观测、syngas 中 CO₂ 通过 V_NDIR_CO2 直接观测，CO₂ 预测应是本场景中精度最高的组分。

### 2.4 N₂ 特殊性

N₂ 在空气中占比 73.8–82%，绝对值大但动态范围仅约 8 个百分点。

潜在影响：

- MSE loss 中 N₂ 的绝对误差可能数值上大于 O₂（因为基数大），但相对误差可能小
- 归一化策略（per-component 标准化）已由 scaler 在数据加载时处理
- Loss 权重 [1.0, 2.0, 1.0] 不需要对 N₂ 额外加权（N₂ 通过声学通道可观测，且占比高时信号稳定）

## 3. 模型选型

### 3.1 复用策略

所有 5 种现有模型/算法直接复用，不修改架构：

| 模型 | 类型 | 预期优势 | 预期劣势 |
|------|------|----------|----------|
| CNN1D | DL | 局部特征提取 | 长程依赖弱 |
| TCN | DL | 时序建模 + 因果卷积 | — |
| LSTM | DL | 序列记忆 | 训练速度慢 |
| PatchTST | DL | 长程注意力 | 序列短时可能 overhead |
| Ridge | ML | 线性基线 + 可解释 | 非线性建模弱 |

### 3.2 输入输出规格

```
输入:
  slow:      (T, 8)      # 8 慢通道
  ultrasonic: (T, 5000)  # 超声波形
  fiber_mic:  (T, 10000) # 光纤麦克风波形
  T = target_timesteps = 512

输出:
  predictions: (3,)  # [x_CO2, x_O2, x_N2]
  output_dim = 3
```

tv3 的 `cnn1d_tcn_fusion` 多模态配置必须使用 `output_mode="raw3"`、`out_dim=3`，走线性三输出。`gas_head` 在 out_dim=3 时表示 ALR/ILR 坐标头语义，在 tv3 下禁止；`raw4` 仅适用于四组分原始百分比场景。

### 3.3 不新增模型结构

理由：
- 第一阶段优先建立基线，量化现有通道能力
- syngas 基线已验证现有架构在多组分回归上的有效性
- 新结构（注意力通道融合、物理约束层等）属于阶段 Ⅲ 扩展

## 4. Loss 选择

### 4.1 推荐 Loss

`weighted_component_mse`，按组分加权：

| 方案 | CO₂ 权重 | O₂ 权重 | N₂ 权重 | 用途 |
|------|:--------:|:-------:|:-------:|------|
| 默认 | 1.0 | 1.0 | 1.0 | 等权对照 |
| **推荐** | **1.0** | **2.0** | **1.0** | O₂ 加权（补偿弱可观测性） |
| 激进 | 1.0 | 3.0 | 1.0 | O₂ 强加权（消融用） |

推荐权重 [1.0, 2.0, 1.0] 的理由：O₂ 的物理可观测信号最弱，给予 2× 权重引导模型在 O₂ 上投入更多学习能力。syngas 中 `weighted_component_mse` 已验证有效。

### 4.2 禁用 Loss

| Loss | 禁用原因 |
|------|----------|
| `compositional_mse` | 依赖闭包残差头，本场景不使用 |
| `ilr_mse` | 依赖 ILR 变换（sum=100% 闭包） |
| `free_component_mse` | 使用 N-1 自由组分，N₂ 变为残差 |
| `weighted_free_component_mse` | 同上 |

`validate_loss_composition_scheme()` 在 `composition_scheme = "tunnel_ventilation"` 时自动拒绝上述 loss。

`target_transform`（ILR/ALR）同样不可用，DL Trainer 与 ML baseline 入口在检测到 `composition_scheme="tunnel_ventilation"` 时都会拒绝。

### 4.3 sum_abs_error 监控

```
sum_abs_error = |x_CO2_pred + x_O2_pred + x_N2_pred - 100|
```

作为评估监控指标记录，不参与 loss 计算，不通过闭包头强制归零。conditional_metrics 固定按 `o2_bins` / `co2_bins` 分箱，避免因 `x_N2` 存在而误走 hydrogen_ng 的 `n2_bins`。

### 4.4 Loss 消融矩阵

| 组 | Loss | 权重 | 目的 |
|----|------|------|------|
| L1 | mse | — | 无权基线 |
| L2 | weighted_component_mse | [1, 1, 1] | 等权基线 |
| L3 | weighted_component_mse | [1, 2, 1] | O₂ 加权（推荐） |
| L4 | weighted_component_mse | [1, 3, 1] | O₂ 强加权 |
| L5 | smooth_l1 | — | 鲁棒性对比 |

## 5. 配置模板

### 5.1 tv3_baseline.json 示例

```json
{
  "experiment_name": "tv3_baseline",
  "data": {
    "dataset_path": "data/tv3-formal",
    "composition_scheme": "tunnel_ventilation",
    "target_timesteps": 512
  },
  "model": {
    "type": "cnn1d",
    "output_dim": 3,
    "slow_input_dim": 8
  },
  "training": {
    "loss": "weighted_component_mse",
    "loss_weights": [1.0, 2.0, 1.0],
    "epochs": 200,
    "batch_size": 32,
    "learning_rate": 0.001,
    "scheduler": "cosine"
  },
  "seed": 42
}
```

### 5.2 配置差异矩阵

| 参数 | baseline | tcn | lstm | patchtst | ridge |
|------|----------|-----|------|----------|-------|
| model.type | cnn1d | tcn | lstm | patchtst | ridge |
| training.loss | weighted_component_mse | 同左 | 同左 | 同左 | — |
| training.epochs | 200 | 200 | 200 | 200 | — |
| training.learning_rate | 0.001 | 0.001 | 0.001 | 0.0005 | — |

`tv3_tcn_multimodal.json` 是额外的方向 B 配置，模型为 `cnn1d_tcn_fusion`，输入 `slow,ultrasonic,fiber_mic`，输出头固定为 `output_mode="raw3"`、`out_dim=3`。

### 5.3 编排脚本

`scripts/run_tv3_baseline.py`：遍历 5 模型 × 3 seeds（固定 `42,123,456`），自动汇总到 `outputs/tv3_baseline/summary.json`。DL run 只要进程返回非零退出码就记录为 `status="fail"`；即使已有 `metrics.json`，也只作为诊断字段，不纳入成功 summary。

## 6. 实验矩阵

### 6.1 基线实验（阶段 Ⅰ-4）

5 模型 × 3 seeds = 15 runs：

| run_id | 模型 | seed | 预期 CO₂ R² | 预期 O₂ R² | 预期 N₂ R² |
|--------|------|-----:|:-----------:|:----------:|:----------:|
| tv3-cnn1d-s42 | CNN1D | 42 | ≥0.95 | ≥0.70 | ≥0.80 |
| tv3-cnn1d-s123 | CNN1D | 123 | ≥0.95 | ≥0.70 | ≥0.80 |
| tv3-cnn1d-s456 | CNN1D | 456 | ≥0.95 | ≥0.70 | ≥0.80 |
| tv3-tcn-s42 | TCN | 42 | ≥0.96 | ≥0.75 | ≥0.85 |
| tv3-tcn-s123 | TCN | 123 | ≥0.96 | ≥0.75 | ≥0.85 |
| tv3-tcn-s456 | TCN | 456 | ≥0.96 | ≥0.75 | ≥0.85 |
| tv3-lstm-s42 | LSTM | 42 | ≥0.93 | ≥0.65 | ≥0.78 |
| tv3-lstm-s123 | LSTM | 123 | ≥0.93 | ≥0.65 | ≥0.78 |
| tv3-lstm-s456 | LSTM | 456 | ≥0.93 | ≥0.65 | ≥0.78 |
| tv3-patchtst-s42 | PatchTST | 42 | ≥0.95 | ≥0.72 | ≥0.82 |
| tv3-patchtst-s123 | PatchTST | 123 | ≥0.95 | ≥0.72 | ≥0.82 |
| tv3-patchtst-s456 | PatchTST | 456 | ≥0.95 | ≥0.72 | ≥0.82 |
| tv3-ridge-s42 | Ridge | 42 | ≥0.90 | ≥0.60 | ≥0.75 |
| tv3-ridge-s123 | Ridge | 123 | ≥0.90 | ≥0.60 | ≥0.75 |
| tv3-ridge-s456 | Ridge | 456 | ≥0.90 | ≥0.60 | ≥0.75 |

预期 R² 为参考估计，基于 syngas 基线结果和 O₂ 可辨识性分析推断。

### 6.2 通道消融（阶段 Ⅱ-1）

基于基线最优模型（预期 TCN），固定 seed=42：

| 实验 ID | 移除通道 | 目的 |
|---------|----------|------|
| tv3-ab-full | 无（baseline） | 对照 |
| tv3-ab-no-tcs | V_TCS | 评估热导贡献 |
| tv3-ab-no-ch4 | V_NDIR_CH4 | 确认 CH₄ 通道无贡献 |
| tv3-ab-no-co2 | V_NDIR_CO2 | 确认 CO₂ NDIR 支配性 |
| tv3-ab-waveform-only | 移除全部慢通道 | 评估波形独立能力 |
| tv3-ab-slow-only | 移除全部波形 | 评估慢通道独立能力 |

### 6.3 O₂ 可辨识性消融（阶段 Ⅱ-2）

| 实验 ID | 输入 | 目的 |
|---------|------|------|
| tv3-o2-ultra-only | 仅超声波形 | 声速差异贡献 |
| tv3-o2-ultra-tcs | 超声 + V_TCS | 热导增量贡献 |
| tv3-o2-full | 全通道 | 对照 |
| tv3-o2-slow-only | 仅慢通道 | 慢通道独立贡献 |

### 6.4 Loss 消融（阶段 Ⅱ-3）

| 实验 ID | Loss | 权重 |
|---------|------|------|
| tv3-loss-mse | mse | — |
| tv3-loss-eq | weighted_component_mse | [1, 1, 1] |
| tv3-loss-o2x2 | weighted_component_mse | [1, 2, 1] |
| tv3-loss-o2x3 | weighted_component_mse | [1, 3, 1] |
| tv3-loss-sl1 | smooth_l1 | — |

## 7. 验收标准

### 7.1 数据验收

| 检查项 | 预期 |
|--------|------|
| `labels/y.npy` shape | `(N, 3)` |
| `manifest.composition_scheme` | `"tunnel_ventilation"` |
| `manifest.background_fields` | `[]` |
| `metadata/label_names.npy` | `["x_CO2", "x_O2", "x_N2"]` |
| 组分总量 | `|sum - 100| < 1e-6` |

### 7.2 DL 分阶段验收

#### 最低标准（必须通过）

| 组分 | R² | MAE (%) | 依据 |
|------|---:|--------:|------|
| CO₂ | ≥ 0.95 | ≤ 0.30 | 有 NDIR 直接通道 |
| O₂ | ≥ 0.70 | ≤ 0.50 | 无直接通道，物理可辨识性弱 |
| N₂ | ≥ 0.80 | ≤ 0.80 | 占比大但动态范围小 |

#### 强指标（期望达到）

| 组分 | R² | MAE (%) |
|------|---:|--------:|
| CO₂ | ≥ 0.98 | ≤ 0.15 |
| O₂ | ≥ 0.85 | ≤ 0.30 |
| N₂ | ≥ 0.90 | ≤ 0.50 |

O₂ 的验收标准有意低于 CO₂，反映物理可观测性的差异。如果 O₂ 无法达到最低标准，需要进入阶段 Ⅲ-1（O₂ 专用通道）。

### 7.3 分层验收

按通风状态分层评估（状态定义见 [sampling_design.md](sampling_design.md)）：

| 状态 | 关注组分 | 额外关注 |
|------|----------|----------|
| fresh_air | 全部 | 正常区间精度 |
| ventilation_decay | CO₂, O₂ | 趋势跟踪能力 |
| co2_accumulation | CO₂ | 高值区精度 |
| oxygen_depletion | O₂ | 低值区精度（安全关键） |

oxygen_depletion 状态下 O₂ 的精度尤其重要——这是实际场景中安全关键的区域。

### 7.4 回归验收

- 现有 `wv4-*` 和 `sg4-*` 测试不因新场景改变默认行为
- 新增代码不修改全局 `COMPONENT_FIELDS` 的含义
- `x_N2` 不被放入任何场景的 `background_fields`（syngas 场景的 N₂ 是 background，但那是 syngas 自己的定义）

## 8. 不推荐路线

1. **不使用闭包残差头预测 N₂**：N₂ 是直接输出目标，残差头（如 `GasHeadNormalize`）会把 N₂ 降级为 `100 - CO₂ - O₂`，丢失独立监督信号，且误差传播到 N₂ 的精度不可控。
2. **不引入 ILR/ALR target_transform**：这些变换假设闭包结构（sum=100%），与不使用闭包头的设计矛盾。
3. **不过早引入 O₂ 专用传感器**：先量化现有四模态通道的 O₂ 辨识极限。如果阶段 Ⅱ 结果不达标，阶段 Ⅲ-1 再引入。
4. **不删除 V_NDIR_CH4 通道**：虽然场景无 CH₄ 该通道仅含噪声，但保留它维持与 hg 场景的通道对齐，通过 ablation 确认无贡献即可。删除通道会改变 `slow_input_dim`，增加跨场景配置的复杂度。

## 9. 从 hg 相位统计方案迁移的技术路线

> 参考 [../DL相位统计稳定提取与保留方案.md](../DL相位统计稳定提取与保留方案.md)。
> hg 场景中 N₂（IR 惰性、弱可观测）的困境与本场景 O₂（同核双原子、无 NDIR、仅声学+TCS 间接推断）高度同构。以下技术按优先级筛选迁移。

### 9.1 问题同构性分析

| 维度 | hg 的 N₂ 问题 | 掘进通风的 O₂ 问题 |
|------|-------------|--------------|
| 直接观测通道 | 无（IR 惰性） | 无（同核双原子，无 IR） |
| 间接推断路径 | 声速 + TCS 热导 | 声速（差 6.4%）+ TCS 热导（差 ~2%） |
| 被压制风险 | V_NDIR_CH4/CO2 强信号压制 | V_NDIR_CO2 强信号压制 |
| Ridge 表现 | 多窗口统计后 N₂ R² 从 0.22 → 0.71 | 待验证 |
| DL 表现 | 端到端 N₂ R² ≈ 0 | 待验证 |

如果掘进通风基线中 O₂ R² 也远低于 Ridge 基线，说明同样的"弱组分信号被压制"问题存在，以下技术路线生效。

### 9.2 直接迁移的技术（阶段 Ⅱ 后可用）

#### T1. TCN Hidden Probe（对应方案 I，低成本诊断）

在基线训练完成后、任何结构改造前执行：

```
冻结基线 TCN → 导出 hidden/pooled features → 线性 probe
  probe → y_true（三组分）
  probe → per-modality features → y_true（定位哪个模态有/无信号）
```

判读：
- probe O₂ R² 高但 final O₂ R² 低 → 信息在融合/输出阶段丢失，修 fusion/head
- probe O₂ R² 也低 → TCN 前端未提取 O₂ 信号，需要结构增强（ROCKET 分支或新 backbone）
- per-modality probe：定位 V_TCS vs 超声波形对 O₂ 的各自贡献

成本：< 0.5 天，只训一层线性回归。

#### T2. 模态级辅助监督（对应方案 D）

为每个模态独立设辅助头，强制各模态保留可预测信息：

```
head_slow     → y (监督 3 组分)
head_ultrasonic → y
head_fiber    → y
head_final    → y (主头)

L = L_final + α · mean(L_slow, L_ultrasonic, L_fiber)
α ∈ {0.05, 0.1, 0.2}  sweep 或 GradNorm 自动平衡
```

目的：V_NDIR_CO2 的强信号通过 slow 通道直接拟合 CO₂，可能导致 ultrasonic/fiber_mic 分支的 O₂/N₂ 信号被梯度忽略。辅助头强制每个模态都保留对 O₂ 的预测能力。

#### T3. 平衡融合 + Modality Dropout（对应方案 E）

```
每个模态 encoder output
  → Linear projection (→ 64-d)
  → LayerNorm
  → modality token

tokens → gated fusion (sigmoid gate per token) → final representation
```

训练时附加 Modality Dropout：随机丢整个 slow/ultrasonic/fiber_mic 分支（p=0.1–0.2），防止 V_NDIR_CO2 所在的 slow 分支独占 CO₂ 预测。

### 9.3 条件迁移的技术（阶段 Ⅲ 备选）

以下技术仅在基线和阶段 Ⅱ ablation 无法达到 O₂ 验收标准时启用：

#### T4. 相位窗口统计分支（对应方案 A+C）

如果 Ridge 多窗口统计在掘进通风场景中也对 O₂ 有增益（exposure/recovery 阶段的声学/热导响应差异携带 O₂/N₂ 辨识信息），则：

1. 生成 `features/phase_stats.npy`（按 exposure/recovery/full 三窗口 × 三模态 × 统计量）
2. 建立 stats-only MLP 基线
3. 接入相位统计专属分支 + 辅助头

前提：需要先验证 Ridge multiwindow 在 tv3 数据上是否对 O₂ 有多窗口增益（类比 hg 中 N₂ 从 0.22 → 0.71）。如果 O₂ 的相位窗口增益不显著（O₂ 在空气中占比稳定，exposure/recovery 的 O₂ 变化可能不如 hg 中 N₂ 的相位变化明显），则此路线不适用。

#### T5. ROCKET/MultiRocket 统计池化分支（对应方案 F/J1）

```
ultrasonic waveform + first-order difference
  → fixed/random 1D kernels (seed 固定)
  → max / mean / std / PPV / slope pooling
  → rocket_features (降维到与 TCN embedding 同量级)

[rocket_features, TCN embedding, slow embedding]
  → balanced fusion → final head
```

适用场景：TCN probe 显示前端未提取 O₂/N₂ 差异信号。ROCKET 的多尺度卷积核 + 多种池化可能捕获 TCN 遗漏的声学模式（例如声速差异导致的 TOF 微小偏移模式）。

#### T6. Ridge Teacher 蒸馏（对应方案 G）

```
L = L_final + β(epoch) · MSE(y_final, y_ridge)
β(epoch) = β₀ · max(0, 1 - epoch / T_anneal)  # 退火
```

蒸馏权重退火：前期借 Ridge 引导收敛，后期切回纯 ground-truth，避免 DL 复制 Ridge 的 bias。仅在 Ridge 对 O₂ 有显著优势时使用。

### 9.4 不迁移的技术

| 技术 | 不迁移原因 |
|------|----------|
| gas_head 闭包残差头分析 | 本场景不使用闭包头，三组分全部直接输出 |
| ILR/ALR target_transform | 依赖 sum=100% 闭包结构 |
| N₂ 特定的 loss 权重策略 | 本场景 N₂ 是高占比组分，不是弱组分 |
| v6 L_m 塌缩对 N₂ 的影响分析 | 掘进通风直接在 v6 链路上构建，无历史链路迁移问题 |
| P-1 证据迁移验证 | 本场景从头开始，不存在旧链路证据需要迁移 |

## 10. 推荐执行顺序

```
P-1  生成 tv3-smoke（链路验证）                               ✅ 已完成
P-2  生成 tv3-formal（600 序列 / 512 时步）                   ✅ 已完成（由原 6000 规模调整，受内存/磁盘限制）
P-3  创建 5 个 tv3 配置 + 编排脚本                            ✅ 已完成
P-4  基线训练（5 模型 × 3 seeds = 15 runs）                   🔶 首轮 TCN+Ridge 完成；脚本 seeds/失败处理已修正，完整 15 runs 待决策
P-5  TCN hidden probe（§9.2 T1，低成本诊断，决定后续分叉）     ⏳ 待执行
P-6  通道消融（阶段 Ⅱ-1，6 runs）                             ⏳ 待执行
P-7  O₂ 可辨识性消融（阶段 Ⅱ-2，4 runs）                      ⏳ 待执行
P-8  Loss 消融（阶段 Ⅱ-3，5 runs）                            ⏳ 待执行
P-9  汇总分析 + probe 结果，决定是否进入阶段 Ⅲ                 ⏳ 待执行
     P-9a 若 probe 显示信息在融合阶段丢失 → T2 模态辅助头 + T3 平衡融合
     P-9b 若 probe 显示前端未提取 → T5 ROCKET 分支
     P-9c 若所有通道组合 O₂ R² < 0.50 → 阶段 Ⅲ-1 O₂ 专用通道  ⚠️ Ridge O₂ R²≈0 已触发，待加入波形模态验证
```

> **首轮基线触发 P-9c 预警**：Ridge (slow-only, 600 序列) O₂ R²=-0.05 < 0.50。但当前仅 slow-only 模态，未使用超声波形（O₂/N₂ 声速差的直接载体）。建议先加入波形模态（方向 B）重跑，确认 O₂ 是否真的不可辨识，再决定是否进入阶段 Ⅲ-1。

停止条件：

- 如果 O₂ R² < 0.50（across all models and all channel combinations）→ 当前通道组合无法有效检测 O₂，必须引入 O₂ 专用通道（阶段 Ⅲ-1）
- 如果 N₂ R² < 0.60 → 检查数据归一化策略和 loss 权重配置
- 如果 CO₂ R² < 0.90 → 异常，检查 V_NDIR_CO2 通道数据生成是否正确
- 如果 T2/T3 使 O₂ R² 提升 > 0.05 → 说明融合压制是主因，继续在融合层优化
- 如果 T5 ROCKET 分支 O₂ R² 仍 < 0.50 → 声学/TCS 物理可辨识性不足，不再在 DL 结构上投入，转向 O₂ 专用传感器

## 11. 文献依据

| 编号 | 主题 | 来源 |
|---:|---|---|
| [1] | 多模态气体传感融合方法 | Huang & Leung 2007, "Simultaneous Classification and Concentration Estimation for Electronic Nose", IEEE Sensors Journal, DOI:10.1109/jsen.2007.894906 |
| [2] | 超声气体组分分析（O₂/N₂ 声速差异） | Bates et al. 2012, "A combined ultrasonic flow meter and binary vapour mixture analyzer for the ATLAS silicon tracker", arXiv:1210.4835（超声 TOF 气体混合物分析方法学）；O₂/N₂ 声速差异数据见 [physics_references.md](physics_references.md) §2.1（NIST 物性） |
| [3] | 热导检测器在气体分析中的应用 | Mukhopadhyay, Das Gupta, Barua 1967, "Thermal conductivity of hydrogen-nitrogen and hydrogen-carbon-dioxide gas mixtures", British Journal of Applied Physics, DOI:10.1088/0508-3443/18/9/312 |
| [4] | 加权损失函数在多任务回归中的应用 | Kendall, Gal, Cipolla 2017, "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics", arXiv:1705.07115 |
| [5] | TCN 在时间序列回归中的表现 | Bai, Kolter, Koltun 2018, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", arXiv:1803.01271 |
| [6] | 矿山通风气体监测技术 | Muduli, Mishra, Jana 2019, "Wireless Sensor Network Based Underground Coal Mine Environmental Monitoring Using Machine Learning Approach", DOI:10.1007/978-981-13-1420-9_66 |
| [7] | 多模态不平衡与弱模态压制 | Wang et al. CVPR 2020; Peng et al. CVPR 2022 (OGM-GE) |
| [8] | 单模态 Teacher 蒸馏 | Du et al. ICML 2023 (UMT), arXiv 2305.01233 |
| [9] | GradNorm 辅助 loss 自动平衡 | Chen et al. ICML 2018, arXiv 1711.02257 |
| [10] | ROCKET/MultiRocket 时间序列统计池化 | Tan et al. 2022; Dempster et al. 2023 (Hydra) |
| [11] | iTransformer 变量中心化表征 | Liu et al. 2023, arXiv 2310.06625 |
| [12] | PatchTST patch Transformer | Nie et al. 2022, arXiv 2211.14730 |
| [13] | 深度监督网络 | Lee et al. AISTATS 2015 (Deeply-Supervised Nets) |
