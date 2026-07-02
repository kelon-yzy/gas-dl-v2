# DL 最佳模型算法框架

**模型名称**: CNN1DTCNFusionRegressor (P0-B 配置)

**当前性能**: Overall R²=+0.4844 (test)

**最后更新**: 2026-06-24

---

## 目录

- [一、输入数据结构](#一输入数据结构)
- [二、模型架构层次](#二模型架构层次)
- [三、训练策略](#三训练策略)
- [四、核心设计原理](#四核心设计原理)
- [五、已证实的失败方向](#五已证实的失败方向)
- [六、当前性能与瓶颈](#六当前性能与瓶颈)
- [七、可探索改进方向](#七可探索改进方向)

---

## 一、输入数据结构

### 输入格式

- **张量形状**: `(Batch, Timesteps, Channels)` — NTC 格式
- **总通道数**: 3008
- **时间步范围**: 50-150 步（取决于相位窗口）

### 通道组成

| 模态                | 通道数  | 索引范围      | 说明                                   |
| ----------------- | ---- | --------- | ------------------------------------ |
| **slow 变量**       | 8    | 0:8       | 温度 T_C、压力 P_MPa、湿度 H_RH、管长 L_m、活塞位置等 |
| **ultrasonic 波形** | 1000 | 8:1008    | 超声传感器时间序列，int16 原始采样                 |
| **fiber_mic 波形**  | 2000 | 1008:3008 | 光纤麦克风时间序列，int16 原始采样                 |

### 目标输出

- **4维气体组分**: `[H₂, CH₄, CO₂, N₂]` 百分比 (%)
- **约束**: sum(components) = 100%

---

## 二、模型架构层次

### 架构总览

```
输入 (B, T, 3008)
    ↓
┌───────────────────────────────────────────────────────────┐
│  第一阶段: 多模态特征编码                                     │
├───────────────────────────────────────────────────────────┤
│  slow (8-d) → SlowFeatureEncoder → (B, T, 64)             │
│  ultrasonic (1000-d) → DeepAcousticEncoder1D → (B, T, 64) │
│  fiber_mic (2000-d) → DeepAcousticEncoder1D → (B, T, 64)  │
└───────────────────────────────────────────────────────────┘
    ↓ concat
  (B, T, 192)
    ↓
┌───────────────────────────────────────────────────────────┐
│  第二阶段: 时序融合建模 (TCN)                                │
├───────────────────────────────────────────────────────────┤
│  NTC → NCT 转置                                            │
│  3× TemporalBlock [64, 64, 64]                            │
│  - dilation: 1 → 2 → 4 (指数增长)                          │
│  - kernel_size: 3                                          │
│  - receptive_field: 29 步                                  │
│  - 残差连接 + BatchNorm + ReLU + Dropout(0.25)             │
└───────────────────────────────────────────────────────────┘
    ↓ NCT → NTC
  (B, T, 64)
    ↓
┌───────────────────────────────────────────────────────────┐
│  多尺度池化                                                 │
├───────────────────────────────────────────────────────────┤
│  last_state  = feats[:, -1, :]        # (B, 64)           │
│  mean_state  = feats.mean(dim=1)      # (B, 64)           │
│  max_state   = feats.max(dim=1)[0]    # (B, 64)           │
│  concat → (B, 192)                                        │
└───────────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────────────────────────┐
│  第四阶段: 回归头                                           │
├───────────────────────────────────────────────────────────┤
│  Shared MLP:                                              │
│    Linear(192, 128) → ReLU → Dropout(0.25)               │
│    Linear(128, 64) → ReLU → Dropout(0.25)                │
│                                                           │
│  GasHeadNormalize (output_mode="gas_head"):              │
│    Linear(64, 4) → 特殊约束变换                            │
│    - 前3维 → Softmax → 自由组分比例                         │
│    - 第4维 → Sigmoid → 自由组分总和 (0-100%)                │
│    - N₂ = 100 - sum(H₂, CH₄, CO₂)                        │
└───────────────────────────────────────────────────────────┘
    ↓
输出 (B, 4) — [H₂%, CH₄%, CO₂%, N₂%]
```

---

## 三、第一阶段：多模态特征编码

### 3.1 声学波形编码器 (DeepAcousticEncoder1D)

**用途**: 独立编码 ultrasonic 和 fiber_mic 波形

#### 输入预处理

```python
# int16 原始波形 → 归一化
waveform_normalized = waveform.float() / 32767.0
```

#### 1D CNN 特征提取

| 层     | 输入通道 | 输出通道 | Kernel | Stride | 作用       |
| ----- | ---- | ---- | ------ | ------ | -------- |
| Conv1 | 1    | 16   | 7      | 2      | 时间下采样 2× |
| Conv2 | 16   | 32   | 7      | 2      | 时间下采样 2× |
| Conv3 | 32   | 64   | 7      | 2      | 时间下采样 2× |
| Conv4 | 64   | 64   | 7      | 1      | 特征精炼     |

**每层结构**:

```
Conv1d (kernel=7, padding=3) → BatchNorm1d → ReLU → Dropout(0.15)
```

#### 全局统计池化

```python
# 从卷积特征图提取三种统计量
avg_pool = AdaptiveAvgPool1d(1)        # 全局平均
max_pool = AdaptiveMaxPool1d(1)        # 全局最大值
log_amplitude = log1p(|waveform|.mean())  # 对数幅度

# 拼接 + 投影
features = concat(avg_pool, max_pool, log_amplitude)  # (64+64+1)
embedding = Linear(129, 64)(features)  # → 64-d
```

**输出**: 每个时间步 64 维嵌入

---

### 3.2 慢变量编码器 (SlowFeatureEncoder)

**用途**: 编码温度、压力、湿度等慢变物理量

#### 结构

```python
slow_encoder = Sequential(
    Linear(8, 32),
    GELU(),
    Linear(32, 64),
)
```

**输出**: 每个时间步 64 维嵌入

---

### 3.3 多模态融合

```python
# 拼接三个模态的嵌入
fused_features = concat([
    ultrasonic_embedding,  # (B, T, 64)
    fiber_mic_embedding,   # (B, T, 64)
    slow_embedding,        # (B, T, 64)
], dim=-1)  # → (B, T, 192)
```

---

## 四、第二阶段：时序融合建模 (TCN)

### 4.1 时态卷积网络 (Temporal Convolutional Network)

#### TemporalBlock 结构

```python
class TemporalBlock(nn.Module):
    """单个 TCN 块，包含两层因果卷积 + 残差连接"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        self.net = Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            BatchNorm1d(out_channels),
            ReLU(),
            Dropout(0.25),

            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            BatchNorm1d(out_channels),
        )

        # 残差投影 (如果维度不匹配)
        self.residual = Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return ReLU(self.net(x) + self.residual(x))
```

#### CausalConv1d 实现

```python
class CausalConv1d(nn.Module):
    """因果卷积：保证时间步 t 只依赖 t 及之前的信息"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        self.padding = (kernel_size - 1) * dilation  # 左侧 padding
        self.conv = Conv1d(in_channels, out_channels, kernel_size, 
                          padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        # 移除右侧 padding，保持因果性
        return out[:, :, :-self.padding] if self.padding > 0 else out
```

### 4.2 TCN 配置

| 参数              | 值            | 说明                     |
| --------------- | ------------ | ---------------------- |
| **层数**          | 3            | 3 个 TemporalBlock      |
| **通道数**         | [64, 64, 64] | 每层 64 通道               |
| **Kernel size** | 3            | 卷积核大小                  |
| **Dilation**    | [1, 2, 4]    | 指数增长的膨胀率               |
| **感受野**         | 29 步         | 1 + 2×(k-1)×Σdilations |
| **Dropout**     | 0.25         | 正则化                    |

#### 感受野计算

```python
receptive_field = 1 + sum(2 * (kernel_size - 1) * d for d in dilations)
                = 1 + 2 * (3 - 1) * (1 + 2 + 4)
                = 1 + 2 * 2 * 7
                = 29 步
```

### 4.3 多尺度时间池化

```python
# TCN 输出: (B, 64, T)
tcn_features = tcn(fused_features.transpose(1, 2))

# 三种池化策略融合
last_state = tcn_features[:, :, -1]      # 最终时刻状态 (B, 64)
mean_state = tcn_features.mean(dim=-1)   # 全局平均 (B, 64)
max_state = tcn_features.amax(dim=-1)    # 全局峰值 (B, 64)

pooled_features = concat([last_state, mean_state, max_state], dim=-1)
# → (B, 192)
```

**设计原理**:

- **last_state**: 捕获最终时刻的隐状态
- **mean_state**: 全局稳态信息
- **max_state**: 瞬时峰值事件

---

## 五、第三阶段：相位统计特征 (已废弃)

### PhaseStatMLP (P3-A 失败方向)

```python
# 尝试接入 Ridge 手工统计特征
phase_stat_mlp = Sequential(
    Linear(420, 128),  # 420-d 统计特征
    ReLU(),
    Dropout(0.25),
    Linear(128, 64),
    ReLU(),
    Dropout(0.25),
)

# 与 TCN 池化特征拼接
final_features = concat([pooled_features, phase_stat_features], dim=-1)
# → (B, 256)
```

**失败原因**:

- 接入后性能从 R²=+0.493 → -0.085 (退化 2.7×)
- 420 维统计特征引入噪声
- 模型容量不足以有效融合 DL + ML 特征

**结论**: **当前最佳配置不使用此分支** (phase_stat_dim=0)

---

## 六、第四阶段：回归头

### 6.1 共享全连接层

```python
shared_head = Sequential(
    Linear(192, 128),  # 或 256 (如果包含 phase_stat)
    ReLU(),
    Dropout(0.25),

    Linear(128, 64),
    ReLU(),
    Dropout(0.25),
)
```

### 6.2 输出头：GasHeadNormalize

**P0-B 最佳配置使用的输出头**

#### 核心约束机制

```python
class GasHeadNormalize(nn.Module):
    """强制满足 sum(components) = 100% 的闭包约束输出头"""

    def __init__(self, in_features=64, output_prior=[9.29, 75.76, 4.99, 9.96]):
        self.linear = Linear(in_features, 4)
        self._init_prior(output_prior)

    def forward(self, features):
        raw = self.linear(features)  # (B, 4)

        # 前3维 → 自由组分比例
        free_ratio = softmax(raw[:, :3], dim=-1)  # (B, 3)

        # 第4维 → 自由组分总和 (0-100%)
        free_total = 100.0 * sigmoid(raw[:, 3:4])  # (B, 1)

        # 计算自由组分 [H₂, CH₄, CO₂]
        free_components = free_total * free_ratio  # (B, 3)

        # N₂ 闭包补全
        n2 = 100.0 - free_components.sum(dim=-1, keepdim=True)  # (B, 1)

        return concat([free_components, n2], dim=-1)  # (B, 4)
```

#### 先验初始化

```python
def _init_prior(self, output_prior):
    """将 bias 初始化为训练集均值的 logit 变换"""
    prior = torch.tensor(output_prior)  # [9.29, 75.76, 4.99, 9.96]

    free = prior[:3]  # [H₂, CH₄, CO₂]
    free_total = free.sum()  # 90.04%

    # 自由组分 logits
    free_logits = log(free / free_total)  # softmax 的逆

    # 总量 logit
    total_prob = free_total / 100.0  # 0.9004
    total_logit = log(total_prob / (1 - total_prob))  # sigmoid 的逆

    # 初始化 bias
    self.linear.bias.data = concat([free_logits, [total_logit]])
```

### 6.3 备选输出模式 (未使用)

| 模式                    | 结构                      | 约束      | 状态                  |
| --------------------- | ----------------------- | ------- | ------------------- |
| **raw4**              | Linear(64, 4)           | 无       | 可用但性能略低             |
| **GasCoordinateHead** | Linear(64, 3) + ILR 逆变换 | ILR 空间  | 完全不可训练              |
| **softmax100**        | Linear + Softmax × 100  | sum=100 | 仅 PhaseWindowTCN 支持 |

---

## 七、训练策略

### 7.1 损失函数 (P0-B 配置)

**WeightedFreeComponentMSELoss**

```python
loss = WeightedFreeComponentMSELoss(
    component_weights=[1.0, 2.0, 3.0],  # [H₂, CH₄, CO₂]
    free_components=3,
)

# 只监督前 3 个组分，N₂ 通过闭包隐式约束
error = pred[:, :3] - target[:, :3]  # (B, 3)
weights = torch.tensor([1.0, 2.0, 3.0])
loss = mean(error² × weights)
```

**权重设计原理**:

- **CH₄ × 2**: 主要组分，基线表现中等 (R²=+0.43)
- **CO₂ × 3**: 最难预测组分，baseline R²=-0.4 → P0-B 首次转正 +0.27
- **H₂ × 1**: 已经表现优秀 (R²=+0.78)，不需要额外权重

### 7.2 优化器

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5,
    betas=(0.9, 0.999),
)
```

### 7.3 学习率调度

**当前 baseline**: 不使用 cosine annealing (已验证会导致性能退化)

```python
# 简单 ReduceLROnPlateau
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
)
```

### 7.4 正则化

| 技术                | 配置   | 位置           |
| ----------------- | ---- | ------------ |
| **Dropout**       | 0.15 | Acoustic 编码器 |
| **Dropout**       | 0.25 | TCN, MLP 头   |
| **BatchNorm**     | -    | 所有卷积层        |
| **Weight Decay**  | 1e-5 | AdamW        |
| **Gradient Clip** | 1.0  | 全局梯度范数       |

### 7.5 权重初始化

```python
def _init_weights(module):
    if isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        # bias 不初始化：GasHeadNormalize 已设置先验

    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
```

---

## 八、核心设计原理

### 8.1 分层编码哲学

```
原始信号空间 (3008-d 波形)
    ↓ CNN
嵌入空间 (192-d 语义特征)
    ↓ TCN
时序依赖 (64-d 时序表示)
    ↓ MLP
决策空间 (4-d 气体组分)
```

**避免**: 端到端直接学习 3000+ 维 → 4 维映射

### 8.2 多尺度时间建模

| 尺度     | 机制             | 感受野     |
| ------ | -------------- | ------- |
| **局部** | CNN (kernel=7) | ~数十采样点  |
| **中期** | TCN (RF=29)    | 29 个时间步 |
| **长期** | 池化 (mean/max)  | 全局      |

### 8.3 模态解耦与后融合

- ultrasonic 和 fiber_mic **独立编码** (不共享权重)
- 在 **嵌入空间融合**，保留各自特征空间语义
- 优势: 避免早期融合的信息干扰

### 8.4 闭包约束输出

```
优势:
✅ 硬约束 sum = 100%，不依赖后处理
✅ 降低优化难度 (4维空间 → 3维自由流形)
✅ 物理可解释性强

劣势:
❌ N₂ 不可独立学习 (R²=-0.01)
❌ 误差累积在 N₂ 上
```

---

## 九、已证实的失败方向

| 方向                     | 配置                                 | 结果                 | 原因分析            |
| ---------------------- | ---------------------------------- | ------------------ | --------------- |
| **gas_head (ILR坐标)**   | output_mode="gas_head" + out_dim=3 | 完全不可训练             | 对数比空间非线性压缩，梯度病态 |
| **PhaseWindowTCN**     | 多窗口架构                              | R²=+0.227          | 容量分散，未超过单窗口     |
| **P3-A phase-stat 分支** | 接入 420-d Ridge 特征                  | R²=-0.085 (退化2.7×) | 统计特征引入噪声，容量不足   |
| **P0-A CO₂×2 单组分**     | component_weights=[1, 1, 2, 0]     | R²=-0.403 (崩溃)     | 单组分过度加权破坏训练平衡   |
| **Cosine annealing**   | CosineAnnealingLR                  | 性能退化               | 末期低学习率困于局部最优    |

---

## 十、当前性能与瓶颈

### 10.1 最佳性能 (P0-B)

| 指标             | 值       | 备注        |
| -------------- | ------- | --------- |
| **Overall R²** | +0.4844 | 首次突破 0.48 |
| **H₂ R²**      | +0.7758 | 优秀        |
| **CH₄ R²**     | +0.4285 | 中等        |
| **CO₂ R²**     | +0.2653 | 首次转正      |
| **N₂ R²**      | -0.0124 | 不可学       |

### 10.2 与传统 ML 对比

| 模型              | Overall R² | N₂ R² | 说明        |
| --------------- | ---------- | ----- | --------- |
| **DL (P0-B)**   | +0.4844    | -0.01 | 端到端学习     |
| **Ridge (多窗口)** | +0.71      | +0.71 | 手工特征 + 线性 |

**DL 劣势**: 小样本 (~4000) 下端到端学习效率低

### 10.3 已知瓶颈

1. **N₂ 闭包残差**: gas_head 硬约束导致 N₂ 成为误差累积器
2. **小样本困境**: 4000 样本支撑 3008 维输入，数据效率低
3. **波形编码容量**: CNN 可能未充分提取声学模式
4. **固定感受野**: TCN RF=29 可能不足以捕获长距离依赖
5. **架构搜索空间**: 未尝试 Transformer / Attention 机制

---

## 十一、参数量统计

### 11.1 分模块参数量

| 模块                     | 参数量       | 占比   |
| ---------------------- | --------- | ---- |
| **Ultrasonic Encoder** | ~24K      | 20%  |
| **Fiber_mic Encoder**  | ~24K      | 20%  |
| **Slow Encoder**       | ~2K       | 2%   |
| **TCN (3层)**           | ~73K      | 60%  |
| **MLP 头**              | ~18K      | 15%  |
| **输出头**                | ~260      | <1%  |
| **总计**                 | **~122K** | 100% |

### 11.2 容量扩张对比

| 配置           | TCN 通道          | 总参数量 | 感受野 |
| ------------ | --------------- | ---- | --- |
| **Baseline** | [64, 64, 64]    | 122K | 29  |
| **P1 扩张**    | [128, 128, 128] | 340K | 29  |
| **P2 扩展**    | [64] × 6        | 180K | 125 |

---

## 十二、可探索改进方向

详见 [improvement_plan.md](./improvement_plan.md)

### 短期 (推荐优先级)

1. **✅ P1: TCN 容量扩张** (1天)
   
   - tcn_channels=[128, 128, 128]
   - 预期 +0.05 R²

2. **✅ P2: TCN 感受野扩展** (1天)
   
   - 6层 block, RF~125
   - 验证长记忆假设

3. **✅ P3: 数据增强** (2天)
   
   - 时间抖动 + 幅度缩放
   - 预期 +0.03-0.08 R²

4. **✅ P4: Multi-scale CNN** (3天)
   
   - 并行 kernel [3, 7, 15]
   - 预期 +0.10 R²

### 中期 (条件性探索)

5. **LSTM 替换 TCN** (5天，高风险)
   
   - 仅在 P2 验证长记忆关键时尝试
   - 训练成本 3-5× 增加

6. **Transformer backbone** (7天)
   
   - Self-attention 替换 TCN
   - 需要位置编码设计

### 长期 (范式转变)

7. **自监督预训练**
   
   - 在大量无标签波形上预训练编码器

8. **物理约束注入**
   
   - 在损失函数显式建模气体混合物理规律

9. **混合模型**
   
   - DL 提取特征 → XGBoost 回归

---

## 附录 A: 代码路径索引

| 模块               | 文件路径                                                       |
| ---------------- | ---------------------------------------------------------- |
| **主模型**          | `src/dl/models/cnn1d_tcn_fusion.py`                        |
| **TCN 基础**       | `src/dl/models/tcn.py`                                     |
| **Acoustic 编码器** | `src/dl/models/cnn1d_tcn_fusion.py::DeepAcousticEncoder1D` |
| **损失函数**         | `src/dl/training/losses.py`                                |
| **数据集**          | `src/dl/data/dataset.py`                                   |
| **训练脚本**         | `run/pipeline/train_dl.py`                                 |

## 附录 B: 关键超参数速查

```yaml
# P0-B 最佳配置
model:
  name: cnn1d_tcn_fusion
  kwargs:
    in_channels: 3008
    out_dim: 4
    waveform_embedding_dim: 64
    waveform_adc_scale: 524287.0
    acoustic_channels: [16, 32, 64, 64]
    acoustic_kernel_size: 7
    acoustic_dropout: 0.15
    tcn_channels: [64, 64, 64]
    tcn_kernel_size: 3
    tcn_dropout: 0.25
    shared_hidden_dims: [128, 64]
    output_mode: "gas_head"
    output_prior: [9.288469, 75.755157, 4.994778, 9.961745]

loss:
  name: weighted_free_component_mse
  component_weights: [1.0, 2.0, 3.0]

training:
  epochs: 50
  batch_size: 32
  lr: 1e-3
  weight_decay: 1e-5
  gradient_clip: 1.0
```

---

**文档版本**: v1.0
**最后更新**: 2026-06-24
**维护者**: DL 团队
