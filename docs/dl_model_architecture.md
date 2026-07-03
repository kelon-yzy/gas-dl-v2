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
- [十三、仿真框架（DL 输入数据来源）](#十三仿真框架dl-输入数据来源)

---

## 一、输入数据结构

### 输入格式

- **张量形状**: `(Batch, Timesteps, Channels)` — NTC 格式
- **总通道数**: 15008（8 slow + 5000 ultrasonic + 10000 fiber_mic）
- **时间步范围**: 50-150 步（取决于相位窗口）

### 通道组成

| 模态                | 通道数  | 索引范围       | 说明                                   |
| ----------------- | ---- | ---------- | ------------------------------------ |
| **slow 变量**       | 8    | 0:8        | 温度 T_C、压力 P_MPa、湿度 H_RH、管长 L_m、活塞位置等 |
| **ultrasonic 波形** | 5000 | 8:5008     | 超声传感器时间序列，int32 原始采样（200kHz 载波 / 1MS/s / 20-bit ADC） |
| **fiber_mic 波形**  | 10000 | 5008:15008 | 光纤麦克风时间序列，int32 原始采样（10ms 窗口 / 1MS/s）  |

### 目标输出

- **4维气体组分**: `[H₂, CH₄, CO₂, N₂]` 百分比 (%)
- **约束**: sum(components) = 100%

---

## 二、模型架构层次

### 架构总览

```
输入 (B, T, 15008)
    ↓
┌───────────────────────────────────────────────────────────┐
│  第一阶段: 多模态特征编码                                     │
├───────────────────────────────────────────────────────────┤
│  slow (8-d) → SlowFeatureEncoder → (B, T, 64)             │
│  ultrasonic (5000-d) → DeepAcousticEncoder1D → (B, T, 64) │
│  fiber_mic (10000-d) → DeepAcousticEncoder1D → (B, T, 64) │
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
# int32 原始波形（20-bit ADC）→ 归一化
waveform_normalized = waveform.float() / 524287.0
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
原始信号空间 (15008-d 波形)
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
2. **小样本困境**: 4000 样本支撑 15008 维输入，数据效率低
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

## 十三、仿真框架（DL 输入数据来源）

DL 模型的输入（slow 变量、超声波形、光纤麦克风波形）全部由 `src/sim/` 仿真框架生成。仿真框架围绕两个气体场景构建：

- **hydrogen_ng (hg)**：H₂/CH₄/CO₂/N₂，sum=100% 闭包，8 慢通道，benchmark `wv4-*`
- **syngas**：H₂/CH₄/CO₂/CO，N₂ 为背景气 sum<100%，9 慢通道（+V_NDIR_CO），benchmark `sg4-*`

两场景共享波形/打包/校验代码，物理后端与 schema 各自独立。本章以 hg 为主线，syngas 差异见 §13.10。

### 13.1 框架总览与数据流

```
BenchmarkGenerationSpec
        │
        ▼
generate_benchmark_dataset(benchmark.py)
        │
        ├─ conditions.py        LHS 采样 → condition_grid（组分 + 环境变量）
        ├─ slow.py              build_sequence_arrays → slow[N,T,8] + ultrasonic[N,T,5000] + fiber_mic[N,T,10000]
        │     ├─ acoustic_physics.py   声速/衰减/TCS/NDIR 经验吸收
        │     ├─ waveforms.py          超声 + 光纤麦克风波形（200kHz/1MS/s/20-bit）
        │     ├─ phases.py             PhaseSchedule（baseline→exposure→steady→recovery）
        │     └─ spectral/             HITRAN 后端（可选，NDIR 光谱前向）
        ├─ packaging/
        │     ├─ arrays.py      写 npy/memmap/npz
        │     ├─ manifest.py    manifest.json（含 sim_revision）
        │     ├─ scalers.py     z_score scaler（train 拟合）
        │     └─ splits.py      mixture_id 主键 70/15/10/rest
        ├─ validation/          完整性校验
        └─ 发布 staging → output_dir
```

### 13.2 整体编排（benchmark.py）

入口 `generate_benchmark_dataset(output_root, spec)`（`benchmark.py:110`）按以下步骤执行：

| 步骤 | 关键调用 | 说明 |
|---|---|---|
| 校验 spec | `_validate_spec` | |
| 生成 conditions | `generate_condition_rows` | LHS 采样 sequence_count 条 |
| 校验 HITRAN 缓存 | `validate_hitran_benchmark_cache` | 仅 HITRAN 后端 |
| 构建 splits | `build_default_split_rows` | mixture_id 主键划分 |
| 构建标签数组 | `_label_array` | `(N, 4)` float32 |
| 构建序列数组 | `build_sequence_arrays` / 并行分块 | slow + 波形 + 辅助 |
| 完整性校验 | `validate_benchmark_assets` | |
| 写入数组 | `write_arrays` | npy/memmap/npz |
| 构建 manifest | `build_manifest` | 含 sim_revision |
| 写 CSV/JSON 元数据 | condition_grid / sequence_index / labels / slow_sequence_long / splits | |
| 拟合 scalers | `fit_z_score_scalers` | train 序列上拟合 |
| 发布 | `_publish_staging_dir` | staging 原子重命名为 output_dir |
| 异常回滚 | `shutil.rmtree(staging_dir)` | |

`BenchmarkGenerationSpec`（`benchmark.py:89-107`）字段：

| 字段 | 默认值 | 含义 |
|---|---|---|
| `dataset_slug` | (必填) | 数据集标识 |
| `sequence_count` | (必填) | 序列数 |
| `seed` | (必填) | 全局随机种子 |
| `timesteps` | 128 | 每序列时间步数 |
| `dt_s` | 0.5 | 时间步间隔（秒） |
| `storage` | "memmap" | memmap/npz/both |
| `multi_path_phase` | "steady" | 多光程扫描阶段 off/baseline/steady |
| `stage_profile` | "standard_exposure" | 相位调度预设 |
| `stage_jitter` | 0.0 | 阶段时长抖动分数 |
| `sampling_strategy` | "lhs" | lhs/random |
| `path_lms` | (0.18,0.20,0.22,0.25,0.28) | 多光程扫描路径（m） |
| `optical_absorption_backend` | "hitran_hapi_v1" | NDIR 后端 |
| `hitran_cache_root` | "data/hitran_cache" | HITRAN 缓存根目录 |
| `workers` / `chunk_size` | 1 / None | 并行分块参数 |

### 13.3 条件采样（conditions.py）

LHS 维度 d=3（H₂/CO₂/N₂ 三个自由度，CH₄ 为闭合余项），`_sample_components_lhs`：

| 组分 | 范围 | 采样逻辑 |
|---|---|---|
| x_H₂ | 0~30% | 双峰映射：15% trace [0,3]、70% mid [0,30]、15% high [25,30] |
| x_CO₂ | 0~15% | 均匀 `u*15` |
| x_N₂ | 0~20% | 均匀 `u*20` |
| x_CH₄ | ≥40% | `100-sum`，若 <40% 压缩 N₂ 保 CH₄≥40% |

环境变量（`generate_condition_rows`）：T_C ∈ [15,35]°C、P_MPa ∈ [0.10,0.709]、H_RH ∈ [20,80]%、**L_m ∈ [0.2,0.3]m**（200kHz 物理约束，见 §13.5）。

### 13.4 慢通道动力学（slow.py）

`build_sequence_arrays`（`slow.py:35`）遍历每条 condition 的每个 timestep，核心是 **multi-tau 双指数动力学**：

- `_multi_tau_channel_step`（`slow.py:391`）：快速分量（fast_tau）+ 慢速分量（slow_tau）加权混合，下降时加 recovery_floor 防归零
- `_dynamic_features_from_equilibrium`（`slow.py:342`）：调上述递推，叠加 random_walk + drift + noise
- `_channel_dynamic_params`（`slow.py:371`）：为每个动态通道生成 tau_rise/tau_decay/fast_fraction/slow_multiplier/fast_weight/recovery_floor 随机参数
- `_blend_equilibrium_features`（`slow.py:269`）：baseline 与 target 间线性插值

**后端切换**（`slow.py:69-72`）：

| 后端 | 动力学模式 | NDIR 计算 |
|---|---|---|
| empirical_v1 + standard_exposure + 无 jitter | legacy 单时间常数（`_dynamic_slow_features`） | 经验线性吸收 `_hidden_absorption_ch4/co2` |
| empirical_v1 其他 | multi-tau 双指数 | 同上 |
| hitran_hapi_v1 | multi-tau 双指数（强制） | HITRAN 光谱前向 `_hitran_ndir_equilibrium` |

**多光程扫描**（`_path_l_m_for_schedule`，`slow.py:418`）：若 `multi_path_phase=baseline` 在 baseline 段扫描 path_lms；`=steady` 在 steady 段扫描；否则固定 L_m_base。5 档默认 (0.18,0.20,0.22,0.25,0.28)m。

### 13.5 声学物理（acoustic_physics.py）

**声速** `hidden_sound_speed_v2`（`acoustic_physics.py:45`）：组分声速加权 + 温度修正 `0.6*(t_c-25)`，下限 200 m/s。常数：H₂=1306、CH₄=446、CO₂=268、N₂=353 m/s。

**衰减** `hidden_attenuation_v2`（`acoustic_physics.py:60`）：返回 `alpha_true_v2`（Np/m）及分量分解：

| 分量 | 机制 | 关键常数 |
|---|---|---|
| alpha_classical | 经典粘滞吸收 ∝f² | K_ref=1.84e-11 |
| alpha_co2 | CO₂ V-T 弛豫 | f_relax=28kHz/atm, λ_max=0.12, H2O 加速 0.015 |
| alpha_ch4 | CH₄ V-T 弛豫 | base=30kHz/atm + slope=120kHz/atm·frac, λ_max=0.034 |
| alpha_h2_diffusion | H₂ 扩散损耗 ∝f² | k=1.6e-3 |
| alpha_n2 | N₂ V-T 弛豫 | 65kHz/atm, λ_max=0.004 |
| alpha_h2o | H₂O V-T 弛豫 | 100kHz/atm, λ_max=0.01 |

**200kHz 物理约束**：CH₄/CO₂ 弛豫峰在 P=0.5MPa 下上移至 ~140kHz，载波落在峰附近，CH₄ 主导气样在 L≥0.5m 信号被吸收淹没。这是 L_m 上限压缩到 0.3m 的根因（见 `docs/Phase0_物理可行性核对记录.md`）。

**主特征** `main_sensor_features`（`acoustic_physics.py:162`）：输出 TOF（含 80µs 系统延迟 + 3µs 触发抖动）、Amp（`exp(-αL)`）、f_peak（载波 + 组分偏移）、A_fft_max、V_NDIR_CH4/CO2（Beer-Lambert `baseline·exp(-A)`）、V_TCS（热导线性模型）、饱和标志、光学/热漂移。`f_hz` 参数化（Phase 1 改造），消除原硬编码 40000。

**TCS 热导**：`_hidden_lambda_mix` 经验线性 `λ=0.034+0.00155·x_H2-0.00011·x_CO2+...`，`_tcs_voltage` 映射到电压。HITRAN 模式下 `thermal_conductivity_sensor_feature` 仅算 TCS。

### 13.6 波形仿真（waveforms.py）

**WaveformSpec** 关键字段（Phase 2/3 改造后）：

| 字段 | 值 | 说明 |
|---|---|---|
| `sample_rate_hz` | 1_000_000 | 1 MS/s（NI-6453 同步采样率） |
| `center_frequency_hz` | 200_000 | 200 kHz（PSC200K） |
| `burst_cycles` | 8 | 超声脉冲周期数 |
| `measurement_window_s` | 0.005 | 5ms 窗口 → 5000 采样点 |
| `daq_bits` | 20 | ADC 位深（NI-6453） |
| `waveform_dtype` | "int32" | 量化 dtype（20-bit 无原生类型） |
| `daq_full_scale_v` | 2.5 | ±2.5V 量程（Phase 0 推荐分辨率最优档） |
| `noise_std_v` | 1e-3 | 前端噪声 1mV |
| `transducer_bandwidth_hz` | 20000 | 换能器带宽（~10% 中心频率估计） |
| `transducer_ringdown_cycles` | 4.0 | 振铃衰减周期 |

`adc_max` property = `2**(daq_bits-1)-1` = 524287，由 `daq_bits` 推导。`adc_max_from_bits` 工具函数（`waveforms.py`）。

**超声波形** `simulate_waveform_measurement`（`waveforms.py:210`）流程：
1. `_compute_physics`（声速 + 衰减，`f_hz=spec.center_frequency_hz`）
2. `tof_true_s = l_m / c_sound`，加系统延迟/电缆延迟/触发抖动
3. `transducer_response_pulse`：burst pulse（Hanning 窗正弦）与二阶谐振核卷积，归一化
4. `_add_pulse_at_peak`：按 `exp(-αL)` 幅度缩放置于 TOF 位置
5. 加高斯噪声 → `_digitize_waveform` 量化（int32）→ TOF quality 评分

**光纤麦克风** `simulate_fiber_mic_measurement`（`waveforms.py:276`）：直接脉冲 + 多次反射（最多 3 次，反射系数 0.08）→ 光学相位解调（压力灵敏度或位移灵敏度两路径）→ 光电噪声 → 量化。窗口 10ms → 10000 采样点。

两函数均支持 `sound_speed_fn`/`attenuation_fn`/`extra_gas_kwargs` 注入，是 syngas 物理后端的挂钩点。

### 13.7 NDIR 光学后端

两套后端，由 `optical_absorption_backend` 切换：

| 维度 | empirical_v1 | hitran_hapi_v1 |
|---|---|---|
| 吸收计算 | 线性经验公式 | HAPI Voigt 线型逐波数积分 |
| 串扰 | `apply_optical_crosstalk` 固定系数 | 光谱积分自动包含多气体交叉 |
| T/P 依赖 | 隐含在经验系数 + 漂移项 | 每个 condition 独立 HitranGridSpec |
| 预处理 | 无 | 需预计算 HITRAN 缓存 |
| 滤光片 | 不涉及（直接给吸收度） | 高斯滤光片（CH₄ 3030cm⁻¹、CO₂ 2347cm⁻¹，**占位符非实际 datasheet**） |

HITRAN 后端位于 `src/sim/generation/spectral/`：`hitran_backend.py`（HAPI 底层 + 缓存）、`filters.py`（高斯滤光核）、`tabulated_backend.py`（表谱积分）、`integration.py`（透射率/吸光度积分）、`cache.py`（文件级缓存）、`defaults.py`（网格/滤镜配置）。编排层 `optical_backend.py` 对 ch4/co2 两通道分别调用。

**注**：NDIR 滤光片 FWHM 目前是行业参考占位值，非 TraceGas 实际 datasheet，待取得后替换（见 `docs/references/传感器硬件资料整理.md` 附录 A.4）。

### 13.8 打包与校验（packaging + validation）

**arrays.py** `write_arrays`：写 14 个数组到 `sequences/`：
- `slow.npy` `(N, T, 8)`、`ultrasonic_<dtype>.npy` `(N, T, 5000)`、`fiber_mic_<dtype>.npy` `(N, T, 10000)`
- 10 个标量辅助数组（tof_s/alpha/quality/peak_index/sound_speed 等）
- `waveform_sequence.npz`（可选压缩包）
- 文件名 dtype 后缀由 `common/waveform.py::waveform_array_filename` 统一拼接，读写端共用

**manifest.py** `build_manifest`：schema_version、composition_scheme、dataset 参数、shapes、通道列表、labels、background_fields、optical/acoustic metadata、**sim_revision**（链路版本 `v5-200khz-20bit-L03`）。

**scalers.py** `fit_z_score_scalers`：z_score，在 train 序列上拟合，输出 `sequence_scaler`（逐通道全局）+ `modal_scaler`（按 optical/thermal/environment 分组）。`Z_SCORE_STD_EPSILON` 防除零。

**splits.py** `build_default_split_rows`：**mixture_id 为主键**，对 unique mixture_id 洗牌后按 70/15/10/rest 划分 train/val/test/extrapolation，确保同一 mixture 的所有序列进同一 split。

**validation/integrity.py** `validate_benchmark_assets`：校验无遗留字段、sequence_id/mixture_id 唯一、组分和=100%（syngas 为 component+background=100%）、split 覆盖完整、数组 shape 匹配。

### 13.9 Phase Schedule 系统（phases.py）

`PhaseSchedule` 把时间轴划分为 baseline → exposure → steady → recovery 四段，每段有 `duration_frac` 和 `blend_shape`。核心方法：`boundaries(timesteps)`、`phase_for_timestep`、`blend_for_timestep`、`resolve_timeline`（批量）、`jittered(rng, jitter)`（段长抖动）。

预设（`phases.py:123`）：

| 预设 | 段比 baseline/exposure/steady/recovery | 特点 |
|---|---|---|
| `standard_exposure` | 25/25/25/25 | 标准四相等分 |
| `variable_onset` | 35/20/25/20 | 长基线 + 短暴露 |
| `fast_transient` | 45/12/8/35 | 极短稳态 |
| `incomplete_recovery` | 25/25/25/25 | recovery 留 20% 残差 |
| `multi_pulse` | 12 段 = 3 个完整循环 | 三次脉冲循环 |

`blend_shape`：`hold0`（0）、`ramp_up`（线性↑）、`hold1`（1）、`ramp_down`（线性↓至 floor）。baseline/target 按 blend 系数插值，驱动慢通道动力学从基线趋向目标浓度再恢复。

### 13.10 syngas 场景差异

| 维度 | hg | syngas |
|---|---|---|
| COMPONENT_FIELDS | (H₂,CH₄,CO₂,N₂) sum=100% | (H₂,CH₄,CO₂,CO) sum<100% |
| BACKGROUND_FIELDS | 空 | (x_N2,) — N₂ 入物理不计预测 |
| SLOW_CHANNELS | 8 | 9 (+V_NDIR_CO) |
| schema_version | v4-benchmark-1 | v4-syngas-1 |
| conditions 采样 | LHS d=3（H₂/CO₂/N₂） | 顺序采样 d=4（CO→H₂→CO₂→CH₄，方案 B 煤气化全谱） |
| 声学物理 | 5 参数 | 6 参数（+x_CO，CO 声速 352 ≈ N₂ 353，**声学近简并**） |
| 衰减 | 5 分量 | +CO V-T 弛豫（f_relax=12kHz/atm, λ_max=0.025） |
| NDIR 通道 | CH₄/CO₂ | +CO（独立网格 1980~2310cm⁻¹，Step 1 无串扰 / Step 2 CO₂↔CO 互扰） |
| HITRAN 缓存 | data/hitran_cache | data/hitran_cache_syngas（隔离） |
| 动力学 | empirical 可走 legacy 单 τ | 始终 multi-tau（无 legacy） |

syngas 的 CO 与 N₂ 摩尔质量均为 28，声速差 <1 m/s，声学通道无法区分 CO/N₂，CO 可观测性依赖 NDIR 光学（见 `docs/syngas/` 系列文档）。

### 13.11 链路版本与历史对齐

2026-07-02 仿真链路对齐实际硬件型号（见 `docs/传感器仿真对齐改造计划.md`）：

| 项 | 旧值 | 新值 | 依据 |
|---|---|---|---|
| 超声载波 | 40 kHz | 200 kHz | PSC200K 型号 |
| 采样率 | 200 kS/s | 1 MS/s | NI-6453 同步采样率 |
| ADC 位深 | 16-bit int16 | 20-bit int32 | NI-6453 |
| 量程 | ±5V | ±2.5V | Phase 0 推荐分辨率最优档 |
| L_m 范围 | 0.2~1.8m | 0.2~0.3m | 200kHz 下 CH₄/CO₂ 弛豫吸收约束 |
| path_lms | (0.20..0.40) | (0.18..0.28) | 同上 |
| DL in_channels | 3008 | 15008 | 波形采样点 ×5 |
| 模型归一化 | waveform_int16_scale=32767 | waveform_adc_scale=524287 | adc_max 同步 |

manifest 的 `sim_revision` 字段标记链路版本，旧 benchmark（`data/_archived_pre_200khz/`）不可用于新链路。

## 附录 A: 代码路径索引

| 模块               | 文件路径                                                       |
| ---------------- | ---------------------------------------------------------- |
| **主模型**          | `src/dl/models/cnn1d_tcn_fusion.py`                        |
| **TCN 基础**       | `src/dl/models/tcn.py`                                     |
| **Acoustic 编码器** | `src/dl/models/cnn1d_tcn_fusion.py::DeepAcousticEncoder1D` |
| **损失函数**         | `src/dl/training/losses.py`                                |
| **数据集**          | `src/dl/data/dataset.py`                                   |
| **训练脚本**         | `run/pipeline/train_dl.py`                                 |
| **仿真编排**        | `src/sim/generation/benchmark.py`                          |
| **条件采样**        | `src/sim/generation/conditions.py`                         |
| **慢通道动力学**     | `src/sim/generation/slow.py`                               |
| **波形仿真**        | `src/sim/generation/waveforms.py`                          |
| **声学物理**        | `src/sim/generation/acoustic_physics.py`                   |
| **相位调度**        | `src/sim/generation/phases.py`                             |
| **HITRAN 后端**    | `src/sim/generation/spectral/hitran_backend.py`            |
| **光学后端编排**     | `src/sim/generation/optical_backend.py`                    |
| **数组打包**        | `src/sim/packaging/arrays.py`                              |
| **manifest**      | `src/sim/packaging/manifest.py`                            |
| **scaler**        | `src/sim/packaging/scalers.py`                             |
| **splits**        | `src/sim/packaging/splits.py`                              |
| **完整性校验**       | `src/sim/validation/integrity.py`                          |
| **波形文件名工具**    | `src/common/waveform.py`                                   |
| **syngas 仿真**    | `src/sim/generation/syngas/`                               |

## 附录 B: 关键超参数速查

```yaml
# P0-B 最佳配置
model:
  name: cnn1d_tcn_fusion
  kwargs:
    in_channels: 15008
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

**文档版本**: v1.1（新增 §13 仿真框架，同步 200kHz/20-bit 链路参数）
**最后更新**: 2026-07-03
**维护者**: DL 团队
