# DL 模型改进计划

**当前基线**: CNN1DTCNFusionRegressor, overall R²=+0.4844 (P0-B 配置)

**目标**: 突破 R²=0.52 门槛，逼近 Ridge 的 0.71 性能

---

## 优先级与路线图

| 优先级 | 方案 | 预期收益 | 实验成本 | 风险等级 | 预计完成 |
|--------|------|---------|---------|---------|---------|
| **P1** | TCN 容量扩张 | +0.05 R² | 1天 | 低 | Day 1 |
| **P2** | TCN 感受野扩展 | +0.02-0.05 R² | 1天 | 低 | Day 2 |
| **P3** | 数据增强 | +0.03-0.08 R² | 2天 | 低 | Day 3-4 |
| **P4** | Multi-scale CNN | +0.10 R² | 3天 | 中 | Day 5-7 |

**总耗时**: 约 7 个工作日

---

## P1: TCN 容量扩张

### 目标
验证当前模型是否受限于参数容量不足，通过增加 TCN 通道数提升表达能力。

### 技术方案

#### 配置变更
```yaml
# 当前 baseline
model:
  name: cnn1d_tcn_fusion
  kwargs:
    tcn_channels: [64, 64, 64]     # 参数量: ~73K
    tcn_kernel_size: 3
    tcn_dropout: 0.25

# 扩张方案
model:
  kwargs:
    tcn_channels: [128, 128, 128]  # 参数量: ~290K (4倍)
    tcn_kernel_size: 3
    tcn_dropout: 0.30              # 增加 dropout 防止过拟合
```

#### 实施步骤
1. **创建配置文件**: `run/conf/p1_tcn_capacity.yaml`
2. **训练**: 3 seeds × 50 epochs
   ```bash
   python run/pipeline/train_dl.py \
       --config conf/p1_tcn_capacity.yaml \
       --seed 20260623,42,1337
   ```
3. **评估**: 对比 baseline 的 R² 提升

#### 验收标准
- ✅ **成功**: test overall R² ≥ 0.53 (至少 +0.05)
- ⚠️ **部分成功**: R² 在 0.50-0.53 之间
- ❌ **失败**: R² < 0.50 或出现严重过拟合 (train-val gap > 0.15)

#### 风险与缓解
| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| 过拟合 | 中 | 增加 dropout 到 0.30-0.35 |
| 训练时间过长 | 低 | 参数量增加 4× 但 TCN 并行，实际增幅约 1.5× |
| 显存不足 | 低 | 减小 batch_size 从 32 → 24 |

---

## P2: TCN 感受野扩展

### 目标
验证任务是否需要 >29 步的长时序依赖，为 LSTM 替换方案提供诊断依据。

### 技术方案

#### 配置变更
```yaml
# 当前 baseline: 感受野 = 29 步
tcn_channels: [64, 64, 64]
# RF = 1 + 2*(k-1)*sum(dilations) = 1 + 2*2*(1+2+4) = 29

# 扩展方案: 感受野 = 125 步
tcn_channels: [64, 64, 64, 64, 64, 64]  # 6层
# RF = 1 + 2*2*(1+2+4+8+16+32) = 1 + 4*63 = 253 步（实际有效约125步）
```

#### 实施步骤
1. **创建配置**: `run/conf/p2_tcn_long_rf.yaml`
2. **对比实验**:
   - Variant A: baseline (RF=29)
   - Variant B: 4层 (RF=61)
   - Variant C: 6层 (RF=125)
3. **分析感受野收益曲线**

#### 验收标准
- ✅ **长记忆关键**: 6层比 baseline 显著提升 (>+0.05 R²)
  - **后续**: 放弃 LSTM，继续优化 TCN
- ⚠️ **边际收益**: 提升 +0.02~0.05
  - **后续**: 保持 3-4 层，投入其他方向
- ❌ **无收益**: 提升 <0.02
  - **后续**: 长记忆不是瓶颈，跳过 LSTM 替换方案

#### 关键诊断指标
```python
# 分析不同组分对感受野的敏感性
components_sensitivity = {
    'H2': rf_impact,   # 快扩散 → 预期低敏感
    'CH4': rf_impact,  # 中等
    'CO2': rf_impact,  # 慢扩散 → 预期高敏感
    'N2': rf_impact,   # 闭包残差
}
```

---

## P3: 数据增强

### 目标
提升模型泛化能力，缓解小样本 (~4000) 过拟合风险。

### 技术方案

#### 增强策略

##### 1. 时间抖动 (Time Jitter)
```python
class TimeJitterAugment:
    """随机平移时间序列起点"""
    def __init__(self, max_shift=5):
        self.max_shift = max_shift
    
    def __call__(self, x):
        # x: (T, C)
        shift = np.random.randint(-self.max_shift, self.max_shift+1)
        if shift > 0:
            x = np.concatenate([x[shift:], x[-shift:]], axis=0)
        elif shift < 0:
            x = np.concatenate([x[:shift], x[:-shift]], axis=0)
        return x
```

##### 2. 幅度缩放 (Amplitude Scaling)
```python
class AmplitudeScaleAugment:
    """对波形通道施加随机缩放"""
    def __init__(self, scale_range=(0.95, 1.05)):
        self.scale_range = scale_range
    
    def __call__(self, x):
        # 只对 ultrasonic 和 fiber_mic 通道缩放
        scale = np.random.uniform(*self.scale_range)
        x[:, 8:] *= scale  # slow 通道不动
        return x
```

##### 3. 高斯噪声 (Gaussian Noise)
```python
class GaussianNoiseAugment:
    """添加微小高斯噪声模拟传感器误差"""
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std
    
    def __call__(self, x):
        noise = np.random.randn(*x.shape) * self.noise_std
        return x + noise
```

#### 配置变更
```yaml
# 在 Dataset 配置中启用增强
dataset:
  augment_config:
    time_jitter:
      enabled: true
      max_shift: 5
    
    amplitude_scale:
      enabled: true
      scale_range: [0.95, 1.05]
      apply_to: ['ultrasonic', 'fiber_mic']  # 不对 slow 缩放
    
    gaussian_noise:
      enabled: true
      noise_std: 0.01
    
    # 概率控制
    apply_prob: 0.5  # 50% 样本应用增强
```

#### 实施步骤
1. **实现增强类**: 在 `src/dl/data/augmentation.py` 中实现
2. **单元测试**: 验证增强不破坏物理约束
   ```python
   def test_augmentation_preserves_sum():
       # gas_head 闭包: sum(components) = 100%
       assert augmented_sample.sum() ≈ 100.0
   ```
3. **消融实验**: 单独测试每种增强的贡献
   - No augment (baseline)
   - Time jitter only
   - Amplitude scale only
   - Gaussian noise only
   - All combined
4. **最优组合**: 选择收益最大的组合

#### 验收标准
- ✅ **显著改进**: test R² ≥ 0.51 且 train-val gap 减小 >0.03
- ⚠️ **轻微改进**: test R² 在 0.49-0.51 之间
- ❌ **无效**: test R² 无提升或 train-val gap 增大

#### 风险与缓解
| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| 破坏物理约束 | 中 | 对 slow 通道（温度/压力）禁用幅度缩放 |
| 训练不稳定 | 低 | 降低 apply_prob 到 0.3-0.4 |
| 增强过度 | 低 | 缩小 scale_range 到 [0.97, 1.03] |

---

## P4: Multi-scale CNN

### 目标
通过并行多尺度卷积提取不同频率的声学特征，增强波形编码器容量。

### 技术方案

#### 架构设计

##### 当前 DeepAcousticEncoder1D
```python
# 单一 kernel_size=7
Conv1d(1, 16, kernel=7, stride=2) -> BN -> ReLU
Conv1d(16, 32, kernel=7, stride=2) -> BN -> ReLU
Conv1d(32, 64, kernel=7, stride=2) -> BN -> ReLU
Conv1d(64, 64, kernel=7, stride=1) -> BN -> ReLU
-> Global Pooling (avg + max + log_amp) -> 64-d embedding
```

##### 改进: MultiScaleAcousticEncoder1D
```python
class MultiScaleAcousticEncoder1D(nn.Module):
    """并行多尺度波形编码器"""
    
    def __init__(
        self,
        waveform_length: int,
        embedding_dim: int = 64,
        kernels: list[int] = [3, 7, 15],  # 多尺度
        channels: list[int] = [16, 32, 64, 64],
    ):
        super().__init__()
        
        # 为每个 kernel 创建独立分支
        self.branches = nn.ModuleList([
            self._build_branch(k, channels)
            for k in kernels
        ])
        
        # 融合多尺度特征
        branch_dim = channels[-1] * 2 + 1  # avg + max + log_amp
        self.fusion = nn.Linear(branch_dim * len(kernels), embedding_dim)
    
    def _build_branch(self, kernel_size, channels):
        """构建单个尺度的卷积分支"""
        layers = []
        current = 1
        for idx, hidden in enumerate(channels):
            stride = 1 if idx == len(channels) - 1 else 2
            padding = kernel_size // 2
            layers.extend([
                nn.Conv1d(current, hidden, kernel_size, stride, padding, bias=False),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(0.15),
            ])
            current = hidden
        return nn.Sequential(*layers)
    
    def forward(self, waveform):
        # waveform: (B, T, L)
        B, T, L = waveform.shape
        flat = waveform.reshape(B*T, 1, L).float() / 32767.0
        
        # 并行提取多尺度特征
        branch_features = []
        for branch in self.branches:
            encoded = branch(flat)
            avg = F.adaptive_avg_pool1d(encoded, 1).squeeze(-1)
            mx = F.adaptive_max_pool1d(encoded, 1).squeeze(-1)
            log_amp = torch.log1p(flat.abs().mean(dim=-1))
            branch_features.append(torch.cat([avg, mx, log_amp], dim=-1))
        
        # 融合
        fused = torch.cat(branch_features, dim=-1)
        embedding = self.fusion(fused)
        return embedding.reshape(B, T, -1)
```

#### 配置变更
```yaml
model:
  name: cnn1d_tcn_fusion
  kwargs:
    # 替换默认编码器
    acoustic_encoder_type: "multiscale"
    acoustic_kernels: [3, 7, 15]  # 小/中/大尺度
    acoustic_channels: [16, 32, 64, 64]
    
    # 其余保持不变
    tcn_channels: [64, 64, 64]
    ...
```

#### 实施步骤
1. **实现 MultiScaleAcousticEncoder1D**: 新建 `src/dl/models/multiscale_encoder.py`
2. **修改 CNN1DTCNFusionRegressor**: 支持 `acoustic_encoder_type` 参数切换
3. **单元测试**: 验证输出维度和梯度流
4. **消融实验**:
   - Kernel [7] (baseline)
   - Kernel [3, 7]
   - Kernel [3, 7, 15]
   - Kernel [3, 7, 15, 31] (极端情况)

#### 验收标准
- ✅ **显著改进**: test R² ≥ 0.54 (至少 +0.10)
- ⚠️ **中等改进**: test R² 在 0.50-0.54 之间
- ❌ **无效**: test R² < 0.50

#### 理论依据
- **物理直觉**: 不同气体混合产生不同频率的声学特征
  - H₂: 高频振动 (小 kernel=3)
  - CH₄: 中频 (中 kernel=7)
  - CO₂: 低频振荡 (大 kernel=15)
- **经验支持**: Inception 网络在视觉任务中验证了多尺度并行的有效性

#### 风险与缓解
| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| 参数量爆炸 | 中 | 限制 kernels 数量 ≤ 3 |
| 训练不稳定 | 低 | 每个分支独立 BN，融合前 LayerNorm |
| 显存不足 | 低 | 减小 channels 或 batch_size |

---

## 实验基础设施

### 统一配置模板
```yaml
# run/conf/experiment_base.yaml
defaults:
  - model: cnn1d_tcn_fusion
  - loss: weighted_free_component_mse
  - optimizer: adamw

# 固定参数 (所有实验保持一致)
training:
  epochs: 50
  batch_size: 32
  lr: 1e-3
  weight_decay: 1e-5
  gradient_clip: 1.0

loss:
  component_weights: [1.0, 2.0, 3.0]  # P0-B 最佳配置

dataset:
  modalities: [slow, ultrasonic, fiber_mic]
  input_format: NTC
  dequantize_waveforms: true

# 随机种子
seeds: [20260623, 42, 1337]
```

### 评估脚本
```bash
# scripts/evaluate_experiment.sh
#!/bin/bash

EXPERIMENT_NAME=$1
CONFIG_PATH="run/conf/${EXPERIMENT_NAME}.yaml"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"

# 训练多个 seed
for seed in 20260623 42 1337; do
    python run/pipeline/train_dl.py \
        --config $CONFIG_PATH \
        --seed $seed \
        --output_dir ${OUTPUT_DIR}/seed_${seed}
done

# 聚合结果
python scripts/aggregate_results.py \
    --experiment_dir $OUTPUT_DIR \
    --baseline_r2 0.4844 \
    --output ${OUTPUT_DIR}/summary.json
```

### 结果汇总模板
```python
# scripts/aggregate_results.py
import json
from pathlib import Path

def aggregate_results(experiment_dir, baseline_r2):
    results = {
        'experiment': experiment_dir.name,
        'baseline_r2': baseline_r2,
        'seeds': [],
    }
    
    for seed_dir in experiment_dir.glob('seed_*'):
        metrics = json.load(open(seed_dir / 'test_metrics.json'))
        results['seeds'].append({
            'seed': seed_dir.name,
            'test_r2_overall': metrics['r2_overall'],
            'test_r2_components': {
                'H2': metrics['r2_H2'],
                'CH4': metrics['r2_CH4'],
                'CO2': metrics['r2_CO2'],
                'N2': metrics['r2_N2'],
            },
        })
    
    # 计算统计量
    r2_values = [s['test_r2_overall'] for s in results['seeds']]
    results['mean_r2'] = np.mean(r2_values)
    results['std_r2'] = np.std(r2_values)
    results['improvement'] = results['mean_r2'] - baseline_r2
    
    return results
```

---

## 里程碑与检查点

| 日期 | 里程碑 | 检查点 | Go/No-Go 决策 |
|------|--------|--------|---------------|
| Day 1 | P1 完成 | R² ≥ 0.53? | Yes → 继续; No → 分析失败原因 |
| Day 2 | P2 完成 | RF 扩展有效? | Yes → 放弃 LSTM; No → 考虑 LSTM |
| Day 4 | P3 完成 | 数据增强收益? | >+0.03 → 后续实验启用 |
| Day 7 | P4 完成 | Multi-scale 突破? | ≥0.54 → 成功; <0.50 → 回退 |

### 最终决策树
```
Day 7 评估总体进展:
├─ 任一方案达到 R²≥0.54
│  └─ ✅ 成功，继续叠加最优配置
│
├─ 组合多方案达到 R²≥0.52
│  └─ ⚠️  部分成功，投入生产但继续探索
│
└─ 所有方案 R²<0.50
   └─ ❌ 当前架构瓶颈，考虑:
      - Transformer backbone
      - 预训练 + 微调
      - 混合模型 (DL特征 + XGBoost)
```

---

## 资源需求

### 计算资源
- GPU: 1× NVIDIA RTX 3090 (24GB)
- 预计总 GPU-hours: ~120 hours
- 并行策略: 不同实验串行，同一实验的多 seed 可并行

### 人力投入
- 深度学习工程师: 7 工作日
- 分工:
  - Day 1-2: 实现 + P1/P2 实验
  - Day 3-4: 数据增强实现 + P3 实验
  - Day 5-7: Multi-scale 实现 + P4 实验 + 总结

### 代码变更估算
| 模块 | 新增代码 | 修改代码 | 测试代码 |
|------|---------|---------|---------|
| P1 | 0 LOC | 10 LOC | 0 LOC |
| P2 | 0 LOC | 10 LOC | 0 LOC |
| P3 | 150 LOC | 30 LOC | 80 LOC |
| P4 | 200 LOC | 50 LOC | 100 LOC |
| **总计** | **350 LOC** | **100 LOC** | **180 LOC** |

---

## 风险管理

### 整体风险评估

| 风险类型 | 概率 | 影响 | 缓解措施 |
|---------|------|------|---------|
| 所有方案均无效 | 低 | 高 | 提前设置 Go/No-Go 阈值，Day 4 中期评估 |
| 过拟合加剧 | 中 | 中 | 监控 train-val gap，早停机制 |
| 实验环境故障 | 低 | 高 | 每日备份检查点，云端同步 |
| 时间超期 | 中 | 低 | P4 可作为可选项，前 3 项优先 |

### 应急预案

#### 场景 1: Day 4 中期评估无进展
- **触发条件**: P1+P2+P3 均 <+0.03 改进
- **应急方案**:
  - 取消 P4 (Multi-scale CNN)
  - 启动诊断分析: 数据质量检查、特征分布验证
  - 考虑转向 Transformer 或混合模型

#### 场景 2: 显存不足
- **触发条件**: P1/P4 OOM 错误
- **应急方案**:
  - 减小 batch_size: 32 → 24 → 16
  - 启用梯度累积: 累积 2-4 步模拟大 batch
  - 减少 TCN 层数或通道数

#### 场景 3: 训练不稳定 (loss NaN)
- **触发条件**: >10% runs 出现 NaN
- **应急方案**:
  - 降低学习率: 1e-3 → 5e-4
  - 增强梯度裁剪: 1.0 → 0.5
  - 检查数据预处理 (Z-score 统计量)

---

## 参考文献与相关工作

### TCN 容量扩张
- Bai et al. (2018). "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
- 经验: 在 UCR 时序数据集上，通道数翻倍平均提升 3-5% 准确率

### 数据增强
- Um et al. (2017). "Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring"
- 时序增强最佳实践: jitter + scaling + magnitude warping

### Multi-scale CNN
- Szegedy et al. (2015). "Going Deeper with Convolutions" (Inception)
- Cui et al. (2016). "Multi-Scale Convolutional Neural Networks for Time Series Classification"

---

## 附录

### A. 配置文件清单
```
run/conf/
├── p1_tcn_capacity.yaml          # P1 实验
├── p2_tcn_long_rf.yaml            # P2 实验
├── p3_data_augment.yaml           # P3 实验
├── p3_ablation_time_jitter.yaml   # P3 消融
├── p3_ablation_amplitude_scale.yaml
├── p3_ablation_gaussian_noise.yaml
├── p4_multiscale_cnn.yaml         # P4 实验
└── experiment_base.yaml           # 基础模板
```

### B. 输出目录结构
```
outputs/
├── p1_tcn_capacity/
│   ├── seed_20260623/
│   │   ├── checkpoints/
│   │   ├── train_metrics.json
│   │   ├── test_metrics.json
│   │   └── config.yaml
│   ├── seed_42/
│   ├── seed_1337/
│   └── summary.json
├── p2_tcn_long_rf/
├── p3_data_augment/
└── p4_multiscale_cnn/
```

### C. 性能跟踪表格
| 实验 | Mean R² | Std R² | Δ Baseline | 最佳组分改进 | Train时间 |
|------|---------|--------|-----------|-------------|---------|
| Baseline (P0-B) | 0.4844 | - | - | CO₂: +0.27 | 2 min/epoch |
| P1 | ? | ? | ? | ? | ? |
| P2 | ? | ? | ? | ? | ? |
| P3 | ? | ? | ? | ? | ? |
| P4 | ? | ? | ? | ? | ? |
| **最佳组合** | ? | ? | ? | ? | ? |

---

**文档版本**: v1.0
**创建日期**: 2026-06-24
**负责人**: DL 工程师
**审核状态**: 待审核
