# DL 模型改进计划

**当前基线**: CNN1DTCNFusionRegressor, overall R²=+0.4844 (P0-B 配置)

**当前最佳**: gaussian_noise 增强, overall R²=**+0.5907** (P3 消融实验结果)

**目标**: 突破 R²=0.60 门槛，在保留 CO₂ 正值前提下逼近 Ridge 的 0.71 性能

---

## 优先级与路线图（2026-06-25 更新：消融实验完成）

| 优先级        | 方案                     | 预期收益              | 实验成本 | 风险等级        | 状态                         |
| ---------- | ---------------------- | ----------------- | ---- | ----------- | -------------------------- |
| ~~**P1**~~ | ~~TCN 容量扩张（加宽）~~       | ~~+0.05 R²~~      | 1天   | ~~低~~ **高** | ❌ **失败** (R²: +0.48→-0.34) |
| **P2**     | TCN 感受野扩展（加深）          | +0.02-0.05 R²     | 1天   | **中**       | ⏸️ **暂缓**（待 P3-N 定型后单独验证） |
| **P3**     | 数据增强                   | +0.03-0.08 R²     | 2天   | 低→**中**    | ⚠️ **部分成功** (Full Aug: +0.50) |
| **P3-N**   | gaussian_noise 参数调优     | +0.05-0.12 R²     | 1天   | 中           | 🎯 **新主线** (单策略 +0.59)     |
| **P4**     | Multi-scale 声学编码器 + 频域 | +0.05-0.10 R²     | 3天   | 中           | 📋 待 P3-N 完成                |

**当前策略**: gaussian_noise 参数调优 → 在保留 CO₂ 正值前提下锁定最优噪声强度 → P4 Multi-scale 叠加

**关键认知更新**:
- 数据量 ~4000 无法支撑大容量模型（参数/样本比应 < 0.02） — P1 验证 ✅
- **P1 失败 ≠ 加深/RF 扩展方向证伪**：P1 改的是宽度（参数 O(C²)，4×），P2 计划改的是深度（参数 O(d)，~2×），两者参数量级、参数/样本比、失败模式都不同。P2 暂缓是排期问题，不是方向被否定 — 详见 §P2
- **gaussian_noise 是强力正则化器**，单独使用使整体 R² 从 +0.48→+0.59（+23%） — P3 消融验证 ✅
- **H₂/CH₄ "饱和"结论被推翻**：gaussian_noise 下 H₂=+0.83（+0.05），CH₄=+0.66（+0.23），之前没找到正确的正则化方法
- **CO₂ 与 gaussian_noise 存在冲突**：全局噪声覆盖了 CO₂ 的微弱信号（-0.23），需要针对性保护

---

## ❌ P1: TCN 容量扩张（已失败）

### 实验结果

**状态**: ❌ 完全失败 (2026-06-24)

**性能对比**:

- P0-B Baseline: test R² = +0.4844
- P1 容量扩张: test R² = -0.3360
- **性能退化**: -0.8204 (下降 169%)

### 失败原因分析

#### 1. 严重过拟合（主因）

- 数据量 ~4000 无法支撑 290K 参数
- 参数/样本比从 0.018 激增至 0.073 (4×)
- Test R² < 0 说明模型性能不如均值预测

#### 2. 正则化不足

- Dropout 0.30 远不足以防止 4× 容量扩张的过拟合
- 应至少 dropout ≥ 0.45 或引入更强正则化

#### 3. 数据量瓶颈

- 290K 参数需要至少 15,000+ 样本才能有效训练
- 经验法则：样本数应 > 参数数 × 10~50

### 经验教训

**技术认知**:

- ✅ **加宽**（O(C²) 参数增长）在 ~4000 样本下风险极高，参数/样本比 > 0.05 大概率训练不收敛
- ✅ 参数/样本比 < 0.02 是安全阈值（baseline 0.018 是临界值）
- ✅ Dropout 缩放规则：容量扩张 N× 需要 dropout 提升 √N×；P1 dropout 仅 +0.05 远不够
- ✅ 数据增强（特别是 gaussian_noise）优先于架构改进 — 已被 P3 消融证实
- ⚠️ **该结论限于"加宽 + 无增强"组合**，不直接适用于加深/RF 扩展（P2，参数 O(d) 线性增长）或"加深 + P3-N 增强"组合

**风险评估修正**:
| 风险 | 原估计 | 实际 | 备注 |
|------|--------|------|------|
| 过拟合 | 中 | **极高** | Dropout 0.30 远远不够 |
| 训练时间 | 低 | 低 | 训练时间不是问题，过拟合才是 |
| 显存不足 | 低 | 低 | 未出现 OOM |
| **数据量不足** | **未评估** | **极高** | **主要 blocker** |

### 详细报告

参见 `docs/整理归档/dl_iteration_plans/p1_analysis_report.md`

---

## ⏸️ P2: TCN 感受野扩展（暂缓，待 P3-N 定型后单独验证）

### 状态

⏸️ **暂缓执行** — 不是因为方向被证伪，而是当前主线（P3-N gaussian_noise 调优）边际收益更高。P2 需要在 P3-N 锁定最佳增强配置后，作为独立架构实验单独验证 RF 是否是瓶颈。

### 原计划

固定 TCN 通道宽度 64，将层数从 3 提升至 6，扩展感受野从 29 步至 125 步，验证任务是否需要长时序依赖。**与 P1（加宽）改的是不同的轴**。

### P1 失败 ≠ P2 一定失败：参数量级与失败模式都不同

| 维度 | P0-B baseline | P1（加宽）| P2 原计划（加深） |
|------|--------------|-----------|------------------|
| TCN 通道宽度 | 64 | **64 → 128** | 64（不变） |
| TCN 层数 | 3 | 3（不变） | **3 → 6** |
| Dilation 感受野 | 29 步 | 29 步 | **29 → 125 步** |
| 参数量 | ~73K | **~290K（4×）** | ~140K 量级（~2×） |
| 参数/样本比 | 0.018 | **0.073** | ~0.035 |

**两个关键差异**：

1. **参数增长量级不同**：卷积加宽是 O(C²)，加深是 O(d)。P1 把参数推到 4×、参数/样本比 0.073；P2 加深至多到 ~2×、参数/样本比 ~0.035，离 baseline 近，离 P1 远。把两者都标成"容量扩张"是把数量级差异抹掉了。
2. **失败模式归因尚有歧义**：`docs/整理归档/dl_iteration_plans/p1_analysis_report.md` 显示 P1 best epoch 的 val R² 始终为负（-0.286），val CH₄=-0.77、val CO₂=-0.51。典型"过拟合"应该是 train R² 高、val R² 落后，而 P1 是**连 val 均值预测水平都没达到**，更像训练未收敛到合理解（4× 加宽 + LR/init/dropout 未同步缩放），未必只是"数据量不够"。

### 新证据：P3-N gaussian_noise 已反推前提

P3 消融实验得到 gaussian_noise 单独使用 test R² = **+0.5907**（H₂=+0.83 / CH₄=+0.66），说明：

- 架构本身在 ~4000 样本下能被推到 0.59，**前提是有合适的正则化**。
- P1 当时所处的"无增强 + 4× 加宽 + dropout 0.30"三角，不等于 P2 即将所处的"P3-N 最佳增强 + 加深 + 同 dropout"三角。
- 把"数据量永远撑不起更复杂模型"作为 P2 暂停理由，已被 P3-N 部分推翻。

### 暂缓（非否决）原因

1. **边际收益排序**：当前 P3-N 已拿到 +0.10 R² 增量，把 CO₂ 从 -0.23 抢回正比 P2 预期的 +0.02~+0.05 更有价值。
2. **依赖关系**：P2 的正确 baseline 应是 P3-N 锁定的最佳增强配置，否则会复现 P1 的混杂归因。
3. **GPU 时间预算**：P3-N 调优 + Phase A 合成气编码并行推进时，P2 优先级排在后面。

### P2 重启的前置条件

P3-N 完成后、P2 重新启动前，必须满足：

1. ✅ P3-N 已锁定最佳 gaussian_noise 配置（noise_std + apply_prob），且 CO₂ R² ≥ 0；
2. ✅ 在该最佳配置下，3 层 baseline 复跑确认 overall R² 稳定在 ≥ 0.55；
3. ✅ P2 实验设计：固定 width=64，分组对比 4 层 / 5 层 / 6 层，叠加 P3-N 最佳增强；
4. ✅ 评估指标按组分拆分（CO₂ / H₂ / CH₄ / N₂），单独看 RF 扩展对慢扩散组分（CO₂）的敏感性，而不是只看 overall R²。

### P2 想验证的真正问题

> **RF=29 步是否够用？拉到 125 步是否带来 CO₂ / 慢扩散组分的提升？**

这是一个 **任务–RF 匹配** 问题，不是表征容量问题。即使 P2 失败，失败模式应该是 "RF 拉长后无收益（说明 29 步已够）" 或 "6 层带来梯度退化（需要 residual/norm 调整）"——和 P1 的 "4× 加宽 + 正则不足导致未收敛" 是两个独立故事。

#### 关键诊断指标

```python
# 分析不同组分对感受野的敏感性（在 P3-N 最佳增强下评估）
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

### 实验执行与结果（2026-06-25）

#### Full Aug（三策略组合）

**配置**: max_shift=5, amplitude_scale=[0.95,1.05], gaussian_noise_std=0.01, apply_prob=0.5

**结果**: Test R² = **+0.4998** (+0.0154 vs P0-B, +3.2%)

| 组分 | P0-B | Full Aug | Δ |
|------|------|----------|---|
| H₂ | +0.78 | +0.79 | +0.01 |
| CH₄ | +0.43 | +0.45 | +0.02 |
| CO₂ | +0.27 | **+0.36** | **+0.09** |
| N₂ | -0.01 | -0.02 | -0.01 |
| **Overall** | +0.48 | **+0.50** | +0.02 |

**训练过程**: 80 epochs 无过拟合（val_loss < train_loss, gap=-0.07），val R²=+0.532 > test R²=+0.500。CO₂ 在 epoch 40→79 持续改进（+0.0195/epoch），从 -0.38 逆转至 +0.38。Early Stopping 未触发（patience=10 但最后 10 epochs 无改进），LR 未衰减。

**初步判断**: ⚠️ 轻微改进但未达标（目标 ≥0.51），CO₂ 显著受益但 H₂/CH₄ 改进微弱。

#### 消融实验（单一策略验证）

| 实验 | Epochs | Test R² | H₂ | CH₄ | CO₂ | N₂ | vs P0-B |
|------|--------|---------|------|------|------|------|------|
| P0-B Baseline | 80 | +0.4844 | +0.78 | +0.43 | +0.27 | -0.01 | — |
| **gaussian_noise** | 72 | **+0.5907** | **+0.83** | **+0.66** | **-0.23** | +0.00 | +0.1063 |
| time_jitter | 60 | +0.3088 | +0.70 | +0.14 | +0.17 | -0.01 | -0.1756 |
| amplitude_scale | 34 | -0.6387 | -0.01 | -1.22 | -0.15 | -0.06 | -1.1231 |

**颠覆性发现**:
1. **gaussian_noise 单独使用是 P0-P3 所有实验中最优配置**，R²=+0.5907 突破所有历史目标线
2. **但 CO₂ 被毁灭**（+0.27→-0.23），高斯噪声覆盖了微弱信号
3. **H₂/CH₄ "饱和"结论错误** — gaussian_noise 正则化使 CH₄ 从 +0.43→+0.66（+53%）
4. **Full Aug 中 CO₂ +0.36 是三策略交互效应**，任何单一策略无法复现
5. **amplitude_scale 单独使用是灾难性的**（R²=-0.64），±5% 幅值变化破坏声学特征量值关系
6. **CH₄ 是对信号扰动最敏感的组分** — gaussian_noise 收益最大（+0.23），amplitude_scale 破坏最大（-1.65）

**策略修正**: 放弃"三策略同时加强"思路，改为 **gaussian_noise 为主体 + CO₂ 保护调优**。

### 当前状态（2026-06-25）

**下一步**: 噪声参数调优（P3-N）
- `gaussian_noise_std=0.005`（减半噪声，预期保留 H₂/CH₄ 增益 + CO₂ 转正）
- `gaussian_noise + time_jitter`（去 amplitude_scale，验证修正版组合）
- 若 CO₂ 转正 → 扫描 apply_prob/noise_std 参数空间
- 若 CO₂ 仍为负 → 考虑 gaussian_noise 为主体 + CO₂ 专用 head 或 loss 加权

### 技术方案（已实现）

#### 增强策略（已实现于 `src/dl/data/augmentation.py`）

1. **时间抖动**: max_shift=5, edge-padding
2. **幅度缩放**: scale_range=[0.95,1.05], apply_from_channel=8（波形通道）
3. **高斯噪声**: noise_std=0.01
4. **概率门控**: apply_prob（per-sample integral probability gate）
5. **可重现性**: per-worker `default_rng(augment_seed + worker_id)`

#### 配置文件

- `configs/experiment/dl_p3_full_aug.json` — 三策略组合
- `configs/experiment/dl_p3_ablation_time_jitter.json` — 时间抖动消融
- `configs/experiment/dl_p3_ablation_amplitude_scale.json` — 幅度缩放消融
- `configs/experiment/dl_p3_ablation_gaussian_noise.json` — 高斯噪声消融

#### 验收标准（修正）

| 标准 | 阈值 | 状态 |
|------|------|------|
| ✅ **显著改进** | test R² ≥ 0.51 且 train-val gap 减小 >0.03 | gaussian_noise +0.59 达成 |
| ⚠️ **部分成功** | test R² 0.49-0.51 之间 | Full Aug +0.50 落入此区间 |
| ❌ **无效** | test R² 无提升 | time_jitter(+0.31) / amplitude_scale(-0.64) |
| 🎯 **新目标** | test R² ≥ 0.55 且 **CO₂ R² > 0** | gaussian_noise 调优目标 |

#### 风险与缓解（修正）

| 风险 | 概率 | 缓解措施 |
|------|------|---------|
| gaussian_noise 毁灭 CO₂ | **高（已验证）** | 降低 noise_std，CO₂ loss 加权，去 amplitude 干扰 |
| amplitude_scale 破坏性 | **高（已验证）** | 仅在其他策略配合下使用，或完全移除 |
| H₂/CH₄ 正则化收益丢失 | 中 | noise_std 精细调优，apply_prob 扫描 |
| 训练配置缺陷（Early Stop/LR） | 高 | patience 修正（ES:10→15, sched:8→5） |

---

---

## P4: Multi-scale 声学编码器 + 频域分支

### 前置条件（2026-06-25 更新）

P3 消融实验确认 **gaussian_noise 单独使用 R²=+0.5907** 为当前最佳（H₂=+0.83, CH₄=+0.66），但 CO₂ 毁灭（-0.23）。

**P4 启动条件**：
- gaussian_noise 参数调优完成 + CO₂ 转正（R² > 0）
- 或 CO₂ 无法转正但接受 gaussian_noise 单策略为最终 DL 基线

**P4 优先级**：P3-N（噪声调优）→ P4 Multi-scale（叠加架构改进）

### 目标

通过改进声学编码器架构（而非增加参数量）提升波形特征提取能力：

1. **时域多尺度并行**：捕获不同频率的声学特征
2. **频域分支**：显式提取频谱特征，补充时域信息
3. **参数量严格控制**：总参数量 ≤ 100K（相对 P0-B 的 73K 仅增加 37%）

### 设计原则（基于 P1 失败教训）

**✅ 做的**:

- 通过架构改进而非容量扩张提升性能
- 多分支并行共享参数，避免参数量爆炸
- 严格控制参数/样本比 < 0.025 (100K / 4000)

**❌ 不做的**:

- 单纯增加通道数或层数
- 引入大量新参数的复杂模块
- 未经数据增强就扩大模型容量

### 技术方案

#### 架构设计

##### 方案 A: 时域 Multi-scale（纯时域）

当前 DeepAcousticEncoder1D 使用单一 kernel_size=7，无法同时捕获高频和低频特征。

**改进: MultiScaleAcousticEncoder1D**

```python
class MultiScaleAcousticEncoder1D(nn.Module):
    """并行多尺度时域编码器

    参数量分析:
    - 3个分支 × [16,32,64,64] 通道 ≈ 3 × 12K = 36K
    - 融合层 (192 × 64) = 12K
    - 总计: ~48K (相对 baseline 单分支 16K 增加 32K)
    """

    def __init__(
        self,
        waveform_length: int,
        embedding_dim: int = 64,
        kernels: list[int] = [3, 7, 15],  # 小/中/大尺度
        channels: list[int] = [16, 32, 64, 64],
        dropout: float = 0.15,
    ):
        super().__init__()

        # 为每个 kernel 创建独立分支
        self.branches = nn.ModuleList([
            self._build_branch(k, channels, dropout)
            for k in kernels
        ])

        # 融合多尺度特征
        branch_dim = channels[-1] * 2 + 1  # avg + max + log_amp = 129-d per branch
        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * len(kernels), embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def _build_branch(self, kernel_size, channels, dropout):
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
                nn.Dropout(dropout),
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
            encoded = branch(flat)  # (B*T, C, L')
            avg = F.adaptive_avg_pool1d(encoded, 1).squeeze(-1)
            mx = F.adaptive_max_pool1d(encoded, 1).squeeze(-1)
            log_amp = torch.log1p(flat.abs().mean(dim=-1))
            branch_features.append(torch.cat([avg, mx, log_amp], dim=-1))

        # 融合
        fused = torch.cat(branch_features, dim=-1)  # (B*T, 129*3=387)
        embedding = self.fusion(fused)  # (B*T, 64)
        return embedding.reshape(B, T, -1)
```

**物理直觉**:

- **H₂**: 高频振动 → kernel=3 捕获快速变化
- **CH₄**: 中频特性 → kernel=7 捕获中等时间尺度
- **CO₂**: 低频振荡 → kernel=15 捕获缓慢变化

##### 方案 B: 时域 + 频域（推荐）

在方案 A 基础上增加显式频域分支，捕获频谱特征。

**改进: HybridAcousticEncoder1D**

```python
class HybridAcousticEncoder1D(nn.Module):
    """混合时频域编码器

    参数量分析:
    - 时域分支 (3-scale): ~48K
    - 频域分支 (FFT → MLP): ~8K
    - 时频融合: ~4K
    - 总计: ~60K (相对 baseline 16K 增加 44K)

    总模型参数量: 60K (acoustic) + 20K (slow) + 15K (TCN) + 5K (head) = 100K
    """

    def __init__(
        self,
        waveform_length: int,
        embedding_dim: int = 64,
        time_kernels: list[int] = [3, 7, 15],
        time_channels: list[int] = [16, 32, 64, 64],
        freq_hidden: int = 64,
        dropout: float = 0.15,
        n_fft: int = 128,  # FFT 窗口大小
        n_mels: int = 32,  # Mel 滤波器数量
    ):
        super().__init__()

        # 时域多尺度分支
        self.time_encoder = MultiScaleAcousticEncoder1D(
            waveform_length, 
            embedding_dim // 2,  # 输出 32-d
            time_kernels, 
            time_channels, 
            dropout
        )

        # 频域分支 (FFT → Mel-spectrogram → MLP)
        self.n_fft = n_fft
        self.n_mels = n_mels

        # Mel 滤波器 (固定参数，不训练)
        self.register_buffer(
            'mel_filterbank',
            self._create_mel_filterbank(waveform_length, n_fft, n_mels)
        )

        # 频域特征提取
        freq_input_dim = n_mels * 3  # mean + std + max
        self.freq_mlp = nn.Sequential(
            nn.Linear(freq_input_dim, freq_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(freq_hidden, embedding_dim // 2),  # 输出 32-d
        )

        # 时频融合
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _create_mel_filterbank(self, waveform_length, n_fft, n_mels):
        """创建 Mel 滤波器"""
        # 简化版，实际应使用 torchaudio.transforms.MelScale
        import torchaudio
        return torchaudio.functional.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=8000.0,  # 假设采样率 16kHz
            n_mels=n_mels,
            sample_rate=16000,
        ).T

    def _extract_freq_features(self, waveform):
        """提取频域特征"""
        # waveform: (B*T, L)
        # FFT
        fft = torch.fft.rfft(waveform, n=self.n_fft, dim=-1)
        magnitude = torch.abs(fft)  # (B*T, n_fft//2+1)

        # Mel 滤波
        mel_spec = torch.matmul(magnitude, self.mel_filterbank)  # (B*T, n_mels)
        mel_spec = torch.log1p(mel_spec)  # Log-scale

        # 统计特征
        mean = mel_spec.mean(dim=-1)
        std = mel_spec.std(dim=-1)
        mx = mel_spec.max(dim=-1)[0]

        return torch.stack([mean, std, mx], dim=-1).reshape(waveform.shape[0], -1)

    def forward(self, waveform):
        # waveform: (B, T, L)
        B, T, L = waveform.shape
        flat = waveform.reshape(B*T, L).float() / 32767.0

        # 时域特征
        time_feat = self.time_encoder(waveform)  # (B, T, 32)

        # 频域特征
        freq_feat_flat = self._extract_freq_features(flat)  # (B*T, n_mels*3)
        freq_feat = self.freq_mlp(freq_feat_flat).reshape(B, T, -1)  # (B, T, 32)

        # 时频融合
        combined = torch.cat([time_feat, freq_feat], dim=-1)  # (B, T, 64)
        return self.fusion(combined)
```

**频域分支优势**:

- 显式捕获频谱特征，补充时域卷积可能遗漏的信息
- Mel 滤波器模拟人耳感知，适合声学信号
- 参数量小（仅 8K），性价比高

#### 配置变更

```yaml
# 方案 A: 纯时域 Multi-scale
model:
  name: cnn1d_tcn_fusion
  kwargs:
    acoustic_encoder_type: "multiscale"
    acoustic_kernels: [3, 7, 15]
    acoustic_channels: [16, 32, 64, 64]
    acoustic_dropout: 0.15
    tcn_channels: [64, 64, 64]
    tcn_dropout: 0.25

# 方案 B: 时域 + 频域（推荐）
model:
  name: cnn1d_tcn_fusion
  kwargs:
    acoustic_encoder_type: "hybrid"
    time_kernels: [3, 7, 15]
    time_channels: [16, 32, 64, 64]
    freq_hidden: 64
    n_fft: 128
    n_mels: 32
    acoustic_dropout: 0.15
    tcn_channels: [64, 64, 64]
    tcn_dropout: 0.25
```

#### 实施步骤

1. **实现编码器**:
   
   - `src/dl/models/multiscale_encoder.py`: MultiScaleAcousticEncoder1D
   - `src/dl/models/hybrid_encoder.py`: HybridAcousticEncoder1D

2. **修改 CNN1DTCNFusionRegressor**:
   
   ```python
   def __init__(self, ..., acoustic_encoder_type="deep", **encoder_kwargs):
       if acoustic_encoder_type == "deep":
           self.encoder = DeepAcousticEncoder1D(...)
       elif acoustic_encoder_type == "multiscale":
           self.encoder = MultiScaleAcousticEncoder1D(...)
       elif acoustic_encoder_type == "hybrid":
           self.encoder = HybridAcousticEncoder1D(...)
   ```

3. **单元测试**:
   
   ```python
   def test_encoder_output_shape():
       encoder = HybridAcousticEncoder1D(1000, embedding_dim=64)
       x = torch.randn(2, 10, 1000)  # (B=2, T=10, L=1000)
       out = encoder(x)
       assert out.shape == (2, 10, 64)
   
   def test_encoder_parameter_count():
       encoder = HybridAcousticEncoder1D(1000)
       params = sum(p.numel() for p in encoder.parameters())
       assert params < 70000, f"Too many params: {params}"
   ```

4. **消融实验**:
   
   - Baseline: kernel [7] (P0-B)
   - A1: Multi-scale [3, 7]
   - A2: Multi-scale [3, 7, 15]
   - B1: Hybrid (time [3,7] + freq)
   - B2: Hybrid (time [3,7,15] + freq)

5. **训练**: seed=20260623 × 80 epochs

#### 验收标准

| 标准           | 阈值                   | 条件               |
| ------------ | -------------------- | ---------------- |
| ✅ **显著改进**   | test R² ≥ 0.54       | 至少 +0.10 相对 P0-B |
| ⚠️ **中等改进**  | 0.50 ≤ R² < 0.54     | +0.02~0.10，可接受   |
| ❌ **无效**     | R² < 0.50            | 未达到 baseline     |
| 🔍 **过拟合检查** | train-val gap < 0.10 | 必须满足             |

#### 理论依据

**物理直觉**:

- 不同气体混合产生不同频率和时间尺度的声学特征
- H₂: 低分子量 → 高频振动、快速扩散
- CH₄: 中等分子量 → 中频特性
- CO₂: 较高分子量 → 低频振荡、慢扩散

**经验支持**:

- Inception (Szegedy et al. 2015): 多尺度并行在视觉任务中验证有效
- SoundNet (Aytar et al. 2016): 时频混合编码器在声学任务中优于纯时域
- Mel-spectrogram: 语音识别标准特征，证明频域信息的价值

#### 风险与缓解

| 风险            | 概率  | 影响  | 缓解措施                                       |
| ------------- | --- | --- | ------------------------------------------ |
| 参数量超标 (>100K) | 中   | 高   | 严格限制分支数 ≤ 3，监控参数统计                         |
| 频域分支无收益       | 中   | 低   | 消融实验验证，可回退纯时域方案 A                          |
| 训练不稳定         | 低   | 中   | 每个分支独立 BN，融合层加 LayerNorm                   |
| 过拟合（P1 教训）    | 中   | 高   | 参数量 < 100K，dropout 0.15→0.20，必须先完成 P3 数据增强 |
| 显存不足          | 低   | 低   | FFT 在 CPU 计算，减小 batch_size 至 12            |

**关键约束** (基于 P1 失败):

- ✅ 参数/样本比 < 0.025 (100K / 4000)
- ✅ 必须在 P3 数据增强完成后执行
- ✅ 训练前验证参数量 < 100K
- ✅ 监控 train-val gap，及时早停

#### 预期收益分析

**乐观场景** (方案 B 成功):

- Multi-scale 时域: +0.03 R²
- 频域分支补充: +0.02 R²
- 总收益: +0.05 R² → test R² ≈ 0.53

**保守场景** (方案 A 部分成功):

- Multi-scale 时域: +0.02 R²
- 总收益: +0.02 R² → test R² ≈ 0.50

**失败场景** (收益 < 0.02):

- 转向 DL 特征 + XGBoost 混合模型
- 或承认 DL 在当前数据量下无法超越 Ridge

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

| 日期 | 里程碑 | 检查点 | 实际结果 |
|------|--------|--------|---------|
| 2026-06-24 | P1 完成 | R² ≥ 0.53? | ❌ R²=-0.34，完全失败 |
| — | P2 暂缓 | RF 扩展有效? | ⏸️ 暂缓（非证伪），待 P3-N 定型后单独验证（详见 §P2） |
| 2026-06-25 | P3 Full Aug | 数据增强收益? | ⚠️ R²=+0.4998 (+0.02)，轻微改进 |
| 2026-06-25 | P3 消融实验 | 单策略贡献? | 🎯 gaussian_noise R²=+0.5907（历史最高） |
| 待执行 | P3-N 噪声调优 | CO₂ 转正 + R²≥0.55? | 待定 |
| 待执行 | P4 Multi-scale | R²≥0.58? | 待 P3-N 完成 |

### 最终决策树（2026-06-25 修正）

```
P3 消融实验已完成:
├─ gaussian_noise 单策略 R²=+0.5907 ✅
│  └─ 🎯 确认 gaussian_noise 为最强正则化器
│     下一步: P3-N 噪声调优 (CO₂ 保护)
│
├─ Full Aug CO₂=+0.36 是三策略交互效应
│  └─ ⚠️  任何单一策略无法复现, 交互路线暂停
│
├─ time_jitter(+0.31) / amplitude_scale(-0.64)
│  └─ ❌ 单独使用均无效, 仅作为辅助策略
│
└─ 下一步决策分支:
   ├─ P3-N CO₂ 转正 + R²≥0.55
   │  └─ ✅ P3 成功, 叠加 P4 Multi-scale
   │
   ├─ P3-N CO₂ 仍为负但 R²≥0.55
   │  └─ ⚠️  接受 CO₂ 不可学的 gaussian_noise 基线, 启动 P4
   │
   └─ P3-N 调优后 CO₂ 仍为负且 R²<0.51
      └─ ❌ 回退 Full Aug 配置 (+0.50, CO₂=+0.36)
         或转向混合模型 (DL embedding + XGBoost)
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

### 代码变更估算（实际）

| 模块 | 新增代码 | 修改代码 | 测试代码 | 状态 |
|------|---------|---------|---------|------|
| P1 | 0 LOC | 10 LOC | 0 LOC | ✅ 已完成 |
| P2 | 0 LOC | 0 LOC | 0 LOC | ⏸️ 暂缓（待 P3-N 定型后单独验证） |
| P3 实现 | 150 LOC | 50 LOC | 110 LOC | ✅ 已完成 |
| P3 配置 | 4 JSON | — | — | ✅ 已完成 |
| P3 分析报告 | 2 文件 | — | — | ✅ 已完成 |
| P3-N 调优 | 待定 | 待定 | — | ⏳ 待执行 |
| P4 | 200 LOC | 50 LOC | 100 LOC | 📋 待 P3-N |
| **已完成** | **150 LOC** | **60 LOC** | **110 LOC** | — |

---

## 风险管理

### 整体风险评估

| 风险类型    | 概率  | 影响  | 缓解措施                        |
| ------- | --- | --- | --------------------------- |
| 所有方案均无效 | 低   | 高   | 提前设置 Go/No-Go 阈值，Day 4 中期评估 |
| 过拟合加剧   | 中   | 中   | 监控 train-val gap，早停机制       |
| 实验环境故障  | 低   | 高   | 每日备份检查点，云端同步                |
| 时间超期    | 中   | 低   | P4 可作为可选项，前 3 项优先           |

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

| 实验 | Test R² | H₂ | CH₄ | CO₂ | N₂ | Δ P0-B | Epochs |
|------|---------|------|------|------|------|------|------|
| P0-B Baseline | +0.4844 | +0.78 | +0.43 | +0.27 | -0.01 | — | 80 |
| P1 TCN 扩张 | -0.3360 | — | — | — | — | -0.82 | 40 |
| P3 Full Aug | +0.4998 | +0.79 | +0.45 | +0.36 | -0.02 | +0.02 | 80 |
| P3 gaussian_noise | **+0.5907** | **+0.83** | **+0.66** | **-0.23** | +0.00 | **+0.11** | 72 |
| P3 time_jitter | +0.3088 | +0.70 | +0.14 | +0.17 | -0.01 | -0.18 | 60 |
| P3 amplitude_scale | -0.6387 | -0.01 | -1.22 | -0.15 | -0.06 | -1.12 | 34 |
| P3-N 噪声调优 | ? | ? | ? | ? | ? | ? | ? |
| P4 Multi-scale | ? | ? | ? | ? | ? | ? | ? |
| **目标** | ≥0.60 | — | — | >0 | — | ≥0.12 | — |

---

**文档版本**: v2.0
**最后更新**: 2026-06-25 (P3 消融实验完成，策略重大调整)
**负责人**: DL 工程师
**审核状态**: P3 消融分析已审核，P3-N 噪声调优待执行
