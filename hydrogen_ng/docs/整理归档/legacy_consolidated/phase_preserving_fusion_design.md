# Phase-Preserving Fusion 设计方案

基于文献调研（2025-2026），针对掺氢天然气 N2 浓度预测的多阶段 DL 架构设计。

## 核心问题

当前 `cnn1d_tcn_fusion` 单窗口裁剪方案失败原因：
- **时间信息丢失**：exposure/recovery 单窗口丢失完整过程上下文
- **阶段边界模糊**：裁剪破坏了相邻阶段的连续性
- **特征不对齐**：不同阶段特征直接拼接无时序对齐

ML 多窗口成功（R²=0.7121）证明 **full+exposure+recovery 三窗口互补信息有效**。

## 文献启发

### 1. 多尺度时序建模（MuST, AMS-PAFN）
- **多时间分辨率金字塔**：对同一输入创建多个采样率序列
- **Multi-Temporal Attention**：跨尺度信息交互
- **应用**：手术阶段识别、癫痫检测

### 2. 相位感知融合（Phase-Aware Fusion）
- **相位差异建模**：显式计算不同分支的时序偏移
- **加权同步**：增强同步特征，抑制不对齐噪声
- **应用**：EEG 信号、CT 多期成像

### 3. 因果 TCN 多阶段架构（TeCNO, MS-TCN）
- **因果膨胀卷积**：保持时序因果性，扩大感受野
- **多阶段精炼**：第一阶段预测 → 后续阶段精炼
- **应用**：手术阶段识别、动作分割

## 设计方案：PhasePreservingTCN

### 架构概览

```
输入：full window (原始完整信号)
  │
  ├─> Branch 1: Phase Exposure Extractor (phase token: "exposure")
  ├─> Branch 2: Phase Steady Extractor (phase token: "steady")  
  └─> Branch 3: Phase Recovery Extractor (phase token: "recovery")
      │
      ├─> 每个 branch：
      │     CNN1D feature extraction
      │     + Phase-specific TCN (causal dilated conv)
      │     + Phase token embedding
      │
      └─> Phase-Aware Fusion Module:
            - Multi-head cross-attention (exposure ↔ steady ↔ recovery)
            - Phase alignment via learnable phase shift
            - Adaptive weighting
      │
      v
  Temporal Consistency Module (long-term TCN)
      │
      v
  Regression Head (H2, CH4, CO2, N2)
```

### 核心模块

#### 1. Phase-Specific Extractor
```python
class PhaseSpecificExtractor(nn.Module):
    """单阶段特征提取器"""
    def __init__(self, phase: str, input_dim: int, hidden_dim: int):
        # phase ∈ {"exposure", "steady", "recovery"}
        self.phase_token = nn.Parameter(torch.randn(1, hidden_dim))
        self.cnn = CNN1D(input_dim, hidden_dim)
        self.tcn = CausalTCN(hidden_dim, num_layers=4, kernel_size=3)
    
    def forward(self, x_full):
        # x_full: (B, T, C) 完整窗口
        feat = self.cnn(x_full)  # (B, T', D)
        feat = feat + self.phase_token  # 注入阶段标识
        feat = self.tcn(feat)  # 因果时序建模
        return feat
```

#### 2. Phase-Aware Fusion Module
```python
class PhaseAwareFusion(nn.Module):
    """相位感知融合模块"""
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads)
        self.phase_shift_estimator = nn.Linear(hidden_dim, 1)  # 学习相位偏移
        self.adaptive_weight = nn.Linear(hidden_dim * 3, 3)  # 自适应权重
    
    def forward(self, feats: Dict[str, Tensor]):
        # feats = {"exposure": (B,T,D), "steady": (B,T,D), "recovery": (B,T,D)}
        
        # 1. 跨阶段交互（Multi-head Attention）
        exp, std, rec = feats["exposure"], feats["steady"], feats["recovery"]
        exp_attn, _ = self.cross_attn(exp, torch.cat([std, rec], 1), torch.cat([std, rec], 1))
        std_attn, _ = self.cross_attn(std, torch.cat([exp, rec], 1), torch.cat([exp, rec], 1))
        rec_attn, _ = self.cross_attn(rec, torch.cat([exp, std], 1), torch.cat([exp, std], 1))
        
        # 2. 相位对齐（学习时序偏移）
        shift_exp = self.phase_shift_estimator(exp_attn).squeeze(-1)  # (B, T)
        shift_std = self.phase_shift_estimator(std_attn).squeeze(-1)
        shift_rec = self.phase_shift_estimator(rec_attn).squeeze(-1)
        
        exp_aligned = self.apply_shift(exp_attn, shift_exp)
        std_aligned = self.apply_shift(std_attn, shift_std)
        rec_aligned = self.apply_shift(rec_attn, shift_rec)
        
        # 3. 自适应加权融合
        concat = torch.cat([exp_aligned, std_aligned, rec_aligned], dim=-1)  # (B, T, 3D)
        weights = F.softmax(self.adaptive_weight(concat), dim=-1)  # (B, T, 3)
        
        fused = (weights[:, :, 0:1] * exp_aligned +
                 weights[:, :, 1:2] * std_aligned +
                 weights[:, :, 2:3] * rec_aligned)  # (B, T, D)
        
        return fused
    
    def apply_shift(self, x, shift):
        # 沿时间轴平移（插值实现）
        # 简化版：使用 roll + 线性插值
        pass
```

#### 3. Temporal Consistency Module
```python
class TemporalConsistencyModule(nn.Module):
    """长期时序一致性模块"""
    def __init__(self, hidden_dim: int):
        self.long_tcn = CausalTCN(hidden_dim, num_layers=6, dilation_base=2)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # x: (B, T, D) 融合后特征
        return self.long_tcn(self.dropout(x))
```

### 完整模型

```python
@register_model("phase_preserving_tcn")
class PhasePreservingTCN(BaseRegressor):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Phase-specific extractors
        self.exp_extractor = PhaseSpecificExtractor("exposure", input_dim, hidden_dim)
        self.std_extractor = PhaseSpecificExtractor("steady", input_dim, hidden_dim)
        self.rec_extractor = PhaseSpecificExtractor("recovery", input_dim, hidden_dim)
        
        # Phase-aware fusion
        self.fusion = PhaseAwareFusion(hidden_dim, num_heads=4)
        
        # Temporal consistency
        self.tcm = TemporalConsistencyModule(hidden_dim)
        
        # Regression head
        self.head = nn.Linear(hidden_dim, 4)  # H2, CH4, CO2, N2
    
    def forward(self, x):
        # x: (B, T, C) 完整窗口输入
        
        # 三阶段并行提取
        exp_feat = self.exp_extractor(x)
        std_feat = self.std_extractor(x)
        rec_feat = self.rec_extractor(x)
        
        # 相位感知融合
        fused = self.fusion({
            "exposure": exp_feat,
            "steady": std_feat,
            "recovery": rec_feat
        })
        
        # 长期时序一致性
        fused = self.tcm(fused)
        
        # 回归预测
        out = self.head(fused[:, -1, :])  # 取最后时刻
        return out
```

## 与 ML 多窗口的对比

| 维度 | ML (ridge_multiwindow) | DL (PhasePreservingTCN) |
|------|------------------------|-------------------------|
| 输入 | 手工拼接 full+exp+rec 特征 | 完整窗口，模型自动学习阶段边界 |
| 阶段划分 | 显式裁剪窗口 | 隐式学习（phase token + attention） |
| 时序建模 | 无（线性模型） | 因果 TCN + 长期依赖 |
| 相位对齐 | 无（假设完美对齐） | 显式相位偏移学习 |
| 参数量 | 极少 (~1K) | 中等 (~500K) |

## 实现计划

### 最小验证版本（2-3天）
1. 实现 `PhaseSpecificExtractor`（复用 `CNN1D` + 新增 `CausalTCN`）
2. 简化 `PhaseAwareFusion`（先用 concat + MLP，跳过 phase shift）
3. 复用 `TemporalConsistencyModule`（已有 TCN 实现）
4. 在 `multiwindow_n2` 数据集上验证

### 完整版本（5-7天）
1. 补全 `apply_shift` 相位对齐（基于可学习插值）
2. 实验多头数、隐藏维度、TCN 层数超参
3. 消融实验：
   - 无 phase token
   - 无 cross-attention
   - 无 phase shift
4. 对比 baseline：`cnn1d_tcn_fusion`、`ridge_multiwindow_all_modalities`

## 预期性能

**保守目标**：
- N2 test R² > 0.50（当前 DL 最好 ~0.30）
- 不退化 H2/CH4/CO2

**理想目标**：
- N2 test R² > 0.65（接近 ML 0.71）
- 泛化性优于 ML（extrapolation R² gain）

## 风险与对策

| 风险 | 对策 |
|------|------|
| Phase token 不收敛 | 改用 position encoding 或固定 one-hot |
| Cross-attention 计算量大 | 降低 num_heads，或改用 linear attention |
| Phase shift 学习失败 | 回退到固定 window，只保留 attention fusion |
| 过拟合（参数多） | Dropout + early stopping + weight decay |

## 参考文献

1. **MuST (MICCAI 2024)**: Multi-Scale Transformers for Surgical Phase Recognition
2. **AMS-PAFN (Frontiers Neurology 2025)**: Adaptive Multi-Scale Phase-Aware Fusion Network
3. **TeCNO (MICCAI 2020)**: Causal TCN for Online Surgical Phase Recognition
4. **MS-TCN (CVPR 2019)**: Multi-Stage Temporal Convolutional Network for Action Segmentation
5. **PULSE (arXiv 2025)**: Generative Phase Evolution for Time Series Forecasting
