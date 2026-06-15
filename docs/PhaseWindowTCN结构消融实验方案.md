# PhaseWindowTCN 结构消融实验方案

> 更新日期：2026-06-15
> 状态：待执行
> 目标：在已完成 `gas_head` 与 `free_component_mse` 负结果验证后，聚焦结构消融，判断 DL 是否还能稳定改善 N2

## 1. 问题背景

当前项目中，传统 ML 的多窗口路线已经成立：

- `ridge_multiwindow_all_modalities` 已在 `full + exposure + recovery` 组合上显著提升 N2
- 该结果说明三窗口信息本身有效
- DL 主线 `PhaseWindowTCN` 的瓶颈不再是“有没有信号”，而是“结构是否能把信号组织好”

PhaseWindowTCN 近期已验证的事实：

- `gas_head` 可以修复闭包输出问题，`sum_abs_error` 已接近 0
- `free_component_mse` 不能单独把 `N2` 拉正
- 因此，继续围绕 loss 小调参的收益已有限

结论是：**下一步只做结构消融，不再继续围绕 head/loss 反复微调。**

## 2. 已知事实

### 2.1 代码现状

当前实现已经具备以下能力：

- `src/dl/models/phase_window_tcn.py`
  - 支持 `PhaseWindowTCNRegressor`
  - 支持 `share_window_encoder`
  - 支持 `output_mode in {"raw4", "softmax100", "gas_head"}`
- `src/dl/training/losses.py`
  - 已注册 `free_component_mse`
- `src/pipeline/experiment_config.py`
  - 已支持 `phase_windows`
  - 已限制 DL 与 ML 的窗口语义分离

这意味着当前不是“补实现”，而是“选下一轮实验方向”。

### 2.2 文献依据

当前方案只保留与本项目最相关、且可直接指导实验顺序的文献结论。

1. **TCN 有效感受野需要实测，不应只看理论堆叠**

   Bai 等对 TCN 的序列建模结果表明，卷积序列模型能建模长依赖，但有效记忆长度是关键，不应默认浅层 TCN 足够。

   参考：Bai, Kolter, Koltun, 2018, *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*  
   [https://arxiv.org/abs/1803.01271](https://arxiv.org/abs/1803.01271)

2. **多任务损失平衡是成熟问题，但不是先验万能药**

   GradNorm 证明多任务训练中可以通过动态梯度平衡改善收敛，但它解决的是“任务竞争”，不是“输入表征不足”。

   参考：Chen et al., 2017, *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks*  
   [https://arxiv.org/abs/1711.02257](https://arxiv.org/abs/1711.02257)

3. **不确定性加权也属于 loss 平衡手段**

   这类方法适合做对照，但不应替代结构验证。

4. **PAF-Net 类方法适合“相位错位 + 频带耦合”的多过程场景**

   PAF-Net 的核心是相位对齐、DCT 频率分解和频率解耦 cross-attention。它能支持“相位信息很重要”的判断，但不直接证明本项目需要完整频率解耦模块。

   参考：Luo et al., 2025, *PAF-Net: Phase-Aligned Frequency Decoupling Network for Multi-Process Manufacturing Quality Prediction*  
   [https://arxiv.org/abs/2507.22840](https://arxiv.org/abs/2507.22840)

## 3. 当前判断

### 3.1 已排除的方向

- 继续只调 `gas_head` 和 `free_component_mse`
- 直接上复杂 cross-attention / DCT 解耦
- 先做大模型，再事后解释结果

### 3.2 仍值得验证的方向

1. `share_window_encoder=false`
2. 更深的 TCN 感受野
3. 仅在前两项有信号后，再考虑 gated fusion 或轻量 attention

### 3.3 不建议作为主线的方向

- 完整 PAF-Net 复刻
- 频域分解 + 对齐 + cross-attention 一步到位
- 多头动态损失权重作为主改进

## 4. 实验目标

### 4.1 主目标

判断以下问题：

1. 共享窗口 encoder 是否确实稀释了 `full / exposure / recovery` 的相位差异
2. 更深 TCN 是否能补足当前感受野不足
3. 结构消融是否能让 `N2` 在 test 和 extrapolation 上同时转正

### 4.2 成功标准

相对当前 `phase_window_tcn_gas_free`：

- `test x_N2 R2` 必须提升
- `extrapolation x_N2 R2` 必须提升
- `sum_abs_error` 维持接近 0
- 其他三组分不能系统性退化

建议最低门槛：

- `test x_N2 R2 > 0`
- `extrapolation x_N2 R2 > 0`
- `macro RMSE` 不高于当前负结果

### 4.3 停止条件

若以下任一成立，停止 DL 主线继续投入：

- `share_window_encoder=false` 仍不能让 N2 转正
- 深 TCN 仍不能带来稳定提升
- 结构改动明显增大复杂度，但验证集和测试集都无收益

这种情况下，正式主线继续保持 `ridge_multiwindow_all_modalities`。

## 5. 实验矩阵

### 5.1 第一批：最小结构消融

| 实验名 | 主要改动 | 目的 |
|---|---|---|
| `phase_window_tcn_gas_free_split` | `share_window_encoder=false` | 验证相位窗口是否需要独立编码器 |
| `phase_window_tcn_gas_free_deep` | `tcn_channels=[64,64,64,64,64]` | 验证感受野是否不足 |
| `phase_window_tcn_gas_free_split_deep` | 独立 encoder + 深 TCN | 验证二者是否有叠加收益 |

### 5.2 第二批：轻量融合对照

仅当第一批有正向信号时再做：

| 实验名 | 主要改动 | 目的 |
|---|---|---|
| `phase_window_tcn_gas_free_gated` | 加 gated fusion | 验证窗口级加权是否比纯 concat 更好 |
| `phase_window_tcn_gas_free_attn` | 轻量 attention | 验证窗口交互是否真有帮助 |

### 5.3 暂缓项

- 完整频率解耦 cross-attention
- DCT 分解
- 相位对齐插值
- phase token 扩展

这些都只有在第一批结果明确支持时再考虑。

## 6. 推荐配置

### 6.1 基线延续

保持当前基线不变：

```json
{
  "name": "phase_window_tcn_gas_free",
  "model": "phase_window_tcn",
  "modalities": ["slow", "ultrasonic", "fiber_mic"],
  "phase_windows": [
    null,
    {"kind": "phase", "value": "exposure"},
    {"kind": "phase", "value": "recovery"}
  ],
  "loss": "free_component_mse",
  "model_kwargs": {
    "window_count": 3,
    "output_mode": "gas_head",
    "waveform_embedding_dim": 64,
    "acoustic_channels": [16, 32, 64, 64],
    "acoustic_kernel_size": 7,
    "acoustic_dropout": 0.15,
    "slow_hidden_dim": 32,
    "slow_embedding_dim": 64,
    "tcn_channels": [64, 64, 64],
    "tcn_kernel_size": 3,
    "tcn_dropout": 0.25,
    "shared_hidden_dims": [128, 64]
  }
}
```

### 6.2 Split 版本

核心只改一项：

```json
{
  "name": "phase_window_tcn_gas_free_split",
  "model_kwargs": {
    "share_window_encoder": false
  }
}
```

### 6.3 Deep 版本

核心只改一项：

```json
{
  "name": "phase_window_tcn_gas_free_deep",
  "model_kwargs": {
    "tcn_channels": [64, 64, 64, 64, 64]
  }
}
```

## 7. 运行顺序

1. 先跑 `phase_window_tcn_gas_free_split`
2. 再跑 `phase_window_tcn_gas_free_deep`
3. 若两者任一有效，再跑 `phase_window_tcn_gas_free_split_deep`
4. 若仍无效，停止 DL 继续扩展

## 8. 验收口径

建议用以下口径判断是否值得继续：

- `test x_N2 R2`
- `extrapolation x_N2 R2`
- `macro RMSE`
- `sum_abs_error`
- 其他三组分的 `R2` 退化幅度

其中，`N2` 必须是首要指标，但不能以明显牺牲其他组分为代价。

## 9. 结论

当前最合理的判断是：

1. `gas_head + free_component_mse` 已经证明闭包问题不是主矛盾
2. `PhaseWindowTCN` 的剩余问题更像是窗口表征和时序感受野不足
3. 所以下一步应该做结构消融，而不是继续做复杂 loss 或直接上大注意力模块

本方案就是 DL 线的最后一轮小规模判定实验。若仍无明显正结果，应收束并把正式主线固定在 `ridge_multiwindow_all_modalities`。

## 10. 备选计划

本节保留前期讨论过的完整候选方案。它们不进入第一批执行矩阵，但作为后续可恢复路线记录下来，避免后续重复调研。

### 10.1 备选方案 A：分离式窗口编码 + 频率解耦注意力

#### 核心思路

该方案来自 PAF-Net 类多过程时序建模思路，目标是同时解决：

- 不同窗口之间的相位差异
- 潜在时序滞后
- 不同频段中有效信息与噪声混杂

候选结构如下：

```python
phase_encoders = {
    "full": TCNEncoder(n_blocks=3, shared=False),
    "exposure": TCNEncoder(n_blocks=3, shared=False),
    "recovery": TCNEncoder(n_blocks=3, shared=False),
}

encoded = [phase_encoders[name](x[name]) for name in ("full", "exposure", "recovery")]
aligned = phase_correlation_alignment(encoded)
fused = frequency_decoupled_cross_attention(aligned, n_frequencies=5)
```

#### 模块拆分

1. **相位独立编码**

   每个窗口使用独立 encoder，避免 `share_window_encoder=true` 把 full、exposure、recovery 的相位差异压成折中表征。

2. **频率域相位对齐**

   对不同窗口的时序特征做 phase-correlation alignment，尝试校正 exposure/recovery 与 full 之间可能存在的响应滞后。

3. **DCT 频率分解**

   将窗口级时序特征拆为多个频段，只在共享有效频段内做窗口交互。

4. **频率解耦 cross-attention**

   对不同频段分别建模窗口依赖，抑制无关频段噪声。

#### 优点

- 能显式建模窗口差异，而不是简单拼接
- 适合多阶段、周期性或准周期性工业过程
- 对“相位错位”和“频段噪声”有明确结构假设

#### 风险

- 当前项目的 `full / exposure / recovery` 是同一样本的不同窗口视图，不一定存在 PAF-Net 场景中那种跨过程时滞
- DCT、相位对齐、cross-attention 同时引入后，失败时难以定位原因
- 训练成本和过拟合风险显著高于当前 PhaseWindowTCN

#### 启用条件

只有满足以下条件之一时，才考虑进入该方案：

- `share_window_encoder=false` 明确提升 N2，但仍未达到验收线
- 深 TCN 明确提升 N2，说明长程结构有效，但简单 concat 融合仍不足
- 窗口级特征分析显示 full/exposure/recovery 存在明显错位或频段差异

#### 当前决策

暂缓。第一批实验只保留它的最低成本子集：

- `share_window_encoder=false`
- 更深 TCN 感受野
- 必要时轻量 gated fusion

### 10.2 备选方案 B：多任务学习 + 动态损失权重

#### 核心思路

该方案把四个组分视为多任务回归问题，通过组分特定输出头和动态损失权重，缓解易预测组分压制困难组分的问题。

候选结构如下：

```python
component_heads = {
    "H2": nn.Linear(hidden_dim, 1),
    "CO2": nn.Linear(hidden_dim, 1),
    "N2": nn.Linear(hidden_dim, 1),
    "CH4": nn.Linear(hidden_dim, 1),
}

loss = sum(
    weight[name] * mse(pred[name], target[name])
    for name in ("H2", "CO2", "N2", "CH4")
)
```

#### 可选权重策略

1. **静态组分权重**

   直接给 N2 更高权重，例如：

   ```text
   H2: 1
   CO2: 1
   CH4: 1
   N2: 2 或 4
   ```

   优点是实现简单、可解释；缺点是需要人工调权重。

2. **方差加权**

   使用训练集方差构造权重：

   ```python
   weight[c] = 1.0 / var(y_c)
   ```

   该方法本质是尺度归一化，不等价于“困难组分更高权重”。如果 N2 方差不是最小，或 N2 噪声更高，它未必能给出正确优化方向。

3. **GradNorm**

   根据每个任务的梯度范数和训练进度动态调整权重，减少某一任务主导共享参数更新的风险。

   优点是比静态权重更自适应；缺点是实现复杂，需要修改训练循环并维护每个任务的 loss 与梯度统计。

4. **不确定性加权**

   学习每个任务的观测不确定性，用不确定性反向决定 loss 权重。

   适合作为比静态权重更稳的多任务 baseline，但仍不能替代表征结构验证。

#### 与闭包约束的冲突

四个独立 component head 会重新引入一个风险：

```text
H2 + CO2 + N2 + CH4 != 100
```

因此，该方案不能直接替换当前 `gas_head`，必须配套以下任一约束：

1. `raw4 + closure_penalty`
2. `softmax100`
3. 预测前三个自由组分，再由闭包得到第四个组分

否则会回到 PhaseWindowTCN MVP 中已经暴露过的 sum error 问题。

#### 推荐低成本对照

若需要验证该方向，优先不要直接实现 GradNorm，而是先做：

```text
raw4 + closure_penalty + N2_weighted_mse
```

建议实验名：

```text
phase_window_tcn_raw4_closure_n2w
```

建议初始权重：

```text
lambda_closure = 0.1 或 1.0
N2_weight = 2.0 或 4.0
```

该对照可以回答一个关键问题：

```text
N2 作为 gas_head 残差项，是否限制了独立学习？
```

#### 启用条件

只有当结构消融后出现以下现象时，才启用本方案：

- `H2/CO2/CH4` 明显改善，但 N2 仍接近 0
- `gas_head` 下闭包正常，但 N2 作为残差项始终被前三项误差牵制
- 训练日志显示不同组分 loss 收敛速度严重不一致

#### 当前决策

低优先级保留。它是 loss/目标分配对照，不是下一轮主线。

### 10.3 备选方案 C：轻量窗口融合模块

#### 核心思路

在第一批结构消融有效后，替换当前简单 concat 融合：

```text
z = concat([z_full, z_exposure, z_recovery])
```

改为窗口级自适应融合：

```python
weights = softmax(gate([z_full, z_exposure, z_recovery]))
z = weights[0] * z_full + weights[1] * z_exposure + weights[2] * z_recovery
```

#### 可选实现

1. **Gated fusion**

   样本级学习三个窗口权重。

2. **Component-aware gated fusion**

   为每个组分学习不同窗口权重，允许 N2 更偏 exposure/recovery，其他组分更偏 full。

3. **Lightweight cross-attention**

   只在窗口级 pooled feature 上做 attention，不在完整时序 token 上做大 attention。

#### 优点

- 比 PAF-Net 完整频域模块简单
- 能直接检验“不同组分是否依赖不同窗口”
- 解释性强，可输出窗口权重

#### 风险

- 如果 split/deep 本身无收益，gating 很可能只是在无效特征上重加权
- component-aware gating 可能增加参数并诱发过拟合

#### 启用条件

- 第一批任一结构消融让 N2 明显提升
- 但提升仍不足以进入正式主线
- 且其他组分没有明显退化

#### 当前决策

作为第二批实验保留。

### 10.4 备选方案 D：完整频域相位融合

这是方案 A 的完整版本，包括：

- phase correlation alignment
- DCT 多频段分解
- frequency-independent patch attention
- frequency-decoupled cross-attention

该方向只在以下条件全部满足时再讨论：

1. 简单结构消融已经证明 DL 多窗口路线有正信号
2. 轻量 gated fusion 或 attention 仍不足
3. 有可视化或统计证据说明不同窗口的有效信息位于不同频段
4. 训练资源允许做多组消融

当前不建议直接实施。
