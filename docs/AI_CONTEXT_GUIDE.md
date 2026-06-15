# 多模态掺氢天然气浓度预测项目：给外部 AI 的自包含上下文

> 更新日期：2026-06-15  
> 用途：这份文档用于复制给网页版 AI 或其他无法访问本地仓库的模型。文档本身应足够说明项目背景、当前结论、问题瓶颈和需要审查的方案。

## 1. 项目目标

本项目研究“多模态掺氢天然气四组分浓度预测”。

输入信号包含三类模态：

1. 慢变量：温度、压力、流量等工况量
2. 超声波形：与声速、衰减、TOF 等气体物理特性相关
3. 光纤麦克风波形：声学响应的另一类波形代理信号

预测目标是四个气体组分浓度：

```text
H2, CO2, N2, CH4
```

标签满足组成闭包约束：

```text
H2 + CO2 + N2 + CH4 = 100
```

其中 `N2` 是当前最难预测的组分。`H2 / CO2 / CH4` 相对更容易学到，`N2` 在很多 DL 模型中接近不可学或 R2 为负。

## 2. 数据和实验设定

当前正式数据主线是 v4 benchmark，核心特点：

- 使用 `mixture_id` 作为唯一配气方案 ID
- `sequence_id` 只是时序样本实例 ID，不能替代 `mixture_id`
- split 按 `mixture_id` 组织，避免同一配气方案泄漏到不同 split
- 正式新 benchmark 不依赖旧字段 `base_condition_id`、`noise_seed_index`、`noise_seed`
- 主要评估 split：
  - `val`
  - `test`
  - `extrapolation`

时序数据包含不同阶段窗口。当前最关键的窗口是：

```text
full + exposure + recovery
```

其中：

- `full`：完整时序
- `exposure`：暴露/响应阶段
- `recovery`：恢复阶段

传统 ML 已证明这三个窗口的组合对 N2 有用。

## 3. 当前最重要实验结论

### 3.1 ML 多窗口路线已经成立

当前最强、最稳定的正式 ML phase-aware 主线是：

```text
ridge_multiwindow_all_modalities
```

它使用：

```text
full + exposure + recovery
```

三个窗口分别提取表格统计特征，再横向拼接给 Ridge 多输出回归。

关键结果：

```text
test N2 R2          = 0.7121
extrapolation N2 R2 = 0.7247
test overall R2     = 0.9253
test macro RMSE     = 2.4133
```

这个结果说明：

1. N2 并非完全不可观测
2. `full + exposure + recovery` 确实包含对 N2 有用的信息
3. DL 失败不能简单解释为“数据没有 N2 信号”

### 3.2 PhaseWindowTCN MVP 失败

PhaseWindowTCN 是一个真实多窗口 DL 模型，输入为：

```text
(B, W, T, C)
```

其中：

```text
W = 3
窗口 = full, exposure, recovery
```

每个窗口先过相同或独立的编码器，再融合为最终预测。

MVP 版本使用：

```text
output_mode = raw4
loss = mse
share_window_encoder = true
tcn_channels = [64, 64, 64]
```

关键失败结果：

```text
test overall R2  = 0.2635
test N2 R2       = -0.0150
extrap N2 R2     = 0.0028
sum_abs_error    = 11.1797
best epoch       = 4
```

主要问题：

1. `raw4` 输出没有表达组成闭包约束
2. 四个组分独立漂移，导致预测总和远离 100
3. N2 仍基本不可学

### 3.3 gas_head 修复了闭包，但没有解决 N2

后来将输出头改为 `gas_head`：

```text
free_ratio = softmax(raw[:3])
free_total = 100 * sigmoid(raw[3])
[H2, CH4, CO2] = free_total * free_ratio
N2 = 100 - H2 - CH4 - CO2
```

注意：项目内部自由组分顺序为前三个非 N2 组分，N2 由闭包残差得到。

实验 1：

```text
run = phase_window_tcn_gas_4mse
output_mode = gas_head
loss = mse 四组分普通 MSE
```

结果：

```text
test overall R2 = 0.4798
test N2 R2      = -0.0066
sum_abs_error   ≈ 2e-6
```

实验 2：

```text
run = phase_window_tcn_gas_free
output_mode = gas_head
loss = free_component_mse
```

`free_component_mse` 只监督前三个自由组分，不直接监督 N2，避免对闭包残差项重复施加 loss。

结果：

```text
test overall R2          = 0.5145
test N2 R2               = -0.0155
extrapolation N2 R2      = -0.0396
sum_abs_error            ≈ 2e-6
```

结论：

1. `gas_head` 确实修复了闭包问题
2. `free_component_mse` 提升 overall，但没有改善 N2
3. 当前主要瓶颈已经从“输出闭包错误”转向“窗口表征与相位融合不足”

## 4. 当前问题诊断

PhaseWindowTCN 现在的核心问题不是输出总和，而是：

```text
DL 已经看到 full/exposure/recovery 三窗口，
但无法像 ML 多窗口 Ridge 那样稳定利用它们提升 N2。
```

主要怀疑点：

### 4.1 共享窗口编码器可能稀释相位差异

当前默认：

```text
share_window_encoder = true
```

这意味着 full、exposure、recovery 三个窗口共用同一套编码器。

问题是：

- full 更像全局稳态/完整上下文
- exposure 更可能包含 N2 相关瞬态响应
- recovery 可能包含恢复动态和滞后信息

共享 encoder 可能迫使三类窗口使用同一套特征抽取规则，导致相位差异被压成折中表征。

### 4.2 当前 TCN 感受野可能不足

当前 TCN 配置：

```text
tcn_channels = [64, 64, 64]
kernel_size = 3
dilation = [1, 2, 4]
```

名义感受野约：

```text
RF = 29
```

输入长度约为 256 timestep 时，RF 只覆盖约 11% 的时间范围。N2 的有用信息可能存在于更长程的响应或跨窗口结构中。

深 TCN 候选：

```text
tcn_channels = [64, 64, 64, 64, 64]
dilation = [1, 2, 4, 8, 16]
RF ≈ 125
```

### 4.3 简单 concat 融合可能不足

当前 PhaseWindowTCN 大致流程：

```text
x_full      -> WindowedFusionEncoder -> z_full
x_exposure  -> WindowedFusionEncoder -> z_exposure
x_recovery  -> WindowedFusionEncoder -> z_recovery

concat([z_full, z_exposure, z_recovery]) -> MLP -> gas_head
```

问题：

- concat 没有显式建模窗口之间的互补性
- 没有给 N2 单独选择 exposure/recovery 的机制
- 没有建模窗口之间可能存在的滞后、频率差异或动态关系

但当前不希望直接上复杂 attention，因为还没证明最小结构消融有效。

## 5. 当前执行方案

当前只做第一批结构消融，不直接上复杂 PAF-Net / DCT / attention。

第一批实验配置包含同一个 seed 下的三项：

```text
phase_window_tcn_gas_free
phase_window_tcn_gas_free_split
phase_window_tcn_gas_free_deep
```

### 5.1 同 seed 基线：phase_window_tcn_gas_free

```text
output_mode = gas_head
loss = free_component_mse
share_window_encoder = true
tcn_channels = [64, 64, 64]
```

目的：作为同一批 ablation 的可比基线。

### 5.2 Split encoder：phase_window_tcn_gas_free_split

只改一项：

```text
share_window_encoder = false
```

目的：

```text
验证 full / exposure / recovery 是否需要独立编码器。
```

### 5.3 Deep TCN：phase_window_tcn_gas_free_deep

只改一项：

```text
tcn_channels = [64, 64, 64, 64, 64]
```

目的：

```text
验证更大感受野是否改善 N2。
```

### 5.4 Followup：phase_window_tcn_gas_free_split_deep

只有当 split 或 deep 任一有效时再运行：

```text
share_window_encoder = false
tcn_channels = [64, 64, 64, 64, 64]
```

目的：

```text
验证独立窗口编码器与更深 TCN 是否存在叠加收益。
```

## 6. 验收标准

所有结构消融必须相对同 seed 的 `phase_window_tcn_gas_free` 判断。

核心指标：

```text
test x_N2 R2
extrapolation x_N2 R2
macro RMSE
sum_abs_error
其他三组分 R2 是否明显退化
```

最低门槛：

```text
test x_N2 R2 > 0
extrapolation x_N2 R2 > 0
macro RMSE 不高于当前负结果
sum_abs_error 继续接近 0
```

若 split 和 deep 都不能改善 N2，则停止 DL 主线继续扩展。正式主线保持 ML 多窗口 Ridge。

## 7. 已讨论但暂缓的备选方案

### 7.1 方案 A：分离式窗口编码 + 频率解耦注意力

灵感来自 PAF-Net 类多过程建模方法：

```text
phase-independent encoders
phase correlation alignment
DCT frequency decomposition
frequency-decoupled cross-attention
```

候选结构：

```python
phase_encoders = {
    "full": TCNEncoder(shared=False),
    "exposure": TCNEncoder(shared=False),
    "recovery": TCNEncoder(shared=False),
}

encoded = [phase_encoders[p](x[p]) for p in ["full", "exposure", "recovery"]]
aligned = phase_correlation_alignment(encoded)
fused = frequency_decoupled_cross_attention(aligned, n_frequencies=5)
```

当前决策：

```text
暂缓。
```

原因：

- 本项目三窗口来自同一样本的不同视图，不一定存在 PAF-Net 场景中的跨过程时滞
- DCT、对齐、cross-attention 同时引入后，失败难以定位
- 只有 split/deep 有正信号后，才值得考虑轻量 fusion 或频域方法

### 7.2 方案 B：多任务学习 + 动态损失权重

候选思路：

```text
每个组分一个 head
N2 使用更高权重
或用 GradNorm / 不确定性加权动态平衡 loss
```

问题：

- 四个独立 head 会重新引入 `sum != 100` 的风险
- 如果不配套 closure penalty 或 simplex head，会回到 raw4 MVP 的闭包失败
- loss 权重只能缓解任务竞争，不能解决窗口表征不足

较低成本对照可以是：

```text
raw4 + closure_penalty + N2_weighted_mse
```

当前决策：

```text
低优先级保留。只有结构消融后 H2/CO2/CH4 改善但 N2 仍被压制时再考虑。
```

### 7.3 方案 C：轻量窗口融合

若 split/deep 有正信号，但仍不足，可考虑：

```text
gated fusion
component-aware gated fusion
lightweight cross-attention over pooled window features
```

当前决策：

```text
作为第二批备选，不进入第一批实验。
```

## 8. 希望外部 AI 审查的问题

请基于以上信息审查以下问题：

1. 当前把第一批实验限制为 `split encoder` 和 `deep TCN` 是否合理？
2. 在 `gas_head + free_component_mse` 已经失败的情况下，N2 仍为负的更可能原因是什么？
3. `share_window_encoder=false` 是否是验证相位差异的最小充分实验？
4. 更深 TCN 感受野是否足以验证长程依赖问题，还是应优先改 pooling / fusion？
5. 是否存在比 gated fusion 更简单、可解释、低风险的窗口融合方式？
6. 是否应该重新考虑 N2 独立 head？如果考虑，如何保持闭包约束？
7. 若 split/deep 都失败，是否应停止 DL 主线，接受 ML 多窗口作为正式结果？

请优先给出：

```text
结论 -> 风险 -> 推荐实验顺序 -> 不建议做的事
```

不要只给泛泛的深度学习建议，要围绕本项目的三个事实：

1. ML 多窗口 Ridge 已经强通过
2. gas_head 已修复闭包但 N2 没改善
3. 当前 DL 的主要疑点是窗口表征和相位融合

## 9. 给外部 AI 的约束提醒

请不要建议：

- 把 `mixture_id` 改回 `sequence_id`
- 使用会泄漏 split 的随机样本级划分
- 直接上复杂大模型而不做可解释消融
- 在没有同 seed 基线的情况下比较实验
- 忽略 `H2 + CO2 + N2 + CH4 = 100` 的闭包约束
- 把 ILR/ALR 或 PhasePreservingTCN 旧方案当成当前主线

当前项目更需要的是：

```text
少变量、可归因、能明确决定是否继续 DL 的实验。
```
