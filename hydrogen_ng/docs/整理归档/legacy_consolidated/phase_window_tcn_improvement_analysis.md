# PhaseWindowTCN MVP 失败根因分析与改进方案

> 日期: 2026-06-13  
> 状态: 2026-06-13 联网资料与本地实现复核后改写  
> 基于: `phase_window_tcn_mvp` 服务器实验结果  
> 对照基线: `ridge_multiwindow_all_modalities` (test N2 R2=0.7121, overall R2=0.9253)

---

## 0. 审查结论

PhaseWindowTCN 的下一步不应从加大模型开始，也不应把 ILR 作为主路线。当前更可靠的改进方向是：

1. 先把 `raw4 + mse` 改成闭包输出头，优先使用现有 `output_mode="gas_head"`。
2. `gas_head` 必须配套自由分量 loss，只监督 `H2/CH4/CO2`，让 `N2 = 100 - sum(H2, CH4, CO2)` 自然闭包生成。
3. `ILR/ALR` 在组成数据理论上成立，但本项目已有 formal_full 结果显示它们对 N2 是负向路线，因此只保留为低优先级对照。
4. `share_window_encoder=false`、加深 TCN、跨窗口 attention 应放在目标/head/loss 之后，避免把目标设计问题误判成架构容量问题。

最小主线实验应是：

```text
PW-GAS-FREE:
  model: phase_window_tcn
  output_mode: gas_head
  loss: free_component_mse  # 新增，只对前三个自由组分算 MSE
  lr: 1e-4
  tcn_channels: [64, 64, 64]
  share_window_encoder: true
```

---

## 1. 资料与实现复核

### 1.1 外部资料结论

- 组成数据是非负、固定和或相对总量约束的数据；百分比和化学组成都属于典型场景。直接把闭包数据当普通欧氏空间向量做回归，容易产生不符合样本空间的不一致结果。参考 Aitchison 的 compositional data 经典工作，以及 Annual Reviews 对 CoDA 的综述。  
  资料: https://www.jstor.org/stable/2345821, https://doi.org/10.1146/annurev-statistics-042720-124436

- ILR 是合法的 simplex 到实坐标变换，Egozcue 等提出的 isometric logratio transformation 能保持度量性质。但“理论合法”不等于“本数据集一定提升 N2”；本项目已有实验显示 ILR/ALR 对 N2 负向。  
  资料: https://doi.org/10.1023/A:1023818214614

- `Softmax` 会把输出缩放到 `[0, 1]` 且沿指定维度求和为 1；`Sigmoid` 可把 total logit 映射到有界区间。因此 `GasHeadNormalize` 的 `softmax + sigmoid + residual N2` 是一种结构性闭包参数化。  
  资料: https://docs.pytorch.org/docs/2.12/generated/torch.nn.Softmax.html, https://docs.pytorch.org/docs/2.12/generated/torch.nn.Sigmoid.html

- PyTorch `MSELoss` 是逐元素平方误差再按 reduction 聚合。对闭包输出而言，如果 `N2` 已由前三项推导，继续把 N2 当独立第 4 项参与 loss，会把残差项误差反向绑回前三项。  
  资料: https://docs.pytorch.org/docs/2.12/generated/torch.nn.MSELoss.html

- TCN 文献支持卷积序列模型可作为序列建模起点，并强调有效记忆长度的重要性。PhaseWindowTCN 当前 3-block TCN 感受野只有 29 步，加深 TCN 有依据，但应在目标/loss 修正后再做。  
  资料: https://arxiv.org/abs/1803.01271

### 1.2 本仓库实现结论

- `PhaseWindowTCNRegressor` 已支持 `output_mode in {"raw4", "softmax100", "gas_head"}`，其中 `gas_head` 复用 `GasHeadNormalize`。
- `GasHeadNormalize` 的实际输出为：

```python
free_ratio = softmax(raw[:, :3])
free_total = 100 * sigmoid(raw[:, 3:4])
free = free_total * free_ratio
n2 = 100 - sum(free)
```

- `src/dl/training/losses.py` 当前没有 `free_component_mse`、component weight loss 或 loss slicing；新增自由分量 loss 是代码级改动，不是纯配置改动。
- `gradient_clip` 当前没有接入 `Trainer.fit()` 或 experiment config；它不能写成可直接运行的配置项。
- `outputs/summary/formal_full_summary.csv` 中已有负向 log-ratio 证据：

| run | target | test overall R2 | test N2 R2 | 结论 |
|---|---|---:|---:|---|
| `ridge_all_modalities` | raw | 0.7968 | 0.2173 | baseline |
| `ridge_ilr_n2_first_all_modalities` | ILR | 0.7123 | 0.1058 | N2 变差 |
| `ridge_alr_ch4_all_modalities` | ALR | 0.7123 | 0.1058 | N2 变差 |
| `cnn1d_tcn_fusion` | gas head/raw label | 0.7138 | -0.0075 | 闭包好，但 N2 仍弱 |
| `cnn1d_tcn_fusion_ilr` | ILR | 0.6630 | -0.0916 | N2 变差 |

这说明：闭包输出能显著修复 sum error，但不保证 N2 自动变好；ILR 不应作为主线。

---

## 2. 现状回顾

### 2.1 MVP 架构

```text
输入: (B, 3, 256, 3008) -> [full窗口 | exposure窗口 | recovery窗口]
                                  |
                                  v
                         共享 WindowedFusionEncoder
                                  |
                         每个窗口输出 192-d
                                  |
                                  v
                         concat 576-d -> 128 -> 64 -> Linear(64, 4)
                                  |
                                  v
                         [H2, CH4, CO2, N2]

Loss: MSELoss(raw4_pred, raw4_label)
```

关键超参：

| 项 | 值 |
|---|---|
| optimizer | AdamW |
| lr | 0.0005 |
| batch_size | 8 |
| epochs | 300 |
| early stopping | patience=25 |
| scheduler | ReduceLROnPlateau |
| output_mode | `raw4` |
| tcn_channels | `[64, 64, 64]` |
| share_window_encoder | `true` |

### 2.2 实验结果

| 指标 | Ridge Multiwindow | PhaseWindowTCN MVP | 差值 |
|---|---:|---:|---:|
| Test overall R2 | 0.9253 | 0.2635 | -0.6618 |
| Test N2 R2 | 0.7121 | -0.0150 | -0.7271 |
| Test H2 R2 | 0.9943 | 0.5745 | -0.4198 |
| Test CH4 R2 | 0.9202 | 0.1632 | -0.7570 |
| Test CO2 R2 | 0.9773 | -0.0770 | -1.0543 |
| Test macro RMSE | 2.4133 | 7.5793 | +5.1660 |
| Test sum abs error | 约 0 | 11.1797 | +11.1797 |
| Extrapolation N2 R2 | 0.7247 | 0.0028 | -0.7219 |
| Best epoch | - | 4 of 29 | - |

验收结论：PhaseWindowTCN MVP 不成立。失败不只是 N2 没有提升，而是整体组成回归能力明显低于 ML 多窗口基线。

---

## 3. 修正后的根因分析

### 根因 1: `raw4 + mse` 没有表达闭包不变量

四组分标签满足：

```text
H2 + CH4 + CO2 + N2 = 100
```

`raw4` 输出没有这个结构约束，训练时的四个输出可以独立漂移，最终产生 test `sum_abs_error=11.1797`。这是必须先修的错误。

需要修正上一版计划中的一个错误判断：N2 不是 60-95% 主组分。本项目数据规格是 `CH4: 50-95`，`N2: 0-20`。因此不能把失败解释为“N2 尺度最大、主导梯度”。更合理的解释是：

```text
raw4 不闭包
  -> 模型可输出不在 simplex 上的组成
  -> CH4/N2 之间的残差关系没有被结构化表达
  -> 弱可观测的 N2 更容易退回均值附近
  -> sum error、CO2 R2、N2 R2 同时崩坏
```

### 根因 2: 闭包输出不能继续照搬 4-task MSE

`gas_head` 下 N2 是由前三个自由组分推导出来的：

```text
N2 = 100 - H2 - CH4 - CO2
```

如果继续使用四组分 MSE：

```text
loss = MSE([H2, CH4, CO2, N2]_pred, [H2, CH4, CO2, N2]_true)
```

那么 N2 误差会再次反向约束前三项，形成重复监督。仓库归档文档中也记录过类似经验：`3-task loss` 只监督自由组分，避免继续对推导出的 N2 单独计算 loss。

因此 PhaseWindowTCN 的主改进不应只是 `gas_head`，而应是：

```text
output_mode = gas_head
loss = free_component_mse(pred[:, :3], target[:, :3])
```

### 根因 3: 优化不稳定是症状，不是首因

MVP best epoch=4，之后验证集持续退化直到早停。降低 lr 有必要，但它只能降低震荡，不能修正目标空间错误。

建议把 lr 从 `5e-4` 降到 `1e-4`，但不要把“延长 patience”作为核心改进。若第 4 轮即 best，继续给更多 epoch 通常只会更晚停止。

### 根因 4: TCN 感受野偏短，但优先级低于 head/loss

当前 3 blocks、kernel=3、dilation `[1,2,4]`：

```text
RF = 1 + 2 * (k - 1) * sum(dilation)
   = 1 + 2 * 2 * (1 + 2 + 4)
   = 29
```

输入长度 256 时只覆盖约 11.3%。加到 5 blocks：

```text
dilation = [1,2,4,8,16]
RF = 1 + 2 * 2 * 31 = 125
```

感受野提升有依据，但如果 `raw4 + mse` 不先修，感受野扩大很可能只是扩大一个错误目标的拟合能力。

### 根因 5: 共享窗口 encoder 可能稀释相位语义

`share_window_encoder=true` 让 full、exposure、recovery 共用同一套编码器。三个窗口的相位含义不同，共享参数可能让 encoder 学到折中表征。

但 ML 多窗口成功说明窗口组合有效，不说明 DL 一定需要复杂 attention。第一步应做最小消融：

```text
share_window_encoder=true  vs  false
```

只有当 `gas_head + free_component_mse` 明确改善后，才值得继续做跨窗口 attention 或 phase token。

---

## 4. 改进方案

### Tier 1: 输出头与 loss 修正

#### 方案 1A: `gas_head + free_component_mse` 主推

```text
输出:
  free_ratio = softmax(logits[:3])
  free_total = 100 * sigmoid(total_logit)
  [H2, CH4, CO2] = free_total * free_ratio
  N2 = 100 - H2 - CH4 - CO2

loss:
  MSE(pred[:, :3], target[:, :3])
```

优点：

- 强制 `sum(pred)=100`，直接消除 sum error。
- N2 作为残差组分，不再被独立 loss 重复监督。
- 与现有 `GasHeadNormalize` 实现一致，代码改动集中在 loss。

风险：

- 如果前三个自由组分里 CH4 仍学不好，N2 仍会差。
- 需要新增 loss，并保证 evaluation 仍按四组分完整计算。

需要改动：

- `src/dl/training/losses.py`: 新增 `FreeComponentMSELoss` 或 `free_component_mse`。
- `src/pipeline/experiment_config.py`: loss registry 校验自动生效；如需限制 output mode，需增加 run 级校验。
- tests: 增加 loss shape、只切前三列、完整四组分评估不变的单测。

#### 方案 1B: `gas_head + mse` 快速闭包对照

这是零代码配置对照：

```json
{
  "model": "phase_window_tcn",
  "loss": "mse",
  "model_kwargs": {
    "output_mode": "gas_head"
  }
}
```

预期：

- `sum_abs_error` 应接近 0。
- N2 R2 未必改善。`cnn1d_tcn_fusion` 已证明闭包输出本身不够，test N2 R2 仍约 -0.0075。

用途：

- 只用于确认 PhaseWindowTCN 接入 `gas_head` 后是否恢复基本组成合法性。
- 不作为最终主实验。

#### 方案 1C: ILR/ALR 低优先级对照

ILR/ALR 在理论上适合 compositional data，但本项目 formal_full 结果已经负向：

```text
ridge_ilr_n2_first: test N2 R2 = 0.1058 < raw ridge 0.2173
cnn1d_tcn_fusion_ilr: test N2 R2 = -0.0916 < gas_head cnn1d_tcn_fusion -0.0075
```

因此它不能作为 PhaseWindowTCN 的第一优先。只有在 `gas_head + free_component_mse` 跑完后，才建议补一个 `PW-ILR` 作为低优先级对照。

### Tier 2: 稳定训练

| 方案 | 当前状态 | 建议 |
|---|---|---|
| lr `5e-4 -> 1e-4` | 配置级可做 | 与 Tier 1 同时采用 |
| gradient clip | 当前未接入 | 先不要写入配置；如需要，作为代码改动单独实现 |
| patience `25 -> 40` | 配置级可做 | 不建议作为主改进；best epoch=4 说明问题不在训练轮数 |
| Cosine + warmup | 当前未接入 | 暂缓，避免变量过多 |

### Tier 3: 架构消融

| 实验 | 改动 | 目的 | 优先级 |
|---|---|---|---|
| split encoder | `share_window_encoder=false` | 验证不同相位窗口是否需要独立 encoder | 中 |
| deep TCN | `tcn_channels=[64,64,64,64,64]` | 感受野从 29 提升到 125 | 中 |
| larger acoustic | `acoustic_channels=[32,64,128,128]` | 降低声学压缩比 | 低 |
| cross-window attention | 新增 attention | 建模窗口间交互 | 低，等上面消融后再做 |
| phase token/gating | 新增 phase embedding | 显式标识窗口身份 | 低，等 split encoder 后再评估 |

---

## 5. 推荐实验路线

### 第一批: 最小验证

| 实验编号 | output | loss | lr | 其他 | 验证假设 |
|---|---|---|---:|---|---|
| PW-GAS-4MSE | gas_head | mse | 1e-4 | 架构不变 | 闭包输出是否修复 sum error |
| PW-GAS-FREE | gas_head | free_component_mse | 1e-4 | 架构不变 | 去掉 N2 重复监督是否改善 N2 |

通过条件：

- `sum_abs_error` 接近 0。
- test N2 R2 至少转正。
- test macro RMSE 不高于 PhaseWindowTCN MVP。
- H2/CH4/CO2 不能出现比 MVP 更大的系统性退化。

### 第二批: 架构消融

只在 `PW-GAS-FREE` 有改善时启动：

| 实验编号 | 改动 | 验证假设 |
|---|---|---|
| PW-GAS-FREE-SPLIT | `share_window_encoder=false` | 独立相位 encoder 是否提升 N2 |
| PW-GAS-FREE-DEEP | `tcn_channels=[64,64,64,64,64]` | 长感受野是否提升 exposure/recovery 趋势建模 |
| PW-GAS-FREE-SPLIT-DEEP | split encoder + deep TCN | 组合收益是否超过单项 |

### 第三批: 低优先级对照

| 实验编号 | 改动 | 目的 |
|---|---|---|
| PW-ILR | `target_transform=ilr_n2_first`, `loss=ilr_mse` | 验证 PhaseWindow 是否改变 ILR 历史负向结论 |
| PW-SOFTMAX100 | `output_mode=softmax100` | 对比完整 4-way simplex 输出与 N2 residual head |
| PW-ATTN | 跨窗口 attention | 仅在 split/deep 明确不足时推进 |

---

## 6. 验收标准

相对 `phase_window_tcn_full_exp_rec_raw4`：

| 指标 | 最低要求 |
|---|---:|
| test sum_abs_error | 从 11.1797 降到接近 0 |
| test N2 R2 | > 0 |
| extrapolation N2 R2 | > 0 |
| test macro RMSE | < 7.5793 |
| best epoch | 不应仍集中在 4 附近且之后快速崩坏 |

相对 ML 正式主线 `ridge_multiwindow_all_modalities`：

| 指标 | 目标 |
|---|---:|
| test N2 R2 | 先达到 0.30，再讨论追 0.7121 |
| overall R2 | 先达到 0.70，再讨论追 0.9253 |
| macro RMSE | 先低于 5.0，再讨论追 2.4133 |

PhaseWindowTCN 当前不应被要求一次超过 ML 多窗口 ridge。第一阶段目标是证明“多窗口 DL + 正确组成 head/loss”能稳定学习，而不是直接替代正式 ML 主线。

---

## 7. 实施清单

### 7.1 代码改动

1. 在 `src/dl/training/losses.py` 增加：

```python
class FreeComponentMSELoss(nn.Module):
    def __init__(self, free_components: int = 3):
        super().__init__()
        self.free_components = free_components
        self.loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.loss(pred[:, : self.free_components], target[:, : self.free_components])
```

2. 注册：

```python
LOSS_REGISTRY["free_component_mse"] = FreeComponentMSELoss
```

3. 增加单测：

- 输入 `(B, 4)` 输出 scalar loss。
- 修改第 4 列 N2 不改变 loss。
- pred/target 非二维或列数不足时显式报错。

### 7.2 配置改动

新增实验配置，例如：

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

训练配置建议：

```json
{
  "lr": 0.0001,
  "batch_size": 8,
  "weight_decay": 0.0001,
  "early_stopping": {
    "enabled": true,
    "monitor": "val_loss",
    "patience": 25,
    "min_delta": 0.0,
    "mode": "min"
  },
  "scheduler": {
    "name": "reduce_on_plateau",
    "factor": 0.5,
    "patience": 8,
    "min_lr": 0.000001
  }
}
```

---

## 8. 最终判断

PhaseWindowTCN 的失败不是因为 `full + exposure + recovery` 窗口信息无效。ML 多窗口 ridge 已证明这组三窗口是有效信息源。

真正需要优先修正的是：

```text
raw4 无闭包输出
  + 4 组分普通 MSE
  + 推导残差关系没有进入模型结构
```

下一步应按以下顺序推进：

```text
gas_head + free_component_mse
  -> share_window_encoder 消融
  -> TCN 感受野消融
  -> attention / phase token
  -> ILR 低优先级负向复核
```

只要 N2 R2 仍未转正，就不应投入复杂融合模块；先把输出空间、loss 和闭包不变量固定下来。

---

## 9. 参考资料

1. Aitchison, J. (1982). The Statistical Analysis of Compositional Data. Journal of the Royal Statistical Society Series B. https://www.jstor.org/stable/2345821
2. Greenacre, M. (2021). Compositional Data Analysis. Annual Review of Statistics and Its Application. https://doi.org/10.1146/annurev-statistics-042720-124436
3. Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G., & Barcelo-Vidal, C. (2003). Isometric Logratio Transformations for Compositional Data Analysis. Mathematical Geology. https://doi.org/10.1023/A:1023818214614
4. PyTorch `Softmax` documentation. https://docs.pytorch.org/docs/2.12/generated/torch.nn.Softmax.html
5. PyTorch `Sigmoid` documentation. https://docs.pytorch.org/docs/2.12/generated/torch.nn.Sigmoid.html
6. PyTorch `MSELoss` documentation. https://docs.pytorch.org/docs/2.12/generated/torch.nn.MSELoss.html
7. Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. https://arxiv.org/abs/1803.01271
