# N2 不可学诊断与 gas_head 参数化分析

日期：2026-06-19

## 结论

所有 DL run 中 N2 组分 R2 ≈ 0，主因是 `gas_head` 的**闭包残差参数化**（N2 = 100 − (H2+CH4+CO2)），不是此前判断的"非线性信息压缩 / ReLU+Dropout 梯度竞争"。判别变量不是"线性 vs 非线性"，而是 **N2 有没有自己的输出参数去直接拟合 N2_true**。

此结论推翻 `rolling_summary.md` 中"非线性压缩"判断，以及由该判断派生的方向 C / D / E / F。

## 证据

### 1. ridge 证明 N2 有强线性信号且跨 split 稳健

`ridge_multiwindow_all_modalities`（线性、与 handcraft_mlp 同特征、4 个独立输出，无闭包），实测：

| split | N2 R2 | H2 R2 | CH4 R2 | CO2 R2 |
|---|---|---|---|---|
| train | 0.741 | 0.995 | 0.924 | 0.978 |
| val | 0.740 | 0.995 | 0.924 | 0.976 |
| **test** | **0.712** | 0.994 | 0.920 | 0.977 |
| extrapolation | 0.725 | 0.994 | 0.915 | 0.980 |

数据来源：`outputs/runs/multiwindow_n2/ridge_multiwindow_all_modalities/metrics.json`。

### 2. N2 在全浓度区间都可学

ridge 在 test 的 N2 分箱（`conditional_metrics.n2_bins`）：

| N2 浓度区间 | count | N2 R2 | rmse |
|---|---|---|---|
| 0.03 – 5.01 | 154 | 0.883 | 2.64 |
| 5.01 – 9.99 | 166 | 0.930 | 2.01 |
| 9.99 – 14.98 | 138 | 0.943 | 1.84 |
| 14.98 – 19.96 | 142 | 0.860 | 3.00 |

N2 不是"低浓度难学"的局部问题，全范围 R2 都在 0.86 以上。原"方向 A 诊断（N2 是否在任何浓度区间可学）"已被 ridge 输出回答：可学。

### 3. 全部 Phase 1 run 被配置校验锁死在 gas_head

`src/dl/training/losses.py:182-188`（`validate_loss_model_output`）强制：`weighted_component_mse` / `free_component_mse` / `weighted_free_component_mse` 这三个损失在 `phase_window_tcn` 上必须 `output_mode='gas_head'`。因此 Phase 1 的 4 个 run（gas_free、gas_varweight、gas_free_varweight、handcraft_mlp）全部在 gas_head（N2=残差）regime 下运行。`raw4`、`softmax100` 从未被测过。"N2 学不到"只在这一个 regime 成立。

### 4. 残差机制的数学

`GasHeadNormalize.forward`（`src/dl/models/cnn1d_tcn_fusion.py:108-113`）：

```python
raw = self.linear(features)
free_ratio = torch.softmax(raw[:, :3], dim=-1)   # H2/CH4/CO2 三者间比例
free_total = 100.0 * torch.sigmoid(raw[:, 3:4])  # 三者总量
free = free_total * free_ratio
n2 = 100.0 - free.sum(dim=-1, keepdim=True)       # N2 = 残差，无自由参数
```

N2 没有任何可学习单元。`N2_pred = 100 − (H2+CH4+CO2)_pred`，所以 N2 误差等于三个自由组分误差之和。DL 的 CH4 R2 仅 0.534（CH4 ≈ 76%，绝对误差大），这份误差整份传导到 N2（N2 std ≈ 5.8）。R2 被压到 ≈ 0 是结构必然，与特征里是否含 N2 信息无关。

即使在损失里给 N2 高权重也无效（`weighted_component_mse` 对 N2 用 1/var 权重，N2 权重很大却仍学不到），因为没有独立 N2 参数可调，只能通过联调 H2/CH4/CO2 间接降低 N2 误差，与它们自身的误差项冲突。这正是此前观察到的"N2 专用损失放大不可行"的真实成因。

## 三种 head 的 N2 机制对比

`src/dl/models/phase_window_tcn.py:181-187`：

| output_mode | 实现 | N2 机制 | 闭包 |
|---|---|---|---|
| `raw4` | `Linear(h2, 4)` | 4 个完全独立输出，N2 有自己的参数 | 不保证 |
| `softmax100` | `Softmax(4) × 100` | 4 个都有 logit，softmax 耦合 | 软保证 sum=100 |
| `gas_head` | `GasHeadNormalize` | N2 = 100 − sum(3)，纯残差 | 硬保证 |

## 更要紧的战略发现

ridge 在**全部四个组分**上都超过最好的 DL：

| 模型 | H2 | CH4 | CO2 | N2 |
|---|---|---|---|---|
| ridge_multiwindow（线性，同特征） | 0.994 | 0.920 | 0.977 | 0.712 |
| handcraft_mlp（非线性，gas_head） | 0.950 | 0.728 | 0.911 | -0.007 |
| PhaseWindowTCN gas_varweight | 0.718 | 0.534 | 0.544 | 0.0003 |

手工特征对线性模型几乎是充分的。这逼出一个问题：**PhaseWindowTCN 主线现在凭什么打过 "手工特征 + ridge"？** DL 的价值若在于从原始波形端到端学特征，需要拿出超过该线性基线的证据，否则 DL 主线本身需要重新论证。

## 为什么"方向 F"是错的修法

之前的方向 F 想在 gas_head 内加"线性旁路直达 N2"。但 gas_head 里 N2 没有输出节点，无处可接。且根本不需要线性旁路——问题是不该把 N2 做成残差，换 head 即可，比加旁路简单得多。

## 建议的解耦实验

代表性单 run（gas_varweight 同款数据，compile 基线），对照现有 gas_head run：

| run | output_mode | loss | N2 机制 | 闭包 | 预期（若残差是主因） |
|---|---|---|---|---|---|
| 对照（已有） | gas_head | weighted_component_mse | 残差 | 硬保证 | N2 ≈ 0（实测） |
| S1 | raw4 | weighted_component_mse | 4 独立输出 | 不保证 | N2 ↑ 接近 ridge 0.71 |
| S2 | softmax100 | weighted_component_mse | 4 个 logit | 软保证 | N2 ↑ 且闭包保留 |

- S1 是 ridge 的非线性版，验证"给 N2 独立参数能否恢复"。
- S2 是潜在生产修法：softmax over 4 既给 N2 直接 logit，又保住 sum=100。
- 判读：若 S1 或 S2 的 N2 R2 跳到 0.5 以上 → 残差参数化确认为主因，gas_head 应被 softmax100 取代；若仍 ≈ 0 → 非线性假设才成立，再考虑其他方向。

### 需要的代码改动

`validate_loss_model_output`（`src/dl/training/losses.py:182-188`）目前把 `weighted_component_mse` 强制绑死 gas_head。该约束是人为的——`WeightedComponentMSELoss` 对全 4 列算 loss，与 head 无关。需放开，允许 `weighted_component_mse` 配 `raw4` / `softmax100`。

注意：`free_component_mse` 与 `weighted_free_component_mse` 只监督 3 列、靠闭包补 N2，应继续锁定 gas_head；只放开监督全 4 列的 `weighted_component_mse`。
