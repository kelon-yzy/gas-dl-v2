# PhasePreservingTCN 详细设计文档

**版本**: v1.1  
**日期**: 2026-06-12  
**目标**: 在不破坏现有 DL pipeline 的前提下，验证 DL 是否能复用 `full + exposure + recovery` 多窗口信息，追近或超过 ML 多窗口 ridge 的 N2 表现。  
**状态**: 设计修订完成，待实现 MVP。

---

## 0. 结论先行

原 v1.0 方向正确，但不能按原设计直接实现。主要原因：

1. 三个 `PhaseSpecificExtractor` 都吃同一个 `x_full`，只加不同 phase token，并没有真正复现 ML 成功的 `full/exposure/recovery` 三窗口信息差异。
2. 文档中的 `input_dim=300` 与现有 DL 数据契约不符；当前三模态输入是 `slow[8] + ultrasonic[1000] + fiber_mic[2000] = 3008`。
3. 直接对 3008 通道做 Conv1d 会把波形采样轴当成普通通道，绕开现有 `DeepAcousticEncoder1D` 的波形归纳偏置。
4. 当前 DL 的 `GasHeadNormalize` 默认用前三组分推出 N2 残差，这可能继续压制 N2 学习；新模型必须显式处理输出头。
5. 配置样例里的 `cosine_annealing` scheduler 当前不被 `experiment_config.py` 接受。

因此本设计改为两阶段：

- **MVP**：`PhaseWindowTCN`。真实构造 `full + exposure + recovery` 三个窗口视图，复用现有 acoustic/slow encoder，各窗口独立 TCN + pooling，最后 concat + MLP 输出。
- **增强版**：在 MVP 通过后，再引入 cross-attention、phase token、phase shift、可视化和消融。

MVP 的目标不是炫技，而是回答一个最关键的问题：**DL 失败是因为没有显式看到多窗口信息，还是因为输出头/训练目标本身限制了 N2？**

---

## 1. 当前事实与约束

### 1.1 已完成实验事实

`ridge_multiwindow_all_modalities` 已在服务器结果中强通过：

| 指标 | full baseline | multiwindow | 变化 |
|---|---:|---:|---:|
| test N2 R² | 0.2173 | 0.7121 | +0.4947 |
| extrapolation N2 R² | 0.2273 | 0.7247 | +0.4974 |
| test macro RMSE | 3.9810 | 2.4133 | -1.5677 |

其他组分 H2/CH4/CO2 相对 full baseline 均无有效 R² drop，反而提升。

该结果已归档：

```text
outputs/archive/multiwindow_n2_20260612/result_analysis.md
```

### 1.2 当前 DL 失败事实

`cnn1d_tcn_fusion` 的单窗口候选均未通过：

- full
- phase:exposure
- phase:recovery
- early:0.50
- early:0.75

失败不能简单归因于“没有 N2 信号”，因为 ML 多窗口已证明信号存在。更合理的解释是：

- 单窗口 DL 只看到一个窗口视图，和 ML 最优输入不一致。
- 现有 DL 输出头让 N2 作为闭包残差，N2 误差受前三组分误差耦合。
- DL 参数更多，若输入信号组织方式不对，会比 ridge 更容易学偏。

### 1.3 现有代码契约

当前 `V4BenchmarkDataset`：

- 返回 `(x, y)`，不返回 phase mask。
- `window=None` 返回完整序列。
- `window={"kind":"phase","value":"exposure"}` 会裁出 exposure 后重采样回原始 timestep 数。
- 多模态 `input_format="NTC"` 时，单样本形状为：

```text
(T, 3008)
3008 = slow[8] + ultrasonic[1000] + fiber_mic[2000]
```

当前 `cnn1d_tcn_fusion`：

- 输入格式：`NTC`
- 先切分 slow / ultrasonic / fiber_mic
- ultrasonic/fiber_mic 每个 timestep 用 `DeepAcousticEncoder1D` 编码
- slow 每个 timestep 用 `SlowFeatureEncoder` 编码
- 再做 TCN + pooling + head

当前 `experiment_config.py`：

- `windows` 只允许 ML run 使用。
- DL 仍只支持单个 `window`。
- scheduler 只允许 `none` 和 `reduce_on_plateau`。

---

## 2. 修订后的设计目标

### 2.1 MVP 目标

1. **真实多窗口输入**：DL 明确看到 `full + exposure + recovery` 三个窗口视图。
2. **复用现有编码器**：不重写波形编码逻辑，沿用 `DeepAcousticEncoder1D` 和 `SlowFeatureEncoder`。
3. **减少变量**：第一版不做 cross-attention、不做 phase shift、不做复杂 phase token。
4. **输出头解耦 N2**：避免默认把 N2 作为前三组分闭包残差的唯一输出方式。
5. **可快速失败**：用最小实现判断 DL 多窗口是否值得继续投入。

### 2.2 性能目标

| 指标 | MVP 通过线 | 强通过线 |
|---|---:|---:|
| test N2 R² | > 0.50 | > 0.65 |
| extrapolation N2 R² | > 0.50 | > 0.70 |
| 其他组分 R² drop | <= 0.05 | <= 0.02 |
| macro RMSE regression | <= 0 | 明显下降 |

说明：ML 多窗口已达到 test N2 R² 0.7121。DL 第一阶段只要求证明“多窗口 DL 有效”，不要求立刻超过 ML。

### 2.3 时间目标

不能再写 `< 2x ML`。ridge 是秒级，DL 不可能只慢 2 倍。

合理目标：

- smoke forward / train 测试：本机分钟级。
- formal 单卡训练：目标 < 2 小时。
- 若单个 run 超过 2 小时仍无明显 val N2 改善，停止该架构继续调参。

---

## 3. MVP 架构：PhaseWindowTCN

### 3.1 总体结构

```text
输入三窗口:
  x_full      (B, T, 3008)
  x_exposure  (B, T, 3008)  # exposure mask 后重采样回 T
  x_recovery  (B, T, 3008)  # recovery mask 后重采样回 T

每个窗口共享或独立编码:
  slow encoder
  ultrasonic waveform encoder
  fiber_mic waveform encoder
  TCN temporal encoder
  temporal pooling

窗口级特征:
  z_full      (B, D)
  z_exposure  (B, D)
  z_recovery  (B, D)

融合:
  z = concat([z_full, z_exposure, z_recovery])  (B, 3D)

输出:
  MLP head -> raw composition prediction (B, 4)
```

### 3.2 为什么 MVP 不用 steady

ML 最优结果使用的是：

```text
full + exposure + recovery
```

不是：

```text
exposure + steady + recovery
```

`full` 已包含 baseline/steady/recovery 全局稳态信息；第一版不额外加入 steady，避免扩大输入和参数空间。若 MVP 通过，再做 `full + exposure + steady + recovery` 消融。

### 3.3 为什么 phase token 不是 MVP 核心

phase token 只能告诉模型“这个分支叫 exposure”，不能保证分支真的看到 exposure 数据。  
MVP 必须先保证数据视图真实不同：

- full branch 看完整序列。
- exposure branch 看 exposure mask 后重采样序列。
- recovery branch 看 recovery mask 后重采样序列。

phase token 可以作为增强版附加项，但不能替代 window/mask。

---

## 4. 模块设计

### 4.1 WindowedFusionEncoder

复用 `cnn1d_tcn_fusion.py` 中的三个底层编码器：

- `DeepAcousticEncoder1D`
- `SlowFeatureEncoder`
- `TemporalBlock`

推荐先实现一个窗口编码器：

```python
class WindowedFusionEncoder(nn.Module):
    input_format = "NTC"

    def __init__(
        self,
        slow_channels: int = 8,
        ultrasonic_channels: int = 1000,
        fiber_mic_channels: int = 2000,
        waveform_embedding_dim: int = 64,
        slow_embedding_dim: int = 64,
        tcn_channels: Sequence[int] = (64, 64, 64),
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.25,
    ):
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 3008)
        # return: pooled feature (B, D)
        ...
```

内部逻辑与 `CNN1DTCNFusionRegressor.forward` 一致：

1. 切分 `slow / ultrasonic / fiber_mic`
2. 对两个 waveform 逐 timestep 编码
3. 对 slow 逐 timestep 编码
4. concat timestep embedding
5. TCN
6. pooling：`last + mean + max`

### 4.2 PhaseWindowTCNRegressor

```python
class PhaseWindowTCNRegressor(BaseRegressor):
    input_format = "multiwindow_ntc"

    def __init__(
        self,
        in_channels: int = 3008,
        out_dim: int = 4,
        windows: tuple[str, ...] = ("full", "exposure", "recovery"),
        share_window_encoder: bool = True,
        output_mode: str = "raw_with_closure_penalty",
        ...
    ):
        ...

    def forward(self, x: dict[str, torch.Tensor], **kwargs: object) -> torch.Tensor:
        z = []
        for name in self.windows:
            z.append(self.encoder_for(name)(x[name]))
        return self.head(torch.cat(z, dim=-1))
```

如果当前训练器不方便接受 dict 输入，可以先在 Dataset 层把三窗口拼成一个张量：

```text
(B, W, T, C)
W = 3
```

模型 forward 接收：

```python
def forward(self, x):
    # x: (B, 3, T, 3008)
    full = x[:, 0]
    exposure = x[:, 1]
    recovery = x[:, 2]
```

这比 dict 更容易兼容 DataLoader 默认 collate。

### 4.3 输出头

MVP 不应默认继续使用 `GasHeadNormalize` 的“前三组分 + N2 残差”模式。

建议支持两个输出模式：

#### 模式 A：raw4 + closure penalty

模型直接输出 4 个组分百分比：

```python
pred = linear(features)  # (B, 4)
```

训练 loss：

```text
MSE(pred, y) + lambda_sum * MSE(pred.sum(dim=-1), 100)
```

优点：

- N2 有独立输出通道。
- 闭包约束仍可通过 loss 保持。
- 最适合验证“残差头是否限制 N2”。

需要修改训练 loss 或新增可选 loss。

#### 模式 B：softmax100

```python
pred = 100 * softmax(linear(features), dim=-1)
```

优点：

- 天然和为 100。
- 四组分都有独立 logit。

风险：

- softmax 会引入组分竞争，可能重新耦合误差。

MVP 优先级：A > B。  
若不想改 loss，先用 raw4 跑通，但分析时必须检查 sum error。

---

## 5. 数据与配置改动

### 5.1 Dataset 改动

新增一个 Dataset 或给 `V4BenchmarkDataset` 增加可选参数：

```python
phase_windows: tuple[WindowConfig | None, ...] | None = None
```

当 `phase_windows is None`：

- 保持当前行为，返回 `(x, y)`。

当 `phase_windows` 非空：

- 对每个 window 构造一个视图。
- `None` 表示 full。
- phase window 沿用现有 `_build_window_masks` 和 `_resample_masked_timesteps`。
- 返回：

```text
x.shape == (W, T, C)  # 单样本
y.shape == (4,)
```

DataLoader 后得到：

```text
(B, W, T, C)
```

不建议第一版返回 phase mask 给模型内部切分，因为现有训练器和 collate 都更容易处理固定张量。

### 5.2 Pipeline 配置

当前 `windows` 仅 ML 可用，因此 DL 需要新增字段，避免和 ML 多窗口语义混淆：

```json
"phase_windows": [
  null,
  {"kind": "phase", "value": "exposure"},
  {"kind": "phase", "value": "recovery"}
]
```

校验规则：

- `phase_windows` 只允许 DL run 使用。
- `phase_windows` 不得与单个 `window` 同时出现。
- `phase_windows` 必须非空。
- 元素可以是 `null` 或合法 `WindowConfig`。

`run_experiment._run_dl` 需要把 `phase_windows` 传给 DL CLI / Dataset。

### 5.3 训练配置样例

当前 scheduler 只支持 `none` / `reduce_on_plateau`，因此配置先使用 `reduce_on_plateau`。

```json
{
  "experiment_name": "phase_window_tcn_mvp",
  "dataset_dir": "data/wv4-formal-hitran-standard-6000",
  "output_root": "outputs",
  "seed": 20260603,
  "device": "cuda",
  "eval_splits": ["val", "test", "extrapolation"],
  "training": {
    "epochs": 300,
    "batch_size": 16,
    "num_workers": 8,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 2,
    "optimizer": "adamw",
    "lr": 0.0005,
    "weight_decay": 0.01,
    "loss": "mse",
    "early_stopping": {
      "enabled": true,
      "monitor": "val_loss",
      "patience": 30,
      "min_delta": 0.0,
      "mode": "min"
    },
    "scheduler": {
      "name": "reduce_on_plateau",
      "factor": 0.5,
      "patience": 10,
      "min_lr": 0.000001
    },
    "amp": {
      "enabled": true,
      "dtype": "float16"
    },
    "progress": {
      "enabled": true,
      "stdout": true,
      "jsonl": true,
      "jsonl_name": "metrics_live.jsonl"
    }
  },
  "ml_runs": [],
  "dl_runs": [
    {
      "name": "phase_window_tcn_mvp",
      "model": "phase_window_tcn",
      "modalities": ["slow", "ultrasonic", "fiber_mic"],
      "phase_windows": [
        null,
        {"kind": "phase", "value": "exposure"},
        {"kind": "phase", "value": "recovery"}
      ],
      "model_kwargs": {
        "share_window_encoder": true,
        "waveform_embedding_dim": 64,
        "slow_embedding_dim": 64,
        "tcn_channels": [64, 64, 64],
        "shared_hidden_dims": [128, 64],
        "output_mode": "raw4"
      }
    }
  ]
}
```

---

## 6. 实现计划

### 6.1 MVP 实现清单

1. `src/dl/data/dataset.py`
   - 增加 `phase_windows` 支持。
   - 多窗口时返回 `(W, T, C)`。
   - 保持原单窗口行为不变。

2. `src/dl/cli.py`
   - 增加 `--phase-windows` JSON 参数。
   - 配置 JSON 读取时支持 `phase_windows`。
   - 传入 Dataset。

3. `src/pipeline/experiment_config.py`
   - DL run 允许 `phase_windows`。
   - 禁止 `phase_windows` 与 `window` 混用。
   - ML 继续使用 `windows`，DL 使用 `phase_windows`，避免语义冲突。

4. `src/pipeline/run_experiment.py`
   - `_run_dl` 传递 `phase_windows`。
   - dry-run detail 记录 `phase_windows`。
   - summary 的 `window` 列可写 `multi:full+exp+rec` 或新增字段；第一版可沿用 `window` 文本。

5. `src/dl/models/phase_window_tcn.py`
   - 新增 `WindowedFusionEncoder`。
   - 新增 `PhaseWindowTCNRegressor`。
   - 复用 `DeepAcousticEncoder1D` / `SlowFeatureEncoder` / `TemporalBlock`。

6. `src/dl/models/registry.py`
   - 注册 `"phase_window_tcn"`。

7. 测试
   - Dataset 多窗口 shape。
   - 模型 forward shape。
   - registry build。
   - pipeline dry-run。
   - smoke train 最小闭环。

### 6.2 不在 MVP 做的事

以下内容先不做，避免一次引入太多变量：

- cross-attention
- phase shift / differentiable time warping
- attention 可视化
- auxiliary phase classification
- transformer 替换 TCN
- 多 seed ensemble
- 知识蒸馏

这些都放到 MVP 证明有效之后。

---

## 7. 实验计划

### 7.1 最小验证

先用 smoke dataset 验证工程闭环：

- Dataset 输出 `(B, 3, T, 3008)`。
- 模型 forward 输出 `(B, 4)`。
- 训练 1-2 epoch 不报错。
- metrics.json / run_config.json / metrics_live.jsonl 正常生成。

### 7.2 formal MVP

运行：

```text
phase_window_tcn_mvp
```

对比：

- `cnn1d_tcn_fusion`
- `ridge_all_modalities`
- `ridge_multiwindow_all_modalities`

验收：

| 验收项 | 阈值 |
|---|---:|
| test N2 R² gain vs DL full | > 0.10 |
| test N2 R² absolute | > 0.50 |
| other component max R² drop vs DL full | <= 0.05 |
| macro RMSE regression vs DL full | <= 0 |
| extrapolation N2 margin | >= -0.10 |

同时必须和 ML 多窗口对比，但不要求 MVP 立刻超过 ML。

### 7.3 消融顺序

若 MVP 有效，再按下面顺序加复杂度：

1. `share_window_encoder=true` vs `false`
2. `full+exposure+recovery` vs `full+exposure+steady+recovery`
3. pooling：`last+mean+max` vs `attention pooling`
4. output head：`raw4` vs `softmax100` vs `GasHeadNormalize`
5. concat fusion vs gated weighted fusion
6. cross-attention
7. phase token
8. phase shift

---

## 8. 风险与对策

| 风险 | 概率 | 影响 | 对策 |
|---|---:|---:|---|
| 多窗口 Dataset 改动破坏单窗口 DL | 中 | 高 | `phase_windows=None` 时完全走旧路径；补回归测试 |
| raw4 输出 sum 不稳定 | 中 | 中 | 记录 sum_abs_error；必要时加 closure penalty |
| 模型仍学不到 N2 | 中 | 高 | 对比 raw4 vs residual head，确认是否输出头瓶颈 |
| 共享窗口 encoder 表达力不足 | 中 | 中 | 消融 `share_window_encoder=false` |
| 独立窗口 encoder 过拟合 | 中 | 中 | dropout、weight_decay、减少 hidden dim |
| 训练时间过长 | 中 | 中 | 先固定 MVP 小模型；不做 attention |
| phase_windows 配置与 ML windows 混淆 | 中 | 中 | ML 字段 `windows`，DL 字段 `phase_windows`，校验互斥 |

---

## 9. 增强版 PhasePreservingTCN

只有当 `PhaseWindowTCN` 达到以下条件时，才进入增强版：

- test N2 R² > 0.50
- 或相比 `cnn1d_tcn_fusion` full 有明确 N2 gain 且其他组分不退化

增强版可以引入：

1. **Gated window fusion**
   - 对 `z_full/z_exposure/z_recovery` 做样本级权重。

2. **Cross-attention**
   - 在窗口级时序 embedding 上交互，而不是直接对原始 3008 通道交互。

3. **Phase token**
   - 作为窗口 identity 补充，而不是替代真实窗口。

4. **Phase shift**
   - 最后再做，因为实现复杂且失败风险高。

增强版目标：

| 指标 | 目标 |
|---|---:|
| test N2 R² | > 0.65 |
| extrapolation N2 R² | >= ML 多窗口或 margin >= -0.05 |
| 其他组分 drop | <= 0.02 |

---

## 10. 最终判定规则

### 10.1 继续投入 DL 的条件

满足任一：

- `phase_window_tcn_mvp` test N2 R² > 0.50，且其他组分不退化。
- 相比 `cnn1d_tcn_fusion` full，N2 gain > 0.10，且 macro RMSE 不退化。
- extrapolation N2 明显优于 ML 多窗口，证明 DL 泛化有价值。

### 10.2 停止投入 DL 的条件

满足任一：

- MVP 多窗口后 N2 仍接近 0。
- raw4 输出头后 N2 仍无改善。
- train N2 高、val/test N2 低，说明主要是过拟合。
- 需要大量复杂 attention/shift 才勉强追近 ML，但训练和解释成本过高。

若停止投入 DL，正式主线保持：

```text
ridge_multiwindow_all_modalities
```

并把 DL 作为负向探索记录。

---

## 11. 文档修订摘要

相对 v1.0，本版本做了以下修订：

- 把直接实现 `PhasePreservingTCN` 改为先实现 `PhaseWindowTCN` MVP。
- 明确三窗口必须是真实 window/mask，不允许只靠 phase token 模拟。
- 修正输入维度：三模态为 3008。
- 明确复用现有 `cnn1d_tcn_fusion` 的 acoustic/slow encoder。
- 明确 DL 新配置字段使用 `phase_windows`，不复用 ML 的 `windows`。
- 修正 scheduler 配置，先使用现有支持的 `reduce_on_plateau`。
- 把 cross-attention、phase shift、可视化和消融移到第二阶段。
- 明确输出头风险，要求 MVP 避免默认 N2 残差头。

---

**文档版本**: v1.1  
**最后更新**: 2026-06-12  
**状态**: 修订完成，建议按 MVP 实施。
