# PhaseWindowTCN 诊断与结构消融实验方案

> 更新日期：2026-06-16
> 状态：待执行
> 目标：在 `gas_head` 与 `free_component_mse` 负结果之后，先用低成本诊断定位 N2 负 R2 的真正机制，再决定结构消融是否值得投入；据此判断 DL 线是否还能稳定改善 N2
> 本版改动：把"直接上结构消融（split/deep）"调整为"先诊断、后消融"。原因是对失败现象和文献的复核显示，N2 负 R2 更可能来自损失尺度与监督方式，而不是窗口编码器结构。

## 0. 本版修订说明（为什么改）

上一版方案把第一批实验直接定为 `share_window_encoder=false`（split）和更深 TCN（deep），即两个"增加模型容量"的结构改动。复核代码、已有结果和文献后，发现这个顺序有三个值得修正的地方：

1. **失败现象与所选变量方向相反**。`free_component_mse` 在 `losses.py` 中用的是无权重 `nn.MSELoss`，只监督前 3 个自由组分（H2/CH4/CO2）。N2 均值约 5%、CO2 均值约 76%，在原始百分比尺度上联合优化时，大尺度组分会主导梯度，小尺度的 N2 容易被忽略——这正是"小尺度目标出现负 R2"的典型多任务现象。split/deep 改的是输入编码器，改不到这个机制。

2. **N2 完全没有直接监督**。`gas_head` 下 `N2 = 100 − H2 − CH4 − CO2` 是纯闭包残差，而 `free_component_mse` 不监督 N2。已有对照里 `gas_4mse`（监督 N2）的 test N2 R2 = −0.0066，`gas_free`（不监督 N2）= −0.0155，去掉 N2 监督让 N2 略变差。方案却选了最不监督 N2 的 `gas_free` 作为消融基线。

3. **split/deep 都在增容，但现象更像快速过拟合**。MVP 的 `best epoch = 4` 是典型的早期过拟合信号。文献一致指出：数据有限时共享 encoder 起正则化作用、等效放大训练样本；过拟合应先加正则，而不是先加容量。

因此本版把实验拆成三批：先做几乎零成本、信息量最高的损失/监督诊断（第一批），用结果决定是否值得做结构消融（第二批），最后才考虑融合与对数比（第三批）。文件名保留不变，因为它被 README、IMPLEMENTATION_PLAN 等 5 处引用；"结构消融"仍是方案主体，只是前面增加了诊断阶段。

## 1. 问题背景

传统 ML 的多窗口路线已经成立：

- `ridge_multiwindow_all_modalities`（`full + exposure + recovery`）显著提升 N2
  - test N2 R2 = 0.7121，extrapolation N2 R2 = 0.7247，test overall R2 = 0.9253，macro RMSE = 2.4133
- 数据源 `data/wv4-formal-hitran-standard-6000`，seed 20260603，各评估 split 为百级样本量，结果统计可信
- 该结果说明三窗口信息本身对 N2 有效，DL 失败不能解释为"数据没有 N2 信号"

PhaseWindowTCN 近期已验证的事实：

- `gas_head` 修复了闭包，`sum_abs_error` 已接近 0
- `free_component_mse` 提升 overall R2（0.5145），但 N2 R2 仍为负（test −0.0155，extrapolation −0.0396）
- 继续围绕 head/loss 做无方向的小调参收益有限

结论是：**下一步先做诊断，确定 N2 负 R2 来自哪一层，再决定是否做结构消融。**

## 2. 已知事实

### 2.1 代码现状

- `src/dl/models/phase_window_tcn.py`
  - `PhaseWindowTCNRegressor` 支持 `share_window_encoder`、`output_mode ∈ {raw4, softmax100, gas_head}`
  - 窗口编码器 `WindowedFusionEncoder.forward` 末端做 `last + mean + max` 三种时间池化（`phase_window_tcn.py:102`），**全局时序统计量已经被 mean/max 捕获**，与卷积感受野无关
  - 名义感受野：3 block(k=3, dilation 1/2/4) → RF=29；5 block(dilation 1..16) → RF≈125
- `src/dl/training/losses.py`
  - 已注册 `mse`、`compositional_mse`、`ilr_mse`、`free_component_mse`、`mae`、`smooth_l1`、`huber`
  - `FreeComponentMSELoss` 用无权重 `nn.MSELoss`，监督前 3 个自由组分
  - **ILR 基础设施已存在**：`ilr_mse` + `target_transform='ilr_n2_first'`（`ILR_N2_FIRST_TRANSFORM`），配 `out_dim=3` 的 `GasCoordinateHead`
  - `build_loss` 支持 loss 配置为带 `name` 的字典并把其余键作为构造参数——便于注册带权重的新 loss
- `src/pipeline/experiment_config.py`
  - 已支持 `phase_windows`，并区分 DL/ML 窗口语义

这意味着当前不是"补实现"，而是"选下一轮实验方向"，且部分诊断臂只需改配置，部分需新增一个 loss 类。

### 2.2 关键数值

| 实验           | output_mode | loss               | test N2 R2 | extrap N2 R2 | overall R2 | sum_abs_error |
| ------------ | ----------- | ------------------ | ---------- | ------------ | ---------- | ------------- |
| MVP          | raw4        | mse                | −0.0150    | 0.0028       | 0.2635     | 11.18         |
| gas_4mse     | gas_head    | mse(4 组分)          | −0.0066    | —            | 0.4798     | ≈2e-6         |
| gas_free（基线） | gas_head    | free_component_mse | −0.0155    | −0.0396      | 0.5145     | ≈2e-6         |
| Ridge 多窗口    | —           | —                  | **0.7121** | **0.7247**   | 0.9253     | —             |

`output_prior = (9.29, 75.76, 4.99, 9.96)`，组分顺序对应内部自由组分 + N2 残差，N2 是均值最小的组分。

## 3. 重新诊断：N2 负 R2 的候选机制

以下按"证据强度 + 验证成本"排序，作为第一批诊断的设计依据。

### 3.1 候选 A：损失尺度被大组分主导（当前最可疑）

`free_component_mse` 在原始百分比上对 H2/CH4/CO2 做等权 MSE。CO2 均值约 76%、N2 约 5%，大尺度组分的平方误差天然更大，联合梯度被其主导，小尺度目标被忽略。

文献支持（置信度：高）：

- 多任务回归中，任务损失尺度差异会导致联合梯度偏向高量级任务，使学习集中在高量级任务上（[Multi-Task Learning with DNN: A Survey, arXiv:2009.09796](https://arxiv.org/pdf/2009.09796)）。
- 小尺度/少数目标出现负 R2，是其被优化器忽略的典型信号；最直接的修正是对每个输出独立标准化（零均值、单位方差）后再计算损失，使各目标对损失与梯度的贡献变成尺度无关（[Loss Functions and Metrics in DL, arXiv:2307.02694](https://arxiv.org/html/2307.02694v5)；[Neural Regression for Scale-Varying Targets, arXiv:2211.07447](https://arxiv.org/pdf/2211.07447)）。
- 直接给少数目标乘大权重往往失败（梯度不稳、主干特征偏移），方差自适应加权（如 SLAW、PopArt 风格的尺度无关更新）更稳（[SLAW, arXiv:2109.08218](https://arxiv.org/pdf/2109.08218)；[Metal Alloy MTL Negative Transfer, arXiv:2512.22740](https://arxiv.org/pdf/2512.22740)）。

含义：若 A 成立，按 `1/σ_c²` 给各组分加权（等价于按训练集方差标准化）就能改善 N2，**不需要动 encoder 结构**。

### 3.2 候选 B：N2 作为无监督闭包残差

`gas_head` 下 N2 是 100 减去三个自由组分，`free_component_mse` 不监督 N2，N2 误差完全由三个大组分的绝对误差决定。CO2 上 1% 的相对误差，落到 5% 的 N2 上就是约 15% 的相对误差。

文献支持（置信度：高）：

- 在常数和（闭包）约束下预测留出/残差组分，出现负 R2 是已知统计现象，因为闭包嵌入了组分间伪相关，几何空间是单纯形而非欧氏空间（[α-regression for compositional data, arXiv:2510.12663](https://arxiv.org/pdf/2510.12663)）。
- softmax 头本身把输出约束在 ALR 空间的对数线性流形上，不满足该约束的真值无法被表示；规避该约束的标准做法是直接预测 d−1 个对数比（ALR/ILR）再反变换回单纯形（[Adaptation of CoDA in Deep Learning, CEUR Vol-3105/paper43](https://ceur-ws.org/Vol-3105/paper43.pdf)）。

含义：给 N2 加直接监督（在闭包下监督全部 4 组分），或在 DL 侧改用对数比目标，是对应 B 的修正。注意：项目已确认 ML 不需要 ILR/ALR（不带对数比的 Ridge 已达 0.71），所以对数比只作为 DL 侧的受限对照臂，不进入正式主线。

### 3.3 候选 C：从原始波形端到端学特征不如手工特征（数据规模相关）

Ridge 用的是每个窗口的手工统计特征 + L2 正则；DL 从 1000/2000 点原始波形端到端学特征。文献一致认为：纯端到端深度模型更"吃数据"，注入手工统计特征的混合模型在数据有限时更稳、常优于纯端到端（[PatternFusion, Nature Sci Rep 2025](https://www.nature.com/articles/s41598-025-28649-4)；[TreNet, IJCAI 2017](https://www.ijcai.org/Proceedings/2017/316)；M4 竞赛冠军 ES-RNN 即统计+神经网络混合）。

含义：把 Ridge 用的手工窗口统计特征喂给一个小 MLP head（同 split、同 seed），是一个判别性诊断——它能区分"瓶颈在波形特征学习"还是"瓶颈在目标/损失侧"。

### 3.4 候选 D：窗口表征 / 相位融合不足（原方案的主假设，现降为候选之一）

共享 encoder 可能稀释 full/exposure/recovery 的相位差异；简单 concat 没有显式建模窗口互补。文献对"共享 vs 独立 encoder"给出的是依数据而定的结论：独立 encoder 只有在分支输入分布真正异质或需不同归纳偏置时才占优；数据有限时共享 encoder 反而是正则（[Multi-Encoder Roles, arXiv:2606.03879](https://arxiv.org/html/2606.03879v1)；[Correlative Channel-Aware Multi-View, arXiv:1911.11561](https://arxiv.org/pdf/1911.11561)）。本项目三窗口是同一样本的子段，异质性不强，所以 split 占优的先验不高。

另外，由于 encoder 已做 mean/max 全局池化，"RF 只覆盖 11%"的说法夸大了缺口——deep TCN 只扩大池化前的局部上下文整合，不是从无到有获得全局信息。所以 deep 是结构消融里更弱的一个变量。

### 3.5 候选 E：过拟合 vs 优化问题

`best epoch = 4` 是早期过拟合的典型信号（数据有限 + 容量偏大 + 缺正则）。文献建议：过拟合应先用早停 + 权重衰减 + dropout + 时序数据增强，并保持结构简单；增容是针对欠拟合的，方向相反（[How to Avoid Overfitting in DL NN, MachineLearningMastery](https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/)）。

含义：任何结构消融都应在固定的正则口径下进行；并且要先用 train/val 曲线确认是过拟合还是优化停滞，再决定加容量还是加正则。

## 4. 文献依据（已联网核实）

| 文献                                                                                                                                      | 用途                                | 核实状态                                 |
| --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ------------------------------------ |
| Bai, Kolter, Koltun 2018, *Empirical Evaluation of Generic Conv & Recurrent Nets*, [arXiv:1803.01271](https://arxiv.org/abs/1803.01271) | TCN 有效感受野需实测                      | 真实，表述准确                              |
| Chen et al. 2017, *GradNorm*, [arXiv:1711.02257](https://arxiv.org/abs/1711.02257)                                                      | 多任务梯度平衡解决任务竞争，不解决表征不足             | 真实，表述准确                              |
| Luo et al. 2025, *PAF-Net*, [arXiv:2507.22840](https://arxiv.org/abs/2507.22840)                                                        | 相位对齐 + DCT + 频率解耦 cross-attention | 真实（2025-07-30），模块描述吻合；本项目无跨过程时滞，仅作备选 |
| Multi-Task Learning Survey, [arXiv:2009.09796](https://arxiv.org/pdf/2009.09796)                                                        | 损失尺度主导联合梯度                        | 真实                                   |
| Scale-Varying Targets, [arXiv:2211.07447](https://arxiv.org/pdf/2211.07447)                                                             | 每输出标准化是小尺度目标的直接修正                 | 真实                                   |
| SLAW, [arXiv:2109.08218](https://arxiv.org/pdf/2109.08218)                                                                              | 方差自适应加权优于固定大权重                    | 真实                                   |
| Metal Alloy MTL Negative Transfer, [arXiv:2512.22740](https://arxiv.org/pdf/2512.22740)                                                 | 固定大权重导致梯度不稳，非平衡                   | 真实                                   |
| CoDA in Deep Learning, [CEUR Vol-3105/paper43](https://ceur-ws.org/Vol-3105/paper43.pdf)                                                | softmax 的对数线性约束；预测 d−1 对数比        | 真实                                   |
| PatternFusion, [Nature Sci Rep 2025](https://www.nature.com/articles/s41598-025-28649-4)                                                | 手工特征注入混合模型在有限数据更稳                 | 真实                                   |
| TreNet, [IJCAI 2017](https://www.ijcai.org/Proceedings/2017/316)                                                                        | 统计特征 + 神经网络融合                     | 真实                                   |

## 5. 实验设计

### 5.1 第一批：低成本诊断与损失/监督修正（先做）

所有实验同 seed、相对同 seed 的 `phase_window_tcn_gas_free` 判断；评估始终在原始百分比空间算 R2，便于和 Ridge、历史结果直接比较。

| 实验名                        | 主要改动                                     | 对应候选  | 实现成本        |
| -------------------------- | ---------------------------------------- | ----- | ----------- |
| `pwtcn_gas_free`（基线）       | 当前 gas_head + free_component_mse         | —     | 无           |
| `pwtcn_gas_varweight`      | gas_head + 按训练集方差加权的 4 组分 MSE（含 N2 直接监督） | A + B | 新增一个 loss 类 |
| `pwtcn_gas_free_varweight` | gas_head + 仅对 3 自由组分做方差加权 MSE（N2 仍残差）    | A     | 新增一个 loss 类 |
| `pwtcn_handcraft_mlp`      | 用 Ridge 同款手工窗口统计特征喂小 MLP（同 split/seed）   | C     | 需新增一个小模型/脚本 |

诊断读法：

- 若 `gas_varweight` 让 N2 转正或明显改善 → 主因是损失尺度 + N2 监督（候选 A/B），结构消融可降级或免做。
- 若仅 `gas_free_varweight` 改善而 `gas_varweight` 更好 → 直接监督 N2 是关键。
- 若 `handcraft_mlp` 能让 N2 ≈ 0.7 → 瓶颈是从原始波形学特征（候选 C），在原始波形 encoder 上做 split/deep 不会有用，应转向特征注入混合模型。
- 若 `handcraft_mlp` 也失败 → 瓶颈在目标/损失/闭包侧，而非窗口/融合。

每个实验都必须保存 train/val/test 三条曲线和 per-component R2（`metrics_live.jsonl` 已有逐 epoch 记录，summary CSV 已有 `x_n2_r2`），用于区分过拟合（候选 E）与优化停滞。

### 5.2 第二批：结构消融（仅当第一批未定位到损失/监督主因时）

进入条件：第一批的损失/监督修正**未能**让 N2 明显改善，且 `handcraft_mlp` 表明特征学习不是唯一瓶颈。

| 实验名                         | 主要改动                            | 目的                            |
| --------------------------- | ------------------------------- | ----------------------------- |
| `pwtcn_gas_free_split`      | `share_window_encoder=false`    | 验证窗口是否需要独立编码器                 |
| `pwtcn_gas_free_deep`       | `tcn_channels=[64,64,64,64,64]` | 验证感受野是否不足（注意已有 mean/max 全局池化） |
| `pwtcn_gas_free_split_deep` | 独立 encoder + 深 TCN              | 仅当 split 或 deep 任一有正信号时再做     |

约束：第二批必须在固定正则口径下进行（统一的 weight decay、dropout、早停），因为现象提示过拟合；split 会把 encoder 参数 ×3，要监控 val 曲线是否更早过拟合。

### 5.3 第三批：融合与对数比对照（条件触发）

仅当第二批出现正信号但不足以达标时：

| 实验名                    | 主要改动                                                          | 目的                    |
| ---------------------- | ------------------------------------------------------------- | --------------------- |
| `pwtcn_gas_free_gated` | gated fusion                                                  | 窗口级自适应加权是否优于纯 concat  |
| `pwtcn_gas_free_attn`  | 轻量 attention（仅在 pooled 窗口特征上）                                 | 窗口交互是否有帮助             |
| `pwtcn_ilr`（受限对照）      | `ilr_mse` + `target_transform='ilr_n2_first'` + `out_dim=3` 头 | 对数比目标在 DL 侧是否缓解闭包残差问题 |

`pwtcn_ilr` 只作为 DL 侧机制对照，不进入正式主线（ML 已证明不需要对数比）。

### 5.4 暂缓项

完整频率解耦 cross-attention、DCT 分解、相位对齐插值、phase token 扩展、component-aware gated fusion、GradNorm/不确定性加权——只有在前面批次明确支持时再考虑（详见第 10 节备选方案）。

## 6. 推荐配置

### 6.1 基线延续（不变）

```json
{
  "name": "pwtcn_gas_free",
  "model": "phase_window_tcn",
  "modalities": ["slow", "ultrasonic", "fiber_mic"],
  "phase_windows": [null, {"kind": "phase", "value": "exposure"}, {"kind": "phase", "value": "recovery"}],
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

### 6.2 方差加权 4 组分（候选 A/B，需新增 loss）

需在 `losses.py` 注册一个按训练集方差加权的组分 MSE，权重 `w_c = 1/σ_c²`（σ_c 取训练集组分标准差），在 gas_head 输出的 4 组分百分比上计算，N2 获得直接但尺度平衡的监督：

```json
{
  "name": "pwtcn_gas_varweight",
  "loss": {"name": "weighted_component_mse", "weighting": "inverse_train_var"},
  "model_kwargs": {"output_mode": "gas_head"}
}
```

实现要点：新 loss 类在 `build_loss` 已支持的"字典 + 额外 kwargs"机制下注册；`weighting="inverse_train_var"` 时从训练集统计读取各组分方差。eval 仍在原始百分比空间算 R2。

### 6.3 仅自由组分方差加权（候选 A 的对照）

```json
{
  "name": "pwtcn_gas_free_varweight",
  "loss": {"name": "weighted_free_component_mse", "weighting": "inverse_train_var"}
}
```

### 6.4 手工特征 + 小 MLP（候选 C，判别诊断，需新增小模型）

复用 Ridge 多窗口的手工窗口统计特征提取，接一个 2 层 MLP + gas_head：

```text
features = ridge_window_stats(full, exposure, recovery)   # 复用现有特征工程
z = MLP(features)                                          # [128, 64]
out = gas_head(z)                                          # 保持闭包
loss = weighted_component_mse 或 free_component_mse
```

目的：检验 N2 信号能否进入一个最简 DL 头。这是判别性诊断，不是候选主线。

### 6.5 Split / Deep（第二批，仅条件触发，配置即可）

```json
{"name": "pwtcn_gas_free_split", "model_kwargs": {"share_window_encoder": false}}
```

```json
{"name": "pwtcn_gas_free_deep", "model_kwargs": {"tcn_channels": [64, 64, 64, 64, 64]}}
```

## 7. 运行顺序

1. **第一批（诊断）**：`pwtcn_gas_free`（基线）、`pwtcn_gas_varweight`、`pwtcn_gas_free_varweight`、`pwtcn_handcraft_mlp`，同 seed 一起跑，先 dry-run。
2. 按 5.1 的读法判断主因落在哪个候选。
3. **若损失/监督修正已让 N2 明显改善** → DL 线的瓶颈被定位为损失尺度/监督，记录结论；是否再做结构消融取决于是否还想进一步逼近 ML。
4. **若修正无效且 `handcraft_mlp` 提示特征学习仍是瓶颈之一** → 进入第二批结构消融（split/deep，固定正则口径）。
5. 第二批有正信号但不达标 → 进入第三批（gated/attn/ilr 对照）。
6. 任一阶段若明确无正向 N2 收益 → 停止 DL 线扩展，正式主线保持 `ridge_multiwindow_all_modalities`，把本轮作为 DL 负结果证据。

配置目录：`configs/experiment/phase_window_tcn_ablation/`（第一批写入主配置，第二批写入 followup）。

## 8. 验收口径

- 首要指标：`test x_N2 R2`、`extrapolation x_N2 R2`，二者必须同时改善
- `sum_abs_error` 维持接近 0（闭包不能被破坏）
- 其他三组分 R2 不能系统性退化
- 每个实验必须报告 train/val/test 三条曲线和 per-component R2，用于区分过拟合与优化停滞
- 评估一律在原始百分比空间，便于和 Ridge（test N2 R2=0.7121）直接比较

最低门槛：`test x_N2 R2 > 0`、`extrapolation x_N2 R2 > 0`、`macro RMSE` 不高于当前负结果。

需要明确：该门槛远低于 ML 的 0.71。即便跨过 0，DL 仍可能远不及 ML 主线。**本轮的产出首先是"瓶颈定位 + go/stop 决策"，而不是一个有竞争力的 DL 模型。**

## 9. 决策框架

停止 DL 线继续投入的条件（任一成立）：

- 第一批损失/监督修正 + 第二批结构消融，都不能让 N2 在 test 与 extrapolation 上同时转正
- 出现正信号但复杂度显著增大而收益不稳定
- `handcraft_mlp` 失败，说明问题不在 DL 结构能解决的范围内

满足停止条件时，正式主线固定为 `ridge_multiwindow_all_modalities`，本方案的全部实验作为 DL 负结果证据归档。

## 10. 备选方案（保留，不进入前两批）

> 这些是前期讨论过的完整候选，作为后续可恢复路线记录，避免重复调研。它们只在前面批次明确支持时才考虑。

### 10.1 备选 A：分离式窗口编码 + 频率解耦注意力（PAF-Net 类）

核心：相位独立编码 + 频率域相位对齐 + DCT 频率分解 + 频率解耦 cross-attention。

```python
phase_encoders = {p: TCNEncoder(n_blocks=3, shared=False) for p in ("full", "exposure", "recovery")}
encoded = [phase_encoders[p](x[p]) for p in ("full", "exposure", "recovery")]
aligned = phase_correlation_alignment(encoded)
fused = frequency_decoupled_cross_attention(aligned, n_frequencies=5)
```

- 优点：显式建模窗口差异，适合相位错位 + 频段噪声场景
- 风险：本项目三窗口是同一样本不同视图，不一定有 PAF-Net 的跨过程时滞；多模块同时引入后失败难定位；过拟合风险显著高
- 启用条件：split 或 deep 明确提升 N2 但未达标，且有证据显示窗口存在明显错位或频段差异
- 当前决策：暂缓

### 10.2 备选 B：多任务学习 + 动态损失权重

把四组分视为多任务回归，组分特定 head + 动态权重（静态权重 / 方差加权 / GradNorm / 不确定性加权）。

- 与闭包冲突：四个独立 head 会重新引入 `sum != 100`，必须配套 `raw4 + closure_penalty`、`softmax100`、或预测三自由组分 + 闭包残差之一
- 注意：本方案 5.1 的 `weighted_component_mse` 已经把"方差加权 + 直接监督 N2"以更低成本纳入第一批，且不破坏 `gas_head` 闭包，应优先于本备选
- GradNorm/不确定性加权实现复杂（需改训练循环、维护每任务梯度统计），只在静态/方差加权都不足时再考虑
- 当前决策：低优先级保留

### 10.3 备选 C：轻量窗口融合模块

在结构消融有正信号后，替换简单 concat：

```python
weights = softmax(gate([z_full, z_exposure, z_recovery]))
z = sum(w_i * z_i)
```

- 可选：gated fusion / component-aware gated fusion / 仅在 pooled 特征上的轻量 attention
- 风险：若 split/deep 本身无收益，gating 只是在无效特征上重加权；component-aware 增参易过拟合
- 当前决策：作为第三批保留

### 10.4 备选 D：完整频域相位融合

方案 A 的完整版（phase correlation alignment + DCT 多频段分解 + frequency-independent patch attention + frequency-decoupled cross-attention）。仅在以下全部满足时讨论：

1. 简单结构消融已证明 DL 多窗口路线有正信号
2. 轻量 gated fusion / attention 仍不足
3. 有可视化或统计证据说明不同窗口有效信息位于不同频段
4. 训练资源允许做多组消融

当前不建议实施。
