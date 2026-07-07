# DL 相位统计稳定提取与保留方案

日期：2026-06-27  
状态：方案整理；经 2026-07-04 复核，经验证据需在 v6-phys-strict 链路重新标定（见下方前提警告）

> **2026-07-04 链路版本前提警告（必读）**
>
> §2 的全部经验证据产自**旧仿真链路**（40kHz / 200kS/s / 16-bit / L_m 0.2-1.8m），对应 CSV 生成于 2026-06-10 ~ 06-24。方案定稿（2026-06-27）之后，链路于 2026-07-02（200kHz / 1MS/s / 20-bit / L_m≤0.3m）与 2026-07-03（v6-phys-strict：理想气声速 + WMS 热导 + 分数延迟 TOF）连续替换，旧 benchmark 已归档且不可用于新链路（见项目 `CLAUDE.md`）。
>
> 影响：N2 在 hydrogen_ng 中为 IR 惰性气体，其超出闭包的独立可观测性主要依赖声学 TOF / 声速；L_m 从 0.2-1.8m 塌缩到 0.18-0.28m（原因：200kHz 下长声程被 CH₄/CO₂ 弛豫吸收淹没）直接削弱该通道。故 §2.1 的 Ridge N2 R²≈0.712、§6.1 的验收门槛均在**物理已不同的信号链上测得**，v6 下可能够不到；一旦够不到，§9 停止条件会在 P2 触发。
>
> 处置：执行 P0 前先完成新增的 **P-1（v6 证据迁移验证，见 §5、§9）**——在 v6 数据上重建 Ridge multiwindow 基线，确认 N2 相位统计信号是否幸存。幸存则 P0–P9 照走（仅需按新 L_m 重定义相位窗口）；大幅衰减则方案前提失效，先改问题定义再谈 DL 改造。§1–§9 原文保留旧链路语境，作为“信号幸存后”的执行蓝本；一切数字与判据以 P-1 在 v6 的重新标定为准。

## 1. 结论

旧 `hg` 中相位统计没有被稳定提取和保留，不能靠普通 `Dropout` 根治。更可靠的路线是：

```text
显式相位统计资产（含 mixture_id split_key 与 composition_scheme）
  + TCN hidden probe 前置诊断（决定后续分叉）
  + 相位统计专属分支 + 辅助监督（aux_weight 须 sweep 或 GradNorm 化）
  + 窗口/模态辅助监督
  + balanced fusion + gated fusion（吸收 TFT/iTransformer gate 思想）
  + 模态级 / 窗口级 dropout
  + ridge multiwindow teacher 蒸馏（带蒸馏权重退火）
  + raw4 + weighted_component_mse（N2 真正独立参数化的最小形态）
```

核心判断：

1. `ridge_multiwindow_all_modalities` 已证明 `full + exposure + recovery` 的显式统计里有可学习信号。
2. 端到端 DL 目前没有稳定学到这些统计，也没有机制保证中间层保留它们。
3. 当前 `cnn1d_tcn_fusion` 已经做了 last/mean/max 三种 pooling，但 pooling 是对整序列、无相位窗口意识；元素级 `Dropout` 只能正则化局部神经元，不能解决 3008 维多模态序列中弱模态、弱窗口、弱组分被主模态压制的问题。
4. 应把”相位统计是否被提取”和”相位统计是否被最终预测头使用”分开诊断，TCN hidden probe 是最低成本的判定手段，应在改造前完成。

## 2. 本地证据

### 2.1 显式相位统计可学

`outputs/summary/multiwindow_n2_summary.csv` 中：

| run                                | window                 | split         | overall R2 | N2 R2  | RMSE   |
| ---------------------------------- | ---------------------- | ------------- | ----------:| ------:| ------:|
| `ridge_all_modalities`             | full                   | test          | 0.7968     | 0.2173 | 3.9810 |
| `ridge_multiwindow_all_modalities` | full+exposure+recovery | test          | 0.9253     | 0.7121 | 2.4133 |
| `ridge_multiwindow_all_modalities` | full+exposure+recovery | extrapolation | 0.9248     | 0.7247 | 2.4075 |

这说明旧 `hg` 的 N2 不是完全不可学；关键增益来自多窗口相位统计拼接。

### 2.2 端到端 DL 没有保住这些信息

`outputs/summary/formal_full_summary.csv` 中：

| run                | split | overall R2 | N2 R2   |
| ------------------ | ----- | ----------:| -------:|
| `cnn1d_tcn_fusion` | test  | 0.7138     | -0.0075 |
| `tcn`              | test  | 0.6008     | -0.0601 |
| `cnn1d`            | test  | 0.1831     | -0.0331 |

`outputs/summary/phaseaware_best_models_summary.csv` 中，DL 即使切成 exposure/recovery/early 窗口，N2 R2 仍基本在 0 附近，没有复现 Ridge 的相位窗口收益。

### 2.3 现有 phase-stat branch 尝试并未解决

`outputs/summary/dl_p3a_phase_stat_branch_summary.csv` 中：

| run                                | split | overall R2 | N2 R2   | sum_abs_error |
| ---------------------------------- | ----- | ----------:| -------:| -------------:|
| `cnn1d_tcn_fusion_raw4_phase_stat` | test  | -0.0851    | -0.1256 | 23.6406       |

这说明“把统计量拼进去”本身不够。当前 `src/dl/models/cnn1d_tcn_fusion.py` 里的 `PhaseStatMLP` 是旁路 MLP，最后只和 TCN pooled feature 拼接；没有统计分支专属 loss、窗口辅助头、模态平衡或蒸馏约束，因此统计信息仍可能被后续融合层忽略。

另外，2026-07-04 复核发现：

```text
data/wv4-formal-hitran-standard-6000/ 本地缺失；且 _archived_pre_200khz/ 与
_archived_pre_phys_strict/ 两个归档目录均不含它 —— 无备份可恢复，只能重生成
data/ 当前仅剩 wv4-smoke / sg4-smoke（sg4-formal 等已随链路切换归档）
```

即 P1/P2 的载体目录不可用，且**不存在“从备份恢复”选项**——只能用 v6-phys-strict 参数重跑 `python -m pipeline.generate_benchmark` 重生成。重生成含 10MS/s 过采样 + 分数延迟重采样，算力不轻；`data/hitran_cache/` 可部分复用，但 L_m 变更后 `hitran_cache_windowed/` 需刷新。重生成并按新 L_m 重定义窗口后，`phase_stats_path=auto` 类实验才可复验。

## 3. 根因拆解

### 3.1 提取问题

端到端 TCN/CNN 当前在 `src/dl/models/cnn1d_tcn_fusion.py:CNN1DTCNFusionRegressor.forward` 中已经做了三种 pooling：

```python
pooled = torch.cat([feats[:, :, -1], feats.mean(dim=-1), feats.amax(dim=-1)], dim=-1)
```

即 last / mean / max 都已存在。但 pooling 是**对整序列做**的，没有按 exposure / recovery 切窗。因此问题不是"TCN 学不到 mean/max"，而是"TCN pooling 没有相位窗口意识"，更弱的 std / delta / slope / peak_index 等也没有结构上的提取通道。

在三类模态和三段窗口混在一起时，模型还会优先拟合强信号：

- CH4、CO2 的 NDIR 直接响应；
- ultrasonic/fiber_mic 的高维波形幅度统计；
- full-window 全局趋势。

相位窗口中的弱统计，如 recovery slope、exposure delta、fiber_mic peak/energy 的细微变化，容易被池化和共享 encoder 抹平。

### 3.2 保留问题

即使前层捕获了相位统计，后续 `concat -> shared_head -> output_head` 也可能把它压掉。原因包括：

- 3008 维原始输入和 8 维 slow 通道天然不平衡；
- full、exposure、recovery 共用 encoder 时，相位语义被折中；
- 没有窗口级/模态级辅助头，梯度只来自最终输出；
- N2 在 `gas_head` 中没有独立输出参数时，会变成前三个组分误差的残差。

### 3.3 Dropout 的边界

当前 `cnn1d_tcn_fusion` 已经在 acoustic encoder 和 TCN block 用了 `acoustic_dropout=0.15` / `tcn_dropout=0.25` 这种元素级 `Dropout`。它们是 L2 风格的局部噪声正则，无法表达”禁止某一整条模态长期独占预测””禁止 full-window 压制 exposure/recovery”这一类结构不变量。

更合适的是结构级 dropout：

- `Modality Dropout`：训练时随机丢整个 slow/ultrasonic/fiber_mic 分支；
- `Window Dropout`：训练时随机丢 full/exposure/recovery 中某个窗口；
- `Branch Dropout`：随机丢 learned branch 或 stat branch，迫使两者都可用。

这些 dropout 只能作为平衡训练手段，不能替代显式统计监督。

## 4. 推荐方案

### 方案 A：固定相位统计资产

目标：先让相位统计成为可复验、可诊断的一等输入。

建议生成：

```text
features/phase_stats.npy
features/phase_stats_scaler.json
features/phase_stats_feature_names.json
features/phase_stats_manifest.json
```

manifest 至少记录：

- dataset slug；
- `composition_scheme`（`hydrogen_ng` / `syngas`）与 `background_fields`，用于后续场景迁移；
- split 主键：明确写 `split_key = "mixture_id"`（项目约定 split 主键是 `mixture_id`，不是 `sequence_id`，见 `CLAUDE.md`），并记录 split 版本；
- windows：`full`、`phase:exposure`、`phase:recovery`；窗口边界来源（piston 位置阈值 / 固定时间帧号）必须显式记录，包括阈值或帧号；**v6 链路 L_m 已塌缩到 0.18-0.28m，活塞行程与曝光/恢复时序随之改变，旧窗口边界不可直接沿用，必须在 v6 数据上重新标定**；
- modalities：`slow`、`ultrasonic`、`fiber_mic`；
- sequence statistics：`mean/std/min/max/last/delta/slope`；
- waveform frame features：`mean/std/mean_abs/max_abs/energy/peak_index`；
- scaler 是否只由 train split 拟合；
- feature order hash。

验收：

1. 用该资产重跑 `ridge_multiwindow_all_modalities`，应复现 test overall R2≈0.925、N2 R2≈0.712。
2. 若复现不了，先查窗口切分、特征顺序、scaler 和 split 对齐，不进入 DL 改造。

### 方案 B：建立 stats-only DL 基线

目标：把“统计资产有没有问题”和“波形端到端学习有没有问题”拆开。

实验：

```text
phase_stats.npy -> MLP -> raw4/softmax100 -> weighted_component_mse
```

对照：

- `ridge_multiwindow_all_modalities`
- `stats_only_mlp_raw4`
- `stats_only_mlp_softmax100`

判据：

- stats-only MLP 应明显优于当前端到端 baseline，硬指标建议 `stats_only_mlp_softmax100 test overall R² >= 0.88, N2 R² >= 0.55`（向 Ridge 0.925 / 0.712 靠拢，但允许 DL 拟合损失）；
- 若 stats-only MLP 也很差，优先查归一化、初始化、loss scale、输出头；
- 若 stats-only MLP 接近 Ridge，则说明统计资产有效，后续问题在融合和保留。

### 方案 C：相位统计专属分支与辅助头

目标：让统计分支必须独立完成预测，而不是只作为可被忽略的附加向量。

结构：

```text
phase_stats
  -> PhaseStatEncoder
  -> stats_embedding
  -> stats_head -> y_stats

raw waveform windows
  -> Window/Modality Encoder
  -> learned_embedding

[stats_embedding, learned_embedding]
  -> balanced fusion
  -> final_head -> y_final
```

损失：

```text
L = L_final
  + 0.2 * L_stats_head
```

如果 `stats_head` 单独效果好但 `final_head` 差，说明融合层压制了统计分支。

### 方案 D：窗口级与模态级辅助监督

目标：强制 `full`、`exposure`、`recovery` 以及每个模态都保留可预测信息。

建议辅助头：

```text
head_full
head_exposure
head_recovery
head_slow
head_ultrasonic
head_fiber
head_final
```

损失：

```text
L = L_final
  + 0.2 * mean(L_full, L_exposure, L_recovery)
  + 0.1 * mean(L_slow, L_ultrasonic, L_fiber)
```

注意：`0.2 / 0.1` 是初值，不要直接固化。任选一条处理路径：

1. 显式 sweep `aux_weight ∈ {0.05, 0.1, 0.2, 0.5}`，window-aux 与 modality-aux 各自扫一遍；
2. 用 GradNorm（Chen et al. ICML 2018）或 Geometric Loss Strategy 让 aux 自动平衡，不调权重。

GradNorm 实现细节（必读，否则会走样）：需指定**参考任务**（reference task）归一化其他任务梯度，本结构以 `head_final`（主任务）为 reference，避免辅助头反客为主压制主任务；并引入单超参 `α` 控制平衡强度，原文用 `α ∈ {0.5, 1, 2}`，需 sweep。

方案 C 中 `0.2 * L_stats_head` 同样适用上述两条规则。

判读：

- 如果 `head_exposure` 或 `head_recovery` 对 N2 有贡献，而 `head_final` 没有，说明融合阶段丢信息；
- 如果所有辅助头都没有 N2 信号，说明前端提取失败或统计资产/窗口定义不对。

### 方案 E：平衡融合

目标：避免高维波形分支淹没慢通道和统计分支。

建议流程：

```text
每个 window × modality:
  encoder output
  -> Linear projection to same dim
  -> LayerNorm/RMSNorm
  -> modality/window token

tokens
  -> gated fusion 或 attention fusion
  -> final representation
```

关键约束：

- 每个模态先等维投影；
- 融合前做归一化；
- 记录 gate/attention 权重；
- 禁止直接用原始高维大小决定融合话语权。

实现优先级：

- **E1 = gated fusion（每 token 一个 sigmoid gate）**：实现简单、可解释，作为方案 E 的首选实现；
- **E2 = attention fusion（token 间多头注意力）**：与方案 J2 phase-token PatchTST 几乎重叠，**不在方案 E 独立做**，统一并入 J2 验证。

即方案 E 主线 = balanced projection + LayerNorm + gated fusion。attention 化的 token 交互留给 J2。

可加训练策略：

- `Modality Dropout`；
- `Window Dropout`；
- `Branch Dropout`；
- OGM-GE（Peng et al. CVPR 2022，arXiv 2203.15332）：监控各模态对学习目标的贡献差异，动态调制各模态梯度幅度，并附加动态高斯噪声避免泛化下降。属梯度级平衡，与上面的 Dropout（结构级）互补，可叠加。比泛泛的“梯度范数监控”更可操作。**执行建议：首轮先只上结构级 Dropout，确认不足后再叠加 OGM-GE；两者同时开启会使模态平衡效果无法归因。**

### 方案 F：ROCKET/MultiRocket 式统计池化分支

目标：补足 TCN 对多尺度局部形态和统计 pooling 的不稳定提取。

结构：

```text
raw window waveform + first-order difference waveform
  -> fixed/random 1D kernels（对原始与差分序列都卷积）
  -> pooling: max / mean / std / PPV / slope
  -> rocket_features
  -> projection
```

“对原始与一阶差分序列都做卷积”是 MultiRocket 相对 MiniRocket 的核心增益之一（Tan et al. 2022，arXiv 2102.00457），不要省略。

然后与：

- explicit phase stats；
- learned TCN embedding；
- slow embedding；

一起融合。

这条路线的意义是：不完全依赖端到端网络自己发现统计量，而是把“多尺度卷积响应 + 多种 pooling”作为结构先验。

实现约束（补）：MultiRocket 特征维度可达上万，与 phase_stats / TCN embedding 融合前必须先降维（PCA 或轻量线性投影到与其它分支同量级），否则 rocket 分支会靠维度优势主导融合，重蹈 §3.2 的高维压制问题；random kernels 必须固定 seed（项目复现规则，见 `rules/experiment-reproducibility.md`）。

### 方案 G：Ridge teacher 蒸馏

目标：用已验证的 `ridge_multiwindow_all_modalities` 指导 DL。

理论先驱：Du et al. ICML 2023（arXiv 2305.01233）提出 UMT（Uni-Modal Teacher）——用单模态 teacher 指导多模态训练，与本项目“用 Ridge（显式统计）teacher 指导端到端 DL”同构。方案 G 可视为 UMT 思想在“统计模型 teacher → DL student”场景的延伸。

可选蒸馏目标：

1. 蒸馏最终预测：

```text
L_distill = MSE(y_final, y_ridge)
```

2. 蒸馏统计分支预测：

```text
L_stats_distill = MSE(y_stats, y_ridge)
```

3. 蒸馏残差：

```text
DL 学 target - ridge_pred
```

推荐从前两种开始。第三种只有在 DL 已能稳定不劣于 Ridge 时再尝试。

**蒸馏权重退火**（必须，避免学出"DL 形状的 Ridge"，损失 extrapolation 表现）：

```text
distill_weight(epoch) = base * max(0, 1 - epoch / T_anneal)
# 或：仅在前 K epoch 启用蒸馏，K = 0.3 * total_epochs
```

理由：DL 若长期被 Ridge 拉向其 bias，extrapolation split 会复制 Ridge 的弱点；前期借 Ridge 引导收敛，后期切回纯 ground-truth 才能让 DL 形成自己的优势。

总损失：

```text
L = L_final
  + 0.2 * L_stats_head
  + 0.2 * L_window_aux
  + 0.1 * L_modality_aux
  + 0.1 * L_distill
```

### 方案 H：输出头与 loss 解耦

目标：避免把 N2 做成前三项误差的残差垃圾桶。

诊断阶段优先：

- `raw4 + weighted_component_mse`
- `softmax100 + weighted_component_mse`

谨慎使用：

- `gas_head + free_component_mse`

原因：

- `raw4` 让四个组分都有独立输出参数，但不保证 sum=100；
- `softmax100` 让四个组分都有 logit，同时软保证 sum=100；但 softmax 的 partition function 让 logit 之间互相牵制，N2 仍非完全独立；
- `gas_head` 硬闭包，但 N2 没有独立参数，容易把 N2 问题和前三项误差耦合在一起。

要真正让 N2 解耦到底，**真正解耦的最小形态是 `raw4 + weighted_component_mse + sum_abs_error monitor`**：4 个独立线性输出，sum 约束通过 loss 加权而不是输出层强制。`softmax100` 是其次。`gas_head` 仅作为生产候选，不作为诊断主线。

判据：

- 若 `raw4` N2 上升但 sum_abs_error 大，说明独立 N2 参数有效，但需要闭包约束；
- 若 `softmax100` N2 上升且 sum_abs_error≈0，则优先进入生产候选；
- 若两者 N2 都不上升，回到特征提取/融合问题，而不是继续调 head。

### 方案 I：TCN hidden linear probe

**优先级说明**：本方案应**提前到 P0/P1 之后立即执行**，作为后续 P4 分叉决策的依据，不要放到 P6。理由：probe 成本最低（只训一层线性），结果直接决定后续要走”修 head/fusion”还是”上 ROCKET/换 backbone”，先做大量改造再诊断容易走弯路。

目标：先判断 TCN 是”没有提取相位统计”，还是”提取了但后续融合/输出头丢掉了”。

做法：

```text
训练现有 TCN/CNN1D-TCN
  -> 冻结模型
  -> 导出 TCN hidden 或 pooled feature
  -> 训练线性 probe / ridge probe
```

probe 任务：

```text
1. hidden -> phase_stats
2. hidden -> ridge_multiwindow prediction
3. hidden -> y_true
4. per-modality hidden -> y_true / phase_stats（每个模态单独 probe）
```

任务 4 是 per-modality probe（Du et al. ICML 2023 的诊断视角，呼应 iTransformer 的 variate-centric 发现）：测每个模态单独的可预测性，能区分“某模态本身无信号”与“模态有信号但被融合压制”，定位欠优化模态。

判读：

- 若 `hidden -> phase_stats` 都很差，说明 TCN 前端没有稳定提取相位统计；
- 若 `hidden -> phase_stats` 好，但 `hidden -> y_true` 或 final head 差，说明信息在融合/输出阶段丢失；
- 若 `hidden -> ridge prediction` 好，但 final prediction 差，优先修 head、loss 和 fusion；
- 若 probe 和 final 都差，再考虑新时间序列 backbone。

这一步成本低，能避免直接换模型导致根因继续不清。

### 方案 J：新时间序列建模算法候选

目标：在不丢掉现有 TCN 前端经验的前提下，引入更适合“相位统计稳定提取与保留”的时间序列结构。

推荐优先级：

```text
J1: ROCKET/MultiRocket/Hydra 式卷积核统计池化分支
J2: Phase-token / PatchTST 式 patch Transformer
J3: TFT/iTransformer 风格 gated window/modal selection
J4: S4/Mamba 状态空间时间编码器
J5: TimesNet/TSMixer/DLinear 轻量对照
```

#### J1：ROCKET/MultiRocket/Hydra 统计池化分支

这是最贴合本问题的候选。旧 `hg` 的强证据是“显式统计 + Ridge”有效，而 ROCKET/MultiRocket/Hydra 的核心正是：

```text
1D convolution kernels
  -> 多种 pooling/statistics
  -> 线性或轻量 head
```

建议实现为旁路分支，而不是直接替换 TCN：

```text
raw waveform / acoustic embedding
  -> fixed/random conv kernels
  -> max / mean / std / PPV / slope / last / delta
  -> rocket_features

[rocket_features, phase_stats, TCN learned features]
  -> balanced fusion
  -> raw4 或 softmax100 head
```

优势：

- 与 Ridge 成功机制一致；
- 可解释性强，能看哪些 kernel/pooling 有贡献；
- 可作为 `phase_stats` 和 TCN learned representation 之间的中间桥梁；
- 对小数据和弱信号通常比纯端到端深层模型更稳。

#### J2：Phase-token / PatchTST 式 patch Transformer

PatchTST 的核心是把时间序列切成 patch token。旧 `hg` 不应只按固定长度 patch，而应把工艺相位显式做成 token：

```text
token_full
token_exposure
token_recovery
token_slow
token_ultrasonic
token_fiber
token_phase_stats
```

token 拆分依据：iTransformer（Liu et al. 2023，arXiv 2310.06625）实证发现“把多变量塞进一个 temporal token 会 fail in learning variate-centric representations，产生 meaningless attention maps”。因此 phase×modality 分开做 token 不是随意拆分，而是有实证支撑——避免变量混合 token 学不到变量中心化表征。

推荐结构：

```text
每个 phase × modality
  -> local encoder / pooling
  -> token projection + LayerNorm

tokens
  -> TransformerEncoder
  -> cls token 或 gated pooling
  -> final head
```

设计变体对照（两者构成完整设计空间，建议都验证）：

- **channel-mixing**（iTransformer 风格，即上图）：phase×modality token 间做 attention 交互，学习变量间关系；
- **channel-independent**（PatchTST 风格）：每个模态独立做 phase-token Transformer（权重共享），末端再融合。PatchTST 摘要明确 channel-independence 是其性能关键之一。

结合 iTransformer 的发现，channel-independent 变体值得作为对照——验证“变量交互是否必要”还是“独立编码已足够”。

适合验证：

- exposure 和 recovery 是否互补；
- slow 与 acoustic 是否冲突；
- phase_stats token 是否被 attention 使用；
- full-window 是否压制相位窗口。

#### J3：TFT/iTransformer 风格 gated selection

注意：J3 的”variable selection / gate”思想与方案 E 的 gated fusion 高度重叠。**不再作为独立主线**，统一并入方案 E（E1 gated fusion 已经覆盖单层 gate；如要 token 间交互再升级，直接走 J2 phase-token PatchTST）。本节仅作为思想出处保留。

Temporal Fusion Transformer 的主要价值不是”预测未来”，而是它有 variable selection 和 gating 思想。旧 `hg` 可以借鉴成：

```text
window/modal token
  -> variable selection / gate
  -> weighted fusion
  -> regression head
```

iTransformer 的启发是把变量或通道当作 token 来建模变量间关系。对旧 `hg` 可改成：

```text
slow token
ultrasonic token
fiber token
phase_stats token
```

优势：

- 直接针对“弱模态被强模态压制”；
- gate/attention 权重可用于解释；
- 比完整大 Transformer 更贴近本项目诊断需求。

#### J4：S4/Mamba 状态空间时间编码器

S4 和 Mamba 适合长序列建模，复杂度比全注意力更友好。可作为 TCN 的替代 temporal encoder：

```text
per-timestep fused embedding
  -> S4/Mamba blocks
  -> phase-aware pooling
  -> final head
```

适用条件：

- TCN hidden probe 证明 TCN 前端确实提取不到相位统计；
- ROCKET/phase-token 分支仍无法提升；
- 需要更强长程依赖建模。

风险：

- 实现和调参成本高；
- 可解释性弱于显式统计和 ROCKET 分支；
- 如果问题根因是融合压制，换 SSM backbone 也可能无效。

#### J5：TimesNet/TSMixer/DLinear 轻量对照

这些模型适合作为对照，不建议优先做主线：

- `DLinear`：证明简单分解/线性模型在很多长序列预测中很强，但旧 `hg` 已经有 Ridge 强基线；
- `TSMixer`：可作为轻量 DL mixing baseline，验证“简单 MLP 混合是否足够”；
- `TimesNet`：适合多周期/2D temporal variation，但旧 `hg` 的相位是工艺阶段，不一定是自然周期。

推荐用途：

```text
如果 ROCKET/phase-token/TFT-style gate 都失败，
再用这些模型做低成本结构对照，
不要把它们作为第一主线。
```

## 5. 推荐实验矩阵

### 第一批：建立事实

| 编号    | 实验                                                   | 目的                    | 通过判据                                                                       |
| ----- | ---------------------------------------------------- | --------------------- | -------------------------------------------------------------------------- |
| P-1   | **v6 证据迁移验证**（新增，最高前置） | 旧链路 Ridge 信号能否在 v6 幸存 | v6 数据上 Ridge multiwindow 的 N2 R² 显著高于同数据 full-only baseline；否则前提失效、停止 |
| A0    | 用 v6 参数重生成 `wv4-formal-hitran-standard-6000`（无备份可恢复） | 该目录本地缺失，是 A1–A2 的前置条件 | benchmark 目录可读，sequence_index/manifest/labels 完整                           |
| A1    | 生成 `phase_stats.npy` + scaler + names                | 固定统计资产                | Ridge 可复现 multiwindow 指标                                                   |
| A2    | `ridge_multiwindow_all_modalities` 复跑                | 校验资产一致性               | v6 基线可复现（N2 R² 以 P-1 标定值为准，不硬编码 0.712）                          |
| B1    | `stats_only_mlp_raw4`                                | 验证 DL 能否吃统计特征         | test overall R² ≥ 0.85, N2 R² ≥ 0.5                                        |
| B2    | `stats_only_mlp_softmax100`                          | 验证闭包 softmax          | test overall R² ≥ 0.88, N2 R² ≥ 0.55, sum_abs_error≈0                      |
| **P** | **`tcn_hidden_probe`（提前）**                           | **决定后续 P4 分叉**        | probe → phase_stats / probe → ridge_prediction / probe → y_true 三个 R² 同时输出 |

### 第二批：统计保留（依赖 probe 结果分叉）

probe 分叉：

- **probe 显示 TCN 已提取 phase_stats**：走 C 系列，修 head/fusion；
- **probe 显示 TCN 未提取 phase_stats**：跳到第四批 E1（ROCKET 分支）作为主干，C 系列降为辅助。

| 编号  | 实验                            | 改动                        | 目的                |
| --- | ----------------------------- | ------------------------- | ----------------- |
| C1  | `learned_only_raw4`           | 不用 phase stats            | learned branch 下限 |
| C2  | `stats_branch_aux_raw4`       | stats head + aux loss     | 检查统计分支是否稳定可用      |
| C3  | `learned_plus_stats_raw4`     | 拼接融合                      | 检查融合是否压制 stats    |
| C4  | `learned_plus_stats_aux_raw4` | stats/window/modality aux | 验证辅助监督是否保留信息      |

### 第三批：融合平衡

| 编号  | 实验                       | 改动                  | 目的                |
| --- | ------------------------ | ------------------- | ----------------- |
| D1  | `balanced_projection_ln` | 等维投影 + LayerNorm    | 消除维度不平衡           |
| D2  | `modality_dropout`       | 丢整个模态               | 防止强模态垄断           |
| D3  | `window_dropout`         | 丢相位窗口               | 防止 full-window 垄断 |
| D4  | `gated_fusion`           | 可解释 gate（合并旧 J3 思想） | 检查实际使用哪些窗口/模态     |

### 第四批：结构增强

| 编号  | 实验                        | 改动                                                         | 目的                         |
| --- | ------------------------- | ---------------------------------------------------------- | -------------------------- |
| E1  | `rocket_pooling_branch`   | 固定/随机卷积核 + 多 pooling                                       | 稳定提取多尺度统计                  |
| E2  | `ridge_distill`（带蒸馏权重退火）  | 蒸馏 Ridge prediction                                        | 让 DL 借 Ridge 引导收敛，后期切回 GT  |
| E3  | `rocket_plus_distill`     | E1 + E2                                                    | 检查是否能超过 stats-only         |
| E4  | `phase_token_patchtst`    | full/exposure/recovery 作为 token；含原 J3 attention 化 token 交互 | 验证 patch/phase token 交互    |
| E5  | `s4_or_mamba_encoder`     | 替换 TCN temporal encoder                                    | 探索长序列状态空间建模（仅在 E1–E4 仍无效时） |
| E6  | `tsmixer_dlinear_control` | 轻量 mixing/linear 对照                                        | 判断是否需要复杂 backbone          |

旧 `E0 tcn_hidden_probe` 已提前到第一批 `P`。旧 `E5 gated_window_modal_selection` 合并入 D4 与 E4。

## 6. 验收标准

### 6.1 统计资产验收

> **注意**：下列数字是**旧链路基线**，仅作历史参照。v6-phys-strict 下必须由 P-1 重新标定；验收判据改为“v6 Ridge multiwindow 的 N2 R² 显著高于同数据 full-only baseline（旧链路对比为 0.712 vs 0.217）”，不再硬编码 0.712。

旧链路基线（历史参照，40kHz / L_m 0.2-1.8m）：

```text
ridge_multiwindow_all_modalities
  test overall R2 ≈ 0.925
  test N2 R2 ≈ 0.712
  extrapolation N2 R2 ≈ 0.725
```

v6 未通过 P-1 前，不进入 DL 改造。

### 6.2 DL 阶段性验收

最低目标：

```text
test overall R2 >= 0.80
test N2 R2 >= 0.50
H2/CH4/CO2 相对当前最佳 DL 不明显退化
```

强目标：

```text
test overall R2 接近 0.90
test N2 R2 接近 0.70
extrapolation N2 R2 不低于 test 太多
```

对 `raw4`：

```text
记录 sum_abs_error，不把它作为唯一否决项。
```

对 `softmax100` / `gas_head`：

```text
sum_abs_error 应接近 0。
```

### 6.3 解释性验收

需要输出：

- 每个 window auxiliary head 的 per-component R2；
- 每个 modality auxiliary head 的 per-component R2；
- stats head 的 per-component R2；
- gated fusion 或 attention 权重；
- Modality/Window Dropout 前后的性能差异。

如果最终模型变好但无法说明它是否使用了 phase stats，仍不能算完成“稳定提取和保留”。

## 7. 不推荐路线

### 7.1 只加普通 Dropout

不推荐作为主方案。它无法保证弱模态被学习，也无法解决输出头和融合压制问题。

### 7.2 只加大 TCN 容量

不推荐优先做。容量更大可能只是更快拟合强模态，不能保证相位统计保留。

### 7.3 继续依赖 `gas_head` 诊断 N2

不推荐作为诊断主线。`gas_head` 中 N2 是残差，没有独立输出参数，容易混淆特征问题和参数化问题。

### 7.4 只看 per-bin N2 R2

不推荐作为主判据。N2 分箱后局部方差很小，R2 分母塌缩，容易出现全局可学但分箱 R2 为负的现象。

## 8. 文献依据

多模态不平衡与弱模态压制：

1. Wang, W.-Y., Tran, D., Feiszli, M. What Makes Training Multi-Modal Classification Networks Hard? CVPR 2020. DOI: `10.1109/CVPR42600.2020.01271`
2. Peng, X. et al. Balanced Multimodal Learning via On-the-fly Gradient Modulation. CVPR 2022. DOI: `10.1109/CVPR52688.2022.00806`
3. Neverova, N. et al. ModDrop: Adaptive Multi-Modal Gesture Recognition. IEEE TPAMI 2015. DOI: `10.1109/TPAMI.2015.2461544`
4. Du, C. et al. On Uni-Modal Feature Learning in Supervised Multi-Modal Learning. ICML 2023. arXiv: `2305.01233`（定量解释"多模态联合训练比单模态更弱"的现象，与本项目 N2 上 DL < Ridge 拼接窗口的观察对应）

辅助监督与多任务权重：

5. Lee, C.-Y., Xie, S., Gallagher, P., Zhang, Z., Tu, Z. Deeply-Supervised Nets. AISTATS 2015.（辅助监督头的标准主引用，比 UNet++ 更贴合"窗口/模态辅助头"语义）
6. Zhou, Z. et al. UNet++: A Nested U-Net Architecture for Medical Image Segmentation. MICCAI Workshops, 2018. DOI: `10.1007/978-3-030-00889-5_1`
7. Chen, Z. et al. GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. ICML 2018. arXiv: `1711.02257`（方案 C/D 中辅助 loss 自动平衡的替代路径）

时间序列统计、卷积核变换与 extrinsic regression：

8. Fawaz, H. I. et al. InceptionTime: Finding AlexNet for time series classification. Data Mining and Knowledge Discovery, 2020. DOI: `10.1007/s10618-020-00710-y`
9. Tan, C. W. et al. MultiRocket: multiple pooling operators and transformations for fast and effective time series classification. Data Mining and Knowledge Discovery, 2022. DOI: `10.1007/s10618-022-00844-1`
10. Dempster, A. et al. Hydra: competing convolutional kernels for fast and accurate time series classification. Data Mining and Knowledge Discovery, 2023. DOI: `10.1007/s10618-023-00939-3`
11. Tan, C. W. et al. Time series extrinsic regression: Predicting numeric values from time series data. Data Mining and Knowledge Discovery, 2021. DOI: `10.1007/s10618-021-00745-9`
12. Foumani, N. M. et al. Deep Learning for Time Series Classification and Extrinsic Regression: A Current Survey. ACM Computing Surveys, 2024. DOI: `10.1145/3649448`

Patch、Transformer、状态空间与轻量时间序列模型：

13. Nie, Y. et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. arXiv, 2022. DOI: `10.48550/arxiv.2211.14730`
14. Lim, B. et al. Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 2021. DOI: `10.1016/j.ijforecast.2021.03.012`
15. Liu, Y. et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. arXiv, 2023. DOI: `10.48550/arxiv.2310.06625`
16. Gu, A. et al. Efficiently Modeling Long Sequences with Structured State Spaces. arXiv, 2021. DOI: `10.48550/arxiv.2111.00396`
17. Gu, A., Dao, T. Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv, 2023. DOI: `10.48550/arxiv.2312.00752`
18. Wu, H. et al. TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. arXiv, 2022. DOI: `10.48550/arxiv.2210.02186`
19. Zeng, A. et al. Are Transformers Effective for Time Series Forecasting? AAAI, 2023. DOI: `10.1609/aaai.v37i9.26317`
20. Chen, S. et al. TSMixer: An All-MLP Architecture for Time Series Forecasting. arXiv, 2023. DOI: `10.48550/arxiv.2303.06053`

## 9. 推荐落地顺序

```text
P-1: v6 证据迁移验证（新增，最高优先）—— 用 v6 参数重生成 wv4-formal，
     重建 Ridge multiwindow 基线，确认 N2 相位统计信号是否在新链路幸存
P0: 用 v6 参数重生成 wv4-formal-hitran-standard-6000 benchmark（无备份，只能重生成）
P1: 生成并固定 phase_stats artifact（含 mixture_id split_key 与 composition_scheme；
    相位窗口按 v6 新 L_m 重新标定）
P2: stats-only MLP + ridge_multiwindow 复现
P3: TCN hidden probe（提前到此处）—— 决定 P4 分叉
P4: 按 probe 结果分叉
    P4a 若 TCN 已提取 phase_stats：stats / window / modality auxiliary heads（含 aux_weight sweep 或 GradNorm）
    P4b 若 TCN 未提取 phase_stats：ROCKET/MultiRocket/Hydra pooling branch 作为主干 + aux heads 辅助
P5: balanced fusion + modality/window dropout + gated fusion（合并旧 J3 思想）
P6: Ridge teacher distillation（带蒸馏权重退火）
P7: phase-token PatchTST（含 attention 化 token 交互，取代旧 J3 独立线）
P8: S4/Mamba encoder 作为后置探索线（仅在 P7 仍无效时）
P9: TSMixer/DLinear 轻量对照（最末，仅用于结构对照）
```

最关键的停止条件：

```text
如果 P-1 显示 v6 下 Ridge multiwindow 的 N2 信号已大幅衰减（不再显著优于
full-only baseline），说明方案前提（“显式相位统计里有可学信号”）在新链路失效，
先改问题定义（N2 是否仍作独立目标 / 是否转向 TCS 热导加权 / 是否接受 N2 走闭包），
不要继续 P0 之后的任何 DL 改造。

如果 P0 benchmark 无法重生成，
不要继续；先确认数据来源。

如果 phase_stats artifact 无法复现 ridge_multiwindow，
不要继续调 DL。

如果 stats-only MLP 接近 ridge，但 learned+stats 变差，
优先修融合和辅助监督（P4 → P5）。

如果 P3 probe 显示 TCN 已经保留 phase_stats，
不要急着换 backbone，优先修 fusion/head/loss（P4a → P5 → P6）。

如果 P3 probe 显示 TCN 没有提取 phase_stats，
优先 ROCKET 分支（P4b），再考虑 phase-token PatchTST（P7）。

如果 P4–P6 仍无法提升 N2，
再推进 P7；P8/P9 仅作为后置探索，不要提前替换 backbone。
```
