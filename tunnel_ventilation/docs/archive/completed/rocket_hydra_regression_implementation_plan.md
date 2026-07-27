# MiniRocket / MultiRocket / Hydra 回归基线实施规划

> 目标：针对 tv3 掘进通风数据集，在当前端到端 DL 训练失效的前提下，优先实现固定卷积核时序特征 + Ridge / ElasticNet / 小 MLP 的稳健回归链路，用于判断超声波形是否真实携带 O2 / N2 可辨识信号。
> 
> **2026-07-08 定位修正**：R0 实测 val O₂ R²=0.603 已越过本规划原设的 R1 判断点（R0 O₂ R²>0.30 即物理特征路线成立）。R1 的科学问题从"raw 波形有无 O₂ 信号"变更为"raw 波形卷积特征能否在 R0 已用满物理标量序列的基础上再贡献至 0.70 验收线"。本规划已据 [dl_training_plan.md §11.4](../legacy/dl_training_plan.md#114-服务器验证结果tv3-formal-6000-50-epoch-单-seed-rtx-5880)（v3_l2 DL fusion 实测 R²=+0.019）与 [波形特征提取算法评估.md](../../methods/波形特征提取算法评估.md)（10 算法排序）重写 §2/§3/§4.2/§8/§9.2/§10。
>
> **2026-07-09 后续回填**：R5 已按 [r5_mlp_implementation_plan.md](../../archive/completed/r5_mlp_implementation_plan.md) 在 **D0-observed 864**（非本文件旧写的 R0 1080）上完成正式 6000，**判据未通过**（val O₂ −0.183）。可部署上限维持 D0-observed Ridge；当前 P0 转为 O₂ 光学通道。详见记忆库 §6.9。

## 0. 实施进度（截至 2026-07-07，状态更新 2026-07-08）

阶段 A 已完成最小可运行落地，R0 正式集结果已回填（6000 序列，`data/tv3-formal-6000`，产物 `outputs/tv3_rocket/r0/metrics.json`）。

### 2026-07-08 状态更新

R0 之后的关键变化已由同日完成的两项工作验证，本规划据此修正 R1 定位：

| 变化                                                                   | 来源                                                         | 对本规划的影响                                                                    |
| -------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------- |
| DL fusion 三层归一化后仍失效（v3_l2 val R²=+0.019，远逊于 R0 的 0.918）              | [dl_training_plan.md §11.4.1](../legacy/dl_training_plan.md#1141-总览) | §2.2 的"固定特征路线"预判从推论升级为已验证结论；R1 不再是"规避 DL 崩溃的保险"，而是主推方向                     |
| P-9c 触发（fusion O₂ R²=-0.061 < 0.50）                                  | [dl_training_plan.md §10](../legacy/dl_training_plan.md#10-推荐执行顺序)   | 端到端 DL raw 波形路线当前架构上限低于物理特征基线，本规划成为 O₂ 推至 0.70 的主路线                        |
| 10 算法排序：Top1 TOF 工程(9.5) / Top2 MultiRocket(9.0) / Top5 wav2vec(8.0) | [波形特征提取算法评估.md](../../methods/波形特征提取算法评估.md)                             | R0 实质已是 Top1 的离线实现（用满 `ultrasonic_tof_s` 等标量序列）；R1 raw 波形卷积特征需对照 R0 而非从零起步 |
| fiber_mic 定位                                                         | [experiment_roadmap.md 方向 D](../legacy/experiment_roadmap.md)        | fiber_mic 是已跳过的现有模态而非新传感器，§3 边界已对齐                                         |

R0 落在原 §9.2 决策阈值的"边际可用"区间（0.50–0.70），距 0.70 验收线剩 0.097 缺口。R1 必须回答的新问题：**raw 波形卷积特征能否填补这 0.097，而不是"raw 波形有无信号"**。

已完成：

- `tv3/ml/rocket_features.py`
  - 实现 `physics_stats_v1` 特征缓存
  - 覆盖 full / phase / early 三类窗口统计
  - 支持 `slow + ultrasonic_tof_s + ultrasonic_tof_observed_s + ultrasonic_peak_index + ultrasonic_sound_speed_m_per_s + ultrasonic_sound_speed_estimated_m_per_s + ultrasonic_alpha_true_npm + ultrasonic_tof_quality + ultrasonic_tof_accepted`
  - 写出 `feature_matrix_{split}.npy`、`feature_names.json`、`feature_manifest.json`
- `tv3/ml/rocket_training.py`
  - 实现 `StandardScaler + RidgeCV`
  - 保留 `ridge_closed_form` 对照路径
  - 输出 `train / val / test / extrapolation` 指标与 top feature group 诊断
- `tv3/pipeline/run_tv3_rocket_baseline.py`
  - 支持 `--feature-set physics_stats --head ridgecv`
- `configs/tv3_rocket_ridge.json`
  - 提供 R0 默认配置
- 测试
  - `tests/test_rocket_features.py`
  - `tests/test_tv3_rocket_pipeline.py`
  - smoke 已通过，当前验收命令：

```powershell
python -m pytest tests/test_rocket_features.py tests/test_tv3_rocket_pipeline.py -v
```

### R0 正式集实测结果（2026-07-07）

数据集 `tv3-formal-6000`（6000 序列，int16 + skip-fiber-mic），特征数 1080，`RidgeCV` 选定 `alpha=0.0001`。

per-component R²：

| 组分    | train | val   | test  | extrapolation |
| ----- |:-----:|:-----:|:-----:|:-------------:|
| x_CO2 | 0.994 | 0.993 | 0.993 | 0.992         |
| x_O2  | 0.661 | 0.603 | 0.639 | 0.557         |
| x_N2  | 0.939 | 0.925 | 0.934 | 0.923         |

整体 R²：val 0.901 / test 0.911 / extrapolation 0.898；`sum_abs_error` 各 split 均 ~6e-7（标签闭包天然满足）。

诊断要点：

- 三个组分的 top-5 feature group 完全一致，均由超声物理量主导：`ultrasonic_alpha_true_npm` > `ultrasonic_sound_speed_m_per_s` > `ultrasonic_tof_s` > `ultrasonic_tof_quality` > `ultrasonic_tof_observed_s`；slow 通道未进入 top-5，O₂ 辨识几乎全部来自超声物理特征。
- O₂ 整体 val R²=0.603，但 `o2_bins` 四个窄分箱内 R² 全部为负（-9.2 ~ -2.6）。含义：模型能区分 O₂ 高/低大档，但在窄浓度区间内无法做精细分辨（区间方差小、残差稍大即 R² 大幅变负）。

对照 §9.2 决策阈值（已按 R0 结果重写，见 §9.2）：

- R0 val O₂ R²=0.603，落在"边际可用（0.50–0.70）"区间，物理特征路线成立但距 0.70 验收线剩 0.097。
- "raw 波形有无 O₂ 信号"已由 R0 的 `ultrasonic_tof_s`/`sound_speed`/`alpha_true` 间接回答（top-5 全是超声物理标量）；R1 要回答的是 raw 5000 点卷积特征能否提供这些标量之外的增量。

### R1a/R1b formal-6000 实测结果（2026-07-08）

R1a（minirocket_scalar，2111 维）/ R1b（minirocket_raw，2367 维）均在 `tv3-formal-6000` 上完成，产物 `outputs/tv3_rocket/r1a|/r1b/metrics.json`。

| run | 特征数  | selected_alpha | val R² | val O₂ R² | val O₂ MAE(%) | O₂ top 特征 abs_coef_sum       |
| --- |:----:|:--------------:|:------:|:---------:|:-------------:|:----------------------------:|
| R0  | 1080 | 1e-4           | 0.901  | **0.603** | 0.46          | 2652（alpha_true）             |
| R1a | 2111 | 1e-2           | 0.879  | 0.515     | 0.52          | 9.4（alpha_true:kernel*）      |
| R1b | 2367 | 1e2            | 0.603  | -0.195    | 0.85          | 0.12（minirocket_raw:kernel*） |

判定（按 §9.2）：

- **R1b − R1a = −0.71，远低于 +0.05 阈值** -> raw 5000 点波形卷积特征**无增量**，raw 波形卷积路线（R1b/R3/Hydra）证伪，不继续投入。
- R1a val O₂ R²=0.515 < R0 的 0.603 -> MiniRocket 作用于标量序列也不如 R0 直接多窗口统计。R1a selected_alpha=1e-2（R0 的 100×）说明特征冗余；top 核贡献高度均匀（9.32–9.43），无核脱颖而出，卷积统计与原序列统计冗余。
- R1b selected_alpha=1e2（R0 的 10⁶×），coef_norms O₂ 仅 1.078（R0 的 680.772）-> raw 卷积特征在 Ridge 线性模型下表达力不足，被正则化压平。
- O₂ 窄分箱三 run 全负且单调恶化（R0 −9.2~-2.6 / R1a −10.2~-3.8 / R1b −32.9~-5.8），R1b 低/高端 bin R² −30 说明 raw 卷积引入系统性偏差。

结论：R0 仍是当前最优（0.603），raw 波形卷积路线证伪。下一步转向 R5（小 MLP on R0 特征，查非线性增益）、方向 D（fiber_mic 增量，但需先评估是否与 R1b 同样失效）、或接受 R0 极限（边际可用区间，优化特征与 split）。

未完成：

- ~~MiniRocket / MultiRocket 波形特征~~（R1a/R1b 已完成，raw 路线证伪，见上）
- ElasticNetCV / 小 MLP（R2/R4/R5，R5 优先：验证 R0 是否已到线性极限）
- ~~Hydra~~（R1b 无增量，§10 阶段 E 不再推进）
- fiber_mic 增量评估（方向 D，需重新生成数据集去 `--skip-fiber-mic`）

## 1. 结论先行

推荐第一版落地顺序：

```text
tv3-formal-6000
  -> ultrasonic waveform / tof / sound_speed / slow channels
  -> MiniRocket-style fixed kernels
  -> sequence-level pooling
  -> StandardScaler
  -> RidgeCV / ElasticNetCV / small MLP
  -> val / test / extrapolation per-component metrics
```

优先级：

| 优先级 | 方法                                          | 用途                | 是否第一版必做 |
| ---:| ------------------------------------------- | ----------------- |:-------:|
| P0  | TOF / sound_speed 序列特征 + Ridge / ElasticNet | 最小可验证声学信号         | 是       |
| P0  | MiniRocket-style 超声帧特征 + RidgeCV            | 稳健主 baseline      | 是       |
| P1  | MultiRocket-style 多池化统计                     | 增强 O2 / N2 弱信号    | 是       |
| P1  | 小 MLP 回归头                                   | 检查轻量非线性增益         | 是       |
| P2  | Hydra-style competing kernels               | 若 MiniRocket 有效再上 | 否，第二轮   |

关键判断：服务器上 6000 序列仍出现 DL 最佳 epoch 为 1，说明不能继续把问题归因于 600 小样本；当前首要任务是用不依赖深层反传稳定性的固定特征路线，把“数据有无 O2 信号”和“端到端训练是否坏掉”分开。

## 2. 背景与触发条件

### 2.1 tv3 数据与任务

- 数据集：服务器侧 `tv3-formal-6000`，每条 sequence 512 timesteps。
- 目标：直接预测 `x_CO2`、`x_O2`、`x_N2`，不使用闭包残差头。
- 主要输入：
  - `slow`: NDIR CO2、TCS、温度、压力、湿度、声程、阶段位置等慢通道。
  - `ultrasonic`: 每 timestep 一帧 200 kHz 超声波形。
  - 已生成声学辅助数组：`ultrasonic_tof_s`、`ultrasonic_tof_observed_s`、`ultrasonic_sound_speed_estimated_m_per_s`、`ultrasonic_peak_index`、`ultrasonic_alpha_true_npm` 等。
- 难点：O2 无直接光学通道，主要依赖 O2 / N2 声速差和 TCS 弱差异。

### 2.2 失效模式（2026-07-08 据 §11.4 实测重写）

已知事实（R0 + v3_l2 实测后）：

- slow-only Ridge 能预测 CO₂（R²=0.91），但 O₂≈均值预测（R²=-0.05）。
- R0（slow + 超声物理标量序列 + RidgeCV）val O₂ R²=0.603，物理特征路线有效但未达 0.70。
- DL fusion 三层归一化后最优 v3_l2：val R²=+0.019 / O₂ R²=-0.061，远逊于 R0 的 0.918；P-9c 触发（fusion O₂ R²<0.50）。

已验证结论（原为 2026-07-07 的推论，现据 §11.4 实测确认）：

1. 仅扩大样本量未解决 DL 训练问题：6000 序列下 v2 best epoch=1、v3_l2 虽训练正常但 R² 仍≈0。根因是 encoder 架构（`avg+max` 池化对平移不敏感，丢掉亚样本 TOF 相位），不是样本量也不是融合层（v3_l3 的 FiLM/gate 无额外收益佐证）。
2. 固定特征路线已成为主推方向：DL 端到端 raw 波形路线当前架构上限低于物理特征基线（0.019 vs 0.918），固定特征不再是"规避 DL 崩溃的保险"，而是把 O₂ 推向 0.70 的主路线。
3. R0 已用满超声物理标量序列（tof_s / sound_speed / alpha_true 等），R1 的 raw 波形卷积特征需对照 R0 证明增量，而非从零起步重复"raw 波形有无信号"的判断。

**停止条件仍生效**：若 R1/R3（含 fiber_mic 增量）O₂ R² 仍 < 0.50，则判断现有通道物理可辨识性不足，转向 O₂ 专用传感器，不再在 ROCKET 特征上投入。

## 3. 不变量与边界

1. 不把 `x_N2` 作为残差目标，三个组分全部直接回归。
2. 不使用 `compositional_mse`、`ilr_mse`、`free_component_mse` 或 target transform。
3. 不把 `mixture_id` 回退或重写为 `sequence_id`。
4. 不用训练失败时的静默 fallback；任何 NaN、维度不匹配、缓存缺失都应直接失败。
5. 本方案只验证现有四模态（slow / ultrasonic / fiber_mic / NDIR）的信号极限，不引入新传感器。**fiber_mic 是已跳过的现有模态而非新传感器**：当前 `--skip-fiber-mic` 是存储优化，代码保留可恢复。R1 在 slow+ultrasonic 下若不达 0.70，恢复 fiber_mic（重新生成数据集去掉 `--skip-fiber-mic`）作 R3 增量备选，与 [experiment_roadmap.md 方向 D](../legacy/experiment_roadmap.md) 一致。
6. 不一次性加载完整 6000 x 512 x 5000 波形到内存；必须 chunk 流式生成特征缓存。
7. 先实现可复现、可审计、可缓存的特征管线，再接回归头。
8. **R1 必须对照 R0 而非从零起步**：R0 已用满 `ultrasonic_tof_s`/`sound_speed`/`alpha_true` 等物理标量序列，R1 的 raw 波形卷积特征需证明在这些标量之外的增量（至少 O₂ R² 提升 > 0.05 才算有效），否则不应继续堆 R3/Hydra。

## 4. 算法路线

### 4.1 P0-A：物理序列统计基线（已落地为 R0）

输入数组：

- `ultrasonic_tof_s`
- `ultrasonic_tof_observed_s`
- `ultrasonic_sound_speed_estimated_m_per_s`
- `ultrasonic_peak_index`
- `ultrasonic_alpha_true_npm`
- slow channels

每条 sequence 提取：

- full window: mean, std, min, max, range, first, last, delta, slope。
- phase windows: baseline, exposure, steady, recovery。
- early windows: first 25%, 50%, 75%。

模型：

- `RidgeCV`
- `ElasticNetCV`
- 现有 closed-form `RidgeRegressor` 作为最小依赖对照。

**已验证结论（R0 实测）**：无需 raw waveform，TOF / sound_speed / alpha 序列统计已使 val O₂ R²=0.603，物理特征路线成立。原"如果 P0-A 的 O2 R2 仍约 0，raw waveform 路线才有必要"的判断点已被跳过——R0 远超该判断线。

**P0-A 在新定位下的角色**：作为 R1 的对照基线（R1 必须在 R0 之上证明增量），不再是"判断 raw 波形是否必要"的前置门。R0 实质上是 [波形特征提取算法评估.md](../../methods/波形特征提取算法评估.md) Top1（TOF 物理特征工程 + Ridge，得分 9.5）的离线实现——它消费的是仿真侧已计算好的 `ultrasonic_*` 标量序列，不是从 raw 5000 点波形端到端提取。

### 4.2 P0-B：MiniRocket-style 超声帧特征（R1 定位重写）

**R1 的新科学问题**：R0 已用满超声物理标量序列（tof_s / sound_speed / alpha_true，top-5 全是这些标量），val O₂ R²=0.603。R1 要回答的不是"raw 5000 点波形有无 O₂ 信号"（R0 间接已有答案），而是 **raw 波形的固定核卷积特征能否提供这些标量之外的增量，把 O₂ R² 从 0.603 推向 0.70**。

这个新问题要求 R1 先做小规模对照，而不是直接上全量 MiniRocket：

1. **R1a 对照先导**：MiniRocket 作用于 raw 波形 vs 作用于 R0 已用的标量序列，哪个对 O₂ 有增量。若 MiniRocket(raw) 不优于 MiniRocket(tof_s 序列)，说明 raw 5000 点的额外信息有限，R1 应转向别的增量方向（fiber_mic 或 TOF 工程的端到端可微版）。
2. **R1b 全量推进**：仅当 R1a 证明 raw 波形有增量时，才上全量 MiniRocket（num_kernels 512 起步）。

输入形态：

```text
one sequence ultrasonic: (T=512, L=5000)
```

不建议把 `(512, 5000)` 直接 flatten 成长序列。按"帧内波形 + 跨 timestep 池化"处理：

1. 对每个 timestep 的 5000 点超声帧应用固定 1D 卷积核。
2. 每个 kernel 对每帧输出 PPV 与 max 两类统计。
3. 对 512 帧的 kernel 统计再做 sequence pooling：mean, std, min, max, slope。
4. 拼接 slow / TOF / sound_speed 统计（与 R0 特征拼接，不是替换）。

第一版 MiniRocket-style 不是完整复刻论文实现，而是项目内可控的固定核特征器：

- kernel length: `{7, 9, 11}`
- dilation: log-spaced，覆盖短窗和长窗。
- weights: zero-mean fixed random kernels，seed 固定。
- bias: 从训练集少量帧的卷积分位数估计。
- features: PPV 为主，max / mean_abs 作为补充。

这样做的原因：

- 不引入额外大依赖。
- 避免把 6000 序列全部波形一次性展开。
- 直接针对超声 TOF 微小位移与衰减形态。

**与 R0 的关键差异**：R0 的 `ultrasonic_tof_s` 等是仿真生成时用 Lagrange 分数延迟 FIR 计算的显式 TOF（亚样本精度 <0.002μs），MiniRocket 从 raw 波形隐式学到的卷积模式若与这些显式 TOF 信息冗余，则 R1 增量有限。R1a 对照正是为确认这一点。实现代码骨架见 [波形特征提取算法代码示例.md](../../methods/波形特征提取算法代码示例.md) 示例 2。

### 4.3 P1：MultiRocket-style 多池化增强

在 P0-B 基础上增加：

- first-order difference waveform。
- 每个 kernel 输出 PPV、max、mean、std。
- phase-aware pooling：baseline / exposure / steady / recovery 分别池化，再拼 full-window 池化。

收益预期：

- 对 O2 / N2 更敏感，因为 O2 信号可能体现在响应阶段变化，而不是单帧绝对形态。
- 比端到端 TCN 更稳，因为只有回归头参与训练。

风险：

- 特征维度可能过高。第一版控制在 2k 到 20k 维，不超过 Ridge / ElasticNet 可承受范围。

### 4.4 P2：Hydra-style competing kernels

Hydra 作为第二轮实现：

- 多组 kernel 在同一 receptive field 内竞争。
- 每组保留 winning pattern 的计数或强度。
- 适合捕获“哪个 acoustic motif 出现”而不是连续卷积幅值。

触发条件：

- MiniRocket / MultiRocket 已证明 O2 R2 有明显正信号，例如 val O2 R2 > 0.30。
- 但仍低于验收线，需要更强非线性模式特征。

不作为第一版必做的原因：

- 实现更复杂。
- 当前更急的是验证训练链路和物理信号，而不是追求最终 SOTA。

## 5. 回归头设计

### 5.1 RidgeCV

用途：主 baseline。

参数：

```text
alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1, 3, 10, 30, 100]
```

要求：

- 所有特征只用 train split 拟合 `StandardScaler`。
- `RidgeCV` 只在 train 内交叉验证，不看 val / test / extrapolation。
- 输出每个目标的 coef norm 与 top feature group，辅助诊断 O2 是否用到 acoustic 特征。

### 5.2 ElasticNetCV

用途：特征选择与稀疏性诊断。

参数：

```text
l1_ratio = [0.05, 0.1, 0.3, 0.5, 0.7]
alphas = logspace(-4, 1, 20)
```

判断：

- 如果 ElasticNet O2 与 Ridge 接近，说明信号可由少量稳定特征解释。
- 如果 ElasticNet 明显差于 Ridge，说明 O2 信号分散在大量弱特征上，后续更适合 Ridge / MLP。

### 5.3 小 MLP

用途：检查轻量非线性增益，不替代 Ridge 主判断。

结构：

```text
input_dim
  -> Linear(256) + GELU + Dropout(0.10)
  -> Linear(128) + GELU + Dropout(0.10)
  -> Linear(3)
```

训练：

- loss: plain MSE 或 weighted component MSE `[1, 2, 1]`。
- optimizer: AdamW。
- lr: `1e-4` 起步，不使用当前 fusion 的 `1e-3` 默认值。
- epochs: 100。
- early stopping: patience 10。
- batch size: 64 或 128。

失败判据：

- 如果 best epoch 仍为 1，且 Ridge / ElasticNet 正常，则小 MLP 的训练配置仍有问题。
- 如果 Ridge / ElasticNet 也无正信号，则不要继续调 MLP。

## 6. 实现文件规划

### 6.1 新增文件

| 文件                                        | 职责                                                         |
| ----------------------------------------- | ---------------------------------------------------------- |
| `tv3/ml/rocket_features.py`               | 固定 kernel 生成、chunk 波形读取、MiniRocket / MultiRocket 特征提取、缓存写入 |
| `tv3/ml/rocket_training.py`               | RidgeCV / ElasticNetCV / small MLP 训练与评估                   |
| `tv3/pipeline/run_tv3_rocket_baseline.py` | tv3 专用编排脚本，生成缓存并跑实验矩阵                                      |
| `configs/tv3_rocket_ridge.json`           | MiniRocket + RidgeCV 配置                                    |
| `configs/tv3_rocket_elasticnet.json`      | MultiRocket + ElasticNetCV 配置                              |
| `configs/tv3_rocket_mlp.json`             | MultiRocket + 小 MLP 配置                                     |
| `tests/test_rocket_features.py`           | kernel、shape、缓存、可复现性测试                                     |
| `tests/test_tv3_rocket_pipeline.py`       | 小数据 smoke pipeline 测试                                      |

阶段 A 当前实际职责更精确地说是：

- `tv3/ml/rocket_features.py`：`physics_stats_v1` 特征缓存与 split 对齐校验
- `tv3/ml/rocket_training.py`：`RidgeCV` / `ridge_closed_form` 的首轮训练与评估
- `tv3/pipeline/run_tv3_rocket_baseline.py`：tv3 `physics_stats` 实验入口
- `configs/tv3_rocket_ridge.json`：R0 配置
- `tests/test_rocket_features.py`、`tests/test_tv3_rocket_pipeline.py`：阶段 A smoke 测试

### 6.2 可复用现有文件

| 文件                               | 复用点                                  |
| -------------------------------- | ------------------------------------ |
| `tv3/ml/features.py`             | slow / waveform 统计特征、multi-window 逻辑 |
| `tv3/ml/models.py`               | 现有 `RidgeRegressor` 可作为最小依赖对照        |
| `tv3/common/metrics.py`          | per-component R2 / MAE / RMSE        |
| `tv3/common/splits.py`           | split CSV 解析                         |
| `tv3/common/waveform.py`         | int16 / int32 waveform 路径解析          |
| `tv3/dl/models/handcraft_mlp.py` | 如接口合适，可复用小 MLP                       |

## 7. 特征缓存契约

缓存目录：

```text
data/<dataset>/features/rocket/
  minirocket_ultra_v1/
    feature_matrix_train.npy
    feature_matrix_val.npy
    feature_matrix_test.npy
    feature_matrix_extrapolation.npy
    feature_names.json
    feature_manifest.json
```

`feature_manifest.json` 必须记录：

- `dataset_slug`
- `schema_version`
- `sequence_count`
- `split_policy`
- `feature_builder`
- `kernel_seed`
- `kernel_count`
- `kernel_lengths`
- `dilations`
- `pooling_stats`
- `modalities`
- `slow_channels`
- `source_arrays`
- `created_at`

缓存校验：

- split 行数必须等于对应 split CSV 行数。
- feature_names 长度必须等于矩阵列数。
- 重新运行相同 seed 必须得到完全相同 feature matrix。
- 任何 NaN / Inf 直接报错。

## 8. 实验矩阵（2026-07-08 据 R0 结果重写）

第一轮只跑 6000 数据集，不再用 600 数据集做主结论。

| 实验 ID | 特征                                        | 模型           | 目的                                                                    | 状态                                          |
| ----- | ----------------------------------------- | ------------ | --------------------------------------------------------------------- | ------------------------------------------- |
| R0    | slow + TOF / sound_speed stats            | RidgeCV      | 物理标量序列基线                                                              | ✅ val O₂ R²=0.603                           |
| R1a   | MiniRocket(tof_s 序列) + slow               | RidgeCV      | 先导对照：MiniRocket 作用于 R0 标量序列                                           | ✅ val O₂ R²=0.515 < R0                      |
| R1b   | MiniRocket(raw 5000 点) + slow stats       | RidgeCV      | 增量验证：raw 波形卷积对 R0 的增量                                                 | ✅ val O₂ R²=-0.195，−R1a=−0.71 < 0.05        |
| R2    | MiniRocket ultrasonic + slow stats        | ElasticNetCV | 稀疏性诊断                                                                 | ⏳ R1b 证伪后优先级降                               |
| R3    | MultiRocket ultrasonic + slow + TOF stats | RidgeCV      | 多池化增强                                                                 | ❌ R1b 无增量，不推进                               |
| R3f   | R3 + fiber_mic（需重新生成数据集）                  | RidgeCV      | fiber_mic 增量备选，对齐 [experiment_roadmap.md 方向 D](../legacy/experiment_roadmap.md) | ⏳ 待评估（fiber_mic 也是 raw 波形，需先确认是否与 R1b 同样失效） |
| R4    | MultiRocket ultrasonic + slow + TOF stats | ElasticNetCV | 高维特征选择                                                                | ❌ R1b 无增量，不推进                               |
| R5    | MultiRocket ultrasonic + slow + TOF stats | small MLP    | 轻量非线性增益                                                               | ⏳ 改为 R0 特征 + MLP，验证 R0 是否已到线性极限（优先）         |
| R6    | Hydra-style ultrasonic + slow + TOF stats | RidgeCV      | 第二轮候选                                                                 | ❌ R1b 无增量，不推进                               |

推荐执行顺序（据 R1a/R1b 实测更新）：

```text
R0(✅ 0.603) -> R1a(✅ 0.515) -> R1b(✅ -0.195)
  -> R1b - R1a = -0.71 < 0.05: raw 波形卷积路线证伪
  -> 不推进 R3/R4/R6(raw 波形特征)
  -> R5(R0 特征 + 小 MLP)优先:验证线性极限
  -> R3f(fiber_mic)待评估:需先确认 fiber_mic 是否与 R1b 同样失效
  -> 若 R5 无增益且 R3f 失效:接受 R0(0.603)作为现有通道极限
```

**R1a/R1b 判定结论**：R1b(val O₂ R²) − R1a(val O₂ R²) = −0.71，远低于 0.05 阈值，raw 5000 点波形相对 R0 标量序列无显著增量。R1a 本身(0.515)也不如 R0(0.603)，MiniRocket 作用于标量序列不如 R0 直接多窗口统计。raw 波形卷积路线（R3/R4/R6）证伪，不再投入。

## 9. 指标与验收

### 9.1 主指标

每个 split 输出：

- overall MAE / RMSE / R2
- `x_CO2` MAE / RMSE / R2
- `x_O2` MAE / RMSE / R2
- `x_N2` MAE / RMSE / R2
- `sum_abs_error`
- `o2_bins` / `co2_bins` conditional metrics

### 9.2 决策阈值（2026-07-08 据 R0 实测重写）

R0 val O₂ R²=0.603 已落在"边际可用"区间，下表标注已达成与待判断的阈值（R1a/R1b 实测后更新）：

| 结果                            | 决策                                                             | R1a/R1b 后状态                    |
| ----------------------------- | -------------------------------------------------------------- | ------------------------------ |
| R0 O2 R2 > 0.30               | TOF / sound_speed 已有可用 O2 信号，优先做物理特征路线                         | ✅ R0 达成（0.603）                 |
| R1b O2 R² > R0 + 0.05         | raw ultrasonic waveform 相对 R0 标量序列提供额外 O2 信息，继续 ROCKET / Hydra | ❌ 未达成（R1b=-0.195，R0=0.603）     |
| R1b O2 R² − R1a O2 R² < 0.05  | raw 5000 点波形无显著增量，转 fiber_mic 或 TOF 工程端到端可微版                   | ✅ 已触发（−0.71，远低于 0.05）          |
| R3 O2 R2 >= 0.70              | 现有通道达到最低验收，可进入消融与稳定性验证                                         | ❌ R1b 证伪，R3 不推进                |
| R3 O2 R2 0.50 到 0.70          | 现有通道边际可用，优先优化特征与 split，不急于加新传感器                                | ✅ R0(0.603)在此区间，R1a/R1b 均不如 R0 |
| R3 O2 R2 < 0.50               | 当前通道组合不足，转向 O2 专用通道评估                                          | ❌ R0 已远离此区间                    |
| Ridge 正常但 MLP best epoch=1    | MLP 训练配置问题，不影响 ROCKET 特征有效性判断                                  | -                              |
| Ridge / ElasticNet / MLP 全部失败 | 优先查数据集、split、label、scale，不换模型                                  | -                              |

**关键变化**：原阈值"R1/R3 O2 R2 > R0 + 0.10"在 R0=0.603 的新基线上过于宽松--R0 已用满标量序列，R1 的 raw 波形增量应至少 0.05 才算有效（否则与 R0 的 `ultrasonic_tof_s` 冗余）。把 +0.10 降为 +0.05 是对"R0 已越过有无信号判断点"的直接反映。

**R1a/R1b 实测判定**：R1b−R1a=−0.71 触发"raw 波形无增量"分支，raw 波形卷积路线（R3/R4/R6）证伪。R1a(0.515)本身也不如 R0(0.603)，说明 MiniRocket 作用于标量序列不如 R0 直接多窗口统计。下一步：R5（R0 特征 + 小 MLP）验证线性极限，R3f（fiber_mic）待评估。

### 9.3 最低验收

第一轮实现验收：

- `tests/test_rocket_features.py` 通过。
- 32 序列 smoke 数据可在 60 秒内完成特征生成与 Ridge 训练。
- 6000 序列特征生成支持逐序列 mmap 流式（R1b 已落地，不 OOM，formal-6000 实测 ~84 分钟）。
- R0 / R1a / R1b 均已产出 val / test / extrapolation 指标 JSON。
- 输出中保留失败证据，不把失败 run 写成成功。

## 10. 实现步骤

### 阶段 A：P0 物理特征基线

1. [已完成] 新增 `tv3/ml/rocket_features.py` 中的物理序列特征读取函数。
2. [已完成] 支持从 `ultrasonic_tof_s`、`ultrasonic_sound_speed_estimated_m_per_s` 等数组提取 full / phase / early stats。
3. [已完成] 新增 `tv3/ml/rocket_training.py`，先接 RidgeCV。
4. [已完成] 新增 `run_tv3_rocket_baseline.py --feature-set physics_stats --head ridgecv`。
5. [已完成] 跑 32 序列 smoke，确认输出格式。
6. [已完成 2026-07-07] 跑服务器正式数据集 R0 并回填结果（`tv3-formal-6000`，val O₂ R²=0.603，详见 §0）。

### 阶段 B：MiniRocket-style 特征（R1 定位重写，已完成 2026-07-08）

**前置认知**：R0 已用满超声物理标量序列（tof_s / sound_speed / alpha_true），val O₂ R²=0.603。阶段 B 回答"raw 波形卷积特征能否在 R0 标量之上贡献增量"。

1. [已完成] 实现 deterministic kernel generator（seed 固定，num_kernels=128）。
2. [已完成] 实现逐序列 mmap waveform reader（朴素加载 formal-6000 OOM，改逐序列 mmap + 按核长度分组批量 einsum，峰值 ~450 MB，formal-6000 实测 ~84 分钟）。
3. [已完成] R1a 先导对照：MiniRocket 作用于 tof_s/sound_speed 等标量序列，val O₂ R²=0.515。
4. [已完成] R1b 增量验证：MiniRocket 作用于 raw 5000 点波形 + slow 统计，val O₂ R²=-0.195。
5. [已完成] 写入 feature cache（minirocket_scalar_v1 / minirocket_raw_v1）。
6. [已完成] **判定**：R1b − R1a = −0.71 < 0.05 -> raw 波形无增量。R1a(0.515)也不如 R0(0.603)。raw 波形卷积路线证伪，转 R5 验证线性极限、R3f 待评估。

### 阶段 C：MultiRocket-style 增强（❌ 不推进，R1b 证伪）

R1b 证明 raw 波形卷积无增量，R3/R4（MultiRocket raw 波形多池化）不再实施。若后续 R3f（fiber_mic）需要，再单独评估 fiber_mic 的 MultiRocket 特征。

### 阶段 D：小 MLP（R5，❌ 正式 6000 判据未通过）

R1b 证伪后，R5 最终按 [r5_mlp_implementation_plan.md](../../archive/completed/r5_mlp_implementation_plan.md) 落地为 **D0-observed 864 + raw3 MLP**（本文件旧草案「R0 1080 + lr 1e-4」已过时，以该专项计划为准）。

**实测（2026-07-09）**：val O₂ R²=−0.183，四 split 全负；相对 D0-observed −0.606，未过 +0.05 判据。浅 MLP 未复现 R5' TabPFN 非线性。可部署上限维持 D0-observed Ridge；详见记忆库 §6.9。

### 阶段 E：Hydra-style 第二轮（❌ 不推进，R1b 证伪）

R1b 无增量，R6 不再实施。

## 11. 验证命令草案

本地 smoke：

```powershell
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark --output-root data --dataset tv3-rocket-smoke --sequences 32 --timesteps 64 --workers 1
python -m tv3.pipeline.run_tv3_rocket_baseline --dataset-dir data\tv3-rocket-smoke --feature-set physics_stats --head ridgecv --output-dir outputs\tv3_rocket_smoke\r0
python -m pytest tests/test_rocket_features.py -v
python -m pytest tests/test_tv3_rocket_pipeline.py -v
```

服务器正式：

```bash
python -m tv3.pipeline.run_tv3_rocket_baseline \
  --dataset-dir data/tv3-formal-6000 \
  --feature-set physics_stats \
  --head ridgecv \
  --output-dir outputs/tv3_rocket/r0

python -m tv3.pipeline.run_tv3_rocket_baseline \
  --dataset-dir data/tv3-formal-6000 \
  --feature-set minirocket_scalar_v1 \
  --head ridgecv \
  --chunk-size 64 \
  --output-dir outputs/tv3_rocket/r1a

python -m tv3.pipeline.run_tv3_rocket_baseline \
  --dataset-dir data/tv3-formal-6000 \
  --feature-set minirocket_raw_v1 \
  --head ridgecv \
  --chunk-size 64 \
  --output-dir outputs/tv3_rocket/r1b

python -m tv3.pipeline.run_tv3_rocket_baseline \
  --dataset-dir data/tv3-formal-6000 \
  --feature-set multirocket_ultra_v1 \
  --head elasticnetcv \
  --chunk-size 64 \
  --output-dir outputs/tv3_rocket/r4
```

> `--feature-set` 的具体取值以 `run_tv3_rocket_baseline.py` 实际支持为准；`minirocket_scalar_v1` / `minirocket_raw_v1` 为阶段 B 新增的 R1a/R1b 两套缓存，落地时按 §7 特征缓存契约实现。

## 12. 风险与排查（2026-07-08 增补 R1 定位相关风险）

| 风险               | 触发信号                    | 处理                                                  |
| ---------------- | ----------------------- | --------------------------------------------------- |
| 波形特征生成太慢         | R1b 特征生成超过可接受时间         | 减 kernel_count，先只用 TOF 附近窗口                         |
| 特征维度太高           | RidgeCV 内存过高            | 降 kernel_count，使用 float32 cache，分组训练                |
| raw 波形无增量        | R1b − R1a < 0.05        | 不继续堆 raw 波形特征，转方向 D（fiber_mic, R3f）或 Top1 端到端可微 TOF |
| R1a/R1b 对照被跳过    | 直接上 R1b 全量无 R1a 基线      | R1a 是 R1b 的判定基线，跳过则 R1b 增量无法归因，必须先跑                 |
| O2 仍不可预测         | R3/R3f O2 R2 < 0.50     | 转 O2 专用通道，不继续堆 DL                                   |
| CO2 下降明显         | CO2 R2 低于 R0 slow Ridge | 检查 scaler、slow 通道漂移、split 对齐                        |
| MLP best epoch=1 | Ridge 正常但 MLP 崩         | 降 lr，检查 target scale 和 loss 权重                      |
| 所有头都失败           | R0 到 R5 都异常             | 检查 labels、split、feature row order、manifest 与实际数组    |

## 13. 文献依据

| 方法            | 依据                                                                                           | 用法                                                                   |
| ------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| ROCKET        | Dempster et al. 2020, Data Mining and Knowledge Discovery, DOI: `10.1007/s10618-020-00701-z` | 固定随机卷积核 + 线性分类器，本项目迁移为回归特征                                           |
| MiniRocket    | Dempster et al. 2021, KDD, DOI: `10.1145/3447548.3467231`                                    | 更快更确定的卷积核特征                                                          |
| MultiRocket   | Tan et al. 2022, Data Mining and Knowledge Discovery, DOI: `10.1007/s10618-022-00844-1`      | 多池化算子(PPV/mean/std/slope)，201 引                                      |
| Hydra         | Dempster et al. 2023, Data Mining and Knowledge Discovery, DOI: `10.1007/s10618-023-00939-3` | competing kernels，作为第二轮增强                                            |
| InceptionTime | Fawaz et al. 2020, Data Mining and Knowledge Discovery, DOI: `10.1007/s10618-020-00710-y`    | 多尺度卷积时序建模参考                                                          |
| 算法排序与代码示例     | [波形特征提取算法评估.md](../../methods/波形特征提取算法评估.md) / [波形特征提取算法代码示例.md](../../methods/波形特征提取算法代码示例.md)                          | 10 算法排序：Top1 TOF 工程(9.5) / Top2 MultiRocket(9.0) / Top5 wav2vec(8.0) |

## 14. 完成定义（2026-07-08 据 R1a/R1b 实测更新）

本方案完成时，应具备：

1. ✅ 一个可复现的 ROCKET 特征缓存格式（physics_stats_v1 / minirocket_scalar_v1 / minirocket_raw_v1 三套）。
2. ✅ R0、R1a、R1b 三个实验完成，且 R1b 相对 R1a 的增量已明确判定（−0.71，无增量）。
3. RidgeCV / small MLP 两类 head：RidgeCV 已跑通（R0/R1a/R1b），small MLP（R5）待跑。
4. ✅ 明确回答：raw 5000 点波形的卷积特征相对 R0 超声标量序列**无增量**（R1b − R1a = −0.71 < 0.05）。
5. ✅ 明确回答：端到端 DL 失效与数据物理不可辨识无关（[dl_training_plan.md §11.4](../legacy/dl_training_plan.md#114-服务器验证结果tv3-formal-6000-50-epoch-单-seed-rtx-5880) DL 失效在 encoder 架构；R0 Ridge 0.603 证明物理可辨识）。
6. ✅ 下一步分叉已定：R1b 无增量 -> raw 波形卷积路线证伪，转 R5（R0 特征 + 小 MLP）验证线性极限；R3f（fiber_mic）待评估；若 R5 无增益且 R3f 失效，接受 R0(0.603)作为现有通道极限。
