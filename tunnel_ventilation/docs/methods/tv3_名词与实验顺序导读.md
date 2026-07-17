# tv3 名词与实验顺序导读

> 面向初学者的名词说明。按**实验推进顺序**分级，便于对照记忆库与专项计划阅读。  
> 正式指标与当前状态以 [掘进通风项目记忆库.md](../掘进通风项目记忆库.md) 为准；本文件只解释概念，不重复承担全局结论源责任。  
> **维护义务**（记忆库 §1.4 / §8.3）：新增或实质变更正式实验、代码模块、算法/builder/头、协议门禁或 verdict 语义时，必须在同一变更批次更新本导读的名词定义、分级位置与索引；未联更不得视为文档齐全。  
> 编写日期：2026-07-13；2026-07-14 增补 `waveform_preprocess` 工程名词。

---

## 0. 怎么用这份文档

1. **第一次读**：先读 §1 心智图，再按 §2→§9 顺着实验线往下。
2. **查词**：用文末 [索引表](#索引表按拼音与英文) 跳转。
3. **读计划时**：先分清该词属于 **物理传感 / 统计信息 / 模型协议** 哪一层。

### 0.1 两层问题（最重要）

| 层次 | 问的是什么 | 典型工作 |
| --- | --- | --- |
| **模型层** | 在现有信号上，换算法能不能更准？ | D0、B1、B7、MLP… |
| **物理 / 信息层** | 这些信号里，O₂ 信息是否本来就不够？ | 可辨识性、误差预算 |

模型 R² 低，不一定是“模型菜”，也可能是**物理上分不清**。

### 0.2 一条故事线（复习用）

```text
场景契约与数据生成
  → D0：O₂ 信息在哪？窄窗是否已有物理墙？
  → D2 失败 / D2b 成功：raw 波形如何变成可信特征？
  → R5 / R5-T / R7：observed 特征上的回归头探索
  → B1 → B6 → B7：RawDSP 默认头冻结与 OOD 协议
  → 模块 C：物理分组失败，停止该分支
  → 可辨识性 v1：物理误差预算；P90=0.4 vol%、nuisance=50%、拒绝率=5% 均失败，flow 未表示 → information-source-upgrade
  → 当前 P0：先在经风速核验的静止空气中验证单向链路，不外推到通风现场
  → 并行 P1-exp：EC-MSW E1d-SB 正式 `parity_passed`；下一步 E1r attachment 联合审计；E2 仍禁止
```

---

## 1. 场景与数据契约（最先建立）

做任何实验前先固定：预测什么、数据长什么样、哪些事禁止做。

### 1.1 场景命名

| 名词 | 说明 |
| --- | --- |
| **tv3 / tunnel_ventilation** | 掘进通风场景。预测空气中 CO₂、O₂、N₂。与掺氢天然气、合成气同属大项目，但 schema 与数据隔离。 |
| **COMPONENT_FIELDS** | 预测目标字段：`x_CO2`, `x_O2`, `x_N2`。 |
| **SLOW_CHANNELS** | 慢通道列表（见 §1.3）。tv3 正式 clean 数据为 7 通道，无 `V_NDIR_CH4`。 |

### 1.2 组分与闭包

| 名词 | 说明 |
| --- | --- |
| **组分 / composition** | 混合气体各气体体积浓度百分比。 |
| **闭包 / closure** | `x_CO2 + x_O2 + x_N2 = 100%`。数据层强制；扰动分析时改一个组分必须明确补偿另一组分。 |
| **sum_abs_error** | 模型三输出之和与 100% 的绝对偏差。**监控项**，不是训练强制约束。 |
| **raw3 / out_dim=3** | 模型直接输出三个浓度，不在模型内用闭包残差结构。 |
| **gas_head** | 某些场景的闭包结构头。**tv3 禁止**。 |
| **target_transform / ILR / ALR** | 依赖严格闭包几何的变换。tv3 正式路线不用。 |

### 1.3 传感模态与慢通道

| 名词 | 说明 |
| --- | --- |
| **NDIR** | 非色散红外。CO₂ 有红外吸收 → 直接可测。O₂/N₂ 同核双原子，几乎无红外活性。 |
| **TCS** | 热导传感器。O₂/N₂ 热导差很小，对区分两者帮助有限。 |
| **ultrasonic / 超声** | 主信息源之一；通过声速 / TOF 间接反映分子量变化。 |
| **fiber_mic** | 光纤麦克风通道；可跳过存储以减小数据集体积，代码可恢复。 |
| **slow channels（慢通道）** | 相对波形的低频/标量通道：`V_NDIR_CO2`, `V_TCS`, `T_C`, `P_MPa`, `H_RH`, `L_m`, `piston_position_m`。 |
| **waveform / raw ultrasonic** | 超声时间序列电压。可从中估 TOF、峰值、衰减等。 |
| **dequantize / scale** | 存储为 int16 时，用 per-timestep `scale` 还原电压：`int16 * scale`。 |
| **normalize_waveforms** | 对 dequant 后波形做**逐帧** z-score（population std）。正式 fusion/E1 与 `waveform_adc_scale=1.0` 联用。 |
| **waveform_stats_features** | 归一化**前**从 dequant 电压算的逐帧统计（如 `log_std`、`log_max_abs`），拼进 slow 侧通道，保留绝对幅度信息。 |
| **waveform_preprocess** | 训练数据通路开关：`cpu`（worker 内组装）或 **`gpu`（正式默认：只搬 int16+scale，设备侧 dequant/stats/z-score）**。不改变数值语义；gpu 不支持 window/phase_windows/augment。见 [server_training_guide §4.5](../operations/server_training_guide.md#45-波形数据通路-waveform_preprocessp1-吞吐)。 |

### 1.4 数据标识与基准集

| 名词 | 说明 |
| --- | --- |
| **mixture_id** | 气体配方 ID（如 `M000001`）。split 主键。 |
| **sequence_id** | 测量序列 ID（如 `Q000001`）。 |
| **benchmark** | 固定规则生成的标准数据集（条件表、波形、标签、manifest…）。 |
| **tv3-formal-6000** | 正式 6000 序列规模数据集。正式结论以此为准。 |
| **clean** | 通道定义干净的版本（无错误 `V_NDIR_CH4` 等）。 |
| **manifest** | 数据集说明书 JSON。下游以 manifest 为准，不硬编码全局常量。 |
| **schema / composition_scheme** | 标签字段、背景气、闭包语义等契约。tv3 为 `tunnel-ventilation-1`。 |
| **sim_revision** | 仿真链路版本标记（载波频率、ADC、声程、物理后端等）。旧 revision 数据不可混训新链路。 |

### 1.5 真值与可部署输入边界

| 名词 | 说明 |
| --- | --- |
| **oracle** | 仿真内部真值特征（真 TOF、真声速、真 alpha）。只用于上界探针，**不能**当部署输入。 |
| **observed** | 测量级特征：带噪声/估计误差的 TOF、峰位置、估计声速、quality 等。比 oracle 接近可部署，但仍可能是仿真同步生成。 |
| **alpha** | 声波衰减系数。真 alpha 通常只进 oracle/诊断。 |
| **auxiliary target** | 辅助学习目标（如 TOF）。不等于可以把真值塞进部署特征。 |

---

## 2. 物理传感核心量（贯穿全实验）

这些量从数据生成到可辨识性审计一直出现。

### 2.1 传播与计时

| 名词 | 说明 |
| --- | --- |
| **TOF / Time of Flight** | 声波飞过声程的时间。近似 `tof ≈ L / c`。O₂ 主信息源之一。 |
| **sound speed / c / 声速** | 混合气中声波速度。常由 TOF 与 L 反推；与 TOF **不是两条独立信息**。 |
| **L / 声程** | 超声传播路径长度。L 误差会伪装成浓度误差。 |
| **trigger jitter** | 触发/计时起点的随机偏差（情景里常用微秒量级）。直接污染 TOF。 |
| **SNR** | 信噪比。低 SNR 使峰值与 TOF 估计更糊。 |

### 2.2 环境与流动干扰

| 名词 | 说明 |
| --- | --- |
| **nuisance（干扰量）** | 不是最终目标、但会改观测的量：T、RH、L、jitter、flow、SNR… 核心问题是 O₂ 信号是否大于 nuisance。 |
| **flow / 流速 / flow projection** | 气流改变有效传播时间。v1 可能未建模 → 标 `not_represented`。 |
| **双向超声 / bidirectional** | 一去一回测 TOF，分离流速与声速。flow 主导误差时的物理升级方向；当前暂停保留。 |
| **静止空气范围 / static air** | 由独立风速核验确认的受控范围，不等于“仿真代码没写 flow”。当前 P0 只在这一范围验证单向 O₂ 可测性。 |
| **湿空气 / RH** | 湿度影响声学与弛豫；若未充分建模，不能写成已验证上限。 |
| **已表示 / 未表示** | 当前仿真链路是否真正建模该量。未表示只能记缺口，不能当已验证证据。 |

### 2.3 信息源升级选项

| 名词 | 说明 |
| --- | --- |
| **TDLAS** | 可调谐二极管激光吸收光谱；可做 O₂ 专用光学通道。声学信息不足时的硬件升级选项，不是当前默认。 |
| **多频超声** | 用多个载波频率增加独立观测维度。 |

---

## 3. 评价与划分协议（所有模型实验共用）

### 3.1 误差与拟合指标

| 名词 | 说明 |
| --- | --- |
| **R²** | 决定系数。1 完美；0 约等于猜均值；**负** 比猜均值还差。 |
| **MAE** | 平均绝对误差。 |
| **P90 误差** | 误差的第 90 百分位：90% 样本不差于该值。误差预算更常看 P90。 |
| **0.8% O₂ bin / 窄窗口** | O₂ 仅在约 0.8 个百分点宽的小区间内评价。全域可粗辨识时，窄窗仍可能全负 R²。 |
| **全域** | 在较大 O₂ 范围（如约 18–21%）上评价。 |

### 3.2 数据划分

| 名词 | 说明 |
| --- | --- |
| **split** | train / val / test 等划分。正式以 `mixture_id` 为主键。 |
| **extrapolation / extrap** | 相对更难的外推划分；仍不等于完整物理 OOD。 |
| **OOD** | 分布外样本（如组分落在训练边界外）。随机 test ≠ 真 OOD。 |
| **S-Y** | 按标签/目标边界的 OOD 选择器（y_margin 一类）。 |
| **S-L** | 按 LHS 采样边界的 OOD 选择器。与 S-Y **分列报告**。 |
| **R / L** | 协议中的 ID 稳定性对照划分，不作“全部 OOD=某单一 R²”的混写依据。 |
| **LHS** | 拉丁超立方采样，用于覆盖组分/条件空间。 |
| **SPXY** | 基于样本距离的划分方法；B7 正式 OOD 使用 observed 统计 profile。 |

### 3.3 门禁与冻结

| 名词 | 说明 |
| --- | --- |
| **protocol_pass** | 预注册的完整协议矩阵通过。 |
| **residual_pass / stable_pass** | 某类头/多 seed 稳定性门禁通过。 |
| **audit_passed / audit_failed** | 审计数值与 provenance 是否完整可信。 |
| **baseline freeze / 基线冻结** | 钉死 B1/B7 配置、结果 hash、split、RawDSP cache、Git commit。审计必须在冻结基线上做。 |
| **provenance** | 产物来源可追溯信息（谁构建、何种 digest、哪次 commit）。 |

---

## 4. 阶段 D0：信息在哪？（特征消融基线）

**目的**：不追求最强模型，先回答“O₂ 信号主要来自哪路特征、上界在哪”。

### 4.1 实验含义

| 名词 | 说明 |
| --- | --- |
| **D0** | 六组特征拆分 + Ridge 基线阶段。 |
| **D0-observed** | observed 级全特征 + Ridge。测量级线性 baseline。 |
| **D0-oracle / R0 oracle** | oracle 真值特征 + Ridge。不可部署的上界探针。 |
| **D0-tof-only** | 仅 TOF 相关特征。用于证明 TOF 是 O₂ 主来源。 |
| **D0-slow-only / no-tof** | 去掉 TOF 后崩溃 → 慢通道 alone 不够。 |
| **D0-no-tcs** | 去掉 TCS 几乎不掉分 → TCS 边际贡献小。 |
| **oracle 膨胀** | oracle 明显高于 observed 的那截差距；说明测量误差占一部分，但不解释窄窗全负。 |

### 4.2 从 D0 应记住的结论骨架

1. O₂ **不是完全无信号**（全域粗辨识可行）。  
2. **窄区间存在物理墙嫌疑**（0.8% bin 内连 oracle R² 也负）。  
3. 后续缺的是**物理误差账**，不是立刻再叠一个复杂头。

---

## 5. 阶段 D2 → D2b：从 raw 波形到可信特征

**目的**：回答“不靠仿真同步 observed 数组，只从保存波形能否恢复同等信息”。

### 5.1 D2（失败路线）

| 名词 | 说明 |
| --- | --- |
| **D2 / TOF-PhaseNet** | 试图用可学习网络从 raw 波形直接做 TOF/组分。正式训练失败，原实现停止。 |
| **matched filter** | 匹配滤波，用已知模板在波形上找回波位置。 |
| **softargmax** | 可微地从响应曲线取“峰位置”的近似。D2 中 temperature 等固定不当会导致不可训。 |
| **固定延迟坐标偏置** | TOF 指标混入装置固定延迟时，会污染评价与学习目标。 |

### 5.2 D2b / RawDSP（成功路线）

| 名词 | 说明 |
| --- | --- |
| **D2b** | 固定 DSP 提取特征 + 下游回归头。与失败的 D2 实现脱钩。 |
| **DSP** | 数字信号处理。 |
| **RawDSP** | 对 raw 超声做固定、可审计处理，输出特征向量（如 1008 维），再喂 B1/B7。 |
| **builder** | 特征构建器命名，如 `d0_raw_dsp_physics_stats_v1`。改特征必须改名。 |
| **frame-level fidelity** | 帧级保真：DSP 估峰/TOF 是否接近真值、各 split 是否过门。失败则先修 DSP。 |
| **cache / feature cache** | 预计算的 RawDSP 特征缓存及 manifest digest。 |
| **parity** | 新特征 + 简单头是否复现旧 observed 线性性能（如 B1 vs D0-observed）。 |

---

## 6. 阶段 R 系列：observed 特征上的回归头探索

**目的**：在已冻结的 observed 特征契约上，试非线性头与正则化，不改变传感前端。

| 名词 | 说明 |
| --- | --- |
| **R5** | 小 MLP on observed。默认配方正式未通过。 |
| **R5-T** | 在 R5 上增加**目标标准化**（按训练目标尺度缩放）。多 seed 稳定通过；说明默认 R5 失败主因常是优化尺度。 |
| **R5' / TabPFN** | TabPFN on observed。证明特征上存在可学非线性；**不**等于 raw 链路已部署。 |
| **R7 / ExtraTrees** | 树模型。正式 6000 未过，训练–验证落差大。 |
| **target scaling / 目标标准化** | 让 CO₂/O₂/N₂ 不同量纲的损失可比较，改善优化。 |
| **Ridge** | L2 正则线性回归。稳定 baseline。 |
| **MLP** | 多层感知机。用于非线性残差或直接回归。 |

---

## 7. 阶段 B1 → B7：RawDSP 默认头与协议冻结

**目的**：在 RawDSP 特征上建立可部署候选头，并用重复 split + 双 OOD 固化结论。

### 7.1 头版本

| 名词 | 说明 |
| --- | --- |
| **B1** | RawDSP + Ridge。对齐 D0-observed 的线性水平。 |
| **B6** | RawDSP + 目标标准化 MLP 等。多 seed `stable_pass`，后被 B7 超越。 |
| **B7** | **OOF Ridge + 残差 MLP**。当前默认 RawDSP 头。 |
| **OOF / Out-Of-Fold** | 折外预测：每折用其余折训练的模型预测本折，减轻残差头偷看训练集过拟合。 |
| **residual / 残差** | `真值 − 基底预测`。B7 中 MLP 只学 Ridge 没学到的部分。 |
| **OOF Ridge residual MLP** | B7 结构全称。 |

### 7.2 协议与 OOD

| 名词 | 说明 |
| --- | --- |
| **B7 protocol** | `R/L/S-Y/S-L × 多 split seed × 多 training seed` 完整矩阵。 |
| **protocol_pass** | 相对 B1 的 test/OOD 增益等门禁全部满足。 |
| **ΔR²** | 相对基线（常为 B1）的 R² 差。 |
| **selector 边界** | S-Y 与 S-L 的绝对 OOD 水平不同，禁止混写成统一“OOD=0.70”。 |

### 7.3 模块 C（旁支证伪）

| 名词 | 说明 |
| --- | --- |
| **模块 C / grouped bottleneck** | 按物理组先独立压缩再融合。C1 完整矩阵 `grouped_failed`。 |
| **C1 physical / C2 permuted** | 物理分组 vs 打乱分组对照。失败主因是早期物理隔离，不是单纯参数量。 |
| **分流结论** | 停止 C1 调参扩展；**B7 保持默认头**。 |

### 7.4 EC-MSW-GatedNet（当前 P1 实验线）

**目的**：检验位置敏感的 learned waveform 表示能否在冻结协议下达到 B1 parity，再逐步判断环境调制和动态窗口是否有独立收益。它不替换 B7，也不推翻 identifiability v1。

| 名词 | 说明 |
| --- | --- |
| **EC-MSW-GatedNet** | Environment-Conditioned Multi-Scale Waveform Gated Network。规划中的环境条件化、多尺度波形、晚期软门控框架。 |
| **E0–E5** | 单因素实验序列：冻结基线 → E1 encoder → FiLM → attention → soft gate → 混合/纯端到端对照。每步未过门不得进入下一步。 |
| **E1 / `ec_msw_e1`** | 已完成正式训练与审计的阶段：共享卷积 stem、三种 kernel/dilation 分支、显式位置统计、固定跨帧聚合和 raw3 小头；正式判定失败。 |
| **位置敏感统计** | 对分支激活同时保留均值、最大值及关于绝对采样坐标的一阶矩，避免只做全局池化后丢失峰位。 |
| **FiLM** | 用可部署环境 token 生成逐通道缩放与偏置。属于 E2，当前未启动。 |
| **attentive statistics pooling** | 动态加权多个 frame/window，并输出加权均值与标准差。属于 E2，当前未启动。 |
| **soft gate / MoE** | 样本级软路由多个尺度或机制专家。属于 E3 以后，当前未启动。 |
| **smoke** | 最小链路运行检查，只证明数据、模型、训练器和产物写出可工作；不证明精度、parity 或 OOD 泛化。 |
| **当前门** | E1d-SB 正式 `parity_passed`；下一步 E1r attachment（冻结 frame + e1d_sb 序列 Ridge）。`e2_allowed=false`，B7 继续作为默认头。 |
| **position fidelity probe** | 冻结 encoder 后，仅用 train frame embedding 拟合线性 peak-index probe，在 val/test/extrapolation 全帧评价；peak target 不进入模型或主损失。 |
| **E1 parity probe** | 冻结 sequence embedding 后另训 train-only Ridge，与 B1 的三个 split R²做非劣比较；原神经网络输出头不参与 parity。 |
| **E1 正式失败** | frame peak MAE 约 `71–72 samples`、P95 约 `155 samples`；冻结 embedding 的 O₂ R²在三个 split 均为负，说明 learned encoder 没有保留 RawDSP 已能恢复的峰位 / TOF 信息。 |
| **E1r / 模板坐标锚点** | 使用 RawDSP train-only baseline median 模板对当前 raw waveform 做冻结匹配滤波，将绝对峰位直接保留为 embedding 第一维。正式 frame MAE约 `0.037 sample`，但冻结 embedding 的 O₂ R²仅 `0.0006–0.0325`，verdict=`b1_parity_failed`。 |
| **frame fidelity 非充分性** | 峰位可被高精度线性恢复，只证明坐标存在；不能证明固定 sequence embedding 同时保留了 phase、校准、声程与质量信息。E1r 正式结果已验证这一边界。 |
| **E1d** | E1r 之后的冻结表示诊断阶段。依次对比时序/phase 聚合、delay/TOF-L 校准组和 waveform quality 组；正式 6000 已完成，verdict=`minimal_deployable_set_found`。 |
| **E1d 入口** | `python scripts/run_ec_msw_e1d.py --config configs/tv3_ec_msw_e1d.json`；正式产物 `outputs/tv3_ec_msw/e1d_s20260704/`。 |
| **E1d feature set** | 真实冻结 E1r 对照为 `e1r_sequence_embedding`、`e1r_peak_lmm`、`e1r_peak_b1_windows`；RawDSP 消融如 `peak_lmm`、`peak_stats7_phase`、`peak_phase_plus_delay`、`cal_plus_*`；`full_b1` 仅正对照。 |
| **E1d 正对照门** | 正式 run 重建 full B1，并以 R²绝对容差 `1e-6` 在 val/test/extrapolation 复现冻结 reference；否则 `positive_control_failed`，不解释其他候选。 |
| **E1d 正式结论** | 正对照通过；E1d-2 校准组未补回 O₂；E1d-3 加 SNR 后过门。compact=`cal_plus_corr_psr_snr`（首选）与 `cal_plus_quality_width`。 |
| **compact parity set** | 扣除冻结 slow 后的诊断特征数 ≤ full B1 诊断块一半，且完整三 split 过 O₂/CO₂/N₂非劣门的最小可部署集合；只有它允许实现新 sequence builder。 |
| **E1d-SB** | 可部署 builder `e1d_sb_cal_plus_corr_psr_snr_v1`；正式 `parity_passed`（O₂ `0.393 / 0.453 / 0.369`）；不是 E2。 |
| **E1r attachment** | probe-only 联合审计：冻结 E1r frame fidelity + e1d_sb 序列 Ridge 替换 `last/mean/max`；入口 `scripts/audit_ec_msw_e1r_attachment.py`；正式产物 `e1r_attach_e1d_sb_s20260704/`。 |
| **E2s / 结构化声速反演头** | 原方案把加权 LS 声速反演当唯一聚合层；前置门 E1d-2 **未过门**，不得单独启动。SNR 加权 LS 仅可作 builder 内可选消融。见 `active/tv3_ec_msw_structured_sequence_head_plan.md`。 |

---

## 8. 阶段 Identifiability：物理可辨识性与误差预算

**目的**：在不改 B1/B7、不改正式 RawDSP builder 的前提下，量化 O₂ 相对 nuisance 的理论边界，并给出分流 verdict。

对应计划：[tv3_identifiability_implementation_plan.md](../active/tv3_identifiability_implementation_plan.md)

### 8.1 目标与非目标

| 名词 | 说明 |
| --- | --- |
| **可辨识性 / identifiability** | 在给定观测与干扰下，O₂ 等参数是否被充分约束。白话：能不能分清“O₂ 变了”还是“温度变了”。 |
| **误差预算 / error budget** | 把总误差拆成各来源贡献并排序。工程误差账本，不是再训一个模型。 |
| **非目标** | 不训练新头；真值 TOF/c/alpha 只进 oracle 列；未表示 nuisance 不伪装成证据。 |

### 8.2 灵敏度与数值审计

| 名词 | 说明 |
| --- | --- |
| **局部灵敏度 / sensitivity** | `∂观测/∂参数`。O₂ 灵敏度大则好测；T 灵敏度大则易淹没 O₂。 |
| **中心差分** | `(f(x+h)-f(x-h))/(2h)` 近似导数。 |
| **单边差分** | 边界处只能朝一侧差分；结果需标记。 |
| **差分步长稳定性** | h 取半/双后导数变化超容差 → 审计失败。 |
| **条件数 / condition number** | 问题病态程度。极大 → 参数近共线，噪声放大严重。 |

### 8.3 统计信息论量

| 名词 | 说明 |
| --- | --- |
| **Fisher information** | 数据中关于参数的信息量。矩阵越大且良态，原则上可估得越准。 |
| **秩 / rank / rank-deficient** | 有效维度。v1 单一 TOF 对多参数 → 秩 1，联合 Fisher 秩亏。 |
| **unavailable_rank_deficient** | 因秩亏声明 Fisher/CRLB 不可用，禁止硬算假精度。 |
| **CRLB** | Cramér–Rao 下界：无偏估计方差的理论下限。Fisher 不可逆时不可用。 |
| **nuisance-marginalized information** | 边缘化干扰参数后，剩余的 O₂ 信息。秩亏时不可靠。 |
| **观测不重复计数** | `c` 与 `tof=L/c` 共享同一 TOF 信息，Fisher 中禁止双计。 |

### 8.4 误差折算与联合传播

| 名词 | 说明 |
| --- | --- |
| **等效 O₂ 误差** | 把 nuisance 不确定度换成看起来像多少 %O₂。直觉：`δO₂ ≈ δ_nuisance / (∂观测/∂O₂)`。 |
| **联合误差 / 协方差传播** | 多误差源按方差/协方差一起传到 O₂。 |
| **correlation_group** | 相关 nuisance 分组，禁止假装独立后乱加。 |
| **情景误差** | 在登记假设（如 1 K、3 μs jitter）下的预算。**不是**部署证书精度。 |
| **主导项排序** | 谁贡献最大误差 → 决定下一步改 flow 还是 RH/设备参数等。 |

### 8.5 业务门限与 verdict

| 名词 | 说明 |
| --- | --- |
| **business threshold / 业务门限** | 业务真正需要的精度与拒绝率，写入配置，禁止实现内隐式默认。v1 已登记 P90=`0.4 vol%`、nuisance=`50%`、拒绝率=`5%`，证据见 `references/tv3_identifiability_business_threshold_evidence.md`。 |
| **target_p90_o2_error_percent** | 可接受的 P90 O₂ 误差上限；v1 为 `0.4 vol%`。这是研究门，不是安全联锁认证。 |
| **max_nuisance_fraction_of_signal** | 干扰相对窄窗信号的最大允许比例；v1 为 `0.50`。单项最坏等效 O₂ P90 除以 `0.8 vol%` 窄窗口宽度，且不得配置得低于 `0.01`。 |
| **max_rejection_rate** | 最大允许拒绝率。v1 为 `0.05`；只要存在阻断性未表示 nuisance，`v1_blocking_nuisance_reject_all` 就拒绝全部审计点。当前 flow 使拒绝率为 100%，因此失败。 |
| **verdict** | 审计后的分流裁决（见下表）。 |

| verdict | 含义 |
| --- | --- |
| `continuous_regression_supported` | 窄窗连续回归仍被物理与业务门限支持；可按主导项改物理。 |
| `coarse_monitoring_only` | 窄窗不够，全域仍有信息 → 改分档/趋势，冻结精细连续回归。 |
| `information_source_upgrade_required` | 全域信息不足或未表示 nuisance 主导 → 评估多频/TDLAS 等。v1 当前状态：flow 未表示、100% 拒绝，触发该 verdict；它只约束部署范围。 |
| `inconclusive_missing_business_threshold` | **缺业务门限、拒绝率/规则或协方差依据**；不得宣布继续或停止。v1 已不处于该状态。 |
| `audit_failed` | 基线 hash、闭包、数值稳定性等失败；先修审计，不训新头。 |

### 8.6 v1 结果应如何理解（读词防误读）

| 说法 | 正确理解 |
| --- | --- |
| `audit=passed` | 审计流程与数值自洽通过。 |
| `verdict=information_source_upgrade_required` | 三项业务门均已量化；flow 未表示导致 100% 拒绝，部署范围内不能继续单向 TOF 窄窗连续回归。当前可在独立核验的静止空气范围开展受控可测性验证，但不能将其误写为现场通过。 |
| 窄窗 P90 ≈ 8.85–12.99% O₂ | **登记情景下的 v1 TOF 误差预算**，不是现场部署精度，也不是“声学路线已失败”。 |
| Fisher 秩 1 | 单一单向 TOF 观测下，多参数联合信息不足；不能假装完成了完整 nuisance 边缘化。 |

---

## 9. 横切工具词（代码与产物）

| 名词 | 说明 |
| --- | --- |
| **config JSON** | 实验/审计超参配置，禁止魔法数散落实现。 |
| **outputs/** | 正式运行产物目录（如 `tv3_d0/`, `tv3_b7_protocol/`, `tv3_identifiability/`）。 |
| **metrics.json** | 单次 run 的指标文件。 |
| **verdict.json** | 协议或审计的最终裁决文件。 |
| **representation_audit.json** | 各参数已表示/未表示清单。 |
| **sensitivity.csv / fisher_information.csv / error_budget.csv** | 可辨识性分项产物。 |
| **nuisance_fraction_summary.csv** | 每个已声明 nuisance 情景相对 `0.8 vol%` 窄窗口的最坏等效 O₂ P90 比例；用于 50% 门。 |
| **不可覆盖输出** | 正式审计目录写过一次后禁止静默覆盖，防止 provenance 污染。 |

---

## 10. 按实验顺序的速查表

| 顺序 | 阶段 | 核心问题 | 关键名词 |
| --- | ---: | --- | --- |
| 0 | 契约 | 预测什么、禁止什么 | tv3、闭包、raw3、manifest、oracle/observed |
| 1 | 物理量 | 信号从哪来、谁在捣乱 | TOF、c、L、nuisance、flow、SNR |
| 2 | 评价协议 | 怎么才算准、怎么划分 | R²、P90、OOD、S-Y/S-L、freeze |
| 3 | D0 | 信息在哪、窄窗有没有墙 | D0-observed、oracle 膨胀、0.8% bin |
| 4 | D2/D2b | 波形能否变成可信特征 | TOF-PhaseNet、RawDSP、fidelity、builder |
| 5 | R 系列 | observed 上非线性头 | R5、R5-T、TabPFN、R7 |
| 6 | B 系列 | RawDSP 默认头 | B1、B6、B7、OOF、residual、protocol_pass |
| 7 | 模块 C | 物理早期分组是否有用 | grouped bottleneck、grouped_failed |
| 8 | Identifiability | 物理上限与分流 | 灵敏度、Fisher、CRLB、误差预算、verdict |
| 9 | 当前并行线 | 受控静止空气可测性与新波形表示是否值得继续 | static air、EC-MSW、E1、E1r、E1d、parity |

---

## 索引表（按拼音与英文）

| 词条 | 章节 |
| --- | --- |
| alpha | [§1.5](#15-真值与可部署输入边界) |
| audit_passed / audit_failed | [§3.3](#33-门禁与冻结) |
| B1 / B6 / B7 | [§7.1](#71-头版本) |
| baseline freeze | [§3.3](#33-门禁与冻结) |
| bidirectional / 双向 | [§2.2](#22-环境与流动干扰) |
| builder | [§5.2](#52-d2b--rawdsp成功路线) |
| business threshold | [§8.5](#85-业务门限与-verdict) |
| clean / tv3-formal-6000 | [§1.4](#14-数据标识与基准集) |
| condition number / 条件数 | [§8.2](#82-灵敏度与数值审计) |
| correlation_group | [§8.4](#84-误差折算与联合传播) |
| CRLB | [§8.3](#83-统计信息论量) |
| D0 / D0-observed / oracle | [§4](#4-阶段-d0信息在哪特征消融基线) |
| D2 / TOF-PhaseNet | [§5.1](#51-d2失败路线) |
| D2b / RawDSP / DSP | [§5.2](#52-d2b--rawdsp成功路线) |
| dequantize / scale | [§1.3](#13-传感模态与慢通道) |
| E0–E5 / E1 / E1r / E1d / E1d-SB / attachment / E2s / `ec_msw_e1` | [§7.4](#74-ec-msw-gatednet当前-p1-实验线) |
| EC-MSW-GatedNet | [§7.4](#74-ec-msw-gatednet当前-p1-实验线) |
| E2s / 结构化声速反演头 | [§7.4](#74-ec-msw-gatednet当前-p1-实验线) |
| error budget / 误差预算 | [§8.1](#81-目标与非目标) |
| fidelity | [§5.2](#52-d2b--rawdsp成功路线) |
| Fisher information | [§8.3](#83-统计信息论量) |
| FiLM / attention / soft gate / MoE | [§7.4](#74-ec-msw-gatednet当前-p1-实验线) |
| flow | [§2.2](#22-环境与流动干扰) |
| gas_head / raw3 | [§1.2](#12-组分与闭包) |
| identifiability / 可辨识性 | [§8.1](#81-目标与非目标) |
| LHS / SPXY | [§3.2](#32-数据划分) |
| L / 声程 | [§2.1](#21-传播与计时) |
| MAE / P90 / R² | [§3.1](#31-误差与拟合指标) |
| manifest / mixture_id | [§1.4](#14-数据标识与基准集) |
| MLP / Ridge | [§6](#6-阶段-r-系列observed-特征上的回归头探索) |
| 模块 C | [§7.3](#73-模块-c旁支证伪) |
| NDIR / TCS | [§1.3](#13-传感模态与慢通道) |
| normalize_waveforms | [§1.3](#13-传感模态与慢通道) |
| nuisance / 已表示未表示 | [§2.2](#22-环境与流动干扰) |
| OOD / S-Y / S-L | [§3.2](#32-数据划分) |
| OOF / residual | [§7.1](#71-头版本) |
| oracle / observed | [§1.5](#15-真值与可部署输入边界) |
| parity | [§5.2](#52-d2b--rawdsp成功路线) |
| position fidelity probe / E1 parity probe | [§7.4](#74-ec-msw-gatednet当前-p1-实验线) |
| protocol_pass | [§3.3](#33-门禁与冻结) / [§7.2](#72-协议与-ood) |
| rank-deficient | [§8.3](#83-统计信息论量) |
| R5 / R5-T / R7 / TabPFN | [§6](#6-阶段-r-系列observed-特征上的回归头探索) |
| sensitivity / 灵敏度 | [§8.2](#82-灵敏度与数值审计) |
| smoke | [§7.4](#74-ec-msw-gatednet当前-p1-实验线) |
| SNR / jitter | [§2.1](#21-传播与计时) |
| sum_abs_error / 闭包 | [§1.2](#12-组分与闭包) |
| TDLAS | [§2.3](#23-信息源升级选项) |
| TOF / 声速 c | [§2.1](#21-传播与计时) |
| tv3 | [§1.1](#11-场景命名) |
| verdict 各状态 | [§8.5](#85-业务门限与-verdict) |
| waveform / raw ultrasonic | [§1.3](#13-传感模态与慢通道) |
| waveform_preprocess | [§1.3](#13-传感模态与慢通道) |
| waveform_stats_features | [§1.3](#13-传感模态与慢通道) |
| 等效 O₂ 误差 | [§8.4](#84-误差折算与联合传播) |
| 窄窗口 0.8% bin | [§3.1](#31-误差与拟合指标) |

---

## 相关文档

| 文档 | 用途 |
| --- | --- |
| [掘进通风项目记忆库.md](../掘进通风项目记忆库.md) | 当前事实与正式结论 |
| [active/tv3_identifiability_implementation_plan.md](../active/tv3_identifiability_implementation_plan.md) | 可辨识性实施计划 |
| [active/tv3_static_air_feasibility_implementation_plan.md](../active/tv3_static_air_feasibility_implementation_plan.md) | 当前 P0：静止空气的测量链校准与独立 holdout |
| [active/tv3_ec_msw_gatednet_implementation_plan.md](../active/tv3_ec_msw_gatednet_implementation_plan.md) | EC-MSW P0 契约、E1/E1r 失败证据、E1d/E1d-SB 正式结论与 attachment 门 |
| [端到端波形动态门控组分反演框架与文献证据.md](../端到端波形动态门控组分反演框架与文献证据.md) | EC-MSW 算法框架与文献证据边界 |
| [active/tv3_ec_msw_structured_sequence_head_plan.md](../active/tv3_ec_msw_structured_sequence_head_plan.md) | E1d-SB 正式过门；E1r attachment 代码已落地；纯 TOF-L LS 仍禁止 |
| [references/tv3_identifiability_business_threshold_evidence.md](../references/tv3_identifiability_business_threshold_evidence.md) | P90 与 nuisance 门限的证据和适用边界 |
| [active/b7_repeated_split_ood_protocol_implementation_plan.md](../active/b7_repeated_split_ood_protocol_implementation_plan.md) | B7 协议 |
| [active/d2b_raw_dsp_implementation_plan.md](../active/d2b_raw_dsp_implementation_plan.md) | D2b / RawDSP |
| [foundation/adaptation_plan.md](../foundation/adaptation_plan.md) | 场景适配与契约 |
| [foundation/physics_references.md](../foundation/physics_references.md) | 物性常数 |
| [operations/server_training_guide.md](../operations/server_training_guide.md) | 服务器训练；§4.5 `waveform_preprocess` |
| [archive/completed/waveform_normalization_plan.md](../archive/completed/waveform_normalization_plan.md) | 三层归一化；§12 设备侧预处理 |
