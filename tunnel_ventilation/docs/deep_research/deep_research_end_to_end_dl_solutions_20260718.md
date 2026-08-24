# 掘进通风端到端 DL：表示修复可行，但 O₂ 窄窗突破必须先增加信息

## 摘要

本文基于 tv3 正式实验、可辨识性审计和经逐页核验的外部文献，区分两个经常被混为一谈的目标：一是修复现有端到端模型丢失已存在信息的问题，二是突破当前单频单向声学观测对 O₂ 的物理信息上限。项目证据表明，E1 先丢绝对峰位，E1r 补回峰位后仍缺少序列校准与 SNR；E1d 进一步证明，`cal_plus_corr_psr_snr` 能恢复 B1 非劣，而仅加 TOF-L 或闭式 LS 几乎没有增量。外部证据则一致提示，小样本逆问题更适合带强归纳偏置的可微测量层、训练期中间表示蒸馏和强基线残差，而不是更大的通用 raw-waveform backbone。由此，当前最值得验证的是“结构化 teacher-student 端到端模型”：推理只读取 raw waveform 与正式 slow 通道，训练期使用冻结 `e1d_sb` 作为 teacher，分阶段监督 peak、corr、PSR、SNR 和 phase-window 校准表示；通过冻结 Ridge parity 后，再接 OOF B7 型低容量 residual raw3 头。该路线可能解决 learned representation 的信息丢失并改善 OOD，但不能创造 Fisher null space 中不存在的信息。真正改善 0.8% O₂ 窄窗，需要多频色散或衰减、双向声学、直接 O₂ 通道等新增独立观测方向。

## 1. 研究问题与项目证据

本报告回答三个冻结问题：

- **RQ1**：E1 与 E1r 的失败究竟是优化失败、表示失败，还是观测不可辨识？
- **RQ2**：在约 4200 个训练序列和现有 raw waveform + slow 输入下，哪种端到端 DL 路线最可能恢复 B1/B7 已利用的信息？
- **RQ3**：怎样用分阶段消融证明改善来自正确机制，而不是数据泄漏、shortcut、随机种子或放宽门限？

项目正式证据来自[算法方向复核](deep_research_algorithm_ideas_20260717.md)、[项目记忆库](../archive/legacy/掘进通风项目记忆库.md)、[EC-MSW 实施计划](../archive/completed/tv3_ec_msw_gatednet_implementation_plan.md)和[端到端框架证据稿](../references/端到端波形动态门控组分反演框架与文献证据.md)。截至 2026-07-17，证据链如下：

| 层次      | 正式事实                                                                              | 对 RQ 的约束                    |
| ------- | --------------------------------------------------------------------------------- | --------------------------- |
| 信息源     | 单向 TOF 对 O₂、CO₂、T、L 的联合 Fisher 秩为 1；verdict=`information_source_upgrade_required` | 网络不能恢复观测 null space 中的信息    |
| 局部任务    | 0.8% O₂ bins 内 oracle、observed、B7 均为负 R²                                          | 全局 R²不能代表精细 O₂ 反演           |
| E1 帧表示  | peak MAE 约 71 samples；冻结 sequence Ridge O₂ R²=`-0.042/-0.027/-0.090`              | 过早卷积、池化丢失绝对到达位置             |
| E1r 帧表示 | peak MAE 约 0.037 sample，帧门通过；sequence Ridge O₂ R²=`0.016/0.033/0.001`             | 峰位必要但不充分，缺口在序列表示            |
| E1d 诊断  | 校准满栈约 `0.13-0.21`；加入 SNR 后达到 `0.393/0.453/0.369`                                  | 必须显式保留校准、相关质量与 SNR          |
| E2s-LS  | 相对 e1d_sb 的 O₂ ΔR²仅约 `+0.0005` 至 `+0.001`                                         | 解析 LS 没有带来实质新信息             |
| B7      | ID test 约 0.70；S-Y 约 0.434，S-L 约 0.700                                            | 强线性底座 + OOF 小残差有效，但 OOD 不统一 |

因此，RQ1 的初步判别不是三选一，而是分层结论：E1/E1r 首先是**表示充分性失败**；整体 O₂ 窄窗随后受**观测不可辨识性**约束；训练过程稳定，不能把主要失败归因于“训练没收敛”。

## 2. 检索与证据方法

检索按四个相互独立的视角展开：

1. 主流方法：raw waveform 可学习前端、位置编码、多模态融合、质量感知汇聚；
2. 反方证据：小样本验证偏差、shortcut、underspecification、逆问题不稳定性；
3. 邻域方法：DDSP、匹配滤波与时延估计、physics-informed learning、heteroscedastic UQ；
4. 直接邻居：超声多组分气体反演及其非唯一性。

工具链为 `smart-search deep` 规划、`exa-search` 发现候选、`fetch` 抓取原文或出版页。主 `search` provider 因区域策略返回 HTTP 403，因此没有使用其生成式摘要；正文主张只使用已抓取的论文摘要、正文或正式出版页。检索覆盖 2017-2026 年，并回溯必要的早期方法论文。最终正文纳入 22 篇已核验文献；题名或主张无法从原页确认的候选未进入报告。

| 证据类型              | 本报告中的用途           | 可信边界               |
| ----------------- | ----------------- | ------------------ |
| tv3 正式产物与冻结计划     | 根因、指标、门限和当前权限     | 对本项目最强；仍只代表当前仿真与协议 |
| 直接超声气体实验          | 证明另一物理系统可从波形预测多组分 | 频率、气体、样本量不同，不能迁移精度 |
| 信号处理与音频 benchmark | 支持结构化可微前端和位置保持    | 只能支持组件可行性          |
| OOD、UQ 与逆问题研究     | 支持验证门、风险边界和反证     | 多为跨领域方法学证据         |

## 3. 按失败层次建立方法分类

为了避免把架构名称当成根因，本报告按端到端链路中的失败层次分类：

| 分支        | 核心问题                   | 可改变什么                             | 不能改变什么             |
| --------- | ---------------------- | --------------------------------- | ------------------ |
| A. 信息层    | 前向观测是否近似单射             | 新频率、新传播方向、新传感器                    | 仅靠网络不能提高 Fisher 秩  |
| B. 表示层    | 峰位、校准、SNR 是否保留下来       | 结构化前端、teacher hint、显式坐标           | 不能创造未观测的 O₂ 差异     |
| C. 优化与头部  | 已保留信息是否被稳定利用           | staged training、OOF residual、小容量头 | 不能替代正确 split 与信息审计 |
| D. 泛化与可信度 | 是否依赖 nuisance shortcut | group holdout、干预、UQ、拒识            | UQ 不能修复均值不可辨识      |

这一分类的关键跨界工作是逆问题研究：Ongie 等指出，监督式 inverse mapping 对 forward operator 的变化或不确定性敏感 [1]；Gottschling 等进一步把 hallucination、稳定性和非唯一性联系到 forward operator 的非平凡 kernel [2]。二者与项目 Fisher 秩 1 的结论共同说明，表示修复与信息源升级必须分开验收。

## 4. 信息层：任何架构都不能绕过 observation collision

设组分为 `y`，nuisance 为 `nu`，传感器前向过程为 `h(y, nu)`。若存在两组状态满足：

```text
distance(h(y1, nu1), h(y2, nu2)) <= registered_measurement_noise
abs(O2(y1) - O2(y2)) > target_o2_tolerance
```

则任何确定性模型都不可能同时正确预测二者；模型至多输出训练分布条件均值、宽区间或拒识。Zhuang 等的超声气体实验同时展示了机会与边界：其 92,873 对 He-Ar-air 波形允许 1D/2D probabilistic CNN 学到多杂质信息，但论文仍明确指出三组分单 TOF 非唯一，预测会沿相似响应轨迹互相补偿 [3]。这与 tv3 并不矛盾，因为其频率、气体弛豫、采样密度和额外波形信息均不同。

由此产生两个不同的成功等级：

- **表示成功**：raw-waveform student 达到 B1 parity，说明没有再丢失已有 deployable 信息；
- **检测突破**：新增观测后，Fisher 有效秩、误差预算和 0.8% bins 同时过业务门。

没有第二项时，第一项不能写成“突破 O₂ 检测上限”。当前优先级仍应是先完成静止空气数字孪生的 observation-collision、Jacobian/Fisher 与误差预算；多频色散或衰减、双向声学和直接 O₂ 通道只有通过同一信息门后才进入模型实验。

## 5. 表示层：把已证实的充分统计量变成训练目标

### 5.1 可微 DSP 的价值是约束搜索空间

DDSP 将经典 DSP 元件嵌入自动微分，并表明强结构先验可以减少黑盒模型负担、保留可解释控制 [4]；LEAF 则把过滤、池化、压缩和归一化限制为小参数可学习前端，在多个音频分类任务上优于固定 mel 特征 [5]。两者共同支持“可学习但受物理结构约束的前端”，却不支持整体照搬：tv3 的主要信号是绝对到达时刻，任意 shift-invariant pooling 都可能消除目标信息。

更接近本项目的是时延估计研究。Cobos 等在显式 FS-GCC 表示上用 CNN 学习噪声与混响下的时延 [6]；Berg 等用 shift-equivariant 网络扩展 GCC-PHAT，在保留理想条件时延性质的同时改善恶劣环境误差 [7]。与通用 raw CNN 相比，这些工作把学习限制在“相关图去噪或残差”上，与 E1r 已成功的 train-only template 匹配滤波更一致。

显式坐标同样有方法学支持。CoordConv 研究表明，普通卷积并不总能高效学习绝对坐标变换，而添加固定坐标通道可直接表达平移依赖 [8]。但项目 E1r 已给出更强的本地反证：一个精确峰坐标仍不足以恢复 sequence parity。因此，本项目不应只加 `t/N` 或 position embedding 后重新训练同一 E1r。

### 5.2 首选：deployable teacher 的分阶段表示蒸馏

FitNets 表明，teacher 的中间表示可作为 hint 分阶段训练 student，而不必只蒸馏最终输出 [9]。对 tv3，更合理的 teacher 不是更大的神经网络，而是已正式过门的 `e1d_sb_cal_plus_corr_psr_snr_v1`。这样可以把 E1d 已定位的信息缺口直接转成监督信号，而 teacher 只在训练与审计中使用；推理仍只读取 raw waveform 与正式 slow 通道。

建议的 student 分三层：

```text
raw waveform
  -> frozen train-only template correlation map
  -> small shift-equivariant residual denoiser
  -> frame heads: peak distribution, corr peak, PSR, SNR
  -> explicit phase/time/L tokens + fixed phase-window statistics
  -> 213-D e1d_sb representation student
  -> frozen Ridge parity probe
  -> OOF baseline + low-capacity raw3 residual head
```

关键约束如下：

1. template 继续使用 train-only baseline median 并固化 digest；不允许让 composition loss 把模板漂移成第二套隐式真相；
2. peak head 以 sample coordinate 为统一量纲，输出每个采样点的概率分布或局部 softargmax；不得混用 seconds、normalized index 和 samples；
3. frame teacher 至少包含 `peak_index`、`corr_peak`、`PSR`、`SNR`；SNR 不能被 LS 或 learned uncertainty 替代；
4. sequence student 显式学习 phase-window 校准表示，而不是再次只做 `last/mean/max`；
5. slow 通道使用正式可部署集合，late fusion 起步；当前不启用 FiLM、attention 或 MoE；
6. offline RawDSP targets 只作为 auxiliary teacher，不进入部署输入，不使用 true TOF、true sound speed 或 true alpha。

这条路线的目标不是让 student 神奇地超过 teacher，而是先回答一个更严格的问题：**raw-waveform DL 能否在不读取部署期 RawDSP cache 的条件下，重建已知有效的 deployable representation？**

## 6. 优化与头部：先固定表示，再学习残差

### 6.1 不把所有损失一次性相加

Physics-informed learning 说明物理约束可在小数据下提供额外结构，但综述也强调需要标准 benchmark 与稳健框架 [10]。更直接的反证是 Wang 等发现，多项 physics loss 的收敛率失配会使 PINN 训练失败 [11]。因此，不建议从第一轮就联合优化 composition、peak、TOF、SNR、physics consistency、OOD consistency 和 uncertainty。

推荐四阶段训练：

| 阶段  | 训练参数             | 目标                                       | 必须通过的门                 |
| --- | ---------------- | ---------------------------------------- | ---------------------- |
| S0  | 无新模型             | 冻结 split、teacher、template、B1/B7 hash     | provenance 完整          |
| S1  | frame student    | 蒸馏 peak distribution、corr、PSR、SNR        | 三 split frame fidelity |
| S2  | sequence student | 蒸馏 213-D e1d_sb 与 phase-window 结构        | 冻结 Ridge 三组分 B1 parity |
| S3  | 小 residual head  | 学习 OOF baseline residual，直接输出 raw3       | B7 配对非劣与 OOD 门         |
| S4  | 单一附加机制           | 每次只试 nuisance pair 或 physics consistency | equal-input 配对增益       |

每一阶段只解冻必要层。S1 失败时修前端；S2 失败时修序列表示；S3 失败时接受 teacher parity 已恢复但没有新预测增量；不得跨门增加模型容量。

### 6.2 OOF 强基线残差比纯神经头更符合项目事实

端到端 student 通过 parity 后，最可行的预测头不是重新从零学习 raw3，而是复用 B7 已验证的不变量：

```text
y_hat_train = y_base_oof + delta_theta(z_student, optional_raw_residual)
y_hat_infer = y_base_full_train + delta_theta(z_student, optional_raw_residual)
```

训练 residual 必须使用 OOF base prediction，正式推理才使用 full-train base；`delta_theta` 零初始化或使用 64-64 级小网络。这样，线性主信号与校准信息不会再次被组分 loss 压掉，raw branch 只有在发现 teacher 未覆盖的信息时才偏离强基线。典型表格 benchmark 中，树模型仍能超过深度模型 [12]，而简单线性模型在多项时间序列预测 benchmark 中也超过了所比较的复杂 Transformer [13]；这些结果不能证明 B7 永远最优，但足以否定“更深即更适合”的先验。

## 7. 泛化与可信度：用干预识别 shortcut，而不是看一张 ID 表

Geirhos 等把 shortcut 定义为标准 benchmark 有效、挑战条件失效的决策规则 [14]。DomainBed 进一步表明，缺少 model-selection strategy 的 domain-generalization 方法是不完整的，严格实现的 ERM 在其 benchmark 中仍是强基线 [15]；group DRO 则需要明显正则化或 early stopping 才可能改善 worst-group generalization，直接套用到过参数网络可能失败 [16]。这些证据共同要求：先定义 train-only nuisance groups 与选择规则，再讨论 OOD loss。

本项目应构造两类可干预配对：

- 固定 mixture，改变 T、RH、P、L、jitter、SNR、device profile 或 calibration session；预测应保持稳定；
- 固定 nuisance，局部改变 O₂；预测应保持正确方向和局部斜率。

只有当 nuisance intervention 显著改变预测，或真实组分 intervention 不引起响应，才能判为 shortcut。单纯从 latent 读出 device id 不是充分淘汰证据。D'Amour 等关于 underspecification 的研究说明，多个 IID 表现等价的模型可能在 stress test 上显著分化 [17]；小样本 CV 还会产生很宽的误差条，fold 间标准误通常低估真实不确定性 [18]。因此，training seed 不能代替独立 selector，S-Y 与 S-L 也不能合并为一个 OOD 数字。

UQ 只承担风险暴露。Faithful heteroscedastic regression 指出，普通异方差 NLL 可能同时损害 mean 与 variance，需阻断 variance 路径对 mean accuracy 的不利影响 [19]；deep ensembles 在分类与回归 benchmark 中提供了较好的校准，并能对 OOD 样本表达更高不确定性 [20]。Tibshirani 等给出的 covariate-shift conformal 保证依赖已知或可由无标签目标域估计的 density ratio [21]，因此普通 split conformal 在未知 S-Y/S-L shift 下不能自动继承 nominal coverage。对 tv3，优先级仍是 train-only split conformal 基线和 risk-coverage，其次才是 faithful heteroscedastic head 或 ensemble。任何 UQ 方法都不能把宽区间写成精度提升，也不能在未知 shift 下宣称无条件 coverage。

## 8. 建议实验矩阵与硬停止条件

### 8.1 最小实验矩阵

| 实验   | 变化                                    | 回答的问题                        | 预期裁决             |
| ---- | ------------------------------------- | ---------------------------- | ---------------- |
| SD-0 | 冻结 E1r、e1d_sb、B1、B7                   | 正负对照能否精确复现                   | 不通过则停止           |
| SD-1 | frame teacher only                    | peak、corr、PSR、SNR 能否从 raw 重建 | 只看 fidelity，不看组分 |
| SD-2 | + fixed phase-window sequence student | 213-D teacher 能否被 student 表达 | 冻结 Ridge parity  |
| SD-3 | + OOF residual raw3 head              | DL 是否在强底座上提供增量               | 与 B7 配对          |
| SD-4 | + nuisance matched-pair consistency   | OOD 改善是否来自正确干预               | 一次只加该机制          |
| SD-5 | + physics consistency                 | 外推是否改善且非额外输入收益               | 等输入消融            |
| SD-6 | + UQ / reject                         | 风险是否可校准                      | 不作为 R²主线         |
| IS-0 | 多频、双向或直接 O₂ 新观测                       | 是否真正增加信息维度                   | 先 Fisher，后模型     |

SD-1 至 SD-3 是当前最值得实施的主实验。SD-4 至 SD-6 只有在前门通过后另立计划；现有 `e2_allowed=false` 不因本报告而改变。FiLM 的通用条件化能力 [22] 与 attentive statistics pooling 的异质帧加权能力 [23] 只保留为远期候选，不能绕过当前停止条件。

### 8.2 硬门

| 门               | 通过要求                                                                                 | 失败动作                                         |
| --------------- | ------------------------------------------------------------------------------------ | -------------------------------------------- |
| G0 信息门          | Fisher 有效秩、条件数、P90、nuisance 比例、拒绝率满足登记门                                              | `information_source_upgrade_required`，停止架构搜索 |
| G1 provenance 门 | outer split 以 `mixture_id` 为最小组；template、scaler、teacher、selector 只在 outer train 拟合   | 任一 overlap 或 hash 不符，整次作废                    |
| G2 frame 门      | val/test/extrapolation 全部通过现有 peak fidelity；SNR/PSR/corr teacher 误差按 train-only 阈值登记 | 只修前端                                         |
| G3 sequence 门   | 冻结 Ridge 在三组分、三 split 达到 B1 非劣；完整正对照复现                                               | 禁止训练新组分头                                     |
| G4 prediction 门 | 同 split 相对 B7 配对；ID、S-Y、S-L、窄窗与 worst-group 不退化                                      | 保留 B7，不替换默认头                                 |
| G5 shortcut 门   | matched-pair、nuisance swap、模态遮蔽、物理扰动稳定性通过                                            | 修数据与表示，不换更大架构                                |
| G6 UQ 门         | 独立 group calibration；coverage、宽度、risk-coverage、拒绝率同时登记                               | 不宣称可信部署                                      |

不得把 `mixture_id` 回退或重写为 `sequence_id`。所有整体 R²改善必须同时报告固定 0.8% bins 的 MAE、bias、P90 与 local slope；只改善整体高低档位时，结论应限定为 coarse monitoring。

## 9. 跨分支综合判断

文献与项目证据在一个并不直观的结论上收敛：**端到端并不等于去除 DSP，而是让从测量到输出的整条推理链可联合审计，并把必要的测量不变量保留在模型内部。** DDSP、LEAF 和时延估计研究支持结构化可微前端；FitNets 支持用中间 hint 训练 student；DomainBed、group DRO 与 underspecification 研究要求强基线和 stress test；逆问题理论则限定所有方法的上界。

这也解释了为什么原 EC-MSW 设想中的 FiLM、attention 和 MoE 目前不应启动。它们解决的是“已有可靠 token 如何融合”，而 E1/E1r 已证明 token 本身尚不充分。先用 teacher 把 `cal_plus_corr_psr_snr` 重建出来，既直接针对根因，也能给后续任何动态模块建立明确输入契约。若 student 已精确重建 teacher 却仍不能超过 B7，结论应是“端到端表示已修复，但当前 waveform 没有额外可利用信息”，而不是继续堆叠门控。

### 推荐优先级

1. **最高优先级：SD-1 + SD-2 表示蒸馏。** 这是修复 E1/E1r 失败机制的最直接实验。
2. **高优先级：SD-3 OOF baseline residual。** 这是当前最可能保持 B7 水平并检测 raw waveform 增量的头部。
3. **条件优先级：SD-4 matched-pair consistency。** 只用于明确登记的 nuisance，不做任意增强。
4. **可信度支线：SD-6 conformal / faithful heteroscedastic / ensemble。** 只改善风险表达。
5. **真正突破主线：IS-0 信息源升级。** 多频或直接通道先过 Fisher 与误差预算，再训练网络。

## 10. 开放问题与反证边界

1. 据本次检索，尚无研究在“200 kHz 单向超声 + NDIR CO₂ + TCS + O₂/N₂/CO₂ raw3 + 双 selector OOD”这一完整设置上验证结构化 teacher-student 方案；算法收益必须由 tv3 消融决定。
2. 当前仿真 waveform 主要是固定 pulse 的时移和标量衰减，student 可能只能复现 RawDSP，无法提供额外残差信息。这是可接受的 no-go 结果。
3. teacher 本身依赖 train-only baseline template。即使没有 label leakage，也必须在 device、session 和 calibration holdout 下检查 template shortcut。
4. teacher hint 会限制 student 的自由度。若 raw waveform 在新多频数据中包含 teacher 未建模的信息，应保留零初始化的小 residual branch，并用等容量、等预算消融判断其价值。
5. 时移不变对比学习、全波形 autoencoder 或大规模 masked modeling 可能优先学习固定 pulse、幅值和噪声，甚至主动抹掉 TOF；在没有真实无标签硬件波形前不应作为 P0。
6. Physics consistency 只能使用独立两侧：`c_measured` 来自 deployable corrected TOF 与 L，`c_from_composition` 来自预测 raw3、测得 T 和登记物性。若两侧来自同一预测量，它只是代数自洽。

## 11. 结论

**RQ1：失败机制是什么？** E1 是峰位与绝对坐标丢失；E1r 证明补回峰位仍不足，真正的表示缺口是 phase-aware 校准、相关质量与 SNR；稳定训练和早停结果不支持把主要失败归因于优化崩溃。更上层的 0.8% O₂ 窄窗失败则受单向观测秩亏约束。

**RQ2：最值得做的方法是什么？** 首选 deployable teacher 驱动的结构化端到端 student：相关图与显式坐标前端、frame-level peak/corr/PSR/SNR hint、phase-window 213-D sequence hint、冻结 Ridge parity probe，以及 OOF B7 型小 residual raw3 头。它比继续加 Transformer、FiLM、attention 或 MoE 更贴合项目根因。

**RQ3：怎样证明有效？** 按 G0 至 G6 串行过门：信息、provenance、frame fidelity、sequence parity、B7 配对、shortcut 干预、UQ calibration。达到 B1 parity 只证明表示修复；稳定超过 B7 且 OOD 与窄窗不退化，才是算法增量；只有新增信息源同时改善 Fisher 与业务误差门，才是精细 O₂ 检测突破。

本报告的贡献不是再列一组候选网络，而是把端到端问题重写为两个可证伪命题：**student 能否重建已知有效的 deployable statistics；新增观测能否提高 nuisance 边缘化后的有效信息秩。** 前者决定 DL 是否值得继续，后者决定 O₂ 目标是否在物理上可达。

## 参考文献

[1] Ongie G, Jalal A, Metzler CA, Baraniuk RG, Dimakis AG, Willett R, "Deep Learning Techniques for Inverse Problems in Imaging," IEEE Journal on Selected Areas in Information Theory, 2020.

[2] Gottschling NM, Antun V, Hansen AC, Adcock B, "The Troublesome Kernel: On Hallucinations, No Free Lunches, and the Accuracy-Stability Tradeoff in Inverse Problems," SIAM Review, 2025.

[3] Zhuang B, Gencturk B, Oberai AA, et al., "Impurity Gas Detection for SNF Canisters Using Probabilistic Deep Learning and Acoustic Sensing," Measurement Science and Technology, 2024.

[4] Engel J, Hantrakul L, Gu C, Roberts A, "DDSP: Differentiable Digital Signal Processing," ICLR, 2020.

[5] Zeghidour N, Teboul O, de Chaumont Quitry F, Tagliasacchi M, "LEAF: A Learnable Frontend for Audio Classification," ICLR, 2021.

[6] Cobos M, Antonacci F, Comanducci L, Sarti A, "Time Difference of Arrival Estimation from Frequency-Sliding Generalized Cross-Correlations Using Convolutional Neural Networks," ICASSP, 2020.

[7] Berg A, O'Connor M, Kenter T, "Extending GCC-PHAT Using Shift Equivariant Neural Networks," Interspeech, 2022.

[8] Liu R, Lehman J, Molino P, et al., "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution," NeurIPS, 2018.

[9] Romero A, Ballas N, Ebrahimi Kahou S, et al., "FitNets: Hints for Thin Deep Nets," ICLR, 2015.

[10] Karniadakis GE, Kevrekidis IG, Lu L, et al., "Physics-Informed Machine Learning," Nature Reviews Physics, 2021.

[11] Wang S, Yu X, Perdikaris P, "When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective," Journal of Computational Physics, 2022.

[12] Grinsztajn L, Oyallon E, Varoquaux G, "Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?" NeurIPS, 2022.

[13] Zeng A, Chen M, Zhang L, Xu Q, "Are Transformers Effective for Time Series Forecasting?" AAAI, 2023.

[14] Geirhos R, Jacobsen JH, Michaelis C, et al., "Shortcut Learning in Deep Neural Networks," Nature Machine Intelligence, 2020.

[15] Gulrajani I, Lopez-Paz D, "In Search of Lost Domain Generalization," ICLR, 2021.

[16] Sagawa S, Koh PW, Hashimoto TB, Liang P, "Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization," ICLR, 2020.

[17] D'Amour A, Heller K, Moldovan D, et al., "Underspecification Presents Challenges for Credibility in Modern Machine Learning," Journal of Machine Learning Research, 2022.

[18] Varoquaux G, "Cross-Validation Failure: Small Sample Sizes Lead to Large Error Bars," NeuroImage, 2018.

[19] Stirn A, Wessels H, Schertzer M, et al., "Faithful Heteroscedastic Regression with Neural Networks," AISTATS, 2023.

[20] Lakshminarayanan B, Pritzel A, Blundell C, "Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles," NeurIPS, 2017.

[21] Tibshirani RJ, Barber RF, Candes EJ, Ramdas A, "Conformal Prediction Under Covariate Shift," NeurIPS, 2019.

[22] Perez E, Strub F, de Vries H, Dumoulin V, Courville A, "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI, 2018.

[23] Okabe K, Koshinaka T, Shinoda K, "Attentive Statistics Pooling for Deep Speaker Embedding," Interspeech, 2018.
