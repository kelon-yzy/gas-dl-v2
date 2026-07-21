# Deep Research：端到端 DL 在 tv3 声学组分反演中的失败根因与可行解法（survey）

> 作用：针对 `deep_research_algorithm_ideas_20260717.md` 与 `掘进通风项目记忆库.md` 的正式结论栈，专门回答"端到端 DL 模型的问题该怎么解"。
> 日期：2026-07-18。定位为文献综述 + 逐 RQ 裁决，不改写任何 A 级正式 verdict。
> 证据分级：**A** = 本项目正式产物（记忆库/计划文档，已冻结）；**B** = 对象/频段/机制接近的已发表实验；**C** = 跨领域方法/预印本，只支持候选、不外推数值。跨领域论文的 R²/RMSE/加速比一律不迁移为 tv3 预期增益。

---

## 摘要

端到端波形网络在 tv3 上全线失败（O₂ R² 从 −0.2 到约 0.01，A 级），而两阶段 B7（RawDSP 手工特征 → OOF Ridge + 残差 MLP）达到 O₂ test R²≈0.70。本综述的核心判断是：**这不是"网络容量不足"，而是三层结构性错配的叠加，且第三层决定了天花板。** (1) 表示层——平移不变的 learned encoder/pooling 破坏了绝对峰位 TOF；(2) 聚合层——`last/mean/max`/attentive pooling 在函数类上表达不出跨 L-sweep 的集合内回归斜率 `c=L/TOF`；(3) 学习机制层——在 Fisher 秩亏 + 中等样本（约 4200）+ 高共线的数据机制下，端到端相对精心设计的两阶段特征管线本就缺乏文献支持的优势。文献可以修复前两层（可微 TOF、可微集合回归 / 声速反演 declarative node），但修完后端到端的上限仍是 B7 的声学物理墙；第三层由多条独立证据汇聚（tabular DL 长期不敌树模型；特征驱动 TabPFN 在小样本高变异生理信号上胜过端到端 Transformer）指向"两阶段是此类问题的合理默认，而非权宜"。因此端到端在 tv3 的合理定位是**可部署单模型工程闭环 + New Setting 可辨识性审计**，不是 R² 突破。真正的 R² 上限只能由信息源升级（多频弛豫谱 / 直接 O₂ 通道）打破，而这与当前单频固定 pulse 仿真机制不兼容。

---

## 1. 研究问题（冻结）

- **RQ1**：如何构造端到端可微网络，在函数类上原生表达"对一组 (TOF, L) 做集合内线性回归求斜率"这类计算？各方法族在小样本、高共线下是否稳定，失败模式为何？
- **RQ2**：端到端 raw 声学信号→物理量回归中，如何在网络内保留/利用 TOF 的平移敏感性？哪些做法真正优于"先 DSP 提 TOF 再回归"的两阶段？
- **RQ3**：物理信息/物理引导端到端学习能否在不增加信息源的前提下改善端到端，而不仅是做正则？
- **RQ4**：信息源受限（秩亏 + 不可辨识窄窗）时，端到端 DL 相对两阶段管线是否有文献支持的实质优势？端到端在"New Setting + 可辨识性审计"论文中的合理定位与新颖性边界？

---

## 2. 方法学与分类轴

按"失败发生在链路的哪一层"做 MECE 分类，每层对应一个可修与不可修的边界：

| 分类层                 | 对应 RQ | 该层能修什么      | 该层修完后的残余上限            |
| ------------------- | ----- | ----------- | --------------------- |
| §3 表示层：TOF 位置保留     | RQ2   | 帧级绝对峰位      | 帧保真是必要非充分（A 级 E1r 已证） |
| §4 聚合层：集合内回归斜率      | RQ1   | 序列聚合的函数类    | 上限逼近 B1≈0.4，不破 0.70   |
| §5 学习机制层：端到端 vs 两阶段 | RQ4   | 优化/归纳偏置错配   | 两阶段在此机制下是合理默认         |
| §6 物理约束层：一致性正则      | RQ3   | 外推/物理合规性    | 不增 Fisher 秩，不创造信息     |
| §7 信息源层（横切约束）       | 全部    | 唯一能破 rank-1 | 与当前单频仿真不兼容            |

---

## 3. 表示层：把 TOF 位置留在网络里（RQ2）

**Claim**：可微 TOF 定位是成熟且必要的一步，但它只是把两阶段管线的第一步"内化"，不改变最终精度上限——本项目 A 级证据已经独立证明了这一点。

平移不变是这层的病根。语音领域 SincNet（Ravanelli & Bengio 2018，B 级）用参数化 sinc 带通滤波器约束首层，使 raw waveform CNN 收敛更快、比标准 CNN 更好，其价值在于给高维原始输入一个物理可解释的窄带先验；但它解决的是"学到有意义的滤波器"，不是"保住绝对到达时间"，且需要输入含可分频谱内容——tv3 单载波 200 kHz 固定 pulse 恰好缺这个内容（A 级：`alpha_lambda_max_o2=0.0`，组分只改到达时间与标量衰减）。相较之下，可微定位算子更贴 TOF 需求：soft-argmax 及其改进 sampling-argmax（Li et al. 2021，C 级）通过对输出分布施加隐式约束缓解概率图形状不受控的问题，是把"峰位坐标"变成可微输出的标准工具。在超声这一具体对象上，深度 TOF 估计已有多条 B 级证据：Jia et al.（2025，GAF 图像编码 + 迁移学习）、Wang et al.（2024，ToFD + 自注意力，声称相对 Hilbert 峰-峰法降低不确定度）、Mimura et al.（2024，2D CNN 在声速 CT 中估 TOF 优于传统信号处理）——三者一致表明 DL 能高精度估 TOF，但**都作用在单波形或多路 TOF 标量上，本质仍是"DL 版的 TOF 提取器"，不是端到端组分回归**。

这与 tv3 的 A 级证据完全吻合：E1r 用 train-only 模板峰位坐标锚点，把帧级 peak MAE 从 E1 的 71 sample 压到 0.037 sample（frame fidelity 通过），但序列层 B1 parity 仍失败（O₂ R²≈0.01 vs B1≈0.4）。**结论是残酷但清晰的：帧保真是必要条件，不是序列充分条件。** 表示层的所有文献方法（SincNet / soft-argmax / matched-filter）都只能解决 E1r 已经解决的那一步。

**表 1｜表示层方法对 tv3 的适配（结论：可微 TOF 只复现 E1r 已达成的帧保真，不触及序列缺口）**

| 方法                         | 机制          | 证据级 | 对 tv3 的净增量     | 与当前仿真兼容性     |
| -------------------------- | ----------- | --- | -------------- | ------------ |
| SincNet 参数化滤波器             | 首层带通先验      | B   | 低：单载波缺可分频谱     | 不兼容（无频率相关形状） |
| soft-/sampling-argmax      | 可微峰位坐标      | C   | 低：E1r 坐标锚点已达同效 | 兼容但冗余        |
| 深度 TOF 估计（GAF/自注意力/2D-CNN） | DL 版 TOF 提取 | B   | 低：仍是两阶段第一步     | 兼容但不端到端      |

**Go/No-go（RQ2）**：

- **No-go**：以"可微 TOF/SincNet 修复端到端"为 R² 突破路线。A 级 E1r 已证帧保真不充分；文献无一给出"端到端 raw→组分"优于两阶段 TOF 管线的对象内证据。
- **Go（受限）**：仅当目标是"单模型可部署"工程闭环时，可微 TOF 定位算子可作为把 E1r 坐标锚点并入同一网络的手段——判据是数值语义与 train-only 模板一致、不引入 exact-simulator 泄漏，且与 B7 配对非劣。

---

## 4. 聚合层：把"集合内回归斜率"变成可微算子（RQ1）

**Claim**：让端到端网络原生表达"对 (TOF, L) 集合做 WLS 求斜率"，在方法上完全可行且有直接类比先例（可微声速反演 = declarative least-squares node）；但本项目 A 级证据显示这条斜率的信息已被更廉价的标量特征吸收，因而该算子的边际价值上界≈0——方法成立，收益不成立。

这层有两条互斥的技术路线，差别在于**是否显式嵌入解析结构**。

**路线一：纯表达力（Deep Sets / Set Transformer）。** Deep Sets（Zaheer et al. 2017，C 级）刻画了置换不变函数的通用形式 `ρ(Σφ(x))`，Set Transformer（Lee et al. 2018，C 级）用注意力 + 诱导点把集合内交互建模到线性复杂度。二者理论上是通用集合函数逼近器，原则上"能"逼近斜率回归；但它们把斜率解交给数据去学，在约 4200 样本 + 高共线下，这正是标准 pooling 失败的同一优化难题的翻版——A 级 E1r 的 attentive pooling 已经在这个函数类里失败过。置换不变性综述（Kimura et al. 2024，C 级）进一步指出 Deep Sets 的行为对聚合函数选择高度敏感，可泛化为拟算术均值——这说明"选对聚合子"本身就是难点，而不是自动获得。

**路线二：解析结构内嵌（可微优化层）。** 这是与 E2s 根因最贴合的方法族。OptNet（Amos & Kolter 2017，C 级，1281 引）把 QP 作为网络层并用 KKT 隐式微分回传；Differentiable Convex Optimization Layers（Agrawal et al. 2019，C 级，892 引）把可微对象推广到 disciplined convex programs 并落地 CVXPY/PyTorch；Deep Declarative Networks（Gould et al. 2019/TPAMI 2022，C 级）用隐函数定理对"由优化问题隐式定义的层"回传，明确指出该框架**囊括最小二乘型节点**。关键是加权最小二乘声速反演本身是标量线性回归，有闭式解、PyTorch 直接可微，根本不需要 OptNet 的 QP 求解器（这与 A 级 E2s 计划的判断一致：OptNet/DDN 只作理论背书）。最直接的类比证据有两条：DecDTW（Xu et al. 2023，C 级）把时间对齐作为 declarative bi-level 层端到端学习，输出真实最优对齐路径而非 soft 近似——这正是"序列内需要一个优化解作为聚合输出"的同构范式；PMaF（Xu et al. 2023，C 级）实现了"least squares on sphere（LESS）"作为 DDN 层，证明最小二乘可作可微聚合节点。electrical 领域的 Ding et al.（2025，C 级）把鲁棒状态估计作为 optimization-embedded 层，报告相对经典端到端与 PINN 在物理一致性上更优——支持"把解析反演当层"能改善物理合规性。

**contradiction（必须直面）**：方法族说这条路可行，但 A 级证据说收益≈0。tv3 的 E2s-LS 正式结论是：additive SNR 加权闭式 LS 声速反演头**过了 B1 门**，但相对已有 compact 特征（`e1d_sb_cal_plus_corr_psr_snr_v1`）的 O₂ ΔR² 仅 `+0.0005~0.001`，判为**不晋升**。条件分析：E1d 诊断（A 级）已定位，TOF-L 校准 alone 补不回 O₂，真正恢复 B1 非劣的是加入 `ultrasonic_snr_db`；也就是说，B1/B7 从 L-sweep 拿到的那点 O₂ 信息，被"校准 TOF + SNR + PSR"这组标量特征几乎完全捕获，显式的 WLS 斜率算子没有额外可榨的量。**因此可微优化层能让一个真·端到端网络"内含"WLS，但它的边际信息价值被 compact 特征集的覆盖度封顶，而后者已接近满覆盖。** DEQ（Bai et al. 2019，C 级）作为隐式定深网络主打大规模序列/内存效率，与 tv3 的小样本标量回归诉求正交，无适配理由。

**表 2｜聚合层方法（结论：解析内嵌路线在方法上成立且有 DecDTW/LESS 类比，但 A 级 E2s-LS 证明其对 tv3 边际增益≈0）**

| 方法族                         | 是否内嵌解析结构  | 最贴 tv3 的类比       | 证据级 | 小样本+共线下风险                      | 对 O₂ R² 的预期       |
| --------------------------- | --------- | ---------------- | --- | ------------------------------ | ----------------- |
| Deep Sets / Set Transformer | 否         | 通用集合逼近           | C   | 高：与 E1r attentive pooling 同类失败 | 不看好（重蹈优化难题）       |
| OptNet / CvxpyLayers        | 是（QP/凸）   | 过重，WLS 无需 QP     | C   | 中：求解器/条件数                      | 方法可行、收益未证         |
| DDN + 闭式 WLS / LESS         | 是（最小二乘节点） | DecDTW、PMaF-LESS | C   | 低：闭式可微                         | **方法成立，A 级证增益≈0** |
| DEQ                         | 是（不动点）    | 无                | C   | 不适配                            | 不适用               |

**Go/No-go（RQ1）**：

- **No-go**：以任何集合回归/可微优化层作为 O₂ R² 突破路线。A 级 E2s-LS 已把该族最贴切成员的增益钉在 `+0.001`。
- **Go（受限，工程价值）**：把闭式 WLS 声速反演作为 declarative 聚合层，嵌入一个从 raw→raw3 的**单一可微网络**，唯一合法目标是"用一个端到端模型证明性地包含 B1 的 L-sweep 回归 + SNR 特征"，服务可部署与可解释，而非刷分。判据：与 B7 配对在 R/L/S-Y/S-L 全非劣、raw3 负值/非闭包语义预先定义且不静默 clamp、增益归因用等输入消融隔离（防止收益来自额外输入 `c_measured` 而非约束本身）。

---

## 5. 学习机制层：端到端 vs 两阶段的机制性证据（RQ4）

**Claim**：多条互相独立的证据汇聚指向同一结论——在中等样本 + 强共线 + 表格化标量的机制下，两阶段特征管线是有原则的默认选择，端到端的劣势是结构性的而非调参不足。这是本综述最强的收敛点。

Grinsztajn et al.（2022，C 级，NeurIPS，>1700 引）在 45 个数据集上系统证明：中等规模（约 10K 样本）表格数据上树模型仍稳定优于 DL，并把原因归结为三条 NN 归纳偏置缺陷——对无信息特征不鲁棒、破坏数据朝向、难学不规则函数。tv3 的 observed/RawDSP 空间恰是"864~1008 维强共线标量表格"（A 级），三条缺陷全中：R5 默认 MLP 直接 train 失败（A 级 val O₂ −0.18），R7 ExtraTrees 显著过拟合（train 0.997 / val 0.45），而线性 Ridge + 低容量残差（B7）反而稳。更直接的对象内类比来自 Jonker et al.（2025，C 级）：在小样本、高个体间变异的多模态生理信号疼痛分级上，特征驱动的 TabPFN（手工特征 + 集成 + 堆叠）显著胜过端到端 Transformer-with-features（60.06% vs 54.02%），作者的结论——"数据有限 + 高变异时，稳健特征工程 + 强归纳偏置模型优于复杂端到端架构"——几乎逐字对应 tv3 的 A 级观察（TabPFN observed O₂ R²≈0.66 为上限探针，端到端波形网络全负）。

这也给 A 级里 R5 失败的正确解读加了外部支撑：不是"输入没标准化/O₂ 没加权"（A 级已澄清这些说法不成立），而是高维共线表格 + 端到端优化的机制性困难。spectral bias 只是候选解释之一：Tancik et al.（2020，C 级）用 NTK 证明标准 MLP 在低维域难学高频、Fourier 特征可缓解——但这支持的是"坐标类回归"的病理，tv3 的病理更偏"共线表格 + 秩亏"，不能把 spectral bias 当作 R5 失败的已证根因（与 A 级 §7"不能过度解释"一致）。

**表 3｜端到端 vs 两阶段的机制证据（结论：三条独立来源汇聚支持"两阶段是此机制下的合理默认"）**

| 证据                  | 对象        | 样本规模   | 结论                          | 对 tv3 的映射                  |
| ------------------- | --------- | ------ | --------------------------- | -------------------------- |
| Grinsztajn 2022 (C) | 45 表格数据集  | ~10K   | 树 > DL；NN 三缺陷               | RawDSP 表格；R5 失败/R7 过拟合     |
| Jonker 2025 (C)     | 多模态生理信号   | 小样本高变异 | 特征+TabPFN > 端到端 Transformer | TabPFN observed 上限 > 端到端波形 |
| tv3 A 级             | 声学 RawDSP | ~4200  | B7 两阶段 ≫ 端到端                | 直接同构                       |

**Go/No-go（RQ4）**：

- **No-go**：期待端到端在 tv3 当前机制下"迟早超过两阶段"。无对象内、无近域证据支持这一期待，反例（Grinsztajn/Jonker）方向一致。
- **Go（论文定位）**：端到端的可发表价值是 **New Setting + 可辨识性审计**，即"在一个此前无直接重合的传感组合（NDIR 仅 CO₂ + 超声 + TCS → O₂/N₂/CO₂）上，用冻结正式协议与双 OOD 门证明端到端为何不优于两阶段、边界在哪"。判据：主张严格分级（A/B/C），不把跨域端到端成功迁移为 tv3 预期；端到端作为对照臂，而非声称的胜者。

---

## 6. 物理约束层：一致性正则的边界（RQ3）

**Claim**：物理一致性能改善外推与合规性，但在信息论上不增 Fisher 秩，因此只能是稳健性支线，不能声称突破物理上限——这与 A 级 B1 裁决一致。

PINN/physics-guided 在逆问题上的成功大多集中在"PDE 结构强、观测可反演"的场景（如逆热传导、稀疏传感流场重建，均为 C 级邻域证据），其增益来自把已知物理当作额外约束缩小解空间，而非创造新可观测。Ding et al.（2025，C 级）的 optimization-embedded 状态估计层是这层最有用的形态：把物理反演作为可微层，改善的是**物理一致性**（更低测量残差），不是信息维度。对 tv3，A 级 B1 计划已给出正确形式：物理一致性 loss 必须用两个独立来源（`c_measured` 来自冻结 RawDSP 的 corrected TOF/L，`c_from_composition` 来自预测 raw3 + 测得 T + 登记物性），否则只是代数自洽；且明确"该 loss 可能改善外推，但不会增加 Fisher 秩"。可微前向声学 + 反演在当前仿真下还有一个致命不兼容：当前生成机制是固定 transducer pulse 的时移 + 标量衰减、O₂ 弛豫项=0（A 级），可微前向模型能建模的频率相关物理在数据里根本不存在，网络只会学噪声或已有标量的重复表达。

**Go/No-go（RQ3）**：

- **No-go**：以 PINN/物理反演突破 O₂ 上限。不增 Fisher 秩，且与单频仿真机制不兼容。
- **Go（受限）**：物理一致性 loss 作为低成本外推/合规性 ablation，权重只在 train 内层选，报告 B7 配对 + 窄窗 + S-Y/S-L，用等输入消融隔离归因（与 A 级 B1 裁决完全一致）。

---

## 7. 信息源层：唯一能破 rank-1 的机制（横切约束）

**Claim**：多频声速色散/衰减谱是文献中唯一被反复验证、能把单频 rank-1 提升为多维敏感度的机制，但它与当前单频仿真不兼容，属于"先改仿真再谈端到端"。

一条清晰的 B 级证据链：Zhu-Shi Ming et al.（2008）用 CO 多弛豫模型 + 声速联合反演 CO 浓度；Zhang et al.（2020）证明捕捉频率相关速度色散的拐点可区分**摩尔质量相同**的混合气（如 CO₂-N₂），仅靠声速把 CO₂ 最大绝对误差从 3.8% 降到 0.2%；Liu et al.（2021）用双频合成弛豫特征分析 CO₂/CH₄/air；Liu et al.（2021）进一步提取分子内比热作为组分内禀量；Shen et al.（2023）用 DBR 光纤激光宽带声学传感重建 CO₂ 弛豫吸收谱（误差 <1.32%）；Iglesias Hernandez et al.（2022）用未涂层 CMUT 同时测声速与衰减区分含 N₂ 混合气。这些一致表明"多频 = 新敏感度维度"，但**没有一篇证明 tv3 的 O₂/N₂ 在常压 200 kHz、当前噪声与 0.18~0.28 m 声程下可分**（A 级 §RQ 明确此风险）。关键约束：这层要动的是仿真物理表示（引入频率相关传递函数），不是网络——在表示门（Fisher 有效秩上升 + 误差预算改善）通过前，任何多频端到端网络都无对象。

**Go/No-go（信息源）**：与 A 级 `deep_research_algorithm_ideas_20260717.md` §4.1 A1-G0 完全一致——先做多频 Jacobian/Fisher 审计，表示门过了才谈端到端。端到端 DL 在此层无独立动作。

---

## 8. 跨层综合讨论

三层修复的净效果可以叠加追踪，但天花板由最外层决定：

- 修表示层（§3）→ 得到帧保真，A 级 E1r 证明这不足以恢复序列 O₂；
- 再修聚合层（§4）→ 得到 B1 非劣（≈0.4），A 级 E2s-LS 证明显式 WLS 相对 compact 特征增益≈0；
- 换更强学习机制（§5）→ 端到端仍不敌两阶段，且这是机制性的（Grinsztajn/Jonker 汇聚）；
- 加物理正则（§6）→ 改善外推/合规，不增秩；
- 唯一破 0.70 的是信息源层（§7），而它把问题推回"先改仿真"。

因此端到端 DL 在 tv3 的"问题"本质是**被三层错配 + 一堵物理墙共同锁死**：文献能逐层拆掉前两层的技术障碍，但拆完后落在 B7 的声学上限上，第三层的独立证据又说明这个上限不是端到端能翻越的。**端到端的真实机会不在"打败 B7"，而在"用一个可微单模型复现 B7 的可部署形态，并把整个失败链做成一篇 New Setting + 可辨识性审计的论文"。**

---

## 9. 开放问题

1. 一个把可微 TOF（§3）+ 闭式 WLS declarative 聚合（§4）串成的**真·端到端 raw→raw3 单网络**，能否在 R/L/S-Y/S-L 全协议上做到与 B7 配对严格非劣？（A 级只跑过 additive LS 消融，未跑过完整端到端单模型 holdout。）
2. 多频仿真表示门（§7 A1-G0）通过后，端到端多频编码器相对多频 RawDSP + Ridge 是否有非劣以上增益？（当前无对象，属 P2。）
3. split conformal + risk-coverage（安全性方向）在端到端单模型上的覆盖率是否与两阶段一致？（可信度层，独立于 R²。）

---

## 10. 结论（逐 RQ 回答）

- **RQ1**：可行方法族已明确——首选"闭式加权最小二乘声速反演作为 Deep Declarative 聚合节点"（DecDTW/PMaF-LESS 为直接类比，OptNet/CvxpyLayers 为过重的理论背书），纯 Deep Sets/Set Transformer 会重蹈 attentive pooling 的优化失败，DEQ 不适配。**但 A 级 E2s-LS 已证该族对 tv3 的边际增益≈0.001。裁决：方法成立、R² 收益不成立；仅作单模型可部署工程闭环的 GO。**
- **RQ2**：可微 TOF（soft-/sampling-argmax、matched-filter、SincNet、深度 TOF 估计）在文献中成熟，但全部只解决"帧级峰位"，即 A 级 E1r 已解决的那一步；无一给出端到端 raw→组分优于两阶段 TOF 管线的对象内证据。**裁决：No-go 作为突破路线；Go 仅作为把 E1r 锚点并入单网络的手段。**
- **RQ3**：物理一致性/PINN/可微反演能改善外推与物理合规，但不增 Fisher 秩，且与当前单频、O₂ 弛豫=0 的仿真机制不兼容。**裁决：Go 仅作低成本稳健性 ablation（严格双来源、等输入消融）；No-go 作为上限突破。**
- **RQ4**：中等样本 + 强共线 + 表格标量机制下，两阶段是有原则的默认——Grinsztajn（树>DL）、Jonker（特征+TabPFN>端到端）与 tv3 A 级三方汇聚。**裁决：端到端的合理定位是 New Setting + 可辨识性审计的对照臂，而非声称的性能胜者；新颖性在"传感组合 + 冻结协议 + 双 OOD 门 + 分级归因"，不在把 SincNet/GP/可微优化层包装成新算法。**

一句话：端到端 DL 在 tv3 值得做的是"**可微单模型 + 审计式论文**"，不是"**更强回归头刷 O₂**"。真正的 R² 天花板由信息源决定，需先改仿真物理，不是改网络。

---

## 11. 参考文献（均经两步存在性 + 摘要一致性核验，标注证据级）

**方法族（C 级）**

1. Zaheer M, Kottur S, Ravanbakhsh S, Póczos B, Salakhutdinov R, Smola A. Deep Sets. *NeurIPS* 2017. arXiv:1703.06114.
2. Lee J, Lee Y, Kim J, Kosiorek AR, Choi S, Teh YW. Set Transformer. *ICML* 2019. arXiv:1810.00825.
3. Amos B, Kolter JZ. OptNet: Differentiable Optimization as a Layer in Neural Networks. *ICML* 2017. arXiv:1703.00443.
4. Agrawal A, Amos B, Barratt S, Boyd S, Diamond S, Kolter JZ. Differentiable Convex Optimization Layers. *NeurIPS* 2019. arXiv:1910.12430.
5. Gould S, Hartley R, Campbell D. Deep Declarative Networks. *IEEE TPAMI* 2022 (arXiv:1909.04866, 2019). doi:10.1109/TPAMI.2021.3059462.
6. Gould S, Campbell D, Ben-Shabat I, Koneputugodage CH, Xu Z. Exploiting Problem Structure in Deep Declarative Networks: Two Case Studies (robust vector pooling, optimal transport). arXiv:2202.12404, 2022.
7. Xu M, Garg S, Milford M, Gould S. DecDTW: Deep Declarative Dynamic Time Warping for End-to-End Learning of Alignment Paths. *ICLR* 2023. arXiv:2303.10778.
8. Xu Z, Wang H, Liu Y, Gould S. PMaF: Deep Declarative Layers for Principal Matrix Features (LESS/IED). arXiv:2306.14759, 2023.
9. Bai S, Kolter JZ, Koltun V. Deep Equilibrium Models. *NeurIPS* 2019. arXiv:1909.01377.
10. Kimura M, Shimizu R, Hirakawa Y, Goto R, Saito Y. On permutation-invariant neural networks. arXiv:2403.17410, 2024.
11. Tancik M, et al. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. *NeurIPS* 2020. arXiv:2006.10739.

**表示层 / 可微 TOF（B–C 级）**
12. Ravanelli M, Bengio Y. Speaker Recognition from Raw Waveform with SincNet. *IEEE SLT* 2018. arXiv:1808.00158.
13. Li J, Chen T, Shi R, Lou Y, Li YL, Lu C. Localization with Sampling-Argmax. *NeurIPS* 2021. arXiv:2110.08825.
14. Jia H, Jiang J, Li F, Xiao G. High-Precision Ultrasonic Flight Time Prediction Based on GAF Image Encoding and Deep Learning. *IEEE TIM* 2025. doi:10.1109/TIM.2025.3633367.
15. Wang Z, Shi F, Ding J, Song X. Ultrasonic Rough Crack Characterization Using ToFD With Self-Attention Neural Network. *IEEE TUFFC* 2024. doi:10.1109/TUFFC.2024.3459619.
16. Mimura Y, et al. Image Reconstruction in Ultrasonic Speed-of-Sound CT Using TOF Estimated by 2D CNN. *Technologies* 2024. doi:10.3390/technologies12080129.

**学习机制层（C 级）**
17. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS* 2022. arXiv:2207.08815.
18. Jonker RAA, et al. When Features Matter More than Sequence: A Case for Tabular In-Context Learning in Pain Classification. 2025. doi:10.1145/3747327.3764785.

**物理约束层（C 级）**
19. Ding Y, et al. Power System Robust State Estimation As a Layer: An Optimization-embedded End-to-end Learning Approach. arXiv:2511.22836, 2025.

**信息源层 / 声学气体传感（B 级）**
20. Zhu-Shi M, Wang S, Wang S-t, Xia D-h. An acoustic gas concentration measurement algorithm for carbon monoxide in mixtures based on molecular multi-relaxation model. *Acta Phys. Sin.* 2008;57:5749. doi:10.7498/aps.57.5749.
21. Zhang X, Wang S, Zhu M. Locating the inflection point of frequency-dependent velocity dispersion by acoustic relaxation to identify gas mixtures. *Meas. Sci. Technol.* 2020. doi:10.1088/1361-6501/ab9375.
22. Liu T, Hu Y, Zhang X, Zhu M. Acoustic analysis of gas compositions based on molecular relaxation features. *Results Phys.* 2021;25:104304. doi:10.1016/j.rinp.2021.104304.
23. Liu T, Wang S, Zhu M. A versatile acoustic gas sensing method via extracting intrinsic molecular internal specific heat. *Phys. Lett. A* 2021. doi:10.1016/j.physleta.2021.127349.
24. Shen K, Yuan J, Li M, Wen X, Lu H. Measurement of the Acoustic Relaxation Absorption Spectrum of CO₂ Using a DBR Fiber Laser. *Sensors* 2023;23:4740. doi:10.3390/s23104740.
25. Iglesias Hernandez L, et al. Gas discrimination by simultaneous sound velocity and attenuation measurements using uncoated CMUTs. *Sci. Rep.* 2022;12:744. doi:10.1038/s41598-021-04689-4.

---

## 12. 与项目不变量对照

| 不变量                                       | 状态                       |
| ----------------------------------------- | ------------------------ |
| raw3 输出、`out_dim=3`，不回填 N₂                | 遵守：单模型端到端建议仍输出 raw3      |
| 不使用 gas_head/ILR/ALR/闭包残差头                | 遵守                       |
| B7 保持默认部署头                                | 遵守：端到端仅作对照臂/工程闭环         |
| `e2_allowed=false`（不开 FiLM/attention/MoE） | 遵守：本文不建议启动 E2            |
| 不把 full B1 包装成端到端改进                       | 遵守：单模型须与 B7 配对非劣并等输入消融归因 |
| 上限逼近 B1≈0.4，不破 0.70                       | 遵守：突破归于信息源层              |
| 未检索到直接重合 ≠ 新颖性证明                          | 遵守：投稿前仍需系统检索复核           |
