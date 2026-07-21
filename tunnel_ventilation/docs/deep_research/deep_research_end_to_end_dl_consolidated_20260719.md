# Deep Research 合并综述：tv3 端到端 DL 失败机制、修复路线与 O₂ 突破方向（consolidated）

> 合并来源：以 [`deep_research_end_to_end_dl_20260718.md`](deep_research_end_to_end_dl_20260718.md)（五层失败分类框架）为基础，整合 [`deep_research_algorithm_ideas_20260717.md`](deep_research_algorithm_ideas_20260717.md)（O₂ 突破方向分诊）与 [`deep_research_end_to_end_dl_solutions_20260718.md`](deep_research_end_to_end_dl_solutions_20260718.md)（teacher-student 实施方案与硬门）。三份原文保留不改。
> 日期：2026-07-19。合并原则：不改写任何 A 级正式 verdict；重叠内容取最严格版本；来源间张力在 §1 显式裁决。
> 证据分级沿用三文档统一约定：**A** = 本项目正式产物（记忆库/冻结计划）；**B** = 对象/频段/机制接近的已发表实验；**C** = 跨领域方法/预印本，只支持候选，不外推数值。跨领域论文的 R²/RMSE/加速比一律不迁移为 tv3 预期增益。

---

## 1. 三份来源的分工与合并裁决

| 来源文档                         | 定位                       | 本合并稿采纳的内容                                                                    |
| ---------------------------- | ------------------------ | ---------------------------------------------------------------------------- |
| `algorithm_ideas_20260717`   | 全局路线分诊（不限端到端）            | 主线 A1/A2/A3 信息源升级、支线 B1–B5、暂缓/拒绝表、B7 解释边界、P0–P2 分阶段、验收指标、最近邻新颖性表             |
| `end_to_end_dl_20260718`（基础） | 端到端 DL 失败的分层综述 + 逐 RQ 裁决 | 五层 MECE 分类、各层 Go/No-go、方法族适配表、"三层错配 + 物理墙"总判断                                |
| `solutions_20260718`         | 端到端修复的可执行方案              | teacher-student 结构化蒸馏架构、S0–S4 分阶段训练、SD-0~SD-6 实验矩阵、G0–G6 硬门、shortcut/UQ 验证文献 |

三份文档在核心事实上完全一致（Fisher 秩 1、B7≈0.70、E1/E1r/E1d/E2s-LS 证据链、唯一突破在信息源）。需要显式裁决的张力有三处：

1. **单模型端到端的实现路径**。基础稿 §4 的 Go 是"闭式 WLS 声速反演作 declarative 聚合层"；solutions 稿的首选是"蒸馏冻结 `e1d_sb_cal_plus_corr_psr_snr_v1` teacher"。**裁决：teacher-student 蒸馏为首选实现**——A 级 E2s-LS 已证显式 WLS 相对 compact 特征 ΔR² 仅 +0.0005~0.001，而 E1d 证明真正恢复 B1 非劣的是"校准 + corr + PSR + SNR"整组特征；蒸馏路线监督的正是这组已验证充分统计量，WLS declarative 层降级为可选组件（方法成立、非必要，见 §7）。
2. **TabPFN 的双重角色**。algorithm_ideas 拒绝"TabPFN 部署堆叠"，基础稿又引 TabPFN 作学习机制层证据。**裁决：两者不矛盾**——TabPFN observed O₂ R²≈0.66 是上限探针与"特征驱动 > 端到端"的机制证据（§8），但其依赖 simulator-derived observed 特征、不可部署，作为正式部署堆叠仍为拒绝。
3. **端到端的论文角色**。algorithm_ideas 定位"New Setting + information-source audit"；基础稿定位"New Setting + 可辨识性审计的对照臂"。**裁决：二者合一**——论文主张是传感组合 + 冻结协议 + 双 OOD 门 + 分级归因，端到端模型作对照臂与可部署单模型演示，不作性能胜者。

---

## 2. 摘要（合并后总判断）

端到端波形网络在 tv3 上全线失败（O₂ R² 从 −0.2 到约 0.01，A 级），而两阶段 B7（RawDSP 手工特征 → OOF Ridge + 残差 MLP）达到 O₂ test R²≈0.70。这不是"网络容量不足"，而是**三层结构性错配 + 一堵物理墙**：

1. **表示层**：平移不变的 learned encoder/pooling 破坏绝对峰位 TOF（E1 peak MAE≈71 sample）；E1r 补回峰位（0.037 sample）后序列仍失败，真正缺口是 phase-aware 校准、相关质量与 SNR（E1d 定位）。
2. **聚合层**：`last/mean/max`/attentive pooling 表达不出跨 L-sweep 的集合内回归斜率 `c=L/TOF`；但 A 级 E2s-LS 证明显式 WLS 算子的边际增益≈0，该信息已被 compact 标量特征覆盖。
3. **学习机制层**：约 4200 样本 + 高共线标量表格机制下，两阶段特征管线是有原则的默认（Grinsztajn 树>DL、Jonker 特征+TabPFN>端到端、tv3 A 级三方汇聚），端到端劣势是结构性的。
4. **物理墙**：单向 TOF 对 O₂/CO₂/T/L 联合 Fisher 秩为 1，正式 verdict `information_source_upgrade_required`；0.8% O₂ bins 内 oracle/observed/B7 均负 R²。物理一致性正则不增 Fisher 秩；唯一能破 rank-1 的是信息源升级（多频弛豫谱 / 直接 O₂ 通道），而这与当前单频固定 pulse 仿真不兼容，须先过 A1-G0 表示门。

由此，路线分为三类：

| 层级   | 路线                                     | 裁决                                                  |
| ---- | -------------------------------------- | --------------------------------------------------- |
| 突破主线 | A1 多频色散/衰减谱、A3 直接 O₂ 通道（IS-0/A1-G0 先行） | 唯一能破 0.70 与窄窗的机制；先审计后建模                             |
| 工程主线 | teacher-student 结构化蒸馏单模型（SD-1→SD-3）    | 修复表示丢失、给出可部署端到端形态；目标是 B1 parity + B7 配对非劣，不承诺 R² 突破 |
| 支线   | 物理一致性、特征消融压缩、独立目标残差、conformal UQ/拒识    | 稳健性与边界刻画；不得包装为信息源突破                                 |

端到端 DL 在 tv3 值得做的是"**可微单模型 + 审计式论文**"，不是"更强回归头刷 O₂"。真正的 R² 天花板由信息源决定，先改仿真物理，不是改网络。

---

## 3. 研究问题（冻结，合并去重）

| RQ  | 问题                                               | 来源                  | 对应判据                      |
| --- | ------------------------------------------------ | ------------------- | ------------------------- |
| RQ1 | E1/E1r 的失败是优化失败、表示失败还是观测不可辨识？                    | solutions           | 分层证据链（§4）                 |
| RQ2 | 端到端网络如何在函数类上保留 TOF 平移敏感性并表达集合内回归？哪些做法真正优于两阶段？    | 基础稿 RQ1+RQ2         | 帧保真门、序列 parity 门          |
| RQ3 | 不增加信息源时，哪些模型/正则改动能改善 OOD 或可信度？                   | ideas RQ2 + 基础稿 RQ3 | R/L/S-Y/S-L 完整协议、覆盖率、拒识风险 |
| RQ4 | 中等样本 + 秩亏 + 高共线机制下，端到端相对两阶段是否有文献支持的实质优势？论文定位是什么？ | 基础稿 RQ4             | 机制证据汇聚、新颖性边界              |
| RQ5 | 新声学观测（多频/双向）或直接 O₂ 通道能否增加独立信息维度？                 | ideas RQ1+RQ4       | Fisher 秩、条件数、误差预算、窄窗 P90  |

---

## 4. A 级正式证据链（合并）

| 层次      | 正式事实                                                                                 | 对路线的约束                      |
| ------- | ------------------------------------------------------------------------------------ | --------------------------- |
| 信息源     | 单向 TOF 对 O₂、CO₂、T、L 联合 Fisher 秩 = 1；verdict=`information_source_upgrade_required`    | 任何网络不能恢复观测 null space 中的信息  |
| 局部任务    | 0.8% O₂ bins 内 oracle、observed、B7 均负 R²                                              | 全局 R² 只支持高低档位辨识，不代表精细反演     |
| E1 帧表示  | peak MAE≈71 sample；冻结 seq Ridge O₂ R²=`-0.042/-0.027/-0.090`                         | 过早卷积/池化丢失绝对到达位置             |
| E1r 帧表示 | peak MAE≈0.037 sample，帧门通过；seq Ridge O₂=`0.016/0.033/0.001`                          | 峰位必要不充分，缺口在序列表示             |
| E1d 诊断  | 校准满栈约 `0.13–0.21`；加入 SNR 后 `0.393/0.453/0.369`（B1 非劣）                                | 必须显式保留校准、相关质量与 SNR          |
| E2s-LS  | 相对 `e1d_sb` 的 O₂ ΔR² 仅 `+0.0005~+0.001`，正式不晋升                                        | 解析 LS 不带来实质新信息              |
| B7      | ID test≈0.70；S-Y≈0.434；S-L≈0.700                                                     | 强线性底座 + OOF 小残差有效，但 OOD 不统一 |
| 学习机制    | R5 默认 MLP val O₂ −0.18；R7 ExtraTrees train 0.997/val 0.45；TabPFN observed≈0.66（上限探针） | 高维共线表格上线性+低容量残差最稳           |
| 仿真机制    | waveform = 固定 transducer pulse 时移 + `exp(-alpha*L)` 标量衰减；`alpha_lambda_max_o2=0.0`   | 波形形状不含 O₂ 频率相关信息，形状特征无物理来源  |

RQ1 的答案由此分层：E1/E1r 首先是**表示充分性失败**；O₂ 窄窗随后受**观测不可辨识性**约束；训练稳定、早停正常，不能把主要失败归因于优化崩溃。

---

## 5. 失败层次 MECE 分类（主框架）

| 分类层                       | 对应 RQ   | 该层能修什么         | 修完后的残余上限                            |
| ------------------------- | ------- | -------------- | ----------------------------------- |
| §6 表示层：TOF 位置 + 校准/SNR 保留 | RQ1/RQ2 | 帧级绝对峰位、序列充分统计量 | 帧保真必要非充分（E1r 已证）；蒸馏上限 = teacher 覆盖度 |
| §7 聚合层：集合内回归斜率            | RQ2     | 序列聚合的函数类       | 上限逼近 B1≈0.4，不破 0.70（E2s-LS 已证增益≈0）  |
| §8 学习机制层：端到端 vs 两阶段       | RQ4     | 优化/归纳偏置错配      | 两阶段是此机制下的合理默认                       |
| §9 物理约束层：一致性正则            | RQ3     | 外推/物理合规性       | 不增 Fisher 秩，不创造信息                   |
| §10 信息源层（横切约束）            | RQ5     | 唯一能破 rank-1    | 与当前单频仿真不兼容，先过表示门                    |

逆问题理论为整个分类划定上界：Ongie et al. 2020（C 级）指出监督式 inverse mapping 对 forward operator 变化敏感；Gottschling et al. 2025（C 级）把 hallucination、稳定性与非唯一性联系到 forward operator 的非平凡 kernel。观测碰撞的形式化判据（solutions 稿 §4）：若两组 `(y, nu)` 状态在登记测量噪声内不可分而 O₂ 差超容差，任何确定性模型至多输出条件均值、宽区间或拒识。由此区分两个成功等级——**表示成功**（student 达到 B1 parity，不再丢失已有 deployable 信息）与**检测突破**（新增观测后 Fisher 有效秩、误差预算和 0.8% bins 同时过业务门）；没有第二项时，第一项不能写成"突破 O₂ 检测上限"。

---

## 6. 表示层：把已证实的充分统计量留在网络里

**Claim**：可微 TOF 定位成熟且必要，但只解决 E1r 已解决的帧保真那一步；真正的表示缺口是 E1d 定位的"校准 + corr + PSR + SNR"，修复手段是结构化前端 + teacher 蒸馏，而非更大的通用 raw-waveform backbone。

### 6.1 文献证据

平移不变是这层的病灶。SincNet（Ravanelli & Bengio 2018，B 级）证明参数化带通先验对 raw waveform 有效，但它解决的是"学到有意义的滤波器"而非"保住绝对到达时间"，且要求输入含可分频谱内容——tv3 单载波 200 kHz 固定 pulse 恰好没有（A 级 `alpha_lambda_max_o2=0.0`）。soft-/sampling-argmax（Li et al. 2021，C 级）是可微峰位坐标的标准工具；深度 TOF 估计（Jia 2025 GAF、Wang 2024 ToFD、Mimura 2024 2D-CNN，均 B 级）一致表明 DL 能高精度估 TOF，但都作用于单波形/多路标量，本质仍是"DL 版 TOF 提取器"。

结构化可微前端有更贴的方向：DDSP（Engel 2020，C 级）与 LEAF（Zeghidour 2021，C 级）支持"可学习但受物理结构约束的前端"；时延估计研究（Cobos 2020 FS-GCC + CNN、Berg 2022 shift-equivariant GCC-PHAT，均 C 级）把学习限制在"相关图去噪/残差"上，与 E1r 已成功的 train-only template 匹配滤波一致。CoordConv（Liu 2018，C 级）支持显式坐标通道，但 E1r 已给出更强的本地反证：一个精确峰坐标不足以恢复 sequence parity，因此不应只加 position embedding 重训同一 E1r。

**表 6-1｜表示层方法适配（结论：可微 TOF 只复现 E1r 已达成的帧保真；结构化前端 + 蒸馏才对准序列缺口）**

| 方法                             | 机制                 | 证据级 | 对 tv3 的净增量                               | 兼容性                |
| ------------------------------ | ------------------ | --- | ---------------------------------------- | ------------------ |
| SincNet 参数化滤波器                 | 首层带通先验             | B   | 低：单载波缺可分频谱                               | 不兼容（无频率相关形状）       |
| soft-/sampling-argmax          | 可微峰位坐标             | C   | 低：E1r 坐标锚点已达同效                           | 兼容但冗余              |
| 深度 TOF 估计（GAF/自注意力/2D-CNN）     | DL 版 TOF 提取        | B   | 低：仍是两阶段第一步                               | 兼容但不端到端            |
| FS-GCC / shift-equivariant GCC | 相关图上受限学习           | C   | 中：与 train-only template 同构，可作 student 前端 | 兼容                 |
| CoordConv / position embedding | 显式坐标               | C   | 低：E1r 本地反证                               | 兼容但不充分             |
| FitNets 式中间表示蒸馏                | teacher hint 分阶段训练 | C   | **高：直接把 E1d 缺口转成监督信号**                   | 兼容（teacher 仅训练期使用） |

### 6.2 首选实现：deployable teacher 的分阶段表示蒸馏（工程主线）

teacher 不是更大的神经网络，而是已正式过门的 `e1d_sb_cal_plus_corr_psr_snr_v1`（213-D）。teacher 只在训练与审计中使用；推理只读取 raw waveform 与正式 slow 通道。student 结构：

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

关键约束（冻结）：

1. template 继续使用 train-only baseline median 并固化 digest；不允许 composition loss 把模板漂移成第二套隐式真相；
2. peak head 以 sample coordinate 为统一量纲；不混用 seconds、normalized index 和 samples；
3. frame teacher 至少包含 `peak_index`、`corr_peak`、`PSR`、`SNR`；SNR 不能被 LS 或 learned uncertainty 替代；
4. sequence student 显式学习 phase-window 校准表示，不再只做 `last/mean/max`；
5. slow 通道使用正式可部署集合，late fusion 起步；不启用 FiLM、attention、MoE（`e2_allowed=false` 不变）;
6. offline RawDSP targets 只作 auxiliary teacher，不进入部署输入，不使用 true TOF / true sound speed / true alpha。

这条路线回答的问题是：**raw-waveform DL 能否在不读取部署期 RawDSP cache 的条件下，重建已知有效的 deployable representation？** 目标不是超过 teacher。若 student 精确重建 teacher 仍不能超过 B7，结论是"端到端表示已修复，但当前 waveform 没有额外可利用信息"——这是可接受的 no-go 结果，不是继续堆叠门控的理由。

**Go/No-go（表示层）**：

- **No-go**：以可微 TOF/SincNet/position embedding 为 R² 突破路线；以对比学习/masked modeling 做波形预训练（可能主动抹掉 TOF，无真实无标签硬件波形前不作 P0）。
- **Go**：teacher-student 蒸馏（SD-1/SD-2），判据为 G2 帧门 + G3 序列 parity 门（§13）。

---

## 7. 聚合层：集合内回归斜率的可微算子

**Claim**：让网络原生表达"对 (TOF, L) 集合做 WLS 求斜率"在方法上完全可行（declarative least-squares node，DecDTW/PMaF-LESS 为直接类比），但 A 级 E2s-LS 证明这条斜率的信息已被"校准 TOF + SNR + PSR"标量特征几乎完全捕获——**方法成立，收益不成立**。

两条互斥路线：

- **纯表达力（Deep Sets / Set Transformer，C 级）**：理论上能逼近斜率回归，但把解交给数据学，在约 4200 样本 + 高共线下重蹈 E1r attentive pooling 的优化失败；聚合函数选择本身高度敏感（Kimura 2024，C 级）。
- **解析内嵌（可微优化层，C 级）**：OptNet/CvxpyLayers/DDN 给出理论背书；WLS 声速反演是标量线性回归，有闭式解、直接可微，无需 QP 求解器。DecDTW（对齐作为 declarative 层）与 PMaF-LESS（最小二乘作 DDN 节点）是最贴的同构范式；Ding 2025 的 optimization-embedded 状态估计层支持"解析反演作层"改善物理合规。

**contradiction（必须直面）**：方法族说可行，A 级说增益≈0。E2s-LS 正式结论：additive SNR 加权闭式 LS 过了 B1 门，但相对 `e1d_sb` ΔR² 仅 +0.0005~0.001，不晋升。E1d 已定位 TOF-L 校准 alone 补不回 O₂，真正恢复 B1 非劣靠 `ultrasonic_snr_db`。**可微优化层能让端到端网络"内含"WLS，但边际信息价值被 compact 特征覆盖度封顶。**

**表 7-1｜聚合层方法（结论：解析内嵌方法成立，A 级证增益≈0）**

| 方法族                         | 内嵌解析结构    | 最贴类比             | 证据级 | 风险                       | 对 O₂ R² 预期        |
| --------------------------- | --------- | ---------------- | --- | ------------------------ | ----------------- |
| Deep Sets / Set Transformer | 否         | 通用集合逼近           | C   | 高：同 attentive pooling 失败 | 不看好               |
| OptNet / CvxpyLayers        | 是（QP/凸）   | 过重，WLS 无需 QP     | C   | 中：求解器/条件数                | 方法可行、收益未证         |
| DDN + 闭式 WLS / LESS         | 是（最小二乘节点） | DecDTW、PMaF-LESS | C   | 低：闭式可微                   | **方法成立，A 级证增益≈0** |
| DEQ                         | 是（不动点）    | 无                | C   | 不适配                      | 不适用               |

**Go/No-go（聚合层）**：

- **No-go**：任何集合回归/可微优化层作为 O₂ 突破路线；"直接声速解析反演头"维持暂缓（无新观测时只是重排同一信息）。
- **Go（可选组件）**：若 §6.2 蒸馏 student 需要显式表达 L-sweep 结构，闭式 WLS 可作 declarative 聚合节点并入单网络；判据为与 B7 配对全协议非劣、raw3 负值/非闭包语义预定义不静默 clamp、等输入消融隔离归因（防止收益来自额外输入 `c_measured`）。相对蒸馏路线，此组件非必要。

---

## 8. 学习机制层：端到端 vs 两阶段的机制性证据

**Claim**：多条独立证据汇聚——中等样本 + 强共线 + 表格化标量机制下，两阶段特征管线是有原则的默认，端到端劣势是结构性的。这是全综述最强收敛点。

- Grinsztajn et al. 2022（C 级，>1700 引）：约 10K 样本表格数据上树模型稳定优于 DL，归因于 NN 三条归纳偏置缺陷（对无信息特征不鲁棒、破坏数据朝向、难学不规则函数）。tv3 的 observed/RawDSP 是 864~1008 维强共线标量表格，三条全中：R5 默认 MLP 失败、R7 ExtraTrees 过拟合、线性 Ridge + 低容量残差（B7）反而稳。
- Jonker et al. 2025（C 级）：小样本高变异生理信号上，特征驱动 TabPFN 胜过端到端 Transformer（60.06% vs 54.02%），结论几乎逐字对应 tv3 观察（TabPFN observed≈0.66 上限探针 vs 端到端波形全负）。
- Zeng et al. 2023（C 级）：简单线性模型在时序预测 benchmark 超过复杂 Transformer——不证明 B7 永远最优，但足以否定"更深即更适合"的先验。
- spectral bias（Tancik 2020，C 级）只是候选解释，不能当作 R5 失败的已证成因（与 A 级"不能过度解释"一致）。

**表 8-1｜B7 结果的正确理论解释边界（自 algorithm_ideas §7）**

| 观察                       | 可以支持的解释                       | 不能支持的过度解释                   |
| ------------------------ | ----------------------------- | --------------------------- |
| Ridge 稳定                 | 主信号强线性/低阶结构，L2 对共线有效          | 最优岭惩罚一定为零或负                 |
| 默认 MLP 失败、R5-T 成功        | 目标尺度是已验证主因                    | Spectral Bias 证明 MLP 先追高频噪声 |
| B7 优于 B1                 | OOF 线性基底 + 低容量非线性 residual 有效 | 等价于某特定非线性 kernel ridge      |
| TabPFN observed 成功       | 当前 observed 契约存在可学习非线性        | 非线性来自真实硬件物理，或可直接部署          |
| C1 grouped bottleneck 失败 | 早期物理分组压缩破坏所需交互                | 所有物理引导/低参数/分组方法都无效          |

**端到端 student 的预测头因此复用 B7 不变量**（solutions §6.2）：

```text
y_hat_train = y_base_oof + delta_theta(z_student, optional_raw_residual)
y_hat_infer = y_base_full_train + delta_theta(z_student, optional_raw_residual)
```

训练 residual 用 OOF base prediction，正式推理用 full-train base；`delta_theta` 零初始化或 64-64 小网络，不扩容。

**Go/No-go（学习机制层）**：

- **No-go**：期待端到端在当前机制下"迟早超过两阶段"；Deep Ridge"一行替换"（无激活多层仍是线性，原方法依赖 anchor graph 专用优化）；GAN-FixMatch 无限仿真数据（标签已知，丢弃标签无成本优势，也不缩小 sim-to-real gap）；TabPFN 部署堆叠（observed 特征不可部署 + 第二套推理契约）。
- **Go（论文定位）**：端到端作 New Setting + 可辨识性审计的对照臂；新颖性在"传感组合 + 冻结协议 + 双 OOD 门 + 分级归因"，不在包装新算法。

---

## 9. 物理约束层：一致性正则的边界

**Claim**：物理一致性能改善外推与合规性，但不增 Fisher 秩，只能是稳健性支线。

可验证形式必须用两个独立来源（A 级 B1 计划）：

```text
L_phys = MSE(c_measured, c_from_composition(y_pred, T_measured))
```

`c_measured` 来自冻结 RawDSP 的 corrected TOF 与 L；`c_from_composition` 由预测 raw3、测得 T 和登记物性计算。两侧同源则只是代数自洽。边界（冻结）：

- 可能改善外推，不增加 Fisher 秩；
- 权重只在 train 内层选，不用 test/OOD 调参；
- 必须报告 B7 配对 + 窄窗 + S-Y/S-L，不得只报 ID 均值；
- 收益须用等输入消融隔离归因（防止来自额外输入 `c_measured`）；
- raw3 负值/非闭包语义预先定义，禁止静默 clamp、重归一化、回填 N₂；无法给出可微兼容定义则不启动。

PINN 的多项 loss 收敛率失配可使训练失败（Wang et al. 2022 NTK，C 级）——这也是 §12 分阶段训练"不把所有损失一次性相加"的依据。可微前向声学在当前仿真下有致命不兼容：固定 pulse 时移 + 标量衰减、O₂ 弛豫=0，频率相关物理在数据里不存在，网络只会学噪声或已有标量的重复表达。

**Go/No-go**：突破上限 No-go；低成本外推/合规 ablation（SD-5，严格双来源 + 等输入消融）Go。

---

## 10. 信息源层：唯一能破 rank-1 的机制（突破主线）

### 10.1 A1 多频声速色散与衰减谱

机制依据（B 级证据链）：Zhu-Shi 2008（CO 多弛豫联合反演）、Zhang 2020（色散拐点区分摩尔质量相同的 CO₂-N₂，误差 3.8%→0.2%）、Liu 2021×2（双频合成弛豫特征、分子内比热内禀量）、Shen 2023（DBR 光纤重建 CO₂ 弛豫谱，误差 <1.32%）、Iglesias Hernandez 2022（CMUT 同测声速+衰减区分含 N₂ 混合气）。一致表明"多频 = 新敏感度维度"，但**没有一篇证明 O₂/N₂ 在常压 200 kHz、当前噪声与 0.18~0.28 m 声程下可分**。

当前缺口：200 kHz 仿真只在中心频率算一次衰减、同一 pulse 时移缩放，不能用于证明多频价值。**先扩展物理表示，不是把现有单频波形送进更复杂网络。**

**决定性实验 A1-G0（= IS-0 = G0 信息门）**：

1. 从文献/可追踪物理模型登记候选频率，不按模型表现挑频率；
2. 对每个频率生成声速、衰减及对 O₂/CO₂/T/L/RH 的 Jacobian；
3. 比较单频、双频、小频组的 Fisher 秩、条件数、nuisance 边缘化预算；
4. 用与 v1 一致的 P90=`0.4 vol%`、单 nuisance=`50%`、拒绝率=`5%` 门；
5. 表示门通过后才构建多频 RawDSP 和 B1/B7 对照。

Go：有效秩增加且误差预算明确改善 → 进入冻结 builder 与模型实验。No-go：灵敏度向量仍近共线或改善低于测量噪声 → 停止多频网络与超参搜索。

### 10.2 A2 当前单频波形形状特征 — 拒绝作为 P0

`waveforms.py` 固定 `transducer_response_pulse(spec)`、`acoustic_physics.py` 固定 `alpha_lambda_max_o2=0.0`、`raw_dsp_features.py` 已含 peak amplitude/SNR/PSR/correlation peak/peak width。此生成机制下归一化包络形状或 coda 比值没有新的 O₂ 物理来源。只有 A1 引入频率依赖传递函数或取得真实宽带波形后重新开放；届时先做单特征条件互信息 + train-only ablation，不直接训练 CNN。

### 10.3 A3 直接 O₂ 传感通道

| 路线                     | 可行性      | 当前定位                        |
| ---------------------- | -------- | --------------------------- |
| 760 nm VCSEL/DFB TDLAS | 高，硬件当前暂停 | 最成熟直接方案；恢复实验条件后优先           |
| 760 nm PAS/光热声学        | 中        | 降低长光程需求，但仍需激光/谐振腔/温控/声学抗扰验证 |
| Raman/FERS             | 物理可行，成本高 | 不符当前资源优先级，长期参考              |

TDLAS/PAS 应独立为硬件融合/系统论文，不与小样本算法、GP、多频声学同篇混写贡献。

---

## 11. 支线软件实验统一映射（B1–B5 ↔ SD-4~SD-6）

| algorithm_ideas 支线    | solutions 实验 | 内容                                                                                             | 裁决                                  |
| --------------------- | ------------ | ---------------------------------------------------------------------------------------------- | ----------------------------------- |
| B1 声速物理一致性正则          | SD-5         | §9 双来源 loss，等输入消融                                                                              | 条件接受，低成本 ablation                   |
| B2 train-only 特征消融与压缩 | —（独立）        | group ablation → stability selection/nested CV → 等维随机对照；selector/scaler 每 outer split 重拟合      | 接受为工程诊断/降成本；新颖性低                    |
| B3 独立目标 residual MLP  | —（独立）        | 三个 single-target MLP 学 OOF residual，仍组合为 raw3                                                  | 接受为诊断；无 gradient cosine 证据不再声称多任务冲突 |
| B4 校准不确定性与拒识          | SD-6         | split conformal → deep ensemble/heteroscedastic → 必要时 DKL/SVGP；报告 coverage/宽度/分组/risk-coverage | 接受为安全性方向，不作 R² 方向                   |
| B5 AdaCap / RC-Mixup  | —（二级候选）      | 仅 B1–B4 完成且形成明确残余机制后；与同参数量/同预算/同 OOF 输入的 B7 对照                                                 | 二级候选                                |
| —（nuisance 一致性）       | SD-4         | matched-pair consistency，只用于明确登记的 nuisance                                                     | 条件优先级，一次只加一个机制                      |

UQ 边界（合并）：Kendall 同方差权重只产生任务级全局 σ，不产生逐点区间，且可能下调 O₂ loss——降级为 loss ablation，不当 UQ。普通异方差 NLL 可能同时损害 mean 与 variance（Stirn 2023 faithful heteroscedastic，C 级）；deep ensembles 校准较好（Lakshminarayanan 2017，C 级）；covariate-shift conformal 的保证依赖可估计的 density ratio（Tibshirani 2019，C 级），普通 split conformal 在未知 S-Y/S-L shift 下不自动继承 nominal coverage。任何 UQ 都不能把宽区间写成精度提升。

---

## 12. 泛化与 shortcut：用干预识别，不看一张 ID 表

Geirhos 2020（C 级）定义 shortcut 为"标准 benchmark 有效、挑战条件失效"；DomainBed（Gulrajani 2021，C 级）表明缺少 model-selection strategy 的 DG 方法不完整、严格 ERM 仍是强基线；group DRO（Sagawa 2020，C 级）需要明显正则化/early stopping 才可能改善 worst-group。要求：先定义 train-only nuisance groups 与选择规则，再讨论 OOD loss。

两类可干预配对：

- 固定 mixture，改变 T/RH/P/L/jitter/SNR/device profile/calibration session → 预测应稳定；
- 固定 nuisance，局部改变 O₂ → 预测应保持正确方向与局部斜率。

只有 nuisance intervention 显著改变预测、或真实组分 intervention 不引起响应，才判 shortcut；从 latent 读出 device id 不是充分淘汰证据。underspecification（D'Amour 2022，C 级）与小样本 CV 宽误差条（Varoquaux 2018，C 级）要求：training seed 不能代替独立 selector，S-Y 与 S-L 不合并为一个 OOD 数字。teacher 依赖 train-only template，即使无 label leakage 也必须在 device/session/calibration holdout 下检查 template shortcut。

分阶段训练（不把所有损失一次性相加，依据 §9 PINN 失配证据）：

| 阶段  | 训练参数             | 目标                                       | 必须通过的门                 |
| --- | ---------------- | ---------------------------------------- | ---------------------- |
| S0  | 无新模型             | 冻结 split、teacher、template、B1/B7 hash     | provenance 完整          |
| S1  | frame student    | 蒸馏 peak distribution、corr、PSR、SNR        | 三 split frame fidelity |
| S2  | sequence student | 蒸馏 213-D e1d_sb 与 phase-window 结构        | 冻结 Ridge 三组分 B1 parity |
| S3  | 小 residual head  | 学 OOF baseline residual，直接输出 raw3        | B7 配对非劣与 OOD 门         |
| S4  | 单一附加机制           | 每次只试 nuisance pair 或 physics consistency | equal-input 配对增益       |

每阶段只解冻必要层。S1 失败修前端；S2 失败修序列表示；S3 失败接受"teacher parity 已恢复但无新预测增量"；不得跨门增加模型容量。

---

## 13. 统一执行计划与硬门

### 13.1 实验矩阵（合并 P0–P2 与 SD-0~SD-6）

| 阶段            | 实验                                                                       | 内容                                          | 回答的问题                                                  | 停止条件                          |
| ------------- | ------------------------------------------------------------------------ | ------------------------------------------- | ------------------------------------------------------ | ----------------------------- |
| P0（信息预审，不训模型） | IS-0 / A1-G0                                                             | 多频 Jacobian/Fisher/误差预算 + 硬件可用频段登记          | 新观测是否真正增加信息维度                                          | 近共线或改善低于噪声则停止；理论可分但硬件不可测则不进模型 |
| P1（工程主线）      | SD-0                                                                     | 冻结 E1r、e1d_sb、B1、B7 正负对照复现                  | provenance 能否精确复现                                      | 不通过则停止                        |
| P1            | SD-1                                                                     | frame teacher 蒸馏                            | peak/corr/PSR/SNR 能否从 raw 重建                           | 只看 fidelity，不看组分              |
| P1            | SD-2                                                                     | + phase-window sequence student             | 213-D teacher 能否被 student 表达                           | 冻结 Ridge parity               |
| P1            | SD-3                                                                     | + OOF residual raw3 head                    | DL 是否在强底座上提供增量                                         | 与 B7 配对                       |
| P1（支线，可并行）    | B2/B3/B4                                                                 | 特征压缩诊断 / 独立目标 residual / conformal 拒识       | 冗余、梯度冲突、风险表达                                           | 不得包装成信息源突破                    |
| P1（条件）        | SD-4 / SD-5(B1)                                                          | nuisance matched-pair / physics consistency | OOD 改善是否来自正确干预                                         | 一次只加一个机制，等输入消融                |
| P1（条件）        | SD-6 / B5                                                                | UQ 拒识 / AdaCap                              | 风险可校准性 / 残余非线性                                         | 不作 R² 主线                      |
| P2（表示/硬件升级）   | 多频 RawDSP + B1 parity + B7 非劣 + 独立参数 holdout；TDLAS/PAS 复核；sim-to-real 审计 | 突破是否真实                                      | A1-G0 未过不启动；learned waveform encoder 只有新表示含频率相关信息后重新立项 |                               |

所有实验保持 B7 默认头，完整 R/L/S-Y/S-L × split seed × training seed 配对协议；单一 val 改善不构成通过。

### 13.2 硬门 G0–G6

| 门               | 通过要求                                                                               | 失败动作                                         |
| --------------- | ---------------------------------------------------------------------------------- | -------------------------------------------- |
| G0 信息门          | Fisher 有效秩、条件数、P90、nuisance 比例、拒绝率满足登记门                                            | `information_source_upgrade_required`，停止架构搜索 |
| G1 provenance 门 | outer split 以 `mixture_id` 为最小组；template/scaler/teacher/selector 只在 outer train 拟合 | 任一 overlap 或 hash 不符，整次作废                    |
| G2 frame 门      | val/test/extrapolation 全部通过 peak fidelity；SNR/PSR/corr teacher 误差按 train-only 阈值登记 | 只修前端                                         |
| G3 sequence 门   | 冻结 Ridge 三组分、三 split 达到 B1 非劣；正对照复现                                                | 禁止训练新组分头                                     |
| G4 prediction 门 | 同 split 相对 B7 配对；ID、S-Y、S-L、窄窗与 worst-group 不退化                                    | 保留 B7，不替换默认头                                 |
| G5 shortcut 门   | matched-pair、nuisance swap、模态遮蔽、物理扰动稳定性通过                                          | 修数据与表示，不换更大架构                                |
| G6 UQ 门         | 独立 group calibration；coverage、宽度、risk-coverage、拒绝率同时登记                             | 不宣称可信部署                                      |

不得把 `mixture_id` 回退为 `sequence_id`。所有整体 R² 改善必须同时报告固定 0.8% bins 的 MAE、bias、P90 与 local slope；只改善整体高低档位时，结论限定为 coarse monitoring。

### 13.3 统一验收指标

- **信息层**：Fisher 有效秩与条件数；O₂ 对 nuisance 的局部灵敏度比；0.8% bins 的 P90/MAE/局部斜率；拒绝率与不可用区域。
- **模型层**：O₂ R²/MAE 按 val/test/S-Y/S-L 分列；B7 配对 Δ 不混写 selector；train-val gap、seed 方差、worst-group；raw3 `sum_abs_error` 仅作监控。
- **可信度层**：interval coverage/宽度/分组 coverage；risk-coverage 与预注册拒识阈值；参数/设备/真实硬件 holdout；所有 selector、scaler、校准器、超参只在 train 内拟合。

---

## 14. 暂缓与拒绝方向汇总（合并）

| 方向                                  | 裁决                | 依据                                                       |
| ----------------------------------- | ----------------- | -------------------------------------------------------- |
| 当前单频波形形状特征（per-cycle 衰减/包络/coda）    | 拒绝作为 P0           | 生成机制无 O₂ 频率相关来源（§10.2）                                   |
| SincNet 修复当前 EC-MSW                 | 暂缓                | 单载波缺可分频谱；参数化滤波器非波形 DL 成功的必要条件                            |
| Deep Ridge"去激活改一行"                  | 拒绝                | 函数上仍线性；原方法依赖 anchor graph 专用优化                           |
| GAN-FixMatch 无限仿真数据                 | 拒绝                | 标签已知，丢标签无成本优势；同域合成不缩小 sim-to-real gap                    |
| TabPFN + Ridge + ExtraTrees 部署堆叠    | 拒绝为正式主线           | observed 特征不可部署；ExtraTrees 过拟合；第二套推理契约（上限探针角色保留，§1 裁决 2） |
| 直接声速解析反演头 / 显式 WLS 聚合层              | 暂缓（可选组件）          | E2s-LS 增益 +0.0005~0.001；无新观测时重排同一信息（§7）                  |
| Kendall 同方差 loss 作 UQ               | 降级为 loss ablation | 任务级全局 σ，无逐点区间，可能下调 O₂ loss                               |
| Deep Sets / Set Transformer 聚合      | 不看好               | 重蹈 attentive pooling 优化失败（§7）                            |
| DEQ                                 | 不适用               | 与小样本标量回归诉求正交                                             |
| FiLM / attention / MoE 动态门控         | 远期候选，当前禁止         | `e2_allowed=false`；token 本身尚不充分，先重建 teacher 表示           |
| 对比学习 / 全波形 AE / masked modeling 预训练 | 不作 P0             | 可能优先学习固定 pulse/幅值/噪声，主动抹掉 TOF                            |
| C1 physical grouped bottleneck 复活   | 停止                | 记忆库冻结结论                                                  |

---

## 15. 论文定位与新颖性边界

定位：**New Setting + information-source audit / 可辨识性审计**。当前未检索到"NDIR 仅测 CO₂ + 超声 + TCS，输出 O₂/N₂/CO₂，并以正式可辨识性和双 OOD 门控评估"的直接重合工作，但方法组件均有先例：

| 最近邻工作                                | 已覆盖内容                                                | tv3 的真实差异轴                                     |
| ------------------------------------ | ---------------------------------------------------- | ---------------------------------------------- |
| Iglesias Hernandez 2022, *Sci. Rep.* | 1–4.5 MHz 同测 TOF 与衰减，区分 N₂-H₂/CO₂/CH₄                | 目标改为 O₂/N₂/CO₂；加 NDIR/TCS；强调 rank 与 nuisance 门 |
| Zhuang 2024, *Meas. Sci. Technol.*   | raw 超声波形 + 概率 CNN，预测 He 中 Ar/air 杂质；论文自证三组分单 TOF 非唯一 | 混合气对象、输入模态、部署协议不同；CNN/UQ 机制不新                  |
| Shen 2023, *Sensors*                 | 多频/变压重建 CO₂ 声弛豫谱                                     | O₂/N₂ 弱差异、常压约束、多模态融合不同                         |
| Liu 2021, *Results Phys.*            | 双频合成弛豫特征分析 CO₂/CH₄/air                               | 双频机制已有先例；tv3 需证 O₂/N₂ 独立灵敏度                    |
| Zhang 2025, *Sensors*                | 760 nm 管道 O₂ TDLAS                                   | 直接 O₂ 检测不新；空间在系统融合与现场协议                        |

新颖性主张边界（冻结）：主张严格分级 A/B/C；不把跨域端到端成功迁移为 tv3 预期；端到端作对照臂；不把 SincNet/GP/AdaCap/可微优化层/残差学习包装为新算法；"未检索到直接重合"不是新颖性证明（未配置 Scopus/ScienceDirect，投稿前需系统检索复核）。

据本次检索，尚无研究在"200 kHz 单向超声 + NDIR CO₂ + TCS + O₂/N₂/CO₂ raw3 + 双 selector OOD"完整设置上验证结构化 teacher-student 方案；算法收益必须由 tv3 消融决定。

---

## 16. 跨层综合与开放问题

三层修复的净效果叠加追踪：修表示层 → 帧保真（E1r 证不充分）；修聚合层 → B1 非劣≈0.4（E2s-LS 证显式算子增益≈0）；换学习机制 → 端到端仍不敌两阶段（机制性）；加物理正则 → 改外推不增秩；唯一破 0.70 的是信息源层，而它把问题推回"先改仿真"。**端到端并不等于去除 DSP，而是让从测量到输出的整条推理链可联合审计，并把必要的测量不变量保留在模型内部。** 全综述重写为两个可证伪命题：**student 能否重建已知有效的 deployable statistics；新增观测能否提高 nuisance 边缘化后的有效信息秩。** 前者决定 DL 是否值得继续，后者决定 O₂ 目标是否物理可达。

开放问题（合并）：

1. 把可微 TOF（§6）+ 蒸馏序列表示（§6.2）+（可选）闭式 WLS declarative 聚合（§7）串成的真·端到端 raw→raw3 单网络，能否在 R/L/S-Y/S-L 全协议与 B7 配对严格非劣？（A 级只跑过 additive LS 消融，未跑过完整端到端单模型 holdout。）
2. 当前仿真下 student 可能只能复现 RawDSP、无额外残差信息——这是可接受的 no-go 结果；若未来多频数据含 teacher 未建模信息，保留零初始化小 residual branch 并用等容量等预算消融判断价值。
3. A1-G0 通过后，端到端多频编码器相对多频 RawDSP + Ridge 是否有非劣以上增益？（当前无对象，P2。）
4. split conformal + risk-coverage 在端到端单模型上的覆盖率是否与两阶段一致？（可信度层，独立于 R²。）
5. teacher hint 限制 student 自由度与 template shortcut 风险，须在 device/session/calibration holdout 下审计。

---

## 17. 风险与非声明（合并）

1. 文献中 CO₂/CH₄/H₂/He 结果不能证明 O₂/N₂ 在 200 kHz 常压下可分。
2. 仿真信息增益不能直接写成真实掘进通风现场能力。
3. 高整体 R² 不能覆盖窄窗全负；安全结论优先报告局部误差与拒识。
4. UQ 只能暴露不确定性，不能创造缺失信息；coverage 未校准不称"可信"。
5. 新硬件通道改变论文类型与系统边界，独立管理成本、标定、响应时间与漂移。
6. 达到 B1 parity 只证明表示修复；稳定超过 B7 且 OOD 与窄窗不退化才是算法增量；只有新增信息源同时改善 Fisher 与业务误差门才是检测突破。

---

## 18. 参考文献（三文档合并去重，标注证据级）

**逆问题与信息边界（C 级）**

1. Ongie G, et al. Deep Learning Techniques for Inverse Problems in Imaging. *IEEE J. Sel. Areas Inf. Theory* 2020.
2. Gottschling NM, Antun V, Hansen AC, Adcock B. The Troublesome Kernel: On Hallucinations, No Free Lunches, and the Accuracy-Stability Tradeoff in Inverse Problems. *SIAM Review* 2025.

**表示层 / 可微前端与 TOF（B–C 级）**

3. Ravanelli M, Bengio Y. Speaker Recognition from Raw Waveform with SincNet. *IEEE SLT* 2018. arXiv:1808.00158.（B）
4. Li J, et al. Localization with Sampling-Argmax. *NeurIPS* 2021. arXiv:2110.08825.（C）
5. Jia H, et al. High-Precision Ultrasonic Flight Time Prediction Based on GAF Image Encoding and Deep Learning. *IEEE TIM* 2025. doi:10.1109/TIM.2025.3633367.（B）
6. Wang Z, et al. Ultrasonic Rough Crack Characterization Using ToFD With Self-Attention Neural Network. *IEEE TUFFC* 2024. doi:10.1109/TUFFC.2024.3459619.（B）
7. Mimura Y, et al. Image Reconstruction in Ultrasonic Speed-of-Sound CT Using TOF Estimated by 2D CNN. *Technologies* 2024. doi:10.3390/technologies12080129.（B）
8. Engel J, et al. DDSP: Differentiable Digital Signal Processing. *ICLR* 2020.（C）
9. Zeghidour N, et al. LEAF: A Learnable Frontend for Audio Classification. *ICLR* 2021.（C）
10. Cobos M, et al. Time Difference of Arrival Estimation from Frequency-Sliding Generalized Cross-Correlations Using CNNs. *ICASSP* 2020.（C）
11. Berg A, O'Connor M, Kenter T. Extending GCC-PHAT Using Shift Equivariant Neural Networks. *Interspeech* 2022.（C）
12. Liu R, et al. An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution. *NeurIPS* 2018.（C）
13. Romero A, et al. FitNets: Hints for Thin Deep Nets. *ICLR* 2015.（C）

**聚合层 / 可微优化与集合函数（C 级）**

14. Zaheer M, et al. Deep Sets. *NeurIPS* 2017. arXiv:1703.06114.
15. Lee J, et al. Set Transformer. *ICML* 2019. arXiv:1810.00825.
16. Amos B, Kolter JZ. OptNet: Differentiable Optimization as a Layer in Neural Networks. *ICML* 2017. arXiv:1703.00443.
17. Agrawal A, et al. Differentiable Convex Optimization Layers. *NeurIPS* 2019. arXiv:1910.12430.
18. Gould S, Hartley R, Campbell D. Deep Declarative Networks. *IEEE TPAMI* 2022. doi:10.1109/TPAMI.2021.3059462.
19. Gould S, et al. Exploiting Problem Structure in Deep Declarative Networks: Two Case Studies. arXiv:2202.12404, 2022.
20. Xu M, et al. DecDTW: Deep Declarative Dynamic Time Warping for End-to-End Learning of Alignment Paths. *ICLR* 2023. arXiv:2303.10778.
21. Xu Z, et al. PMaF: Deep Declarative Layers for Principal Matrix Features (LESS/IED). arXiv:2306.14759, 2023.
22. Bai S, Kolter JZ, Koltun V. Deep Equilibrium Models. *NeurIPS* 2019. arXiv:1909.01377.
23. Kimura M, et al. On permutation-invariant neural networks. arXiv:2403.17410, 2024.
24. Tancik M, et al. Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. *NeurIPS* 2020. arXiv:2006.10739.

**学习机制层（C 级）**

25. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS* 2022. arXiv:2207.08815.
26. Jonker RAA, et al. When Features Matter More than Sequence: A Case for Tabular In-Context Learning in Pain Classification. 2025. doi:10.1145/3747327.3764785.
27. Zeng A, Chen M, Zhang L, Xu Q. Are Transformers Effective for Time Series Forecasting? *AAAI* 2023.
28. Breiman L. Stacked regressions. *Machine Learning* 1996;24:49–64.
29. Belucci B, Lounici K, Meziani K. AdaCap: An adaptive contrastive approach for small-data neural networks. arXiv:2511.20170.
30. Fang Y, Ren Y. Deep Ridge Regression with Anchor Graph. *IEEE ICFTIC* 2024. doi:10.1109/ICFTIC64248.2024.10913017.

**物理约束层（C 级）**

31. Karniadakis GE, et al. Physics-Informed Machine Learning. *Nature Reviews Physics* 2021.
32. Wang S, Yu X, Perdikaris P. When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective. *J. Comput. Phys.* 2022.
33. Ding Y, et al. Power System Robust State Estimation As a Layer: An Optimization-embedded End-to-end Learning Approach. arXiv:2511.22836, 2025.
34. Chang CC, Zeng T. A hybrid data-driven physics-constrained Gaussian process regression framework with deep kernel for uncertainty quantification. *J. Comput. Phys.* 2023;486:112129.

**泛化 / shortcut / UQ（C 级）**

35. Geirhos R, et al. Shortcut Learning in Deep Neural Networks. *Nature Machine Intelligence* 2020.
36. Gulrajani I, Lopez-Paz D. In Search of Lost Domain Generalization. *ICLR* 2021.
37. Sagawa S, et al. Distributionally Robust Neural Networks for Group Shifts. *ICLR* 2020.
38. D'Amour A, et al. Underspecification Presents Challenges for Credibility in Modern Machine Learning. *JMLR* 2022.
39. Varoquaux G. Cross-Validation Failure: Small Sample Sizes Lead to Large Error Bars. *NeuroImage* 2018.
40. Stirn A, et al. Faithful Heteroscedastic Regression with Neural Networks. *AISTATS* 2023.
41. Lakshminarayanan B, Pritzel A, Blundell C. Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles. *NeurIPS* 2017.
42. Tibshirani RJ, et al. Conformal Prediction Under Covariate Shift. *NeurIPS* 2019.
43. Kendall A, Gal Y, Cipolla R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *CVPR* 2018.
44. Perez E, et al. FiLM: Visual Reasoning with a General Conditioning Layer. *AAAI* 2018.
45. Okabe K, Koshinaka T, Shinoda K. Attentive Statistics Pooling for Deep Speaker Embedding. *Interspeech* 2018.

**信息源层 / 声学气体传感（B 级）**

46. Phillips S, Dain Y, Lueptow RM. Theory for a gas composition sensor based on acoustic properties. *Meas. Sci. Technol.* 2003;14(1):70.
47. Ejakov SG, et al. Acoustic attenuation in gas mixtures with nitrogen: Experimental data and calculations. *JASA* 2003;113:1871–1879.
48. Zhu-Shi M, et al. An acoustic gas concentration measurement algorithm for carbon monoxide in mixtures based on molecular multi-relaxation model. *Acta Phys. Sin.* 2008;57:5749.
49. Shu Y, Wang S. Reconstruction algorithm of relaxation attenuation spectrum in polyatomic gas. *Acta Phys. Sin.* 2008;57(7):4282–4291.
50. Zhang X, Wang S, Zhu M. Locating the inflection point of frequency-dependent velocity dispersion by acoustic relaxation to identify gas mixtures. *Meas. Sci. Technol.* 2020. doi:10.1088/1361-6501/ab9375.
51. Liu T, et al. Acoustic analysis of gas compositions based on molecular relaxation features. *Results Phys.* 2021;25:104304.
52. Liu T, Wang S, Zhu M. A versatile acoustic gas sensing method via extracting intrinsic molecular internal specific heat. *Phys. Lett. A* 2021. doi:10.1016/j.physleta.2021.127349.
53. Shen K, et al. Measurement of the Acoustic Relaxation Absorption Spectrum of CO₂ Using a DBR Fiber Laser. *Sensors* 2023;23:4740.
54. Iglesias Hernandez L, et al. Gas discrimination by simultaneous sound velocity and attenuation measurements using uncoated CMUTs. *Sci. Rep.* 2022;12:744.
55. Zhuang B, et al. Impurity Gas Detection for SNF Canisters Using Probabilistic Deep Learning and Acoustic Sensing. *Meas. Sci. Technol.* 2024;35:126005.

**直接 O₂ 通道（B 级）**

56. Zhang Y, et al. On-site and sensitive pipeline oxygen detection equipment based on TDLAS. *Sensors* 2025;25:4027.
57. Guo YM, et al. Ppm-level photoacoustic oxygen gas sensor with a 3 W red diode laser. *Opt. Express* 2026. doi:10.1364/OE.589797.

---

## 19. 与项目不变量对照（合并）

| 不变量                                       | 状态  | 本合并稿约束                               |
| ----------------------------------------- | --- | ------------------------------------ |
| raw3 输出、`out_dim=3`，不回填 N₂                | 遵守  | 蒸馏 student 与所有支线仍输出 raw3             |
| 不使用 gas_head/ILR/ALR/闭包残差头                | 遵守  | 物理 loss 仍约束 raw3                     |
| `sum_abs_error` 仅作监控                      | 遵守  | 不作主通过门                               |
| `e2_allowed=false`（不开 FiLM/attention/MoE） | 遵守  | FiLM/attentive pooling 只作远期候选引用      |
| B7 保持默认部署头                                | 遵守  | 所有候选先作冻结对照；G4 失败保留 B7                |
| 不扩大 residual MLP                          | 遵守  | `delta_theta` 零初始化或 64-64，不以加宽/加深作路线 |
| OOD 分别报告 S-Y/S-L                          | 遵守  | 不合并 selector 结论                      |
| 停止 C1 physical grouped bottleneck         | 遵守  | 不复活维度/dropout/gating 搜索              |
| v1 verdict 不改写                            | 遵守  | 多频与直接通道另立 schema/manifest 后重新审计      |
| split 主键 `mixture_id`                     | 遵守  | G1 门强制，不回退 `sequence_id`             |
| 不把 full B1 包装成端到端改进                       | 遵守  | 单模型须 B7 配对非劣 + 等输入消融归因               |
| 上限逼近 B1≈0.4，不破 0.70                       | 遵守  | 突破归于信息源层                             |
| 未检索到直接重合 ≠ 新颖性证明                          | 遵守  | 投稿前系统检索复核                            |
