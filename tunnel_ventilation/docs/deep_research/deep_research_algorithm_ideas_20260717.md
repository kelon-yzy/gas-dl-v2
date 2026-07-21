# Deep Research: tv3 掘进通风 O₂ 预测突破方向

> 基于[项目记忆库](../掘进通风项目记忆库.md) §二至§六的正式结论栈，评估在 B7、双 selector OOD 和单向 TOF 可辨识性审计之后，哪些研究方向仍具备物理可行性与论文价值。
>
> 日期：2026-07-17；2026-07-17 完成可行性复核与路线重排。

---

## 1. 执行摘要

当前 tv3 的核心问题不是“还缺一个更强回归头”，而是现有观测在目标部署 nuisance 下是否包含足够的 O₂ 信息。正式证据已经固定：

1. B7 OOF Ridge residual MLP 是当前默认 RawDSP 头，ID test O₂ R²约 0.70；
2. S-L OOD O₂ R²约 0.70，但 S-Y OOD 仅约 0.43，不能声称所有 OOD 均达到 0.70；
3. 0.8% O₂ bins 内，oracle、observed 和 B7 均为负 R²，只能支持整体高低档位辨识；
4. 单向 TOF 对 O₂、CO₂、T、L 的联合 Fisher 秩为 1，正式 verdict 为 `information_source_upgrade_required`；
5. 当前仿真 waveform 是固定 transducer pulse 的时移与标量衰减，O₂ 弛豫衰减项被设为 0，RawDSP 已提取 amplitude、SNR、PSR 与 peak width。

因此，研究路线重新划分为三类：

| 层级 | 路线 | 当前裁决 |
|---|---|---|
| 主线 | 多频声速色散与衰减谱，或直接 O₂ 传感通道 | 唯一具备突破可辨识性墙的机制；先做信息增益审计 |
| 支线 | 物理一致性、train-only 特征消融、独立目标、校准不确定性 | 可做稳健性和边界刻画；不得承诺突破物理上限 |
| 暂缓或拒绝 | 当前单频波形形状、SincNet、Deep Ridge 一行替换、GAN-FixMatch、TabPFN 部署堆叠 | 与当前生成机制、正式证据或部署契约不匹配 |

本文不再给出未经实验支持的 `O₂ R² +0.0x` 预期增益。所有候选方向改用可审计的 go/no-go 门和停止条件。

---

## 2. 研究问题与证据等级

| RQ | 问题 | 对应判据 |
|---|---|---|
| RQ1 | 新声学观测是否增加相对单频 TOF 独立的信息维度？ | Fisher 秩、条件数、误差预算、窄窗 P90 |
| RQ2 | 在不增加信息源时，哪些模型改动能改善 OOD 或可信度？ | R/L/S-Y/S-L 完整协议、校准覆盖率、拒识风险 |
| RQ3 | 约 4200 个训练样本下，哪些轻量方法值得做诊断？ | nested train-only 选择、多 seed、B7 配对非劣 |
| RQ4 | 若声学信息仍不足，哪种直接 O₂ 通道最符合项目条件？ | 灵敏度、响应时间、集成复杂度、真实硬件可得性 |

证据按以下等级使用：

- **A 级**：项目正式产物、冻结协议和本地实现，可直接约束路线。
- **B 级**：已发表且对象、频段或传感机制接近的实验工作，可支持机制可行性。
- **C 级**：跨领域方法、预印本或 2026 候选论文，只能支持试验候选，不能外推项目收益。

跨领域论文中的 R²、RMSE 或速度增益不得迁移为 tv3 的预期数值。

---

## 3. 新颖性边界与最近邻工作

当前未检索到“NDIR 仅测 CO₂ + 超声 + TCS，输出 O₂/N₂/CO₂，并以正式可辨识性和双 OOD 门控评估”的直接重合工作。但方法组件本身已有明确先例：

| 最近邻工作 | 已覆盖内容 | tv3 的真实差异轴 |
|---|---|---|
| Iglesias Hernandez et al., 2022, *Scientific Reports* | 1–4.5 MHz 下同时测量 TOF 与衰减，区分 N₂-H₂/CO₂/CH₄ | 目标组分改为 O₂/N₂/CO₂；加入 NDIR/TCS；强调 rank 与 nuisance 门 |
| Zhuang et al., 2024, *Meas. Sci. Technol.* | 原始超声波形和概率 CNN，预测 He 中 Ar/air 杂质 | 混合气对象、输入模态和部署协议不同；CNN/UQ 机制不新 |
| Shen et al., 2023, *Sensors* | 多频与变压恢复 CO₂ 声弛豫吸收谱 | O₂/N₂ 弱差异、常压约束和多模态融合不同 |
| Liu et al., 2021, *Results in Physics* | 双频合成弛豫特征，分析 CO₂/CH₄/air | 双频机制已有先例；tv3 需证明对 O₂/N₂ 有独立灵敏度 |
| Zhang et al., 2025, *Sensors* | 小型化 760 nm 管道 O₂ TDLAS | 直接 O₂ 检测并不新；可发表空间在系统融合和现场协议 |

论文定位应是 **New Setting + information-source audit**，而不是把 SincNet、GP、AdaCap 或残差学习包装为新算法。

---

## 4. 主线 A：信息源升级

### 4.1 A1 多频声速色散与衰减谱

#### 机制依据

- 分子弛豫使声速和吸收随频率变化；不同组分的弛豫时间和强度不同。
- 单频 TOF 只有一个主要观测方向，多频声速或衰减可增加新的敏感度向量。
- 已有工作分别在 25–40 kHz、约 200 kHz 和 1–4.5 MHz 验证了弛豫谱或 TOF/衰减联合测量，但没有证明 tv3 的 O₂/N₂ 在常压、当前噪声和声程下可分。

#### 当前缺口

现有 200 kHz 仿真不能用于证明多频价值：当前声学模型只在中心频率计算一次衰减，并将同一 pulse 进行时移和标量缩放。启动新模型训练前，必须先扩展物理表示，而不是把现有单频波形送入更复杂网络。

#### 决定性实验 A1-G0

1. 从文献或可追踪物理模型登记候选频率，不先按模型表现挑频率；
2. 对每个频率生成声速、衰减及其对 O₂/CO₂/T/L/RH 的 Jacobian；
3. 比较单频、双频和小型频率组的 Fisher 秩、条件数和 nuisance 边缘化预算；
4. 使用与 v1 一致的 P90=`0.4 vol%`、单 nuisance=`50%`、拒绝率=`5%` 门；
5. 只有表示门通过后，才构建多频 RawDSP 和 B1/B7 对照。

#### Go / no-go

- **Go**：新增频率使有效秩增加，且误差预算相对单频有明确改善；随后进入冻结 builder 与模型实验。
- **No-go**：灵敏度向量仍近共线，或改善完全低于测量噪声；停止多频网络和超参搜索。

### 4.2 A2 当前单频波形形状特征

原建议包括 per-cycle 衰减、上升/下降沿不对称、脉冲展宽和 coda energy。对当前 benchmark，裁决为 **拒绝作为 P0**：

- [`waveforms.py`](../../tv3/sim/generation/waveforms.py) 使用固定 `transducer_response_pulse(spec)`，组分只改变到达时间与 `exp(-alpha*L)` 标量；
- [`acoustic_physics.py`](../../tv3/sim/generation/tunnel_ventilation/acoustic_physics.py) 固定 `alpha_lambda_max_o2=0.0`，不存在 O₂ 引起的频率相关形状变化；
- [`raw_dsp_features.py`](../../tv3/ml/raw_dsp_features.py) 已包含 peak amplitude、SNR、PSR、correlation peak 和 peak width；
- 在此生成机制下，归一化包络形状或 coda 比值没有新的 O₂ 物理来源，新增特征更可能学习噪声或已有标量的重复表达。

该路线只能在 A1 引入频率依赖传递函数，或取得真实宽带波形后重新开放。届时先做单特征条件互信息与 train-only ablation，不直接训练 CNN。

### 4.3 A3 直接 O₂ 传感通道

若目标是实际突破 O₂ 窄窗口，而不是维持纯声学约束，直接通道的成功概率高于继续更换回归头。

| 路线 | 可行性 | 当前定位 |
|---|---|---|
| 760 nm VCSEL/DFB TDLAS | 高，但硬件当前暂停 | 最成熟的直接 O₂ 方案；恢复实验条件后优先 |
| 760 nm PAS/光热声学 | 中 | 可降低长光程需求，但仍需激光、谐振腔、温控与声学抗扰验证；不能直接称为更简单 |
| Raman/FERS | 物理上可行，成本与复杂度高 | 不符合当前项目资源优先级，保留为长期参考 |

TDLAS/PAS 应独立为硬件融合或系统论文，不与小样本算法、GP 和多频声学同时作为一篇论文的贡献。

---

## 5. 支线 B：不增加信息源的软件实验

### 5.1 B1 声速物理一致性正则

可验证形式必须使用两个独立来源：

```text
L_phys = MSE(c_measured, c_from_composition(y_pred, T_measured))
```

其中 `c_measured` 来自冻结 RawDSP 的 corrected TOF 与 L，`c_from_composition` 由预测 raw3、测得温度和登记物性计算。若等式两侧都由同一预测量生成，则只是代数自洽，不是物理监督。

边界：

- 该 loss 可能降低不物理预测或改善外推，但不会增加 Fisher 秩；
- 权重只能在 train 内层选择，不能用 test 或 OOD 调参；
- 必须报告 B7 配对结果、窄窗口和 S-Y/S-L，不得只报告 ID 均值；
- 若收益来自额外输入 `c_measured` 而非约束本身，必须用等输入消融隔离归因。
- 必须预先定义 raw3 负值和非闭包预测的处理语义；禁止在 loss 或 evaluator 中静默 clamp、重归一化或回填 N₂。若无法给出可微且与 raw3 契约兼容的定义，则不启动该实验。

裁决：**条件接受，作为低成本 ablation，不作为突破主线。**

### 5.2 B2 train-only 特征消融与压缩

特征冗余是合理诊断，但原“Ridge 系数 + RF importance 压到 50–150 维”的方案会引入两个问题：高度共线下单次 Ridge 系数不稳定；当前 ExtraTrees 已显著过拟合，不能作为可靠 selector。

推荐协议：

1. 先按物理来源和统计族做 group ablation；
2. 在每个 outer split 的 train 内做 stability selection 或 nested CV；
3. selector、scaler 和模型必须在每个 outer split 重新拟合；
4. 比较 B7 全特征、压缩特征与等维随机/分组对照；
5. 只有 R/L/S-Y/S-L 配对非劣且至少一个预注册风险指标改善，才保留压缩 builder。

裁决：**接受为工程诊断和降成本支线；新颖性低。**

### 5.3 B3 独立目标 residual MLP

三个 single-target MLP 分别学习 CO₂/O₂/N₂ 的 OOF Ridge residual，最终仍组合为 raw3。它用于回答共享 trunk 是否存在梯度冲突，不改变输出契约，不回填 N₂。

裁决：**接受为诊断实验。** 若没有 gradient cosine 或独立目标对照证据，不再声称多任务冲突是根因。

### 5.4 B4 校准不确定性与拒识

安全场景需要知道何时不应信任预测，但“不确定性大”不会自动修复不可辨识性。

最低基线顺序：

1. B7 residual 的 split conformal prediction；
2. deep ensemble 或 heteroscedastic head；
3. 只有简单基线不足时，再评估 latent DKL/SVGP 或非平稳 GP。

评价指标必须包含区间 coverage、平均宽度、分组 coverage、S-Y/S-L coverage、拒识后的 risk-coverage 曲线。不得把 Kendall 同方差任务权重当作逐点不确定性；它每个任务只学习一个全局 σ，并可能下调噪声较大的 O₂ loss。

裁决：**接受为 Stronger/安全性方向，不作为 Higher/R² 方向。**

### 5.5 B5 AdaCap 与回归增强

AdaCap 在小样本 residual 模型上有候选证据，但在 500–10,000 样本区间效果并不一致，且当前主要证据来自跨领域 benchmark。RC-Mixup 同样可能破坏高维共线特征的物理流形。

裁决：**仅在 B1–B4 完成后作为二级候选。** 需要与同参数量、同训练预算和同 OOF residual 输入的 B7 对照，不能只与默认 MLP 比较。

---

## 6. 暂缓或拒绝的方向

| 方向 | 裁决 | 原因 |
|---|---|---|
| Kendall 同方差 loss 作为 UQ | 降级为 loss ablation | 只产生任务级全局 σ，不产生逐点区间；可能与当前 O₂ 上权重目标相反 |
| SincNet 修复当前 EC-MSW | 暂缓 | SincNet 在语音中证明参数化滤波器有效，不证明其是波形 DL 成功的必要条件；当前单载波也缺少可分频谱内容 |
| Deep Ridge“去激活改一行” | 拒绝 | 无激活多层映射在函数上仍为线性；原方法还依赖 anchor graph 和专用优化，不等于替换激活函数 |
| GAN-FixMatch 使用无限仿真数据 | 拒绝 | 仿真生成时组分标签已知，主动丢弃标签没有成本优势；同域合成数据也不缩小 sim-to-real gap |
| TabPFN + Ridge + ExtraTrees 部署堆叠 | 拒绝为正式主线 | TabPFN 当前使用 simulator-derived observed 特征且不可部署；ExtraTrees 已过拟合；堆叠增加第二套推理契约 |
| 直接声速解析反演头 | 暂缓 | E2s-LS 已证明 TOF-L alone 几乎无增益；没有新观测时只是重排同一信息 |

---

## 7. B7 结果的正确理论解释

| 观察 | 可以支持的解释 | 不能支持的过度解释 |
|---|---|---|
| Ridge 稳定 | 主信号具有强线性/低阶结构，L2 对共线特征有效 | 最优岭惩罚一定为零或负 |
| 默认 MLP 失败、R5-T 成功 | 目标尺度是已验证主因；高维共线和优化仍可能影响 | Spectral Bias 证明 MLP 先追逐高频噪声 |
| B7 优于 B1 | OOF 线性基底加低容量非线性 residual 在当前协议有效 | 等价于某个特定非线性 kernel ridge |
| TabPFN observed 成功 | 当前 observed 契约存在可学习非线性 | 非线性一定来自真实硬件物理，或可直接部署 |
| C1 grouped bottleneck 失败 | 早期物理分组压缩破坏了当前任务所需交互 | 所有物理引导、低参数或分组方法都无效 |

Breiman 堆叠、Spectral Bias、表格 DL benchmark 和残差学习文献可以解释候选机制，但项目内归因仍以冻结 ablation 为准。

---

## 8. 分阶段执行与验收

### P0：信息增益预审，不训练新模型

| 顺序 | 动作 | 产出 | 停止条件 |
|---|---|---|---|
| 1 | 冻结单频 v1 作为对照 | manifest、频率与 nuisance 表 | 不改写现有 verdict |
| 2 | 建立多频声速/衰减 Jacobian | sensitivity 与 Fisher 审计 | 新频率仍近共线则停止 |
| 3 | 复用业务门计算误差预算 | P90、nuisance 比例、拒绝率 | 三项门无明确改善则停止 |
| 4 | 登记真实硬件可用频段和噪声 | 频段可实现性表 | 仅理论可分但硬件不可测则不进入模型 |

### P1：低成本软件诊断

P1 与 P0 可并行，但不得包装成信息源突破：

1. B1 物理一致性 loss；
2. B2 nested train-only group ablation/feature selection；
3. B3 independent-target residual MLP；
4. B4 split conformal 与 risk-coverage；
5. 只有前四项形成明确残余机制时，才试 AdaCap 或 RC-Mixup。

所有实验保持 B7 默认头，使用完整 R/L/S-Y/S-L × split seed × training seed 配对协议。单一 val 改善不构成通过。

### P2：表示和硬件升级

- A1-G0 通过后：实现多频 RawDSP、B1 parity、B7 非劣和独立参数 holdout；
- 实验条件恢复后：优先复核 760 nm TDLAS，再根据集成成本评估 PAS；
- 真实硬件数据到位后：单独审计 sim-to-real、设备 profile、flow、T/RH/P/L/SNR OOD；
- SincNet 或其他 learned waveform encoder 只有在新表示确实包含频率相关信息后才可重新立项。

---

## 9. 统一验收指标

### 9.1 信息层

- Fisher 有效秩与条件数；
- O₂ 对 nuisance 的局部灵敏度比；
- 0.8% bins 的 P90、MAE、局部斜率；
- 拒绝率与不可用区域。

### 9.2 模型层

- O₂ R²/MAE：val、test、S-Y、S-L 分列；
- B7 配对 Δ，不混写不同 selector；
- train-val gap、seed 方差和 worst-group；
- raw3 `sum_abs_error` 仅作监控，不作主通过门。

### 9.3 可信度层

- interval coverage、平均宽度和分组 coverage；
- risk-coverage 与预注册拒识阈值；
- 参数 holdout、设备 holdout 和真实硬件 OOD；
- 所有 selector、scaler、校准器和超参数只在 train 内拟合。

---

## 10. 风险与非声明

1. 文献中的 CO₂/CH₄/H₂/He 结果不能证明 O₂/N₂ 在 200 kHz 常压下可分。
2. 仿真信息增益不能直接写成真实掘进通风现场能力。
3. 高整体 R²不能覆盖窄窗口全负；安全结论必须优先报告局部误差和拒识。
4. UQ 只能暴露不确定性，不能创造缺失信息；coverage 未校准时不称“可信”。
5. 新硬件通道改变论文类型和系统边界，应独立管理成本、标定、响应时间和漂移。
6. 未配置 Scopus/ScienceDirect；“未检索到直接重合”不是新颖性证明，投稿前仍需系统检索复核。

---

## 11. 关键参考文献

1. Phillips S, Dain Y, Lueptow RM. Theory for a gas composition sensor based on acoustic properties. *Meas. Sci. Technol.* 2003;14(1):70. doi:10.1088/0957-0233/14/1/311
2. Ejakov SG, Phillips S, Dain Y, Lueptow RM, Visser JH. Acoustic attenuation in gas mixtures with nitrogen: Experimental data and calculations. *JASA*. 2003;113:1871–1879. doi:10.1121/1.1559177
3. Shu Y, Wang S. Reconstruction algorithm of relaxation attenuation spectrum in polyatomic gas. *Acta Phys. Sin.* 2008;57(7):4282–4291. doi:10.7498/aps.57.4282
4. Liu T, Hu Y, Zhang X, Zhu M. Acoustic analysis of gas compositions based on molecular relaxation features. *Results Phys.* 2021;25:104304.
5. Iglesias Hernandez L, Shanmugam P, Michaud JF, et al. Gas discrimination by simultaneous sound velocity and attenuation measurements using uncoated CMUTs. *Sci. Rep.* 2022;12:744. doi:10.1038/s41598-021-04689-4
6. Shen KC, Yuan J, Li M, Wen X, Lu H. Measurement of the acoustic relaxation absorption spectrum of CO₂ using a DBR fiber laser. *Sensors*. 2023;23:4740. doi:10.3390/s23104740
7. Zhuang B, Gencturk B, Oberai AA, et al. Impurity gas detection for SNF canisters using probabilistic deep learning and acoustic sensing. *Meas. Sci. Technol.* 2024;35:126005. doi:10.1088/1361-6501/ad730d
8. Ravanelli M, Bengio Y. Speaker recognition from raw waveform with SincNet. *IEEE SLT*. 2018. arXiv:1808.00158
9. Kendall A, Gal Y, Cipolla R. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *CVPR*. 2018.
10. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS*. 2022.
11. Breiman L. Stacked regressions. *Machine Learning*. 1996;24:49–64.
12. Chang CC, Zeng T. A hybrid data-driven physics-constrained Gaussian process regression framework with deep kernel for uncertainty quantification. *J. Comput. Phys.* 2023;486:112129. doi:10.1016/j.jcp.2023.112129
13. Belucci B, Lounici K, Meziani K. AdaCap: An adaptive contrastive approach for small-data neural networks. arXiv:2511.20170.
14. Fang Y, Ren Y. Deep Ridge Regression with Anchor Graph. *IEEE ICFTIC*. 2024. doi:10.1109/ICFTIC64248.2024.10913017
15. Zhang Y, Yuan K, Yu Z, et al. On-site and sensitive pipeline oxygen detection equipment based on TDLAS. *Sensors*. 2025;25:4027. doi:10.3390/s25134027
16. Guo YM, Liu Z, Liang Y, Yin X, Ma B. Ppm-level photoacoustic oxygen gas sensor with a 3 W red diode laser. *Opt. Express*. 2026. doi:10.1364/OE.589797

---

## 12. 与项目记忆库的不变量对照

| 不变量 | 状态 | 本文约束 |
|---|---|---|
| raw3 输出、`out_dim=3` | 遵守 | 软件支线不回填 N₂，不改输出契约 |
| 不使用 gas_head/ILR/ALR | 遵守 | 物理 loss 仍约束 raw3 |
| `sum_abs_error` 仅作监控 | 遵守 | 不作为主通过门 |
| `e2_allowed=false` | 遵守 | 不启动 FiLM/attention/MoE |
| B7 保持默认头 | 遵守 | 所有候选先作冻结对照，不直接替换 B7 |
| 不扩大 residual MLP | 遵守 | 不以加宽/加深作为路线 |
| OOD 分别报告 S-Y/S-L | 遵守 | 不合并 selector 结论 |
| 停止 C1 physical grouped bottleneck | 遵守 | 不复活 C1 维度、dropout 或 gating 搜索 |
| v1 verdict 不改写 | 遵守 | 多频与直接通道另立 schema/manifest 后重新审计 |
