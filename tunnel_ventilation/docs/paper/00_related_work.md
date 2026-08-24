# 相关工作：已核实参考文献与领域边界判定

> 检索日期：2026-08-20
> 检索工具：CrossRef / PubMed / arXiv（`mcp__academic-search`、`mcp__paper-search`）
> 用途：第 1 节引言的写作依据，以及全文可声明创新点的边界
> 纪律：只收录题名、作者、期刊、DOI 全部核对成功的条目。未核实的一律标注。

---

## 1. 已核实的参考（全部带 DOI）

### 1.1 本方案的直接概念前身

| 文献                                                                                                    | 期刊                      | DOI                          | 引用数    |
| ----------------------------------------------------------------------------------------------------- | ----------------------- | ---------------------------- | ------:|
| Lueptow & Phillips 1994, *Acoustic sensor for determining combustion properties of natural gas*       | Meas. Sci. Technol.     | `10.1088/0957-0233/5/11/010` | 20     |
| **Phillips, Dain & Lueptow 2003, *Theory for a gas composition sensor based on acoustic properties*** | **Meas. Sci. Technol.** | `10.1088/0957-0233/14/1/311` | **50** |

Phillips 2003 是最接近的前身：用**声速 + 声衰减**确定气体组成，并在理论上演示了三组分混合气。其摘要指出：对于三组分气体混合物，测得的声速和声衰减会分别在其中两个气体的组分平面上定义直线，两条直线的交点确定气体组成。

**这条对本文的定位有直接影响**：传感概念本身不是新的，且已发表在目标期刊上 23 年。本文能补充的是：当把温度、声程、触发延迟这些干扰参数加进那个"两线相交"的几何后，交点条件数如何劣化，以及由此产生的精度上限。

### 1.2 弛豫吸收物理

| 文献                                                                                                                                      | 期刊   | DOI                 | 引用数 |
| --------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------------------- | ---:|
| Dain & Lueptow 2001, *Acoustic attenuation in three-component gas mixtures—Theory*                                                      | JASA | `10.1121/1.1352087` | 70  |
| Dain & Lueptow 2001, *Acoustic attenuation in a three-gas mixture: Results*                                                             | JASA | `10.1121/1.1413999` | 34  |
| Ejakov, Phillips, Dain, Lueptow & Visser 2003, *Acoustic attenuation in gas mixtures with nitrogen: Experimental data and calculations* | JASA | `10.1121/1.1559177` | 75  |

### 1.3 声学弛豫谱反演与深度学习

| 文献                                                                                                                                                 | 期刊                  | DOI                          | 引用数 |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | ---------------------------- | ---:|
| Zhu, Liu, Zhang & Li 2018, *A simple measurement method of molecular relaxation in a gas by reconstructing acoustic velocity dispersion*           | Meas. Sci. Technol. | `10.1088/1361-6501/aa96da`   | 10  |
| Zhang, Wang & Zhu 2020, *Locating the inflection point of frequency-dependent velocity dispersion by acoustic relaxation to identify gas mixtures* | Meas. Sci. Technol. | `10.1088/1361-6501/ab9375`   | 17  |
| **Liu, Mei, Zhu & Cheng 2023, *Identifying gas mixtures based on acoustic relaxation spectroscopy and attention recurrent neural network***        | Results in Physics  | `10.1016/j.rinp.2023.106558` | 4   |

Liu 2023（ARGD）是**必须显式承认的反例**：BiGRU + 注意力，输入六个频点的 `c(f)` 或归一化吸收，三组分 CO₂/CH₄/N₂，约 4.98 万仿真样本。"首次把深度学习用于声学弛豫谱组分反演"这句话不能用。

### 1.4 信息量分析与可辨识性

| 文献                                                                                                                                             | 出处                    | DOI                                | 引用数      |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ---------------------------------- | --------:|
| **Rodgers 2000, *Inverse Methods for Atmospheric Sounding: Theory and Practice***                                                              | World Scientific      | `10.1142/3171`                     | **2273** |
| — 其中第 2 章 *Information Aspects*                                                                                                                | 同上                    | `10.1142/9789812813718_0002`       | —        |
| Fewster & Jupp 2013, *Information on parameters of interest decreases under transformations*                                                   | J. Multivariate Anal. | `10.1016/j.jmva.2013.05.010`       | 4        |
| Diong, Roueff, Lasaygues & Litman 2015, *Precision analysis based on Cramér–Rao bound for 2D acoustics and electromagnetic inverse scattering* | Inverse Problems      | `10.1088/0266-5611/31/7/075003`    | 8        |
| Zois & Mitra 2014, *Controlled sensing: a myopic Fisher information sensor selection algorithm*                                                | IEEE GLOBECOM         | `10.1109/glocom.2014.7037333`      | 2        |
| Sibug-Torres & Enriquez 2019, *Design of potentiometric sensor arrays using Fisher information and genetic algorithm*                          | IEEE ICECIE           | `10.1109/icecie47765.2019.8974846` | 2        |

### 1.5 可分离非线性最小二乘与干扰参数

| 文献                                                                                                             | 期刊               | DOI                              | 引用数 |
| -------------------------------------------------------------------------------------------------------------- | ---------------- | -------------------------------- | ---:|
| Golub & Pereyra 2003, *Separable nonlinear least squares: the variable projection method and its applications* | Inverse Problems | `10.1088/0266-5611/19/2/201`     | 612 |
| Kaufman 1975, *A variable projection method for solving separable nonlinear least squares problems*            | BIT              | `10.1007/bf01932995`             | 191 |
| Aravkin & van Leeuwen 2012, *Estimating nuisance parameters in inverse problems*                               | Inverse Problems | `10.1088/0266-5611/28/11/115016` | —   |

### 1.6 后验校准与诊断

| 文献                                                                                                                              | 出处                       | 标识                   |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------------------ | -------------------- |
| Talts, Betancourt, Simpson, Vehtari & Gelman 2018, *Validating Bayesian inference algorithms with simulation-based calibration* | arXiv（未见正式期刊版）           | `arXiv:1804.06788v2` |
| **Vehtari, Simpson, Gelman, Yao & Gabry, *Pareto smoothed importance sampling***                                                | arXiv（v9, 2024-03-13 更新） | `arXiv:1507.02646v9` |

> **PSIS 引用状态更新**：§4 里原标注"未核实"的那条，作者与题名现已核对无误（五作者，非三作者）。arXiv 上无 DOI。该文通常被引为 JMLR 25(72), 2024，**但该期刊卷期号本次未独立核实**，投稿前需确认。

---

## 2. 领地判定：哪些声明不能提

按证据强度排序。

| #   | 不能声明                       | 占领者                                          | 强度     |
| --- | -------------------------- | -------------------------------------------- | ------ |
| 1   | "先算信息量、再选估计器"是新的工作顺序       | **Rodgers 2000 第 2 章**，大气遥感界的教科书做法，2273 引    | **极强** |
| 2   | 用声速 + 声衰减做三组分气体组成传感是新概念    | Phillips 2003（目标期刊）、Lueptow 1994             | **强**  |
| 3   | 首次把深度学习用于声学弛豫谱组分反演         | Liu 2023 (ARGD)                              | **强**  |
| 4   | 首次用 Cramér–Rao 界分析声学反问题的精度 | Diong 2015                                   | 中强     |
| 5   | 首次用 Fisher 信息做传感器方案设计      | Sibug-Torres 2019、Zois 2014                  | 中      |
| 6   | 变量投影消去干扰参数是新方法             | Golub–Pereyra 2003、Kaufman 1975、Aravkin 2012 | **强**  |

**第 1 条是本次检索最重要的结果，它实质性削弱了原定的 §1 框架。** 原计划把"审计先于建模"作为核心方法学贡献；这个顺序在大气遥感反演界已经成熟三十年，Rodgers 的信息量分析、平均核、信号自由度（DFS）就是为此设计的。

---

## 3. 检索后仍然站得住的贡献

逐条给出与既有工作的差别。

**（a）秩亏时仍可计算的目标单位判据。** Rodgers 体系的信息量分析（DFS、Shannon 信息量、平均核）建立在**先验加权后的系统可逆**这个前提上。本工作的场景在单标量观测下联合 Fisher 秩恒为 1（189/189 个点实测），干扰参数块无法边缘化，DFS 与 CRB 都不给出目标单位下的数。等效目标误差 P90 恰好在这个区域仍然可算，代价是放弃联合最优性、只做一阶传播。**这是窄但真实的技术差别，且实现只有 34 行（27 行非空）、无领域依赖。**

**（b）审计失败后的硬件规格反演。** 最优实验设计文献（Attia 2025、Capellari 2016 等）解的是"给定预算选最优设计"。本工作解的是反向问题：给定精度目标，扫描规格空间，报告**约束排序**与**饱和层**。二者数学上相关，但输出的用途不同——前者产出一个设计，后者产出一份可证伪的采购规格。检索未发现气体传感领域的同类做法，但该说法只能限定为"未检索到"，不能写成"首次"。

**（c）该做法在多组分气体传感领域尚未常规化。** 检索显示该领域文献以分类、传感器融合与阵列设计算法为主（Solorzano 2017、Zhang 2021、Day & Wilmer 2020 等），未见把信息量充分性作为前置门的工作。**这是一个可以声明的"迁移"贡献，不是"发明"贡献**——把大气遥感的成熟做法引入一个尚未采用它的领域，并给出秩亏情形下的替代判据。

**（d）治理层与负结果报告纪律。** 预注册 + 版本化冻结 + 冻结后一致性审查 + 拒绝率与两种覆盖率并报 + 宽度对量程的检查。其中**冻结后一致性审查**（每个可达 verdict 必须有可达路径）与**描述漂移**（五例，数字都在冻结产物里、错的是描述那个数的句子）是本工作自己的发现，未检索到既有表述。

**（e）经验证据本身。** 六个方法各异、相互独立的测量收敛到同一个物理墙位置；五种结构不同的端到端网络在三个 split 上全部负 R²；同参数量随机置换对照反超物理分组。这类证据密度在传感器论文里少见。

---

## 4. 对 §1 与投稿方案的影响

### 4.1 §1 必须改的框架

**不能写**："本文提出在算法开发之前先做信息量审计的方法。"
**可以写**："信息量审计在大气遥感反演中是成熟做法（Rodgers 2000），但在多组分气体传感文献中尚未常规化；本文将其迁移过来，并针对该场景的秩亏结构给出一个 CRB 失效时仍可计算的目标单位判据。"

§1 的叙事应从"我们提出一个新顺序"改成"**一个成熟领域的标准做法，在相邻领域缺席，代价是什么**"——本文的负结果链条正好量化了这个代价：两条先训练后审计的路线各消耗一次完整训练，才得到审计几分钟就能给出的结论。

### 4.2 §7.6 需要相应收窄

§7.6 现在写的是"该领域的组织方式有盲点"。这条仍然成立，但必须补一句：**这个盲点在相邻领域并不存在**，并给出 Rodgers 作为对照。这样反而更有力——不是"没人想到"，而是"想到了的人在另一个学科"。

### 4.3 §4 的 PSIS 引用

作者与题名已核实。JMLR 卷期号未核实，投稿前确认。

### 4.4 对目标期刊的支持

首选 Measurement Science and Technology 的判断得到加强：本课题的**三条最相关文献**（Phillips 2003、Zhu 2018、Zhang 2020）全部发表在该刊。审稿人熟悉这条线，也意味着 Phillips 2003 必须被正面处理而非回避。

---

## 5. 还没检索的

| 项                         | 为什么需要          | 优先级    |
| ------------------------- | -------------- | ------ |
| 测量科学中的预注册与负结果报告实践         | §7.3 的治理层定位    | 中      |
| JMLR 上 PSIS 的正式卷期号        | §4 引用完整性       | 投稿前必办  |
| 摊销推断（SBI）在传感反演中的既有应用      | §7.5 的开放问题定位   | 中      |
| 声学测氧的替代路线（TDLAS 760 nm 等） | §7.4 的迁移条件     | 低      |
| 中文文献（CNKI），尤其矿井通风气体检测     | 学位论文需要，期刊论文非必需 | 学位论文阶段 |

网页检索在本次检索中两次失败（一次 ECONNRESET、一次 429），补充检索改用学术数据库 MCP 完成。上表各项待后续补齐。
