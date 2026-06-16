# ML 模型改进方向分析

- 状态：参考资料（分析文档，非执行计划）
- 负责路线：传统 ML（ridge 主线及其扩展）
- 创建日期：2026-06-16
- 关联主线：`ridge_multiwindow_all_modalities`（正式 ML 主线，已验收）
- 关联归档：`outputs/archive/multiwindow_n2_20260612`
- 关联文档：
  - `docs/整理归档/N2改进计划_多窗口特征拼接.md`（多窗口主线方案，已完成）
  - `docs/整理归档/N2改进计划_ilr_alr.md`（log-ratio 路线，已判负向）
  - `docs/AI_CONTEXT_GUIDE.md`（外部审查材料）

---

## 1. 文档目的

本文档基于「当前 ML 代码现状 + 联网文献检索」两条证据线，系统梳理传统 ML 主线的可改进方向，作为后续实验迭代的长期参考。它不规定执行顺序，只给出每个方向的依据、改动面、预期与风险。具体的执行序列见 §6，待决策项见 §7。

ML 主线当前已强通过验收（N2 从 full baseline 0.2173 提升到 0.7121），本文档的核心问题是：**在 multiwindow ridge 已达天花板的情况下，下一个增益点在哪里。**

---

## 2. 当前 ML 主线的事实基线

以下事实来自 `src/ml/` 代码探索，是判断改进方向的客观起点。

### 2.1 模型与训练

| 维度 | 现状 | 代码位置 |
|---|---|---|
| 主线模型 | 纯 numpy 闭式解 Ridge，`(XᵀX + αI)⁻¹Xᵀy`，多输出一次解出 | `src/ml/models.py:41` |
| 正则强度 | `alpha=1.0` 硬编码，**无 CV、无 RidgeCV、无 LOO 选 alpha** | `multiwindow_n2.json:53`、`models.py:64` |
| 截距处理 | `fit_intercept=True`，截距项不参与正则 | `models.py:67` |
| 标准化 | Ridge 内部 `standardize=True` 按列 z-score；slow 可选外部 scaler（主线未用） | `models.py:48`、`features.py:75` |
| 可选模型 | `mean`（均值基线）、`dynamic_stacking_svr`（RBF-SVR + 蒙特卡洛动态加权，唯一用 sklearn 的模型） | `models.py:21,108` |
| **缺失模型** | 无 KernelRidge / GaussianProcess / PLS / 树模型（RF/XGBoost/GBDT）/ Lasso / BayesianRidge | — |

关键判断：`alpha=1.0` 是未经搜索的硬编码值。420 维多窗口拼接下，正则强度对 train/test/extrapolation 的偏差-方差平衡影响显著，这是一个**几乎零成本、尚未尝试**的改进点。

### 2.2 特征工程（最大改进面）

| 模态 | 当前特征 | 缺失 |
|---|---|---|
| 慢变量（T/P/RH/流量/活塞位等 8 通道） | 7 个纯时域统计量：mean/std/min/max/last/delta/slope | 物理派生量（如声速理论值、密度估算） |
| 超声波形 | 帧级 6 个幅度统计（mean/std/mean_abs/max_abs/energy/peak_index）→ 再 7 时域统计 | **TOF / 声速 / FFT 频谱 / 小波包能量** |
| 光纤麦克风波形 | 同超声 | FFT / 谱质心 / 频带能量 / 小波 |

- 时域统计定义：`DEFAULT_SEQUENCE_STATISTICS = (mean,std,min,max,last,delta,slope)`（`features.py:15`），`slope` 为最小二乘斜率，`delta = last − first`。
- 波形特征两步降维：`waveform_stat_features`（`features.py:174`）→ `_waveform_frame_descriptors`（`features.py:236`），`peak_index = argmax(|w|)/N`，`energy = mean(w²)`。
- **波形侧完全丢失频域信息**：原始 int16 波形被压成幅度统计，TOF、频移、衰减模式等对介质组成敏感的物理量均未显式提取。

### 2.3 多窗口拼接

- 配置 `windows: [null, {phase:exposure}, {phase:recovery}]`（`multiwindow_n2.json:54`），`null` 代表 full。
- 装配：每窗口约 140 维，横向 `np.concatenate` 得 ≈ 420 维，特征名加前缀 `full|` / `ph_exposure|` / `ph_recovery|`（`features.py:116,128`）。
- 不变量：三窗口的 `y` / `label_names` / `sequence_ids` 必须一致（`features.py:132` 有断言）。
- 互斥：`windows` 不得与 `window` / `protocol` 同时出现，仅 ML 可用（`experiment_config.py:294`）。

### 2.4 评估现状与缺口

- 指标：R²、RMSE、MAE 均为 numpy 版，**逐组分单列**计算（`metrics.py:9,29`），非 pooled。
- 划分：train/val/test/extrapolation 四集，按 `mixture_id` 业务主键预固定（`splits.py`），代码只读不算。
- **闭包误差缺口**：ML 侧 `sum_abs_error`（四组分预测和与 100% 的偏差）**恒为空字符串**（`run_experiment.py:318`），只有 DL 侧填该列（`run_experiment.py:343`）。即 ML 路径上**未量化闭包误差**，无法判断当前 ridge 预测是否满足浓度和约束。

### 2.5 已验收基线（对照基准）

`outputs/archive/multiwindow_n2_20260612`，test split：

| 指标 | full baseline | multiwindow | 变化 |
|---|---:|---:|---:|
| N2 R² | 0.2173 | **0.7121** | +0.4948 |
| overall R² | 0.7968 | 0.9253 | +0.1285 |
| macro RMSE | 3.9810 | 2.4133 | −1.5677 |

- extrapolation N2 R² = 0.7247（margin +0.0126，过线）。
- 验收阈值（`N2改进计划_多窗口特征拼接.md §5`）：N2 gain ≥ 0.10、其他组分 drop ≤ 0.05、extrapolation margin ≥ −0.10。所有后续方案须对照 `ridge_all_modalities` 与 `ridge_multiwindow_all_modalities` 两个 run 判定。

---

## 3. 文献证据

联网检索（2026-06-16）给出三条与本改进直接相关的硬证据。

### 证据 1：超声 TOF / 声速对 H₂-CH₄ 组成极其敏感

- H₂ 声速 ≈ 1304 m/s，CH₄ ≈ 430 m/s（STP），二者差异近 3 倍。
- 声速是混合比的**近线性敏感函数**，主流方法直接用超声飞行时间（ToF）测氢掺混比。
- 来源：[Ultrasonic ToF of Hydrogen Blending Ratios (IJOE 2025)](https://www.sciencedirect.com/science/article/pii/S0360319925051390)、[Speed of Sound in Gas Mixtures (DIVA)](https://www.diva-portal.org/smash/get/diva2:1003574/FULLTEXT01.pdf)、[ToF for Binary Gas Mixtures (IEEE)](https://ieeexplore.ieee.org/iel2/886/6033/00234173.pdf)、[ML Prediction of Speed of Sound (PMC12630880)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12630880/)。

**对本项目的含义**：当前代码从超声波形只取幅度统计，**完全未提取 TOF/声速**。声速 → 混合比是文献验证过的强信号路径，对全部四组分都有用。把 H₂ 的强信号显式化后，可借助「四组分和=100%」的闭包约束反推 N2。这是当前**最大的低垂果实**。

### 证据 2：N₂ 是本征难测的惰性气体

- N₂ 是同核双原子分子，无偶极矩变化，**TDLAS / 红外吸收无法直接测**。
- Raman 截面弱（N₂ 振动拉曼位移约 2330 cm⁻¹），需增强腔（cavity-enhanced / fiber-enhanced / 高压）才能达 ppb 级。
- 工业上普遍用**化学计量学间接推断**：从其他可测组分反推 N₂。
- 来源：[Ultrasensitive Raman for N₂ (PMC11412228)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412228/)、[Cavity-Enhanced Raman (ACS)](https://pubs.acs.org/doi/10.1021/acs.analchem.2c05432)、[TDLAS vs Raman (Beamonics)](https://tdlas-and-raman-spectroscopy-how-two-optical-methods-compare-for-gas-analysis/)。

**对本项目的含义**：佐证归档文档「N2 天花板需物理特征或接受固有上限」的判断。**N2 的提升必须走间接耦合路径** —— 喂足 H₂/CO₂/CH₄ 的强信号与物理派生量，让模型从闭包约束反推 N2，而非试图直接增强 N2 的（本就微弱的）直接信号。

### 证据 3：光谱气体回归的标准工具是 PLS / Kernel-PLS，而非裸 Ridge

- PLSR 在有吸收干扰时，浓度反演误差可比多元线性回归好 5 倍。
- Kernel PLS / KernelRidge 处理非线性。
- 420 维多窗口拼接**必有强共线性**（composition 数据 + 多窗口重复特征），PLS 的潜变量投影天然处理共线性。
- 来源：[PLSR to Retrieve Gas Concentrations (PMC8009469)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8009469/)、[Kernel PLS non-linear calibration](https://www.sciencedirect.com/science/article/abs/pii/S0169743922002477)、[sklearn KernelRidge](https://scikit-learn.org/stable/modules/kernel_ridge.html)、[Evaluating ML in Two-Step Calibration (AMT)](https://www.amt.copernicus.org/articles/19/2923/2026/amt-19-2923-2026.pdf)。

**对本项目的含义**：当前主线是裸 Ridge，缺 PLS 这一光谱回归事实标准。PLS 的降维能力可能比单纯加 L2 正则更适合高维共线特征。

---

## 4. 改进方向（按投入产出比排序）

### 方向 A：物理派生特征（最高优先）

**核心思路**：在 `features.py` 的波形通道里显式提取**声速/ToF** 和**频谱特征**，而非只取幅度统计。

#### A1. 超声 TOF / 声速

- **依据**：证据 1。声速 → 混合比近线性强信号。
- **方法**：
  - 从原始超声波形提取峰值到达时间或过零点间隔，结合已知传播路径换算声速。
  - 若波形时间轴标定信息不全，可退而用「峰值位置 `peak_index` 的物理化」或「自相关周期」作为声速代理。
- **改动面**：扩展 `DEFAULT_WAVEFORM_FRAME_FEATURES`（`features.py:16`）或在 `_waveform_frame_descriptors`（`features.py:236`）新增分支。代码已支持按 config 传 `waveform_frame_features`，无需改主流程装配逻辑。
- **前置核对**：确认超声波形的时间轴采样率、收发换能器间距等物理标定参数（见 `src/sim` 物理建模），否则 ToF 无法换算为绝对声速。
- **预期**：对 N2 的间接耦合增益最大，是文献与归档文档共同指向的主改进面。
- **风险**：中。若波形时间轴标定缺失，需先补物理建模侧的标定元数据。

#### A2. 频谱特征（FFT / 小波包能量）

- **依据**：超声/光纤信号的频移、衰减模式对介质组成敏感；声发射信号常用小波包能量作特征。
- **方法**：
  - FFT 幅度谱的质心（spectral centroid）、带宽、指定频带能量。
  - 小波包分解（WPT）后各子带能量。
- **改动面**：同 A1，在波形特征分支新增。
- **参考实现**：[Wavelet Packet Coefficient Selection (PMC3231411)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3231411/)、[Wavelet Packet Energy in Python (GitHub)](https://github.com/biswajitsahoo1111/cbm_codes_open/blob/master/notebooks/Wavelet_packet_energy_features_python.ipynb)。
- **预期**：补足频域信息，与 A1 互补。
- **风险**：中。需控制特征维度膨胀，420 维已偏高，加频谱后可能需配合特征筛选或 PLS。

### 方向 B：模型层 —— alpha 搜索 + 非线性/PLS 对照（中优先）

#### B1. alpha CV（LOO / GCV）

- **现状**：`alpha=1.0` 硬编，无搜索（`multiwindow_n2.json:53`）。
- **方法**：Ridge 闭式解下可用 LOO 误差的解析形式（Allen's formula 或 GCV）选 alpha，无需引入 sklearn；或直接用 `sklearn.linear_model.RidgeCV`。
- **改动面**：`models.py` 新增 `RidgeCVRegressor` 并注册。
- **预期**：低成本基线微调，先建立「正则是否被低估/高估」的事实。
- **风险**：极低。

#### B2. PLS / KernelRidge 对照

- **依据**：证据 3。PLS 是光谱气体回归事实标准，处理高维共线。
- **方法**：引入 `sklearn.cross_decomposition.PLSRegression` 或 `KernelRidge(kernel='rbf')` 作为对照 run（不替换主线）。
- **改动面**：`models.py` 注册新回归器，走与 ridge 平行的 run。
- **依赖决策**：需先放行 sklearn 作为正式依赖（见 §7）。当前 `DynamicStackingSVRRegressor` 已用 sklearn，路径已开，但 `IMPLEMENTATION_PLAN P5` 仍标「待决」。
- **预期**：验证是否优于 Ridge；PLS 的潜变量降维可能更适配 420 维共线特征。
- **风险**：中（依赖决策 + 引入新模型族）。

### 方向 C：约束 / 闭包建模（中优先，针对 N2 间接推断）

**核心思路**：显式利用「四组分和 = 100%」的物理闭包约束，把 N2 从独立回归变为「由其他三组分 + 约束反推」。

#### C1. 前置：补 ML 侧 sum_abs_error 评估列

- **现状**：ML 侧 `sum_abs_error` 恒为空（`run_experiment.py:318`），闭包误差未量化。
- **改动**：在 ML 汇总分支填该列（参考 DL 侧 `run_experiment.py:343`）。
- **意义**：任何闭包相关改进都需此列量化，否则无法判定。
- **风险**：极低。

#### C2. 闭包后处理（N2 = 100% − Σ其余）

- **依据**：证据 2。N2 难直接测，工业上普遍间接推断。
- **方法**：先回归 H₂/CO₂/CH₄（强信号组分），N2 由约束反推。
- **与历史决策的关系**：归档「方案 B 分组分回归」当时判「暂不启动」，前提是 multiwindow 尚未强通过。**前提已变**：现在其余三组分预测可靠（drop=0），基于它们做减法的前提成立。
- **风险**：低。可作 N2 间接推断的理论上界参照。

#### C3. 约束最小二乘（constrained LS with sum-to-one）

- **依据**：组分回归的标准约束方法。
- **方法**：约束最小二乘，四输出加和=1 约束。
- **参考**：[Constrained LS Simplicial Regression (Statistics and Computing)](https://link.springer.com/article/10.1007/s11222-024-10560-z)、[Multivariate Mixture Regression for Constrained Responses (Bayesian Analysis)](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/A-Multivariate-Mixture-Regression-Model-for-Constrained-Responses/10.1214/22-BA1359.pdf)。
- **与 log-ratio 的区别**：约束 LS 比 ILR/ALR 温和，不在对数空间强制耦合，本项目未试过。

### 方向 D：窗口与特征选择（低优先，调参级）

#### D1. 窗口边界细化

- 归档「方案 C」：更窄/过渡段窗口。
- 优先级最低：已验收的多窗口足够强。

#### D2. 特征筛选

- 420 维下可加互信息 / 方差阈值筛选，缓解共线性。
- **冗余提醒**：若已上 PLS（B2），PLS 自带降维，此方向无必要。

---

## 5. 已排除方向（避免后人重试）

### log-ratio（ILR / ALR）

- **判定**：负向，不再投入。
- **依据**：formal_full 全窗口结果（`N2改进计划_ilr_alr.md`）——
  - `ridge_all_modalities`（原始）test N2 R² = 0.2173
  - `ridge_ilr_n2_first`（ILR）= 0.1058（更差）
  - `ridge_alr_ch4`（ALR）= 0.1058（更差）
- **原因**：log-ratio 把其他组分误差通过比值耦合进 N2 坐标，对「单独抬弱信号 N2」是负担。
- **注意**：方向 C3 的约束 LS 与 log-ratio 不同，前者不在对数空间强制耦合，未被排除。

---

## 6. 建议的实验序列

以下序列按「先建事实、再攻主增益、后做对照」组织。验收基准统一对照 `ridge_multiwindow_all_modalities`（test N2 R²=0.7121、extrapolation N2 R²=0.7247）。

| 序号 | 实验 | 改动量 | 风险 | 目的 / 预期 |
|---|---|---|---|---|
| 1 | alpha CV（LOO/GCV） | 极小 | 极低 | 建立正则是否被低估的事实，基线微调 |
| 2 | 补 ML 侧 sum_abs_error 列 | 小 | 极低 | 解锁闭包改进的量化基础 |
| 3 | 超声 TOF / 声速特征 | 中 | 中 | **N2 间接耦合主增益来源** |
| 4 | FFT / 小波包能量特征 | 中 | 中 | 补频域信息，与 3 互补 |
| 5 | PLS / KernelRidge 对照 | 中 | 中 | 验证是否优于 Ridge，处理共线性 |
| 6 | 闭包后处理（N2=100−Σ其余） | 小 | 低 | N2 间接推断的理论上界 |

**起手建议**：先做 1+2（纯零风险，建立 baseline 事实），再集中攻 3+4（主增益），5、6 作为对照。

**关键里程碑**：若 3+4 把 N2 推过 0.80，DL 主线（PhaseWindowTCN）的压力可大幅缓解 —— 归档文档判断「N2 0.8+ 需物理特征」，方向 A 正对应此路径。

**收束纪律**：保持与 DL 主线相同的克制 —— 若某方向无正向 N2 增益（gain < 0.10）或导致其他组分退化（drop > 0.05），停止扩展，不堆叠多个无效改进。

---

## 7. 待决策项

### 决策 1：是否引入 sklearn 作为正式依赖

- **影响**：方向 B2（PLS / KernelRidge）、方向 C3（约束 LS）。
- **现状**：`DynamicStackingSVRRegressor` 已用 sklearn（`models.py:108`），路径已开；但 `IMPLEMENTATION_PLAN P5` 标「SVR、RandomForest 可选依赖待决」。
- **不影响**：方向 A（物理特征）、B1（alpha CV 可纯 numpy）、C1/C2、D 均不依赖此决策。
- **建议**：若只做 alpha CV + 物理特征，可暂不引入；若要上 PLS / 约束 LS，需先拍板。

---

## 8. 参考资料

### 气体传感与浓度预测
- [Ultrasonic ToF of Hydrogen Blending Ratios (IJOE 2025)](https://www.sciencedirect.com/science/article/pii/S0360319925051390)
- [Speed of Sound in Gas Mixtures (DIVA Portal)](https://www.diva-portal.org/smash/get/diva2:1003574/FULLTEXT01.pdf)
- [ToF for Binary Gas Mixtures (IEEE)](https://ieeexplore.ieee.org/iel2/886/6033/00234173.pdf)
- [ML Prediction of Speed of Sound in Gas Mixtures (PMC12630880)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12630880/)
- [Analysis and Prediction of Hydrogen-Blended Natural Gas (IJOE 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0360319923062444)
- [In-Situ Concentration Measurement of Blended Hydrogen (ASME IPC 2024)](https://asmedigitalcollection.asme.org/IPC/proceedings/IPC2024/88582/V005T09A001/1210775)

### N₂ 检测难点
- [Ultrasensitive Raman Gas Spectroscopy for N₂ (PMC11412228)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11412228/)
- [Multiple Gas Detection by Cavity-Enhanced Raman (ACS)](https://pubs.acs.org/doi/10.1021/acs.analchem.2c05432)
- [TDLAS vs Raman for Gas Analysis (Beamonics)](https://tdlas-and-raman-spectroscopy-how-two-optical-methods-compare-for-gas-analysis/)
- [Raman Gas Sensing Technology Review (Sheffield Hallam)](https://shura.shu.ac.uk/35116/1/Majumder-RamanGasSensing(VoR).pdf)

### 回归方法学
- [PLSR to Retrieve Gas Concentrations (PMC8009469)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8009469/)
- [Spectral Non-linear Calibration with Kernel PLS](https://www.sciencedirect.com/science/article/abs/pii/S0169743922002477)
- [Kernel Ridge Regression (sklearn)](https://scikit-learn.org/stable/modules/kernel_ridge.html)
- [Evaluating ML in Two-Step Calibration (AMT Copernicus)](https://www.amt.copernicus.org/articles/19/2923/2026/amt-19-2923-2026.pdf)
- [Anisotropic KRR with Extrapolation (PhysRevC)](https://link.aps.org/doi/10.1103/PhysRevC.110.034322)
- [Beyond Linearity in Spectroscopic Data (Spectroscopy Online)](https://www.spectroscopyonline.com/view/beyond-linearity-identifying-and-managing-nonlinear-effects-in-spectroscopic-data)

### 波形 / 频域特征
- [Wavelet Packet Coefficient Selection for AE (PMC3231411)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3231411/)
- [Wavelet Packet Energy Features in Python (GitHub)](https://github.com/biswajitsahoo1111/cbm_codes_open/blob/master/notebooks/Wavelet_packet_energy_features_python.ipynb)
- [Analysis of AE Waveforms by Wavelet Packet (MDPI)](https://www.mdpi.com/2076-3417/15/15/8435)

### 组分 / 闭包约束回归
- [Multivariate Mixture Regression for Constrained Responses (Bayesian Analysis)](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/A-Multivariate-Mixture-Regression-Model-for-Constrained-Responses/10.1214/22-BA1359.pdf)
- [Constrained LS Simplicial Regression (Statistics and Computing)](https://link.springer.com/article/10.1007/s11222-024-10560-z)
- [Latent Variable Mixture Model for Composition-on-Composition (PMC12448131)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12448131/)
- [Log-Ratio Transformations in ML](https://medium.com/@nextgendatascientist/a-guide-for-data-scientists-log-ratio-transformations-in-machine-learning-a2db44e2a455)
