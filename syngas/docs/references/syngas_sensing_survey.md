# 合成气在线监测综述 — 商用系统、测量方法学与多模态融合可行性

> 检索范围：Exa（web）+ paper-search MCP（arXiv / Crossref / Semantic Scholar）+ 厂商官方公开资料。
> 检索时间：2026-06-25。
> 目的：为本项目（NDIR + 超声 + 光纤麦克风 + TCS 热导）从掺氢天然气向合成气场景扩展，做工业可行性与文献支撑评估。
> **置信度图例**：H = 厂商官方文档 / 国标全文 / 同行评议期刊；M = 多源转述一致；L = 单源、二手或商业供应商网页。

---

## 1. 摘要（先看这一段）

1. **合成气在线监测的工业主流是“NDIR (CO/CO₂/CH₄/CnHm) + TCD (H₂) + EC (O₂)”三合一模块化分析仪**。武汉四方光电 Gasboard-3100、MRU VARIOluxx SYNGAS、ESEgas IR-GAS-600P、ABYSS、Aquagas ABYSS、CUBIC Gasboard-3100P 都是这一架构的近似实现，**与本项目慢通道（V_NDIR_CH4、V_NDIR_CO2、V_TCS）严格同构**。这意味着把项目从掺氢天然气迁移到合成气，慢通道部分有充分工业先例，不是开新路。【H, 多源】
2. **高端 / 苛刻工况上有两条替代路线**：(a) TDLAS（Siemens SITRANS TDL、Yokogawa TDLS8000、Endress+Hauser SpectraSensors），ppb 级精度、in situ、抗水汽干扰；(b) 拉曼光谱（Endress+Hauser Raman Rxn5），可一次测全 H₂ / CO / CO₂ / CH₄ / H₂S / NH₃，对原料生煤气 / 重整器出口流耐液滴和颗粒污染。两条都比 NDIR+TCD 贵 5–20 倍。【H】
3. **同时使用 NDIR + 声学 + 热导的同行架构没有公开文献直接命中**。最接近的两个工作是：(a) Calgary 大学 2025 硕士论文用 TCD + 超声波 + ML 测掺氢天然气中 H₂；(b) Frontiers in Energy Research 2025 理论分析掺氢天然气中“热导 + 声速 / 密度”组合的可识别性。**项目把 NDIR 加进来组成四模态，是这条线的延伸，没有现成对手**。【H】
4. **CO 单组分多模态测量**在 2022–2026 的公开文献里也未见 NDIR + 光声 + 超声组合，光声路线（PAS / QEPAS / 光纤悬臂梁）做单组分 ppm–ppb 级 CO 是热点，但都是单模态。【H】
5. **国标层面**，GB/T 40789-2021《气体分析 一氧化碳含量、二氧化碳含量和氧气含量在线自动测量系统 性能特征的确定》是覆盖 CO/CO₂/O₂ 在线 AMS 性能指标的对应标准；GB/T 8984-2025 / GB/T 28124-2025 给微量 CO / H₂ 的 GC 法。HJ 系列里 HJ 1241-2022（固定污染源 CO + HCl CEMS 技术规范）是工业 CO 在线监测的执法依据。**专门覆盖“合成气在线监测”的整套国标目前未找到**，行业更多按工艺气分析仪 + 化工 SH 标准走。【H】
6. **本项目的目标精度 R² > 0.7（参考 Ridge baseline 0.71）远低于商用分析仪 ±1–2 % FS**。换算下来，商用系统对应的浓度回归 R² 通常远在 0.95 以上。**这说明项目的精度指标是“低端基线”而非“工业可用”，从合成气场景做产品化前要预留 1–2 个量级的提升空间**。【推断，基于精度指标量级换算】

---

## 2. 商用合成气在线监测系统对比

> 表 1：覆盖 5 组分（CO / CO₂ / CH₄ / H₂ / O₂）的多组分合成气分析仪。所有量程、精度均录自官方 datasheet 或厂商页面，未经第三方比对。

| # | 厂商 | 型号 | 测量原理 | 组分 / 量程 | 精度 | 响应 T90 | URL | 置信度 |
|---|------|------|----------|------------|------|----------|-----|--------|
| 1 | 四方光电（武汉） Cubic Sensor & Instrument | Gasboard-3100 / 3100P / 9031EX（防爆） | NDIR (CO/CO₂/CH₄/CnHm/C₂H₂/C₂H₄) + TCD (H₂) + ECD (O₂) | CO/CO₂/CH₄/H₂: 0–100 %；O₂: 0–25 %；CnHm: 0–10 % | CO/CO₂/CH₄/CnHm: ±1 % FS；H₂/O₂: ±2 % FS | < 15 s (NDIR) | https://en.gassensor.com.cn/GasAnalyzer/info_itemid_287.html | H |
| 2 | MRU Instruments (DE) | VARIOluxx SYNGAS（便携） | NDIR (CO/CO₂/CH₄) + TCD (H₂) + EC (O₂, H₂S) | CO/CO₂/CH₄/H₂: 0–100 %；O₂: 0–25 %；H₂S: 0–5000 ppm | CO: ±0.1 % 或 ±2 % 读数；CO₂: ±0.3 % 或 ±2 %；CH₄ / H₂: ±0.1 % / ±2 %；O₂: ±0.2 % | 未在 datasheet 中明示 | https://mru-instruments.com/wp-content/uploads/2023/03/MRU_Brochure_VARIOluxx-SYNGAS_ENGLISH_WEB-1.pdf | H |
| 3 | ESEgas (CN) | IR-GAS-600P（便携合成气） | NDIR (CO/CO₂/CH₄/CnHm) + TCD (H₂) + EC (O₂) | CO/CO₂/H₂/CH₄: 0–100 %；CnHm: 0–10 %；O₂: 0–25 % | CO/CO₂/CnHm/CH₄: ±2 % FS；H₂/O₂: ±3 % FS | T90 < 15 s (NDIR) | https://esegas.com/product/portable-syngas-analyzer-ir-gas-600p/ | H |
| 4 | Siemens (DE) | ULTRAMAT 23 / SIPROCESS GA 700（CALOMAT 7 + ULTRAMAT 7） | NDIR + UV + EC (ULTRAMAT 23)；TCD (CALOMAT 7) + NDIR (ULTRAMAT 7) + 顺磁 O₂ (OXYMAT 7)（GA 700） | ULTRAMAT 23：4 组分同时；GA 700：模块化 H₂、CO、CO₂、CH₄、O₂ 等 | 未在公开摘要给统一数值，按 EN 15267 QAL1 认证 | T90 一般几秒（GA 700 模块如 OXYMAT 7 = 1.9 s） | https://www.siemens.com/en-us/products/process-analytics/ultramat-23/ ；https://cache.industry.siemens.com/dl/files/345/109794345/att_1057307/v1/GA700_iPDF_en.pdf | H |
| 5 | ABB (CH/SE) | EL3000 系列（EL3020 / EL3040），Uras26 + Caldos27 + Magnos206 + 可选 ZO23 | NDIR (Uras26) + TCD (Caldos27) + 顺磁 O₂ (Magnos206) | Caldos27 二元混合：CO in H₂ 0–3 vol % / 99–100 vol %；H₂ in CO 0–11 vol %；CH₄ in H₂ 0–14 vol %；CO₂ in H₂ 0–13 vol %；H₂ in N₂ 0–11 vol % | EN 15267 QAL1 认证，按用户范围确定 | 未在该摘要明示 | https://library.e.abb.com/public/359441019c18638cc1257b0c00546b88/10-24-410-09-EN.pdf | H |
| 6 | Yokogawa (JP) | TDLS8000（in situ TDLAS） | TDLAS | O₂: 0–1 %–0–25 %；CO: 0–200 ppm 至 0–10000 ppm 或 0–50 %；CO + CH₄；CO + CO₂；CO₂: 0–1 % 至 0–50 %；NH₃、H₂O、HCl 等 | SIL 2 单机 / SIL 3 双机；具体精度按组分和路径长度而定 | TDLAS 典型亚秒级 | https://www.yokogawa.com/solutions/products-and-services/measurement/analyzers/gas-analyzers/tunable-diode-laser-spectrometer/tdls8000-tunable-diode-laser-spectrometer/ | H |
| 7 | Endress+Hauser (CH) | Raman Rxn5 + OptoDRS（生煤气原气） | 拉曼光谱 | H₂ 25–45 mol %、CO 30–50 mol %、CO₂ 10–30 mol %、CH₄ 0–10 mol %、H₂S/NH₃ ≥0.1 mol %；典型精度 k=2 时 0.01–0.03 mol % | k=2 precision: H₂ 0.02、CO 0.02、CO₂ 0.03、CH₄ 0.01 mol % | 未在摘要给出 | https://bdih-download.endress.com/files/DLA/005056A500261EEC9A988F25D8B90A7F/AI01336CEN_0121.pdf | H |
| 8 | ABB (CH/SE) | ACF5000（FTIR + FID + 顺磁 O₂） | FTIR 多组分 + FID + ZrO₂ | 同时测 15 组分（含 CO、CO₂、CH₄ 及污染物）；按 EN 15267 认证 | 未在摘要明示数值精度 | https://new.abb.com/products/measurement-products/analytical/cga-system-solution/acf5000 | H |
| 9 | Aquagas / Ankersmid (AU/EU) | ABYSS / ABY SynGas（NDIR + TCD + EC） | NDIR (CnHm/CH₄/CO/CO₂) + TCD (H₂) + EC (O₂) | CnHm 0–5 %；CH₄ 0–10 %；CO₂ 0–20 %；CO 0–40 %；H₂ 0–5 %；O₂ 0–25 %（不同型号量程差异较大） | 2 % FS | < 15 s (NDIR) | https://www.aquagas.com.au/environmental-monitoring-australia/emissions-and-process-gas/cems-and-process-gas/abyss-syngas-analyser/ ；https://ankersmid.eu/wp-content/uploads/2021/04/10.2-ABY-SynGas-Analyzers.pdf | H |
| 10 | HORIBA (JP) | 51 系列防爆（EIA-51 + TCA-51 + MPA/PMA-51） | NDIR (CO/CO₂/CH₄) + TCD (H₂) + 顺磁 (O₂) | NDIR: 各组分 0.11 vol % – 100 vol %（或 50 ppm – 2000 ppm，T 型）；H₂ TCD: 0–10 vol % 至 0–100 vol %；O₂: 0–5 vol % 至 0–25 vol % | 未在摘要明示统一精度 | NDIR 通常 ≈ 10 s | https://www.horiba.com/usa/process-and-environmental/products/detail/action/show/Product/51-series-177/ | H |

> **观察**：第 1–3、9、10 行（NDIR + TCD + EC 模块化）与本项目慢通道架构一致；第 4 行 Siemens GA 700 是上述架构的模块化高端形态；第 5 行 ABB EL3000 给了非常关键的“二元混合量程上下限”表，能直接对照本项目掺氢 / 合成气切换时 TCD 通道的工作区间。

---

## 3. 学术文献（多模态融合 + 合成气 + 声学 / 光声）

### 3.1 直接相关（多模态融合气体测量）

**[1] Alcantara Costa, M. (2025).** *In-situ monitoring of hydrogen concentration using multimodal sensor fusion enhanced by machine learning*. Master's thesis, University of Calgary.
URL: https://ucalgary.scholaris.ca/items/d5b62caa-b021-44c6-9fbe-f6769a13257f
**与项目关系**：热导 + 超声 + ML 测掺氢天然气 H₂ 浓度；额外含温湿压补偿。**这是和本项目最接近的公开工作**，但仅两个模态，没有 NDIR 光学通道，组分目标只有 H₂ 一项。精度数值未在公开摘要给出。【H — 同校学位论文】

**[2] Mengoni, F.; Saponaro, N.; Aziz, M. et al. (2024).** *Rapid Communications: First experimental realization of a thermoacoustic-based flue-gas analyzer*. Journal of Sound and Vibration, S0022-460X(24)00334-1.
URL: https://www.sciencedirect.com/science/article/pii/S0022460X24003341
**与项目关系**：首次实验验证热声学（onset 温差 + 谐振频率）测烟气 CO₂/O₂/N₂；类比于本项目“声学波形”作为浓度通道的合理性论证。**不直接做 NDIR + 声学融合，但提供了“声学单模态可测烟气组分”的实验证据**。【H】

**[3] Monsalve, J.M.; Schenk, H.A.G.; Stolz, M. et al. (2024).** *Rapid characterisation of mixtures of hydrogen and natural gas by means of ultrasonic time-delay estimation*. Journal of Sensors and Sensor Systems, 13, 179–192.
URL: https://jsss.copernicus.org/articles/13/179/2024/
**与项目关系**：MUT（微机械超声换能器）测掺氢天然气声速，分辨率 < 2 mol %（高端 ≈ 1 mol %）。该论文已与 GC 和密度传感器互比验证。【H】

**[4] Frontiers in Energy Research (2025).** *A theoretical assessment of the on-site monitoring of hydrogen-enriched natural gas by its thermodynamic properties*. Front. Energy Res., 13:1339598. DOI: 10.3389/fenrg.2025.1339598.
URL: https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2025.1339598/full
**与项目关系**：用 Monte Carlo 论证“热导 + 声速 / 密度”组合作为 PGC 的替代可行性，结论是当总测量不确定度 ≤ 1 % 时，可以稳健估计 H₂ 浓度与热值。**这给本项目热导 + 超声组合提供了理论上限边界**。【H】

**[5] Hanson, R.K. et al. (Stanford).** *Development of robust TDLAS sensors for combustion products at high pressure and temperature in energy systems* (multi-paper, 中试到工程规模煤气化炉，1 ton/day–50 tons/day).
URL: https://purl.stanford.edu/bq161vy0151 ；https://www.sciencedirect.com/science/article/abs/pii/S1540748912000193
**与项目关系**：在真实煤气化炉上 18 atm、1800 K、含颗粒散射条件下用 WMS-2f TDLAS 测 CO / CO₂ / CH₄ / H₂O，并以质量守恒推 H₂。3 s 时间分辨。**这是合成气场景的 TDLAS 黄金参考**，给本项目“工业级合成气在线”的精度天花板。【H】

### 3.2 间接相关（光声 / 光纤 / 多模态 ML）

**[6] Zhang, T.; Wang, W.; et al. (2024).** *All-Optical Photoacoustic Spectroscopy-Based Dual-Component Greenhouse Gas Analyzer*. Anal. Chem. 96(37): 14819–14825. DOI: 10.1021/acs.analchem.4c02440.
**与项目关系**：光声 + 光纤悬臂梁同时测 CO₂ 和 CH₄，CO₂ LOD 76.5 ppb / CH₄ LOD 1.9 ppb。**比 NDIR 强约 4 个数量级，但只测两组分**。【H】

**[7] PMC10682669 (2023).** *All-optical non-resonant photoacoustic spectroscopy for multicomponent gas detection based on aseismic photoacoustic cell*. 六组分（CO₂ / CO / CH₄ / C₂H₆ / C₂H₄ / C₂H₂）检测限分别 62.66 ppb / 929.11 ppb / 1494.97 ppb / 212.94 ppb / 1153.36 ppb / 417.61 ppb。
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10682669/
**与项目关系**：硅悬臂光纤麦克风做多组分 PAS，CO 单组分 LOD ≈ 0.93 ppm。**与本项目“光纤麦克风波形”路径完全同构，但人家用激光 + 多波长锁相，远比本项目的被动光声系统复杂**。【H】

**[8] Zhang, E.; Zhang, E. (2024).** *Development of A Multimodal Deep Feature Fusion with Ensemble Learning Architecture for Real-Time Gas Leak Detection*. IEEE ICMI 2024. DOI: 10.1109/icmi60790.2024.10585716.
**与项目关系**：多模态特征融合 + 集成学习做气体泄漏检测，方法论可借鉴；非合成气场景。【M — 会议论文】

**[9] Pignanelli, E.; Kuhn, K.; Schütze, A. (2011).** *Versatile gas measurement system based on combined NDIR transmission and photoacoustic spectroscopy*. IEEE SENSORS 2011, 813–816. DOI: 10.1109/icsens.2011.6127048.
**与项目关系**：**直接做 NDIR + PAS 双模态气体测量**，2011 年早期工作。和本项目“NDIR + 光纤麦克风（PAS 等效）”在原理上同构。【H — 老但极相关】

**[10] Jadhav, P. et al. (2025).** *Multimodal Gas Detection Using E-Nose and Thermal Images: An Approach Utilizing SRGAN and Sparse Autoencoder*. Computers, Materials & Continua 83(2): 3493–3517.
URL: https://www.techscience.com/cmc/v83n2/60537
**与项目关系**：E-Nose + 热成像融合做气体分类（4 类，准确率 97.89–98.55 %），属分类而非浓度回归；方法论可借鉴。【M — 期刊但研究气体非合成气】

**[11] PMC12630880 (2025).** *Machine learning-based prediction and SHAP sensitivity analysis of sound speed in hydrogen-rich gas mixtures*. URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12630880/
**与项目关系**：用 ML 预测 H₂-rich 混合气声速，SHAP 分析显示 H₂ 含量是声速第一驱动因素，CO₂ / 温度 / CH₄ 影响弱。**直接支持本项目“超声波模态对 H₂ 高度敏感”的物理假设**。【H】

---

## 4. CO 测量的核心挑战

### 4.1 NDIR 通道 CO / CO₂ 光谱串扰

**事实**：CO 的 NDIR 吸收带在 4.65 μm 附近，CO₂ 在 4.26 μm，**两者光谱相邻但中心波长可分**，主流商用 NDIR 都用窄带光学滤光片 + 多层探测器分离。在水汽 / 烃类共存时仍会引入读数偏置，**消除方式有三种**：(a) 商用做法：内置补偿探测器 + 出厂多组分交叉灵敏度矩阵；(b) 改用 TDLAS（窄线宽激光器，无串扰）；(c) FTIR + 多元算法（HORIBA MEXA-ONE-FT 路线）。【H — Beamonics 2024、Endress+Hauser、HORIBA datasheet】

**对项目的启示**：本项目慢通道是单束 NDIR，没有反向补偿。**若 V_NDIR_CH4 / V_NDIR_CO2 通道还要扩展到 CO，必须重新讨论光路与滤光片配置**，不能直接复用现有 CH4 + CO2 通道分时复用。

### 4.2 高 H₂ 浓度下的热导非线性

**事实**：TCD 在二元混合中近似线性，但合成气中 H₂ 与 CO、CO₂、CH₄ 同时存在，热导对组分组合是非线性的。Energies 18(4): 971 (2025)（MEMS-based TCD）明确指出：“thermal conductivity of gases varies with the square root of the temperature, …non-linear nature of the changes”，并且需要多组分查表 / 模型反演。**Cubic 和四方光电的 datasheet 都强调“H₂ 通道做了对 CO / CO₂ / CH₄ 的智能补偿”**，说明工业产品已用补偿算法处理这个问题。ABB Caldos27 / Siemens CALOMAT 7 都按“二元 + 一组背景组分”的方式预先配置量程区间。【H】

**对项目的启示**：项目 V_TCS 通道若要做合成气 H₂ 量化，**不能用简单线性回归，要么用 ML 隐式学习多元非线性，要么按工业做法做组分组合预校准**。Ridge baseline = 0.71 这个数字在合成气场景下不一定能直接迁移，**因为本项目掺氢天然气背景气主要是 CH₄，而合成气背景是 CO + CO₂ + N₂，TCD 响应函数完全不同**。

### 4.3 工业气样中粉尘 / 水汽 / 硫化物干扰

**事实**：所有主流合成气分析仪要求样气“无尘、无水、无油”（Cubic Gasboard-3100、ESEgas IR-GAS-600P、HORIBA 51 系列、ABB EL3000 都明示），需要电伴热取样管 + 多级过滤 + 冷凝除水 + 不锈钢 + PTFE 流路。高炉煤气场景下，深圳华谊环保 HY/QT-ZXS 方案采用“双流路自动切换 + 反吹除水”，西安聚能高炉炉顶煤气方案要求“样气含尘 ≤ 200 g/Nm³、含水 ≤ 30 % VOL”的取样条件后做预处理。【M — 多家厂商一致】

**对项目的启示**：本项目超声 + 光纤麦克风波形若进现场，**对粉尘 / 凝液耐受性是核心实验风险**。光纤麦克风（光声原理）对样气清洁度的要求与 NDIR 同级，**不会比 NDIR 更耐脏**。

---

## 5. 国标 / 行标对照

| 标准号 | 名称 | 与项目相关性 | 置信度 |
|--------|------|------------|--------|
| GB/T 40789-2021 | 气体分析 一氧化碳含量、二氧化碳含量和氧气含量在线自动测量系统 性能特征的确定 | **最直接**：CO / CO₂ / O₂ 在线 AMS 的性能特征定义、QAQC 程序、报告内容；抽取式 + 原位式都覆盖 | H |
| GB/T 8984-2025 | 气体分析 气体中微量一氧化碳、二氧化碳和碳氢化合物含量的测定 火焰离子化气相色谱法 | 微量 CO/CO₂/HC 测定的 GC 法，可作参考真值 | H |
| GB/T 28124-2025 | 气体分析 惰性气体中微量氢、氧、甲烷、一氧化碳含量的测定 氧化锆气相色谱法 | 微量 H₂/O₂/CH₄/CO 测定 GC-ZrO₂ 法；适用 N₂/Ar/He 等惰气背景，**直接迁移到合成气有保留** | H |
| HJ 1241-2022（征求意见稿名义） | 固定污染源废气 一氧化碳和氯化氢连续监测系统技术规范 | 工业 CO CEMS 的执法标准，含全程高温采样、零点 / 量程校准、系统响应时间检测；**项目要进入污染源监测场景必须对标** | H |
| GB/T 3394-2023 | 工业用乙烯、丙烯中微量一氧化碳、二氧化碳和乙炔的测定 气相色谱法 | 化工原料气中微量 CO/CO₂ 的 GC 法 | H |
| HJ 75 / HJ 76 | 固定污染源烟气 SO₂/NOₓ/颗粒物 CEMS 规范与方法 | 不直接含 CO/H₂，但 CEMS 的安装、校准、传输标准与 HJ 1241 同源 | H |

**关键空缺**：检索范围内**没有找到专门针对“合成气在线监测”整套系统的国标 / 行标**。工业实践按“工艺气分析仪 + 化工 SH/T 行标 + 厂内规范”落地，公开标准只覆盖单组分性能要求。【推断，基于现有检索结果，可能漏检 SH/T 系列化工行标】

---

## 6. 多模态融合可行性评估

### 6.1 同行架构对比

| 架构（按模态组合） | 是否有公开同行 | 代表工作 |
|-------------------|--------------|----------|
| NDIR + TCD + EC（电化学） | **有大量工业先例** | Cubic Gasboard-3100、MRU VARIOluxx、ESEgas IR-GAS-600P、HORIBA 51 系列、ABB EL3000、Siemens ULTRAMAT/CALOMAT、四方光电 9031EX |
| TCD + 超声波 + ML | **有，1 篇硕士论文** | Alcantara Costa 2025 [文献 1] |
| TCD + 声速 + 密度（理论） | **有 1 篇理论分析** | Frontiers in Energy Research 2025 [文献 4] |
| NDIR + 光声（PAS） | **有 1 篇较老的工程会议论文** | Pignanelli 2011 [文献 9] |
| 多组分光声 + 光纤麦克风 | **有，单模态高灵敏度** | Anal. Chem. 2024 [文献 6]、PMC10682669 [文献 7] |
| **NDIR + 超声 + 光纤麦克风（PAS） + TCD 四模态** | **未检索到直接同行** | — |

**结论**：项目的“NDIR + 超声 + 光纤麦克风 + TCS 热导”四模态组合，**在公开文献和商用产品中都没有完全同构的对手**。最接近的是把上述前三种两模态工作叠在一起。**这既是机会（差异化），也是风险（无先例可直接对标精度边界）**。

### 6.2 模态互补性物理分析

| 模态 | 对 CO 敏感性 | 对 CO₂ 敏感性 | 对 CH₄ 敏感性 | 对 H₂ 敏感性 | 主要价值 |
|------|------------|--------------|--------------|------------|----------|
| NDIR (4.6 μm CO, 4.26 μm CO₂, 3.3 μm CH₄) | 高（直接吸收） | 高 | 高 | 0（H₂ 无 IR 吸收） | 三大碳源直接量化 |
| 超声波（声速反演） | 中（声速对所有组分都受影响，但 CO 与 N₂ 接近，对 CO 分辨弱） | 中（CO₂ 比 CH₄/N₂ 重，声速降明显） | 中 | **极高**（H₂ 声速是 N₂ 的约 3.7 倍） | **H₂ 主通道** |
| 光纤麦克风（光声） | 中–高（取决于激发波长） | 中–高 | 中–高 | 0（同 NDIR） | 高灵敏度补强 |
| TCD（热导） | 弱（CO 与 N₂ 热导接近） | 弱 | 中 | **极高** | **H₂ 二通道，与超声构成冗余** |

**结论**：四模态对 H₂ 形成双通道（超声 + TCD），对 CO/CO₂/CH₄ 形成 NDIR + 光声双通道。**理论上 4 组分都至少有 2 个独立模态覆盖**，融合是合理的。**但 CO 与 N₂ 在超声和热导通道近似简并**，CO 主要靠 NDIR / 光声的 IR 吸收来定位，**这一点决定了 CO 是 4 组分中最难做高精度的组分**。【推断，基于物理常数】

---

## 7. 项目架构与工业现状的差距分析

| 维度 | 工业主流（合成气在线分析仪） | 本项目当前 | 差距 / 备注 |
|------|---------------------------|----------|------------|
| 测量原理 | NDIR + TCD + EC | NDIR + TCD + 超声 + 光纤麦克风 | 多两个被动声学模态，工业上没有先例；缺 EC 氧通道 |
| 精度（CO/CO₂/CH₄） | ±1–2 % FS（对应单点 R² ≫ 0.95） | R² > 0.7（Ridge 0.71） | **差 1–2 个数量级** |
| 精度（H₂） | TCD ±2–3 % FS；TDLAS 至 ppb | 同 4 组分一并 R² > 0.7 | 同上 |
| 响应时间 | T90 < 15 s (NDIR) / < 3 s (TDLAS) | 未在本次提示中明示 | 项目需明确给出 T90 对照 |
| 量程 | CO/CO₂/CH₄/H₂ 0–100 %；CnHm 0–10 %；O₂ 0–25 % | 掺氢天然气场景仅覆盖 H₂ 0–30 %，迁移到合成气需重定标 | **必须做量程重新校准** |
| 样气预处理 | 全程高温电伴热 + 多级过滤 + 除水 + PTFE 流路 | 实验室级 | 工业部署需大规模工程化 |
| 防爆 / 认证 | Ex d IIC T4 Gb；EN 15267 QAL1；SIL 2/3 | 无 | 商用化需补 |
| 国标对标 | GB/T 40789-2021；HJ 1241-2022 | 未对标 | 项目立项前要明确 AMS 性能特征 |
| 多模态融合 | **几乎全部是 NDIR + TCD + EC 的“硬规则交叉补偿”**，不是 ML 融合 | ML 融合（Ridge / 后续 DL） | **这是项目唯一的差异化空间** |
| 工业落地 H₂ 量化 | TCD 二元背景 + 智能补偿（厂商专利） | 同上目标 | 工业方法已成熟，**项目需证明 ML 比规则补偿好** |

**总评**：本项目从“可行性”维度看，慢通道（NDIR + TCD）有充分工业先例；超声与光纤麦克风两个声学模态，理论支撑充足但**没有人在合成气场景上做过同样的四模态组合**。**项目的核心论证应该是“四模态 ML 融合 vs. 三模态规则补偿”的精度 / 鲁棒性对比**，而不是“能不能测”。

---

## 8. 建议下一步检索 / 落地

1. **补检 SH/T 化工行标**（石油化工和煤化工子系列），目前 web 检索未覆盖完整，CNKI 学位论文 + 行标库可能给到合成气专用规范。
2. **复制 Calgary 2025 硕士论文的实验设计**（[文献 1]）做内部对照基线，验证“TCD + 超声 + ML”在掺氢 / 合成气切换时的迁移性能。
3. **联系四方光电 / Cubic** 索取 Gasboard-3100 的“组分交叉补偿矩阵”技术白皮书，作为本项目 Ridge baseline 的工业对标。
4. **TDLAS 路线评估**：若项目想冲击工业级精度，需要把 NDIR 通道升级到 TDLAS（参考 Yokogawa TDLS8000 / Siemens SITRANS TDL），代价是激光器 + 调制 + 锁相系统。是否值得，要看产品定位。
5. **数据集层面**：项目当前掺氢天然气数据，迁移到合成气前，**TCD 通道的训练域几乎完全不重叠**（背景气从 CH₄-dominant 变成 N₂ + CO + CO₂-dominant），必须新建合成气标定气配置矩阵。

---

## 9. 检索局限与未验证项

- 本次检索未使用 CNKI 实际浏览器接口（需要 Chrome DevTools MCP，当前环境不可用），中文学术文献覆盖偏弱，**可能漏检“煤气化炉煤气在线监测”“高炉煤气多组分分析”这类中文期刊论文**。
- 表 1 中的精度数据全部来自厂商官方 datasheet，**没有独立第三方比对**；同一指标在不同实验室的实测值可能与 datasheet 有出入。
- 国标列表是基于 web 检索结果，**未检索到“合成气在线监测专用整体标准”不等于不存在**，可能在 SH/T、HG/T、JJG 系列中有遗漏。
- “未找到完全同构的四模态对手”这一结论的置信度为 **M**：检索覆盖了 arXiv / Semantic Scholar / Crossref / Exa 但未覆盖 IEEE Xplore、Scopus、CNKI、Web of Science 全量。**结论可能因这些缺失库的检索而被推翻**。
- 项目目标精度（R² > 0.7）与商用 ±1–2 % FS 的换算是基于“假设浓度变化覆盖 0–100 % 量程，且 FS 误差近似为 RMSE”的粗略推断，**严格对应关系需要项目方给出具体 RMSE 数值后重新换算**。
