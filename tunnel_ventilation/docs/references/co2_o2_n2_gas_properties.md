# CO₂/O₂/N₂ 气体物性参数文献汇总

> 本文档汇集掘进通风场景所需的三组分纯组分及混合物性参数的文献来源。
> 标注 `[代码已有]` 的条目已在主线 `tv3/sim/generation/acoustic_physics.py` 中使用（来源 NIST @298.15 K），可直接复用。
> O₂ 参数来源见各表"来源"列；置信度标注于关键条目。
>
> **数据核实状态（2026-07-04 经 smart-search 多源验证）**：O₂ cp 经 NIST WebBook Shomate 方程手算确认；O₂ λ/η 经 NIST/CRC/Engineering ToolBox/NBS TN.350 交叉确认；O₂ 弛豫机制经 Bass 1990 JASA 公式确认；CO₂ NDIR 滤光片经 Umicore/MicroHybrid datasheet 确认。
> **勘误（2026-07-24）**：① N₂ V-T 弛豫频率修正为 dry air fr,N ≈ 9 Hz/atm（旧值 65 kHz/atm 错误约 4 个数量级；代码 `f_relax_n2_per_atm` 已同步修正）；② Bass 1990 DOI 修正为 10.1121/1.400176（旧引 10.1121/1.400476 实际指向 Brown 1991 音乐信号处理文献）。核验依据见 [声速法_N2-O2辨识_深度学习突破路径_综述.md](声速法_N2-O2辨识_深度学习突破路径_综述.md) §4-A3 / §7。

## 1. 声学参数

### 1.1 纯组分声速

| 气体 | c₀ (m/s, 300 K, 1 atm) | γ | M (g/mol) | cp (J/mol·K) | 来源 |
|------|------------------------:|----:|----------:|-------------:|------|
| CO₂ | ~270.3 (理想气计算) | 1.289 | 44.01 | 37.13 | [代码已有] NIST @298.15K |
| O₂ | ~330.4 (理想气计算)；329 @ 298K (查表) | 1.400 | 31.9988 | 29.38 | NIST WebBook (Chase 1998, NIST-JANAF)；Engineering ToolBox |
| N₂ | ~353.1 (理想气计算) | 1.400 | 28.01 | 29.12 | [代码已有] NIST @298.15K |

理想气声速公式：`c = sqrt(γ · R · T / M)`，其中 R = 8.314 J/(mol·K)。
CO₂ 和 N₂ 的 γ 和 M 已在 `acoustic_physics.py` 的 `hidden_sound_speed_v2` 中编码。

O₂ cp = 29.38 J/(mol·K) @ 298.15 K 来自 NIST WebBook Shomate 方程，与主线 N₂ 的 29.12 接近（双原子理想气理论值约 29.1 J/mol·K），符合预期。置信度：高（已验证）。

**CoolProp 交叉验证（2026-07-20）**：近常压干气混合声速与 CoolProp `HEOS` 偏差约 0.02%；H₂O 0→5 mol% 使 c 约 +2.83 m/s（当前代码声速通道未含水汽）。详见 [tv3_acoustic_simulation_fidelity_review.md](tv3_acoustic_simulation_fidelity_review.md) §4 / §5.1 与 [physics_references.md](../foundation/physics_references.md) §2.2.1。

**Shomate 方程手算验证**：NIST WebBook O₂ 参数（100–700K：A=31.32234, B=-20.23531, C=57.86644, D=-36.50624, E=-0.007374），t=T/1000=0.29815 代入 Cp = A + B·t + C·t² + D·t³ + E/t² = 31.32234 − 6.0322 + 5.1444 − 0.9676 − 0.0830 = 29.384 ≈ 29.38 ✓

### 1.2 弛豫参数

| 气体 | 弛豫类型 | 弛豫频率 f_r | 弛豫强度 λ_max | 来源 | 置信度 |
|------|----------|----------:|----------:|------|--------|
| CO₂ | V-T 弛豫 | 28 kHz/atm | 0.12 | [代码已有] `hidden_attenuation_v2` | 高 |
| O₂ | V-T 弛豫 | dry air (h=0) 下 **fr,O ≈ 24 Hz/atm**；受 H₂O 显著加速（V-V 传能），项目采样域内 ≈2.3–166 kHz，典型大气条件仍 ≪ 200 kHz | 对 200 kHz 贡献可忽略 | Bass, Sutherland, Zuckerwar 1990, "Atmospheric absorption of sound: Update", JASA 88(4), DOI:10.1121/1.400176 | 高（公式已验证）|
| N₂ | V-T 弛豫 | dry air (h=0) 下 **fr,N ≈ 9 Hz/atm**（2026-07-24 修正；旧值 65 kHz/atm 错误约 4 个数量级）；受 H₂O 催化后典型大气条件 ≲1.3 kHz | 0.004（经验值；MRS 线拟按 C_vib 重新推导，见 active MRS 计划） | Bass 1990 (DOI:10.1121/1.400176)；[代码已修正 2026-07-24] `hidden_attenuation_v2` | 高 |

CO₂ 的振动弛豫在 200 kHz 载波下可能产生显著的频率依赖衰减，需要特别关注。O₂ 和 N₂ 的振动弛豫频率远低于 200 kHz，对波形影响有限。

**O₂ 弛豫机制补充**（Bass 1990 JASA 公式）：

```
fr,O = (Ps/Pso) · [24 + 4.04×10⁴ · h·(0.02+h) / (0.391+h)]
fr,N = (Ps/Pso) · (T/T0)^(-1/2) · [9 + 280h · exp(−4.17·((T/T0)^(-1/3) − 1))]
```

其中 h 为水蒸气摩尔浓度（%），Ps 为大气压力，Pso 为参考压力（1 atm），T0=293.15 K。

- dry air (h=0)：fr,O = 24 Hz/atm，fr,N = 9 Hz/atm（均远低于 200 kHz）
- 潮湿空气：fr,O 随湿度上升但仍 ≪ 200 kHz
- 200 kHz 载波远高于 O₂/N₂ 弛豫峰，**O₂/N₂ 弛豫衰减贡献可忽略**

参考文献：
- Bass, Sutherland, Zuckerwar 1990, JASA 88(4), "Atmospheric absorption of sound: Update"
- Bass & Sutherland 2004, "Atmospheric absorption in the atmosphere up to 160 km", JASA, DOI:10.1121/1.1631937
- Shields & Lee, "Effect of Light Molecules on Vibrational Relaxation in Oxygen", J. Chem. Phys., DOI:10.1063/1.1725200
- Bass & Shields 1976, "Vibrational energy transition rates from recent sound absorption measurements in moist air and in moist nitrogen", JASA, DOI:10.1121/1.2003276

工程实现建议：alpha_o2 取 0 或仅保留经典吸收分量。

### 1.3 声衰减

CO₂ 在 200 kHz 下的声衰减系数显著高于 O₂ 和 N₂，是超声波形中 CO₂ 浓度的间接可观测信号。

| 气体 | α (dB/m, 200 kHz, 300 K) | 来源 | 置信度 |
|------|------------------------:|------|--------|
| CO₂ | 需根据 `hidden_attenuation_v2` 模型计算（alpha_classical + alpha_co2 弛豫，P=0.5 MPa 时弛豫峰移近 200 kHz） | [代码已有] §13.5 | 计算依赖 P/T/湿度 |
| O₂ | 经典吸收为主（alpha_classical ∝ f²），弛豫贡献可忽略（fr,O ≈ 24 Hz/atm ≪ 200 kHz）；具体值需根据 K_ref=1.84e-11 计算 | Bass 1990 JASA + [代码已有] §13.5 | 计算依赖 P/T |
| N₂ | 同上，经典吸收为主，弛豫峰 9 Hz/atm（Bass 1990；2026-07-24 修正，旧文误作 65 kHz/atm）远低于 200 kHz | Bass 1990 JASA + [代码已修正] §13.5 | 计算依赖 P/T |

> 200 kHz 下纯组分声衰减系数无简单查表值，需根据主线 `hidden_attenuation_v2` 公式（经典吸收 + 各组分弛豫）在给定 P/T/湿度条件下计算。编码后用单元测试覆盖典型工况。

## 2. 热导参数

### 2.1 纯组分热导率

| 气体 | λ (mW/m·K, 298 K) | 幂律指数 n | 来源 | 置信度 |
|------|-------------------:|----------:|------|--------|
| CO₂ | 16.6 | 0.87 | [代码已有] `acoustic_physics.py` NIST | 高 |
| O₂ | 26.4 | 0.80 | NIST Thermophysical Properties / CRC Handbook / Engineering ToolBox | 高（已验证）|
| N₂ | 25.8 | 0.77 | [代码已有] `acoustic_physics.py` NIST | 高 |

温度修正：`λ(T) = λ₀ · (T / T₀)^n`（与主线代码 `_hidden_lambda_mix` 一致）

O₂ λ @ 298K = 26.4 mW/m·K（NIST/CRC 共识中值，search 多源确认 26.3–26.5 @ 300K，Engineering ToolBox 给 26 @ 298K）。n = 0.80（diatomic gases 幂律指数 0.7–0.9，O₂ ~0.80；不同来源 0.77–0.88）。

### 2.2 动力粘度

| 气体 | η (Pa·s, 298 K) | 来源 | 置信度 |
|------|----------------:|------|--------|
| CO₂ | 1.491e-5 | [代码已有] `acoustic_physics.py` | 高 |
| O₂ | 2.058e-5 | NBS Technical Note 350 (Childs 1966), DOI:10.6028/nbs.tn.350；NIST REFPROP (Lemmon & Jacobsen) | 高（已验证）|
| N₂ | 1.781e-5 | [代码已有] `acoustic_physics.py` | 高 |

O₂ η = 2.058e-5 Pa·s 高于 N₂ 的 1.781e-5（O₂ 分子稍大但碰撞截面与质量影响综合使其粘度略高），符合双原子气体趋势。

### 2.3 Wassiljewa-Mason-Saxena 混合规则

二元交互参数 φ_ij 由 Wilke 1950 公式从 M 和 η 计算（`φ_ij = (1/√8)·(1+M_i/M_j)^(-1/2)·[1+(η_i/η_j)^(1/2)·(M_j/M_i)^(1/4)]²`），**无需查表**。

| i \ j | CO₂ | O₂ | N₂ |
|-------|----:|---:|---:|
| CO₂ | 1.000 | ≈0.709（计算） | [代码已有] |
| O₂ | ≈1.414（计算） | 1.000 | ≈0.934（计算） |
| N₂ | [代码已有] | ≈1.059（计算） | 1.000 |

> 上表 φ_ij 由 Wilke 公式手算给出示意值（注意 φ_ij ≠ φ_ji）。编码时直接实现公式，不硬编码。CO₂-N₂ 和 N₂-CO₂ 复用主线代码现有值。

### 2.4 O₂ 与 N₂ 热导率对比

O₂ 和 N₂ 的热导率在 298 K 下非常接近，这是 TCS 通道区分两者的主要物理限制。具体数值：

- λ_O₂ ≈ 26.4 mW/m·K（@ 298K，NIST/CRC 共识）
- λ_N₂ ≈ 25.8 mW/m·K（@ 298K，[代码已有]）
- 差值 Δλ ≈ 0.6 mW/m·K
- 相对差异 ≈ 2.3%

来源：NIST Thermophysical Properties / CRC Handbook / Engineering ToolBox（search 多源交叉确认：N₂ 25.9–26.0，O₂ 26.3–26.5 @ 300K）。

> 与文档早期估计 ~2% 一致。TCS 通道对 O₂/N₂ 辨识力为边际水平（Δλ ≈ 0.6 mW/m·K），作为声学通道的独立补充信息源。

## 3. 光学参数

### 3.1 CO₂ NDIR 吸收

| 参数 | 值 | 来源 | 置信度 |
|------|------|------|--------|
| 基频吸收带 | ν₃ 反对称伸缩，2349 cm⁻¹ (4.26 μm) | HITRAN | 高 |
| HITRAN 分子编号 | 2 | HITRAN2020 | 高 |
| 典型 NDIR 滤光片中心 (CWL) | 4.26 μm (4265 nm) ± 0.5% | Umicore 4.26µm CO2 NBP datasheet | 高（已验证）|
| 典型 NDIR 滤光片半宽 (FWHM/HBW) | 窄带 60–120 nm，宽带 ~180 nm | Umicore HBW 105±10 nm；MicroHybrid MTS4SENS44-3C-1 HBW 120±10 nm (CWL 4265±25 nm)；MicroHybrid MTS4SENS44-3C-2 HBW 60±10 nm (CWL 4415±30 nm) | 高（厂商 datasheet 已验证）|

主线代码 `spectral/defaults.py` 中滤光片中心为 2347 cm⁻¹（占位值），FWHM 当前为占位值，编码时应依据目标厂商 datasheet 确认最终采用值（典型窄带 100 nm 量级）。

### 3.2 O₂ 红外活性

O₂ 为同核双原子分子，无永久偶极矩，基频振动模式（1580 cm⁻¹）不产生红外吸收。NDIR 方法对 O₂ 无效。

O₂ 的可选光学检测方式（非本阶段范围）：
- 顺磁检测（paramagnetic）
- 荧光猝灭（fluorescence quenching）
- 电化学（electrochemical）
- 调谐二极管激光（TDLAS，A-band 760 nm）

### 3.3 N₂ 红外活性

N₂ 同样为同核双原子分子，无红外活性。NDIR 和 HITRAN 方法均不适用。

## 4. 摩尔质量与混合规则

| 气体 | M (g/mol) | 来源 |
|------|----------:|------|
| CO₂ | 44.010 | IUPAC |
| O₂ | 31.9988 | NIST WebBook |
| N₂ | 28.014 | IUPAC |

混合气摩尔质量：`M_mix = Σ(x_i · M_i)`

混合气比热比：理想气近似 `γ_mix = Σ(x_i · Cp_i) / Σ(x_i · Cv_i)`

## 5. 来源索引

| 编号 | 来源 | 覆盖内容 | 置信度 |
|---:|---|---|---|
| [1] | NIST WebBook (webbook.nist.gov), Chase 1998 NIST-JANAF Thermochemical Tables | O₂/CO₂/N₂ 气相热化学：cp（Shomate 方程）、γ、标准熵 | 高（O₂ cp 手算验证）|
| [2] | NBS Technical Note 350, Childs 1966, "The viscosity and thermal conductivity coefficients of dilute nitrogen and oxygen", DOI:10.6028/nbs.tn.350 | O₂/N₂ 稀薄气体粘度与热导率系数 | 高 |
| [3] | CRC Handbook of Chemistry and Physics | 气体热导率、粘度、幂律温度指数 n | 高 |
| [4] | HITRAN2020 (hitran.org) | CO₂ 红外光谱参数（ν₃ 带，分子编号 2） | 高 |
| [5] | Bass, Sutherland, Zuckerwar 1990, "Atmospheric absorption of sound: Update", JASA 88(4), DOI:10.1121/1.400176（2026-07-24 修正；旧引 10.1121/1.400476 指向无关文献） | O₂/N₂ 振动弛豫频率公式（fr,O、fr,N） | 高（公式已验证）|
| [6] | Bass & Sutherland 2004, "Atmospheric absorption in the atmosphere up to 160 km", JASA, DOI:10.1121/1.1631937 | 大气吸收算法（高海拔扩展） | 高 |
| [7] | Mukhopadhyay, Das Gupta, Barua 1967, "Thermal conductivity of hydrogen-nitrogen and hydrogen-carbon-dioxide gas mixtures", British Journal of Applied Physics, DOI:10.1088/0508-3443/18/9/312 | 气体混合物热导率实验与 Wassiljewa 混合规则验证 | 高 |
| [8] | Wilke 1950, "A Viscosity Equation for Gas Mixtures", J. Chem. Phys. | Wilke 粘度混合规则（φ_ij 公式来源） | 高 |
| [9] | NIST Thermophysical Properties of Fluid Systems / NIST REFPROP (Lemmon & Jacobsen) | 气体热导率、粘度温度依赖（λ₀, n, η） | 高 |
| [10] | Engineering ToolBox, "Oxygen - Thermophysical properties" (engineeringtoolbox.com/oxygen-d_1422.html) | O₂ 物性速查（M、λ、cp、γ、声速） | 高（与 NIST 一致）|
| [11] | Umicore 4.26µm CO2 NBP datasheet (eom.umicore.com) | CO₂ NDIR 滤光片 CWL/HBW | 高（厂商 datasheet）|
| [12] | MicroHybrid MTS4SENS44-3C-1/3C-2 datasheet (shop.microhybrid.com) | CO₂ NDIR 多通道热电堆探测器滤光片规格 | 高（厂商 datasheet）|
| [13] | CoolProp 8.0.0（coolprop.org；HEOS 混合，GERG/Lemmon 系） | 干/湿 CO₂–O₂–N₂–H₂O 声速交叉验证；非正式编码默认 | 高（2026-07-20 本机实跑）|
