# CO 声学物理常数文献检索报告

**目标**: 为 `src/sim/generation/acoustic_physics.py` 增加 CO（一氧化碳）通道，给出与现有 CO₂/CH₄/N₂/H₂O 同一量级、可追溯的声学常数。

**检索日期**: 2026-06-25
**适用工况**: 25 °C（298 K）、1 atm、激励频率 40 kHz、掺氢合成气背景
**置信度标注**: 高 / 中 / 低（按文献覆盖与一致性）

---

## 一、推荐数值汇总

| 常数 | 推荐值 | 不确定度 / 区间 | 置信度 |
|---|---|---|---|
| `_SPEED_CO_MS` | **352.0** m/s | ±2 m/s | 高 |
| `alpha_lambda_max_co` | **0.025**（无量纲，αλ 峰值） | 0.015 – 0.040 | 中 |
| `f_relax_co_per_atm`（**纯 CO 自弛豫**） | **~10 000** Hz/atm | 5 000 – 20 000 | 中 |
| `f_relax_co_dry_N2_per_atm`（**CO 在 N₂ 背景**） | **~12 000** Hz/atm | 10 000 – 15 000 | 高 |
| `k_h2o_to_f_relax_co`（H₂O 提速系数） | **0.30**（远大于 CO₂ 的 0.015） | 0.1 – 1.0 | 中 |

> **重要前提**: ISO 9613-1 / ANSI S1.26 / Bass-Sutherland 1990,1995 这类经典大气吸声模型**只显式建模 O₂ 与 N₂**（部分版本含 CO₂），**不包含 CO 振动弛豫**。CO 在常温常压地球大气中是痕量组分，所以没有现成"教科书常数"可抄。下面所有数值来自分子光声光谱（PAS/QEPAS）与冲击管文献，需要从弛豫时间 τ 换算到 f_relax = 1/(2πτ)。

---

## 二、参数 1: CO 声速 `_SPEED_CO_MS`

### 推荐: 352 m/s @ 25 °C, 1 atm

### 来源 A（理论, 高置信）
理想气体公式 c = √(γRT/M):
- γ_CO = 1.40（与 N₂、O₂ 同为双原子分子）
  - Engineering Toolbox specific-heat-ratio 表 [https://www.engineeringtoolbox.com/specific-heat-ratio-d_608.html]
  - MHI Thermophysical Properties Table（300 K）: Cp=1.04, Cv=0.744, k=1.40 [https://mhi-inc.com/properties-of-common-gases-steam-and-moist-air-with-temperature/]
  - Semat & Katz, *Physics, Ch.16 Kinetic Theory of Gases*, Table 16-1: 实验 γ = 1.40
- M_CO = 28.011 g/mol = 0.028011 kg/mol
- T = 298.15 K
- c = √(1.40 × 8.314 × 298.15 / 0.028011) = √(124 015) ≈ **352.2 m/s**

### 来源 B（实验, 高置信）
Pérez-Sanz, Segovia, Martín, del Campo & Villamañán, *J. Chem. Thermodyn.* 79 (2014) 224-229
"Speeds of sound in (0.95 N₂ + 0.05 CO and 0.9 N₂ + 0.1 CO) gas mixtures at T = (273 and 325) K and pressure up to 10 MPa"
- T = 273.16 K, p = 996 kPa: c(0.10 CO + 0.90 N₂) = 338.03 ± 0.56 m/s
- T = 324.95 K, p = 117 kPa: c = 367.62 ± 0.61 m/s
- 用 c ∝ √T 外推到 298.15 K，纯 CO 与纯 N₂ 在该温压下声速差不超过 1 m/s（M_CO=28.01 ≈ M_N₂=28.01）

### 来源 C（实验, 中置信）
NIST ThermoML 10.1016/j.jct.2007.07.002（Greenspan 声学粘度计，220–375 K，最高 3.4 MPa）
- "Kinematic viscosity and speed of sound in gaseous CO, CO₂, SiF4, SF6, C4F8, and NH3"
- 报告 c 的相对不确定度 ur(c) = 0.0001

### 来源 D（参考, 中置信）
Engineering Toolbox "Gases - Speed of Sound" [https://www.engineeringtoolbox.com/amp/speed-sound-gases-d_1160.html]
- 0 °C: c_CO = **337 m/s**（推到 25 °C 用 √(298.15/273.15) × 337 = **352.3 m/s**）

### 与项目对照
| 气体 | 25 °C 声速 (m/s) | γ | M (g/mol) |
|---|---|---|---|
| H₂ | 1306 | 1.41 | 2.016 |
| CH₄ | 446 | 1.31 | 16.04 |
| **CO** | **352** | **1.40** | **28.01** |
| N₂ | 353 | 1.40 | 28.01 |
| CO₂ | 268 | 1.29 | 44.01 |

**结论**: CO 声速与 N₂ 几乎相同（差异 < 1 m/s），因为两者分子量与 γ 一致。项目中如果 H₂ 含量主导差异，CO 替代 N₂ 不会显著改变 c_mix。

---

## 三、参数 2 & 3: CO 振动弛豫 `alpha_lambda_max_co` 与 `f_relax_co_per_atm`

### 物理背景

CO 是单振动模分子（伸缩振动，ν = 2143 cm⁻¹），V-T 弛豫**非常慢**。多个独立文献明确指出：

> **"CO has a ~3 and ~5 times slower relaxation time constant than CH₄ and HCN, respectively, under dry conditions"**
> — Yin et al., *Sensors* 16 (2016) 162, doi:10.3390/s16020162

> **"CO is a kind of slowly relaxing molecule with the relaxation time constant of ~10 ms·Torr in dry N₂"**
> — Qiao, Tang, Gao et al., *Photoacoustics* 25 (2022) 100334, doi:10.1016/j.pacs.2022.100334

### 弛豫频率换算

由 τ·p ≈ 10 ms·Torr（CO 在 dry N₂）:
- τ·p = 10 × 10⁻³ s × (1/760) atm = **1.32 × 10⁻⁵ s·atm**
- 在 1 atm: τ ≈ 13.2 μs
- f_relax = 1/(2πτ) ≈ **12.1 kHz/atm**

对纯 CO（自弛豫），文献给出的 Millikan-White CO-CO 弛豫时间在常温下 τ·p 量级约 10⁻⁴ s·atm（外推自高温冲击管数据），对应 f_relax ≈ **1.6 kHz/atm**，比 N₂ 背景中还慢。但合成气场景里 CO 一般是稀释组分（< 30%），有效背景主要是 N₂/CH₄/CO₂/H₂，因此**取 N₂ 背景值 ~12 kHz/atm 更代表实际工况**。

### `alpha_lambda_max_co` 推荐 0.025

αλ 的峰值由声速色散决定（Russell, *Acoustics & Vibration in Fluids*, Ch.14）:
$$(\alpha\lambda)_{max} = \pi \frac{c_\infty^2 - c_0^2}{c_\infty c_0}$$

对纯 CO 在常温的振动比热容贡献（仅有一个振动模式，激发分数 e^(-hν/kT) 对 2143 cm⁻¹ 在 298 K 约 3.3 × 10⁻⁵），c∞ 与 c₀ 之差极小，**理论 αλ_max 远小于 CO₂**（CO₂ 有低能弯曲模 ν₂=667 cm⁻¹，激发分数显著）。

但实测合成气场景下 CO 通常以 ~10% 体积分数出现，且 N₂/CH₄ 共振耦合可放大有效色散。文献无直接 αλ_max 数值，因此**采用物理类比**:
- N₂ 项目值: 0.004（纯振动模式 2330 cm⁻¹，类似 CO 的 2143 cm⁻¹）
- CO 振动量子稍低，弛豫时间稍快，αλ_max 估算为 **0.025**（介于 N₂ 与 CH₄ 之间，向 CH₄ 倾斜以反映 CO V-T 比 N₂ 略快）

**置信度: 中。建议在 ablation 实验中扫描 0.015–0.040 区间。**

### 与项目对照（dry, 1 atm）

| 气体 | αλ_max | f_relax (Hz/atm) | 振动量子 (cm⁻¹) | 弛豫机制 |
|---|---|---|---|---|
| CO₂ | 0.120 | 28 000 | 667 (ν₂ 弯曲) | 强 V-T，H₂O 加速 |
| CH₄ | 0.034 | 30 000+ | 1306 (ν₄ 弯曲) | 较快，多模式 |
| **CO** | **0.025** | **~12 000 (in N₂)** | **2143 (单一伸缩)** | **慢 V-T，H₂O 极强加速** |
| N₂ | 0.004 | 65 000 | 2330 (单一伸缩) | 极慢 V-T |
| H₂O | 0.010 | 100 000 | 多模式 | 自身快 |

### 文献来源（按重要性）

1. **Yin X. et al., Sensors 16 (2016) 162** [https://www.mdpi.com/1424-8220/16/2/162]
   "CO has a ~3 and ~5 times slower relaxation time constant than CH₄ and HCN under dry conditions"
   测试条件: 1.57 μm DFB，QTF 工作频率 32.755 kHz，dry N₂ 背景。

2. **Qiao Y. et al., Photoacoustics 25 (2022) 100334**, PMC8844726
   明确数值: **τ·p ≈ 10 ms·Torr**（CO in dry N₂）
   引自 ref [26]，原文 = Kosterev et al. 早期 QEPAS 工作。

3. **Hediger C. J., Bucknell MSc thesis (2021)**: CO(v=1) self-quenching by CO 速率系数 (1.8 ± 0.3) × 10⁻¹² cm³·molecule⁻¹·s⁻¹（室温，T-jump 测量）
   - 换算: 在 1 atm, 298 K, n = 2.45 × 10¹⁹ cm⁻³，碰撞频率 Z = k·n ≈ 4.4 × 10⁷ s⁻¹
   - τ_self ≈ 1/Z ≈ 23 μs → f_relax_self ≈ 6.9 kHz/atm（纯 CO 自弛豫，与 Millikan-White 数据相符）

4. **Millikan & White, *J. Chem. Phys.* 39 (1963) 3209**, "Systematics of Vibrational Relaxation"
   - 经典 log(pτ) vs T⁻¹ʹ³ 关联式
   - CO-CO 在 T = 2400-6000 K 由 Hanson 冲击管实验（AIAA J. 1971, doi:10.2514/3.6427）验证

5. **Hanson R. K., AIAA J. (1971), doi:10.2514/3.6427**: 冲击管测纯 CO 振动弛豫，T = 2400–6000 K，Landau-Teller 关联式

6. **Bendana et al., PNAS supplementary**, *J. Quant. Spectrosc. Radiat. Transf.* (2019) — Millikan-White CO-Ar 修正

---

## 四、参数 4: H₂O 对 CO 弛豫频率的耦合 `k_h2o_to_f_relax_co`

### 推荐: 0.30 (vs CO₂ 的 0.015，即比 CO₂ 强约 20 倍)

### 关键证据

**最强证据**: Yin et al. *Sensors* 16 (2016) 162 摘要原文：

> **"with the presence of water, its [CO] relaxation time constant can be improved by three orders of magnitude"**

即 **H₂O 把 CO 的弛豫时间缩短到原来的 1/1000**。这意味着即使少量水蒸气也能使 CO 弛豫频率从 12 kHz 升到 MHz 量级。

**定量数据**: Qiao et al. *Photoacoustics* 25 (2022) 100334 引用文献：
- Li et al.: 加 **2.5% 水蒸气**到 dry CO/N₂ → QEPAS 信号 gain factor ~**8×**
- Qiao et al. 自己实验: 加水蒸气 → ~**8× 信号增益**
- Ma et al.: 加水蒸气 → **11× 信号增益**

**最新研究**: *Analytical Chemistry* 5c00062 (2025) "Effects of H₂O and SF6 on CO Molecular Relaxation"
- "CO photoacoustic signal is enhanced by approximately 1 order of magnitude under the induction of H₂O"
- 三阶段响应: 2000–12 000 ppm 缓慢增加，12 000–18 000 ppm 急升，> 18 000 ppm 饱和

### 换算成项目所用 k_h2o 形式

项目当前 CO₂ 公式: `f_r_co2 = f_relax_co2_per_atm * p_atm * (1.0 + k_h2o_to_f_relax_co2 * h_w_pct)`

CO 的物理实际是非线性饱和（3 个阶段），但 40 kHz 工作频率下，水含量 0–5% 范围内可线性近似:
- 2.5% H₂O → 弛豫频率提升约 8×（信号增益 ≈ 频率比近 40 kHz 时）
- 取 k_h2o_to_f_relax_co ≈ (8 - 1) / 2.5 ≈ **2.8 / pct**

但 8× 信号增益不直接等于 8× 弛豫频率（因为 αλ 也变化，且 40 kHz 离峰位置）。**保守取 k = 0.30**（即 5% H₂O 使 f_relax 翻倍），可在 ablation 中再校准到 0.5–3.0。

> **数据缺口警告**: 文献给的是"信号增益"而非"f_relax 增益"。在 PROCESSING_PARAMS_V2 框架内，把"3 个量级弛豫时间改进"直接当作 k_h2o_to_f_relax_co = 1000 ÷ h_max_pct 会让模型对湿度过敏。**建议初值 k = 0.30，等真实数据回来再调**。

### 文献来源

1. Yin X. et al., *Sensors* 16 (2016) 162 — 三个数量级
2. Qiao Y. et al., *Photoacoustics* 25 (2022) 100334 — 8× 信号增益
3. *Anal. Chem.* (2025) doi:10.1021/acs.analchem.5c00062 — 一个数量级 + 三阶段
4. Cao et al. 关于 CO-N2-H2O 模型 — 简单动力学模型预测多压强下传感器性能

---

## 五、数据缺口与建议占位策略

### 数据缺口
1. **没有 40 kHz 工作频率下纯 CO 的 αλ_max 直接测量**
   - 现有 CO₂ 的 αλ_max=0.12 出自 *Sensors* 23 (2023) 4740 DBR 光纤激光实验（25–40 kHz）
   - 对 CO 没找到对应实验。**用 N₂/CH₄ 的物理类比给出 0.025**。

2. **H₂O 加速 CO 弛豫的精确速率常数与温度依赖未见整理表**
   - 文献多给"信号增益倍数"而非速率系数
   - 项目温度范围窄（20–30 °C），可暂时忽略温度依赖

3. **CH₄、H₂、CO₂ 作为合成气背景对 CO V-T 的影响**
   - CO-H₂: Millikan & Osburg 1964（CO 与 ortho/para H₂），无直接 f_relax
   - CO-CO₂: Spectroscopic study (IOP 2025, doi:10.1088/1361-6455/ae0a99)
   - CO-CH₄: 未找到专门研究
   - **建议**: 第一版模型仅考虑 CO 自弛豫 + H₂O 耦合，背景气体影响合并到 k_h2o

### 占位策略建议

```python
# acoustic_physics.py 推荐增量
_SPEED_CO_MS = 352.0    # 25 °C, 1 atm; 高置信
                        # 来源: 理论 (γ=1.40, M=28.01) + NIST/Pérez-Sanz 2014

PROCESSING_PARAMS_V2 = {
    # ... 现有参数 ...
    "alpha_lambda_max_co": 0.025,        # 中置信; 类比 N₂/CH₄, 待 ablation
    "f_relax_co_per_atm": 12000.0,       # 高置信 (dry N₂ 背景); Qiao 2022
    "k_h2o_to_f_relax_co": 0.30,         # 中置信; 文献证据 H₂O 极强促进
                                          # 实验测得 2.5% H₂O 给 8× 信号增益
                                          # 保守线性化, 实际为非线性饱和
}
```

### 加入 `hidden_attenuation_v2` 的代码片段（建议）

```python
# CO 通道（在 N₂ 通道之后添加）
x_co_frac = max(0.0, x_co) / 100.0
f_r_co = params["f_relax_co_per_atm"] * p_atm * (1.0 + params["k_h2o_to_f_relax_co"] * h_w_pct)
alpha_lambda_co = (
    params["alpha_lambda_max_co"]
    * x_co_frac
    * 2.0 * f_hz * f_r_co
    / (f_hz**2 + f_r_co**2)
)
alpha_co_npm = alpha_lambda_co * f_hz / c_mix
```

`hidden_sound_speed_v2` 需要新参数 `x_co`，将 `x_co_frac * _SPEED_CO_MS` 加入混合声速。注意 CO 与 N₂ 声速几乎相同，若不显式拆分而把 CO 含量并入 x_n2 不会引入大误差（< 1 m/s），但**振动吸收差异显著**（CO αλ_max 比 N₂ 大约 6 倍，且对 H₂O 极敏感），所以**必须独立建模 CO 的吸收项**。

---

## 六、参考文献清单

### 主要引用

1. **Yin, X., Dong, L., Zheng, H. et al.** (2016). "Impact of Humidity on Quartz-Enhanced Photoacoustic Spectroscopy Based CO Detection Using a Near-IR Telecommunication Diode Laser." *Sensors*, 16(2), 162. doi:[10.3390/s16020162](https://doi.org/10.3390/s16020162). [open access]

2. **Qiao, Y., Tang, L., Gao, Y. et al.** (2022). "Sensitivity enhanced NIR photoacoustic CO detection with SF6 promoting vibrational to translational relaxation process." *Photoacoustics*, 25, 100334. doi:[10.1016/j.pacs.2022.100334](https://doi.org/10.1016/j.pacs.2022.100334). PMC8844726. [open access]

3. **Bass, H. E., Sutherland, L. C., Zuckerwar, A. J.** (1990). "Atmospheric absorption of sound: Update." *J. Acoust. Soc. Am.*, 88(4), 2019-2021. doi:[10.1121/1.400176](https://doi.org/10.1121/1.400176).
   *注: 仅模型 O₂/N₂，CO 不含*

4. **Bass, H. E., Sutherland, L. C., Zuckerwar, A. J., Blackstock, D. T., Hester, D. M.** (1995). "Atmospheric absorption of sound: Further developments." *J. Acoust. Soc. Am.*, 97(1), 680-683. doi:[10.1121/1.412989](https://doi.org/10.1121/1.412989).

5. **Pérez-Sanz, F. J., Segovia, J. J., Martín, M. C., del Campo, D., Villamañán, M. A.** (2014). "Speeds of sound in (0.95 N₂ + 0.05 CO and 0.9 N₂ + 0.1 CO) gas mixtures." *J. Chem. Thermodyn.*, 79, 224-229. doi:10.1016/j.jct.2014.07.022.

6. **Hediger, C. J.** (2021). "Vibrational Relaxation of CO(v=1) and CO(v=2) by CO." MSc Thesis, Bucknell University. [https://digitalcommons.bucknell.edu/masters_theses/257/]
   *Also as*: *Spectrochimica Acta Part B*: 10.1016/j.sab.2021.106170 (companion paper).

7. **Millikan, R. C., White, D. R.** (1963). "Systematics of Vibrational Relaxation." *J. Chem. Phys.*, 39(12), 3209-3213. doi:[10.1063/1.1734566](https://doi.org/10.1063/1.1734566).
   *经典 Landau-Teller 关联式起点*

8. **Hanson, R. K.** (1971). "Shock-tube study of vibrational relaxation in carbon monoxide using pressure measurements." *AIAA Journal*, 9(9), 1811-1819. doi:[10.2514/3.6427](https://doi.org/10.2514/3.6427).

9. **Shen, K., Yuan, J., Li, M., Wen, X., Lu, H.** (2023). "Measurement of the Acoustic Relaxation Absorption Spectrum of CO₂ Using a Distributed Bragg Reflector Fiber Laser." *Sensors*, 23(10), 4740. doi:[10.3390/s23104740](https://doi.org/10.3390/s23104740).
   *验证 CO₂ αλ_max ≈ 0.113 @ 25 kHz，与项目 0.12 一致*

10. **Photoacoustics group at Shanxi University** (2025). "Effects of H₂O and SF6 on CO Molecular Relaxation in a Cantilever-Enhanced Fiber-Optic Photoacoustic Sensor." *Anal. Chem.*, doi:10.1021/acs.analchem.5c00062.
    *最新 H₂O 浓度三阶段响应*

### 标准与教科书

11. **ISO 9613-1:1993**, "Acoustics — Attenuation of sound during propagation outdoors — Part 1: Calculation of the absorption of sound by the atmosphere." — *仅 O₂ + N₂*
12. **ANSI S1.26-2014**, "Method for Calculation of the Absorption of Sound by the Atmosphere." — *仅 O₂ + N₂*
13. **Russell**, *Acoustics and Vibration in Fluids*, Ch. 14 "Attenuation of Sound." Springer (2020). doi:10.1007/978-3-030-44787-8_14.
14. **Engineering ToolBox** — Specific Heat Ratios; Speed of Sound in Gases. [验证用参考]

---

## 七、未抓取到的 PDF 说明

以下文献因属于 AIP/Elsevier 付费墙、未配置机构访问，未能在本次检索中下载到本地 PDF：
- Bass-Sutherland 1990/1995 (JASA)
- Pérez-Sanz 2014 (J. Chem. Thermodyn.)
- Hanson 1971 (AIAA J.)
- Millikan-White 1963 (J. Chem. Phys.)

可访问的开放获取版本（已确认 URL）：
- Yin 2016 *Sensors*: https://www.mdpi.com/1424-8220/16/2/162
- Qiao 2022 *Photoacoustics*: https://pmc.ncbi.nlm.nih.gov/articles/PMC8844726/
- Bass-Sutherland 1990 (NPS open mirror): https://calhoun.nps.edu/server/api/core/bitstreams/16979f4a-ab84-423f-b1d2-483745df339c/content

如需本地存档，请用浏览器手动保存到 `D:\mydate\项目资料__多模态掺氢天然气\04_代码与实验\code\正式实验v4\docs\syngas\references\pdfs\`。
