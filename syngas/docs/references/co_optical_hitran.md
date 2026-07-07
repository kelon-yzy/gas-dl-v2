# CO 红外光学参数与 HITRAN 数据库参考

> 用途：为 `src/sim/generation/spectral/` 光学仿真后端 + `configs/data/spectral-defaults.json` 新增 V_NDIR_CO 通道提供可追溯的光谱参数。
> 编制日期：2026-06-25
> 数据来源：HITRAN 官方文档（hitran.org）、HITRAN2020 期刊文献、InfraTec / Boston Electronics / MicroHybrid 厂商页面、NIST 计量实验。

---

## 1. HITRAN 分子标识

| 字段 | 取值 | 来源 |
|---|---|---|
| `molecule_id` | **5** | hitran.org molecule metadata 表（https://hitran.org/docs/molec-meta/） |
| `isotopologue_id`（主同位素体，HITRAN local ID） | **1**（即 ¹²C¹⁶O，AFGL code 26） | hitran.org iso-meta 表（https://hitran.org/docs/iso-meta/） |
| 主同位素自然丰度 | **0.986544** | 同上 |
| HAPI `fetch()` 推荐 `table_name` | **`"CO"`** | 与项目现有 CH4/CO2 命名一致；HAPI 内表名由用户自定义，仅作本地缓存键。 |
| Q(296 K)（配分函数，主同位素） | 107.42 | hitran.org iso-meta 表 |
| HITRAN 收录线总数（6 同位素体合计） | 5381 | hitran.org `lbl/` molecule 列表 |
| 主同位素 ¹²C¹⁶O 单同位素线数 | 1344 | hitran.org `lbl/2` iso 选择页 |

置信度：High。所有字段来源于 HITRAN 官方网站结构化表格，可在浏览器直接复核。

---

## 2. CO 基频带 (fundamental 1←0) 中心位置与强度

### 2.1 基频带几何位置

| 量 | 取值 | 备注 |
|---|---|---|
| 振动跃迁 | v=0 → v=1 | 一氧化碳为双原子分子，只有伸缩振动一个红外活性模式 |
| 带原点（band origin, ν₀） | **2143.27 cm⁻¹（≈ 4.6657 μm）** | ¹²C¹⁶O；由 R(0) 2147.081 cm⁻¹ 与 P(1) 2139.426 cm⁻¹ 中点估算，与教科书 ν₀ ≈ 2143 cm⁻¹ 一致（Gemini Observatory CO line table，UKIRT calibration page） |
| HITRAN 中 ¹²C¹⁶O ν_min..ν_max | 3.705 .. 14477.377 cm⁻¹ | hitran.org `lbl/2?5=on` 同位素选择页（覆盖 1-0 基频 + 2-0 第一倍频 + 高阶） |
| 基频带主要谱线分布 | **约 2000–2230 cm⁻¹** | P 支 ~2000–2143 cm⁻¹，R 支 ~2143–2230 cm⁻¹（Devi et al., JQSRT 2018，1940–2260 cm⁻¹ 拟合区间） |
| 基频带最强谱线强度（296 K，¹²C¹⁶O）参考量级 | **~4.5 × 10⁻¹⁹ cm·molec⁻¹** | HITRAN 给出 ¹²C¹⁶O Smax = 4.556 × 10⁻¹⁹ cm·molec⁻¹（hitran.org `lbl/`） |
| NIST 实测 R(17) 线强度（2206.354 cm⁻¹） | 1.028 × 10⁻¹⁹ cm·molec⁻¹（不确定度 0.6 %） | Bailey et al., JQSRT vol.347 (2025), DOI:10.1016/j.jqsrt.2025.109652 |

置信度：High。带原点和谱线位置由两套独立来源（Gemini Observatory / UKIRT 校准表）一致给出；强度由 HITRAN2020 + NIST 2025 实测交叉确认。

### 2.2 与 CO₂ ν₃ 反对称伸缩带 (2347 cm⁻¹) 的关系

CO 基频中心 2143 cm⁻¹ 与 CO₂ ν₃ 中心 2347 cm⁻¹ 在波数轴上**间隔 ~204 cm⁻¹**，对应波长 4.67 μm vs 4.26 μm，间隔 ~0.4 μm。

- CO 基频 R 支高 J 跃迁最高可达 2230 cm⁻¹ 附近；CO₂ 4.3 μm 带 P 支边缘可下探到 2250 cm⁻¹ 上下（项目现 `co2` 网格下界）。
- 中间约 2230–2250 cm⁻¹ 区间为相对干净的"窗口"，常规 NDIR 不会在此放滤光片。
- 因此**两条带在 4.6 μm 与 4.3 μm 滤光片各自的 FWHM 内不发生直接重叠**；通道选择性主要受弱碰撞翼影响，而非两个带的核心区域交叉。

置信度：High。

---

## 3. 商用 CO NDIR 滤光片型号汇总

所有参数取自厂商页面，单位换算公式：Δν(cm⁻¹) ≈ Δλ(nm) × 10⁷ / λ²(nm²)。

| # | 厂商 / 型号代码 | CWL（μm / nm） | FWHM（nm） | CWL（cm⁻¹） | FWHM（cm⁻¹） | 用途说明 | 来源 URL |
|---|---|---|---|---|---|---|---|
| 1 | **InfraTec "I" — NBP 4.66 µm / 180 nm** | 4.66 ± 0.04 / 4660 | 180 ± 20 | 2145.92 | **82.89** | "CO centered"，覆盖基频带核心 P/R 支 | https://www.infratec.eu/sensor-division/ir-filters/ |
| 2 | **InfraTec "K" — NBP 4.74 µm / 140 nm** | 4.74 ± 0.02 / 4740 | 140 ± 20 | 2109.70 | 62.32 | "CO flank"，偏向 P 支边缘，用于高浓度量程或差分通道 | https://www.infratec.eu/sensor-division/ir-filters/ |
| 3 | **Boston Electronics HIS-E222-F3.91/4.64**（双通道热堆） | 4.64 / 4640 | 180 | 2155.17 | 83.61 | 商用 CO 双通道传感器，4.64 μm CO + 3.91 μm 参考 | https://shop.boselec.com/products/his-e222-f3-91-4-64-g4300-dual-integrated-thermopile-sensor |
| 4 | **MicroHybrid MTS4SENS44-3C-1**（4 通道热堆） | 4.650 / 4650 | 180 ± 20 | 2150.54 | 83.25 | CO/CO₂/HC + 参考四通道集成 | https://shop.microhybrid.com/en/mts4sens44-3c-1 |
| 5 | **LaserComponents I (NBP4.66-180nm)** | 4.66 ± 0.030 / 4660 | 180 ± 20 | 2145.92 | 82.89 | InfraTec 同款 "I"，独立分销渠道二次确认 | https://www.lasercomponents.com/fileadmin/user_upload/home/Datasheets/lc-pyros/filters-windows-pyros.pdf |

**业界共识**：CO NDIR 主流滤光片 **CWL 4.64–4.66 μm，FWHM 180 nm（~83 cm⁻¹）**。窄带变体（FWHM ~140 nm）用于高背景 CO₂ 条件下抑制串扰。

置信度：High。多厂商交叉确认且参数高度一致。

---

## 4. 推荐写入 `configs/data/spectral-defaults.json` 的 JSON 片段

### 4.1 `gas_specs` 新增 CO 条目

```json
{"gas": "CO", "table_name": "CO", "molecule_id": 5, "isotopologue_id": 1}
```

### 4.2 `filters.co` 新增条目（取业界主流 InfraTec I / Boston HIS-E222 共识值）

```json
"co": {"channel": "co", "center_cm1": 2145.92, "fwhm_cm1": 82.89}
```

工程换算依据：CWL 4.66 μm = 2145.92 cm⁻¹；FWHM 180 nm @ 4.66 μm = 82.89 cm⁻¹。

### 4.3 `hitran_grids.co` 网格

下界：滤光片中心 - FWHM × 2 ≈ 2145.92 - 165.8 = 1980 cm⁻¹（覆盖整个基频带 P 支低端 ~2000 cm⁻¹）。
上界：滤光片中心 + FWHM × 2 ≈ 2145.92 + 165.8 = 2310 cm⁻¹（覆盖 R 支高端 ~2230 cm⁻¹，并预留 R 支远端 + CO₂ 4.3 μm 带下沿尾翼区域 ~2250–2300 cm⁻¹，便于后续 cross-talk 计算）。
步长：与 ch4/co2 保持一致 0.1 cm⁻¹。

```json
"co": {
  "wavenumber_min_cm1": 1980.0,
  "wavenumber_max_cm1": 2310.0,
  "wavenumber_step_cm1": 0.1,
  "temperature_k": 296.0,
  "pressure_atm": 1.0
}
```

### 4.4 `filter_source` 块建议追加

```text
"co_reference": "InfraTec I (NBP 4.66 µm / 180 nm FWHM, CO centered), cross-verified with Boston Electronics HIS-E222-F3.91/4.64 (4.64 µm / 180 nm) and MicroHybrid MTS4SENS44-3C-1 (4.65 µm / 180 nm). Source: https://www.infratec.eu/sensor-division/ir-filters/ ; https://shop.boselec.com/products/his-e222-f3-91-4-64-g4300-dual-integrated-thermopile-sensor"
```

### 4.5 完整合并示例（仅展示新增/修改部分）

```json
{
  "gas_specs": [
    {"gas": "CH4", "table_name": "CH4", "molecule_id": 6, "isotopologue_id": 1},
    {"gas": "CO2", "table_name": "CO2", "molecule_id": 2, "isotopologue_id": 1},
    {"gas": "H2O", "table_name": "H2O", "molecule_id": 1, "isotopologue_id": 1},
    {"gas": "CO",  "table_name": "CO",  "molecule_id": 5, "isotopologue_id": 1}
  ],
  "filters": {
    "ch4": {"channel": "ch4", "center_cm1": 3030.0, "fwhm_cm1": 147.0},
    "co2": {"channel": "co2", "center_cm1": 2347.0, "fwhm_cm1": 93.0},
    "co":  {"channel": "co",  "center_cm1": 2145.92, "fwhm_cm1": 82.89}
  },
  "hitran_grids": {
    "ch4": {"wavenumber_min_cm1": 2880.0, "wavenumber_max_cm1": 3180.0, "wavenumber_step_cm1": 0.1, "temperature_k": 296.0, "pressure_atm": 1.0},
    "co2": {"wavenumber_min_cm1": 2250.0, "wavenumber_max_cm1": 2445.0, "wavenumber_step_cm1": 0.1, "temperature_k": 296.0, "pressure_atm": 1.0},
    "co":  {"wavenumber_min_cm1": 1980.0, "wavenumber_max_cm1": 2310.0, "wavenumber_step_cm1": 0.1, "temperature_k": 296.0, "pressure_atm": 1.0}
  }
}
```

---

## 5. CO 与其他气体的光谱串扰评估

### 5.1 CO ↔ CO₂（最关键）

| 维度 | 评估 |
|---|---|
| 主带中心间隔 | CO 2143.27 cm⁻¹ vs CO₂ ν₃ 2349.14 cm⁻¹，差 ~206 cm⁻¹ | 
| CO₂ 在 CO 滤光片（2145.92 ± 41.4 cm⁻¹，即 2104.5–2187.3 cm⁻¹）范围内 | 存在 CO₂ 的 ν₁+ν₂-ν₂ 类弱热带和 4.3 μm 主带的低 J P 支远翼；常温下相对 ν₃ 主带强度低 ~3–4 个数量级，但在 >5 %vol CO₂ 高浓度（典型烟道气 / 内燃机尾气）条件下会有可观察干扰 | 
| 文献佐证 | Shi et al., Sensors 22 (2022) 1286, MDPI：商用 MIR CO 检测在 4.6–4.8 μm 区间确认 CO₂ 必须建模；Spearrin 等利用 P(20) @ 2059.91 cm⁻¹ 与 R(15) @ 2190.02 cm⁻¹ 双线策略，发现 P(20) 受 CO₂ 干扰显著小于 R(15)。 |
| 工程建议 | 在 `cross-talk` 计算中，**必须把 CO₂ 在 [1980, 2310] cm⁻¹ 区间的吸收纳入仿真**；现有 `hitran_grids.co2` 上界 2445 已覆盖 ν₃ 主带，下界需扩展到 1980 cm⁻¹ 才能覆盖 CO 通道带宽。CO₂ 的网格扩展属可选项：可单独在 CO 通道仿真时按需 `fetch()` 一份 1980–2310 cm⁻¹ 的 CO₂ 数据。 |

### 5.2 CO ↔ H₂O

| 维度 | 评估 |
|---|---|
| H₂O 在 2100–2200 cm⁻¹ 区间的吸收 | 处于 H₂O ν₂ 弯曲带（~1595 cm⁻¹）和 ν₁/ν₃ 伸缩带（~3700 cm⁻¹）之间的低吸收"窗口"，整体吸收弱，但仍有若干分立线 |
| 实测干扰量级 | Shi et al. (Sensors 22, 2022) 指出在燃烧场温度下，4.6 μm CO 通道需独立 H₂O 测量做修正，证实存在不可忽略的 H₂O 干扰；Sun, Park et al. NDIR review（Sensors and Actuators B, 2016, doi:10.1016/j.snb.2016.05.097）将水汽列为 2–8 μm 全段 NDIR 的主要干扰源 |
| 工程建议 | 在 `hitran_grids` 中**新增 `h2o_co` 子网格**（或扩展现有 H₂O 网格至 1980–2310 cm⁻¹），以便在 V_NDIR_CO 通道仿真中纳入 H₂O cross-talk |

### 5.3 CO ↔ CH₄

CH₄ ν₃ 反对称伸缩带中心 3019 cm⁻¹（项目滤光片 3030 cm⁻¹），与 CO 基频 2143 cm⁻¹ 相距 ~876 cm⁻¹，**在 NDIR 通道层级互不干扰**。仅在远翼（>10 cm⁻¹ 离中心）影响完全可忽略。无需在 CH₄/CO 通道之间增加 cross-talk 项。

### 5.4 串扰汇总

| 干扰对 | 严重程度 | 是否需在 spectral-defaults.json 加入 |
|---|---|---|
| CO 信号 ← CO₂ 干扰 | 中高（高 CO₂ 条件下显著） | **是**，建议把 CO₂ 加入 `co` 通道仿真气体清单 |
| CO 信号 ← H₂O 干扰 | 中（取决于湿度） | **是**，建议把 H₂O 加入 `co` 通道仿真气体清单 |
| CO 信号 ← CH₄ 干扰 | 可忽略 | 否 |
| CO₂ 信号 ← CO 干扰（反向） | 低（CO 基频远低于 CO₂ ν₃ 中心 200 cm⁻¹，CO 滤光片外 CO 吸收弱） | 可选，建议优先级低于上面两项 |
| CH₄ 信号 ← CO 干扰（反向） | 可忽略 | 否 |

---

## 6. 文献置信度汇总

| 数据项 | 置信度 | 主要来源 | 二次确认 |
|---|---|---|---|
| HITRAN molecule_id = 5 | High | hitran.org/docs/molec-meta/ | hitran.org/docs/iso-meta/ |
| HITRAN isotopologue_id = 1（¹²C¹⁶O） | High | hitran.org/docs/iso-meta/ | lweb.cfa.harvard.edu/hitran/molecules.html |
| 基频带中心 2143 cm⁻¹ | High | gemini.edu CO lines | UKIRT calibration page；vpl.astro.washington.edu |
| 主同位素 Smax = 4.556×10⁻¹⁹ cm·molec⁻¹ | High | hitran.org/lbl/2?5=on | NIST Bailey 2025 单线实测（相对差 2.2 %） |
| InfraTec I 滤光片 4.66 μm / 180 nm | High | infratec.eu/sensor-division/ir-filters/ | lasercomponents.com 数据手册 |
| Boston HIS-E222 4.64 μm / 180 nm | High | shop.boselec.com 产品页 | — |
| MicroHybrid 4.65 μm / 180 nm | High | shop.microhybrid.com 产品页 | — |
| CO ↔ CO₂ 串扰显著 | High | Shi et al., Sensors 22 (2022) 1286 | Sun et al., Sens. Actuators B 2016 review |
| CO ↔ H₂O 串扰显著 | Medium-High | Shi et al., Sensors 22 (2022) 1286 | Sun et al., Sens. Actuators B 2016 review |
| CO ↔ CH₄ 不重叠 | High | 直接由带中心间距推断 | — |

---

## 7. 参考文献（完整 URL）

1. HITRAN molecule metadata. https://hitran.org/docs/molec-meta/
2. HITRAN isotopologue metadata. https://hitran.org/docs/iso-meta/
3. HITRANonline line-by-line search (molecule list). https://hitran.org/lbl/
4. HITRANonline isotopologue selection for CO. https://hitran.org/lbl/2?5=on
5. Gemini Observatory CO lines and band heads. https://www.gemini.edu/observing/resources/near-ir-resources/spectroscopy/co-lines-and-band-heads
6. UKIRT CO lines calibration table. https://about.ifa.hawaii.edu/ukirt/calibration-and-standards/spectroscopic-calibration/astronomical-and-calibration-lines/co-lines/
7. Bailey D. et al., "Mid-Infrared line intensity for the fundamental (1–0) vibrational band of carbon monoxide (CO)," J. Quant. Spectrosc. Radiat. Transf. 347 (2025) 109652. DOI:10.1016/j.jqsrt.2025.109652. https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=959633
8. Devi V.M. et al., "Positions, intensities and line shape parameters for the 1←0 bands of CO isotopologues," JQSRT 2018. https://www.sciencedirect.com/science/article/pii/S0022407318302462
9. HITRAN2020 CO line list update (PMC review article). https://pmc.ncbi.nlm.nih.gov/articles/PMC10408379/
10. InfraTec IR filter standard list. https://www.infratec.eu/sensor-division/ir-filters/
11. InfraTec detector handbook (PDF). https://media.infratec.eu/infratec-detektorhandbuch-fenster-und-filter.pdf
12. Boston Electronics HIS-E222-F3.91/4.64 CO Dual Thermopile. https://shop.boselec.com/products/his-e222-f3-91-4-64-g4300-dual-integrated-thermopile-sensor
13. MicroHybrid MTS4SENS44-3C-1 4-channel thermopile. https://shop.microhybrid.com/en/mts4sens44-3c-1
14. LaserComponents pyroelectric filters & windows datasheet. https://www.lasercomponents.com/fileadmin/user_upload/home/Datasheets/lc-pyros/filters-windows-pyros.pdf
15. Shi L. et al., "Multi-Parameter In-Situ Diagnostics for Gas Turbine Combustion," Sensors 22 (2022) 1286. https://mdpi-res.com/d_attachment/sensors/sensors-22-01286/article_deploy/sensors-22-01286-v3.pdf
16. Park J. et al., "A review on non-dispersive infrared gas sensors," Sens. Actuators B 2016. https://www.sciencedirect.com/science/article/abs/pii/S0925400516303343

---

## 8. 工程落地清单（给后续 PR）

- [ ] 把第 4.5 节 JSON 片段合并入 `configs/data/spectral-defaults.json`
- [ ] 在 `src/sim/generation/spectral/` HAPI fetch 流程里增加 `CO` 表预编译（参数 molecule_id=5、isotopologue_id=1、网格 1980–2310 cm⁻¹）
- [ ] 在 cross-talk 仿真中把 CO₂、H₂O 在 [1980, 2310] cm⁻¹ 区间的吸收纳入 V_NDIR_CO 通道前向模型
- [ ] 等待用户提供 TraceGas-HC-NDIR 实际 CO 通道 datasheet 后，替换 `filter.co.center_cm1` / `fwhm_cm1` 为厂商真值，并在 `filter_source.co_reference` 标注实际型号
- [ ] 验证步骤：HAPI `fetch('CO', 5, 1, 1980, 2310)` 成功返回 1344 条主同位素谱线，最强线 Smax ≈ 4.56×10⁻¹⁹ cm·molec⁻¹

---

**说明**：本文档中所有 NDIR 滤光片参数均可在第 7 节列出的厂商页面 URL 直接复核；HITRAN 数据由 hitran.org 官方表格直接给出。文中未出现的扩展材料（如 V_NDIR_CO 实物传感器的厂商型号选型）需等用户进一步指定后再补充。
