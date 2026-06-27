# 合成气 / 煤气化 / 生物质气化 组分分布文献综述

> **用途**：为正式实验 v4 的 LHS 采样设计提供组分浓度区间的工业实测数据支撑。
> **检测目标**：CO / CO₂ / CH₄ / H₂，背景气 N₂。
> **检索日期**：2026-06-25。
> **检索方式**：NETL gasifipedia、MDPI Energies、ACS Omega、Lupine 等公开学术 / 政府报告。

---

## 1. 文献来源清单

| # | 来源 | 作者 / 机构 | 年份 | 类型 | URL / DOI | 置信度 |
|---|------|-------------|------|------|-----------|--------|
| **[R1]** | Gasifipedia §5.1.5 *Syngas Composition* | NETL (US DOE) | n.d. (持续维护) | 政府机构技术文档 | https://www.netl.doe.gov/research/coal/energy-systems/gasification/gasifipedia/syngas-composition | **High** |
| **[R2]** | *Gasification Processes Old and New: A Basic Review of the Major Technologies*, **Energies 3(2):216–240** | R.W. Breault, NETL-US DOE | 2010 | 同行评议综述 | DOI: 10.3390/en3020216 | **High** |
| **[R3]** | NETL Gas Turbine Handbook §1.2 *Composition of Raw Syngas from Coal Fed Gasifiers* | NETL (US DOE) | n.d. | 政府机构技术手册 | https://www.netl.doe.gov/sites/default/files/gas-turbine-handbook/1-2.pdf | **High** |
| **[R4]** | *Syngas Compositions, Cold Gas and Carbon Conversion Efficiencies for Different Coal Gasification Processes and all Coal Ranks* | Said M.A. Ibrahim, Mostafa E.M. Samy (Al-Azhar Univ.) | 2020 | 期刊文章 (J. Mining & Mech. Eng.) | DOI: 10.32474/JOMME.2020.01.000109 | **Medium** |
| **[R5]** | *Syngas Production from Biomass Gasification: Influences of Feedstock Properties, Reactor Type, and Reaction Parameters*, **ACS Omega 8(35):31620–31631** | Gao et al. (Nanjing Forestry Univ.) | 2023 | 同行评议综述 (PMC10483670) | DOI: 10.1021/acsomega.3c03050 | **High** |
| **[R6]** | *A Comprehensive Review of Syngas Production, Fuel Properties, and Operational Parameters for Biomass Conversion*, **Energies 17(15):3646** | Khlifi, Pozzobon, Lajili | 2024 | 同行评议综述 | DOI: 10.3390/en17153646 | **High** |
| **[R7]** | *Gasification of Coal* (IspatGuru technical note) | M. Sahu / IspatGuru | n.d. | 行业技术资料 | https://www.ispatguru.com/gasification-of-coal/ | Medium |
| **[R8]** | *Syngas Analyzer Applying in Gasifier* — Cubic Optoelectronics | Wuhan Cubic Optoelectronics Co. | 2019 | 厂商技术资料 (在线监测视角) | https://www.syngas-analyzer.com/new/SYNGAS-ANALYZER-APPLYING-IN-GASIFIER.html | Low–Medium |
| **[R9]** | NETL Gasifipedia §5.1.6 *Syngas Optimized for Intended Products* | NETL (US DOE) | n.d. | 政府机构技术文档 | https://netl.doe.gov/research/coal/energy-systems/gasification/gasifipedia/syngas-optimization | **High** |

> 注：[R1]/[R3]/[R9] 是 NETL gasifipedia 的子页，引用时合并为 NETL gasifipedia。

---

## 2. 不同气化技术 / 工艺下的典型合成气组分（干基，vol %）

### 2.1 NETL Gas Turbine Handbook 实测数据 [R3]

来源：典型 oxygen-blown 煤气化炉的"raw syngas"组分，干基。

| Gasifier | H₂ | CO | CO₂ | CH₄ | N₂+Ar | H₂/CO |
|----------|------|------|-----|-----|-------|-------|
| Sasol / Lurgi (固定床，dry-ash) | 52.2 | 29.5 | 5.6 | 4.4 | — | 1.77 |
| BGL (slagging 固定床) | 30.3 | 39.6 | 10.8 | 0.1 | — | 0.77 |
| Texaco / GE Energy (气流床, slurry feed) | 26.4 | 45.8 | 2.9 | 3.8 | — | 0.58 |
| E-Gas / ConocoPhillips (气流床, slurry feed) | 33.5 | 44.9 | 16.0 | 1.8 | — | 0.75 |
| Shell / Uhde (气流床, dry feed) | 26.7 | 63.1 | 1.5 | 0.03 | — | 0.42 |

### 2.2 Wabash River E-Gas 商业运行数据 (1996–1999, four-year demonstration) [R1]

> 实测年均最低 / 最高浓度。

| 年份 | H₂ Low–High | CO Low–High | CO₂ Low–High | CH₄ Low–High |
|------|-------------|-------------|--------------|--------------|
| 1996 | 32.87–34.21 | 42.34–46.03 | 14.89–17.13 | 1.26–1.99 |
| 1997 | 32.90–34.40 | 42.20–46.70 | 16.60–16.90 | 1.04–2.02 |
| 1998 | 32.71–33.82 | 44.25–46.73 | 14.92–16.06 | 1.90–2.09 |
| 1999 | 32.31–33.44 | 44.44–46.31 | 15.25–16.22 | 1.88–2.17 |

### 2.3 不同煤阶 × 气化炉类型 [R4]（Ibrahim & Samy, 2020）

> 单位 mol %（干基，含 H₂S/H₂O 已剔除时的相对组分）。

**Table — Entrained Flow (气流床)**

| 组分 | Anthracite | Bituminous | Sub-bit. | Lignite | min | max |
|------|-----------|-----------|----------|---------|-----|-----|
| CO   | 58.79 | 52.96 | 49.30 | 42.93 | 42.93 | 58.79 |
| H₂   | 30.13 | 38.08 | 31.33 | 24.42 | 24.42 | 38.08 |
| CO₂  | 7.56  | 6.81  | 6.34  | 5.52  | 5.52  | 7.56 |
| CH₄  | 0     | 0     | 0     | 0     | 0     | 0 |
| N₂   | 0.20  | 0.36  | 0.56  | 0.34  | 0.20  | 0.56 |

**Table — Fluidized Bed (流化床)**

| 组分 | Anthracite | Bituminous | Sub-bit. | Lignite | min | max |
|------|-----------|-----------|----------|---------|-----|-----|
| CO   | 47.12 | 42.33 | 39.53 | 34.45 | 34.45 | 47.12 |
| H₂   | 32.56 | 40.43 | 33.31 | 26.32 | 26.32 | 40.43 |
| CO₂  | 12.77 | 11.47 | 10.72 | 9.34  | 9.34  | 12.77 |
| CH₄  | 4.24  | 3.81  | 3.55  | 3.10  | 3.10  | 4.24 |
| N₂   | 0.20  | 0.36  | 0.56  | 0.34  | 0.20  | 0.56 |

**Table — Fixed Bed (固定床)**

| 组分 | Anthracite | Bituminous | Sub-bit. | Lignite | min | max |
|------|-----------|-----------|----------|---------|-----|-----|
| CO   | 68.05 | 58.79 | 53.71 | 44.98 | 44.98 | 68.05 |
| H₂   | 4.88  | 18.78 | 11.77 | 6.27  | 4.88  | 18.78 |
| CO₂  | 3.87  | 3.34  | 3.05  | 2.56  | 2.56  | 3.87 |
| CH₄  | 18.32 | 15.83 | 14.46 | 12.11 | 12.11 | 18.32 |
| N₂   | 0.28  | 0.48  | 0.73  | 0.43  | 0.28  | 0.73 |

> 警示：[R4] 的 fixed-bed CH₄ 偏高（12–18%），与 [R3] Lurgi 实测 4.4 % 有显著差异。原因是 [R4] 用的是热力学平衡模型估算（而非工业测点），固定床低温区有利于甲烷化反应。**实际工业 Lurgi 数据应以 [R3]/[R7] 为准（CH₄ ≈ 4–10%）。**

### 2.4 国内典型煤气化炉组分范围 [R8]（Wuhan Cubic Optoelectronics，在线监测厂商资料）

| 炉型 | H₂ % | CO % | CO₂ % | CH₄ % |
|------|------|------|-------|-------|
| UGI 煤气炉 (空气 + 蒸汽，移动床) | 50 | 18–20 | 5 | 3–4 |
| Lurgi 煤气炉 | 37–39 | 17–18 | 32 | 8–10 |
| Winkler 煤气炉 (流化床) | 35–46 | 30–40 | 13–25 | 1–2 |
| K-T 煤气炉 (Koppers-Totzek 气流床) | 31 | 58 | 10 | 0.1 |
| Texaco 煤气炉 | 35–36 | 44–51 | 13–18 | 0.1 |

### 2.5 生物质气化组分（按气化剂区分）[R5][R6]

> 气化剂决定 N₂ 含量，决定背景气特性，**这是与煤气化最大的差异**。

| 气化剂 | H₂ % | CO % | CO₂ % | CH₄ % | N₂ % | LHV (MJ·Nm⁻³) | 备注 |
|--------|------|------|-------|-------|------|---------------|------|
| **空气 (air)** | 5–16 | 10–22 | 9–19 | 2–6 | **45–60** | 4–6 | 高 N₂ 稀释（producer gas），不属于"传感器目标场景"通常关注的合成气 |
| **氧气 (O₂)** | 25–40 | 30–55 | 5–15 | 0–5 | <2 | 10–15 | "纯净" syngas，是工业 IGCC / Sasol 主线 |
| **蒸汽 (steam)** | 35–55 | 20–35 | 15–25 | 5–12 | <1 | 12–18 | H₂ 富集型，H₂/CO 高 |
| **O₂ + steam 混合** | 30–45 | 30–50 | 10–20 | 0–5 | <2 | 10–15 | Sasol / Lurgi 主流工艺 |

数据综合自 [R5]（ACS Omega 综述）及 [R6]（Energies 综述）。

### 2.6 NETL gasifipedia 给出的"典型合成气"区间 [R1]

> 这是最被广泛引用的工业总体区间：

- **CO**: 30–60 %
- **H₂**: 25–30 %
- **CH₄**: 0–5 %
- **CO₂**: 5–15 %
- **plus** H₂O、H₂S、COS、NH₃ 等杂质

---

## 3. H₂ / CO 比值（工业典型）

| 工艺 / 用途 | H₂/CO 目标 | 数据来源 |
|------------|------------|----------|
| Fischer–Tropsch (Fe 催化剂) | 0.5–0.7 | [R9] |
| Fischer–Tropsch (Co 催化剂) | ~2.0 | [R9] |
| Methanol 合成 | ~2.0；(H₂−CO₂)/(CO+CO₂) ≈ 2.05 | [R9] |
| Hydrogen 生产 | 经 WGS 调到 high | [R9] |
| 气流床直出 (Shell / Texaco) | 0.4–0.8 | [R3] |
| 流化床直出 | 0.7–1.0 | [R4] |
| 固定床 Lurgi 直出 | 1.5–2.0 | [R3][R7] |
| Wabash E-Gas 实测 | > 0.7 | [R1] |

**工业 raw syngas 的 H₂/CO 区间总体在 0.4–2.0**，下游通过 WGS（水煤气变换）调整。

---

## 4. 工业在线监测的检测目标浓度区间 [R8]

来源：武汉四方光电（Cubic Optoelectronics）syngas analyzer 应用说明（在线监测厂商资料，Low–Medium 置信度，仅作为传感器视角参考）。

典型在线气体分析仪同时检测 **CO / CO₂ / CH₄ / H₂ / O₂ / CₙHₘ** 六组分，工作浓度区间普遍是：

- CO: **0–70 %**（覆盖所有炉型）
- H₂: **0–60 %**
- CO₂: **0–35 %**
- CH₄: **0–20 %**（覆盖 Lurgi 等高 CH₄ 炉型）
- O₂: 0–25 %（用于点火前安全监测）

> 在线监测的关键场景：① 反应状态判断；② 点火前确认 CO/H₂ 浓度低于爆炸下限；③ 半水煤气 O₂ 关键控制；④ 投料前 N₂ 置换完整性确认。

---

## 5. 对用户初步采样区间的评估

### 用户给出区间
| 组分 | 用户区间 | 工业实测综合区间（多源） | 评估 |
|------|----------|------------------------|------|
| **CO**  | 45–60 % | 17–68 %（全炉型）；30–60 %（NETL 主流） | 用户区间偏窄、偏高。**仅覆盖气流床 / Shell-Texaco 高 CO 工况，未涵盖 Lurgi 固定床 (17–30 %) 与流化床 (30–47 %)。** |
| **H₂**  | 25–35 % | 5–55 %（全炉型）；25–34 %（NETL 主流） | 用户区间合理对应气流床实测主区间；但未覆盖蒸汽气化（35–55 % H₂）与 Lurgi（37–52 %）。 |
| **CO₂** | 5–15 % | 1.5–32 %（全炉型）；5–17 %（NETL 主流） | 用户区间合理；上限可酌情扩到 20 % 覆盖 Lurgi（30%+ 较极端，可不必）。 |
| **CH₄** | 0–5 %  | 0–18 %（全炉型）；0–5 %（气流床）；4–10 %（Lurgi 实测）；12–18 %（固定床平衡模型） | 用户区间合理对应气流床 / 流化床；若要兼容 Lurgi 等固定床移动床炉型，应扩到 0–10 %。 |
| **N₂**  | 残量    | <2 %（O₂/蒸汽气化）；45–60 %（空气气化 producer gas） | 用户区间默认是 O₂-blown 合成气；如要兼容 air-blown 生物质气化（producer gas），需引入 N₂ 大范围背景气工况。 |

### 5.1 结论

- 用户提出的区间 **CO 45–60 / H₂ 25–35 / CO₂ 5–15 / CH₄ 0–5 / N₂ 残量** 是**气流床合成气（Texaco / Shell / E-Gas）的典型工况**，与 NETL Wabash 多年实测数据（CO 42–47, H₂ 32–34, CO₂ 15–17, CH₄ 1–2）一致性较好。
- **覆盖窄**：不涵盖 Lurgi 固定床（高 CH₄ + 高 CO₂ + 高 H₂）、流化床（中等 CO + 较高 CH₄）、空气气化（高 N₂ 背景），也不涵盖蒸汽气化（H₂ 富集型）。
- 如果项目目标只是"针对气流床型典型合成气的在线监测"，**用户区间已经合理可用**；如果目标是"覆盖工业常见气化技术全谱以增强模型泛化",则区间需要扩展。

---

## 6. 推荐 LHS 采样区间

### 6.1 方案 A — 保守区间（仅气流床合成气，与用户原意一致）

| 组分 | min | max | 备注 |
|------|-----|-----|------|
| CO   | 40 % | 65 % | 略放宽下限至 40 %，覆盖 Wabash 低点与 Texaco 上限 |
| H₂   | 24 % | 38 % | 覆盖 Shell/E-Gas/Texaco/Anthracite-Bituminous 实测 |
| CO₂  | 3 %  | 18 % | 覆盖 Shell (1.5) ~ Wabash (17) |
| CH₄  | 0 %  | 5 %  | 用户原区间 |
| N₂   | balance | balance | 残量；上限受其他组分约束 |

### 6.2 方案 B — 宽区间（覆盖煤气化主要技术全谱，推荐）

| 组分 | min | max | 备注 |
|------|-----|-----|------|
| CO   | 15 % | 65 % | 涵盖 Lurgi (17) → Shell (63) |
| H₂   | 5 %  | 55 % | 涵盖 Lurgi 移动床低 H₂ → 蒸汽气化高 H₂ |
| CO₂  | 2 %  | 30 % | 涵盖 Shell (1.5) → Lurgi (32) |
| CH₄  | 0 %  | 12 % | 涵盖气流床近零 → Lurgi (8–10) |
| N₂   | 0.2 % | 5 % | O₂/蒸汽气化背景；如要兼容空气气化需单独建子区间 |

### 6.3 方案 C — 全谱（含 air-blown producer gas）

如果项目目标必须包含 air-blown 生物质气化或 producer gas，建议作为独立子数据集（"高 N₂ 工况"）：

| 组分 | min | max |
|------|-----|-----|
| CO   | 10 % | 25 % |
| H₂   | 5 %  | 20 % |
| CO₂  | 8 %  | 20 % |
| CH₄  | 1 %  | 6 %  |
| N₂   | 40 % | 60 % |

**建议**：方案 B 与方案 C 不要混在一个 LHS 中采样，因为 N₂ 量级差异 10 倍以上，会让方案 B 的样本被边界稀释。若两者都要，分两批独立 LHS。

### 6.4 联合可行性约束

无论选哪个方案，LHS 必采的约束：

1. **质量守恒**：`CO + H₂ + CO₂ + CH₄ + N₂ = 100 %`（vol%, 干基）
   - 通常将 N₂ 设为 balance（被动），其余 4 组分独立采样。
   - 若 4 组分独立采样后 N₂ < 0 或超过上限，则丢弃该样本。

2. **H₂/CO 比物理可行**：保留 H₂/CO ∈ [0.1, 4.0]
   - 工业实测最大约 2.0（Lurgi 蒸汽气化），最小约 0.4（Shell）；放宽到 [0.1, 4.0] 兼顾边界。

3. **CO₂/CO 物理可行**：保留 CO₂/CO ∈ [0.02, 1.5]
   - 防止极端非工业组合（如 CO=15%, CO₂=30% 同时出现仅在 Lurgi 才可能）。

4. **C 平衡**：`CO + CO₂ + CH₄` 应保持在合理总碳范围（约 35 %–75 %），低于此则非典型工业 syngas。

5. **如需对应特定下游用途**，可附加 H₂/CO 区间约束（见 §3）。

---

## 7. 置信度总结

| 数据维度 | 置信度 | 依据 |
|---------|--------|------|
| 气流床 (Shell/Texaco/E-Gas) 组分区间 | **High** | NETL 政府报告 + Wabash 4 年实测 + 多源交叉 |
| 流化床组分区间 | **Medium–High** | Breault 2010 综述 + Ibrahim 2020（含平衡模型） |
| 固定床 / Lurgi 组分区间 | **Medium** | NETL Gas Turbine Handbook + IspatGuru，但与 [R4] 平衡模型差异显著 |
| 生物质气化（O₂/steam）组分区间 | **High** | ACS Omega 2023 + Energies 2024 同行评议综述 |
| 生物质气化（air-blown）组分区间 | **High** | 同上 |
| 在线监测传感器工作区间 | **Low–Medium** | 仅厂商资料 [R8]，需 cross-check 标气厂商手册 |
| H₂/CO 工业目标比 | **High** | NETL gasifipedia §5.1.6 |

---

## 8. 给项目的下一步建议

1. **若 LHS 设计目标 = 气流床主流 syngas**：直接用用户原区间，但建议把 CO 下限从 45 % 放宽到 40 %，CO₂ 上限从 15 % 放宽到 18 %，与 Wabash 实测对齐。
2. **若 LHS 设计目标 = 覆盖煤气化技术全谱**：采用 §6.2 方案 B，并附 §6.4 的可行性约束剔除非物理样本。
3. **air-blown producer gas 工况是否需要**？这取决于实际检测器服务场景。如果检测器只装在 IGCC / Sasol 等 O₂/蒸汽气化主线上，则不需要；如果要兼容小型生物质气化器，需单独建子区间（方案 C）。
4. **传感器工作区间需独立核对**：建议查阅项目实际使用的传感器（NDIR / TCD / 半导体？）的官方手册，[R8] 仅供方向性参考。

---

## 9. 未解决项

- 未找到针对 **中国国内典型煤化工合成氨 / 甲醇厂的合成气在线监测点位实测数据集**（权威同行评议来源），如需该信息需要进一步检索中文文献（CNKI / 万方），可在后续轮次补充。
- 未找到针对"煤焦化煤气"（COG，焦炉煤气）的对照组分数据，COG 通常 H₂ ~55%, CH₄ ~25%, CO ~7%，与本报告范围（合成气 / 气化制气）不同领域，未纳入。
