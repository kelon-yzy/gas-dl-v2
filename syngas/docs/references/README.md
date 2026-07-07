# Syngas 适配文献检索汇总

> 本目录是为「正式实验 v4 检测目标切换到 CO/CO₂/CH₄/H₂」做的前期文献调研。
> 检索日期：2026-06-25。四份子报告独立，本文件只做汇总与决策接口。

---

## 文件索引

| 文件 | 主题 | 关键产物 |
|---|---|---|
| [syngas_composition_ranges.md](syngas_composition_ranges.md) | 合成气 / 煤气化 / 生物质气化典型组分分布 | 9 个文献来源 + 3 套 LHS 采样方案 |
| [co_acoustic_constants.md](co_acoustic_constants.md) | CO 声速 / 弛豫频率 / 与 H₂O 耦合 | 5 个常数推荐值 + ablation 区间 |
| [co_optical_hitran.md](co_optical_hitran.md) | CO HITRAN 谱 + NDIR 滤光片 + 串扰 | 可直接合并的 JSON 片段 + 串扰评估 |
| [syngas_sensing_survey.md](syngas_sensing_survey.md) | 商用传感器现状 + 多模态融合可行性 | 10 个商用系统对比 + 国标对标 |

---

## 一、可直接编码的常数（高置信）

### 1.1 光学侧 —— 直接合并到 `configs/data/spectral-defaults.json`

```json
{
  "gas_specs": [
    ...,
    {"gas": "CO", "table_name": "CO", "molecule_id": 5, "isotopologue_id": 1}
  ],
  "filters": {
    "co": {"channel": "co", "center_cm1": 2145.92, "fwhm_cm1": 82.89}
  },
  "hitran_grids": {
    "co": {
      "wavenumber_min_cm1": 1980.0,
      "wavenumber_max_cm1": 2310.0,
      "wavenumber_step_cm1": 0.1,
      "temperature_k": 296.0,
      "pressure_atm": 1.0
    }
  }
}
```

来源：InfraTec I (4.66 μm / 180 nm) + Boston HIS-E222 + MicroHybrid 三厂商交叉确认，HITRAN molecule_id=5 来自 hitran.org 官方表。详见 [co_optical_hitran.md §4](co_optical_hitran.md)。

### 1.2 声学侧 —— 直接写入 `src/sim/generation/acoustic_physics.py`

```python
# 声速（25°C, 1 atm，与 N₂ 几乎相同，差异 <1 m/s）
_SPEED_CO_MS = 352.0

# 弛豫常数（CO 在 N₂ 背景下的 V-T 弛豫）
PROCESSING_PARAMS_V2 = {
    ...,
    "alpha_lambda_max_co": 0.025,        # 中置信，建议 0.015-0.040 区间 ablation
    "f_relax_co_per_atm": 12000.0,       # 高置信，CO 在 dry N₂ 背景
    "k_h2o_to_f_relax_co": 0.30,         # 中置信，H₂O 对 CO 加速比 CO₂ 强 20 倍
}
```

来源：
- `_SPEED_CO_MS=352`：理论 γ=1.40 + Pérez-Sanz 2014 实测交叉确认（高置信）
- `f_relax_co_per_atm=12000`：Qiao 2022 *Photoacoustics* τ·p ≈ 10 ms·Torr in dry N₂（高置信）
- `k_h2o_to_f_relax_co=0.30`：Yin 2016 *Sensors* 报告 2.5% H₂O 给 8-11× 信号增益（中置信）

详见 [co_acoustic_constants.md §一](co_acoustic_constants.md)。

---

## 二、对原规划文档的修正项

对照 [../adaptation_plan.md](../adaptation_plan.md)（即 `C:\Users\kelon\.claude\plans\stateful-prancing-papert.md` 的项目内副本），有以下修正：

### 修正 1：CO 与 N₂ 在声速维度几乎不可分辨

| 气体 | 25°C 声速 (m/s) | γ | M (g/mol) |
|---|---|---|---|
| CO | 352 | 1.40 | 28.01 |
| N₂ | 353 | 1.40 | 28.01 |

**影响**：原规划在 Phase A3 假设 CO 与 N₂ 在声速混合公式里是不同的项，可保留，但需要明确"超声波通道对 CO 几乎不敏感，CO 的可观测性主要由 NDIR + 光纤麦克风（光声）+ 弛豫吸收提供"。这与 [syngas_sensing_survey.md §3](syngas_sensing_survey.md) 独立得出的结论一致：CO 在 TCD 和声速通道与 N₂ 简并，**高精度 CO 测量只能靠 IR 吸收**。

**对采样设计的影响**：如果背景 N₂ 浓度变化区间窄（方案 A 残量 ~5%），声速通道难以区分 CO 是 45% 还是 60%。需要在训练曲线上观察 CO 的 R² 是否依赖光学通道（用 ablation 验证）。

### 修正 2：HITRAN 网格需要扩展 CO₂ / H₂O 覆盖范围

原规划 A4 只在 `hitran_grids.co` 加新条目。文献分析指出 **CO 通道前向模型必须包含 CO₂ 与 H₂O 在 1980–2310 cm⁻¹ 区间的吸收**（CO ↔ CO₂ 串扰中高，CO ↔ H₂O 中等）。

**建议增补到 Phase A4**：
- `hitran_grids.co2` 下界从 2250 cm⁻¹ 扩到 1980 cm⁻¹（或单独加 `co2_for_co_channel` 子网格）
- 新增 `hitran_grids.h2o`（项目当前 H₂O 已在 `gas_specs` 但没有专门网格，靠 cross-talk 现场 fetch）→ 评估是否值得做静态网格
- `optical_backend.py:collect_hitran_cache_requirements` 的 channels 改为 `("ch4", "co2", "co")` 后，CO 通道下要 fetch CH4/CO2/H2O/CO 四套谱在 [1980,2310] 区间

### 修正 3：LHS 采样方案已确认

文献给出三套方案，用户原区间属于「方案 A 保守 / 气流床合成气」：

| 方案 | CO | H₂ | CO₂ | CH₄ | N₂ | 适用 |
|---|---|---|---|---|---|---|
| A 保守 | 40–65 | 24–38 | 3–18 | 0–5 | balance | 气流床（Texaco/Shell/E-Gas）|
| **B 推荐** | 15–65 | 5–55 | 2–30 | 0–12 | 0.2–5 | **煤气化技术全谱** |
| C air-blown | 10–25 | 5–20 | 8–20 | 1–6 | 40–60 | 生物质气化 / producer gas（独立子集）|

[syngas_composition_ranges.md §5](syngas_composition_ranges.md) 评估：**方案 A 与用户原区间一致**，覆盖窄但物理工况清晰；**方案 B 推荐用于训练泛化能力更强的模型**。当前主方案已确认采用 **方案 B + 条件顺序采样**，方案 A 可作为气流床 holdout。实施规范见 [../lhs_sampling_design.md](../lhs_sampling_design.md)。

### 修正 4：联合可行性约束（原规划 N₂ ≥ 1% 太宽松）

文献给出更严的约束（[syngas_composition_ranges.md §6.4](syngas_composition_ranges.md)）：

1. **质量守恒**：CO + H₂ + CO₂ + CH₄ + N₂ = 100%
2. **H₂/CO ∈ [0.1, 4.0]**（工业实测范围）
3. **CO₂/CO ∈ [0.02, 1.5]**（防止极端非工业组合）
4. **C 平衡**：CO + CO₂ + CH₄ ∈ [35%, 75%]（总碳合理）
5. N₂ 残量在方案 A/B 下应 ≥ 0.2%（O₂/蒸汽气化背景），方案 C 下 ≥ 40%

建议把这些约束实现为 `src/sim/generation/conditions.py` 的拒绝采样规则。

---

## 三、已确认与仍需标定的事项

### 3.1 高优先级（影响 Phase A 编码）

| 项 | 状态 | 决策需求 |
|---|---|---|
| LHS 采样方案 | **已确认** | 采用方案 B + 条件顺序采样；方案 A 作为 holdout，见 [../lhs_sampling_design.md](../lhs_sampling_design.md) |
| `alpha_lambda_max_co` 精确值 | 0.025 占位 | ablation 实验扫 0.015-0.040 |
| CO 通道 cross-talk | **已确认** | 完整串扰，分两步实施，见 [../co_crosstalk_design.md](../co_crosstalk_design.md) |

### 3.2 中优先级（不阻塞编码，但影响仿真精度）

- **CO V-T 弛豫与温度的耦合**：文献无 25-100°C 温度修正系数，按 N₂ 类比线性外推
- **真实合成气检测目标传感器的 datasheet**：当前 CO NDIR 滤光片用 InfraTec I 4.66 μm 占位（与 CH4/CO2 通道同性质），正式 benchmark 前需替换
- **H₂O 对 CO 弛豫的非线性饱和**：当前 `k_h2o_to_f_relax_co=0.30` 是线性近似，实测是三段非线性

### 3.3 低优先级（文档说明清楚即可）

- **未检索 CNKI 中文文献**（国内煤化工厂实测数据），目前组分分布以 NETL/欧美文献为主
- **未对 IEEE Xplore / Scopus 做全量检索**，syngas_sensing_survey 的「未找到完全同构四模态对手」结论置信度 M
- **OOM / 显存**：原规划未涉及，但 hitran_grids 扩展 4 套气体 × 3000 cm⁻¹ 步长 0.1 = ~120k 行/气体 × 3 通道，HITRAN cache 体积估算 ~50MB（可接受）

---

## 四、关于项目竞争力的几个独立观察

来自 [syngas_sensing_survey.md](syngas_sensing_survey.md) 的独立洞察，与原规划不冲突但值得用户知道：

1. **工业合成气在线监测主流就是 NDIR + TCD + EC 三合一**，项目慢通道架构与之同构，不是开新路。
2. **没有公开同行做过 NDIR + 超声 + 光纤麦克风 + TCD 四模态融合**，项目的差异化点是「四模态 ML 融合 vs 工业三模态规则补偿」。
3. **CO 在超声波和热导通道与 N₂ 近似简并**，CO 高精度只能靠 IR 吸收 → 这是 CO 检测精度天花板的物理根源。
4. **目标精度 R² > 0.7 vs 工业 ±1-2% FS 量级差 1-2 个数量级**，Ridge baseline 0.71 是低端基线而非工业可用。
5. **国标对标**：GB/T 40789-2021（CO/CO₂/O₂ 在线 AMS）+ HJ 1241-2022（固定污染源 CO/HCl CEMS）是可对标的整体标准；**专门覆盖"合成气在线监测"的整套国标未找到**。

---

## 五、下一步建议

当前 LHS 方案和 CO 通道串扰范围已确认。Phase A 编码可直接参考：

1. [../adaptation_plan.md](../adaptation_plan.md)：总体实施路线和验证流程。
2. [../lhs_sampling_design.md](../lhs_sampling_design.md)：方案 B + 条件顺序采样实现规范。
3. [../co_crosstalk_design.md](../co_crosstalk_design.md)：CO 通道完整串扰的分步实现。
4. 本目录文献报告：用于追溯物理常数和滤光片参数来源。
