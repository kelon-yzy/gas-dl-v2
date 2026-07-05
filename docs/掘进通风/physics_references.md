# 掘进通风场景物性常数速查

> 本文档提供 CO₂/O₂/N₂ 三组分的可编码物性常数。
> 详细文献来源见 [references/co2_o2_n2_gas_properties.md](references/co2_o2_n2_gas_properties.md)。
> 目标代码文件：`src/sim/generation/tunnel_ventilation/acoustic_physics.py`
>
> CO₂ 和 N₂ 的部分常数已在主线代码中使用（来源：[../dl_model_architecture.md §13.5](../dl_model_architecture.md#135-声学物理acoustic_physicspy)，原始数据为 NIST @298.15 K），标注为 `[代码已有]`。
> O₂ 为新增组分，参数来源见各表"来源"列。
>
> **数据核实状态（2026-07-04 经 smart-search 多源验证）**：O₂ cp/λ/η/M 已通过 NIST WebBook Shomate 方程手算 + Engineering ToolBox + search 多源交叉确认。O₂ 弛豫机制经 Bass 1990 JASA 公式确认。国标条款经应急管理部官方 PDF 确认。置信度标注于各条目。

## 1. 基本物性

| 气体 | M (kg/mol) | M (g/mol) | γ (300 K) | 自由度 | 来源 |
|------|----------:|----------:|----------:|-------:|------|
| CO₂ | 0.04401 | 44.01 | 1.289 | 多原子 | [代码已有] NIST |
| O₂ | 0.031998 | 31.9988 | 1.400 | 双原子 | NIST WebBook（M.W. 31.9988）|
| N₂ | 0.02801 | 28.01 | 1.400 | 双原子 | [代码已有] NIST |

## 2. 声学参数

### 2.1 纯组分声速 (300 K, 1 atm)

理想气声速：`c = sqrt(γ · R · T / M)`，R = 8.314 J/(mol·K)。

与主线 hg/sg 代码使用同一公式（`acoustic_physics.py:48` `hidden_sound_speed_v2`）。

| 气体 | γ | M (kg/mol) | c₀ 理想气计算 (m/s) | c₀ 实测/查表 (m/s) | 来源 |
|------|----:|----------:|------------------:|------------------:|------|
| CO₂ | 1.289 | 0.04401 | 270.3 | — | [代码已有] §13.5 |
| O₂ | 1.400 | 0.031998 | 330.4 | 329 @ 298K | 理想气公式 + Engineering ToolBox |
| N₂ | 1.400 | 0.02801 | 353.1 | — | [代码已有] §13.5 |

O₂ 与 N₂ 的声速差 ≈ 22.7 m/s（约 6.4%），是超声通道区分两者的物理基础。CO₂ 声速显著低于 O₂/N₂（~270 vs ~330–353），对混合气声速贡献敏感。

### 2.2 混合气声速

与主线完全相同的公式（`acoustic_physics.py:48`）：

```python
# 理想气混合（与 hg/sg 代码一致）
# M_mix = Σ(x_i · M_i)
# cp_mix = Σ(x_i · cp_i)
# γ_mix = cp_mix / (cp_mix - R)
# c_mix = sqrt(γ_mix · R · T_K / M_mix)
```

### 2.3 比热容

| 气体 | cp (J/mol·K, 298 K) | 来源 | 置信度 |
|------|--------------------:|------|--------|
| CO₂ | 37.13 | [代码已有] `acoustic_physics.py` NIST @298.15K | 高 |
| O₂ | 29.38 | NIST WebBook Shomate 方程手算确认（Chase 1998 NIST-JANAF） | 高（已验证）|
| N₂ | 29.12 | [代码已有] `acoustic_physics.py` NIST @298.15K | 高 |

cv 由 `cp - R` 计算（理想气近似）。O₂ cv ≈ 20.79 J/mol·K，γ = 29.38/20.79 ≈ 1.413（与双原子理论 1.40 一致）。

**验证**：NIST WebBook O₂ Shomate 参数（100–700K：A=31.32234, B=-20.23531, C=57.86644, D=-36.50624, E=-0.007374），t=0.29815 代入 Cp = A + B·t + C·t² + D·t³ + E/t² = 29.384 ≈ 29.38 ✓

### 2.4 弛豫参数

主线 hg 代码（`acoustic_physics.py:72` `hidden_attenuation_v2`）中已有的衰减分量：

| 分量 | 机制 | 关键常数 | 本场景适用性 | 来源 |
|------|------|----------|------------|------|
| alpha_classical | 经典粘滞吸收 ∝f² | K_ref=1.84e-11 | 适用（通用） | [代码已有] §13.5 |
| alpha_co2 | CO₂ V-T 弛豫 | f_relax=28 kHz/atm, λ_max=0.12 | 适用（CO₂ 是目标组分） | [代码已有] §13.5 |
| alpha_n2 | N₂ V-T 弛豫 | 65 kHz/atm, λ_max=0.004 | 适用（N₂ 是目标组分） | [代码已有] §13.5 |
| alpha_h2o | H₂O V-T 弛豫 | 100 kHz/atm, λ_max=0.01 | 适用（湿度影响） | [代码已有] §13.5 |
| alpha_ch4 | CH₄ V-T 弛豫 | — | **不适用**（场景无 CH₄） | — |
| alpha_h2_diffusion | H₂ 扩散损耗 | — | **不适用**（场景无 H₂） | — |

**需新增**：

| 分量 | 机制 | 关键常数 | 来源 | 置信度 |
|------|------|----------|------|--------|
| alpha_o2 | O₂ V-T 弛豫 | dry air (h=0) 下 **fr,O ≈ 24 Hz/atm**（Bass 1990 公式：`fr,O = (Ps/Pso)·[24 + 4.04×10⁴·h(0.02+h)/(0.391+h)]`，h 为水蒸气摩尔浓度 %）；受 H₂O 显著加速但典型大气条件下仍 ≪ 200 kHz。**200 kHz 载波远高于 O₂ 弛豫峰，对 200 kHz 衰减贡献可忽略**，工程实现取 alpha_o2 ≈ 0 或仅保留经典吸收分量 | Bass, Sutherland, Zuckerwar 1990, "Atmospheric absorption of sound: Update", J. Acoust. Soc. Am. 88(4), DOI:10.1121/1.400476；Bass & Sutherland 2004, JASA, DOI:10.1121/1.1631937 | 高（机制与公式已验证）|

O₂ 的振动弛豫频率远低于 200 kHz 载波（O₂ 基频 1556 cm⁻¹ 对应的振动能级高，弛豫极慢），预期对 200 kHz 衰减贡献很小。

### 2.5 200 kHz 物理约束

与主线相同：200 kHz 载波下 CO₂ 弛豫峰（f_relax ~28 kHz/atm，P=0.5 MPa 时上移至 ~140 kHz）落在载波附近，高 CO₂ 浓度 + 长声程时信号衰减严重。L_m 限制在 0.2–0.3 m（见 `docs/Phase0_物理可行性核对记录.md`）。

本场景 CO₂ 最高 5%（远低于 hg 的 15%），衰减压力较小，但仍沿用 L_m 0.2–0.3 m 保持一致性。

## 3. 热导参数

### 3.1 纯组分热导率 (298 K)

| 气体 | λ₀ (W/m·K) | λ₀ (mW/m·K) | 幂律指数 n | 来源 | 置信度 |
|------|----------:|------------:|----------:|------|--------|
| CO₂ | 0.0166 | 16.6 | 0.87 | [代码已有] `acoustic_physics.py` §13.5 | 高 |
| O₂ | 0.0264 | 26.4 | 0.80 | NIST Thermophysical Properties / CRC Handbook / Engineering ToolBox | 高（已验证）|
| N₂ | 0.0258 | 25.8 | 0.77 | [代码已有] `acoustic_physics.py` §13.5 | 高 |

温度修正（与主线相同）：`λ(T) = λ₀ · (T / 298.15)^n`

> O₂ λ @ 298K：Engineering ToolBox 给 0.026 W/m·K（26 mW/m·K），search 多源共识 26.3–26.5 mW/m·K @ 300K，本表取 26.4 mW/m·K @ 298K（NIST/CRC 共识中值）。
> O₂ n：search 共识 ~0.80（diatomic gases 幂律指数 0.7–0.9，O₂ ~0.80）。不同来源 0.77–0.88，建议编码时与 N₂ 的 n=0.77 保持来源层级一致。

### 3.2 动力粘度（Wilke 混合用）

| 气体 | η (Pa·s, 298 K) | 来源 | 置信度 |
|------|----------------:|------|--------|
| CO₂ | 1.491e-5 | [代码已有] `acoustic_physics.py` §13.5 | 高 |
| O₂ | 2.058e-5 | NBS Technical Note 350 (Childs 1966), DOI:10.6028/nbs.tn.350；NIST REFPROP (Lemmon & Jacobsen) | 高（已验证）|
| N₂ | 1.781e-5 | [代码已有] `acoustic_physics.py` §13.5 | 高 |

### 3.3 O₂ 与 N₂ 热导率对比

```
λ_O₂ ≈ 26.4 mW/m·K   (NIST/CRC 共识，@ 298K)
λ_N₂ = 25.8 mW/m·K   [代码已有，@ 298K]
Δλ(O₂ - N₂) ≈ 0.6 mW/m·K
相对差异 ≈ 2.3%
```

来源：NIST Thermophysical Properties / CRC Handbook / Engineering ToolBox（详见 [references/co2_o2_n2_gas_properties.md](references/co2_o2_n2_gas_properties.md) §2.4）。

这一差异是 TCS 通道区分 O₂ 和 N₂ 的物理上限。差值约 0.6 mW/m·K（2.3%），与文档早期估计 ~2% 一致；TCS 通道提供边际辨识力，作为声学通道的独立补充信息源。

### 3.4 Wassiljewa-Mason-Saxena 混合规则

与主线代码（`_hidden_lambda_mix`）使用同一公式：

```python
# λ_mix = Σ_i (y_i · λ_i(T) / Σ_j y_j · φ_ij)
# φ_ij 用 Wilke 粘度混合形式：
# φ_ij = (1/√8) · (1 + M_i/M_j)^(-1/2) · [1 + (η_i/η_j)^(1/2) · (M_j/M_i)^(1/4)]²
```

φ_ij 由 Wilke 公式从 M 和 η 直接计算，**无需查表**。CO₂-N₂ 的 φ_ij 已在主线代码中实现。需新增的二元组合（计算示例，编码时由公式自动生成）：

| 组合 | φ_ij 计算值（示意，298 K） | 来源 |
|------|--------------------------:|------|
| CO₂-O₂ | ≈ 0.709 | Wilke 公式：M_CO₂=44.01, M_O₂=32.00, η_CO₂=1.491e-5, η_O₂=2.058e-5 |
| O₂-CO₂ | ≈ 1.414 | 同上（注意 φ_ij ≠ φ_ji） |
| O₂-N₂ | ≈ 0.934 | Wilke 公式：M_O₂=32.00, M_N₂=28.01, η_O₂=2.058e-5, η_N₂=1.781e-5 |
| N₂-O₂ | ≈ 1.059 | 同上 |

> 上表示意值由 Wilke 公式手算给出，编码时应直接实现公式，不硬编码这些数值。CO₂-N₂ 和 N₂-CO₂ 可直接复用主线代码中的现有值。

## 4. 光学参数

### 4.1 CO₂ NDIR

| 参数 | 值 | 来源 | 置信度 |
|------|------|------|--------|
| 吸收带中心 | ν₃ 反对称伸缩 2349 cm⁻¹ (4.26 μm) | HITRAN | 高 |
| HITRAN 分子编号 | 2 | HITRAN2020 | 高 |
| 主线代码滤光片中心 | 2347 cm⁻¹ | [代码已有] `spectral/defaults.py`（占位值） | 高 |
| 典型滤光片 CWL | 4.26 μm (4265 nm) ± 0.5% | Umicore 4.26µm CO2 NBP datasheet | 高（已验证）|
| 典型滤光片 FWHM (HBW) | 窄带 60–120 nm，宽带 ~180 nm（依赖厂商 datasheet） | Umicore HBW 105±10 nm；MicroHybrid MTS4SENS44-3C-1 HBW 120±10 nm（CWL 4265±25 nm）；MicroHybrid MTS4SENS44-3C-2 HBW 60±10 nm（CWL 4415±30 nm） | 高（厂商 datasheet 已验证）|
| 经验吸收公式 | `V_NDIR_CO2 = baseline · exp(-A_co2)` | [代码已有] `acoustic_physics.py:174` | 高 |

经验后端（`empirical_v1`）中 CO₂ 吸收由 `_hidden_absorption_co2` 计算，可直接复用。

### 4.2 O₂ 和 N₂

两者均为同核双原子分子，**无红外活性**。本场景不为 O₂ 和 N₂ 设置光学检测通道。

主线代码中 `V_NDIR_CH4` 通道保留但在本场景中不携带组分信息（无 CH₄），仅含噪声基线。

## 5. 波形规格

沿用 v6-phys-strict 链路，与主线完全一致（`waveforms.py`）：

| 参数 | 值 | 来源 |
|------|------|------|
| 超声载波 | 200 kHz (PSC200K) | [代码已有] §13.6 |
| 采样率 | 1 MS/s (NI-6453) | [代码已有] §13.6 |
| ADC 位深 | 20-bit int32 | [代码已有] §13.6 |
| adc_max | 524287 | `2^(20-1)-1` |
| 超声窗口 | 5 ms → 5000 点 | [代码已有] §13.6 |
| 光纤麦克风窗口 | 10 ms → 10000 点 | [代码已有] §13.6 |
| 量程 | ±2.5 V | [代码已有] §13.6 |
| 声程 L_m | 0.20–0.30 m | [代码已有] §13.11 |
| 分数延迟 | Lagrange 5 阶 FIR | [代码已有] §13.6 |

## 6. 常数复用与新增汇总

### 可直接复用（代码已有）

| 常数 | 主线代码位置 | 备注 |
|------|------------|------|
| CO₂: M, cp, λ, η, n | `acoustic_physics.py` 全局常量 | 完全复用 |
| N₂: M, cp, λ, η, n | 同上 | 完全复用 |
| CO₂ 弛豫: f_relax, λ_max | `hidden_attenuation_v2` | 完全复用 |
| N₂ 弛豫: f_relax, λ_max | 同上 | 完全复用 |
| 经典衰减 K_ref | 同上 | 完全复用 |
| H₂O 弛豫 | 同上 | 完全复用（湿度影响） |
| WMS 混合规则公式 | `_hidden_lambda_mix` | 公式复用，需扩展组分 |
| 声速公式 | `hidden_sound_speed_v2` | 公式复用，需替换组分 |
| 波形仿真 | `waveforms.py` | 完全复用 |

### 需新增（O₂ 参数）

| 常数 | 状态 | 优先级 | 来源 |
|------|------|--------|------|
| O₂: cp = 29.38 J/(mol·K) | 已验证 | 高（声速计算必需） | NIST WebBook Shomate 手算确认 (Chase 1998) |
| O₂: λ₀ = 0.0264 W/m·K, n = 0.80 | 已验证 | 高（TCS 计算必需） | NIST/CRC/Engineering ToolBox 共识 |
| O₂: η = 2.058e-5 Pa·s | 已验证 | 高（Wilke φ_ij 计算必需） | NBS Technical Note 350 (Childs 1966) / NIST REFPROP |
| O₂: 弛豫 f_relax ≈ 24 Hz/atm (dry air), 200 kHz 下可忽略 | 已验证 | 中（工程取 alpha_o2 ≈ 0） | Bass 1990 JASA 公式 |
| CO₂-O₂, O₂-N₂ 交互参数 | 由 Wilke 公式计算，无需查表 | 高（WMS 混合规则） | Wilke 1950 混合规则公式 |

### 需移除（本场景不适用）

| 常数 | 原因 |
|------|------|
| H₂: M, cp, λ, η | 场景无 H₂ |
| CH₄: M, cp, λ, η | 场景无 CH₄ |
| alpha_h2_diffusion | 场景无 H₂ |
| alpha_ch4 | 场景无 CH₄ |
| CO NDIR 相关 | 场景无 CO |

## 7. 编码检查清单

- [x] O₂ cp 值已确认（29.38 J/mol·K, NIST WebBook Shomate 手算验证, Chase 1998）
- [x] O₂ λ₀ 和 n 值已确认（0.0264 W/m·K, n=0.80, NIST/CRC/Engineering ToolBox）
- [x] O₂ η 值已确认（2.058e-5 Pa·s, NBS Technical Note 350 / NIST REFPROP）
- [x] O₂/N₂ 热导率差值已量化（Δλ ≈ 0.6 mW/m·K, 2.3%, TCS 边际辨识力）
- [x] O₂ 弛豫参数已确认（dry air fr,O ≈ 24 Hz/atm，200 kHz 下可忽略，Bass 1990 公式）
- [x] CO₂-O₂ 和 O₂-N₂ Wilke φ_ij 已明确由公式计算（不硬编码）
- [x] CO₂ NDIR 滤光片 FWHM 已确认（窄带 60–120 nm，宽带 ~180 nm，Umicore/MicroHybrid datasheet）
- [x] 空气组成（~78% N₂ + 21% O₂ + 0.04% CO₂）下的 c_mix 与已知空气声速 (~346 m/s @25°C) 交叉验证 — 已通过（`test_sound_speed_air_mixture_close_to_346`，c_mix ≈ 346 m/s）
- [x] 所有常数写入 `acoustic_physics.py` 后，单元测试覆盖边界条件 — 已完成（25 tests，`tests/test_tunnel_ventilation_physics.py`）
