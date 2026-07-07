# CO 物理常数编码速查

> 本文档从 `docs/syngas/references/` 的 4 份详细报告中提取可直接写入代码的常数和代码片段。
> 每个参数标注置信度（High / Medium / Low）和一级文献来源，详细推导见对应子报告。

---

## 1. 声学常数 → `src/sim/generation/acoustic_physics.py`

### 1.1 声速

```python
_SPEED_CO_MS = 352.0  # 25°C, 1 atm; High
# 来源: 理论 c=√(γRT/M), γ=1.40, M=28.01 g/mol → 352.2 m/s
# 实验: Pérez-Sanz et al., J.Chem.Thermodyn. 79 (2014) 224-229
# 注意: 与 _SPEED_N2_MS=353.0 仅差 1 m/s (同 M, 同 γ)
```

### 1.2 弛豫参数

```python
PROCESSING_PARAMS_V2 = {
    # ... 现有参数保留 ...
    "alpha_lambda_max_co": 0.025,   # Medium; N₂/CH₄ 物理类比, 建议 ablation 0.015-0.040
    "f_relax_co_per_atm": 12000.0,  # High; CO 在 dry N₂ 背景, Qiao 2022 Photoacoustics
    "k_h2o_to_f_relax_co": 0.30,    # Medium; 文献报告 H₂O 给 8-11× 信号增益
                                     # 保守线性化, 实际为非线性三阶段饱和
}
```

### 1.3 声速混合公式新增 CO 项

```python
def hidden_sound_speed_v2(
    x_h2: float, x_ch4: float, x_co2: float, x_n2: float,
    x_co: float,  # 新增
    t_c: float,
) -> float:
    x_h2_frac = max(0.0, x_h2) / 100.0
    x_ch4_frac = max(0.0, x_ch4) / 100.0
    x_co2_frac = max(0.0, x_co2) / 100.0
    x_n2_frac = max(0.0, x_n2) / 100.0
    x_co_frac = max(0.0, x_co) / 100.0   # 新增
    c_mix = (
        x_h2_frac * _SPEED_H2_MS
        + x_ch4_frac * _SPEED_CH4_MS
        + x_co2_frac * _SPEED_CO2_MS
        + x_n2_frac * _SPEED_N2_MS
        + x_co_frac * _SPEED_CO_MS        # 新增
    )
    c_mix += 0.6 * (t_c - 25.0)
    return max(c_mix, 200.0)
```

### 1.4 衰减公式新增 CO 项

```python
# 在 hidden_attenuation_v2 中, N₂ 通道之后添加:
x_co_frac = max(0.0, x_co) / 100.0
f_r_co = params["f_relax_co_per_atm"] * p_atm * (1.0 + params["k_h2o_to_f_relax_co"] * h_w_pct)
alpha_lambda_co = (
    params["alpha_lambda_max_co"]
    * x_co_frac
    * 2.0 * f_hz * f_r_co
    / (f_hz**2 + f_r_co**2)
)
alpha_co_npm = alpha_lambda_co * f_hz / c_mix

# alpha_true 加入 alpha_co_npm
# 返回 dict 加入 "alpha_co_v2": alpha_co_npm, "f_relax_co_eff": f_r_co
```

### 1.5 各气体声学参数对照表

| 气体 | 25°C 声速 (m/s) | γ | M (g/mol) | αλ_max | f_relax (Hz/atm) | 置信度 |
|------|----------------|-----|-----------|--------|-------------------|--------|
| H₂ | 1306 | 1.41 | 2.016 | — (扩散衰减) | — | High |
| CH₄ | 446 | 1.31 | 16.04 | 0.034 | 30000+ | High |
| CO₂ | 268 | 1.29 | 44.01 | 0.120 | 28000 | High |
| **CO** | **352** | **1.40** | **28.01** | **0.025** | **12000** | **Medium** |
| N₂ | 353 | 1.40 | 28.01 | 0.004 | 65000 | High |
| H₂O | — | — | 18.02 | 0.010 | 100000 | High |

---

## 2. 热导率 → `acoustic_physics.py:_hidden_lambda_mix`

```python
def _hidden_lambda_mix(x_h2: float, x_co2: float, x_co: float, t_c: float) -> float:
    # CO 热导率 ~25 mW/(m·K) vs N₂ ~26 mW/(m·K), 差异小但 60% 占比下累积 ~0.6 mW/(m·K)
    return 0.034 + 0.00155 * x_h2 - 0.00011 * x_co2 - 0.00005 * x_co + 0.00002 * (t_c - 25.0)
    # x_co 系数 -0.00005: 占位, Medium 置信度, 需文献标定
```

---

## 3. 光学参数 → `configs/data/spectral-defaults.json`

### 3.1 完整 JSON 合并片段

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

**来源**: InfraTec I (4.66 μm / 180 nm) + Boston HIS-E222 + MicroHybrid 三厂商交叉确认。
滤光片 CWL 4.66 μm = 2145.92 cm⁻¹, FWHM 180 nm @ 4.66 μm = 82.89 cm⁻¹。置信度 High。

### 3.2 HITRAN 验证标准

HAPI `fetch('CO', 5, 1, 1980, 2310)` 应返回 ~1344 条主同位素谱线，最强线 Smax ≈ 4.56×10⁻¹⁹ cm·molec⁻¹。

### 3.3 CO 隐式吸收函数（待新增）

```python
def _hidden_absorption_co(x_co: float, h_rh: float, p_mpa: float, t_c: float) -> float:
    # CO 4.7μm 基频带吸收, 类比 _hidden_absorption_co2 结构
    # 系数需从 HITRAN 积分或实测标定, 以下为占位 (Medium)
    return 0.035 * x_co + 0.0005 * h_rh + 0.010 * p_mpa + 0.00015 * (t_c - 25.0)
```

### 3.4 光学串扰矩阵（`optical_crosstalk.py` 扩展）

| 干扰对 | 严重程度 | 处理方式 |
|--------|---------|---------|
| CO 信号 ← CO₂ 干扰 | 中高 | 必须在 CO 通道仿真中纳入 CO₂ 在 1980-2310 cm⁻¹ 的吸收 |
| CO 信号 ← H₂O 干扰 | 中 | 纳入 H₂O 在 1980-2310 cm⁻¹ 的吸收 |
| CO 信号 ← CH₄ 干扰 | 可忽略 | 带中心间距 876 cm⁻¹, 不处理 |
| CO₂ 信号 ← CO 干扰 | 低 | 可选, 优先级低 |

---

## 4. 采样区间 → `src/sim/generation/conditions.py`

### 4.1 方案 A — 气流床合成气（与用户原意一致）

| 组分 | min (%) | max (%) | 来源 |
|------|---------|---------|------|
| CO | 40 | 65 | NETL Wabash 实测 + Shell 上限 |
| H₂ | 24 | 38 | Shell/E-Gas/Texaco 实测 |
| CO₂ | 3 | 18 | Shell ~ Wabash 实测 |
| CH₄ | 0 | 5 | 用户原区间, 气流床一致 |
| N₂ | balance | balance | ≥ 0.2%, 被动计算 |

### 4.2 方案 B — 煤气化技术全谱（推荐, 泛化更强）

| 组分 | min (%) | max (%) |
|------|---------|---------|
| CO | 15 | 65 |
| H₂ | 5 | 55 |
| CO₂ | 2 | 30 |
| CH₄ | 0 | 12 |
| N₂ | 0.2 | 5 |

### 4.3 联合约束（两方案通用）

```python
# 拒绝采样规则
def is_feasible(x_h2, x_co, x_co2, x_ch4, *, n2_min=0.2):
    x_n2 = 100.0 - x_h2 - x_co - x_co2 - x_ch4
    if x_n2 < n2_min:
        return False
    if x_co < 1e-6:
        return False
    h2_co_ratio = x_h2 / x_co
    if not (0.1 <= h2_co_ratio <= 4.0):
        return False
    co2_co_ratio = x_co2 / x_co
    if not (0.02 <= co2_co_ratio <= 1.5):
        return False
    total_carbon = x_co + x_co2 + x_ch4
    if not (35.0 <= total_carbon <= 75.0):
        return False
    return True
```

---

## 5. 一级文献索引

| 编号 | 引用 | 用于 |
|------|------|------|
| [1] | Yin et al., Sensors 16 (2016) 162 | CO V-T 弛豫, H₂O 加速 |
| [2] | Qiao et al., Photoacoustics 25 (2022) 100334 | CO τ·p ≈ 10 ms·Torr |
| [3] | Pérez-Sanz et al., J.Chem.Thermodyn. 79 (2014) 224 | CO-N₂ 混合声速 |
| [4] | HITRAN 2020 (hitran.org) | molecule_id=5, 谱线参数 |
| [5] | InfraTec IR filter catalog | NBP 4.66 μm / 180 nm |
| [6] | NETL gasifipedia / Gas Turbine Handbook | 合成气组分分布 |
| [7] | Anal. Chem. (2025) 10.1021/acs.analchem.5c00062 | H₂O 对 CO 非线性三阶段 |

详细文献列表见 `docs/syngas/references/co_acoustic_constants.md §六` 和 `docs/syngas/references/co_optical_hitran.md §7`。
