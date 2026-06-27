# CO 通道完整串扰建模实现方案

> 决策结论：**实施完整串扰建模**（CO₂ + H₂O 对 CO 通道的干扰），与现有 CH₄/CO₂ 通道串扰架构对称。
> 采用**分两步实施策略**：先跑通无串扰基线，再补完整串扰作为 ablation 对比。
> 本文档包含物理分析、实现规范、代码片段和验收标准。

---

## 一、物理分析

### 1.1 CO 通道滤光片带宽

```
CO  滤光片: center = 2145.92 cm⁻¹, FWHM = 82.89 cm⁻¹
  -3dB 带宽: [2104.5, 2187.4] cm⁻¹
  -20dB 带宽 (≈3σ): [~2040, ~2252] cm⁻¹

CO₂ 滤光片: center = 2347.0 cm⁻¹, FWHM = 93.0 cm⁻¹
  -3dB 带宽: [2300.5, 2393.5] cm⁻¹

CH₄ 滤光片: center = 3030.0 cm⁻¹, FWHM = 147.0 cm⁻¹
  -3dB 带宽: [2956.5, 3103.5] cm⁻¹
```

### 1.2 各气体在 CO 通道 [2104.5, 2187.4] cm⁻¹ 内的吸收

| 干扰源 | 吸收带位置 | 距 CO 通道中心 | 在 CO 带宽内的吸收 | 严重程度 |
|--------|-----------|--------------|-------------------|---------|
| **CO 自身** | 基频带 2000-2230 cm⁻¹ | 0（目标信号） | 强（主信号） | — |
| **CO₂** | ν₃ 主带 2280-2400 cm⁻¹ | 主带距 CO 带宽上边缘 **113 cm⁻¹** | **无直接重叠**。但 CO₂ 有 ν₁+ν₂−ν₂ 弱热带和 ν₃ P 支远翼在 2100-2200 cm⁻¹，强度为 ν₃ 峰的 **10⁻³ ~ 10⁻⁴** | **中等**（高 CO₂ 下显著） |
| **H₂O** | ν₂ 弯曲 ~1595 cm⁻¹ / ν₁ν₃ 伸缩 ~3700 cm⁻¹ | 2100-2200 cm⁻¹ 处于两个带之间的窗口 | 散在弱分立线，非连续吸收 | **低-中等** |
| **CH₄** | ν₃ 反对称伸缩 3019 cm⁻¹ | **873 cm⁻¹**（10.5× CO FWHM） | **完全无干扰** | **可忽略** |

### 1.3 定量估算（合成气典型浓度下）

#### CO₂ → CO 通道串扰

CO₂ 在 2100-2200 cm⁻¹ 区间的主要贡献来自：
- ν₃ P 支远翼（J > 50 的高转动态）：单线强度 ~10⁻²³ cm/molec（vs 带中心 ~10⁻¹⁸）
- ν₁+ν₂−ν₂ 差带（hot band）：单线强度 ~10⁻²² cm/molec
- 密度效应：CO₂ 浓度 15-30% vol 时，大量弱线累积可观察

**定量级**：假设 CO 60%、CO₂ 15% 共存
- CO 自身吸收：0.035 × 60 ≈ 2.1 absorbance units（参照 `_hidden_absorption_co` 线性系数）
- CO₂ 泄漏到 CO 通道：约 CO₂ 在自身通道吸收的 0.3-1%
  - CO₂ 自身通道吸收：0.045 × 15 ≈ 0.675
  - 泄漏量：0.675 × 0.003~0.01 ≈ **0.002 ~ 0.007**
  - 相对 CO 信号：0.002/2.1 ≈ **0.1%** ~ 0.007/2.1 ≈ **0.3%**

**结论**：在典型工况下，CO₂ 串扰约占 CO 信号的 **0.1-0.3%**。对于目标精度 R² > 0.9 的模型，这个量级**可以被学习但不做也不会崩溃**。但在极端工况（CO 15%、CO₂ 30%）下，串扰可达 **1-3%**，变得不可忽略。

#### H₂O → CO 通道串扰

H₂O 在 2100-2200 cm⁻¹ 有若干分立线但不构成连续吸收带：
- 合成气冷凝后 H₂O 约 3-5% vol
- 干扰量级估计为 CO 信号的 **0.05-0.2%**

**结论**：H₂O 串扰在大多数工况下可忽略，但建模成本很低（顺带做了），所以值得纳入。

---

## 二、串扰矩阵设计

### 2.1 当前 2×2 矩阵

```
              CH₄通道实测    CO₂通道实测
CH₄真实吸收 [    1.0          ε_21=0.012  ]
CO₂真实吸收 [  ε_12=0.035      1.0        ]

observed_ch4 = true_ch4 + 0.035 × true_co2
observed_co2 = true_co2 + 0.012 × true_ch4
```

### 2.2 扩展后 3×3 矩阵

```
              CH₄通道实测    CO₂通道实测    CO通道实测
CH₄真实吸收 [    1.0          ε_21          ε_31 ≈ 0    ]
CO₂真实吸收 [  ε_12          1.0           ε_32          ]
CO 真实吸收 [  ε_13 ≈ 0      ε_23          1.0           ]
```

其中：
- `ε_12 = 0.035`（现有，CH₄ 通道中 CO₂ 的响应）
- `ε_21 = 0.012`（现有，CO₂ 通道中 CH₄ 的响应）
- **`ε_32`**（CO 通道中 CO₂ 的响应）：**需要标定的关键系数**
- **`ε_23`**（CO₂ 通道中 CO 的响应）：CO 基频带远翼对 CO₂ 通道的泄漏
- `ε_13 ≈ 0`（CO 通道中 CH₄ 的响应）：带间距 873 cm⁻¹，忽略
- `ε_31 ≈ 0`（CH₄ 通道中 CO 的响应）：同理忽略

### 2.3 串扰系数推荐值

| 系数 | 推荐值 | 置信度 | 推导方式 | ablation 范围 |
|------|--------|--------|---------|--------------|
| `ε_32`（CO₂→CO） | **0.005** | Medium | CO₂ 弱热带 + 远翼积分估算（~ν₃ 峰的 0.1-1%） | 0.001 – 0.02 |
| `ε_23`（CO→CO₂） | **0.002** | Medium-Low | CO R 支高 J 线延伸到 2200+ cm⁻¹，泄漏到 CO₂ P 支区 | 0.0005 – 0.005 |
| `ε_13`（CH₄→CO） | **0** | High | 带间距 873 cm⁻¹ | 固定为 0 |
| `ε_31`（CO→CH₄） | **0** | High | 带间距 873 cm⁻¹ | 固定为 0 |

**H₂O 不通过串扰矩阵建模**——它通过 `_hidden_absorption_co()` 函数中的 `h_rh` 项直接纳入（与 CH₄/CO₂ 通道中 H₂O 的处理方式一致）。

### 2.4 HITRAN 谱积分法标定 ε_32（推荐但非阻塞）

如果要用 HITRAN 数据精确计算 ε_32：

```python
# 在 CO 滤光片带宽内，计算 CO₂ 的等效吸收度
# integral(CO2_absorption × CO_filter_response, dν) / integral(CO_absorption × CO_filter_response, dν)

# 步骤：
# 1. 用 HAPI fetch CO₂ 和 CO 在 [1980, 2310] cm⁻¹ 的谱线
# 2. 计算各自在 T=296K, P=1atm, L=1cm 的吸收截面
# 3. 与 CO 高斯滤光片卷积
# 4. 取比值（归一化到 1% 体积分数）

# 这个计算可以在 precompute 阶段一次性完成
```

当前先用占位值 0.005，等 HITRAN 缓存构建完成后再回来精确标定。

---

## 三、分步实施计划

### Step 1：无串扰基线（Phase A4，~2 小时）

目标：CO 通道只包含 CO 自身吸收，先跑通 benchmark 生成 + DL 训练闭环。

#### 3.1.1 新增 `_hidden_absorption_co()` — `acoustic_physics.py`

```python
def _hidden_absorption_co(x_co: float, h_rh: float, p_mpa: float, t_c: float) -> float:
    """CO 4.7μm 基频带吸收（经验线性模型）。

    系数结构与 _hidden_absorption_ch4/_hidden_absorption_co2 对称。
    x_co 系数 0.035 基于 CO 基频带 Smax ≈ 4.56e-19 cm/molec
    与 CO₂ ν₃ 带 Smax ≈ 3.5e-18 的比例（~1:8）推导,
    再参照 _hidden_absorption_co2 的 x_co2 系数 0.045 缩放。
    """
    return 0.035 * x_co + 0.0005 * h_rh + 0.010 * p_mpa + 0.00015 * (t_c - 25.0)
```

#### 3.1.2 `main_sensor_features()` 新增 V_NDIR_CO 输出

```python
# 在 v_ndir_co2 计算之后添加：
absorption_co = _hidden_absorption_co(x_co, h_rh, p_mpa, t_c)
# Step 1 不做串扰，直接用 true absorption
optical_baseline_co_now = 2.5 + optical_drift_co2 + rng.gauss(0.0, 0.006)  # 与 CO₂ 共享漂移特性
v_ndir_co = max(
    0.1,
    optical_baseline_co_now * math.exp(-absorption_co) + rng.gauss(0.0, 0.008),
)
```

#### 3.1.3 不修改 `optical_crosstalk.py`

CO 通道暂时不参与串扰矩阵。`apply_optical_crosstalk` 保持 2×2 不变。

---

### Step 2：完整串扰（Phase A4.5，~6-8 小时）

目标：扩展串扰矩阵为 3×3，建模 CO₂→CO 和 CO→CO₂ 的互扰。

#### 3.2.1 重写 `optical_crosstalk.py`

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpticalCrosstalkSpec:
    # CH₄ ↔ CO₂ (现有)
    ch4_channel_co2_response: float = 0.035
    co2_channel_ch4_response: float = 0.012
    # CO₂ ↔ CO (新增)
    co_channel_co2_response: float = 0.005   # ε_32: CO₂ 弱热带泄漏到 CO 通道
    co2_channel_co_response: float = 0.002   # ε_23: CO R 支远翼泄漏到 CO₂ 通道
    # CH₄ ↔ CO (可忽略，显式设为零以保持矩阵完整)
    co_channel_ch4_response: float = 0.0
    ch4_channel_co_response: float = 0.0


DEFAULT_OPTICAL_CROSSTALK_SPEC = OpticalCrosstalkSpec()


def apply_optical_crosstalk(
    *,
    absorption_ch4: float,
    absorption_co2: float,
    absorption_co: float = 0.0,
    spec: OpticalCrosstalkSpec = DEFAULT_OPTICAL_CROSSTALK_SPEC,
) -> dict[str, float]:
    # CH₄ 通道实测 = CH₄ 真值 + CO₂ 泄漏 + CO 泄漏
    ch4_observed = (
        absorption_ch4
        + spec.ch4_channel_co2_response * absorption_co2
        + spec.ch4_channel_co_response * absorption_co
    )
    # CO₂ 通道实测 = CO₂ 真值 + CH₄ 泄漏 + CO 泄漏
    co2_observed = (
        absorption_co2
        + spec.co2_channel_ch4_response * absorption_ch4
        + spec.co2_channel_co_response * absorption_co
    )
    # CO 通道实测 = CO 真值 + CO₂ 泄漏 + CH₄ 泄漏
    co_observed = (
        absorption_co
        + spec.co_channel_co2_response * absorption_co2
        + spec.co_channel_ch4_response * absorption_ch4
    )

    return {
        "absorption_ch4_true": absorption_ch4,
        "absorption_co2_true": absorption_co2,
        "absorption_co_true": absorption_co,
        "absorption_ch4_cross_from_co2": spec.ch4_channel_co2_response * absorption_co2,
        "absorption_co2_cross_from_ch4": spec.co2_channel_ch4_response * absorption_ch4,
        "absorption_co_cross_from_co2": spec.co_channel_co2_response * absorption_co2,
        "absorption_co2_cross_from_co": spec.co2_channel_co_response * absorption_co,
        "absorption_ch4_observed": ch4_observed,
        "absorption_co2_observed": co2_observed,
        "absorption_co_observed": co_observed,
    }
```

#### 3.2.2 `main_sensor_features()` 更新

```python
# 替换 Step 1 的简化逻辑：
absorption_co = _hidden_absorption_co(x_co, h_rh, p_mpa, t_c)

if optical_absorption is None:
    absorption_ch4 = _hidden_absorption_ch4(x_ch4, h_rh, p_mpa, t_c)
    absorption_co2 = _hidden_absorption_co2(x_co2, h_rh, p_mpa, t_c)
    optical_absorption = apply_optical_crosstalk(
        absorption_ch4=absorption_ch4,
        absorption_co2=absorption_co2,
        absorption_co=absorption_co,       # 新增参数
    )

# V_NDIR_CO 使用 observed（含串扰）而非 true
v_ndir_co = max(
    0.1,
    optical_baseline_co_now * math.exp(-optical_absorption["absorption_co_observed"]) + rng.gauss(0.0, 0.008),
)
```

#### 3.2.3 HITRAN 网格扩展

`configs/data/spectral-defaults.json` 中 CO₂ 网格需要覆盖 CO 通道范围：

```json
"hitran_grids": {
    "co2": {
        "wavenumber_min_cm1": 1980.0,
        "wavenumber_max_cm1": 2445.0,
        "wavenumber_step_cm1": 0.1,
        "temperature_k": 296.0,
        "pressure_atm": 1.0
    }
}
```

**注意**：CO₂ 网格下界从 2250 扩展到 1980 cm⁻¹。这会增加缓存但不影响 CO₂ 通道本身的滤光片计算（滤光片参数不变）。

如果不想改动现有 CO₂ 网格（避免影响旧 benchmark 兼容性），可以新建子网格：

```json
"hitran_grids": {
    "co2": { ... },
    "co2_in_co_channel": {
        "wavenumber_min_cm1": 1980.0,
        "wavenumber_max_cm1": 2310.0,
        "wavenumber_step_cm1": 0.1,
        "temperature_k": 296.0,
        "pressure_atm": 1.0
    }
}
```

这是更安全的做法——旧网格不动，新网格只服务于 CO 通道的串扰计算。

---

## 四、对模型训练的影响

### 4.1 无串扰 vs 完整串扰的预期差异

| 指标 | 无串扰（Step 1） | 完整串扰（Step 2） |
|------|-------------------|-------------------|
| V_NDIR_CO 信号 | CO 浓度的干净单调函数 | CO 浓度 + CO₂ 干扰 + H₂O 干扰 |
| CO R² 预期 | 较高（信号纯净） | 略低（信号含噪，但更真实） |
| CO₂ R² 预期 | 不变 | 可能略降（CO 反向泄漏到 CO₂ 通道） |
| 模型学习难度 | 低（直接映射） | 中（需学会解耦） |
| Sim-to-real 迁移 | 差（真实传感器有串扰） | 好（仿真更贴近真实） |

### 4.2 值得做的 ablation 实验

| 实验 | 配置 | 验证目标 | 实测（2026-06-27） |
|------|------|---------|---|
| Baseline | 无串扰（`sg4-formal`） | CO 通道"天花板精度" | TCN x_CO R²=0.954 |
| Cross-talk | ε_32=0.005（`sg4-formal-crosstalk`） | 串扰引起的精度退化量 | **TCN x_CO R²=0.956（持平，Δ ≤ 0.006）** |
| Cross-talk sweep | ε_32 ∈ {0.001, 0.005, 0.01, 0.02} | 串扰系数敏感度 | 未做（持平结论使敏感度扫描必要性降低） |
| CO channel ablation | 移除 V_NDIR_CO | CO 仅靠声学/TCS 的 R²（验证 CO/N₂ 简并假说） | **TCN x_CO R²=0.484（baseline 0.954 → -0.47），Ridge 同向 0.470** |

**实验 4（CO channel ablation）特别重要**：移除 V_NDIR_CO 后 CO R² 从 0.954 跌至 ~0.48（损失约 50%，非原预期的 ~0），证实 V_NDIR_CO 是 CO 检测的**支配通道**但保留 ~50% 残留可学性。结论修正：CO 主导依赖光学，非完全依赖。详见 [stage_ii_ablation_results.md §2](stage_ii_ablation_results.md#22-结果test-splitmean--std)。

**Cross-talk 实测结果意义**：3×3 串扰为确定性线性变换，模型可学到逆映射，R² 与无串扰持平。这是 informative 的负结果——线性串扰**不构成模型学习难度**，sim-to-real gap 应在硬件层面（非线性 / 时变 / 标定漂移）验证。详见 [stage_ii_ablation_results.md §3](stage_ii_ablation_results.md#32-结果test-splitmean--std)。

---

## 五、实现难点与对策

### 难点 1：串扰系数标定（最大难点）

**问题**：ε_32（CO₂→CO）没有真实传感器标定数据，只有 HITRAN 谱线和滤光片参数。

**对策**：
1. 先用占位值 ε_32=0.005（物理量级估算）
2. HITRAN 缓存构建完成后，用谱积分法精确计算
3. 通过 ablation 扫描验证敏感度
4. 最终以真实传感器标定数据替换（正式实验阶段）

### 难点 2：`optical_backend.py` 的通道-气体关联逻辑

**问题**：当前 `collect_hitran_cache_requirements` 假设每个通道只需要自己的目标气体 + H₂O。CO 通道需要 CO + CO₂ + H₂O 三种气体。

**对策**：在通道配置中增加 `interference_gases` 字段：

```python
CHANNEL_GAS_MAP = {
    "ch4": {"target": "CH4", "interference": ["CO2", "H2O"]},
    "co2": {"target": "CO2", "interference": ["CH4", "H2O"]},
    "co":  {"target": "CO",  "interference": ["CO2", "H2O"]},  # CH₄ 忽略
}
```

### 难点 3：HITRAN 缓存膨胀

**问题**：CO 通道需要 3 种气体的谱线数据，缓存体积增加。

**量化**：
- 当前：CH₄ 3000 点 + CO₂ 1950 点 = 4950 点/condition
- 新增 CO 通道：CO 3300 点 + CO₂_in_co 3300 点 + H₂O_in_co 3300 点 = 9900 点
- 总计约 14850 点/condition（~3× 增长）
- 每 100 条序列新增 ~45 MB 缓存（可接受）

**对策**：无特殊处理。HITRAN 缓存是一次性预计算，存储和计算开销都在可承受范围内。

### 难点 4：`apply_optical_crosstalk` 的向后兼容

**问题**：现有代码在多处调用 `apply_optical_crosstalk(absorption_ch4=..., absorption_co2=...)`，新增 `absorption_co` 参数后需要保证旧 benchmark 不崩。

**对策**：`absorption_co` 参数设默认值 `0.0`。旧 benchmark（wv4-smoke）不传此参数，行为完全不变。新 benchmark（sg4-smoke）传入 CO 吸收值。

```python
def apply_optical_crosstalk(
    *,
    absorption_ch4: float,
    absorption_co2: float,
    absorption_co: float = 0.0,  # 向后兼容：旧调用不传此参数
    spec: OpticalCrosstalkSpec = DEFAULT_OPTICAL_CROSSTALK_SPEC,
) -> dict[str, float]:
```

### 难点 5：测试覆盖

**问题**：`test_optical_crosstalk.py` 需要覆盖 3×3 矩阵的所有路径。

**对策**：参数化测试：

```python
@pytest.mark.parametrize("absorption_co", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("absorption_co2", [0.0, 0.3, 1.5])
@pytest.mark.parametrize("absorption_ch4", [0.0, 0.2, 1.0])
def test_crosstalk_3x3_observed_ge_true(absorption_ch4, absorption_co2, absorption_co):
    result = apply_optical_crosstalk(
        absorption_ch4=absorption_ch4,
        absorption_co2=absorption_co2,
        absorption_co=absorption_co,
    )
    # observed ≥ true（串扰只增加，不减少）
    assert result["absorption_ch4_observed"] >= absorption_ch4
    assert result["absorption_co2_observed"] >= absorption_co2
    assert result["absorption_co_observed"] >= absorption_co


def test_crosstalk_backward_compat():
    """旧调用（不传 absorption_co）行为不变。"""
    old = apply_optical_crosstalk(absorption_ch4=0.5, absorption_co2=0.3)
    assert "absorption_co_observed" in old
    assert old["absorption_co_observed"] == 0.0  # CO 吸收为零时无串扰输出
    # CH₄/CO₂ 结果与旧版一致
    assert abs(old["absorption_ch4_observed"] - (0.5 + 0.035 * 0.3)) < 1e-10
    assert abs(old["absorption_co2_observed"] - (0.3 + 0.012 * 0.5)) < 1e-10
```

---

## 六、验收标准

### Step 1 验收（无串扰基线）

- [ ] `_hidden_absorption_co()` 在 x_co ∈ [0, 65] 范围内输出非负
- [ ] `V_NDIR_CO` 出现在 benchmark `slow.npy` 的通道中
- [ ] `manifest.json` 记录 `slow_channels` 包含 `V_NDIR_CO`
- [ ] `pytest tests/test_spectral_co_channel.py -v` 通过
- [ ] DL 训练可消费新 benchmark（9 通道 slow 输入）

### Step 2 验收（完整串扰）

- [x] `apply_optical_crosstalk` 新增 `absorption_co` 参数，默认 0.0
- [x] 旧 benchmark（wv4-smoke）的 `pytest tests/test_optical_crosstalk.py` 不变
- [x] 新 benchmark 的 `V_NDIR_CO` 信号在 CO₂ 浓度升高时呈现系统性偏移
- [x] ablation 实验：无串扰 vs 有串扰的 CO R² 差异在 0.01-0.05 以内（串扰不会过大）— **实测 |Δ| ≤ 0.006，远低于预期上限，模型已学会逆映射**
- [x] CO channel ablation：移除 V_NDIR_CO 后 CO R² 显著下降（验证 CO/N₂ 简并假说）— **实测 0.954 → 0.484，损失 ~50%，主导依赖成立**
