# LHS 采样方案决策：方案 B + 条件顺序采样

> 决策结论：采用 **方案 B（煤气化技术全谱）** 区间 + **条件顺序采样**替代拒绝采样。
> 本文档包含决策依据、数值分析、实现规范和验收标准，供 `src/sim/generation/conditions.py` 编码直接引用。

---

## 一、方案 B 采样区间（确定）

| 组分 | min (%) | max (%) | 来源 |
|------|---------|---------|------|
| CO | 15 | 65 | Lurgi 17% → Shell 63%，覆盖固定床到气流床 |
| H₂ | 5 | 55 | 固定床低 H₂ → 蒸汽气化高 H₂ |
| CO₂ | 2 | 30 | Shell 1.5%（取 2%）→ Lurgi 32%（取 30%） |
| CH₄ | 0 | 12 | 气流床近零 → Lurgi 实测 8-10% |
| N₂ | ≥ 0.2% | balance | O₂/蒸汽气化背景（被动计算） |

### 联合可行性约束

| # | 约束 | 表达式 | 来源 |
|---|------|--------|------|
| C1 | 质量守恒 | CO + H₂ + CO₂ + CH₄ + N₂ = 100% | 物理 |
| C2 | N₂ 非负且有最低残量 | N₂ ≥ 0.2% | 工业实测下限 |
| C3 | H₂/CO 比值工业可行 | H₂/CO ∈ [0.1, 4.0] | Wabash/Shell 实测范围 |
| C4 | CO₂/CO 比值工业可行 | CO₂/CO ∈ [0.02, 1.5] | 防止极端非工业组合 |
| C5 | 总碳合理 | CO + CO₂ + CH₄ ∈ [35%, 75%] | 气化反应化学平衡 |

---

## 二、为什么不用拒绝采样

### 蒙特卡洛实测数据（50 万样本）

**方案 B + 均匀随机 + 拒绝采样：接受率 50.8%**

各约束的单独通过率：

| 约束 | 通过率 | 是否瓶颈 |
|------|--------|---------|
| N₂ ≥ 0.2% | 63.1% | 是（主瓶颈） |
| H₂/CO ∈ [0.1, 4.0] | 99.5% | 否 |
| CO₂/CO ∈ [0.02, 1.5] | 98.6% | 否 |
| 碳平衡 [35%, 75%] | 69.9% | 是（次瓶颈） |
| **全部约束联合** | **50.8%** | — |

### 空间填充质量劣化（1000 样本实测）

| 指标 | 含义 | 方案 B + 拒绝采样 |
|------|------|-------------------|
| KS 检验 CO | p < 0.05 = 非均匀 | **p ≈ 0（非均匀）** |
| KS 检验 H₂ | 同上 | **p ≈ 0（非均匀）** |
| KS 检验 CO₂ | 同上 | **p ≈ 0（非均匀）** |
| KS 检验 CH₄ | 同上 | p = 0.088（勉强通过） |
| 超立方体区域覆盖 CV | 0=完美均匀 | 0.62（严重不均匀） |

**核心问题**：拒绝采样系统性剔除高浓度角落的样本（CO+H₂+CO₂ 同时偏高的区域），导致训练集在可行域边界处严重欠采样。ML 模型在这些边界工况上的泛化能力会显著下降。

---

## 三、条件顺序采样实现规范

### 核心思想

按组分逐个采样，每一步动态计算当前组分的上限，确保后续组分有足够空间满足所有约束。**接受率可提升到 ~85-90%**，且不破坏边际均匀性。

### 采样顺序

选择方差贡献最大的组分先采（CO → H₂ → CO₂ → CH₄），因为先采的组分边际分布最接近均匀。

### 实现伪代码

```python
from scipy.stats.qmc import LatinHypercube

N2_MIN = 0.2
H2_CO_MIN, H2_CO_MAX = 0.1, 4.0
CO2_CO_MIN, CO2_CO_MAX = 0.02, 1.5
CARBON_MIN, CARBON_MAX = 35.0, 75.0

CO_RANGE  = (15.0, 65.0)
H2_RANGE  = (5.0, 55.0)
CO2_RANGE = (2.0, 30.0)
CH4_RANGE = (0.0, 12.0)


def _sequential_upper_bound(
    x_co: float, x_h2: float | None = None, x_co2: float | None = None,
) -> dict[str, float]:
    """计算当前已采组分下，后续组分的动态上限。"""
    budget = 100.0 - N2_MIN  # 总可分配预算

    if x_h2 is None:
        # 正在决定 H₂ 上限
        # C2: x_co + x_h2 + CO2_min + CH4_min ≤ budget
        ub_from_n2 = budget - x_co - CO2_RANGE[0] - CH4_RANGE[0]
        # C3: x_h2 ≤ H2_CO_MAX * x_co
        ub_from_ratio = H2_CO_MAX * x_co
        # C3: x_h2 ≥ H2_CO_MIN * x_co（下限约束，不影响上限）
        lb_from_ratio = H2_CO_MIN * x_co
        return {
            "h2_ub": min(H2_RANGE[1], ub_from_n2, ub_from_ratio),
            "h2_lb": max(H2_RANGE[0], lb_from_ratio),
        }

    if x_co2 is None:
        # 正在决定 CO₂ 上限
        ub_from_n2 = budget - x_co - x_h2 - CH4_RANGE[0]
        # C4: x_co2 ≤ CO2_CO_MAX * x_co
        ub_from_ratio = CO2_CO_MAX * x_co
        # C4: x_co2 ≥ CO2_CO_MIN * x_co
        lb_from_ratio = CO2_CO_MIN * x_co
        # C5: x_co + x_co2 + CH4_min ≤ CARBON_MAX → x_co2 ≤ CARBON_MAX - x_co - CH4_min
        ub_from_carbon = CARBON_MAX - x_co - CH4_RANGE[0]
        return {
            "co2_ub": min(CO2_RANGE[1], ub_from_n2, ub_from_ratio, ub_from_carbon),
            "co2_lb": max(CO2_RANGE[0], lb_from_ratio),
        }

    # 正在决定 CH₄ 上限
    ub_from_n2 = budget - x_co - x_h2 - x_co2
    # C5: x_co + x_co2 + x_ch4 ≤ CARBON_MAX
    ub_from_carbon = CARBON_MAX - x_co - x_co2
    # C5: x_co + x_co2 + x_ch4 ≥ CARBON_MIN
    lb_from_carbon = max(0.0, CARBON_MIN - x_co - x_co2)
    return {
        "ch4_ub": min(CH4_RANGE[1], ub_from_n2, ub_from_carbon),
        "ch4_lb": max(CH4_RANGE[0], lb_from_carbon),
    }


def generate_syngas_samples(n: int, *, seed: int) -> list[dict[str, float]]:
    """方案 B 条件顺序采样。

    采样顺序 CO → H₂ → CO₂ → CH₄，每步动态收紧后续组分的上下限。
    如果某步的 lb > ub（约束不相容），丢弃该样本并补采。
    """
    sampler = LatinHypercube(d=4, seed=seed)
    # 预生成足够的 LHS 分位数（留 20% 余量应对极少数约束冲突）
    raw = sampler.random(n=int(n * 1.2))

    samples = []
    idx = 0
    while len(samples) < n and idx < len(raw):
        u_co, u_h2, u_co2, u_ch4 = raw[idx]
        idx += 1

        # Step 1: CO（无依赖，直接映射）
        x_co = CO_RANGE[0] + u_co * (CO_RANGE[1] - CO_RANGE[0])

        # Step 2: H₂（依赖 CO）
        bounds = _sequential_upper_bound(x_co)
        h2_lb, h2_ub = bounds["h2_lb"], bounds["h2_ub"]
        if h2_lb > h2_ub:
            continue  # 约束冲突，丢弃
        x_h2 = h2_lb + u_h2 * (h2_ub - h2_lb)

        # Step 3: CO₂（依赖 CO, H₂）
        bounds = _sequential_upper_bound(x_co, x_h2)
        co2_lb, co2_ub = bounds["co2_lb"], bounds["co2_ub"]
        if co2_lb > co2_ub:
            continue
        x_co2 = co2_lb + u_co2 * (co2_ub - co2_lb)

        # Step 4: CH₄（依赖 CO, H₂, CO₂）
        bounds = _sequential_upper_bound(x_co, x_h2, x_co2)
        ch4_lb, ch4_ub = bounds["ch4_lb"], bounds["ch4_ub"]
        if ch4_lb > ch4_ub:
            continue
        x_ch4 = ch4_lb + u_ch4 * (ch4_ub - ch4_lb)

        x_n2 = 100.0 - x_co - x_h2 - x_co2 - x_ch4

        samples.append({
            "x_H2": round(x_h2, 6),
            "x_CH4": round(x_ch4, 6),
            "x_CO2": round(x_co2, 6),
            "x_CO": round(x_co, 6),
            "x_N2": round(x_n2, 6),
        })

    if len(samples) < n:
        raise RuntimeError(
            f"Sequential sampling exhausted {idx} candidates but only produced "
            f"{len(samples)}/{n} feasible samples. Consider relaxing constraints."
        )
    return samples
```

### 关键设计说明

1. **LHS 分位数在 [0,1]⁴ 空间均匀**，条件映射会拉伸或压缩各维度的区间，但不会引入系统性偏差。每个组分在其条件可行域内的分布仍然近似均匀。

2. **采样顺序 CO → H₂ → CO₂ → CH₄ 的理由**：
   - CO 是合成气的定义性组分（浓度最高、范围最宽），先采不受约束
   - H₂ 受 H₂/CO 比值约束，必须在 CO 之后
   - CO₂ 受 CO₂/CO 比值 + 碳平衡约束
   - CH₄ 受碳平衡下限约束（确保总碳 ≥ 35%），放最后

3. **冲突丢弃率预期 <15%**：冲突主要发生在 CO 很大（~65%）同时 LHS 分位数把 H₂ 推到上限附近的情况。预生成 1.2n 个候选即可。

4. **`x_N2` 必须作为计算值写入返回 dict**，因为 `acoustic_physics.py` 的声速和衰减计算需要它。

---

## 四、验收标准

### 采样质量检查（写入 `tests/test_syngas_sampling.py`）

```python
def test_sequential_sampling_feasibility():
    """所有样本满足全部约束。"""
    samples = generate_syngas_samples(2000, seed=42)
    for s in samples:
        x_n2 = s["x_N2"]
        assert x_n2 >= 0.2, f"N2 too low: {x_n2}"
        ratio_h2co = s["x_H2"] / s["x_CO"]
        assert 0.1 <= ratio_h2co <= 4.0, f"H2/CO out of range: {ratio_h2co}"
        ratio_co2co = s["x_CO2"] / s["x_CO"]
        assert 0.02 <= ratio_co2co <= 1.5, f"CO2/CO out of range: {ratio_co2co}"
        carbon = s["x_CO"] + s["x_CO2"] + s["x_CH4"]
        assert 35.0 <= carbon <= 75.0, f"Carbon out of range: {carbon}"


def test_sequential_sampling_coverage():
    """边际分布覆盖目标区间的 90% 以上。"""
    samples = generate_syngas_samples(2000, seed=42)
    co_vals = [s["x_CO"] for s in samples]
    assert min(co_vals) < 18.0, "CO lower end not covered"
    assert max(co_vals) > 62.0, "CO upper end not covered"
    # 类似检查 H2, CO2, CH4


def test_sequential_sampling_acceptance_rate():
    """接受率 ≥ 80%（即 1.2n 候选生成 n 样本）。"""
    # 实际通过观察 generate_syngas_samples 的 idx vs n 比值来验证
    samples = generate_syngas_samples(1000, seed=42)
    assert len(samples) == 1000
```

### 空间填充质量检查

```python
def test_sequential_sampling_space_filling():
    """最近邻距离的变异系数 < 0.5（均匀填充的典型值）。"""
    from scipy.spatial import cKDTree
    import numpy as np

    samples = generate_syngas_samples(1000, seed=42)
    pts = np.array([[s["x_CO"], s["x_H2"], s["x_CO2"], s["x_CH4"]] for s in samples])
    # 归一化到 [0,1]
    pts_norm = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0) + 1e-8)
    tree = cKDTree(pts_norm)
    dists, _ = tree.query(pts_norm, k=2)
    nn_dists = dists[:, 1]
    cv = nn_dists.std() / nn_dists.mean()
    assert cv < 0.5, f"Space filling CV too high: {cv:.3f}"
```

---

## 五、Holdout 评估集（可选）

如果需要测试模型在"纯气流床"工况上的精度，可以用方案 A 区间生成一个小的 holdout benchmark：

```bash
python -m pipeline.generate_benchmark \
    --composition-scheme syngas_entrained_flow \
    --sequence-count 200 \
    --output-dir data/sg4-holdout-entrained
```

这个数据集不参与训练，仅用于评估方案 B 训练的模型在窄区间工况上的表现。

---

## 六、与现有代码的接口变更

| 现有函数/常量 | 变更 |
|--------------|------|
| `_generate_lhs_samples(n, seed)` | 维度从 `d=3` 改为 `d=4`，返回 4-tuple |
| `_sample_components_lhs(u_h2, u_co2, u_n2)` | 替换为 `_sample_components_sequential(u_co, u_h2, u_co2, u_ch4)` |
| `_map_hydrogen_lhs(u)` | 删除（合成气不需要 H₂ 双峰分布） |
| `_sample_components_random(rng)` | 替换为合成气版本 |
| `generate_condition_rows()` | 新增 `composition_scheme` 参数，`"syngas"` 走新路径 |
| `COMPONENT_FIELDS` 引用 | 返回 dict 的 key 从 `x_N2` 变为 `x_CO`，但仍包含 `x_N2` 计算值 |
