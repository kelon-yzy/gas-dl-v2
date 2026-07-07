# 新气体检测目标 (CO/CO₂/CH₄/H₂) 适配规划

> **2026-06-25 更新**：文献检索完成，规划已根据调研结果修正。详见 [§ 决策落地（2026-06-25 更新）](#决策落地2026-06-25-更新) 与 [docs/syngas/references/README.md](syngas/references/README.md)。

## Context

当前正式实验v4 项目检测目标为 H₂/CH₄/CO₂/N₂（掺氢天然气，sum=100% 严格闭包），架构覆盖：物理仿真（声学 + 光学 NDIR + TCS）→ DL/ML 训练 → 评估。

用户要求适配新场景：**合成气 / 煤气化制气**，检测目标改为 **H₂ / CH₄ / CO₂ / CO**，N₂ 保留为背景稀释气（不预测），4 个目标变量 sum < 100。其他架构不变。

用户给出的典型浓度区间：

| 组分  | 范围                 |
| --- | ------------------ |
| CO  | 45–60%             |
| H₂  | 25–35%             |
| CO₂ | 5–15%              |
| CH₄ | 0–5%               |
| N₂  | = 100% − Σ（背景，不预测） |

**可行性约束**：4 个目标变量最大和 = 60+35+15+5 = 115%，需联合约束保证 N₂ ≥ 0，且建议 N₂ 留有合理范围（如 ≥ 1%）。

---

## 关键架构决策

1. **组件顺序**：内部仍为 4 列 tuple `(x_H2, x_CH4, x_CO2, x_CO)`，把原第 4 列 N₂ 位替换为 CO。下游 ILR/ALR 数学结构与 free-component 列切片 **作废**（因为 sum<100，不再有闭包），但**列序保留**降低改动面。
2. **N₂ 双重角色**：仍是仿真层物理变量（声学声速/衰减、光学交叉项），但不是 DL/ML 标签。`COMPONENT_FIELDS` 改为 4 个预测目标；`CONDITION_GRID_FIELDS` 单独保留 `x_N2` 作为 condition 列（不入 labels）。
3. **CO NDIR 通道**：新增 `V_NDIR_CO` 慢通道，HITRAN 谱在 ~4.7 μm / 2143 cm⁻¹（CO 基频吸收带）。`SLOW_CHANNELS` 从 8→9，`SLOW_DYNAMIC_CHANNELS` 从 3→4。
4. **Loss 路线**：sum<100 后，`compositional_mse` / `ilr_mse` / `free_component_mse` / `weighted_free_component_mse` 全部失效。新基线损失走 `weighted_component_mse`（直接监督全 4 列），或 `mse`。`alr_ch4` 与 `ilr_n2_first` 目标变换标记为 deprecated 但代码保留（向后兼容老 benchmark）。
5. **Benchmark 命名**：现 benchmark = `wv4-smoke`（wet v4 hydrogen-natural-gas）。新 benchmark = `sg4-smoke` (synthesis gas 4-component) 或 `coalgas-smoke`，与旧数据并存，不覆盖。

---

## 实现路线（按依赖顺序分阶段）

### Phase A — 物理建模与采样（sim 层）

**A1. Schema 与组件字段** — `src/sim/core/schema.py`

- `COMPONENT_FIELDS = ("x_H2", "x_CH4", "x_CO2", "x_CO")` ← 把 `x_N2` 改为 `x_CO`
- 新增常量 `BACKGROUND_FIELDS = ("x_N2",)`
- `CONDITION_GRID_FIELDS` 中显式加 `"x_N2"`（不通过 `*COMPONENT_FIELDS`）
- `SLOW_CHANNELS` 加 `"V_NDIR_CO"`（位置：紧跟 `V_NDIR_CO2`）
- `SLOW_DYNAMIC_CHANNELS` 加 `"V_NDIR_CO"`
- `SLOW_MODAL_GROUPS["optical"]` 加 `"V_NDIR_CO"`

**A2. LHS 采样** — `src/sim/generation/conditions.py`

需要文献验证后落到联合可行性约束的实现（推断方案：拒绝采样）：

- `LatinHypercube(d=4)`：四维 LHS 分别映射到 H₂/CO/CO₂/CH₄ 的边际区间（用户给的 25-35 / 45-60 / 5-15 / 0-5）
- 计算 `x_N2 = 100 - x_H2 - x_CO - x_CO2 - x_CH4`
- 若 `x_N2 < N2_MIN_PERCENT`（建议 1.0），拒绝该样本，重新抽样补足 N
- 移除原 H₂ 双峰分布逻辑（用户场景不需要）
- 移除原 CH₄ 闭包逻辑（CH₄ 是采样变量，不再做余量）

**文献验证项**：合成气/煤气化气典型组分分布、N₂ 稀释比例、是否有现成 LHS 拒绝采样实现可复用。建议规划阶段先输出"可行性验证报告"再实现。

**A3. 声学物理** — `src/sim/generation/acoustic_physics.py`

- 新增 CO 声速常数：`_SPEED_CO_MS = 352.0`（25°C 估值，需文献核对）
- `hidden_sound_speed_v2` 形参增加 `x_co`，N₂ 项保留（仍参与混合）
- 新增 CO 弛豫常数（`alpha_lambda_max_co`、`f_relax_co_per_atm`，需文献查证或暂定为 N₂ 同量级）
- `hidden_attenuation_v2` 加 CO 衰减项

**文献验证项**：CO 在 40 kHz 声学频段的弛豫吸收系数、温度修正、与 H₂O 耦合。

**A4. 光学后端** — `configs/data/spectral-defaults.json` + `src/sim/generation/spectral/defaults.py`

JSON 新增条目（结构对齐现有 `ch4`/`co2`）：

```json
"gas_specs": [
  ...,
  {"gas": "CO", "table_name": "CO", "molecule_id": 5, "isotopologue_id": 1}
],
"filters": {
  "co": {"channel": "co", "center_cm1": 2143.0, "fwhm_cm1": 80.0}
},
"hitran_grids": {
  "co": {"wavenumber_min_cm1": 2050.0, "wavenumber_max_cm1": 2230.0, ...}
}
```

filter 参数 `center_cm1=2143`（CO 基频带中心）、`fwhm_cm1=80`（占位，需替换为目标传感器型号 datasheet）。

`optical_backend.py:collect_hitran_cache_requirements` 的 `channels` 参数从 `("ch4", "co2")` 扩到 `("ch4", "co2", "co")`。

**A5. 慢通道响应曲线** — `src/sim/generation/slow.py`

- `TAU_RISE_SYSTEM_S` / `TAU_DECAY_SYSTEM_S` / `NOISE_FRACTION` 加 `V_NDIR_CO` 条目（默认与 V_NDIR_CO2 同量级）
- `_main_feature_condition`（行 ~247）写入 dict 新增 `x_CO` 字段、保留 `x_N2`
- `equilibrium_features` 中 V_NDIR_CO 的响应曲线（参考 V_NDIR_CO2 模式）

**A6. HITRAN 缓存预生成** — `src/pipeline/precompute_hitran_benchmark_cache.py`

需要重新跑一遍（用户手动触发）：

```powershell
python -m hg.pipeline.precompute_hitran_benchmark_cache --benchmark sg4-smoke
```

---

### Phase B — 通用工具层（common）

**B1. composition.py**

- 保留 `ALR_CH4_TRANSFORM` / `ILR_N2_FIRST_TRANSFORM` / `N2_FIRST_ILR_BASIS`（不删除，标记 `deprecated_for_v4_sum100_only`）
- `replace_zeros_multiplicative` / `close_to_unit_interval` 仍可用，但调用方需自己保证 sum=1
- 新增 `validate_open_composition(values, *, min_n2_percent=1.0)`：检查 4 列 sum<100 且隐含 N₂ ≥ 阈值

**B2. metrics.py**

- `conditional_component_metrics` 的 `n2_bins` 改名为 `co_bins`，`component_name="x_N2"` 改为 `"x_CO"`
- `ch4_bins` 保留（CH₄ 仍在）

---

### Phase C — 训练层（dl）

**C1. losses.py**

- 标记 deprecated：`COMPOSITIONAL_MSE_LOSS`、`ILR_MSE_LOSS`、`FREE_COMPONENT_MSE_LOSS`、`WEIGHTED_FREE_COMPONENT_MSE_LOSS`
- `validate_loss_target_transform` 在 sum<100 模式下拒绝以上 loss
- 推荐新基线 loss：`weighted_component_mse`（监督全 4 列，`component_weights` 按训练集方差倒数生成）
- 移除 `DEFAULT_FREE_COMPONENT_COUNT` 的隐式 N₂ 闭包注释，改写为"4 个独立目标，无闭包假设"

**C2. trainer.py**

- ILR 逆变换路径在 sum<100 模式下跳过（保留旧 benchmark 兼容性）
- `conditional_component_metrics` 调用方更新 key 名（`n2_bins → co_bins`）

**C3. dataset.py / scalers.py**

- labels 维度仍为 4，但语义不同；YScaler 不变
- 新增数据校验：加载 benchmark 时检查 `labels/y.npy` shape=(N, 4)，并断言 `metadata/component_fields == ("x_H2","x_CH4","x_CO2","x_CO")`

---

### Phase D — Pipeline 与配置

**D1. 评估管线** — `src/pipeline/run_experiment.py`

- 报告字段 `x_n2_r2` → `x_co_r2`（行 ~327, 354, 394）
- `_summary_metrics` 中 N₂ 引用全部替换为 CO

**D2. N2 改进分析工具**

- `src/pipeline/analyze_n2_improvement.py` 改名为 `analyze_co_improvement.py`，函数同步重命名
- `src/pipeline/run_n2_improvement_workflow.py` 类似处理
- 旧文件保留并标记 deprecated（避免破坏 PhaseWindow 历史实验脚本）

**D3. 配置文件**

- `configs/` 下所有现有 JSON 保留（属于旧 benchmark 范围），不批量改
- 新建 `configs/sg4/` 目录，放新 benchmark 的实验配置：
  - `sg4_baseline.json`（对应旧的 P0-B baseline）
  - `sg4_cnn1d_tcn.json`
  - `sg4_ridge_baseline.json`（ML 基线对照）
- `component_weights` 在新配置里按合成气场景的训练集方差重新生成（启动后第一次实验完成即可估出）

**D4. Benchmark 生成入口** — `src/pipeline/generate_benchmark.py`

- 新增 `--composition-scheme {hydrogen_ng,syngas}` 选项，syngas 走新 LHS 采样
- 输出目录 `data/sg4-smoke/`，与 `data/wv4-smoke/` 并列

---

### Phase E — 测试

**E1. 新增测试**

- `tests/test_syngas_sampling.py`：验证 LHS 采样落在用户给定区间、N₂ 残量 ≥ 阈值、拒绝采样收敛
- `tests/test_acoustic_physics_syngas.py`：CO 声速/衰减项参与混合公式
- `tests/test_spectral_co_channel.py`：CO HITRAN 谱缓存生成、NDIR 通道吸收度计算

**E2. 修改现有测试**

- `tests/test_benchmark_generation.py`：参数化为两种 scheme，断言对应字段
- `tests/test_dl_losses.py`：sum<100 模式下 `weighted_component_mse` 通过、`free_component_mse` 拒绝

---

### Phase F — 文档

- 新增 `docs/syngas/adaptation_plan.md`（本规划文档落地版）
- 新增 `docs/syngas/physics_references.md`（CO 声速/弛豫/光学常数的文献来源 — **需要文献检索 skill 协助**）
- 更新 `README.md` 项目使命段、`docs/ARCHITECTURE.md` 范围段
- `context_brief.md`（recallloom）更新使命描述

---

## 关键文件清单（按改动量排序）

| 文件                                                                               | 改动类型          | 说明                               |
| -------------------------------------------------------------------------------- | ------------- | -------------------------------- |
| [src/sim/core/schema.py](src/sim/core/schema.py)                                 | 改             | COMPONENT_FIELDS + SLOW_CHANNELS |
| [src/sim/generation/conditions.py](src/sim/generation/conditions.py)             | 重写            | LHS 采样改为合成气场景                    |
| [src/sim/generation/acoustic_physics.py](src/sim/generation/acoustic_physics.py) | 加 CO 项        | 声速、衰减                            |
| [src/sim/generation/slow.py](src/sim/generation/slow.py)                         | 加 V_NDIR_CO   | 慢通道响应曲线                          |
| [configs/data/spectral-defaults.json](configs/data/spectral-defaults.json)       | 加 CO          | gas_spec / filter / grid         |
| [src/dl/training/losses.py](src/dl/training/losses.py)                           | 标记 deprecated | sum<100 模式校验                     |
| [src/common/metrics.py](src/common/metrics.py)                                   | 改             | n2_bins → co_bins                |
| [src/pipeline/run_experiment.py](src/pipeline/run_experiment.py)                 | 改             | 报告字段重命名                          |
| [src/pipeline/analyze_n2_improvement.py](src/pipeline/analyze_n2_improvement.py) | 改名            | analyze_co_improvement.py        |
| `configs/sg4/*.json`                                                  | 新建            | 新场景实验入口                          |
| `tests/test_syngas_*.py`                                                         | 新建            | 采样/物理/光学新测试                      |
| `data/sg4-smoke/`                                                                | 新建            | 重生成 benchmark                    |

---

## 验证流程

1. **采样可行性**：`python -m hg.pipeline.generate_benchmark --composition-scheme syngas --sequence-count 100 --dry-run`，确认 N₂ 残量分布、采样无失败
2. **HITRAN 缓存**：`python -m hg.pipeline.precompute_hitran_benchmark_cache --benchmark sg4-smoke` 跑通，`data/hitran_cache/co/` 生成
3. **小规模 benchmark**：`--sequence-count 100` 完整跑通，检查 `manifest.json` 字段、`labels/y.npy` shape
4. **物理单元测试**：`pytest tests/test_syngas_sampling.py tests/test_acoustic_physics_syngas.py tests/test_spectral_co_channel.py -v`
5. **DL 基线训练**：`python -m hg.dl.cli train --config configs/sg4/sg4_baseline.json --epochs 20`，确认 loss 下降、4 组分预测 shape 正确、N₂ 不出现在 labels
6. **回归测试**：`pytest tests/ -v`，确认旧 wv4-smoke 流水线未受影响

---

## 风险与未决项

- **CO 物理常数文献覆盖**：声速、声学弛豫、HITRAN 谱线参数需要文献支撑。建议先用 `cnki-search` + `nature-academic-search` skill 做一轮检索，落到 `docs/syngas/physics_references.md` 再编码。
- **采样可行性**：用户给的边际区间存在最大求和 115%，需联合约束。拒绝采样的接受率需经验证（可能需要把上界压缩，例如 H₂ 上限改 30%、CO 上限改 55%）。
- **NDIR CO 通道滤光片实际型号**：当前规划用 InfraTec 占位参数，正式 benchmark 前需替换为合成气检测目标传感器的真实 datasheet。
- **旧 wv4-smoke 数据是否保留**：本规划假设并存（两套 benchmark + 两组 configs/experiment），如用户希望整仓切换，回收旧代码量额外。
- **改动面**：约 25 个源码文件 + 5 个新文件 + ~30 处文档更新，预计 1-2 个工作日（不含文献调研与物理常数核实）。

---

## 不在本规划范围

- 模型架构本身（CNN1D / TCN / PhaseWindowTCN 等）不变
- 训练流水线（trainer、metrics 计算逻辑、scheduler）不变
- 现有 wv4-smoke 实验结果与 P3 数据增强分析不受影响
- 真实合成气数据采集 / 标定方案

---

## 决策落地（2026-06-25 更新）

文献检索完成，4 份调研报告 + 1 份汇总位于 [docs/syngas/references/](syngas/references/)。本节记录基于文献的最终决策，对原规划的修正以此节为准。

### 决策 1：LHS 采样走两轮（A → B）

- **第一轮（A 保守，sg4a-smoke）**：CO 40-65 / H₂ 24-38 / CO₂ 3-18 / CH₄ 0-5 / N₂ balance。与用户原区间一致，对应 Texaco/Shell/E-Gas 气流床合成气。目的：跑通整套流水线，验证架构可行性。
- **第二轮（B 推荐，sg4b-formal）**：CO 15-65 / H₂ 5-55 / CO₂ 2-30 / CH₄ 0-12 / N₂ 0.2-5。覆盖煤气化技术全谱（Lurgi 固定床 + 流化床 + 气流床 + 蒸汽气化）。目的：泛化验证。
- **air-blown producer gas（方案 C，N₂ 40-60%）暂不做**，需要时单独建子数据集。

### 决策 2：CO NDIR 通道做完整四气体串扰

V_NDIR_CO 通道（1980-2310 cm⁻¹）仿真时，HITRAN 谱图同时考虑 CO + CO₂ + H₂O + CH₄ 四气体的吸收叠加。与现有 CH4/CO2 通道架构对称。HITRAN cache 体积估算 ~50 MB，可接受。

### 决策 3：联合可行性约束（覆盖原 A2 章节）

`src/sim/generation/conditions.py` 的拒绝采样规则按以下顺序应用：

1. **质量守恒**：`CO + H₂ + CO₂ + CH₄ + N₂ = 100%`（vol%, 干基），独立采前 4 个，N₂ 由 balance 推出
2. **N₂ 残量**：方案 A 下 N₂ ∈ [0.2, balance_max]，方案 B 下 N₂ ∈ [0.2, 5]
3. **H₂/CO ∈ [0.1, 4.0]**：覆盖工业实测范围
4. **CO₂/CO ∈ [0.02, 1.5]**：防止极端非工业组合
5. **C 平衡**：`CO + CO₂ + CH₄ ∈ [35, 75]%`

### 决策 4：可直接编码的物理常数（高置信，来自文献）

**光学侧** — [docs/syngas/references/co_optical_hitran.md §4](syngas/references/co_optical_hitran.md)：

```python
# configs/data/spectral-defaults.json 直接追加
gas_specs.append({"gas": "CO", "table_name": "CO", "molecule_id": 5, "isotopologue_id": 1})
filters["co"] = {"channel": "co", "center_cm1": 2145.92, "fwhm_cm1": 82.89}  # InfraTec I 4.66μm/180nm
hitran_grids["co"] = {
    "wavenumber_min_cm1": 1980.0,
    "wavenumber_max_cm1": 2310.0,
    "wavenumber_step_cm1": 0.1,
    "temperature_k": 296.0,
    "pressure_atm": 1.0,
}
```

**声学侧** — [docs/syngas/references/co_acoustic_constants.md §一](syngas/references/co_acoustic_constants.md)：

```python
# src/sim/generation/acoustic_physics.py 直接追加
_SPEED_CO_MS = 352.0  # 25°C, 1 atm；与 N₂ 仅差 1 m/s
PROCESSING_PARAMS_V2.update({
    "alpha_lambda_max_co": 0.025,        # 中置信，建议 ablation 扫描 0.015-0.040
    "f_relax_co_per_atm": 12000.0,       # 高置信，CO 在 dry N₂ 背景 (Qiao 2022)
    "k_h2o_to_f_relax_co": 0.30,         # 中置信，H₂O 对 CO 加速比 CO₂ 强 20× (Yin 2016)
})
```

### 决策 5：HITRAN 网格扩展（补充原 A4）

CO 通道下 CO₂ 与 H₂O 在 1980-2310 cm⁻¹ 区间的吸收要纳入仿真，但**不修改现有 `hitran_grids.co2` 边界**（避免影响 wv4 旧数据）。实现方式：

- 在 `optical_backend.py:collect_hitran_cache_requirements` 增加 `co_channel_crosstalk_gases = ("co2", "h2o", "ch4")` 参数
- 对每个串扰气体在 CO 通道网格 [1980, 2310] cm⁻¹ 单独 fetch 一份谱并缓存（命名空间 `data/hitran_cache/co_channel/{gas}/`）
- V_NDIR_CO 通道前向模型把 CO + 三个串扰气体的吸收叠加

### 决策 6：CO 物理可观测性提示（架构层）

CO 与 N₂ 在声速（352 vs 353 m/s）与热导（M、γ 相同）维度近似简并 → **CO 高精度只能靠 NDIR + 光纤麦克风（光声）+ 弛豫吸收**。这一物理事实需要：

- 在 `docs/ARCHITECTURE.md` 新场景章节明确写出
- DL/ML 模型评估时单独跟踪 CO 在仅光学 vs 多模态融合下的 R² 差异
- 工业精度天花板：传感器组合的物理可观测性已先决定 CO R² 上限

### 决策 7：Benchmark 命名

- 第一轮：`data/sg4a-smoke/`（synthesis-gas 4-component, scheme A, smoke smoke 规模）
- 第二轮：`data/sg4b-formal/`（scheme B 正式规模）
- 与 `data/wv4-smoke/`、`data/wv4-formal/` 并存，不覆盖旧数据
- HITRAN cache 命名 `data/hitran_cache/co/` 与 `data/hitran_cache/co_channel/{co2,h2o,ch4}/`

### 仍未决（不阻塞编码）

1. `alpha_lambda_max_co=0.025` 是物理类比占位值（中置信），第一轮训练完成后做 ablation 扫描 0.015-0.040
2. CO NDIR 滤光片当前用 InfraTec I 占位，正式工程项目前需替换为目标传感器实际 datasheet
3. CNKI 中文文献（国内煤化工厂实测）未覆盖，方案 B 浓度区间以欧美 NETL 文献为主
4. 国标对标：GB/T 40789-2021（CO/CO₂/O₂ AMS 性能）+ HJ 1241-2022（CO/HCl CEMS）；专门覆盖合成气在线监测的整套国标未找到

### 下一步

可以开始 Phase A 编码。建议执行顺序：

1. `src/sim/core/schema.py` 改 `COMPONENT_FIELDS` + `SLOW_CHANNELS`
2. `src/sim/generation/conditions.py` 重写为方案 A 的 LHS + 拒绝采样
3. `configs/data/spectral-defaults.json` 合并决策 4 的 JSON 片段
4. `src/sim/generation/acoustic_physics.py` 加 CO 常数与混合公式项
5. 运行 `precompute_hitran_benchmark_cache --benchmark sg4a-smoke` 预生成谱图
6. 小规模 benchmark 跑通后，进入 Phase B-F
