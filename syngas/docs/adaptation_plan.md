# 合成气四元检测目标适配方案（整合审查修正版）

> 本文档是 `.claude/plans/stateful-prancing-papert.md`（原始规划）与 `stateful-prancing-papert-review.md`（审查报告）的整合落地版。
> 所有审查修正项已直接合并到对应章节，以 `[审查修正]` 标记。
> 文献常数已从 `docs/syngas/references/` 同步，以 `[文献确认]` 标记。

> ⚠️ **2026-06-27 时间轴对齐变更**：sg4 系列正式数据集 timesteps 已从 128 改为 512（与 hg `wv4-formal-hitran-standard-6000` 一致）。下文"实施进度"中描述的 sg4-formal (128 步) 数据集与 Stage Ⅰ-3 / Stage Ⅱ 基于它的 41 个 run 结果**已废弃，待 512 步重跑**。文中具体命令、shape 记录保留为历史事实。新生成命令以 `configs/experiment/sg4/README.md` 与 `docs/syngas/README.md` 顶部为准。

---

## 实施进度（截至 2026-06-27）

### 总体状态

阶段 1–4 已完成并验证；阶段 Ⅰ 已完成 14/15 runs：sg4-formal benchmark 已生成、5 模型 × 3 seeds 基线训练 TCN ≈ Ridge pool R²≈0.96，PatchTST 修复后 0.935，CNN1D / LSTM ≈ 0.93。**阶段 Ⅱ 已完成 27/27 runs**（CO 通道 / 串扰 / Loss 三组 ablation），见 [stage_ii_ablation_results.md](stage_ii_ablation_results.md)。HITRAN 后端（阶段 3c）作为独立后续步骤。

| 阶段 | 范围 | 状态 |
|---|---|---|
| 1 | schema + 采样 | 完成，18 测试通过 |
| 2 | 物理（声学/光学/热导/吸收/3×3 串扰） | 完成，49 测试通过 |
| 3a/3b | slow + benchmark + CLI（empirical 后端） | 完成，15 测试通过，端到端可生成 `data/sg4-smoke/` |
| 3c | HITRAN 后端 + CO/CO₂/H₂O 三气体谱线缓存 | 未完成，syngas slow.py 在 HITRAN backend 下抛 `NotImplementedError` |
| 4 | DL 训练适配（dataset/loss/trainer/metrics/cli） | 完成，9 测试通过，`python -m dl.cli --config configs/experiment/sg4/sg4_baseline.json` 可跑 |
| **Ⅰ-1** | sg4-formal benchmark 生成（6000 序列） | 完成，validation pass，split 4200/900/600/300 |
| **Ⅰ-2** | 配置矩阵：CNN1D / TCN / LSTM / PatchTST / Ridge | 完成，5 个 sg4 config 就绪 |
| **Ⅰ-3** | 基线训练（5 模型 × 3 seeds = 15 runs） | 完成 14/15：TCN ≈ Ridge pool R²≈0.96，PatchTST 修复后 0.935，CNN1D / LSTM ≈ 0.93（LSTM seed=123 单 seed 不收敛）。详见 [stage_i3_baseline_results.md](stage_i3_baseline_results.md) |
| **Ⅱ-1** | CO 通道 ablation（B 去 V_NDIR_CO / C 仅 V_NDIR_CO+环境，TCN+Ridge × 3 seeds = 12 runs） | 完成。B 组 x_CO R²: 0.954 → **0.484**（TCN/Ridge 同向），C 组 x_CO R²: 0.93-0.94 |
| **Ⅱ-2** | 串扰 ablation（`sg4-formal-crosstalk` + TCN × 3 seeds = 3 runs） | 完成。所有组分 R² 与 baseline 差异 ≤0.006 |
| **Ⅱ-3** | Loss 对比（TCN × {mse,mae,huber,smooth_l1} × 3 seeds = 12 runs） | 完成。未加权 loss 让 x_CH4 R² 跌至 0.39-0.44（基线 0.827） |
| **Ⅱ 配套代码** | channel 子集选择 + crosstalk 透传 + 测试 | 完成，全量 462 passed，hg 零回归 |

全量测试：462 passed = 353 baseline + 109 syngas 增量（含 Ⅱ 的 18 个新测试）。详见 [experiment_roadmap.md §阶段 Ⅰ 执行记录](experiment_roadmap.md#阶段-ⅰ-执行记录2026-06-26)、[experiment_roadmap.md §阶段 Ⅱ 执行记录](experiment_roadmap.md#阶段-ⅱ-执行记录2026-06-27)、[stage_i3_baseline_results.md](stage_i3_baseline_results.md) 和 [stage_ii_ablation_results.md](stage_ii_ablation_results.md)。

### 实施路线偏离

本次实施未按本文档第二节"组分列序"的字面意思**直接替换全局 `COMPONENT_FIELDS`**，而是采用**分支隔离**：

- `src/sim/core/syngas_schema.py`：syngas 专用 schema。
- `src/sim/generation/syngas/`：syngas 物理与 benchmark 子包（`conditions.py` / `acoustic_physics.py` / `optical_crosstalk.py` / `slow.py` / `benchmark.py`）。
- `src/pipeline/generate_syngas_benchmark.py`：syngas CLI 入口。
- `configs/experiment/sg4/sg4_baseline.json`：syngas DL 配置。
- `tests/test_syngas_*.py`：4 个新测试文件。

原因：hydrogen_ng `wv4-smoke` benchmark、187+ 既有测试与全局 `COMPONENT_FIELDS` 强耦合；直接替换会破坏旧数据语义和回归基线。分支隔离让两套场景独立演进，并存运行。

### 共用代码改造（向后兼容）

下列共用文件做了显式参数化，**默认行为与原版完全一致**：

| 文件 | 新增能力 | hydrogen_ng 默认行为 |
|---|---|---|
| `src/sim/generation/waveforms.py` | 可注入 `sound_speed_fn` / `attenuation_fn` / `extra_gas_kwargs` | 沿用 hg `hidden_sound_speed_v2` / `hidden_attenuation_v2`，签名不变 |
| `src/sim/packaging/manifest.py` | 新增 `schema_version` / `composition_scheme` / `background_fields` 可选参数 | `composition_scheme="hydrogen_ng"`，`background_fields=[]` |
| `src/sim/validation/integrity.py` | `validate_benchmark_assets` 接受 `component_fields` / `slow_channels` / `background_fields` / `require_sum_100` | 默认值取 hg schema |
| `src/common/metrics.py` | `conditional_component_metrics` 新增 `bin_components` 可选参数 | 默认 `("x_N2", "x_CH4")` |
| `src/dl/training/losses.py` | 新增 `validate_loss_composition_scheme()`，`validate_loss_model_output` 接受 `composition_scheme` | 默认 `hydrogen_ng`，闭包类 loss 限制不变 |
| `src/dl/training/trainer.py` | `Trainer` 构造接受 `component_names` / `composition_scheme`；`evaluate` / `_metric_predictions` / `_compositional_metrics` 三处 `.cpu().numpy()` 加 `.float()` 转 fp32 | 默认 hg；evaluate 行为不变；fp32 转换不改变数值精度，仅解除 AMP bf16 的 numpy 兼容性限制 |
| `src/dl/cli.py` | 自动从 manifest 读 `composition_scheme` + label_names 注入 trainer | 旧 wv4 manifest 无 `composition_scheme` 字段时 fallback 到 hg |
| `src/ml/training.py` | 新增 `_default_bin_components(label_names)` helper；`SplitEvaluation.sum_abs_error` 改为 `float | None`；按 label 是否含 `x_N2` 选 conditional 分箱 / 跳过 sum=100% 检查 | 含 `x_N2` 时走原 hg 路径 |

> **2026-06-26 追加修复（共两处）**：
>
> 1. **`ml/training.py` conditional bin_components**：`evaluate_regressor` 原本无条件调用 `conditional_component_metrics(predictions, y, label_names)`，默认 bin_components `("x_N2","x_CH4")` 与 syngas labels `(x_H2,x_CH4,x_CO2,x_CO)` 冲突，Ridge 在 sg4 上首次运行直接报 `ValueError: Unknown component 'x_N2'`。修复后：`_default_bin_components` 根据 label_names 自动选 `("x_N2","x_CH4")`（hg）或 `("x_CO","x_CH4")`（syngas），与 trainer 的 `_conditional_bin_components` 行为对齐。
> 2. **`dl/training/trainer.py` AMP bf16 numpy 兼容**：`evaluate` / `_metric_predictions` / `_compositional_metrics` 三处 `.detach().cpu().numpy()` 在 AMP `bfloat16` 下报 `TypeError: Got unsupported ScalarType BFloat16`。修复后每处加 `.float()` 显式转 fp32，PatchTST 才能使用 bf16/fp16 AMP（即便最终配置选择 fp32，该修复仍是 trainer 健壮性必需）。
>
> 两处修复后全量回归 444 passed 通过，hg 零破坏。

### 已落地的关键契约

1. **预测目标**：`("x_H2", "x_CH4", "x_CO2", "x_CO")`，4 列 sum<100%。
2. **背景气**：`x_N2 = 100 - sum(targets)`，写入 condition grid 但不入 labels。
3. **慢通道**：9 个（含 `V_NDIR_CO`）。
4. **CO 物理常数**：
   - `_SPEED_CO_MS = 352.0`（与 N₂ 差 <1 m/s）
   - `alpha_lambda_max_co = 0.025`（中置信，ablation 区间 0.015–0.040）
   - `f_relax_co_per_atm = 12000.0`（高置信，dry N₂ 背景）
   - `k_h2o_to_f_relax_co = 0.30`（中置信，线性近似）
   - 热导 `x_co` 系数 -0.00005（占位，需文献标定）
5. **CO 光学**：滤光片 `center_cm1=2145.92, fwhm_cm1=82.89`（InfraTec I 4.66 μm / 180 nm 行业参考占位）。
6. **3×3 串扰**：`enable_co_crosstalk=False` 为默认（Step 1）；切到 `True` 启用 CO₂↔CO 互扰（Step 2）。当前 benchmark 默认 Step 1。
7. **采样**：方案 B（CO 15–65 / H₂ 5–55 / CO₂ 2–30 / CH₄ 0–12 / N₂ ≥0.2 %）+ 条件顺序采样。
8. **Loss 路线**：闭包类 loss（`compositional_mse` / `ilr_mse` / `free_component_mse` / `weighted_free_component_mse`）在 syngas 场景被自动拒绝；推荐 `mse`、`weighted_component_mse`、`mae`、`smooth_l1`、`huber`。
9. **Target transform**：syngas 不允许（ILR/ALR 依赖 sum=100% 闭包）。

### 端到端验证记录

#### sg4-smoke（链路冒烟）

```text
# 生成 sg4-smoke
python -m pipeline.generate_syngas_benchmark \
    --output-root data --dataset sg4-smoke --sequences 16 --seed 20260626 \
    --timesteps 16 --dt-s 0.5 --optical-absorption-backend empirical_v1 --workers 1
→ data/sg4-smoke/manifest.json:
    composition_scheme: syngas
    schema_version: v4-syngas-1
    labels: [x_H2, x_CH4, x_CO2, x_CO]
    background_fields: [x_N2]
    slow_channels: [..., V_NDIR_CO, ...]   # 9 个
→ data/sg4-smoke/labels/y.npy shape (16, 4)
→ data/sg4-smoke/sequences/slow.npy shape (16, 16, 9)
→ data/sg4-smoke/sequence_labels.csv: 仅 sequence_id + 4 列目标，不含 x_N2

# DL 训练
python -m dl.cli --config configs/experiment/sg4/sg4_baseline.json --epochs 3 --batch-size 4
epoch=1 train_loss=6.08 val_loss=5.49
epoch=2 train_loss=5.52 val_loss=4.28
epoch=3 train_loss=5.12 val_loss=4.05
→ outputs/sg4_baseline/metrics.json:
    component_metrics keys: x_H2 / x_CH4 / x_CO2 / x_CO
    conditional_metrics keys: co_bins / ch4_bins   # 按 CO 分箱
    sum_abs_error: null   # syngas 不计算
    compositional_metrics: null
```

> 注：3 epoch + 16 序列 + CPU 跑出负 R² 属预期；本次仅做闭环可用性验证，不评估收敛质量。

#### sg4-formal（正式实验数据，2026-06-26）

```text
# 生成 sg4-formal（6000 序列 / 128 时步 / 24 workers，实测 <2 分钟）
python -m pipeline.generate_syngas_benchmark \
    --output-root data --dataset sg4-formal --sequences 6000 --seed 20260626 \
    --timesteps 128 --dt-s 0.5 --optical-absorption-backend empirical_v1 \
    --storage memmap --workers 24
→ data/sg4-formal/manifest.json:
    composition_scheme: syngas
    schema_version: v4-syngas-1
    sequence_count: 6000
    timesteps: 128
    sampling_strategy: lhs
    optical_absorption_backend: empirical_v1
→ data/sg4-formal/labels/y.npy shape (6000, 4)
→ data/sg4-formal/sequences/slow.npy shape (6000, 128, 9)
→ data/sg4-formal/quality/validation_summary.json.status: pass
→ split 划分：train 4200 / val 900 / test 600 / extrapolation 300

# 5 模型 × 3 seeds 基线训练（运行中，编排脚本 scripts/run_sg4_baseline.py）
python scripts/run_sg4_baseline.py
→ outputs/sg4_baseline/{cnn1d,tcn,lstm,patchtst,ridge}/seed{42,123,2026}/metrics.json
→ outputs/sg4_baseline/summary.json   # 5 模型 × 3 seeds 汇总，覆盖 per-component metrics
```

CNN1D 3-epoch 预热的 test split 信号（仅链路校验）：x_H2 R²=0.84 / x_CO2 R²=0.89 / x_CO R²=0.82 / x_CH4 R²=0.39。CH₄ 低浓度区间表现与风险矩阵预期一致；x_CO 与 V_NDIR_CO 通道贡献吻合，待 Ⅱ-1 ablation 验证 CO/N₂ 简并假说。

### 未完成（阶段 3c：HITRAN syngas 后端）

`sim.generation.syngas.slow.build_sequence_arrays` 在 `optical_absorption_backend="hitran_hapi_v1"` 时抛 `NotImplementedError`。要落地需要：

- 联网用 HAPI fetch CO / CO₂ / H₂O 在 [1980, 2310] cm⁻¹ 的谱线（约 50 MB 缓存）。
- 新增 syngas 专用 spectral defaults 或扩展现有 `configs/data/spectral-defaults.json`（CO 通道 + CO₂ 网格下界从 2250 扩到 1980）。
- `optical_backend.compute_hitran_optical_absorption` 支持 channels=("ch4","co2","co")，并把 CO₂/H₂O 在 CO 通道的吸收纳入前向模型。
- 适合在正式实验前作为独立步骤补充。

---

## 一、背景

当前 v4 项目检测 H₂/CH₄/CO₂/N₂（掺氢天然气，sum=100% 闭包）。

适配新场景：**合成气 / 煤气化制气**，检测目标改为 **H₂/CH₄/CO₂/CO**，N₂ 保留为背景稀释气（不预测），4 个目标变量 sum<100%。

## 二、关键架构决策

| # | 决策 | 说明 |
|---|------|------|
| 1 | 组分列序 | `COMPONENT_FIELDS = ("x_H2", "x_CH4", "x_CO2", "x_CO")`，第 4 列 N₂→CO |
| 2 | N₂ 双重角色 | 不入预测目标，但参与所有物理仿真（声学、衰减、热导）。以 `x_N2 = 100 - sum(targets)` 计算传递 |
| 3 | CO NDIR 通道 | 新增 `V_NDIR_CO`（4.66 μm / 2145.92 cm⁻¹），SLOW_CHANNELS 8→9，SLOW_DYNAMIC_CHANNELS 3→4 |
| 4 | Loss 路线 | 闭包类 loss 全部 deprecated，新基线 `weighted_component_mse`（监督全 4 列） |
| 5 | Benchmark | 新 benchmark = `sg4-smoke`，与 `wv4-smoke` 并存 |

---

## 三、实施路线

### Phase 0 — 前置验证（不阻塞编码但影响参数选择）

| 任务 | 状态 | 产物 |
|------|------|------|
| CO 物理常数文献检索 | **已完成** | `docs/syngas/references/co_acoustic_constants.md` |
| CO 光学 HITRAN 参数检索 | **已完成** | `docs/syngas/references/co_optical_hitran.md` |
| 合成气组分分布文献综述 | **已完成** | `docs/syngas/references/syngas_composition_ranges.md` |
| 商用传感器 & 可行性评估 | **已完成** | `docs/syngas/references/syngas_sensing_survey.md` |
| **LHS 采样方案选择** | **已确认** | 方案 B + 条件顺序采样，详见 §四与 `docs/syngas/lhs_sampling_design.md` |
| **CO 通道串扰建模范围** | **已确认** | 完整串扰，分两步实施，详见 §四与 `docs/syngas/co_crosstalk_design.md` |

### Phase A — 物理建模与采样

#### A1. Schema 与组件字段 — `src/sim/core/schema.py`

```python
COMPONENT_FIELDS = ("x_H2", "x_CH4", "x_CO2", "x_CO")        # x_N2 → x_CO
BACKGROUND_FIELDS = ("x_N2",)                                  # 新增
# CONDITION_GRID_FIELDS 显式加 "x_N2"（不通过 *COMPONENT_FIELDS）
SLOW_CHANNELS = (..., "V_NDIR_CO")                             # 紧跟 V_NDIR_CO2
SLOW_DYNAMIC_CHANNELS = ("V_NDIR_CH4", "V_NDIR_CO2", "V_NDIR_CO", "V_TCS")
SLOW_MODAL_GROUPS["optical"] = ("V_NDIR_CH4", "V_NDIR_CO2", "V_NDIR_CO")
```

`[审查修正]` `SEQUENCE_LABEL_FIELDS = ("sequence_id", *COMPONENT_FIELDS)` 会自动跟随变化——这是正确行为，但需在 PR 中显式验证。

#### A2. LHS 采样 — `src/sim/generation/conditions.py`

- `LatinHypercube(d=4)`：四维 LHS 映射到 H₂/CO/CO₂/CH₄ 的边际区间
- 计算 `x_N2 = 100 - x_H2 - x_CO - x_CO2 - x_CH4`
- 联合可行性约束（拒绝采样）：
  - N₂ ≥ N2_MIN_PERCENT（方案 A: 0.2%）
  - H₂/CO ∈ [0.1, 4.0]
  - CO₂/CO ∈ [0.02, 1.5]
  - 总碳 CO+CO₂+CH₄ ∈ [35%, 75%]
- 移除 H₂ 双峰分布逻辑
- 移除 CH₄ 闭包逻辑

`[审查修正]` 采样函数返回 dict **必须**包含 `x_N2` 键（作为计算值，不是采样值）：

```python
return {
    "x_H2": x_h2, "x_CH4": x_ch4, "x_CO2": x_co2, "x_CO": x_co,
    "x_N2": round(100.0 - x_h2 - x_ch4 - x_co2 - x_co, 6),
}
```

`[审查修正]` 拒绝采样会破坏 LHS 空间填充性。如果接受率 <70%，考虑改用条件顺序采样（先采 CO，再采 H₂ 上限受约束），保证 100% 接受率。

#### A3. 声学物理 — `src/sim/generation/acoustic_physics.py`

`[文献确认]` 新增常数：

```python
_SPEED_CO_MS = 352.0                         # High; 理论+实验交叉确认
"alpha_lambda_max_co": 0.025                 # Medium; ablation 0.015-0.040
"f_relax_co_per_atm": 12000.0                # High; Qiao 2022
"k_h2o_to_f_relax_co": 0.30                  # Medium; 线性近似
```

- `hidden_sound_speed_v2` 形参增加 `x_co`，新增 `x_co_frac * _SPEED_CO_MS` 项
- `hidden_attenuation_v2` 形参增加 `x_co`，新增 CO 弛豫衰减项
- N₂ 项保留（N₂ 仍参与物理混合）

`[审查修正]` CO 与 N₂ 声速差 <1 m/s（同 M=28, 同 γ=1.40）。超声波通道对 CO 几乎不敏感，CO 的可观测性主要由 NDIR 通道提供。

#### A3.5 热导率更新（新增） — `acoustic_physics.py:_hidden_lambda_mix`

`[审查修正]` 原函数仅依赖 x_h2 和 x_co2，需增加 x_co 项：

```python
def _hidden_lambda_mix(x_h2: float, x_co2: float, x_co: float, t_c: float) -> float:
    return 0.034 + 0.00155 * x_h2 - 0.00011 * x_co2 - 0.00005 * x_co + 0.00002 * (t_c - 25.0)
```

CO 热导率 ~25 mW/(m·K) vs N₂ ~26 mW/(m·K)，差异小但 60% 占比下累积效应不可忽略。

#### A4. 光学后端 — `configs/data/spectral-defaults.json` + `src/sim/generation/spectral/`

`[文献确认]` JSON 新增条目：

```json
{"gas": "CO", "table_name": "CO", "molecule_id": 5, "isotopologue_id": 1}
```

滤光片：`center_cm1=2145.92, fwhm_cm1=82.89`（InfraTec I / Boston HIS-E222 交叉确认）。
HITRAN 网格：`[1980.0, 2310.0] cm⁻¹`，步长 0.1。

`[审查修正]` `optical_backend.py:collect_hitran_cache_requirements` 的 channels 从 `("ch4", "co2")` 扩到 `("ch4", "co2", "co")`。CO 通道下需 fetch CO/CO₂/H₂O 三套谱在 [1980,2310] 区间。

#### A4.5 CO 吸收函数与光学串扰（新增）

`[审查修正]` 新增 `_hidden_absorption_co()` 函数（类比现有 CH₄/CO₂）：

```python
def _hidden_absorption_co(x_co: float, h_rh: float, p_mpa: float, t_c: float) -> float:
    return 0.035 * x_co + 0.0005 * h_rh + 0.010 * p_mpa + 0.00015 * (t_c - 25.0)
```

`[审查修正]` 扩展 `optical_crosstalk.py` 的串扰矩阵从 2×2 到 3×3：

```
          CH₄目标    CO₂目标    CO目标
CH₄通道 [  1.0       ε₁₂       ~0    ]
CO₂通道 [  ε₂₁       1.0       ε₂₃   ]   ← ε₂₃ CO₂↔CO 最显著
CO 通道 [  ~0        ε₃₂       1.0   ]
```

#### A5. 慢通道响应曲线 — `src/sim/generation/slow.py`

- `TAU_RISE_SYSTEM_S` / `TAU_DECAY_SYSTEM_S` / `NOISE_FRACTION` 加 `V_NDIR_CO` 条目
- `_main_feature_condition` 写入 dict 新增 `x_CO` 字段，保留 `x_N2`
- `equilibrium_features` 新增 V_NDIR_CO 响应曲线

`[审查修正]` `_main_feature_condition` 必须保证 `x_N2` 作为计算值传入（不依赖 COMPONENT_FIELDS）。

#### A6. HITRAN 缓存预生成

```powershell
python -m pipeline.precompute_hitran_benchmark_cache --benchmark sg4-smoke
```

---

### Phase B — 通用工具层

#### B1. composition.py

- 保留 `ALR_CH4_TRANSFORM` / `ILR_N2_FIRST_TRANSFORM` / `N2_FIRST_ILR_BASIS`（标记 deprecated）
- 新增 `validate_open_composition(values, *, min_n2_percent=0.2)`

#### B2. metrics.py

- `conditional_component_metrics` 的 `n2_bins` → `co_bins`
- `component_name="x_N2"` → `"x_CO"`

---

### Phase C — 训练层

#### C1. losses.py

- 标记 deprecated：`compositional_mse`, `ilr_mse`, `free_component_mse`, `weighted_free_component_mse`
- `validate_loss_target_transform` 在 sum<100 模式下拒绝以上 loss
- 新基线：`weighted_component_mse`（component_weights 按训练集方差倒数）

#### C2. trainer.py

- ILR 逆变换路径在 sum<100 模式下跳过
- `conditional_component_metrics` key 更新

#### C3. dataset.py / scalers.py

- 新增数据校验：加载 benchmark 时断言 `metadata/component_fields == ("x_H2","x_CH4","x_CO2","x_CO")`

---

### Phase D — Pipeline 与配置

#### D1. 评估管线 — `src/pipeline/run_experiment.py`

- 报告字段 `x_n2_r2` → `x_co_r2`
- `_summary_metrics` 中 N₂ 引用全部替换为 CO

#### D2. 分析工具重命名

- `analyze_n2_improvement.py` → `analyze_co_improvement.py`
- `run_n2_improvement_workflow.py` → 同步重命名
- 旧文件保留并标记 deprecated

#### D3. 新 benchmark 配置

```
configs/experiment/sg4/
├── sg4_baseline.json
├── sg4_cnn1d_tcn.json
└── sg4_ridge_baseline.json
```

#### D4. Benchmark 生成入口 — `generate_benchmark.py`

- 新增 `--composition-scheme {hydrogen_ng,syngas}` 选项
- 输出目录 `data/sg4-smoke/`

---

### Phase E — 测试

#### E1. 新增测试

| 测试文件 | 验证内容 |
|----------|---------|
| `tests/test_syngas_sampling.py` | LHS 落在目标区间、N₂ ≥ 阈值、可行性约束、空间填充度 |
| `tests/test_acoustic_physics_syngas.py` | CO 声速/衰减项参与混合公式 |
| `tests/test_spectral_co_channel.py` | CO HITRAN 缓存生成、NDIR 通道吸收度 |

#### E2. 修改现有测试

- `test_benchmark_generation.py`：参数化为两种 scheme
- `test_dl_losses.py`：sum<100 模式下 `weighted_component_mse` 通过、闭包类拒绝

---

### Phase F — 文档更新

- [x] `docs/syngas/adaptation_plan.md`（本文档）
- [x] `docs/syngas/physics_references.md`（物理常数速查）
- [x] `docs/syngas/lhs_sampling_design.md`（LHS 采样设计）
- [x] `docs/syngas/co_crosstalk_design.md`（CO 串扰设计）
- [x] `docs/syngas/references/`（文献报告统一目录）
- [x] `docs/syngas/README.md`（统一文档导航）
- [x] `CLAUDE.md`（项目级 AI 协作指引）
- [x] `docs/ARCHITECTURE.md`（合成气适配段更新到已落地状态）
- [x] `README.md`（使命段 + 当前状态新增 syngas）
- [ ] `.recallloom/context_brief.md`（使命描述更新）

---

## 四、已确认决策

### 决策 1：LHS 采样方案 → **方案 B + 条件顺序采样**

- 区间：CO [15,65]%, H₂ [5,55]%, CO₂ [2,30]%, CH₄ [0,12]%, N₂ ≥ 0.2%
- 采样方式：条件顺序采样（非拒绝采样），接受率 ~85-90%
- 方案 A 区间可作为 holdout 评估集（~200 条序列）
- 详细设计见 `docs/syngas/lhs_sampling_design.md`

### 决策 2：CO 通道串扰建模 → **完整串扰，分两步实施**

- Step 1：先做 CO 自身吸收（无串扰基线），跑通闭环
- Step 2：扩展 `optical_crosstalk.py` 为 3×3 矩阵（CO₂→CO ε=0.005, CO→CO₂ ε=0.002）
- 两步对比本身构成 ablation 实验
- 详细设计见 `docs/syngas/co_crosstalk_design.md`

---

## 五、关键文件清单

> 计划版以"改动 hg 路径"为蓝本；实际实施采用**分支隔离**，hg 路径文件完全保留。
> 下表"实际处理"列反映 2026-06-26 实施后的真实情况。

| 文件 | 计划改动类型 | Phase | 实际处理 |
|------|---------|-------|---------|
| `src/sim/core/schema.py` | 改 | A1 | 不动，新增 `src/sim/core/syngas_schema.py` |
| `src/sim/generation/conditions.py` | 重写 | A2 | 不动，新增 `src/sim/generation/syngas/conditions.py` |
| `src/sim/generation/acoustic_physics.py` | 加 CO 项 + 改签名 | A3, A3.5 | 不动，新增 `src/sim/generation/syngas/acoustic_physics.py` |
| `src/sim/generation/optical_crosstalk.py` | 2×2→3×3 矩阵 | A4.5 | 不动，新增 `src/sim/generation/syngas/optical_crosstalk.py`（3×3，Step 1/2 切换） |
| `src/sim/generation/slow.py` | 加 V_NDIR_CO | A5 | 不动，新增 `src/sim/generation/syngas/slow.py` |
| `src/sim/generation/benchmark.py` | — | — | 不动，新增 `src/sim/generation/syngas/benchmark.py` |
| `src/sim/generation/waveforms.py` | — | — | 共用，新增可选 `sound_speed_fn` / `attenuation_fn` / `extra_gas_kwargs` 注入 |
| `src/sim/packaging/manifest.py` | — | — | 共用，新增 `composition_scheme` / `background_fields` 可选参数 |
| `src/sim/validation/integrity.py` | — | — | 共用，新增 `component_fields` / `background_fields` / `require_sum_100` 可选参数 |
| `configs/data/spectral-defaults.json` | 加 CO | A4 | 暂不动（HITRAN syngas 后端为阶段 3c，empirical 路径不依赖此文件） |
| `src/common/composition.py` | 加 validate_open + deprecated | B1 | 暂不动（syngas 不走 ILR/ALR，trainer 直接拒绝 target_transform） |
| `src/common/metrics.py` | n2→co 重命名 | B2 | 不重命名，`conditional_component_metrics` 新增 `bin_components` 可选参数，syngas 传 `("x_CO", "x_CH4")` |
| `src/dl/training/losses.py` | deprecated + 校验 | C1 | 新增 `validate_loss_composition_scheme()`，syngas 自动拒绝闭包类 loss；`validate_loss_model_output` 接受 `composition_scheme` 放开 syngas 对 gas-head 的限制 |
| `src/dl/training/trainer.py` | 条件跳过 ILR 逆变换 | C2 | `Trainer` 接受 `component_names` / `composition_scheme`；syngas 直接拒绝 target_transform；evaluate 按 scheme 决定分箱、`sum_abs_error`、`compositional_metrics` |
| `src/dl/cli.py` | — | — | 自动从 `manifest.json` 读 `composition_scheme` + `metadata/label_names.npy` 注入 trainer 与 loss 校验 |
| `src/pipeline/generate_benchmark.py` | 加 `--composition-scheme` | D4 | 不动，新增 `src/pipeline/generate_syngas_benchmark.py` 独立入口 |
| `src/pipeline/run_experiment.py` | 报告字段重命名 | D1 | 暂不动（syngas DL 走 `dl.cli`，未接 run_experiment 多 run 编排） |
| `src/pipeline/analyze_n2_improvement.py` | 改名 | D2 | 暂不动（syngas 没有对应 CO 改进工作流，按需后续补） |
| `configs/experiment/sg4/*.json` | 新建 | D3 | 已新增 `sg4_baseline.json` 和 `README.md` |
| `tests/test_syngas_*.py` | 新建 | E1 | 已新增 `test_syngas_sampling.py` / `test_syngas_acoustic_physics.py` / `test_syngas_benchmark_generation.py` / `test_syngas_dl_training.py`（共 91 测试） |
| `data/sg4-smoke/` | 新建 benchmark | D4 | 已可生成（empirical 后端） |

**实际总改动面**：8 个新源码文件 + 7 个共用文件向后兼容改造 + 4 个新测试文件 + 配置/文档若干。

---

## 六、验证流程

> 计划版命令按"hg 路径加 `--composition-scheme syngas`"草拟；实际入口为独立 `pipeline.generate_syngas_benchmark`。下面给出实际可执行命令。

```
1. 小规模 benchmark（empirical 后端）
   python -m pipeline.generate_syngas_benchmark \
       --output-root data --dataset sg4-smoke --sequences 32 --seed 20260626 \
       --timesteps 32 --dt-s 0.5 --optical-absorption-backend empirical_v1 --workers 1
   → data/sg4-smoke/manifest.json: composition_scheme=syngas, schema_version=v4-syngas-1
   → labels/y.npy shape (32, 4)
   → sequences/slow.npy shape (32, 32, 9)

2. HITRAN 缓存（阶段 3c，未完成）
   尚未实现 syngas 专用 HITRAN 缓存预计算入口；调用 syngas slow.py 的 HITRAN backend 会抛 NotImplementedError。

3. 物理 + 数据单元测试
   pytest tests/test_syngas_sampling.py tests/test_syngas_acoustic_physics.py \
          tests/test_syngas_benchmark_generation.py tests/test_syngas_dl_training.py -v
   → 91 passed（截至 2026-06-26）

4. DL 基线训练
   python -m dl.cli --config configs/experiment/sg4/sg4_baseline.json --epochs 3 --batch-size 4
   → loss 下降，metrics.json 输出含 component_metrics (x_H2/x_CH4/x_CO2/x_CO),
     conditional_metrics (co_bins/ch4_bins), sum_abs_error=null

5. 回归测试
   pytest tests/ -q
   → 444 passed（旧 wv4-smoke 流水线完全未受影响）
```

---

## 七、风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| CO/N₂ 声学不可辨识，模型退化为依赖单通道 | 高 | 高 | Fisher 信息分析 + 物理先验约束 + NDIR_CO ablation |
| 采样拒绝率高，训练集边界欠采样 | 中 | 中 | 条件顺序采样替代拒绝采样 |
| CO NDIR 滤光片参数为占位值 | 确定 | 中 | 正式实验前替换为实际 datasheet |
| CO 弛豫常数文献缺失（αλ_max） | 中 | 低 | 物理类比合理 + ablation 扫描 0.015-0.040 |
| 旧 wv4-smoke 回归破坏 | 低 | 高 | Phase E2 参数化测试 |
| CO/CO₂ 光学串扰被低估 | 中 | 中 | 完整 3×3 串扰矩阵 + HITRAN 谱积分验证 |

---

## 八、工作量估计

| Phase | 预估 | 依赖 |
|-------|------|------|
| Phase 0（前置验证） | 已完成 | — |
| Phase A（物理建模） | 2 天 | Phase 0 + 用户决策 |
| Phase B（通用工具层） | 0.5 天 | Phase A |
| Phase C（训练层） | 0.5 天 | Phase A |
| Phase D（Pipeline） | 0.5 天 | Phase A-C |
| Phase E（测试） | 1 天 | Phase A-D |
| Phase F（文档） | 部分已完成 | Phase A-E |
| **总计** | **~4 天**（不含物理常数 ablation） | |

`[审查修正]` 原规划估计 1-2 天偏乐观，含光学串扰扩展和完整测试覆盖后调整为 3-4 天。
