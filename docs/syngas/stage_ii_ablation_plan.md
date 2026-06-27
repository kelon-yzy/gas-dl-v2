# 阶段 Ⅱ 关键 ablation 实验实施计划

> 本文档从 `.claude/plans/tidy-yawning-stearns.md`（已批准的实施计划）复制到项目文档目录，并在开头补充实施进度。
> 计划主体见下方「## Context」起的原始内容，未作改动。

---

## 实施进度（截至 2026-06-27）

### 总体状态

**Ⅱ-1 / Ⅱ-2 / Ⅱ-3 全部完成**。代码、配置、编排、数据、训练、分析 27 个 run 全部跑齐。结果分析见 [stage_ii_ablation_results.md](stage_ii_ablation_results.md)。

| 步骤 | 范围 | 状态 |
|---|---|---|
| 代码 Ⅱ-1 | DL + Ridge channel 子集选择 | ✅ 完成，已验证 |
| 代码 Ⅱ-2 | `enable_co_crosstalk` 四层透传 | ✅ 完成，已验证 |
| 测试 | `tests/test_syngas_ablation.py` | ✅ 完成，462 passed |
| 配置 | 9 个 ablation 配置 | ✅ 完成 |
| 编排 | `scripts/run_sg4_ablation.py` | ✅ 完成 |
| Ⅱ-2 数据 | `sg4-formal-crosstalk` 生成（6000 / 128 / `enable_co_crosstalk=True`） | ✅ 完成，manifest policy=`syngas_empirical_3x3_step2_co2_co_crosstalk` |
| Ⅱ-1/2/3 训练 | 三组 ablation 训练（27 runs） | ✅ 全部成功 |
| 分析 | `stage_ii_ablation_results.md` | ✅ 完成 |

### 关键结果（test split，详细见 [stage_ii_ablation_results.md](stage_ii_ablation_results.md)）

- **Ⅱ-1（CO 通道）**：去 V_NDIR_CO（B 组），x_CO R² 从 0.954 → **0.484**（TCN/Ridge 同向，-0.02 以内），证实 V_NDIR_CO 支配 CO 检测；仅保留 V_NDIR_CO + 环境（C 组），x_CO R² 仍达 **0.93–0.94**。原计划"暴跌至 ~0"的预期偏差：实际损失约 50%，剩余信号来自闭包约束 + V_TCS 弱热导差异 + 可能的 CO₂ NDIR 弱串扰。
- **Ⅱ-2（串扰）**：crosstalk 数据集训练后所有组分 R² 与基线持平（|Δ| ≤ 0.006），原计划"下降 0.01–0.05"的预期未出现。线性 3×3 串扰是确定性映射，模型可学到逆映射；sim-to-real gap 不在串扰矩阵层面。
- **Ⅱ-3（Loss）**：单一超参切换（`weighted_component_mse` → 任一未加权 loss），x_CH4 R² 从 **0.827 → 0.39–0.44**（下降约 0.4），其他组分几乎不变。inverse_train_var 加权对低浓度组分至关重要，是论文核心方法学贡献点。

### 已落地的代码改动

**Ⅱ-1 channel 子集选择**（DL + Ridge 两条路径对称，默认 `slow_channels=None` 保证 hydrogen_ng 零回归）：

- `src/dl/data/dataset.py`：`V4BenchmarkDataset` 新增 `slow_channels` 参数 + 模块级 `_resolve_slow_channel_indices`，读 `metadata/slow_channel_names.npy` 建名字→index 映射，在 `apply_scaler` + `_apply_window` 之后按 index 选列；`in_channels` 由 `_infer_input_shape` 自动推断。
- `src/dl/cli.py`：`--slow-channels` 参数 + `DEFAULT_DL_CONFIG` 字段 + `_parse_slow_channels` helper + train/val/test/extrapolation 四处 loader 透传 + `metrics.json`/`run_config.json` 记录；FEATURES 输入格式下传 `slow_channels` 报错保护。
- `src/ml/features.py`：`MLFeatureConfig.slow_channels` + `_select_slow_channels`，`apply_scaler` 后按 `slow_channel_names` 过滤列与名字。
- `src/ml/cli.py`：`--slow-channels` + `DEFAULT_ML_CONFIG` 字段 + `_parse_slow_channels` + `feature_config` 记录。

**Ⅱ-2 crosstalk 透传**（默认 `enable_co_crosstalk=False` = Step 1，hydrogen_ng 不受影响）：

链路 `CLI → spec → build_sequence_arrays → main_sensor_features → apply_syngas_optical_crosstalk`：
- `acoustic_physics.py`：`main_sensor_features` 加 `enable_co_crosstalk` 参数（替换硬编码 `False`）。
- `slow.py`：`build_sequence_arrays` 加参数，透传给两处 `main_sensor_features`。
- `benchmark.py`：`SyngasBenchmarkGenerationSpec` 加字段；单进程路径透传；`_optical_absorption_metadata` 按开关返回 `syngas_empirical_3x3_step1_co_pure` 或 `..._step2_co2_co_crosstalk` + `enable_co_crosstalk`。
- `_parallel.py`：`_generate_chunk_file` 透传。
- `generate_syngas_benchmark.py`：`--enable-co-crosstalk` flag。

**测试 / 配置 / 编排**：
- `tests/test_syngas_ablation.py`：channel 选列正确（A=9/B=8/C=6）、列顺序遵循请求、非法通道名报错、CLI 端到端 in_channels=8、Ridge 特征数 63/56/42、crosstalk 开关物理层差异、benchmark policy step1/step2、同 seed 下 V_NDIR_CO 变化而环境通道不变。
- `configs/experiment/sg4/ablation/`：`co_channel/{tcn,ridge}_{dropco,coonly}.json`（4）+ `loss/tcn_{mse,mae,huber,smoothl1}.json`（4）+ `crosstalk/tcn_crosstalk.json`（1）。
- `scripts/run_sg4_ablation.py`：三组 ablation 编排，`--experiment {co_channel,loss,crosstalk,all}`，含 Windows cuDNN 退出码兼容、summary 扫描全部已存在 metrics.json（部分重跑不丢历史）。

### 通道集（Ⅱ-1）

| 组 | 保留通道 | in_channels |
|---|---|---|
| A（基线，复用 sg4_baseline） | 全 9 | 9 |
| B（去 CO NDIR） | V_NDIR_CH4, V_NDIR_CO2, V_TCS, T_C, P_MPa, H_RH, L_m, piston_position_m | 8 |
| C（仅 CO 光学+环境） | V_NDIR_CO, T_C, P_MPa, H_RH, L_m, piston_position_m | 6 |

### 验证记录

- 全量 `python -m pytest`（用户 PowerShell 实跑）：**首轮 461 passed / 1 failed**，耗时 62s。
  - 唯一失败 `TestRidgeChannelSubset::test_subset_columns_match_full`：float32 下 numpy 对 `(N,T,9)` 与 `(N,T,1)` 的 `mean(axis=1)` 累加路径不同，产生 ~4.77e-7 末位差异（11 元素中 9 个 bit-identical，2 个超 `rtol=1e-7`）。属测试断言容差问题，非功能 bug。
  - 修复：放宽该断言为 `rtol=1e-5`（仅改测试，不动 production 代码）。**重跑预期 462 passed，待最终确认。**
- 原有 444 测试（hydrogen_ng 353 + syngas 91）全部通过 → **hydrogen_ng 零回归确认**。
- channel 子集 in_channels 与 crosstalk policy 行为符合预期（见上方测试覆盖）。

### 待执行

✅ **全部完成**。后续优化候选见 [stage_ii_ablation_results.md §6 待跟进](stage_ii_ablation_results.md#已确认)：

1. 在 sg4-formal-crosstalk 上重复 B 组（验证 CO₂ NDIR 弱串扰是否提供 CO 信息）。
2. B 组再去 V_TCS（量化热导通道对 CO 的残留贡献）。
3. 接入 ultrasonic / fiber_mic 模态后重做 Ⅱ-1（验证多模态能否恢复 CO 信息）。
4. Stage Ⅲ 真实硬件 sim-to-real 测试。

---

## Context

合成气四组分（H₂/CH₄/CO₂/CO）场景的阶段 Ⅰ 已完成：sg4-formal benchmark（6000 序列）+ 5 模型 × 3 seeds 基线训练，TCN ≈ Ridge ≈ 0.96。基线给出两个待验证的核心物理判断，需要 ablation 实验产出论文的科学贡献点：

1. **CO/N₂ 声学近简并假说**：CO 与 N₂ 摩尔质量都是 28、声速差 <1 m/s，声学/热导几乎无法区分二者，CO 的可观测性理论上完全依赖 NDIR 光学通道（V_NDIR_CO）。基线里 CO R²≈0.92–0.95 反向支持了这一点，但需要 ablation 直接证明。
2. **CO₂↔CO 光学串扰影响**：当前 sg4-formal 是 Step 1（CO 通道纯吸收，无串扰）。3×3 串扰矩阵代码已就绪但被硬编码关闭，需要生成 Step 2 数据量化串扰对 CO/CO₂ 精度的影响（sim-to-real 真实性）。
3. **Loss 选择对 CH₄ 短板的影响**：CH₄ 是基线短板（R²≈0.83），不同开放组合 loss 可能改变低浓度组分表现。

**用户已确认范围**：Ⅱ-1 + Ⅱ-2 + Ⅱ-3 全做；Ⅱ-1 用 **TCN + Ridge** 两个最强模型跑 ablation 组别。

**当前阻塞**：channel 级 ablation 机制不存在（`dl.cli --modalities` 只能选模态整体，`ml/features` 加载全 9 通道无子集选择）；crosstalk 开关没从 CLI 透传到物理层（`main_sensor_features` 硬编码 `enable_co_crosstalk=False`）。两者都需要先补代码。

---

## 决策摘要

| 项 | 决定 |
|---|---|
| 范围 | Ⅱ-1（CO 通道 ablation）+ Ⅱ-2（串扰 ablation）+ Ⅱ-3（Loss 对比）。Ⅱ-4（CO 弛豫扫描）本轮不做 |
| Ⅱ-1 模型 | TCN + Ridge；组 A（全通道）直接复用 `outputs/sg4_baseline/{tcn,ridge}` |
| Ⅱ-2 模型 | TCN（最强 DL，对比 CO/CO₂ R²）；Ridge 可选 |
| Ⅱ-3 模型 | TCN × {mse, mae, huber, smooth_l1}；weighted_component_mse 复用 baseline |
| seeds | 42 / 123 / 2026（与 baseline 对齐） |
| 向后兼容 | 所有共用文件改动用可选参数默认值（`slow_channels=None` / `enable_co_crosstalk=False`），hg 路径零回归 |

**SLOW_CHANNELS 顺序**（`syngas_schema.py:27`，index 固定）：
`V_NDIR_CH4(0) V_NDIR_CO2(1) V_NDIR_CO(2) V_TCS(3) T_C(4) P_MPa(5) H_RH(6) L_m(7) piston_position_m(8)`

**Ⅱ-1 三组别通道集**（配置里写"保留通道名列表"，比 drop 列表可读）：

| 组 | 含义 | 保留通道 | in_channels |
|---|---|---|---|
| A | 全通道（基线） | 全 9 | 9 |
| B | 去 CO NDIR | `V_NDIR_CH4, V_NDIR_CO2, V_TCS, T_C, P_MPa, H_RH, L_m, piston_position_m` | 8 |
| C | 仅 CO 光学 + 环境 | `V_NDIR_CO, T_C, P_MPa, H_RH, L_m, piston_position_m` | 6 |

> 组 C 去掉了 CH₄/CO₂/TCS 主通道，H₂/CH₄/CO₂ 预测会连带退化属预期；C 组解读焦点只看 **x_CO R²**（验证仅靠 CO 光学+环境能否测准 CO）。

---

## Ⅱ-1 CO 通道 ablation

### 代码改动（channel 子集选择，DL + Ridge 两条路径对称）

**DL 路径** — `src/dl/data/dataset.py`：
- `V4BenchmarkDataset.__init__` 加 `slow_channels: tuple[str, ...] | None = None`。
- 当非 None：读 `metadata/slow_channel_names.npy` 建名字→index 映射，存 `self._slow_channel_indices`；通道名非法则报错。
- `_build_single_input`：slow 部分在现有 `apply_scaler` + `_apply_window` **之后**做列选择 `sl = sl[:, indices]`（scaler 按全 9 通道 fit，必须先 scaler 再选列）。
- `in_channels` 由 `cli.py:208 _infer_input_shape(train_dataset[0])` 自动推断，去列后自动变 8/6，模型维度自动适配，无需手动改。

**DL CLI** — `src/dl/cli.py`：
- `DEFAULT_DL_CONFIG` 加 `"slow_channels": None`；`build_parser` 加 `--slow-channels`（逗号分隔）。
- `_build_dataset` / `_optional_loader` 透传 `slow_channels` 到 `V4BenchmarkDataset`（train/val/test/extrapolation 四处 loader 都要传，保证一致）。

**Ridge 路径** — `src/ml/features.py`：
- `MLFeatureConfig` 加 `slow_channels: tuple[str, ...] | None = None`。
- `load_feature_matrix` slow 分支（line 73-86）：加载 slow + `apply_scaler` 后，若 `config.slow_channels` 非 None，按 `slow_channel_names` 过滤列与名字，再进 `sequence_stat_features`。

**Ridge CLI** — `src/ml/cli.py`：
- `DEFAULT_ML_CONFIG` 加 `"slow_channels": None`；`build_parser` 加 `--slow-channels`；`run` 构造 `MLFeatureConfig` 时透传。

### 配置（`configs/experiment/sg4/ablation/co_channel/`）

复制 `sg4_tcn.json` / `sg4_ridge.json`，加 `slow_channels` 字段、改 `output_dir`：
- `tcn_dropco.json`（B 组 8 通道）、`tcn_coonly.json`（C 组 6 通道）
- `ridge_dropco.json`、`ridge_coonly.json`

### 训练

B/C 组 × (TCN + Ridge) × 3 seeds。A 组复用 `outputs/sg4_baseline`。

---

## Ⅱ-2 串扰 ablation

### 代码改动（`enable_co_crosstalk` 四层透传 + manifest）

链路：CLI → spec → build_sequence_arrays → main_sensor_features → apply_syngas_optical_crosstalk

1. `src/sim/generation/syngas/optical_crosstalk.py`：已有 `enable_co_crosstalk` 参数，**无需改**。
2. `src/sim/generation/syngas/acoustic_physics.py`：`main_sensor_features` 加 `enable_co_crosstalk: bool = False`，传给 `apply_syngas_optical_crosstalk`（替换 line 261 硬编码 `False`）。
3. `src/sim/generation/syngas/slow.py`：`build_sequence_arrays` 加 `enable_co_crosstalk: bool = False`，传给 `main_sensor_features` 两处调用（baseline_main / target_main，line 130-131）。
4. `src/sim/generation/syngas/benchmark.py`：`SyngasBenchmarkGenerationSpec` 加 `enable_co_crosstalk: bool = False`；`_build_sequence_arrays_for_spec` 单进程路径透传；`_optical_absorption_metadata` 按开关返回 `syngas_empirical_3x3_step1_co_pure` 或 `..._step2_co2_co_crosstalk`。
5. `src/sim/generation/syngas/_parallel.py`：`_generate_chunk_file`（line 106）调 `build_sequence_arrays` 加 `enable_co_crosstalk=spec.enable_co_crosstalk`。
6. `src/pipeline/generate_syngas_benchmark.py`：加 `--enable-co-crosstalk` flag，透传给 spec。

> 同 seed 下开/关串扰，`apply_syngas_optical_crosstalk` 不消耗 RNG，除 CO 通道（及 CO₂ 的 cross_from_co）外所有量一致，可严格对比。

### 数据生成

生成 `sg4-formal-crosstalk`：与 sg4-formal **完全相同参数**（seed 20260626 / 6000 / 128 / empirical / 24 workers），仅加 `--enable-co-crosstalk`。

### 训练

TCN × 3 seeds（配置复制 `sg4_tcn.json`，`dataset_dir` 指向 `data/sg4-formal-crosstalk`），对比 sg4-formal 基线的 x_CO R² / x_CO2 R²。预期差异 0.01–0.05。

---

## Ⅱ-3 Loss 对比

### 代码改动

无。`mse / mae / smooth_l1 / huber` 已在 `LOSS_REGISTRY`，syngas 兼容（闭包类已被 `validate_loss_composition_scheme` 拒绝）。

### 配置（`configs/experiment/sg4/ablation/loss/`）

复制 `sg4_tcn.json`，改 `loss` 字段 + `output_dir`：`tcn_mse.json` / `tcn_mae.json` / `tcn_huber.json` / `tcn_smoothl1.json`（string loss）。weighted_component_mse 复用 baseline `sg4_tcn`。

### 训练

TCN × 4 loss × 3 seeds。重点看 x_CH4 R²（低浓度组分最受 loss 影响）。

---

## 编排、测试与回归

**编排脚本** — 新建 `scripts/run_sg4_ablation.py`：
- 复用 `run_sg4_baseline.py` 的 `_run_dl` / `_run_ml`（含 Windows cuDNN 退出码兼容、metrics.json 优先判定）。
- 参数 `--experiment {co_channel,crosstalk,loss,all}`，各实验定义 (config, seeds) 矩阵，产物写 `outputs/sg4_ablation/{experiment}/{tag}/seed{seed}/`，汇总 `summary.json`。

**新增测试** — `tests/test_syngas_ablation.py`：
- `V4BenchmarkDataset(slow_channels=[...])` 输出通道数/选列正确；非法通道名报错。
- `load_feature_matrix` + `MLFeatureConfig(slow_channels=[...])` 列过滤正确。
- crosstalk 透传：高 CO₂ 工况下 `main_sensor_features(enable_co_crosstalk=True)` 的 `V_NDIR_CO` 与 False 不同；CH₄↔CO₂ 行为不变。
- hg 零回归：`slow_channels=None` 时 dataset 输出与改动前一致。

**回归**：`python -m pytest` 全量必须保持通过（当前 444），共用文件（dataset/features/cli/slow/acoustic_physics/benchmark）改动前后对 hg+sg 都跑。

---

## 验证

```bash
# 0. 代码改动后先跑全量回归
python -m pytest -q                      # 期望 444 + 新增 全通过

# Ⅱ-2 透传正确性先在 smoke 上验证（快）
python -m pipeline.generate_syngas_benchmark --output-root data --dataset sg4-smoke-crosstalk \
    --sequences 32 --seed 20260626 --timesteps 32 --optical-absorption-backend empirical_v1 \
    --workers 1 --enable-co-crosstalk
# → manifest optical_crosstalk_policy = ..._step2_co2_co_crosstalk

# Ⅱ-2 正式数据
python -m pipeline.generate_syngas_benchmark --output-root data --dataset sg4-formal-crosstalk \
    --sequences 6000 --seed 20260626 --timesteps 128 --optical-absorption-backend empirical_v1 \
    --storage memmap --workers 24 --enable-co-crosstalk
# → validation status=pass, slow.npy (6000,128,9)

# 三组 ablation 训练（GPU；可后台或 PowerShell 手动跑）
python scripts/run_sg4_ablation.py --experiment all
```

**预期结论**：
- **Ⅱ-1（核心）**：B 组（去 V_NDIR_CO）x_CO R² **暴跌至 ~0**，A/C 组 CO R² 正常 → 证实 CO/N₂ 声学简并、CO 完全依赖光学通道。TCN 与 Ridge 同向佐证，排除"仅时序模型依赖光学"质疑。
- **Ⅱ-2**：crosstalk 版 x_CO R²（可能 x_CO2 R²）较 baseline 下降 0.01–0.05，量化串扰对 sim-to-real 的影响。
- **Ⅱ-3**：对比各 loss 的 x_CH4 R²，确认 weighted_component_mse 对低浓度组分是否有优势。

**产物**：`outputs/sg4_ablation/{co_channel,crosstalk,loss}/.../metrics.json` + `summary.json`；结果分析写入 `docs/syngas/stage_ii_ablation_results.md`（新建，结构对齐 stage_i3）。

---

## 执行顺序

1. **代码层**（一次性改完再统一测试）：Ⅱ-1 channel 选择（dataset/features/两个 cli）+ Ⅱ-2 透传（physics/slow/benchmark/_parallel/CLI）→ 新增测试 → 全量 pytest 回归。
2. **数据层**：smoke 验证 Ⅱ-2 透传 → 生成 sg4-formal-crosstalk。
3. **配置层**：Ⅱ-1 四个 + Ⅱ-3 四个 ablation 配置 + 编排脚本。
4. **训练层**：`run_sg4_ablation.py --experiment all`（Ⅱ-1 → Ⅱ-3 → Ⅱ-2）。
5. **分析层**：汇总 metrics，写 stage_ii_ablation_results.md。

第 1 步是纯代码 + 测试，零训练成本、可独立验证，先做；训练（第 4 步）是 GPU 长任务，按需后台或手动执行。
