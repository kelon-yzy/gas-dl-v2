# 合成气四组分文档导航

> 本目录统一管理合成气 / 煤气化制气新四组分适配相关文档。
> 新四组分指 `H₂ / CH₄ / CO₂ / CO`，其中 `N₂` 作为背景气参与物理仿真，但不作为预测目标。

> ⚠️ **2026-06-27 时间轴对齐变更**：sg4 系列正式数据集 timesteps 已从 128 改为 512（与 hg `wv4-formal-hitran-standard-6000` 一致）。当前的 `data/sg4-formal` (128 步) 与基于它产出的 Stage Ⅰ-3 / Stage Ⅱ 全部 41 个 run 结果**已废弃，待 512 步重跑**。所有正式 benchmark 生成命令、TCN 配置 `target_timesteps`、PatchTST attention cost 等均已同步更新；历史结果文档保留原数字，顶部标注废弃状态。

## 实施状态（截至 2026-06-26）

阶段 1–4 已完成；阶段 Ⅰ 正式实验已启动：

```text
pipeline.generate_syngas_benchmark  →  data/sg4-formal/ (6000 seq)  →  dl.cli + sg4_{baseline,tcn,lstm,patchtst,ridge}.json
```

| 阶段 | 范围 | 状态 |
|---|---|---|
| 1 | schema + 采样 | ✅ 18 测试通过 |
| 2 | 物理（声学 / 热导 / 光学 / 3×3 串扰） | ✅ 49 测试通过 |
| 3a/3b | slow + benchmark + CLI（empirical 后端） | ✅ 15 测试通过 |
| 3c | HITRAN 后端 + CO/CO₂/H₂O 三气体缓存 | ⏳ 留位（HITRAN backend raise NotImplementedError） |
| 4 | DL 训练适配 | ✅ 9 测试通过 |
| **Ⅰ-1** | sg4-formal benchmark 生成 | ⚠️ 6000 序列 / 128 时步 / 9 慢通道，validation pass — **已废弃，待 512 时步重跑** |
| **Ⅰ-2** | 实验配置矩阵（CNN1D / TCN / LSTM / PatchTST / Ridge） | ✅ 5 配置就绪 |
| **Ⅰ-3** | 基线训练（5 模型 × 3 seeds = 15 runs，PatchTST 重跑 +3） | ⚠️ 14/15 收敛（TCN≈Ridge pool R²≈0.96，PatchTST/CNN1D/LSTM ≈ 0.93），见 [stage_i3_baseline_results.md](stage_i3_baseline_results.md) — **基于 128 步，已废弃，待 512 步重跑** |
| Ⅰ-3 配套修复 | 编排脚本 LSTM 退出码兼容 + trainer AMP bf16 兼容（全量 444 测试通过，hg 零回归） | ✅ |
| **Ⅱ** | ablation 实验（CO 通道 / 串扰 / Loss，27 runs） | ⚠️ 全部成功，V_NDIR_CO 支配 CO 检测、串扰可学、CH₄ 受 loss 加权决定，见 [stage_ii_ablation_results.md](stage_ii_ablation_results.md) — **基于 128 步，已废弃，待 512 步重跑** |
| Ⅱ 配套代码 | channel 子集选择（DL+Ridge）+ crosstalk CLI 四层透传 + 测试（全量 462 测试通过，hg 零回归） | ✅ |

全量回归 462 passed（baseline 353 + syngas 增量 109，含 Stage Ⅱ ablation 18 个测试）。

详情见 [adaptation_plan.md §实施进度](adaptation_plan.md#实施进度截至-2026-06-27)、[experiment_roadmap.md §阶段 Ⅰ 执行记录](experiment_roadmap.md#阶段-ⅰ-执行记录2026-06-26) 和 [experiment_roadmap.md §阶段 Ⅱ 执行记录](experiment_roadmap.md#阶段-ⅱ-执行记录2026-06-27)。

## 阅读顺序

| 顺序 | 文件 | 用途 | 适合什么时候读 |
|---:|---|---|---|
| 1 | [adaptation_plan.md](adaptation_plan.md) | 主实施方案，包含架构决策、阶段计划、验证流程和风险矩阵 | 开始编码、审查整体方案、确认改动范围 |
| 2 | [experiment_roadmap.md](experiment_roadmap.md) | 后续实验路线图（正式 benchmark → 基线 → ablation → 基础设施） | 规划下一步实验、确认依赖和优先级 |
| 3 | [stage_i3_baseline_results.md](stage_i3_baseline_results.md) | 阶段 Ⅰ-3 基线训练结果（5 模型 × 3 seeds）与分析 | 查阅 baseline R²、解读 TCN/Ridge 平手、CH₄/CO 表现 |
| 4 | [stage_ii_ablation_plan.md](stage_ii_ablation_plan.md) | 阶段 Ⅱ ablation 实施计划（CO 通道 / 串扰 / Loss）+ 实施进度 | 跟进 Ⅱ-1/Ⅱ-2/Ⅱ-3 实现状态、channel 子集与 crosstalk 透传改动 |
| 5 | [stage_ii_ablation_results.md](stage_ii_ablation_results.md) | 阶段 Ⅱ ablation 实验结果（27 runs）与分析 | 查阅 V_NDIR_CO 支配性、CO₂↔CO 串扰可学性、loss 加权对 CH₄ 的影响 |
| 6 | [lhs_sampling_design.md](lhs_sampling_design.md) | 方案 B + 条件顺序采样的实现规范和验收标准 | 修改 `conditions.py` 或设计 `sg4` benchmark |
| 7 | [physics_references.md](physics_references.md) | 可直接写入代码的 CO 声学、热导、光学常数速查 | 修改 `acoustic_physics.py`、`spectral-defaults.json`、`slow.py` |
| 8 | [co_crosstalk_design.md](co_crosstalk_design.md) | CO 通道串扰矩阵、分步实施和消融实验设计 | 修改 `optical_crosstalk.py` 或做串扰 ablation |
| 9 | [references/README.md](references/README.md) | 文献报告索引和证据来源入口 | 需要追溯参数来源、引用文献或补充调研 |

## 当前已确认的设计

1. **预测目标**：`("x_H2", "x_CH4", "x_CO2", "x_CO")`。
2. **背景气**：`x_N2 = 100 - x_H2 - x_CH4 - x_CO2 - x_CO`，不写入 labels，但参与声学、衰减、热导和光学仿真。
3. **采样方案**：采用方案 B（煤气化技术全谱）+ 条件顺序采样；方案 A 可作为气流床 holdout。
4. **新增慢通道**：`V_NDIR_CO`，slow 通道从 8 个变为 9 个。
5. **光学串扰**：分两步实施；先跑通 CO 自身吸收基线，再扩展 3×3 串扰矩阵。
6. **Loss 路线**：sum<100 场景使用开放组合 loss，例如 `weighted_component_mse` 或 `mse`；闭包类 loss 仅保留旧场景兼容。
7. **Benchmark**：新场景使用 `sg4-*` 系列，与 `wv4-*` 掺氢天然气数据并存。

## 文件职责

### 实施与架构

- [adaptation_plan.md](adaptation_plan.md)：主计划。后续实现、审查、验收应以这个文件为准。
- [experiment_roadmap.md](experiment_roadmap.md)：实验路线图。正式 benchmark 生成、基线训练、ablation 实验、基础设施补全的优先级和依赖关系。
- [stage_i3_baseline_results.md](stage_i3_baseline_results.md)：阶段 Ⅰ-3 基线训练结果。5 模型 × 3 seeds 的 per-component R²/MAE、关键发现、待跟进项。
- [stage_ii_ablation_plan.md](stage_ii_ablation_plan.md)：阶段 Ⅱ ablation 实施计划与进度。CO 通道 / 串扰 / Loss 三组消融的代码改动、配置、编排脚本和验证状态。
- [stage_ii_ablation_results.md](stage_ii_ablation_results.md)：阶段 Ⅱ ablation 实验结果。27 runs 的 per-component R²/MAE、三组消融的物理与方法学结论、待跟进项。
- [lhs_sampling_design.md](lhs_sampling_design.md)：采样设计。给出方案 B 区间、联合约束、条件顺序采样伪代码、测试建议。
- [co_crosstalk_design.md](co_crosstalk_design.md)：CO NDIR 串扰设计。给出 3×3 矩阵、推荐系数和无串扰 / 有串扰对比实验。

### 参数与文献

- [physics_references.md](physics_references.md)：从文献报告中提取的编码速查表。
- [references/co_acoustic_constants.md](references/co_acoustic_constants.md)：CO 声速、弛豫频率、H₂O 耦合文献证据。
- [references/co_optical_hitran.md](references/co_optical_hitran.md)：CO HITRAN 标识、滤光片、网格和光谱串扰证据。
- [references/syngas_composition_ranges.md](references/syngas_composition_ranges.md)：合成气组分范围、方案 A/B/C 来源。
- [references/syngas_sensing_survey.md](references/syngas_sensing_survey.md)：商用系统、国标、四模态融合可行性。

## 编码前检查点

- `manifest.json` 应记录 `composition_scheme` / `labels` / `background_fields` / `slow_channels`，下游加载器应以 manifest 为准，不要硬编码全局常量。
- `x_N2` 必须保留在 condition / physics 输入中，但不能作为 syngas labels 目标列。
- 旧 `wv4` 数据和新 `sg4` 数据完全并存：实施采用**分支隔离**，syngas 走 `src/sim/core/syngas_schema.py` 和 `src/sim/generation/syngas/` 子包；hg 路径未改动。
- CO 与 N₂ 在声学和热导上近似简并，CO 精度需要重点检查 `V_NDIR_CO` 贡献；建议做 ablation 移除 `V_NDIR_CO` 看 CO R² 是否暴跌。
- CO NDIR 滤光片参数当前是行业参考占位（InfraTec I 4.66 μm / 180 nm），正式实验前需要替换为目标传感器 datasheet。
- syngas 不允许 `target_transform`（ILR/ALR 依赖 sum=100% 闭包），Trainer 直接拒绝。
- 闭包类 loss（`compositional_mse` / `ilr_mse` / `free_component_mse` / `weighted_free_component_mse`）在 syngas 场景被自动拒绝。

## 实际入口与最小闭环

```powershell
# 1a. 生成 sg4-smoke benchmark（empirical 后端，约 16 序列、3 秒，用于链路验证）
python -m pipeline.generate_syngas_benchmark `
    --output-root data --dataset sg4-smoke --sequences 32 --seed 20260626 `
    --timesteps 32 --dt-s 0.5 --optical-absorption-backend empirical_v1 --workers 1

# 1b. 生成 sg4-formal benchmark（6000 序列 / 512 时步 / 24 workers，约 6–8 分钟，与 hg `wv4-formal-hitran-standard-6000` 时间轴对齐）
python -m pipeline.generate_syngas_benchmark `
    --output-root data --dataset sg4-formal --sequences 6000 --seed 20260626 `
    --timesteps 512 --dt-s 0.5 --optical-absorption-backend empirical_v1 `
    --storage memmap --workers 24

# 2. DL 基线训练（cnn1d + weighted_component_mse，单 seed）
python -m dl.cli --config configs/experiment/sg4/sg4_baseline.json

# 3. 全量基线矩阵（5 模型 × 3 seeds = 15 runs，自动汇总到 outputs/sg4_baseline/summary.json）
python scripts/run_sg4_baseline.py

# 4. 单元测试
python -m pytest tests/test_syngas_*.py -v
```

## 关键源码入口

| 入口 | 文件 |
|---|---|
| schema | `src/sim/core/syngas_schema.py` |
| 采样 | `src/sim/generation/syngas/conditions.py` |
| 声学 / 热导 / 吸收 | `src/sim/generation/syngas/acoustic_physics.py` |
| 3×3 光学串扰（Step 1/2 切换） | `src/sim/generation/syngas/optical_crosstalk.py` |
| 慢通道（9 通道） | `src/sim/generation/syngas/slow.py` |
| benchmark 编排 | `src/sim/generation/syngas/benchmark.py` |
| CLI | `src/pipeline/generate_syngas_benchmark.py` |
| DL 配置（5 个） | `configs/experiment/sg4/sg4_{baseline,tcn,lstm,patchtst,ridge}.json` |
| 训练编排脚本 | `scripts/run_sg4_baseline.py` |
| 共用 metrics 适配 | `src/common/metrics.py` (`bin_components` 参数化) ; `src/ml/training.py` (`_default_bin_components`) |

## 旧路径迁移说明

原先散落在 `docs/` 顶层和 `docs/syngas_references/` 下的合成气文档已统一迁移到本目录：

| 旧路径 | 新路径 |
|---|---|
| `docs/syngas_adaptation_plan.md` | `docs/syngas/adaptation_plan.md` |
| `docs/syngas_physics_references.md` | `docs/syngas/physics_references.md` |
| `docs/syngas_lhs_sampling_design.md` | `docs/syngas/lhs_sampling_design.md` |
| `docs/syngas_co_crosstalk_design.md` | `docs/syngas/co_crosstalk_design.md` |
| `docs/syngas_references/*.md` | `docs/syngas/references/*.md` |
