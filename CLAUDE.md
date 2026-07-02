# gas-dl-v2 — 多模态气体组分检测 v4

## 项目使命

基于 **NDIR 光学 + 超声波 + 光纤麦克风 + TCS 热导** 四模态传感器仿真信号，使用 DL/ML 模型预测混合气体各组分浓度。

已落地两个检测场景，可并存：

- **掺氢天然气（hydrogen_ng）**：H₂/CH₄/CO₂/N₂，sum=100% 闭包。benchmark `wv4-*`。
- **合成气 / 煤气化制气（syngas）**：H₂/CH₄/CO₂/CO，N₂ 为背景气，sum<100%。Stage Ⅰ 基线 + Stage Ⅱ ablation 完成，benchmark `sg4-smoke` / `sg4-formal`（empirical 后端，6000 序列）可用，HITRAN 后端待补。独立模块路径。

## 代码结构

```
src/
├── sim/          # 物理仿真：声学、光学、慢通道、波形生成、打包
│   ├── core/     #   schema.py (hg) + syngas_schema.py (sg)
│   ├── generation/  # conditions / acoustic_physics / slow / optical_* (hg)
│   │   └── syngas/  #   合成气专用子包：conditions / acoustic_physics / optical_crosstalk / slow / benchmark
│   ├── packaging/   # 数据打包、manifest、scalers（schema 无关，两场景共用）
│   └── validation/  # 数据完整性校验（可注入 component_fields / background_fields）
├── dl/           # 深度学习
│   ├── data/     #   dataset.py（manifest 驱动，自动兼容两场景）
│   ├── models/   #   CNN1D / TCN / CNN1DTCNFusion / PhaseWindowTCN / LSTM / Transformer / PatchTST
│   └── training/ #   losses.py / trainer.py / metrics.py（按 composition_scheme 切换分箱/loss 校验）
├── ml/           # 传统 ML (Ridge / Mean baseline)
├── pipeline/     # CLI 编排、实验运行、benchmark 生成
│   ├── generate_benchmark.py         # hg benchmark CLI
│   └── generate_syngas_benchmark.py  # sg benchmark CLI
└── common/       # 共享工具：composition.py / metrics.py / scalers.py
```

关键入口：
- `python -m pipeline.generate_benchmark` — hg benchmark 生成
- `python -m pipeline.generate_syngas_benchmark` — sg benchmark 生成（empirical 后端可用，HITRAN 后端未实现）
- `python -m pipeline.precompute_hitran_benchmark_cache` — hg HITRAN 谱线缓存预计算
- `python -m dl.cli --config <json>` — DL 训练（manifest 自动决定走 hg 还是 sg 路径）
- `python -m pipeline.run_experiment --config <json>` — hg 多 run 实验编排（sg 暂未接入）
- `python -m pytest` — 全量测试（以实际通过数为准；当前主线 462 passed = hg 353 + sg 109，含 Stage Ⅱ ablation 18 个）

## 核心概念

### 组分字段约定

- **hydrogen_ng**：`src/sim/core/schema.py` 中 `COMPONENT_FIELDS = ("x_H2", "x_CH4", "x_CO2", "x_N2")`，4 列 sum=100%。
- **syngas**：`src/sim/core/syngas_schema.py` 中 `COMPONENT_FIELDS = ("x_H2", "x_CH4", "x_CO2", "x_CO")` + `BACKGROUND_FIELDS = ("x_N2",)`，4 列 sum<100%；`x_N2 = 100 - sum(targets)`，写入 condition grid 但不入 labels。

下游加载器优先读 `manifest.composition_scheme` 与 `metadata/label_names.npy`，不要直接 import 全局常量推断 schema。

### 慢通道

- **hydrogen_ng**：8 个 — V_NDIR_CH4, V_NDIR_CO2, V_TCS, T_C, P_MPa, H_RH, L_m, piston_position_m。
- **syngas**：9 个，在 hg 基础上新增 `V_NDIR_CO`。

### Loss 体系

- 闭包类（sum=100%，仅 hg 可用）：`compositional_mse`, `ilr_mse`, `free_component_mse`, `weighted_free_component_mse`
- 开放类（两场景兼容）：`weighted_component_mse`, `mse`, `mae`, `smooth_l1`, `huber`

syngas 场景下闭包类 loss 由 `validate_loss_composition_scheme()` 自动拒绝；syngas 也不允许 `target_transform`（ILR/ALR 依赖 sum=100%）。

### Benchmark 命名

> **2026-07-02 仿真链路对齐**：超声载波 40kHz→200kHz（PSC200K）、采样率 200k→1MS/s、ADC 16-bit→20-bit（NI-6453）、L_m 范围 0.2~1.8m→0.2~0.3m（200kHz 下长声程信号被 CH₄/CO₂ 弛豫吸收淹没，见 `docs/Phase0_物理可行性核对记录.md`）。旧 benchmark 已归档至 `data/_archived_pre_200khz/`，**不可用于新链路训练**。manifest 新增 `sim_revision` 字段标记链路版本（`v5-200khz-20bit-L03`）。`wv4-smoke` / `sg4-smoke` 已按新链路重生成；formal 集（`wv4-formal-hitran-standard-6000`、`sg4-formal`、`sg4-formal-crosstalk`、`sg4-hitran-smoke`）需用原参数重生成后才能使用。

- `wv4-smoke` / `wv4-formal*` — 掺氢天然气；正式 6000 序列集 `wv4-formal-hitran-standard-6000` 可通过 `--experiment-preset formal-hitran-standard-6000` 一键固定。
- `sg4-smoke` / `sg4-formal` — 合成气 smoke / 正式集（empirical 后端，6000 序列）

## 开发注意事项

### 物理建模关键约束

1. **CO 与 N₂ 声学近简并**：两者摩尔质量均为 28 g/mol，声速差 <1 m/s。声学和 TCS 通道几乎无法区分 CO 和 N₂，CO 的可观测性主要依赖 NDIR 光学通道。
2. **N₂ 双重角色**：syngas 场景中 N₂ 不在预测目标中，但声学/衰减/热导计算仍需要 N₂ 浓度（`x_N2 = 100 - sum(targets)`），由 syngas `conditions.py` 与 `slow.py._main_feature_condition` 自动透传。
3. **CO NDIR 串扰**：CO 基频 2143 cm⁻¹ 与 CO₂ ν₃ 2349 cm⁻¹ 间隔 ~200 cm⁻¹，宽带滤光片下存在串扰。已实现 3×3 矩阵 `src/sim/generation/syngas/optical_crosstalk.py`；默认 `enable_co_crosstalk=False`（Step 1：CO 通道仅含自身吸收），切到 `True` 启用 CO₂↔CO 互扰（Step 2 ablation）。

### 文件约定

- ID 规范：MixtureId = `M000001`，SequenceId = `Q000001`
- split 主键：`mixture_id`（不是 sequence_id）
- benchmark 产物：`condition_grid_sequence.csv`, `sequence_index.csv`, `manifest.json`, `labels/y.npy`, `sequences/slow.npy`, `metadata/label_names.npy` 等
- manifest 必须含 `composition_scheme` / `background_fields` / `slow_channels` / `labels`，下游以 manifest 为准而非全局常量
- 配置驱动：所有超参数通过 `configs/experiment/*.json`，不硬编码

### 测试

修改 `src/` 下代码后必须运行 `python -m pytest`，以实际通过数为准（当前主线 462）。新增 syngas 功能在 `tests/test_syngas_*.py` 系列。修改共用文件（waveforms / manifest / validation / metrics / losses / trainer / cli）前后都要对两个场景的测试都跑一遍。

## 相关文档

| 文档 | 位置 | 内容 |
|------|------|------|
| 当前实验导读 | `docs/AI_CONTEXT_GUIDE.md` | 比 ARCHITECTURE.md 更接近当前状态的动态主线导读 |
| 当前 DL 主线 | `docs/DL相位统计稳定提取与保留方案.md` | 当前活跃的 DL 方案；旧 PhaseWindowTCN 归档于 `docs/整理归档/dl_iteration_plans/` |
| 架构说明 | `docs/ARCHITECTURE.md` | 已落地的 v4 架构契约 |
| 合成气文档导航 | `docs/syngas/README.md` | 新四组分文档索引、阅读顺序与迁移说明 |
| 合成气适配方案 | `docs/syngas/adaptation_plan.md` | 整合审查修正的完整实施方案 |
| 物理常数速查 | `docs/syngas/physics_references.md` | CO 声学/光学常数 + 编码片段 |
| LHS 采样设计 | `docs/syngas/lhs_sampling_design.md` | 方案 B + 条件顺序采样实现规范 |
| CO 串扰设计 | `docs/syngas/co_crosstalk_design.md` | CO 通道 3×3 光学串扰与消融设计 |
| 文献检索汇总 | `docs/syngas/references/README.md` | 4 份子报告索引 + 可编码常数 |
| CO 声学常数 | `docs/syngas/references/co_acoustic_constants.md` | 声速/弛豫/H₂O 耦合详细文献 |
| CO 光学参数 | `docs/syngas/references/co_optical_hitran.md` | HITRAN 谱线/NDIR 滤光片/串扰 |
| 组分分布 | `docs/syngas/references/syngas_composition_ranges.md` | LHS 采样区间文献支撑 |
| 传感器综述 | `docs/syngas/references/syngas_sensing_survey.md` | 商用系统对比 + 可行性评估 |
| 学长 RCDW 复现 | `docs/学长算法/RCDW_实施完成情况.md` | 独立子工程 `rcdw_mgda/` 的端到端落地状态（与主线 src/ 完全隔离，互不影响主线 462 tests） |
| 工作原则 | `AGENTS.md` | AI 协作规则与边界 |

## 环境

- Python 3.10–3.13（排除 3.14）；当前已验证环境为 Python 3.12 虚拟环境
- 核心依赖：numpy, scipy, scikit-learn, torch, hitran-api, pytest
- 安装：优先 `pip install -r requirements.txt`；开发用 `pip install -e .[dev]`
