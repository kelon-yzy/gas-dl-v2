# docs 索引

整理日期：2026-06-27

## 顶层活动文档

| 文件                            | 用途                                      |
| ----------------------------- | --------------------------------------- |
| `AI_CONTEXT_GUIDE.md`         | 外部 AI / 新人快速上下文导读                       |
| `ARCHITECTURE.md`             | v4 当前已落地的目标架构契约                         |
| `IMPLEMENTATION_PLAN.md`      | 阶段实施脉络（保留历史进度索引）                        |
| `improvement_plan.md`         | DL 模型改进计划（当前进度索引）                       |
| `dl_model_architecture.md`    | 当前最佳 DL 模型（CNN1DTCNFusionRegressor）算法框架 |
| `DL相位统计稳定提取与保留方案.md`          | **当前 DL 主线方案**（2026-06-27）              |
| `stateful-prancing-papert.md` | 合成气场景（CO/CO₂/CH₄/H₂）适配主线规划              |

## 子目录

| 路径            | 内容                                      |
| ------------- | --------------------------------------- |
| `syngas/`     | 合成气场景文档集，含适配方案、物理常数、LHS 采样、CO 串扰设计、文献参考 |
| `references/` | 文献检索报告与背景分析（含 PDF）                      |
| `patent/`     | 专利材料                                    |
| `assets/`     | 图表 / 二进制（drawio 等）                      |
| `整理归档/`       | 历史归档，分三类，下方展开                           |

## 整理归档

### `整理归档/legacy_consolidated/`

2026-06-15 之前的整合性文档：项目总览、数据集设计、模型架构总览、实验设计、代码结构、N2 改进早期方案、PhasePreservingTCN 详细设计、phase-preserving fusion 设计、PhaseWindowTCN 早期分析。

### `整理归档/dl_iteration_plans/`

DL 改进历代计划与诊断报告：

- `ML模型改进方向分析.md`（2026-06-16）
- `PhaseWindowTCN结构消融实验方案.md`（2026-06-16，被新方案取代）
- `PhaseWindowTCN实验执行与验收流程.md`（2026-06-16，配套执行手册）
- `N2不可学诊断与gas_head参数化分析.md`（2026-06-19）
- `DL下一步改进分步实验计划_20260623.md`
- `DL下一步改进计划_v2_20260623.md`
- `DL改进实验顺序_v3_20260623.md`
- `p1_tcn_capacity_plan.md`（2026-06-24）
- `p1_analysis_report.md`（2026-06-24，P1 容量扩张失败分析）
- `生成正式 HITRAN 标准数据集计划.md`（2026-06-05）

> 当前 DL 主线为顶层 `DL相位统计稳定提取与保留方案.md`，本目录均为历史方案与诊断证据。

### `整理归档/runtime_tuning/`

训练运行历史调优记录：

- `训练配置优化方案.md`（48 GiB 服务器主调优方案，仍可参考）
- `Phase1_Phase2运行指南.md`
- `OOM修正记录.md`（旧 24 GiB 环境）
- `训练配置优化应用记录.md`（历史应用）
- `训练时间优化执行总结.md`（历史总结）

## 引用变更说明

2026-06-27 docs 整理后，下列引用已统一更新：

- 项目根 `README.md`、`docs/IMPLEMENTATION_PLAN.md`、`docs/improvement_plan.md`
- `test_optimized_config.py`、`scripts/compare_training_configs.py`、`scripts/run_phase1_phase2_optimized.sh`

`.recallloom/` 下的日志属于时间快照，未改。整理归档内文件之间的相对引用未改（归档不再维护）。
