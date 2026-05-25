# v4 架构落地说明

本文件记录 v4 当前已落地的目标架构片段。主参考仍为 V3 项目中的 `docs/新项目目标架构说明.md`。

## 已落地

- 顶层目录契约：`src/configs/data/outputs/docs/experiments/tests`。
- 配置分组契约：`configs/data`、`configs/model`、`configs/train`、`configs/eval`、`configs/experiment`。
- 输出分区契约：`outputs/runs`、`outputs/summary`、`outputs/reports`、`outputs/archive`。
- ID 契约：`MixtureId` 使用 `M000001` 风格，`SequenceId` 使用 `Q000001` 风格。
- split 打包契约：默认按 `mixture_id` 分组，不把 `mixture_id` 改写为 `sequence_id`。
- run 最小产物契约：`config.json`、`summary.json`、`component_metrics.csv`、`predictions.csv`、`train_log.csv`、`report.md`。
- benchmark 生成主线：`src/pipeline/generate_benchmark.py` 调度 `src/sim/generation/benchmark.py`，写出 `condition_grid_sequence.csv`、`sequence_index.csv`、`labels/*`、`sequences/*`、`metadata/*`、`splits/*.csv`、`scalers/*`、`manifest.json` 和 `quality/validation_summary.json`。默认使用拉丁超立方采样（LHS），可通过 `sampling_strategy="random"` 切回纯随机采样。
- 采样策略：`src/sim/generation/conditions.py` 基于 `scipy.stats.qmc.LatinHypercube` 进行 3D LHS 空间填充采样，H₂ 双峰分布保留，策略写入 manifest。
- 声程候选：`BenchmarkGenerationSpec.path_lms` 默认使用 `(0.20, 0.25, 0.30, 0.35, 0.40)`，CLI 通过 `--path-lms` 覆盖，候选值写入 `manifest.json` 和 `metadata/waveform_spec.json`。
- DL 数据加载：`V4BenchmarkDataset` 消费 v4 benchmark 目录，支持慢变量/超声/光纤麦克风三模态、NTC/NCT 格式切换、lazy memmap、按 split 消费。
- DL 模型注册：`MODEL_REGISTRY` + `build_model()` 工厂，已落地 `CNN1DRegressor`（Conv1D + AdaptiveAvgPool + MLP head）。
- v4 输出契约：正式 split 只使用 `splits/train.csv`、`splits/val.csv`、`splits/test.csv`、`splits/extrapolation.csv`；不写 V3 的 `*_sequence_ids.csv` 旧命名。

## 未迁移

- 更复杂的光学谱线重叠、环境串扰参数化。
- ML feature package 导出与传统模型训练。
- DL 训练编排（loss/metrics/trainer）、checkpoint 管理。
- 汇总、绘图、状态管理和报告生成。
