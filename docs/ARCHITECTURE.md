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
- 光学 backend：benchmark 默认使用 `hitran_hapi_v1`，生成前按同一批 conditions 校验 HITRAN cache 完整性，生成中只读 cache、不联网、不写谱线缓存；`empirical_v1` 仍可显式选择，用于旧合成经验链路和回归对照。
- HITRAN 光谱积分支撑：`src/sim/generation/spectral/` 已具备表格谱积分、HAPI 适配、缓存、单位换算和默认滤光片/网格配置；`src/pipeline/precompute_hitran_benchmark_cache.py` 按 benchmark 的 `sequence_count/seed/sampling_strategy` 预计算每条 condition 的 T/P cache，`src/pipeline/precompute_hitran_spectra.py` 保留通用通道预计算入口。默认滤光片当前使用 InfraTec NBP 行业参考占位（`filter_source.type=industry_reference_only`），待目标传感器 TraceGas-HC-NDIR 实际 datasheet 替换。
- DL 数据加载：`V4BenchmarkDataset` 消费 v4 benchmark 目录，支持慢变量/超声/光纤麦克风三模态、NTC/NCT 格式切换、lazy memmap、按 split 消费。
- DL 模型注册：`MODEL_REGISTRY` + `build_model()` 工厂，已落地 `CNN1DRegressor`（Conv1D + AdaptiveAvgPool + MLP head）和 `TCNRegressor`（因果 Conv1D 残差块 + AdaptiveAvgPool + MLP head），`TCNRegressor.receptive_field` 记录模型时间感受野。
- v4 输出契约：正式 split 只使用 `splits/train.csv`、`splits/val.csv`、`splits/test.csv`、`splits/extrapolation.csv`；不写 V3 的 `*_sequence_ids.csv` 旧命名。

## 未迁移

- 更复杂的环境串扰参数化。
- 目标传感器 TraceGas-HC-NDIR 实际 datasheet 替换当前行业参考占位、真实 PNNL/NIST 或仪器外部谱表对照和更完整谱源版本管理；当前已具备 HITRAN benchmark 默认接入、通用 CSV sanity check、HAPI 适配层、谱线缓存和 InfraTec NBP 行业参考占位（含 `filter_source` 元信息）。
- ML feature package 导出与传统模型训练。
- DL 训练编排（loss/metrics/trainer）、checkpoint 管理。
- 汇总、绘图、状态管理和报告生成。
