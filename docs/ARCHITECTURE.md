# v4 架构落地说明

本文件记录 v4 当前已落地的目标架构片段。主参考仍为 V3 项目中的 `docs/新项目目标架构说明.md`。

## 已落地

- 顶层目录契约：`src/configs/data/outputs/docs/experiments/tests`。
- 配置分组契约：`configs/data`、`configs/model`、`configs/train`、`configs/eval`、`configs/experiment`。
- 输出分区契约：`outputs/runs`、`outputs/summary`、`outputs/reports`、`outputs/archive`。
- 环境依赖契约：`pyproject.toml` 声明项目元数据、Python 范围和运行依赖，`requirements.txt` 提供普通 pip 安装入口；当前主环境建议 Python 3.10-3.13，不使用 Python 3.14 作为正式运行环境。
- ID 契约：`MixtureId` 使用 `M000001` 风格，`SequenceId` 使用 `Q000001` 风格。
- split 打包契约：默认按 `mixture_id` 分组，不把 `mixture_id` 改写为 `sequence_id`。
- run 最小产物契约：`config.json`、`summary.json`、`component_metrics.csv`、`predictions.csv`、`train_log.csv`、`report.md`。
- benchmark 生成主线：`src/pipeline/generate_benchmark.py` 调度 `src/sim/generation/benchmark.py`，写出 `condition_grid_sequence.csv`、`sequence_index.csv`、`labels/*`、`sequences/*`、`metadata/*`、`splits/*.csv`、`scalers/*`、`manifest.json` 和 `quality/validation_summary.json`。其中超声链路会额外写出 `ultrasonic_tof_s`、`ultrasonic_tof_observed_s`、`ultrasonic_peak_index`、`ultrasonic_sound_speed_m_per_s`、`ultrasonic_sound_speed_estimated_m_per_s`、`ultrasonic_alpha_true_npm`、`ultrasonic_tof_quality`、`ultrasonic_tof_accepted` 派生数组。默认使用拉丁超立方采样（LHS），可通过 `sampling_strategy="random"` 切回纯随机采样。CLI 生成路径支持按 sequence chunk 多进程并行，最终 dataset 文件仍由主进程顺序合并写出。
- 性能参数契约：`pipeline.generate_benchmark` 的 CLI 未传 `--workers` 时使用 `default_worker_count(sequence_count)`，即默认保留 2 个逻辑线程给系统并最多使用 24 个 worker；内存密集型 `pipeline.precompute_hitran_benchmark_cache` 默认最多使用 4 个 worker，并通过有界 pending 队列限制进程池任务驻留。`BenchmarkGenerationSpec.workers` 的 Python API 默认值仍是 `1`，以保留程序化调用的串行语义。`--chunk-size` 默认按 `ceil(sequences / workers)`，`--temp-dir` 默认使用 staging 目录下的 `.chunks`，`--keep-chunks` 只用于调试。
- 长时序协议：benchmark 支持 `short/standard/long/xlong` 时间轴预设、显式 `timesteps/dt_s` 覆盖、`standard_exposure/variable_onset/fast_transient/incomplete_recovery/multi_pulse` 阶段 profile，以及 per-sequence `stage_jitter`。`manifest.json` 与 `metadata/waveform_spec.json` 写入 `stage_profile`、`stage_jitter` 和 `phase_schedule`，旧四分位 `standard_exposure` 兼容层保留。
- 采样策略：`src/sim/generation/conditions.py` 基于 `scipy.stats.qmc.LatinHypercube` 进行 3D LHS 空间填充采样，H₂ 双峰分布保留，策略写入 manifest。
- 声程候选：`BenchmarkGenerationSpec.path_lms` 默认使用 `(0.20, 0.25, 0.30, 0.35, 0.40)`，CLI 通过 `--path-lms` 覆盖，候选值写入 `manifest.json` 和 `metadata/waveform_spec.json`。
- 声学模型契约：`manifest.json` 和 `metadata/waveform_spec.json` 会记录 `ultrasonic_model = "tof_observed_transducer_proxy_v1"`、`ultrasonic_system_delay_model = "fixed_delay_plus_trigger_jitter_v1"`、`ultrasonic_transducer_response_model = "second_order_resonant_bandpass_proxy_v1"`、`fiber_mic_model = "fiber_interferometric_proxy_v1"`、`fiber_optical_demodulation_model = "linear_phase_demodulation_proxy_v1"`、`acoustic_attenuation_model = "semi_empirical_multigas_relaxation_proxy_v2"`。当前 `fiber_mic` 已实现光纤干涉式代理链路，但参数仍是仿真占位，正式实验前需要用真实探头、解调器和 DAQ 标定值替换。
- 光学 backend：benchmark 默认使用 `hitran_hapi_v1`，生成前按同一批 conditions 校验 HITRAN cache 完整性，生成中只读 cache、不联网、不写谱线缓存；`empirical_v1` 仍可显式选择，用于旧合成经验链路和回归对照。
- HITRAN 光谱积分支撑：`src/sim/generation/spectral/` 已具备表格谱积分、HAPI 适配、缓存、单位换算和默认滤光片/网格配置；`src/pipeline/precompute_hitran_benchmark_cache.py` 按 benchmark 的 `sequence_count/seed/sampling_strategy` 预计算每条 condition 的 T/P cache，先串行确保 HAPI 原始表，再以保守进程数和有界 pending 队列补齐缺失 `.npz` cache，已有 cache 直接跳过；`src/pipeline/precompute_hitran_spectra.py` 保留通用通道预计算入口。默认滤光片当前使用 InfraTec NBP 行业参考占位（`filter_source.type=industry_reference_only`），待目标传感器 TraceGas-HC-NDIR 实际 datasheet 替换。
- DL 数据加载：`V4BenchmarkDataset` 消费 v4 benchmark 目录，支持慢变量/超声/光纤麦克风三模态、NTC/NCT 格式切换、lazy memmap、按 split 消费；训练期可选 `TimeSeriesAugmentConfig` 做窗口切片/重采样抖动，默认关闭。
- DL 模型注册：`MODEL_REGISTRY` + `build_model()` 工厂，已落地 `CNN1DRegressor`、`TCNRegressor`、`LSTMRegressor`、`TransformerRegressor` 和 `PatchTSTRegressor`。CNN/TCN 支持 `mean/last/attention` 时序聚合；`TCNRegressor.receptive_field` 记录模型时间感受野，并可按 `target_timesteps` 自动扩展通道层数。
- DL 训练闭环：`src/dl/training` 已落地 `WeightedMSELoss`、`SumConstraintLoss`、回归指标、`build_optimizer`、轻量 `Trainer.fit/evaluate/predict` 和 checkpoint 保存/加载。当前定位为单卡最小闭环，不包含正式 argparse CLI、LR scheduler、early stopping 或完整 run report 写出。
- v4 输出契约：正式 split 只使用 `splits/train.csv`、`splits/val.csv`、`splits/test.csv`、`splits/extrapolation.csv`；不写 V3 的 `*_sequence_ids.csv` 旧命名。
- 传统 ML 与协议评估：`src/ml` 支持 Mean/Ridge baseline、full/per-phase/early-window 特征窗口，`python -m ml.cli --protocol` 可输出 JSON 或 Markdown baseline protocol report。
- 存储策略：正式大规模数据集默认推荐 `storage=memmap`，下游 ML/DL 已能直接读取 `.npy`/memmap。若需要兼容压缩包，生成后用 `python -m pipeline.bundle_waveform_sequence --dataset-dir <dataset>` 单独打包 `sequences/waveform_sequence.npz`；`storage=npz/both` 保留兼容，但不作为大规模主路径。
- 验证基线：核心依赖为 `numpy/scipy/torch/pytest/hitran-api`；`python -m pytest` 当前为 187 passed。

## 未迁移

- 更复杂的环境串扰参数化。
- 目标传感器 TraceGas-HC-NDIR 实际 datasheet 替换当前行业参考占位、真实 PNNL/NIST 或仪器外部谱表对照和更完整谱源版本管理；当前已具备 HITRAN benchmark 默认接入、通用 CSV sanity check、HAPI 适配层、谱线缓存和 InfraTec NBP 行业参考占位（含 `filter_source` 元信息）。
- 独立 ML feature package 落盘导出尚未迁移；传统 ML baseline 与 baseline protocol report 入口已落地。
- DL 正式训练 CLI、LR scheduler、early stopping、多 GPU/分布式训练、完整 run report 写出。
- 跨 run 汇总、绘图和状态管理。
