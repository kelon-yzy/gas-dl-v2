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
- PhaseWindowTCN 多窗口 DL 链路：`V4BenchmarkDataset` 支持 `phase_windows` 返回 `(W, T, C)` 多窗口输入；`PhaseWindowTCNRegressor` 消费真实 `full + exposure + recovery` 窗口视图，支持 `share_window_encoder`、`output_mode={"raw4","softmax100","gas_head"}` 和更深 TCN 通道配置。当前活跃结构消融配置位于 `configs/phase_window_tcn_ablation/`。
- DL 训练闭环：`src/dl/training` 已落地 loss registry、`free_component_mse`、回归指标、`build_optimizer`、轻量 `Trainer.fit/evaluate/predict`、checkpoint 保存/加载、early stopping、ReduceLROnPlateau、AMP、训练进度 JSONL 和 run_config/metrics JSON 写出。当前定位为单卡实验闭环，不包含多 GPU/分布式训练。
- v4 输出契约：正式 split 只使用 `splits/train.csv`、`splits/val.csv`、`splits/test.csv`、`splits/extrapolation.csv`；不写 V3 的 `*_sequence_ids.csv` 旧命名。
- 传统 ML 与协议评估：`src/ml` 支持 Mean/Ridge baseline、full/per-phase/early-window 特征窗口，以及 `full + exposure + recovery` 多窗口特征拼接；`ridge_multiwindow_all_modalities` 是当前正式 phase-aware ML 主线。
- 实验编排：`src/pipeline/experiment_config.py` 与 `src/pipeline/run_experiment.py` 支持 JSON 实验配置、ML/DL run 列表、`phase_windows`、dry-run、summary CSV 和 Markdown report。当前 PhaseWindowTCN 结构消融首批和 followup 配置已落地。
- 存储策略：正式大规模数据集默认推荐 `storage=memmap`，下游 ML/DL 已能直接读取 `.npy`/memmap。若需要兼容压缩包，生成后用 `python -m hg.pipeline.bundle_waveform_sequence --dataset-dir <dataset>` 单独打包 `sequences/waveform_sequence.npz`；`storage=npz/both` 保留兼容，但不作为大规模主路径。
- 验证基线：核心依赖为 `numpy/scipy/torch/pytest/hitran-api`；`python -m pytest` 当前为 462 passed（hydrogen_ng 353 + syngas 109，含 Stage Ⅱ ablation 18 个新增测试）。

## 合成气适配（已落地，2026-06-26）

> 详细方案见 `docs/syngas/adaptation_plan.md`，文档导航见 `docs/syngas/README.md`，物理常数见 `docs/syngas/physics_references.md`。

**目标**：新增合成气检测场景（H₂/CH₄/CO₂/CO，N₂ 为背景气，sum<100%），与现有掺氢天然气场景并存。

**实施策略**：采用**分支隔离**，syngas 走独立 schema 与子包，hydrogen_ng 路径完全保留；共用 packaging / waveforms / validation / dl trainer 等只做向后兼容的可选参数化改造。

**已落地的核心入口**：

- `src/sim/core/syngas_schema.py` — syngas 专用 schema：`COMPONENT_FIELDS = ("x_H2","x_CH4","x_CO2","x_CO")`，`BACKGROUND_FIELDS = ("x_N2",)`，9 个 SLOW_CHANNELS（含 `V_NDIR_CO`）。
- `src/sim/generation/syngas/` 子包：`conditions.py`（方案 B + 条件顺序采样）、`acoustic_physics.py`（CO 声速 / 弛豫衰减 / 热导 / 吸收）、`optical_crosstalk.py`（3×3 矩阵，Step 1/2 切换）、`slow.py`（含 V_NDIR_CO 慢通道）、`benchmark.py`（独立 benchmark 编排）。
- `src/pipeline/generate_syngas_benchmark.py` — CLI 入口，默认 `empirical_v1` 光学后端。
- `configs/sg4/sg4_baseline.json` — DL baseline 配置：`cnn1d` + `weighted_component_mse` + `weighting=inverse_train_var`。
- `tests/test_syngas_{sampling,acoustic_physics,benchmark_generation,dl_training,ablation}.py` — 109 个新测试（91 个基础 + 18 个 Stage Ⅱ ablation）。

**共用代码兼容性改造**（默认行为不变）：

- `src/sim/generation/waveforms.py` — `simulate_waveform_measurement` / `simulate_fiber_mic_measurement` 新增可选 `sound_speed_fn` / `attenuation_fn` / `extra_gas_kwargs` 注入；syngas 走 syngas 物理后端，hg 调用方完全不感知。
- `src/sim/packaging/manifest.py` — `build_manifest` 新增 `schema_version` / `composition_scheme` / `background_fields` 可选参数，默认值匹配 hg 行为。
- `src/sim/validation/integrity.py` — `validate_benchmark_assets` 接受 `component_fields` / `slow_channels` / `background_fields` / `require_sum_100`，默认走 hg schema。
- `src/common/metrics.py` — `conditional_component_metrics` 新增 `bin_components` 可选参数，syngas 传 `("x_CO", "x_CH4")`，hg 默认 `("x_N2", "x_CH4")`。
- `src/dl/training/losses.py` — 新增 `validate_loss_composition_scheme()`，syngas 自动拒绝闭包类 loss；`validate_loss_model_output` 接受 `composition_scheme` 放开 syngas 对 gas-head 模型的限制。
- `src/dl/training/trainer.py` — `Trainer` 构造接受 `component_names` / `composition_scheme`；syngas 拒绝 `target_transform`；evaluate 按 scheme 决定分箱（`co_bins` vs `n2_bins`）、是否计算 `sum_abs_error` / `compositional_metrics`。
- `src/dl/cli.py` — 自动从 `manifest.json` 读 `composition_scheme`、从 `metadata/label_names.npy` 读 label_names，注入 trainer 与 loss 校验；旧 wv4 manifest 无该字段时 fallback 到 hg。

**Manifest 契约扩展**：

- 新增字段：`schema_version`（"v4-syngas-1" or "v4-benchmark-1"）、`composition_scheme`（"syngas" or "hydrogen_ng"）、`background_fields`（syngas: `["x_N2"]`）。
- 旧 wv4 manifest 无 `composition_scheme` 字段时 CLI 自动 fallback 到 "hydrogen_ng"。

**已知物理限制**：CO 与 N₂ 摩尔质量相同（28 g/mol），声速差 <1 m/s，超声波和热导通道对 CO 几乎不敏感。CO 的可观测性主要依赖 NDIR 光学通道。**Stage Ⅱ-1 实测确认**（详见 `docs/syngas/stage_ii_ablation_results.md`）：移除 V_NDIR_CO 后 x_CO R² 从 0.954 跌至 0.484（损失 ~50%，TCN/Ridge 同向），仅保留 V_NDIR_CO + 环境通道即可恢复 0.93-0.94。结论修正：CO **主导依赖** NDIR 光学（非完全依赖），残留 ~50% 可学性来自闭包约束 / V_TCS 热导差异 / 可能的 CO₂ NDIR 弱串扰。

**未完成**：

- HITRAN syngas 后端（阶段 3c）：`syngas.slow.build_sequence_arrays` 在 `optical_absorption_backend="hitran_hapi_v1"` 时抛 `NotImplementedError`；需要联网拉取 CO/CO₂/H₂O 在 [1980, 2310] cm⁻¹ 谱线，并扩展 `optical_backend` 支持三气体通道。
- `run_experiment` 多 run 编排尚未接入 syngas（当前 syngas DL 走 `dl.cli` 单 run）。
- CO 改进分析工具（对标 `analyze_n2_improvement.py`）按需后续补。

## 未迁移

- 更复杂的环境串扰参数化。
- 目标传感器 TraceGas-HC-NDIR 实际 datasheet 替换当前行业参考占位、真实 PNNL/NIST 或仪器外部谱表对照和更完整谱源版本管理；当前已具备 HITRAN benchmark 默认接入、通用 CSV sanity check、HAPI 适配层、谱线缓存和 InfraTec NBP 行业参考占位（含 `filter_source` 元信息）。
- 独立 ML feature package 落盘导出尚未迁移；传统 ML baseline 与 baseline protocol report 入口已落地。
- 多 GPU/分布式训练。
- 跨 run 绘图和更完整状态管理。
