<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-26 -->
<!-- file-state: revision=24 | updated-at=2026-05-26T11:55:36+08:00 | writer-id=Codex | base-workspace-revision=45 -->

<!-- section: current_state -->
- sim 核心模块完成度约 90%：core、generation、packaging、validation 已落地；光学链路包含 empirical_v1 兼容路径、hitran_hapi_v1 默认 benchmark backend、HITRAN cache-only 预检查、RH 到 H2O 光学气体换算、本地光谱积分、单位换算、缓存、预计算 CLI，以及外部定量谱表 sanity check 通用 CSV 入口。
- benchmark 主线默认 optical_absorption_backend 是 hitran_hapi_v1；empirical_v1 保留为显式 opt-in 回归/对照路径。2026-05-26 代码审查修复后，HITRAN 主线不再调用 empirical main_sensor_features 计算 NDIR，只用 thermal_conductivity_sensor_feature 计算 V_TCS，NDIR 由 cache-only HITRAN 光谱积分给出。
- benchmark 生成阶段对 HITRAN 采用 cache_only_prechecked：生成 dataset 文件前收集同批 conditions 所需 cache keys，缺失则一次性显式失败，不现场导入 HAPI、不联网、不写 cache。
- 同一 (channel, HitranGridSpec) 的表格谱会准备成 PreparedTabulatedSpectra 后复用，栅格一致性与滤光片响应只在 cache 载入阶段处理一次。
- configs/data/spectral-defaults.json 已作为运行时 spectral 默认值 source-of-truth；defaults.py 读取该 JSON 并构造 HitranGasSpec、NDIRFilter 和 HitranGridSpec。
- slow.py 的 HITRAN NDIR equilibrium 逐 timestep 使用当前 blend 组分、当前 L_m、condition 固定 T_C_base/P_MPa_base/H_RH_base 计算；短 phase 声程扫描现在会覆盖声程候选端点。
- dl/data 完成度约 45%：V4BenchmarkDataset 支持慢变量、超声、光纤三模态，支持 NTC/NCT、split 加载、真正 lazy memmap 和 scaler 归一化；数组保持 memmap 到单条样本取出时才转 float32。
- scaler 契约已收敛：Z_SCORE_STD_EPSILON=1e-12 位于 src/sim/packaging/constants.py，fit/apply 共用阈值；fit_z_score_scalers 固定 channels-last，不再暴露未生效 channel_axis 参数。
- dl/models 完成度约 25%：MODEL_REGISTRY + build_model 工厂 + CNN1DRegressor + TCNRegressor 已落地，TCN 暴露 dilations 与 receptive_field。
- dl/training 完成度约 20%：loss 注册与构造、整体回归指标、按组分回归指标已落地；regression_metrics 的 R2 口径已明确为 pooled R2，不是逐组分宏平均；trainer、checkpoint 管理和训练配置仍待实现。
- pipeline 完成度约 30%：layout、benchmark 生成 CLI、benchmark HITRAN 预计算 CLI、optical backend 对照 CLI，以及外部定量谱表 sanity check CLI/API 可用。
- configs 完成度约 8%：spectral defaults 使用 InfraTec 行业参考占位，CH4 3030/147 cm-1、CO2 2347/93 cm-1，并记录 filter_source；默认 HITRAN grid 为 CH4 2880-3180 cm-1、CO2 2250-2445 cm-1。
- tests 共 132 个，当前全部通过；最新验证为 python -m pytest tests 132 passed，git diff --check 通过，verify-change 通过。
- 最新提交已创建：0c004abafa352747ee53c2c613f52418adb54f2f，subject 为 fix(benchmark): resolve review findings；提交包含代码修复、测试、文档同步、docs/CODE_REVIEW_2026-05-26.md 和 src/sim/packaging/constants.py。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约，不追求功能完整度。
- benchmark 默认生成主线使用 hitran_hapi_v1；empirical_v1 只作为显式 opt-in 兼容、回归和对照路径。
- benchmark 生成永不现场下载 HITRAN；真实下载和 cache 写入只发生在预计算 CLI。
- HITRAN T/P cache key 粒度固定为 temperature_k=round(T_C+273.15,3)，pressure_atm=round(P_MPa/0.101325,6)。
- H2O 由 T/P/RH 换算，只作为光学吸收气体，不进入 label 组分和 100% 组分校验。
- HITRAN 多气体滤光片积分表达通道交叉响应，默认路径不再叠加 empirical apply_optical_crosstalk。
- 默认 HITRAN grid 必须至少覆盖当前滤光片 center +/- FWHM；grid 或滤光片规格变化后必须重新预计算 cache。
- configs/data/spectral-defaults.json 是 spectral 默认值唯一运行时来源，不再维护 Python 常量镜像作为第二来源。
- 外部定量谱表 sanity check 的标准入口是通用 CSV contract：显式 wavenumber_cm1、absorption_coeff 和单位；禁止单位猜测，重采样禁止外推。
- 滤光片占位现在使用 InfraTec NBP 行业参考值，属 industry_reference_only，正式 benchmark 前必须替换为目标传感器 TraceGas-HC-NDIR 的实际 datasheet。
- 训练模块先建立显式、可测试的 loss/metrics 前置契约，再进入 trainer、checkpoint 和实验追踪。

<!-- section: risks_open_questions -->
- 目标传感器 TraceGas-HC-NDIR datasheet 仍未到手，当前 CH4 fwhm 147 cm-1 / CO2 fwhm 93 cm-1 是 InfraTec NBP 行业参考值占位，非厂商实际值。
- 真实 PNNL/NIST 或仪器定量谱表仍未导入；通用 CSV sanity check 路径已完成，但外部实测/数据库对照仍待执行。
- 真实大规模 benchmark 使用默认 HITRAN 前必须先用相同 sequences、seed、sampling_strategy 预计算对应 conditions 的 cache；cache 文件是本地运行产物，不纳入 git。
- 端到端 trainer、checkpoint 管理、训练配置和实验追踪仍未实现。
- TCN 具体通道数、kernel_size、层数仍未进入正式模型配置和实验对照；时间步分布仍为固定四等分。

<!-- section: next_step -->
若继续工程主线，优先实现最小 Trainer 与 checkpoint/训练配置 smoke 链路；若继续光学链路，先用目标 benchmark spec 运行 pipeline.precompute_hitran_benchmark_cache 生成真实 cache，再生成一个小规模默认 HITRAN smoke benchmark 做数值抽查，并继续推进 TraceGas-HC-NDIR datasheet 与真实 PNNL/NIST 或仪器谱表对照。

<!-- section: recent_pivots -->
- 2026-05-26：根据 docs/CODE_REVIEW_2026-05-26.md 完成审查修复并提交 0c004ab，修复 lazy memmap、scaler 阈值契约、HITRAN 主线 empirical NDIR 死代码、PreparedTabulatedSpectra 缓存和 spectral defaults source-of-truth；验证 python -m pytest tests 为 132 passed。
- 2026-05-26：完成 hitran_hapi_v1 对 benchmark 主线的默认接入，新增 optical_backend/gas_state、benchmark cache-only 预检查、benchmark HITRAN 预计算 CLI、逐 timestep NDIR equilibrium，并验证 pytest -q 为 125 passed。
- 2026-05-26：完成 PNNL/NIST 外部定量谱表 sanity check 通用 CSV 入口，新增 quantitative_table.py、sanity_check_tabulated_spectra.py 和对应测试；全量 pytest -q 为 119 passed。
- 2026-05-26：继续推进 hitran_hapi_v1，默认 HITRAN grid 扩大为 CH4 2880-3180 cm-1、CO2 2250-2445 cm-1，并新增测试固定 grid 覆盖滤光片 center +/- FWHM。
- 2026-05-26：用真实 HAPI 1.3.0.0 在新窗口重新预计算本地 cache；precompute 输出 CH4 通道吸收约 0.8601、CO2 通道吸收约 1.4407，compare_optical_backends 对照跑通。
- 2026-05-26：完成 Phase 4.2 训练基础组件最小实现，新增 dl.training loss/metrics，并验证 pytest -q 为 107 passed。
- 2026-05-25：完成 HITRAN 主路径原型和真实下载验证，修复 HAPI 表名按气体复用导致的跨窗口缓存污染风险，并把 HITRAN cache 作为本地运行产物忽略。
- 2026-05-25：spectral defaults 中 CH4/CO2 FWHM 从 smoke 值切到 InfraTec NBP 行业参考值，目标传感器 TraceGas-HC-NDIR 实际 datasheet 仍待获取。
