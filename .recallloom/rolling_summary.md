<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-26 -->
<!-- file-state: revision=23 | updated-at=2026-05-26T10:19:33+08:00 | writer-id=Codex | base-workspace-revision=43 -->

<!-- section: current_state -->
- sim 核心模块完成度约 90%：core、generation、packaging、validation 已落地；光学链路包含 empirical_v1 兼容路径、hitran_hapi_v1 默认 benchmark backend、HITRAN cache-only 预检查、RH 到 H2O 光学气体换算、本地光谱积分、单位换算、缓存、预计算 CLI，以及外部定量谱表 sanity check 通用 CSV 入口。
- benchmark 主线默认 optical_absorption_backend 已切换为 hitran_hapi_v1；empirical_v1 保留为显式 opt-in 回归/对照路径。
- benchmark 生成阶段对 HITRAN 采用 cache-only_prechecked：生成 dataset 文件前收集同批 conditions 所需 cache keys，缺失则一次性显式失败，不现场导入 HAPI、不联网、不写 cache。
- 新增 pipeline.precompute_hitran_benchmark_cache，可用与生成相同的 sequences、seed、sampling_strategy 派生 conditions，并按 CH4/CO2/H2O、ch4/co2 通道、逐 condition T/P 写入 cache。
- slow.py 的 HITRAN NDIR equilibrium 已改为逐 timestep 使用当前 blend 组分、当前 L_m、condition 固定 T_C_base/P_MPa_base/H_RH_base 计算；动态滞后、漂移、随机游走和噪声仍沿用现有 slow channel 机制。
- dl/data 完成度约 40%：V4BenchmarkDataset 支持慢变量、超声、光纤三模态，支持 NTC/NCT、split 加载、lazy memmap 和 scaler 归一化。
- dl/models 完成度约 25%：MODEL_REGISTRY + build_model 工厂 + CNN1DRegressor + TCNRegressor 已落地，TCN 暴露 dilations 与 receptive_field。
- dl/training 完成度约 20%：loss 注册与构造、整体回归指标、按组分回归指标已落地；trainer、checkpoint 管理和训练配置仍待实现。
- pipeline 完成度约 30%：layout、benchmark 生成 CLI、benchmark HITRAN 预计算 CLI、optical backend 对照 CLI，以及外部定量谱表 sanity check CLI/API 可用。
- configs 完成度约 6%：spectral defaults 使用 InfraTec 行业参考占位，CH4 3030/147 cm-1、CO2 2347/93 cm-1，并记录 filter_source；默认 HITRAN grid 为 CH4 2880-3180 cm-1、CO2 2250-2445 cm-1。
- tests 共 125 个，当前全部通过；git diff --check 无 whitespace error，仅有 Windows LF/CRLF 提示。
- 本地已用 hitran-api 1.3.0.0 在新默认窗口重新预计算 CH4、CO2、H2O 谱线缓存；HITRAN cache 属于本地运行产物，不进 git。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约，不追求功能完整度。
- benchmark 默认生成主线使用 hitran_hapi_v1；empirical_v1 只作为显式 opt-in 兼容、回归和对照路径。
- benchmark 生成永不现场下载 HITRAN；真实下载和 cache 写入只发生在预计算 CLI。
- HITRAN T/P cache key 粒度固定为 temperature_k=round(T_C+273.15,3)，pressure_atm=round(P_MPa/0.101325,6)。
- H2O 由 T/P/RH 换算，只作为光学吸收气体，不进入 label 组分和 100% 组分校验。
- HITRAN 多气体滤光片积分表达通道交叉响应，默认路径不再叠加 empirical apply_optical_crosstalk。
- 默认 HITRAN grid 必须至少覆盖当前滤光片 center +/- FWHM；grid 或滤光片规格变化后必须重新预计算 cache。
- 旧 2960-3100 / 2280-2410 cm-1 窗口只作为早期真实下载验证记录，不再是当前默认窗口。
- HAPI raw table name 必须绑定气体和波数窗口，避免不同 NDIR 通道复用错误谱线范围。
- 真实谱线缓存属于本地运行产物，不纳入 git；代码和配置只保留可复现入口。
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
若继续光学链路，先用目标 benchmark spec 运行 pipeline.precompute_hitran_benchmark_cache 生成真实 cache，再生成一个小规模默认 HITRAN smoke benchmark 做数值抽查；同时继续推进 TraceGas-HC-NDIR datasheet 与真实 PNNL/NIST 或仪器谱表对照。

<!-- section: recent_pivots -->
- 2026-05-26：完成 hitran_hapi_v1 对 benchmark 主线的默认接入，新增 optical_backend/gas_state、benchmark cache-only 预检查、benchmark HITRAN 预计算 CLI、逐 timestep NDIR equilibrium，并验证 pytest -q 为 125 passed。
- 2026-05-26：完成 PNNL/NIST 外部定量谱表 sanity check 通用 CSV 入口，新增 quantitative_table.py、sanity_check_tabulated_spectra.py 和对应测试；全量 pytest -q 为 119 passed。
- 2026-05-26：继续推进 hitran_hapi_v1，默认 HITRAN grid 扩大为 CH4 2880-3180 cm-1、CO2 2250-2445 cm-1，并新增测试固定 grid 覆盖滤光片 center +/- FWHM。
- 2026-05-26：用真实 HAPI 1.3.0.0 在新窗口重新预计算本地 cache；precompute 输出 CH4 通道吸收约 0.8601、CO2 通道吸收约 1.4407，compare_optical_backends 对照跑通。
- 2026-05-26：完成 Phase 4.2 训练基础组件最小实现，新增 dl.training loss/metrics，并验证 pytest -q 为 107 passed。
- 2026-05-25：完成 HITRAN 主路径原型和真实下载验证，修复 HAPI 表名按气体复用导致的跨窗口缓存污染风险，并把 HITRAN cache 作为本地运行产物忽略。
- 2026-05-25：spectral defaults 中 CH4/CO2 FWHM 从 smoke 值切到 InfraTec NBP 行业参考值，目标传感器 TraceGas-HC-NDIR 实际 datasheet 仍待获取。
