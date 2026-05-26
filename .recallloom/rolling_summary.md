<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-26 -->
<!-- file-state: revision=21 | updated-at=2026-05-26T08:49:47+08:00 | writer-id=Codex | base-workspace-revision=39 -->

<!-- section: current_state -->
- sim 核心模块完成度约 87%：core、generation、packaging、validation 已落地，光学链路包含经验 NDIR、交叉敏感度、本地光谱积分、HITRAN HAPI 适配、单位换算、缓存和新默认窗口真实预计算。
- dl/data 完成度约 40%：V4BenchmarkDataset 支持慢变量、超声、光纤三模态，支持 NTC/NCT、split 加载、lazy memmap 和 scaler 归一化。
- dl/models 完成度约 25%：MODEL_REGISTRY + build_model 工厂 + CNN1DRegressor + TCNRegressor 已落地，TCN 暴露 dilations 与 receptive_field。
- dl/training 完成度约 20%：loss 注册与构造、整体回归指标、按组分回归指标已落地；trainer、checkpoint 管理和训练配置仍待实现。
- pipeline 完成度约 22%：layout、benchmark 生成 CLI、HITRAN 预计算 CLI 和 optical backend 对照 CLI 可用。
- configs 完成度约 6%：spectral defaults 使用 InfraTec 行业参考占位，CH4 3030/147 cm-1、CO2 2347/93 cm-1，并记录 filter_source；默认 HITRAN grid 已扩大为 CH4 2880-3180 cm-1、CO2 2250-2445 cm-1。
- tests 共 107 个，覆盖 sim、dl、pipeline，当前全部通过。
- 本地已用 hitran-api 1.3.0.0 在新默认窗口重新预计算 CH4、CO2、H2O 谱线缓存；HITRAN cache 属于本地运行产物，不进 git。
- 当前 benchmark 仍使用 empirical_v1，并记录 optical_absorption_backend；hitran_hapi_v1 仍作为显式预计算和对照入口存在。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约，不追求功能完整度。
- 经验吸收系数不得表述为真实标定值；如需物理支撑，应使用 HITRAN HAPI 或 PNNL、NIST 谱库积分 backend，并在 manifest 记录 optical_absorption_backend。
- 默认 HITRAN grid 必须至少覆盖当前滤光片 center +/- FWHM；grid 或滤光片规格变化后必须重新预计算 cache。
- 旧 2960-3100 / 2280-2410 cm-1 窗口只作为早期真实下载验证记录，不再是当前默认窗口。
- HAPI raw table name 必须绑定气体和波数窗口，避免不同 NDIR 通道复用错误谱线范围。
- 真实谱线缓存属于本地运行产物，不纳入 git；代码和配置只保留可复现入口。
- 滤光片占位现在使用 InfraTec NBP 行业参考值，属 industry_reference_only，正式 benchmark 前必须替换为目标传感器 TraceGas-HC-NDIR 的实际 datasheet。
- 训练模块先建立显式、可测试的 loss/metrics 前置契约，再进入 trainer、checkpoint 和实验追踪。

<!-- section: risks_open_questions -->
- 目标传感器 TraceGas-HC-NDIR datasheet 仍未到手，当前 CH4 fwhm 147 cm-1 / CO2 fwhm 93 cm-1 是 InfraTec NBP 行业参考值占位，非厂商实际值。
- HITRAN 新窗口预计算和 empirical 对照已跑通，但尚未用仪器、PNNL 或 NIST 数据做数值 sanity check。
- PNNL 与 NIST 对照尚未实施，缺少谱表 parser 和外部 sanity check。
- benchmark 主线尚未支持显式切换到 spectral backend。
- 端到端 trainer、checkpoint 管理、训练配置和实验追踪仍未实现。
- TCN 具体通道数、kernel_size、层数仍未进入正式模型配置和实验对照；时间步分布仍为固定四等分。

<!-- section: next_step -->
继续 hitran_hapi_v1 的优先下一步：导入 PNNL/NIST 定量谱表做外部 sanity check；或实现 benchmark 生成时显式选择 optical_absorption_backend 的配置入口，但正式切换前仍需 TraceGas-HC-NDIR datasheet。

<!-- section: recent_pivots -->
- 2026-05-26：继续推进 hitran_hapi_v1，默认 HITRAN grid 扩大为 CH4 2880-3180 cm-1、CO2 2250-2445 cm-1，并新增测试固定 grid 覆盖滤光片 center +/- FWHM。
- 2026-05-26：用真实 HAPI 1.3.0.0 在新窗口重新预计算本地 cache；precompute 输出 CH4 通道吸收约 0.8601、CO2 通道吸收约 1.4407，compare_optical_backends 对照跑通。
- 2026-05-26：完成 Phase 4.2 训练基础组件最小实现，新增 dl.training loss/metrics，并验证 pytest -q 为 107 passed。
- 2026-05-25：完成 HITRAN 主路径原型和真实下载验证，修复 HAPI 表名按气体复用导致的跨窗口缓存污染风险，并把 HITRAN cache 作为本地运行产物忽略。
- 2026-05-25：spectral defaults 中 CH4/CO2 FWHM 从 smoke 值切到 InfraTec NBP 行业参考值，目标传感器 TraceGas-HC-NDIR 实际 datasheet 仍待获取。
