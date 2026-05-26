<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-26 -->
<!-- file-state: revision=22 | updated-at=2026-05-26T09:22:36+08:00 | writer-id=Codex | base-workspace-revision=41 -->

<!-- section: current_state -->
- sim 核心模块完成度约 88%：core、generation、packaging、validation 已落地，光学链路包含经验 NDIR、交叉敏感度、本地光谱积分、HITRAN HAPI 适配、单位换算、缓存、新默认窗口真实预计算，以及外部定量谱表 sanity check 通用 CSV 入口。
- dl/data 完成度约 40%：V4BenchmarkDataset 支持慢变量、超声、光纤三模态，支持 NTC/NCT、split 加载、lazy memmap 和 scaler 归一化。
- dl/models 完成度约 25%：MODEL_REGISTRY + build_model 工厂 + CNN1DRegressor + TCNRegressor 已落地，TCN 暴露 dilations 与 receptive_field。
- dl/training 完成度约 20%：loss 注册与构造、整体回归指标、按组分回归指标已落地；trainer、checkpoint 管理和训练配置仍待实现。
- pipeline 完成度约 24%：layout、benchmark 生成 CLI、HITRAN 预计算 CLI、optical backend 对照 CLI，以及外部定量谱表 sanity check CLI/API 可用。
- configs 完成度约 6%：spectral defaults 使用 InfraTec 行业参考占位，CH4 3030/147 cm-1、CO2 2347/93 cm-1，并记录 filter_source；默认 HITRAN grid 为 CH4 2880-3180 cm-1、CO2 2250-2445 cm-1。
- tests 共 119 个，当前全部通过；git diff --check 无 whitespace error，仅有 Windows LF/CRLF 提示。
- 本地已用 hitran-api 1.3.0.0 在新默认窗口重新预计算 CH4、CO2、H2O 谱线缓存；HITRAN cache 属于本地运行产物，不进 git。
- 当前 benchmark 仍使用 empirical_v1，并记录 optical_absorption_backend；hitran_hapi_v1 和外部 tabulated sanity check 均为显式诊断/对照入口。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约，不追求功能完整度。
- 经验吸收系数不得表述为真实标定值；如需物理支撑，应使用 HITRAN HAPI 或 PNNL、NIST、仪器谱表积分 backend，并在 manifest 记录 optical_absorption_backend。
- 默认 HITRAN grid 必须至少覆盖当前滤光片 center +/- FWHM；grid 或滤光片规格变化后必须重新预计算 cache。
- 旧 2960-3100 / 2280-2410 cm-1 窗口只作为早期真实下载验证记录，不再是当前默认窗口。
- HAPI raw table name 必须绑定气体和波数窗口，避免不同 NDIR 通道复用错误谱线范围。
- 真实谱线缓存属于本地运行产物，不纳入 git；代码和配置只保留可复现入口。
- 外部定量谱表 sanity check 的标准入口是通用 CSV contract：显式 wavenumber_cm1、absorption_coeff 和单位；禁止单位猜测，重采样禁止外推。
- 滤光片占位现在使用 InfraTec NBP 行业参考值，属 industry_reference_only，正式 benchmark 前必须替换为目标传感器 TraceGas-HC-NDIR 的实际 datasheet。
- 训练模块先建立显式、可测试的 loss/metrics 前置契约，再进入 trainer、checkpoint 和实验追踪。

<!-- section: risks_open_questions -->
- 目标传感器 TraceGas-HC-NDIR datasheet 仍未到手，当前 CH4 fwhm 147 cm-1 / CO2 fwhm 93 cm-1 是 InfraTec NBP 行业参考值占位，非厂商实际值。
- PNNL/NIST 外部 sanity check 路径已实现，但真实 PNNL/NIST 或仪器数据尚未导入；当前只完成通用 CSV contract、合成谱表测试和伪 HAPI 对照测试。
- benchmark 主线尚未支持显式切换到 spectral backend。
- 端到端 trainer、checkpoint 管理、训练配置和实验追踪仍未实现。
- TCN 具体通道数、kernel_size、层数仍未进入正式模型配置和实验对照；时间步分布仍为固定四等分。

<!-- section: next_step -->
如果继续光学链路，优先用真实 PNNL/NIST 或仪器导出的谱表转换为通用 CSV contract 后运行 sanity_check_tabulated_spectra；若暂无真实谱表，则转向实现 benchmark 生成时显式选择 optical_absorption_backend 的配置入口。

<!-- section: recent_pivots -->
- 2026-05-26：完成 PNNL/NIST 外部定量谱表 sanity check 通用 CSV 入口，新增 quantitative_table.py、sanity_check_tabulated_spectra.py 和对应测试；全量 pytest -q 为 119 passed。
- 2026-05-26：继续推进 hitran_hapi_v1，默认 HITRAN grid 扩大为 CH4 2880-3180 cm-1、CO2 2250-2445 cm-1，并新增测试固定 grid 覆盖滤光片 center +/- FWHM。
- 2026-05-26：用真实 HAPI 1.3.0.0 在新窗口重新预计算本地 cache；precompute 输出 CH4 通道吸收约 0.8601、CO2 通道吸收约 1.4407，compare_optical_backends 对照跑通。
- 2026-05-26：完成 Phase 4.2 训练基础组件最小实现，新增 dl.training loss/metrics，并验证 pytest -q 为 107 passed。
- 2026-05-25：完成 HITRAN 主路径原型和真实下载验证，修复 HAPI 表名按气体复用导致的跨窗口缓存污染风险，并把 HITRAN cache 作为本地运行产物忽略。
- 2026-05-25：spectral defaults 中 CH4/CO2 FWHM 从 smoke 值切到 InfraTec NBP 行业参考值，目标传感器 TraceGas-HC-NDIR 实际 datasheet 仍待获取。
