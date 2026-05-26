<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-26 -->
<!-- file-state: revision=20 | updated-at=2026-05-26T08:32:52+08:00 | writer-id=Codex | base-workspace-revision=37 -->

<!-- section: current_state -->
- sim 核心模块完成度约 86%：core、generation、packaging、validation 均已落地，包含 LHS 采样、声程候选配置化、声学回归测试、光学交叉敏感度、本地光谱积分原型、HITRAN HAPI 适配层、单位换算和真实谱线下载验证。
- dl/data 完成度约 40%：V4BenchmarkDataset 支持慢变量、超声、光纤三模态，支持 NTC/NCT、split 加载、lazy memmap 和 scaler 归一化。
- dl/models 完成度约 25%：MODEL_REGISTRY + build_model 工厂 + CNN1DRegressor + TCNRegressor 已落地，TCN 暴露 dilations 与 receptive_field。
- dl/training 完成度约 20%：loss 注册与构造、整体回归指标、按组分回归指标已落地；trainer、checkpoint 管理和训练配置仍待实现。
- ml 仅有占位初始化文件。
- pipeline 完成度约 22%：layout、benchmark 生成 CLI、HITRAN 预计算 CLI 和 optical backend 对照 CLI 可用。
- configs 完成度约 5%：spectral defaults 已从 smoke 占位切到行业参考占位（CH4 fwhm 147 cm-1、CO2 fwhm 93 cm-1，中心保持 3030/2347 cm-1 不变），并加 filter_source 元信息记录数据出处；其余模型、训练、评估和实验配置未落地。
- experiments 目录仍无实验脚本。
- tests 共 106 个，覆盖 sim、dl、pipeline，当前全部通过。
- Phase 2.5 HITRAN 与 PNNL 光谱积分支撑已完成 HITRAN 主路径原型：默认气体、滤光片、网格配置、缓存、单位换算、真实 HAPI 下载、预计算 CLI 和经验模型对照 CLI 均已落地。
- 本地已用 hitran-api 1.3.0.0 下载 CH4、CO2、H2O 在 2960-3100 cm-1 与 2280-2410 cm-1 两个窗口的 HITRAN 谱线缓存；HITRAN cache 属于本地运行产物，不进 git。
- 当前 benchmark 仍使用 empirical_v1，并记录 optical_absorption_backend；hitran_hapi_v1 只作为显式预计算和对照入口存在。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约，不追求功能完整度。
- LHS 采样优于纯随机采样，作为默认策略。
- CNN1D 和 TCN 作为 DL 基线模型已足够支撑后续训练链路，复杂融合与 Transformer 后续扩展。
- 训练模块先建立显式、可测试的 loss/metrics 前置契约，再进入 trainer、checkpoint 和实验追踪。
- R2 在常量目标下采用显式语义：完全匹配为 1.0，不匹配为 0.0。
- 不复制 V3 代码，所有迁移必须显式适配 v4 主键语义。
- 输出契约严格使用 v4 命名，不使用 V3 旧 split 命名。
- 声程候选属于 benchmark 生成契约，应由生成 spec、CLI 和 manifest 共同记录。
- 经验吸收系数不得表述为真实标定值；如需物理支撑，应使用 HITRAN HAPI 或 PNNL、NIST 谱库积分 backend，并在 manifest 记录 optical_absorption_backend。
- HAPI raw table name 必须绑定气体和波数窗口，避免不同 NDIR 通道复用错误谱线范围。
- 真实谱线缓存属于本地运行产物，不纳入 git；代码和配置只保留可复现入口。
- 光学链路原理文档与光谱积分实施计划形成「原理 + 实施计划」分工。
- 滤光片占位现在使用 InfraTec NBP 行业参考值（CH4 LIM-262 3.3 μm/160 nm、CO2 4.26 μm/170 nm），属 industry_reference_only，正式 benchmark 前必须替换为目标传感器 TraceGas-HC-NDIR 的实际 datasheet。

<!-- section: risks_open_questions -->
- 端到端 trainer、checkpoint 管理、训练配置和实验追踪仍未实现。
- metrics 当前返回 Python float，适合 epoch 汇总；若后续需要分布式训练聚合，需要在 trainer 层另行设计。
- TCN 感受野已在模型上记录，但具体通道数、kernel_size、层数仍未进入正式模型配置和实验对照。
- 时间步分布仍为固定四等分，未动态调整。
- configs 仍缺模型、训练、评估和实验正式配置。
- 目标传感器 TraceGas-HC-NDIR（深圳市痕量气体传感科技有限公司）datasheet 仍未到手，公开渠道找不到具体型号滤光片规格；当前 CH4 fwhm 147 cm-1 / CO2 fwhm 93 cm-1 是 InfraTec NBP 行业参考值占位，非厂商实际值。
- 新 FWHM 占位扩大后，HITRAN grid（CH4 ±70、CO2 ±63 cm-1）已窄于 filter 主响应（±FWHM 即 ±147 / ±93 cm-1），高斯滤光片在 grid 边缘会被截断；若要更高保真，需要扩大 hitran_grids 并重新下载 HITRAN cache。
- HITRAN 单位换算已完成并跑通真实 HAPI 输出，但尚未用仪器、PNNL 或 NIST 数据做数值 sanity check。
- PNNL 与 NIST 对照尚未实施，缺少谱表 parser 和外部 sanity check。
- benchmark 主线尚未支持显式切换到 spectral backend。

<!-- section: next_step -->
继续 Phase 4.2：实现最小 Trainer 与 train/eval epoch 循环，接入 V4BenchmarkDataset、build_model、build_loss、regression_metrics，并补 checkpoint 保存与最小 smoke 测试；外部并行项仍是获取 TraceGas-HC-NDIR 实际 datasheet 后替换 spectral defaults。

<!-- section: recent_pivots -->
- 2026-05-26：完成 Phase 4.2 训练基础组件最小实现，新增 src/dl/training 的 loss registry、build_loss、RegressionMetrics、regression_metrics 和 component_regression_metrics，并在 src/dl/__init__.py 暴露入口。
- 2026-05-26：新增 tests/test_dl_training.py 覆盖 loss 构造、整体和按组分指标、常量目标 R2 与错误形状显式失败；验证 pytest -q 为 106 passed。
- 2026-05-25：完成 Phase 3.1 TCNRegressor，保留 V3 因果卷积残差块思路，但改为 v4 BaseRegressor、默认 in_channels 为 8、out_dim 为 4，并记录 receptive_field。
- 2026-05-25：完成 HITRAN 主路径原型和真实下载验证，修复 HAPI 表名按气体复用导致的跨窗口缓存污染风险，并把 HITRAN cache 作为本地运行产物忽略。
- 2026-05-25：spectral defaults 中 CH4/CO2 FWHM 从 smoke 值（30/24 cm-1）切到 InfraTec NBP 行业参考值（147/93 cm-1）作为占位，中心波数 3030/2347 保持不变，spectral-defaults.json 增加 filter_source 元信息记录数据出处；目标传感器 TraceGas-HC-NDIR 实际 datasheet 仍待获取。
