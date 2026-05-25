<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-25 -->
<!-- file-state: revision=18 | updated-at=2026-05-25T15:58:57+08:00 | writer-id=Codex | base-workspace-revision=32 -->

<!-- section: current_state -->
- sim 核心模块完成度约 86%：core、generation、packaging、validation 均已落地，包含 LHS 采样、声程候选配置化、声学回归测试、光学交叉敏感度、本地光谱积分原型、HITRAN HAPI 适配层、单位换算和真实谱线下载验证。
- dl/data 完成度约 40%：V4BenchmarkDataset 支持慢变量、超声、光纤三模态，支持 NTC/NCT、split 加载、lazy memmap 和 scaler 归一化。
- dl/models 完成度约 25%：MODEL_REGISTRY + build_model 工厂 + CNN1DRegressor + TCNRegressor 已落地，TCN 暴露 dilations 与 receptive_field。
- dl/training 未开始：loss、metrics、trainer、checkpoint 管理均待实现。
- ml 仅有占位初始化文件。
- pipeline 完成度约 22%：layout、benchmark 生成 CLI、HITRAN 预计算 CLI 和 optical backend 对照 CLI 可用。
- configs 完成度约 5%：已新增 spectral defaults 作为 HITRAN smoke 参考配置，其余模型、训练、评估和实验配置未落地。
- experiments 目录仍无实验脚本。
- tests 共 95 个，覆盖 sim、dl、pipeline，当前全部通过。
- Phase 2.5 HITRAN 与 PNNL 光谱积分支撑已完成 HITRAN 主路径原型：默认气体、滤光片、网格配置、缓存、单位换算、真实 HAPI 下载、预计算 CLI 和经验模型对照 CLI 均已落地。
- 本地已用 hitran-api 1.3.0.0 下载 CH4、CO2、H2O 在 2960-3100 cm-1 与 2280-2410 cm-1 两个窗口的 HITRAN 谱线缓存；HITRAN cache 属于本地运行产物，不进 git。
- 当前 benchmark 仍使用 empirical_v1，并记录 optical_absorption_backend；hitran_hapi_v1 只作为显式预计算和对照入口存在。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约，不追求功能完整度。
- LHS 采样优于纯随机采样，作为默认策略。
- CNN1D 和 TCN 作为 DL 基线模型已足够支撑后续训练链路，复杂融合与 Transformer 后续扩展。
- 不复制 V3 代码，所有迁移必须显式适配 v4 主键语义。
- 输出契约严格使用 v4 命名，不使用 V3 旧 split 命名。
- 声程候选属于 benchmark 生成契约，应由生成 spec、CLI 和 manifest 共同记录。
- 经验吸收系数不得表述为真实标定值；如需物理支撑，应使用 HITRAN HAPI 或 PNNL、NIST 谱库积分 backend，并在 manifest 记录 optical_absorption_backend。
- HAPI raw table name 必须绑定气体和波数窗口，避免不同 NDIR 通道复用错误谱线范围。
- 真实谱线缓存属于本地运行产物，不纳入 git；代码和配置只保留可复现入口。
- 光学链路原理文档与光谱积分实施计划形成「原理 + 实施计划」分工。

<!-- section: risks_open_questions -->
- DL training 模块未开始，端到端训练链路未建立。
- TCN 感受野已在模型上记录，但具体通道数、kernel_size、层数仍未进入正式模型配置和实验对照。
- 时间步分布仍为固定四等分，未动态调整。
- configs 仍缺模型、训练、评估和实验正式配置。
- 无 checkpoint 管理和实验追踪机制。
- 滤光片中心波数和 FWHM 仍是 smoke 参考值，尚未替换为目标传感器规格并记录出处。
- HITRAN 单位换算已完成并跑通真实 HAPI 输出，但尚未用仪器、PNNL 或 NIST 数据做数值 sanity check。
- PNNL 与 NIST 对照尚未实施，缺少谱表 parser 和外部 sanity check。
- benchmark 主线尚未支持显式切换到 spectral backend。

<!-- section: next_step -->
若继续光学物理支撑，下一步是获取目标传感器滤光片中心波数和 FWHM 并替换 spectral defaults，再实现 PNNL 与 NIST 谱表 parser 做外部 sanity check；若回到训练主线，则进入 Phase 4.2 的 loss 与 metrics 最小实现。

<!-- section: recent_pivots -->
- 2026-05-25：完成 Phase 3.1 TCNRegressor，保留 V3 因果卷积残差块思路，但改为 v4 BaseRegressor、默认 in_channels 为 8、out_dim 为 4，并记录 receptive_field。
- 2026-05-25：完成 HITRAN 主路径原型和真实下载验证，修复 HAPI 表名按气体复用导致的跨窗口缓存污染风险，并把 HITRAN cache 作为本地运行产物忽略。
