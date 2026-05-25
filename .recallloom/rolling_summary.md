<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-25 -->
<!-- file-state: revision=17 | updated-at=2026-05-25T15:14:16+08:00 | writer-id=Codex | base-workspace-revision=27 -->

<!-- section: current_state -->
- sim 核心模块完成度约 82%：core、generation、packaging、validation 均已落地，包含 LHS 采样、声程候选配置化、声学回归测试、光学交叉敏感度、本地光谱积分原型和 HITRAN HAPI 适配层
- dl/data 完成度约 40%：V4BenchmarkDataset 支持慢变量、超声、光纤三模态，支持 NTC/NCT、split 加载、lazy memmap 和 scaler 归一化
- dl/models 完成度约 25%：MODEL_REGISTRY + build_model 工厂 + CNN1DRegressor + TCNRegressor 已落地，TCN 暴露 dilations 与 receptive_field
- dl/training 未开始：loss、metrics、trainer、checkpoint 管理均待实现
- ml 仅有占位初始化文件
- pipeline 完成度约 15%：layout + generate_benchmark CLI 可用，path_lms 已支持逗号分隔声程候选
- configs 目录仍无实际配置，experiments 目录仍无实验脚本
- tests 共 87 个，覆盖 sim、dl、pipeline，当前全部通过
- LHS 采样已落地，H₂ 双峰分布保留，策略写入 manifest
- 声程候选已迁入 BenchmarkGenerationSpec.path_lms，正式默认值为 (0.20, 0.25, 0.30, 0.35, 0.40)，并写入 manifest 与 waveform metadata
- Phase 2.3 声学物理确定性回归测试已完成
- Phase 2.4 光谱交叉敏感度显式建模已完成
- HITRAN/PNNL 光谱积分资料调研、实施方案和 NDIR 光学链路原理文档已完成
- 本地光谱积分核心、tabulated_spectrum_v1 backend、hitran_hapi_v1 适配层和谱线缓存已新增；当前 benchmark 仍使用 empirical_v1，并记录 optical_absorption_backend

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约，不追求功能完整度
- LHS 采样优于纯随机采样，作为默认策略
- CNN1D 和 TCN 作为 DL 基线模型已足够支撑后续训练链路，复杂融合与 Transformer 后续扩展
- 不复制 V3 代码，所有迁移必须显式适配 v4 主键语义
- 输出契约严格使用 v4 命名，不使用 V3 旧 split 命名
- 声程候选属于 benchmark 生成契约，应由生成 spec、CLI 和 manifest 共同记录
- 经验吸收系数不得表述为真实标定值；如需物理支撑，应使用 HITRAN HAPI 或 PNNL/NIST 谱库积分 backend，并在 manifest 记录 optical_absorption_backend
- hitran_hapi_v1 当前是可测试适配层，真实下载仍需要安装 HAPI、配置数据库目录、补充真实滤光片响应和谱源版本管理
- 光学链路原理文档与光谱积分实施计划形成「原理 + 实施计划」分工

<!-- section: risks_open_questions -->
- DL training 模块未开始，端到端训练链路未建立
- TCN 感受野已在模型上记录，但具体通道数、kernel_size、层数仍未进入正式模型配置和实验对照
- 时间步分布仍为固定四等分，未动态调整
- configs 目录仍为空，无实际 Hydra 配置
- 无 checkpoint 管理和实验追踪机制
- 滤光片中心波数和 FWHM 尚未做项目级正式配置
- hitran_hapi_v1 缺单位换算层：HAPI 在 HITRAN_units=True 下返回的谱系数当前被直接当作 absorption_coeff_per_percent_m 使用
- 真实 HITRAN/PNNL 数据尚未集成，谱表缓存目录、T/P 网格策略和滤光片响应配置仍需决策
- PNNL/NIST 对照尚未实施，缺少光谱积分 vs 经验模型的 sanity check 脚本

<!-- section: next_step -->
继续 Phase 3 时，下一步是实现 LSTMRegressor / GRURegressor 并注册到 MODEL_REGISTRY；若优先补训练主线，则进入 Phase 4.2 的 loss 与 metrics 最小实现。

<!-- section: recent_pivots -->
- 2026-05-25：回到实验主线完成 Phase 3.1 TCNRegressor，保留 V3 因果卷积残差块思路，但改为 v4 BaseRegressor、默认 in_channels=8/out_dim=4，并记录 receptive_field。
