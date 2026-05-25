<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Claude] | 2026-05-24 -->
<!-- file-state: revision=7 | updated-at=2026-05-24T16:39:48+08:00 | writer-id=Claude | base-workspace-revision=12 -->

<!-- section: current_state -->
- sim 核心模块完成度约 75%：core（ID 语义）、generation（LHS 采样 + 声程候选配置化 + 物理建模 + 确定性回归测试）、packaging（split + manifest + 输出契约）、validation（旧字段校验）均已落地
- dl/data 完成度约 40%：V4BenchmarkDataset 支持慢变量/超声/光纤三模态、NTC/NCT 格式、split 加载、lazy memmap、scaler 归一化
- dl/models 完成度约 10%：MODEL_REGISTRY + build_model 工厂 + CNN1DRegressor 已落地
- dl/training 未开始：loss、metrics、trainer、checkpoint 管理均待实现
- ml 仅有占位 __init__.py
- pipeline 完成度约 15%：layout + generate_benchmark CLI 可用，--path-lms 已支持逗号分隔声程候选
- configs 全部 .gitkeep，无实际配置
- experiments 仅有 .gitkeep
- tests 共 69 个（sim + dl + pipeline），全部通过
- LHS 采样已落地，H₂ 双峰分布保留，策略写入 manifest
- 声程候选已迁入 BenchmarkGenerationSpec.path_lms，正式默认值为 (0.20, 0.25, 0.30, 0.35, 0.40)，并写入 manifest.json 和 metadata/waveform_spec.json
- Phase 2.3 声学物理确定性回归测试已完成：11 个新测试覆盖 hidden_sound_speed_v2（5 cases）、hidden_attenuation_v2（3 cases）、main_sensor_features（3 cases），固定输入到固定输出，rel=1e-12

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约，不追求功能完整度
- LHS 采样优于纯随机采样，作为默认策略
- CNN1D 作为 DL 基线模型已足够，复杂模型（TCN/Transformer）后续扩展
- 不复制 V3 代码，所有迁移必须显式适配 v4 主键语义
- 输出契约严格使用 v4 命名（splits/train.csv），不使用 V3 旧命名
- 声程候选属于 benchmark 生成契约，应由 BenchmarkGenerationSpec、CLI 和 manifest 共同记录；当前正式默认声程候选为 (0.20, 0.25, 0.30, 0.35, 0.40)
- 物理模型的确定性回归测试是防止数值漂移的核心防线，新增物理参数或公式变更必须同步更新回归基线

<!-- section: risks_open_questions -->
- DL training 模块未开始，端到端训练链路未建立
- TCN 感受野较短（PLAN 问题 #2），模型层尚未开始处理
- 时间步分布仍为固定四等分（PLAN 问题 #3），未动态调整
- 光谱交叉敏感度未显式建模（PLAN 问题 #6），吸收层已有但谱线重叠未建模
- configs 目录仍为空，无实际 Hydra 配置
- 无 checkpoint 管理和实验追踪机制

<!-- section: next_step -->
进入 Phase 2.4：在 src/sim/generation/optical_crosstalk.py 中实现光谱交叉敏感度显式建模，建模 CH₄/CO₂ 吸收谱线重叠对 NDIR 传感器读数的影响

<!-- section: recent_pivots -->
