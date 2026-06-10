<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-09 -->
<!-- file-state: revision=53 | updated-at=2026-06-09T15:52:19+08:00 | writer-id=Codex | base-workspace-revision=105 -->

<!-- section: current_state -->
- N2 组成数据目标空间改造已完成工程落地：统一 composition 工具、机器学习目标变换、深度学习三维变换坐标训练、标签审计工具和 N2 增益验收工具均已实现。
- N2 增益验收工具已支持协议窗口分析：当 metrics 含 full/per_phase/early 结构时，会额外输出 per_phase 与 early 窗口的 N2 R2 增益、RMSE 变化和 Aitchison mean。
- ML/DL 评估结果已统一记录 n2_bins 与 ch4_bins conditional metrics，用于检查 N2 低/高浓度区间和 CH4 reference 区间内的误差变化。
- analyze_n2_improvement 已纳入 Conditional Bins 和 Protocol Conditional Bins 报告区块，会比较 baseline/candidate 在 full-window 与 per_phase/early 窗口内的 n2_bins/ch4_bins 分箱 N2 R2 gain 和 RMSE regression。
- analyze_n2_improvement 已支持 --output-path，Markdown 与 JSON 报告均可显式落盘，同时保留 stdout。
- run_experiment dry-run 已输出 ml_run_details 与 dl_run_details，可在服务器正式执行前直接审查每个默认 run 的 model、modalities、target_transform 和 protocol。
- 正式默认实验已包含 ALR Ridge、ILR Ridge 和 ILR fusion 三个候选方案。
- 候选方案默认使用训练集最小正值一半作为零值替换阈值。
- 运行配置和指标元数据都会记录解析后的目标变换与实际零值阈值。
- 计划文档已同步当前工程状态、协议窗口验收边界、conditional metrics 覆盖和正式运行顺序。
- 本地验证已完成：正式 dry-run、目标测试、相关实验测试和全量 pytest 均通过。

<!-- section: active_judgments -->
- 当前先隔离验证目标空间改造收益，不把瞬态感知建模混入同一轮实验。
- ILR 和 ALR 改造解决组成数据闭合约束问题，不能单独证明 N2 瞬态信息已被充分利用。
- Phase-aware N2 建模保留为下一阶段支线，需等 ALR 和 ILR 单独实验收益明确后再推进。
- N2/CH4 分箱指标是正式实验验收观测项，不在本地伪造真实收益。
- 服务器正式执行前应先用 dry-run 审查默认注入的 run 详情和 target_transform。
- 服务器正式验收报告使用 analyze_n2_improvement --output-path 显式落盘，不依赖 shell 重定向。
- 正式主线继续保持 mixture id 语义，不回退为 sequence id。
- 正式 HITRAN 生成仍保持 cache only 原则。

<!-- section: risks_open_questions -->
- 本地仍缺少正式数据集，无法验证真实 N2 收益。
- 完整收益需要在服务器或正式数据环境运行后，用 test N2 R2、macro RMSE、其他组分 R2 降幅、Aitchison mean、协议窗口 N2 增益以及 full/protocol conditional bins 判断。
- 正式配置默认启用 AMP，若在非 CUDA 环境或数值不稳定环境运行，需要显式关闭。
- 工作区仍存在与本轮无关的既有文档删除和分析输出未跟踪项，未处理。

<!-- section: next_step -->
- 在服务器正式数据环境先运行 run_experiment dry-run，确认 target_transform 详情。
- 随后运行标签审计和正式完整实验。
- 完成后运行 analyze_n2_improvement --output-path，查看主表、Protocol Windows、Conditional Bins 和 Protocol Conditional Bins。
- 如果 ALR 或 ILR 在 full 与协议窗口中有明确收益且其他组分没有明显退化，再进入瞬态感知 N2 建模。

<!-- section: recent_pivots -->
- 2026-06-09：为 analyze_n2_improvement 增加 --output-path，并通过全量测试。
- 2026-06-09：扩展 run_experiment dry-run，输出默认 run 的 target_transform 详情，并通过全量测试。
- 2026-06-09：补齐 analyze_n2_improvement Markdown 中的 Protocol Conditional Bins 报告，并通过全量测试。
- 2026-06-09：将 analyze_n2_improvement 接入 n2_bins/ch4_bins 分箱验收报告，并通过全量测试。
- 2026-06-09：补齐 ML/DL 统一的 n2_bins 与 ch4_bins conditional metrics，并通过全量测试。
- 2026-06-09：补齐 N2 结果验收中的 per_phase 与 early 协议窗口对比，并通过全量测试。
- 2026-06-09：完成 N2 组成数据目标空间改造，并通过全量测试。
- 2026-06-09：补齐运行配置中的解析后目标变换记录，确保 data driven zero floor 可复现。
- 2026-06-08：确定先单独验证 ALR 和 ILR 收益，瞬态感知 N2 建模延后。
