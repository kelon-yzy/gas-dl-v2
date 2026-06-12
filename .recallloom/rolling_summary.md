<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [unknown] | 2026-06-12 -->
<!-- file-state: revision=56 | updated-at=2026-06-12T10:17:11+08:00 | writer-id=unknown | base-workspace-revision=111 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮服务器验证已完成，实验名 phaseaware_best_models，结果来自 outputs/服务器 outputs。
- 多窗口 N2 改进实验 multiwindow_n2 已完成并归档到 outputs/archive/multiwindow_n2_20260612。
- multiwindow_n2 归档包含实验配置、summary、pipeline report、两个 ridge run 的 metrics.json/run_config.json，以及 result_analysis.md。
- ML 多窗口路线验证有效：ridge_multiwindow_all_modalities 使用 full+exposure+recovery 三窗口拼接，在 test split 将 N2 R2 从 0.2173 提升到 0.7121。
- 多窗口方案未造成 H2/CH4/CO2 退化；三者 R2 均相对 full baseline 提升。
- DL 路线当前无效：cnn1d_tcn_fusion 的 phase/early 单窗口候选均未达到 N2 gain >= 0.10。

<!-- section: active_judgments -->
- Phase-aware N2 当前结论升级为强通过：传统 ML 多窗口特征拼接显著通过全部验收。
- 正式 ML phase-aware 主线采用 ridge_multiwindow_all_modalities，即 full+exposure+recovery 多窗口特征拼接。
- ridge_all_modalities_phase_recovery 从主候选降级为历史单窗口对照。
- ridge_all_modalities_early_075 保留为保守对照。
- ridge_all_modalities_phase_exposure 只保留为诊断证据：单独 exposure N2 R2 强，但其他组分 R2 drop 超阈值。
- 当前 DL 单窗口裁剪不进入默认有效方案；后续若继续 DL，应设计 phase-preserving fusion，而不是继续裁剪单一窗口。

<!-- section: risks_open_questions -->
- 本次 multiwindow_n2 归档基于服务器输出导入目录 outputs/服务器 outputs；正式数据集未在本机 data 目录中复跑。
- DL 路线仍未学到有效 N2 信号，当前 cnn1d_tcn_fusion backbone 需要结构级调整。
- 多窗口拼接当前验证的是 ML 特征装配有效性；后续正式配置和论文引用需要同步改为 multiwindow 主线。

<!-- section: next_step -->
- 将 ridge_multiwindow_all_modalities 写入后续正式 ML phase-aware 主线。
- 更新正式配置、论文/报告结果引用和后续实验说明，引用 outputs/archive/multiwindow_n2_20260612/result_analysis.md。
- 保留 ridge_all_modalities_phase_recovery、ridge_all_modalities_early_075、ridge_all_modalities_phase_exposure 作为历史对照和诊断证据。
- 如继续推进 DL，设计包含 exposure/steady/recovery 多分支或 phase token/gating 的 phase-preserving fusion。

<!-- section: recent_pivots -->
- 2026-06-12：完成 multiwindow_n2 服务器结果分析与归档；full+exposure+recovery 多窗口拼接强通过，test N2 R2=0.7121，extrapolation N2 R2=0.7247。
- 2026-06-12：完成 phase-aware N2 服务器结果分析与归档；ML recovery 和 early 0.75 通过，DL 全部不通过。
- 2026-06-12：补齐 phase-aware 输入窗口能力、默认候选和分析报告，并提交 feat(n2): add phase-aware benchmark runs。
- 2026-06-09：完成 N2 组成数据目标空间改造与 ALR/ILR 验收链路。
