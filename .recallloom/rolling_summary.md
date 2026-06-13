<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-12 -->
<!-- file-state: revision=58 | updated-at=2026-06-12T16:12:01+08:00 | writer-id=Codex | base-workspace-revision=115 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮和 ML 多窗口实验均已完成归档；ML 主线仍是 full+exposure+recovery 的 ridge_multiwindow_all_modalities。
- PhaseWindowTCN MVP 服务器实验已完成并归档，run 为 phase_window_tcn_full_exp_rec_raw4。
- PhaseWindowTCN MVP 真实使用 full、exposure、recovery 三窗口和 raw4 输出，但 test N2 R2=-0.0150，extrapolation N2 R2=0.0028，未通过验收。
- 本次归档包含实验配置、summary、report、完整 run 产物、训练日志、checkpoint、best checkpoint 和人工分析报告。

<!-- section: active_judgments -->
- PhaseWindowTCN MVP 不成立，不能作为正式 DL 主线。
- 失败不是单纯 N2 未提升，而是整体组成回归弱：test overall R2=0.2635，CO2 R2=-0.0770，CH4 R2=0.1632。
- 正式主线继续保持 ridge_multiwindow_all_modalities。
- 短期不继续扩大 PhaseWindowTCN raw4 调参；若继续 DL，应优先验证 ILR/ALR target 或 softmax100/gas_head 等组成约束输出。

<!-- section: risks_open_questions -->
- PhaseWindowTCN 多窗口输入通路已跑通，但当前模型和训练目标无法复现 ML 多窗口 ridge 的 N2 收益。
- raw4 输出的 sum abs error 在 test split 为 11.1797，组成和约束问题仍明显。
- 本地工作区仍有既有输出目录删除和若干无关未跟踪文件，需避免误纳入后续提交。

<!-- section: next_step -->
- 将 phase_window_tcn_mvp 作为失败的 DL MVP 证据引用，后续报告强调 ML 多窗口主线。
- 如继续 DL，先做 ILR/ALR target 或 softmax100/gas_head 对照，再考虑窗口独立 encoder 或 cross-window attention。
- 保持 ridge_multiwindow_all_modalities 作为正式 ML phase-aware 主线。

<!-- section: recent_pivots -->
- 2026-06-12：完成 PhaseWindowTCN MVP 服务器结果分析与归档；test N2 R2=-0.0150，extrapolation N2 R2=0.0028，判定未通过。
- 2026-06-12：完成 PhaseWindowTCN MVP 实现，新增 DL phase_windows 契约、phase_window_tcn 模型、正式配置和集成测试；全量测试 279 passed。
- 2026-06-12：完成 multiwindow_n2 服务器结果分析与归档；full+exposure+recovery 多窗口拼接强通过，test N2 R2=0.7121，extrapolation N2 R2=0.7247。
- 2026-06-12：完成 phase-aware N2 服务器结果分析与归档；ML recovery 和 early 0.75 通过，DL 全部不通过。
- 2026-06-09：完成 N2 组成数据目标空间改造与 ALR/ILR 验收链路。
