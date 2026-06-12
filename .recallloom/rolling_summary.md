<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-12 -->
<!-- file-state: revision=57 | updated-at=2026-06-12T13:19:54+08:00 | writer-id=Codex | base-workspace-revision=113 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮和 ML 多窗口实验均已完成归档；ML 主线仍是 full+exposure+recovery 的 ridge_multiwindow_all_modalities。
- PhaseWindowTCN MVP 已完成代码实现：DL Dataset 支持真实多窗口输入，CLI/pipeline 接入 phase_windows，模型 phase_window_tcn 已注册。
- phase_window_tcn_mvp 正式实验配置已新增，默认验证 full+exposure+recovery 三窗口 DL 输入。
- 本机验证已完成：受影响测试 92 passed，全量测试 279 passed，phase_window_tcn_mvp dry-run 通过。
- 正式数据集上的 PhaseWindowTCN 训练尚未在本机执行。

<!-- section: active_judgments -->
- Phase-aware N2 当前结论仍为 ML 多窗口强通过；DL 是否可追近 ML 需要 phase_window_tcn_mvp 正式训练结果验证。
- PhaseWindowTCN MVP 先不引入 cross-attention、phase token 或 phase shift，优先隔离真实多窗口输入和 raw4 输出头的效果。
- phase_windows 是 DL 专属契约；ML 多窗口特征拼接继续使用 windows 字段。
- 现有单窗口 cnn1d_tcn_fusion 不再作为继续调参主线。

<!-- section: risks_open_questions -->
- multiwindow_n2 和 phase-aware 归档结果来自服务器输出导入目录；本机未复跑正式数据集。
- PhaseWindowTCN 目前只有代码测试、dry-run 和小数据 1 epoch 冒烟，尚无 test/extrapolation N2 R2。
- 本地工作区仍有既有输出目录删除和若干无关未跟踪文件，需避免误纳入后续提交。

<!-- section: next_step -->
- 在服务器或具备正式数据集和 GPU 的环境运行 phase_window_tcn_mvp。
- 训练完成后比较 phase_window_tcn_full_exp_rec_raw4 与 ridge_multiwindow_all_modalities 的 test/extrapolation N2 R2、其他组分 R2 drop 和 macro RMSE。
- 若 PhaseWindowTCN N2 仍明显低于 ML 多窗口，再考虑输出目标变换、窗口独立 encoder、或增强版 cross-window attention。

<!-- section: recent_pivots -->
- 2026-06-12：完成 PhaseWindowTCN MVP 实现，新增 DL phase_windows 契约、phase_window_tcn 模型、正式配置和集成测试；全量测试 279 passed。
- 2026-06-12：完成 multiwindow_n2 服务器结果分析与归档；full+exposure+recovery 多窗口拼接强通过，test N2 R2=0.7121，extrapolation N2 R2=0.7247。
- 2026-06-12：完成 phase-aware N2 服务器结果分析与归档；ML recovery 和 early 0.75 通过，DL 全部不通过。
- 2026-06-09：完成 N2 组成数据目标空间改造与 ALR/ILR 验收链路。
