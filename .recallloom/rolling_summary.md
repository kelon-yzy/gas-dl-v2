<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-13 -->
<!-- file-state: revision=59 | updated-at=2026-06-13T17:02:31+08:00 | writer-id=Codex | base-workspace-revision=116 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮和 ML 多窗口实验均已完成归档；ML 主线仍是 full+exposure+recovery 的 ridge_multiwindow_all_modalities。
- PhaseWindowTCN MVP 服务器实验已完成并归档，run 为 phase_window_tcn_full_exp_rec_raw4。
- PhaseWindowTCN MVP 真实使用 full、exposure、recovery 三窗口和 raw4 输出，但 test N2 R2=-0.0150，extrapolation N2 R2=0.0028，未通过验收。
- `docs/phase_window_tcn_improvement_analysis.md` 已在 2026-06-13 联网资料与本地实现复核后改写；当前结论是 PhaseWindowTCN 下一步应优先验证 `gas_head + free_component_mse`，而不是继续把 ILR 作为主路线。

<!-- section: active_judgments -->
- PhaseWindowTCN MVP 不成立，不能作为正式 DL 主线。
- 失败不是单纯 N2 未提升，而是整体组成回归弱：test overall R2=0.2635，CO2 R2=-0.0770，CH4 R2=0.1632。
- 更正旧判断：N2 不是 60-95% 主组分；本项目数据规格中 CH4 是主组分，N2 范围约 0-20。
- 当前优先根因判断是 `raw4 + 4组分 MSE` 没有表达闭包不变量；`gas_head` 能修复 sum error，但必须配套只监督 H2/CH4/CO2 的自由分量 loss，避免对推导出的 N2 重复监督。
- ILR/ALR 在理论上适合 composition，但本项目 formal_full 已显示 ridge_ilr/ridge_alr 和 cnn1d_tcn_fusion_ilr 对 N2 负向；ILR 只保留为低优先级复核。
- 正式 ML 主线继续保持 ridge_multiwindow_all_modalities。

<!-- section: risks_open_questions -->
- `src/dl/training/losses.py` 当前还没有 `free_component_mse`、component weight loss 或 loss slicing；这是下一步必须实现的代码改动。
- `gradient_clip` 当前没有接入 Trainer 或 experiment config，不能作为可直接运行的配置项记录。
- `gas_head + mse` 预计能把 sum_abs_error 降到接近 0，但不保证 N2 变好；需用 `PW-GAS-4MSE` 和 `PW-GAS-FREE` 区分闭包收益与 loss 口径收益。
- 本地工作区仍有既有输出目录删除和若干无关未跟踪文件，需避免误纳入后续提交。

<!-- section: next_step -->
- 实现 `free_component_mse`：只对 pred/target 的前三个自由组分 H2/CH4/CO2 计算 MSE，并补单测验证第 4 列 N2 不影响 loss。
- 新增 PhaseWindowTCN 对照配置并先跑两组最小实验：`PW-GAS-4MSE` (`output_mode=gas_head`, `loss=mse`, `lr=1e-4`) 和 `PW-GAS-FREE` (`output_mode=gas_head`, `loss=free_component_mse`, `lr=1e-4`)。
- 只有当 `PW-GAS-FREE` 的 test/extrapolation N2 R2 转正且 macro RMSE 明显低于 MVP 时，再启动 `share_window_encoder=false` 和 5-block TCN 感受野消融。
- `PW-ILR` 仅作为低优先级负向复核，不再作为 PhaseWindowTCN 改进主路线。
- 保持 ridge_multiwindow_all_modalities 作为正式 ML phase-aware 主线。

<!-- section: recent_pivots -->
- 2026-06-13：联网复核并改写 `docs/phase_window_tcn_improvement_analysis.md`；主路线从 ILR 改为 `gas_head + free_component_mse`，并记录 ILR/ALR 为低优先级负向复核。
- 2026-06-12：完成 PhaseWindowTCN MVP 服务器结果分析与归档；test N2 R2=-0.0150，extrapolation N2 R2=0.0028，判定未通过。
- 2026-06-12：完成 PhaseWindowTCN MVP 实现，新增 DL phase_windows 契约、phase_window_tcn 模型、正式配置和集成测试；全量测试 279 passed。
- 2026-06-12：完成 multiwindow_n2 服务器结果分析与归档；full+exposure+recovery 多窗口拼接强通过，test N2 R2=0.7121，extrapolation N2 R2=0.7247。
- 2026-06-12：完成 phase-aware N2 服务器结果分析与归档；ML recovery 和 early 0.75 通过，DL 全部不通过。
- 2026-06-09：完成 N2 组成数据目标空间改造与 ALR/ILR 验收链路。
