<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [unknown] | 2026-06-15 -->
<!-- file-state: revision=60 | updated-at=2026-06-15T14:33:04+08:00 | writer-id=unknown | base-workspace-revision=118 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮和 ML 多窗口实验均已完成归档；ML 主线仍是 full+exposure+recovery 的 `ridge_multiwindow_all_modalities`。
- PhaseWindowTCN MVP 服务器实验已完成归档，run 为 `phase_window_tcn_full_exp_rec_raw4`。
- `phase_window_tcn_improvement` 服务器实验已完成并归档到 `outputs/archive/phase_window_tcn_improvement_20260615`。
- 本次实验完成了两个最小对照：`phase_window_tcn_gas_4mse` 与 `phase_window_tcn_gas_free`。
- `gas_head` 输出头已显著修复闭包问题，`sum_abs_error` 从 `raw4` MVP 的 `11.1797` 降到约 `2e-6`。
- `phase_window_tcn_gas_free` 的 overall 指标优于 `phase_window_tcn_gas_4mse`，但 `test/extrapolation x_N2 R2` 仍未转正。

<!-- section: active_judgments -->
- PhaseWindowTCN 仍不能作为正式 DL 主线。
- `gas_head` 是正确输出头方向，应保留为后续 PhaseWindowTCN 默认 head。
- `free_component_mse` 没有把 `N2` 拉起来，当前不能再把它视为足以单独过线的主改进。
- 当前 PhaseWindowTCN 的主要瓶颈已从“闭包/重复监督”转移到“窗口表征与相位融合能力不足”。
- 正式 ML 主线继续保持 `ridge_multiwindow_all_modalities`。

<!-- section: risks_open_questions -->
- `share_window_encoder=true` 可能仍在稀释 `full/exposure/recovery` 的相位差异，这个结构假设还未被验证。
- 当前 3-block TCN 可能无法充分表达对 `N2` 有用的长程或跨窗口结构信息。
- `free_component_mse` 提升了 overall，但没有改善 `N2`，说明继续只调 loss 的收益可能已经有限。
- 本地工作区仍有既有输出目录删除和若干无关未跟踪文件，需避免误纳入后续提交。

<!-- section: next_step -->
- 若继续推进 PhaseWindowTCN，下一轮优先做结构消融，而不是继续围绕 `gas_head/free_component_mse` 小调参。
- 结构消融优先级：`share_window_encoder=false`，再评估更深 TCN 感受野；只有这些无效后再考虑 attention / phase token。
- 保持 `ridge_multiwindow_all_modalities` 作为正式 ML phase-aware 主线，并在报告中把 `phase_window_tcn_improvement` 作为 DL 负结果证据引用。

<!-- section: recent_pivots -->
- 2026-06-15：完成 `phase_window_tcn_improvement` 服务器实验与归档；验证 `gas_head` 修复闭包有效，但 `free_component_mse` 未能改善 `N2`。
- 2026-06-13：联网复核并改写 `docs/phase_window_tcn_improvement_analysis.md`；主路线从 ILR 改为 `gas_head + free_component_mse`。
- 2026-06-12：完成 PhaseWindowTCN MVP 服务器结果分析与归档；`raw4` 输出头失败。
- 2026-06-12：完成 multiwindow_n2 服务器结果分析与归档；full+exposure+recovery 多窗口拼接强通过，test `N2 R2=0.7121`，extrapolation `N2 R2=0.7247`。
