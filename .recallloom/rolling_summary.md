<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [codex] | 2026-06-15 -->
<!-- file-state: revision=61 | updated-at=2026-06-15T18:18:48+08:00 | writer-id=codex | base-workspace-revision=119 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮和 ML 多窗口实验均已完成归档；ML 主线仍是 full+exposure+recovery 的 `ridge_multiwindow_all_modalities`。
- PhaseWindowTCN MVP 与 `phase_window_tcn_improvement` 服务器实验均已完成归档；`gas_head` 已显著修复闭包问题，但 `free_component_mse` 未能让 `test/extrapolation x_N2 R2` 转正。
- 当前已新增 `docs/PhaseWindowTCN结构消融实验方案.md` 作为 PhaseWindowTCN 后续工作的唯一活跃方案，并将 ILR/ALR、多窗口 ML 计划、PhasePreservingTCN 等旧方案移入 `docs/整理归档/`。
- 当前已新增 `docs/AI_CONTEXT_GUIDE.md`，其内容是可直接复制给网页版 AI 的自包含项目上下文，说明项目目标、关键结果、当前瓶颈、结构消融方案和希望外部 AI 审查的问题。
- 当前已新增 `configs/experiment/phase_window_tcn_ablation/`：首批配置 `phase_window_tcn_ablation.json` 包含同 seed 的 `phase_window_tcn_gas_free` 基线、`phase_window_tcn_gas_free_split` 和 `phase_window_tcn_gas_free_deep`；followup 配置只包含 `phase_window_tcn_gas_free_split_deep`。
- README、ARCHITECTURE、IMPLEMENTATION_PLAN 和整理归档 README 已补充最新导读入口与当前实验状态，避免后续 AI 误把历史文档当成当前主线。

<!-- section: active_judgments -->
- PhaseWindowTCN 仍不能作为正式 DL 主线；正式 ML 主线继续保持 `ridge_multiwindow_all_modalities`。
- `gas_head` 是正确输出头方向，应保留为后续 PhaseWindowTCN 默认 head。
- `free_component_mse` 没有把 `N2` 拉起来，当前不能再把它视为足以单独过线的主改进。
- 当前 PhaseWindowTCN 的主要瓶颈已从闭包/重复监督转移到窗口表征、感受野和相位融合能力不足。
- 外部 AI 审查应基于三个事实：ML 多窗口 Ridge 已强通过，`gas_head` 已修复闭包但 N2 没改善，当前 DL 主要疑点是窗口表征和相位融合。

<!-- section: risks_open_questions -->
- `share_window_encoder=true` 可能仍在稀释 `full/exposure/recovery` 的相位差异，这个结构假设还未被验证。
- 当前 3-block TCN 可能无法充分表达对 `N2` 有用的长程或跨窗口结构信息。
- 第一批结构消融必须相对同 seed 的 `phase_window_tcn_gas_free` 判断，不能把不同 seed 或旧归档结果直接混作同批对照。
- 如果 split/deep 都无效，应停止 DL 主线继续扩展，不应直接上完整 PAF-Net、DCT 频率解耦或复杂 attention。
- 工作区仍有文档、测试和 `configs/experiment/phase_window_tcn_ablation/` 未提交改动，后续提交需避免误纳入无关输出或 RecallLoom 状态文件。

<!-- section: next_step -->
- 下一步运行或交给外部 AI 审查 `docs/AI_CONTEXT_GUIDE.md` 中的自包含问题说明，重点判断 split encoder 与 deep TCN 是否是当前最小充分消融。
- 若继续本地实验，先 dry-run 并运行 `configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation.json`，只在 split 或 deep 有正向 N2 增益后再运行 followup。
- 结果判定优先看 `test x_N2 R2`、`extrapolation x_N2 R2`、`macro RMSE`、`sum_abs_error` 和其他三组分退化幅度。
- 若第一批消融无效，收束 DL 主线，把 `phase_window_tcn_improvement` 和 ablation 作为 DL 负结果证据，正式结果继续引用 ML 多窗口主线。

<!-- section: recent_pivots -->
- 2026-06-15：新增面向网页版 AI 的自包含 `docs/AI_CONTEXT_GUIDE.md`，不再假设外部 AI 能访问本地文件。
- 2026-06-15：新增并修复 PhaseWindowTCN ablation 配置；首批配置包含同 seed 基线、split、deep，followup 单独运行 split_deep；相关 dry-run 与目标测试通过 `5 passed`。
- 2026-06-15：更新 README、ARCHITECTURE、IMPLEMENTATION_PLAN 和整理归档 README，使其他 AI 优先读取最新导读与活跃实验方案。
- 2026-06-15：完成 `phase_window_tcn_improvement` 服务器实验与归档；验证 `gas_head` 修复闭包有效，但 `free_component_mse` 未能改善 `N2`。
- 2026-06-12：完成 multiwindow_n2 服务器结果分析与归档；full+exposure+recovery 多窗口拼接强通过，test `N2 R2=0.7121`，extrapolation `N2 R2=0.7247`。
