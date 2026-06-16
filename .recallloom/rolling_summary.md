<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [unknown] | 2026-06-16 -->
<!-- file-state: revision=62 | updated-at=2026-06-16T09:59:17+08:00 | writer-id=unknown | base-workspace-revision=121 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮和 ML 多窗口实验均已完成归档；ML 主线仍是 full+exposure+recovery 的 `ridge_multiwindow_all_modalities`，test N2 R2=0.7121。
- PhaseWindowTCN MVP 与 `phase_window_tcn_improvement` 服务器实验均已完成归档；`gas_head` 已显著修复闭包问题，但 `free_component_mse` 未能让 `test/extrapolation x_N2 R2` 转正。
- 当前 DL 策略已从"直接结构消融"转向"诊断优先"：先用低成本损失/监督诊断定位 N2 负 R2 的真正机制，再决定是否做结构消融。
- 新增 `docs/ML模型改进方向分析.md`：基于代码现状与文献检索，系统梳理 ML 主线改进方向（状态：参考资料，非执行计划）。
- 重写 `docs/PhaseWindowTCN结构消融实验方案.md`：改为"诊断与结构消融"，增加 Phase 0-3 三批实验框架与候选机制分析。
- 新增 `docs/PhaseWindowTCN实验执行与验收流程.md`：可执行手册，包含前置准备、运行命令、验收门槛、决策门 G0-G3、归档与文档同步。
- 当前已有 `docs/AI_CONTEXT_GUIDE.md`，其内容是可直接复制给网页版 AI 的自包含项目上下文。
- README、ARCHITECTURE、IMPLEMENTATION_PLAN 和整理归档 README 已补充最新导读入口与当前实验状态。

<!-- section: active_judgments -->
- PhaseWindowTCN 仍不能作为正式 DL 主线；正式 ML 主线继续保持 `ridge_multiwindow_all_modalities`。
- `gas_head` 是正确输出头方向，应保留为后续 PhaseWindowTCN 默认 head。
- `free_component_mse` 失败的主因候选（按证据强度排序）：① 损失尺度不平衡（大尺度 CO2 主导梯度，小尺度 N2 被忽略）> ② N2 无直接监督（gas_head 下 N2 是纯闭包残差）> ③ 早期过拟合（MVP best epoch=4）> ④ 窗口编码器共享稀释相位差异 > ⑤ TCN 感受野不足。
- DL 诊断优先：第一批做低成本损失/监督诊断（weighted loss + handcraft MLP），结果决定是否做结构消融（split/deep）。
- ML 改进序列（按投入产出比）：A 物理派生特征（ToF/声速/FFT，最高优先）> B alpha CV + PLS/KernelRidge 对照 > C 约束/闭包建模 > D 窗口与特征选择。
- 关键文献证据：① 超声声速对 H2-CH4 组成近线性强敏感（H2≈1304、CH4≈430 m/s），当前代码完全未提取 ToF/声速——最大低垂果实；② N2 是本征难测惰性气体（TDLAS 不可测、Raman 截面弱），工业上靠化学计量学间接推断；③ 光谱气体回归标准工具是 PLS/Kernel-PLS，当前主线缺这一类。
- ML 主线现状：特征全是时域统计量，波形侧无 TOF/声速/FFT/小波；模型是裸 Ridge、`alpha=1.0` 硬编、无 CV；ML 侧 `sum_abs_error` 恒为空（闭包误差未量化）。

<!-- section: risks_open_questions -->
- DL 侧两份非本会话产生的文档改动（方案修订 + 执行流程新文档）来源未确认，已如实记录但未提交；需用户确认是否纳入。
- ML 改进方向文档与 DL 诊断方案均为"待执行"，本轮无实验结果；任何新方案须对照同 seed 基线判定（ML 验收：N2 gain≥0.10、其他组分 drop≤0.05、extrapolation margin≥−0.10）。
- DL 诊断批次若 weighted loss / handcraft MLP 均无正向 N2 增益，应收束 DL 线，不应直接进入 split/deep 结构消融。
- ML 物理特征提取（ToF/声速）需要原始波形访问与信号处理能力，当前代码框架是否支持待验证。
- 是否引入 sklearn 作正式依赖（影响 PLS/KernelRidge 与约束 LS；物理特征/alpha CV/闭包后处理不受影响）待决策。
- 工作区仍有文档改动和 RecallLoom 状态文件未提交，后续提交需避免误纳入无关输出。

<!-- section: next_step -->
- ML 侧起手做 alpha CV（LOO/GCV）+ 补 ML 侧 sum_abs_error 列（纯零风险），建立正则与闭包事实基线。
- ML 侧主增益集中攻超声 ToF/声速 + FFT/小波特征，PLS/KernelRidge 与闭包后处理作对照。
- DL 侧按执行流程文档 Phase 0 完成前置（weighted_component_mse / weighted_free_component_mse loss 类、handcraft_mlp 诊断模型、配置改写、dry-run + 单测），再进 Phase 1 诊断批。
- Phase 1 诊断批结果判读优先看 `test x_N2 R2`、`extrapolation x_N2 R2`、per-component R2 分布、`best epoch` 与 train/val loss 曲线（判断过拟合）。
- 若 Phase 1 诊断显示损失/监督是主因，采用加权损失为新 DL 基线并记录结论；若显示特征学习是主因，转向特征注入混合模型或收束 DL；仅当结构问题有证据时才进 Phase 2 结构消融。
- 提交本批文档改动前需用户确认两份 DL 文档来源。

<!-- section: recent_pivots -->
- 2026-06-16：DL 策略转向"诊断优先"——复核 `free_component_mse` 代码与文献后，判定 N2 负 R2 更可能来自损失尺度与监督方式而非窗口编码器结构；重写方案文档为三批实验（诊断 > 结构消融 > 融合/对数比）+ 决策门框架。
- 2026-06-16：新增 ML 改进方向分析文档——基于代码现状与文献检索，确认超声 ToF/声速是最大低垂果实（当前完全未提取），alpha CV 与 PLS 是次优先项，log-ratio(ILR/ALR) 已判负向不重试。
- 2026-06-16：新增 PhaseWindowTCN 执行流程文档——把诊断优先方案落成 Phase 0-3 + 决策门 G0-G3 的可执行手册（前置准备、运行命令、验收门槛、决策树、归档与文档同步）。
- 2026-06-15：新增面向网页版 AI 的自包含 `docs/AI_CONTEXT_GUIDE.md`，不再假设外部 AI 能访问本地文件。
- 2026-06-15：更新 README、ARCHITECTURE、IMPLEMENTATION_PLAN 和整理归档 README，使其他 AI 优先读取最新导读与活跃实验方案。
- 2026-06-15：完成 `phase_window_tcn_improvement` 服务器实验与归档；验证 `gas_head` 修复闭包有效，但 `free_component_mse` 未能改善 `N2`。
- 2026-06-12：完成 multiwindow_n2 服务器结果分析与归档；full+exposure+recovery 多窗口拼接强通过，test `N2 R2=0.7121`，extrapolation `N2 R2=0.7247`。
