<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-12 -->
<!-- file-state: revision=55 | updated-at=2026-06-12T08:23:51+08:00 | writer-id=Codex | base-workspace-revision=109 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮服务器验证已完成，实验名 phaseaware_best_models，结果来自 outputs/服务器 outputs。
- 实验结果已整理并归档到 outputs/archive/phaseaware_best_models_20260612，归档包含 summary、报告、metrics/run_config，以及 DL checkpoint.pt、best_checkpoint.pt、metrics_live.jsonl。
- 正式实验报告已生成：outputs/reports/phaseaware_best_models_experiment_report.md。
- phase-aware 分析报告已生成：outputs/服务器 outputs/reports/phaseaware_best_models_phase_aware_n2.md/json，并复制到归档目录。
- ML 路线验证有效：ridge_all_modalities_phase_recovery 与 ridge_all_modalities_early_075 满足主验收。
- DL 路线当前无效：cnn1d_tcn_fusion 的 phase/early 单窗口候选均未达到 N2 gain >= 0.10。

<!-- section: active_judgments -->
- Phase-aware N2 当前结论为弱通过：仅传统 ML 路线满足验收。
- 正式 ML phase-aware 主候选采用 ridge_all_modalities_phase_recovery。
- ridge_all_modalities_early_075 作为稳健备选。
- ridge_all_modalities_phase_exposure 只保留为诊断证据：N2 R2 最高，但其他组分 R2 drop 超过阈值。
- 当前 DL 单窗口裁剪不进入默认有效方案；后续若继续 DL，应设计 phase-preserving fusion，而不是继续裁剪单一窗口。

<!-- section: risks_open_questions -->
- DL 路线仍未学到有效 N2 信号，当前 cnn1d_tcn_fusion backbone 需要结构级调整。
- Exposure-only 虽然 N2 R2 最高，但对其他组分有明显退化风险。
- 当前 phase-aware 验证的是输入窗口裁剪有效性，不等同于最终多阶段融合模型。
- 若把 recovery-only 作为 ML 主线，需要在后续正式配置和文档中明确其对输入信息的取舍。

<!-- section: next_step -->
- 将 ridge_all_modalities_phase_recovery 写入后续正式 ML phase-aware 主线。
- 保留 ridge_all_modalities_early_075 作为保守对照。
- 如继续推进 DL，设计包含 exposure/steady/recovery 多分支或 phase token/gating 的 phase-preserving fusion。
- 后续文档和论文结果应引用 outputs/reports/phaseaware_best_models_experiment_report.md 与 outputs/archive/phaseaware_best_models_20260612。

<!-- section: recent_pivots -->
- 2026-06-12：完成 phase-aware N2 服务器结果分析与归档；ML recovery 和 early 0.75 通过，DL 全部不通过。
- 2026-06-12：补齐 phase-aware 输入窗口能力、默认候选和分析报告，并提交 feat(n2): add phase-aware benchmark runs。
- 2026-06-09：完成 N2 组成数据目标空间改造与 ALR/ILR 验收链路。
