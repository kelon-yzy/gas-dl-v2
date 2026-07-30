# Archive 文档

归档文档用于追溯决策，不再作为当前路线入口。

## completed

已完成、已证伪或已有正式结果的专项计划：

| 文档 | 归档原因 |
| --- | --- |
| [completed/d2_tof_phasenet_implementation_plan.md](completed/d2_tof_phasenet_implementation_plan.md) | D2 正式训练失败，原实现停止 |
| [completed/r5_tabpfn_implementation_plan.md](completed/r5_tabpfn_implementation_plan.md) | 正式 6000 完成，作为非部署上限探针 |
| [completed/waveform_normalization_plan.md](completed/waveform_normalization_plan.md) | 三层归一化已实施并完成结论回填 |
| [completed/rocket_hydra_regression_implementation_plan.md](completed/rocket_hydra_regression_implementation_plan.md) | R0/R1/R5 历史路线已完成或被专项计划取代 |
| [completed/module_c_grouped_bottleneck_implementation_plan.md](completed/module_c_grouped_bottleneck_implementation_plan.md) | 模块 C P0 完整 24 条矩阵判定 `grouped_failed`，分支停止 |
| [completed/d2b_raw_dsp_mlp_implementation_plan.md](completed/d2b_raw_dsp_mlp_implementation_plan.md) | B6 单 seed + 三新增 seed 均通过；已被 B7 residual 超越，保留为 flat-MLP 对照锚点 |
| [completed/b7_oof_ridge_residual_mlp_implementation_plan.md](completed/b7_oof_ridge_residual_mlp_implementation_plan.md) | B7 residual_pass 的实现与 random-split 正式记录；后续协议由 b7_repeated_split 计划承接 |
| [completed/r5t_b6_multiseed_replication_plan.md](completed/r5t_b6_multiseed_replication_plan.md) | R5-T / B6 六次冻结复核完成，均 stable_pass |
| [completed/r7_extratrees_implementation_plan.md](completed/r7_extratrees_implementation_plan.md) | R7 正式 6000 未通过（显著训练-验证落差），失败证据保留 |
| [completed/d2b_raw_dsp_implementation_plan.md](completed/d2b_raw_dsp_implementation_plan.md) | RawDSP 前端、B1 parity、fidelity、B6/B7 正式通过；默认 RawDSP 头契约 |
| [completed/b7_repeated_split_ood_protocol_implementation_plan.md](completed/b7_repeated_split_ood_protocol_implementation_plan.md) | 完整 12-split × 3 training seed 矩阵 `protocol_pass`；B7 升级为默认头候选 |
| [completed/spxy_split_implementation_plan.md](completed/spxy_split_implementation_plan.md) | 通用 SPXY 已落地；B7 正式 OOD 用 `spxy_observed_stats_v1` |
| [completed/r5_mlp_implementation_plan.md](completed/r5_mlp_implementation_plan.md) | 默认 R5 正式失败；R5-T 目标标准化正式通过，可部署 MLP 对照 |
| [completed/tv3_identifiability_implementation_plan.md](completed/tv3_identifiability_implementation_plan.md) | v1 单向 TOF 审计完成，`information_source_upgrade_required`；后续由 F 线 / 静止空气承接 |
| [completed/tv3_composition_range_widening_plan.md](completed/tv3_composition_range_widening_plan.md) | 宽域 F0'–F4-wide 通过；F5-wide=`f5_model_protocol_failed`，不进 F6-wide |
| [completed/tv3_bidir_f5_performance_optimization_plan.md](completed/tv3_bidir_f5_performance_optimization_plan.md) | F5 模型协议代码侧并行/缓存优化已落地；墙钟加速比待服务器实测 |
| [completed/tv3_ec_msw_gatednet_implementation_plan.md](completed/tv3_ec_msw_gatednet_implementation_plan.md) | E1/E1r 失败证据、E1d/E1d-SB/attachment/LS 正式结论；`e2_allowed=false` |
| [completed/tv3_ec_msw_structured_sequence_head_plan.md](completed/tv3_ec_msw_structured_sequence_head_plan.md) | E1d-SB / attachment / LS 正式完成；LS 不晋升 |
| [completed/tv3_ec_msw_e1d_sb_deployable_joint_system_plan.md](completed/tv3_ec_msw_e1d_sb_deployable_joint_system_plan.md) | D1 正式 `deploy_probe_passed`；D2 打包可选；不替换 B7 |
| [completed/tv3_multifreq_relaxation_spectroscopy_dl_implementation_plan.md](completed/tv3_multifreq_relaxation_spectroscopy_dl_implementation_plan.md) | MRS-2 升秩但未过精度门，MRS-6 交付后原 MRS 线关闭 |
| [completed/tv3_mrs6_hardware_requirements.md](completed/tv3_mrs6_hardware_requirements.md) | MRS-6 硬件需求已交付；K4 和粗精度成本参照已被 MRS-EI 继承 |
| [completed/tv3_mrs_ei_versioned_refreeze_execution_report.md](completed/tv3_mrs_ei_versioned_refreeze_execution_report.md) | 上一轮 MEI-0/1 重冻结历史报告，后续状态已更新 |
| [completed/tv3_mrs_ei_f2_f5_disposition_execution_report.md](completed/tv3_mrs_ei_f2_f5_disposition_execution_report.md) | F2--F5 证据处置完成，MEI-1 固定 K4 并放行 MEI-3 |
| [completed/tv3_mrs_ei_mei3_phase_a_execution_report.md](completed/tv3_mrs_ei_mei3_phase_a_execution_report.md) | MEI-3 Phase A 历史报告；当前 B5 冻结与 MEI-4 准入边界见 active 执行计划 |
| [completed/tv3_mrs_ei_mei3_b4_review_and_analysis_report.md](completed/tv3_mrs_ei_mei3_b4_review_and_analysis_report.md) | B4 冻结结果的代码复核与再分析：机制归因（改善全部来自共享 max_iterations 样本）、CI 稳健性、代码问题清单与 B5 注记建议；verdict 不变 |

## legacy

已被当前记忆库和 active 专项计划替代的旧综合路线：

| 文档 | 归档原因 |
| --- | --- |
| [legacy/experiment_roadmap.md](legacy/experiment_roadmap.md) | 旧阶段路线，保留实验历史 |
| [legacy/dl_training_plan.md](legacy/dl_training_plan.md) | 旧 DL 总计划，多个优先级已变化 |
| [legacy/三组分检测深度学习新框架方案.md](legacy/三组分检测深度学习新框架方案.md) | D0–D5 历史框架，D2b 已修订 raw 路线 |
| [legacy/掘进通风_深度学习算法研究方向与文献路线.md](legacy/掘进通风_深度学习算法研究方向与文献路线.md) | 已由统一路线吸收；保留算法结论与文献追溯 |
| [legacy/tv3_掘进通风项目改进方案.md](legacy/tv3_掘进通风项目改进方案.md) | 已由统一路线吸收；保留测量系统改进与硬件路线追溯 |

当前执行路线统一查看[统一研究与实施路线](../掘进通风_统一研究与实施路线.md)；当前事实和术语查看[项目记忆库](../掘进通风项目记忆库.md)。
