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
