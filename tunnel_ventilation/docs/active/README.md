# Active 文档

本目录只放正在实施、等待正式验收或直接影响下一步决策的专项方案。

| 优先级 | 文档 | 当前状态 |
| --- | --- | --- |
| P0 | [d2b_raw_dsp_implementation_plan.md](d2b_raw_dsp_implementation_plan.md) | clean 6000 构建契约、Ridge parity 与帧级 fidelity 均通过；B6 三新增 seed 稳定通过 |
| P0 | [d2b_raw_dsp_mlp_implementation_plan.md](d2b_raw_dsp_mlp_implementation_plan.md) | B6 三新增 seed 稳定通过；可进入 B7 residual 对照 |
| P0 | [r5_mlp_implementation_plan.md](r5_mlp_implementation_plan.md) | R5 已失败；R5-T 三新增 seed 稳定通过，待新 split 验证 |
| P0 | [r5t_b6_multiseed_replication_plan.md](r5t_b6_multiseed_replication_plan.md) | 6 次冻结复核已完成；R5-T 与 B6 均 stable_pass |
| P0 | [b7_oof_ridge_residual_mlp_implementation_plan.md](b7_oof_ridge_residual_mlp_implementation_plan.md) | 代码与 smoke 已通过；待服务器 seed42 预检与三 seed 正式复核 |
| P0 | [r7_extratrees_implementation_plan.md](r7_extratrees_implementation_plan.md) | 正式 6000 未通过，存在明显训练-验证落差 |
| P2 | [spxy_split_implementation_plan.md](spxy_split_implementation_plan.md) | 正式数据仍用 random split，待新数据或重划分验证 |

全局执行顺序、阈值和停止条件见[项目记忆库](../掘进通风项目记忆库.md)。任务完成或证伪后应移入 `../archive/completed/`。
