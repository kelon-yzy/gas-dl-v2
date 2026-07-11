# Active 文档

本目录只放正在实施、等待正式验收或直接影响下一步决策的专项方案。

| 优先级 | 文档 | 当前状态 |
| --- | --- | --- |
| P0 | [d2b_raw_dsp_implementation_plan.md](d2b_raw_dsp_implementation_plan.md) | clean 6000 构建契约与 Ridge parity 通过；帧级 fidelity 数值待补齐 |
| P0 | [r5_mlp_implementation_plan.md](r5_mlp_implementation_plan.md) | R5 已失败；R5-T 正式 6000 通过，待稳定性复核 |
| P0 | [r7_extratrees_implementation_plan.md](r7_extratrees_implementation_plan.md) | 正式 6000 未通过，存在明显训练-验证落差 |
| P2 | [spxy_split_implementation_plan.md](spxy_split_implementation_plan.md) | 正式数据仍用 random split，待新数据或重划分验证 |

全局执行顺序、阈值和停止条件见[项目记忆库](../掘进通风项目记忆库.md)。任务完成或证伪后应移入 `../archive/completed/`。
