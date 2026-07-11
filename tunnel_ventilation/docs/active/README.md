# Active 文档

本目录只放正在实施、等待正式验收或直接影响下一步决策的专项方案。

| 优先级 | 文档 | 当前状态 |
| --- | --- | --- |
| P0 | [b7_repeated_split_ood_protocol_implementation_plan.md](b7_repeated_split_ood_protocol_implementation_plan.md) | 代码已落地（observed profile / 派生 skip RawDSP / 协议编排）；正式 12-split 矩阵待服务器跑 |
| P0 | [d2b_raw_dsp_implementation_plan.md](d2b_raw_dsp_implementation_plan.md) | B1/fidelity/B6 通过；B7 residual_pass，可作为默认 raw-DSP 头候选 |
| P0 | [d2b_raw_dsp_mlp_implementation_plan.md](d2b_raw_dsp_mlp_implementation_plan.md) | B6 三新增 seed stable_pass；已被 B7 residual 超越 |
| P0 | [r5_mlp_implementation_plan.md](r5_mlp_implementation_plan.md) | R5 已失败；R5-T 三新增 seed 稳定通过，待新 split 验证 |
| P0 | [r5t_b6_multiseed_replication_plan.md](r5t_b6_multiseed_replication_plan.md) | 6 次冻结复核已完成；R5-T 与 B6 均 stable_pass |
| P0 | [b7_oof_ridge_residual_mlp_implementation_plan.md](b7_oof_ridge_residual_mlp_implementation_plan.md) | **residual_pass** 的实现与 random-split 正式记录；后续协议见上一行 |
| P0 | [r7_extratrees_implementation_plan.md](r7_extratrees_implementation_plan.md) | 正式 6000 未通过，存在明显训练-验证落差 |
| P2 | [spxy_split_implementation_plan.md](spxy_split_implementation_plan.md) | 通用 SPXY 已落地；B7 正式 OOD 改用 `spxy_observed_stats_v1`，oracle profile 仅作敏感性 |

全局执行顺序、阈值和停止条件见[项目记忆库](../掘进通风项目记忆库.md)。任务完成或证伪后应移入 `../archive/completed/`。
