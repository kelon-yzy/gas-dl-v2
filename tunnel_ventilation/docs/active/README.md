# Active 文档

本目录只放正在实施、等待正式验收或直接影响下一步决策的专项方案。

| 优先级 | 文档                                                                                                             | 当前状态                                                                                   |
| --- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| P1  | [tv3_ec_msw_gatednet_implementation_plan.md](tv3_ec_msw_gatednet_implementation_plan.md)                       | LS 正式不晋升；`e2_allowed=false`                                                            |
| P1  | [tv3_ec_msw_structured_sequence_head_plan.md](tv3_ec_msw_structured_sequence_head_plan.md)                     | E1d-SB / attachment / LS 正式结论                                                          |
| P1  | [tv3_ec_msw_e1d_sb_deployable_joint_system_plan.md](tv3_ec_msw_e1d_sb_deployable_joint_system_plan.md)         | D1 正式 `deploy_probe_passed`；可选 D2 打包；不替换 B7                                            |
| P1  | [tv3_comsol_multiphysics_dl_implementation_plan.md](tv3_comsol_multiphysics_dl_implementation_plan.md)         | COMSOL 通风输运 → 传感器投影 → DL；G1 `g1_cfd_smoke_passed`；下一步 G2 输运；正式 DOE 仍阻断 |
| P0  | [tv3_static_air_feasibility_implementation_plan.md](tv3_static_air_feasibility_implementation_plan.md)         | 当前仿真主线：`flow=0` 扰动设计、可辨识性与独立参数 holdout；真实实验暂缓                                          |
| ▶️  | [tv3_bidirectional_ultrasound_implementation_plan.md](tv3_bidirectional_ultrasound_implementation_plan.md)     | 双向超声 F 线：F0–F4 完成（F4=`coarse_monitoring_only`）；F5 代码就绪，正式 6000 待服务器；F6 未执行                       |
| ⏸   | [tv3_identifiability_implementation_plan.md](tv3_identifiability_implementation_plan.md)                       | v1 审计已完成、`information_source_upgrade_required`；v1 verdict 不改写；双向恢复计划见上一行的 F 线立项        |
| P0  | [b7_repeated_split_ood_protocol_implementation_plan.md](b7_repeated_split_ood_protocol_implementation_plan.md) | 完整 12-split × 3 training seed 矩阵已完成，判定 **protocol_pass**                               |
| P0  | [d2b_raw_dsp_implementation_plan.md](d2b_raw_dsp_implementation_plan.md)                                       | B1/fidelity/B6 通过；B7 residual_pass，可作为默认 raw-DSP 头候选                                   |
| P0  | [r5_mlp_implementation_plan.md](r5_mlp_implementation_plan.md)                                                 | R5 已失败；R5-T 三新增 seed 稳定通过，待新 split 验证                                                  |
| P2  | [spxy_split_implementation_plan.md](spxy_split_implementation_plan.md)                                         | 通用 SPXY 已落地；B7 正式 OOD 改用 `spxy_observed_stats_v1`，oracle profile 仅作敏感性                 |

全局执行顺序、阈值和停止条件见[项目记忆库](../掘进通风项目记忆库.md)。任务完成或证伪后应移入 `../archive/completed/`。
