# Active 文档

本目录只放正在实施、等待正式验收或直接影响下一步决策的专项方案。

| 优先级 | 文档 | 当前状态 |
| --- | --- | --- |
| ▶️ | [tv3_mrs_ei_mei4_execution_plan.md](tv3_mrs_ei_mei4_execution_plan.md) | **MEI-4 执行计划（规划中）**：C0 契约冻结是唯一允许的第一步；C0--C2 无需新授权，C3 起观测空间抽样需 MEI-4 范围独立授权 |
| 已冻结 | [tv3_mrs_ei_mei3_execution_plan.md](tv3_mrs_ei_mei3_execution_plan.md) | **MEI-3 执行记录**：B5 已冻结 `mei3_full_parameter_baseline_retained`；后续 MEI-4 确定性基线固定为 S1 |
| ▶️ | [tv3_mrs_information_efficient_inversion_experiment_plan.md](tv3_mrs_information_efficient_inversion_experiment_plan.md) | **MRS-EI 上位主线**：B0--B5 已完成；MEI-4 须先建立独立版本化契约。波形 / benchmark / 硬件仍禁止 |
| 操作依据 | [tv3_mrs_ei_versioned_refreeze_execution_guide.md](tv3_mrs_ei_versioned_refreeze_execution_guide.md) | **保留在 active 的可复现入口**：当前配置和冻结脚本仍按该路径引用，不作当前状态汇总 |
| 并行 | [tv3_static_air_feasibility_implementation_plan.md](tv3_static_air_feasibility_implementation_plan.md) | 静止空气 `flow=0` 扰动与 holdout；不阻塞 MRS-EI，不作现场声明 |
| ⏸ | [tv3_bidirectional_ultrasound_implementation_plan.md](tv3_bidirectional_ultrasound_implementation_plan.md) | **暂缓**：F4=`coarse_monitoring_only`；F5-wide=`f5_model_protocol_failed`；窄域 F5 / F6 不排期。MEI-0 未恢复 F 线，后续恢复需另行决策 |
| ⏸ | [tv3_comsol_multiphysics_dl_implementation_plan.md](tv3_comsol_multiphysics_dl_implementation_plan.md) | **暂缓**：G1 `g1_cfd_smoke_passed` 已冻结；G2 及以后不排期。MEI-0 未恢复 G 线，后续恢复需另行决策 |

已完成、已证伪或已冻结结论的专项计划见 [archive/completed](../archive/completed/)（含 B7 协议、D2b/RawDSP、SPXY、R5/R5-T、identifiability v1、组分宽域 F5-wide、EC-MSW 全线、F5 性能优化等）。

全局执行顺序、阈值和停止条件见[项目记忆库](../掘进通风项目记忆库.md)。任务完成或证伪后应移入 `../archive/completed/`。
