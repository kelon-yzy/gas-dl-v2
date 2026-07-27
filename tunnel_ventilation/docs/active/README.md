# Active 文档

本目录只放正在实施、等待正式验收或直接影响下一步决策的专项方案。

| 优先级 | 文档 | 当前状态 |
| --- | --- | --- |
| ▶️ | [tv3_mrs_information_efficient_inversion_experiment_plan.md](tv3_mrs_information_efficient_inversion_experiment_plan.md) | **MRS-EI 当前主线**：MEI-0=`mei0_registry_frozen`；MEI-1=`mei1_inconclusive_forward_model`（F2–F5 未表征 + 设计落在 `delta_num` 等价带），**不放行 MEI-2**。不授权正式波形，也不恢复 MRS-3 |
| ✅ | [tv3_multifreq_relaxation_spectroscopy_dl_implementation_plan.md](tv3_multifreq_relaxation_spectroscopy_dl_implementation_plan.md) | **MRS 线（已收尾）**：MRS-2=`mrs2_rank_upgraded_p90_fail`；MRS-3 未进入；MRS-6 已交付（2026-07-25），`allowed_next_stage=MRS_line_closed` |
| ✅ | [tv3_mrs6_hardware_requirements.md](tv3_mrs6_hardware_requirements.md) | **MRS-6 硬件需求说明书（正式版）**：(σ_TOF×T) 全扫描不过 0.4 参考线（L 先验为第一约束）；full_stack（10 ns+0.1 K+L 10 μm+α 0.1%+双湿双压）可达 0.287 但属极端栈；K=4 足够。**§0：2026-07-25 起 0.4 精度要求暂缓强制、仅参考标注**，粗精度方案（obs-cfreq K4+0.5 μs+0.1 K ≈1.5/1.2 vol%）成为候选。证据 `outputs/tv3_mrs/mrs6_hardware/` |
| 并行 | [tv3_static_air_feasibility_implementation_plan.md](tv3_static_air_feasibility_implementation_plan.md) | 静止空气 `flow=0` 扰动与 holdout；不阻塞 MRS-EI，不作现场声明 |
| ⏸ | [tv3_bidirectional_ultrasound_implementation_plan.md](tv3_bidirectional_ultrasound_implementation_plan.md) | **暂缓**：F4=`coarse_monitoring_only`；F5-wide=`f5_model_protocol_failed`；窄域 F5 / F6 不排期。MEI-0 未恢复 F 线，后续恢复需另行决策 |
| ⏸ | [tv3_comsol_multiphysics_dl_implementation_plan.md](tv3_comsol_multiphysics_dl_implementation_plan.md) | **暂缓**：G1 `g1_cfd_smoke_passed` 已冻结；G2 及以后不排期。MEI-0 未恢复 G 线，后续恢复需另行决策 |

已完成、已证伪或已冻结结论的专项计划见 [archive/completed](../archive/completed/)（含 B7 协议、D2b/RawDSP、SPXY、R5/R5-T、identifiability v1、组分宽域 F5-wide、EC-MSW 全线、F5 性能优化等）。

全局执行顺序、阈值和停止条件见[项目记忆库](../掘进通风项目记忆库.md)。任务完成或证伪后应移入 `../archive/completed/`。
