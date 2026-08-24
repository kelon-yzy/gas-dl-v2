# Active 文档

本目录原则上只放正在实施、等待正式验收或直接影响下一步决策的专项方案。

**当前没有活跃执行线。** MRS-EI 线已于 2026-08-20 按 P-C 路径收尾，tv3 转入方法学论文写作，不再排期新实验。

## 收尾状态（2026-08-20）

| 项 | 值 |
| --- | --- |
| MEI-4 verdict | `mei4_closed_on_c2_evidence`，语义 `not_passed` |
| 线状态 | `mrs_ei_closed_for_publication` |
| 收尾冻结 | `outputs/runs/tv3_mrs_ei/mei4_posterior_calibration/freezes/20260820T040953067397Z_7dff9e9ef255` |
| manifest SHA256 | `fb81deffd83286bdc0b22a6d50856c0e4162b75873987cc0988320074d0ad565` |
| 收尾契约 | `configs/tv3_mrs_ei/mei4_closure_contract.json` |
| 证据核对 | 51 条论文引用数值逐条重算，全部一致 |

收尾的完整经过见[实验日志 §2.13](../掘进通风实验日志.md)，契约层面的约束见[代码契约事实源 §10.2](../掘进通风代码契约事实源.md)。

**C3 / C4 / C5 的恢复执行已进入禁令清单。** 重开需要新立项文档、对 C0 三处矛盾的显式处置和新的授权记录。

## 本目录文档的当前定位

以下四份不再是执行计划，转为**收尾证据的上下文材料**：论文写作时需要它们说明契约设计、阶段顺序与停止理由。

| 文档 | 现在的用途 |
| --- | --- |
| [tv3_mrs_ei_mei4_execution_plan.md](tv3_mrs_ei_mei4_execution_plan.md) | §0.1 机制分解、§0.2 三项契约矛盾、§10 三条路径——收尾契约的直接依据 |
| [tv3_mrs_ei_mei4_execution_progress.md](tv3_mrs_ei_mei4_execution_progress.md) | 各阶段 freeze 索引与 C3 停止现状 |
| [tv3_mrs_ei_mei4_c3_compute_optimization_plan.md](tv3_mrs_ei_mei4_c3_compute_optimization_plan.md) | 并行改造记录；说明"中止理由已消失但跑完也不改判定"这一张力 |
| [tv3_mrs_information_efficient_inversion_experiment_plan.md](tv3_mrs_information_efficient_inversion_experiment_plan.md) | MRS-EI 上位主线的问题分段与门设计 |
| [tv3_mrs_ei_versioned_refreeze_execution_guide.md](tv3_mrs_ei_versioned_refreeze_execution_guide.md) | 冻结脚本的可复现入口 |

**为什么不移入 `archive/completed/`**：`scripts/run_tv3_mei0_registry_freeze.py`（`:47`、`:63`）与 `tests/test_tunnel_ventilation_mei0_registry.py`（`:540`）在运行时真正读取 `docs/active/tv3_mrs_information_efficient_inversion_experiment_plan.md` 这个路径。按[事实源 §13.3](../掘进通风代码契约事实源.md)，这类路径必须跟随移动更新，而移动本身没有收益。四份文档因此留在原地，状态由本文件说明。

## 已移出本目录（2026-08-16）

| 文档 | 去向 | 原因 |
| --- | --- | --- |
| tv3_mrs_ei_mei3_execution_plan.md | [archive/completed](../archive/completed/tv3_mrs_ei_mei3_execution_plan.md) | B5 已关闭 MEI-3，无剩余阶段 |
| tv3_bidirectional_ultrasound_implementation_plan.md | [archive/parked](../archive/parked/tv3_bidirectional_ultrasound_implementation_plan.md) | F5-wide 失败，窄域 F5 / F6 不排期 |
| tv3_comsol_multiphysics_dl_implementation_plan.md | [archive/parked](../archive/parked/tv3_comsol_multiphysics_dl_implementation_plan.md) | G1 通过后不排期 |
| tv3_static_air_feasibility_implementation_plan.md | [archive/parked](../archive/parked/tv3_static_air_feasibility_implementation_plan.md) | 2026-07-24 立项至今无产物，标记为未启动 |

其余已完成或已证伪的专项见 [archive/completed](../archive/completed/)。

---

论文写作的入口是 [methods/tv3_论文结构与投稿方案.md](../methods/tv3_论文结构与投稿方案.md)，素材在 [methods/tv3_算法方法论文说明.md](../methods/tv3_算法方法论文说明.md)。代码层面的约束与禁令见[代码契约事实源](../掘进通风代码契约事实源.md)；实验顺序与教训见[实验日志](../掘进通风实验日志.md)。
