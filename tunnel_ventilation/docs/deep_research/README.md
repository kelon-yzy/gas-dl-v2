# Deep Research 报告索引

本目录保存 deep research 产出的综述、方向分诊与审计报告。这些报告是路线决策的输入，不改写任何 A 级正式 verdict；当前有效事实仍以[项目记忆库](../掘进通风项目记忆库.md)为准。

## 阅读入口

首选合并稿，三份原始报告按需回溯：

| 文档 | 日期 | 责任 |
| --- | --- | --- |
| [deep_research_end_to_end_dl_consolidated_20260719.md](deep_research_end_to_end_dl_consolidated_20260719.md) | 2026-07-19 | **合并稿（入口）**：五层失败分类 + 路线分诊 + teacher-student 实施方案 + G0–G6 硬门，来源间张力已显式裁决 |
| [deep_research_algorithm_ideas_20260717.md](deep_research_algorithm_ideas_20260717.md) | 2026-07-17 | 原始报告：O₂ 突破方向分诊（主线 A1/A2/A3、支线 B1–B5、暂缓/拒绝表、P0–P2） |
| [deep_research_end_to_end_dl_20260718.md](deep_research_end_to_end_dl_20260718.md) | 2026-07-18 | 原始报告：端到端 DL 失败成因五层综述 + 逐 RQ Go/No-go |
| [deep_research_end_to_end_dl_solutions_20260718.md](deep_research_end_to_end_dl_solutions_20260718.md) | 2026-07-18 | 原始报告：teacher-student 结构化蒸馏方案、SD-0~SD-6 实验矩阵、G0–G6 硬门 |
| [仿真链路与多模态融合审计_20260718.md](仿真链路与多模态融合审计_20260718.md) | 2026-07-18 | 仿真链路与真实硬件链对照审计、多模态融合契约问题 |

## 使用约束

- 证据分级统一为 A（项目正式产物）/ B（近域已发表实验）/ C（跨领域方法），跨领域数值不迁移为 tv3 预期增益。
- 三份端到端原始报告的重叠结论以合并稿裁决为准；合并稿 §1 记录了裁决依据。
- 报告中的候选方向进入实施前，仍需按 `active/` 计划文档立项并过正式门。
