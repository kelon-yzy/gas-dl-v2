# 掘进通风（tv3）文档中心

本目录按“当前事实、基础契约、活跃任务、复用方法、运行手册、参考资料、历史归档”分类。项目现状与下一步判断以[项目记忆库](掘进通风项目记忆库.md)为唯一入口；专项文档不重复承担全局状态汇总责任。

## 推荐阅读顺序

1. [掘进通风项目记忆库.md](掘进通风项目记忆库.md)：当前有效事实、正式结果、硬不变量和执行门。
2. [methods/tv3_名词与实验顺序导读.md](methods/tv3_名词与实验顺序导读.md)：初学者名词说明，按实验顺序分级检索。
3. [active/tv3_static_air_feasibility_implementation_plan.md](active/tv3_static_air_feasibility_implementation_plan.md)：当前仿真 P0，`flow=0` 静止空气扰动、可辨识性与独立参数 holdout。
3b. [active/tv3_bidirectional_ultrasound_implementation_plan.md](active/tv3_bidirectional_ultrasound_implementation_plan.md)：双向超声 F 线立项规划（未启动执行）：解除 v1 flow 阻断的信息源升级，先验误差预算 + F0–F6 门。
4. [active/tv3_comsol_multiphysics_dl_implementation_plan.md](active/tv3_comsol_multiphysics_dl_implementation_plan.md)：P1 并行线：隧道 CFD / 输运 / 局部声学与 DL；G1 smoke 已通过，下一步 G2；正式 DOE 仍阻断。
5. [active/tv3_ec_msw_gatednet_implementation_plan.md](active/tv3_ec_msw_gatednet_implementation_plan.md)：算法实验线；LS 正式不晋升；下一步见 deploy-joint 计划，E2 仍禁止。
5b. [active/tv3_ec_msw_e1d_sb_deployable_joint_system_plan.md](active/tv3_ec_msw_e1d_sb_deployable_joint_system_plan.md)：e1d_sb（无 LS）可部署联合系统。
6. [active/b7_repeated_split_ood_protocol_implementation_plan.md](active/b7_repeated_split_ood_protocol_implementation_plan.md)：B7 冻结后的重复 split、独立 OOD、审计和通过门。
7. [active/README.md](active/README.md)：D2b、B6、B7、R5-T、R7、SPXY、COMSOL 多物理场等当前工作。
8. [foundation/README.md](foundation/README.md)：场景、适配、采样和物理基础。
9. [operations/README.md](operations/README.md)：服务器训练与运行操作（含波形 `waveform_preprocess` gpu/cpu 通路）。
10. [references/README.md](references/README.md)：物性、传感器和算法文献。
11. [../COMSOL/README.md](../COMSOL/README.md)：气室 P0 孪生入口；隧道输运见 `COMSOL/tunnel_transport/`。

## 目录结构

```text
docs/
├── README.md                         # 文档导航
├── 掘进通风项目记忆库.md              # 当前事实源
├── foundation/                      # 稳定背景与核心契约
├── active/                          # 正在实施或等待正式验收
├── methods/                         # 可复用方法和算法资料
├── deep_research/                   # deep research 综述、方向分诊与审计报告
├── operations/                      # 运行与服务器手册
├── references/                      # 外部资料与文献综述
└── archive/
    ├── completed/                   # 已完成、已证伪或已回填的专项计划
    └── legacy/                      # 被当前记忆库替代的旧总路线
```

## 分类索引

| 分类 | 入口 | 责任 |
| --- | --- | --- |
| 当前事实 | [掘进通风项目记忆库.md](掘进通风项目记忆库.md) | 当前状态、正式指标、结论和停止条件 |
| 基础设计 | [foundation/README.md](foundation/README.md) | 场景定义、数据与物理契约 |
| 活跃任务 | [active/README.md](active/README.md) | 待实现或待服务器验收的方案 |
| 方法资料 | [methods/README.md](methods/README.md) | 名词导读、波形、小样本和模型方法 |
| 深研报告 | [deep_research/README.md](deep_research/README.md) | 端到端 DL、算法方向与仿真链路的 deep research 综述 |
| 运行手册 | [operations/README.md](operations/README.md) | 环境、命令和服务器操作 |
| 参考资料 | [references/README.md](references/README.md) | 文献、物性和硬件资料 |
| 历史归档 | [archive/README.md](archive/README.md) | 完成记录与过时路线 |

## 文档生命周期

- 新的正式方向先进入 `active/`。
- 通过、失败或被新方案替代后移入 `archive/completed/`。
- 旧版全局路线、已被记忆库取代的综合方案移入 `archive/legacy/`。
- 稳定场景契约和物理资料留在 `foundation/`。
- 只有正式指标、契约变化或停止条件触发才更新项目记忆库。

归档不等于删除；历史文档保留原始判断，并由当前记忆库说明哪些结论已被修订。
