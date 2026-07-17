# 掘进通风（tv3）文档中心

本目录按“当前事实、基础契约、活跃任务、复用方法、运行手册、参考资料、历史归档”分类。项目现状与下一步判断以[项目记忆库](掘进通风项目记忆库.md)为唯一入口；专项文档不重复承担全局状态汇总责任。

## 推荐阅读顺序

1. [掘进通风项目记忆库.md](掘进通风项目记忆库.md)：当前有效事实、正式结果、硬不变量和执行门。
2. [methods/tv3_名词与实验顺序导读.md](methods/tv3_名词与实验顺序导读.md)：初学者名词说明，按实验顺序分级检索。
3. [active/tv3_static_air_feasibility_implementation_plan.md](active/tv3_static_air_feasibility_implementation_plan.md)：当前 P0，静止空气范围、测量链校准与独立 holdout。
4. [active/tv3_ec_msw_gatednet_implementation_plan.md](active/tv3_ec_msw_gatednet_implementation_plan.md)：P1 算法实验线；attachment 过门，E2s-LS 消融正式待跑，E2 仍禁止。
5. [active/b7_repeated_split_ood_protocol_implementation_plan.md](active/b7_repeated_split_ood_protocol_implementation_plan.md)：B7 冻结后的重复 split、独立 OOD、审计和通过门。
6. [active/README.md](active/README.md)：D2b、B6、B7、R5-T、R7、SPXY 等当前工作。
7. [foundation/README.md](foundation/README.md)：场景、适配、采样和物理基础。
8. [operations/README.md](operations/README.md)：服务器训练与运行操作（含波形 `waveform_preprocess` gpu/cpu 通路）。
9. [references/README.md](references/README.md)：物性、传感器和算法文献。

## 目录结构

```text
docs/
├── README.md                         # 文档导航
├── 掘进通风项目记忆库.md              # 当前事实源
├── foundation/                      # 稳定背景与核心契约
├── active/                          # 正在实施或等待正式验收
├── methods/                         # 可复用方法和算法资料
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
