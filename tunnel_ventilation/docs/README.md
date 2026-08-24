# 掘进通风（tv3）文档中心

本目录有两个主入口，分工明确：

| 入口                               | 回答什么                               | 什么时候读             |
| -------------------------------- | ---------------------------------- | ----------------- |
| [掘进通风代码契约事实源.md](掘进通风代码契约事实源.md) | **能不能这么写**——字段、常量、冻结默认、禁令、报告要求     | 改代码、改配置、写报告之前     |
| [掘进通风实验日志.md](掘进通风实验日志.md)       | **怎么走到这一步、学到了什么**——时间轴、因果、教训、已封闭方向 | 接手工作、判断新方向、避免重复犯错 |

两者冲突时：契约条目以事实源为准，判断与教训以日志为准。

新人从实验日志 §1（八条核心事实）开始，再按需查事实源。

---

## 推荐阅读顺序

1. [掘进通风实验日志.md](掘进通风实验日志.md) §1 — 十分钟了解项目走到哪、结论是什么。
2. [掘进通风代码契约事实源.md](掘进通风代码契约事实源.md) §0 — 按你要动的模块跳到对应章节。
3. [active/README.md](active/README.md) — MRS-EI 收尾状态，以及因运行时路径约束继续留在 `active/` 的上下文材料。
4. [foundation/README.md](foundation/README.md) — 场景定义、采样设计与物理基础。
5. [operations/README.md](operations/README.md) — 服务器训练与运行操作。
6. [methods/README.md](methods/README.md) — 可复用的波形与小样本方法资料。
7. [references/README.md](references/README.md) — 物性、传感器与算法文献。
8. [deep_research/README.md](deep_research/README.md) — 端到端 DL 与算法方向的深研综述。
9. [archive/README.md](archive/README.md) — 已完成、已证伪、已暂缓与历史全局文档。
10. [../COMSOL/README.md](../COMSOL/README.md) — 气室 P0 孪生入口。

---

## 目录结构

```text
docs/
├── README.md                       # 本文件
├── 掘进通风代码契约事实源.md          # 硬约束：字段、常量、冻结默认、禁令、报告要求
├── 掘进通风实验日志.md               # 经验：时间轴、规律、已封闭方向、开放问题
├── foundation/                     # 场景、适配、采样与物理基础
├── active/                         # 活跃计划；当前也保留运行时依赖固定路径的收尾材料
├── methods/                        # 可复用方法与算法资料
├── deep_research/                  # deep research 综述与审计报告
├── operations/                     # 运行与服务器手册
├── references/                     # 外部文献与硬件资料
└── archive/
    ├── completed/                  # 已完成、已证伪或已关闭的专项
    ├── parked/                     # 有正式结论但不排期，重开需新立项
    └── legacy/                     # 被当前结构替代的旧全局文档
```

---

## 2026-08-16 文档结构调整

此前项目状态在六处并行维护（记忆库、统一路线、docs/README、active/README、名词导读、审查报告），出现过时条目与相互矛盾的状态注记。现调整为上述两个入口：

| 原文档                                                        | 去向                   | 内容分流                                     |
| ---------------------------------------------------------- | -------------------- | ---------------------------------------- |
| 掘进通风项目记忆库.md                                               | `archive/legacy/`    | 契约与不变量 → 事实源；结论与指标 → 实验日志                |
| 掘进通风_统一研究与实施路线.md                                          | `archive/legacy/`    | §2.1/§5.1/§5.2/§6 → 事实源；因果框架与执行顺序 → 实验日志 |
| methods/tv3_名词与实验顺序导读.md                                   | `archive/legacy/`    | 契约类名词 → 事实源；实验顺序 → 实验日志                  |
| 进度审查报告8.15.md                                              | `archive/legacy/`    | 结论 → 实验日志 §2.12；待处置项 → MEI-4 执行计划 §0.2   |
| active/tv3_mrs_ei_mei3_execution_plan.md                   | `archive/completed/` | MEI-3 已由 B5 关闭                           |
| active/tv3_bidirectional_ultrasound_implementation_plan.md | `archive/parked/`    | F5-wide 失败，窄域不排期                         |
| active/tv3_comsol_multiphysics_dl_implementation_plan.md   | `archive/parked/`    | G1 后不排期                                  |
| active/tv3_static_air_feasibility_implementation_plan.md   | `archive/parked/`    | 立项至今无产物，标记为未启动                           |

归档不等于删除。历史文档保留原始判断，每份头部注明归档原因与内容去向。

---

## 文档生命周期

- 新的正式方向先进入 `active/`。
- 产生正式 verdict 后移入 `archive/completed/`，并在[实验日志](掘进通风实验日志.md)追加一条时间轴条目（问题 / 动作 / 结果 / 教训）。
- 有正式结论但决定不排期的，移入 `archive/parked/`；重开需要新立项文档与授权，不在旧计划上续接。
- 被新结构替代的全局文档移入 `archive/legacy/`。
- 稳定的场景契约与物理资料留在 `foundation/`。

**更新触发条件**：

- 事实源：schema、契约、冻结默认、仿真常量、禁令或报告要求变化时更新。
- 实验日志：某个实验产生正式 verdict 时追加时间轴条目。
- 两者都不记：冒烟测试、未验收草稿、纯代码改动、命令与路径细节。
