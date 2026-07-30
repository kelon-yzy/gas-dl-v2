# 掘进通风（tv3）文档中心

本目录按“当前事实、基础契约、活跃任务、复用方法、运行手册、参考资料、历史归档”分类。项目现状与下一步判断以[项目记忆库](掘进通风项目记忆库.md)为唯一入口；专项文档不重复承担全局状态汇总责任。

## 推荐阅读顺序

1. [掘进通风项目记忆库.md](掘进通风项目记忆库.md)：当前有效事实、正式结果、硬不变量和执行门。
2. [methods/tv3_名词与实验顺序导读.md](methods/tv3_名词与实验顺序导读.md)：初学者名词说明，按实验顺序分级检索。
3. [active/tv3_mrs_ei_mei3_execution_plan.md](active/tv3_mrs_ei_mei3_execution_plan.md)：**已关闭的 MEI-3 执行记录**——B5 已冻结 `mei3_full_parameter_baseline_retained`，并固定后续 MEI-4 确定性基线为 S1。
4. [active/tv3_mrs_information_efficient_inversion_experiment_plan.md](active/tv3_mrs_information_efficient_inversion_experiment_plan.md)：**MRS-EI 上位主线**——固定 D0 K4，跳过 MEI-2；B0--B5 已完成。MEI-4 尚须独立登记版本化契约；波形 / benchmark / 硬件仍禁止。
5. [active/tv3_static_air_feasibility_implementation_plan.md](active/tv3_static_air_feasibility_implementation_plan.md)：并行 `flow=0` 静止空气扰动与留出检验；不作现场声明。
6. [active/tv3_bidirectional_ultrasound_implementation_plan.md](active/tv3_bidirectional_ultrasound_implementation_plan.md)：**暂缓**——F 线：F4=`coarse_monitoring_only`；F5-wide=`f5_model_protocol_failed`；窄域 F5 / F6 不排期。
7. [active/tv3_comsol_multiphysics_dl_implementation_plan.md](active/tv3_comsol_multiphysics_dl_implementation_plan.md)：**暂缓**——G1 冒烟测试已通过；G2 及后续不排期。
8. [active/README.md](active/README.md)：当前待评审、推进或暂缓的专项计划索引。
9. [archive/README.md](archive/README.md)：已完成 / 已证伪专项（B7、D2b、SPXY、R5、identifiability v1、宽域 F5-wide、EC-MSW 全线等）。
10. [foundation/README.md](foundation/README.md)：场景、适配、采样和物理基础。
11. [operations/README.md](operations/README.md)：服务器训练与运行操作（含波形 `waveform_preprocess` gpu/cpu 通路）。
12. [references/README.md](references/README.md)：物性、传感器和算法文献。
13. [../COMSOL/README.md](../COMSOL/README.md)：气室 P0 孪生入口；隧道输运见 `COMSOL/tunnel_transport/`。

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
