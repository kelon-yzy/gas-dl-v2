# 项目级文档

本目录只放**跨场景**的项目级文档——阶段规划、跨场景方法学、阶段产出。

**场景专属文档不放这里**，在各场景模块自己的 `docs/` 下：

| 场景        | 文档位置                       | 入口                                      |
| --------- | -------------------------- | --------------------------------------- |
| 掺氢天然气 hg  | `hydrogen_ng/docs/`        | `AI_CONTEXT_GUIDE.md`、`ARCHITECTURE.md` |
| 合成气 sg    | `syngas/docs/`             | `README.md`                             |
| 掘进通风 tv3  | `tunnel_ventilation/docs/` | `掘进通风代码契约事实源.md`（硬约束）、`掘进通风实验日志.md`（经验） |
| 学长算法 RCDW | `rcdw_mgda/`               | 该子工程内                                   |

> 场景隔离重构（见根目录 `场景隔离重构计划.md`）之前，`docs/ARCHITECTURE.md` 与 `docs/AI_CONTEXT_GUIDE.md` 在仓库根的 `docs/`。它们现在属于 hg 场景，已迁到 `hydrogen_ng/docs/`。根 `CLAUDE.md` 里仍按旧路径写，尚未同步。

---

## 当前主线

长期研究边界见 [多模态气体检测通用融合算法项目指导方向](../general_fusion/多模态气体检测通用融合算法_项目指导方向.md)，算法执行唯一以 [项目总体规划](../general_fusion/项目总体规划.md)（v8，2026-08-29）为准。两份文档与新主线代码同在 `general_fusion/` 子工程内，各阶段契约与评审记录在 `general_fusion/docs/algorithm/`。

```text
A0  统一任务、数据与算法接口契约              ✅ 2026-08-27
 │
A1  Ar-He-CO₂ 仿真 benchmark v1             ✅ 2026-08-28
 │
A2  完整传感器配置下的通用融合核心             负结果关闭 2026-08-28
 │
A2H 高难度仿真与分布外泛化实验                负结果关闭 2026-08-28
 │
A2M 主流深度学习架构对照与强基线收口           MLP_RETAINED 2026-08-29
 │
A3  xylene-e-nose 外部数据集验证              ← 当前阶段
 │
A4  可变传感器集合与可靠性扩展
 │
A5  小规模真实 Ar-He-CO₂ 混合气验证
```

当前位置为 **A3**。A2M 冻结 `A2M-MLP / mlp_lbfgs_width32` 为完整输入参考，时序对照矩阵冻结在 `general_fusion/configs/experiment/a3_temporal_matrix.json`。论文写作 A6 从 A2H 冻结结论后并行推进。

旧 P0–P3 已结束，不再充当当前主线的前置执行阶段。历史入口保留为：[P1 正式关闭审查](p1/09_P1补充检索与正式关闭审查.md)、[P2 状态](p2/README.md)、[P3 评审记录](p3/P3评审记录.md)；旧 P4 未授权且不再恢复。
