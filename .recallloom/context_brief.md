<!-- recallloom:file=context_brief version=1.0 lang=zh-CN -->
<!-- file-state: revision=2 | updated-at=2026-05-24T14:13:03+08:00 | writer-id=RecallLoom-init | base-workspace-revision=2 -->

<!-- section: mission -->
# 项目使命

构建多模态掺氢天然气浓度预测系统：基于慢变量（温度、压力、流量）、超声波形、光纤麦克风波形三种模态信号，预测 H₂、CO₂、N₂、CH₄ 四组分浓度。

目标是将 V3 实验代码重构为架构清晰、主键语义统一、可复现的正式实验主线。

<!-- section: audience_stakeholders -->
# 受众与相关方

- 研究团队（直接使用者）
- 论文审稿人（产出消费者）
- 后续实验迭代者

<!-- section: current_phase -->
# 当前阶段

Phase 1 — 核心架构落地。已完成 sim（数据生成+物理建模）最小垂直切片、DL 数据加载和基础 CNN1D 模型。正在向 Phase 2（声程配置化、声学物理单元测试、LHS 采样收尾）推进。

<!-- section: scope -->
# 范围

- `src/sim`：数据生成、物理建模、打包、质量检查
- `src/dl`：深度学习数据读取、模型、训练、评估
- `src/ml`：传统 ML 特征工程、训练、评估、融合基线
- `src/pipeline`：CLI 编排、状态管理、汇总、图表、报告
- `configs`：按 data/model/train/eval/experiment 拆分正式配置
- `outputs`：按 runs/summary/reports/archive 管理运行产物

<!-- section: source_of_truth -->
# 事实来源

- `AGENTS.md`：工作原则与目标边界
- `docs/ARCHITECTURE.md`：已落地架构说明
- `docs/IMPLEMENTATION_PLAN.md`：实施计划与优先级
- `README.md`：项目入口说明
- V3 项目中的 `docs/新项目目标架构说明.md`：目标架构参考

<!-- section: core_workflow -->
# 核心工作流

1. benchmark 生成（LHS 采样 → 物理建模 → 打包 → 质量检查）
2. DL 数据加载（V4BenchmarkDataset，三模态，split 过滤，lazy memmap）
3. 模型训练（注册表 + 工厂模式，CNN1D/TCN/LSTM/Transformer）
4. 评估与报告（component_metrics, predictions, summary, report）
5. 实验编排（Hydra 配置驱动，批量运行，汇总对比）

<!-- section: boundaries -->
# 边界与约束

- `mixture_id` 是唯一配气方案 ID，是 split 和汇总的业务主键；不得回退为 `sequence_id`
- 正式新 benchmark 不依赖 `base_condition_id`、`noise_seed_index`、`noise_seed`
- 第一阶段先固定核心契约，不整仓复制 V3 历史代码
- 可复用逻辑迁移时必须显式去除旧主键语义和临时产物命名
- 正式 split 只使用 `splits/train.csv` 等新命名，不写 V3 旧命名
- 文件修改使用 `functions.apply_patch`；命令执行使用 `functions.exec_command`
