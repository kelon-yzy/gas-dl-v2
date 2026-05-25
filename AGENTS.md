# AGENTS.md

本目录是多模态掺氢天然气正式实验 v4 重构主线。

## 工作原则

- 中文优先；代码、命令、API 名保持原文。
- 禁止无依据的防御性编程和隐式兜底逻辑。
- 新正式主线不得把 `mixture_id` 回退或重写为 `sequence_id`。
- 新正式 benchmark 不依赖 `base_condition_id`、`noise_seed_index`、`noise_seed`。
- 文件修改使用 `functions.apply_patch`；命令执行使用 `functions.exec_command` 并显式设置 `workdir`。

## 目标边界

- `src/sim`：数据生成、物理建模、打包、质量检查。
- `src/dl`：深度学习数据读取、模型、训练、评估。
- `src/ml`：传统 ML 特征、训练、评估、融合基线。
- `src/pipeline`：CLI 编排、状态、汇总、图表、报告。
- `configs`：按 `data/model/train/eval/experiment` 拆分正式配置。
- `outputs`：只按 `runs/summary/reports/archive` 管理运行产物。

## 第一阶段约束

第一阶段先固定核心契约，不整仓复制 V3 历史代码。可复用逻辑必须在迁移时显式去除旧主键语义和临时产物命名。
