# general_fusion — 通用多模态气体融合主线

对应 [项目总体规划.md](项目总体规划.md) v8 的 A0–A6 新主线。方向与论证边界见
[多模态气体检测通用融合算法_项目指导方向.md](多模态气体检测通用融合算法_项目指导方向.md)。

## 证据链

1. Ar-He-CO₂ 多模态仿真数据集 — 主开发数据（`src/gf/sim/`）
2. `xylene-e-nose` 外部数据集 — 按完整工作簿分组的外部验证（适配器在 `src/gf/dl/`）
3. 小规模真实 Ar-He-CO₂ 实验 — 最终应用验证（A5，不在本骨架内实现）

## 目录与职责（总体规划 §1.3）

| 位置 | 唯一职责 |
| --- | --- |
| `src/gf/sim/` | Ar-He-CO₂ 数据生成、物理模型、打包和质量检查 |
| `src/gf/dl/` | 数据适配器、通用融合核心、任务头、训练与评估 |
| `src/gf/ml/` | Ridge、GBDT 等传统机器学习基线 |
| `src/gf/pipeline/` | CLI 编排、运行状态、汇总、图表和报告 |
| `configs/{data,model,train,eval,experiment}/` | 正式配置事实源 |
| `docs/algorithm/` | A0 后形成的算法契约与阶段记录 |
| `outputs/{runs,summary,reports,archive}/` | 运行产物（不进 git） |
| `data/` | benchmark 数据集（不进 git） |

## 不变量（自总体规划 §1.2，写代码前必读）

- 融合核心不得硬编码气体名称、组分数量、固定通道数或数据集路径分支。
- 数据集差异只允许出现在适配器（输入侧）与任务头（输出侧）。
- Ar-He-CO₂ 按 `mixture_id` 分组切分；xylene 按完整工作簿分组切分。
- scaler 只由训练组拟合。
- 跨数据集分别训练只声称架构复用，不声称零样本迁移。

## 当前状态

A0 已于 2026-08-27 通过评审，A1 已于 2026-08-28 通过评审并冻结 formal v1。当前已具备统一样本与批次契约、
Ar-He-CO₂ 和 xylene 两个适配器、无数据集分支的共享 `FusionCore`、任务头、
分组与 scaler 测试，以及 A1 数据生成、信息审计、基线矩阵和报告编排。

A0 的执行顺序、阶段门和完成证据见 [A0 分步执行计划](docs/algorithm/00_A0分步执行计划.md)。
冻结契约见 [统一任务与接口契约](docs/algorithm/01_统一任务与接口契约.md) 和
[评价协议与基线矩阵](docs/algorithm/02_评价协议与基线矩阵.md)，最终结论见
[A0 评审记录](docs/algorithm/03_A0评审记录.md)。A1 的轻量执行顺序、数据规模和停止条件见
[A1 分步执行计划](docs/algorithm/04_A1分步执行计划.md)。A1 执行证据见
[A1 数据与物理规格](docs/algorithm/05_A1数据与物理规格.md) 和
[A1 评审记录](docs/algorithm/06_A1评审记录.md)。A2 已完成 A2-0 至 A2-4 的 train / val 与 grouped OOF 诊断，结果为负：Deep Sets 与 residual 备选均未通过预注册门，A2-5 formal test 未进入且保持锁定。详见
[A2 评审记录](docs/algorithm/07_A2评审记录.md)。

A2H 已完成 A2H-0 至 A2H-7 并以 `NEGATIVE_RESULT` 关闭：正式 v2 数据的 calibration、environment、joint、noise 四个困难轴通过资格审计，composition 未通过；C1、M1 未通过候选晋级门，B5 被冻结为主强基线。hard-test 按冻结证据一次解锁且未用于调参。详见
[A2H 分步执行计划](docs/algorithm/09_A2H分步执行计划.md)、[A2H 正式报告](outputs/reports/a2h_v2/A2H正式报告.md) 和
[A2H 评审记录](outputs/reports/a2h_v2/A2H评审记录.md)。

A2M 已完成 A2M-0 至 A2M-6，并以 `MLP_RETAINED` 关闭：A1 历史 B5 明确区分为 `B5-SK`，当前运行时建立 `A2M-MLP` 新参考；独立 formal holdout 在 `FROZEN` 状态下一次解锁，RESNET 和 FTT 均未通过开发与 formal 晋级门。A3 完整输入参考冻结为 `A2M-MLP / mlp_lbfgs_width32`；TCN、GRU 和时序 Transformer 矩阵仅由 A3 的真实时间维协议执行。详见
[A2M 主流架构对照分步执行计划](docs/algorithm/10_A2M主流架构对照分步执行计划.md)、[A2M 评审记录](docs/algorithm/11_A2M评审记录.md) 和 [A3 时序架构矩阵](configs/experiment/a3_temporal_matrix.json)。

当前下一阶段为 A3 外部数据集验证。

```powershell
python -m pytest -q
```
