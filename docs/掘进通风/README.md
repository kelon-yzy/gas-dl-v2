# 掘进通风气体检测文档导航

> 本目录规划掘进通风场景下的 `CO₂ / O₂ / N₂` 三组分气体检测任务。
> 该场景面向巷道通风状态感知与气体组成估计，仿真链路复用 [../dl_model_architecture.md](../dl_model_architecture.md) 第十三节，只替换组分种类、组分范围和场景命名。

## 实施状态（截至 2026-07-05）

仿真链路适配（阶段 1–3）+ DL 训练适配（阶段 4）+ formal 数据集 + 初步基线已落地。tv3-formal（600 序列）已生成，Ridge/TCN 基线已完成首轮。阶段 5 剩余：完整 15 runs 基线矩阵 + ablation（待决策方向，见 [experiment_roadmap.md](experiment_roadmap.md) 基线结果分析）。2026-07-05 审查修复后，tv3 多模态 fusion 使用 `raw3` 三输出，`gas_head` / `target_transform` 在 tv3 路径下显式拒绝。

| 阶段 | 范围 | 状态 |
|------|------|------|
| 1 | Schema + 采样 | ✅ 已完成（18 tests） |
| 2 | 声学 / 热导 / 光学物理适配 | ✅ 已完成（25 tests） |
| 3 | 慢通道 + benchmark + CLI | ✅ 已完成（18 tests） |
| 4 | DL 训练适配 | ✅ 已完成（13 tests） |
| 5 | formal 数据集 + 基线训练 + ablation | 🔶 进行中（tv3-formal 已生成 + Ridge/TCN 首轮基线完成） |

## 阅读顺序

| 顺序 | 文件 | 用途 | 适合什么时候读 |
|---:|---|---|---|
| 1 | [CO2_O2_N2_气体检测场景规划.md](CO2_O2_N2_气体检测场景规划.md) | 场景目标、组分语义、采样范围、数据契约、实施路线框架 | 了解场景全貌、确认组分定义和约束 |
| 2 | [adaptation_plan.md](adaptation_plan.md) | 主实施方案：架构决策、分阶段任务、文件清单、验证流程、风险矩阵 | 开始编码、审查整体方案、确认改动范围 |
| 3 | [sampling_design.md](sampling_design.md) | 采样方案：CO₂/O₂/N₂ 区间、联合约束、LHS 实现规范、状态分层 | 修改 `conditions.py` 或设计 `tv3` benchmark |
| 4 | [physics_references.md](physics_references.md) | 可编码物性常数速查：声速、弛豫、热导、NDIR 参数 | 修改 `acoustic_physics.py`、编码物理模型 |
| 5 | [experiment_roadmap.md](experiment_roadmap.md) | 实验路线图：benchmark 生成 → 基线 → ablation | 规划下一步实验、确认依赖和优先级 |
| 6 | [dl_training_plan.md](dl_training_plan.md) | DL 训练方案：通道可辨识性、模型选型、Loss、实验矩阵、验收标准 | 配置 DL 实验、评估 O₂ 可辨识性 |
| 7 | [references/README.md](references/README.md) | 文献报告索引和证据来源入口 | 追溯参数来源、补充文献 |

## 已确认的初始约束

1. **检测组分**：`CO₂ / O₂ / N₂`。
2. **预测目标**：显式预测 `x_CO₂`、`x_O₂`、`x_N₂`，不使用由其他组分残差补 `N₂` 的闭包预测头。
3. **场景边界**：掘进通风，不引入可燃气体或毒性气体扩展；后续若加入 CH₄、CO，应作为新阶段单独评估。
4. **命名空间**：`tv3-*` 数据集前缀，`composition_scheme = "tunnel_ventilation"`，`schema_version = "tunnel-ventilation-1"`。
5. **实现方式**：复用现有 slow、ultrasonic、fiber_mic、phase schedule、packaging、validation 链路；只做组分字段与范围的场景化适配。
6. **DL 框架**：复用 CNN1D / TCN / LSTM / PatchTST / Ridge，`weighted_component_mse` 为推荐 loss，O₂ 加权 [1.0, 2.0, 1.0]；`cnn1d_tcn_fusion` 在 tv3 下必须使用 `output_mode="raw3"`、`out_dim=3`。

## 关键源码入口（已落地）

| 入口 | 文件 | 状态 |
|------|------|------|
| Schema | `src/sim/core/tunnel_ventilation_schema.py` | ✅ |
| 采样 | `src/sim/generation/tunnel_ventilation/conditions.py` | ✅ |
| 声学 / 热导 | `src/sim/generation/tunnel_ventilation/acoustic_physics.py` | ✅ |
| 慢通道 | `src/sim/generation/tunnel_ventilation/slow.py` | ✅ |
| Benchmark 编排 | `src/sim/generation/tunnel_ventilation/benchmark.py` | ✅ |
| CLI | `src/pipeline/generate_tunnel_ventilation_benchmark.py` | ✅ |
| DL 配置 | `configs/experiment/tv3/tv3_{baseline,tcn,lstm,patchtst,ridge}.json` | ✅ |
| 训练编排 | `scripts/run_tv3_baseline.py` | ✅ |

阶段 1–4 + 编排脚本已落地，DL CLI（`python -m dl.cli --config configs/experiment/tv3/tv3_baseline.json`）可直接消费 tv3 数据集。阶段 5 的 tv3-formal 已按 600 序列规模生成；完整 15-run 基线矩阵仍需按当前决策继续执行。

## 编码前检查点

- 不把 `N₂` 当作旧合成气场景里的背景字段；在本场景中 `x_N₂` 是显式组分。
- 不把 `mixture_id` 回退或重写为 `sequence_id`。
- 不依赖 `base_condition_id`、`noise_seed_index`、`noise_seed`。
- `manifest.json` 应记录 `composition_scheme`、`schema_version`、`labels`、`slow_channels` 和采样范围。
- `sum_abs_error` 可作为评估监控项，但模型输出不做强闭包归一化。
- 任何安全阈值只作为实验标签或风险分层使用；正式报警阈值需另行按目标规范确认。
- 闭包类 Loss（`compositional_mse` / `ilr_mse` / `free_component_mse`）不可用于本场景。
- `scripts/run_tv3_baseline.py` 固定 seeds 为 `42,123,456`；DL run 非零退出码必须暴露为失败，即使已写出 `metrics.json`。

## 文件职责

### 实施与架构

- [CO2_O2_N2_气体检测场景规划.md](CO2_O2_N2_气体检测场景规划.md)：场景定义文档。组分语义、采样范围、数据契约、实施路线框架。
- [adaptation_plan.md](adaptation_plan.md)：主实施方案。后续实现、审查、验收应以这个文件为准。
- [sampling_design.md](sampling_design.md)：采样设计。CO₂/O₂/N₂ 三组分 LHS 方案、约束、状态分层、伪代码。
- [experiment_roadmap.md](experiment_roadmap.md)：实验路线图。benchmark 生成、基线训练、ablation 消融的优先级和依赖关系。
- [dl_training_plan.md](dl_training_plan.md)：DL 训练方案。通道可辨识性分析、模型选型、Loss 选择、实验矩阵、验收标准。

### 参数与文献

- [physics_references.md](physics_references.md)：可编码物性常数速查表。
- [references/co2_o2_n2_gas_properties.md](references/co2_o2_n2_gas_properties.md)：CO₂/O₂/N₂ 声学、热导、光学物性文献来源。
- [references/tunnel_ventilation_sensing_survey.md](references/tunnel_ventilation_sensing_survey.md)：掘进通风气体检测技术综述。

