# 掘进通风气体检测文档导航

> 本目录规划掘进通风场景下的 `CO₂ / O₂ / N₂` 三组分气体检测任务。
> 该场景面向巷道通风状态感知与气体组成估计，仿真链路复用 [../dl_model_architecture.md](../dl_model_architecture.md) 第十三节，只替换组分种类、组分范围和场景命名。

## 实施状态（截至 2026-07-06）

仿真链路适配（阶段 1–3）+ DL 训练适配（阶段 4）+ formal 数据集 + 初步基线已落地。tv3-formal（600 序列）已生成，Ridge/TCN 基线已完成首轮。2026-07-06 新增固定特征回归分支第一阶段：`physics_stats + RidgeCV` 已实现，包含 `tv3/ml/rocket_features.py`、`tv3/ml/rocket_training.py`、`tv3/pipeline/run_tv3_rocket_baseline.py`、`configs/tv3_rocket_ridge.json`，并已通过 smoke 测试。2026-07-07 场景隔离重构后，tv3 成为自包含子工程（包名 `tv3`，独立 `pyproject.toml`），原 `src/` 下的 sim/dl/ml/pipeline/common 全部迁入 `tv3/` 下，`pip install -e ./tunnel_ventilation` 后以 `python -m tv3.xxx` 调用。阶段 5 现拆为两条线并行：一条继续整理 DL / baseline 结果，一条推进 rocket 物理特征与后续 MiniRocket / MultiRocket。2026-07-05 审查修复后，tv3 多模态 fusion 使用 `raw3` 三输出，`gas_head` / `target_transform` 在 tv3 路径下显式拒绝。2026-07-05 存储优化：tv3 默认 int16 + per-timestep scale + `--skip-fiber-mic`，数据集 17 GB → 3 GB（600 序列），精度损失可忽略（误差/噪声 ≈ 1%），光纤代码保留可恢复，详见 [server_training_guide.md](server_training_guide.md)。

| 阶段 | 范围 | 状态 |
|------|------|------|
| 1 | Schema + 采样 | ✅ 已完成（18 tests） |
| 2 | 声学 / 热导 / 光学物理适配 | ✅ 已完成（25 tests） |
| 3 | 慢通道 + benchmark + CLI | ✅ 已完成（18 tests） |
| 4 | DL 训练适配 | ✅ 已完成（13 tests） |
| 5 | formal 数据集 + 基线训练 + ablation | 🔶 进行中（tv3-formal 已生成 + Ridge/TCN 首轮基线完成 + rocket 阶段 A 已落地） |

## 阅读顺序

| 顺序 | 文件 | 用途 | 适合什么时候读 |
|---:|---|---|---|
| 1 | [CO2_O2_N2_气体检测场景规划.md](CO2_O2_N2_气体检测场景规划.md) | 场景目标、组分语义、采样范围、数据契约、实施路线框架 | 了解场景全貌、确认组分定义和约束 |
| 2 | [adaptation_plan.md](adaptation_plan.md) | 主实施方案：架构决策、分阶段任务、文件清单、验证流程、风险矩阵 | 开始编码、审查整体方案、确认改动范围 |
| 3 | [sampling_design.md](sampling_design.md) | 采样方案：CO₂/O₂/N₂ 区间、联合约束、LHS 实现规范、状态分层 | 修改 `conditions.py` 或设计 `tv3` benchmark |
| 4 | [physics_references.md](physics_references.md) | 可编码物性常数速查：声速、弛豫、热导、NDIR 参数 | 修改 `acoustic_physics.py`、编码物理模型 |
| 5 | [experiment_roadmap.md](experiment_roadmap.md) | 实验路线图：benchmark 生成 → 基线 → ablation | 规划下一步实验、确认依赖和优先级 |
| 6 | [dl_training_plan.md](dl_training_plan.md) | DL 训练方案：通道可辨识性、模型选型、Loss、实验矩阵、验收标准 | 配置 DL 实验、评估 O₂ 可辨识性 |
| 7 | [server_training_guide.md](server_training_guide.md) | 服务器训练操作手册：环境/生成/训练/回收完整步骤（Linux + RTX 5880 48GB） | 在服务器上执行正式训练 |
| 8 | [small_sample_dl_strategies.md](small_sample_dl_strategies.md) | 小样本 DL 训练策略：9 类策略（数据增强/正则化/轻量模型/集成/蒸馏/元学习/自监督/半监督/物理约束）+ 文献 + 优先级 | 600 序列约束下提升 DL 表现 |
| 9 | [rocket_hydra_regression_implementation_plan.md](rocket_hydra_regression_implementation_plan.md) | 固定特征回归专项方案：physics_stats / MiniRocket / MultiRocket / Hydra 的实现与验收路线 | 判断超声波形是否真实提升 O₂ / N₂ 可辨识性 |
| 10 | [波形特征提取算法评估.md](波形特征提取算法评估.md) | 现有 CNN1D 对 5000 点波形特征提取能力不足的算法评估：10 类算法排序、对比表、文献锚点、实施步骤 | 选择替代/补充 encoder 架构、判断 TOF/MultiRocket/wav2vec 适用性 |
| 11 | [波形特征提取算法代码示例.md](波形特征提取算法代码示例.md) | Top3 算法(TOF 提取/MultiRocket/wav2vec)的可落地 PyTorch 实现示例与接入方式 | 落地所选算法时参考代码骨架 |
| 12 | [references/README.md](references/README.md) | 文献报告索引和证据来源入口 | 追溯参数来源、补充文献 |

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
| Schema | `tv3/sim/core/tunnel_ventilation_schema.py` | ✅ |
| 采样 | `tv3/sim/generation/tunnel_ventilation/conditions.py` | ✅ |
| 声学 / 热导 | `tv3/sim/generation/tunnel_ventilation/acoustic_physics.py` | ✅ |
| 慢通道 | `tv3/sim/generation/tunnel_ventilation/slow.py` | ✅ |
| Benchmark 编排 | `tv3/sim/generation/tunnel_ventilation/benchmark.py` | ✅ |
| CLI | `tv3/pipeline/generate_tunnel_ventilation_benchmark.py` | ✅ |
| DL 配置 | `configs/tv3_{baseline,tcn,lstm,patchtst,ridge}.json` | ✅ |
| 训练编排 | `scripts/run_tv3_baseline.py` | ✅ |
| Rocket 物理特征 | `tv3/ml/rocket_features.py` | ✅（阶段 A） |
| Rocket 训练与评估 | `tv3/ml/rocket_training.py` | ✅（阶段 A） |
| Rocket CLI | `tv3/pipeline/run_tv3_rocket_baseline.py` | ✅（阶段 A） |
| Rocket 配置 | `configs/tv3_rocket_ridge.json` | ✅（阶段 A） |

阶段 1–4 + 编排脚本已落地，DL CLI（`python -m tv3.dl.cli --config configs/tv3_baseline.json`）可直接消费 tv3 数据集。Rocket 阶段 A 现已支持 `python -m tv3.pipeline.run_tv3_rocket_baseline --dataset-dir data/tv3-rocket-smoke --feature-set physics_stats --head ridgecv --output-dir outputs/tv3_rocket_smoke/r0`。阶段 5 的 tv3-formal 已按 600 序列规模生成；完整 15-run 基线矩阵和 rocket 正式集结果仍需继续执行。

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
- [server_training_guide.md](server_training_guide.md)：服务器训练操作手册。Linux + RTX 5880 48GB 环境下的完整执行步骤（环境/生成/训练/回收）。
- [small_sample_dl_strategies.md](small_sample_dl_strategies.md)：小样本 DL 训练策略。9 类策略（数据增强/正则化/轻量模型/集成/蒸馏/元学习/自监督/半监督/物理约束）的原理、文献、tv3 适用性、优先级。
- [波形特征提取算法评估.md](波形特征提取算法评估.md)：高维波形特征提取算法评估。现有 CNN1D 对 5000 点超声波形特征提取能力不足的诊断、10 类替代算法排序、对比表、文献锚点与实施步骤。
- [波形特征提取算法代码示例.md](波形特征提取算法代码示例.md)：Top3 算法代码示例。TOF 物理特征提取/MultiRocket 固定核分支/wav2vec 式 raw 编码器的 PyTorch 实现与接入方式。

### 参数与文献

- [physics_references.md](physics_references.md)：可编码物性常数速查表。
- [references/co2_o2_n2_gas_properties.md](references/co2_o2_n2_gas_properties.md)：CO₂/O₂/N₂ 声学、热导、光学物性文献来源。
- [references/tunnel_ventilation_sensing_survey.md](references/tunnel_ventilation_sensing_survey.md)：掘进通风气体检测技术综述。
