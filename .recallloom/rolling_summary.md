<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Claude Code] | 2026-06-27 -->
<!-- file-state: revision=88 | updated-at=2026-06-27T14:12:00+08:00 | writer-id=Claude Code | base-workspace-revision=171 -->

<!-- section: current_state -->
- **合成气 Stage Ⅱ ablation 全部完成（27/27 runs，2026-06-27）**。三组消融在 `sg4-formal`（Ⅱ-1/Ⅱ-3）与新生成的 `sg4-formal-crosstalk`（Ⅱ-2）上展开，结果已写入 `docs/syngas/stage_ii_ablation_results.md`，相关文档（experiment_roadmap / stage_ii_ablation_plan / adaptation_plan / co_crosstalk_design / syngas/README）已同步更新。全量测试 462 passed（hg 零回归）。
- **Ⅱ-1 物理证据**：移除 V_NDIR_CO 通道，x_CO R² 从 0.954 → 0.484（TCN/Ridge 同向 0.470），损失约 50% 而非原预期的 ~0；仅保留 V_NDIR_CO+环境可恢复 0.93-0.94。结论修正：CO 主导依赖光学通道，残留 ~50% 可学性来自闭包约束 / V_TCS 弱热导差 / 可能的 CO₂ NDIR 弱串扰。
- **Ⅱ-2 鲁棒性**：3×3 CO₂↔CO 光学串扰为确定性线性变换，模型可学到逆映射，所有组分 R² 与无串扰持平（|Δ| ≤ 0.006），不出现原预期 0.01-0.05 下降。sim-to-real gap 不在串扰矩阵层面，应在 Stage Ⅲ 硬件层面验证。
- **Ⅱ-3 方法学**：weighted_component_mse(inverse_train_var) 切换至任一未加权 loss（mse/mae/huber/smooth_l1），x_CH4 R² 从 0.827 跌至 0.39-0.44（-0.4），其他组分几乎不变。逆方差加权对低浓度组分是论文核心方法学贡献。
- **掺氢天然气 P3 状态保留**：gaussian_noise 单独使用 Test R²=+0.5907（P0-P3 历史最高）但 CO₂ 毁灭（-0.23），noise_std=0.005 调优待跑。两条工程线（hg 噪声调优 + syngas Stage Ⅲ）独立推进，不互相阻塞。

<!-- section: active_judgments -->
- **syngas 阶段 Ⅱ 三大结论可写入论文**：(1) V_NDIR_CO 支配 CO 检测（跨 TCN/Ridge 一致）；(2) 线性串扰对模型可学性无影响；(3) inverse_train_var 加权让 CH₄ R² 翻倍。论文叙事重心：Ⅱ-1 + Ⅱ-3 是科学/方法学贡献，Ⅱ-2 作为 informative 负结果反衬 Stage Ⅲ 硬件验证的必要性。
- **syngas baseline 排序确认**：TCN ≈ Ridge ≈ 0.96 ≥ PatchTST/CNN1D/LSTM ≈ 0.93，慢通道+手工统计量在该规模（6000 序列）接近性能上限，深度模型边际收益有限。
- **gaussian_noise CO₂ 毁灭仍未解决（hg）**：+0.5907 整体最佳以 CO₂ -0.23 为代价。noise_std=0.005 / apply_prob 扫描尚未启动，与 syngas Stage Ⅱ 并行不冲突。
- **N₂ 不可学本质未变（hg）**：所有 P0-P3 实验 N₂ R² 均为 0 或负。

<!-- section: risks_open_questions -->
- **Ⅱ-1 残留 50% CO 可学性的物理来源未量化**：闭包约束 + V_TCS 热导 + 可能的 CO₂ NDIR 弱串扰，三因素相对贡献需要后续 ablation（B 组在 sg4-formal-crosstalk 上重复 / B 组再去 V_TCS）。
- **Ⅱ-2 与 sim-to-real 关系待落地**：仿真层面证明线性串扰不构成学习难度，但真实硬件的非线性 / 时变 / 标定漂移因素需要 Stage Ⅲ 实测数据验证。Stage Ⅲ 路线未启动。
- **gaussian_noise CO₂ 毁灭问题（hg）**：noise_std=0.005 + apply_prob 扫描待跑，6-12h GPU。
- **DL vs Ridge 差距（hg）**：缩小但仍在 0.12，超声/光纤麦克风模态尚未接入 syngas 训练，未验证多模态能否拉开 DL 与 Ridge。
- **PatchTST AMP 敏感性与 LSTM seed 不收敛**：已记录、不阻塞主结论。

<!-- section: next_step -->
- **Stage Ⅲ 真实硬件 sim-to-real 测试规划**（优先级 P0，论文工程闭环）：需要先确认硬件可用性、数据采集协议、对标传感器型号。Ⅱ-2 持平结论说明仿真已不是 gap 主因。
- **Ⅱ 后续可选 ablation**（优先级 P1）：(1) B 组在 sg4-formal-crosstalk 上重复（验证 CO₂ NDIR 弱串扰假设）；(2) B 组再去 V_TCS（量化热导贡献）；(3) 接入 ultrasonic / fiber_mic 后重做 Ⅱ-1（验证多模态恢复 CO）。
- **hg P3 噪声调优**（优先级 P1，GPU 6-12h）：先跑 `gaussian_noise_std=0.005` 和 `gaussian_noise_only + apply_prob=0.7`，观察 CO₂ 是否转正。
- **论文初稿启动**（优先级 P2）：以 Ⅰ-3 + Ⅱ 的 27 runs 数据为基础，结构对齐 IMRaD，重点突出 Ⅱ-1 物理证据 + Ⅱ-3 方法学贡献。

<!-- section: recent_pivots -->
- 2026-06-27：**syngas Stage Ⅱ ablation 全部完成（27 runs）**。三组消融结论清晰，文档全栈同步（stage_ii_ablation_results 新建 + experiment_roadmap/stage_ii_ablation_plan/adaptation_plan/co_crosstalk_design/syngas-README 联动更新）。下一步主线转向 Stage Ⅲ 硬件 sim-to-real 与论文写作。
- 2026-06-26：syngas Stage Ⅰ-3 基线训练完成（5 模型 × 3 seeds = 15 runs），TCN ≈ Ridge ≈ 0.96，CNN1D/LSTM/PatchTST ≈ 0.93。PatchTST 配置修复 + trainer AMP bf16 兼容 + 编排脚本 LSTM 退出码兼容三项配套修复落地。
- 2026-06-25：P3 消融实验结果颠覆性发现。gaussian_noise 单独使用 R²=+0.5907（P0-P3 历史最高），CO₂ 毁灭 -0.23，time_jitter +0.31，amplitude_scale -0.64。
- 2026-06-25：合成气适配文献检索完成 + 7 项决策落地。CO/CO₂/CH₄/H₂ 检测目标确认，Phase A 编码准备就绪。
- 2026-06-24：P1 TCN 容量扩张实验失败（test R²: +0.48 → -0.34）。放弃容量扩张路线（P1/P2），P3 数据增强提升至最高优先级。
