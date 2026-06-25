<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Claude Code] | 2026-06-25 -->
<!-- file-state: revision=87 | updated-at=2026-06-25T17:00:00+08:00 | writer-id=Claude Code | base-workspace-revision=170 -->

<!-- section: current_state -->
- 阶段状态：**P3 消融实验完成，发现重大意外结果**。gaussian_noise 单独使用 Test R² = **+0.5907**（P0-P3 历史最高，突破 0.51/0.53 目标），H₂=+0.83/CH₄=+0.66 均为历史最高。但 CO₂ 毁灭（+0.27→-0.23）。time_jitter 全面退化（+0.31），amplitude_scale 完全失败（-0.64）。
- **根本发现：消融实验推翻了 Full Aug 分析的多个假设**：(1) H₂/CH₄ 并非"已饱和"——之前没找到正确的正则化方法，gaussian_noise 独用时 H₂ 从 +0.78→+0.83，CH₄ 从 +0.43→+0.66；(2) Full Aug 中 CO₂ +0.36 是三种策略的交互效应，任何单一策略无法复现；(3) amplitude_scale 是 Full Aug 中破坏性最强的策略，将 gaussian_noise 的正则化收益从 +0.59 抵消至 +0.50。
- **合成气适配（新工程分支）**：Phase A 编码准备就绪。文献检索已完成（4 份报告 + 7 项决策）。与 P3 实验并行不冲突。
- 训练过程特征：80 epochs 无过拟合（val_loss < train_loss，gap=-0.07），val R²=+0.532 > test R²=+0.500 说明泛化良好。CO₂ 在 epoch 40→79 持续改进（+0.0195/epoch），从 -0.38 逆转至 +0.38，验证时序扰动增强了 CO₂ 特征鲁棒性。
- 训练配置缺陷：Early Stopping 应在 epoch 79 触发但未生效（最后 10 epochs 无改进），LR 始终 0.00015 未衰减（ReduceLROnPlateau patience=8 过宽）。

<!-- section: active_judgments -->
- **gaussian_noise 是 P3 最大发现**：+0.5907 是 P0-P3 所有实验最佳结果，证明高斯噪声是强力正则化器的关键。但 CO₂ 在噪声下毁灭（-0.23），需要参数调优找到 CO₂ 可容忍的噪声强度。**CH₄ 是高噪声下的主要受益者**（+0.43→+0.66），幅度缩放单独使用时 CH₄ 也是受伤最重的（-1.22），说明 CH₄ 特征对信号扰动极其敏感。
- **Full Aug 的交互效应本质**：amplitude_scale"中和"了 gaussian_noise 的过强正则化，同时与其他策略叠加为 CO₂ 创造了更多样的扰动，帮助 CO₂ 学习鲁棒特征。CO₂ +0.36 是"幸存者偏差"——CH₄/H₂ 在交互中被抑制，CO₂ 反而受益。
- **调整后的策略优先级**：gaussian_noise_std 从 0.01 降至 0.005（预期保留 H₂/CH₄ 增益 + 减轻 CO₂ 破坏）；gaussian_noise + time_jitter（去 amplitude_scale）作为修正版组合；apply_prob 和 noise_std 扫描精细调优。
- **训练配置需修复**：early_stopping.patience 从 10 提升至 15，scheduler.patience 从 8 降至 5，下次实验应用。
- **N₂ 不可学本质未改变**：所有实验 N₂ R² 均为 0 或负，验证纯端到端方法无法学习 N₂。

<!-- section: risks_open_questions -->
- **gaussian_noise CO₂ 毁灭问题**：+0.5907 的历史最佳整体 R² 以 CO₂ 完全不可学（-0.23）为代价。核心问题是找到 CO₂ 可容忍的最大噪声强度，不损失 H₂/CH₄ 正则化收益。noise_std 从 0.01→0.005 是最直接尝试。
- **amplitude_scale 破坏性本质不明**：±5% 的幅值变化就导致模型完全崩溃（R²=-0.64），可能涉及 ultrasonic 通道的幅值量级关系被破坏。为何与 noise+jitter 配合时破坏性消失？需要在 Full Aug 中去掉 amplitude_scale 验证。
- **参数调优过拟合风险**：gaussian_noise 的参数扫描（noise_std/apply_prob）需要 3-4 个实验点，可能又花 6-12 小时 GPU 时间，需谨慎选择扫描范围和分辨率。
- **P4 Multi-scale 可行性未知**：即使 gaussian_noise 调优后 CO₂ 转正，整体 R² 突破 0.60 仍有难度。多尺度架构能否在 100K 内进一步提升 H₂/CH₄ 仍不确定。
- **DL vs Ridge 差距缩小但仍在**：gaussian_noise 将差距从 0.23 缩小至 0.12（+0.59 vs +0.71），但仍有明显差距。N₂ 组分根本无法学习的问题未解决。
- **合成气 Phase A 与 P3 噪声调优可并行**：编码工作不占用 GPU，两者可同时推进。

<!-- section: next_step -->
- **P3 噪声参数调优**（优先级 P0，GPU 6-12h）：先跑 `gaussian_noise_std=0.005`（减半噪声）和 `gaussian_noise_only + apply_prob=0.7`（更高增强比例），观察 CO₂ 是否转正且不损失整体 R²。若 CO₂ 仍为负→继续降 noise_std 至 0.002；若 CO₂ 转正→扫描 apply_prob 确认最优比例。
- **修正版组合实验**（优先级 P1）：`gaussian_noise + time_jitter`（去 amplitude_scale），验证是否能在保留 H₂/CH₄ 增益的同时让 CO₂ 受益（预期整体 R² 0.52-0.56）。
- **Phase A 编码**（优先级 P1，CPU 编码）：按 `docs/stateful-prancing-papert.md` 顺序修改 6 个核心文件。
- **分析报告**：消融实验分析报告已保存至 `outputs/reports/p3_ablation_analysis.md` + 数据 JSON。

<!-- section: recent_pivots -->
- 2026-06-25：**P3 消融实验结果颠覆性发现**。gaussian_noise 单独使用 R²=+0.5907（P0-P3 历史最高，突破所有目标线），但 CO₂ 毁灭（-0.23）。time_jitter 全面退化（+0.31），amplitude_scale 完全失败（-0.64）。推翻"H₂/CH₄ 饱和"和"CO₂ 是主要受益者"的 Full Aug 分析结论。新主线：gaussian_noise 参数调优，目标在保留 H₂/CH₄ 增益的同时让 CO₂ 转正。
- 2026-06-25：**合成气适配文献检索完成 + 7 项决策落地**。检测目标切换到 CO/CO₂/CH₄/H₂。LHS 两轮采样，CO NDIR 完整四气体串扰设计。Phase A 编码准备就绪。RecallLoom 侧车修复完成。
- 2026-06-25：**P3 数据增强实验（Full Aug）完成，⚠️ 轻微改进但未达标**。Test R² +0.4998，CO₂ 显著改进 +34%，H₂/CH₄ 改进 <2%。决策：执行消融实验定位 CO₂ 改进来源。
- 2026-06-24：**P1 TCN 容量扩张实验失败**（test R²: +0.48 → -0.34）。放弃容量扩张路线（P1/P2），P3 数据增强提升至最高优先级。
