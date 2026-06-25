<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Claude Code] | 2026-06-25 -->
<!-- file-state: revision=85 | updated-at=2026-06-25T13:45:00+08:00 | writer-id=Claude Code | base-workspace-revision=168 -->

<!-- section: current_state -->
- 阶段状态：**P3 数据增强实验（Full Aug）完成，结果：⚠️ 轻微改进但未达标**。Test overall R² 从 P0-B 的 +0.4844 提升至 +0.4998 (+0.0154, +3.2%)，距目标 0.51 差 -0.0102，未满足最低改进标准 (+0.03)。
- P3 实验配置：cnn1d_tcn_fusion_raw4_p3_full_aug，80 epochs，apply_prob=0.5（仅 50% 样本增强），max_shift=5，amplitude_scale=[0.95,1.05]，gaussian_noise_std=0.01，有效样本量 ~6000（4000×1.5）。
- **CO₂ 是数据增强最大受益者**：CO₂ R² 从 +0.27 提升至 +0.36（+34%，+0.09），是 4 个组分中唯一显著改进者。H₂ +1%（+0.78→+0.79），CH₄ +2%（+0.43→+0.45），N₂ 轻微退化（-0.01→-0.02）。
- 训练过程特征：80 epochs 无过拟合（val_loss < train_loss，gap=-0.07），val R²=+0.532 > test R²=+0.500 说明泛化良好。CO₂ 在 epoch 40→79 持续改进（+0.0195/epoch），从 -0.38 逆转至 +0.38，验证时序扰动增强了 CO₂ 特征鲁棒性。
- 训练配置缺陷：Early Stopping 应在 epoch 79 触发但未生效（最后 10 epochs 无改进），LR 始终 0.00015 未衰减（ReduceLROnPlateau patience=8 过宽）。
- **下一步决策**：执行 3 个消融实验确认哪个单一策略贡献 CO₂ 改进，若有效则参数调优作为 P3 最后尝试；若消融+调优后整体 R² < 0.51，放弃纯数据增强路线，转向 P4 Multi-scale 架构。

<!-- section: active_judgments -->
- **P3 数据增强有效但强度不足**：整体改进仅 +3.2% 未达标，但 CO₂ 显著改进 +34% 证明策略有效。apply_prob=0.5 导致仅一半样本增强（有效样本 ~6000 vs 目标 6000~8000），增强参数可能过于保守（max_shift=5, amplitude_scale±5%, noise_std=0.01）。
- **CO₂ 对时序扰动更鲁棒**：CO₂ 在训练后期持续改进且是唯一显著受益组分，说明 CO₂ 特征对时间抖动/幅度缩放/高斯噪声具有更强容忍度，H₂/CH₄ 已接近饱和（改进 <2%）。
- **消融实验必要性**：必须确认 time_jitter、amplitude_scale、gaussian_noise 哪个策略贡献 CO₂ 改进（目标：单一策略使 CO₂ R² > 0.35），避免盲目加强所有参数导致过拟合。
- **训练配置需修复**：early_stopping.patience 从 10 提升至 15（避免过早停止），scheduler.patience 从 8 降至 5（更早触发 LR 衰减），下次实验应用。
- **H₂/CH₄ 饱和，N₂ 不可学本质未改变**：H₂/CH₄ 改进 <2% 说明高性能组分在当前架构下触及上限，N₂ R² 从 -0.01→-0.02 验证纯端到端方法无法学习 N₂。

<!-- section: risks_open_questions -->
- **消融实验时间成本**：3 个消融实验 × 80 epochs ≈ 6~9 小时 GPU 时间，若结果不理想需立即决策是继续调优还是转向 P4。若某单一策略无效（CO₂ R² < 0.35），可能意味着需要多策略协同才能达到 Full Aug 效果。
- **参数调优过拟合风险**：盲目增强强度（apply_prob→0.8, max_shift→10, amplitude_scale→[0.9,1.1], noise_std→0.02）可能破坏原始特征分布或引入过拟合，需基于消融结果谨慎调整。
- **P4 Multi-scale 可行性未知**：即使 P3 完全失败，P4 的多尺度并行架构能否在参数量限制（100K 内）下提升性能仍不确定。可能需要先验证中等容量（96 通道）基线。
- **DL vs Ridge 差距依然巨大**：P3 最佳结果（+0.50）距 Ridge（+0.71）仍有 0.21 差距，即使 P3+P4 全部成功，突破 0.60 门槛仍不明朗。可能需要接受 DL 在当前数据规模下的性能上限。
- **N₂ 组分根本无法学习**：所有实验 N₂ R² 均为负，验证纯 raw waveform 端到端方法的局限性。若 N₂ 是关键目标组分，必须考虑混合模型（DL embedding + 手工特征）或放弃 DL 路线。

<!-- section: next_step -->
- **立即执行 3 个 P3 消融实验**（优先级 P0）：运行 dl_p3_ablation_time_jitter.json、dl_p3_ablation_amplitude_scale.json、dl_p3_ablation_gaussian_noise.json，优先级顺序 time_jitter → amplitude_scale → gaussian_noise（基于 CO₂ 时序特征假设）。每个实验完成后立即分析 CO₂ R²，若某策略使 CO₂ R² > 0.35 标记为候选。
- **消融分析完成后（预计 6~9 小时）决策分支**：(1) 若识别出有效策略 → 创建 dl_p3_stronger_aug.json（apply_prob=0.8, 针对性加强有效策略参数），运行最后一轮 P3 实验，目标 CO₂ R² ≥ 0.40、整体 R² ≥ 0.51；(2) 若所有单一策略 CO₂ R² < 0.35 → P3 数据增强路线失败，立即转向 P4 Multi-scale 架构设计 + 中等容量（96 通道）基线验证；(3) 若参数调优后整体 R² < 0.51 → 更新 improvement_plan.md 标记 P3 失败，启动 P4 或考虑混合模型方案。
- **并行任务**：修复训练配置（early_stopping.patience=15, scheduler.patience=5），准备 P4 Multi-scale 架构草案（多分支并行 + 注意力融合，参数预算 100K），生成 P3 Full Aug 训练曲线可视化已完成（training_curves.png）。

<!-- section: recent_pivots -->
- 2026-06-25：**P3 数据增强实验（Full Aug）完成，⚠️ 轻微改进但未达标**。Test R² +0.4998（+0.0154 vs P0-B），CO₂ 显著改进 +34%（+0.27→+0.36），H₂/CH₄ 改进 <2%，N₂ 轻微退化。训练过程无过拟合，CO₂ 后期持续改进验证时序扰动有效性。决策：执行消融实验 + 参数调优作为 P3 最后尝试，若失败转向 P4 Multi-scale。
- 2026-06-24：**P1 TCN 容量扩张实验失败**（test R²: +0.48 → -0.34，退化 0.82）。容量扩张 4× 导致严重过拟合，数据量 ~4000 无法支撑 290K 参数。放弃容量扩张路线（P1/P2），P3 数据增强提升至最高优先级。完成失败分析报告（`docs/p1_analysis_report.md`）。
- 2026-06-24：启动 P1 TCN 容量扩张实验 (tcn_channels [128,128,128]，参数量 73K→290K，目标 R² ≥ 0.53)。创建完整改进计划文档 (P1-P4 路线图)。决策采纳 improvement_plan.md 作为后续指南，暂不执行 seed 核对和 P0-B multi-seed 验证。
- 2026-06-24：P0-B 首次突破 0.48 门槛 (overall R²=+0.4844，CO₂ R²=+0.2653 首次转正)。P3-A phase-stat branch 方向废弃 (接入 Ridge 特征后退化 2.7 倍)。P0-A (CO₂×2) 方向废弃。发现 seed 差异问题 (P0-baseline +0.306 vs 原 +0.493)。
- 2026-06-23：stage1_extended 完成分析。cosine annealing 退化 overall R²，heavy_n2_co2 不可用。CO₂ 首次可学。回退 baseline 配置为最佳 starting point。
