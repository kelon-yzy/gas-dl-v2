<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [ClaudeCode] | 2026-06-24 -->
<!-- file-state: revision=83 | updated-at=2026-06-24T15:54:00+08:00 | writer-id=ClaudeCode | base-workspace-revision=166 -->

<!-- section: current_state -->
- 阶段状态：**P1 TCN 容量扩张实验已启动**。P0-B (CH₄×2+CO₂×3) 首次突破 test overall R²=+0.4844，基于此配置创建 P1 实验 (tcn_channels [128,128,128])，参数量从 73K 扩张至 290K (4倍)，目标 R² ≥ 0.53。
- P0 实验完成结果：P0-baseline (+0.306)、P0-A CO₂×2 (-0.403，方向废弃)、**P0-B CH₄×2+CO₂×3 (+0.4844，CO₂ R²=+0.2653 首次转正)**。
- P3-A 方向失败：接入 Ridge 多窗口统计特征后 overall R²=-0.0851，性能退化 2.7 倍，说明"DL backbone + ML 手工特征"混合方向无效。
- **改进计划文档已创建** (`docs/improvement_plan.md`)：包含 P1-P4 完整路线图，总耗时约 7 个工作日，P1 作为最低风险首选方案。
- 代码侧：今日完成 P1 配置文件 (`configs/experiment/dl_p1_tcn_capacity.json`)、实验计划文档 (`docs/p1_tcn_capacity_plan.md`) 创建，提交 32eb503 并推送到远程仓库。

<!-- section: active_judgments -->
- **采纳 improvement_plan.md 作为后续行动指南**：按 P1→P2→P3→P4 优先级顺序执行改进实验，暂不执行原 next_step 中的 seed 核对和 P0-B multi-seed 验证，优先验证架构改进潜力。
- **P1 是最低风险容量扩张方案**：TCN 通道数翻倍 (64→128)，dropout 提升 (0.25→0.30) 防止过拟合，保持 P0-B 温和权重。若成功 (R² ≥ 0.53) 则继续 P2；若失败则转向 P4 或承认 DL 局限性。
- P0-B 是当前最佳配置：CH₄ R²=+0.4285、CO₂ R²=+0.2653、H₂ R²=+0.7758，N₂ R²=-0.0124 (仍为负但接近 0)。
- P3-A phase-stat branch 方向证伪：接入 420-d 统计特征反而退化，不再尝试混合 DL+ML 特征。

<!-- section: risks_open_questions -->
- **P1 实验尚未运行**：配置已推送到 GitHub，等待在 Linux 服务器上训练 (预计 30-40 分钟)。
- improvement_plan.md 中 P2 描述与 P1 有重叠 (都提到 tcn_channels [128,128,128])，需澄清 P2 的实际目标是感受野扩展 (增加 TCN 层数) 而非容量扩张。
- Seed 差异问题暂未核对：P0-baseline (+0.306) vs 原 stage1 baseline (+0.493) 差距 0.187，可能影响后续基线对比有效性。
- N₂ R² 在所有配置仍为负或接近 0，可能本质不可学 (Ridge 依赖显式相位窗口，DL 未学到类似特征)。

<!-- section: next_step -->
- **等待 P1 实验结果**：在 Linux 服务器上运行 `python src/pipeline/run_experiment.py --config configs/experiment/dl_p1_tcn_capacity.json`，训练完成后查看 `outputs/reports/dl_p1_tcn_capacity.md`。
- 根据 test overall R² 决策：(1) R² ≥ 0.53 → 接受 P1 为新 baseline，继续 P2 感受野扩展 (增加 TCN 层数至 4-6 层)；(2) 0.50 ≤ R² < 0.53 → multi-seed 验证稳定性；(3) R² < 0.50 → 分析失败原因，考虑 P4 Multi-scale CNN 或回退核对 seed 差异。
- 若 P1 失败：评估是继续 P2/P3 (数据增强) 还是直接跳至 P4 (Multi-scale CNN 架构改进)。

<!-- section: recent_pivots -->
- 2026-06-24：**启动 P1 TCN 容量扩张实验** (tcn_channels [128,128,128]，参数量 73K→290K，目标 R² ≥ 0.53)。创建完整改进计划文档 (P1-P4 路线图)。决策采纳 improvement_plan.md 作为后续指南，暂不执行 seed 核对和 P0-B multi-seed 验证。
- 2026-06-24：P0-B 首次突破 0.48 门槛 (overall R²=+0.4844，CO₂ R²=+0.2653 首次转正)。P3-A phase-stat branch 方向废弃 (接入 Ridge 特征后退化 2.7 倍)。P0-A (CO₂×2) 方向废弃。发现 seed 差异问题 (P0-baseline +0.306 vs 原 +0.493)。
- 2026-06-23：stage1_extended 完成分析。cosine annealing 退化 overall R²，heavy_n2_co2 不可用。CO₂ 首次可学。回退 baseline 配置为最佳 starting point。
- 2026-06-23：dl_feature_upgrade_stage1 完成分析。CNN1DTCNFusion raw4 overall R² +0.493 远优于 PhaseWindowTCN；gas_head 在两个 backbone 证实不可训练，方向废弃。
