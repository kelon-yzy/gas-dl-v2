<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [ClaudeCode] | 2026-06-23 -->
<!-- file-state: revision=81 | updated-at=2026-06-23T17:40:33+08:00 | writer-id=ClaudeCode | base-workspace-revision=164 -->

<!-- section: current_state -->
- 阶段状态：Stage 1.5 延长训练实验完成。extended (160ep+cosine) overall R²=+0.427 劣于 stage1 baseline (+0.493)；heavy_n2_co2 ([1,1,5,5]) 完全不可用 (R²=+0.197)。
- 最佳 test R² 仍是 stage1 baseline 的 +0.493 (CNN1DTCNFusion + raw4 + dequantized_scale5, epochs=80, 固定 LR 1.5e-4)。0.6 门槛未过。
- CO₂ 首次可学：extended 中 test CO₂ R²=+0.144 (baseline -0.050)，证明 CO₂ 并非该数据上本质不可学。
- Cosine annealing 从 epoch 1 开始衰减在当前架构上劣于固定 LR：epoch 79 时 extended val_loss=0.905 vs baseline 0.689。
- 代码侧：T_max 修复、init_weights 注释、test 补强、typo 修复已完成，328 测试通过。

<!-- section: active_judgments -->
- Cosine annealing 不适合当前问题：LR 衰减太快导致收敛速度受损，且低 LR 阶段 inverse_train_var 权重偏差 (CO₂ 高权重) 引发 CH₄ 退化。
- [1,1,5,5] 硬权重策略不可行：loss scale 放大 160×，梯度方向被 CO₂/N₂ 贡献淹没，应完全放弃。
- 延长训练不是当前瓶颈：80→160 epoch val_loss 仅改善 0.005，test R² 反而退化。瓶颈可能在 backbone 容量或数据质量。
- CO₂ 可学是重要信号：说明适当策略可以提取 CO₂ 信息。需要的是温和加权而非强制加权。
- 当前最佳路径：回退 baseline 配置 + 温和 CO₂ 权重试跑；若仍不达 0.52+，增大 backbone 容量。

<!-- section: risks_open_questions -->
- test overall R² 最高 +0.493，距 0.6 门槛仍有 ~0.1-0.15。N₂ R² 始终为负，可能本质不可学。
- CH₄ 退化风险：inverse_train_var 权重策略内嵌 CO₂ 偏好，任何延长训练方案都需监控 CH₄ R² 是否被牺牲。
- Raw4 闭包退化：sum_abs_error 从 ~7 升至 ~8.4，对累加检测场景可能引入偏差。
- 当前只跑了 n=1 per run，所有结论需 multi-seed 验证。

<!-- section: next_step -->
- 立即试跑温和 CO₂ 权重：[1, 1, 2, 1] 或 [1, 2, 3, 1]，epochs=80，固定 LR 1.5e-4，其余同 baseline。
- 若温和权重不达 overall R² ≥ 0.52：增大 backbone (acoustic_channels [32,64,128,128], tcn_channels [128,128,128])。
- 若任一方案 overall R² ≥ 0.6 且 N₂ R² > 0：multi-seed 验证 (3 seeds)。
- 保留 PhaseWindowTCN + window_attention (Stage 2) 为后备方向。

<!-- section: recent_pivots -->
- 2026-06-23：stage1_extended 完成分析。cosine annealing 退化 overall R²，heavy_n2_co2 不可用。CO₂ 首次可学。回退 baseline 配置为最佳 starting point。
- 2026-06-23：dl_feature_upgrade_stage1 完成分析。CNN1DTCNFusion raw4 overall R² +0.493 远优于 PhaseWindowTCN；gas_head 在两个 backbone 证实不可训练，方向废弃。
- 2026-06-23：n2_input_contract_ablation 完成。反量化输入优于 raw int16 证实，输入契约对照线收束。
- 2026-06-23：n2_head_sweep 完成。gas_head 闭包残差为 N2 不可学主因证实；head 参数化对照线收束。
- 2026-06-23：完成三处小修并通过目标测试。
- 2026-06-22：PyTorch 最佳实践审查 DL 代码，全量 314 测试通过。
- 2026-06-19：N2 诊断修正，主因是 gas_head 闭包残差参数化而非非线性压缩。
