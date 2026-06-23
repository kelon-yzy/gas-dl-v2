<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [ZCode] | 2026-06-23 -->
<!-- file-state: revision=80 | updated-at=2026-06-23T15:20:00+08:00 | writer-id=ZCode | base-workspace-revision=160 -->

<!-- section: current_state -->
- 阶段状态：速度优化已关闭；DL 模型改进继续推进。dl_feature_upgrade_stage1 实验在服务器完成并分析。
- CNN1DTCNFusion + raw4 + dequantized_scale5 取得 test overall R²=+0.493（vs PhaseWindowTCN 最优 -0.305，提升 +0.798），H2 R²=+0.869，CH4 R²=+0.440，N2 R²=-0.006，CO2 R²≈0。CNN1DTCNFusion backbone 远优于 PhaseWindowTCN 已证实。
- gas_head 闭包残差参数化在 CNN1DTCNFusion 上同样完全不可训练（epoch 1 即停滞，val_loss 11 epoch 恒定 4.7846），该方向在两个 backbone 上均被证实为死路，正式废弃。
- 训练在 epoch 80 未触发早停（best epoch=79），val_loss 持续下降趋势中——更多 epoch 仍有改善空间。当前 overall R² 0.493 处于计划中段（>0.3 但 <0.6），未达"可继续投入"门槛但也未触发停止条件。

<!-- section: active_judgments -->
- 强历史 backbone 假设获得强证实：CNN1DTCNFusion 的特征提取能力显著优于 PhaseWindowTCN，是当前 DL 主线的正确方向。
- raw4 独立参数化是唯一可训练输出模式，已在两个 backbone 上验证。gas_head 闭包残差方向彻底关闭。
- H2 可学到高精度（R² 0.869），CH4 中等（R² 0.44），CO2 和 N2 仍困难——可能是数据内在约束（CO2 范围窄 0-15%、N2 为调节变量）而非纯架构问题。
- 当前结果处于灰色区域：overall R² 0.493 > 0.3（非停止区）但 < 0.6（非确认区）。延长训练 + 损失加权调整可能再提升 0.1-0.15。

<!-- section: risks_open_questions -->
- CO2 R²≈0 和 N2 R²≈0 是否属于数据本质限制（范围窄 + 协方差结构）尚未确定——不排除在该数据集上 DL 根本不可学这两组分。
- 当前只跑了 n=1 per run，存在运行间方差混淆风险。所有单次结论需 multi-seed 验证。
- CNN1DTCNFusion 训练在 epoch 80 仍在改善，最佳 epoch 数和最终收敛水平未知。

<!-- section: next_step -->
- 延长训练 + 损失加权 —— Stage 1.5：epochs=160、lr 余弦衰减、patience=20、CO2/N2 高权重配置。
- 若延长后 overall R²>=0.6 且 N2 R²>0，进入 multi-seed 验证（3 seeds）。
- 若延长后 overall R² 仍 <0.6，考虑增大 backbone 容量（acoustic_channels [32,64,128,128]）。
- 保留 PhaseWindowTCN + window_attention（Stage 2）为后备方向，但当前优先挖掘 CNN1DTCNFusion 最大潜力。

<!-- section: recent_pivots -->
- 2026-06-23：dl_feature_upgrade_stage1 完成分析。CNN1DTCNFusion raw4 overall R² +0.493 远优于 PhaseWindowTCN；gas_head 在两个 backbone 证实不可训练，方向废弃。
- 2026-06-23：n2_input_contract_ablation 完成。反量化输入优于 raw int16 证实，输入契约对照线收束。
- 2026-06-23：n2_head_sweep 完成。gas_head 闭包残差为 N2 不可学主因证实；head 参数化对照线收束。
- 2026-06-23：完成三处小修并通过目标测试。
- 2026-06-22：PyTorch 最佳实践审查 DL 代码，全量 314 测试通过。
- 2026-06-19：N2 诊断修正，主因是 gas_head 闭包残差参数化而非非线性压缩。
