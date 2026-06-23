<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Claude-Code] | 2026-06-23 -->
<!-- file-state: revision=79 | updated-at=2026-06-23T13:42:26+08:00 | writer-id=Claude-Code | base-workspace-revision=160 -->

<!-- section: current_state -->
- 阶段状态：速度优化已关闭；DL 模型改进继续推进。n2_head_sweep 与 n2_input_contract_ablation 两组实验均在服务器 wv4-formal-hitran-standard-6000 上执行完毕并完成分析。
- gas_head 闭包残差是 N2 不可学的主因已证实：raw4 test N2 R²=-0.11 vs gas_head N2 R²=-2.79。softmax100 从 epoch 1 完全停滞，已废弃。raw4 独立参数确认为 DL 唯一基础模式。
- 反量化电压输入优于 raw int16：test overall R² 从 -0.561 升至 -0.305（+0.256），CH4 R² 从 -1.392 升至 -0.866（+0.526），sum_abs_error 减少 19.5%。反量化已确认为更好的默认输入契约。
- 全四组分 R² 仍为负，DL 仍未超过 ridge 线性基线（N2 R²=+0.712）。PhaseWindowTCN 特征提取能力不足是核心瓶颈，head 参数化与输入契约两条对照线均已收束。
- 两次实验间同名 raw4_int16 配置存在不可忽略的运行间方差（H2 R² 在 n2_head_sweep 中为 +0.20，在 ablation 中为 +0.47），单次对比结论需谨慎。
- 2026-06-23 早年已完成 PyTorch 最佳实践修复和三处小修（grad_clip_norm 透传、sum_abs_error 输出、dequantize_waveforms 输入契约）。

<!-- section: active_judgments -->
- N2 主因判断正确且已实验证实：gas_head 闭包残差参数化 vs raw4 独立输出参数，判别变量是 N2 是否拥有独立参数。
- head 参数化线已收束：raw4 优于 gas_head，softmax100 无效。后续 DL 不再在 head 层面对照。
- 输入契约线已收束：反量化电压 + int16_scale=5.0 优于 raw int16 + 32767，建议作为 raw4 标配。但改善幅度有限，不能解决根本瓶颈。
- DL 主线瓶颈在特征提取层：当前 PhaseWindowTCN 无论输入契约和 head 如何变更，全四组分 overall R² 始终为负，证明特征提取能力不足以捕捉各组分的变化。
- PhaseWindowTCN 训练存在较大随机波动（同配置不同运行 H2 R² 差异可达 0.27），结论需跨多次实验验证。
- 后续应聚焦特征提取架构提升：Transformer backbone、更大 acoustic encoder、跨模态注意力融合。

<!-- section: risks_open_questions -->
- DL 是否能在当前 6000 样本规模下超越 ridge 线性基线尚未可知。PhaseWindowTCN 特征提取能力可能根本不足。
- 反量化优势仅为单次对照，可能是运行间方差。在最终 DL 架构确定后需 multi-seed 验证。
- CH4 和 CO2 在所有配置下 R² 均为负值，这两组分的可学性可能依赖于更强的特征提取或多任务架构。
- 2026-06-23 小修只跑了目标测试，未重跑全量 314 项测试。

<!-- section: next_step -->
- 关闭 head 参数化和输入契约两条对照线，将全部资源转向 DL 特征提取能力提升。
- 设计并实验替代 backbone：Transformer encoder 替代 TCN、增大 acoustic_channels、增加跨模态注意力融合。
- 在改动架构前，补跑 test_dl_data、test_ml_baselines、test_run_experiment、test_dl_training、test_dl_models 确保基线正确。

<!-- section: recent_pivots -->
- 2026-06-23：n2_input_contract_ablation 完成。反量化输入优于 raw int16 获证实，输入契约对照线收束。CH4 对输入格式最敏感。
- 2026-06-23：n2_head_sweep 三组对照完成。gas_head 闭包残差为 N2 不可学主因证实；softmax100 废弃；head 参数化对照线收束。
- 2026-06-23：完成三处小修并通过目标测试。pipeline 透传 grad_clip_norm；ML 输出 sum_abs_error；DL 增加显式 dequantize_waveforms 与 waveform_int16_scale 输入契约 ablation 支持。
- 2026-06-22：PyTorch 最佳实践审查 DL 代码，全量 314 测试通过。
- 2026-06-19：N2 诊断修正，主因是 gas_head 闭包残差参数化而非非线性压缩。
