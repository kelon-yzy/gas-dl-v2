<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-23 -->
<!-- file-state: revision=77 | updated-at=2026-06-23T09:15:19+08:00 | writer-id=Codex | base-workspace-revision=156 -->

<!-- section: current_state -->
- 阶段状态：速度优化已关闭；DL 模型改进继续推进。N2 诊断已修正，head 对照配置已就绪；本次完成实验可靠性与输入契约三处小修，等待服务器执行正式 n2_head_sweep。
- 速度优化基线仍为 compile=true、compile_mode=reduce-overhead、drop_last=true、batch_size=16、FP16，batch=16 是 compile 下 sweet spot，num_workers=2 是拐点。
- N2 不可学主因已修正为 gas_head 闭包残差参数化，判别变量是 N2 是否有独立输出参数，不是线性或非线性。ridge_multiwindow_all_modalities 仍是正式 ML 主线，test N2 R2=0.712，且 H2、CH4、CO2 也显著高于现有 DL。
- S1 raw4 与 S2 softmax100 已可配置运行。configs/experiment/n2_head_sweep/n2_head_sweep.json 含 gas_head 对照、raw4、softmax100 三 run；本地缺 wv4-formal-hitran-standard-6000，需服务器执行。
- 2026-06-22 已完成 PyTorch 最佳实践修复：grad_clip_norm 训练支持、随机种子补齐、显式权重初始化、evaluate 端设备累积；全量 314 项测试当时通过，n2_head_sweep dry-run 当时通过。
- 2026-06-23 完成三处小修：pipeline.run_experiment 透传 training.grad_clip_norm；ML SplitEvaluation、ML CLI 和 pipeline 输出 sum_abs_error；DL Dataset 新增默认关闭的 dequantize_waveforms，用于 raw int16 与 int16 * scale 输入契约 ablation。CNN1D-TCN 与 PhaseWindowTCN 顶层模型新增 waveform_int16_scale，默认 32767.0 不改变既有行为。

<!-- section: active_judgments -->
- N2 问题主因 = gas_head 闭包残差参数化。gas_head 内 N2 = 100 - (H2 + CH4 + CO2)，没有独立输出参数。
- weighted_component_mse 监督全 4 列，允许 phase_window_tcn 配 raw4、softmax100、gas_head；free_component_mse 与 weighted_free_component_mse 只监督前三列，继续要求 gas_head。
- 若 S1 或 S2 的 N2 仍低，不能直接复活非线性压缩假设；必须先看 H2、CH4、CO2 是否也低于 ridge。若四组分整体低，根因应回到 PhaseWindowTCN 特征提取能力不足。
- 反量化输入必须是显式 ablation，不应静默替换现有 DL 输入。打开 dequantize_waveforms 时，建议同步在 model_kwargs 中设置 waveform_int16_scale=5.0，避免反量化电压再次按 32767 缩放。
- ML 闭包误差的单一事实来源放在 evaluate_regressor 的 SplitEvaluation.sum_abs_error，pipeline 和 ML CLI 复用该字段。

<!-- section: risks_open_questions -->
- S1/S2 仍未在正式 6000 数据集上执行；当前无法判定 raw4 或 softmax100 是否能恢复 N2。
- DL 主线价值仍存疑：手工特征加 ridge 在全四组分强于现有 PhaseWindowTCN，DL 需要证明超过线性手工特征基线。
- 2026-06-23 小修只跑了目标测试：6 个配置与数据测试、2 个 pipeline 训练烟测、63 个 DL CLI 与模型测试、24 个 run_experiment 测试；未重跑全量 314 项测试。
- dequantize_waveforms 只是输入契约 ablation 开关，默认 false。后续实验需分别记录 raw int16、反量化电压及 waveform_int16_scale 设置，否则结果不可比。

<!-- section: next_step -->
- 在服务器执行 n2_head_sweep：PYTHONPATH=src python -m pipeline.run_experiment --config configs/experiment/n2_head_sweep/n2_head_sweep.json。
- 保留原三 run 作为 head 对照；另派生一个小型输入契约 ablation 配置，对比 raw int16 输入与 dequantize_waveforms=true 加 waveform_int16_scale=5.0。
- 取 test split 的 x_N2.r2，同时检查 H2、CH4、CO2、overall R2 和 sum_abs_error；按 docs/N2不可学诊断与gas_head参数化分析.md 的三分支规则判读。
- 如准备提交这批小修，建议先跑全量 pytest 或至少补跑 tests/test_dl_data.py、tests/test_ml_baselines.py、tests/test_run_experiment.py、tests/test_dl_training.py、tests/test_dl_models.py。

<!-- section: recent_pivots -->
- 2026-06-23：完成三处小修并通过目标测试。pipeline 透传 grad_clip_norm；ML 输出 sum_abs_error；DL 增加显式 dequantize_waveforms 与 waveform_int16_scale 输入契约 ablation 支持。
- 2026-06-22：使用 PyTorch 最佳实践审查 DL 代码，修复梯度裁剪、随机种子、权重初始化和 evaluate 同步问题；当时全量 314 测试通过。
- 2026-06-20：代码改动完成，losses.py 拆分 head 约束，S1 raw4 与 S2 softmax100 配置就绪；本地数据集缺失，待服务器执行。
- 2026-06-20：证据校正，ridge test 的 N2 分箱列 R2 实测全为负，分箱方差塌缩导致不可判读；全局 N2 R2=0.712 仍证明可学。
- 2026-06-19：N2 诊断修正，主因是 gas_head 闭包残差参数化而非非线性压缩；方向 C、D、E、F 废弃。
- 2026-06-18：完成速度优化收束，compile=reduce-overhead 转正，batch=16 与 num_workers=2 成为当前基线。
