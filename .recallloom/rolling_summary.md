<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [ZCode] | 2026-06-18 -->
<!-- file-state: revision=70 | updated-at=2026-06-18T14:41:32+08:00 | writer-id=ZCode | base-workspace-revision=138 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮和 ML 多窗口实验均已完成归档；ML 主线仍是 full+exposure+recovery 的 `ridge_multiwindow_all_modalities`，test N2 R2=0.7121。
- PhaseWindowTCN MVP 与 `phase_window_tcn_improvement` 服务器实验均已完成归档；`gas_head` 已显著修复闭包问题，但 `free_component_mse` 未能让 `test/extrapolation x_N2 R2` 转正。
- 当前 DL 策略保持诊断优先：先用低成本损失/监督诊断定位 N2 负 R2 的真正机制，再决定是否做结构消融；Phase 2 仍必须经过 G1/G2 决策门。
- 训练加速主口径已重写为 `docs/训练配置优化方案.md`；当前正式基线为 batch_size=16、num_workers=2、persistent_workers=false、lr=0.00015、epochs=80、patience=10、AMP float16、TF32/cudnn_benchmark 开启、compile 关闭。
- B 组 DataLoader worker sweep 已收束：B0（workers=0）失败、B2（workers=4）与基线持平、B3（workers=8）回退；当前正式训练基线继续保持 num_workers=2、persistent_workers=false。
- B4 persistent_workers 单 run 已完成，判定不足以替代正式基线。
- 新增目标服务器口径：NVIDIA RTX5880 48 GiB、AMD vCPU 32 核、内存 64 GB。
- 代码级加速已完成：`performance.tf32=true` 时同步 `torch.set_float32_matmul_precision("high")`；`Trainer.evaluate()`/`predict()` 使用 `torch.inference_mode()`。
- compile 线（C1）单 run 已完成并通过 §C 四道硬门槛：+68.5% 吞吐、21 GiB 显存稳定、无 graph break/NaN。
- compile 转正 Phase 1 联合回归配置 `phase_window_tcn_ablation_compile.json` 已建（4 run），待运行或运行中。
- batch 上探线：跳过 D1（batch=20 独立），直接创建 C1+D2 组合配置 `phase_window_tcn_ablation_c1d2_compile_batch24_screen.json`（batch=24、lr=0.00018 √k、min_lr=1.2e-6 √k、compile=reduce-overhead、drop_last=true、FP16、gas_varweight 单 run）。打破 compile/batch 不可叠加铁律，理据：C1 独立通过 21 GiB 稳定、48 GiB 余量充分。

<!-- section: active_judgments -->
- PhaseWindowTCN 仍不能作为正式 DL 主线；正式 ML 主线继续保持 `ridge_multiwindow_all_modalities`。
- `gas_head` 是正确输出头方向，应保留为后续 PhaseWindowTCN 默认 head。
- `free_component_mse` 失败的主因候选（按证据强度排序）：① 损失尺度不平衡 > ② N2 无直接监督 > ③ 早期过拟合 > ④ 窗口编码器共享稀释相位差异 > ⑤ TCN 感受野不足。
- micro_batch_size / gradient accumulation 不作为当前提速方案。
- B 组 worker sweep 已证实当前拐点仍是 num_workers=2。
- compile=reduce-overhead 已通过 C1 单 run 验证（+68.5% 吞吐），进入转正联合回归阶段。
- C1+D2 组合打破 compile/batch 不可叠加铁律，但在 C1 独立通过 + 48 GiB 余量充分的前提下风险可控。若失败则回退到纯 batch 上探线。
- ML 改进序列（按投入产出比）：A 物理派生特征 > B alpha CV + PLS/KernelRidge 对照 > C 约束/闭包建模 > D 窗口与特征选择。

<!-- section: risks_open_questions -->
- compile Phase 1 跨 loss/跨模型兼容性待联合回归验证；handcraft_mlp compile 可能触发 graph break。
- C1+D2 组合的 CUDA graphs + batch 放大双重显存压力待实测；若 reserved > 38 GiB 或持续增长需立即停止。
- 48 GiB 上 batch=32 的真实收益仍未知。
- 工作区未提交改动持续累积。
- 当前输出未记录 CPU 内存指标。

<!-- section: next_step -->
- 并行执行（互不冲突）：① compile Phase 1 `phase_window_tcn_ablation_compile.json` ② C1+D2 `phase_window_tcn_ablation_c1d2_compile_batch24_screen.json`。
- C1+D2 对照：C1（batch=16 compile, 82.84 samples/s）+ B1（batch=16 无compile, 49.16 samples/s）。重点监控 epoch1 显存峰值与 epoch2+ 稳定度。
- 若 C1+D2 因显存/gen break 失败：回退到纯 D2（batch=24、compile=false、lr=0.00018）。
- ML 侧起手做 alpha CV（LOO/GCV）+ 补 ML 侧 sum_abs_error 列。

<!-- section: recent_pivots -->
- 2026-06-18：跳过 D1，直接创建 C1+D2 组合配置（batch=24、lr=0.00018 √k、compile=reduce-overhead），打破铁律但 C1 已验证 safe。
- 2026-06-18：决策 compile 转正，创建 Phase 1 联合回归配置 `phase_window_tcn_ablation_compile.json`（4 run、drop_last=true、compile=reduce-overhead）。
- 2026-06-18：C1 compile=reduce-overhead 单 run 完成并通过 §C 四道硬门槛——+68.5% 吞吐、21 GiB 显存稳定、无 graph break/NaN、val_loss 改善。
- 2026-06-18：完成 compile 线代码前置项 drop_last 全链路支持。
- 2026-06-18：完成 B4 persistent_workers 单 run，判定不足以替代正式基线。
- 2026-06-17：完成 B 组 DataLoader worker sweep，确认 num_workers=2 仍是当前拐点。
- 2026-06-16：训练加速主口径重写——RTX5880 48 GiB 服务器口径，batch=16 基线 + worker/compile/batch 候选实测矩阵。
