<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-17 -->
<!-- file-state: revision=64 | updated-at=2026-06-17T16:20:24+08:00 | writer-id=Codex | base-workspace-revision=127 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮和 ML 多窗口实验均已完成归档；ML 主线仍是 full+exposure+recovery 的 `ridge_multiwindow_all_modalities`，test N2 R2=0.7121。
- PhaseWindowTCN MVP 与 `phase_window_tcn_improvement` 服务器实验均已完成归档；`gas_head` 已显著修复闭包问题，但 `free_component_mse` 未能让 `test/extrapolation x_N2 R2` 转正。
- 当前 DL 策略保持诊断优先：先用低成本损失/监督诊断定位 N2 负 R2 的真正机制，再决定是否做结构消融；Phase 2 仍必须经过 G1/G2 决策门。
- 训练加速主口径已重写为 `docs/训练配置优化方案.md`：当前正式基线为 batch_size=16、num_workers=2、persistent_workers=false、lr=0.00015、epochs=80、patience=10、AMP float16、TF32/cudnn_benchmark 开启、compile 关闭。
- B 组 DataLoader worker sweep 已收束：B0（workers=0）失败、B2（workers=4）与基线持平、B3（workers=8）回退；当前正式训练基线继续保持 num_workers=2、persistent_workers=false。
- 已创建 B4 单 run 配置 `configs/experiment/phase_window_tcn_ablation/phase_window_tcn_ablation_b4_workers2_persistent_screen.json`，用于验证 `persistent_workers=true` 是否能减少 epoch 间 worker 重启开销。
- 新增目标服务器口径：NVIDIA RTX5880 48 GiB、AMD vCPU 32 核、内存 64 GB；旧 24 GiB 环境下 batch_size=32 OOM 只作为历史事实，新服务器可候选实测 batch_size=24/32，但不能直接转正。
- 代码级加速已完成：`performance.tf32=true` 时同步 `torch.set_float32_matmul_precision("high")`；`Trainer.evaluate()`/`predict()` 使用 `torch.inference_mode()`。

<!-- section: active_judgments -->
- PhaseWindowTCN 仍不能作为正式 DL 主线；正式 ML 主线继续保持 `ridge_multiwindow_all_modalities`。
- `gas_head` 是正确输出头方向，应保留为后续 PhaseWindowTCN 默认 head。
- `free_component_mse` 失败的主因候选（按证据强度排序）：① 损失尺度不平衡（大尺度 CO2 主导梯度，小尺度 N2 被忽略）> ② N2 无直接监督（gas_head 下 N2 是纯闭包残差）> ③ 早期过拟合（MVP best epoch=4）> ④ 窗口编码器共享稀释相位差异 > ⑤ TCN 感受野不足。
- micro_batch_size / gradient accumulation 不作为当前提速方案；它主要降低峰值显存，通常不会提升整体训练速度。
- B 组 worker sweep 已证实当前拐点仍是 num_workers=2；B4 只验证 persistent_workers=true，若吞吐不升则不改正式配置。
- torch.compile 和 batch_size 上探仍必须在 B4 之后逐个单 run 验证，且不得与其他候选叠加。
- ML 改进序列（按投入产出比）：A 物理派生特征（ToF/声速/FFT，最高优先）> B alpha CV + PLS/KernelRidge 对照 > C 约束/闭包建模 > D 窗口与特征选择。

<!-- section: risks_open_questions -->
- B4 persistent_workers=true 是否能在不明显增加 CPU/内存压力的前提下带来吞吐改善仍未知。
- torch.compile 可能增加显存或触发 compile/graph break 问题；只能先做单 run 验证。
- 48 GiB 服务器上 batch_size=24/32 的真实收益仍未知。
- DL 诊断批次若 weighted loss / handcraft MLP 均无正向 N2 增益，应收束 DL 线，不应直接进入 split/deep 结构消融。
- ML 物理特征提取（ToF/声速）需要原始波形访问与信号处理能力，当前代码框架是否支持待验证。
- 是否引入 sklearn 作正式依赖（影响 PLS/KernelRidge 与约束 LS；物理特征/alpha CV/闭包后处理不受影响）待决策。
- 工作区仍有本轮文档、脚本、DL 训练代码与测试改动未提交；提交时需避免误纳入无关输出。

<!-- section: next_step -->
- 在 48 GiB 服务器上先运行 B4 单 run，记录 epoch_seconds、train_samples_per_second、gpu_memory_allocated_mb/reserved_mb、CPU 内存、best_epoch、val_loss、test/extrapolation x_N2 R2。
- 若 B4 吞吐不高于 B1 或内存压力明显增加，则锁定正式训练基线为 batch_size=16、num_workers=2、persistent_workers=false，并转入 `torch.compile(mode="reduce-overhead")` 单 run 验证。
- 若 B4 通过，再按 `docs/训练配置优化方案.md` 的验收表依次推进 `torch.compile(mode="reduce-overhead")` 和 batch_size=20/24/32 单 run。
- ML 侧起手做 alpha CV（LOO/GCV）+ 补 ML 侧 sum_abs_error 列，建立正则与闭包事实基线。
- Phase 1 诊断批结果判读优先看 `test x_N2 R2`、`extrapolation x_N2 R2`、per-component R2 分布、`best_epoch` 与 train/val loss 曲线。

<!-- section: recent_pivots -->
- 2026-06-17：完成 B 组 DataLoader worker sweep 单 run；B0=21.1 samples/s 失败，B2=91.98 s/epoch + 49.0 samples/s 持平，B3=99.40 s/epoch + 45.2 samples/s 回退，确认 num_workers=2 仍是当前拐点。
- 2026-06-17：新增 B4 配置 `phase_window_tcn_ablation_b4_workers2_persistent_screen.json`，用于验证 `persistent_workers=true` 是否能减少 epoch 间 worker 重启开销。
- 2026-06-16：训练加速主口径重写——加入 NVIDIA RTX5880 48 GiB / AMD vCPU 32 核 / 64 GB RAM 服务器配置；废弃旧 `batch_size=32 -> 6-8GB` 估算，改为 batch=16 基线 + worker/compile/batch 候选实测矩阵。
- 2026-06-16：完成训练代码小幅加速——TF32 配置同步 `torch.set_float32_matmul_precision("high")`，验证/预测阶段切到 `torch.inference_mode()`；`tests/test_dl_training.py tests/test_run_experiment.py` 55 passed，`git diff --check` 通过。
- 2026-06-16：DL 策略转向诊断优先——复核 `free_component_mse` 代码与文献后，判定 N2 负 R2 更可能来自损失尺度与监督方式而非窗口编码器结构；重写方案文档为诊断 > 结构消融 > 融合/对数比 + 决策门框架。
