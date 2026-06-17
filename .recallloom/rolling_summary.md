<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-16 -->
<!-- file-state: revision=63 | updated-at=2026-06-16T18:22:44+08:00 | writer-id=Codex | base-workspace-revision=123 -->

<!-- section: current_state -->
- Phase-aware N2 第一轮和 ML 多窗口实验均已完成归档；ML 主线仍是 full+exposure+recovery 的 `ridge_multiwindow_all_modalities`，test N2 R2=0.7121。
- PhaseWindowTCN MVP 与 `phase_window_tcn_improvement` 服务器实验均已完成归档；`gas_head` 已显著修复闭包问题，但 `free_component_mse` 未能让 `test/extrapolation x_N2 R2` 转正。
- 当前 DL 策略保持诊断优先：先用低成本损失/监督诊断定位 N2 负 R2 的真正机制，再决定是否做结构消融；Phase 2 仍必须经过 G1/G2 决策门。
- 训练加速主口径已重写为 `docs/训练配置优化方案.md`：当前正式基线为 batch_size=16、num_workers=2、persistent_workers=false、lr=0.00015、epochs=80、patience=10、AMP float16、TF32/cudnn_benchmark 开启、compile 关闭。
- 新增目标服务器口径：NVIDIA RTX5880 48 GiB、AMD vCPU 32 核、内存 64 GB；旧 24 GiB 环境下 batch_size=32 OOM 只作为历史事实，新服务器可候选实测 batch_size=24/32，但不能直接转正。
- 训练优化相关旧文档已整理：`docs/OOM修正记录.md`、`docs/训练配置优化应用记录.md`、`docs/训练时间优化执行总结.md` 降级为历史记录；`docs/Phase1_Phase2运行指南.md` 与辅助脚本已对齐当前基线和候选实测流程。
- 代码级加速已完成：`performance.tf32=true` 时同步 `torch.set_float32_matmul_precision("high")`；`Trainer.evaluate()`/`predict()` 使用 `torch.inference_mode()`。

<!-- section: active_judgments -->
- PhaseWindowTCN 仍不能作为正式 DL 主线；正式 ML 主线继续保持 `ridge_multiwindow_all_modalities`。
- `gas_head` 是正确输出头方向，应保留为后续 PhaseWindowTCN 默认 head。
- `free_component_mse` 失败的主因候选（按证据强度排序）：① 损失尺度不平衡（大尺度 CO2 主导梯度，小尺度 N2 被忽略）> ② N2 无直接监督（gas_head 下 N2 是纯闭包残差）> ③ 早期过拟合（MVP best epoch=4）> ④ 窗口编码器共享稀释相位差异 > ⑤ TCN 感受野不足。
- micro_batch_size / gradient accumulation 不作为当前提速方案；它主要降低峰值显存，通常不会提升整体训练速度。
- 训练加速候选顺序固定为：先跑当前 Phase 1 基线，再做 num_workers=0/2/4/8 sweep，再单 run 测 `torch.compile(mode="reduce-overhead")`，最后按 batch_size=20 -> 24 -> 32 逐步上探。
- batch 变大不是纯性能开关，会改变 optimizer step 数和收敛轨迹；只有速度、显存、val_loss、test/extrapolation x_N2 R2 全部通过，候选才可写入正式配置。
- ML 改进序列（按投入产出比）：A 物理派生特征（ToF/声速/FFT，最高优先）> B alpha CV + PLS/KernelRidge 对照 > C 约束/闭包建模 > D 窗口与特征选择。

<!-- section: risks_open_questions -->
- 48 GiB 服务器尚未完成当前基线实测；batch_size=24/32、num_workers=4/8 和 torch.compile 的真实收益仍未知。
- torch.compile 可能增加显存或触发 compile/graph break 问题；只能先做单 run 验证。
- DL 诊断批次若 weighted loss / handcraft MLP 均无正向 N2 增益，应收束 DL 线，不应直接进入 split/deep 结构消融。
- ML 物理特征提取（ToF/声速）需要原始波形访问与信号处理能力，当前代码框架是否支持待验证。
- 是否引入 sklearn 作正式依赖（影响 PLS/KernelRidge 与约束 LS；物理特征/alpha CV/闭包后处理不受影响）待决策。
- 工作区仍有本轮文档、脚本、DL 训练代码与测试改动未提交；提交时需避免误纳入无关输出。

<!-- section: next_step -->
- 在新 48 GiB 服务器上先运行 `phase_window_tcn_ablation` 当前基线，记录 epoch_seconds、train_samples_per_second、gpu_memory_allocated_mb/reserved_mb、best_epoch、val_loss、test/extrapolation x_N2 R2。
- 基线完成后按 `docs/训练配置优化方案.md` 的验收表依次测试 worker sweep、torch.compile reduce-overhead、batch_size=20/24/32。
- ML 侧起手做 alpha CV（LOO/GCV）+ 补 ML 侧 sum_abs_error 列，建立正则与闭包事实基线。
- ML 侧主增益集中攻超声 ToF/声速 + FFT/小波特征，PLS/KernelRidge 与闭包后处理作对照。
- Phase 1 诊断批结果判读优先看 `test x_N2 R2`、`extrapolation x_N2 R2`、per-component R2 分布、`best_epoch` 与 train/val loss 曲线。

<!-- section: recent_pivots -->
- 2026-06-16：训练加速主口径重写——加入 NVIDIA RTX5880 48 GiB / AMD vCPU 32 核 / 64 GB RAM 服务器配置；废弃旧 `batch_size=32 -> 6-8GB` 估算，改为 batch=16 基线 + worker/compile/batch 候选实测矩阵。
- 2026-06-16：完成训练代码小幅加速——TF32 配置同步 `torch.set_float32_matmul_precision("high")`，验证/预测阶段切到 `torch.inference_mode()`；`tests/test_dl_training.py tests/test_run_experiment.py` 55 passed，`git diff --check` 通过。
- 2026-06-16：DL 策略转向诊断优先——复核 `free_component_mse` 代码与文献后，判定 N2 负 R2 更可能来自损失尺度与监督方式而非窗口编码器结构；重写方案文档为诊断 > 结构消融 > 融合/对数比 + 决策门框架。
- 2026-06-16：新增 ML 改进方向分析文档——基于代码现状与文献检索，确认超声 ToF/声速是最大低垂果实，alpha CV 与 PLS 是次优先项，log-ratio(ILR/ALR) 已判负向不重试。
- 2026-06-12：完成 multiwindow_n2 服务器结果分析与归档；full+exposure+recovery 多窗口拼接强通过，test `N2 R2=0.7121`，extrapolation `N2 R2=0.7247`。
