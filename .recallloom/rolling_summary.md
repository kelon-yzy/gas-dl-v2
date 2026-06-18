<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [ZCode] | 2026-06-18 -->
<!-- file-state: revision=72 | updated-at=2026-06-18T17:30:00+08:00 | writer-id=ZCode | base-workspace-revision=140 -->

<!-- section: current_state -->
## 阶段状态：速度优化 ✅ 已关闭 → DL 模型改进 ▶️ 进行中

### 速度优化阶段（已关闭）
- **compile=reduce-overhead 已确认为唯一有效加速方案**：+68.5% 吞吐（82.84 vs 49.16 samples/s）、21 GiB 显存稳定、无 graph break/NaN、收敛不降。
- **batch=16 确认为 compile 下最优 sweet spot**：C1+D2（batch=24）吞吐持平 C1、+10 GiB 显存无回报、val_loss 劣化 +15%、H2/CH4 per-component R2 下降。D1/D2/D3 不再跑。
- **drop_last 全链路支持已完成**，为 compile CUDA graphs 消除末 batch 二次 capture。
- **compile Phase 1 联合回归配置已建**（4 run），状态待确认。
- **结论**：compile=true + drop_last=true + compile_mode=reduce-overhead + batch_size=16 写入正式训练基线（待测试环境确认 compile Phase 1 无阻塞后立即生效）。

### DL 模型改进阶段（当前）
- PhaseWindowTCN 仍不能作为正式 DL 主线：**N2 组分 R2 ≈ 0**（test x_N2 R2=0.001, extrapolation x_N2 R2=0.003），H2/CH4/CO2 R2 在 0.5-0.7 可接受区间。
- Phase 1 完成 2/4 run（gas_varweight ✅、gas_free ✅），gas_free_varweight 被中断未完成，handcraft_mlp 未运行。
- **gas_varweight 最佳 run**（weighted_component_mse + gas_head）：val_loss=0.5821、test R2=0.5006。per-component：H2 0.646、CH4 0.497、CO2 0.515、N2 **0.001**。
- **gas_free 对照**（free_component_mse + gas_head）：H2 0.715、CH4 0.492、CO2 **0.251**、N2 **0.005**。CO2 明显劣化，free_component_mse 不可行。
- **gas_head 已确认为正确输出方向**，闭包问题已修复（sum_abs_error≈0），但 N2 仍不可学。
- 正式 ML 主线继续保持 `ridge_multiwindow_all_modalities`（test N2 R2=0.7121）。

<!-- section: active_judgments -->
### 速度优化（已关闭判断）
- **compile=reduce-overhead 是唯一有效的训练加速方案**（+68.5% 吞吐、21 GiB 稳定、收敛不降）。
- **batch=16 是 compile 下的最优 sweet spot**。
- micro_batch_size / gradient accumulation / persistent_workers 不采用。num_workers=2 是拐点。

### DL 模型改进（当前判断）
- **核心问题：N2 组分 R2 ≈ 0 对所有损失函数和 head 配置鲁棒**。这说明问题不在 loss/head 设计，而在特征提取层未捕获 N2 相关信息。
- **gas_head 应保留为默认 head**：已修复闭包、在 H2/CH4/CO2 上工作正常。
- **free_component_mse 不可行**：CO2 劣化严重，N2 仍不可学。
- **weighted_component_mse 为当前最佳损失**：H2/CH4/CO2 均衡，N2 持平。
- **结构实验（share_window_encoder=false、5-block TCN）尝试了结构消融但未解决 N2**，且 free_component_mse 作为背景损失可能掩盖了结构收益。
- N2 不可学的主因候选重新排序：① **输入信号中 N2 相关信息不足/噪声过大** > ② **共享特征提取对低信噪比组分不利** > ③ **TCN 感受野不足以捕获 N2 特征时间尺度** > ④ **训练信号被高方差组分主导**。
- ML 改进序列（按投入产出比）：A 物理派生特征 > B alpha CV + PLS/KernelRidge 对照 > C 约束/闭包建模 > D 窗口与特征选择。

<!-- section: risks_open_questions -->
- compile Phase 1 联合回归结果待确认；若全部通过，compile 写入正式训练基线。
- gas_free_varweight（weighted_free_component_mse）被中断，handcraft_mlp 未运行。
- 工作区未提交改动持续累积（cli.py / run_experiment.py / 多个配置 / 多个输出目录 / 训练配置优化方案.md）。
- **DL N2 问题可能是数据固有难题**：如果传感器数据中 N2 浓度与声学/光学信号之间的相关性本身就弱，模型可能永远无法达到 ML ridge 水平（N2 R2=0.71）。

<!-- section: next_step -->
### 速度优化收尾（立即）
- [ ] 确认 compile Phase 1 结果（服务器端），无阻塞后写入正式训练基线。
- [ ] 提交工作区累积改动。

### DL 策略改进方向（按优先级）
- **方向 A — 诊断先行**：用 conditional_metrics（N2 bins）确认 N2 在哪些浓度区间可学/不可学；对模型中间特征做 PCA/UMAP 可视化，检查编码器是否提取了 N2 区分性特征。
- **方向 B — 逐组分预测头**：修改 shared_head，为每个组分（H2/CH4/CO2/N2）分配独立的 MLP 分支，让 N2 学习自己的特征映射，不受其他组分梯度干扰。
- **方向 C — N2 专用损失提升**：增大 N2 损失权重（单独 N2 MSE 乘系数），与 weighted_component_mse 组合。
- **方向 D — 窗口注意力融合**：在 shared_head 之前加入跨窗口注意力机制，让模型可以显式对比 null/exposure/recovery 之间的差异。
- **方向 E — 重新审视结构消融**：用 weighted_component_mse + unshared window encoder + 5-block TCN 跑完整评估，排除 free_component_mse 掩盖结构收益的可能。

<!-- section: recent_pivots -->
- 2026-06-18：C1+D2 不通过，batch 上探线收束——compile 下 batch=16 是最优 sweet spot，退回 batch=16 固定。
- 2026-06-18：C1+D2（compile=reduce-overhead + batch=24）单 run 完成判读：吞吐持平 C1、+10GiB 显存无回报、val_loss 劣化、H2/CH4 per-component R2 下降。
- 2026-06-18：决策 compile 转正，创建 Phase 1 联合回归配置 `phase_window_tcn_ablation_compile.json`（4 run）。
- 2026-06-18：C1 compile=reduce-overhead 单 run 完成并通过 §C 四道硬门槛——+68.5% 吞吐、21 GiB 显存稳定、无 graph break/NaN、val_loss 改善。
- 2026-06-18：完成 compile 线代码前置项 drop_last 全链路支持。
- 2026-06-18：完成 B4 persistent_workers 单 run，判定不足以替代正式基线。
- 2026-06-17：完成 B 组 DataLoader worker sweep，确认 num_workers=2 是拐点。
