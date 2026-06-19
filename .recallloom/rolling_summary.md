<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [ZCode] | 2026-06-19 -->
<!-- file-state: revision=72 | updated-at=2026-06-19T09:33:47+08:00 | writer-id=ZCode | base-workspace-revision=144 -->

<!-- section: current_state -->
## 阶段状态：速度优化 ✅ 已关闭 → DL 模型改进 ▶️ 进行中

### 速度优化阶段（已关闭）
- **compile=reduce-overhead 已确认为唯一有效加速方案**：+68.5% 吞吐、21 GiB 显存稳定、无 graph break/NaN、收敛不降。
- **batch=16 确认为 compile 下最优 sweet spot**。
- **drop_last 全链路支持已完成**。
- **compile Phase 1 联合回归（4 run）已全部完成并通过判读**。
- **正式训练基线**：compile=true + compile_mode=reduce-overhead + drop_last=true + batch_size=16 + FP16。相位相关任务（phase_window/phase_windows）**必须 pypy 预处理后使用**。

### DL 模型改进阶段（当前）
- **PhaseWindowTCN 仍不能作为正式 DL 主线**：N2 组分 R2≈0 对所有配置鲁棒。
- **compile Phase 1 联合回归（4 run 跨 loss/跨模型）已全部完成**：
  - gas_free ❌ 不可行：CO2 R2=0.115，N2 R2≈-0.004
  - gas_varweight ✅ 最佳 PhasedWindowTCN 配置：test H2=0.718, CH4=0.534, CO2=0.544, N2=0.0003
  - gas_free_varweight ⚠️ 加权帮助 CO2（0.328）但仍不如 gas_varweight
  - **handcraft_mlp 🔥 关键发现**：test H2=0.950, CO2=0.911, CH4=0.728, N2=-0.007
- **正式 ML 主线继续保持** `ridge_multiwindow_all_modalities`（test N2 R2=0.712）。

<!-- section: active_judgments -->
### 速度优化（已关闭判断）
- **compile=reduce-overhead 是唯一有效的训练加速方案**，已写入正式训练基线。
- **batch=16 是 compile 下的最优 sweet spot**。
- micro_batch_size / gradient accumulation / persistent_workers 不采用。num_workers=2 是拐点。

### DL 模型改进（当前判断）
- **N2 问题已精确定位**：不是 loss 设计、head 设计、窗口编码器结构或 TCN 感受野的问题。
- **核心证据：handcraft_mlp 与 ridge 结果对比**：
  - ridge 线性回归（同特征）：N2 R2=0.712
  - HandcraftMLP 非线性（同特征）：N2 R2=-0.007
  - → **N2 只有极弱的线性信号，所有非线性模型（MLP/TCN）因 ReLU dropout 共享梯度竞争而丢失 N2**
- **gas_head 应保留为默认 head**：闭包修复有效（sum_abs_error≈0）。
- **weighted_component_mse 为默认损失**。
- **free_component_mse 系不可行**（CO2 劣化）。
- **N2 专用损失放大不可行**：非线形空间中 N2 无有效信号可放大。
- **窗口注意力/结构消融不可行**：handcraft_mlp 无 TCN/波形编码器同样学不到 N2。
- **新方向 F — 线性 / N2 专用残差通路**：在 gas_head 前加一条线性通路专用于 N2，绕过共享非线性压缩。等价于在 DL 中嵌入 mini ridge regression。

<!-- section: risks_open_questions -->
- ✅ compile Phase 1 联合回归已全部完成并通过判读。
- 工作区未提交改动持续累积（handcraft_mlp_compile.json + 累积改动）。
- **N2 线性通路能否在 PhaseWindowTCN 中实现**需要工程验证。
- handcraft_mlp 的 H2/CO2 R2 接近 0.95，说明**统计特征已含足够信息，DL 瓶颈在于非线性信息压缩**。

<!-- section: next_step -->
### 速度优化收尾（已完成）
- ✅ compile Phase 1 全部完成并判读通过。
- ✅ compile 写入正式训练基线的条件全部满足。
- ⏳ 提交工作区累积改动。

### DL 方向 F — 线性 / N2 专用残差通路
- 在共享特征提取（window encoder + TCN）之后，加一条并行线性分支：
  - 线性分支：特征 → Linear(out_dim) → 直接输出，无激活/无 dropout
  - gas_head 分支：保持不变
  - 合并：gas_head 输出（H2/CH4/CO2） + 线性分支 N2 输出
- 或者更简单：在 gas_head 的 shared_hidden 后加一条 residual linear skip 直达 N2
- 评估指标：N2 R2 > 0，至少正数

<!-- section: recent_pivots -->
- 2026-06-18：C1+D2 不通过，batch 上探线收束——compile 下 batch=16 是最优 sweet spot，退回 batch=16 固定。
- 2026-06-18：C1+D2（compile=reduce-overhead + batch=24）单 run 完成判读：吞吐持平 C1、+10GiB 显存无回报、val_loss 劣化、H2/CH4 per-component R2 下降。
- 2026-06-18：决策 compile 转正，创建 Phase 1 联合回归配置 `phase_window_tcn_ablation_compile.json`（4 run）。
- 2026-06-18：C1 compile=reduce-overhead 单 run 完成并通过 §C 四道硬门槛——+68.5% 吞吐、21 GiB 显存稳定、无 graph break/NaN、val_loss 改善。
- 2026-06-18：完成 compile 线代码前置项 drop_last 全链路支持。
- 2026-06-18：完成 B4 persistent_workers 单 run，判定不足以替代正式基线。
- 2026-06-17：完成 B 组 DataLoader worker sweep，确认 num_workers=2 是拐点。
