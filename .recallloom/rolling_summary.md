<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [unknown] | 2026-06-20 -->
<!-- file-state: revision=75 | updated-at=2026-06-20T09:08:37+08:00 | writer-id=unknown | base-workspace-revision=150 -->

<!-- section: current_state -->
## 阶段状态：速度优化 ✅ 已关闭 → DL 模型改进 ▶️ 进行中（N2 诊断已修正、代码已改、S1/S2 配置就绪待执行）

### 速度优化阶段（已关闭）
- 正式训练基线：compile=true + compile_mode=reduce-overhead + drop_last=true + batch_size=16 + FP16（+68.5% 吞吐、21 GiB 稳定）。相位相关任务必须 pypy 预处理后使用。

### DL 模型改进阶段（当前，2026-06-20）
- **N2 不可学的主因已修正**：是 gas_head 的闭包残差参数化（N2 = 100 − sum），不是此前判断的非线性压缩。详见 docs/N2不可学诊断与gas_head参数化分析.md。
- 关键证据：ridge（线性、同特征、4 独立输出）test N2 R2=0.712，跨四 split 稳定在 0.71-0.74，确立 N2 全局可学；而所有 DL run 被配置校验锁死在 gas_head（N2=残差），raw4 与 softmax100 从未测过。
- **证据校正（2026-06-20）**：此前误把 ridge test 的 N2 分箱聚合 R2（bin.metrics.r2）当成 N2 列 R2。实测 N2 列分箱 R2 全为负（-5.95 / -2.33 / -1.65 / -5.57），是分箱方差塌缩所致，不可据分箱判可学性。可学性仅由全局 N2 R2=0.712 确立。详见 docs 与 daily_logs/2026-06-20.md entry-1。
- **代码改动（2026-06-20）**：losses.py 的 validate_loss_model_output 已拆分校验——weighted_component_mse（全 4 列监督）允许 phase_window_tcn 配 raw4/softmax100/gas_head；free 类闭包损失（仅 3 列监督）继续锁 gas_head。详见 daily_logs/2026-06-20.md entry-2。
- **S1/S2 配置就绪**：configs/experiment/n2_head_sweep/n2_head_sweep.json 含三 run（gas_varweight 对照 / S1 raw4 / S2 softmax100），compile 基线，dry-run 通过。本地无 wv4-formal-hitran-standard-6000 数据集，须在服务器执行。
- 战略问题：ridge 在全部四组分都超过最好的 DL（H2 0.994、CH4 0.920、CO2 0.977、N2 0.712），手工特征对线性模型近乎充分，PhaseWindowTCN 主线需重新论证相对 ridge 的价值。
- 正式 ML 主线继续保持 ridge_multiwindow_all_modalities（test N2 R2=0.712）。

<!-- section: active_judgments -->
### 速度优化（已关闭判断）
- compile=reduce-overhead 唯一有效加速；batch=16 是 compile 下 sweet spot；num_workers=2 是拐点。

### DL 模型改进（当前判断，2026-06-20 更新）
- **N2 问题主因 = gas_head 闭包残差参数化**：N2 = 100 − (H2+CH4+CO2)，无独立输出参数，误差由三个自由组分误差之和决定。
- **判别变量是“N2 有无独立输出参数”，不是线性 vs 非线性**：ridge 4 独立输出 → N2 0.71；gas_head 残差 → N2 ≈ 0。
- **以下旧判断已推翻**：“非线性压缩”证伪；方向 C D E F 废弃。方向 F 尤其不可行——gas_head 内 N2 无输出节点。
- **新方向 = 换 head**：raw4 或 softmax100 给 N2 独立输出参数；softmax100 兼顾 sum=100 闭包。weighted_component_mse 已允许配 raw4/softmax100。
- S1/S2 判读盲区规则（已补）：N2 仍低时不能直接复活非线性假设，须先查四组分整体是否低于 ridge；仅四组分追平而独 N2 低时才考虑非线性耦合。

<!-- section: risks_open_questions -->
- DL 主线价值存疑：手工特征 + ridge 在全四组分碾压 DL，PhaseWindowTCN 需证明超过该线性基线。
- S1/S2 未执行：本地仅有 wv4-smoke（36 条），须在服务器运行。命令：`PYTHONPATH=src python -m pipeline.run_experiment --config configs/experiment/n2_head_sweep/n2_head_sweep.json`
- 判读规则已补盲区，待 S1/S2 结果后按三分支判读。

<!-- section: next_step -->
### 在服务器执行 n2_head_sweep 实验
- 传 configs/experiment/n2_head_sweep/n2_head_sweep.json 到有 wv4-formal-hitran-standard-6000 的环境。
- 三 run 串行约 45 分钟。输出到 outputs/runs/n2_head_sweep/{run_name}/。
- 取 test split 的 x_N2.r2 对照 docs/N2不可学诊断与gas_head参数化分析.md 的判读规则分析。

<!-- section: recent_pivots -->
- 2026-06-20：代码改动完成——losses.py 拆分 head 约束，S1 (raw4) 与 S2 (softmax100) 配置就绪；本地数据集缺失，待服务器执行。
- 2026-06-20：证据校正——ridge test 的 N2 分箱列 R2 实测全为负（分箱方差塌缩），此前“分箱 R2 均≥0.86”是误读。
- 2026-06-19：N2 诊断修正——主因是 gas_head 闭包残差参数化而非非线性压缩；ridge 全四组分碾压 DL；方向 C D E F 废弃。
- 2026-06-18：C1+D2 不通过，batch 上探线收束——compile 下 batch=16 是 sweet spot。
- 2026-06-18：决策 compile 转正，完成 Phase 1 联合回归 4 run 并通过判读。
- 2026-06-17：完成 B 组 DataLoader worker sweep，确认 num_workers=2 是拐点。
