<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [ZCode] | 2026-06-19 -->
<!-- file-state: revision=73 | updated-at=2026-06-19T10:19:53+08:00 | writer-id=ZCode | base-workspace-revision=146 -->

<!-- section: current_state -->
## 阶段状态：速度优化 ✅ 已关闭 → DL 模型改进 ▶️ 进行中（N2 诊断已修正）

### 速度优化阶段（已关闭）
- 正式训练基线：compile=true + compile_mode=reduce-overhead + drop_last=true + batch_size=16 + FP16（+68.5% 吞吐、21 GiB 稳定）。相位相关任务必须 pypy 预处理后使用。

### DL 模型改进阶段（当前，2026-06-19 诊断修正）
- **N2 不可学的主因已修正**：是 gas_head 的闭包残差参数化（N2 = 100 − sum），不是此前判断的非线性压缩。详见 docs/N2不可学诊断与gas_head参数化分析.md。
- 关键证据：ridge（线性、同特征、4 独立输出）test N2 R2=0.712，且全浓度分箱 R2 均≥0.86；而所有 DL run 被配置校验锁死在 gas_head（N2=残差），raw4 与 softmax100 从未测过。
- 战略问题：ridge 在全部四组分都超过最好的 DL（H2 0.994、CH4 0.920、CO2 0.977、N2 0.712），手工特征对线性模型近乎充分，PhaseWindowTCN 主线需重新论证相对 ridge 的价值。
- 正式 ML 主线继续保持 ridge_multiwindow_all_modalities（test N2 R2=0.712）。

<!-- section: active_judgments -->
### 速度优化（已关闭判断）
- compile=reduce-overhead 唯一有效加速；batch=16 是 compile 下 sweet spot；num_workers=2 是拐点。

### DL 模型改进（当前判断，2026-06-19 修正）
- **N2 问题主因 = gas_head 闭包残差参数化**：N2 = 100 − (H2+CH4+CO2)，无独立输出参数，误差由三个自由组分误差之和决定。CH4（约 76%）DL 预测差，误差整份灌入 N2，R2 ≈ 0 是结构必然，与特征是否含 N2 信息无关。
- **判别变量是“N2 有无独立输出参数”，不是线性 vs 非线性**：ridge 4 独立输出 → N2 0.71；gas_head 残差 → N2 ≈ 0。
- **以下旧判断已推翻**：
  - “非线性信息压缩 / ReLU dropout 梯度竞争”——证伪，残差参数化才是主因。
  - 方向 C（N2 专用损失）、D（窗口注意力）、E（结构消融）、F（gas_head 内线性旁路）——均基于错误诊断，废弃。方向 F 尤其不可行：gas_head 内 N2 无输出节点可接旁路。
- **新方向 = 换 head**：raw4 或 softmax100 给 N2 独立输出参数；softmax100 兼顾 sum=100 闭包，是潜在生产修法。
- gas_head 保留为可选 head；weighted_component_mse 监督全 4 列、与 head 无关，配 raw4 或 softmax100 即可正常监督 N2。

<!-- section: risks_open_questions -->
- DL 主线价值存疑：手工特征 + ridge 在全四组分碾压 DL，PhaseWindowTCN（原始波形端到端）需证明超过该线性基线，否则主线需重新论证。
- 解耦实验（S1 raw4、S2 softmax100，均配 weighted_component_mse）尚未运行，残差假设待最终确认。
- 需放开 src/dl/training/losses.py 的 validate_loss_model_output 约束，允许 weighted_component_mse 配 raw4 与 softmax100（free 类损失继续锁 gas_head）。

<!-- section: next_step -->
### DL 解耦实验（验证残差假设）
- 改 src/dl/training/losses.py:182-188：允许 weighted_component_mse 配 output_mode=raw4 与 softmax100；free_component_mse 与 weighted_free_component_mse 继续锁 gas_head。
- 建 S1（raw4 + weighted_component_mse）与 S2（softmax100 + weighted_component_mse）两个筛选 run，对照现有 gas_varweight（gas_head），均用 compile 基线、gas_varweight 同款数据。
- 判读：S1 或 S2 的 N2 R2 跳到 0.5 以上 → 残差是主因，gas_head 应被 softmax100 取代；仍 ≈ 0 → 重新考虑非线性假设。
- 详见 docs/N2不可学诊断与gas_head参数化分析.md。

<!-- section: recent_pivots -->
- 2026-06-19：N2 诊断修正——主因是 gas_head 闭包残差参数化而非非线性压缩；ridge 全四组分碾压 DL；方向 C D E F 废弃，转 raw4 与 softmax100 换 head 实验。详见 docs/N2不可学诊断与gas_head参数化分析.md。
- 2026-06-18：C1+D2 不通过，batch 上探线收束——compile 下 batch=16 是最优 sweet spot，退回 batch=16 固定。
- 2026-06-18：决策 compile 转正，完成 Phase 1 联合回归 4 run 并通过判读。
- 2026-06-18：C1 compile=reduce-overhead 单 run 通过四道硬门槛——+68.5% 吞吐、21 GiB 稳定、无 graph break NaN、val_loss 改善。
- 2026-06-18：完成 compile 线前置项 drop_last 全链路支持；完成 B4 persistent_workers 单 run，判定不转正。
- 2026-06-17：完成 B 组 DataLoader worker sweep，确认 num_workers=2 是拐点。
