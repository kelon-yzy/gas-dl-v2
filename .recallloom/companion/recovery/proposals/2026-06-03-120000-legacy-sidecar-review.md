# RecallLoom Legacy Sidecar Review Proposal

## 来源摘要

当前项目已有 `.recallloom` sidecar，`preflight_context_check.py` 显示其结构可读，但 provenance metadata 缺失，状态为 `review_required`。本提案用于审核并导入该 legacy baseline，使后续可通过 helper 记录新的 daily log。

## 来源类型与可信级别

来源类型为本项目现有 RecallLoom sidecar 和本轮本地验证结果。可信级别为 Tier B：结构化文件存在、helper 能读取当前摘要和状态，但缺少 receipt-backed provenance，因此不声明 helper_evidenced。

## 候选当前状态事实

- 当前 rolling summary 可读，记录项目处于正式实验 v4 重构主线。
- 当前任务为执行 `docs/LONG_SEQUENCE_PROTOCOL_PROPOSAL_2026-06-02.md` 的长时序实验协议改进计划。
- 本轮已经完成 S0-S5 的可运行代码闭环，并通过全量测试。

## 候选里程碑事件

- 完成 PhaseSchedule 抽象、长序列时间轴预设、stage_profile 库和 stage_jitter。
- 完成多时间常数慢传感器动态、LSTM/Transformer/PatchTST 模型、保序聚合头、DL 数据增强。
- 完成 ML baseline 的 per-phase 与 early-window 评估入口。
- 验证结果为目标测试 76 passed，全量测试 168 passed。

## 候选判断反转

无需要反转的既有判断。现有 sidecar 中关于长时序问题根因和 S0-S5 推进方向的判断仍成立。

## 候选下一步变化

下一步应从代码落地转向实验执行：生成 standard/long/multi_pulse 数据集，运行 ridge、CNN/TCN、LSTM/Transformer/PatchTST 对比，并汇总 full/per-phase/early 指标。

## 与当前 sidecar 的冲突

未发现与当前 sidecar 的实质冲突。唯一阻塞是 provenance metadata 缺失导致 helper 拒绝 daily_logs/ 写入。

## 建议提升动作

批准将现有 legacy sidecar 作为 reviewed imported baseline。后续使用 helper 将本轮完成的工作追加到 daily_logs/，必要时再同步 rolling_summary.md。

## 审阅结论

建议接受该 proposal。该动作只确认当前 sidecar 可作为 reviewed import baseline，不声明 receipt-backed helper evidence。
