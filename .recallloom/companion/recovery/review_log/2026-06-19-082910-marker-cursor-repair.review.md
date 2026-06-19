# Marker 与游标修复 Recovery Review

## 提案引用
Marker 与游标修复 recovery proposal（2026-06-19-082910-marker-cursor-repair）。

## 审阅结论
通过。现有 rolling_summary.md 内容与 daily_logs/2026-06-18.md（含 entry-8/9）接受为 reviewed imported baseline，可用于 helper-managed writes。内容真实有效，仅标记损坏需修复。

## 通过项
- 将 rolling_summary.md 现有内容作为 reviewed imported baseline，授权 helper 重写以生成合法 file-state marker。
- 同步 daily-log 游标 state.json 到 entry-9。
- 保留 entry-8/9 的非标准 section 不动（append-only 历史记录）。

## 拒绝项
- 不重构任何 section 内容。
- 不改变任何实验结论或判断。

## 提升状态
批准提升。没有 hint-only 项。

## 下一步
运行 prepare_recovery_promotion.py，重新执行 preflight，然后用 dispatcher write 重写 rolling_summary、用 repair-daily-log-cursor 修复游标。

Hint-only handling: no items remain hint-only.
