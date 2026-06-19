# Marker 与游标修复 Recovery Proposal

## 来源摘要
外部写入方 ZCode 在 2026-06-18 绕过 helper 直接手写了 `rolling_summary.md`（更新到 compile 转正 + DL 模型改进阶段）以及 `daily_logs/2026-06-18.md` 的 entry-8、entry-9。手写导致两处结构损坏：rolling_summary 的 file-state marker 把 `base-workspace-revision` 写成 git 短哈希 `ee69455`（应为数字），daily-log 游标 state.json 仍停在 entry-7 而文件已到 entry-9。内容本身真实有效，仅元数据标记损坏。

## 来源类型与可信级别
来源类型：本地 sidecar 文件（已存在的项目连续性内容）。可信级别：高——内容与 git 提交 ee69455、816f807、76a5194、bd533a6、b2815fb 及实验输出一致，无内容冲突，仅标记格式损坏。

## 候选当前状态事实
- 速度优化阶段已关闭：compile=reduce-overhead + batch=16 + drop_last=true + FP16 为正式训练基线。
- DL 模型改进阶段进行中：N2 R2≈0 对所有配置鲁棒。
- compile Phase 1 联合回归 4 run 全部完成并判读通过。
- handcraft_mlp test H2=0.950/CO2=0.911/CH4=0.728/N2=-0.007。

## 候选里程碑事件
- compile Phase 1 联合回归 4 run 完竣（entry-9）。
- 速度优化阶段正式关闭、DL 模型改进阶段启动（entry-8）。

## 候选判断反转
- 无判断反转。本次仅修复标记损坏，不改变任何实验结论。

## 候选下一步变化
- 修复 rolling_summary file-state marker 为合法数字 base-workspace-revision。
- 同步 daily-log 游标 state.json 到 entry-9。

## 与当前 sidecar 的冲突
无内容冲突。冲突仅为元数据层：file-state marker 格式非法、state.json 游标落后于实际文件。entry-8/9 使用了非标准 section（phase_transition/experimental_results/active_dl_directions），属 append-only 历史记录，保留不动。

## 建议提升动作
批准把现有 rolling_summary.md 内容与 daily_logs/2026-06-18.md（含 entry-8/9）作为 reviewed imported baseline，授权 helper 重新生成合法 marker 并修复游标。提升目标：rolling_summary.md 与 state.json 游标。

## 审阅结论
建议接受。内容为可持续的真实项目连续性，仅需修复标记与游标，不涉及内容重构。
