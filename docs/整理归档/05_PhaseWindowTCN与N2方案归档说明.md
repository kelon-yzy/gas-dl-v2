# PhaseWindowTCN 与 N2 方案归档说明

> 归档日期：2026-06-15
> 说明：本页记录已被新实验方案覆盖的旧文档，便于追溯历史判断

## 已归档文档

- `N2改进计划_ilr_alr.md`
- `N2改进计划_多窗口特征拼接.md`
- `phase_window_tcn_improvement_analysis.md`
- `phase_preserving_fusion_design.md`
- `PhasePreservingTCN_详细设计.md`

## 归档原因

这些文档分别对应以下已被当前结论覆盖的路线：

- `ILR/ALR` 组成数据目标变换主线
- ML 多窗口特征拼接主线
- PhaseWindowTCN 的头部 / loss 改进分析
- PhasePreserving / phase-aware 复杂融合设想

当前新的活跃实验文档已统一为：

- [PhaseWindowTCN 结构消融实验方案](../PhaseWindowTCN结构消融实验方案.md)

## 当前判断

新的活跃方案只保留最小可验证方向：

- `share_window_encoder=false`
- 更深 TCN 感受野
- 必要时再考虑轻量融合

旧文档保留为历史证据，不再作为当前主线执行依据。
