# MEI-3 VarPro Phase A 执行报告

> 执行日期：2026-07-28  
> 结论：条件线性结构与数值等价审计通过，输出 `mei3_phase_a_structure_supported`。项目继续留在 `MEI-3_varpro_audit`，不得将本结论写成 S2 已通过正式求解门。
>
> 后续状态（2026-07-29）：本报告正文保留 Phase A 完成时的历史截面；B0 后续已以干基等式约束和二维正交切空间闭合，当前 verdict 为 `mei3_b0_representation_closed`。最新契约与 freeze 见 [MEI-3 后续执行计划](tv3_mrs_ei_mei3_execution_plan.md) §11。

## 1. 执行边界

Phase A 只回答“当前观测表示中是否存在可靠的条件线性干扰参数块”。它使用内存中的非正式数值等价 fixture，没有生成登记稀疏仿真数据、正式波形、benchmark 或硬件数据。

父 MEI-1 freeze：

```text
outputs/runs/tv3_mrs_ei/mei1_forward_envelope/freezes/
  20260728T064100731550Z_1b55aa2e09cb/
```

父 manifest SHA256：`faf397f9457b8eadc8871c55e488da0d62671826bf724ac3fd66f9c03b029396`。

## 2. 结构审计

| 参数块 | 结论 | 准入条件 |
| --- | --- | --- |
| 公共延迟 | 准入 | 使用 `raw_tof_s` 和已由独立预处理固定整数分支的 `unwrapped_phase_rad` |
| 对数幅度增益 | 准入 | 在 `log_amplitude` 中为加性仿射项 |
| 逐频标定偏移 | 准入 | 在 `device_profile_id × frequency_hz` 上跨样本共享，并带独立标定先验 |
| 声程 | 不准入 | 按冻结联合观测政策保留在非线性块 `beta` |
| 复传递函数中的延迟 | 不准入 | 延迟通过正弦和余弦进入实部与虚部，不是条件线性项 |

## 3. 数值等价

- 固定频点：D0 K4 `{25,63,100,200}` kHz。
- 条件数：3；观测数：36；线性参数：6。
- 增广系统秩：6，满列秩。
- VarPro 与联合正规方程参考的最大参数差：`1.0408340855860843e-17`。
- 最大增广残差差：`7.105427357601002e-15`。
- 原始单位下增广条件数：`592616.8787082422`；该值作为后续 S1 / S2 参数尺度化的诊断，不是本阶段失败门。

## 4. 冻结产物

```text
outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/
  20260728T080522165154Z_7cd8443230fa/
```

- verdict：`mei3_phase_a_structure_supported`
- issues：`[]`
- evidence manifest SHA256：`82488add1a9ab11e15a6dfe11ac60d2c7b9cfe91e5d238f538c9e7993b21121f`
- `allowed_next_stage`：`MEI-3_varpro_audit`
- `formal_solver_gate_ready`：`false`
- blocker：`registered_sparse_simulation_generation_forbidden`

## 5. 后续边界

Phase A 之后可以继续实现 S1 和 S2 的确定性求解器核心、参数尺度化与合成单元验证。在登记稀疏仿真数据获得独立授权前，不得运行正式 test / OOD 求解门，也不得输出 `mei3_varpro_supported`。

四项授权继续全部为 `forbidden_until_explicit_authorization`。

## 6. 后续实际进展（2026-07-29 追记）

阶段 B 的 B0 秩审计**未通过**：共享 MRS-1 前向对 `(x_CO2, x_O2, x_N2)` 的整体缩放精确不变，`raw3` 的总量方向是前向精确零空间，5 个登记点在 `1e-7 / 1e-6 / 1e-5` 三档容差下秩一律为 2。这触发后续执行计划的停止条件第 1 条，S1/S2 实现暂停。

本报告的结论**不受影响**：Phase A 只审计条件线性干扰参数块的结构与内层线性求解的数值正确性，与 `raw3` 参数化的零方向是两个独立问题，`mei3_phase_a_structure_supported` 保持有效。完整探针数据与处置候选见 [MEI-3 后续执行计划](tv3_mrs_ei_mei3_execution_plan.md) §11。
