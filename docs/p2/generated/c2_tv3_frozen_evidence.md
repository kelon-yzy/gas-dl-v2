# C2 TV3 冻结证据（自动生成）

> 本文件由 `docs/p2/tools/render_c2_frozen_evidence.py` 从冻结 CSV 生成；不手工维护运行数值。

## 输入产物

- table6：`tunnel_ventilation/docs/paper/artifacts/table6_solver_efficiency.csv`，SHA256 `8468c1fce2325945a86a8efe6782f3285c4db4fe9948cf951c69695a22030219`
- table7：`tunnel_ventilation/docs/paper/artifacts/table7_structural_verification.csv`，SHA256 `d3ace599963aee266b7ce89ae6a9665adf323007631c4e3b4a3e85e1306166ce`

## table6_solver_efficiency.csv

| domain | n_resamples | ci_level | relative_improvement_point | relative_improvement_ci_lower | relative_improvement_ci_upper | practical_equivalence_band | clears_band | recomputed_clears_band | S1_p90_abs_err_o2 | S2_p90_abs_err_o2 |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| test | 2000 | 0.95 | 0.0278 | 0.0116 | 0.0735 | 0.02 | False | False | 1.6604 | 1.6142 |
| ood | 2000 | 0.95 | 0.0245 | 0.0074 | 0.0541 | 0.02 | False | False | 0.7161 | 0.6985 |

## 冻结门解释

门的重算规则是 `relative_improvement_ci_lower > practical_equivalence_band`；点估计超过等价带本身不足以通过。
- `test`：95% CI 下界 `0.0116`，等价带 `0.02`，重算 `clears_band=False`。
- `ood`：95% CI 下界 `0.0074`，等价带 `0.02`，重算 `clears_band=False`。

综合 `clears_band`：`False`。该结果只说明冻结观测下的 C2 原收益门证据，不授予 C2 或 C5 授权。

## table7_structural_verification.csv

| check | quantity | value | n_linear_parameters | source |
|---|---|---:|---:|---|
| variable_projection_vs_joint_reference_parameters | max absolute parameter difference | 1.0408340855860843e-17 | 6 | outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260728T080522165154Z_7cd8443230fa/mei3_structure_audit.json |
| variable_projection_vs_joint_reference_residuals | max projected residual difference | 7.105427357601002e-15 | 6 | outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260728T080522165154Z_7cd8443230fa/mei3_structure_audit.json |
| projected_jacobian | max relative error against finite differences | 8.56712878193837e-10 | 6 | outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T081421139186Z_c0ade3f5df14/projected_jacobian_report.json |

## 证据边界

- table7 证明变量投影结构与数值 Jacobian 对照一致，不证明效率收益已过门。
- 当前冻结表只提供 P90、迭代结构对照和 bootstrap；统一 wall-clock、推理延迟、数据效率和增量信息仍须按 P2-08 新协议测量。
- C2 的原门、新 endpoint、非劣带、paired split/seed、硬件和计时口径必须在授权前保持显式可区分。
