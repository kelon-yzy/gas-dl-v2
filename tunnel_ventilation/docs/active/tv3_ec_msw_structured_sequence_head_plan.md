# tv3 EC-MSW：结构化序列表示（E2s / compact builder / E1r attachment / LS 消融）

> 状态（2026-07-17）：**E1d-SB / attachment / E2s-LS 正式均已完成。**  
> - E1d-SB：`parity_passed`。  
> - attachment：`attachment_passed`。  
> - E2s-LS：`ls_ablation_passed`，相对 e1d_sb 的 O₂ ΔR² 仅 `+0.0005～0.001` → **不晋升 LS**。  
> - 下一步（另立计划）：`tv3_ec_msw_e1d_sb_deployable_joint_system_plan.md`（e1d_sb **无 LS** 推理探针）。  
> - `e2_allowed=false`。B7 仍是默认 RawDSP 头。

## 1. 证据链

| 阶段 | 结论 |
| --- | --- |
| E1 | learned encoder 丢失绝对峰位 → `frame_fidelity_failed` |
| E1r | 峰位补回，但 LMM sequence parity 失败 |
| E1d-3 / E1d-SB | 校准栈 + SNR → compact 过门 |
| E1r-attach | 冻结 frame + e1d_sb 序列 → `attachment_passed` |
| E2s-LS | additive LS 过 B1 门，但对 e1d_sb 几乎无增益 → 不晋升 |
| Deploy-joint | 下一步：e1d_sb（无 LS）接入推理探针 |

## 2. P0 / P0b（已完成正式）

见 `e1d_sb_s20260704/`、`e1r_attach_e1d_sb_s20260704/`。

## 3. P1 — SNR 加权闭式 LS 消融（正式完成）

```powershell
python scripts/run_ec_msw_e1d_sb_ls.py --config configs/tv3_ec_msw_e1d_sb_ls.json
```

正式：`outputs/tv3_ec_msw/e1d_sb_ls_s20260704/` → `ls_ablation_passed`；`snr_retained=true`；`e2_allowed=false`。

| split | O₂ R² | Δ vs B1 | ΔO₂ vs e1d_sb |
| --- | --- | --- | --- |
| val | 0.394 | −0.034 | +0.00057 |
| test | 0.453 | −0.026 | +0.00051 |
| extrapolation | 0.370 | +0.0003 | +0.00101 |

**结论**：LS 不伤门，但不带来实质新信息；默认序列表示仍用 **无 LS 的 e1d_sb**。

## 4. P2 — 可部署联合系统（另立计划）

见 [`tv3_ec_msw_e1d_sb_deployable_joint_system_plan.md`](tv3_ec_msw_e1d_sb_deployable_joint_system_plan.md)。

## 5. 硬边界

1. 不创造新物理信息源；上限逼近 B1 O₂ ≈ `0.4`。
2. 不开 E2 FiLM/attention/MoE。
3. 不把 full B1 包装成端到端改进。
4. 不覆盖既有正式证据目录。
5. B7 仍是默认部署头。
6. 不晋升 LS 变体。
