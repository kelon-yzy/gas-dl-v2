# tv3 EC-MSW：结构化序列表示（E2s / compact builder / E1r attachment）

> 状态（2026-07-17）：**E1d-SB 正式 `parity_passed`；E1r attachment 代码已落地。**  
> - 原「纯 TOF-L 加权 LS 声速头」：前置门 E1d-2 **未过门**，不得单独启动。  
> - E1d-SB 正式：`outputs/tv3_ec_msw/e1d_sb_s20260704/` → `parity_passed`，`continue_e1r_attachment=true`。  
> - E1r attachment（probe-only）：`scripts/audit_ec_msw_e1r_attachment.py`；正式产物 `outputs/tv3_ec_msw/e1r_attach_e1d_sb_s20260704/`（待服务器执行）。  
> - `e2_allowed=false`。B7 仍是默认 RawDSP 头。

## 1. E1d 之后的问题重定位

E1 → E1r → E1d → E1d-SB 证据链：

| 阶段 | 结论 |
| --- | --- |
| E1 | learned encoder 丢失绝对峰位 → `frame_fidelity_failed` |
| E1r | 模板匹配把峰位补回（MAE ≈ `0.037 sample`），但 LMM sequence parity 仍失败 |
| E1d-1/2 | embedding / peak / TOF-L 校准 alone 未补回 O₂ |
| E1d-3 | 校准栈 + **SNR** → compact `cal_plus_corr_psr_snr` 过门 |
| E1d-SB | 可部署 builder 正式复现同一集合 → `parity_passed` |

## 2. P0 — E1d-SB（已完成正式）

```powershell
python scripts/run_ec_msw_e1d_sb.py --config configs/tv3_ec_msw_e1d_sb.json
```

正式 verdict：`parity_passed`；O₂ R² `0.393 / 0.453 / 0.369`；ΔO₂ `-0.035 / -0.026 / -0.001`。

## 3. P0b — E1r attachment（代码已落地，正式待跑）

**目标**：probe-only 联合审计——冻结 E1r 的 frame fidelity 仍通过，同时用 `e1d_sb_cal_plus_corr_psr_snr_v1` **替换** E1r 的 `last/mean/max` 序列表示做 B1 parity。不重训深网，不开 E2。

```powershell
python scripts/audit_ec_msw_e1r_attachment.py --config configs/tv3_ec_msw_e1r_attachment_smoke.json
python scripts/audit_ec_msw_e1r_attachment.py --config configs/tv3_ec_msw_e1r_attachment.json
```

**实现**：

- `tv3/dl/evaluation/ec_msw_e1r_attachment_audit.py`
- `configs/tv3_ec_msw_e1r_attachment*.json`
- `tests/test_ec_msw_e1r_attachment.py`（6 passed）

**验收**：

- E1r frame fidelity 三 split 过门
- e1d_sb 序列 Ridge 相对 B1 非劣（同 E1d-SB 门）
- 窄窗口写出；独立目录 `e1r_attach_e1d_sb_s20260704/`
- `verdict.e2_allowed=false`；通过则为 `attachment_passed`

## 4. P1 — 可选消融：SNR 加权闭式 LS

仅在 attachment 通过后允许；不得删掉帧级 SNR 只留 LS 标量。

## 5. 硬边界

1. 不创造新物理信息源；上限逼近 B1 O₂ ≈ `0.4`。
2. 不开 E2 FiLM/attention/MoE。
3. 不把 full B1 包装成端到端改进。
4. 不覆盖既有 E1/E1r/E1d/E1d-SB 产物。
