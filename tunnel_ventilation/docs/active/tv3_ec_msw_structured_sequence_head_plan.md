# tv3 EC-MSW：结构化序列表示（E2s / compact builder / E1r attachment / LS 消融）

> 状态（2026-07-17）：**E1d-SB `parity_passed`；attachment `attachment_passed`；E2s-LS 代码已落地。**  
> - 原「纯 TOF-L 加权 LS 声速头」：前置门 E1d-2 **未过门**，不得单独启动。  
> - E1d-SB / attachment 正式过门。  
> - P1 消融：`e1d_sb_cal_plus_corr_psr_snr_ls_v1`（**保留**帧级 SNR，仅 **追加** SNR 加权闭式 LS 标量）。  
> - `e2_allowed=false`。B7 仍是默认 RawDSP 头。

## 1. 证据链

| 阶段 | 结论 |
| --- | --- |
| E1 | learned encoder 丢失绝对峰位 → `frame_fidelity_failed` |
| E1r | 峰位补回，但 LMM sequence parity 失败 |
| E1d-3 / E1d-SB | 校准栈 + SNR → compact 过门 |
| E1r-attach | 冻结 frame + e1d_sb 序列 → `attachment_passed` |
| E2s-LS | 在 e1d_sb 上 **additive** SNR 加权 LS；正式待跑 |

## 2. P0 / P0b（已完成正式）

见 `e1d_sb_s20260704/`、`e1r_attach_e1d_sb_s20260704/`。

## 3. P1 — SNR 加权闭式 LS 消融（代码已落地）

**规则**：不得删掉 `ultrasonic_snr_db`；仅追加

- `ultrasonic_tof_l_m_intercept_snr_weighted_ls_s`
- `ultrasonic_sound_speed_snr_weighted_ls_m_per_s`

权重默认 `amplitude`：`w = 10**(snr_db/20)`，在 steady（及 accepted）帧上对 `tof ≈ a + b·L_m` 做加权闭式 OLS，`c=1/b`。

```powershell
python scripts/run_ec_msw_e1d_sb_ls.py --config configs/tv3_ec_msw_e1d_sb_ls_smoke.json
python scripts/run_ec_msw_e1d_sb_ls.py --config configs/tv3_ec_msw_e1d_sb_ls.json
```

正式产物：`outputs/tv3_ec_msw/e1d_sb_ls_s20260704/`（不得覆盖 `e1d_sb_*` / `e1r_*` / `attach_*`）。

**门禁**：

- formal 需要 `attachment_verdict_path` 且 `status=attachment_passed`
- 相对 B1 非劣（同 E1d-SB 门）
- 写出相对 `e1d_sb` baseline 的 ΔR²（信息项，非 E2 解锁）
- `verdict.e2_allowed=false`；`snr_retained=true`

**实现**：

- `tv3/ml/raw_dsp_features.py` → `fit_tof_vs_path_length_snr_weighted`
- `tv3/ml/e1d_sb_features.py` → `build_e1d_sb_ls_feature_matrix`
- `tv3/dl/evaluation/ec_msw_e1d_sb_ls_audit.py`
- `tests/test_ec_msw_e1d_sb_ls.py`

## 4. 硬边界

1. 不创造新物理信息源；上限逼近 B1 O₂ ≈ `0.4`。
2. 不开 E2 FiLM/attention/MoE。
3. 不把 full B1 包装成端到端改进。
4. 不覆盖既有 E1/E1r/E1d/E1d-SB/attachment 产物。
5. B7 仍是默认部署头。
6. 「可部署联合系统」（e1d_sb 接入推理链路）需另立计划，不在本消融内。
