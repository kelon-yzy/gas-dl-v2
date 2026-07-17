# tv3 EC-MSW：e1d_sb 可部署联合系统实施计划

> 状态（2026-07-17）：**D1 代码已落地；正式 `deploy_probe_*` 待跑。**  
> 前置门：E1d-SB `parity_passed`、E1r attachment `attachment_passed`、E2s-LS `ls_ablation_passed` 但 ΔR² 可忽略 → **不晋升 LS**。  
> 目标：把 **`e1d_sb_cal_plus_corr_psr_snr_v1`（无 LS）** 接到可审计的推理链路。  
> 硬边界：`e2_allowed=false`；B7 仍是默认 RawDSP 头；本系统是结构化 compact **旁路**，不是 B7 替换。

## 1. 为什么现在做

| 证据 | 含义 |
| --- | --- |
| `e1d_sb_s20260704/` | compact builder 相对 B1 非劣 |
| `e1r_attach_e1d_sb_s20260704/` | 冻结 E1r 帧保真 + e1d_sb 序列联合过门 |
| `e1d_sb_ls_s20260704/` | additive LS 过 B1 门，但相对 e1d_sb 的 O₂ ΔR² 仅 `+0.0005～0.001` |

现有入口全是训练 / 审计；**没有** waveform→e1d_sb→Ridge→raw3 的独立推理探针。  
attachment 证明的是审计联合，不是部署接线。

## 2. P0 契约（唯一允许的主链）

```text
int16 ultrasonic + scale + slow(7)
  → extract_raw_dsp_sequence（train-baseline median 模板）
  → e1d_sb_cal_plus_corr_psr_snr_v1   # 含 SNR；不含 LS
  → train-only StandardScaler + RidgeCV
  → raw3 (x_CO2, x_O2, x_N2)
```

可选旁路（监控，非主预测）：冻结 E1r 峰位 probe（attachment 已证 MAE≈0.037 sample）。

### 输入 / 输出

| 名称 | 约束 |
| --- | --- |
| waveform | 可部署 ultrasonic int16 + scale；禁止 oracle TOF / true speed / true alpha |
| slow | 正式 7 通道；含 `L_m` |
| template | 仅 train-only baseline median；digest 必须与 RawDSP / E1r 正式 digest 一致或显式记录 |
| prediction | `raw3`，`out_dim=3`；`sum_abs_error` 只监控 |
| feature_builder | 固定 `e1d_sb_cal_plus_corr_psr_snr_v1`；`ls_promoted=false` |

## 3. 分阶段任务

### D0 — 计划与门禁冻结（本文）

- 固定 builder、头、split、parity 门、产物目录命名。
- 明确：不晋升 LS；不开 E2；不替换 B7。

### D1 — 推理探针（代码已落地；正式跑待服务器）

| 文件 | 作用 |
| --- | --- |
| `tv3/ml/e1d_sb_inference.py` | fit / artifact 导出 / frozen predict（无 LS） |
| `tv3/dl/evaluation/ec_msw_e1d_sb_deploy_probe.py` | 对齐审计 + B1 非劣 + 前置门 + 产物 |
| `scripts/probe_ec_msw_e1d_sb_inference.py` | CLI |
| `configs/tv3_ec_msw_e1d_sb_deploy_probe_smoke.json` | smoke |
| `configs/tv3_ec_msw_e1d_sb_deploy_probe.json` | formal |
| `tests/test_ec_msw_e1d_sb_inference.py` | 契约与 smoke 链路（7 passed） |

正式产物目录：`outputs/tv3_ec_msw/e1d_sb_deploy_probe_s20260704/`  
（不得覆盖 `e1*` / `e1d*` / `e1d_sb_*` / `attach*` / `e1d_sb_ls_*`）

**D1 验收**：

1. smoke：链路写出 manifest / predictions / verdict=`smoke_only`。
2. formal：`feature_source=waveform` 与 `raw_dsp_cache` 在 **train + val/test/extrapolation** 特征对齐（atol 登记，默认 `1e-5`）。
3. formal：train-only Ridge 在 val/test/extrapolation 相对 B1 非劣（O₂ ≥ −0.05，CO₂/N₂ ≥ −0.03）。
4. `verdict.e2_allowed=false`；`ls_promoted=false`；`default_head_remains=B7`。
5. provenance：dataset、RawDSP template digest、e1d_sb/attachment/LS 前置 SHA、B1 reference SHA。
6. 前置门身份：e1d_sb 须 `parity_passed` + `feature_builder=e1d_sb_cal_plus_corr_psr_snr_v1` + `e2_allowed=false`；attachment 须 `attachment_passed` + 同 builder + `e2_allowed=false` + `frame_fidelity_passed` + `sequence_parity_passed`。

### D2 — 可选：artifact 打包（仅 D1 通过后）

- 固化 train scaler、Ridge 系数、template、builder manifest。
- 仍不宣称现场部署；不替换 B7 默认入口。

## 4. 判定

| verdict | 含义 | 后续 |
| --- | --- | --- |
| `deploy_probe_passed` | waveform 路径 parity 通过且 provenance 完整 | 可进入 D2 打包；仍不升格为默认头 |
| `deploy_probe_failed` | waveform 对齐失败或 B1 非劣失败 | 修提取 / builder；不得开 E2 或换 LS |
| `smoke_only` | 仅链路 | 非正式证据 |
| `gate_blocked` | 缺少 attachment / e1d_sb 前置 | 停止 |

## 5. Non-goals

1. 不晋升 `e1d_sb_cal_plus_corr_psr_snr_ls_v1`。
2. 不删除 `ultrasonic_snr_db`。
3. 不开 FiLM / attentive pooling / MoE / soft gate。
4. 不把 full B1 / 1008 RawDSP 包装成端到端改进。
5. **不替换 B7** 为默认部署头。
6. 不覆盖既有正式证据目录。
7. 不用闭包残差头 / N₂ 回填 / `gas_head` / ILR。
8. 不因接线成功宣称突破声学 O₂ 上限（仍逼近 B1 ≈ 0.4）。
9. 不把本旁路写成掘进通风现场能力（identifiability `information_source_upgrade_required` 不变）。

## 6. 与相邻计划的关系

| 文档 | 关系 |
| --- | --- |
| `tv3_ec_msw_structured_sequence_head_plan.md` | 表示 / LS 消融已完成；联合系统从本文接棒 |
| `tv3_ec_msw_gatednet_implementation_plan.md` | E2 仍禁止；本计划是 E1d-SB 旁路接线，不是 E2 |
| `b7_oof_ridge_residual_mlp_implementation_plan.md` | B7 保持默认头 |
| `tv3_static_air_feasibility_implementation_plan.md` | 现场 / 静止空气可测性独立；本计划不替代 |

## 7. 服务器命令（D1 启动后）

```powershell
python scripts/probe_ec_msw_e1d_sb_inference.py --config configs/tv3_ec_msw_e1d_sb_deploy_probe_smoke.json
python scripts/probe_ec_msw_e1d_sb_inference.py --config configs/tv3_ec_msw_e1d_sb_deploy_probe.json
```

## 8. 当前进度

- [x] 前置：E1d-SB / attachment / LS 正式结论回填
- [x] D0：本文冻结契约与门禁
- [x] D1 代码：inference / deploy_probe / CLI / configs / tests（7 passed）
- [ ] D1 正式：`outputs/tv3_ec_msw/e1d_sb_deploy_probe_s20260704/` → `deploy_probe_*`
- [ ] D2：可选 artifact 打包（仅正式 `deploy_probe_passed` 后）
