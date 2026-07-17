# tv3 EC-MSW：结构化序列表示（E2s / compact builder）

> 状态（2026-07-17）：**E1d 正式已完成；E1d-SB 代码已落地。**  
> - 原「纯 TOF-L 加权 LS 声速头」：前置门 E1d-2 **未过门**，不得单独启动。  
> - E1d-SB：`tv3/ml/e1d_sb_features.py` + `scripts/run_ec_msw_e1d_sb.py`；builder=`e1d_sb_cal_plus_corr_psr_snr_v1`；单元测试 8 passed。  
> - 正式 6000 审计待服务器执行：`python scripts/run_ec_msw_e1d_sb.py --config configs/tv3_ec_msw_e1d_sb.json` → `outputs/tv3_ec_msw/e1d_sb_s20260704/`。  
> - `e2_allowed=false`。B7 仍是默认 RawDSP 头。  
> 正式 E1d 证据：`outputs/tv3_ec_msw/e1d_s20260704/`。

## 1. E1d 之后的问题重定位

E1 → E1r → E1d 证据链：

| 阶段 | 结论 |
| --- | --- |
| E1 | learned encoder 丢失绝对峰位 → `frame_fidelity_failed` |
| E1r | 模板匹配把峰位补回（MAE ≈ `0.037 sample`），但 sequence parity 仍失败 |
| E1d-1 | embedding / peak LMM / phase 窗 O₂ 仍约 `0.00–0.20`，相对 B1 差 `0.24–0.45` |
| E1d-2 | delay、corrected TOF、TOF-L intercept/slope、estimated sound speed **累加后仍不过门**（O₂ ≈ `0.13–0.21`） |
| E1d-3 | 在校准满栈上加入 **`ultrasonic_snr_db`** 后 O₂ 跳到 `0.393 / 0.453 / 0.369`，三 split 过非劣门 |

因此：

1. **「TOF-L 斜率 alone 就是 B1 的 O₂ 充分统计」被证伪。** 原 E2s 方案 A（只换闭式加权 LS 聚合）物理前提不成立。
2. **缺口在「校准栈 + 质量（尤其 SNR）」的联合表示**，不是再堆 FiLM/attention。
3. compact 首选集合是 `cal_plus_corr_psr_snr`（诊断 213 维 ≤ full B1 诊断块一半）；次选 `cal_plus_quality_width`。`cal_plus_quality_full` 过门但非 compact，不得作为部署目标。

`last/mean/max` 仍然表达不出 within-sequence 校准与质量加权结构；下一步是 **显式 builder**，不是通用聚合器自学。

## 2. 文献用法（不变，但角色降级）

| 机制 | 文献 | 在本轮的合法用法 |
| --- | --- | --- |
| 可微 QP / 最小二乘层 | OptNet, arXiv:1703.00443 | 仅作 builder **内可选子算子**的理论背书；不得替代 SNR 特征 |
| Deep Declarative Networks | DDN, arXiv:1909.04866 | 同左；A 有效后再考虑 |
| Attentive Statistics Pooling | Okabe et al. 2018 | 只允许用 raw waveform 导出的 SNR/PSR 生成样本权重，不作主聚合 |
| Set Transformer | Lee et al. 2019 | 消融对照，不作主路径 |

## 3. 下一步实验矩阵（按优先级）

### P0 — E1d-SB：`cal_plus_corr_psr_snr` 可部署 builder（代码已落地，正式审计待跑）

**入口**：

```powershell
python scripts/run_ec_msw_e1d_sb.py --config configs/tv3_ec_msw_e1d_sb_smoke.json
python scripts/run_ec_msw_e1d_sb.py --config configs/tv3_ec_msw_e1d_sb.json
```

**实现**：

- `tv3/ml/e1d_sb_features.py` — `e1d_sb_cal_plus_corr_psr_snr_v1`；`feature_source=raw_dsp_cache|waveform`
- `tv3/dl/evaluation/ec_msw_e1d_sb_audit.py` — train-only Ridge + B1 parity + 窄窗口
- `tests/test_ec_msw_e1d_sb.py` — 与 E1d 矩阵 bit-identical、waveform 对齐、smoke 产物

**目标**：把 E1d 诊断里的 RawDSP 特征语义，改写成「仅依赖 raw waveform + 可部署慢通道」的单一 sequence feature builder，再在冻结 split 上用 train-only `StandardScaler + RidgeCV` 复现 parity。

必须包含（与 `default_e1d_specs()` 中该集合一致的语义，不必逐名字拷贝 B1 内部实现）：

1. peak / phase 统计（与 `peak_stats7_phase` 同级的可部署提取）
2. delay calibration、corrected TOF、TOF-L intercept/slope、estimated sound speed
3. correlation peak、PSR、**SNR(dB)**
4. 完整 B1 慢通道窗口（冻结，不计入 compact 诊断维）

验收：

- val/test/extrapolation 相对冻结 B1：O₂ ΔR² ≥ `-0.05`，CO₂/N₂ ΔR² ≥ `-0.03`
- 固定 0.8 vol% O₂ 窄窗口 MAE/P90/bias/局部斜率单列，对照 `e1d_s20260704/narrow_o2_windows.csv`
- 诊断特征数仍 ≤ full B1 诊断块一半
- 输出独立目录，不得覆盖 `e1_*` / `e1r_*` / `e1d_*`
- 通过后：`continue` 才允许把该 builder 接到 E1r 前端做端到端审计；**仍不打开** `e2_allowed`

失败动作：builder 无法在三 split 复现 compact 集合 → 停止 EC-MSW learned 分支，保留 B7；记录为「诊断特征不可部署复现」。

### P1 — 可选消融：SNR 加权闭式 LS（原方案 A 的降级版）

仅在 P0 builder 已通过或并行做对照时允许：

```
[ â, b̂ ] = argmin Σ wᵢ ( τᵢ − ( a·Lᵢ + b ) )²
wᵢ = f(SNRᵢ, PSRᵢ)   # 由可部署质量量生成，不是 learned attention
```

- 输出 `â`（→ `ĉ`）、`b̂`、加权残差，作为 builder 的额外标量，**不能删掉帧级 SNR 特征只留 LS 标量**（E1d-2 已证明不够）。
- 若「仅 LS 标量 + peak/phase」过不了门，而「LS + 显式 SNR 帧特征」能过，则记录：LS 是压缩，SNR 是必要信息源。

### P2 — 明确不做

- 单独启动原 E2s「只用 TOF-L LS 替换 `last/mean/max`」
- E2a FiLM、E2b attentive pooling、E3 MoE（`e2_allowed=false`）
- 把 `full_b1` / `cal_plus_quality_full` / `b1_arrays_without_tof_observed` 包装成端到端改进
- 声称突破 O₂ R² `0.70`（identifiability v1 上限仍在）

## 4. 硬边界

1. 不创造新物理信息源；最好结果是逼近 B1 O₂ ≈ `0.4`。
2. E1d-2 失败 ⇒ 纯声速 LS 头无启动资格；E1d-3 compact 通过 ⇒ 只启动校准+SNR builder。
3. 沿用冻结 B1 parity 门与 mixture 级 split；`raw3`；`sum_abs_error` 仅监控。
4. provenance 与既有 E1/E1r/E1d 产物不覆盖。

## 5. 与主线计划的关系

- 门禁与命令：`active/tv3_ec_msw_gatednet_implementation_plan.md` §8–§8.2
- 全局事实：`docs/掘进通风项目记忆库.md` §6.6
- 正式数字：`outputs/tv3_ec_msw/e1d_s20260704/{verdict,summary,ablation_table}.json|csv`

## 参考文献

1. Amos B, Kolter JZ. OptNet. *ICML 2017*. arXiv:1703.00443.
2. Gould S, Hartley R, Campbell D. Deep Declarative Networks. *IEEE TPAMI* 2021. arXiv:1909.04866.
3. Pan J, et al. BPQP. 2024. arXiv:2411.19285.

（Set Transformer、Attentive Statistics Pooling 见框架文档参考文献。）
