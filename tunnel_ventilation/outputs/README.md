# tv3 运行产物索引

> 整理日期：2026-07-14。正式结论以 clean `tv3-formal-6000` 为准；路径与 [掘进通风项目记忆库](../docs/掘进通风项目记忆库.md) §七 对齐。

## 顶层结构

| 目录 | 用途 | 状态 |
| --- | --- | --- |
| `tv3_d0/` | D0 六组特征拆分（Ridge，6000） | 正式 baseline |
| `tv3_d2/` | D2 TOF-PhaseNet 失败证据 | 已停止 |
| `tv3_d2b/` | RawDSP fidelity、B1 reference、B6 MLP | 当前主线 |
| `tv3_r5/` | TabPFN、R5 MLP、R5-T | 正式 / 失败证据 |
| `tv3_r7/` | ExtraTrees observed | 失败证据 |
| `tv3_rocket/` | R0 / R1a / R1b 固定特征回归 | 正式 |
| `tv3_r5t_b6_multiseed/` | R5-T / B6 三新增 seed 稳定性复核 | ✅ `stable_pass` |
| `tv3_b7_protocol/` | B7 repeated split + 双 OOD selector 协议 | ✅ `protocol_pass` |
| `tv3_module_c_grouped_bottleneck/` | 模块 C 分组 bottleneck C1 / C2 协议 | ❌ 24 条正式矩阵 `grouped_failed` |
| `tv3_baseline_freeze/` | B1/B7 配置、结果、split、RawDSP 与环境 hash 冻结 | ✅ `frozen` |
| `tv3_identifiability/` | 冻结 v1 单向 TOF 的敏感度、Fisher 与误差预算审计 | ✅ `audit=passed`；P90=`0.4 vol%`、nuisance=`50%`、拒绝率=`5%` 均失败，flow 未表示，verdict=`information_source_upgrade_required` |
| `tv3_ec_msw/` | EC-MSW-GatedNet 分阶段实验 | E1 fail；E1r frame pass / B1 parity fail；E1d 管线已落地（正式待跑）；E2 禁止 |
| `summary/` | 跨运行汇总产物 | 汇总索引 |
| `archive/` | 调试、本地 600、历史 DL、已取代子运行 | 仅追溯 |

新实验默认写入对应顶层目录；调试与 smoke 直接写入 `archive/debug/`。

`tv3_ec_msw/e1_s20260704/` 是旧 E1 的 `frame_fidelity_failed` 证据。`e1r_s20260704/` 是 E1r clean 6000 正式证据：53 epochs 早停、最佳 epoch 41；frame fidelity 三 split 全部通过，但冻结 sequence embedding 相对 B1 的 O₂ ΔR²为 `-0.4118 / -0.4461 / -0.3689`，最终 verdict=`b1_parity_failed`、`e2_allowed=false`。`e1d_smoke_s20260704/` 是 E1d 冻结表示诊断链路产物（非正式）；正式 E1d 目标目录为 `e1d_s20260704/`。`e1_smoke_s20260704/`、`e1r_smoke_s20260704/` 与 `e1r_preflight_s20260704/` 只作链路追溯。

## 正式产物（configs 与记忆库引用）

| 路径 | 说明 |
| --- | --- |
| `tv3_d0/observed_ridge/metrics.json` | D0-observed 测量级线性 baseline |
| `tv3_d0/oracle_ridge/metrics.json` | oracle 上限（不可部署） |
| `tv3_d2/tof_phasenet_s20260704/` | D2 正式训练失败记录 |
| `tv3_d2b/raw_dsp_frame_fidelity/metrics.json` | 帧级 fidelity 审计（passed） |
| `tv3_d2b/raw_dsp_ridge_provenance/metrics.json` | D2b B1 parity reference |
| `tv3_d2b/raw_dsp_mlp_target_scaled_v2/metrics.json` | B6 单 seed 正式结果 |
| `tv3_r5t_b6_multiseed/summary.json` | R5-T / B6 多 seed 稳定性汇总 |
| `tv3_b7_protocol/split_metrics.json` | B7 36 条 B1 / B7 配对行、审计与 `protocol_pass` 判定 |
| `tv3_b7_protocol/result_matrix.md` | B7 的 R/L/S-Y/S-L × split seed × training seed 结果矩阵 |
| `tv3_module_c_grouped_bottleneck/split_metrics.json` | C1 / C2 各 12 条配对行与 `grouped_failed` 判定 |
| `tv3_module_c_grouped_bottleneck/result_matrix.md` | 模块 C 的逐行 C0 / C1 / C2 对照 |
| `tv3_baseline_freeze/manifest.json` | B1/B7 唯一比较基线及其完整 provenance |
| `tv3_baseline_freeze/verdict.json` | 冻结基线 `frozen` 判定 |
| `tv3_r5/tabpfn_observed/metrics.json` | R5' TabPFN 上限探针 |
| `tv3_r5/mlp_observed/metrics.json` | R5 默认 MLP 失败证据 |
| `tv3_r5/mlp_observed_target_scaled/metrics.json` | R5-T 正式通过 |
| `tv3_r7/extratrees_observed/metrics.json` | R7 过拟合失败证据 |
| `tv3_rocket/r0|r1a|r1b/metrics.json` | Rocket 固定特征回归 |
| `tv3_identifiability/verdict.json` | 单向 TOF 可辨识性正式 verdict |
| `tv3_identifiability/nuisance_fraction_summary.csv` | 窄窗口内每个已声明 nuisance 情景相对 `0.8 vol%` 信号的最坏 P90 比例 |
| `tv3_ec_msw/e1_s20260704/metrics.json` | EC-MSW E1 正式训练曲线与神经头指标 |
| `tv3_ec_msw/e1_s20260704/audit/verdict.json` | EC-MSW E1 正式 `frame_fidelity_failed` 判定与 E2 门 |
| `tv3_ec_msw/e1_s20260704/audit/frame_fidelity.json` | learned frame embedding 的正式 peak-index fidelity 失败证据 |
| `tv3_ec_msw/e1_s20260704/audit/b1_parity.json` | 冻结 sequence embedding 相对 B1 的非劣审计 |
| `tv3_ec_msw/e1r_s20260704/metrics.json` | EC-MSW E1r 正式训练曲线与神经头指标 |
| `tv3_ec_msw/e1r_s20260704/audit/frame_fidelity.json` | E1r 模板坐标锚点正式 frame fidelity 通过证据 |
| `tv3_ec_msw/e1r_s20260704/audit/b1_parity.json` | E1r 冻结 sequence embedding 的正式非劣失败证据 |
| `tv3_ec_msw/e1r_s20260704/audit/verdict.json` | E1r 正式 `b1_parity_failed` 与 E2 门 |
| `tv3_ec_msw/e1d_smoke_s20260704/verdict.json` | E1d smoke 诊断 verdict（非正式） |
| `tv3_ec_msw/e1d_smoke_s20260704/ablation_table.csv` | E1d smoke 组增量消融表 |
| `tv3_ec_msw/e1d_s20260704/` | E1d clean 6000 正式诊断（待跑） |

## archive 分类

| 子目录 | 内容 | 归档原因 |
| --- | --- | --- |
| `archive/debug/` | `_tmp_*`、`tv3_quick_test`、`tv3_mm_debug` | 临时调试与 smoke |
| `archive/legacy_dl/` | `tv3_tcn_multimodal*`、`tv3_tcn_s42`、`tv3_tcn_mm_s42` | 波形 fusion DL 历史路线（O₂ 未学到） |
| `archive/local_600/` | `tv3_d0_local` | 旧本地 600 序列，slow 含 `V_NDIR_CH4` 污染 |
| `archive/early/` | `tv3_baseline`、`tv3_rocket_smoke`、`spxy_compare`、`tv3_ridge_result.json` | 早期基线与 split 对比 |
| `archive/superseded/` | `tv3_d2b/raw_dsp_ridge`、`raw_dsp_mlp_target_scaled`、`tv3_r5/tabpfn_observed_smoke` | 已被 provenance / v2 / 正式 run 取代 |
| `archive/tv3_d2b.zip` | D2b 产物打包备份 | 与目录内正式文件重复 |

## 路径迁移说明

以下历史文档路径已迁入 `archive/`，查阅时加 `archive/` 前缀：

- `outputs/tv3_tcn_multimodal_v2/s42/` → `outputs/archive/legacy_dl/tv3_tcn_multimodal_v2/s42/`
- `outputs/tv3_d0_local/` → `outputs/archive/local_600/tv3_d0_local/`
- `outputs/tv3_d2b/raw_dsp_ridge/` → `outputs/archive/superseded/tv3_d2b/raw_dsp_ridge/`
- `outputs/tv3_d2b/raw_dsp_mlp_target_scaled/` → `outputs/archive/superseded/tv3_d2b/raw_dsp_mlp_target_scaled/`

configs 中 `output_dir` 未改；重跑实验仍写入原约定路径。仅既有产物做了物理归档。
