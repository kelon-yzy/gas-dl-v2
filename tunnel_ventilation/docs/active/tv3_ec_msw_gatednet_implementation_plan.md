# tv3 EC-MSW-GatedNet 实施计划

> 状态：**E1d clean 6000 正式完成（2026-07-17）**：`verdict=minimal_deployable_set_found`，compact=`cal_plus_corr_psr_snr`；`continue_structured_builder=true`，`e2_allowed=false`。E1r 正式 frame pass / B1 parity fail 不变。
> 基线边界：B7 仍是冻结默认 RawDSP 头，identifiability v1 的 `information_source_upgrade_required` verdict 不变。
> 下一步：实现 `cal_plus_corr_psr_snr` 语义的可部署结构化 sequence builder（见 `tv3_ec_msw_structured_sequence_head_plan.md`）。纯 TOF-L LS 头因 E1d-2 失败不得单独启动；FiLM/attention/MoE 仍禁止。

## 1. 范围与不变量

本计划把 `端到端波形动态门控组分反演框架与文献证据.md` 转为独立实验线，不改写 B7、D2b 和 identifiability v1 产物。所有阶段保持以下不变量：

- 输出为 `raw3`，`out_dim=3`；N₂ 不由闭包回填，`sum_abs_error` 只监控。
- split 沿用 benchmark 的 mixture 级正式划分；不得按 frame 重拆。
- 模型输入只含可部署波形和慢通道；oracle TOF、真实声速、真实衰减与标签不得进入 encoder、head 或后续 gate。
- E1 未通过前不实现 FiLM、attention 或 MoE。

## 2. P0 数据契约

| 名称 | E1 形状 | 来源与约束 |
| --- | --- | --- |
| `waveform` | `(B,T,5000)` | `ultrasonic` 解量化后逐帧归一化；绝对幅度由下列统计保留 |
| `slow` | `(B,T,9)` | 7 个可部署慢通道加 `log_std`、`log_max_abs`；使用 train scaler |
| `label` | `(B,3)` | `x_CO2`、`x_O2`、`x_N2`，只作为 loss target |
| `frame_embedding` | `(B,T,64)` | 共享多尺度波形 encoder 输出 |
| `prediction` | `(B,3)` | 独立 raw3 线性输出 |

数据通路（工程，非算法门）：正式 E1/E1r 配置默认 `waveform_preprocess: "gpu"`——DataLoader 只搬运 int16 与 scale，在训练设备上完成 dequant、归一化前 stats 与 z-score，再组装为上表 NTC 输入。数值语义与 `cpu` 路径对齐；审计读 `run_config.waveform_preprocess`。细节见 [server_training_guide.md §4.5](../operations/server_training_guide.md#45-波形数据通路-waveform_preprocessp1-吞吐)。

E1 不读取 `aux_target_arrays`。真实 TOF 与 peak index 仅允许由离线审计程序读取，用于 frame fidelity 评价，不作为训练输入或 loss target。

P2 及以后预留但在 E1 禁用的契约如下：

| 名称 | 形状 | 允许来源 |
| --- | --- | --- |
| `environment_token` | `(B,T,5)` | `T_C`、`P_MPa`、`H_RH`、`L_m`、`piston_position_m` |
| `quality_token` | `(B,T,2)` | 直接由当前帧 raw waveform 计算的 `log_std`、`log_max_abs` |
| `phase_token` | `(B,T,4)` | benchmark 的 baseline、exposure、steady、recovery phase 标识 |
| `window_embedding` | `(B,W,D)` | 同一 encoder 对预注册 phase/window 视图的输出；`W` 由配置固定 |

若后续需要 SNR、TOF quality 或 accepted ratio，必须先实现并审计 raw waveform 到该量的可部署提取器；当前 simulator 同步数组不能直接接入 P2。

## 3. 预注册实验矩阵

| 实验 | 唯一变化 | 进入下一阶段的门 | 失败动作 |
| --- | --- | --- | --- |
| E0 | 冻结 B1/B7 | 已完成 | 不改写基线 |
| E1 | 位置敏感多尺度 encoder 加固定聚合 | frame fidelity 通过且固定头达到下列 B1 parity | 停止组分扩展，修前端 |
| E1d | 冻结 E1r 与 B1，逐组诊断坐标、聚合、校准和质量特征 | ✅ 正式找到 compact 集合 `cal_plus_corr_psr_snr` | 找不到则停止 learned encoder 分支 |
| E1d-SB | 可部署 builder 复现 compact 集合语义 | 三 split B1 parity + 窄窗口 | 不可部署复现则停，保留 B7 |
| E2s-LS | 可选：builder 内 SNR 加权闭式 LS 子算子 | 不得替代 SNR；仅作消融 | E1d-2 已证伪「仅 LS」充分性 |
| E2a | E1 加环境 FiLM | ⛔ `e2_allowed=false` | 不进入 attention |
| E2b | E2a 加 attentive statistics pooling | ⛔ | 保留固定聚合 |
| E3 | 晚期共享 soft gate | ⛔ | 判为路由失败 |
| E4 | component-specific multi-gate 消融 | ⛔ | 保留共享 gate |
| E5 | 混合分支与纯端到端对照 | ⛔ | 保留 RawDSP 锚点 |

## 4. E1 实现与验收

E1 使用共享卷积 stem 和三个 kernel、dilation 分支。每个分支固定输出均值、最大值和激活绝对值关于归一化采样坐标的一阶矩，确保峰位形成显式特征后才压缩波形轴。跨 frame 仅使用固定 `last/mean/max` 聚合。

运行入口：

```powershell
python -m tv3.dl.cli --config configs/tv3_ec_msw_e1_smoke.json
python -m tv3.dl.cli --config configs/tv3_ec_msw_e1.json
```

验收顺序：

1. 单元测试证明位置统计随人工峰位单调移动，encoder 梯度有限且形状正确。
2. smoke 训练完成并写出 checkpoint、run config 和 val/test 指标。
3. 正式运行报告 CO₂、O₂、N₂ 的 MAE、RMSE、R²、bias 与 `sum_abs_error`。
4. 用冻结 B1 协议比较 E1；未达到 parity 时不得开始 E2。
5. 固定 0.8 vol% O₂ 窗口的 MAE、RMSE、P90 和局部斜率必须单列，不能用全局 R² 替代。

B1 parity 沿用冻结门：val、test、extrapolation 的 O₂ R²相对 B1 差距均不超过 `0.05`，CO₂和 N₂ 不出现大于 `0.03` 的系统性退化，并完整报告 `sum_abs_error`、O₂ bins 与 CO₂ bins。smoke 数据只证明链路可运行，不构成 parity 或组分性能证据。

frame fidelity 使用冻结 encoder 的纯 waveform frame embedding：仅在 train frames 上拟合 StandardScaler + RidgeCV peak-index probe，val/test/extrapolation 全帧评价。正式 train 最多按固定 seed `20260704` 无放回抽取 200,000 帧拟合 probe；这只限制拟合成本，不抽样 eval frames。通过门为：

- peak MAE ≤ `0.15 sample`；
- peak P95 绝对误差 ≤ `0.25 sample`；
- peak bias 绝对值 ≤ `0.05 sample`；
- 三个 eval split 必须全部通过。

sequence parity 使用同一冻结 encoder 的固定 `last/mean/max` embedding，另行在 train split 拟合 StandardScaler + RidgeCV；神经网络原输出头不参与 parity 判定。审计入口为：

```powershell
python scripts/audit_ec_msw_e1.py --config configs/tv3_ec_msw_e1_audit_smoke.json
python scripts/audit_ec_msw_e1.py --config configs/tv3_ec_msw_e1_audit.json
```

正式审计生成 `frame_fidelity.json`、`b1_parity.json`、`narrow_o2_windows.csv`、`verdict.json` 和带 SHA-256 provenance 的 `manifest.json`。只有 `verdict.status="e1_pass"` 才令 `e2_allowed=true`。

## 5. 当前完成记录

| 项目 | 状态 | 证据 |
| --- | --- | --- |
| P0 数据与部署输入契约 | ✅ | 本文 §2；E1 明确禁用 simulator 同步辅助数组 |
| E0–E5 门与停止条件 | ✅ | 本文 §3；B1 parity 阈值沿用冻结协议 |
| E1/E1r 位置敏感多尺度 encoder | ✅ | `tv3/dl/models/ec_msw_e1.py`；E1r 新增冻结模板坐标锚点 |
| 模型注册与 CLI 集成 | ✅ | `tv3/dl/models/registry.py`、`tv3/dl/cli.py` |
| E1/E1r 正式、smoke 与 preflight 配置 | ✅ | `configs/tv3_ec_msw_e1*.json`、`configs/tv3_ec_msw_e1r*.json`；正式配置含 `waveform_preprocess: "gpu"` |
| 波形设备侧预处理（P1 吞吐） | ✅ | `tv3/dl/data/waveform_preprocess.py`；Dataset/CLI/Trainer/E1 审计联通 |
| 新增及相关回归测试 | ✅ | 含 `test_waveform_device_preprocess.py`；E1/D2/归一化相关套件通过 |
| 1 epoch 本地 smoke | ✅ | checkpoint、run config、val/test metrics 已生成；不作性能证据 |
| 独立 fidelity/parity 审计器 | ✅ | train-only probes、固定窄窗、verdict 与 provenance 已实现 |
| 本地 smoke 审计 | ✅ 链路 / ❌ 正式门 | `frame_fidelity_failed`，峰位 P95 约 90–141 samples；`e2_allowed=false`，不作正式性能结论 |
| clean 6000 正式训练 | ✅ | 80 epochs；最佳 epoch 72，val loss `0.7272`；约 `1.57 h` |
| clean 6000 frame fidelity | ❌ | val/test/extrap peak MAE `71.19 / 71.87 / 72.33 samples`，P95 `154.96 / 154.97 / 154.83 samples` |
| clean 6000 B1 parity | ❌ | O₂ ΔR² `-0.4697 / -0.5052 / -0.4590`；N₂亦未通过，只有 CO₂通过 |
| 固定 O₂ 窄窗口 | ❌ | 边缘 MAE `1.16–1.21 vol.%`，局部斜率 `-0.168–0.088`，输出向均值收缩 |
| 正式 provenance | ✅ | config、checkpoint、run config、B1 reference 的 SHA-256 与 manifest 完全一致 |
| E2 FiLM/attention | ⛔ | `e2_allowed=false`；不得启动 |
| E1r 前端修复 | ✅ | train-only baseline median 模板匹配滤波；绝对峰位坐标绕过 learned projection |
| E1r smoke frame fidelity | ✅ | val/test/extrap MAE `0.00526 / 0.00573 / 0.00455`，P95 `0.01045 / 0.00819 / 0.01042 sample` |
| clean 6000 E1r 训练 | ✅ | 53/80 epochs 早停；最佳 epoch 41，val loss `0.79995`，约 `1.05 h` |
| clean 6000 E1r frame fidelity | ✅ | MAE `0.03717 / 0.03746 / 0.03774 sample`，P95 `0.08643 / 0.08643 / 0.08716 sample` |
| clean 6000 E1r B1 parity | ❌ | O₂ ΔR² `-0.4118 / -0.4461 / -0.3689`；CO₂、N₂亦未通过 |
| clean 6000 E1r verdict | ❌ | `b1_parity_failed`、`e2_allowed=false` |
| E1d 诊断管线（代码 / 配置 / 测试） | ✅ | `tv3/dl/evaluation/ec_msw_e1d_diagnosis.py`、`scripts/run_ec_msw_e1d.py`、`configs/tv3_ec_msw_e1d*.json`；`tests/test_ec_msw_e1d.py` 12 passed |
| E1d smoke7 链路 | ✅ 链路 / ❌ 非正式 | `outputs/tv3_ec_msw/e1d_smoke_s20260704/`；n=16 不作 parity 结论 |
| clean 6000 正式 E1d | ✅ | `outputs/tv3_ec_msw/e1d_s20260704/`；`minimal_deployable_set_found`；compact=`cal_plus_corr_psr_snr` |
| E1d-SB compact builder | ✅ 代码 / ▶️ 正式审计 | `e1d_sb_cal_plus_corr_psr_snr_v1`；`tests/test_ec_msw_e1d_sb.py` 8 passed；正式见 `configs/tv3_ec_msw_e1d_sb.json` |

E1 与 E1r 正式结果分别位于 `outputs/tv3_ec_msw/e1_s20260704/` 和 `outputs/tv3_ec_msw/e1r_s20260704/`。E1r audit manifest 已固定训练配置、checkpoint、run config 与 B1 reference 的 SHA-256；模型输入排除了 peak/TOF 真值、真实声速、真实衰减与组分标签。

## 6. 正式失败分析与修复门

本次训练过程稳定：80 epochs 完整结束，验证 loss 在 epoch 72 达到最优，最终验证 loss 仅比最优高 `1.51%`，train/val 间隙较小。train frame probe 的 peak MAE/P95 为 `70.92 / 154.49 samples`，与三个评价 split 几乎一致，因此失败不是 probe 过拟合、split 漂移或 checkpoint 选择错误。

冻结 sequence Ridge 对 CO₂ 有改善，但 O₂ R²在三个 split 均为负，N₂也显著落后 B1。B1 的 O₂ R²仍为 `0.4280 / 0.4786 / 0.3695`，说明 RawDSP 能恢复部分 O₂信息，而当前 learned encoder 没有保留相应峰位 / TOF 表示。按预注册失败动作，后续只允许修波形前端，不允许通过调整门限、延长同结构训练或添加 FiLM、attention、MoE 绕过 E1。

修复后的验收顺序保持不变：

1. 在波形轴下采样前保留由当前 raw waveform 可部署计算的位置坐标或高分辨率位置特征；不得把 oracle peak/TOF 作为部署输入。
2. 先运行冻结 frame peak-index probe，三个 split 全部通过 fidelity 门后再继续。
3. 再运行冻结 sequence Ridge 与 B1 parity；O₂、CO₂、N₂全部通过非劣门后，才允许恢复 E2 讨论。
4. 原神经头的正式指标需补齐 component bias；`sum_abs_error` 继续只监控，不进行 N₂闭包回填。

## 7. E1r 实施记录

E1r 新增 `MatchedFilterPeakCoordinate`，输入只包含当前帧归一化 raw waveform 与 RawDSP 已冻结的 train-only baseline median 模板。模板相关峰位除以 `waveform_length-1` 后成为 frame embedding 第一维；其余维度仍由原多尺度卷积分支生成。坐标不经过 learned projection，因此组分损失不能再次把它压掉。模板在 CLI 解析时转为 float32 数组并写入 run config，checkpoint 内保存同一冻结 buffer 与 digest。

新增配置：

```powershell
python -m tv3.dl.cli --config configs/tv3_ec_msw_e1r_smoke.json
python scripts/audit_ec_msw_e1.py --config configs/tv3_ec_msw_e1r_audit_smoke.json

python -m tv3.dl.cli --config configs/tv3_ec_msw_e1r.json --epochs 1 --output-dir outputs/tv3_ec_msw/e1r_preflight_s20260704
python scripts/audit_ec_msw_e1.py --config configs/tv3_ec_msw_e1r_preflight_audit.json

python -m tv3.dl.cli --config configs/tv3_ec_msw_e1r.json
python scripts/audit_ec_msw_e1.py --config configs/tv3_ec_msw_e1r_audit.json
```

smoke 与 1-epoch preflight 只保留为链路证据，不参与正式 parity 结论。clean 6000 E1r 已按预注册顺序完成，原始产物不得由后续诊断覆盖。

clean 6000 E1r 已按上述顺序完成。正式 frame fidelity 通过，但冻结 sequence embedding 的 val/test/extrapolation O₂ R²仅 `0.0162 / 0.0325 / 0.0006`，相对 B1 下降 `0.4118 / 0.4461 / 0.3689`；CO₂和 N₂也未过门。神经头 O₂ R²为 `-0.0089 / -0.0097 / -0.0165`，且整个训练的最佳 val O₂ R²仅 `0.0055`。因此不再把问题表述为峰位坐标丢失，而是当前 sequence representation 信息不足。

## 8. E1d 冻结表示诊断

E1d 的目标是定位 E1r 与 B1 之间的信息缺口，不是构造第二套 RawDSP 或提前实现 E2。所有实验使用冻结 split、train-only `StandardScaler + RidgeCV` 和现有 parity 门。慢通道固定为完整 B1 窗口契约，仅消融超声 / RawDSP 导出量，避免把 slow 变化误判为波形表示收益。

| 阶段 | 输入对照 | 要回答的问题 | 产物 |
| --- | --- | --- | --- |
| E1d-0 | 完整 B1 RawDSP | reference、feature names、split 和指标能否原样复现 | 冻结 manifest 与正对照指标 |
| E1d-1 | 冻结 E1r sequence embedding 与实际 matched peak；再对照 RawDSP peak 的 `last/mean/max`、七统计与四 phase 分窗 | learned embedding、固定聚合或 peak 语义在哪一层丢失信息 | 逐 split 指标和 ΔR² |
| E1d-2 | 在 phase 峰位集合上逐组加 delay calibration、corrected TOF、TOF-L intercept/slope、sound speed | B1 的 O₂能力是否来自序列内校准与声程解耦 | 组增量消融表 |
| E1d-3 | 再逐组加 corr peak、PSR、SNR、peak width、quality/accepted | 质量筛选是否是缺失信息源 | 组增量消融表 |

预注册 feature set 名称见 `default_e1d_specs()`：真实 E1r 对照 `e1r_sequence_embedding`、`e1r_peak_lmm`、`e1r_peak_b1_windows`；RawDSP 对照 `full_b1`、`peak_lmm`、`peak_stats7`、`peak_stats7_phase`、`peak_b1_windows`；校准组 `peak_phase_plus_*`、质量组 `cal_plus_*`。

运行入口：

```powershell
python scripts/run_ec_msw_e1d.py --config configs/tv3_ec_msw_e1d_smoke.json
python scripts/run_ec_msw_e1d.py --config configs/tv3_ec_msw_e1d.json
```

正式配置要求：

1. `data/tv3-formal-6000` 完整可用；
2. `features/raw_dsp/raw_dsp_frame_v1/` 为 `train_baseline_median`、`complete_dataset=true`，且含帧级与序列级数组（含 delay / TOF-L / corrected TOF / PSR / width）；
3. `training_run_dir` 指向冻结 E1r 正式 run，checkpoint 的模板 digest 必须与 RawDSP cache 一致；
4. `b1_reference_metrics` 指向冻结 B1 provenance metrics；
5. `eval_splits` 必须精确为 val/test/extrapolation；输出目录不得覆盖既有 E1 / E1r 产物。

产物：`manifest.json`、`feature_sets.json`、`summary.json`、`ablation_table.csv`、`narrow_o2_windows.csv`、`verdict.json`。`verdict.e2_allowed` 恒为 `false`。

执行约束：

1. full B1 只作正对照，不得将完整 RawDSP 拼接后命名为端到端改进。
2. full B1 每次正式 run 重建，并须以 R²绝对容差 `1e-6` 在完整三 split 复现冻结 reference；否则 verdict=`positive_control_failed`，停止解释其他候选。
3. 每一步都报告 val/test/extrapolation 的三组分 R²、MAE、bias、`sum_abs_error` 与固定 O₂窄窗口；缺失任一 split 直接失败。
4. 扣除冻结 slow 后的诊断特征数 ≤ full B1 诊断块一半，且通过 O₂ `-0.05`、CO₂/N₂ `-0.03` 非劣门后，才能实现新的单一结构化 sequence builder。
5. 若只有接近完整 B1 的集合通过，或改善不能同步到三个 split，则停止 EC-MSW learned encoder 分支，保留 B7，E2不启动；smoke verdict 固定为 `smoke_only`。

### 8.1 本地完成记录（2026-07-14）

- 代码与配置已合入；`tests/test_ec_msw_e1d.py` 12 passed；真实 E1r smoke checkpoint 的 embedding/coordinate 提取集成通过。
- smoke7 写出 `outputs/tv3_ec_msw/e1d_smoke_s20260704/`，链路完整；n=16 的 R²不可作正式判断。
- 本机 `data/tv3-formal-6000` 仅有 RawDSP manifest、缺少数据集主体与特征数组，正式 E1d 必须在服务器完整数据上执行。

### 8.2 clean 6000 正式 E1d 结论（2026-07-17）

服务器完整 RawDSP cache + 冻结 E1r checkpoint 上执行 `configs/tv3_ec_msw_e1d.json`，产物 `outputs/tv3_ec_msw/e1d_s20260704/`。

| 项 | 结果 |
| --- | --- |
| 正对照 | `full_b1` 三 split ΔR²=`0`（容差 `1e-6`）通过 |
| E1d-1 | E1r embedding / peak 聚合全部失败；O₂ embedding `0.014 / 0.032 / 0.000` |
| E1d-2 | 校准满栈（含 TOF-L / sound speed）仍失败；O₂ 停留在约 `0.13–0.21` |
| E1d-3 | 加入 SNR 后过门：`cal_plus_corr_psr_snr` O₂ `0.393 / 0.453 / 0.369` |
| verdict | `minimal_deployable_set_found`；`continue_structured_builder=true`；`e2_allowed=false` |
| compact | `cal_plus_corr_psr_snr`、`cal_plus_quality_width`（首选前者） |

**下一步**：实现可部署 E1d-SB builder（校准栈 + corr/PSR/SNR），重过 parity；不得启动纯 TOF-L LS 头，不得启动 E2 FiLM/attention。细节见 `tv3_ec_msw_structured_sequence_head_plan.md`。