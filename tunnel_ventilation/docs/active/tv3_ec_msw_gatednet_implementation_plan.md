# tv3 EC-MSW-GatedNet 实施计划

> 状态：**P0 契约已冻结；P1/E1 模型与独立审计链已实现；本地 smoke 审计按正式门清晰失败且仅作链路证据；clean 6000 frame fidelity/B1 parity 待执行；E2 未启动。**
> 基线边界：B7 仍是冻结默认 RawDSP 头，identifiability v1 的 `information_source_upgrade_required` verdict 不变。

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
| E2a | E1 加环境 FiLM | 固定 holdout 的 P90 和 MAE 改善 | 不进入 attention |
| E2b | E2a 加 attentive statistics pooling | test、S-Y、S-L 同步改善 | 保留固定聚合 |
| E3 | 晚期共享 soft gate | 优于等权和单专家且 gate 不塌缩 | 判为路由失败 |
| E4 | component-specific multi-gate 消融 | 跨 seed 稳定优于共享 gate | 保留共享 gate |
| E5 | 混合分支与纯端到端对照 | 纯端到端不劣且 provenance 完整 | 保留 RawDSP 锚点 |

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
| E1 位置敏感多尺度 encoder | ✅ | `tv3/dl/models/ec_msw_e1.py` |
| 模型注册与 CLI 集成 | ✅ | `tv3/dl/models/registry.py`、`tv3/dl/cli.py` |
| 正式与 smoke 配置 | ✅ | `configs/tv3_ec_msw_e1.json`、`configs/tv3_ec_msw_e1_smoke.json` |
| 新增及相关回归测试 | ✅ | E1 与审计共 11 项、D2 与波形归一化 58 项，共 69 项通过 |
| 1 epoch 本地 smoke | ✅ | checkpoint、run config、val/test metrics 已生成；不作性能证据 |
| 独立 fidelity/parity 审计器 | ✅ | train-only probes、固定窄窗、verdict 与 provenance 已实现 |
| 本地 smoke 审计 | ✅ 链路 / ❌ 正式门 | `frame_fidelity_failed`，峰位 P95 约 90–141 samples；`e2_allowed=false`，不作正式性能结论 |
| clean 6000 frame fidelity | ⏳ | 未执行 |
| clean 6000 B1 parity | ⏳ | 未执行；未过门前禁止 E2 |
| E2 FiLM/attention | ⛔ | 未启动，等待 E1 正式门 |

当前本机 `data/tv3-formal-6000/` 只有 RawDSP feature cache，缺少 `sequences/ultrasonic_int16.npy`、`metadata/sequence_ids.npy`，且尚无 `outputs/tv3_ec_msw/e1_s20260704/checkpoint.pt`。因此 clean 6000 训练与正式审计必须在持有完整 benchmark arrays 的服务器执行；不得从 feature cache 或 smoke 产物伪造正式输入。
