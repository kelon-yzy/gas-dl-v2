# tv3 静止空气仿真可辨识性实施计划

> 状态：**当前仿真 P0；真实测量与硬件闭环暂缓**
>
> 目的：在显式固定 `flow=0` 的仿真边界内，系统量化现有单向声学链路对 O₂ 的信息量、测量链扰动敏感度和独立参数 holdout 表现。本计划的任何通过结论都只表示“在已登记仿真分布内成立”，不能外推为真实静止空气、掘进通风现场或安全联锁能力。

## Context

`tv3_identifiability` v1 已完成。它暴露：单向 TOF 的 O₂ / CO₂ / T / L 联合 Fisher 秩为 1，1 K 温度和 3 μs trigger jitter 情景的窄窗口 P90 为 8.85–12.99 vol% O₂；当前数字孪生没有 flow 传播，部署范围 verdict 为 `information_source_upgrade_required`。

实际实验暂时无法进行，因此当前不能用实测风速、真实设备漂移或参考仪表数据替换 v1 假设。后续证据必须严格区分：

- `implemented_physics`：代码已经显式表示且可复现的物理量；
- `literature_bound`：有文献或硬件规格依据、但未经本项目实测校准的参数范围；
- `engineering_scenario`：仅用于敏感性扫描的工程情景，不得写成真实分布；
- `not_represented`：尚未进入仿真器的 nuisance，继续作为阻断项。

双向声学和 flow holdout 不在本计划范围内（已另行立项为 F 线，见 [tv3_bidirectional_ultrasound_implementation_plan.md](tv3_bidirectional_ultrasound_implementation_plan.md)；F0 已冻结，F1 起并行推进但不挤占本线正式排期）。静止空气在本计划中只表示仿真条件 `flow=0`，不再表述为经风速仪核验的真实实验范围。

## Task

### 1. 仿真边界与不变量

1. 新实验必须使用独立的配置、manifest、输出目录和版本标识；不得改写 `tv3_identifiability` v1、B1/B7 或现有 RawDSP 正式产物。
2. `flow=0` 必须在配置和 manifest 中显式登记；它是情景假设，不是由当前波形反演得到的结论。
3. 部署输入边界保持不变：波形、7 个 slow 通道和经配置登记的元数据。true TOF、true sound speed、true alpha 和标签只可用于 oracle、灵敏度或审计。
4. trigger jitter、固定延迟、声程、T/RH/P、SNR 和设备响应的每个范围必须登记来源类型、单位、分布、相关组和是否已由当前仿真器表示。缺少依据时输出 `inconclusive_parameter_bounds`，不使用隐藏默认值。
5. 正式模型仍直接输出 raw3；不使用 N₂ 回填、闭包残差头、ILR/ALR 或新的 E2 动态模块。

### 2. 执行顺序

1. **S0：参数表示与边界登记。** 建立 nuisance registry，核对 flow、湿空气、设备响应、延迟漂移、jitter、声程误差和 SNR 在当前代码中的表示状态。为每个扫描范围记录 `implemented_physics`、`literature_bound` 或 `engineering_scenario` 来源。
2. **S1：静止空气扰动仿真。** 在 `flow=0` 下生成独立设计矩阵，联合扫描组分、T/RH/P、L、固定延迟、trigger jitter、SNR 和可实现的设备 profile。若需要新增湿空气或设备物理，必须创建新 builder / physics version，不能混入冻结 v1。
3. **S2：可辨识性与误差预算。** 先做灵敏度、联合 Fisher、nuisance 边缘化条件检查和 Monte Carlo P90；继续使用 P90 O₂ <= 0.4 vol%、单项 nuisance 比例 <= 50%、拒绝率 <= 5% 三个研究门。该步骤不训练新模型。
4. **S3：独立参数 holdout。** 仅当 S0-S2 的参数来源和数值审计完整时，固定 RawDSP builder，并用冻结 B1 / B7 比较 composition、T/RH/P、L、jitter、SNR 和 device-profile holdout。不得按同一设计矩阵随机拆帧形成伪 OOD。
5. **S4：路线判定。** 汇总窄窗口、worst-group、selector、拒绝率和仿真支持边界；决定继续单向仿真、补充新的物理表示，或维持 `information_source_upgrade_required`。

### 3. 正式产物

建议输出到独立目录 `outputs/tv3_static_air_simulation/`，至少包含：

- `parameter_registry.json`：参数范围、来源类型、表示状态与相关组；
- `design_matrix.csv`：仿真实验矩阵及配置标识；
- `representation_audit.json`：已表示 / 未表示 nuisance；
- `sensitivity_metrics.json`：灵敏度、Fisher、P90 与 nuisance 比例；
- `split_hashes.json`：各参数 holdout 的样本集合与 hash；
- `model_metrics.json`：冻结 B1 / B7 的 ID、各 holdout 与 worst-group 指标；
- `verdict.json`：仿真范围内的最终判定。

### 4. 判定边界

| 结果 | 含义 | 后续动作 |
| --- | --- | --- |
| `static_sim_supported` | 在已登记仿真分布内三项研究门通过，且独立参数 holdout 稳定 | 保留为仿真可行性证据；不升级为真实能力，可继续扩展湿空气或设备物理 |
| `static_sim_information_insufficient` | 仿真范围内仍有 P90、nuisance、秩或拒绝率失败 | 停止增加回归头；定位主导 nuisance 或评估新信息源 |
| `inconclusive_parameter_bounds` | 参数来源、协方差或物理表示不足 | 补充文献边界或仿真物理，不给继续 / 停止结论 |
| `static_sim_audit_failed` | manifest、split、hash、数值稳定性或禁止输入审计失败 | 修仿真与审计链路，不训练模型 |

## Non-goals

- 不宣称完成真实静止空气、真实硬件、Sim2Real 或矿井通风现场验证。
- 不用“仿真中 `flow=0`”替代真实风速核验，也不据此撤销 v1 的 `information_source_upgrade_required`。
- 不生成 `raw_dsp_bidirectional_v1`，不建立 flow holdout；双向属独立 F 线计划，不混入本计划产物。
- 不启动 E2 FiLM / attention / MoE，不晋升 LS，不替换 B7。
- 不改变 raw3、`mixture_id`、B1/B7、现有 RawDSP builder 或冻结基线。
