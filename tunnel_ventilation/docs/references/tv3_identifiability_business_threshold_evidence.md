# tv3 可辨识性业务门限与证据

> 记录日期：2026-07-13。用途：为 `tv3_identifiability` 的 O₂ 精度与 nuisance 门限提供可追溯的预注册依据；不是矿用安全仪表认证文件，也不授予该模型安全联锁资格。
>
> **状态更新（2026-07-25）**：`target_p90_o2_error_percent=0.4` 经业务决策**暂缓强制门属性，降级为参考标注**（决策登记见 `../archive/completed/tv3_mrs6_hardware_requirements.md` §0）。本文档的推导链与数值保留不变，作为 0.4 参考线的证据来源；已在强制门下判定的历史 verdict（v1 / F4 / F5-wide / MRS-2）不改写。恢复强制门需业务侧再决策并另行登记。

## 1. 已确定的业务门限

| 配置字段 | 值 | 含义 |
| --- | ---: | --- |
| `target_p90_o2_error_percent` | `0.4` | 窄窗口 O₂ 绝对误差的 P90 不得超过 `0.4 vol% O₂`。 |
| `max_nuisance_fraction_of_signal` | `0.50` | 任一已声明 nuisance 情景的最坏等效 O₂ P90，不得超过 `0.8 vol%` 窄窗口宽度的 50%，即 `0.4 vol% O₂`。 |
| `max_rejection_rate` | `0.05` | 研究可用性门：拒绝率不得超过 5%。不是法规值；被拒绝样本必须交由经认证的现场安全监测链路，不得由该研究模型继续给出连续 O₂ 数值。 |

`max_nuisance_fraction_of_signal` 是小数比例，不是百分数数值；实现要求其取值在 `[0.01, 1.0]`，因此满足“不低于 1%”的约束。联合 P90 与单一 nuisance 比例是两个独立门：前者限制总预算，后者防止单个机制吞没窄窗口信号。5% 拒绝率是保证研究路线至少保留 95% 可用观测的预注册可用性门，不冒充法规或产品认证指标。

## 2. 证据链与推导

1. **法规安全边界。** [《煤矿安全规程》本地归档](煤矿安全规程.pdf) PDF 第 60 页第一百五十六条规定：采掘工作面的进风流中氧气浓度不低于 `20%`。这给出业务语义的硬下限，但没有直接规定模型的 P90 误差。
2. **声学精度可行性参照。** Fukuoka 等在近大气、二元、温湿度补偿的超声 TOF 实验中报告补偿后 O₂ 测量误差约 `0.4%` 或更低，[Review of Scientific Instruments 94, 035001 (2023)](https://doi.org/10.1063/5.0113877)。该条件远小于 tv3 的三元、单向 TOF、未表示 flow 场景，故它只能作为严格的研究目标量级，不能迁移为部署精度承诺。
3. **工业安全边界。** Willett 的综述指出，矿山等工业安全场景需要可靠测量 O₂ 并对异常状态发出警告，[Sensors 14, 6084–6103 (2014)](https://doi.org/10.3390/s140406084)。这支持将模型输出定位为安全预警研究证据，而非替代经认证的安全监测或闭锁仪表。

推导为：法规下限为 `20 vol%`；以 `20.4 vol%` 作为研究性的预警守护点时，`0.4 vol%` 是需要被 P90 误差覆盖的绝对裕度。同时，预注册窄窗口宽度为 `0.8 vol%`，其一半正好是 `0.4 vol%`，所以关键单一 nuisance 最多可消耗 50% 的窗口信号。这个 50% 门限远高于 1%，但仍阻止“干扰大于有效信号”的连续回归主张。

## 3. v1 的判读边界

当前冻结 v1 只有单向 TOF 独立观测，`O₂ / CO₂ / T / L` 的联合 Fisher 秩为 1，`flow_projection` 是 `not_represented` 且阻止 go verdict。重跑后的窄窗口联合 P90 最大值为 `12.99 vol% O₂`，超过 `0.4 vol%` 门 32.5 倍；温度情景的最坏单项比例为 `5.17`，trigger jitter 为 `15.50`，也均高于 `0.50` 门。

v1 采用 `v1_blocking_nuisance_reject_all`：只要 representation audit 存在 `blocks_go_verdict=true` 的未表示 nuisance，就拒绝全部审计点，不输出连续回归可用性结论。当前 flow 触发此规则，189/189 点被拒绝，观测拒绝率为 `100%`，超过 5% 门。因此正式 verdict 是 `information_source_upgrade_required`。这是对 v1 信息边界的审计结论，不是对声学路线在补齐双向 flow、湿空气和设备物理后的最终否定。

输出中的 `nuisance_fraction_summary.csv` 使用以下定义：

```text
nuisance_fraction_of_signal
  = max_over_narrow_window_points(P90 equivalent O₂ error for one declared scenario)
    / narrow_window_width_percent
```

它只适用于登记为独立标准差的 v1 温度与 trigger jitter 情景，不能把未建模的 RH、设备漂移或 flow 当成已验证的零贡献。完整的连续回归主张仍需：flow 的表示/校正、湿空气和设备机制、独立 calibration split，以及联合协方差依据；拒绝策略和拒绝率已在 v1 中登记并量化。

## 4. 当前执行范围

双向声学和 flow holdout 当前暂停。下一步仅在独立风速核验满足静止条件的受控范围内，校准单向测量链并验证 O₂ 可测性。这个范围收缩不改变本文件的法规、文献和 v1 审计证据：它只是把下一步问题缩小为“静止空气里能否测”，而不是声称已经解决通风现场的 flow 混叠。具体计划见 `docs/active/tv3_static_air_feasibility_implementation_plan.md`。
