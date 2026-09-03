# 下一轮 A2：Ar–He–CO₂ 动态时间序列仿真与数据分布规划

> 制定日期：2026-08-31 \
> 文档状态：`DESIGN_COMPLETE / A2-DYN-0_R4_PROTOCOL_FROZEN / A2-DYN-1R4_EXECUTED / A2-DYN-1_PHYSICS_VERIFIED / A2-DYN-2R4_PILOT_QUALIFIED / A2-DYN-3_DEVELOPMENT_GENERATED / A2-DYN-3_DIFFICULTY_QUALIFIED / A2-DYN-4_TEST_GENERATED / A2-DYN-4_DATA_FROZEN` \
> 工作包定位：下一轮 A2 的数据与问题重定义子工作包，不新增 A2I 或平行顶层阶段 \
> 上位事实源：[项目总体规划](../../项目总体规划.md) \
> 相关契约：[统一任务与接口契约](01_统一任务与接口契约.md)、[A1 数据与物理规格](05_A1数据与物理规格.md)、[A2H 分步执行计划](09_A2H分步执行计划.md) \
> 传感器证据：[传感器仿真文献调研与 A2-DYN 对比](14_传感器仿真文献调研与A2-DYN对比.md) \
> 声速生成定义：CoolProp HEOS 8.0.0 直接作为 `a2dyn_direct_multifluid_eos_v1` 的合成声速算子；历史 pair-v2 仅保留为失败证据 \
> 历史失败边界：[TQIF 失败归档](12_TQIF新算法设计与A2至A2M验证计划.md)

## 0. 结论与执行摘要

当前 Ar–He–CO₂ 正式 A1、A2、A2H 和 A2M 数据都是 `T=1` 的稳态标量。仓库虽已有 A0 的 32 点一阶响应 smoke，但它只有 3 个配方、固定时间常数和单次纯 Ar 到目标气体的确定性过渡，只证明统一接口能承载时间维，不能作为正式动态 benchmark。

下一轮 A2 在设计新算法前，先增加 `A2-DYN` 动态数据子工作包。它要建立真正随时间演化的气室组成、进气协议、传感器响应、时序噪声和在线前缀评价，而不是把一个稳态值重复 1,200 次。当前 R4 机器协议已冻结，开发数据已生成并通过难度审计，A2-DYN-4 已生成 test 并聚合为 6,300 观测 / 4,410 组的完整数据包且冻结 hash（`DATA_FROZEN`，2026-09-03）；完整基线与时间增量信息门仍未执行。

- 三路低频传感器输出：超声 ToF、热导电压、CO₂ NDIR 电压；
- 共同气室、局部传感器输运、平衡物性、器件或采集、观测扰动五层物理链；
- 超声采用外层过程时间轴与内部短波形采集的双时间尺度，只持久化 ToF 和审计质量量；
- 240 s 序列、5 Hz 采样、`T=1200`（由 pilot 在 1 / 2 / 5 Hz 中选定）；
- baseline、transition、steady、recovery 四阶段；
- 4,410 个唯一 `mixture_id`，6,300 条动态观测；
- train、val、stress_val、test 四个逻辑 split；
- 动力学、进气协议、时序噪声与漂移、环境与标定、联合压力六个数据族；
- 5 s、15 s、30 s、60 s、120 s、150 s 因果前缀评价；
- 一个核心资格门：时间序列基线必须在至少两个早期前缀上稳定优于“只看当前末值”的匹配标量基线，否则关闭动态算法方向。

本规划不恢复、改名或修补 TQIF。TQIF 的 `SCIENTIFIC_FAILURE / ABANDONED` 结论保持不变。`A2-DYN` 通过只表示动态问题具有可学习且未饱和的时序信息，不代表任何新算法已经成立。

`A2-DYN-1` 的两次自建维里声速路径均因压力方向不一致而终止。`A2-DYN-1R4` 已把固定版本 CoolProp HEOS 直接定义为正式声速生成算子：完整 185,436 点与 10,000 个离网点的生成器一致性差值均为 `0`，压力方向不一致为 `0 / 30,906`，恢复、超声多径和 NDIR 设备门均通过。`A2-DYN-2R4` pilot 完成 240 组资格审计，冻结 `5 Hz / 240 s` 与 `US-CHIRP-XCORR-PARABOLIC-1`（`reference_xcorr_parabolic`），当前终态为 `PILOT_QUALIFIED`。HEOS 验证范围仍明确限定为 `generator_consistency`，不宣称独立验证 HEOS。

## 1. Context：现状、根因与边界

### 1.1 当前数据事实

| 数据或代码 | 时间形态 | 当前用途 | 能否作为动态正式证据 |
| --- | --- | --- | --- |
| `configs/data/ar_he_co2_a0_smoke.json` | 32 点，`dt=1 s` | A0 统一接口 smoke | 否；只有 3 个配方，固定一阶时间常数 |
| `src/gf/sim/ar_he_co2.py::build_pilot_record` | 纯 Ar 到目标配方的一阶响应 | 生成 A0 smoke | 否；没有气室状态、协议分布和时序噪声 |
| A1 formal v1 | `T=1` | 稳态数据与强基线 | 否 |
| A2 首轮与 TQIF | `T=1` | 稳态融合机制筛选 | 否 |
| A2H v2 | `T=1`，跨观测扰动 | 噪声、环境、标定和组成压力 | 否；重复观测不是单条序列内动态 |
| A2M formal | `T=1` | 表格架构对照 | 否 |
| tv3 | `T≫1`，阶段化序列和早期窗口 | 掘进通风专用时间序列 | 只借鉴阶段、窗口和审计思想，不 import 代码或沿用主键 |

因此，“A2 是否有时间序列”的准确回答是：正式 A2 没有；只有 A0 接口 smoke 中存在 32 点的简化时序占位。

### 1.2 为什么不能直接扩大 A0 pilot

A0 pilot 的三路输出分别满足固定时间常数的一阶响应：

$$
y_s(t)=y_{s,0}+\left(y_{s,\infty}-y_{s,0}\right)
\left(1-\exp\left(-t/\tau_s\right)\right).
$$

当 `\tau_s` 对所有样本固定、输入始终是同一个 step、没有气室混合和漂移时，整条曲线几乎可由终点幅值唯一确定。这样的时间维没有独立信息，只会把稳态映射展开成高度冗余的序列。若直接据此训练 TCN、GRU 或 Transformer，模型复杂度增加，但研究问题没有实质变化。

正式动态 benchmark 必须同时引入：

1. 可解释的共同气室状态；
2. 不同进气与恢复协议；
3. 分传感器动态滞后；
4. 序列级与时间相关扰动；
5. 只允许使用当前及过去信息的在线前缀任务；
6. 证明轨迹相对当前末值确有增量信息的资格审计。

### 1.3 研究边界

本工作包负责：

- Ar–He–CO₂ 动态气体暴露仿真；
- 动态数据分布、split、schema 和生成器；
- 物理一致性、动态真实性和信息增量审计；
- 标量、统计特征和轻量时序基线；
- 离线整序列与在线因果前缀评价；
- 为后续新算法提供冻结的数据和强基线。

本工作包不负责：

- 重新运行、调参或重命名 TQIF；
- 在数据资格审计前确定新算法结构；
- 声称 sensitivity tier 等于真实硬件误差分布；
- 把 MHz 级超声波形作为主模型输入，或完整持久化每个时刻的高频 ADC；
- 每条训练序列逐点运行 KLM、FEM、CFD 或完整多频声学弛豫谱；
- 传感器整路缺失、未见传感器组合和故障路由；这些仍属于 A4；
- 把当前时序仿真结果回写成既有 A1、A2H 或 A2M 的成绩；
- 新建 A2I、A2T 等顶层阶段。

### 1.4 隔离与访问策略

按当前项目决策，v1 不建设复杂的物理文件隔离、权限令牌、标签解锁服务或专门的访问门测试。允许所有 split 保存在同一聚合数据包，并由 `split` 字段做逻辑过滤。

但以下科学不变量仍必须满足：

- 同一 `mixture_id` 的所有动态观测只能属于一个 split；
- scaler、特征统计和数据驱动阈值只使用 train；
- 模型选择只使用 train、val 和 stress_val；
- test 在 recipe 冻结后统一报告，不用于回调分布、窗口或模型；
- oracle 字段不进入可部署模型输入；
- test 结果一旦用于决策，后续数据或协议修改必须升版。

这是一套轻量、可审计的研究流程约束，不扩展为文件访问控制子系统。

## 2. Task：要回答的科学问题

### 2.1 主问题

> 在目标混合物尚未完全达到稳态、气室交换和传感器响应速度未知或变化时，三路传感器的因果前缀轨迹是否比同一时刻的三个末值标量更有助于提前估计目标 Ar、He、CO₂ 配比？

该问题把“复杂模型是否有用”改写为先可证伪的数据问题。只有前缀轨迹存在稳定增量信息，才值得为时序机制设计新算法。

### 2.2 次问题

1. 早期误差来自混合尚未完成、传感器滞后、噪声，还是标定漂移？
2. 不同模态的响应速度差异能否帮助区分目标幅值与动力学 nuisance？
3. 时序收益只存在于名义 step，还是能迁移到 ramp、延迟起始、短脉冲和不完全恢复？
4. 简单统计特征或轻量 TCN 是否已经吃尽时序信息？
5. 动态数据是否仍保持三组分可辨识，而不是靠不可逆噪声制造难度？
6. 在线预测达到可用误差需要多少秒，推理耗时是否低于数据更新周期？

### 2.3 主任务与标签

v1 主任务保持现有三组分样本级回归，不修改任务语义：

- 输入：从 baseline 开始到当前预测时刻的三路因果时间序列；
- 目标：暴露阶段进气端的固定目标配方 `[x_Ar_pct, x_He_pct, x_CO2_pct]`；
- 单位：mol%；
- 约束：各项非负且总和为 `100±1e-6`；
- 分组：原始 `mixture_id`；
- 主输出：各指定前缀时刻的目标配方预测。

气室内瞬时组成 `x_chamber(t)` 是生成器的特权状态，只用于物理审计和 oracle，不作为 v1 可部署输入。若未来增加逐时刻状态跟踪任务，必须升版统一任务契约，不能静默把样本级标签改成稠密标签。

### 2.4 成功、失败与停止语义

| 终态 | 条件 | 后续 |
| --- | --- | --- |
| `DYNAMIC_QUALIFIED` | 物理、数据和增量信息门均通过 | 冻结数据，开始新算法构思 |
| `TEMPORAL_REDUNDANT` | 动态有效，但轨迹不稳定优于末值 | 不设计复杂时序算法；回到稳态或重新定义物理激励 |
| `DYNAMIC_UNIDENTIFIABLE` | oracle 也无法恢复目标，或压力范围破坏可辨识性 | 缩回扰动并升版数据设计 |
| `PHYSICS_INVALID` | 守恒、单位、稳态一致性或动态响应审计失败 | 修复共享物理实现，不生成正式数据 |
| `BASELINE_SATURATED` | 简单统计或轻量基线接近 oracle，复杂模型缺少余量 | 保留简单方法，不强行构造新算法 |
| `INVALID_PROTOCOL` | split、标签、禁用字段或结果回流违反契约 | 结果作废，修正协议后升版 |

“生成了 6,300 条序列”不是完成条件；只有明确终态和对应证据才算关闭 `A2-DYN`。

## 3. 版本、身份与不变量

### 3.1 计划版本

| 项目 | 计划值 | 说明 |
| --- | --- | --- |
| schema version | `gf-a2-dynamic-data-1` | 动态数据契约 |
| dataset id | `ar_he_co2` | 与主线一致 |
| 计划数据版本前缀 | `gf-a2-dynamic-v1` | 正式生成时追加冻结日期和 revision |
| observation mode | `dynamic_exposure_sequence` | 与稳态重复观测显式区分 |
| target order | `x_Ar_pct, x_He_pct, x_CO2_pct` | 不改变标签顺序 |
| group key | `mixture_id` | 唯一 split 与 bootstrap 单位 |
| row identity | `observation_id` | 只追踪单条序列，不参与分组 |
| canonical timestep | 1,200 | 240 s × 5 Hz |
| sensor count | 3 | v1 始终完整三路 |

机器配置已由 `A2-DYN-0` 的 R4 revision 冻结；声速生成语义由 CoolProp HEOS 直连。本文中的数据规模仍是 v1 计划值，不是已经生成的正式数据 manifest 事实。

### 3.2 永久不变量

1. `mixture_id` 来自配方生成器，绝不从 `observation_id`、数组行号或其他字段推断。
2. 不把 `mixture_id` 回退或重写为 `sequence_id`。
3. 新 benchmark 不生成、不读取、不依赖 `base_condition_id`、`noise_seed_index`、`noise_seed`。
4. 随机状态只登记在数据配置和运行 manifest，不作为模型特征或逐行身份。
5. 热导和 NDIR 平衡物理由 `src/gf/sim/ar_he_co2.py` 提供；A2-DYN 声速由 `src/gf/sim/a2dyn_sound_speed.py` 的 CoolProp HEOS 直连算子提供。
6. pipeline 不复制声速、WMS 热导或 NDIR 方程。
7. 可部署模型不能读取目标、特权动力学参数、clean signal 或气室真实状态。
8. 时间序列按真实时间排序；任何前缀评价都不得读取预测时刻后的数值。
9. v1 不用静默裁剪修复信号越界；越界 profile 直接由审计判定无效。
10. A1、A2H 和 A2M 历史数据与结果保持只读，动态结果使用新目录和新 schema。

### 3.3 来源等级

所有动态参数必须带 `source_level`：

| 等级 | 语义 | 使用规则 |
| --- | --- | --- |
| `shared_physics` | 已在当前 Ar–He–CO₂ 物理链中使用 | 可作为名义公式 |
| `literature_structure` | 调研文献支持的模型结构、变量关系或估计流程 | 可确定结构，不能直接复制单台仪器数值 |
| `literature_anchor` | 文献中的具体仪器参数 | 只能形成有限候选或 fixture，不写成通用硬件真值 |
| `existing_a0_proxy` | A0 smoke 的固定时间常数或响应假设 | 只作历史对照，不进入 v1 正式 profile |
| `a2h_registered_range` | A2H 已登记的环境、噪声或标定档 | 可复用其相对层级 |
| `application_range` | 有明确应用范围依据 | 需在配置注释来源 |
| `sensitivity_tier_1` | 中等敏感性压力，无硬件标定声明 | 只作开发压力 |
| `sensitivity_tier_2` | 较强 OOD 压力 | 必须通过 oracle 与信号边界审计 |
| `hardware_calibrated` | 未来真实实验拟合 | 只有 A5 或独立标定证据可使用 |

报告中不得把 `literature_anchor`、`existing_a0_proxy` 或 `sensitivity_tier_*` 写成实测传感器响应。

## 4. 动态物理链

### 4.1 状态与符号

| 符号 | shape | 含义 |
| --- | --- | --- |
| `x_purge` | 3 | 基线吹扫气体组成，v1 沿用纯 Ar |
| `x_target` | 3 | 本条序列的目标进气组成 |
| `b(t)` | scalar | 进气协议混合系数，范围 `[0,1]` |
| `u(t)` | 3 | 进入气室的瞬时组成 |
| `x_ch(t)` | 3 | 气室内共同真实组成 |
| `x_s(t)` | 3 | 传感器 `s` 的局部声路或小气室组成 |
| `p_s(t)` | scalar 或 vector | 由局部组成得到的平衡物性 |
| `q_s(t)` | sensor-specific | 器件或内部采集状态 |
| `z_s(t)` | scalar | 器件或估计器输出的 clean 低频读数 |
| `y_s(t)` | scalar | 加入标定、漂移、噪声与量化后的观测 |

进气组成定义为：

$$
u(t)=(1-b(t))x_{purge}+b(t)x_{target}.
$$

`b(t)` 由 step、ramp、pulse 等协议决定，但 `x_target` 在一条观测内固定。

### 4.2 共同气室与局部输运

用连续搅拌气室的一阶质量守恒近似：

$$
\frac{d x_{ch}(t)}{dt}=\frac{1}{\tau_{mix}}\left(u(t)-x_{ch}(t)\right),
\qquad \tau_{mix}=\frac{V_{chamber}}{Q}.
$$

当一个采样步内 `u_k` 不变时，使用解析离散更新：

$$
x_{ch,k+1}=u_k+
\left(x_{ch,k}-u_k\right)\exp\left(-\frac{\Delta t}{\tau_{mix}}\right).
$$

选择解析更新而不是一般 Euler 步进，是为了在 `u_k` 和初始状态位于组成单纯形时自然保持非负与闭合。任何浮点误差修正只能是有记录的数值容差检查，不能每步重新归一化来隐藏积分错误。

共同气室只是 well-mixed v1 假设，不再同时代表三台传感器的局部死体积。每路局部组成满足：

$$
\frac{d x_s(t)}{dt}
=\frac{x_{ch}(t)-x_s(t)}{\tau_{transport,s}}.
$$

当一个外层采样步内 `x_ch,k` 不变时，仍使用解析更新。若传感器直接位于共同气室且没有可辨别的局部换气体积，对应配置字段显式设置为零，此时 `x_s=x_ch`；不能用一个不明来源的固定 `tau_s` 同时表示局部换气、换能器响应和输出平均。机器字段统一使用 `tau_transport_ultrasonic_s`、`tau_transport_thermal_s` 和 `tau_transport_ndir_s`，不把公式下标直接当作字段名。

### 4.3 平衡物性与 A1 一致性

每个时刻先对 `x_s(t)` 调用现有平衡物理：

| 传感器 | 平衡物性 | 已有事实源 | v1 设备层用途 |
| --- | --- | --- |
| 超声 ToF | 版本化混合声速与理论声程 ToF | `src/gf/sim/a2dyn_sound_speed.py` 中显式选择的声速算子 | `a1_constant_cp_ideal_v1` 只保留历史回归；正式 A2-DYN 使用 `a2dyn_direct_multifluid_eos_v1` 决定传播时延 |
| 热导电压 | WMS 混合热导率与 A1 电压锚点 | `wms_thermal_conductivity`、`thermal_conductivity_voltage` | 热导率进入电热状态；A1 电压只作名义稳态 parity |
| NDIR | HITRAN2020 CO₂ 线强与 Voigt 吸收的 active/reference 带宽积分 | `a2_sensor_devices.py` 的注册 HITRAN 表 | 吸收进入 active/reference 光学链；A1 标量电压只作历史迁移参考 |

名义环境和名义 hardware profile 下，热导设备层稳态输出通过一次校准映射对齐 A1 deterministic signal 的冻结容差；NDIR 不再把 A1 标量电压当作等价物理公式，而是以注册 HITRAN2020 active/reference 核直接审计零点、低浓度灵敏度和高浓度饱和。超声则以 A2-DYN 配置显式选择的新声速算子为物理事实源；A1 ToF 只保留为旧算子回归和迁移差值，不再要求新物性通过 gain、delay 或其他校准强制回到旧数值。动态模块不得复制声速、WMS 或 Beer–Lambert 公式；设备层只负责把共享物性变成可测读数。

`A2-DYN-1` 的两套自建低压维里声速均已因压力方向门失败而退出正式生成路径。R4 直接使用固定版本 CoolProp HEOS 生成声速；适配层只冻结版本、相态、组成、温压域和运行时 hash，不复制 EOS 公式。具体执行与验收见第 19.2 节。生成器不可用或越域时显式失败，不得用校准、噪声、旧模型或缩小审计域掩盖问题。

### 4.4 超声双时间尺度采集

v1 冻结为横穿共同气室、与主流近似正交的单声程几何，因此名义传播时延为：

$$
\tau_{prop,k}=\frac{L_{us}}{c(x_{us,k},T_k,p_k)}.
$$

该几何不把流速作为 v1 超声输入。若未来改成斜置流动声路，必须升版并同时生成上、下游 ToF，不能在现有单程公式上追加经验流速偏置。

每个外层时刻内部临时生成短接收波形：

$$
a_k(t)=A_k\left[
r(t-\tau_{prop,k})
+\sum_m \beta_m r(t-\tau_{prop,k}-\Delta_m)
\right]+n_k(t),
$$

其中 `r(t)` 是 hardware profile 的参考波包，`A_k` 表示几何扩散和组成相关衰减，`beta_m` 与 `Delta_m` 表示有限个预注册多径分量，`n_k(t)` 是内部 ADC 与电子噪声。批量生成不得逐点运行 KLM 或 FEM；参考波包采用带限解析模板或独立验证的固定资产。

内部采集固定执行：重复发射与平均 → 带通滤波 → 参考波形互相关粗定位 → 注册的三点抛物线亚采样精化（若候选启用）→ 输出 `tof`。`A2-DYN-2` 只在预注册的 `bandlimited_burst` 与 `linear_chirp` 候选间做 pilot，随后冻结一个正式 excitation profile。内部载频、ADC rate、重复频率和平均次数属于 acquisition profile，不等于外层 1、2 或 5 Hz 过程采样率。

超声同时产生 `peak_correlation`、`snr`、`estimated_tof_uncertainty` 和 `lock_status` 审计量。v1 模型输入仍只有低频 ToF；质量审计量默认不进入 `FusionCore`。失锁必须显式拒绝该 observation 或 profile，不得回退到理论 `L/c`。A0 的 `0.5 s` 不再作为正式超声时间常数。

### 4.5 热导集总电热模型

热导通道使用一个加热器热节点：

$$
C_h\frac{dT_h}{dt}
=P_{Joule}
-G_g(k_{mix},v)(T_h-T_g)
-G_s(T_h-T_{sub}),
$$

$$
R_h=R_0\left[1+\alpha_R(T_h-T_0)\right].
$$

`k_mix` 只能来自共享 WMS 算子；`C_h`、`G_s`、气体传热几何系数、加热功率、TCR 和桥压来自冻结的 thermal hardware profile。由 `R_h` 和唯一桥路方程得到 clean 电压。响应幅值与电热时间常数应随气体组成自然变化，不再使用 A0 的固定 `10 s × multiplier` 作为正式响应。

传感器宏观换气只由 `x_thermal(t)` 和 `tau_transport_thermal_s` 表达，微桥热惯性只由上述能量平衡表达。名义稳态通过一次显式校准映射对齐 A1 `thermal_conductivity_voltage`；该校准不能在不同 split 重新拟合。

### 4.6 NDIR 光学与局部气室模型

v1 采用单一高量程短光程 profile，覆盖注册的 0–100 mol% CO₂ 范围，不在序列内切换量程。active 和 reference 通道分别为：

$$
V_{act,k}\propto
\int S(\lambda)F_{act}(\lambda)D(\lambda)
\exp[-\kappa(\lambda,T_k,p_k)x_{CO2,k}L]d\lambda,
$$

$$
V_{ref,k}\propto
\int S(\lambda)F_{ref}(\lambda)D(\lambda)d\lambda.
$$

正式 R4 设备模型直接调用 HAPI 的 HITRAN2020 CO₂ 表，在注册波数网格上计算 Voigt 吸收，再用 active/reference 高斯带宽积分形成透过率；压力展宽使用 `air` 稀释代理，属于 `sensitivity_tier_1` 假设，不是实测器件标定。`optical_path_m`、active/reference band、source spectrum、detector response 和电子热响应均属于 NDIR hardware profile。

NDIR 慢响应拆为两部分：`x_ndir(t)` 表示光学气室换气或扩散；可选 `tau_emitter_detector` 只表示光源与探测器电子热响应。A0 的固定 `8 s` 只有在被重新登记为局部气室输运候选并通过 pilot 后才可复用，不能继续称为通用 NDIR 时间常数。

最终 clean 输出由冻结的 active/reference 比值映射产生；零 CO₂ 基线和低 CO₂ 灵敏度由 R4 HAPI 参考核直接审计，A1 `ndir_co2_voltage` 不再被当作等价物理公式。全组成网格必须报告饱和比例、量化平台长度和低 CO₂ 灵敏度；若高量程 profile 不能同时满足边界与可辨识性要求，v1 终态为 `PHYSICS_INVALID`，不能靠随机抖动恢复信息。

### 4.7 观测链

每路最终观测按固定次序形成：

$$
y_s(t)=Q_s\left[
g_s z_s(t)+o_s+r_s(t)+c_s(t)+w_s(t)
\right].
$$

其中：

- `g_s`：序列级 gain；
- `o_s`：序列级 offset；
- `r_s(t)`：慢漂移项；
- `c_s(t)`：时间相关噪声或共享环境噪声；
- `w_s(t)`：白噪声；
- `Q_s`：传感器量化算子。

处理顺序固定为：共同气室 → 局部输运 → 共享平衡物性 → 器件或采集 → gain/offset → 漂移 → 相关噪声 → 白噪声 → 量化 → 边界审计。不能在量化后再次添加连续噪声，也不能通过 clip 让越界样本伪装合法。

超声内部波形噪声先于 ToF 估计，外层 `w_s(t)` 则表示估计结果或模块输出上的剩余噪声；二者必须使用不同字段和 profile，不能对同一噪声源重复计数。

### 4.8 时序噪声

相关噪声用显式 AR(1) 过程：

$$
c_{s,k}=\rho_s c_{s,k-1}
+\sqrt{1-\rho_s^2}\,\epsilon_{s,k}.
$$

`\epsilon` 的边际尺度由每传感器 noise profile 决定。共享扰动通过一个公共过程和固定通道载荷向量投影，不能把三路噪声简单复制成完全相同。

漂移是序列级随机截距与低频斜率，不允许每个时间点独立重抽标定参数。重复观测共享 `mixture_id`，但拥有不同 `observation_id` 和独立观测噪声。

## 5. 时间轴、阶段与在线窗口

### 5.1 Canonical v1 时间轴（R4 冻结）

| 项目 | R4 冻结值 |
| --- | ---: |
| 总时长 | 240 s |
| 采样率 | 5 Hz |
| `dt` | 0.2 s |
| 时间点数 | 1,200 |
| 时间范围 | 0.0–239.8 s |
| 传感器同步 | v1 三路共用同一时间轴 |
| 超声内部采集 | 每个外层时刻独立短窗口，由 acquisition profile 定义 |
| 缺测 | v1 主数据不主动制造整路缺失或异步缺口 |

2 Hz 只是共同气室、局部输运和低频输出的候选，不是超声载频或 ADC rate。R4 pilot 在同一直接 HEOS 轨迹上比较 1 Hz、2 Hz 和 5 Hz 的任务信息增益、过程混叠和数据量：1 Hz 不满足 `0.5 s` 实时刷新门，2 Hz 与最佳信息分数相差超过注册的 `0.02`，因此冻结 5 Hz。超声内部采集仍按自己的短波形时间轴验证；不得为解析超声载波把整条 240 s 序列提升到 MHz。

### 5.2 标准阶段

| phase | 时间 | 点数 | 进气语义 | 主用途 |
| --- | --- | ---: | --- | --- |
| `baseline` | 0–30 s | 150 | 纯 Ar purge | 估计初始水平和短时噪声 |
| `transition` | 30–90 s | 300 | 目标气体 step 或 ramp 进入 | 早期在线估计与动力学识别 |
| `steady` | 90–180 s | 450 | 目标进气保持 | 中后期估计与稳态接近度 |
| `recovery` | 180–240 s | 300 | 恢复 purge | 响应闭环和滞回审计 |

阶段名称借鉴 tv3 的阶段化表达，但不 import tv3 实现。`phase_id` 是评价与审计字段，默认不作为模型特征。

### 5.3 因果预测窗口

所有在线窗口相对于真实暴露起点 `t_onset` 定义。每个窗口包含完整 baseline 和暴露后的当前前缀：

| 窗口 ID | 暴露后时长 | 标准 onset 下累计点数 | 用途 |
| --- | ---: | ---: | --- |
| `P005` | 5 s | 175 | 极早期可用性 |
| `P015` | 15 s | 225 | 快速预警 |
| `P030` | 30 s | 300 | 主早期指标之一 |
| `P060` | 60 s | 450 | 中期主指标 |
| `P120` | 120 s | 750 | 接近稳态前的主指标 |
| `P150` | 150 s | 900 | recovery 前完整暴露 |
| `FULL` | 240 s | 1,200 | 离线诊断，不能代表实时主结果 |

主评价以 `P015/P030/P060/P120` 为核心。`P005` 可能受不可约滞后支配，单列报告；`P150` 用于连接稳态结果；recovery 只用于机制审计，不作为在线预测必须拥有的未来信息。

### 5.4 进气协议

`b(t)` 至少支持：

| profile | 定义 | 训练可见性 |
| --- | --- | --- |
| `STEP_STANDARD` | 标准 onset 后立即切换到目标 | train、val、所有参照族 |
| `RAMP_LINEAR` | 5–30 s 线性上升 | train、val |
| `RAMP_SMOOTH` | 10–45 s 平滑 S 型上升 | stress_val、test |
| `ONSET_SHIFT` | onset 相对 30 s 偏移 | 少量 train，主要 stress_val |
| `SHORT_PULSE` | 尚未充分稳定即开始恢复 | stress_val、test |
| `MULTI_PULSE` | 同一目标配方重复暴露 2–3 次 | test 诊断 |
| `INCOMPLETE_RECOVERY` | recovery 结束时仍有残留 | stress_val、test |

多脉冲始终使用同一个 `x_target`，不在一条 v1 序列中切换到第二个目标配方。多目标切换属于未来稠密状态跟踪任务。

`SHORT_PULSE` 的首次有效暴露时长不得短于 60 s，因此 P015、P030 和 P060 仍可评价。每条记录保存 `exposure_end_s`；cutoff 晚于首次暴露结束的 horizon 不进入实时主指标，只标记为 protocol diagnostic。不能把恢复后的未来信息用于补齐 P120 或 P150。

## 6. 动力学与扰动分布

### 6.1 过程、局部输运与 hardware profile

#### 6.1.1 共同气室与局部输运

以下范围是 v1 sensitivity 计划值，需经 `A2-DYN-2` pilot 确认：

| 参数 | train | val | stress_val | test |
| --- | --- | --- | --- | --- |
| `tau_mix_s` | 6–18 s | 8–22 s | 24–45 s | 45–75 s |
| `tau_transport_ultrasonic_s` | 0–1 s | 0–2 s | 1–4 s | 2–6 s |
| `tau_transport_thermal_s` | 1–6 s | 2–8 s | 8–18 s | 15–30 s |
| `tau_transport_ndir_s` | 2–10 s | 4–14 s | 12–28 s | 24–45 s |
| phase duration jitter | 0–5% | 0–8% | 8–20% | 15–30% |

`tau_mix_s` 和非零局部输运时间使用 log-uniform。局部输运范围是 `sensitivity_tier_*`，只表示主气室到声路或小气室的换气，不代表换能器、微桥或光电器件固有响应。train 不含 test 的极慢区间；stress_val 和 test 仍必须通过边界与可辨识性审计。

#### 6.1.2 超声 acquisition profile

| 字段 | v1 规则 | 来源等级 |
| --- | --- | --- |
| `geometry` | 固定 `transverse_single_path` | `literature_structure` |
| `path_length_m` | A2-DYN-0 冻结一个合成声程 | `sensitivity_tier_1` |
| `excitation_type` | pilot 候选仅 `bandlimited_burst`、`linear_chirp` | `literature_structure` |
| `center_frequency_hz`、`fractional_bandwidth` | 随候选 profile 固定，不逐 observation 重抽 | `literature_anchor` |
| `adc_rate_hz` | 至少满足带限波形离散和亚采样精化测试 | `literature_anchor` |
| `pulse_repetition_hz`、`average_count` | 共同决定 acquisition window | `literature_anchor` |
| `tof_estimator` | 固定 `reference_xcorr` 或 `reference_xcorr_parabolic` | `literature_structure` |
| `multipath_profile` | train 仅轻微；stress_val、test 使用冻结 OOD 档 | `sensitivity_tier_*` |

具体频率、声程、ADC rate 和平均次数不能由不同论文拼成所谓典型仪器。A2-DYN-0 只冻结两个内部一致的完整候选；A2-DYN-2 根据无噪声偏差、失锁率、SNR、计算量和任务信息冻结一个正式 profile。test 不使用未在 pilot 中注册的新估计器。

#### 6.1.3 Thermal hardware profile

唯一名义 profile 为 `TCD-LUMPED-SYNTH-1`，至少包含：

- `heater_heat_capacity`；
- `gas_conductance_scale`；
- `substrate_conductance`；
- `heater_power`；
- `tcr`；
- `bridge_voltage`；
- `flow_coupling`。

这些字段按设备级 profile 固定，不逐时间点重抽。train、val、stress_val 和 test 的差异通过完整 profile ID 表达，不能再使用公共 sensor multiplier。任何 OOD profile 都必须保持正热容、正热导、有限稳态温升和能量守恒。

#### 6.1.4 NDIR hardware profile

唯一主量程语义为 `NDIR-HIGHRANGE-SHORTPATH-1`，至少包含：

- `optical_path_m`；
- `active_band_id` 与 `reference_band_id`；
- `source_spectrum_id`；
- `detector_response_id`；
- `effective_absorption_model_id`；
- `tau_emitter_detector`；
- `range_min_mol_pct=0`、`range_max_mol_pct=100`。

v1 不使用依据当前读数自动切换的双量程，避免引入隐藏路由。若单一高量程 profile 无法在全单纯形同时通过低浓度灵敏度和高浓度饱和门，应判定该组成域的物理设计失败并升版，不得静默切换增益。

所有真实 `tau_mix_s`、三路 `tau_transport_*_s`、hardware 参数、内部波形参数和 device state 都是特权生成参数，不作为模型输入。

### 6.2 环境 profile

沿用 A2H 已登记的环境块作为序列平均工况：

| split | environment | 温度 | 压力 | 来源等级 |
| --- | --- | ---: | ---: | --- |
| train | `ENV-TRAIN-LOW` | 293.15 K | 98,000 Pa | registered train block |
| train | `ENV-NOMINAL` | 298.15 K | 101,325 Pa | A1 reference |
| train | `ENV-TRAIN-HIGH` | 303.15 K | 105,000 Pa | registered train block |
| val | `ENV-NEAR` | 308.15 K | 108,000 Pa | application range |
| stress_val | `ENV-MID` | 313.15 K | 112,000 Pa | sensitivity tier 1 |
| test | `ENV-FAR` | 278.15 K | 90,000 Pa | sensitivity tier 2 |

v1 的单条序列内温压默认保持常量，以先隔离气体交换和传感器动力学。慢温压变化只放入 `D-JOINT` 诊断，不与主动力学族混合。温压作为可观测上下文时，所有基线必须使用同一 context arm；未知环境 arm 则所有模型都不读取。

### 6.3 标定 profile

沿用 A2H 的四级结构：

- train：`CAL-NOMINAL`；
- val：`CAL-LIGHT`；
- stress_val：`CAL-SHARED-DRIFT`；
- test：`CAL-CONFLICT`。

gain、offset 和物理 scale 的具体数值复用 A2H 配置事实源，不在动态生成器中复制。动态层只增加“序列内漂移斜率”，且该斜率必须独立登记。

### 6.4 噪声与漂移 profile

| 参数 | train | val | stress_val | test |
| --- | --- | --- | --- | --- |
| 白噪声档 | `NOISE-1X` | `NOISE-2X` | `NOISE-5X` | `NOISE-10X` |
| AR(1) `rho` | 0.00–0.40 | 0.20–0.60 | 0.65–0.85 | 0.85–0.97 |
| 共享相关载荷 | 0–0.10 | 0.10–0.20 | 0.20–0.35 | 0.35–0.50 |
| 漂移强度 | 0–0.10% 动态范围/min | 0.10–0.25%/min | 0.25–0.75%/min | 0.75–1.50%/min |
| 独立重复数 | 3 | 3 | 3 | 3 |

`NOISE-CORR-5X` 可作为 stress_val 的预注册 profile。test 范围必须经 oracle 和信号边界审计；若 10× 加强相关噪声后 oracle 同样失效，应判 `DYNAMIC_UNIDENTIFIABLE`，不能把它作为“更困难”的有效 test。

### 6.5 组成分布

三组分总和约束使组成空间只有两个自由度。v1 不依赖纯随机 Dirichlet 填满样本，而使用单纯形分层与低差异采样：

| 区域 | 计划比例 | 定义 |
| --- | ---: | --- |
| interior | 50% | 每个组分至少 5 mol% |
| near-boundary | 30% | 恰有一个组分位于 0.1–5 mol% |
| binary | 20% | 恰有一个组分为 0，另外两个远离纯气端点 |
| pure vertices | 3 个规范组 | 三个纯气端点，只作为专门 stress 或 test slice |

组成生成规则：

1. 在 0.01 mol% 量化后，全数据的非纯气配方坐标不得重复。
2. 三个 pure vertex 各只有一个规范 `mixture_id`，不能为跨 split 方便伪造多个组 ID。
3. 同一配方的重复噪声、协议或环境观测共享同一 `mixture_id`。
4. family 和 split 的抽样先冻结配额，再抽配方；不能看模型误差后移动 simplex 区域。
5. stress_val 与 test 的组成边际尽量与对应参照族匹配，避免把动力学 OOD 和组成 OOD 混成一个不可归因差异。
6. 只有 `D-JOINT` 的专门 slice 允许把边界组成与动态压力结合。

按全部 family 汇总后的精确 group 配额为：

| split | interior | near-boundary | binary | pure | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1,260 | 756 | 504 | 0 | 2,520 |
| val | 315 | 189 | 126 | 0 | 630 |
| stress_val | 315 | 189 | 126 | 0 | 630 |
| test | 315 | 189 | 123 | 3 | 630 |

三个规范 pure vertex 全部放入 `D-JOINT/test`，替换该格原有的三个 binary 配额，形成明确的纯气外推诊断。它们不复制到其他 split，也不计入动态非退化幅值比例的分母。

## 7. 数据族、规模与 split

### 7.1 六个数据族

| family | 主变量 | 固定变量 | 回答的问题 |
| --- | --- | --- | --- |
| `D-IID` | 独立配方和名义噪声 | 标准 step、train-range 动力学、名义环境标定 | 新动态生成器的同分布基线 |
| `D-KINETICS` | `tau_mix_s`、局部输运和已注册 hardware profile | 标准 step、名义环境标定 | 未见过程、换气与器件响应是否可泛化 |
| `D-PROTOCOL` | ramp、onset、pulse、recovery | train-range 动力学、名义环境标定 | 激励协议改变时是否仍能早期估计 |
| `D-NOISE-DRIFT` | 白噪声、AR(1)、共享噪声、漂移 | 标准 step、名义环境 | 轨迹收益是否只是平滑噪声 |
| `D-ENV-CAL` | 温压块和标定 profile | 标准 step、受控动力学 | 现有 A2H 压力进入序列后的影响 |
| `D-JOINT` | 最多两个已合格主压力加固定噪声 | 组合在训练前冻结 | 时序收益能否在联合变化下保留 |

每个 family 的配方组互相独立，避免同一 `mixture_id` 因属于多个 family 而产生隐式重复加权。若需要对同一配方做 paired 反事实生成，应建立专门 audit 数据，不并入主训练计数。

### 7.2 计划规模

| family | train groups | val groups | stress_val groups | test groups | repeat | observation rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `D-IID` | 720 | 180 | 180 | 180 | 1 | 1,260 |
| `D-KINETICS` | 360 | 90 | 90 | 90 | 1 | 630 |
| `D-PROTOCOL` | 360 | 90 | 90 | 90 | 1 | 630 |
| `D-NOISE-DRIFT` | 360 | 90 | 90 | 90 | 3 | 1,890 |
| `D-ENV-CAL` | 360 | 90 | 90 | 90 | 1 | 630 |
| `D-JOINT` | 360 | 90 | 90 | 90 | 2 | 1,260 |
| **合计** | **2,520** | **630** | **630** | **630** | — | **6,300** |

独立统计单位：

- 唯一 `mixture_id`：4,410；
- train 观测：3,600；
- val 观测：900；
- stress_val 观测：900；
- test 观测：900；
- 每条观测：3 × 1,200 个标量。

重复观测增加噪声统计稳定性，但 bootstrap 和 split 的独立单位仍是 4,410 个 `mixture_id`，不能把 6,300 行当作 6,300 个独立配方。

### 7.3 为什么采用该规模

1. 4,410 个二维组成组足以覆盖多 family，同时保持每个压力轴有独立组。
2. 6,300 × 3 × 1,200 的 float32 signals 为 90,720,000 bytes（约 86.5 MiB），主数组规模可在普通工作站一次加载。
3. 加上 clean signal、共同气室、低频 device state、质量审计、mask、phase 和元数据，未压缩规模预计约 0.6–0.7 GB；`A2-DYN-2` 必须记录实际峰值内存并以资源门复核。
4. 超声短波形只在生成时临时存在。若保存每个外层时刻的 4,300 点单向平均波形，仅 float32 波形就约 129,830,400,000 bytes（约 121 GiB）；v1 禁止把它并入正式聚合数据包。
5. 只允许为解析 fixture 和少量可视化样本保存内部波形，数量和用途写入 manifest。
6. 规模足够运行 5 seed 轻量基线，又不会在数据资格尚未确认前进入大规模训练。
7. 若 pilot 显示 2,520 个 train group 已饱和，应下调规模并记录学习曲线；若明显未收敛，只能依据嵌套学习曲线升版，不能随意加样本。

### 7.4 split 规则

1. 先生成唯一组成和 `mixture_id`，再按 family 配额分配 split。
2. 一个 `mixture_id` 的全部 observation 必须同 split。
3. test 不要求物理文件隔离，但在 recipe 冻结前不用于模型或分布选择。
4. val 用于普通选择，stress_val 用于判断动态 OOD 和资格门。
5. test 同时报告 iid、单轴和 joint，不只给一个汇总平均。
6. split manifest 保存每个 split 的有序 group 列表和 SHA-256。
7. group bootstrap 必须在 `mixture_id` 层配对抽样。

## 8. 数据契约与存储

### 8.1 聚合数据包

计划目录：

```text
data/a2_dynamic_v1/
  config_snapshot.json
  manifest.json
  records.jsonl
  observations.npz
  oracle.npz
  device_audit.npz
  waveform_fixtures.npz
  audit.json
```

允许 train、val、stress_val 和 test 共同存放在 `observations.npz`。`records.jsonl` 提供逐 observation 的身份和分层字段；`oracle.npz` 保存共同气室与低频 clean state；`device_audit.npz` 保存不供模型读取的设备质量量。`waveform_fixtures.npz` 只保存 manifest 登记的小规模超声解析与可视化样本，不包含全部 observation 的高频波形，也不建立权限系统。

### 8.2 主数组

| 数组 | dtype / shape | 模型可见 | 说明 |
| --- | --- | --- | --- |
| `signals` | `float32[N,3,1200,1]` | 是 | 三路最终观测 |
| `valid_mask` | `bool[N,3,1200,1]` | 是 | v1 主数据应全真 |
| `quality` | `float32[N,3,1200]` | 是 | 模型可见的有效性标记固定为 1；组成相关设备质量只写入 `device_audit`，不暴露给模型 |
| `time_s` | `float64[1200]` | 是 | 全数据共享时间轴 |
| `target` | `float32[N,3]` | 训练 split 可见 | 目标进气配方 |
| `phase_id` | `int8[N,1200]` | 默认否 | 评价与审计 |
| `observation_index` | `int64[N]` | 否 | 对齐 records |

### 8.3 Oracle 数组

| 数组 | dtype / shape | 用途 |
| --- | --- | --- |
| `inlet_composition` | `float32[N,1200,3]` | 检查进气协议和闭合 |
| `chamber_composition` | `float32[N,1200,3]` | 检查质量守恒和响应 |
| `equilibrium_reference_signals` | `float32[N,3,1200]` | 配置选择的共享平衡物性输出；超声使用新声速，热导用于 A1 稳态 parity，NDIR 使用 HAPI active/reference 核 |
| `clean_device_signals` | `float32[N,3,1200]` | 局部输运和设备层之后、观测扰动之前的读数 |
| `device_states` | `float32[N,3,1200]` | 超声估计状态、加热器温度和 NDIR 光学比值的统一审计投影 |
| `privileged_parameters` | 结构化数值表 | 过程、局部输运、hardware 和 oracle 分层 |

`device_audit.npz` 至少包含超声 `peak_correlation`、`snr`、`estimated_tof_uncertainty`、`lock_status`，热导能量平衡 residual，以及 NDIR active/reference 电压、饱和标记和量化平台长度。训练 adapter 默认只打开 `observations.npz` 和公开 records 字段；oracle 与设备审计可由审计 pipeline 读取，但不能通过 metadata 传入 `FusionCore`。

### 8.4 records 字段

每行至少包含：

| 字段 | 约束 |
| --- | --- |
| `schema_version` | 固定 `gf-a2-dynamic-record-1` |
| `observation_id` | 全数据唯一 |
| `mixture_id` | 非空，真实 group key |
| `split` | train、val、stress_val、test |
| `family` | 六个 family 之一 |
| `composition_region` | interior、near_boundary、binary、pure |
| `protocol_profile_id` | 指向冻结 protocol |
| `transport_profile_id` | 指向共同气室与局部输运 profile |
| `ultrasonic_profile_id` | 指向冻结的几何、激励和估计器 profile |
| `thermal_profile_id` | 指向冻结电热 profile |
| `ndir_profile_id` | 指向冻结光学与量程 profile |
| `environment_id` | 指向冻结环境块 |
| `calibration_profile_id` | 指向冻结标定 |
| `noise_profile_id` | 指向冻结噪声 |
| `exposure_onset_s`、`exposure_end_s` | 定义协议的首次有效暴露区间 |
| `timesteps` | 1,200 |
| `dt_s` | 0.2 |
| `status` | generated、audited、rejected |

禁止字段：

- `sequence_id` 作为 group；
- `base_condition_id`；
- `noise_seed_index`；
- `noise_seed`；
- 可直接恢复标签的文件名编码；
- 把真实输运时间、hardware 参数、内部波形质量量、device state 或 clean signal 填入模型可见 metadata。

### 8.5 UnifiedSample 映射

每条 observation 映射为：

- `signals`：长度 3 的 tuple，每项 `float32[1200,1]`；
- `sensor_id`：固定三路唯一 ID；
- `sensor_type`：沿用现有注册；
- `valid_mask`：与 signal 同 shape；
- `quality`：每路 `float32[1200]`；v1 模型可见有效性标记固定为 1，实际设备质量放在审计专用 `device_audit`，不用隐藏 noise profile 编码；
- `time`：三路共享但分别提供严格递增数组；
- `target`：`float32[3]`；
- `target_mask`：三个真；
- `group_id`：与原始 `mixture_id` 完全相同；
- `dataset_id`：`ar_he_co2`；
- `metadata["mixture_id"]`：原始值。

现有 collate 已支持 `T>1`，不为动态数据另建第二套 batch schema。

## 9. 实时与流式仿真接口

### 9.1 离线生成和实时回放分离

物理数据先离线、确定性生成，再通过回放器逐点发布：

1. `offline_generate`：一次生成完整序列、manifest 和 oracle；
2. `stream_replay`：按时间顺序输出当前帧或当前前缀；
3. `online_infer`：模型维护状态或重算当前前缀；
4. `emit_prediction`：在注册 horizon 输出预测和耗时；
5. `aggregate`：离线计算准确率、time-to-accuracy 和延迟。

回放器不重新抽噪声，确保离线与在线看到完全相同的观测。实时模式不是第二套数据生成器。

### 9.2 回放模式

| 模式 | 时间推进 | 用途 |
| --- | --- | --- |
| `virtual_clock` | 不等待墙钟，按 timestamp 顺序立即推进 | 单元测试、训练、批量评价 |
| `accelerated` | 按可配置倍速推进 | 人工演示和集成测试 |
| `wall_clock_1x` | 0.2 s 一次更新 | 最终实时 smoke |

自动测试必须使用 virtual clock，不能为 240 s 序列真实 sleep。`wall_clock_1x` 只作为人工或专门集成检查，不进入默认测试。

### 9.3 流式消息最小契约

每次更新至少包含：

| 字段 | 说明 |
| --- | --- |
| `observation_id` | 当前观测身份 |
| `timestamp_s` | 当前时间，严格递增 |
| `sensor_values` | 三路当前值 |
| `valid_mask` | 当前三路有效性 |
| `quality` | 当前三路质量 |
| `is_end` | 是否序列结束 |

消息不包含 target、split、oracle state、动力学真值或后续 timestamp。模型状态在新的 `observation_id` 开始时必须显式 reset。

### 9.4 因果性检查

- prefix slicer 只能返回 `timestamp <= cutoff` 的点；
- 改动 cutoff 后，历史部分字节必须一致；
- 在线预测与对同一前缀做离线预测应在数值容差内相同；
- 模型不得通过 padding 长度、预先分配的完整 mask 或 phase 数组看到未来；
- 运行日志记录每次预测实际使用的最大 timestamp；
- 若模型使用双向编码器，只能在 FULL 离线对照中运行，不能标记为实时。

### 9.5 运行延迟

每个模型报告：

- 单帧更新延迟 p50、p95、max；
- 首次预测延迟；
- 每个注册 horizon 的累计推理耗时；
- peak CPU 内存和 GPU 显存；
- 测试硬件、线程数、device、dtype。

实时资格要求 p95 单帧处理时间小于 `dt=0.2 s`。若模型按前缀全量重算，也必须按实际耗时报告，不能只报告单次 batch 推理。

对 `SHORT_PULSE`、`MULTI_PULSE` 等协议，只有 cutoff 不晚于 `exposure_end_s` 的 horizon 进入实时准确率聚合。无效 horizon 使用显式 `horizon_valid=false`，不得用末次有效预测、0 或序列末值填充。

## 10. 质量、物理与动态真实性审计

### 10.1 Schema 审计

- 数组数量、dtype 和 shape 与 manifest 一致；
- `N=6300`、`T=1200`、`S=3`；
- records 行数与 observation index 一一对应；
- 所有时间有限且严格递增；
- target 有限、非负且和为 100；
- `mixture_id` 非空，同组不跨 split；
- observation ID 唯一；
- 禁止字段不存在；
- content hash、config hash 和 split hash 可重算。

### 10.2 质量守恒

对每个时间点检查：

$$
x_{Ar}(t)+x_{He}(t)+x_{CO2}(t)=100\%.
$$

计划容差：

- inlet composition：绝对和误差不超过 `1e-6 mol%`；
- chamber composition：绝对和误差不超过 `1e-5 mol%`；
- 任一组分不得低于 `-1e-7 mol%`；
- 超出容差立即失败，不通过逐步归一化隐藏。

### 10.3 平衡一致性

1. 对相同组成和温压，动态模块调用的声速、WMS 热导率和 Beer–Lambert 算子与配置显式选择的共享物理算子绝对差不超过 `1e-12`；pipeline 和设备层不得另建公式。
2. A2-DYN 声速的正式合成定义是固定版本 CoolProp HEOS `speed_sound()`。完整组成单纯形、注册温压块和固定 seed 离网点必须与该同一生成算子逐点一致，最大相对差为 `0`；同时保存生成器版本、二进制 hash、运行入口 hash、压力方向统计和明确的 `generator_consistency` 验证范围。本项不把 HEOS 自身准确性写成独立验证结论。
3. 名义 hardware profile 的稳态热导电压仍与 A1 deterministic signal 做 parity；NDIR 则以注册 HITRAN2020 active/reference 结果做零点、灵敏度和饱和审计，不把 A1 标量 NDIR 公式冒充为同一物理算子。超声波形估计 ToF 与新声速算子的理论传播时延按内部采样误差门验收；A1 旧 ToF 只报告迁移差值，不作为新声速的校准目标。设备层校准只在名义 profile 建立一次，不能按 split 重拟合。
4. 对足够长的名义 step audit，局部组成、热导电热状态和 NDIR 光电状态分别收敛到其理论或高精度数值稳态；不能再用统一 `5 tau_s` 判断三路。
5. A1、A2H 的原文件和 content hash 不变；A2-DYN r1/r2 与 pair-v2 只作为只读失败证据。
6. 单位转换只发生在共享物理边界，不能在 dataset、device model 和 adapter 各做一次。

### 10.4 分层解析与设备回归

对无观测扰动的固定 fixture 分层检查：

- 共同 CSTR 与每路非零局部输运的数值序列逐点匹配解析指数更新，`t50`、`t90` 与理论值偏差不超过一个外层采样间隔；
- 超声无噪声、无多径时，`reference_xcorr` 的 ToF 偏差不超过一个内部 ADC sample；`reference_xcorr_parabolic` 的偏差不超过 `0.25` 个内部 sample；
- 超声改变内部 ADC rate 后估计误差按预注册方向收敛，低相关或相位歧义必须显式失锁；
- 热导电热状态满足离散能量平衡，稳态温升与集总解析解一致，改变 `k_mix` 会同时改变稳态幅值和响应时间；
- NDIR 在零 CO₂ 时回到 active/reference 校准基线，直接 HITRAN 带宽积分的 active/reference 输出必须通过低 CO₂ 灵敏度和高量程饱和审计；
- recovery 使用同一共同气室、局部输运和设备方程返回 purge，不完全恢复 residual 与配置一致；
- 任一 fixture 失败都保留错误，不得回退到理论 ToF、A1 电压或旧固定一阶曲线。

### 10.5 动态非退化

排除“时间轴存在但信号实为常量或稳态复制”：

1. 非纯气序列中，至少 95% 有两路以上 peak-to-peak 大于各自名义噪声标准差的 5 倍。
2. 每路有效动态序列至少覆盖 10 个不同量化级。
3. 至少 70% 的名义序列存在两路低频观测 `t50` 相差两个以上外层采样点；超声内部传播时间不参与该统计。
4. baseline、transition、steady、recovery 均有非空点。
5. clean signal 的 transition 方差不能全部由白噪声解释。
6. 任意 family 若超过 5% 序列被判为动态退化，该 family 不进入正式训练。

纯气或目标等于 purge 的序列不能用于上面幅值比例的分母，应单列为边界审计。

### 10.6 边界、饱和与量化

- 沿用 A2H signal bounds；
- 报告每路 min、max、P01、P99；
- 报告越界率、靠近边界 1% 的比例和量化唯一值数；
- 越界不 clip，标记 profile 无效；
- 超声报告失锁率、低相关率、ToF 不确定度和多径 stress 的峰选择错误率；
- 热导报告温升范围、能量 residual、桥路饱和率和组成相关时间常数分位数；
- NDIR 报告 active/reference 范围、低 CO₂ 灵敏度、高 CO₂ 饱和率和连续量化平台长度，防止指数压缩导致假可辨识；
- pure CO₂ 若进入饱和区，只能作为明确失败或 stress 证据，不能计入正常可回归样本；
- ToF 与热导若被量化成大段平台，调整采样或 resolution profile，而不是加随机抖动掩盖。

### 10.7 可辨识性

审计至少包括：

1. 固定动力学 nuisance 时，目标组成到多 horizon 观测的有限差分 Jacobian；
2. 未知 nuisance 时，目标组成与 `tau_mix_s`、局部输运、冻结 hardware profile 的联合局部 Jacobian；
3. 各 family、split 和 horizon 的 rank fraction；
4. condition number P50、P95、max；
5. pure 和结构零边界单列；
6. 特权 kinetics oracle 与可部署基线的 headroom。

默认资格：

- 目标组成 Jacobian full-rank fraction ≥0.99；
- P95 condition number <1000；
- oracle 预测有限且组成合法；
- 至少两个早期 horizon 留有可部署学习器可利用的 headroom。

## 11. 基线、指标与时间信息资格门

### 11.1 数据资格基线

| ID | 输入 | 模型 | 作用 |
| --- | --- | --- | --- |
| `B-LAST` | 当前 horizon 的三个末值 | A2M-MLP 同级小 MLP | 检查只看当前标量能做到什么 |
| `B-DELTA` | baseline 均值、当前末值、两者差 | Ridge 与小 MLP | 检查简单基线校正 |
| `B-EWMA` | 截止当前的三路指数平滑值与一阶差分 | Ridge 与小 MLP | 检查时序收益是否只是简单降噪和平滑 |
| `B-STAT` | 每路均值、斜率、AUC、方差、分位数、observed half-range crossing | Ridge、GBDT、小 MLP | 强统计特征基线 |
| `B-TCN` | 原始三路因果前缀 | 轻量 causal TCN | 最小时序神经基线 |
| `B-GRU` | 原始三路因果前缀 | 单层 GRU | 架构类型诊断 |
| `B-STEADY` | P150 的三个低频读数 | A2M-MLP 同级小 MLP | 只作接近稳态的静态对照，不参与早期因果竞争 |
| `O-EQ` | clean equilibrium reference signal | 特权平衡 oracle | 目标可达上界 |
| `O-KIN` | clean device signal 加真实输运与 hardware 参数 | 特权数值 oracle | 区分 nuisance 与不可辨识 |

这些基线只用于数据资格和 headroom 判断，不自动成为新算法候选。所有可部署基线使用相同 split、scaler、seed、训练预算和输出头。`B-EWMA` 的平滑系数只用 train 选择并在 val、stress_val、test 固定。`B-STEADY` 使用未来稳定读数，只能作为 P150 和 FULL 的静态 sanity check，不能与 P015、P030、P060 的实时模型做可部署性等价比较。

### 11.2 主指标

沿用现有三组分 `macro_RNMAE` 为每个 horizon 的主准确率指标，并报告：

- 三组分 RNMAE、MAE、RMSE、R²；
- 组成和偏差、负值率、超过 100% 比例；
- family、protocol、共同气室、局部输运、hardware、环境、标定、噪声和组成区域分层；
- P90 group MAE、worst-slice MAE；
- 5 seed 均值、标准差和逐 seed 结果；
- 2,000 次 `mixture_id` group bootstrap 置信区间。

### 11.3 动态专用指标

1. `Error@P005/P015/P030/P060/P120/P150`：各前缀误差。
2. `AUEC`：误差—时间曲线在 P015–P120 的归一化面积，越低越好。
3. `TTA@5mol%`：首次达到样本平均绝对组分误差不超过 5 mol% 的时间。
4. `TTA@2mol%`：首次达到 2 mol% 的时间。
5. `EarlyGain`：相对冻结 `B-REF` 的前缀误差改善。
6. `RecoveryConsistency`：用同一模型在 recovery 诊断时的状态一致性，只作辅助。
7. `LatencyP95`：在线单帧推理 p95。

若某条序列从未达到阈值，TTA 记为右删失，不填为序列末时刻或 0。

`B-STAT` 的 half-range crossing 只能由当前 prefix 的 baseline 和已观测 min/max 计算，不能使用真实目标终值。AUEC 主值只在 P015–P120 全部有效的 observation 上计算；短脉冲另报 AUEC@P060。TTA 的删失上界取 `exposure_end_s`，不是整条序列结束时间。

### 11.4 动态难度门

至少满足：

1. `B-LAST` 在 P015、P030、P060 中至少两个 horizon 相对 P150 退化 ≥25%，证明早期问题不是稳态复制。
2. `O-KIN` 在相同早期 horizon 相对 `B-LAST` 保留 ≥20% 的误差改善空间。
3. 早期 oracle 没有因极端噪声或 incomplete exposure 全面失效。
4. 各 family 的关键动态参数能由审计区分，不能所有变化只映射为同一个幅值缩放。

未通过则关闭为 `TEMPORAL_REDUNDANT` 或 `DYNAMIC_UNIDENTIFIABLE`，不训练更多时序架构。

### 11.5 时间增量信息门

每个 horizon 先在冻结 val recipe 上从 `B-LAST`、`B-DELTA`、`B-EWMA` 中选出误差最低的因果参考，记为 `B-REF`；选择结果在 test 前冻结。`B-STAT` 或 `B-TCN` 必须相对 `B-REF` 满足：

- 在 P015、P030、P060 中至少两个 horizon 的平均 macro_RNMAE 改善 ≥10%；
- 每个用于晋级的 horizon 至少 4/5 seed 改善；
- 2,000 次 paired group bootstrap 的 95% CI 上界小于 0；
- 任一组分 RNMAE 不退化超过 0.005；
- 改善至少在 `D-IID` 与一个合格压力 family 上同时成立；
- 结果不是通过读取 phase、真实输运或 hardware 参数、future padding 或 oracle metadata 获得；
- 在 P150 上相对 `B-STEADY` 的 macro_RNMAE 退化不超过 5%，避免用牺牲稳态准确率换取早期表面收益。

门通过才形成 `DYNAMIC_QUALIFIED`。只在 FULL 或 recovery 后改善不算实时增量信息。

### 11.6 新算法 headroom 门

动态数据合格后，还要判断是否值得设计复杂方法：

- 若 `B-EWMA` 或 `B-STAT` 已进入 `O-KIN` 的 5% 等价带，优先保留简单方法；
- 若 `B-TCN` 相对 `B-STAT` 无稳定收益，不能仅因“深度学习”继续扩展 Transformer；
- 若 `B-TCN` 已接近 oracle，但在线成本高，后续研究问题只能转向效率或压缩，不能声称更高精度 headroom；
- 只有至少一个合格压力轴仍保留 ≥10% 的 oracle headroom，才进入新融合算法构思。

## 12. A2-DYN 分步执行

### 12.1 A2-DYN-0：冻结机器协议与候选设备 profile（2–3 天）

目标：把本文的 schema、时间轴、分布、基线、指标和停止规则转为机器配置。

计划动作：

1. 新建 `configs/data/ar_he_co2_a2_dynamic_v1.json`。
2. 新建 `configs/eval/a2_dynamic_eval.json`。
3. 新建 `configs/experiment/a2_dynamic_protocol.json`。
4. 固定 schema、family、split、group 数、repeat、参数来源等级和输出目录。
5. 固定 `transverse_single_path` 超声几何、两个 excitation 候选、ToF 估计器候选、`TCD-LUMPED-SYNTH-1` 和 `NDIR-HIGHRANGE-SHORTPATH-1`。
6. 固定 CoolProp HEOS 8.0.0 生成版本、HITRAN2020 参考资产、查询网格和误差门；HEOS 误差审计按 `generator_consistency` 记录，不宣称独立物理验证。
7. 配置 validator 拒绝禁用字段、未知或不完整 profile、重复组成和 split 交叉。
8. 记录 A1、A2H 共享物理、设备候选和参考资产 hash。

阶段门：

- 配置互相引用一致；
- 所有计划范围有 source level；
- 每个 hardware profile 字段完整且单位明确；
- 不再引用 A0 的 0.5 s、10 s、8 s 作为正式传感器响应；
- 没有模型结果驱动的参数；
- v1 不实现复杂访问控制；
- status 为 `PROTOCOL_FROZEN`。

### 12.2 A2-DYN-1：实现分层物理、设备采集与解析测试（5–8 天）

目标：建立共同气室、局部输运、共享物性、设备采集和观测扰动的唯一实现，不复制平衡物理。

计划动作：

1. 新建 `src/gf/sim/a2_dynamic_physics.py`。
2. 新建 `src/gf/sim/a2_sensor_devices.py`，集中实现超声 acquisition、TCD 电热状态和 NDIR active/reference；pipeline 不复制设备公式。
3. 实现 inlet profile、共同气室和三路局部输运解析更新。
4. 热导复用 `ar_he_co2.py` 的 WMS 共享算子；NDIR 通过注册 HITRAN2020 表直接完成 Voigt 吸收与 active/reference 带宽积分；声速通过配置显式选择 `a2dyn_direct_multifluid_eos_v1`，直接调用固定版本 CoolProp HEOS，并完成全网格、离网和压力方向的一致性核对。
5. 实现超声短波形、重复平均、滤波、互相关和可选相位精化；内部波形默认只在内存中存在。
6. 实现 TCD 能量平衡、温度电阻和桥路输出，以及 NDIR 高量程窄带、reference channel 和光电响应。
7. 实现 AR(1)、共享噪声、漂移和量化顺序，区分超声内部波形噪声与外层剩余噪声。
8. 建立 step、ramp、recovery、输运、ToF、能量守恒、NDIR 零点和饱和解析 fixture。
9. 对组成闭合、非负、单位、声速新旧迁移、热导 A1 parity、NDIR HAPI 零点/灵敏度/饱和和失锁显式失败做单元测试。

阶段门：

- 全部解析测试通过；
- 共享平衡算子没有第二实现；
- 超声估计器失败不回退到理论 ToF；
- 热导和 NDIR 不回退到旧线性电压加固定一阶曲线；
- 不存在 clip、静默归一化或默认成功；
- status 为 `PHYSICS_VERIFIED`。

`A2-DYN-1R4` 已完成当前声速生成定义与分层设备 smoke：CoolProp HEOS 直接算子在完整网格、10,000 个离网点和 `30,906` 个压力方向样本上分别得到最大相对差 `0`、`0` 和 `0` 个不一致；超声无理论 ToF 回退，TCD 能量、NDIR 饱和、组成闭合和协议 hash 检查通过。该状态的验证范围是 `generator_consistency`；它解除声速物性对 `A2-DYN-2` 的阻塞，但不替代后续 pilot 对采样率、设备候选、动态非退化和资源边界的资格审计。

### 12.3 A2-DYN-2：小规模 pilot、设备候选与采样率资格（3–5 天）

目标：在生成正式数据前验证时间尺度、动态非退化和数据量。

pilot 规模：

- 240 个唯一 `mixture_id`；
- 每个 family 40 个组；
- train、val、stress_val 均有小样本，暂不生成正式 test；
- 比较 1 Hz、2 Hz、5 Hz；
- 比较 120 s、240 s、360 s；
- 比较 `bandlimited_burst` 与 `linear_chirp`，以及 `reference_xcorr` 与 `reference_xcorr_parabolic`；
- 只运行 B-LAST、B-EWMA、B-STAT 和 O-KIN。

必须回答：

1. 5 Hz 是否解析共同气室、局部输运和三路低频输出；
2. 240 s 是否覆盖足够 transition 和 recovery；
3. 哪个超声 excitation 与估计器组合在精度、失锁率、SNR 和计算量间合格；
4. TCD 组成相关动态和 NDIR 高量程是否产生有效且未饱和的跨模态信息；
5. stress tier 是否仍可辨识；
6. 正式数组是否在目标资源范围内，并确认完整高频波形未进入数据包。

阶段门：

- 选出唯一采样率和时长；
- 选出唯一超声 acquisition profile；
- TCD 与 NDIR hardware profile 通过设备专用审计；
- 动态非退化门通过；
- 任何范围调整都回写配置并提高 protocol revision；
- status 为 `PILOT_QUALIFIED` 或明确失败终态。

`A2-DYN-2R4` 已执行完成：240 个唯一 `mixture_id`（六个 family 各 40，train / val / stress_val 为 120 / 60 / 60），在 5 Hz / 360 s 参考轨迹上一次调用 HEOS，再重采样比较 1 / 2 / 5 Hz 与 120 / 240 / 360 s；低频序列、TCD、NDIR、四个预注册基线和 36 点超声设备探针均写入摘要，不生成正式数据包或保存全量高频波形。5 Hz / 240 s 的动态非退化门全部通过（双通道有效率 `1.000`、量化级数门 `1.000`、低频 t50 双通道分离率 `0.9875`、六族退化率 `0`），TCD 最大能量残差约 `2.70e-14 W`、NDIR 饱和率 `0`、信号越界 `0`。超声候选均锁定率 `1.0`，最终按 p95 ToF 误差优先选择 `US-CHIRP-XCORR-PARABOLIC-1 / reference_xcorr_parabolic`（linear chirp p95 约 `3.19e-9 s`）；stress P060 的 O-KIN 相对 B-LAST 保留 `44.99%` 信息增益。正式数据配置和实验配置已同步选择，执行证据见 [pilot_audit_r4.json](../../outputs/summary/a2_dynamic_v1/pilot_audit_r4.json) 与 [A2-DYN-2R4 manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-2r4-pilot/manifest.json)。

### 12.4 A2-DYN-3：生成开发数据与难度审计（3–5 天）

目标：先生成 train、val、stress_val，验证各轴困难但可学习。

计划动作：

1. 生成 3,600 train、900 val、900 stress_val observation。
2. 运行 schema、守恒、EOS、声速新旧迁移、热导 A1 parity、NDIR HAPI 零点/灵敏度/饱和、超声失锁、TCD 能量、动态非退化、边界和 Jacobian 审计。
3. 运行 B-LAST 与两个 oracle。
4. 只让通过 `11.4` 的 family 进入基线比较。
5. 输出 `eligible_dynamic_axes.json`。

阶段门：

- 至少两个相互独立的动态压力轴合格；
- `D-IID` 必须合格；
- 不合格 family 保留证据，不静默替换参数；
- status 为 `DIFFICULTY_QUALIFIED`，否则停止。

### 12.5 A2-DYN-4：生成 test 与冻结数据（1–2 天）

> 已执行（2026-09-03）：`generate-test` 阶段生成 900 条 test 并聚合成完整包，冻结审计通过，见 [§22 A2-DYN-4 执行事实](#22-a2-dyn-4-执行事实)。

目标：在分布和难度已经由开发数据确认后生成完整数据包。

计划动作：

1. 生成 900 条 test observation。
2. 聚合为 6,300 条 observation 和 4,410 个 group。
3. 写 manifest、records、observations、oracle、device audit、有限 waveform fixtures 和 audit。
4. 重算 content hash、split hash、config hash 和 source hash。
5. test 与其他 split 可同文件存储，但 pipeline 默认不在开发命令中评价。

阶段门：

- 所有 hash 可重算；
- group 零交集；
- test 分布与冻结配置一致；
- status 为 `DATA_FROZEN`。

### 12.6 A2-DYN-5：时间信息基线与实时回放（4–7 天）

目标：判断时间维是否真的产生增量价值。

计划动作：

1. 运行 B-LAST、B-DELTA、B-EWMA、B-STAT、B-TCN、B-GRU 和 B-STEADY 五 seed。
2. 对 P005 至 P150 和 FULL 生成统一 predictions。
3. 运行 virtual-clock 全量回放和 wall-clock 小规模 smoke。
4. 计算 EarlyGain、AUEC、TTA、逐 family 指标和延迟。
5. 按 `11.5` 相对冻结的 B-REF 判定 `DYNAMIC_QUALIFIED` 或 `TEMPORAL_REDUNDANT`。
6. 检查 B-EWMA、B-STAT、B-TCN 与 oracle 距离，决定是否存在新算法 headroom。

阶段门：

- predictions 可重算全部 metrics；
- 不使用 test 调整模型或窗口；
- test 在冻结 recipe 后统一报告；
- 形成唯一数据终态。

### 12.7 A2-DYN-6：新算法交接或关闭（1–2 天）

若通过：

- 输出 `dynamic_handoff.json`；
- 冻结数据版本、合格轴、主 horizon、强基线和 oracle headroom；
- 新算法必须使用新 ID，不得使用 TQIF 名称或 checkpoint；
- 后续仍沿 A2 → A2H → A2M 主线验证。

若未通过：

- 输出明确失败状态和对应审计；
- 不启动新时序算法搜索；
- 不通过增加网络、seed 或隐藏窗口改变结论；
- 如需改变物理激励或标签任务，建立 v2 规划而非覆盖 v1。

## 13. 计划影响文件与职责

| 位置 | 计划文件 | 唯一职责 |
| --- | --- | --- |
| 数据配置 | `configs/data/ar_he_co2_a2_dynamic_v1.json` | 外层时间轴、数据分布、输运与 hardware profile、规模 |
| 评价配置 | `configs/eval/a2_dynamic_eval.json` | horizon、指标、资格门、seed、bootstrap |
| 实验配置 | `configs/experiment/a2_dynamic_protocol.json` | 工作包顺序、允许 split、pilot 轴、候选和冻结选择、产物目录 |
| 平衡物理 | `src/gf/sim/ar_he_co2.py` | 继续作为三路平衡公式唯一事实源 |
| A2-DYN 声速生成资产 | `configs/data/a2dyn_direct_heos_v1.json` | CoolProp HEOS 版本、source revision、二进制 hash、组成/温压域和运行入口 hash |
| A2-DYN 历史物性草案 | `configs/data/a2dyn_eos_coefficients_v1.json`、`configs/data/a2dyn_eos_coefficients_v2.json` | 仅作为旧近似与失败审计证据，不进入正式 R4 生成路径 |
| 动态物理 | `src/gf/sim/a2_dynamic_physics.py` | 进气协议、共同气室、局部输运和时序扰动 |
| 设备模型 | `src/gf/sim/a2_sensor_devices.py` | 超声短波形与 ToF、TCD 电热状态、NDIR 光学与参考通道 |
| A2-DYN-2 pilot | `src/gf/pipeline/a2_dynamic_pilot.py` | 固定 240 组 pilot、采样率/时长比较、设备探针、基线和资源门 |
| 数据生成 | `src/gf/sim/a2_dynamic_dataset.py` | 配方、observation、manifest、打包 |
| 数据审计 | `src/gf/sim/a2_dynamic_audit.py` | 守恒、EOS、设备状态、动态、边界、Jacobian、oracle |
| 数据适配 | `src/gf/dl/adapters/ar_he_co2.py` | 将动态包映射到 UnifiedSample |
| 基线 | `src/gf/dl/temporal_baselines.py` | B-LAST、B-DELTA、B-EWMA、B-STAT、B-TCN、B-GRU、B-STEADY 与 B-REF 选择 |
| 编排 | `src/gf/pipeline/a2_dynamic_benchmark.py` | generate、audit、baseline、replay、report |
| 测试 | `tests/test_a2_dynamic_*.py` | 解析输运、设备采集、schema、因果前缀、最小 smoke |
| 数据 | `data/a2_dynamic_v1/` | 聚合动态数据包 |
| 运行 | `outputs/runs/a2_dynamic_v1/` | 单次运行 manifest、预测和 checkpoint |
| 汇总 | `outputs/summary/a2_dynamic_v1/` | 指标、比较、合格轴和 hash |
| 报告 | `outputs/reports/a2_dynamic_v1/` | 数据与时间信息资格报告 |

不得在 pipeline 中再实现一套 profile 抽样或指标公式。配置是范围事实源，sim 是生成事实源，dl 是模型事实源，pipeline 只编排。

## 14. 计划 CLI

以下是阶段接口；当前已实现 `protocol`、`physics-smoke`、`pilot`、`generate-development`、`audit` 与 `generate-test`，其余阶段（`baselines`、`replay-smoke`、`report`）未实现时必须明确失败，不得返回伪造通过状态：

```powershell
python -m gf.pipeline.a2_dynamic_benchmark --stage protocol --project-root .
python -m gf.pipeline.a2_dynamic_benchmark --stage physics-smoke --project-root .
python -m gf.pipeline.a2_dynamic_benchmark --stage pilot --project-root .
python -m gf.pipeline.a2_dynamic_benchmark --stage generate-development --project-root .
python -m gf.pipeline.a2_dynamic_benchmark --stage audit --project-root .
python -m gf.pipeline.a2_dynamic_benchmark --stage generate-test --project-root .
python -m gf.pipeline.a2_dynamic_benchmark --stage baselines --project-root .
python -m gf.pipeline.a2_dynamic_benchmark --stage replay-smoke --project-root .
python -m gf.pipeline.a2_dynamic_benchmark --stage report --project-root .
```

CLI 只选择已经注册的 stage 和项目根。family、split 数、时间轴、阈值、seed 和模型矩阵不能通过临时命令行覆盖冻结配置。

## 15. 验证矩阵

| 层级 | 必做验证 |
| --- | --- |
| 单元测试 | 解析混合与局部输运、step/ramp、AR(1)、量化顺序 |
| 超声设备 | 波形时移、平均与滤波、互相关、相位精化、失锁、多径、内部采样率收敛 |
| 热导设备 | 电热能量守恒、稳态温升、组成相关幅值与时间常数、桥路边界 |
| NDIR 设备 | HITRAN 网格误差、active/reference 零点、高量程饱和与低浓度灵敏度 |
| 物理测试 | 闭合、非负、CoolProp HEOS 生成器一致性全网格与离网点对照、压力方向、A1 旧算子回归、热导 A1 parity、NDIR HAPI 零点/灵敏度/饱和、t50/t90、recovery |
| schema | shape、dtype、records 对齐、禁用字段、group 互斥、hash |
| 因果测试 | prefix 无未来、在线与离线同前缀一致、state reset |
| pilot | 1/2/5 Hz、120/240/360 s、超声候选、动态非退化、内存 |
| 难度 | Jacobian、condition number、oracle headroom、family 资格 |
| 基线 | B-REF 冻结、五 seed、相同 scaler/预算、逐 horizon、逐 family |
| 统计 | 2,000 次 mixture group bootstrap、逐组分退化界 |
| 实时 | virtual-clock 全量、wall-clock 小规模、p95 延迟 |
| 回归 | A1/A2H 文件 hash 不变、全量项目测试 |

代码实现阶段的最小命令：

```powershell
python -m pytest -q tests/test_a2_dynamic_physics.py
python -m pytest -q tests/test_a2_sensor_devices.py
python -m pytest -q tests/test_a2_dynamic_dataset.py
python -m pytest -q tests/test_a2_dynamic_pilot.py
python -m pytest -q tests/test_a2_dynamic_protocol.py
python -m pytest -q tests/test_a2_dynamic_benchmark.py
python -m pytest -q
git diff --check
git status --short
```

默认单元测试超时 60 s；正式生成、五 seed 训练和 bootstrap 不塞入普通单元测试。

## 16. 产物与报告

### 16.1 单次运行

```text
outputs/runs/a2_dynamic_v1/<run_id>/
  run_manifest.json
  resolved_config.json
  train.log
  metrics.json
  predictions.csv
  checkpoints/
  diagnostics/
```

### 16.2 predictions 最小字段

| 字段 | 说明 |
| --- | --- |
| `run_id` | 与 manifest 一致 |
| `model_id` | 基线或后续新算法 |
| `seed` | 固定 seed |
| `observation_id` | 行级追踪 |
| `mixture_id` | group 单位 |
| `split`、`family` | 分层 |
| `horizon_id` | P005 至 FULL |
| `cutoff_s` | 实际最大输入时间 |
| `horizon_valid` | 当前协议下该实时 horizon 是否仍位于首次有效暴露内 |
| `y_true_*`、`y_pred_*` | 三组分原尺度 |
| `latency_ms` | 当前预测耗时 |

一条 prediction 必须能由 `mixture_id + observation_id + model_id + seed + horizon_id` 唯一定位。

### 16.3 数据资格报告

正式报告至少包含：

1. 数据版本、配置和 hash；
2. group 与 observation 数；
3. family、split、组成区域分布；
4. 共同气室、局部输运、hardware、协议、环境、标定和噪声的实际采样分位数；
5. 物理守恒、EOS 对照、声速新旧迁移、热导 A1 稳态 parity 以及 NDIR HAPI 零点/灵敏度/饱和；
6. 超声估计误差与失锁、TCD 能量 residual、NDIR 饱和与量程审计；
7. 动态非退化、t50/t90 和跨模态滞后；
8. Jacobian 与 oracle；
9. B-REF 选择和全部可部署基线逐 horizon 结果；
10. 时间增量信息门逐项判断；
11. 实时延迟与生成峰值内存；
12. worst slice 和失败案例；
13. 唯一终态与是否允许新算法设计。

## 17. 风险与预注册处置

| 风险 | 识别信号 | 处置 |
| --- | --- | --- |
| 时间序列只是稳态复制 | B-LAST 与时序模型等价，轨迹由终点幅值决定 | `TEMPORAL_REDUNDANT`，停止复杂模型 |
| 输运、器件和采集再次混成统一一阶项 | profile 中出现公共 sensor multiplier 或无物理归属的 `tau_s` | `PHYSICS_INVALID`，回到分层状态修复 |
| 合成硬件参数被写成实测 | 结果只在某个 sensitivity profile 成立 | 限定为仿真敏感性，不写成硬件结论 |
| 超声估计器静默成功 | 低相关或多径错峰时输出恰等于理论 `L/c` | 明确失锁并拒绝 observation 或 profile |
| EOS 定义与审计口径 | A2-DYN 生成器和审计器未使用同一注册 HEOS 算子，或把生成器一致性误写成 HEOS 独立物理验证 | 生成器必须固定为 CoolProp HEOS；审计报告标记 `generator_consistency` 和 `independent_physics_validation=NOT_CLAIMED`，不得伪称独立验证 |
| 用 A1 parity 压回旧声速 | 新声速经 gain、delay 或查表后重新等于固定热容 A1 ToF | A1 只做旧算子回归；A2-DYN 超声按新物性和波形估计误差门验收 |
| 为通过 EOS 门修改范围或阈值 | 删除纯气端点、缩小温压块或把 `0.5%` 调高 | 判 `INVALID_PROTOCOL`；保留原网格和门限，修正物理模型并升 revision |
| 高频波形导致数据膨胀 | 正式包出现全量 ADC 数组或达到几十 GiB | 仅生成时临时使用，正式包只存 ToF 和有限 fixture |
| NDIR 高量程大面积饱和 | 高 CO₂ 平台或低 CO₂ 无可辨灵敏度 | `PHYSICS_INVALID`，修改量程并升版 |
| 压力过强导致不可辨识 | O-KIN 同时失败、Jacobian 退化 | `DYNAMIC_UNIDENTIFIABLE`，缩回并升版 |
| 外层采样率太低 | 过程或低频 t50 少于两个点、量化平台 | pilot 比较 1/2/5 Hz 后冻结 |
| 混淆外层与超声内部采样 | 试图以 2 Hz 离散载波或把 240 s 提升到 MHz | 强制双时间尺度接口 |
| 采样率过高造成冗余 | 相邻点高度重复且 5 Hz 无增益 | 保留较低采样率 |
| recovery 泄漏未来 | 只有 FULL 明显改善 | 不计入实时主结论 |
| protocol 与组成混杂 | 不同 family 的组成边际明显不同 | 重做配额匹配，不用模型修正 |
| 重复观测虚增样本 | 行级 bootstrap 给出过窄 CI | 强制 mixture group bootstrap |
| phase 字段泄漏 | 模型直接读取真实 phase 或 target onset | 默认不提供；另立 matched context arm |
| test 结果回流 | 看 test 后修改窗口、范围或 recipe | 当前结果只作探索，升版后重做 |
| 简单基线已饱和 | B-STAT 接近 oracle | `BASELINE_SATURATED`，不构造复杂新算法 |
| TQIF 被变相恢复 | 复用名称、checkpoint 或核心结构 | 拒绝交接，新候选必须新 ID 和新假设 |

## 18. 完成定义

### 18.1 设计完成

- [x] 明确既有 A1 正式数据为 `T=1`，A0 32 点只属 smoke；A2-DYN-3 开发数据已生成，A2-DYN-4 已生成 test 并聚合完整正式数据（2026-09-03）；
- [x] 明确共同气室、局部输运、平衡物性、设备采集和观测扰动五层物理链；
- [x] 明确超声双时间尺度、TCD 集总电热和 NDIR 高量程 active/reference 语义；
- [x] 明确时间轴、阶段、窗口和协议；
- [x] 明确六个数据族、4,410 个 group 和 6,300 条 observation；
- [x] 明确 schema、split、存储和永久不变量；
- [x] 明确物理、动态、可辨识性和时间增量信息门；
- [x] 明确工作包、影响文件、验证和杀停规则；
- [x] 明确不恢复 TQIF、不新增顶层阶段。

### 18.2 实现完成

- [x] A2-DYN-0 R4 机器协议冻结；
- [x] 分层动态物理与设备采集已实现，EOS 以外的 A2-DYN-1 解析和设备检查通过；
- [x] `a2dyn_direct_multifluid_eos_v1` 已直接使用固定 CoolProp HEOS，完整网格、离网、压力方向与设备门通过，A2-DYN-1 为 `PHYSICS_VERIFIED`；
- [x] A2-DYN-2R4 pilot 选定外层采样率、时长、超声 acquisition 和有效参数范围（`5 Hz / 240 s / US-CHIRP-XCORR-PARABOLIC-1`）；
- [x] 开发数据生成并通过难度审计（`DIFFICULTY_QUALIFIED`）；
- [x] test 和完整数据包生成、hash 冻结（A2-DYN-4 `DATA_FROZEN`，2026-09-03）；
- [ ] B-LAST、B-DELTA、B-EWMA、B-STAT、B-TCN、B-GRU、B-STEADY 与 oracle 五 seed 完整；
- [ ] 在线回放和延迟报告完整；
- [ ] 时间增量信息门形成唯一终态；
- [ ] 只有 `DYNAMIC_QUALIFIED` 才生成新算法 handoff。

## 19. A2-DYN-1 执行事实

### 19.1 已终止的近似声速路径

首轮固定热容模型与后续 pair-virial 模型都能通过绝对声速误差门，但压力方向分别有 `1,091 / 30,906` 和 `1,398 / 30,906` 个不一致。根因是截断维里近似与 HEOS 多流体 Helmholtz 状态方程不是同一生成定义；pair-v2 因此保持为草案失败证据，不进入正式 R4。

### 19.2 当前正式声速生成定义

`a2dyn_direct_multifluid_eos_v1` 直接调用固定版本 CoolProp 8.0.0 的 HEOS `speed_sound()`。适配层只负责显式气相、`Ar/He/CO2` 组分顺序、注册温压域、运行时版本和二进制 hash 校验，不实现第二套 EOS，也不提供旧模型回退。完整网格、离网审计和压力方向结果如下：

| 审计项目 | A2-DYN-1R4 结果 |
| --- | ---: |
| 协议与生成资产 | `PASS` |
| 完整网格 | 185,436 点，最大相对差 `0` |
| 固定 seed 离网审计 | 10,000 点，最大相对差 `0` |
| 压力方向 | `0 / 30,906` 不一致 |
| 共享物性与设备 smoke | `PASS`，无理论 ToF 回退 |
| 验证范围 | `generator_consistency` |

执行产物为 [physics_audit_r4.json](../../outputs/summary/a2_dynamic_v1/physics_audit_r4.json) 和 [A2-DYN-1R4 manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-1r4-physics-smoke/manifest.json)。`A2-DYN-1` 的声速生成门通过，允许进入 `A2-DYN-2`；这不等同于独立验证 HEOS 的物理准确性。

## 20. A2-DYN-2 执行事实

`A2-DYN-2R4` 已由 `src/gf/pipeline/a2_dynamic_pilot.py` 完成并返回 `PILOT_QUALIFIED`。执行严格使用实验配置中的 240 组、六族各 40 组和 train / val / stress_val = 120 / 60 / 60；不生成 test、不写入正式 `data/a2_dynamic_v1`，也不把 `noise_seed`、`sequence_id` 或高频 ADC 数组写入产物。

| 项目 | 结果 |
| --- | ---: |
| 外层比较 | 1 / 2 / 5 Hz；120 / 240 / 360 s |
| 冻结外层轴 | `5 Hz / 240 s` |
| 冻结超声 | `US-CHIRP-XCORR-PARABOLIC-1`，`reference_xcorr_parabolic` |
| 超声探针 | 36 点；锁定率 `1.000`；linear chirp p95 ToF 误差约 `3.19e-9 s` |
| 动态双通道有效率 | `1.000` |
| 低频 t50 双通道分离率 | `0.9875` |
| 六族动态退化率 | `0` |
| TCD 最大能量 residual | `2.70e-14 W` |
| NDIR 最大饱和率 / 信号越界 | `0 / 0` |
| stress P060 O-KIN 相对 B-LAST | `44.99%` 信息增益 |
| 正式信号数组估算 | `90,720,000` bytes（signals 单数组） |
| pilot 资源峰值 / 高频波形持久化 | 约 `226 MB`（实际值随进程运行记录） / `0` bytes |

为保证 HEOS 只在同一物理轨迹上比较，pilot 在 5 Hz / 360 s 参考网格上完成一次直接 HEOS 计算，低频候选只做外层重采样与时长截断；正式冻结为 5 Hz / 240 s。正式选择已回写到 `configs/data/ar_he_co2_a2_dynamic_v1.json` 和 `configs/experiment/a2_dynamic_protocol.json`。证据文件为 [pilot_audit_r4.json](../../outputs/summary/a2_dynamic_v1/pilot_audit_r4.json)、[A2-DYN-2R4 manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-2r4-pilot/manifest.json) 和 [resolved_config.json](../../outputs/runs/a2_dynamic_v1/a2-dyn-2r4-pilot/resolved_config.json)。

## 21. A2-DYN-3 执行事实

`A2-DYN-3` 已由 `src/gf/pipeline/a2_dynamic_benchmark.py` 完成并返回 `DIFFICULTY_QUALIFIED`。开发包只生成 train、val、stress_val，不生成 test；6 个 family 全部合格，`D-IID` 通过，5 个相互独立的动态压力轴进入 `eligible_dynamic_axes.json`，无失败要求。审计前强制重跑 physics smoke，并对 19 个源依赖和配置执行 freshness 校验，结果均为 `PASS`。

| 项目 | 结果 |
| --- | ---: |
| 唯一 `mixture_id` / group | `3,780` |
| train | 2,520 groups / 3,600 observations |
| val | 630 groups / 900 observations |
| stress_val | 630 groups / 900 observations |
| signals | `[5400, 3, 1200, 1]`，float32 |
| oracle clean / device state | `[5400, 3, 1200]` |
| 合格 family | `D-IID`、`D-KINETICS`、`D-PROTOCOL`、`D-NOISE-DRIFT`、`D-ENV-CAL`、`D-JOINT` |
| eligible 动态轴 | `D-KINETICS`、`D-PROTOCOL`、`D-NOISE-DRIFT`、`D-ENV-CAL`、`D-JOINT` |
| 动态非退化 / Jacobian | `PASS / PASS`；有效通道率与量化级数率均为 `1.000`，低频 t50 配对率 `0.8335`；固定目标、联合参数和剔除 nuisance 后目标满秩率均为 `1.000` |
| Jacobian 条件数 | 固定目标 P95 `45.56`；联合目标 P95 `38.88`，均低于门限 `1000` |
| 设备与边界 | NDIR 饱和率 `0`，信号越界率 `0`，超声锁定率 `1.000`；peak correlation `0.96494–0.97706`、SNR `318.37–397.32`、ToF 不确定度 `5.03e-10–6.28e-10 s`；TCD 最大能量 residual `2.7263e-14 W` |
| 数据 `content_sha256` | `82837b52b54f1d76ebc5b72a5eae796c931e3e86ad26ee52b7f1161661355a1d` |
| 审计 `audit_sha256` | `3f4ac9d19c614f88c235b2491c358c23b5d4b7b8438c61814153289fc8d8132b` |

基线资格门中各 family 的 B-LAST、O-EQ、O-KIN 拟合状态均为 `PASS`。B-LAST 使用显式记录的 `lbfgs`、`max_iter=2000`、`tol=1e-3`，不再硬编码成功状态；O-KIN 使用 clean device signal、inlet protocol 和特权 kinetics 参数，先以注册 HITRAN 曲线反演 CO₂，再在注册 HEOS 的 1% simplex 分段线性 ToF 曲线上做有界一维反演，不读取 `target` 或 `chamber_composition`。每个 family 的 O-KIN headroom 在 3 个早期前缀通过；D-IID、D-KINETICS、D-PROTOCOL 的相对退化在 2 个早期前缀通过，D-NOISE-DRIFT、D-ENV-CAL、D-JOINT 在 3 个早期前缀通过，满足至少 2 个前缀的阶段门。oracle 组成在 float32 序列化后最大和误差为 `0`，由显式序列化闭合逻辑保证，不靠审计放宽容差。

执行产物为 [a2_dyn_3_audit.json](../../outputs/summary/a2_dynamic_v1/a2_dyn_3_audit.json)、[eligible_dynamic_axes.json](../../outputs/summary/a2_dynamic_v1/eligible_dynamic_axes.json)、[数据 manifest](../../data/a2_dynamic_v1/manifest.json)、[audit.json](../../data/a2_dynamic_v1/audit.json) 和 [A2-DYN-3 audit manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-3-development/audit_manifest.json)。A2-DYN-4 的 test 生成与完整数据冻结已执行，见 [§22](#22-a2-dyn-4-执行事实)。

## 22. A2-DYN-4 执行事实

`A2-DYN-4` 已由 `src/gf/pipeline/a2_dynamic_benchmark.py --stage generate-test` 完成并返回 `DATA_FROZEN`（2026-09-03）。生成前先重跑 physics smoke 并强制 `PASS`；开发子集（3,780 groups / 5,400 observations）的 manifest 与 audit 先备份到 [development_subset_backup](../../outputs/runs/a2_dynamic_v1/a2-dyn-4-test/development_subset_backup/)（内容 hash `82837b52…` 与 A2-DYN-3 存档一致），随后 test 配方从同一低差异池中、以开发 3,780 个已占用坐标之外的唯一点抽取；3 个规范 pure 顶点按配置替换 D-JOINT/test 的 3 个 binary 配额，固定在该 family 尾部，不混入数字编号流（test 数字组 ID 连续为 `a2dyn-mix-0003781…0004407`）。聚合写盘后执行冻结审计（schema / 守恒与设备 / test 动态非退化 / pure 边界四类），并重算 content hash 与 audit hash。执行过程中修正了两处实现缺陷：数组拼接的键映射错误（`inlet_composition` vs `inlet`）与 pure 顶点边界审计把 pure-He / pure-CO2 误当 purge 恒等序列；两者均导致审计显式失败后修复，未引入静默降级。

| 项目 | 结果 |
| --- | ---: |
| 唯一 `mixture_id` / group | `4,410` |
| train | 2,520 groups / 3,600 observations |
| val | 630 groups / 900 observations |
| stress_val | 630 groups / 900 observations |
| test | 630 groups / 900 observations |
| signals | `[6300, 3, 1200, 1]`，float32 |
| test 区域配额（interior / near_boundary / binary / pure） | 组级 `315 / 189 / 123 / 3`，行级与冻结配置一致 |
| pure 顶点 | `a2dyn-mix-pure-Ar/He/CO2`，仅 D-JOINT/test，各 2 条观测 |
| 开发 ↔ test group / 组成零交集 | `PASS`（4,410 组全部唯一；非纯气坐标 4,407 个互不重复） |
| 完整包 schema 审计 | `PASS`（records / groups / 区域 / hash / 数组不变量全部通过） |
| 守恒与设备审计（6,294 非 pure 行） | `PASS`：信号越界 `0`、NDIR 饱和率 `0`、超声锁率 `1.000`、TCD 最大能量残差 `2.7263e-14 W` |
| test 动态非退化（897 非 pure 行） | `PASS`：有效通道率 `1.000`、量化级数率 `1.000`、t50 配对率 `0.9004`、六族退化率 `0` |
| pure 边界审计 | `PASS`：pure-Ar 目标等于 purge 且 clean 全程静态；pure-He / pure-CO2 clean 有预期动态；越界 `0`、饱和 `0`、锁率 `1.000` |
| 完整包 `content_sha256` | `3da0e478eca52bb6a31e1fe2c2d5b3d066341fec3516be1a29fe7ed3077aeb95` |
| 冻结审计 `audit_sha256` | `87344bb532ab56dfd0a28b6b938c8b9641e9231725043a3869bceda75a9a54f0` |
| 开发子集 content（存档，不覆盖） | `82837b52b54f1d76ebc5b72a5eae796c931e3e86ad26ee52b7f1161661355a1d` |
| physics smoke 重跑 / dataset freshness | `PASS / PASS`（源依赖 hash 已在最终冻结时重绑定） |

test 与开发 split 可同文件存储（§8.1），`data/a2_dynamic_v1/audit.json` 已替换为 A2-DYN-4 冻结审计；A2-DYN-3 难度审计完整存档于 `outputs/summary/a2_dynamic_v1/a2_dyn_3_audit.json`、`outputs/runs/a2_dynamic_v1/a2-dyn-3-development/audit_manifest.json` 与 `development_subset_backup/`。执行产物为 [a2_dyn_4_freeze_audit.json](../../outputs/summary/a2_dynamic_v1/a2_dyn_4_freeze_audit.json)、[A2-DYN-4 manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-4-test/manifest.json) 和 [resolved_config.json](../../outputs/runs/a2_dynamic_v1/a2-dyn-4-test/resolved_config.json)。A2-DYN-5 的基线、B-REF 冻结、在线回放与时间增量信息门仍未执行。
