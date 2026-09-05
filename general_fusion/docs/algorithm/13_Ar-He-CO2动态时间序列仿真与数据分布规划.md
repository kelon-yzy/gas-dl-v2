# 下一轮 A2：Ar–He–CO₂ 动态时间序列仿真与数据分布规划

> **暂停注记（2026-09-05）**：[项目总体规划](../../项目总体规划.md) v11 §2.2 已暂停本工作包的后续扩建、完整 v2 生成和新增时序架构筛选，并且不把动态 v2 冻结设为新研究的前置条件。本文与从属的 13a / 13b 继续作为 v1 数据、前向模型、扰动链、因果前缀和基线工具的规格与执行事实源供复用，**已发生的机器终态、阈值和 hash 不修改**；其中「后续动作」「下一步」类条文不再排期。`15_A2-DYN审计缺陷修复规划.md` 已于同日归档至 [archive/](archive/README.md)。

> 制定日期：2026-08-31 \
> 文档状态：`DESIGN_COMPLETE / A2-DYN-0_R4_PROTOCOL_FROZEN / A2-DYN-1R4_EXECUTED / A2-DYN-1_PHYSICS_VERIFIED / A2-DYN-2R4_PILOT_QUALIFIED / A2-DYN-3R2_DIFFICULTY_QUALIFIED / A2-DYN-4_TEST_GENERATED / A2-DYN-4R2_FREEZE_FAILED_KNOWN_DEFECT / A2-DYN-5_DEVELOPMENT_COMPLETE_TEMPORAL_REDUNDANT / A2-DYN-5_FORMAL_BLOCKED / A2-DYN-6_BLOCKED_DATA_FREEZE_FAILED` \
> 工作包定位：下一轮 A2 的数据与问题重定义子工作包，不新增 A2I 或平行顶层阶段 \
> 上位事实源：[项目总体规划](../../项目总体规划.md) \
> 相关契约：[统一任务与接口契约](01_统一任务与接口契约.md)、[A1 数据与物理规格](05_A1数据与物理规格.md)、[A2H 分步执行计划](archive/09_A2H分步执行计划.md)（已归档） \
> 传感器证据：[传感器仿真文献调研与 A2-DYN 对比](14_传感器仿真文献调研与A2-DYN对比.md) \
> 声速生成定义：CoolProp HEOS 8.0.0 直接作为 `a2dyn_direct_multifluid_eos_v1` 的合成声速算子；历史 pair-v2 仅保留为失败证据 \
> 历史失败边界：[TQIF 失败归档](12_TQIF新算法设计与A2至A2M验证计划.md)

## 文档结构与归属（2026-09-04 D5 拆分）

主规划回答"要做什么、为什么、什么条件下停"；以下两类内容已按职责拆分，状态由本文件统一持有：

| 承接文件 | 承接的 §13 节 | 内容 |
| --- | --- | --- |
| [13a_A2-DYN执行记录.md](13a_A2-DYN执行记录.md)（从属于 13） | §19–§22（及后续 R2 执行事实节） | 各阶段已发生的执行事实 |
| [13b_A2-DYN物理与审计规格.md](13b_A2-DYN物理与审计规格.md)（从属于 13） | §4、§6、§8、§10、§13、§15 | 被代码与主规划共同引用的稳定规格 |

三份文件节号沿用 §13 编号，跨文件引用写作「13b §x」「13a §x」。正文中被迁出节的位置保留指针行。

## 0. 结论与执行摘要

当前 Ar–He–CO₂ 正式 A1、A2、A2H 和 A2M 数据都是 `T=1` 的稳态标量。仓库虽已有 A0 的 32 点一阶响应 smoke，但它只有 3 个配方、固定时间常数和单次纯 Ar 到目标气体的确定性过渡，只证明统一接口能承载时间维，不能作为正式动态 benchmark。

下一轮 A2 在设计新算法前，先增加 `A2-DYN` 动态数据子工作包。它要建立真正随时间演化的气室组成、进气协议、传感器响应、时序噪声和在线前缀评价，而不是把一个稳态值重复 1,200 次。当前 R4 机器协议已冻结；开发数据已生成，难度审计在 R2 修正口径（配对总体 + O-KIN-OBS 噪声受限 headroom）下为 `DIFFICULTY_QUALIFIED`（A2-DYN-3R2，2026-09-04，见 [13a §23](13a_A2-DYN执行记录.md)）；A2-DYN-4 已生成 test 并聚合为 6,300 观测 / 4,410 组的完整数据包，内容 hash 冻结（`3da0e478…`）但 R2 口径的冻结审计失败（test NOISE-10X 两族 11–20% 行低于可辨识线），冻结状态降级为 `DATA_FREEZE_FAILED` 并登记缺陷（见 [13a §24](13a_A2-DYN执行记录.md)），v1 数据保留不重新生成。A2-DYN-5 已完成 train / val / stress_val 基线、B-REF、动态指标和因果回放，开发证据为 `TEMPORAL_REDUNDANT`；正式 test 与完整阶段终态仍由 `DATA_FROZEN` 硬门阻塞。A2-DYN-6 已生成显式阻断记录，未生成新算法 handoff；v1 不以限制口径绕过该门。

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
| `P150` | 150 s | 900 | recovery 前完整暴露；cutoff 与 steady/recovery 边界重合（标准 onset 下恰为 180 s），受 phase duration jitter 影响约半数序列的该行失效，只作副参照（13a §23/§24 实测 P150 有效行 31–141，不参与门判定） |
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
| `O-EQ` | clean equilibrium reference signal | 与 B-LAST 同模型类的小 MLP（同结构、同 seed、同标准化） | 同模型类、clean 平衡输入的上界；若实测劣于 B-LAST 则降级为参照并单独记录 |
| `O-KIN` | clean device signal 加真实输运与 hardware 参数 | 特权数值 oracle | 前向模型可逆性上界（无噪输入），不衡量可部署 headroom，只报告不判定 |
| `O-KIN-OBS` | 最终观测信号（含标定、漂移、AR(1)、白噪、量化）加真实输运与 hardware 参数 | 与 O-KIN 完全相同的反演算子 | 噪声受限 headroom 参照；§11.4 门 2 与 §11.6 均以它为准 |

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

1. `B-LAST` 在 P015、P030、P060 中至少两个 horizon 相对晚期参照退化 ≥25%，证明早期问题不是稳态复制。晚期参照主值为 P120，副值为 P150；比值的分子与分母**必须在同时于早期 horizon 与晚期参照 horizon 有效的同一批行**上计算（`horizon_valid` 同时为真）。任一 family 在主参照上的配对行数低于 60 时，该 family 判 `FAILED` 而不是照常出数。P150 因与 recovery 边界重合会损失约半数序列，只作副值报告，不参与判定。
2. `O-KIN-OBS`（与 O-KIN 完全相同的反演算子和特权动力学参数、输入最终观测信号）在相同早期 horizon 相对 `B-LAST` 保留 ≥20% 的误差改善空间；headroom 与 B-LAST 在 O-KIN-OBS 反演成功的同一批 val 行上配对计算，反演失败按行显式计入 `inversion_failure_fraction`，不静默丢弃、不放宽反演容差。`O-KIN`（clean 输入）保留为前向模型可逆性上界，只报告不判定。
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
- 在 P150 上相对 `B-STEADY` 的 macro_RNMAE 退化不超过 5%，避免用牺牲稳态准确率换取早期表面收益；该比值同样必须在 `B-STEADY` 与被测模型同时有效的同一批行上配对计算，P150 因 recovery 边界失效的序列不得进入分子或分母。

门通过才形成 `DYNAMIC_QUALIFIED`。只在 FULL 或 recovery 后改善不算实时增量信息。

### 11.6 新算法 headroom 门

动态数据合格后，还要判断是否值得设计复杂方法。本节的 oracle 一律指 `O-KIN-OBS`（噪声受限 headroom 参照）；`O-KIN` 只证明前向模型可逆，不作为 headroom 依据：

- 若 `B-EWMA` 或 `B-STAT` 已进入 `O-KIN-OBS` 的 5% 等价带，优先保留简单方法；
- 若 `B-TCN` 相对 `B-STAT` 无稳定收益，不能仅因“深度学习”继续扩展 Transformer；
- 若 `B-TCN` 已接近 `O-KIN-OBS`，但在线成本高，后续研究问题只能转向效率或压缩，不能声称更高精度 headroom；
- 只有至少一个合格压力轴仍保留 ≥10% 的 `O-KIN-OBS` headroom，才进入新融合算法构思。

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

> 已执行（2026-09-03）：`generate-test` 阶段生成 900 条 test 并聚合成完整包，R1 口径冻结审计通过；2026-09-04 的 R2 修正口径重审失败（`DATA_FREEZE_FAILED`，test NOISE-10X 可辨识缺陷登记），执行事实见 [13a §22](13a_A2-DYN执行记录.md) 与 [13a §24](13a_A2-DYN执行记录.md)。

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

> 当前门状态：仅开发侧实现解除阻塞，允许读取 train、val 和 stress_val；由于 A2-DYN-4R2 为 `DATA_FREEZE_FAILED`，正式 test 评价、完整阶段终态和 `DYNAMIC_QUALIFIED` 均保持阻塞。恢复正式执行必须由 v2 数据通过 `DATA_FROZEN`，不得把 v1 test 缺陷改写为受限解锁。

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

## 14. 计划 CLI

以下是阶段接口；`protocol`、`physics-smoke`、`pilot`、`generate-development`、`audit`、`generate-test`、`baselines`、`replay-smoke`、`report` 与 `handoff` 均已实现。由于 A2-DYN-4R2 为 `DATA_FREEZE_FAILED`，后四个阶段当前只允许开发侧 train / val / stress_val，并在产物中显式保留正式 test 硬门；`handoff` 会写出 A2-DYN-6 阻断事实而不会伪造 handoff。任何未实现或门禁失败都必须明确失败，不得返回伪造通过状态：

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
python -m gf.pipeline.a2_dynamic_benchmark --stage handoff --project-root .
```

CLI 只选择已经注册的 stage 和项目根。family、split 数、时间轴、阈值、seed 和模型矩阵不能通过临时命令行覆盖冻结配置。

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
| 晚期参照 horizon 与 phase 边界重合 | 某 horizon 的 `horizon_valid` 行数显著低于早期 horizon；B-LAST 误差随 horizon 非单调 | 参照 horizon 必须落在 phase 内部而非边界；判据改配对总体（主参照 P120、副 P150）；不通过挪动 phase 时长补救 |
| 高噪声档压过可辨识线 | test 冻结审计中某噪声档的 active fraction 或退化率超标（R2 实测 NOISE-10X 族 11–20%） | 按预注册门判 `DATA_FREEZE_FAILED` 并登记缺陷；不放宽 5σ 判据或退化率门限；v1 test 只保留冻结审计证据，不进入 A2-DYN-5 模型评价；v2 数据规划时调整 test 噪声档分布并重新冻结 |
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
- [x] 开发数据生成并通过难度审计（`DIFFICULTY_QUALIFIED`；2026-09-04 R2 修正口径重审计仍 `DIFFICULTY_QUALIFIED`，终态记 A2-DYN-3R2，见 [13a §23](13a_A2-DYN执行记录.md)）；
- [x] test 和完整数据包生成（2026-09-03）；R1 口径冻结审计通过并 `DATA_FROZEN`，2026-09-04 R2 修正口径重审失败降级为 `DATA_FREEZE_FAILED`（缺陷登记，见 [13a §24](13a_A2-DYN执行记录.md)），内容 hash `3da0e478…` 不变，v1 数据不重新生成；
- [x] A2-DYN-5 开发侧 B-LAST、B-DELTA、B-EWMA、B-STAT、B-TCN、B-GRU、B-STEADY 与 oracle 矩阵完成（train / val / stress_val；正式 test 仍被 `DATA_FROZEN` 硬门阻塞）；
- [x] 开发侧在线回放和延迟 smoke 完成；正式 test 延迟未宣称；
- [x] 开发侧时间增量信息门已形成证据（当前候选为 `TEMPORAL_REDUNDANT`）；正式阶段终态仍为 `FORMAL_BLOCKED_DATA_FREEZE_FAILED`；
- [x] A2-DYN-6 handoff/关闭阶段已执行并写出 `A2_DYN_6_BLOCKED_DATA_FREEZE_FAILED`；未生成 `dynamic_handoff.json`，未启动新算法搜索；
- [ ] 只有 `DYNAMIC_QUALIFIED` 才生成新算法 handoff。

