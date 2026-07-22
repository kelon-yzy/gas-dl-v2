# tv3 双向超声（F 线）实施规划

> 状态：**F5-S 代码已落地（2026-07-22）**：`bidir_spxy_observed_ab_v1` + S-Y/S-L×3 seeds 十二格判据 (d) 已接线；`stage_status.f5*=f5s_code_ready_awaiting_formal_matrix`。F0–F4 / F*-wide 已通过。**下一步：smoke 端到端 → 服务器正式 `tv3-bidir-6000`（及 `-wide`）矩阵**；F6 未执行。
>
> 责任：给出恢复双向声学路线的完整实施契约：物理与观测模型、数据 schema、部署级估计器、F0–F6 阶段门与 verdict 分流。本计划立项**不**改写 v1 `information_source_upgrade_required`、**不**撤销静止空气 P0、**不**替换 B7 默认头；执行启动以本文前置条件为准。
>
> 命名：本线阶段前缀为 **F**（flow）。数据契约 `tunnel-ventilation-bidir-1`，benchmark `tv3-bidir-*`，正式 builder `raw_dsp_bidirectional_v1`。

---

## Context

### 1. 为什么是双向：v1 审计定位的信息缺口

冻结 v1 单向 TOF 可辨识性审计（`outputs/tv3_identifiability/`，2026-07-13）给出三个事实：

1. 单一 TOF 观测下 `O₂ / CO₂ / T / L` 联合 Fisher 秩为 1，nuisance 无法边缘化；
2. 1 K 温度 + 3 μs trigger jitter 情景的 0.8% 窄窗口联合 P90 为 8.85–12.99 vol% O₂，远超 `0.4 vol%` 门；
3. `flow_projection` 为 `not_represented` 且 `blocks_go_verdict=true`，`v1_blocking_nuisance_reject_all` 拒绝 189/189 点。

其中第 3 条是结构性的：掘进巷道按《煤矿安全规程》必须维持风流（岩巷 ≥0.25 m/s、煤巷 ≥0.5 m/s，上限 4 m/s），部署中 flow 不是可选情景而是必然存在的量。单向 TOF 中流速与声速线性混叠 `t = L/(c ± v_path)`，任何合规风速都会污染组分反演（量化见下节）。双向对射测量是声学流量计与超声风速仪的标准解法（transit-time 原理）：

```text
t_ab = L / (c + v_path) + τ_ab + j_ab        # 顺流
t_ba = L / (c - v_path) + τ_ba + j_ba        # 逆流

ĉ      = (L / 2) × (1/t'_ab + 1/t'_ba)       # t' = t_meas − τ̂_dir（延迟校正后）
v̂_path = (L / 2) × (1/t'_ab − 1/t'_ba)
```

关键性质（解析可证，进入 F1 单元测试）：

- 均匀轴向流下 reciprocal-sum 声速估计器**对 v 精确**（无一阶、无二阶项）；非均匀流剖面下残余偏差为 `−Var(v)/c`（相对量 `Var(v)/c²`），量级可忽略（见预算表）。
- 共模固定延迟 `τ̄=(τ_ab+τ_ba)/2` 对 ĉ 的偏置与单向情形相同，仍由延迟校正处理；方向不对称 `δτ=τ_ab−τ_ba` 只进入 v̂（偏置 `−c²δτ/(2L)`，0.1 μs → ≈0.024 m/s），对 ĉ 为二阶。
- v̂ 是 flow 的直接观测，使 flow 从 `not_represented` 变为 `implemented_physics + estimable`，解除 v1 的阻断拒绝规则——这正是 verdict 要求的"信息源升级"中成本最低的一种（同一对换能器互为收发，无新传感原理）。

### 2. 先验误差预算（立项依据，同时是 F4 的 sanity 锚点）

参考点：`x_CO2=1%, x_O2=20%, x_N2=79%, T=20°C` → `c≈342.9 m/s`，`TOF@L=0.25m≈729 μs`。O₂ 灵敏度：1 vol% O₂↔N₂ 交换 → `δc/c ≈ −7.1×10⁻⁴`（≈0.24 m/s，ΔTOF≈0.51 μs @0.25 m）。γ 项贡献仅 −3.5×10⁻⁵/vol%（O₂/N₂ 同为双原子），信号本质来自摩尔质量差。

| 误差项 | 情景 | 等效 O₂（单帧） | 64 帧序列后 | 性质 |
| --- | --- | ---: | ---: | --- |
| flow 未校正 | v_path=1 m/s | 4.1 vol% | 4.1 vol% | 偏置，不随平均下降 |
| flow 未校正 | 规程范围 0.25–4 m/s | 1.0–16.5 vol% | 同左 | 偏置 |
| flow 双向解耦后残余 | σ_v,turb=0.4 m/s 剖面非均匀 | <0.01 vol% | <0.01 vol% | `−Var(v)/c` 二阶项 |
| trigger jitter | 3 μs（v1 保守登记） | 5.8 vol% | 0.73 vol%(1σ) / P90≈1.2 | 随机，√N 下降 |
| trigger jitter | ≤0.5 μs（待 F0 依硬件规格重登记） | 0.97 vol% | P90≈0.20 vol% | 随机 |
| 温度误差 | 1 K | 2.4 vol% | 2.4 vol% | 偏置/慢漂 |
| 温度误差 | 0.1 K | 0.24 vol% | 0.24 vol% | 偏置/慢漂 |
| 固定延迟残余 | δτ̂=0.1 μs | 0.19 vol% | 0.19 vol% | 偏置，session 内稳定 |
| 声程误差 | 0.5 mm 未标定 | 2.8 vol% | 2.8 vol% | 偏置，由 span 标定吸收 |
| 峰值定位噪声 | 0.02 sample（D2b 实测水平） | 0.04 vol% | <0.01 vol% | 随机 |

三条立项判断：

1. **双向消除的是当前最大且不可平均的单项**。任何合规风速的未校正偏置（≥1.0 vol%）都单独超过 0.4 vol% 门；解耦后该项低于 0.01 vol%。
2. **双向是必要条件，不是充分条件**。解除 flow 阻断后，主导项转为温度感测（需 ~0.1 K 级）与 trigger jitter 登记值（3 μs 是工程保守情景；NI 硬件触发 DAQ 的实际 jitter 为 ns–亚 μs 级，F0 必须按 datasheet 重推导并双情景登记）。0.4 vol% P90 是否可达由 F4 定量回答，本表只提供先验量级。
3. **延迟标定策略影响精度上限**。多 L 截距自标定（斜率 1/c、截距 τ）虽然消除 τ 偏置，但杠杆臂只有 std(L)≈0.035 m，jitter 噪声放大约 7 倍；正确做法是慢时间尺度（session 级、多帧聚合）估计 τ̂，序列内用固定 τ̂ 做平均。此结构写入 F3 估计器设计。

### 3. 双向能解决什么、不能解决什么

| 能 | 不能 |
| --- | --- |
| 解除 v1 flow 阻断拒绝，恢复可辨识性审计的量化意义 | 不改变 `c=√(γRT/M)` 中 T–M 混叠：ĉ 仍是一个方程两个未知量，T 必须由外部测量承担 |
| 把流速从 nuisance 变成副产品输出（风速监测本身有安全价值） | 不突破 0.8% 窄窗口物理墙的结论（oracle 仍全负），不承诺窄窗精细回归 |
| 提供 reciprocity 残差作为链路健康与质量信号 | 不构成真实硬件 / Sim2Real 证据，全部结论限定在已登记仿真分布内 |
| 消除延迟共模项与流速偏置的耦合，改善 span 标定的可迁移性 | 不替代 TDLAS：若 F4 后 T/jitter 预算仍不达标，`information_source_upgrade_required` 对"更高精度 O₂"仍然成立 |

### 4. 与现有路线的关系

- **静止空气 P0（S 线）不被撤销**：S 线量化 `flow=0` 边界内单向链路的极限，其 S0 参数登记与 S2 审计机制是 F 线的直接前置资产（F0 复用 registry 结构，F4 复用灵敏度/Fisher/P90 实现）。F 线的 `flow=0` 锚点子集与 S 线结果互为对照。
- **COMSOL P1 不受影响**：隧道输运线（G 系列）继续独立推进；若后续需要非均匀流剖面的 `Var(v)` 依据，G 线 CFD 是天然来源（登记为 `literature_bound`→`implemented_physics` 的升级路径）。
- **v1 verdict 不改写**：v1 是单向情景的正式历史结论。F4 产物写入独立目录（identifiability v2），v1 目录不可覆盖。
- **B1/B7/E1d-SB 全部冻结**：F5 模型对照只使用冻结头配方在新数据契约上重训，不修改任何现有正式产物。

---

## Task

### 1. 目标与非目标

目标：

1. 在仿真链路中显式表示轴向流传播，生成双向（AB/BA）波形与逐方向 observed 数组；
2. 实现部署级双向 DSP 估计器（延迟自标定 + reciprocal-sum 声速 + 流速 + reciprocity 残差），冻结为 `raw_dsp_bidirectional_v1`；
3. 以 v1 同门限（P90 ≤0.4 vol%、单项 nuisance ≤50%、拒绝率 ≤5%）完成解除 flow 阻断后的可辨识性 v2 审计；
4. 在 `tv3-bidir-6000` 上完成五臂模型对照与 S-Flow holdout，量化双向相对单向的组分反演增益。

非目标：

- 不做真实硬件、风洞或现场实验声明；不宣称 Sim2Real。
- 不启动多频超声、E2 FiLM/attention/MoE、TDLAS 硬件；不晋升 LS。
- 不修改 raw3 输出契约、B1/B7、`d0_raw_dsp_physics_stats_v1`、v1 审计产物与 S 线计划。
- 不在本线内建模湿空气新物理与设备 profile 分布（沿用现有表示；`acoustic_measurement_v2` 属独立后续）。

### 2. 物理与观测模型（F0 冻结项）

| 项 | 契约 | 来源类型 |
| --- | --- | --- |
| 流速表示 | 直接采样 `v_path_m_per_s`（沿声路投影，签名区分方向），序列内恒定 | `engineering_scenario`（几何投影抽象） |
| 采样范围 | `v_path ∈ [−4.0, +4.0] m/s` 约束 LHS；≥10% 序列固定 `v_path=0` 作 S 线对照锚点 | `literature_bound`（煤矿安全规程风速界） |
| 湍流波动 | v1 版不表示（序列内恒定），在 registry 显式标 `not_represented`，并给出 `−Var(v)/c` 先验上界说明其非阻断 | 登记缺口 |
| 传播模型 | `t_true,ab = L/(c+v_path)`、`t_true,ba = L/(c−v_path)`；衰减双向同 `α(f)`；束偏移/折射不表示（标注） | `implemented_physics`（落地后） |
| 时序拓扑 | ping-pong 交替对射：同一 timestep 内 AB、BA 两次独立发射（间隔 ≤2.5 ms，登记 `pair_interval_s`），两条 5 ms 接收窗 | 设计决策 |
| jitter 相关性 | 基线 `independent`（每次发射独立 jitter）；`shared_trigger` 作为可选情景开关（同步交叉发射的理想化） | 设计决策 + 情景 |
| 固定延迟 | `τ_ab = τ_ba = 82 μs` 基线 + 可配置不对称 `delay_asymmetry_s`（默认 0，扫描情景 ≤0.2 μs） | `engineering_scenario` |
| trigger jitter 值 | F0 必须依 DAQ/触发硬件规格重推导：保守情景保留 3 μs，新增 nominal 情景（预期 ≤0.5 μs），双情景全程并行报告 | `literature_bound`（datasheet） |

明确不表示（registry 中标注，不得伪装为已验证零贡献）：湍流时变、速度剖面非均匀性、束漂移与折射、Doppler 展宽、换能器角度安装误差。

### 3. 数据契约

- schema：`tunnel-ventilation-bidir-1`（新模块 `tv3/sim/core/tunnel_ventilation_bidir_schema.py`，组分/慢通道复用基 schema：`COMPONENT_FIELDS` 与 7 个 `SLOW_CHANNELS` 不变——**flow 不进 slow 通道**，部署中它由声学对推断而非独立仪表测量）。
- condition grid 新增：`v_path_m_per_s`（真值，仅 oracle/审计可用）、`flow_scenario`、`pair_interval_s`、`delay_asymmetry_s`、`jitter_correlation`。
- 波形数组（沿用 packaging 命名规则，成对新增）：

```text
sequences/ultrasonic_ab_int16.npy + ultrasonic_ab_scale.npy
sequences/ultrasonic_ba_int16.npy + ultrasonic_ba_scale.npy
sequences/ultrasonic_tof_observed_ab_s.npy / _ba_s.npy
sequences/ultrasonic_peak_index_ab.npy / _ba.npy
sequences/ultrasonic_tof_quality_ab.npy / _ba.npy
sequences/ultrasonic_tof_accepted_ab.npy / _ba.npy
sequences/ultrasonic_tof_true_ab_s.npy / _ba_s.npy          # oracle
sequences/ultrasonic_v_path_true_m_per_s.npy                 # oracle
sequences/ultrasonic_sound_speed_m_per_s.npy                 # oracle（介质 c）
```

- manifest：`sim_revision.tag = "v7-bidir-flow-v1"`、`physics_backend` 追加 flow 模型标识、`schema=tunnel-ventilation-bidir-1`、flow registry 摘要与 jitter 双情景登记。缺任一必需数组或元数据直接失败。
- benchmark：`tv3-bidir-smoke`（本地链路验证）→ `tv3-bidir-6000`（服务器正式）。存储估计：双向两条 5 ms int16 窗 ≈ 7.7 GB@6000（现单向约 3.8 GB 的两倍）；保持 5 ms/方向不裁剪（避免触碰窗口长度相关代码），裁剪至 2.5 ms/方向留作后续存储优化项。
- 组分/环境采样沿用 `tv3-formal-6000` 的约束 LHS 与 L 扫描（steady 0.18–0.28 m 五档），保证与现有结论可比。
- **组分域变体（2026-07-22 立项，A1 仅 F 线 + B1 并行留档已确认）**：本计划 F0–F5 的组分域为窄域（CO₂ 0.03–5%、O₂ 18–21.2%）。另立**独立注册宽域**（CO₂ 0.03–10%、O₂ 15–25%，覆盖 OSHA 缺氧线 <19.5%/富氧火险线 >23.5% 与超 IDLH CO₂ 积聚），作用范围仅 F 线、与窄域并行留档，`-wide` 后缀独立命名与目录，不覆盖窄域任何冻结产物（含 v1/F3/F4/registry dc61d9e7）。误差预算锚点 O₂=20% 仍在宽域内，§Context.2 先验表沿用；宽域不改写 `coarse_monitoring_only` 物理墙，只拓宽部署包络。完整改动契约与分阶段步骤见 `docs/active/tv3_composition_range_widening_plan.md`。

### 4. 部署级估计器与 builder（F3 冻结项）

`raw_dsp_bidirectional_v1` 输入仅限部署可得：双向波形 + 7 slow 通道 + 配置登记元数据。分四层：

1. **帧级逐方向**：复用 D2b 匹配滤波（train-only baseline median 模板，逐方向独立模板与 digest），输出连续峰位、`tof_obs`、SNR、PSR、quality、accepted。
2. **session 级延迟自标定**：对每个方向，用 train split 的多帧多 L 稳态帧做鲁棒回归 `t_dir(L) = τ_dir + L/c_eff,dir`，取截距为 `τ̂_dir`；train-only、随 run config 固化数值与 digest（与 E1r 模板锚点同一纪律，禁止 exact simulator 值）。
3. **帧对物理量**：`ĉ = (L/2)(1/t'_ab + 1/t'_ba)`、`v̂_path = (L/2)(1/t'_ab − 1/t'_ba)`、reciprocity 残差（帧对声速与序列聚合声速之差、以及 `t'_ab+t'_ba` 对多 L 拟合的残差），SNR 加权、accepted 过滤。
4. **序列特征**：E1d 结论直接迁移——compact 集合 = 逐方向校准栈（corrected TOF、TOF-L、estimated c）+ PSR + `ultrasonic_snr_db` + 新增 pair 物理块（`ĉ_seq`、`v̂_seq`、`τ̂_ab/τ̂_ba`、reciprocity 残差统计、逐 L ĉ 离散度）。禁止把完整双倍 RawDSP 堆栈包装为端到端改进；维度控制以 E1d-SB 纪律为准。

### 5. 阶段与验收门

| 阶段 | 内容 | 通过门 | 失败动作 |
| --- | --- | --- | --- |
| **F0** 契约与参数重登记 | flow registry、jitter 双情景推导、拓扑/时序/不对称决策、schema 草案评审 | registry 完整且每项有来源类型；jitter nominal 情景有 datasheet 依据 | 缺依据 → `inconclusive_parameter_bounds`，不进 F1 |
| **F1** 仿真物理与单元测试 | flow 传播、双向波形生成、schema/packaging/manifest | 零噪零 jitter 网格上 ĉ 恢复 ≤0.01 m/s、v̂ ≤0.01 m/s；`v_path=0` 时 AB 帧 TOF 字段与现单向生成器逐点一致；全 pytest 通过 | 修物理，不生成数据 |
| **F2** smoke benchmark | `tv3-bidir-smoke` 生成 + 校验 + 存储审计 | validation 全过；int16+scale 往返误差与现契约同档；体积符合估计 | 修 packaging |
| **F3** DSP 估计器与保真 | 帧级 fidelity + 延迟自标定 + 物理量恢复 | 逐方向 peak P95 ≤0.25 sample（train-calibrated 模板）；`|τ̂−τ_true| ≤0.10 μs`；全 flow 网格 ĉ 偏置 ≤0.05 m/s（≈0.2 vol% 等效）；v̂ 偏置 ≤0.05 m/s；reciprocity 残差 P95 ≤0.10 μs | 先修 DSP，不训练模型（停止条件 3 同款） |
| **F4** 可辨识性 v2 审计 | 观测集 = 双向 TOF（多 L 多帧）+ 登记 T/NDIR/TCS 协方差；flow 转 `implemented_physics`；同门限 P90/nuisance/拒绝率；jitter 双情景并行 | 联合 Fisher 非秩亏；产出各情景窄窗 P90 与主导项排序；与 §Context.2 先验表交叉核对（偏离超量级须解释） | 按 verdict 分流（见 §7），不训练模型 |
| **F5** 正式数据与模型对照 | `tv3-bidir-6000` + 五臂对照 + S-Flow selector | 见下表 | 记录失败臂，不扩大结构 |
| **F6** verdict 与回填 | 汇总 F4/F5，更新记忆库/统一路线/导读 | 记忆库结论表新增 F 行并保持 v1 行不变 | — |

**F5 五臂对照**（全部使用冻结 B1 Ridge 与 B7 配方重训，不改超参；S-Flow：train `|v_path|≤2.5`，OOD `|v_path|∈(2.5,4]`；同时报告 S-Y/S-L 与 `v_path=0` 锚点子集）：

| 臂 | 输入 | 回答的问题 |
| --- | --- | --- |
| A1 单向 | 仅 AB 方向特征 | flow 污染下单向基线退化多少（相对静止锚点） |
| A2 单向 + oracle v 校正 | AB + true `v_path`（仅审计臂，不可部署） | 完美流速信息的收益上限 |
| A3 双向解耦 | pair 物理块（ĉ、v̂、reciprocity）+ 逐方向 compact 栈 | 部署级解耦能否逼近 A2 |
| A4 双向全特征 | A3 + 全部逐方向统计 | 增量维度是否还有信息 |
| A5 双向去 flow 列 | A3 移除 v̂ 及其派生 | 增益来自解耦后的 ĉ 还是 v̂ 本身 |

F5 预注册判据：a) A3 在 S-Flow OOD 上的 O₂ MAE 显著优于 A1（幅度门在 F4 后依误差预算预注册，先验预期 ≥0.5 vol% 级）；b) A3 与 A2 差距 ≤ 预注册残余（解耦充分性）；c) A3 的 `v_path=0` 锚点子集相对 S 线/现有 B1 非劣（Δ ≤0.05）；d) S-Y/S-L 非劣（ΔR² ≥ −0.01）；e) corrected sound speed bias 相对 A1 下降且 reciprocity 可稳定标定（沿用统一路线 0.10 μs 初始目标）。任一增益仅出现在 val 不同步到 test/OOD → 判未通过。

#### F5-S：bidir secondary selectors（判据 d 实施契约）

**目标**：为 F5 判据 (d) 生成与双向数据契约相容的 S-Y/S-L 派生集，并在相同派生 split 内比较 A3 与 A1。F5-S 只改变数据划分与重建顺序，不改五臂特征、B1/B7 超参或 F4 已冻结阈值。窄域与 wide 域复用同一实现，但 source、bootstrap、派生目录和结果必须按 `composition_domain` 隔离；F5-S 未通过前，F5 verdict 只能是 `f5_model_protocol_incomplete`。

**1. SPXY observed X 冻结**

正式 profile 命名为 `bidir_spxy_observed_ab_v1`，summary 写 `x_feature_profile=bidir_spxy_observed_ab_stats_v1`。X 只使用部署可得的 7 slow 通道和 **AB 单向 RawDSP**，保持 selector 对 A1/A3 中立，不能预先利用只有双向臂才有的解耦信息。

| X 块 | 数组 | 统计 | 维数 |
| --- | --- | --- | ---: |
| slow | 7 个 `SLOW_CHANNELS` | mean/std/min/max/trend | 35 |
| AB corrected TOF | `ultrasonic_tof_corrected_ab_raw_dsp_s` | mean/std/trend | 3 |
| AB peak | `ultrasonic_peak_index_ab_raw_dsp` | mean/std | 2 |
| AB sound speed proxy | `ultrasonic_sound_speed_ab_raw_dsp_m_per_s` | mean/std | 2 |
| AB SNR | `ultrasonic_snr_db_ab` | mean/std | 2 |
| AB PSR | `ultrasonic_psr_ab` | mean/std | 2 |
| AB quality | `ultrasonic_quality_ab_raw_dsp` | mean/std | 2 |
| AB accepted | `ultrasonic_accepted_ab_raw_dsp` | mean/std | 2 |

总维数固定为 **50**，逐列 `StandardScaler` 后进入 SPXY 距离。以下字段禁止进入 X：任何 BA 数组、pair ĉ、v̂、reciprocity、`v_path_m_per_s`、true/oracle TOF、true sound speed、true alpha、组分标签及 simulator 内部量。标签只允许作为 SPXY 的标准化 Y 和 `y_margin_ood` selector 输入，不得拼入 X。缺列、非有限值、维数/feature-name digest 漂移直接失败，不回退到 `oracle_v1` 或旧单向 `observed_v1`。

**2. bootstrap 与 train-only 重建纪律**

1. 先在 source benchmark 的冻结 base train split 上构建一次 `raw_dsp_bidirectional_v1` cache，角色登记为 `split_selection_bootstrap_only`；它只为 SPXY X 提供 AB 数组。
2. 用 bootstrap X 派生 S-Y/S-L 后，派生目录只链接物理数组、slow、labels、metadata 与 condition grid；禁止链接 source 的 `features/`。
3. 每个派生 split 必须以其自身 `train.csv` 重新标定 AB/BA template 与 session delay，再重建 bidir frame cache 和 A1–A5 arm cache。模型训练只能读取这份 split-specific cache。
4. bootstrap manifest、source manifest、split hash 或 template source sequence-id digest 任一不匹配即停止；不得复用 random/S-Flow train 标定结果伪装为 selector-specific cache。

**3. selector 与正式矩阵**

| ID | SPXY | 独立 OOD selector | 作用 |
| --- | --- | --- | --- |
| S-Y | `alpha=0.5` + `bidir_spxy_observed_ab_v1` | `y_margin_ood` | 组分边界 OOD |
| S-L | `alpha=0.5` + `bidir_spxy_observed_ab_v1` | `lhs_boundary` | LHS 几何边界 OOD |

- split seeds 冻结为 `20260704 / 20260712 / 20260720`；保持 `mixture_id` 分组与 train/val/test/extrapolation=`70/15/10/5`。
- 正式矩阵为 `2 selectors × 3 split seeds × 5 arms × 2 heads`。每个派生 split 都训练冻结 B1 Ridge 与 B7 residual；不得只跑表现较好的 selector、seed 或臂。
- 建议目录：`${splits_root}/s_y/spxy_ab_a05_ymargin_s<seed>/` 与 `${splits_root}/s_l/spxy_ab_a05_lhsboundary_s<seed>/`；wide 由独立 `${splits_root}` 保持 `-wide` 命名空间，不与窄域共享 cache。
- 每个 split summary 必须写 source/condition/label/bootstrap hash、`x_feature_names` 与 digest、X matrix hash、split hash、`ood_set_hash`、集合互斥/总数守恒、各 split 组分与环境范围；selector 退化或不同 seed OOD 集相同必须显式报告，不能静默换 selector。

**4. 判据 (d) 的唯一口径**

主门只用冻结主头 `b1_ridge`，B7 结果完整报告但不新增通过条件。对每个 selector、split seed 和 `test/extrapolation` 分别计算：

```text
delta_r2_o2 = R2_O2(A3, same selector/seed/split) - R2_O2(A1, same selector/seed/split)
```

`selector_gate_d.passed=true` 当且仅当：矩阵与 provenance 审计完整，且所有 12 个配对值（2 selectors × 3 seeds × 2 splits）均满足 `delta_r2_o2 >= -0.01`。禁止把不同 selector、不同 seed 或 S-Flow 的绝对 R²互减；禁止用均值掩盖单个失败格。缺结果/审计失败 → `f5_model_protocol_incomplete`（CLI exit 2）；矩阵完整但任一格低于门 → `f5_model_protocol_failed`（CLI exit 1）；判据 a–e 全通过才允许 `f5_model_protocol_passed`（CLI exit 0）并进入 F6。

**5. 实现落点与验证**

| 文件 | 动作 |
| --- | --- |
| `tv3/sim/packaging/spxy_split.py` | 注册 `bidir_spxy_observed_ab_v1` 及 50 维字段契约；复用标准化 SPXY 与现有 OOD selector，不复制算法 |
| `scripts/recompute_tv3_split.py` | 支持 bidir AB bootstrap adapter、`-wide` provenance 和派生目录不链接 feature cache |
| `scripts/run_tv3_bidir_model_protocol.py` | 新增 F5-S derive/audit/rebuild/train/report 编排；以真实 12 格配对结果替换 `blocked_unimplemented` |
| `configs/tv3_bidir_model_protocol*.json` | 冻结 profile、selectors、split seeds、目录和 `selector_r2_noninferior_delta=-0.01` |
| `tests/test_tunnel_ventilation_bidir_secondary_selectors.py` | 覆盖 50 维字段、oracle/BA/pair 拒绝、分组互斥、hash 绑定、split-specific cache 重建、12 格门与退出码 |

最小执行顺序：smoke 上完成 profile 与两 selector 派生审计 → 生成正式 6000 benchmark → 构建 source bootstrap → 派生 6 个正式 split → 各 split 重建 bidir cache → 跑完整五臂双头矩阵 → 汇总判据 (d)。任何派生 split 未通过审计时禁止开始该 split 的模型训练。

**实施状态（2026-07-22）**：上表代码路径已落地（profile / recompute / 协议编排 / 配置 / 单元测试）。`derive_secondary_selectors=true`。正式数值矩阵仍须先 smoke 端到端，再服务器 6000；incomplete → CLI exit 2，矩阵完整但任一格失败 → exit 1。

### 6. 实施范围

| 文件 | 动作 | 约束 |
| --- | --- | --- |
| `tv3/sim/core/tunnel_ventilation_bidir_schema.py` | 新增；复用基 schema 常量，定义双向数组名与 condition 字段 | 不改基 schema |
| `tv3/sim/generation/tunnel_ventilation/flow_physics.py` | 新增；v_path 采样、双向有效声速、registry 描述 | 常数进 config，不散落 |
| `tv3/sim/generation/waveforms.py` | 新增 `simulate_bidirectional_waveform_measurement`（组合两次现核心单发射） | 不改现函数语义；现测试不回归 |
| `tv3/sim/generation/tunnel_ventilation/{conditions,benchmark,slow}.py` | 扩展 bidir 生成路径 | flow 真值只进 condition grid/oracle 数组 |
| `tv3/sim/packaging/arrays.py` | 新增 `_ab/_ba` 数组写读 | schema 驱动，旧数据不受影响 |
| `tv3/pipeline/generate_tunnel_ventilation_benchmark.py` | 新增 `--bidirectional` 入口与 manifest 字段 | 默认关闭 |
| `tv3/ml/bidir_features.py` | 新增 `raw_dsp_bidirectional_v1` 估计器 + builder | train-only 标定，digest 固化 |
| `tv3/audit/identifiability*.py` | 扩展多观测 Fisher（v2），复用 S2 机制 | v1 代码路径与产物不动 |
| `scripts/run_tv3_bidir_*.py`、`configs/tv3_bidir_*.json` | 新增各阶段入口 | 输出不可覆盖 |
| `tests/test_tunnel_ventilation_bidir_*.py` | 新增：解析恢复、零流回归一致、延迟不对称传播、jitter 相关性模式、schema/packaging 往返、估计器门 | 纯小型数值 fixture |

正式产物目录：

```text
outputs/tv3_bidir/
  benchmark_audit/            # F2 存储与校验
  dsp_fidelity/               # F3 帧保真 + 延迟标定 + 物理量恢复
  identifiability_v2/         # F4 审计（独立于 outputs/tv3_identifiability/）
  model_protocol/             # F5 五臂 × selector 矩阵
  verdict.json                # F6
```

### 7. verdict 分流（沿用 v1 词表，范围限定"已登记仿真分布内"）

| verdict | 条件 | 后续 |
| --- | --- | --- |
| `continuous_regression_supported`（bidir 范围） | F4 三门通过（至少 nominal jitter 情景）且 F5 判据 a–e 通过 | 进入湿空气/设备 `acoustic_measurement_v2` 与 UQ；仍不外推现场能力 |
| `coarse_monitoring_only` | flow 解耦成功但窄窗 P90 仍超门（预期主导：T 或保守 jitter） | 组分输出保持分档/趋势；v̂ 流速输出可独立保留价值 |
| `information_source_upgrade_required` | 解耦后仍有阻断性缺口或全域信息不足 | 维持 v1 判断，升级方向转多频/TDLAS |
| `inconclusive_parameter_bounds` | F0 依据不足 | 不给继续/停止结论 |
| `audit_failed` / `estimator_failed` | hash、数值、闭包或 F3 门失败 | 修链路，不训练模型 |

---

## Format

### 1. 执行顺序与前置条件

1. **前置门**：静止空气 S 线的 S2 审计机制落地后，F4 才可执行（复用实现）；F0/F1/F2（契约、物理、smoke）不依赖 S 线，可在用户确认启动后并行推进，但不得挤占 S 线正式排期。
2. F0 评审通过 → F1 → F2 → F3；F3 未过帧保真门前禁止任何模型训练（与 D2b 停止条件同构）。
3. F4 与 F5 数据生成可并行准备，但 F5 判据的幅度门必须在 F4 产出后预注册，防止事后选择。
4. 每阶段产物一次写入，不可覆盖；复现 run 写独立目录。

### 2. 最小验证

```bash
# F1
python -m pytest -q tests/test_tunnel_ventilation_bidir_physics.py
# F2
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark --bidirectional --preset bidir-smoke
python scripts/check_slow_channels.py data/tv3-bidir-smoke
# F3
python scripts/run_tv3_bidir_dsp_fidelity.py --config configs/tv3_bidir_dsp_fidelity.json
# F4
python scripts/run_tv3_identifiability_v2.py --config configs/tv3_bidir_identifiability_v2.json
# F5 (server; formal 6000)
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark --output-root data --preset bidir-formal-6000
python scripts/run_tv3_bidir_model_protocol.py --config configs/tv3_bidir_model_protocol.json --stage all --device cuda
python -m pytest -q tests/test_tunnel_ventilation_bidir_model_protocol.py
```

测试至少证明：reciprocal-sum 对均匀流精确、`−Var(v)/c` 二阶残余符号与量级正确、零流退化一致、延迟不对称只进 v̂、独立/共模 jitter 两种相关性统计行为区分、oracle 字段不可进入部署输入路径（校验拒绝）。

### 3. 文档回填与联更义务

- F0 冻结、每阶段正式 verdict 产生时更新本文件实施记录；只有 F4/F5/F6 正式产物可进入项目记忆库结论表。
- 按记忆库 §1.4：本计划涉及的新名词（F 线、`raw_dsp_bidirectional_v1`、reciprocity error、v_path、ping-pong、reciprocal-sum、identifiability v2）已随立项写入 `methods/tv3_名词与实验顺序导读.md`；后续阶段名或 builder 变更须同批次联更。
- v1 目录 `outputs/tv3_identifiability/` 永不改写；F4 结论以"v2（双向情景）"身份并列报告。

---

## 备选设计与明确不做（记录取舍，防止重复讨论）

| 备选 | 取舍 | 状态 |
| --- | --- | --- |
| 正交双径（横向路径测 c、轴向测 v） | 免解耦数学，但需 4 换能器与新几何契约；对射双向 2 换能器即可达成同等一阶效果 | 不实施，保留记录 |
| 同步交叉发射（共模 jitter 消除） | Δt 精度大幅提升（v̂ 噪声降约两个量级），但需收发隔离/编码分离，硬件复杂度高 | 以 `shared_trigger` 情景开关保留在 F4 审计中，不进基线契约 |
| 单窗双脉冲复用（两发射共用一条 5 ms 窗） | 省一半存储，但混合两个接收通道的物理身份，破坏逐方向模板与质量标记 | 不实施 |
| sing-around / ring-around 高精度 c | 提升 c 分辨率，但改变帧结构契约且不解决 T–M 混叠 | 不实施 |
| 序列内时变湍流 AR(1) | 更真实，但 v1 版先以恒定流 + `not_represented` 登记；`Var(v)` 先验证明其非阻断 | 后续情景扩展 |
| 多频色散（CO₂ 弛豫）通道 | 独立信息维度，属于另一条信息源升级线 | 独立立项，不混入 F 线 |

---

## 实施记录

| 日期 | 阶段 | verdict | 产物 |
| --- | --- | --- | --- |
| 2026-07-21 | F0 | `f0_registry_frozen` | `configs/tv3_bidir/parameter_registry.json`（冻结 sha256 `dc61d9e7…`）；进度外置于 `stage_status.json`；oracle 含 `ultrasonic_alpha_true_npm`；`outputs/tv3_bidir/f0_registry/` 标注 supersedes 旧哈希 `67752835…`；测试 `tests/test_tunnel_ventilation_bidir_f0.py` |
| 2026-07-21 | F1 | `f1_physics_passed` | `flow_physics.py`；`simulate_bidirectional_waveform_measurement`；`slow.build_sequence_arrays(bidirectional=True)`；`write_bidirectional_arrays`；`tests/test_tunnel_ventilation_bidir_physics.py` |
| 2026-07-21 | F2 | `f2_smoke_passed` | `data/tv3-bidir-smoke`（16×32，AB/BA int16 + alpha oracle）；F2 审计校验 registry sha256 ≡ F0；`int16_storage_self_consistency`；体积 ~9.9 MB；零锚点 12.5% |
| 2026-07-21 | pre-F3 | provenance 修复 | registry 去可变状态；`sample_v_path` 改为 1D LHS；alpha 落盘；`MAX_PAIR_INTERVAL_S`；workers=2 小规模验证通过 |
| 2026-07-21 | F3 | `f3_dsp_passed` | `raw_dsp_bidirectional_v1`；`data/tv3-bidir-f3`（零 jitter）；`outputs/tv3_bidir/dsp_fidelity/`；peak P95 AB/BA 0.087/0.046；τ̂ 误差 45 ns；steady 网格 ĉ 偏置 0.004 m/s、v̂ 偏置 −0.011 m/s；reciprocity P95 91 ns；测试 `tests/test_tunnel_ventilation_bidir_dsp.py` |
| 2026-07-21 | F4 | `coarse_monitoring_only`（stage pass） | `outputs/tv3_bidir/identifiability_v2/`；flow=`implemented_physics`，拒绝率 0；joint rank≥2（+T 时 3）；窄窗 P90 max：nominal 4.37 / conservative 9.58 vol% O₂；主导项：nominal=T(1K)，conservative=jitter(3μs)；先验交叉核对通过；F5 幅度门已预注册；v1 目录未改写 |
| 2026-07-21 | F5 code | `code_ready_awaiting_server_formal_6000` | preset `bidir-formal-6000`；`tv3/ml/bidir_s_flow.py`；`build_tv3_bidir_features.py`；`bidir_arm_features.py`（A1–A5）；`scripts/run_tv3_bidir_model_protocol.py`；`configs/tv3_bidir_model_protocol.json`；测试 `tests/test_tunnel_ventilation_bidir_model_protocol.py`；正式 6000 训练待服务器 |
| 2026-07-22 | wide 域立项 | `scope_confirmed_a1_flow_only_b1_parallel` | 组分宽域 CO₂ 0.03–10%/O₂ 15–25% 独立注册域；A1 仅 F 线 + B1 并行留档已确认；改动契约 `docs/active/tv3_composition_range_widening_plan.md`；窄域 F0–F5 冻结不动 |
| 2026-07-22 | F0'-wide / F1-wide | `f0_wide_registry_frozen` + `f1_wide_physics_passed` | `WIDE_COMPOSITION_RANGES`；spec/`--composition-domain` 线穿；`parameter_registry_wide.json`（独立 sha256，窄域 dc61d9e7… 未动）；`tests/test_tunnel_ventilation_wide_composition.py` |
| 2026-07-22 | F2-wide | `f2_wide_smoke_passed` | `data/tv3-bidir-smoke-wide`（16×32）；`outputs/tv3_bidir/f0_registry_wide/` + `benchmark_audit_wide/`；int16 自洽；零锚点 12.5%；CO₂ max≈9.87 / O₂∈[15.47,24.86]；窄域 `stage_status.f2`/`allowed_next_stage` 未改写 |
| 2026-07-22 | F3-wide | `f3_wide_dsp_passed` | `data/tv3-bidir-f3-wide`（32×32，零 jitter）；`outputs/tv3_bidir/dsp_fidelity_wide/`；peak P95 AB/BA≈0.087/0.046；stress(CO₂≥8%,L≥0.28) 30 帧 max≈0.090/0.047；τ≈53 ns；ĉ bias 0.008、v̂ bias −0.010；reciprocity P95 97 ns；窄域 `dsp_fidelity/` 未改写 |
| 2026-07-22 | F4-wide | `coarse_monitoring_only`（stage pass） | `outputs/tv3_bidir/identifiability_v2_wide/`；六窗危害锚定；拒绝率 0；窄窗 P90 max 名义≈4.50 / 保守≈9.74 vol% O₂；先验交叉核对通过；F5-wide 幅度门已预注册（判据 c=`in_domain_a1_v_path_zero`）；窄域 F4/`identifiability_v2/` 与 v1 未改写 |
| 2026-07-22 | F5-S code | `f5s_code_ready_awaiting_formal_matrix` | `bidir_spxy_observed_ab_v1`（50 维 AB-only）；`recompute_tv3_split` bidir adapter；`tv3/ml/bidir_f5_secondary.py`；协议 `derive_secondary_selectors=true` + 12 格判据 (d)；窄/宽 `configs/tv3_bidir_model_protocol*.json`；测试 `tests/test_tunnel_ventilation_bidir_secondary_selectors.py`；下一步 smoke 端到端 → 服务器 6000 矩阵 |

F0 要点：

- flow：`v_path ∈ [-4,4]` + ≥10% 零锚点；湍流时变等显式 `not_represented`（`−Var(v)/c` 先验非阻断）。
- jitter 双情景：`conservative_v1` = 3 μs（`engineering_scenario`）；`nominal_daq_half_sample` = 0.5 μs（`literature_bound`，由 NI USB-6453 1 MS/s 半采样上界推导；datasheet 无 RMS jitter 声明）。
- 拓扑：ping-pong，`pair_interval_s=2.5 ms`；`delay_asymmetry_s` 默认 0。
- schema 草案：`tunnel-ventilation-bidir-1`；flow 不进 slow；oracle 含 `ultrasonic_alpha_true_npm`。
- **冻结语义**：`parameter_registry.json` 只含参数证据；阶段进度在 `configs/tv3_bidir/stage_status.json` 与 `outputs/tv3_bidir/f*_verdict.json`；F0 gate 的 `allowed_next_stage_on_pass` 固定为 `F1_physics_unit_tests`。

F1 要点：

- 均匀流 reciprocal-sum 解析精确；`−Var(v)/c` 二阶残余符号与量级正确。
- 零噪零 jitter 网格 ĉ/v̂ 恢复 ≤0.01 m/s；`v_path=0` 时 `path_velocity=0` 单发射与现单向生成器逐点一致。
- 延迟不对称在方向校正后不污染 ĉ；共模 τ 校正时 v̂ 偏置符合 `−c²δτ/(2L)`。
- `independent` / `shared_trigger` jitter 相关性可区分。
- `sample_v_path_m_per_s` 为 1D LHS + 零锚点（对齐 registry `constrained_lhs_with_zero_anchor`）。

F2 要点：

- `--bidirectional` / `--preset bidir-smoke` 写入 `tunnel-ventilation-bidir-1`；flow 进 condition grid，不进 slow。
- validation pass；无单向 `ultrasonic*.npy`；AB/BA int16+scale **存储自洽**（非 float 量化保真）；alpha oracle 落盘。
- F2+ 审计要求当前 registry sha256 与 F0 verdict 一致。
- 允许进入 F3（DSP 估计器与保真）；尚未训练模型。
- workers=2 小规模（4×4）已验证并行合并路径。

F3 要点：

- 冻结 builder：`raw_dsp_bidirectional_v1`（train-only 逐方向 median 模板 + session 级 mid-pair shared τ̂ + reciprocal-sum）。
- F3 数据集 `tv3-bidir-f3` 使用零 trigger jitter；名义/保守 jitter 留给 F4 双情景。
- 物理量恢复门在 **steady + accepted** 多 L 帧上聚合（SNR 加权 ĉ/v̂ vs oracle steady median）。全序列 `ĉ_seq` 混入 baseline/exposure/recovery 组分变化，不作 F3 偏置门。
- 首轮失败曾把全相位聚合误判为 ĉ 偏置 0.18 m/s； steady 网格真实偏置 0.004 m/s。
- `delay_asymmetry_s=0` 时用 shared mid-pair τ，避免 τ̂_ab−τ̂_ba 噪声污染 v̂。
- 允许进入 F4（identifiability v2）；仍禁止模型训练。

F4 要点：

- 复用 v1 灵敏度 / 等效 O₂ / P90 / verdict 词表；新模块 `identifiability_v2.py`，**不**改写 `outputs/tv3_identifiability/`。
- 观测：AB/BA TOF + 登记 T（Fisher 附加行）；误差预算用 mid-pair TOF（独立 jitter ⇒ σ/√2）。
- `flow_projection` → `implemented_physics`，`blocks_go_verdict=false`；拒绝率 0（v1 为 189/189）。
- 双 jitter 并行：`conservative_v1`（3 μs）与 `nominal_daq_half_sample`（0.5 μs）。
- 业务三门：拒绝率通过；P90 与 nuisance 比例两门在双情景均失败（1 K 温度情景下名义臂窄窗 max P90≈4.37 vol%；保守臂≈9.58）。
- 主导项：名义臂为温度 1 K；保守臂为单帧 jitter。与 §Context.2 先验交叉核对通过（bidir mid-pair 使 jitter 等效约为单向表的 1/√2）。
- 业务 verdict=`coarse_monitoring_only`（flow 解耦成功，连续回归门未达）；阶段门通过，允许 F5。
- F5 幅度门已预注册：A3−A1 O₂ MAE ≥0.5 vol%；A3−A2 ≤0.25；零锚点 Δ≤0.05；selector ΔR²≥−0.01。

F5 要点（F5-S 代码已落地，正式矩阵待 smoke → 服务器）：

- 数据：`--preset bidir-formal-6000` → `data/tv3-bidir-6000`（6000×512，int16，skip-fiber-mic）。
- S-Flow：mixture 中位 `|v_path|≤2.5` 入域随机划分；`(2.5,4]` 进 extrapolation OOD；保持 mixture_id 分组。
- F5-S：`bidir_spxy_observed_ab_v1`（50 维 AB-only）派生 S-Y/S-L×3 seeds；12 格判据 (d) 配对 A3−A1 O₂ ΔR²≥−0.01；`derive_secondary_selectors=true`。
- 特征：train-calibrated `raw_dsp_bidirectional_v1` 帧缓存 + 五臂 rocket 矩阵（A2 为 oracle-v 审计臂）；各派生 split 独立重建 cache。
- 头：冻结 B1 RidgeCV alpha 网格与 B7 OOF residual MLP 超参；产物写入 `outputs/tv3_bidir/model_protocol/`，不覆盖 `outputs/tv3_d2b/`。
- 入口：`python scripts/run_tv3_bidir_model_protocol.py --config configs/tv3_bidir_model_protocol.json --stage all`；incomplete→exit 2，(d) 失败→exit 1。
