# A2-DYN 物理与审计规格（从属于 13）

> 文档状态：从属于 [13_Ar-He-CO2动态时间序列仿真与数据分布规划](13_Ar-He-CO2动态时间序列仿真与数据分布规划.md)，阶段状态由主规划统一持有 \
> 承接内容：§13 的 §4 动态物理链、§6 动力学与扰动分布、§8 数据契约与存储、§10 质量、物理与动态真实性审计、§13 计划影响文件与职责、§15 验证矩阵 \
> 迁移日期：2026-09-04（D5 结构拆分，纯搬运逐字保留；节号沿用 §13，跨文件引用写作「13b §x」）\
> 拆分验收：三份文件合并正文与拆分前逐字一致

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

1. 非纯气序列中，至少 95% 有两路以上 peak-to-peak 大于各自噪声标准差的 5 倍；噪声标准差按该行注册 noise profile 的 `white_noise_scale` 缩放（NOISE-10X 行不得享受相对放宽 10 倍的名义门）。未缩放口径的历史值同时报告作对照。
2. 每路有效动态序列至少覆盖 10 个不同量化级。
3. 至少 70% 的名义序列存在两路低频观测 `t50` 相差两个以上外层采样点；超声内部传播时间不参与该统计。
4. baseline、transition、steady、recovery 均有非空点；任一阶段为空判该行退化。
5. clean signal 的 transition 方差不能全部由白噪声解释：逐通道计算 transition 段方差与该行白噪方差之比，任一通道比值 ≥4（`minimum_transition_variance_ratio`）即通过该行——binary 无 CO₂ 等组成使单通道恒定属合法动态，不判退化；通过比例的门限为 95%（`minimum_transition_variance_ratio_pass_fraction`）。
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
3. rank fraction 与 condition number P50、P95、max，覆盖 train / val / stress_val 三个 split、每 family 每 split 12 个确定性等距采样行（合计 216 个样本）；按 family × split × horizon 组织，逐 horizon（P015 / P030 / P060 单独）与堆叠矩阵两种口径分别报告；这是**采样结论而非全量结论**，产物中 `sampled_rows / total_rows` 显式声明覆盖规模；
4. pure 和结构零边界单列：pure 顶点的边界审计由 A2-DYN-4 的 pure 边界审计承担，Jacobian 层不覆盖 pure；
5. 特权 kinetics oracle 与可部署基线的 headroom（O-KIN-OBS，见 §11.1）。

默认资格：

- 目标组成 Jacobian full-rank fraction ≥0.99；
- P95 condition number <1000；
- oracle 预测有限且组成合法；
- 至少两个早期 horizon 留有可部署学习器可利用的 headroom。

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
| 数据审计 | `src/gf/sim/a2_dynamic_audit/`（包：`__init__` 编排 + `_schema` / `_physics` / `_dynamic` / `_baselines` / `_heos_interpolation` / `_jacobian` / `_freeze` / `_shared`） | 守恒、EOS、设备状态、动态、边界、Jacobian、oracle；2026-09-04 拆分子模块，公开签名不变 |
| 数据适配 | `src/gf/dl/adapters/ar_he_co2.py` | 将动态包映射到 UnifiedSample |
| 基线 | `src/gf/dl/temporal_baselines.py` | B-LAST、B-DELTA、B-EWMA、B-STAT、B-TCN、B-GRU、B-STEADY 与 B-REF 选择 |
| 编排 | `src/gf/pipeline/a2_dynamic_benchmark.py` | generate、audit、baseline、replay、report |
| 测试 | `tests/test_a2_dynamic_*.py` | 解析输运、设备采集、schema、因果前缀、最小 smoke |
| 数据 | `data/a2_dynamic_v1/` | 聚合动态数据包 |
| 运行 | `outputs/runs/a2_dynamic_v1/` | 单次运行 manifest、预测和 checkpoint |
| 汇总 | `outputs/summary/a2_dynamic_v1/` | 指标、比较、合格轴和 hash |
| 报告 | `outputs/reports/a2_dynamic_v1/` | 数据与时间信息资格报告 |

不得在 pipeline 中再实现一套 profile 抽样或指标公式。配置是范围事实源，sim 是生成事实源，dl 是模型事实源，pipeline 只编排。

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

