# 掘进通风（tv3）项目改进方案

> 文档用途：基于当前项目记忆库、正式实验结论和补充资料检索，形成下一阶段统一执行方案。  
> 适用范围：`tv3` 掘进工作面 CO₂/O₂/N₂ 空气质量监测项目。  
> 当前基线日期：2026-07-12。  
> 正式数据基线：clean `tv3-formal-6000`。  
> 当前默认 RawDSP 预测头：B7 OOF Ridge + residual MLP。  
> 本文不修改现有正式数据、特征和 `raw3` 输出契约；任何新物理、新特征或新协议必须独立版本化。

---

## 1. 执行摘要

当前项目已经完成从原始超声波形到稳定 RawDSP 特征、线性基线和轻量非线性残差头的主要算法闭环：

- D2b RawDSP 已通过 waveform fidelity 和 D0-observed parity；
- B7 已通过 repeated split 和双 OOD selector 协议；
- B7 是当前默认 RawDSP 头；
- C1 physical grouped bottleneck 已经由完整正式矩阵证伪；
- 单纯扩大 residual MLP、继续早期物理分组或恢复旧 D2 补丁路线，优先级应降为零；
- O₂ 在整体 18.0%–21.2% 范围存在可辨识信号，但在 0.8% 窄窗口内不具备可靠精细分辨能力；
- S-Y 与 S-L 的绝对 OOD 结果差异明显，说明模型能力强烈依赖分布偏移机制；
- 当前最大缺口不再是回归头容量，而是：
  1. 声学可辨识性与误差预算；
  2. 气流、温度、湿度、声程、延迟、SNR 等环境扰动；
  3. 仿真到真实硬件的域差异；
  4. 部署时的预测区间、拒绝预测和风险控制。

因此，下一阶段应从“继续优化模型”转为“验证测量系统是否成立”。

总体优先顺序：

```text
P0  双向声学与气流解耦
P0  可辨识性、灵敏度和误差预算
P0  湿空气与测量系统数字孪生
P0  环境/设备/时间 OOD 协议
P0  最小真实硬件闭环

P1  B7 不确定性与拒绝预测
P1  独立 O₂ residual、小模型和参数匹配 flat TabM
P1  强表格模型上限探针

P2  多频声学信息量验证
P2  基于少量实测数据的仿真参数校准

P3  大规模真实无标签数据到位后，再启动自监督和端到端时序模型

Fallback  声学能力不足时恢复 O₂ 专用 TDLAS 路线
```

---

## 2. 冻结事实与硬约束

### 2.1 数据和输出契约

1. 正式标签为：

```text
x_CO2 + x_O2 + x_N2 = 100%
```

2. 正式模型直接输出：

```text
raw3
out_dim = 3
```

3. 不采用：

- `gas_head`
- N₂ 闭包回填
- ILR/ALR
- `target_transform`
- `free_component_mse`
- 闭包残差头

4. `sum_abs_error` 只作为监控指标，不作为强制输出约束。
5. 仿真真值 TOF、真实声速和 true alpha 只能用于 oracle 审计或 auxiliary target，不能进入部署输入。
6. 现有 observed builder 继续冻结为：

```text
d0_observed_physics_stats_v1
```

7. 新增湿空气、双向声学、多频或流速补偿特征时，必须创建新 builder 和新 manifest。
8. 缺失 manifest、waveform scale、slow metadata 或必需数组时直接失败，不允许静默 fallback。
9. 正式比较只使用 clean `tv3-formal-6000`。
10. 旧本地 600 数据只能用于代码、DSP 和 waveform fidelity 调试，不能用于正式性能声明。

### 2.2 当前模型结论

| 路线 | 当前结论 | 后续处理 |
|---|---|---|
| D0-observed Ridge | 测量级线性 baseline | 保留 |
| R5-T observed MLP | 目标标准化后稳定有效 | 保留为 observed 对照 |
| D2 TOF-PhaseNet | 原实现失败 | 不再补丁式扩展 |
| D2b RawDSP Ridge | 已通过 parity | 保留 |
| B6 RawDSP MLP | 已稳定通过 | 保留历史锚点 |
| B7 OOF Ridge residual | 当前默认 RawDSP 头 | 冻结 |
| C1 physical grouped bottleneck | 完整矩阵失败 | 停止 |
| ExtraTrees | 泛化不足 | 不继续调参 |
| TabPFN observed | 非部署上限探针 | 只作上限参考 |
| TDLAS O₂ | 硬件暂停 | 保留 fallback |

### 2.3 能力边界

固定采用以下结论：

> O₂ 可分辨整体高低档位，但窄浓度窗口内精细辨识不足。

不得将：

- S-L OOD `≈0.70`
- 某一随机划分的 test R²
- simulator-derived observed 特征结果
- 单一设备或单一环境结果

外推为所有真实巷道工况下的 O₂ 精确测量能力。

---

## 3. 关键问题重定义

下一阶段不再围绕“哪个模型 R² 更高”组织，而围绕以下五个问题组织：

### Q1：物理上能否达到目标精度？

需要回答：

- O₂ 变化引起的 TOF、相位、吸收和色散变化有多大？
- 温度、湿度、气流、声程、系统延迟和 SNR 的影响有多大？
- 目标业务精度是否低于系统的可辨识极限？

### Q2：当前仿真是否过度理想化？

需要回答：

- 干空气近似是否低估湿度影响？
- 单向 TOF 是否混入沿声路气流？
- 换能器、TX/RX 延迟、频响和老化是否被充分建模？
- 当前 B7 是否利用了真实硬件中不存在的仿真规律？

### Q3：当前 OOD 验证是否接近部署变化？

需要从组分边界扩展到：

- 温度；
- 湿度；
- 声程；
- 气流；
- SNR；
- 设备；
- 日期；
- 校准批次；
- 配气批次；
- 多因素极端组合。

### Q4：预测何时可信？

需要输出：

- 点预测；
- 预测区间；
- OOD 距离；
- 质量标记；
- 拒绝预测状态。

### Q5：声学路线是否值得继续投入？

必须建立明确停止条件。若经过流速解耦、湿空气建模、真实硬件校准和多频验证后仍达不到目标，应恢复 O₂ 专用传感通道。

---

## 4. 总体技术路线

```text
原始双向/多频波形
        │
        ▼
版本化 RawDSP / quality / calibration 特征
        │
        ├── 物理校正：气流、湿空气、声程、延迟
        │
        ├── 质量诊断：SNR、reciprocity、多径、漂移
        │
        ▼
冻结 Ridge 主干
        │
        ▼
B7 residual 或有限替代头
        │
        ▼
不确定性校准与 OOD 检测
        │
        ▼
点预测 + 区间 + 质量等级 + 拒绝状态
```

原则：

1. 先证明测量链路，再优化模型。
2. 先做显式物理解耦，再让模型学习剩余非线性。
3. 新增物理变量时必须版本隔离。
4. OOD 必须按变化机制独立报告。
5. 模型替换必须同时改善 ID、双 selector OOD、环境 OOD 和可靠性。
6. 不以单一整体 R² 作为部署验收依据。

---

# 5. 工作包设计

## WP0：基线冻结与复现实验

### 目标

在任何新实验前锁定当前正式状态，保证所有后续增益可归因。

### 执行动作

1. 冻结以下代码和产物版本：
   - RawDSP builder；
   - B1 Ridge；
   - B6；
   - B7；
   - repeated split selectors；
   - OOD set hash；
   - train-only template；
   - target scaler；
   - feature scaler。
2. 输出统一运行 manifest：
   - git commit；
   - Python 和依赖版本；
   - 数据集 hash；
   - builder 名称；
   - split seed；
   - training seed；
   - selector；
   - model config；
   - calibration config。
3. 建立正式结果索引：
   - baseline；
   - new physics；
   - OOD；
   - hardware；
   - uncertainty；
   - model probes。

### 验收门

- B1、B6、B7 正式指标可复现；
- repeated protocol 行数、hash 和 verdict 一致；
- 不允许使用旧本地 600 数据生成正式结论。

### 产物

```text
outputs/tv3_baseline_freeze/
  manifest.json
  baseline_metrics.json
  split_hashes.json
  environment.json
```

---

## WP1：可辨识性、灵敏度和误差预算

### 目标

确定 O₂ 声学信号相对环境扰动和硬件误差的真实量级，形成系统级测量能力边界。

### 变量

目标变量：

```text
x_O2
x_CO2
x_N2
```

nuisance variables：

```text
T
P
RH
L
airflow_projection
system_delay
tx_delay
rx_delay
trigger_jitter
SNR
gain
phase
transducer_response
multipath
dust_attenuation
```

### 实验设计

对冻结仿真器和新物理仿真器分别进行局部数值扰动，计算：

```text
∂TOF / ∂x_O2
∂TOF / ∂T
∂TOF / ∂RH
∂TOF / ∂L
∂TOF / ∂flow
∂TOF / ∂delay
∂feature_j / ∂variable_k
```

进一步计算：

- Jacobian；
- 特征条件数；
- Fisher information；
- 近似 Cramér–Rao lower bound；
- nuisance-marginalized O₂ information；
- 每个工况下的最小可分辨 O₂ 变化；
- O₂ 信号与 nuisance 影响比；
- 误差传播到最终 O₂ 预测的贡献。

### 核心输出

1. 全 O₂ 范围对应的 TOF 变化；
2. 0.8% O₂ 窄窗口对应的 TOF 变化；
3. 1 K 温度误差的等效 O₂ 误差；
4. RH 误差的等效 O₂ 误差；
5. 0.5/1.0 m/s 沿程流速的等效 O₂ 误差；
6. 声程误差、固定延迟误差和 jitter 的等效 O₂ 误差；
7. 每个环境条件下的理论分辨率。

### 通过门

至少满足：

- 在目标业务环境范围内，理论 P90 O₂ 误差优于业务要求；
- 关键 nuisance 校正后的残余影响不超过窄窗口 O₂ 信号的 50%；
- O₂ 信息在联合 nuisance 条件下不接近奇异；
- 若理论上无法满足，则停止窄区间精确回归目标，改为分档或趋势监测。

### 失败后的处理

若理想条件下仍无法满足目标：

- 不再扩大模型；
- 固定输出能力为粗粒度分档；
- 进入多频或 TDLAS 决策。

### 产物

```text
outputs/tv3_identifiability/
  sensitivity_matrix.parquet
  fisher_information.json
  error_budget.json
  equivalent_o2_error.csv
  identifiability_report.md
```

---

## WP2：双向声学与气流解耦

### 目标

消除真实通风流速对单向 TOF 的一阶混叠。

### 物理模型

顺流和逆流传播：

```text
t_down = L / (c + v_path)
t_up   = L / (c - v_path)
```

解耦：

```text
c      = L/2 × (1/t_down + 1/t_up)
v_path = L/2 × (1/t_down - 1/t_up)
```

其中：

```text
v_path = v × cos(alpha)
```

### 新数据契约

建议新建：

```text
raw_dsp_bidirectional_v1
```

必需输入：

```text
waveform_ab
waveform_ba
L_m
T_C
P_MPa
H_RH
pair_id
calibration_session_id
```

输出特征：

```text
tof_ab
tof_ba
peak_ab
peak_ba
snr_ab
snr_ba
quality_ab
quality_ba
sound_speed_bidirectional
flow_projection
reciprocity_error
corrected_tof
```

### 关键诊断

- AB/BA 峰值差；
- corrected sound speed 一致性；
- reciprocity error；
- 两方向 SNR 差；
- 多径不对称；
- TX/RX 链路差异；
- flow residual。

### 对照矩阵

| 输入 | 头 | 目的 |
|---|---|---|
| 单向 RawDSP | Ridge/B7 | 当前基线 |
| 单向 + 实测风速校正 | Ridge/B7 | 外部流速补偿 |
| 双向解耦声速 | Ridge/B7 | 主方案 |
| 双向全部特征 | Ridge/B7 | 检查残余信息 |
| 双向但不提供 flow | Ridge/B7 | 检查显式解耦价值 |

### 通过门

- 风速变化条件下 corrected sound speed bias 显著低于单向；
- 在 flow holdout 上 O₂ MAE 改善；
- 对 S-Y、S-L 不产生明显退化；
- 设备间 reciprocity 指标可稳定校准；
- 校正后的残余 TOF 偏差达到项目定义的可接受阈值。

建议初始阈值：

```text
目标       ≤ 0.10 μs
警告区间   0.10–0.20 μs
拒绝预测   > 0.20 μs
```

最终阈值应以 WP1 误差预算为准。

### 产物

```text
outputs/tv3_bidirectional/
  frame_fidelity/
  flow_decoupling/
  model_comparison/
  reciprocity_audit/
  protocol_metrics.json
```

---

## WP3：湿空气与测量系统数字孪生

### 目标

将仿真从“理想组分波形生成器”升级为“可校准的测量系统数字孪生”。

### 3.1 湿空气传播

标签保持干基：

```text
x_CO2 + x_O2 + x_N2 = 1
```

根据 T、P、RH 计算水蒸气分压：

```text
x_H2O = p_vapor / P
```

传播介质转换为湿基：

```text
x_i_wet = (1 - x_H2O) × x_i_dry
```

湿基组成参与：

- 混合摩尔质量；
- Cp、Cv、γ；
- 密度；
- 声速；
- 频率相关吸收；
- 分子弛豫；
- 换能器负载。

最终监督目标仍为干基 `raw3`。

### 3.2 设备级参数

显式建模：

```text
transducer_id
transducer_pair_id
frequency_response
temperature_response
tx_delay
rx_delay
gain
phase
noise_psd
clock_drift
aging_state
installation_angle
path_offset
```

### 3.3 现场扰动

可选建模：

```text
airflow_profile
dust_attenuation
condensation
multipath
reflector_position
nonuniform_temperature
nonuniform_humidity
```

### 3.4 版本设计

新物理版本：

```text
acoustic_measurement_v2
```

新 manifest 至少包含：

```text
physics_version
humidity_model
flow_model
transducer_profile_id
noise_profile_id
delay_profile_id
multipath_profile_id
calibration_session_id
randomization_seed
```

### 四组正式对照

| 训练域 | 测试域 | 目的 |
|---|---|---|
| dry v1 | dry v1 | 当前基准 |
| humid v2 | humid v2 | 新物理内部性能 |
| dry v1 | humid v2 | 旧模型 Sim2Real 风险 |
| humid v2 | dry v1 | 检查对 RH 的过度依赖 |

增加设备对照：

| 训练设备 | 测试设备 |
|---|---|
| 同设备 | 同设备 |
| 多设备 | 已见设备 |
| 多设备 | 完全留出设备 |
| 仿真设备分布 | 实际设备 |

### 通过门

- 新物理能解释真实波形的主要统计量；
- `dry→humid` 的退化被 `humid→humid` 明显恢复；
- device holdout 相对单设备训练显著改善；
- 增益不依赖 oracle 真值；
- 新 builder 在 B1 parity 或对应新基线上完成闭环。

### 产物

```text
outputs/tv3_measurement_v2/
  simulator_validation/
  domain_gap/
  device_profiles/
  humidity_ablation/
  physics_manifest/
```

---

## WP4：正式环境与硬件 OOD 协议

### 目标

将 OOD 从组分边界扩展为真实部署变化机制。

### 4.1 保留现有 selector

```text
R
L
S-Y
S-L
```

S-Y 与 S-L 必须继续独立报告。

### 4.2 新增单变量 selector

| selector | 留出内容 |
|---|---|
| S-T | 连续温度区间 |
| S-RH | 连续湿度区间 |
| S-Flow | 风速或沿程流速区间 |
| S-Path | 声程或安装偏差 |
| S-SNR | 低 SNR 区间 |
| S-Delay | 系统延迟区间 |
| S-Dust | 衰减或粉尘区间 |
| S-Multipath | 多径强度区间 |

### 4.3 设备和时间 selector

| selector | 留出单位 |
|---|---|
| S-Device | 整台换能器 |
| S-Pair | 整对 TX/RX |
| S-Day | 整个采集日期 |
| S-Cal | 整个校准批次 |
| S-Cylinder | 整个气瓶或配气批次 |
| S-Site | 整个安装点或现场 |

### 4.4 联合 corner OOD

至少包括：

```text
高湿 + 低温 + 低 SNR
高流速 + 声程边界
留出设备 + 留出日期
高湿 + 粉尘衰减 + 增益下降
非均匀温度 + 多径
```

### 4.5 协议规则

1. 每个 selector 独立形成 OOD 集；
2. 保存 selector config 和 OOD set hash；
3. 不允许将多个 OOD 合并为单一平均 R²；
4. 训练 seed 与 split seed 分开；
5. 设备、日期和批次划分以组为单位；
6. 任何 feature selection 仅使用训练集；
7. calibration split 不与 test/OOD 重叠；
8. 同一 OOD 集上的重复 seed 只能量化训练方差，不能称为独立 OOD 证据。

### 正式指标

每个 selector 至少报告：

```text
O2 R²
O2 MAE
O2 RMSE
O2 bias
P90 absolute error
worst-bin MAE
worst-group MAE
sum_abs_error
coverage
interval_width
rejection_rate
train-support distance
```

### 通过门

新模型或新物理替代现有 B7 时，必须：

- ID test 不退化超过 `0.01 R²`；
- S-Y 不退化；
- S-L 不退化；
- 环境 OOD 平均和 worst-group 不退化；
- device/date holdout 不退化；
- 不确定性 coverage 不恶化；
- 3 split seed × 3 training seed 稳定。

### 产物

```text
outputs/tv3_ood_protocol_v2/
  selector_configs/
  split_hashes/
  split_metrics.parquet
  worst_group_metrics.parquet
  verdict.json
```

---

## WP5：最小真实硬件闭环

### 目标

用最小但有判别力的真实实验，验证仿真、DSP、校准、模型和 OOD 结论能否迁移。

### 5.1 实验维度

#### 组分

建议至少：

- O₂：5 个档位；
- CO₂：3 个档位；
- 包含边界、中间点和困难对照；
- 包含相同 O₂、不同 CO₂；
- 包含近似相同声速、不同组成。

#### 环境

建议至少：

- 3 个温度档；
- 3 个湿度档；
- 3 个流速档；
- 2 个声程；
- 2–3 对换能器；
- 3 个独立采集日期；
- 多个独立配气或气瓶批次。

### 5.2 每个实验循环

```text
零气
标准气
目标气
标准气复测
零气复测
```

用于估计：

- 零点漂移；
- span 漂移；
- hysteresis；
- 日内漂移；
- 切换残留；
- 响应时间；
- 恢复时间。

### 5.3 必需 metadata

```text
gas_certificate_id
gas_batch_id
reference_analyzer_id
reference_uncertainty
temperature_reference
humidity_reference
flow_reference
pressure_reference
transducer_pair_id
installation_geometry
calibration_session_id
acquisition_day
operator
```

### 5.4 划分规则

不得随机按帧拆分。

正式划分单位：

- 气瓶/配气批次；
- 日期；
- 换能器；
- 校准会话；
- 安装点。

### 5.5 分层验证

1. 仿真训练 → 仿真测试；
2. 仿真训练 → 实物测试；
3. 仿真训练 + 少量实物校准 → 实物测试；
4. 多设备训练 → 留出设备；
5. 多日期训练 → 留出日期。

### 通过门

- RawDSP 在真实波形上完成 fidelity 审计；
- corrected sound speed 对参考条件具有稳定 bias；
- B1/B7 在留出设备和留出日期上取得正向有效结果；
- prediction interval coverage 达标；
- 模型误差明显高于或可区分于参考标签不确定度；
- 不允许只用同日随机帧划分宣称成功。

### 产物

```text
outputs/tv3_hardware_pilot/
  acquisition_manifest/
  calibration/
  waveform_fidelity/
  device_holdout/
  day_holdout/
  sim2real/
  uncertainty/
  final_verdict.json
```

---

## WP6：B7 不确定性与拒绝预测

### 目标

在不扩大 B7 residual 容量的前提下，增加部署可信度输出。

### 6.1 输出形式

每个样本输出：

```text
prediction_raw3
o2_lower
o2_upper
confidence_level
support_distance
quality_score
ood_flag
reject_flag
reject_reason
```

### 6.2 基础 conformal

使用与训练、测试完全独立的 calibration split：

```text
residual_i = |y_i - yhat_i|
```

按残差分位数生成 90% 和 95% 区间。

### 6.3 条件区间

区间宽度可依赖：

```text
T
RH
flow
SNR
quality
reciprocity_error
device_id
support_distance
```

可比较：

- split conformal；
- conformalized quantile regression；
- 分组 conformal；
- covariate-shift weighted conformal。

### 6.4 OOD 检测

至少实现一种轻量支持距离：

- kNN distance；
- Mahalanobis distance；
- PCA/whitened distance；
- density ratio；
- calibration residual model。

### 6.5 拒绝策略

示例：

```text
区间宽度超过阈值
support distance 超过阈值
reciprocity error 超过阈值
双向校正残差超过阈值
SNR 低于阈值
关键 metadata 缺失
```

### 验收门

按每个 selector 独立评估：

- 90% 区间实际 coverage：目标 88%–92%；
- 95% 区间实际 coverage：目标 93%–97%；
- 每个关键环境组 coverage 不低于 nominal −5%；
- 区间宽度与样本难度具有合理单调关系；
- 拒绝后风险下降；
- 必须同时报告 rejection rate；
- 不允许通过拒绝大量样本只展示提高后的 R²。

### 产物

```text
outputs/tv3_b7_uq/
  calibration_manifest.json
  coverage_metrics.parquet
  risk_coverage_curve.csv
  rejection_audit.json
  verdict.json
```

---

## WP7：有限模型诊断，不开展无边界架构搜索

### 目标

只验证少数仍有明确科学问题的模型结构，不以刷新单一 R² 为目标。

### 7.1 独立 O₂ residual

结构：

```text
Ridge raw3
+
O2-only residual model
```

最终仍输出三组分 raw3，不回填 N₂。

回答：

- 共享 residual trunk 是否存在多任务梯度冲突？
- O₂ 是否需要独立容量？
- 能否以更少参数达到 B7 水平？

### 7.2 小型 flat residual MLP

固定候选：

```text
(32, 32)
(64, 32)
(64, 64)
```

目的：

- 判断 B7 是否过参数化；
- 判断更小模型是否更稳；
- 不继续扩大结构。

### 7.3 参数匹配 flat TabM

要求：

- flat 输入；
- 不使用物理早期分组；
- 与 B7 residual 参数量接近；
- 小型固定超参数网格；
- 同一协议比较。

### 7.4 强表格模型上限探针

用途：

- 判断 RawDSP 特征中是否仍存在 B7 未利用的非线性；
- 只作非部署上限；
- 必须覆盖环境和设备 OOD；
- 不得只跑 random split。

### 统一替代门

任何新头替代 B7，必须同时满足：

```text
test O2 R² 平均提升 ≥ 0.01
S-Y OOD 不下降
S-L OOD 不下降
环境 OOD worst-group 不下降
device/date holdout 不下降
closure monitor 不恶化
coverage 不恶化
3×3 seed 稳定
```

否则：

```text
保留 B7
停止该结构
不继续扩大搜索
```

### 明确禁止

- 重启 C1 bottleneck；
- C1 group dropout；
- C1 group gating；
- C1 encoder TabM；
- N₂ 回填；
- 继续给旧 D2 打补丁；
- 仅凭 val 增益推进；
- 只报告最优 seed。

---

## WP8：多频声学信息量预实验

### 目标

验证多频相位、吸收和色散是否能提供单频 TOF 之外的新 O₂ 信息。

### 8.1 候选激励

- 多个离散频点；
- 短 chirp；
- 多载波 burst；
- 在换能器有效带宽内进行。

### 8.2 每个频率提取

```text
tof_f
phase_f
amplitude_f
snr_f
coherence_f
quality_f
```

构造抗公共漂移特征：

```text
phase_fi - phase_fref
log(amplitude_fi / amplitude_fref)
tof_fi - tof_fref
group_delay
dispersion_slope
```

### 8.3 频率选择原则

不采用简单等间距。应最大化：

```text
O2 conditional Fisher information
```

并同时约束：

- 换能器响应；
- 湿度敏感度；
- 温度敏感度；
- 衰减；
- SNR；
- 设备相位稳定性；
- 增益漂移。

### 8.4 仿真矩阵

| 模式 | 目的 |
|---|---|
| 单频 TOF | 基线 |
| 多频 TOF | 检查相位速度增益 |
| 多频幅值 | 检查吸收信息 |
| 多频差分 | 抑制增益漂移 |
| 多频全部特征 | 上限 |
| 增益随机化后多频 | 检查真实性 |
| 湿空气 + 多频 | 检查 RH 混叠 |
| 设备 holdout + 多频 | 检查频响迁移 |

### 启动硬件门

只有同时满足：

- nuisance-marginalized O₂ information 明显增加；
- S-Y、S-L 均改善；
- 环境 OOD worst-group 改善；
- 窄窗口 O₂ MAE 改善；
- 增益和设备随机化后仍有效；
- 预测收益不是由仿真器泄漏造成；

才进入硬件多频实现。

---

## WP9：基于少量实测数据的 Sim2Real 校准

### 目标

使用少量真实波形拟合仿真 nuisance 参数分布，而不是无限扩大随机化范围。

### 流程

1. 获取无标签或少标签真实波形；
2. 估计真实噪声 PSD；
3. 估计换能器频响；
4. 估计固定延迟和漂移；
5. 估计增益、相位和 SNR 分布；
6. 估计多径和异常波形比例；
7. 拟合仿真参数后验或可信区间；
8. 在可信区间内 domain randomization；
9. 保留尾部压力测试域；
10. 比较校准前后 Sim2Real 性能。

### 对照

```text
固定仿真
宽范围盲随机化
实测校准随机化
实测微调
校准随机化 + 少量微调
```

### 通过门

- 仿真与实物的核心波形统计距离下降；
- 实物 holdout 性能改善；
- 不以牺牲仿真/环境 OOD 稳健性换取同设备拟合；
- 校准参数可解释并可复现。

---

## WP10：TDLAS fallback 决策

### 目标

建立声学路线停止条件，避免在物理上限附近无限调模。

### 重启条件

完成以下工作后：

- 双向流速补偿；
- 湿空气传播；
- 环境和设备 OOD；
- 最小真实硬件闭环；
- B7-UQ；
- 多频信息量验证；

若出现任一情况，则恢复 760 nm O₂ TDLAS：

1. 真实硬件 O₂ test R² 持续低于目标；
2. 0.8% 窄窗口 MAE 不满足业务要求；
3. prediction interval 过宽；
4. 拒绝率超过可接受范围；
5. device holdout 明显失败；
6. WP1 误差预算表明 nuisance floor 高于目标精度；
7. 多频没有增加可用信息；
8. 声学路线成本、复杂度或维护要求超过直接 O₂ 传感。

### 混合路线

若声学路线适合粗粒度监测但不适合精细 O₂：

```text
NDIR CO2
+
双向超声粗粒度 O2/N2
+
TDLAS O2 精确通道
```

声学可继续用于：

- 交叉验证；
- 冗余诊断；
- 传感器故障检测；
- 流速估计；
- 组合气体一致性检查。

---

# 6. 正式指标体系

## 6.1 整体回归

每个组分报告：

```text
R²
MAE
RMSE
bias
P90 absolute error
```

### 6.2 O₂ 固定业务区间

对预先固定的 O₂ bins 报告：

- MAE；
- bias；
- P90 absolute error；
- 相邻档位可分概率；
- 区间覆盖率；
- 样本数量。

bin 内 R²只保留为辅助诊断，不作为主要验收门。

### 6.3 Worst-group

按以下变量分组：

```text
T
RH
L
flow
SNR
device
day
calibration_session
gas_batch
composition_corner
```

报告：

- worst-group MAE；
- worst-group bias；
- worst-group coverage；
- worst-group interval width。

### 6.4 可靠性

```text
90% coverage
95% coverage
mean interval width
conditional coverage
risk-coverage curve
rejection rate
false-safe rate
```

### 6.5 物理与链路健康度

```text
peak P95
sound speed bias
flow residual
reciprocity error
SNR
multipath score
delay drift
zero drift
span drift
```

### 6.6 闭包监控

继续报告：

```text
sum_abs_error
```

但不得：

- 用闭包误差替代 O₂性能；
- 因 closure 较小宣称模型更正确；
- 混入 N₂ 回填。

---

# 7. 实验优先级矩阵

| 优先级 | 工作包 | 核心问题 | 是否改变模型 |
|---|---|---|---|
| P0 | WP0 基线冻结 | 后续结论能否复现 | 否 |
| P0 | WP1 可辨识性 | 物理上能否达到目标 | 否 |
| P0 | WP2 双向气流解耦 | 单向 TOF 是否被流速污染 | 否/特征变更 |
| P0 | WP3 数字孪生 v2 | 仿真是否过度理想化 | 否/物理变更 |
| P0 | WP4 OOD v2 | 是否接近部署变化 | 否 |
| P0 | WP5 实物闭环 | 仿真结论能否迁移 | 否 |
| P1 | WP6 B7-UQ | 预测何时可信 | 否 |
| P1 | WP7 有限模型诊断 | B7 是否还有有限结构收益 | 是 |
| P2 | WP8 多频 | 是否存在新的物理信息 | 可能 |
| P2 | WP9 Sim2Real 校准 | 如何缩小域差异 | 否 |
| Fallback | WP10 TDLAS | 声学失败后如何继续 | 硬件路线 |

---

# 8. 阶段化执行顺序

## 阶段 A：测量能力审计

执行：

```text
WP0 → WP1
```

输出：

- 基线冻结；
- 灵敏度矩阵；
- 误差预算；
- 理论分辨率；
- 是否继续追求窄区间回归的判定。

决策：

```text
若物理不可辨识：
    转为分档/趋势目标
    或进入多频/TDLAS
否则：
    进入阶段 B
```

## 阶段 B：显式物理解耦

执行：

```text
WP2 → WP3
```

输出：

- 双向声速；
- flow projection；
- 湿空气传播；
- 设备级参数；
- 新 builder 和 manifest；
- dry/humid、single/bidirectional 对照。

决策：

```text
若新物理无法改善环境稳定性：
    不扩大模型
    检查硬件和业务目标
否则：
    进入阶段 C
```

## 阶段 C：正式 OOD 和真实硬件

执行：

```text
WP4 → WP5
```

输出：

- 环境 OOD；
- 设备/date/batch holdout；
- 真实 waveform fidelity；
- Sim2Real gap；
- 实物 B1/B7 结果。

决策：

```text
若真实硬件失败：
    优先修物理、校准和传感链路
    不进行模型扩容
若真实硬件基本成立：
    进入阶段 D
```

## 阶段 D：部署可信度和有限模型验证

执行：

```text
WP6 → WP7
```

输出：

- prediction interval；
- OOD flag；
- reject flag；
- 风险-覆盖曲线；
- 独立 O₂ residual；
- 小模型和 flat TabM 对照。

决策：

```text
只有满足统一替代门的新头才能替换 B7
否则冻结 B7
```

## 阶段 E：信息源升级

执行：

```text
WP8 → WP9 → WP10 决策
```

输出：

- 多频信息量；
- 校准后仿真；
- 声学继续/停止 verdict；
- TDLAS 是否重启。

---

# 9. 停止条件

出现以下任一情况时停止对应方向：

## 9.1 模型方向

1. 新模型只提升 val，不提升 test 和 OOD；
2. 只在一个 selector 提升；
3. 只在一个 seed 提升；
4. 环境或设备 worst-group 退化；
5. coverage 退化；
6. 需要显著扩大参数量但提升 `<0.01 R²`；
7. flat 小模型已达到 B7 水平；
8. 新模型依赖非部署输入。

## 9.2 物理方向

1. WP1 显示理论误差下限高于业务目标；
2. 双向校正后流速残余仍不可控；
3. 湿空气和设备建模无法解释实测域差异；
4. 多频增益在设备随机化后消失；
5. 真实设备间迁移失败；
6. prediction interval 过宽；
7. 可接受拒绝率下仍不满足精度。

## 9.3 已冻结停止项

不得重启：

```text
C1 physical grouped bottleneck
C1 bottleneck dimension tuning
C1 group dropout
C1 group gating
C1 encoder TabM
旧 D2 patch 路线
ExtraTrees 无边界调参
N2 闭包回填
```

除非出现新的正式证据推翻当前结论，并另立冻结计划。

---

# 10. 决策树

```text
开始
 │
 ├─ WP1：理论上可达到目标吗？
 │      ├─ 否 → 粗粒度分档 / 多频 / TDLAS
 │      └─ 是
 │
 ├─ WP2：双向流速解耦有效吗？
 │      ├─ 否 → 修改声路与硬件结构
 │      └─ 是
 │
 ├─ WP3：湿空气与设备数字孪生能缩小域差异吗？
 │      ├─ 否 → 真实硬件参数识别 / 传感链路重构
 │      └─ 是
 │
 ├─ WP4/WP5：环境、设备和日期 OOD 通过吗？
 │      ├─ 否 → 修物理、校准和数据覆盖
 │      └─ 是
 │
 ├─ WP6：预测区间和拒绝机制达标吗？
 │      ├─ 否 → 不部署精确数值输出
 │      └─ 是
 │
 ├─ WP7：新头全面优于 B7 吗？
 │      ├─ 否 → 保留 B7
 │      └─ 是 → 替换默认头
 │
 └─ 最终部署 / 或根据停止条件重启 TDLAS
```

---

# 11. 建议新增工程入口

```text
tv3/physics/humid_air.py
tv3/physics/bidirectional_acoustics.py
tv3/physics/acoustic_measurement_v2.py

tv3/dsp/bidirectional_raw_dsp.py
tv3/dsp/multifrequency_raw_dsp.py
tv3/dsp/quality_metrics.py

tv3/ml/conformal.py
tv3/ml/ood_distance.py
tv3/ml/o2_residual_head.py
tv3/ml/flat_tabm_head.py

tv3/protocols/environment_ood.py
tv3/protocols/device_holdout.py
tv3/protocols/calibration_split.py

tv3/audit/identifiability.py
tv3/audit/error_budget.py
tv3/audit/sim2real_gap.py
```

建议命令：

```bash
python -m tv3.audit.identifiability ...
python -m tv3.protocols.environment_ood ...
python -m tv3.dsp.bidirectional_raw_dsp ...
python -m tv3.ml.conformal ...
python -m tv3.audit.sim2real_gap ...
```

---

# 12. 正式产物目录建议

```text
outputs/
├── tv3_baseline_freeze/
├── tv3_identifiability/
├── tv3_bidirectional/
├── tv3_measurement_v2/
├── tv3_ood_protocol_v2/
├── tv3_hardware_pilot/
├── tv3_b7_uq/
├── tv3_model_probes/
├── tv3_multifrequency/
├── tv3_sim2real_calibration/
└── tv3_final_route_decision/
```

每个正式目录至少包含：

```text
manifest.json
config.json
metrics.json
split_hashes.json
audit.json
verdict.json
README.md
```

---

# 13. 首批执行清单

## 必须立即执行

- [ ] 冻结 B7、RawDSP、split selector 和正式产物版本；
- [ ] 建立 O₂/T/RH/L/flow/delay/SNR 灵敏度矩阵；
- [ ] 计算全范围与窄窗口 O₂ 的等效 TOF 信号；
- [ ] 建立系统误差预算；
- [ ] 在仿真中加入双向流速传播；
- [ ] 创建 `raw_dsp_bidirectional_v1`；
- [ ] 设计 humid-air `acoustic_measurement_v2`；
- [ ] 固定环境 OOD selector 配置；
- [ ] 设计最小真实硬件采集矩阵；
- [ ] 定义 B7-UQ calibration split。

## 第二批执行

- [ ] 完成 dry/humid 和 single/bidirectional 对照；
- [ ] 完成 S-T、S-RH、S-Flow、S-SNR；
- [ ] 完成 S-Device、S-Day、S-Cal；
- [ ] 建立真实 waveform fidelity 审计；
- [ ] 建立 prediction interval 和 reject flag；
- [ ] 运行独立 O₂ residual；
- [ ] 运行小型 flat residual；
- [ ] 运行参数匹配 flat TabM。

## 条件启动

- [ ] 多频声学信息量仿真；
- [ ] 实测校准 domain randomization；
- [ ] 大规模无标签真实波形自监督；
- [ ] TDLAS 路线重启。

---

# 14. 最终建议

当前最合理的项目方向不是继续扩大模型，而是依次完成：

1. **确认物理可辨识性和误差下限；**
2. **解决单向 TOF 与通风流速混叠；**
3. **将湿度、设备、延迟和现场扰动纳入测量系统数字孪生；**
4. **建立环境、设备、日期和批次级 OOD；**
5. **完成小规模真实硬件闭环；**
6. **为冻结的 B7 增加预测区间和拒绝预测；**
7. **只保留少量有明确诊断目的的模型对照；**
8. **若声学路线仍无法满足目标，及时转向多频或 O₂ TDLAS。**

项目后续成功的标准应从：

```text
某个 split 上取得更高 O₂ R²
```

转为：

```text
在明确环境、设备和时间边界下，
系统具有可解释的误差预算、
稳定的 worst-group 性能、
校准有效的预测区间、
明确的拒绝机制，
并能在真实硬件上复现。
```
