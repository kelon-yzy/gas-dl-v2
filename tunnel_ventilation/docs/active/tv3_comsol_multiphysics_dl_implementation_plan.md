---
title: tv3 COMSOL 多物理场辅助掘进通风 DL 实施计划
date: 2026-07-20
status: paused
priority: paused
project: tv3 / tunnel ventilation
schema_target: tunnel-ventilation-comsol-1
---

# tv3 COMSOL 多物理场辅助掘进通风 DL 实施计划

> 状态：**⏸ 暂缓（2026-07-24）**。既有正式结论不变：G1 CFD smoke 已通过（2026-07-20），verdict=`g1_cfd_smoke_passed`；三档网格质量守恒与 medium→fine KPI 门通过。G2 及以后、正式 DOE / DL benchmark **不排期**。MRS-EI 的 MEI-0 没有恢复 G 线；后续恢复需另行决策。当前执行入口见 [MRS-EI 计划](tv3_mrs_information_efficient_inversion_experiment_plan.md)。
> 目标：用 COMSOL 6.3 补齐当前 `tv3` 缺失的通风输运、空间非均匀性和沿声路流速表示，形成可追溯的“隧道多物理场 → 局部传感器状态 → 现有传感器正演 → DL 训练与 OOD 审计”链路。
> 结论边界：本计划不撤销静止空气 v1 的 `information_source_upgrade_required`，不替换 B1/B7，不把数值仿真写成真实矿井、现场部署或安全联锁能力。

## Context

当前 `tv3` 已具备 `CO2/O2/N2` 三组分条件生成、NDIR/TCS 慢通道、单向超声波形、RawDSP、B1/B7 基线以及参数 holdout 审计。现有仿真仍是“均匀气体 + 单一声程 + 阶段曝光”的传感器观测代理：温湿压在单条序列内基本固定，组分由阶段 blend 变化，`flow_projection` 未表示。该链路适合受控静止气体的可辨识性研究，但不能表达风筒射流、工作面涡旋、旁路、低速滞留区、源项位置、空间浓度梯度和传感器安装位置。

COMSOL 6.3 本机预检已经确认：

- 会话版本为 `6.3`，standalone，16 核；
- `CFD`、`CHEM`、`HEATTRANSFER`、`ACOUSTICS`、`OPTICS`、`WAVEOPTICS`、`ACDC` 许可证均允许 checkout；
- 现有 `COMSOL/tv3_acoustic_p0_clean.mph` 只有 `Convected Wave Equation, Time Explicit`、`flow=0` 和瞬态研究，没有 multiphysics、solution 或 dataset；
- COMSOL 6.3 官方文档支持 RANS 流动与组分输运、非等温流，以及通过 `Background Fluid Flow Coupling + Mapping` 将 CFD 背景流单向映射到声学网格。

外部证据只用于确定建模要素，不直接迁移数值结论。掘进施工通风研究表明，工作面附近涡旋、低速区和旁路会形成污染物滞留，风筒位置、风量与源项时序共同改变清除过程；现场验证 CFD 是判断模型可信度的必要环节。跨领域 CFD 代理模型研究表明，DL 可以加速参数探索，但不能替代模型验证与 OOD 审计。

## Task

### 1. 研究问题与成功定义

| 编号  | 研究问题                                       | 必需证据                                                       | 可接受结论                   |
| --- | ------------------------------------------ | ---------------------------------------------------------- | ----------------------- |
| RQ1 | COMSOL 是否能表达当前生成器缺失的通风空间状态？                | 质量守恒、网格收敛、流场与浓度场验证、滞留区稳定性                                  | 数值模型在登记边界内可用于生成局部物理状态   |
| RQ2 | COMSOL 局部状态驱动的传感器数据是否改善 flow/geometry OOD？ | 与 E0 基线配对的独立 field-case holdout                            | 在已登记仿真域内改善，不外推为现场能力     |
| RQ3 | 双向声学是否能降低沿声路流速对组分反演的混叠？                    | AB/BA TOF、`u_parallel`、reciprocity residual 和 flow holdout | 支持建立新 builder；不得改写旧单向结果 |
| RQ4 | 多任务辅助监督是否优于仅扩大模型容量？                        | 固定主干与参数量的消融，辅助目标只用于训练                                      | 辅助目标在独立 OOD 上提供可重复增益    |
| RQ5 | 是否值得训练独立 CFD 代理模型？                         | COMSOL 数据规模、场重建误差、守恒误差、未见几何测试                              | 仅用于快速场预测或主动采样，不替代高保真验证集 |

本计划的完成不是“成功跑出一个 `.mph`”，而是同时满足：

1. 模型版本、几何、物理接口、边界、网格、study 和导出均可复现；
2. COMSOL 输出与 Python 数据契约之间只有一条显式转换链；
3. 每个训练样本能追溯到 COMSOL case、传感器布局、组分、设备 profile 和导出表达式；
4. 训练、校准和测试按独立物理 case 分组，不按帧或探针行随机拆分；
5. 任何通过结论都有对应的数值门、OOD 门和适用范围；
6. 任一关键输入、许可证、求解或导出失败时显式失败，不生成成功 manifest。

### 2. 硬不变量

#### 2.1 项目与标识

1. `mixture_id` 继续表示组分身份，格式沿用 `M000001`；不得回退、复制或重写为 `sequence_id`。
2. `sequence_id` 继续表示传感器时间序列实例，格式沿用 `Q000001`。
3. 新增 `field_case_id` 表示一次完整 COMSOL 输运求解，建议格式 `F000001`。
4. 新增 `sensor_layout_id` 表示传感器位置、姿态和声路集合，建议格式 `SL0001`。
5. 新增 `acoustic_case_id` 表示一个局部声学求解，建议格式 `A000001`。
6. 新正式 benchmark 不依赖 `base_condition_id`、`noise_seed_index` 或 `noise_seed`。
7. 随机性只在 manifest 的生成配置中登记；不得把随机种子当作业务主键或 split 依据。

#### 2.2 标签与模型

1. 正式主标签仍为 `x_CO2`、`x_O2`、`x_N2` 三列，单位为干基 `vol%`。
2. COMSOL 可在内部使用质量分数和约束方程，但导出层必须显式给出三列摩尔百分数并校验非负与和为 100%。
3. 模型仍直接输出 raw3；不使用 N2 回填、闭包残差头、ILR/ALR、`gas_head` 或新的隐藏 target transform。
4. true TOF、true sound speed、局部真值组分、CFD 流速和湍流量只能作为 oracle 审计或辅助目标，不能成为部署输入。
5. B1/B7、现有 RawDSP builder、旧数据集和旧 verdict 保持冻结；新物理必须使用新 schema、builder、manifest 和输出目录。

#### 2.3 COMSOL 6.3

1. 只使用 COMSOL 6.3 官方帮助、本机 6.3 GUI 生成 Java 或已核验的 6.3 API。
2. 每次自动建模前记录会话版本、连接方式、核心数、目标许可证 checkout 和已加载模型。
3. 空间物理接口必须显式绑定 geometry tag；不得依赖当前 MCP 包装器中的两参数 physics create 或整数 component 索引。
4. 边界不得靠猜编号配置。必须用命名选择或由几何构建脚本稳定生成的选择 tag。
5. 每个 MCP/MPh/Java 调用检查内部 `success` 或实际 Java 异常；外层调用完成不代表 COMSOL 操作成功。
6. `.mph` 文件被 GUI 或其他客户端锁定时停止保存，不静默改文件名。
7. 正式模型必须实际存在 solution、dataset，且目标表达式可求值；只有空 study 的模型不得标记 solved。

### 3. 总体架构

```text
场景参数与来源登记
  ├─ 几何：巷道、工作面、风筒、障碍物
  ├─ 通风：风量、出口速度、方向、压力边界
  ├─ 源项：位置、强度、持续时间、组分
  └─ 环境：T、RH、绝对压力、壁温
        ↓
COMSOL A：隧道输运模型
  稳态 RANS → 瞬态组分输运 → 可选热湿耦合
        ↓ 点/线/面/体派生量
局部状态中间层
  sensor point + acoustic path + face-zone metrics
        ├──────────────────────────────┐
        ↓                              ↓
Python 传感器正演                    COMSOL B：局部流动声学
NDIR/TCS/设备动态/噪声               CWE + background flow mapping
        └──────────────┬───────────────┘
                       ↓
现有 tv3 数组 + 物理辅助目标
  slow / ultrasonic / labels / aux_targets / provenance
                       ↓
DL-1 组分反演                  DL-2 可选场代理
  raw3 + interval + reject       稀疏观测 → 场/KPI
```

隧道尺度 CFD 与 200 kHz 声学不得在同一个全域瞬态 study 中强耦合。声学只在传感器附近的缩小几何中求解，并通过插值函数或 `Background Fluid Flow Coupling + Mapping` 接收局部背景状态。

### 4. COMSOL A：隧道输运模型

#### 4.1 几何层级

| 层级             | 用途                 | 几何要求                    | 是否产生正式证据 |
| -------------- | ------------------ | ----------------------- | -------- |
| G0-parametric  | API、命名选择、求解链 smoke | 参数化直巷道、单风筒、平工作面         | 否        |
| G1-engineering | DOE 与 DL 数据主来源     | 实际断面近似、风筒、主要设备障碍物、传感器位置 | 条件通过     |
| G2-site        | 最终现场验证             | 现场 CAD/BIM、实测风筒与设备布局    | 是，需实测校准  |

G0 参数化几何至少包含：

- 巷道长度、宽度、高度或拱形断面参数；
- 工作面和开放端；
- 风筒直径、出口到工作面距离、横向/竖向偏置和朝向；
- 掘进机或等效阻塞体；
- 传感器点、声路、呼吸高度平面和工作面危险区命名选择；
- 入口延长段，避免入口速度条件与壁面条件在尖角直接冲突。

正式 G1/G2 不得使用 `COMSOL/gas_chamber_simplified.*` 代替隧道几何；该几何只属于局部气室声学模型。

#### 4.2 物理接口选择

| 物理过程   | COMSOL 6.3 接口                       | 首轮设置                       | 进入条件                |
| ------ | ----------------------------------- | -------------------------- | ------------------- |
| 湍流通风   | `Turbulent Flow, SST`               | 不可压或弱可压 RANS；重力按场景登记       | G1 高保真与验证子集         |
| 批量筛选流场 | `Turbulent Flow, k-epsilon`         | 壁函数；用于低成本 DOE 候选           | 必须与 SST 子集比较后才可批量使用 |
| 三组分输运  | `Transport of Concentrated Species` | CO2/O2/N2，无反应，对流 + 分子/湍流扩散 | G1 必需               |
| 温度场    | `Heat Transfer in Fluids`           | 入口温度、壁面热边界、设备热源            | 仅在等温模型通过后加入         |
| 流热耦合   | `Nonisothermal Flow`                | 单向或双向浮力耦合由 Gr/Re 审计决定      | 热浮力不可忽略时启用          |
| 湿度     | H2O 作为附加组分或经 6.3 核验的湿空气接口           | 标签仍为干基 raw3                | 有现场 RH 边界或明确工程情景后加入 |
| 粉尘     | Particle Tracing 或多相流               | 不进入首轮三组分 benchmark         | 独立项目目标获批后再做         |

首轮不模拟化学反应。若使用名称中带 `Reacting Flow` 的预定义 multiphysics，反应源项必须为空并经质量守恒验证；不得为了使用预定义接口添加虚构反应。

#### 4.3 组分与基准转换

项目标签是干基摩尔百分数，COMSOL 输运优先使用质量分数。转换必须由单一模块完成：

```text
dry mole fraction → mixture molar mass → mass fraction
COMSOL mass-fraction field → wet/dry basis audit → dry mole fraction
```

每个探针时刻必须检查：

- `w_i >= 0`，`sum(w_i) = 1` 在数值容差内；
- `x_i >= 0`，`sum(x_CO2, x_O2, x_N2) = 100%` 在导出容差内；
- 加入 H2O 后，先从湿基总量剔除水蒸气，再输出干基 raw3；
- 质量分数和摩尔分数单位、基准、转换版本写入 manifest。

#### 4.4 边界与源项

G0 资料门必须收集或明确标记下列输入：

| 参数组 | 必需项                   | 缺失时处理                            |
| --- | --------------------- | -------------------------------- |
| 巷道  | 断面、有效长度、坡度、粗糙度        | 不进入 G1；只能跑 smoke                 |
| 风筒  | 直径、出口位置/姿态、风量或速度时程    | 不生成正式 flow 数据                    |
| 出口  | 开放端位置、绝对/表压定义         | 停止求解配置                           |
| 环境  | 入口 T、RH、现场绝对压力或海拔     | 不沿用旧 `0.10–0.709 MPa` 扫描作为隧道环境真值 |
| 源项  | 位置、体积/质量流率、组分、开始和持续时间 | 不创建隐藏默认源                         |
| 设备  | 主要阻塞体尺寸与位置            | 标为未表示并阻断现场外推                     |
| 验证  | 风速/浓度测点、仪器和不确定度       | 只能得到 numerical-only verdict      |

文献中的风筒速度、出口距离和污染物清除时间只可登记为 `literature_bound` 或 `engineering_scenario`，不得直接成为现场分布。首轮组合范围以项目当前组分边界为锚：`x_CO2=0.03–5.0%`、`x_O2=18.0–21.2%`，其余边界必须由 G0 registry 提供。

#### 4.5 当前检测模态下的参数影响分层

这里必须区分三个对象：

1. `y = [x_CO2, x_O2, x_N2]` 真值标签；
2. 传感器观测 `x`，包括 7 个 slow 通道和超声波形；
3. 风速/浓度验证测点，它们只用于判断正演是否可信。

对当前均匀气体、单声程、`flow=0` 的 `tv3` 模态，坡度、壁面粗糙度、风筒风量、源项体积流率和设备阻塞体都没有进入正演方程；验证测点永远不会改变正演结果。现场绝对压力不改变当前理想气体声速 `c_mix`，但会进入 `rho_mix`、`P_MPa` slow 通道和 NDIR 经验吸收项，因此不能从观测 nuisance 中删除。

| 参数        | 当前 `tv3` 标签影响 | 当前传感器观测影响 | G1/G2 COMSOL 输运模型处理                                                      |
| --------- | ------------- | --------- | ------------------------------------------------------------------------ |
| 坡度        | 无             | 无         | 若关闭重力且为等温不可压短段，可延后；启用高程静压、浮力或非等温流后重新纳入                                   |
| 壁面粗糙度     | 无             | 无         | 影响壁面摩阻、射流和回流；G1 用等效粗糙度，未实测时标为 `engineering_scenario`                     |
| 风筒风量      | 无             | 无         | 直接决定射流、稀释和局部流速，G1 必需                                                     |
| 源项体积流率    | 无             | 无         | 直接决定释放质量和组分时程，G2 必需；不得用阶段 blend 代替                                       |
| 设备阻塞体测绘   | 无             | 无         | 影响分离、回流和传感器局部输运；G1 可用等效阻塞体，G2/G7 再替换实测几何                                 |
| 现场绝对压力    | 无             | 有         | 保留为密度、吸收、压力通道和可压缩性 nuisance；不得直接继承旧 `0.10–0.709 MPa` 范围                  |
| 风速/浓度验证测点 | 无             | 无         | 不作为 forward input，移入 `validation_registry.json`；缺失时只能得到 `numerical-only` |

因此，G0 不应要求所有参数都先变成可扫描的求解变量：坡度、粗糙度和详细阻塞体可以在 MVP 阶段登记后延后；风筒风量、源项流率和传感器位置不能延后；验证测点不参与正演，但必须在进入现场校准前冻结。

### 5. COMSOL B：局部流动声学模型

#### 5.1 与现有 P0 的关系

- `tv3_acoustic_p0_clean.mph` 保留为 `flow=0`、均匀混合气的回归基准。
- 新模型建议命名 `tv3_acoustic_flow_p1.mph`，使用独立脚本、参数表和验收 CSV。
- 不在 P0 内添加流动、双向换能器或新材料逻辑，避免旧验收含义漂移。

#### 5.2 求解链

1. 从 COMSOL A 导出传感器局部体积中的 `u/v/w`、T、P、组分和密度，或通过 Mapping study 映射。
2. 在局部声学几何上建立 AB 与 BA 两个传播方向；换能器位置、路径长度和安装角度由 `sensor_layout_id` 决定。
3. 使用 `Convected Wave Equation, Time Explicit` 求解 tone burst。
4. 导出两端压力时程、峰值时刻、互相关 TOF、SNR 和 boundary-hit 标志。
5. 计算：

```text
t_ab = L / (c + u_parallel) + delay_ab
t_ba = L / (c - u_parallel) + delay_ba
c_est = L/2 * (1/t_ab_corr + 1/t_ba_corr)
u_est = L/2 * (1/t_ab_corr - 1/t_ba_corr)
```

6. 设备延迟不得假设完全相消。`delay_ab`、`delay_ba`、clock drift 和 reciprocity residual 必须独立登记。

#### 5.3 成本控制

不得为每个 CFD case 求解完整声学瞬态。采用三层策略：

1. P0 解析式与现有 Python 波形生成器覆盖大规模样本；
2. COMSOL 声学对选定的组分、温度、流速梯度和安装角度子集求解；
3. 用 COMSOL 子集校准或证伪 Python 传感器传递模型，并把模型差异作为设备/物理 profile，而不是把 COMSOL 波形复制扩增。

#### 5.4 NDIR：用 COMSOL 建模光路和吸收，不把它当作现成 NDIR 接口

COMSOL 6.3 官方 Heat Transfer 文档提供 `Radiative Beam in Absorbing Media (rbam)`。该接口在半透明介质中按 Beer–Lambert 定律求光束衰减，并可把吸收能量作为热源；它不是带有 CO2 NDIR 光源、滤光片、探测器、ADC 和标定曲线的专用 NDIR 传感器接口。

首轮 NDIR 子模型建议为局部气室的 2D axisymmetric 或简化 3D：

```text
光源/准直孔 → CO2 吸收气室 → 窄带滤光片/参考通道 → 探测面
```

在气室域定义频带相关吸收系数：

```text
kappa_lambda = sum_i[n_i * S_i(T) * g_i(lambda, T, P)] + kappa_H2O + kappa_window
I_det(lambda) = I_src(lambda) * exp(-integral(kappa_lambda * ds))
V_NDIR = detector_gain * integral_A(I_det * responsivity(lambda) dA) + electronics_noise
```

其中 `i` 至少包括 CO2，并按需要加入 H2O 和窗口/粉尘损失。`S_i`、线型 `g_i`、压力展宽、温度展宽和滤光片带宽必须来自外部光谱资料或实测标定；COMSOL 不会自动从 `x_CO2` 推导真实 4.26 μm 光谱吸收。没有这些数据时，只能使用显式的 `kappa_model_version` 经验模型，不得把 `rbam` 求解标成真实 NDIR 复现。

NDIR 的 COMSOL 变量应作为中间物理量导出：`I_src`、`I_det`、光程积分吸收率、`kappa_eff`、探测面功率和参考/测量通道比值。ADC 量化、零点/span、T90、漂移、噪声 PSD 和异常帧继续由 Python 设备层生成，除非有完整电子学和标定模型。

`Electromagnetic Waves, Frequency Domain` 只在需要分析窗口干涉、滤光片、电磁散射或局部器件结构时启用。直接用全波 Maxwell 模型覆盖 4.26 μm 在 0.2 m 级气室中的传播会产生极高网格成本，而且不能自动解决分子谱线参数；它不是 NDIR 首选路线。

NDIR 进入正式数据集前必须通过：

1. 均匀气体 Beer–Lambert 解析对照；
2. `x_CO2` 单调性、`x_O2/x_N2` 近似无交叉敏感性；
3. T、RH、P、光程和窗口损失独立扫描；
4. 测量/参考双通道比值对光源增益扰动的抑制；
5. 至少一个独立气体或参考吸收测量校准。

#### 5.5 TCS：用热-电耦合量化热导边际，不假定存在专用 TCS 接口

TCS 在本项目中应理解为加热元件/热导桥对气体导热、对流散热和温度变化的测量。COMSOL 6.3 可用下列基础接口组合：

```text
Electric Currents
  + Heat Transfer in Fluids/Solids
  + Joule Heating
  + 可选 Laminar Flow 或 Conjugate Heat Transfer
```

建议建立局部传感器 cell，而不是把热导丝直接放进全巷道网格：

1. 细丝、膜片、参考元件和气室壁采用真实或等效几何；
2. Electric Currents 求加热功率与电阻；
3. Heat Transfer 求固体导热、气体导热、壁面散热和局部对流；
4. 气体 `k_mix(T,x_CO2,x_O2,x_N2,x_H2O)`、`rho_mix`、`cp_mix`、黏度和必要的流动边界由自定义材料表达式或插值函数给出；
5. 输出热丝温度、热流、稳态电阻/桥路差分和瞬态时间常数，再映射到 `V_TCS`。

COMSOL 的 Moist Air Properties 可提供干空气与蒸汽的混合物热物性，但这不等于 CO2/O2/N2 TCS 标定。CO2、O2、N2 的导热率、热容、黏度和压力修正仍需明确物性来源；若使用经验混合律，必须写入 `tcs_property_model_version`。

TCS 的首要用途不是假设它能显著恢复 O2，而是量化：

```text
composition → k_mix / heat loss → filament temperature → V_TCS
```

在当前 O2/N2 变化范围内，若 `Delta V_TCS` 小于温度、流速、压力、加热功率和电子噪声造成的合成不确定度，则 TCS 只保留为环境/质量通道，不作为主要组分信息源。

TCS COMSOL 子模型必须通过：

1. 单组分气体导热和热丝温升解析/实验对照；
2. CO2/O2/N2 混合物 `k_mix`、`rho_mix`、`cp_mix` 的物性单元测试；
3. 加热功率、壁温、局部流速和压力的独立敏感性；
4. 稳态 `V_TCS` 单调性与滞后/T90 对照；
5. O2 窄窗口的 nuisance-marginalized Fisher 或有限差分信号审计。

#### 5.6 NDIR/TCS 的路线判定

| 通道   | COMSOL 可解决的核心问题              | 不能自动解决的部分                 | 推荐路线                                 |
| ---- | ---------------------------- | ------------------------- | ------------------------------------ |
| NDIR | 光程、组分/温压/RH 相关吸收、窗口/光路和探测面功率 | 分子光谱参数、滤光片实测响应、ADC、漂移、T90 | 选定 case 用 `rbam` 校准，批量仍用 Python 传感器层 |
| TCS  | 热丝、电热、气体导热、局部流动、热时间常数        | 真实器件结构、桥路电子学、物性与小信号标定     | 选定 case 用热-电耦合验证，批量仍用 Python 层       |
| 超声   | 背景流、声速、TOF、AB/BA 传播          | 换能器真实频响、线缆、时钟和安装耦合        | COMSOL 局部高保真校准，现有 P0/Python 批量正演     |

正式原则是“COMSOL 物理子集校准 + Python 大规模观测生成”，而不是把三种传感器全部重写成 COMSOL 模型。只有当 COMSOL 子集显著改变误差预算、解释实测域差，并且通过独立设备/日期 holdout，才允许升级相应的 `physics_version`。

### 6. Study、网格和数值验证

#### 6.1 Study 顺序

| Study | 内容                | 输入                 | 输出                        |
| ----- | ----------------- | ------------------ | ------------------------- |
| ST1   | 稳态 RANS           | 几何、风筒、出口、壁面        | `u,v,w,p,k,omega/epsilon` |
| ST2   | 瞬态组分输运            | ST1 冻结流场、源项时程      | `w_CO2,w_O2,w_N2`         |
| ST3   | 非等温流              | ST1/ST2、热边界        | `T,rho` 与浮力修正             |
| ST4   | Mapping           | CFD solution 与声学网格 | 映射后的背景流和热力状态              |
| ST5   | CWE Time Explicit | ST4、TX/RX          | AB/BA 压力波形与 TOF           |

ST3–ST5 逐门开启。ST1 或 ST2 未通过时不得继续下游求解。

#### 6.2 网格策略

1. CFD 网格至少包含 coarse、medium、fine 三档；风筒出口、射流剪切层、工作面、障碍物尾流和传感器区域局部加密。
2. SST 子集根据近壁处理登记目标 y+；实际 y+ 分布作为结果导出，不只记录目标值。
3. 组分边界层和高梯度源区不得仅依赖全局 element size。
4. 声学网格独立于 CFD 网格，按波长和 CWE 离散阶次设计。
5. `Background Fluid Flow Coupling` 使用独立 Mapping study；曲面几何的 shape function 与 CWE 高阶离散必须一致。

#### 6.3 数值门

建议首轮门限如下；正式值可在 G0 评审时收紧，但不得无记录放宽：

| 指标                      | 门限               | 失败动作           |
| ----------------------- | ---------------- | -------------- |
| 全局质量流入/流出不平衡            | <= 1%            | 修边界、网格或求解器     |
| 组分总质量预算残差               | <= 1%            | 修源项、出口或时间步     |
| medium→fine 关键 KPI 相对变化 | <= 5%            | 继续加密或降级结论      |
| 探针组分闭包误差                | <= 1e-4 fraction | 停止导出与打包        |
| 负质量分数                   | 不允许超过登记数值容差      | 修离散稳定性，不截断伪装成功 |
| 稳态残差与监控量                | 达到登记收敛准则且无持续漂移   | 不接受“迭代到上限”作为收敛 |
| AB/BA 无流 reciprocity    | 在设备延迟校正容差内       | 修声学选择、网格或延迟模型  |
| P0 `c/TOF` 对照           | 延续现有 `<0.1%` 声速门 | 不进入流动声学        |

### 7. DOE 与多保真采样

#### 7.1 参数分组

| 分组          | 参数示例             | split 角色          |
| ----------- | ---------------- | ----------------- |
| composition | CO2、O2、N2        | mixture holdout   |
| geometry    | 断面、长度、坡度、障碍物     | geometry OOD      |
| ventilation | 风量、风筒距离/偏置/角度、模式 | flow OOD          |
| source      | 位置、强度、持续时间、开始时刻  | source OOD        |
| environment | T、RH、绝对压力、壁温     | environment OOD   |
| layout      | 传感器坐标、姿态、路径长度    | sensor-layout OOD |
| instrument  | 换能器、延迟、增益、频响、噪声  | device OOD        |

#### 7.2 采样顺序

1. **screening**：Morris 或小型正交设计识别主导参数，不训练 DL。
2. **core DOE**：对主导连续变量做 LHS；类别变量做分层组合。
3. **boundary DOE**：主动加入射流刚好到达/未到达工作面、回流区突变和浓度安全边界附近样本。
4. **replication**：相同物理 case 不通过不同 noise seed 伪装独立 case；设备噪声只生成同 case 下的 sequence 实例。
5. **active learning**：首轮模型训练后，以 ensemble disagreement、field reconstruction error 和物理残差选择新增 COMSOL case。

#### 7.3 多保真层级

| fidelity | 模型                      | 作用                | 可否进入 test |
| -------- | ----------------------- | ----------------- | --------- |
| F0       | 2D/粗 3D、简化湍流            | API、灵敏度、排错        | 否         |
| F1       | 3D k-epsilon 或经验证的低成本模型 | 批量 DOE            | 仅 ID/训练   |
| F2       | 3D SST、medium/fine 网格   | 高保真标签与模型差异审计      | 是         |
| F3       | 现场几何 + 实测边界             | 最终 Sim2Real 校准与测试 | 是，需真实参考测量 |

F1 只有在 F2 配对子集上通过速度、回流区和组分 KPI 差异门后才能成为训练主来源。F2/F3 测试集不得参与代理模型主动采样后的再训练。

### 8. 数据契约

#### 8.1 中间层 schema

新 schema 建议为 `tunnel-ventilation-comsol-1`。COMSOL case registry 与最终 DL benchmark 分开：

```text
data/comsol_transport/
  case_registry.csv
  parameter_registry.json
  geometry_registry.json
  sensor_layout_registry.json
  cases/F000001/
    model_manifest.json
    mesh_metrics.json
    solver_metrics.json
    probe_timeseries.csv
    path_timeseries.csv
    zone_metrics.csv
    export_manifest.json

data/tv3_comsol_benchmark/
  manifest.json
  condition_grid_sequence.csv
  sequence_index.csv
  splits.csv
  sequences/
    slow.npy
    ultrasonic_int16.npy
    ultrasonic_scale.npy
    local_state.npy
    path_state.npy
    flow_projection.npy
    acoustic_tof_ab.npy
    acoustic_tof_ba.npy
    ndir_path_absorption.npy
    ndir_detector_power.npy
    tcs_filament_temperature.npy
    tcs_voltage.npy
  labels/y.npy
  metadata/
    sequence_ids.npy
    mixture_ids.npy
    field_case_ids.npy
    sensor_layout_ids.npy
    slow_channel_names.npy
    aux_target_names.npy
```

#### 8.2 核心表字段

`case_registry.csv` 至少包含：

```text
field_case_id, geometry_id, ventilation_id, source_id,
mixture_id, environment_id, fidelity_level,
model_sha256, parameter_sha256, mesh_sha256,
study_status, solution_tag, dataset_tag, export_status
```

`sequence_index.csv` 至少包含：

```text
sequence_id, mixture_id, field_case_id, sensor_layout_id,
device_profile_id, calibration_profile_id,
n_timesteps, dt_s, status
```

每个 array 的 shape、dtype、单位、坐标基准、插值方式和表达式写入 `export_manifest.json`。数组之间只按 `sequence_id` 显式对齐，不依赖目录排序或隐式行号。

#### 8.3 局部状态定义

`local_state.npy` 建议形状 `[N, T, S, F]`：

```text
F = x_CO2, x_O2, x_N2, T_C, P_Pa, H_RH,
    u_x, u_y, u_z, speed, turbulence_k, turbulence_scale
```

`path_state.npy` 建议形状 `[N, T, P, F_path]`：

```text
F_path = path_mean_x_CO2, path_mean_x_O2, path_mean_x_N2,
         path_mean_T, path_mean_u_parallel,
         path_std_T, path_std_u_parallel, path_gradient_score
```

`zone_metrics.csv` 保存工作面和呼吸高度区域 KPI，例如区域均值、P95、最大值、低速体积分数、回流体积分数和达到登记阈值的清除时间。它们服务于通风场代理任务，不进入组分反演的部署输入。

### 9. DL 训练路线

#### 9.1 DL-1：组分反演主线

主任务继续使用现有 `TunnelVentilationDataset` 和 raw3 输出。第一阶段通过 `aux_target_arrays` 接入：

- `flow_projection`；
- `path_mean_T`；
- `path_mean_x_CO2/x_O2/x_N2`；
- `tof_ab/tof_ba`；
- `reciprocity_error`；
- 可选 `sensor_support_score`。

NDIR 的 `ndir_path_absorption`、`ndir_detector_power` 和 TCS 的 `tcs_filament_temperature`、`tcs_voltage` 默认属于物理审计与辅助监督，不自动进入部署输入。只有当它们能由实际硬件观测或可部署校准量稳定替代时，才可以在独立的 observed builder 中加入；COMSOL 真值本身不得直接喂给正式推理模型。

辅助目标只在训练时使用。验证和部署输入仍限定为真实可观测 slow、waveform、设备登记元数据。loss 形式建议为：

```text
L = L_raw3
  + lambda_flow * L_flow
  + lambda_tof * L_tof
  + lambda_state * L_path_state
```

`lambda_*` 必须由训练集或 calibration split 决定，不得查看 test/OOD 后调整。辅助任务缺失时显式报错；不静默置零继续训练。

#### 9.2 DL-2：通风场/KPI 代理

这是独立任务，输入为场景参数、稀疏传感器和风机控制量，输出为：

- 指定平面浓度/速度场；或
- POD/autoencoder latent；或
- 工作面区域 KPI 与清除时间。

推荐顺序：

1. POD + Ridge/MLP 作为最低复杂度基线；
2. 规则网格切片的 U-Net/CNN；
3. 几何变化明显时再评估 graph/operator learning；
4. 只有低阶基线无法满足未见 case 门时才增加复杂度。

场代理必须报告质量/组分守恒误差和物理边界违例，不能只报告像素 MSE。

#### 9.3 Sim2Real 使用顺序

1. 用少量实测数据估计边界与设备参数，不直接掩盖仿真域差；
2. 比较 fixed simulation、宽范围盲随机化、实测校准随机化、实测微调和二者组合；
3. calibration 设备、日期和气瓶批次与 test 完全隔离；
4. 真实数据不足时结论为 `simulation_only`，不输出现场性能数字。

### 10. 正式实验矩阵

| 实验    | 数据                                  | 模型          | 目的                   |
| ----- | ----------------------------------- | ----------- | -------------------- |
| E0    | 现有均匀 `flow=0`                       | 冻结 B1/B7    | 旧物理基线                |
| E1    | COMSOL 局部状态 → 现有传感器正演               | 冻结 B1/B7    | 判断空间/流动扰动造成的域差       |
| E1-NT | COMSOL NDIR/TCS 物理子集 → P1/P2 观测通道   | 冻结 B1/B7    | 判断光学/热导高保真模型是否改变通道结论 |
| E2    | E1 + `u_parallel/T/path-state` 辅助目标 | 相同主干、相近参数量  | 判断多任务是否提高 OOD        |
| E3    | E1 + 单向超声                           | 冻结头         | 流速混叠基线               |
| E4    | E1 + 双向超声                           | 冻结头         | 双向解耦增益               |
| E5    | E4 + COMSOL 声学校准 profile            | 冻结头         | 判断解析波形与 FEM 差异       |
| E6    | F1 批量训练 + F2 测试                     | B7/候选 DL    | 多保真泛化                |
| E7    | 实测校准随机化                             | 冻结 protocol | Sim2Real；资料恢复后执行     |

每个实验至少报告：

- val/test 与每个 OOD selector 的 CO2/O2/N2 R2、MAE、RMSE、bias、P90 absolute error；
- `sum_abs_error`、区间 coverage/width、rejection rate；
- flow、geometry、source、layout、device worst-group；
- 相对 E0/E1/B7 的配对差值和多 seed 稳定性；
- COMSOL case 数、fidelity 构成、失败 case 和剔除原因。

### 11. Split 与泄漏控制

1. 不按 timestep、probe row 或同一 case 的不同噪声实例随机拆分。
2. `mixture_id` group split 继续用于组成泛化；flow 路线新增 `field_case_id` 与 `sensor_layout_id` 分组。
3. 正式 selector 至少包含：

```text
R-composition
S-Flow
S-Geometry
S-Source
S-Environment
S-Layout
S-Device
S-JointExtreme
```

4. 相同 geometry/ventilation family 的相邻参数点不得跨 train/test 造成插值伪 OOD；family 规则和 hash 必须保存。
5. scaler、POD basis、特征选择、template 和 domain-randomization 参数只由 train 或独立 calibration split 拟合。
6. F2/F3 高保真 test 在实验开始前冻结；主动学习不得查询其误差来选新训练 case。

### 12. G0–G7 执行阶段

#### G0：资料、版本与参数来源冻结

输入：现场或工程几何、风筒、环境、源项、传感器和验证测点资料。

任务：

- 建立 `parameter_registry.json` 和 `geometry_registry.json`；
- 每个参数标记 `implemented_physics`、`measured_bound`、`literature_bound`、`engineering_scenario` 或 `not_represented`；
- 核验 COMSOL 6.3 接口、Java 签名、许可证和输出路径；
- 冻结 schema、ID、单位、坐标系和命名选择规范；
- 记录旧压力范围与现场绝对压力的差异，不默认继承。

通过：所有 G1 必需参数有来源，阻断项为零。否则 verdict=`g0_input_blocked`。

#### G1：最小 CFD smoke 与稳定几何

任务：

- 用原生 COMSOL 几何建立参数化直巷道、风筒和工作面；
- 建立稳定命名选择；
- 完成 ST1、三档网格和质量守恒检查；
- 导出速度、压力、流线、回流区和探针表；
- 保存可重新加载的 `.mph` 与 MPh/Java 构建脚本。

通过：`g1_cfd_smoke_passed`。smoke 不进入正式训练。

#### G2：三组分瞬态输运

任务：

- 加入 Concentrated Species 和源项时程；
- 完成质量/摩尔分数转换；
- 运行 ST2，验证闭包、质量预算和时间步收敛；
- 与零流/充分混合解析极限做 sanity check；
- 生成首批 probe/path/zone 导出。

通过：`g2_transport_validated`；否则停止下游数据打包。

#### G3：传感器投影与 Python 对齐

任务：

- 定义传感器点、声路、局部体积与区域算子；
- COMSOL 局部状态转换为当前 slow 与 waveform 正演输入；
- 对均匀 `flow=0` case 与现有 P0/Python 做 parity；
- 建立 `field_case_id → sequence_id` 显式映射；
- 完成数组 shape、dtype、单位和 hash 审计。

通过：`g3_sensor_projection_passed`。

#### G3a：NDIR/TCS 物理子集校准

该阶段独立于隧道全域 CFD，可以在局部气室、有限组分和有限设备 profile 上执行。

任务：

- NDIR：`rbam` Beer–Lambert 解析对照、测量/参考通道、吸收系数版本和探测面功率导出；
- TCS：Electric Currents、Heat Transfer、Joule Heating 及局部流动边界的热丝/桥路模型；
- 将 COMSOL 中间量映射为 `V_NDIR_CO2` 与 `V_TCS`，并与现有 Python P1/P2 通道对齐；
- 分别执行 composition、T/RH/P、flow、path、gain 和 noise 敏感性；
- 判断 NDIR/TCS 是主信息通道、环境校正通道，还是仅保留为审计通道。

通过：`g3a_ndir_tcs_physics_passed`。缺少光谱参数、TCS 物性或器件标定时，允许输出 `physics_subset_only`，但不得把它写成真实传感器复现。

#### G4：双向流动声学

任务：

- 新建 `tv3_acoustic_flow_p1.mph`；
- 完成 ST4/ST5、AB/BA 波形、TOF 和 reciprocity；
- 对均匀背景流解析式与 COMSOL 结果做对照；
- 扫描局部梯度、安装角和 flow projection；
- 冻结新 acoustic builder 与设备延迟版本。

通过：`g4_bidirectional_acoustics_passed`。失败时保留单向 flow nuisance，不训练宣称已解耦的模型。

#### G5：数据打包与基线重放

任务：

- 写入 `tunnel-ventilation-comsol-1` 数据集；
- 增加 schema、integrity、split 和 aux target 测试；
- 在 E0 数据重放 B1/B7，确认代码改动未改变旧结果；
- 在 E1 运行 frozen baseline，量化纯物理域差；
- 冻结 F2 test 和所有 set hash。

通过：`g5_dataset_frozen`。

#### G6：DL 训练与消融

任务：

- 按 E1、E1-NT、E2–E6 顺序执行，不跳过 frozen baseline；
- 辅助监督、双向声学和多保真逐项加入；
- 至少 3 split seed × 3 training seed；
- 报告 ID、所有 OOD、worst-group、区间与拒绝指标；
- 记录失败实验，不通过调参覆盖失败证据。

通过：`g6_simulation_ood_supported` 只表示登记仿真域支持。

#### G7：现场校准与最终判定

前置：恢复真实实验，并具备独立参考分析仪、风速、温湿压、时间戳、设备/日期/气瓶批次信息。

任务：

- 校准边界、噪声 PSD、换能器频响、固定延迟、漂移和多径；
- 执行留出设备、日期、气瓶和安装点测试；
- 评估仿真/实测统计距离、区间覆盖和拒绝策略；
- 决定 `site_calibrated_supported`、`simulation_only` 或 `information_source_upgrade_required`。

### 13. 判定门

| verdict                               | 条件                                 | 后续                   |
| ------------------------------------- | ---------------------------------- | -------------------- |
| `g0_input_blocked`                    | 现场边界或参数来源不足                        | 补资料；不得批量求解           |
| `numerical_model_failed`              | 守恒、网格、闭包或收敛失败                      | 修 COMSOL 根因；不得导出训练数据 |
| `g3_sensor_projection_passed`         | 均匀极限 parity 与数据契约通过                | 可生成 E1               |
| `g4_bidirectional_acoustics_passed`   | AB/BA 解析对照、reciprocity 和 flow 扫描通过 | 可进入 E4/E5            |
| `g6_simulation_ood_supported`         | 多 seed、flow/geometry OOD 与研究门通过    | 保留为仿真证据；不宣称现场能力      |
| `simulation_information_insufficient` | 加入 flow/space 后仍无法达到信息门            | 停止扩大网络；评估传感器/信息源升级   |
| `simulation_only`                     | 无真实校准或现场 OOD 失败                    | 不输出现场性能结论            |
| `site_calibrated_supported`           | G7 独立硬件/日期/场景测试与可靠性门通过             | 才可进入现场试运行评审          |

研究门继续沿用项目当前口径：O2 P90 absolute error `<= 0.4 vol%`、主导 nuisance 占窄窗口信号 `<= 50%`、rejection rate `<= 5%`。它们是研究判定门，不是法规或安全联锁声明。

### 14. 预期代码与文件边界

建议新增而非改写：

```text
tunnel_ventilation/
  COMSOL/
    tunnel_transport/
      README.md
      models/
        tv3_tunnel_transport_g0.mph
        tv3_tunnel_transport_g1.mph
        tv3_acoustic_flow_p1.mph
      scripts/
        build_tunnel_transport_g0.py
        verify_tunnel_transport.py
        export_sensor_states.py
        build_acoustic_flow_p1.py
        compare_bidirectional_tof.py
      configs/
        parameter_registry.json
        geometry_registry.json
        sensor_layout_registry.json
  tv3/sim/comsol/
    schema.py
    ids.py
    composition.py
    registry.py
    export_reader.py
    sensor_projection.py
    integrity.py
  tv3/pipeline/
    generate_comsol_tunnel_benchmark.py
    audit_comsol_tunnel_benchmark.py
  configs/
    tv3_comsol_transport_smoke.json
    tv3_comsol_transport_formal.json
    tv3_comsol_dl_e1.json
    tv3_comsol_dl_e2_aux.json
  tests/
    test_comsol_transport_schema.py
    test_comsol_composition_conversion.py
    test_comsol_export_integrity.py
    test_comsol_sensor_projection.py
    test_comsol_field_case_splits.py
```

复杂 COMSOL 建模优先使用经 6.3 文档核验的 MPh/Java 脚本，并保留 GUI 生成的 Java 作为签名证据。当前 MCP 适合会话、加载、检查和结果读取；已知 component/physics create overload 风险未修复前，不用通用 MCP 建模工具批量创建正式模型。

### 15. 测试与验证清单

#### 15.1 单元测试

- 摩尔分数 ↔ 质量分数往返误差；
- 湿基 → 干基 raw3 转换；
- `field_case_id`、`mixture_id`、`sequence_id` 唯一性与语义；
- export manifest 的 shape/dtype/unit；
- probe/path 时间轴对齐；
- 同一 field case 不跨 split；
- 缺失 solution/dataset/export 时显式失败；
- 负组分、闭包误差和单位不一致时失败；
- `aux_target_arrays` 名称、维度与 sequence count 对齐。

#### 15.2 COMSOL 验证

- 会话版本与许可证；
- 模型树 component/geometry/mesh/physics/study/solution/dataset；
- 三档网格、质量守恒、时间步收敛；
- 零源项、零流、均匀场和充分混合极限；
- AB/BA 无流 reciprocity 与均匀流解析 TOF；
- `.mph` 保存、卸载、重新加载和表达式求值。

#### 15.3 端到端 smoke

```text
1 个 G0 geometry
× 2 个 ventilation cases
× 2 个 mixtures
× 1 个 sensor layout
→ COMSOL solve
→ probe/path export
→ Python sensor projection
→ benchmark integrity
→ 1 epoch DL smoke
```

smoke 只验证链路，不产生性能 verdict。

### 16. 风险、停止条件与排错顺序

| 风险                  | 证据                                   | 停止/修复顺序                                          |
| ------------------- | ------------------------------------ | ------------------------------------------------ |
| 边界资料不足              | 参数 registry 存在 `not_represented` 阻断项 | 停在 G0，不添加默认值                                     |
| RANS 模型不稳定          | k-epsilon/SST 回流区与 KPI 差异大           | 增加验证，不批量生成数据                                     |
| 全域模型过大              | 内存、时间或网格不可接受                         | 分离 steady flow、transient species、local acoustics |
| 组分数值振荡              | 负质量分数或闭包失败                           | 修网格/离散/时间步，不裁剪结果                                 |
| COMSOL/Python 语义分叉  | 单位、干湿基或表达式不一致                        | 统一转换模块，删除重复公式                                    |
| DL 记住 geometry/case | random split 好、field holdout 失败      | 修 split，不增加网络容量                                  |
| COMSOL 数据量不足        | 高方差或 active-learning 不收敛             | 优先补主导 case，不复制噪声实例                               |
| Sim2Real 域差大        | 实测统计与仿真明显分离                          | 校准物理与设备 profile，不用少量微调掩盖                         |
| 双向声学无增益             | reciprocity/flow holdout 不改善         | 保留为否定证据，评估其他信息源                                  |

统一排错顺序：输入与单位 → 几何/选择 → 网格 → 物理边界 → study 依赖 → solver → dataset/export → Python 转换 → DL。不得在下游训练中补偿上游物理或数据契约错误。

### 17. Non-goals

1. 不在本计划中建立火灾、瓦斯爆炸、粉尘爆炸或安全联锁模型。
2. 不用道路隧道、室内 HVAC 或其他矿井的数值范围替代本项目现场参数。
3. 不把 CO、NOx 或粉尘标签混入当前 `CO2/O2/N2` raw3 benchmark；如需研究，另立 schema 和任务。
4. 不把 COMSOL 整个非结构网格直接塞入现有组分反演网络。
5. 不用 PINN、FNO、GNN 或更深网络作为立项理由；先通过 POD/Ridge/MLP 与冻结 B7 基线。
6. 不覆盖现有 P0 `.mph`、P0/P1/P2 acceptance CSV、B1/B7 或静止空气产物。
7. 不生成隐藏 fallback、mock solve、空 dataset 成功标记或缺失 case 的模拟结果。
8. 不因仿真 OOD 通过而改写真实静止空气、真实通风或现场能力声明。

## Format

### 18. 阶段交付格式

每个 G 阶段必须提交：

```text
1. 输入与来源：registry + SHA256
2. 实现：代码、配置、模型路径和 COMSOL 6.3 API 证据
3. 数值结果：solver/mesh/conservation/export metrics
4. 数据结果：schema/shape/unit/split/hash
5. DL 结果：baseline、OOD、worst-group、interval、reject
6. 失败 case：原始错误、原因和处理状态
7. verdict：枚举值、通过门、阻断项和适用范围
```

`verdict.json` 最小结构：

```json
{
  "schema_version": "tunnel-ventilation-comsol-1",
  "stage": "G2",
  "verdict": "g2_transport_validated",
  "comsol_version": "6.3",
  "input_registry_sha256": "...",
  "model_sha256": "...",
  "dataset_sha256": "...",
  "passed_gates": [],
  "failed_gates": [],
  "blocking_items": [],
  "claim_scope": "registered_simulation_domain_only"
}
```

### 19. 正式产物目录

```text
outputs/runs/tv3_comsol_multiphysics/
  g0_registry/
  g1_cfd_smoke/
  g2_transport_validation/
  g3_sensor_projection/
  g4_bidirectional_acoustics/
  g5_dataset_freeze/
  g6_dl_protocol/
  g7_site_calibration/
outputs/summary/tv3_comsol_multiphysics/
    model_inventory.json
    dataset_inventory.json
    gate_matrix.csv
    final_verdict.json
```

阶段目录只追加新 run，不覆盖旧证据。正式 summary 只能引用已完成 run 的 manifest 和 hash，不能手工重录指标作为第二来源真相。

### 20. 执行检查表

- [x] COMSOL 6.3 会话与四项许可证只读预检
- [x] 现有 P0 模型树只读检查
- [x] COMSOL 6.3 流动、组分输运与流动声学官方接口核验
- [x] 本地代码、schema、DL `aux_target_arrays` 与现有审计检查
- [x] 外部通风 CFD 与 CFD-DL 资料检索
- [x] G0：工程参数 registry 与坐标系冻结（engineering_scenario；正式阻断项已列出）
- [x] G0：新 schema、ID 与目录评审（`tunnel-ventilation-comsol-1`；测试 13 passed）
- [x] G1：参数化 CFD smoke（`tv3_tunnel_transport_g0.mph`，`TurbulentFlowkeps`，命名选择非空）
- [x] G1：三档网格与质量守恒（imbalance ≤0.15%；medium→fine mdot 相对变化 ≈0.006% ≤5%）
- [ ] G2：三组分瞬态输运
- [ ] G2：质量分数/干基摩尔分数对齐
- [ ] G3：探针、声路和区域投影
- [ ] G3：现有 Python 传感器正演 parity
- [ ] G3a：NDIR `rbam` 光路与吸收系数审计
- [ ] G3a：TCS 热-电耦合与物性敏感性审计
- [ ] G3a：NDIR/TCS 与 Python P1/P2 通道对齐
- [ ] G4：双向流动声学与 Mapping study
- [ ] G5：benchmark 打包、split 与 integrity
- [ ] G6：E1–E6 多 seed 正式实验
- [ ] G7：真实测量校准与现场 OOD

### 21. 执行进度记录

| 日期         | 阶段  | verdict / 结果                                                                                 | 关键产物                                                                                                              |
| ---------- | --- | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 2026-07-20 | G0  | formal=`g0_input_blocked`；`smoke_allowed=true`（engineering_scenario）                         | `outputs/runs/tv3_comsol_multiphysics/g0_registry/`；`tv3/sim/comsol/`；契约测试 13 passed                              |
| 2026-07-20 | G1  | `g1_cfd_smoke_passed`；physint=`TurbulentFlowkeps`；imbalance ≤0.15%；medium→fine mdot Δ≈0.006% | `COMSOL/tunnel_transport/models/tv3_tunnel_transport_g0.mph`；`outputs/runs/tv3_comsol_multiphysics/g1_cfd_smoke/` |
| —          | G2  | 未开始                                                                                          | 三组分瞬态输运 + 摩尔/质量分数对齐                                                                                               |

文档同步（同日）：记忆库、`docs/README.md`、`active/README.md`、`COMSOL/README.md`、`COMSOL/tunnel_transport/README.md`、`outputs/README.md`、`methods/tv3_名词与实验顺序导读.md`、统一研究路线。

## 参考证据

### 项目内

- `COMSOL/README.md`：当前 P0 数字孪生边界与 acceptance 入口。
- `COMSOL/tv3_comsol_params.md`：P0 参数与 Python 对齐。
- `docs/deep_research/仿真链路与多模态融合审计_20260718.md`：空间输运、器件链和融合缺口。
- `docs/掘进通风_统一研究与实施路线.md`：当前基线、硬门和暂停路线。
- `docs/active/tv3_static_air_feasibility_implementation_plan.md`：静止空气 P0 的执行边界。

### COMSOL 6.3 官方资料

- `com.comsol.help.cfd/cfd_ug_chemsptrans.11.081.html`：SST 与 Concentrated Species 耦合。
- `com.comsol.help.aco/aco_ug_ultrasound.10.02.html`：Convected Wave Equation, Time Explicit。
- `com.comsol.help.aco/aco_ug_ultrasound.10.04.html`：空间变化背景流与 CFD 数据输入。
- `com.comsol.help.aco/aco_ug_multiphysics_couplings.14.17.html`：Background Fluid Flow Coupling 与 Mapping study。
- `com.comsol.help.models.cfd.displacement_ventilation/displacement_ventilation.html`：通风射流、湍流和实验对照示例。
- `com.comsol.help.heat/heat_ug_interfaces.08.65.html`：Radiative Beam in Absorbing Media 与 Beer–Lambert 衰减。
- `com.comsol.help.heat/heat_ug_theory.07.071.html`：Radiative Beam in Absorbing Media 理论和吸收系数。
- `com.comsol.help.acdc/acdc_introduction.02.02.html`：Electric Currents、Joule Heating 和热耦合。
- `com.comsol.help.heat/heat_ug_multiphysics_interfaces.11.01.html`：Heat Transfer Module 多物理场接口与 Joule Heating 入口。
- `com.comsol.help.heat/heat_ug_theory.07.013.html`：湿空气混合物密度、热容和热导率属性。

### 外部资料

- Chang X, et al. *Study on tunnel ventilation and pollutant diffusion mechanism during construction period*. PLOS ONE, 2025. <https://doi.org/10.1371/journal.pone.0322984>
- Toraño J, et al. *Auxiliary ventilation in mining roadways driven with roadheaders: Validated CFD modelling of dust behaviour*. Tunnelling and Underground Space Technology, 2011. <https://doi.org/10.1016/j.tust.2010.07.005>
- COMSOL. *CFD Modelling of Urban Road Tunnels*. <https://www.comsol.com/paper/cfd-modelling-of-urban-road-tunnels-122671>
- D'Aquilio A, et al. *A surrogate CFD model using Machine Learning for fast design explorations of the indoor environment*. Building Simulation 2023. <https://doi.org/10.26868/25222708.2023.1590>
