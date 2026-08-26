# P2 专用 bench 规格书

> 本文件当前包含 P2-03 至 P2-12 已交付的工程身份、科学问题、参数与候选体系、纯前向筛选、S1 3 × 3 刻度、S2 模态互补、S3 独立开关、S4 精度/效率协议、数据契约、S5 来源/偏差接口、S6 工程冻结纪律和 P3 G3-4 晋级矩阵。H1 已授权 C2/C5 的门值与条件式候选，H2 已批准命名空间、主键和所有权；S5 采用 `controlled_synthetic` profile，只支持同条件相对算法比较，不作目标硬件保真声明。本文件不授权在 P2 生成数据、训练模型或宣称候选进入 P4。
> 依据：[根 README](../../README.md)、[根 AGENTS](../../AGENTS.md)、[P2 能力复用审计](./P2能力复用审计.md)。

## 1. 目的、科学问题与边界

### 1.1 科学问题

本 bench 面向低风险动态四组分混合气，研究在目标组分组成与温度、压力、湿度、声程、传感器状态及交叉敏感共同变化时，多模态观测能否稳定区分目标组分，以及信息总量和 target–nuisance 可分离性如何分别影响误差与求解难度。

统一观测模型固定为：

```text
y = F(theta, eta; design) + epsilon + delta
```

- `theta` 只包含目标组分的物理组成，不包含温度、压力、湿度、传感器状态或任何由真值构造的 oracle 特征。
- `eta` 只包含可影响观测但不是预测目标的 nuisance；其最小字段在第 3 节登记。
- `design` 记录模态开关、噪声 profile、采样窗口、组分工作点和 headroom 开关；这些字段由后续任务冻结，不改变主键语义。
- `epsilon` 是观测噪声；`delta` 是预留的 model discrepancy 接口。P2 纯前向筛选默认 `delta = 0`，不以自注入偏差证明算法有效。

### 1.2 阶段边界

P2-04 只完成可执行的参数、组分和观测契约登记。具体物理常数、噪声频响、Fisher/CRB 数值和候选通过 verdict 必须由 P2-05 及后续任务产生；本任务不生成 pilot 或正式数据，不训练模型，不声称任何算法优势。

## 2. 工程身份、命名空间与所有权

### 2.1 命名空间冻结

| 项目 | 冻结值 | 约束与说明 |
|---|---|---|
| 子工程目录 | `gas_information_bench/` | 与 `hydrogen_ng/`、`syngas/`、`tunnel_ventilation/`、`rcdw_mgda/` 平级；H2 已通过。 |
| Python 包 | `gib` | 只承载本 bench 的代码；不得作为旧包的兼容别名。 |
| distribution | `gas-information-bench` | 独立 `pyproject.toml` 的 distribution 名。 |
| schema version | `gib-benchmark-1` | 新 schema 命名空间；不复用任何历史场景 schema。 |
| `mixture_id` 前缀 | `GIB-M` | 组成主键与 split group；示例值为 `GIB-M000001`；同一组成允许对应多个观测实例。 |
| `sequence_id` 前缀 | `GIB-Q` | 观测实例键；示例值为 `GIB-Q000001`；不得与 `mixture_id` 互相回退、别名化或重写。 |
| CLI 前缀 | `gib` | 命令名必须与独立 distribution 和包所有权一致。 |
| 输出根目录 | `gas_information_bench/outputs/` | 只由新子工程写入；历史 `outputs/` 不作为运行时依赖。 |

### 2.2 主键与 split 不变量

1. `mixture_id` 表示组成实例，是 manifest 的 `primary_key` 和 split 的 `split_group_field`。
2. `sequence_id` 表示一次观测实例，是 manifest 的 `instance_key`；多个 `sequence_id` 可以属于同一 `mixture_id`。
3. split validation 必须以 `mixture_id` 检查 train/val/test 互斥；同一组成不能因不同噪声、时间窗或模态 profile 分裂到不同 group。
4. 任何字段缺失都必须显式失败；不得将 `sequence_id` 回退为 `mixture_id`，也不得反向回退。
5. 新 schema 不引入历史噪声种子主键或兼容别名；主键、实例键和 split group 的含义只能在本章节定义一次。

### 2.3 所有权边界

`gas_information_bench/` 是新 bench 的单一事实所有者，至少拥有以下目录：

```text
gas_information_bench/
├── gib/
│   ├── audit/
│   ├── cli.py
│   ├── contract.py
│   ├── freeze.py
│   └── s5_contract.py
├── configs/
├── docs/
├── tests/
├── outputs/
├── pyproject.toml
└── README.md
```

- `gib` 是唯一 Python 包；当前实现及未来保留 owner 均以 §11 registry 为准。
- `gib.audit` 拥有当前 Fisher、有效 CRB、Jacobian 子空间夹角、纯前向候选审计和 3 × 3 刻度计算。
- P3 才实现的 owner 只在 registry 中标记为 `reserved`，P2 不创建空模块冒充可运行能力。
- `configs`、`docs`、`tests` 和 `outputs` 只服务新 bench 的配置、证据和报告，不作为历史场景事实的第二来源。

新子工程可以提取 P2-01 标记为“可直接迁移”的纯函数行为，但必须复制到 `gib` 的唯一所有者并删除旧命名空间；对标记为“需解耦”或“不得复用”的来源不得建立运行时 import。新子工程对 `hg`、`sg`、`tv3`、`rcdw` 私有包的 import 必须为零。

### 2.4 CLI 与输出布局

CLI 前缀为 `gib`，所有正式入口都从 `gas_information_bench/` 的独立安装环境解析配置、输入和输出。输出根目录如下：

```text
gas_information_bench/outputs/
├── runs/
│   └── attempts/
├── summary/
├── reports/
└── archive/
    └── freezes/
```

- `runs/attempts/` 保存单次 attempt 和成功运行的输入绑定；
- `summary/` 保存由冻结产物聚合的机器可读汇总；
- `reports/` 保存由脚本生成的正式报告；
- `archive/freezes/` 保存 append-only 正式证据。

具体 freeze 文件布局、输入 hash 集合和 attempts/freezes 物理分离实现已在 §11 冻结；本章节不授权生成数据。

### 2.5 H2 人工批准记录

H2 批准记录如下：

```text
review_gate: H2
proposed_namespace:
  directory: gas_information_bench/
  python_package: gib
  distribution: gas-information-bench
  schema_version: gib-benchmark-1
  mixture_id_prefix: GIB-M
  sequence_id_prefix: GIB-Q
  cli_prefix: gib
  output_root: gas_information_bench/outputs/
reviewer: kelon
review_date: 2026-08-24
review_verdict: approved
review_reason: 批准拟定目录、包名、distribution、schema、ID 前缀、CLI 前缀、输出根目录及单一所有权边界，允许创建 P2-03 空骨架。
```

H2 已批准。本任务只创建与上述身份一致的空骨架；不生成数据、不训练模型、不运行新包测试，不建立正式 CLI 入口或运行产物。

## 3. 目标组分、nuisance、工作域和单位

### 3.1 目标参数 `theta`

首选体系固定登记为 `N2/CO2/O2/Ar`。四个物理组分都属于目标输出，不能把其中一个静默降级为无误差背景值。

| 名称 | 符号 | 单位 | 工作范围 | 先验 / 设计分层 | 可观测性 | 允许进入部署模型 | 来源与边界 |
|---|---|---|---|---|---|---|---|
| 氮气摩尔分数 | `c_N2` | mol/mol | `0–1`，受四组分 simplex 约束 | 有界 simplex 先验；主筛选网格与低浓度分层由 P2-05 冻结 | 联合多模态可观测，需由 rank/CRB 审计确认 | 作为预测目标输出；不得作为真值输入 | 总体规划 §6.2；具体网格是 P2 bench 设计，不是已验证物性 |
| 二氧化碳摩尔分数 | `c_CO2` | mol/mol | `0–1`，受四组分 simplex 约束 | 同上 | NDIR 提供主要光学观测，其他模态提供辅助约束 | 作为预测目标输出；不得作为真值输入 | 总体规划 §6.2；工业调研 §2.1 |
| 氧气摩尔分数 | `c_O2` | mol/mol | `0–1`，受四组分 simplex 约束 | 同上；与 Ar 形成光学近简并候选对 | 由声学/热导和联合模态观测，需由前向审计确认 | 作为预测目标输出；不得作为真值输入 | 总体规划 §6.2；具体灵敏度待 P2-05 |
| 氩气摩尔分数 | `c_Ar` | mol/mol | `0–1`，受四组分 simplex 约束；设置 `0.01–0.05` 低浓度筛选层 | 低浓度分层；精确采样点由 P2-05 冻结 | 声学/热导低浓度观测，需由 CRB 和尺度审计确认 | 作为预测目标输出；不得作为真值输入 | 总体规划 §6.2；低浓度层是 bench 设计要求 |

物理输出满足：

```text
c_N2 + c_CO2 + c_O2 + c_Ar = 1
c_i >= 0
```

Fisher 参数化采用三个自由坐标，例如 `theta_free = (c_N2, c_CO2, c_O2)`，并以 `c_Ar = 1 - c_N2 - c_CO2 - c_O2` 计算约束方向；这不是把 Ar 当作已知无误差背景。报告和数据字段仍保留四个物理组分名称，闭包残差必须进入验证结果。

### 3.2 Nuisance 参数 `eta`

以下是 P2-04 的最小 nuisance 注册表。范围是纯前向 bench 的工作包络，不是器件常数；P2-05 必须用物理来源和负对照检查其可实现性。

| 名称 | 符号 | 单位 | 工作范围 | 先验 | 可观测性 | 允许进入部署模型 | 来源与边界 |
|---|---|---|---|---|---|---|---|
| 温度 | `T` | K | `293.15–333.15` | nominal 附近有界扰动；profile 记录实际值 | T 慢通道直接观测或校准观测 | 仅以部署时可获得的 T 测量进入；不得使用真值 oracle | 总体规划 §6.2 的 T 扰动要求；范围待 P2-05 审计 |
| 压力 | `P` | kPa | `90–110` | nominal 附近有界扰动 | P 慢通道直接观测或校准观测 | 同上 | 总体规划 §6.2；范围待 P2-05 审计 |
| 相对湿度 | `RH` | %RH | `10–90` | 有界扰动，按 profile 采样 | RH 慢通道直接观测或校准观测 | 同上 | 总体规划 §6.2；范围待 P2-05 审计 |
| 声程 | `L` | m | `0.10–1.00` | 设备 profile 固定或窄范围变化 | 由装置标定记录 | 仅允许部署可测/可标定值 | 总体规划 §6.1；具体装置范围待来源登记 |
| 传感器增益 | `gain_m` | 无量纲 | `0.90–1.10` | 每模态独立有界校准误差 | 由校准过程或在线 reference 估计 | 允许使用部署校准参数，不允许使用仿真真值 | P2 bench 设计字段；P2-10 登记来源 |
| 传感器 baseline | `baseline_m` | 归一化信号单位 | `-0.05–0.05` | 每模态独立有界漂移 | 由空白/基线校准观测 | 仅允许真实可测校准量 | P2 bench 设计字段；P2-10 登记来源 |
| 波形 delay | `delay_m` | ms | `-2–2` | 触发抖动 profile | 由同步标记或校准过程观测 | 仅允许部署时可估计的同步量 | 总体规划 §6.1 的触发抖动要求；范围待审计 |
| 交叉敏感系数 | `crosstalk_mn` | 无量纲 | 非对角系数 `-0.10–0.10` | 稀疏、有界；S3 开关可置零 | 由校准或多通道联合拟合估计 | 只有部署链路能测/校准时才允许进入 | 总体规划 §6.3；具体机制待 P2-05/P2-07 |
| 总流量 | `q_flow` | L/min | `0.1–10` | MFC profile 与动态阶跃 | MFC 慢通道直接观测 | 允许以部署 MFC 读数进入 | 工业调研 §2.1；具体设备范围待 P2-10 |

`eta` 的真值可以用于 P2 的 Fisher/CRB 审计和 oracle 隔离，但不得作为部署 dataset loader 的隐式输入。任何部署可用 nuisance 都必须保存其实际测量值、校准来源和缺失状态。

### 3.3 工作域与观测设计

首选工作域是 MFC 动态混气：气瓶经 MFC 进入混气室，经过参考分析仪和多传感器腔体后排气；P2 只冻结这个前向抽象，不启动真实供气。设计字段至少包含：

| 设计字段 | 取值结构 | 作用 |
|---|---|---|
| `mixture_profile` | 四组分 simplex 工作点与低浓度层 | 调节目标组成尺度 |
| `modality_mask` | NDIR、超声 Raw、声学 DSP、热导、慢通道和校准观测的逐项开关 | 调节有效信息总量 |
| `noise_profile` | 各模态噪声与同步抖动 profile | 调节有效信息总量 |
| `nuisance_prior_profile` | `T/P/RH/L` 等先验宽度 | 调节边缘化后的有效信息 |
| `window_profile` | 宽域 / 窄窗及时间窗 | 调节观测覆盖与增量信息 |
| `headroom_switches` | 快通道、非线性 nuisance、交叉敏感的独立开关 | 为 S3 消融预留接口 |

观测层固定区分 Raw 与 DSP：超声压力波形属于 Raw；从同一 Raw 经过明确配置得到的 TOF、衰减、频域或统计量属于 DSP；DSP 不得读取目标真值、oracle nuisance 或未声明的额外通道。

## 4. 候选体系与模态分工

### 4.1 首选与备选登记

| 候选 ID | 组分体系 | 状态 | 四项要求覆盖 | 安全与范围约束 |
|---|---|---|---|---|
| `GIB-C4-LR` | `N2/CO2/O2/Ar` | `selected_for_forward_screen` | CO2 作为光学锚点；N2/O2/Ar 由声学或热导提供非光学约束；O2/Ar 作为光学近简并候选对；Ar 设置低浓度层 | 首选低风险模拟载体；真实硬件仍需独立通风、氧浓度和供气安全审查 |
| `GIB-C4-CH4` | `N2/CO2/O2/CH4` | `backup_only_safety_gate_required` | 仅作为含 CH4 的备选；不用于 P2-04 首选，也不默认进入 P3 | 必须先取得独立可燃气体安全门和物理来源审查；未通过前不得生成数据或实现 |

首选体系的“光学锚点”“近简并”和“仅某模态可见”均是待 P2-05 以白化 Jacobian、联合秩和有效 Fisher 证实的前向假设；P2-04 不把候选登记等同于物理筛选通过。

### 4.2 组分—模态映射

| 目标组分 | NDIR | 超声 Raw | 声学 DSP | 热导 | T/P/RH/flow 慢通道 |
|---|---|---|---|---|---|
| N2 | 弱/近零光学响应假设 | 主要声速、衰减和波形约束 | 从 Raw 派生的可部署约束 | 主要热物性约束 | 提供 nuisance 校正 |
| CO2 | 主要光学吸收约束 | 辅助传播/衰减约束 | 从 Raw 派生的辅助特征 | 辅助热物性约束 | 提供 nuisance 校正 |
| O2 | 弱/近零光学响应假设 | 主要传播约束；与 Ar 构成跨模态消歧候选 | 同一 Raw 的 DSP 表达 | 辅助热物性约束 | 提供 nuisance 校正 |
| Ar | 弱/近零光学响应假设 | 低浓度下的传播约束 | 同一 Raw 的 DSP 表达 | 低浓度下的热物性约束 | 提供 nuisance 校正 |

每个模态必须对应 `F(theta, eta; design)` 的显式观测块：`y_ndir`、`y_us_raw`、`y_ac_dsp`、`y_tc`、`y_slow=(T,P,RH,q_flow)` 和独立的 `y_calibration=(L,gain,baseline,delay,crosstalk)`。校准观测必须来自部署时可获得的装置记录、空白测量、同步标记或多通道校准，不得使用仿真真值；同一 Raw 与 DSP 不得被当作两个独立物理传感器重复计数。

### 4.3 P2-04 验收边界

- 目标组分与 nuisance 无重名；四个物理组分均保留在输出 schema 中。
- `c_Ar` 的低浓度层用于检验尺度与加权问题，不能由闭包约束直接推算后当作无误差标签。
- NDIR、超声 Raw、声学 DSP、热导、T/P/RH/flow 慢通道和独立校准观测均已有前向映射；是否形成足够互补性留给 P2-05/P2-07。
- 含 CH4 体系只保留为需要独立安全门的备选，不改变首选体系。

### 4.4 P2-05 纯前向候选筛选

P2-05 的实现位于 [`gib.audit.forward`](../../gas_information_bench/gib/audit/forward.py)，只包含确定性前向函数、有限差分 Jacobian、白化、Schur complement、CRB、主夹角和负对照；没有数据生成器、训练代码或历史场景 import。

| 审计对象 | 实现约束 | 来源边界 |
|---|---|---|
| 白化 Jacobian | `J_theta`、`J_eta` 均按声明的观测噪声逐行白化；联合秩使用 prior-scaled nuisance 列，避免 `delay_s` 单位主导 SVD | `hydrogen_ng/docs/物理模型严格化实施计划.md` §2 的理想气体声速公式；纯函数在新包内重写 |
| 有效信息 | `F_eff = F_tt - F_tη(F_ηη + Λ_prior)^-1F_ηt`；报告的 CRB 不使用未边缘化的 target-only Fisher 代替 | P2-01 审计中的 Schur complement 不变量 |
| 观测 profile | NDIR、超声 Raw 或 DSP（二者互斥）、热导、T/P/RH/flow 慢通道及独立校准观测；噪声 profile 与硬件资料显式登记 | `hydrogen_ng/docs/references/传感器硬件资料整理.md` §2–§8 |
| 结果字段 | 分模态白化灵敏度、nuisance-adjusted effective information share、六组分对相似度、联合秩和条件数 | P2-05 任务卡；O2/Ar 常数仍需 P2-10 正式来源 registry |

| 候选 ID | 纯前向 verdict | 负对照 | P2 决策 |
|---|---|---|---|
| `GIB-C4-LR` | `candidate_selected` | 单组分扰动、总量缩放、模态关闭、噪声单调性、重复运行均通过 | 进入 P2-06；仍不是 P3 算法晋级 |
| `GIB-C4-CH4` | 数学筛选 `candidate_selected` | 同上 | 仅保留为 `safety_gate_required` 备选，不改变首选体系 |

首选体系未出现 S2 前向失败，因此没有用增加模型复杂度掩盖物理缺口。P2-05 的候选 verdict 只表示通过纯前向负对照，可进入 S1 刻度；真实气体安全、物理来源完整性和 P3 资格仍由后续任务独立判断。

## 5. S1 信息量三档与 Jacobian 夹角三档

### 5.1 目标误差容许值与信息档

四个物理组分的容许误差先冻结为以下 `tau_j`。它们是 bench 的验收尺度，不是器件精度的替代声明；CO2 的硬件参考来自 TraceGas-NDIR-CO2 profile，其余尺度服务于四组分闭包和低浓度 Ar 分层，后续来源 registry 不能静默改写。

| 组分 | `tau_j`（mol/mol） | 来源 |
|---|---:|---|
| N2 | 0.08 | P2-04 首选体系与 MFC 标签误差的 bench screening tolerance |
| CO2 | 0.03 | P2-04 首选体系；TraceGas-NDIR-CO2 硬件 profile |
| O2 | 0.10 | P2-04 低风险四组分 bench contract |
| Ar | 0.05 | P2-04 低浓度筛选层的 bench tolerance |

物理四组分 CRB 由三个自由坐标的 `crb` 通过闭包变换 `D = [I; -1 -1 -1]` 得到；信息比定义为：

```text
r_j = CRB_P90_j / tau_j
information_ratio = max_j(r_j)
```

信息档门值冻结为：

| 信息档 | 判定 |
|---|---|
| `sufficient` | `max(r_j) <= 0.5` |
| `critical` | `0.8 <= max(r_j) <= 1.2` |
| `insufficient` | `max(r_j) >= 2.0` |

`0.5–0.8` 与 `1.2–2.0` 是未分类间隔，不得被标成任何信息档。P2-06 没有看到 pilot 结果，不修改计划默认门值。

### 5.2 共线档与正交旋钮

共线性使用白化后的 `J_theta` 与 `J_eta` 子空间最小主夹角；目标中心和容许偏差冻结为：

| 共线档 | 目标中心 | 容许偏差 |
|---|---:|---:|
| `high_collinearity` | 10° | ±5° |
| `medium_collinearity` | 45° | ±5° |
| `low_collinearity` | 80° | ±5° |

信息轴只改变 `noise_scale`（同一全模态集合），共线轴只改变已登记 nuisance coupling 的 `coupling_strength`；不使用同一个无解释噪声倍数同时冒充两个轴。`grid.py` 对 coupling strength 做确定性可达性搜索，若任一目标角超出 ±5° 直接失败，不留空格。

### 5.3 3 × 3 配置与生成证据

9 个配置的实际 Fisher、四组分 CRB P90、主夹角、条件数和 `accessible` 字段由 [`p2_s1_grid.json`](../../gas_information_bench/configs/p2_s1_grid.json) 生成，源代码为 [`gib.audit.grid`](../../gas_information_bench/gib/audit/grid.py)。[`S1 3 × 3 冻结表`](./generated/s1_grid_table.md) 由该 JSON 自动复算 Fisher trace、CRB P90、实际夹角和条件数，完整矩阵仍只以 JSON 为事实源。

| 配置 ID | 信息档 | 共线档 | 生成证据字段 |
|---|---|---|---|
| `GIB-S1-SUF-HIG` | sufficient | high_collinearity | `cells[0]`: `effective_fisher`、`crb_p90`、`actual_angle_deg`、`condition_number`、`accessible` |
| `GIB-S1-SUF-MED` | sufficient | medium_collinearity | `cells[1]`: 同上 |
| `GIB-S1-SUF-LOW` | sufficient | low_collinearity | `cells[2]`: 同上 |
| `GIB-S1-CRI-HIG` | critical | high_collinearity | `cells[3]`: 同上 |
| `GIB-S1-CRI-MED` | critical | medium_collinearity | `cells[4]`: 同上 |
| `GIB-S1-CRI-LOW` | critical | low_collinearity | `cells[5]`: 同上 |
| `GIB-S1-INS-HIG` | insufficient | high_collinearity | `cells[6]`: 同上 |
| `GIB-S1-INS-MED` | insufficient | medium_collinearity | `cells[7]`: 同上 |
| `GIB-S1-INS-LOW` | insufficient | low_collinearity | `cells[8]`: 同上 |

`test_grid.py` 强制检查：9 格全部可达；信息档比值分离且单调；角度全部在目标容差内；固定信息档改变角度时只改变 coupling strength；同一信息档内的角度变化不会造成超过 2 倍的信息尺度漂移。任何失败都返回 `redesign_required`，不把空格交给 P3。

## 6. S2 四组分模态分工

### 6.1 互补性对象与冻结阈值

P2-07 只审计首选低风险体系 `GIB-C4-LR = N2/CO2/O2/Ar`。四项 S2 要求均绑定到具体观测行、组分或组分对，并使用白化 Jacobian 灵敏度、白化灵敏度余弦或已冻结 S1 的 CRB 比值；不使用波形图肉眼判断：

| S2 要求 | 具体对象 | 量化指标 | 冻结阈值 | 解释边界 |
|---|---|---|---:|---|
| 光学主可见 | NDIR 的 `ndir_co2` 行与 CO2 | `abs(whitened row · CO2 direction)` | `>= 10.0` | 证明 CO2 有直接光学响应；不声称 NDIR 单模态能独立消歧 CO2/Ar |
| 声学或热导主可见 | 热导 `thermal_primary` 行与 Ar | Ar 灵敏度 / 其他候选组分最大灵敏度 | `>= 1.5` | 证明低浓度 Ar 有非光学主约束 |
| 单模态近简并 | NDIR 单模态中的 CO2/Ar | 白化目标方向绝对余弦 | `>= 0.95` | 近简并是待消歧对象，不是通过条件 |
| 跨模态消歧 | NDIR + 超声 Raw + 热导中的 CO2/Ar | 同一对的白化目标方向绝对余弦 | `<= 0.50` | 只有跨模态余弦低于阈值才证明消歧有效 |
| 低浓度目标 | `c_Ar` 与 `GIB-S1-CRI-LOW` | `0.01 <= c_Ar <= 0.05`、全 profile 白化灵敏度非零、`CRB_P90_Ar / tau_Ar` | `> 0`、`<= 1.2` | Ar 仍是物理目标；不得由闭包约束当作无误差背景 |

光学近简并和跨模态消歧使用同一 CO2/Ar 对，避免把单模态可见性误写成可识别性。低浓度证据复用已经冻结的 S1 单元 `GIB-S1-CRI-LOW`，不另造信息档或修改 P2-06 门值。

### 6.2 生成证据与字段所有者

S2 的运行数值只能由 [`gib.audit.s2_s3`](../../gas_information_bench/gib/audit/s2_s3.py) 从纯前向函数和 S1 冻结产物生成；[`p2_s2_s3_frozen_evidence.json`](../../gas_information_bench/configs/p2_s2_s3_frozen_evidence.json) 是本任务的数值证据文件，规格书不手工复制其中数值：

| 证据 | 生成字段 |
|---|---|
| 光学主可见 | `s2.optical_primary` |
| 声学或热导主可见 | `s2.acoustic_or_thermal_primary` |
| 单模态近简并 | `s2.single_modality_near_degeneracy` |
| 跨模态消歧 | `s2.cross_modal_disambiguation` |
| 低浓度目标 | `s2.low_concentration_target`，引用 `cell_id=GIB-S1-CRI-LOW` |

上述五项全部通过时，C4 的 P2-07 前置 verdict 写为 `eligible_for_P3_test`；该 verdict 只表示具备进入后续 P3 前置测试的物理互补性，不代表 C4 算法晋级、G3-4 占位或 P2 已关闭。H1 已于 2026-08-25 授权，C4 已按该前置进入 §12 的 active 最小证伪行。

### 6.3 S2 失败路径

任一灵敏度、余弦或 CRB 阈值失败，都返回 `redesign_required`，并回到 P2-04/P2-05 修改组分、模态或前向物理设计。不得通过增大模型容量、添加 oracle 特征或调整 P1 门值补救；失败机制必须保留在任务记录中。

## 7. S3 headroom 来源与独立开关

### 7.1 开关注册表

三个开关属于 `AuditConfig` 的显式设计字段，默认均为 `true`。P2-07 使用 [`p2_s2_s3_audit.json`](../../gas_information_bench/configs/p2_s2_s3_audit.json) 冻结开关 profile 和 probe nuisance：

| 开关 | `on` | `off` | 唯一差异 | 预期变化 | 不应变化的量 |
|---|---|---|---|---|---|
| `fast_waveform` | 选择同一 Raw 源的 `acoustic_raw` 视图 | 将同一声明的 Raw 源解析为互斥的 `acoustic_dsp` 视图 | 只改变 Raw/DSP 观测视图，不同时计数 | Raw 与 DSP 标签集合按证据文件切换 | NDIR、热导和慢通道逐元素不变 |
| `nonlinear_nuisance_coupling` | T/P/RH/L 环境因子保留乘性交互 | 使用相同一阶 nuisance 偏导的加性展开 | 只改变环境 nuisance 的高阶交互 | `ndir_co2`、`ndir_null`、`us_amplitude_raw` 改变 | TOF、phase、speed、热导和慢通道不变 |
| `cross_sensitivity` | 保留显式 `crosstalk` 交叉项 | 将显式交叉贡献置零 | 只改变传感器交叉敏感项 | NDIR 两行、超声 amplitude/phase、`thermal_primary` 改变 | TOF、speed、`thermal_auxiliary` 和慢通道不变 |

`nonlinear_nuisance_coupling=off` 不会删除 T/P/RH/L 的一阶 nuisance 或慢通道；`cross_sensitivity=off` 也不会删除 crosstalk 的实测字段，只关闭其进入前向观测的交叉项。`fast_waveform` 是同一 Raw 源的视图选择，不是把 Raw 和 DSP 当成两个独立传感器。

### 7.2 单独关闭、负对照与可见失败

对每个开关分别建立 `on`/`off` 配置；两个配置的 `config_delta_fields` 必须恰好只有该开关，实际改变的观测标签必须等于预注册集合，不应变化的标签必须逐元素相等。三个单独关闭结果由 `s3.switches` 保存；三个同时关闭只出现在 `s3.all_off_negative_control`，字段 `negative_control_only=true`，不得作为目标档位或 P3 profile。

验收入口为：

```powershell
python -m gib.audit.s2_s3
python -m pytest -q tests/test_s2_s3.py
```

测试失败时保留异常和实际标签差异，不吞错、不将空差异解释为成功。P2-07 只有在 S2 与三个独立开关均通过时才取 `pass`；否则取 `redesign_required`。

## 8. S4 CRB、oracle、测量级、强基线和效率协议

### 8.1 协议边界与授权状态

P2-08 只冻结“如何测量、如何配对、如何报告和如何判定字段”，不运行正式模型比较。完整 registry 位于 [`p2_s4_metric_registry.json`](../../gas_information_bench/configs/p2_s4_metric_registry.json)，由新子工程 `configs/` 单一所有者维护。

H1 已于 2026-08-25 授权“精度非劣前提下效率优越”的联合 endpoint。候选相对配对强基线的分组分 P90 绝对差之 95% CI 上界须分别不超过 `N2=0.008`、`CO2=0.003`、`O2=0.010`、`Ar=0.005 mol/mol`；这是 P2-06 `tau_j` 的 10%，四组分必须同时通过。效率侧要求 `iterations/forward_calls` 至少下降 30%，或 `solver_wall_clock/batch-size-1 latency` 至少下降 20%，且其他 primary efficiency metric 回退不超过 5%。P1 原门保留为历史事实，但不再是新 C2 的当前联合 verdict。

### 8.2 四层精度参照

每个 `GIB-S1-*` 信息量 × 夹角单元、每个 split 和 seed 都分别登记四层结果。oracle 与部署结果必须使用不同结果表，不能在同一汇总表中混列：

| 层 | 输入 | 输出 | 部署允许 | 约束 |
|---|---|---|---|---|
| `crb` | 样本级有效 Fisher 与 nuisance 先验 | CRB、CRB P90 | 不适用 | 使用 Schur complement；不把 target-only Fisher 当作已边缘化结果 |
| `oracle` | 真值派生审计特征 | `oracle_results` | 禁止 | 只用于上限/空档审计；oracle 字段不得进入部署 loader |
| `measurement_dsp` | 从 Raw 按声明配置派生的 DSP 特征 | `deployment_results` | 允许 | 绑定 Raw manifest hash、DSP config hash、代码 hash |
| `strong_baseline` | 与候选相同的测量级 DSP 输入 | `deployment_results` | 允许 | 不得获得 oracle 特征或额外真值 nuisance |

精度 registry 的 primary metric 是每个物理组分的绝对误差 P90；RMSE、MAE 和 R² 是 secondary metrics。先报告 `N2/CO2/O2/Ar` 分量，再给宏平均；不得把闭包组分静默删除。原始预测、失败样本、收敛状态和分母都要保留。

### 8.3 强基线和固定容量深度基线

强基线 allowlist 固定为：

| `method_id` | 角色 | 容量/调参纪律 |
|---|---|---|
| `ridge` | 线性强基线 | 预注册配置；只能使用训练组选择参数 |
| `gbdt` | GBDT 强基线 | 固定 GBDT 配置；不能看测试结果调容量 |
| `xgboost_strong_table` | 强表格模型 | 固定 XGBoost 配置；环境缺失时显式失败，不回退到其他模型 |
| `mlp_fixed` | 固定容量深度参照 | `mlp-128-64-relu-v1`，容量锁定，禁止增容 |
| `tcn_fixed` | 固定容量时序参照 | `tcn-32x2-dilation-1-2-4-kernel-5-v1`，容量锁定，禁止增容 |

`mlp_fixed` 和 `tcn_fixed` 不是“看结果后扩大模型”的入口。任何超出 allowlist 的模型、容量或特征层都必须新增授权配置，不得在同一 P3 运行中临时加入。

### 8.4 配对 split、seed 与统计

固定使用 5 个 `mixture_id` group split：`GIB-SPLIT-01` 至 `GIB-SPLIT-05`，以及 3 个预注册 seed：`101/202/303`。同一 `mixture_id` 是所有候选、四层精度参照和数据效率子集的配对单位；所有方法使用相同组、相同 3 × 3 单元、相同预算和相同硬件。

报告粒度至少包含：

```text
grid_cell_id + split_id + seed + method_id + component
```

置信区间使用 95% paired group bootstrap，重采样单位为 `mixture_id`，固定 10,000 次重采样和 registry 中的 bootstrap seed；不得以样本独立重采样破坏组配对。点估计和 CI 按信息单元分层报告，不把不同单元或不同 OOD 构造压成一个数字。

### 8.5 训练、推理和 solver difficulty 计时

训练耗时拆成 `preprocess`、`fit`、`validation` 三段，使用单调 `time.perf_counter_ns`；记录 `duration_ns`、`repeat_index`、硬件指纹、CPU/GPU、线程数、操作系统、Python/框架/方法包版本和 git commit。固定 30 次独立重复，配对方法在同一 3 × 3 单元、split 和 seed 内采用随机交错顺序；单次计时永远不得形成结论。

推理计时先 warm-up 10 个 batch，再测 30 个 batch，batch size 固定记录 `1/32/256`，报告批量吞吐和单样本延迟；所有方法共享硬件、线程和 batch profile。硬件或线程不一致时该配对无效，不通过删行或插值修复。

solver difficulty 长表键固定为：

```text
sequence_id + method_id + split_id + seed
```

至少记录 `iterations`、`forward_calls`、`convergence`、`condition_number`、`final_residual` 和单调时钟 `runtime_ns`，并在同一 solver 记录中绑定硬件指纹、逻辑 CPU 与 BLAS/OMP/MKL/framework 线程数、操作系统与 Python/NumPy/framework/方法包版本、git commit 和 `repeat_index`。一次 `forward_calls` 是 solver 内一次声明的完整前向模型评估；不收敛行保留并计入失败率，不能静默丢弃。

### 8.6 数据效率与增量信息

数据效率使用嵌套的训练 `mixture_id` group 子集：`10% / 25% / 50% / 75% / 100%`。每个 split 先冻结 group 顺序，再取前缀；不同比例不得重新随机抽样。验证组和测试组在所有比例保持不变，记录训练 group 数、样本数、P90/RMSE 及三段训练耗时。首次达到主精度目标的 `precision_target_point` 冻结为四组分绝对 P90 均不超过 `N2/CO2/O2/Ar = 0.08/0.03/0.10/0.05 mol/mol`；该绝对达标点与候选相对基线的非劣带是两个不同字段，不得混用。

增量信息按新增时间窗、新增采样点和新增模态分别登记：

```text
Delta I_vector = diag(F_eff_after) - diag(F_eff_before)
Delta I = trace(F_eff_after) - trace(F_eff_before)
Delta I / Delta cost
```

`Delta cost` 必须保留原生单位：毫秒时间窗、采样点数或激活模态数；未经授权不得把三种成本用任意权重合成为单一成本。增量信息必须在同一 3 × 3 单元、split 和 seed 内配对。

### 8.7 联合 verdict 与失败可见性

P2-08 只冻结 verdict 计算逻辑，不签发候选通过：

1. `precision_non_inferiority`：所有注册的 primary precision CI 上界均落在授权的非劣带内。
2. `efficiency_superiority`：至少一个注册效率指标清除授权效率门，且其他 primary efficiency metrics 不超过授权的非回退带。
3. `combined_verdict = precision_non_inferiority AND efficiency_superiority`。

授权值缺失时结果只能是 `pending_authorization`，不能写成 `pass`、`基本通过` 或默认零带宽。缺失字段、硬件不一致、split 泄漏、oracle 字段进入部署输入、非收敛和计时异常都必须保留显式失败证据；不得静默丢弃、插值或回退。

P2-08 验收以 [`p2_s4_metric_registry.json`](../../gas_information_bench/configs/p2_s4_metric_registry.json) 的字段直接检查协议完整性；本任务不启动训练、不运行 Ridge/GBDT/XGBoost/MLP/TCN 比较，不生成 pilot。

## 9. 数据契约、manifest、ID 与 split

### 9.1 契约所有者与实体主键

P2-09 的唯一数据契约由新子工程的三份配置和 `gib.contract` 校验模块共同维护：[`p2_data_schema.json`](../../gas_information_bench/configs/p2_data_schema.json)、[`p2_manifest_schema.json`](../../gas_information_bench/configs/p2_manifest_schema.json)、[`p2_split_contract.json`](../../gas_information_bench/configs/p2_split_contract.json) 和 [`contract.py`](../../gas_information_bench/gib/contract.py)。三份配置的 `schema_version` 固定为 `gib-benchmark-1`，当前 `contract_status` 为 `contract_frozen`；本节冻结格式，不创建 pilot 数据或实际 artifact manifest。

`mixture_id` 是组成主键，格式为 `GIB-M-` 加 16 位大写 SHA256 截断摘要；摘要输入只包含 `candidate_id` 和四组分 `composition`。nuisance、档位、模态、单位、噪声、时间窗和校准 profile 均属于观测实例，不得改变 `mixture_id`。`sequence_id` 是实例键，格式为 `GIB-Q-` 加 16 位大写 SHA256 截断摘要；摘要输入必须包含一个已验证的 `mixture_id`、非负 `sequence_index` 和非空 `sequence_profile_id`。样本校验必须重算两个 ID 并与记录值比较，不能只检查格式。

样本记录的顶层字段固定为：

| 字段 | 冻结内容 |
|---|---|
| `mixture_id` | 由 `candidate_id + composition` 唯一重算的组成主键 |
| `sequence_id` | 由 `mixture_id + sequence_index + sequence_profile_id` 唯一重算的观测实例键 |
| `sequence_index`、`sequence_profile_id` | sequence 身份来源字段；必须随样本保存以支持完整性复算 |
| `candidate_id` | `GIB-C4-LR` 候选命名空间 |
| `composition` | `N2/CO2/O2/Ar`，单位 `mol/mol`，非负且和为 1 |
| `nuisance` | `T/P/RH/L/gain_m/baseline_m/delay_m/crosstalk_mn/q_flow` 及单位 |
| `grade` | `grid_id`、`grid_cell_id`、`information_band`、`angle_band`，引用 `GIB-S1-3x3-v1` |
| `modality_profile` | profile ID、启用模态、Raw/DSP 视图；`acoustic_raw` 与 `acoustic_dsp` 不得同时计数 |
| `units` | 组成、温度、压力、湿度、声程、流量、波形和标签单位 |
| `sources` | 每类来源的类型、ID、revision、SHA256 和定位信息 |
| `split_assignment` | `split_id` 与 `train/val/test` partition |

### 9.2 数组层与派生关系

数组层字段及其来源关系固定如下；每个数组 descriptor 必须给出 `file_ref`、`dtype`、符号化 `shape_spec`、单位、存储类型和 `derived_from`。`shape_spec` 必须与冻结 axes 完全一致，`file_ref` 必须是 artifact root 内的相对路径，并在同一 manifest 中以相同 `artifact_type` 登记；实际文件的 SHA256 只在 manifest 中登记。

| 数组层 | 轴/记录 | 来源与部署边界 |
|---|---|---|
| `raw_waveform` | `channel × time` | 实测 Raw；允许部署输入 |
| `slow_channels` | `channel × time`，通道为 `T/P/RH/q_flow` | 实测慢通道；允许部署输入 |
| `calibration_channels` | `calibration_parameter`，通道为 `L/gain_m/baseline_m/delay_m/crosstalk_mn` | 部署可得校准观测；与慢通道分离，允许部署输入 |
| `dsp_features` | `time × feature` | 只能由 `raw_waveform` 按声明配置派生；允许部署输入 |
| `labels` | `component`，顺序 `N2/CO2/O2/Ar` | 标签层；不得进入部署输入 |
| `sample_fisher` | `parameter × parameter` | 审计层；不得进入部署输入 |
| `effective_fisher` | `target_parameter × target_parameter` | nuisance 边缘化后的审计层；不得进入部署输入 |
| `crb` | `component × component` | CRB 审计层；不得进入部署输入 |
| `crb_p90` | `component` | CRB P90 审计层；不得进入部署输入 |
| `principal_angle` | 标量 `minimum_principal_angle_deg` | 由 `sample_fisher` 中白化 `J_theta/J_eta` 的 Gram block 复算；不得错误标记为仅由 `effective_fisher` 派生；不得进入部署输入 |
| `incremental_information` | 增量记录 | 含 `increment_type`、`delta_I_vector`、`delta_I_trace`、`delta_cost`、`delta_I_per_delta_cost`；不得进入部署输入 |

`dsp_provenance` 必须同时保存 `source_raw_manifest_id`、Raw manifest SHA256、DSP 配置 SHA256、代码 SHA256 和 `derived_from=raw_waveform`。样本校验必须接收 manifest 与三个外部期望 hash，禁止用记录自身的值与自身比较；manifest ID 或任一 hash 不匹配都必须显式失败，不得静默复用缓存。oracle 特征、真值 nuisance、真值派生特征和 `oracle_results` 属于独立 audit layer，禁止出现在 deployment dataset loader 的允许字段中。

### 9.3 solver difficulty 长表

长表主键固定为：

```text
sequence_id + method_id + split_id + seed
```

除上述键外，字段至少包含 `iterations`、`forward_calls`、`convergence`、`condition_number`、`final_residual`、单调时钟 `runtime_ns`、硬件指纹、逻辑 CPU、BLAS/OMP/MKL/framework 线程数、操作系统、Python/NumPy/framework/方法包版本、git commit 和 `repeat_index`。`forward_calls` 的定义沿用 S4；不收敛行保留，不得用删除、插值或默认值隐藏失败。

### 9.4 manifest 与 split validation

manifest 必须声明 `primary_key=mixture_id`、`instance_key=sequence_id`、`split_group_field=mixture_id`、schema version、ID namespace、source snapshots，以及每个 artifact 文件的相对路径、artifact type、schema version 和 SHA256。manifest schema 只冻结格式，不代表已有数据文件。

split 固定为 `GIB-SPLIT-01` 至 `GIB-SPLIT-05`，seed 固定为 `101/202/303`。每个 split 必须包含 `train`、`val`、`test` 三个 partition；以 `mixture_id` 为 group 验证，同一 split 内一个 `mixture_id` 只能属于一个 partition，但允许在该 partition 内对应多个 `sequence_id`；同一 split 内一个 `sequence_id` 只能出现一次并映射到一个 `mixture_id`。不同 split 之间允许同一 `mixture_id` 重新参与分组，以支持固定的多 split 评估；不得引入历史噪声种子字段作为分组键。

P2-09 的测试入口固定为新子工程根目录：

```powershell
python -m pytest -q
rg -n "base_condition_id|noise_seed_index|noise_seed" .
rg -n "from (hg|sg|tv3|rcdw)|import (hg|sg|tv3|rcdw)" .
```

测试覆盖 JSON 契约、ID 确定性、manifest hash 必填、DSP provenance hash mismatch、solver difficulty 必填、oracle/deployment 字段隔离和 group split 泄漏拒绝；不生成 pilot 数据。

## 10. S5 物理来源 registry 与 model discrepancy 接口

P2-10 的来源事实由 [`p2_s5_source_registry.json`](../../gas_information_bench/configs/p2_s5_source_registry.json) 维护，接口格式由 [`p2_s5_discrepancy_contract.json`](../../gas_information_bench/configs/p2_s5_discrepancy_contract.json) 和 [`s5_contract.py`](../../gas_information_bench/gib/s5_contract.py) 共同维护。当前 verdict 为 `source_complete`，其精确定义是：进入合成前向方程的量均有明确值、单位、类别和代码绑定，不进入模型的现实硬件字段明确为 `null/not_modeled`。该 verdict 不等于目标器件物理验证，也不单独授权在 P2 生成数据或训练。

### 10.1 来源分类与前向绑定

registry 只允许三类来源：`peer_reviewed`、`device_manual` 和 `engineering_assumption`。`engineering_assumption` 必须保留非验证状态；P2-05 的筛选 proxy、数值微分步长、通道噪声标准差、NDIR 光学系数和耦合系数均登记为假设，不能写成器件或文献已验证值。每个进入前向方程的数量都按 `forward_inventory` 登记，并通过代码 hash、行号和来源 locator 绑定到 `forward.py`。

| inventory_id | 进入方程或审计的位置 | 当前来源状态 |
|---|---|---|
| `IDEAL_GAS_CONSTANT_R`、`MOLAR_MASS`、`CP_MOLAR` | 混合物质量、`gamma_mix` 与理想气体声速 | Ar 已绑定 IUPAC 原子量技术报告和气态 Ar 热容论文；`39.948` 明确为参考气名义组成，不冒充任意天然样品精确值 |
| `THERMAL_CONDUCTIVITY` | 热导率混合与温度修正 | Ar 已由两份气态 Ar 同行评议参考数据绑定至环境温区稀薄气体 screening 基准 |
| `ATTENUATION_WEIGHT`、`PHASE_WEIGHT` | 声学衰减、振幅和相位 proxy | P2-05 engineering assumption，未验证 |
| `ETA_DEFAULT`、`ETA_PRIOR_STD`、`ETA_STEPS` | nuisance 中心、先验和有限差分 | 设计/数值假设，未作为实测标定 |
| `NORMALIZATION_CONSTANTS` | 参考温压、声程、声速、热导率及耦合系数 | P2-05 screening constants，未验证 |
| `NDIR_ABSORPTION_COEFFICIENTS` | CO2/CH4 吸收和 null 通道 | 授权的无量纲 Beer–Lambert proxy；所有方法共享，不表示 TraceGas 标定曲线 |
| `OBSERVATION_NOISE_STD` | 各模态白化噪声尺度 | 授权的逐通道 synthetic noise；所有配对方法共享同一数组和 `noise_scale` |
| `MODALITY_FREQUENCY_RESPONSE` | 具名硬件参考与模型表达边界 | 器件资料仅作 P5 参考；P2 光学/声学 proxy 不表达滤光片曲线或激励频响，对应字段显式 `null/not_modeled` |
| `OPERATING_RANGES` | P2 bench nuisance envelope 与硬件范围 | 设计工作域和硬件摘要分开登记，不把设计域当硬件验证 |

2026-08-25 的来源检索和合成 profile 边界见 [`S5来源检索记录.md`](./S5来源检索记录.md)。现实硬件仍缺目标 TraceGas filter/calibration、装配态声学频响和完整逐通道实测噪声，但这些量不进入当前合成模型，因此不再是 P2/P3 相对算法比较的必填字段。不得用邻近器件数字填补它们；若 P5 要声明绝对设备性能，必须按来源记录中的采集流程重新标定。

### 10.2 discrepancy 接口

接口签名冻结为：

```text
delta(observation, condition, modality, discrepancy_profile)
```

`discrepancy_profile` 必须显式传入或使用默认值 `off`。`off` 在元素级返回 observation 的相同数值副本，不改变输入；P2 不自注入偏差。`p5_reserved` 仅保留给 P5 sim-to-real 阶段，在 P2 请求时必须显式失败；未知 profile、缺失 condition 字段和不支持的单位转换同样显式失败。该接口不改动 P2-05 `forward.py`，也不生成 bias 数据。

condition 的冻结字段为 `T_K`、`P_kPa`、`RH_frac`、`L_m`、`gain`、`baseline`、`delay_s`、`crosstalk` 和 `q_flow`；modality 为 `ndir`、`acoustic_raw`、`acoustic_dsp`、`thermal`、`slow`、`calibration`。单位转换表只登记 C/K、kPa/Pa、ms/s、mol/mol/% 和 mW/(m*K)/W/(m*K) 的显式换算，并由纯接口测试验证。

S5 验收入口固定为新子工程根目录：

```powershell
python -m pytest -q tests/test_s5_contract.py
python -m pytest -q
```

验收要求为 `source_complete`、所有工程假设保持“非硬件验证”状态、现实硬件声明被禁用、配对方法强制共享一个 profile、单位换算测试通过，以及 `off` 输出元素级不变且不修改输入；本任务不运行前向比较、数据生成、训练或部署评估。

## 11. S6 工程归属与冻结纪律

S6 的机器可读事实源是
[`p2_s6_ownership_registry.json`](../../gas_information_bench/configs/p2_s6_ownership_registry.json)，
冻结实现由 [`freeze.py`](../../gas_information_bench/gib/freeze.py) 唯一维护。历史子工程只保留为
P2-01 迁移出处；`gib` 运行时不得导入 `hg`、`sg`、`tv3` 或 `rcdw` 私有包。

### 11.1 单一所有者

P2-01 的全部能力都映射到一个规范 owner。`active` 表示 P2 已存在可验证实现，
`reserved` 只冻结未来实现位置，不表示功能已完成，也不得提供空成功路径。

| P2-01 能力 | 规范 owner | 当前状态 |
|---|---|---|
| ID、主键与索引 | `gib.contract` | `active` |
| schema 与 manifest | `gib.contract` | `active` |
| 数组写入与存储布局 | `gib.sim.packaging.arrays` | `reserved` |
| 通用文件 I/O | `gib.common.io` | `reserved` |
| split 与分组 | `gib.contract` | `active` |
| 质量检查与输入验证 | `gib.contract` | `active` |
| Fisher、CRB 与秩 | `gib.audit.forward` | `active` |
| Jacobian 主夹角 | `gib.audit.grid` | `active` |
| VarPro、求解器与迭代轨迹 | `gib.audit.solver` | `reserved` |
| Raw 与 DSP 派生链 | `gib.pipeline.raw_dsp` | `reserved` |
| append-only freeze 与证据 manifest | `gib.freeze` | `active` |
| 效率与统计协议 | `gib.audit.metrics` | `reserved` |
| 运行输出与报告纪律 | `gib.pipeline.report` | `reserved` |

配置文件可以描述 schema、门值和协议，但不能成为第二份代码 owner。新增能力必须先更新
registry，再在登记的 owner 中实现；禁止为了兼容历史场景另建第二套 ID、split、provenance、
freeze 或报告逻辑。

### 11.2 独立工程入口与输出布局

`gas_information_bench/` 自带 `pyproject.toml`、`gib` CLI、`tests/`、`configs/`、
`docs/` 和 `outputs/` 说明。`configs/` 是唯一配置事实源，同时作为 Python 资源包随 wheel 分发；普通非 editable 安装必须能在源码树外读取契约、运行审计入口和 CLI。

```powershell
pip install .[dev]
python -m pytest -q
gib --help
```

运行尝试固定写入 `outputs/runs/attempts/`，正式冻结固定写入
`outputs/archive/freezes/`；`summary/` 与 `reports/` 只能由冻结证据派生。attempt 与 freeze
目录必须物理分离，不能互为父子目录。

### 11.3 append-only freeze 不变量

只有含 `attempt_manifest.json` 且 `status=complete` 的 attempt 可以冻结。每次冻结必须：

1. 使用新的 `GIB-FREEZE-*` 目录，目标已存在时显式失败；
2. 先写唯一 staging 目录，完整后以目录级原子提升发布；
3. 复制 attempt、输入文件和独立 source snapshots；
4. 在 `evidence_manifest.json` 中登记 `SHA256` 并允许逐文件重算；
5. 至少绑定 `config`、`schema`、`gate`、`code`、`source_registry` 五类输入；
6. 拒绝工作区外输入、符号链接、重复角色归属、未完成 attempt 和任何 hash 不一致。

`gib freeze` 负责创建正式目录，`gib verify-freeze` 负责重算证据。正式 freeze 不回写 live
config，不覆盖历史 freeze，也不把失败 attempt 聚合为成功产物。

S6 验收由 ownership registry 测试、freeze 契约测试、全量子工程测试以及私有跨包 import
扫描共同完成；验收只证明工程边界与冻结纪律，不授权数据生成、训练或候选晋级。

## 12. P3 G3-4 候选晋级矩阵

### 12.1 共用判定口径

C1 只提供信息谱分层、3 × 3 单元和报告框架，不占算法候选行。C3 固定为
`deferred_to_P5`，P3 不实现 discrepancy 学习。所有 active/conditional 行共享 5 个
`mixture_id` group split、3 个 seed、9 个 S1 单元、强基线、H1 硬件 profile 和 30 次配对
计时；oracle 只作上限审计。表中门值缩写如下：

- `NI`：候选减配对对照的分组分 P90 差之 95% CI 上界均不超过
  `N2/CO2/O2/Ar = 0.008/0.003/0.010/0.005 mol/mol`。
- `E30`：iterations 或 forward calls 的相对下降之 95% CI 下界达到 30%。
- `E20`：solver wall-clock 或 batch size 1 latency 的相对下降之 95% CI 下界达到 20%。
- `NR5`：未被选作优势证据的其他 primary cost 相对回退不超过 5%。
- `P90-target`：四组分绝对 P90 均不超过 `0.08/0.03/0.10/0.05 mol/mol`。

### 12.2 候选矩阵

| 行 / 状态 | 机制与廉价前置 | 配对对照 | 所需 3 × 3 单元 | primary / secondary endpoint | 门值 | 负对照 | 停止条件 | 通过后的唯一下一步 |
|---|---|---|---|---|---|---|---|---|
| `C2-S0` / active preflight | 先确认普通 VP 是否存在可测 headroom；只运行已实现 solver，不建立展开网络 | Joint LM、Classical VP、VPLR、TSVD/Ridge-VP | 9/9；逐单元、split、seed 报告 | primary：P90、iterations、forward calls、solver wall-clock、失败率；secondary：RMSE、condition number、final residual | 至少一个 VP 系方法相对 Joint LM 满足 `NI AND (E30 OR E20) AND NR5`，且不得由单一单元/seed 驱动 | 固定预算下打乱信息档标签；普通 VP 与同实现但关闭投影的 Joint LM 配对 | 无方法清除联合门、优势只在一个单元/seed、或非收敛率上升时停止 C2 | 仅解锁 `C2-IC-RDU-VP` 最小实现 |
| `C2-IC-RDU-VP` / conditional | 仅在 `C2-S0` 证明普通 VP headroom 后，实现信息条件化正则、可学习更新与自适应深度；每项可独立关闭 | `C2-S0` 中最强 VP、VPLR、TSVD/Ridge-VP、固定深度展开 | 9/9，必须覆盖三种信息档与三种夹角档 | primary：`NI` 下 iterations/forward calls/wall-clock；secondary：P90/RMSE、深度分布、收敛失败率 | `NI AND (E30 OR E20) AND NR5`；固定深度展开也须用同容量/预算 | 关闭信息条件化、固定正则、固定深度、打乱单元标签 | 任一消融与完整模型等价、收益来自增容、或跨单元门不成立即停止 | 仅提交 P3 C2 G3-4 verdict，不自动进入 P4 |
| `C5-A-CRB-ADAPT` / active | 先用冻结 Fisher/CRB 与增量信息模拟停止策略，不训练网络 | 完整采样、固定短窗口、uncertainty early exit、CRB early exit、CRB + 动态模态 | 9/9；必须包含 critical 与 insufficient 档 | primary：`NI` 下测量时间、采样点数、激活模态数；secondary：`Delta I`、`Delta I / Delta cost`、FLOPs | `NI` 且至少一项原生成本下降 20%，其余已登记成本满足 `NR5` | 随机停止、等长度固定窗口、打乱 CRB 排序 | CRB early exit 不优于固定短窗口、成本记录不闭合、或只在单一单元/seed 有效即停止 | 仅解锁 P3 自适应采样最小实验 |
| `C5-B-FO-MPLSELM` / active | 先在冻结 DSP 特征和嵌套训练组上运行轻量表格基线；不使用 oracle | Ridge、PLS、ELM、PLSELM、XGBoost、轻量 CNN | 9/9；训练比例 `10/25/50/75/100%` 全部配对 | primary：首次达到 `P90-target` 的训练组比例、训练 wall-clock、batch size 1 latency；secondary：learning curve、RMSE、跨噪声稳定性 | `NI` 且首次达标比例不高于 75%，或训练 wall-clock/latency 下降至少 20%；其他 primary cost 满足 `NR5` | 随机正交方向、移除 Fisher 权重、随机非嵌套子集仅作泄漏检查且不参与主结论 | 无样本/计算优势、只在单一比例或 seed 优势、或 PLSELM 基线等价即停止 | 仅提交 P3 C5-B G3-4 verdict |
| `C4-ID-MULTIVIEW` / active | P2-07 已给出 `eligible_for_P3_test`；保持 encoder 容量一致，先做共享/私有可关闭实现 | early fusion、共享/私有分支、同参数量随机分组 | 9/9；重点预注册高共线与 critical/insufficient 单元 | primary：遮蔽或未见 nuisance OOD 的 MAE；secondary：完整模态 R²、共享表示线性探测、模态贡献 | OOD MAE 相对 early fusion 下降至少 10%，且完整模态 R² 回退不超过 0.01；95% paired CI 支持 | 错误配对、模态遮蔽、nuisance OOD、同参数量随机分组 | 随机分组不劣、共享表示不可线性探测、或只改善完整模态时停止 | 仅提交 P3 C4 G3-4 verdict |
| `C5-C-CR-PKD` / conditional | raw teacher 必须先相对 DSP 强基线证明真实信息空档；未通过时不建 student | raw teacher、DSP 强基线、普通 KD、Physics/Fisher KD | 9/9；raw/DSP 同一 Raw manifest 派生并配对 | primary：student 的 `NI` 与 CPU latency/FLOPs/输入带宽；secondary：参数量、表示信息差、遮蔽稳定性 | 激活门：teacher 至少一组分 P90 改善 10%、其余满足 `NI`；student 门：`NI AND E20 AND NR5` | 打乱 teacher logits、关闭 Physics/Fisher 项、等参数普通 KD | teacher 无显著空档、student 无部署成本优势、或使用真值/未部署字段即停止 | 仅提交 P3 C5-C G3-4 verdict |
| `C5-D-FIGS` / conditional | `C2-S0` 必须先证明至少两个 solver 在不同区域分别胜出；第一版只用物理规则路由 | 全域最强单 solver、oracle 区域路由上限、物理规则路由、轻量修正规则 | 9/9；每个被路由区域至少覆盖两个 split 与三个 seed | primary：`NI` 下 solver wall-clock；secondary：错误路由代价、区域胜率、失败率 | 激活门：两个 solver 在不同区域分别清除 `E30` 或 `E20`；路由门：`NI AND E20 AND NR5` | 打乱区域标签、固定最强 solver、oracle 路由仅作上限 | 一个 solver 全域不劣、没有两个区域化赢家、规则路由不降成本即停止 | 仅解锁轻量修正规则路由；不得直接转黑箱 MoE |

矩阵 verdict 为 `matrix_ready`：C2-S0、C5-A、C5-B 与 C4 均已有廉价、可停止且可由字段直接计算的 active 行；C2 主模型、C5-C 和 C5-D 保持条件式，不存在“先做完整模型再看”的路径。

## 13. P3 G3-1 至 G3-5 消费契约

### 13.1 全局不变量

P3 只消费本规格冻结的配置、schema、门值、代码和 source registry hash。所有候选和对照共享 `controlled_synthetic` profile、9 个 S1 单元、5 个 `mixture_id` group split、3 个 seed、预算和硬件指纹；任何方法不得获得私有噪声、不同前向系数或 oracle 字段。现实硬件数据不是 P3 的输入，P3 结论不得外推设备绝对性能。

### 13.2 各门输入、输出与停止路径

| 门 | 输入 | 通过与输出 | 失败与停止路径 |
|---|---|---|---|
| G3-1 | `forward.py`、S2/S3 配置、synthetic source registry | 前向确定性、单组分扰动、总量缩放、模态关闭、噪声单调性全部通过；输出带代码与配置 hash 的 forward audit | 任一负对照失败即停止数据生成，返回 P2-05/P2-07 修正前向或开关 |
| G3-2 | `p2_s1_grid.json`、forward audit | 9/9 单元可达；信息档满足 `<=0.5 / 0.8–1.2 / >=2.0`，夹角达到 `10/45/80 ±5 deg`；输出冻结 grid 与 [生成数值表](./generated/s1_grid_table.md) | 任一格不可达、档位不单调或 JSON 与生成表不一致即停止，返回 P2-06 调整旋钮 |
| G3-3 | 通过 G3-1/G3-2 的 pilot、数据契约、Ridge/GBDT/XGBoost/固定 MLP/TCN 与 oracle | manifest、group split、Raw→DSP provenance 和 oracle 隔离全部通过；在预注册三个 critical 单元中至少 2 格满足 `oracle_r2 - strongest_deployment_baseline_r2 >= 0.05`，或至少一个组分的 `strongest_baseline_p90 - oracle_p90` 超过该组分 `NI` 带；输出冻结 pilot 与 `baseline_sufficient` | 完整性失败先修生成链路；强基线打平 oracle 则判 `no_algorithmic_headroom`，返回 P2-04 至 P2-07 重设 bench，不进入 G3-4 |
| G3-4 | G3-3 冻结 pilot、§12 候选矩阵、S4 联合门 | 逐行执行廉价前置、配对 CI、负对照和停止条件；至少一项候选取得 `进入 P4` 才通过；输出候选机制、适用单元和失败候选清单 | 所有 active 行失败则停止 P4，返回 P1/P2 重新立项；不得临时放宽 NI/效率门或激活 conditional 行 |
| G3-5 | 通过 G3-4 的冻结配置、正式规模生成计划、manifest/schema | 正式数据集覆盖全部 9 格、固定 split/seed、Raw + DSP、样本级 Fisher/CRB、增量信息和 solver difficulty；所有 artifact hash 验证通过后输出 append-only freeze | 生成异常、hash 不符、split 泄漏、profile 漂移或 oracle 泄漏均使该 freeze 失败；修正后创建新 attempt，不覆盖旧产物 |

### 13.3 P2-13 一致性 verdict

S1–S6、字段 schema、指标 registry、候选矩阵和 G3-1 至 G3-5 的 pass/fail 路径均已闭合。规范名称只保留 `mixture_id` 作为组成与 split group 主键、`sequence_id` 作为观测实例键；C1 不占候选行，C3 保持 `deferred_to_P5`，C2/C5 使用 H1 授权门，C4 使用 P2-07 前置。P2-13 verdict 为 `ready_for_review`。

## P2-03 执行记录

```text
task_id: P2-03
input_versions:
  - README.md | workspace_sha256=47D3092AF2FA3C6F3732E27B01107B984FEF4E8D899C5E69E1AABCAEC0358FC7
  - AGENTS.md | workspace_sha256=4E39B21127DC98DFE0AC6CF81D8E3C0E8981A61011966FDAC6580BF502F74540
  - docs/p2/P2能力复用审计.md | workspace_sha256=9C6B27438B28535B8534A6C50CA4CDC7327F2BCDF6173B4CBA97D7AD7B506AFC
changed_files:
  - docs/p2/P2_bench规格书.md
  - gas_information_bench/README.md
  - gas_information_bench/pyproject.toml
  - gas_information_bench/gib/__init__.py
  - gas_information_bench/sim/core/.gitkeep
  - gas_information_bench/sim/packaging/.gitkeep
  - gas_information_bench/sim/validation/.gitkeep
  - gas_information_bench/audit/.gitkeep
  - gas_information_bench/configs/.gitkeep
  - gas_information_bench/docs/.gitkeep
  - gas_information_bench/tests/.gitkeep
  - gas_information_bench/outputs/runs/.gitkeep
  - gas_information_bench/outputs/summary/.gitkeep
  - gas_information_bench/outputs/reports/.gitkeep
  - gas_information_bench/outputs/archive/.gitkeep
  - docs/p2/README.md
commands:
  - Get-FileHash -Algorithm SHA256 -LiteralPath README.md, AGENTS.md, docs/p2/P2能力复用审计.md
  - New-Item -ItemType Directory -Force -Path gas_information_bench/sim/core, gas_information_bench/sim/packaging, gas_information_bench/sim/validation, gas_information_bench/audit, gas_information_bench/configs, gas_information_bench/docs, gas_information_bench/tests, gas_information_bench/outputs/runs, gas_information_bench/outputs/summary, gas_information_bench/outputs/reports, gas_information_bench/outputs/archive
  - rg -n "gas_information_bench|gas-information-bench|gib-benchmark-1|GIB-M|GIB-Q|mixture_id|sequence_id|review_gate|review_verdict" docs/p2/P2_bench规格书.md gas_information_bench
  - rg -n "base_condition_id|noise_seed_index|noise_seed" gas_information_bench
  - rg -n "^\s*(from|import)\s+(hg|sg|tv3|rcdw)(\.|$)" gas_information_bench
  - Test-Path -LiteralPath gas_information_bench, gas_information_bench/README.md, gas_information_bench/pyproject.toml, gas_information_bench/gib/__init__.py, gas_information_bench/sim/core, gas_information_bench/sim/packaging, gas_information_bench/sim/validation, gas_information_bench/audit, gas_information_bench/configs, gas_information_bench/docs, gas_information_bench/tests, gas_information_bench/outputs/runs, gas_information_bench/outputs/summary, gas_information_bench/outputs/reports, gas_information_bench/outputs/archive, docs/p2/P2_bench规格书.md
  - git diff --check -- docs/p2/P2_bench规格书.md gas_information_bench docs/p2/README.md
exit_codes:
  - input hash: 0
  - skeleton creation: 0
  - namespace marker rg: 0
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - skeleton path Test-Path: 0（全部为 True）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/ P2-03 空骨架
artifact_sha256:
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - gas_information_bench/README.md=190F8A84E56C821521B75175723404FFE127F92FF046C110969EDB8F465817FD
  - gas_information_bench/pyproject.toml=90B34254ECB807C545467D8331870EA292CDAC6E492C0B4A3134AA18F01D1F29
  - gas_information_bench/gib/__init__.py=1623E2D8B7AC8174FF50E12392228ABBADA23AE2628A115C31A25D20954824DD
  - gas_information_bench/<eleven .gitkeep files>=E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
failed_checks: []
verdict: approved
next_allowed_task: P2-04
```

## P2-04 执行记录

```text
task_id: P2-04
input_versions:
  - 项目总体规划.md §6.1–§6.3 | workspace_sha256=49E674945540D0A9E077C48A2011D29A6A0C6E1DEF62B3C5B33D72ED062CA7E8
  - docs/工业多传感器气体组分检测_工业应用方向调研报告.md §2.1 | workspace_sha256=2BB8C20E4767E42541B786D1CC13CCDA15906DCAEEFF0F6362F7B9CB224AFA88
  - docs/多传感器多组分气体检测算法创新深度调研报告.md §6 | workspace_sha256=5D7BC2A2F4D9816084BFC892D2C804505620FE13D95CEE7B17975544F6355BF6
  - P2-03 前置 | README status=completed/approved；H2 review_verdict=approved
changed_files:
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - Get-FileHash -Algorithm SHA256 -LiteralPath 项目总体规划.md, docs/工业多传感器气体组分检测_工业应用方向调研报告.md, docs/多传感器多组分气体检测算法创新深度调研报告.md
  - rg -n 'P2-03 \\| completed \\| `approved`|review_verdict: approved' docs/p2/README.md docs/p2/P2_bench规格书.md
  - rg -n 'N2/CO2/O2/Ar|c_N2|c_CO2|c_O2|c_Ar|theta|eta|T|P|RH|声程|gain|baseline|delay|crosstalk|NDIR|超声 Raw|声学 DSP|热导|flow|CH4|闭包|GIB-C4-LR' docs/p2/P2_bench规格书.md
  - rg -n 'y_ndir|y_us_raw|y_ac_dsp|y_tc|y_slow|T/P/RH/q_flow|Raw 与 DSP' docs/p2/P2_bench规格书.md
  - rg -n '草案|提案值|提案前缀|输出根目录提案|待人工填写|在 H2 人工批准前' docs/p2/P2_bench规格书.md
  - rg -n '[ \\t]+$' docs/p2/P2_bench规格书.md
  - git diff --check -- docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - input hash: 0
  - P2-03 prerequisite rg: 0
  - parameter/candidate marker rg: 0
  - modality mapping rg: 0
  - stale approval marker rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - docs/p2/P2_bench规格书.md
artifact_sha256:
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
failed_checks: []
verdict: ready_for_forward_screen
next_allowed_task: P2-05
```

## P2-05 执行记录

```text
task_id: P2-05
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=F33DAC6939845C753220CB484DED99556155707ABA13C436006ADCC298FFE8EA
  - docs/p2/P2_bench规格书.md | P2-04 verdict=ready_for_forward_screen；append-only 文档不对自身内容建立循环哈希
  - hydrogen_ng/docs/references/传感器硬件资料整理.md | workspace_sha256=D1C44BEF8E463809985570BA40E54CE25B2DCA182D41EBB0FC5BCDD66A438AC9
  - tunnel_ventilation/docs/references/传感器硬件资料整理.md | workspace_sha256=5D7FAF30639AB9F5E91324FF7B9B321E1244AB4EEF2044D5232227E7E1C0860E
  - hydrogen_ng/docs/物理模型严格化实施计划.md | workspace_sha256=953887D913D0FF7D9032F0C1D90BFEECA676047D0B206BE0725B56DBA2EE1514
  - syngas/docs/references/co_acoustic_constants.md | workspace_sha256=A2913A114F72DBE9C123F8C2ECD404D4DE9C67CCC5BA129EC55F7B224359304B
changed_files:
  - gas_information_bench/pyproject.toml
  - gas_information_bench/gib/audit/__init__.py
  - gas_information_bench/gib/audit/forward.py
  - gas_information_bench/tests/test_forward_audit.py
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - rg -n -A 75 -B 5 'P2-05|P2-06' docs/p2/P2执行计划.md
  - python -m pytest -q tests/test_forward_audit.py
  - python -c "from gib.audit.forward import AuditConfig, screen_candidate; results={key: screen_candidate(AuditConfig(candidate_id=key)) for key in ('GIB-C4-LR','GIB-C4-CH4')}; print([(key, item['candidate_verdict'], item['result'].joint_rank, all(value['passed'] for value in item['negative_controls'].values()), item['safety_gate_required']) for key, item in results.items()])"
  - rg -n 'base_condition_id|noise_seed_index|noise_seed' gas_information_bench
  - rg -n '^\s*(from|import)\s+(hg|sg|tv3|rcdw)(\.|$)' gas_information_bench
  - rg -n '[ \t]+$' gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - task card rg: 0
  - forward audit tests: 0（7 passed）
  - candidate screen: 0；GIB-C4-LR 与 GIB-C4-CH4 均为 candidate_selected，joint_rank=11，四类负对照均通过；CH4 safety_gate_required=True
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/gib/audit/forward.py
  - gas_information_bench/tests/test_forward_audit.py
  - docs/p2/P2_bench规格书.md §4.4 候选筛选表
artifact_sha256:
  - gas_information_bench/pyproject.toml=71CE666711ECC8758A67D5B35114C229E5B8DDAED3D14B5C413ABF4DE5EE63D0
  - gas_information_bench/gib/audit/__init__.py=C0A8793A19E84E889E08D8A1BD3639438278E36B4BC7CF81F9E188BBA48B7763
  - gas_information_bench/gib/audit/forward.py=12650B418B5027A4FC2EC1540619A2E319FE4D22EDE14156AAD3332416DB81EF
  - gas_information_bench/tests/test_forward_audit.py=DB4194194ADC9CB83A8920A8F94A13D6B67F2326CC6080CE085B08B4872A1A8C
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks:
  - 初轮联合秩直接使用未缩放 nuisance 列时，1e-5 的 SVD 秩与其他容差不一致；已改为按声明先验尺度缩放 nuisance 列，最终三档容差均为 joint_rank=11。
  - 初轮候选判定错误要求 joint_rank 等于目标维度；已修正为不少于目标维度，最终两个候选均通过纯前向筛选。
  - 两次临时候选探针分别误传 CandidateProfile、误将 AuditResult 直接 JSON 序列化而失败；未改变实现，改用 AuditConfig 并只读取 verdict/审计字段后 exit 0。
verdict: candidate_selected
next_allowed_task: P2-06
```

## P2-06 执行记录

```text
task_id: P2-06
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=F33DAC6939845C753220CB484DED99556155707ABA13C436006ADCC298FFE8EA
  - docs/p2/P2_bench规格书.md | P2-05 primary candidate=GIB-C4-LR；append-only 文档不对自身内容建立循环哈希
  - gas_information_bench/gib/audit/forward.py | workspace_sha256=12650B418B5027A4FC2EC1540619A2E319FE4D22EDE14156AAD3332416DB81EF
  - gas_information_bench/tests/test_forward_audit.py | workspace_sha256=DB4194194ADC9CB83A8920A8F94A13D6B67F2326CC6080CE085B08B4872A1A8C
changed_files:
  - gas_information_bench/gib/audit/__init__.py
  - gas_information_bench/gib/audit/grid.py
  - gas_information_bench/tests/test_grid.py
  - gas_information_bench/configs/p2_s1_grid.json
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - python -m pytest -q tests/test_grid.py
  - python -m gib.audit.grid
  - python -c "import json; json.load(open('configs/p2_s1_grid.json', encoding='utf-8')); print('valid')"
  - python -c "import json; from gib.audit.grid import grid_summary; actual=json.load(open('configs/p2_s1_grid.json', encoding='utf-8')); expected=json.loads(json.dumps(grid_summary(), ensure_ascii=False, sort_keys=True)); assert actual == expected; print('generated_config_matches_grid_summary')"
  - rg -n 'GIB-S1-(SUF|CRI|INS)' gas_information_bench/configs/p2_s1_grid.json docs/p2/P2_bench规格书.md
  - rg -n 'base_condition_id|noise_seed_index|noise_seed' gas_information_bench
  - rg -n '^\s*(from|import)\s+(hg|sg|tv3|rcdw)(\.|$)' gas_information_bench
  - rg -n '[ \t]+$' gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - grid tests: 0（4 passed）
  - grid generation: 0；9 格均生成
  - generated JSON parse: 0（valid）
  - generated config comparison: 0（JSON 规范化后与 grid_summary 完全一致）
  - 9-cell marker rg: 0
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/gib/audit/grid.py
  - gas_information_bench/tests/test_grid.py
  - gas_information_bench/configs/p2_s1_grid.json
  - docs/p2/P2_bench规格书.md §5 S1 信息量与夹角刻度
artifact_sha256:
  - gas_information_bench/gib/audit/__init__.py=C0A8793A19E84E889E08D8A1BD3639438278E36B4BC7CF81F9E188BBA48B7763
  - gas_information_bench/gib/audit/grid.py=F5B43CE11BE8179FAD295B3FCE77A9959CD96A5E7615AE7821B76D16DBC7E14E
  - gas_information_bench/tests/test_grid.py=732E98B1EBC175933FF4474CF482A233017449CE059670E9F554EE8F8968EA1A
  - gas_information_bench/configs/p2_s1_grid.json=F7D834DB51B13F47A8A7716252346B425876ECE3A5195DD61103B41196CB765F
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks:
  - 初轮 sufficient/high 单元 information_ratio=0.517，超过 0.5 门值；已调整确定性 sufficient noise profile 到 0.35，门值本身未改变，最终该单元 ratio=0.4618。
  - 初轮测试断言误检查固定噪声跨角度，而计划要求固定信息档只改变 coupling；已修正断言轴，最终 9 格全部可达且测试通过。
  - 初次生成时 `gib.audit.__init__` 预先导入 grid 触发 runpy 警告，旧捕获文件将该警告带入 JSON 顶部；已移除 package-level grid import 并清理旧生成前缀，最终生成命令无该警告且 `json.load` exit 0。
  - 初次用 Python 对象直接比较落盘 JSON 与 `grid_summary()` 时，tuple/list 序列化差异导致断言失败；改用 JSON 规范化等价比较后通过，逐格数值与配置 ID 一致。
verdict: grid_frozen
next_allowed_task: P2-07
```

## P2-07 执行记录

```text
task_id: P2-07
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=8F77263A3A8F08DDEFF85BB22F0FBA29B39B09F3B86667580B36E25F1F0222DD
  - docs/p2/P2_bench规格书.md §5 S1 3 × 3 | P2-06 verdict=grid_frozen；append-only 文档不对自身内容建立循环哈希
  - gas_information_bench/configs/p2_s1_grid.json | workspace_sha256=F7D834DB51B13F47A8A7716252346B425876ECE3A5195DD61103B41196CB765F
changed_files:
  - gas_information_bench/gib/audit/forward.py
  - gas_information_bench/gib/audit/s2_s3.py
  - gas_information_bench/tests/test_s2_s3.py
  - gas_information_bench/configs/p2_s2_s3_audit.json
  - gas_information_bench/configs/p2_s2_s3_frozen_evidence.json
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - python -m gib.audit.s2_s3
  - python -m pytest -q tests/test_s2_s3.py
  - python -m pytest -q
  - python -c "import json; from gib.audit.s2_s3 import audit_summary; actual=json.load(open('configs/p2_s2_s3_frozen_evidence.json',encoding='utf-8')); expected=audit_summary(); assert actual == expected; print('frozen_evidence_matches; verdict=' + expected['verdict'] + '; c4=' + expected['s2']['c4_pre_verdict'])"
  - rg -n 'base_condition_id|noise_seed_index|noise_seed' gas_information_bench
  - rg -n '^\s*(from|import)\s+(hg|sg|tv3|rcdw)(\.|$)' gas_information_bench
  - rg -n '[ \t]+$' gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- docs/p2/P2_bench规格书.md gas_information_bench docs/p2/README.md
exit_codes:
  - S2/S3 audit: 0；S2 五项量化检查、S3 三个独立关闭检查和同时关闭负对照均通过；`verdict=pass`，`c4_pre_verdict=eligible_for_P3_test`
  - S2/S3 tests: 0（3 passed）
  - full new-subproject tests: 0（14 passed）
  - frozen evidence comparison: 0（generated evidence matches audit_summary）
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/gib/audit/forward.py | AuditConfig 三个显式 headroom 开关及其前向语义
  - gas_information_bench/gib/audit/s2_s3.py | S2/S3 纯前向审计与证据生成器
  - gas_information_bench/configs/p2_s2_s3_audit.json | 阈值、probe nuisance 和独立开关 profile
  - gas_information_bench/configs/p2_s2_s3_frozen_evidence.json | S2/S3 生成数值与 verdict
  - gas_information_bench/tests/test_s2_s3.py | S2/S3 阈值、独立关闭和冻结证据测试
  - docs/p2/P2_bench规格书.md §6 S2、§7 S3
artifact_sha256:
  - gas_information_bench/gib/audit/forward.py=51DE91424E3E607AB4A10A02FDE1AE2C646725136A1B608BA5E8288C89E68DFD
  - gas_information_bench/gib/audit/s2_s3.py=92815A5271157EBB1BDAD298D3030C69244C963A8FDDF25E3D5A9FDB6EC013C4
  - gas_information_bench/tests/test_s2_s3.py=52E924BA7E16DEA8F437EAEB894867E7FC0CCB5CFADDDEC3B69B0BF8782C5C06
  - gas_information_bench/configs/p2_s2_s3_audit.json=01C113230635075A1159EE87E6A13F4ACA5E9A40A59FABAF0A97232C151C4FAA
  - gas_information_bench/configs/p2_s2_s3_frozen_evidence.json=EB240B0C7EE34875C5D4F163B507301FDEF2E5066D10C072F46BDC775C47B474
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 任务记录，不对自身内容建立循环哈希
failed_checks:
  - 首轮冻结证据严格比较因线性代数末位漂移失败，差异约 1e-9；已由审计脚本统一数值精度并保留小于 1e-6 的阈值量，未改变任何门值或 verdict。
  - 8 位小数仍有一个值跨舍入边界；已收敛到 6 位小数并重新生成证据，最终严格比较通过。
verdict: pass
next_allowed_task: P2-08
```

## P2-08 执行记录

```text
task_id: P2-08
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=8F77263A3A8F08DDEFF85BB22F0FBA29B39B09F3B86667580B36E25F1F0222DD
  - docs/p1/创新点候选集.md §3 | workspace_sha256=DE10F12F5A6D6F87B86983B5901CC6DE9F57ABB8C3CD63768E05FC8DE02C0263
  - docs/p2/P1候选修订记录.md §1–§4 | workspace_sha256=6724D9343C44D37D57E47C4DD99A84A7694C05F5A29C3A6CBF0D6877F1D7F38F
  - docs/多传感器多组分气体检测算法创新深度调研报告.md §3 | workspace_sha256=5D7BC2A2F4D9816084BFC892D2C804505620FE13D95CEE7B17975544F6355BF6
  - 项目总体规划.md §6.4 | workspace_sha256=49E674945540D0A9E077C48A2011D29A6A0C6E1DEF62B3C5B33D72ED062CA7E8
  - P2-07 前置 | README status=completed/pass
changed_files:
  - gas_information_bench/configs/p2_s4_metric_registry.json
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - python -c "load JSON and assert S4 layers, methods, paired splits/seeds, timing, solver, nested efficiency and pending authorization fields"
  - Test-Path all P2-08 input documents and generated registry
  - rg -n "base_condition_id|noise_seed_index|noise_seed" gas_information_bench
  - rg -n "^\s*(from|import)\s+(hg|sg|tv3|rcdw)(\.|$)" gas_information_bench
  - rg -n "[ \t]+$" gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - registry schema and field validation: 0（四层、5 split × 3 seed、嵌套五档、计时、solver 和联合 verdict 均通过）
  - input path validation: 0（6 个输入/产物路径均存在）
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/configs/p2_s4_metric_registry.json | S4 精度、效率、solver 和联合 verdict registry
  - docs/p2/P2_bench规格书.md §8 | S4 CRB、oracle、测量级、强基线和效率协议
  - docs/p2/README.md | P2-08 状态与执行记录
artifact_sha256:
  - gas_information_bench/configs/p2_s4_metric_registry.json=F68817A8C781A8A6DE17FA01BCBAD5842A9570F613406320C6FA6169AD898B66
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks:
  - 初次辅助路径核对误写为 docs/p1/候选集.md，结果为 False；未进入验收结论，随后按仓库实际路径 docs/p1/创新点候选集.md 复核通过。
  - P1 修订中的非劣带、效率门、硬件 profile 和精确重复次数仍为 pending_authorization/requires_human_value；本任务未填默认值，也未据此运行比较。
verdict: protocol_frozen
next_allowed_task: P2-09
```

## P2-09 执行记录

```text
task_id: P2-09
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=8F77263A3A8F08DDEFF85BB22F0FBA29B39B09F3B86667580B36E25F1F0222DD
  - P2-03 前置 | README status=completed/approved；gib-benchmark-1；当前 README sha256=9D4C05038E9D199917F4425D0C3ABD3D7712F7206D452BEDC8FF6FB6DEC7DD6B；pyproject sha256=71CE666711ECC8758A67D5B35114C229E5B8DDAED3D14B5C413ABF4DE5EE63D0
  - P2-06 前置 | README status=completed/grid_frozen；configs/p2_s1_grid.json sha256=F7D834DB51B13F47A8A7716252346B425876ECE3A5195DD61103B41196CB765F
  - P2-08 前置 | README status=completed/protocol_frozen；configs/p2_s4_metric_registry.json sha256=F68817A8C781A8A6DE17FA01BCBAD5842A9570F613406320C6FA6169AD898B66
changed_files:
  - gas_information_bench/configs/p2_data_schema.json
  - gas_information_bench/configs/p2_manifest_schema.json
  - gas_information_bench/configs/p2_split_contract.json
  - gas_information_bench/gib/contract.py
  - gas_information_bench/tests/test_contract.py
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
commands:
  - python -m pytest -q tests/test_contract.py
  - python -m py_compile gib/contract.py
  - python -m pytest -q
  - python -c "parse p2_data_schema.json, p2_manifest_schema.json and p2_split_contract.json"
  - rg -n "base_condition_id|noise_seed_index|noise_seed" .
  - rg -n "from (hg|sg|tv3|rcdw)|import (hg|sg|tv3|rcdw)" .
  - rg -n "[ \t]+$" gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - contract tests: 0（8 passed）
  - contract module compile: 0
  - full new-subproject tests: 0（22 passed）
  - contract JSON parse: 0
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/configs/p2_data_schema.json | 样本记录与数组层 schema
  - gas_information_bench/configs/p2_manifest_schema.json | manifest 字段与 SHA256 规则；仅契约，不是数据 manifest
  - gas_information_bench/configs/p2_split_contract.json | 5 split、3 partition、group isolation 规则
  - gas_information_bench/gib/contract.py | ID、样本、manifest、DSP provenance、solver、split 和 deployment/oracle 校验
  - gas_information_bench/tests/test_contract.py | P2-09 契约与泄漏拒绝测试
  - docs/p2/P2_bench规格书.md §9 | 数据契约、manifest、ID 与 split
artifact_sha256:
  - gas_information_bench/configs/p2_data_schema.json=6E2A2C95F025913D48911263047B2EB2863320985A6110F69305A3914637CDFA
  - gas_information_bench/configs/p2_manifest_schema.json=A1320C2BD0E808740BB6E9DBF293A8A13FB3ACC497E60B61AB863B360E795D20
  - gas_information_bench/configs/p2_split_contract.json=AED3A8F9BA037F581B0176D50D42382C2ED3E0C09DD39C75AFD208D83D595588
  - gas_information_bench/gib/contract.py=E42C875716E5CA683FD338B9A6B1B6223649D598B3AA6F3CFB601394E5033A17
  - gas_information_bench/tests/test_contract.py=333201701ED70B50DA98D96AAE935FFF99048C946F6D13E94F96B995620EEB6A
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks: []
verdict: contract_frozen
next_allowed_task: P2-10
```

## P2-10 执行记录

```text
task_id: P2-10
input_versions:
  - docs/p2/P2执行计划.md | workspace_sha256=8F77263A3A8F08DDEFF85BB22F0FBA29B39B09F3B86667580B36E25F1F0222DD
  - P2-04 前置 | README status=completed/ready_for_forward_screen；参数表 §3.2–§3.3；append-only 规格书不对自身内容建立循环哈希
  - P2-05 前置 | README status=completed/candidate_selected；forward.py workspace_sha256=51DE91424E3E607AB4A10A02FDE1AE2C646725136A1B608BA5E8288C89E68DFD
  - hydrogen_ng/docs/物理模型严格化实施计划.md | workspace_sha256=953887D913D0FF7D9032F0C1D90BFEECA676047D0B206BE0725B56DBA2EE1514
  - hydrogen_ng/docs/references/传感器硬件资料整理.md | workspace_sha256=D1C44BEF8E463809985570BA40E54CE25B2DCA182D41EBB0FC5BCDD66A438AC9
  - tunnel_ventilation/docs/references/传感器硬件资料整理.md | workspace_sha256=5D7FAF30639AB9F5E91324FF7B9B321E1244AB4EEF2044D5232227E7E1C0860E
  - tunnel_ventilation/docs/references/co2_o2_n2_gas_properties.md | workspace_sha256=1F347722388298F6660FDDB0B8EFF61AD34F4594348EABE733D61AF519391C5E
  - syngas/docs/references/co_acoustic_constants.md | workspace_sha256=A2913A114F72DBE9C123F8C2ECD404D4DE9C67CCC5BA129EC55F7B224359304B
changed_files:
  - gas_information_bench/configs/p2_s5_source_registry.json
  - gas_information_bench/configs/p2_s5_discrepancy_contract.json
  - gas_information_bench/gib/s5_contract.py
  - gas_information_bench/tests/test_s5_contract.py
  - docs/p2/P2_bench规格书.md
  - docs/p2/README.md
  - gas_information_bench/gib/audit/forward.py | 未修改，仅登记代码 hash 与参数绑定
commands:
  - Get-FileHash -Algorithm SHA256 -LiteralPath <P2-10 registry、contract、module、test、forward.py、P2执行计划>
  - python -m pytest -q tests/test_s5_contract.py
  - python -m py_compile gib/s5_contract.py
  - python -c "parse p2_s5_source_registry.json and p2_s5_discrepancy_contract.json"
  - python -m pytest -q
  - rg -n "base_condition_id|noise_seed_index|noise_seed" .
  - rg -n "from (hg|sg|tv3|rcdw)|import (hg|sg|tv3|rcdw)" .
  - rg -n "[ \\t]+$" gas_information_bench docs/p2/P2_bench规格书.md docs/p2/README.md
  - git diff --check -- gas_information_bench/configs/p2_s5_source_registry.json gas_information_bench/configs/p2_s5_discrepancy_contract.json gas_information_bench/gib/s5_contract.py gas_information_bench/tests/test_s5_contract.py docs/p2/P2_bench规格书.md docs/p2/README.md
exit_codes:
  - input hash: 0
  - S5 pure interface tests: 0（12 passed）
  - S5 module compile: 0
  - S5 JSON parse: 0
  - full new-subproject tests: 0（34 passed）
  - forbidden historical field rg: 1（无命中）
  - private cross-package import rg: 1（无命中）
  - trailing whitespace rg: 1（无命中）
  - git diff --check: 0
generated_artifacts:
  - gas_information_bench/configs/p2_s5_source_registry.json | 前向数量、来源分类、代码绑定、缺失来源和 verdict registry
  - gas_information_bench/configs/p2_s5_discrepancy_contract.json | delta 签名、off/P5 profile、字段与单位契约
  - gas_information_bench/gib/s5_contract.py | registry 校验、显式单位换算和 discrepancy 接口
  - gas_information_bench/tests/test_s5_contract.py | 来源阻塞、单位换算、off 不变和 P5 禁止注入测试
  - docs/p2/P2_bench规格书.md §10 | S5 来源与 discrepancy 接口规范
artifact_sha256:
  - gas_information_bench/configs/p2_s5_source_registry.json=6FAFFAEBCB91EAC5855C705A153E1F6B5DF53EF8D6D744F9370F3A53E7A64946
  - gas_information_bench/configs/p2_s5_discrepancy_contract.json=1507F9F77301445F3CE46B2A3F90829CDD4ABF41AFAA15BC6FC1E4658338A3E3
  - gas_information_bench/gib/s5_contract.py=3A91013F0B60777BE0F71AEE43DFCD818E15A7B487B7F2E6D568C0A8B6CA364F
  - gas_information_bench/tests/test_s5_contract.py=DB22E46FD64640441AFDF50BBEA7BC25C28B5737B0FAFB2C1BE3C796CB34794F
  - gas_information_bench/gib/audit/forward.py=51DE91424E3E607AB4A10A02FDE1AE2C646725136A1B608BA5E8288C89E68DFD
  - docs/p2/P2_bench规格书.md=append-only 任务记录，不对自身内容建立循环哈希
  - docs/p2/README.md=append-only 状态页，不对自身内容建立循环哈希
failed_checks:
  - source_complete 未满足：Ar 三项物性、目标 TraceGas filter response、NDIR proxy 标定、声学频率绑定和逐通道噪声标定证据仍缺失；按计划给出 blocked_source_missing，未做邻近数值替代。
verdict: blocked_source_missing
next_allowed_task: P2-11
```

## P2-11 执行记录

```text
task_id: P2-11
inputs:
  - P2-01 pass
  - P2-03 approved
  - P2-09 contract_frozen
changed_files:
  - gas_information_bench/pyproject.toml
  - gas_information_bench/gib/freeze.py
  - gas_information_bench/gib/cli.py
  - gas_information_bench/configs/p2_s6_ownership_registry.json
  - gas_information_bench/tests/test_freeze_contract.py
  - gas_information_bench/tests/test_ownership_registry.py
  - gas_information_bench/README.md
  - gas_information_bench/docs/README.md
  - gas_information_bench/outputs/
  - README.md
  - docs/p2/P2_bench规格书.md
validation:
  - freeze contract tests: 5 passed
  - full new-subproject tests: 43 passed
  - independent test collection: 43 tests collected
  - independent editable install and installed gib CLI: passed
  - forbidden historical fields: no match
  - historical private-package imports: no match
  - git diff --check: passed
verdict: ownership_frozen
next_allowed_task: P2-12_after_H1_authorized
```

## H1 授权与 S5 来源补充记录

```text
date: 2026-08-25
authorization:
  authority: project_owner_delegated_evidence_based_freeze
  verdict: authorized
  precision_non_inferiority_band_mol_per_mol:
    N2: 0.008
    CO2: 0.003
    O2: 0.010
    Ar: 0.005
  efficiency_reduction:
    iterations: 0.30
    forward_calls: 0.30
    solver_wall_clock: 0.20
    single_sample_latency: 0.20
  maximum_other_primary_regression: 0.05
  independent_repeats: 30
  hardware_profile: GIB-HW-WIN-R9-8940HX-RTX5060L-20260825
s5_source_update:
  verified: MOLAR_MASS.Ar, CP_MOLAR.Ar, THERMAL_CONDUCTIVITY.Ar
  remaining_blockers: NDIR.filter_response, NDIR_ABSORPTION_COEFFICIENTS, MODALITY_FREQUENCY_RESPONSE.forward_acoustic_frequency_hz, OBSERVATION_NOISE_STD.instrument_traceability
  verdict: blocked_source_missing
validation:
  - S4 authorization and S5 targeted tests: 15 passed
  - full new-subproject tests: 46 passed
  - S5 registry validation: 14 inventory entries, 15 sources, 4 blockers
artifact_sha256:
  - gas_information_bench/configs/p2_s4_metric_registry.json=C287C9B48A74C420614E617859458E1D750967EF9542AFBB5FBE7918F02A1D5E
  - gas_information_bench/configs/p2_s5_source_registry.json=B53FA053809A6C5ED15A830D649F6CB2F3A3ED611C46C3D71C255F626D95A5F8
  - gas_information_bench/tests/test_s4_authorization.py=109FCB8E1C7A3E3098ABDCD4CD27164242079488B3652376CF11AB23C0685F5D
  - docs/p2/S5来源检索记录.md=3BBB7B996823C8F74027643EB4964ADA6A5555BC56E2073E2F2E2191CF13996B
```

## P2-12 执行记录

```text
task_id: P2-12
input_versions:
  - H1 authorization | docs/p2/P1候选修订记录.md sha256=606E664AC87E3B723165D6E73B42EEDAF608FFF38000DAF5779EDBE8CC183C61
  - P2-06 | configs/p2_s1_grid.json sha256=D193585A12932C78A8383A73058AFB989318DAB83EAE4953E4C4B74F1C682310
  - P2-07 | configs/p2_s2_s3_frozen_evidence.json sha256=357874A6121DB1770B238BDCDF22B7D5A6A8E78B06D5EDC7C1C312D456FD1E4F
  - P2-08 | configs/p2_s4_metric_registry.json sha256=C287C9B48A74C420614E617859458E1D750967EF9542AFBB5FBE7918F02A1D5E
  - P2-09 | configs/p2_data_schema.json sha256=1FA14AEDA5A384D2EFE0F28682910E304719367352A31221BAD78A8654F3D90E
changed_files:
  - docs/p2/P2_bench规格书.md §12
validation:
  - matrix rows and matrix_ready token audit: passed
  - H1 registry JSON parse and direct-judgeability audit: passed
  - full new-subproject tests: 46 passed
  - forbidden historical fields: no match
  - historical private-package imports: no match
  - git diff --check: passed
matrix_rows:
  - C2-S0 active preflight
  - C2-IC-RDU-VP conditional
  - C5-A-CRB-ADAPT active
  - C5-B-FO-MPLSELM active
  - C4-ID-MULTIVIEW active
  - C5-C-CR-PKD conditional
  - C5-D-FIGS conditional
failed_checks: []
verdict: matrix_ready
next_allowed_task: P2-13_after_P2-10_source_complete
```

## P2-10 合成 profile 授权修订记录

```text
date: 2026-08-25
authorization: project owner permits controlled synthetic benchmarking without target-device fidelity claims
changed_files:
  - gas_information_bench/configs/p2_s5_source_registry.json
  - gas_information_bench/gib/s5_contract.py
  - gas_information_bench/tests/test_s5_contract.py
  - docs/p2/S5来源检索记录.md
  - docs/p2/P2_bench规格书.md §10
invariants:
  - every paired method uses the identical forward/noise/profile hash
  - unmodeled filter and acoustic-frequency fields remain explicit null/not_modeled
  - engineering assumptions remain not hardware-verified
  - real-hardware performance claims are forbidden before P5
validation:
  - S5 targeted tests: 13 passed
  - full new-subproject tests: 47 passed
  - source registry: 14 inventory entries, 16 sources, 0 missing blockers
artifact_sha256:
  - gas_information_bench/configs/p2_s5_source_registry.json=7573FC247B490BAC2DB5613E534A37814C558DD7EF2B75E6100F5ABC0BFC6995
  - gas_information_bench/gib/s5_contract.py=A66F005101A04662558A64EAC2BA0D2EA6B901CDA2F0419CEC2C53774F8D039A
  - gas_information_bench/tests/test_s5_contract.py=C36D9D25E66CA2A1D58C0288462FF2D92F967C3A0896BCA158DC6E2E6A030C77
  - docs/p2/S5来源检索记录.md=A67787702B4263A4D3B6C6CEEFDEFA9CDE85ABCAF7409B3A5502994B17B109B6
failed_checks: []
verdict: source_complete
next_allowed_task: P2-13
```

## P2-13 执行记录

```text
task_id: P2-13
inputs:
  - P2-01 through P2-12 acceptable terminal artifacts
  - P2-10 revised verdict=source_complete
changed_files:
  - docs/p2/P2_bench规格书.md §5.3, §13
  - docs/p2/tools/render_s1_grid_table.py
  - docs/p2/generated/s1_grid_table.md
validation:
  - generated S1 table matches frozen grid
  - 23 local Markdown links resolved before task-record append
  - active specification terminology audit passed
  - placeholder-token scan: no match
  - forbidden historical fields in runtime project: no match
  - historical private-package imports: no match
  - full new-subproject tests: 47 passed
artifact_sha256:
  - gas_information_bench/configs/p2_s1_grid.json=D193585A12932C78A8383A73058AFB989318DAB83EAE4953E4C4B74F1C682310
  - docs/p2/tools/render_s1_grid_table.py=46E81011C36B6548FF5CCBAEF185000DE4D407CBBB016001AB5E9AA842F56D0E
  - docs/p2/generated/s1_grid_table.md=B89AC664D3EAC68C3AE67EC6ECC0177A048A8337F2C5A45AE1E359C27C5AFAB2
  - docs/p2/P2_bench规格书.md=append-only task record; no self-hash
failed_checks:
  - first active-spec audit command used PowerShell-sensitive backticks and failed to parse; rerun with ASCII verdict tokens passed
  - second audit counted noise_seed as a substring of noise_seed_index; rerun with token boundaries passed
verdict: ready_for_review
next_allowed_task: P2-14
```

## P2 关闭审查修复记录

2026-08-25 的关闭审查修复将 `configs/` 冻结为随普通 wheel 分发的唯一资源包；样本校验改为同时消费 manifest 与外部 DSP 期望 hash，并拒绝非法 crosstalk、虚构 S1 单元、Raw/DSP view 冲突、错误 axes 和逃逸路径；freeze verifier 改为精确校验角色、字段、hash policy 与重复路径；`principal_angle` 来源统一为 `sample_fisher` 中白化 Jacobian 的 Gram block。目标测试、普通 wheel 冒烟和全量测试通过后，P2 正式证据提升为 `outputs/archive/freezes/GIB-FREEZE-P2-20260825-01`。
