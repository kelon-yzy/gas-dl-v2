# TQIF 新算法设计与 A2 至 A2M 验证计划（失败归档）

> 文档状态：`ARCHIVED / SCIENTIFIC_FAILURE / ABANDONED / NO_REUSE`  
> 当前工作包：TQIF 失败归档完成；下一工作包必须重新构思新算法，尚未登记候选  
> 阶段归属：TQIF 已在 A2 终止，未进入 A2H、A2M 或 A3  
> 历史边界：既有 A2、A2H v2、A2M 第一轮结果只读，不覆盖、不改名、不参与新候选的正式选择
> 执行就绪状态：`ABANDONED`；实现与产物只读保留，不得启动 TQIF 续跑或局部修补

> 实施结论：A2-TQIF-0/1/2 全部完成；A2-TQIF-2 完成 15 个基线运行，A2-TQIF-3 完成两档 full 对 C0 的 20 个逻辑格。token16 和 token32 的完整 TQIF 相对 C0 分别退化 343.05% 和 77.99%，均为 0/5 seed 改善。机器终态为 `NEGATIVE_RESULT`，项目决策为 `SCIENTIFIC_FAILURE / ABANDONED`。

## 0. 文档职责

本文是 `GF-I14` 目标查询式传感器交互融合网络（Target-Query Interaction Fusion Network，`TQIF-Net`）的算法设计和验证事实源，负责冻结以下内容：

1. TQIF 要修复的已观测问题、结构假设和不变量；
2. 完整的数据流、融合模块、输出契约和复杂度边界；
3. A2 机制筛选、A2H 压力验证、A2M 主流架构收口和 A3 外部验证的顺序；
4. matched 对照、必要消融、晋级门、停止规则和失败声明；
5. 后续新算法必须满足的完整度门，避免把单层改动或模块堆叠登记为独立算法。

既有 [A2 分步执行计划](archive/07_A2分步执行计划.md)、[A2H 分步执行计划](archive/09_A2H分步执行计划.md) 和 [A2M 主流架构对照分步执行计划](archive/10_A2M主流架构对照分步执行计划.md)（三份已于 2026-09-05 归档至 `archive/`）继续记录第一轮协议和历史结果。本文只追加一个新的、预注册的候选验证轮次，不重写历史结论。

## 1. 研究问题与设计依据

### 1.1 已有实验暴露的问题

当前证据支持四个事实：

1. 单传感器 Ridge、GBDT 和预测级后融合显著弱于多传感器联合建模，说明目标依赖跨传感器互补信息；
2. 不同目标组分的最佳单传感器不同，例如热导率通道对 He 更敏感，NDIR 通道对 CO₂ 更敏感，统一全局融合向量可能掩盖目标特异关系；
3. `M1` Deep Sets 的统一 masked mean 在固定三路完整传感器上没有超过 matched concat，说明先把全部 token 压成一个全局均值不是充分机制；
4. A2M 中 MLP 明显优于 ResNet 和 FTT，说明新算法不能依赖扩大容量，必须针对 MLP 没有显式表达的结构提出可证伪改进。

TQIF 的研究问题因此冻结为：

> 能否把每个目标组分建模为独立查询，使其分别读取单传感器证据和传感器对互补证据，并在不硬编码气体名称、传感器数量和数据集分支的前提下，稳定超过等参数量 concat MLP？

### 1.2 论文级方法主张

TQIF 不把 Attention、pairwise feature 或 softmax 包装为单独创新。候选主张限定为：

> 提出面向多组分定量的目标—传感器—交互融合结构：任务输出槽以查询 token 表达，传感器 token 提供局部证据，无序传感器对 token 提供互补证据；同一融合方程可在不同传感体系和不同目标组分上重新训练。

这是一项方法假设，不是当前结果。只有 A2、A2H 和 A2M 的预注册门全部通过后，才允许把 TQIF 冻结为 A3 输入模型。

### 1.3 不变量

- `mixture_id` 始终是 Ar-He-CO₂ 仿真的分组主键，不回退或重写为 `sequence_id`。
- 新 benchmark 和结果不得依赖 `base_condition_id`、`noise_seed_index` 或 `noise_seed`。
- 查询槽只使用任务适配器声明的 `target_slot_id`；融合核心不得出现 `Ar`、`He`、`CO2`、`m-xylene` 等名称分支。
- 传感器顺序变化不得改变输出；传感器身份、类型和可用性不得被删除。
- A2 与 A3 可以使用不同传感器编码器，但必须使用同一 TQIF 融合方程、mask 语义和任务槽协议。
- attention 和 interaction gate 只表示模型内部路由，不等于物理贡献、可靠性或置信度。
- 旧 A1 test、A2H hard-test 和 A2M formal holdout 已被查看，只能作历史参照，不能用于新候选选择。

## 2. TQIF 完整架构

### 2.1 输入与统一 token

对一个样本，传感器编码器输出：

\[
Z = \{z_i\}_{i=1}^{S}, \qquad z_i \in \mathbb{R}^{d}
\]

其中 `S` 是当前可用传感器数。A2 使用稳态标量编码器；A3 使用统计、TCN、GRU 或其他预注册时间编码器。融合核心只读取统一字段：

- `sensor_embedding`：传感器编码结果；
- `sensor_id_embedding`：物理身份；
- `sensor_type_embedding`：测量原理或涂层类型；
- `sensor_mask`：是否可用；
- `quality_embedding`：显式质量字段的编码，不作为真实精度；
- `target_slot_embedding`：任务适配器声明的目标查询槽。

最终传感器 token 为：

\[
\tilde z_i = \operatorname{LayerNorm}
\left(
W_z z_i + e_i^{id} + e_i^{type} + W_q q_i
\right)
\]

缺失传感器不生成有效 token，不能以全零向量伪装成正常观测。

### 2.2 无序传感器对 token

TQIF 不只汇总单个传感器，还为每个有效无序传感器对 `(i,j)` 构造交互 token：

\[
r_{ij}=\phi_{pair}\left(
\left[
\tilde z_i+\tilde z_j,
|\tilde z_i-\tilde z_j|,
\tilde z_i\odot\tilde z_j,
e_{ij}^{type}
\right]
\right), \quad i<j
\]

这里使用和、绝对差和逐元素积，保证交换 `i,j` 不改变 pair token。`e_{ij}^{type}` 由无序类型对生成，例如声学—热学与 QCM—QCM；不得为每个固定通道对注册一套独立网络。

`φ_pair` 使用共享的低秩两层 MLP。这样既能显式建模传感器互补关系，又避免参数量随具体通道对线性复制。若 `S<2`，pair 集合显式为空，并按结构契约令 `h_k^{pair}=0`、`g_k=0`，模型退化为目标查询式单传感器模型；这一路径必须有单元测试，不得依赖 attention 对空张量的隐式行为。

### 2.3 目标查询的两级取证

任务适配器声明 `K` 个目标槽，融合核心为每个槽构造查询 `u_k`。查询先读取单传感器证据：

\[
h_k^{sensor}
=\operatorname{MHA}
\left(u_k,\{\tilde z_i\},\{\tilde z_i\};m_i\right)
\]

再读取传感器对证据：

\[
h_k^{pair}
=\operatorname{MHA}
\left(u_k+h_k^{sensor},\{r_{ij}\},\{r_{ij}\};m_i m_j\right)
\]

两路证据通过目标特异门融合：

\[
g_k=\sigma\left(W_g[u_k,h_k^{sensor},h_k^{pair}]\right)
\]

\[
h_k=\operatorname{LayerNorm}
\left(u_k+h_k^{sensor}+g_k\odot h_k^{pair}\right)
\]

`g_k` 用于判断当前目标是否需要 pair evidence。它不得被解释为传感器可靠性，也不得直接用于故障抑制或置信区间。

### 2.4 输出头

TQIF 复用可配置总量—比例契约，不在融合核心中硬编码固定总量。

每个目标槽输出一个比例 logit：

\[
a_k=w_a^\top h_k+b_a,
\qquad
p=\operatorname{softmax}(a)
\]

固定总量任务：

\[
\hat y=T_{fixed}p
\]

可变总量任务：

\[
\hat T=\operatorname{softplus}
\left(g_T(\operatorname{Pool}(\{h_k\}))\right),
\qquad
\hat y=\hat T p
\]

A2 的 Ar-He-CO₂ 使用固定 100 mol% 语义；A3 的 xylene 使用可变 ppm 总量。softmax 只保证非负和闭合，不保证结构零的精确输出；纯组分、二元和三元样本必须分层报告，不能用后处理阈值隐藏失败。

### 2.5 损失与训练边界

第一轮 TQIF 只复用当前冻结的组成回归损失和 train-only scaler，不同时引入 attention 熵、查询正交、对比学习、蒸馏或可靠性损失。可变总量任务允许增加独立登记的总量损失：

\[
\mathcal L
=\mathcal L_{component}
+\lambda_T\mathcal L_{total}
\]

其中 `λ_T` 必须在 formal 解锁前由配置冻结。若新正则项没有独立诊断和消融，它不能进入正式候选。

训练必须记录：

- 五个固定 seed；
- 参数量、峰值内存、训练时间和推理时间；
- 每个查询的传感器 attention、pair attention 和 gate 分布，仅作诊断；
- 梯度有限值、checkpoint round-trip 和确定性指纹；
- 所有 scaler、early stopping 和 recipe 选择的 fit scope。

### 2.6 容量与复杂度

TQIF 的计算复杂度主要为：

\[
O(S^2d)+O(KSd)+O(KS^2d)
\]

A2 的 `S=3` 和 A3 的 `S=6` 均处于小集合范围。首轮只允许两档容量：

| recipe | token dim | attention heads | pair hidden | query FFN | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| `tqif_token16_pair16` | 16 | 2 | 16 | 32 | 低容量主候选 |
| `tqif_token32_pair32` | 32 | 4 | 32 | 64 | 容量敏感性对照 |

实现后必须登记真实参数量。每档 TQIF 都要有参数量容差内的 concat MLP；不能用更大模型对比 227 参数 MLP 后把容量收益归因于融合机制。

## 3. 新算法完整度门

TQIF 以及后续任何新算法，在进入正式实现前必须同时具备以下内容：

1. **问题闭环**：明确由哪一条现有失败或诊断触发，不能从热门模型反向寻找任务；
2. **结构闭环**：输入、核心表示、交互、输出和 mask 行为完整，不能只改激活函数、层数或优化器；
3. **训练闭环**：损失、scaler、seed、预算、checkpoint 和失败语义明确；
4. **归因闭环**：至少一个等参数量对照和能够拆开核心机制的必要消融；
5. **评价闭环**：iid、压力轴、逐组分、关键子组、效率和失败案例均有预注册出口；
6. **迁移闭环**：说明哪些模块跨数据集保持不变，哪些只属于适配器或任务头；
7. **停止闭环**：核心消融失败时直接关闭，不追加第二个复杂模块制造不可归因收益；
8. **文献边界**：明确借鉴部件和本项目新增结构，不以模块拼接数量作为创新性。

以下变化不能单独登记为新算法：

- 只增加隐藏层、注意力头、宽度或训练轮数；
- 只更换 optimizer、scheduler、activation 或 normalization；
- 在同一 concat MLP 后追加未校准 gate、ErrorNet 或后处理；
- 把已有 TCN、GRU、Transformer、Deep Sets 或 GBDT 改名后重新排序；
- 没有 matched 对照和核心消融的多模块串联。

“完整”不等于无限复杂。每个候选只能有一个主机制，辅助模块必须服务同一个研究问题，并能被单独移除验证。

### 3.1 共用预注册常量

TQIF 新一轮继承现有 A2 至 A2M 的统计纪律，并在任何训练前把以下值写入机器配置：

| 配置项 | 冻结值 | 判定语义 |
| --- | ---: | --- |
| `PARAM_MATCH_REL_DIFF_MAX` | 0.10 | 主比较双方真实参数量相对差不超过 10% |
| `BASELINE_REPRO_REL_DIFF_MAX` | 0.03 | 当前复现的五 seed 平均主指标相对历史锚点偏差不超过 3% |
| `MIN_MACRO_REL_GAIN` | 0.05 | 相对指定强基线的平均主指标改善至少 5% |
| `MAX_IID_REL_DEGRADATION` | 0.05 | iid 主指标相对 A2M-MLP 的平均退化不超过 5% |
| `SEED_DIRECTION_MIN` | 4 / 5 | 五个独立 seed 至少四个改善方向一致 |
| `COMPONENT_RNMAE_DEGRADATION_MAX` | 0.005 | 任一组分 RNMAE 绝对退化不超过 0.005 |
| `GROUP_BOOTSTRAP_REPEATS` | 2000 | 以独立 `mixture_id` 为单位做配对 bootstrap |
| `PAIRED_CI_LEVEL` | 95% | `TQIF - baseline` 差值的 CI 上界严格小于 0 |

若新数据版本需要修改任一常量，必须在候选训练前升版协议并给出物理或统计依据；看到开发或正式结果后不得修改。

## 4. A2：创新机制筛选

### 4.1 A2 第一轮事实边界

A2 第一轮的 Deep Sets、输出头和 grouped OOF residual 结果保持只读，终态继续是 `NEGATIVE_RESULT / FORMAL_TEST_NOT_ENTERED`。新轮次不修改这些结果，只使用允许的 train、val 和开发证据建立 TQIF 候选。

### 4.2 A2-TQIF 工作包

| 步骤 | 工作 | 产物 | 阶段门 |
| --- | --- | --- | --- |
| A2-TQIF-0 | 冻结输入、模型、预算、seed、对照和访问集合 | protocol、model recipes、hash | 任何训练前全部冻结 |
| A2-TQIF-1 | 实现 sensor/pair/query/head 与契约测试 | 模型、单元测试、smoke | 排列不变、身份保留、mask 正确、梯度有限 |
| A2-TQIF-2 | 复现 A2M-MLP 与 matched concat | 基线回归、参数审计 | 基线不漂移，matched 容差通过 |
| A2-TQIF-3 | 两档 recipe、五 seed 开发比较 | dev summary、checkpoint | 不读取旧 test 或 formal 标签 |
| A2-TQIF-4 | 核心机制消融和失败案例 | ablation summary | 主机制收益可独立归因 |
| A2-TQIF-5 | 冻结唯一候选或负结果关闭 | selection manifest | 至多一个 TQIF recipe 进入 A2H |

### 4.3 A2 matched 矩阵

| ID | sensor evidence | pair evidence | target query | head | 作用 |
| --- | --- | --- | --- | --- | --- |
| C0 | ordered concat | 无 | 共享全局表示 | H0 | 当前强基线语义 |
| C1 | token attention | 无 | 单个共享查询 | H0 | 隔离 token attention |
| Q1 | token attention | 无 | 每目标独立查询 | H0 | 隔离 target query |
| I1 | token attention | 有 | 单个共享查询 | H0 | 隔离 pair interaction |
| TQIF-H0 | token attention | 有 | 每目标独立查询 | H0 | 完整融合核心 |
| TQIF-STR | token attention | 有 | 每目标独立查询 | STR | 融合核心加输出契约 |

`H0` 是无闭合约束的逐组分线性回归头，只用于控制融合表示；`STR` 是 §2.4 的总量—比例约束头。A2 同时保留二者，是为了避免把输出几何收益记到 TQIF 融合核心名下。

参数匹配的主比较固定为：

- `TQIF-H0 - matched C0`：完整融合机制收益；
- `Q1 - C1` 与 `TQIF-H0 - I1`：无 pair 与有 pair 条件下的目标查询收益；
- `I1 - C1` 与 `TQIF-H0 - Q1`：共享查询与目标查询条件下的 pair evidence 收益；
- `TQIF-STR - TQIF-H0`：输出头收益，不归因于融合核心。

`C1/Q1/I1/TQIF-H0` 构成“共享或目标查询 × 无 pair 或有 pair”的 2×2 因子矩阵。I1 使用单个共享查询先读取 sensor token，再读取 pair token；其余方程与 TQIF 相同。所有主比较都必须通过隐藏维度调整达到预注册参数量容差，不能只对 C0 做容量匹配。

### 4.4 A2 晋级门

TQIF 进入 A2H 前必须同时满足：

1. 相对 matched concat 的平均主指标改善至少 5%，且五 seed 中至少四个同向；
2. 2000 次 paired group bootstrap 的 95% CI 上界严格小于 0；
3. 三个组分均不超过预注册退化上限；
4. 目标查询效应在 `Q1-C1` 或 `TQIF-H0-I1` 中至少一项通过，pair 效应在 `I1-C1` 或 `TQIF-H0-Q1` 中至少一项通过；两类效应缺一不可，且完整收益不能完全由 STR head 解释；
5. 两档容量只允许冻结一个 recipe，不按单 seed 最优值选择；
6. attention 和 gate 的数值有限，传感器置换测试严格通过；
7. 若核心机制未通过，A2 以新增候选负结果关闭，不进入 A2H。

旧 A1 test 已公开，不作为新候选 formal。A2 的职责是机制筛选，不重新把已查看 test 包装成未知测试集。

## 5. A2H：压力与 OOD 验证

### 5.1 开发数据与正式数据边界

通过 A2 的唯一 TQIF recipe 可以在 A2H 已允许的 train、val、stress-val 上开发。既有 A2H v2 hard-test 已解锁，只能作历史参照。新的正式判断必须：

- 升版数据和协议；
- 使用全新 `mixture_id`；
- 生成新的 content hash、split hash 和 test lock；
- 不依赖 `base_condition_id`、`noise_seed_index` 或 `noise_seed`；
- 在候选、checkpoint 规则和门槛冻结后一次解锁。

### 5.2 压力轴

主晋级轴沿用通过资格审计的：

- calibration；
- environment；
- joint；
- noise。

composition 继续只作诊断，除非新版本在候选结果产生前重新通过困难度资格审计。不得因为 TQIF 在 composition 上表现好而事后降低资格门。

### 5.3 A2H 对照与门

每个轴至少比较：

- B3 Ridge；
- B4 GBDT；
- `A2M-MLP / mlp_lbfgs_width32`；
- TQIF 的 matched concat；
- 冻结 TQIF；
- A2 的 `Q1` 与 `I1`，仅用于机制诊断。

进入 A2M 新一轮比较要求：

1. iid 主指标相对 A2M-MLP 的平均退化不超过 5%；
2. 至少两个合格压力轴相对 A2M-MLP 和 matched concat 同时改善至少 5%；
3. 每个晋级轴五 seed 至少四个同向，且 2000 次 paired group bootstrap 的 95% CI 上界严格小于 0；
4. 任一关键组分退化不超过上限；
5. 最差环境、最差组成区域和失败案例不被总体均值掩盖；
6. 训练与推理成本完整登记；
7. 未通过时形成 `NEGATIVE_RESULT`，不得直接进入 A3。

## 6. A2M：主流架构综合收口

### 6.1 新一轮模型矩阵

A2M 第一轮 `MLP_RETAINED` 对当时的候选集合继续有效。只有通过 A2H 的 TQIF 才能触发 A2M 新一轮比较：

| 模型 | 角色 | recipe 处理 |
| --- | --- | --- |
| B3 Ridge | 低容量线性基线 | 唯一冻结实现 |
| B4 GBDT | 传统强基线 | 唯一冻结实现 |
| A2M-MLP | 主强基线 | 复用冻结 recipe |
| A2M-RESNET | 主流负对照 | 复用冻结 recipe |
| A2M-FTT | 主流负对照 | 复用冻结 recipe |
| matched concat MLP | TQIF 容量对照 | 与 TQIF 参数量匹配 |
| TQIF | 新方法候选 | A2H 冻结的唯一 recipe |

ResNet、FTT 和 MLP 不因加入 TQIF 重新开放无上限搜索；TQIF 也不得比其他架构获得更多 recipe、seed 或 formal 访问。

### 6.2 新 formal holdout

旧 A2M formal 已被查看，不能评价新候选。A2M 新一轮必须生成独立 holdout，并在开发结果前登记：

- 新数据版本和 schema；
- 全新的 `mixture_id` 命名空间；
- iid、calibration、environment、joint、noise 主轴；
- composition 的诊断身份；
- content、split、profile、protocol 和 checkpoint hash；
- formal 一次解锁状态机。

### 6.3 最终选择

TQIF 只有在新 formal 上继续满足以下条件，才可替换 MLP：

1. iid 主指标相对 A2M-MLP 的平均退化不超过 5%；
2. 至少两个合格压力轴相对 A2M-MLP 和 matched concat 同时达到 5% 改善，并通过 4/5 seed、paired CI 和组分退化门；
3. 相对 matched concat 的收益仍存在；
4. 核心消融与开发阶段方向一致；
5. formal 结果未回流 recipe、seed、epoch、门槛或数据范围。

允许的终态沿用现有 A2M 语义：

| 终态 | 条件 | A3 交接 |
| --- | --- | --- |
| `POSITIVE_RESULT` | TQIF 通过全部开发与 formal 门 | 冻结 TQIF 为 A3 完整输入核心，MLP 保留为强基线 |
| `MLP_RETAINED` | TQIF 未通过或 formal 未确认 | A3 正式创新算法验证保持阻塞；MLP 仅保留为后续强基线 |
| `INVALID` | test 泄漏、hash 不一致、预算不公平或证据不完整 | 结果作废，不进入 A3 |

## 7. A3：泛用性验证边界

A3 不再承担 TQIF 的发明和主选择。只有 A2M 终态为 `POSITIVE_RESULT` 时，A3 才验证 TQIF 在 xylene 上的泛用性。

### 7.1 保持不变的部分

- target-query 与 pair-evidence 融合方程；
- sensor、pair、target slot 和 mask 契约；
- 两级 attention 和 interaction gate；
- 输出头的固定总量、可变总量配置规则；
- 候选预算、seed 数、文件级 split 和 test lock 原则。

### 7.2 允许变化的部分

- xylene 数据适配器；
- 单传感器时间编码器；
- `target_slot_id` 对应的 m/o/p 任务语义；
- 可变 ppm 总量头；
- train-only scaler 和文件级窗口构造。

模型权重按 xylene 训练集重新训练；A3 声称架构复用，不声称跨气体零样本迁移。

### 7.3 归因矩阵

A3 必须在相同时间编码器下比较 concat 与 TQIF：

| 时间表示 | concat | TQIF | 主要回答 |
| --- | --- | --- | --- |
| 统计特征 | 必须 | 必须 | 稳态统计下融合核心是否复现收益 |
| 轻量 TCN | 必须 | 必须 | 局部时间模式下是否复现 |
| 单层 GRU | 必须 | 必须 | 递归状态下是否复现 |
| 轻量 temporal Transformer | 必须 | 必须 | 注意力时间编码下是否复现 |
| NCDE-lite | 条件进入 | 条件进入 | 不规则时间诊断满足时是否复现 |

如果只有某个时间编码器获益，结论必须归因于“编码器与 TQIF 的组合”，不能单独声称通用融合核心成立。

## 8. 计划文件与唯一职责

| 位置 | 计划职责 |
| --- | --- |
| `src/gf/dl/tqif.py` | sensor/pair/query 融合核心，不读取 dataset 名称 |
| `src/gf/dl/task_heads.py` | 固定总量与可变总量输出契约 |
| `configs/model/tqif_*.json` | 两档 TQIF recipe 与 matched concat |
| `configs/experiment/a2_tqif_*.json` | A2 机制筛选与消融 |
| `configs/experiment/a2h_tqif_*.json` | A2H 压力验证、数据升版与 test lock |
| `configs/experiment/a2m_tqif_*.json` | A2M 模型矩阵、新 formal 和关闭门 |
| `src/gf/pipeline/` | 沿用各阶段编排边界，不新建 A2I pipeline |
| `tests/test_tqif_*.py` | 排列、mask、pair、query、head、参数与 checkpoint 测试 |
| `outputs/summary/{a2,a2h,a2m}/` | 按新运行版本追加机器事实，不覆盖历史 summary |

实际文件名在编码前由协议配置冻结。结果目录必须能同时保留第一轮历史结果和 TQIF 新轮次结果。

## 9. 验证矩阵

| 类型 | 必须验证 |
| --- | --- |
| 契约 | 任意 `S≥1`、任意 `K≥1`；目标名和 dataset 名不进入核心 |
| 排列 | 传感器和 pair token 重排后输出一致，sensor identity 保留 |
| mask | 缺失传感器及关联 pair 完全不参与 attention |
| pair | `(i,j)` 与 `(j,i)` 生成同一 token；`S<2` 显式退化 |
| query | 联合置换 target slot 与标签时输出对应置换；共享、独立和错误 query identity 可独立比较 |
| head | 固定总量、可变总量、非负、闭合和结构零分层 |
| 数值 | forward、backward、有限值、checkpoint round-trip、确定性重跑 |
| 公平性 | 参数量、recipe、seed、训练预算、scaler 和允许 split 一致 |
| 统计 | group bootstrap、4/5 seed、逐组分门、最差子组和失败案例 |
| test lock | 旧 test 只读；新 hard-test 和 formal 默认锁定且一次解锁 |

## 10. 停止规则

| 观察 | 动作 |
| --- | --- |
| TQIF 不优于 matched concat | 关闭主机制，不增加更大 Transformer 或 MoE |
| 只有 STR head 改善 | 归因于输出契约，不声称 TQIF 有效 |
| 只有 pair 分支改善 | 收缩为 interaction-only 候选，重新预注册后再评价 |
| 只有 target query 改善 | 收缩为 query-only 候选，不保留无收益 pair 分支 |
| 收益只存在于一个 seed 或一个组分 | 不晋级，保留失败案例 |
| A2 通过但 A2H 失败 | 结论限定为 iid 机制收益，不进入 A2M formal |
| A2H 通过但 A2M formal 失败 | 终态 `MLP_RETAINED`，不进入 A3 主模型 |
| A3 未复现 | 报告泛用性负结果，不回写 A2M 排名或修改 test 门 |
| attention 与移除实验不一致 | 不解释 attention，不影响点预测事实但删除机理声明 |

## 11. 完成定义

本文对应工作完成必须同时满足：

1. TQIF 架构、两档 recipe、matched 矩阵和损失配置机器可读；
2. A2 机制筛选产生唯一候选或明确负结果；
3. 通过 A2 的候选完成 A2H 新版本压力验证；
4. 通过 A2H 的候选完成 A2M 新独立 formal；
5. 历史 A2、A2H v2 和 A2M 第一轮结果保持只读；
6. 只有 A2M `POSITIVE_RESULT` 才更新 A3 参考模型；
7. 结果记录按问题、设计、结果、解读和边界追加，不把 `NO_RESULT` 写成完成；
8. 全量测试、配置 hash、diff 检查和评审记录一致。

## 12. 文献边界

- Set Transformer 的 pooling-by-attention 提供了可学习查询汇总集合的基础，但未针对多组分气体目标槽和传感器对证据设计：[Lee et al., 2019](https://proceedings.mlr.press/v97/lee19d.html)。
- Factorization Machines 说明低秩显式二阶交互可以在受控参数量下表达特征组合，但不是本项目的完整融合结构：[Rendle, 2010](https://doi.org/10.1109/ICDM.2010.127)。
- 已有气体阵列图模型使用图卷积和注意力完成组分识别或浓度估计，TQIF 必须通过 target-query、pair token 和严格组外验证与其区分：[Wang et al., 2025](https://doi.org/10.1109/TIM.2025.3588932)。

上述工作只支撑部件来源和最接近方法定位。TQIF 的新颖性仍需在实现前做专项全文检索；“未检索到同名模型”不能作为新颖性证据。

## 13. 当前实现基线与启动阻断项

本节是继续实施前的事实基线。后续执行者必须先核对本节，不得根据文件名、测试通过或配置存在就推断阶段已经完成。

### 13.1 已有实现

| 类别 | 当前已有内容 | 可确认范围 | 不得推断 |
| --- | --- | --- | --- |
| 模型核心 | `src/gf/dl/tqif.py` | 槽位 ID 查询、传感器注册表、低秩 pair token、容量控制块和诊断开关已实现 | 不代表算法数值有效 |
| 任务头 | `src/gf/dl/task_heads.py` | TQIF 任务头统一由一个 builder 构造，支持 H0、STR 和 variable-total | 不代表任务头在数据上通过晋级门 |
| 配置 | model、train、eval、experiment 下的 TQIF 配置 | 两档 recipe 全字段精确校验，seed、allowlist、bootstrap 和消融矩阵已冻结 | 不代表基线复现已完成 |
| 测试 | TQIF 核心、recipe、梯度、artifact 和 pipeline 测试 | R1 至 R8、T01 至 T12 的契约证据已落盘 | 不代表 A2、A2H 或 A2M 数值验证通过 |
| 执行与产物 | `src/gf/pipeline/a2_tqif_benchmark.py`、`outputs/runs/a2/tqif/`、`outputs/summary/a2/tqif/` | protocol、parameter profile、smoke 和 baseline 阻断证据已生成 | `A2-TQIF-2` 未通过前不得进入完整模型训练 |

R1 至 R8 已闭合。A2M-MLP 在当前 A2 数据上重建为 `rebuild_current_a2_v1` 新锚点，旧 A2M formal 只提供模型和 recipe 身份，不参与数值漂移门。A2-TQIF-3 已形成有效负结果，A2-TQIF-4 按容量停止规则不执行，A2-TQIF-5 已生成唯一 `NEGATIVE_RESULT` 终态。

### 13.2 实施前置审查清单

| ID | 阻断问题 | 已观察风险 | 必须实施的修正 | 关闭证据 |
| --- | --- | --- | --- | --- |
| R1 | pipeline 与协议 hash 缺失 | 无法证明数据、配置、代码和阶段门一致 | 实现 A2 TQIF pipeline、阶段状态机、访问门和 hash manifest | pipeline 测试、`protocol_manifest.json`、失败用例日志 |
| R2 | pair 消融参数不匹配 | C1 对 I1、Q1 对 full 的参数差明显超过 10% | 给无 pair 的控制组增加真实参与前向的容量控制块，并在训练前自动搜索隐藏维度 | `parameter_profiles.json` 中所有主比较差异不超过 10% |
| R3 | target 槽位身份未闭环 | 交换 `target_slot_ids` 可能不改变输出；checkpoint 无槽位身份摘要 | 槽位嵌入或槽位参数必须由 ID 索引；保存并校验有序槽位列表与 hash | 槽位置换测试、checkpoint 契约测试 |
| R4 | 传感器 ID 与 type 可错配 | 两者只各自验证词表成员，错误组合可能被接受 | 建立唯一的 `sensor_id -> sensor_type` 注册表并验证每个样本 | 错配必须显式失败的单元测试 |
| R5 | recipe 张量规格可漂移 | token16 的维度字段可被任意修改仍通过检查 | 对每个 recipe 校验完整字段和值，不只校验名称 | 配置篡改参数化测试 |
| R6 | 任务头存在双实现 | TQIF 内直接构建，同时另有 builder；共享查询控制无法统一表达 | 只保留一个 builder 作为构造事实源，模型通过 builder 注入任务头 | 构造路径测试与代码检索无重复实现 |
| R7 | 普通前向计算诊断量 | 非诊断训练也可能生成 attention 权重 | 以显式 `return_diagnostics` 控制，默认路径不得请求权重 | 普通前向与诊断前向测试、性能冒烟记录 |
| R8 | 正式运行源快照不明确 | dirty worktree 或未版本化文件会让 hash 不可复现 | formal 前要求提交态源码；开发运行记录 commit 与 dirty diff hash | manifest 中 `git_commit`、`git_dirty`、`source_diff_hash` |

关闭顺序固定为：R3、R4、R5、R6、R7 → R2 → R1 → R8。R2 依赖稳定模型结构，R1 依赖最终协议字段；不得倒序用临时 hash 或虚参数填充。

### 13.3 容量控制块设计

无 pair 控制组不得添加未参与前向的 dummy parameter。容量控制块必须对传感器证据执行真实变换：

\[
\tilde e_i=e_i+W_2\phi(W_1\operatorname{LN}(e_i)),
\]

其中隐藏维度 `d_c` 只用于匹配容量，不改变输入输出契约。实现要求如下：

1. `d_c` 由离散候选集合自动枚举，不能在查看验证指标后调整；
2. 搜索目标按“参数差异最小、控制块参数量最小、隐藏维度最小”排序；
3. 参数匹配以可训练参数计数为准，差异定义为

\[
\Delta_P(a,b)=\frac{|P_a-P_b|}{\max(P_a,P_b)}.
\]

4. 主比较 `TQIF-H0↔C0`、`Q1↔C1`、`TQIF-H0↔I1`、`I1↔C1`、`TQIF-H0↔Q1`、`TQIF-STR↔TQIF-H0` 均要求 `Delta_P <= 0.10`；
5. 一旦生成 `parameter_profiles.json`，训练期间不得修改维度；找不到合格维度时阶段状态为 `BLOCKED`，不能放宽阈值；
6. 报告必须同时列出参数量、训练时峰值显存和单批前向耗时，参数匹配不等同于计算量完全匹配。

## 14. 统一执行契约

### 14.1 唯一事实源与优先级

发生冲突时按以下顺序判断：

1. 当前阶段的 `run_manifest.json` 和输入文件 hash；
2. 已冻结且被 manifest 引用的配置文件；
3. 本文定义的门禁、状态机和统计口径；
4. 机器生成的 summary 与 comparison；
5. 人工撰写的 report、结果记录和 README。

低优先级文件不得覆盖高优先级事实。报告中的数字必须由 summary 引用生成，不能手工复制后独立维护。

### 14.2 阶段状态机

每个 stage 只能取一个状态：

| 状态 | 含义 | 是否允许进入下一阶段 |
| --- | --- | --- |
| `NOT_STARTED` | 尚未执行 | 否 |
| `IN_PROGRESS` | 已锁定输入，执行尚未结束 | 否 |
| `PASS` | 执行有效且达到本阶段门 | 是 |
| `NEGATIVE_RESULT` | 执行有效但未达到科学门 | 仅允许进入预定义的负结果收尾 |
| `INVALID` | 数据访问、hash、缺失运行或统计口径无效 | 否，修正后必须重跑受影响阶段 |
| `BLOCKED` | 前置条件或资源未满足，尚未开始有效执行 | 否 |

状态转换只能是 `NOT_STARTED -> IN_PROGRESS -> PASS/NEGATIVE_RESULT/INVALID`，或 `NOT_STARTED -> BLOCKED`。不得把 `INVALID` 改写成负结果，也不得通过补写 summary 把未运行阶段改成 `PASS`。

### 14.3 数据访问矩阵

| 阶段 | 可读数据 | 禁止访问 | 选择可使用的信息 |
| --- | --- | --- | --- |
| A2 开发 | A2 train、OOF、val | A2 旧 test、A2H、A2M、A3 | val 指标、OOF 诊断 |
| A2 收尾 | 已冻结的 A2 开发产物 | A2 旧 test | 机制和容量选择，不做 formal 宣称 |
| A2H 开发 | 新版 A2H train、val、stress-val | 新版 A2H hard-test | val 与 stress-val |
| A2H formal | 已冻结候选、新版 hard-test | 任何解冻或调参 | 一次性评估 |
| A2M 开发 | 新版 A2M train、val、stress-val | 新版 A2M formal-test | val 与 stress-val |
| A2M formal | 已冻结候选、新版 formal-test | 任何解冻或调参 | 一次性评估 |
| A3 | A3 独立文件级划分 | 回流修改 A2M test 门 | 泛用性评价 |

所有读取器必须满足：

- `mixture_id` 是独立样本组标识，不得回退、复制或重写为 `sequence_id`；
- 新正式 benchmark 不依赖 `base_condition_id`、`noise_seed_index` 或 `noise_seed`；
- 组划分在训练前物化为不可变清单，按 ID 检查 train、val、test 零交集；
- pipeline 可以读取冻结的聚合 manifest 或观测文件后按 split allowlist 在内存中过滤，不要求物理文件隔离或文件打开审计；
- 训练器、选择器、预测表或指标一旦接收到禁止 split，立即标记 `PROTOCOL_ACCESS_VIOLATION`，本次 run 为 `INVALID`。

### 14.4 Run manifest 最小字段

每个可计入结论的 run 都必须写出以下结构；字段只能新增，不能在不同阶段改变语义：

    {
      "schema_version": "tqif-run-1",
      "run_id": "A2-TQIF-3__token16__full__seed17",
      "phase": "A2",
      "stage": "A2-TQIF-3",
      "status": "PASS",
      "model_id": "tqif_full",
      "recipe_id": "token16",
      "seed": 17,
      "dataset_manifest_hash": "...",
      "split_manifest_hash": "...",
      "model_config_hash": "...",
      "train_config_hash": "...",
      "eval_config_hash": "...",
      "protocol_hash": "...",
      "target_slot_ids": ["slot_0", "slot_1", "slot_2"],
      "target_slot_hash": "...",
      "sensor_registry_hash": "...",
      "git_commit": "...",
      "git_dirty": false,
      "source_diff_hash": null,
      "started_at": "...",
      "finished_at": "...",
      "artifact_paths": {}
    }

hash 要求：

- 文本和 JSON 先统一 UTF-8、LF；JSON 键递归排序后计算 SHA-256；
- `protocol_hash` 覆盖阶段定义、数据 allowlist、候选集合、seed、阈值和选择规则；
- checkpoint 重载时重新计算 model config、目标槽、传感器注册表 hash，不一致必须失败；
- 开发运行可允许 `git_dirty=true`，但必须记录非空 `source_diff_hash`；
- A2H hard-test 和 A2M formal-test 要求 `git_dirty=false`。

### 14.5 幂等、恢复与并发

1. 相同 `run_id` 与相同输入 hash 已成功完成时，默认拒绝覆盖并返回已有产物；
2. 相同 `run_id` 但任一输入 hash 不同，必须生成新 run ID，不能复用目录；
3. `IN_PROGRESS` run 通过锁文件防止并发写；异常退出保留 manifest 和日志；
4. 恢复只能从已校验 checkpoint 开始，并在 manifest 写明 `resumed_from`；
5. summary 只聚合 `status=PASS` 且产物完整的 run；缺 seed 必须显式列入 `missing_runs`；
6. 禁止静默跳过坏样本、缺文件、NaN 或 checkpoint 不兼容。

## 15. A2 逐步执行手册

A2 的目标不是证明跨分布泛用性，而是在原 A2 数据边界内回答三个问题：

1. target-query 是否比同容量的普通聚合有效；
2. 低秩 pair token 是否提供独立于容量的交互收益；
3. 结构化任务头是否提供独立收益，完整 TQIF 是否值得进入 A2H。

所有 A2 工作包都属于原 A2 验证链，不增设 A2I。

### 15.1 A2-TQIF-0：实现执行 pipeline

#### 15.1.1 影响文件

| 文件 | 动作 | 单一职责 |
| --- | --- | --- |
| `src/gf/pipeline/a2_tqif_benchmark.py` | 新增 | 阶段编排、状态机、访问门、manifest 和退出码 |
| `src/gf/pipeline/tqif_common.py` | 仅确有 A2H/A2M 复用时新增 | hash、run ID、参数统计和完整性检查 |
| `tests/test_a2_tqif_pipeline.py` | 新增 | 阶段顺序、幂等、越界访问和失败传播 |
| `tests/test_tqif_artifact_contract.py` | 新增 | manifest、prediction、summary schema |
| `configs/experiment/a2_tqif_protocol.json`、`a2_tqif_ablation.json` | 修订 | 冻结协议和消融矩阵，不复制 pipeline 状态机 |
| `configs/eval/a2_tqif_eval.json`、`configs/train/a2_tqif_train.json` | 修订 | 冻结指标、bootstrap、seed 和训练预算 |

若现有 pipeline 已有通用 manifest 或原子写入实现，应直接复用；不得为 TQIF 建立第二套语义相同的运行框架。

#### 15.1.2 CLI 阶段

pipeline 必须暴露以下 stage：

| stage | 输入 | 输出 | 成功条件 |
| --- | --- | --- | --- |
| `protocol` | 冻结配置、数据清单、源码状态 | 协议 manifest、参数 profile | R1 至 R8 关闭，hash 完整 |
| `smoke` | 极小训练子集、所有模型变体 | smoke manifests | 前向、反向、保存、重载、预测均成功 |
| `baseline` | A2M-MLP、C0、5 seeds | baseline runs 与 summary | 运行完整且复现偏差合格 |
| `development` | 完整 TQIF 两档 recipe | full-model runs 与 capacity summary | 20 个计划运行完整 |
| `ablation` | 冻结容量、2×2 与 head 变体 | ablation runs 与 comparison | 参数门、运行门和统计文件完整 |
| `select` | 只读 summary 与 comparison | selection manifest | 唯一终态且规则可重算 |
| `all` | 前述全部 | 按依赖顺序执行 | 任一失败即非零退出 |

pipeline 不得捕获异常后写入成功状态。预期协议错误用稳定错误码表示，未预期异常保留堆栈并写入失败日志。

#### 15.1.3 pipeline 验收

- 在没有 `protocol` 通过时调用 `baseline`，应返回 `PREREQUISITE_NOT_PASSED`；
- 训练 batch、预测表和 summary 的 split 审计必须只包含 allowlist 中的 split；不要求专门验证底层文件是否曾被打开；
- 修改已完成 run 的模型配置后复用 run ID，应返回 `RUN_ID_HASH_CONFLICT`；
- 人为删除一个 seed 的 prediction，`select` 必须返回 `INCOMPLETE_RUN_SET`；
- 同一成功 run 重复执行不得覆盖 checkpoint；
- 失败命令退出码必须非零，`stderr` 和 run log 保留根因。

### 15.2 A2-TQIF-1：关闭模型契约阻断项

#### 15.2.1 实施步骤

1. 定义不可变的目标槽注册表，包含有序 `slot_id`、输出列、合法范围和损失权重；A2 适配器固定映射 `slot_0 -> x_Ar_pct`、`slot_1 -> x_He_pct`、`slot_2 -> x_CO2_pct`；
2. target query 由槽位 ID 查表产生；位置只能作为辅助信息，不能成为唯一身份；
3. checkpoint 写入有序槽位清单及 hash，重载时逐项匹配；
4. 定义不可变传感器注册表；每个 `sensor_id` 只映射到一个 `sensor_type`；
5. dataset 到模型边界同时验证 ID、type 和形状；错误组合立即失败；
6. 将两档 recipe 的全部维度、层数、rank、dropout、任务头类型写成精确规格；
7. 将任务头构造收敛到唯一 builder，TQIFModel 只接收构造结果或 builder 配置；
8. 默认前向只返回预测；只有 `return_diagnostics=true` 时返回 attention、pair 强度等诊断；
9. 实现 §13.3 的容量控制块和参数 profile 生成器；
10. 删除被替代的重复实现、死分支和仅用于绕过测试的兼容逻辑。

#### 15.2.2 必须新增或强化的测试

| 测试 ID | 场景 | 预期 |
| --- | --- | --- |
| T01 | 同一输入交换两个 target slot ID | 对应目标输出或查询表示发生可解释置换，不得完全不变 |
| T02 | checkpoint 槽位顺序与当前配置不一致 | 加载失败 |
| T03 | checkpoint 槽位集合缺失或新增 | 加载失败 |
| T04 | 合法 sensor ID 搭配错误 type | 输入校验失败 |
| T05 | 未知 sensor ID 或 type | 输入校验失败 |
| T06 | 篡改 token16 任一固定维度 | recipe 校验失败 |
| T07 | 篡改 token32 任一固定维度 | recipe 校验失败 |
| T08 | 普通前向 | 不生成或返回 attention 权重 |
| T09 | 诊断前向 | 诊断字段完整，主预测与普通前向在 eval 模式一致 |
| T10 | 四组主比较参数 profile | 每组 `Delta_P <= 0.10` |
| T11 | 所有消融模型反向传播 | 所有应训练参数获得有限梯度 |
| T12 | builder 构造三类任务头 | 类型、输出形状和共享参数关系正确 |

此工作包完成后只可写 `A2-TQIF-1=PASS`，不得写“算法验证通过”。

### 15.3 A2-TQIF-2：基线复现

#### 15.3.1 候选与运行数

固定 seed 为 `[17, 29, 43, 71, 101]`。运行：

| 模型 | recipe | seed 数 | 用途 |
| --- | --- | ---: | --- |
| A2M-MLP | 历史正式配置 | 5 | 常规对照和历史锚点 |
| C0 matched concat | token16 | 5 | 小容量结构对照 |
| C0 matched concat | token32 | 5 | 大容量结构对照 |

共 15 个正式开发运行。smoke 不计入这 15 个运行，也不得进入统计。

#### 15.3.2 训练与选择约束

- 使用同一 A2 train/val 清单、归一化器拟合边界、优化器族、最大 epoch 和 early-stop 规则；
- 模型间只允许预注册的 batch size 差异，若因显存调整，记录 effective batch 和梯度累积；
- 每个 seed 独立拟合，不共享最佳 epoch 或初始化；
- checkpoint 只由 val 主指标选择，不读取旧 test；
- 预测必须按样本 ID 排序后落盘，保存原尺度预测和目标；
- 历史 A2M-MLP 若不能按当前协议复现，先停止并调查数据、配置或评估漂移。

#### 15.3.3 基线漂移门

将当前复现与历史只读结果比较：

\[
r_d=\frac{|M_{\mathrm{new}}-M_{\mathrm{hist}}|}{\max(|M_{\mathrm{hist}}|,\epsilon)}.
\]

主指标均值相对偏差原则上不超过 3%，且 5-seed 均值应落在历史可解释波动区间内。若历史产物缺少足够信息，不能伪造区间；应把状态设为 `BLOCKED`，补齐可审计历史依据或明确重建新基线版本。

#### 15.3.4 输出字段

`baseline_summary.json` 至少包含：

- 每个模型、recipe、seed 的 run ID 和 hash；
- val `macro_RNMAE` 主指标，以及 overall MAE、RMSE、R² 辅指标；
- Ar、He、CO₂ 的 `component_RNMAE`、MAE 和 RMSE；
- 参数量、最佳 epoch、训练时长、峰值显存；
- 均值、标准差、95% bootstrap CI；
- 缺失运行、异常运行和历史漂移判定。

### 15.4 A2-TQIF-3：完整模型容量筛选

#### 15.4.1 运行矩阵

| 模型 | recipe | seeds | 运行数 |
| --- | --- | --- | ---: |
| C0 | token16 | 5 个固定 seed | 5 |
| TQIF full | token16 | 5 个固定 seed | 5 |
| C0 | token32 | 5 个固定 seed | 5 |
| TQIF full | token32 | 5 个固定 seed | 5 |

若 A2-TQIF-2 的 C0 运行 hash 与本工作包完全相同，应引用而不是重训；summary 仍需显示其来源。完整矩阵逻辑上为 20 格，物理新增运行可为 10 个。

#### 15.4.2 容量选择顺序

先对每个 recipe 计算 full 相对 C0 的配对 seed 差值，再按以下字典序选择：

1. 是否达到 full 对 C0 的主改善门；
2. median seed 改善是否更大；
3. worst-seed 是否更稳健；
4. 参数量和耗时是否更低；
5. 若前述无实质差异，固定选择 token16。

不得先在全部模型中挑最好单次结果，也不得因 token32 数值略优但不满足预注册实质差异就选大模型。

#### 15.4.3 容量入围门

每个 recipe 的 full 相对 C0 至少满足：

- 5 个 seed 中至少 4 个 `macro_RNMAE` 改善；
- mean relative `macro_RNMAE` improvement ≥ 5%；
- 配对 bootstrap 95% CI 的上界小于 0；
- 任一目标组分 `RNMAE` 的平均退化不超过 0.005 绝对值；
- 参数匹配 `Delta_P <= 0.10`。

两档均不满足时，A2 终态可直接进入 `NEGATIVE_RESULT`，不允许通过扩大模型或增加 seed 搜索挽救。

### 15.5 A2-TQIF-4：2×2 机制消融与任务头消融

#### 15.5.1 冻结后的模型定义

| ID | target-query | pair token | 任务头 | 容量控制 | 回答的问题 |
| --- | --- | --- | --- | --- | --- |
| C0 | 否 | 否 | 独立头 H0 | matched concat 主干 | 普通融合基线 |
| C1 | 否 | 否 | 独立头 H0 | 开启无 pair 容量控制 | I1 的容量对照 |
| Q1 | 是 | 否 | 独立头 H0 | 开启无 pair 容量控制 | target-query 主效应 |
| I1 | 否 | 是 | 独立头 H0 | 无 | pair 主效应 |
| TQIF-H0 | 是 | 是 | 独立头 H0 | 无 | 完整融合机制 |
| TQIF-STR | 是 | 是 | STR | 无 | 结构化任务头增益 |

每个变体均运行 5 个固定 seed。可复用 hash 完全相同的 C0 与 full 运行；其余不得从别的配置复制指标。

#### 15.5.2 预注册对比

| 对比 | 统计量 | 机制归因 |
| --- | --- | --- |
| Q1 − C1 | 配对 seed 的 `macro_RNMAE` 差 | 无 pair 条件下 target-query 收益 |
| TQIF-H0 − I1 | 配对 seed 的 `macro_RNMAE` 差 | 有 pair 条件下 target-query 收益 |
| I1 − C1 | 配对 seed 的 `macro_RNMAE` 差 | 无 query 条件下 pair 收益 |
| TQIF-H0 − Q1 | 配对 seed 的 `macro_RNMAE` 差 | 有 query 条件下 pair 收益 |
| TQIF-STR − TQIF-H0 | 配对 seed 的 `macro_RNMAE` 差 | 任务头收益 |
| TQIF-H0 − C0 | 配对 seed 的 `macro_RNMAE` 差 | 完整融合核心收益 |

target-query 机制至少要在前两个对比中有一个通过；pair 机制至少要在对应两个对比中有一个通过。通过定义为：

- 至少 4/5 seed 的差值方向有利；
- mean relative `macro_RNMAE` improvement ≥ 5%；
- 配对 bootstrap 95% CI 上界小于 0；
- 对应参数差异不超过 10%；
- 任一目标组分平均 `RNMAE` 退化不超过 0.005。

完整融合核心 `TQIF-H0` 对 C0 仍执行 5% 的主改善门。`TQIF-STR` 只决定最终是否采用结构化输出头；任务头若改善但融合机制不通过，只能归因为 head，不得声称 TQIF 融合有效。

#### 15.5.3 统计与诊断

- 以 seed 为配对单元，不能把样本级预测当成独立 seed；
- bootstrap 重采样固定 2,000 次，并在协议中固定 bootstrap seed；
- 同时报告 mean、median、worst-seed 和 5 个原始差值；
- attention 仅用于事后诊断，必须与 token 移除或传感器遮蔽实验交叉验证；
- 诊断不参与候选选择；出现不一致时删除机理解释，不删除点预测结果；
- 不为获得显著性临时增加 seed、切换指标或排除异常 seed。

### 15.6 A2-TQIF-5：唯一候选判定与冻结

选择器只读取已冻结 comparison，按顺序输出一个且仅一个状态：

| 状态 | 必要条件 | 可进入下一阶段的模型 |
| --- | --- | --- |
| `A2_POSITIVE_CANDIDATE` | query、pair、整体门均通过；head 按结果选 H0 或 STR | 冻结的完整候选 |
| `QUERY_ONLY` | query 通过、pair 不通过，且 Q1 对 C0 达到整体门 | 不直接进入 A2H；若继续研究，必须升版协议并回到 A2 重新预注册 |
| `PAIR_ONLY` | pair 通过、query 不通过，且 I1 对 C0 达到整体门 | 不直接进入 A2H；若继续研究，必须升版协议并回到 A2 重新预注册 |
| `HEAD_ONLY` | 只有 STR 对 H0 有收益，融合机制均不通过 | 不进入 A2H；作为任务头对照记录 |
| `NEGATIVE_RESULT` | 运行有效但没有候选达到门 | 无 |
| `INVALID` | 运行缺失、hash 冲突、参数门失败或访问违规 | 无，修复后重跑 |

`selection_manifest.json` 至少包含：

    {
      "schema_version": "tqif-selection-1",
      "phase": "A2",
      "selection_status": "A2_POSITIVE_CANDIDATE",
      "selected_model_id": "tqif_str",
      "selected_recipe_id": "token16",
      "selected_checkpoint_policy": "per_seed_best_val",
      "evidence_comparison_hash": "...",
      "parameter_profile_hash": "...",
      "protocol_hash": "...",
      "allowed_next_phase": "A2H"
    }

冻结动作必须复制配置快照和 hash 引用，不能复制或改写原始预测。A2 结束后不得根据 A2H 表现回头更换 A2 候选。

### 15.7 A2 执行结论（2026-08-31）

A2-TQIF-2 使用当前 A1 train/val 重建 `A2M-MLP / mlp_lbfgs_width32` 锚点，并完成两档 C0 matched concat，共 15 个运行。A2-TQIF-3 完成两档完整 TQIF 的 10 个新增运行，连同复用的 C0 构成 20 个逻辑格。

| 模型 | mean val macro_RNMAE | std | 95% seed bootstrap CI | 参数量 |
| --- | ---: | ---: | ---: | ---: |
| A2M-MLP | 0.005637 | 0.000386 | [0.005329, 0.005961] | 227 |
| C0 token16 | 0.007890 | 0.001124 | [0.006996, 0.008833] | 5,987 |
| TQIF-H0 token16 | 0.034959 | 0.025749 | [0.015157, 0.060317] | 6,033 |
| C0 token32 | 0.006176 | 0.000542 | [0.005730, 0.006653] | 23,123 |
| TQIF-H0 token32 | 0.010993 | 0.002081 | [0.009249, 0.012788] | 22,817 |

token16 相对 C0 退化 343.05%，0/5 seed 改善，paired group bootstrap 95% CI 为 [0.017962, 0.021428]；三个组分 RNMAE 分别退化 0.043234、0.024079 和 0.013893。token32 相对 C0 退化 77.99%，0/5 seed 改善，paired CI 为 [0.003081, 0.004389]；CO₂ RNMAE 退化 0.006066，超过 0.005 上限。两档参数差分别为 0.76% 和 1.32%，容量匹配有效，负结果不能归因于参数量不匹配。执行中发现模型初始权重曾在设置运行 seed 前构造，已改为构造前复位 seed；上述数值来自修复后的确定性批次，旧批次归档在 `outputs/archive/tqif_pre_seed_fix_20260831_r1/`。

两档均未通过 A2-TQIF-3 容量门。按 §15.4.3，A2-TQIF-4 不执行，不通过扩大模型、增加 seed 或追加模块挽救；A2-TQIF-5 的机器终态为 `NEGATIVE_RESULT`，`allowed_next_phase=null`。证据见 [baseline summary](../../outputs/summary/a2/tqif/baseline_summary.json)、[capacity summary](../../outputs/summary/a2/tqif/capacity_summary.json) 和 [selection manifest](../../outputs/runs/a2/tqif/selection_manifest.json)。

### 15.8 失败分析与放弃决定

失败事实不是参数量或单个 seed 造成的：两档 full 与 C0 的参数差均低于 2%，但五个 seed 全部退化，配对置信区间完全大于 0。TQIF token16 的 `macro_RNMAE` 标准差达到 0.025749、最差 seed 为 0.079707，显示明显优化不稳定；token32 虽较稳定，均值仍是 C0 的 1.78 倍。两个 TQIF 配方预测组分和偏离 100 mol% 的平均绝对值约为 4.83 和 1.47 mol%，也明显高于 C0 的 0.44 和 0.50 mol%。

这些证据支持以下解释：当前任务只有三个稳态标量和 840 个训练配方，简单拼接网络已经能够学习主要低维映射；token、pair evidence、两级 attention 与独立 target query 增加了优化自由度，却没有增加观测信息，并削弱了组成输出的一致性。由于容量门失败后未执行 2×2 消融，不能把失败唯一归因于 query、pair 或 H0 head，也不能假设补跑 STR head 可以修复整体结构。

项目决定将 `GF-I14 TQIF-Net` 标记为 `SCIENTIFIC_FAILURE / ABANDONED`。保留实现、配置与产物仅用于复现，不再扩容、延长训练、增加 seed、替换优化器、补跑消融或形成 TQIF-v2。下一候选必须采用新算法 ID 和新预注册计划，从已观测任务瓶颈重新提出机制，不继承 TQIF 的候选身份或成绩。

以下 §16 至 §18 仅保存未执行的历史预注册设计，不构成当前待办，也不得据此启动 TQIF 的后续阶段。

## 16. A2H 逐步执行手册

A2H 只接收 A2 selection manifest 指定的唯一候选，目标是验证该机制在新的组外困难样本上是否仍稳定。A2H 不重新搜索 TQIF 架构，不允许加入第三档 recipe。

### 16.1 所需实现

| 文件 | 动作 | 内容 |
| --- | --- | --- |
| `src/gf/pipeline/a2h_tqif_benchmark.py` | 新增 | A2H 协议、开发矩阵、冻结和 hard-test 门 |
| `configs/experiment/a2h_tqif_protocol.json` | 新增 | 新数据版本、对照集合、seed 和阈值 |
| `tests/test_a2h_tqif_pipeline.py` | 新增 | A2 入门条件、hard-test 锁和一次性运行 |
| `tests/test_tqif_group_split.py` | 新增或扩充 | 新 `mixture_id`、组零交集和禁用字段 |

### 16.2 A2H-TQIF-0：前置条件与协议冻结

执行者逐项完成：

1. 读取 A2 `selection_manifest.json`，校验 selection、protocol 和 parameter profile hash；
2. 仅接受同时通过 query、pair 和整体门的 `A2_POSITIVE_CANDIDATE`；
3. 固定候选模型、recipe、任务头、训练预算和 5 个 seed；
4. 创建全新的 A2H 数据版本和 split manifest；
5. 将 `hard-test` 路径写入锁定区，但开发进程无读取权限；
6. 输出 `a2h_protocol_manifest.json`，状态从 `NOT_STARTED` 变为 `PASS`。

若 A2 为 `QUERY_ONLY`、`PAIR_ONLY`、`HEAD_ONLY`、`NEGATIVE_RESULT` 或 `INVALID`，A2H pipeline 必须拒绝启动，不能由人工指定模型绕过。

### 16.3 A2H-TQIF-1：新数据与 iid parity

新数据版本要求：

- 原始条件、噪声生成和标签文件具有新 dataset version；
- 每个样本使用新的真实 `mixture_id`，与历史 A2H v2 的组 ID 不复用；
- train、val、stress-val、hard-test 按 mixture group 物化；
- 归一化器只在 train 拟合；
- iid 子集应保持与 A2 可比较的物理量纲、目标范围和评估口径；
- 旧 A2H v2 和新 A2H 产物分别只读保存，不覆盖目录。

iid parity 只比较数据难度和基线尺度，不用于挑选候选。若基础模型在 iid 子集出现无法解释的整体漂移，阶段设为 `BLOCKED`，先审查数据生成与评估契约。

### 16.4 A2H-TQIF-2：困难度审计

在训练 TQIF 前，使用冻结的非 TQIF 基线进行困难度审计：

| 审计项 | 判定要求 |
| --- | --- |
| iid 与 stress 差异 | 至少一个预注册 stress 维度使基线 `macro_RNMAE` 明显上升 |
| 目标覆盖 | Ar、He、CO₂ 的合法范围和关键边界样本均有覆盖 |
| group 独立性 | 四个 split 的 `mixture_id` 零交集 |
| 样本量 | 每个 split 达到配置声明数量，无静默丢样 |
| 元数据 | 不依赖三个禁用字段；sensor registry 完整 |
| hard-test 锁 | 审计过程不能打开 hard-test 标签或预测 |

如果 stress-val 对基线不比 nominal-val 更困难，数据版本为 `INVALID`；不得继续并把其称为压力验证。

### 16.5 A2H-TQIF-3：对照与容量复核

固定比较：

1. B3 Ridge；
2. B4 GBDT；
3. `A2M-MLP / mlp_lbfgs_width32`；
4. A2 冻结的 C0 matched concat；
5. A2 selection manifest 指定的完整 TQIF 候选；
6. A2 的 Q1 与 I1，仅作机制诊断，不参与晋级。

每个模型在新版 A2H train 上从头训练，不直接把 A2 checkpoint 当正式结果。每个模型运行 5 个固定 seed。重新生成参数 profile，确认数据输入维度变化没有破坏 10% 容量门。

### 16.6 A2H-TQIF-4：开发评估

开发选择只使用 val 和 stress-val：

- 主指标：各合格压力轴的 stress-val `macro_RNMAE`；
- iid 守门：iid val `macro_RNMAE`；
- 辅助指标：各组分 RNMAE、MAE、RMSE、R²、worst stress slice；
- 稳定性：5-seed median 和 worst-seed；
- 效率：参数量、训练时长、峰值显存；
- 所有指标由同一 prediction schema 聚合。

A2H 开发通过门：

1. iid `macro_RNMAE` 相对 A2M-MLP 的平均退化不超过 5%；
2. calibration、environment、joint、noise 中至少两个通过困难度审计的压力轴，TQIF 相对 A2M-MLP 和 C0 均达到至少 5% 的 mean relative `macro_RNMAE` improvement；
3. 每个用于晋级的压力轴均至少 4/5 seed 改善；
4. 每个用于晋级的压力轴，其 2,000 次 paired group bootstrap 95% CI 上界小于 0；
5. 任一用于晋级的轴、任一目标组分平均 RNMAE 退化不超过 0.005；
6. 最差环境、最差组成区域、composition 诊断轴和失败样本完整报告，不用总体均值掩盖；
7. 参数公平性、训练预算和运行完整性全部通过。

任一主门不满足，输出 `A2H_NEGATIVE_RESULT`，停止在开发阶段，不打开 hard-test。

### 16.7 A2H-TQIF-5：候选冻结

开发通过后执行：

1. 固定唯一模型 ID、recipe、任务头和训练配置；
2. 固定每个 seed 的 checkpoint 选择规则，而非手挑某个 checkpoint；
3. 冻结 hard-test 评估脚本、指标字段和置信区间实现；
4. 保存候选配置、源码 commit、数据和 split hash；
5. 运行 dry-run，只验证文件可访问性和输出 schema，不读取 hard-test 标签；
6. 生成 `a2h_freeze_manifest.json`，由执行者与审查者分别记录确认。

冻结后任何代码、配置、数据或阈值变化都会产生新 protocol hash；旧 freeze manifest 自动失效，不能沿用 formal 次数。

### 16.8 A2H-TQIF-6：一次性 hard-test

hard-test 解锁条件：

- A2H-TQIF-0 至 5 全部为 `PASS`；
- `git_dirty=false`；
- 计划的 5 个 seed checkpoint 全部存在且 hash 匹配；
- `formal_run_status=FROZEN`；
- `hard_test_unlock_count=0`。

解锁后一次生成所有模型预测和 summary，随即把 `hard_test_unlock_count` 写为 1。中途基础设施失败时保留失败证据，由审查记录判断是否属于可重试的 `INVALID`；不能删除记录并伪装为首次。

A2H formal 通过门与开发门保持同一语义：

- iid `macro_RNMAE` 相对 A2M-MLP 的平均退化不超过 5%；
- 至少两个已预注册且通过困难度审计的压力轴，相对 A2M-MLP 和 C0 均改善至少 5%；
- 每个晋级轴至少 4/5 seed 改善，2,000 次 paired group bootstrap 的 95% CI 上界小于 0；
- 每个晋级轴的逐组分 RNMAE 不越过 0.005 退化界；
- worst slice 和失败案例不触发预注册的完整性或安全边界。

满足则输出 `A2H_POSITIVE_RESULT` 并允许进入 A2M；执行有效但不满足则输出 `A2H_NEGATIVE_RESULT`，结论限定为 A2 iid 收益。

## 17. A2M 逐步执行手册

A2M 是进入 A3 前的最终综合对比。它使用新的独立 formal 数据版本，回答 TQIF 相对于常规融合、残差网络、表格 Transformer 和 matched concat 是否仍有综合优势。

### 17.1 所需实现

| 文件 | 动作 | 内容 |
| --- | --- | --- |
| `src/gf/pipeline/a2m_tqif_benchmark.py` | 新增 | A2M 开发矩阵、冻结、formal 和终态 |
| `configs/experiment/a2m_tqif_protocol.json` | 新增 | 对照、预算、数据版本、seed 和门 |
| `tests/test_a2m_tqif_pipeline.py` | 新增 | A2H 入门、对照完整性、formal 锁和终态 |
| `tests/test_tqif_a3_handoff.py` | 新增 | 只有正结果可生成 A3 交接 |

### 17.2 A2M-TQIF-0：候选身份校验

只接受 `A2H_POSITIVE_RESULT`，并验证其：

- selected model、recipe、head 与 A2 freeze 链一致；
- A2H protocol、data、split、checkpoint、comparison hash 完整；
- 没有在 A2H formal 后发生模型结构或选择规则变化；
- formal 次数和最终状态一致。

任一身份不一致都输出 `INVALID`，不得重新指定一个“更好”的 TQIF 进入 A2M。

### 17.3 A2M-TQIF-1：新独立数据

1. 创建不同于历史 A2M 第一轮的新 dataset version；
2. 使用新的 group split，保持 `mixture_id` 真实独立；
3. 物化 train、val、stress-val 和 formal-test；
4. formal-test 标签在冻结前不可读；
5. 对各组分分布、传感器完整性、缺失模式和范围进行质量检查；
6. 记录与 A2、A2H 的分布差异，但不按 TQIF 表现调整数据。

### 17.4 A2M-TQIF-2：smoke 与预算对齐

所有对照先运行同一 smoke：

- 1 个固定 seed；
- 相同小样本清单；
- 完成训练、checkpoint、重载和预测；
- 输出合法 prediction schema；
- 无 NaN、Inf、空组、丢样和标签泄漏。

预算对齐至少统一最大 epoch、early-stop patience、优化器搜索次数和 seed 数。若某模型需要特有学习率区间，应预注册等量搜索预算，不能给 TQIF 更多试验次数。

### 17.5 A2M-TQIF-3：开发矩阵

| ID | 模型 | 角色 |
| --- | --- | --- |
| B3 | 历史 A2 综合表现最佳的传统或深度融合基线 | 历史强基线 |
| B4 | 当前正式 concat 基线 | 主线结构基线 |
| MLP | A2M-MLP | 常规低复杂度对照 |
| RES | ResNet/残差 MLP 正式实现 | 深层表格基线 |
| FTT | FT-Transformer 正式实现 | 注意力表格基线 |
| C0 | 与 TQIF 容量匹配的 concat | 公平容量基线 |
| TQIF | A2H 冻结候选 | 待评价新算法 |

只有仓库中已有正式、可复现实现的模型才能进入矩阵。若 B3、B4、RES 或 FTT 的命名与现有项目不同，配置引用现有正式 model ID，文档别名不形成第二套实现。

每个模型运行 5 个固定 seed，共 35 个开发运行。所有模型在相同 train 上从头训练，在 val 和 stress-val 上评估。

### 17.6 A2M-TQIF-4：开发选择与 formal 冻结

TQIF 必须同时满足：

1. iid `macro_RNMAE` 相对 A2M-MLP 的平均退化不超过 5%；
2. 至少两个合格压力轴相对 A2M-MLP 和 C0 均达到至少 5% 的 mean relative `macro_RNMAE` improvement；
3. 每个用于晋级的压力轴均至少 4/5 seed 改善，2,000 次 paired group bootstrap 的 95% CI 上界小于 0；
4. 每个用于晋级的压力轴，任一组分平均 RNMAE 退化不超过 0.005；
5. 相对 C0 的融合收益方向与 A2、A2H 一致；
6. 全部模型的参数量、预算、失败运行、平均排名和 worst slice 完整报告；
7. 无协议、容量、运行完整性或数据质量问题。

不满足时输出 `A2M_DEVELOPMENT_NEGATIVE`，不打开 formal-test。满足时冻结：

- 全部模型与其训练配置；
- 每 seed checkpoint 规则；
- formal prediction 和评价脚本；
- 排名、非劣和正收益阈值；
- protocol、源码和数据 hash。

### 17.7 A2M-TQIF-5：一次性 formal-test

formal 运行与 §16.8 使用相同锁机制，并额外要求开发矩阵 35 个逻辑格完整。一次性生成全部模型预测，禁止先看 TQIF 再决定是否运行强基线。

formal 正结果要求：

- iid `macro_RNMAE` 相对 A2M-MLP 的平均退化不超过 5%；
- 至少两个合格压力轴相对 A2M-MLP 和 C0 均改善至少 5%；
- 每个晋级轴至少 4/5 seed 改善，2,000 次 paired group bootstrap 的 95% CI 上界小于 0；
- 每个晋级轴的逐组分 RNMAE 不越过 0.005 退化界；
- 相对 C0 的收益仍存在，核心消融方向与开发阶段一致；
- formal 结果未回流 recipe、seed、epoch、门槛或数据范围；
- formal 运行和全部 hash 有效。

### 17.8 A2M-TQIF-6：终态

| 终态 | 条件 | 后续 |
| --- | --- | --- |
| `POSITIVE_RESULT` | formal 有效且全部主门通过 | 生成 A3 handoff |
| `MLP_RETAINED` | formal 有效但 TQIF 未通过；A2M-MLP 或其他正式基线更合适 | 不生成 TQIF A3 handoff |
| `INVALID` | formal 数据访问、hash、运行完整性或统计无效 | 审查后决定重跑，不形成科学结论 |

“排名第二但有创新性”不能替代预注册门。只有 `POSITIVE_RESULT` 才能把 TQIF 写成 A3 参考算法。

## 18. A3 交接与泛用性边界

### 18.1 交接文件

`a3_handoff.json` 只能由 A2M selection pipeline 生成，至少包含：

    {
      "schema_version": "tqif-a3-handoff-1",
      "source_phase": "A2M",
      "source_status": "POSITIVE_RESULT",
      "model_id": "tqif_str",
      "recipe_id": "token16",
      "model_config_hash": "...",
      "checkpoint_policy": "per_seed_best_val",
      "target_slot_hash": "...",
      "sensor_registry_hash": "...",
      "a2m_selection_manifest_hash": "...",
      "allowed_a3_changes": [
        "file_level_input_adapter",
        "time_encoder",
        "VAR_TOTAL"
      ]
    }

若 source status 不是 `POSITIVE_RESULT`，生成命令必须失败。

### 18.2 A3 允许和禁止的变化

允许：

- 为文件级序列增加输入 adapter；
- 增加与 A3 输入契约匹配的时间编码；
- 增加预注册的 `VAR_TOTAL` 输出；
- 重新训练，但保持融合核心和目标查询机制不变。

禁止：

- 根据 A3 test 结果修改 TQIF rank、层数、pair 构造或任务头；
- 将 A3 结果回写为 A2M 算法排名证据；
- 仅对 TQIF 增加额外数据、搜索次数或人工清洗；
- 用 A3 成败改写 A2、A2H、A2M 的既有结论。

A3 至少比较“同一 encoder + concat 聚合”和“同一 encoder + TQIF 聚合”。若 TQIF 未复现优势，结论是泛用性未通过，不是重新开放 A2M 调参。

## 19. 产物目录与数据契约

### 19.1 固定目录

所有路径相对于 `general_fusion/`：

    outputs/
      runs/
        a2/tqif/<run_id>/
        a2h_tqif/<run_id>/
        a2m_tqif/<run_id>/
      summary/
        a2/tqif/
        a2h_tqif/
        a2m_tqif/
      reports/
        tqif/
      archive/

单 run 目录固定包含：

    <run_id>/
      run_manifest.json
      resolved_config.json
      train.log
      metrics.json
      predictions.csv
      checkpoints/
      diagnostics/

`diagnostics/` 可为空，但不能用诊断产物替代主预测。运行目录不得混入聚合 summary；summary 也不得复制 checkpoint。

### 19.2 预测表 schema

`predictions.csv` 一行对应一个样本和一个 seed，字段固定为：

| 字段 | 类型 | 约束 |
| --- | --- | --- |
| `schema_version` | string | 固定 `tqif-prediction-1` |
| `run_id` | string | 与 manifest 一致 |
| `phase` | enum | A2、A2H、A2M 或 A3 |
| `split` | string | 必须在阶段 allowlist 中 |
| `seed` | integer | 与 run 一致 |
| `sample_id` | string | split 内唯一 |
| `mixture_id` | string | 原始真实组 ID，不从 sequence 生成 |
| `y_true_x_Ar_pct`、`y_pred_x_Ar_pct` | float | 原尺度、有限值 |
| `y_true_x_He_pct`、`y_pred_x_He_pct` | float | 原尺度、有限值 |
| `y_true_x_CO2_pct`、`y_pred_x_CO2_pct` | float | 原尺度、有限值 |
| `model_id` | string | 与配置一致 |
| `recipe_id` | string | 非 TQIF 可写 `not_applicable` |

禁止写入 `base_condition_id`、`noise_seed_index` 和 `noise_seed`。如果 A3 增加 `VAR_TOTAL`，通过新 schema version 显式扩展，不改变旧字段含义。

### 19.3 Metrics 与 comparison schema

`metrics.json` 保存单 run 指标：

    {
      "schema_version": "tqif-metrics-1",
      "run_id": "...",
      "split": "val",
      "n_samples": 0,
      "primary": {"macro_RNMAE": 0.0},
      "overall_auxiliary": {"rmse": 0.0, "mae": 0.0, "r2": 0.0},
      "components": {
        "x_Ar_pct": {"RNMAE": 0.0, "rmse": 0.0, "mae": 0.0},
        "x_He_pct": {"RNMAE": 0.0, "rmse": 0.0, "mae": 0.0},
        "x_CO2_pct": {"RNMAE": 0.0, "rmse": 0.0, "mae": 0.0}
      },
      "slices": {},
      "integrity": {
        "prediction_rows": 0,
        "duplicate_sample_ids": 0,
        "missing_sample_ids": 0,
        "non_finite_values": 0
      }
    }

`comparison.json` 保存配对统计：

    {
      "schema_version": "tqif-comparison-1",
      "comparison_id": "TQIF-H0__vs__Q1",
      "metric": "primary.macro_RNMAE",
      "direction": "lower_is_better",
      "left_model_id": "tqif_h0",
      "right_model_id": "q1",
      "paired_seeds": [17, 29, 43, 71, 101],
      "seed_differences": [],
      "mean_difference": 0.0,
      "median_difference": 0.0,
      "relative_improvement": 0.0,
      "bootstrap_ci_95": [0.0, 0.0],
      "favorable_seed_count": 0,
      "parameter_delta": 0.0,
      "gate_checks": {},
      "gate_passed": false
    }

聚合器必须从 predictions 重算 metrics，再从 metrics 生成 comparison。若现有正式评估器已定义 overall 指标，应调用该实现，不在 TQIF pipeline 复制公式。

### 19.4 稳定失败码

| 失败码 | 触发条件 | 阶段状态 |
| --- | --- | --- |
| `PREREQUISITE_NOT_PASSED` | 前一工作包未通过 | BLOCKED |
| `PROTOCOL_ACCESS_VIOLATION` | 读取禁止 split 或字段 | INVALID |
| `RUN_ID_HASH_CONFLICT` | 同 run ID 输入 hash 不同 | INVALID |
| `CHECKPOINT_CONTRACT_MISMATCH` | 模型、槽位或注册表 hash 不一致 | INVALID |
| `PARAMETER_PARITY_FAILED` | 主比较参数差异超过 10% | BLOCKED |
| `INCOMPLETE_RUN_SET` | 缺 seed、预测或指标 | INVALID |
| `NON_FINITE_METRIC` | 预测或指标含 NaN/Inf | INVALID |
| `FORMAL_NOT_FROZEN` | 未冻结就请求 formal | BLOCKED |
| `FORMAL_ALREADY_UNLOCKED` | 重复打开 formal | INVALID |
| `GATE_NOT_PASSED` | 执行有效但科学门未通过 | NEGATIVE_RESULT |

错误码不能替代具体错误信息；日志必须同时给出 stage、run ID、相关路径和不变量。

## 20. 命令、验证层级与严格执行顺序

### 20.1 当前可执行的验证

在 `general_fusion/` 为工作目录执行：

    python -m pytest -q tests/test_tqif_core.py tests/test_tqif_gradients.py tests/test_tqif_task_heads.py tests/test_tqif_recipe.py
    python -m pytest -q

第一条用于模型局部回归，第二条用于项目全量回归。代码修改后还应执行：

    git diff --check
    git status --short

`git diff --check` 不会检查未跟踪文件；未跟踪的 TQIF 文件必须先单独审阅，formal 前纳入版本控制。

### 20.2 计划中的 CLI

以下命令是待 §15 至 §17 pipeline 实现后的验收接口，当前模块不存在时必须显式失败，不能建立 mock success：

    python -m gf.pipeline.a2_tqif_benchmark --stage protocol --project-root .
    python -m gf.pipeline.a2_tqif_benchmark --stage smoke --project-root .
    python -m gf.pipeline.a2_tqif_benchmark --stage baseline --project-root .
    python -m gf.pipeline.a2_tqif_benchmark --stage development --project-root .
    python -m gf.pipeline.a2_tqif_benchmark --stage ablation --project-root .
    python -m gf.pipeline.a2_tqif_benchmark --stage select --project-root .

    python -m gf.pipeline.a2h_tqif_benchmark --stage protocol --project-root .
    python -m gf.pipeline.a2h_tqif_benchmark --stage difficulty-audit --project-root .
    python -m gf.pipeline.a2h_tqif_benchmark --stage development --project-root .
    python -m gf.pipeline.a2h_tqif_benchmark --stage freeze --project-root .
    python -m gf.pipeline.a2h_tqif_benchmark --stage formal --project-root .

    python -m gf.pipeline.a2m_tqif_benchmark --stage protocol --project-root .
    python -m gf.pipeline.a2m_tqif_benchmark --stage smoke --project-root .
    python -m gf.pipeline.a2m_tqif_benchmark --stage development --project-root .
    python -m gf.pipeline.a2m_tqif_benchmark --stage freeze --project-root .
    python -m gf.pipeline.a2m_tqif_benchmark --stage formal --project-root .
    python -m gf.pipeline.a2m_tqif_benchmark --stage handoff --project-root .

CLI 参数只能选择已经定义的 stage 和项目根；模型集合、seed、阈值和 split 不应通过临时命令行覆盖冻结配置。

### 20.3 每次代码变更的验证层级

| 变更范围 | 最小验证 |
| --- | --- |
| TQIF 核心、任务头 | T01 至 T12、现有四个专项测试 |
| 配置或 recipe | recipe 篡改测试、resolved config snapshot、hash 测试 |
| 数据读取和 split | group 零交集、禁用字段、错配传感器和访问门测试 |
| pipeline | 对应 pipeline 测试、失败码、幂等和中断恢复 |
| 聚合与统计 | 人工小样本 fixture、手算配对差值、CI 稳定性 |
| formal 锁 | 未冻结拒绝、重复解锁拒绝、dirty source 拒绝 |
| 任一跨模块改动 | 全量测试和 diff 检查 |

### 20.4 不得改变的执行顺序

    R1-R8 阻断项关闭
      -> A2-TQIF-0/1
      -> A2-TQIF-2 基线
      -> A2-TQIF-3 容量
      -> A2-TQIF-4 消融
      -> A2-TQIF-5 选择
      -> A2H-0..4 开发
      -> A2H-5 冻结
      -> A2H-6 hard-test
      -> A2M-0..3 开发
      -> A2M-4 冻结
      -> A2M-5 formal-test
      -> A2M-6 终态
      -> A3 handoff

任一箭头左侧不是 `PASS` 或明确允许的 positive selection，右侧都不得开始。

## 21. 供后续模型使用的工作包模板

后续模型每次只领取一个最小工作包。开始前必须在回复中给出以下结构：

### 21.1 开始前结构

#### Context

- 当前阶段与 stage 状态；
- 已通过的前置工作包；
- 本次允许读取的 split；
- 本次明确禁止读取的 split 和字段；
- 当前 commit、dirty 状态和关键 protocol hash；
- 本工作包涉及的源文件、配置和测试。

#### Task

- 本次只解决的根因或实验问题；
- 要维持的不变量；
- 预计修改文件；
- 预期生成产物；
- 成功门与失败状态；
- 不在本次范围内的后续阶段。

#### Validation

- 先运行的目标测试；
- 再运行的相关集成测试或全量测试；
- 产物 schema、hash、运行数和数据交集检查；
- diff 中需要人工复核的高风险点。

#### Format

完成回复必须依次报告：

1. stage 最终状态；
2. 修改文件及职责；
3. 生成产物及 hash；
4. 实际运行命令和结果；
5. 门禁逐项判定；
6. 未解决风险；
7. 唯一允许的下一动作。

### 21.2 标准工作包示例

    Context
    - 阶段：A2-TQIF-1，前置 A2-TQIF-0=PASS。
    - 可读：A2 train/val fixture；禁止：A2 old test、A2H、A2M。
    - 不变量：mixture_id 不回退；三个禁用字段不参与 benchmark。

    Task
    - 关闭 R3、R4、R5。
    - 修改 tqif.py、recipe validator 和对应测试。
    - 不运行候选训练，不生成数值结论。

    Validation
    - 运行槽位、sensor registry、recipe 参数化测试。
    - 运行四个 TQIF 专项测试和全量测试。

    Format
    - 报告三个阻断项的 PASS/INVALID 状态、测试数和下一步。

示例只校准格式，不授权跳过 §13.2 的关闭顺序。

### 21.3 失败处理

| 场景 | 执行动作 | 禁止动作 |
| --- | --- | --- |
| 测试失败 | 保留错误和堆栈，定位根因，stage 不通过 | 吞错、xfail 掩盖、删除断言 |
| 运行中断 | 保留 IN_PROGRESS manifest 和日志，按恢复契约处理 | 删除目录后声称首次运行 |
| 缺失输入 | 标记 BLOCKED 并列出确切文件/hash | 生成占位数据或默认路径 |
| 指标为负结果 | 写 NEGATIVE_RESULT 和完整证据 | 临时调阈值、增加模型容量 |
| 协议无效 | 写 INVALID，确定受影响的最早阶段 | 把 INVALID 合并进均值 |
| 文档与代码冲突 | 以 manifest、冻结配置和代码事实核查，修正文档 | 仅修改报告数字使其一致 |

## 22. 最终验收清单

### 22.1 实现验收

- [x] R1 至 R8 均有代码、测试和产物证据；
- [x] target slot 身份参与计算并进入 checkpoint 契约；
- [x] sensor ID 与 type 使用唯一注册表配对验证；
- [x] 两档 recipe 精确校验；
- [x] 任务头只有一个构造事实源；
- [x] 普通前向不计算诊断权重；
- [x] 所有已执行主比较的参数差异不超过 10%；
- [ ] A2、A2H、A2M pipeline 均有失败传播、幂等和访问门测试。

最后一项保留未勾选：A2 pipeline 已实现并验证；A2H 和 A2M 因 A2 失败不得进入，也未实现 TQIF 专用 pipeline。本计划现为失败归档，不再补齐该项。

### 22.2 A2 验收

- [x] 15 个基线逻辑运行完整或有 hash 一致的合法复用；
- [x] 两档 full 对 C0 容量筛选完整；
- [ ] 2×2 机制和任务头消融 5-seed 完整；N/A：两档容量门均失败，按 §15.4.3 必须停止，A2-TQIF-4 不执行；
- [x] comparison 可由 predictions 重算；
- [x] selection manifest 只有一个终态；
- [x] A2 旧 test 未进入训练、选择、预测或指标；允许读取冻结全集后按 split 逻辑过滤；
- [x] 没有新增 A2I。

### 22.3 A2H 验收

本节为 N/A：A2 没有唯一正候选，按阶段门不得进入 A2H；未执行不构成阻塞或缺失实验。

- [ ] 只接收 A2 唯一候选；
- [ ] 使用新 dataset version 和新 group split；
- [ ] 困难度审计在 TQIF 开发前完成；
- [ ] val、stress-val 开发与 hard-test 严格隔离；
- [ ] 冻结 manifest 完整；
- [ ] hard-test 解锁不超过一次；
- [ ] 最终状态为正结果、负结果或 INVALID 之一。

### 22.4 A2M 与 A3 验收

本节为 N/A：A2 已在容量门冻结负结果，A2M 与 A3 均无合规输入；未执行不构成阻塞或缺失实验。

- [ ] 只接收 A2H 正结果；
- [ ] 35 个开发逻辑运行完整；
- [ ] 新 formal-test 在冻结前未访问；
- [ ] 全部强基线与 TQIF 一次性 formal；
- [ ] 终态唯一；
- [ ] 只有 `POSITIVE_RESULT` 生成 A3 handoff；
- [ ] A3 不回写或重开 A2M 选择。

### 22.5 文档状态更新规则

本计划已经失败归档，不再追求 `IMPLEMENTED` 或 `SELECTED_FOR_A3`。A2 机器产物保持 `NEGATIVE_RESULT`，项目决策固定为 `SCIENTIFIC_FAILURE / ABANDONED`；A2H、A2M 与 A3 条目均为未进入的历史设计。数值结果必须继续链接对应 summary、comparison 和 selection manifest，后续新算法不得覆盖或复用 TQIF 的终态。
