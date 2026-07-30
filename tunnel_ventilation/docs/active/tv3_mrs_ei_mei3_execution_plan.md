# MEI-3 确定性 VarPro 后续执行计划

> 状态：B5 已完成并关闭 MEI-3，verdict=`mei3_full_parameter_baseline_retained`  
> 前置：Phase A=`mei3_phase_a_structure_supported`；B3 protocol-2 方案 A 已重冻；pre-B4=`mei3_pre_b4_technical_ready`  
> 当前阶段：`b5_verdict_freeze` / MEI-3 已关闭  
> 结论范围：`registered_simulation_domain_only`  
> 上位计划：[tv3 MRS-EI 实验计划](tv3_mrs_information_efficient_inversion_experiment_plan.md)  
> Phase A 证据：[MEI-3 Phase A 执行报告](../archive/completed/tv3_mrs_ei_mei3_phase_a_execution_report.md)  
> B0 结论：本文 §11。无约束三维 `raw3` 依然无效；登记的干基闭包物理域在二维切空间中满秩。  
> 授权记录：`stage_status.mei3.authorizations.registered_sparse_simulation_generation=authorized`（其余三项仍禁止）。  
> 当前 B4 freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T120958962354Z_cf7ed57312d9`；manifest SHA256=`604a5fe6a26c51963b8b5197748002b77ad2177461ff11c3bc5e7cd174f747d8`。  
> 当前 B5 freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260730T011247690033Z_f1246e54ccb0`；manifest SHA256=`a2b2ce51322e0420971d8503ba61a26c01c179486e5bc6ae15f5af3b22910be5`；MEI-4 基线=`S1`。  
> 历史作废 freeze（未 join 已知 T/L/RH 条件先验）：`.../20260729T115530120687Z_4a32b504e5fe`，不得引用其求解门指标。  
> 并行旁证 freeze（条件先验已修复，但被后完成的有效 freeze 覆盖 stage_status）：`.../20260729T120817575667Z_cf7ed57312d9`，manifest=`fe34a54f51d4a0d922a10bc377540e21064eb80f75ca7a82064c63e3abd10026`；不得替换有效 freeze。`parent_pre_b4_manifest_sha256` 已校正为 `94bdd50d…b2a9`。  
> B4 复核报告（2026-07-29）：[MEI-3 B4 代码复核与结果再分析](../archive/completed/tv3_mrs_ei_mei3_b4_review_and_analysis_report.md)——独立复算确认主指标与 CI 判定；机制归因：P90 改善全部来自共享 `max_iterations` 样本，S2 为同解省算力优化器；B5 契约注记建议见其 §10。

---

## 1. Context：当前已知道什么

MEI-1 已固定 D0 K4 `{25,63,100,200}` kHz 并跳过 MEI-2。MEI-3 Phase A 已完成条件线性结构审计：

- 公共延迟在 `raw_tof_s` 与固定整数分支的 `unwrapped_phase_rad` 中为仿射项，准入时必须在 `device_profile_id × view_id` 上跨样本共享；
- 对数幅度增益在 `log_amplitude` 中为仿射项，准入时必须在 `device_profile_id × view_id` 上跨样本共享；
- 逐频标定偏移只有在 `device_profile_id × frequency_hz` 上跨样本共享并带独立先验时才准入；
- 声程 `L` 按当前联合观测政策留在非线性块；
- 复传递函数实部与虚部中的延迟不具有条件线性，不准入。

36 个内存观测、6 个线性参数的数值 fixture 中，VarPro 解与联合正规方程参考的最大参数差为 `1.0408340855860843e-17`，最大增广残差差为 `7.105427357601002e-15`。这只证明内层线性求解正确，尚未证明 S2 优于 S1。

### 1.1 B0 结构缺口的处置

1. **原始观测算子已建立。** `mrs_observation.py` 是 `raw_tof_s + log_amplitude + unwrapped_phase_rad + covariance` 的单一来源；行顺序为频率优先，单位为秒、无量纲、弧度，相位整数分支必须显式传入。
2. **`raw3` 物理域已闭合。** 三列输入和输出保持 `out_dim=3`，但干基定义要求非负且 `sum=100%`，有效物理维数为 2。求解器从第一步起在正交 `sum-zero` 切空间中联合更新三个组分，不回填 N2，不做事后投影，非闭包输入在调用前向前显式失败。

---

## 2. Task：本计划要回答什么

唯一核心问题是：

> 在观测、噪声、初值和评价样本全部相同时，S2 是否比 S1 更稳定、更高效地利用固定 K4 信息？

本计划不优化频点，不训练神经网络，不估计后验覆盖率，不生成正式波形，不做硬件或现场能力声明。

---

## 3. 全程不变量

1. 固定 D0 K4 `{25,63,100,200}` kHz，不恢复 MEI-2。
2. 点估计仍为 `raw3` 和 `out_dim=3`；运行方法 S1--S3 从求解开始即强制干基非负与 `sum=100%` 物理域。不做 N2 闭包回填、`target_transform`、静默归一化或求解后单纯形投影。
3. `mixture_id` 是组分主键；不回退或重写为 `sequence_id`。新比较不依赖 `base_condition_id`、`noise_seed_index` 或 `noise_seed`。
4. S1--S3 必须使用同一前向观测算子、协方差、边界、先验、初值集和停止条件。
5. S3 可读取真值干扰参数，但只作上限审计；S1/S2 不得读取任何仿真内部真值。
6. 所有标准化、标定先验和超参数只能由训练集或独立标定集确定。
7. 失败必须暴露为异常、非零退出、失败测试或冻结失败 verdict；不吞错，不自动换求解器。
8. 当前四项授权全部保持 `forbidden_until_explicit_authorization`。代码和合成单元验证不能被写成正式数据生成。

---

## 4. 求解器比较矩阵

| ID  | 内容                   | 用途             | 是否可成为后续基线 |
| --- | -------------------- | -------------- | --------- |
| S1  | 全参数尺度化阻尼高斯--牛顿       | VarPro 的直接机制对照 | S2 未通过时保留 |
| S2  | 条件线性投影 + 尺度化阻尼高斯--牛顿 | 主候选            | 只有通过求解门后  |
| S3  | S2 + 已知真值干扰参数        | 仿真上限和误差归因      | 否         |

S0 不属于运行矩阵。旧 MRS-5 H1 只是“弛豫重建 + Ridge”的未执行设计，状态冻结为 `historical_h1_not_instantiated`，只作非运行历史说明。未来若实现类似方法，必须使用新的独立 method ID，不得称为现有历史 S0。S1 必须在查看 S2 正式结果之前冻结为唯一主对照，不允许事后更换基线。

---

## 5. 分阶段执行路径

### B0：表示与前向算子闭合

> 执行状态（2026-07-29）：**通过**。观测算子契约通过；5 个登记点的二维切空间雅可比在三档容差下均为满秩 2。详见 §11。

**目标**：在编写 S1/S2 前，先确保非线性问题本身有唯一且一致的含义。

1. 在共享 MRS-1 前向之上建立单一观测算子，输出理想登记仿真域的 TOF、对数幅度和已解缠相位及其协方差。
2. 明确相位整数分支来自独立预处理；未知分支时直接判为当前 S2 不适用。
3. 同时保留无约束三维零方向证据，并对登记的二维正交切空间计算白化雅可比和奇异值。
4. 三个组分必须通过仿射切空间映射联合生成；不得偷换成 N2 闭包回填、静默归一化或事后单纯形投影。

**产物**：`mei3_observation_operator_audit.json`、`mei3_raw3_forward_rank_audit.json`。

**通过条件**：观测单位、符号、协方差和相位分支全部可复算；非闭包输入显式失败；`raw3` 的登记二维有效参数化在全部容差下满秩。

### B1：S0 历史处置、S1 和数值尺度化

> 执行状态（2026-07-29）：**通过**，verdict=`mei3_b1_s1_frozen`。S0=`historical_h1_not_instantiated`，不进入运行矩阵；S1 尺度化阻尼高斯--牛顿核心、梯度对照、尺度不变性与三个冻结初值的合成数值审计已冻结。freeze=`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T072036945412Z_646ad3f1c878`；manifest SHA256=`d6af49868c317fff04609e191276dd3758caeb646ff23e6809e7e1b1eabe2e84`。该结果只属于内存合成数值验证，不是正式求解门证据。

1. 冻结 S0 的未实例化历史处置，不实现或替换第二套 H1 公式。
2. 实现 S1 的统一残差、参数边界、先验、阻尼、线搜索和停止条件。
3. 对秒、弧度、百分数、摄氏度和米建立显式尺度。Phase A 原始单位增广条件数 `5.926e5` 作为诊断基准。
4. 冻结多初值集。初值只依赖允许的观测、训练标定或登记常数，不读取真值。

**产物**：S0 历史处置、S1 单元测试、梯度对照、尺度不变性报告、多初值收敛报告。

### B2：S2/S3 核心与机制验证

> 执行状态（2026-07-29）：**通过**，verdict=`mei3_solver_core_verified`。投影雅可比、S1/S2 合成等价性、S3 显式真值干扰入口及四项负对照均通过。freeze=`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T081421139186Z_c0ade3f5df14`；manifest SHA256=`ba8bafe0e903f2efc6d6cca8c081d479a98071749d4cdf32674f365d2e3ea8eb`。该结果仍是内存合成机制验证，不是正式求解门证据。

1. S2 每次非线性迭代都调用 Phase A 已验证的带先验条件线性求解器。
2. 实现投影残差对 `beta` 的雅可比，并与高精度中心差分逐点对照。
3. 在相同合成问题上验证 S1 和 S2 的最优目标值与组分解一致。
4. S3 只固定干扰参数真值，其他前向、噪声和初值与 S2 相同。
5. 添加负对照：错误相位分支、每样本无约束 K4 偏移、非正定协方差、将声程强行放入线性块。所有负对照必须失败。

**技术状态**：`mei3_solver_core_verified` 或 `mei3_solver_core_invalid`。技术通过不等于求解门通过。

### B3：正式数据授权就绪包

> 执行状态（2026-07-29）：**通过（protocol-2 重冻）**，verdict=`mei3_registered_data_authorization_ready`。方案 A 已写入 `view_nuisance_calibration_priors` 与 `view_protocol`。当前 freeze=`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T104111344740Z_b435a5b57d0f`；manifest SHA256=`7badf39329a7b77cffbdaa862dc330add956e262b520ea30e8adc221f0579557`。初始 freeze `.../20260729T090738478475Z_073bc6489d9e` 仅作历史证据。该状态只表示可以提交独立授权评审，不是数据生成授权。

在申请 `registered_sparse_simulation_generation` 之前，必须冻结：

- 原始观测字段、dtype、shape、单位和协方差布局；
- `mixture_id`、`device_profile_id`、`view_id` 和观测行 ID 契约；
- `view_protocol`：当前登记稀疏域冻结 `views_per_mixture=1`、`registered_view_ids=["view_0"]`，且 `view_id` 跨 `mixture_id` 复用；`replicates_per_condition` 只产生不同 `mixture_id`，不增加视图数；
- 独立标定集与两类共享层级：逐频偏移在 `device_profile_id × frequency_hz`；公共延迟与对数幅度增益在 `device_profile_id × view_id`。评价域策略为标定集估计后验，再 join 为 S1/S2 先验（方案 A），禁止每样本无约束估计；
- train/test/OOD 划分单元实际冻结为 `calibration/test/ood`，同一 `mixture_id` 的所有视图必须在同一划分；
- 样本量与功效计算；不允许因为计算方便事后减小样本数；
- 三个数据划分随机种子、冻结多初值集、`2000` 次分层配对 bootstrap；
- 每个运行求解器读取字段白名单，S3 真值字段必须与 S1/S2 物理隔离；S1/S2 必须能读取偏移与 view-nuisance 标定先验字段；
- schema、配置、划分、生成器、求解器和指标实现的 SHA256（不含会随文档改写而漂移的 execution_plan）。

就绪包输出 `mei3_registered_data_authorization_ready` 或列出具体缺口。只有独立授权记录把 `registered_sparse_simulation_generation` 改为已授权后，才能进入 B4。其他三项授权不随之改变。B3 通过后 `allowed_next_stage=null` 是刻意的单向停点：在独立数据授权前不允许自动推进。

### B4：正式 S1--S3 配对比较

> 执行状态（2026-07-29）：**通过执行并冻结**，verdict=`mei3_full_parameter_baseline_retained`。S2 未通过正式求解门；后续基线保留 S1。当前 freeze=`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T120958962354Z_cf7ed57312d9`；manifest SHA256=`604a5fe6a26c51963b8b5197748002b77ad2177461ff11c3bc5e7cd174f747d8`。详见 §15。

1. 生成一次冻结的登记稀疏观测，S1--S3 对每个观测行做精确配对；主比较仅为 S1/S2，S3 只作上限审计。
2. 先完成并冻结 S1，再运行 S2；S3 最后运行，只用于解释干扰参数造成的可恢复误差。
3. 确定性求解器不报告“训练种子”；改为报告冻结初值索引与数据划分种子。
4. 收敛失败保留在主指标中，不允许只对成功样本计算 P90。
5. 观测表中的 `T_C` / `L_m` / `H_RH` 作为 `solver_allowed_condition`，必须按混合物 join 为先验均值与初值；不得使用 B1 fixture 的全局 25°C / 0.25 m / 50% RH 先验覆盖全网格。

### B5：裁决与冻结

B5 必须将 B4 的冻结裁决写入版本化 MEI-3 执行契约，不重跑或改写 B4 freeze：

> 执行状态（2026-07-30）：**已完成并冻结**。B5 已验证 B4 manifest 后写入独立 freeze `20260730T011247690033Z_f1246e54ccb0`，verdict=`mei3_full_parameter_baseline_retained`，MEI-4 基线=`S1`；B5 未生成数据，`allowed_next_stage=null`，因为 MEI-4 尚无独立的登记执行契约。

| 状态                                           | 条件                      | 后续基线   |
| -------------------------------------------- | ----------------------- | ------ |
| `mei3_varpro_supported`                      | S2 在 test 和 OOD 同时通过求解门 | S2     |
| `mei3_full_parameter_baseline_retained`      | 结构适用，但 S2 未通过求解门        | S1     |
| `mei3_solver_core_invalid`                   | 观测、`raw3` 或数值不变量未闭合     | 无，停止   |
| `mei3_waiting_registered_data_authorization` | 核心通过，但正式数据仍未授权          | 不作科学裁决 |

Phase A 已证明结构适用，因此后续“S2 可实现但性能无改善”不得写成 `mei3_varpro_not_applicable`。新增 `mei3_full_parameter_baseline_retained` 前，必须以子阶段契约明确它的条件与 S1 后续语义；不回写历史 MEI-0/1 freeze。

#### B4 结果解释约束

1. S2 的 P90 改善全部来自双方共同 `max_iterations=100` 的样本；它表示固定迭代预算内更快的优化进展，不构成“统计效率更高的估计器”主张。若改变迭代上限或停止判据，必须建立新契约并重新比较。
2. 收敛失败仍按该次求解得到的有限估计计算主指标；只有估计非有限时才使用 `failure_abs_error_percent`。因此 B4 的失败率表示紧停止判据下的慢收敛，不等于不可用输出比例。
3. 不得引用 `mean_relative_crb_efficiency_o2`。CRB 诊断仅可引用逐样本相对效率的中位数，或先分别聚合 `crb_o2_std_percent^2` 与 `abs_err_o2^2` 后相除的比值。
4. 当前 B4 freeze 的 `forward_calls` 漏计 Jacobian 中心预测；该 freeze 的真实物理前向调用比约为 S1:S2 = 1.5:1，而非记录的约 2.1:1。修正后的代码只用于未来新契约复跑，不能回写当前 freeze。

---

## 6. 正式求解门

### 6.1 主指标

S2 相对冻结 S1 必须同时满足：

1. test 的 O2 配对 P90 相对改善 `> delta_practical=0.02`；
2. OOD 的 O2 配对 P90 相对改善 `> 0.02`；
3. test 和 OOD 的分层配对 95% 置信区间下界均 `> 0.02`。

### 6.2 非退化门

以下任一项的绝对增量超过其 `solver_gate.non_degeneration.thresholds` 阈值，S2 均不通过。组分误差单位为 pp，失败率与边界命中率单位为比例点：

- O2、CO2、N2 MAE；
- 收敛失败率；
- 最坏组误差；
- 边界命中率或局部极小值率。

### 6.3 机制诊断

以下指标完整报告，但不能替代主指标：

- 相对 CRB 效率；
- 迭代次数、前向调用次数与墙钟时间；
- 原始与尺度化条件数；
- 多初值解分散度和局部极小值率；
- S3 与 S2 的误差差，用于估计干扰参数的可恢复上限。

---

## 7. 计划代码与产物边界

### 7.1 代码

| 范围          | 计划文件                                                                                                                                                                                                                                                                                                   |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 观测算子        | `tv3/sim/generation/tunnel_ventilation/mrs_observation.py`                                                                                                                                                                                                                                             |
| S1/S2/S3    | `tv3/ml/mrs_varpro.py`                                                                                                                                                                                                                                                                                 |
| 求解门与统计      | `tv3/audit/mrs_ei_solver_gate.py`                                                                                                                                                                                                                                                                      |
| B4 正式比较     | `tv3/audit/mrs_ei_b4_formal.py`                                                                                                                                                                                                                                                                        |
| 运行契约        | `configs/tv3_mrs_ei/mei3_solver_audit.json`                                                                                                                                                                                                                                                            |
| 数据授权就绪包     | `configs/tv3_mrs_ei/mei3_solver_data_protocol.json`                                                                                                                                                                                                                                                    |
| B4 授权门控契约   | `configs/tv3_mrs_ei/mei3_b4_formal_gate.json`                                                                                                                                                                                                                                                          |
| B5 裁决契约     | `configs/tv3_mrs_ei/mei3_b5_verdict_contract.json`                                                                                                                                                                                                                                                     |
| B0 执行入口     | `scripts/run_tv3_mei3_varpro_audit.py`                                                                                                                                                                                                                                                                 |
| B1 执行入口     | `scripts/run_tv3_mei3_b1_solver_audit.py`                                                                                                                                                                                                                                                              |
| B2 执行入口     | `scripts/run_tv3_mei3_solver_audit.py`（已实现并冻结）                                                                                                                                                                                                                                                         |
| B3 执行入口     | `scripts/run_tv3_mei3_b3_data_readiness.py`（已实现并冻结）                                                                                                                                                                                                                                                    |
| pre-B4 执行入口 | `scripts/run_tv3_mei3_pre_b4_technical_audit.py`（已实现并冻结）                                                                                                                                                                                                                                               |
| B4 执行入口     | `scripts/run_tv3_mei3_b4_formal_comparison.py`（已实现；授权后正式生成与配对比较）                                                                                                                                                                                                                                       |
| B5 执行入口     | `scripts/run_tv3_mei3_b5_verdict_freeze.py`（已实现并冻结 B5 裁决契约）                                                                                                                                                                                                                                            |
| 测试          | `tests/test_tunnel_ventilation_mei3_varpro.py`、`tests/test_tunnel_ventilation_mei3_solver.py`、`tests/test_tunnel_ventilation_mei3_b3_readiness.py`、`tests/test_tunnel_ventilation_mei3_pre_b4.py`、`tests/test_tunnel_ventilation_mei3_b4_formal.py`、`tests/test_tunnel_ventilation_mei3_b5_verdict.py` |

共享弛豫谱公式仍只存在于现有 `relaxation_spectrum.py`；观测算子、S1 和 S2 不得各自复制一份。

### 7.2 产物

```text
outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/<freeze_id>/
  mei3_solver_run_config.json
  s0_historical_disposition.json
  s1_parameter_scale_report.json
  s1_gradient_report.json
  s1_multi_initialization_report.json
  s1_scale_invariance_report.json
  mei3_b1_verdict.json
  # B2--B5 追加对应阶段 freeze，而不覆盖 B1：
  solver_comparison.csv
  convergence_report.json
  crb_efficiency.csv
  bootstrap_report.json
  mei3_verdict.json
  # B5 only:
  mei3_b5_verdict_contract.json
  mei3_b5_verdict.json
  parent_b4_manifest.json
  parent_b4_verdict.json
  evidence_manifest.json
  source_snapshots/
```

阶段目录只追加。汇总文档只引用 freeze 路径和 manifest SHA256，不手工重录一份可编辑指标表。

---

## 8. 验证矩阵

### 8.1 单元与数值测试

- 观测算子单位、符号和 shape 契约；
- `raw3` 前向零空间与秩审计；
- 解析 / 自动 / 有限差分雅可比一致性；
- S1/S2 在可分离问题上的目标值与解一致；
- 参数单位重缩放不改变物理解；
- 边界、阻尼、线搜索和停止原因可追踪；
- 非正定协方差、错误相位分支、无先验逐频偏移和禁止真值输入显式失败；
- 两个独立新进程复算的数值一致性。

### 8.2 回归测试

- MEI-0/1/3 Phase A 专项回归；
- MRS physics、identifiability 与 MRS-6 budget 回归；
- 三个有效上游 manifest 独立验证；
- 四项授权在未获独立批准时始终保持禁止。

---

## 9. 执行顺序与停止条件

```text
B0 观测与 raw3 表示审计
  → B1 S0 历史处置 + S1 冻结 + 尺度化
  → B2 S2/S3 + 合成机制验证
  → B3 正式数据授权就绪包
  → [独立授权决策]
  → B4 正式 S1--S3 配对比较（S1/S2 主比较，S3 上限）
  → B5 裁决与冻结
```

任一条触发即停止当前分支：

1. 求解器接受非闭包 `raw3`、在三维无约束空间迭代，或重新引入尺度零方向；
2. 相位分支无法从允许输入独立确定；
3. S1/S2 在可分离单元问题上不等价；
4. 尺度化后物理解改变，或雅可比与有限差分不一致；
5. 正式数据授权未获批却试图执行 B4；
6. S2 只在 validation 改善，未同时进入 test 和 OOD；
7. S2 改善未越过 2% 实践界或置信区间下界未过界；
8. 收敛失败率、最坏组或任一组分 MAE 退化超界。

---

## 10. 当前可立即执行的最小批次

> 更新（2026-07-30）：B0--B5 已完成。B5 已将 `mei3_full_parameter_baseline_retained` 写入版本化契约，MEI-4 基线固定为 S1；不得事后减小样本、更换基线或改写 B4 freeze。

已完成：

1. B0 观测算子契约与受约束 `raw3` 前向秩审计 —— 已通过，见 §11；
2. B1 参数尺度表、S1 残差和多初值框架 —— 已通过并冻结；
3. B2 投影雅可比与 S1/S2 合成等价测试 —— 已通过并冻结，见 §12；
4. B3 数据契约、功效计算方法与授权检查器 —— 已通过并冻结，见 §13；
5. pre-B4：方案 A 真值恢复、相对 CRB、触界可记录失败、200 kHz 相位负对照 —— 已通过并冻结，见 §14；
6. B4：独立授权后正式生成与 S1→S2→S3 配对比较 —— 已冻结，见 §15。
7. B5：验证 B4 父 manifest、冻结裁决契约与 S1 后续基线 —— 已冻结，见 §16。

当前状态：MEI-3 已关闭。MEI-4 只有在建立独立登记契约后才能执行。

---

## 11. B0 表示闭合结论

> 执行日期：2026-07-29  
> 性质：代码化的内存观测契约与数值秩审计，不是正式数据生成或求解器性能证据。  
> freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T023618492318Z_bdc968bc2f93`；manifest SHA256=`6e7823be7b056e0af86e3197c8b7096f3a6a330a570baeb2c2ad899c582ffbb2`。  
> 结论范围：`registered_simulation_domain_only`。

### 11.1 观测到什么

共享 MRS-1 前向 `relaxation_spectrum` 对 `(x_CO2, x_O2, x_N2)` 的**整体缩放精确不变**。把三个组分同时乘 1.03：

| 量               | 变化                                           |
| --------------- | -------------------------------------------- |
| `c_f` 最大绝对差     | `0.0`（精确零，不是小量）                              |
| `alpha_f` 最大绝对差 | `2.220446049250313e-16`（浮点舍入量级）              |
| `c_eq`          | `344.1377033503531` → `344.1377033503531`，不变 |

因此把 `raw3` 三个分量当作无约束自由参数时，总量方向不产生任何观测变化。

### 11.2 缩放雅可比秩审计

观测行取 `raw_tof_s`、`log_amplitude`、`unwrapped_phase_rad` 各 4 个频点（D0 K4 `{25,63,100,200}` kHz），共 12 行；行尺度用 Phase A fixture 的 `observation_std`（`5e-7 s` / `0.02` / `0.1 rad`）。参数为 `raw3` 三个百分数。秩按 `metric_registry.numerical_protocol.rank_reporting_protocol` 要求的三档相对容差 `1e-7 / 1e-6 / 1e-5` 同时报告。

| 登记点                                    | σ₁        | σ₂        | σ₃        | 秩 @1e-7/1e-6/1e-5 | 零向量与组分向量 \|cos\| |
| -------------------------------------- | --------- | --------- | --------- | ----------------- | ---------------- |
| co2=0.030 o2=18.4 T=15 P=0.1013 L=0.20 | 2.535e+01 | 8.073e-01 | 4.584e-09 | 2/2/2             | 1.000000000000   |
| co2=2.515 o2=19.6 T=25 P=0.1013 L=0.25 | 2.984e+01 | 8.730e-01 | 1.971e-09 | 2/2/2             | 1.000000000000   |
| co2=5.000 o2=21.2 T=35 P=0.1013 L=0.30 | 3.375e+01 | 9.617e-01 | 4.618e-09 | 2/2/2             | 1.000000000000   |
| co2=2.515 o2=19.6 T=25 P=0.5000 L=0.25 | 3.357e+01 | 2.090e+00 | 3.947e-09 | 2/2/2             | 1.000000000000   |
| co2=0.030 o2=21.2 T=15 P=0.7090 L=0.20 | 2.873e+01 | 1.723e+00 | 3.208e-09 | 2/2/2             | 1.000000000000   |

三档容差下秩一致为 2，满足 registry 的"离散秩结论必须在每个登记容差下成立"要求。σ₃ 落在 `2e-9`--`5e-9`，是中心差分步长 `1e-4` 的截断残留，不是弱可辨识信号。最小奇异值对应的右奇异向量与归一化组分向量的 `|cos|` 在全部 5 个点（含两个高压点 `0.5` / `0.709` MPa）上都等于 1，即零方向就是组分向量自身、也就是总量缩放方向。

### 11.3 根因

`tv3/sim/generation/tunnel_ventilation/relaxation_spectrum.py` 的 `_normalized_fracs`（第 74--83 行）在进入任何物理计算前把三个组分除以其总和；`tv3/sim/generation/tunnel_ventilation/acoustic_physics.py` 的 `hidden_sound_speed_v2`（第 78--86 行）做同样的归一化。前向只依赖摩尔分数**比值**，不依赖百分数总量。总压由独立参数 `P_MPa` 承担，与组分百分数解耦，所以总量方向在观测里没有任何投影。

这不是实现缺陷 —— 它是"组分用归一化摩尔分数、压力单列"这一参数化的必然结果。缺陷在于 `raw3` 把一个 2 自由度的量写成 3 个无约束参数，却没有登记第三个自由度的信息来源。

### 11.4 处置决议：登记干基物理域

本阶段不把总量当作先验驱动的第三个待估自由度。对当前只含 CO2/O2/N2 的干基 schema，`sum=100%` 是组分定义，不是观测信息或统计先验。登记参数化为：

```text
x = (100/3) * [1,1,1] + B z
B^T B = I,  [1,1,1] B = 0,  z in R^2
x_CO2 >= 0, x_O2 >= 0, x_N2 >= 0
```

`B` 的两列是固定正交 `sum-zero` 基。每次前向调用前都验证非负和 `sum=100%`；整体乘 1.03 的非闭包输入显式报错。三列输出由同一仿射映射联合生成，不存在 N2 回填或求解后投影。

### 11.5 受约束秩审计

| 登记点                                    | σ₁        | σ₂        | 秩 @1e-7/1e-6/1e-5 |
| -------------------------------------- | --------- | --------- | ----------------- |
| co2=0.030 o2=18.4 T=15 P=0.1013 L=0.20 | 1.919e+01 | 7.364e-01 | 2/2/2             |
| co2=2.515 o2=19.6 T=25 P=0.1013 L=0.25 | 2.331e+01 | 8.031e-01 | 2/2/2             |
| co2=5.000 o2=21.2 T=35 P=0.1013 L=0.30 | 2.721e+01 | 8.949e-01 | 2/2/2             |
| co2=2.515 o2=19.6 T=25 P=0.5000 L=0.25 | 2.650e+01 | 1.902e+00 | 2/2/2             |
| co2=0.030 o2=21.2 T=15 P=0.7090 L=0.20 | 2.216e+01 | 1.576e+00 | 2/2/2             |

因此无约束三维表示的零方向保留为负对照证据，但它不属于登记物理域的切空间。B0 状态为 `mei3_b0_representation_closed`。

### 11.6 本节不主张什么

- B0 通过不表示 S1/S2 已实现或通过求解门。
- 不主张观测辨识了总量尺度；总量不是当前干基 schema 的待估量。
- Phase A 的 `mei3_phase_a_structure_supported` 保持有效。
- 四项授权保持 `forbidden_until_explicit_authorization`。

---

## 12. B2 求解器核心结论

> 执行日期：2026-07-29  
> 性质：代码化的内存合成机制审计，不是登记稀疏数据生成或正式 S1--S3 性能比较。  
> freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T081421139186Z_c0ade3f5df14`；manifest SHA256=`ba8bafe0e903f2efc6d6cca8c081d479a98071749d4cdf32674f365d2e3ea8eb`。

### 12.1 数值结果

| 检查                   | 结果                      |
| -------------------- | ----------------------- |
| 投影雅可比最大相对误差          | `8.56712878193837e-10`  |
| S1/S2 目标值差           | `3.848310559106949e-14` |
| S1/S2 最大参数差          | `6.417597475660841e-07` |
| S1/S2 最大 raw3 差（百分点） | `5.649982028899103e-07` |
| S3 显式真值干扰入口          | 通过，且不具备正式配对资格           |

错误相位分支、每样本无约束 K4 偏移、非正定协方差和把声程强行放入线性块四项负对照均显式失败。条件线性求解器已迁入 `tv3/ml/mrs_varpro.py`，Phase A 与 B2 共用同一实现，不保留第二来源。

### 12.2 当前边界

- B2 技术通过不表示 S2 已通过 test/OOD 正式求解门。
- B2 的历史 blocker `mei3_registered_data_authorization_ready_package_not_frozen` 已由 B3 关闭。
- 当前 blocker 为 `registered_sparse_simulation_generation_forbidden_pending_independent_authorization`；在独立授权前不得执行 B4。
- 四项授权继续保持 `forbidden_until_explicit_authorization`。

---

## 13. B3 数据授权就绪结论

> 执行日期：2026-07-29  
> 性质：正式数据生成前的契约与授权就绪审计；没有生成观测、波形或 benchmark。  
> 当前 freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T104111344740Z_b435a5b57d0f`；manifest SHA256=`7badf39329a7b77cffbdaa862dc330add956e262b520ea30e8adc221f0579557`。  
> 历史 freeze（protocol-1，缺 view-nuisance 标定表）：`.../20260729T090738478475Z_073bc6489d9e`；manifest SHA256=`0bc988c3459c752d5eabaf0727ce5d660212473de1d383b2db0f11fbdb81c485`。

### 13.1 已冻结内容

- `mixture_id` 是组分主键；`observation_row_id` 是观测行主键，不存在 `sequence_id` 回退。
- 观测长表固定 D0 K4，每个视图 4 行；字段为 `raw_tof_s`、`log_amplitude`、`unwrapped_phase_rad`，协方差块固定为 `12 × 12`，行序为频率优先后接三类观测。
- `view_protocol`：`views_per_mixture=1`，`registered_view_ids=["view_0"]`，`view_id` 跨 mixture 复用；条件重复只产生不同 `mixture_id`。
- 逐频偏移在 `device_profile_id × frequency_hz` 上跨样本共享，先验来自与评价混合物物理隔离的独立标定集。
- 公共延迟与对数幅度增益在 `device_profile_id × view_id` 上跨样本共享；B4 评价策略为方案 A：在 calibration 划分估计后验，再 join 为 S1/S2 先验，禁止每样本宽先验自由估计。真值仍只存在于隔离的 `s3_truth_nuisance`。
- 划分标签冻结为 `calibration/test/ood`；test 使用 `ambient_core_216`，OOD 使用 `pressure_extension_low_rh_216`；同一 `mixture_id` 的全部视图保持在同一划分。
- 三个划分种子、三个 B1 冻结初值和 `2000` 次按 `design_condition_id` 分层的配对 bootstrap 已冻结。
- S1/S2 字段白名单完全相同：含偏移与 view-nuisance 标定先验，不含组分真值或 S3 干扰真值；S3 真值干扰存放在 S1/S2 运行时不可连接的独立文件中。
- 输入契约哈希 inventory 不含 `execution_plan`，避免文档补写导致 contract SHA 不可复现。

### 13.2 样本量与功效

规划估计量为 S2 相对 S1 的 O2 配对 P90 相对改善。零界为 `0.02`，规划备择为 `0.05`，配对 P90 影响函数标准差上界为 `0.25`，目标功效为 `0.80`，95% CI。正态近似得到每个评价域至少 `546` 个 `mixture_id`；按每域 216 个登记条件整层取整后，test 和 OOD 各冻结 `648` 个，即每条件 3 个。冻结样本量下规划功效为 `0.8631841310`，禁止事后减小。

### 13.3 当前授权边界

`registered_sparse_simulation_generation_review_eligible=true` 只表示就绪包可进入独立评审。`registered_sparse_simulation_generation` 以及波形、benchmark、硬件三项授权仍全部为 `forbidden_until_explicit_authorization`；`formal_solver_gate_ready=false`，B4 不可执行。`allowed_next_stage=null` 是授权前的刻意停点，不是状态丢失。

### 13.4 方案 A 定案与 B1/B2 偏差诊断的关系

B1/B2 零噪声 fixture 上 S1/S2 相对真值约 +4 pp 的 O2 系统偏差，来自每样本宽先验下的 `log_amplitude_gain`（及延迟）可吸收弛豫幅度特征；该 fixture 未执行方案 A 的标定共享语义，因此不能直接外推为 B4 正式门失败。pre-B4 已在方案 A 语义下完成真值恢复与相对 CRB 报告（见 §14）；protocol-2 重冻后，授权依据为当前 B3 freeze，不再使用 protocol-1 历史包。

---

## 14. pre-B4 技术就绪结论

> 执行日期：2026-07-29  
> 性质：进入 B4 前的非正技术门；不生成登记稀疏观测，不改变四项授权。  
> freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T105306238861Z_464cbb0c1285`；manifest SHA256=`94bdd50df59222064bc1349b26bf5c2473ec6c2f3ff194da56ef6b7cd5a6b2a9`。

### 14.1 已关闭的审查缺口

| 缺口                     | 处置                                                           |
| ---------------------- | ------------------------------------------------------------ |
| 方案 A 延迟/增益共享与先验来源      | B3 protocol-2 已登记；pre-B4 用紧标定后验复现                            |
| 零噪声 4 pp O2 偏置         | 宽先验下仍约 `+4.01` pp；方案 A 后降到约 `0.067` pp                       |
| 相对 CRB                 | 已在零噪声 fixture 上报告；方案 A 误差约为 CRB 的 `0.041` 倍                  |
| 触界抛异常                  | 改为返回 `success=False` / `bound_hit=True`；S1/S2 线搜索对条件线性越界对称跳过 |
| Phase A fixture 复制观测算子 | `_build_numerical_fixture` 改为只调用 `ideal_mrs_observation`     |
| 相位负对照只打 25 kHz         | B2 负对照改为扰动 200 kHz 分支                                        |

### 14.2 B4 入口状态

- `b4_technical_ready=true`
- `formal_solver_gate_ready=false`
- blocker 仅剩 `registered_sparse_simulation_generation_forbidden_pending_independent_authorization`
- `scripts/run_tv3_mei3_b4_formal_comparison.py` 在未授权时返回退出码 `5`，verdict=`mei3_waiting_registered_data_authorization`
- 独立授权写入 `stage_status.mei3.authorizations.registered_sparse_simulation_generation=authorized` 后，才允许实现并执行正式配对生成

### 14.3 授权后 B4 仍需补齐的实现项

授权通过后，B4 入口还需从“拒绝”切换到正式生成与配对比较实现：一次性冻结登记稀疏观测、S1→S2→S3 精确配对、失败样本保留在主指标中、写出 `solver_comparison.csv` / `crb_efficiency.csv` / `bootstrap_report.json`。这些代码在未授权前故意不启用。

---

## 15. B4 正式配对比较结论

> 执行日期：2026-07-29  
> 性质：已授权的登记稀疏仿真生成与 S1/S2/S3 正式配对比较；不是波形、benchmark 或硬件声明。  
> 当前 freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T120958962354Z_cf7ed57312d9`；manifest SHA256=`604a5fe6a26c51963b8b5197748002b77ad2177461ff11c3bc5e7cd174f747d8`。  
> 历史作废 freeze：`.../20260729T115530120687Z_4a32b504e5fe`（未把测量 T/L/RH join 为条件先验，O2 P90 约 78%，不得引用）。  
> 并行旁证 freeze：`.../20260729T120817575667Z_cf7ed57312d9`（条件先验已修复，但不是 `stage_status` 当前指针）。  
> 复核报告：[MEI-3 B4 代码复核与结果再分析](../archive/completed/tv3_mrs_ei_mei3_b4_review_and_analysis_report.md)——机制归因、CI 稳健性、代码问题清单（P1 效率计数不对称等）与 B5 注记建议。

### 15.1 授权与实现

- 用户显式授权 `registered_sparse_simulation_generation=authorized`；波形 / benchmark / 硬件三项保持禁止。
- 实现入口：`tv3/audit/mrs_ei_b4_formal.py` + `scripts/run_tv3_mei3_b4_formal_comparison.py`。
- 一次生成 calibration/test/ood；方案 A 在 calibration 上估计共享延迟/增益/逐频偏移后验，再 join 为 S1/S2 先验。
- 观测表中的 `T_C` / `L_m` / `H_RH` 作为已知条件，按混合物 join 为先验均值与初值。
- 评价域各 648 个 `mixture_id`；S1 全部完成后再跑 S2，最后 S3；失败样本保留在主指标中；bootstrap=2000。

### 15.2 主指标

| 域    | S1 O2 P90 | S2 O2 P90 | 相对改善    | bootstrap 95% CI 下界 | 是否过门            |
| ---- | --------- | --------- | ------- | ------------------- | --------------- |
| test | 1.6604    | 1.6142    | 0.02783 | 0.01157             | 否（CI 下界 ≤ 0.02） |
| ood  | 0.7161    | 0.6985    | 0.02453 | 0.00742             | 否（CI 下界 ≤ 0.02） |

点估计在 test 与 OOD 均略高于 `delta_practical=0.02`，但分层配对 bootstrap 置信区间下界均未过界。本 freeze 的否决仅来自 CI 下界。

### 15.3 机制诊断（不替代主指标）

- S3 O2 P90：test `1.5232`，OOD `0.6392`，相对 S2 仍有可恢复干扰误差上限。
- test/OOD 收敛失败率：S1 与 S2 同为约 `0.137` / `0.128`。
- 第一轮错误 freeze 已证明：若不把 `T_C`/`L_m`/`H_RH` 作为已知条件先验，求解门指标无物理意义。

### 15.4 裁决含义

- 正式状态：`mei3_full_parameter_baseline_retained`。
- Phase A 的结构适用结论保持有效；本结果表示“S2 可实现但未通过求解门”，不得回写成 `mei3_varpro_not_applicable`。
- 后续 MEI-4 学习基线应使用冻结的 S1，而不是 S2。
- B5 已将该 verdict 写入版本化 MEI-3 执行契约并关闭本阶段；未重跑 B4，见 §16。

---

## 16. B5 裁决契约冻结

> 执行日期：2026-07-30  
> B5 freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260730T011247690033Z_f1246e54ccb0`；manifest SHA256=`a2b2ce51322e0420971d8503ba61a26c01c179486e5bc6ae15f5af3b22910be5`。  
> 上游 B4 manifest SHA256=`604a5fe6a26c51963b8b5197748002b77ad2177461ff11c3bc5e7cd174f747d8`，已在 B5 写入前验证。

- 正式 verdict：`mei3_full_parameter_baseline_retained`。
- 后续 MEI-4 基线：`S1`；S2 保留为条件线性结构已验证、固定 `max_iterations=100` 预算下进展更快的优化路径，不作为学习基线。
- B5 不生成观测、波形、benchmark 或硬件数据，也不重跑或改写 B4 freeze。
- `allowed_next_stage=null`：B5 仅关闭 MEI-3，不能隐式启动尚未登记的 MEI-4。
