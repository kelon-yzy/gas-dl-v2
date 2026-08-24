# MEI-4 后验基线与覆盖率审计执行计划

> 状态：C0-C2 已冻结；C3 已获 MEI-4 专属授权，但全量观测空间计算于 2026-07-30 按用户要求停止，未形成 C3 完成 freeze 或科学 verdict。2026-08-15 完成一次冻结证据复读与代码契约审查，产出 §0.1 机制分解与 §0.2 三项待处置发现，未运行任何新计算，未改写既有 freeze。详见 [MEI-4 执行进度记录](tv3_mrs_ei_mei4_execution_progress.md)。  
> 前置：MEI-3 已由 B5 关闭，verdict=`mei3_full_parameter_baseline_retained`，MEI-4 确定性基线固定为 `S1`  
> 准入约束：B5 的 `allowed_next_stage=null` 与 `no_mei4_transition=true` 表示 MEI-4 不被隐式启动；必须先冻结本计划对应的独立版本化执行契约（C0）  
> 结论范围：`registered_simulation_domain_only`  
> 上位计划：[tv3 MRS-EI 实验计划](tv3_mrs_information_efficient_inversion_experiment_plan.md) §MEI-4  
> 上游 B4 freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T120958962354Z_cf7ed57312d9`；manifest SHA256=`604a5fe6a26c51963b8b5197748002b77ad2177461ff11c3bc5e7cd174f747d8`  
> 上游 B5 freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260730T011247690033Z_f1246e54ccb0`；manifest SHA256=`a2b2ce51322e0420971d8503ba61a26c01c179486e5bc6ae15f5af3b22910be5`  
> S2 解释约束：沿用 B5 契约 `interpretation_constraints`（S2 效应仅为固定 `max_iterations=100` 预算内的优化进展；CRB 只允许逐样本中位数或先聚合平方再相除；B4 `forward_calls` 记录失真，物理比约 1.5:1）  
> B4 复核报告：[MEI-3 B4 代码复核与结果再分析](../archive/completed/tv3_mrs_ei_mei3_b4_review_and_analysis_report.md)

---

## 0. 执行进度与停止现状（2026-07-30）

以下完成项均有 append-only freeze 及 manifest 约束；它们是当前唯一可用于研究结论的 MEI-4 证据。

| 阶段 | 冻结状态 | freeze 与 manifest SHA256 | 已完成内容 |
| --- | --- | --- | --- |
| C0 | `mei4_contract_frozen` | `20260730T025640042212Z_d3505e1a3e0c`，`057a9e249c57e5d5a3224709b3b7b428b6aa393ce3011497bcf3d71a71e1aa61` | 固定 S1 后验基线、参数化、主覆盖门、M2 PSIS 阈值、C3 抽样规模与 C4 触发阈值。 |
| C1 | `mei4_posterior_core_verified` | `20260730T053939880570Z_d4b88b625f0c`，`ec294ad30ba4453b1aa884671ba97b0d7a91f19a2f7858dd4daedba6a7ce1a71` | 完成拉普拉斯、截断、覆盖 / NLL / CRPS / SBC 估计量、PSIS 及负对照的合成机制审计。 |
| C2 | `mei4_c2_deterministic_evaluation_complete` | `20260730T071532806157Z_76811228bcea`，`1375a2bc737d512196eed56d4afdbd7483c7ddac21442eb07cd6ea3c18671dbf` | 完成 B4 冻结数据上的 M1、M1b、M2 评价；24 个 S1 复解探针全部一致。 |
| C3 就绪与授权 | 已冻结 | 就绪包 `20260730T071644285661Z_806e5e05e951`，`2fbf34d36ca2d952f7215bf4417228f9ed0f4d7eadf560a6e37eafe01aee6ed7`；授权 `20260730T080818819647Z_6c6b2da21139`，`29054a9a70bc7cd17774f67a3d1a0287c1a7a4c28b3e29fdb38671c0b7626215` | 仅授权 SBC、PPC 和 PSIS 触发后的 M2b；未授权 CC-SBI 训练、波形、benchmark 或硬件工作。 |

### C2 已得到的研究事实

1. S1 复解一致性检查通过：test 与 OOD 各 12 个、共 24 个登记混合物的 `raw3_percent` 和 `objective` 均与 B4 冻结解一致。
2. M1、M1b、M2 均未通过 24 条主覆盖带。拒绝样本按未覆盖计入：M1 的 test / OOD 为 `205 / 42`，M1b 为 `183 / 42`，M2 为 `240 / 71`。
3. M2 的 PSIS `k_hat` 超阈率为 test `35 / 648 = 5.40%`、OOD `29 / 648 = 4.48%`；test 超过 C0 冻结上限 5%，因此 M2b 被合规触发。
4. 这些是确定性后验的中间评价结果，不构成通过或失败类 MEI-4 科学 verdict；SBC、PPC 与 M2b 的正式聚合证据尚缺。

### C3 停止前状态

- 已在 MEI-4 专属 `registered_sparse_simulation_generation` 授权范围内开始全量运行，日志确认 SBC 的 test 与 OOD 各完成 `1000 / 1000` 次迭代，PPC 的 test 与 OOD 各完成 `648 / 648` 个冻结混合物。
- M2b 只记录到 `1 / 1296` 的进度后即按用户要求终止；所有 C3 结果当时仍在进程内存中，未写出 `sbc_rank_histograms.json`、`ppc_report.json`、`bootstrap_posterior_report.json` 或新的 C3 完成 freeze。
- 因而上述 SBC/PPC 进度不是可复核的正式审计证据，不得用于更新任何覆盖、PPC、C4 触发或 C5 verdict 结论。当前可恢复状态保持为 `mei4_mc_authorized_pending_execution`。
- C4 仅完成了触发审计与授权停点的代码、契约及单测准备，尚未运行；没有生成 CC-SBI 训练抽样、模型或评价产物。

## 0.1 C2 冻结证据的机制分解（2026-08-15 复读）

本节只重新读取 C2 freeze `20260730T071532806157Z_76811228bcea` 内已存在的 `coverage_report.json`、`laplace_diagnostics.json` 与 `posterior_intervals_{test,ood}.csv`，不新增计算、不改写 freeze、不产生 verdict。C2 原报告只记录了"M1、M1b、M2 均未通过 24 条主覆盖带"，未分解失败构成；以下三项是同一机制的三种表现。

**（1）拒绝率主导 test 域；M1/M1b 的拒绝原因单一，M2 混有 PSIS 拒绝。**

| 方法 | test 拒绝 | OOD 拒绝 | `rejection_reasons` 构成 |
| --- | ---: | ---: | --- |
| M1 | `205 / 648 = 31.6%` | `42 / 648 = 6.5%` | 全部为 `truncation_interval_numerical_failure` |
| M1b | `183 / 648 = 28.2%` | `42 / 648 = 6.5%` | 同上 |
| M2 | `240 / 648 = 37.0%` | `71 / 648 = 11.0%` | test：`m1_proposal_unavailable` 205 + `psis_k_hat_exceeded_for_M2` 35；OOD：42 + 29 |

M2 的 205 次 `m1_proposal_unavailable` 与 M1 的截断失败数量一致，可归因于同一机制；但另有 35 次（占其拒绝的 14.6%，占 648 的 5.40%）是 PSIS 超阈，与截断无关。**M2 的拒绝不能写成"全部由截断造成"。**

`rejection_policy` 列出的四类拒绝条件中，`curvature_not_positive_definite` 与 `curvature_condition_number_exceeded` 在 C2 的六个方法×域分组中一次都没有出现。截断的实际触发点是 `tv3/ml/mrs_posterior.py` 的 `sample_nonnegative_tangent_gaussian`：`initial_candidates=65536` 个打乱 Sobol 候选中落入非负域的数量低于 `minimum_accepted_candidates=2048`，即**超过 96.9% 的切空间高斯质量位于物理定义域之外**。

**（2）能构造出区间的样本上，经验覆盖率系统性高于名义水平。**

M1 的选择条件覆盖率（`covered / (n - rejected)`）：

| 域 | 组分 | 50% | 80% | 90% | 95% |
| --- | --- | ---: | ---: | ---: | ---: |
| test | O₂ | 0.607 | 0.912 | 0.984 | 0.998 |
| ood | O₂ | 0.789 | 0.990 | 1.000 | 1.000 |

M1b 与 M2 的形态相同。方向是过覆盖，不是过度自信。

注意 `coverage_report.json` 的 `coverage` 字段与 `within_acceptance_band` 判定用的是**无条件**覆盖率（`covered / n`）。M1 test O₂ 的无条件值为 `0.4151 / 0.6235 / 0.6728 / 0.6821`；M1 的 8 条 O₂ 主覆盖带中只有 `ood @ 0.95` 落在接受带内。引用覆盖率时两种统计必须同时给出并标注样本量。

**（3）区间宽度超过组分工作量程。**

M1 的等尾边缘区间宽度中位数，单位为百分点：

| 域 | 组分 | 50% | 90% | 95% | 采样量程 |
| --- | --- | ---: | ---: | ---: | ---: |
| test | O₂ | 1.967 | 4.797 | **5.717** | **3.20** |
| ood | O₂ | 1.103 | 2.690 | 3.206 | 3.20 |
| test | CO₂ | 0.418 | 1.010 | 1.201 | 4.97 |
| ood | CO₂ | 0.135 | 0.326 | 0.384 | 4.97 |

test 域的 O₂ 95% 区间是全量程（`18.00–21.20%`）的 1.79 倍，90% 区间同样超量程；OOD 域的 95% 区间恰好等于量程。M2 与 M1 相差在 1% 以内。

**（4）曲率本身正确。** `o2_laplace_to_crb_ratio` 中位数为 test `1.00035`、OOD `1.00004`，即拉普拉斯标准差与逐样本 CRB 一致。`condition_number` 中位数为 test `1.97e7`、OOD `6.59e6`。`truncation_mass_loss` 中位数为 `0.0`、p90 约 `0.485`、均值约 `0.096`，因此 C0 的 T3 触发条件（中位数 `> 0.05`）不成立。

**（5）S1 点估计本身可能位于物理域外。** `laplace_diagnostics.json` 的 `complete_hessian` 审计中，24 个探针有 **4 个**状态为 `unavailable`，错误信息为 `S1 parameters are outside the registered physical domain`（test `M000649` / `M001001`，OOD `M001297` / `M001473`）。同一文件的 `s1_replay` 为 24/24 一致，但那只证明复解可重现，**不证明解落在物理域内**，两者不可互相代替陈述。`posterior_intervals_test.csv` 的 O₂ 区间下界最小到 `14.931%`（OOD 最小 `16.509%`），低于 narrow 采样下界 `18.00%` 三个百分点以上。

**判读。** 曲率正确、区间宽于量程、过覆盖、96.9% 质量出域、S1 解本身出域这五项互相印证，指向同一件事：在冻结 K4 与登记噪声下，单样本 O₂ 后验的信息量不足以支撑比"落在工作量程内"更细的陈述。该判读与 B4 的 S1 test O₂ P90 `1.6604`、MRS-2 最佳臂 median P90 ≈ `2.40 vol%`、MRS-6 的 `1.0–1.3 vol%` 饱和层一致。它是对既有冻结证据的解释，**不是**新的 verdict，也不改写 C2 的 `mei4_c2_intermediate_no_pass_verdict` 状态。

第（5）项对 §0.2 发现 3 的处置方式有直接约束：既然似然的峰有一部分本就在物理域外，把组分先验换成登记 LHS 规格之后后验将由先验主导、测量贡献接近零。**因此先验规格修订与主指标定义必须在同一次契约冻结中完成，主指标用先验宽度到后验宽度的收缩比，不得用绝对区间宽度**，否则会得到符合形式判据的假通过。

## 0.2 C3 恢复前必须处置的三项发现（2026-08-15 代码与契约审查）

以下三项来自对 C0 冻结契约与 C3/C4 实现的一致性审查，不是实验结果。它们都指向同一个后果：**按当前冻结契约执行完整 C3，无法产生 `mei4_deterministic_posterior_retained`**。是否修订契约属于版本化冻结决策，不得在 C3 运行现场调整。

**发现 1：M2b 在结构上无法通过完整校准门。**

C0 的 `mc_protocol` 中 `sbc.methods` 与 `ppc.methods` 均为 `["M1", "M1b", "M2"]`，不含 M2b；`tv3/audit/mrs_ei_mei4_mc.py` 的 `MC_METHODS` 与之一致，M2b 只产出覆盖与 NLL/CRPS，没有 SBC 秩直方图和 PPC 统计量。而 `verdict_state_machine` 要求 `M1_or_M2_or_M2b_passes_complete_calibration_gate`，完整校准门 = 主覆盖门 + §6.2 非退化门。因此 `tv3/audit/mrs_ei_mei4_c4.py` 的 `_m2b_gate()` 将 M2b 的 `complete_calibration_gate_passed` 直接置为 `False`，理由为 `M2b has no registered SBC/PPC evidence in the frozen C3 protocol`。

这是对冻结契约的忠实实现，但与 §4 方法矩阵中"M2b 可否成为冻结后验基线 = 可（替代 M2）"相矛盾。叠加 C2 已确定的 M1/M1b 覆盖失败，**T1 在 C2 结束时即已注定为 `true`**，与 M2b 的实际覆盖结果无关。

**发现 2：T4 触发条件的实现存在循环依赖。**

`mrs_ei_mei4_c4.py` 从 `m2b_report["cost"]["forward_calls"]` 读取实测值判定 `forward_calls > budget`。但 T4 的语义是"M2b 必要但成本超出登记预算"，其用途是**跳过** M2b 直接进入条件性 CC-SBI；实现却要求 M2b 先完整执行。按 B4 `solver_comparison.csv` 的 S1 平均 `forward_calls=395`（含 3 个冻结初值）事前估算，M2b 登记规模需 `1296 × 200 × 395 ≈ 1.02e8` 次前向调用，为 C0 登记预算 `1.0e6` 的约 102 倍。若 T4 改用事前估算，它在 C3 启动前即成立。

**发现 3：确定性方法与 CC-SBI 的组分先验规格不一致。**

C0 的 `posterior_parameterization.domain` 为 `raw3_percent_components_nonnegative_and_sum_100_percent`，`method_input_policy.allowed_assets` 只包含干扰参数标定先验（`calibration_priors`、`view_nuisance_calibration_priors`），**没有组分先验**。因此 M1/M1b/M2 在单纯形上使用平坦组分先验。而登记观测由受约束 LHS 生成（`x_CO2` `0.03–5.00%`、`x_O2` `18.00–21.20%`），§5 C4 又要求 CC-SBI"训练集联合采样组分、干扰参数、实验设计与模型族"，即 CC-SBI 按构造使用真实生成先验。

两者不在同一规格下比较：CC-SBI 的区间收窄会同时包含先验规格差异与方法差异，无法归因。§0.1 的区间超量程与 96.9% 质量出域也直接由平坦先验产生。处置方向有二，须择一预注册：把登记 LHS 组分域纳入 M1/M1b/M2 的后验定义域（它是真实生成分布，不属于 §9 停止条件第 2 条禁止的未登记重标定），或者要求 CC-SBI 的主指标改为**先验宽度到后验宽度的收缩比**而非绝对区间宽度。若两者都不做，CC-SBI 可能在观测信息贡献接近零的情况下通过覆盖门，构成对本计划自身标准的假阳性。

## 0.3 C3 计算成本的本机实测（非正式工程测量）

以下为 2026-08-15 在本机对**已冻结观测**做参数空间复解的计时，属于 [C3 计算效率优化计划](tv3_mrs_ei_mei4_c3_compute_optimization_plan.md) §10 P0 所指的工程基准。未执行任何观测空间抽样，未写入 attempt 分片或 freeze，**不得作为正式 C3 证据**。

| 量 | 实测值 | 测量方式 |
| --- | ---: | --- |
| 单次前向 `ideal_mrs_observation` | `73.6 μs` | 2000 次重复，K4 四频 |
| S1 单初值求解 | `204.6 ms` | 10 个 test 混合物 × 3 冻结初值 |
| S1 每混合物（3 初值） | `613.9 ms` | 同上 |

按此推算 M2b 登记规模 `1296 × 200 × 3 = 777,600` 次求解器启动：单核约 `44.2 h`，12 worker 理想线性约 `3.7 h`。SBC 的约 12,000 次求解与 PPC 的约 248,832 次前向预测成本远低于 M2b。因此 2026-07-30 中止时"M2b 需数十小时"的判断成立于当时的串行实现；P1/P2 并行改造完成后，全量 C3 的规模已降到单次夜间运行可完成的范围。正式 worker 数仍须按 §10 P3 的固定基准确定，本节数据不替代该基准。

## 1. Context：当前已知道什么

MEI-3 已回答确定性点估计问题：S2 结构适用但未通过求解门，S1 是冻结基线。MEI-4 回答的是主计划第四个研究问题：

> 当问题病态或后验多峰时，能否给出覆盖率正确、允许拒绝输出的后验分布？

B4 freeze 中已存在可直接复用、无需任何新数据生成的资产：

| 资产 | 内容 | MEI-4 用途 |
| --- | --- | --- |
| `registered_observations.json` | 1944 个 `mixture_id`（calibration/test/ood 各 648）真值组分表；7776 行观测（每混合物 4 频点，`raw_tof_s` / `log_amplitude` / `unwrapped_phase_rad`）；1944 个 `12×12` 协方差块 | 观测与协方差是后验构造输入；真值组分列仅进审计层 |
| `paired_solutions.json` | 1296 条 test/OOD 逐样本记录：S1/S2/S3 完整参数向量、`raw3_percent`、`success` / `stop_reason` / `bound_hit`、`truth_raw3_percent`、`crb_o2_std_percent` | S1 参数向量是拉普拉斯展开点；真值与 CRB 列仅进审计层 |
| `calibration_priors.json`、`view_nuisance_calibration_priors.json` | 方案 A 在 calibration 划分估计的逐频偏移与 view-nuisance 先验 | 直接作为后验的干扰参数先验，不重估 |
| `s3_truth_nuisance.json` | 1944 条干扰参数真值 | 仅审计层；后验方法物理隔离 |
| `mei3_solver_run_config.json`、`mei3_solver_data_protocol.json` | B4 冻结求解配置、三个冻结初值、划分与 bootstrap 协议 | S1 复解一致性抽查与统计协议沿用 |

与 MEI-4 直接相关的 B4 数值事实：S1 的 O2 P90 为 test `1.6604` / OOD `0.7161`；收敛失败率约 `0.137` / `0.128`（失败样本保留有限估计）；`bound_hit` 存在。这意味着后验方法必须显式处理"估计触界或未收敛时区间怎么给"，不能只在收敛样本上报告覆盖率。

## 2. Task：本计划要回答什么

在 B4 冻结观测、冻结 S1 解与冻结方案 A 先验之上：

1. S1 全参数惩罚似然的拉普拉斯近似，能否在 test 与 OOD 上同时给出覆盖率正确的三组分区间？
2. 若拉普拉斯不足，参数空间重要性采样修正能否修复覆盖，而不引入观测空间新抽样？
3. 只有前两类方法表现出多峰、边界偏差或计算成本问题时，才评估 CC-SBI 是否值得训练。
4. 若覆盖率正确但区间宽，结论只能是"观测信息不足"，不得用任何重标定手段制造高精度。

本计划不优化频点，不改动 S1 求解器与停止条件，不生成正式波形，不做 benchmark 打包或硬件声明。

## 3. 全程不变量

1. 固定 D0 K4 `{25,63,100,200}` kHz；B4/B5 freeze 只读，不重跑、不改写；S1 超参数、边界、初值、停止条件不得为 MEI-4 重新调整——任何改动等于新契约。
2. 点估计与后验都在 B0 登记的干基物理域内：后验分布定义在二维正交 `sum-zero` 切空间 `z`，经同一仿射映射生成三组分报告；非负截断是 C0 预注册的后验域定义的一部分，不是事后投影；不做 N2 回填、`target_transform` 或静默归一化。
3. `mixture_id` 是组分主键，不回退为 `sequence_id`。
4. 组分真值（`x_*_percent`、`truth_raw3_percent`）、干扰真值（`s3_truth_nuisance.json`）与 `crb_o2_std_percent` 只进覆盖率 / SBC / 诊断等审计层；后验构造的读取白名单与 B4 的 S1/S2 白名单一致。
5. calibration 划分只以 B4 已冻结的方案 A 先验形式进入方法；不得在 test/OOD 上拟合任何方法参数；温度缩放、先验缩窄、conformal 等未登记的重标定一律禁止。
6. 评价六件套——SBC 秩直方图、经验覆盖率、NLL、CRPS、后验预测检验、OOD 覆盖率——必须在同一冻结划分上完整报告；缺任何一件不得输出通过类 verdict。
7. 无法构造区间的样本（曲率非正定、数值失败等）按"未覆盖"计入主覆盖率并打 `rejected` 标志；主门样本数固定为每域 648，不允许通过拒绝样本改善覆盖统计。
8. 计算分两类并区别对待：**参数空间计算**（在冻结观测处求前向 / 雅可比、拉普拉斯曲率、参数空间蒙特卡洛抽样）不需要新授权；**观测空间合成**（SBC 的 `(theta, y)` 抽样、后验预测 `y_rep`、参数自助 `y*`、CC-SBI 训练集）属于 `registered_sparse_simulation_generation` 活动类，必须取得 MEI-4 范围的新授权记录，不得引用 B4 的授权记录。波形 / benchmark / 硬件三项授权继续禁止。
9. S2 剖面似然路径的任何结果都受 B5 解释约束限制，不得写成 S2 统计效率主张；该路径不能成为冻结后验基线。
10. 失败必须暴露为异常、非零退出、失败测试或冻结失败 verdict；不吞错，不静默替换方法。

## 4. 后验方法矩阵

| ID | 内容 | 抽样类别 | 启动条件 | 可否成为冻结后验基线 |
| --- | --- | --- | --- | --- |
| M1 | S1 全参数惩罚似然的 MAP-拉普拉斯：白化增广系统的高斯--牛顿曲率，Schur 边缘化到 `z`，截断到非负域 | 参数空间 | C0 冻结后即可 | 可（第一优先） |
| M1b | S2 剖面似然拉普拉斯 | 参数空间 | 与 M1 同批 | 否，仅登记受控比较 |
| M2 | 重要性采样修正后验：M1 为提议分布，惩罚似然重加权，PSIS 诊断 | 参数空间 | 与 M1 同批 | 可（第二优先） |
| M2b | 参数自助法后验 | 观测空间 | 仅当 M2 的 PSIS `k_hat` 超过 C0 冻结阈值 | 可（替代 M2） |
| M3 | CC-SBI：置换不变集合编码器，单纯形约束三组分后验，训练时联合采样组分、干扰参数、设计与模型族 | 观测空间（训练集） | 仅当 §5 C4 触发条件命中且训练抽样获独立授权 | 可（最后顺位） |

主计划中"参数自助法或重要性采样"二选一：本计划选 M2 重要性采样为主路径，理由是它只做参数空间计算、不触发观测空间授权，且逐样本成本远低于按混合物重复求解的自助法；M2b 仅作 PSIS 失效时的登记替代。多方法同时通过校准门时，按 M1 → M2/M2b → M3 的简单优先序冻结基线；更复杂方法要替换已过门的更简单方法，必须两域 O2 区间平均宽度相对改善 `> delta_practical=0.02` 且分层配对 bootstrap 95% CI 下界 `> 0.02`，同时自身全部校准带命中。

## 5. 分阶段执行路径

### C0：执行契约冻结与上游资产盘点

**目标**：建立主计划 §MEI-4 要求的独立版本化执行契约，这是解除 `no_mei4_transition` 的唯一途径。

必须冻结：

1. 父 manifest 绑定：B4 与 B5 的 freeze 路径、`evidence_manifest` SHA256，写入前实际校验；
2. 资产盘点：逐文件记录 §1 表中 B4 资产的路径、SHA256、行数与字段清单；审计真值白名单（`x_*_percent`、`truth_raw3_percent`、`s3_truth_nuisance.json`、`crb_o2_std_percent`）与方法读取白名单分列；
3. 后验参数化：切空间 `z` 的高斯族、非负截断规则、三组分边缘区间的构造方法（等尾、4 个名义水平 `50/80/90/95%`）、截断边缘密度的数值方法与容差；
4. 干扰参数处理：白化增广系统（含方案 A 先验惩罚项）的高斯--牛顿曲率、Schur 边缘化定义；高斯--牛顿近似与完整 Hessian 差异的抽查协议；
5. 校准门全部数值：每域 648 的精确二项接受区间、检验族定义（3 组分 × 4 水平 × 2 域 = 24 条带为一族）、Šidák 族内校正规则、SBC 均匀性检验方法与显著性水平；
6. 拒绝语义：`rejected` 判定条件（曲率非正定、区间构造数值失败、PSIS `k_hat` 超阈）、主覆盖率的"未覆盖"计入规则、选择条件覆盖率的报告格式；
7. 观测空间蒙特卡洛登记表：SBC / PPC / M2b / CC-SBI 训练各自的用途边界、抽样规模、种子表（建议 SBC 每域每方法 1000 次、PPC 每混合物 64 次 `y_rep`、M2b 每混合物 200 次，正式数值以 C0 冻结为准）；
8. C4 触发条件与阈值（见 C4）；verdict 状态机（见 C5）；append-only 输出目录 `outputs/runs/tv3_mrs_ei/mei4_posterior_calibration/freezes/<freeze_id>/`；
9. S1 复解一致性抽查协议：每域抽 12 个混合物按 B4 冻结配置重跑 S1，`raw3_percent` 与 `objective` 一致性容差。

**产物**：`mei4_execution_contract.json`、`b4_asset_inventory.json`、`parent_b4_manifest.json`、`parent_b5_manifest.json`、`evidence_manifest.json`。

**状态**：`mei4_contract_frozen` 或列出缺口的 `mei4_contract_incomplete`；后者不允许进入 C1。

### C1：后验机制合成审计

**性质**：内存合成数值验证，与 B1/B2 同类，不是正式校准证据，不读取 B4 观测。

1. 线性高斯 fixture：拉普拉斯后验必须等于解析后验（协方差与边缘区间在数值容差内一致），名义覆盖在自洽模拟下命中二项接受带；
2. 截断一致性：真值远离边界时截断实现与非截断解析结果一致；人为贴边 fixture 中截断质量损失与区间形变可复算；
3. 估计量对照：覆盖率、NLL、CRPS、SBC 秩统计的实现与已知分布 fixture 的解析值一致；SBC 在自洽模型下秩均匀，检验不拒绝；
4. M2 机制：对故意错标协方差的 fixture，重要性权重与 PSIS `k_hat` 必须报警；
5. 负对照（全部必须显式失败）：后验构造读取真值字段；非正定曲率静默继续；错误相位分支；未登记的重标定入口。

**状态**：`mei4_posterior_core_verified` 或 `mei4_posterior_core_invalid`。技术通过不等于校准门通过。

### C2：冻结数据上的确定性后验正式评价（无观测空间抽样）

1. 对 test/OOD 各 648 个混合物，从 B4 冻结 S1 参数向量出发复算白化增广雅可比与曲率，构造 M1 区间；先完成 S1 复解一致性抽查，不一致即停止；
2. 同批运行 M1b（受控比较）与 M2（PSIS 诊断齐报）；
3. 报告：三组分 × 4 水平 × 2 域经验覆盖率与二项带判定、O2 边缘 NLL 与 CRPS（CO2/N2 同报，`z` 空间联合 NLL 作诊断）、区间宽度分布、`rejected` 率、选择条件覆盖率、按 `design_condition_id` 与压力 / RH 分组的覆盖率、高斯--牛顿对完整 Hessian 的抽查差异、截断质量损失分布、拉普拉斯标准差与 `crb_o2_std_percent` 的比较（受 B5 CRB 引用限制约束）；
4. 本阶段不输出通过类 verdict：评价六件套中 SBC 与 PPC 尚缺，只冻结中间报告。

**产物**：`posterior_intervals_test.csv`、`posterior_intervals_ood.csv`、`coverage_report.json`、`nll_crps_report.json`、`group_coverage_report.json`、`laplace_diagnostics.json`。

### C3：观测空间蒙特卡洛就绪包与授权停点

1. 就绪包按 C0 登记表列出全部观测空间抽样：类别、规模、种子、用途边界、禁止用途（不得作为新的正式评价集、不得进入 benchmark）；
2. 输出 `mei4_mc_review_eligible=true` 后停止——这是与 B3→B4 相同的单向停点；
3. 只有 `stage_status.mei4.authorizations` 写入 MEI-4 范围的新 `registered_sparse_simulation_generation` 授权记录后，才执行：每域每方法的 SBC（`theta` 从该域登记生成分布抽样，检验语义是登记总体上的平均校准，不主张逐点校准）、冻结观测上的 PPC（白化残差范数与逐通道分位数统计量按 C0 预注册）、以及仅在 PSIS 超阈时的 M2b；
4. 授权未获时冻结保持状态 `mei4_waiting_mc_authorization`，已有 C2 结果不得单独升格为 verdict。

**产物**：`mei4_mc_protocol.json`、`sbc_rank_histograms.json`、`ppc_report.json`、（条件）`bootstrap_posterior_report.json`。

### C4：条件性 CC-SBI

启动必须同时满足：

- 触发条件命中（C0 冻结阈值）：T1 M1、M1b、M2/M2b 全部未过校准门；或 T2 多峰证据（多初值后验模式分裂超阈）；或 T3 边界偏差（截断质量损失中位数超阈）；或 T4 M2b 必要但成本超出登记预算；
- `mei4_cc_sbi_training_draws` 获得独立授权记录（训练集是新的观测空间合成，B4 与 C3 的授权都不覆盖它）。

执行要求：3 个训练种子；训练集联合采样组分、干扰参数、实验设计与登记模型族；输出满足单纯形约束的三组分后验；在与 M1/M2 完全相同的冻结划分与评价六件套上评价；训练诊断（loss、SBC 预检）只使用训练与 calibration 资源。

触发条件不满足时整体跳过，记 `mei4_cc_sbi_skipped_not_triggered`；触发但授权未获时记 `mei4_waiting_cc_sbi_training_authorization`。

### C5：裁决与冻结

C5 校验 C2/C3（及条件性 C4）freeze 的 manifest 后写入独立 verdict freeze，不重跑或改写任何上游 freeze：

| verdict | 条件 | 冻结后验基线 |
| --- | --- | --- |
| `mei4_deterministic_posterior_retained` | M1 或 M2/M2b 通过完整校准门 | 按序最简通过方法（M1 优先） |
| `mei4_posterior_calibrated` | 非学习方法全部未过门，CC-SBI 通过 | CC-SBI |
| `mei4_uncertainty_failed` | 全部方法未过门 | 无；任何后验不得用于拒绝策略或精度声明 |

保持状态 `mei4_waiting_mc_authorization` 与 `mei4_waiting_cc_sbi_training_authorization` 不是科学裁决，与 `mei3_waiting_registered_data_authorization` 同类。任何 verdict 都必须同时给出相对 0.4 vol% 参考线与固定 K4 粗精度参考下的区间宽度语境；覆盖正确但区间宽时，结论固定表述为"观测信息不足"。C5 保持 `allowed_next_stage=null`：MEI-5 需要硬件证据，MEI-6 需要按主计划前置另行评审，均不由 C5 隐式放行。

## 6. 正式校准门

### 6.1 主门

对每个方法：3 组分 × 4 名义水平（50/80/90/95%）× 2 域（test、OOD）共 24 条经验覆盖率，全部落入 C0 冻结的精确二项接受区间。族错误率 5%，Šidák 校正到每条带（`alpha_each = 1 - 0.95^(1/24)`），接受区间以整数临界值形式冻结。示意（正式数值以 C0 冻结为准）：`n=648`、名义 95% 时未校正接受带约 `[93.3%, 96.7%]`，Šidák 校正后约 `[92.4%, 97.6%]`。主覆盖率在每域全部 648 个样本上计算，`rejected` 样本按未覆盖计。

### 6.2 非退化门

- SBC 秩均匀性检验（C0 冻结的 ECDF 检验与显著性水平）在每域不得拒绝；
- PPC 预注册统计量不得出现 C0 冻结阈值以上的系统偏离；
- M2 的 PSIS `k_hat` 分布超阈样本率不得高于 C0 冻结上限（超限则改走 M2b）；
- 选择条件覆盖率与分组覆盖率完整报告；粗分组偏离只记警告与诊断，不单独否门。

### 6.3 机制诊断（不替代主门）

区间宽度分布及其相对两条参考线的语境、`rejected` 率与构成、截断质量损失、高斯--牛顿对完整 Hessian 差异、拉普拉斯标准差与逐样本 CRB 的中位数比较、方法间 NLL / CRPS 配对差（沿用 B4 的 2000 次按 `design_condition_id` 分层配对 bootstrap）、墙钟与前向调用成本（按复核报告修正后的计数口径重新实现，不回写 B4 记录）。

## 7. 计划代码与产物边界

### 7.1 代码

| 范围 | 计划文件 | 创建条件 |
| --- | --- | --- |
| M1/M1b/M2 后验构造 | `tv3/ml/mrs_posterior.py` | C1 |
| 覆盖 / NLL / CRPS / SBC 估计量与二项带 | `tv3/audit/mrs_ei_posterior_gate.py` | C1 |
| C2/C3 正式运行编排 | `tv3/audit/mrs_ei_mei4_formal.py` | C2 |
| CC-SBI | `tv3/dl/models/mrs_sbi.py` | 仅 C4 触发后（主计划 §9.2 已预留） |
| C0 契约 | `configs/tv3_mrs_ei/mei4_execution_contract.json` | C0 |
| C1 审计配置 | `configs/tv3_mrs_ei/mei4_posterior_audit.json` | C1 |
| C3 就绪包 | `configs/tv3_mrs_ei/mei4_mc_protocol.json` | C3 |
| C4 契约 | `configs/tv3_mrs_ei/mei4_cc_sbi_contract.json` | 条件 |
| C5 裁决契约 | `configs/tv3_mrs_ei/mei4_verdict_contract.json` | C5 |
| 执行入口 | `scripts/run_tv3_mei4_c0_contract_freeze.py`、`run_tv3_mei4_c1_posterior_audit.py`、`run_tv3_mei4_c2_deterministic_posterior.py`、`run_tv3_mei4_c3_mc_calibration.py`、`run_tv3_mei4_c4_cc_sbi.py`（条件）、`run_tv3_mei4_c5_verdict_freeze.py` | 对应阶段 |
| 测试 | `tests/test_tunnel_ventilation_mei4_posterior.py`、`tests/test_tunnel_ventilation_mei4_formal.py`、`tests/test_tunnel_ventilation_mei4_verdict.py` | 对应阶段 |

残差、雅可比与前向预测复用 `tv3/ml/mrs_varpro.py` 的 `augmented_residual` / `finite_difference_jacobian` / `predict_s1` 等现有接口；弛豫公式继续只存在于 `relaxation_spectrum.py`，后验模块不得复制第二份。单文件保持 200--400 行，超限即拆分。

### 7.2 产物

```text
outputs/runs/tv3_mrs_ei/mei4_posterior_calibration/freezes/<freeze_id>/
  mei4_execution_contract.json
  b4_asset_inventory.json
  parent_b4_manifest.json
  parent_b5_manifest.json
  # C1:
  mei4_posterior_core_report.json
  mei4_negative_controls_report.json
  # C2:
  posterior_intervals_test.csv
  posterior_intervals_ood.csv
  coverage_report.json
  nll_crps_report.json
  group_coverage_report.json
  laplace_diagnostics.json
  # C3:
  mei4_mc_protocol.json
  sbc_rank_histograms.json
  ppc_report.json
  bootstrap_posterior_report.json   # 仅 M2b 启动时
  # C4（条件）:
  cc_sbi_training_report.json
  cc_sbi_eval_report.json
  # C5:
  mei4_verdict_contract.json
  mei4_verdict.json
  evidence_manifest.json
  source_snapshots/
```

阶段目录只追加。`outputs/summary/tv3_mrs_ei/calibration_report.json` 只引用 freeze 路径与 manifest SHA256，不手工重录指标。

## 8. 验证矩阵

### 8.1 单元与数值测试

- 切空间仿射映射与截断域判定的往返一致性；
- 线性高斯 fixture 上拉普拉斯等于解析后验；
- 覆盖率 / NLL / CRPS / SBC 秩估计量对照已知分布 fixture；
- 二项接受区间与 Šidák 校正的整数临界值可复算；
- Schur 边缘化与直接联合协方差求逆一致；
- 无法构造区间按未覆盖计入且打 `rejected` 标志；
- 后验构造读取真值字段、非正定曲率静默继续、未登记重标定入口均显式失败；
- 两个独立新进程复算 C2 主报告数值一致。

### 8.2 回归测试

- MEI-0/1/3 专项回归与三个上游 manifest 独立校验保持通过；
- B4/B5 freeze 内容不因 MEI-4 运行发生任何字节变化；
- 四项授权在未写入 MEI-4 范围新记录时始终拒绝观测空间抽样与 CC-SBI 训练。

## 9. 执行顺序与停止条件

```text
C0 契约冻结与资产盘点
  → C1 后验机制合成审计
  → C2 冻结数据正式评价（参数空间）
  → C3 蒙特卡洛就绪包 → [独立授权决策] → SBC / PPC / 条件 M2b
  → C4 条件性 CC-SBI（触发 + 独立训练授权）或跳过
  → C5 裁决与冻结
```

任一条触发即停止当前分支：

1. 后验构造读取任何审计层真值字段；
2. 在 test/OOD 上拟合或调整任何方法参数，或引入温度缩放、先验缩窄等未登记重标定；
3. S1 复解一致性抽查超出冻结容差；
4. 未获 MEI-4 范围授权而执行观测空间抽样或 CC-SBI 训练；
5. SBC 显著非均匀或 PPC 系统偏离却宣称通过；
6. 覆盖率只在单域成立（test 与 OOD 必须同时）；
7. 试图重跑、改写 B4/B5 freeze，或为 MEI-4 重调 S1；
8. CC-SBI 触发条件未命中却启动训练。

## 10. 当前恢复点

技术上，下一次恢复从已授权的 C3 全量运行开始：重新执行完整的 SBC、PPC 与 PSIS 触发 M2b，不复用已终止进程的内存结果。只有 C3 写出新的完成 freeze 且 manifest 校验通过，才可读取聚合报告并运行 C4 触发审计。

但 §0.2 的三项发现意味着恢复前需要先做一次显式决策，因为按现契约执行完整 C3 得到的 verdict 只能是 `mei4_uncertainty_failed`（或在 CC-SBI 通过时为 `mei4_posterior_calibrated`），`mei4_deterministic_posterior_retained` 不可达。三条路径的取舍如下，任选其一都必须在启动 C3 前登记：

| 路径 | 内容 | 代价 | 结果 |
| --- | --- | --- | --- |
| P-A | 先冻结 C0′ 版本化契约修订：补齐 M2b 的 SBC/PPC 登记（或明确把 M2b 降级为诊断路径并写明其 verdict 不可达）、把 T4 改为事前预算估算、按 §0.2 发现 3 择一处置组分先验规格；随后执行 C3 全量 | 一次契约冻结工作量 + 约一个夜间的 C3 运行 | 证据链完整，C4 与 C5 的比较可归因 |
| P-B | 不改契约，按现登记执行 C3 全量，接受 M2b 产出一份结构上不可能过门的覆盖报告 | 约一个夜间的 C3 运行，其中 M2b 约 `3.7 h` 用于不可用于裁决的结果 | 六件套齐备，verdict 为 `mei4_uncertainty_failed` |
| P-C | 以 §0.1 的机制分解加 MRS-2 / MRS-6 直接进入 MEI-8 收尾 | 需显式契约修订以豁免"六件套完整报告"要求 | 证据链留缺口，不推荐 |

无论选哪条，§9 的停止条件与四类授权边界不变：`registered_sparse_simulation_generation` 的 MEI-4 记录仍只覆盖 SBC、PPC 和条件 M2b；波形、benchmark、硬件与 `mei4_cc_sbi_training_draws` 均未获授权。C4/C5 不得在 C3 完成前运行。§0.2 与 §0.3 都不构成 verdict，也不解除任何授权。
