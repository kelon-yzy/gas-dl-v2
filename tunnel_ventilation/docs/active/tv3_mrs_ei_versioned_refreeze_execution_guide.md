# tv3 MRS-EI 版本化 MEI-0 / MEI-1 重冻结分步执行指导

> 文档状态：F2--F5 证据处置重冻结已执行并验收（2026-07-28）
> 
> 适用范围：`registered_simulation_domain_only`
> 
> 当前事实：MEI-0 `20260728T063115704201Z_8c02b8635dd7` 已冻结；MEI-1 `20260728T064100731550Z_1b55aa2e09cb` 为 `mei1_fixed_k4_retained`；`allowed_next_stage=MEI-3_varpro_audit`
> 
> 目标：建立新的版本化 MEI-0 契约 freeze，并基于该不可变 freeze 建立新的 MEI-1 前向包络审计 freeze

> 执行结果：[MEI-0 / MEI-1 版本化重冻结执行报告](../archive/completed/tv3_mrs_ei_versioned_refreeze_execution_report.md)
> 
> 非目标：本指导不启动 MEI-2，不生成正式波形，不打包新 benchmark，不启动硬件试验，不恢复 MRS-3、F 线或 COMSOL G2+

## 0. F2--F5 证据处置补充（优先于下文历史执行条款）

本节只用于下一次版本化 MEI-0 / MEI-1。下文关于上一有效 freeze 的 ID、`not_represented` 阻断和 `allowed_next_stage=null` 仍是历史事实，但不再定义新一轮状态机。

1. 证据审查事实源为 `docs/references/tv3_mrs_ei_f2_f5_public_evidence_review_20260728.md`，必须以路径和 SHA256 纳入新 freeze。

2. 新状态 `parked_nonblocking` 表示机制或方法路径已有文献支持，但项目域数值包络或设备数据尚未闭合。它不等于 `represented_traceable`，也不能设置 `can_clear_not_represented=true`。

3. 每个搁置项必须冻结 `nonblocking_scope`、`still_blocks`、`unresolved_reason` 和 `revisit_trigger`。缺一项则 MEI-0 preflight 失败。

4. F2/F3 仅在 `obs-cfreq` 登记仿真信息设计中非阻断；F4/F5 仅在不含设备传播与幅相声明的理想信息设计中非阻断。

5. 高压力点继续计算和报告，但其证据状态为诊断性搁置，不进入现场或高压力能力声明。

6. 若不存在真正的物理 blocker、稳定性翻转或执行问题，而全部 K4 仍未越过 `delta_practical`，MEI-1 输出 `mei1_fixed_k4_retained`，冻结 D0 `{25,63,100,200} kHz`，跳过 MEI-2 并进入 `MEI-3_varpro_audit`。

7. `mei1_fixed_k4_retained` 不授权登记数据生成、正式波形、benchmark 打包或硬件试验；四项授权继续为 `forbidden_until_explicit_authorization`。
   
   > 上位计划：[tv3_mrs_information_efficient_inversion_experiment_plan.md](tv3_mrs_information_efficient_inversion_experiment_plan.md)，重点执行其第 13.2 节

---

## 1. Context：执行者必须先理解的事实

### 1.1 为什么必须重冻结

当前 MEI-0 和 MEI-1 的代码与产物可以复现，但契约存在以下已知问题：

1. `delta_num` 把实际数值误差和 2% 实践等价界混在一个字段中。
2. MEI-1 使用 `registered_mrs2` 噪声，即 TOF 标准差 3 μs、温度先验 1 K；低成本候选参考使用 0.5 μs、0.1 K，两者不能直接比较。
3. 216 点正式网格只有 `0.101325 MPa`，没有审计 MEI-2 将使用的高压力域。
4. 当前成本字段把输入驱动代理写成 `total_acoustic_energy`，但 F5 换能器响应没有标定，不能声称实际声能已配平。
5. VarPro 需要原始 TOF、幅度、相位或复传递函数，现有拟议数据字段不足以保证这些结构存在。
6. `raw3` 点估计与单纯形投影之间的公开输出语义没有统一。
7. MEI-3、MEI-4、MEI-6 的基线选择和阶段转移没有形成单一、显式状态表。
8. F2 至 F5 仍缺少可追溯数值包络或独立留出证据。

因此，不允许直接修改旧 freeze，也不允许直接运行 MEI-2。必须先建立一个新版本的 MEI-0 契约，再运行新版本 MEI-1。

### 1.2 “建立 freeze”和“通过阶段”不是同一件事

- 新 MEI-0 只有在契约完整、数值复算成功、manifest 完整时才允许冻结为 `mei0_registry_frozen`。
- 新 MEI-1 可以成功生成审计 freeze，但科学结论仍可能是 `mei1_inconclusive_forward_model`。
- MEI-1 科学门未通过时，脚本可以返回非零退出码并留下有效的不可变失败证据。这不是技术运行失败。
- 不得反复调整阈值并重跑，直到得到 `supported`。
- 只有输入证据、实现或预注册契约发生了可追溯变化，才允许创建下一次新 freeze。

### 1.3 当前旧证据必须保持只读

以下目录及其内容不得修改、删除、覆盖或重命名：

```text
outputs/runs/tv3_mrs_ei/mei0_registry/freezes/20260727T071921821957Z_f209e893a9e5/
outputs/runs/tv3_mrs_ei/mei1_forward_envelope/freezes/20260727T081544975672Z_ea58c1e41e30/
```

旧 MEI-1 的更早 freeze 也必须保留。新的 `stage_status.json` 只把“当前事实源”指向新 freeze，不回写旧目录。

---

## 2. Task：最终必须交付什么

执行完成后必须存在两类新产物。

### 2.1 新 MEI-0 freeze

目标目录：

```text
outputs/runs/tv3_mrs_ei/mei0_registry/freezes/<NEW_MEI0_FREEZE_ID>/
```

最低产物：

```text
model_family_registry.json
design_space.json
metric_registry.json
stage_status.json
numerical_stability_recompute.json
domain_point_manifest.json
registry_change_log.json
mei0_verdict.json
mei0_summary.md
evidence_manifest.json
```

### 2.2 新 MEI-1 freeze

目标目录：

```text
outputs/runs/tv3_mrs_ei/mei1_forward_envelope/freezes/<NEW_MEI1_FREEZE_ID>/
```

最低产物：

```text
model_family_registry.json
design_space.json
metric_registry.json
mei1_run_config.json
parent_mei0_manifest.json
stage_status.json
domain_point_manifest.json
noise_profile_comparison.json
design_ranking.csv
family_envelope_report.json
mei1_forward_envelope.json
mei1_verdict.json
mei1_summary.md
evidence_manifest.json
```

配置和结果必须分别写入以下两个文件，不得使用同名文件混合两种职责：

```text
mei1_run_config.json
mei1_forward_envelope.json
```

推荐采用后一种命名，避免配置和结果混淆。

---

## 3. Format：每一步的固定执行格式

每个步骤都必须按以下顺序执行：

1. 读取“输入”。
2. 检查“开始条件”。
3. 只修改“允许修改的文件”。
4. 运行“验证命令”。
5. 对照“预期结果”。
6. 任一结果不符时执行“失败处理”，不得自行猜测或继续下一步。
7. 将完成证据记录到执行日志。

执行日志每一步至少记录：

```text
step_id:
started_at_utc:
finished_at_utc:
input_paths:
changed_paths:
commands:
exit_codes:
test_summary:
artifact_paths:
artifact_sha256:
decision:
blockers:
```

禁止只写“已完成”“测试通过”而不记录命令、退出码和产物路径。

---

## 4. 全程不变量

执行者必须遵守以下规则。违反任意一条，当前执行无效。

1. 不修改任何旧 freeze。
2. 不把 `mixture_id` 回退或改写为 `sequence_id`。
3. 新 benchmark 不得依赖 `base_condition_id`、`noise_seed_index` 或 `noise_seed`。
4. 正式点估计继续输出 `raw3`、`out_dim=3`。
5. 不使用 N2 闭包回填、`target_transform` 或闭包残差头。
6. 不把真值声速、真值衰减或真值干扰参数加入部署输入。
7. 不给缺失噪声、协方差、来源、数组或哈希添加默认值。
8. 不使用 F3、F4、F5 的诊断代理清除 `not_represented`。
9. 不把仿真共模噪声当成 MEI-5 硬件证据。
10. 不把驱动预算写成实际入射声能。
11. 不静默归一化 `raw3`。
12. 不手工填写性能数字到汇总文件；汇总必须读取阶段产物。
13. 不因 MEI-1 运行成功就自动授权波形、benchmark 或硬件试验。
14. 不创建 MEI-2 代码、配置或产物，除非新 MEI-1 明确输出 `mei1_forward_envelope_supported`。

---

## 5. 版本和命名规则

### 5.1 Registry schema

旧值：

```text
tunnel-ventilation-mrs-ei-1
```

新 registry 契约推荐值：

```text
tunnel-ventilation-mrs-ei-registry-2
```

未来正式数据模式继续保留：

```text
tunnel-ventilation-mrs-ei-1
```

不得把 registry schema 和未来 benchmark schema 写成同一个字段。

建议使用两个明确字段：

```json
{
  "registry_schema_version": "tunnel-ventilation-mrs-ei-registry-2",
  "reserved_benchmark_schema_version": "tunnel-ventilation-mrs-ei-1"
}
```

### 5.2 Freeze manifest schema

新 manifest 使用：

```text
tunnel-ventilation-mrs-ei-freeze-manifest-2
```

新字段至少包括：

```text
parent_freeze_id
parent_manifest_path
parent_manifest_sha256
plan_path
plan_sha256
git_commit
git_relevant_paths_dirty
input_contract_sha256
artifact_sha256
source_sha256
environment
```

`plan_path` 和 `source_sha256` 必须指向 freeze 内的不可变快照，不得指向冻结后还需要更新的活动文档或工作源码。最低快照包括上位计划、执行指南、冻结脚本、registry 审计、MEI-1 审计、共享前向和登记为 traceable 的证据文件。

### 5.3 Freeze ID

保留“UTC 时间戳 + 输入契约哈希前缀”的格式：

```text
<UTC_TIMESTAMP>_<CONTRACT_SHA256_PREFIX>
```

MEI-0 的哈希必须来自全部 registry 的规范化组合，不得只使用单个文件哈希。

MEI-1 的哈希必须至少覆盖：

```text
父 MEI-0 manifest SHA256
mei1_run_config SHA256
MEI-1 审计代码 SHA256
共享 MRS-1 前向代码 SHA256
```

---

## 6. 影响文件和唯一职责

| 文件                                              | 唯一职责                         | 不得包含             |
| ----------------------------------------------- | ---------------------------- | ---------------- |
| `configs/tv3_mrs_ei/model_family_registry.json` | 模型族、来源、数值包络和证据状态             | 阶段运行结果           |
| `configs/tv3_mrs_ei/design_space.json`          | 点集、频率、噪声口径、成本和设计臂            | 可变 verdict       |
| `configs/tv3_mrs_ei/metric_registry.json`       | 指标、门值、统计协议、输出语义、状态转移和授权策略    | 手工抄录的运行指标        |
| `configs/tv3_mrs_ei/mei1_forward_envelope.json` | MEI-1 运行矩阵和门                 | MEI-1 结果         |
| `configs/tv3_mrs_ei/stage_status.json`          | 当前 freeze 指针和当前 verdict      | 指标第二副本           |
| `tv3/audit/mrs_ei_registry.py`                  | MEI-0 构建、校验和数值稳定性复算          | MEI-1 独立业务逻辑     |
| `tv3/audit/mrs_ei_forward_envelope.py`          | MEI-1 双噪声、双域、模型族审计           | 弛豫公式副本           |
| `scripts/run_tv3_mei0_registry_freeze.py`       | MEI-0 追加式冻结                  | 覆盖旧 freeze 的逻辑   |
| `scripts/run_tv3_mei1_forward_envelope.py`      | 从指定 MEI-0 freeze 运行并冻结 MEI-1 | 从可变 config 偷读父契约 |

共享弛豫公式必须继续来自：

```text
tv3/sim/generation/tunnel_ventilation/relaxation_spectrum.py
```

不得在 registry、MEI-1 审计或脚本中复制公式。

---

## 7. Step 0：执行前基线检查

### 7.1 输入

```text
docs/active/tv3_mrs_information_efficient_inversion_experiment_plan.md
configs/tv3_mrs_ei/
tv3/audit/mrs_ei_registry.py
tv3/audit/mrs_ei_forward_envelope.py
scripts/run_tv3_mei0_registry_freeze.py
scripts/run_tv3_mei1_forward_envelope.py
tests/test_tunnel_ventilation_mei0_registry.py
tests/test_tunnel_ventilation_mei1_forward_envelope.py
```

### 7.2 开始条件

- 当前目录必须是 `tunnel_ventilation/`。
- 旧 freeze 必须存在。
- 当前 `stage_status.json` 必须仍显示旧 MEI-1 为 inconclusive 和 `allowed_next_stage=null`。
- 相关旧证据 manifest 必须通过哈希验证。
- 必须记录当前 Git commit 和相关文件 dirty 状态。

### 7.3 命令

```powershell
git status --short
python -m pytest tests/test_tunnel_ventilation_mei0_registry.py tests/test_tunnel_ventilation_mei1_forward_envelope.py -q
```

### 7.4 预期结果

- 旧基线测试应为 22 项通过，除非后续仓库已合法增加测试。
- 如果基线测试失败，先定位现有回归，不得开始 schema 改造。
- 用户已有未提交文档修改不得被覆盖或回退。

### 7.5 失败处理

输出失败测试、文件和行号。停止执行，不创建任何新 freeze。

---

## 8. Step 1：升级 MEI-0 registry 契约

本步骤只修改配置契约和校验器，不运行正式 freeze。

### 8.1 拆分数值带和实践界

删除 `delta_num` 作为新契约字段。改为：

```json
{
  "decision_thresholds": {
    "delta_numerical": {
      "formula": "max(2*fd_relative_change, 2*fresh_process_relative_change, optimizer_relative_tolerance)",
      "recompute_required_at_freeze": true,
      "by_noise_profile": {},
      "shared_upper_bound": null
    },
    "delta_practical": {
      "value": 0.02,
      "role": "minimum_meaningful_relative_improvement",
      "source": "pre_registered_practical_equivalence_policy",
      "not_a_numerical_error": true
    }
  }
}
```

必须满足：

- `delta_numerical` 不含固定 2% floor。
- 每个正式噪声口径独立复算。
- `shared_upper_bound` 等于各口径实际数值带最大值。
- 排名同时输出数值等价类和实践等价类。
- 科学通过门使用 `delta_practical`。
- 数值稳定性判断使用 `delta_numerical`。

### 8.2 冻结两套完整噪声口径

`design_space.json` 必须包含：

```text
low_cost_k4_primary
registered_mrs2_stress
```

每套口径都必须显式提供：

```text
jitter_std_s
relative_amp_std
covariance_model
prior_std.t_c
prior_std.path_length_m
prior_std.h_rh
prior_std.co2_percent
fixed_delay_s 或其来源引用
source
refs
```

不得让 `low_cost_k4_primary` 隐式继承 `registered_mrs2_stress` 的字段。

已知值：

```text
low_cost_k4_primary.jitter_std_s = 5e-7
low_cost_k4_primary.prior_std.t_c = 0.1
registered_mrs2_stress.jitter_std_s = 3e-6
registered_mrs2_stress.prior_std.t_c = 1.0
```

其余低成本口径字段如果没有可追溯来源，必须停止并输出：

```text
mei0_registry_incomplete: low_cost_noise_profile_missing_traceable_fields
```

不得把缺失值填成 0、`null` 后继续，也不得复制压力测试口径。

### 8.3 建立两个命名点集

保留：

```text
ambient_core_216
```

其轴为：

```text
O2 window: 4
CO2: 0.03, 2.515, 5.0
T_C: 15, 25, 35
L_m: 0.2, 0.25, 0.3
H_RH: 30, 50, 70
P_MPa: 0.101325
```

新增：

```text
pressure_extension_low_rh_216
```

其轴为：

```text
O2 window: 4
CO2: 0.03, 2.515, 5.0
T_C: 15, 25, 35
L_m: 0.2, 0.25, 0.3
H_RH: 30
P_MPa: 0.5, 0.709
```

正式 MEI-1 点集：

```text
formal_mei1_432 = ambient_core_216 union pressure_extension_low_rh_216
```

必须检查：

- core 恰好 216 点；
- pressure extension 恰好 216 点；
- union 恰好 432 个唯一点；
- 点 ID 稳定且包含 point set 名；
- 高压力点不能被当成附加绘图点，必须进入正式门。

### 8.4 把驱动预算和实际声能分开

将新契约中的以下命名：

```text
total_acoustic_energy_relative_s
total_acoustic_energy
equal_total_energy
```

替换为：

```text
total_drive_budget_relative_s
total_drive_budget
equal_input_drive_budget
```

D0 数值可以继续使用：

```text
total_drive_budget_relative_s = 0.000375238095238095
total_measurement_time_s = 0.23
```

D4 和 D5 继续保持：

```text
D4 total_measurement_time_s = 5.23
D5 total_measurement_time_s = 0.47
D4 eligible_for_information_gate = false
D5 eligible_for_information_gate = false
```

新增声明：

```text
actual_incident_acoustic_energy_status = unavailable_without_F5_calibration
```

### 8.5 冻结统计估计量

`metric_registry.json` 至少拆成三类统计协议。

有限登记信息审计：

```text
estimand = exact worst-case CRB-P90 over frozen finite point and model set
random_unit = none
bootstrap = forbidden
report = exact paired difference plus delta_numerical bound
```

学习型求解实验：

```text
random_unit = mixture_id
strata = mixture_id and design_condition
n_data_split_seeds = 3
n_train_seeds = 3
n_bootstrap_resamples = 2000
ci_level = 0.95
```

后验校准实验：

```text
nominal_coverages = 0.5, 0.8, 0.9, 0.95
report_marginal_coverage = true
report_group_conditional_coverage = true
report_selection_conditional_coverage_after_rejection = true
report_interval_width = true
report_rejection_rate = true
```

如果 432 点被定义为完整有限登记集合，不得对它伪造 bootstrap 置信区间。

### 8.6 冻结 VarPro 观测契约

本步骤只登记未来 MEI-3 输入要求，不生成数据。

必须登记：

```text
raw_tof_s
log_amplitude 或 amplitude
phase_rad 或 complex_transfer_real/imag
observation_covariance
frequency_hz
device_profile_id
view_id
T_C
P_MPa
H_RH
L_m
```

逐频标定偏移的层级固定为：

```text
device_profile_id × frequency_hz 上跨样本共享
```

并要求独立标定先验。禁止为单次 K4 样本设置四个无约束自由偏移。

如果只有 `c_observed`，必须输出：

```text
mei3_varpro_not_applicable_to_c_observed_only
```

不得伪造线性块。

### 8.7 统一 raw3 和后验输出语义

冻结以下公开契约：

```text
point_estimate = raw3, out_dim=3, no silent normalization
posterior = joint distribution over raw3, no silent normalization
closure_sum_abs_error = audit monitor only
simplex_projected_view = optional explicit audit view only
```

单纯形投影不得用于：

```text
训练标签重写
主性能指标
部署输出
N2 闭包回填
```

### 8.8 建立唯一状态转移表

在 `metric_registry.json` 中建立 `stage_transition_policy`，不要再在多个脚本分别写规则。

最低规则：

| 当前状态                              | 条件                                        | 下一状态或基线                                                   |
| --------------------------------- | ----------------------------------------- | --------------------------------------------------------- |
| `mei0_registry_frozen`            | manifest 通过                               | 允许运行 MEI-1                                                |
| `mei1_forward_envelope_supported` | 所有门通过                                     | `allowed_next_stage=MEI-2_robust_design`                  |
| `mei1_inconclusive_forward_model` | 任一阻断项存在                                   | `allowed_next_stage=null`                                 |
| `mei1_fixed_k4_retained`          | 无物理 blocker，但所有 K4 均未越过 `delta_practical` | 固定 D0 K4、跳过 MEI-2，`allowed_next_stage=MEI-3_varpro_audit` |
| `mei3_varpro_supported`           | S2 通过求解门                                  | MEI-4 使用 S2                                               |
| `mei3_varpro_not_applicable`      | 线性结构不成立                                   | MEI-4 使用 S1                                               |
| MEI-3 存在稳定可学习残差                   | 证据字段为真                                    | MEI-6 可进入资格评审                                             |
| 其他 MEI-3 状态                       | 无资格证据                                     | 不允许 NP-VPNet                                              |

授权字段必须分开：

| 字段                                        | 授权后允许的活动                     | 未授权时仍可做什么                |
| ----------------------------------------- | ---------------------------- | ------------------------ |
| `registered_sparse_simulation_generation` | 生成新的正式登记稀疏谱仿真观测              | 使用既有冻结前向做解析和数值审计         |
| `formal_waveform_generation`              | 生成包含设备链路的完整时域波形              | 不生成波形的 Fisher、CRB 和求解器审计 |
| `benchmark_packaging`                     | 发布正式 schema、划分、manifest 和评价包 | 维护阶段内部审计产物，不称为 benchmark |
| `hardware_trial`                          | 开展真实换能器、管道或压力台架采集            | 仅做不依赖新硬件证据的仿真与方法工作       |

MEI-0 freeze 后四者全部为 `forbidden_until_explicit_authorization`。

即使 MEI-1 supported，也只能设置：

```text
registered_sparse_simulation_generation_review_eligible = true
```

不能自动改成 authorized。

当前 `mei1_fixed_k4_retained` 的 `allowed_next_stage=MEI-3_varpro_audit` 只授权阶段推进，不改变上述四个字段。`forbidden_until_explicit_authorization` 是等待独立审批，不是永久禁止。

### 8.9 本步骤验证

必须新增或更新测试，至少覆盖：

```text
test_registry_v2_schema_is_distinct_from_benchmark_schema
test_delta_numerical_has_no_practical_floor
test_delta_practical_is_separate_and_pre_registered
test_low_cost_noise_profile_requires_all_fields
test_noise_profiles_do_not_inherit_from_each_other
test_point_sets_are_216_216_and_432_unique
test_high_pressure_points_are_in_formal_mei1_gate
test_drive_budget_is_not_labeled_acoustic_energy
test_finite_registry_information_gate_forbids_bootstrap
test_raw3_contract_forbids_silent_normalization
test_stage_transition_selects_s1_when_varpro_not_applicable
test_authorization_fields_are_independent
```

运行：

```powershell
python -m pytest tests/test_tunnel_ventilation_mei0_registry.py -q
```

失败时停止，不运行正式 freeze。

---

## 9. Step 2：修改 MEI-0 校验和冻结实现

### 9.1 修改 `mrs_ei_registry.py`

必须完成：

1. 将单一 `SCHEMA_VERSION` 改为 registry schema 常量，不再与 benchmark schema 共用。
2. 将 `build_narrow_points` 改为按命名点集构建。
3. 新增点 ID 去重和 expected count 校验。
4. 删除对 `registered_mrs2` 的硬编码读取。
5. 数值复算按配置列出的两个噪声 profile 运行。
6. 输出 `delta_numerical_by_profile` 和共享上界。
7. 删除 2% floor 对数值带的参与。
8. 校验 `delta_practical` 的来源和固定值。
9. 将成本校验改为 drive budget 语义。
10. 校验统计协议、raw3 契约、VarPro 契约、状态转移和独立授权字段。

不要保留同时生效的新旧两套门。旧 `delta_num` 只允许出现在旧 freeze 或迁移说明中；新 registry 出现该字段应直接失败。

### 9.2 修改 MEI-0 冻结脚本

必须完成：

1. freeze ID 使用全部 registry 的组合哈希。
2. manifest 升级到 v2。
3. manifest 纳入上位计划文件哈希。
4. manifest 记录 Git commit 和相关路径是否 dirty。
5. 生成 `domain_point_manifest.json`。
6. 生成 `registry_change_log.json`，列出相对父 MEI-0 的契约变化。
7. `stage_status.json` 更新到新 MEI-0 时，移除旧 MEI-1 当前指针。
8. `allowed_next_stage` 只能为 `MEI-1_forward_envelope`。
9. 四类授权全部保持 forbidden。
10. 保留临时目录写入、原子 rename、拒绝覆盖和写后 manifest 验证。
11. MEI-0 必须通过 `--parent-mei0-freeze-dir` 显式指定父 freeze，不得硬编码固定祖先。
12. 生成 `experiment_plan_snapshot.md`、`refreeze_execution_guide_snapshot.md` 和 `source_snapshots/`。

### 9.3 MEI-0 集成测试

新增测试至少覆盖：

```text
test_mei0_v2_freeze_contains_parent_and_plan_hash
test_mei0_v2_freeze_uses_combined_contract_hash
test_mei0_v2_freeze_clears_stale_mei1_pointer
test_mei0_v2_freeze_keeps_all_authorizations_forbidden
test_mei0_v2_freeze_refuses_existing_output_dir
test_mei0_v2_manifest_detects_tampered_source
```

运行：

```powershell
python -m pytest tests/test_tunnel_ventilation_mei0_registry.py -q
python -m pytest tests/test_tunnel_ventilation_mei1_forward_envelope.py -q
```

两条命令都通过后才能进入 Step 3。

---

## 10. Step 3：创建正式 MEI-0 v2 freeze

### 10.1 正式运行前检查

确认：

- 三份 registry 均为 v2；
- 两套噪声字段完整；
- 432 点构建测试通过；
- 数值复算使用两个 profile；
- 相关代码和配置没有未说明修改；
- 旧 freeze 未变化；
- 当前阶段仍未放行 MEI-2。

### 10.2 完成改造后的目标命令

冻结脚本完成上述改造后运行：

```powershell
python scripts/run_tv3_mei0_registry_freeze.py --config-dir configs/tv3_mrs_ei --parent-mei0-freeze-dir <CURRENT_MEI0_FREEZE_DIR>
```

`<CURRENT_MEI0_FREEZE_DIR>` 必须替换为 `stage_status.json` 当前登记的 MEI-0 freeze 路径，不得原样传入占位符。不要手工指定一个已经存在的 `--output-dir`。

### 10.3 预期退出状态

- 退出码 0：freeze 创建成功，继续核验。
- 退出码非 0：MEI-0 未完成。不得创建 MEI-1。

### 10.4 创建后核验

必须核对：

```text
verdict = mei0_registry_frozen
allowed_next_stage = MEI-1_forward_envelope
formal_waveform_generation = forbidden_until_explicit_authorization
benchmark_packaging = forbidden_until_explicit_authorization
hardware_trial = forbidden_until_explicit_authorization
point_count = 432
noise_profile_count = 2
delta_numerical_by_profile 已填充
delta_practical = 0.02
```

使用项目 manifest 校验函数核对新目录。校验结果必须为空列表。

### 10.5 禁止的修复方式

- 不得手工修改 freeze 内 JSON。
- 不得手工修正哈希。
- 不得复制旧 `delta_num_recompute.json` 充当新复算。
- 不得在 freeze 完成后修改源配置并继续使用该 freeze 作为新配置证据。

如果 freeze 内容错误，保留错误 freeze 作为失败证据，修正源代码或配置后创建另一个新 freeze。

---

## 11. Step 4：准备 MEI-1 v2 配置和父 freeze 读取

### 11.1 MEI-1 必须读取不可变父契约

修改 `run_tv3_mei1_forward_envelope.py`，增加必需参数：

```text
--parent-mei0-freeze-dir
```

目标帮助信息中必须出现该参数。MEI-1 应从父 freeze 读取：

```text
model_family_registry.json
design_space.json
metric_registry.json
stage_status.json
evidence_manifest.json
```

不得从 `configs/tv3_mrs_ei/` 重新读取这三份 registry 作为正式输入。

`--config-dir` 只用于读取当前 MEI-1 运行配置，不能覆盖父 registry。

### 11.2 父 freeze 校验

运行 MEI-1 前必须检查：

```text
父 verdict = mei0_registry_frozen
父 registry schema = tunnel-ventilation-mrs-ei-registry-2
父 manifest 哈希正确
父 allowed_next_stage = MEI-1_forward_envelope
当前 stage_status 指向同一父 freeze
```

任一项不符时退出，不创建 MEI-1 freeze。

### 11.3 MEI-1 运行矩阵

`mei1_forward_envelope.json` 必须显式包含：

```text
noise_profiles = [low_cost_k4_primary, registered_mrs2_stress]
point_sets = [ambient_core_216, pressure_extension_low_rh_216]
formal_point_union = formal_mei1_432
design_count = 15
primary_metric = max_p90_o2_percent
secondary_metric = median_p90_o2_percent
numerical_equivalence_source = delta_numerical
decision_equivalence_source = delta_practical
```

正式运行不允许 stride 抽样。

### 11.4 模型族证据规则

每个 F1 至 F5 必须具有以下字段：

```text
status
source
refs
implementation_or_holdout_path
evidence_sha256
parameter_or_bias_bounds
bound_semantics
can_clear_not_represented
```

允许状态：

```text
represented_traceable
independent_holdout_available
not_represented
```

只有前两个状态在证据和哈希完整时可能清除阻断。

结构代理必须固定：

```text
can_clear_not_represented = false
```

F2 至 F5 任一仍为 `not_represented` 时，MEI-1 必须判 inconclusive。

### 11.5 高压力有效性检查

在计算排名前，逐个高压力点检查：

```text
前向函数返回有限值
参数来源声明覆盖该压力
有限差分两侧均在有效域
雅可比无 NaN 或 Inf
CRB 求解未使用隐藏正则化
```

如果参数模型没有证据覆盖 0.5 或 0.709 MPa，输出：

```text
pressure_domain_not_validated
```

该状态阻止 `supported`。不得删除高压力点来让测试通过。

---

## 12. Step 5：修改 MEI-1 审计实现

### 12.1 双噪声循环

`run_mei1_audit` 不得再固定读取：

```python
design["observation_baselines"]["registered_mrs2"]
```

应按配置中的 `noise_profiles` 逐个运行，并以 profile ID 作为结果第一层键。

每个 profile 必须独立报告：

```text
baseline K4 max P90
baseline K4 median P90
15 个设计排名
numerical equivalence classes
practical equivalence classes
best_vs_baseline exact improvement
ranking span
bottleneck counts
rank interval
```

### 12.2 双域报告

每个 profile 下分别报告：

```text
ambient_core_216
pressure_extension_low_rh_216
formal_mei1_432
```

正式 verdict 使用 `formal_mei1_432`，但两个子域均不得退化或出现未报告失败。

### 12.3 等价类和通过门

每个设计排名输出两组 rank：

```text
rank_numerical
rank_practical
```

规则：

- 差值不超过 `delta_numerical`：数值不可分辨。
- 差值超过数值带但不超过 `delta_practical`：数值可分辨，但没有实践意义。
- 只有超过 `delta_practical` 才能成为决策上可分辨的设计层级。

不得因浮点排序稳定就写 `stable optimum`。

### 12.4 MEI-1 verdict

`mei1_forward_envelope_supported` 必须同时满足：

1. 无运行或契约 issues。
2. F2 至 F5 不存在未清除的 `not_represented`。
3. 两套噪声口径均完成。
4. 432 点全部进入正式门。
5. 高压力域有效性通过。
6. 设计排名在实践界下可分辨。
7. 跨可追溯模型族没有排名翻转。
8. 主要瓶颈结论稳定。
9. 逐点主角门通过。
10. 合成平行、正交、混合偏差只用于证伪，没有替代真实包络。

否则按原因输出：

```text
mei1_inconclusive_forward_model
```

如果是代码、配置、哈希或数组错误，输出：

```text
mei1_audit_failed
```

不要把技术失败和科学不确定混为一个状态。

### 12.5 测试

新增测试至少覆盖：

```text
test_mei1_requires_explicit_parent_freeze
test_mei1_rejects_parent_manifest_mismatch
test_mei1_reads_registries_from_parent_freeze
test_mei1_runs_both_noise_profiles
test_mei1_reports_core_pressure_and_union
test_mei1_practical_rank_is_separate_from_numerical_rank
test_mei1_blocks_unvalidated_pressure_domain
test_mei1_blocks_each_unrepresented_family
test_proxy_never_clears_not_represented
test_mei1_supported_requires_all_profiles_and_domains
test_mei1_inconclusive_keeps_allowed_next_stage_null
test_mei1_supported_does_not_authorize_waveform_or_hardware
```

运行：

```powershell
python -m pytest tests/test_tunnel_ventilation_mei1_forward_envelope.py -q
python -m pytest tests/test_tunnel_ventilation_mei0_registry.py tests/test_tunnel_ventilation_mei1_forward_envelope.py -q
```

全部通过后才能正式运行 MEI-1。

---

## 13. Step 6：创建正式 MEI-1 v2 freeze

### 13.1 运行前取得父路径

从新 `configs/tv3_mrs_ei/stage_status.json` 读取新 MEI-0 的 `freeze_dir`。

记为：

```text
<NEW_MEI0_FREEZE_DIR>
```

执行者必须把命令中的占位符替换为实际路径。不得把尖括号占位符原样传给命令。

### 13.2 确认目标 CLI 已实现

先运行：

```powershell
python scripts/run_tv3_mei1_forward_envelope.py --help
```

帮助信息必须包含：

```text
--parent-mei0-freeze-dir
```

如果没有该参数，说明代码改造未完成。停止，不得使用旧 CLI 运行新 freeze。

### 13.3 正式命令

```powershell
python scripts/run_tv3_mei1_forward_envelope.py --config-dir configs/tv3_mrs_ei --parent-mei0-freeze-dir <NEW_MEI0_FREEZE_DIR>
```

### 13.4 解释退出码

建议脚本保持以下语义：

| 退出码 | 含义                    | 是否可能有有效 freeze |
| --- | --------------------- | -------------- |
| 0   | 审计完成且科学门通过            | 是              |
| 2   | 审计完成但科学门 inconclusive | 是              |
| 3   | 前置契约、父 manifest 或配置失败 | 否              |
| 4   | 数值运行或产物校验失败           | 否              |

退出码 2 时，不得马上重跑。先检查是否已生成 manifest 完整的 inconclusive freeze。

### 13.5 产物核验

无论 verdict 是否 supported，都必须检查：

```text
parent_mei0_manifest_sha256 与新 MEI-0 一致
point_count = 432
noise_profiles = 2
design_count = 15
core、pressure、union 三组结果存在
F2 至 F5 状态完整
numerical 和 practical 两类排名存在
manifest 校验通过
旧 freeze 哈希未变化
```

### 13.6 状态提升规则

如果 verdict 为：

```text
mei1_forward_envelope_supported
```

则：

```text
allowed_next_stage = MEI-2_robust_design
registered_sparse_simulation_generation_review_eligible = true
formal_waveform_generation = forbidden_until_explicit_authorization
benchmark_packaging = forbidden_until_explicit_authorization
hardware_trial = forbidden_until_explicit_authorization
```

如果 verdict 为：

```text
mei1_inconclusive_forward_model
```

则：

```text
allowed_next_stage = null
四类授权继续 forbidden
```

如果 verdict 为 `mei1_audit_failed`，不得把该运行设为科学事实源，必须先修复技术错误并创建新的运行目录。

---

## 14. F2 至 F5 证据工作包

这些工作包决定 MEI-1 是否可能通过，但不允许伪造证据。

### 14.1 F2：H2O 弛豫参数

必须取得：

```text
可追溯文献或独立数据
参数定义和单位
适用温度、湿度、压力、频率范围
上下界及其语义
实现路径
来源哈希
```

如果文献不覆盖 0.709 MPa，不得外推并写成 represented。

### 14.2 F3：耦合弛豫

可接受：

```text
共享前向模块中的可追溯耦合模型实现
或独立数值参考形成的留出模型族
```

不可接受：

```text
任意 5% cross-mix 代理
仅凭代理未翻转排名就清除阻断
```

### 14.3 F4：衍射和近场

可接受：

```text
适用几何和频带下的可追溯解析包络
或已经存在且获得使用授权的独立高保真子集
```

不得因为本任务自动启动 COMSOL G2+。缺少已有证据时保持 `not_represented`。

### 14.4 F5：换能器响应

可接受：

```text
已有台架标定
可追溯厂商幅相曲线及其不确定度
或独立设备响应数据
```

没有硬件授权时不得新采集。没有标定时：

- 保持 `not_represented`；
- 成本只称 drive budget；
- MEI-1 保持 inconclusive。

---

## 15. 最终验证矩阵

### 15.1 单元测试

```powershell
python -m pytest tests/test_tunnel_ventilation_mei0_registry.py -q
python -m pytest tests/test_tunnel_ventilation_mei1_forward_envelope.py -q
```

目标：全部通过。任一目标测试不应超过 60 秒；超过时应报告性能问题，不得提高超时隐藏异常。

### 15.2 相关回归测试

至少运行 MRS-1 前向和 identifiability 相关测试，确保没有复制或改变共享公式。

先定位测试：

```powershell
rg --files tests
rg -n "relaxation_spectrum|identifiability_v3_mrs" tests
```

然后运行实际存在的相关测试文件。不得虚构测试路径。

### 15.3 全套测试

```powershell
python -m pytest -q
```

如果全套测试因与本任务无关的既有失败而不通过，必须记录：

```text
失败测试名
是否在修改前已失败
与本任务的关联判断
目标测试结果
```

不得删除无关失败，也不得声明全套通过。

### 15.4 Manifest 验证

新 MEI-0 和新 MEI-1 的 manifest 均必须：

- 自身 SHA256 与 `stage_status.json` 一致；
- 所有 artifact 存在且哈希一致；
- 所有 source 存在且哈希一致；
- 父 manifest 哈希一致；
- 上位计划哈希一致。

### 15.5 Diff 检查

结束前检查：

```powershell
git status --short
git diff --stat
git diff -- configs/tv3_mrs_ei tv3/audit scripts tests docs/active
```

重点排查：

```text
旧 delta_num 是否仍在新契约生效
registered_mrs2 是否仍被硬编码为唯一 profile
216 是否仍被硬编码为正式总点数
total_acoustic_energy 是否仍被用于未标定声明
是否存在 silent fallback
是否复制了弛豫公式
是否手工写了 stage 指标
是否出现第二套状态机
是否误授权波形、benchmark 或硬件
```

---

## 16. 停止条件和失败处理表

| 现象             | 正确动作                          | 禁止动作                 |
| -------------- | ----------------------------- | -------------------- |
| 低成本噪声字段缺来源     | MEI-0 incomplete，补来源          | 复制 MRS-2 值           |
| 数值重复性差         | 保留失败证据，定位 FD 或求解器             | 提高 `delta_practical` |
| 高压力模型无有效性来源    | MEI-1 inconclusive            | 删除高压力点               |
| F2 至 F5 缺证据    | 保持 `not_represented`          | 用代理清除阻断              |
| 设计差异小于实践界      | 保留固定 K4 候选，MEI-1 inconclusive | 改小 2% 门后重跑           |
| MEI-1 退出码 2    | 验证并保留 inconclusive freeze     | 当作运行失败反复重跑           |
| Manifest 哈希不一致 | 技术失败，停止提升状态                   | 手工改哈希                |
| 旧 freeze 被改动   | 停止并报告数据完整性事故                  | 用新哈希覆盖记录             |
| 测试只在验证集改善      | 判未通过                          | 继续调参直到通过             |
| 想启动硬件或 COMSOL  | 请求独立授权                        | 由 MEI-1 隐式放行         |

---

## 17. 完成定义

只有同时满足以下条件，才能宣布“版本化 MEI-0 / MEI-1 重冻结工作完成”：

### 17.1 MEI-0 完成条件

- registry schema 已升级并与 benchmark schema 分离；
- 两套噪声口径字段完整且有来源；
- `delta_numerical` 和 `delta_practical` 已拆分；
- 432 点合同已冻结；
- drive budget 语义已修正；
- 统计、VarPro、raw3、状态机和授权合同已冻结；
- 新 MEI-0 manifest 校验通过；
- `allowed_next_stage` 仅为 MEI-1；
- 没有授权正式数据、波形或硬件。

### 17.2 MEI-1 完成条件

- 显式绑定新 MEI-0 父 freeze；
- 两套噪声、两个子域和 432 点 union 均已运行；
- 15 个 K4 设计均已报告；
- 数值和实践等价类分开；
- F2 至 F5 状态和证据完整；
- 新 MEI-1 manifest 校验通过；
- verdict 与实际门结果一致；
- inconclusive 时 `allowed_next_stage=null`；
- supported 时也没有自动授权波形、benchmark 或硬件。

### 17.3 不要求的结果

本任务不要求 MEI-1 必须输出 supported。

如果严谨执行后仍得到：

```text
mei1_inconclusive_forward_model
```

但所有输入、计算、门和证据均正确，这次重冻结仍然是合格交付。它说明当前证据不足，而不是执行失败。

---

## 18. 最终交付报告模板

执行者最终只能按事实填写，不得省略失败项。

```markdown
# MEI-0 / MEI-1 版本化重冻结执行报告

## 代码与配置

- registry schema:
- manifest schema:
- git commit:
- relevant paths dirty:
- changed files:

## MEI-0

- freeze_id:
- freeze_dir:
- verdict:
- manifest_sha256:
- point_count_core:
- point_count_pressure:
- point_count_total:
- noise_profiles:
- delta_numerical_by_profile:
- delta_practical:
- allowed_next_stage:
- authorizations:

## MEI-1

- parent_mei0_freeze_id:
- freeze_id:
- freeze_dir:
- verdict:
- manifest_sha256:
- blockers:
- low_cost ranking span:
- registered_mrs2 ranking span:
- ambient result:
- pressure result:
- F2 status:
- F3 status:
- F4 status:
- F5 status:
- allowed_next_stage:
- authorizations:

## 验证

- focused tests:
- related regression tests:
- full tests:
- MEI-0 manifest verification:
- MEI-1 manifest verification:
- old freeze integrity verification:

## 未完成或风险

- unresolved blockers:
- unavailable evidence:
- unauthorized branches:
```

报告中不得写“基本通过”“大致稳定”“应该没问题”。只能使用脚本输出的明确状态、数值和哈希。

---

## 19. 本次重冻结后的阻断处置指南

本节用于处理有效 MEI-1 `20260728T021714539197Z_18a9e241d705` 暴露出的科学阻断。它不回写本次 freeze，也不授权 MEI-2、硬件试验或 COMSOL G2+。

### 19.1 先理解三个阻断

1. F2–F5 `not_represented`：只有人为代理，没有能限定真实偏差的可追溯证据。
2. 排名未越过 `delta_practical`：15 个 K4 在数值上有顺序，但工程改善只有 `0.148%–0.330%`，不足 2%。
3. 压力域未验证：216 个高压力点已经计算，但没有证据证明当前参数在 `0.5–0.709 MPa` 仍有效。

三项必须分别关闭。任何单项通过都不能替代另外两项。

### 19.2 Step A：建立证据矩阵

创建后续任务文档时，为 F2、F3、F4、F5 和高压力域逐行登记：

```text
对象
现有文献或数据路径
证据类型
参数或偏差定义
单位
温度范围
湿度范围
压力范围
频率范围
组分范围
上下界及其语义
实现或 holdout 路径
证据 SHA256
缺口
是否需要外部授权
```

缺少任一关键范围时写 `not_covered`，不得填“默认适用”或把空值写成零。

### 19.3 Step B：先做 F5 可行性决策

按以下顺序查找：已有台架标定、带不确定度的厂商幅相曲线、独立设备数据。三者均不存在时，记录：

```text
F5_status = blocked_by_missing_transducer_evidence
requires_explicit_hardware_authorization = true
```

没有明确硬件授权时停止 F5 新采集，不创建模拟标定数据，不把衰减 ripple 代理改名为设备证据。其他工作可以继续，但必须知道 MEI-1 仍不能 supported。

### 19.4 Step C：联合处理 F2 和压力域

优先寻找同时覆盖 25–200 kHz、目标温湿度和 `0.5–0.709 MPa` 的 H2O 弛豫参数。若只能取得常压证据：

```text
F2_ambient = candidate_traceable
F2_pressure = not_covered
pressure_domain = not_validated
```

不得把常压参数线性外推后登记为高压力已验证。压力验证必须预先定义比较量、允许误差、校准集、holdout 和失败处理，再读取正式结果。

### 19.5 Step D：处理 F3 和 F4

F3 只能采用可追溯耦合实现或独立参考。最低测试包括零耦合退化到 F0、参数单位和边界、有限差分梯度、目标域无非有限值。

F4 先冻结实际几何、孔径、声程、边界条件和频带，再选择解析包络或已授权高保真子集。结果必须转换为 MEI-1 实际使用的 TOF、相位或幅度偏差，不能只交付声场图片。

### 19.6 Step E：创建新 MEI-0

只有候选证据通过独立核对后，才修改活动 registry。每个拟清除阻断的 family 必须同时具备：

```text
status = represented_traceable 或 independent_holdout_available
refs 非空
implementation_or_holdout_path 非空
evidence_path 指向真实文件
evidence_sha256 与文件一致
parameter_or_bias_bounds 非空且有来源
bound_semantics 明确
can_clear_not_represented = true
```

压力域必须使用 `validated_traceable`，`validated_range_mpa` 同时覆盖 `0.5` 和 `0.709`。随后运行 MEI-0 专项测试、registry preflight、版本化 freeze 和 manifest 独立校验。旧 freeze 不改、不删。

### 19.7 Step F：重跑 MEI-1 并分流

新 MEI-1 继续运行两个噪声口径、环境 216、压力 216 和正式 432。禁止 stride 代替正式门。

| 结果                       | 处置                                       |
| ------------------------ | ---------------------------------------- |
| F2–F5、压力域、六个排名域和稳定性门全部通过 | 输出 supported；只获得 MEI-2 评审资格，不自动授权其他活动    |
| 物理和压力证据完整，但排名仍低于 2%      | 保留固定 K4；若需在 MEI-1 终止，先通过新 MEI-0 冻结新的状态转移 |
| 证据仍缺失                    | 保持 `mei1_inconclusive_forward_model`     |
| 合理模型族导致排名或瓶颈翻转           | 保持 inconclusive，先修正前向模型                  |
| 代码、配置、哈希或点数错误            | 输出 `mei1_audit_failed`，不得提升为科学事实源        |

任何情况下都不得按已看到的 0.456%–0.758% 排名跨度事后降低 `delta_practical=2%`。若要更换频带、观测类型、K 或路径长度，必须先在另一个新 MEI-0 中预注册成本和判定门。
