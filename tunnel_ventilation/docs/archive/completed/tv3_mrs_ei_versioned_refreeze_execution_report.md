# MEI-0 / MEI-1 版本化重冻结执行报告

> 执行日期：2026-07-28
>
> 验收结论：版本化重冻结技术执行通过；MEI-1 科学门未通过，结论为 `mei1_inconclusive_forward_model`。`allowed_next_stage=null`，不得进入 MEI-2。
>
> 历史状态说明：本报告只记录上一轮 freeze。当前状态已由 [F2--F5 证据处置执行报告](tv3_mrs_ei_f2_f5_disposition_execution_report.md) 更新为 `mei1_fixed_k4_retained` 和 `allowed_next_stage=MEI-3_varpro_audit`；四类授权的当前释义也以新报告为准。

## 1. 代码与配置

- registry schema：`tunnel-ventilation-mrs-ei-registry-2`
- manifest schema：`tunnel-ventilation-mrs-ei-freeze-manifest-2`
- freeze 记录的 Git commit：`bab35f0041ac4d518467fccc0b92ce4e35d60183`
- freeze 记录的 relevant paths dirty：`true`
- registry 配置：`configs/tv3_mrs_ei/model_family_registry.json`、`design_space.json`、`metric_registry.json`
- MEI-1 运行配置：`configs/tv3_mrs_ei/mei1_forward_envelope.json`
- 核心审计实现：`tv3/audit/mrs_ei_registry.py`、`tv3/audit/mrs_ei_forward_envelope.py`
- 冻结入口：`scripts/run_tv3_mei0_registry_freeze.py`、`scripts/run_tv3_mei1_forward_envelope.py`
- 专项测试：`tests/test_tunnel_ventilation_mei0_registry.py`、`tests/test_tunnel_ventilation_mei1_forward_envelope.py`

本次修复关闭了审查发现的 P1 / P2：F0 / F1 证据同时要求路径、SHA256、引用和实现或留出路径；F2–F5 不再按 family ID 永久硬锁，但只有结构化可追溯证据才能改变状态；高压力有效性改为结构化范围证据校验，不再接受关键词命中；每个噪声口径的环境域、压力域和正式并集排名必须分别可判定；求解门的配对置信区间下界必须超过 `delta_practical`；冻结 manifest 改为校验 freeze 内不可变快照。

## 2. MEI-0

- freeze_id：`20260728T020915548649Z_f47f6f51d1b1`
- freeze_dir：`outputs/runs/tv3_mrs_ei/mei0_registry/freezes/20260728T020915548649Z_f47f6f51d1b1`
- parent freeze_id：`20260728T012236268746Z_228bce47cb0d`
- verdict：`mei0_registry_frozen`
- manifest SHA256：`cc6572ad81e9fe0e1e52d6859f400e664bfca6b8b243485e29a8aa7ffa68eea1`
- point_count_core：`216`
- point_count_pressure：`216`
- point_count_total：`432`
- noise_profiles：`low_cost_k4_primary`、`registered_mrs2_stress`
- `delta_numerical_by_profile.low_cost_k4_primary`：`2.8210630817324963e-05`
- `delta_numerical_by_profile.registered_mrs2_stress`：`9.055394884241296e-05`
- `delta_numerical_shared_upper_bound`：`9.055394884241296e-05`
- `delta_practical`：`0.02`
- freeze 时 `allowed_next_stage`：`MEI-1_forward_envelope`
- 最终当前 `allowed_next_stage`：`null`
- authorizations：四项均为 `forbidden_until_explicit_authorization`

MEI-0 manifest 保存并校验 registry、活动计划与执行指南快照、冻结脚本、registry 审计、共享前向、MRS 上游判决及 F0 / F1 来源证据。后续活动文档修改不会改变该冻结的 manifest 校验结果。

## 3. MEI-1

- parent MEI-0 freeze_id：`20260728T020915548649Z_f47f6f51d1b1`
- parent MEI-0 manifest SHA256：`cc6572ad81e9fe0e1e52d6859f400e664bfca6b8b243485e29a8aa7ffa68eea1`
- freeze_id：`20260728T021714539197Z_18a9e241d705`
- freeze_dir：`outputs/runs/tv3_mrs_ei/mei1_forward_envelope/freezes/20260728T021714539197Z_18a9e241d705`
- verdict：`mei1_inconclusive_forward_model`
- manifest SHA256：`76d08c9604690e5f66b77553eb83f18edc315f25751f88788bbe2212dca51423`
- formal point set：`formal_mei1_432`
- point sets：`ambient_core_216`、`pressure_extension_low_rh_216`
- design count：`15`
- noise profiles：`low_cost_k4_primary`、`registered_mrs2_stress`
- COMSOL holdout：`unavailable`
- registered sparse simulation generation review eligible：`false`
- allowed_next_stage：`null`
- authorizations：四项均为 `forbidden_until_explicit_authorization`

### 3.1 阻断项

1. `unrepresented_families_without_flip_proof:F2_h2o_relaxation_params,F3_coupled_relaxation,F4_diffraction_near_field,F5_transducer_response`
2. `design_ranking_not_resolvable_within_delta_practical`
3. `pressure_domain_not_validated`

F2、F3、F4、F5 均为 `not_represented`。代理扰动只能作诊断，不能清除该状态。高压力点没有被删除，216 个压力扩展点已进入正式门；但 `pressure_domain_evidence.status=not_validated`、`validated_range_mpa=null`，所以压力域仍被显式阻断。

### 3.2 双噪声、双域结果

| 噪声口径 | 环境域排名跨度 | 压力域排名跨度 | 正式并集排名跨度 | 基线 K4 最大 / 中位 P90 |
| --- | ---: | ---: | ---: | ---: |
| `low_cost_k4_primary` | `0.007581900768250417` | `0.004564833712193834` | `0.004564833712193834` | `1.7013596774527533 / 1.3717225374211606` |
| `registered_mrs2_stress` | `0.0071170280697116966` | `0.0067251760506776255` | `0.0067251760506776255` | `7.572841247748836 / 6.448175119196189` |

六个噪声口径与排名域组合的 `ranking_resolvable_numerical=true`，但 `ranking_resolvable_practical=false`；全部 15 个设计仍属于同一个实践等价类。不得把数值可排序误写为具有实践意义的设计优越性。

### 3.3 阻断项解释和关闭条件

这些阻断项不是技术运行失败，而是科学证据不足。三项互相独立：补齐模型族证据不会自动让设计差异超过 2%，加入压力点也不会自动证明压力模型有效。

#### F2–F5

| Family | 当前事实 | 关闭条件 |
| --- | --- | --- |
| F2 | H2O 弛豫强度和催化参数是未验证 proxy | 取得覆盖目标温湿度、频率和压力的参数定义、上下界和来源；实现可信包络或独立 holdout |
| F3 | 5% cross-mix 是诊断假设，不是耦合模型 | 实现可追溯耦合模型或使用独立数值参考；验证零耦合退化、梯度和适用域 |
| F4 | 衍射代理未绑定实际几何和可信误差 | 使用适用几何的解析包络，或经授权的独立高保真子集；报告 TOF、相位和幅度偏差 |
| F5 | 衰减 ripple 不代表真实换能器幅相和群延迟 | 使用已有台架、厂商曲线及不确定度或独立设备数据；缺少数据和硬件授权时保持阻断 |

每项正式证据必须包含引用、实现或 holdout 路径、证据文件、匹配 SHA256、参数或偏差边界及其语义。当前代理均未发生瓶颈翻转，但代理边界没有物理来源，因此该结果不能清除 `not_represented`。

#### 设计排名

当前最佳 K4 相对固定 K4 的改善只有 `0.148%–0.330%`，最好与最坏设计的跨度只有 `0.456%–0.758%`。这些差异明显高于数值误差，但仍低于 `delta_practical=2%`，说明当前 15 个 K4 在工程上等价。合理选择是先补齐物理后复算；若结论不变，保留固定 K4，或通过新的 MEI-0 预注册更有差异的频带、观测类型、K 或路径长度。不得按本次结果事后降低 2% 门。

#### 高压力域

MEI-1 已计算 216 个 `0.5 / 0.709 MPa` 点，但 `pressure_domain_evidence.status=not_validated`。关闭条件是取得覆盖完整压力范围的可追溯文献、独立模型或台架证据，并预先定义温湿度、组分、频率、比较指标、允许误差和 holdout 规则。仅有高压力输出或单一平衡声速证据不足以验证完整前向链路。

## 4. 技术失败运行

- 目录：`outputs/runs/tv3_mrs_ei/mei1_forward_envelope/freezes/20260728T021007588245Z_18a9e241d705`
- 原因：`mei1_run_config.json` 的 artifact SHA256 与 manifest 不一致。
- 处置：修复 manifest 生成逻辑后重新运行；失败目录只读保留，不删除、不改写、不提升到 `configs/tv3_mrs_ei/stage_status.json`。
- 有效替代运行：`20260728T021714539197Z_18a9e241d705`。

## 5. 验证

- focused tests：`56 passed in 2.21s`。
- related MRS regression tests：`34 passed in 1.00s`；测试集合为 MRS physics、MRS identifiability、MRS-6 budget 和通用 tv3 identifiability。
- compile checks：通过。
- MEI-0 registry preflight：`mei0_registry_frozen`，issues 为 `[]`。
- MEI-0 manifest verification：`[]`。
- MEI-1 manifest verification：`[]`。
- stage status consistency：MEI-0 与 MEI-1 路径均精确匹配有效 freeze，`allowed_next_stage=null`，8 个授权字段全部为 `forbidden_until_explicit_authorization`，review eligible 为 `false`。
- full tests：此前整套测试首先失败于与本次改动无关的 `test_d2b_frame_fidelity_audit.py`；Windows 占用 `slow.npy` 导致 `WinError 32`。未添加绕过、mock 或吞错逻辑。

文档更新后已重新执行专项测试、相关回归、两个有效 manifest 的独立校验和状态文件一致性检查。`git diff --check` 未发现空白错误；仅报告用户已有文档的 CRLF 到 LF 提示，不影响本次验收。

## 6. 未完成或风险

- unresolved blockers：F2–F5 缺少可追溯的参数包络、实现或独立留出证据；15 个 K4 在 `delta_practical=0.02` 内不可分辨；0.5–0.709 MPa 范围缺少结构化参数有效性或留出证据。
- unavailable evidence：COMSOL 本地声学 holdout 不可用；换能器实测幅相与声功率证据不可用。
- unauthorized branches：MEI-2、登记稀疏仿真生成、正式波形生成、benchmark 打包和硬件试验均未授权。
- 状态机要求：`configs/tv3_mrs_ei/stage_status.json` 必须继续精确指向本报告列出的两个有效 freeze，并保持 `allowed_next_stage=null`。

## 7. 后续执行顺序

1. 建立 F2–F5 与高压力域证据矩阵，盘点现成文献、数据、适用范围和授权缺口。
2. 优先确认 F5 是否存在可用标定；无数据且无新采集授权时，立即记录为持续阻断，不先投入大规模算法开发。
3. 联合处理 F2 与高压力证据，禁止把常压参数外推到 0.5–0.709 MPa。
4. 为 F3 建立耦合实现或独立参考；为 F4 建立解析包络或经授权的高保真子集。
5. 若 F5 必须新采集，先完成独立硬件授权和最小标定协议，不由本报告隐式授权。
6. 证据和实现完成后创建新的版本化 MEI-0，冻结路径、哈希、边界、适用域和状态转移。
7. 从新 MEI-0 重跑 432 点、双噪声、三排名域 MEI-1，并重新验证 manifest。
8. 全部门通过才评审 MEI-2；物理证据完整但排名仍低于 2% 时保留固定 K4；证据缺失或排名翻转时继续保持 inconclusive。

执行过程中不得删除压力点、用代理清除 `not_represented`、根据当前结果降低 `delta_practical`，也不得修改旧 freeze。
