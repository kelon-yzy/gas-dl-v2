# A2-DYN 执行记录（从属于 13）

> 文档状态：从属于 [13_Ar-He-CO2动态时间序列仿真与数据分布规划](13_Ar-He-CO2动态时间序列仿真与数据分布规划.md)，阶段状态由主规划统一持有 \
> 承接内容：§13 的 §19–§22 执行事实；后续修订（A2-DYN-3R2 / A2-DYN-4R2）的执行事实追加在本文件 \
> 迁移日期：2026-09-04（D5 结构拆分，纯搬运逐字保留；节号沿用 §13，跨文件引用写作「13a §x」）\
> 拆分验收：三份文件合并正文与拆分前逐字一致

## 19. A2-DYN-1 执行事实

### 19.1 已终止的近似声速路径

首轮固定热容模型与后续 pair-virial 模型都能通过绝对声速误差门，但压力方向分别有 `1,091 / 30,906` 和 `1,398 / 30,906` 个不一致。根因是截断维里近似与 HEOS 多流体 Helmholtz 状态方程不是同一生成定义；pair-v2 因此保持为草案失败证据，不进入正式 R4。

### 19.2 当前正式声速生成定义

`a2dyn_direct_multifluid_eos_v1` 直接调用固定版本 CoolProp 8.0.0 的 HEOS `speed_sound()`。适配层只负责显式气相、`Ar/He/CO2` 组分顺序、注册温压域、运行时版本和二进制 hash 校验，不实现第二套 EOS，也不提供旧模型回退。完整网格、离网审计和压力方向结果如下：

| 审计项目 | A2-DYN-1R4 结果 |
| --- | ---: |
| 协议与生成资产 | `PASS` |
| 完整网格 | 185,436 点，最大相对差 `0` |
| 固定 seed 离网审计 | 10,000 点，最大相对差 `0` |
| 压力方向 | `0 / 30,906` 不一致 |
| 共享物性与设备 smoke | `PASS`，无理论 ToF 回退 |
| 验证范围 | `generator_consistency` |

执行产物为 [physics_audit_r4.json](../../outputs/summary/a2_dynamic_v1/physics_audit_r4.json) 和 [A2-DYN-1R4 manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-1r4-physics-smoke/manifest.json)。`A2-DYN-1` 的声速生成门通过，允许进入 `A2-DYN-2`；这不等同于独立验证 HEOS 的物理准确性。

## 20. A2-DYN-2 执行事实

`A2-DYN-2R4` 已由 `src/gf/pipeline/a2_dynamic_pilot.py` 完成并返回 `PILOT_QUALIFIED`。执行严格使用实验配置中的 240 组、六族各 40 组和 train / val / stress_val = 120 / 60 / 60；不生成 test、不写入正式 `data/a2_dynamic_v1`，也不把 `noise_seed`、`sequence_id` 或高频 ADC 数组写入产物。

| 项目 | 结果 |
| --- | ---: |
| 外层比较 | 1 / 2 / 5 Hz；120 / 240 / 360 s |
| 冻结外层轴 | `5 Hz / 240 s` |
| 冻结超声 | `US-CHIRP-XCORR-PARABOLIC-1`，`reference_xcorr_parabolic` |
| 超声探针 | 36 点；锁定率 `1.000`；linear chirp p95 ToF 误差约 `3.19e-9 s` |
| 动态双通道有效率 | `1.000` |
| 低频 t50 双通道分离率 | `0.9875` |
| 六族动态退化率 | `0` |
| TCD 最大能量 residual | `2.70e-14 W` |
| NDIR 最大饱和率 / 信号越界 | `0 / 0` |
| stress P060 O-KIN 相对 B-LAST | `44.99%` 信息增益 |
| 正式信号数组估算 | `90,720,000` bytes（signals 单数组） |
| pilot 资源峰值 / 高频波形持久化 | 约 `226 MB`（实际值随进程运行记录） / `0` bytes |

为保证 HEOS 只在同一物理轨迹上比较，pilot 在 5 Hz / 360 s 参考网格上完成一次直接 HEOS 计算，低频候选只做外层重采样与时长截断；正式冻结为 5 Hz / 240 s。正式选择已回写到 `configs/data/ar_he_co2_a2_dynamic_v1.json` 和 `configs/experiment/a2_dynamic_protocol.json`。证据文件为 [pilot_audit_r4.json](../../outputs/summary/a2_dynamic_v1/pilot_audit_r4.json)、[A2-DYN-2R4 manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-2r4-pilot/manifest.json) 和 [resolved_config.json](../../outputs/runs/a2_dynamic_v1/a2-dyn-2r4-pilot/resolved_config.json)。

## 21. A2-DYN-3 执行事实

> **R1 口径作废注记（2026-09-04）**：本节为 A2-DYN-3 首轮（R1 口径）执行事实。R1 的 `relative_degradation` 分子分母来自不同样本总体、oracle 门为空门（F1/F2/F3，见 [15_A2-DYN审计缺陷修复规划](archive/15_A2-DYN审计缺陷修复规划.md)，已归档），本节退化与 headroom 数值不得按面值引用；修正口径的重审计见 [§23 A2-DYN-3R2 执行事实](#23-a2-dyn-3r2-难度重审计执行事实)。原始产物保留于 `outputs/summary/a2_dynamic_v1/a2_dyn_3_audit.json`、`outputs/runs/a2_dynamic_v1/a2-dyn-3-development/` 供对照。

`A2-DYN-3` 已由 `src/gf/pipeline/a2_dynamic_benchmark.py` 完成并返回 `DIFFICULTY_QUALIFIED`。开发包只生成 train、val、stress_val，不生成 test；6 个 family 全部合格，`D-IID` 通过，5 个相互独立的动态压力轴进入 `eligible_dynamic_axes.json`，无失败要求。审计前强制重跑 physics smoke，并对 19 个源依赖和配置执行 freshness 校验，结果均为 `PASS`。

| 项目 | 结果 |
| --- | ---: |
| 唯一 `mixture_id` / group | `3,780` |
| train | 2,520 groups / 3,600 observations |
| val | 630 groups / 900 observations |
| stress_val | 630 groups / 900 observations |
| signals | `[5400, 3, 1200, 1]`，float32 |
| oracle clean / device state | `[5400, 3, 1200]` |
| 合格 family | `D-IID`、`D-KINETICS`、`D-PROTOCOL`、`D-NOISE-DRIFT`、`D-ENV-CAL`、`D-JOINT` |
| eligible 动态轴 | `D-KINETICS`、`D-PROTOCOL`、`D-NOISE-DRIFT`、`D-ENV-CAL`、`D-JOINT` |
| 动态非退化 / Jacobian | `PASS / PASS`；有效通道率与量化级数率均为 `1.000`，低频 t50 配对率 `0.8335`；固定目标、联合参数和剔除 nuisance 后目标满秩率均为 `1.000` |
| Jacobian 条件数 | 固定目标 P95 `45.56`；联合目标 P95 `38.88`，均低于门限 `1000` |
| 设备与边界 | NDIR 饱和率 `0`，信号越界率 `0`，超声锁定率 `1.000`；peak correlation `0.96494–0.97706`、SNR `318.37–397.32`、ToF 不确定度 `5.03e-10–6.28e-10 s`；TCD 最大能量 residual `2.7263e-14 W` |
| 数据 `content_sha256` | `82837b52b54f1d76ebc5b72a5eae796c931e3e86ad26ee52b7f1161661355a1d` |
| 审计 `audit_sha256` | `3f4ac9d19c614f88c235b2491c358c23b5d4b7b8438c61814153289fc8d8132b` |

基线资格门中各 family 的 B-LAST、O-EQ、O-KIN 拟合状态均为 `PASS`。B-LAST 使用显式记录的 `lbfgs`、`max_iter=2000`、`tol=1e-3`，不再硬编码成功状态；O-KIN 使用 clean device signal、inlet protocol 和特权 kinetics 参数，先以注册 HITRAN 曲线反演 CO₂，再在注册 HEOS 的 1% simplex 分段线性 ToF 曲线上做有界一维反演，不读取 `target` 或 `chamber_composition`。每个 family 的 O-KIN headroom 在 3 个早期前缀通过；D-IID、D-KINETICS、D-PROTOCOL 的相对退化在 2 个早期前缀通过，D-NOISE-DRIFT、D-ENV-CAL、D-JOINT 在 3 个早期前缀通过，满足至少 2 个前缀的阶段门。oracle 组成在 float32 序列化后最大和误差为 `0`，由显式序列化闭合逻辑保证，不靠审计放宽容差。

执行产物为 [a2_dyn_3_audit.json](../../outputs/summary/a2_dynamic_v1/a2_dyn_3_audit.json)、[eligible_dynamic_axes.json](../../outputs/summary/a2_dynamic_v1/eligible_dynamic_axes.json)、[数据 manifest](../../data/a2_dynamic_v1/manifest.json)、[audit.json](../../data/a2_dynamic_v1/audit.json) 和 [A2-DYN-3 audit manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-3-development/audit_manifest.json)。A2-DYN-4 的 test 生成与完整数据冻结已执行，见 [§22](#22-a2-dyn-4-执行事实)。

## 22. A2-DYN-4 执行事实

> **R1 口径作废注记（2026-09-04）**：本节为 A2-DYN-4 首轮（R1 口径）冻结执行事实，其中"test 动态非退化 `PASS`（有效通道率 `1.000`）"基于未按行噪声缩放的名义判据（F4），不得按面值引用；修正口径的冻结重审计失败，见 [§24 A2-DYN-4R2 冻结重审计执行事实](#24-a2-dyn-4r2-冻结重审计执行事实)。原始产物保留于 `outputs/summary/a2_dynamic_v1/a2_dyn_4_freeze_audit.json`、`outputs/runs/a2_dynamic_v1/a2-dyn-4-test/` 供对照。

`A2-DYN-4` 已由 `src/gf/pipeline/a2_dynamic_benchmark.py --stage generate-test` 完成并返回 `DATA_FROZEN`（2026-09-03）。生成前先重跑 physics smoke 并强制 `PASS`；开发子集（3,780 groups / 5,400 observations）的 manifest 与 audit 先备份到 [development_subset_backup](../../outputs/runs/a2_dynamic_v1/a2-dyn-4-test/development_subset_backup/)（内容 hash `82837b52…` 与 A2-DYN-3 存档一致），随后 test 配方从同一低差异池中、以开发 3,780 个已占用坐标之外的唯一点抽取；3 个规范 pure 顶点按配置替换 D-JOINT/test 的 3 个 binary 配额，固定在该 family 尾部，不混入数字编号流（test 数字组 ID 连续为 `a2dyn-mix-0003781…0004407`）。聚合写盘后执行冻结审计（schema / 守恒与设备 / test 动态非退化 / pure 边界四类），并重算 content hash 与 audit hash。执行过程中修正了两处实现缺陷：数组拼接的键映射错误（`inlet_composition` vs `inlet`）与 pure 顶点边界审计把 pure-He / pure-CO2 误当 purge 恒等序列；两者均导致审计显式失败后修复，未引入静默降级。

| 项目 | 结果 |
| --- | ---: |
| 唯一 `mixture_id` / group | `4,410` |
| train | 2,520 groups / 3,600 observations |
| val | 630 groups / 900 observations |
| stress_val | 630 groups / 900 observations |
| test | 630 groups / 900 observations |
| signals | `[6300, 3, 1200, 1]`，float32 |
| test 区域配额（interior / near_boundary / binary / pure） | 组级 `315 / 189 / 123 / 3`，行级与冻结配置一致 |
| pure 顶点 | `a2dyn-mix-pure-Ar/He/CO2`，仅 D-JOINT/test，各 2 条观测 |
| 开发 ↔ test group / 组成零交集 | `PASS`（4,410 组全部唯一；非纯气坐标 4,407 个互不重复） |
| 完整包 schema 审计 | `PASS`（records / groups / 区域 / hash / 数组不变量全部通过） |
| 守恒与设备审计（6,294 非 pure 行） | `PASS`：信号越界 `0`、NDIR 饱和率 `0`、超声锁率 `1.000`、TCD 最大能量残差 `2.7263e-14 W` |
| test 动态非退化（897 非 pure 行） | `PASS`：有效通道率 `1.000`、量化级数率 `1.000`、t50 配对率 `0.9004`、六族退化率 `0` |
| pure 边界审计 | `PASS`：pure-Ar 目标等于 purge 且 clean 全程静态；pure-He / pure-CO2 clean 有预期动态；越界 `0`、饱和 `0`、锁率 `1.000` |
| 完整包 `content_sha256` | `3da0e478eca52bb6a31e1fe2c2d5b3d066341fec3516be1a29fe7ed3077aeb95` |
| 冻结审计 `audit_sha256` | `87344bb532ab56dfd0a28b6b938c8b9641e9231725043a3869bceda75a9a54f0` |
| 开发子集 content（存档，不覆盖） | `82837b52b54f1d76ebc5b72a5eae796c931e3e86ad26ee52b7f1161661355a1d` |
| physics smoke 重跑 / dataset freshness | `PASS / PASS`（源依赖 hash 已在最终冻结时重绑定） |

test 与开发 split 可同文件存储（§8.1），`data/a2_dynamic_v1/audit.json` 已替换为 A2-DYN-4 冻结审计；A2-DYN-3 难度审计完整存档于 `outputs/summary/a2_dynamic_v1/a2_dyn_3_audit.json`、`outputs/runs/a2_dynamic_v1/a2-dyn-3-development/audit_manifest.json` 与 `development_subset_backup/`。执行产物为 [a2_dyn_4_freeze_audit.json](../../outputs/summary/a2_dynamic_v1/a2_dyn_4_freeze_audit.json)、[A2-DYN-4 manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-4-test/manifest.json) 和 [resolved_config.json](../../outputs/runs/a2_dynamic_v1/a2-dyn-4-test/resolved_config.json)。截至本节对应的 2026-09-03 记录，A2-DYN-5 尚未执行；当前开发侧执行事实见 §25。

## 23. A2-DYN-3R2 难度重审计执行事实

A2-DYN-3 的 R2 修正口径重审计（2026-09-04，工作包 [15_A2-DYN审计缺陷修复规划](archive/15_A2-DYN审计缺陷修复规划.md) 的 F1/F2/F3/F5/F6）在冻结数据上完成。数据内容零变化：开发子集由完整包按 A2-DYN-4 备份 manifest 重建，`content_sha256` 逐位复现 `82837b52b54f1d76ebc5b72a5eae796c931e3e86ad26ee52b7f1161661355a1d`；未重新生成任何序列。判定终态：**`DIFFICULTY_QUALIFIED`（A2-DYN-3R2）**，解除 A2-DYN-5 的开发侧实现阻塞；正式 test 与完整阶段终态仍受 A2-DYN-4 的 `DATA_FROZEN` 门约束。

| 项目 | 结果 |
| --- | ---: |
| 口径修订 | `relative_degradation` 配对总体（主参照 P120、副 P150）、门控 oracle 改 O-KIN-OBS（噪声受限）、O-EQ 改同模型类 `_fit_small_mlp`、Jacobian 三 split × 216 样本采样声明、动态非退化按行噪声缩放 + phase 覆盖 + transition 方差比 |
| 配对行数（P120 主参照） | `D-IID` 180 / `D-KINETICS` 90 / `D-PROTOCOL` 68 / `D-NOISE-DRIFT` 270 / `D-ENV-CAL` 90 / `D-JOINT` 180，全部 ≥ 60 门 |
| P150 副参照配对行数 | 31–141（P150 与 recovery 边界重合，约半数序列失效，只报告不判定） |
| 配对 `relative_degradation`（P015 / P030 / P060） | P015 全族 `2.15–5.85`（远超 0.25 门）；P030 全族 `0.78–3.10`；P060 部分族低于 0.25（`D-IID` 0.122、`D-NOISE-DRIFT` 0.159、`D-ENV-CAL` 0.195、`D-KINETICS` 0.188），无碍——每族 P015 + P030 两个 horizon 过门 |
| O-KIN-OBS val headroom | 0.42–0.92（全部 ≥ 0.20 门），反演失败率按行显式记录 0–6.7%（`D-PROTOCOL` P030 最高 0.067） |
| O-KIN（clean）可逆性上界 | 只报告不判定（F2 语义） |
| O-EQ ≤ B-LAST | 17/18 格点成立；唯一违反 `D-IID` P060（O-EQ 0.02180 vs B-LAST 0.02033），按 F3 记为需单独记录的物理发现，保留方案 A，不回退 Ridge |
| Jacobian（216 样本，采样声明） | `PASS`：fixed / joint-parameter / joint-target 满秩率均 `1.000`，fixed 条件数 P95 `114.3`（P015）、`80.1`（P030）、`62.7`（P060），joint-target P95 `38.3`，均低于门限 `1000`；逐 horizon 只报 fixed 口径（单 horizon joint 投影不适定，见代码注释） |
| 动态非退化（开发，全 split） | `PASS`：scaled active `0.9981`、退化率 max `0.0037`、phase 覆盖 `1.0`、transition 方差比通过率 `0.9986` |
| family gate | 6 family 全部 `QUALIFIED`；`eligible_dynamic_axes` = `D-KINETICS`、`D-PROTOCOL`、`D-NOISE-DRIFT`、`D-ENV-CAL`、`D-JOINT`；`failed_requirements` = `[]` |
| 审计 `audit_sha256` | `a8ac1cf624cd5cbf235e03961eefbf269084d8954469d31d5d05edfc54691576`（审计模块结构复核后重登记；数据内容与数值结论不变） |

执行产物：[a2_dyn_3r2_audit.json](../../outputs/summary/a2_dynamic_v1/a2_dyn_3r2_audit.json)、[A2-DYN-3R2 audit manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-3r2-reaudit/audit_manifest.json)；旧 `eligible_dynamic_axes.json` 归档至 [eligible_dynamic_axes_a2dyn3_r1.json](../../outputs/archive/a2_dynamic_v1/eligible_dynamic_axes_a2dyn3_r1.json)。R1 口径数值见 §21 并作废（R1 首轮 `audit_sha256` `3f4ac9d1…` 与 §21 表同属作废口径）。

## 24. A2-DYN-4R2 冻结重审计执行事实

A2-DYN-4 的 R2 修正口径冻结重审计（2026-09-04）在完整包上完成。数据内容零变化：完整包 `content_sha256` 仍为 `3da0e478eca52bb6a31e1fe2c2d5b3d066341fec3516be1a29fe7ed3077aeb95`（不 rebind source hash）。判定终态：**`DATA_FREEZE_FAILED`（A2-DYN-4R2）**，唯一失败项 `test_dynamic_non_degenerate`。R1 口径的 `DATA_FROZEN` 声明按项目决策降级为 `DATA_FREEZE_FAILED` 并登记缺陷（2026-09-04）；v1 数据保留不重新生成，禁止调阈值补救。

| 项目 | 结果 |
| --- | ---: |
| schema / physics / pure 边界（完整包） | `PASS / PASS / PASS` |
| test 动态非退化（897 非 pure 行） | **`FAIL`**：scaled active `0.9329` < 0.95 门；`D-JOINT` 退化率 `0.1954`、`D-NOISE-DRIFT` `0.1111` > 0.05 门（unscaled 口径 `1.000 / 0` 作对照——R1 名义门虚高确认） |
| 退化根因 | NOISE-10X 只分布于 test（`D-NOISE-DRIFT` 270 行 + `D-JOINT` 180 行 = 450/897 行），10 倍噪声档下 11–20% 行双通道 peak-to-peak < 50σ（低于自身噪声底的可辨识线） |
| 处置（S5 判定，2026-09-04；2026-09-04 契约复核收紧） | 冻结状态降级 `DATA_FREEZE_FAILED` + 缺陷登记；v1 test 只保留冻结审计证据，不进入 A2-DYN-5 模型评价；v2 数据规划时调整 test 噪声档分布并重新冻结；不改 §6 分布、不放宽 §10.5 判据 |
| 审计 `audit_sha256` | `cfcd8d6bb210dab9b079ad9a50b7a279e473f76bd55dcd0a4e23d931b05a97e8`（审计模块结构复核后重登记；数据内容与失败结论不变） |

`data/a2_dynamic_v1/manifest.json` 的 `audit_status` / `status` 已由 pipeline 重写为 `DATA_FREEZE_FAILED`（`content_sha256` 不在重写范围）。执行产物：[a2_dyn_4r2_freeze_audit.json](../../outputs/summary/a2_dynamic_v1/a2_dyn_4r2_freeze_audit.json)、[A2-DYN-4R2 freeze manifest](../../outputs/runs/a2_dynamic_v1/a2-dyn-3r2-reaudit/freeze_audit_manifest.json)。R1 口径数值见 §22 并作废。A2-DYN-5 的开发侧基线、B-REF、回放与时间门证据追加于 §25；正式 test 仍不得读取。

## 25. A2-DYN-5 开发侧基线与因果回放执行事实

A2-DYN-5 在 2026-09-04 继续执行。实现入口为 `gf.pipeline.a2_dynamic_benchmark` 的 `baselines`、`replay-smoke`、`report` 和 `handoff` stage；模型实现集中在 `src/gf/dl/temporal_baselines.py`，编排集中在 `src/gf/pipeline/a2_dynamic_baselines.py`。运行层只读取 train / val / stress_val，test 行数为 `0`；`DATA_FREEZE_FAILED` 不被改写为受限解锁或正式通过。

| 项目 | 结果 |
| --- | ---: |
| 开发基线状态 | `DEVELOPMENT_BASELINES_COMPLETE` |
| 数据状态 / content hash | `DATA_FREEZE_FAILED` / `3da0e478eca52bb6a31e1fe2c2d5b3d066341fec3516be1a29fe7ed3077aeb95` |
| 模型矩阵 | B-LAST、B-DELTA、B-EWMA、B-STAT、B-TCN、B-GRU、B-STEADY、O-EQ、O-KIN、O-KIN-OBS；可训练模型五个 seed，数值 oracle 明确标记 deterministic |
| 预测物化 | `1,309,528` 行；包含 train / val / stress_val，不含 test；唯一键无重复 |
| B-REF | P005 / FULL 选 B-EWMA；P015 / P030 / P060 / P120 / P150 选 B-LAST；选择只看 val 的 D-IID，未读取 test |
| 时间增量信息门（开发证据） | `TEMPORAL_REDUNDANT`：B-STAT、B-TCN 在 P015 / P030 / P060 均未满足预注册改善、seed 同向、组件退化和 paired group bootstrap 联合条件；D-IID 与压力族均分开计算 |
| O-KIN-OBS headroom | stress_val P060 相对简单候选仍保留约 `0.80`，反演失败按行记录，不作为正式 qualification |
| 因果回放 | `REPLAY_SMOKE_COMPLETE`；12 条 val observation、14,400 次 virtual-clock 更新、无未来填充、跨 observation 状态重置；wall-clock p95 约 `0.0271 ms`，低于 `0.2 s` 更新周期 |
| 报告 | [a2_dyn_5_report.md](../../outputs/reports/a2_dynamic_v1/a2_dyn_5_report.md)；机器汇总 [a2_dyn_5_baselines.json](../../outputs/summary/a2_dynamic_v1/a2_dyn_5_baselines.json)、[a2_dyn_5_replay_smoke.json](../../outputs/summary/a2_dynamic_v1/a2_dyn_5_replay_smoke.json) |
| A2-DYN-6 handoff/关闭 | `A2_DYN_6_BLOCKED_DATA_FREEZE_FAILED`；见 [a2_dyn_6_closure.json](../../outputs/summary/a2_dynamic_v1/a2_dyn_6_closure.json)，未生成 `dynamic_handoff.json`，未启动新算法搜索 |

本节的 `TEMPORAL_REDUNDANT` 是开发 split 上的候选结论，不能越过 A2-DYN-4R2 的冻结硬门。正式 test 评价、完整 A2-DYN-5 终态和 A2-DYN-6 handoff 仍保持 `FORMAL_BLOCKED_DATA_FREEZE_FAILED`；v1 不因该结果启动新算法搜索。
