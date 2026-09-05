# A2-DYN 审计缺陷修复规划（A2-DYN-3R2）

> **归档说明（2026-09-05）**：本文件已按总体规划 v11 移入 `docs/algorithm/archive/`，状态 `ARCHIVED / REMEDIATION_COMPLETE`。S1–S7 修复已全部执行完成，工作单本身没有未完成项，因此不再作为活跃规划。v11 §2.2 暂停 A2-DYN 后续扩建与完整 v2 生成，本文提到的后续动作（P1/P2 剩余整改、v2 相关准备）不再排期。审计口径修正的**结论与数值**以 [13a 执行记录 §23/§24](../13a_A2-DYN执行记录.md) 为准，物理与审计规格以 [13b](../13b_A2-DYN物理与审计规格.md) 为准；本文只保留 F1–F9 / D1–D6 的缺陷定义与修复依据，供 `criteria_revision` 字段溯源。归档范围见 [archive/README](README.md)。

> 制定日期：2026-09-04 \
> 文档状态：`REMEDIATION_COMPLETE / S1-S7_DONE / CHANGES_UNCOMMITTED` \
> 执行进度：见 §6 执行进度（§6.7 记录 S3–S5，§6.8 记录 S6–S7） \
> 工作包定位：`A2-DYN-3` 难度审计的口径与实现修复，不改动已冻结的 `data/a2_dynamic_v1` 数据内容 \
> 上位事实源：[13_Ar-He-CO2动态时间序列仿真与数据分布规划](../13_Ar-He-CO2动态时间序列仿真与数据分布规划.md)（2026-09-04 起按 D5 拆为 13 主规划 + [13a 执行记录](../13a_A2-DYN执行记录.md) + [13b 规格](../13b_A2-DYN物理与审计规格.md)） \
> 前置状态：A2-DYN-3 `DIFFICULTY_QUALIFIED`（R2 口径，2026-09-04）、A2-DYN-4 `DATA_FREEZE_FAILED`（R2 口径，2026-09-04） \
> 阻塞关系：F1、F2、F3 已在 S1 完成；A2-DYN-5 仅开发侧实现解除阻塞，正式 test、完整阶段终态与 `DYNAMIC_QUALIFIED` 仍由 A2-DYN-4 的 `DATA_FROZEN` 硬门阻塞

> **引用说明（2026-09-04 S7 后）**：正文中的「§13 第 X 行」行号引用基于拆分前 1,382 行版本，已随 D5 拆分失效；「§13 §x.y」条文引用按新归属理解为：主规划保留 §0–§3/§5/§7/§9/§11/§12/§14/§16/§17/§18，§4/§6/§8/§10/§13(原)/§15 迁至 13b，§19–§22 迁至 13a（节号均沿用原编号）。

## 0. 结论与摘要

对规划文档 §13、A2-DYN-3 相关的 6 个源文件（约 9,700 行）和两份审计产物做了一次完整复核。结论分三条：

1. **`DIFFICULTY_QUALIFIED` 的定性结论不被推翻。** P015 的 B-LAST 相对退化为 1.59–4.20 倍、P030 为 0.46–2.29 倍，离 0.25 阈值有 2–17 倍余量；已识别的偏差量级撑不到这个尺度。
2. **但 §21 报告的具体退化数值不成立，不得按面值引用。** 晚期参照 P150 只在约一半样本上有效，且是"暴露时间更长"的那一半，比值同时混入了 horizon 效应与子集效应，方向是把退化幅度放大。
3. **两个 oracle 门当前是空门。** O-KIN 在无噪 clean 信号上反演，误差比 B-LAST 低三个数量级，`min_oracle_headroom_vs_last=0.20` 在任何数据上都会通过；O-EQ 名义为"可达上界"，实测比 B-LAST 差 2.6 倍。这两项直接决定 A2-DYN-5 的 §11.6 能否回答"是否值得设计新算法"，必须在 A2-DYN-5 前修。

本规划共 15 个修复项：`F1–F9` 为代码与配置层，`D1–D6` 为规划文档 §13 自身的规格与表述层。分 P0/P1/P2 三档：P0（F1–F3、D1）阻塞 A2-DYN-5；P1（F4–F7、D2–D4）为审计可信度补强，可与 A2-DYN-5 并行；P2（F8–F9、D5–D6）为工程与文档整改，不阻塞。

文档层问题不是附带的排版整改。D1 是 F1 的规格层根因——§11.4 从未规定退化比值的样本总体，实现只是填了个空；只修实现不修规格，下一轮会重新长出同一个缺陷。因此 D1 与 F1 必须同批提交，D2/D3/D4 与对应的 F5/F6/F2+F3 同批提交。

**不做的事**：不重新生成数据、不修改 `content_sha256`、不放宽任何已冻结阈值、不因修复结果反向调整 §6 分布。若修复后 D-IID 或压力轴的难度门反转，按 §12.4 的既有规则判失败终态并升 revision，不做参数补救。

## 1. 缺陷清单与证据

| ID | 缺陷 | 严重度 | 证据 | 影响面 |
| --- | --- | --- | --- | --- |
| F1 | `relative_degradation` 的分子分母来自不同样本总体 | P0 | 见 §1.1 | A2-DYN-3 §11.4 门 1；A2-DYN-5 §11.5 的 B-STEADY/P150 late 门 |
| F2 | O-KIN headroom 门恒真 | P0 | 见 §1.2 | A2-DYN-3 §11.4 门 2；A2-DYN-5 §11.6 `min_oracle_headroom=0.10` |
| F3 | O-EQ 不是上界，且不参与任何门控 | P0 | 见 §1.3 | §11.1 基线语义；报告可信度 |
| F4 | 动态非退化的 5σ 判据使用固定 1× 名义噪声 | P1 | 见 §1.4 | §10.5 门 1、`active_channel_fraction` |
| F5 | Jacobian 审计仅覆盖 stress_val、72 样本、堆叠 horizon | P1 | audit.py:1479-1484 | §10.7 口径 |
| F6 | §10.5 第 4、5 条判据未实现但文档写成已定义 | P1 | audit.py:762-874 | §10.5 完整性 |
| F7 | `a2_dynamic_audit.py` 无专用单元测试 | P1 | `tests/` 无 `test_a2_dynamic_audit.py` | 判决代码无回归保护 |
| F8 | 6 个模块全部超过 800 行上限 | P2 | 见 §1.5 | 编码规范 |
| D1 | §11.4 未定义退化比值的样本总体 | P0 | §13 第 916 行 | 是 F1 的规格层根因 |
| D2 | §10.7 口径大于实现 | P1 | §13 第 855–867 行 | 报告可信度 |
| D3 | §10.5 第 4、5 条写成已定义判据但未实现 | P1 | §13 第 833–834 行 | 报告可信度 |
| D4 | §11.1 对 O-EQ / O-KIN 的角色描述与实测不符 | P1 | §13 第 882–883 行 | 结论表述 |
| D5 | 单文件 1,382 行，plan 与 execution facts 混排 | P2 | §13 §0 / §18.2 / §19–§22 | 可维护性、评审成本 |
| D6 | §5.3 未记录 P150 与 recovery 边界重合的已知后果 | P2 | §13 第 404 行 | 设计可追溯性 |

### 1.1 F1：晚期参照的样本总体不一致

`_horizon_indices`（`src/gf/sim/a2_dynamic_audit.py:1056-1088`）把 `cutoff >= exposure_end_s` 的行置 `-1` 并在基线拟合前排除。P150 的 cutoff 为 `exposure_onset_s + 150 - dt = onset + 149.8 s`，标准协议下恰好压在 steady/recovery 边界（绝对 180 s）；phase duration jitter（train 0–5%、stress 8–20%、test 15–30%）一旦让 recovery 提前，该行的 P150 即失效。

val split 实际有效行数（本次统计）：

| family | P015 | P030 | P060 | P120 | P150 |
| --- | ---: | ---: | ---: | ---: | ---: |
| D-IID | 180 | 180 | 180 | 180 | **93** |
| D-KINETICS | 90 | 90 | 90 | 90 | **47** |
| D-PROTOCOL | 90 | 90 | 90 | 68 | **31** |
| D-NOISE-DRIFT | 270 | 270 | 270 | 270 | **141** |
| D-ENV-CAL | 90 | 90 | 90 | 90 | **46** |
| D-JOINT | 180 | 180 | 180 | 180 | **94** |

D-IID 只使用 `STEP_STANDARD`，同样掉到 93/180，说明这不是协议族特有问题而是边界设计问题。后果：

- 分母 `late_error` 在"暴露更长、更接近稳态"的子集上计算，系统性偏低，比值被放大；
- B-LAST 误差非单调——D-IID / D-KINETICS / D-PROTOCOL 的 P150 误差**高于** P120（D-IID：0.02205 vs 0.01812），D-IID 的 P060 相对退化为 **-0.078**；
- D-PROTOCOL 的 P150 仅 31 行，用作参照点的统计意义薄弱。

同一机制会影响 A2-DYN-5：`temporal_information` 门的 `late_static_reference=B-STEADY / late_horizon=P150 / max_late_relative_degradation=0.05` 也会落在这个半数子集上。

### 1.2 F2：O-KIN headroom 门恒真

`_kinetic_oracle_predictions`（audit.py:1186-1311）在 `clean_device_signals` 上反演，输入无观测噪声、无标定漂移、无量化，并直接读取特权 `privileged_parameters`。实测 val macro_RNMAE：

| family | O-KIN P015 | O-KIN P060 | B-LAST P060 | headroom |
| --- | ---: | ---: | ---: | ---: |
| D-IID | 2.77e-05 | 2.15e-05 | 0.02033 | 0.9989 |
| D-PROTOCOL | 5.30e-05 | 2.20e-05 | 0.02503 | 0.9991 |

六个 family、三个早期 horizon 共 18 个格点的 headroom 全部 ≥0.9989，阈值 0.20。这道门证明的是"前向模型可逆"，不是"可部署模型还有多少空间"。文档 §11.1 称 O-KIN 能"区分 nuisance 与不可辨识"——只做到了前半句：它无法区分"误差来自 nuisance"与"误差来自不可约观测噪声"。

### 1.3 F3：O-EQ 不构成上界

O-EQ 用 `Ridge(alpha=1e-8)` 拟合 3 维 clean 平衡特征（audit.py:1165-1185），B-LAST 用 `MLPRegressor(hidden_layer_sizes=(24,), tanh, lbfgs)` 拟合 3 维带噪端点。D-IID val：

| horizon | O-EQ | B-LAST |
| --- | ---: | ---: |
| P015 | 0.06327 | 0.05704 |
| P060 | 0.05280 | 0.02033 |
| P150 | 0.05947 | 0.02205 |

特权 oracle 在所有 horizon 上都劣于带噪基线，P060 差 2.6 倍。原因是模型类不对等（线性 vs 非线性），不是物理不可辨识。当前 O-EQ 既不是上界，也不进入任何门控条件，是死重。

### 1.4 F4：非退化判据的噪声基准

`_audit_dynamic_non_degenerate`（audit.py:786-815）读取 `experiment_config.pilot.observation_noise_std_by_sensor = [1e-6, 2e-3, 2e-3]`，对所有行统一用 `p2p > 5 × σ_nominal` 判定通道有效，从不乘该行 `noise_profile` 的 `white_noise_scale`（NOISE-1X/2X/5X/10X）。因此 NOISE-10X 行的实际判据比 NOISE-1X 松 10 倍，报出的 `active_channel_fraction = 1.000` 对高噪声族偏乐观。

附带观察：t50 配对率从 pilot 的 0.9875 掉到开发数据的 0.8335（test 0.9004），门限 0.70。通过没有问题，但说明 240 组 pilot 对该量的估计偏差约 0.15，后续不应把 pilot 分位数当作全量预期。

### 1.5 F8：文件规模

| 文件 | 行数 |
| --- | ---: |
| `src/gf/sim/a2_dynamic_dataset.py` | 2,015 |
| `src/gf/sim/a2_dynamic_audit.py` | 1,772 |
| `src/gf/pipeline/a2_dynamic_protocol.py` | 1,492 |
| `src/gf/sim/a2_sensor_devices.py` | 1,214 |
| `src/gf/pipeline/a2_dynamic_pilot.py` | 1,148 |
| `src/gf/sim/a2_dynamic_physics.py` | 1,098 |

全局编码规范禁止单文件超过 800 行。`a2_dynamic_audit.py` 内含 schema / physics / dynamic / baselines / jacobian / freeze 六类互不依赖的审计。

### 1.6 D1–D6：规划文档 §13 自身的问题

§13 整体质量高——终态可证伪、物理分层不含糊、来源等级制度完整、失败路径（§19.1 两次维里声速）保留而非删除、规模与配额数字自洽。下列问题不影响它作为主事实源的地位，但需要在本轮一并整改。

**D1（P0）——§11.4 门 1 的规格不完整。** 原文只写"`B-LAST` 在 P015、P030、P060 中至少两个 horizon 相对 P150 退化 ≥25%"，没有规定分子与分母在哪些行上计算。F1 是这条规格缺口在实现层的直接后果：实现选择了"各 horizon 各自的有效行"，在 P150 被 recovery 边界截断后就产生了总体不一致。**只修 F1 不修 D1，下一个实现者会重新踩同一个洞。**

**D2（P1）——§10.7 承诺的口径大于实现。** 文档写"各 family、split 和 horizon 的 rank fraction"，实现只做 stress_val 单 split、72 个采样行、P015/P030/P060 堆叠成一个矩阵（见 F5）。§21 因此把"满秩率 1.000"写成了看似全量的结论。同条第 5 项"pure 和结构零边界单列"在开发数据阶段无对应实现——pure 顶点只存在于 test，A2-DYN-4 的 pure 边界审计也不含 Jacobian。

**D3（P1）——§10.5 第 4、5 条是纸面判据。** 第 4 条（四阶段均有非空点）与第 5 条（clean signal 的 transition 方差不能全部由白噪声解释）在 `_audit_dynamic_non_degenerate` 中没有任何对应代码，但文档把六条并列写成同一套审计的组成部分，§21 的 `dynamic_non_degenerate: PASS` 会被读成六条全过。

**D4（P1）——§11.1 的 oracle 角色描述与实测不符。** 表中 O-EQ 标为"目标可达上界"，实测在所有 horizon 上劣于 B-LAST（F3）；O-KIN 标为"区分 nuisance 与不可辨识"，实际只能证明前向模型可逆（F2）。这两个描述如果原样进论文，属于对证据强度的高估。

**D5（P2）——单文件 1,382 行，plan 与 execution facts 混排。** 这是"过长"和"混排"两个问题叠在一起，拆开看：

*长度本身*。22 个一级节，最大的几节各自已经是独立文档的体量：

| 节 | 行数 | 性质 |
| --- | ---: | --- |
| §4 动态物理链 | 167 | 物理规格 |
| §12 A2-DYN 分步执行 | 164 | 执行计划 |
| §6 动力学与扰动分布 | 132 | 数据规格 |
| §8 数据契约与存储 | 97 | 接口契约 |
| §10 质量与审计 | 93 | 验收规格 |
| §19–§22 执行事实 | 91 | 已发生的结果 |
| §11 基线与资格门 | 77 | 验收规格 |

对照仓库既有做法，A1 已经拆成"04 分步执行计划 + 05 数据与物理规格"两份，A2 拆成"07 分步执行 + 07 评审记录"。§13 把 A1 的两份、A2 的两份、外加执行记录全部塞进一个文件，是这条线上唯一的例外。后果是实际使用成本高：改一次状态要翻 1,382 行找齐 3–4 处；评审时无法只审物理规格或只审执行结果；本次复核中定位 §11.4 与其实现的对应关系，需要在 §5.3、§6.1.1、§10.5、§11.4 四处之间来回跳转才能拼出完整判据（D1、D6 两个缺陷能长期存在，与这个可读性成本直接相关）。

*混排*。同一批 pilot 数字（`5 Hz / 240 s`、`US-CHIRP-XCORR-PARABOLIC-1`、`44.99%`）在 §0 摘要、§12.3 阶段块、§20 执行事实中出现三次；A2-DYN-3/4 的状态同时写在文档状态行、§0、§18.2 和 §21/§22。计划（应当稳定）与结果（每个阶段都变）耦合在一起，导致每次执行都要改动计划部分的文字。

*不做过度拆分*。§13 的论证链条（现状 → 为什么不能扩大 A0 → 物理链 → 分布 → 门 → 停止规则）是它质量高的原因，拆碎会破坏这条链。目标是拆成"一份可读的主规划 + 两份被引用的规格/记录"，不是拆成十份。

**D6（P2）——P150 与 recovery 边界重合未被记录为已知风险。** §5.2 把 steady 定为 90–180 s，§5.3 把 P150 定为 onset 后 150 s（标准 onset 下即 180 s），两者恰好重合；§6.1.1 又对 phase duration jitter 给了 0–30% 的范围。三处设计各自合理，交叉后的后果（约半数序列 P150 失效）没有在 §17 风险表中登记。§17 有"recovery 泄漏未来"一条，但讲的是相反方向的问题。

### 1.7 已确认没有问题的部分

复核中未发现下列风险，记录在案避免重复排查：

- 全模块只有一处 `np.clip`（`a2_sensor_devices.py:497`），前置有显式范围检查加 float32 eps 容差，属合法收边；无静默归一化、无吞错 `try/except`、无默认成功分支。
- 超声失锁 `raise UltrasonicLockError` 且不回退理论 ToF；O-KIN 在 CO₂ 越界、He 反演越过量化边界、组成不闭合时全部显式 `raise`。
- `B-LAST` 记录 solver / n_iter / max_iter / loss，`n_iter >= max_iter` 判 FAIL。
- 门控组合逻辑（audit.py:88-160）阈值全部来自 `eval_config`，`failed_requirements` 逐项累积。
- §6.5 配额表、§7.2 规模表、§5.3 窗口点数、独立统计单位（4,410 groups / 6,300 obs / train 3,600）互相自洽。
- `tests/test_a2_dynamic_{physics,dataset,protocol,benchmark,pilot}.py` 与 `tests/test_a2_sensor_devices.py` 共 43 项，2026-09-04 实跑 **43 passed in 27.9 s**。

## 2. 修复项

### F1 — 晚期参照改为配对总体（P0）

**根因**：门 1 的比值定义没有约束分子分母的行集合；P150 的 cutoff 落在 steady/recovery 边界上，被 jitter 系统性截断。

**修改**：

1. `src/gf/sim/a2_dynamic_audit.py::_audit_baselines`：`relative_degradation` 改为配对计算——仅在早期 horizon 与晚期参照 horizon **同时有效**的行子集上分别计算 B-LAST val macro_RNMAE 再取比值，并在 `difficulty[horizon]` 中新增 `paired_row_count`、`early_row_count`、`late_row_count` 三个字段。
2. 晚期参照主值改用 **P120**（除 D-PROTOCOL 外全族 100% 有效），同时保留 P150 作为副参照并两者都写入证据；`family_gate` 的判定只用主参照。
3. `configs/eval/a2_dynamic_eval.json`：`qualification_gates.dynamic_difficulty` 增加 `late_reference_horizon: "P120"`、`secondary_late_reference_horizon: "P150"`、`pairing: "valid_at_both_horizons"`，并在 `temporal_information` 中把 `late_horizon` 的语义标注为同样使用配对总体。
4. 若某 family 在主参照上配对行数低于 60，审计显式 `FAIL` 而不是照常出数。

**不做**：不放宽 `min_relative_degradation=0.25`；不删除 P150 报告。

**验证**：新增测试用合成 records 构造"一半行 P150 无效"的场景，断言配对比值与非配对比值不同且配对值可复算；重跑 `--stage audit`，比对新旧 `relative_degradation` 并在 §21 更新数值。

**预期影响**：§13 §21 表中的退化数字会变化（方向为变小）；`DIFFICULTY_QUALIFIED` 是否保持需以实际重跑为准，不预判。

### F2 — 增加噪声受限 oracle O-KIN-OBS（P0）

**根因**：唯一的 kinetics oracle 建立在无噪 clean 信号上，headroom 定义因此退化为常数。

**修改**：

1. 新增 `O-KIN-OBS`：与 O-KIN 使用完全相同的反演算子和特权动力学参数，但输入换成 `dataset.signals`（含 gain/offset/漂移/AR(1)/白噪/量化的最终观测）。反演失败（越界、不可辨识）按行显式记录并计入 `inversion_failure_fraction`，不静默丢弃。
2. `min_oracle_headroom_vs_last` 的判定改用 `O-KIN-OBS`；`O-KIN` 保留为 `identifiability_upper_bound`，只报告不判定。
3. `configs/eval/a2_dynamic_eval.json` 的 `baseline_registry` 增加 `O-KIN-OBS` 条目，`reference_role: "noise_limited_headroom_audit"`；`qualification_gates.dynamic_difficulty.oracle_model` 与 `new_algorithm_headroom.min_oracle_headroom` 同步指向它。

**验证**：新增测试断言在 NOISE-1X 与 NOISE-10X 两组行上 `O-KIN-OBS` 的误差有可分辨差异（否则说明输入接错）；断言 `O-KIN-OBS` 误差严格大于 `O-KIN`。

**风险**：`O-KIN-OBS` 可能在高噪声族上大量反演失败。这是有意义的负结果——若失败率过高，说明该族的 nuisance 已经压过可辨识性，应按 §2.4 走 `DYNAMIC_UNIDENTIFIABLE` 判定流程，不得靠放宽反演容差救回。

### F3 — 修正 O-EQ 的模型类与语义（P0）

**根因**：oracle 与被比较基线的模型类不对等，"上界"由更弱的假设空间实现。

**修改**（二选一，推荐方案 A）：

- **方案 A**：O-EQ 改用与 B-LAST 完全相同的 `_fit_small_mlp`（同结构、同 `random_state`、同标准化流程），只有输入特征不同（clean 平衡信号 vs 带噪端点）。这样 O-EQ 与 B-LAST 的差就是"观测扰动的代价"，语义干净。
- **方案 B**：保留 Ridge，但把 `reference_role` 从 `equilibrium_upper_bound` 改为 `linear_equilibrium_reference`，并在文档 §11.1 同步降级表述。

采用 A 后仍需断言 `O-EQ ≤ B-LAST`；若断言失败，说明 clean 平衡信号确实不足以确定目标，属于需要单独记录的物理发现，此时转方案 B 并在报告中写明原因。

**验证**：新增测试断言 O-EQ 与 B-LAST 使用同一 `_fit_small_mlp` 配置；重跑后比对二者关系。

### F4 — 非退化判据按行缩放噪声（P1）

**修改**：`_audit_dynamic_non_degenerate` 读取每行 `noise_profile_id` 对应的 `white_noise_scale`，判据改为 `p2p > 5 × σ_nominal × white_noise_scale`。产物中同时报告缩放前后的 `active_channel_fraction`，便于与历史数值对照。

**验证**：新增测试构造 NOISE-10X 行，断言缩放后判据比缩放前严格；重跑 `--stage audit` 并更新 §21、§22 中的 `active_channel_fraction`。

**注意**：`maximum_family_degenerate_fraction=0.05` 不放宽。若缩放后某族退化率超标，该族按 §12.4 退出正式训练，不调阈值。

### F5 — Jacobian 审计扩面（P1）

**修改**：

1. 采样从 stress_val 单 split 扩到 `train / val / stress_val` 三个 split，每 family 每 split 至少 12 行（共 ≥216 样本），仍用确定性等距抽样并记录 `row` 列表。
2. 除堆叠矩阵外，追加逐 horizon（P015 / P030 / P060 单独）的 rank 与条件数，按 family × split × horizon 组织。
3. 产物中显式记录采样策略与总体规模（`sampled_rows / total_rows`），避免"满秩率 1.000"被读成全量结论。

**不做**：不做全量 5,400 行的有限差分 Jacobian（成本不合理）；口径以采样声明为准。

### F6 — 补齐 §10.5 第 4、5 条（P1）

**修改**：

1. 第 4 条：审计每条序列的 `phase_id` 在 baseline / transition / steady / recovery 四段均非空，任一为空则该行判退化。
2. 第 5 条：对 clean signal 的 transition 段方差与该行噪声 profile 下的白噪方差做比值检验，报告 `transition_variance_ratio` 的分位数并设最小门限（建议初值 4.0，写入 `pilot.dynamic_gate`）。
3. 若判定第 5 条在 v1 不实现，则在 §13 §10.5 明确标注 `NOT_IMPLEMENTED` 及原因——不允许文档保留未实现的判据描述。

### F7 — 补 `tests/test_a2_dynamic_audit.py`（P1）

覆盖至少：门控组合逻辑（各失败分支能正确进入 `failed_requirements`）、`_horizon_indices` 的 exposure_end 失效规则、F1 的配对比值、O-KIN / O-KIN-OBS 反演的显式失败路径、Jacobian 投影的秩计算、`eligible_dynamic_axes` 的推导。使用小规模合成 `DynamicDataset`，单测超时 60 s 内。

### F8 — 拆分超长模块（P2）

`a2_dynamic_audit.py` 按审计类别拆为 `audit/schema.py`、`audit/physics.py`、`audit/dynamic.py`、`audit/baselines.py`、`audit/jacobian.py`、`audit/freeze.py`，顶层 `audit/__init__.py` 保留 `run_a2_dynamic_difficulty_audit` / `run_a2_dynamic_freeze_audit` 的公开签名与 `__all__`，调用方不变。其余五个文件在本轮只登记不动，列入后续技术债。

**约束**：纯结构调整，禁止夹带数值行为变更；拆分前后数据 `content_sha256`、阶段状态、失败项、family 资格和数值证据必须一致。`audit_sha256` 包含 freshness 与依赖身份，源文件布局变化后必须重算并登记新值，不能把其变化误写成数值行为变化。

### D1 — 补全 §11.4 门 1 的规格（P0，与 F1 同批）

**改 §13 §11.4 第 1 条**，把判据写全：

> `B-LAST` 在 P015、P030、P060 中至少两个 horizon 相对晚期参照退化 ≥25%。晚期参照主值为 P120，副值为 P150；比值的分子与分母**必须在同时于早期 horizon 与晚期参照 horizon 有效的同一批行**上计算（`horizon_valid` 同时为真）。任一 family 的配对行数低于 60 时，该 family 判 `FAILED` 而不是照常出数。P150 因与 recovery 边界重合会损失约半数序列，只作副值报告，不参与判定。

同批在 §11.5 的 late 门补同一句配对约束（`B-STEADY` / P150 同样受影响）。**D1 的文字与 F1 的实现必须在同一次提交中落地**，否则规格与实现再次分叉。

### D2 — §10.7 口径对齐实现（P1，与 F5 同批）

按 F5 完成后的实际能力改写：写明采样策略（确定性等距）、覆盖的 split（train / val / stress_val）、每 family 每 split 的采样行数与总样本数、逐 horizon 与堆叠两种矩阵的分别口径，并显式声明**这是采样结论而非全量结论**。§10.7 第 5 项"pure 和结构零边界单列"改为指向 A2-DYN-4 的 pure 边界审计，或标注 Jacobian 层不覆盖 pure。§21 的"满秩率 1.000"同步补上样本规模。

### D3 — §10.5 第 4、5 条给出确定归属（P1，与 F6 同批）

两条路径二选一，不允许维持现状：

- 按 F6 实现，§10.5 保持原文并在 §21 增加 `phase_coverage`、`transition_variance_ratio` 两项证据；
- 判定 v1 不实现，则在 §10.5 该两条后直接标注 `NOT_IMPLEMENTED（v1）` 与原因，并在 §21 的 `dynamic_non_degenerate` 结论旁注明"覆盖第 1、2、3、6 条"。

### D4 — §11.1 oracle 角色重写（P1，与 F2/F3 同批）

按 F2/F3 的实际结果改写 §11.1 表：

| ID | 改后描述 |
| --- | --- |
| `O-EQ` | 方案 A 下为"同模型类、clean 平衡输入的上界"；方案 B 下降级为"线性平衡参照，不构成上界" |
| `O-KIN` | 改为"前向模型可逆性上界（无噪）"，明确它不衡量可部署 headroom |
| `O-KIN-OBS` | 新增，"噪声受限 headroom 参照"，§11.4 门 2 与 §11.6 均以它为准 |

同时修订 §11.6 的四条 headroom 判据，把 `O-KIN` 替换为 `O-KIN-OBS`，否则 A2-DYN-5 的"是否值得设计新算法"仍然无法回答。

### D5 — 按职责拆分 §13（P2）

**目标结构**：一份主规划 + 两份被引用文件，对齐 A1（04 计划 / 05 规格）的既有做法。

| 文件 | 承接内容 | 预计行数 |
| --- | --- | ---: |
| `13_Ar-He-CO2动态时间序列仿真与数据分布规划.md`（主规划） | §0–§3、§5、§7、§9、§11、§12、§14、§16、§17、§18 | 约 620 |
| `13b_A2-DYN物理与审计规格.md`（新增） | §4 动态物理链、§6 动力学与扰动分布、§8 数据契约与存储、§10 质量与审计、§13 影响文件、§15 验证矩阵 | 约 670 |
| `13a_A2-DYN执行记录.md`（新增） | §19–§22 执行事实 | 约 95 |

划分依据：主规划回答"要做什么、为什么、什么条件下停"，读一遍能拿到完整论证链；13b 是被主规划和代码同时引用的稳定规格，改动频率低；13a 是每个阶段都会追加的结果，改动频率高。三者的改动节奏不同，是拆分的实际理由。

**保留的整体性**：§0 摘要、§1 Context、§2 Task、§12 分步执行必须留在主规划且顺序不变——§13 的论证链条是它质量高的原因，不能为了均分行数打散。主规划中被移走的每一节保留一行指针（`§4 动态物理链 → 13b §1`）。

**迁移约束**：

1. 纯搬运，逐字保留，不改写、不补充、不删减；数值更新在 S7 的 F9 步统一做。
2. 交叉引用全部改为跨文件链接并逐条点开验证，不留悬空锚点。
3. 三份文件的文档状态行统一由主规划持有，13a / 13b 只写 `从属于 13`，避免状态再次分裂。
4. 迁移完成后，同一数字在三份文件中只出现一次；`5 Hz / 240 s` 这类冻结选择由 13a 持有，主规划引用而不复述。
5. §18.2 的实现完成清单留在主规划（属于计划），其中的具体数值改为指向 13a 的链接。
6. 外部引用同步：`general_fusion/项目总体规划.md`、根 `CLAUDE.md` 的相关文档表、`14_传感器仿真文献调研与A2-DYN对比.md` 中指向 §13 具体小节的链接，按新归属改到 13a / 13b；指向文档整体的链接保持指向主规划。

**验收**：迁移前后三份文件合并的正文内容与原文逐字一致（可用 diff 校验）；主规划单独读完能回答"A2-DYN 要证明什么、通过条件是什么、当前到哪一步"。

### D6 — 登记 P150 边界风险（P2）

§17 风险表新增一行：

| 风险 | 识别信号 | 处置 |
| --- | --- | --- |
| 晚期参照 horizon 与 phase 边界重合 | 某 horizon 的 `horizon_valid` 行数显著低于早期 horizon；B-LAST 误差随 horizon 非单调 | 参照 horizon 必须落在 phase 内部而非边界；判据改配对总体；不通过挪动 phase 时长补救 |

§5.3 的窗口表在 P150 行补注"cutoff 与 steady/recovery 边界重合，受 jitter 影响约半数序列失效，仅作副参照"。

### F9 — §21/§22 数值更新（P2，S7 执行）

§21、§22 中受 F1（退化比值）、F4（`active_channel_fraction`）、F5（Jacobian 口径）影响的全部数值，按 S4 重跑结果逐项更新，并注明原数值因口径问题作废、原始产物路径保留在 `outputs/summary/a2_dynamic_v1/` 供对照。

## 3. 执行顺序与验收

| 步骤 | 内容 | 门 |
| --- | --- | --- |
| S1 | 实现 F1 + F2 + F3，同批完成 D1（§11.4/§11.5 规格）与 D4（§11.1 oracle 角色） | 新增单测通过；`python -m pytest -q` 全绿；§11.4 文字与实现的总体定义逐字一致 |
| S2 | 实现 F4 + F5 + F6，同批完成 D2（§10.7 口径）与 D3（§10.5 第 4/5 条归属） | 同上；§10.7 中的采样规模与产物 `sampled_rows` 一致 |
| S3 | 补 F7 | `test_a2_dynamic_audit.py` 覆盖 §2 中列出的六类路径 |
| S4 | 对已冻结数据重跑 `--stage audit` | `content_sha256` 必须仍为 `82837b52…`（开发子集）/ `3da0e478…`（完整包）；`audit_sha256` 变化并记录新值 |
| S5 | 判定终态 | 若仍 `DIFFICULTY_QUALIFIED` → 记 `A2-DYN-3R2`，解除 A2-DYN-5 开发侧实现阻塞；正式 test 仍要求 A2-DYN-4 `DATA_FROZEN`；若反转 → 按 §2.4 走对应失败终态，**不修改分布或阈值补救** |
| S6 | F8 结构拆分 | 数据 `content_sha256`、阶段状态、失败项、family 资格和数值证据一致；包含依赖身份的 `audit_sha256` 重算并登记 |
| S7 | D5 按职责拆分 §13（主规划 / 13b 规格 / 13a 记录）+ D6 风险登记 + F9 数值更新 | 三份文件合并后与原文逐字一致；交叉引用无悬空锚点；§21/§22 数值与 S4 产物逐项对齐；同一数字在三份文件中只出现一次 |

**文档与实现的同批约束**：S1、S2 的每次提交必须同时包含代码改动与对应的 §13 条文改动。审查时以"条文写的总体 = 代码算的总体"为通过条件；只改一侧的提交不予合入。

最小验证命令：

```powershell
python -m pytest -q tests/test_a2_dynamic_audit.py
python -m pytest -q tests/test_a2_dynamic_physics.py tests/test_a2_sensor_devices.py
python -m pytest -q tests/test_a2_dynamic_dataset.py tests/test_a2_dynamic_protocol.py tests/test_a2_dynamic_benchmark.py
python -m pytest -q
python -m gf.pipeline.a2_dynamic_benchmark --stage audit --project-root .
git diff --check
git status --short
```

## 4. 版本与不变量影响

| 项目 | 是否变化 | 说明 |
| --- | --- | --- |
| `data/a2_dynamic_v1/` 数组与 records | 否 | 本轮不重新生成数据 |
| 数据 `content_sha256` | 否 | S4 必须复算并确认不变 |
| `audit.json` / `audit_sha256` | 是 | 口径变更，新旧值都需在执行记录中留存 |
| `configs/eval/a2_dynamic_eval.json` | 是 | F1/F2/F3 的 gate 与 registry 变更，需升 eval revision |
| `configs/data/ar_he_co2_a2_dynamic_v1.json` | 否 | 分布与 profile 不动 |
| `configs/experiment/a2_dynamic_protocol.json` | 是（仅 F6） | 新增 `transition_variance_ratio` 门限 |
| A2-DYN-3 终态 | 待定 | 以 S4 重跑结果为准 |
| A2-DYN-4 `DATA_FROZEN` | 否 | 数据冻结不受审计口径变更影响 |
| A1 / A2H / A2M 历史产物 | 否 | 只读不变量维持 |
| `docs/algorithm/13_…md` | 是 | D1–D6 的条文修订；D5 后拆为主规划（约 620 行） |
| `docs/algorithm/13a_A2-DYN执行记录.md` | 新增 | D5 迁出的执行事实（§19–§22） |
| `docs/algorithm/13b_A2-DYN物理与审计规格.md` | 新增 | D5 迁出的物理链、分布、存储契约、审计与验证矩阵 |

**杀停规则**：S5 若判定反转，禁止通过调整 `min_relative_degradation`、更换晚期参照、缩小 family 或增加 seed 来翻回 `DIFFICULTY_QUALIFIED`。按 §17 的预注册处置，该情形对应 `TEMPORAL_REDUNDANT` 或 `INVALID_PROTOCOL`，需要建立 v2 数据规划而非覆盖 v1。

## 5. 未验证项声明

- 本规划中所有缺陷判断来自代码阅读、审计产物 JSON 数值和一次 records 统计（§1.1 表格为实际统计结果），**未重跑难度审计本身**（需要完整 5,400 条开发数据，耗时较长）。
- F1 修复后退化数值的具体变化量、`DIFFICULTY_QUALIFIED` 是否保持，均未预测。
- F2 引入的 `O-KIN-OBS` 在高噪声族的反演成功率未知。
- F3 方案 A 下 `O-EQ ≤ B-LAST` 是否成立未知。
- 43 项现有单元测试通过为 2026-09-04 实跑结果；本规划尚未产生任何代码改动。
- D1–D6 为文本层判断，依据是 §13 的条文与实现代码的逐条比对（引用行号基于 2026-09-04 的 §13 版本，共 1,382 行）；D5 的分节行数由一级标题行号差计算得出，拆分后的预计行数为估算值，"同一数字出现三次"为人工计数，未做脚本校验。

---

## 6. 执行进度（2026-09-04 第一轮会话交接记录）

> 本节由执行 AI 写入，供后续会话/换模型续做使用。状态：**S1、S2 已完成（代码+条文+真实数据冒烟验证），S3 进行中，S4–S7 未开始**。

### 6.1 已完成的代码改动（全部未提交 git）

| 文件 | 改动内容 |
| --- | --- |
| `src/gf/sim/a2_sensor_devices.py` | `estimate_ndir_equilibrium_co2_series` 新增 `domain_tolerance` 参数（默认 None=原 float32 预算，行为不变；观测反演显式传预算）。clean 路径行为与原实现逐位一致。 |
| `src/gf/sim/a2_dynamic_audit.py` | ① `AUDIT_SCHEMA_VERSION` 升 `gf-a2-dynamic-audit-2`；② F1：`_audit_baselines` 重写，`relative_degradation` 改配对总体（主参照 P120、副参照 P150，新增 `_paired_late_reference_evidence`，`difficulty[h]` 含 `paired_row_count/early_row_count/late_row_count` 与两套参照证据）；③ F2：`_kinetic_oracle_predictions` 新增 `input_mode="observed"`（O-KIN-OBS），失败按行记 `inversion_failures` 并 NaN 占位；headroom 用 O-KIN-OBS 在反演成功的同一批 val 行上与 B-LAST 配对计算；新增 `_observed_admission_budgets`（按行注册 noise/calibration 包络，常量 `OBSERVED_ADMISSION_SIGMA_FACTOR=5.0`）；④ F3：删除 `_fit_ridge`，O-EQ 改用与 B-LAST 相同的 `_fit_small_mlp`，新增 `oeq_upper_bound_of_blast_holds` 字段；⑤ F4：`_audit_dynamic_non_degenerate` 判据按行 `white_noise_scale` 缩放，`active_channel_fraction_unscaled` 同时报告；⑥ F5：`_audit_jacobian` 重写（train/val/stress_val 三 split×6 family×12 行=216 样本，逐 horizon + 堆叠两种口径，`sampled_rows/total_rows` 声明，per-horizon 检查全部进 checks）；⑦ F6：第 4 条 phase 覆盖（`_phase_index`）、第 5 条 `_transition_variance_ratio`（**any-channel** 语义，见 6.3） |
| `src/gf/pipeline/a2_dynamic_protocol.py` | `EVAL_SCHEMA_VERSION` 升 `gf-a2-dynamic-eval-2`；eval 校验器：baseline_registry 含 O-KIN-OBS（role 必须 `noise_limited_headroom_audit`）、dynamic_difficulty gate 新冻结值（P120/副 P150/pairing/min_paired_rows=60/oracle_model=O-KIN-OBS）、temporal_information 加 `late_horizon_pairing`、new_algorithm_headroom 加 `oracle_model: O-KIN-OBS`；pilot 校验器 dynamic_gate 加 `minimum_transition_variance_ratio=4.0` 与 `minimum_transition_variance_ratio_pass_fraction=0.95` |
| `src/gf/pipeline/a2_dynamic_benchmark.py` | `run_a2_dynamic_difficulty_audit` 重写为 **A2-DYN-3R2 冻结重审计**：从完整包重建开发子集（`_development_subset_of_frozen_package`，manifest 取 `outputs/runs/a2_dynamic_v1/a2-dyn-4-test/development_subset_backup/manifest.json`，content hash 必须逐位复现 `82837b52…`）；同次运行完整包冻结审计 `A2-DYN-4R2`（**不 rebind source hash**，content hash 必须仍为 `3da0e478…`）；新增 `_frozen_reaudit_freshness`（数据配置 hash 必须不变；源文件只允许审计口径白名单 `_REAUDIT_ALLOWED_CHANGED_DEPENDENCIES` 内变化：eval/experiment 配置、benchmark/protocol/audit/sensor_devices 六个）；产物写 `outputs/summary/a2_dynamic_v1/a2_dyn_3r2_audit.json`、`a2_dyn_4r2_freeze_audit.json`、run 目录 `a2-dyn-3r2-reaudit/`，旧 `eligible_dynamic_axes.json` 归档至 `outputs/archive/a2_dynamic_v1/eligible_dynamic_axes_a2dyn3_r1.json`；主返回 status 取难度审计，freeze FAIL 时为 `DATA_FREEZE_FAILED` |
| `configs/eval/a2_dynamic_eval.json` | schema 升 `gf-a2-dynamic-eval-2`；O-EQ/O-KIN role 重写、新增 O-KIN-OBS 条目；dynamic_difficulty gate 新字段；temporal_information 加 `late_horizon_pairing`；new_algorithm_headroom 加 `oracle_model` |
| `configs/experiment/a2_dynamic_protocol.json` | dynamic_gate 加两个 transition variance 门限 |
| `docs/algorithm/13_…md` | D1（§11.4 全文重写为配对总体规格 + §11.5 late 门补配对约束）、D4（§11.1 O-EQ/O-KIN/O-KIN-OBS 角色重写 + §11.6 oracle 一律指 O-KIN-OBS）、D2（§10.7 采样口径）、D3（§10.5 第 1/4/5 条按实现改写）已落地 |

### 6.2 已验证（真实数据实跑）

- **开发子集重建**：从完整包前 5,400 行 + 备份 manifest 重建，`dynamic_content_sha256` 逐位复现 `82837b52b54f1d76…`（MATCH）。
- **`_audit_baselines` 全量冒烟**（约 186 s，5,400 行六 family）：所有 family fit=PASS/PASS；配对行数 P120 全部 ≥68（D-IID 180、D-PROTOCOL 68 等）、P150 副参照 31–141；配对 RD（P120 主参照）P015 全族 2.1–5.9、P030 0.78–3.1、P060 0.12–0.46；O-KIN-OBS val macro_RNMAE 0.0074–0.0162、反演失败率 0–6.7%（显式记录）、headroom 0.42–0.92（全部 ≥0.20 门）；O-EQ 在 17/18 格点 ≤ B-LAST（**唯一违反：D-IID P060 O-EQ=0.02180 vs B-LAST=0.02033**，已用字段记录，按规划 F3 属"需单独记录的物理发现"，保留方案 A 并报告，不回退 Ridge）。
- **F4/F6 在真实数据**：开发全 split PASS（scaled active 0.9886、tv ratio pass 0.9986、退化率 max 0.0317）；**test split FAIL**（详见 6.4）。
- 编译通过；新 eval/experiment 配置通过 `validate_a2_dynamic_eval_config` / `validate_a2_dynamic_experiment_config`。

### 6.3 实现中的两个关键语义决定（后续会话不得悄悄改回）

1. **§10.5 第 5 条采用 any-channel 语义**：逐通道算 transition 方差/白噪方差之比，**任一通道 ≥4 即该行通过**。原因（真实数据实测）：NDIR 比值 p05=0 的行是 binary 无 CO₂ 组成——CO₂ 恒为 0 时 NDIR 恒定是物理合法动态，all-channel 解读会把 20% binary 配额误判退化。§13 §10.5 条文已按此改写（"任一通道比值 ≥4 即通过该行——binary 无 CO₂ 等组成使单通道恒定属合法动态"）。
2. **O-KIN-OBS 准入预算**（`_observed_admission_budgets`）：NDIR 比值域预算 = |gain-1| + |offset|/BASELINE + p2p·drift% + 5σ·去卷积增益/2.5；ToF 域预算 = |gain-1|·max|tof| + |offset| + p2p·drift% + 5σ。去卷积增益 = sqrt(1+decay²)/(1-decay)，decay=exp(-0.2/0.75)。预算内越界投影回物理域端点（与 clean 口径 float32 预算同款处理），预算外按行记失败。这不是"放宽容差"——反演算子与 O-KIN 完全相同，预算只是判定观测是否仍在该行注册扰动包络内。

### 6.4 ⚠️ 重大发现（S5 判定时必须呈报）

**修正后的 F4 判据使 test split 动态非退化 FAIL**：scaled active=0.9329 < 0.95，D-JOINT 退化率 0.195、D-NOISE-DRIFT 0.111 > 0.05（未缩放口径为 1.000/0）。开发 split（train/val/stress_val 聚合）仍 PASS。含义：NOISE-10X 下约 7% test 行的双通道 peak-to-peak 低于其自身 5σ 噪声底——原 `active_channel_fraction=1.000` 是名义噪声门虚高。这与本规划 §4 "A2-DYN-4 DATA_FROZEN 不受审计口径变更影响"矛盾：若按 §12.4/§10.5 规则严格执行，test 侧动态非退化不过 → 完整包冻结审计（A2-DYN-4R2）将 FAIL（`run_a2_dynamic_freeze_audit` 的 dynamic 审计用 `subset_split="test"`）。**处置选择留给 S5**：按预注册规则走失败终态（`_finalize` 路径的 freeze FAIL），或升 revision 建 v2——**禁止调阈值补救**。注意：难度审计（A2-DYN-3R2）只看开发 split，仍应 QUALIFIED；已把 freeze FAIL 时主返回状态设为 `DATA_FREEZE_FAILED` 以避免静默。

### 6.5 未完成（按顺序）

- **S3（进行中）**：`tests/test_a2_dynamic_audit.py` 尚未创建。规划 §2 F7 要求覆盖六类路径。已确认的构造输入：合成 `DynamicDataset` 需满足 `__post_init__` 全部校验（signals (N,3,1200,1) float32、privileged (N,12) float64、device_audit 九键含特定 shape、inlet/chamber 逐时步闭包等）；`validate_a2_dynamic_records` 需 records 完整字段（namespace `a2dyn-obs-`/`a2dyn-mix-` 等，直接从真实 records 抽样改造比从零构造省力）；`_metrics` 依赖 `evaluate_predictions`（groups/indices 合法即可）。注意 `run_a2_dynamic_difficulty_audit` 现在**要求 a2h_config**（新增参数），单测调 sim 层函数时必须传。
- **S4**：`python -m gf.pipeline.a2_dynamic_benchmark --stage audit --project-root .`（约 3–5 min baselines + Jacobian 216 样本 + physics smoke）。预期开发子集 content=`82837b52…`、完整包 content=`3da0e478…` 不变；难度审计大概率 QUALIFIED（冒烟数据支持）；**freeze 审计预期 FAIL（见 6.4）**。
- **S5**：终态判定。按 6.4 的矛盾呈报；难度门若 QUALIFIED 记 `A2-DYN-3R2`。
- **S6**：F8 拆分（注意：文件现在约 2,100+ 行；拆分前重跑 audit 记录基线 hash，拆分后必须一致；`_development_subset_of_frozen_package` 与 `_frozen_reaudit_freshness` 在 benchmark.py 也要考虑归属）。
- **S7**：D5 拆 §13（主规划/13b 规格/13a 执行记录，纯搬运）+ D6（§17 风险表 + §5.3 P150 行补注）+ F9（§21/§22 数值按 S4 结果更新，注明旧口径数值作废）。**注意**：S4 后 §21/§22 要写 A2-DYN-3R2/4R2 两个新执行事实节，且 §0/文档状态行/§18.2 需同步新终态。
- 提交策略：建议按 S1（F1/F2/F3+D1/D4）、S2（F4/F5/F6+D2/D3）分两个 commit，S3 单独一个，S4/S5 一个（含产物与文档状态），S6、S7 各一个。当前全部改动未提交。

### 6.6 已知风险与注意事项

- `_audit_jacobian` 的 per-horizon 检查全部进入 `checks`（P015/P030/P060 各 4 项），若任一 horizon 条件数超标会整体 FAIL——这是口径严格化的预期行为，未跑过真实数据（S4 首次验证）。
- Jacobian 216 样本 × 每 `_jacobian_sample` 调 3 次 `_stacked_equilibrium_jacobians`（逐 horizon 各一次）+ 堆叠，计算量约为旧版（72 样本单次）的 9 倍；若 S4 耗时过长，先确认不是卡死。
- `data/a2_dynamic_v1/manifest.json` 与 `audit.json` 会在 S4 被 pipeline 重写（audit_status/audit_sha256/status 字段）；`content_sha256` 不在其中（阶段状态字段不属于内容身份，已由 `dynamic_content_sha256` 排除），数据数组不动。
- `run_a2_dynamic_freeze_audit` 签名未变；`_audit_dynamic_non_degenerate` 对 pure 行为不变（exclude_pure 路径）。
- 现有 43 项测试中 `test_a2_dynamic_physics_artifact_is_fresh_and_independently_gated` 会校验 `physics_audit_r4.json` 的 dependency_hashes——`a2_sensor_devices.py` 改动后该 hash 会变，physics-smoke 会重写产物，测试应仍过（它每次实跑重算）；但 **`test_a2_dynamic_pilot.py` 若有对 audit/eval schema 的断言需要先跑一遍确认**（S3 前先 `python -m pytest -q tests/` 看哪些旧测试因 schema 升版挂掉，逐个修）。

### 6.7 第二轮会话交接记录（2026-09-04）

状态：**S3、S4、S5 已完成（判定结论见下），S6、S7 未开始，全部改动仍未提交 git**。

**S3 已完成**：`tests/test_a2_dynamic_audit.py` 落地（13 项测试，13.3–17.4 s，其中 1 项在无 data 包环境 skip；含纯函数组与冻结包依赖组）。覆盖 F7 六类路径：门控组合各失败分支进 `failed_requirements`、`_horizon_indices` exposure_end 失效规则、`_paired_late_reference_evidence` 交集配对与 naive 口径区分、O-KIN-OBS 显式失败（`ndir_ratio_domain` / `tof_quantization_boundary` stage 记录 + NaN 占位）与 clean 模式 raise、O-KIN-OBS 噪声敏感（D-NOISE-DRIFT 1X vs 5X）与误差 > O-KIN、`_condition_number` / `_per_horizon_jacobian_summary`、`_audit_jacobian` 采样声明口径、`_audit_dynamic_non_degenerate` 缩放判据与 dev PASS / test FAIL 真实回归。

**S3 期间修复了 F5 重写遗留的三个真实缺陷**（均属"从未被真实数据执行过"的路径，S4 预演暴露）：

1. **`_stacked_equilibrium_jacobians` 逐 horizon 调用断裂**：`_jacobian_sample` 传单 endpoint，但函数强制 `len(endpoints)==3` → 必抛。修复：提出 `_equilibrium_jacobian_block`（单 endpoint），`_stacked_equilibrium_jacobians` 由 block 堆叠，`_jacobian_sample` 逐 horizon 用 block。
2. **per-horizon joint_target 语义不适定**：单 horizon 3 输出 × 4 nuisance 列，行满秩时 nuisance 吸收整个输出空间 → 投影残差≈0 → 条件数全 inf（216/216 样本 NaN），joint_target per-horizon 恒 FAIL。修复：joint nuisance 投影只保留堆叠口径；per-horizon 只报 fixed（目标切向）rank/条件数（P015/P030/P060 p95 条件数 63–114，全过）；checks 同步调整。
3. **checks key 错位**：`_audit_jacobian` 读 `summary["fixed_p95_condition_number"]`，summary 实际输出 `fixed_condition_number_p95` → KeyError。修复 key 对齐。
4. （附带）`_kinetic_oracle_predictions` zero-He 分支失败消息引用未定义 `boundary_tolerance_s`（应 `zero_he_tolerance_s`）→ NameError 隐患。已修。该分支在冻结开发数据不可达（纯 CO₂ 只在 test pure，不进 O-KIN 路径），属防御性修正，无行为测试覆盖。

**S4 已完成**（`python -m gf.pipeline.a2_dynamic_benchmark --stage audit --project-root .`，约 4 分钟）：

- 开发子集 content 逐位复现 `82837b52b54f1d76…`（MATCH）；完整包 `3da0e478eca52bb6…` 不变。
- **难度审计 A2-DYN-3R2 = `DIFFICULTY_QUALIFIED`**：6 family 全过；配对行数（P120 主参照）68–270 全 ≥60；P015 配对比值 2.15–5.85（远超 0.25 门）、P030 0.78–3.10、P060 0.12–0.46（部分族 <0.25，无碍——每族 P015+P030 双 horizon 过门）；O-KIN-OBS val headroom 0.42–0.92（全 ≥0.20 门），反演失败率 0–6.7%（显式按行记录）；O-EQ ≤ B-LAST 在 17/18 格点成立，唯一违反仍为 D-IID P060（0.02180 vs 0.02033，已按 F3 记为需单独记录的物理发现，保留方案 A）。Jacobian 216 样本全 PASS（fixed/joint 秩 1.0，条件数全过）。dynamic 非退化 dev 全 PASS（scaled active 0.9981/退化率 max 0.0037/tv ratio pass 0.9986）。audit_sha256 难度 = `85f305f575fcf833b5060b47e6c6b221adf58e51d2516ad5c094cec9a5943a1a`。
- **冻结审计 A2-DYN-4R2 = `DATA_FREEZE_FAILED`**（仅 `test_dynamic_non_degenerate` 失败）：test（897 非 pure 行）scaled active 0.9329 < 0.95、D-JOINT 退化率 0.1954、D-NOISE-DRIFT 0.1111 > 0.05（unscaled 口径 1.000/0——旧门虚高确认）。根因：NOISE-10X 只分布于 test（D-NOISE-DRIFT 270 行 + D-JOINT 180 行），10 倍噪声档在 R2 判据下 11–20% 行双通道 p2p < 50σ（可辨识线以下）。audit_sha256 冻结 = `75aab988848fdc2c85bdc97bba90372e8bf356a48dcbcfaf961cd3f214c42d36`。

**S5 判定已完成（2026-09-04；契约复核后收紧 test 边界）**：

1. 难度终态记 **A2-DYN-3R2 = `DIFFICULTY_QUALIFIED`**，A2-DYN-5 仅开发侧实现解除阻塞（§11.4 门在 R2 口径下有效）。
2. A2-DYN-4 冻结状态**降级为 `DATA_FREEZE_FAILED`**（R2 口径），缺陷登记在案：test NOISE-10X 两族可辨识行超标；v1 数据保留不重新生成；v1 test 只保留冻结审计证据，不进入 A2-DYN-5 模型评价；v2 数据规划时调整 10X 档并重新冻结。**禁止调阈值补救**。
3. data/a2_dynamic_v1/manifest.json 的 audit_status/status 已由 pipeline 重写为 `DATA_FREEZE_FAILED`（content hash 不变）。

**S6（未开始）注意事项更新**：
- 拆分对象行数已增至约 2,100+（F5 修复后）。S4 的难度 `85f305f5…`、冻结 `75aab988…` 是拆分前审计身份；拆分后对同一数据重跑 `--stage audit`，数据 hash、状态和数值证据必须一致，包含依赖身份的 audit hash 重新登记。
- 我新增了 `_equilibrium_jacobian_block`（拆到 jacobian 组）、常量 OBSERVED_ADMISSION_*（baselines 组）；`_horizon_indices`/`_dataset_arrays`/`_metrics`/`_condition_number` 是跨类共享（建议 `_core.py` 或各自归属组内被引用组 import）。
- `tests/test_a2_dynamic_audit.py` 从 `gf.sim.a2_dynamic_audit` 导入私有函数——拆分后凡以 `gf.sim.a2_dynamic_audit.X` 引用者不受影响（包名不变），但 `from gf.sim.a2_dynamic_audit import _kinetic_oracle_predictions` 这类跨子模块导入必须经 `__init__.py` 重导出或改导入路径——注意两种 import 语法在拆分后的差异。
- 现有 165 项测试全绿（2026-09-04 实跑，58.6 s）。

### 6.8 第三段交接记录（2026-09-04，S6/S7 完成）

- **S6（F8）已完成**：`src/gf/sim/a2_dynamic_audit.py`（2,433 行）拆为包 `src/gf/sim/a2_dynamic_audit/`；后续复核又将 HEOS 单纯形插值从 `_baselines.py` 提取到 `_heos_interpolation.py`，所有子模块均低于 800 行。`__init__` 重导出全部私有符号保持 import 兼容；`a2_dynamic_benchmark.py` 的依赖 hash 枚举与重审计白名单同步覆盖全部子模块。行为一致性以 §7 终检为准。
- **S7 已完成**：D5 按职责拆分 §13 为主规划、13a 执行记录和 13b 稳定规格；D6 风险登记与 F9 数值更新已同步到主规划和执行记录，外部引用同步至根 CLAUDE.md 与 general_fusion README。终检见 §7。

## 7. 最终验证

- `python -m gf.pipeline.a2_dynamic_benchmark --stage protocol --project-root .`：`PASS`；机器配置继续只允许 train、val、stress_val，正式 test 仍要求 `DATA_FROZEN`。
- `python -m gf.pipeline.a2_dynamic_benchmark --stage audit --project-root .`：开发 content hash `82837b52…`、完整包 content hash `3da0e478…` 不变；难度状态仍为 `DIFFICULTY_QUALIFIED`，冻结状态仍为 `DATA_FREEZE_FAILED`，唯一失败项仍为 `test_dynamic_non_degenerate`。依赖身份更新后的难度 audit hash 为 `a8ac1cf624cd5cbf235e03961eefbf269084d8954469d31d5d05edfc54691576`，冻结 audit hash 为 `cfcd8d6bb210dab9b079ad9a50b7a279e473f76bd55dcd0a4e23d931b05a97e8`。
- `python -m pytest -q`：166 项通过；新增回归测试拒绝以 `DATA_FREEZE_FAILED` 解锁 test。
- 审计包最大子模块为 `_baselines.py` 790 行，其余子模块均低于 800 行；`_registered_heos_interpolated_tof` 仍经包级入口重导出。

### 7.1 后续 A2-DYN-5/6 执行指针（2026-09-04）

本规划的 S1–S7 修复完成后，A2-DYN-5 开发侧基线、因果回放和 A2-DYN-6 handoff/关闭阶段已继续执行；最新数值、输入哈希、预测键审计、回放延迟与冻结阻断状态统一记录于 [13a §25](../13a_A2-DYN执行记录.md)。当前全量回归为 `172 passed`；A2-DYN-6 状态为 `A2_DYN_6_BLOCKED_DATA_FREEZE_FAILED`，不生成 `dynamic_handoff.json`。
