# MEI-3 B4 代码复核与结果再分析报告

> 审计日期：2026-07-29  
> 性质：对已冻结 B4 正式配对比较的只读复核——代码 review 加上从 freeze 产物独立复算的结果再分析。不修改任何 freeze，不改变 `mei3_full_parameter_baseline_retained` 裁决，不构成新的正式证据。  
> 审计对象 freeze：`outputs/runs/tv3_mrs_ei/mei3_varpro_audit/freezes/20260729T120958962354Z_cf7ed57312d9`；manifest SHA256=`604a5fe6a26c51963b8b5197748002b77ad2177461ff11c3bc5e7cd174f747d8`（本次已用 `verify_evidence_manifest` 复核，无 issue）。  
> 权威数字以 freeze 为准；本文表格是引用与复算，如有出入以 freeze 为准。  
> 代码版本：工作区四个核心源文件（`mrs_ei_b4_formal.py` / `mrs_varpro.py` / `run_tv3_mei3_b4_formal_comparison.py` / `mrs_observation.py`）与 freeze 内 `source_snapshots/` 逐字节一致（`cmp` 已核对），本文行号同时适用于两者。  
> 结论范围：`registered_simulation_domain_only`。  
> 上位计划：[MEI-3 执行计划](tv3_mrs_ei_mei3_execution_plan.md) §15。

## 0. 三条主结论

1. **裁决 `mei3_full_parameter_baseline_retained` 成立，且比 §15 文档呈现的更稳固。** 主指标、bootstrap、非退化门与冻结产物逐项一致；换 CI 方法（basic / BCa）只会对 S2 更不利；独立种子复算 bootstrap 与冻结结果行为一致。
2. **机制归因（本报告新增）：S2 的全部 P90 改善来自 test 89 个、OOD 83 个双方共同 `max_iterations` 的样本。** 在两法都收敛的约 86% 样本上 S1 与 S2 逐样本相同（成功配对最大 |ΔO₂ 误差| < 1e-5 pp）。S2 不是统计效率更高的估计器，而是同一估计器在固定迭代预算内进展更快的优化器；点改善 2.5–2.8% 是 `max_iterations=100` 这一冻结值的函数。
3. **代码 review 发现 1 个影响 §6.3 效率诊断公平性的计数问题**（S2 真实前向调用被系统性少计约 33%，真实效率比约 1.5× 而非报告的 2.1×），另有若干不影响主指标与裁决的实现 / 协议问题，见 §7–§8。

---

## 1. 审计范围与方法

- 数据来源：freeze 内 `solver_comparison.csv`（1296 mixture × 3 方法）、`crb_efficiency.csv`、`bootstrap_report.json`、`convergence_report.json`、`calibration_report.json`、`mei3_verdict.json`、`registered_observations.json`（提取每 mixture 的 T/P/RH/L）。
- 复算工具：一次性 Python 脚本（numpy/scipy），写在系统临时目录，未入库；所有统计可从上述 CSV/JSON 重新推导。
- 代码 review 覆盖：`tv3/audit/mrs_ei_b4_formal.py`、`scripts/run_tv3_mei3_b4_formal_comparison.py`、`tests/test_tunnel_ventilation_mei3_b4_formal.py`、`configs/tv3_mrs_ei/mei3_b4_formal_gate.json`，以及 B4 交互面上的 `tv3/ml/mrs_varpro.py`（`solve_s1` / `_solve_reduced` / `varpro_projected_jacobian` / `finite_difference_jacobian` / `composition_crb_o2_std`）与 `mrs_observation.py`。
- 独立种子 bootstrap（§5）是稳健性旁证，不是冻结协议的复算，已显式标注。

## 2. 主指标复核（引自 freeze，逐项与 §15 文档一致）

| 域    | S1 P90 | S2 P90 | S3 P90 | 相对改善    | bootstrap 95% CI 下界 | S1/S2 收敛失败率  |
| ---- | ------ | ------ | ------ | ------- | ------------------- | ------------ |
| test | 1.6604 | 1.6142 | 1.5232 | 0.02783 | 0.01157             | 0.1373（两法相同） |
| ood  | 0.7161 | 0.6985 | 0.6392 | 0.02453 | 0.00742             | 0.1281（两法相同） |

非退化门全部无退化；test 域 CO₂ MAE 两法之差仅 2.6e-9 pp。误差分位补充（test，O₂，pp）：S1 P50/P75/P95/P99 = 0.636/1.109/2.045/2.630；S2 = 0.581/1.055/1.961/2.619。OOD 整体约为 test 的 43%。

## 3. 机制归因：改善全部来自共享的 max_iterations 样本

从 `solver_comparison.csv` 配对复算：

| 量                               | test                     | ood                      |
| ------------------------------- | ------------------------ | ------------------------ |
| S1 失败数 / S2 失败数 / 交集            | 89 / 89 / 89             | 83 / 83 / 83             |
| 失败 stop_reason                  | 全部 `max_iterations`(100) | 全部 `max_iterations`(100) |
| 成功样本平均迭代数                       | ~4.5                     | ~4.3                     |
| 配对差 \|ΔO₂err\|<1e-6 的比例         | 0.846                    | 0.867                    |
| 成功配对最大 \|ΔO₂err\|（pp）           | <1e-5                    | 7e-5                     |
| 失败样本上 S2 目标值更低的比例               | 1.000                    | 1.000                    |
| 失败样本上 S2 误差更小的比例                | 0.989                    | 0.988                    |
| 反事实：仅失败样本换 S2 误差的 P90 改善        | **0.02783（=全量）**         | **0.02453（=全量）**         |
| 反事实：仅成功样本换 S2 误差的 P90 改善        | −0.00000                 | −0.00000                 |
| Spearman(Δobjective, Δ误差)（差异配对） | 0.955                    | 0.862                    |

解释：S1/S2 优化同一增广目标函数（B2 已证合成等价），收敛即同解；`objective_tolerance=1e-12`（相对变化）非常紧，13–14% 样本 100 次迭代内未触发。S2 每次迭代的前向成本约为 S1 一半，同样 100 次迭代内目标值下降更多，于是只在这批样本上取得更小误差。

**推论**：若提高迭代上限或放宽停止判据，两法可能在这些样本上同样收敛、点改善趋向 0。计划 §2 的问题“S2 是否更稳定、更高效地利用固定 K4 信息”，在估计器意义上答案是否定的；S2 的真实价值是**同解但省算力**。未来若重启 S2 主张，属于新预算 / 新停止判据下的重新对比，需要新契约，不得复用本 freeze 结论。

## 4. 失败样本剖析

- 失败样本误差并不灾难：test 域失败样本 O₂ 误差中位数 0.70 pp，成功样本 0.61 pp；失败样本在 S1 误差分布中的位置中位于第 54（test）/ 62（ood）百分位，仅约 10% 落在 P90 以上。“收敛失败率 13.7%”更接近紧停止判据下的慢收敛，而非发散或不可用输出。
- 失败集中在 `co2=0.03`、`T=35°C` 格点（CO₂ 弛豫信号最弱、目标面最平坦处）；S1/S2/S3 失败集合几乎一致（S3 略多：test 100 vs 89），说明是数据难度驱动，不是某个求解器的缺陷。
- S2 增益最大的 5 个 test 分层全部为 `co2=0.03`（多数 `T=35`）；而 S1 误差最大的若干分层两法增益为 0（双方同样失败、同样进展）。
- 条件相关性弱：Spearman(T, S1 O₂err) ≈ +0.08，Spearman(L, ·) ≈ −0.12~−0.15（长声程略有利），P/RH/组分真值无显著相关。

## 5. CI 判定稳健性与功效复盘

- **独立种子旁证（非冻结协议）**：seed=12345、4000 次分层配对重采样，test 域 percentile CI = (0.0124, 0.0724)，与冻结值 (0.0116, 0.0735) 行为一致——冻结判定不是种子巧合。
- bootstrap 分布系统性上偏（test：mean 0.040 > 点估计 0.028；BCa 偏差项 z₀ ≈ −0.64；basic CI 下界 −0.017）。**percentile 已是对 S2 最有利的区间方法，仍未过 0.02**；判定对 CI 方法选择稳健。
- 功效复盘：B3 预登记配对 P90 影响函数 std 上界 0.25，实测（bootstrap std × √648）test ≈ 0.41、ood ≈ 0.30，均超出上界；对备择 0.05 的真实功效约 0.6 而非规划的 0.86。但这不是败因——观测效应本身只有约 0.025，低于规划备择 0.05；即使每域样本加到约 7000，CI 下界也难越过 0.02。**结论：不是抽样运气，是效应真的小（且依赖迭代预算，见 §3）。**

## 6. 其他观察

1. **OOD 比 test 容易**（P90 0.72 vs 1.66）：高压提升弱方向条件数（B0 中 σ₂ 约 0.8 → 1.9），`pressure_extension_low_rh_216` 在本设计中不构成更难的域；gate 对 OOD 的实际语义是“在更易域上也要改善”。
2. **S3 上限差距很小**：S2−S3 平均 O₂ 误差差仅 0.006 pp（test）/ 0.019 pp（ood）——干扰参数可恢复余量接近用尽，残余误差由噪声与弱方向本身主导。
3. **方案 A 偏置问题确认解决**：正式数据上 O₂ 符号偏差 −0.07~−0.11 pp（pre-B4 宽先验 fixture 曾 +4 pp）；三组分闭包保持在 2.6e-14。
4. **相对 CRB 效率的聚合方式失真**：`mean_relative_crb_efficiency_o2`（S1/S2 约 2600，S3 达 2.6e6）是重尾比值 `crb²/err²` 的均值，被个别误差趋零的样本主导（单样本最大 1.7e9）；中位数只有 6–10。且 RMSE 低于逐点 CRB（比值 0.51–0.64）——MAP 估计带先验收缩、有偏，逐点 CRB 在此不构成下界。**引用该诊断应改用中位数或 `crb²/mean(err²)`（2.7–4.8），弃用 mean 值。**
5. 标定后验结构：`common_delay_s` 后验被下限 1e-8 s 截住（数据强约束）；`log_amplitude_gain` 与逐频偏移个体后验 std ≈ 0.01 基本等于先验——增益与偏移只有和被数据确定（每视图 4 条 log-amp 行对 5 个参数，增广条件数 8.65e6 即此简并）。对预测无害，S3 差距小说明代价可忽略。

## 7. 代码 review：确认正确的关键契约

- 授权门双重校验（`mrs_ei_b4_formal.py:903-922`）：merge 判定 + `stage_status` 直查不一致即抛 `RuntimeError`；未授权路径 exit 5，测试覆盖。
- S1 全部完成后才运行 S2、再 S3（`solve_paired_methods`）；三法共享同一 observation / covariance / spec / 冻结初值集 / 停止条件，精确配对；S3 真值隔离在独立文件、只经显式入参进入。
- 失败样本保留在主指标中（P90/MAE 不做 success 过滤）；`T_C/L_m/H_RH` 按混合物 join 为先验均值与初值（`apply_known_condition_priors` + `_initial_parameters`）——正是第一轮作废 freeze 缺失项。
- 正式规模强制校验 648/域（`generate_registered_sparse_dataset:270-276`）；smoke 必须 `--allow-incomplete-smoke` 且不写 stage_status；freeze 目录 staging+rename 原子写入、拒绝覆盖、写后即验 manifest。
- 标定集使用 `raw3_percent` 真值合法：B3 契约定义标定气组分已知，且只输出 nuisance 后验，不向 S1/S2 泄露评价域组分真值；字段白名单与 protocol 一致。
- `test_tunnel_ventilation_mei3_b4_formal.py` 5 项测试通过（31.9 s，2026-07-29 本地复跑）。

## 8. 代码 review：问题清单（按影响排序；行号为 2026-07-29 snapshot）

| 编号  | 问题                                                                                                                                                                                                                                        | 位置                                                                  | 影响                          |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------- |
| P1  | **forward_calls 计数不对称**：`varpro_projected_jacobian` 每列循环内无条件计算 `center_prediction`，双侧可行分支不使用——每迭代浪费 5 次真实前向且未计数（计数器只按 2×5 计）；S1 也有每迭代 1 次未计 center。真实物理前向比约 **1.5×**（成功样本 ~118 vs ~78），非报告的 2.1×（407.6 vs 193.7）。把 center 提出循环既修性能又让计数变诚实 | `mrs_varpro.py:439`（浪费）、`:862`、`:641`（计数）、`:574`                    | §6.3 效率诊断被夸大；不影响主指标与裁决      |
| P2  | `_component_errors` 的 success 分支是死逻辑：success 真假走同一计算；`failure_abs_error_percent=100` 只在估计非有限时触发（本次从未触发）。行为符合“失败保留主指标、用其实际估计”，但代码表达误导                                                                                                      | `mrs_ei_b4_formal.py:498-520`                                       | 可读性；B5 文档应写明失败样本按实际估计计误差    |
| P3  | `_promote_stage_status` 的 `parent_pre_b4_manifest_sha256` 取 `previous.evidence_manifest_sha256`：重复运行 B4 时 previous 已是上次 B4，指针错指——即执行计划头部第 13 行需手工校正的原因（当前 stage_status 已校正为 `94bdd50d…b2a9`，本次核对无误）                                       | `run_tv3_mei3_b4_formal_comparison.py:177`                          | 复跑时再次踩中；建议读专用字段并保持不变        |
| P4  | 相位噪声 0.1 rad 硬编码：design_space noise profile 无相位字段，单一事实来源落在代码字面量（与 B1 冻结值一致、生成与求解共用同一 dict，无内部矛盾）                                                                                                                                          | `mrs_ei_b4_formal.py:93`                                            | 契约卫生                        |
| P5  | 标定 GLS 构造 7776×7776 稠密协方差（约 480 MB）+ Cholesky，而协方差按构造是对角的（`mrs_observation.py:128`）；且联合后验拆成独立边缘先验 join，丢掉 gain↔offsets 强相关                                                                                                                | `mrs_ei_b4_formal.py:330-334, 360`                                  | 资源浪费；后验简并属登记实现选择（§6.5），非错误  |
| P6  | bootstrap 种子复用 split 种子（20260805/20260812 同时用于噪声生成与重采样）                                                                                                                                                                                   | `mrs_ei_b4_formal.py:832`                                           | 统计卫生 nit，实际影响可忽略            |
| P7  | §6.3 诊断缺口：墙钟时间未记录；多初值候选在 `_select_best` 后丢弃，freeze 无法复算多初值分散度 / 局部极小值率；B4 级条件数未入 freeze                                                                                                                                                   | `mrs_ei_b4_formal.py:578, 623`                                      | 计划 §6.3 列明“完整报告”的项缺失        |
| P8  | 非退化门语义歧义：计划 §6.2 写“相对退化超过 0.02”，实现为绝对差 `s2−s1 > 0.02`（对 MAE 是更严的绝对 0.02 pp，对失败率是更松的绝对 2 pp）                                                                                                                                               | `mrs_ei_b4_formal.py:834-849`                                       | 本次 S2 无退化未受影响；B5 契约应统一文字与实现 |
| P9  | 小项：`_promote_stage_status` 内 smoke 分支为死代码（main 在 smoke 时不调用）；`--mixture-limit-per-split` 实际限制条件数（×3 replicates），命名与语义不符；`evaluate_solver_gate` 把“域缺行”归入 `baseline_retained` 而非硬错误                                                         | `run_…comparison.py:160-165`、`:44-46`；`mrs_ei_b4_formal.py:820-822` | 可读性 / 边界行为                  |
| P10 | 测试容差偏松：delay 断言容差 5e-6 大于真值 2e-6 本身、gain 容差 0.05 大于真值 0.03（断言为 0 也能过）；smoke 误差上限（S3<2.0、S1<5.0）同属宽松烟测                                                                                                                                     | `test_…_b4_formal.py:100-101, 129-132`                              | 回归灵敏度低                      |

## 9. 验证状态

- **已验证**：manifest SHA256 与文档 / stage_status 一致且 `verify_evidence_manifest` 无 issue；执行计划 §15 全部数字与 freeze 逐项一致；三个历史 / 旁证 freeze 目录存在；B4 测试文件 5 项通过；本文所有统计从 `solver_comparison.csv` / `crb_efficiency.csv` / `registered_observations.json` 独立复算；工作区四个源文件与 freeze snapshot 逐字节一致。
- **未验证**：未运行三场景全量 pytest（本次审计未修改任何仓库代码）；§5 独立种子 bootstrap 是旁证，不替代冻结协议本身。

## 10. 对 B5 的建议

verdict 维持 `mei3_full_parameter_baseline_retained`、MEI-4 基线用 S1，均无需重议。建议 B5 契约额外写入四条注记，避免后人误读：

1. **机制归因**：S2 的 P90 改善全部来自共享 `max_iterations` 样本，属固定迭代预算内的优化器进展，不是统计效率差异；量级依赖 `max_iterations=100`。未来重启 S2 主张须以新预算 / 新停止判据的新契约重新对比。
2. **失败率语义**：13.7%/12.8% 是紧停止判据（`objective_tolerance=1e-12`）下的慢收敛，失败样本误差仅略高于成功样本，不等于不可用输出比例。
3. **CRB 诊断引用规则**：只引用中位数或 `crb²/mean(err²)`，弃用 `mean_relative_crb_efficiency_o2`。
4. **效率数字引用规则**：引用前向调用比时说明计数器不计 jacobian 中心点预测；按真实物理前向计，S2 约为 S1 的 1/1.5（修复 P1 后计数器即与真实一致）。

P1 计数修正只在未来复跑时有意义；当前 freeze 按只追加原则不动。

## 11. 复算方法附注

- P90 与冻结实现一致使用 `np.quantile(·, 0.9, method="linear")`；配对差容差 1e-6 pp。
- 反事实分解：`P90(把 X 子集换成 S2 误差、其余用 S1)` 与全量 P90 改善对比，X 分别取失败集与成功集。
- Wilcoxon（去零配对）：test n=100，p≈2e-17；ood n=86，p≈3e-15——方向显著（S2 优），与幅度门判定（未过 0.02 CI 界）不矛盾。
- 独立 bootstrap：与冻结实现同构（`design_condition_id` 分层、层内有放回、层大小 3），仅种子与次数不同（12345 / 4000）。
