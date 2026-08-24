# MEI-4 执行进度记录

> 记录时间：2026-07-30；2026-08-15 追加复读与审查记录  
> 当前阶段状态：`mei4_mc_authorized_pending_execution`（未变）  
> 结论状态：未产生 MEI-4 科学 verdict

## 已冻结的研究工作

MEI-4 已完成 C0 执行契约、C1 后验机制审计和 C2 确定性后验评价。所有计算只读取 B4/B5 freeze；`mixture_id` 保持组分主键，方法层未读取真值、CRB 或干扰真值，未使用温度缩放、先验缩窄或 conformal 校准。

| 阶段 | 证据 | 可作出的事实陈述 |
| --- | --- | --- |
| C0 | `20260730T025640042212Z_d3505e1a3e0c` | 后验域、拒绝语义、24 条覆盖接受带、M2 PSIS 门和 C3/C4 协议均已预注册。 |
| C1 | `20260730T053939880570Z_d4b88b625f0c` | 后验构造与估计量通过合成机制审计，负对照可显式失败；这不是正式校准证据。 |
| C2 | `20260730T071532806157Z_76811228bcea` | 24 个 S1 复解探针全部一致。M1、M1b、M2 均未通过完整主覆盖门；M2 在 test 的 PSIS 超阈率为 `35 / 648 = 5.40%`，合规触发 M2b。 |
| C3 授权 | `20260730T080818819647Z_6c6b2da21139` | 用户已授权 MEI-4 范围的 SBC、PPC 和条件 M2b；CC-SBI 训练抽样仍未授权。 |

## 已停止的 C3 运行

2026-07-30 启动了 `run_tv3_mei4_c3_mc_calibration.py --run-authorized-mc`。运行日志表明：

- SBC 的 test 与 OOD 域均到达 `1000 / 1000`；
- PPC 的 test 与 OOD 域均到达 `648 / 648`；
- M2b 日志只确认到 `1 / 1296`，随后按用户要求终止后台进程。

该实现将 C3 聚合报告和 freeze 写入放在 SBC、PPC、M2b 全部成功之后。因此终止时没有 C3 完成 manifest，没有任何 C3 聚合 JSON，也没有可供 C4 或 C5 使用的数值结果。内存中的 SBC/PPC 结果已随进程终止丢失，不能视作正式证据。

## 未执行与边界

- C3 全量 M2b、C3 完成 freeze 和 C3 manifest 校验尚未完成。
- C4 触发审计尚未运行；虽已准备审计代码和契约，但不得据此声称 CC-SBI 被触发。
- 未生成 CC-SBI 训练样本，未训练模型，未生成波形，未打包 benchmark，未开展硬件工作。
- C5 裁决与 MEI-4 verdict 尚未执行；`allowed_next_stage` 继续为 `null`。

## 恢复条件

恢复 C3 时必须从 C0/C2/C3 授权 freeze 重新执行完整登记的 SBC、PPC 和 M2b 流程，并仅在生成新的 C3 完成 freeze、核验 manifest 后读取结果。现有 `registered_sparse_simulation_generation` 记录仍限定为 MEI-4 的 SBC、PPC 与条件 M2b；任何 CC-SBI 训练仍需新的 `mei4_cc_sbi_training_draws` 明确授权。

---

## 2026-08-15 复读与审查记录

本次只读取既有 freeze 与源码，未运行任何观测空间抽样，未写入 attempt 分片、报告或 freeze，阶段状态与全部授权均未改变。完整内容见 [MEI-4 执行计划](tv3_mrs_ei_mei4_execution_plan.md) §0.1–§0.3。

### 可作出的新事实陈述

1. **C2 的覆盖失败可分解到机制，但拒绝原因按方法不同。** M1/M1b/M2 在 test 域的拒绝率为 `31.6% / 28.2% / 37.0%`，OOD 域为 `6.5% / 6.5% / 11.0%`。`laplace_diagnostics.json` 的 `rejection_reasons`：M1 的 205 次与 M1b 的 183 次拒绝**全部**为 `truncation_interval_numerical_failure`；M2 的 240 次由 `m1_proposal_unavailable` 205 次与 `psis_k_hat_exceeded_for_M2` 35 次组成，后者与截断无关。`curvature_not_positive_definite` 与 `curvature_condition_number_exceeded` 在六个分组中一次都未出现。截断的触发点是非负域内接受的 Sobol 候选低于 `2048 / 65536`，即超过 96.9% 的切空间高斯质量位于物理定义域之外。
2. **在能构造区间的样本上，覆盖率高于名义值。** M1 的 test O₂ 选择条件覆盖率为 `0.6072 / 0.9120 / 0.9842 / 0.9977`，OOD 为 `0.7888 / 0.9901 / 1.0000 / 1.0000`，对应名义 `0.5 / 0.8 / 0.9 / 0.95`。方向是过覆盖。注意 `coverage_report.json` 的 `coverage` 字段与 `within_acceptance_band` 判定用的都是无条件值（M1 test O₂ 为 `0.4151 / 0.6235 / 0.6728 / 0.6821`）；M1 的 8 条 O₂ 主覆盖带中只有 `ood @ 0.95` 落在接受带内。
3. **区间宽度超过组分工作量程。** M1 的 test O₂ 区间宽度中位数为 `1.9669 / 3.7383 / 4.7971 / 5.7171` 个百分点（50/80/90/95%），而 O₂ narrow 采样量程只有 `3.20` 个百分点；OOD 的 95% 区间为 `3.2061`，恰好等于量程。`posterior_intervals_test.csv` 中 O₂ 区间下界最小到 `14.931%`，低于采样下界 `18.00%` 三个百分点以上。
4. **拉普拉斯曲率本身正确。** `o2_laplace_to_crb_ratio` 中位数为 test `1.00035`、OOD `1.00004`。
5. **S1 点估计本身可能出域。** `complete_hessian` 审计的 24 个探针中有 4 个状态为 `unavailable`，错误为 `S1 parameters are outside the registered physical domain`（test `M000649` / `M001001`，OOD `M001297` / `M001473`）。`s1_replay` 的 24/24 一致只说明复解可重现，不说明解落在物理域内。这条说明问题不限于后验尾部泄漏，似然的峰有一部分本就在物理域外。
6. 上述各项互相印证，判读为：冻结 K4 与登记噪声下，单样本 O₂ 后验不足以支撑比"落在工作量程内"更细的陈述。该判读与 B4 的 S1 test O₂ P90 `1.6604`、MRS-2 最佳臂 median P90 ≈ `2.40 vol%`、MRS-6 的 `1.0–1.3 vol%` 饱和层一致。
7. **对后续指标选法的直接结论。** 因第 5 条成立，把组分先验换成登记 LHS 规格之后，后验将由先验主导、测量贡献接近零。主指标必须用先验宽度到后验宽度的收缩比，用绝对区间宽度会得到假通过。这与 §5 对 CC-SBI 的既有要求一致，但此前未对 M1/M1b/M2 登记。

契约常量、分方法拒绝构成与实测宽度表已写入[代码契约事实源](../掘进通风代码契约事实源.md) §10.1，本文档不再重复维护这些数字。

以上均为对 C2 freeze `20260730T071532806157Z_76811228bcea` 内既有产物的复读，**不是**新 verdict，不改写 `mei4_c2_intermediate_no_pass_verdict`。

### 需要在 C3 恢复前处置的三项审查发现

1. **M2b 结构上无法通过完整校准门**：C0 的 `mc_protocol` 未为 M2b 登记 SBC/PPC，`mrs_ei_mei4_c4.py` 的 `_m2b_gate()` 因此把其 `complete_calibration_gate_passed` 置为 `False`；叠加 C2 已确定的 M1/M1b 覆盖失败，T1 在 C2 结束时即已注定为 `true`，`mei4_deterministic_posterior_retained` 不可达。
2. **T4 触发条件的实现存在循环依赖**：实现从实测 `m2b_report["cost"]["forward_calls"]` 判定，而 T4 的用途是跳过 M2b。按 B4 的 S1 平均 `forward_calls=395` 事前估算，M2b 登记规模需约 `1.02e8` 次前向调用，为登记预算 `1.0e6` 的约 102 倍。
3. **组分先验规格不一致**：M1/M1b/M2 在单纯形上使用平坦组分先验，而 CC-SBI 按 §5 要求联合采样组分，即使用真实 LHS 生成先验。二者不可直接比较，须在 C4 启动前择一预注册处置方式。

### 非正式工程测量

本机对已冻结观测做参数空间复解：单次前向 `73.6 μs`，S1 单初值求解 `204.6 ms`，每混合物 3 初值 `613.9 ms`。据此 M2b 登记规模 `777,600` 次求解器启动约为单核 `44.2 h`、12 worker 理想线性 `3.7 h`。这些数字属于 C3 优化计划 §10 P0 的工程基准，不得写入正式 C3 freeze，也不替代 §10 P3 的正式 worker 基准。

### 仍未改变的边界

C3 全量、C3 完成 freeze、C4 触发审计、CC-SBI 训练与 C5 裁决均未执行；`allowed_next_stage` 继续为 `null`；波形、benchmark、硬件与 `mei4_cc_sbi_training_draws` 四项授权继续禁止。

