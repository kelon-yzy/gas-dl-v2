# MEI-4 执行进度记录

> 记录时间：2026-07-30  
> 当前阶段状态：`mei4_mc_authorized_pending_execution`  
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
