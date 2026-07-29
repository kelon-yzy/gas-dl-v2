# MEI-0 / MEI-1 F2--F5 证据处置执行报告

> 执行日期：2026-07-28  
> 结论：F2--F5 已按限定结论范围转为 `parked_nonblocking`；MEI-1 完整审计通过并固定 D0 K4，下一阶段为 `MEI-3_varpro_audit`。

## 1. 文献与公开数据

事实源：`docs/references/tv3_mrs_ei_f2_f5_public_evidence_review_20260728.md`，SHA256 `04efacec4dc8e90eff5413b162759fbd9dc18baeee1d4777cc1565df59740795`。

- F2：确认 Bass/ISO O2/N2 湿度与压力缩放链，以及 H2O--O2/N2 实验速率；完整目标域 H2O 强度 holdout 未找到。
- F3：确认 Dain--Lueptow 多组分耦合理论及 Ejakov 等多频变压实验验证；当前项目完整耦合实现未完成。
- F4：确认有限孔径对幅度和相位的影响及流量计衍射修正路径；项目实际孔径和安装几何未冻结。
- F5：确认空气中 50--300 kHz 互易校准方法、个体幅相差和流量计非互易；当前设备校准数组不存在。

没有发现可直接作为当前设备与完整温压域独立 holdout 的公开原始数组。未解决部分均登记了非阻断范围、继续禁止的声明和复查触发条件，没有升级为 `represented_traceable`。

## 2. 新 MEI-0

- freeze：`20260728T063115704201Z_8c02b8635dd7`
- parent：`20260728T020915548649Z_f47f6f51d1b1`
- verdict：`mei0_registry_frozen`
- input contract：`8c02b8635dd7d407a82f8bb882b4ba2468e3fde7ba1fb4ab1a503c0e51ae5d50`
- manifest SHA256：`8ae3874750cccd284dd2a842acd4ff721847ec28ffa2f36fa4926eb80c1e8bda`
- registry preflight issues：`[]`
- `delta_numerical`：低成本 `2.8210630817324963e-05`；stress `9.055394884241296e-05`
- `delta_practical`：`0.02`

## 3. 新 MEI-1

- freeze：`20260728T064100731550Z_1b55aa2e09cb`
- parent MEI-0 manifest：`8ae3874750cccd284dd2a842acd4ff721847ec28ffa2f36fa4926eb80c1e8bda`
- verdict：`mei1_fixed_k4_retained`
- passed：`true`
- manifest SHA256：`faf397f9457b8eadc8871c55e488da0d62671826bf724ac3fd66f9c03b029396`
- points / designs：`432 / 15`
- blockers / issues：`[] / []`
- parked families：F2、F3、F4、F5
- pressure domain：`parked_nonblocking`、`diagnostic_only_not_primary_gate`
- decision reason：`design_ranking_not_resolvable_within_delta_practical`
- frozen design：`D0_fixed_k4_25_63_100_200_khz`
- allowed next stage：`MEI-3_varpro_audit`

低成本正式并集排名跨度仍为 `0.004564833712193834`，最佳 K4 相对固定 K4 改善 `0.0014822820748954114`，均低于 2% 实践界。因此没有启动 MEI-2 优化，而是固定 D0 K4。

首次技术通过运行 `20260728T063303617434Z_a0ff46eab644` 的物理结果相同，但顶层产物未显式抄出 parked family、decision reason 和 frozen design。补齐输出契约后重新运行；首次目录只读保留，不提升为当前状态。

## 4. 授权边界

以下四项继续为 `forbidden_until_explicit_authorization`：

| 字段 | 含义 | 本次未授权原因 |
| --- | --- | --- |
| `registered_sparse_simulation_generation` | 生成新的正式登记稀疏谱仿真观测 | MEI-3 只获准进行确定性求解审计，尚未批准新的正式数据生产 |
| `formal_waveform_generation` | 生成带激励、传播、设备响应和采样噪声的完整波形 | F4/F5 的实际几何和设备响应仍为搁置项 |
| `benchmark_packaging` | 将数据、schema、划分、manifest 和指标打包为正式 benchmark | MEI-6 的数据与模型协议尚未获准 |
| `hardware_trial` | 开展真实换能器、管道或压力台架试验 | 尚无独立设备、风险、成本和采集协议授权 |

`allowed_next_stage=MEI-3_varpro_audit` 只允许开展下一阶段数值审计，不等于以上任一活动已获授权。`forbidden_until_explicit_authorization` 表示等待单独审批，不是永久禁止。

F2/F3 的搁置不支持绝对衰减或完整耦合物理声明；F4/F5 的搁置不支持设备幅相、实际声能、硬件精度或现场泛化声明。

## 5. 验证

- JSON 解析：通过。
- Python compile：通过。
- MEI-0 / MEI-1 专项测试：`59 passed in 2.36s`，包含搁置字段完整性和真实未表示项回退阻断测试。
- MEI-0 preflight：`mei0_registry_frozen`，issues `[]`。
- MEI-1 正式运行：退出码 `0`。
- 两个有效 manifest 独立校验：均为 `[]`。
- 相关 MRS 回归：`34 passed in 0.97s`。
- Python compile 与 `git diff --check`：通过；仅有用户已有文档的 CRLF 到 LF 提示。
