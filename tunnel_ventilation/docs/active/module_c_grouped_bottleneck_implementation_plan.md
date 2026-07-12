# 模块 C：分组 Bottleneck 单变量对照实施计划

> 状态：**代码已落地，等待服务器正式 24 条矩阵与 verdict**
> 日期：2026-07-12
> 前置条件：B7 OOF Ridge residual 已完成 repeated split 与双 selector OOD 复核，判定 protocol_pass。
> 范围：仅检验冻结 RawDSP 特征上的物理分组 bottleneck；不在本轮引入 TabM、group gating、新特征、可微 DSP 或新数据划分。
> 成本约束：不做 multi training seed；冻结单一 training seed=42。矩阵为 4×3×1×2=24 条。

## Context

### 1. 当前有效证据

正式 clean tv3-formal-6000 的 RawDSP 链路已完成 frame fidelity、B1 Ridge parity、B6 target-scaled MLP 和 B7 OOF Ridge residual 验证。B7 的完整 R / L / S-Y / S-L × 3 split seed × 3 training seed 矩阵判定 protocol_pass，因而成为默认 RawDSP 回归头。

| 协议 | B7 test O2 R2 | B7 OOD O2 R2 | 相对 B1 的 test / OOD 增益 |
| --- | ---: | ---: | ---: |
| R | 0.6880 ± 0.0190 | 0.6429 ± 0.0256 | +0.2323 / +0.1923 |
| L | 0.6590 ± 0.0198 | 0.7006 ± 0.0231 | +0.2208 / +0.2606 |
| S-Y | 0.6756 ± 0.0119 | 0.4340 ± 0.0279 | +0.1740 / +0.7009 |
| S-L | 0.6975 ± 0.0100 | 0.7002 ± 0.0260 | +0.1895 / +0.2570 |

S-Y 与 S-L 的绝对 OOD 表现分化明显。S-Y 的三个 split seed 共享同一 target-margin OOD 集，而 S-L 才提供随 split seed 变化的 lhs-boundary OOD 复核。因此，本计划只对两个 selector 分列报告，不把任一单 selector 写成统一 OOD 或部署泛化水平。

当前 B7 residual MLP 直接读取 1008 维 flat RawDSP 特征，hidden dims 为 64、64，参数量为 68,931。训练样本约 4,200 条，输入由相邻窗口和统计量构成，存在强共线冗余。模块 C 要回答的是：

> 在不改变 Ridge 主趋势、RawDSP 链路和 raw3 输出契约的前提下，物理上可解释的分组压缩能否比同容量随机分组更好地保留非线性残差，并相对 B7 保持两类 OOD 表现？

### 2. 分组边界的必要澄清

现有 1008 列是“信号通道 × full 或 phase 或 early window × sequence statistic”的组合。因而“phase/window 统计”不是可以与 TOF、声速等并列的独立输入组；若同时按模态和按窗口分组，同一列会进入多个 encoder，破坏单一来源和单变量判读。

本计划采用按信号语义划分的、穷尽且互斥的 7 组。窗口和统计语义保留在各自原始列中，但不再构成重复分组。本轮不新增 corrected TOF、peak width、curvature、sidelobe、1/c² 或任何新物理坐标；它们不属于冻结 feature builder。

## Task

### 1. 正式不变量

1. 数据只使用 clean tv3-formal-6000，以及 B7 protocol 已审计的 R、L、S-Y、S-L 共 12 个派生 split。训练和评估仍按 mixture_id 分组；禁止把 mixture_id 改写或回退为 sequence_id。
2. 特征固定为 d0_raw_dsp_physics_stats_v1 的 1008 列、7 个正式 slow channels、7 个已锁定 RawDSP physics arrays、既有 sequence statistics、phase windows 和 early fractions。不得读取 simulator-derived observed 或 oracle 数组。
3. 每个派生 split 必须复审 split hash、RawDSP manifest、train-only template source、frame fidelity、feature names digest 和 B1 / B7 provenance。任一不匹配即失败，不重建物理仿真，不静默复用不匹配 cache。
4. 预测固定为三维 raw3：x_CO2、x_O2、x_N2。禁止 N2 回填、gas_head、ILR/ALR、target_transform、闭包强制损失和闭包残差头。sum_abs_error 仅监控。
5. Ridge 继续以全部 1008 个冻结特征训练；只有 residual MLP 的输入编码替换为分组 bottleneck。不得把 Ridge 改为分组 Ridge，也不得改变 Ridge alpha grid、OOF folds 或 OOF seed。
6. 残差训练严格沿用 B7：5-fold OOF Ridge target、full-train Ridge 推理、残差目标逐目标标准化、输出层零初始化、val 组合 O2 R2 early stopping。test 与 extrapolation 不得参与早停或模型选择。
7. 本轮唯一研究变量是 group assignment：物理分组与固定随机置换分组的架构、参数、优化器、训练 seed、OOF seed、split、Ridge 预测和 early stopping 完全相同。
8. group dropout 在本轮固定为 0。B7 同等的 activation dropout 固定为 0.1；group gating、组删除、group dropout 率和 bottleneck 维度搜索都属于后续实验。
9. 不允许用单次 val、单一 random split、L 或 S-L 的高分替代双 selector 结论；不报告 p 值。为控制训练成本，本轮冻结单一 training seed=42（与 B7 C0 配对），不做 42/123/456 multi-seed。
10. O2 的 0.8% bins 继续完整输出，但已知其局部 R2 为负，不作为本轮通过门。

### 2. 冻结特征分组

每个特征名都必须由唯一的通道 token 归入下表一组。实现应从缓存 feature_names 动态构建索引，而不是依赖当前列号；未知列、重复归组、漏归组或总数不等于 1008 时直接报错。

| group_id | 信号语义 | 归属通道或数组 | 列数 |
| --- | --- | --- | ---: |
| G1_tof | 到达时间 | ultrasonic_tof_observed_raw_dsp_s | 72 |
| G2_sound_speed | 重建声速 | ultrasonic_sound_speed_raw_dsp_m_per_s | 72 |
| G3_peak_response | 峰位置与相关峰响应 | ultrasonic_peak_index_raw_dsp、ultrasonic_corr_peak | 144 |
| G4_signal_quality | 信噪与帧质量 | ultrasonic_snr_db、ultrasonic_raw_dsp_quality、ultrasonic_raw_dsp_accepted | 216 |
| G5_ndir_co2 | CO2 光学慢通道 | V_NDIR_CO2 | 72 |
| G6_tcs | 热导慢通道 | V_TCS | 72 |
| G7_environment_geometry | 环境和几何慢通道 | T_C、P_MPa、H_RH、L_m、piston_position_m | 360 |
| total | — | — | 1008 |

对每组的 72 列，列语义仍包含 full、baseline、exposure、steady、recovery 和 early fractions，以及 mean、std、min、max、range、first、last、delta、slope。G3、G4、G7 的列数分别为 2、3、5 个通道的 72 倍。

### 3. P0 模型

公开预测保持：

~~~
Y_hat(X) = Ridge_full(X) + Residual_grouped(X)
~~~

其中 Ridge_full 的训练、预测和 OOF residual target 与 B7 完全一致。对 residual_grouped：

~~~text
每组输入 Xi
  Linear(dim(Xi) -> 16)
  LayerNorm(16)
  SiLU
  Dropout(0.1)

concat(G1 ... G7) = 112
  Linear(112 -> 64) -> ReLU -> Dropout(0.1)
  Linear(64 -> 64)  -> ReLU -> Dropout(0.1)
  Linear(64 -> 3, zero initialized)
~~~

该配方的可训练参数量为 28,051，较 B7 的 68,931 减少约 59%。最后一层必须严格零初始化，因此任一训练开始前均满足：

~~~text
Residual_grouped(X) = 0
Y_hat(X) = Ridge_full(X)
~~~

每个 group encoder 的输入 scaler 只 fit 当前 train split；残差 target scaler 只 fit 当前 train 的 OOF residual。val residual 仅由 full-train Ridge 产生。所有公开预测均反变换回原始百分比单位的 raw3。

### 4. 单变量对照矩阵

| variant | group assignment | 用途 | 新训练数 |
| --- | --- | --- | ---: |
| C0_flat_b7 | 既有 flat 1008 输入，B7 冻结结果 | 实用性能与成本锚点 | 0 |
| C1_physical_grouped | 上述 G1 至 G7 物理分组 | 模块 C 主候选 | 12 |
| C2_permuted_grouped | 固定随机置换后按同一组尺寸切分为 72、72、144、216、72、72、360 | 排除“只因稀疏化或降参” | 12 |

C1 与 C2 是本 P0 的因果对照：二者的网络、参数量、优化变量、数据和 training seed 相同，仅列到 group encoder 的分配不同。C0 与 C1 的比较只用于判断候选是否能替代默认 B7，不用于将性能差异全部归因于物理分组。

C2 的 permutation_seed 预注册为 20260712。置换后的 feature names digest、permutation digest、组尺寸与所有 group index 必须写入 metrics；不得在任何 test 或 OOD 结果后更换置换。

每个 C1 / C2 运行覆盖：

~~~text
R / L / S-Y / S-L
  × split seed 20260704 / 20260712 / 20260720
  × training seed 42   # 单 seed，不做 multi-seed
~~~

每个候选共 12 条训练，P0 共 24 条新增训练。derived split、RawDSP cache 与 frame fidelity 只在通过 provenance 审计后复用；本实验不重新划分数据、重建模板或重跑 DSP。

### 5. 预注册验收与停止条件

主指标是 O2 R2 的同 split、同 training seed paired difference。R、L 的 test 结果用于 ID 稳定性背景；S-Y、S-L 的 test 和 extrapolation 是正式 selector 证据。

#### 5.1 运行完整性

以下任一项不满足即为 failed，不进入数值比较：

1. C1 或 C2 的 12 行矩阵不完整，或同一行缺少 C0 B7（seed=42）配对记录。
2. split / cache / template / fidelity / feature names / config hash 任一审计失败。
3. feature group mapping 不是 1008 列的严格划分，或 C2 的 permutation digest 不等于预注册值。
4. OOF coverage 不完整、Ridge 读取 val/test/OOD 标签、early stopping 不为 val 组合 O2 R2、或输出不是有限 raw3。
5. 缺失 CO2、O2、N2、sum_abs_error、0.8% bins、parameter count、训练时长或训练配置。

#### 5.2 bottleneck_pass

只有同时满足下列条件，C1 才判为 bottleneck_pass：

1. 在 R、L、S-Y、S-L 的 test paired mean delta O2 R2 中，C1 相对 C0 均不低于 -0.01。
2. 在 S-Y、S-L 的 extrapolation paired mean delta O2 R2 中，C1 相对 C0 均不低于 -0.01。
3. 在 S-Y、S-L 中，C1 相对 C2 的 test 与 extrapolation paired mean delta O2 R2 均不为负；两类 OOD 中至少一类的 extrapolation paired mean delta 不低于 +0.01。
4. C1 的参数量必须等于预注册架构的 28,051，且 C1 / C2 参数量相同。

该判定表示：物理分组在保持 B7 主要性能的同时，给出了超出随机分组的方向一致证据。它不是统计显著性、真实硬件泛化或突破窄区间物理上限的声明。因跳过 multi-seed，结论强度弱于原 72 条矩阵设计。

#### 5.3 分流

| 结果 | 判读 | 动作 |
| --- | --- | --- |
| bottleneck_pass | 物理分组在双 selector 下可承接 B7，且优于随机分组 | 只以 C1 encoder 启动下一步 residual TabM 计划 |
| compression_only | C1 对 C0 非劣，但未超过 C2 | 可报告压缩效率，不将其解释为物理分组收益；停止 group 结构扩展 |
| grouped_failed | C1 相对 C0 任一主门低于 -0.01，或出现全负 seed cluster | 停止 grouped bottleneck 调参，保留 B7 默认头 |
| audit_failed | 证据或契约无效 | 先修审计或实现；不产生性能结论 |

本 P0 不执行 group gating、bottleneck 维度 8 / 16 消融、特征选择或 TabM。residual TabM 只能在本计划生成完整结论且 bottleneck_pass 后立项。

## Format

### 1. 实施范围

| 范围 | 必需改动 |
| --- | --- |
| tv3/ml/grouped_bottleneck.py | 定义冻结的 7 组 mapping、feature names 校验、C2 permutation 和 grouped encoder module。不得生成新特征。 |
| tv3/ml/grouped_ridge_residual_head.py | 新增 grouped OOF Ridge residual regressor；复用 B7 的 Ridge、OOF、raw3、target scaling、early stopping 与零初始化语义。B7 实现保持冻结。 |
| tv3/ml/rocket_training.py | 注册 grouped OOF residual head，强制 RawDSP evidence / B7 protocol provenance，写入 group diagnostics。 |
| tv3/pipeline/run_tv3_rocket_baseline.py | 增加 head 与配置白名单，仅暴露预注册的 group spec、bottleneck dim、permutation seed 和 group dropout。 |
| configs/tv3_module_c_grouped_bottleneck_physical.json | C1 冻结配置，继承 B7 RawDSP 契约，只替换 grouped residual 字段。 |
| configs/tv3_module_c_grouped_bottleneck_permuted.json | C2 冻结配置；除 group assignment、control 名称与 pre-registered permutation seed 外必须等于 C1。 |
| scripts/run_module_c_grouped_bottleneck_protocol.py | 复审 B7 protocol 的 12 个派生 split 和证据后，编排 C1 / C2 的 24 条运行（单 training seed=42）、paired 汇总与 verdict。 |
| tests/test_tv3_module_c_grouped_bottleneck.py | 覆盖 mapping、permutation、参数量、OOF、零初始化、raw3、early stopping 和 diagnostics。 |
| tests/test_tv3_module_c_grouped_bottleneck_protocol.py | 覆盖矩阵完整性、B7 配对、provenance、S-Y / S-L 分列和 verdict。 |

实现不得复制第二套 Ridge、MLP、指标或 split 审计逻辑。若既有 B7 helper 未暴露可复用的稳定接口，应先抽取单一公共 helper，并由 B7 回归测试锁定其行为。

### 2. metrics 与汇总产物

正式运行产物与 tv3_b7_protocol 同层，按模块 C 实验统一归档：

~~~text
outputs/tv3_module_c_grouped_bottleneck/
  physical/
    R|L|S-Y|S-L/
      split_<seed>/seed_<seed>/metrics.json
  permuted/
    R|L|S-Y|S-L/
      split_<seed>/seed_<seed>/metrics.json
  protocol_manifest.json
  result_matrix.csv
  split_metrics.json
  result_matrix.md
  verdict.md
~~~

每个 metrics.json 除既有 train、val、test、extrapolation 指标外，必须包含：

~~~json
{
  "head": "grouped_oof_ridge_residual_mlp",
  "grouped_bottleneck": {
    "group_spec": "raw_dsp_physics_groups_v1",
    "group_assignment": "physical|permuted",
    "group_counts": {},
    "group_bottleneck_dim": 16,
    "group_dropout": 0.0,
    "feature_names_digest": "",
    "permutation_seed": null,
    "permutation_digest": "",
    "parameter_count": 28051
  },
  "early_stopping": {
    "monitor": "val_o2_r2",
    "uses_combined_ridge_prediction": true
  }
}
~~~

### 3. 最小验证

~~~bash
python -m pytest -q tests/test_tv3_b7_oof_residual.py tests/test_tv3_raw_dsp_pipeline.py tests/test_tv3_module_c_grouped_bottleneck.py tests/test_tv3_module_c_grouped_bottleneck_protocol.py
python scripts/run_module_c_grouped_bottleneck_protocol.py --dry-run
~~~

在服务器完成全部正式矩阵后，额外复审：

1. C1 / C2 每组均有 12 条成功训练，且 24 条运行均能关联同一行 C0 B7（seed=42）；
2. B7 split_metrics 必须为 protocol_pass、matrix_complete 且完整含 36 条冻结配对行；
3. 所有派生 split 的 hash、cache manifest、template source、fidelity、feature names digest 与 frozen config 一致；
4. S-Y 与 S-L 分列的 paired delta、最差 split seed 和未聚合行完整可追溯；
5. 输出没有覆盖历史 B7 或 B1 正式 metrics。

### 4. 文档回填规则

在完整 24 条运行、审计和 verdict 生成前，不更新项目记忆库的正式结论，也不把任何单 split 结果写为模块 C 结论。

完成后才回填：

1. 项目记忆库的当前执行路线、主结果表、停止条件和工程入口；
2. 深度学习算法研究方向与文献路线的模块 C 状态和实验矩阵；
3. 本文“实施记录”；
4. active/README.md 的状态与链接。

## 实施记录

### 2026-07-12 — 规划建立

已根据 B7 protocol_pass 的完整正式矩阵，冻结 P0 的问题定义、7 组不重叠 mapping、C1 / C2 单变量对照、验收门与停止条件。

### 2026-07-12 — 代码落地

已实现并本地验证：

| 产物 | 路径 |
| --- | --- |
| 分组 mapping / encoder | `tv3/ml/grouped_bottleneck.py` |
| grouped OOF residual head | `tv3/ml/grouped_ridge_residual_head.py` |
| 训练注册 | `tv3/ml/rocket_training.py`、`tv3/pipeline/run_tv3_rocket_baseline.py` |
| C1 / C2 冻结配置 | `configs/tv3_module_c_grouped_bottleneck_physical.json`、`..._permuted.json` |
| 协议编排 | `scripts/run_module_c_grouped_bottleneck_protocol.py` |
| 测试 | `tests/test_tv3_module_c_grouped_bottleneck.py`、`tests/test_tv3_module_c_grouped_bottleneck_protocol.py` |

实现要点：

1. 从 B7 抽出公共 `build_oof_ridge_predictions`；B7 行为由既有测试锁定。
2. Ridge 仍吃全部 1008 列；仅 residual 路径换成 7 组 bottleneck（参数量 28,051）。
3. C1 / C2 仅 `group_assignment` 与 `permutation_seed` 不同；C2 `permutation_seed=20260712`。
4. 本地验证：`pytest` Module C + B7 residual 全过。

### 2026-07-12 — 砍 multi-seed，矩阵缩为 24 条

应成本约束，取消 training seed 42/123/456 三重复：

- 冻结 `TRAINING_SEEDS = (42,)`，与 B7 C0 的 seed=42 配对；
- 正式矩阵：4 protocols × 3 split seeds × 1 training seed × 2 variants = **24** 条；
- 去掉 multi-seed cluster 门，仅保留 split 间 paired mean 门；
- C0 加载从 B7 的 36 行中过滤出 seed=42 的 12 行。

**尚未**在服务器跑完整 24 条正式训练，**未**产生正式性能结论，也未回填项目记忆库。

服务器入口：

```bash
python scripts/run_module_c_grouped_bottleneck_protocol.py --dry-run
python scripts/run_module_c_grouped_bottleneck_protocol.py --stage all
```
