# tv3 可辨识性与误差预算实施计划

> 状态：**规划中；前置 B1/B7 基线冻结未完成**
>
> 责任：在不新增回归模型、不改正式 RawDSP builder 的条件下，量化 O₂ 声学信息相对 nuisance 不确定度的理论边界，并给出是否继续追求窄区间连续回归的可审计 verdict。

## Context

### 1. 已冻结证据与当前问题

- clean `tv3-formal-6000` 上，`d0_raw_dsp_physics_stats_v1`、B1 Ridge、B7 OOF Ridge residual 的重复 split 与 S-Y、S-L 双 selector 协议均已通过；B7 是当前默认 RawDSP 头。
- 0.8% O₂ bins 内，即使 oracle 特征的 R²仍为负；当前缺少的不是又一个回归头，而是“组分信号是否大于温度、湿度、流速、声程、延迟、jitter 与 SNR 误差”的定量答案。
- 当前仿真未覆盖或未显式建模的 nuisance 不能被伪装为已有证据。该类变量必须在 manifest 中标为 `not_represented`，其误差预算只可作为待补齐物理的需求，不能写成已验证上限。

### 2. 前置基线门

执行数值审计前，必须在 clean Git worktree 上运行：

```bash
python scripts/freeze_tv3_baseline.py
```

产物固定为 `outputs/tv3_baseline_freeze/`，至少含 B1/B7 配置与结果 hash、12 个派生 split 的 split/OOD hash、12 个 RawDSP cache 的 build/template digest、帧级 fidelity、环境和 Git commit。若任一项不一致或工作树未清理，命令必须失败，不产生“部分冻结”结果。

## Task

### 1. 目标与非目标

目标：

1. 对当前冻结物理链路中的 O₂、CO₂、T、P、RH、L、flow projection、固定延迟、trigger jitter、SNR 等变量建立“已表示 / 未表示”清单。
2. 在全 O₂ 范围与预注册 0.8% 窄窗口内，计算声速、TOF 和已声明独立的波形统计量的局部灵敏度、条件数、Fisher information、CRLB 与 nuisance-marginalized O₂ information。
3. 将每项 nuisance 误差折算为等效 O₂ 误差，给出联合 P90 误差与主导项排序。
4. 输出连续回归、分档趋势监测或信息源升级的 verdict；不以本审计选择或训练新模型。

非目标：

- 不修改 `d0_raw_dsp_physics_stats_v1`、B1、B7、训练 split、OOD selector 或 raw3 输出契约。
- 不把 true TOF、true sound speed 或 true alpha 加入预测输入；它们只可出现在 oracle 审计列。
- 不以当前 v1 中缺失的湿空气、双向流速或设备参数构造虚假的端到端性能结论。
- 不启动 TabM、DeepSets、可微 RawDSP、蒸馏或自监督分支。

### 2. 输入契约与参数来源

新增审计配置 `configs/tv3_identifiability.json` 必须显式给出：

- 基线目录、B1/B7 builder 名称与 `tv3_baseline_freeze/manifest.json` 的 hash；
- 组分域、窄窗口中心及宽度、各 nuisance 的物理范围、标准差或协方差来源；
- `target_p90_o2_error_percent`、`max_nuisance_fraction_of_signal` 与业务可接受拒绝率。

业务精度未定义时，执行只允许输出 `inconclusive_missing_business_threshold`，不得自行选择阈值并给出继续或停止结论。

参数清单的每一行包括 `name`、unit、representation、source、distribution、correlation_group、是否部署可测、是否由当前 v1 表示。范围必须来自 schema、物理配置或校准记录；禁止在实现中复制魔法数。

### 3. 计算与不变量

1. 在每个组成点保持 `x_CO2 + x_O2 + x_N2 = 100%`；扰动一个自由组分时明确记录补偿组分，禁止产生不闭合点。
2. 对 `c`、`tof = L/c` 和候选统计量计算中心差分；边界使用单边差分并在结果中标记。差分步长需要做稳定性检查，导数随半步与双步变化超过预注册容差时显式失败。
3. 共享同一 TOF 信息的 `c`、TOF 派生量不得在 Fisher 矩阵中重复计数；观测协方差必须说明来源。协方差不可逆、条件数超阈或未建模变量主导时，Fisher / CRLB 结论必须标为不可用。
4. 单 nuisance 等效 O₂ 误差定义为其观测扰动除以对应 O₂ 灵敏度；联合误差使用登记协方差传播。所有除零、非有限数或缺失物理映射直接报错。
5. 分别报告全域和窄窗口；bin 内 R²只作已有模型诊断，不替代本审计的 MAE / P90 物理误差判断。

### 4. 实施范围

| 文件 | 动作 | 责任 |
| --- | --- | --- |
| `tv3/audit/identifiability.py` | 新增局部敏感度、参数表示审计、Fisher 与条件数计算 | 不训练模型 |
| `tv3/audit/error_budget.py` | 新增单项及联合等效 O₂ 误差传播与 P90 汇总 | 不吞掉未表示变量 |
| `scripts/run_tv3_identifiability.py` | 读取冻结基线和配置，写入唯一正式目录 | 输出不可覆盖 |
| `configs/tv3_identifiability.json` | 新增参数域、协方差来源与业务门限 | 无隐式默认阈值 |
| `tests/test_tv3_identifiability.py` | 覆盖差分、闭包、Fisher、未表示变量和 verdict 分流 | 纯小型数值 fixture |

正式产物目录为：

```text
outputs/tv3_identifiability/
  manifest.json
  representation_audit.json
  sensitivity.csv
  fisher_information.csv
  error_budget.csv
  narrow_window_summary.csv
  metrics.json
  audit.json
  verdict.json
  README.md
```

### 5. 判定与分流

| verdict | 条件 | 后续动作 |
| --- | --- | --- |
| `continuous_regression_supported` | P90 O₂ 误差满足业务门限；联合信息不近奇异；关键 nuisance 残余小于窄窗口信号的预注册比例 | 才进入双向 / 湿空气等按主导项排序的物理改进 |
| `coarse_monitoring_only` | 窄窗口 P90 或 nuisance floor 不满足门限，但全域仍有可辨识信息 | 冻结精细连续回归，转为分档与趋势报告 |
| `information_source_upgrade_required` | 全域信息不足、未表示 nuisance 主导或可测校正后仍超门限 | 评估多频与恢复 TDLAS 的条件 |
| `inconclusive_missing_business_threshold` | 缺少业务精度或协方差依据 | 不做继续 / 停止声明，补齐输入 |
| `audit_failed` | 基线 hash、物理映射、数值稳定性或闭包审计失败 | 修审计或物理参数，不训练新头 |

若 flow projection 是主导可测误差，下一步是 `raw_dsp_bidirectional_v1` 的单向 / 外部风速校正 / 双向解耦对照；若 RH 或设备参数主导，下一步是 `acoustic_measurement_v2`。不得在两种机制尚未量化前同时修改前端、数字孪生和模型头。

## Format

### 1. 执行顺序

1. 运行基线冻结并校验 `verdict.json.status=frozen`。
2. 审计每个参数在当前物理链路中的表示状态与来源，缺失项停止在 `representation_audit.json`。
3. 先完成解析可验证的声速 / TOF 敏感度，再加入可审计的波形统计量与协方差传播。
4. 分别生成全域、四个预注册窄窗口和 worst-case nuisance 组合结果。
5. 仅在业务门限和协方差齐备时生成最终 verdict；将结论同步回项目记忆库和统一路线。

### 2. 最小验证

```bash
python -m pytest -q tests/test_tv3_baseline_freeze.py
python -m pytest -q tests/test_tv3_identifiability.py
python scripts/freeze_tv3_baseline.py
python scripts/run_tv3_identifiability.py --config configs/tv3_identifiability.json
```

测试至少证明：解析 / 数值导数在已知声速公式上一致、标签扰动严格闭包、边界差分被标记、重复观测未被双重计入、奇异协方差失败可见、未表示 nuisance 阻止最终 go verdict、正式输出不能覆盖。

### 3. 文档回填规则

在 `audit_failed`、`inconclusive_missing_business_threshold` 或任何单项敏感度结果阶段，不更新项目能力结论。只有正式 `verdict.json` 产生后，才更新项目记忆库中的物理上限、当前执行路线和风险控制；S-Y 与 S-L 的既有模型结论仍保持分列。
