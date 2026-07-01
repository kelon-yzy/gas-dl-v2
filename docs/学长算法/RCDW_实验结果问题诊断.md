# RCDW 实验结果问题诊断

## 0. 文档定位

- **日期**：2026-07-01
- **范围**：`rcdw_mgda` Phase 6B/6D/6E 结果复查，重点分析 128-seq HITRAN smoke、scaler-on、Phase 6E `pressure_drift` 扰动实验。
- **输入证据**：
  - 数据集：`rcdw_mgda/data/rcdw-hitran-smoke-128`
  - 模型：`rcdw_mgda/runs/stage_b/rcdw.pt`
  - 配置：`rcdw_mgda/configs/phase6e-hitran-smoke-128-pressure.yaml`
  - 扰动图：`rcdw_mgda/runs/phase6e_pressure/perturb/`
  - 代码入口：`rcdw_mgda/scripts/perturb.py`

本文档记录当前实验暴露的问题，不替代 `RCDW_实施完成情况.md` 和 `RCDW_数据集主线对齐_Phase6路线.md` 的进度记录。

---

## 1. 关键结论

当前 Phase 6E 已完成实验闭环，但实验结果的主要问题不是“未跑完”，而是**评估语义和可靠性机制还不足以支撑强结论**：

1. 扰动发生在 `BenchmarkDataset` 输出之后，因此 scaler-on 情况下扰动 level 作用在**标准化输入空间**，不是原始物理量纲。
2. `hard_suppress` 在 clean baseline 上已经大量触发，并让整体指标变差。
3. `ErrorNet` 的预测误差与真实单模态误差相关性很弱，可靠性估计尚未校准。
4. 目前只有 `temperature` 扰动产生强响应；`pressure_drift` 没有形成可观测退化。
5. O2/CO2 极低真值会放大 MRE/MaxRE，当前正式结论应优先看 MAE/RMSE。
6. 12 维 input scaler 的 passthrough 文档语义与实际行为需要再对齐。

---

## 2. 当前结果摘要

### 2.1 标签分布

128-seq HITRAN smoke 的 test split 共有 20 条 sequence、500 个窗口。低浓度样本会显著影响相对误差：

| 气体 | mean | std | min | p05 | p50 | max |
|------|-----:|----:|----:|----:|----:|----:|
| O2 | 0.11553 | 0.06406 | 0.00092336 | 0.03802 | 0.09518 | 0.23872 |
| CO2 | 0.09246 | 0.04946 | 0.00008927 | 0.00223 | 0.10822 | 0.17681 |
| N2 | 0.79201 | 0.07994 | 0.62237 | 0.65915 | 0.79499 | 0.91145 |

结论：O2 与 CO2 允许接近 0 的测试样本，`MRE=e/ref` 在这些样本上会爆炸。报告质量时应以 MAE/RMSE 为主，MRE 需要阈值过滤或浓度分箱。

### 2.2 Clean baseline：raw fusion vs hard_suppress

| 输出 | overall MAE | overall RMSE | 备注 |
|------|------------:|-------------:|------|
| raw fusion | 0.05648 | 0.06742 | `model(x)["C"]` |
| hard_suppress fusion | 0.05803 | 0.06872 | `scripts.perturb` 当前使用路径 |

`hard_suppress` 在 clean test 上使指标变差，说明当前 hard suppression 不能直接作为默认正式指标。

### 2.3 单模态表现

| 模态 | overall MAE | overall RMSE | O2 MAE | CO2 MAE | N2 MAE |
|------|------------:|-------------:|-------:|--------:|-------:|
| NDIR | 0.05769 | 0.06897 | 0.06170 | 0.04805 | 0.06331 |
| TCD | 0.06058 | 0.07302 | 0.05739 | 0.05588 | 0.06847 |
| US | 0.07833 | 0.09293 | 0.08081 | 0.09246 | 0.06174 |

观察：

- NDIR 是当前总体最强单模态，尤其 CO2 优于其他模态。
- US 对 N2 最好，但整体最差。
- 融合只比 NDIR 略好或相近，说明 RCDW 融合增益还没有拉开。

---

## 3. 当前问题

### P0-1：扰动 level 与物理量纲不一致

`scripts.perturb` 先用 `BenchmarkDataset(..., apply_input_scaler=None)` 取数据，再调用 `inject(X_test, kind, level)`。在当前数据集 manifest 中 `input_normalization.applied=true`，所以 `X_test` 已经是标准化输入。

因此 `inject.py` 中的扰动不是原始物理量纲扰动：

| 扰动 | 当前实现 | scaler-on 实际含义 |
|------|----------|-------------------|
| `temperature` | `T_C += level * 80.0` | `level=0.11` 为 `+8.8` 个标准差，约 `+49.9°C` |
| `pressure_drift` | `P_MPa += level * 2.0` | `level=0.11` 为 `+0.22` 个标准差，约 `+0.0385 MPa` |
| `optical_atten` | `V_NDIR_CO2 *= 1-level` | 对标准化值做乘性衰减，物理上不等同于原始电压衰减 |
| `thermal` | `V_TCS *= 1+level*randn` | 对标准化值做乘性噪声，物理含义弱 |

关键数据：

| 通道 | mean | std |
|------|-----:|----:|
| `T_C` | 26.07442 | 5.66611 |
| `P_MPa` | 0.40119 | 0.17519 |
| `V_NDIR_CO2` | 1.30688 | 0.67043 |
| `V_TCS` | 1.12323 | 0.03824 |

影响：Phase 6E 的扰动曲线只能解释为“标准化输入空间敏感性”，不能直接解释为物理传感器退化或真实环境漂移。

建议：

1. 增加 `perturb_space: standardized|physical` 配置。
2. physical 模式下先对原始物理量纲注入扰动，再应用 input scaler。
3. `temperature` 与 `pressure_drift` 至少补一版物理单位实验。

### P0-2：hard_suppress 在 clean 数据上过度触发

clean test 诊断：

| 气体 | 模态 | baseline W | hard_suppress 后 W | degraded rate |
|------|------|-----------:|-------------------:|--------------:|
| O2 | NDIR | 0.1852 | 0.2261 | 0.414 |
| O2 | TCD | 0.4358 | 0.4855 | 0.238 |
| O2 | US | 0.3789 | 0.2885 | 0.482 |
| CO2 | NDIR | 0.5250 | 0.4865 | 0.218 |
| CO2 | TCD | 0.2509 | 0.2994 | 0.168 |
| CO2 | US | 0.2241 | 0.2141 | 0.294 |
| N2 | NDIR | 0.1961 | 0.2422 | 0.368 |
| N2 | TCD | 0.4135 | 0.4831 | 0.202 |
| N2 | US | 0.3903 | 0.2747 | 0.502 |

clean 样本中任意模态-气体对触发 degraded 的比例为 0.666。当前 `degraded=True` 在每个扰动 level 都出现，因此不能作为“扰动导致退化”的证据。

建议：

1. `scripts.perturb` 同时报告 raw fusion 与 hard_suppress fusion。
2. hard suppression 的门槛需要先在 clean validation 上校准，例如目标 clean 触发率低于 5% 或 10%。
3. 在报告中增加 `degraded_rate`，不要只打印 `degraded.any()`。

### P0-3：ErrorNet 尚未形成可靠排序

clean test 上，`E_pred` 与真实单模态绝对误差 `|Y_modal-C_ref|` 的相关性如下：

| 模态-气体 | actual mean | E_pred mean | corr |
|-----------|------------:|------------:|-----:|
| NDIR-O2 | 0.06170 | 0.05162 | 0.000 |
| TCD-O2 | 0.05739 | 0.03357 | 0.121 |
| US-O2 | 0.08081 | 0.05432 | 0.047 |
| NDIR-CO2 | 0.04805 | 0.05781 | -0.012 |
| TCD-CO2 | 0.05588 | 0.04657 | 0.065 |
| US-CO2 | 0.09246 | 0.06460 | -0.010 |
| NDIR-N2 | 0.06331 | 0.06078 | -0.010 |
| TCD-N2 | 0.06847 | 0.04156 | -0.365 |
| US-N2 | 0.06174 | 0.07106 | 0.082 |

问题：`E_pred` 的排序对融合权重和 hard_suppress 至关重要，但当前相关性接近 0 或为负。

可能原因：

- Stage B 只有 20 epoch smoke 训练，数据量与扰动覆盖都不足。
- ErrorNet 的输入特征主要来自 clean 滑窗统计，缺少真实扰动标签或 domain shift 监督。
- `lambda_error=1.0` 未必足以让误差预测稳定；融合损失可能主导了可用但不校准的权重。

建议：

1. 训练后自动输出 ErrorNet calibration 表，包括 Pearson/Spearman、排序准确率、ECE-like 分箱。
2. 在 Stage B 加入 validation calibration early stopping 或额外指标。
3. 用扰动增强样本训练 ErrorNet，或先关闭 hard_suppress，仅观察 soft fusion。

### P1-1：只有 temperature 产生强响应

固定 seed 重复诊断（5 次）显示：

| 扰动 | level=0 HS MAE | level=0.11 HS MAE | level=0.11 raw MAE | 结论 |
|------|---------------:|------------------:|-------------------:|------|
| optical_atten | 0.05803 | 0.05804 | 0.05651 | 基本不变 |
| optical_scat | 0.05803 | 0.05819 | 0.05654 | 基本不变 |
| thermal | 0.05803 | 0.05764 | 0.05656 | 轻微改善 |
| ultrasonic | 0.05803 | 0.05813 | 0.05648 | 基本不变 |
| temperature | 0.05803 | 0.11293 | 0.10471 | 强退化 |
| pressure_drift | 0.05803 | 0.05756 | 0.05611 | 轻微改善 |

解释：

- 当前结果强烈说明模型对标准化温度通道敏感。
- 不能说明模型对 optical/thermal/ultrasonic/pressure 物理退化鲁棒，因为这些扰动并未重生成物理链路。
- `pressure_drift` 当前只移动 `P_MPa` 输入，不重算声速、吸收、慢响应等压力派生观测，因此未退化是预期内结果。

### P1-2：MRE/MaxRE 不适合作为当前主指标

`metrics.py` 中 MRE 定义为 `abs(pred-ref)/(abs(ref)+1e-8)`。当 O2/CO2 真值接近 0 时，MRE/MaxRE 会被极小分母放大。

当前 128-seq test 中 CO2 最小值为 0.00008927，因此 MaxRE 出现几万到十万级不代表模型整体失效。

建议：

1. 总表以 MAE/RMSE 为主。
2. MRE 增加 `ref >= threshold` 版本，例如 `MRE@ref>=0.01`。
3. 对 O2/CO2 做浓度分箱：`[0,1%)`、`[1%,5%)`、`[5%,15%)`、`[15%,25%]`。

### P1-3：input scaler passthrough 策略需要重新对齐

代码注释中 `DEFAULT_PASSTHROUGH_CHANNELS` 包含：

- `ultrasonic_peak_index`
- `ultrasonic_tof_quality`
- `ultrasonic_tof_accepted`

但当前 12 维 `input_scaler.json` 实际只有：

- `ultrasonic_tof_accepted`: `passthrough`

而：

- `ultrasonic_peak_index`: `z_score`
- `ultrasonic_tof_quality`: `z_score`

这可能是后续 H1 修复中的有意收缩，但文档和命名容易让人误判“所有离散/质量通道都 passthrough”。

建议：

1. 明确 12 维模型输入 scaler 的策略：是否只 passthrough `tof_accepted`。
2. 若 `peak_index/tof_quality` 保持 z-score，更新 docstring、manifest 说明和测试命名。
3. 若要保留原语义，则把 `INPUT_PASSTHROUGH_CHANNELS` 扩展回三个通道，并重生成数据和重训。

---

## 4. 建议修复顺序

| 优先级 | 任务 | 目的 | 验收 |
|--------|------|------|------|
| P0 | 增加 physical perturb 路径 | 修正扰动物理语义 | `temperature` 以 °C、`pressure_drift` 以 MPa 注入后再 scaler |
| P0 | perturb 同时报 raw 与 hard_suppress | 分离模型输出与抑制规则影响 | 图表和控制台均有 raw/HS 两套指标 |
| P0 | ErrorNet calibration 报告 | 判断可靠性估计是否可用 | 输出相关性、排序准确率、分箱误差 |
| P1 | hard_suppress 阈值校准 | 避免 clean 数据大面积误触发 | clean degraded rate 低于约 5%-10% |
| P1 | MRE 阈值化或分箱 | 避免极低真值污染结论 | 报告 `MRE@ref>=0.01` 或分箱 MRE |
| P1 | input scaler passthrough 决策 | 对齐文档、manifest、实际 scaler | 测试覆盖 `peak_index/tof_quality` 策略 |

---

## 5. 当前可引用的保守表述

可以写入阶段报告的结论：

> 在 128-seq HITRAN smoke、scaler-on 设置下，RCDW-MGDA 的 raw fusion clean test overall MAE/RMSE 为 0.05648/0.06742；当前 perturb 脚本采用 hard_suppress 后为 0.05803/0.06872。Phase 6E 的六类标准化输入空间扰动中，temperature 扰动产生最明显退化，level=0.11 时 hard_suppress MAE/RMSE 上升至 0.11293/0.14629；pressure_drift 未造成可观测退化，level=0.11 时为 0.05756/0.06780。由于扰动注入发生在 input scaler 之后，当前曲线应解释为标准化输入敏感性，而非严格物理量纲退化实验。

不建议写入阶段报告的强结论：

- “模型已证明对压力漂移鲁棒。”
- “temperature level=0.11 等于 +8.8°C。”
- “`degraded=True` 证明所有扰动都触发退化。”
- “MRE/MaxRE 显示模型完全失效。”

---

## 6. 下一步建议命令

先增加诊断脚本或扩展 `scripts.perturb`，不要继续只生成 PNG：

```bash
cd rcdw_mgda
.\.venv\Scripts\python.exe -m scripts.perturb \
  --ckpt runs/stage_b/rcdw.pt \
  --config configs/phase6e-hitran-smoke-128-pressure.yaml \
  --output-dir runs/phase6e_pressure/perturb \
  --report-json runs/phase6e_pressure/perturb/metrics.json
```

建议新增 JSON 字段：

- raw fusion per-gas metrics
- hard_suppress fusion per-gas metrics
- degraded rate per modality/gas
- `E_pred` calibration metrics
- perturb space：`standardized` 或 `physical`
- physical delta metadata：例如 `temperature_delta_C`、`pressure_delta_MPa`

---

## 7. 修复计划

### 7.1 总原则

先修“实验可解释性”，再修模型可靠性，最后调性能。当前不应直接靠增加训练轮数或调 `ErrorNet` 来掩盖评估口径问题。

### 7.2 Phase R1：诊断报告化（当前开始执行）

目标：让每次 perturb 都能回答“到底是谁让指标变了”。

实施项：

1. `scripts.perturb` 增加 `--report-json`。
2. 同时报 raw fusion 与 hard_suppress fusion 两套指标。
3. 输出 `degraded_rate`，按 gas 与 modality 展开，不再只打印 `degraded=True/False`。
4. 输出 `E_pred` 校准表，包括 Pearson、best-modality 排序准确率、真实误差均值和预测误差均值。
5. 在 JSON 中记录 `apply_input_scaler`、`perturbation.space`、dataset、ckpt、levels、kinds 等复现实验元数据。

验收：

- `scripts.perturb --report-json <path>` 能生成结构化 JSON。
- JSON 中每个扰动 level 同时包含 `raw` 与 `hard_suppress` 指标。
- JSON 中有 clean baseline 的 ErrorNet calibration 和 degraded rate。
- 既有 PNG 输出不破坏。

### 7.3 Phase R2：扰动物理量纲修复

目标：把“标准化输入空间敏感性”和“物理量纲扰动”分开。

实施项：

1. 增加 `perturbation.space: standardized|physical`。
2. `standardized` 保留当前行为，用于历史兼容。
3. `physical` 模式下，Dataset 以 `apply_input_scaler=False` 读取原始量纲，扰动后再应用 `input_scaler.json`。
4. `temperature` 用 °C 增量；`pressure_drift` 用 MPa 增量。
5. JSON 报告记录 physical delta metadata。

验收：

- `temperature level=0.11` 的含义能明确追溯到 °C，而不是标准差。
- `pressure_drift level=0.11` 的含义能明确追溯到 MPa。
- 标准化扰动与物理扰动分别出图和出 JSON。

### 7.4 Phase R3：hard_suppress 阈值校准

目标：降低 clean 数据上的误触发。

实施项：

1. 增加一个阈值扫描脚本或配置实验，扫描 `ratio` 与 `cap`。
2. 在 validation clean split 上统计 degraded rate 与 MAE/RMSE。
3. 必要时引入绝对 margin，例如 `E_pred > max(ratio * min_E, min_E + margin)`。

验收：

- clean validation 的 degraded rate 降到约 5%-10%。
- hard_suppress 后 clean 指标不应明显差于 raw fusion。

### 7.5 Phase R4：ErrorNet 校准修复

目标：让 `E_pred` 能代表单模态误差排序。

实施项：

1. 每次 Stage B 后输出 ErrorNet calibration report。
2. 如果校准仍差，再评估增加 Stage B epoch、调大 `lambda_error`、对误差目标做归一化或增加 ranking loss。
3. 可选：用扰动增强样本训练 ErrorNet。

验收：

- `E_pred` 与真实误差相关性明显为正。
- best-modality 排序准确率高于随机基线。
- hard_suppress 基于校准后的 `E_pred` 才进入正式结果。

### 7.6 Phase R5：指标与 scaler 契约清理

目标：消除报告指标和 scaler 文档语义歧义。

实施项：

1. 保留原始 MRE，但正式报告主用 MAE/RMSE 与 `MRE@ref>=0.01`。
2. 明确 input scaler 是否只 passthrough `ultrasonic_tof_accepted`。
3. 若 `peak_index/tof_quality` 保持 z-score，则更新 docstring、manifest 说明和测试命名。

验收：

- 文档、manifest、input scaler JSON 与测试对 passthrough 策略一致。
- 报告不再用极低真值样本的 MRE/MaxRE 作为主结论。
