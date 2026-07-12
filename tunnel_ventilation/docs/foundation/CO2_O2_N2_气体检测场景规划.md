# 掘进通风 CO2 / O2 / N2 气体检测场景规划

## 目标

建立一个独立于掺氢天然气与合成气主线的掘进通风气体检测场景，用于模拟并评估 `CO2 / O2 / N2` 三组分浓度估计能力。

该场景的仿真链路复用 [../dl_model_architecture.md](../../../hydrogen_ng/docs/dl_model_architecture.md) 第十三节：沿用 slow 变量、超声波形、光纤麦克风波形、phase schedule、打包与校验流程，只替换组分种类、组分范围和场景命名。

该场景的核心问题不是识别复杂燃料气组成，而是在空气主背景下捕捉通风稀释、人员与设备扰动、局部积聚等因素导致的 `CO2`、`O2`、`N2` 组成变化。`N2` 是显式预测输出，不再作为闭包残差补全项。

## 场景定义

| 项目                | 设定                                      |
| ----------------- | --------------------------------------- |
| 场景名称              | 掘进通风三组分检测                               |
| 建议数据集前缀           | `tv3-*`                                 |
| 建议 schema_version | `tunnel-ventilation-1`                  |
| 组分                | `x_CO2`、`x_O2`、`x_N2`                   |
| 数据一致性             | 组分总量按气体组成数据校验，训练输出不做闭包残差预测              |
| 预测目标              | `x_CO2`、`x_O2`、`x_N2` 三组分均作为 label 直接输出 |
| 主要任务              | 回归三组分浓度；按通风状态做分层评估                      |
| 非目标范围             | 暂不加入 `CH4`、`CO`、粉尘、湿度报警逻辑或法规阈值判定        |

## 组分语义

1. `x_CO2`：通风稀释效果和人员 / 设备排放扰动的主要可观测指标。
2. `x_O2`：空气新鲜度与缺氧风险趋势指标。
3. `x_N2`：空气主背景气体，作为显式 label 与模型输出直接监督，不由 `CO2` 与 `O2` 的预测残差补全。

与 `syngas` 场景不同，本场景的 `N2` 不是背景字段，也不应从 labels 中移除。

与旧闭包头不同，本场景不采用 `GasHeadNormalize` 或类似残差补全头；`sum_abs_error` 只作为输出一致性监控。

## 采样设计

### 初始区间

以下区间用于仿真数据规划，不直接作为现场安全报警阈值。

| 变量                 | 建议范围              | 用途                       |
| ------------------ | -----------------:| ------------------------ |
| `x_CO2`            | 0.03-5.00 %       | 覆盖新鲜空气、通风不足与局部积聚         |
| `x_O2`             | 18.00-21.20 %     | 覆盖正常空气到低氧扰动              |
| `x_N2`             | 73.80-81.97 %     | 空气主背景；作为直接 label 输出      |
| `T_C`              | 沿用现有链路范围，必要时场景化收窄 | 环境温度扰动                   |
| `P_MPa`            | 沿用现有链路范围，必要时场景化收窄 | 压力变化                     |
| `H_RH`             | 沿用现有链路范围，必要时场景化收窄 | 湿度对声学、热导与光学通道的影响         |
| `L_m` / `path_lms` | 沿用 v6 物理严格化链路     | 与 200 kHz 超声传播和多光程扫描保持一致 |

### 采样规则

1. 复用现有 LHS / random 采样入口，仅替换 `COMPONENT_FIELDS` 与组分范围。
2. `x_N2` 写入 condition grid、labels 和 manifest，不作为 `background_fields`。
3. 数据生成阶段可保留组分总量一致性校验；模型输出阶段不使用闭包残差补全。
4. 保留状态分层标签，但不把状态标签作为模型输入的真值捷径。

### 建议状态分层

| 状态                  | 判定依据                    | 目的          |
| ------------------- | ----------------------- | ----------- |
| `fresh_air`         | 低 `CO2`、接近常规空气的 `O2`    | 检查正常区间精度    |
| `ventilation_decay` | `CO2` 上升且 `O2` 下降       | 检查通风变差趋势    |
| `co2_accumulation`  | `CO2` 高值但 `O2` 未必同步大幅下降 | 检查 CO2 通道贡献 |
| `oxygen_depletion`  | `O2` 低值                 | 检查 O2 可辨识性  |

具体边界应写入配置文件，避免散落在生成、训练和评估代码中。

## 仿真链路复用

本场景第一阶段不新增硬件链路，直接复用 `dl_model_architecture.md` 第十三节中的仿真框架。

| 模块                     | 复用方式                                        | 本场景仅变动                                                  |
| ---------------------- | ------------------------------------------- | ------------------------------------------------------- |
| slow 变量                | 复用现有 8 slow 通道组织方式                          | 气体组分输入改为 `CO2 / O2 / N2`                                |
| ultrasonic 波形          | 复用 200 kHz、1 MS/s、20-bit ADC、5000 点窗口       | 声速与衰减物理参数换成三组分空气背景；tv3 默认 int16 + per-timestep scale 存储 |
| fiber_mic 波形           | 复用 10000 点窗口与反射 / 相位解调链路                    | 气体物性输入换成三组分空气背景；tv3 默认 `--skip-fiber-mic` 跳过生成（代码保留）    |
| phase schedule         | 复用 baseline、exposure、steady、recovery 四阶段    | 状态语义映射为通风扰动                                             |
| packaging / validation | 复用 arrays、manifest、scalers、splits、integrity | labels 改为三列；N2 不进 background_fields                     |
| HITRAN / empirical 光学  | 复用现有后端开关                                    | 第一阶段只保留 CO2 相关吸收差异，O2 / N2 不新增光学专用通道                    |

第一阶段不规划 `V_paramagnetic_O2`、风速通道或新传感器。如果后续要加入 O2 专用硬件，应作为新阶段单独设计。

## 数据契约

### labels

```text
label_names = ["x_CO2", "x_O2", "x_N2"]
composition_scheme = "tunnel_ventilation"
schema_version = "tunnel-ventilation-1"
```

### manifest 必备字段

```json
{
  "schema_version": "tunnel-ventilation-1",
  "composition_scheme": "tunnel_ventilation",
  "labels": ["x_CO2", "x_O2", "x_N2"],
  "background_fields": [],
  "slow_channels": "reuse_v6_default",
  "track_sum_abs_error": true
}
```

模型输出契约：tv3 多模态 `cnn1d_tcn_fusion` 使用 `output_mode="raw3"`、`out_dim=3`，三组分直接线性输出；`gas_head` 与 `target_transform` 在 tv3 路径下显式拒绝。

### 命名建议

| 类型        | 建议命名                                             |
| --------- | ------------------------------------------------ |
| smoke 数据集 | `tv3-smoke`                                      |
| 正式数据集     | `tv3-formal`                                     |
| 配置目录      | `configs/`                                       |
| 生成入口      | `pipeline.generate_tunnel_ventilation_benchmark` |
| schema 文件 | `tv3/sim/core/tunnel_ventilation_schema.py`      |
| 生成子包      | `tv3/sim/generation/tunnel_ventilation/`         |

## 实施路线

> 实施状态（2026-07-08）：阶段 1–3 仿真链路 + 阶段 4 DL/ML 适配 + tv3-formal（600 序列）+ Ridge/TCN 首轮基线已落地；固定特征回归分支阶段 A/B（`physics_stats / MiniRocket + RidgeCV`，R0/R1a/R1b）已完成，D0 oracle/observed/tof_only/slow_only 四组特征拆分实验已在本地 600 序列完成并可视化分析（见 `outputs/tv3_d0_local/d0_analysis.png` 与 [掘进通风项目记忆库.md](../掘进通风项目记忆库.md)）。完整 15 runs 基线矩阵、服务器端 `tv3-formal-6000` 重跑与后续 D1/D2 物理引导深度学习方案待继续执行。详见 [adaptation_plan.md](adaptation_plan.md) §实施进度、[rocket_hydra_regression_implementation_plan.md](../archive/completed/rocket_hydra_regression_implementation_plan.md) 与 [三组分检测深度学习新框架方案.md](../archive/legacy/三组分检测深度学习新框架方案.md)。

### 阶段 1：契约与采样 ✅

- 新增 `tunnel_ventilation` 专用 schema。
- 复用现有采样入口，只替换 `COMPONENT_FIELDS = ("x_CO2", "x_O2", "x_N2")` 与组分范围。
- 生成 `tv3-smoke`，验证 labels、condition grid、manifest 和 split。
- 单元测试覆盖组分总量一致性、字段顺序、非法样本拒绝和 seed 可复现。

### 阶段 2：仿真链路复用 ✅

- 复用 v6 声速、衰减、热导、波形和 slow 动力学链路。
- 仅补齐 `CO2 / O2 / N2` 三组分所需的纯组分物性常数与范围映射。
- 验证 slow、ultrasonic、fiber_mic shape 与现有链路一致。

### 阶段 3：benchmark 闭环 ✅

- 新增 `tv3-smoke` CLI。
- 打包 `labels/y.npy`、`sequence_labels.csv`、`manifest.json`、splits 和 scalers。
- 输出 per-component 数据统计与 `sum_abs_error` 数据一致性检查。

### 阶段 4：算法框架 ✅（DL/ML 适配完成，首轮基线已执行）

训练适配已完成：losses/trainer 与 ML baseline 均按 `composition_scheme="tunnel_ventilation"` 处理 tv3，conditional metrics 使用 `o2_bins/co2_bins`，闭包类 loss、`target_transform` 和 `gas_head` 在 tv3 下拒绝。基线训练 + ablation 见 [dl_training_plan.md](../archive/legacy/dl_training_plan.md) §10 推荐执行顺序 P-2 ~ P-9。

## 验收标准

### 数据验收 ✅ 已通过

- `labels/y.npy` shape 为 `(N, 3)` — ✅ tv3-smoke/tv3-formal 均通过
- `sequence_labels.csv` 包含 `sequence_id` 与 `x_CO2 / x_O2 / x_N2` — ✅
- 每条样本保留组分总量一致性检查 — ✅ `|sum-100|<1e-6`
- `manifest.json` 中 `composition_scheme == "tunnel_ventilation"` — ✅
- `background_fields == []` — ✅

### 输出契约验收 ✅ 已通过

- 指标按 `x_CO2`、`x_O2`、`x_N2` 分别输出 — ✅ component_metrics 含 3 组分
- `sum_abs_error` 可计算，但不能通过闭包残差头强行保证 — ✅ Ridge sum_abs_error≈0（线性模型天然闭包），TCN sum_abs_error≈58（未收敛）；模型层无闭包残差头
- 状态分层指标至少覆盖 `fresh_air`、`ventilation_decay`、`co2_accumulation`、`oxygen_depletion` — ⏳ 待实施（当前 conditional_metrics 按 o2_bins/co2_bins 分箱，状态分层评估待阶段 Ⅱ）

### 回归验收 ✅ 已通过

- 既有 `wv4-*` 与 `sg4-*` 测试不因新场景改变默认行为 — ✅ 全量 548 passed，0 failed
- 新增代码不得修改全局 `COMPONENT_FIELDS` 的含义 — ✅ hg schema 不变
- 新场景不得把 `x_N2` 写成 syngas 的 `background_fields` — ✅ tv3 `BACKGROUND_FIELDS=()`

### 回归性能验收 🔶 首轮结果（slow-only，600 序列）

| 组分  | 最低标准              | Ridge (val)        | TCN (val) | 状态                  |
| --- | ----------------- |:------------------:|:---------:| ------------------- |
| CO₂ | R²≥0.95, MAE≤0.30 | R²=0.91, MAE=0.34  | R²=-0.05  | Ridge 接近达标，DL 数据量不足 |
| O₂  | R²≥0.70, MAE≤0.50 | R²=-0.05, MAE=0.81 | R²=-0.14  | ❌ 物理可辨识性不足，触发停止条件   |
| N₂  | R²≥0.80, MAE≤0.80 | R²=0.65, MAE=0.92  | R²=-0.53  | ❌ Ridge 接近但未达标      |

详见 [experiment_roadmap.md](../archive/legacy/experiment_roadmap.md) 基线结果分析。

## 主要风险

1. **N2 直接可辨识性弱**：N2 在空气中占比高，直接监督不等于容易学习。
2. **O2 可观测性风险**：第一阶段不新增 O2 专用通道，需在后续算法阶段再评估可辨识性。
3. **过度改造风险**：本阶段目标是复用仿真链路，避免提前引入新硬件通道、新 loss 或新模型结构。
4. **安全阈值误用**：本文档的区间用于实验覆盖，不应直接转写为现场报警阈值。

## 最小闭环建议

```powershell
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark `
    --output-root data --dataset tv3-smoke --sequences 32 --seed 20260704 `
    --timesteps 32 --dt-s 0.5 --optical-absorption-backend empirical_v1 --workers 1

python -m pytest tests/test_tunnel_ventilation_*.py -v
```

上述命令用于 smoke 链路回归。formal 数据与首轮 Ridge/TCN 基线已可用；完整 15 runs 由 `scripts/run_tv3_baseline.py` 编排，固定 seeds 为 `42,123,456`，DL 非零退出码按失败记录，即使存在 `metrics.json` 也不纳入成功 summary。
