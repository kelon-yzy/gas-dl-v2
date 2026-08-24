# 掘进通风实验路线图

> 本文档规划掘进通风场景（CO₂/O₂/N₂）的正式实验路线，从 benchmark 生成到 DL 基线训练到 ablation 消融实验。
> 仿真链路基于 v6-phys-strict（200 kHz 超声、1 MS/s 采样、20-bit ADC、L_m 0.2–0.3 m）。
> 仿真链路适配方案见 [adaptation_plan.md](../../foundation/adaptation_plan.md)。

## 当前状态

```
pipeline.generate_tunnel_ventilation_benchmark  →  data/tv3-smoke/   →  链路验证 ✅
                                                →  data/tv3-formal/  →  dl.cli + tv3_*.json  →  metrics
```

| 已完成                                         | 说明                                                                               |
| ------------------------------------------- | -------------------------------------------------------------------------------- |
| 仿真链路适配（阶段 1–3）                              | schema/物理/慢通道/benchmark/CLI 全部落地，tv3-smoke 生成通过                                  |
| DL 训练适配（阶段 4）                               | losses/trainer 适配 tv3 scheme，CLI 端到端训练通过                                         |
| tv3-smoke 生成                                | 32 序列链路验证通过                                                                      |
| tv3-formal 生成                               | 600 序列 512 时步生成通过（int16 + skip-fiber-mic，3 GB；详见 Ⅰ-2）                            |
| 存储优化（2026-07-05）                            | int16 + per-timestep scale + `--skip-fiber-mic`；数据集 17→3 GB，误差/噪声 ≈ 1%           |
| DL 配置矩阵                                     | 5 个基线配置 + `tv3_tcn_multimodal.json` + 编排脚本已创建；fusion 配置使用 `raw3` 三输出             |
| CNN1D 1 epoch 验证                            | 管线正常，3 组分 metrics + o2_bins/co2_bins 分箱 + sum_abs_error 可计算                      |
| TCN 50 epochs（seed=42）                      | val R²≈0（CO₂=-0.05, O₂=-0.14, N₂=-0.53），600 序列对 DL 不够                            |
| Ridge 基线                                    | val: CO₂ R²=0.91 ✅, O₂ R²=-0.05 ❌, N₂ R²=0.65 ❌（见下方分析）                           |
| Rocket 阶段 A（2026-07-06 落地，2026-07-07 R0 回填） | `physics_stats + RidgeCV` 链路落地；R0 正式集（6000 序列）val O₂ R²=0.603、CO₂=0.993、N₂=0.925 |
| D0 oracle/observed 特征拆分（2026-07-08 clean 6000 完成） | 6 组 Ridge 配置（oracle/observed/tof_only/slow_only/no_tof/no_tcs）已在服务器 tv3-formal-6000（CLEAN）上完成；oracle 膨胀 0.18，o2_bins 物理极限确认；`scripts/check_slow_channels.py` 核查数据集无 V_NDIR_CH4；结论 D2 优先、D1 暂缓，详见 [记忆库 §6.4](掘进通风项目记忆库.md) |

### 初步基线结果分析（2026-07-04）

| 模型              | CO₂ R² (val) | O₂ R² (val) | N₂ R² (val) | 说明                |
| --------------- |:------------:|:-----------:|:-----------:| ----------------- |
| TCN (50 epochs) | -0.05        | -0.14       | -0.53       | 600 序列对 DL 太少，未收敛 |
| Ridge           | 0.91         | -0.05       | 0.65        | CO₂ 达标，O₂/N₂ 未达标  |

**关键发现**：

1. **CO₂ 可辨识性高**（Ridge R²=0.91）：V_NDIR_CO2 直接通道有效，Ridge 能提取信号。
2. **O₂ 可辨识性极低**（Ridge R²≈0）：O₂ 在 slow 通道（声速+热导）信号弱，Ridge 也无法预测。触发 dl_training_plan §10 停止条件（O₂ R²<0.50）。
3. **N₂ 中等可辨识**（Ridge R²=0.65）：通过声学/TCS 间接推断部分有效，但未达标。
4. **DL vs Ridge 差距大**：Ridge CO₂ R²=0.91 vs TCN CO₂ R²≈0，说明 600 序列对 DL 严重不足。
5. **sum_abs_error≈0**（Ridge）：Ridge 线性模型天然满足 sum=100% 闭包（标签闭包），但 O₂ 预测接近均值。

**O₂ 停止条件已触发**：Ridge O₂ R²<0.50，按 dl_training_plan §10 应考虑阶段 Ⅲ-1（O₂ 专用通道）。但当前仅 slow-only 模态 + 600 序列，加入波形模态（ultrasonic+fiber_mic）可能改善 O₂ 辨识（声学通道携带 O₂/N₂ 声速差信号）。

### 方向 B：加入波形模态重跑（训练动力学修复中 🔶）

#### 失效确认与诊断（2026-07-07）

原配置 `configs/tv3_tcn_multimodal.json` 在服务器 6000 序列上跑过，best epoch=1、O₂ R²≈0，DL 失效。结合 R0（同 6000 序列 physics_stats+RidgeCV val O₂ R²=0.603），排除"数据量不足"和"数据无信号"，根因是训练动力学或输入尺度问题。

本地 600 序列历史 `metrics_live.jsonl` 诊断佐证：

- slow-only TCN（`tv3_tcn_s42`）：best epoch=20，train 1336→15 正常收敛，val R²=-0.31（600 序列过拟合）。
- fusion（`tv3_tcn_multimodal/s42`）：train 28756→62 持续下降，val_loss epoch 5 后剧烈震荡（43→119→183→80→215），best epoch=10 但 val R²=-98，数值发散。

三个叠加根因：

1. `lr=0.001` 对多模态输入过大（train_loss 首轮暴跌跨过最优区间，val 剧烈震荡）
2. `scaler_path=null`，slow 通道尺度差异大（T_C≈24、H_RH≈50、L_m≈0.24）未标准化，与波形拼接后梯度失衡
3. 波形侧 `dequantize_waveforms=true` + `waveform_adc_scale=5.0` 压到 ±0.2 量级，尺度本身合理

#### v2 修复配置（2026-07-07）

新增 `configs/tv3_tcn_multimodal_v2.json`，保留原配置作失效对照：

| 参数            | 原值                           | v2                                                       | 依据                                             |
| ------------- | ---------------------------- | -------------------------------------------------------- | ---------------------------------------------- |
| `lr`          | 0.001                        | 0.0001                                                   | rocket 方案 §5.3 建议 1e-4；fusion 默认 1e-3 致 val 震荡 |
| `batch_size`  | 32                           | 16                                                       | 降梯度方差，缓解震荡                                     |
| `scaler_path` | null                         | `data/tv3-formal-6000/scalers/scaler_slow_sequence.json` | slow 通道 z-score 标准化                            |
| `modalities`  | slow,ultrasonic,fiber_mic    | slow,ultrasonic                                          | 数据集跳过 fiber_mic                                |
| `eval_splits` | val,test                     | val,test,extrapolation                                   | 与 R0 三 split 对照                                |
| `output_dir`  | `outputs/tv3_tcn_multimodal` | `outputs/tv3_tcn_multimodal_v2`                          | 不覆盖原产物                                         |

本地 smoke（tv3-smoke7，16 序列）已验证 v2 配置端到端跑通：scaler 7 通道维度匹配、train/val/test/extrapolation 四 split 评估正常。

#### 服务器重跑判定标准

- best epoch 不再是 1、val_loss 平稳下降 → 训练动力学修复成功
- val O₂ R² > R0 的 0.603 → DL 路线超过物理特征基线，继续调优
- val O₂ R² ≤ 0.603 但训练正常 → DL 当前架构上限低于 R0，转 R1 MiniRocket
- best epoch 仍为 1 → 配置调整不够，进一步诊断 grad 规模 / loss 权重尺度

| 未完成                      | 阻塞程度                                                                                                                                                                                   |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ~~v2 配置服务器单 seed 验证~~    | **已由 v3_l2 证伪**（见 [dl_training_plan.md §11.4](dl_training_plan.md#114-服务器验证结果tv3-formal-6000-50-epoch-单-seed-rtx-5880)）：三层归一化后 best epoch 不再是 1，但 val R²=+0.019 仍远逊 R0 的 0.918，P-9c 触发 |
| 完整基线训练（15 runs）          | 阻塞 Ⅱ，DL 端到端 raw 波形路线当前架构上限低于 R0，优先推进 R1 固定特征                                                                                                                                           |
| ~~Rocket R1 MiniRocket~~ | **R1a/R1b 已完成（2026-07-08）**：R1b−R1a=−0.71 < 0.05，raw 波形卷积路线证伪。下一步 R5（R0 特征 + 小 MLP）验证线性极限，R3f（fiber_mic）待评估，见 [rocket_hydra §0/§8](../completed/rocket_hydra_regression_implementation_plan.md)     |

### 方向 C：固定特征 Rocket 基线（2026-07-06 已启动）

目的：把“端到端 DL 训练失败”和“现有通道是否真的没有 O₂ / N₂ 信息”拆开验证。

当前已落地内容：

- `tv3/ml/rocket_features.py`：tv3 `physics_stats_v1` 特征缓存，覆盖 `slow + ultrasonic_tof_s + ultrasonic_tof_observed_s + ultrasonic_peak_index + ultrasonic_sound_speed_m_per_s + ultrasonic_sound_speed_estimated_m_per_s + ultrasonic_alpha_true_npm + ultrasonic_tof_quality + ultrasonic_tof_accepted`
- `tv3/ml/rocket_training.py`：`StandardScaler + RidgeCV` 主链路，保留 `ridge_closed_form` 对照
- `tv3/pipeline/run_tv3_rocket_baseline.py`：tv3 专用 CLI，输出 `train/val/test/extrapolation` 指标 JSON
- `configs/tv3_rocket_ridge.json`：R0 默认配置
- `tests/test_rocket_features.py`、`tests/test_tv3_rocket_pipeline.py`：smoke 验证已通过

当前范围（2026-07-08 R1a/R1b 完成后更新）：

- 已实现：`physics_stats`（R0）、`minirocket_scalar`（R1a）、`minirocket_raw`（R1b）
- 已完成：R0 val CO₂ R²=0.993、O₂ R²=0.603、N₂ R²=0.925；R1a val O₂ R²=0.515；R1b val O₂ R²=-0.195
- **R1b − R1a = −0.71 < 0.05**：raw 波形卷积路线证伪，R3/R4/R6 不推进
- 未实现：小 MLP（R5，改为 R0 特征 + MLP，优先）、fiber_mic 增量（R3f，待评估）

建议执行顺序更新为（据 R1a/R1b 实测）：

```text
R0(✅ 0.603) -> R1a(✅ 0.515) -> R1b(✅ -0.195)
  -> R1b - R1a = -0.71 < 0.05: raw 波形卷积路线证伪
  -> 不推进 R3/R4/R6(raw 波形特征)
  -> R5(R0 特征 + 小 MLP)优先: 验证线性极限
  -> R3f(fiber_mic)待评估: 需先确认 fiber_mic 是否与 R1b 同样失效
  -> 若 R5 无增益且 R3f 失效: 接受 R0(0.603)作为现有通道极限
```

详见 [rocket_hydra_regression_implementation_plan.md §0/§8/§9.2](../completed/rocket_hydra_regression_implementation_plan.md)。

### 方向 D：模态链路完整度评估（2026-07-07）

R0/v2 实测后，对各模态仿真链路现状与下一步价值做一次盘点（详见 [dl_training_plan.md §2.5](dl_training_plan.md#25-模态实现现状与实测验证2026-07-07)）。

| 模态         | 物理实现                          | 存储                         | DL 可用 | 对 O₂ 价值                 | 下一步优先级               |
| ---------- | ----------------------------- | -------------------------- | ----- | ----------------------- | -------------------- |
| slow（7 通道） | 完整                            | slow.npy                   | ✅     | 弱（仅 V_TCS 2.3% 热导差）     | 已被 R0 用满             |
| ultrasonic | 完整（200kHz/20bit/Lagrange TOF） | int16 + per-timestep scale | ✅     | 强（声速差 6.4%，R0 top 特征来源） | R1 MiniRocket 主战场    |
| fiber_mic  | 完整（反射+解调）                     | 同上                         | 默认跳过  | 待验证（声压相位或对 O₂/N₂ 有增量）   | R1 不达标时作增量备选         |
| NDIR 光学    | empirical 完整，HITRAN 禁用        | 并入 slow                    | ✅     | 无（O₂/N₂ 无红外吸收）          | HITRAN 后端优先级低，不解决 O₂ |

关键结论：

- R0 已把 slow + ultrasonic 物理统计量用满（val O₂ R²=0.603），剩余 0.10 到 0.70 验收线的缺口在 raw 波形未利用。
- v2 DL fusion 证明 raw 波形直接进网络有尺度问题，R1 MiniRocket 用固定核 + 池化绕开此问题，是当前最合理的增量方向。
- fiber_mic 是 R1 不达标时的备选，需重新生成数据集（去掉 `--skip-fiber-mic`）。
- HITRAN 光学后端（原 §9 Ⅲ-2）对 O₂ 无帮助，优先级下调。

---

## 阶段 Ⅰ — 正式 benchmark + 基线实验

### Ⅰ-1 tv3-smoke 生成 ✅ 已完成

目的：端到端链路验证（32 序列、32 时步）。

```powershell
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark `
    --output-root data --dataset tv3-smoke --sequences 32 --seed 20260704 `
    --timesteps 32 --dt-s 0.5 --optical-absorption-backend empirical_v1 --workers 1
```

验证清单：

| 检查项                           | 预期值                              |
| ----------------------------- | -------------------------------- |
| `labels/y.npy` shape          | `(N, 3)`                         |
| `metadata/label_names.npy`    | `["x_CO2", "x_O2", "x_N2"]`      |
| `manifest.composition_scheme` | `"tunnel_ventilation"`           |
| `manifest.background_fields`  | `[]`                             |
| `sequences/slow.npy` 最后一维     | 7                                |
| 组分总量                          | `                                |
| `sequence_labels.csv` 列       | `sequence_id, x_CO2, x_O2, x_N2` |

### Ⅰ-2 tv3-formal 生成 ⚠️ 规模调整 + 存储优化

目的：训练规模数据集，与 hg `wv4-formal-hitran-standard-6000` 时间轴对齐。

> **内存/磁盘双重限制（原始 int32 + fiber_mic）**：计划 6000 序列 × 512 时步受限于两个瓶颈：
> 
> 1. **磁盘**：memmap 需 184 GB，多进程峰值（chunk + memmap）约 368 GB，D 盘剩余 251.8 GB 不足。
> 2. **内存**：`build_sequence_arrays` 在内存中预分配整个 chunk 数组，每 worker chunk 内存 = chunk_size × 512 × 15008 × 4。系统内存 33.6 GB（可用约 25 GB），3000 序列 workers=4（chunk=750）每 worker 22.5 GB，4 worker 并行 90 GB，OOM。
> 
> **2026-07-05 存储优化（方案 B）**：tv3 默认采用 int16 + per-timestep 自适应 scale + 跳过 fiber_mic，数据集大幅压缩：
> 
> - int32 → int16：per-timestep scale 按每时步波形峰值定标，实测峰值占满量程 ~22%，per-timestep 比固定 scale 量化步长小 4.6×；量化误差 max ~1e-5 V，远小于噪声 std 1e-3 V（误差/噪声 ≈ 1%），精度损失可忽略
> - 跳过 fiber_mic（`--skip-fiber-mic`）：光纤代码全部保留，后续去掉开关即可恢复
> - 物理 ADC 仍为 20-bit（`daq_bits=20`），存储 dtype 改为 int16
> 
> 调整为 **600 序列 × 512 时步 workers=4 --skip-fiber-mic**（int16 下每 worker ~0.77 GB，总 ~3 GB）。保持 512 时步与 sg4-formal 时间轴对齐。600 序列 ≈ 400 训练样本，可训练初步基线。

| 规模      | int32 + fiber_mic | int16 + skip-fiber-mic | 减幅   |
| ------- |:-----------------:|:----------------------:|:----:|
| 600 序列  | 17 GB             | 3 GB                   | -82% |
| 6000 序列 | 172 GB            | 29 GB                  | -83% |

```bash
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark \
    --output-root data --dataset tv3-formal --sequences 600 --seed 20260704 \
    --timesteps 512 --dt-s 0.5 --optical-absorption-backend empirical_v1 \
    --storage memmap --workers 4 --skip-fiber-mic
```

预计耗时：1–2 分钟（600 序列 int16 + skip-fiber-mic，workers=4）。

> DL 端通过 `metadata/waveform_spec.json` 自动识别 `waveform_dtype=int16`，加载 `ultrasonic_int16.npy`，dequantize 用 per-timestep `ultrasonic_scale.npy` 还原电压。多模态训练需 `--modalities slow,ultrasonic`（去掉 fiber_mic）。详见 [server_training_guide.md](../../operations/server_training_guide.md)。

### Ⅰ-3 配置矩阵 ✅ 已完成

5 个基线配置和 1 个多模态方向 B 配置在 `configs/` 下：

| 配置文件                      | 模型                 | Loss                   | 特殊参数                                                        |
| ------------------------- | ------------------ | ---------------------- | ----------------------------------------------------------- |
| `tv3_baseline.json`       | CNN1D              | weighted_component_mse | loss_weights: [1.0, 2.0, 1.0]                               |
| `tv3_tcn.json`            | TCN                | weighted_component_mse | loss_weights: [1.0, 2.0, 1.0]                               |
| `tv3_lstm.json`           | LSTM               | weighted_component_mse | loss_weights: [1.0, 2.0, 1.0]                               |
| `tv3_patchtst.json`       | PatchTST           | weighted_component_mse | loss_weights: [1.0, 2.0, 1.0]                               |
| `tv3_ridge.json`          | Ridge              | —                      | repo `RidgeRegressor`（closed-form）                          |
| `tv3_tcn_multimodal.json` | `cnn1d_tcn_fusion` | weighted_component_mse | slow+ultrasonic+fiber_mic；`output_mode="raw3"`, `out_dim=3` |

共享关键参数：

- `slow_input_dim = 7`
- `output_dim = 3`
- `target_timesteps = 512`
- `composition_scheme = "tunnel_ventilation"`
- `dataset_path = "data/tv3-formal"`
- tv3 fusion 输出头：`output_mode="raw3"`；`gas_head` 和 `target_transform` 禁用

### Ⅰ-4 基线训练（5 模型 × 3 seeds = 15 runs）🔶 首轮单 seed 已完成

指标记录：per-component R²、MAE、RMSE（x_CO2、x_O2、x_N2）+ sum_abs_error。

编排脚本：`scripts/run_tv3_baseline.py`（已创建；seeds 固定为 `42,123,456`；DL 非零退出码按失败暴露，即使存在 `metrics.json`）

首轮已完成 TCN seed=42（50 epochs）+ Ridge（closed-form）。完整 15 runs 待决策（见基线结果分析，O₂ 停止条件已触发，需先决定方向 B/D）。

```powershell
python scripts/run_tv3_baseline.py
```

预期观察：

- CO₂ 应有最高 R²（V_NDIR_CO2 直接通道）
- O₂ 可能 R² 最低（无直接光学通道，依赖声学 + TCS 间接推断）
- N₂ R² 中等（占比大但动态范围仅 ~8%）
- 参考 syngas 结果，TCN/PatchTST 可能优于 CNN1D/LSTM

---

## 阶段 Ⅱ — 关键 ablation

基于阶段 Ⅰ 的最优模型开展消融实验。

### Ⅱ-0 TCN Hidden Probe（低成本前置诊断）

> 迁移自 [../DL相位统计稳定提取与保留方案.md](../../../../hydrogen_ng/docs/DL相位统计稳定提取与保留方案.md) 方案 I。在通道消融之前执行，成本 < 0.5 天。

冻结基线最优模型 → 导出 TCN hidden/pooled features → 线性 probe：

| probe 任务                     | 目的                               |
| ---------------------------- | -------------------------------- |
| hidden → y_true (3 组分)       | TCN 是否已提取足够信息                    |
| per-modality hidden → y_true | 定位各模态对 O₂ 的贡献                    |
| hidden → per-component R²    | 对比 final head 的 per-component R² |

判读与分叉：

- probe O₂ R² 高、final O₂ R² 低 → **融合/输出阶段丢信息**，走 Ⅱ-4 模态辅助头 + 平衡融合
- probe O₂ R² 也低 → **前端未提取 O₂ 信号**，走阶段 Ⅲ ROCKET 分支或 O₂ 专用通道

### Ⅱ-1 通道贡献消融

#### Ⅱ-1a 移除 V_TCS

**假设**：O₂ 和 N₂ 精度显著下降。TCS 是区分 O₂/N₂ 的关键补充通道（虽然热导率差异仅约 2%，但提供了声学通道之外的独立信息）。

| 组        | 输入通道         | 预期 CO₂ R² | 预期 O₂ R² | 预期 N₂ R² |
| -------- | ------------ |:---------:|:--------:|:--------:|
| baseline | 全部 7 通道 + 波形 | 高         | 中        | 中        |
| -V_TCS   | 6 通道 + 波形    | 不变        | 下降       | 下降       |

#### Ⅱ-1b ~~移除 V_NDIR_CH4~~

~~假设：无影响。场景中不存在 CH₄，此通道应只含噪声。~~

> **已执行**：V_NDIR_CH4 已从 tv3 schema 中移除（7 通道），无需再作为 ablation 项验证。以下为历史记录。

~~| 组           | 输入通道         | 预期变化         |
~~| ----------- | ------------ | ------------ |
~~| baseline    | 全部 7 通道 + 波形 | —            |
~~| -V_NDIR_CH4 | 7 通道 + 波形    | 各组分 R² 无显著变化 |~~

#### Ⅱ-1c 移除 V_NDIR_CO2

**假设**：CO₂ 精度大幅崩溃。NDIR 是 CO₂ 唯一的直接可观测通道。

| 组           | 输入通道      | 预期 CO₂ R² | 预期 O₂ R² | 预期 N₂ R² |
| ----------- | --------- |:---------:|:--------:|:--------:|
| baseline    | 全部        | 高         | —        | —        |
| -V_NDIR_CO2 | 6 通道 + 波形 | 大幅下降      | 不变或微升    | 不变或微升    |

### Ⅱ-2 O₂ 可辨识性评估

核心科学问题：现有通道能否有效区分 O₂ 和 N₂？

| 组   | 输入            | 目的               |
| --- | ------------- | ---------------- |
| A   | 仅超声波形         | 评估声速差异（~6.4%）的贡献 |
| B   | 超声波形 + V_TCS  | 评估热导额外贡献         |
| C   | 全通道（baseline） | 对照               |
| D   | 仅慢通道（无波形）     | 评估慢通道独立贡献        |

关键判据：

- 如果组 B 的 O₂ R² 显著高于组 A → TCS 对 O₂ 辨识有独立贡献
- 如果所有组 O₂ R² < 0.50 → 需要考虑引入 O₂ 专用传感器（阶段 Ⅲ-1）

### Ⅱ-3 Loss 选择消融

| 组   | Loss                   | 权重                |
| --- | ---------------------- | ----------------- |
| L1  | mse                    | —                 |
| L2  | weighted_component_mse | [1, 1, 1]         |
| L3  | weighted_component_mse | [1, 2, 1]（O₂ 加权）  |
| L4  | weighted_component_mse | [1, 3, 1]（O₂ 强加权） |
| L5  | smooth_l1              | —                 |

关注：O₂ 加权是否改善 O₂ R² 且不损害 CO₂/N₂ 精度。

---

## 阶段 Ⅲ — 扩展（可选，依据阶段 Ⅱ 结果决策）

### Ⅲ-1 O₂ 专用通道评估

触发条件：阶段 Ⅱ-2 所有通道组合下 O₂ R² < 0.70。

方案：模拟顺磁 O₂ 传感器通道 `V_PARAMAGNETIC_O2`，慢通道从 8 → 9。重跑基线评估 O₂ 提升幅度。

### Ⅲ-2 HITRAN 后端适配

将 CO₂ 光学后端从 empirical_v1 升级到 HITRAN line-by-line 计算，提高仿真保真度。

### Ⅲ-3 稳态阶段分析

按 phase schedule 阶段（baseline → exposure → steady → recovery）分析模型性能，映射为通风扰动语义。

### Ⅲ-4 状态分层评估

按四种通风状态（[sampling_design.md](../../foundation/sampling_design.md) 定义）分层评估：

| 状态                | 关注指标            |
| ----------------- | --------------- |
| fresh_air         | 各组分在正常区间的精度     |
| ventilation_decay | CO₂↑ O₂↓ 趋势跟踪能力 |
| co2_accumulation  | CO₂ 高值区精度       |
| oxygen_depletion  | O₂ 低值区精度（安全关键）  |

### Ⅲ-5 模态辅助头 + 平衡融合（Ⅱ-0 probe 触发）

> 迁移自 [../DL相位统计稳定提取与保留方案.md](../../../../hydrogen_ng/docs/DL相位统计稳定提取与保留方案.md) 方案 D/E。
> 触发条件：Ⅱ-0 probe 显示 O₂ 信息在融合阶段丢失。

- 模态级辅助头：slow / ultrasonic / fiber_mic 各设独立预测头
- 平衡融合：等维投影 + LayerNorm + gated fusion
- Modality Dropout：训练时随机丢整个模态分支（p=0.1–0.2）
- 详见 [dl_training_plan.md §9.2 T2/T3](dl_training_plan.md#92-直接迁移的技术阶段-ⅱ-后可用)

### Ⅲ-6 ROCKET 统计池化分支（Ⅱ-0 probe 触发；阶段 A 已先实现 physics_stats）

> 迁移自 [../DL相位统计稳定提取与保留方案.md](../../../../hydrogen_ng/docs/DL相位统计稳定提取与保留方案.md) 方案 F/J1。
> 触发条件：Ⅱ-0 probe 显示 TCN 前端未提取 O₂ 信号。

- 固定/随机 1D 卷积核 + 多种池化（max/mean/std/PPV/slope）
- 降维后与 TCN embedding 平衡融合
- 详见 [dl_training_plan.md §9.3 T5](dl_training_plan.md#93-条件迁移的技术阶段-ⅲ-备选)

---

## 执行顺序与依赖

```
仿真适配（阶段 1-3）
    │
    ▼
Ⅰ-1 tv3-smoke ──→ Ⅰ-2 tv3-formal ──→ Ⅰ-3 配置矩阵 ──→ Ⅰ-4 基线训练
                                                                │
                                                                ▼
                                                         Ⅱ-0 TCN probe  ⏳ 待执行
                                                                │
                                          ┌─────────────┬───────┤
                                          ▼             ▼       ▼
                                       Ⅱ-1 通道消融  Ⅱ-2 O₂   Ⅱ-3 Loss
                                          │          可辨识性      │
                                          └─────┬───────┘         │
                                                ▼                 │
                                          Ⅱ 结果汇总 ◄────────────┘
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
              probe 显示融合丢信息     probe 显示前端未提取       O₂ R² < 0.50
                    ▼                      ▼                      ▼
              Ⅲ-5 辅助头+融合        Ⅲ-6 ROCKET 分支         Ⅲ-1 O₂ 专用通道
                    │                      │
                    └──────────┬───────────┘
                               ▼
                    Ⅲ-2/3/4 HITRAN/阶段/分层

注（2026-07-08）：P-9c 已确认触发（fusion v3_l2 O₂ R²=-0.061 < 0.50，见
[dl_training_plan.md §10](dl_training_plan.md#10-推荐执行顺序)）。但 Ⅱ-0 probe 尚未执行，
当前直接推进方向 C 的 R1a/R1b 先导对照（固定特征路线），probe 可与 R1a 并行。
```

## 风险提醒

| 风险               | 概率  | 影响  | 缓解措施               | 当前状态  |
| ---------------- | --- | --- | ------------------ | ----- |
| O₂/N₂ 热导率接近（~2%） | 高   | 高   | Ⅱ-2 专项评估           | 待阶段 Ⅱ |
| 与 hg/sg 测试回归     | 低   | 高   | 分支隔离 + 全量 pytest   | 持续    |
| formal 集生成耗时     | 中   | 低   | 先 smoke 验证再 formal | —     |
| O₂ R² 不达标        | 中   | 高   | 阶段 Ⅲ-1 专用通道后备      | 待阶段 Ⅱ |
