# 合成气实验路线图

> 基于 2026-06-26 代码审查后的进度盘点。阶段 1–4 最小链路已通，本文档规划后续实验与基础设施。
> 阶段 Ⅰ 已完成 14/15 runs（TCN ≈ Ridge pool R²≈0.96，PatchTST 修复后 0.935），详见 [stage_i3_baseline_results.md](stage_i3_baseline_results.md)。
> 阶段 Ⅱ 已完成 27/27 runs，三组 ablation 结论见 [stage_ii_ablation_results.md](stage_ii_ablation_results.md)。

> ⚠️ **2026-06-27 时间轴对齐变更**：sg4 系列正式数据集 timesteps 从 128 改为 512（与 hg `wv4-formal-hitran-standard-6000` 一致）。下方"当前状态"与"阶段 Ⅰ/Ⅱ 执行记录"中描述的 sg4-formal (128 步) 与 41 个 run 结果**已废弃，待 512 步重跑**；"阶段 Ⅰ — 正式 benchmark + 基线实验"章节往后是 512 步下的新计划，命令模板与产物 shape 已同步更新。

---

## 当前状态

```
generate_syngas_benchmark → data/sg4-formal/ (6000 seq) → dl.cli + sg4_{baseline,tcn,lstm,patchtst,ridge}.json → metrics.json
```

| 已完成 | 说明 |
|--------|------|
| schema + 采样 | `syngas_schema.py`，条件顺序采样，18 测试 |
| 物理仿真 | 声速/衰减/热导/CO NDIR/3×3 串扰，49 测试 |
| 数据生成 | slow 9 通道 + benchmark 编排 + CLI，15 测试 |
| DL 训练 | loss 校验 + trainer 适配 + cli manifest 感知，9 测试 |
| 共用层 | 7 文件向后兼容参数化，hg 353 测试不变 |
| 全量回归 | 462 passed（含 Stage Ⅱ ablation 18 个新测试） |
| **Ⅰ-1 sg4-formal benchmark** | 6000 序列 / 128 时步 / 9 慢通道，validation pass，split 4200/900/600/300 |
| **Ⅰ-2 实验配置矩阵** | `sg4_baseline / sg4_tcn / sg4_lstm / sg4_patchtst / sg4_ridge` 全部指向 `data/sg4-formal` |
| **共用 metrics bug 修复** | `ml/training.py` 中 `conditional_component_metrics` 硬编码 `x_N2` 导致 syngas Ridge 报错；改为按 label_names 自动选 bin_components；`sum_abs_error` 在非 sum=100% 场景置 None |
| **Ⅰ-3 基线训练（14/15 收敛）** | TCN pool R²=0.958、Ridge=0.957、PatchTST=0.935（修复后）、CNN1D=0.931、LSTM 2/3 seeds 0.93。详见 [stage_i3_baseline_results.md](stage_i3_baseline_results.md) |
| **编排脚本 LSTM 退出码兼容** | `scripts/run_sg4_baseline.py` returncode!=0 时优先检查 metrics.json，存在则视为 ok 并附 warning（Windows cuDNN teardown 误报兼容） |
| **trainer AMP bf16 兼容** | `src/dl/training/trainer.py` 三处 `.cpu().numpy()` 加 `.float()`，AMP bfloat16 输出不再报 `Got unsupported ScalarType BFloat16`；hg/sg 全量 444 测试通过 |
| **PatchTST 配置修复** | `sg4_patchtst.json`: AMP fp16 → fp32 / epochs 50 → 80 / patience 10 → 15。原 AMP 配置让 attention 卡在 val_loss=1.00 平台或后续 NaN 早停；fp32 后 3 seeds 全部收敛 |
| **Ⅱ-1 CO 通道 ablation** | `outputs/sg4_ablation/co_channel/`：4 tag × 3 seeds = 12 runs。B 组 x_CO R²: 0.954 → 0.484（TCN/Ridge 同向）；C 组 x_CO R²: 0.93-0.94 |
| **Ⅱ-2 串扰 ablation** | `data/sg4-formal-crosstalk` 生成成功，TCN × 3 seeds = 3 runs，所有组分 R² 与 baseline 差异 ≤0.006 |
| **Ⅱ-3 Loss 对比** | TCN × {mse,mae,huber,smooth_l1} × 3 seeds = 12 runs。x_CH4 R² 从 weighted 的 0.827 跌至 0.39-0.44 |
| **代码改动配套** | channel 子集选择（dataset/cli/features）+ crosstalk 透传（acoustic_physics/slow/benchmark/_parallel/CLI）+ 测试 `test_syngas_ablation.py`，全量 462 passed，hg 零回归 |

| 未完成 | 阻塞程度 |
|--------|----------|
| HITRAN 后端（阶段 3c） | 不阻塞（empirical 后端可用） |
| run_experiment 多 run 编排 | 仅 hg 路径可用 |
| CO 改进分析工具 | hg 有 `analyze_n2_improvement.py`，syngas 无对应 |
| LSTM seed=123 重跑（lr 调整） | 低（2/3 可用） |
| Stage Ⅲ 真实硬件 sim-to-real 测试 | 高（论文工程闭环），未启动 |

---

## 阶段 Ⅰ 执行记录（2026-06-26）

### Ⅰ-1 实测数据规模决策

> 原方案 `--sequences 512` 在采样维度（8 维 LHS）和模型训练量上不足，讨论后改用与 hg `wv4-formal-hitran-standard-6000` 对齐的 6000 序列。

实际命令：

```powershell
python -m pipeline.generate_syngas_benchmark `
    --output-root data --dataset sg4-formal --sequences 6000 `
    --seed 20260626 --timesteps 128 --dt-s 0.5 `
    --optical-absorption-backend empirical_v1 `
    --storage memmap --workers 24
```

实测产物（全部通过预期验收）：

| 检查项 | 实测值 |
|---|---|
| `manifest.composition_scheme` | `syngas` |
| `manifest.schema_version` | `v4-syngas-1` |
| `labels/y.npy` | `(6000, 4)` |
| `sequences/slow.npy` | `(6000, 128, 9)` |
| `metadata/label_names.npy` | `["x_H2", "x_CH4", "x_CO2", "x_CO"]` |
| `condition_grid_sequence.csv` 含 `x_N2` 列 | 是 |
| `quality/validation_summary.json.status` | `pass` |
| split 划分 train/val/test/extrap | 4200 / 900 / 600 / 300 |

### Ⅰ-2 配置矩阵实际产出

`configs/experiment/sg4/` 目录新增 4 个配置（外加更新 baseline）：

| 配置 | 模型 | 关键参数 |
|---|---|---|
| `sg4_baseline.json` | CNN1D | `hidden=[16,32,32]`，kernel=5，dropout=0.1 |
| `sg4_tcn.json` | TCN | `target_timesteps=128`（Stage Ⅰ-3 实测值，**当前已更新为 512**），kernel=3，dropout=0.1 |
| `sg4_lstm.json` | LSTM | `hidden=64`，`num_layers=2`，单向 |
| `sg4_patchtst.json` | PatchTST | `patch_len=16`，`stride=8`，`d_model=64`，`nhead=4` |
| `sg4_ridge.json` | Ridge | `alpha=1.0`，slow modality，7 个 sequence statistics |

公共参数：`dataset_dir=data/sg4-formal`，`modalities=slow`，`epochs=50`，early_stopping patience=10，AdamW + ReduceOnPlateau，AMP fp16，cudnn benchmark + TF32。

### Ⅰ-3 训练编排

`scripts/run_sg4_baseline.py`：5 模型 × 3 seeds (`42 / 123 / 2026`) = 15 runs，产物路径 `outputs/sg4_baseline/{model}/seed{seed}/`，自动汇总到 `outputs/sg4_baseline/summary.json` 和 `runs.jsonl`。

### 共用代码 bug 修复

`src/ml/training.py`：

1. `evaluate_regressor` 直接调用 `conditional_component_metrics(predictions, y, label_names)`，但默认 bin_components 是 `("x_N2","x_CH4")`，syngas label_names 不含 `x_N2`，Ridge 在 sg4 上直接崩。
2. 新增 helper `_default_bin_components(label_names)`：含 `x_N2` 走 hg 历史路径，否则改用 `x_CO`（与 trainer 的 `_conditional_bin_components` 对齐）。
3. `SplitEvaluation.sum_abs_error` 类型由 `float` 改为 `float | None`，syngas 场景 label 不含 `x_N2` 时置 None（同 trainer 处理逻辑）。
4. 全量回归 444 passed，hg 零破坏。

### Ⅰ-3 训练结果（test split 平均 ± std，3 seeds，仅收敛 runs）

完整数据、单 run 明细、分箱分析与待跟进项见 [stage_i3_baseline_results.md](stage_i3_baseline_results.md)。

| 模型 | seeds | pool R² | x_H2 | x_CH4 | x_CO2 | x_CO |
|---|---|---|---|---|---|---|
| **TCN** | 3 | **0.958 ± 0.001** | 0.968 ± 0.001 | 0.827 ± 0.003 | 0.969 ± 0.002 | **0.954 ± 0.000** |
| **Ridge** | 3 | **0.957 ± 0.000** | **0.977** | 0.826 | 0.966 | 0.946 |
| PatchTST¹ | 3 | 0.935 ± 0.008 | 0.955 ± 0.002 | 0.686 ± 0.031 | 0.964 ± 0.007 | 0.926 ± 0.016 |
| CNN1D | 3 | 0.931 ± 0.005 | 0.945 ± 0.007 | 0.727 ± 0.031 | 0.960 ± 0.001 | 0.924 ± 0.003 |
| LSTM | 2/3 | 0.930 ± 0.003 | 0.945 ± 0.005 | 0.428 ± 0.018 | 0.959 ± 0.003 | 0.940 ± 0.001 |

¹ PatchTST 首轮 3 seeds 在 fp16 AMP + lr=1e-3 下全部 NaN 早停；同日改 fp32 + epochs=80 + patience=15 重跑 3 seeds 全部收敛。

关键结论：

1. **TCN ≈ Ridge ≈ 0.96**，比 PatchTST/CNN1D/LSTM 高出 2–3 个百分点。Ridge 仅用 63 维手工统计量 + closed-form L2，在 x_H2 上反而压过所有 DL 模型。**慢通道 + 手工特征已接近性能上限**，深度模型边际收益有限——需写进论文 discussion 重点讨论。
2. **CO 检测可靠（R²≈0.92–0.95）**，反向支持 CO/N₂ 声学简并假说：V_NDIR_CO 单通道有效，Ⅱ-1 CO channel ablation 应能直接量化。
3. **CH₄ 是短板**（R²≈0.83 上限），与 roadmap 风险矩阵预期一致。Transformer 类（PatchTST 0.69）比 CNN/Ridge 更差。
4. **PatchTST 对 AMP 极度敏感**：fp16/bf16 让 attention 卡在常数预测平台（val_loss=1.00 不动）或后续 NaN 早停，必须 fp32 才能学。修复后表现仍不及 TCN。
5. **LSTM seed=123 不收敛**（best_epoch=1），单 seed 偶发，2/3 可用。

副产品修复（同日，全量 444 测试通过）：

- `scripts/run_sg4_baseline.py`：returncode != 0 时优先检查 metrics.json（Windows cuDNN teardown 误报兼容）
- `src/dl/training/trainer.py`：3 处 `.cpu().numpy()` 加 `.float()`（AMP bf16 兼容）
- `configs/experiment/sg4/sg4_patchtst.json`：AMP fp16 → fp32，epochs 50 → 80，patience 10 → 15

---

## 阶段 Ⅱ 执行记录（2026-06-27）

完整设计与代码改动见 [stage_ii_ablation_plan.md](stage_ii_ablation_plan.md)，完整结果与单 run 明细见 [stage_ii_ablation_results.md](stage_ii_ablation_results.md)。

### 实测命令

```powershell
# Ⅱ-2 数据集（与 sg4-formal 同参数，仅加 --enable-co-crosstalk）
python -m pipeline.generate_syngas_benchmark `
    --output-root data --dataset sg4-formal-crosstalk `
    --sequences 6000 --seed 20260626 --timesteps 128 `
    --optical-absorption-backend empirical_v1 `
    --storage memmap --workers 24 --enable-co-crosstalk

# 三组 ablation 训练（27 runs，TCN seed=42/123/2026 + Ridge closed-form）
python scripts/run_sg4_ablation.py --experiment all
```

manifest `optical_crosstalk_policy=syngas_empirical_3x3_step2_co2_co_crosstalk`，`enable_co_crosstalk=True`。

### 关键结果（test split，mean ± std over 3 seeds）

| ablation | x_H2 | x_CH4 | x_CO2 | **x_CO** | pool |
|---|---|---|---|---|---|
| A baseline TCN | 0.968 | 0.827 | 0.969 | **0.954** | 0.958 |
| B TCN dropco（去 V_NDIR_CO） | 0.967 | 0.791 | 0.962 | **0.484** | 0.745 |
| B Ridge dropco | 0.977 | 0.808 | 0.966 | **0.470** | 0.744 |
| C TCN coonly（仅 V_NDIR_CO+环境） | 0.037 | 0.214 | 0.151 | **0.928** | 0.454 |
| C Ridge coonly | 0.044 | 0.220 | 0.141 | **0.941** | 0.461 |
| Ⅱ-2 TCN crosstalk | 0.967 | 0.821 | 0.968 | **0.956** | 0.958 |
| Ⅱ-3 TCN mse | 0.973 | **0.414** | 0.970 | 0.956 | 0.949 |
| Ⅱ-3 TCN mae | 0.969 | **0.392** | 0.964 | 0.948 | 0.943 |
| Ⅱ-3 TCN huber | 0.970 | **0.395** | 0.972 | 0.951 | 0.946 |
| Ⅱ-3 TCN smooth_l1 | 0.971 | **0.440** | 0.965 | 0.954 | 0.947 |

### 关键结论

1. **Ⅱ-1（物理证据）**：V_NDIR_CO 是 CO 检测的支配通道——移除后 x_CO R² 损失 ~50%（0.954 → 0.484，不是原预期的 ~0）；仅保留 V_NDIR_CO + 环境即可恢复 95% 以上 CO 精度。TCN 与 Ridge 同向，排除"仅时序模型依赖光学"质疑。**论文叙事应从"完全依赖"修正为"主导依赖"**。
2. **Ⅱ-2（鲁棒性）**：CO₂↔CO 线性串扰不构成模型学习难度——所有组分 R² 与基线持平（|Δ| ≤ 0.006），不出现原预期的 0.01–0.05 下降。3×3 串扰是确定性映射，模型可学到逆映射；真实 sim-to-real gap 应在 Stage Ⅲ 硬件层面验证（湿度敏感性、温漂、CO/H₂O 真实交叉等）。
3. **Ⅱ-3（方法学）**：单一超参（loss 加权）决定 CH₄ 性能——`weighted_component_mse(inverse_train_var)` 让 x_CH4 R² 翻倍（未加权 0.39–0.44 → 加权 0.83），其他组分几乎不变。低浓度组分的方差小，未加权 loss 下被高方差组分梯度淹没。

### 与原计划预期偏差

| 项 | 原预期 | 实测 | 影响 |
|---|---|---|---|
| Ⅱ-1 B 组 x_CO R² | ~0 | 0.484 | 物理叙事修正：CO 主导依赖光学，非完全依赖 |
| Ⅱ-2 x_CO/x_CO2 R² 下降 | 0.01–0.05 | -0.006~+0.002（持平） | 串扰不构成学习难度，论文转为"模型对线性串扰具备隐式校正能力" |
| Ⅱ-3 各 loss 对 CH₄ 的差异 | 待量化 | 加权 vs 未加权：+0.4 R²；未加权之间 ≤0.03 | 加权是低浓度组分的关键，loss 类型本身次要 |

### 产物

| 产物 | 路径 |
|---|---|
| Ⅱ-1/2/3 各 run metrics | `outputs/sg4_ablation/{experiment}/{tag}/seed{seed}/metrics.json` |
| 汇总 JSON | `outputs/sg4_ablation/summary.json` |
| 编排脚本 | `scripts/run_sg4_ablation.py` |
| crosstalk 数据集 | `data/sg4-formal-crosstalk/` |
| 实验配置 | `configs/experiment/sg4/ablation/{co_channel,crosstalk,loss}/*.json` |
| 完整分析 | [stage_ii_ablation_results.md](stage_ii_ablation_results.md) |

---

## 阶段 Ⅰ — 正式 benchmark + 基线实验

> 目标：获得各模型在正式数据上的组分级 R²/MAE/RMSE，建立 syngas 基线。
>
> 不依赖 HITRAN，empirical 后端即可。

### Ⅰ-1. 生成正式规模 benchmark

参考 hg `wv4-formal-hitran-standard-6000` 的规模（6000 序列 / 512 时步），完整对齐时间轴。

```powershell
python -m pipeline.generate_syngas_benchmark `
    --output-root data `
    --dataset sg4-formal `
    --sequences 6000 `
    --seed 20260626 `
    --timesteps 512 `
    --dt-s 0.5 `
    --optical-absorption-backend empirical_v1 `
    --storage memmap `
    --workers 24
```

产物验证：

| 检查项 | 预期 |
|--------|------|
| `manifest.json` → `composition_scheme` | `"syngas"` |
| `labels/y.npy` shape | `(6000, 4)` |
| `sequences/slow.npy` shape | `(6000, 512, 9)` |
| `metadata/label_names.npy` | `["x_H2", "x_CH4", "x_CO2", "x_CO"]` |
| `condition_grid_sequence.csv` 含 `x_N2` 列 | 是 |
| `quality/validation_summary.json` → `status` | `"pass"` |
| split 划分 | train 4200 / val 900 / test 600 / extrapolation 300 |

### Ⅰ-2. 补全实验配置矩阵

在 `configs/experiment/sg4/` 下新建：

| 配置文件 | 模型 | Loss | 说明 |
|----------|------|------|------|
| `sg4_baseline.json` | CNN1D | weighted_component_mse | 已有 |
| `sg4_tcn.json` | TCN | weighted_component_mse | 对标 hg TCN 基线 |
| `sg4_lstm.json` | LSTM | weighted_component_mse | 对标 hg LSTM 基线 |
| `sg4_patchtst.json` | PatchTST | weighted_component_mse | 对标 hg PatchTST |
| `sg4_ridge.json` | Ridge (ML) | — | 传统 ML baseline |

公共参数约束：

- `dataset_dir`: `data/sg4-formal`
- `modalities`: `slow`（首轮仅用慢通道，与 hg 基线对齐）
- `in_channels`: 9（自动从数据读取）
- `out_dim`: 4（自动从 label_names 读取）
- 不设 `target_transform`（syngas 禁用）
- Loss 选 `weighted_component_mse`（`inverse_train_var`，`component_count=4`）
- `epochs`: 50，`early_stopping`: patience=10

### Ⅰ-3. 跑基线训练

每个配置跑 3 seeds（42 / 123 / 2026），记录：

| 指标 | 粒度 | 说明 |
|------|------|------|
| overall R² / MAE / RMSE | pooled | 四组分混合 |
| per-component R² / MAE | x_H2 / x_CH4 / x_CO2 / x_CO | 重点关注 CO |
| conditional metrics | co_bins / ch4_bins | 分箱精度分布 |
| val_loss 收敛曲线 | epoch-level | 过拟合判断 |

预期观察：

- H₂ R² 最高（声速贡献最大，动态范围宽）
- CH₄ R² 受限（浓度区间仅 0–12%，信噪比低）
- CO R² 取决于 V_NDIR_CO 通道的有效性（声学/TCS 近简并，光学是主要信息源）
- CO₂ R² 与 hg 场景可比（NDIR 通道信号强）

---

## 阶段 Ⅱ — 关键 ablation 实验

> 目标：验证核心物理假说，产出论文的科学贡献点。
> **状态：已完成 27/27 runs（2026-06-27）**。完整结果见 [stage_ii_ablation_results.md](stage_ii_ablation_results.md)，原始设计与代码改动见 [stage_ii_ablation_plan.md](stage_ii_ablation_plan.md)，本节保留为计划存档。

### Ⅱ-1. CO 通道 ablation（最高优先级）✅

**假说**：CO 与 N₂ 声学近简并（M=28, γ=1.40, Δc < 1 m/s），CO 的可观测性完全依赖 V_NDIR_CO 光学通道。

**实测**：

| 组别 | 输入通道 | 预期 | 实测 x_CO R²（TCN / Ridge） |
|------|----------|------|---|
| A: 全通道 | 9 通道（含 V_NDIR_CO） | CO R² 正常 | 0.954 / 0.946 |
| B: 去 CO NDIR | 8 通道（去掉 V_NDIR_CO） | ~0 | **0.484 / 0.470** |
| C: 仅 CO NDIR + 环境 | 6 通道 | 接近 A | 0.928 / 0.941 |

实现方式：`V4BenchmarkDataset(slow_channels=[...])` + `MLFeatureConfig(slow_channels=[...])`，scaler 按全 9 通道 fit 后按列选择，`in_channels` 由 `_infer_input_shape` 自动推断。

**结论修正**：B 组未"暴跌至 ~0"，而是损失约 50%。CO 主导依赖光学通道但保留 ~50% 残留可学性，可能来自闭包约束（`x_N2 = 100 - sum`）、V_TCS 热导差异、CO₂ NDIR 弱串扰。这是论文最强证据之一。

### Ⅱ-2. 串扰 ablation ✅

**假说**：CO₂↔CO 光学串扰在高 CO₂ 低 CO 工况下显著影响 CO 检测精度。

**实测**：

| 组别 | 串扰设定 | benchmark | x_CO R² | x_CO2 R² |
|------|----------|-----------|---|---|
| Step 1 | `enable_co_crosstalk=False` | `sg4-formal` | 0.954 | 0.969 |
| Step 2 | `enable_co_crosstalk=True` | `sg4-formal-crosstalk` | **0.956** | **0.968** |

差异在 ±0.006 以内，**不构成可学习问题**。3×3 矩阵是确定性线性变换，模型学到逆映射。CLI 透传链：`--enable-co-crosstalk` → `SyngasBenchmarkGenerationSpec` → `build_sequence_arrays` → `main_sensor_features` → `apply_syngas_optical_crosstalk`。

**论文意义**：sim-to-real gap 不在串扰矩阵层面，应在 Stage Ⅲ 真实硬件层面验证。

### Ⅱ-3. Loss 对比 ✅

实测（TCN × 3 seeds × 5 loss，详见 stage_ii_ablation_results §4）：

| Loss | x_CH4 R² | x_CO R² | pool R² |
|------|----------|---------|---------|
| `weighted_component_mse`（inverse_train_var） | **0.827** | 0.954 | 0.958 |
| `mse` | 0.414 | 0.956 | 0.949 |
| `mae` | 0.392 | 0.948 | 0.943 |
| `huber` | 0.395 | 0.951 | 0.946 |
| `smooth_l1` | 0.440 | 0.954 | 0.947 |

**结论**：单一超参（loss 加权）决定 CH₄ R²，翻倍（0.40 → 0.83）且不牺牲其他组分。这是论文核心方法学贡献。

### Ⅱ-4. CO 弛豫系数扫描（本轮不做）

Ⅱ-1 已确认 CO 主导依赖光学，但声学/热导通道仍贡献 ~50% 可学性，CO 弛豫系数扫描转为低优先级。**未在 Stage Ⅱ 执行**，可在 Stage Ⅲ 硬件标定数据可用后再做。

---

## 阶段 Ⅲ — 基础设施补全

> 按论文需要选做，不阻塞阶段 Ⅰ/Ⅱ 的结论。

### Ⅲ-1. HITRAN 后端

| 工作项 | 说明 |
|--------|------|
| 联网 fetch CO/CO₂/H₂O 谱线 [1980, 2310] cm⁻¹ | ~50 MB 缓存 |
| 扩展 `spectral-defaults.json` 新增 CO 通道配置 | 或新建 syngas 专用 spectral defaults |
| `optical_backend.compute_hitran_optical_absorption` 支持 `channels=("ch4","co2","co")` | CO 通道需 CO+CO₂+H₂O 三气体前向 |
| 移除 `syngas/slow.py` 的 `NotImplementedError` | 落地实际计算逻辑 |

用途：对比 empirical vs HITRAN 光学通道精度差异，增加论文严谨性。也可用 HITRAN 谱积分精确标定串扰系数 ε₃₂。

### Ⅲ-2. run_experiment 多 run 编排

当前 `pipeline/run_experiment.py` 仅支持 hg。syngas 接入需要：

| 改动 | 说明 |
|------|------|
| `experiment_config.py` 支持 `composition_scheme` 感知 | 报告字段 x_N2 → x_CO |
| `run_experiment.py` 评估指标分流 | sum_abs_error / compositional_metrics 按 scheme 跳过 |
| `_summary_metrics` 中 N₂ 引用替换为 CO | 或参数化 |

在实验规模大（>5 配置 × 3 seeds）时值得做。小规模用 `dl.cli` 手动跑也可以。

### Ⅲ-3. CO 改进分析工具

类似 hg 的 `analyze_n2_improvement.py`：

- 按 CO 浓度分箱（低/中/高）分析各模型的精度分布
- 识别模型在哪些 CO 浓度区间性能退化
- 支撑论文中"不同工况下的检测可靠性"分析

### Ⅲ-4. 占位参数替换

| 参数 | 当前值 | 替换时机 |
|------|--------|----------|
| CO NDIR 滤光片 | InfraTec I 4.66 μm 行业参考 | 获得目标传感器 datasheet 后 |
| 热导 x_co 系数 | -0.00005（占位） | 文献标定后 |
| CO 弛豫 alpha_lambda_max_co | 0.025（中置信） | 阶段 Ⅱ-4 ablation 确认后 |
| 串扰 ε₃₂ | 0.005（物理估算） | HITRAN 谱积分或实测标定后 |

---

## 执行顺序与依赖

```
Ⅰ-1 生成 sg4-formal ──┐
                        ├─ Ⅰ-3 基线训练 ✅ ──┬─ Ⅱ-1 CO channel ablation ✅
Ⅰ-2 补全配置矩阵 ✅ ───┘                       ├─ Ⅱ-2 串扰 ablation ✅
                                                ├─ Ⅱ-3 Loss 对比 ✅
                                                └─ Ⅱ-4 CO 弛豫扫描（本轮不做）

Ⅲ-1 HITRAN 后端 ──────── 独立，不阻塞 Ⅰ/Ⅱ
Ⅲ-2 run_experiment 接入 ─ 实验规模大时做
Ⅲ-3 CO 分析工具 ──────── 写论文时做
Ⅲ-4 占位参数替换 ──────── 获得实测数据后做
```

---

## 风险提醒

| 风险 | 概率 | 影响 | 缓解 | 现状 |
|------|------|------|------|------|
| CO/N₂ 声学简并 → 模型退化为单通道依赖 | 高 | 高 | Ⅱ-1 ablation 验证；若成立，论文转为"证实+量化简并效应" | **已验证（部分成立）**：B 组 R² 损失 ~50%（非 100%），主导依赖而非完全依赖 |
| CH₄ 低浓度区间（0–12%）精度不足 | 中 | 中 | weighted_component_mse 加权补偿；分箱分析低浓度表现 | **已验证（关键）**：Ⅱ-3 证明加权可让 R² 翻倍（0.40 → 0.83），是论文方法学核心 |
| 串扰系数为占位值，影响 sim-to-real 迁移 | 中 | 中 | Ⅱ-2 ablation 量化敏感度；HITRAN 谱积分标定 | **Ⅱ-2 结果**：线性串扰对模型可学性无影响，sim-to-real gap 在硬件层面而非串扰矩阵 |
| 正式 benchmark 生成耗时 | 低 | 低 | 并行 workers；6000 序列 + empirical 后端 + 24 workers 实测 <2 分钟 | 已验证 |
