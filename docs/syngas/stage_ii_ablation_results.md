# 阶段 Ⅱ ablation 实验结果（2026-06-27）

> 三组 ablation 共 27 runs，全部在 `data/sg4-formal`（6000 序列 / 128 时步 / 9 慢通道，empirical 后端，`enable_co_crosstalk=False`，split 4200/900/600/300）上展开，仅 Ⅱ-2 使用同参数 `enable_co_crosstalk=True` 重新生成的 `data/sg4-formal-crosstalk`（6000 / 128，optical_crosstalk_policy=`syngas_empirical_3x3_step2_co2_co_crosstalk`）。
> 训练参数：epochs=50，AdamW lr=1e-3，weight_decay=1e-4，AMP fp16，ReduceOnPlateau（factor=0.5 / patience=5 / min_lr=1e-6），early_stopping patience=10。
> Loss：除 Ⅱ-3 外均为 `weighted_component_mse`（inverse_train_var）。
> seeds：42 / 123 / 2026，与 stage Ⅰ-3 基线对齐；A 组（全通道）结果直接复用 `outputs/sg4_baseline/{tcn,ridge}`。

## 1. 实验完整性

`python scripts/run_sg4_ablation.py --experiment all` 期望产出 27 个 run，全部成功：

| experiment | 配置 × seeds | metrics.json |
|---|---|---|
| co_channel | tcn_dropco / tcn_coonly / ridge_dropco / ridge_coonly × 3 | 12 ✓ |
| crosstalk | tcn_crosstalk × 3 | 3 ✓ |
| loss | tcn_mse / tcn_mae / tcn_huber / tcn_smoothl1 × 3 | 12 ✓ |
| **合计** | | **27 / 27 ✓** |

汇总：`outputs/sg4_ablation/summary.json`（49 KB）。Ridge 三个 seed 因 closed-form 解结果完全一致，差异为 0。TCN 多数 run 跑满 50 epoch 或在 early_stopping 触发前几个 epoch 收敛，仅 `tcn_coonly` 在 13–29 epoch 早停（val_loss 在低位平台）。

## 2. Ⅱ-1 CO 通道 ablation

### 2.1 设计

| 组 | 含义 | 保留通道 | in_channels |
|---|---|---|---|
| A | 全通道（基线） | 全 9 | 9 |
| B | 去 V_NDIR_CO | `V_NDIR_CH4, V_NDIR_CO2, V_TCS, T_C, P_MPa, H_RH, L_m, piston_position_m` | 8 |
| C | 仅 CO 光学 + 环境 | `V_NDIR_CO, T_C, P_MPa, H_RH, L_m, piston_position_m` | 6 |

### 2.2 结果（test split，mean ± std）

| 组 | 模型 | pool R² | x_H2 | x_CH4 | x_CO2 | **x_CO** |
|---|---|---|---|---|---|---|
| A | TCN | 0.958±0.001 | 0.968 | 0.827 | 0.969 | **0.954** |
| A | Ridge | 0.957 | 0.977 | 0.826 | 0.966 | **0.946** |
| B | TCN dropco | 0.745±0.002 | 0.967±0.001 | 0.791±0.016 | 0.962±0.009 | **0.484±0.005** |
| B | Ridge dropco | 0.744 | 0.977 | 0.808 | 0.966 | **0.470** |
| C | TCN coonly | 0.454±0.009 | 0.037±0.005 | 0.214±0.003 | 0.151±0.001 | **0.928±0.019** |
| C | Ridge coonly | 0.461 | 0.044 | 0.220 | 0.141 | **0.941** |

MAE（pool / x_CO，test，mean）：

| 组 | TCN pool MAE | TCN x_CO MAE | Ridge pool MAE | Ridge x_CO MAE |
|---|---|---|---|---|
| A | 1.603 | 2.393 | 1.648 | 2.698 |
| B | 3.128 | 8.279 | 3.130 | 8.567 |
| C | 5.680 | 2.928 | 5.662 | 2.813 |

### 2.3 发现

**B 组：去 V_NDIR_CO 后 x_CO R² 从 0.954 → 0.48，MAE 从 2.4 → 8.3 mol%**。其余三个组分（H₂/CH₄/CO₂）几乎不受影响（ΔR² ≤ 0.04），说明 V_NDIR_CO 通道承担的信息几乎全部用于 CO 预测，没有跨组分耦合。TCN 与 Ridge 两个独立模型给出几乎相同的退化幅度（0.484 vs 0.470），跨模型一致性排除"模型相关性"质疑。

**与原计划预期偏差**：plan `stage_ii_ablation_plan.md` §Ⅱ-1 写"B 组 x_CO R² 暴跌至 ~0"。实际结果是 ~0.48，不是 0。说明 CO 并非 100% 依赖 NDIR 光学通道，仍有约一半的可预测信号来自其他通道。可能的物理来源：

1. `x_N2 = 100 - sum(x_H2, x_CH4, x_CO2, x_CO)` 的闭包约束——只要剩下三个组分预测准确，CO 在 sum<100 框架里仍有间接约束；
2. V_TCS（热导）通道在 H₂ / CH₄ / CO₂ 的浓度组合给定后，仍能对 CO 提供轻量信息——尽管 CO 与 N₂ 摩尔质量相同，热导率 λ_CO ≈ 25.1 mW/(m·K) 与 λ_N₂ ≈ 25.8 mW/(m·K) 差 ~3%；
3. NDIR_CO2 通道与 CO 的弱光学串扰（数据集 `enable_co_crosstalk=False`，但 CO₂ 自身的滤光片可能在 CO 吸收带边缘有微弱响应残留——需要 sg4-formal-crosstalk 上重复 B 组进一步验证）。

**C 组：仅 CO 光学 + 环境**，x_CO R² 仍达 0.93（TCN）/ 0.94（Ridge），证明 V_NDIR_CO 单通道在工程上可独立支撑 CO 检测。其余组分的暴跌（H₂ 0.04 / CH₄ 0.21 / CO₂ 0.14）属预期：主光学通道与 V_TCS 被全部移除。

**论文写作建议**：原 roadmap 表述"CO 的可观测性几乎完全依赖光学通道"应修正为"**V_NDIR_CO 是 CO 可观测性的支配通道**，移除后 R² 损失约 50%（0.96 → 0.48）；仅保留 V_NDIR_CO 即可恢复 95% 以上的 CO 检测精度"。这一支配关系是 stage Ⅰ-3 CO/N₂ 声学简并假说的直接验证。

### 2.4 单 run 明细（test）

| 模型 / 组 | seed | best_ep | pool R² | x_H2 | x_CH4 | x_CO2 | x_CO | stop |
|---|---|---|---|---|---|---|---|---|
| tcn_dropco | 42 | 36 | 0.748 | 0.968 | 0.792 | 0.951 | 0.490 | early-stop |
| tcn_dropco | 123 | 49 | 0.746 | 0.967 | 0.811 | 0.972 | 0.481 | completed |
| tcn_dropco | 2026 | 46 | 0.742 | 0.965 | 0.771 | 0.963 | 0.480 | completed |
| tcn_coonly | 42 | 29 | 0.465 | 0.043 | 0.215 | 0.152 | 0.948 | early-stop |
| tcn_coonly | 123 | 13 | 0.453 | 0.032 | 0.210 | 0.151 | 0.933 | early-stop |
| tcn_coonly | 2026 | 13 | 0.442 | 0.037 | 0.217 | 0.149 | 0.903 | early-stop |
| ridge_dropco | — | — | 0.744 | 0.977 | 0.808 | 0.966 | 0.470 | closed-form |
| ridge_coonly | — | — | 0.461 | 0.044 | 0.220 | 0.141 | 0.941 | closed-form |

TCN dropco 跨 seed 标准差 0.002（pool），早停后 CO R² 在 0.48-0.49 之间稳定，确认是真实信号上限而非训练不足。TCN coonly 早停较早（13-29 epoch），val_loss 在低位平台，进一步训练对剩余三个组分无收益（H₂/CH₄/CO₂ 信号源已被移除）。

## 3. Ⅱ-2 串扰 ablation

### 3.1 设计

数据集对比：

| 数据集 | enable_co_crosstalk | optical_crosstalk_policy | 其它参数 |
|---|---|---|---|
| sg4-formal | False | syngas_empirical_3x3_step1_co_pure | seed=20260626 / 6000 / 128 / empirical_v1 |
| sg4-formal-crosstalk | True | syngas_empirical_3x3_step2_co2_co_crosstalk | 同上 |

CLI 透传链：`generate_syngas_benchmark --enable-co-crosstalk` → `SyngasBenchmarkGenerationSpec.enable_co_crosstalk` → `build_sequence_arrays` → `main_sensor_features` → `apply_syngas_optical_crosstalk`。3×3 矩阵在 CO 通道引入 CO₂ 串扰（CO₂ 浓度→CO 通道信号的线性混合）+ CO 自身吸收。

训练：TCN × 3 seeds，配置完全等同 baseline `sg4_tcn.json`，仅 `dataset_dir` 切换。

### 3.2 结果（test split，mean ± std）

| 数据集 | pool R² | x_H2 | x_CH4 | x_CO2 | x_CO |
|---|---|---|---|---|---|
| sg4-formal（baseline TCN） | 0.958±0.001 | 0.968±0.001 | 0.827±0.003 | 0.969±0.002 | 0.954±0.000 |
| sg4-formal-crosstalk | 0.958±0.004 | 0.967±0.005 | 0.821±0.011 | 0.968±0.007 | 0.956±0.003 |
| **Δ** | **+0.000** | **-0.001** | **-0.006** | **-0.001** | **+0.002** |

MAE 同样：

| 数据集 | pool | x_H2 | x_CH4 | x_CO2 | x_CO |
|---|---|---|---|---|---|
| baseline | 1.603 | 1.907 | 1.151 | 0.963 | 2.393 |
| crosstalk | 1.598 | 1.905 | 1.168 | 0.957 | 2.360 |

### 3.3 发现

**所有组分 R² 差异 |Δ| ≤ 0.006，pool R² 完全持平**。原计划预期 0.01–0.05 的 x_CO/x_CO2 下降未出现。

物理解释：3×3 串扰矩阵是**确定性线性变换** `V_NDIR_CO_observed = α·c_CO + β·c_CO2 + noise`。在训练数据中既给模型看到 NDIR 信号又给出真值标签的前提下，TCN 完全可以学到逆映射并恢复纯 CO 信号。串扰让物理仿真**更接近真实硬件特性**，但**不增加可学习问题的难度**。

这是一个 informative 的负结果：

1. **对论文叙事**：可写作"光学通道间的线性串扰不构成 sim-to-real gap，模型本身具备隐式校正能力"。
2. **对 sim-to-real 风险评估**：真正的 gap 应来自非线性 / 时变 / 标定漂移因素（湿度敏感性、温度依赖、传感器老化、CO 与 H₂O 在 2143 cm⁻¹ 附近的真实交叉），不在串扰矩阵层面。Stage Ⅲ 在真实硬件上的对比仍然必要。
3. **数据集的工程意义保留**：sg4-formal-crosstalk 比 sg4-formal 更接近真实 NDIR 滤光片响应，未来训练应优先使用 crosstalk 数据集——虽然指标持平，但模型学到的内部表征更符合真实物理过程。

### 3.4 单 run 明细（test）

| seed | best_ep | pool R² | x_H2 | x_CH4 | x_CO2 | x_CO | stop |
|---|---|---|---|---|---|---|---|
| 42 | 47 | 0.953 | 0.962 | 0.820 | 0.959 | 0.951 | completed |
| 123 | 50 | 0.960 | 0.966 | 0.834 | 0.973 | 0.959 | completed |
| 2026 | 50 | 0.961 | 0.973 | 0.808 | 0.972 | 0.957 | completed |

跨 seed 标准差 0.004，与 baseline TCN（0.001）同量级，无收敛异常。

## 4. Ⅱ-3 Loss 对比

### 4.1 设计

TCN × 4 个开放类 loss × 3 seeds。`weighted_component_mse(inverse_train_var)` 复用 baseline `sg4_tcn`，其余配置完全相同。闭包类 loss（`compositional_mse / ilr_mse / *_free_component_mse`）被 syngas 的 `validate_loss_composition_scheme` 拒绝，不在对比范围。

### 4.2 结果（test split，mean ± std）

| loss | pool R² | x_H2 | **x_CH4** | x_CO2 | x_CO |
|---|---|---|---|---|---|
| **weighted_component_mse**（baseline） | **0.958±0.001** | 0.968 | **0.827±0.003** | 0.969 | 0.954 |
| mse | 0.949±0.001 | 0.973±0.000 | **0.414±0.010** | 0.970±0.002 | 0.956±0.002 |
| mae | 0.943±0.006 | 0.969±0.004 | **0.392±0.053** | 0.964±0.007 | 0.948±0.005 |
| huber | 0.946±0.002 | 0.970±0.002 | **0.395±0.005** | 0.972±0.001 | 0.951±0.005 |
| smooth_l1 | 0.947±0.004 | 0.971±0.005 | **0.440±0.030** | 0.965±0.007 | 0.954±0.003 |

MAE（pool / x_CH4，test）：

| loss | pool MAE | x_CH4 MAE | x_CO MAE |
|---|---|---|---|
| weighted_component_mse | 1.603 | 1.151 | 2.393 |
| mse | 1.800 | 2.217 | 2.317 |
| mae | 1.869 | 2.168 | 2.506 |
| huber | 1.811 | 2.164 | 2.421 |
| smooth_l1 | 1.799 | 2.107 | 2.358 |

### 4.3 发现

**单一超参（loss 函数）的切换，使 x_CH4 R² 从 0.827 跌至 0.39-0.44，下降 ~0.4**。其余组分（H₂/CO₂/CO）的 R² 在 0.94-0.97 之间几乎不变，pool R² 在 0.94-0.95 之间。

物理原因：

- CH₄ 浓度区间 0-12 mol%，方差 ~5（mol%）²，是四组分中最低；H₂ 0-100，方差 ~700+；CO 0-50，方差 ~150+；
- 未加权 loss 下，MSE 梯度按组分平方误差求和，**H₂ / CO 的大数值梯度淹没 CH₄**，模型选择牺牲 CH₄ 来降低总 loss；
- `inverse_train_var` 加权按 `1/Var(y_train)` 缩放每个组分的 loss 项，把 CH₄ 的梯度权重放大约 100×，迫使模型平等对待四个组分。

**这是论文最有说服力的方法学贡献点**：单行代码（loss 配置切换），CH₄ R² 翻倍（0.40 → 0.83），且**不牺牲其他组分**——H₂ R² 在加权版本下 0.968，未加权版本下 0.973，反而是加权版本略低 0.005。这是低浓度组分检测的典型解法，可推广到其他多组分浓度反演问题。

延伸观察：

- 四个未加权 loss 之间差异很小（pool R² 在 0.943-0.949 之间），mse / smooth_l1 在数值上略优于 mae / huber，但跨 seed 差异（±0.001 - ±0.006）相近；
- smooth_l1 的 x_CH4 R² 最高（0.440），比 mse（0.414）略好，可能是 Huber-like 平滑性在 CH₄ 低值区域抑制了极端梯度。但提升幅度（+0.026）远小于换 weighted_component_mse 带来的提升（+0.41）。

### 4.4 单 run 明细（test）

| loss | seed | best_ep | pool R² | x_H2 | x_CH4 | x_CO2 | x_CO | stop |
|---|---|---|---|---|---|---|---|---|
| tcn_mse | 42 | 36 | 0.948 | 0.973 | 0.390 | 0.972 | 0.958 | early-stop |
| tcn_mse | 123 | 48 | 0.949 | 0.973 | 0.422 | 0.967 | 0.956 | completed |
| tcn_mse | 2026 | 49 | 0.950 | 0.973 | 0.420 | 0.971 | 0.957 | completed |
| tcn_mae | 42 | 36 | 0.935 | 0.967 | 0.318 | 0.954 | 0.940 | early-stop |
| tcn_mae | 123 | 50 | 0.944 | 0.967 | 0.433 | 0.967 | 0.950 | completed |
| tcn_mae | 2026 | 46 | 0.949 | 0.974 | 0.426 | 0.971 | 0.953 | completed |
| tcn_huber | 42 | 42 | 0.943 | 0.969 | 0.400 | 0.971 | 0.946 | completed |
| tcn_huber | 123 | 50 | 0.948 | 0.969 | 0.388 | 0.972 | 0.958 | completed |
| tcn_huber | 2026 | 47 | 0.946 | 0.973 | 0.396 | 0.972 | 0.950 | completed |
| tcn_smoothl1 | 42 | 49 | 0.942 | 0.964 | 0.429 | 0.956 | 0.951 | completed |
| tcn_smoothl1 | 123 | 50 | 0.953 | 0.976 | 0.481 | 0.966 | 0.958 | completed |
| tcn_smoothl1 | 2026 | 49 | 0.948 | 0.972 | 0.410 | 0.973 | 0.953 | completed |

mae seed=42 的 x_CH4 R²=0.318 是最低值（其他 seed 0.43），early-stop 触发，提示 MAE 在 lr=1e-3 + 16 batch + 50 epoch 配置下偶发对初始化敏感，但不改变排序结论。

## 5. 三组 ablation 横向对比

| ablation | 自变量 | x_CO 关键变化 | x_CH4 关键变化 | 解读 |
|---|---|---|---|---|
| Ⅱ-1 B | 去 V_NDIR_CO | **0.954 → 0.484**（-0.47） | -0.04 | V_NDIR_CO 是 CO 检测的支配通道 |
| Ⅱ-1 C | 仅 V_NDIR_CO + 环境 | -0.02 | -0.61 | 单光学通道可独立支撑 CO 检测 |
| Ⅱ-2 | 启用 CO₂↔CO 串扰 | +0.002 | -0.006 | 线性串扰不构成可学习问题 |
| Ⅱ-3 | 取消 inverse_train_var 加权 | -0.001 | **0.827 → 0.41**（-0.42） | 低浓度组分对 loss 加权高度敏感 |

按论文贡献度排序：Ⅱ-1（物理证据） ≈ Ⅱ-3（方法学）> Ⅱ-2（鲁棒性）。

## 6. 结论与下一步

### 已确认

1. **V_NDIR_CO 通道支配 CO 可观测性**：移除后 x_CO R² 从 0.954 → 0.484，TCN 与 Ridge 同向（差异 <0.02）。原 roadmap 的"完全依赖"描述应修正为"主导依赖（损失约 50%）"。
2. **仅 V_NDIR_CO + 环境**即可恢复 x_CO R²=0.93-0.94，工程上可作为 CO 单通道传感方案的可行性证据。
3. **CO₂↔CO 线性串扰不构成模型学习难度**：crosstalk 数据集训练后所有组分 R² 与基线持平。sim-to-real gap 的关键不在串扰矩阵层面。
4. **逆方差加权 loss 对低浓度组分至关重要**：CH₄ R² 从 0.41（未加权）→ 0.83（inverse_train_var），翻倍且不牺牲其他组分。这是 sg4 训练的关键方法学选择。

### 待跟进

| 项 | 状态 | 优先级 |
|---|---|---|
| 在 sg4-formal-crosstalk 上重复 B 组（验证 §2.3 假设 3：CO₂ NDIR 串扰是否提供 CO 信息） | 待启动 | 中 |
| 进一步 ablation：B 组再去 V_TCS，验证 §2.3 假设 2（热导贡献量化） | 待启动 | 低 |
| Ⅱ-3 加 lr 调优后重跑（mae/huber 可能因 lr 不当低估，目前对结论无影响） | 待评估 | 低 |
| 接入 ultrasonic / fiber_mic 模态后重做 Ⅱ-1（验证多模态是否能恢复 CO 信息） | 待启动 | 中 |
| Stage Ⅲ 真实硬件 sim-to-real 测试 | 待规划 | 高（论文工程闭环） |

### 产物

| 产物 | 路径 |
|---|---|
| 各 run metrics | `outputs/sg4_ablation/{experiment}/{tag}/seed{seed}/metrics.json` |
| 汇总 JSON | `outputs/sg4_ablation/summary.json` |
| 编排脚本 | `scripts/run_sg4_ablation.py` |
| crosstalk 数据集 | `data/sg4-formal-crosstalk/`（manifest.optical_crosstalk_policy=`syngas_empirical_3x3_step2_co2_co_crosstalk`） |
| 实验配置 | `configs/experiment/sg4/ablation/{co_channel,crosstalk,loss}/*.json` |
