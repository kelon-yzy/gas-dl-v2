# PhaseWindowTCN 实验执行与验收流程

> 更新日期：2026-06-16
> 状态：待执行
> 配套文档：`docs/PhaseWindowTCN结构消融实验方案.md`（讲"为什么这么做"），本文件讲"怎么跑、怎么判、之后怎么处理"
> 适用数据：`data/wv4-formal-hitran-standard-6000`（在服务器，本地无）

## 0. 与方案文档的关系

- 方案文档负责诊断逻辑、候选机制（A–E）、文献依据。
- 本文件是可执行手册：前置准备、运行命令、记录项、验收门槛、决策树、归档与文档同步。
- 实验命名以现有配置的 `phase_window_tcn_*` 前缀为准（方案文档里的 `pwtcn_*` 是同一批实验的简写）。

## 1. 总览：三批实验 + 三个决策门

```text
Phase 0  前置准备 ──G0──> Phase 1 诊断批 ──G1──> [按病因分流]
                                                  │
                          ┌───────────────────────┼───────────────────────┐
                          │                       │                       │
                   损失/监督是主因          特征学习是主因            都不是 → 结构问题
                          │                       │                       │
                   采用加权损失为           转向特征注入混合           Phase 2 结构消融
                   新 DL 基线，记录结论       模型 或 收束 DL          ──G2──> Phase 3 融合/对数比 ──G3──> 收束
```

- **同批同 seed**：同一批内所有 arm 用同一个 seed（当前配置 `seed=20260615`），相互之间才可比。
- **外部基准**：Ridge 多窗口（`seed=20260603`，test N2 R2=0.7121 / extrapolation=0.7247）是固定参照，不与 DL 同 seed，只用于判断 DL 离正式主线还有多远。
- **运行入口**（README 已记录，PowerShell 设 `$env:PYTHONPATH = "src"`）：
  
  ```text
  python -m pipeline.run_experiment --config <配置路径> --dry-run   # 校验并打印计划
  python -m pipeline.run_experiment --config <配置路径>             # 实跑
  ```

## 2. Phase 0：前置准备（开跑前必须完成）

| 编号   | 任务                                     | 改动位置                                            | 类型   |
| ---- | -------------------------------------- | ----------------------------------------------- | ---- |
| P0.1 | 实现按方差加权的组分损失                           | `src/dl/training/losses.py`                     | 新增代码 |
| P0.2 | 构建手工特征 + 小 MLP 诊断模型                    | `src/dl/models/` + 配置                           | 新增代码 |
| P0.3 | 确认逐 epoch 与 per-component 日志可用         | 验证现有 `metrics_live.jsonl` / summary CSV         | 验证   |
| P0.4 | 改写第一批配置为诊断批，旧 split/deep 移到 Phase 2 配置 | `configs/experiment/phase_window_tcn_ablation/` | 配置   |
| P0.5 | 全部新配置 dry-run 通过 + 新 loss 单测通过         | —                                               | 验证   |

### P0.1 加权损失

在 `losses.py` 注册一个新 loss，权重 `w_c = 1/σ_c²`（σ_c 取训练集各组分标准差）：

- `weighted_component_mse`：在 **4 组分百分比**上做加权 MSE，监督全 4 列、与 head 无关，N2 获得直接但尺度平衡的监督。
- `weighted_free_component_mse`：只对 **3 个自由组分**做方差加权，N2 仍为残差（候选 A 的对照）。

实现要点：

- 走 `build_loss` 已支持的"字典 + 额外 kwargs"机制，配置写 `{"name": "weighted_component_mse", "weighting": "inverse_train_var"}`。
- `weighting="inverse_train_var"` 时从训练集统计读组分方差；权重在构造时固定，不随 batch 变化。
- `validate_loss_model_output` 的 head 约束按监督列数区分：`weighted_component_mse` 监督全 4 列、不依赖闭包，允许 `phase_window_tcn` 配 `output_mode` 为 `raw4` / `softmax100` / `gas_head`（`out_dim` 必须为 4）；`free_component_mse` 与 `weighted_free_component_mse` 只监督前 3 列、靠 sum=100 闭包补 N2，仍强制 `output_mode='gas_head'`。
- **eval 一律在原始百分比空间算 R2**，加权只作用于训练损失，不改评估口径。

需写最小单测：权重计算正确、shape 校验、与 `gas_head` 组合校验通过。

### P0.2 手工特征 + 小 MLP 诊断

复用 ML 侧 `ml.features` 的窗口统计特征（与 `ridge_multiwindow_all_modalities` 同款），接 2 层 MLP + gas_head：

```text
features = 多窗口手工统计特征(full, exposure, recovery)   # 复用现有 MLFeatureConfig 特征
z = MLP(features)        # [128, 64] + ReLU + Dropout
out = gas_head(z)        # 保持闭包
loss = weighted_component_mse
```

目的：检验 N2 信号能否进入一个最简 DL 头。这是判别诊断，不是候选主线。实现成本以"能跑通、可比"为限，不追求调优。

### P0.3 日志确认

确认每个 run 落盘以下内容（现有 `metrics_live.jsonl` 已逐 epoch 记录，summary CSV 已含 `x_n2_r2`）：

- 逐 epoch 的 train/val loss（判断过拟合）
- val/test/extrapolation 三 split 的 per-component R2 与 `x_n2_r2`
- `sum_abs_error`、`macro RMSE`、overall R2

若 per-component R2 未落盘，先补齐再开跑——否则无法判读候选 A/B/E。

### P0.4 配置改写

把现有 `phase_window_tcn_ablation.json`（当前是 gas_free/split/deep）改为第一批诊断批：

- 保留 `phase_window_tcn_gas_free`（基线）
- 新增 `phase_window_tcn_gas_varweight`（loss=weighted_component_mse）
- 新增 `phase_window_tcn_gas_free_varweight`（loss=weighted_free_component_mse）
- 新增 `phase_window_tcn_handcraft_mlp`（手工特征 MLP）

把旧的 `split`、`deep`、`split_deep` 迁到 Phase 2 配置（可命名 `phase_window_tcn_ablation_structure.json`），followup 配置保持只含 `split_deep`。training 块（epochs 300 / batch 16 / adamw / lr 1e-4 / weight_decay 1e-4 / early_stopping patience 25 / reduce_on_plateau / AMP）保持不变，作为固定正则口径。

### 决策门 G0（进入 Phase 1 的条件）

- 全部新配置 `--dry-run` 通过
- 新 loss 单测通过
- per-component 日志确认可用

任一不满足 → 留在 Phase 0，不开跑。

## 3. Phase 1：诊断批（必做）

### 实验清单

| 实验名                                   | loss                        | 关键改动             | 对应候选  |
| ------------------------------------- | --------------------------- | ---------------- | ----- |
| `phase_window_tcn_gas_free`           | free_component_mse          | 无（基线）            | —     |
| `phase_window_tcn_gas_varweight`      | weighted_component_mse      | N2 被直接监督 + 方差加权  | A + B |
| `phase_window_tcn_gas_free_varweight` | weighted_free_component_mse | 仅自由组分方差加权，N2 仍残差 | A     |
| `phase_window_tcn_handcraft_mlp`      | weighted_component_mse      | 手工特征 + 小 MLP     | C     |

### 运行步骤

1. `--dry-run` 全部配置，确认计划无误。
2. 实跑（同 seed 20260615，一次性跑完整批）。
3. 收集 summary CSV、各 run 的 `metrics.json`、`metrics_live.jsonl`。
4. 按下表判读。

### 判读表（结果 → 病因 → 分流）

| 现象                                           | 病因定位               | 分流动作                                                |
| -------------------------------------------- | ------------------ | --------------------------------------------------- |
| `gas_varweight` 让 N2 达到 Tier 1+              | 损失尺度 + N2 监督（A+B）  | 采用加权损失为新 DL 基线；是否进 Phase 2 取决于是否想进一步逼近 ML           |
| 仅 `gas_free_varweight` 改善、`gas_varweight` 更好 | 损失尺度为主，直接监督 N2 是关键 | 同上，确认 4 组分加权为默认                                     |
| 只有 `handcraft_mlp` 达 Tier 1+                 | 从原始波形学特征是瓶颈（C）     | **跳过** Phase 2 原始波形 split/deep；转向特征注入混合模型，或接受 ML 主线 |
| 四个 arm 的 N2 都 ≤ Tier 0                       | 不在损失/监督/特征学习层      | 进入 Phase 2 结构消融                                     |
| 任意 arm overall R2 上升但 N2 仍负                  | 损失尺度被进一步确认         | 检查 per-component：是否大组分变好、N2 没动                      |

### 决策门 G1

- 若已定位到损失/监督主因且 N2 达标 → 记录结论，DL 瓶颈定位完成；结构消融变为可选。
- 若 `handcraft_mlp` 指向特征学习 → 不在原始波形 encoder 上做结构消融。
- 若全部无信号 → 进入 Phase 2。

## 4. Phase 2：结构消融（条件触发）

**进入条件**：Phase 1 的损失/监督修正未能让 N2 改善，且 `handcraft_mlp` 表明特征学习不是唯一瓶颈。

**固定正则口径**：training 块保持与 Phase 1 一致。若 Phase 1 的 train/val 曲线确认过拟合，先把 `tcn_dropout`、`weight_decay` 调强（如 dropout 0.25→0.4、weight_decay 1e-4→1e-3）作为统一基线，再做结构对比，避免把"加容量导致的过拟合"误读成"结构无效"。

| 实验名                                    | 改动                              | 说明                                          |
| -------------------------------------- | ------------------------------- | ------------------------------------------- |
| `phase_window_tcn_gas_free_split`      | `share_window_encoder=false`    | encoder 参数 ×3，重点监控 val 是否更早过拟合              |
| `phase_window_tcn_gas_free_deep`       | `tcn_channels=[64,64,64,64,64]` | 已有 mean/max 全局池化，deep 是较弱变量                 |
| `phase_window_tcn_gas_free_split_deep` | split + deep                    | 仅当 split 或 deep 任一达 Tier 1+ 时跑（followup 配置） |

运行顺序：先跑 `split` 与 `deep`（同 seed），按 G2 决定是否跑 `split_deep`。

### 决策门 G2

- `split` 或 `deep` 任一达 Tier 1+ → 跑 `split_deep`，再进 Phase 3。
- 都 ≤ Tier 0 → **收束 DL 线**，不跑 followup（见第 7 节）。

## 5. Phase 3：融合与对数比（条件触发）

**进入条件**：Phase 2 出现正信号但未达标。

| 实验名                               | 改动                                                            | 目的                         |
| --------------------------------- | ------------------------------------------------------------- | -------------------------- |
| `phase_window_tcn_gas_free_gated` | gated fusion                                                  | 窗口级自适应加权是否优于纯 concat       |
| `phase_window_tcn_gas_free_attn`  | pooled 特征上的轻量 attention                                       | 窗口交互是否有帮助                  |
| `phase_window_tcn_ilr`            | `ilr_mse` + `target_transform='ilr_n2_first'` + `out_dim=3` 头 | 对数比目标是否缓解闭包残差（受限对照，不进正式主线） |

### 决策门 G3

- 任一达验收标准（见第 6 节强信号档） → 作为 DL 结果记录，并与 ML 诚实对比。
- 都不达标 → 收束 DL 线。

## 6. 验收标准（统一口径）

### 6.1 指标定义

- **首要**：`test x_N2 R2` 与 `extrapolation x_N2 R2`，必须**同时**改善才算有效。
- **闭包**：`sum_abs_error`（预测四组分和与 100 的偏差）。
- **非退化**：H2 / CO2 / CH4 三组分各自的 test R2。
- **过拟合**：train 与 val loss 曲线、best epoch 位置。
- 所有 R2 在原始百分比空间计算。

### 6.2 分层门槛（相对同 seed 的 `phase_window_tcn_gas_free` 基线）

| 档位          | 条件                             | 含义                      |
| ----------- | ------------------------------ | ----------------------- |
| Tier 0（失败）  | test 或 extrapolation N2 R2 ≤ 0 | 该 arm 无效                |
| Tier 1（有信号） | 两 split N2 R2 均 > 0，且 < 0.3    | 病因/方向被确认，但 DL 大概率仍不及 ML |
| Tier 2（强信号） | 两 split N2 R2 均 ≥ 0.3          | 值得继续推进到下一批              |

> 门槛数值为建议值，可按需要调整。Ridge 已在 0.71/0.72，Tier 2 的 0.3 仍远低于 ML，跨过只代表方向成立，不代表 DL 可作正式主线。

### 6.3 闭包约束

- `sum_abs_error` 必须维持在当前量级（≈2e-6），上限建议 < 0.01。
- 任何破坏闭包的 arm（如不配约束的独立 head）直接判不通过。

### 6.4 非退化约束

- H2 / CO2 / CH4 每个组分的 test R2 相对基线下降不超过 0.02，且三者无系统性同向下降。
- 若 N2 改善以明显牺牲其他组分为代价，判为不通过。

### 6.5 过拟合判定

- best epoch 过早（如 < 10）且 train loss 持续下降而 val loss 回升 → 判为过拟合，先调正则再比结构。
- 若加容量（split/deep）使 best epoch 更早、val 更差 → 支持"增容方向错误"的结论。

### 6.6 比较基准说明

- **同批可比**：同一批内 arm 共享 seed 20260615。
- **跨方法参照**：Ridge（seed 20260603）是外部固定基准，不与 DL 同 seed，只用于判断差距，不作同 seed 严格对照。
- 不得把不同 seed 或历史归档结果直接混作同批对照。

## 7. 决策树与之后的处理

### 7.1 完整决策树

```text
Phase 1 诊断批
├─ gas_varweight 达 Tier 1+        → 病因=损失/监督
│   ├─ 想逼近 ML → 用加权损失为新基线，进 Phase 2
│   └─ 不必逼近 → 记录结论，DL 定位完成，正式主线仍用 Ridge
├─ 仅 handcraft_mlp 达 Tier 1+      → 病因=特征学习
│   ├─ 转向特征注入混合模型（方案文档 10.x）
│   └─ 或直接收束 DL，接受 ML 主线
└─ 全部 ≤ Tier 0                    → 进 Phase 2 结构消融
    ├─ split/deep 达 Tier 1+        → 跑 split_deep → Phase 3
    │   ├─ Phase 3 达强信号 → 记录 DL 结果，与 ML 诚实对比
    │   └─ Phase 3 不达标   → 收束 DL
    └─ split/deep ≤ Tier 0          → 收束 DL
```

### 7.2 收束 DL 线时的处理（任一收束分支触发）

1. **归档**：把本轮全部 run 整理到 `outputs/archive/phase_window_tcn_diagnosis_<日期>/`，含 summary CSV、各 run metrics、配置、结论说明。
2. **正式结果**：正式主线固定为 `ridge_multiwindow_all_modalities`（test N2 R2=0.7121 / extrapolation=0.7247）。
3. **DL 定位为负结果证据**：PhaseWindowTCN 全线（MVP、gas_head、free_component_mse、本轮诊断与消融）作为 DL 负结果记录，写清"病因定位在损失/监督还是结构/特征学习"。
4. **文档同步**（重要，避免后续 AI 误读）：
   - `docs/AI_CONTEXT_GUIDE.md`：当前仍写"第一批=split/deep"，需更新为"诊断优先 + 病因结论"。
   - `README.md` / `docs/IMPLEMENTATION_PLAN.md`：更新实验状态与配置入口。
   - RecallLoom `rolling_summary` / 里程碑：**经 helper 追加，不手改** `.recallloom` 文件。
5. **提交**：按 Conventional Commits，只纳入相关改动，排除无关输出与 RecallLoom 状态文件。

### 7.3 继续 DL 时的处理

- 把验证有效的 arm（如加权损失）设为新的 DL 默认配置，并在方案文档与 AI_CONTEXT_GUIDE 标注。
- 进入下一批前，重跑一次基线 + 新基线的同 seed 对照，确认改善稳定（非单次波动）。
- 每跨一个决策门，更新一次 RecallLoom 里程碑（经 helper）。

## 8. 记录与可复现

- **seed**：同批固定（当前 20260615）；记录在 run_config.json。
- **输出目录**：沿用 `outputs/runs/<experiment_name>/<run_name>/`，归档用 `outputs/archive/<名称>_<日期>/`。
- **每个 run 必存**：`run_config.json`、`metrics.json`、`metrics_live.jsonl`、`best_checkpoint.pt`、per-component R2。
- **环境**：在归档目录存 `pip freeze` 输出。
- **数据**：记录 `data/wv4-formal-hitran-standard-6000` 的来源/哈希，确认与 Ridge 基准同一份数据。
- **决策留痕**：每个决策门的判读结果与分流原因写入归档目录的结论说明。

## 9. 风险与回退

| 风险                    | 表现                    | 回退                                      |
| --------------------- | --------------------- | --------------------------------------- |
| 加权损失实现错误              | 权重方向反了、N2 反而更差        | 先用单测 + 小规模 dry-run 验证权重数值               |
| 方差加权数值不稳              | 大权重导致梯度不稳、训练发散        | 改用对数空间标准化或夹住权重上限，参照 SLAW 思路             |
| handcraft_mlp 复用特征不一致 | 与 Ridge 特征口径不同导致结论失真  | 直接复用 `MLFeatureConfig`，核对特征维度与 Ridge 一致 |
| 把不同 seed 结果混比         | 误判某 arm 有效/无效         | 严格同批同 seed；Ridge 仅作参照                   |
| 过拟合误读为结构无效            | split/deep 看似无效，实为过拟合 | 先固定并调强正则，再比结构                           |
| 改文件名/引用断裂             | README 等引用失效          | 方案文档文件名保持不变；新增本执行文档而非改名                 |

---

> 一句话定位：本文件把"诊断优先"的方案落成可执行步骤——先用四个低成本 arm 定位 N2 负 R2 的病因，再按病因决定是否动结构，最后无论结果如何都明确归档与文档同步，正式主线在 DL 未达标时保持 `ridge_multiwindow_all_modalities`。
