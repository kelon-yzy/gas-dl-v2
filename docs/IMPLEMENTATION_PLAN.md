# v4 正式实验 实施计划

> 基于 `新项目PLAN.md` 6 项改进要求 + `docs/新项目目标架构说明.md` 目标架构，当前状态评估后制定。

## 当前状态总览

| 子系统               | 状态                                              | 完成度 |
| ----------------- | ----------------------------------------------- | --- |
| `src/sim`         | 最小垂直切片完成（core/generation/packaging/validation）  | 70% |
| `src/dl/data`     | P0 完成：V4BenchmarkDataset + splits + scalers     | 40% |
| `src/dl/models`   | P0 完成：BaseRegressor + CNN1DRegressor + registry | 10% |
| `src/dl/training` | 未迁移                                             | 0%  |
| `src/ml`          | 仅有占位 `__init__.py`                              | 0%  |
| `src/pipeline`    | 仅有 layout + generate_benchmark CLI              | 15% |
| `configs/`        | 全部 `.gitkeep`，无实际配置                             | 0%  |
| `experiments/`    | 仅有 `.gitkeep`                                   | 0%  |
| `tests/`          | 58 个测试（sim + dl + pipeline）                     | 40% |

## PLAN 6 项问题对照

| #   | 问题                                                 | 状态                            |
| --- | -------------------------------------------------- | ----------------------------- |
| 1   | 删除 base_condition_id / noise_seed 旧列，mixture_id 唯一 | ✅ v4 sim 已落地                  |
| 2   | TCN 感受野较短                                          | ❌ 模型层未开始                      |
| 3   | 时间步分布不合理                                           | ⚠️ phase 仍为固定四等分              |
| 4   | LHS 采样 + Dropout 语义归位                              | ⚠️ LHS 已完成，Dropout 待 training |
| 5   | 文件命名过长，结果混乱                                        | ✅ output 分区 + run 契约已定义       |
| 6   | 光学变量显式分层建模                                         | ⚠️ 吸收层已有，谱线重叠/交叉敏感未建模         |

---

## Phase 2: sim 完善（P1 事项）

### 2.1 LHS 采样替换 `conditions.py` ✅

- **状态**：已完成。`generate_condition_rows` 默认使用 `scipy.stats.qmc.LatinHypercube` 进行 3D 空间填充采样
- **实现**：H₂/CO₂/N₂ 三自由度 LHS，CH₄ 由减法得到，约束保持 CH₄ ≥ 40%
- **后向兼容**：`sampling_strategy="random"` 可切回纯随机采样
- **manifest 记录**：`sampling_strategy` 字段写入 manifest.json

### 2.2 声程候选配置化 ✅

- **状态**：已完成。声程候选已从 `slow.py` 硬编码常量迁入 `BenchmarkGenerationSpec.path_lms`
- **CLI 暴露**：`generate_benchmark.py` 已增加 `--path-lms` 参数，格式为逗号分隔浮点数
- **manifest 记录**：声程候选写入 `manifest.json` 和 `metadata/waveform_spec.json`
- **默认值**：`(0.20, 0.25, 0.30, 0.35, 0.40)` 作为正式默认声程候选

### 2.3 声学物理单元测试

- 对 `acoustic_physics.py` 中 `hidden_sound_speed_v2`、`hidden_attenuation_v2`、`main_sensor_features` 编写确定性回归测试
- 固定输入 → 固定输出，防止物理公式意外变更

### 2.4 光谱交叉敏感度显式建模

- 新增 `src/sim/generation/optical_crosstalk.py`
- 定义交叉敏感矩阵参数（CH₄ 通道对 CO₂ 的响应系数、CO₂ 通道对 CH₄ 的响应系数）
- 在 `main_sensor_features` 中插入交叉层

---

## Phase 3: DL 模型扩充

### 3.1 TCN 回归器

- 文件：`src/dl/models/tcn.py`
- 从 V3 迁移 `TCNRegressor`，适配 v4 接口
- 记录感受野参数

### 3.2 LSTM / GRU 回归器

- 文件：`src/dl/models/lstm.py`、`src/dl/models/gru.py`
- 统一接口：`forward(x) -> Tensor[batch, out_dim]`

### 3.3 多模态融合模型

- 文件：`src/dl/models/multimodal_fusion.py`
- 慢变量 CNN1D encoder + 波形 encoder → concat → head
- 适配 Dataset 多模态输出（slow + ultrasonic + fiber_mic 拼接后的通道维）

### 3.4 Transformer encoder

- 文件：`src/dl/models/transformer.py`
- 标准 TransformerEncoder + pooling → head

---

## Phase 4: 训练模块

### 4.1 训练配置

- 文件：`configs/train/adamw-cosine.yaml`（Hydra 结构）
- 字段：`optimizer`、`scheduler`、`loss`、`batch_size`、`epochs`、`grad_clip`

### 4.2 Loss 与 Metrics

- 文件：`src/dl/training/losses.py`
  - `WeightedMSELoss`（组分权重可调）
  - `SumConstraintLoss`（总和=100% 约束）
- 文件：`src/dl/training/metrics.py`
  - `macro_RMSE`、`macro_MAE`、`per_component_R2`、`sum_error`

### 4.3 Trainer

- 文件：`src/dl/training/trainer.py`
- 标准训练循环：epoch → batch → forward → loss → backward → step
- Early stopping + checkpoint 保存
- 输出契约：`config.json`、`summary.json`、`component_metrics.csv`、`predictions.csv`、`train_log.csv`、`report.md`

### 4.4 Seed 管理

- 文件：`src/dl/training/seed.py`
- `set_seed()` 统一入口

---

## Phase 5: ML 模块

### 5.1 特征打包

- 文件：`src/sim/generation/feature_package.py`
- 从 benchmark 数据导出传统 ML 特征表（慢变量统计量 + 声学特征）
- 输出到 `data/<slug>/feature_package/`

### 5.2 传统模型训练

- 文件：`src/ml/train.py`
- SVR、Ridge、RandomForest 基线
- 共享 split 和 metrics 契约

---

## Phase 6: 实验编排与报告

### 6.1 Experiment 配置

- 文件：`configs/experiment/` 下各实验 yaml
- 组合 data/model/train/eval 四类配置

### 6.2 批量运行入口

- 文件：`experiments/run_baseline.py`
- 读实验配置 → 生成 benchmark（如需要）→ 训练 → 评估 → 写报告

### 6.3 汇总与绘图

- 文件：`src/pipeline/summary.py`
- 跨 run 汇总表、model comparison 表

---

## 实施优先级

| 优先级       | 范围                                | 预计文件数    | 依赖  |
| --------- | --------------------------------- | -------- | --- |
| **P0** ✅  | dl data + cnn1d 最小可用链路            | 10       | —   |
| **P1** ✅  | LHS 采样完成                          | 1 (done) | P0  |
| **P1** 🔜 | 声程配置化 + acoustic 测试               | 3        | P0  |
| **P2**    | TCN + LSTM/GRU + Transformer 模型   | 5        | P0  |
| **P3**    | training 模块（loss/metrics/trainer） | 5        | P0  |
| **P4**    | 光谱交叉敏感建模                          | 2        | —   |
| **P5**    | ML 特征导出 + 传统模型                    | 4        | P3  |
| **P6**    | 配置落地 + 实验编排 + 报告                  | 5        | P3  |

## 文件规模预估

| 已落地     | 待新增     | 预估总行数   |
| ------- | ------- | ------- |
| ~1600 行 | ~1800 行 | ~3400 行 |

## 完成标准

1. `python -m pytest tests/` 全部通过
2. 端到端链路可运行：配置 → benchmark 生成 → DL 数据加载 → 训练 → 评估 → 报告
3. 所有 PLAN 6 项问题有明确对应实现或归档决策
