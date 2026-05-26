# v4 正式实验 实施计划

> 基于 `新项目PLAN.md` 6 项改进要求 + `docs/新项目目标架构说明.md` 目标架构，当前状态评估后制定。

## 当前状态总览

| 子系统               | 状态                                                              | 完成度 |
| ----------------- | --------------------------------------------------------------- | --- |
| `src/sim`         | 核心链路完成（core/generation/packaging/validation + HITRAN benchmark 默认接入 + 外部谱表 sanity check 入口 + spectral 默认配置 source-of-truth） | 90% |
| `src/dl/data`     | P0 完成：V4BenchmarkDataset + splits + scalers；lazy memmap 与 scaler 阈值契约已收敛 | 45% |
| `src/dl/models`   | P0+ 完成：BaseRegressor + CNN1DRegressor + TCNRegressor + registry | 25% |
| `src/dl/training` | loss/metrics 基础完成，trainer/checkpoint 待实现                       | 20% |
| `src/ml`          | 仅有占位 `__init__.py`                                              | 0%  |
| `src/pipeline`    | layout + generate_benchmark + HITRAN benchmark cache 预计算/对照 CLI + 外部谱表 sanity check CLI | 26% |
| `configs/`        | 已新增 `configs/data/spectral-defaults.json`（运行时 spectral 默认值 source-of-truth，含 `filter_source` 行业参考占位元信息），其余配置未落地 | 8%  |
| `experiments/`    | 仅有 `.gitkeep`                                                   | 0%  |
| `tests/`          | 132 个测试（sim + dl + pipeline）                                    | 62% |

## PLAN 6 项问题对照

| #   | 问题                                                 | 状态                                                 |
| --- | -------------------------------------------------- | -------------------------------------------------- |
| 1   | 删除 base_condition_id / noise_seed 旧列，mixture_id 唯一 | ✅ v4 sim 已落地                                       |
| 2   | TCN 感受野较短                                          | ⚠️ TCNRegressor 已落地并记录 receptive_field，感受野调参待实验配置化 |
| 3   | 时间步分布不合理                                           | ⚠️ phase 仍为固定四等分                                   |
| 4   | LHS 采样 + Dropout 语义归位                              | ⚠️ LHS 已完成，Dropout 待 training                      |
| 5   | 文件命名过长，结果混乱                                        | ✅ output 分区 + run 契约已定义                            |
| 6   | 光学变量显式分层建模                                         | ✅ NDIR 默认走 HITRAN 多气体光谱积分，empirical 交叉敏感度保留为显式兼容路径 |

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

### 2.3 声学物理单元测试 ✅

- **状态**：已完成。`tests/test_acoustic_physics_regression.py` 覆盖 `hidden_sound_speed_v2`、`hidden_attenuation_v2`、`main_sensor_features`
- **策略**：固定输入 → 固定输出，防止物理公式意外变更

### 2.4 光谱交叉敏感度显式建模 ✅

- **状态**：已完成。新增 `src/sim/generation/optical_crosstalk.py`
- **实现**：定义交叉敏感矩阵参数，CH4 通道响应 3.5% CO2 吸收，CO2 通道响应 1.2% CH4 吸收
- **接入**：在 `main_sensor_features` 中把原始吸收转换为 NDIR 观测吸收，并用于 NDIR 电压与饱和判断
- **测试**：`tests/test_optical_crosstalk.py` 覆盖默认矩阵和显式矩阵
- **边界**：该矩阵仅用于显式 `empirical_v1` 兼容路径；默认 `hitran_hapi_v1` 已由多气体滤光片积分表达交叉响应，不再叠加经验矩阵

### 2.5 HITRAN/PNNL 光谱积分支撑

- **状态**：HITRAN 主路径已接入 benchmark 默认生成；PNNL/NIST 外部真实数据对照仍待补实测/数据库文件，见 `docs/SPECTRAL_INTEGRATION_PLAN.md`
- **HITRAN 路线**：用 HAPI 下载 CH4/CO2/H2O line-by-line 数据，按温度、压力、浓度、光程和滤光片响应积分得到 NDIR 通道吸收
- **PNNL/NIST 路线**：读取定量 IR absorption coefficient 或 cross-section 谱，按浓度和光程缩放后做滤光片窗口积分
- **已落地**：新增 `src/sim/generation/spectral/` 本地积分核心、`tabulated_spectrum_v1` backend、外部定量谱表 CSV 读取与重采样、`hitran_hapi_v1` 适配层、谱线缓存、HITRAN 单位换算、默认滤光片/网格配置、HITRAN benchmark cache-only 接入、benchmark 专用预计算 CLI、empirical/HITRAN 小规模对照 CLI 和外部谱表 sanity check CLI，并在 manifest 中记录 `optical_absorption_backend` 与 HITRAN cache policy
- **契约收敛**：`configs/data/spectral-defaults.json` 已作为运行时默认值来源；HITRAN 主线不再调用 empirical `main_sensor_features` 计算 NDIR；同一 `(channel, grid)` 的谱表会准备成 `PreparedTabulatedSpectra` 后复用
- **真实下载验证**：本地已用 `hitran-api 1.3.0.0` 下载 CH4/CO2/H2O 在早期 `2960-3100 cm-1` 与 `2280-2410 cm-1` 两个窗口的 HITRAN 谱线；HAPI 原始表名已绑定波数窗口，避免不同通道缓存互相污染。当前默认窗口已扩大为 CH4 `2880-3180 cm-1`、CO2 `2250-2445 cm-1`，需重新预计算生成新缓存
- **后续实现**：当前默认滤光片已用行业参考占位（CH4 InfraTec LIM-262 3.3 μm/160 nm，对应 147 cm⁻¹ FWHM；CO2 InfraTec 4.26 μm/170 nm，对应 93 cm⁻¹ FWHM），默认 `hitran_grids` 已扩大到覆盖当前滤光片 `center ± FWHM`；仍需替换为目标传感器 TraceGas-HC-NDIR 实际 datasheet，并在 grid 变化后重下 HITRAN benchmark cache；获取真实 PNNL/NIST 或仪器定量谱表并运行 sanity check

---

## Phase 3: DL 模型扩充

### 3.1 TCN 回归器

- **状态**：已完成。新增 `src/dl/models/tcn.py`
- **实现**：从 V3 的因果卷积残差块思路迁移，适配 v4 `BaseRegressor`、`forward(x) -> Tensor[batch, out_dim]`、默认 `in_channels=8` / `out_dim=4`
- **注册**：`MODEL_REGISTRY["tcn"]` + `build_model({"name": "tcn", ...})`
- **感受野记录**：`TCNRegressor.dilations` 与 `TCNRegressor.receptive_field`
- **测试**：`tests/test_dl_models.py` 覆盖注册、参数透传、NCT forward、梯度、因果卷积长度保持和 dataset → TCN 前向

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
| **P1** ✅  | 声程配置化 + acoustic 测试               | 3        | P0  |
| **P2**    | TCN + LSTM/GRU + Transformer 模型   | 5        | P0  |
| **P3**    | training 模块（loss/metrics/trainer） | 5        | P0  |
| **P4** ✅  | 光谱交叉敏感建模                          | 2        | —   |
| **P4** 🔜 | TraceGas-HC-NDIR datasheet 替换占位 + 按需扩 HITRAN grid + PNNL/NIST 外部对照 | 3        | P4  |
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
