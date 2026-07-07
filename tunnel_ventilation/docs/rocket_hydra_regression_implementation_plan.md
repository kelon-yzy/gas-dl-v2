# MiniRocket / MultiRocket / Hydra 回归基线实施规划

> 目标：针对 tv3 掘进通风数据集，在当前端到端 DL 训练失效的前提下，优先实现固定卷积核时序特征 + Ridge / ElasticNet / 小 MLP 的稳健回归链路，用于判断超声波形是否真实携带 O2 / N2 可辨识信号。

## 0. 实施进度（截至 2026-07-06）

阶段 A 已完成最小可运行落地，但正式集结果还未回填。

已完成：

- `tv3/ml/rocket_features.py`
  - 实现 `physics_stats_v1` 特征缓存
  - 覆盖 full / phase / early 三类窗口统计
  - 支持 `slow + ultrasonic_tof_s + ultrasonic_tof_observed_s + ultrasonic_peak_index + ultrasonic_sound_speed_m_per_s + ultrasonic_sound_speed_estimated_m_per_s + ultrasonic_alpha_true_npm + ultrasonic_tof_quality + ultrasonic_tof_accepted`
  - 写出 `feature_matrix_{split}.npy`、`feature_names.json`、`feature_manifest.json`
- `tv3/ml/rocket_training.py`
  - 实现 `StandardScaler + RidgeCV`
  - 保留 `ridge_closed_form` 对照路径
  - 输出 `train / val / test / extrapolation` 指标与 top feature group 诊断
- `tv3/pipeline/run_tv3_rocket_baseline.py`
  - 支持 `--feature-set physics_stats --head ridgecv`
- `configs/tv3_rocket_ridge.json`
  - 提供 R0 默认配置
- 测试
  - `tests/test_rocket_features.py`
  - `tests/test_tv3_rocket_pipeline.py`
  - smoke 已通过，当前验收命令：

```powershell
python -m pytest tests/test_rocket_features.py tests/test_tv3_rocket_pipeline.py -v
```

未完成：

- 正式集 R0 指标回填
- MiniRocket / MultiRocket 波形特征
- ElasticNetCV / 小 MLP / Hydra

## 1. 结论先行

推荐第一版落地顺序：

```text
tv3-formal-6000
  -> ultrasonic waveform / tof / sound_speed / slow channels
  -> MiniRocket-style fixed kernels
  -> sequence-level pooling
  -> StandardScaler
  -> RidgeCV / ElasticNetCV / small MLP
  -> val / test / extrapolation per-component metrics
```

优先级：

| 优先级 | 方法                                          | 用途                | 是否第一版必做 |
| ---:| ------------------------------------------- | ----------------- |:-------:|
| P0  | TOF / sound_speed 序列特征 + Ridge / ElasticNet | 最小可验证声学信号         | 是       |
| P0  | MiniRocket-style 超声帧特征 + RidgeCV            | 稳健主 baseline      | 是       |
| P1  | MultiRocket-style 多池化统计                     | 增强 O2 / N2 弱信号    | 是       |
| P1  | 小 MLP 回归头                                   | 检查轻量非线性增益         | 是       |
| P2  | Hydra-style competing kernels               | 若 MiniRocket 有效再上 | 否，第二轮   |

关键判断：服务器上 6000 序列仍出现 DL 最佳 epoch 为 1，说明不能继续把问题归因于 600 小样本；当前首要任务是用不依赖深层反传稳定性的固定特征路线，把“数据有无 O2 信号”和“端到端训练是否坏掉”分开。

## 2. 背景与触发条件

### 2.1 tv3 数据与任务

- 数据集：服务器侧 `tv3-formal-6000`，每条 sequence 512 timesteps。
- 目标：直接预测 `x_CO2`、`x_O2`、`x_N2`，不使用闭包残差头。
- 主要输入：
  - `slow`: NDIR CO2、TCS、温度、压力、湿度、声程、阶段位置等慢通道。
  - `ultrasonic`: 每 timestep 一帧 200 kHz 超声波形。
  - 已生成声学辅助数组：`ultrasonic_tof_s`、`ultrasonic_tof_observed_s`、`ultrasonic_sound_speed_estimated_m_per_s`、`ultrasonic_peak_index`、`ultrasonic_alpha_true_npm` 等。
- 难点：O2 无直接光学通道，主要依赖 O2 / N2 声速差和 TCS 弱差异。

### 2.2 当前失效模式

已知事实：

- slow-only Ridge 能预测 CO2，但 O2 接近均值预测。
- 端到端 TCN / fusion 在 600 序列下不收敛。
- 用户补充：服务器 6000 序列下当前 DL 算法仍训练失效，最佳 epoch 为 1。

推论：

1. 仅扩大样本量未解决训练问题，当前端到端 DL 链路存在训练动力学或输入尺度问题。
2. 必须先引入固定特征路线，避免所有结论都被 DL 训练崩溃污染。
3. 如果 ROCKET 类特征也无法让 O2 R2 超过 0.50，则应优先判断现有通道物理可辨识性不足，而不是继续换大模型。

## 3. 不变量与边界

1. 不把 `x_N2` 作为残差目标，三个组分全部直接回归。
2. 不使用 `compositional_mse`、`ilr_mse`、`free_component_mse` 或 target transform。
3. 不把 `mixture_id` 回退或重写为 `sequence_id`。
4. 不用训练失败时的静默 fallback；任何 NaN、维度不匹配、缓存缺失都应直接失败。
5. 不先引入新传感器；本方案只验证现有 slow + ultrasonic 的信号极限。
6. 不一次性加载完整 6000 x 512 x 5000 波形到内存；必须 chunk 流式生成特征缓存。
7. 先实现可复现、可审计、可缓存的特征管线，再接回归头。

## 4. 算法路线

### 4.1 P0-A：物理序列统计基线

输入数组：

- `ultrasonic_tof_s`
- `ultrasonic_tof_observed_s`
- `ultrasonic_sound_speed_estimated_m_per_s`
- `ultrasonic_peak_index`
- `ultrasonic_alpha_true_npm`
- slow channels

每条 sequence 提取：

- full window: mean, std, min, max, range, first, last, delta, slope。
- phase windows: baseline, exposure, steady, recovery。
- early windows: first 25%, 50%, 75%。

模型：

- `RidgeCV`
- `ElasticNetCV`
- 现有 closed-form `RidgeRegressor` 作为最小依赖对照。

目的：

- 判断无需 raw waveform 时，TOF / sound_speed 是否已足够解释 O2。
- 如果 P0-A 的 O2 R2 仍约 0，raw waveform 路线才有必要。

### 4.2 P0-B：MiniRocket-style 超声帧特征

输入形态：

```text
one sequence ultrasonic: (T=512, L=5000)
```

不建议把 `(512, 5000)` 直接 flatten 成长序列。第一版按“帧内波形 + 跨 timestep 池化”处理：

1. 对每个 timestep 的 5000 点超声帧应用固定 1D 卷积核。
2. 每个 kernel 对每帧输出 PPV 与 max 两类统计。
3. 对 512 帧的 kernel 统计再做 sequence pooling：mean, std, min, max, slope。
4. 拼接 slow / TOF / sound_speed 统计。

第一版 MiniRocket-style 不是完整复刻论文实现，而是项目内可控的固定核特征器：

- kernel length: `{7, 9, 11}`
- dilation: log-spaced，覆盖短窗和长窗。
- weights: zero-mean fixed random kernels，seed 固定。
- bias: 从训练集少量帧的卷积分位数估计。
- features: PPV 为主，max / mean_abs 作为补充。

这样做的原因：

- 不引入额外大依赖。
- 避免把 6000 序列全部波形一次性展开。
- 直接针对超声 TOF 微小位移与衰减形态。

### 4.3 P1：MultiRocket-style 多池化增强

在 P0-B 基础上增加：

- first-order difference waveform。
- 每个 kernel 输出 PPV、max、mean、std。
- phase-aware pooling：baseline / exposure / steady / recovery 分别池化，再拼 full-window 池化。

收益预期：

- 对 O2 / N2 更敏感，因为 O2 信号可能体现在响应阶段变化，而不是单帧绝对形态。
- 比端到端 TCN 更稳，因为只有回归头参与训练。

风险：

- 特征维度可能过高。第一版控制在 2k 到 20k 维，不超过 Ridge / ElasticNet 可承受范围。

### 4.4 P2：Hydra-style competing kernels

Hydra 作为第二轮实现：

- 多组 kernel 在同一 receptive field 内竞争。
- 每组保留 winning pattern 的计数或强度。
- 适合捕获“哪个 acoustic motif 出现”而不是连续卷积幅值。

触发条件：

- MiniRocket / MultiRocket 已证明 O2 R2 有明显正信号，例如 val O2 R2 > 0.30。
- 但仍低于验收线，需要更强非线性模式特征。

不作为第一版必做的原因：

- 实现更复杂。
- 当前更急的是验证训练链路和物理信号，而不是追求最终 SOTA。

## 5. 回归头设计

### 5.1 RidgeCV

用途：主 baseline。

参数：

```text
alphas = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1, 3, 10, 30, 100]
```

要求：

- 所有特征只用 train split 拟合 `StandardScaler`。
- `RidgeCV` 只在 train 内交叉验证，不看 val / test / extrapolation。
- 输出每个目标的 coef norm 与 top feature group，辅助诊断 O2 是否用到 acoustic 特征。

### 5.2 ElasticNetCV

用途：特征选择与稀疏性诊断。

参数：

```text
l1_ratio = [0.05, 0.1, 0.3, 0.5, 0.7]
alphas = logspace(-4, 1, 20)
```

判断：

- 如果 ElasticNet O2 与 Ridge 接近，说明信号可由少量稳定特征解释。
- 如果 ElasticNet 明显差于 Ridge，说明 O2 信号分散在大量弱特征上，后续更适合 Ridge / MLP。

### 5.3 小 MLP

用途：检查轻量非线性增益，不替代 Ridge 主判断。

结构：

```text
input_dim
  -> Linear(256) + GELU + Dropout(0.10)
  -> Linear(128) + GELU + Dropout(0.10)
  -> Linear(3)
```

训练：

- loss: plain MSE 或 weighted component MSE `[1, 2, 1]`。
- optimizer: AdamW。
- lr: `1e-4` 起步，不使用当前 fusion 的 `1e-3` 默认值。
- epochs: 100。
- early stopping: patience 10。
- batch size: 64 或 128。

失败判据：

- 如果 best epoch 仍为 1，且 Ridge / ElasticNet 正常，则小 MLP 的训练配置仍有问题。
- 如果 Ridge / ElasticNet 也无正信号，则不要继续调 MLP。

## 6. 实现文件规划

### 6.1 新增文件

| 文件                                        | 职责                                                         |
| ----------------------------------------- | ---------------------------------------------------------- |
| `tv3/ml/rocket_features.py`               | 固定 kernel 生成、chunk 波形读取、MiniRocket / MultiRocket 特征提取、缓存写入 |
| `tv3/ml/rocket_training.py`               | RidgeCV / ElasticNetCV / small MLP 训练与评估                   |
| `tv3/pipeline/run_tv3_rocket_baseline.py` | tv3 专用编排脚本，生成缓存并跑实验矩阵                                      |
| `configs/tv3_rocket_ridge.json`           | MiniRocket + RidgeCV 配置                                    |
| `configs/tv3_rocket_elasticnet.json`      | MultiRocket + ElasticNetCV 配置                              |
| `configs/tv3_rocket_mlp.json`             | MultiRocket + 小 MLP 配置                                     |
| `tests/test_rocket_features.py`           | kernel、shape、缓存、可复现性测试                                     |
| `tests/test_tv3_rocket_pipeline.py`       | 小数据 smoke pipeline 测试                                      |

阶段 A 当前实际职责更精确地说是：

- `tv3/ml/rocket_features.py`：`physics_stats_v1` 特征缓存与 split 对齐校验
- `tv3/ml/rocket_training.py`：`RidgeCV` / `ridge_closed_form` 的首轮训练与评估
- `tv3/pipeline/run_tv3_rocket_baseline.py`：tv3 `physics_stats` 实验入口
- `configs/tv3_rocket_ridge.json`：R0 配置
- `tests/test_rocket_features.py`、`tests/test_tv3_rocket_pipeline.py`：阶段 A smoke 测试

### 6.2 可复用现有文件

| 文件                               | 复用点                                  |
| -------------------------------- | ------------------------------------ |
| `tv3/ml/features.py`             | slow / waveform 统计特征、multi-window 逻辑 |
| `tv3/ml/models.py`               | 现有 `RidgeRegressor` 可作为最小依赖对照        |
| `tv3/common/metrics.py`          | per-component R2 / MAE / RMSE        |
| `tv3/common/splits.py`           | split CSV 解析                         |
| `tv3/common/waveform.py`         | int16 / int32 waveform 路径解析          |
| `tv3/dl/models/handcraft_mlp.py` | 如接口合适，可复用小 MLP                       |

## 7. 特征缓存契约

缓存目录：

```text
data/<dataset>/features/rocket/
  minirocket_ultra_v1/
    feature_matrix_train.npy
    feature_matrix_val.npy
    feature_matrix_test.npy
    feature_matrix_extrapolation.npy
    feature_names.json
    feature_manifest.json
```

`feature_manifest.json` 必须记录：

- `dataset_slug`
- `schema_version`
- `sequence_count`
- `split_policy`
- `feature_builder`
- `kernel_seed`
- `kernel_count`
- `kernel_lengths`
- `dilations`
- `pooling_stats`
- `modalities`
- `slow_channels`
- `source_arrays`
- `created_at`

缓存校验：

- split 行数必须等于对应 split CSV 行数。
- feature_names 长度必须等于矩阵列数。
- 重新运行相同 seed 必须得到完全相同 feature matrix。
- 任何 NaN / Inf 直接报错。

## 8. 实验矩阵

第一轮只跑 6000 数据集，不再用 600 数据集做主结论。

| 实验 ID | 特征                                        | 模型           | 目的         |
| ----- | ----------------------------------------- | ------------ | ---------- |
| R0    | slow + TOF / sound_speed stats            | RidgeCV      | 最小声学信号验证   |
| R1    | MiniRocket ultrasonic + slow stats        | RidgeCV      | 主 baseline |
| R2    | MiniRocket ultrasonic + slow stats        | ElasticNetCV | 稀疏性诊断      |
| R3    | MultiRocket ultrasonic + slow + TOF stats | RidgeCV      | 多池化增强      |
| R4    | MultiRocket ultrasonic + slow + TOF stats | ElasticNetCV | 高维特征选择     |
| R5    | MultiRocket ultrasonic + slow + TOF stats | small MLP    | 轻量非线性增益    |
| R6    | Hydra-style ultrasonic + slow + TOF stats | RidgeCV      | 第二轮候选      |

推荐执行顺序：

```text
R0 -> R1 -> R3 -> R2 / R4 -> R5 -> R6
```

## 9. 指标与验收

### 9.1 主指标

每个 split 输出：

- overall MAE / RMSE / R2
- `x_CO2` MAE / RMSE / R2
- `x_O2` MAE / RMSE / R2
- `x_N2` MAE / RMSE / R2
- `sum_abs_error`
- `o2_bins` / `co2_bins` conditional metrics

### 9.2 决策阈值

| 结果                            | 决策                                                   |
| ----------------------------- | ---------------------------------------------------- |
| R0 O2 R2 > 0.30               | TOF / sound_speed 已有可用 O2 信号，优先做物理特征路线               |
| R1/R3 O2 R2 > R0 + 0.10       | raw ultrasonic waveform 提供额外 O2 信息，继续 ROCKET / Hydra |
| R3 O2 R2 >= 0.70              | 现有通道达到最低验收，可进入消融与稳定性验证                               |
| R3 O2 R2 0.50 到 0.70          | 现有通道边际可用，优先优化特征与 split，不急于加新传感器                      |
| R3 O2 R2 < 0.50               | 当前通道组合不足，转向 O2 专用通道评估                                |
| Ridge 正常但 MLP best epoch=1    | MLP 训练配置问题，不影响 ROCKET 特征有效性判断                        |
| Ridge / ElasticNet / MLP 全部失败 | 优先查数据集、split、label、scale，不换模型                        |

### 9.3 最低验收

第一轮实现验收：

- `tests/test_rocket_features.py` 通过。
- 32 序列 smoke 数据可在 60 秒内完成特征生成与 Ridge 训练。
- 6000 序列特征生成支持 chunk，不 OOM。
- R0 / R1 至少产出 val / test / extrapolation 指标 JSON。
- 输出中保留失败证据，不把失败 run 写成成功。

## 10. 实现步骤

### 阶段 A：P0 物理特征基线

1. [已完成] 新增 `tv3/ml/rocket_features.py` 中的物理序列特征读取函数。
2. [已完成] 支持从 `ultrasonic_tof_s`、`ultrasonic_sound_speed_estimated_m_per_s` 等数组提取 full / phase / early stats。
3. [已完成] 新增 `tv3/ml/rocket_training.py`，先接 RidgeCV。
4. [已完成] 新增 `run_tv3_rocket_baseline.py --feature-set physics_stats --head ridgecv`。
5. [已完成] 跑 32 序列 smoke，确认输出格式。
6. [未完成] 跑服务器正式数据集 R0 并回填结果。

### 阶段 B：MiniRocket-style 特征

1. 实现 deterministic kernel generator。
2. 实现 chunk waveform reader，支持 int16 / int32 + scale dequantization。
3. 每个 sequence 逐帧提取 kernel PPV / max，再跨 timestep pooling。
4. 写入 feature cache。
5. 接 RidgeCV 与 ElasticNetCV。
6. 跑 R1 / R2。

### 阶段 C：MultiRocket-style 增强

1. 增加 first-order difference waveform。
2. 增加 mean / std pooling。
3. 增加 phase-aware pooling。
4. 控制特征维度并记录 feature groups。
5. 跑 R3 / R4。

### 阶段 D：小 MLP

1. 复用或新增 tabular MLP。
2. 输入使用已缓存的 scaled feature matrix。
3. 固定低学习率 `1e-4`。
4. 跑 R5，检查是否还 best epoch=1。

### 阶段 E：Hydra-style 第二轮

1. 在 MiniRocket 有正信号后再实现。
2. 增加 competing kernel groups。
3. 输出 group-level activation counts。
4. 跑 R6。

## 11. 验证命令草案

本地 smoke：

```powershell
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark --output-root data --dataset tv3-rocket-smoke --sequences 32 --timesteps 64 --workers 1
python -m tv3.pipeline.run_tv3_rocket_baseline --dataset-dir data\tv3-rocket-smoke --feature-set physics_stats --head ridgecv --output-dir outputs\tv3_rocket_smoke\r0
python -m pytest tests/test_rocket_features.py -v
python -m pytest tests/test_tv3_rocket_pipeline.py -v
```

服务器正式：

```bash
python -m tv3.pipeline.run_tv3_rocket_baseline \
  --dataset-dir data/tv3-formal-6000 \
  --feature-set physics_stats \
  --head ridgecv \
  --output-dir outputs/tv3_rocket/r0

python -m tv3.pipeline.run_tv3_rocket_baseline \
  --dataset-dir data/tv3-formal-6000 \
  --feature-set minirocket_ultra_v1 \
  --head ridgecv \
  --chunk-size 64 \
  --output-dir outputs/tv3_rocket/r1

python -m tv3.pipeline.run_tv3_rocket_baseline \
  --dataset-dir data/tv3-formal-6000 \
  --feature-set multirocket_ultra_v1 \
  --head elasticnetcv \
  --chunk-size 64 \
  --output-dir outputs/tv3_rocket/r4
```

## 12. 风险与排查

| 风险               | 触发信号                 | 处理                                               |
| ---------------- | -------------------- | ------------------------------------------------ |
| 波形特征生成太慢         | R1 特征生成超过可接受时间       | 减 kernel_count，先只用 TOF 附近窗口                      |
| 特征维度太高           | RidgeCV 内存过高         | 降 kernel_count，使用 float32 cache，分组训练             |
| O2 仍不可预测         | R3 O2 R2 < 0.50      | 转 O2 专用通道，不继续堆 DL                                |
| CO2 下降明显         | CO2 R2 低于 slow Ridge | 检查 scaler、slow 通道漂移、split 对齐                     |
| MLP best epoch=1 | Ridge 正常但 MLP 崩      | 降 lr，检查 target scale 和 loss 权重                   |
| 所有头都失败           | R0 到 R5 都异常          | 检查 labels、split、feature row order、manifest 与实际数组 |

## 13. 文献依据

| 方法            | 依据                                                                                           | 用法                         |
| ------------- | -------------------------------------------------------------------------------------------- | -------------------------- |
| ROCKET        | Dempster et al. 2020, Data Mining and Knowledge Discovery, DOI: `10.1007/s10618-020-00701-z` | 固定随机卷积核 + 线性分类器，本项目迁移为回归特征 |
| MiniRocket    | Dempster et al. 2021, KDD, DOI: `10.1145/3447548.3467231`                                    | 更快更确定的卷积核特征                |
| Hydra         | Dempster et al. 2023, Data Mining and Knowledge Discovery, DOI: `10.1007/s10618-023-00939-3` | competing kernels，作为第二轮增强  |
| InceptionTime | Fawaz et al. 2020, Data Mining and Knowledge Discovery, DOI: `10.1007/s10618-020-00710-y`    | 多尺度卷积时序建模参考                |

## 14. 完成定义

本方案完成时，应具备：

1. 一个可复现的 ROCKET 特征缓存格式。
2. 至少 R0、R1、R3 三个实验完成。
3. RidgeCV / ElasticNetCV / small MLP 三类 head 至少两个可跑通。
4. 明确回答：当前 ultrasonic 是否提升 O2 / N2。
5. 明确回答：端到端 DL 失效是否与数据物理不可辨识无关。
6. 给出下一步分叉：继续 Hydra / 轻量卷积，或转 O2 专用通道。
