# D2 可微 TOF-PhaseNet 实施计划

> 状态：实施前规划
> 日期：2026-07-08
> 依据：[掘进通风项目记忆库.md](掘进通风项目记忆库.md)、[三组分检测深度学习新框架方案.md](三组分检测深度学习新框架方案.md)、现有 `tv3.dl` 训练入口与 D0 clean 6000 结果。

## 结论

D2 的实施重点不是继续加深普通 CNN 融合模型，而是把 raw ultrasonic waveform 中的 TOF / phase / peak 信息显式、可微地提取出来，再接入三组分直接回归。第一版应采用 fixed matched filter + envelope + softargmax lag 的最小闭环，先验证辅助 TOF 任务是否学得动；若 TOF auxiliary 无法优于现有 observed TOF 估计器，应停止 raw waveform 深挖，不继续叠复杂可学习滤波器。

目标基线固定为 D0-observed Ridge：val O2 R2=0.4226、test O2 R2=0.4571、extrapolation O2 R2=0.3708。D2 成功标准是 O2 在 val / test / extrapolation 均相对 D0-observed 提升超过 0.05，同时 CO2 不明显退化，且 `o2_bins` 不比 D0-observed 明显恶化。D0-oracle O2 R2=0.6025 只作为上限参照，不作为可部署 baseline。

## 背景事实

| 事实 | D2 含义 |
| --- | --- |
| D0-oracle 与 D0-observed 的 O2 gap 为 0.1799 | 主要缺口来自 oracle TOF / true sound speed / true alpha，而不是普通融合层容量不足。 |
| D0-tof-only O2 R2=0.3879，接近 observed 0.4226 | O2 可辨识性主要由 TOF 相关观测支撑。 |
| D0-no-tof 与 slow-only O2 R2 均约为 -0.0013 | 移除 TOF 后 O2 基本失效，D2 必须围绕 TOF / phase 做前端。 |
| D0-no-tcs O2 R2=0.4297，略高于 observed | TCS 热导信号边际贡献很小，第一版 D2 不应优先做复杂 thermal fusion。 |
| oracle 的 0.8% O2 bins R2 仍全负 | D2 不应承诺窄区间精细辨识；论文式结论仍是“可分辨大档位，窄区间精细辨识不足”。 |

## 不变量

1. tv3 只预测 `x_CO2`、`x_O2`、`x_N2` 三列，模型输出必须保持 `raw3` / `out_dim=3`。
2. 不使用 `gas_head`、`target_transform`、ILR / ALR、`free_component_mse` 或 N2 闭包残差头。
3. 可部署输入不得包含 `ultrasonic_tof_s`、`ultrasonic_sound_speed_m_per_s`、`ultrasonic_alpha_true_npm` 等仿真真值特征。
4. `ultrasonic_tof_s` 若为仿真真值，只能作为 D2 auxiliary target 或审计指标，不能进入模型 forward 输入。
5. `ultrasonic_tof_observed_s`、`ultrasonic_peak_index`、`ultrasonic_tof_quality`、`ultrasonic_tof_accepted` 可以作为辅助监督、mask 或 baseline 评估来源；若作为输入，必须单独声明为 observed-physics 路线，不能混入 D2 raw waveform 主线。
6. 训练与评估统一对比 D0-observed，不对比含 oracle 特征的 R0 / D0-oracle。
7. 只使用 clean tv3-formal-6000。旧本地 600 数据集含 V_NDIR_CH4 污染，不能用于 D2 结论。
8. 缺失 auxiliary target、slow channel metadata、waveform scale 时直接报错，不添加静默降级或模拟成功路径。

## 当前工程切入点

| 层 | 现状 | D2 需要的扩展 |
| --- | --- | --- |
| Dataset | `V4BenchmarkDataset` 输出 `(x, y)`，dict batch 已支持 `{"x": xs, ...}` 形式 | 增加 auxiliary target 读取，并保证 aux 只进 loss / metrics，不进 model input。 |
| Model registry | `MODEL_REGISTRY` 通过 `model` 字段构造模型 | 新增 `tof_phase_net` 注册名。 |
| Trainer | `loss_fn(pred, target)`，metrics 默认把 pred 当主三组分输出 | 支持结构化输出 `{"prediction": y_hat, "aux": ...}`，主指标只看 `prediction`。 |
| Loss | `weighted_component_mse` 已支持 tv3 raw3 | 新增 D2 composite loss：主三组分 loss + TOF / peak auxiliary loss。 |
| CLI | 配置驱动，已有 `dequantize_waveforms`、`normalize_waveforms`、`waveform_stats_features` | 增加 D2 配置项：`aux_target_arrays`、`aux_loss`、`aux_metrics`。 |
| 输出 | `metrics.json` 写主任务指标、conditional metrics、sum_abs_error | 增加 `auxiliary_metrics`，记录 TOF MAE、peak MAE、TOF correlation 和 observed baseline。 |

## 最小实现闭环

第一版只做 D2-minimal，不做完整研究路线：

```text
raw ultrasonic frame (B, T, 5000)
  -> fixed quadrature / matched filters
  -> envelope or correlation score over sample index
  -> softargmax index and peak sharpness
  -> per-timestep acoustic feature sequence
  -> TCN over timesteps
  -> raw3 component head
  -> auxiliary heads: tof_s, peak_index
```

输入建议沿用 v3_l2 的数据设置：

- `modalities`: `slow,ultrasonic`
- `dequantize_waveforms`: `true`
- `normalize_waveforms`: `true`
- `waveform_stats_features`: `log_std,log_max_abs`
- `slow_channels` in model: `9`（7 个 slow 通道 + 2 个 waveform stats）
- `ultrasonic_channels`: `5000`
- `fiber_mic_channels`: `0`

## 分阶段实施步骤

### D2-0：前置核查

目的：确认数据与 baseline 可以作为 D2 依据。

实施：

1. 在服务器或同步后的正式环境运行 `scripts/check_slow_channels.py data/tv3-formal-6000`。
2. 确认 `data/tv3-formal-6000/sequences/` 下存在：
   - `ultrasonic.npy` 或当前 `waveform_array_path()` 解析出的 ultrasonic 波形文件
   - `ultrasonic_scale.npy`
   - `ultrasonic_tof_s.npy`
   - `ultrasonic_tof_observed_s.npy`
   - `ultrasonic_peak_index.npy`
   - `ultrasonic_tof_quality.npy`
   - `ultrasonic_tof_accepted.npy`
3. 记录 auxiliary baseline：`MAE(ultrasonic_tof_observed_s, ultrasonic_tof_s)`、`corr(ultrasonic_tof_observed_s, ultrasonic_tof_s)`。
4. 确认 D0-observed metrics 文件可追溯：`outputs/tv3_d0/observed_ridge/metrics.json`。

产物：

- `outputs/tv3_d2/preflight/aux_baseline.json`
- 数据核查日志或命令输出摘录。

### D2-1：Dataset 增加 auxiliary target

影响文件：

- `tv3/dl/data/dataset.py`
- `tv3/dl/cli.py`
- `tests/test_d2_tof_phase_net.py`

配置草案：

```json
{
  "aux_target_arrays": {
    "tof_true_s": "ultrasonic_tof_s",
    "tof_observed_s": "ultrasonic_tof_observed_s",
    "peak_index": "ultrasonic_peak_index",
    "tof_quality": "ultrasonic_tof_quality",
    "tof_accepted": "ultrasonic_tof_accepted"
  }
}
```

实现要求：

1. `V4BenchmarkDataset` 增加 `aux_target_arrays` 参数。
2. 每个 aux array 从 `dataset_dir/sequences/{array_name}.npy` 读取，shape 必须是 `(N, T)`。
3. `__getitem__` 返回 `({"x": xs, "aux_targets": aux}, y)`。
4. `Trainer._unpack_batch()` 必须把 `aux_targets` 分离到 loss / metrics kwargs，不传给 model forward，避免真实 TOF 泄漏进输入。
5. 缺失文件、shape 不匹配、非有限值直接报错。

测试要点：

- smoke dataset 能读取 aux target。
- aux target 的 split 索引与主 label 对齐。
- 模型 forward 收到的 kwargs 不包含 `aux_targets`。

### D2-2：实现可微 TOF / phase 前端

影响文件：

- `tv3/dl/models/tof_phase_net.py`
- `tests/test_d2_tof_phase_net.py`

建议模块：

| 模块 | 作用 |
| --- | --- |
| `FixedQuadratureFilterBank` | 用固定正弦 / 余弦或短窗 matched filters 生成 I/Q 响应。 |
| `EnvelopeExtractor` | 计算 `sqrt(i^2 + q^2 + eps)`，得到 envelope。 |
| `SoftArgmaxLag` | 对 sample 维做温度控制 softargmax，输出连续 peak index。 |
| `PeakShapeFeatures` | 输出 peak sharpness、local energy、peak ratio 等辅助特征。 |

实现边界：

1. 第一版滤波器固定，不训练；确认 D2-minimal 站住后再开 learnable filter。
2. softargmax 输出先用归一化 sample index，再换算为 `tof_s = index / sample_rate_hz`。
3. `peak_index` loss 建议在归一化 index 上计算，metrics 再还原为 samples。
4. 不引入 SciPy 运行依赖；Hilbert-like magnitude 用 PyTorch conv 近似。
5. 每个 batch 只对 ultrasonic raw 段操作，slow / waveform stats 作为上下文分支输入。

测试要点：

- 合成单峰波形平移 `k` samples 后，softargmax index 单调跟随平移。
- forward 支持 `(B, T, 5009)`，输出主预测 `(B, 3)`。
- auxiliary 输出至少包含 `tof_s` 和 `peak_index`，shape 为 `(B, T)`。
- CPU 下小 batch 无 NaN / inf。

### D2-3：实现 `TOFPhaseNetRegressor`

影响文件：

- `tv3/dl/models/tof_phase_net.py`
- `tv3/dl/models/registry.py`

模型结构：

```text
input x (B, T, C)
  split slow_context (B, T, slow_channels)
  split ultrasonic_raw (B, T, 5000)
  acoustic_frontend -> per_timestep_features (B, T, F)
  slow_encoder -> slow_embedding (B, T, S)
  concat -> LayerNorm -> TCN
  sequence pooling: last + mean + max
  component_head -> raw3
  aux heads from frontend / fused features
```

构造参数建议：

```json
{
  "output_mode": "raw3",
  "slow_channels": 9,
  "ultrasonic_channels": 5000,
  "fiber_mic_channels": 0,
  "sample_rate_hz": 1000000.0,
  "carrier_hz": 200000.0,
  "softargmax_temperature": 0.05,
  "acoustic_feature_dim": 32,
  "slow_embedding_dim": 32,
  "tcn_channels": [64, 64, 64],
  "tcn_kernel_size": 3,
  "tcn_dropout": 0.2,
  "shared_hidden_dims": [128, 64]
}
```

输出契约：

```python
{
    "prediction": Tensor[B, 3],
    "aux": {
        "tof_s": Tensor[B, T],
        "peak_index": Tensor[B, T],
        "peak_sharpness": Tensor[B, T],
    },
}
```

兼容要求：

- `prediction` 是唯一主任务预测。
- 现有 `Trainer.predict()`、主 component metrics、conditional metrics 只能消费 `prediction`。
- checkpoint 保存仍使用普通 `state_dict`。

### D2-4：Composite loss 与 Trainer 支持结构化输出

影响文件：

- `tv3/dl/training/losses.py`
- `tv3/dl/training/trainer.py`
- `tv3/dl/training/metrics.py` 或新增 `tv3/dl/training/auxiliary_metrics.py`
- `tests/test_d2_tof_phase_net.py`

loss 配置草案：

```json
{
  "name": "d2_tof_phase_loss",
  "component_loss": {
    "name": "weighted_component_mse",
    "weighting": "fixed",
    "loss_weights": [1.0, 2.0, 1.0],
    "component_count": 3
  },
  "aux_weights": {
    "tof_s": 0.2,
    "peak_index": 0.05
  },
  "tof_loss": "smooth_l1",
  "peak_loss": "smooth_l1"
}
```

实现要求：

1. `D2TOFPhaseLoss` 接收结构化 pred、主 target、`aux_targets`。
2. 主 loss 复用现有 `weighted_component_mse`，不复制权重逻辑。
3. `tof_s` loss 只在 `tof_accepted == 1` 或质量 mask 通过的位置计算；mask 全空时直接报错，不返回 0。
4. `peak_index` loss 对归一化 index 计算，避免 5000 sample 尺度压过主任务。
5. loss payload 写入 `metrics.json.resolved_loss`，包含 component weights 和 aux weights。

Trainer 改动：

1. `_unpack_batch()` 返回 `model_kwargs` 与 `loss_kwargs` 两组。
2. `fit()` 调用 `_compute_loss(pred, y, loss_kwargs)`。
3. `evaluate()` 用 `_main_prediction(pred)` 提取主预测。
4. `evaluate()` 汇总 auxiliary metrics，写入 `result["auxiliary_metrics"]`。
5. 现有普通 tensor 模型路径行为不变。

### D2-5：CLI 与配置文件

影响文件：

- `tv3/dl/cli.py`
- `configs/tv3_d2_tof_phasenet_smoke.json`
- `configs/tv3_d2_tof_phasenet.json`

CLI 扩展：

| 配置项 | 默认 | 说明 |
| --- | --- | --- |
| `aux_target_arrays` | `null` | dict，启用 D2 auxiliary targets。 |
| `aux_metrics` | `null` | list 或 dict，声明要写出的 auxiliary metrics。 |
| `model`: `tof_phase_net` | 无 | 新模型名。 |

正式配置建议：

```json
{
  "dataset_dir": "data/tv3-formal-6000",
  "output_dir": "outputs/tv3_d2/tof_phasenet_s20260704",
  "model": "tof_phase_net",
  "model_kwargs": {
    "output_mode": "raw3",
    "slow_channels": 9,
    "ultrasonic_channels": 5000,
    "fiber_mic_channels": 0,
    "sample_rate_hz": 1000000.0,
    "carrier_hz": 200000.0,
    "raw_output_prior": "auto"
  },
  "modalities": "slow,ultrasonic",
  "scaler_path": "data/tv3-formal-6000/scalers/scaler_slow_sequence.json",
  "dequantize_waveforms": true,
  "normalize_waveforms": true,
  "waveform_stats_features": "log_std,log_max_abs",
  "aux_target_arrays": {
    "tof_true_s": "ultrasonic_tof_s",
    "tof_observed_s": "ultrasonic_tof_observed_s",
    "peak_index": "ultrasonic_peak_index",
    "tof_quality": "ultrasonic_tof_quality",
    "tof_accepted": "ultrasonic_tof_accepted"
  },
  "loss": {
    "name": "d2_tof_phase_loss",
    "component_loss": {
      "name": "weighted_component_mse",
      "weighting": "fixed",
      "loss_weights": [1.0, 2.0, 1.0],
      "component_count": 3
    },
    "aux_weights": {
      "tof_s": 0.2,
      "peak_index": 0.05
    }
  },
  "epochs": 80,
  "batch_size": 16,
  "num_workers": 4,
  "pin_memory": true,
  "persistent_workers": true,
  "prefetch_factor": 4,
  "seed": 20260704,
  "device": "cuda",
  "optimizer": "adamw",
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "grad_clip_norm": 1.0,
  "eval_splits": "val,test,extrapolation",
  "early_stopping": {
    "enabled": true,
    "monitor": "val_loss",
    "patience": 12,
    "min_delta": 0.0,
    "mode": "min"
  },
  "scheduler": {
    "name": "reduce_on_plateau",
    "factor": 0.5,
    "patience": 6,
    "min_lr": 1e-6
  },
  "amp": {
    "enabled": true,
    "dtype": "float16"
  }
}
```

说明：`raw_output_prior: "auto"` 需要 CLI 解析为 train label mean；如果不做该语法扩展，则由 CLI 在 `tof_phase_net` + `raw3` 下复用现有 `cnn1d_tcn_fusion` 的 target mean prior 逻辑。

### D2-6：评估输出

`metrics.json` 每个 split 增加：

```json
{
  "auxiliary_metrics": {
    "tof_mae_s": 0.0,
    "tof_mae_us": 0.0,
    "tof_corr": 0.0,
    "peak_index_mae_samples": 0.0,
    "observed_tof_mae_s": 0.0,
    "observed_tof_corr": 0.0,
    "d2_minus_observed_tof_mae_s": 0.0
  }
}
```

判读方式：

1. `d2_minus_observed_tof_mae_s < 0` 才说明 D2 front-end 对 true TOF 的恢复优于现有 observed TOF。
2. 若 auxiliary 变好但 O2 R2 不变，说明 TOF 误差不是唯一瓶颈，应转向 D4 residual 或 D1 observed sequence。
3. 若 auxiliary 不变或变差，不进入 learnable filter / Transformer 大模型扩展。

### D2-7：实验顺序

| 顺序 | 实验 | 目的 | 通过条件 |
| ---: | --- | --- | --- |
| 1 | synthetic softargmax unit test | 验证可微 lag 提取不是空转 | 平移单峰的 soft index 单调，误差小于 1 sample。 |
| 2 | tv3-smoke 1 epoch CPU | 验证 CLI / Dataset / Loss / Trainer 闭环 | 写出 metrics.json，主指标和 aux metrics 均存在。 |
| 3 | tv3-formal-6000 auxiliary probe | 只看 TOF / peak 学习质量 | D2 TOF MAE 优于 observed TOF baseline。 |
| 4 | tv3-formal-6000 full D2 single seed | 验证主任务收益 | O2 val 至少超过 0.4726，test / extrap 同步改善。 |
| 5 | seeds 42 / 123 / 456 | 验证稳定性 | O2 提升均值超过 +0.05，且方差可接受。 |

## 影响文件清单

| 文件 | 改动类型 | 说明 |
| --- | --- | --- |
| `tv3/dl/data/dataset.py` | 修改 | 增加 auxiliary target 读取与 batch 返回。 |
| `tv3/dl/models/tof_phase_net.py` | 新增 | D2 前端与 `TOFPhaseNetRegressor`。 |
| `tv3/dl/models/registry.py` | 修改 | 注册 `tof_phase_net`。 |
| `tv3/dl/training/losses.py` | 修改 | 新增 `d2_tof_phase_loss`，复用主 component loss。 |
| `tv3/dl/training/trainer.py` | 修改 | 支持结构化输出、aux loss kwargs、aux metrics。 |
| `tv3/dl/cli.py` | 修改 | 解析 aux target、D2 config、输出 payload。 |
| `configs/tv3_d2_tof_phasenet_smoke.json` | 新增 | 本地 smoke 配置。 |
| `configs/tv3_d2_tof_phasenet.json` | 新增 | 正式 6000 配置。 |
| `tests/test_d2_tof_phase_net.py` | 新增 | D2 前端、dataset、trainer、CLI smoke 覆盖。 |
| `docs/掘进通风项目记忆库.md` | 修改 | 登记本实施计划。 |

## 验证命令

以下命令默认在 `tunnel_ventilation` 目录执行。

```bash
python -m pytest tests/test_d2_tof_phase_net.py -q
python -m pytest tests/test_tunnel_ventilation_dl_training.py tests/test_tv3_waveform_normalization.py -q
```

smoke 训练：

```bash
python -m tv3.dl.cli --config configs/tv3_d2_tof_phasenet_smoke.json
```

正式数据核查：

```bash
python scripts/check_slow_channels.py data/tv3-formal-6000
```

正式单 seed：

```bash
python -m tv3.dl.cli --config configs/tv3_d2_tof_phasenet.json
```

结果读取重点：

1. `metrics.json.evaluations.val.component_metrics.x_O2.r2`
2. `metrics.json.evaluations.test.component_metrics.x_O2.r2`
3. `metrics.json.evaluations.extrapolation.component_metrics.x_O2.r2`
4. `metrics.json.evaluations.*.auxiliary_metrics.tof_mae_us`
5. `metrics.json.evaluations.*.auxiliary_metrics.d2_minus_observed_tof_mae_s`
6. `metrics.json.evaluations.*.conditional_metrics.o2_bins`

## 验收标准

| 层级 | 标准 |
| --- | --- |
| 工程验收 | D2 tests 通过；现有 tv3 DL 训练与 waveform normalization tests 不回归。 |
| 数据验收 | clean 6000 无 V_NDIR_CH4；aux target 文件齐全且 split 对齐。 |
| 辅助任务验收 | val split 上 D2 soft TOF 对 true TOF 的 MAE 小于 observed TOF baseline。 |
| 主任务验收 | O2 val / test / extrapolation R2 均相对 D0-observed 提升超过 0.05。 |
| 退化约束 | CO2 R2 不低于 D0-observed 超过 0.02；N2 不出现大幅退化。 |
| 物理解释 | `o2_bins` 允许仍为负，但不得明显劣于 D0-observed；结论保持“大档位可分辨，窄区间不足”。 |

## 停止条件

1. D2 front-end 在 synthetic shift 测试中无法稳定定位峰值：停止，先修 front-end。
2. D2 TOF MAE 不能优于 observed TOF baseline：停止，不做 learnable filter / deeper Transformer。
3. Auxiliary 明显改善但 O2 主任务不提升：暂停 D2 主线，转向 D4 residual 或 D1 observed-physics sequence。
4. O2 提升只出现在 val，不出现在 test / extrapolation：视为不通过，优先检查 split 或过拟合。
5. 为跑通而需要把 oracle 特征作为输入：直接判定路线违规。

## 暂不做

- 不引入 fiber_mic，R3f 另行评估。
- 不引入 HITRAN，CO2 光学保真不是当前 O2 缺口根因。
- 不实现 GasGraph / D3。
- 不实现 self-supervised pretraining / D5。
- 不改数据集 split 策略；SPXY 是独立计划。
- 不把 O2 专用传感器并入 D2；那属于阶段 III 后备硬件路线。

## 实施检查清单

- [ ] D2-0 数据与 D0 baseline 核查完成。
- [ ] Dataset 能读取 aux targets，且不把 aux 传给 model。
- [ ] `SoftArgmaxLag` synthetic shift 单测通过。
- [ ] `TOFPhaseNetRegressor` forward shape 与 aux 输出单测通过。
- [ ] `D2TOFPhaseLoss` 能同时计算主 loss 与 aux loss。
- [ ] `Trainer.evaluate()` 写出主指标和 auxiliary metrics。
- [ ] `tof_phase_net` 注册到 `MODEL_REGISTRY`。
- [ ] smoke config 1 epoch 跑通。
- [ ] 正式 config 单 seed 完成。
- [ ] 与 D0-observed 对比表回填到项目记忆库或实验报告。
