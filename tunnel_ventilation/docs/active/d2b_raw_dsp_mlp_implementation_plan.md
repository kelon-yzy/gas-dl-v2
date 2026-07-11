# B6：RawDSP Frozen Features + Target-Scaled MLP 实施计划

> 状态：**代码与配置已就绪，待服务器正式训练**  
> 日期：2026-07-11  
> 前置证据：[D2b RawDSP 计划](d2b_raw_dsp_implementation_plan.md)的 B1 Ridge parity 与 clean 6000 帧级 fidelity 均已通过。  
> 目标：在不读取 simulator-derived observed 数组的条件下，检验 R5-T 的非线性收益能否由已验证的 RawDSP 特征链路承接。

## 1. 当前事实与问题定义

### 1.1 已通过前置门

正式 `tv3-formal-6000` 的 RawDSP cache 满足：

- `template_mode=train_baseline_median`；
- `template_source_split=train`；
- `diagnostic_only=false`；
- 6000 条序列、3,072,000 帧全量审计；
- frame fidelity 的 val/test/extrapolation 全部通过。

最差 eval split 为 extrapolation：peak MAE `0.02184 sample`、peak P95 `0.04443 sample`、peak bias `0.01471 sample`、sound-speed MAE `0.13671 m/s`。全部帧 accepted，且没有 boundary hit 或 clipping。

因此 B6 不再回答“raw waveform 能否恢复测量特征”，而只回答：**在同一 RawDSP frozen 特征契约下，目标标准化 MLP 是否能稳定超过 B1 Ridge。**

### 1.2 固定比较对象

| 实验 | val O₂ R² | test O₂ R² | extrapolation O₂ R² | 语义 |
| --- | ---: | ---: | ---: | --- |
| B1 RawDSP Ridge | 0.4280 | 0.4786 | 0.3695 | 已通过的可部署线性基线 |
| R5-T observed MLP | 0.6642 | 0.6462 | 0.5815 | simulator-derived observed 特征上的非线性上限对照 |
| B6 RawDSP MLP | 待测 | 待测 | 待测 | 本计划目标 |

R5-T 不是 B6 的通过线。B6 的正式比较对象是 B1，因为两者共享可部署的 RawDSP 特征来源。

## 2. 不变量与边界

1. 输出保持 `raw3`、`out_dim=3`，直接预测 `x_CO2`、`x_O2`、`x_N2`。
2. 禁止 `gas_head`、N₂ 回填、ILR/ALR、`target_transform`、闭包残差头。
3. `mlp_standardize_targets=true` 仅改变训练损失空间；公开预测必须反变换回原始百分比 `raw3`。
4. B6 必须使用 `d0_raw_dsp_physics_stats_v1`，不得切换为 `d0_observed_physics_stats_v1`，也不得读取 `ultrasonic_tof_observed_s`、`ultrasonic_peak_index` 或 `ultrasonic_sound_speed_estimated_m_per_s` 作为模型输入。
5. 数据固定为 clean `tv3-formal-6000`，split 固定为当前 `random_mixture_id_split_v4`；本轮不混入 SPXY 或新 OOD 策略。
6. RawDSP cache 的 `build_signature=57927d707d2c6f8dd449fc66f4c7c5aac42524123072114364954367022980e7` 与 `template_digest=864a37af53a5b1d29c00082a9eb07ad551ca08fa426d297f3613f18a629fc190` 必须记录在 B6 `metrics.json` 的输入追溯信息中。
7. 缺失 cache、manifest、required array 或 7 个正式 slow channel 时直接失败；不回退到 observed 数组或旧本地 600 数据。

## 3. 冻结的 B6 配方

### 3.1 特征契约

与 B1 完全一致：

```text
feature_builder = d0_raw_dsp_physics_stats_v1
slow channels = V_NDIR_CO2, V_TCS, T_C, P_MPa, H_RH, L_m, piston_position_m
physics arrays =
  ultrasonic_tof_observed_raw_dsp_s
  ultrasonic_peak_index_raw_dsp
  ultrasonic_sound_speed_raw_dsp_m_per_s
  ultrasonic_corr_peak
  ultrasonic_snr_db
  ultrasonic_raw_dsp_quality
  ultrasonic_raw_dsp_accepted
statistics = mean, std, min, max, range, first, last, delta, slope
phase windows = baseline, exposure, steady, recovery
early fractions = 0.25, 0.5, 0.75
```

`ultrasonic_peak_to_sidelobe_ratio`、`ultrasonic_peak_width_samples`、`ultrasonic_peak_amplitude_raw_dsp_v`、clipping 与 boundary-hit 等额外数组不得在 B6 临时加入；本轮只验证已由 B1 Ridge 通过的 frozen contract。

### 3.2 MLP 配方

直接复用 R5-T 已通过的训练变量：

| 参数 | 值 |
| --- | --- |
| head | `mlp` |
| hidden dims | `[256, 128]` |
| dropout | `0.1` |
| optimizer | AdamW |
| lr | `1e-3` |
| weight decay | `1e-4` |
| batch size | `256` |
| max epochs | `200` |
| early stopping | val O₂ R²，patience `20`，回滚 best checkpoint |
| loss weights | `[1.0, 2.0, 1.0]` |
| target scaling | per-target `StandardScaler`，仅训练损失空间 |
| seed | `20260704` |
| device | `cuda` |

本轮不加入 RealMLP、TabM、独立 heads、PCGrad、feature selection 或 B7 residual 学习。任一新增变量都会破坏“RawDSP 替代 observed 特征”的单因素判断。

## 4. 实施文件与命令

### 4.1 新增配置与 provenance

新增 `configs/tv3_d2b_raw_dsp_mlp_target_scaled.json`。该配置应等于 `tv3_d2b_raw_dsp_ridge.json` 的特征字段，加上 R5-T 的 MLP 字段：

```json
{
  "dataset_dir": "data/tv3-formal-6000",
  "output_dir": "outputs/tv3_d2b/raw_dsp_mlp_target_scaled",
  "feature_set": "physics_stats",
  "head": "mlp",
  "feature_builder": "d0_raw_dsp_physics_stats_v1",
  "include_slow": true,
  "mlp_hidden_dims": [256, 128],
  "mlp_dropout": 0.1,
  "mlp_weight_decay": 0.0001,
  "mlp_lr": 0.001,
  "mlp_batch_size": 256,
  "mlp_max_epochs": 200,
  "mlp_patience": 20,
  "mlp_loss_weights": [1.0, 2.0, 1.0],
  "mlp_standardize_targets": true,
  "seed": 20260704,
  "device": "cuda"
}
```

配置中必须保留 B1 的完整 `slow_channels`、`physics_arrays`、statistics、phase windows、early fractions 与 eval splits；上面片段仅展示 MLP 差异字段。

现有 `rocket_training_payload` 不会读取 RawDSP manifest。执行 B6 前，扩展 `tv3/ml/rocket_training.py`：当 `feature_builder=d0_raw_dsp_physics_stats_v1` 时，将 cache 的 `manifest.json` 中的 `build_signature`、`template_digest`、`template_mode`、`template_source_split` 与 `diagnostic_only` 写入 `metrics.json` 的 `raw_dsp_provenance`。缺失或不符合正式 cache 契约时直接失败。

### 4.2 服务器执行顺序

```bash
# 1. 确认 fidelity 证据存在且通过
python -c "import json; p=json.load(open('outputs/tv3_d2b/raw_dsp_frame_fidelity/metrics.json', encoding='utf-8')); assert p['status'] == 'passed'; print(p['source']['cache_build_signature'])"

# 2. 重跑 B1，生成带 RawDSP provenance 的锁定参考基线
python -m tv3.pipeline.run_tv3_rocket_baseline \
  --config configs/tv3_d2b_raw_dsp_ridge.json \
  --output-dir outputs/tv3_d2b/raw_dsp_ridge_provenance

# 3. 运行 B6
python -m tv3.pipeline.run_tv3_rocket_baseline \
  --config configs/tv3_d2b_raw_dsp_mlp_target_scaled.json
```

旧 `outputs/tv3_d2b/raw_dsp_ridge/metrics.json` 不含 provenance，不能作为 B6 reference。若 `outputs/tv3_d2b/raw_dsp_mlp_target_scaled/` 已存在，训练入口会拒绝覆盖；必须新建结果目录或显式 `--overwrite` 并记录原因。

## 5. 结果审计与验收

### 5.1 必须输出

`outputs/tv3_d2b/raw_dsp_mlp_target_scaled/metrics.json` 必须包含：

1. train/val/test/extrapolation 的三组分 R²、MAE、RMSE、`sum_abs_error` 与 bins；
2. best epoch、best val O₂ R²、parameter count、seed 和完整 MLP config；
3. feature builder、feature names / count、RawDSP cache path、build signature 与 template digest；
4. train-val O₂ gap；
5. 与 B1 的 val/test/extrap O₂ R² 差值表。

### 5.2 通过线

B6 只有同时满足以下条件才通过：

1. val O₂ R² `>= 0.4726`，即 B1 val `+0.05`；
2. test O₂ R²严格高于 B1 的 `0.4786`；
3. extrapolation O₂ R²严格高于 B1 的 `0.3695`；
4. 增益不能仅来自 train 或 val；
5. `sum_abs_error` 作为 raw3 监控项完整报告，但本轮不以闭包误差替代 O₂ 通过判据。

### 5.3 诊断矩阵

| 结果 | 判断 | 后续动作 |
| --- | --- | --- |
| 三 eval split 全通过 | RawDSP 可部署非线性头成立 | 先做多 seed 稳定性复核，再决定是否运行 B7 |
| 仅 val 通过 | 选择偏差或泛化失败 | 判失败，不调大模型；进入 B7 OOF Ridge residual 对照 |
| B6 三 split 均不超过 B1 | RawDSP 直接非线性承接失败 | 运行 B7，检验残差式非线性是否仍有信息 |
| train 明显高、eval 不提升 | 高维表格过拟合 | 保留 B1，B7 前不得继续加宽网络 |
| cache tracing 不匹配或 fidelity 未通过 | 输入证据无效 | 停止 B6，先重建或审计 RawDSP cache |

R5-T observed 的 `0.6642 / 0.6462 / 0.5815` 只用于解释可学习非线性上限；B6 即使低于该结果，只要超过 B1 并满足三 split 同步条件，仍可作为可部署增益成立。

## 6. 代码、测试与文档更新

`tv3.pipeline.run_tv3_rocket_baseline` 已支持 `head=mlp` 和 `mlp_standardize_targets`，不新增第二个训练实现。实施范围限于 B6 配置、RawDSP provenance payload 和对应测试。

最小验证：

```bash
python -m pytest tests/test_d2b_frame_fidelity_audit.py -q
python -m pytest tests/test_tv3_r5_mlp.py tests/test_tv3_raw_dsp_pipeline.py -q
python -m tv3.pipeline.run_tv3_rocket_baseline \
  --config configs/tv3_d2b_raw_dsp_mlp_target_scaled.json
```

服务器正式结果回填后，更新本文件、`d2b_raw_dsp_implementation_plan.md`、项目记忆库和 `docs/active/README.md`；未验收的单 seed 结果不得提前写为“可部署模型通过”。

## 7. 检查清单

- [x] B1 parity 已通过。
- [x] clean 6000 train-calibrated frame fidelity 已通过。
- [x] 新增 B6 配置，特征字段逐项与 B1 一致。
- [x] 为 RawDSP feature builder 写入并测试 `raw_dsp_provenance`。
- [ ] 运行 B6 单 seed 正式训练。
- [ ] 审计三 eval split、bins、closure、feature tracing 与 train-val gap。
- [ ] 按通过线决定多 seed 复核或进入 B7。

### 7.1 2026-07-11 本地实施记录

已完成：

1. `configs/tv3_d2b_raw_dsp_mlp_target_scaled.json`：特征契约与 B1 一致，MLP 字段复用 R5-T。
2. `tv3/ml/rocket_training.py`：RawDSP 路径写入 `raw_dsp_provenance` 与 `o2_audit`（含 train−val O₂ gap、相对 B1 差值）；正式 cache 契约不符时直接失败。
3. 最小验证通过：`tests/test_d2b_frame_fidelity_audit.py`、`tests/test_tv3_r5_mlp.py`、`tests/test_tv3_raw_dsp_pipeline.py` 共 23 passed。

未完成：

- 本地 `data/tv3-formal-6000/` 仅有 features 缓存，缺少 `manifest.json` 与完整序列数组，无法在本机跑正式 B6。
- 服务器命令仍按 §4.2 执行；结果回填前不得把单 seed 写成可部署通过。
