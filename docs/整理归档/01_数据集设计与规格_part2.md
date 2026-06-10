> 复核说明：本页承接 `01_数据集设计与规格.md`，内容聚焦物理参数、split 与 scaler 规格；不单独定义实验进度口径

## 7. 物理参数清单

### 7.1 共用参数

| 参数 | 值 | 说明 |
|------|---|------|
| 载频 | 40 kHz | 超声中心频率 |
| 采样率 | 200 kHz | 两通道统一采样率 |
| 激励 | Hanning 8-cycle burst | 发射脉冲形式 |
| ADC | 16-bit | 模数转换精度 |
| timesteps | 120 | 每序列时步数 |
| dt_s | 1.0 s | 时步间隔 |
| calibration | pending | 校准状态 |

### 7.2 通道 1 专属参数

| 参数 | 值 |
|------|---|
| measurement_window_s | 0.005 |
| waveform_samples | 1000 |
| noise_std_us | 1e-3 V |

### 7.3 通道 2 专属参数

| 参数 | 值 |
|------|---|
| measurement_window_s | 0.010 |
| waveform_samples | 2000 |
| l_direct_factor | 0.5 |
| wall_reflection_coef | 0.5 |
| max_reflections | 15 |
| noise_std_fm | 1e-3 V |

## 8. Split 策略

### 8.1 seedpath_formal 策略

- **策略名称**：`stratified_group_by_mixture_id_with_extrapolation_holdout`
- **分组依据**：按 `mixture_id` 分组，避免同 mixture 跨 split
- **4 路划分**：train / val / test / extrapolation
- **比例**：约 58% / 13.5% / 13.5% / 15%

| Split | 序列数 | 比例 | 用途 |
|-------|--------|------|------|
| train | 17400 | 58% | 训练 + scaler 拟合 |
| val | 4050 | 13.5% | 验证 / 早停 |
| test | 4050 | 13.5% | 测试评估（主线指标）|
| extrapolation | 4500 | 15% | 外推泛化测试（可选）|

### 8.2 与旧版差异

| 项 | waveform_v3 (旧) | waveform_v3_seedpath_formal (新) |
|---|---|---|
| 总序列数 | 10000 | 30000 |
| split 数 | 3 | 4 |
| train | 7028 | 17400 |
| val | 1478 | 4050 |
| test | 1494 | 4050 |
| extrapolation | — | 4500 |

**重要**：两份数据集的 `mixture_id` 命名空间不同，**不要混用 split 文件**。

## 9. Scaler 策略

| 数据 | Scaler | 拟合范围 | 说明 |
|------|--------|----------|------|
| slow [N, 120, 8] | ChannelStandardScaler（按通道 z-score）| 仅 train split | 固化到 scaler_slow_sequence.json |
| ultrasonic 波形 | 不单独保存 scaler；训练时先反量化，再用 train 统计归一化 | 仅 train split | int16 × scale → float32 |
| fiber_mic 波形 | 同上 | 仅 train split | int16 × scale → float32 |

**固化文件**：
- `scalers/scaler_slow_sequence.json` - 8 通道 scaler
- `scalers/scaler_slow_sequence_modal.json` - 模态归属信息

## 10. 工况采样与配气

完全复用 V3.0 工况逻辑，不改 `condition_grid_sequence.csv` 的字段：

| 字段 | 说明 |
|------|------|
| sequence_id | Q000001 ~ Q030000（seedpath_formal）|
| mixture_id | 同 mixture 下多 L_m 配置 |
| x_H2, x_CH4, x_CO2, x_N2 | 四组分，加和 100% |
| T_C_base, P_MPa_base, H_RH_base | 工况基线 |
| L_m_base | 0.2, 0.6, 1.0, 1.4 m 阶梯 |
| status | synthetic_measurement |

**配气范围**：

| 组分 | 范围（vol%）| 说明 |
|------|------------|------|
| x_H2 | 0 - 25 | 掺氢天然气主要关注组分 |
| x_CH4 | 50 - 95 | 天然气主成分 |
| x_CO2 | 0 - 10 | 常见杂质 |
| x_N2 | 0 - 20 | 常见杂质 |

**工况范围**：

| 参数 | 范围 | 说明 |
|------|------|------|
| T_C | 0 - 50 °C | 常温范围 |
| P_MPa | 0.1 - 1.0 MPa | 低中压范围 |
| H_RH | 0 - 80 %RH | 相对湿度 |
| L_m | 0.2 - 1.4 m | 声程变化 |

## 11. 方向性与质量检查

由 `src/sim/scripts/check_waveform_directionality.py` 输出 `quality/waveform_quality_summary.json`。

### 11.1 通道 1 必查项

| 检查 | 通过条件 | 物理意义 |
|------|----------|----------|
| L_m ↑ → peak_index ↑ | Pearson r > 0.99 | 声程增加，TOF 增加 |
| c_sound ↑ → peak_index ↓ | 单调下降 | 升 H2 含量，声速上升，TOF 下降 |
| alpha ↑ → 峰幅度 ↓ | 单调下降 | 升 CO2，声衰减增大 |
| 噪声 SNR | > 40 dB | 信号质量 |

### 11.2 通道 2 必查项

| 检查 | 通过条件 | 物理意义 |
|------|----------|----------|
| alpha ↑ → tau ↓ | Pearson r < -0.85 | 声衰减增大，衰减时间常数减小 |
| L_m ↑ → T_round ↑ | 单调 | 腔体尺寸增加，反射周期增加 |
| R 调高 → 包络尾巴更长 | 单调 | 反射系数增大，混响延长 |
| 反射峰可识别（前 3 次）| 在 90% 序列中可见 | 混响结构清晰 |
| tau 估计与 1/(alpha·c) 相关 | Pearson r > 0.90 | 物理模型一致性 |

### 11.3 质量文件格式

```json
{
  "ultrasonic": {
    "peak_index_distribution": {
      "min": 0,
      "max": 0,
      "mean": 0,
      "p95": 0
    },
    "peak_amplitude_v": {
      "min": 0,
      "max": 0,
      "mean": 0
    },
    "snr_db_estimate": 0,
    "tof_directionality_passed": true
  },
  "fiber_mic": {
    "decay_tau_ms_distribution": {
      "min": 0,
      "max": 0,
      "mean": 0
    },
    "envelope_peak_count_mean": 0,
    "snr_db_estimate": 0,
    "alpha_directionality_passed": true
  }
}
```

## 12. metadata 规范

`metadata/waveform_v3_spec.json` 至少包含：

```json
{
  "dataset_version": "V3.1",
  "channels": {
    "ultrasonic": {
      "sample_rate_hz": 200000,
      "center_frequency_hz": 40000.0,
      "burst_cycles": 8,
      "measurement_window_s": 0.005,
      "waveform_samples": 1000,
      "noise_std_v": 1e-3,
      "adc_max_int16": 32767,
      "calibration_status": "pending"
    },
    "fiber_mic": {
      "sample_rate_hz": 200000,
      "center_frequency_hz": 40000.0,
      "burst_cycles": 8,
      "measurement_window_s": 0.010,
      "waveform_samples": 2000,
      "noise_std_v": 1e-3,
      "adc_max_int16": 32767,
      "l_direct_factor": 0.5,
      "wall_reflection_coef": 0.5,
      "max_reflections": 15,
      "calibration_status": "pending"
    }
  },
  "slow_channels": [
    "V_NDIR_CH4", "V_NDIR_CO2", "V_TCS",
    "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"
  ],
  "labels": ["x_H2", "x_CH4", "x_CO2", "x_N2"],
  "sequences": 30000,
  "timesteps": 120,
  "dt_s": 1.0,
  "generation_seed": "20260514"
}
```
