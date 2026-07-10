# D2b / RawDSP 物理波形特征提取实施计划

> 状态：**D2b-0～D2b-2 与 D2b-3 B1 基础设施及本地验证已完成，clean 6000 cache、fidelity 与 Ridge parity 待服务器执行**  
> 日期：2026-07-10  
> 依据：[掘进通风项目记忆库.md](../掘进通风项目记忆库.md)、[d2_tof_phasenet_implementation_plan.md](../archive/completed/d2_tof_phasenet_implementation_plan.md)、`tv3-formal-6000` 的 D0、D2、R5' 与 R5 结果，以及本地 raw waveform 只读诊断。

## 1. 结论

D2b 不继续扩展当前 `TOFPhaseNetRegressor`，也不把 5000 点 raw waveform 直接交给更大的 CNN、PatchTST 或 Transformer。第一目标是建立一个可审计的 RawDSP 前端：从去量化波形中恢复连续峰位、校正 TOF、估计声速与质量指标，再复用 D0 的 phase/window 统计口径和 Ridge/ExtraTrees/目标标准化 MLP 回归头。

推荐主链路：

```text
dequantized ultrasonic waveform
  -> calibration template / matched correlation
  -> physical lag search window
  -> local peak + three-point sub-sample interpolation
  -> per-sequence delay calibration
  -> corrected TOF + estimated sound speed + quality features
  -> D0-compatible phase/window statistics
  -> Ridge parity gate
  -> ExtraTrees / target-scaled residual MLP
```

D2b 的首要验收不是直接把 O₂ R² 推到 0.70，而是证明 raw waveform 可以在不读取仿真派生 observed 数组的前提下，重建 D0-observed 的测量特征空间。只有通过这一 parity gate，后续非线性模型的收益才具备端到端部署语义。

### 1.1 2026-07-10 实施进度

已落地：

1. `tv3-raw-dsp-frame-1` 帧级缓存与 `d0_raw_dsp_physics_stats_v1` 序列特征契约。
2. exact simulator 与 train-only baseline median 两种模板模式；模板参考峰 offset、极性和 digest 均写入 manifest。
3. 物理 lag window、normalized matched correlation、相关包络 phase lock、三点抛物线亚采样峰值、per-sequence baseline delay calibration 和 `tof vs L_m` 诊断。
4. timing 与 amplitude 双路语义、质量指标、分块 memmap、CPU 单进程与多进程数值一致性。
5. RawDSP cache 到 D0 统计再到 RidgeCV 的 smoke 闭环。

本地旧 600 的只读波形验证中，exact-template 与 train-calibrated template 均通过各自 512-frame peak gate。该数据仍含额外 `V_NDIR_CH4`，因此只作为 waveform fidelity 证据，不作为模型性能结论。

尚未执行：clean `tv3-formal-6000` 正式 cache、val/test/extrap fidelity、B1 Ridge parity，以及 parity 通过后的 B5–B7。

## 2. 背景与已验证事实

### 2.1 D2 失败仍然成立

正式 `tv3-formal-6000` 上，D2 的 O₂ val R² 为 −0.015，CO₂ 与 N₂ 也全面低于 D0-observed。当前 D2 配置与训练结果不得改写为成功，既有停止条件仍然有效。

### 2.2 D2 失败不能外推为所有 raw waveform 路线失败

代码复核确认：

1. `FixedQuadratureFilterBank` 的核是 buffer，`SoftArgmaxLag.temperature` 是固定常数。
2. `tof_s` 与 `peak_index` 在可训练 `AcousticFrontend.proj` 之前直接输出。
3. 实测 `prediction.requires_grad=true`，但 `aux.tof_s.requires_grad=false`、`aux.peak_index.requires_grad=false`。
4. 因此 D2 的 TOF / peak auxiliary loss 不会更新峰值估计器，只会给主 loss 加入常数项。
5. TOF 使用秒作为 SmoothL1 输入，peak 使用 0–1 归一化位置，按现有误差量级，两个辅助项约为 `1e-9`，相对约 `1.0` 的 component loss 可忽略。

所以 D2 证伪的是“固定短窗 I/Q softargmax + 隐式 TCN 融合 + 当前复合 loss”这一实现，不是基于完整标定模板的物理匹配滤波路线。

### 2.3 当前 D2 TOF 指标坐标不一致

仿真波形峰位对应：

```text
tof_observed = tof_true + system_delay + cable_delay + trigger_jitter
```

默认固定延迟为 `80 μs + 2 μs = 82 μs`。当前 D2 把峰位直接换算为 `tof_s`，却与 `tof_true_s` 比较；同时 peak loss 又与包含固定延迟的 `ultrasonic_peak_index` 比较。84.2 μs 的 D2 TOF MAE主要包含 82 μs 坐标偏置，不能单独证明峰值定位误差达到 84 μs。

后续必须分开报告：

- raw peak 对 `tof_observed_s` 的定位误差；
- 延迟校正后 TOF 对 `tof_true_s` 的误差；
- `L_m` 与固定延迟残差化后的声速误差；
- 不再用被 `L_m` 扫描主导的绝对 TOF correlation 作为成功判据。

### 2.4 D0-observed 是测量级上限，不是当前 raw 链路产物

`d0_observed_physics_stats_v1` 直接读取：

- `ultrasonic_tof_observed_s`
- `ultrasonic_peak_index`
- `ultrasonic_sound_speed_estimated_m_per_s`
- `ultrasonic_tof_quality`
- `ultrasonic_tof_accepted`

这些数组由仿真器在生成波形时同步计算，并非现有代码从保存的 raw waveform 重建。它们可代表传感器 DSP 已经输出的测量级特征，但在 D2b parity 通过前，不应称为当前 raw-to-prediction 链路的端到端可部署证据。

### 2.5 本地局部诊断

在旧本地 `tv3-formal` 的 8 条序列、每条前 64 帧，共 512 个 ultrasonic frame 上执行只读诊断。该数据集的第 8 个 slow 通道受 `V_NDIR_CH4` 污染，因此本诊断只评价 raw waveform 与仿真 TOF 数组的对应关系，不用于声明模型性能。

使用完整 `transducer_response_pulse` 模板、FFT 相关和三点抛物线插值后：

| 指标 | 结果 |
| --- | ---: |
| 连续峰位相对 `tof_observed_s` MAE | 0.019 sample，即 0.019 μs |
| 连续峰位 bias | −0.00017 sample |
| 连续峰位误差 std | 0.021 sample |
| 连续峰位 P95 绝对误差 | 0.030 sample |
| 重建声速相对 `sound_speed_estimated` MAE | 0.010 m/s |

这证明当前仿真波形中保留了足以重建 observed TOF 与 estimated sound speed 的信息。正式结论仍需在 clean `tv3-formal-6000` 上按本计划复验。

## 3. 目标与非目标

### 3.1 目标

1. 从 raw ultrasonic waveform 重建 observed TOF、连续 peak、估计声速和质量特征。
2. 建立不读取仿真 observed 数组的 `d0_raw_dsp_physics_stats_v1` 特征契约。
3. 用 Ridge 进行 D0-observed parity 验证。
4. parity 通过后，将 R7 ExtraTrees、R5-T 或 Ridge residual MLP 切换到 RawDSP 特征。
5. 为真实硬件保留模板标定、延迟漂移与局部神经残差修正接口。

### 3.2 非目标

1. 不重启当前 D2 softargmax 模型的超参数搜索。
2. 不直接训练更大的 5000 点 raw waveform encoder。
3. 不以 exact simulator template 的结果声明 sim-to-real 可部署性。
4. 不承诺使 0.8% O₂ bins R² 转正；oracle 已确认该尺度存在物理辨识上限。
5. 不在第一版恢复 fiber_mic，不修改 TDLAS 暂缓状态。

## 4. 不变量

1. tv3 输出保持 `raw3`、`out_dim=3`，直接预测 `x_CO2`、`x_O2`、`x_N2`。
2. 不使用 `gas_head`、ILR / ALR、`target_transform`、闭包残差头或 N₂ 回填。
3. 正式 RawDSP feature builder 不得读取 `ultrasonic_tof_s`、`ultrasonic_tof_observed_s`、`ultrasonic_peak_index`、`ultrasonic_sound_speed_m_per_s`、`ultrasonic_sound_speed_estimated_m_per_s` 或 `ultrasonic_alpha_true_npm` 作为输入。
4. 上述仿真数组只允许用于离线审计、单元测试 target 和正式结果后的 fidelity metrics。
5. 模板只能来自硬件标定、训练 split baseline 波形或显式 diagnostic exact-template 模式；val/test/extrapolation 不得参与模板拟合。
6. per-sequence baseline 延迟校准只能使用该序列的 baseline 输入、已知新鲜空气组成和环境测量，不使用该序列标签。
7. slow channel 必须按 metadata 名称解析，不硬编码列索引。
8. derived arrays 写入 `features/raw_dsp/`，不得混入 `sequences/` 冒充原始数据真相。
9. 正式模型结论只使用 clean `tv3-formal-6000`；旧本地 600 仅用于波形算法 smoke 与 fidelity 调试。
10. 缺失 waveform scale、phase metadata、校准模板或必须的 slow channel 时直接报错，不添加静默 fallback。

## 5. 数据与特征契约

### 5.1 帧级缓存

建议目录：

```text
data/<dataset>/features/raw_dsp/raw_dsp_frame_v1/
  manifest.json
  template.npy
  ultrasonic_peak_index_raw_dsp.npy
  ultrasonic_tof_observed_raw_dsp_s.npy
  ultrasonic_delay_calibration_s.npy
  ultrasonic_tof_corrected_raw_dsp_s.npy
  ultrasonic_sound_speed_raw_dsp_m_per_s.npy
  ultrasonic_corr_peak.npy
  ultrasonic_snr_db.npy
  ultrasonic_peak_width_samples.npy
  ultrasonic_raw_dsp_quality.npy
  ultrasonic_raw_dsp_accepted.npy
```

`manifest.json` 至少记录：

- schema version，例如 `tv3-raw-dsp-frame-1`；
- source dataset 与 source waveform dtype；
- template mode、template source split、template digest；
- sample rate、carrier frequency、search window bounds；
- peak interpolation method；
- delay calibration method；
- 输入文件摘要与输出 shapes；
- 生成时间和代码版本。

### 5.2 序列级特征缓存

新增 builder：`d0_raw_dsp_physics_stats_v1`。

首版 physics arrays：

```text
ultrasonic_tof_observed_raw_dsp_s
ultrasonic_peak_index_raw_dsp
ultrasonic_sound_speed_raw_dsp_m_per_s
ultrasonic_corr_peak
ultrasonic_snr_db
ultrasonic_raw_dsp_quality
ultrasonic_raw_dsp_accepted
```

统计口径必须与 D0 一致：

- statistics：`mean,std,min,max,range,first,last,delta,slope`
- phase windows：`baseline,exposure,steady,recovery`
- early fractions：`0.25,0.5,0.75`
- slow 特征仍来自正式 7 通道 schema。

### 5.3 双路波形语义

- timing path：去量化后进行局部去均值或归一化相关，只用于峰位和相位。
- amplitude path：保留去量化电压尺度，用于 peak amplitude、SNR、衰减和 clipping quality。

不得用 per-frame z-score 后的波形计算幅度或衰减特征。

## 6. RawDSP 算法设计

### 6.1 模板构建

支持两个明确模式：

1. `exact_simulator_debug`：使用 `transducer_response_pulse`，只验证实现上限，不进入正式性能结论。
2. `train_baseline_median`：从 train split 的 baseline frame 中选取高质量波形，按粗峰对齐、幅度归一化后取稳健中位数，作为正式模板。

真实硬件接入时替换为独立标定波形，不改变后续接口。

### 6.2 物理搜索窗口

对每帧根据 `L_m`、允许声速范围、sample rate 和延迟范围计算 lag window。第一版允许使用宽物理边界，例如 `c ∈ [250, 400] m/s`，不能从标签反推窗口。

只在窗口内搜索相关峰，避免全 5000 点 softargmax 被噪声或边界响应影响。

### 6.3 亚采样峰值

1. 对完整模板做 normalized cross-correlation 或等价 matched convolution。
2. 取局部最大相关峰。
3. 使用三点抛物线插值估计 fractional sample offset。
4. 输出连续 peak sample 与 `tof_observed_raw_dsp_s`。

第一版不需要可微，因为前端是固定、可测试的物理算子。若以后加入神经残差头，只允许在局部相关峰窗口上学习小修正。

### 6.4 延迟校准

优先采用 per-sequence baseline self-calibration：

```text
c_fresh = physics_model(fresh_air_composition, T_C, P_MPa, H_RH)
delay_seq = robust_mean(tof_peak_baseline - L_m_baseline / c_fresh)
tof_corrected = tof_peak - delay_seq
c_estimated = L_m / tof_corrected
```

`manifest.delay_correction_s` 只作为 simulator debug 对照，不作为正式 RawDSP 唯一来源。

同时输出 steady phase 的 `tof vs L_m` 稳健线性拟合：

```text
tof = intercept + slowness * L_m
c_slope = 1 / slowness
```

该结果作为固定延迟漂移诊断与可选声速特征，不替代主校准路径。

### 6.5 质量指标

至少计算：

- normalized correlation peak；
- peak-to-sidelobe ratio；
- local peak width；
- dequantized waveform SNR；
- clipping flag；
- physical-window boundary hit；
- accepted flag。

quality 规则必须显式、可审计，并在 manifest 中记录阈值。

## 7. 分阶段实施

### D2b-0：冻结契约与 preflight

目标：先固定 RawDSP 的输入边界和比较基线。

任务：

1. 确认 clean 6000 的 waveform path、scale、slow metadata、phase csv 与 split 文件。
2. 记录 D0-observed val/test/extrap 指标和 frozen feature contract。
3. 增加配置 schema 草案与 feature cache manifest schema。
4. 确认旧本地 600 只用于 fidelity smoke。

产物：配置草案、preflight 输出和契约测试。

### D2b-1：实现帧级 matched TOF extractor

建议新增：

- `tv3/ml/raw_dsp_features.py`
- `tests/test_raw_dsp_features.py`

必须覆盖：

1. exact template 无噪声 fractional shift 测试；
2. noise、幅度缩放和 int16 / int32 去量化一致性；
3. physical lag window 边界；
4. parabolic interpolation 单调性和误差；
5. amplitude path 不受 timing normalization 污染；
6. local 512-frame fidelity 复现。

### D2b-2：实现缓存编排与 CLI

建议新增：

- `tv3/pipeline/build_tv3_raw_dsp_features.py`
- `configs/tv3_d2b_raw_dsp_features.json`
- `tests/test_tv3_raw_dsp_pipeline.py`

要求：

1. 分块读取 memmap，不能一次载入 6000 × 512 × 5000 波形。
2. 支持 CPU 多进程或 GPU batch，但两种后端必须通过数值一致性测试。
3. train template 先独立拟合并冻结，再处理全部 split。
4. cache 已存在且 manifest 不匹配时直接报错，不静默复用。
5. 不写入 `sequences/`。

### D2b-3：D0-RawDSP Ridge parity

建议扩展：

- `tv3/ml/rocket_features.py`
- `configs/tv3_d2b_raw_dsp_ridge.json`
- `scripts/analyze_d2b_results.py`

实验：

| 编号 | 特征 | 模型 | 目的 |
| --- | --- | --- | --- |
| B0 | D0-observed simulator-derived | RidgeCV | 现有测量级基线 |
| B1 | RawDSP TOF / peak / speed / quality + slow | RidgeCV | 主 parity gate |
| B2 | RawDSP TOF / peak + slow，不含 sound speed | RidgeCV | 验证显式物理变换价值 |
| B3 | RawDSP sound speed + slow，不含 raw TOF | RidgeCV | 验证声速是否为核心充分特征 |
| B4 | RawDSP exact simulator template | RidgeCV | diagnostic upper bound，不作正式结论 |

### D2b-4：非线性回归头

只有 B1 通过 parity gate 后执行：

| 编号 | 特征 | 模型 | 目的 |
| --- | --- | --- | --- |
| B5 | RawDSP frozen contract | ExtraTrees | 承接 R7 的可部署非线性交互 |
| B6 | RawDSP frozen contract | target-scaled MLP | 复用 R5-T 的目标尺度修复 |
| B7 | RawDSP frozen contract | Ridge residual MLP | 只学习线性基线残差，降低弱信号优化难度 |

B7 形式：

```text
y_hat = y_ridge + residual_mlp(z_raw_dsp)
```

Ridge 预测和残差 target 必须按 train split 生成，val/test/extrap 不参与拟合。

### D2b-5：可选局部神经残差修正

仅在 exact-template fidelity 通过、train-calibrated template fidelity 未通过时启动。

流程：

1. 固定 matched filter 得到 coarse peak。
2. 截取峰附近约 32–64 sample 的 waveform / correlation patch。
3. 小型 CNN 或 MLP 预测 fractional-sample residual。
4. target 使用 `tof_observed_s * fs - coarse_peak` 或校准后的 fractional residual，单位统一为 sample，并按训练集标准差缩放。
5. 先做帧级监督预训练，再冻结或低学习率接入组分模型。

不得重新让网络从整段 5000 点隐式学习绝对 TOF。

## 8. 评估与验收

### 8.1 帧级 fidelity

| 指标 | exact-template gate | train-calibrated template gate |
| --- | ---: | ---: |
| peak MAE | ≤0.05 sample | ≤0.15 sample |
| peak P95 绝对误差 | ≤0.10 sample | ≤0.25 sample |
| calibration 后 peak bias | ≤0.02 sample | ≤0.05 sample |
| sound speed 对 stored estimated MAE | ≤0.05 m/s | ≤0.15 m/s |

正式报告同时给出 val/test/extrap fidelity，仿真 truth 只作为审计 target，不进入特征。

### 8.2 模型 parity

B1 相对 D0-observed：

- val O₂ R² 差距不超过 0.05；
- test 与 extrapolation O₂ R² 差距均不超过 0.05；
- CO₂ / N₂ 不出现大于 0.03 的系统性退化；
- `sum_abs_error`、O₂ bins、CO₂ bins 完整报告。

通过该 gate 后，D0-observed 才可由“测量级上限”升级为已实现 RawDSP 链路的可复现 baseline。

### 8.3 非线性头

B5–B7 的共同通过线：

- val O₂ R² ≥ 0.4726；
- test 与 extrapolation 同步优于 B1；
- train-val gap 有解释，不能只提升 train；
- `sum_abs_error` 不得明显劣于对应 observed 头；
- 不以 o2_bins 全部转正作为通过条件。

### 8.4 必须新增的诊断

1. peak error histogram 和 phase 分层误差；
2. corrected TOF error，而非绝对 TOF error；
3. sound speed error vs `L_m/T_C/P_MPa/H_RH`；
4. `tof vs L_m` slope/intercept residual；
5. RawDSP 与 simulator-derived observed 特征的逐数组差异；
6. 模型 feature importance 或 permutation importance；
7. val/test/extrap 三 split 的 O₂ bins。

## 9. 停止条件与判断路径

1. **exact template 未过 fidelity gate**：实现错误或峰位映射错误，停止模型训练，先修 extractor。
2. **exact template 通过、train template 未通过**：模板标定或搜索窗口问题，允许进入 D2b-5，不允许直接换大模型。
3. **frame fidelity 通过、B1 未达 parity**：检查 phase/window 统计和 feature contract 是否与 D0 对齐；不得把问题归因于模型容量。
4. **B1 达 parity、B5–B7 无增益**：波形提取问题已解决，observed 特征的可部署非线性承接失败；保留 Ridge，停止扩大 DL。
5. **B1 与非线性头均通过但 O₂ <0.70**：接受声学路线的整体档位辨识上限，等待 TDLAS 或其他 O₂ 专用通道。
6. **仅 val 提升，test/extrap 不提升**：判为未通过，不追加正则化补丁掩盖。

## 10. 影响文件规划

| 文件 | 动作 | 责任 |
| --- | --- | --- |
| `tv3/ml/raw_dsp_features.py` | 新增 | 模板、相关、峰值插值、校准、帧级质量特征 |
| `tv3/pipeline/build_tv3_raw_dsp_features.py` | 新增 | 分块缓存构建与 manifest |
| `tv3/ml/rocket_features.py` | 扩展 | `d0_raw_dsp_physics_stats_v1` builder |
| `configs/tv3_d2b_raw_dsp_features.json` | 新增 | RawDSP 缓存配置 |
| `configs/tv3_d2b_raw_dsp_ridge.json` | 新增 | parity Ridge 配置 |
| `configs/tv3_d2b_raw_dsp_extratrees.json` | 新增 | RawDSP ExtraTrees 配置 |
| `configs/tv3_d2b_raw_dsp_mlp_target_scaled.json` | 新增 | RawDSP 目标标准化 MLP 配置 |
| `scripts/analyze_d2b_results.py` | 新增 | fidelity、parity 和 split 对比图表 |
| `tests/test_raw_dsp_features.py` | 新增 | 帧级算法测试 |
| `tests/test_tv3_raw_dsp_pipeline.py` | 新增 | 缓存、契约和 CLI 测试 |

第一版不修改 `tv3/dl/models/tof_phase_net.py`；D2 保留为已证伪历史实现，避免在坏抽象上继续叠补丁。

## 11. 建议命令

以下是计划中的稳定入口，代码落地时应保持配置驱动：

```bash
# 本地 exact-template smoke / fidelity
python -m tv3.pipeline.build_tv3_raw_dsp_features \
  --config configs/tv3_d2b_raw_dsp_features.json \
  --template-mode exact_simulator_debug \
  --max-sequences 8

# 服务器 clean 6000 正式 RawDSP cache
python -m tv3.pipeline.build_tv3_raw_dsp_features \
  --config configs/tv3_d2b_raw_dsp_features.json \
  --template-mode train_baseline_median

# D0-RawDSP Ridge parity
python -m tv3.pipeline.run_tv3_rocket_baseline \
  --config configs/tv3_d2b_raw_dsp_ridge.json

# parity 通过后
python -m tv3.pipeline.run_tv3_extratrees_baseline \
  --config configs/tv3_d2b_raw_dsp_extratrees.json

python -m tv3.pipeline.run_tv3_rocket_baseline \
  --config configs/tv3_d2b_raw_dsp_mlp_target_scaled.json
```

## 12. 验证要求

代码实施后至少执行：

```bash
python -m pytest tests/test_raw_dsp_features.py -v
python -m pytest tests/test_tv3_raw_dsp_pipeline.py -v
python -m pytest tests/test_rocket_features.py -v
python -m pytest tests/test_tv3_r7_extratrees.py -v
python -m pytest tests/test_tv3_r5_mlp.py -v
python -m pytest -q
```

后端单元测试默认 60 秒超时。正式 6000 cache 构建和模型训练属于服务器任务，不以本地 600 smoke 替代。

## 13. 风险与控制

| 风险 | 影响 | 控制 |
| --- | --- | --- |
| exact simulator template 造成过于乐观的 fidelity | 高 | 只作为 debug；正式使用 train baseline template |
| template 使用 val/test 数据 | 高 | manifest 记录 source split 与 digest，契约测试强制 train-only |
| 固定延迟或温漂变化 | 高 | per-sequence baseline 校准 + slope/intercept 诊断 |
| z-score 删除幅度信息 | 中 | timing/amplitude 双路，幅度特征使用去量化原波形 |
| 6000 数据缓存成本高 | 中 | memmap 分块、可恢复但不静默跳过的显式 chunk manifest |
| frame fidelity 好但 O₂ 无增益 | 中 | 先做 Ridge parity，区分 extractor 与回归头责任 |
| 物理上限被误判为算法失败 | 高 | 保留 oracle bins 结论，不承诺窄区间 R² 转正 |

## 14. 实施检查清单

- [x] 冻结 `tv3-raw-dsp-frame-1` 与 `d0_raw_dsp_physics_stats_v1` 契约。
- [x] exact-template synthetic 与本地 512-frame fidelity 测试通过。
- [x] train-only baseline template 构建与 digest 记录完成。
- [x] per-sequence baseline delay calibration 完成。
- [x] timing/amplitude 双路语义测试完成。
- [x] RawDSP cache 不写入 `sequences/`。
- [ ] B1 Ridge parity 在 clean 6000 完成。
- [ ] parity 通过后再运行 B5–B7。
- [ ] val/test/extrap、O₂ bins、sum_abs_error 与 fidelity 指标完整回填。
- [ ] 将正式结果同步回记忆库与本文档顶部状态。
