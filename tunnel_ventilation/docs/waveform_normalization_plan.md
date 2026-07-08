# tv3 raw 波形归一化实施计划

> 本文档落地 [dl_training_plan.md §3.4](dl_training_plan.md#34-raw-波形尺度问题与归一化方案2026-07-07) 的三层归一化方案,给出每层的精确改动点、代码片段、配置开关、验证标准与风险。2026-07-07 复审后补齐四项工程收口:归一化前幅度统计侧通道、inverse-var 权重 mean-one 归一化、CLI `--no-*` 回退开关、增强通道边界推断。
> 
> 前置:v2(`tv3_tcn_multimodal_v2`,cnn1d_tcn_fusion @ 6000 序列,slow+ultrasonic)实测 val R² 全负,已确认 raw 波形与 slow 标量尺度失衡是主因。本文档不重复物理判断,只解决工程落地。

## 1. 目标与非目标

**目标**

- 让 fusion 模型能正常训练:train_loss 不再从 6517 起步,val O₂ R² 超过 R0 基线的 0.603。
- 按 §3.4 顺序落地三层方案,每层独立可验证、可回退。
- 改动限定在 tv3 独立模块(`tunnel_ventilation/tv3/`),不侵入主线 `src/`。

**非目标**

- 不改物理仿真链路(声学/光学/慢通道生成代码)。
- 不改数据集已落盘的 int16 + per-timestep scale 存储格式。
- 不动 `phase_window_tcn.py`(旧方案已归档,虽共用 `waveform_adc_scale` 但不同步改)。
- 不引入新模型结构(层 3 的 FiLM/gated fusion 属于 §3.4 既定方案,不算新结构)。

## 2. v2 失效现状复盘

实测来源:`outputs/tv3_tcn_multimodal_v2/s42/metrics.json`,配置 `configs/tv3_tcn_multimodal_v2.json`。

| 指标         | epoch 1 | best(epoch 6) | epoch 16(early stop) |
| ---------- | -------:| -------------:| --------------------:|
| train_loss | 6517.59 | 86.99         | 54.79                |
| val_loss   | 221.70  | 15.74         | 49.73                |
| val CO₂ R² | -33.70  | -0.34         | -0.29                |
| val O₂ R²  | -111.25 | -3.12         | -30.51               |
| val N₂ R²  | -545.02 | -43.17        | -121.89              |

判读:

- train_loss 从 6517 起步,证明首步前向的 raw 波形幅度与 slow 标量差几个数量级,梯度被波形分支主导。
- best epoch 6 的 CO₂ R² 接近 0,O₂/N₂ 仍深度负值,说明模型只在 CO₂(有 V_NDIR_CO2 强信号)上学到一点东西,O₂/N₂ 完全没学到。
- best 后 val_loss 重新发散,early stop 在 epoch 16 触发(patience=10)。
- 配置 `waveform_adc_scale: 5.0` 是唯一缩放,dequantize 后电压除以 5,与 slow scaler 后的 z-score 标量仍差几个数量级。

**v2 配置与文档 §4.1 的差异(实施前需决策)**:v2 用 `loss.weighting: "inverse_train_var"`,文档 §4.1 推荐 `loss_weights: [1.0, 2.0, 1.0]`。两者不同:inverse_train_var 按训练集各组分方差取逆加权,N₂ 占比高、方差大,会拿到更小权重,与"O₂ 加 2×"的意图相反。本计划在 v3 中先保留 `inverse_train_var`（仅改归一化,便于单变量归因）,另建 v3b 配置切到 [1,2,1] 做对照实验（见 §5.1b）。

## 3. 现状代码分析

### 3.1 数据层 `tv3/dl/data/dataset.py`

`V4BenchmarkDataset._waveform_values`(第 229-243 行)是波形进入网络的唯一入口:

```python
def _waveform_values(self, waveform, scale, src_idx, window_masks, *, modality):
    values = waveform[src_idx]
    if self._dequantize_waveforms:
        if scale is None:
            raise ValueError(f"{modality}_scale is required when dequantize_waveforms=true")
        values = values.astype(np.float32) * scale[src_idx].astype(np.float32)[:, np.newaxis]
    return self._apply_window(values, src_idx, window_masks)
```

- `scale[src_idx]` 是 per-timestep 标量,`[:, np.newaxis]` 广播到 5000 点,所以每帧所有点共用一个 scale。
- dequantize 后无任何归一化,直接返回原始电压值。
- slow 通道在 `_build_single_input`(第 189-190 行)经 `apply_scaler` 做 z-score,波形没有对应处理。

构造函数(第 45-60 行)参数列表,目前无 `normalize_waveforms` 开关。

### 3.2 模型层 `tv3/dl/models/cnn1d_tcn_fusion.py`

`DeepAcousticEncoder1D.forward`(第 56-70 行):

```python
flat = waveform.reshape(batch_size * timesteps, 1, waveform_length).float() / self.waveform_adc_scale
encoded = self.encoder(flat)  # encoder 内部第一层是 BatchNorm1d(第 46 行)
```

- `waveform_adc_scale` 默认 524287.0(2^19-1,20-bit ADC 满量程),v2 配置覆盖为 5.0。
- encoder 内部用 `BatchNorm1d`(第 46 行),batch=16 临界,小样本下统计量不稳。

`CNN1DTCNFusionRegressor.forward`(第 313-343 行)拼接处(第 325-330 行):

```python
parts = [self.ultrasonic_encoder(ultrasonic)]
if self.fiber_mic_encoder is not None:
    fiber_mic = x[:, :, ultrasonic_end:]
    parts.append(self.fiber_mic_encoder(fiber_mic))
parts.append(self.slow_encoder(slow))
fused = torch.cat(parts, dim=-1)
```

拼接前无任何 norm,ultrasonic embedding(64 维)与 slow embedding(64 维)直接 concat。

### 3.3 配置层 `tv3/dl/cli.py`

配置传入路径:`DEFAULT_DL_CONFIG`(第 49 行 `dequantize_waveforms: False`)→ `_resolve_args` 合并 JSON 与 CLI → `run()` 调用 `_build_dataset`(第 212 行传参)→ `_build_dataset`(第 611-652 行)构造 `V4BenchmarkDataset`。

`dequantize_waveforms` 在 cli.py 出现的位置:第 49(默认)、212(train 构造)、295(val `_optional_loader` 调用)、357(eval splits 循环，遍历 val/test/extrapolation)、389(metrics payload 记录)、620(`_build_dataset` 签名)、646(传入 `V4BenchmarkDataset`)、1013/1030(`_optional_loader` 签名及内部转发给 `_build_dataset`)、1109(`_run_config_payload` 记录)。加 `normalize_waveforms` 开关需同步这些位置。

### 3.4 增强顺序

`_build_single_input`(第 185-221 行)流程:`_waveform_values`(dequantize)→ concatenate → `augment_sequence`。层 1 z-score 会插在 dequantize 之后、concatenate 之前,即增强在归一化之后执行。`augment_sequence` 的变换包括 window_fraction、max_shift、amplitude_scale_range、jitter_std(用 `np.std` 做尺度)、gaussian_noise_std(绝对噪声)。

## 4. 三层方案落地设计

### 层 1:数据层 per-timestep z-score

**对标**:wav2vec 2.0 "raw waveform normalized to zero mean and unit variance"。

**改动点**:

1. `dataset.py` 构造函数加参数:

```python
def __init__(
    self,
    ...
    dequantize_waveforms: bool = False,
    normalize_waveforms: bool = False,   # 新增
    ...
):
    ...
    self._dequantize_waveforms = bool(dequantize_waveforms)
    self._normalize_waveforms = bool(normalize_waveforms)   # 新增
```

2. `dataset.py` `_waveform_values` 加 z-score:

```python
def _waveform_values(self, waveform, scale, src_idx, window_masks, *, modality):
    values = waveform[src_idx]
    if self._dequantize_waveforms:
        if scale is None:
            raise ValueError(f"{modality}_scale is required when dequantize_waveforms=true")
        values = values.astype(np.float32) * scale[src_idx].astype(np.float32)[:, np.newaxis]
    if self._normalize_waveforms:
        # per-timestep z-score:每帧 5000 点独立 zero-mean unit-var
        # 对标 wav2vec 2.0 整段波形 z-score;这里按帧做是因为 tv3 每帧独立承载一个声学事件
        mean = values.mean(axis=-1, keepdims=True)
        std = np.maximum(values.std(axis=-1, keepdims=True), 1e-6)
        values = (values - mean) / std
    return self._apply_window(values, src_idx, window_masks)
```

2b. `dataset.py` 可选保留归一化前幅度统计:

```python
waveform_stats_features=("log_std", "log_max_abs")
```

启用后,每个波形模态每帧追加 `log_std` / `log_max_abs` 两列到 slow 分支,顺序为 `slow + waveform_stats + waveform`。CLI 会自动把 `cnn1d_tcn_fusion.model_kwargs.slow_channels` 从物理 slow 通道数 7 扩展为 9（仅启用 ultrasonic 时）,避免 z-score 抹掉幅度/能量线索。

3. `cli.py` 加配置开关:
   
   - `DEFAULT_DL_CONFIG` 第 49 行后加 `"normalize_waveforms": False`
   - `_build_dataset` 签名(第 620 行)加 `normalize_waveforms: bool` 参数,第 646 行传入 `V4BenchmarkDataset`
   - `run()` 中 train 构造(第 212 行)、val `_optional_loader` 调用(第 295 行)、eval splits 循环(第 357 行,遍历 val/test/extrapolation)传 `normalize_waveforms=args.normalize_waveforms`
   - metrics payload(第 389 行)与 `_run_config_payload`(第 1109 行)加 `"normalize_waveforms": args.normalize_waveforms`

4. 配置 json:`"normalize_waveforms": true`,`waveform_adc_scale` 设为 `1.0`(z-score 后 adc 缩放失去意义,保留参数避免模型构造校验失败)。

**与 `waveform_adc_scale` 的关系**:层 1 生效后,进入 `DeepAcousticEncoder1D` 的波形已是 unit variance,`waveform_adc_scale` 应设为 1.0。不删除参数是为了不破坏模型构造校验(第 32-33 行 `waveform_adc_scale > 0` 检查)和 `phase_window_tcn.py` 的兼容。

**预期效果**:train_loss 不再从 6517 起步,首步前向的波形分支与 slow 分支梯度量级一致。

### 层 2:模型层拼接前 LayerNorm

**对标**:UTOPYA "各模态独立编码到共享维度,拼接前 LayerNorm 稳定尺度"。

**改动点**:

1. `cnn1d_tcn_fusion.py` `CNN1DTCNFusionRegressor.__init__` 加 norm 模块:

```python
# 在 slow_encoder 构造之后(第 267 行后)加:
self.ultrasonic_norm = nn.LayerNorm(waveform_embedding_dim)
if fiber_mic_channels > 0:
    self.fiber_mic_norm = nn.LayerNorm(waveform_embedding_dim)
else:
    self.fiber_mic_norm = None
self.slow_norm = nn.LayerNorm(slow_embedding_dim)
```

2. `forward` 拼接处(第 325-330 行)改:

```python
parts = [self.ultrasonic_norm(self.ultrasonic_encoder(ultrasonic))]
if self.fiber_mic_encoder is not None:
    fiber_mic = x[:, :, ultrasonic_end:]
    parts.append(self.fiber_mic_norm(self.fiber_mic_encoder(fiber_mic)))
parts.append(self.slow_norm(self.slow_encoder(slow)))
fused = torch.cat(parts, dim=-1)
```

3. 配置开关:加 `"fusion_layer_norm": true`(默认开,可通过配置关闭做消融)。

**与层 1 的关系**:独立,可叠加。层 1 解决输入尺度,层 2 解决 embedding 尺度。即使层 1 已做 z-score,encoder 输出的 embedding 仍可能因各分支容量不同而尺度不一,LayerNorm 保证拼接前对齐。

**不改动 `DeepAcousticEncoder1D` 内部的 BatchNorm1d**(第 46 行):batch=16 临界但 v2 未报告 BatchNorm 失效证据,先不动;若层 1+2 后仍不稳,再考虑换 LayerNorm(属层 3 范围)。

### 层 3:FiLM 调制 + gated fusion

**对标**:UTOPYA FiLM 条件化 + gated fusion。仅在层 1+2 后 O₂ R² 仍低于 R0 的 0.603 时启用。

**改动点**:

1. 新增 `FiLMModulation` 模块(slow embedding 作 context 调制 ultrasonic embedding):

```python
class FiLMModulation(nn.Module):
    """slow context 调制 ultrasonic embedding:z' = γ ⊙ z + β。
    初始化 γ=1, β=0(恒等启动),避免早期破坏编码器特征。"""
    def __init__(self, context_dim, feature_dim):
        super().__init__()
        self.proj = nn.Linear(context_dim, 2 * feature_dim)
        # 初始化为恒等:γ=1, β=0
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.proj.bias.data[:feature_dim].fill_(1.0)       # γ=1(前半段)
    def forward(self, feature, context):
        gb = self.proj(context)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma * feature + beta
```

2. 新增 `GatedFusion` 模块:

```python
class GatedFusion(nn.Module):
    """门控加权融合替代直接 concat:g_i = σ(W[ẑ_i; c])。"""
    def __init__(self, *dims):
        super().__init__()
        self.gates = nn.ModuleList([nn.Linear(sum(dims), d) for d in dims])
    def forward(self, *embeddings):
        concat = torch.cat(embeddings, dim=-1)
        weighted = []
        for gate, emb in zip(self.gates, embeddings):
            g = torch.sigmoid(gate(concat))
            weighted.append(g * emb)
        return torch.cat(weighted, dim=-1)
```

3. `CNN1DTCNFusionRegressor` 加 `fusion_mode: str = "concat"` 参数,支持 `"concat"`(默认,层 2 行为)/`"film_gate"`(层 3)。`forward` 按 `fusion_mode` 分支:`film_gate` 时 `ultrasonic' = FiLM(ultrasonic, slow)`,再 `GatedFusion(ultrasonic', slow)`。

4. 配置:`"fusion_mode": "film_gate"`。

**物理合理性**:O₂ 主要辨识力在超声(声速差 6.4%),slow 提供 T/P/RH 环境 context。用 slow 调制 ultrasonic 比直接拼接更符合"环境条件影响声学读数"的物理结构。

## 5. 配置模板

### 5.1 层 1 配置 `tv3_tcn_multimodal_v3.json`

基于 v2 配置改动，**仅改归一化，保留 v2 的 loss weighting**（便于单变量归因）：

```json
{
  "output_dir": "outputs/tv3_tcn_multimodal_v3",
  "model": "cnn1d_tcn_fusion",
  "model_kwargs": {
    "output_mode": "raw3",
    "slow_channels": 7,
    "ultrasonic_channels": 5000,
    "fiber_mic_channels": 0,
    "waveform_embedding_dim": 64,
    "waveform_adc_scale": 1.0,
    "acoustic_channels": [16, 32, 64, 64],
    "acoustic_kernel_size": 7,
    "acoustic_dropout": 0.15,
    "slow_hidden_dim": 32,
    "slow_embedding_dim": 64,
    "tcn_channels": [128, 128, 128],
    "tcn_kernel_size": 3,
    "tcn_dropout": 0.30,
    "shared_hidden_dims": [128, 64]
  },
  "modalities": "slow,ultrasonic",
  "scaler_path": "data/tv3-formal-6000/scalers/scaler_slow_sequence.json",
  "dequantize_waveforms": true,
  "normalize_waveforms": true,
  "waveform_stats_features": "log_std,log_max_abs",
  "augment": null,
  "epochs": 50,
  "batch_size": 16,
  "lr": 0.0001,
  "loss": {
    "name": "weighted_component_mse",
    "weighting": "inverse_train_var",
    "weight_normalization": "mean_one",
    "component_count": 3
  }
}
```

相对 v2 的改动：`waveform_adc_scale` 5.0→1.0、新增 `normalize_waveforms: true`、新增 `waveform_stats_features` 保留幅度统计、显式 `"augment": null`。loss weighting 仍使用 inverse train variance,但新增 `weight_normalization:"mean_one"` 控制整体 loss 尺度。

### 5.1b 层 1 对照配置 `tv3_tcn_multimodal_v3b.json`

在 v3 基础上，将 loss weighting 改为 §4.1 推荐的 fixed [1,2,1]，用于归因 loss weighting 的独立影响：

```json
{
  "output_dir": "outputs/tv3_tcn_multimodal_v3b",
  "loss": {
    "name": "weighted_component_mse",
    "weighting": "fixed",
    "loss_weights": [1.0, 2.0, 1.0],
    "component_count": 3
  }
}
```

其余字段与 v3 一致。v3 与 v3b 的唯一差异是 loss weighting（inverse_train_var vs fixed [1,2,1]），对比即可归因。算力紧张时可跳过 v3b，直接用 v3 结果推进。

### 5.2 层 2 配置 `tv3_tcn_multimodal_v3_l2.json`

层 1 基础上加:

```json
{
  "output_dir": "outputs/tv3_tcn_multimodal_v3_l2",
  "model_kwargs": { ..., "fusion_layer_norm": true },
  "normalize_waveforms": true,
  "augment": null
}
```

### 5.3 层 3 配置 `tv3_tcn_multimodal_v3_l3.json`

层 2 基础上加:

```json
{
  "output_dir": "outputs/tv3_tcn_multimodal_v3_l3",
  "model_kwargs": { ..., "fusion_mode": "film_gate" },
  "augment": null
}
```

## 6. 执行顺序与验证

### 6.1 执行顺序

```
S1  层 1 代码改动(dataset.py + cli.py)
       → 本地 smoke(tv3-smoke,50 step)验证 train_loss 起步值下降
       → 验证:train_loss 起步 < 100(对比 v2 的 6517)
S2  层 1 服务器重跑 v3(6000 序列,50 epoch,单 seed,inverse_train_var)
       → 验证:val O₂ R² > 0(超过 v2 的 -3.12);若 > 0.603 则层 1 已足够,停止
S2b (可选) v3b(fixed [1,2,1])与 v3 对照,归因 loss weighting 影响
S3  若 S2 未达标,叠加层 2(模型 LayerNorm)
       → 本地 smoke 验证不报错
       → 服务器重跑 v3_l2
       → 验证:val O₂ R² 相对 S2 有提升
S4  若 S3 仍 < 0.603,上层 3(FiLM + gated fusion)
       → 本地 smoke 验证 forward/backward 通路
       → 服务器重跑 v3_l3
       → 验证:val O₂ R² 是否达到 0.603
S5  汇总三层结果,回填 dl_training_plan.md §10 执行顺序表
```

### 6.2 smoke 验证命令

本地用 tv3-smoke(小规模)跑 2-3 epoch,确认通路与起步 loss:

```bash
# 在 tv3 独立模块下
python -m tv3.dl.cli --config configs/tv3_tcn_multimodal_v3.json \
  --dataset-dir data/tv3-smoke --epochs 3 --output-dir outputs/tv3_v3_smoke
```

(具体命令以 cli.py 实际参数为准,实施时核对 `--config` 是否支持覆盖 `--dataset-dir`。)

### 6.3 服务器验证标准

每层单 seed 50 epoch,看以下指标:

| 指标              | v2 基线  | 层 1 目标 | 层 2 目标 | 层 3 目标  |
| --------------- | ------:| ------:| ------:| -------:|
| train_loss 起步   | 6517   | < 100  | 同层 1   | 同层 1    |
| val O₂ R² best  | -3.12  | > 0    | > 层 1  | ≥ 0.603 |
| val N₂ R² best  | -43.17 | > 0    | > 层 1  | > 0     |
| val CO₂ R² best | -0.34  | > 0.5  | > 0.8  | > 0.9   |

层 1 的核心验证点是 train_loss 起步值——若仍从数千起步,说明 z-score 未生效或位置不对,需排查。

## 7. 风险与注意事项

### 7.1 augmentation 与归一化的交互

层 1 z-score 在增强之前执行。`augment_sequence` 的变换:

- `jitter_std`:用 `np.std(augmented, axis=0)` 做尺度,z-score 后各通道 std≈1,jitter 实际退化为准绝对噪声(与 `gaussian_noise_std` 类似),`jitter_std` 的配置值直接决定噪声标准差,不再自适应。启用增强时需注意两者的效果重叠。
- `gaussian_noise_std`:**绝对噪声**,z-score 后数据 unit variance,原配置的噪声值(针对原始电压尺度)会过大。启用增强时需把 `gaussian_noise_std` 重新标定到 0.01-0.1 量级。
- `amplitude_scale_range`:已修复为由 CLI 按实际 slow 通道数推断 `amplitude_apply_from_channel`。tv3 full slow 为 7 通道;若启用 `waveform_stats_features`,这些统计列会被视为 slow 分支的一部分并跳过幅度缩放,缩放只作用于 raw waveform 列。

**建议**:层 1/2/3 验证阶段先关增强(`augment: null`),确认归一化效果后再单独开启增强并标定参数。

### 7.2 loss weighting 决策

v2 用 `inverse_train_var`，文档 §4.1 推荐 [1,2,1]。本计划在 §5.1 中将两者拆分为独立配置：v3 保留 `inverse_train_var`（仅改归一化，单变量归因），v3b 切到 fixed [1,2,1]（归因 loss weighting 的独立影响）。算力紧张时可跳过 v3b，但需在结果中注明 loss weighting 未单独验证。

### 7.3 BatchNorm 在小 batch 下的风险

`DeepAcousticEncoder1D` 内部第一层是 `BatchNorm1d`(第 46 行),batch=16 临界。层 1/2 不动它。若层 1+2 后训练仍不稳(表现为 val_loss 抖动大),可考虑把 encoder 内部 BatchNorm1d 换成 LayerNorm——但这属于结构改动,归入层 3 的备选项,需单独消融。

### 7.4 `waveform_adc_scale` 的遗留

层 1 z-score 后,`waveform_adc_scale` 设为 1.0 仅作占位。`phase_window_tcn.py` 也用这个参数(第 29/50/58/124/158 行),如果未来恢复旧方案,需把该值改回 524287.0 或 5.0。配置层面建议在 v3 配置注释里标明。

### 7.5 fiber_mic 恢复时的兼容

当前 `--skip-fiber-mic`,fiber_mic_channels=0。层 1 的 z-score 对 fiber_mic 同样生效(走同一个 `_waveform_values`)。层 2 的 `fiber_mic_norm` 在 `fiber_mic_channels=0` 时不构造。未来恢复 fiber_mic 不需要额外改动。

## 8. 回退方案

每层都是配置开关控制,回退只需改配置:

- 层 1 回退:`normalize_waveforms: false` + `waveform_stats_features: null` + `waveform_adc_scale: 5.0`(回到 v2)。CLI 也支持 `--no-normalize-waveforms` / `--no-dequantize-waveforms` 覆盖配置文件中的 true。
- 层 2 回退:`fusion_layer_norm: false`。
- 层 3 回退:`fusion_mode: "concat"`(回到层 2)。

代码改动保持向后兼容:所有新参数默认值与 v2 行为一致(`normalize_waveforms=False`、`waveform_stats_features=None`、`fusion_layer_norm=False`、`fusion_mode="concat"`),不破坏现有 v2 配置的可复现性。v3 使用 `inverse_train_var + mean_one`,回退需同时移除 `weight_normalization`。

## 9. 改动文件清单

| 层   | 文件                                         | 改动                                                                                                          |
| --- | ------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| 1   | `tv3/dl/data/dataset.py`                   | 构造函数加 `normalize_waveforms` / `waveform_stats_features`;`_waveform_values` 加 per-timestep z-score 与归一化前幅度统计 |
| 1   | `tv3/dl/cli.py`                            | `DEFAULT_DL_CONFIG` 加默认值;全链路传参;metrics/run_config 记录;归一化契约校验;`--no-*` 回退开关                                  |
| 1   | `configs/tv3_tcn_multimodal_v3.json`       | 新建,基于 v2 改 `normalize_waveforms`/`waveform_adc_scale`,保留 inverse_train_var                                  |
| 1   | `configs/tv3_tcn_multimodal_v3b.json`      | 新建(可选),v3 基础上改 loss weighting 为 fixed [1,2,1],用于对照归因                                                        |
| 2   | `tv3/dl/models/cnn1d_tcn_fusion.py`        | `__init__` 加 LayerNorm 模块;`forward` 拼接前过 norm;加 `fusion_layer_norm` 开关                                      |
| 2   | `configs/tv3_tcn_multimodal_v3_l2.json`    | 新建                                                                                                          |
| 3   | `tv3/dl/models/cnn1d_tcn_fusion.py`        | 新增 `FiLMModulation`/`GatedFusion` 类;`__init__` 加 `fusion_mode`;`forward` 分支                                 |
| 3   | `configs/tv3_tcn_multimodal_v3_l3.json`    | 新建                                                                                                          |
| 测试  | `tests/test_tv3_waveform_normalization.py` | 新增 z-score、幅度统计侧通道、LayerNorm、FiLM/gate、loss weighting、CLI 配置边界测试                                            |

## 10. 与 dl_training_plan.md 的对应关系

本计划是 §3.4 的工程落地,不改变 §3.4 的技术结论。落地后已回填:

- §10 执行顺序表 P-4b 状态更新为"代码+服务器验证全部完成"，记录 v3_l2 最优结果及 P-9c 确认触发。
- §2.5 v2 失效解释更新：不仅是 raw 波形输入尺度问题，更关键的是 encoder 输出 embedding 尺度不对齐（层 2 LayerNorm 为主要改善）。
- §3.4 执行顺序标记三层实测结果，确认均未达标。
- P-9c 从"预警"升级为"已确认触发"：fusion v3_l2 O₂ R²=-0.061 < 0.50。

层 3 仍不达标,按 §10 进入 P-9c（阶段 Ⅲ-1 O₂ 专用通道）的判断。

## 11. 实施记录（2026-07-07）

### 11.1 代码落地状态

| 层   | 文件                                         | 改动                                                                                         | 状态   |
| --- | ------------------------------------------ | ------------------------------------------------------------------------------------------ | ---- |
| 1   | `tv3/dl/data/dataset.py`                   | `normalize_waveforms` 参数 + per-timestep z-score + `waveform_stats_features` 幅度统计侧通道        | ✅    |
| 1   | `tv3/dl/cli.py`                            | `--normalize-waveforms/--no-normalize-waveforms` 开关 + 全链路传参 + metrics/run_config 记录 + 契约校验 | ✅    |
| 2   | `tv3/dl/models/cnn1d_tcn_fusion.py`        | `fusion_layer_norm` 开关 + ultrasonic/fiber/slow LayerNorm                                   | ✅    |
| 3   | 同上                                         | `FiLMModulation`（含恒等初始化）+ `GatedFusion` + `fusion_mode` 分支                                 | ✅    |
| 补齐  | `tv3/dl/training/losses.py`                | `weighted_component_mse` 加 `weighting:"fixed"` 分支 + `weight_normalization:"mean_one"`      | ✅    |
| 配置  | `configs/`                                 | `tv3_tcn_multimodal_v3` / `v3b` / `v3_l2` / `v3_l3`                                        | ✅    |
| 测试  | `tests/test_tv3_waveform_normalization.py` | 26 个归一化专项测试                                                                                | ✅ 全过 |

专项验证通过:`test_tv3_waveform_normalization.py` 26 passed;相邻 DL 回归联跑 `test_tv3_waveform_normalization.py + test_tunnel_ventilation_dl_training.py` 35 passed。

### 11.2 与计划的偏差

1. **`fusion_layer_norm` 默认改为 False**（计划 §4/§8 写"默认开"）。理由：v2/v3 配置均不含此键,默认开会破坏 v2 可复现性（§8 既定要求）并使 v3 不再是"仅改归一化"（§5.1 既定目标）。默认 False 是唯一自洽选择,v3_l2 显式置 true。若需改回默认开是一行改动。
2. **补齐 `losses.py` 的 `fixed` 分支**。计划 §5.1b 配置用 `weighting:"fixed"` + `loss_weights`,但原 `_resolve_weighted_loss_kwargs` 只支持 `inverse_train_var`,跑 v3b 会报 `unknown weighting`。加了 fixed 分支读取 `loss_weights` → `component_weights`（tv3 独立模块,不影响 hg/sg）。
3. **新增 `waveform_stats_features` 幅度统计侧通道**。per-timestep z-score 会去除 raw 波形的绝对幅度/能量信息,而 CO₂ 衰减与声学能量仍可能有辨识价值。当前用 `log_std,log_max_abs` 两列保留归一化前幅度统计,由 slow 分支消费;启用 ultrasonic 时 slow 分支输入从 7 维自动扩展为 9 维。
4. **inverse-var 权重改为显式 mean-one 归一化**。v3/v3_l2/v3_l3 仍按训练集方差反比决定相对权重,但配置加 `weight_normalization:"mean_one"` 控制整体 loss 尺度;v3b fixed [1,2,1] 不做归一化。

### 11.5 复审修复闭环（2026-07-07）

复审发现的代码质量问题已收口:

- 增强 `amplitude_apply_from_channel` 不再硬编码 8,改为按实际 slow 通道数推断。
- `normalize_waveforms=true` 会校验 `dequantize_waveforms=true` 且 `waveform_adc_scale=1.0`。
- `raw3` 输出头由训练标签均值初始化 bias,降低百分比尺度目标的初始偏置。
- `resolved_loss.component_weights` 写入 `metrics.json` 与 `run_config.json`,便于复盘。
- `film_gate` 强制要求 `fusion_layer_norm=true`;GatedFusion 中性初始化为 `2*sigmoid(0)=1`。
- CLI 支持 `--no-normalize-waveforms` / `--no-dequantize-waveforms` 覆盖配置文件。

### 11.3 smoke 验证结果（tv3-smoke 32 序列，3 epoch，CPU，v3 完整模型规模）

| 指标                | v2 设置（normalize=false, adc=5.0） | v3 设置（normalize=true, adc=1.0） |
| ----------------- | -------------------------------:| ------------------------------:|
| epoch1 train_loss | 27551                           | 26161                          |
| epoch3 train_loss | 27413                           | 21058                          |
| epoch1 val_loss   | 1229                            | 420                            |
| epoch3 val_loss   | 1273（发散）                        | 326（持续下降）                      |

判读：

- v3 val_loss 比 v2 低约 3 倍且持续下降,v2 val_loss 反而发散。z-score 改善了训练动态。
- **起步 train_loss 未达 §6.1 S1 的"< 100"目标**（smoke 上 2.6 万）。分析：encoder 内部 `BatchNorm1d` 已部分抹平波形输入尺度差异,z-score 对起步 forward loss 影响有限；起步 loss 高更可能来自 `inverse_train_var` 在小样本上的极端权重 + output head 初始化。
- 这提示 §2/§3.4 把 v2 失效主因完全归结为"raw 波形尺度失衡"可能不完全准确,但 z-score 仍有正面价值（val_loss 改善）。

### 11.4 服务器验证结果（tv3-formal-6000，50 epoch，单 seed，RTX 5880）

四组实验均在 `tv3-formal-6000`（6000 序列 / 600 mixture）上完成，patience=10 early stopping，seed=20260704。

#### 11.4.1 总览

| 配置 | 改动 | best ep | val_loss | val R² | val MAE (%) | early stop ep |
| --- | --- | ---:| ---:| ---:| ---:| ---:|
| R0 (Ridge) | 传统 ML 基线 | — | — | **0.918** | **0.28** | — |
| v2 | 原始 fusion, adc=5.0 | 6 | 86.99 | -295 | 18.47 | 16 |
| **v3** | +z-score +幅度统计 +inv-var | 11 | 6.53 | -3.03 | 2.22 | 21 |
| **v3b** | 同 v3, loss 改 fixed [1,2,1] | 3 | 5.78 | -1.66 | 1.77 | 13 |
| **v3_l2** | v3 + fusion LayerNorm | 14 | **1.48** | **+0.019** | **1.13** | 24 |
| **v3_l3** | v3_l2 + FiLM gate | 10 | 1.48 | -0.001 | 1.14 | 20 |

#### 11.4.2 各组分 best epoch 指标

| 配置 | CO₂ R² | O₂ R² | N₂ R² | CO₂ MAE | O₂ MAE | N₂ MAE | sum_abs_err |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| R0 (Ridge) | 0.994 | 0.661 | 0.939 | 0.09 | 0.42 | 0.33 | — |
| v2 | -33.7 | -111.2 | -545 | 7.18 | 9.27 | 38.9 | — |
| **v3** | -5.97 | -3.42 | -0.73 | 3.28 | 1.62 | 1.77 | 3.59 |
| **v3b** | -2.96 | -1.55 | -0.72 | 2.34 | 1.19 | 1.78 | 3.09 |
| **v3_l2** | **+0.040** | -0.061 | **+0.028** | **1.21** | **0.82** | **1.36** | **0.12** |
| **v3_l3** | +0.003 | -0.004 | -0.002 | 1.24 | 0.80 | 1.39 | 0.03 |

#### 11.4.3 与 §6.3 验证标准的对比

| 指标 | v2 基线 | 层 1 目标 | 层 1 实际 | 层 2 目标 | 层 2 实际 | 层 3 目标 | 层 3 实际 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:| ---:|
| train_loss 起步 | 6517 | < 100 | 10389 ❌ | 同层 1 | 891 ❌ | 同层 1 | 750 ❌ |
| val O₂ R² best | -3.12 | > 0 | -3.42 ❌ | > 层 1 | -0.061 ✅ | ≥ 0.603 | -0.004 ❌ |
| val N₂ R² best | -43.17 | > 0 | -0.73 ❌ | > 层 1 | +0.028 ✅ | > 0 | -0.002 ❌ |
| val CO₂ R² best | -0.34 | > 0.5 | -5.97 ❌ | > 0.8 | +0.040 ❌ | > 0.9 | +0.003 ❌ |

**结论：没有任何一层达到计划设定的目标。** 最好的 v3_l2 在 R² 上仅略微为正（0.019），距离 R0 Ridge 基线的 0.918 差距巨大。

#### 11.4.4 逐层归因分析

**层 1（v3 vs v2）——z-score 解决了训练崩溃，但 R² 仍全负**

v2 完全失败（R² 在 -100 到 -500），v3 把 val_loss 从 87 降到 6.5，R² 从 -295 升到 -3.0。z-score 确实解决了训练不收敛，但 R² 仍然全负。train_loss 第 1 epoch 仍高达 10389（起步高于 v2 的 6517），但第 2 epoch 即降到 24.3、第 3 epoch 降到 1.55 并趋于稳定。起步 loss 高源于 `raw_output_prior` 初始化后第一轮梯度更新未生效，不代表 z-score 无效。

**v3b vs v3——fixed [1,2,1] 优于 inverse_train_var**

v3b 在所有指标上优于 v3：val R² -1.66 vs -3.03，val MAE 1.77 vs 2.22，收敛更快（best epoch 3 vs 11）。inverse_train_var 的权重 [0.73, 1.76, 0.51] 把 N₂ 权重压到 0.51（N₂ 方差大→反比权重小），削弱了 N₂ 的学习信号。fixed [1,2,1] 对 N₂ 保留了 1.0 权重，O₂ 显式加权 2.0。两者 N₂ R² 几乎相同（-0.72 vs -0.73），但 CO₂ 和 O₂ 差距明显。

**层 2（v3_l2 vs v3）——fusion LayerNorm 是最关键的改进**

这是四组实验中最大的单步提升：val_loss 从 6.53 降到 1.48（4.4 倍），R² 首次转正到 +0.019，val MAE 从 2.22 降到 1.13（减半），O₂ MAE 从 1.62 降到 0.82（减半），sum_abs_error 从 3.59% 降到 0.12%。LayerNorm 在 embedding 拼接前对齐了 ultrasonic encoder 和 slow encoder 输出的尺度，效果比输入层的 z-score 更直接。这说明 v2 失效的关键不仅是"raw 波形输入尺度大"，更在于"两个 encoder 输出的 embedding 尺度不一致导致 TCN 无法有效融合"。

**层 3（v3_l3 vs v3_l2）——FiLM + gated fusion 无额外收益**

v3_l3 与 v3_l2 几乎一致，val_loss 从 1.476 微增到 1.483，R² 从 +0.019 回退到 -0.001。FiLM 调制和门控融合增加了模型复杂度但没有带来收益，可能原因：当前架构的表达力不是瓶颈，而是 CNN1D encoder 从 waveform 中提取的特征本身缺乏区分性；600 mixture 的数据量也不足以支撑更复杂的融合架构。

#### 11.4.5 test / extrapolation 泛化验证

v3b 和 v3_l2 的 test set 结果与 val set 基本一致，未见过拟合：

| 配置 | test R² | test MAE | extrap R²（仅 v3b） |
| --- | ---:| ---:| ---:|
| v3b | -1.66 | 1.78 | -1.56 |
| v3_l2 | +0.014 | 1.14 | —（未配置） |
| v3_l3 | -0.004 | 1.16 | — |

v3b 的 extrapolation split R² = -1.56，与 val/test 持平，说明模型在外推区域也没有严重退化。

#### 11.4.6 条件分析（O₂ 分箱）

v3_l2 在 O₂ 浓度中间段（18.8%–20.4%）表现最好，O₂ MAE 降到 0.42–0.46%；但在高 O₂ 段（20.4%–21.2%）和低 O₂ 段（18.0%–18.8%），O₂ MAE 回升到 1.19–1.22%，R² 降到 -27 至 -30。这与 O₂ 浓度范围窄（18%–21%、仅 3 个百分点跨度）导致 bin 内方差极小有关——R² 的分母很小，微小偏差就会导致 R² 深度负值。

CO₂ 分箱中，v3_l2 在中间段（1.3%–3.8%）MAE 仅 0.54–0.67%，但在低 CO₂ 段（0–1.3%）和高 CO₂ 段（3.8%–5.0%）MAE 分别为 1.66% 和 1.90%，存在明显的边界效应。

### 11.5 结论与后续方向

#### 三层归一化方案的整体评价

三层方案成功地把 fusion 模型从完全失败（v2, R² = -295）带到了勉强有效（v3_l2, R² ≈ 0），但距离 R0 Ridge 基线（R² = 0.918）仍有数量级差距。三层中 **fusion LayerNorm（层 2）贡献了绝大部分改善**，z-score（层 1）解决了"不崩溃"，FiLM/gate（层 3）无额外收益。

#### DL fusion 远逊于 Ridge 的根因分析

1. **Ridge 的优势在于特征层面**：Ridge 直接在 7 个 slow 特征上做线性回归，这些特征（V_NDIR_CO2、V_TCS 等）与目标有强线性/近线性相关，Ridge 天然适合这种场景。
2. **CNN1D encoder 未从 waveform 中提取到有效特征**：5000 点 raw waveform 经 4 层 CNN1D → 64 维 embedding，这个 embedding 可能只捕获了波形的粗糙统计量（能量、包络），而非物理上关键的 TOF（飞行时间）差异。TOF 差异体现在亚样本级别的相位偏移（~0.002μs），通用 CNN 架构难以精确提取。
3. **数据量不足**：600 mixture / 6000 序列对于端到端从 waveform 学习组分浓度而言偏少，模型难以泛化。
4. **slow 分支被淹没**：虽然 LayerNorm 对齐了 embedding 尺度，但 TCN 融合后 slow 分支的 7/9 个强特征的信息可能在 128 维 concat 空间中被 64 维 waveform embedding 稀释。

#### 后续可选方向

按计划 §10，三层方案未达标后进入 P-9c（阶段 Ⅲ-1）。可选路径：

1. **v3_l2 + fixed [1,2,1]**：v3b 证明 fixed [1,2,1] 优于 inverse_train_var，但 v3b 未叠加 LayerNorm。在 v3_l2 基础上改用 fixed [1,2,1] 可能进一步改善。
2. **O₂ 专用通道**（§10 P-9c）：为 O₂ 设计专用的声学特征提取通道，而非共用通用 CNN1D encoder。
3. **混合架构**：slow 分支直接用线性头（对齐 Ridge 的优势），waveform 分支作为辅助特征拼接进来。
4. **TOF 特征工程**：不依赖 CNN1D 端到端学习 TOF，而是在数据层显式提取 TOF 作为额外 slow 特征（类似 `waveform_stats_features` 的思路）。
5. **数据增强 + 更大数据集**：当前 augment=null，可尝试启用增强；或增大 formal 集规模。
