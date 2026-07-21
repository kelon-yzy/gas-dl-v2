# tv3 双向超声 F0：trigger jitter 双情景推导

> 状态：F0 冻结证据（2026-07-21）  
> 范围：仅支撑 `configs/tv3_bidir/parameter_registry.json` 中的双情景登记；**不是**现场部署证书，也不是 NI 对 RMS trigger jitter 的直接规格声明。

## 1. 硬件锚点（datasheet）

设备：NI USB-6453（项目超声 AI 链路）。

| 规格项 | 值 | 出处 |
| --- | --- | --- |
| 同步采样率 | 1 MS/s/ch | USB-6453 Specifications；本地摘要 `传感器硬件资料整理.md` §9.1 |
| Timing resolution | 10 ns | 同上 |
| Timing accuracy | 50 ppm of sample rate | 同上 |
| Base clock accuracy | 50 ppm | 同上 |
| External digital trigger | PFI → AI Start Trigger 等 | 同上 § External Digital Triggers |

**明确缺口**：该 datasheet **未公布** μs 量级的 RMS “trigger jitter”。因此不得把任一情景写成“厂商保证 RMS”。

## 2. 双情景定义

### 2.1 `conservative_v1`（std = 3.0 μs）

- **来源类型**：`engineering_scenario`
- **依据**：`WaveformSpec.trigger_jitter_std_s` 与 identifiability v1 登记值
- **用途**：与 v1 误差预算对照；保守上界扫描
- **不是**：datasheet RMS

### 2.2 `nominal_daq_half_sample`（std = 0.5 μs）

- **来源类型**：`literature_bound`（由 datasheet 采样率推导的工程上界）
- **推导**：
  1. \(T_s = 1 / 10^6\,\mathrm{s} = 1.0\,\mu\mathrm{s}\)（1 MS/s）
  2. 取半采样作为数字 Start Trigger + 样本时钟量化的 **上界**：\(0.5\,T_s = 0.5\,\mu\mathrm{s}\)
  3. 对照：timing resolution 地板 = 10 ns；50 ppm 时钟在 TOF≈729 μs 上贡献 ≈36 ns，远小于 0.5 μs
- **用途**：F4/F5 与保守情景并行报告的 nominal 臂
- **不声称**：子样本匹配滤波后的残余噪声；实测校准值；Sim2Real

## 3. 使用规则

1. F0 及以后凡报告 jitter 相关 P90 / nuisance，必须双情景并列，禁止只报较优臂。
2. 若后续获得厂商 RMS、板级实测或硬件触发示波器统计，应新增情景并改 `source`，不得静默覆盖本文件已冻结的两臂 ID。
3. 本推导只解除 F0「nominal 缺 datasheet 依据」门；不预判 F4 是否通过 0.4 vol% 门。
