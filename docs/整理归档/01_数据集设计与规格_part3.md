> 复核说明：本页承接 `01_数据集设计与规格.md`，内容聚焦生成流程与使用方式；当前实验结果与输出目录状态请以总览/流程页为准

## 13. 数据生成流程

### 13.1 生成脚本

| 脚本 | 功能 |
|------|------|
| `src/sim/scripts/generate_waveform_dataset.py` | 主入口，生成整个数据集 |
| `src/sim/scripts/acoustic_waveform_v3.py` | 通道 1 超声对射波形生成 |
| `src/sim/scripts/acoustic_fiber_mic_v3.py` | 通道 2 光纤麦克风波形生成 |
| `src/sim/scripts/acoustic_v2.py` | 共用物理函数（声速、声衰减）|
| `src/sim/scripts/check_waveform_directionality.py` | 方向性与质量检查 |

### 13.2 共用物理量

| 量 | 来源 |
|---|------|
| `c_sound` | `_hidden_sound_speed_v2(...)` |
| `alpha(f)` | `_hidden_attenuation_v2(..., f_hz=f_c)["alpha_true_v2"]` |
| 配气 + 工况 | 复用 V3.0 工况采样逻辑 |
| phase 序列 | `sim_common.phase_boundaries` |
| slow 通道 120 时步演化 | 复用现有动力学叠加 |

**注意**：两通道在每个 timestep 共享同一组 `(gas, T, P, RH, L)`，天然时间对齐。

### 13.3 生成步骤

1. **工况采样**：生成 10000 个 base condition（4 组分配比 + 温压湿 + 声程）
2. **多噪声种子**：每个 base condition 生成 3 个噪声实例（noise seed）
3. **序列生成**：每个噪声实例生成 120 时步序列
   - 慢通道：8 个传感器信号随时间演化
   - 波形通道 1：每时步 1000 采样点超声波形
   - 波形通道 2：每时步 2000 采样点光纤麦克风波形
4. **量化存储**：波形用 int16 + scale 压缩存储
5. **Split 生成**：按 mixture_id 分层分组，生成 4 路 split
6. **Scaler 拟合**：仅在 train split 上拟合 slow scaler
7. **质量检查**：运行方向性检查脚本，输出质量报告

## 14. 传统 ML 特征导出

### 14.1 特征导出脚本

- **入口**：`src/pipeline/feature_extraction.py`
- **功能**：从双通道波形提取传统 ML 表格特征
- **输出**：每个 120 时步序列抽样为 4 个代表性时间点

### 14.2 特征定义

**时间点采样策略**：
- 基线段：1 个时间点
- 目标气体稳定段：2 个时间点
- 恢复段：1 个时间点

**通道 1（超声）特征**：
- 传播时延（TOF）
- 主峰幅值
- 频域峰值
- 声速估计
- 声衰减近似

**通道 2（光纤麦克风）特征**：
- 包络衰减时间常数 tau
- 反射峰间隔
- 反射峰数量
- 尾部能量
- 混响衰减特征

**慢通道特征**：
- NDIR、TCS 传感器信号
- 温压湿环境变量
- 相对基线变化量
- 环境派生量

### 14.3 输出文件

| 文件 | 说明 |
|------|------|
| `train_acoustic.csv` | 声学模态特征（基于通道 1）|
| `train_optical.csv` | 光学模态特征（NDIR）|
| `train_thermal.csv` | 热导模态特征（TCS）|
| `feature_table.csv` | 完整特征表 |
| `feature_table_env.csv` | 含环境特征的完整表 |
| `feature_manifest.json` | 特征元数据 |

### 14.4 特征表规模

| 数据集 | 序列数 | 抽样点/序列 | 总样本数 |
|--------|--------|------------|----------|
| waveform_v3 (旧) | 10000 | 4 | 40000 |
| waveform_v3_seedpath_formal (新) | 30000 | 4 | 120000 |

## 15. 已知问题与限制

| 问题 | 影响 | 处理方式 |
|------|------|----------|
| calibration_status: pending | 还不能声称等价真实硬件采集 | 论文中只声明链路结构对齐 |
| 通道 2 反射系数 R 未校准 | tau / 尾部形态存在模型误差 | 把 R 写入 metadata，后续可重生成 |
| 光纤探头位置 l_direct_factor=0.5 是设计值 | 真实几何敏感性未验证 | 写入 metadata，必要时做参数扫描 |
| 数据体积较大（~21.6 GB）| 生成、训练、存储成本上升 | 必要时把 fiber_mic 窗口从 10 ms 缩到 5 ms |
| mixture_id 命名空间不同 | 新旧数据集不能混用 split | 明确标注版本，严格分开使用 |

## 16. 数据版本管理

### 16.1 版本锁定

- **生成 seed**：`20260514`（固定，不可变）
- **split seed**：由 split 策略自动生成（基于 mixture_id）
- **冻结策略**：数据生成完成后冻结，不再修改

### 16.2 版本兼容性

| 组件 | waveform_v3 (旧) | waveform_v3_seedpath_formal (新) |
|------|------------------|----------------------------------|
| 张量字段 | ✅ 兼容 | ✅ 兼容 |
| 慢通道顺序 | ✅ 兼容 | ✅ 兼容 |
| 标签顺序 | ✅ 兼容 | ✅ 兼容 |
| split 文件 | ❌ 不兼容 | ❌ 不兼容 |
| scaler 文件 | ❌ 不兼容 | ❌ 不兼容 |
| mixture_id | ❌ 不兼容 | ❌ 不兼容 |

**重要**：新旧数据集可以共存，但训练时必须明确指定数据集路径和对应的 split 目录。

## 17. 使用指南

### 17.1 数据加载

```python
import numpy as np

# 加载数据
data_dir = "data/waveform_v3_seedpath_formal"

# 加载波形数据（需要反量化）
ultrasonic_int16 = np.load(f"{data_dir}/sequences/ultrasonic_int16.npy")
ultrasonic_scale = np.load(f"{data_dir}/sequences/ultrasonic_scale.npy")
ultrasonic_float = ultrasonic_int16 * ultrasonic_scale[:, :, None]

fiber_mic_int16 = np.load(f"{data_dir}/sequences/fiber_mic_int16.npy")
fiber_mic_scale = np.load(f"{data_dir}/sequences/fiber_mic_scale.npy")
fiber_mic_float = fiber_mic_int16 * fiber_mic_scale[:, :, None]

# 加载慢通道和标签
slow = np.load(f"{data_dir}/sequences/slow.npy")
y = np.load(f"{data_dir}/labels/y.npy")
```

### 17.2 Split 使用

```python
import pandas as pd

# 加载 split
split_dir = "data/waveform_v3_seedpath_formal/splits"
train_ids = pd.read_csv(f"{split_dir}/train_sequence_ids.csv")
val_ids = pd.read_csv(f"{split_dir}/val_sequence_ids.csv")
test_ids = pd.read_csv(f"{split_dir}/test_sequence_ids.csv")

# 根据 sequence_id 筛选数据
train_indices = train_ids["sequence_id"].str.replace("Q", "").astype(int) - 1
train_slow = slow[train_indices]
train_y = y[train_indices]
```

### 17.3 Scaler 使用

```python
import json

# 加载 slow scaler
with open(f"{data_dir}/scalers/scaler_slow_sequence.json", "r") as f:
    scaler_config = json.load(f)

# 应用标准化
slow_normalized = (slow - scaler_config["mean"]) / scaler_config["std"]
```

## 18. 参考文档

| 文档 | 内容 |
|------|------|
| `docs/数据集生成设计.md` | V3.1 双通道数据集生成设计详细说明 |
| `docs/data_card.md` | V3.1 双通道数据卡片（本文档的原始版本）|
| `docs/waveform_v3_seedpath_formal_数据分析报告.md` | seedpath_formal 版本数据分析 |
| `docs/waveform_v3_seedpath_formal_适配说明.md` | 新数据集适配说明（DL + ML）|
| `docs/课题数据集关键信息.md` | 数据集关键信息速查 |

## 19. 数据引用

如果使用本数据集，请引用：

```
V3.1 Dual-Channel Waveform Dataset for Hydrogen-Blended Natural Gas Detection
Version: seedpath_formal (30000 sequences)
Generation seed: 20260514
Created: 2026-05-19
```

## 20. 更新日志

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-05-14 | v0.1 | 初始 waveform_v3 设计（10000 序列）|
| 2026-05-19 | v0.2 | 上线 waveform_v3_seedpath_formal（30000 序列 + 4 路 split）|
| 2026-06-08 | v0.3 | 整理归档文档，明确版本差异 |
