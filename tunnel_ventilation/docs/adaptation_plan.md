# 掘进通风 CO₂/O₂/N₂ 三组分检测适配方案

> 本文档定义掘进通风场景的仿真链路适配方案。
> 场景目标、组分定义和数据契约见 [CO2_O2_N2_气体检测场景规划.md](CO2_O2_N2_气体检测场景规划.md)。
> 仿真框架详细说明见 [../dl_model_architecture.md §13](../dl_model_architecture.md#十三仿真框架dl-输入数据来源)，本文档在其基础上做组分替换和场景适配。

## 实施进度（截至 2026-07-06）

| 阶段 | 范围 | 状态 |
|------|------|------|
| 1 | Schema + 采样 | ✅ 已完成（18 tests） |
| 2 | 声学 / 热导 / 光学物理适配 | ✅ 已完成（25 tests） |
| 3 | 慢通道 + benchmark + CLI | ✅ 已完成（18 tests） |
| 4 | DL 训练适配 | ✅ 已完成（13 tests） |
| 5 | formal 数据集 + 基线训练 | 🔶 进行中（tv3-formal 600 序列已生成 + Ridge/TCN 首轮基线完成 + rocket 阶段 A 已落地） |

已落地契约：
- `composition_scheme = "tunnel_ventilation"`，`schema_version = "tunnel-ventilation-1"`
- `COMPONENT_FIELDS = ("x_CO2", "x_O2", "x_N2")`，`BACKGROUND_FIELDS = ()`
- 7 慢通道（V_NDIR_CO2 / V_TCS / T_C / P_MPa / H_RH / L_m / piston_position_m，无 V_NDIR_CH4）
- 闭包类 loss / target_transform / `gas_head` 在 tv3 下被拒绝；多模态 `cnn1d_tcn_fusion` 使用 `raw3` 直接三输出
- conditional_metrics 按 `o2_bins` / `co2_bins` 分箱
- sum_abs_error 在 tv3 下计算（数据层 sum=100% 闭包，模型层不强制归一化）
- HITRAN 后端阶段 1 未实现，CLI 拒绝 `hitran_hapi_v1`
- **2026-07-05 存储优化**：tv3 默认 `WaveformSpec(per_timestep_scale=True, waveform_dtype="int16")` + CLI `--skip-fiber-mic`；物理 ADC 仍 20-bit，存储 int16 + per-timestep scale（误差/噪声 ≈ 1%）；fiber_mic 代码保留但默认不生成；数据集 17 GB → 3 GB（600 序列）
- **2026-07-06 固定特征回归分支**：已新增 `tv3/ml/rocket_features.py`、`tv3/ml/rocket_training.py`、`tv3/pipeline/run_tv3_rocket_baseline.py` 与 `configs/tv3_rocket_ridge.json`；阶段 A 先支持 `physics_stats + RidgeCV`，用于把 O₂ / N₂ 物理信号验证与端到端 DL 训练失败解耦
- **2026-07-07 场景隔离重构**：tv3 子工程自包含化，原 `src/sim|dl|ml|pipeline|common` 全部迁入 `tunnel_ventilation/tv3/` 下，包名 `tv3`，独立 `pyproject.toml`；以下文件清单中的 `tv3/...` 路径为隔离后的实际位置（重构前位于 `src/...`）

首轮基线结果（slow-only，600 序列）：
- Ridge: CO₂ R²=0.91 ✅, O₂ R²=-0.05 ❌, N₂ R²=0.65 ❌
- TCN: 全组分 R²≈0（600 序列对 DL 严重不足）
- Rocket 阶段 A：smoke 测试已通过；R0 正式集（6000 序列）已回填——val CO₂ R²=0.993、O₂ R²=0.603、N₂ R²=0.925，O₂ 物理特征有正信号但窄分箱 R² 全负
- 详见 [experiment_roadmap.md](experiment_roadmap.md) 基线结果分析与 rocket 方向 C

---

## 一、背景

当前系统已实现两个检测场景：

- **hydrogen_ng**（掺氢天然气）：H₂/CH₄/CO₂/N₂，sum=100% 闭包，benchmark `wv4-*`
- **syngas**（合成气）：H₂/CH₄/CO₂/CO，N₂ 为背景气，sum<100%，benchmark `sg4-*`

掘进通风场景的目标是估计巷道掘进面附近空气中 CO₂/O₂/N₂ 三组分浓度，用于通风质量感知。与前两个场景的本质差异：

1. **空气基底**：组分以空气为主背景（N₂ ≈ 78%，O₂ ≈ 21%），而非可燃气体混合物
2. **N₂ 是预测目标**：不同于 syngas 中 N₂ 作为背景气，本场景直接预测 N₂
3. **O₂ 可观测性受限**：O₂ 为同核双原子，无红外指纹，仅能通过声学和热导间接推断

## 二、关键架构决策

| 编号 | 决策 | 理由 |
|---:|---|---|
| 1 | Schema 独立：`tunnel_ventilation_schema.py` | 组分种类和语义与 hg/sg 完全不同，不应复用现有 schema |
| 2 | `COMPONENT_FIELDS = ("x_CO2", "x_O2", "x_N2")` | 三组分全部为显式预测目标 |
| 3 | `BACKGROUND_FIELDS = ()` | N₂ 不是背景气，与 syngas 不同 |
| 4 | `composition_scheme = "tunnel_ventilation"` | manifest 标识，下游自动分流 |
| 5 | 不使用闭包残差头 | `sum_abs_error` 仅作监控，模型输出不做强制归一化 |
| 6 | 第一阶段不新增 O₂ 专用传感器 | 先评估现有四模态的 O₂ 可辨识性极限 |
| 7 | 复用 v6-phys-strict 仿真链路 | 200 kHz 超声、1 MS/s、20-bit ADC、L_m 0.2–0.3 m |
| 8 | 8 个慢通道（沿用 hg 默认） | 不新增 V_NDIR_CO 或 V_PARAMAGNETIC_O2 |
| 9 | 仅 CO₂ 有 NDIR 光学通道 | O₂/N₂ 为同核双原子，无红外活性 |
| 10 | 数据集前缀 `tv3-*` | 与 `wv4-*` / `sg4-*` 并存 |

## 三、实施路线

### A. Schema 与采样（阶段 1）

#### A1. 创建 `tv3/sim/core/tunnel_ventilation_schema.py`

```python
COMPONENT_FIELDS = ("x_CO2", "x_O2", "x_N2")
BACKGROUND_FIELDS = ()  # N₂ 是目标，不是背景
SLOW_CHANNELS = (
    "V_NDIR_CO2", "V_TCS",
    "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m",
)  # 7 通道，无 V_NDIR_CH4（场景无 CH₄）
```

#### A2. 创建 `tv3/sim/generation/tunnel_ventilation/conditions.py`

- 组分范围：CO₂ 0.03–5.00%，O₂ 18.00–21.20%，N₂ = 100 - CO₂ - O₂
- LHS 采样在 (CO₂, O₂) 二维空间进行
- 环境变量沿用 v6 链路范围

详见 [sampling_design.md](sampling_design.md)。

#### A3. 单元测试 `tests/test_tunnel_ventilation_schema.py`

- 组分总量一致性：`|x_CO2 + x_O2 + x_N2 - 100| < 1e-6`
- 字段顺序与 schema 一致
- 非法样本（超出范围）被拒绝
- seed 可复现性

### B. 声学与热导物理适配（阶段 2）

参考 [../dl_model_architecture.md §13.5](../dl_model_architecture.md#135-声学物理acoustic_physicspy) 中的现有实现。

#### B1. 创建 `tv3/sim/generation/tunnel_ventilation/acoustic_physics.py`

纯组分物性常数（具体数值见 [physics_references.md](physics_references.md)）：

- CO₂ 和 N₂ 的 M、cp、λ、η、弛豫参数可直接从主线 `acoustic_physics.py` 复用
- O₂ 物性已确认：cp=29.38 J/(mol·K)（NIST WebBook Shomate 手算验证）、λ₀=0.0264 W/m·K、n=0.80（NIST/CRC/Engineering ToolBox 共识）、η=2.058e-5 Pa·s（NBS TN.350 / NIST REFPROP）
- CO₂-O₂ 和 O₂-N₂ 的 Wilke φ_ij 由公式计算（无需查表），示意值见 [physics_references.md](physics_references.md) §3.4
- O₂ V-T 弛豫在 200 kHz 下贡献可忽略（dry air fr,O ≈ 24 Hz/atm，Bass 1990 JASA 公式验证），工程取 alpha_o2 ≈ 0

#### B2. 混合规则复用

声速：理想气混合 `c = sqrt(γ_mix · R · T / M_mix)`，与主线 `hidden_sound_speed_v2`（`acoustic_physics.py:48`）相同的数学形式，替换组分字段为 CO₂/O₂/N₂。

热导：Wassiljewa-Mason-Saxena 混合规则，与主线 `_hidden_lambda_mix` 相同的数学形式。CO₂-N₂ 交互参数可复用，需新增 O₂ 相关的二元组合。

衰减：从 `hidden_attenuation_v2`（`acoustic_physics.py:72`）裁剪，保留 alpha_classical + alpha_co2 + alpha_n2 + alpha_h2o，移除 alpha_ch4 + alpha_h2_diffusion，新增 alpha_o2。

#### B3. 波形验证

波形仿真（[§13.6](../dl_model_architecture.md#136-波形仿真waveformspy)）完全复用，不修改 `waveforms.py`。通过注入 `sound_speed_fn` / `attenuation_fn` 挂钩三组分物理后端。

验证项：
- 空气组成（~78% N₂ + 21% O₂ + 0.04% CO₂）下 c_mix ≈ 346 m/s（与已知空气声速交叉验证）
- 超声波形 shape：`(T, 5000)`，int32，20-bit 动态范围
- 光纤麦克风波形 shape：`(T, 10000)`

### C. 光学通道适配（阶段 2）

参考 [../dl_model_architecture.md §13.7](../dl_model_architecture.md#137-ndir-光学后端)。

#### C1. CO₂ NDIR 通道

复用 V_NDIR_CO2 通道的 empirical/HITRAN 后端。CO₂ 的 ν₃ 吸收带（2349 cm⁻¹ / 4.26 μm）是 NDIR 检测的标准波段。经验后端的 `_hidden_absorption_co2` 函数可直接调用。

#### C2. O₂/N₂ 无光学通道

O₂ 和 N₂ 均为同核双原子分子，无永久偶极矩，基频振动不产生红外吸收。不为这两个组分设置光学检测通道。

#### C3. 无串扰矩阵

与 syngas 场景不同（CO₂↔CO 存在 NDIR 串扰），本场景只有 CO₂ 一个红外活性组分，不需要串扰矩阵。

### D. 慢通道与 benchmark（阶段 3）

参考 [../dl_model_architecture.md §13.4](../dl_model_architecture.md#134-慢通道动力学slowpy)（慢通道动力学）和 [§13.8](../dl_model_architecture.md#138-打包与校验packaging--validation)（打包与校验）。

#### D1. 创建 `tv3/sim/generation/tunnel_ventilation/slow.py`

8 个慢通道，与 hg 默认通道组织方式相同。气体组分输入替换为 CO₂/O₂/N₂。动力学模式复用 multi-tau 双指数（`_multi_tau_channel_step`），phase schedule 复用 `standard_exposure` 预设（[§13.9](../dl_model_architecture.md#139-phase-schedule-系统phasepy)）。phase blend 的 baseline 语义固定为标准新鲜空气（CO₂ 0.04%、O₂ 20.90%、N₂ 79.06%），不是纯 N₂。

#### D2. Benchmark 编排

- `tv3/sim/generation/tunnel_ventilation/benchmark.py`：数据生成编排
- `tv3/pipeline/generate_tunnel_ventilation_benchmark.py`：CLI 入口

#### D3. tv3-smoke 生成与验证

```powershell
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark `
    --output-root data --dataset tv3-smoke --sequences 32 --seed 20260704 `
    --timesteps 32 --dt-s 0.5 --optical-absorption-backend empirical_v1 --workers 1
```

验证清单：
- `labels/y.npy` shape = `(N, 3)`
- `metadata/label_names.npy` = `["x_CO2", "x_O2", "x_N2"]`
- `manifest.json` 中 `composition_scheme == "tunnel_ventilation"`
- `manifest.json` 中 `background_fields == []`
- `sequences/slow.npy` 最后一维 = 7

### E. DL 训练适配（阶段 4）

#### E1. Dataset 加载兼容

`tv3/dl/data/dataset.py` 通过 `manifest.composition_scheme` 自动适配。如果现有代码已支持 manifest 驱动加载，则无需修改。

#### E2. Loss / Metrics 适配

- 允许的 Loss：`weighted_component_mse`、`mse`、`mae`、`smooth_l1`、`huber`
- 拒绝的 Loss：`compositional_mse`、`ilr_mse`、`free_component_mse`、`weighted_free_component_mse`（闭包类）
- `validate_loss_composition_scheme()` 自动拒绝闭包 loss
- `target_transform` 不可用（ILR/ALR 依赖 sum=100% 闭包头）；ML 与 DL 入口均按 manifest 读取 `composition_scheme` 后拒绝 tv3 transform

#### E3. 配置矩阵

在 `configs/` 下创建 5 个配置文件：

| 配置 | 模型 | Loss |
|------|------|------|
| `tv3_baseline.json` | CNN1D | weighted_component_mse |
| `tv3_tcn.json` | TCN | weighted_component_mse |
| `tv3_lstm.json` | LSTM | weighted_component_mse |
| `tv3_patchtst.json` | PatchTST | weighted_component_mse |
| `tv3_ridge.json` | Ridge | — (repo `RidgeRegressor`) |
| `tv3_tcn_multimodal.json` | `cnn1d_tcn_fusion` | weighted_component_mse (`raw3`, out_dim=3) |

详见 [dl_training_plan.md](dl_training_plan.md)。

## 四、已确认决策

1. N₂ 作为显式预测目标，不放入 `background_fields`。
2. 不使用闭包残差头（GasHeadNormalize 等），`sum_abs_error` 仅监控；tv3 fusion 配置必须是 `output_mode="raw3"`、`out_dim=3`。
3. 第一阶段不新增 O₂ 专用传感器通道。
4. 复用 v6-phys-strict 仿真链路（200 kHz、1 MS/s、20-bit、L_m 0.2–0.3 m）。
5. 数据集前缀 `tv3-*`，schema_version `tunnel-ventilation-1`。
6. `composition_scheme = "tunnel_ventilation"`，下游加载器以 manifest 为准。
7. 移除 `V_NDIR_CH4` 通道（场景无 CH₄，该通道仅含噪声无信息）。

## 五、关键文件清单

| 文件 | 改动类型 | 阶段 | 实际结果 |
|------|----------|------|----------|
| `tv3/sim/core/tunnel_ventilation_schema.py` | 新增 | A | ✅ 已完成 |
| `tv3/sim/generation/tunnel_ventilation/__init__.py` | 新增 | A | ✅ 已完成 |
| `tv3/sim/generation/tunnel_ventilation/conditions.py` | 新增 | A | ✅ 已完成 |
| `tv3/sim/generation/tunnel_ventilation/acoustic_physics.py` | 新增 | B | ✅ 已完成 |
| `tv3/sim/generation/tunnel_ventilation/slow.py` | 新增 | D | ✅ 已完成 |
| `tv3/sim/generation/tunnel_ventilation/_parallel.py` | 新增 | D | ✅ 已完成 |
| `tv3/sim/generation/tunnel_ventilation/benchmark.py` | 新增 | D | ✅ 已完成 |
| `tv3/pipeline/generate_tunnel_ventilation_benchmark.py` | 新增 | D | ✅ 已完成 |
| `configs/tv3_baseline.json` | 新增 | E | ✅ 已完成 |
| `configs/tv3_tcn.json` | 新增 | E | ✅ 已完成 |
| `configs/tv3_lstm.json` | 新增 | E | ✅ 已完成 |
| `configs/tv3_patchtst.json` | 新增 | E | ✅ 已完成 |
| `configs/tv3_ridge.json` | 新增 | E | ✅ 已完成 |
| `configs/tv3_tcn_multimodal.json` | 新增/修正 | E | ✅ 已完成（`raw3` 直接三输出） |
| `scripts/run_tv3_baseline.py` | 新增/修正 | E | ✅ 已完成（seeds=42/123/456；非零退出码不再按 metrics.json 误判成功） |
| `tests/test_tunnel_ventilation_schema.py` | 新增 | A | ✅ 已完成（18 tests） |
| `tests/test_tunnel_ventilation_physics.py` | 新增 | B | ✅ 已完成（25 tests） |
| `tests/test_tunnel_ventilation_benchmark.py` | 新增 | D | ✅ 已完成（18 tests） |
| `tests/test_tunnel_ventilation_dl_training.py` | 新增 | E | ✅ 已完成（13 tests） |
| `tv3/dl/data/dataset.py` | 兼容检查 | E | ✅ 无需修改（manifest 驱动自动适配） |
| `tv3/dl/models/cnn1d_tcn_fusion.py` | 修改 | E | ✅ 已完成（新增 `raw3` 输出模式） |
| `tv3/dl/training/losses.py` | 修改 | E | ✅ 已完成（tv3 拒绝闭包 loss + gas-head 校验） |
| `tv3/ml/training.py` | 修改 | E | ✅ 已完成（按 `composition_scheme` 选择 `o2_bins/co2_bins`，tv3 拒绝 target_transform） |
| `tv3/dl/training/trainer.py` | 修改 | E | ✅ 已完成（tv3 scheme + bin components + sum_abs_error） |
| `tv3/sim/generation/waveforms.py` | 修改 | E | ✅ 已完成（`_digitize_waveform` 支持 `per_timestep_scale`；WaveformSpec/FiberMicSpec 添加字段） |
| `tv3/sim/packaging/arrays.py` | 修改 | E | ✅ 已完成（`write_arrays` 支持可选 fiber_mic） |
| `tv3/sim/validation/integrity.py` | 修改 | E | ✅ 已完成（`_validate_array_shapes` 支持可选 fiber_mic） |
| `docs/server_training_guide.md` | 新增 | E | ✅ 已完成（Linux + RTX 5880 48GB 服务器训练操作手册） |

## 六、验证流程

```powershell
# 1. tv3-smoke 链路验证
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark `
    --output-root data --dataset tv3-smoke --sequences 32 --seed 20260704 `
    --timesteps 32 --dt-s 0.5 --optical-absorption-backend empirical_v1 --workers 1

# 2. 掘进通风单元测试
python -m pytest tests/test_tunnel_ventilation_*.py -v

# 3. 全量回归（确认 hg/sg 无回归）
python -m pytest

# 4. DL 单 seed 训练验证（需 tv3-formal 或临时改用 tv3-smoke 路径）
python -m tv3.dl.cli --config configs/tv3_baseline.json
```

实际结果（2026-07-04）：
- ✅ tv3-smoke 生成成功，validation pass（32 序列，split: train=22/val=4/test=3/extrapolation=3）
- ✅ 掘进通风测试全部通过（74 tests = schema 18 + physics 25 + benchmark 18 + dl_training 13）
- ✅ 全量回归通过：548 passed，0 failed（原主线 462 + tv3 新增 74 + 其他 12），hg/sg 无回归
- ✅ DL CLI 端到端验证通过（tv3-smoke 数据，CNN1D 7 通道 / out_dim=3 / o2_bins+co2_bins 分箱 / sum_abs_error 可计算）

## 七、风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| O₂ 物理可辨识性不足 | 高 | 高 | 阶段 Ⅱ-2 通道消融评估；后备方案：引入顺磁 O₂ 传感器 |
| O₂/N₂ 热导率差异过小（~2%） | 高 | 高 | 依赖声学通道补充（声速差 ~6.4%） |
| N₂ 动态范围小（73.8–82%） | 中 | 中 | weighted_component_mse 加权 + 数据归一化 |
| ~~V_NDIR_CH4 通道无信息（无 CH₄）~~ | — | — | 已移除，不再构成风险 |
| 过度改造主线代码 | 低 | 高 | 分支隔离，tunnel_ventilation 子包独立 |
| 仿真精度限制实际迁移性 | 中 | 中 | 后续引入 HITRAN 后端提高保真度 |

## 八、工作量估计

| 阶段 | 估计 | 依赖 |
|------|------|------|
| 阶段 1：Schema + 采样 | 1–2 天 | 物性常数已填充（[physics_references.md](physics_references.md)） |
| 阶段 2：声学 / 热导 / 光学 | 2–3 天 | 阶段 1 |
| 阶段 3：慢通道 + benchmark + CLI | 1–2 天 | 阶段 2 |
| 阶段 4：DL 适配 | 1–2 天 | 阶段 3（tv3-smoke 可用） |
| **总计** | **5–9 天** | 不含文献检索和参数验证 |
