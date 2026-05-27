# 光学链路物理仿真原理

本文记录 v4 正式实验中 NDIR 光学通道（`V_NDIR_CH4`、`V_NDIR_CO2`）从配气方案到电压信号的完整物理仿真链路、默认 `hitran_hapi_v1` 光谱积分路径、显式兼容的 `empirical_v1` 经验路径，以及两条路径之间的契约关系。

更新日期：2026-05-26。最新状态以 `manifest.json` 的 `optical_absorption_backend` 字段为准。

## 1. 在系统里的位置

光学链路属于慢变量（slow channels）模态，与超声波形、光纤麦克风波形并列。每个序列在每个时步输出：

| 字段                   | 含义                           |
| -------------------- | ---------------------------- |
| `V_NDIR_CH4`         | CH4 通道 NDIR 探测器电压（含基线、漂移、噪声） |
| `V_NDIR_CO2`         | CO2 通道 NDIR 探测器电压            |
| `ndir_ch4_saturated` | CH4 通道吸光度是否超过饱和阈值            |
| `ndir_co2_saturated` | CO2 通道吸光度是否超过饱和阈值            |

注意：`V_TCS` 是热导式氢传感器，不走光学链路。`empirical_v1` 兼容路径仍在 `main_sensor_features` 中同时生成 NDIR 与 TCS；默认 `hitran_hapi_v1` benchmark 主线会改走 `thermal_conductivity_sensor_feature` 只计算 TCS，NDIR 由 HITRAN 光谱积分单独给出，避免顺带跑 empirical NDIR。

仿真方向与真实测量相反：真实仪器是「红外光 → 经过混合气 → 滤光片 → 电压 → 反推吸光度」，仿真是从配气方案直接给出真值吸光度，再退化成电压。

## 2. 双路径架构

当前代码同时存在两条实现路径，由 `optical_absorption_backend` 字段标记当前 benchmark 用的是哪一条：

| backend                 | 代码位置                                                                                  | 状态             | 用途                       |
| ----------------------- | ------------------------------------------------------------------------------------- | -------------- | ------------------------ |
| `hitran_hapi_v1`        | `src/sim/generation/optical_backend.py`<br>`src/sim/generation/spectral/hitran_backend.py` | benchmark 默认使用 | HITRAN line-by-line 谱线积分 |
| `empirical_v1`          | `src/sim/generation/acoustic_physics.py`<br>`src/sim/generation/optical_crosstalk.py` | 显式兼容路径 | 可解释、可回归测试的合成吸收模型 |
| `tabulated_spectrum_v1` | `src/sim/generation/spectral/tabulated_backend.py` | 已实现，作为外部 sanity check 支撑 | 预制气体吸收谱表的滤光片积分 |

`tabulated_spectrum_v1` 和 `hitran_hapi_v1` 共用同一套积分公式，区别只在「谱来自哪里」。

## 3. hitran_hapi_v1：默认 benchmark 链路

默认 `pipeline.generate_benchmark` 使用 `hitran_hapi_v1`。生成前会按同一批 `conditions` 收集 CH4/CO2/H2O 在 CH4 与 CO2 两个 NDIR 通道窗口、每条 condition 的温压状态下需要的 HITRAN cache key；只要有缺失，就在写出 dataset 前失败，并提示先运行 `pipeline.precompute_hitran_benchmark_cache`。

关键契约：

- `temperature_k = round(T_C + 273.15, 3)`。
- `pressure_atm = round(P_MPa / 0.101325, 6)`。
- `H2O` 不进入 label 组分和 100% 组分校验，只由 `T/P/RH` 换算为光学吸收中的水汽 mole percent。
- 每个 timestep 使用当前 `blend` 后的组分和当前 `L_m` 计算 NDIR equilibrium，因此 steady/baseline 声程扫描会真实影响 `V_NDIR_CH4` 与 `V_NDIR_CO2`。
- HITRAN 多气体滤光片积分已经表达通道交叉响应，不再叠加 `apply_optical_crosstalk` 的经验矩阵，避免双计数。
- 生成阶段只读 `.npz` cache，不导入 HAPI、不联网、不写谱线缓存。
- 同一 `(channel, HitranGridSpec)` 会准备成 `PreparedTabulatedSpectra` 缓存在生成过程中复用；栅格一致性和滤光片响应只在 cache 载入时处理一次。

推荐流程：

```powershell
python -m pipeline.precompute_hitran_benchmark_cache --cache-root data/hitran_cache --sequences 32 --seed 42
python -m pipeline.generate_benchmark --output-root data --dataset wv4-smoke --sequences 32 --seed 42 --storage npz
```

`manifest.json` 与 `metadata/waveform_spec.json` 会记录：

- `optical_absorption_backend = "hitran_hapi_v1"`
- `hitran_cache_policy = "cache_only_prechecked"`
- `hitran_temperature_pressure_mode = "per_condition"`
- `h2o_policy = "rh_to_mole_pct"`
- `optical_crosstalk_policy = "spectral_multigas_integral"`

## 4. empirical_v1：显式兼容链路

如需旧合成经验路径，可显式传 `--optical-absorption-backend empirical_v1`。入口在 `acoustic_physics.py` 的 `main_sensor_features`。给定一个稳态条件，按以下顺序生成两路 NDIR 信号。

### 4.1 单气体真值吸光度

由两条经验线性公式生成（`acoustic_physics.py:214`、`:218`）：

```text
A_CH4_true = 0.008 · x_CH4 + 0.0008 · H_RH + 0.015 · P_MPa + 0.0002 · (T_C − 25)
A_CO2_true = 0.045 · x_CO2 + 0.0006 · H_RH + 0.012 · P_MPa + 0.00015 · (T_C − 25)
```

设计意图：

- 主项是目标组分浓度，主导信号。
- 副项保留湿度、压力、温度的弱依赖，强制下游模型必须处理交叉影响。
- 系数是合成 benchmark 的可解释值，不是仪器标定，也不是谱线积分结果。

参考阅读：`docs/SPECTRAL_INTEGRATION_PLAN.md` 对这些系数为何不能当作物理标定值的说明。

### 4.2 通道交叉敏感度

`apply_optical_crosstalk` 在 `optical_crosstalk.py:15`，把真值吸光度转换成通道观测吸光度：

```text
A_CH4_observed = A_CH4_true + 0.035 · A_CO2_true
A_CO2_observed = A_CO2_true + 0.012 · A_CH4_true
```

- 参数集中在 `OpticalCrosstalkSpec`：`ch4_channel_co2_response=0.035`，`co2_channel_ch4_response=0.012`。
- 物理含义：CH4 通道的滤光带宽内有一部分 CO2 的吸收漏入，反之亦然。
- 当切换到光谱积分 backend 时，这一层会被取消，因为通道间串扰直接由滤光片响应 + 各气体真实谱形给出，不再需要单独的交叉矩阵。

### 4.3 基线 + 漂移 + 噪声 → 探测器电压

NDIR 探测器电压用 Beer–Lambert 形式的指数衰减（`acoustic_physics.py:175`）：

```text
V_NDIR_CH4 = max(0.1, baseline_CH4_now · exp(−A_CH4_observed) + ε)
V_NDIR_CO2 = max(0.1, baseline_CO2_now · exp(−A_CO2_observed) + ε)
```

其中 baseline 不是常数，分三层叠加：

| 来源     | 公式                                                             |
| ------ | -------------------------------------------------------------- |
| 起始基线   | `optical_baseline_*_init = 2.5`                                |
| 漂移项    | `0.0007 · (H_RH − 50) + 0.004 · (P_MPa − 1) + Gauss(0, 0.004)` |
| 出口高斯抖动 | `Gauss(0, 0.006)`                                              |

漂移项把湿度差和压力差分别耦合进 baseline，让 baseline 在不同条件下不再是常数；出口抖动模拟探测器电压的瞬时白噪声。

饱和判断在 `acoustic_physics.py:198`：阈值是吸光度而不是电压，`optical_saturation_absorbance = 4.0`，对应透射率约 1.8%。

### 4.4 时序：从静态条件展开为序列

`src/sim/generation/slow.py:50` 把单条件快照扩展成 `(timesteps,)` 序列：

1. 计算两份 `main_sensor_features`（仅 `empirical_v1` 兼容路径）：
   - `baseline_main`：零浓度初态。
   - `target_main`：目标配气稳态。
2. 把 timesteps 四等分（`phases.py`，边界为 `q1, q2, q3`），每段含义：
   - 第 1 段：稳定在 baseline。
   - 第 2 段：指数上升到 target，时间常数 `tau_rise_system_s`。
   - 第 3 段：稳定在 target（实际靠步 2 末尾的渐近值近似）。
   - 第 4 段：指数衰减回 baseline，时间常数 `tau_decay_system_s`。
3. 在指数轨迹上再叠加：
   - drift 斜率项：模拟仪器随时间的零点漂移。
   - 随机游走项：模拟 baseline 的低频不平稳。
   - 独立高斯噪声项：模拟瞬时白噪声。

时间常数从配置区间均匀采样（`slow.py:14-23`）：

| 通道           | tau_rise (s) | tau_decay (s) |
| ------------ | ------------ | ------------- |
| `V_NDIR_CH4` | 8.0 – 20.0   | 12.0 – 30.0   |
| `V_NDIR_CO2` | 6.0 – 18.0   | 10.0 – 28.0   |
| `V_TCS`      | 10.0 – 35.0  | 20.0 – 60.0   |

输出落入 `slow[seq, t, channel_index]` 三维数组对应槽位，同时写入 `sequences/slow_sequence_long.csv` 长表。

## 5. spectral 子模块：物理支撑路径

`src/sim/generation/spectral/` 下的实现遵循 NDIR + Beer–Lambert + 滤光片积分的标准流程。benchmark 默认通过 `src/sim/generation/optical_backend.py` 走 cache-only `hitran_hapi_v1`；也可通过 `pipeline.precompute_hitran_spectra` 做通用通道预计算，通过 `pipeline.compare_optical_backends` 与 empirical_v1 做小规模对照，并通过 `pipeline.sanity_check_tabulated_spectra` 把外部定量谱表与 `hitran_hapi_v1` 做同条件 sanity check。本地已用真实 HAPI 环境完成 CH4/CO2/H2O 两个 NDIR 窗口的下载验证。

### 5.1 滤光片响应

`filters.py:8` 定义 `NDIRFilter(channel, center_cm1, fwhm_cm1)`。`gaussian_filter` 根据 FWHM 推 σ：

```text
σ = FWHM / (2 · sqrt(2 · ln 2)) ≈ FWHM / 2.3548
R(ν) = exp(−½ · ((ν − ν0) / σ)²)
```

只使用归一形状，量纲在通道归一时约掉。`np.trapezoid(response, ν)` 必须 > 0，否则报错。

`configs/data/spectral-defaults.json` 是运行时默认值来源，`defaults.py` 读取该 JSON 构造 dataclass 常量。当前配置使用行业参考占位（`filter_source.type=industry_reference_only`，实际取值由目标传感器 TraceGas-HC-NDIR 决定）：

| 通道  | 中心波长     | 中心波数       | FWHM       | 来源占位                                                                                                |
| --- | -------- | ---------- | ---------- | --------------------------------------------------------------------------------------------------- |
| CO2 | ~4.26 μm | 2347 cm⁻¹  | 93 cm⁻¹    | InfraTec 标准 CO2 NBP filter 4.26 μm / 170 nm HPBW（infratec-infrared.com）                              |
| CH4 | ~3.3 μm  | 3030 cm⁻¹  | 147 cm⁻¹   | InfraTec LIM-262 pyroelectric methane detector NBP 3.3 μm / 160 nm FWHM（MDPI Sensors 2012, doi:10.3390/s120912729） |

以上属于 `industry_reference_only` 占位，正式 benchmark 前必须替换为目标传感器 TraceGas-HC-NDIR（深圳市痕量气体传感科技有限公司）实际 datasheet。

### 5.2 单气体光学深度

`tabulated_backend.py`：

```text
τ_i(ν) = k_i(ν) · concentration_i_pct · L_m
```

`TabulatedSpectrum.absorption_coeff_per_percent_m` 的单位约定是「每 1% 体积浓度、每米光程」的吸收系数，方便直接乘以 percent 浓度。所有气体必须共享同一套 wavenumber grid；`prepare_tabulated_spectra` 会用 `np.allclose` 校验并构造滤光片响应，HITRAN benchmark 生成会缓存这个 prepared 对象。

### 5.3 通道透射率与吸光度

`integration.py:8` 是核心公式：

```text
T_channel = ∫ R(ν) · exp(−Σ_i τ_i(ν)) dν  /  ∫ R(ν) dν
A_channel = −ln(T_channel)
```

实现要点：

- 所有气体的 τ 在指数里相加，先求总 τ 再做一次滤光片加权积分。这是正确的多组分吸收形式，避免「单气体吸光度可加但单气体透射率不可加」的常见错误。
- 透射率必须 > 0，防止 `log(0)`。
- 同时返回 `transmittance_channel` 和 `filter_area`，便于排查滤光片配置。
- 每个气体还会单独跑一次 `integrate_channel_absorbance`，得到 `absorbance_by_gas`，方便后续推导交叉敏感度矩阵或单组分诊断。

### 5.4 HITRAN 适配层

`hitran_backend.py:35` 的 `compute_hitran_ndir_absorbance` 内部委托给 `compute_tabulated_ndir_absorbance`，只负责「把 HAPI 算出来的吸收系数包成 `TabulatedSpectrum`」。流程：

1. 构造 `SpectralCacheKey`（backend + gas + source_version + 波数范围 + 步长 + T + P）。
2. 查 `cache.py:27` 缓存：命中就直接读 npz，缓存元数据必须与请求 key 完全一致，否则报错。
3. 缓存未命中时调用 HAPI：
   - `hapi.db_begin(cache_root)` 设置本地数据库目录；
   - `hapi.fetch(table, molecule_id, isotopologue_id, ν_min, ν_max)` 下载谱线；
   - `hapi.absorptionCoefficient_Voigt(...)` 用 Voigt 线型生成 `(ν, k(ν))`，传 `HITRAN_units=True` 锁定 HITRAN 的 cm²/molecule 系数。
4. HAPI 原始表名使用气体名 + 波数窗口，例如 `CH4_2960p0000_3100p0000`，避免 CH4/CO2 通道用不同窗口时复用或覆盖错误谱线范围。
5. 写缓存（`cache.py:38`），缓存保存 HAPI 原始 cm²/molecule 系数，文件名嵌入全部 key 字段，便于人工检查。
6. `convert_hitran_coeff_to_per_percent_m()` 使用理想气体数密度，把 cm²/molecule 换算为 `TabulatedSpectrum.absorption_coeff_per_percent_m` 需要的「每 1% 体积浓度、每米光程」光学深度系数。

为了让测试不依赖外网，`compute_hitran_ndir_absorbance` 接受 `hapi_module` 注入参数。`tests/test_spectral_hitran_backend.py` 用一个 fake HAPI 验证了缓存 miss/hit 行为、缓存 roundtrip 和 HITRAN 单位换算。

### 5.5 公共输出契约

`compute_tabulated_ndir_absorbance` 和 `compute_hitran_ndir_absorbance` 返回同一组字段：

| 字段                      | 含义                                         |
| ----------------------- | ------------------------------------------ |
| `absorbance_observed`   | 通道观测吸光度                                    |
| `absorbance_by_gas`     | 每个气体单独的通道吸光度                               |
| `transmittance_channel` | 通道透射率                                      |
| `filter_center_cm1`     | 滤光片中心波数                                    |
| `filter_fwhm_cm1`       | 滤光片 FWHM                                   |
| `backend`               | `tabulated_spectrum_v1` 或 `hitran_hapi_v1` |
| `source_version`        | 每气体的谱源版本字典                                 |

这是 empirical 对照、HITRAN 主线和外部谱表 sanity check 的共同对接面。默认 benchmark 已在 `optical_backend.py` 中调用 HITRAN cache-only 积分，并把 `absorbance_observed` 映射为 `V_NDIR_CH4` / `V_NDIR_CO2` 的 equilibrium。

## 5. 测试覆盖

| 测试文件                                        | 覆盖内容                                                           |
| ------------------------------------------- | -------------------------------------------------------------- |
| `tests/test_acoustic_physics_regression.py` | `main_sensor_features` 固定种子回归基线（含 empirical 吸收 + 交叉敏感度 + 电压退化） |
| `tests/test_optical_crosstalk.py`           | `apply_optical_crosstalk` 的对称性和单调性                             |
| `tests/test_spectral_integration.py`        | 常数 optical depth、Gaussian 滤光片响应、表格谱交叉响应、prepared 谱表复用       |
| `tests/test_spectral_hitran_backend.py`     | fake HAPI 调用、缓存命中、缓存 roundtrip、单位换算                         |
| `tests/test_spectral_pipeline.py`           | 默认谱学配置、HITRAN 预计算 CLI、empirical/HITRAN 对照入口                  |
| `tests/test_quantitative_table.py`          | 外部定量谱表 CSV 读取、单位转换、grid 重采样和拒绝外推                         |
| `tests/test_spectral_sanity_check.py`       | 外部表格谱与 `hitran_hapi_v1` sanity check CLI 核心契约                    |

执行：

```bash
python -m pytest tests
```

当前新机器 Python 3.12.10 虚拟环境下，`python -m pytest tests` 为 134 passed（2026-05-27 状态）。

## 6. 当前缺口

物理支撑路径已实现的部分：滤光片高斯响应、通道积分公式、表格谱 backend、外部定量谱表 CSV 读取与单位转换、HITRAN 适配层、缓存读写、HITRAN cm²/molecule 到 per-percent-per-meter 的单位换算、默认滤光片/网格配置、HITRAN 预计算 CLI、empirical/HITRAN 对照 CLI、外部表格谱 sanity check CLI，以及本地真实 HAPI 下载验证。

仍缺的部分（按集成难度排序）：

1. **真实滤光片规格未替换**。当前 `spectral-defaults.json` 已从 smoke 占位（CH4 30 cm⁻¹ / CO2 24 cm⁻¹ FWHM）切到行业参考占位（CH4 147 cm⁻¹ / CO2 93 cm⁻¹，来源见 `filter_source` 字段），但仍非目标传感器 TraceGas-HC-NDIR 的实际 datasheet。默认 `hitran_grids` 已扩大到覆盖当前滤光片 `center ± FWHM`；grid 变化后需要重下 HITRAN cache，未来拿到真实 datasheet 后仍需再次复核窗口。
2. **真实单位标定仍需外部对照**。代码已经完成 HITRAN cm²/molecule 到 `absorption_coeff_per_percent_m` 的理想气体换算，并已用真实 HAPI 输出跑通缓存生成；通用 CSV sanity check 入口已可用，但尚未接入真实仪器/PNNL/NIST 数据文件。
3. **PNNL/NIST 原始格式适配未做**。当前支持显式列名和显式单位的通用 CSV，真实数据库导出格式若不同，需要新增薄适配器转换到该 CSV 契约。
4. **真实谱表对照仍待补实测数据**。benchmark 默认已接入 `hitran_hapi_v1`，但真实 PNNL/NIST 或仪器数据尚未导入，当前只完成通用 CSV sanity check 路径。

## 7. 下一步选项

按收益和成本排序：

| 选项  | 内容                                                                                                                   | 成本  |
| --- | -------------------------------------------------------------------------------------------------------------------- | --- |
| A   | 获取目标传感器 TraceGas-HC-NDIR 实际 datasheet 替换当前行业参考占位；按实际 FWHM 复核 `hitran_grids` 并重下 HITRAN cache。 | 中   |
| B   | 获取真实 PNNL/NIST 或仪器定量谱表，转换为通用 CSV 契约并运行 sanity check；若原始格式复杂，再补专用适配器。 | 中   |
| C   | 在获得真实 datasheet 后复核滤光片窗口与 HITRAN grid，重新预计算 benchmark cache，并重新运行外部 sanity check。 | 中 |

## 8. 相关文档

- `docs/SPECTRAL_INTEGRATION_PLAN.md`：HITRAN/PNNL/NIST 资料依据 + 实现路线 + 验证标准。
- `docs/ARCHITECTURE.md`：sim 模块整体架构。
- `docs/IMPLEMENTATION_PLAN.md`：阶段任务清单和优先级。
- `README.md`：项目入口和 backend 字段说明。
