# 光学链路物理仿真原理

本文记录 v4 正式实验中 NDIR 光学通道（`V_NDIR_CH4`、`V_NDIR_CO2`）从配气方案到电压信号的完整物理仿真链路、当前实际使用的经验模型、已实现但未启用的光谱积分原型，以及两条路径之间的契约关系。

更新日期：2026-05-25。最新状态以 `manifest.json` 的 `optical_absorption_backend` 字段为准。

## 1. 在系统里的位置

光学链路属于慢变量（slow channels）模态，与超声波形、光纤麦克风波形并列。每个序列在每个时步输出：

| 字段                   | 含义                           |
| -------------------- | ---------------------------- |
| `V_NDIR_CH4`         | CH4 通道 NDIR 探测器电压（含基线、漂移、噪声） |
| `V_NDIR_CO2`         | CO2 通道 NDIR 探测器电压            |
| `ndir_ch4_saturated` | CH4 通道吸光度是否超过饱和阈值            |
| `ndir_co2_saturated` | CO2 通道吸光度是否超过饱和阈值            |

注意：`V_TCS` 是热导式氢传感器，不走光学链路。代码上和 NDIR 共用同一个函数 `main_sensor_features`，只是因为复用 baseline + drift + noise 的处理风格。

仿真方向与真实测量相反：真实仪器是「红外光 → 经过混合气 → 滤光片 → 电压 → 反推吸光度」，仿真是从配气方案直接给出真值吸光度，再退化成电压。

## 2. 双路径架构

当前代码同时存在两条实现路径，由 `optical_absorption_backend` 字段标记当前 benchmark 用的是哪一条：

| backend                 | 代码位置                                                                                  | 状态             | 用途                       |
| ----------------------- | ------------------------------------------------------------------------------------- | -------------- | ------------------------ |
| `empirical_v1`          | `src/sim/generation/acoustic_physics.py`<br>`src/sim/generation/optical_crosstalk.py` | benchmark 实际使用 | 可解释、可回归测试的合成吸收模型         |
| `tabulated_spectrum_v1` | `src/sim/generation/spectral/tabulated_backend.py`                                    | 已实现，未启用        | 预制气体吸收谱表的滤光片积分           |
| `hitran_hapi_v1`        | `src/sim/generation/spectral/hitran_backend.py`                                       | 适配层就位，真实数据未集成  | HITRAN line-by-line 谱线积分 |

`tabulated_spectrum_v1` 和 `hitran_hapi_v1` 共用同一套积分公式，区别只在「谱来自哪里」。

## 3. empirical_v1：当前 benchmark 跑的链路

入口在 `acoustic_physics.py:145` 的 `main_sensor_features`。给定一个稳态条件，按以下顺序生成两路 NDIR 信号。

### 3.1 单气体真值吸光度

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

### 3.2 通道交叉敏感度

`apply_optical_crosstalk` 在 `optical_crosstalk.py:15`，把真值吸光度转换成通道观测吸光度：

```text
A_CH4_observed = A_CH4_true + 0.035 · A_CO2_true
A_CO2_observed = A_CO2_true + 0.012 · A_CH4_true
```

- 参数集中在 `OpticalCrosstalkSpec`：`ch4_channel_co2_response=0.035`，`co2_channel_ch4_response=0.012`。
- 物理含义：CH4 通道的滤光带宽内有一部分 CO2 的吸收漏入，反之亦然。
- 当切换到光谱积分 backend 时，这一层会被取消，因为通道间串扰直接由滤光片响应 + 各气体真实谱形给出，不再需要单独的交叉矩阵。

### 3.3 基线 + 漂移 + 噪声 → 探测器电压

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

### 3.4 时序：从静态条件展开为序列

`src/sim/generation/slow.py:50` 把单条件快照扩展成 `(timesteps,)` 序列：

1. 计算两份 `main_sensor_features`：
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

### 3.5 manifest 标记

`src/sim/generation/benchmark.py:35` 硬编码：

```python
OPTICAL_ABSORPTION_BACKEND = "empirical_v1"
```

由 `build_manifest`（`packaging/manifest.py:17`）和 `metadata/waveform_spec.json`（`benchmark.py:127`）同步写出。这是「这一份 benchmark 用了哪个吸收模型」的唯一可信来源，后续切换 backend 时必须同步修改。

## 4. spectral 子模块：物理支撑路径

`src/sim/generation/spectral/` 下的实现遵循 NDIR + Beer–Lambert + 滤光片积分的标准流程。当前不集成进 benchmark，只通过测试覆盖。

### 4.1 滤光片响应

`filters.py:8` 定义 `NDIRFilter(channel, center_cm1, fwhm_cm1)`。`gaussian_filter` 根据 FWHM 推 σ：

```text
σ = FWHM / (2 · sqrt(2 · ln 2)) ≈ FWHM / 2.3548
R(ν) = exp(−½ · ((ν − ν0) / σ)²)
```

只使用归一形状，量纲在通道归一时约掉。`np.trapezoid(response, ν)` 必须 > 0，否则报错。

文档级参考中心波长（实际取值由传感器滤光片决定）：

| 通道  | 中心波长     | 波数         |
| --- | -------- | ---------- |
| CO2 | ~4.26 μm | ~2347 cm⁻¹ |
| CH4 | ~3.3 μm  | ~3030 cm⁻¹ |

### 4.2 单气体光学深度

`tabulated_backend.py:36`：

```text
τ_i(ν) = k_i(ν) · concentration_i_pct · L_m
```

`TabulatedSpectrum.absorption_coeff_per_percent_m` 的单位约定是「每 1% 体积浓度、每米光程」的吸收系数，方便直接乘以 percent 浓度。所有气体必须共享同一套 wavenumber grid，函数内部用 `np.allclose` 校验。

### 4.3 通道透射率与吸光度

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

### 4.4 HITRAN 适配层

`hitran_backend.py:35` 的 `compute_hitran_ndir_absorbance` 内部委托给 `compute_tabulated_ndir_absorbance`，只负责「把 HAPI 算出来的吸收系数包成 `TabulatedSpectrum`」。流程：

1. 构造 `SpectralCacheKey`（backend + gas + source_version + 波数范围 + 步长 + T + P）。
2. 查 `cache.py:27` 缓存：命中就直接读 npz，缓存元数据必须与请求 key 完全一致，否则报错。
3. 缓存未命中时调用 HAPI：
   - `hapi.db_begin(cache_root)` 设置本地数据库目录；
   - `hapi.fetch(table, molecule_id, isotopologue_id, ν_min, ν_max)` 下载谱线；
   - `hapi.absorptionCoefficient_Voigt(...)` 用 Voigt 线型生成 `(ν, k(ν))`，传 `HITRAN_units=True` 锁定 HITRAN 的 cm²/molecule 系数。
4. 写缓存（`cache.py:38`），缓存保存 HAPI 原始 cm²/molecule 系数，文件名嵌入全部 key 字段，便于人工检查。
5. `convert_hitran_coeff_to_per_percent_m()` 使用理想气体数密度，把 cm²/molecule 换算为 `TabulatedSpectrum.absorption_coeff_per_percent_m` 需要的「每 1% 体积浓度、每米光程」光学深度系数。

为了让测试不依赖外网，`compute_hitran_ndir_absorbance` 接受 `hapi_module` 注入参数。`tests/test_spectral_hitran_backend.py` 用一个 fake HAPI 验证了缓存 miss/hit 行为、缓存 roundtrip 和 HITRAN 单位换算。

### 4.5 公共输出契约

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

这是后续替换 empirical 链路时的对接面。要切换到光谱积分时，把 `main_sensor_features` 里 3.1 + 3.2 两层换成调用 `compute_*_ndir_absorbance` 两次，分别取 `absorbance_observed` 作为 `absorption_ch4_observed` / `absorption_co2_observed`，下游 3.3 / 3.4 / 3.5 不需要改动。

## 5. 测试覆盖

| 测试文件                                        | 覆盖内容                                                           |
| ------------------------------------------- | -------------------------------------------------------------- |
| `tests/test_acoustic_physics_regression.py` | `main_sensor_features` 固定种子回归基线（含 empirical 吸收 + 交叉敏感度 + 电压退化） |
| `tests/test_optical_crosstalk.py`           | `apply_optical_crosstalk` 的对称性和单调性                             |
| `tests/test_spectral_integration.py`        | 常数 optical depth、Gaussian 滤光片响应、表格谱交叉响应                        |
| `tests/test_spectral_hitran_backend.py`     | fake HAPI 调用、缓存命中、缓存 roundtrip                                 |

执行：

```bash
python -m pytest tests
```

当前 89 个测试全部通过（2026-05-25 状态）。

## 6. 当前缺口

物理支撑路径已实现的部分：滤光片高斯响应、通道积分公式、表格谱 backend、HITRAN 适配层、缓存读写、HITRAN cm²/molecule 到 per-percent-per-meter 的单位换算。

仍缺的部分（按集成难度排序）：

1. **滤光片配置不存在**。`NDIRFilter` 的 `center_cm1`、`fwhm_cm1` 还没有项目级正式取值。需要根据目标传感器滤光片实际带宽决定，且必须有出处。
2. **真实谱表 / HITRAN 数据未集成**。`TabulatedSpectrum` 目前只在测试里手工构造；HITRAN 路径需要安装 HAPI、配置本地数据库目录、决定 T/P 网格策略（多少格点、插值还是按条件下载）。
3. **真实单位标定仍需外部对照**。代码已经完成 HITRAN cm²/molecule 到 `absorption_coeff_per_percent_m` 的理想气体换算，但尚未用真实 HAPI 输出和仪器/PNNL/NIST 数据做数值 sanity check。
4. **PNNL/NIST 对照尚未做**。`docs/SPECTRAL_INTEGRATION_PLAN.md` 提到把 PNNL/NIST 作为 sanity check 或标定参考，目前没有拉数据，也没有对照脚本。

## 7. 下一步选项

按收益和成本排序：

| 选项  | 内容                                                                                                                   | 成本  |
| --- | -------------------------------------------------------------------------------------------------------------------- | --- |
| A   | 把 `OPTICAL_ABSORPTION_BACKEND` 改成可选项，让 benchmark 可以跑 `tabulated_spectrum_v1`，配一个本地合成小型谱表，给 empirical_v1 保留独立的回归测试基线。 | 中   |
| B   | 装 HAPI、写谱线下载脚本、做一次 HITRAN vs empirical 的小规模 benchmark 对照，并用 PNNL/NIST 做 sanity check。                                | 高   |
| C   | 暂停物理支撑，回到 Phase 3 实现 TCNRegressor，物理路径作为后续可插拔升级。                                                                     | 低   |

## 8. 相关文档

- `docs/SPECTRAL_INTEGRATION_PLAN.md`：HITRAN/PNNL/NIST 资料依据 + 实现路线 + 验证标准。
- `docs/ARCHITECTURE.md`：sim 模块整体架构。
- `docs/IMPLEMENTATION_PLAN.md`：阶段任务清单和优先级。
- `README.md`：项目入口和 backend 字段说明。
