# HITRAN/PNNL 光谱积分实施方案

本文记录 NDIR CH4/CO2 吸收量从经验合成系数升级为光谱积分系数的依据和实现路线。

## 当前状态

当前 `src/sim/generation/acoustic_physics.py` 中的 NDIR 吸收量仍是合成 benchmark 的经验模型：

```text
CH4_abs = 0.008 * x_CH4 + 0.0008 * H_RH + 0.015 * P_MPa + 0.0002 * (T_C - 25)
CO2_abs = 0.045 * x_CO2 + 0.0006 * H_RH + 0.012 * P_MPa + 0.00015 * (T_C - 25)
```

`src/sim/generation/optical_crosstalk.py` 再把原始吸收转换为 NDIR 通道观测吸收：

```text
CH4_observed = CH4_abs + 0.035 * CO2_abs
CO2_observed = CO2_abs + 0.012 * CH4_abs
```

这些系数的作用是生成可解释、可回归测试的合成数据。它们不是仪器标定值，也不是 HITRAN/PNNL 谱线积分结果。

## 资料依据

- HITRAN HAPI：官方 Python API，可下载 HITRAN line-by-line 数据并计算 absorption coefficient、cross-section、transmittance 等谱量。资料：<https://hitran.org/hapi/>
- HITRAN units and definitions：定义 absorption coefficient、column number density、optical depth 等量。资料：<https://hitran.org/docs/definitions-and-units/>
- PNNL quantitative IR database：PNNL 气相红外定量数据库，约 0.1 cm-1 分辨率，N2 broadened to one atmosphere，面向场景化气体检测/遥感。资料：<https://www.pnnl.gov/publications/gas-phase-database-quantitative-infrared-spectroscopy>
- PNNL/NWIR 数据库说明：近 500 种纯物质气相红外定量谱，0.1 cm-1 分辨率，1 atm N2 展宽。资料：<https://www.pnnl.gov/publications/northwest-infrared-nwir-gas-phase-spectral-database-industrial-and-environmental>
- NIST Quantitative Infrared Database：提供 absorption coefficient spectra，并建议比较时使用 integrated spectral features；吸收谱可按浓度和光程缩放得到吸光度。资料：<https://webbook.nist.gov/chemistry/quant-ir/>
- NDIR/Beer-Lambert 原理：吸收量与浓度、吸收系数、光程相关，透射强度遵循指数衰减。资料：<https://www.horiba.com/gbr/process-and-environmental/measuring-principles/ndir/home/>、<https://www.analog.com/en/resources/analog-dialogue/articles/complete-gas-sensor-circuit-using-nondispersive-infrared.html>

## 推荐实现路线

### 路线 A：HITRAN line-by-line 计算

适用于 CH4、CO2、H2O 这类 HITRAN 覆盖较好的小分子。推荐作为正式实现主线。

1. 选择 NDIR 通道波段和滤光片响应函数。
   - CO2 常用中心波长约 4.26 um，对应约 2347 cm-1。
   - CH4 常用强吸收带约 3.3 um，对应约 3030 cm-1。
   - 具体窗口必须由目标传感器滤光片带宽决定，不能只用中心波长。
2. 用 HAPI 下载目标分子谱线：
   - CH4
   - CO2
   - H2O
3. 在给定 `T_C`、`P_MPa`、组分浓度和光程 `L_m` 下计算吸收系数 `k_i(nu, T, P)`。
4. 将浓度和光程转成 column amount 或等价 optical depth：

```text
tau_i(nu) = k_i(nu, T, P) * column_i
```

5. 合成通道透射率：

```text
T_channel = integral R_channel(nu) * exp(-sum_i tau_i(nu)) dnu
            / integral R_channel(nu) dnu
```

6. 通道观测吸收定义为：

```text
A_channel = -ln(T_channel)
```

7. 交叉敏感度矩阵可由单组分扰动积分得到：

```text
M[channel, gas] = d A_channel / d concentration_gas
```

或者在 benchmark 生成中直接使用 `A_channel`，不再显式保留经验交叉矩阵。

### 路线 B：PNNL/NIST absorption coefficient 谱库积分

适用于需要实验参考谱、且条件接近 1 atm N2 展宽的场景。

1. 获取目标气体的定量吸收系数谱或 cross-section 谱。
2. 插值到统一 wavenumber grid。
3. 将各气体谱按浓度和光程缩放为吸光度。
4. 对滤光片响应窗口做积分：

```text
A_channel = integral R_channel(nu) * sum_i A_i(nu) dnu
            / integral R_channel(nu) dnu
```

5. 对比路线 A 的 HITRAN line-by-line 结果，决定是否用 PNNL/NIST 作为校准参考或回归基线。

## 代码结构

```text
src/sim/generation/spectral/
  __init__.py
  defaults.py             # 默认气体、滤光片和 HITRAN 网格配置
  filters.py              # NDIR 通道滤光片响应函数
  hitran_backend.py       # HAPI/HITRAN line-by-line 计算适配层
  tabulated_backend.py    # PNNL/NIST 谱表读取与积分
  integration.py          # 光谱积分公共接口
  cache.py                # 光谱网格与吸收系数缓存
```

已落地的是本地光谱积分核心、表格谱 backend、HAPI 适配层、缓存、HITRAN 单位换算、默认滤光片/网格配置、HITRAN 预计算 CLI 和 empirical/HITRAN 对照 CLI。`hitran_backend.py` 支持注入 HAPI-like 对象做离线回归测试；本地已用真实 `hitran-api 1.3.0.0` 下载 CH4/CO2/H2O 在 `2960-3100 cm-1` 与 `2280-2410 cm-1` 两个窗口的谱线，并生成 `.data/.header/.npz` 缓存。当前入口：

```text
configs/
  data/spectral-defaults.json
data/
  hitran_cache*/           # 运行 precompute 后生成，本地缓存不进 git
src/pipeline/
  precompute_hitran_spectra.py
  compare_optical_backends.py
```

`spectral-defaults.json` 当前已切到行业参考占位（`filter_source.type=industry_reference_only`，CH4 来源 InfraTec LIM-262 NBP 3.3 μm/160 nm，对应 ~147 cm⁻¹ FWHM；CO2 来源 InfraTec 标准 CO2 NBP 4.26 μm/170 nm，对应 ~93 cm⁻¹ FWHM），不是目标仪器 TraceGas-HC-NDIR（深圳市痕量气体传感科技有限公司）的最终滤光片标定，正式 benchmark 前必须替换为厂商 datasheet。

建议公共接口：

```python
def compute_ndir_absorbance(
    *,
    x_ch4: float,
    x_co2: float,
    x_h2o: float,
    t_c: float,
    p_mpa: float,
    l_m: float,
    channel: str,
) -> dict[str, float]:
    ...
```

返回字段建议包含：

- `absorbance_observed`
- `absorbance_by_gas`
- `transmittance_channel`
- `filter_center_cm1`
- `filter_fwhm_cm1`
- `backend`
- `source_version`

## 验证标准

- 单气体浓度增加时，目标通道吸收单调增加。
- 非目标气体对目标通道的交叉响应非负，且小于主响应。
- HITRAN HAPI 输出的 cm²/molecule 系数必须先按理想气体数密度换算为 per-percent-per-meter 系数，再进入 `TabulatedSpectrum`。
- HAPI 原始表名必须绑定气体和波数窗口，避免不同 NDIR 通道复用错误谱线范围。
- `main_sensor_features` 的固定种子回归测试必须更新并记录谱源版本。
- 文档和 manifest 必须记录 `optical_absorption_backend`，例如 `empirical_v1`、`hitran_hapi_v1` 或 `pnnl_tabulated_v1`。

## 当前结论

短期内保留经验模型作为 `empirical_v1`，但文档和论文表述必须说明其为合成经验系数。当前已实现 `tabulated_spectrum_v1` 本地积分原型、`hitran_hapi_v1` 适配层、HITRAN 单位换算、真实 HAPI 谱线下载、预计算入口和 empirical/HITRAN 对照入口；默认滤光片已从 smoke 占位（CH4 30 / CO2 24 cm⁻¹ FWHM）切到行业参考占位（CH4 147 / CO2 93 cm⁻¹ FWHM，来源见 `filter_source`），仍需获取 TraceGas-HC-NDIR 实际 datasheet 才能进入正式标定，再用 PNNL/NIST 定量 IR 数据进行 sanity check 或标定对照。
