# tv3 声学仿真链路保真度审查

> 审查对象：`tv3/sim/generation/tunnel_ventilation/acoustic_physics.py` + `tv3/sim/generation/waveforms.py` 的声速 / 衰减 / TOF / 波形链路；2026-07-20 补充审查同文件的 TCS / 热导链路（`_hidden_lambda_mix` / `_tcs_voltage`，见第八节）。
> 审查日期：2026-07-20（声学）；CoolProp 8.0.0 复核同日完成；热学链路补充审查同日完成。
> 边界声明：本报告评估**仿真链路与真实物理的对齐程度**，并给出外部数据库/手段建议。**不改写** B1/B7 正式结论、不改写现场掘进通风精度声明、不改写 `tv3-formal-6000` 指标。属评估综述性质，不进记忆库正式结论表。
> 验证状态约定：`[已验证]` 用项目常数实算或代码/CSV/CoolProp 直接确认；`[工程判断]` 基于常识与文献量级估计，未实测；`[待验证]` 需外部工具或实验确认。

---

## 一、结论摘要

1. **主导物理可信**：理想气声速公式在近常压 CO₂/O₂/N₂ 下与 CoolProp `HEOS` 偏差约 **0.02%**（T=35°C、P=0.1 MPa），量程高压端 P=0.709 MPa 约 **0.25%**；空气 346 m/s 交叉验证通过（`test_sound_speed_air_mixture_close_to_346`）。`[已验证：空气点 + CoolProp]`
2. **三个影响结论解读的保真度缺口**：
   - 湿度只进衰减、不进声速/TOF（最关键，见第四节；CoolProp 已把湿度 Δc 升为数据库级证据）；
   - 波形链路偏乐观（单脉冲、无多径/混响/衍射）；
   - 衰减与换能器均为自标注 proxy，未绑定验证过的标准/数据库。
3. **COMSOL P0 对照仅验证公式转写**：`p0_sound_speed_acceptance.csv` 全部 `rel_err=0.0`，因为 COMSOL 求的是同一条 `sqrt(γ·R·T/M)` 解析式，不构成对波形/多径/损耗的独立物理校验。`[已验证：CSV]`
4. **外部工具**：CoolProp 四元声速路径已打通（见 §5.1）；对 tv3 近常压，**把 H₂O 纳入声速比换真实气体 EOS 更值钱**。ISO 9613-1 / Bass 1995、HITRAN、COMSOL 时域 CWE 仍按原优先级可选。
5. **热学链路同源诊断（补充审查，见第八节）**：热导混合物理（WMS + 逐组分幂律温度修正 + NIST 纯组分 λ）与声速理想气公式同属"主导物理忠实"层；`_tcs_voltage` 是未标定 proxy，量级与声学换能器 proxy 平行。**湿度未进 `_hidden_lambda_mix`**，与声速缺湿度是**同一处干/湿不一致**（衰减含水、声速与热导都不含）。"用热导补偿声学 O₂/N₂" 方向物理正确但信号上限低（Δλ(O₂−N₂)≈2.3%，单次 TCS 的 O₂ SNR≈0.05，比声学 O₂ 墙 0.43 还低约一个量级），只提供边际辨识力，不能突破 O₂ 物理墙；真正的"环境参数补偿声学"杠杆是**湿度进声速 + `H_RH` 校正 TOF**（§六 A′），与热导无关。`[工程估算：项目常数手算，未跑 CoolProp 热导实测]`

---

## 二、被审查的完整链路

```
组分(CO₂/O₂/N₂) → 理想气声速 c = √(γ_mix·R·T/M_mix)          [hidden_sound_speed_v2]
               → TOF = L/c + 固定延迟 82μs + 高斯抖动(σ=3μs)   [simulate_waveform_measurement]
               → 8 周期 Hanning burst ⊛ 二阶谐振换能器 proxy
               → Lagrange 分数延迟定位 → 幅度 ×exp(-α·L) → 加性高斯噪声 → 20-bit ADC
衰减 α = 经典吸收(∝f²·√T/P) + CO₂/N₂/O₂/H₂O V-T 弛豫 proxy     [hidden_attenuation_v2]
         （O₂ 弛豫在 200 kHz 取 0；Bass 1990 fr,O≈24 Hz/atm）
```

模型命名已诚实标注性质：`semi_empirical_multigas_relaxation_proxy_v2`、`tof_observed_transducer_proxy_v1`、`second_order_resonant_bandpass_proxy_v1`——衰减和换能器是工程 proxy，不是标定过的物理。`[已验证：代码常量]`

采样范围（双域）：
- **narrow（默认）**：`x_CO2 ∈ [0.03, 5.00]%`、`x_O2 ∈ [18.00, 21.20]%`、`x_N2` 残差闭包。
- **wide（仅 F 线）**：`x_CO2 ∈ [0.03, 10.00]%`、`x_O2 ∈ [15.00, 25.00]%`、`x_N2` 残差闭包；独立 registry / 数据集，不改写窄域冻结结论。
共用环境：`T ∈ [15,35]°C`、`P ∈ [0.10, 0.709] MPa`、`H_RH ∈ [20,80]%`、`L ∈ [0.20, 0.30] m`。`[已验证：conditions.py]`

---

## 三、分组件对齐评估

| 组件            | 对齐程度      | 具体判断                                                                                                                                                                                | 验证状态             |
| ------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| **声速（理想气）**   | 高（均值），有缺口 | 近常压 vs CoolProp `HEOS` 约 0.02%，够用。缺三项：① **湿度→声速完全缺失**（见第四节，最关键；CoolProp 湿 5 mol% Δc≈+2.83 m/s）；② 真实气体 P 依赖缺失（0.709 MPa 约 +0.25%）；③ CO₂ 弛豫附近声速色散 c₀→c∞ 未建（CO₂≤5% 下小）                 | `[已验证：CoolProp]` |
| **衰减**        | 中（proxy）  | 经典吸收用单一 air 标定的 `K_ref=1.84e-11`，与组分无关；弛豫 `λ_max`（CO₂ 0.12 / N₂ 0.004 / H₂O 0.01）是 proxy 量级，非验证数据库。作为弱通道影响有限，但湿度耦合放在这里而非声速里，干/湿质量不一致                                                | `[已验证：代码]`       |
| **波形 / TOF**  | 偏乐观       | 单条干净脉冲：**无多径、无 Ø215×320 mm 气室壁反射/混响、无衍射、无声束扩散**；换能器带宽是估的（PSC200K datasheet 未给带宽，按 10% 中心频率取 20 kHz）；噪声纯高斯。记忆库自注 “exact simulator template 过于乐观”，D2b 达 0.019 sample 峰值 MAE 即此乐观性直接后果 | `[已验证：代码 + 记忆库]` |
| **结构性**       | 忠实但单薄     | 声学通道里 O₂ 信息 = **单一标量 c**，DSP 只把脉冲位置读回，没有独立物理能额外增/减信息。作为下限忠实；真实系统有更多（或更脏）通道                                                                                                          | `[已验证：链路结构]`     |
| **COMSOL 对照** | 仅转写一致     | `p0_sound_speed_acceptance.csv` 全部 `rel_err=0.0`，COMSOL 算同一条 `sqrt(γRT/M)`。验证参数搬运正确，未做时域 CWE 波形求解，不构成对波形/多径/损耗的独立物理校验                                                               | `[已验证：CSV]`      |
| **TCS / 热导混合** | 高（均值），有缺口 | `_hidden_lambda_mix` 用 WMS 混合规则 + 逐组分幂律温度修正，纯组分 λ（CO₂ 16.6 / O₂ 26.4 / N₂ 25.8 mW/m·K）来自 NIST/CRC，φ_ij 由 Wilke 公式算不查表，与声速理想气同属主导物理忠实。缺口：**H₂O 完全未进 λ_mix**（水汽 λ≈18–19 mW/m·K、本场景≤5 mol%，与声速缺湿度同源不一致）；压力仅进漂移项非热导池物理（常压 λ 与 P 无关，此近似可接受）           | `[已验证：代码 + NIST 常数]` |
| **TCS 传递函数** | 低（proxy）  | `_tcs_voltage = 1.1 + 15.0·(λ−0.026) + 0.004·(T−20) + gauss(0,0.006)`：斜率/offset/baseline 均自标注 proxy，未绑定真实热导池（惠斯通电桥 + 热丝几何/对流散热）传递函数。与声学换能器 proxy 同性质          | `[已验证：代码常量]`   |

---

## 四、关键量化发现（已用项目常数验证）

T=35°C、L=0.2 m。下表「解析」列用 `acoustic_physics.py` 的 `_GAS_M`/`_GAS_CP` 加水汽（M=0.018015, cp=33.58）；「CoolProp」列用 `HEOS` 四元混合（CarbonDioxide&Oxygen&Nitrogen&Water）同点复算。`[已验证：实算 + CoolProp 8.0.0]`

| 扰动                              | Δc（解析）    | Δc（CoolProp）      | ΔTOF @0.2 m（CoolProp） | 相对参照                 |
| ------------------------------- | --------- | ----------------- | --------------------- | -------------------- |
| **O₂ 全量程 18→21.2%**             | −0.80 m/s | −0.80 m/s         | **+1.30 μs**          | 声学通道 O₂ 的全部信号        |
| **触发抖动 σ**                      | —         | —                 | **3.0 μs**            | O₂ 信号的 2.3×          |
| **H₂O 0→5 mol%（当前未进声速）**        | +2.83 m/s | **+2.83 m/s**     | **−4.54 μs**          | O₂ 信号的 3.5×、抖动的 1.5× |
| **P 0.1→0.709 MPa（干气，理想气无此效应）** | 0         | +0.87 m/s (0.25%) | **−1.30 μs**          | 与整段 O₂ TOF 同量级       |

两个含义：

1. **O₂ 物理墙真实且被忠实建模**：整段 O₂ 量程的 TOF 跨度(1.30μs) < 抖动(3μs)，比值 0.43；窄 bin 内被噪声淹没。这解释了 oracle 也只有 O₂ R²≈0.60、窄 0.8% bin 全负（与记忆库 §4.1 / §4.3 一致）。
2. **湿度缺口已升为数据库级证据**：水汽本场景最高 5 mol%（Buck 方程 + `min(…,5.0)` clamp，`gas_state.py:13` 确认）；CoolProp 确认其对 TOF 的影响压过 O₂ 本身。当前仿真里湿度对 TOF 完全隐形，既没暴露这个混淆、也没测试“用测到的 `H_RH` 校正 TOF”。真实系统里这一步绕不开。

> 补充：`H_RH` 是慢通道且真实可测，因此湿度在真实系统中既是混淆项也可被校正；关键在于当前仿真链路两者都没体现。

---

## 五、外部数据库 / 手段（按性价比排序）

### 1. CoolProp（开源，Python）/ NIST REFPROP —— 最高性价比（路径已验证）

- CoolProp 免费；混合物性挂在 **`HEOS`** 后端（理论来源含 GERG-2004/2008 + Lemmon 空气混合物）。**`GERG2008` 不是** CoolProp 8.0.0 的合法 backend 名（会报 `Invalid backend name`）。
- REFPROP 是付费金标准；CoolProp 可包装为 `AbstractState("REFPROP", …)`，但需本机 `REFPRP64.dll` / `COOLPROP_REFPROP_ROOT`。本机复核时 DLL 未安装，REFPROP 标 `[未就绪]`。
- 文档：[Mixtures](https://coolprop.org/fluid_properties/Mixtures.html)、[High-level API](https://coolprop.org/coolprop/HighLevelAPI.html)（`A` / `speed_of_sound`）、[Humid Air](https://coolprop.org/fluid_properties/HumidAir.html)、[REFPROP 接口](https://coolprop.org/coolprop/REFPROP.html)。

#### 1.1 已验证调用路径（CoolProp 8.0.0，2026-07-20）

| 路径    | API                                                                                                                          | 是否适合 tv3 变组分                                     | 状态         |
| ----- | ---------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ---------- |
| 高阶一行式 | `PropsSI("A","T",T,"P",P,"CarbonDioxide[x]&Oxygen[y]&Nitrogen[z]&Water[w]")`                                                 | 是（摩尔分数 0–1）                                      | `[已验证]`    |
| 低阶循环  | `AbstractState("HEOS","CarbonDioxide&Oxygen&Nitrogen&Water")` → `set_mole_fractions` → `update(PT_INPUTS)` → `speed_sound()` | 是（扫网格推荐）                                         | `[已验证]`    |
| 标准湿空气 | `HAPropsSI("speed_of_sound","T",…,"P",…,"R",RH)`                                                                             | **否**（干空气伪纯固定；且不认键名 `"A"`，须用 `"speed_of_sound"`） | `[已验证：边界]` |

单位易错点：CoolProp 用 **Pa / K / 摩尔分数(0–1)**；项目用 **MPa / °C / 百分数**。

同点对照（干气 CO₂=1%、O₂=20.9%、N₂=78.1%，T=35°C）：

| 工况                     | 项目理想气 c       | CoolProp HEOS c | Δc / 相对          |
| ---------------------- | ------------- | --------------- | ---------------- |
| P=0.1 MPa              | 351.29        | 351.36          | +0.07 m/s（0.02%） |
| P=0.709 MPa            | 351.29        | 352.16          | +0.87 m/s（0.25%） |
| 同上 + H₂O 5 mol%（P=0.1） | 354.15（理想气加水） | 354.18          | 与 HEOS 差 <0.02%  |

结论：**近常压不必为换 EOS 而换 EOS**；优先把 H₂O 纳入声速（理想气加水汽与 CoolProp 常压几乎重合）。高压端若要做真实气体表，P∈[0.1, 0.709] MPa 必须单独扫。

#### 1.2 在本项目中的接入方式（未改正式训练链路）

当前：`hidden_sound_speed_v2` 无水/RH/P；`hidden_attenuation_v2` 经 `h2o_mole_percent_from_rh` 含水；`waveforms._compute_physics` 可注入 `sound_speed_fn`，但声速函数**收不到** `p_mpa` / `h_rh`（仅衰减函数收）。

| 落法                         | 内容                                                                                                 | 何时用                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------ |
| **A. 离线对照表**               | 脚本扫网格：`c_ideal` vs `c_coolprop`（干/湿/高低压）+ O₂×H₂O 交叉灵敏度 → CSV                                       | 方向 A 剩余工作；**不改**正式 `tv3-formal-6000` |
| **B. 湿度进理想气**              | `x_H2O = h2o_mole_percent_from_rh(...)`，干组分按 `(1−x_H2O)` 缩放后闭包，常数加 H₂O（M=0.018015, cp=33.58）       | 修干/湿不一致；CoolProp 作验收，不必训练时每次调用       |
| **C. 可选 CoolProp backend** | 闭包或扩展 `sound_speed_fn` 签名以传入 P/RH，经 `simulate_waveform_measurement(..., sound_speed_fn=...)` 生成消融集 | 湿度校正 TOF 消融；**不**作默认正式 backend       |

骨架（落法 A / 验收用）：

```python
import CoolProp.CoolProp as CP
from tv3.sim.generation.gas_state import h2o_mole_percent_from_rh

def coolprop_c(x_co2, x_o2, x_n2, t_c, p_mpa, h_rh):
    x_h2o = h2o_mole_percent_from_rh(t_c, p_mpa, h_rh) / 100.0
    dry = 1.0 - x_h2o
    z = [x_co2/100*dry, x_o2/100*dry, x_n2/100*dry, x_h2o]
    s = sum(z); z = [v/s for v in z]
    AS = CP.AbstractState("HEOS", "CarbonDioxide&Oxygen&Nitrogen&Water")
    AS.set_mole_fractions(z)
    AS.update(CP.PT_INPUTS, p_mpa * 1e6, t_c + 273.15)
    return AS.speed_sound()
```

依赖：`pip install CoolProp`（可选，不进主训练硬依赖）。正式训练默认 backend **暂不替换**。

#### 1.3 同源热导输出（补充，`[待验证：未实跑]`）

同一 `AbstractState("HEOS", "CarbonDioxide&Oxygen&Nitrogen&Water")` 除声速外可直接出热导：`AS.conductivity()` 或 `PropsSI("conductivity"/"L", ...)`。用途：① 给含水汽的真实 `λ_mix` 做参照，补第八节的 λ_mix 湿度缺口；② 验收 `_hidden_lambda_mix` 的 WMS 混合规则在 tv3 网格上的偏差。**不需要新工具，与声速验收同源**。REFPROP 是 λ_mix 金标准但本机 DLL 未装（`[未就绪]`）。混合规则本身有 Mukhopadhyay 1967 文献支撑，不必再找库。

### 2. ISO 9613-1:1993 / Bass et al. 1995 JASA 大气吸收 —— 替换衰减 proxy

- 国际标准 / 权威文献，给出含 N₂、O₂ 振动弛豫和显式湿度/温度依赖的验证过的吸收公式。代码现只引用 Bass 1990 的 O₂ 弛豫频率，未实现完整吸收模型。
- Python `acoustics` 包含大气吸收实现可参照（`[待验证]` 具体 API 未实跑）。
- 文献：Bass, Sutherland, Zuckerwar 1990, JASA 88(4), DOI:10.1121/1.400176（2026-07-24 修正；旧引 10.1121/1.400476 指向无关文献）；Bass & Sutherland 2004, JASA, DOI:10.1121/1.1631937。

### 3. HITRAN / HITEMP（hapi）—— NDIR 与 O₂ 光学通道

- 逐线算 CO₂ 4.26μm 吸收替换经验式 `0.045·x_CO2 + …`；记忆库已把 tv3 HITRAN 后端列为“待后续阶段”。
- 关键：HITRAN 有 **O₂ A-band 760 nm** 谱线——这是记忆库里 TDLAS 路线、也是唯一能给 O₂ 独立光学信息、真正突破 0.70 的物理后备。
- hapi 在主线 hydrogen_ng 已使用，工具链存在。

### 4. COMSOL 完整时域 CWE 求解 —— 最高保真的波形目标

- 现 P0 只对了解析 c。真正升级是在 3D 气室几何里解热粘声波方程，得到**含衍射 / 壁反射 / 热粘损耗的接收波形**，替换“单脉冲太干净”的代理。几何、mph、脚本已就位（`COMSOL/`），成本高但基础设施在。

### 5. 实测换能器脉冲响应（PSC200K 表征）

- 替换估计的 20 kHz / 二阶谐振 proxy，直接影响 TOF 分辨率真实性。需真实器件测量或厂商完整 datasheet。

### 6. 已发表超声气体传感器精度数据

- 用真实报道的 c/TOF 精度、SNR 标定抖动和噪声量级，判断 σ=3μs、噪声 1 mV 是否偏乐观。参照 `references/tunnel_ventilation_sensing_survey.md`。

---

## 六、可选下一步（独立可选）

| 方向     | 内容                                                                   | 成本  | 状态 / 直接价值                                          |
| ------ | -------------------------------------------------------------------- | --- | -------------------------------------------------- |
| **A**  | CoolProp：真实气体 c vs 理想气 + 湿度→声速灵敏度                                    | 低   | **冒烟已完成**（§4 / §5.1）；剩余：正式网格 CSV 落盘脚本。湿度结论已为数据库级证据 |
| **A′** | 落法 B：湿度进理想气声速（可加 `H_RH` 校正 TOF 消融）                                   | 低–中 | 修干/湿不一致；CoolProp 作验收。**尚未改代码**                     |
| **B**  | 按 ISO 9613-1 / Bass 1995 实现验证过的大气吸收，替换 `hidden_attenuation_v2` proxy | 中   | 衰减通道；湿度物理一致性与 A′ 配合                                |
| **C**  | 跑一次 COMSOL 时域 CWE，导出真实气室接收波形，与单脉冲 proxy 对比多径/拖尾差异                    | 高   | 量化 D2b 的乐观性                                        |

推荐：方向 A 冒烟已够支撑湿度缺口判断；下一步优先 **A′（湿度进声速）** 或 A 的网格 CSV 落盘，再视需要做 B/C。

---

## 七、验证方法与数据来源

- **量化实算**：用 `acoustic_physics.py` 的 `_GAS_M`/`_GAS_CP` + 水汽 (M=0.018015, cp=33.58) 计算扰动 Δc/ΔTOF（第四节表），T=35°C、L=0.2 m。
- **CoolProp 复核（2026-07-20）**：CoolProp 8.0.0；`HEOS` 四元混合声速；同点对照见 §4 / §5.1。本机 `pip install CoolProp` 后实跑；未改项目源码与正式 benchmark。
- **代码核对**：`hidden_sound_speed_v2` 签名无水/RH 入参（声速不含湿度）；`hidden_attenuation_v2` 经 `h2o_mole_percent_from_rh` 引入水汽（衰减含湿度）——两者不对称，为干/湿不一致来源。
- **CSV 核对**：`COMSOL/p0_sound_speed_acceptance.csv` 全 `rel_err=0.0`。
- **未执行**：REFPROP DLL 未安装；CoolProp 全网格 CSV 脚本未入库；COMSOL 时域 CWE 未求解；`acoustics` 包 API 未实跑；落法 A′/C 代码未落地。
- **热学补充审查（第八节）方法**：代码核对 `_hidden_lambda_mix` / `_wilke_phi` / `_lambda_at_t` / `_tcs_voltage`（`acoustic_physics.py:264–358`）；纯组分 λ/η/n 与 `physics_references.md` §3 交叉核对；第八节 SNR 与 λ_mix 灵敏度为 `[工程估算]`（项目常数手算，**未跑 CoolProp 热导实测、未跑训练验证**）。

---

## 八、TCS / 热导链路补充审查（2026-07-20）

原第七版仅系统审查声学链路，第八节明确"NDIR 与 TCS 未系统审查"。本节补上 TCS/热导链路，结论与声学链路**同源**。

### 8.1 被审查链路

```
组分(CO₂/O₂/N₂) → 纯组分 λ_i(T)=λ₀·(T/298.15)^n              [_lambda_at_t]
               → WMS 混合 λ_mix = Σ y_i·λ_i / Σ y_j·φ_ij     [_hidden_lambda_mix, φ_ij 由 Wilke 公式算]
               → V_TCS = 1.1 + 15.0·(λ_mix−0.026) + 0.004·(T−20) + 漂移 + gauss(0,0.006)  [_tcs_voltage]
纯组分常数：λ(mW/m·K)=CO₂ 16.6 / O₂ 26.4 / N₂ 25.8；n=0.87/0.80/0.77；η(Pa·s)=1.491e-5/2.058e-5/1.781e-5
```

### 8.2 分层对齐（详见 §三新增两行）

- **混合物理层：高（主导物理忠实）**。WMS（Wassiljewa-Mason-Saxena）是被 Mukhopadhyay 1967 等实验验证过的标准混合规则，φ_ij 由 M、η 经 Wilke 公式直接算不查表，纯组分 λ 来自 NIST/CRC。与声速理想气公式属同一"下限忠实"层。`[已验证：代码 + NIST 常数]`
- **传递函数层：低（proxy）**。`_tcs_voltage` 的斜率 15.0、offset 0.026、baseline 1.1 是自标注 proxy，未绑定真实热导池（惠斯通电桥 + 热丝温度 + 气室对流散热几何）的传递函数。与声学换能器 proxy 同性质、同局限。`[已验证：代码常量]`

### 8.3 两个缺口

1. **湿度未进 λ_mix**（关键，与声速缺湿度同源）。`_hidden_lambda_mix` 只含 CO₂/O₂/N₂，不含 H₂O。水汽 λ≈18–19 mW/m·K、本场景最高 5 mol%，且 `H_RH` 可测。这与 §七"声速不含湿、衰减含湿"是**同一处干/湿系统性不一致**——三条链路里只有衰减进了水汽。补法与声速 A′ 一致：把 `h2o_mole_percent_from_rh` 的产物纳入 λ_mix，用 §5.1.3 的 CoolProp `conductivity` 验收。`[已验证：代码]`
2. **压力仅进漂移项**。`_thermal_baseline_drift` 的 `0.004·(p−1)` 是拟合漂移，不是热导池物理。常压下 λ 与 P 无关的近似本身可接受（Knudsen 过渡仅低压显现）；真实 TCS 对流速敏感，但 P0 为静止空气（flow=0），当前范围内不影响。`[工程判断]`

### 8.4 "热导补偿声学"可行性

分两问，结论不同：

- **(A) 热导作为独立通道约束 O₂/N₂：方向对，上限低**。声速由 M_mix/γ_mix 决定，热导由 λ_i/φ_ij 决定，对 (CO₂,O₂,N₂) 雅可比方向不同，联立能缩小 O₂/N₂ 简并——这正是 raw3 三输出融合、TCS 值得保留的物理依据。但信号弱：`[工程估算]` O₂ 18→21.2%（N₂ 反向）→ Δλ_mix≈1.9e-5 W/m·K → ΔV_TCS≈2.9e-4 V，噪声 σ=6e-3 V，单次 SNR≈0.05，比声学 O₂ 墙（TOF 跨度 1.30μs / 抖动 3μs ≈0.43）还低约一个量级。TCS 对 O₂/N₂ 只给"边际辨识力"（与 `physics_references.md` §3.3 一致），不能突破 D0 的 O₂ 物理墙（oracle val R²≈0.60）。序列多步平均可压噪，放进模型有意义，属锦上添花。TCS 真正的强对比是 CO₂（λ_CO₂=16.6 vs 空气~26），但 CO₂ 已被 NDIR 观测良好，热导增量有限。
- **(B) 用环境参数补偿声速：杠杆是湿度，不是热导**。T 已是声速显式输入，P 在理想气声速无效应（真实气体才 0.25%），都不需要热导来补。真正缺口是湿度：H₂O 0→5 mol% → ΔTOF≈−4.54μs，是 O₂ 信号（1.30μs）的 3.5×（§四）。`H_RH` 可测，故"测到的湿度校正 TOF"（§六 A′）绕不开——热导在这里帮不上忙，它自己也缺湿度。

---

## 九、边界与免责

- 本报告覆盖**声学链路**（声速/衰减/TOF/波形）与**热学链路**（TCS/热导，第八节）；NDIR 通道仅在对照时提及，未系统审查。
- 所有结论限定在**当前仿真数字孪生范围**，不得改写为真实静止空气或掘进通风现场能力（与记忆库 §2.2-15、`COMSOL/README.md` 一致）。
- 第四节的湿度结论、第八节的 TCS SNR / λ_mix 缺口均是**仿真链路缺口**判断，不是对 O₂ 可辨识性最终否定；CoolProp 已把声速湿度 Δc/ΔTOF 升为数据库级证据，热导侧灵敏度仍为 `[工程估算]` 未实测。
- CoolProp 复核不构成对正式集重生成的授权；默认 backend 仍以理想气三组分声速为准，直至实验矩阵明确启用 A′/消融集。热导补湿度、TCS proxy 标定同理，未获重生成授权。
- 与记忆库冲突时，以记忆库正式指标与不变量为准。
