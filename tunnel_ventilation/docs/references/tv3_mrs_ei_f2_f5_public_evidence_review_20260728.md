# tv3 MRS-EI F2-F5 文献与公开数据证据审查

> 审查日期：2026-07-28  
> 适用阶段：下一次版本化 MEI-0 / MEI-1 重新冻结  
> 结论范围：`obs-cfreq`、25--200 kHz、登记仿真域内的信息设计；不授权正式波形、benchmark 打包或硬件试验。

## 1. 研究问题

1. F2--F5 是否已有可追溯文献，足以确认机制、公式或校准路径？
2. 是否存在覆盖当前组分、15--35 °C、25--200 kHz、0.101325--0.709 MPa 以及具体设备几何的公开数据，可直接形成数值包络或独立 holdout？
3. 证据不足时，怎样缩小结论范围，使未解决项不再阻断登记仿真研究，同时不制造硬件能力声明？

## 2. 检索与核验方法

使用已挂载的学术检索 MCP 对 CrossRef、Semantic Scholar、OpenAlex、Scopus 和 ScienceDirect 做宽检索与定向复核。Scopus 对请求字段返回未授权；Zenodo 连接器返回日期解析错误。其余来源继续使用。候选文献必须至少由 DOI 元数据或机构库记录核验；只有摘要明确给出实验条件、公式用途或定量结果时，才用于本审查的事实判断。

公开数据筛选区分三类：

- `direct_holdout`：公开原始数组覆盖项目目标域，可直接用于独立比较；
- `method_and_parameter_evidence`：论文给出公式、速率、校准法或实验趋势，但没有可直接复用的项目域原始数组；
- `discovery_only`：只有题录或不匹配当前设备 / 几何，不能形成数值边界。

本轮未找到覆盖 F2--F5 全部目标域的 `direct_holdout`。这不是把所有项目继续设为 MEI-1 阻断的理由；应按结论层级处置。

## 3. 证据矩阵

| Family | 已确定事实 | 公开数据结论 | 本轮处置 |
| --- | --- | --- | --- |
| F2 H2O 弛豫参数 | H2O 与 O2/N2 的能量交换、湿度催化和压力缩放有经典实验及 Bass 公式链 | 有参数与实验论文，无覆盖本项目全温压域的原始 holdout | `parked_nonblocking`；对 `obs-cfreq` 信息设计非阻断，对衰减精度和硬件声明仍限制 |
| F3 耦合弛豫 | 多组分 V-T / V-V 耦合矩阵已有 Dain--Lueptow 理论，且多类 N2 混合气有变压多频实验验证 | 有模型和实验趋势，无当前 O2/CO2/N2/H2O 全域现成实现或数据集 | `parked_nonblocking`；保留 5% 结构代理诊断，不把代理称为模型验证 |
| F4 衍射 / 近场 | 有限孔径会改变幅度和相位；超声渡越时间流量计文献直接要求衍射修正 | 2026 年开放论文给出 150/500 kHz 圆活塞模型，但项目未登记孔径、管径和安装几何 | `parked_nonblocking`；不阻断理想传播信息设计，继续阻断设备级传播声明 |
| F5 换能器响应 | 空气耦合换能器存在幅相个体差、窄带响应和非互易；50--300 kHz 互易校准方法公开 | 无当前 `device_profile_id` 的公开校准数组；其他器件数据不能替代本机标定 | `parked_nonblocking`；在 TOF-only 信息门中排除设备响应声明，正式波形与硬件仍禁止 |

## 4. F2：H2O 弛豫参数

### 4.1 已确认

- Fujii、Lindsay 和 Urushihara 在约 65 kHz 测量 N2、O2 和水蒸气吸收；水蒸气测量覆盖 42--183 °C，并报告约 `1.5e-8 s` 的振动弛豫时间量级（DOI `10.1121/1.1918640`）。
- Bass、Keeton 和 Williams 使用 H2O/O2 超声吸收及其他实验建立随温度变化的能量交换速率（DOI `10.1121/1.381050`）。
- Keeton 和 Bass 对 H2O 与 N2/Ar 混合物给出带不确定度的去激活速率，但主要实验温度为 500 K（DOI `10.1121/1.381051`）。
- Bass、Sutherland 和 Zuckerwar 给出 O2/N2 振动弛豫频率的湿度公式（DOI `10.1121/1.400176`）；1995 年工作显式给出吸收、频率和湿度相对压力的缩放，并有 1996 erratum（DOI `10.1121/1.412989`、`10.1121/1.415223`）。

当前代码已经实现 Bass O2/N2 频率公式和压力缩放。未确定的是独立 H2O 过程使用的 `alpha_lambda_max_h2o=0.01`，而不是 O2/N2 湿度催化本身。

### 4.2 项目内量级复核

在当前 6 个候选频率和 216 点环境域上，现有 H2O 单弛豫项对 `c_f` 的最大贡献为 `0.02495837186858439 m/s`，95 分位为 `0.022059148041716714 m/s`。即使把该项按 100% 不确定度处理，在 `L=0.3 m`、`c=320 m/s` 的保守换算下，最大 TOF 变化约 `0.0732 us`，低于低成本口径 `0.5 us` 的单次 TOF 标准差。

这只支持“F2 不阻断当前 TOF-only 频点信息比较”，不支持衰减预测已验证，也不支持把 H2O 强度误差设为零。

### 4.3 搁置边界

- 非阻断范围：`obs-cfreq`、登记仿真域、K4 相对信息排序。
- 继续限制：使用 `alpha_observed` 的定量结论、正式波形、硬件精度声明。
- 复查触发：取得 15--35 °C、目标湿度与压力下的 H2O/O2/N2 吸收数组，或实现带文献速率的耦合模型并通过独立 holdout。

## 5. F3：耦合弛豫

### 5.1 已确认

- Zuckerwar 和 Griffin 的 N2--H2O 分析表明 VVR 交换可显著强于简单 VT 路径，并给出声学实验速率（DOI `10.1121/1.386227`）。
- Dain 和 Lueptow 建立三组分气体的 V-T / V-V 耦合理论，明确多种有效弛豫频率由组分耦合产生（DOI `10.1121/1.1352087`）。
- Ejakov 等使用四组不同中心频率的换能器并改变压力覆盖宽 `f/p`，对 air、O2/N2、CH4/N2、CO2/N2、H2/N2 等混合物比较模型与实验；经过衍射修正后，模型能匹配广泛混合物的衰减谱趋势（DOI `10.1121/1.1559177`）。

因此“耦合机制是否存在”和“有无可实现理论路径”已经确定。尚未确定的是完整耦合模型相对当前单弛豫加和模型在本项目完整组分 / 温压域的数值偏差。

### 5.2 项目内诊断

当前 MEI-1 的 5% cross-mix 代理在 432 点、双噪声审计中没有瓶颈翻转；正式并集上相对固定 K4 最大 P90 的变化不超过约 `0.0362%`，主角最大约 `0.02394 deg`，均远低于 `delta_practical=2%`。代理没有物理边界，故不能升级为 `represented_traceable`；但它说明继续把 F3 当作无限大、无条件阻断也缺少比例原则。

### 5.3 搁置边界

- 非阻断范围：当前单弛豫加和模型下的仿真信息结论，并强制标注 `single_relaxation_sum_model`。
- 继续限制：声称完整耦合物理已验证、现场衰减绝对精度、外推到未登记组分。
- 复查触发：实现 Dain--Lueptow 耦合矩阵，或获得当前组分与温压域的独立多频吸收 / 声速数据。

## 6. F4：衍射与近场

### 6.1 已确认

- 超声渡越时间流量计文献明确指出有限孔径衍射会改变幅度和相位，并可能影响精确渡越时间（DOI `10.1121/1.4988231`、`10.1121/10.0019023`）。
- Tchatat Ngaha 和 Frøysa 的开放 Acta Acustica 论文给出圆活塞、均匀 / 抛物线横向流的三维窄角抛物方程，并在 150 kHz 和 500 kHz 展示幅度、慢变相位及波束弯曲（DOI `10.1051/aacus/2026047`）。
- Ejakov 等的实验需要对每个换能器对做衍射修正后，不同频率在相同 `f/p` 下的数据才重合（DOI `10.1121/1.4777297`、`10.1121/1.1559177`）。

### 6.2 为什么不能直接生成项目包络

衍射修正依赖孔径、接收面、声程、管径、安装角度、流场和边界。当前 MRS-EI 只冻结 `L=0.2--0.3 m`，没有冻结这些几何量。用其他装置的修正曲线作为本项目数值包络会形成错误的第二来源真相。

当前 0.2% 诊断代理在正式并集上的 P90 相对变化不超过约 `0.0209%`，无瓶颈翻转；它继续保留为敏感性检查，不作为物理验证。

### 6.3 搁置边界

- 非阻断范围：理想均匀传播、点接收假设下的频点信息比较。
- 继续限制：设备几何下的绝对 TOF、幅相、声功率和现场精度声明。
- 复查触发：冻结实际孔径 / 管径 / 安装几何后运行解析圆活塞或独立 FEM / 台架 holdout。

## 7. F5：换能器幅相和群延迟

### 7.1 已确认

- van Deventer 和 Delsing 报告超声流量计中的表观换能器非互易（DOI `10.1016/S0041-624X(02)00152-X`）。
- Allevato 等对 64 个 40 kHz Murata MA40S4S 实测幅相差，表明制造公差可显著改变波束；校准后与理想波束更接近（DOI `10.1109/IUS54386.2022.9957576`）。
- Mosland 的公开学位论文给出空气中三换能器互易校准，覆盖 50--300 kHz，并显式包含空气吸收、衍射及收发电子学修正（机构库 `hdl:1956/7164`）。

这些来源确定了校准方法和设备依赖性，但不能提供当前未知设备的可迁移幅相 / 群延迟数组。

### 7.2 与当前观测契约的关系

当前 MEI-1 主臂是 `obs-cfreq`，F5 代理只改变 `alpha`，所以当前审计中完全 inert。后续若使用原始 TOF、相位或复传递函数，必须以 `device_profile_id x frequency_hz` 共享校准偏移及独立先验进入观测契约；不得把未知器件响应归到气体组分。

### 7.3 搁置边界

- 非阻断范围：不含设备幅相声明的 `c_observed` 信息设计。
- 继续限制：正式波形生成、同步多正弦、实际等声能比较、硬件精度和跨设备泛化。
- 复查触发：确定设备型号并取得逐频幅度、相位、群延迟和声功率校准数组及不确定度。

## 8. 公开数据筛选结论

| 来源 | 可访问内容 | 是否可作当前 holdout | 原因 |
| --- | --- | --- | --- |
| JASA / CrossRef DOI 论文链 | 题录、摘要，部分有公开 PDF 地址 | 否，作为公式与参数来源 | 多数未提供机器可读原始数组，且目标域不完整 |
| Acta Acustica 2026 | 开放全文，圆活塞衍射模型 | 否，作为实现路径 | 频率只有 150/500 kHz 示例，缺项目几何 |
| UiB `hdl:1956/7164` | 50--300 kHz 空气中互易校准论文 / 学位论文 | 否，作为校准协议 | 器件不是本项目 `device_profile_id` |
| Zenodo 检索 | 连接器日期解析失败 | 未确认 | 未据此声称无数据，只记录本轮检索失败 |

没有发现可直接复制进项目的 F2--F5 原始 holdout。公开论文足以确定模型和校准路径，但不足以把设备相关未知量伪装成通用数值边界。

## 9. 状态与阶段处置

下一版本 MEI-0 应新增 `parked_nonblocking`，其语义是：

1. 未解决项被明确记录，不能计为 `represented_traceable`；
2. 必须列出非阻断结论范围、仍被禁止的声明和复查触发条件；
3. 在限定的登记仿真信息设计中不再形成 MEI-1 blocker；
4. 四项正式授权继续保持 `forbidden_until_explicit_authorization`。

若重新运行 MEI-1 后 15 个 K4 仍全部位于 `delta_practical=2%` 内，应输出 `mei1_fixed_k4_retained`，跳过没有实践收益的 MEI-2 频点优化，固定 D0 `{25,63,100,200} kHz` 进入 MEI-3。该状态不授权任何新数据或硬件活动。

## 10. 核心参考文献

1. Fujii Y, Lindsay RB, Urushihara K. Ultrasonic Absorption and Relaxation Times in Nitrogen, Oxygen, and Water Vapor. JASA 35, 961--966 (1963). DOI `10.1121/1.1918640`.
2. Bass HE, Sutherland LC, Zuckerwar AJ. Atmospheric absorption of sound: Update. JASA 88, 2019--2021 (1990). DOI `10.1121/1.400176`.
3. Bass HE et al. Atmospheric absorption of sound: Further developments. JASA 97, 680--683 (1995); erratum JASA 99, 1259 (1996). DOI `10.1121/1.412989`, `10.1121/1.415223`.
4. Bass HE, Keeton RG, Williams D. Vibrational and rotational relaxation in mixtures of water vapor and oxygen. JASA 60, 74--77 (1976). DOI `10.1121/1.381050`.
5. Keeton RG, Bass HE. Vibrational and rotational relaxation of water vapor by water vapor, nitrogen, and argon at 500 K. JASA 60, 78--82 (1976). DOI `10.1121/1.381051`.
6. Zuckerwar AJ, Griffin WA. Vibrational-rotational energy transfer in mixtures of nitrogen and water vapor. JASA 69, S45 (1981). DOI `10.1121/1.386227`.
7. Dain Y, Lueptow RM. Acoustic attenuation in three-component gas mixtures--Theory. JASA 109, 1955--1964 (2001). DOI `10.1121/1.1352087`.
8. Ejakov SG et al. Acoustic attenuation in gas mixtures with nitrogen: Experimental data and calculations. JASA 113, 1871--1879 (2003). DOI `10.1121/1.1559177`.
9. Tchatat Ngaha D, Frøysa KE. Diffraction effects for acoustic beams propagating through a parabolic and uniform flow. Acta Acustica 10, 53 (2026). DOI `10.1051/aacus/2026047`.
10. van Deventer J, Delsing J. Apparent transducer non-reciprocity in an ultrasonic flow meter. Ultrasonics 40, 403--405 (2002). DOI `10.1016/S0041-624X(02)00152-X`.
11. Allevato G et al. Calibration of Air-Coupled Ultrasonic Phased Arrays. Is it worth it? IEEE IUS (2022). DOI `10.1109/IUS54386.2022.9957576`.
12. Mosland EN. Reciprocity calibration method for ultrasonic piezoelectric transducers in air. University of Bergen (2013). Handle `1956/7164`.
