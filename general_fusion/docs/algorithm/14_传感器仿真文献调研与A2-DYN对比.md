# 传感器仿真文献调研与 A2-DYN 对比

> 文档状态：调研结论，供修订 `13_Ar-He-CO2动态时间序列仿真与数据分布规划.md` 使用  
> 调研日期：2026-08-31  
> 适用对象：Ar–He–CO₂ 三传感器动态 benchmark  
> 结论性质：文献支持的设计评审，不代表 A2-DYN 配置已经冻结或代码已经实现

> 版本说明：本文记录 R4 实施前的评审快照；其中“当前计划”“2 Hz 首选”和“尚未实现”等表述不覆盖后续执行事实。R4 已在文档 13 对应代码中完成 direct HEOS、设备链和 pilot 资格审计，并冻结 `5 Hz / 240 s` 与 `US-CHIRP-XCORR-PARABOLIC-1`；正式数据生成仍未完成。

## 摘要

本次调研重点核对气体传感器论文如何划定仿真边界，尤其检查超声传感器是否只仿真“组成到声速或 ToF 的标量映射”，还是继续仿真换能器、传播波形和 ToF 估计。结论如下。

1. 当前 A2-DYN 的共同气室、组成守恒、进气协议、时序扰动和因果前缀评价是合理的过程级骨架，应保留。
2. 当前超声链路只使用理想气体声速和固定声程得到平衡 ToF，再施加 `tau=0.5 s` 的一阶响应。这可以作为最低保真度的组成映射，但不能代表论文中的完整超声测量过程。实验论文通常显式包含激励波形、声程与流速、接收波形、带通滤波、重复平均、参考信号以及互相关、相位或两者结合的时延估计 [1–6]。
3. 超声测量本身发生在微秒到毫秒尺度，论文中的最终过程输出可以是 2 Hz，但这不等于超声器件具有 0.5 s 固有一阶时间常数 [4,5]。当前计划把快速声学采集、测量平均和慢速气体交换混在了同一个 `tau_s` 中，这是最主要的结构性问题。
4. 在低压、已知气体集合和受控温度下，使用理想气体声速并非错误；多篇二元气体和掺氢天然气实验直接采用这一近似 [1,3–5]。但它只说明平衡声速算子可保留，不说明 ToF 观测可直接加独立高斯噪声后视为硬件信号。正式生成前仍应在整个 Ar–He–CO₂ 单纯形上与 REFPROP 或 CoolProp 做数值核对 [11,12]。
5. 单一声速是一个标量，无法单独唯一恢复任意三元组成。论文通过已知基气、固定天然气来源、二元假设，或同时测量声衰减来恢复可辨识性 [4,7,8]。A2-DYN 使用超声、热导和 NDIR 三路融合，因此不存在“超声单路必须独立解出三组分”的要求；但报告中不得把单一 ToF 写成三元超声分析仪。
6. 当前热导链路的 WMS 混合热导率可作为材料物性层，但“热导率直接线性映射到电压，再加固定 10 s 一阶响应”弱于论文常见的焦耳加热、热容、对流和导热耦合模型。µTCD 论文还表明，瞬态时间常数本身依赖气体热导率，而不是与组成无关的固定参数 [13–15]。
7. 当前 NDIR 的 Beer–Lambert 主干正确，但实际论文通常再积分光源谱、滤光片和探测器响应，使用检测通道与参考通道，并把气室扩散、光程平均、温湿度和器件温漂纳入模型 [16–20]。当前计划覆盖到近纯 CO₂ 端点，必须定义高量程短光程或双量程硬件 profile；否则普通 ppm 级 NDIR 会在高浓度区域产生没有研究意义的饱和。
8. 建议将 A2-DYN 定位为“中等保真度、双时间尺度、输出仍为三路低频传感器读数”的 benchmark：外层继续使用 1、2、5 Hz 比较气室与任务动态；超声在每个外层时刻内部合成短波形并执行 ToF 估计，只保存 ToF 和审计质量量，不保存全部高频 ADC。这样能纠正主要物理缺口，又不会把数据规模从约 120–180 MB 推高到几十 GB 甚至 TB。

综合判定：当前计划不是整体推倒重来，而是需要在 A2-DYN-0 冻结机器协议前，对三路“平衡物性 → 实际观测”之间的设备层进行结构性修订。优先级最高的是拆分超声快速采集与慢速输运；其次是把热导固定一阶响应改成电热状态模型，以及明确 NDIR 高浓度量程。

---

## 1. 调研问题

本次调研冻结以下四个研究问题，避免按搜索结果临时改变评审标准。

### RQ1：气体超声传感器通常仿真哪些层？

区分以下对象是否被建模：

- 组成到声速的热力学映射；
- 气体在管道、气室和声程中的输运；
- 换能器与电子学响应；
- 发射、传播和接收波形；
- ToF、相位、互相关或声衰减估计；
- 最终低频浓度或流量输出。

### RQ2：热导和 NDIR 传感器的常见仿真边界是什么？

重点检查论文是否把材料物性直接映射为电压，还是继续建模热、电、流体、光谱、气室扩散和参考通道。

### RQ3：当前 A2-DYN 计划在哪个保真度层级？

逐项比较共同气室、平衡算子、传感器动态、采样率、噪声、组成分布和验证协议。

### RQ4：哪些修订是正式 benchmark 的必要条件？

区分：

- v1 必须修订；
- pilot 中应比较；
- 只有真实硬件参数后才值得实现；
- 当前研究范围内不需要实现。

---

## 2. 检索与证据方法

### 2.1 检索范围

检索围绕五组关键词进行两轮收敛：

1. 气体组成、超声、声速、ToF、互相关、相位和衰减；
2. 超声换能器等效电路、有限元、气体传播和实验验证；
3. 热导检测器、焦耳加热、瞬态响应和有限元；
4. NDIR、Beer–Lambert、HITRAN、气室扩散、参考通道和动态响应；
5. REFPROP、CoolProp、理想气体和混合物物性。

优先保留期刊原文、出版社页面、PubMed、期刊全文页和官方数据库说明。综述、透视文章和厂商材料只用于发现术语，不单独支撑关键设计决策。

### 2.2 纳入标准

纳入文献至少满足一项：

- 给出可复现的传感器物理或信号处理链；
- 把仿真与实验波形、物性或浓度结果比较；
- 明确列出动态响应、采样、平均、滤波或校准方式；
- 是本项目使用的物性或光谱数据库的正式说明。

排除仅报告机器学习精度、没有说明传感器生成过程的论文；排除与气体耦合机制无关的通用图像超声工作。

### 2.3 证据类型与用途

| 证据类型 | 代表来源 | 可支撑的结论 | 不能支撑的结论 |
| --- | --- | --- | --- |
| 气体组成超声实验 | [1–5,7] | 声速关系、激励和 ToF 估计、温度与流速影响 | Ar–He–CO₂ 专用硬件参数 |
| 超声系统仿真并与实验比较 | [6,9,10] | KLM 或 FEM 可产生频域和时域系统响应 | v1 必须逐样本运行全 FEM |
| 热导传感器仿真并与实验比较 | [13–15] | 电热耦合、组成相关时间常数、流速影响 | 当前宏观传感器一定为微秒响应 |
| NDIR 设计与动态实验 | [16–20] | 光谱积分、参考通道、扩散和温漂的重要性 | 任意光程可覆盖 0–100% CO₂ |
| 官方物性与光谱数据库 | [11,12,21] | 平衡物性和谱线计算的可信事实源 | 未校准硬件噪声和时间常数 |

### 2.4 证据限制

- 没有找到与本项目完全一致的 Ar–He–CO₂、同一气室、三传感器同步动态公开数据集。
- 文献硬件覆盖 50 kHz 至数 MHz、毫米至几十厘米声程，参数不能直接拼成一个“典型超声传感器”。
- µTCD 的微秒级器件响应和商用宏观模块的秒级换气响应属于不同系统边界，不能直接比较大小后判定对错。
- NDIR 文献常面向 ppm 到几个体积分数百分比；本项目的全单纯形包含高 CO₂ 区域，必须另外定义量程。
- 因此本报告给出结构、参数关系和需要验证的范围，不把文献中的单个数值当作本项目硬件真值。

---

## 3. 传感器仿真的五层分类

不同论文所谓“传感器仿真”并不是同一对象。为避免把不同层级混合，统一分为五层。

| 层级 | 输入 | 输出 | 典型方法 | 当前 A2-DYN |
| --- | --- | --- | --- | --- |
| L0 物性层 | 组成、温度、压力 | 声速、热导率、吸收系数 | 理想气体、EOS、WMS、HITRAN | 已有 |
| L1 过程与输运层 | 进气协议、流量、体积 | 局部或路径组成 | CSTR、扩散方程、CFD | 有共同 CSTR，无各传感器死体积 |
| L2 器件层 | 局部物性、激励 | 温度、位移、光强、电荷 | 热网络、KLM、FEM、光线追迹 | 基本缺失 |
| L3 采集与估计层 | 模拟波形 | ToF、电压、质量量 | 滤波、平均、互相关、相位、锁相 | 超声缺失；其余仅简化电压 |
| L4 任务数据层 | 低频读数与扰动 | 模型可见序列 | 漂移、AR(1)、量化、缺测 | 规划较完整 |

当前计划的优势集中在 L0、L1 和 L4，主要缺口是 L2 和 L3。正式 benchmark 不需要把三路都提升到全三维 FEM，但至少需要一个可解释的中间器件状态，使“物性变化为何形成观测变化”不再完全由固定线性映射和任意一阶时间常数代替。

---

## 4. 超声传感器：论文中的仿真与测量链

### 4.1 第一类：只建模平衡声速和组成

二元气体分析中，常见起点是

$$
c=\sqrt{\frac{\gamma RT}{M}},
$$

并按摩尔分数混合摩尔质量和定压、定容热容。早期二元气体分析覆盖过 CO₂–空气、He–空气和 Ar–SF₆；后续仪器将声速曲线、温度和压力映射为二元组分含量 [1,3]。

掺氢天然气的近期实验也在低压条件下采用理想气体关系。该工作明确指出，若天然气来源保持不变，可把天然气看成具有有效物性的一个组分，从声速估计氢摩尔分数；若天然气自身组成变化，这一简化就不再成立 [4]。

这类工作说明：

- 当前 `ideal_gas_sound_speed` 适合作为低压 v1 的名义算子；
- 温度必须进入声速或补偿链；
- 对固定已知二元体系，可用查表或标定曲线反演浓度；
- 不能由此推出单一声速能唯一决定任意三元组成。

### 4.2 第二类：显式考虑声程、流速和双向传播

流动气体中的上、下游 ToF 为

$$
t_d=\frac{L}{c+v\cos\phi},
\qquad
t_u=\frac{L}{c-v\cos\phi}.
$$

双向传播可同时估计平均流速和声速 [4,5]。实际结构还要考虑换能器没有正好位于流道壁面、无流死区、声程角度和速度剖面，部分修正项需要实验标定 [5]。

对当前计划的含义是：

- 如果超声通道宣称是在线流动测量，固定 `ToF=L/c` 会把流速影响完全删掉；
- 若 v1 明确采用横穿静态气室、声路与主流近似正交，则可以忽略一阶流速项，但必须把这个几何假设写进 hardware profile；
- 若保留斜置双换能器，应生成 `t_up` 和 `t_down`，再得到声速，而不是只生成一个抽象 ToF。

### 4.3 第三类：从激励波形到 ToF 估计

实验系统并不直接观测理论 ToF。典型链路是：

$$
x_{tx}(t)
\rightarrow h_{tx}
\rightarrow h_{path}(c,\alpha,L,v)
\rightarrow h_{rx}
\rightarrow y_{adc}(t)
\rightarrow \widehat{t}_{ToF}.
$$

文献中存在多种具体实现：

- 50 kHz 脉冲串、密封光滑气体管、40 MHz 计时器和声速查表 [3]；
- 1.65 MHz CMUT 使用发射与接收信号的相位差；约 16–20 kHz 线性啁啾使用互相关峰值 [4]；
- 490 kHz 单周期方波、500 Hz 重复频率、50 次平均、400–800 kHz 四阶 Butterworth 带通，再用参考波形互相关做粗时移、瞬时相位做亚采样精化 [5]；
- 通过超声信号幅度图样而不是单一阈值到达时间实现毫秒级气体浓度变化测量 [2]。

这些论文共同说明：

1. ToF 误差不是独立于波形的抽象白噪声；它取决于带宽、SNR、脉冲形状、相关峰、相位、采样率和多径。
2. 测量平均会形成独立的时间窗口。以 500 Hz 脉冲重复并平均 50 次为例，仅平均窗口就是约 0.1 s，但这仍不是换能器本体的 0.5 s 一阶惯性 [5]。
3. 参考信号和声程校准是测量链的一部分。路径温度误差会直接变成组成误差 [3–5]。
4. 过程输出频率可低于内部 ADC 数个数量级。2 Hz 输出不要求以 2 Hz 采样超声载波。

### 4.4 第四类：换能器等效电路或有限元

在器件设计论文中，压电换能器常使用 Mason、Redwood 或 KLM 等效电路；KLM 可在频域和时域表示压电层、背衬、匹配层和损耗，并可生成脉冲回波 [9]。更高保真度工作使用轴对称或三维有限元同时模拟压电发射器、接收器、气体传播、线缆和电子学，并与空气中的传递函数和端口电压波形比较 [6,10]。

这些方法适合：

- 设计换能器尺寸、谐振频率、背衬和匹配层；
- 建立一套固定硬件的脉冲响应；
- 研究近场、模态耦合、衍射和封装。

它们不适合在 6,300 条、每条 480 点的训练数据生成中逐点运行。对 A2-DYN 更合理的用法是：离线得到或假定一个已验证的参考脉冲响应，再在批量生成中进行时移、幅度、衰减、多径和噪声扰动。

### 4.5 第五类：声速之外的衰减和频散

单一声速主要反映等效摩尔质量和热容比，不能可靠地区分具有相近声速响应的未知多组分气体。研究工作因此同时测量 ToF 和声衰减，并利用 1–4.5 MHz 的频率响应区分 N₂ 中的 H₂、CO₂ 和 CH₄，同时估计温度和湿度影响 [7]。声学弛豫模型进一步把频率相关吸收和频散与分子能量交换联系起来 [8]。

对 A2-DYN 的含义不是立即增加一套完整 QARS，而是：

- 单一 ToF 应被描述为融合阵列的一路物理投影；
- 可以从内部超声波形额外导出 `attenuation`、`peak_correlation` 或 `snr` 作为审计量；
- v1 是否把衰减作为模型输入必须预注册，不能在看到结果后追加；
- 完整多频声弛豫属于后续声学增强版，不是 v1 的必要条件。

### 4.6 超声时间尺度必须拆开

超声链至少包含四个不同时间尺度：

| 时间尺度 | 物理来源 | 典型数量级 | 当前计划处理 |
| --- | --- | ---: | --- |
| 声传播 | `L/c` | 数十 µs 至约 1 ms [4,5] | 被压成平衡 ToF |
| 单次发射与振铃 | 载波周期、带宽和换能器脉冲响应 | µs 至 ms | 缺失 |
| 重复平均与输出刷新 | 脉冲重复、平均次数和处理 | 约 0.01–0.1 s 或设备定义 [5] | 混入 `tau=0.5 s` |
| 气体输运 | 主气室、传感器死体积、流量和扩散 | 可能为秒至分钟 | 由 CSTR 和 `tau_s` 混合表示 |

因此，当前 `tau_ultrasonic=0.5 s` 不能继续同时代表以上四类过程。建议把它拆成：

- `tau_mix`：共同气室交换；
- `tau_path_us`：超声声路或传感器小腔体的气体输运，可为 0 或小值；
- `acquisition_window_us`：重复平均与刷新窗口；
- `h_us`：高频参考脉冲响应；
- 不再设置抽象的“超声固有一阶时间常数”，除非有硬件阶跃标定支持。

---

## 5. 热导传感器：论文中的仿真方式

### 5.1 物性层

WMS 混合规则适合从组成得到混合热导率，是热导链的 L0 层。它没有描述热导率如何改变加热元件温度、电阻和桥路电压。

### 5.2 集总电热模型

µTCD 论文以焦耳加热和气体导热建立能量平衡。简化后可写成

$$
C_h\frac{dT_h}{dt}
=P_{Joule}-G_g(k_{mix},v,\rho,c_p)(T_h-T_g)-G_s(T_h-T_{sub}),
$$

再通过温度电阻系数和 Wheatstone bridge 得到电压。瞬态实验与仿真表明，微桥温度或电阻在电流阶跃后呈指数变化，但其时间常数与周围气体热导率相关 [13]。这与“先算平衡电压，再乘一个和组成无关的固定 10 s 一阶滤波器”有本质区别。

### 5.3 多物理场有限元

另一类论文在 COMSOL 或类似平台中耦合：

- 焦耳加热；
- 固体热传导；
- 气体热传导；
- 强制或自然对流；
- 流速、密度和定压热容；
- 加热器和热敏电阻几何。

动态 MEMS 热流传感器在 200 Hz 交流加热下用温度幅值和相位分离热导率与流速影响 [14]。金属 TCD 有限元则直接计算不同气体中悬空微桥的温度和电阻 [15]。

### 5.4 对 A2-DYN 的最小充分改法

v1 不需要逐样本三维热流 FEM，但应把热导通道改成一个集总电热状态模型：

1. `k_mix(x,T,p)` 仍由唯一 WMS 事实源提供；
2. `G_g` 显式依赖 `k_mix`，必要时依赖流速；
3. 对加热器温度进行解析或稳定数值更新；
4. 由 `R(T_h)` 和桥路得到电压；
5. 让响应幅度和时间常数随组成自然变化；
6. 将宏观传感器换气滞后与微桥热惯性分开。

若暂时没有器件几何，`C_h`、`G_s`、桥压和 TCR 可以定义为研究用 hardware profile，但必须标为合成参数，并通过无噪声阶跃、能量平衡和单调性测试约束。

---

## 6. NDIR：论文中的仿真方式

### 6.1 Beer–Lambert 是主干，不是完整设备

NDIR 的基础是

$$
I(\lambda)=I_0(\lambda)
\exp[-\kappa(\lambda,T,p)x_{CO_2}L].
$$

实际探测器读数通常是光源谱、光程分布、气体吸收、滤光片透过率和探测器响应的带宽积分：

$$
V_{act}\propto
\int S(\lambda)F_{act}(\lambda)D(\lambda)
\exp[-\kappa(\lambda)xL]d\lambda.
$$

多气体 NDIR 论文会依据 HITRAN 选择吸收带，模拟气室光程和光通量，并使用不同滤光通道检测不同气体 [16,21]。检测通道与不吸收或弱吸收的参考通道之比可抵消部分光源衰减和公共漂移 [16–18]。

### 6.2 气室扩散决定动态响应

NDIR 气室内的浓度并不一定等于入口或共同主气室的瞬时浓度。研究使用 Fluent 建立气室扩散与流动模型，发现光程上的归一化浓度主要随入口流速和扩散时间变化，并用分段指数模型近似 CFD 结果 [19]。小型片上 NDIR 也报告扩散限制的约 2.5 s 响应，同时观察到器件温度变化会影响 LED 和光电探测器 [20]。

因此，当前 `tau_ndir=8 s` 只有在它明确代表具体光学气室的换气与扩散，并经硬件或 CFD surrogate 标定时才有物理意义。它不应被写成所有 NDIR 的固有响应。

### 6.3 高 CO₂ 量程是当前计划的特殊风险

A2-DYN 组成分布覆盖 interior、边界、binary 和三个纯气端点，即 CO₂ 可能从接近 0 到接近 100 mol%。普通环境 CO₂ NDIR 常为 ppm 或几个体积分数百分比量程；Beer–Lambert 指数在固定长光程下会迅速饱和 [16–20]。

v1 必须三选一：

1. 定义可覆盖全单纯形的短光程高量程 CO₂ 通道；
2. 定义双光程或双增益量程，并冻结量程切换规则；
3. 把主 benchmark 的 CO₂ 域限制在硬件可测范围，纯 CO₂ 仅保留为明确的饱和 stress，不计入常规回归结论。

当前计划已要求单独报告 NDIR 低电压区，这是必要但不充分的。若大块组成区域都压到量化底部，不能靠更小噪声或随机抖动恢复信息。

---

## 7. 与当前 A2-DYN 计划逐项对比

### 7.1 总体评审

| 计划项 | 文献对照 | 判定 | 动作 |
| --- | --- | --- | --- |
| 共同 CSTR 气室 | 工程上常用低成本输运近似；NDIR 和超声论文显示局部死体积仍可能重要 [5,19] | 保留但降格为 well-mixed v1 假设 | 增加路径或传感器小腔体状态 |
| 理想气体声速 | 低压二元和掺氢天然气实验常用 [1,3–5] | 可保留 | 在全单纯形与 REFPROP、CoolProp 核对 |
| 固定声程 ToF | 静止横向声路可用；流动斜声路需双向传播 [4,5] | 条件保留 | hardware profile 冻结几何和流动假设 |
| 超声固定 0.5 s 一阶响应 | 文献的传播和估计是 µs–ms，刷新与平均另计 [4,5] | 必须修订 | 拆分气体输运、采集窗口和波形估计 |
| 不仿真原始超声波形 | 会绕过 ToF 估计误差和多径失效 | 必须调整边界 | 内部合成短波形，默认不作为模型输入或持久化数据 |
| WMS 热导率 | 是合理物性层 | 保留 | 不再直接等同完整传感器电压 |
| 热导率线性电压 + 10 s 固定响应 | 弱于电热耦合和组成相关瞬态 [13–15] | 必须修订 | 使用集总热网络；10 s 只可作为外部换气候选值 |
| Beer–Lambert NDIR | 正确主干 [16–21] | 保留 | 增加带宽积分、参考通道或明确有效系数的适用量程 |
| NDIR 8 s 固定响应 | 可能代表气室扩散，但不是通用器件常数 [19,20] | 条件保留 | 改名为局部气室输运或由 CFD surrogate 决定 |
| 2 Hz、240 s 外层序列 | 2 Hz 与过程级在线输出相符，相关论文也以至少 2 Hz 为目标 [5] | 保留为 pilot 候选 | 不用它解析超声载波；仍比较 1、2、5 Hz 任务收益 |
| AR(1)、漂移、量化 | 合理的 L4 统计扰动 | 保留 | 超声再增加由估计器产生的异方差和失锁事件审计 |
| 全单纯形分层抽样 | 对融合可辨识性有价值 | 保留 | NDIR 必须匹配高量程；超声不得单路声称三元可逆 |
| 因果前缀与动态基线 | 与真实在线任务一致 | 保留 | 时间收益必须与简单平滑和稳态幅值基线比较 |

### 7.2 当前计划最值得保留的部分

以下设计与文献调研并不冲突：

1. 用真实时间变化的气室组成替代把稳态值复制 480 次。
2. `mixture_id` 只表示真实配方组，重复观测使用独立 `observation_id`。
3. 目标组成、真实气室状态、真实动力学参数和 clean signal 不作为可部署输入。
4. 按 step、ramp、pulse、recovery 组织过程协议。
5. 区分 train、val、stress_val 和 test，并预注册分布。
6. 使用因果前缀评价早期估计，而不是只看完整序列离线精度。
7. 在 pilot 中比较 1、2、5 Hz，而不是把 2 Hz 写成硬件事实。
8. 对量化平台、NDIR 低电压区、动力学不可辨识和 oracle 失效设置停止规则。

### 7.3 当前计划必须修订的三处根因

#### 根因一：把输运、器件和估计混成一个一阶滤波器

共同气室 CSTR 已经表示一次慢输运，之后每路再施加固定一阶滤波，但这个滤波没有明确是传感器小腔体、器件热惯性、光学气室扩散，还是输出平均。不同来源的参数关系不同，无法由一个 `tau_s` 同时表达。

#### 根因二：平衡物性直接跳到最终电压或 ToF

声速、热导率和吸收系数是气体属性，不是 ADC 输出。直接跳转会让模型只学习三个平滑的解析投影，且噪声与失效模式主要由人工分布决定，而非由测量算法自然产生。

#### 根因三：硬件量程和几何没有参与数据分布

当前 sensitivity tier 主要扰动时间常数、噪声和漂移，但没有把声程、载频、带宽、发射波形、NDIR 光程、滤波带、TCD 热容和桥路作为分层 profile。结果可能在统计上复杂，却仍只代表一台没有几何和量程的抽象传感器。

---

## 8. 建议的 A2-DYN v1 双时间尺度架构

### 8.1 外层过程时间轴

保留：

- 240 s 总时长；
- 1、2、5 Hz pilot，2 Hz 为当前首选；
- baseline、transition、steady、recovery；
- 共同气室解析 CSTR；
- step、ramp 和 pulse 进气协议。

外层时刻 `t_k` 先得到共同气室组成 `x_ch,k`，再进入各传感器局部状态。

### 8.2 局部输运层

每个传感器使用明确命名的局部气体状态：

$$
x_{s,k+1}=x_{ch,k}+
(x_{s,k}-x_{ch,k})
\exp(-\Delta t/\tau_{transport,s}).
$$

其中：

- 超声若声路直接位于主气室，`tau_transport,us=0`；
- TCD 和 NDIR 若有旁路或小腔体，使用由体积、流量或 CFD surrogate 决定的输运时间；
- 该状态只表达气体交换，不再混入器件电热或信号处理时间。

### 8.3 超声内部采集层

对每个外层时刻生成一个短接收波形：

$$
y_{us,k}(t)=A_k
\left[r(t-\tau_k)
+\sum_m a_m r(t-\tau_k-\Delta_m)\right]
+n_k(t),
$$

其中：

- `r(t)`：固定硬件 profile 的参考波包，可来自实测、KLM 或参数化带限脉冲；
- `tau_k`：由声速、声程和可选流速计算的传播时延；
- `A_k`：几何扩散、经典吸收和可选组成相关衰减；
- 多径项用于形成相关峰歧义，但参数必须预注册；
- `n_k(t)`：ADC 噪声、电子耦合或干扰。

然后固定执行：

1. 重复平均；
2. 带通滤波；
3. 与参考波形互相关；
4. 可选相位精化；
5. 输出 `tof`；
6. 保存 `peak_correlation`、`snr`、`estimated_uncertainty` 为审计字段。

v1 的模型可见输入仍只保留 `tof`，从而不改变“三路传感器低频融合”的主任务。质量量默认只用于检查生成器是否产生合理失效，不应在看到性能后临时加入模型。

### 8.4 热导集总器件层

使用一个或两个热节点：

$$
C_h\dot T_h=P_{Joule}-G_g(k_{mix},v)(T_h-T_g)-G_s(T_h-T_{sub}),
$$

$$
R_h=R_0[1+\alpha_R(T_h-T_0)],
$$

再由固定桥路计算输出。这样可自然得到：

- 组成相关稳态幅值；
- 组成相关时间常数；
- 激励功率变化；
- 流速对冷却的交叉敏感性；
- 热容、基底导热和 TCR 的 hardware-profile 差异。

### 8.5 NDIR 光学与局部气室层

最低可接受的 v1 有两种实现：

#### 方案 A：窄带等效模型

保留单一有效吸收系数，但明确其适用温压、滤光带和 CO₂ 量程；增加 reference channel 与局部光学气室输运。

#### 方案 B：带宽积分模型

使用 HITRAN2020 计算吸收谱，积分光源、滤光片和探测器响应，生成 active 与 reference 两路，再映射为现有 NDIR 电压。该方案更可信，但计算和配置复杂度更高。

对当前 v1，建议先用 A 做正式候选，同时用 B 在组成和温压网格上验证 A 的误差；若高 CO₂ 区域误差或饱和不可接受，直接冻结为高量程 profile，而不是通过噪声参数修补。

---

## 9. 数据分布需要新增的变量

### 9.1 超声 hardware profile

| 参数 | 语义 | 建议分层方式 |
| --- | --- | --- |
| `path_length_m` | 声程 | 设备级固定，不逐时刻重抽 |
| `path_angle_deg` | 声路与流动夹角 | 与是否双向传播绑定 |
| `center_frequency_hz` | 载频 | 每个 hardware profile 固定 |
| `fractional_bandwidth` | 波包宽度和振铃 | 每个 hardware profile 固定 |
| `excitation_type` | pulse、burst 或 chirp | pilot 比较后冻结一种主配置 |
| `pulse_repetition_hz` | 重复发射频率 | 与平均窗口一致 |
| `average_count` | 重复平均次数 | 决定 SNR 与时间分辨率 |
| `reference_template_id` | 参考波形 | 设备级固定 |
| `multipath_profile` | 多径幅度与时延 | train 与 stress 分层 |
| `adc_rate_hz` | 内部波形采样率 | 只用于内部仿真，不等于外层采样率 |

### 9.2 热导 hardware profile

| 参数 | 语义 |
| --- | --- |
| `heater_heat_capacity` | 加热器等效热容 |
| `gas_conductance_scale` | 气体传热几何系数 |
| `substrate_conductance` | 基底散热 |
| `heater_power` | 焦耳激励 |
| `tcr` | 温度电阻系数 |
| `bridge_voltage` | 桥路激励 |
| `local_cell_volume` | 局部换气体积 |
| `flow_coupling` | 对流项强度 |

### 9.3 NDIR hardware profile

| 参数 | 语义 |
| --- | --- |
| `optical_path_m` | 有效光程或光程分布 |
| `active_band_id` | CO₂ 检测带 |
| `reference_band_id` | 参考带 |
| `source_spectrum_id` | 光源谱 |
| `detector_response_id` | 探测器谱响应 |
| `range_mode` | 高量程、低量程或双量程 |
| `local_cell_volume` | 光学气室体积 |
| `diffusion_profile` | 局部扩散 surrogate |
| `emitter_detector_tau` | 光源与探测器电子热响应 |

### 9.4 分组与 split 约束

新增 hardware profile 不改变项目主键不变量：

1. `mixture_id` 仍只表示真实组成组，绝不回退或重写为 `sequence_id`。
2. 同一配方的重复观测共享 `mixture_id`。
3. hardware profile、噪声和协议是观测条件，不产生伪造的配方 ID。
4. 若研究跨硬件泛化，应建立预注册的 `hardware_holdout` slice；不能把硬件差异混进普通 IID test 后再解释。
5. 新 benchmark 继续不依赖 `base_condition_id`、`noise_seed_index` 或 `noise_seed`。

---

## 10. 数据规模影响

当前计划的 6,300 条序列、每条 480 点，如果只保存三路低频读数、状态审计量和元数据，原规划的约 120–180 MB 仍然合理。

若按文献 [5] 的单个 4,300 点超声窗口保存每个外层时刻的一条 `float32` 波形，则仅一个方向、仅平均后的波形为：

$$
6300\times480\times4300\times4
=52,012,800,000\ \text{bytes}
\approx48.4\ \text{GiB}.
$$

若同时保存上、下游两个方向，约为 96.9 GiB。若把平均前 50 次重复全部保存，单方向约为 2.42 TiB，还没有计入中间滤波数组和元数据。

因此 v1 应采用：

- 生成时内部短暂创建波形；
- 立即执行固定估计器；
- 只持久化低频 `tof` 与少量质量审计量；
- 仅为小规模 fixture 或可视化样本保存原始波形。

这样既符合论文中的测量过程，又不破坏当前数据规模设计。

---

## 11. 修订优先级

### P0：A2-DYN-0 冻结机器协议前必须完成

1. 把超声 `tau=0.5 s` 拆成局部输运、内部采集窗口和波形估计；没有标定依据时删除固有一阶惯性。
2. 把“v1 不仿真原始超声波形”改为“内部仿真短波形，但默认不作为模型输入或完整持久化”。
3. 冻结超声几何：静态横向单程，或流动斜置双向；不能两种语义混用。
4. 在整个 Ar–He–CO₂ 组成、温度和压力网格上，把理想气体声速与 REFPROP 或 CoolProp 比较并形成误差表。
5. 将热导固定一阶响应替换为集总电热状态，至少让幅值和时间常数通过热导率进入同一能量平衡。
6. 冻结 NDIR 量程、光程和高 CO₂ 饱和规则。
7. 将共同气室、局部传感器气室和器件状态写成不同变量，避免第二来源真相。

### P1：A2-DYN-2 pilot 必须比较

1. 超声直接理论 ToF 与波形估计 ToF 的误差、失锁率和计算成本。
2. pulse、burst、chirp 三者中至少比较两种，再冻结一个主配置。
3. 互相关单独估计与“互相关 + 相位精化”的精度和稳健性。
4. 1、2、5 Hz 外层输出对早期组成估计的影响。
5. TCD 固定一阶代理与电热模型的动态信息差异。
6. NDIR 窄带等效模型与 HITRAN 带宽积分的网格误差。
7. 高 CO₂ 区域的饱和比例、量化平台长度和 oracle 可辨识性。

### P2：有真实硬件 profile 后再做

1. KLM 或 FEM 生成真实换能器脉冲响应。
2. 特定流道的 CFD 和声学近场仿真。
3. TCD 三维焦耳热、固体导热和流体对流联合 FEM。
4. NDIR 完整光线追迹、表面反射和热源有限元。
5. 由实测 reference waveform 和阶跃实验拟合局部输运参数。

### P3：当前不纳入 v1

1. 每条训练序列逐点运行全三维多物理场仿真。
2. 把 MHz 级原始波形作为主模型输入。
3. 完整声学弛豫谱或多频 QARS。
4. 为模拟硬件故障添加没有实验依据的复杂保护或静默 fallback。

---

## 12. 建议追加的验证不变量

### 12.1 超声

1. 无噪声、无流、无多径时，估计 ToF 与 `L/c` 的偏差低于预注册阈值。
2. 双向传播时，由 `t_up` 和 `t_down` 恢复的声速和流速同时通过解析 fixture。
3. 改变内部 ADC 率时，估计器误差按预期收敛，不能由输出裁剪制造一致。
4. `peak_correlation` 低于阈值时显式记为失锁；不得回退到理论 ToF。
5. 温度偏差、声程偏差、SNR 和多径分别做单因素误差传播。
6. 外层 2 Hz 只控制结果刷新，不参与载波离散化。

### 12.2 热导

1. 无气体传热时，能量守恒误差可见并使测试失败。
2. 稳态热平衡与解析解一致。
3. 在固定几何和功率下，改变 `k_mix` 会同时改变稳态温升和瞬态时间常数。
4. 桥路电压必须由同一加热器状态计算，不能另建线性电压事实源。
5. 所有合成硬件参数有单位、范围和 profile ID。

### 12.3 NDIR

1. 零 CO₂ 时 active/reference 比值回到校准基线。
2. 高 CO₂ 区域的饱和和量化平台必须显式报告。
3. 局部气室状态的光程平均与共同气室状态分开审计。
4. 窄带等效模型在冻结网格上与带宽积分参考的最大误差受限。
5. active 与 reference 共享光源漂移，但保留各自独立电子噪声。

### 12.4 任务级

1. 三路 clean equilibrium signal 仍与 A1 冻结事实源一致。
2. 任何设备增强都不能向模型泄露目标组成、真实 `tau`、clean signal 或气室真实状态。
3. 时间模型必须同时超过当前值基线、稳态三标量基线和简单指数平滑基线，才能声称利用了动态结构。
4. 若加入超声衰减或质量量作为输入，必须创建新任务配置，不能静默改变三通道契约。

---

## 13. 反方审查

### 13.1 是否必须仿真完整超声波形？

不必须。二元气体分析和低压掺氢天然气论文表明，理论声速、校准声程和温度补偿足以支持有效浓度测量 [1,3,4]。如果研究目标只是比较三个平衡物性投影，当前标量 ToF 模型可接受。

但 A2-DYN 的目标是证明实时轨迹包含稳态标量之外的可学习信息。此时若超声仍是“理论 ToF + 任意一阶滤波 + 人工噪声”，动态信息来自人为设定的滤波器，而不是超声采集过程。内部波形 surrogate 因而是中等保真动态 benchmark 的必要修订，不等于要求全 FEM。

### 13.2 是否应把 2 Hz 提高到 MHz？

不应。MHz 是内部载波或 ADC 时间尺度，2 Hz 是过程输出时间尺度。文献中的超声系统同样可以在数百 kHz 激励和数十 MHz ADC 下，最终形成约 2 Hz 的过程测量 [5]。正确做法是嵌套时间轴，不是把整条 240 s 序列提升到 MHz。

### 13.3 是否应直接采用论文中的 490 kHz、65 MHz 和 50 次平均？

不应直接冻结。这组参数属于一台具体仪器 [5]；另有研究使用 50 kHz、1.65 MHz 和 16–20 kHz chirp [3,4]。本项目应选择一个明确的合成 hardware profile，并用论文参数作为可行性锚点，而非平均成所谓典型值。

### 13.4 是否应删除所有一阶响应？

不应。CSTR、局部换气和集总热网络都可能产生一阶或多指数响应。问题不在“一阶”形式，而在当前一阶参数没有对应唯一物理状态。修订后仍可使用解析指数更新，但每个时间常数必须只表达一个可命名过程。

### 13.5 完整器件模型会不会让 benchmark 失去可解释性？

全 FEM 会带来过高成本和新的参数不确定性，所以不建议作为 v1 生成器。建议的中等保真 surrogate 保留了解析可审计性：组成决定物性，物性进入一个明确器件状态，器件状态经过固定估计器形成观测。它比直接标量映射多一层，但仍能逐层做 oracle 和单因素测试。

---

## 14. 按研究问题回答

### RQ1 回答

气体超声论文至少存在三种保真度：只算声速或查表的热力学模型；加入声程、流速、波形和时延估计的系统模型；用于器件设计的 KLM 或 FEM 模型。组成测量实验通常不会逐样本运行全 FEM，但会真实执行发射、接收、平均、滤波和 ToF 估计。当前 A2-DYN 只达到第一层。

### RQ2 回答

热导论文通常从焦耳加热和热传递得到加热器温度、电阻和桥路信号；NDIR 论文从 Beer–Lambert 出发，再考虑谱带、光程、参考通道、气室扩散和温漂。当前计划的两路平衡物性合理，但设备层都过度简化。

### RQ3 回答

A2-DYN 是较好的过程与数据分布规划，但不是完整的传感器仿真规划。其主要优点是共同气室、因果任务、分布和审计；主要缺点是 L0 物性直接跳到 L4 数据，并用固定一阶滤波代替 L2 器件和 L3 估计。

### RQ4 回答

v1 必须实现双时间尺度超声 surrogate、热导集总电热状态、NDIR 量程和局部气室语义。全 KLM、全 FEM、完整 QARS 和全部原始高频数据可以延后。修订后仍可保持三路低频输入、约 120–180 MB 数据规模和当前 split 主键不变量。

---

## 15. 最终结论

当前计划应继续推进，但不能直接进入原定 A2-DYN-0 配置冻结。应先把第 4 节动态物理链改成：

$$
\text{inlet}
\rightarrow \text{common chamber}
\rightarrow \text{local transport}
\rightarrow \text{equilibrium property}
\rightarrow \text{device/acquisition}
\rightarrow \text{observation}.
$$

三路对应为：

- 超声：组成 → 声速 → 传播与短波形 → 固定 ToF 估计器 → 低频 ToF；
- 热导：组成 → 混合热导率 → 电热状态 → 电阻和桥路 → 低频电压；
- NDIR：组成 → 光谱吸收 → 光学气室与 active/reference → 低频电压。

从文献证据看，最小但足够可信的修订不是“增加更多随机噪声”，也不是“全面上 FEM”，而是补齐每路最关键的设备状态和估计步骤，并把所有慢响应绑定到具体输运或器件过程。完成这些修订后，A2-DYN 才能回答其核心问题：模型是否真正利用了传感器和气体过程的时序结构，而不是利用人为固定的一阶曲线。

---

## 参考文献

[1] R. Joos et al. An ultrasonic sensor for the analysis of binary gas mixtures. *Sensors and Actuators B: Chemical*, 16:413–419, 1993.

[2] H. Toda et al. High-speed gas concentration measurement using ultrasound. *Sensors and Actuators A: Physical*, 144:1–6, 2008.

[3] R. Bates et al. Implementation of ultrasonic sensing for high resolution measurement of binary gas mixture fractions. *Sensors*, 14:11260–11276, 2014.

[4] J. M. Monsalve et al. Rapid characterisation of mixtures of hydrogen and natural gas by means of ultrasonic time-delay estimation. *Journal of Sensors and Sensor Systems*, 13:179–185, 2024.

[5] J. Sablowski et al. Ultrasonic time-of-flight measurements for inline gas analysis and process monitoring in the phenolic urethane cold box process. *Journal of Sensors and Sensor Systems*, 14:99–109, 2025.

[6] R. Øyerhamn et al. Finite element modeling of ultrasound measurement systems for gas: comparison with experiments in air. *The Journal of the Acoustical Society of America*, 144:2613, 2018.

[7] L. Iglesias Hernandez et al. Gas discrimination by simultaneous sound velocity and attenuation measurements using uncoated capacitive micromachined ultrasonic transducers. *Scientific Reports*, 12:744, 2022.

[8] T. Liu, S. Wang, and M. Zhu. Predicting acoustic relaxation absorption in gas mixtures for extraction of composition relaxation contributions. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 473:20170496, 2017.

[9] M. Castillo et al. KLM model for lossy piezoelectric transducers. *Ultrasonics*, 41:671–679, 2003.

[10] G. Massimino et al. Piezo-micro-ultrasound-transducers for air-coupled arrays: modeling and experiments in the linear and non-linear regimes. *Extreme Mechanics Letters*, 40:100968, 2020.

[11] E. W. Lemmon, M. L. Huber, and M. O. McLinden. NIST Standard Reference Database 23: Reference Fluid Thermodynamic and Transport Properties—REFPROP, Version 9.1. National Institute of Standards and Technology, 2013.

[12] I. H. Bell et al. Pure and pseudo-pure fluid thermophysical property evaluation and the open-source thermophysical property library CoolProp. *Industrial & Engineering Chemistry Research*, 53:2498–2508, 2014.

[13] A. Mahdavifar et al. Transient thermal response of micro-thermal conductivity detector for the identification of gas mixtures: an ultra-fast and low power method. *Microsystems & Nanoengineering*, 1:15025, 2015.

[14] A. S. Cubukcu, D. F. Reyes Romero, and G. A. Urban. A dynamic thermal flow sensor for simultaneous measurement of thermal conductivity and flow velocity of gases. *Sensors and Actuators A: Physical*, 208:73–87, 2014.

[15] J. Mallah and L. G. Occhipinti. Finite element simulation model of metallic thermal conductivity detectors for compact air pollution monitoring devices. *Sensors*, 24:4683, 2024.

[16] J. Hodgkinson et al. Non-dispersive infra-red (NDIR) measurement of carbon dioxide at 4.2 µm in a compact and optically efficient sensor. *Sensors and Actuators B: Chemical*, 186:580–588, 2013.

[17] M. Xu, B. Peng, X. Zhu, and Y. Guo. Multi-gas detection system based on non-dispersive infrared spectral technology. *Sensors*, 22:836, 2022.

[18] L. Zhou, Y. He, Q. Zhang, and L. Zhang. Carbon dioxide sensor module based on NDIR technology. *Micromachines*, 12:845, 2021.

[19] Z. Qiang, X. Wang, and W. Zhang. Real-time correction of gas concentration in nondispersive infrared sensor. *IEEE Transactions on Instrumentation and Measurement*, 72:1–10, 2023.

[20] X. Jia, J. Roels, R. Baets, and G. Roelkens. A miniaturised, fully integrated NDIR CO₂ sensor on-chip. *Sensors*, 21:5347, 2021.

[21] I. E. Gordon et al. The HITRAN2020 molecular spectroscopic database. *Journal of Quantitative Spectroscopy and Radiative Transfer*, 277:107949, 2022.
