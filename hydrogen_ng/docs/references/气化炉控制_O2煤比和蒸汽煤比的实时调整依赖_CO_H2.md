# 气化炉控制：O₂/煤比和蒸汽/煤比的实时调整依赖 CO/H₂

> 来源：原 PDF《气化炉控制：O₂_煤比和蒸汽_煤比的实时调整依赖 CO_H₂.pdf》（同目录），由 AI 生成的研究综述。本 MD 版按 PDF 正文结构提取，保留所有文献内嵌超链接。

**报告日期**：2026 年 6 月 24 日
**标题**：基于在线组分反馈的煤气化炉氧煤比与汽煤比实时优化控制

---

## 摘要

煤气化技术是现代煤化工与清洁能源战略的核心环节，其运行效率与产品质量直接影响整个产业链的经济性与稳定性。气化炉控制系统的核心目标在于维持最佳的反应条件，以最大化碳转化率、冷煤气效率和目标合成气的产率。本报告深入探讨了气化炉控制的一个关键难题：如何通过对合成气中关键组分（CO、H₂、CO₂）的在线实时监测，对氧气/煤比（O₂/煤比）和蒸汽/煤比（汽煤比）进行精确的闭环反馈调节。研究发现，偏离最优操作区间会导致冷煤气效率显著下降、合成气质量剧烈波动，甚至引发操作安全问题。

本报告系统性地分析了影响最优氧煤比和汽煤比的关键因素，包括煤种特性、气化炉类型、运行负荷与压力等。报告详细梳理了用于合成气组分在线分析的先进技术，如非分散红外光谱（NDIR）和气相色谱（GC），并探讨了这些技术固有的测量延迟对控制系统性能的挑战及其缓解策略。在控制策略层面，报告从经典的 PID 控制出发，重点阐述了模型预测控制（MPC）和人工智能（AI）等先进过程控制技术在应对气化过程非线性、多变量、强耦合特性方面的应用优势。报告进一步解构了控制回路的实现机制，即如何将在线组分测量偏差转化为对氧气和蒸汽调节阀的具体操作指令，并讨论了多变量协同控制中温度稳定性与合成气质量之间的权衡管理。最后，报告总结了当前技术面临的挑战，并对未来的发展方向，如高频智能传感器、AI 与机理模型深度融合以及全流程优化等，进行了展望。本报告旨在为煤气化过程的精细化控制和智能化运行提供全面的理论参考和技术洞察。

---

## 1. 引言

### 1.1 研究背景与重要性

在全球能源结构转型和"双碳"目标的大背景下，煤炭的清洁高效利用成为保障国家能源安全和促进工业可持续发展的关键路径。煤气化技术通过在高温高压条件下使煤与气化剂（如氧气、蒸汽、空气等）发生一系列复杂的物理化学反应，将固体煤转化为富含一氧化碳（CO）和氢气（H₂）的合成气（Syngas）。合成气不仅是生产甲醇、合成氨、乙二醇、合成油品等大宗化学品的基础原料，也是燃气轮机联合循环（IGCC）发电和氢能源生产的核心燃料。因此，气化炉作为这一转化过程的核心反应器，其运行的稳定性、效率和经济性至关重要。

然而，气化过程是一个典型的多变量、非线性、强耦合且具有显著时滞的复杂工业过程。任何微小的操作参数波动都可能引起连锁反应，导致炉内温度、压力、产物组分等关键指标发生剧烈变化。其中，两种关键的气化剂——氧气和蒸汽——的注入量与入炉煤量的比例，即氧煤比（O₂/Coal Ratio）和汽煤比（Steam/Coal Ratio），是决定气化反应方向、程度和最终产物分布的最核心的控制变量。一个精确且稳定的氧煤比和汽煤比控制系统，是实现气化炉"安、稳、长、满、优"运行的根本保障。

### 1.2 核心控制挑战

传统的气化炉控制往往依赖于经验模型和开环或简单的反馈控制，难以应对煤种变化、负荷波动等工况扰动。现代精细化生产要求控制系统必须具备实时响应和动态优化的能力。这一需求催生了本报告的核心研究课题：建立一个基于合成气组分（CO/H₂/CO₂）在线反馈的、对氧煤比和汽煤比进行实时闭环调整的先进控制系统。

该控制理念的逻辑基础在于：合成气中 CO、H₂ 和 CO₂ 的相对比例是气化炉内部化学反应平衡和动力学状态最直接、最灵敏的"窗口"。

- **CO/H₂ 比** 直接决定了下游合成产品的选择性和反应效率。
- **CO₂ 浓度** 则反映了煤的氧化程度，过高意味着宝贵的碳资源被过度氧化，导致有效气体（CO+H₂）产率下降，即冷煤气效率降低。

这些组分浓度的实时变化，为我们提供了一个精确的反馈信号，用以判断当前的氧煤比和汽煤比是否处于最优区间。

然而，实现这一闭环控制面临诸多挑战：

1. **最优操作区间的动态性**：最优的氧煤比和汽煤比并非定值，它随着煤种（灰熔点、反应性、水分）、负荷、压力等因素的变化而漂移。
2. **在线测量的延迟与噪声**：从气化炉出口采样到分析仪表给出可靠的组分数据，存在数秒到数分钟的延迟，这严重影响了控制回路的响应速度和稳定性 [[1]][[2]]。
3. **多变量的强耦合性**：氧煤比和汽煤比的调整不仅相互影响，还同时对炉温、合成气组分、碳转化率等多个输出变量产生复杂的影响，必须进行协同控制。
4. **控制算法的复杂性**：简单的线性控制器（如 PID）难以有效处理气化过程的非线性和约束条件，需要更先进的控制策略。

### 1.3 报告结构

为系统性地解决上述挑战，本报告将从以下几个方面展开深入研究：

- **第二章** 将详细分析气化过程的关键控制参数（氧煤比、汽煤比）及其对核心性能指标（冷煤气效率、合成气质量）的影响，并探讨不同工况下最优参数的范围。
- **第三章** 将聚焦于实时反馈控制的"眼睛"——在线监测技术，评估各类分析技术的性能，并深入讨论测量延迟问题及其解决方案。
- **第四章** 将从控制策略的"大脑"层面，系统梳理从经典 PID 到模型预测控制（MPC）和人工智能（AI）等先进算法在气化炉控制中的应用。
- **第五章** 将探讨控制回路的"手脚"——执行机构，分析如何将控制算法的输出转化为对氧气和蒸汽阀门的精确、协同操作。
- **第六章** 将对全文进行总结，并对气化炉智能控制技术的未来发展趋势进行展望。

---

## 2. 气化过程关键控制参数及其优化

气化炉的精细化控制，本质上是对其内部复杂的化学反应网络进行调控。氧煤比和汽煤比作为调控气化剂供给的核心手段，直接决定了燃烧、气化、变换等一系列基元反应的相对速率和平衡点。

### 2.1 氧煤比（O₂/Coal Ratio）的核心作用

氧煤比，通常定义为单位质量（或摩尔）干煤所消耗的纯氧质量（或摩尔），是气化过程中最具决定性的参数。它直接控制着气化炉内的氧化还原气氛和能量平衡。

- **温度控制**：煤中的碳与氧气发生燃烧反应是气化过程主要的放热来源，为吸热的气化反应（碳与水蒸气、二氧化碳的反应）提供所需能量。因此，提高氧煤比会显著提升炉内温度 [[3]][[4]][[5]]。温度是影响反应速率、碳转化率、合成气组分平衡以及炉渣流动性的关键因素。
- **合成气组分**：氧煤比直接影响产物的氧化深度。
    - **氧煤比过低**：氧气供应不足，导致燃烧反应不充分，炉温偏低。这会降低气化反应速率，导致碳转化率下降，合成气中残余的甲烷（CH₄）含量增加，冷煤气效率低下 [[6]]。
    - **氧煤比过高**：氧气供应过量，部分本应用于气化的碳被完全氧化成 CO₂，同时宝贵的中间产物 CO 也会被进一步氧化。这导致有效气体（CO+H₂）的浓度下降，CO₂ 浓度急剧上升，同样造成冷煤气效率的损失 [[7]][[8]][[9]]。过高的氧煤比还会导致局部过热，对耐火材料造成不可逆的损害 [[10]]。

因此，存在一个最佳氧煤比，能够在该点实现炉温、碳转化率、有效气体产率和设备安全等多重目标的平衡。

### 2.2 典型最优氧煤比范围与影响因素

最优氧煤比并非一个普适的常数，它受到多种因素的制约，其范围在不同文献和工况中有所不同，通常在 0.47 到 1.3 之间波动 [[11]][[12]]。

#### 2.2.1 不同气化炉类型的最优值

气化炉的结构设计和操作原理直接决定了其最优氧煤比的范围。

- **壳牌（Shell）气流床气化炉**：这类干粉煤进料的气流床气化炉，操作温度高，反应时间短。研究模型显示，其最优氧煤比约为 0.85 [[13]]。另一项研究则指出，对于短停留时间的反应器，0.8 至 0.9 的氧煤比可以实现较好的碳转化率和冷煤气效率 [[14]]。
- **德士古（Texaco）水煤浆气化炉**：采用水煤浆进料，部分能量需用于蒸发水分。在完全燃烧的化学计量条件下，氧气供应量约为 1 公斤氧气对应 1 公斤干煤，即氧煤比为 1.0。实际操作中，氧碳比通常在 0.9–0.95 范围内 [[15]]。
- **BGL（British Gas/Lurgi）移动床气化炉**：这是一种固定床熔渣气化炉，其操作条件与气流床不同。研究表明，BGL 气化炉的最佳操作条件对应的氧煤比在 0.47–0.49 的较低范围 [[16]]。
- **流化床气化炉**：对于哥伦比亚煤的流化床气化研究显示，为实现高效运行，氧碳比不应超过 0.8，而最优氧煤比为 0.91 [[17]]。

这些数据清晰地表明，不存在统一的最优氧煤比，其设定必须紧密结合具体的气化炉技术类型。

#### 2.2.2 煤种特性的影响

不同煤种的元素组成、灰分特性、水分含量和反应活性差异巨大，直接影响最佳氧气消耗率 [[18]]。

- **煤阶与反应性**：煤阶（rank）越高的煤（如无烟煤），其固定碳含量高，挥发分低，反应活性相对较差，通常需要更高的温度和氧煤比来实现充分的转化。相反，褐煤等低阶煤反应活性高，可在相对温和的条件下气化 [[19]]。研究比较了不同煤种，发现烟煤通常比次烟煤需要更多的氧气 [[20]]，而泥煤的单位氧气消耗则比烟煤低约 15% [[21]]。
- **水分含量**：煤中的水分在进入气化炉后需要吸收大量热量才能蒸发。为了维持必要的反应温度，必须增加氧气的注入量来补偿这部分热量损失。因此，煤的水分含量越高，所需的氧煤比也越高 [[22]][[23]][[24]]。这对于水煤浆气化炉尤其重要。在开发前馈控制策略时，必须将在线监测的煤中水分含量作为一个关键输入变量，用于动态修正氧气注入的设定值 [[25]][[26]]。
- **灰熔点**：对于排渣方式为固态排渣的气化炉（如 Lurgi 干灰炉），炉温必须严格控制在灰熔点以下，以防结渣 [[27]][[28]]。这限制了氧煤比的上限。而对于液态排渣的气化炉（如 Shell 炉、Texaco 炉），则要求炉温必须高于灰熔融温度，以保证炉渣的顺畅排出，这反过来又设定了氧煤比的下限 [[29]]。

#### 2.2.3 运行负荷与压力的影响

- **负荷变化**：气化炉的负荷，即煤处理量，在实际生产中经常需要调整。运行数据显示，随着气化炉负荷从 55.0% 增加到 90.0%，最优的氧煤比呈现出轻微下降的趋势（从 0.797 降至 0.764）[[30]]。这可能是因为在高负荷下，单位体积内的反应物浓度增加，传热传质效率改善，使得维持相同反应温度所需的相对氧气量减少。因此，先进的控制系统必须能够根据负荷的变化，对氧煤比的设定值进行动态调整。
- **操作压力**：压力是影响气化反应平衡的另一个重要参数。研究表明，对于大同煤，在较低压力下，最优氧煤比约为 0.9；而当压力升高到 12–14 个大气压时，最优氧煤比降至约 0.6 [[31]]。这说明压力的升高有利于气化反应向生成 CO 和 H₂ 的方向进行，从而降低了对氧气的需求。

#### 2.2.4 安全与设备约束（热力学极限）

氧煤比的设定还必须考虑设备的安全边界。最主要的约束来自耐火材料的耐温极限。过高的氧煤比会导致局部区域温度急剧升高，超过耐火砖的最高使用温度，引发材料的软化、熔融甚至化学侵蚀，从而显著缩短气化炉的使用寿命，甚至导致灾难性事故 [[32]]。有研究指出，为防止反应器损坏，最高温度限制被设定在 1150 °C [[33]]。德国的固定床气化炉操作经验表明，为防止结渣和保护耐火材料，氧气浓度上限约为 22% [[34]]。这些安全阈值构成了氧煤比控制的硬约束，任何控制策略都必须在此边界内运行。

### 2.3 汽煤比（Steam/Coal Ratio）的协同作用

蒸汽是另一种关键的气化剂，其作用与氧气相辅相成。汽煤比的调节对气化过程同样至关重要。

- **作为气化剂**：蒸汽是水煤气反应（C + H₂O → CO + H₂）的主要反应物，这是生成氢气的主要途径。
- **调节炉温**：水煤气反应是强吸热反应。因此，增加蒸汽注入量可以有效降低气化炉温度，这是一种重要的调温手段，常用于与氧气注入协同控制炉温，特别是在防止灰渣结块和保护耐火材料方面 [[35]][[36]]。
- **促进水煤气变换反应**：蒸汽也参与水煤气变换反应（CO + H₂O ↔ CO₂ + H₂）。提高汽煤比会使该反应平衡向右移动，从而提高 H₂ 的浓度，降低 CO 的浓度，即调整合成气中的 H₂/CO 比。这对于某些下游应用（如合成氨或高氢碳比的合成油）是必需的。
- **协同优化**：氧煤比和汽煤比的设定需要协同考虑。例如，一项研究指出，在氧煤比为 0.8 和汽煤比为 0.4 的条件下，可以同时获得最佳的碳转化率和冷煤气效率 [[37]]。在 BGL 气化炉中，当氧煤比为 0.47 时，蒸汽的分解率达到最大 [[38]]。控制系统必须能够同时调整这两个比率，以在温度稳定和气体成分质量之间找到最佳平衡点 [[39]]。

### 2.4 控制目标：冷煤气效率与合成气质量

对氧煤比和汽煤比进行精细控制的最终目的，是优化两个核心经济技术指标：冷煤气效率和合成气质量。

- **冷煤气效率（Cold Gas Efficiency, CGE）**：这是衡量气化过程能量转化效率的最常用指标，定义为产出的干合成气的总化学能（以其低位热值计算）与输入煤的化学能之比 [[40]]。当氧煤比偏离最优值时，无论是碳转化不完全（氧煤比过低）还是过度氧化生成 CO₂（氧煤比过高），都会导致合成气中有效成分（CO+H₂）的能量占比下降，从而直接降低冷煤气效率 [[41]][[42]]。定量模型和实验数据显示，合成气组成偏差与冷煤气效率损失之间存在明确的负相关关系 [[43]]。例如，燃料含水量增加 10%，平均会导致冷煤气效率下降 3.5% [[44]]，这背后就是因为需要更高的氧煤比来补偿，从而增加了能量损失。
- **合成气质量**：主要指合成气的组分构成，特别是有效气体（CO+H₂）的纯度以及 H₂/CO 的比例。下游化工合成过程对合成气质量有严格要求，任何组分的波动都会影响催化剂的活性、寿命以及最终产品的收率和质量。通过实时在线反馈 CO/H₂/CO₂ 的比例，并据此调整氧煤比和汽煤比，是实现合成气质量稳定控制、满足下游工艺需求的根本途径。

---

## 3. 在线监测：实时反馈控制的基石

要实现基于合成气组分反馈的闭环控制，首先必须拥有一套能够快速、准确、连续地测量高温高压、含尘含焦油的粗煤气中 CO、H₂、CO₂ 浓度的在线分析系统。这个系统是整个先进控制回路的"眼睛"，其性能直接决定了控制的精度和时效性。

### 3.1 合成气组分（CO/H₂/CO₂）在线分析技术

目前工业上已应用多种先进分析技术来实现合成气的在线监测。

- **非分散红外光谱（Non-Dispersive Infrared, NDIR）**：NDIR 是测量 CO 和 CO₂ 最成熟、最常用的技术。它利用了这些异核双原子或多原子分子在红外波段有特征吸收峰的原理。NDIR 分析仪具有结构简单、响应速度快、选择性好、维护量小等优点，非常适合工业现场的连续监测 [[45]][[46]][[47]]。一些先进的分析仪，如 Rosemont 的 CAT 200，可以利用 NDIR 同时测量 CH₄ 和 CO [[48]]。
- **气相色谱（Gas Chromatography, GC）**：气相色谱，特别是微型气相色谱（Micro GC），是能够同时分离和定量分析合成气中几乎所有组分（包括 H₂、CO、CO₂、CH₄、N₂ 等）的强大工具。它通过色谱柱对混合气体进行分离，再用检测器（如热导检测器 TCD）进行检测。GC 的优点是分析精度高、组分覆盖全。然而，其主要缺点是分析周期较长，存在明显的测量延迟（通常为数分钟），这对于需要快速响应的控制回路是一个巨大挑战 [[49]][[50]]。
- **热导检测器（Thermal Conductivity Detector, TCD）**：H₂ 是一种同核双原子分子，没有红外吸收，因此不能用 NDIR 检测。TCD 是测量 H₂ 浓度的常用方法，因为它利用了氢气与其他气体在热导率上的巨大差异。TCD 通常与 GC 联用，也可以作为独立的分析模块 [[51]][[52]]。
- **激光拉曼光谱（Raman Spectroscopy）**：这是一种新兴的在线分析技术，它通过测量分子被激光激发后产生的拉曼散射光谱来识别和定量组分。拉曼光谱的优势在于它可以同时测量多种组分（包括 H₂、N₂ 等红外非活性分子），且样品预处理要求低，可以直接进行原位（in-situ）测量，响应速度快。研究表明，激光拉曼光谱可用于在线分析天然气中的 CO₂ 等组分 [[53]]，其在苛刻的煤气化合成气分析中也展现出巨大潜力。
- **可调谐半导体激光吸收光谱（TDLAS）**：TDLAS 是一种高灵敏度、高选择性、响应极快的在线分析技术，特别适用于测量 CO₂、H₂O 等组分 [[54]]。其抗干扰能力强，能够有效克服背景气体和粉尘的影响。

在实际应用中，往往采用组合式分析仪，例如将 NDIR 用于 CO/CO₂ 的快速测量，TCD 用于 H₂ 的测量，再辅以 GC 进行周期性的全组分校准，以兼顾响应速度和分析的全面性 [[55]]。

### 3.2 测量延迟及其对控制系统的挑战

在线分析系统的测量延迟（Dead Time/Lag）是闭环控制系统中最棘手的问题之一。这种延迟主要来自两个方面：

1. **传输延迟**：从气化炉高压取样点到分析小屋，粗煤气需要经过一段长长的取样管线。气体在这段管线中的传输过程会产生数秒到数十秒的延迟 [[56]]。
2. **分析延迟**：样品进入分析仪后，还需要经过预处理（降温、降压、除尘、除焦油）、分离和检测等步骤。特别是对于气相色谱（GC），其固有的分析周期（保留时间）就是一种显著的延迟，可能长达数分钟 [[57]][[58]]。

总的测量延迟通常在几十秒到几分钟之间 [[59]]。这种延迟对控制系统会产生一系列负面影响：

- **降低控制响应速度**：控制器无法立即获知过程的变化，导致其做出的调节动作总是"慢半拍"。
- **恶化系统稳定性**：严重的延迟会使得控制系统的相位裕度减小，容易引发振荡甚至失控。一个设计用于快速过程的控制器，如果反馈信号延迟过大，其调节作用可能反而加剧了过程的波动。
- **增加超调和波动**：在应对扰动（如负荷变化）时，由于信息滞后，控制器可能会过度调节，导致被控变量（如合成气组分）出现更大的超调和更长的稳定时间。研究表明，将采样时间从 10 秒增加到 30 秒，会显著恶化组分控制性能，甚至导致系统不稳定 [[60]]。

### 3.3 信号延迟的缓解策略

鉴于测量延迟的严重危害，工业界采取了多种策略来缓解其影响。

- **优化取样系统**：通过缩短取样管线长度、提高管内流速、采用加热保温措施防止冷凝等方式，可以最大限度地减少传输延迟 [[61]][[62]]。
- **采用快速分析技术**：优先选用 NDIR、TDLAS、拉曼光谱等响应速度更快的分析技术，而将 GC 作为辅助或校准手段。
- **采用先进控制算法**：
    - **Smith 预估器**：这是一种专门为含纯滞后环节的过程设计的控制算法。它通过建立一个不含延迟的过程模型，在控制器内部提前"预估"出过程的真实响应，从而对延迟进行补偿。
    - **模型预测控制（MPC）**：MPC 在其预测模型中可以显式地包含时滞环节，从而在其未来的优化计算中自然地考虑并补偿延迟的影响 [[63]]。
- **数据驱动的延迟校正**：可以利用历史数据，通过系统辨识等方法建立分析仪信号与真实过程变量之间的动态模型，并在线地对测量信号进行去噪和延迟校正 [[64]]。例如，有专利技术提出了一种根据接收到的调制电磁辐射强度变化来补偿检测器信号延迟的方法 [[65]]。

通过硬件优化和软件补偿相结合的策略，可以在一定程度上克服测量延迟带来的挑战，为实现稳定、高效的闭环控制奠定基础。

---

## 4. 气化炉先进控制策略与算法

拥有了可靠的在线反馈信号后，控制系统的"大脑"——控制策略与算法——便成为决定控制性能的关键。气化炉的复杂特性要求控制系统必须超越传统的单回路控制框架。

### 4.1 从经典 PID 到先进过程控制

#### 4.1.1 PID 控制的应用与局限性

比例-积分-微分（Proportional-Integral-Derivative, PID）控制器因其结构简单、鲁棒性好、易于实现，至今仍在工业过程控制中占据主导地位。在气化炉控制中，PID 可用于底层的流量、压力、温度等单变量控制回路 [[66]][[67]]。例如，一个氧气浓度控制回路可以通过 PID 控制器比较设定值与测量值，计算出误差，并据此调整氧气阀门的开度 [[68]][[69]]。

然而，对于氧煤比和汽煤比这种需要协同调节的核心参数，传统 PID 控制面临巨大挑战：

- **多变量耦合问题**：氧煤比和汽煤比的调整会同时影响炉温和多种气体组分，它们之间存在强耦合。使用多个独立的 PID 控制器分别控制不同的变量，往往会相互干扰，导致系统振荡，难以达到全局最优。
- **非线性问题**：气化过程的动力学特性是高度非线性的。一套在某个工况点（如 80% 负荷）整定好的 PID 参数，在工况发生较大变化（如降到 50% 负荷）后，其控制性能可能会急剧恶化 [[70]]。
- **约束处理能力弱**：生产过程总是伴随着各种约束，如阀门开度的物理限制（0–100%）、氧煤比的安全上限等。标准 PID 算法无法显式地处理这些约束，容易导致饱和积分等问题。

#### 4.1.2 比例增益参数的整定挑战（以煤种反应性为例）

PID 控制性能在很大程度上取决于其参数（比例增益 Kp、积分时间 Ti、微分时间 Td）的整定。然而，为气化炉这样的复杂对象整定 PID 参数极具挑战性。以氧煤比反馈控制回路的比例增益（Kp）整定为例，它需要根据煤种的反应性进行调整。

- **高反应性煤种（如褐煤）**：这种煤对氧气量的变化响应迅速且剧烈。如果比例增益设置过大，微小的组分偏差就会导致氧气阀门的大幅度动作，容易引发炉温和组分的剧烈振荡。因此，需要相对较小的 Kp 以保证系统的稳定性。
- **低反应性煤种（如无烟煤）**：这种煤对氧气变化的响应较为迟钝。如果比例增益设置过小，控制作用会非常微弱，无法及时纠正偏差，导致控制过程缓慢，抗扰动能力差。因此，需要相对较大的 Kp 来提高响应速度。

由于缺乏直接的、公开的关于如何根据煤种反应性定量整定 PID 增益的成熟方法或模型，现场工程师往往依赖经验和试凑法，这不仅耗时耗力，而且难以保证最优性。为解决此问题，自适应 PID 控制方案被提出，它通过在线辨识过程模型，并根据模型变化自动调整 PID 参数，从而适应煤种变化等工况扰动 [[71]]。

### 4.2 模型预测控制（MPC）的应用

模型预测控制（Model Predictive Control, MPC）是 20 世纪 80 年代发展起来的一种先进过程控制技术，被认为是解决复杂工业过程控制难题的有效工具 [[72]][[73]]。其核心思想是：在每个控制时刻，基于一个描述过程动态行为的模型，在线反复求解一个有限时域的优化问题，以预测未来一段时间内（预测时域）系统的行为，并计算出一系列最优的控制动作序列。然后，将此序列的第一个控制动作施加于被控对象，并在下一个时刻重复此过程（滚动优化）。

MPC 在气化炉控制中具有天然的优势：

- **处理多变量耦合**：MPC 可以自然地处理多输入多输出（MIMO）系统。它可以将氧气阀和蒸汽阀的开度作为输入，将 CO/H₂/CO₂ 浓度、炉温等作为输出，建立一个统一的多变量预测模型，从而在优化计算中自动协调各个输入变量，实现全局最优控制 [[74]]。
- **处理约束**：MPC 可以在其优化问题中显式地包含各种等式和不等式约束，如输入约束（阀门开度范围）、输出约束（炉温安全上下限）、变化率约束等。这使得控制系统总能保证操作的安全性和可行性。
- **处理时滞**：如前所述，MPC 的预测模型可以很方便地将测量延迟包含进去，从而实现对时滞的有效补偿 [[75]]。
- **优化能力**：MPC 框架内嵌了一个优化求解器，可以直接以经济指标（如最大化冷煤气效率）或性能指标（如最小化组分偏差）作为优化目标，实现真正意义上的优化控制。

针对气化炉的非线性特性，发展出了多种 MPC 策略，如广义预测控制（GPC）[[76]][[77]] 以及在整个操作空间内建立多个线性模型，并根据当前工况进行切换或加权的多模型预测控制（MMPC）[[78]][[79]]。这些策略在循环流化床锅炉的氧含量控制等类似应用中已证明了其优越性 [[80]]。

#### 4.2.1 关键参数的优化：预测时域与采样间隔

MPC 的性能同样依赖于其关键参数的整定，主要是预测时域（Prediction Horizon, P）和采样间隔（Sampling Interval, T）。

- **采样间隔（T）**：应足够小，以捕捉过程的关键动态特性，通常应小于或远小于被控过程的时间常数 [[81]][[82]]。但过小的采样间隔会急剧增加计算负担，因为 MPC 的在线优化计算必须在每个采样间隔内完成 [[83]]。对于气化炉这种相对缓慢的过程，采样间隔可能在数秒到几十秒的量级。
- **预测时域（P）**：定义了控制器"看得多远"。P 应足够长，以覆盖过程的主要动态响应过程，通常应大于过程的整定时间 [[84]][[85]]。较长的 P 可以提高控制性能和对扰动的预见性，但同样会显著增加优化问题的维度和计算量 [[86]]。

这些参数的优化选择是一个在控制性能和计算可行性之间的权衡，通常需要通过仿真和现场实验来确定。

### 4.3 人工智能（AI）与机器学习的集成

随着计算能力的提升和数据量的积累，人工智能（AI），特别是机器学习，为气化炉这种难以精确机理建模的复杂过程控制开辟了新途径。

- **AI 用于建模与预测**：可以利用神经网络（如 RBF 神经网络）、支持向量机等 AI 方法，直接从大量的历史运行数据中学习，建立气化过程的非线性数据驱动模型。这种模型可以用于预测合成气组分 [[87]]，或直接作为 MPC 中的预测模型，即所谓的"AI-MPC" [[88]]。
- **AI 用于优化与控制**：强化学习（Reinforcement Learning）等 AI 技术可以直接学习一个从过程状态到最优控制动作的映射（即控制策略），而无需显式的过程模型。
- **AI 的潜力**：AI 在处理不确定性、挖掘数据中隐含的复杂关联、实现自适应优化方面具有独特优势 [[89]][[90]]。研究表明，AI 在气化系统的性能建模、预测和优化方面展现出显著的潜力 [[91]]。

目前，关于 AI 集成控制是否全面优于传统 MPC 的直接对比研究还较少。一个可能的发展方向是，将 AI 的数据驱动建模能力与 MPC 基于模型的优化控制框架相结合，形成混合智能控制系统，取长补短，实现更优的控制性能。

### 4.4 串级控制与前馈控制策略

- **串级控制（Cascade Control）**：这是一种通过引入辅助测量变量来提高主变量控制性能的有效策略。在气化炉温度控制中，可以构建一个串级结构：主控制器（外环）以炉温为被控量，其输出是氧气流量的设定值；副控制器（内环）以氧气流量为被控量，直接控制氧气阀门 [[92]]。由于流量回路响应远快于温度回路，这种结构可以快速抑制氧气压力波动等对流量的扰动，从而提高温度控制的稳定性和精度。有研究提出了将氧/油比、汽/油比与温度进行串级控制的系统方案 [[93]]。
- **前馈控制（Feedforward Control）**：当系统中存在可测量的主要扰动时，前馈控制可以提前做出补偿动作，以减小扰动对输出的影响。对于气化炉，煤质（特别是水分含量）的变化是一个主要扰动。可以建立一个前馈通道：在线测量入炉煤的水分含量，根据水分含量与所需氧煤比之间的关系模型，直接计算出一个补偿性的氧气流量调整量，叠加到反馈控制器的输出上 [[94]]。这样，在反馈控制器尚未检测到由水分变化引起的组分偏差之前，前馈作用就已经开始进行补偿，从而大大提高了系统的抗扰动性能。

---

## 5. 控制回路的实现与执行

控制策略最终需要通过具体的算法和执行机构（阀门）来实现对物理过程的操控。本节将探讨从测量偏差到阀门指令的转换过程，以及多阀门的协同调节问题。

### 5.1 从测量偏差到阀门指令的转换

控制回路的核心任务是根据被控变量的设定值（Setpoint, SP）和测量值（Process Variable, PV）之间的偏差（Error, e = SP − PV），计算出控制器的输出（Controller Output, OP），并最终将其转换为阀门的位置指令。

**控制算法的数学实现**：

- **PID 算法**：控制器输出是偏差的比例、积分和微分项的加权和。其离散化的数学公式为：

    ```
    OP(k) = OP(k-1) + Kp · [ (e(k) − e(k-1))
                            + (T/Ti) · e(k)
                            + (Td/T) · (e(k) − 2·e(k-1) + e(k-2)) ]
    ```

    这个 OP 值（通常是 0–100%）被发送给阀门定位器，驱动阀门动作。

- **MPC 算法**：MPC 的计算过程更为复杂。它通过求解一个二次规划（QP）或非线性规划（NLP）优化问题，得到未来一系列的阀门开度变化量（Δu）。然后，将第一个变化量施加于当前阀门开度：u(k) = u(k-1) + Δu(k) [[95]]。

**传递函数在系统建模中的角色**：在进行控制系统设计和仿真时，通常需要用数学模型来描述从阀门开度变化（输入）到合成气组分变化（输出）之间的动态关系。传递函数（Transfer Function）是描述线性时不变系统动态特性的经典工具 [[96]][[97]]。虽然气化炉是高度非线性的，但在某个特定的工作点附近，可以将其线性化，得到一个近似的传递函数模型。这个模型对于 PID 参数的初步整定、控制回路的稳定性分析以及构建 MPC 的线性预测模型至关重要。然而，目前公开文献中很少有专门针对"合成气组分偏差到阀门执行信号"的标准化传递函数，这通常需要通过现场的阶跃响应实验来辨识获得 [[98]]。

### 5.2 氧气与蒸汽阀门的协同调节

气化炉控制的复杂性在于，氧气和蒸汽的调节需要紧密协同，以在多个相互冲突的目标之间寻求平衡。

- **管理温度与组分质量的权衡**：这是一个典型的控制权衡问题。例如，当检测到碳转化率不足（表现为 CO/CO₂ 比值偏低）时，控制器需要增加氧煤比。但增加氧气会使炉温升高。如果此时炉温已接近安全上限，就不能简单地开大氧气阀。协同控制系统此时可能会同时适度增加蒸汽注入量，利用水煤气反应的吸热效应来"压住"温度，从而在提高转化率的同时维持温度稳定 [[99]]。
- **比例控制与多变量协调策略**：
    - **比例控制（Ratio Control）**：在许多工况下，氧气和蒸汽之间需要维持一个特定的比例。控制系统可以采用比例控制策略，将煤流量作为主变量，氧气和蒸汽流量的设定值则根据预设的氧煤比和汽煤比自动计算得出，从而保证在负荷变化时，三者能够按比例联动 [[100]]。
    - **解耦控制**：对于强耦合的多变量系统，可以设计解耦控制器。解耦控制通过引入补偿环节，试图抵消变量之间的耦合效应，从而将一个 MIMO 系统近似转化为多个独立的 SISO（单输入单输出）系统，之后便可应用单变量控制技术。
    - **MPC 的内在协调**：如前所述，MPC 是解决这类问题的最有力工具。它在一个统一的优化框架内，同时考虑了所有输入（氧气、蒸汽）和输出（温度、CO、H₂、CO₂）以及它们之间的相互影响，自然地实现了协同调节，找到了满足所有约束条件下的最优操作路径。

### 5.3 控制回路的动态性能要求

为了有效地应对扰动和负荷变化，控制回路必须满足一定的动态性能要求。

- **响应时间与更新速率**：
    - **响应时间**：指系统在受到扰动或设定值改变后，恢复到新稳态所需的时间。对于稳态操作中的组分偏差修正，自动化系统需要足够快的响应时间。虽然没有统一的"标准"，但控制回路的响应时间应远小于过程的开环响应时间，才能实现有效控制。氧气传感器的典型响应时间为 5 秒 [[101]]，而一些智能控制系统可以实现 3–4 秒的稳定时间 [[102]]。整个控制回路的响应时间则取决于传感器、控制器和执行器的综合性能。
    - **采样频率与控制更新率**：指控制器读取测量值和计算并发送控制指令的频率。根据奈奎斯特采样定理，采样频率必须至少是系统有效带宽的两倍。对于气化炉的组分控制，由于存在测量延迟和过程惯性，更新率可能不需要非常高，但过低的更新率（过长的采样时间）会严重损害控制性能 [[103]]。典型的采样和控制周期可能在秒级到十秒级。
- **负荷变动期间的稳定性维持**：在化工生产中，气化炉经常需要进行升降负荷操作（Load Ramps）。在负荷快速变化期间，系统的动态特性会发生改变，这对控制系统是一个严峻的考验 [[104]]。一个优秀的控制系统，如设计良好的 MPC，能够在负荷爬坡过程中，动态调整氧煤比和汽煤比，不仅要跟踪负荷变化，还要始终将炉温和合成气组分维持在允许的范围内，确保过程的平稳过渡。

---

## 6. 结论与未来展望

### 6.1 核心研究发现总结

本研究报告系统地阐述了基于在线合成气组分反馈对煤气化炉氧煤比和汽煤比进行实时优化控制的理论、技术与挑战。核心结论如下：

1. **参数优化的关键性**：氧煤比和汽煤比是决定气化炉效率、产品质量和安全运行的核心控制变量。其最优值是一个受煤种、炉型、负荷、压力等多因素影响的动态区间，而非固定值。偏离此区间将直接导致冷煤气效率下降和合成气质量波动。
2. **在线监测是前提**：快速、准确的在线分析技术（如 NDIR、GC、拉曼光谱等）是实现闭环反馈控制的物理基础。然而，测量延迟是该技术路线上的主要障碍，必须通过优化取样系统和采用具有时滞补偿功能的先进控制算法来克服。
3. **先进控制是核心**：面对气化过程的多变量、非线性、强耦合和约束特性，传统的 PID 控制能力有限。模型预测控制（MPC）以其处理多变量、约束和时滞的内在优势，成为当前最适合气化炉优化控制的先进策略。同时，人工智能（AI）在非线性建模和自适应优化方面展现出巨大潜力。
4. **协同控制是保障**：必须对氧气和蒸汽的注入进行协同调节，通过比例控制、串级控制、前馈补偿以及 MPC 等多变量优化策略，有效管理炉温稳定性和合成气质量之间的复杂权衡。

### 6.2 当前技术的挑战与局限

尽管气化炉控制技术已取得长足进步，但仍面临诸多挑战：

- **模型失配问题**：MPC 等基于模型的控制策略，其性能高度依赖于模型的准确性。气化过程机理复杂，难以建立高精度的全工况范围机理模型；而数据驱动模型则存在泛化能力不足、数据质量要求高等问题。
- **传感器技术的瓶颈**：尽管分析技术不断发展，但在高温、高压、强腐蚀、高粉尘的恶劣工况下，实现长期、稳定、免维护、低延迟的全组分在线测量，仍然是一个技术难题。
- **经济性与性能的权衡**：实施先进控制系统（如 MPC）需要较高的初始投资（软件、硬件、建模、人力）和持续的维护成本，企业需要在预期的效益提升与成本投入之间做出决策。

### 6.3 未来发展方向

展望未来，气化炉的控制技术将朝着更智能、更精细、更一体化的方向发展：

- **智能传感与软测量技术**：开发响应更快、抗干扰能力更强、集成度更高的原位（in-situ）智能传感器。同时，大力发展软测量（Soft Sensor）技术，利用易于测量的变量（如温度、压力、流量）和 AI 模型，来实时估算难以直接测量的关键变量（如碳转化率、合成气热值），从而为控制系统提供更丰富、更及时的反馈信息。
- **混合智能控制**：将机理模型、数据驱动模型和专家知识进行深度融合。利用机理模型保证控制的基本规律和边界，利用 AI 模型补偿机理模型的未知动态和非线性，形成"机理+AI"的混合预测模型，并将其嵌入 MPC 框架，有望实现控制性能的跃升。
- **全流程一体化优化**：打破气化炉的"孤岛"式控制，将控制范围向上游的磨煤、制浆系统和下游的合成气净化、化工合成单元延伸。通过建立覆盖整个工厂的动态模型和优化系统，实现从煤炭进厂到最终产品产出的全流程实时优化（Real-Time Optimization, RTO），最大化整体经济效益。
- **数字孪生与预测性维护**：构建气化炉的高保真数字孪生（Digital Twin）体，不仅用于控制策略的离线仿真与测试，还可用于操作人员培训、异常工况预警和设备健康状态的预测性维护，从而将控制从被动的调节提升到主动的预防和预测层面。

综上所述，随着传感技术、控制理论和人工智能的不断进步，基于在线反馈的实时优化控制必将推动煤气化技术向着更加高效、清洁、智能和经济的方向迈进，为全球能源化工行业的可持续发展提供坚实的技术支撑。

---

## 文献来源

> 说明：原 PDF 的引用以 `[N]` 形式出现于正文，并在末尾以"引用组—文献条目"的方式列出（一条文献对应多个正文引用编号）。本节按 PDF 原顺序逐条罗列，保留 PDF 中可点击的 DOI / 检索 / 全文 PDF 链接。

### 主要文献（直接引用）

1. [4,14,56,323] One-Dimensional Modeling of an Entrained Coal Gasification Process Using Kinetic Parameters — <https://doi.org/10.3390/EN9020099>
2. [5,16,317] Gasification characteristics in an entrained flow coal gasifier — <https://doi.org/10.22634/KSME-B.1997.21.12.1690>
3. [6,119,290,293,295,315] Effect of oxygen-coal ratio on gasification process in entrained-flow gasifier — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Effect%20of%20oxygen-coal%20ratio%20on%20gasification%20process%20in%20entrained-flow%20gasifier&sort=relevance)
4. [13,316] A Study on the Improvement of ASU Process Control Method for Stabilization of Oxygen Supply Pressure in IGCC Gasifier — <https://doi.org/10.47116/apjcri.2023.08.03>
5. [21,32,35] 预测型 PID 算法在气化炉控制系统中的研究 — [万方数据](https://s.wanfangdata.com.cn/paper?q=%E9%A2%84%E6%B5%8B%E5%9E%8BPID%E7%AE%97%E6%B3%95%E5%9C%A8%E6%B0%94%E5%8C%96%E7%82%89%E6%8E%A7%E5%88%B6%E7%B3%BB%E7%BB%9F%E4%B8%AD%E7%9A%84%E7%A0%94%E7%A9%B6)
6. [22] Explicit stochastic predictive control of combustion plants based on Gaussian process models — <https://doi.org/10.1016/j.automatica.2008.04.002>
7. [24] Hierarchical predictive control. Application to reheating furnaces in the steel industry. — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Hierarchical%20predictive%20control.%20Application%20to%20reheating%20furnaces%20in%20the%20steel%20industry.&sort=relevance)
8. [25,30] Design and Implementation of Hybrid Modeling and PFC for Oxygen Content Regulation in a Coke Furnace — <https://doi.org/10.1109/TII.2018.2815717>
9. [29,36,365] ALSTOM 气化炉的多模型预测控制 — [万方数据](https://s.wanfangdata.com.cn/paper?q=ALSTOM%E6%B0%94%E5%8C%96%E7%82%89%E7%9A%84%E5%A4%9A%E6%A8%A1%E5%9E%8B%E9%A2%84%E6%B5%8B%E6%8E%A7%E5%88%B6)
10. [38,45,47,50,331,334,337,341] NOVEL GAS CLEANING / CONDITIONING FOR INTEGRATED GASIFICATION COMBINED CYCLE — <https://doi.org/10.2172/881044>
11. [39,43,48,332,336] Kinetic modeling and sensitivity analysis of plasma-assisted combustion — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Kinetic%20modeling%20and%20sensitivity%20analysis%20of%20plasma-assisted%20combustion&sort=relevance)
12. [40,340] 大尺度区域 CO₂ 和 H₂O 的激光在线检测技术 — [维普期刊](https://www.cqvip.com/search?k=%E5%A4%A7%E5%B0%BA%E5%BA%A6%E5%8C%BA%E5%9F%9FCO2%E5%92%8CH2O%E7%9A%84%E6%BF%80%E5%85%89%E5%9C%A8%E7%BA%BF%E6%A3%80%E6%B5%8B%E6%8A%80%E6%9C%AF)
13. [41,333] 红外线二氧化碳气体分析器在碳酸盐生产中应用 — [维普期刊](https://www.cqvip.com/search?k=%E7%BA%A2%E5%A4%96%E7%BA%BF%E4%BA%8C%E6%B0%A7%E5%8C%96%E7%A2%B3%E6%B0%94%E4%BD%93%E5%88%86%E6%9E%90%E5%99%A8%E5%9C%A8%E7%A2%B3%E9%85%B8%E7%9B%90%E7%94%9F%E4%BA%A7%E4%B8%AD%E5%BA%94%E7%94%A8)
14. [42,52,335] Catalytic methanation of CO₂ in biogas: experimental results from a reactor at full scale — <https://doi.org/10.1039/c9re00351g>
15. [44] CRDS 和 GC 方法在线监测大气 CH₄ 和 CO 的结果对比（附视频）— [万方数据](https://s.wanfangdata.com.cn/paper?q=CRDS%E5%92%8CGC%E6%96%B9%E6%B3%95%E5%9C%A8%E7%BA%BF%E7%9B%91%E6%B5%8B%E5%A4%A7%E6%B0%94CH4%E5%92%8CCO%E7%9A%84%E7%BB%93%E6%9E%9C%E5%AF%B9%E6%AF%94)
16. [46,338] Advanced Hydrogen Transport Membrane for Coal Gasification — <https://doi.org/10.2172/1238351>
17. [49,339] 激光拉曼光谱法天然气原料气中 CO₂ 和 H₂S 在线分析研究及应用 — [万方数据](https://s.wanfangdata.com.cn/paper?q=%E6%BF%80%E5%85%89%E6%8B%89%E6%9B%BC%E5%85%89%E8%B0%B1%E6%B3%95%E5%A4%A9%E7%84%B6%E6%B0%94%E5%8E%9F%E6%96%99%E6%B0%94%E4%B8%ADCO2%E5%92%8CH2S%E5%9C%A8%E7%BA%BF%E5%88%86%E6%9E%90%E7%A0%94%E7%A9%B6%E5%8F%8A%E5%BA%94%E7%94%A8)
18. [51] Development of the Advanced Aqueous Ammonia Based Post Combustion Capture Technology Progress Report — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Development%20of%20the%20Advanced%20Aqueous%20Ammonia%20Based%20Post%20Combustion%20Capture%20Technology%20Progress%20Report&sort=relevance)
19. [54,63] 주요 운전 변수에 따른 석탄의 가스화 성능 예측 (主要运行变量对煤气化性能的预测) — [Semantic Scholar search](https://www.semanticscholar.org/search?q=%EC%A3%BC%EC%9A%94%20%EC%9A%B4%EC%A0%84%20%EB%B3%80%EC%88%98%EC%97%90%20%EB%94%B0%EB%A5%B8%20%EC%84%9D%ED%83%84%EC%9D%98%20%EA%B0%80%EC%8A%A4%ED%99%94%20%EC%84%B1%EB%8A%A5%20%EC%98%88%EC%B8%A1&sort=relevance)
20. [58] Relevance of the coal rank on the performance of the in situ gasification chemical-looping combustion — <https://doi.org/10.1016/J.CEJ.2012.04.052>
21. [59] Release of inorganic trace elements from high-temperature gasification of coal — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Release%20of%20inorganic%20trace%20elements%20from%20high-temperature%20gasification%20of%20coal&sort=relevance)
22. [61] Chemistry of coal utilization — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Chemistry%20of%20coal%20utilization&sort=relevance)
23. [62] Numerical simulation of coal gasification in entrained flow coal gasifier — <https://doi.org/10.1016/J.FUEL.2006.02.002>
24. [65] Hazardous waste treatment method and apparatus — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Hazardous%20waste%20treatment%20method%20and%20apparatus&sort=relevance)
25. [66] Local learning-based model-free adaptive predictive control for adjustment of oxygen concentration in syngas manufacturing industry — <https://doi.org/10.1049/IET-CTA.2015.0835>
26. [73,80,384] Development of Computational Approaches for Simulation and Advanced Controls for Hybrid Combustion-Gasification Chemical Looping — <https://doi.org/10.2172/1132632>
27. [74] Modelling and Control of Dry and Wet Gas Compressors — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Modelling%20and%20Control%20of%20Dry%20and%20Wet%20Gas%20Compressors&sort=relevance)
28. [75] Instrument Engineers' Handbook, Fourth Edition, Volume Two: Process Control and Optimization — <https://doi.org/10.1201/9781420064001>
29. [76] Plantwide control design procedure — <https://doi.org/10.1002/AIC.690431205>
30. [77] Multivariable Feedback Control: Analysis and Design — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Multivariable%20Feedback%20Control%3A%20Analysis%20and%20Design&sort=relevance)
31. [78] Plantwide control: the search for the self-optimizing control structure — <https://doi.org/10.1016/s0959-1524(00)00023-8>
32. [79,81,390] Flexible operation, optimisation and stabilising control of a quench cooled ammonia reactor for power-to-ammonia — <https://doi.org/10.1016/j.compchemeng.2023.108316>
33. [83] Snowball effects in reactor/separator processes with recycle — <https://doi.org/10.1021/IE00026A019>
34. [84,92,352,381] A System for Dissolved oxygen Control in Industrial aeration tank — <https://doi.org/10.5755/j01.itc.41.1.921>
35. [86,88,96,354] On-line, adaptive, optimal control of a high-density, fed-batch fermentation of Streptomyces C5 — [Semantic Scholar search](https://www.semanticscholar.org/search?q=On-line%2C%20adaptive%2C%20optimal%20control%20of%20a%20high-density%2C%20fed-batch%20fermentation%20of%20Streptomyces%20C5%20/&sort=relevance)
36. [87,356] Improved combustion efficiency of a H₂/O₂ steam generator for spinning reserve application — <https://doi.org/10.1016/S0360-3199(97)00034-7>
37. [90,355] A microcontroller-based approach to control oxygen and establish flexible acclimation regimes — <https://doi.org/10.1101/2024.06.03.597140>
38. [94] Extending DEVS-Scheme for control of an oxygen production test bed — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Extending%20DEVS-Scheme%20for%20control%20of%20an%20oxygen%20production%20test%20bed&sort=relevance)
39. [98,342,343,349,361] Lecture notes for the course Advanced Control of Industrial Processes — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Lecture%20notes%20for%20the%20course%20Advanced%20Control%20of%20Industrial%20Processes&sort=relevance)
40. [99,287] Pulverized coal combustion boiler efficient control — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Pulverized%20coal%20combustion%20boiler%20efficient%20control&sort=relevance)
41. [100,288,345,347] ON LINE GAS ANALYSIS — <https://doi.org/10.2172/4600874>
42. [101] ArtEMon: Artificial Intelligence and Internet of Things Powered Greenhouse Gas Sensing for Real-Time Emissions Monitoring — <https://doi.org/10.3390/s23187971>
43. [102,350] Industrial Control under Non-Ideal Measurements: Data-Based Signal Processing as an Alternative to Controller Retuning — <https://doi.org/10.3390/s21041237>
44. [103,351] System and method of compensating for system delay in analyte analyzation — [Semantic Scholar search](https://www.semanticscholar.org/search?q=System%20and%20method%20of%20compensating%20for%20system%20delay%20in%20analyte%20analyzation&sort=relevance)
45. [104,344] 缩短在线气相色谱仪分析滞后时间采取的措施和效益分析 — [万方数据](https://s.wanfangdata.com.cn/paper?q=%E7%BC%A9%E7%9F%AD%E5%9C%A8%E7%BA%BF%E6%B0%94%E7%9B%B8%E8%89%B2%E8%B0%B1%E4%BB%AA%E5%88%86%E6%9E%90%E6%BB%9E%E5%90%8E%E6%97%B6%E9%97%B4%E9%87%87%E5%8F%96%E7%9A%84%E6%8E%AA%E6%96%BD%E5%92%8C%E6%95%88%E7%9B%8A%E5%88%86%E6%9E%90)
46. [105,348] 改进在线取样方式及样品预处理的使用 — [维普期刊](https://www.cqvip.com/search?k=%E6%94%B9%E8%BF%9B%E5%9C%A8%E7%BA%BF%E5%8F%96%E6%A0%B7%E6%96%B9%E5%BC%8F%E5%8F%8A%E6%A0%B7%E5%93%81%E9%A2%84%E5%A4%84%E7%90%86%E7%9A%84%E4%BD%BF%E7%94%A8)
47. [107] Automated on-line analysis for controlling industrial processes — <https://doi.org/10.1351/PAC197749101609>
48. [118,300] A review on the effect of feed oxygen, water concentration, temperature and pressure on gasification process — [Semantic Scholar search](https://www.semanticscholar.org/search?q=A%20review%20on%20the%20effect%20of%20feed%20oxygen%2C%20water%20concentration%2C%20temperature%20and%20pressure%20on%20gasification%20proce&sort=relevance)
49. [120,126,129,382] Application of a New Dataset Selection Procedure for the Prediction of the Syngas Composition of a Gasification Plant — <https://doi.org/10.3182/20120710-4-SG-2026.00065>
50. [121,123,383] An Acceleration Switching Valve Control System With Machine Tool Application — <https://doi.org/10.2172/4620456>
51. [122,124,127] A High-Fidelity Real-Time Simulation Code of Gas Turbine Dynamics for Control Applications — <https://doi.org/10.1115/GT2002-30039>
52. [125] Invariant Method of Load Independent Pressure Control in Steam Boiler — <https://doi.org/10.2478/v10314-012-0001-4>
53. [130,133,158,367] Predictive Monitoring of Basic Oxygen Steel Refining — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Predictive%20Monitoring%20of%20Basic%20Oxygen%20Steel%20Refining&sort=relevance)
54. [135,143,156,157,165,369] Model-Based Predictive Control of a Solar Hybrid Thermochemical Reactor for High-Temperature Steam Gasification of Biomass — <https://doi.org/10.3390/cleantechnol5010018>
55. [169,175,320] Thermodynamic equilibrium for Wyoming Coal: new calculations — <https://doi.org/10.2172/7292260>
56. [171,179] A Computational Workbench Environment for Virtual Power Plant Simulation, Quarterly Progress Report (Oct 1, 2003 – Dec 31, 2003) — [Semantic Scholar search](https://www.semanticscholar.org/search?q=A%20Computational%20Workbench%20Environment%20for%20Virtual%20Power%20Plant%20Simulation%20Quarterly%20Progress%20Report%20Reporting%20Period%20Start%20Date%20%3A%20October%201%20%2C%202003%20Reporting%20Period%20End%20Date%20%3A%20December%2031%20%2C%202003&sort=relevance)
57. [173] Net carbon-di-oxide conversion and other novel features of packed bed biomass gasification with O₂/CO₂ mixtures — <https://doi.org/10.1016/J.FUEL.2019.01.171>
58. [176,182,313,321] Phase I: the pipeline gas demonstration plant. Technical support program report — <https://doi.org/10.2172/6226293>
59. [180] Design eines hocheffizienten Festoxid-Brennstoffzellensystems mit integrierter Schutzgaserzeugung — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Design%20eines%20hocheffizienten%20Festoxid-Brennstoffzellensystems%20mit%20integrierter%20Schutzgaserzeugung&sort=relevance)
60. [183] A survey of industrial model predictive control technology — <https://doi.org/10.1016/S0967-0661(02)00186-7>
61. [184] Lower Order Modeling and Control of Alstom Fluidized Bed Gasifier — <https://doi.org/10.5772/48674>
62. [186] Process model based control of a fluidized bed gasifier: A comparison of two strategies — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Process%20model%20based%20control%20of%20a%20fluidized%20bed%20gasifier%3A%20A%20comparison%20of%20two%20strategies&sort=relevance)
63. [187] Utilisation of non-linear modelling methods in flue-gas oxygen-content control — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Utilisation%20of%20non-linear%20modelling%20methods%20in%20flue-gas%20oxygen-content%20control&sort=relevance)
64. [190,194,375] Artificial Intelligence-Based Modeling and Control of Fluidized Bed Combustion — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Artificial%20Intelligence-Based%20Modeling%20and%20Control%20of%20Fluidized%20Bed%20Combustion&sort=relevance)
65. [191,193,376,377] Biomass Gasification and Applied Intelligent Retrieval in Modeling — <https://doi.org/10.3390/en16186524>
66. [195,198,378] Contribution to the study and design of advanced controllers: application to smelting furnaces — <https://doi.org/10.5821/dissertation-2117-95238>
67. [196] 29 Heat Exchanger Control and Optimization — [Semantic Scholar search](https://www.semanticscholar.org/search?q=29%20Heat%20Exchanger%20Control%20and%20Optimization&sort=relevance)
68. [197,199,379] 气化炉氧/油比值、汽/油比值与温度串级控制系统 — [万方数据](https://s.wanfangdata.com.cn/paper?q=%E6%B0%94%E5%8C%96%E7%82%89%E6%B0%A7/%E6%B2%B9%E6%AF%94%E5%80%BC%E3%80%81%E6%B1%BD/%E6%B2%B9%E6%AF%94%E5%80%BC%E4%B8%8E%E6%B8%A9%E5%BA%A6%E4%B8%B2%E7%BA%A7%E6%8E%A7%E5%88%B6%E7%B3%BB%E7%BB%9F)
69. [201] Industrial oxygen plants: a technology overview for users of coal gasification-combined-cycle systems — <https://doi.org/10.2172/6632782>
70. [202] Steam Reforming, 6-in. Bench-Scale Design and Testing Project — Technical and Functional Requirements Description — <https://doi.org/10.2172/910657>
71. [204] JV Task-129 Advanced Conversion Test — Bulgarian Lignite — <https://doi.org/10.2172/990807>
72. [205,207,308,311] Subtask 3.16 — Low-Cost Coal-Water Fuel for Entrained-Flow Gasification — <https://doi.org/10.2172/3812>
73. [206] Improving heat capture for power generation in coal gasification plants — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Improving%20heat%20capture%20for%20power%20generation%20in%20coal%20gasification%20plants&sort=relevance)
74. [209,310] Performance Analysis on an Entrained-Flow Gasifier by Coal Moisture — <https://doi.org/10.1002/CEAT.201700129>
75. [215] An Applied Numerical Simulation of Entrained-Flow Coal Gasification with Improved Sub-models — [Semantic Scholar search](https://www.semanticscholar.org/search?q=An%20Applied%20Numerical%20Simulation%20of%20Entrained-Flow%20Coal%20Gasification%20with%20Improved%20Sub-models&sort=relevance)
76. [217] Numerical Analysis of Gasification Performance via Finite-Rate Model in a Cross-Type Two-Stage Gasifier — <https://doi.org/10.1016/J.IJHEATMASSTRANSFER.2012.10.026>
77. [218,219] Design, Modelling and Optimization of a Heat Integrated Coal Gasification Process — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Design%2C%20Modelling%20and%20Optimization%20of%20a%20Heat%20Integrated%20Coal%20Gasification%20Process&sort=relevance)
78. [220,307] On the Thermal and Kinetic Performance of a Coal-CO₂ Slurry-fed Gasifier: Optimization of CO₂ and H₂O flow using CO₂ skimming and steam injection — [Semantic Scholar search](https://www.semanticscholar.org/search?q=On%20the%20Thermal%20and%20Kinetic%20Performance%20of%20a%20Coal-CO%202%20Slurry-fed%20Gasifier%20%3A%20Optimization%20of%20CO%202%20and%20H%202%20O%20flow%20using%20CO%202%20skimming%20and%20steam%20injection&sort=relevance)
79. [221,305,306] Countercombustion coal gasification — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Countercombustion%20coal%20gasification&sort=relevance)
80. [222,304] Effect of coal type on the gasification rate — <https://doi.org/10.3775/JIE.65.10_798>
81. [224] Gasifier study for Mobil coal to gasoline processes. Final report — <https://doi.org/10.2172/6120614>
82. [225,228,232,327] Mathematical modelling of a small biomass gasifier for synthesis gas production — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Mathematical%20modelling%20of%20a%20small%20biomass%20gasifier%20for%20synthesis%20gas%20production&sort=relevance)
83. [226,229,328,330] Biomass Gasification Thermodynamic Model Including Tar and Char — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Biomass%20Gasification%20Thermodynamic%20Model%20Including%20Tar%20and%20Char&sort=relevance)
84. [227,230,326] Thermochemical Equilibrium Model of Synthetic Natural Gas Production from Coal Gasification Using Aspen Plus — <https://doi.org/10.1155/2014/192057>
85. [231,329] Plasma Gasification With Municipal Solid Waste As A Method Of Energy Self Sustained For Better Urban Built Environment: Modeling and Simulation — <https://doi.org/10.1088/1755-1315/396/1/012002>
86. [233] Biomass Steam Gasification: A Comparison of Syngas Composition between a 1-D MATLAB Kinetic Model and a 0-D Aspen Plus Quasi-Equilibrium Model — <https://doi.org/10.3390/COMPUTATION8040086>
87. [234] An overview of advances in biomass gasification — <https://doi.org/10.1039/C6EE00935B>
88. [235] Review and analysis of biomass gasification models — <https://doi.org/10.1016/J.RSER.2010.07.030>
89. [236] Simulation of hybrid biomass gasification using Aspen plus: A comparative performance analysis for food, municipal solid and poultry waste — <https://doi.org/10.1016/J.BIOMBIOE.2011.06.005>
90. [237,247] Closed-loop controlled inspired oxygen concentration for mechanically ventilated very low birth weight infants with frequent episodes of hypoxemia — <https://doi.org/10.1542/PEDS.107.5.1120>
91. [238,248] Multicenter Crossover Study of Automated Control of Inspired Oxygen in Ventilated Preterm Infants — <https://doi.org/10.1542/peds.2010-0939>
92. [240] Increased System Fidelity for Navy Aviation Hypoxia Training — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Increased%20System%20Fidelity%20for%20Navy%20Aviation%20Hypoxia%20Training&sort=relevance)
93. [242] Measurement and control of impurity distribution in ultra pure gas delivery systems — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Measurement%20and%20control%20of%20impurity%20distribution%20in%20ultra%20pure%20gas%20delivery%20systems.&sort=relevance)
94. [245] Response by an Automated Inspired Oxygen Control System to Hypoxemic Episodes: Assessment of Damping — [Semantic Scholar search](https://www.semanticscholar.org/search?q=RESPONSE%20BY%20AN%20AUTOMATED%20INSPIRED%20OXYGEN%20CONTROL%20SYSTEM%20TO%20HYPOXEMIC%20EPISODES%3A%20ASSESSMENT%20OF%20DAMPING&sort=relevance)
95. [249] Automated Oxygen Flow Titration to Maintain Constant Oxygenation — <https://doi.org/10.4187/respcare.01343>
96. [250,262] Thermal efficiency improvement through fuel gas rate and excess oxygen control — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Thermal%20efficiency%20improvement%20through%20fuel%20gas%20rate%20and%20excess%20oxygen%20control&sort=relevance)
97. [251,264,265] Development of Optimal Control System for Air Separation Unit — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Development%20of%20Optimal%20Control%20System%20for%20Air%20Separation%20Unit&sort=relevance)
98. [252,259,386] Conceptual Process Design — <https://doi.org/10.1007/978-3-642-12525-6_11>
99. [253] Approaches to the Gas Control in UCG — <https://doi.org/10.14311/AP.2017.57.0182>
100. [255] United States Patent No.: US 7,531,135 B2 (d'Haene) — [Semantic Scholar search](https://www.semanticscholar.org/search?q=United%20States%20Patent%20(%2010%20)%20Patent%20No%20.%20%3A%20US%207%20%2C%20531%20%2C%20135%20B%202%20d%20'%20Haene&sort=relevance)
101. [257,263,266] Modeling and Simulation of Energy Systems — <https://doi.org/10.3390/books978-3-03921-519-5>
102. [268,275] Intelligent feed-forward and feedback control for oxygen ratio in the fuel cell stack — <https://doi.org/10.4314/IJEST.V3I10.1>
103. [269] Identification and control of a fed-batch process: application to culture of Saccharomyces cerevisiae — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Identification%20and%20control%20of%20a%20fed-batch%20process%20%3A%20application%20to%20culture%20of%20Saccharomyces%20cerevisiae&sort=relevance)
104. [270] A predictive dynamic model of a smart cogeneration plant fuelled with fast pyrolysis bio-oil — <https://doi.org/10.13044/j.sdewes.d10.0430>
105. [273,357] 基于自适应遗传算法的气化炉氧碳比控制 — [万方数据](https://s.wanfangdata.com.cn/paper?q=%E5%9F%BA%E4%BA%8E%E8%87%AA%E9%80%82%E5%BA%94%E9%81%97%E4%BC%A0%E7%AE%97%E6%B3%95%E7%9A%84%E6%B0%94%E5%8C%96%E7%82%89%E6%B0%A7%E7%A2%B3%E6%AF%94%E6%8E%A7%E5%88%B6)
106. [276] Apparatus and method for controlling fuel injection of internal combustion engine, and internal combustion engine — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Apparatus%20and%20method%20for%20controlling%20fuel%20injection%20of%20internal%20combustion%20engine%2C%20and%20internal%20combustion%20engine&sort=relevance)
107. [279] Gas-phase measurements of a bituminous coal in a pressurized entrained-flow gasifier — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Gas-phase%20measurements%20of%20a%20bituminous%20coal%20in%20a%20pressurized%20entrained-flow%20gasifier&sort=relevance)
108. [283] Aspects of the choice of sampling frequency in the control system of a gas turbine — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Aspects%20of%20the%20choice%20of%20sampling%20frequency%20in%20the%20control%20system%20of%20a%20gas%20turbine&sort=relevance)
109. [284] Biomass Gasifier "Tars": Their Nature, Formation, and Conversion — <https://doi.org/10.2172/3726>
110. [285] A review of the primary measures for tar elimination in biomass gasification processes — <https://doi.org/10.1016/S0961-9534(02)00102-2>
111. [286] The reduction and control technology of tar during biomass gasification / pyrolysis: An overview — <https://doi.org/10.1016/J.RSER.2006.07.015>

### 其他来源（直接引用）

112. [1,2,9,292,294,299] 基于 Aspen Plus 的气流床煤气化炉建模及其变工况特性研究 — [PDF](https://www.pgtjournal.com/CN/article/downloadArticleFile.do?attachType=PDF&id=942)
113. [3,17,18,214] Modeling of a Pressurized Entrained-Flow Coal Gasifier for Power Plant Simulation — [PDF (DLR)](https://elib.dlr.de/85503/1/Krueger_2014.pdf)
114. [7,8,109,110,112,113,114,116,172,181,291,296,297,318] Research on 6.5 MPa Multi-Nozzle Opposed Coal-Water-Slurry Gasifier based on CFD — [PDF (EPJ)](https://www.epj-conferences.org/articles/epjconf/pdf/2025/17/epjconf_icpms2025_01009.pdf)
115. [10,303] Performance Analysis of a Mui Basin Coal Gasifier Based on Feed Oxygen, Water Concentration and Heat Regulation — [PDF (JKUAT thesis)](http://ir.jkuat.ac.ke/bitstream/handle/123456789/5861/Oyugi%20George%20-THESIS%20FINAL-2.pdf?sequence=1&isAllowed=y)
116. [11,15,111,115,117,298,302,324] BGL 气化炉新型建模方法及优化分析 — [PDF (SciEngine)](https://dds.sciengine.com/cfs/files/pdfs/0023-074X/E40B488967AC4174ACD3C580C3595088-mark.pdf)
117. [12,301] Performance Study of Coal-Water Gasification Process in a Texaco Gasifier — [PDF (IIETA)](https://www.iieta.org/download/file/fid/28151)
118. [19,23,31,358] 城市固废焚烧过程烟气含氧量自适应预测控制 — [PDF (自动化学报)](http://www.aas.net.cn/cn/article/pdf/preview/10.16383/j.aas.c210935.pdf)
119. [20,28,34,37,359,360,364] 基于多模型切换控制的煤气化工业过程先进控制 — [PDF](http://www.chinacaj.net/d/file/2022-08-23/63584eabd07f2948b76d80e8d3c04cba.pdf)
120. [26,27,362,363] Nonlinear Generalized Predictive Control for Air Flow Rate Regulation in the PEM Fuel Cell System — [PDF (IJCA)](http://article.nadiapub.com/IJCA/vol9_no10/8.pdf)
121. [33,166] Model-based predictive control of a solar hybrid thermochemical reactor for high-temperature steam gasification of biomass — [PDF (HAL)](https://hal.science/hal-04020748v1/file/Model-Based%20Predictive%20Control%20of%20a%20Solar%20Hybrid.pdf)
122. [53,64] Coal Gasification Performance with Key Operating Variables — [PDF (KoreaScience)](https://koreascience.kr/article/CFKO200727465738971.pdf)
123. [55,289] 基于 Aspen Plus 的废锅流程粉煤气化炉稳态流程模拟 — [PDF (煤炭学报)](https://mtxb.com.cn/cn/article/pdf/preview/93be74dd-ebf7-4518-bc82-6f3eb5570fcd.pdf)
124. [57] A Review on Explorations of the Oxygen Blast Furnace Process — [PDF (Åbo Akademi)](https://research.abo.fi/ws/files/33590150/OBF_Review_AboCris.pdf)
125. [60] Stoichiometric Approach to the Analysis of Coal Gasification Process — [PDF (InTech)](https://cdn.intechopen.com/pdfs/35403/InTech-Stoichiometric_approach_to_the_analysis_of_coal_gasification_process.pdf)
126. [67,68,72,200,203,260,325,385] The Proceedings of the 1981 Symposium on Instrumentation and Control for Fossil-Energy Processes — [PDF (UNT Digital Library)](https://digital.library.unt.edu/ark:/67531/metadc283514/m2/1/high_res_d/metadc283514.pdf)
127. [69] High Efficiency Furnace with Oxy-Fuel Combustion and Zero-Emission by CO₂ Recovery — [PDF (WGC 2009)](http://members.igu.org/html/wgc2009/papers/docs/wgcFinal00580.pdf)
128. [70] Life Cycle Assessment of a Hydrogen Valley — [PDF (Politecnico di Torino)](https://webthesis.biblio.polito.it/38322/1/tesi.pdf)
129. [71,168] Dynamic simulation and control of a continuous gasifier for solar fuel production — [PDF (HAL theses)](https://theses.hal.science/tel-05293722/document)
130. [82] Theory and Operation of Purification Systems — [PDF](https://kh.aquaenergyexpo.com/wp-content/uploads/2024/01/Theory-and-Operation-of-Purification-Systems.pdf)
131. [85,95,97] Dissolved Oxygen Control Based in Real-Time Oxygen Uptake Rate Estimation — [PDF (FWRJ)](https://fwrj.com/techarticles/0413%20FWRJ_tech%205.pdf)
132. [89,272,353] Commande par logique floue d'un circuit de combustion d'une chaudière de type compact (CEVITAL-Bejaia) — [PDF (Univ. Bejaia)](https://www.univ-bejaia.dz/xmlui/bitstream/handle/123456789/8941/Commande%20par%20logique%20floue%20d%E2%80%99un%20circuit%20de%20combustion%20d%E2%80%99une%20chaudi%C3%A8re%20de%20type%20compact%20R%C3%A9alis%C3%A9%20%C3%A0%20CEVITAL-Bejaia.pdf)
133. [91,93,239,241,388] Intelligent oxygen delivery: a portable concentrator combining closed-loop automation, PSA, and Q-learning for optimized performance — [PDF (Springer)](https://link.springer.com/content/pdf/10.1007/s42452-025-07698-4.pdf)
134. [106,108] 数字取证：网络安全防御的重要技术 — [PDF](http://www.cechina.cn/mag/pdf/201406.pdf)
135. [128] 基于扩张状态观测器的高空台进气环境模拟控制技术研究 — [PDF (SciEngine)](https://www.sciengine.com/doi/pdfView/17F5D82A33E64A56BED5AD4A13E699C9)
136. [131,134,155,159,368] Nonlinear Model Predictive Control for the ALSTOM Gasifier Benchmark Problem — [PDF (Skogestad/IFAC 2005)](https://skoge.folk.ntnu.no/prost/proceedings/ifac2005/Fullpapers/03017.pdf)
137. [132,136,139,161] Real-time optimization and model predictive control for aerospace and automotive applications — [PDF (MERL)](https://www.merl.com/publications/docs/TR2018-086.pdf)
138. [137,144,150,153,167,280,281,282,346,389] Dynamic Simulation and Control of a Hybrid Coal Gasifier / Steam Methane Reformer System — [PDF (McMaster Univ.)](https://macsphere.mcmaster.ca/bitstream/11375/15346/1/fulltext.pdf)
139. [138,142,146,147,154,160,372] Optimal Control of Biogas Plants using Nonlinear Model Predictive Control — [PDF (Maynooth Univ.)](https://eprints.maynoothuniversity.ie/id/eprint/3648/1/SMcL_optimal_control.pdf)
140. [140,145,148,163,370] Modeling and Predictive Control of Heating Systems via Gaussian Mixture Models — Executive Summary — [PDF (Politecnico di Milano)](https://www.politesi.polimi.it/retrieve/a352f3b1-5297-44b9-b2d2-f68b624a43e5/2024_04_Lobriglio.pdf)
141. [141,151,152,371] Leading the Lorenz-63 system toward the prescribed regime by model predictive control coupled with data assimilation — [PDF (Copernicus)](https://npg.copernicus.org/preprints/npg-2024-4/npg-2024-4-manuscript-version3.pdf)
142. [149] Modeling and Control of the Open Plate Reactor — [PDF (Lund Univ.)](https://lucris.lub.lu.se/ws/portalfiles/portal/4487051/8682383.pdf)
143. [162,164] Model Predictive Control Application in the Energy Saving Technology of Basic Oxygen Furnace — [PDF (MCIT)](https://mcitdoc.org.ua/index.php/ITConf/article/download/50/22/289)
144. [170,211,213] Reduced order modeling of the Shell-Prenflo entrained flow gasifier — [PDF (MIT DSpace)](https://dspace.mit.edu/bitstream/handle/1721.1/105407/ShellROM_rev_final.pdf?sequence=1)
145. [174,319] CO₂ Conversion by Oxygen-Enriched Gasification of Wood Chips — [PDF (RWTH Aachen)](https://publications.rwth-aachen.de/record/995465/files/995465.pdf)
146. [177,322] Allothermal Gasification of High-Ash Coals — [PDF (ECN)](https://www.ecn.nl/publicaties/PdfFetch.aspx?nr=ECN-M--14-004)
147. [178,314] Biomass for Renewable Energy, Fuels, and Chemicals — [PDF](https://lib.zu.edu.pk/ebookdata/Engineering/Energy%20System/Biomass%20for%20renewable%20energy,%20fuels,%20and%20chemicals%20%20by%20Donald%20L.%20Klass%20.pdf)
148. [185,189,366] Cascade Generalized Predictive Control — Applications in Power Plant Control — [PDF (Oulu Univ.)](https://oulurepo.oulu.fi/bitstream/handle/10024/35160/isbn951-42-8032-6.pdf?sequence=1)
149. [188,374] An Overview of Artificial Intelligence Application for Optimal Control of Municipal Solid Waste Incineration Process — [PDF (Semantic Scholar)](https://pdfs.semanticscholar.org/c55f/6380fdd65510ea6a9e6ad345402c2acc8831.pdf)
150. [192,373] Fluidized Bed Scale-Up for Sustainability Challenges. 1. Tomorrow's Tools — [PDF (Chalmers)](https://research.chalmers.se/publication/539978/file/539978_Fulltext.pdf)
151. [208,210,309,312,380] Gasification for Synthetic Fuel Production: Fundamentals, Processes, and Applications — [PDF (rexresearch1)](https://rexresearch1.com/WoodGasifierLibrary/GasificationSyntheticFuelProductionFundamentals.pdf)
152. [212] Observer-Based Fuel Control Using Oxygen Measurement — A study based on a first-principles model of a pulverized coal fired Benson Boiler — [PDF (OSTI)](https://www.osti.gov/etdeweb/servlets/purl/20588014)
153. [216] Simulation of Ash Deposition Behavior in an Entrained Flow Coal Gasifier — [PDF (SCIRP)](https://www.scirp.org/pdf/ijcce_2015052814035853.pdf)
154. [223] 循环流化床富氧气化运行特性研究 — [PDF (中国动力工程)](https://epjournal.csee.org.cn/dlkcsj/cn/article/pdf/preview/10.13500/j.dlkcsj.issn1671-9913.2021.03.005.pdf)
155. [243] 기계호흡 환자에서 흡입 산소 분율의 변화후 시간에 따른 동맥혈 산소 분압 측정 — [PDF (대한임상의학지)](https://www.ekjm.org/upload/42832196.pdf)
156. [244,387] Yet Another Benchmark — Part I — [PDF (Divetable)](https://www.divetable.eu/TDM/TDM_Issue011.pdf)
157. [246] Steady-state monitoring of oxygen in a high-throughput organ-on-chip platform enables rapid and non-invasive assessment of drug-induced nephrotoxicity — [PDF (RSC)](https://pubs.rsc.org/en/content/getauthorversionpdf/D3AN00380A)
158. [254] Official Journal of the Patent Office (India) — [PDF (IP India)](https://ipindia.gov.in/writereaddata/Portal/IPOJournal/1_1546_1/Part-1.pdf)
159. [256] ガス再循環による蒸気温度調整 (Steam Temperature Control by Gas Recirculation) — [PDF (Hitachi Hyoron)](https://www.hitachihyoron.com/jp/pdf/1957/03/1957_03_03.pdf)
160. [258,261] 燃气质量管理方法与实践——当今液化天然气质量与互换性研究进展论述之三 — [PDF (Gas800)](http://gas800.com/qikan/QiKan/20111215103117598.pdf)
161. [267,274] Modeling and Simulation of an Industrial Furnace with Flue Gas Recirculation for NOₓ Control — [PDF (UWO Scholarship)](https://uwo.scholaris.ca/bitstreams/50bf36b8-3a7b-4dec-b55a-f954855a843e/download)
162. [271] Understanding Process Dynamics and Control (Kravaris, Kookos) — [PDF](https://elmoukrie.com/wp-content/uploads/2022/06/understanding-process-dynamics-and-control-costas-kravaris-ioannis-k.-kookos-z-lib.org_.pdf)
163. [277,278] Reduced Order Modeling and Scale-up of an Entrained Flow Gasifier — [PDF (UWaterloo)](https://www.uwspace.uwaterloo.ca/bitstreams/03a47cd6-d729-4d60-a202-913e2ee1aa35/download)

### 未被直接引用的来源

下列文献在 PDF 原始来源列表中归入"以下来源未被直接引用"段，与本报告正文未直接挂钩，但作为背景资料保留：

- Development and Application of Optimal Design Capability for Coal Gasification Systems: Performance, Emissions, and Cost of Texaco Gasifier-Based Systems Using ASPEN — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Development%20and%20Application%20of%20Optimal%20Design%20Capability%20for%20Coal%20Gasification%20Systems%3A%20Performance%2C%20Emissions%2C%20and%20Cost%20of%20Texaco%20Gasifier-Based%20Systems%20Using%20ASPEN&sort=relevance)
- I-1 Stability of TITO Predictive Functional Control Systems — [Semantic Scholar search](https://www.semanticscholar.org/search?q=I-1%20STABILITY%20OF%20TITO%20PREDICTIVE%20FUNCTIONAL%20CONTROL%20SYSTEMS&sort=relevance)
- Fuzzy model predictive control for small-scale biomass combustion furnaces — <https://doi.org/10.1016/j.apenergy.2020.115339>
- Carbon dioxide methanation over hydrotalcite-based nickel catalysts — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Carbon%20dioxide%20methanation%20over%20hydrotalcite-based%20nickel%20catalysts&sort=relevance)
- Coal devolatilization at high temperatures — <https://doi.org/10.1016/S0082-0784(77)80341-X>
- Noncatalytic Heterogeneous Solid-Fluid Reaction Models — <https://doi.org/10.1021/IE50705A007>
- Control structure design for complete chemical plants — <https://doi.org/10.1016/j.compchemeng.2003.08.002>
- 氧气顶吹转炉计算机静态控制数学模型 — [万方数据](https://s.wanfangdata.com.cn/paper?q=%E6%B0%A7%E6%B0%94%E9%A1%B6%E5%90%B9%E8%BD%AC%E7%82%89%E8%AE%A1%E7%AE%97%E6%9C%BA%E9%9D%99%E6%80%81%E6%8E%A7%E5%88%B6%E6%95%B0%E5%AD%A6%E6%A8%A1%E5%9E%8B)
- CFD simulation of entrained-flow coal gasification: Coal particle density / size fraction effects — <https://doi.org/10.1016/J.POWTEC.2010.03.029>
- Entrainment Coal Gasification Modeling — <https://doi.org/10.1021/I260072A020>
- Numerical study on the coal gasification characteristics in an entrained flow coal gasifier — <https://doi.org/10.1016/S0016-2361(01)00101-6>
- Rapidly Accelerated Synchronous Generators in CAES Systems for Frequency Support in Power Grids — <https://doi.org/10.26686/wgtn.17142311>
- Flight control systems properties and problems, volume 1 — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Flight%20control%20systems%20properties%20and%20problems%2C%20volume%201&sort=relevance)
- The Detection of Malfunction Using a Process Computer — [Semantic Scholar search](https://www.semanticscholar.org/search?q=The%20Detection%20of%20Malfunction%20Using%20a%20Process%20Computer&sort=relevance)
- Mathematical Modelling and Hierarchical Encourage Particle Swarm Optimization Genetic Algorithm for Jet Pipe Servo Valve — <https://doi.org/10.1155/2022/9155248>
- 伺服阀的传递函数 — [维普期刊](https://www.cqvip.com/search?k=%E4%BC%BA%E6%9C%8D%E9%98%80%E7%9A%84%E4%BC%A0%E9%80%92%E5%87%BD%E6%95%B0)
- Predictive control: with constraints — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Predictive%20control%20%3A%20with%20constraints&sort=relevance)
- 采用工件与炉膛串级控温的新型工业炉 — [万方数据](https://s.wanfangdata.com.cn/paper?q=%E9%87%87%E7%94%A8%E5%B7%A5%E4%BB%B6%E4%B8%8E%E7%82%89%E8%86%9B%E4%B8%B2%E7%BA%A7%E6%8E%A7%E6%B8%A9%E7%9A%84%E6%96%B0%E5%9E%8B%E5%B7%A5%E4%B8%9A%E7%82%89)
- A multivariable control strategy for an industrial gas-phase polyethylene reactor — [Semantic Scholar search](https://www.semanticscholar.org/search?q=A%20multivariable%20control%20strategy%20for%20an%20industrial%20gas-phase%20polyethylene%20reactor&sort=relevance)
- BP Statistical Review of World Energy — <https://doi.org/10.2307/3324639>
- Economics of the H-Coal Process — [Semantic Scholar search](https://www.semanticscholar.org/search?q=Economics%20of%20the%20H-Coal%20Process&sort=relevance)
- Oxy-fuel combustion of pulverized coal: Characterization, fundamentals, stabilization and CFD modeling — <https://doi.org/10.1016/J.PECS.2011.09.003>
- Thermodynamic Equilibrium Model and Second Law Analysis of a Downdraft Waste Gasifier — <https://doi.org/10.1016/J.ENERGY.2007.01.010>
- The effect of air preheating in a biomass CFB gasifier using ASPEN Plus simulation — <https://doi.org/10.1016/J.BIOMBIOE.2009.05.004>
- MINCON, a BASIC program to control temperature and oxygen fugacity in furnaces — <https://doi.org/10.3133/OFR811324>
- Catalytic destruction of tar in biomass derived producer gas — <https://doi.org/10.1016/J.ENCONMAN.2003.08.016>
- Modelling of an Industrial Off-Gas Cleaning System — [PDF (Library and Archives Canada)](https://www.collectionscanada.gc.ca/obj/thesescanada/vol2/002/MR41960.PDF?oclc_number=679504663)
- 蒸汽炉温度的一种自适应预测控制 — [PDF (广东工业大学学报)](https://xbzrb.gdut.edu.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=1982)
- Dynamic Modelling and Simulation of an Industrial Boiler System — [PDF (U. Alberta)](https://ualberta.scholaris.ca/bitstreams/051d6d31-7d3b-4b3c-82d9-1aa5c9b83303/download)
- Monitoring and Control System of Underground Coal Gasification based on Industrial Ethernet and PLC — [PDF](https://dacemirror.sci-hub.se/journal-article/ce318803178341995235dfed626b7e92/li2014.pdf)
- Simulación de un control no lineal de oxígeno disuelto y demanda química de oxígeno en un reactor de lodos activados — [PDF (U. Chile)](https://repositorio.uchile.cl/tesis/uchile/2007/cortes_lm/sources/cortes_lm.pdf)
- PEMFC Gas-Feeding Control: Critical Insights and Review — [MDPI Actuators](https://www.mdpi.com/2076-0825/13/11/455)
- Application of Cascade Control in Solid Oxide Fuel Cell Thermal Management System — [PDF (HJCET)](https://pdf.hanspub.org/HJCET20110200000_71902832.pdf)
- 脱硫、硫磺回收与在线分析技术（上）— [PDF (MS17)](https://www.ms17.cn/uploadfile/jszx/uploadfile/201011/20101103030348869.pdf)
- Numerical Investigation on Performance of Coal Gasifier of 150 kW under Various Injection Conditions — [PDF (KNS Korea)](https://www.kns.org/files/pre_paper/33/15S-506%EC%B5%9C%ED%98%84%EA%B2%BD.pdf)
- 基于汽氧比优化与阶段性气化反应分析的煤炭地下气化过程 — [PDF (煤炭学报)](https://mtxb.com.cn/cn/article/pdf/preview/10.13225/j.cnki.jccs.QH24.1537.pdf)
- Energy Conservation through Control (Shinskey, 1978) — [PDF (Skogestad)](https://skoge.folk.ntnu.no/puublications_others/books/Shinskey-1978-Energy%20Conservation%20Through%20Control-SEARCHABLE.pdf)
