# 多模态多组分气体检测算法创新调研报告（第一轮）

## 1. 研究目标

目标任务定义为：

> 面向焊接保护气及其他多组分气体定量检测，融合超声飞行时间/原始波形、热导率、NDIR（非分散红外）及温压湿等异构传感信号，在动态工况下实现多组分浓度回归，并进一步解决传感器缺失、信号质量退化、漂移以及跨设备/跨传感体系泛化问题。

当前研究不建议继续围绕“换 CNN、TCN、LSTM 或 Transformer”进行局部结构搜索，而应将创新拆成五个可独立验证的算法模块：

1. 异构传感数据编码
2. 质量感知动态融合
3. 缺失模态鲁棒学习
4. 跨设备/跨传感体系域泛化
5. 物理约束多组分回归

---

# 2. 模块一：异构传感编码

## 核心问题

超声原始波形、NDIR、热导率和温压湿数据存在：

- 数据维度不同；
- 采样频率不同；
- 信号机理不同；
- 时间尺度不同；
- 信息密度不同。

因此简单执行：

$$
z=[z_{\mathrm{US}},z_{\mathrm{NDIR}},z_{\mathrm{TCD}},z_{\mathrm{env}}]
$$

再直接拼接，容易造成高维超声信息压制低维慢变量，或者网络过度依赖某个强模态。

## 重点论文

[1] [Graph-Driven Models for Gas Mixture Identification and Concentration Estimation on Heterogeneous Sensor Array Signals](https://consensus.app/papers/graphdriven-models-for-gas-mixture-identification-and-wang-wang/bce23fc4753f5e5ab7b7bc36a942afd5/?utm_source=chatgpt)  
Ding Wang 等，2024，IEEE Transactions on Instrumentation and Measurement，引用 10 次。论文利用图卷积网络与自注意力直接处理结构不同的传感阵列，GraphANet 在不同气体组分浓度估计中达到 $R^2>0.96$，而且固定配置可以应用于不同传感器、不同气室和不同气体体系。

**迁移价值：★★★★★**

建议将超声、热导、NDIR、环境传感器定义成异构节点，而不是普通特征通道：

$$
G=(V,E)
$$

$$
V=
\{
v_{\mathrm{US}},
v_{\mathrm{TCD}},
v_{\mathrm{NDIR}},
v_T,
v_P,
v_H
\}
$$

这是目前与你“可配置传感体系”最匹配的一篇论文。

[2] [A multi-rate sampling data fusion method for fault diagnosis and its industrial applications](https://consensus.app/papers/a-multirate-sampling-data-fusion-method-for-fault-huang-wu/8f413613c1c4523b995296a30ce5506b/?utm_source=chatgpt)  
Keke Huang 等，2021，Journal of Process Control，引用 68 次。针对工业传感器不同采样速率问题进行深度融合。

**迁移价值：★★★★☆**

适合解决你的高频超声波形与慢速 NDIR/热导数据之间的时间尺度差异。

[3] [Deep Learning for Data Modeling of Multirate Quality Variables in Industrial Processes](https://consensus.app/papers/deep-learning-for-data-modeling-of-multirate-quality-yuan-feng/97de52a7566855e5ae8989f855136278/?utm_source=chatgpt)  
Xiaofeng Yuan 等，2021，IEEE Transactions on Instrumentation and Measurement，引用 41 次。提出多速率堆叠自编码器，采用“共享表示 + 任务专属表示”。

可迁移为：

$$
z_i=E_i(x_i)
$$

$$
z_i=z_i^{shared}+z_i^{private}
$$

即同时学习气体组成共同信息和传感器特有信息。

[4] [Frame-Dilated Convolutional Fusion Network and GRU-Based Self-Attention Dual-Channel Network for Soft-Sensor Modeling of Industrial Process Quality Indexes](https://consensus.app/papers/framedilated-convolutional-fusion-network-and-grubased-liu-he/17b272b956a2580a8114989cb113186b/?utm_source=chatgpt)  
Jinping Liu 等，2022，IEEE Transactions on Systems, Man, and Cybernetics: Systems，引用 48 次。针对不同采样频率和不同数据形态设计双通道编码，然后采用自注意力融合。

**非常适合作为你的基础多模态基线。**

[5] [Deep fusion of time series and visual data through temporal Features: A soft-sensor model for FeO content in sintering process](https://consensus.app/papers/deep-fusion-of-time-series-and-visual-data-through-temporal-yang-yang/f2bb0bf4a29c5333af80c86e952bc89a/?utm_source=chatgpt)  
Chong Yang 等，2024，Expert Systems with Applications，引用 13 次。采用双分支结构处理异构工业数据并进行软测量回归。

### 模块一判断

最值得发展的结构不是：

$$
CNN_{\mathrm{US}}+MLP_{\mathrm{slow}}\rightarrow Concat
$$

而是：

$$
\boxed{
\text{模态专属编码}
\rightarrow
\text{统一Token/节点表示}
\rightarrow
\text{异构图或交叉注意力}
}
$$

---

# 3. 模块二：质量感知动态融合

这是目前最值得加入你算法的创新之一。

实际运行中：

- 超声可能受到回波变弱、噪声、错峰影响；
- NDIR 可能出现光源衰减、污染、饱和；
- 热导传感器可能发生温漂；
- 环境参数本身存在测量误差。

所以模态的重要程度应当随样本和时间变化：

$$
w_i=w_i(x,t)
$$

而不是固定：

$$
w_i=\mathrm{constant}
$$

## 重点论文

[6] [Provable Dynamic Fusion for Low-Quality Multimodal Data](https://consensus.app/papers/provable-dynamic-fusion-for-lowquality-multimodal-data-zhang-wu/36903a03d8ae59439c667c5c095ba9aa/?utm_source=chatgpt)  
Qingyang Zhang 等，2023，引用 178 次。提出质量感知多模态融合 QMF，通过不确定度评价模态质量，再动态调整模态贡献。

**迁移价值：★★★★★**

核心可写成：

$$
q_i=f_q(z_i)
$$

$$
w_i=
\frac{\exp(-q_i)}
{\sum_j\exp(-q_j)}
$$

$$
z_{\mathrm{fusion}}=\sum_iw_iz_i
$$

[7] [DMAW: A Dynamic Multimodal Measurement Fusion Network With Attention for Reliable Welding Process Monitoring Under Harsh Industrial Environments](https://consensus.app/papers/dmaw-a-dynamic-multimodal-measurement-fusion-network-with-fan-yuan/2e9ae20637d75b2f8c9262170321c588/?utm_source=chatgpt)  
Jiawei Fan 等，2026，IEEE Sensors Journal，引用 1 次。提出模态专属特征提取、交叉注意力以及动态可靠性加权，专门针对恶劣焊接环境中的噪声和模态退化。

**与课题场景匹配度：★★★★★**

这篇应列入第一批精读论文。

[8] [MIFDELN: A multi-sensor information fusion deep ensemble learning network for diagnosing bearing faults in noisy scenarios](https://consensus.app/papers/mifdeln-a-multisensor-information-fusion-deep-ensemble-ye-yan/1334d60f46675e7885b16c07886738ed/?utm_source=chatgpt)  
Maoyou Ye 等，2024，Knowledge-Based Systems，引用 122 次。重点解决强噪声环境下多传感器可靠融合。

[9] [Information Complementary Fusion Stacked Autoencoders for Soft Sensor Applications in Multimode Industrial Processes](https://consensus.app/papers/information-complementary-fusion-stacked-autoencoders-zhang-he/59be6540275a5106a467e02596d5a906/?utm_source=chatgpt)  
Xinmin Zhang 等，2024，IEEE Transactions on Industrial Informatics，引用 31 次。使用门控模块控制信息流，并通过逐层融合过滤底层噪声。

[10] [Multimodal Fusion on Low-quality Data: A Comprehensive Survey](https://consensus.app/papers/multimodal-fusion-on-lowquality-data-a-comprehensive-zhang-wei/081a15f7c1c15800b3318715c6ee9ad6/?utm_source=chatgpt)  
Qingyang Zhang 等，2024，引用 113 次。系统将低质量多模态问题划分为噪声、不完整、模态不平衡和样本级质量动态变化四类。

### 建议创新

建立“传感器质量分数”：

$$
Q_i=
f(
SNR_i,
\mathrm{drift}_i,
\mathrm{missing}_i,
\mathrm{saturation}_i,
\mathrm{uncertainty}_i
)
$$

再进行：

$$
z=\sum_iw(Q_i)z_i
$$

这比普通注意力的解释性更强，因为权重直接对应传感器可靠程度。

---

# 4. 模块三：缺失模态鲁棒学习

## 重点论文

[11] [Deep Multimodal Learning with Missing Modality: A Survey](https://consensus.app/papers/deep-multimodal-learning-with-missing-modality-a-survey-wu-wang/aa6f4d12475252289ffc2c4c5dc06e85/?utm_source=chatgpt)  
Renjie Wu 等，2024，Transactions on Machine Learning Research，引用 141 次。系统总结训练阶段和测试阶段缺失模态学习。

[12] [M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities](https://consensus.app/papers/m3ae-multimodal-representation-learning-for-brain-tumor-liu-wei/499e1ebdd9915713a3356a0f0afcf8b0/?utm_source=chatgpt)  
Hong Liu 等，2023，引用 127 次。随机删除整个模态和局部数据，并通过掩码自编码器进行重建，再进行自蒸馏。代码公开。

**迁移价值：★★★★★**

可以直接修改成：

训练时随机执行：

$$
US\rightarrow0
$$

或

$$
NDIR\rightarrow0
$$

或

$$
TCD\rightarrow0
$$

要求网络仍能预测气体浓度。

[13] [SMIL: Multimodal Learning with Severely Missing Modality](https://consensus.app/papers/smil-multimodal-learning-with-severely-missing-modality-ma-ren/433f906c2b035992be3e0bc6bff96a10/?utm_source=chatgpt)  
Mengmeng Ma 等，2021，引用 461 次。利用贝叶斯元学习处理训练和测试阶段严重缺失模态情况。

[14] [Maximum Likelihood Estimation for Multimodal Learning with Missing Modality](https://consensus.app/papers/maximum-likelihood-estimation-for-multimodal-learning-ma-xu/b6c1a9bc2f17596fa26e64533f7beb03/?utm_source=chatgpt)  
Fei Ma 等，2021，引用 18 次。即使训练数据中 95% 存在模态缺失，方法仍能利用这些样本。

[15] [Employing multimodal co-learning to evaluate the robustness of sensor fusion for industry 5.0 tasks](https://consensus.app/papers/employing-multimodal-colearning-to-evaluate-the-rahate-mandaokar/6f46fa5481c852e0af3bfcfa63be8c82/?utm_source=chatgpt)  
Anil Rahate 等，2022，Soft Computing，引用 35 次。直接以气体检测为案例研究缺失与噪声模态，多任务融合对缺失模态的鲁棒性明显强于普通中间融合。

[16] [Learning Contrastive Multimodal Fusion with Improved Modality Dropout for Disease Detection and Prediction](https://consensus.app/papers/learning-contrastive-multimodal-fusion-with-improved-gu-saito/42795372fea75146be1392c16de5d3fe/?utm_source=chatgpt)  
Yi Gu 等，2025，引用 9 次。采用模态随机失活、可学习模态 Token 和对比学习，并公开代码。

### 最推荐实现方案

第一阶段不需要直接采用复杂生成模型，而采用：

$$
\boxed{
Modality\ Dropout
+
Missing\ Token
+
Consistency\ Loss
}
$$

完整模态预测：

$$
\hat y^{full}
$$

随机缺失后的预测：

$$
\hat y^{mask}
$$

增加：

$$
L_{\mathrm{cons}}
=
\|
\hat y^{full}
-
\hat y^{mask}
\|_2^2
$$

优点是实现简单、计算量低，并且非常容易做消融实验。

---

# 5. 模块四：域泛化与跨传感器迁移

该模块是当前最有机会形成独立创新点的方向。

需要区分三个难度层级。

### Level 1：时间漂移

同一个传感器：

$$
S(t_0)\rightarrow S(t_1)
$$

### Level 2：跨设备

$$
S_A\rightarrow S_B
$$

例如更换同类型 NDIR 或超声传感器。

### Level 3：跨传感体系

$$
\{
US,TCD,NDIR_A
\}
\rightarrow
\{
US',TCD',NDIR_B
\}
$$

第三种才真正对应“可配置传感体系”。

## 重点论文

[17] [TDACNN: Target-domain-free Domain Adaptation Convolutional Neural Network for Drift Compensation in Gas Sensors](https://consensus.app/papers/tdacnn-targetdomainfree-domain-adaptation-zhang-yan/a36c025b40265fd197748fda4bc876b0/?utm_source=chatgpt)  
Yuelin Zhang 等，2021，引用 74 次。针对气体传感漂移设计无需目标域参与训练的域适应方法。

[18] [Interpretable Calibration Transfer and Drift Compensation for MOS Gas Sensors in Complex Gas Mixtures](https://consensus.app/papers/interpretable-calibration-transfer-and-drift-schauer-morsch/a81314db82d4579c97e83c27c716c1df/?utm_source=chatgpt)  
J. Schauer 等，2026，Sensors。研究传感器批次差异、老化和校准迁移；仅使用约 10% 新域校准数据就能显著恢复定量性能。

**与你未来真实传感器更换场景高度匹配。**

[19] [Online Drift Compensation Framework Based on Active Learning for Gas Classification and Concentration Prediction](https://consensus.app/papers/online-drift-compensation-framework-based-on-active-se-song/5e0684a372785ac0a9c73fd89ea6b7bb/?utm_source=chatgpt)  
Haifeng Se 等，2023，Sensors and Actuators B: Chemical，引用 33 次。特别重要之处在于它不仅做分类，也研究**气体浓度预测**和在线漂移补偿。

[20] [Semi-supervised comparative learning compensation method for chemical gas sensor drift](https://consensus.app/papers/semisupervised-comparative-learning-compensation-xiong-wang/dbf54e7a27df530e99e8b07b0aaa7aa9/?utm_source=chatgpt)  
Lijian Xiong 等，2024，Analytical and Bioanalytical Chemistry，引用 7 次。通过少量目标域样本和对比学习建立漂移前后的表示对应关系。

[21] [Electronic Nose Drift Suppression Based on Smooth Conditional Domain Adversarial Networks](https://consensus.app/papers/electronic-nose-drift-suppression-based-on-smooth-zhu-wu/d2dc313bbe2a52b59ec07c396603d016/?utm_source=chatgpt)  
Huichao Zhu 等，2024，Sensors，引用 9 次。将条件域对抗网络与 SAM（锐度感知最小化）优化器结合，提取不同漂移域之间的公共特征。

[22] [MDFE-Net: A Meta-Learning Driven Dual-Branch Feature Extraction Network for E-Nose Sensor Drift Adaptation](https://consensus.app/papers/mdfenet-a-metalearning-driven-dualbranch-feature-yang-liu/f4be6c904ce55c6ea5cdd7a785a63c6a/?utm_source=chatgpt)  
Qilong Yang 等，2026，ACS Sensors，引用 3 次。采用模型无关元学习 MAML，利用少量标注快速适配新漂移条件，在跨设备实验中达到 97.89% 准确率。

[23] [Efficient Unsupervised Domain Adaptation Regression for Spatial–Temporal Sensor Fusion](https://consensus.app/papers/efficient-unsupervised-domain-adaptation-regression-for-niresi-nejjar/231031ebd2f9558aa24e30846e86108e/?utm_source=chatgpt)  
Keivan Faghih Niresi 等，2024，IEEE Internet of Things Journal，引用 6 次。针对**连续值回归**设计无监督域适应，与时空图网络结合，且代码公开。

### 模块四最佳路线

建议最终采用：

$$
\boxed{
\text{域不变表示}
+
\text{少样本元学习}
}
$$

基础损失：

$$
L=
L_{\mathrm{reg}}
+
\lambda L_{\mathrm{domain}}
$$

进一步加入少量目标设备标定样本：

$$
\theta'
=
\theta-\alpha\nabla_\theta L_{\mathrm{target}}
$$

目标从“重新训练新传感器模型”转变为：

> 新设备仅提供少量标定气样即可完成快速迁移。

工程价值明显更高。

---

# 6. 模块五：物理约束多组分回归

这是最容易做、同时又最不应该省略的一部分。

## 6.1 组分闭合约束

假设预测 $K$ 种气体：

$$
\mathbf y=
[y_1,\ldots,y_K]
$$

应严格满足：

$$
y_i\geq0
$$

$$
\sum_{i=1}^{K}y_i=1
$$

[24] [Physics-Informed Neural Networks with Hard Linear Equality Constraints](https://consensus.app/papers/physicsinformed-neural-networks-with-hard-linear-chen-constante-flores/2479dc9f89d659d8919331e3a359561d/?utm_source=chatgpt)  
Hao Chen 等，2024，Computers & Chemical Engineering，引用 74 次。证明单纯在损失函数增加软物理惩罚并不能严格满足约束，因此提出硬线性等式约束投影层。

[25] [A self-adaptive deep learning algorithm for accelerating multi-component flash calculation](https://consensus.app/papers/a-selfadaptive-deep-learning-algorithm-for-accelerating-zhang-li/5006bde5f90e59b9911d2cbb38fdff5c/?utm_source=chatgpt)  
Tao Zhang 等，2020，Computer Methods in Applied Mechanics and Engineering，引用 93 次。多组分摩尔分数预测中直接采用 Softmax 保证摩尔分数和为 1。

因此最简单可靠的输出层是：

$$
\hat y_i=
\frac{\exp(o_i)}
{\sum_j\exp(o_j)}
$$

自动保证：

$$
0<\hat y_i<1
$$

及：

$$
\sum_i\hat y_i=1
$$

比：

$$
L_{\mathrm{closure}}
=
(\sum_i\hat y_i-1)^2
$$

更值得优先采用。

[26] [Development of Steady-State and Dynamic Mass and Energy Constrained Neural Networks for Distributed Chemical Systems Using Noisy Transient Data](https://consensus.app/papers/development-of-steadystate-and-dynamic-mass-and-energy-mukherjee-bhattacharyya/06f925da8fa85e84ac780462c2c4d4a3/?utm_source=chatgpt)  
Angan Mukherjee 等，2024，Industrial & Engineering Chemistry Research，引用 14 次。研究动态化工过程，并强调硬质量守恒相较软惩罚的优势。

[27] [Physics-informed recurrent neural network modeling for predictive control of nonlinear processes](https://consensus.app/papers/physicsinformed-recurrent-neural-network-modeling-for-zheng-hu/6b0feff86bcd5f718eac3eee5b12b054/?utm_source=chatgpt)  
Yingzhe Zheng 等，2023，Journal of Process Control，引用 94 次。将动态机理与循环神经网络结合，说明物理信息可改善噪声环境下的泛化。

## 6.2 超声物理约束

[28] [Gas discrimination by simultaneous sound velocity and attenuation measurements using uncoated capacitive micromachined ultrasonic transducers](https://consensus.app/papers/gas-discrimination-by-simultaneous-sound-velocity-and-hernandez-shanmugam/9d64ad6f9127506e84f4d7b39bb8c814/?utm_source=chatgpt)  
Luis Iglesias Hernandez 等，2022，Scientific Reports，引用 5 次。证明飞行时间与衰减联合能够辨识 H₂、CO₂ 和 CH₄，并可识别温湿度漂移。

因此超声辅助头可以预测：

$$
\hat t_{\mathrm{TOF}},\quad
\hat\alpha
$$

并增加：

$$
L_{\mathrm{US}}
=
L(\hat t_{\mathrm{TOF}},t_{\mathrm{TOF}})
+
L(\hat\alpha,\alpha)
$$

从而迫使超声编码器学习有物理意义的特征。

## 6.3 NDIR辅助约束

[29] [Rapid Recognition and Concentration Prediction of Gas Mixtures Based on SMLP](https://consensus.app/papers/rapid-recognition-and-concentration-prediction-of-gas-sun-liu/4e8578ca6432564b941ded7a7d486f67/?utm_source=chatgpt)  
Qifang Sun 等，2024，IEEE Transactions on Instrumentation and Measurement，引用 11 次。NDIR 数据通过“气体类型判别 → 对应浓度回归”的两阶段结构实现 CO₂/CH₄ 快速定量，0.5 s 数据下分类准确率达到 98.21%。

可增加：

$$
L_{\mathrm{gas-type}}
$$

作为辅助任务。

## 6.4 热导率物理约束

[30] [Concentration Measurement Technology for Ternary Gas Mixtures Using a Thermal Conductivity Gas Sensor](https://consensus.app/papers/concentration-measurement-technology-for-ternary-gas-akasaka-nakahara/7a00e5061bc25287ba143b1eb9c03a1b/?utm_source=chatgpt)  
Shunsuke Akasaka、Ken Nakahara，2026，IEEE Sensors Journal。利用多个加热工作点产生不同热导响应，实现 CH₄/CO₂/H₂ 三组分定量。

说明热导率不应只作为普通慢变量，而应被视为独立物理模态。

---

# 7. 焊接应用支撑

[31] [Wire composition and shielding gas flow monitoring based on image and spectrum multimodal network](https://consensus.app/papers/wire-composition-and-shielding-gas-flow-monitoring-based-chen-sun/48518bb536cc52dd9071852f9483ca0b/?utm_source=chatgpt)  
Xiaoyu Chen 等，2020，Measurement，引用 7 次。直接研究焊丝成分与保护气流量的多模态在线监测。

[32] [End-to-end online quality prediction for ultrasonic metal welding using sensor fusion and deep learning](https://consensus.app/papers/endtoend-online-quality-prediction-for-ultrasonic-metal-wu-meng/3ae3b64e1a1d52a7863751a640583c3a/?utm_source=chatgpt)  
Yulun Wu 等，2022，Journal of Manufacturing Processes，引用 36 次。研究多传感器深度融合，并系统比较早期、中期和晚期融合方式。

因此，“焊接保护气在线监测 + 多模态传感融合”并非人为构造的应用场景，而有明确文献基础。

---

# 8. 第一轮筛选后的 5 个算法创新 Idea

## Idea 1：质量感知异构图多模态气体定量网络

推荐等级：★★★★★

核心组合：

**GraphANet + DMAW + QMF**

结构：

$$
\text{独立编码器}
\rightarrow
\text{异构传感器图}
\rightarrow
\text{交叉注意力}
\rightarrow
\text{质量估计}
\rightarrow
\text{动态加权融合}
\rightarrow
\text{时序模型}
\rightarrow
\text{组分回归}
$$

主要解决：

- 简单拼接无法描述传感器物理关系；
- 不同模态量纲和信息量不平衡；
- 低质量传感器污染融合结果。

创新性：高  
实现难度：中高  
与现有数据兼容性：高

---

## Idea 2：缺失模态鲁棒的可配置气体传感网络

推荐等级：★★★★★

核心组合：

**Modality Dropout + Missing Token + 对比一致性学习 + M3AE**

训练阶段主动随机制造：

- 超声完全失效；
- NDIR失效；
- 热导失效；
- 两模态同时缺失；
- 局部时间段丢包。

损失：

$$
L=
L_{\mathrm{reg}}
+
\lambda_1L_{\mathrm{cons}}
+
\lambda_2L_{\mathrm{contrast}}
$$

最大的论文价值在于：

> 同一模型无需重新训练即可接受不同传感器子集。

这真正对应“可配置多模态气体检测”。

---

## Idea 3：跨设备少样本气体浓度回归网络

推荐等级：★★★★★

核心组合：

**TikUDA + MDFE-Net + 对比学习**

训练目标：

设备 A：

$$
D_A
$$

直接迁移到设备 B：

$$
D_B
$$

首先做无监督表示对齐：

$$
L_{\mathrm{domain}}
$$

若有少量设备 B 标定气样，再通过元学习快速更新：

$$
\theta_B
=
\theta_A-\alpha\nabla L_B
$$

最适合作为第二篇独立算法论文。

---

## Idea 4：物理一致性的多任务浓度回归网络

推荐等级：★★★★☆

结构：

$$
\text{融合特征}
\rightarrow
\begin{cases}
\text{组分浓度}\\
\text{气体存在状态}\\
\text{声速/飞行时间}\\
\text{衰减}\\
\text{NDIR吸收量}\\
\text{热导率}
\end{cases}
$$

总损失：

$$
L=
L_{\mathrm{conc}}
+
\lambda_1L_{\mathrm{US}}
+
\lambda_2L_{\mathrm{NDIR}}
+
\lambda_3L_{\mathrm{TCD}}
$$

浓度输出采用 Softmax 或硬约束投影：

$$
\sum_i\hat y_i=1
$$

该方案实现成本低，适合作为所有后续模型的公共基础。

---

## Idea 5：质量退化与缺失统一建模网络

推荐等级：★★★★★

不要把：

“传感器正常”

和：

“传感器完全坏掉”

视为两个离散状态，而建立连续退化过程：

$$
Q_i\in[0,1]
$$

模拟：

$$
x'_i
=
a_ix_i+b_i+\epsilon_i+d_i(t)
$$

其中：

- $a_i$：灵敏度衰减；
- $b_i$：零点漂移；
- $\epsilon_i$：随机噪声；
- $d_i(t)$：长期漂移。

当：

$$
Q_i\rightarrow0
$$

自然退化为“模态缺失”。

这样可以把“质量退化”和“传感器缺失”统一成一个问题，而不是分别设计两套网络。

这是目前五个 Idea 中我认为**方法论最完整**的一个。

---

# 9. 最推荐的总架构

综合第一轮文献，不建议一步构造过于复杂的模型。

推荐最终逐步发展为：

$$
\boxed{
\text{物理专属编码器}
\rightarrow
\text{异构传感表示}
\rightarrow
\text{交叉模态交互}
\rightarrow
\text{质量感知门控}
\rightarrow
\text{TCN/Transformer}
\rightarrow
\text{硬约束多组分回归}
}
$$

训练阶段附加：

$$
\boxed{
\text{Modality Dropout}
+
\text{传感器质量退化增强}
+
\text{域对齐}
}
$$

整体目标函数可设计为：

$$
L=
L_{\mathrm{reg}}
+
\lambda_1L_{\mathrm{aux}}
+
\lambda_2L_{\mathrm{consistency}}
+
\lambda_3L_{\mathrm{domain}}
+
\lambda_4L_{\mathrm{quality}}
$$

其中由于输出层已经满足：

$$
\sum_i\hat y_i=1
$$

通常不需要再额外设置强闭合惩罚。

---

# 10. 推荐执行顺序

优先级 1：先复现简单基线：

$$
E_{\mathrm{US}}
+
E_{\mathrm{TCD}}
+
E_{\mathrm{NDIR}}
\rightarrow
Concat
\rightarrow
TCN
\rightarrow
Softmax
$$

优先级 2：加入 **Modality Dropout**，验证传感器缺失。

优先级 3：加入 **质量感知门控 QMF/DMAW**，验证噪声、漂移、灵敏度下降。

优先级 4：Concat 替换成**异构图/交叉注意力**。

优先级 5：最后增加**域适应 + 元学习**，进行真正的跨传感器实验。

这种顺序每一步都有独立消融结果，能够清楚回答：

> 性能提升到底来自哪个创新模块？

而不是直接构建一个大型网络后无法解释性能来源。

---

# 11. 当前最应该精读的论文顺序

第一批建议按以下顺序：

1. **Wang et al., 2024 — GraphANet**：解决异构传感体系。
2. **Fan et al., 2026 — DMAW**：解决动态质量退化，且直接来自焊接场景。
3. **Zhang et al., 2023 — QMF**：建立质量感知融合理论。
4. **Liu et al., 2023 — M3AE**：建立缺失模态机制。
5. **Niresi et al., 2024 — TikUDA**：解决回归任务跨域迁移。
6. **Yang et al., 2026 — MDFE-Net**：解决跨设备少样本快速适应。
7. **Chen et al., 2024 — KKT-hPINN**：解决硬物理约束。