# 稀疏多频物理谱反演的通用算法改进：从“换网络”转向可辨识设计、有效估计与可信校准

> 文档类型：文献核验与算法构想推演
> 检索日期：2026-07-27
> 项目起点：[`tv3_multifreq_relaxation_spectroscopy_dl_implementation_plan.md`](../archive/completed/tv3_multifreq_relaxation_spectroscopy_dl_implementation_plan.md)
> 项目证据：[`MRS-2 指标`](../../outputs/tv3_mrs/identifiability_mrs2/metrics.json)、[`MRS-6 硬件需求说明书`](../archive/completed/tv3_mrs6_hardware_requirements.md)
> 适用边界：本文只提出新研究线候选，不改写 `mrs2_rank_upgraded_p90_fail`，不授权重启 MRS-3，也不把 0.4 vol% O2 重新设为强制门。

## 摘要

现有 MRS 线已经回答了一个基础问题：增加频率、衰减、湿度和压力观测，确实带来了新的独立信息。联合费舍尔秩（Fisher rank）从 1 提升到 3--5，就是直接证据。问题在于，这些新信息仍然太弱。在登记噪声下，最佳观测臂的窄窗 O2 最大 P90 约为 4.03 vol%。噪声预算还表明，4 个频点和 8 个频点的结果相差不到 1%。继续增加相似频点，收益已经很小；真正限制精度的是飞行时间（time of flight, TOF）、温度、声程、幅度标定以及前向模型误差。因此，若输入和采样方式不变，只把卷积神经网络（CNN）换成循环神经网络（RNN）、注意力网络或变换器网络（Transformer），缺少充分的理论依据。

本报告把这类任务统一看作“稀疏多频观测下的非线性物理反演”。目标参数与干扰参数（nuisance parameter）会共同改变观测结果；与此同时，训练数据有限，仿真模型与真实系统之间还可能存在模型偏差（model discrepancy）。因此，问题不只是“选哪一种网络”，而是怎样从测量、求解和校准三个层面提高可信度。

沿着这条思路，本文提出六类改进：带干扰参数消元的变量投影展开网络（Variable Projection Unrolling）、稳健目标导向 c 最优实验设计（robust goal-oriented c-optimal design）、同步多正弦激励与共模误差消除（simultaneous multisine）、覆盖率校准的基于仿真后验推断（coverage-calibrated simulation-based inference）、正交模型偏差灰箱反演（orthogonal model-discrepancy grey-box inversion），以及共享潜变量的多视图反演（shared-latent multi-view inversion）。

这六类方法各有分工。实验设计和同步激励负责增加有效信息；变量投影和后验推断负责更充分地利用已有信息；正交模型偏差和多视图方法则主要处理仿真到真实的偏差与弱标签数据。因此，它们不是六个互相竞争的网络，而是六个可以按需组合的环节。

## 1. 研究问题与关键更正

### 1.1 研究问题

- **研究问题一（RQ1）：** 对一般的稀疏物理谱反问题，算法上限由什么决定？哪些方法能增加信息，哪些方法只能提高估计效率？
- **研究问题二（RQ2）：** 在现有解析反演 H1 和物理信息反演 H2 的基础上，哪些改进可以跨气体、跨谱学模态和跨传感器复用？
- **研究问题三（RQ3）：** 每种构想在什么条件下成立，怎样用最小实验把它证伪？如何避免用更深的模型掩盖硬件或前向模型问题？

### 1.2 对原综述“白地”判断的更正

Zhu 等使用 `2N+1` 个声速测点重建多弛豫色散 [1]，Zhang 等进一步通过色散曲线的拐点识别声速相同的混合物 [2]。这些工作说明，即使只测 `c(f)`，也可以利用分子弛豫信息。原综述把“声学弛豫谱与现代深度学习（deep learning, DL）的结合”视为研究空白，但这项判断已经过时。2023 年，Liu 等提出注意力循环气体检测网络（attention recurrent gas detection, ARGD）：输入是六个频率的 `c(f)` 或归一化吸收 `αλ(f)`，网络由双向门控循环单元（BiGRU）、注意力模块和分类/浓度双任务输出头组成，用于识别 CO2/CH4/N2 混合物 [3]。

该工作在单一温度下生成了约 4.98 万个仿真样本。论文报告的两类输入分类准确率分别为 99.14% 和 99.86%，三组分均方根误差（RMSE）为 0.28%--0.67%；作者还用少量 CO2/N2 与 CH4/N2 实验点作了验证 [3]。因此，下面这句话不能再作为创新声明：

> “首次把现代深度学习用于声学弛豫谱组分反演。”

不过，ARGD 并没有直接解决本项目面对的几个难点：O2/N2 浓度窗口很窄，温度、湿度、压力和声程需要联合估计，频点尚未围绕 O2 目标专门设计，网络也没有给出经过校准的后验分布。它同样没有证明普通的注意力循环网络能够接近含干扰参数的克拉默--拉奥下界（Cramér--Rao bound, CRB）。所以，真正值得研究的问题已经不是“能不能把深度学习用于弛豫谱”，而是“怎样让采样包含更多目标信息，怎样减少干扰参数的影响，怎样给出可信的不确定度，以及怎样防止仿真偏差误导反演”。

## 2. 方法与证据范围

### 2.1 检索视角

检索按六个互相制衡的视角进行：

1. 声学弛豫谱的解析模型、稀疏重建和直接深度学习；
2. 可分离非线性反问题、变量投影（Variable Projection）和算法展开（algorithm unrolling）；
3. c 最优设计、目标导向设计和稳健贝叶斯实验设计；
4. 同步多正弦激励、费舍尔最优功率分配和时基抖动；
5. 基于仿真的推断（simulation-based inference, SBI）、后验覆盖率和仿真校准；
6. 计算模型偏差、正交校准和多视图可辨识性。

检索以论文题名、数字对象标识符（DOI）页面、期刊全文页、学术会议页面和 arXiv 全文为证据。只有题名、作者、摘要或正文主张能够核对的文献才进入报告。检索先看通用理论，再用具体气体实验校准适用边界；同时主动寻找反例，因此发现了已经发表的 ARGD 工作。

### 2.2 证据类型

| 证据类型              | 能支持的结论            | 不能支持的结论       |
| ----------------- | ----------------- | ------------- |
| 解析理论与统计定理         | 可辨识条件、信息上限、优化等价性  | 具体硬件能达到的误差    |
| 仿真基准实验（benchmark） | 算法在已知仿真器和噪声下的相对表现 | 现场泛化与绝对校准     |
| 实验室气体实验           | 特定装置、组分和工况下的可行性   | O2 窄窗与矿井现场能力  |
| 项目费舍尔信息与 CRB 审计   | 已登记域内任何算法都不能越过的下界 | 存在未建模误差时的真实性能 |
| 相邻谱学与系统辨识         | 可以迁移的采样和反演思路      | 未经验证的声学数值收益   |

## 3. 统一问题形式：先区分“信息增益”和“估计增益”

先把问题写成一个通用形式。这样做，是为了看清误差究竟来自目标本身、干扰参数、实验设计，还是模型偏差：

```text
y_s = F(theta, eta_s; d_s, psi) + delta(z_s, d_s) + epsilon_s,
epsilon_s ~ (0, Sigma_s).
```

各符号的含义如下：

- `theta` 是需要反演的目标参数，例如气体组分、材料参数或生理参数；
- `eta_s` 是第 `s` 个观测视图（view）中的干扰参数，例如温度、湿度、声程、增益和触发延迟；
- `d_s` 是实验设计变量，例如频率、功率、压力、湿度和观测时长；
- `psi` 是各次观测共享的物理参数；
- `delta` 表示仿真器没有表达出来的模型偏差；
- `Sigma_s` 是观测噪声的协方差，它既可以包含各频点自己的噪声，也可以包含跨频共模噪声。

在某个工况附近，前向模型对参数的导数构成雅可比矩阵（Jacobian）。将其分成目标块 `J_theta` 和干扰块 `J_eta` 后，可以用舒尔补（Schur complement）写出目标参数的有效费舍尔信息：

```text
I_eff(theta)
  = I_theta,theta
  - I_theta,eta (I_eta,eta + Lambda_eta)^(-1) I_eta,theta,

I_ab = J_a^T Sigma^(-1) J_b.
```

`Lambda_eta` 表示有真实依据的干扰参数先验。Fewster 与 Jupp 证明，在存在干扰参数时，目标参数的有效费舍尔信息仍具有单调性；增加新的观测来源，只有在它确实提供了新的条件信息时，才会缩短置信区间 [6]。用直白的话说，重复测量同一种信息不会自动提高可辨识性。由此得到四条判断规则：

1. 改变实验设计 `d` 或前向物理 `F`，才可能改变灵敏度方向 `J`，从源头上增加信息；
2. 改变采样协议或噪声协方差 `Sigma`，可以降低噪声，或者利用原本被忽略的共模结构；
3. 增强先验信息 `Lambda_eta` 必须依靠真实标定或可靠知识，不能让网络凭训练数据自行假定；
4. 如果观测不变，只更换反演器，最多让估计结果更接近 `I_eff^-1`，不能越过同一观测模型的 CRB。

这也是本文不再讨论普通注意力网络、特征线性调制（FiLM）、专家混合模型（MoE）和同输入 Transformer 变体的原因：它们没有改变信息来源。

## 4. 六类通用算法构想

### 4.1 构想 A：干扰参数投影的变量投影展开网络（Nuisance-Projected Variable Projection Unrolling, NP-VPNet）

#### 通用定义

许多谱反问题都可以拆成两组参数。一组参数以非线性方式决定谱形，例如组分和弛豫时间；另一组参数在前一组给定后，以线性或仿射方式进入观测，例如固定延迟和幅度偏移。这类问题称为可分离非线性最小二乘（separable nonlinear least squares）。将非线性参数记为 `beta`，条件线性参数记为 `a`：

```text
min_{beta, a} ||W^(1/2) [y - A(beta) a - b(beta)]||_2^2
              + R_beta(beta) + R_a(a).
```

变量投影的做法很直接：每给定一个 `beta`，先把较容易求解的 `a` 算出来：

```text
a*(beta) = argmin_a ||W^(1/2) [y - A(beta) a - b(beta)]||_2^2 + R_a(a),
```

然后只优化降维后的目标 `phi(beta)=L(beta,a*(beta))`。Golub 与 Pereyra 的综述表明，变量投影不仅减少了待优化参数的数量，通常还会改善问题的条件数，使求解过程不容易被小扰动放大 [7]。Español 与 Pasha 又把它扩展到带一般吉洪诺夫正则化（Tikhonov regularization）的非线性反问题，并分析了高斯--牛顿法（Gauss--Newton）使用近似雅可比矩阵时的收敛性 [8]。

NP-VPNet 不让神经网络从头学习整个逆映射，而是把固定步数的阻尼高斯--牛顿法（damped Gauss--Newton）展开成网络：

```text
beta_{t+1} = Project_C[
  beta_t - P_t (J_t^T W_t J_t + lambda_t I)^(-1) J_t^T W_t r_t
].
```

网络只学习预条件矩阵 `P_t`、阻尼系数 `lambda_t`、异方差权重（heteroscedastic weight）或小型残差先验（residual prior）。前向模型 `F`、组分闭包约束和每一步的数据一致性（data consistency）仍然明确写在模型中。算法展开的价值就在这里：保留成熟求解器的结构，只把难以手工设定的少量部分交给数据学习，因此比完全黑盒的端到端映射更节省样本，也更容易解释 [9]。

#### 在 tv3 的具体化

- 非线性参数块包括三组分、弛豫时间、弛豫强度，以及可能的温度和湿度修正参数；
- 条件线性或近线性参数块包括公共固定延迟、声程 `L`、对数幅度增益和每个频率的标定偏移；
- 不同观测臂共享组分，而各观测臂自己的延迟与增益由变量投影消去；
- 组分使用单纯形投影（simplex projection）维持闭包，真值审计数组（oracle array）仍禁止进入部署输入；
- H1 解析反演作为第 0 层初值，H2 中的黑盒编码器（encoder）改为“物理迭代 + 小型学习预条件器”。

#### 为什么可能改善

它可以减少干扰参数与目标参数在数值优化中的纠缠。例如，声程误差本应由声程参数解释，而不应被组分输出头“吃掉”。在样本较少时，这种结构也更有机会接近最大似然解（maximum likelihood）或最大后验解（maximum a posteriori, MAP）。只要前向模型可微，并且存在可以先消去的参数块，这个思路就能迁移到声学、阻抗、光谱、磁共振成像（MRI）和材料参数反演。

#### 它不能做什么

NP-VPNet 不能突破同一 `I_eff` 对应的 CRB。MRS-6 已经表明，登记条件下的主要误差下限来自声程 `L`、幅度和 TOF。如果这些量与组分在观测上仍然高度共线，变量投影只能更稳定地算出同一个宽置信区间。声称它“仅靠算法就能把 4.03 降到 0.4”，没有理论依据。

#### 最小证伪实验

1. 在同一仿真数据、同一频点和同一噪声下，比较 H1、多层感知机（MLP）/ARGD、纯高斯--牛顿法和 NP-VPNet；
2. 除平均绝对误差（MAE）外，还要报告相对 CRB 效率、失败率、迭代收敛情况和分布外（OOD）干扰参数表现；
3. 如果只有把干扰参数固定为真值时 NP-VPNet 才有增益，说明问题仍在干扰参数，而不是谱表示；
4. 如果 NP-VPNet 在测试集和 OOD 条件下都不优于纯高斯--牛顿法，就停止增加展开层数。

### 4.2 构想 B：稳健目标导向 c 最优实验设计（Robust Goal-Oriented c-Optimal Design, RG-cOED）

#### 通用定义

当前 K=4 与 K=8 的结果几乎相同，这说明增加更多相似频点已经进入冗余区。此时不该继续问“还要加几个点”，而应问“在频率、压力、湿度、功率和停留时间都受限时，怎样选择测量条件，才能让目标参数的不确定度最小”。把这些可控条件记为 `d=(f,P,RH,power,dwell)`。

如果只关心参数向量中的某个方向 `c`，局部 c 最优设计的目标是：

```text
min_d  c^T I_eff(d, xi)^(-1) c + lambda_cost * Cost(d),
```

其中 `xi` 表示工况和模型不确定性。只在中心工况上优化，容易得到“中心点很好、边界点很差”的设计。为避免这种情况，可以采用极小极大（minimax）或风险规避形式：

```text
min_d  max_{xi in Xi} c^T I_eff(d, xi)^(-1) c,
```

还可以直接最小化误差分布的高分位数。Sagnol 证明，在候选测量有限且每次测量有多个响应量时，c、A、T、D 等最优设计可以转化为凸优化问题，并能加入测量次数、能量和时间等线性约束 [10]。目标导向贝叶斯实验设计（goal-oriented Bayesian experimental design）进一步指出，设计目标应直接针对真正关心的量，而不是平均照顾所有参数 [11]。如果系统中还有次级参数或模型不确定性，就必须把它们纳入积分或稳健优化；否则所谓“最优频点”可能只对一个错误的前向模型最优 [12]。

#### 两阶段自适应版本

1. 粗测阶段（scout batch）：先用 2--3 个低成本测点估计弛豫拐点的大致位置和干扰参数后验；
2. 定向测量阶段（target batch）：根据第一阶段结果，选择最能增加 O2 有效信息的频率、压力或湿度；
3. 停止规则（stop rule）：预计信息增益低于成本阈值时停止，不再机械地采满 K=8。

这种两阶段做法并不局限于气体弛豫谱。只要谱峰或拐点会随工况移动，就可以先粗略定位，再把第二批测点放到当前样本最有信息的位置。

#### 在 tv3 的具体化

- 目标函数不是整个费舍尔矩阵的行列式 `det(Fisher)`，而是 O2 窄窗方向的 `c^T I_eff^-1 c`；
- 设计空间包含硬件可实现的频率、RH/压力切换、脉冲串（burst）能量和总测量时长；
- 对 216 个登记窄窗点和前向参数不确定性做最坏情况或条件风险价值（CVaR）优化；
- 把低 RH、高压力死角写成硬约束（hard constraint），而不是只在优化完成后画一张附带热力图；
- 0.4 vol% 仅作为参考线，新实验的门值必须另行预注册。

#### 为什么可能真正改善

RG-cOED 会直接改变雅可比矩阵 `J`，所以它是六个构想中少数有机会真正降低 CRB 的方法。K4 与 K8 几乎等价，也可以用这个框架解释：如果新增频点的目标灵敏度仍落在已有列空间内，`I_eff` 就几乎不增长。只有让测点跨过弛豫拐点，或者改变压力、湿度与功率分配，才可能带来新的有效信息。

#### 失效条件

- 候选硬件频带内所有 `J_theta` 都近共线；
- 用来设计频点的仿真器存在未建模偏差；
- 自适应选择时间长于被测状态稳定时间；
- 用面向全部参数的 D 最优目标代替 O2 目标，导致测量资源花在无关干扰参数上。

#### 最小证伪实验

在频点数、总声能和总时长完全相同的条件下，比较固定 K4、随机 K4、局部 c 最优、稳健 c 最优和两阶段自适应策略。评价先在纯前向 CRB 上完成，不生成波形。只有稳健或自适应方案在留出工况和扰动后的前向模型上都降低最坏 P90，才允许进入波形生成阶段。

### 4.3 构想 C：同步多正弦与共模剖面似然（Simultaneous Multisine and Common-Mode Profile Likelihood, SIMD-MRS）

#### 通用定义

顺序扫频有一个容易忽略的问题：不同频点来自不同触发时刻，被测状态也可能在扫描期间缓慢变化。同步多正弦激励（multisine excitation）把多个选定频率放进同一个时间窗，可以让它们共享一次触发和近似相同的环境状态。一般的复传递函数可以写成：

```text
H_k = G_k exp[-alpha_k(theta, eta) L]
      * exp{-j 2*pi*f_k [L/c_k(theta, eta) + tau_common]}
      + noise_k.
```

其中，`tau_common` 是所有频点共享的触发或时基偏移，`G_k` 是仪器的频率响应。可以用联合剖面似然（joint profile likelihood）同时估计 `theta`、`tau_common` 和低维 `G_k`；也可以加入参考声程或参考通道，直接消除电子链路带来的公共相位。若相位抖动可以分解为

```text
Sigma_phase = sigma_ind^2 I + sigma_common^2 v v^T,
```

那么广义最小二乘（generalized least squares）就能利用其中的低秩共模结构。这里必须保持两方面的谨慎：把公共抖动误写成 K 个彼此独立的 3 us 噪声，会低估系统能力；反过来，把真正独立的抖动当成共模噪声，又会虚高性能。

阻抗谱（impedance spectroscopy）的研究已经证明，多正弦激励既能缩短测量时间，也能按照费舍尔信息分配各频率的激励功率。Sanchez 等给出了 D 最优多正弦功率谱的理论和实验验证 [13]。针对时基失真与抖动，Schoukens 等的正弦拟合（sine-wave fitting）研究表明，这些误差应进入联合估计模型，而不应一律当作独立白噪声 [14]。

#### 在 tv3 的具体化

- 先用构想 B 选出的 4 个频率生成低峰均比（low crest factor）多正弦信号；
- 在同一个脉冲串中估计各频率的复传递函数 `H(f_k)`，减少逐频扫描带来的状态漂移；
- 参考通道测电子延迟，测量通道保留气体传播相位；
- 将 `c_eq` 对应的近线性相位与弛豫色散残差分开建模，不能简单去趋势后丢掉全部基线声速信息；
- 用克拉默斯--克勒尼希关系（Kramers--Kronig relation）和因果性残差检查频谱能否由线性稳定系统解释；由于测量带宽有限，这里只能使用软约束。

#### 为什么可能真正改善

SIMD-MRS 同时改变采样协议和噪声协方差 `Sigma`。如果当前 3 us 抖动主要来自公共触发，而不是每个频点独立的相位噪声，那么同步激励和参考通道就有机会消除其中的大部分。同步测量还可以减少 RH、压力和温度在扫频期间的漂移。这个思路也适用于电化学阻抗谱（EIS）、核磁共振（NMR）、太赫兹（THz）光谱，以及其他需要测量动态对象频率响应的任务。

#### 失效条件

- 抖动主要来自每个频率独立的相位估计，而不是公共触发；
- 换能器或功率放大器的互调使系统不再近似线性；
- 多正弦信号的峰均比迫使单频能量下降，信噪比（SNR）损失超过共模消除带来的收益；
- 未建模 `G_k` 吸收了 `alpha_k`，形成新的幅度不可辨识性。

#### 最小证伪实验

先在空腔或参考负载（reference load）上重复采集，估计跨频误差协方差的特征值谱。只有第一主成分足够显著，并且与 `f_k` 或公共延迟模式一致，才有证据支持共模假设。随后在总能量相同的条件下，比较顺序 K4 和多正弦 K4 的 `Sigma`、CRB、互调残差与 Kramers--Kronig 残差。在共模结构得到实测确认之前，禁止在仿真器中预设有利的跨频相关性。

### 4.4 构想 D：覆盖率校准的基于仿真后验推断（Coverage-Calibrated Simulation-Based Posterior Inference, CC-SBI）

#### 通用定义

点回归只给一个答案。在病态反问题中，这可能造成明显误导：如果两组不同参数都能解释同一观测，均方误差训练往往会输出二者的平均值，而这个平均值本身可能不符合物理。CC-SBI 不直接学习单点答案，而是学习完整的条件后验分布：

```text
q_phi(theta | {f_k, y_k, quality_k}, slow_channels, design),
```

输入端采用置换不变集合编码器（permutation-invariant set encoder），因此频点顺序不会影响结果，也能处理频点数量变化、个别频点缺测和自适应测量。输出端使用条件归一化流（conditional normalizing flow）或其他显式概率密度模型。训练时，从仿真器中联合采样 `theta`、干扰参数、噪声和实验设计，直接学习后验密度；部署时只需一次网络前向计算。

当似然函数（likelihood）难以直接计算、但可以运行仿真器时，SBI 是一类常用的科学推断方法 [15]。不过，它并不天然可靠。Lueckmann 等的基准研究表明，SBI 的效果会随任务和评价指标明显变化，即使先进方法也会失效 [16]。Falkiewicz 等进一步发现，神经 SBI 容易给出过度自信的后验，并提出可微覆盖率损失（coverage loss）来改善经验覆盖率 [17]。因此，CC-SBI 必须同时进行基于仿真的校准（simulation-based calibration, SBC），不能只报告负对数似然（NLL）或后验均值 [18]。

#### 在 tv3 的具体化

- `theta` 表示满足单纯形约束的 O2/CO2/N2 三组分；
- 被边缘化的干扰参数包括 T、RH、L、逐频增益、公共/独立抖动和物理参数不确定性；
- 多湿度、多压力观测通过专家乘积（product of experts）或联合集合似然合并；
- 输出 O2 后验 P90、拒绝概率和分布外评分，而不只输出三组分均值；
- H1 与 NP-VPNet 的解可以作为后验提议分布或摘要统计，但不能作为真值审计标签。

#### 为什么可能改善

CC-SBI 不会凭空降低信息下限。它主要解决四个问题：

1. 在非线性、多峰或边界后验中，避免均方误差（MSE）造成的均值偏置；
2. 显式边缘化干扰参数，而不是把每个干扰参数都压成单点估计；
3. 对缺频和自适应设计提供统一推理接口；
4. 用经过校准的不确定度决定何时拒绝输出，把“错误但自信”变成能够被审计的失败。

这些能力不仅适用于气体检测，也适用于其他由仿真器驱动的传感反演任务。

#### 失效条件

- 仿真器没有覆盖真实数据的生成机制；
- 先验分布（prior）把测试样本排除在外；
- 只在仿真数据上检查覆盖率，随后把结论直接外推到现场；
- 只评价后验均值的 MAE，掩盖覆盖率崩溃。

#### 最小证伪实验

除平均绝对误差（MAE）和第 90 百分位误差（P90）外，还必须报告基于仿真的校准（SBC）秩直方图、50%/80%/90%/95% 经验覆盖率、负对数似然（NLL）、后验预测检验（posterior predictive check）和分布外（OOD）条件覆盖率。与 H1、NP-VPNet 比较时，如果 CC-SBI 只改善负对数似然，却让覆盖率变差，应判为失败。如果覆盖率正确但区间很宽，结论应是“观测信息不足”，不能人为缩窄区间来制造高精度。

### 4.5 构想 E：正交模型偏差灰箱反演（Orthogonal Model-Discrepancy Grey-Box Inversion, OMD-GreyBox）

#### 通用定义

仿真器不可能表达真实系统的全部细节。更符合实际的观测模型是：

```text
y_real = F_sim(theta, eta; d) + delta(d, z) + epsilon,
```

其中 `delta` 表示模型偏差。如果让 `delta` 完全自由，一个容量很大的残差网络（residual network）可能直接吸收本应由组分解释的变化。结果会出现一种危险现象：频谱重建得很好，但组分参数失去物理意义。Kennedy--O'Hagan 校准框架面对的主要难点，正是仿真器参数与模型偏差之间难以区分 [19]。

Plumlee 提出的非精确计算模型校准（inexact computer-model calibration）给出了一种处理办法：让模型偏差的先验与仿真器关于校准参数的梯度正交。这样可以减少“偏差函数吞掉目标参数”的问题 [20]。正交高斯过程（orthogonal Gaussian process）又把这个思想推广到一般的半参数模型，用来处理随机项与均值模型之间的可辨识冲突 [21]。离散形式可以写成：

```text
Phi_perp = [I - J (J^T W J)^(-1) J^T W] Phi,
delta(d) = Phi_perp(d) w,
```

从而在当前线性化点满足

```text
J^T W delta = 0.
```

`Phi` 可以是高斯过程基函数、样条基函数、低秩神经基函数，或有限元（FEM）残差基函数。无论选哪一种，正交约束都必须相对于目标参数的物理雅可比矩阵来定义。

#### 在 tv3 的具体化

把当前 MRS-1 单弛豫加和模型作为低保真模型（low-fidelity model）`F_sim`，再按以下顺序补充缺失物理：

1. 湿空气对 `c_eq` 的真实影响；
2. H2O 弛豫与 CO2 湿度系数；
3. Dain--Lueptow 耦合弛豫矩阵；
4. 低频衍射、近场和换能器频响。

Dain 与 Lueptow 已经表明，多组分之间的振动能量交换会产生彼此耦合的弛豫过程，而且有效弛豫频率随组分变化 [4]。Ejakov 等的多频、变压实验验证了含 N2 混合物的吸收谱趋势，同时也说明装置和频段条件不能忽略 [5]。因此，H2 的解码器不能永久限定为单个洛伦兹项（Lorentzian）简单相加，也不能直接加入一个没有物理约束的自由残差。

#### 为什么可能改善

OMD-GreyBox 关注的是仿真到真实偏差（sim-to-real bias），而不是仿真数据内部的模型排名。它允许使用少量真实标定数据修正系统误差，同时在一阶近似下保护组分灵敏度，避免它被模型偏差吸收。凡是“机理模型并不完整，但仍然具有解释价值”的数字孪生（digital twin）反演，都可以考虑这种结构。

#### 失效条件

- 没有真实数据或更高保真数据时，`delta` 只能重复仿真器自身的偏差；
- 正交性只在局部成立，远离线性化点仍可能全局混淆；
- 如果仿真器的雅可比矩阵本身错误，计算出的正交空间也会错误；
- 真实偏差恰好沿目标灵敏度方向，强制正交会低估不确定度。

#### 最小证伪实验

可以构造一个有意加入偏差的高保真数据生成器，也可以使用小规模真实标定集。实验比较三种情况：不建模偏差、使用自由残差、使用正交模型偏差。至少报告留出工况的组分偏差、覆盖率、后验预测残差、物理参数漂移和 Kramers--Kronig 残差。如果自由残差的频谱重建最好，但组分偏差反而最大，就说明它确实在吞掉目标参数；这正是 OMD-GreyBox 要避免的失败。

### 4.6 构想 F：共享潜变量的多视图一致性反演（Shared-Latent Multi-View Consistency Inversion, SMVCI）

#### 通用定义

很多实验都能在目标不变时主动改变测量条件。例如，在一小段稳定时间内，气体组分 `theta` 可以近似不变，但湿度、压力、频率集合和仪器增益可以变化。把每一种条件下的观测看作一个视图：

```text
y_s = F(theta_shared, eta_s; d_s) + epsilon_s,  s=1,...,S.
```

联合后验满足

```text
p(theta | y_1:S) proportional_to p(theta) * product_s p(y_s | theta, d_s),
```

彼此独立的视图，其费舍尔信息可以相加。只要新视图提供了非零条件信息，目标参数的置信区间就会缩短 [6]。算法中只设置一个共享组分潜变量，同时为每个视图设置自己的干扰潜变量，再用物理解码器重建各视图：

```text
L = sum_s L_recon(y_s, F(theta_bar, eta_s; d_s))
    + lambda_cons * sum_s ||theta_s - theta_bar||^2
    + lambda_sup * L_label(theta_bar, theta_true).
```

配对视图还可以用于对比训练（contrastive training）：同一混合物在不同条件下的观测是正样本，不同混合物的错误配对是负样本。Gresele 等证明，当不同视图足够多样，并满足相应噪声条件时，多视图可以让原本在单视图中不可识别的非线性潜变量变得可识别 [22]。Yao 等又给出了部分可观测条件下共享信息的多视图可辨识条件 [23]。

#### 在 tv3 的具体化

- 同一 `mixture_id` 下的双湿、双压和不同频率子集共享三组分潜变量；
- T、RH、P、增益、延迟和短时漂移使用各视图独立的潜变量；
- 有标签仿真配对用于监督，少量或无标签真实配对用于重建损失与一致性损失；
- 禁止把 `mixture_id` 回退或重写为 `sequence_id`；配对关系必须由正式的新数据模式（schema）明确表达；
- 将时间顺序打乱作为负对照（negative control）：打乱 RH/P 视图后，一致性约束不应仍然带来同样增益。

#### 为什么可能改善

SMVCI 同时利用两个事实：受控扰动可能提供新的雅可比方向，而目标组分在多个视图之间保持不变。因此，它有机会利用无标签真实数据校正表示。这个思路也适用于重复测量谱学（repeated-measures spectroscopy）、多角度成像、多传感器融合，以及同一样本在多种实验条件下的联合反演。

#### 失效条件

- 采样期间真实组分发生变化，共享潜变量假设不再成立；
- 不同视图实际只是在重复同一灵敏度方向，没有提供新的条件信息；
- 模型通过 `mixture_id`、时间戳或采样顺序泄漏目标；
- 各视图自己的编码器容量过大，把组分信息偷偷藏进干扰潜变量。

#### 最小证伪实验

执行四个负对照：错误配对、时间顺序打乱、单视图、屏蔽受控条件标签。只有正确配对相对这些对照，在测试集、OOD 条件和无标签真实数据的后验预测检验上都得到改善，才能把收益归因于共享潜变量机制。如果错误配对也能带来同样改善，应优先怀疑数据泄漏或普通正则化效应。

## 5. 六个构想的横向比较

| 构想            | 主要作用层级     | 能否改变 CRB       | 是否需要新硬件或新采样  | 主要数据需求        | 通用价值           | 首要风险           |
| ------------- | ---------- | -------------- | ------------ | ------------- | -------------- | -------------- |
| A NP-VPNet    | 求解器与干扰参数消元 | 否，只能逼近下界       | 不需要          | 仿真数据与标签       | 小样本、可解释、优化稳定   | 把数值求解改进误写成信息增益 |
| B RG-cOED     | 测量条件选择     | 能              | 需要可调频、压力或湿度  | 可微前向模型与不确定性范围 | 围绕目标参数分配测量资源   | 设计过度依赖不准确的仿真器  |
| C SIMD-MRS    | 激励方式与噪声协方差 | 能              | 需要同步多正弦与参考通道 | 台架协方差实测       | 动态频谱测量与共模消除    | 共模噪声假设未经实测     |
| D CC-SBI      | 后验分布与拒绝机制  | 否，只能更诚实地利用信息   | 不需要          | 大量仿真采样        | 多峰后验、缺频和快速在线推断 | 仿真器与真实系统不一致    |
| E OMD-GreyBox | 前向模型校准     | 间接作用，修正错误的 `J` | 通常不需要        | 少量真实或高保真数据    | 仿真到真实迁移与参数解释   | 模型偏差再次与目标参数混淆  |
| F SMVCI       | 多视图联合反演    | 条件成立时可以        | 需要成对扰动协议     | 配对视图，可使用弱标签   | 重复测量与无标签真实数据   | 目标在视图间不再保持不变   |

这张表不是在挑选“最强网络”。B 和 C 尝试增加可用信息，A 和 D 负责把已有信息反演得更充分、更可信，E 用来防止仿真偏差带歪结论，F 则把受控多条件和无标签真实数据变成额外约束。

## 6. 组合路线与优先级

### 6.1 推荐的结构化组合

```text
前向可信度审计 E
       |
       v
稳健选频/选条件 B ---> 同步多正弦与协方差识别 C
       |                              |
       +--------------+---------------+
                      v
             NP-VarPro 求解器 A
                      |
             +--------+--------+
             v                 v
          校准后验 D          多视图一致性 F
```

这条路线遵守“先信息、后模型”的项目惯例：

1. 先做不生成波形的 B 和 E 审计。用高、低保真前向模型和稳健 c 最优设计，判断是否真的存在优于固定 K4 的测量条件；如果没有，就停止自适应选频路线。
2. 再做小型台架试验验证 C。实测跨频协方差，确认共模误差是否存在；如果不存在，就不启动多正弦深度学习。
3. 有了新的观测协议后再做 A。H1、高斯--牛顿法和 NP-VPNet 形成从解析求解到有限学习的清晰对照。
4. D 是不确定度层，不是“过门器”。如果后验区间变宽但覆盖率更正确，这也是合法结果。
5. E 和 F 必须使用真实数据或更高保真配对数据。仅在仿真器内部自洽，不能证明仿真到真实的性能得到改善。

### 6.2 两种不同的“优先级”

按改变信息上限的能力排序，可粗略写成 `B ≈ C > F > A ≈ D`。E 的作用是修正错误的前向模型，因此不适合与其余方法放在同一条轴上比较。

按当前条件下的低成本可验证性排序，则是 `A > B > D > E > F > C`。不过，容易验证不等于最可能突破。A 和 D 适合先检查现有算法离理论下界还有多远；真正有机会降低下界的主候选仍是 B 和 C。

## 7. 预注册实验矩阵

| 假设  | 唯一新增机制          | 冻结对照          | 主要指标              | 必做负对照         | 停止条件        |
| --- | --------------- | ------------- | ----------------- | ------------- | ----------- |
| H-A | 变量投影与展开的高斯--牛顿法 | H1 与纯高斯--牛顿法  | CRB 效率、OOD MAE    | 干扰参数真值、打乱干扰参数 | 不优于纯高斯--牛顿法 |
| H-B | 稳健 c 最优设计       | 固定 K4         | 留出工况的最坏 CRB       | 随机 K4、只对中心点优化 | 留出工况没有改善    |
| H-C | 共模多正弦           | 顺序 K4         | 实测 `Sigma`、CRB、互调 | 独立抖动仿真        | 共模特征不显著     |
| H-D | 经校准的后验          | H1 与 NP-VPNet | 覆盖率、NLL、SBC       | 未校准的归一化流      | 覆盖率没有改善     |
| H-E | 正交模型偏差          | 无偏差模型、自由残差    | 真实工况偏差、覆盖率        | 自由残差          | 真实留出工况没有改善  |
| H-F | 共享潜变量视图         | 直接拼接          | 配对 OOD、预测检验       | 错误配对、顺序打乱     | 错误配对也能改善    |

所有实验都应继续遵守项目通则：增益必须同时出现在测试集和 OOD 条件；真值数据只能用于审计；任何泄漏负对照一旦触发，先冻结该次运行。由于 0.4 vol% 已经降为参考标注，新的通过/失败数值必须在新计划中重新预注册，不能沿用旧门，更不能看到结果后再调整门值。

## 8. 关键开放问题

1. 跨频误差究竟彼此独立，还是主要来自共模？这是构想 C 能否成立的关键，也是 MRS-2“逐频独立 3 us”假设最值得用台架试验复核的部分。
2. 弛豫模型的误差是否恰好沿着 O2 灵敏度方向？如果是，简单的正交模型偏差可能低估真实不确定度，需要更高保真物理或额外参考气体。
3. 受控 RH/压力变化期间，组分能否近似不变？如果稳定时间内存在真实漂移，构想 F 的共享潜变量会产生系统偏差。
4. 自适应设计的控制开销是否抵消信息收益？目标函数必须包含切换时间、稳态等待时间和总能量，不能只看费舍尔信息。
5. 真实 O2 窄窗数据仍然缺失。ARGD 的 CO2/CH4/N2 结果不能直接支持 O2/N2 窄窗精度；它只能证明“弛豫谱与深度学习结合”并非空白。

## 9. 结论

研究问题一的答案是：算法上限由目标方向的有效费舍尔信息决定。改变实验设计 `d`、噪声协方差 `Sigma`，或加入经过真实标定的先验，才可能降低 CRB；只更换反演网络，只能让结果更接近这个下界。

研究问题二的答案是：最有通用性的改进不是某个专用主干网络，而是六个可以组合的模块，即变量投影展开求解、稳健 c 最优设计、同步多正弦共模建模、覆盖率校准的 SBI、正交模型偏差，以及共享潜变量的多视图推断。它们可以迁移到稀疏谱学、阻抗测量、成像和数字孪生反问题。

研究问题三的答案是：每个构想都必须配套反例、负对照和停止条件。对 tv3 而言，近期最值得先做的是不生成波形的稳健 c 最优设计、NP-VarPro 数值审计，以及低成本台架协方差识别。在共模结构、真实配对数据或模型偏差得到证据支持之前，不应直接启动新的大规模深度学习训练。

综合来看，下一轮工作的重点不应再是“弛豫谱用了什么网络”，而应转向四个更基础的问题：怎样设计对目标有用的观测，怎样消除干扰参数，怎样校准后验不确定度，以及怎样约束仿真器的模型偏差。这个定位比“首次把深度学习用于 MRS”更准确，也更容易迁移到其他反问题。

## 参考文献

[1] Ming Zhu, Tingting Liu, Xiangqun Zhang, et al., “A simple measurement method of molecular relaxation in a gas by reconstructing acoustic velocity dispersion,” Measurement Science and Technology, 2018.

[2] Xiangqun Zhang, Shu Wang, Ming Zhu, “Locating the inflection point of frequency-dependent velocity dispersion by acoustic relaxation to identify gas mixtures,” Measurement Science and Technology, 2020.

[3] Shuangling Liu, Jie Mei, Ming Zhu, et al., “Identifying gas mixtures based on acoustic relaxation spectroscopy and attention recurrent neural network,” Results in Physics, 2023.

[4] Yefim Dain, Richard M. Lueptow, “Acoustic attenuation in three-component gas mixtures—Theory,” Journal of the Acoustical Society of America, 2001.

[5] Sally G. Ejakov, Scott Phillips, Yefim Dain, et al., “Acoustic attenuation in gas mixtures with nitrogen: Experimental data and calculations,” Journal of the Acoustical Society of America, 2003.

[6] R. M. Fewster, P. E. Jupp, “Information on parameters of interest decreases under transformations,” Journal of Multivariate Analysis, 2013.

[7] Gene H. Golub, Víctor Pereyra, “Separable nonlinear least squares: the variable projection method and its applications,” Inverse Problems, 2003.

[8] Malena I. Español, Mirjeta Pasha, “Variable projection methods for separable nonlinear inverse problems with general-form Tikhonov regularization,” Inverse Problems, 2023.

[9] Vishal Monga, Yuelong Li, Yonina C. Eldar, “Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing,” IEEE Signal Processing Magazine, 2021.

[10] Guillaume Sagnol, “Computing Optimal Designs of Multiresponse Experiments Reduces to Second-Order Cone Programming,” arXiv:0912.5467, 2009.

[11] Keyi Wu, Peng Chen, Omar Ghattas, “An Efficient Method for Goal-Oriented Linear Bayesian Optimal Experimental Design: Application to Optimal Sensor Placement,” arXiv:2102.06627, 2021.

[12] Alen Alexanderian, Ruanui Nicholson, Noémi Petra, “Optimal Design of Large-Scale Nonlinear Bayesian Inverse Problems under Model Uncertainty,” arXiv:2211.03952, 2022.

[13] Benjamin Sanchez, Cristian R. Rojas, Gerd Vandersteen, et al., “On the calculation of the D-optimal multisine excitation power spectrum for broadband impedance spectroscopy measurements,” Measurement Science and Technology, 2012.

[14] Johan Schoukens, Rik Pintelon, Gerd Vandersteen, “A sinewave fitting procedure for characterizing data acquisition channels in the presence of time base distortion and time jitter,” IEEE Transactions on Instrumentation and Measurement, 1997.

[15] Kyle Cranmer, Johann Brehmer, Gilles Louppe, “The frontier of simulation-based inference,” Proceedings of the National Academy of Sciences, 2020.

[16] Jan-Matthis Lueckmann, Jan Boelts, David S. Greenberg, et al., “Benchmarking Simulation-Based Inference,” Proceedings of AISTATS, 2021.

[17] Maciej Falkiewicz, Naoya Takeishi, Imahn Shekhzadeh, et al., “Calibrating Neural Simulation-Based Inference with Differentiable Coverage Probability,” Advances in Neural Information Processing Systems, 2023.

[18] Sean Talts, Michael Betancourt, Daniel Simpson, et al., “Validating Bayesian Inference Algorithms with Simulation-Based Calibration,” arXiv:1804.06788, 2018.

[19] Marc C. Kennedy, Anthony O’Hagan, “Bayesian calibration of computer models,” Journal of the Royal Statistical Society: Series B, 2001.

[20] Matthew Plumlee, “Bayesian Calibration of Inexact Computer Models,” Journal of the American Statistical Association, 2017.

[21] Matthew Plumlee, V. Roshan Joseph, “Orthogonal Gaussian process models,” Statistica Sinica, 2018.

[22] Luigi Gresele, Paul K. Rubenstein, Arash Mehrjou, et al., “The Incomplete Rosetta Stone Problem: Identifiability Results for Multi-view Nonlinear ICA,” Proceedings of UAI, 2020.

[23] Dingling Yao, Danru Xu, Sébastien Lachapelle, et al., “Multi-View Causal Representation Learning with Partial Observability,” Proceedings of ICLR, 2024.
