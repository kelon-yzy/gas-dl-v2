# 现有观测模态 O₂ 识别算法文献综述

> 更新：2026-07-10
> 范围：TDLAS O₂ 硬件暂缓期间，使用现有超声、慢通道与环境观测量提升 CO₂/O₂/N₂ 的识别能力。
> 结论边界：本文不把文献中的独立硬件精度直接迁移为 tv3 指标，也不改变项目的 raw3、无 oracle 特征与无闭包回填约束。

## 项目事实与问题定义

tv3-formal-6000 的可部署基线是 D0-observed Ridge：O₂ `R²` 为 val 0.4226、test 0.4571、extrapolation 0.3708。O₂ 的主要有效信息来自观测 TOF；去除 TOF 后，O₂ 接近均值预测。窄 O₂ 分箱在 oracle 特征下仍为负 `R²`，因此算法的现实目标是提高大档位与跨 split 识别，而不是承诺突破 0.8% 窗口的物理分辨率墙。

R5' TabPFN 在同一 864 维 observed 特征上达到 val O₂ 0.6673，证明非线性存在，但其三组分独立回归的 `sum_abs_error` 约为 0.16，不能直接部署。R5 小 MLP 的 val O₂ 为 -0.1834，未承接这一增益。故当前研究问题收敛为：在不新增硬件、不使用真值声速、真值 TOF 或真值衰减的前提下，可部署模型能否利用 TOF 与 `T_C/P_MPa/H_RH/L_m` 的联合信息获得稳定的非线性增益。

## 中文文献

### 超声浓度测量与环境补偿

1. 李博宸. *具有温湿度补偿功能的超声波甲烷检测技术*. 哈尔滨理工大学, 2018.
   - CNKI 摘要明确建立了浓度、声速、相位差、温度和湿度之间的关系，并将温湿度补偿用于超声浓度测量。
   - 对 tv3 的含义：`T_C` 与 `H_RH` 不能仅作为弱辅助统计量，应与 observed TOF 或 estimated sound speed 联合建模。

2. 丁欣. *双声道超声波气体传感器的构型优化及检测方法研究*. 哈尔滨理工大学, 2024.
   - CNKI 摘要将精度低、环境影响大列为超声气体检测的核心工程问题，研究管道声场、多周期相位差和环境温湿度补偿。
   - 对 tv3 的含义：算法应优先利用现有的相位或 TOF 质量特征与环境变量，而不是回到已证伪的原始波形端到端路径。

3. 孙慧. *基于声学特性的多组分气体浓度检测机理及方法研究*. 哈尔滨理工大学, 2021.
   - CNKI 摘要提出声速和声衰减与多组分浓度的关系模型，并讨论多频相位差测量。
   - 对 tv3 的含义：TOF 为主通道、TCS 为边际通道的项目诊断与声学多组分检测机理一致；不应把 TCS 当作替代 O₂ 直接通道。

### 漂移补偿的适用边界

4. 陈浩天. *基于子空间迁移学习的气体传感器漂移补偿方法研究*. 吉林大学, 2025.
   - CNKI 摘要基于跨批次时序传感器数据研究域适应和漂移补偿。
   - 对 tv3 的含义：它支持未来现场跨设备、跨季节校准的研究方向，但当前 tv3 只有受控仿真 split，尚无独立无标签目标域或跨批次漂移数据，不能把域适应作为本轮正式对照。

## 英文文献

1. Fukuoka H, Taskin M, Teii K, Kato Y. Measurement of oxygen concentration in atmospheric air using ultrasound time of flight with humidity compensation. *Review of Scientific Instruments*. 2023;94(3). doi:10.1063/5.0113877.
   - 该研究指出空气与 O₂ 的声速差很小，必须计算补偿温度和湿度影响；其近大气空气实验的补偿后误差约为 0.4% 或更低。
   - 适用限制：该结果针对其仪器与近大气二元条件，不能直接作为 tv3 精度承诺；但它直接支持将 `T_C/H_RH` 作为 TOF-O₂ 的条件变量。

2. Zhan X, Yang Y, Liang J, Shi T, Li X. Temperature effects and compensation in ultrasonic concentration measurement of multicomponent mixture. *Sensors and Actuators A: Physical*. 2016. doi:10.1016/j.sna.2016.10.036.
   - 主题直接覆盖多组分超声浓度测量中的温度效应与补偿。
   - 对 tv3 的含义：应比较“原始 observed 统计特征”与“环境条件化特征”，而不是只调整回归器容量。

3. Matsuda M, Takakura Y, Nakabayashi Y, Morioka T. Ultrasonic gas flow and concentration measurement in hydrogen and nitrogen gas mixtures. *Measurement: Sensors*. 2025;11:101535. doi:10.1016/j.measen.2024.101535.
   - 该研究验证了超声同时用于气流与气体浓度测量的相邻应用场景。
   - 对 tv3 的含义：`L_m`、TOF 和流动工况的耦合不应被独立统计特征完全掩盖，后续可做同步条件化特征消融。

4. Yao N, Ma T, Lin W, et al. Acoustical modeling and analysis of proton-exchange membrane fuel cell stack anode exhaust gas from the perspective of ultrasound testing. *Clean Energy*. 2025. doi:10.1093/ce/zkae086.
   - 研究范围包含温度、压力、湿度对超声气体测试的影响。
   - 对 tv3 的含义：`P_MPa` 同样应保留为补偿条件；在没有独立湿度或压力观测时，不能期望 TOF 单独消除这些混杂量。

5. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on tabular data? 2022. doi:10.48550/arXiv.2207.08815.
   - 该基准研究说明，在典型中小样本表格任务上，树模型常优于默认深度网络。
   - 对 tv3 的含义：R5 MLP 的失败不等于 observed 特征无效；应先用受正则化的树模型验证可部署非线性容量。

## 算法路线判断

| 路线 | 文献与项目依据 | 本轮状态 | 不变量 |
|---|---|---|---|
| R7 ExtraTrees observed | 小样本表格非线性与环境条件联合建模 | 代码和本地 smoke 已完成，正式 6000 待服务器 | 864 维 D0-observed；raw3；无 oracle；不做闭包回填 |
| R5 目标标准化 MLP | 表格 MLP 的目标尺度优化 | 配置和单元测试已完成，正式 6000 待服务器 | 只改变训练损失尺度，预测反变换回原始百分比 |
| 显式环境条件化特征 | 超声 TOF 温湿压补偿文献 | R7 后的独立消融 | 不硬编码未知混合声速公式；逐项确认输入可观测 |
| 域适应或漂移补偿 | 跨批次传感器漂移文献 | 暂缓 | 需要真实跨批次或无标签目标域数据 |
| 原始超声波形端到端 | 项目 R1b/D2 已证伪 | 关闭 | 不重启 MiniRocket raw 或 TOF-PhaseNet |

## R7 服务器验收

先运行以下单一配置：

```bash
python -s -m tv3.pipeline.run_tv3_extratrees_baseline --config configs/tv3_r7_extratrees_observed.json
```

与 D0-observed Ridge 比较 train、val、test、extrapolation 四个 split，并记录 `sum_abs_error` 与 O₂ 分箱指标。通过条件为：

1. val O₂ `R² >= 0.4726`，即相对 D0 至少 +0.05。
2. test 与 extrapolation 也同步优于各自的 D0 基线，不能只在 val 提升。
3. `sum_abs_error` 必须一并报告；R7 虽直接预测 raw3，但不强制三组分闭包。

若 R7 未通过，不将失败解释为声学信息消失。下一步是把 observed TOF、estimated sound speed 与每个时间步的 `T_C/P_MPa/H_RH/L_m` 组合成可解释的条件化特征，再做与 R7 完全相同的树模型对照；不得混入任何真值物理数组。

## 检索记录

- CNKI：`超声波 气体浓度 温度补偿`，19 条结果；以上选取 3 篇与多组分、温湿度补偿或相位测量直接相关的论文。
- CNKI：`气体传感器 漂移补偿 迁移学习`，18 条结果；选取 1 篇用于界定域适应的前置数据要求。
- CNKI：`掘进通风 氧气浓度 监测` 本次页面返回“暂无数据，请稍后重试”，未据此补写结论。
- CrossRef / OpenAlex：核验英文论文 DOI；初始 arXiv 源返回 429，后续以 OpenAlex 精确题名检索补充元数据。
