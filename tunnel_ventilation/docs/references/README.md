# 掘进通风场景文献报告索引

> 本目录收集掘进通风 CO₂/O₂/N₂ 三组分检测相关的文献证据。
> 所有物性常数需可追溯到本目录下的文献来源，不得凭估计填入代码。

## 子报告

| 序号  | 文件                                                                                                       | 覆盖范围                                                                           |
| ---:| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| 1   | [co2_o2_n2_gas_properties.md](co2_o2_n2_gas_properties.md)                                               | CO₂/O₂/N₂ 声学、热导、光学物性参数及其文献来源                                                   |
| 2   | [tunnel_ventilation_sensing_survey.md](tunnel_ventilation_sensing_survey.md)                             | 掘进通风气体检测技术综述：矿用传感器国标、商用系统、多模态融合可行性                                             |
| 3   | [observed_o2_algorithm_review.md](observed_o2_algorithm_review.md)                                       | TDLAS 暂缓期的 observed O₂ 识别算法综述：TOF 温湿压补偿、树模型、漂移补偿适用边界                           |
| 4   | [传感器硬件资料整理.md](传感器硬件资料整理.md)                                                                             | TCS205、NDIR、超声、光纤麦克风和数据采集硬件规格汇总                                                |
| 5   | [tv3_identifiability_business_threshold_evidence.md](tv3_identifiability_business_threshold_evidence.md) | O₂ P90 与 nuisance 比例门限的法规、文献和项目内推导；同时记录安全联锁边界                                  |
| 6   | [端到端波形动态门控组分反演框架与文献证据.md](端到端波形动态门控组分反演框架与文献证据.md)                                                       | EC-MSW-GatedNet 算法框架、文献证据与 P0–P4 实施顺序（附同名 .docx；配图 `dl框架1.png`）                |
| 7   | [tv3_acoustic_simulation_fidelity_review.md](tv3_acoustic_simulation_fidelity_review.md)                 | 声学仿真链路保真度审查；CoolProp 8.0.0 已复核四元声速/湿度灵敏度（§5.1）；含 ISO 9613-1/HITRAN/COMSOL 后续选项 |
| 8   | [文献检索_煤矿进风流安全检测_20260702.md](文献检索_煤矿进风流安全检测_20260702.md)                                             | 煤矿进风流安全检测文献检索记录（2026-07-02）                                             |
| 9   | [tv3_bidir_trigger_jitter_scenarios.md](tv3_bidir_trigger_jitter_scenarios.md)                               | F 线 F0：NI USB-6453 双情景 trigger jitter 推导（3 μs 保守 / 0.5 μs nominal 半采样上界） |

## 附件资料（非 md）

| 文件 | 说明 |
| --- | --- |
| [煤矿安全规程.pdf](煤矿安全规程.pdf) | 法规原文本地归档；O₂ ≥20% 进风流下限出处（第一百五十六条），被 `tv3_identifiability_business_threshold_evidence.md` 引用 |
| [面向煤矿掘进工作面进风流安全监测的 O₂CO₂N₂ 三元气体组分反演方法研究(1).pdf](<面向煤矿掘进工作面进风流安全监测的 O₂CO₂N₂ 三元气体组分反演方法研究(1).pdf>) | 项目论文稿（历史版本存档） |
| `dl框架1.png` | EC-MSW 框架配图，随第 6 项文档使用 |

## 与编码文件的对应

| 本目录文献                                                | 对应速查表                                                                                                     | 目标代码文件                                                                                                                        |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `co2_o2_n2_gas_properties.md`                        | [foundation/physics_references.md](../foundation/physics_references.md)                                   | `tv3/sim/generation/tunnel_ventilation/acoustic_physics.py`                                                                   |
| `tunnel_ventilation_sensing_survey.md`               | —                                                                                                         | 场景设计与可行性判断依据                                                                                                                  |
| `observed_o2_algorithm_review.md`                    | [archive/completed/r7_extratrees_implementation_plan.md](../archive/completed/r7_extratrees_implementation_plan.md) | `tv3/ml/extratrees_head.py` / `tv3/ml/extratrees_training.py`                                                                 |
| `传感器硬件资料整理.md`                                       | —                                                                                                         | 传感器选型、信号链和采集硬件核对                                                                                                              |
| `tv3_identifiability_business_threshold_evidence.md` | [archive/completed/tv3_identifiability_implementation_plan.md](../archive/completed/tv3_identifiability_implementation_plan.md) | `configs/tv3_identifiability.json` / `scripts/run_tv3_identifiability.py`                                                     |
| `tv3_acoustic_simulation_fidelity_review.md`         | [foundation/physics_references.md](../foundation/physics_references.md) §2.2 / §2.2.1                     | `tv3/sim/generation/tunnel_ventilation/acoustic_physics.py` / `tv3/sim/generation/waveforms.py`（CoolProp 为可选验收，非正式默认 backend） |

## 使用说明

1. `physics_references.md` 是编码速查表，只列最终采用的数值和公式。
2. 本目录下的文献报告提供详细的来源推导、多源对比和置信度评估。
3. 修改物性参数前，先查本目录下的文献报告确认来源和适用范围。
