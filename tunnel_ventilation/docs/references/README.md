# 掘进通风场景文献报告索引

> 本目录收集掘进通风 CO₂/O₂/N₂ 三组分检测相关的文献证据。
> 所有物性常数需可追溯到本目录下的文献来源，不得凭估计填入代码。

## 子报告

| 序号 | 文件 | 覆盖范围 |
|---:|---|---|
| 1 | [co2_o2_n2_gas_properties.md](co2_o2_n2_gas_properties.md) | CO₂/O₂/N₂ 声学、热导、光学物性参数及其文献来源 |
| 2 | [tunnel_ventilation_sensing_survey.md](tunnel_ventilation_sensing_survey.md) | 掘进通风气体检测技术综述：矿用传感器国标、商用系统、多模态融合可行性 |
| 3 | [observed_o2_algorithm_review.md](observed_o2_algorithm_review.md) | TDLAS 暂缓期的 observed O₂ 识别算法综述：TOF 温湿压补偿、树模型、漂移补偿适用边界 |
| 4 | [传感器硬件资料整理.md](传感器硬件资料整理.md) | TCS205、NDIR、超声、光纤麦克风和数据采集硬件规格汇总 |
| 5 | [tv3_identifiability_business_threshold_evidence.md](tv3_identifiability_business_threshold_evidence.md) | O₂ P90 与 nuisance 比例门限的法规、文献和项目内推导；同时记录安全联锁边界 |

## 与编码文件的对应

| 本目录文献 | 对应速查表 | 目标代码文件 |
|---|---|---|
| `co2_o2_n2_gas_properties.md` | [foundation/physics_references.md](../foundation/physics_references.md) | `tv3/sim/generation/tunnel_ventilation/acoustic_physics.py` |
| `tunnel_ventilation_sensing_survey.md` | — | 场景设计与可行性判断依据 |
| `observed_o2_algorithm_review.md` | [active/r7_extratrees_implementation_plan.md](../active/r7_extratrees_implementation_plan.md) | `tv3/ml/extratrees_head.py` / `tv3/ml/extratrees_training.py` |
| `传感器硬件资料整理.md` | — | 传感器选型、信号链和采集硬件核对 |
| `tv3_identifiability_business_threshold_evidence.md` | [active/tv3_identifiability_implementation_plan.md](../active/tv3_identifiability_implementation_plan.md) | `configs/tv3_identifiability.json` / `scripts/run_tv3_identifiability.py` |

## 使用说明

1. `physics_references.md` 是编码速查表，只列最终采用的数值和公式。
2. 本目录下的文献报告提供详细的来源推导、多源对比和置信度评估。
3. 修改物性参数前，先查本目录下的文献报告确认来源和适用范围。
