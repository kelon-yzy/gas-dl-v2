# 掘进通风场景文献报告索引

> 本目录收集掘进通风 CO₂/O₂/N₂ 三组分检测相关的文献证据。
> 所有物性常数需可追溯到本目录下的文献来源，不得凭估计填入代码。

## 子报告

| 序号 | 文件 | 覆盖范围 |
|---:|---|---|
| 1 | [co2_o2_n2_gas_properties.md](co2_o2_n2_gas_properties.md) | CO₂/O₂/N₂ 声学、热导、光学物性参数及其文献来源 |
| 2 | [tunnel_ventilation_sensing_survey.md](tunnel_ventilation_sensing_survey.md) | 掘进通风气体检测技术综述：矿用传感器国标、商用系统、多模态融合可行性 |

## 与编码文件的对应

| 本目录文献 | 对应速查表 | 目标代码文件 |
|---|---|---|
| `co2_o2_n2_gas_properties.md` | [../physics_references.md](../physics_references.md) | `src/sim/generation/tunnel_ventilation/acoustic_physics.py` |
| `tunnel_ventilation_sensing_survey.md` | — | 场景设计与可行性判断依据 |

## 使用说明

1. `physics_references.md` 是编码速查表，只列最终采用的数值和公式。
2. 本目录下的文献报告提供详细的来源推导、多源对比和置信度评估。
3. 修改物性参数前，先查本目录下的文献报告确认来源和适用范围。
