# Methods 文档

本目录保存可跨实验复用的方法资料，不代表当前路线优先级。

| 文档 | 责任 |
| --- | --- |
| [tv3_名词与实验顺序导读.md](tv3_名词与实验顺序导读.md) | 初学者名词说明；按实验顺序分级（D0→D2b→B7→可辨识性→COMSOL G0/G1） |
| [端到端波形动态门控组分反演框架与文献证据.md](../references/端到端波形动态门控组分反演框架与文献证据.md) | EC-MSW-GatedNet 算法框架、证据边界与 P0–P4 实施顺序 |
| [波形特征提取算法评估.md](波形特征提取算法评估.md) | 高维波形算法比较与历史判断 |
| [波形特征提取算法代码示例.md](波形特征提取算法代码示例.md) | TOF、固定核和 wav2vec 风格示例 |
| [small_sample_dl_strategies.md](small_sample_dl_strategies.md) | 小样本训练策略与风险 |

Raw waveform 的当前优先结论以 [D2b / RawDSP 计划](../active/d2b_raw_dsp_implementation_plan.md)为准；方法文档中的早期算法排序不覆盖正式实验结果。

波形训练工程通路（dequant / z-score / 设备侧组装）见 [server_training_guide.md §4.5](../operations/server_training_guide.md#45-波形数据通路-waveform_preprocessp1-吞吐) 与 [waveform_normalization_plan.md §12](../archive/completed/waveform_normalization_plan.md#12-设备侧预处理扩展2026-07-14)。
