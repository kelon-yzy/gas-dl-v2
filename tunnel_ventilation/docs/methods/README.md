# Methods 文档

本目录保存可跨实验复用的方法资料，以及 tv3 方法学论文的写作入口。

| 文档 | 责任 |
| --- | --- |
| [tv3_论文结构与投稿方案.md](tv3_论文结构与投稿方案.md) | **论文写作入口**：定位与核心论点、章节结构与证据映射、主图表清单、候选期刊与投稿顺序、审稿风险与应对 |
| [tv3_算法方法论文说明.md](tv3_算法方法论文说明.md) | 论文素材：前向物理模型、RawDSP、反演算法族、不确定度量化、可辨识性审计的完整方法链与已验证数值；§8.4 给出代码资产耦合分层与跨场景迁移条件；§9 含 11 条投稿须声明的局限 |
| [端到端波形动态门控组分反演框架与文献证据.md](../references/端到端波形动态门控组分反演框架与文献证据.md) | EC-MSW-GatedNet 算法框架、证据边界与 P0–P4 实施顺序 |
| [波形特征提取算法评估.md](波形特征提取算法评估.md) | 高维波形算法比较与历史判断 |
| [波形特征提取算法代码示例.md](波形特征提取算法代码示例.md) | TOF、固定核和 wav2vec 风格示例 |
| [small_sample_dl_strategies.md](small_sample_dl_strategies.md) | 小样本训练策略与风险 |

名词与实验顺序导读已于 2026-08-16 归档至 [archive/legacy](../archive/legacy/tv3_名词与实验顺序导读.md)：契约类名词进入[代码契约事实源](../掘进通风代码契约事实源.md)，实验顺序由[实验日志](../掘进通风实验日志.md)承接。

Raw waveform 的当前优先结论以 [D2b / RawDSP 计划](../archive/completed/d2b_raw_dsp_implementation_plan.md)为准；方法文档中的早期算法排序不覆盖正式实验结果。

波形训练工程通路（dequant / z-score / 设备侧组装）见 [server_training_guide.md §4.5](../operations/server_training_guide.md#45-波形数据通路-waveform_preprocessp1-吞吐) 与 [waveform_normalization_plan.md §12](../archive/completed/waveform_normalization_plan.md#12-设备侧预处理扩展2026-07-14)。
