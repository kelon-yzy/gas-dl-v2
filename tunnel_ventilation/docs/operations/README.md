# Operations 文档

| 文档 | 责任 |
| --- | --- |
| [server_training_guide.md](server_training_guide.md) | 服务器环境、数据同步、训练、产物回收；**§4.5 波形 `waveform_preprocess`（gpu/cpu）** |

测试与验证要求见[代码契约事实源](../掘进通风代码契约事实源.md) §14；当前活跃执行线的命令见 [active/README.md](../active/README.md) 指向的专项计划。

波形训练吞吐相关：正式配置默认 `waveform_preprocess: "gpu"`（int16 + scale 上卡后 dequant/normalize）；细节与回退见 [server_training_guide.md §4.5](server_training_guide.md#45-波形数据通路-waveform_preprocessp1-吞吐)。