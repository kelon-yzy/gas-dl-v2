<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-28 -->
<!-- file-state: revision=30 | updated-at=2026-05-28T16:39:32+08:00 | writer-id=Codex | base-workspace-revision=57 -->

<!-- section: current_state -->
- 2026-05-28 声学链路改造已按 docs/当前声学链路问题.md 完成工程落地，并同步 README、架构、实施计划、光学链路说明和当前问题文档。
- 超声链路已从 simplified_tof_proxy_v1 升级为 tof_observed_transducer_proxy_v1：生成真值 TOF、观测 TOF、固定系统/电缆延迟、延迟修正、触发抖动、二阶谐振换能器响应、估计声速、TOF 质量和接收标志。
- 光纤链路已从 acoustic_proxy_v1 升级为 fiber_interferometric_proxy_v1：新增 FiberProbeSpec，建模探头声压、声反射、压力/腔长到相位转导、光链路损耗、线性相位解调、电噪声、饱和和固定 DAQ 量化。
- 声衰减模型已升级为 semi_empirical_multigas_relaxation_proxy_v2，并在 hidden_attenuation_v2 中加入 N2 背景弛豫项和对应元数据。
- 新增超声观测派生数组 ultrasonic_tof_observed_s、ultrasonic_sound_speed_estimated_m_per_s、ultrasonic_tof_quality、ultrasonic_tof_accepted，已写入 npy/memmap、waveform_sequence.npz、manifest/metadata 和完整性校验。
- 验证结果：python -m pytest tests 通过，145 passed；git diff --check 通过，仅存在 LF 到 CRLF 的 Git 提示。

<!-- section: active_judgments -->
- 第一阶段仍聚焦核心契约和可复现实验链路，不追求完整硬件仪器级标定。
- 当前声学实现是可校准仿真代理模型，不宣称已经等价于真实超声硬件或真实光纤干涉测量系统。
- 正式主线保持 mixture_id 语义，不回退或重写为 sequence_id；benchmark 不恢复 base_condition_id、noise_seed_index、noise_seed 依赖。
- HITRAN benchmark 仍保持 cache-only 原则：生成阶段不隐式联网、不补写缓存，缺 cache 必须显式失败。
- 新增声学资产已进入打包和完整性校验，下游 dl/ml 消费者应以新契约为准。

<!-- section: risks_open_questions -->
- 真实硬件标定参数仍缺失，包括超声系统延迟、换能器频响、前端响应、触发同步、光纤探头灵敏度、光电探测器、放大器和 DAQ 实测参数。
- 既有已生成数据集不包含新的声学字段和 metadata，需要重生成后才能代表当前正式契约。
- 新增声学观测字段虽然已通过生成、打包和完整性测试，但尚未跑完整训练管线验证 DL/ML 下游特征选择和报告兼容性。
- TraceGas-HC-NDIR 目标 datasheet 仍缺失；当前 CH4/CO2 带宽参数仍是行业参考占位。
- 外部 PNNL/NIST 或仪器定量谱数据仍未导入，当前外部 CSV sanity-check 路径需要真实数据补齐。

<!-- section: next_step -->
- 重新生成当前正式 smoke benchmark，确认 manifest、metadata、npz/npy 产物包含新的声学字段。
- 跑最小 DL/ML 下游冒烟流程，确认新增字段不会破坏数据读取、特征提取、训练评估和报告汇总。
- 如进入仪器级建模，先收集并文档化超声与光纤硬件标定参数，再升级模型名，避免把代理模型误写为实测模型。

<!-- section: recent_pivots -->
- 2026-05-28: 完成 docs/当前声学链路问题.md 第 9 节方案的代码落地和实施审核，明确当前边界是可校准代理模型。
- 2026-05-28: 超声链路加入观测 TOF、延迟修正、触发抖动、换能器响应、估计声速、TOF 质量和接收标志。
- 2026-05-28: 光纤链路加入 FiberProbeSpec、探头声压、相位/腔长转导、光链路损耗、线性解调、电噪声、饱和和 DAQ 量化。
- 2026-05-28: 声衰减模型加入 N2 背景弛豫项，测试期望和文档同步更新。
- 2026-05-28: 补充 tests/test_waveform_physics.py，并更新 benchmark、声学物理回归测试；全量 tests 145 passed。
