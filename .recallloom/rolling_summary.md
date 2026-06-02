<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Claude] | 2026-06-02 -->
<!-- file-state: revision=32 | updated-at=2026-06-02T10:35:26+08:00 | writer-id=Claude | base-workspace-revision=61 -->

<!-- section: current_state -->
- 2026-05-29：重新生成正式 smoke benchmark data/wv4-smoke（32 sequences，LHS，hitran_hapi_v1 cache-only），validation pass；2026-05-28 落地的声学链路改造字段已端到端写入正式产物。
- 已校验 8 个超声派生数组（tof_s/tof_observed_s/peak_index/sound_speed_m_per_s/sound_speed_estimated_m_per_s/alpha_true_npm/tof_quality/tof_accepted）落盘并进入 waveform_sequence.npz；manifest 与 metadata/waveform_spec 声学模型名齐全（tof_observed_transducer_proxy_v1、fiber_interferometric_proxy_v1、linear_phase_demodulation_proxy_v1、semi_empirical_multigas_relaxation_proxy_v2）。
- DL/ML 下游冒烟通过：ML 三模态（140 维特征）ridge 训练 + 四 split 评估 + 按组分 component_metrics；DL V4BenchmarkDataset(slow,NCT,scaler) + CNN1D 前向 (4,4)，三模态 NTC 拼接 (128,3008)；新字段未破坏数据读取、特征提取、训练评估与报告汇总链路。
- 当前阶段：Phase 1 核心契约与可校准声学代理链路稳定，下游可直接消费含新声学字段的契约数据集。
- 2026-06-02：完成 Karpathy 代码审查与报告 P0 去重重构——新建 src/common（RegressionMetrics/scalers/splits 单一真相源）与 dl/models/heads.py（CNN1D/TCN 共享 head），ml/dl 重复副本收敛为 re-export；新增 .gitattributes 统一 LF（根治 129 文件 CRLF 污染）；完整测试 145 passed。仅代码结构与卫生改动，未触及实验契约与数据。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约和可复现实验链路，不追求完整硬件仪器级标定。
- 当前声学实现是可校准仿真代理模型，不宣称等价于真实超声硬件或真实光纤干涉测量系统。
- 正式主线保持 mixture_id 语义，不回退或重写为 sequence_id；benchmark 不恢复 base_condition_id、noise_seed_index、noise_seed 依赖。
- HITRAN benchmark 保持 cache-only：生成阶段不隐式联网、不补写缓存，缺 cache 必须显式失败。
- 新增声学派生数组对 DL/ML 是额外文件，现有消费者（V4BenchmarkDataset、ml.features）按既有路径读取即兼容新契约，无需改动。
- ml 与 dl 的共享逻辑（指标数据结构、scaler 加载/应用、split 工具）以 src/common 为单一真相源；新增同类共享逻辑应放 common，避免再次出现 ml/dl 双份副本。

<!-- section: risks_open_questions -->
- 真实硬件标定参数仍缺失（超声系统延迟、换能器频响、前端响应、触发同步、光纤探头灵敏度、光电探测器、放大器、DAQ 实测参数）。
- TraceGas-HC-NDIR 目标 datasheet 仍缺失；当前 CH4/CO2 带宽参数仍是行业参考占位。
- 外部 PNNL/NIST 或仪器定量谱数据仍未导入，外部 CSV sanity-check 路径需真实数据补齐。
- smoke 规模（32 条）下 DL/ML 泛化指标（val/extrapolation R2 为负）无参考意义，正式基线需更大规模数据集；当前 DL 侧尚无完整训练 CLI，仅有 Dataset + 模型工厂 + losses/metrics。

<!-- section: next_step -->
- 代码侧（报告 P1）：为 train_regressor_on_dataset 加 argparse CLI（--dataset-dir/--model）并在 data/wv4-smoke 跑出 ridge/mean 基线指标；之后实现 DL Trainer（fit→eval→checkpoint）打通训练验证通道。
- 如继续实验主线：在更大规模数据集上复跑 DL/ML 建立可比基线，并评估是否补齐 DL 训练编排（Trainer/checkpoint/训练配置）。
- 如进入仪器级建模：先收集并文档化超声/光纤硬件标定参数，再升级模型名，避免把代理模型误写为实测模型。
- 导入外部定量谱（PNNL/NIST）并补齐 sanity-check 路径。

<!-- section: recent_pivots -->
- 2026-06-02：Karpathy 审查 + 报告 P0 去重（src/common 单一真相源 + 共享 head）+ CRLF 根治（.gitattributes/全树 LF），测试 145 passed；属代码质量重构，实验契约不变。
- 2026-05-29：重生成 wv4-smoke（含新声学字段）并完成 DL/ML 下游兼容性冒烟，确认新契约端到端可用。
- 2026-05-28：完成声学链路改造代码落地，明确当前边界是可校准代理模型。
- 2026-05-28：超声升级为 tof_observed_transducer_proxy_v1（观测 TOF、延迟修正、触发抖动、换能器响应、估计声速、TOF 质量/接收标志）。
- 2026-05-28：光纤升级为 fiber_interferometric_proxy_v1（FiberProbeSpec、相位/腔长转导、线性解调、电噪声、饱和、DAQ 量化）。
- 2026-05-28：声衰减升级为 semi_empirical_multigas_relaxation_proxy_v2（新增 N2 背景弛豫项）。
