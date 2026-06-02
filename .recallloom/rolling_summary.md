<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Claude] | 2026-06-02 -->
<!-- file-state: revision=33 | updated-at=2026-06-02T14:19:20+08:00 | writer-id=Claude | base-workspace-revision=62 -->

<!-- section: current_state -->
- 2026-05-29：重新生成正式 smoke benchmark data/wv4-smoke（32 sequences，LHS，hitran_hapi_v1 cache-only），validation pass；2026-05-28 落地的声学链路改造字段已端到端写入正式产物。
- 已校验 8 个超声派生数组（tof_s/tof_observed_s/peak_index/sound_speed_m_per_s/sound_speed_estimated_m_per_s/alpha_true_npm/tof_quality/tof_accepted）落盘并进入 waveform_sequence.npz；manifest 与 metadata/waveform_spec 声学模型名齐全。
- DL/ML 下游冒烟通过：ML 三模态（140 维特征）ridge 训练 + 四 split 评估 + 按组分 component_metrics；DL V4BenchmarkDataset(slow,NCT,scaler) + CNN1D 前向 (4,4)，三模态 NTC 拼接 (128,3008)；新字段未破坏数据读取、特征提取、训练评估与报告汇总链路。
- 当前阶段：Phase 1 核心契约与可校准声学代理链路稳定，下游可直接消费含新声学字段的契约数据集。
- 2026-06-02：完成 Karpathy 代码审查与报告 P0/P1/P2 全部修复——P0 去重重构（src/common  + shared heads），P1 补齐验证通道（ML CLI ridge/mean 基线 + DL Trainer fit→eval→checkpoint），P2 dtype 注释 + RegressorProtocol 替换；新增 .gitattributes 统一 LF；全量测试 145 passed。
- ML CLI：`python -m ml.cli --dataset-dir data/wv4-smoke --model ridge`，ridge train R²=0.93, val R²=0.24 (smoke 32 条)；支持 --json、--modalities、--scaler-path。
- DL Trainer：`dl.training.trainer.Trainer` 完整 fit→eval→checkpoint 闭环，通过 checkpoint roundtrip 验证；`build_optimizer` 支持 adam/adamw/sgd。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约和可复现实验链路，不追求完整硬件仪器级标定。
- 当前声学实现是可校准仿真代理模型，不宣称等价于真实超声硬件或真实光纤干涉测量系统。
- 正式主线保持 mixture_id 语义，不回退或重写为 sequence_id。
- HITRAN benchmark 保持 cache-only：生成阶段不隐式联网、不补写缓存，缺 cache 必须显式失败。
- 新增声学派生数组对 DL/ML 是额外文件，现有消费者（V4BenchmarkDataset、ml.features）按既有路径读取即兼容新契约，无需改动。
- ml 与 dl 的共享逻辑（指标数据结构、scaler 加载/应用、split 工具）以 src/common 为单一真相源；新增同类共享逻辑应放 common。
- DL Trainer 为最小闭环设计（单卡、无分布式），等训练规模扩大后再评估是否需要 LR scheduler/early stopping/多 GPU。

<!-- section: risks_open_questions -->
- 真实硬件标定参数仍缺失（超声系统延迟、换能器频响、前端响应、触发同步、光纤探头灵敏度、光电探测器、放大器、DAQ 实测参数）。
- TraceGas-HC-NDIR 目标 datasheet 仍缺失；当前 CH4/CO2 带宽参数仍是行业参考占位。
- 外部 PNNL/NIST 或仪器定量谱数据仍未导入，外部 CSV sanity-check 路径需真实数据补齐。
- smoke 规模（32 条）下 DL/ML 泛化指标（val/extrapolation R² 为负）无参考意义，正式基线需更大规模数据集。
- DL Trainer 尚无 argparse CLI 入口（configs/train/ 为空 .gitkeep），训练配置仅能通过 Python API 使用。

<!-- section: next_step -->
- 在更大规模数据集上复跑 DL/ML 建立可比基线（ridge train R²=0.93 可作为当前 smoke 参考值）。
- 为 DL Trainer 加 argparse CLI（--dataset-dir/--model/--epochs/--output-dir），在 wv4-smoke 上跑通 CNN1D/TCN 训练并记录指标。
- 如进入仪器级建模：先收集并文档化超声/光纤硬件标定参数，再升级模型名，避免把代理模型误写为实测模型。
- 导入外部定量谱（PNNL/NIST）并补齐 sanity-check 路径。
- 评估是否在 Trainer 中加入 LR scheduler 与 early stopping（当前可先保持极简）。

<!-- section: recent_pivots -->
- 2026-06-02 PM：Karpathy 审查 P1/P2 修复——ML CLI（ml.cli）+ DL Trainer（trainer.py）+ dtype 注释 + RegressorProtocol 替换，145 tests passed。
- 2026-06-02 AM：Karpathy 审查 + 报告 P0 去重（src/common 单一真相源 + 共享 head）+ CRLF 根治（.gitattributes/全树 LF），测试 145 passed；属代码质量重构，实验契约不变。
- 2026-05-29：重生成 wv4-smoke（含新声学字段）并完成 DL/ML 下游兼容性冒烟，确认新契约端到端可用。
- 2026-05-28：完成声学链路改造代码落地，明确当前边界是可校准代理模型。
