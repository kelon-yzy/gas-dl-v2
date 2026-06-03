<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-03 -->
<!-- file-state: revision=36 | updated-at=2026-06-03T10:28:11+08:00 | writer-id=Codex | base-workspace-revision=71 -->

<!-- section: current_state -->
- 2026-06-03：docs/LONG_SEQUENCE_PROTOCOL_PROPOSAL_2026-06-02.md 的 S0-S5 主体已落地为可运行代码闭环。
- sim 侧已支持 short/standard/long/xlong 时间轴预设、动态 PhaseSchedule、standard_exposure/variable_onset/fast_transient/incomplete_recovery/multi_pulse 阶段 profile、stage_jitter，并把 stage_profile、stage_jitter、phase_schedule 写入 manifest.json 和 metadata/waveform_spec.json。
- 非标准阶段 profile 的慢传感器动态已改为 blend equilibrium + 多时间常数通道更新，支持不完全恢复和跨脉冲记忆效应；默认 standard_exposure 且 stage_jitter=0 的兼容路径保留。
- DL 侧已新增 LSTMRegressor、TransformerRegressor、PatchTSTRegressor；CNN/TCN 支持 mean/last/attention 聚合；TCN 支持 target_timesteps 自动扩展层数并断言 receptive_field >= target_timesteps。
- DL 数据集已支持显式 TimeSeriesAugmentConfig 做窗口切片/重采样抖动增强，默认关闭。
- ML baseline 已支持基于实际 phase_id 的 per-phase 特征窗口和 early-window 特征；run_baseline_protocol(...) 与 python -m ml.cli --protocol 可输出 full/per-phase/early baseline protocol JSON 或 Markdown report。
- docs/ARCHITECTURE.md 与 docs/IMPLEMENTATION_PLAN.md 已同步当前长时序协议、模型扩充、baseline protocol report 入口和测试状态。
- 当前验证：python -m pytest tests/test_ml_baselines.py 为 13 passed；python -m pytest 为 170 passed；git diff --check 通过时仅有 Git CRLF 规范化提示。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约和可复现实验链路，不追求完整硬件仪器级标定。
- 当前声学实现是可校准仿真代理模型，不宣称等价于真实超声硬件或真实光纤干涉测量系统。
- 正式主线保持 mixture_id 语义，不回退或重写为 sequence_id。
- HITRAN benchmark 保持 cache-only：生成阶段不隐式联网、不补写缓存，缺 cache 必须显式失败。
- 新增声学派生数组对 DL/ML 是额外文件，现有消费者按既有路径读取即兼容新契约。
- ml 与 dl 的共享逻辑以 src/common 为单一真相源；新增同类共享逻辑应放 common。
- DL Trainer 为最小闭环设计（单卡、无分布式），等训练规模扩大后再评估是否需要 LR scheduler/early stopping/多 GPU。
- 时序协议改进遵循配置驱动+向后兼容+可证伪：默认保持现状可复现，长时序优势以序列感知模型 vs 顺序不敏感强 baseline 的 R2 差距随 T 增大而扩大验收。
- Mamba/SSM 属本轮提案中的可选项，当前不引入外部依赖；先以 TCN/LSTM/Transformer/PatchTST 建立可验证闭环。

<!-- section: risks_open_questions -->
- 尚未实际生成 long/multi_pulse 大规模数据集并跑完整 DL vs ridge 对比报告。
- 真实硬件标定参数仍缺失。
- TraceGas-HC-NDIR 目标 datasheet 仍缺失；当前 CH4/CO2 带宽参数仍是行业参考占位。
- 外部 PNNL/NIST 或仪器定量谱数据仍未导入，外部 CSV sanity-check 路径需真实数据补齐。
- smoke 规模下 DL/ML 泛化指标无参考意义，正式基线需更大规模数据集。
- DL Trainer 尚无 argparse CLI 入口，训练配置仅能通过 Python API 使用。
- 跨 run 汇总、绘图和完整实验状态管理尚未落地；当前 report 入口覆盖传统 ML 单数据集 protocol。

<!-- section: next_step -->
- 生成 standard、long、multi_pulse 数据集，先用 python -m ml.cli --protocol --report-path <path> 产出传统 ML baseline report。
- 运行 CNN/TCN/LSTM/Transformer/PatchTST 对比实验，汇总 full/per-phase/early 指标，验证长序列模型相对顺序不敏感 baseline 的差距是否随 T 增大。
- 为 DL Trainer 加 argparse CLI（--dataset-dir/--model/--epochs/--output-dir），在 wv4-smoke 上跑通 CNN1D/TCN 训练并记录指标。
- 如进入仪器级建模：先收集并文档化超声/光纤硬件标定参数，再升级模型名，避免把代理模型误写为实测模型。
- 导入外部定量谱（PNNL/NIST）并补齐 sanity-check 路径。

<!-- section: recent_pivots -->
- 2026-06-03：长时序协议 S0-S5 代码闭环落地，新增阶段调度、长时间轴、多 profile、多时间常数动态、序列模型、训练期增强和 baseline protocol。
- 2026-06-03：补齐 ml.cli --protocol，可输出 full/per-phase/early baseline protocol JSON/Markdown report，并同步架构/实施计划文档；全量测试 170 passed。
- 2026-06-02：完成时序阶段划分分析 + 联网调研，产出长时序协议改进提案 docs/LONG_SEQUENCE_PROTOCOL_PROPOSAL_2026-06-02.md。
- 2026-06-02 PM：Karpathy 审查 P1/P2 修复，145 tests passed。
- 2026-06-02 AM：Karpathy 审查 + P0 去重 + CRLF 根治，测试 145 passed。
- 2026-05-29：重生成 wv4-smoke（含新声学字段）并完成 DL/ML 下游兼容性冒烟。
- 2026-05-28：完成声学链路改造代码落地，明确当前边界是可校准代理模型。
