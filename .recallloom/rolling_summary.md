<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Claude] | 2026-06-03 -->
<!-- file-state: revision=37 | updated-at=2026-06-03T13:10:24+08:00 | writer-id=Claude | base-workspace-revision=73 -->

<!-- section: current_state -->
- 2026-06-03：长时序协议 S0-S5 已落地为可运行代码闭环；同日完成该提交的代码质量审查并落实修复。
- sim 侧支持 short/standard/long/xlong 时间轴预设、动态 PhaseSchedule、standard_exposure/variable_onset/fast_transient/incomplete_recovery/multi_pulse 阶段 profile、stage_jitter，并把 stage_profile/stage_jitter/phase_schedule 写入 manifest.json 和 metadata/waveform_spec.json。
- 传感器动态：empirical 后端在 standard_exposure + stage_jitter=0 时保留旧单时间常数兼容路径（wv4-smoke 逐位重现）；其余 empirical 路径与全部 HITRAN 后端（含 V_TCS）统一走 blend equilibrium + 多时间常数动力学，支持不完全恢复与跨脉冲记忆。
- PhaseSchedule.boundaries 增加严格递增校验防止小 timesteps 下阶段塌缩；新增 resolve_timeline 整段一次计算 phase_id/blend。
- DL 侧有 LSTM/Transformer/PatchTST，CNN/TCN 支持 mean/last/attention 聚合，TCN 支持 target_timesteps 自动扩展并断言 receptive_field>=target；transformer 位置编码改用 register_buffer；数据集增强 RNG 按 DataLoader worker_id 派生。
- ML baseline 支持基于实际 phase_id 的 per-phase 与 early-window 特征，run_baseline_protocol 与 python -m ml.cli --protocol 输出 full/per-phase/early 的 JSON 或 Markdown report；phase CSV 读取已加 lru_cache 缓存。
- 当前验证：python -m pytest tests/ 为 175 passed。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约和可复现实验链路，不追求完整硬件仪器级标定。
- 当前声学实现是可校准仿真代理模型，不宣称等价于真实超声硬件或真实光纤干涉测量系统。
- 正式主线保持 mixture_id 语义，不回退或重写为 sequence_id。
- HITRAN benchmark 保持 cache-only：生成阶段不隐式联网、不补写缓存，缺 cache 必须显式失败。
- 向后兼容只承诺 empirical/wv4-smoke 逐位重现；HITRAN 后端动力学统一为多时间常数，旧 HITRAN 数据集（outputs/runs/hitran-*）不重新生成。
- ml 与 dl 的共享逻辑以 src/common 为单一真相源；新增同类共享逻辑应放 common。
- DL Trainer 为最小闭环设计（单卡、无分布式），训练规模扩大后再评估 LR scheduler/early stopping/多 GPU。
- 时序协议改进遵循配置驱动+向后兼容+可证伪：默认保持现状可复现，长时序优势以序列感知模型 vs 顺序不敏感强 baseline 的 R2 差距随 T 增大验收。
- Mamba/SSM 为本轮提案的可选项，当前不引入外部依赖；先以 TCN/LSTM/Transformer/PatchTST 建立可验证闭环。

<!-- section: risks_open_questions -->
- 尚未实际生成 long/multi_pulse 大规模数据集并跑完整 DL vs ridge 对比报告。
- outputs/runs 下旧 HITRAN 数据集仍是旧单时间常数 V_TCS，与新代码生成结果在 V_TCS 上不可数值对比。
- 真实硬件标定参数仍缺失；TraceGas-HC-NDIR datasheet 仍缺失，CH4/CO2 带宽参数仍是行业参考占位。
- 外部 PNNL/NIST 定量谱数据未导入，外部 CSV sanity-check 路径需真实数据补齐。
- smoke 规模下 DL/ML 泛化指标无参考意义，正式基线需更大规模数据集。
- DL Trainer 尚无 argparse CLI 入口，训练配置仅能通过 Python API 使用。
- 环境未安装 ruff/mypy，静态检查目前仅用 py_compile。

<!-- section: next_step -->
- 生成 standard/long/multi_pulse 数据集，先用 python -m ml.cli --protocol --report-path <path> 产出传统 ML baseline report。
- 运行 CNN/TCN/LSTM/Transformer/PatchTST 对比实验，汇总 full/per-phase/early 指标，验证长序列模型相对顺序不敏感 baseline 的差距是否随 T 增大。
- 为 DL Trainer 加 argparse CLI（--dataset-dir/--model/--epochs/--output-dir），在 wv4-smoke 上跑通 CNN1D/TCN 训练并记录指标。
- 如进入仪器级建模：先收集并文档化超声/光纤硬件标定参数，再升级模型名，避免把代理模型误写为实测模型。
- 导入外部定量谱（PNNL/NIST）并补齐 sanity-check 路径。

<!-- section: recent_pivots -->
- 2026-06-03：审查当日时序改造提交，按方向 A 统一 HITRAN 后端多时间常数动力学，清理孤儿函数/边界塌缩/性能/worker RNG/类型标注等审查问题，175 passed。
- 2026-06-03：长时序协议 S0-S5 代码闭环落地，新增阶段调度、长时间轴、多 profile、多时间常数动态、序列模型、训练期增强和 baseline protocol。
- 2026-06-03：补齐 ml.cli --protocol，输出 full/per-phase/early baseline protocol JSON/Markdown report，并同步架构/实施计划文档。
- 2026-06-02：完成时序阶段划分分析 + 联网调研，产出长时序协议改进提案 docs/LONG_SEQUENCE_PROTOCOL_PROPOSAL_2026-06-02.md。
- 2026-06-02 PM：Karpathy 审查 P1/P2 修复，145 tests passed。
- 2026-05-29：重生成 wv4-smoke（含新声学字段）并完成 DL/ML 下游兼容性冒烟。
