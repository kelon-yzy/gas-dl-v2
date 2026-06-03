<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-03 -->
<!-- file-state: revision=38 | updated-at=2026-06-03T15:01:32+08:00 | writer-id=Codex | base-workspace-revision=75 -->

<!-- section: current_state -->
- 2026-06-03：长时序协议 S0-S5 已落地为可运行代码闭环；同日完成该提交的代码质量审查并落实修复。
- 2026-06-03：基于专利文档（三）算法设计，对 ML 模型完成专利基准对齐改进：新增 DynamicStackingSVRRegressor，实现声/光/热三模态独立 RBF-SVR、MC 漂移 MSE 逆不确定性动态权重和 Ridge 元学习器融合。
- sim 侧支持 short/standard/long/xlong 时间轴预设、动态 PhaseSchedule、standard_exposure/variable_onset/fast_transient/incomplete_recovery/multi_pulse 阶段 profile、stage_jitter，并把 stage_profile/stage_jitter/phase_schedule 写入 manifest.json 和 metadata/waveform_spec.json。
- DL 侧有 LSTM/Transformer/PatchTST，CNN/TCN 支持 mean/last/attention 聚合，TCN 支持 target_timesteps 自动扩展并断言 receptive_field>=target。
- ML baseline 支持基于实际 phase_id 的 per-phase 与 early-window 特征，run_baseline_protocol 与 python -m ml.cli --protocol 输出 full/per-phase/early 的 JSON 或 Markdown report；新增 dynamic_stacking_svr CLI 模型选项。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约和可复现实验链路，不追求完整硬件仪器级标定。
- 当前声学实现是可校准仿真代理模型，不宣称等价于真实超声硬件或真实光纤干涉测量系统。
- 正式主线保持 mixture_id 语义，不回退或重写为 sequence_id。
- HITRAN benchmark 保持 cache-only：生成阶段不隐式联网、不补写缓存，缺 cache 必须显式失败。
- ml 与 dl 的共享逻辑以 src/common 为单一真相源；新增同类共享逻辑应放 common。
- 动态 stacking ML 模型以专利算法设计为基准：缺少 feature_names 或缺失声/光/热任一模态时显式失败，不做 ridge fallback。

<!-- section: risks_open_questions -->
- 尚未实际生成 long/multi_pulse 大规模数据集并跑完整 DL vs ridge 对比报告。
- 动态 stacking 模型已在 wv4-smoke 冒烟通过，但该规模指标无泛化参考意义；正式指标仍需在 standard/long/multi_pulse 数据集上验证。
- outputs/runs 下旧 HITRAN 数据集仍是旧单时间常数 V_TCS，与新代码生成结果在 V_TCS 上不可数值对比。
- 真实硬件标定参数仍缺失；TraceGas-HC-NDIR datasheet 仍缺失，CH4/CO2 带宽参数仍是行业参考占位。
- 外部 PNNL/NIST 定量谱数据未导入，外部 CSV sanity-check 路径需真实数据补齐。
- DL Trainer 尚无 argparse CLI 入口，训练配置仅能通过 Python API 使用。

<!-- section: next_step -->
- 生成 standard/long/multi_pulse 数据集，先用 python -m ml.cli --protocol --report-path <path> 产出传统 ML baseline report。
- 用 python -m ml.cli --model dynamic_stacking_svr --modalities slow,ultrasonic,fiber_mic 在正式数据集上产出 full/per-phase/early baseline report，并与 ridge baseline 对比。
- 运行 CNN/TCN/LSTM/Transformer/PatchTST 对比实验，汇总 full/per-phase/early 指标，验证长序列模型相对顺序不敏感 baseline 的差距是否随 T 增大。
- 为 DL Trainer 加 argparse CLI（--dataset-dir/--model/--epochs/--output-dir），在 wv4-smoke 上跑通 CNN1D/TCN 训练并记录指标。

<!-- section: recent_pivots -->
- 2026-06-03：按专利（三）算法设计对齐 ML 模型，新增 dynamic_stacking_svr：三模态解耦 RBF-SVR、MC 漂移 MSE 动态权重、Ridge 元学习器；tests/test_ml_baselines.py 15 passed，tests/ 177 passed，wv4-smoke CLI 冒烟通过。
- 2026-06-03：审查当日时序改造提交，按方向 A 统一 HITRAN 后端多时间常数动力学，清理孤儿函数/边界塌缩/性能/worker RNG/类型标注等审查问题，175 passed。
- 2026-06-03：长时序协议 S0-S5 代码闭环落地，新增阶段调度、长时间轴、多 profile、多时间常数动态、序列模型、训练期增强和 baseline protocol。
- 2026-06-03：补齐 ml.cli --protocol，输出 full/per-phase/early baseline protocol JSON/Markdown report，并同步架构/实施计划文档。
