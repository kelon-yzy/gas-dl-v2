<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-04 -->
<!-- file-state: revision=40 | updated-at=2026-06-04T08:58:49+08:00 | writer-id=Codex | base-workspace-revision=79 -->

<!-- section: current_state -->
- 2026-06-04：benchmark/HITRAN cache 数据集生成性能优化已提交（commit 7703e99）；正式 512 序列标准数据集尚未生成。
- 生成链路现支持 sequence chunk 多进程、worker 临时 chunk、主进程 memmap 顺序合并、staging 原子发布、HITRAN cache 并行预计算与谱 cache 原子写；memmap 数据集可在生成后单独打包 waveform_sequence.npz。
- 2026-06-03：长时序协议 S0-S5、时序改造审查修复及专利基准对齐的 DynamicStackingSVRRegressor 已落地。
- DL 已覆盖 CNN/TCN/LSTM/Transformer/PatchTST；ML protocol 已覆盖 full/per-phase/early，并支持 dynamic_stacking_svr CLI。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约和可复现实验链路，不追求完整硬件仪器级标定。
- 正式主线保持 mixture_id 语义，不回退或重写为 sequence_id。
- HITRAN benchmark 保持 cache-only：生成阶段不隐式联网、不补写缓存，缺 cache 必须显式失败。
- ml 与 dl 的共享逻辑以 src/common 为单一真相源。
- 动态 stacking 缺少 feature_names 或声/光/热任一模态时显式失败，不做 ridge fallback。
- 并行随机源按 (seed, global_sequence_index, stream_name) 稳定派生；大规模数据集使用 storage=memmap，npz 压缩包生成后单独打包。

<!-- section: risks_open_questions -->
- 正式数据集 data/wv4-formal-hitran-standard-512 尚未生成；现有 data/hitran_cache 约 201 个 .npz 不足以覆盖，seed=20260603 的预计算大概率未跑。
- 尚未实际生成 long/multi_pulse 大规模数据集并跑完整 DL、ridge 与 dynamic_stacking_svr 对比。
- outputs/runs 下旧 HITRAN 数据集使用旧单时间常数 V_TCS，不能与新代码结果做 V_TCS 数值对比。
- 真实硬件标定参数与 TraceGas-HC-NDIR datasheet 仍缺失；外部 PNNL/NIST 定量谱数据未导入。
- DL Trainer 尚无 argparse CLI 入口。

<!-- section: next_step -->
- 按 docs/生成正式 HITRAN 标准数据集计划.md 先运行 precompute_hitran_benchmark_cache（--sequences 512 --seed 20260603 --workers 24），再生成 wv4-formal-hitran-standard-512。
- 按 Validation Plan 校验 slow [512,512,8]、ultrasonic [512,512,1000]、fiber_mic [512,512,2000]、y [512,4]、splits 358/76/51/27 与 quality status=pass。
- 正式数据集就绪后运行 ml.cli --protocol、ridge 与 dynamic_stacking_svr，并执行 CNN/TCN/LSTM/Transformer/PatchTST 对比实验。
- 为 DL Trainer 增加 argparse CLI，并在 wv4-smoke 上跑通 CNN1D/TCN 训练。

<!-- section: recent_pivots -->
- 2026-06-04：提交 benchmark/HITRAN cache 生成性能优化（commit 7703e99，14 文件 +766/-127）；性能相关 29 passed、全量 182 passed。
- 2026-06-03：新增 dynamic_stacking_svr，完成声/光/热三模态独立 RBF-SVR、MC 漂移 MSE 动态权重和 Ridge 元学习器融合。
- 2026-06-03：统一 HITRAN 后端多时间常数动力学并落实时序改造审查修复，175 passed。
- 2026-06-03：长时序协议 S0-S5 代码闭环落地。
