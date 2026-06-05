<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-05 -->
<!-- file-state: revision=42 | updated-at=2026-06-05T09:45:53+08:00 | writer-id=Codex | base-workspace-revision=83 -->

<!-- section: current_state -->
- 2026-06-05：DL 训练 CLI 已落地，python -m dl.cli 支持单数据集训练、checkpoint、run_config 与 metrics JSON 输出；根层 dl launcher 支持从仓库根目录运行。
- 2026-06-05：新增 v4 兼容 cnn1d_tcn_fusion 模型，支持 slow/ultrasonic/fiber_mic 三路编码、TCN 融合、last+mean+max 池化与 bounded simplex 输出约束。
- DL 模型注册表已覆盖 CNN/TCN/LSTM/Transformer/PatchTST/cnn1d_tcn_fusion；ML protocol 已覆盖 full/per-phase/early，并支持 dynamic_stacking_svr CLI。
- 正式 HITRAN 标准数据集方案为 6000 条组分，每条保持 512 timesteps；正式数据集尚未生成。
- 生成链路现支持 sequence chunk 多进程、worker 临时 chunk、主进程 memmap 顺序合并、staging 原子发布、HITRAN cache 并行预计算与谱 cache 原子写；memmap 数据集可在生成后单独打包 waveform_sequence.npz。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约和可复现实验链路，不追求完整硬件仪器级标定。
- 正式主线保持 mixture_id 语义，不回退或重写为 sequence_id。
- 正式数据集固定为 6000 sequences × 512 timesteps，输出目录为 data/wv4-formal-hitran-standard-6000。
- HITRAN benchmark 保持 cache-only：生成阶段不隐式联网、不补写缓存，缺 cache 必须显式失败。
- ml 与 dl 的共享逻辑以 src/common 为单一真相源。
- 动态 stacking 缺少 feature_names 或声/光/热任一模态时显式失败，不做 ridge fallback。
- cnn1d_tcn_fusion 在当前 v4 Dataset 中按 slow[8] + ultrasonic[1000] + fiber_mic[2000] 的拼接通道切分；旧 waveform scale 字段未进入拼接张量，因此模型使用波形自身 log_amplitude 作为幅值旁路。

<!-- section: risks_open_questions -->
- 正式数据集 data/wv4-formal-hitran-standard-6000 尚未生成；现有 data/hitran_cache 约 201 个 .npz 远不足以覆盖，seed=20260603 的预计算尚未执行。
- 6000 条正式数据集主要数组预计约 17.4 GiB；并行 chunk、合并数组、staging 和 HITRAN cache 需要额外空间，生成前至少预留 50 GiB。
- 尚未实际生成 long/multi_pulse 大规模数据集并跑完整 DL、ridge 与 dynamic_stacking_svr 对比。
- 新增 DL CLI 与 cnn1d_tcn_fusion 已通过测试，但尚未在正式 6000 条数据集上跑完整模型对比。
- DL Trainer 仍未包含 LR scheduler、early stopping 或分布式训练。
- 真实硬件标定参数与 TraceGas-HC-NDIR datasheet 仍缺失；外部 PNNL/NIST 定量谱数据未导入。

<!-- section: next_step -->
- 按 docs/生成正式 HITRAN 标准数据集计划.md 先运行 precompute_hitran_benchmark_cache（--sequences 6000 --seed 20260603 --workers 24），再生成 wv4-formal-hitran-standard-6000。
- 按 Validation Plan 校验 slow [6000,512,8]、ultrasonic [6000,512,1000]、fiber_mic [6000,512,2000]、y [6000,4]、splits 4200/900/600/300 与 quality status=pass。
- 正式数据集就绪后运行 ml.cli --protocol、ridge、dynamic_stacking_svr，并使用 python -m dl.cli 对 CNN/TCN/LSTM/Transformer/PatchTST/cnn1d_tcn_fusion 做对比实验。

<!-- section: recent_pivots -->
- 2026-06-05：DL CLI、根层 dl launcher 与 cnn1d_tcn_fusion 已落地；全量测试 197 passed。
- 2026-06-04：正式 HITRAN 标准数据集规模由 512 条组分调整为 6000 条，固定 512 timesteps，预期 splits 为 4200/900/600/300。
- 2026-06-04：提交 benchmark/HITRAN cache 生成性能优化（commit 7703e99，14 文件 +766/-127）；性能相关 29 passed、全量 182 passed。
- 2026-06-03：新增 dynamic_stacking_svr，完成声/光/热三模态独立 RBF-SVR、MC 漂移 MSE 动态权重和 Ridge 元学习器融合。
- 2026-06-03：统一 HITRAN 后端多时间常数动力学并落实时序改造审查修复，175 passed。
- 2026-06-03：长时序协议 S0-S5 代码闭环落地。
