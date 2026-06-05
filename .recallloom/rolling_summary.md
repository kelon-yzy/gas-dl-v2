<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-06-05 -->
<!-- file-state: revision=43 | updated-at=2026-06-05T13:30:41+08:00 | writer-id=Codex | base-workspace-revision=85 -->

<!-- section: current_state -->
- 2026-06-05：JSON 配置驱动的正式实验总控已落地，主命令为 python -m pipeline.run_experiment --config configs/experiment/formal_full.json，支持 dry-run、ML+DL 顺序执行、run/summary/report 输出。
- configs/experiment/formal_full.json 默认展开 ridge_slow、ridge_all_modalities、dynamic_stacking_svr_all_modalities，以及 cnn1d/tcn/lstm/transformer/patchtst/cnn1d_tcn_fusion。
- dl.cli 与 ml.cli 均支持 --config JSON，显式 CLI 参数覆盖配置；根层 ml launcher 已补齐，仓库根目录可运行 python -m ml.cli。
- DL Trainer 已支持 early stopping、ReduceLROnPlateau/none scheduler、best_checkpoint.pt、learning_rates 与 stopped_early/stop_reason/best_checkpoint_path metrics。
- DL 模型注册表覆盖 CNN/TCN/LSTM/Transformer/PatchTST/cnn1d_tcn_fusion；ML protocol 覆盖 full/per-phase/early，并支持 dynamic_stacking_svr CLI。
- 正式 HITRAN 标准数据集方案为 6000 条组分，每条保持 512 timesteps；本地仓库仍未生成 data/wv4-formal-hitran-standard-6000。
- 生成链路现支持 sequence chunk 多进程、worker 临时 chunk、主进程 memmap 顺序合并、staging 原子发布、HITRAN cache 并行预计算与谱 cache 原子写；memmap 数据集可在生成后单独打包 waveform_sequence.npz。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约和可复现实验链路，不追求完整硬件仪器级标定。
- 正式主线保持 mixture_id 语义，不回退或重写为 sequence_id。
- 正式数据集固定为 6000 sequences x 512 timesteps，默认输出目录为 data/wv4-formal-hitran-standard-6000。
- HITRAN benchmark 保持 cache-only：生成阶段不隐式联网、不补写缓存，缺 cache 必须显式失败。
- ml 与 dl 的共享逻辑以 src/common 为单一真相源。
- 动态 stacking 缺少 feature_names 或声/光/热任一模态时显式失败，不做 ridge fallback。
- cnn1d_tcn_fusion 在当前 v4 Dataset 中按 slow[8] + ultrasonic[1000] + fiber_mic[2000] 的拼接通道切分；旧 waveform scale 字段未进入拼接张量，因此模型使用波形自身 log_amplitude 作为幅值旁路。
- 第一批不做 AMP、DDP、RevIN、DLinear、iTransformer、不确定性输出和统一 simplex head，保留为后续阶段。
- 默认不做 CUDA 静默回退；formal_full.json 写 cuda 时服务器无 CUDA 将显式失败，需用 --device cpu 或修改配置显式切换。

<!-- section: risks_open_questions -->
- 本地仓库仍没有 data/wv4-formal-hitran-standard-6000；完整正式实验需在服务器已有合格数据集后运行。
- 若需要重新生成正式数据集，现有 data/hitran_cache 约 201 个 .npz 远不足以覆盖，seed=20260603 的预计算仍需执行。
- 6000 条正式数据集主要数组预计约 17.4 GiB；并行 chunk、合并数组、staging 和 HITRAN cache 需要额外空间，生成前至少预留 50 GiB。
- 尚未在正式 6000 条数据集上跑完整 ML+DL 对比；dynamic_stacking_svr_all_modalities 与 cnn1d_tcn_fusion 可能较重，服务器需关注 CPU/RAM/GPU 资源。
- 真实硬件标定参数与 TraceGas-HC-NDIR datasheet 仍缺失；外部 PNNL/NIST 定量谱数据未导入。

<!-- section: next_step -->
- 在有正式数据集的服务器上先运行 python -m pipeline.run_experiment --config configs/experiment/formal_full.json --dry-run 核对计划。
- 按 Validation Plan 校验 slow [6000,512,8]、ultrasonic [6000,512,1000]、fiber_mic [6000,512,2000]、y [6000,4]、splits 4200/900/600/300 与 quality status=pass。
- 正式数据集通过校验后，运行 python -m pipeline.run_experiment --config configs/experiment/formal_full.json --dataset-dir data/wv4-formal-hitran-standard-6000 --device cuda；无 CUDA 时显式改用 --device cpu。
- 若服务器也缺少正式数据集，则按 docs/生成正式 HITRAN 标准数据集计划.md 先运行 precompute_hitran_benchmark_cache，再生成 wv4-formal-hitran-standard-6000。
- 第二阶段再规划 AMP/性能参数、统一 composition head、DLinear/iTransformer、RevIN 与不确定性输出。

<!-- section: recent_pivots -->
- 2026-06-05：提交并推送 dbf4c05 feat(pipeline): add configured experiment runner；新增 JSON 实验总控、配置化 ML/DL CLI、DL early stopping/scheduler/best checkpoint、根层 ml launcher；全量测试 211 passed。
- 2026-06-05：DL CLI、根层 dl launcher 与 cnn1d_tcn_fusion 已落地；全量测试 197 passed。
- 2026-06-04：正式 HITRAN 标准数据集规模由 512 条组分调整为 6000 条，固定 512 timesteps，预期 splits 为 4200/900/600/300。
- 2026-06-04：提交 benchmark/HITRAN cache 生成性能优化（commit 7703e99，14 文件 +766/-127）；性能相关 29 passed、全量 182 passed。
- 2026-06-03：新增 dynamic_stacking_svr，完成声/光/热三模态独立 RBF-SVR、MC 漂移 MSE 动态权重和 Ridge 元学习器融合。
- 2026-06-03：统一 HITRAN 后端多时间常数动力学并落实时序改造审查修复，175 passed。
- 2026-06-03：长时序协议 S0-S5 代码闭环落地。
