<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [unknown] | 2026-06-08 -->
<!-- file-state: revision=46 | updated-at=2026-06-08T10:29:13+08:00 | writer-id=unknown | base-workspace-revision=91 -->

<!-- section: current_state -->
- 2026-06-05：JSON 配置驱动的正式实验总控已落地，主命令为 python -m pipeline.run_experiment --config configs/experiment/formal_full.json，支持 dry-run、ML+DL 顺序执行、run/summary/report 输出。
- configs/experiment/formal_full.json 默认展开 ridge_slow、ridge_all_modalities、dynamic_stacking_svr_all_modalities，以及 cnn1d/tcn/lstm/transformer/patchtst/cnn1d_tcn_fusion。
- DL Trainer 已支持 early stopping、ReduceLROnPlateau/none scheduler、best_checkpoint.pt、learning_rates、stopped_early/stop_reason/best_checkpoint_path metrics，并已新增 AMP、CUDA non_blocking batch 搬运、epoch/train/val 计时、samples/s 与 GPU memory 指标。
- dl.cli 与 ml.cli 均支持 --config JSON；dl.cli 已支持 pin_memory、persistent_workers、prefetch_factor、amp 与 metrics_live.jsonl 性能指标输出。
- formal_full.json 面向单 GPU 24GB + 32 核 CPU 默认启用 num_workers=8、pin_memory=true、persistent_workers=true、prefetch_factor=2、amp.enabled=true。
- DynamicStackingSVRRegressor 已支持 n_jobs；默认 dynamic_stacking_svr_all_modalities 使用 n_jobs=4 并传给 MultiOutputRegressor。
- 已基于 outputs/runs/formal_full 的 9 个模型/基线结果生成 outputs/reports/formal_full_experiment_report.md；报告按 test RMSE 排序并包含外推、组分级误差、Ridge 协议诊断与 DL 早停状态。
- 正式 HITRAN 标准数据集方案为 6000 条组分，每条保持 512 timesteps；本地仓库仍未生成 data/wv4-formal-hitran-standard-6000。
- 2026-06-08：已新增 docs/N2误差改进计划_ilr_alr.md，收集 ALR/ILR、零值处理与当前仓库改造点，形成面向 N2 误差的组成数据回归改进计划。

<!-- section: active_judgments -->
- 第一阶段聚焦核心契约和可复现实验链路，不追求完整硬件仪器级标定。
- 正式主线保持 mixture_id 语义，不回退或重写为 sequence_id。
- 正式数据集固定为 6000 sequences x 512 timesteps，默认输出目录为 data/wv4-formal-hitran-standard-6000。
- HITRAN benchmark 保持 cache-only：生成阶段不隐式联网、不补写缓存，缺 cache 必须显式失败。
- 单 GPU 服务器第一阶段不做 DDP、不并发运行多个 DL 模型，避免单卡显存竞争。
- AMP 在 formal_full.json 中默认启用；CPU 设备若显式启用 AMP 会直接失败，要求用户显式改 amp.enabled=false。
- ML 并行第一步只启用 MultiOutputRegressor 的 n_jobs，不引入三模态嵌套并行或 MC drift 并行。
- 当前结果报告以 outputs/runs/formal_full 中实际运行目录为准，不改写原始运行产物；主表按 test 集整体 RMSE 排序。
- ILR/ALR 解决的是组成数据目标空间与闭合约束问题，不能单独解决 N2 瞬态信息在 full-window 汇总中被冲淡的问题。
- 当前优先级是先单独验证 ALR/ILR 的收益，再决定是否推进 phase-aware N2 建模。

<!-- section: risks_open_questions -->
- 本地仓库仍没有 data/wv4-formal-hitran-standard-6000；完整正式实验需在服务器已有合格数据集后运行或复现。
- 性能收益需要在服务器正式数据集和 CUDA 环境上观察；本地只能通过测试和 dry-run 验证接口。
- formal_full.json 默认开启 AMP；若服务器 CUDA/PyTorch 环境或某模型数值不稳定，需要显式关闭 AMP 后复跑。
- 若需要重新生成正式数据集，现有 data/hitran_cache 约 201 个 .npz 远不足以覆盖，seed=20260603 的预计算仍需执行。
- N2 组分是当前主要误差来源，多数模型 test R2 接近 0 或为负。
- outputs/reports/* 被 .gitignore 忽略，新生成的 formal_full_experiment_report.md 默认不会进入 git diff。
- 正式数据集各组分零值比例尚未统计；若 H2/CO2/N2 存在精确 0，log-ratio 方案必须引入显式 zero replacement 与审计。

<!-- section: next_step -->
- 提交本次文档与记忆更新后，先在 ML baseline 上实现 Ridge + ALR-CH4 与 Ridge + ILR-N2-first 两个对照实验。
- 对照实验重点观察 test x_N2 R2、macro_RMSE，以及 H2/CH4/CO2 是否出现明显退化。
- 在推进任何 DL compositional head 之前，先统计正式数据集各组分零值比例并确定 zero replacement 策略。
- 若需要 CPU 跑正式配置，先将 training.amp.enabled 显式改为 false。
- ILR/ALR + phase-aware N2 modeling 仅保留为下一步计划，待 ILR/ALR 单独实验收益明确后再讨论。

<!-- section: recent_pivots -->
- 2026-06-08：基于 formal_full 结果进一步定位 N2 为跨 ML/DL 的系统性弱项，并确认 exposure/recovery 窗口下 N2 可学性明显高于 steady/full-window。
- 2026-06-08：新增 docs/N2误差改进计划_ilr_alr.md，确定先单独验证组成数据回归；phase-aware N2 modeling 保留为下一步计划。
- 2026-06-05：实现单 GPU 24GB + 32 核 CPU 性能优化：DL AMP/DataLoader/性能指标、ML n_jobs；目标测试 45 passed、全量 219 passed。
- 2026-06-05：基于 outputs/runs/formal_full 生成 formal_full_experiment_report.md；ridge_all_modalities/full 为当前 test RMSE 最佳，cnn1d_tcn_fusion 为最佳 DL。
- 2026-06-05：提交并推送 dbf4c05 feat(pipeline): add configured experiment runner；新增 JSON 实验总控、配置化 ML/DL CLI、DL early stopping/scheduler/best checkpoint、根层 ml launcher；全量测试 211 passed。
- 2026-06-05：DL CLI、根层 dl launcher 与 cnn1d_tcn_fusion 已落地；全量测试 197 passed。
- 2026-06-04：正式 HITRAN 标准数据集规模由 512 条组分调整为 6000 条，固定 512 timesteps，预期 splits 为 4200/900/600/300。
- 2026-06-04：提交 benchmark/HITRAN cache 生成性能优化（commit 7703e99，14 文件 +766/-127）；性能相关 29 passed、全量 182 passed。
- 2026-06-03：新增 dynamic_stacking_svr，完成声/光/热三模态独立 RBF-SVR、MC 漂移 MSE 动态权重和 Ridge 元学习器融合。
