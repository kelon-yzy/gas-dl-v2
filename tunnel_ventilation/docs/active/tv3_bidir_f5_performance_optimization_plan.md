# tv3 双向 F5 模型协议性能优化规划

> 目标命令：
> ```bash
> python scripts/run_tv3_bidir_model_protocol.py \
>   --config configs/tv3_bidir_model_protocol_wide.json \
>   --stage all --device cuda
> ```
> 服务器：NVIDIA RTX 5880 Ada 48 GiB × 1 + 32 vCPU（Linux，fork 可用）。
>
> 本规划只改**执行效率**，不改数值结果、不改 F5 判据/阈值、不改特征契约与冻结产物（B1/B7 配方、`raw_dsp_bidirectional_v1` 帧定义、arm 契约、`build_signature`/template digest 全部保持不变）。所有优化项都必须以「并行/串行输出逐数组相等」为验收前提。
>
> 状态：**已落地（代码侧）**。P0–P2（除明确不做的 P2-2）已实现；单元级与帧提取串行/并行逐元素相等已在 `tests/test_tunnel_ventilation_bidir_f5_perf.py` 验证。正式集墙钟加速比须在服务器 smoke/wide 上实测后写入 `outputs/tv3_bidir/perf/`。
>
> 落地摘要：
> - P0-1/P0-2：`build_tv3_bidir_feature_cache(..., workers=)` + ProcessPool；worker 内锁定 BLAS/FFT=1；默认 `min(30, cpu_count-2)`；`--workers 1` 回退串行。
> - P0-3：`overwrite=false` 且 manifest 的 `feature_builder`/`schema_version`/`build_signature` 齐全时跳过重建。
> - P1-1：`build_arm_feature_caches` 每 split 只算一次 slow windowed 块，五 arm 复用。
> - P1-2：`--arm-workers N`（默认 1=共享 slow 批构建；>1 则 ProcessPool 各 arm 独立构建）。
> - P1-3：`mlp_head` 训练张量一次性 `.to(device)`，去掉 batch 内逐次拷贝。
> - P2-1：`train_arm_head` 四 split 矩阵只加载一次；`load_arm_split_matrix` 使用 `mmap_mode="r"`。
> - P2-2：不实施（过订阅）。

---

## 0. 结论

- 唯一值得投入的方向是**把帧提取从单核串行改成 32 核并行**。它是墙钟主导项，且逐序列独立、可无损并行。
- `--device cuda` 对这条命令几乎无收益：GPU 只承担 35 个极小 MLP，48 GiB 显存和 5880 的算力全程近空闲。GPU 不是优化对象，单卡串行这些小 MLP 绰绰有余，甚至改 CPU 更快。
- 次级收益来自消除重复计算（slow 特征块 5× 重算、训练矩阵重复加载）与修正可恢复性（帧缓存 `overwrite=false` 时抛错而非跳过）。

---

## 1. 现状诊断

### 1.1 `--stage all`（wide）的计算量

| 工作项 | 次数 | 单次规模 | 设备 |
| --- | ---: | --- | --- |
| 全量帧缓存 `build_tv3_bidir_feature_cache` | 8 | 6000 序列 × 512 步 × 2 方向 ≈ 6.1M 帧逐帧匹配滤波 + `hilbert` | CPU 单核 |
| arm 特征缓存 `build_arm_feature_cache` | 35 | 4 split × windowed 统计 | CPU 单核 |
| 头训练 `train_arm_head` | 70 | 35× B1 Ridge + 35× B7(5 折 OOF Ridge + 1 MLP) | Ridge=CPU / MLP=GPU |

8 = bootstrap 1 + s_flow 1 + F5-S(2 selector × 3 seed = 6)。
35 = s_flow 5 arm + 6 split × 5 arm。
70 = 上述 35 arm-split × 2 head。

帧提取合计 ≈ **4900 万次逐帧 `np.correlate`**，全部在 `build_tv3_bidir_features.py:336` 的 `for seq_idx in range(n_seq)` 单核串行。校准阶段（`:276`）对 train 序列**再算一遍**逐帧匹配滤波。

### 1.2 瓶颈定位

- **主导项**：8 次帧提取的单核循环。32 核服务器上除 Ridge 的 BLAS 线程外，主循环只用 1 核，CPU 综合利用率约 3%（1/32，推断）。
- **GPU 空闲**：`device` 仅传给 B7 的 MLP（`run_tv3_bidir_model_protocol.py:169`）。MLP 为 hidden `(64,64)`、out 3、样本约 4000、batch 256。`mlp_head.py:144-146` 每 batch 每 epoch `.to(device)`，小张量下拷贝+kernel 启动开销大于计算，GPU occupancy 近 0%。
- **冗余计算**：`bidir_arm_features.py:205-221` 中 slow 通道 windowed 块（35 维，5 个 arm 完全相同）在每个 split 内被 5 个 arm 各算一遍；`train_arm_head` 每个 (arm,head) 重载 train/val/test/extrapolation 矩阵。
- **可恢复性不一致**：`train_arm_head` 遇 `metrics.json` 跳过（`run_...:145`），但 `build_tv3_bidir_feature_cache` 在 `overwrite=false` 且缓存存在时抛 `FileExistsError`（`build_...:169`）。中途失败重跑会在帧缓存阶段崩溃。

---

## 2. 优化项（分级）

每项给出：位置 / 改法 / 预期 / 数值不变性保证 / 风险。

### P0-1 帧提取序列级并行（最大收益）

- **位置**：`tv3/pipeline/build_tv3_bidir_features.py` 的两处循环——主提取 `:336 for seq_idx in range(n_seq)`、校准 TOF `:276 for sequence_index in train_indices`。
- **改法**：用 `concurrent.futures.ProcessPoolExecutor` 按序列分块并行。worker 函数接收 `(seq_idx_chunk, dataset_dir, template digest 路径, calibration, config)`，返回该 chunk 的每序列结果数组，主进程按 `seq_idx` 写回 `frame_store`。
  - Linux fork 下 memmap 波形（`wave_ab/ba`、`scale`、`slow`）与模板只读继承，worker 直接切片读，IPC 只传回小结果（每序列 512×17 ≈ 35 KB）。
  - chunk 大小取 `ceil(n_seq / workers)`（如 6000/30 = 200），摊薄 fork 与 IPC 成本。
  - worker 内**必须**锁定单线程 BLAS/FFT（见 P0-2），否则 30 进程 × 多线程 FFT 过订阅反而变慢。
- **预期**：帧提取部分近线性加速至约 `workers` 倍（推断 ~25–30×，`hilbert` 的 FFT 与内存带宽有次线性折损）。因帧提取是墙钟主导项，整命令墙钟大幅下降（Amdahl 下推断 5–10×，须实测）。
- **数值不变性**：逐序列独立、确定性，按 `seq_idx` 写回 → 输出数组与串行逐元素相等。校准 `calibrate_session_delay_shared_s` 为 Theil-Sen 截距，对样本聚合顺序不敏感 → τ̂ 相同。`build_signature`/template digest 不依赖提取顺序 → 不变。
- **风险**：内存峰值上升（各 worker 结果驻留）；`hilbert` 线程未锁会过订阅。均由 P0-2 与分块控制。

### P0-2 收紧 BLAS / FFT 线程数

- **位置**：进程池 worker 初始化，或命令入口 env。
- **改法**：worker `initializer` 内设 `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`；主进程做 Ridge 时可临时放开。
- **预期**：消除 P0-1 并行下的线程过订阅，保证近线性加速。
- **数值不变性**：线程数不影响 numpy/scipy 数值结果（同 dtype 同算法）。
- **风险**：无。

### P0-3 帧缓存可恢复（skip-if-exists）

- **位置**：`build_tv3_bidir_feature_cache` `:169`。
- **改法**：缓存已存在且 `manifest.json` 的 `build_signature` 与当前配置一致时**跳过并返回既有 manifest**，与 `train_arm_head` 行为一致；仅签名不匹配才报错或按 `overwrite` 重建。
- **预期**：8 次帧构建、35 次 arm 构建、70 次训练支持断点续跑，中途失败不必从头再来。
- **数值不变性**：签名一致才复用，签名绑定 template + calibration → 复用结果与重算相等。
- **风险**：需保证签名覆盖影响输出的全部输入（当前签名含 feature_builder/schema/template digest/calibration digest，足够）。

### P1-1 共享 slow windowed 特征块

- **位置**：`tv3/ml/bidir_arm_features.py:205-221`（`assemble_arm_feature_matrix` 内 slow 块），由 `build_arm_feature_cache` 对 5 个 arm 各调一次。
- **改法**：在一个 split 内先算一次 slow windowed 块（35 维）并缓存，5 个 arm 复用；仅 arm 专属的 frame/scalar 块各自计算。
- **预期**：slow 块计算量从 5×→1×/split；arm 构建阶段整体明显下降（该阶段非主导，绝对收益中等）。
- **数值不变性**：同输入同统计函数 → 特征逐列相等，`feature_names` 与 digest 不变。
- **风险**：低，注意 float32 拷贝一致性。

### P1-2 arm 缓存跨 arm 并行

- **位置**：`run_tv3_bidir_model_protocol.py:496-498` 与 `:676-677` 的 `for arm_id in arms` 循环。
- **改法**：5 个 arm 相互独立，用小进程池（如 5 worker）并行 `build_arm_feature_cache`。注意与 P0-1 不要嵌套过订阅——arm 构建发生在帧提取之后，两阶段不重叠，可各自吃满 32 核。
- **预期**：arm 构建阶段 ~5×（受 I/O 与共享 slow 块影响，实测为准）。
- **数值不变性**：各 arm 输出独立文件，互不影响。
- **风险**：I/O 竞争；建议先做 P1-1 再并行。

### P1-3 B7 MLP 去 GPU 或一次性搬运

- **位置**：`tv3/ml/mlp_head.py:135-152`。
- **改法**（二选一）：
  - **推荐**：这条协议的 B7 MLP 用 `--device cpu`（模型极小，CPU 更快且免占 GPU），把 GPU 让给需要它的正式训练线；或
  - 保留 cuda 但把整份 `x_tensor/y_tensor` 一次性 `.to(device)`，删除 batch 循环内的逐 batch `.to()`。
- **预期**：MLP 训练不再被 host↔device 拷贝主导；训练阶段（非主导项）小幅下降。
- **数值不变性**：设备与搬运时机不改变前向/反向数值（同 float32、同 seed）；早停与 checkpoint 逻辑不变。RTX 5880 与 CPU 间可能有浮点末位差异，若要求 bit 一致则固定用 CPU。
- **风险**：设备切换的浮点末位差异——验证时用组件级指标容差而非 bit 相等。

### P2-1 训练矩阵加载去重

- **位置**：`train_arm_head` `:148-149、:193`。
- **改法**：同一 (arm, split) 的矩阵在 B1/B7 两 head 间复用；或对 `feature_matrix_*.npy` 用 `mmap_mode="r"` 避免全量物化。
- **预期**：减少重复磁盘读与 float 拷贝，训练阶段边际下降。
- **数值不变性**：同数据 → 同结果。
- **风险**：低。

### P2-2 编排级并发（谨慎）

- **位置**：F5-S 的 6 个派生 split（`_run_f5s_secondary_matrix`）。
- **改法**：**不建议**在帧提取已吃满 32 核时再对 split 做进程级并发（过订阅）。仅当把帧提取并行度降到 `32/6` 时才考虑跨 split 并发。默认保持 split 串行、内部并行。
- **结论**：本项记录取舍，默认不实施。

---

## 3. 资源预算（32 vCPU + 48 GiB GPU）

- **进程数**：`workers = 30`（预留 2 核给主进程与 I/O）。可配 `--workers` 覆盖，默认取 `min(30, cpu_count-2)`。
- **BLAS/FFT 线程**：worker 内固定 1。
- **内存**：主进程 `frame_store` 17 数组 × 6000×512×4B ≈ 209 MB/build；波形走 memmap 不入常驻。并行结果分片峰值推断 < 1 GB。服务器 RAM 需求低（假设 ≥ 32 GiB，未核实实际配置）。
- **GPU 显存**：MLP 参数千级，占用 MB 级，48 GiB 完全过剩；若并发多 MLP 也不受显存限制，瓶颈是 kernel 串行——故建议 CPU。
- **磁盘**：8 帧缓存 + 35 arm 缓存写入量与现状一致，并行不改变落地体积。

---

## 4. 数值一致性验证方案（验收前提）

1. **单元级**：`python -m pytest -q tests/test_tunnel_ventilation_bidir_*.py` 全过。
2. **帧提取等价**：smoke 数据上分别用串行与并行构建帧缓存，对全部 `.npy` 数组 `np.array_equal`（浮点用 `np.allclose(rtol=0, atol=0)` 期望完全相等）；`session_delay_calibration.json` 的 τ̂/digest 相同；`manifest.json` 的 `build_signature` 相同。
3. **arm/特征等价**：P1-1 前后 `feature_names.json` 与 `feature_names_digest` 不变，`feature_matrix_*.npy` 逐元素相等。
4. **端到端**：smoke 配置 `--stage all` 优化前后 `f5_verdict.json` 的 verdict / stage_passed / 12 格 ΔR² 一致；退出码一致。
5. **加速比留档**：记录优化前后各阶段墙钟与 CPU 利用率（`/usr/bin/time -v` 或 `psutil` 采样），写入 `outputs/tv3_bidir/perf/`。

---

## 5. 落地顺序

1. P0-2（线程锁）+ P0-3（可恢复）——低风险、独立，先做。
2. P0-1（帧提取并行）——核心收益；先 smoke 验证 §4.2 等价，再测加速比。
3. P1-1（共享 slow 块）→ P1-2（arm 并行）——特征阶段收益。
4. P1-3（MLP 去 GPU）——训练阶段与 GPU 释放。
5. P2-1——边际优化，可选。
6. 全部通过 §4 后，才在服务器上跑正式 `configs/tv3_bidir_model_protocol_wide.json --stage all`。

**验证命令（smoke 优先）**：
```bash
# 正确性与加速比（窄/宽 smoke）
python scripts/run_tv3_bidir_model_protocol.py \
  --config configs/tv3_bidir_model_protocol_wide_smoke.json --stage all --device cpu
python -m pytest -q tests/test_tunnel_ventilation_bidir_model_protocol.py
```

---

## 6. 风险与回退

- **并行数值漂移**：若并行帧提取与串行不逐元素相等，说明引入了跨序列状态或非确定性——停止并排查，不得以「指标接近」放行。
- **过订阅**：BLAS/FFT 线程未锁会使并行比串行更慢——P0-2 是 P0-1 的前置。
- **设备浮点差异**：P1-3 若要求 bit 一致，固定 CPU；否则用组件指标容差验证。
- **回退**：每项优化以 `--workers 1` / 配置开关保留串行路径；出现问题可单项回退到现行为，不影响冻结产物。

---

## 7. 明确不做

- 不改 F5 判据、阈值、preregistration、冻结 B1/B7 超参。
- 不改 `raw_dsp_bidirectional_v1` 帧定义、arm 特征契约、schema、manifest 字段。
- 不为提速裁剪序列数、timestep 或窗口长度。
- 不引入近似匹配滤波（如 FFT 卷积替换 `np.correlate`）除非另行验证逐元素等价——本规划范围内视为数值变更，不纳入。
