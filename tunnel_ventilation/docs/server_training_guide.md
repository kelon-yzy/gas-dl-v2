# 掘进通风服务器训练操作手册

> 本文档给出在 Linux + RTX 5880 48GB 服务器上执行 tv3 正式训练的完整步骤。
> 场景背景见 [README.md](README.md)，训练方案见 [dl_training_plan.md](dl_training_plan.md)，实验路线见 [experiment_roadmap.md](experiment_roadmap.md)。

## 1. 训练内容

| 项目      | 规模                               | 说明                                                                                |
| ------- | -------------------------------- | --------------------------------------------------------------------------------- |
| 数据集     | tv3-formal 600 序列 × 512 时步       | 服务器上由 CLI 重新生成，**跳过 fiber_mic**（`--skip-fiber-mic`）+ **int16 per-timestep scale** |
| 基线训练    | 5 模型 × 3 seeds = 15 runs         | `scripts/run_tv3_baseline.py` 编排                                                  |
| 多模态方向 B | cnn1d_tcn_fusion，slow+ultrasonic | `tv3_tcn_multimodal.json` + `--modalities slow,ultrasonic`，验证 O₂ 是否可辨识            |
| GPU     | RTX 5880 48GB                    | 多模态 batch_size 可从 2 调到 8（见 §4.2）                                                  |

> **波形存储优化**：tv3 默认采用 int16 + per-timestep 自适应 scale（方案 B），物理 ADC 仍为 20-bit（`daq_bits=20`），存储时按每 timestep 峰值定标压缩为 int16。实测峰值占满量程 ~22%，per-timestep scale 比固定 scale 量化步长小 ~4.6×；int16 量化误差 max ~1e-5 V，远小于噪声 std 1e-3 V（误差/噪声 ≈ 1%），精度损失可忽略。数据集从 int32 的 ~6 GB 降至 int16 的 ~3 GB。

> **跳过 fiber_mic 说明**：fiber_mic 波形占数据集 66%（600 序列下 11.4 GB），当前阶段先只跑超声链路。光纤麦克风代码全部保留（`FiberMicSpec` / `simulate_fiber_mic_measurement` / `waveforms.py` 未改），后续去掉 `--skip-fiber-mic` 即可恢复完整三模态生成。DL 端需同步去掉 `fiber_mic` 模态（`--modalities slow,ultrasonic`），否则会因找不到 `fiber_mic_int32.npy` 报错。

> **数据量限制说明**：600 序列 ≈ 400 训练样本，首轮 TCN 50 epochs 全组分 R²≈0（见 [experiment_roadmap.md](experiment_roadmap.md) 基线结果分析）。服务器训练能加速 epoch，但无法解决数据量不足的根本问题。如需更好效果，后续应扩大数据集规模（见 §6）。

## 2. 环境准备

### 2.1 系统要求

| 项目     | 要求                                    | 说明                                                |
| ------ | ------------------------------------- | ------------------------------------------------- |
| OS     | Linux（Ubuntu 22.04 推荐）                | 脚本按 bash 编写                                       |
| Python | 3.10–3.13（排除 3.14）                    | 项目 pyproject.toml 约束                              |
| CUDA   | 11.8+                                 | PyTorch GPU 版本需匹配驱动                               |
| GPU 显存 | ≥ 8 GB（基线）/ ≥ 16 GB（多模态 batch_size=8） | RTX 5880 48GB 充足                                  |
| 磁盘     | ≥ 8 GB                                | 数据集 ~3 GB（int16 + 跳过 fiber_mic）+ 临时 chunk + 输出    |
| 内存     | ≥ 8 GB                                | 600 序列 workers=4 生成峰值 ~3 GB（int16 + 跳过 fiber_mic） |

### 2.2 获取代码

```bash
git clone https://github.com/kelon-yzy/gas-dl-v2.git
cd gas-dl-v2
git checkout feat/ultrasonic-200khz-adc-20bit-alignment

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> `requirements.txt` 内容为 `-e .[dev]`，会安装 numpy / scipy / scikit-learn / torch / hitran-api / pytest。
> 若服务器无 GPU 版 torch，需先按 [PyTorch 官方指南](https://pytorch.org/get-started/locally/) 安装匹配 CUDA 的版本，再 `pip install -r requirements.txt`。

### 2.3 验证安装

```bash
# 掘进通风单元测试（应 75 passed）
python -m pytest tests/test_tunnel_ventilation_schema.py \
                 tests/test_tunnel_ventilation_physics.py \
                 tests/test_tunnel_ventilation_benchmark.py \
                 tests/test_tunnel_ventilation_dl_training.py -q
```

## 3. 数据集生成

数据集不进 git（fiber_mic_int32.npy 11.7 GB 超 GitHub 100 MB 限制），在服务器上由 CLI 重新生成。

### 3.1 tv3-smoke 链路验证（可选，~30 秒）

```bash
python -m pipeline.generate_tunnel_ventilation_benchmark \
    --output-root data --dataset tv3-smoke --sequences 32 --seed 20260704 \
    --timesteps 32 --dt-s 0.5 --optical-absorption-backend empirical_v1 --workers 1
```

验证清单：

| 检查项                           | 预期                          |
| ----------------------------- | --------------------------- |
| `labels/y.npy` shape          | `(32, 3)`                   |
| `metadata/label_names.npy`    | `["x_CO2", "x_O2", "x_N2"]` |
| `manifest.composition_scheme` | `"tunnel_ventilation"`      |
| `manifest.background_fields`  | `[]`                        |
| `sequences/slow.npy` 最后一维     | 7                           |

### 3.2 tv3-formal 正式集（600 序列，~1–2 分钟，int16 + 跳过 fiber_mic）

```bash
python -m pipeline.generate_tunnel_ventilation_benchmark \
    --output-root data --dataset tv3-formal --sequences 600 --seed 20260704 \
    --timesteps 512 --dt-s 0.5 --optical-absorption-backend empirical_v1 \
    --storage memmap --workers 4 --skip-fiber-mic
```

生成完成后 `data/tv3-formal/` 约 3 GB（int16 + 跳过 fiber_mic），关键产物：

| 文件                               | shape                  | 说明                                                                                |
| -------------------------------- | ---------------------- | --------------------------------------------------------------------------------- |
| `sequences/ultrasonic_int16.npy` | (600, 512, 5000) int16 | 超声波形（per-timestep scale 压缩）                                                       |
| `sequences/ultrasonic_scale.npy` | (600, 512) float32     | per-timestep scale_factor（每时步不同）                                                  |
| `sequences/slow.npy`             | (600, 512, 7) float32  | 7 慢通道（V_NDIR_CO2 / V_TCS / T_C / P_MPa / H_RH / L_m / piston_position_m）        |
| `labels/y.npy`                   | (600, 3)               | CO₂/O₂/N₂ 浓度                                                                      |
| `manifest.json`                  | —                      | composition_scheme + sim_revision，`fiber_mic_model: null`，`waveform_dtype: int16` |

> tv3 默认 int16 + per-timestep scale（方案 B），无需额外参数。数据集从 int32 的 17 GB 降到 int16 + 跳过 fiber_mic 的 ~3 GB（-82%）。DL 端通过 `waveform_spec.json` 自动识别 dtype，加载 `ultrasonic_int16.npy`。
> `ultrasonic_scale.npy` 现在是 per-timestep 自适应（每个时步值不同，按该时步波形峰值定标），dequantize 时 `waveform_int16 * scale` 还原电压。
> 若服务器内存 < 8 GB，降低 `--workers`（如 `--workers 2`）。

## 4. 训练执行

### 4.1 基线 15 runs（5 模型 × 3 seeds）

```bash
python scripts/run_tv3_baseline.py
```

编排脚本固定 seeds = `42, 123, 456`，依次运行：

| 模型       | 配置                  | 默认 epochs      | 默认 batch_size |
| -------- | ------------------- |:--------------:|:-------------:|
| cnn1d    | `tv3_baseline.json` | 50             | 16            |
| tcn      | `tv3_tcn.json`      | 50             | 16            |
| lstm     | `tv3_lstm.json`     | 50             | 16            |
| patchtst | `tv3_patchtst.json` | 80             | 16            |
| ridge    | `tv3_ridge.json`    | —（closed-form） | —             |

可选参数：

```bash
# 只跑部分模型 / seeds
python scripts/run_tv3_baseline.py --models tcn,ridge --seeds 42

# 覆盖 epochs
python scripts/run_tv3_baseline.py --epochs 100

# 只打印命令不执行
python scripts/run_tv3_baseline.py --dry-run
```

输出位置：`outputs/tv3_baseline/{model}/seed{seed}/metrics.json`，汇总在 `outputs/tv3_baseline/summary.json`。

> DL run 非零退出码按失败暴露，即使已写出 `metrics.json`。`runs.jsonl` 记录所有 run 状态，partial rerun 会合并已有记录。

### 4.2 多模态方向 B（cnn1d_tcn_fusion）

多模态配置 `tv3_tcn_multimodal.json` 默认 `batch_size=2`（受本地 8 GB 显存限制）、`modalities="slow,ultrasonic,fiber_mic"`。当前数据集跳过了 fiber_mic，需用 `--modalities slow,ultrasonic` 覆盖。RTX 5880 48 GB 可调大 batch_size：

```bash
python -m dl.cli \
    --config configs/experiment/tv3/tv3_tcn_multimodal.json \
    --modalities slow,ultrasonic \
    --batch-size 8 \
    --output-dir outputs/tv3_tcn_multimodal/s42 \
    --seed 42
```

> `--modalities slow,ultrasonic` 必须显式传入，否则配置默认含 `fiber_mic` 会因找不到 `fiber_mic_int32.npy` 报错。

batch_size 选择建议（slow+ultrasonic 两模态，无 fiber_mic，显存占用比三模态低）：

| batch_size | 预计显存占用 | 适用                 |
|:----------:|:------:| ------------------ |
| 2          | ~5 GB  | 配置默认（保守）           |
| 4          | ~9 GB  | 中等                 |
| **8**      | ~18 GB | **推荐**（48 GB 显存充足） |
| 16         | ~34 GB | 激进（48 GB 显存仍有余量）   |

> 若 OOM，降到 4 或 8；若显存有余量，可试 16。AMP fp16 已在配置中启用。
> 多模态训练 epochs=50，early stopping patience=10。如需多 seed，手动改 `--seed` 和 `--output-dir` 重复运行。

### 4.3 多模态多 seed（可选）

配置脚本 `run_tv3_baseline.py` 不含多模态。如需多模态多 seed，手动循环：

```bash
for seed in 42 123 456; do
    python -m dl.cli \
        --config configs/experiment/tv3/tv3_tcn_multimodal.json \
        --modalities slow,ultrasonic \
        --batch-size 8 \
        --output-dir outputs/tv3_tcn_multimodal/s${seed} \
        --seed ${seed}
done
```

## 5. 结果回收

训练完成后，需回收的产物：

| 文件                   | 位置                                                     | 大小       | 用途            |
| -------------------- | ------------------------------------------------------ | -------- | ------------- |
| `summary.json`       | `outputs/tv3_baseline/summary.json`                    | 小        | 基线 15 runs 汇总 |
| `runs.jsonl`         | `outputs/tv3_baseline/runs.jsonl`                      | 小        | 每个 run 状态记录   |
| `metrics.json`       | `outputs/tv3_baseline/{model}/seed{seed}/metrics.json` | 小        | 单 run 完整指标    |
| `metrics.json`       | `outputs/tv3_tcn_multimodal/s*/metrics.json`           | 小        | 多模态指标         |
| `best_checkpoint.pt` | 同上目录                                                   | 0.5–3 MB | 最优模型权重（按需）    |
| `metrics_live.jsonl` | 同上目录                                                   | 小        | 每 epoch 训练日志  |

打包回收（在服务器上）：

```bash
# 只回收 metrics（小文件）
tar czf tv3_results_metrics.tar.gz \
    outputs/tv3_baseline/summary.json \
    outputs/tv3_baseline/runs.jsonl \
    outputs/tv3_baseline/*/seed*/metrics.json \
    outputs/tv3_baseline/*/seed*/run_config.json \
    outputs/tv3_tcn_multimodal/s*/metrics.json \
    outputs/tv3_tcn_multimodal/s*/run_config.json

# 含 checkpoint（按需）
tar czf tv3_results_full.tar.gz outputs/tv3_baseline outputs/tv3_tcn_multimodal
```

下载到本地（在本地执行）：

```bash
scp user@server:/path/to/gas-dl-v2/tv3_results_metrics.tar.gz .
```

## 6. 后续扩展

### 6.1 扩大数据集

600 序列对 DL 严重不足。若服务器资源允许，可生成 6000 序列。int16 + 跳过 fiber_mic 时磁盘/内存需求大幅降低：

```bash
python -m pipeline.generate_tunnel_ventilation_benchmark \
    --output-root data --dataset tv3-formal-6000 --sequences 6000 --seed 20260704 \
    --timesteps 512 --dt-s 0.5 --optical-absorption-backend empirical_v1 \
    --storage memmap --workers 4 --skip-fiber-mic
```

| 规模      | 磁盘（int16 + 跳过 fiber_mic） | 生成内存峰值（workers=4） |
| ------- | ------------------------ | ----------------- |
| 600 序列  | ~3 GB                    | ~3 GB             |
| 6000 序列 | ~29 GB                   | ~15 GB            |

> 6000 序列 int16 + 跳过 fiber_mic 后 memmap ~29 GB（原始 int32 + 含 fiber_mic 为 172 GB，减 83%）。`build_sequence_arrays` 在内存中预分配 chunk 数组，workers=4 chunk=750 每 worker ~3.8 GB（int16）。若内存不足，降低 `--workers`。
> 若需完整三模态 6000 序列（int32 + fiber_mic），需改回 `WaveformSpec()`（去掉 per_timestep_scale + waveform_dtype="int16"）并去掉 `--skip-fiber-mic`，磁盘需 ≥350 GB、内存 ≥90 GB。

### 6.2 阶段 Ⅱ ablation

见 [experiment_roadmap.md](experiment_roadmap.md) 阶段 Ⅱ：

- 通道消融（`--slow-channels` 参数移除指定通道）
- O₂ 可辨识性消融（`--modalities` 参数切换模态组合）
- Loss 消融（`--loss` 参数切换 loss）

### 6.3 长时间训练后台运行

```bash
# nohup 后台运行，日志写文件
nohup python scripts/run_tv3_baseline.py > tv3_baseline.log 2>&1 &

# 或 tmux
tmux new -s tv3
python scripts/run_tv3_baseline.py
# Ctrl+B D 脱离
```

## 7. 故障排查

| 问题                   | 排查方向                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------ |
| `CUDA out of memory` | 降低 `--batch-size`；多模态从 8 降到 4 或 2                                                          |
| 数据生成 OOM             | 降低 `--workers`；`build_sequence_arrays` 预分配 chunk 内存                                        |
| 磁盘不足                 | 清理 `data/tv3-formal/.chunks/` 临时文件；`du -sh data/tv3-formal/`                               |
| 测试失败                 | `python -m pytest tests/test_tunnel_ventilation_*.py -v` 查看详情                              |
| DL run 非零退出          | 查看 `outputs/tv3_baseline/{model}/seed{seed}/` 下是否有 `metrics.json`（诊断用），`runs.jsonl` 记录失败原因 |
| 多模态 `gas_head` 报错    | tv3 下 `output_mode` 必须为 `raw3`、`out_dim=3`，`gas_head` / `target_transform` 被拒绝             |
