# 生成正式 HITRAN 标准数据集计划

## Summary

生成一套完整正式基线数据集，采用用户选择的 `hitran_hapi_v1` 光学后端、标准时间轴和合理默认特征范围。

固定参数：

- 输出目录：`data/wv4-formal-hitran-standard-6000`
- 样本数：`6000 sequences`
- 时间轴：`standard`，即 `512 timesteps × 0.5s`
- 光学后端：`hitran_hapi_v1`
- 采样：`lhs`
- 阶段：`standard_exposure`
- 存储：`memmap`，正式大规模生成优先走 `.npy`/memmap，避免压缩包阻塞主生成路径；如需 `waveform_sequence.npz`，生成完成后单独打包
- 容量：主要数组预计约 `17.4 GiB`；并行 chunk、合并数组、staging 和 HITRAN cache 会额外占用空间，生成前至少预留 `50 GiB` 可用磁盘
- 随机种子：`20260603`
- 性能：CLI 未传 `--workers` 时默认自动并行；在当前 `16 核 / 32 线程` 机器上等价于默认最多 24 个 worker。注意 Python API 的 `BenchmarkGenerationSpec.workers` 仍默认是 `1`，只有 CLI 默认并行，脚本调用必须显式传 `workers`

## Key Changes / Commands

先预计算 HITRAN cache，参数必须与生成数据集一致：

```powershell
python -m pipeline.precompute_hitran_benchmark_cache `
  --cache-root data/hitran_cache `
  --sequences 6000 `
  --seed 20260603 `
  --sampling-strategy lhs `
  --workers 24
```

再生成完整数据集：

```powershell
python -m pipeline.generate_benchmark `
  --output-root data `
  --dataset wv4-formal-hitran-standard-6000 `
  --sequences 6000 `
  --seed 20260603 `
  --time-axis-preset standard `
  --storage memmap `
  --multi-path-phase steady `
  --stage-profile standard_exposure `
  --stage-jitter 0 `
  --sampling-strategy lhs `
  --path-lms 0.20,0.25,0.30,0.35,0.40 `
  --optical-absorption-backend hitran_hapi_v1 `
  --hitran-cache-root data/hitran_cache `
  --workers 24
```

如果需要生成压缩包，再单独执行：

```powershell
python -m pipeline.bundle_waveform_sequence `
  --dataset-dir data/wv4-formal-hitran-standard-6000
```

说明：上面显式写 `--workers 24` 是为了固定当前 32 线程机器的性能配置；也可以省略，CLI 会按 `default_worker_count(sequence_count)` 自动选择。程序化 API 不会自动并行，`BenchmarkGenerationSpec.workers` 默认保持 `1`。

采用当前代码内置的合理特征范围：

- 组分：`H2 0-30%`，含低氢 `<3%` 和高氢 `25-30%` 覆盖；`CO2 0-15%`；`N2 0-20%`；`CH4` 为补足项且不低于 `40%`
- 环境：`T_C 15-35`，`P_MPa 0.10-0.709`，`H_RH 20-80`
- 光程：基础 `L_m 0.2-1.8`，多声程候选 `0.20,0.25,0.30,0.35,0.40`
- 输出模态：slow 8 通道、ultrasonic waveform、fiber_mic waveform、TOF/声速/衰减/质量派生数组、labels、splits、metadata、scalers、manifest、quality summary

## Validation Plan

生成后执行以下检查：

```powershell
python -m pytest tests/test_benchmark_generation.py tests/test_generate_benchmark_cli.py
```

对生成目录做文件完整性检查，必须包含：

- 根表：`condition_grid_sequence.csv`、`sequence_index.csv`、`sequence_labels.csv`、`manifest.json`
- splits：`train.csv`、`val.csv`、`test.csv`、`extrapolation.csv`、`split_summary.json`
- arrays：`slow.npy`、`ultrasonic_int16.npy`、`fiber_mic_int16.npy`、全部 ultrasonic 派生 `.npy`；若已执行 bundle 命令，还应包含 `waveform_sequence.npz`
- labels/metadata/scalers/quality：`labels/y.npy`、`metadata/*.npy`、`metadata/waveform_spec.json`、`scalers/*.json`、`quality/validation_summary.json`

验收形状：

- `slow`: `[6000, 512, 8]`
- `ultrasonic`: `[6000, 512, 1000]`
- `fiber_mic`: `[6000, 512, 2000]`
- `y`: `[6000, 4]`
- splits 预期：train `4200`、val `900`、test `600`、extrapolation `300`
- `quality/validation_summary.json.status == "pass"`

再做下游读取冒烟：

- ML：`load_feature_matrix(..., modalities=("slow","ultrasonic","fiber_mic"))` 能读取 train split
- DL：`V4BenchmarkDataset(..., modalities=("slow","ultrasonic","fiber_mic"))` 单样本 `x/y` 均 finite

## Assumptions

- 使用 HITRAN 正式路径时，缺 cache 必须失败；不改成 empirical fallback。
- 不修改代码，只生成数据文件。
- 若本地 HAPI 原始表缺失，预计算阶段允许使用真实 HAPI 下载；不允许生成 fake 谱线。
- `storage=memmap` 是正式大规模生成推荐路径；如需“完整文件”中的压缩包，生成后用 `pipeline.bundle_waveform_sequence` 单独补齐。
