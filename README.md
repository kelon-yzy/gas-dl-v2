# 正式实验 v4

本项目是 `V3_正式实验` 的目标架构重构版。当前主线已经从第一阶段的核心契约固定，推进到可运行的 benchmark 生成、传统 ML baseline、DL 数据/模型/轻量训练闭环和长时序协议验证阶段。正式大规模实验、跨 run 汇总和真实硬件/谱表标定仍待完成。

## 当前状态

- 已建立 `src/sim/core`：定义 benchmark、mixture、sequence 的核心 ID 语义。
- 已建立 `src/sim/generation`：提供正式 benchmark 生成的最小垂直切片，默认使用拉丁超立方采样（LHS），NDIR 默认走 `hitran_hapi_v1` cache-only 光谱积分；HITRAN 主线只计算 TCS 热导慢变量，不再顺带跑 empirical NDIR；`empirical_v1` 保留为显式兼容/对照路径。
- 已建立 `src/sim/packaging`：定义 `condition_grid`、`sequence_index`、split rows、manifest、scaler 常量和正式 run 最小输出契约。
- 已建立 `src/sim/validation`：校验旧字段、ID 唯一性、组分和 split 覆盖。
- 已建立 `src/ml`：提供 dependency-light 的传统 ML baseline，包含 v4 benchmark 表格特征抽取、`MeanRegressor`、闭式解 `RidgeRegressor`、numpy 回归指标、`train_regressor_on_dataset` 训练/评估入口，以及 full/per-phase/early baseline protocol report，不依赖 scikit-learn。
- 已建立 `src/dl/data`：`V4BenchmarkDataset` 消费 v4 benchmark，支持慢变量/超声/光纤三模态、NTC/NCT 格式、split 加载、真正 lazy memmap（取单条样本时才转 float32）、scaler 归一化；训练期增强通过显式 `TimeSeriesAugmentConfig` 开启，默认关闭。
- 已建立 `src/dl/models`：模型注册表 `MODEL_REGISTRY` + `build_model` 工厂，已落地 `CNN1DRegressor`、`TCNRegressor`、`LSTMRegressor`、`TransformerRegressor` 和 `PatchTSTRegressor`。CNN/TCN 支持 `mean/last/attention` 聚合，TCN 支持按 `target_timesteps` 自动扩展感受野。
- 已建立 `src/dl/training`：提供 loss、metrics、轻量 `Trainer`、optimizer 构造、evaluate/predict 和 checkpoint 保存/加载。当前仍没有 argparse 训练 CLI、LR scheduler、early stopping 或完整 run 报告输出。
- 已建立 `src/pipeline/layout.py`：定义顶层目录、配置分组和输出分区。
- 已建立 `src/pipeline/generate_benchmark.py`：正式 benchmark 生成入口，CLI 默认按 CPU 自动启用 sequence chunk 多进程生成。
- 已建立 `src/pipeline/precompute_hitran_spectra.py`、`src/pipeline/precompute_hitran_benchmark_cache.py`、`src/pipeline/compare_optical_backends.py`、`src/pipeline/sanity_check_tabulated_spectra.py` 和 `src/pipeline/bundle_waveform_sequence.py`：HITRAN 谱缓存预计算、benchmark 专用 cache 预计算、empirical/HITRAN 小规模对照、外部定量谱表 sanity check、生成后 waveform 压缩包打包入口；本地已用真实 HAPI 下载 CH4/CO2/H2O 两个 NDIR 窗口谱线缓存。
- 已建立长时序协议：支持 `short/standard/long/xlong` 时间轴预设、显式 `timesteps/dt_s` 覆盖、动态 `PhaseSchedule`、`stage_jitter`、`standard_exposure/variable_onset/fast_transient/incomplete_recovery/multi_pulse` 阶段 profile，相关溯源写入 `manifest.json` 和 `metadata/waveform_spec.json`。
- 已建立测试入口：`python -m pytest tests`（覆盖 sim + ml + dl + pipeline）。当前全量验证为 184 passed。

## 目标目录

```text
src/
configs/
data/
outputs/
docs/
experiments/
tests/
```

`configs` 固定拆为 `data/model/train/eval/experiment`。`outputs` 固定拆为 `runs/summary/reports/archive`。

## 新机器环境初始化

依赖入口已经版本化：`pyproject.toml` 声明运行依赖，`requirements.txt` 提供普通 pip 安装入口。建议使用 Python 3.10-3.13。当前项目在 `pyproject.toml` 中显式排除了 Python 3.14，避免科学计算和深度学习依赖在新解释器上暂未提供稳定 wheel 时安装失败。

Windows PowerShell 示例：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pytest tests
```

如果本机只有 Python 3.14，请先安装 Python 3.12 或 3.13 后再创建虚拟环境。若需要 GPU 版 PyTorch，可按本机 CUDA 版本参考 PyTorch 官方安装命令替换默认 `torch` 安装方式。

`data/hitran_cache*/` 与 `outputs/runs/*` 是本地缓存和实验产物，已被 `.gitignore` 排除，不会随远程仓库同步。新机器需要从旧机器复制这些目录，或按下面的 HITRAN 预计算命令重新生成缓存。

当前新机器已验证的可用环境为 Python 3.12.10 虚拟环境，核心依赖包括 `numpy 2.4.6`、`scipy 1.17.1`、`torch 2.12.0+cpu`、`pytest 9.0.3` 和 `hitran-api 1.3.0.0`；`.\.venv\Scripts\python -m pytest tests` 当前全量通过 184 个测试。

## 核心语义

- `mixture_id` 是唯一配气方案 ID，是正式 split 和汇总的业务主键。
- `sequence_id` 是时序样本实例 ID，只服务存储、索引、张量对齐和调试。
- 正式新数据不得再使用 `mixture_id = sequence_id` 的隐式语义。

## 生成 smoke benchmark

```powershell
python -m pipeline.precompute_hitran_benchmark_cache --cache-root data/hitran_cache --sequences 32 --seed 42
python -m pipeline.generate_benchmark --output-root data --dataset wv4-smoke --sequences 32 --seed 42 --storage npz
# 或切换随机采样：
python -m pipeline.precompute_hitran_benchmark_cache --cache-root data/hitran_cache --sequences 32 --seed 42 --sampling-strategy random
python -m pipeline.generate_benchmark --output-root data --dataset wv4-smoke --sequences 32 --seed 42 --storage npz --sampling-strategy random
# 或覆盖声程候选：
python -m pipeline.generate_benchmark --output-root data --dataset wv4-smoke --sequences 32 --seed 42 --storage npz --path-lms 0.20,0.25,0.30,0.35,0.40
# 或生成长时序 / 多脉冲协议数据：
python -m pipeline.generate_benchmark --output-root data --dataset wv4-long --sequences 32 --seed 42 --storage npz --time-axis-preset long --stage-profile multi_pulse --stage-jitter 0.05
# 或显式使用旧经验光学路径（不需要 HITRAN cache）：
python -m pipeline.generate_benchmark --output-root data --dataset wv4-smoke-empirical --sequences 32 --seed 42 --storage npz --optical-absorption-backend empirical_v1
```

当前生成主线已经落地条件表、索引表、标签表、split、manifest、validation summary、slow 张量、超声 waveform、显式超声 TOF/观测 TOF/质量派生数组、光纤麦克风 waveform、metadata 和 scaler。默认使用 LHS 采样，默认声程候选为 `(0.20, 0.25, 0.30, 0.35, 0.40)`，短 phase 下会均匀覆盖声程候选端点。长时序协议通过 `--time-axis-preset`、`--timesteps`、`--dt-s`、`--stage-profile` 和 `--stage-jitter` 控制，默认 `standard_exposure` 且 `stage_jitter=0` 时保持旧四阶段兼容语义。NDIR 光学通道默认使用 `hitran_hapi_v1`，按每条 condition 的温压、每个 timestep 的当前组分和 `L_m` 做滤光片积分；H2O 由 `T/P/RH` 换算。benchmark 生成只读 cache，cache 缺失会在写出 dataset 前失败；`empirical_v1` 仍可通过 CLI 显式选择。本地表格谱积分原型为 `tabulated_spectrum_v1`，HITRAN/PNNL 谱线积分路线见 `docs/SPECTRAL_INTEGRATION_PLAN.md`。声学链路的当前模型名会写入 `manifest.json` 和 `metadata/waveform_spec.json`：`ultrasonic` 是 `tof_observed_transducer_proxy_v1`，包含系统延迟、触发抖动、延迟修正、二阶谐振换能器响应和 TOF 质量指标；`fiber_mic` 是 `fiber_interferometric_proxy_v1`，包含探头声压、光纤相位转导、线性解调、电噪声、饱和和 DAQ 量化代理。正式 v4 只写 `splits/train.csv` 这类新命名，不写 V3 的 `train_sequence_ids.csv` 等旧命名。

性能参数需要区分 CLI 与 Python API：`python -m pipeline.generate_benchmark` 未传 `--workers` 时会使用 `default_worker_count(sequence_count)`，即默认保留 2 个逻辑线程给系统并最多使用 24 个 worker；`python -m pipeline.precompute_hitran_benchmark_cache` 是内存密集型 HAPI 任务，CLI 默认最多使用 4 个 worker，并通过有界 pending 队列避免一次性提交全部 cache requirement。程序化调用 `BenchmarkGenerationSpec(...)` 时，`workers` 仍默认是 `1`，目的是保留既有 Python API 的串行语义。可选参数 `--chunk-size` 控制每个数据生成 chunk 的 sequence 数，默认按 `ceil(sequences / workers)`；`--temp-dir` 指定 chunk 临时目录；`--keep-chunks` 仅用于调试。大规模正式数据集推荐 `--storage memmap`，需要 `sequences/waveform_sequence.npz` 时在生成后单独运行 `python -m pipeline.bundle_waveform_sequence --dataset-dir data/<dataset>`，避免压缩打包阻塞主生成路径。

## 传统 ML baseline

`src/ml` 提供不依赖 scikit-learn 的传统机器学习最小闭环，用于快速评估 v4 benchmark 生成质量和作为 DL 模型对照基线。

当前能力：

- `ml.features.load_feature_matrix(...)`：按 split CSV 顺序读取 `slow`、`ultrasonic`、`fiber_mic` 模态并打包为表格特征。
- `MLFeatureConfig`：控制模态选择、序列统计量、波形帧级描述符和 slow scaler。
- `MeanRegressor`：多输出均值基线。
- `RidgeRegressor`：纯 numpy 闭式解多输出 ridge baseline，支持标准化和截距项。
- `regression_metrics(...)` / `component_regression_metrics(...)`：numpy 版 MAE、RMSE、R2 和按组分指标。
- `train_regressor_on_dataset(...)`：加载训练 split、拟合模型并评估 train/val/test/extrapolation split。

示例：

```python
from ml import MLFeatureConfig, train_regressor_on_dataset

result = train_regressor_on_dataset(
    "data/wv4-smoke",
    model_config={"name": "ridge", "alpha": 1.0},
    feature_config=MLFeatureConfig(
        modalities=("slow",),
        sequence_statistics=("mean", "last", "slope"),
    ),
    eval_splits=("train", "val", "test"),
)

print(result.evaluations["val"].metrics)
```

新增测试入口：

```powershell
python -m pytest tests/test_ml_baselines.py
```

命令行 baseline protocol report：

```powershell
python -m ml.cli --dataset-dir data/wv4-smoke --protocol --report-path outputs/reports/wv4-smoke-baseline.md
python -m ml.cli --dataset-dir data/wv4-smoke --protocol --json
```

该协议会输出 full-window、按实际 `phase_id` 切分的 per-phase 窗口，以及 early-window 指标，用于正式长时序实验前先检查顺序不敏感 baseline 的表现。

## DL 数据、模型与训练

`src/dl` 当前提供最小训练闭环：

- `V4BenchmarkDataset`：读取 v4 benchmark，按 split 过滤，支持 NTC/NCT、lazy memmap、scaler 和显式时间序列增强。
- `build_model(...)`：通过 `cnn1d`、`tcn`、`lstm`、`transformer`、`patchtst` 构造模型。
- `Trainer`：支持 `fit`、`evaluate`、`predict`、checkpoint 保存/加载。

当前边界：训练配置和 run 输出契约尚未整理成正式 CLI；`Trainer` 不含分布式训练、LR scheduler 或 early stopping。后续正式实验需要先补 `--dataset-dir/--model/--epochs/--output-dir` 训练入口，再跑长时序模型对比。

## HITRAN 光谱预计算

```powershell
python -m pipeline.precompute_hitran_spectra --cache-root data/hitran_cache --channels ch4,co2
python -m pipeline.precompute_hitran_benchmark_cache --cache-root data/hitran_cache --sequences 32 --seed 42
python -m pipeline.compare_optical_backends --cache-root data/hitran_cache
```

从仓库根目录运行时，根层 `pipeline` launcher 包会把 `src` 加入 Python import path，并把子模块解析转发到 `src/pipeline`，因此 README 中的 `python -m pipeline...` 命令可直接执行。预计算入口需要真实 HAPI 环境；缺少 HAPI 时会直接报错，不会生成 fake 谱线。benchmark 专用预计算必须与后续生成使用相同的 `--sequences`、`--seed` 和 `--sampling-strategy`，因为 `hitran_hapi_v1` cache key 按每条 condition 的 `T_C_base/P_MPa_base` 派生。若对应窗口的 HAPI 原始 `.data/.header` 表已在本地，预计算会复用本地真实谱线表并直接通过 `absorptionCoefficient_Voigt` 生成当前温压的 `.npz` cache；只有原始表缺失时才调用 `hapi.fetch` 下载。benchmark 专用预计算实现会先串行确保 HAPI 原始表存在，再以最多 4 个 worker 和有界 pending 队列补齐缺失 `.npz` cache，已有 cache 会直接跳过；若服务器仍出现 OOM 或 `BrokenProcessPool`，保留 cache 并以更低 `--workers` 重跑即可。当前本地环境已验证 `hitran-api 1.3.0.0` 可用，早期已成功下载 CH4/CO2/H2O 在 `2960-3100 cm-1` 与 `2280-2410 cm-1` 两个窗口的谱线，并生成 `.npz` 预计算缓存；当前默认 HITRAN grid 已扩大为 CH4 `2880-3180 cm-1`、CO2 `2250-2445 cm-1`，以覆盖行业参考滤光片 `center ± FWHM`。运行时默认滤光片、气体和 grid 以 `configs/data/spectral-defaults.json` 为 source-of-truth，`src/sim/generation/spectral/defaults.py` 只负责读取并构造 dataclass 常量。`data/hitran_cache*/` 是本地运行缓存，已加入 `.gitignore`，不纳入版本库。

## 外部定量谱表 sanity check

```powershell
$env:PYTHONPATH = "src"
python -m pipeline.sanity_check_tabulated_spectra --cache-root data/hitran_cache --unit per_percent_m --ch4-spectrum path/to/ch4.csv --co2-spectrum path/to/co2.csv --channels ch4,co2
```

外部谱表 CSV 默认列名为 `wavenumber_cm1,absorption_coeff`，单位必须显式指定为 `per_percent_m`、`per_fraction_m` 或 `per_ppm_m`，不会自动猜测。该入口用于把 PNNL/NIST 或仪器导出的定量谱表重采样到当前 HITRAN grid，并与 `hitran_hapi_v1` 做同条件滤光片积分对照。
