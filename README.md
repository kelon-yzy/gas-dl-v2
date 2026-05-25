# 正式实验 v4

本项目是 `V3_正式实验` 的目标架构重构版。第一阶段目标是先建立清晰主线：稳定主键语义、统一数据资产边界、统一 run 输出契约，再逐步迁移 `sim`、`ml`、`dl` 和 `pipeline` 能力。

## 当前状态

- 已建立 `src/sim/core`：定义 benchmark、mixture、sequence 的核心 ID 语义。
- 已建立 `src/sim/generation`：提供正式 benchmark 生成的最小垂直切片，默认使用拉丁超立方采样（LHS）。
- 已建立 `src/sim/packaging`：定义 `condition_grid`、`sequence_index`、split rows、manifest 和正式 run 最小输出契约。
- 已建立 `src/sim/validation`：校验旧字段、ID 唯一性、组分和 split 覆盖。
- 已建立 `src/dl/data`：`V4BenchmarkDataset` 消费 v4 benchmark，支持慢变量/超声/光纤三模态、NTC/NCT 格式、split 加载、lazy memmap、scaler 归一化。
- 已建立 `src/dl/models`：模型注册表 `MODEL_REGISTRY` + `build_model` 工厂，已落地 `CNN1DRegressor`。
- 已建立 `src/pipeline/layout.py`：定义顶层目录、配置分组和输出分区。
- 已建立 `src/pipeline/generate_benchmark.py`：正式 benchmark 生成入口。
- 已建立测试入口：`python -m pytest tests`（57 个测试，覆盖 sim + dl）。

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

## 核心语义

- `mixture_id` 是唯一配气方案 ID，是正式 split 和汇总的业务主键。
- `sequence_id` 是时序样本实例 ID，只服务存储、索引、张量对齐和调试。
- 正式新数据不得再使用 `mixture_id = sequence_id` 的隐式语义。

## 生成 smoke benchmark

```powershell
python -m pipeline.generate_benchmark --output-root data --dataset wv4-smoke --sequences 32 --seed 42 --storage npz
# 或切换随机采样：
python -m pipeline.generate_benchmark --output-root data --dataset wv4-smoke --sequences 32 --seed 42 --storage npz --sampling-strategy random
# 或覆盖声程候选：
python -m pipeline.generate_benchmark --output-root data --dataset wv4-smoke --sequences 32 --seed 42 --storage npz --path-lms 0.20,0.25,0.30,0.35,0.40
```

当前生成主线已经落地条件表、索引表、标签表、split、manifest、validation summary、slow 张量、超声 waveform、光纤麦克风 waveform、metadata 和 scaler。默认使用 LHS 采样，默认声程候选为 `(0.20, 0.25, 0.30, 0.35, 0.40)`。正式 v4 只写 `splits/train.csv` 这类新命名，不写 V3 的 `train_sequence_ids.csv` 等旧命名。
