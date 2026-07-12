# B7 冻结、重复 Split 与独立 OOD 协议实施计划

> 状态：**P0，代码已落地；正式矩阵待服务器执行**
> 日期：2026-07-11
> 前置证据：B7 在 clean `tv3-formal-6000` 的当前 random split 上取得 `residual_pass`；三训练 seed 的 O₂ R²均值为 val/test/extrap `0.6964 / 0.7001 / 0.6157`。
> 本文责任：冻结 B7 后的泛化验证协议、数据派生、运行矩阵、审计与判定。B7 的模型实现与既有 random-split 复核记录保留在 `b7_oof_ridge_residual_mlp_implementation_plan.md`；通用 SPXY 算法细节保留在 `spxy_split_implementation_plan.md`。

## Context

B7 OOF Ridge residual MLP 已证明：冻结 `d0_raw_dsp_physics_stats_v1` 的 RawDSP 特征上，Ridge 可承担稳定线性主趋势，低容量 `(64, 64)` residual MLP 能稳定拟合非线性残差。RawDSP 的 frame fidelity 和 B1 Ridge parity 也均已通过，因此当前瓶颈不是继续优化波形提取，也不是扩大 B7。

现有 `raw_dsp_frame_v1/manifest.json` 的模板来源是当前 random split 的 `train.csv`。这对既有 B7 正式结果是正确的，但对任何派生 split 都不能直接复用：若新 test / OOD 行曾参与旧模板构建，RawDSP 前端已在特征阶段接触新评价数据。派生 split 可以硬链接物理波形和 slow 数据，**不能硬链接或复用 split-dependent 的 RawDSP frame cache**；必须用该派生 split 的 train rows 重建 train-calibrated cache 并重新通过 fidelity 审计。

但现有 `random_mixture_id_split_v4` 的 extrapolation 只是 shuffle 剩余集，不能解释为物理 OOD。已有三 training seed 只量化了模型初始化与训练方差，尚未量化数据划分方差，也没有独立的 OOD 选择器。因此，B7 目前只能称为**默认 raw-DSP 头候选**，不能据此写出部署泛化结论。

本协议的原则是先固定模型，再改变评价条件。任何由新 split 产生的结果都不得反向触发 B7 的结构、特征、损失或训练超参数搜索。

## Task

### 1. 目标与非目标

目标：

1. 冻结 B7 为唯一的 raw-DSP 非线性候选，复核其相对 B1 RawDSP Ridge 的增益是否跨 split、训练 seed 与独立 OOD 选择器保持一致。
2. 将 random、LHS 分层、SPXY 训练覆盖和 OOD selector 的作用严格分开，输出可复现的 split × seed × model 结果表。
3. 在全部协议通过前，不把当前约 `0.70` 的 test O₂ R²表述为 OOD 或部署性能。

非目标：

- 不调整 B7 的 `(64,64)`、dropout、学习率、OOF folds 或 OOF seed；不加宽 residual MLP。
- 不同时引入 grouped bottleneck、TabM、可微 RawDSP、质量集合学习、独立 heads、蒸馏或闭包头。
- 不把 O₂ 的 0.8% concentration bins 负 R²解释为模型失败；该现象是已有 oracle 审计确认的物理分辨率边界。

### 2. 冻结契约

| 项目 | 冻结值 | 要求 |
| --- | --- | --- |
| 数据源 | clean `data/tv3-formal-6000` | 只重算 split 与 split-specific RawDSP cache；不重跑仿真，不使用本地旧 600 数据 |
| 特征 | `d0_raw_dsp_physics_stats_v1`，1008 维 | 特征 schema 冻结；每个派生 split 均以自己的 train rows 重建 RawDSP cache，不读 observed / oracle 物理数组作为模型输入 |
| 头部 | `oof_ridge_residual_mlp` | full-train Ridge + OOF 残差目标 + `(64,64)` residual MLP |
| OOF | 5 folds，seed `20260711` | 每条 train row 恰好一次 holdout |
| 训练 | 原 B7 config | target scaling、`[1,2,1]` 权重、raw3、val O₂ R²早停均保持不变 |
| 训练 seeds | `42`、`123`、`456` | 所有新数据集一律使用；不得按 split 替换 seed |
| 对照 | B1 RawDSP Ridge | 每个派生 split 必跑；B6 / R5-T 仅作历史锚点，不作为新 split 的通过线 |
| 早停与选择 | 仅 val | test、extrapolation / OOD 永不参与早停、模型选择或重跑决策 |

标签始终是 `x_CO2`、`x_O2`、`x_N2` 三维 raw3。禁止 N₂ 回填、ILR/ALR、强制闭包损失、true TOF、true sound speed 或 true alpha 进入模型输入。

### 3. Split 与 OOD 定义

#### 3.1 观测量限定

现有 `spxy_split.py` 的通用 `spxy_v1` X 统计含 `ultrasonic_tof_s`、`ultrasonic_sound_speed_m_per_s` 与 `ultrasonic_alpha_true_npm`。它们均在 RawDSP 契约中被列为 simulator / oracle 数组；即使不进入回归模型，若用它们决定 train 覆盖或 OOD 集，评价协议仍会依赖部署时不可观测的隐变量。

本协议必须先实现并显式命名 `spxy_observed_stats_v1`：

- 保留 7 个 slow channels 的 mean、std、min、max、trend；
- 仅使用当个派生 split 的 train-calibrated RawDSP 输出：`ultrasonic_tof_observed_raw_dsp_s`、`ultrasonic_peak_index_raw_dsp`、`ultrasonic_sound_speed_raw_dsp_m_per_s`、`ultrasonic_corr_peak`、`ultrasonic_snr_db`、`ultrasonic_raw_dsp_quality`、`ultrasonic_raw_dsp_accepted` 的序列统计；
- 显式排除 `ultrasonic_tof_s`、`ultrasonic_tof_observed_s`、`ultrasonic_peak_index`、`ultrasonic_sound_speed_m_per_s`、`ultrasonic_sound_speed_estimated_m_per_s` 与 `ultrasonic_alpha_true_npm`，不以任何 true / oracle 物理量替代；
- 在 `split_summary.json` 写入 `x_feature_profile`、特征名、特征数与特征矩阵 hash。

不得静默修改旧 `spxy_v1` 的特征含义。旧 profile 如需保留，只能登记为 `oracle_split_sensitivity`，不得成为 B7 正式 OOD 结论依据。

#### 3.2 协议矩阵

所有比例维持 train/val/test/extrapolation = `70/15/10/5`。split seeds 固定为 `20260704`、`20260712`、`20260720`；首个 seed 与既有正式 random 数据保持可追溯关联。

| ID | 数据划分 | 作用 | 是否可作为 OOD 证据 |
| --- | --- | --- | --- |
| R | `random_mixture_id_split_v4`，3 个 split seeds | 现有基线与纯随机方差控制 | 否；extrapolation 仅标记为 random remainder |
| L | `lhs_stratified_split_v1`，3 个 split seeds | 更简单的 ID 覆盖对照 | 否；仅用于 ID 稳定性比较 |
| S-Y | `spxy_observed_stats_v1(alpha=0.5)` + `y_margin_ood`，3 个 split seeds | 主协议：ID 覆盖与组分边界 OOD 分离 | 是，组分边界 OOD |
| S-L | `spxy_observed_stats_v1(alpha=0.5)` + `lhs_boundary`，3 个 split seeds | 独立 selector 复核 | 是，LHS 边界 OOD |

`alpha=0.3` 仅在 S-Y 通过后作为预声明敏感性实验；它不能替代 S-L。`kmeans_boundary` 仅在 `lhs_boundary` 的几何诊断退化时，由失败记录明确触发，不允许自动回退或混写结果。

若某 selector 在不同 split seed 下产生相同 OOD 集，必须在汇总中报告 `ood_set_hash` 与重叠率。此时重复 seed 只能量化 train、val、test 的划分方差，不能被误称为 OOD selector 方差；S-Y 与 S-L 的差异才承担 selector 独立性检验。

### 4. 数据派生与审计

派生数据集只硬链接既有物理波形、waveform scale、slow、labels、metadata 和 condition grid。`features/raw_dsp/` 必须跳过硬链接，并在新 split 写入后重建；新增目录保存 split CSV、split-specific RawDSP cache、summary 和派生审计信息：

```text
data/tv3-formal-6000-splits/
  random_s20260704/
  random_s20260712/
  random_s20260720/
  lhs_s20260704/
  ...
  spxy_observed_a05_ymargin_s20260704/
  ...
  spxy_observed_a05_lhsboundary_s20260704/
  ...
```

每个派生目录必须记录：

1. source dataset manifest / labels / condition grid 的 hash；
2. `split_policy`、`split_seed`、`spxy_alpha`、`x_feature_profile`、OOD selector 与 `ood_set_hash`；
3. 四个 split 的 sample count、`mixture_id` count、集合互斥与总数守恒；
4. CO₂、O₂、N₂、L、T、RH 的范围和分位数；
5. ID test 相对 train 的最近邻距离，以及 OOD 相对 train 的最近邻、凸包或 selector 距离摘要；
6. RawDSP manifest 的 `template_source_split=train`、template source sequence-id hash、template digest、frame cache build signature 与 split hash；
7. 所有诊断失败、selector 退化、cache provenance 不匹配或 hard-link 失败的明确错误信息。

最小命令模板如下；正式执行前须先在小目录 smoke 并确认 `spxy_observed_stats_v1` 已实际写入 summary：

```bash
python scripts/recompute_tv3_split.py \
  --source-dir data/tv3-formal-6000 \
  --output-dir data/tv3-formal-6000-splits/spxy_observed_a05_ymargin_s20260704 \
  --split-strategy spxy_v1 \
  --spxy-alpha 0.5 \
  --extrapolation-strategy y_margin_ood \
  --seed 20260704
```

`recompute_tv3_split.py` 需新增显式 `--spxy-x-profile observed_v1` 或等价的冻结 config，并默认跳过 source 的 `features/raw_dsp/`。RawDSP builder 必须在新 split 写入后，以新 train rows 构建相同 schema 的 frame cache；未指定 profile 或 cache provenance 不匹配时，不得把旧的含 true alpha profile 或旧 random-train cache 当作本协议默认值。

### 5. 运行矩阵与产物

对每个派生 split：先完成该 split 的 RawDSP frame fidelity 与 cache provenance 审计，再运行一次 B1，最后运行 B7 的三个 training seeds。B1 不依赖 training seed，但必须保留该 split 的独立 metrics，以免把旧 random-split 的 B1 当作新 split 基线。

```text
outputs/tv3_b7_protocol/
  protocol_manifest.json
  runs.jsonl
  split_metrics.json
  result_matrix.csv
  result_matrix.md
  random_s20260704/
    b1/metrics.json
    b7_s42/metrics.json
    b7_s123/metrics.json
    b7_s456/metrics.json
  spxy_observed_a05_ymargin_s20260704/
    ...
```

新增 `scripts/run_b7_repeated_split_ood_protocol.py` 只负责编排现有 B1 / B7 入口、读取已派生数据集、审计产物和汇总；不得复制训练代码或重定义 B7 超参数。每条 run 记录至少包含：

- protocol ID、split seed、training seed、dataset/split hash、B7 config hash；
- O₂、CO₂、N₂ 的 R²、MAE、RMSE，`sum_abs_error`，train/val gap；
- O₂ 0.8% bins 指标；
- 相对同 split B1 的 `ΔR²`；
- 运行状态、错误和时长。

### 6. 判定、停止条件与后续分流

#### 6.1 `protocol_pass`

B7 保持“默认 raw-DSP 头候选”需要同时满足：

1. 所有 split、RawDSP frame cache 与模型审计通过，无 ID overlap、无 source / cache hash 漂移、无旧 random-train template 复用、无 oracle X profile；
2. 在 R、L、S-Y、S-L 的每个协议内，B7 相对同 split B1 的 test O₂ `ΔR²` 均值为正；
3. 在 S-Y 与 S-L 的 OOD split 内，B7 相对 B1 的 O₂ `ΔR²` 均值为正，且不存在三个 training seed 全部为负的 split seed；
4. 任何增益都同步出现在 test 与对应 OOD，而非仅 val；
5. `sum_abs_error`、CO₂/N₂ 误差和 O₂ bins 指标均完整报告，不以强制闭包替换 raw3 结果。

汇总必须报告 `mean ± std`、最差 training seed、最差 split seed 和逐条 paired difference。3 个 seed 不足以作独立显著性宣称，不报告 p 值；本协议判断的是预先冻结的方向一致性和可复现性。

#### 6.2 停止与分流

| 触发条件 | 结论与动作 |
| --- | --- |
| split 审计或 observed-only profile 未通过 | 先修 split 实现；不训练任何新模型 |
| B7 只在 random / L 协议优于 B1 | 保留其为 ID 候选，不写 OOD 泛化；不进入 TabM |
| B7 在 S-Y、S-L 的任一协议内 test/OOD 均值不优于 B1 | 接受当前非线性增益依赖划分；停止扩大回归头，转入 shift 诊断 |
| S-Y 通过、S-L 失败 | 结论限于组分边界 OOD；补做 L、T、RH、SNR 条件 holdout，不引入新模型 |
| 两个 selector 都通过 | 进入模块 C 的分组 bottleneck → residual TabM 单变量消融 |
| 所有严格 OOD 下稳定低于 0.70 | 报告声学路线与数据覆盖的综合上限；不以持续调参追逐阈值 |

模块 C 的开始条件是 `protocol_pass`，而不是某个 random-split 单次 R²。B7 通过后，首个新实验只能是同一残差结构上的 grouped bottleneck；residual TabM 只能在该对照完成后启动。

## Format

### 1. 预期代码与测试范围

| 范围 | 必需改动 |
| --- | --- |
| `tv3/sim/packaging/spxy_split.py` | 新增显式 observed-only X profile；保留旧含 oracle profile 的兼容入口但禁止其作为本协议默认值 |
| `scripts/recompute_tv3_split.py` | 接收 profile / protocol metadata，跳过 split-dependent RawDSP cache 的硬链接，写入 hash 与诊断，不重跑物理仿真 |
| `tv3/pipeline/build_tv3_raw_dsp_features.py` 及其配置入口 | 对每个派生 split 以本 split 的 `train.csv` 重建 train-calibrated RawDSP cache，并写 template source / build signature |
| `scripts/run_b7_repeated_split_ood_protocol.py` | 新增协议编排、已存在产物跳过规则和统一汇总 |
| `tv3/ml/*` | 不改 B7 模型逻辑；只复用既有 B1 / B7 训练入口 |
| `tests/test_spxy_split.py` | 覆盖 observed-only profile、oracle profile 显式标记、selector 退化和 split 审计 |
| RawDSP cache 测试 | 覆盖派生 split 不复用 source cache、模板只来自本 split train、manifest 与 split hash 绑定 |
| 新增协议测试 | 覆盖矩阵完整性、B1 / B7 配对、OOD 不参与早停、result matrix 字段与 hash 一致性 |

### 2. 最小验证

```bash
python -m pytest -q tests/test_spxy_split.py tests/test_tv3_b7_oof_residual.py
python scripts/recompute_tv3_split.py --help
python scripts/run_b7_repeated_split_ood_protocol.py --dry-run
```

在完整派生和服务器训练结束后，额外验证：所有 `split_summary.json` 的 hash 可追溯、所有 B7 run 均使用冻结 config、结果表没有缺失的 B1 配对行。

### 3. 文档回填规则

只有完成完整协议矩阵且生成 `result_matrix.md` 后，才更新：

1. `掘进通风项目记忆库.md` 的正式结论、执行门和停止条件；
2. `掘进通风_深度学习算法研究方向与文献路线.md` 的 P0 状态与实验矩阵；
3. 本文的“实施记录”。

中途 smoke、单个 split 或失败日志只写入本文件的实施记录与 `outputs/tv3_b7_protocol/`，不得提前提升 B7 的泛化结论。

## 实施记录

### 2026-07-11 — 代码落地（协议编排就绪，正式矩阵未跑）

已完成：

1. `tv3/sim/packaging/spxy_split.py`
   - 新增显式 X profile：`observed_v1` → summary `x_feature_profile=spxy_observed_stats_v1`（50 维，仅 slow + RawDSP observed 序列统计）
   - 旧默认保留为 `oracle_v1` → summary `oracle_split_sensitivity`（42 维，含 true tof/speed/alpha）；不得作为 B7 正式 OOD 依据
   - summary 写入 `x_feature_names` / `x_feature_count` / `x_feature_matrix_hash` / `ood_set_hash` / `split_seed`

2. `scripts/recompute_tv3_split.py`
   - `--spxy-x-profile {observed_v1,oracle_v1}`；`observed_v1` 强制 `--raw-dsp-cache-dir`（仅 SPXY X bootstrap，角色写入 `raw_dsp_bootstrap`）
   - 默认跳过硬链接 `features/`（含 `features/raw_dsp/`），写入 `RAW_DSP_MUST_REBUILD.txt`
   - 写入 `source_hashes`、`split_hash`、组分/条件范围分位数

3. `tv3/pipeline/build_tv3_raw_dsp_features.py`
   - RawDSP manifest / build_signature 绑定 `split_hash`、`split_policy`、`split_seed`、`template_source_sequence_ids_digest`

4. `scripts/run_b7_repeated_split_ood_protocol.py`
   - 编排 R / L / S-Y / S-L × 3 split seeds；每 split 重建 RawDSP → fidelity → B1 → B7×{42,123,456}
   - 产出 `protocol_manifest.json` / `runs.jsonl` / `result_matrix.csv|md` / `split_metrics.json`
   - `--dry-run` 可用；已存在产物可跳过

5. 测试
   - `tests/test_spxy_split.py`：observed/oracle profile、selector 退化、recompute 跳过 RawDSP hardlink
   - `tests/test_b7_repeated_split_ood_protocol.py`：矩阵完整性、B1 配对字段、val-only 早停审计、`protocol_pass` 判定、dry-run manifest

最小验证（已跑通）：

```text
python -m pytest -q tests/test_spxy_split.py tests/test_tv3_b7_oof_residual.py
# 40 passed
python -m pytest -q tests/test_b7_repeated_split_ood_protocol.py
# 5 passed
python scripts/recompute_tv3_split.py --help
python scripts/run_b7_repeated_split_ood_protocol.py --dry-run
```

尚未执行（需服务器 `data/tv3-formal-6000` + source RawDSP bootstrap）：

- 完整 12 个派生 split 的 RawDSP 重建与 fidelity
- B1 / B7 全矩阵训练与 `result_matrix.md` 回填
- 记忆库 / 文献路线的正式结论更新（按本文档回填规则，等完整协议后再写）

SPXY observed 的 X 划分目前显式借用 source RawDSP 作 bootstrap（写入 summary），模型特征仍必须用派生 split 的 train-calibrated 重建 cache；不是静默复用旧 random-train 模型特征。
