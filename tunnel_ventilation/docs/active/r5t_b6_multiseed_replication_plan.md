# R5-T 与 B6 三 Seed 稳定性复核计划

> 状态：待执行
> 日期：2026-07-11
> 目标：在不改变数据、split、特征、模型和超参数的前提下，分别用 3 个新增训练随机种子复核 R5-T observed MLP 与 B6 RawDSP MLP 的单 seed 结论。
> 范围：只评估优化随机性；不在本轮混入 SPXY、新 OOD、B7 residual、特征选择、模型结构或超参数搜索。

## 1. 现有锚点与本轮问题

| 实验 | 历史锚点 seed | val / test / extrap O2 R2 | 对比基线 |
| --- | ---: | ---: | --- |
| R5-T observed MLP | `20260704` | `0.6642 / 0.6462 / 0.5815` | D0-observed Ridge：`0.4226 / 0.4571 / 0.3708` |
| B6 RawDSP MLP | `20260704` | `0.5629 / 0.5244 / 0.4957` | B1 RawDSP Ridge：`0.4280 / 0.4786 / 0.3695` |

两项历史结果都只有一个训练 seed。R5-T 的 observed 输入仍含 simulator 同步生成的测量级数组；B6 已绑定通过的 RawDSP fidelity 与 B1 provenance，是当前可部署特征链路的主验证对象。历史锚点仅保留用于上下文对照，**不计入本轮三 seed 通过判定**。

## 2. 预注册不变量

1. 数据固定为 clean `data/tv3-formal-6000`，继续使用 `random_mixture_id_split_v4` 的既有 split。
2. R5-T 固定 `d0_observed_physics_stats_v1`、864 特征、`raw3` 输出与 `mlp_standardize_targets=true`。
3. B6 固定 `d0_raw_dsp_physics_stats_v1`、1008 特征、`raw3` 输出、同一 RawDSP cache、fidelity metrics 与 B1 reference metrics；不得读取 observed physics 数组。
4. 两组均固定 `(256, 128)`、dropout `0.1`、weight decay `1e-4`、learning rate `1e-3`、batch size `256`、max epochs `200`、patience `20`、loss weights `[1, 2, 1]` 与 CUDA 设备。
5. 不覆盖现有正式 `metrics.json`，每次运行写入唯一的 seed 目录；失败必须保留错误和日志，不得改用 fallback 数据或输入。

## 3. 实验矩阵

新增 seed 固定为 `42`、`123`、`456`。两个模型分别运行 3 次，共 6 次正式训练。6 组结果统一写入 `outputs/tv3_r5t_b6_multiseed/`，由 `scripts/run_r5t_b6_multiseed.py` 一次性编排。

| 组别 | seed | 输出目录 |
| --- | ---: | --- |
| R5-T | 42 | `outputs/tv3_r5t_b6_multiseed/r5t_s42` |
| R5-T | 123 | `outputs/tv3_r5t_b6_multiseed/r5t_s123` |
| R5-T | 456 | `outputs/tv3_r5t_b6_multiseed/r5t_s456` |
| B6 | 42 | `outputs/tv3_r5t_b6_multiseed/b6_s42` |
| B6 | 123 | `outputs/tv3_r5t_b6_multiseed/b6_s123` |
| B6 | 456 | `outputs/tv3_r5t_b6_multiseed/b6_s456` |

汇总文件（同目录）：

| 文件 | 内容 |
| --- | --- |
| `runs.jsonl` | 每次运行的状态、耗时、审计结果 |
| `summary.json` | 6 组运行记录摘要 |
| `replication_report.json` | O2 R2 统计、Ridge 差值、单 seed 门槛与三 seed 判定 |

## 4. 执行前核查

在服务器的 `tunnel_ventilation` 根目录执行：

```bash
python -m pytest tests/test_tv3_r5_mlp.py tests/test_tv3_raw_dsp_pipeline.py tests/test_d2b_frame_fidelity_audit.py -q
python -c "import json; p=json.load(open('outputs/tv3_d2b/raw_dsp_frame_fidelity/metrics.json', encoding='utf-8')); assert p['status'] == 'passed'; print(p['source']['cache_build_signature'])"
```

第二条只适用于 B6，但可在所有运行前统一执行。若测试或 RawDSP fidelity 未通过，停止本轮；不得以新的 cache、模板、特征或数据替换当前证据链。

## 5. 正式运行命令

数据集已在服务器时，在 `tunnel_ventilation` 根目录一键执行 6 组训练、审计与汇总：

```bash
python scripts/run_r5t_b6_multiseed.py
```

可选参数：

```bash
# 数据集路径非默认时
DATASET_DIR=/path/to/tv3-formal-6000 python scripts/run_r5t_b6_multiseed.py --dataset-dir "$DATASET_DIR"

# 只重跑某一组或某一 seed
python scripts/run_r5t_b6_multiseed.py --groups b6 --seeds 42
python scripts/run_r5t_b6_multiseed.py --skip-preflight

# 仅打印将执行的命令
python scripts/run_r5t_b6_multiseed.py --dry-run
```

脚本顺序：preflight 测试 → RawDSP fidelity 核查 → 6 次 `run_tv3_rocket_baseline` → 写入 `runs.jsonl` / `summary.json` / `replication_report.json`。每次训练仍写入独立 seed 子目录的 `metrics.json`，不覆盖历史 `outputs/tv3_r5/` 与 `outputs/tv3_d2b/` 中的单 seed 锚点。

如需手工单条复现，等价命令为：

```bash
python -m tv3.pipeline.run_tv3_rocket_baseline --config configs/tv3_r5_mlp_target_scaled.json --seed 42 --output-dir outputs/tv3_r5t_b6_multiseed/r5t_s42
python -m tv3.pipeline.run_tv3_rocket_baseline --config configs/tv3_d2b_raw_dsp_mlp_target_scaled.json --seed 42 --output-dir outputs/tv3_r5t_b6_multiseed/b6_s42
```

## 6. 审计与汇总

每个 `metrics.json` 必须核对：

1. `diagnostics.model_config.seed` 等于目录名中的 seed；`standardize_targets=true`，`out_dim=3`。
2. R5-T 的 feature builder/count 为 `d0_observed_physics_stats_v1` / `864`；B6 为 `d0_raw_dsp_physics_stats_v1` / `1008`。
3. B6 的 `raw_dsp_fidelity.status=passed`、`raw_dsp_provenance`、B1 reference path 与历史正式锚点一致。
4. train、val、test、extrapolation 的 CO2/O2/N2 R2、MAE、RMSE、`sum_abs_error`、best epoch、parameter count 与 train-val O2 gap 完整存在。

汇总表按“模型 × seed”逐行记录，不只保留最佳结果。对每个模型额外报告 val/test/extrap O2 R2 的均值、标准差、最小值、最大值，以及相对对应 Ridge 基线的逐 seed 差值。

## 7. 通过与分支规则

### 7.1 单 seed 门槛

| 模型 | val | test | extrapolation |
| --- | --- | --- | --- |
| R5-T | `>= 0.4726`（D0 + 0.05） | `>= 0.5071`（D0 + 0.05） | `>= 0.4208`（D0 + 0.05） |
| B6 | `>= 0.4780`（B1 + 0.05） | `> 0.4786`（B1） | `> 0.3695`（B1） |

三项评估集必须在同一 seed 同时通过；仅 val 通过不计为通过。`sum_abs_error` 仅作 raw3 监控，不改写组分预测或替代 O2 判据。

### 7.2 三 seed 结论

- **稳定通过**：三个新增 seed 均满足对应单 seed 门槛。R5-T 与 B6 才可分别写为“在当前随机 split 下稳定复核通过”。
- **证据不足**：恰有 2 个 seed 通过。保留结果和方差，不继续调 `(256,128)` 超参数；进入下一阶段的 split 稳定性审查，再决定是否做 residual 对照。
- **未通过**：至多 1 个 seed 通过。单 seed 增益视为不稳定，不将其写成可部署非线性头结论；B6 转入 OOF Ridge residual 对照，R5-T 保留为 observed-space 候选。

无论结论如何，都不得把当前 `extrapolation` 称为物理 OOD；它仍是 random split 的剩余集合。

## 8. 本轮完成后的唯一决策

1. 两者均稳定通过：冻结本轮特征和训练配方，优先实现 B7 的 OOF Ridge residual MLP，并在新 split 协议下比较 absolute 与 residual 头。
2. R5-T 稳定、B6 不稳定：定位为 observed 特征中仍有未由 RawDSP 恢复的信息，先做 RawDSP 特征契约差异审计，不扩大可微前端。
3. B6 稳定、R5-T 不稳定：以 B6 作为可部署主线，observed MLP 不作泛化结论。
4. 两者均不稳定：停止当前 flat MLP 配方的重复调参，保留 Ridge 基线并转入 OOF residual 对照与新 split 验证。

## 9. 交付物

统一目录 `outputs/tv3_r5t_b6_multiseed/`：

- 6 个不可覆盖的 seed 级 `metrics.json`（`r5t_s*` / `b6_s*` 子目录）；
- `runs.jsonl`、`summary.json`、`replication_report.json`（含 `model × seed × split` 表、均值 / 标准差 / 最差 seed、Ridge 差值、运行时间与三 seed 判定）；
- 对项目记忆库、R5 与 B6 实施计划的正式回填，只在 3 个 seed 全部完成并审计后进行。
