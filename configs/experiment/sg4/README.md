# 合成气 (syngas) 实验配置

本目录包含合成气 / 煤气化制气场景下的实验配置。

## 文件

| 文件 | 模型 | 用途 |
|---|---|---|
| `sg4_baseline.json` | CNN1D | DL 基线：`python -m dl.cli --config configs/experiment/sg4/sg4_baseline.json` |
| `sg4_tcn.json` | TCN | 时序卷积基线，target_timesteps=512 |
| `sg4_lstm.json` | LSTM | 循环网络基线，hidden=64 / num_layers=2 |
| `sg4_patchtst.json` | PatchTST | Patch-based Transformer 基线，patch_len=16 / stride=8 |
| `sg4_ridge.json` | Ridge | 传统 ML 基线（闭式解），`python -m ml.cli --config configs/experiment/sg4/sg4_ridge.json` |

5 个 DL/ML 配置通用：`dataset_dir=data/sg4-formal`、`modalities=slow`、`epochs=50`、AdamW + ReduceOnPlateau + early stopping、AMP fp16（DL）。

## 配套数据

`sg4-smoke` benchmark（链路验证用）：

```powershell
python -m pipeline.generate_syngas_benchmark `
    --output-root data --dataset sg4-smoke --sequences 32 --seed 20260626 `
    --timesteps 32 --dt-s 0.5 --optical-absorption-backend empirical_v1 --workers 1
```

`sg4-formal` benchmark（正式实验数据，6000 序列 / 512 时步，与 hg `wv4-formal-hitran-standard-6000` 时间轴对齐）：

```powershell
python -m pipeline.generate_syngas_benchmark `
    --output-root data --dataset sg4-formal --sequences 6000 --seed 20260626 `
    --timesteps 512 --dt-s 0.5 --optical-absorption-backend empirical_v1 `
    --storage memmap --workers 24
```

## 训练编排

`scripts/run_sg4_baseline.py`：5 模型 × 3 seeds (`42 / 123 / 2026`) = 15 runs，自动汇总到 `outputs/sg4_baseline/summary.json` 和 `runs.jsonl`。

## 关键约束

- **预测目标**：4 列 `(x_H2, x_CH4, x_CO2, x_CO)`，sum<100%。
- **背景气**：`x_N2 = 100 - sum(targets)`，仅参与物理仿真，不入 labels。
- **慢通道**：9 个（含 `V_NDIR_CO`）。
- **Loss 限制**：闭包类 loss（`compositional_mse` / `ilr_mse` / `free_component_mse` / `weighted_free_component_mse`）在 syngas 场景被自动拒绝；推荐用 `mse`、`weighted_component_mse`、`mae`、`smooth_l1` 或 `huber`。
- **Target transform**：不允许（ILR/ALR 依赖 sum=100% 闭包假设）。

## 相关文档

- 整体方案：`docs/syngas/adaptation_plan.md`
- LHS 采样：`docs/syngas/lhs_sampling_design.md`
- 物理常数：`docs/syngas/physics_references.md`
- CO 串扰：`docs/syngas/co_crosstalk_design.md`

## 待替换的占位

- CO NDIR 滤光片当前使用 InfraTec I 4.66 μm 行业参考占位（`docs/syngas/physics_references.md` §3）。正式实验前需替换为目标传感器 datasheet。
- CO V-T 弛豫 `alpha_lambda_max_co=0.025` 为中置信类比值，建议做 ablation 扫描 0.015–0.040。
- 串扰系数 `ε_32 (CO2→CO) = 0.005` / `ε_23 (CO→CO2) = 0.002` 为物理量级估算（Step 1 默认无串扰，3×3 矩阵作为 Step 2 ablation）。
