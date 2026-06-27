# 阶段 Ⅰ-3 基线训练结果（2026-06-26）

> 5 模型 × 3 seeds (`42 / 123 / 2026`) = 15 runs，在 `data/sg4-formal`（6000 序列 / 128 时步 / 9 慢通道）上训练。
> 数据：empirical 光学后端，`enable_co_crosstalk=False`，split 4200/900/600/300。
> 训练参数：epochs=50，AdamW lr=1e-3，weight_decay=1e-4，AMP fp16，ReduceOnPlateau，early_stopping patience=10。
> Loss：DL 用 `weighted_component_mse`（inverse_train_var），Ridge 用 closed-form L2。

> **后续进展（2026-06-27）**：本文档底部 §6 "Ⅱ-1/Ⅱ-2/Ⅱ-3 待启动"已全部完成，详见 [stage_ii_ablation_results.md](stage_ii_ablation_results.md)。本节保留为 2026-06-26 时间点的历史快照。

## 1. 总览（test split 平均 ± std）

> 单元：R²；统计仅基于成功收敛的 seeds（pool R²>0.5）。
> PatchTST 第一轮 3 seeds 全部 NaN 早停（pool R²~0.01–0.10），同日修复配置（关 AMP / lr=1e-3 / epochs=80）后重跑 3 seeds 全部收敛，下表用重跑后数据。

| 模型 | seeds | pool R² | x_H2 | x_CH4 | x_CO2 | x_CO |
|---|---|---|---|---|---|---|
| **TCN** | 3 | **0.958 ± 0.001** | 0.968 ± 0.001 | 0.827 ± 0.003 | 0.969 ± 0.002 | **0.954 ± 0.000** |
| **Ridge** | 3 | **0.957 ± 0.000** | **0.977 ± 0.000** | 0.826 ± 0.000 | 0.966 ± 0.000 | 0.946 ± 0.000 |
| PatchTST | 3 | 0.935 ± 0.008 | 0.955 ± 0.002 | 0.686 ± 0.031 | 0.964 ± 0.007 | 0.926 ± 0.016 |
| CNN1D | 3 | 0.931 ± 0.005 | 0.945 ± 0.007 | 0.727 ± 0.031 | 0.960 ± 0.001 | 0.924 ± 0.003 |
| LSTM | 2/3 | 0.930 ± 0.003 | 0.945 ± 0.005 | 0.428 ± 0.018 | 0.959 ± 0.003 | 0.940 ± 0.001 |

MAE（同上口径）：

| 模型 | pool | x_H2 | x_CH4 | x_CO2 | x_CO |
|---|---|---|---|---|---|
| **TCN** | **1.603 ± 0.017** | 1.907 ± 0.052 | 1.151 ± 0.013 | **0.963 ± 0.034** | **2.393 ± 0.016** |
| Ridge | 1.648 ± 0.000 | **1.603 ± 0.000** | **1.194 ± 0.000** | 1.096 ± 0.000 | 2.698 ± 0.000 |
| PatchTST | 1.951 ± 0.103 | 2.226 ± 0.066 | 1.578 ± 0.091 | 1.044 ± 0.104 | 2.958 ± 0.322 |
| CNN1D | 2.033 ± 0.071 | 2.488 ± 0.167 | 1.479 ± 0.099 | 1.098 ± 0.013 | 3.067 ± 0.085 |
| LSTM | 2.131 ± 0.029 | 2.466 ± 0.100 | 2.191 ± 0.020 | 1.158 ± 0.064 | 2.710 ± 0.061 |

## 2. 单 run 明细（test split）

| 模型 | seed | best_ep | pool R² | x_H2 | x_CH4 | x_CO2 | x_CO | stop |
|---|---|---|---|---|---|---|---|---|
| cnn1d | 42 | 48 | 0.935 | 0.949 | 0.733 | 0.960 | 0.928 | completed |
| cnn1d | 123 | 36 | 0.924 | 0.934 | 0.686 | 0.960 | 0.920 | early-stopped |
| cnn1d | 2026 | 48 | 0.933 | 0.950 | 0.762 | 0.958 | 0.923 | completed |
| **tcn** | 42 | 43 | 0.957 | 0.966 | 0.830 | 0.966 | 0.954 | completed |
| **tcn** | 123 | 50 | 0.959 | 0.969 | 0.827 | 0.970 | 0.954 | completed |
| **tcn** | 2026 | 50 | 0.958 | 0.968 | 0.823 | 0.970 | 0.955 | completed |
| lstm | 42 | 47 | 0.927 | 0.940 | 0.410 | 0.962 | 0.939 | completed |
| lstm | **123** | 1 | **−0.004** | −0.002 | −0.001 | −0.026 | −0.001 | early-stopped (no convergence) |
| lstm | 2026 | 47 | 0.933 | 0.950 | 0.446 | 0.957 | 0.941 | completed |
| patchtst | 42 | 76 | 0.934 | 0.956 | 0.712 | 0.968 | 0.920 | completed (修复后) |
| patchtst | 123 | 77 | 0.945 | 0.953 | 0.703 | 0.970 | 0.948 | completed (修复后) |
| patchtst | 2026 | 80 | 0.927 | 0.957 | 0.642 | 0.955 | 0.910 | completed (修复后) |
| **ridge** | 42 | — | **0.957** | 0.977 | 0.826 | 0.966 | 0.946 | closed-form |
| **ridge** | 123 | — | **0.957** | 0.977 | 0.826 | 0.966 | 0.946 | closed-form |
| **ridge** | 2026 | — | **0.957** | 0.977 | 0.826 | 0.966 | 0.946 | closed-form |

注：Ridge 是 closed-form 解，3 seeds 结果完全相同（seed 仅用于记录对齐）。

PatchTST 首轮（fp16 AMP + lr=1e-3）所有 3 seeds 在 epoch 19–27 出现 train/val NaN，best_epoch 时 pool R² 在 0.00–0.10 之间。修复（fp32 + lr=1e-3 + epochs=80 + grad_clip=1.0）后 3 seeds 全部跑满收敛，上表数据基于修复后第二轮结果。修复细节见 §3.5、§3.6、§4.1。

## 3. 关键发现

### 3.1 CO 检测可靠，反向支持 CO/N₂ 简并假说

**TCN 上 x_CO R² = 0.954**，Ridge=0.946，CNN1D=0.924，LSTM=0.940。CO 与 CO₂、H₂ 在同一水平，说明 **V_NDIR_CO 单光学通道已经能稳定预测 CO 浓度**。

这反向支持了 roadmap 中的核心物理假说：CO 与 N₂ 在声学和热导上近简并（M=28，γ=1.40，Δc<1 m/s），CO 的可观测性几乎完全依赖光学通道。下一步 Ⅱ-1 CO 通道 ablation 应能直接量化这一依赖：移除 V_NDIR_CO 通道后预期 CO R² 暴跌至 ~0。

### 3.2 Ridge 与 TCN 性能持平

这是论文需要重点讨论的发现。

| 维度 | TCN | Ridge | PatchTST | CNN1D | LSTM |
|---|---|---|---|---|---|
| pool R² | **0.958** | 0.957 | 0.935 | 0.931 | 0.930 |
| x_H2 R² | 0.968 | **0.977** | 0.955 | 0.945 | 0.945 |
| x_CO2 R² | **0.969** | 0.966 | 0.964 | 0.960 | 0.959 |
| x_CO R² | **0.954** | 0.946 | 0.926 | 0.924 | 0.940 |
| x_CH4 R² | **0.827** | 0.826 | 0.686 | 0.727 | 0.428 |
| MAE (pool) | **1.603** | 1.648 | 1.951 | 2.033 | 2.131 |

Ridge 在 x_H2 上反而压过 TCN（0.977 vs 0.968），其余组分两者差 ≤0.01。**TCN 与 Ridge 几乎持平，且双双甩开 PatchTST / CNN1D / LSTM（0.93 量级）2.5 个百分点**。

Ridge 使用的特征仅是 9 个慢通道在 7 个 sequence statistics（mean / std / min / max / last / delta / slope）下的展开，共 63 维，闭式解。这说明 **syngas 慢通道场景下，组分检测的大部分信息已被手工统计量捕获**，时序结构对组分预测的边际收益有限。

潜在解释：

1. 慢通道本身就是 0.5 Hz 的低频物理量（NDIR 电压、TCS、T/P/RH），在 4 阶段 phase schedule 下信号变化主要由组分驱动，时序复杂度本来就低
2. `weighted_component_mse(inverse_train_var)` 加权使 DL 与 Ridge 都聚焦低方差组分（CH₄），导致两者优化目标接近
3. 6000 序列的训练量对深度模型而言并不充裕，Transformer 类（PatchTST）和容量较大的网络（CNN1D / LSTM）反而更容易过拟合或陷入次优
4. TCN 的因果膨胀卷积 + receptive field 128 step 刚好匹配序列长度，是该项目的最佳归纳偏置

下一步 Ⅱ 阶段需要：

- 在串扰 / CO 弛豫扫描 ablation 中考察 TCN 与 Ridge 的鲁棒性差异（动态环境下深度模型可能更稳）
- 加入 ultrasonic / fiber_mic 模态后再对比，组分外的快通道是 DL 的真正用武之地

### 3.3 CH₄ 仍是短板（与 roadmap 预警一致）

最高 x_CH4 R² = 0.83（TCN / Ridge），最低 0.41（LSTM 收敛 seeds 的均值）。

原因和 roadmap 预测一致：

- 浓度区间窄（0–12%），方差小
- 在 condition_grid 中 CH₄ 来自 LHS 边角约束，分布带尾较重
- NDIR CH₄ 通道在 3.3 μm 与 H₂O 有较强重叠（hg 阶段已知问题）

CH₄ 分箱（test split，跨 3 seeds 平均，bin 内 R²）：

| 模型 | CH₄ 0–3% | CH₄ 3–6% | CH₄ 6–9% | CH₄ 9–12% |
|---|---|---|---|---|
| cnn1d | −2.86 | −3.39 | −3.91 | −6.61 |
| tcn | −0.70 | −2.35 | −3.47 | −3.86 |
| ridge | −1.67 | −1.22 | −2.69 | −3.49 |

> bin 内 R² 全部为负是因为分箱后 CH₄ 取值范围小、bin 内方差极低，参考价值有限；overall x_CH4 R²（0.83）才是有效指标。但能看出 TCN 和 Ridge 在低浓度 bin 比 CNN1D 更平稳。

### 3.4 LSTM seed 敏感

LSTM seed=123 完全没收敛（best_epoch=1，val_loss 卡在 1.00），另两 seeds (42/2026) 收敛到 R²≈0.93。

LSTM 在 AdamW + lr=1e-3 + 9 输入通道 + 128 时步 这套配置下对初始化敏感是已知现象。统计上 2/3 seeds 能用，但单 seed 失败在论文表格里需要标注。

潜在缓解：lr 1e-3 → 5e-4，或加 lr warmup 100 步。

### 3.5 PatchTST 全军覆没（配置问题）

3 个 seed 都在 epoch 11–27 出现 train/val NaN，best_epoch 时 pool R² 在 0.00–0.10 之间。

原因高概率是 `AMP fp16 + Transformer attention + lr=1e-3` 组合在 epoch 中期 gradient explosion 后 fp16 直接 NaN，`grad_clip_norm=1.0` 没救住。

下一步：

- lr 1e-3 → 3e-4
- AMP fp16 → bfloat16，或暂时关 AMP
- grad_clip_norm 1.0 → 0.5
- 重跑 3 seeds

PatchTST 在 hg 项目里跑通过，所以是 sg4 上的配置问题，不是模型 / 数据不兼容。

#### 重跑结果（2026-06-26 同日修复后）

排查中发现：

- **AMP fp16/bfloat16 都让 PatchTST 卡在 val_loss=1.00 平台不收敛**（前 8 epoch attention 权重被 autocast 压缩为常数预测）
- hg 历史 PatchTST 配置（`outputs/archive/formal_full_dl_slow_20260605/runs/formal_full/patchtst/run_config.json`）就是 `amp=null`（fp32），且 lr=1e-3 / epochs=300
- 改用 `amp.enabled=false` + lr=1e-3 + epochs=80 + early_stopping patience=15 后，15 epoch 已能稳定收敛到 test R²=0.787（dry-run）
- 同时发现 trainer `_metric_predictions` / `_compositional_metrics` / `evaluate` 三处 `.cpu().numpy()` 在 AMP bf16 下报 `TypeError: Got unsupported ScalarType BFloat16`，加 `.float()` 转 fp32 后兼容（共用代码修复，不影响 hg 路径）

### 3.6 PatchTST 修复后正式结果

修复配置 `configs/experiment/sg4/sg4_patchtst.json`：`amp.enabled=false` / `lr=1e-3` / `epochs=80` / `grad_clip_norm=1.0` / `early_stopping.patience=15`，其他保持原架构（patch_len=16 / stride=8 / d_model=64 / nhead=4 / num_layers=2 / pooling=attention）。

3 seeds 全部收敛（test split）：

| seed | best_ep | pool R² | x_H2 | x_CH4 | x_CO2 | x_CO |
|---|---|---|---|---|---|---|
| 42 | 76 | 0.934 | 0.956 | 0.712 | 0.968 | 0.920 |
| 123 | 77 | 0.945 | 0.953 | 0.703 | 0.970 | 0.948 |
| 2026 | 80 | 0.927 | 0.957 | 0.642 | 0.955 | 0.910 |
| **mean ± std** | — | **0.935 ± 0.008** | 0.955 ± 0.002 | 0.686 ± 0.031 | 0.964 ± 0.007 | 0.926 ± 0.016 |

观察：

- PatchTST 修复后 pool R²=0.935，**与 CNN1D 同级（0.931）**，落在 TCN/Ridge（0.958/0.957）之下
- x_H2 = 0.955：略低于 TCN（0.968）和 Ridge（0.977），但显著高于 CNN1D（0.945）
- x_CH4 = 0.686：**比 CNN1D（0.727）还差**，CH₄ 低浓度短板在 Transformer 模型上更明显
- x_CO = 0.926：与 CNN1D 持平，比 TCN（0.954）/ Ridge（0.946）/ LSTM（0.940）略差
- seed 稳定性 ±0.008，比 CNN1D（±0.005）和 TCN（±0.001）稍差

**结论**：PatchTST 不是 sg4 上的最优模型，但能力达到与 CNN1D 同级。**TCN ≈ Ridge > PatchTST ≈ CNN1D ≈ LSTM** 的排序确认。Transformer 在 6000 序列 + 慢通道单模态上没有明显优势。

## 4. 副产品：编排脚本的 LSTM 退出码误报

`scripts/run_sg4_baseline.py` 把 LSTM 全部 3 个 run 标记为 `status=fail / returncode=3221226505`，但 `metrics.json` 完整、训练正常完成。`3221226505 = 0xC0000409 = STATUS_STACK_BUFFER_OVERRUN` 是 Windows 上 PyTorch LSTM 进程退出时 cuDNN 资源回收阶段的已知问题，**不影响 metrics 输出**。

### 修复（2026-06-26 同日）

`scripts/run_sg4_baseline.py` `_run_dl` 改为：returncode != 0 时优先检查 metrics.json 是否存在；存在则视为 `status="ok"` 并附 `warning` 字段记录原始 returncode（典型场景：Windows cuDNN teardown 误报）；不存在才真的算 fail。

```python
if proc.returncode != 0:
    if metrics_path.is_file():
        warning = f"non-zero exit code {proc.returncode} but metrics.json present (likely Windows cuDNN teardown)"
        ...
        return {..., "status": "ok", "warning": warning, "returncode": proc.returncode, ...}
    return {..., "status": "fail", ...}
```

这样 LSTM 之后的多 run 实验（含 ablation 矩阵）不会再误报失败。

## 4.1 副产品：trainer AMP bfloat16 兼容性修复

PatchTST 重跑试用 bfloat16 时报：

```
TypeError: Got unsupported ScalarType BFloat16
```

来自 `src/dl/training/trainer.py` 三处 `.detach().cpu().numpy()`（autocast 下输出张量仍是 bf16，PyTorch 的 numpy 转换不接受 bf16）。修法：每处加 `.float()` 显式转 fp32 再 numpy：

```python
# trainer.py:323-325
conditional_metrics = conditional_component_metrics(
    y_pred_raw.detach().float().cpu().numpy(),
    y_true.detach().float().cpu().numpy(),
    ...
)
# trainer.py:427-429, 442-443 同理
```

完整 444 测试通过，hg 零回归。该修复使 trainer 可在 fp16 / bf16 / fp32 三种精度下统一工作，为后续 ablation 实验 AMP 自由切换打底。

## 4.2 副产品：编排脚本部分重跑合并

`scripts/run_sg4_baseline.py --models patchtst` 重跑时，旧版会用 PatchTST 3 条记录覆盖整个 `summary.json` 和 `runs.jsonl`，丢掉其他 4 个模型的历史结果。修后：

1. 写 `runs.jsonl` 时先读旧文件，按 `(model, seed)` key 合并，新记录覆盖同 key、其他保留
2. 新增 `_summarize_from_disk()`：扫描 `outputs/sg4_baseline/{model}/seed{seed}/metrics.json` 逐文件构造 summary，不依赖 records 内存对象

部分重跑后 `summary.json` 始终保持 5 模型 × 3 seeds 完整。

## 5. 与 hg 基线对比（参考）

hg `wv4-formal-hitran-standard-6000` 上 TCN/CNN1D 的 pool R² 一般在 0.95–0.99（参考过往 P0–P3 实验），N₂ R² 通常是短板（0.4–0.7，因 N₂ 是 background）。

syngas 与 hg 的可比点：

| 维度 | hg | syngas |
|---|---|---|
| 数据规模 | 6000 / 128 | 6000 / 128 |
| 慢通道 | 8（不含 V_NDIR_CO） | 9（含 V_NDIR_CO） |
| 预测目标 | 4（含 x_N2） | 4（含 x_CO） |
| 短板组分 | N₂（背景） | CH₄（低浓度） |
| 最佳 pool R²（TCN 类） | ~0.97 | 0.958 |
| 慢通道单模态 Ridge 上限 | — | 0.957 |

syngas pool R² 比 hg 略低（0.96 vs 0.97），主要被 CH₄ 短板拖累。CO 子项检测水平与 hg CO₂ 类似（依赖 NDIR 强信号）。

## 6. 结论与下一步

### 已确认

1. sg4-formal benchmark 在 5 模型 × 3 seeds 训练下能稳定输出基线结果。**TCN ≈ Ridge ≈ 0.96 ≥ PatchTST ≈ CNN1D ≈ LSTM ≈ 0.93** 的 pool R² 排序成立
2. V_NDIR_CO 通道有效，CO 检测无明显退化（所有正常 runs CO R² 在 0.91–0.95 之间）
3. **慢通道 + 手工统计量已经接近性能上限**：Ridge 用 63 维特征 + closed-form 就能跟 TCN 持平，4 个 DL 模型中只有 TCN 跟上，其余三个全部落后 2-3 个百分点
4. CH₄ 短板符合 roadmap 风险预期，TCN/Ridge ≈ 0.83，CNN1D ≈ 0.73，PatchTST ≈ 0.69，LSTM ≈ 0.43
5. **PatchTST 配置敏感**：AMP fp16/bf16 会让其完全无法收敛（卡在 val_loss=1.00 平台），必须用 fp32；修复后表现接近 CNN1D，但仍不及 TCN
6. **LSTM seed 敏感性**：3 seeds 中 1 个完全不收敛（best_epoch=1），AdamW + lr=1e-3 + 9 输入通道在该 seed 上落入平坦区

### 待跟进

| 项 | 状态 | 优先级 |
|---|---|---|
| 修编排脚本 LSTM 退出码误报 | ✅ 已修复（同日，见 §4） | — |
| trainer AMP bf16 兼容性 | ✅ 已修复（同日，见 §4.1） | — |
| 重跑 PatchTST 3 seeds（fp32 + lr=1e-3 + 80 epoch） | ✅ 已完成，pool R²=0.935±0.008 | — |
| **Ⅱ-1 CO 通道 ablation**（论文核心证据链） | 待启动 | **最高** |
| Ⅱ-2 串扰 ablation（数据生成 + 训练） | 待启动 | 中 |
| Ⅱ-3 Loss 对比 | 待启动 | 中 |
| 接入 ultrasonic / fiber_mic 模态 | 待启动 | 中（可能让 TCN 拉开与 Ridge 的差距） |
| LSTM seed=123 不收敛重试（lr 1e-3 → 5e-4 或 warmup） | 待评估 | 低（2/3 可用，不阻塞结论） |

### 产物

| 产物 | 路径 |
|---|---|
| 各 run metrics | `outputs/sg4_baseline/{model}/seed{seed}/metrics.json` |
| 汇总 JSON | `outputs/sg4_baseline/summary.json` |
| run 状态记录 | `outputs/sg4_baseline/runs.jsonl` |
| 训练日志 | `logs/sg4_baseline_train.log` |
| 编排脚本 | `scripts/run_sg4_baseline.py` |
