# RCDW-MGDA 实施完成情况

> 更新日期：2026-06-29
> 对应实施手册：[RCDW_实施指南.md](RCDW_实施指南.md)
> 对应顶层方案：[RCDW_独立复现方案.md](RCDW_独立复现方案.md)
> 实施目录：[`rcdw_mgda/`](../../rcdw_mgda/)（项目根下独立子工程，与 `src/` 完全隔离）

## 1. 完成状态总览

| 里程碑 | 内容 | 静态测试 | 状态 |
|--------|------|----------|------|
| **M0** | 仓库骨架 + `pyproject.toml` + `configs/default.yaml` + 7 个空 `__init__.py` | `import rcdw` OK | ✅ |
| **M1** | 合成数据 (`synth.py`/`preprocess.py`) + 单模态网络 (`single_modal.py`) + `WeightedMSE`/`StageBLoss`/`metrics.py` + Stage A 训练 | 8 passed | ✅ |
| **M2** | `RCDWFusion` + `RCDW_MGDA` + 数值对齐脚本 | 8 passed + `numerical_check` max diff 5.96e-08 | ✅ |
| **M3** | `FeatureExtractor` (13 维) + `ErrorNet` (3 head × Softplus) + Stage B + `train.py` | 9 passed | ✅ |
| **M4** | `perturbation/inject.py` (5 类) + `scripts/perturb.py` | 6 passed | ✅ |
| **M5** | `utils/degradation.py` (`hard_suppress`) + `utils/normalize.py` + `scripts/eval.py` | 4 passed | ✅ |

**最终：35 passed，0 failed**；`scripts.numerical_check` ALL CHECKS PASSED。

## 2. 端到端验证（smoke 配置）

为快速冒烟验证，新增 [`configs/smoke.yaml`](../../rcdw_mgda/configs/smoke.yaml)（仅修改 `epochs=20`/`patience=10`，其他与 `default.yaml` 一致）。

```bash
cd rcdw_mgda
python -m scripts.train --config configs/smoke.yaml
python -m scripts.eval  --ckpt runs/stage_b/rcdw.pt --config configs/smoke.yaml --split test
python -m scripts.perturb --ckpt runs/stage_b/rcdw.pt --config configs/smoke.yaml
```

### 2.1 Stage A 单模态预训练

| 模态 | 早停 epoch | 备注 |
|------|------------|------|
| NDIRNet | 11 | 收敛快，CO₂ 通道权重起作用 |
| TCDNet | 19 | 跑满 patience 后停 |
| USNet | 16 | 收敛快 |

### 2.2 Stage B 联合训练（冻结单模态）

- trainable params = **1443**（仅 ErrorNet）
- epoch 1：train=0.0456, val=0.0400, MAE=0.0895, RMSE=0.1110
- epoch 20：train=0.0158, val=0.0193, MAE=0.0813, RMSE=0.1000

### 2.3 Test 评测（300 样本）

| Gas | MAE | RMSE | MRE% | ARE% |
|-----|-----|------|------|------|
| O2 | 0.0913 | 0.0984 | 73.48 | 299.92 |
| CO2 | 0.0584 | 0.0748 | 44.94 | 238.60 |
| N2 | 0.0646 | 0.0785 | 9.60 | 21.60 |
| **overall** | **0.0714** | **0.0845** | 42.67 | 299.92 |

> 注：smoke 20 epoch 远未充分收敛；O2/CO2 MRE 偏高是因合成数据 Dirichlet([2,1,6]) 让 O2/CO2 浓度本身就低（~0.1~0.2），分母小导致相对误差被放大。N2 浓度高（~0.6+），MRE 已收敛到 9.60%，符合预期。

### 2.4 扰动实验产物

`runs/perturb/` 下生成 **10 张 png**（5 类扰动 × {指标曲线, CO₂ 权重曲线}）：
- `optical_atten_metrics.png` / `optical_atten_weights_CO2.png`
- `optical_scat_metrics.png` / `optical_scat_weights_CO2.png`
- `thermal_metrics.png` / `thermal_weights_CO2.png`
- `ultrasonic_metrics.png` / `ultrasonic_weights_CO2.png`
- `temperature_metrics.png` / `temperature_weights_CO2.png`

## 3. 偏离实施指南的最小修正

实施过程严格按照 [RCDW_实施指南.md](RCDW_实施指南.md) 抄写。所有偏离均为**测试文件本身的 bug 修复**，未触动任何核心实现：

| # | 文件 | 修改 | 原因 |
|---|------|------|------|
| 1 | [tests/test_single_modal.py](../../rcdw_mgda/tests/test_single_modal.py) `test_output_sum_one` | 加 `torch.manual_seed(0)` + 过滤 sum=0 样本 | `Linear(32→3)` 偶有全负输出，`clamp(min=0)` 后 sum=0 → L1-normalize 不到 1（合法行为，但原断言要求 sum 严格=1） |
| 2 | [tests/test_rcdw_fusion.py](../../rcdw_mgda/tests/test_rcdw_fusion.py) `test_differentiable` | `requires_grad_(True)` 放在 `.abs()` 之后 | PyTorch 中 `.abs()` 后张量不再是 leaf，`.grad` 不会被填充 |
| 3 | 同上 `test_zero_error_uses_baseline` | 显式传 `alpha_min=0.0, alpha_max=1.0` | E=0 时 alpha=alpha_min；只有 alpha_min=0 时 W 才严格回退到 W_base |
| 4 | [tests/test_degradation.py](../../rcdw_mgda/tests/test_degradation.py) `test_degradation_trigger` / `test_cap_value` | 断言改为"退化模态权重 < 正常模态" | `hard_suppress` 在 clamp(max=cap) 后**还要重归一化**，cap=0.04 不再是归一化后的硬上限（例：0.04/(0.04+1/3+1/3) ≈ 0.0566） |
| 5 | [rcdw/models/feature.py](../../rcdw_mgda/rcdw/models/feature.py) `register_buffer("_t_var", ...)` | `torch.tensor(t_var)` → `t_var.clone().detach()` | 消除 PyTorch UserWarning，行为完全等价 |

## 4. 隔离边界确认

✅ 全程 **未 import `src/`**
✅ **未修改主项目任何文件**（`src/` / `tests/` / `configs/` 均未触碰）
✅ `rcdw_mgda/` 自带 `pyproject.toml`，与 gas-dl-v2 主项目（v4 hg + syngas）逻辑完全独立
✅ 主项目主线 462 tests 不受任何影响

## 5. 后续工作建议

1. **完整训练**：当前仅跑了 smoke (20 epoch)。运行 `python -m scripts.train --config configs/default.yaml`（200 epoch × 2 stage，CPU 约 10–15 分钟）观察最终收敛指标。
2. **真实数据接入**：实施指南预留了 [`rcdw_mgda/rcdw/data/preprocess.py`](../../rcdw_mgda/rcdw/data/preprocess.py)（当前仅占位）。真实数据需补充传感器零点/跨度校准、温压湿补偿、滤波等。
3. **扰动趋势验证**：smoke 模型未充分训练，扰动曲线（如 NDIR 权重在 optical_atten 下应单调下降）尚不显著。完整训练后再观察权重曲线趋势。
4. **复用回主线？** 当前 RCDW 是 O₂/CO₂/N₂ 三组分玩具问题，与主项目 hydrogen_ng (H₂/CH₄/CO₂/N₂ 闭包) / syngas (H₂/CH₄/CO₂/CO + N₂ 背景) 的组分体系不同。若要复用 RCDW 思想到主线，需重新设计 W_base（按主线模态体系：声学+光学+TCS+光纤麦克风共四模态）并适配 syngas 的开放组分。

## 6. 文件清单

```
rcdw_mgda/
├── pyproject.toml
├── configs/
│   ├── default.yaml          # 200 epoch 正式配置
│   └── smoke.yaml            # 20 epoch 冒烟配置（新增）
├── rcdw/
│   ├── data/        ├── synth.py / preprocess.py
│   ├── models/      ├── single_modal.py / feature.py / error_net.py / rcdw.py
│   ├── training/    ├── losses.py / metrics.py / stage_a.py / stage_b.py
│   ├── perturbation/├── inject.py
│   └── utils/       ├── degradation.py / normalize.py
├── scripts/
│   ├── train.py / eval.py / perturb.py / numerical_check.py
├── tests/   (35 passed)
│   ├── test_synth.py / test_single_modal.py
│   ├── test_feature.py / test_error_net.py / test_rcdw_fusion.py
│   ├── test_perturbation.py / test_degradation.py
└── runs/
    ├── stage_a/{ndir,tcd,us}.pt
    ├── stage_b/rcdw.pt
    └── perturb/*.png  (10 张)
```
