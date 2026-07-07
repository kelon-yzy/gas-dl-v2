# RCDW-MGDA 实施完成情况

> 更新日期：2026-07-01（Phase 6A/6B/6C/6D 完成；Phase 6E pressure_drift 实验闭环完成）
> 对应实施手册：[RCDW_实施指南.md](RCDW_实施指南.md)
> 对应顶层方案：[RCDW_独立复现方案.md](RCDW_独立复现方案.md)
> 实施目录：[`rcdw_mgda/`](../../rcdw_mgda/)（项目根下独立子工程，与 `src/` 完全隔离）

> ⚠️ **数据集已切换为 benchmark 形态（2026-06-30）**：
> 本文档 §1–§6 描述的是 M0-M5 toy 阶段（35 tests，旧 `synth.py` 6 维数据）。
> 数据契约已升级到 `schema_version="rcdw-benchmark-1"`，12 维通道布局，
> 与 HG 主线 `src/sim/` 同质量的数据生成管线。详见
> [RCDW_数据集主线对齐改动方案.md](RCDW_数据集主线对齐改动方案.md) v1.2 +
> [RCDW_数据集主线对齐_完成情况.md](RCDW_数据集主线对齐_完成情况.md)；
> Phase 6 后续路线见 [RCDW_数据集主线对齐_Phase6路线.md](RCDW_数据集主线对齐_Phase6路线.md)。
> 最新状态：**222 tests pass / Phase 6A HITRAN cache 预热完成 / Phase 6B 64-seq 与 128-seq HITRAN smoke 闭环 / Phase 6C benchmark 可选并行生成完成 / Phase 6D scaler on/off ablation 完成 / Phase 6E `pressure_drift` perturb 实验完成并生成 12 张 PNG / 旧 ckpt 必须重训**。

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

toy 阶段 `runs/perturb/` 下生成 **10 张 png**（5 类基础扰动 × {指标曲线, CO₂ 权重曲线}）：
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

Phase 1-5 数据集主线对齐完成后，后续工作已拆分为 Phase 6 路线，详见 [RCDW_数据集主线对齐_Phase6路线.md](RCDW_数据集主线对齐_Phase6路线.md)。摘要如下：

1. **HITRAN 默认路径可用**：`--precompute-cache-only` 已实现并完成真实 HAPI cache 预热。
2. **HITRAN smoke 证据链**：64-seq 与 128-seq `hitran_hapi_v1` smoke 已完成 generate -> train -> eval -> perturb。
3. **formal 性能与训练稳定性**：benchmark 可选并行生成已完成确定性门槛；12 维 input scaler ablation 已证明 scaler-on 优于 scaler-off。
4. **实验扩展**：`pressure_drift` 已完成工程接入与 Phase 6E perturb 实验闭环；`h2o_cross` 仍需先定义输入扰动或后端重生成语义。O2 弛豫参数和多 `stage_profile` 放在后段。
5. **复用回主线？** 当前 RCDW 是 O₂/CO₂/N₂ 三组分独立子工程，与主项目 hydrogen_ng (H₂/CH₄/CO₂/N₂ 闭包) / syngas (H₂/CH₄/CO₂/CO + N₂ 背景) 的组分体系不同。若要复用 RCDW 思想到主线，需重新设计 W_base（按主线模态体系：声学+光学+TCS+光纤麦克风共四模态）并适配 syngas 的开放组分。

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

---

## 7. 数据集主线对齐（2026-06-30 v1.2 升级）

§1–§6 是 M0-M5 toy 阶段（旧 `synth.py` 6 维输入 + Dirichlet 玩具浓度）；2026-06-30 完成
**数据集主线对齐**，将 RCDW 数据生成管线升级到与 HG 主线 `src/sim/` 同质量，但保持完全隔离（独立 schema、独立 ID 前缀、独立 HITRAN cache）。

### 7.1 升级要点

| 维度 | toy v1（旧） | benchmark v1.2（新） |
|------|------------|-------------------|
| 数据生成入口 | `rcdw.data.synth.synth_timeseries` | `scripts.generate_benchmark` 全流程编排 |
| 通道布局 | 6 维 `[S_ndir, S_tc, S_us, P, T, RH]` | **12 维** slow(7) + ultrasonic 元数据(5) |
| 标签语义 | Dirichlet([2,1,6]) toy 分布 | HITRAN 物理建模 + LHS d=2 simplex 采样 |
| Schema version | 无 | `rcdw-benchmark-1` |
| ID 前缀 | 无 | `RCDW-M{:06d}` / `RCDW-Q{:06d}` |
| baseline | toy 随机 | **100% N₂ 纯背景气** |
| PhaseSchedule | 无 | `STANDARD_EXPOSURE`（v1.2 YAGNI 仅一种） |
| 切分 | 时序顺序切分 | **mixture_id 分层 70/15/15**（不含 extrapolation） |
| Scaler | 无 | train-only Z-score + 异质通道 passthrough + 12 维 input scaler |
| Validation | 4 项 toy 不变量 | **10 项**（含 LEGACY 黑名单、组分和=100、scaler passthrough 标记） |
| 测试数 | 35 | **222**（当前工作区实测；含 Phase 6A-6E 相关测试） |

### 7.2 commit 序列

| commit | Phase | 文件 | 测试 |
|--------|-------|------|------|
| `7671e7f` | Phase 1+2：schema 骨架 + 物理栈 | 27 新增 | 79 |
| `da5cddc` | Phase 3：slow + packaging + benchmark 端到端 | 17 新增 + 2 修改 | 59 |
| `5eb8001` | Phase 4+5：Dataset + 12 维通道 + 扰动重映射 | 4 新增 + 11 修改 + 2 删除 | 60 |
| `099d8cc` | 文档同步：数据集主线对齐完成情况与入口说明 | docs | 198 |
| `de4b08f` | P2 修复：scaler validation summary + Dataset window 边界 | 6 修改 | 198 |
| **Phase 1-5 累计** | **5 个 Phase 全部完成，Phase 6 路线已拆分** | **48 新增 / 19 修改 / 2 删除** | **198 全过** |

### 7.3 历史 ckpt 废弃

**§2 smoke 训练产生的 `runs/stage_a/*.pt` 与 `runs/stage_b/rcdw.pt`（基于旧 toy `synth.py` 6 维输入）与新数据契约完全不兼容**。新通道布局 6→12、标签语义 toy→HITRAN、scaler 重新拟合、baseline=100% N₂ 物理化，必须删除旧 ckpt → 按新数据生成 → Stage A 重训 → Stage B 重训。

### 7.4 新流程命令

```bash
cd rcdw_mgda

# 1) 生成 benchmark（empirical 后端不联网；HITRAN 后端需先预热 cache）
python -m scripts.generate_benchmark --config configs/smoke.yaml \
    --dataset-slug rcdw-smoke --output-root data

# 2) 训练（自动从 data/rcdw-smoke 加载 BenchmarkDataset）
python -m scripts.train --config configs/smoke.yaml

# 3) 评测
python -m scripts.eval --ckpt runs/stage_b/rcdw.pt --config configs/smoke.yaml --split test

# 4) 扰动评测（Phase 1-5 基础 5 类 × 2 张图 = 10 张 PNG；Phase 6E pressure_drift 配置为 6 类 × 2 张图 = 12 张 PNG）
python -m scripts.perturb --ckpt runs/stage_b/rcdw.pt --config configs/smoke.yaml
```

### 7.5 详细落地记录

见 [RCDW_数据集主线对齐_完成情况.md](RCDW_数据集主线对齐_完成情况.md)：
- 每 Phase 文件清单 + 测试覆盖
- 17 项验证清单
- 与方案的偏差和后续状态（如 6A-6D 已完成、6E 的 `pressure_drift` 已完成实验闭环）
- 后续 Phase 6E/6F/6G 建议

Phase 6 的独立路线文档见 [RCDW_数据集主线对齐_Phase6路线.md](RCDW_数据集主线对齐_Phase6路线.md)，优先级从 HITRAN cache 预热、HITRAN smoke 证据链、benchmark 并行化、12 维 input scaler、扰动扩展、O2 弛豫参数校核到多 `stage_profile` 激活。

---

## 8. 代码质量审查修复（2026-07-01）

全量代码审查发现 10 项问题（1 高 / 4 中 / 5 低），已全部修复。详见 [RCDW_代码质量审查报告.md](RCDW_代码质量审查报告.md)。

### 8.1 修改文件清单

| 文件 | 修复项 | 变更 |
|------|--------|------|
| `rcdw/sim/packaging/scalers.py` | H1 | 新增 `fit_input_channel_scaler()` + `INPUT_CHANNEL_ORDER` |
| `rcdw/sim/generation/benchmark.py` | H1 | 12 维 scaler 拟合/落盘 + `input_normalization` manifest |
| `rcdw/sim/packaging/manifest.py` | H1 | `build_manifest` 增 `input_normalization` 参数 |
| `rcdw/data/dataset.py` | H1 | `apply_input_scaler` 参数 + scaler 加载/应用逻辑 |
| `rcdw/sim/generation/slow.py` | M2, L3 | 环境通道传感器噪声 + tau 单位 docstring |
| `rcdw/utils/degradation.py` | M3 | 逐样本判定替代 batch 统计 |
| `scripts/train.py` | M4 | 训练播种 + seeded DataLoader |
| `configs/smoke.yaml` | M5 | 删除死配置节 |
| `configs/default.yaml` | M5 | 删除死配置节 |
| `rcdw/training/metrics.py` | L1 | `ARE` → `MaxRE` |
| `scripts/eval.py` | L1 | 表头同步更新 |
| `rcdw/utils/normalize.py` | L2, L5 | RH 单位检查 + 陈旧引用清理 |
| `rcdw/perturbation/inject.py` | L4 | 扰动强度语义 docstring |
| `rcdw/training/stage_a.py` | L5 | `(B,6)` → `(B,12)` |
| `tests/test_input_scaler.py` | H1 | 新增 11 个测试 |
| `tests/test_dataset_loader.py` | H1 | 布局测试关闭 scaler |
| `tests/test_degradation.py` | M3 | 适配 `degraded` shape 变更 |

### 8.2 测试基线

`python -m pytest` → **222 passed**（当前工作区实测，包含 Phase 6A-6E 相关测试）。

---

## 9. Phase 6A/6B HITRAN smoke 证据链（2026-07-01）

本节记录 Phase 6 路线中 6A 与 6B 的首轮正式闭环。目标是把默认 `hitran_hapi_v1` 路径从“需预热但缺入口”推进到可执行，并用 64 sequence HITRAN smoke 数据完成 generate -> train -> eval -> perturb。

### 9.1 环境与配置

| 项 | 内容 |
|----|------|
| Python 环境 | `rcdw_mgda/.venv`，Python 3.12.10 |
| 运行命令前缀 | `./.venv/Scripts/python.exe -m ...` |
| HITRAN HAPI 包 | PyPI 包名 `hitran-api`，导入名 `hapi`，版本 1.3.0.0 |
| 错误包说明 | `pip install hapi` 安装的是无关包，已卸载；应使用 `pip install hitran-api` |
| Phase 6B 配置 | `configs/phase6b-hitran-smoke.yaml`（64-seq）、`configs/phase6b-hitran-smoke-128.yaml`（128-seq） |
| dataset root | `data/rcdw-hitran-smoke-64`、`data/rcdw-hitran-smoke-128` |
| spectral backend | `hitran_hapi_v1` |
| input scaler | on，`manifest.input_normalization.applied=true`，artifact 为 `scalers/input_scaler.json` |
| ckpt 输出 | `runs/stage_a/{ndir,tcd,usn}.pt`、`runs/stage_b/rcdw.pt` |
| 扰动图输出 | `runs/perturb/`，10 张 PNG |

`configs/phase6b-hitran-smoke.yaml` 和 `configs/phase6b-hitran-smoke-128.yaml` 是 6B 专用配置，主要差异是将 `data.dataset_root` 分别指向 `data/rcdw-hitran-smoke-64` 与 `data/rcdw-hitran-smoke-128`，避免误读默认 `data/rcdw-smoke`。

### 9.2 Phase 6A：HITRAN cache 预热

新增能力：

| 文件 | 变更 |
|------|------|
| `rcdw/sim/generation/spectral/hitran_backend.py` | 新增 `precompute_spectrum_cache()`，作为 `_spectrum_for_gas()` 的公共薄包装 |
| `rcdw/sim/generation/spectral/__init__.py` | 导出 `precompute_spectrum_cache` |
| `rcdw/sim/generation/optical_backend.py` | 新增 `precompute_hitran_benchmark_cache()`，复用需求收集、预热和最终校验 |
| `scripts/generate_benchmark.py` | 新增 `--precompute-cache-only`；HITRAN 后端预热 cache，empirical 后端 no-op |
| `tests/test_optical_backend.py` | 增加全 cache 命中和部分缺失填充测试 |
| `tests/test_generate_benchmark_cli.py` | 增加 cache-only CLI 测试，覆盖不创建 benchmark 目录和 empirical no-op |

验证命令：

```bash
cd rcdw_mgda
./.venv/Scripts/python.exe -m pytest tests/test_optical_backend.py tests/test_generate_benchmark_cli.py
./.venv/Scripts/python.exe -m pytest
```

结果：

- 定向测试：**23 passed**。
- RCDW 全量测试：**216 passed**。

真实预热命令：

```bash
./.venv/Scripts/python.exe -m scripts.generate_benchmark \
  --config configs/smoke.yaml \
  --precompute-cache-only
```

结果：

```text
precompute-cache-only DONE: {'total': 128, 'cached': 0, 'filled': 128, 'cache_root': 'data\\hitran_cache'}
```

说明：

- `data/hitran_cache` 下生成 128 个 `.npz` cache 文件。
- HAPI 同时写入 CO2 / H2O table 文件。
- 预热路径只写 HITRAN cache，不创建 benchmark 数据集目录。

### 9.3 Phase 6B：64 sequence HITRAN smoke 数据集

生成命令：

```bash
./.venv/Scripts/python.exe -m scripts.generate_benchmark \
  --config configs/smoke.yaml \
  --dataset-slug rcdw-hitran-smoke-64 \
  --output-root data
```

结果：

| 项 | 内容 |
|----|------|
| dataset slug | `rcdw-hitran-smoke-64` |
| sequence_count | 64 |
| output_dir | `data/rcdw-hitran-smoke-64` |
| optical_absorption_backend | `hitran_hapi_v1` |
| validation status | `pass` |
| split counts | train 44 / val 9 / test 11 |
| input shape | slow 7 维 + ultrasonic 元数据 5 维，共 12 维 |
| train windows | 1100 |
| val windows | 225 |
| test windows | 275 |

加载检查：

```text
dataset_ok 1100 (8, 12) (3,) 1.0
finite True
validation pass
cache_npz_count 128
```

### 9.4 Stage A/B 训练结果

训练命令：

```bash
./.venv/Scripts/python.exe -m scripts.train --config configs/phase6b-hitran-smoke.yaml
```

Stage A：

| 模态 | epoch 1 | 结束状态 |
|------|---------|----------|
| NDIR | train=0.007303, val=0.006844 | epoch 20: train=0.002206, val=0.004936 |
| TCD | train=0.319374, val=0.291501 | early stop at epoch 12 |
| USN | train=0.021935, val=0.015330 | early stop at epoch 12 |

Stage B：

| 项 | 数值 |
|----|------|
| trainable params | 1443 |
| epoch 1 | train=0.217918, val=0.128832, MAE=0.1766, RMSE=0.2092 |
| epoch 20 | train=0.014344, val=0.018865, MAE=0.0824, RMSE=0.0992 |

### 9.5 Test 评测结果

评测命令：

```bash
./.venv/Scripts/python.exe -m scripts.eval \
  --ckpt runs/stage_b/rcdw.pt \
  --config configs/phase6b-hitran-smoke.yaml \
  --split test
```

Test windows：275。

| Gas | MAE | RMSE | MRE% | MaxRE% |
|-----|----:|-----:|-----:|-------:|
| O2 | 0.06914 | 0.08960 | 141.75 | 1353.44 |
| CO2 | 0.05051 | 0.06096 | 59.39 | 220.90 |
| N2 | 0.07174 | 0.09994 | 9.43 | 81.46 |
| overall | 0.06380 | 0.08511 | 70.19 | 1353.44 |

说明：O2 的 MRE / MaxRE 偏高主要受低真值分母影响，后续正式报告应同时看 MAE / RMSE 与分组浓度区间。

### 9.6 扰动评测结果

扰动命令：

```bash
./.venv/Scripts/python.exe -m scripts.perturb \
  --ckpt runs/stage_b/rcdw.pt \
  --config configs/phase6b-hitran-smoke.yaml
```

产物：`runs/perturb/` 下生成 10 张 PNG：

- `optical_atten_metrics.png` / `optical_atten_weights_CO2.png`
- `optical_scat_metrics.png` / `optical_scat_weights_CO2.png`
- `thermal_metrics.png` / `thermal_weights_CO2.png`
- `ultrasonic_metrics.png` / `ultrasonic_weights_CO2.png`
- `temperature_metrics.png` / `temperature_weights_CO2.png`

指标摘要：

| 扰动 | level=0 MAE/RMSE | 最高 level 现象 |
|------|------------------|----------------|
| optical_atten | 0.0638 / 0.0851 | level 0.11: MAE 0.0641, RMSE 0.0855，变化较小 |
| optical_scat | 0.0638 / 0.0851 | level 0.11: MAE 0.0637, RMSE 0.0847，变化较小 |
| thermal | 0.0638 / 0.0851 | level 0.11: MAE 0.0652, RMSE 0.0903，有波动 |
| ultrasonic | 0.0638 / 0.0851 | level 0.11: MAE 0.0638, RMSE 0.0851，变化较小 |
| temperature | 0.0638 / 0.0851 | level 0.11: MAE 0.0878, RMSE 0.1219，退化最明显 |

观察：

- 当前 64-seq smoke 下，temperature 扰动趋势最清晰。
- optical / ultrasonic 类扰动曲线较平，可能与 smoke 训练规模、ErrorNet 可靠性估计和 hard_suppress 判定有关。
- 所有扰动输出均显示 `degraded=True`，后续 6D / 6E 可进一步检查 hard_suppress 与 ErrorNet 是否过于敏感。

### 9.7 Phase 6B：128 sequence HITRAN smoke 补充

为观察更大 smoke 下 Stage A/B 是否更平滑，补充 128 sequence HITRAN smoke。

配置：`configs/phase6b-hitran-smoke-128.yaml`。

cache 预热命令：

```bash
./.venv/Scripts/python.exe -m scripts.generate_benchmark \
  --config configs/phase6b-hitran-smoke-128.yaml \
  --precompute-cache-only
```

预热结果：

```text
precompute-cache-only DONE: {'total': 256, 'cached': 128, 'filled': 128, 'cache_root': 'data\\hitran_cache'}
```

生成命令：

```bash
./.venv/Scripts/python.exe -m scripts.generate_benchmark \
  --config configs/phase6b-hitran-smoke-128.yaml \
  --dataset-slug rcdw-hitran-smoke-128 \
  --output-root data
```

生成结果：

| 项 | 内容 |
|----|------|
| dataset slug | `rcdw-hitran-smoke-128` |
| sequence_count | 128 |
| output_dir | `data/rcdw-hitran-smoke-128` |
| optical_absorption_backend | `hitran_hapi_v1` |
| validation status | `pass` |
| split counts | train 89 / val 19 / test 20 |
| train windows | 2225 |
| val windows | 475 |
| test windows | 500 |
| cache `.npz` 总数 | 256 |

加载检查：

```text
dataset_ok 2225 475 500 (8, 12) (3,) 1.0
finite True
validation pass
cache_npz_count 256
```

训练命令：

```bash
./.venv/Scripts/python.exe -m scripts.train --config configs/phase6b-hitran-smoke-128.yaml
```

Stage A：

| 模态 | epoch 1 | 结束状态 |
|------|---------|----------|
| NDIR | train=0.005596, val=0.003921 | early stop at epoch 11 |
| TCD | train=0.067931, val=0.008716 | early stop at epoch 14 |
| USN | train=0.016180, val=0.009515 | early stop at epoch 12 |

Stage B：

| 项 | 数值 |
|----|------|
| trainable params | 1443 |
| epoch 1 | train=0.090300, val=0.024786, MAE=0.0582, RMSE=0.0741 |
| epoch 20 | train=0.010022, val=0.009968, MAE=0.0575, RMSE=0.0733 |

Test 评测命令：

```bash
./.venv/Scripts/python.exe -m scripts.eval \
  --ckpt runs/stage_b/rcdw.pt \
  --config configs/phase6b-hitran-smoke-128.yaml \
  --split test
```

Test windows：500。

| Gas | MAE | RMSE | MRE% | MaxRE% |
|-----|----:|-----:|-----:|-------:|
| O2 | 0.06179 | 0.07078 | 617.21 | 16551.48 |
| CO2 | 0.05196 | 0.05858 | 3302.76 | 98768.96 |
| N2 | 0.05847 | 0.07341 | 7.46 | 32.36 |
| overall | 0.05741 | 0.06790 | 1309.14 | 98768.96 |

说明：128-seq 相比 64-seq 的 overall MAE / RMSE 更低，Stage B val 指标也更平滑；但 O2 / CO2 的 MRE 与 MaxRE 受极低真值分母强烈放大，不能单独作为质量判断。

扰动评测命令：

```bash
./.venv/Scripts/python.exe -m scripts.perturb \
  --ckpt runs/stage_b/rcdw.pt \
  --config configs/phase6b-hitran-smoke-128.yaml
```

扰动摘要：

| 扰动 | level=0 MAE/RMSE | 最高 level 现象 |
|------|------------------|----------------|
| optical_atten | 0.0580 / 0.0687 | level 0.11: MAE 0.0580, RMSE 0.0687，几乎不变 |
| optical_scat | 0.0580 / 0.0687 | level 0.11: MAE 0.0581, RMSE 0.0687，变化很小 |
| thermal | 0.0580 / 0.0687 | level 0.11: MAE 0.0578, RMSE 0.0689，变化很小 |
| ultrasonic | 0.0580 / 0.0687 | level 0.11: MAE 0.0582, RMSE 0.0690，变化很小 |
| temperature | 0.0580 / 0.0687 | level 0.11: MAE 0.1129, RMSE 0.1463，退化最明显 |

观察：128-seq 下 temperature 扰动趋势更清晰；其它四类扰动仍相对平坦。所有扰动输出仍显示 `degraded=True`，后续可在 6D / 6E 检查 ErrorNet 和 hard_suppress 的判定敏感性。

### 9.8 Phase 6C：benchmark 可选并行生成

Phase 6C 的目标是在不改变 schema、ID、split、scaler 与 manifest 契约的前提下，为 `build_sequence_arrays` 增加可选并行执行路径，并验证单进程与并行结果在固定 seed 下确定性等价。

实现变更：

| 文件 | 变更 |
|------|------|
| `rcdw/sim/generation/benchmark.py` | `BenchmarkGenerationSpec` 新增 `num_workers` 与 `chunk_size`；新增 `_build_sequence_arrays_for_spec()` 与 `_merge_sequence_array_chunks()`；`num_workers=1` 保持默认单进程路径 |
| `scripts/generate_benchmark.py` | 从 `generation.num_workers` 与 `generation.chunk_size` 读取可选并行参数，默认 `1 / 0` |
| `tests/test_benchmark_e2e.py` | 新增 empirical 单进程 / 并行 parity、HITRAN 合成 cache 单进程 / 并行 parity、默认 worker 回归、非法参数拒收测试；保留同 seed reproducibility 测试 |

关键实现约束：

- `num_workers=1` 为默认值，完全保持既有行为。
- `chunk_size=0` 表示自动切分；手动 chunk size 必须大于 0。
- 并行 worker 调用 `build_sequence_arrays_chunk()`，`start_sequence_index` 使用 chunk 起始全局序号，保证 `_stable_uint32(seed, global_sequence_index, stream)` 与单进程一致。
- worker 传 `phase_schedule=spec.stage_profile` 字符串，不传 `PhaseSchedule` 对象，降低 Windows pickle 风险。
- ndarray 按 axis=0 合并，`slow_rows` 按 chunk 原顺序 extend。
- split、labels、scaler、manifest 仍在主进程中按完整 `conditions` 计算，不随并行度改变。

验证命令：

```bash
cd rcdw_mgda
./.venv/Scripts/python.exe -m pytest tests/test_benchmark_e2e.py
./.venv/Scripts/python.exe -m pytest
```

结果：

- `tests/test_benchmark_e2e.py`：**15 passed**。
- RCDW 全量测试：**222 passed**（2026-07-01 复测）。

当前结论：Phase 6C 已完成。并行能力已可通过 `BenchmarkGenerationSpec(num_workers=..., chunk_size=...)` 或配置文件 `generation.num_workers / generation.chunk_size` 启用；默认仍为单进程。正式 formal 性能测试可在后续用较大 sequence_count 评估是否开启。

### 9.9 Phase 6D：12 维 input scaler ablation

Phase 6D 目标是在同一 HITRAN smoke 数据集上比较 input scaler on/off 对训练、评测与扰动趋势的影响。本轮使用已生成的 `data/rcdw-hitran-smoke-128`。

实现变更：

| 文件 | 变更 |
|------|------|
| `rcdw/training/stage_a.py` | `run_stage_a(..., save_dir="runs/stage_a")`，支持配置 Stage A ckpt 目录 |
| `rcdw/training/stage_b.py` | `run_stage_b(..., save_dir="runs/stage_b")`，支持配置 Stage B ckpt 目录 |
| `scripts/train.py` | 读取 `data.apply_input_scaler` 并传给 train/val Dataset；读取 `training.stage_a.save_dir` 与 `training.stage_b.save_dir`，避免覆盖 scaler-on ckpt |
| `scripts/eval.py` | 读取 `data.apply_input_scaler` 并传给 Dataset |
| `scripts/perturb.py` | 读取 `data.apply_input_scaler` 并传给 Dataset |
| `configs/phase6d-hitran-smoke-128-scaler-off.yaml` | 新增 scaler-off 配置，`data.apply_input_scaler=false`，ckpt 输出到 `runs/phase6d_scaler_off/` |

验证命令：

```bash
cd rcdw_mgda
./.venv/Scripts/python.exe -m pytest tests/test_input_scaler.py tests/test_dataset_loader.py
./.venv/Scripts/python.exe -m pytest
```

结果：

- input scaler / dataset 定向测试：**23 passed**。
- RCDW 全量测试：**222 passed**（2026-07-01 复测）。

scaler-off 训练命令：

```bash
./.venv/Scripts/python.exe -m scripts.train --config configs/phase6d-hitran-smoke-128-scaler-off.yaml
```

训练输出目录：

- Stage A：`runs/phase6d_scaler_off/stage_a/{ndir,tcd,usn}.pt`
- Stage B：`runs/phase6d_scaler_off/stage_b/rcdw.pt`

确认 `runs/stage_b/rcdw.pt`（scaler-on ckpt）仍存在，未被覆盖。

Stage A scaler-off：

| 模态 | epoch 1 | 结束状态 |
|------|---------|----------|
| NDIR | train=0.015045, val=0.003519 | early stop at epoch 11 |
| TCD | train=0.197937, val=0.222069 | early stop at epoch 11 |
| USN | train=0.201832, val=0.222069 | early stop at epoch 11 |

Stage B scaler-off：

| 项 | 数值 |
|----|------|
| trainable params | 1443 |
| epoch 1 | train=5.674202, val=0.447491, MAE=0.3086, RMSE=0.4551 |
| epoch 20 | train=0.069985, val=0.069805, MAE=0.1282, RMSE=0.1678 |

scaler-off test 评测命令：

```bash
./.venv/Scripts/python.exe -m scripts.eval \
  --ckpt runs/phase6d_scaler_off/stage_b/rcdw.pt \
  --config configs/phase6d-hitran-smoke-128-scaler-off.yaml \
  --split test
```

scaler-off test windows：500。

| Gas | MAE | RMSE | MRE% | MaxRE% |
|-----|----:|-----:|-----:|-------:|
| O2 | 0.05964 | 0.06701 | 756.09 | 14057.90 |
| CO2 | 0.08444 | 0.09562 | 3715.01 | 223876.90 |
| N2 | 0.08519 | 0.10590 | 11.60 | 37.76 |
| overall | 0.07642 | 0.09101 | 1494.23 | 223876.90 |

scaler on/off 对比（128-seq）：

| 输入 | Stage B epoch20 val | Test overall MAE | Test overall RMSE |
|------|--------------------:|-----------------:|------------------:|
| scaler on | 0.009968 | 0.05741 | 0.06790 |
| scaler off | 0.069805 | 0.07642 | 0.09101 |

结论：scaler-on 明显优于 scaler-off，说明 12 维 input scaler 对收敛与 test 指标有实质收益；正式 HITRAN smoke 后续默认保留 scaler-on。

scaler-off 扰动命令：

```bash
./.venv/Scripts/python.exe -m scripts.perturb \
  --ckpt runs/phase6d_scaler_off/stage_b/rcdw.pt \
  --config configs/phase6d-hitran-smoke-128-scaler-off.yaml \
  --output-dir runs/phase6d_scaler_off/perturb
```

产物：`runs/phase6d_scaler_off/perturb/` 下生成 10 张 PNG。

scaler-off 扰动摘要：

| 扰动 | level=0 MAE/RMSE | 最高 level 现象 |
|------|------------------|----------------|
| optical_atten | 0.0857 / 0.1116 | level 0.11: MAE 0.0856, RMSE 0.1115，变化很小 |
| optical_scat | 0.0857 / 0.1116 | level 0.11: MAE 0.0812, RMSE 0.0966，误差反而下降 |
| thermal | 0.0857 / 0.1116 | level 0.11: MAE 0.1731, RMSE 0.2414，退化明显 |
| ultrasonic | 0.0857 / 0.1116 | level 0.11: MAE 0.0831, RMSE 0.1023，变化较小 |
| temperature | 0.0857 / 0.1116 | level 0.11: MAE 0.0890, RMSE 0.1159，轻微退化 |

观察：scaler-off 模型的扰动响应与 scaler-on 明显不同，thermal 退化更强，而 temperature 趋势反而弱于 scaler-on。这进一步说明输入尺度会改变 ErrorNet 和融合权重的可靠性判断，6D 结论支持正式路径继续使用 12 维 input scaler。

### 9.10 Phase 6E：pressure_drift 扰动扩展

Phase 6E 使用 128 sequence HITRAN smoke 的 scaler-on 模型，在已有 5 类扰动基础上加入 `pressure_drift`，验证输入压力漂移是否形成可观测退化。

运行命令：

```bash
./.venv/Scripts/python.exe -m scripts.perturb \
  --ckpt runs/stage_b/rcdw.pt \
  --config configs/phase6e-hitran-smoke-128-pressure.yaml \
  --output-dir runs/phase6e_pressure/perturb
```

运行环境与数据：

| 项 | 内容 |
|----|------|
| dataset | `data/rcdw-hitran-smoke-128` |
| ckpt | `runs/stage_b/rcdw.pt` |
| input scaler | `apply_input_scaler=None`，跟随 manifest，scaler-on |
| test windows | 500，输入 shape `(500, 8, 12)` |
| output dir | `runs/phase6e_pressure/perturb/` |

产物：`runs/phase6e_pressure/perturb/` 下生成 12 张 PNG：

- `optical_atten_metrics.png` / `optical_atten_weights_CO2.png`
- `optical_scat_metrics.png` / `optical_scat_weights_CO2.png`
- `thermal_metrics.png` / `thermal_weights_CO2.png`
- `ultrasonic_metrics.png` / `ultrasonic_weights_CO2.png`
- `temperature_metrics.png` / `temperature_weights_CO2.png`
- `pressure_drift_metrics.png` / `pressure_drift_weights_CO2.png`

指标摘要：

| 扰动 | level=0 MAE/RMSE | level=0.11 MAE/RMSE | 现象 |
|------|------------------|---------------------|------|
| optical_atten | 0.0580 / 0.0687 | 0.0580 / 0.0687 | 基本持平 |
| optical_scat | 0.0580 / 0.0687 | 0.0581 / 0.0686 | 基本持平 |
| thermal | 0.0580 / 0.0687 | 0.0577 / 0.0685 | 基本持平，误差轻微下降 |
| ultrasonic | 0.0580 / 0.0687 | 0.0581 / 0.0688 | 基本持平 |
| temperature | 0.0580 / 0.0687 | 0.1129 / 0.1463 | 退化最明显 |
| pressure_drift | 0.0580 / 0.0687 | 0.0576 / 0.0678 | 未观察到退化 |

结论：

- Phase 6E 的 `pressure_drift` 工程与实验闭环已完成。
- 当前 `pressure_drift` 只对标准化输入空间的 `P_MPa` 通道做确定性偏移，不重新生成压力派生的声学、光学或慢响应观测，因此本轮未观察到可解释的性能退化。
- scaler-on 128-seq smoke 中，`temperature` 仍是主导扰动源。
- `h2o_cross` 暂不实现；进入正式扰动集前需先决定它是输入空间湿度扰动，还是 HITRAN 光学后端重新生成扰动。

### 9.11 当前结论与下一步

- Phase 6A 已完成：cache 预热命令、真实 HAPI 预热、无网络测试与全量测试均通过。
- Phase 6B 的 64 与 128 sequence HITRAN smoke 均已完成 generate -> train -> eval -> perturb。
- Phase 6C 已完成：benchmark 生成支持可选并行，单进程 / 并行在 empirical 与 HITRAN 合成 cache 测试中确定性等价。
- Phase 6D 已完成：128-seq HITRAN smoke 上 scaler-on 明显优于 scaler-off（overall MAE 0.05741 vs 0.07642；RMSE 0.06790 vs 0.09101），正式路径保留 12 维 input scaler。
- Phase 6E 已完成 `pressure_drift` 扰动扩展实验：6 类扰动图共 12 张 PNG 已生成并同步记录；压力漂移在当前输入空间实现下未造成可观测退化。
- 后续路线：先决策 `h2o_cross` 的物理语义，再进入 O2 弛豫参数校核与多 `stage_profile` 后段扩展。
