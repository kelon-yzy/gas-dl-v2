# R5 小 MLP 观测特征回归 实施计划

> 状态：**默认 R5 正式 6000 未通过（2026-07-09）；R5-T 目标标准化正式 6000 通过（2026-07-10）**
> 日期：2026-07-09  
> 依据：[掘进通风项目记忆库.md](../../掘进通风项目记忆库.md) §5.4 / §6.8 / §6.9 / §8.4；[r5_tabpfn_implementation_plan.md](./r5_tabpfn_implementation_plan.md)（R5' 已完成）；[rocket_hydra_regression_implementation_plan.md](./rocket_hydra_regression_implementation_plan.md) 阶段 D；[small_sample_dl_strategies.md](../../methods/small_sample_dl_strategies.md) S2/S3；下列表格 MLP 文献。

## 结论

R5 在 **D0-observed 864 维** 特征上，把回归头从 RidgeCV / TabPFN 换成**可部署的小 MLP**，验证 R5' 揭示的非线性增益能否用轻量、离线、无第三方许可依赖的模型复现一部分。

| 对照 | val O₂ R² | 角色 |
|------|:---------:|------|
| D0-observed Ridge | 0.4226 | 可部署线性基线；R5 必须超过它 |
| R5' TabPFN | 0.6673 | 非线性上限探针（不可部署）；R5 报告相对差距 |
| R5 小 MLP（本计划） | **−0.1834** | ❌ 判据未通过；不作部署头 |
| R5-T 逐目标标准化 MLP | **0.6642** | ✅ 三个 eval split 均超过 D0 + 0.05；仅改变训练损失尺度 |

**判据（沿用 §6.4 / R5'）**：val O₂ 相对 D0-observed 提升是否 ≥ **+0.05**（即 ≥0.4726），且 test/extrap 同步提升。  
**默认 R5 实测**：四 split O₂ 全负；触发停止条件「全 split < D0+0.05」。
**R5-T 实测**：train/val/test/extrap O₂ R² 为 `0.8792 / 0.6642 / 0.6462 / 0.5815`；三个 eval split 均超过 D0 + 0.05，验证目标标准化是默认 R5 优化失败的主要原因。R5-T 不撤销默认 R5 的失败记录，而是作为独立、已通过的可部署 MLP 对照。

**非目标**：冲破 o2_bins 物理墙；单独达到验收线 0.70；复现 TabPFN 的全部 +0.245。

---

## 背景事实

1. **特征口径已冻结**：`d0_observed_physics_stats_v1`，864 维，与 `configs/tv3_d0_observed_ridge.json` / `configs/tv3_r5_tabpfn.json` 的 `physics_arrays` 逐位一致。禁止混入 oracle（true TOF / true sound speed / true alpha）。
2. **R5' 证明 observed 有非线性空间**（+0.245），但 TabPFN per-target 破坏闭包（`sum_abs_error`≈0.16），不可部署。
3. **旧 rocket 计划阶段 D 过时**：仍写「R0 1080 维 + lr 1e-4」。本计划改为 **D0-observed 864**，与 R5' 对齐。
4. **现有 `HandcraftMLPRegressor` 不可直接用**：绑定 `out_dim=4` + `gas_head`（hg 闭包头）。tv3 需要 **raw3、out_dim=3、无 gas_head**。
5. **工程入口已就绪**：`_build_head` 目前仅有 `ridgecv` / `ridge_closed_form` / `tabpfn`；扩展 `mlp` 即可复用特征缓存与 `evaluate_regressor`。

---

## 文献要点（支撑设计选择）

检索来源：CrossRef / arXiv（2026-07-09），workflow = multi-source-search。

| 文献 | 关键结论 | 对 R5 的用法 |
|------|----------|--------------|
| Holzmüller, Grinsztajn, Steinwart. **RealMLP**, arXiv:2407.04491 (NeurIPS 2024). DOI 会议版 [10.52202/079017-0837](https://doi.org/10.52202/079017-0837) | 强默认 MLP：robust scale + clip、3×256、AdamW（β₂=0.95）、dropout、目标标准化、多周期 lr、按 val 回滚；在 1K–500K 表数据上可与 GBDT 竞争 | **默认配方来源**：标准化、AdamW、dropout、按 val 选 checkpoint；第一版不引入 PBLD 数值 embedding（RealMLP-TD-S 简化版思路） |
| Gorishniy et al. **Revisiting Deep Learning Models for Tabular Data**, arXiv:2106.11959 | ResNet-like MLP 是常被漏掉的强基线；无万能架构 | R5 用「浅 MLP / 可选残差」作基线，不急上 Transformer |
| Gorishniy, Rubachev, Babenko. **On Embeddings for Numerical Features**, arXiv:2203.05556 | 数值特征 embedding（PLR 等）可显著抬升 MLP | **第二版可选消融**；第一版先 StandardScaler + 裸 MLP，控制变量 |
| Grinsztajn, Oyallon, Varoquaux. **Why do tree-based models still outperform DL on tabular data?**, arXiv:2207.08815 | ~10K 样本上树模型仍常胜 NN；NN 需抗无关特征、保方向、学不规则函数 | 预期 R5 可能达不到 TabPFN；成功标准是相对 **Ridge**，不是相对 TabPFN |
| Gorishniy et al. **Benchmarking Optimizers for MLPs in Tabular DL**, arXiv:2604.15297 | AdamW 仍是常用默认；Muon 更强但有开销；EMA 可改善 AdamW | 第一版 **AdamW**；不引入 Muon |
| 项目 [small_sample_dl_strategies.md](../../methods/small_sample_dl_strategies.md) S2/S3 | 小样本下轻量 + dropout/weight decay/early stop | hidden 不宜过大；dropout 0.1–0.25；early stop 必开 |
| Hollmann et al. *Nature* 2025; Chen et al. *JCIM* 2026 | TabPFN 为上限参考 | 只作对照，不参与 R5 训练 |

**设计翻译（第一版默认）**：

- 特征：`StandardScaler`（与 Ridge 一致；与 TabPFN「禁止外部缩放」相反——MLP 需要）
- 目标：可选 per-component 标准化后回归再反变换（RealMLP 回归做法）；第一版先 **直接预测百分浓度**，用 `weighted_component_mse` 权重 `[1, 2, 1]`
- 结构：共享 trunk + 三输出线性头（raw3），**不做** softmax/gas_head
- 优化：AdamW，lr `1e-3`（若不稳再试 `3e-4` / `1e-4`），weight decay `1e-4`，dropout `0.1`
- 训练：最多 200 epoch，batch 256（或全量若 GPU 显存允许），**val O₂ R² 或 val weighted loss early stop**，patience 20，回滚 best checkpoint
- 复杂度：优先 `(256, 128)`；若过拟合再试 `(128, 64)`

---

## 不变量

1. **特征与 D0-observed / R5' 逐位一致**（同一 `feature_builder` 与 `physics_arrays`）。
2. **输出 raw3**：`out_dim=3`，无闭包残差头、无 `target_transform`。
3. **评估口径不变**：train/val/test/extrapolation + `o2_bins` / `co2_bins` + `sum_abs_error`。
4. **判据锚点**：相对 D0-observed O₂ val **+0.05**；三 eval split 同步；禁止仅 train/val 虚高。
5. **不使用 TabPFN 输出作蒸馏标签**（R5 是纯自研监督；与已撤回的「TabPFN 可部署蒸馏」方案隔离）。
6. **缺失数组 / 特征名漂移 / 非有限值 → 直接失败**，无静默兜底。

---

## 当前工程切入点

| 入口 | 现状 | R5 改动 |
|------|------|---------|
| `tv3/ml/rocket_training.py` `_build_head` | ridge / tabpfn | 增加 `"mlp"` |
| 同上 `_model_diagnostics` | 依赖 `coef_` | `mlp` 返回结构摘要（层宽、参数量、best_epoch） |
| `tv3/pipeline/run_tv3_rocket_baseline.py` | `--head` choices 无 mlp | 增加 `mlp` + MLP 超参透传 |
| `configs/tv3_r5_tabpfn.json` | 特征母本 | 复制为 `tv3_r5_mlp.json`，只换 head |
| `tv3/dl/models/handcraft_mlp.py` | hg gas_head | **不复用**；新建 tv3 raw3 MLP 包装类 |
| 特征缓存 | D0-observed 已有 | 直接复用，不重建 |

---

## 最小实现闭环

### 模型契约

```python
class _ScaledMLPRegressor:
    """fit(x, y, feature_names=...) / predict(x) → (N, 3) float32"""
    # 内部: StandardScaler(x) → MLP → raw3
    # loss: weighted MSE, weights=(1.0, 2.0, 1.0)
```

建议实现位置：`tv3/ml/rocket_training.py` 内私有类（与 `_TabPFNMultiRegressor` 同级），或 `tv3/ml/mlp_head.py` 若文件过长再拆。

### 默认超参（写入 config）

| 键 | 默认值 | 说明 |
|----|--------|------|
| `head` | `"mlp"` | |
| `mlp_hidden_dims` | `[256, 128]` | 消融可改 `[128, 64]` |
| `mlp_dropout` | `0.1` | S2 起点 |
| `mlp_weight_decay` | `1e-4` | AdamW |
| `mlp_lr` | `1e-3` | 不稳则降到 `1e-4`（旧 rocket 草案） |
| `mlp_batch_size` | `256` | RealMLP 同量级 |
| `mlp_max_epochs` | `200` | |
| `mlp_patience` | `20` | early stop |
| `mlp_loss_weights` | `[1.0, 2.0, 1.0]` | O₂ 加倍 |
| `device` | `"cuda"` / `"auto"` | |
| `seed` | `20260704` | 与正式集一致；可选补 42/123/456 |

### 配置文件

`configs/tv3_r5_mlp.json`：在 `tv3_r5_tabpfn.json` 上改：

- `"head": "mlp"`
- `"output_dir": "outputs/tv3_r5/mlp_observed"`
- 增加上表 `mlp_*` 字段
- `physics_arrays` / `feature_builder` **一字不动**

### CLI

```bash
# 本地 smoke（小数据集）
python -m tv3.pipeline.run_tv3_rocket_baseline --config configs/tv3_r5_mlp_smoke.json

# 正式 6000
python -m tv3.pipeline.run_tv3_rocket_baseline --config configs/tv3_r5_mlp.json
```

---

## 分阶段实施步骤

### R5-0：对照表冻结（0.5 h）

- [x] 确认本地/服务器存在 D0-observed 与 R5' 的 `metrics.json`
- [x] 确认 `data/tv3-formal-6000/features/rocket/d0_observed_physics_stats_v1` 缓存可用
- [x] 写清对照数字：D0 O₂ val=0.4226；R5' =0.6673

### R5-1：MLP head 代码（0.5–1 天）

- [x] 实现 `_ScaledMLPRegressor`（PyTorch；延迟 import torch）
- [x] `_build_head("mlp")` + diagnostics
- [x] CLI `--head mlp` 与 config 透传
- [x] 单元测试：形状 `(N,F)→(N,3)`、无 NaN、early stop 回滚、ridge/tabpfn 零回归

### R5-2：配置与 smoke（0.5 天）

- [x] `configs/tv3_r5_mlp.json`（6000）
- [x] `configs/tv3_r5_mlp_smoke.json`（`data/tv3-smoke`，5 epoch 本地冒烟）
- [x] 单元测试覆盖 mlp head（`tests/test_tv3_r5_mlp.py`）；正式 6000 已直接跑通（smoke 可选）

### R5-3：正式 6000 单 seed（0.5–1 天）

- [x] 服务器跑默认 `(256,128)` → 产物 `outputs/tv3_r5/mlp_observed/`
- [ ] ~~若 val O₂ 相对 D0 <+0.02 且 train 远高于 val：改 `(128,64)` 或加大 dropout 到 0.2 再跑一版~~（未触发：train 已负，属欠拟合而非过拟合；可选低优先级救援，不阻塞光学通道）
- [ ] ~~若仍不稳：lr→`1e-4`~~（同上，可选）

### R5-4：判读与回填（0.5 天）

- [x] 填 §「预期结果」实测表
- [x] 更新记忆库 §5.4 / §6.9 / §8.4
- [x] 按停止条件决定：接受 D0 Ridge 为可部署上限；D1 降级；初始 P0 为 O₂ 光学通道。2026-07-10 起 TDLAS 硬件暂缓，R5-T 与 R7 作为服务器算法对照。

### R5-5（可选）：稳定性

- [ ] seeds `42,123,456` — **跳过**（单 seed 未过 +0.05）

### R5-T：逐目标标准化救援对照（TDLAS 暂缓期）

- [x] 新增 `mlp_standardize_targets`：仅在训练损失空间按组分标准化，预测反变换为原始 `raw3` 百分比。
- [x] 新增 `configs/tv3_r5_mlp_target_scaled.json`；特征与 `tv3_r5_mlp.json` 逐位一致。
- [x] 单元测试验证输出仍为 raw3；本地 R5/R7 联合测试 13 项通过。
- [x] 服务器执行正式 `tv3-formal-6000`：`outputs/tv3_r5/mlp_observed_target_scaled/metrics.json`，best epoch=35。

R5-T 是独立救援对照，不撤销默认 R5 的失败结论。2026-07-11 的新增 seeds `42/123/456` 均达到三个 eval split 的 D0+0.05 门槛，val/test/extrap O₂ R²均值±标准差为 `0.6631±0.0082 / 0.6438±0.0055 / 0.5890±0.0197`。正式汇总见 `outputs/tv3_r5t_b6_multiseed/replication_report.json`；当前 random split 下 stable_pass，下一步改为独立 split / OOD 验证，而不是继续搜索当前 MLP 超参数。

---

## 影响文件清单

| 文件 | 改动 |
|------|------|
| `tv3/ml/rocket_training.py` | `_ScaledMLPRegressor`；`_build_head` / diagnostics / 训练超参透传 |
| `tv3/pipeline/run_tv3_rocket_baseline.py` | `--head mlp`；`mlp_*` CLI/config |
| `configs/tv3_r5_mlp.json` | 正式配置 |
| `configs/tv3_r5_mlp_smoke.json` | smoke 配置 |
| `configs/tv3_r5_mlp_target_scaled.json` | R5-T 正式配置（已执行） |
| `tests/test_tv3_rocket_pipeline.py` 或新建 `tests/test_tv3_r5_mlp.py` | mlp head smoke |
| `outputs/tv3_r5/mlp_observed/` | 训练产物 |
| `docs/掘进通风项目记忆库.md` | 结果回填 |

**不改**：特征构建、D0/R5' 配置、D2 代码、TabPFN 路径、schema。

---

## 验收标准

| 项 | 标准 |
|----|------|
| 工程 | `head=mlp` 跑通；`metrics.json` 完整；ridge/tabpfn 测试零回归 |
| 特征 | 864 维，builder=`d0_observed_physics_stats_v1`，无 oracle |
| 主判据 | val O₂ ≥ D0 + 0.05（≥0.4726）；test/extrap 同步 ≥ D0 + 0.05 |
| 对照报告 | 写出相对 R5' 的差距（不要求追上） |
| 闭包监控 | 报告 `sum_abs_error`（不强制 ≤某阈值，但若 ≫ Ridge 需在结论中说明） |
| 健康度 | 记录 train−val O₂ gap；若 gap>0.35 且 val 未过判据 → 判过拟合失败 |

---

## 停止条件与分支

| 结果 | 动作 |
|------|------|
| 三 split 均 ≥ D0+0.05 | **通过**；可部署非线性头成立；D1 可排期但优先级仍低于光学通道 |
| 仅 val 过、test/extrap 不过 | **不通过**（与 R5' 停止条件 3 同逻辑） |
| 全 split < D0+0.05 | TabPFN 增益难用浅 MLP 复现；接受 D0 Ridge 为可部署上限；D1 降级 |
| 接近 TabPFN（差距 <0.05） | 强成功；后续以 R5 为部署默认，不必再碰 TabPFN |
| o2_bins 仍全负 | **预期内**，不作为失败条件 |

---

## 暂不做

- 不用 TabPFN / Ridge 软标签做知识蒸馏（保持 R5 为纯硬标签实验）。
- 不上 RealMLP 全套 PBLD embedding / 多周期 coslog（第一版控制变量；过判据后再考虑）。
- 不引入 `gas_head` / softmax 强制闭包。
- 不改特征口径去凑分。
- 不把 R5 当作冲 0.70 验收线的手段。

---

## 预期结果（实施前推断）与实测

| 指标 | 推断 | 实测（2026-07-09） | 判读 |
|------|------|-------------------|------|
| val O₂ | 0.48–0.60 | **−0.1834** | 远低于推断；低于均值预测 |
| vs D0 +0.05 | 有机会通过 | **未通过**（Δ≈−0.606） | 触发「全 split < D0+0.05」 |
| vs R5' | 大概率低 0.05–0.20 | 低约 **0.85** | 浅 MLP 未承接 foundation 增益 |
| o2_bins | 仍全负 | 全负且比 D0 更差 | 预期内；无额外信息 |
| `sum_abs_error` | 高于 Ridge、低于 TabPFN | val≈**1.65**（> TabPFN 0.16） | 闭包也差于线性基线 |
| train O₂ | — | **−0.1118** | 欠拟合 / 弱信号优化失败，非过拟合 |

R5-T 正式结果：val/test/extrap O₂ R² 为 `0.6642 / 0.6462 / 0.5815`，相对 D0 分别提升 `+0.2416 / +0.1891 / +0.2107`。其 val O₂ R² 距 R5' TabPFN 仅 `0.0031`，说明当前 observed 特征中的非线性可由小型 MLP 承接。R5-T val `sum_abs_error=0.0307`，高于 Ridge 的近零水平，作为 raw3 直接输出的闭包监控项保留，但不构成当前验收否决条件。

---

## 关键文献

1. Holzmüller D, Grinsztajn L, Steinwart I. Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data. arXiv:2407.04491. 2024. https://arxiv.org/abs/2407.04491  
2. Gorishniy Y, Rubachev I, Khrulkov V, Babenko A. Revisiting Deep Learning Models for Tabular Data. arXiv:2106.11959. 2021. https://arxiv.org/abs/2106.11959  
3. Gorishniy Y, Rubachev I, Babenko A. On Embeddings for Numerical Features in Tabular Deep Learning. arXiv:2203.05556. 2022. https://arxiv.org/abs/2203.05556  
4. Grinsztajn L, Oyallon E, Varoquaux G. Why do tree-based models still outperform deep learning on tabular data? arXiv:2207.08815. 2022. https://arxiv.org/abs/2207.08815  
5. Gorishniy Y, et al. Benchmarking Optimizers for MLPs in Tabular Deep Learning. arXiv:2604.15297. 2026. https://arxiv.org/abs/2604.15297  
6. Hollmann N, et al. Accurate predictions on small data with a tabular foundation model. *Nature*. 2025;637:319-326. doi:10.1038/s41586-024-08328-6  
7. Chen W, et al. TabPFN Opens New Avenues for Small-Data Tabular Learning in Drug Discovery. *J Chem Inf Model*. 2026;66:3525-3539. doi:10.1021/acs.jcim.5c02823  

项目内：

- [r5_tabpfn_implementation_plan.md](./r5_tabpfn_implementation_plan.md) — 上限探针与判据  
- [rocket_hydra_regression_implementation_plan.md](./rocket_hydra_regression_implementation_plan.md) — 阶段 D（本计划已修正特征口径）  
- [small_sample_dl_strategies.md](../../methods/small_sample_dl_strategies.md) — S2/S3  
- [references/observed_o2_algorithm_review.md](../../references/observed_o2_algorithm_review.md) — TDLAS 暂缓期的 TOF 环境补偿与表格回归文献依据

---

## 实施记录

| 日期 | 事件 |
|------|------|
| 2026-07-09 | 文献检索完成；本实施计划落盘。代码未开工。 |
| 2026-07-09 | R5-1/R5-2 落地：`tv3/ml/mlp_head.py`、`_build_head("mlp")`、CLI/config、`tv3_r5_mlp*.json`、`tests/test_tv3_r5_mlp.py`（7 项 rocket+mlp 测试通过）。 |
| 2026-07-09 | R5-3 正式 6000：`outputs/tv3_r5/mlp_observed/metrics.json`。best_epoch=175，params=254723。val O₂=−0.183；train/test/extrap O₂ 全负。 |
| 2026-07-09 | R5-4 判读：**判据未通过**。停止条件「全 split < D0+0.05」触发 → 可部署上限维持 D0-observed Ridge；D1 降级；下一步 P0 = O₂ 光学通道（TDLAS）。已回填记忆库 §5.4 / §6.9 / §8.4。 |
| 2026-07-10 | TDLAS 硬件暂缓；新增 R5-T 逐目标标准化 MLP。代码与本地测试完成，正式 tv3-formal-6000 仅在服务器，结果待回填。 |
| 2026-07-10 | R5-T 正式 6000：`outputs/tv3_r5/mlp_observed_target_scaled/metrics.json`。best epoch=35，val/test/extrap O₂ R²=`0.6642/0.6462/0.5815`，三个 eval split 均通过 D0+0.05 判据。 |
