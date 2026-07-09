# R5 小 MLP 观测特征回归 实施计划

> 状态：R5-1 / R5-2 已实施（代码 + 配置 + 测试）；R5-3 正式训练待跑  
> 日期：2026-07-09  
> 依据：[掘进通风项目记忆库.md](掘进通风项目记忆库.md) §5.4 / §6.8 / §8.4；[r5_tabpfn_implementation_plan.md](r5_tabpfn_implementation_plan.md)（R5' 已完成）；[rocket_hydra_regression_implementation_plan.md](rocket_hydra_regression_implementation_plan.md) 阶段 D；[small_sample_dl_strategies.md](small_sample_dl_strategies.md) S2/S3；下列表格 MLP 文献。

## 结论

R5 在 **D0-observed 864 维** 特征上，把回归头从 RidgeCV / TabPFN 换成**可部署的小 MLP**，验证 R5' 揭示的非线性增益能否用轻量、离线、无第三方许可依赖的模型复现一部分。

| 对照 | val O₂ R² | 角色 |
|------|:---------:|------|
| D0-observed Ridge | 0.4226 | 可部署线性基线；R5 必须超过它 |
| R5' TabPFN | 0.6673 | 非线性上限探针（不可部署）；R5 报告相对差距 |
| R5 小 MLP（本计划） | 待测 | 可部署非线性候选 |

**判据（沿用 §6.4 / R5'）**：val O₂ 相对 D0-observed 提升是否 ≥ **+0.05**（即 ≥0.4726），且 test/extrap 同步提升。  
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
| 项目 [small_sample_dl_strategies.md](small_sample_dl_strategies.md) S2/S3 | 小样本下轻量 + dropout/weight decay/early stop | hidden 不宜过大；dropout 0.1–0.25；early stop 必开 |
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

- [ ] 确认本地/服务器存在 D0-observed 与 R5' 的 `metrics.json`
- [ ] 确认 `data/tv3-formal-6000/features/rocket/d0_observed_physics_stats_v1` 缓存可用
- [ ] 写清对照数字：D0 O₂ val=0.4226；R5' =0.6673

### R5-1：MLP head 代码（0.5–1 天）

- [x] 实现 `_ScaledMLPRegressor`（PyTorch；延迟 import torch）
- [x] `_build_head("mlp")` + diagnostics
- [x] CLI `--head mlp` 与 config 透传
- [x] 单元测试：形状 `(N,F)→(N,3)`、无 NaN、early stop 回滚、ridge/tabpfn 零回归

### R5-2：配置与 smoke（0.5 天）

- [x] `configs/tv3_r5_mlp.json`（6000）
- [x] `configs/tv3_r5_mlp_smoke.json`（`data/tv3-smoke`，5 epoch 本地冒烟）
- [ ] 本地 CPU/GPU smoke 跑通并写出 `metrics.json`（需先生成 `data/tv3-smoke`）

### R5-3：正式 6000 单 seed（0.5–1 天）

- [ ] 服务器跑默认 `(256,128)`
- [ ] 若 val O₂ 相对 D0 <+0.02 且 train 远高于 val：改 `(128,64)` 或加大 dropout 到 0.2 再跑一版
- [ ] 若仍不稳：lr→`1e-4`（对齐旧 rocket 草案）

### R5-4：判读与回填（0.5 天）

- [ ] 填 §「预期结果」实测表
- [ ] 更新记忆库 §5.4 / §6 / §8.4
- [ ] 按停止条件决定 D1 / 接受 D0 / 加深 MLP

### R5-5（可选）：稳定性

- [ ] seeds `42,123,456` 仅在单 seed **通过 +0.05** 后执行
- [ ] 报告 mean±std O₂ R²

---

## 影响文件清单

| 文件 | 改动 |
|------|------|
| `tv3/ml/rocket_training.py` | `_ScaledMLPRegressor`；`_build_head` / diagnostics / 训练超参透传 |
| `tv3/pipeline/run_tv3_rocket_baseline.py` | `--head mlp`；`mlp_*` CLI/config |
| `configs/tv3_r5_mlp.json` | 正式配置 |
| `configs/tv3_r5_mlp_smoke.json` | smoke 配置 |
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

## 预期结果（实施前推断）

| 指标 | 推断 | 依据 |
|------|------|------|
| val O₂ | 0.48–0.60 | 介于 Ridge 0.42 与 TabPFN 0.67；浅 MLP 通常吃不下全部 foundation 增益（Grinsztajn 2022） |
| vs D0 +0.05 | 有机会通过 | R5' 已证明非线性存在 |
| vs R5' | 大概率低 0.05–0.20 | RealMLP 也需强默认与调参才近 GBDT；本计划刻意轻量 |
| o2_bins | 仍全负 | 物理墙 |
| `sum_abs_error` | 高于 Ridge、低于 TabPFN | 共享 trunk 会学到组分相关，但不保证闭包 |

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

- [r5_tabpfn_implementation_plan.md](r5_tabpfn_implementation_plan.md) — 上限探针与判据  
- [rocket_hydra_regression_implementation_plan.md](rocket_hydra_regression_implementation_plan.md) — 阶段 D（本计划已修正特征口径）  
- [small_sample_dl_strategies.md](small_sample_dl_strategies.md) — S2/S3  

---

## 实施记录

| 日期 | 事件 |
|------|------|
| 2026-07-09 | 文献检索完成；本实施计划落盘。代码未开工。 |
| 2026-07-09 | R5-1/R5-2 落地：`tv3/ml/mlp_head.py`、`_build_head("mlp")`、CLI/config、`tv3_r5_mlp*.json`、`tests/test_tv3_r5_mlp.py`（7 项 rocket+mlp 测试通过）。 |
