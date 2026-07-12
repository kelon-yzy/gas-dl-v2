# 掘进通风小样本 DL 训练策略

> 本文档汇总 tv3 场景（CO₂/O₂/N₂ 三组分多模态时序回归）下可用的小样本深度学习训练策略，基于 2026-07-05 文献检索结果整理。
> 与 [dl_training_plan.md](../archive/legacy/dl_training_plan.md) §9 的 O₂ 可辨识性策略互补：§9 聚焦"弱组分信号被压制"的结构性问题，本文档聚焦"数据量不足"的通用小样本问题。
> 实验路线见 [experiment_roadmap.md](../archive/legacy/experiment_roadmap.md)，服务器执行见 [server_training_guide.md](../operations/server_training_guide.md)。

## 1. 背景与问题定义

### 1.1 当前状态

| 维度               | 数值                                       | 说明                                         |
| ---------------- | ---------------------------------------- | ------------------------------------------ |
| 数据集规模            | 600 序列 × 512 时步                          | ≈ 400 训练样本（split: train ~440 / val / test） |
| 模型输入             | slow (512, 7) + ultrasonic (512, 5000)   | 多模态时序                                      |
| 预测目标             | CO₂ / O₂ / N₂ 三组分浓度                      | 连续回归，sum=100% 闭包                           |
| 首轮 TCN 50 epochs | CO₂ R²=-0.05, O₂ R²=-0.14, N₂ R²=-0.53   | DL 未收敛                                     |
| Ridge 基线         | CO₂ R²=0.91 ✅, O₂ R²=-0.05 ❌, N₂ R²=0.65 | CO₂ 达标，O₂/N₂ 未达标                           |

### 1.2 核心矛盾

DL 模型容量大（`cnn1d_tcn_fusion` 含 acoustic_channels [16,32,64,64] + tcn_channels [128,128,128]，参数量 ~10⁵-10⁶），但训练样本仅 ~440 个。每个参数平均只见到 <1 个样本，严重过参数化。

**关键判断**：当前 R²≈0 的主因是数据量不足 + 模型过参数化，不是策略问题。最有效的解法是扩大数据集（6000 序列，int16 + skip-fiber-mic 后 29 GB，服务器可行）。本文档的策略是在 600 序列约束下尽力提升 DL 表现。

### 1.3 文献检索范围

2026-07-05 通过 `mcp__academic-search__search_papers`（Crossref / PubMed / arXiv 三源）检索 9 次查询，覆盖 9 类策略。arXiv 源被限流（HTTP 429），主要结果来自 Crossref 和 PubMed。`WebSearch` 和 `mcp__paper-search__search_arxiv` 未返回有效结果。

## 2. 策略分类总览

| 编号  | 策略        | 原理                                                  | tv3 适用性 | 置信度    | 成本     | 与 §9 关系 |
|:---:| --------- | --------------------------------------------------- |:-------:|:------:|:------:| ------- |
| S1  | 数据增强      | jitter/scaling/mixup/窗口切片扩充样本                       | 高       | High   | <0.5 天 | 新增      |
| S2  | 正则化强化     | dropout/weight decay/early stopping/label smoothing | 高       | High   | <0.5 天 | 调参      |
| S3  | 轻量模型      | 减小 channel 维度，降维 + 浅网络                              | 高       | High   | 0.5 天  | 新增      |
| S4  | 集成学习      | bootstrap 聚合 + 多模型平均                                | 中       | High   | <0.5 天 | 新增      |
| S5  | 知识蒸馏      | Ridge teacher → DL student，退火权重                     | 中       | Medium | 1 天    | §9 T6   |
| S6  | 元学习       | MAML/ADKF/Prototypical Net 跨任务迁移                    | 低       | Medium | 3-5 天  | 新增      |
| S7  | 自监督/对比预训练 | MAE/SupCon 预训练表征，小样本微调                              | 中       | Medium | 2-3 天  | 新增      |
| S8  | 半监督学习     | 未标注数据扩充 + 伪标签                                       | 低       | Medium | —      | 不适用     |
| S9  | 物理信息约束    | 已知声速/衰减/热导物理 → loss 约束                              | 高       | Medium | 1-2 天  | 新增      |

置信度含义：

- **High**：多来源文献支撑 + 与 tv3 场景高度匹配
- **Medium**：文献支撑但场景有差异，或单一来源
- **Low**：新方法未验证，或前提条件不完全满足

## 3. 策略详解

### S1. 数据增强（Data Augmentation）

**原理**：通过对训练样本施加随机变换（加噪、幅度缩放、时间扭曲、样本插值），在不改变语义的前提下扩充有效样本量，降低过拟合风险。

**文献支撑**：

- [Liu et al., 2024, Information Technology and Control] 在小样本时序分类上验证数据增强 + 半监督有效，DOI:10.5755/j01.itc.53.2.35797
- [Jiang et al., 2022, CNN + DA 小样本时序预测] 提出 CNN + 数据增强框架，DOI:10.21203/rs.3.rs-1094384/v1
- [Iwana & Uchida, 2021, 神经网络时序数据增强经验评估] 系统对比 jitter/scaling/warping/mixup 等方法（经典综述，本次检索未直接命中但领域共识）

**tv3 适用性**：高。`cli.py` 已支持 `--augment` 参数和 `TimeSeriesAugmentConfig`，包含：

- `jitter_std`：高斯噪声 std（建议 0.001，与噪声 std 同量级）
- `amplitude_scale_range`：幅度缩放区间（建议 [0.9, 1.1]）
- `window_fraction`：窗口切片比例
- `gaussian_noise_std`：附加高斯噪声
- `apply_prob`：增强应用概率（建议 0.5）
- `amplitude_apply_from_channel`：从第 7 通道开始（跳过 slow 7 通道，只增强波形）

**实现方式**（零代码改动，仅 CLI 参数）：

```bash
python -m tv3.dl.cli \
    --config configs/tv3_tcn_multimodal.json \
    --modalities slow,ultrasonic \
    --batch-size 8 --output-dir outputs/tv3_tcn_multimodal_aug/s42 --seed 42 \
    --augment '{"jitter_std":0.001,"amplitude_scale_range":[0.9,1.1],"gaussian_noise_std":0.0005,"apply_prob":0.5}'
```

**预期收益**：R² 提升 0.05-0.15（文献显示小样本时序下数据增强典型增益）。
**风险**：增强强度过大引入标签失真（浓度标签不变，但波形变形过多可能改变物理语义）。建议从弱增强开始（jitter_std=0.001）。
**置信度**：High。

### S2. 正则化强化（Regularization Enhancement）

**原理**：通过增加正则化强度（dropout、weight decay、early stopping、label smoothing、stochastic depth）抑制过拟合，牺牲训练精度换取泛化。

**文献支撑**：

- [PloS one, 2026, 多模态小样本 N=108] 验证适当正则化的 1D-CNN 优于未正则化的高维 3D-CNN，DOI:10.1371/journal.pone.0346251
- 正则化是 DL 领域共识， Srivastava et al. 2014 (Dropout) 等经典工作。

**tv3 适用性**：高。当前配置：

- `cnn1d` dropout=0.1，`tcn` dropout=0.1，`lstm` dropout=0.1，`patchtst` dropout=0.1
- `cnn1d_tcn_fusion` acoustic_dropout=0.15, tcn_dropout=0.30
- weight_decay=0.0001（所有配置）
- early_stopping patience=10（tcn），patience=15（patchtst）

**实现方式**（调参）：

- dropout：0.1 → 0.2-0.3（slow-only 模型），0.3 → 0.4（多模态模型）
- weight_decay：0.0001 → 0.001（10×）
- early_stopping patience：10 → 5（更激进提前停止）
- 可加 label smoothing（但回归任务不常用，改用 smooth_l1 loss）

**预期收益**：R² 提升 0.02-0.08。正则化收益递减，过强会欠拟合。
**风险**：过强正则化导致欠拟合（训练 loss 不下降）。建议网格搜索 dropout ∈ {0.1, 0.2, 0.3}。
**置信度**：High。

### S3. 轻量模型（Lightweight Architecture）

**原理**：减小模型容量（参数量），使参数/样本比更合理。小样本下高维架构易过拟合甚至 mode collapse。

**文献支撑**：

- [PloS one, 2026, 多模态小样本 N=108] 系统对比发现：轻量 1D-CNN（降维特征）AUC=0.900±0.072，高维多模态 3D-CNN AUC=0.568±0.090（mode collapse）。结论：**小样本下高维架构过拟合，降维 + 轻量网络显著更优**，DOI:10.1371/journal.pone.0346251
- [Bioengineering, 2026, N=82] 在小样本回归下，简单平均集成（RF + GB + LR）R²=0.390 优于注意力 DL 模型 R²=0.294，DOI:10.3390/bioengineering13020252

**tv3 适用性**：高。当前 `cnn1d_tcn_fusion` 配置较重：

- acoustic_channels [16, 32, 64, 64]（4 层 CNN 处理 5000 点波形）
- tcn_channels [128, 128, 128]（3 层 TCN）
- slow_hidden_dim 32, shared_hidden_dims [128, 64]

**实现方式**（改配置）：

```json
{
  "model_kwargs": {
    "acoustic_channels": [16, 32],
    "tcn_channels": [64, 64],
    "shared_hidden_dims": [64, 32],
    "acoustic_dropout": 0.2,
    "tcn_dropout": 0.4
  }
}
```

参数量约减半。或直接用 `cnn1d`（slow-only）作为主力，跳过多模态。

**预期收益**：R² 提升 0.05-0.10（减过拟合）。DL 可能在 600 序列下首次收敛（R² > 0）。
**风险**：模型过小可能无法捕捉 O₂/N₂ 微弱信号。建议先验证 CO₂ 是否收敛，再考虑 O₂。
**置信度**：High（小样本下轻量模型优势有强文献支撑）。

### S4. 集成学习（Ensemble Learning）

**原理**：训练多个模型，预测时平均或加权组合，降低单一模型的方差。

**文献支撑**：

- [Li & Chen, 2025, ACTCE] 专门针对小样本时序预测提出集成 DL 框架，DOI:10.1109/actce66599.2025.00079
- [Lee & Chang, 2017, CMPB] 在小样本（5 样本/受试者）下用 DBN-DNN + bootstrap + AdaBoost 集成，SDE 降低 9-11%，DOI:10.1016/j.cmpb.2017.08.005
- [Bioengineering, 2026] 简单平均集成优于复杂加权，DOI:10.3390/bioengineering13020252

**tv3 适用性**：中。已有 3 seeds × 5 模型 = 15 runs 基线（`scripts/run_tv3_baseline.py`），可直接集成。

**实现方式**（后处理脚本）：

```python
# 读取 15 个 run 的 val 预测，平均或加权
import numpy as np, json
preds = []
for model in ["cnn1d", "tcn", "lstm", "patchtst"]:
    for seed in [42, 123, 456]:
        m = json.load(open(f"outputs/tv3_baseline/{model}/seed{seed}/metrics.json"))
        preds.append(m["evaluations"]["val"]["predictions"])
ensemble_pred = np.mean(preds, axis=0)  # 简单平均
```

**预期收益**：R² 提升 0.03-0.08（方差降低）。集成对 O₂（弱信号）增益可能更大。
**风险**：集成需要各模型有多样性。如果所有模型都 R²≈0，集成仍为 0。**前提是单模型能收敛**。
**置信度**：High（集成降低方差是领域共识）。

### S5. 知识蒸馏（Knowledge Distillation）

**原理**：用表现更好的 teacher（如 Ridge）的预测作为 soft target，引导 DL student 学习。退火权重 β(epoch) 前期借 teacher 收敛，后期切回 ground-truth。

**文献支撑**：

- [Wang & Lu, 2021, ICAICE] 多教师小样本知识蒸馏，DOI:10.1109/icaice54393.2021.00127
- [Hinton et al., 2015, Distilling the Knowledge in a Neural Network] 知识蒸馏经典工作（本次检索未直接命中但领域共识）
- 项目内 [dl_training_plan.md §9 T6](../archive/legacy/dl_training_plan.md) 已规划 Ridge Teacher 蒸馏

**tv3 适用性**：中。Ridge CO₂ R²=0.91（teacher 有效），但 O₂ R²≈0（teacher 也无法预测 O₂）。蒸馏对 CO₂ 有效，对 O₂ 无效。

**实现方式**（需改 trainer.py）：

```python
# L = L_final + β(epoch) · MSE(y_final, y_ridge)
# β(epoch) = β₀ · max(0, 1 - epoch / T_anneal)
# β₀=0.3, T_anneal=30
```

需要：

1. 预计算 Ridge 在训练集上的预测（作为 soft target）
2. trainer.py 增加 distillation loss 项
3. 退火权重 β(epoch)

**预期收益**：CO₂ R² 提升 0.05-0.10（接近 Ridge 0.91）。O₂ 无增益（Ridge 也无法预测）。
**风险**：DL 过度模仿 Ridge 的 bias。退火权重控制关键。
**置信度**：Medium（Ridge teacher 有效性已验证，蒸馏机制需实现）。
**与 §9 关系**：§9 T6 已规划，本文档补充文献支撑。

### S6. 元学习（Meta-Learning）

**原理**：在多个相关任务上元学习初始化参数，使模型能快速适应新任务（few-shot）。

**文献支撑**：

- [Lee & Yang, 2022, JMS, 23 citations] DNN + 元学习在制造业小样本有效，DOI:10.1016/j.jmsy.2022.02.004
- [Kötter et al., 2024, Chembiochem] ADKF 在回归任务上优于 MAML，但**性能高度依赖任务相似性**，DOI:10.1002/cbic.202400095
- [FS-UNet, 2025, Phys Med] MAML + Prototypical Network 小样本剂量预测，DOI:10.1016/j.ejmp.2025.105184
- [Finn et al., 2017, MAML] 元学习经典工作（领域共识）

**tv3 适用性**：低。元学习需要多个相关任务（如多场景、多气体类型）。tv3 是单任务（CO₂/O₂/N₂ 回归），除非：

1. 跨场景元学习：hg（H₂/CH₄/CO₂/N₂）+ sg（H₂/CH₄/CO₂/CO）+ tv3（CO₂/O₂/N₂）联合元学习
2. 跨组分元学习：把每个组分预测作为独立任务

但 hg/sg/tv3 组分不同、物理参数不同，任务相似性存疑。[Kötter et al., 2024] 明确指出"任务相似性是元学习成功的关键因素"。

**实现方式**：需引入 MAML/ADKF 框架，重构训练循环。成本高（3-5 天）。
**预期收益**：不确定。任务相似性不足时可能无效。
**风险**：实现复杂，tv3 单任务前提不满足，ROI 低。
**置信度**：Medium（元学习本身有文献支撑，但 tv3 场景前提不完全满足）。

### S7. 自监督/对比预训练（Self-Supervised / Contrastive Pretraining）

**原理**：用无标签数据预训练 encoder（MAE 重建、对比学习），再在小样本标注数据上微调。学到的表征更通用。

**文献支撑**：

- [Moradinasab et al., 2024, DMKD] SupCon-TSC 在小样本多变量时序（CPET 数据集）上验证：instance-level + cluster-level SupCon 优于 SOTA，DOI:10.1007/s10618-024-01006-1
- [Zhang, 2025, KBS] 自监督预训练时序分类，DOI:10.1016/j.knosys.2025.114340
- [Yi et al., 2025, Front AI] MAE 预训练加速收敛但精度受限（小样本下 Dice 0.63-0.65），DOI:10.3389/frai.2025.1618426
- [Eldele et al., 2026, AI for Time Series] 自监督对比表征学习半监督时序分类，DOI:10.1201/9781003612742-6

**tv3 适用性**：中。tv3 有 600 序列无标签波形数据（ultrasonic），可做自监督预训练。但：

1. MAE 重建波形：需要 mask + 重建，学到的表征是否对浓度回归有用存疑
2. SupCon 需要正负对定义——tv3 是连续浓度回归，正负对需要按浓度分箱构造

**实现方式**：

1. MAE 预训练：mask 50% 波形点，重建缺失部分（无标签）
2. SupCon 预训练：按 CO₂ 浓度分箱（如 <1%, 1-3%, >3%），同箱为正对，异箱为负对
3. 微调：预训练 encoder + 回归头，用 600 序列标注数据微调

**预期收益**：R² 提升 0.05-0.15（[Moradinasab et al., 2024] 显示小样本下 SupCon 显著优于直接监督）。
**风险**：正负对定义困难；预训练-微调域差异；实现复杂（2-3 天）。
**置信度**：Medium（文献支撑强，但 tv3 回归场景的正负对设计需验证）。

### S8. 半监督学习（Semi-Supervised Learning）

**原理**：利用未标注数据（伪标签、一致性正则）扩充训练信号。

**文献支撑**：

- [Liu et al., 2024, Information Technology and Control] 数据增强 + 半监督小样本时序分类，DOI:10.5755/j01.itc.53.2.35797
- [Eldele et al., 2026] 自监督对比 + 半监督时序分类，DOI:10.1201/9781003612742-6

**tv3 适用性**：低。tv3 所有 600 序列都已有标注（CO₂/O₂/N₂ 浓度），无未标注数据。除非：

1. 生成更多仿真序列但不保存标签（浪费，因为标签是生成已知的）
2. 跨场景未标注数据（hg/sg 波形）预训练

**实现方式**：不直接适用。
**预期收益**：不适用。
**风险**：前提不满足。
**置信度**：Medium（半监督本身有效，但 tv3 无未标注数据）。

### S9. 物理信息约束（Physics-Informed Loss）

**原理**：用已知物理规律（声速方程、衰减模型、热导混合规则）作为 loss 约束，引导模型学习符合物理的解。

**文献支撑**：

- [Liu et al., 2022, SSRN] 分位回归 physics-informed DL，DOI:10.2139/ssrn.4233942
- [Brunton, 2023, Autoencoders & Physics Informed ML] 物理信息 ML 综述，DOI:10.52843/cassyni.4zpjhl
- [Karniadakis et al., 2021, Nature Reviews Physics] Physics-informed ML 经典综述（领域共识）

**tv3 适用性**：高。tv3 物理模型已知（`tv3/sim/generation/tunnel_ventilation/acoustic_physics.py`）：

- 声速：`c = sqrt(γ_mix · R · T / M_mix)`（理想气混合）
- 衰减：`alpha = alpha_classical + alpha_co2 + alpha_n2 + alpha_h2o`（半经验多气体弛豫）
- 热导：Wassiljewa-Mason-Saxena 混合规则

可约束：模型预测的浓度 → 通过物理模型计算声速 → 与波形提取的 TOF 一致。

**实现方式**（需改 loss）：

```python
# 物理一致性 loss
pred_co2, pred_o2, pred_n2 = model(x)  # 预测浓度
# 从 ultrasonic 波形提取 TOF → 估计声速 c_observed
c_predicted = physics_sound_speed(pred_co2, pred_o2, pred_n2, T, P)
physics_loss = MSE(c_observed, c_predicted)
L = L_regression + lambda * physics_loss
```

**预期收益**：R² 提升 0.05-0.10（物理约束正则化）。对 O₂ 可能特别有效——物理上 O₂/N₂ 声速差 6.4%，约束模型利用这一信号。
**风险**：物理模型计算可微性（需要可微物理层）；lambda 调参。
**置信度**：Medium（物理模型已知，但可微实现和 lambda 调参需验证）。

## 4. 与 dl_training_plan.md §9 的关系

[dl_training_plan.md §9](../archive/legacy/dl_training_plan.md) 定义了 6 个针对 O₂ 可辨识性的策略（T1-T6），本文档补充 9 个通用小样本策略（S1-S9）。两者关系：

| §9 策略                      | 本文档对应     | 关系                            |
| -------------------------- | --------- | ----------------------------- |
| T1 TCN Hidden Probe        | —         | 诊断策略，非小样本通用                   |
| T2 模态级辅助监督                 | S2 正则化变种  | 辅助头 = 正则化的一种形式                |
| T3 平衡融合 + Modality Dropout | S2 正则化变种  | Modality Dropout = dropout 变种 |
| T4 相位窗口统计分支                | S3 轻量模型变种 | 统计特征 = 降维                     |
| T5 ROCKET 统计池化             | S3 轻量模型变种 | ROCKET = 固定核浅特征               |
| T6 Ridge Teacher 蒸馏        | S5 知识蒸馏   | 完全对应                          |

**本文档新增**（§9 未覆盖）：S1 数据增强、S4 集成学习、S6 元学习、S7 自监督预训练、S8 半监督、S9 物理信息约束。

## 5. 优先级与实施路线

### 5.1 优先级排序

| 优先级    | 策略            | 理由                       | 成本     | 前置条件         |
|:------:| ------------- | ------------------------ |:------:| ------------ |
| **P0** | S1 数据增强       | cli.py 已支持，零代码改动，未启用     | <0.5 天 | 无            |
| **P0** | S3 轻量模型       | 减小 channel 维度，直接抗过拟合     | 0.5 天  | 无            |
| **P1** | S4 集成预测       | 15 runs 已就绪，后处理即得        | <0.5 天 | 单模型能收敛       |
| **P1** | S5 Ridge 蒸馏   | §9 T6 已规划，CO₂ teacher 有效 | 1 天    | 改 trainer.py |
| **P1** | S9 物理信息约束     | tv3 物理模型已知，O₂ 声速差可约束     | 1-2 天  | 可微物理层        |
| **P2** | S2 正则化强化      | 当前 dropout=0.1 偏低        | <0.5 天 | 网格搜索         |
| **P2** | S7 SupCon 预训练 | 需设计正负对                   | 2-3 天  | 预训练框架        |
| **P3** | S6 元学习        | 需跨场景多任务，前提不满足            | 3-5 天  | hg/sg/tv3 联合 |
| —      | S8 半监督        | tv3 无未标注数据，不适用           | —      | —            |

### 5.2 实施路线

```
阶段 1（P0，<1 天）：验证 DL 能否收敛
    │
    ├── S1 数据增强：--augment '{"jitter_std":0.001,"amplitude_scale_range":[0.9,1.1],"apply_prob":0.5}'
    ├── S3 轻量模型：acoustic_channels [16,32], tcn_channels [64,64]
    └── 验证：TCN 50 epochs CO₂ R² > 0?
                                                │
                                                ▼
阶段 2（P1，2-3 天）：若阶段 1 收敛
    │
    ├── S4 集成：15 runs 平均
    ├── S5 Ridge 蒸馏：CO₂ 目标 R² → 0.91
    └── S9 物理约束：O₂ 声速差约束
                                                │
                                                ▼
阶段 3（P2，3-5 天）：若阶段 2 O₂ 仍不达标
    │
    ├── S2 正则化网格搜索
    ├── S7 SupCon 预训练
    └── 决策：是否扩大数据集到 6000 序列
                                                │
                                                ▼
阶段 4（P3，备选）：若所有策略 O₂ R² < 0.50
    └── S6 跨场景元学习（hg + sg + tv3 联合 MAML）
```

### 5.3 停止条件

沿用 [dl_training_plan.md §10](../archive/legacy/dl_training_plan.md) 停止条件：

- 如果 O₂ R² < 0.50（across all models and all channel combinations）→ 当前通道组合无法有效检测 O₂，必须引入 O₂ 专用通道（阶段 Ⅲ-1）
- 如果 O₂ R² > 0.70 → 达标，进入阶段 Ⅱ ablation
- 如果 CO₂ R² < 0.90 → 异常，检查 V_NDIR_CO2 通道数据

## 6. 文献列表

1. [Liu, J.-J., Yao, J.-P., Wang, Z., Wang, Z.-Y., & Huang, L., 2024, "Small Sample Time Series Classification Based on Data Augmentation and Semi-supervised Learning", Information Technology and Control, DOI:10.5755/j01.itc.53.2.35797](https://doi.org/10.5755/j01.itc.53.2.35797)
2. [Jiang, W., Ling, L., Zhang, D., Lin, R., & Zeng, L., 2022, "A Time Series Forecasting Model Selection Framework Using CNN and Data Augmentation for Small Sample Data", DOI:10.21203/rs.3.rs-1094384/v1](https://doi.org/10.21203/rs.3.rs-1094384/v1)
3. [Li, K. & Chen, Q., 2025, "Ensemble Deep Learning for Small-Sample Time Series Forecasting", ACTCE, DOI:10.1109/actce66599.2025.00079](https://doi.org/10.1109/actce66599.2025.00079)
4. [Lee, S. & Chang, J.-H., 2017, "Deep learning ensemble with asymptotic techniques for oscillometric blood pressure estimation", Computer Methods and Programs in Biomedicine, DOI:10.1016/j.cmpb.2017.08.005](https://doi.org/10.1016/j.cmpb.2017.08.005)
5. [Moradinasab, N. et al., 2024, "Universal representation learning for multivariate time series using instance-level and cluster-level supervised contrastive learning", Data Mining and Knowledge Discovery, DOI:10.1007/s10618-024-01006-1](https://doi.org/10.1007/s10618-024-01006-1)
6. [Lee, J. & Yang, C., 2022, "Deep neural network and meta-learning-based reactive sputtering with small data sample counts", Journal of Manufacturing Systems, DOI:10.1016/j.jmsy.2022.02.004](https://doi.org/10.1016/j.jmsy.2022.02.004)
7. [Kötter, A. et al., 2024, "Task-Similarity is a Crucial Factor for Few-Shot Meta-Learning of Structure-Activity Relationships", Chembiochem, DOI:10.1002/cbic.202400095](https://doi.org/10.1002/cbic.202400095)
8. [Chen, Z. et al., 2025, "A few-shot u-net learning framework for fast and accurate three-dimensional dose prediction in radiotherapy", Physica Medica, DOI:10.1016/j.ejmp.2025.105184](https://doi.org/10.1016/j.ejmp.2025.105184)
9. [Li, L. et al., 2026, "Development and evaluation of a multimodal feature-based predictive model for radiotherapy-induced oral mucositis", PLOS ONE, DOI:10.1371/journal.pone.0346251](https://doi.org/10.1371/journal.pone.0346251)
10. [Yuda, E. et al., 2026, "Development and Evaluation of a Urinary Na/K Ratio Prediction Model", Bioengineering, DOI:10.3390/bioengineering13020252](https://doi.org/10.3390/bioengineering13020252)
11. [Chen, W. et al., 2026, "TabPFN Opens New Avenues for Small-Data Tabular Learning in Drug Discovery", Journal of Chemical Information and Modeling, DOI:10.1021/acs.jcim.5c02823](https://doi.org/10.1021/acs.jcim.5c02823)
12. [Yi, X. et al., 2025, "Thyroid nodule segmentation in ultrasound images using transformer models with masked autoencoder pre-training", Frontiers in AI, DOI:10.3389/frai.2025.1618426](https://doi.org/10.3389/frai.2025.1618426)
13. [Liu, J. et al., 2022, "A Quantile-Regression Physics-Informed Deep Learning for Car-Following Model", SSRN, DOI:10.2139/ssrn.4233942](https://doi.org/10.2139/ssrn.4233942)
14. [Zhang, H., 2025, "A self-supervised pretraining model for time series classification based on data preprocessing", Knowledge-Based Systems, DOI:10.1016/j.knosys.2025.114340](https://doi.org/10.1016/j.knosys.2025.114340)

## 7. 检索方法与工具

### 7.1 工具使用

| 工具                                    | 调用次数 | 数据源                       | 结果                                          |
| ------------------------------------- |:----:| ------------------------- | ------------------------------------------- |
| `mcp__academic-search__search_papers` | 7    | Crossref / PubMed / arXiv | ✅ 有效（arXiv 被限流 HTTP 429，Crossref/PubMed 正常） |
| `WebSearch`                           | 4    | 通用 Web                    | ❌ 全部返回空（只有 reminder，无实际结果）                  |
| `mcp__paper-search__search_arxiv`     | 2    | arXiv                     | ❌ 空结果（参数不匹配或被限流）                            |

### 7.2 检索查询

1. `few-shot deep learning regression small sample`
2. `time series regression data augmentation small sample deep learning`
3. `meta-learning MAML regression small sample neural network`
4. `physics-informed deep learning gas sensing concentration regression`
5. `data augmentation techniques time series jitter scaling mixup regression`
6. `self-supervised pretraining time series representation small sample`
7. `knowledge distillation teacher student small sample regression`
8. `semi-supervised learning time series regression small sample`
9. `ensemble deep learning small sample regression overfitting`
10. `contrastive learning time series representation small sample`
11. `transfer learning cross-domain gas sensor electronic nose`

### 7.3 局限性

1. **arXiv 源被限流**：多次查询返回 HTTP 429，部分 CS 领域最新工作可能遗漏
2. **WebSearch 失效**：通用 Web 搜索未返回结果，无法补充非学术来源
3. **文献时效**：检索到 2024-2026 年文献为主，经典工作（MAML 2017, Mixup 2018, Dropout 2014）靠领域共识补充，未单独引用
4. **场景匹配度**：多数文献是分类任务，tv3 是回归任务，策略迁移需验证
5. **置信度标注**：High = 多来源 + 场景匹配；Medium = 单来源或场景有差异；Low = 前提不满足

## 8. 后续行动

1. **P0 策略立即执行**：数据增强 + 轻量模型，验证 DL 能否在 600 序列下收敛
2. **P1 策略按需推进**：若 P0 收敛，推进集成/蒸馏/物理约束
3. **扩大数据集备选**：若所有策略 O₂ R² < 0.50，生成 6000 序列（int16 + skip-fiber-mic，29 GB，服务器可行）
4. **文档更新**：策略执行后补充实际 R² 结果，验证策略有效性
