# R7 ExtraTrees 观测特征回归实施计划

> 状态：正式 tv3-formal-6000 已完成（2026-07-10），未通过部署候选判据。
> 文献依据：[observed_o2_algorithm_review.md](../../references/observed_o2_algorithm_review.md)；[掘进通风项目记忆库.md](../../掘进通风项目记忆库.md) §5.4、§6.4、§6.8、§6.9、§7.1。
> 定位：TDLAS O₂ 硬件暂缓期间的可部署非线性回归探针，不宣称突破窄 O₂ 分箱物理极限。

## 目标

在 D0-observed 的 864 维固定特征上使用 ExtraTrees，检验树模型能否利用 observed TOF、estimated sound speed 与 `T_C/P_MPa/H_RH/L_m` 的联合信息获得相对线性基线的非线性增益，并在 val/test/extrapolation 同时超过 D0-observed Ridge。

## 不变量

1. 特征必须与 `configs/tv3_d0_observed_ridge.json` 逐位一致：仅 observed TOF、peak index、estimated sound speed、TOF quality、TOF accepted 与 slow 通道统计量。
2. 不使用 `ultrasonic_tof_s`、真值声速、真值衰减或任何 oracle 数组。
3. 直接输出 `x_CO2/x_O2/x_N2` 三列原始百分比；不使用 ILR/ALR、闭包损失、N₂ 残差或预测后回填。
4. 不重启 raw waveform、MiniRocket raw 或 TOF-PhaseNet 路线。

## 实现

- 回归器：[tv3/ml/extratrees_head.py](../../../tv3/ml/extratrees_head.py)，`ExtraTreesRegressor` 多输出直接回归 raw3。
- 训练入口：[tv3/ml/extratrees_training.py](../../../tv3/ml/extratrees_training.py)。
- 服务器命令入口：[tv3/pipeline/run_tv3_extratrees_baseline.py](../../../tv3/pipeline/run_tv3_extratrees_baseline.py)。
- 正式配置：[tv3_r7_extratrees_observed.json](../../../configs/tv3_r7_extratrees_observed.json)。
- 本地测试：[test_tv3_r7_extratrees.py](../../../tests/test_tv3_r7_extratrees.py)，已通过 4 项；覆盖 864 维契约、oracle 拒绝、正式 JSON 对齐和 CLI 产物。

默认超参为 600 棵树、`max_features=0.7`、`min_samples_leaf=2`、无深度上限、全核并行。`min_samples_leaf=2` 用于降低单样本叶子对 OOD split 的脆弱性；正式结果必须验证，而不是把该选择当成性能保证。

## 正式执行

```bash
python -s -m tv3.pipeline.run_tv3_extratrees_baseline --config configs/tv3_r7_extratrees_observed.json
```

产物：`outputs/tv3_r7/extratrees_observed/metrics.json`。它保持 frozen D0-observed 864 维契约，使用 600 棵树、`max_features=0.7`、`min_samples_leaf=2` 与 seed `20260704`。

## 验收与分支

| 结果 | 判断 | 后续 |
|---|---|---|
| val/test/extrap O₂ 均超过 D0，且 val 至少 0.4726 | R7 通过 | 将 impurity feature importance 仅作特征线索，不据此确认交互；随后做种子稳定性 |
| 仅 val 提升 | 不通过 | 视为泛化不足，不作部署候选 |
| 三个 eval split 均未到 D0+0.05 | 不通过 | 进入显式环境条件化特征消融；不回到 raw 波形 |
| O₂ 提升但 `sum_abs_error` 明显恶化 | 需单列风险 | 保留 raw3 结果，不做闭包回填，评估部署校准方案 |

R7 无论结果如何，都不改变“窄 O₂ 分箱在 oracle 下仍不可精细辨识”的项目结论。

## 正式结果

| split | train | val | test | extrapolation |
|---|---:|---:|---:|---:|
| O₂ R² | 0.9969 | 0.4516 | 0.4473 | 0.4211 |
| O₂ MAE | 0.0354 | 0.5699 | 0.5814 | 0.5580 |

相对 D0-observed Ridge，R7 的 O₂ R² 在 val 为 `+0.0290`、test 为 `-0.0097`、extrapolation 为 `+0.0503`。val 未达到 `0.4726`，且 test 低于 D0；train-val gap 为 `0.5453`，表明当前无深度上限的树配置主要记忆训练集局部组合。`sum_abs_error` 在三个评估 split 均近零，因此失败原因是 O₂ 泛化不足而非闭包退化。

结论：R7 未通过，不做部署候选，也不在该配置上追加种子稳定性或调参补丁。后续若重启 observed-space 树模型，应作为显式环境条件化或特征消融实验，且与当前结果分开登记。
