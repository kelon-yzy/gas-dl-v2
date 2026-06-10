# N2 误差改进计划：基于 ILR/ALR 的组成数据回归

## 1. 背景与结论

当前正式实验里，`N2` 是跨 `ML/DL` 的系统性弱项，不是单一模型失效：

- `ridge_all_modalities` 的 test `x_N2 R²` 仅约 `0.22`
- `cnn1d_tcn_fusion` 的 test `x_N2 R²` 约 `-0.0066`
- `dynamic_stacking_svr_all_modalities` 的 test `x_N2 R²` 约 `0.325`

与此同时，`H2/CH4/CO2` 的 `R²` 明显更高，说明问题不在“整体回归完全失败”，而在 `N2` 的表示方式和可辨识性。

结合现有协议结果，`N2` 在 `exposure/recovery` 窗口内可学性显著更强，但在 `steady/full-window` 统计中明显恶化。这表明：

1. `N2` 的信息主要存在于瞬态，而不是全窗口稳态平均。
2. 当前把四组分直接当普通欧式向量做回归，会把 `N2` 继续压缩成“其余组分误差的剩余项”。

本计划的核心判断是：

- 四组分标签本质上是 **compositional data**，样本空间是 simplex，不应直接按普通 `R^4` 目标做 `MSE/MAE`
- 最有价值的第一步不是继续加深模型，而是先把目标空间改成 **log-ratio coordinates**
- 当前项目应优先做两条线：
  - `ALR`：作为低风险、可解释、实现快的对照
  - `ILR`：作为更稳健的正式主线

## 2. 为什么普通回归不适合当前标签

当前标签满足：

- 四个分量非负
- 四个分量和为 `100%`
- 任一分量变化都会强制影响其余分量

这意味着标签不在普通欧式空间，而在一个三维 simplex 中。直接对 `[H2, CH4, CO2, N2]` 做逐维回归会出现几个问题：

1. **闭合约束耦合**
   
   - `N2 = 100 - (H2 + CH4 + CO2)` 的残差天然和前三项绑定
   - 当前 `N2` 很容易退化为 `CH4` 或环境漂移的“误差吸收器”

2. **欧式距离失真**
   
   - 对 composition 来说，真正有意义的是比例关系而不是绝对坐标差
   - 例如 `2% -> 4%` 与 `40% -> 42%` 的统计意义并不相同

3. **原始 MSE 会错误放大大组分、弱化小组分**
   
   - 当前数据设计中 `CH4` 为补足项且不低于 `40%`
   - `N2` 上限约 `20%`，更容易被闭合约束吞掉

## 3. ILR / ALR 是什么

### 3.1 ALR

`ALR`（additive log-ratio）把 `D` 维 composition 转成 `D-1` 个 log-ratio：

```text
alr(x)_i = log(x_i / x_ref)
```

其中 `x_ref` 是选定的参考分量。

优点：

- 公式简单
- 逆变换简单
- 工程落地快
- 适合做第一轮对照实验

缺点：

- 依赖参考分量选择
- 几何上不是 isometric
- 对 reference 噪声更敏感

### 3.2 ILR

`ILR`（isometric log-ratio）把 composition 映射到正交归一的 `R^(D-1)` 坐标系中。

优点：

- 保持 Aitchison 几何
- 各坐标正交，适合直接接普通 `ML/DL`
- 对距离、协方差、正则化更自然

缺点：

- 比 ALR 更难解释
- 需要显式定义 basis

## 4. 对当前项目的具体建议

### 4.1 优先建议：正式主线用 ILR，先做 ALR 对照

建议分两步：

1. **先做 ALR 对照**
   
   - 用最小改动验证“目标空间变换”是否真能改善 `N2`
   - 工程复杂度低，方便快速出结论

2. **再做 ILR 正式版**
   
   - 作为可长期保留的主线实现
   - 避免把模型表现绑定到某个 reference part

### 4.2 ALR 在本项目里的推荐定义

不建议用 `N2` 作 ALR 分母，因为 `N2` 正是当前最不稳定、最弱可观测的组分。

更合理的第一版是用 `CH4` 作为 reference：

```text
z1 = log(H2 / CH4)
z2 = log(CO2 / CH4)
z3 = log(N2 / CH4)
```

原因：

- `CH4` 在当前数据设计中始终较大且不低于 `40%`
- `CH4` 在现有模型中可预测性高于 `N2`
- `z3 = log(N2 / CH4)` 直接把当前最难问题变成“`N2` 相对 `CH4` 的比例建模”

逆变换可写成：

```text
r1 = exp(z1)
r2 = exp(z2)
r3 = exp(z3)

CH4 = 1 / (1 + r1 + r2 + r3)
H2  = r1 * CH4
CO2 = r2 * CH4
N2  = r3 * CH4
```

最终再乘回 `100`。

### 4.3 ILR 在本项目里的推荐 basis

当前目标不是追求“任意正交 basis 都行”，而是希望 **第一个坐标直接刻画 `N2` 与其余三项的对比**。

建议使用面向 `N2` 的 pivot / balance 设计，例如：

```text
u1 = sqrt(3/4) * log( N2 / (H2 * CH4 * CO2)^(1/3) )
u2 = sqrt(2/3) * log( CH4 / (H2 * CO2)^(1/2) )
u3 = sqrt(1/2) * log( H2 / CO2 )
```

解释：

- `u1` 直接对应 `N2` 相对其余三项的 balance
- `u2/u3` 负责解释其余组分内部结构
- 这比“把 `N2` 作为原空间第四个头硬回归”更符合当前问题结构

工程上，推荐把这个 basis 固化为代码中的命名常量，而不是运行时隐式生成。

## 5. 需要先解决的零值问题

这是当前方案成败的前提。

### 5.1 为什么必须单独处理

`log-ratio` 变换要求所有分量严格大于 `0`。但当前数据设计允许：

- `H2 0-30%`
- `CO2 0-15%`
- `N2 0-20%`

因此真实标签中可能出现 `0`，这会使 `ALR/ILR` 不可定义。

### 5.2 当前项目的建议策略

第一阶段不做隐式吞错，显式采用 **multiplicative replacement / floor + closure** 策略：

1. 保留原始标签文件不变
2. 只在“训练前目标变换”时做零值替换
3. 替换后重新 closure 到 `100%`
4. 记录每个 split、每个组分被替换的样本数
5. 把 `epsilon` 写入 run config 与 metrics 元数据

建议初始策略：

```text
eps = 1e-4  (比例空间中即 0.0001，对应 0.01%)
```

更稳妥的策略：

- 用训练集最小正值的一半作为 data-driven floor
- 再对全体样本统一应用

不建议：

- 在 transform 时静默 `clip(min=1e-12)` 却不记录
- 训练和评估使用不同的零值处理规则

## 6. 对当前仓库的具体改造点

### 6.1 建议新增一个统一的 composition 工具层

建议新增一个独立模块，例如：

`src/common/composition.py`

职责只做这几件事：

- `close_to_unit_interval()` / `close_to_100()`
- `replace_zeros_multiplicative()`
- `alr_forward()` / `alr_inverse()`
- `ilr_forward()` / `ilr_inverse()`
- basis 常量定义
- `aitchison_distance()`

这样可避免把相同逻辑散在 `ML/DL/metrics/cli` 多处。

### 6.2 ML 改造点

当前 `ML` 训练直接使用原始 `y`：

- [src/ml/training.py](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/正式实验v4/src/ml/training.py:59)

建议新增一个目标变换层：

1. 训练前：
   
   - `y_raw -> zero handling -> closure -> alr/ilr -> y_transformed`

2. 模型训练：
   
   - Ridge / SVR 都回归 `3` 维 transformed target

3. 推理后：
   
   - `y_hat_transformed -> inverse log-ratio -> y_hat_raw`

4. 指标：
   
   - 继续在原始四组分空间评估现有 `RMSE/MAE/R²`
   - 新增 `Aitchison distance`

建议优先做两个 ML 对照：

- `ridge_alr_ch4`
- `ridge_ilr_n2_first`

如果这两条都不能明显提升 `N2`，再考虑更复杂模型。

### 6.3 DL 改造点

当前 `DL` loss 只支持普通欧式损失：

- [src/dl/training/losses.py](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/正式实验v4/src/dl/training/losses.py:8)

而 `cnn1d_tcn_fusion` 当前 head 已经在做 simplex 输出：

- [src/dl/models/cnn1d_tcn_fusion.py](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/正式实验v4/src/dl/models/cnn1d_tcn_fusion.py:92)

如果切到 `ALR/ILR` 路线，建议不要再让模型直接输出 4 个原始百分比，而是：

1. 网络 head 改为输出 `3` 维 transformed coordinates
2. loss 在 transformed space 计算
3. metrics 在 inverse 后的原始空间计算

这会带来两个直接收益：

- 不再需要显式学一个“闭合约束下的第四维残差”
- `N2` 不再是被动的末位输出，而成为某个可解释的 log-ratio 坐标的一部分

建议 `DL` 先做一个最小切换版本：

- 保留 `cnn1d_tcn_fusion` 编码器不变
- 替换最后 head 为 `3` 维输出
- 新增 `compositional_mse` 或 `ilr_mse`

### 6.4 指标层改造

当前项目指标仍主要是原空间：

- `macro_RMSE`
- `macro_MAE`
- `per-component R²`

这些要保留，因为最终业务解释仍在原始四组分空间。

但新增 compositional 路线后，建议增加：

1. **Aitchison distance**
   
   - 作为 transformed-space 下更自然的整体误差

2. **`N2` conditional metrics**
   
   - 按 `N2` 分箱
   - 按 phase 窗口
   - 按 `CH4` 分箱

3. **zero replacement audit**
   
   - 每个 split 被替换的样本数
   - 每个组分的最大替换幅度

## 7. ILR 和 ALR 在本项目里的取舍建议

### 7.1 什么时候先上 ALR

适合先上 ALR 的条件：

- 想尽快验证方向是否有效
- 想做 `ML` baseline 的第一轮实验
- 想把 `N2` 与 `CH4` 的耦合直接显式化

本项目中，ALR 最适合作为：

- 第一轮低成本验证
- 与现有 `ridge_all_modalities`、`dynamic_stacking_svr` 的快速 A/B

### 7.2 什么时候切到 ILR

一旦 ALR 显示 `N2` 确实改善，下一步应切到 ILR，原因有三点：

1. ILR 更符合 simplex 几何
2. 便于在 `DL` 中用标准优化器和正则化
3. 避免把性能稳定性绑定到单个 reference part

因此推荐策略是：

- **短期验证：ALR-CH4**
- **中期主线：ILR-N2-first basis**

## 8. 分阶段实施计划

### Phase 0：只做分析和基础设施

目标：

- 新增 `src/common/composition.py`
- 加入 `ALR/ILR/closure/zero replacement`
- 补单元测试

验收：

- `forward -> inverse` 数值误差小于阈值
- 变换前后总和守恒
- 零值替换日志完整

### Phase 1：ML 快速验证

目标：

- 在 `ridge` 上实现 `ALR-CH4`
- 在 `ridge` 上实现 `ILR-N2-first`
- 和当前 `ridge_all_modalities` 直接对比

主要判断标准：

- test `x_N2 R²` 是否显著高于 `0.22`
- `H2/CH4/CO2` 是否没有明显退化
- `macro_RMSE` 是否至少不劣化明显

推荐阈值：

- `N2 R²` 提升 `>= 0.10`
- 其余三组分 `R²` 降幅不超过 `0.02`

### Phase 2：DL 最小切换

目标：

- `cnn1d_tcn_fusion` 切到 `3` 维 transformed target
- 保持编码器不变，只替换 head + loss

验收：

- 训练稳定
- inverse 后 `sum_error` 接近 `0`
- test `x_N2 R²` 高于当前 `-0.0066`

### Phase 3：与 phase-aware 方案联动

如果 Phase 1/2 有增益，再与现有发现联动：

- 对 `N2` 单独做 `exposure/recovery` 强化
- 在 `ML` 中增加 phase-specific transformed targets
- 在 `DL` 中加入 phase-aware pooling

因为当前本地结果已经表明：`N2` 在 `exposure/recovery` 本来就更可学。

## 9. 风险与决策点

### 9.1 零值频率过高

如果训练集里真实零值很多，`ALR/ILR` 的零值替换可能带来额外噪声。

对策：

- 先统计每个组分的零值比例
- 若比例高，先做“去掉精确 0 端点”的数据设计实验

### 9.2 ALR reference 选择不当

如果 reference 本身高噪声，会把所有坐标拖坏。

对策：

- 第一版用 `CH4`
- 同时记录训练集 log-reference 的方差
- 不把 `N2` 作为 ALR 分母

### 9.3 transformed-space 改善但原空间指标不升

这说明几何建模虽更合理，但当前传感链路对 `N2` 的真实可观测性仍不足。

对策：

- 保留 transformed-space 作为更合理目标表示
- 同时推进 phase-aware 特征和新模态方案

## 10. 本项目的推荐决策

当前建议不是“直接全面替换主线”，而是：

1. 先在 `ML` 上做 `ALR-CH4` 和 `ILR-N2-first`
2. 只要 `N2` 有明确改善，就把 `ILR` 推进到 `DL`
3. `DL` 切换后，再和 phase-aware pooling 联动

优先级排序：

1. `Ridge + ALR-CH4`
2. `Ridge + ILR-N2-first`
3. `CNN1D-TCN + ILR target head`
4. `ILR + phase-aware N2 branch`

当前工程落地状态：

- `src/common/composition.py` 已提供 closure、zero replacement、ALR/ILR 正逆变换和 Aitchison distance。
- `ALR/ILR` 逆变换已使用稳定化归一化，避免极端 transformed coordinate 在服务器正式训练早期触发指数溢出。
- zero replacement 已支持固定 `epsilon` 与 `train_min_positive_half` 数据驱动策略；正式默认 `ALR/ILR` run 使用训练集最小正值一半作为统一 floor。
- zero replacement 审计允许空二维 eval split 生成零行审计，避免正式数据变体中空 `extrapolation` 阶段阻塞目标变换元数据记录；一维空输入仍会显式报错。
- `ML` 路线已支持 `target_transform=alr_ch4` 与 `target_transform=ilr_n2_first`，并在原始四组分空间输出指标。
- `DL` 路线已支持 `cnn1d_tcn_fusion` 在 `target_transform=ilr_n2_first` 下输出 3 维 transformed coordinates，并在评估阶段 inverse 回四组分空间。
- `DL` loss 注册表已提供 `compositional_mse` 与 `ilr_mse` 语义别名；`ilr_mse` 会被校验为只能搭配 `target_transform=ilr_n2_first`，正式默认 `cnn1d_tcn_fusion_ilr` 使用 `ilr_mse`。
- `ML/DL` 的 transformed target 路线都会在 `metrics.json` 中记录 split 级 zero replacement audit。
- `ML/DL` 的评估结果都会记录 `n2_bins` 与 `ch4_bins` conditional metrics，用于检查 `N2` 低/高浓度区间和 `CH4` reference 区间内的误差变化。
- `ML/DL` 的 run config 额外记录 `resolved_target_transform`，用于追踪 data-driven 策略解析后的实际 `epsilon`。
- `formal_full` 默认实验计划已加入 `ridge_alr_ch4_all_modalities`、`ridge_ilr_n2_first_all_modalities` 和 `cnn1d_tcn_fusion_ilr`。
- `pipeline.inspect_composition_labels` 已提供正式实验前置标签审计工具，用于统计 split 级零值比例、最小正值、推荐 `epsilon` 和 `CH4` ALR reference 的 log 方差。
- `pipeline.analyze_n2_improvement` 已提供正式结果验收工具，用于比较 baseline 与 ALR/ILR 方案的 `x_N2 R²`、macro RMSE、其他组分退化和 Aitchison mean；当 metrics 含协议窗口结果时，也会同步输出 `per_phase` 与 `early` 窗口的 `N2` 增益；当 metrics 含 `n2_bins/ch4_bins` 时，也会输出 full-window 与协议窗口内的分箱 `N2` 增益。full/window/bin 层级都会按同一阈值输出 pass 标记，用于判断是否进入 phase-aware 支线。
- 实际收益仍需在 `data/wv4-formal-hitran-standard-6000` 上运行后，以 `test x_N2 R²`、`macro_RMSE`、其他三组分退化幅度和 `Aitchison distance` 判断。

正式运行顺序建议：

```bash
python -m pipeline.inspect_composition_labels --dataset-dir data/wv4-formal-hitran-standard-6000 --output-path outputs/reports/formal_full_composition_labels.md --json-output-path outputs/reports/formal_full_composition_labels.json
python -m pipeline.run_experiment --config configs/experiment/formal_full.json --dry-run
python -m pipeline.run_experiment --config configs/experiment/formal_full.json
python -m pipeline.analyze_n2_improvement --run-root outputs/runs/formal_full --output-path outputs/reports/formal_full_n2_improvement.md --json-output-path outputs/reports/formal_full_n2_improvement.json
```

也可先用 workflow 入口生成同一组服务器命令，确认无误后再显式执行。该入口默认读取 `configs/experiment/formal_full.json` 中的 `dataset_dir`、`output_root`、`device` 和 `experiment_name`，避免手工维护第二套运行路径；需要时再用同名参数覆盖：

```bash
python -m pipeline.run_n2_improvement_workflow --validate-only --json
python -m pipeline.run_n2_improvement_workflow --execute
```

`--validate-only --json` 输出包含 `artifacts` 字段，可直接读取 `composition_label_report/json`、`n2_improvement_report/json` 和 `run_root`，不需要从命令数组反向解析路径。

若服务器脚本需要解析 workflow 的 JSON 结果，可使用 `--execute --json`；workflow 自身日志和子命令输出会写入 `stderr`，`stdout` 保持为最终 JSON。

## 10.1 关于“双主线版本”的当前边界

这里的“双主线版本”指：

- `ILR/ALR` 目标空间改造
- `phase-aware N2 modeling` 瞬态优先建模

当前决定是：

- 将“双主线版本”**保留为下一步计划**
- 不纳入本轮立即实施范围
- 先完成 `ILR/ALR` 的目标空间验证，再决定是否推进 `phase-aware` 支线

这样做的原因是：

1. 先隔离变量，避免同时改 `target space` 和 `input/pooling` 后无法判断增益来源
2. 当前最需要先验证的是：仅靠组成数据回归改造，是否已经能显著改善 `N2`
3. 若 `ILR/ALR` 单独收益有限，再把“双主线版本”作为下一步正式议题展开

因此，后续讨论顺序固定为：

1. 先评估 `ALR/ILR` 单独方案
2. 再决定是否进入 `ILR/ALR + phase-aware N2 modeling` 的联合方案讨论

## 11. 参考资料

### 基础理论

1. Aitchison, J. (1982). *The Statistical Analysis of Compositional Data*. Journal of the Royal Statistical Society. Series B, 44(2), 139-160.  
   DOI: [10.1111/j.2517-6161.1982.tb01195.x](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x)

2. Aitchison, J. (1986). *The Statistical Analysis of Compositional Data*.  
   Chapter: [Logratio analysis of compositions](https://doi.org/10.1007/978-94-009-4109-0_7)

3. Egozcue, J. J., Pawlowsky-Glahn, V., Mateu-Figueras, G., & Barceló-Vidal, C. (2003). *Isometric Logratio Transformations for Compositional Data Analysis*.  
   DOI: [10.1023/A:1023818214614](https://doi.org/10.1023/A:1023818214614)

### 实践与变换取舍

4. Greenacre, M. (2018). *Compositional Data Analysis in Practice*.  
   Book DOI: [10.1201/9780429455537](https://doi.org/10.1201/9780429455537)

5. Greenacre, M., Martínez-Álvaro, M., & Blasco, A. (2021). *Compositional Data Analysis of Microbiome and Any-Omics Datasets: A Validation of the Additive Logratio Transformation*.  
   DOI: [10.3389/fmicb.2021.727398](https://doi.org/10.3389/fmicb.2021.727398)

6. Wang, Z., Shi, W., Zhou, W., Li, X., & Yue, T. (2020). *Comparison of additive and isometric log-ratio transformations combined with machine learning and regression kriging models for mapping soil particle size fractions*.  
   DOI: [10.1016/j.geoderma.2020.114214](https://doi.org/10.1016/j.geoderma.2020.114214)

### 零值处理

7. Martín-Fernández, J. A., Hron, K., Templ, M., Filzmoser, P., & Palarea-Albaladejo, J. (2012). *Model-based replacement of rounded zeros in compositional data: Classical and robust approaches*.  
   DOI: [10.1016/j.csda.2012.02.012](https://doi.org/10.1016/j.csda.2012.02.012)

8. Lubbe, S., Filzmoser, P., & Templ, M. (2021). *Comparison of zero replacement strategies for compositional data with large numbers of zeros*.  
   DOI: [10.1016/j.chemolab.2021.104248](https://doi.org/10.1016/j.chemolab.2021.104248)

### 当前项目相关的本地依据

- [src/dl/training/losses.py](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/正式实验v4/src/dl/training/losses.py:8)
- [src/ml/training.py](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/正式实验v4/src/ml/training.py:59)
- [src/dl/models/cnn1d_tcn_fusion.py](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/正式实验v4/src/dl/models/cnn1d_tcn_fusion.py:92)
- [docs/生成正式 HITRAN 标准数据集计划.md](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/正式实验v4/docs/生成正式 HITRAN 标准数据集计划.md:54)
- [docs/整理归档/02_模型架构总览_part2.md](D:/mydate/项目资料__多模态掺氢天然气/04_代码与实验/code/正式实验v4/docs/整理归档/02_模型架构总览_part2.md:105)
