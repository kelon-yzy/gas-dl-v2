# B7：OOF Ridge Residual MLP 实施计划

> 状态：代码与 smoke 已通过；待服务器 seed42 预检与三 seed 正式复核
> 日期：2026-07-11
> 上游证据：B1 RawDSP Ridge、RawDSP frame fidelity、B6 三新增 seed stable_pass。

## Context

B6 在冻结 `d0_raw_dsp_physics_stats_v1` 的 1008 个 RawDSP 特征上，三个新增 seed 的 O2 R2 均值为 val/test/extrap `0.5581 / 0.5356 / 0.4835`，已稳定优于 B1 RawDSP Ridge 的 `0.4280 / 0.4786 / 0.3695`。但 B6 的 flat MLP 仍需同时学习线性主趋势和非线性细节；Ridge 已证明前者稳定可学。

B7 要检验的不是“更大网络是否更高”，而是：**固定 Ridge 物理主趋势后，低容量 MLP 是否能稳定学习其未解释残差，并在 test 与当前 extrapolation 上优于 B6。** 当前 extrapolation 仍是 random split 的剩余集，不能写成物理 OOD。

## Task

### 1. 正式不变量

1. 数据固定为 clean `data/tv3-formal-6000` 和现有 `random_mixture_id_split_v4`。
2. 特征固定为 `d0_raw_dsp_physics_stats_v1`、1008 特征、7 个正式 slow channels 与 B1 / B6 相同的 RawDSP cache；不读取 observed physics 数组。
3. 输出固定为三维 `raw3`：`x_CO2`、`x_O2`、`x_N2`。禁止 N2 回填、ILR/ALR、`gas_head`、true TOF/true sound speed 输入和强制闭包损失。
4. RawDSP fidelity、cache provenance 和 B1 reference metrics 必须沿用 B6 的正式证据；任一缺失或 hash 不一致时直接失败。
5. B7 首轮只改变回归头，不加入分组 bottleneck、TabM、特征选择、可微 DSP、独立 heads 或新 split。

### 2. 模型与 OOF 数据流

令训练特征与标签为 `(X_train, Y_train)`。B7 的公开预测为：

\[
\hat Y(X)=\hat Y_{\mathrm{Ridge,full}}(X)+\hat R_{\mathrm{MLP}}(X)
\]

其中 residual MLP 只拟合 Ridge 残差，仍输出三个 raw 值的修正量。

1. 在训练 split 的 `mixture_id` 行级样本上构造固定 5-fold `KFold(shuffle=True, random_state=20260711)`；每个样本必须恰好一次作为 holdout，且折间不得共享行。
2. 每一折仅用其余 4 折 fit `RidgeCV`（包括该折自己的特征标准化与 alpha 选择），预测该折 holdout，拼成 `Y_ridge_oof`。
3. 构造训练残差：`R_train = Y_train - Y_ridge_oof`。不得以 full-train Ridge 的训练内预测代替 OOF 预测。
4. 使用全部 train rows fit 一份 `RidgeCV`，得到 `ridge_full`。它是 val/test/extrap 推理时唯一允许使用的 Ridge。
5. 对 val 生成 `R_val = Y_val - ridge_full(X_val)`；残差 MLP 以 `(X_train, R_train)` 训练、以 `(X_val, R_val)` 早停，但 early stopping 的 O2 R2 必须用组合预测 `ridge_full(X_val) + mlp_residual(X_val)` 对原始 `Y_val` 计算。
6. 对 train/test/extrap 统一输出 `ridge_full(X) + mlp_residual(X)`，再使用既有组分评估器。不得分别报告 Ridge 或 residual 的 R2 来替代组合结果。

### 3. 固定 B7 配方

| 项 | 值 | 说明 |
| --- | --- | --- |
| Ridge | 复用 B1 `RidgeCV` 和 `DEFAULT_RIDGE_ALPHAS` | 每个 OOF 折独立选 alpha；full Ridge 也独立记录 alpha |
| OOF | 5 folds，seed `20260711` | OOF 划分固定，不随 MLP training seed 改变 |
| residual MLP | hidden dims `(64, 64)`，约 68,931 参数 | 相比 B6 的 291,587 参数显著缩小 |
| 输出层 | 零初始化 | 训练开始时 `MLP_residual(X)=0`，组合模型严格等价于 Ridge |
| 输入标准化 | 仅 fit train，再 transform val/test/extrap | 与 B6 一致 |
| 残差目标标准化 | 每目标 `StandardScaler`，仅 fit `R_train` | 预测反变换回原始残差单位 |
| 训练 | dropout `0.1`、AdamW lr `1e-3`、weight decay `1e-4`、batch `256` | 冻结，不做搜索 |
| 早停 | val 组合 O2 R2，max epochs `200`、patience `20` | 不以 residual 自身 R2 早停 |
| 损失 | 残差空间加权 MSE，权重 `[1,2,1]` | 仅改变训练损失尺度，公开输出仍是 raw3 |

### 4. 实施范围

| 文件 | 改动 |
| --- | --- |
| `tv3/ml/ridge_residual_head.py` | 新增 `OofRidgeResidualMlpRegressor`：OOF Ridge、full Ridge、残差 MLP、组合预测与防泄漏审计。 |
| `tv3/ml/mlp_head.py` | 仅抽取可复用的 raw3 MLP 构建、输入/目标缩放和训练循环；增加“输出层零初始化”和“组合预测 early stopping”的显式接口。不得复制第二套 MLP 训练代码。 |
| `tv3/ml/rocket_training.py` | 注册 `oof_ridge_residual_mlp` head，传递 val matrix，写入 B7 diagnostics、RawDSP provenance 和 B1 / B6 对照。 |
| `tv3/pipeline/run_tv3_rocket_baseline.py` | 增加 head、`oof_folds`、`oof_seed` 和 B7 config 的 CLI / 配置白名单；保持旧 head 行为不变。 |
| `configs/tv3_d2b_oof_ridge_residual_mlp.json` | 新建 B7 正式配置；继承 B6 的全部 RawDSP 特征与证据路径，只替换 head、OOF 字段与 `(64,64)`。 |
| `tests/test_tv3_b7_oof_residual.py` | 新增 OOF、防泄漏、零初始化、组合预测、诊断 payload 和 CLI smoke 测试。 |
| `tests/test_tv3_raw_dsp_pipeline.py` | 扩展正式 RawDSP evidence 与 B7 payload 的契约测试。 |
| `scripts/run_r5t_b6_multiseed.py` 或抽取后的通用 helper | 复用既有 JSON 汇总逻辑运行 B7 的 `42/123/456`；不复制另一套 report 判定代码。 |

### 5. 必测的不变量

1. OOF prediction 行数等于训练行数，每行恰好被一个 holdout fold 写入一次。
2. 对每折，fit indices 与 holdout indices 没有交集；fold Ridge 的 scaler 和 alpha 不得接触 holdout 标签。
3. val residual 只能来自 `ridge_full`（其 fit 数据仅为 train）；test/extrap 同理。
4. MLP 输出层初始化后 residual 预测全零，组合预测逐元素等于 `ridge_full.predict(X)`。
5. `standardize_targets=true` 仅标准化 residual 训练损失；最终 `predict` 返回原始百分比的 raw3 组合预测。
6. B7 metrics 必须记录 feature builder/count、OOF fold seed、每折训练/holdout 行数、每折与 full Ridge alpha、OOF coverage、best epoch、parameter count、B1 / B6 metrics path 与 hash。
7. 缺失 fidelity、RawDSP manifest、B1 reference、B6 multiseed report 或其契约不匹配时直接报错；不得自动降级为 B6 或 Ridge。

### 6. 验收与停止条件

正式运行使用 seeds `42/123/456`，OOF seed 固定为 `20260711`。每个 seed 首先必须满足 B6 的 B1 门槛：

| split | B7 单 seed 门槛 |
| --- | ---: |
| val | O2 R2 `>= 0.4780` |
| test | O2 R2 `> 0.4786` |
| extrapolation | O2 R2 `> 0.3695` |

相对 B6 的主判据使用同一 training seed 的 paired difference，并预先固定 B6 三 seed均值为 test `0.5356`、extrapolation `0.4835`：

- **residual_pass**：三个 seed 均通过 B1 门槛；B7 的 test 与 extrapolation 配对均值差均不为负，且至少一个 split 的均值提升 `>= 0.01`；B7 的 test/extrapolation 标准差不得同时高于 B6。
- **noninferior_only**：满足 B1 门槛但未满足 residual_pass。保留 B6 为默认 raw-DSP 头，不扩大 B7 结构。
- **failed**：任一 seed 未通过 B1 门槛，或 test/extrapolation 的配对均值任一低于 B6 超过 `0.01`。停止当前 residual 头调参，保留 B6 与 Ridge。

无论结果如何，只有新 split / 独立 OOD 复核后才能讨论泛化；O2 0.8% bins 仍作为物理上限审计，不设“全部正 R2”目标。

## Format

### 1. 正式输出目录

```text
outputs/tv3_d2b/b7_oof_ridge_residual_mlp/
  s42/metrics.json
  s123/metrics.json
  s456/metrics.json
  summary.json
  replication_report.json
```

输出目录不可覆盖。每个 metrics.json 必须包含原有 evaluations、conditional metrics、`sum_abs_error`、RawDSP provenance / fidelity，以及：

```json
{
  "head": "oof_ridge_residual_mlp",
  "diagnostics": {
    "oof": {
      "fold_count": 5,
      "fold_seed": 20260711,
      "coverage_complete": true,
      "folds": []
    },
    "ridge": {"full_selected_alpha": 0.0},
    "residual_mlp": {"parameter_count": 68931, "best_epoch": 0}
  }
}
```

### 2. 实施阶段

1. 先完成 head 与纯 numpy / smoke 数据单元测试，验证 OOF 语义和组合预测。
2. 扩展 Rocket CLI / config 并完成 RawDSP smoke，验证 evidence tracing 与 payload。
3. 在服务器做一次 seed `42` 的正式预检；只核查产物结构、契约和数值有限性，不据此调参。
4. 通过预检后，执行 `42/123/456` 的冻结正式复核并生成 paired B6 对照报告。
5. 审计完成后，才回填项目记忆库、B6 计划和算法路线图。

### 3. 最小验证命令

```bash
python -m pytest tests/test_tv3_b7_oof_residual.py tests/test_tv3_r5_mlp.py tests/test_tv3_raw_dsp_pipeline.py tests/test_d2b_frame_fidelity_audit.py -q
python -m tv3.pipeline.run_tv3_rocket_baseline --config configs/tv3_d2b_oof_ridge_residual_mlp.json
```

正式服务器命令、输出目录和多 seed 汇总脚本只在代码与 smoke 测试通过后写入实施记录；不以本地旧 600 数据作为性能证据。

## 实施记录

### 2026-07-11 代码与 smoke

已落地：

1. `tv3/ml/mlp_head.py`：公开 `build_raw3_mlp`；新增 `zero_init_output` 与 `early_stop_combine_base`。
2. `tv3/ml/ridge_residual_head.py`：`OofRidgeResidualMlpRegressor`（5-fold OOF Ridge + full Ridge + residual MLP）。
3. `tv3/ml/rocket_training.py` / `run_tv3_rocket_baseline.py`：注册 `oof_ridge_residual_mlp`，强制校验 fidelity / B1 / B6 report。
4. `configs/tv3_d2b_oof_ridge_residual_mlp.json`：B7 正式配置。
5. `scripts/run_b7_oof_residual_multiseed.py`：复用 B6 汇总 helper，输出 `s42/s123/s456` + paired verdict。
6. 最小验证：`tests/test_tv3_b7_oof_residual.py` + RawDSP/B6/R5 相关测试 **36 passed**。

服务器下一步：

```bash
# seed 42 预检（只查产物结构与有限性，不调参）
python -m tv3.pipeline.run_tv3_rocket_baseline \
  --config configs/tv3_d2b_oof_ridge_residual_mlp.json \
  --seed 42 \
  --output-dir outputs/tv3_d2b/b7_oof_ridge_residual_mlp/s42

# 通过后三 seed 正式复核
python scripts/run_b7_oof_residual_multiseed.py
```

注意：若 `s42` 已由预检写入，正式三 seed 编排会拒绝覆盖；预检通过后可直接跑 `123/456`，或换新 output-root 后重跑全套。
