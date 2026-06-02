# Karpathy Guidelines 代码审查报告

> **审查范围**：`src/ml/`、`src/dl/`、`src/sim/` 核心模块
> **审查原则**：[Karpathy Guidelines](https://github.com/multica-ai/andrej-karpathy-skills) 四项行为准则
> **审查日期**：2026-06-02
> **二次核实**：2026-06-02（对照真实代码逐条复查后修订，详见下方「修订记录」）

---

## 修订记录（二次核实）

> 第二轮以「对照真实代码逐条核实」为标准复查了全部发现，修订如下：

- **3.3 撤销** — 129 个改动文件忽略空白后语义增删为 **0**，整片 diff 纯属 CRLF 行尾符变更，不存在功能或测试逻辑改动；原「功能/测试混合、无法区分」的论据不成立，并入 3.1。
- **2.5 药方修正** — `input_format`/`out_dim` 实际被 `dl/data/dataset.py` 与测试消费，并非死抽象；其过薄是 Trainer 尚未落地（4.2）的症状。改为「保留并暂缓」，撤销原「用 nn.Module 直接替代」的建议。
- **4.1 计数订正** — ML 测试为 **9** 个用例（原写 8）。
- **1.1 漏报补充** — `_apply_scaler`/`apply_scaler` 同样逐行重复；ML 侧自定义 epsilon、DL 侧从 `sim.packaging.constants` 导入，epsilon 存在两个真相源。
- **3.2 漏报补充** — `_resolve_split_indices`/`resolve_split_indices` 同样重复，split 工具三件套整体重复。
- **3.1 实证坐实** — 方向确认为「工作区被污染」（HEAD=LF → 工作区 CRLF），根因 `core.autocrlf` 未设、仓库无 `.gitattributes`。

---

## 总览

| 原则 | 整体评分 | 主要问题 |
|------|----------|----------|
| Think Before Coding | ⚠️ 一般 | scaler/split 逻辑分叉，dtype 隐式转换无注释 |
| **Simplicity First** |   **最差** | **代码重复是最严重的违规**：metrics 重复、head 重复、split 加载重复 |
| Surgical Changes | ⚠️ 一般 | 工作区 129 文件纯 CRLF 行尾符污染（零功能改动），仓库缺 .gitattributes |
| Goal-Driven |   较差 | DL 无 Trainer、ML 无 CLI——验证通道断裂 |

---

## 1. Think Before Coding（编码前思考）

> "不假设、不隐藏困惑、展示权衡"

| # | 发现 | 严重程度 | 文件 | 说明 |
|---|------|----------|------|------|
| 1.1 | **scaler 加载逻辑分叉** |   中 | `ml/features.py` L260-267 vs `dl/data/scalers.py` L11-25 | 同一功能两个独立实现。如果一方改了校验逻辑（如新增 scaler method），另一方不会感知。当初做了"ML 和 DL 独立演化"的隐式假设，但这个假设既未文档化、也未评估维护成本。**【核实补充】**`_apply_scaler`/`apply_scaler` 亦逐行重复；且 ML 侧自定义 `_Z_SCORE_STD_EPSILON`、DL 侧从 `sim.packaging.constants` 导入，epsilon 双真相源。 |
| 1.2 | **dtype 隐式转换链** |   中 | `features.py` → `models.py` → `RidgeRegressor` | float32 加载 → `_as_2d_features` 转为 float64 → 闭形式求解 → predict 输出 float32。三次 dtype 变换，无一处注释说明"为什么用 float64"（数值稳定性？精度？）。 |
| 1.3 | **`_as_2d_array` 语义重复命名** |   低 | `ml/metrics.py` L62-70 vs `ml/models.py` L124-141 | `_as_2d_array`、`_as_2d_features`、`_as_2d_targets` 是同一函数的不同变体，分散在两个文件中。暗示当初不确定 helper 归属——于是两边都放了。 |

---

## 2. Simplicity First（简洁优先）

> "最少代码解决问题，不投机"

| # | 发现 | 严重程度 | 文件 | 说明 |
|---|------|----------|------|------|
| 2.1 | **CNN1D / TCN head 完全重复** |   **高** | `cnn1d.py` L42-50 vs `tcn.py` L83-91 | MLP head（Linear128→ReLU→Drop→Linear64→ReLU→Drop→Linear4）一模一样，逐行复制。这是 Karpathy 反模式的教科书案例——要么提取为共用组件，要么说明两个模型的不同 head 设计。 |
| 2.2 | **ML / DL metrics 完全重复** |   **高** | `ml/metrics.py` vs `dl/training/metrics.py` | `RegressionMetrics`、`regression_metrics`、`component_regression_metrics` 是两个几乎相同的实现，差异仅在于 NumPy vs PyTorch 张量操作。若要新增指标（如 MAAPE、MAPE），需修改两个文件。这是**"100 行能搞定的写了 200 行"**的精确案例。 |
| 2.3 | **`_validate_split_rows` 重复** |   中 | `ml/features.py` L252-257 vs `dl/data/splits.py` L61-67 | 相同的函数，两个位置。ML 模块拒绝依赖 DL 模块，所以复制了一份——但这违反了"不为一次性代码建抽象"的原则。 |
| 2.4 | **`RegressorProtocol` 过度抽象** |   低 | `ml/training.py` L14-19 | 仅有两个回归器（Mean/Ridge）的情况下定义 Protocol 仅用于类型注解。运行时不被强制执行，增加了概念复杂度。等有 5+ 个回归器时再引入才有意义。 |
| 2.5 | **`BaseRegressor` 基类过薄** |   低 | `dl/models/base.py` L7-21 | 整个类只有 `input_format`/`out_dim` 和未实现的 `forward`。它没提供任何共享逻辑（forward hook、权重初始化、参数统计），更像是接口标记。**【核实修正】**但 `input_format`/`out_dim` 已被 `dl/data/dataset.py`（驱动 NTC/NCT 布局）与测试消费，并非死代码；其单薄实为 Trainer 缺失（4.2）的症状——Trainer 落地后需靠此契约对齐布局与模型声明。**建议保留并暂缓，勿删。** |

---

## 3. Surgical Changes（精准修改）

> "只动必须改的，只清理自己造成的混乱"

| # | 发现 | 严重程度 | 说明 |
|---|------|----------|------|
| 3.1 | **工作区大规模 CRLF 行尾符污染** |   **高** | `git diff` 显示 **129 个文件、14127+ / 14127- 行**。**【核实坐实】**忽略空白后语义增删=**0**，确为纯 LF→CRLF 行尾符变更（HEAD=LF、工作区被污染成 CRLF），零功能改动；根因 `core.autocrlf` 未设、无 `.gitattributes`，应丢弃误改并新增 `.gitattributes`。虽然似乎是格式化操作（无功能变化），但若与功能改动混合提交，完全违反了"只动必须改的"原则。格式化应该作为独立 commit。 |
| 3.2 | **ML 和 DL 各自维护 split 加载** |   中 | `ml/features.py:_load_splits` 和 `dl/data/splits.py:load_splits` 是两个独立实现。ML 模块创建时决定不依赖 DL，于是把所有工具函数重写了一份。方便自己，但对后续维护者是负担。**【核实补充】**`_resolve_split_indices`/`resolve_split_indices` 亦重复——split 工具实为 load/resolve/validate 三件套整体重复。 |
| 3.3 | ~~**测试和功能代码同步修改**~~ ❌ 已撤销（见修订记录，并入 3.1） |   中 | 当前工作区同时修改了核心代码和所有测试文件，无法区分"功能改动"和"测试适配"。违反了 Goal-Driven 的"每次提交一个可验证的步骤"。 |

---

## 4. Goal-Driven Execution（目标驱动执行）

> "定义成功标准，循环直到验证通过"

| # | 发现 | 严重程度 | 说明 |
|---|------|----------|------|
| 4.1 | **ML 有测试但从未在真实 benchmark 上运行** |   **高** | 9 个测试用例（核实订正：原写 8）只在 pytest `tmp_path` 的临时烟雾数据上运行。`train_regressor_on_dataset` 可以返回完整的 `MLTrainingResult`，但没有任何 CLI 入口让它跑在实际的 `data/wv4-smoke/` 上。验证闭环在测试环境完成了，但在真实数据上从未执行。 |
| 4.2 | **DL 有 Dataset + 模型 + loss/metrics，但没有 Trainer** |   **高** | `src/dl/training/losses.py` 和 `metrics.py` 已就绪，但训练器（trainer.py）不存在。DL 训练路径的验证通道断裂——可以构造模型和前向，但不能做完整的 fit→eval→checkpoint 循环。 |
| 4.3 | **缺失 CLI 入口** |   中 | ML 模块只能通过 Python API 调用，DL 训练完全无法运行。`configs/train/` 和 `configs/experiment/` 均为空 `.gitkeep`。 |

---

## 代码优点（值得保留的实践）

-   模块间边界清晰（sim / ml / dl / pipeline），职责分离合理
-   文档注释充分，函数签名完整，类型注解覆盖率高
-   纯 NumPy 实现传统 ML，刻意避免依赖膨胀——**Simplicity First 的正面案例**
-   frozen dataclass 用于数据容器（`MLFeatureMatrix`、`SplitEvaluation`）是正确的选择
-   `__all__` 在每个 `__init__.py` 中显式声明，API 边界干净
-   功能代码和测试代码都使用了 `from __future__ import annotations`，延迟注解一致

---

## 修复优先级

### P0 — 消除核心重复（Simplicity First）

| 行动 | 涉及文件 | 预期行数变化 |
|------|----------|-------------|
| 统一 ML/DL metrics：提取共用 `RegressionMetrics` 数据结构，NumPy/Torch 计算层分离 | `ml/metrics.py`, `dl/training/metrics.py` | ~-70 行 |
| 提取共用 MLP head：CNN1D 和 TCN 共享一个 `build_regression_head(out_dim, dropout)` | `cnn1d.py`, `tcn.py` | ~-20 行 |
| 统一 scaler 加载/应用：提取 `load_scaler`+`apply_scaler`（含 epsilon）到 `src/common/scalers.py`，ML/DL 共用 | `ml/features.py`, `dl/data/scalers.py` | ~-40 行 |
| 统一 split 工具：提取 `load_splits`/`resolve_split_indices`/`_validate_split_rows` 到 `src/common/splits.py` | `ml/features.py`, `dl/data/splits.py` | ~-40 行 |

### P1 — 补齐验证通道（Goal-Driven）

| 行动 | 说明 |
|------|------|
| 为 `train_regressor_on_dataset` 添加 Minimal CLI 入口 | 使用 argparse，只需 `--dataset-dir` 和 `--model` 两个参数 |
| 在 `data/wv4-smoke/` 上跑通 `ridge` 和 `mean` 基线，记录指标 | 验证 ML 模块在实际数据上的可用性 |
| 丢弃 129 文件 CRLF 误改并新增 `.gitattributes`（`* text=auto eol=lf`） | 根因是 autocrlf 未设、无 .gitattributes；仅独立 commit 无法防复发 |

### P2 — 合理简化（Think Before Coding）

| 行动 | 说明 |
|------|------|
| 为 dtype 转换链路添加注释说明 | 在 `_as_2d_features` 和 `_fit_transform_x` 处标注 float64 用途 |
| 评估是否移除非功能性 `RegressorProtocol` | 如果只是为了类型注解，考虑直接用 `MeanRegressor | RidgeRegressor` |
| `BaseRegressor` 保留并暂缓，勿删 | `input_format`/`out_dim` 已被 dataset 与测试消费；待 Trainer 落地后由其消费该契约 |

---

## 附录：违规计数

| 原则 | 高 | 中 | 低 | 合计 |
|------|----|----|----|------|
| Think Before Coding | 0 | 2 | 1 | 3 |
| Simplicity First | 2 | 1 | 2 | 5 |
| Surgical Changes | 1 | 1 | 0 | 2 |
| Goal-Driven Execution | 2 | 1 | 0 | 3 |
| **合计** | **5** | **5** | **3** | **13** |
