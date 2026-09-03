# AGENTS.md

本目录是多模态掺氢天然气正式实验 v4 重构主线。

## 工作原则

- 中文优先；代码、命令、API 名保持原文。
- 禁止无依据的防御性编程和隐式兜底逻辑。
- 新正式主线不得把 `mixture_id` 回退或重写为 `sequence_id`。
- 新正式 benchmark 不依赖 `base_condition_id`、`noise_seed_index`、`noise_seed`。
- `rcdw_mgda` 子工程遵循上述同一组不变量；其独立 schema_version 为 `rcdw-benchmark-1`，ID 前缀 `RCDW-M/Q`，与主线 `wv4-*` / `sg4-*` 命名空间隔离。
- 文件修改使用 `functions.apply_patch`；命令执行使用当前会话可用 shell 工具，并显式设置 `workdir`。

## 代码语义检索规约

- 需要理解代码行为、模块关系、调用流程、影响范围，或尚不确定相关文件位置时，必须优先调用 `mcp__ace_tool_rs__search_context`；它是本项目代码语义检索的首选入口。
- 调用时，`project_root_path` 使用本项目根目录的绝对路径并采用 `/` 作为分隔符，`query` 使用自然语言描述目标行为，并可附加关键词。
- 已知标识符的全量匹配和精确文本查找仍使用 `rg`；不得用语义检索替代需要完整枚举的查找。
- `search_context` 报错或超时不得伪造成功或静默忽略；保留错误证据，并明确说明改用本地精确检索的原因。

## 提示词与协作规则

- 非简单任务按 `Context / Task / Format` 组织：先交代项目背景、数据与代码边界，再明确要执行的动作、约束和验收方式，最后说明期望输出格式。
- 复杂任务动手前先复述目标、关键约束、影响文件、修复路径和验证方式；只有缺失信息会实质改变结果或带来明显风险时才提问。
- 稳定角色、行为准则、项目不变量放在本文件；一次性示例、临时输出格式和当前任务材料放在用户请求或任务文档中，不写入全局规则。
- 指令、背景材料、约束、示例和正式任务要分区呈现；长规则优先用标题、列表或标签分隔，避免把多类信息混成一段。
- 当格式、分类边界或业务规则难以用文字说清时，用少量高质量示例校准；示例必须结构一致、覆盖关键边界，并与正式任务清楚隔开。
- 多步骤推理、代码调试、实验设计和架构取舍要先建立可验证的不变量与判断路径，再给结论；简单转换、摘要和格式化任务不额外展开推理。
- 输出不符合预期时，优先回到源规则、源提示词或源代码修正，不用连续追加临时补丁制造上下文噪声。
- 多阶段工作可以拆成不同角色视角，但阶段之间只传递必要结论、接口契约和证据，不搬运完整冗余上下文。

## 目标边界

- `src/sim`：数据生成、物理建模、打包、质量检查。
- `src/dl`：深度学习数据读取、模型、训练、评估。
- `src/ml`：传统 ML 特征、训练、评估、融合基线。
- `src/pipeline`：CLI 编排、状态、汇总、图表、报告。
- `configs`：按 `data/model/train/eval/experiment` 拆分正式配置。
- `outputs`：只按 `runs/summary/reports/archive` 管理运行产物。

## 第一阶段约束

第一阶段先固定核心契约，不整仓复制 V3 历史代码。可复用逻辑必须在迁移时显式去除旧主键语义和临时产物命名。

## RecallLoom 写入规约

写入 `.recallloom/` sidecar（rolling_summary、daily_logs、context_brief、update_protocol）必须遵守以下两条，否则 helper 写入会失败，被迫手写会写坏 marker。

### 1. 调用统一带 PYTHONUTF8=1

RecallLoom helper 输出含 emoji（✅ ▶️ 🔥 等），Windows 控制台默认 GBK 编码会崩在 `UnicodeEncodeError`，且 helper 会把这个崩溃误报成 `damaged_sidecar`。所有 RecallLoom 调用前缀 `PYTHONUTF8=1`：

```
PYTHONUTF8=1 python <recallloom>/recallloom.py status . --json
```

### 2. section 正文避免“中文字符紧跟斜杠再接字母数字”

helper 的 attached-text 安全扫描会把这种写法误判成 `absolute_path_dump`（绝对路径转储）而拒绝写入。精确触发条件：一个斜杠，前面是中文字符（或空格、标点），后面紧跟字母数字段。

- 触发误报：中文词后直接用裸斜杠连英文或数字 token（例如把“线性”和“N2”裸斜杠连写）。
- 安全写法：斜杠两侧加空格（`线性 / N2`），或改用顿号 `线性、N2`、连字符 `线性-N2`、或“与”。
- 不受影响：纯字母数字之间的斜杠，如 `H2/CH4/CO2`、`val/test`、`free_component_mse/weighted`——斜杠前是字母或数字，不触发。

只有“中文 + 斜杠 + 字母数字”这一种组合需要规避，其余斜杠写法照常使用。
