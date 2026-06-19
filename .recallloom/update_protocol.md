<!-- recallloom:file=update_protocol version=1.0 lang=zh-CN -->
<!-- file-state: revision=2 | updated-at=2026-06-19T09:39:23+08:00 | writer-id=ZCode | base-workspace-revision=145 -->

<!-- section: project_specific_overrides -->
# 项目级约束与补充说明

## RecallLoom 写入规约（写入 sidecar 前必读）

### 1. 调用统一带 PYTHONUTF8=1
RecallLoom helper 输出含 emoji，Windows 控制台默认 GBK 编码会崩在 UnicodeEncodeError，并被 helper 误报成 damaged_sidecar。所有 RecallLoom 调用前缀 PYTHONUTF8=1。

### 2. section 正文避免“中文字符紧跟裸斜杠再接字母数字”
helper 的 attached-text 安全扫描会把这种写法误判成 absolute_path_dump 而拒绝写入。
- 精确触发：一个斜杠，前面是中文字符或空格、标点，后面紧跟字母数字段。
- 安全写法：斜杠两侧加空格（写成 线性 / N2），或改用顿号、连字符、或“与”。
- 不受影响：纯字母数字之间的斜杠，如 H2/CH4/CO2、val/test、free_component_mse/weighted，斜杠前是字母或数字，不触发。
只有“中文 + 斜杠 + 字母数字”这一种组合需要规避，其余斜杠照常使用。

## 其他项目级约束
按需补充读取顺序、写入规则、归档规则、证据优先级等更强约束。
