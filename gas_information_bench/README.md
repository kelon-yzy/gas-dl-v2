# gas_information_bench

P2 专用 bench 子工程，当前已完成工程身份、前向审计、数据契约、来源接口与
append-only freeze 契约。

- Python 包：`gib`
- distribution：`gas-information-bench`
- schema version：`gib-benchmark-1`
- `mixture_id` 前缀：`GIB-M`
- `sequence_id` 前缀：`GIB-Q`
- CLI 前缀：`gib`
- 输出根目录：`outputs/`

当前包含纯前向审计、S1 信息量 × Jacobian 夹角 3 × 3 刻度、S2/S3 证据、
S4 指标协议、S5 来源接口和 S6 冻结纪律。尚未实现的 P3 能力在所有者 registry
中显式标记为 `reserved`，不视为可运行实现；本子工程仍不包含 pilot 数据或训练代码。

## 独立安装与验证

```powershell
pip install .[dev]
python -m pytest -q
gib --help
```

`configs/` 是唯一配置事实源，并作为 package data 随普通 wheel 安装；契约读取不依赖
editable source path。

`gib freeze` 将一个 `status=complete` 的 attempt 提升到新的 freeze 目录；
`gib verify-freeze <freeze_dir>` 会重算 manifest 中的所有 SHA256。输出布局与输入
角色见 [`outputs/README.md`](outputs/README.md)，子工程文档入口见
[`docs/README.md`](docs/README.md)。

完整契约以 [`docs/p2/P2_bench规格书.md`](../docs/p2/P2_bench规格书.md) 为准。
