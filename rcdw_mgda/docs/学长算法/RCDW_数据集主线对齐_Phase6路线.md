# RCDW 数据集主线对齐 — Phase 6 后续路线

## 0. 文档定位

- **版本**：v1.7（Phase 6E pressure_drift 实验闭环后更新)
- **日期**：2026-07-01
- **前置状态**：Phase 1-5 已落地，`schema_version="rcdw-benchmark-1"`，12 维 benchmark 数据契约已启用。
- **最新进展**：Phase 6A、6B、6C、6D 已完成；Phase 6E 的 `pressure_drift` 已完成代码、配置、测试与 perturb 实验闭环，`runs/phase6e_pressure/perturb/` 生成 12 张 PNG；代码测试基线为 222 passed。
- **相关文档**：
  - [RCDW_数据集主线对齐改动方案.md](RCDW_数据集主线对齐改动方案.md)
  - [RCDW_数据集主线对齐_完成情况.md](RCDW_数据集主线对齐_完成情况.md)
  - [RCDW_实施完成情况.md](RCDW_实施完成情况.md)

本文档只规划 Phase 6 的后续路线，不改变 Phase 1-5 已固定的数据契约。除非明确进入破坏性升级，否则 RCDW-MGDA 继续使用独立 schema `rcdw-benchmark-1`、ID 前缀 `RCDW-M/Q`，并保持与主线 `wv4-*` / `sg4-*` 命名空间隔离。

---

## 1. 当前基线

### 1.1 已完成

- Phase 1-5 已完成：schema、conditions、phases、gas_state、声学 / 光学 / waveform 物理栈、slow、benchmark 落盘、validation、Dataset、12 维模型通道、Stage A/B、扰动评测全部贯通。
- empirical smoke 已能完成 generate -> train -> eval -> perturb，并生成 10 张扰动图。
- `BenchmarkDataset` 从磁盘读取 slow 7 维 + ultrasonic 元数据 5 维，训练侧标签从百分比转换为 `[0,1]` 闭包。
- validation 已覆盖 legacy 字段黑名单、组分和、split、manifest、scaler passthrough 标记等核心不变量。
- P2 修复已完成：validation summary 会写出并校验 scaler 字段；窗口长度小于 2 时在 Dataset 初始化阶段直接拒收。
- 2026-07-01 代码质量审查修复已完成：12 维 input scaler 已接入 `BenchmarkDataset`，默认跟随 `manifest.input_normalization.applied`；训练入口已播种；`hard_suppress` 改为逐样本判定；环境通道加入测量噪声；死配置节已清理。
- Phase 6A 已完成：`--precompute-cache-only` 已实现；真实 HAPI 预热生成 128 个 HITRAN cache `.npz`；定向测试 23 passed，全量测试 216 passed。
- Phase 6B 64 sequence HITRAN smoke 已闭环：`data/rcdw-hitran-smoke-64` validation pass，Stage B test overall MAE=0.06380 / RMSE=0.08511，扰动评测生成 10 张 PNG。
- Phase 6B 128 sequence HITRAN smoke 已闭环：`data/rcdw-hitran-smoke-128` validation pass，Stage B test overall MAE=0.05741 / RMSE=0.06790，扰动评测生成 10 张 PNG。
- Phase 6C 已完成：benchmark 生成支持可选并行，`num_workers=1` 默认保持单进程；empirical 与 HITRAN 合成 cache parity 测试通过。
- Phase 6D 已完成：128-seq HITRAN smoke 上 scaler-on 明显优于 scaler-off（overall MAE 0.05741 vs 0.07642；RMSE 0.06790 vs 0.09101），正式路径保留 12 维 input scaler。
- Phase 6E 的 `pressure_drift` 已闭环：`PERTURBATION_KINDS` 已包含 `pressure_drift`，`configs/phase6e-hitran-smoke-128-pressure.yaml` 已加入 6 类扰动配置，测试覆盖 `test_pressure_drift_targets_p_mpa`；`scripts.perturb` 已生成 6 类扰动图与趋势记录。
- 当前 RCDW 全量测试：222 passed。

### 1.2 仍需进入 Phase 6 的问题

| 优先级 | 问题 | 当前影响 |
|--------|------|----------|
| P2 | 扰动实验闭环仍有一项语义待定 | `pressure_drift` 已完成代码/配置/测试/实验闭环；`h2o_cross` 尚未实现，需先定义输入扰动或后端重生成语义 |
| P3 | O2 弛豫参数仍是 placeholder | 物理参数需要文献或实测校核 |
| P3 | 多 `stage_profile` 未启用 | 需要 condition 1:N、split 分组和 schedule 注册表同步扩展 |

---

## 2. Phase 6 总路线

| 子阶段 | 优先级 | 目标 | 验收门槛 |
|--------|--------|------|----------|
| 6A | P0 | HITRAN cache 预热命令与默认 smoke 可用性 | **已完成**：空 cache 预热成功，生成 128 个 HITRAN cache `.npz`；`hitran_hapi_v1` smoke generate 成功 |
| 6B | P1 | HITRAN smoke 训练、评测、扰动证据链 | **已完成**：64 与 128 sequence 均已完成 generate -> train -> eval -> perturb，并记录指标与图 |
| 6C | P1 | benchmark 并行生成 | **已完成**：`num_workers/chunk_size` 可选并行路径已实现；单进程与并行结果在 empirical 与 HITRAN 合成 cache 测试中确定性等价 |
| 6D | P2 | 12 维 input scaler ablation 与证据收口 | **已完成**：scaler-on 明显优于 scaler-off；manifest 标志位生效；旧 ckpt 不可混用 |
| 6E | P2 | 扰动评测扩展 | **pressure_drift 已完成**：代码/配置/测试/6 类 perturb 图与趋势说明均已同步；`h2o_cross` 是否加入仍需语义决策 |
| 6F | P3 | O2 弛豫参数校核 | placeholder 参数替换为可追溯来源或实测流程 |
| 6G | P3 | 多 `stage_profile` 激活 | `standard_exposure` 之外 profile 有明确 schema 与 split 证据 |

建议执行顺序：**6E 剩余语义决策 -> 6F -> 6G**。其中 6A、6B、6C、6D 已完成；6E 的 `pressure_drift` 已闭环，下一步只需决定 `h2o_cross` 是否以输入扰动或光学后端重生成方式进入正式扰动集。

---

## 3. Phase 6A：HITRAN cache 预热与默认 smoke 可用

### 3.1 目标

让默认配置中 `spectral.backend="hitran_hapi_v1"` 的路径真正可执行。当前缺口不是物理后端本身，而是用户遇到 cache 缺失时，缺少一个与错误提示一致的 CLI 预热入口。

### 3.2 建议改动

| 文件 | 改动 |
|------|------|
| `rcdw_mgda/scripts/generate_benchmark.py` | 新增 `--precompute-cache-only` 选项；只预热 cache，不落盘 benchmark |
| `rcdw_mgda/rcdw/sim/generation/optical_backend.py` | 复用 `collect_hitran_cache_requirements` 的需求收集逻辑，避免 CLI 自己拼 spectrum 列表 |
| `rcdw_mgda/rcdw/sim/generation/spectral/hitran_backend.py` | 如已有 fetch/cache 原语，则只补 CLI 调用层；不要重复实现 HAPI 表下载逻辑 |
| `rcdw_mgda/tests/` | 增加无网络 mock 的 CLI 行为测试：cache 缺失提示、precompute 分支调用、empirical 后端不触发预热 |

### 3.3 验收

```bash
cd rcdw_mgda

# 空 cache 下先预热
python -m scripts.generate_benchmark --config configs/smoke.yaml --precompute-cache-only

# 再生成 HITRAN smoke
python -m scripts.generate_benchmark --config configs/smoke.yaml --dataset-slug rcdw-hitran-smoke --output-root data
```

验收条件：
- cache 预热命令存在，且错误提示不再指向不存在的操作。
- cache 预热只写 HITRAN cache，不创建半成品 benchmark 目录。
- empirical 后端继续不需要联网、不需要 cache。
- `validation_summary.json` 仍为 `status="pass"`，并包含 scaler validation 结果。

---

## 4. Phase 6B：HITRAN smoke 证据链

### 4.1 目标

把 Phase 1-5 的 empirical 端到端证据升级为 HITRAN smoke 证据，确认默认物理后端下的数据、训练、评测和扰动曲线都能闭环。

### 4.2 建议流程

1. 生成 64 sequence HITRAN smoke，用于快速校验。
2. 如 64 sequence 稳定，再生成 128 sequence HITRAN smoke，用于观察 Stage A/B 是否更平滑。
3. 删除旧 ckpt，重新训练 Stage A/B。
4. 运行 eval 与 perturb，保存指标和 10 张图。
5. 与 empirical smoke 的 MAE、RMSE、扰动趋势做表格对照。

### 4.3 记录要求

建议在完成情况文档中新增 “Phase 6B HITRAN smoke 证据” 小节，至少记录：

| 项 | 内容 |
|----|------|
| dataset slug | 如 `rcdw-hitran-smoke-64`、`rcdw-hitran-smoke-128` |
| spectral backend | `hitran_hapi_v1` |
| cache 状态 | cold / warm，预热耗时 |
| input scaler | on / off、manifest 标志、artifact 路径 |
| generation 耗时 | sequence 数、总耗时、平均每 sequence 耗时 |
| train/eval 指标 | input scaler 状态、overall MAE/RMSE + 各气体 MAE |
| perturb 产物 | 10 张图路径和趋势结论 |

---

## 5. Phase 6C：benchmark 并行生成

### 5.1 触发条件

只有在 6A 与 6B 确认 HITRAN 路径正确后，再做并行化。否则性能优化会放大调试成本。

### 5.2 设计原则

- 保持固定 seed 下单进程与多进程输出等价。
- 继续使用 blake2b 派生 RNG，避免 worker 调度顺序影响随机数。
- chunk 合并时稳定排序，`mixture_id`、`sequence_id` 与 split 结果不可随并行度改变。
- Windows 下避免把不可 picklable 对象传入 worker。

### 5.3 验收

| 验收项 | 期望 |
|--------|------|
| deterministic parity | 单进程与 2 workers 生成的 manifest、sequence_index、labels、splits 一致 |
| validation | 并行生成结果 `status="pass"` |
| performance | formal 规模耗时有明确下降；若下降不明显，不强行保留复杂并行代码 |

---

## 6. Phase 6D：12 维 input scaler ablation 与证据收口

### 6.1 当前状态

2026-07-01 H1 修复后，12 维 input scaler 已完成工程接入：

- `fit_input_channel_scaler()` 覆盖 slow 7 维 + ultrasonic 元数据 5 维，包含 USNet 主输入 `ultrasonic_sound_speed_estimated_m_per_s`。
- `input_scaler.json` 由 train split 拟合，记录 `coverage="input_12ch"` 与 `version="rcdw-input-scaler-1"`。
- `manifest.input_normalization.applied=true` 控制默认应用，artifact 指向 `scalers/input_scaler.json`。
- `BenchmarkDataset(apply_input_scaler=None)` 默认跟随 manifest；`apply_input_scaler=False` 可强制返回原始物理量纲，用于 layout 测试和 ablation。
- 旧数据集无 `input_normalization` 字段时默认不标准化，保持向后兼容；显式 `apply_input_scaler=True` 会拒收未声明 scaler 的旧数据集。
- passthrough 通道保持恒等变换，零方差通道不会产生 NaN / Inf。

### 6.2 剩余任务

Phase 6D 不再是工程实现任务，而是实验验证任务：

1. 在 HITRAN smoke 数据集上分别跑 scaler on / off 对照。
2. 比较 Stage A/B 收敛曲线、overall MAE / RMSE、各气体 MAE / RMSE。
3. 确认 passthrough 通道不变量：`ultrasonic_tof_accepted` 保持 0/1 语义，零方差通道不产生 NaN / Inf。
4. 记录开启 scaler 后输入分布改变，旧 ckpt 不可复用，必须重训。
5. 将 scaler 状态写入 HITRAN smoke 证据表，避免混淆原始量纲训练与标准化训练指标。

### 6.3 验收

- `BenchmarkDataset()` 默认应用新数据集 manifest 声明的 input scaler。
- `BenchmarkDataset(apply_input_scaler=False)` 可用于 scaler-off ablation。
- scaler on / off 的 Stage A/B、eval 指标有记录。
- 文档明确：input distribution 改变后旧 ckpt 不可复用。
- 若 scaler-on 指标不稳定，保留 scaler-off 对照并记录原因，不把不稳定结果纳入正式结论。

---

## 7. Phase 6E：扰动评测扩展

### 7.1 顺序建议

先做 `pressure_drift`，再评估 `h2o_cross`。当前 `pressure_drift` 已按输入空间 `P_MPa` 漂移闭环；H2O cross 需要更仔细地区分光谱建模中的真实 H2O 交叉与训练侧扰动注入。

### 7.1.1 当前代码与实验状态（2026-07-01）

已完成：

- `rcdw/perturbation/inject.py`：`PERTURBATION_KINDS` 增加 `pressure_drift`，实现为对 `IDX_P_MPa` 的确定性输入偏移 `level*2.0`。
- `configs/phase6e-hitran-smoke-128-pressure.yaml`：基于 128-seq HITRAN smoke 与 scaler-on 路径，扰动列表增加 `pressure_drift`。
- `tests/test_perturbation.py`：新增 `test_pressure_drift_targets_p_mpa`，确认只改压力通道。
- `scripts.perturb` Phase 6E 配置已运行，输出目录为 `runs/phase6e_pressure/perturb/`，生成 12 张 PNG：
  - `optical_atten_metrics.png` / `optical_atten_weights_CO2.png`
  - `optical_scat_metrics.png` / `optical_scat_weights_CO2.png`
  - `thermal_metrics.png` / `thermal_weights_CO2.png`
  - `ultrasonic_metrics.png` / `ultrasonic_weights_CO2.png`
  - `temperature_metrics.png` / `temperature_weights_CO2.png`
  - `pressure_drift_metrics.png` / `pressure_drift_weights_CO2.png`
- 全量测试：222 passed。

待完成：

- `h2o_cross` 尚未加入 `PERTURBATION_KINDS`；如后续实现，需要先明确它是输入空间湿度扰动，还是光学后端重新生成扰动。

Phase 6E 运行摘要（128-seq HITRAN smoke，scaler-on，test windows=500，`apply_input_scaler=None` 跟随 manifest）：

| 扰动 | level=0 MAE/RMSE | level=0.11 MAE/RMSE | 趋势结论 |
|------|------------------|---------------------|----------|
| optical_atten | 0.0580 / 0.0687 | 0.0580 / 0.0687 | 基本持平 |
| optical_scat | 0.0580 / 0.0687 | 0.0581 / 0.0686 | 基本持平 |
| thermal | 0.0580 / 0.0687 | 0.0577 / 0.0685 | 基本持平，误差轻微下降 |
| ultrasonic | 0.0580 / 0.0687 | 0.0581 / 0.0688 | 基本持平 |
| temperature | 0.0580 / 0.0687 | 0.1129 / 0.1463 | 退化最明显，仍是当前主导扰动 |
| pressure_drift | 0.0580 / 0.0687 | 0.0576 / 0.0678 | 未观察到退化，误差轻微下降 |

解释：当前 `pressure_drift` 只移动标准化后的输入压力通道，不重新生成由压力派生的物理观测通道，因此它不是本轮 128-seq scaler-on smoke 中的可观测退化源。所有扰动 level 均打印 `degraded=True`，这一点延续 6B/6D 中 ErrorNet 与 hard_suppress 判定偏敏感的现象，不能单独作为扰动严重性的证据。

### 7.2 验收

| 扰动 | 验收重点 |
|------|----------|
| `pressure_drift` | 当前实现只影响 `P_MPa` 输入通道；结果显示无明显退化，文档需明确未做压力派生通道重生成 |
| `h2o_cross` | 明确是输入扰动还是光学后端生成扰动；不得与真实 HITRAN 吸收语义混淆 |

新增扰动后，`scripts/perturb.py` 的图数量、文件名和完成情况文档都要同步更新。

---

## 8. Phase 6F：O2 弛豫参数校核

### 8.1 目标

把 `acoustic_model_metadata.o2_relaxation_source="placeholder_v1"` 替换为可追溯来源，或至少补齐实测校核流程。

### 8.2 验收

- 文档记录 O2 弛豫参数来源、适用温压范围和对 40 kHz 工作点的影响。
- 如果参数发生变化，重新生成 HITRAN smoke 并重训评估。
- manifest 或 metadata 中能追踪参数版本。

---

## 9. Phase 6G：多 stage_profile 激活

### 9.1 前置条件

多 profile 会改变数据生成结构，建议放在 Phase 6 后段或 v2。不要在 HITRAN smoke、并行化、scaler ablation 尚未稳定时引入。

### 9.2 影响面

| 模块 | 影响 |
|------|------|
| `conditions.py` | 从 condition 到 sequence 的关系可能变为 1:N |
| `phases.py` | 注册表从仅 `standard_exposure` 扩展到多个 profile |
| `splits.py` | 必须继续按 `mixture_id` 分组，避免同一 mixture 泄漏到不同 split |
| `manifest.py` | 需要记录 stage profile 与 profile version |
| tests | 需要新增同 mixture 多 profile 的 split 不泄漏测试 |

---

## 10. 非目标

Phase 6 不做以下事情：

- 不把 RCDW-MGDA 合并回主线 `src/`。
- 不把 `mixture_id` 回退或重写为 `sequence_id`。
- 不引入 `base_condition_id`、`noise_seed_index`、`noise_seed` 作为新 benchmark 依赖。
- 不改变 RCDW 独立 ID 前缀 `RCDW-M/Q`。
- 不默认升级 schema_version；只有落盘字段或兼容性契约发生破坏性变化时，才讨论 schema 版本策略。
- 不在没有 HITRAN smoke 证据前优化 formal 性能或扩展复杂 profile。

---

## 11. Phase 6 验收总表

| Gate | 通过标准 | 失败时处理 |
|------|----------|------------|
| Gate 1：默认 smoke 可用 | 空 cache 预热 + HITRAN smoke generate 成功 | **已通过**：`--precompute-cache-only` 真实预热成功，64-seq HITRAN smoke validation pass |
| Gate 2：HITRAN 科学 sanity | HITRAN smoke train/eval/perturb 全链路有记录 | **已通过**：64-seq overall MAE=0.06380 / RMSE=0.08511；128-seq overall MAE=0.05741 / RMSE=0.06790；扰动图均已生成 |
| Gate 3：并行化收益 | 并行结果确定性等价且耗时下降 | **已通过确定性门槛**：empirical 与 HITRAN 合成 cache 单进程 / 并行 parity 测试通过；formal 性能收益待更大规模实测 |
| Gate 4：scaler ablation | scaler on/off 指标清晰，passthrough 不变量通过 | **已通过**：128-seq scaler-on overall MAE/RMSE 0.05741/0.06790，scaler-off 0.07642/0.09101；正式路径保留 scaler-on |
| Gate 5：扰动扩展 | 新扰动图与趋势说明同步文档 | **pressure_drift 已通过**：6 类扰动图与趋势说明已同步；`pressure_drift` 结论为输入空间压力漂移未造成可观测退化；`h2o_cross` 暂不纳入正式扰动结论 |
