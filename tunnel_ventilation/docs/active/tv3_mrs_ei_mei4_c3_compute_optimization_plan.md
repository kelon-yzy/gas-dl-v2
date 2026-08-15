# MEI-4 C3 计算效率与可恢复执行优化计划

> 文档状态：核心代码改造已完成；正式性能基准与全量 C3 未执行  
> 编制日期：2026-07-30  
> 适用阶段：MEI-4 C3 已授权的 SBC、PPC 与 PSIS 触发后的 M2b  
> 当前实验状态：`mei4_mc_authorized_pending_execution`  
> 关联文档：[MEI-4 执行计划](tv3_mrs_ei_mei4_execution_plan.md)、[MEI-4 执行进度](tv3_mrs_ei_mei4_execution_progress.md)

## 实施状态

| 阶段 | 状态 | 当前证据 |
| --- | --- | --- |
| P0 | 部分完成 | 已完成缩小契约与双进程微型验证；尚未执行 `1/4/8/12/16` worker 性能矩阵。 |
| P1 | 已完成 | 已实现稳定任务、attempt manifest、原子分片、严格校验、串行恢复和固定顺序归并。 |
| P2 | 已完成 | 已实现 attempt 级进程池、有界 Future、单线程 BLAS、异常传播和失败状态记录。 |
| P3 | 未执行 | 需要单独运行非正式性能基准后确定正式 worker 数。 |
| P4 | 未触发 | 当前没有 profiler 证据支持继续修改科学计算热点。 |
| P5 | 未执行 | 未启动正式 C3，未生成 C3 完成 freeze，未更新实验阶段状态。 |

本次改造没有运行 `--run-authorized-mc`，不产生新的科学证据或 verdict。

## 1. 结论

当前 C3 的主要问题不是抽样规模设置错误，而是执行架构不能保存阶段性成果，也没有利用混合物之间天然独立的计算结构。优化应按以下顺序实施：

1. 将 SBC、PPC、M2b 拆成有稳定身份的独立任务，并增加可校验、可恢复的分片检查点；
2. 在保持现有随机数和归并顺序不变的前提下，使用 `ProcessPoolExecutor` 做确定性多进程并行；
3. 将每个 worker 的 BLAS 线程固定为 1，避免“多进程乘以多线程”造成 CPU 过度订阅；
4. 主进程按冻结顺序严格归并全部分片，校验完整性后再生成唯一的正式 C3 freeze；
5. 只有完成上述结构性修复且性能仍不足时，才评估缓存、前向批处理和重复抽样复用。

检查点只用于恢复计算，不是科学证据。只有最终 C3 报告完整、输入与代码哈希一致、manifest 校验通过并完成状态晋升，才可作为 C4/C5 的正式输入。

## 2. 当前停止位置与计算负担

### 2.1 停止位置

2026-07-30 停止运行时，日志记录为：

| 子阶段 | 日志进度 | 正式落盘状态 | 恢复结论 |
| --- | ---: | --- | --- |
| SBC test | `1000 / 1000` | 未落盘 | 旧实现下必须重算 |
| SBC OOD | `1000 / 1000` | 未落盘 | 旧实现下必须重算 |
| PPC test | `648 / 648` | 未落盘 | 旧实现下必须重算 |
| PPC OOD | `648 / 648` | 未落盘 | 旧实现下必须重算 |
| M2b | `1 / 1296` | 未落盘 | 只完成极少部分 |

当前程序先在内存中顺序生成三个完整结果，最后一次性写入 `sbc_rank_histograms.json`、`ppc_report.json` 和 `bootstrap_posterior_report.json`。进程在 M2b 中断后，已完成的 SBC/PPC 随进程一起丢失，无法通过 manifest 复核，因此不能作为正式证据。

### 2.2 主要计算量

冻结契约规定：

- SBC：test/OOD 各 1000 次，每次评价 M1、M1b、M2，后验秩抽样数为 256；
- PPC：test/OOD 各 648 个冻结混合物，每种方法每个混合物生成 64 个 `y_rep`；
- M2b：1296 个混合物，每个混合物执行 200 次参数自助重求解；
- S1：每次求解使用 3 个冻结初值，单次优化最多 100 次迭代。

M2b 单独需要：

```text
1296 个混合物 × 200 次重抽样 × 3 个冻结初值
= 777,600 次 S1 优化器启动
```

每次优化器启动又包含多次前向模型计算、残差计算和数值线性代数。实际成本由前向调用总量决定，明显高于 777,600 次简单函数调用。

SBC 也需要对 2000 个合成观测分别运行 S1 和 S2 的 3 个冻结初值，按当前实现至少产生 12,000 次求解器启动。PPC 不重复优化 S1，但需要对 1296 个混合物、3 种方法和每种方法 64 个样本执行约 248,832 次前向预测，同时重复构造后验和线性代数对象。

## 3. 根因

### 3.1 写盘边界过大

正式 runner 仅在 `run_c3_mc_calibration()` 完整返回后写 freeze。SBC、PPC 和 M2b 之间没有持久化边界，单个子阶段内部也没有分片检查点。任何中断都会丢失本次进程的全部结果。

### 3.2 调度完全串行

SBC replicate、PPC 混合物和 M2b 混合物彼此独立，但当前代码都由单进程 `for` 循环执行。本机 16 个物理核心不能被有效利用。

### 3.3 BLAS 线程存在过度订阅风险

环境盘点时 OpenBLAS 默认使用 24 个线程。如果直接启动 12 或 16 个 Python worker，理论线程数会膨胀到 288 或 384，导致上下文切换、缓存争用和内存压力，性能可能低于串行。

### 3.4 重复构造与重复前向计算

参数化、solver settings、calibration、后验对象和部分矩阵分解在循环中重复构造。PPC 对多个样本逐条调用 `predict_s1`。这些不是第一优先级根因，但会限制并行完成后的进一步加速。

### 3.5 正式证据与运行中间态耦合

当前只有“全部成功后形成正式 freeze”这一层。缺少独立的 attempt 和 checkpoint 层，使“可恢复计算状态”与“正式科学证据”无法同时满足。

## 4. 优化目标

### 4.1 必须达到

1. 中断后只重算缺失或未完成的分片，不重算已校验分片；
2. `workers=1` 的任务级和最终聚合结果与现有串行实现一致；
3. `workers=1/4/8/12/16` 的最终规范化 JSON 内容一致；
4. worker 完成顺序不得影响随机数、样本顺序、统计量或最终 manifest 内容；
5. 损坏、重复、重叠、跨输入或跨代码版本的分片必须显式失败；
6. 峰值物理内存低于 24 GB，给操作系统和文件缓存保留空间；
7. 正式 freeze 只能在全部任务完整且验证通过后生成。

### 4.2 性能目标

- 首轮候选 worker 数：`1、4、8、12、16`；
- 推荐起点：12 个 worker，每个 worker 1 个 BLAS 线程；
- 稳态 CPU 利用率目标：80% 到 95%；
- M2b 墙钟加速目标：相对新架构 `workers=1` 达到 6 到 10 倍；
- 每完成一个 M2b 混合物即具备可恢复结果，强制终止时最多损失正在运行的混合物任务。

这些是实测目标，不是预先承诺。最终 worker 数由固定基准结果决定。

## 5. 不变量与边界

本优化只改变工程执行方式，不改变科学计算契约。

1. SBC、PPC、M2b 的规模、方法集合、阈值和种子不变；
2. 三个冻结初值不变；
3. S1/S2 的参数化、边界、最大 100 次迭代和停止条件不变；
4. `mixture_id` 继续作为组分主键，不得回退或改写为 `sequence_id`；
5. test/OOD 顺序和 B4 冻结记录顺序不变；
6. 拒绝样本语义和覆盖门不变，不得通过重试或过滤改善结果；
7. 不引入温度缩放、先验缩窄、conformal 或其他未登记重标定；
8. 不修改 B4/B5/C0/C2/C3 授权 freeze；
9. 不在本优化中启动 CC-SBI、波形生成、benchmark 打包或硬件工作；
10. 不以 GPU 重写求解器作为首轮方案。

运行参数只能控制 worker 数、分片大小、在途任务数和检查点目录，不得复制或覆盖科学参数。科学参数唯一来源仍是冻结的 C0 execution contract。

## 6. 目标架构

### 6.1 三层结构

```text
冻结输入与科学契约
        |
        v
任务层：生成稳定 task_id，执行单个 replicate 或混合物
        |
        v
恢复层：原子写入并校验 attempt 分片
        |
        v
证据层：按冻结顺序归并，生成正式报告和 C3 freeze
```

- 任务层只负责一个确定输入到一个确定输出；
- 恢复层保存非正式中间结果；
- 证据层不重新计算，只验证和归并完整分片。

### 6.2 任务划分

| 子阶段 | 最小任务单位 | 建议分片 | 原因 |
| --- | --- | --- | --- |
| SBC | 一个 `(domain, replicate_index)` | 每片 8 到 16 个 replicate | 单任务成本中等，分片可降低写盘频率 |
| PPC | 一个 `mixture_id` | 每片 8 个混合物 | 单任务成本较小，批量写盘更合适 |
| M2b | 一个 `mixture_id` | 每个混合物一个分片 | 单任务很重，细粒度恢复价值最高 |

分片大小属于运行参数。正式基准可调整分片大小，但不能改变任务身份、任务内容和随机数映射。

### 6.3 稳定任务身份

任务 ID 必须由科学身份直接构造：

```text
SBC  : sbc/<domain>/<replicate_index>
PPC  : ppc/<domain>/<mixture_id>
M2b  : m2b/<domain>/<mixture_id>
```

任务 ID 不包含 worker 编号、进程号、开始时间或完成顺序。这样同一任务在串行、并行和恢复运行中保持同一身份。

### 6.4 attempt 目录

```text
outputs/runs/tv3_mrs_ei/mei4_posterior_calibration/
  attempts/<attempt_id>/
    attempt_manifest.json
    attempt_status.json
    shards/
      sbc/
      ppc/
      m2b/
```

`attempt_id` 绑定以下 SHA256：

- B4 evidence manifest；
- C0 execution contract；
- C2 evidence manifest；
- C3 authorization manifest；
- 直接参与执行的源文件；
- Python、NumPy、SciPy 和 `threadpoolctl` 版本；
- 运行配置及任务计划摘要。

恢复时任一绑定项不一致都应拒绝继续。不得自动迁移旧分片，也不得提供忽略哈希的静默选项。

### 6.5 分片格式与原子发布

每个分片至少记录：

- schema version、attempt ID、阶段和任务 ID；
- domain、`mixture_id` 或 replicate 索引；
- 输入绑定摘要和任务计划摘要；
- 单任务结果及可归并的原始统计贡献；
- 计算成本、开始时间、结束时间和 worker 信息；
- 规范化 payload 的 SHA256。

写入流程：

1. 主进程接收 worker 结果；
2. 写入同目录临时文件；
3. 重新读取并校验 schema、任务 ID 和 payload SHA256；
4. 使用原子 rename 发布正式分片；
5. 已存在同名有效分片时跳过，内容冲突时立即失败。

worker 不直接写正式分片。`attempt_status.json` 仅用于展示进度，可由已验证分片重建，不作为完成判定的唯一来源。

### 6.6 分片保存内容

- SBC 保存每个 replicate、每种方法和每个组分的 rank 或 rejection；
- PPC 保存每个混合物、每种方法的白化残差尾概率、12 通道经验分位数或 rejection；
- M2b 保存每个混合物的区间、覆盖事件、NLL/CRPS 贡献、rejection 和 forward-call 计数；
- 不为恢复目的重复保存可以从上述内容无损聚合出的第二套统计报告。

## 7. 确定性并行设计

### 7.1 进程模型

采用 `concurrent.futures.ProcessPoolExecutor`。Windows 使用 spawn 模式，因此 worker 函数、initializer 和任务数据结构必须定义在模块顶层，并可被 pickle。

主进程职责：

- 验证 freeze 与 attempt 绑定；
- 生成任务计划；
- 跳过已完成且校验通过的任务；
- 控制在途任务数量；
- 接收结果并原子写分片；
- 按冻结顺序归并并生成正式 freeze。

worker 职责：

- 初始化一次不可变上下文；
- 执行收到的纯任务；
- 返回结果或向主进程传播原始异常；
- 不更新 `stage_status.json`，不创建 freeze，不吞错。

### 7.2 worker 初始化

每个 worker 只加载一次：

- B4 records 和 audit tables；
- solver config、calibration 和 C0 contract；
- 参数化对象和可安全共享的只读索引；
- BLAS 线程限制。

首次实现允许 worker 各自加载只读数据。只有实测内存或启动时间不达标时，再评估 memmap 或更紧凑的任务传输，避免提前增加复杂度。

### 7.3 BLAS 线程限制

在创建进程池前将 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS` 和 `NUMEXPR_NUM_THREADS` 设为 1；worker initializer 再用 `threadpoolctl.threadpool_limits(1)` 约束已加载的数值库，并记录实际线程池信息。

`threadpoolctl` 应作为直接依赖声明，不能依赖其他包的间接安装结果。启动后若任一 worker 报告 BLAS 线程数大于 1，基准和正式运行都应停止。

### 7.4 随机数保持

PPC 和 M2b 已按 domain、method、`mixture_id` 等稳定字段派生独立种子，可直接并行。

SBC 当前按 domain 使用一个顺序 RNG 流，模板选择和观测噪声共同消耗该 RNG。为保持现有语义，主进程必须按原顺序重建 SBC task 描述，再将确定的 task 输入交给 worker。恢复时即使前部任务已有分片，也要按原顺序推进任务描述生成，不能按 worker 编号重新分配种子。

不得将 SBC 改成“每个 replicate 新派生一个种子”，因为这会改变已冻结运行的随机流语义。

### 7.5 有界调度

`max_inflight` 默认设为 `2 × workers`。主进程只保持有限数量的未完成 Future，避免一次提交全部 M2b 任务造成内存和序列化压力。

收到中断时：

- 停止提交新任务；
- 允许当前最小任务完成并写入分片；
- 取消尚未开始的任务；
- 以非零状态退出，MEI-4 状态保持 `mei4_mc_authorized_pending_execution`；
- 下次从已验证分片恢复。

### 7.6 确定性归并

Future 的完成顺序只能影响分片到达时间，不能影响结果顺序。归并顺序固定为：

1. domain：`test` 后 `ood`；
2. SBC：replicate 索引升序；
3. PPC/M2b：B4 冻结 records 原始顺序；
4. method：`M1`、`M1b`、`M2`；
5. component：现有 `COMPONENTS` 顺序；
6. nominal level：C0 contract 原始顺序。

worker 只返回逐任务贡献，不在 worker 内做跨任务 reduction。最终均值、分位数、KS 检验、覆盖计数和成本汇总在主进程按上述顺序执行，避免浮点加法顺序变化。

## 8. 正式 freeze 生成规则

正式 C3 freeze 前必须完成以下检查：

1. 任务计划中的 task ID 全部存在；
2. 无重复、无范围重叠、无未知 task ID；
3. 每个分片 schema、attempt ID、输入摘要和 payload SHA256 有效；
4. 每域任务数与 C0 contract 一致；
5. SBC、PPC、M2b 聚合报告均通过结构校验；
6. 聚合结果能由分片重新生成；
7. 父 manifest 和所有相关源文件哈希未变化；
8. 正式 freeze 的 evidence manifest 自校验通过。

只有完成上述检查，runner 才能：

- 写入三个正式 C3 JSON；
- 生成 append-only freeze；
- 将状态晋升为 `mei4_c3_mc_calibration_complete`；
- 设置 `c4_review_eligible=true`。

任何失败都保留 attempt 分片用于诊断，但不更新正式阶段状态。

## 9. 本地资源使用方案

当前机器盘点：

| 资源 | 盘点结果 | 使用策略 |
| --- | --- | --- |
| CPU | Ryzen 9 8940HX，16 核 32 线程 | 优先使用 8 到 16 个进程，以物理核心为上限 |
| RAM | 32 GB | 正式门限设为峰值低于 24 GB |
| GPU | RTX 5060 Laptop，8 GB | 首轮不使用 |
| D 盘 | 可用约 277.5 GiB | attempt 分片写入现有输出目录 |
| 数值库 | OpenBLAS 盘点时默认 24 线程 | 每个 worker 固定为 1 线程 |

不把 32 个逻辑线程直接作为默认 worker 数。当前任务包含 Python 调度、SciPy 优化和小型线性代数，超线程收益需要实测，且 32 worker 更容易增加内存与调度开销。

GPU 暂不作为主路径。当前核心是 SciPy 求解器和大量小型分支任务，迁移到 GPU 需要改写求解器与数据流，会扩大数值等价性风险。只有 CPU 并行和批处理完成后仍无法满足预算，且 profiling 显示前向模型占绝对主要成本时，才单独立项评估 GPU。

## 10. 分阶段实施

### P0：冻结基准与剖析

**目标**：建立优化前可比较基线。

实施：

- 固定一组非正式工程基准任务，覆盖 test/OOD、正常、触界和 rejected 情形；
- 分别记录 SBC、PPC、M2b 的墙钟、CPU、峰值内存、forward calls 和单任务耗时分布；
- 保存旧串行实现的逐任务结果和规范化聚合 JSON；
- 确认 OpenBLAS 实际线程数。

验收：基准任务和输出摘要有固定 SHA256，可供后续所有 worker 数对比。工程基准不得写入正式 C3 freeze。

### P1：纯任务函数与串行检查点

**目标**：先解决中断丢失，再引入并行。

实施：

- 从现有循环提取 SBC、PPC、M2b 单任务函数；
- 建立 task ID、attempt manifest 和分片 schema；
- 主进程实现原子写入、读取校验、缺失任务发现和严格归并；
- 用 `workers=1` 完成可恢复串行路径。

验收：

- 新串行路径与旧串行路径逐任务一致；
- 中断后只执行缺失任务；
- 删除、截断或篡改单个分片时显式失败；
- attempt 分片不会触发正式状态晋升。

### P2：确定性多进程

**目标**：利用本机多核，不改变输出。

实施：

- 增加顶层 worker initializer 和任务入口；
- 使用 `ProcessPoolExecutor` 和有界 Future 队列；
- 限制每个 worker 的 BLAS 线程；
- 主进程单独负责写盘和归并；
- 增加进度、吞吐率、ETA、完成分片数和 rejection 统计日志。

验收：

- `workers=1/4/8/12/16` 的规范化聚合 JSON 一致；
- 多次运行和不同 Future 完成顺序下结果一致；
- worker 异常直接导致非零退出，并保留已完成分片；
- Windows spawn 下无递归启动和 pickle 错误。

### P3：资源基准与参数定型

**目标**：确定正式运行配置。

实施：

- 对固定 M2b 工程基准测试 `workers=1/4/8/12/16`；
- 每组至少重复 3 次，报告中位墙钟和离散程度；
- 同时记录 CPU、峰值内存、每秒完成任务数和 BLAS 线程；
- 在 8、12、16 worker 之间选择吞吐最高且内存低于 24 GB 的配置。

决策规则：若 16 worker 相比 12 worker 的中位吞吐提升不足 5%，选择 12 worker，保留温度、内存和系统交互余量。

### P4：第二阶段热点优化

**启动条件**：P2/P3 完成后仍未达到性能目标，并且 profiler 已定位明确热点。

按收益和风险排序：

1. worker 内缓存不可变的 parameterization、solver settings、calibration 索引和混合物 Cholesky；
2. 对 PPC 的 `predict_s1` 做等价批量前向，减少 Python 循环；
3. 对 M2 或 PPC 内重复生成的相同候选样本做任务内复用；
4. 减少主进程与 worker 之间的重复序列化。

每项单独实施、单独基准、单独做数值等价验证。没有 profiler 证据的优化不实施。任何会改变随机数消耗顺序、候选样本顺序或求解器调用顺序的方案都应拒绝。

### P5：正式恢复执行

前置门全部通过后，创建新的正式 attempt：

1. 重新验证 B4、C0、C2 和 C3 authorization manifest；
2. 记录正式 runtime config 和全部代码哈希；
3. 依次执行或恢复 SBC、PPC、M2b；
4. 完成全任务盘点和确定性归并；
5. 生成新的 C3 freeze；
6. 校验 manifest；
7. 更新 `stage_status.json`；
8. 更新执行进度文档。

本计划的编写不等于启动 P5。正式全量运行应在 P0 至 P4 的适用门通过后单独执行。

## 11. 计划影响文件

| 文件 | 计划改动 |
| --- | --- |
| `tv3/audit/mrs_ei_mei4_mc.py` | 提取单任务计算和确定性聚合接口，保留科学逻辑唯一来源 |
| `tv3/audit/mrs_ei_mei4_parallel.py` | 新增 attempt、分片校验、worker 初始化和进程池编排 |
| `scripts/run_tv3_mei4_c3_mc_calibration.py` | 增加 runtime config、fresh attempt、resume 和完整性门 |
| `configs/tv3_mrs_ei/mei4_c3_runtime.json` | 只保存 worker、分片和在途任务等工程参数 |
| `tests/test_tunnel_ventilation_mei4_mc.py` | 增加串行等价、恢复、损坏分片和聚合顺序测试 |
| `tests/test_tunnel_ventilation_mei4_mc_parallel.py` | 增加多 worker、spawn、过度订阅和故障传播测试 |
| `pyproject.toml` | 将 `threadpoolctl` 声明为直接依赖 |
| `docs/active/tv3_mrs_ei_mei4_execution_progress.md` | P5 完成后记录新 freeze 与实测成本 |

若实现时现有模块边界足以容纳并行代码，可不新增第二个测试文件；不得为了匹配本表制造空模块或重复逻辑。

## 12. 验证矩阵

### 12.1 单元测试

- task ID 稳定且唯一；
- SBC 顺序 RNG 任务描述与旧实现一致；
- PPC/M2b 每任务种子与旧实现一致；
- 分片规范化序列化和 payload SHA256 可复算；
- 临时文件不会被误判为已完成分片；
- 重复、重叠、缺失、未知、损坏分片显式失败；
- 跨 contract、跨授权、跨代码版本恢复显式失败；
- rejection、区间、coverage event、NLL、CRPS 和成本聚合一致。

### 12.2 集成测试

- `workers=1` 与旧串行结果一致；
- `workers=4/8/12/16` 与 `workers=1` 一致；
- 在 SBC、PPC、M2b 各阶段注入中断，恢复后只补齐缺失任务；
- worker 抛出真实异常时 runner 非零退出，不更新正式状态；
- 合并顺序不受人为延迟和 Future 完成顺序影响；
- 最终 freeze 可通过现有 `verify_evidence_manifest`。

后端单元测试默认限制在 60 秒内。长时间并行和性能测试单独标记，不混入日常快速测试。

### 12.3 性能验收

| 指标 | 门限 |
| --- | --- |
| 峰值内存 | `< 24 GB` |
| worker BLAS 线程 | `= 1` |
| 稳态 CPU 利用率 | 目标 `80% 到 95%` |
| M2b 加速比 | 目标 `6 到 10 倍`，以 `workers=1` 为基线 |
| 中断重算范围 | 仅缺失或未完成任务 |
| 正式输出一致性 | 不受 worker 数和完成顺序影响 |

## 13. 失败处理与停止条件

以下任一情况发生时停止，不生成正式 C3 freeze：

1. 新串行路径无法复现旧串行结果；
2. 不同 worker 数产生不同规范化报告；
3. BLAS 实际线程数不等于 1；
4. attempt 绑定哈希变化；
5. 分片损坏、重复、重叠或任务集合不完整；
6. 峰值内存达到或超过 24 GB；
7. worker 异常、求解器异常或聚合异常；
8. 正式 manifest 校验失败；
9. 发现优化改变了 C0 冻结的科学参数、随机数语义或 rejection 规则。

失败后保留日志和已验证分片，不吞错，不生成占位报告，不把部分结果标记为成功。

## 14. 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| 并行改变 RNG 流 | SBC 在主进程按旧顺序物化任务；PPC/M2b 沿用稳定派生种子 |
| Future 完成顺序改变浮点结果 | 保存逐任务贡献，按冻结顺序单进程聚合 |
| 多进程乘以 BLAS 多线程 | 创建进程池前设置环境变量，worker 内用 `threadpoolctl` 验证 |
| worker 重复加载导致内存过高 | 先实测；必要时再引入只读 memmap，不提前复杂化 |
| checkpoint 被误当正式证据 | attempts 与 freezes 物理分离；只有 manifest 完整的 freeze 可晋升状态 |
| 代码更新后误用旧分片 | attempt ID 绑定直接参与执行的源码 SHA256 和依赖版本 |
| 主进程崩溃时分片半写 | 临时文件、校验、原子 rename；启动时忽略未发布临时文件 |
| 运行状态成为第二来源真相 | 完成集合从已验证分片重建，`attempt_status.json` 只作摘要 |
| 过细分片造成 I/O 开销 | SBC/PPC 使用小批量分片，M2b 因任务重而保持每混合物一片 |

## 15. 完成定义

工程优化完成必须同时满足：

1. P0 至 P3 验收全部通过；
2. P4 若未触发，记录未触发原因；若触发，每项均有 profiler 证据和等价测试；
3. 选定的正式 worker 配置有可复算基准；
4. 中断恢复测试证明只重算缺失任务；
5. 并行结果与串行基准一致；
6. 正式 C3 全量运行完成；
7. 三份正式报告和 evidence manifest 校验通过；
8. `stage_status.json` 正确晋升；
9. 执行进度文档记录实际墙钟、峰值内存、worker 数、加速比和 freeze SHA256。

## 16. 参考依据

- Python 3.13 `ProcessPoolExecutor`：<https://docs.python.org/3.13/library/concurrent.futures.html>
- scikit-learn 并行与过度订阅说明：<https://scikit-learn.org/stable/computing/parallelism.html>
- NumPy 线程安全说明：<https://numpy.org/doc/stable/reference/thread_safety.html>
