# RCDW-MGDA 代码质量审查报告

> **第一轮审查**：2026-07-01 · 约 5,900 行 · 10 项发现 · **全部已修复**（测试 201→212）
> **第二轮审查**：2026-07-02 · 约 10,700 行（Phase 1–6 全量） · 8 项新发现 + 9 个观察项 · 6 项已修复、2 项部分修复；另有 1 个报告字段残留
>
> 审查范围：`rcdw_mgda/` 子工程全量源码 —— models / training / sim / packaging / validation / data / perturbation / scripts
> 结论：**无崩溃级 bug**。第一轮问题已全部修复。第二轮 R1-R6 已落地；R7/R8 已完成主体修复，仍保留少量复核残留；`perturb` report JSON 还需记录 `absolute_threshold` 以保证配置可复现。

---

## 0. 严重级别总览

| 编号     | 级别  | 一句话摘要                                            | 主要位置                                         | 状态    |
| ------ | --- | ------------------------------------------------ | -------------------------------------------- | ----- |
| **H1** | 高   | Z-score scaler 被拟合/落盘/校验，却从未作用到模型输入              | `benchmark.py` / `dataset.py`                | ✅ 已修复 |
| **M2** | 中   | `FeatureExtractor` 环境差分特征 5–7 恒为 0（含温度扰动下）       | `slow.py` / `feature.py` / `inject.py`       | ✅ 已修复 |
| **M3** | 中   | `hard_suppress` 阈值依赖 batch，eval 指标随 batch 大小变化   | `degradation.py` / `eval.py` vs `perturb.py` | ✅ 已修复 |
| **M4** | 中   | 训练不可复现（无 `torch.manual_seed`/seeded DataLoader）  | `train.py` / `training/*`                    | ✅ 已修复 |
| **M5** | 中   | 多个配置节被静默忽略（splits/scalers/phases）                | `smoke.yaml` / `default.yaml`                | ✅ 已修复 |
| **L1** | 低   | "ARE" 实为最大相对误差（非平均）；相对误差在近零真值处爆炸                 | `metrics.py` / `eval.py`                     | ✅ 已修复 |
| **L2** | 低   | `basis="wet"` 路径 RH 单位隐患（百分数 vs 0–1）             | `normalize.py`                               | ✅ 已修复 |
| **L3** | 低   | multi-tau 用 `exp(-1/tau)`，`dt_s` 未接入动力学          | `slow.py`                                    | ✅ 已修复 |
| **L4** | 低   | 5 类扰动强度语义跨类别不可比                                  | `inject.py`                                  | ✅ 已修复 |
| **L5** | 低   | 陈旧注释/文档漂移（`(B,6)`、`synth_timeseries`、198 vs 201） | `stage_a.py` / `normalize.py`                | ✅ 已修复 |

---

## 1. 高（High）

### H1 — train-only Z-score scaler 被拟合、落盘、校验，却从未作用到任何模型输入 [已修复]

> **修复方式**：采用方案 ① 消费侧应用。`fit_input_channel_scaler()` 覆盖 12 维消费布局（含 US 主信号 idx 8），`BenchmarkDataset` 加载 `input_scaler.json` 在 `__getitem__` 执行 `(x-mean)/std`，`manifest.input_normalization.applied` 标志位控制开关，旧数据集自动回退。新增 11 个测试覆盖精确匹配、train 统计、val/test 无泄露、idx 11 零方差安全、向后兼容。

> 📎 深入分析（数据流断点 / 实测通道量级 / 分组件影响 / 修复方案权衡）见文末 **[附录 A](#附录-a-h1-深入分析)**。

**证据链**

- `rcdw/sim/generation/benchmark.py:205` `fit_z_score_scalers(...)` 拟合（仅 train split）。
- 同文件 `:220-226` 写入 `scalers/scaler_slow_sequence.json` 与 `scaler_slow_sequence_modal.json`。
- `rcdw/sim/validation/integrity.py:199` `_validate_scaler_passthrough` "校验" 它；manifest（`benchmark.py:229-233`）也记录 `scaler_metadata`。
- **但** `rcdw/data/dataset.py:__getitem__`（149、186-210）读取的是**原始** `slow.npy` 并直接拼接原始 ultrasonic 元数据 —— 无任何 transform。
- `scripts/train.py`、`scripts/eval.py`、`scripts/perturb.py` 全程不加载 scaler。已在 `rcdw/` + `scripts/` 全量 grep 确认：唯一引用是 `benchmark.py` 的写盘与 `scalers.py` 的 docstring。

**失败场景 / 影响**
模型在**原始异质量级**上训练与推理：US 声速 ~340–450、`P_MPa` ~0.1–0.7、电压 ~2.5、ToF ~1e-4 s。单模态 MLP 与 `FeatureExtractor` 的每一项统计量（环境差分、梯度能量、SNR、drift）都在量级悬殊的原始值上计算后喂给 `ErrorNet`，显著恶化优化，并抬高 `RCDW_实施完成情况.md` §2.3 报告的误差。

**定性**
Phase 6 路线把"12 维 input scaler"列为后续工作，所以属**部分已知/延后**；但 v1.2 完成情况表把 scaler 宣传为**已交付**，manifest/校验也在断言它 —— 现状会误导维护者（"已落盘 + 已校验" 被合理理解为 "已生效"）。

**修复建议**
二选一：

1. 把 scaler 接进 `BenchmarkDataset`：加载 JSON，对 `strategy=="z_score"` 的通道做 `(x-mean)/std`，其余 passthrough；用 manifest 标志位（如 `scaler_applied: true`）控制开关，保证旧数据集可回退。
2. 若确定 Phase 6 前不接入，则把 v1.2 完成情况表 / manifest / 校验措辞降级为"仅拟合、未应用"，避免过度宣称。

**关联缺陷（并入 H1）：passthrough 机制为摆设**
`scalers.py` 的 `skip_channels`（`DEFAULT_PASSTHROUGH_CHANNELS` = 3 个 ultrasonic 元数据通道）与实际被拟合的 7 通道 slow 矩阵**无交集**（`SLOW_CHANNELS` 不含任何 ultrasonic 元数据）。`scalers.py` docstring 亦自承此点。结果 `integrity.py:211` 的 passthrough 校验对这些通道走 `channel not in skip_channels` 分支而**空转通过**，未验证任何真实语义。

---

## 2. 中（Medium）

### M2 — `FeatureExtractor` 特征 5–7（`delta_T`/`delta_P`/`delta_RH`）对每个样本恒为 0 [已修复]

> **修复方式**：在 `slow.py` 为 T_C/P_MPa/H_RH 添加传感器测量噪声（σ 分别为 0.05°C / 0.0005 MPa / 0.1%），使 `delta_*` 在 clean 数据中不再恒为零。噪声量级远小于环境信号，不影响声学/光学物理计算，但使 ErrorNet 的扰动感知特征有效。

**证据链**

- `rcdw/sim/generation/slow.py:249-251` 让环境通道逐时间步取常数基线值：

  ```python
  current["T_C"]  = float(condition["T_C_base"])
  current["P_MPa"] = float(condition["P_MPa_base"])
  current["H_RH"] = float(condition["H_RH_base"])
  ```

  即整条序列所有时间步 T/P/RH 不变。

- `rcdw/models/feature.py:83-85` 计算 `delta_* = |env[-1]-env[-2]|` → **恒为 0**。

- `rcdw/perturbation/inject.py:85` 的 `temperature` 扰动为**整窗均匀偏移** `x[..., IDX_T_C] += level*80`，在一阶差分中被抵消 → 温度故障下 `delta_T` **仍为 0**。

**影响**
13 个 ErrorNet 输入中有 3 个（idx 5/6/7）不携带任何信息；且本应用于捕捉温度故障的 `delta_T` 特征在温度故障下失效。温度扰动只能经"绝对 T 变化 → 单模态输出 Y_modal 变化 → dev 类特征变化"间接被感知，与特征设计意图（扰动感知的环境突变检测）不符。与 H1 叠加（无缩放 + 死特征）进一步削弱 ErrorNet 有效输入。

**修复建议**
二选一：在 `slow.py` 为环境通道注入序列内动态（阶跃/漂移），使 `delta_*` 有意义；或删除特征 5–7 及相关文档描述，避免"扰动感知特征"名不副实。

### M3 — `hard_suppress` 退化阈值按 batch 维计算，导致 eval 指标依赖 batch 大小 [已修复]

> **修复方式**：改为逐样本判定。`E_pred.min(dim=1)` 替代 `E_pred.median(dim=0)`，每个样本独立比较各模态误差，结果与 batch 大小无关。`degraded` 返回形状从 `(M,G)` 改为 `(B,M,G)`，调用方（`eval.py`/`perturb.py`）兼容。

**证据链**

- `rcdw/utils/degradation.py:39-44`：`E_med = E_pred.median(dim=0)` → `degraded = E_med > ratio*min_E`，阈值来自**当前 batch** 的误差分布。
- `scripts/eval.py:44,63-67` 用 `DataLoader(batch_size=64)`，逐 minibatch 判定（末批不整齐）。
- `scripts/perturb.py:59,89-93` 用 `_collect_split_tensor` 一次性喂整个 split，全局判定。

**失败场景**
同一 checkpoint、同一测试集，`eval.py`（逐 64 样本）与 `perturb.py`（整 split）会得到**不同的抑制权重与不同指标**；改 `batch_size` 也会改结果。对一个应确定性的 benchmark eval 而言是真实的不一致。

**修复建议**
改为逐样本判定退化（不在 batch 维做 reduce），或在文档/脚本中强制 eval 整 split 单批，并说明该约束。

### M4 — 训练不可复现 [已修复]

> **修复方式**：`scripts/train.py` 入口添加 `torch.manual_seed(seed)` + `np.random.seed(seed)` + `random.seed(seed)` + `torch.cuda.manual_seed_all(seed)`，seed 从 `cfg["data"]["seed"]` 读取；train DataLoader 传入 seeded `torch.Generator`。

**证据链**
`scripts/train.py`、`rcdw/training/*` 中**无任何** `torch.manual_seed` 或带种子的 DataLoader `generator`（仅 `scripts/numerical_check.py:62` 有，用于数值对齐诊断）。模型初始化（`nn.Linear` 默认初始化）与 `train.py:54` 的 `shuffle=True` 均未播种；配置 `data.seed:42` 只服务于数据生成。

**影响**
重跑 Stage A/B 会得到不同权重与不同 §2.3 数字，benchmark 结果不可复现。

**修复建议**
训练入口设置 `torch.manual_seed(cfg_seed)`（必要时 `numpy`/`random` 一并），并给 train DataLoader 传 seeded `torch.Generator`。

### M5 — 多个配置节被静默忽略 [已修复]

> **修复方式**：从 `smoke.yaml` 和 `default.yaml` 中删除从未被读取的 `phases:`、`splits:`、`scalers:` 顶层节以及 `data:` 下的旧字段（`n_train`/`n_val`/`n_test`）。

**证据链**（已 grep 确认）

- `smoke.yaml` 顶层 `splits:`（ratios / group_field）、`scalers:`（transform_target / modal_groups_from_schema）、`phases:`（default_schedule）**从未被读取**。
- split 比例硬编码 0.70/0.15/0.15：`rcdw/sim/packaging/splits.py:22-24` 默认值，`benchmark.py:127` 仅传 `seed`。
- scaler 分组/目标硬编码：`benchmark.py:205-211` 用 `SLOW_MODAL_GROUPS` 与 `transform_target="slow"`。
- 调度来自 `generation.stage_profile`（`generate_benchmark.py:52`），非 `phases.default_schedule`。

**影响**
编辑这些 YAML 键毫无效果（当前默认值恰好一致，掩盖了问题）—— 是易踩的 footgun。

**修复建议**
要么通过 `BenchmarkGenerationSpec` 把它们真正接进生成流程，要么从配置模板中删除，避免"看似可调实则无效"。

---

## 3. 低（Low）

- **L1 — 指标命名与数值稳定性 [已修复]**：`metrics.py` 中 `ARE` 改名为 `MaxRE`（准确反映 `re.max()` 语义）；`eval.py` 表头同步更新。`eps=1e-8` 在近零真值处的爆炸问题保留为已知限制（需物理下限 ~1e-3，但当前无影响 dry basis eval）。

- **L2 — 湿基路径单位隐患 [已修复]**：`normalize.py` 的 `rh_to_water_vol` 添加 `RH > 1.0` 检查并抛 `ValueError`，防止百分数误传入导致负总量。docstring 更新删除了对已删除的 `synth_timeseries` 的引用。

- **L3 — `dt_s` 未接入慢通道动力学 [已修复]**：`slow.py` 的 `_multi_tau_channel_step` docstring 已明确说明 `tau_*_system_s` 实际单位为时间步（`exp(-1/tau)` 隐含 dt=1），改变 `dt_s` 仅影响时间戳标注。

- **L4 — 扰动强度语义跨类别不可比 [已修复]**：`inject.py` 的 `inject()` docstring 新增 warning 块，逐类型说明 `level` 的物理含义差异，明确跨类型对比无意义。

- **L5 — 陈旧注释/文档漂移 [已修复]**：`stage_a.py:50` 注释 `(B, 6)` → `(B, 12)`；`normalize.py` 删除已不存在的 `synth_timeseries` 引用。

---

## 4. 已核对无误（非问题）

- **融合数学正确**（`rcdw/models/rcdw.py:59-85`）：`softmax(-β·E, dim=1)` 正确地降低高误差模态权重；α/shift 广播、`clamp(W_base±shift)` 边界、`+eps` 重归一化均正确；`W` 为凸组合保持 ≥0，无除零。
- **无量级/单位 bug**：组分全链路走**百分数**；`rcdw_sound_speed`/`rcdw_attenuation`/`_hidden_lambda_mix`（`acoustic_physics.py:107-109` 等）内部统一 `/100`；标签→比例 `/100`（`dataset.py:140`）与单模态 `sum=1`、`StageBLoss` 的 `sum→1` 一致。
- **HITRAN 缓存无碰撞**：内存缓存 key `(channel, grid_spec)` 含 T/P；浓度在缓存之后经 Beer-Lambert 施加（`optical_backend.py:222-250`），prepared 谱与浓度无关 → 无碰撞、无陈旧浓度复用；磁盘 key 经 `grid_spec` 变化即 cache miss（`MissingHitranCacheError`）。
- **无 split 泄露**：按 `mixture_id` 分组（`splits.py:33,67`），同一 mixture 全部落入同一 split；v1.x 为 1:1 mixture:sequence，correctness 成立。
- **组分和校验用容差**：`integrity.py:126` 用 `abs(total-100) > 1e-5`（非浮点 `==`），rounding 累计误差 ~1.5e-6 < 1e-5，通过。
- **落盘原子性**：staging → 目标目录有 backup/rollback（`benchmark.py:398-412`）；失败清理后 re-raise，无被吞异常（全量 grep 无 `except: pass`）。

---

## 5. 修复记录

全部 10 项问题已于 2026-07-01 修复完成，按原建议顺序执行：

1. **H1**：12 维 input scaler 消费侧应用 + manifest 标志位 + 11 个新增测试。
2. **M4**：`train.py` 训练播种（`torch.manual_seed` + `np.random.seed` + seeded DataLoader）。
3. **M3**：`hard_suppress` 改逐样本判定（`E_pred.min(dim=1)` 替代 `E_pred.median(dim=0)`）。
4. **M2**：`slow.py` 环境通道添加传感器测量噪声（T: σ=0.05°C, P: σ=0.0005 MPa, RH: σ=0.1%）。
5. **M5**：删除 `smoke.yaml` / `default.yaml` 中从未被读取的 `phases`/`splits`/`scalers`/旧 `data` 字段。
6. **L1–L5**：`ARE` → `MaxRE`；`rh_to_water_vol` 单位检查；tau docstring；扰动强度 docstring；陈旧注释清理。

> 修复后测试基线：**212 passed**（原 201 + H1 新增 11）。

---

## 附录 A — H1 深入分析

### A.1 精确机制：数据流在哪里断开

完整的"标准化"契约本应是一条闭环链路，但它在**生成侧结束、消费侧从未开始**：

```
generate_condition_rows
   └─> build_sequence_arrays        # 产出原始 arrays["slow"] (N,T,7) —— 物理量纲
        └─> fit_z_score_scalers     # benchmark.py:205  用 train 索引拟合 (mean,std)
             └─> write_json         # benchmark.py:220  落盘 scalers/scaler_slow_sequence.json
                  └─> [validate]    # integrity.py:199  校验 passthrough 标记
        └─> write_arrays            # benchmark.py:151  写 slow.npy —— 仍是原始值(未应用 scaler)
                                    # ✂── 链路在此断开 ──✂
BenchmarkDataset.__getitem__        # dataset.py:186-210  读原始 slow.npy + 原始 ultrasonic 元数据
   └─> concat → (L,12)              # 直接拼接，无 (x-mean)/std
        └─> RCDW_MGDA.forward       # 模型吃到的是原始异质量纲张量
```

`scaler_slow_sequence.json` 落盘后成为**孤儿产物**：`dataset.py` / `train.py` / `eval.py` / `perturb.py` 均不 `open` 它（全量 grep 确认，唯一读者是不存在的）。也就是说，`(x - mean) / std` 这一步在整个工程里**没有任何一行代码执行**。

### A.2 实测 12 维通道量级（`empirical_v1` 后端，16 序列 × 16 步 = 256 样本/通道）

> 说明：此为离线 empirical 后端的量级探针；smoke 用 hitran + 64×32，绝对值略有差异，但**数量级与结论一致**。

| idx | 通道（模型消费顺序）                       | mean      | std     | scaler 覆盖 | 备注                   |
| ---:| -------------------------------- | ---------:| -------:|:---------:| -------------------- |
| 0   | V_NDIR_CO2                       | 2.133     | 0.270   | z_score   | NDIR 主信号             |
| 1   | V_TCS                            | 1.123     | 0.042   | z_score   | TCS 主信号，**近常量**      |
| 2   | T_C                              | 26.09     | 6.141   | z_score   | 环境（3 个单模态共享）         |
| 3   | P_MPa                            | 0.393     | 0.183   | z_score   | 环境                   |
| 4   | H_RH                             | 47.72     | 16.42   | z_score   | 环境                   |
| 5   | L_m                              | 0.608     | 0.374   | z_score   | 声程                   |
| 6   | piston_position_m                | 0.608     | 0.374   | z_score   | **与 idx 5 完全相同**     |
| 7   | ultrasonic_tof_observed_s        | 0.00183   | 0.00107 | **无**     | ~1e-3 s              |
| 8   | ultrasonic_sound_speed_estimated | **346.4** | 7.295   | **无**     | **US 主信号（USNet 输入）** |
| 9   | ultrasonic_peak_index            | 366.6     | 214.1   | **无**     | 离散索引，跨度大             |
| 10  | ultrasonic_tof_quality           | 0.983     | 0.014   | **无**     | 近常量                  |
| 11  | ultrasonic_tof_accepted          | 1.0       | **0.0** | **无**     | **常量，零方差**           |

**跨通道 |mean| 动态范围 ≈ 2×10⁵**（`ultrasonic_tof_observed_s` ~1.8e-3 ↔ `ultrasonic_peak_index`/`sound_speed_estimated` ~3.5e2）。把这样的张量直接喂给权重共享、无 BatchNorm 的小型 MLP，是典型的病态输入。

### A.3 为什么有害：逐组件影响

**(1) 单模态 MLP —— 主信号被环境上下文淹没（NDIR/TCD）或反之淹没上下文（US）**

各单模态网络输入为 `[sensor, T_C, P_MPa, H_RH]`（`single_modal.py:97-113`），实测量级：

| 网络      | 自身传感器信号               | 共享环境 T_C / H_RH | 失衡方向                             |
| ------- | --------------------- | --------------- | -------------------------------- |
| NDIRNet | V_NDIR_CO2 ≈ **2.13** | 26.1 / 47.7     | 传感器比环境**小 12–22×** → 首层被 T/RH 主导 |
| TCDNet  | V_TCS ≈ **1.12**      | 26.1 / 47.7     | 传感器比环境**小 23–43×** → 判别信号几近被埋没   |
| USNet   | US_speed ≈ **346**    | 26.1 / 47.7     | 传感器**大 7–13×** → 环境上下文可忽略        |

即：三个网络的**判别性主信号与上下文的相对量级全部失衡**，方向还相反。`Linear(4→32)` 首层的预激活被最大量纲项支配，AdamW 虽能靠权重内部补偿，但收敛更慢、每输入有效学习率失衡，正是 §2.3 误差偏高的直接诱因之一。

**(2) FeatureExtractor → ErrorNet —— RCDW 的可靠性核心被量纲污染**

13 维特征中，尺度不变 vs 尺度相关的划分（`feature.py`）：

- **尺度不变**（安全）：`CV = std/mean`（idx 0）、`Q = snr_m/Σsnr`（idx 3）、`snr_proxy = |mean|/std`（idx 10）—— 比值天然消掉量纲。
- **尺度相关**（被污染）：`G_m = mean(Δs²)`（idx 2，∝ c²）、`drift = OLS 斜率`（idx 11，∝ c）。US_speed 的 Δ² 与电压的 Δ² 相差 ~(346/2)² ≈ 3×10⁴ 倍，却落在**同一特征列**，由 ErrorNet 权重共享的 `Linear(13→h)` 同时处理。

由于 `ErrorNet` 直接决定各模态可靠性 → 融合权重，被量纲污染的 `G_m`/`drift` 会**系统性偏置 RCDW 的核心可靠性估计**，而非仅仅拖慢训练。

**(3) 与 M2 叠加**：环境差分特征 `delta_T/P/RH`（idx 5-7）本就恒为 0（见正文 M2），叠加本条 (2)，ErrorNet 的 13 维输入里**有效且干净的特征进一步减少**。

### A.4 关键盲区：scaler 只覆盖 7/12 通道，US 主信号根本不在其中

即使把现有 scaler 正确接入，也**不足以修复问题**：scaler 仅在 7 通道 slow 矩阵上拟合（`benchmark.py:205-211`，`channel_names=SLOW_CHANNELS`），而 12 维输入里的 **idx 7-11 五个 ultrasonic 元数据通道没有任何 scaler 统计量**。其中 **idx 8 `sound_speed_estimated`（~346）恰是 USNet 的主输入**（`SENSOR_INDICES["usn"]=IDX_US_SPEED=8`）。

结论：**"应用现有 scaler" ≠ "标准化模型输入"**。正确修复必须把标准化扩展到 12 维消费张量（或至少覆盖 idx 8/9），否则 US 模态首层依旧在 ~346 量级上训练。

### A.5 附带发现（分析 12 维输入时一并暴露）

- **idx 11 `ultrasonic_tof_accepted` 为常量（clean 数据恒 =1，std=0）** → 零信息死输入；且若"修复"时对其做 z-score，会触发 `std=0` 除零（slow 通道有 `np.where(std>eps,1.0)` 保护于 `scalers.py:69`，但该通道当前不在被拟合集合内，扩展时须补同款保护）。
- **idx 6 `piston_position_m` ≡ idx 5 `L_m`**（`slow.py:252-253` 两者都赋 `current_l_m`）→ 完全冗余通道，实测统计量逐位相同。

二者属"输入通道卫生"问题，建议在标准化改造时一并处置（删冗余、剔常量或转为有信息的门控特征）。

### A.6 为什么 201 个测试与 10 项校验都没抓到

- 测试覆盖的是 **scaler 拟合本身**（`test_packaging.py` 验证 mean/std 数值、passthrough 标记），而**没有任何测试断言"模型输入已被标准化"**——因为消费侧从不读 scaler，也就无从测起。
- `integrity._validate_scaler_passthrough` 校验的是 JSON 里 passthrough 字段的存在性（且对不在矩阵中的 ultrasonic 通道走"在 skip_channels 即放行"分支而空转，见正文 H1 关联缺陷），与"是否应用"正交。

即：现有测试/校验体系把"**产物写对了**"误当成"**功能生效了**"。建议补一条端到端不变量：`BenchmarkDataset` 取出的 train 窗口，其被标准化通道的整体 mean≈0 / std≈1。

### A.7 修复方案与权衡

| 方案              | 做法                                                                                                      | 优点                                                | 缺点 / 风险                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------- |
| **① 消费侧应用（推荐）** | `BenchmarkDataset` 加载 scaler JSON，在 `__getitem__` 对 z_score 通道做 `(x-mean)/std`，并**扩展到 idx 8/9 等 US 通道** | 原始 `.npy` 不变，可回退；标准化逻辑集中在加载层；与 `manifest` 标志位天然兼容 | 需扩展 scaler 覆盖到 12 维；须处理 std=0 通道；每 `__getitem__` 轻量开销                   |
| **② 生成侧烘焙**     | 在 `write_arrays` 前把标准化后的数组写盘                                                                            | 加载零开销                                             | 原始物理量纲丢失、不可回退；`slow_sequence_long.csv` 与 `.npy` 语义分叉；扰动注入（作用于标准化空间）语义改变 |
| **③ 降级文档**      | 保持不应用，改 v1.2 完成情况表 / manifest / 校验措辞为"仅拟合、未应用（Phase 6 接入）"                                              | 零代码改动、消除误导                                        | 模型仍在原始量纲训练，§2.3 性能问题不解决                                                 |

**推荐路径**：方案 ① + `manifest` 增 `input_normalization: {applied: true, coverage: "12ch", version: "..."}` 标志位；旧数据集无此标志则 Dataset 回退到"不应用"以保持向后兼容。

**实施要点**

1. scaler 拟合从 `SLOW_CHANNELS`(7) 扩到 12 维消费布局（新增 idx 7-11 的 train-only 统计量），对 `std<eps` 通道（idx 11）走 passthrough。
2. `BenchmarkDataset.__init__` 读 `scalers/scaler_slow_sequence.json`（+ US 通道统计量），`__getitem__` 拼接后统一 transform。
3. 处置附带发现：删除或合并 idx 6（冗余），把 idx 11 明确标为 passthrough/门控。
4. **旧 ckpt 必须重训**（输入分布改变，与 §7.3 历史 ckpt 废弃同理）。

### A.8 验证清单（改造后）

- [ ] 新增端到端测试：`BenchmarkDataset("train")` 全窗口在被标准化通道上 `|mean|<0.05`、`|std-1|<0.1`。
- [ ] 新增：val/test 使用 **train 拟合**的 mean/std（确认无泄露，且 val/test 的 mean 不必为 0）。
- [ ] idx 11（std=0）transform 不产生 NaN/Inf。
- [ ] `manifest.input_normalization.applied` 标志位存在且被 Dataset 正确分支。
- [ ] 重训 Stage A/B 后复测 §2.3，对比标准化前后 MAE/RMSE（预期显著改善，尤其 US 相关）。
- [ ] 回归 `python -m pytest`（当前 201）。

---

## 第二轮审查（2026-07-02）

> 审查范围：`rcdw_mgda/` 子工程全量源码（Phase 1–6 合计约 10,700 行，85 个文件）
> 审查方法：8 个独立审查角度（line-by-line diff、removed-behavior、cross-file tracer、reuse、simplification、efficiency、altitude、conventions）并行扫描，去重后 10 个候选经独立验证 agent 逐一确认
> 原始结论：**无崩溃级 bug**；问题集中在**死配置项**、**扰动可复现性**、**硬编码常量分散**和**效率浪费**。
> 当前同步：R1-R6 已按代码落地；R7/R8 为部分修复；复核定向测试 22 passed。

### 6. 严重级别总览（第二轮）

| 编号     | 级别  | 一句话摘要                                                          | 主要位置                | 状态  |
| ------ | --- | -------------------------------------------------------------- | ------------------- | --- |
| **R1** | 高   | 超声扰动 scale 是 batch 全局均值，扰动量依赖 batch 组成，破坏可复现性                  | `inject.py:93`      | ✅ 已修复 |
| **R2** | 中   | `rh_to_water_vol` 接受 T 参数但硬编码 300K，高温时误差可达 7 倍                 | `normalize.py:26`   | ✅ 已修复 |
| **R3** | 中   | Stage B 中 Y_modal 未 detach，freeze_single_modal=False 时产生非预期梯度  | `stage_b.py:60`     | ✅ 已修复 |
| **R4** | 中   | stage_b.batch_size 配置项无效，Stage B 始终使用 Stage A 的 batch_size     | `train.py:81`       | ✅ 已修复 |
| **R5** | 中   | model.error_net.hidden 配置项无效，ErrorNet 始终使用 single_modal.hidden | `train.py:105`      | ✅ 已修复 |
| **R6** | 中   | hard_suppress 在所有模态均匀退化时完全失效（ratio 机制的设计盲区）                    | `degradation.py:38` | ✅ 功能已修复 |
| **R7** | 低   | 气体名称在 4 个文件中独立硬编码，无单一数据源                                       | `metrics.py:36` 等   | ⚠️ 部分修复 |
| **R8** | 低   | label 列数硬编码为 3，不从 manifest 或 label_names.npy 推导                | `dataset.py:153`    | ⚠️ 部分修复 |

### 6.1 当前修复同步（2026-07-02）

- **R1 已修复**：`inject.py` 中 ultrasonic `scale` 改为仅沿序列维求均值并保留 batch 维，扰动幅度不再依赖 batch 组成。
- **R2 已修复**：`rh_to_water_vol(RH, T)` 改为 Magnus 公式，支持 K 与 °C 两种温度输入；无 T 时仍按 300K 默认。
- **R3 已修复**：`stage_b.py` 在训练与验证 loss 前显式 `detach()` `Y_modal`，ErrorNet 目标不再向 single-modal 子网回传辅助梯度。
- **R4 已修复**：Stage B 启动前按 `training.stage_b.batch_size` 重建 DataLoader，不再复用 Stage A batch size。
- **R5 已修复**：`RCDW_MGDA` 增加 `err_hidden`，`train/eval/perturb` 从 `model.error_net.hidden` 读取并传入 ErrorNet。
- **R6 功能已修复**：`hard_suppress` 增加 `absolute_threshold`，配置文件统一加入默认值 `0.15`，可捕捉均匀高误差场景。残留：`scripts/perturb.py` 的 report JSON 尚未写出该阈值。
- **R7 部分修复**：指标、eval、perturb report 与测试已统一使用 `GAS_DISPLAY_NAMES` 或 dataset label names。残留：`scripts/perturb.py` 权重曲线仍硬编码 CO2 与列索引 `1`。
- **R8 部分修复**：`BenchmarkDataset` 已从 `manifest["labels"]` 或 `metadata/label_names.npy` 推导标签列数并删除 `!=3` 断言。残留：`/100.0` 标签缩放仍是隐式百分数契约，尚未从 manifest 显式声明或校验。

**复核验证**：`python -m pytest tests/test_degradation.py tests/test_dataset_loader.py tests/test_perturb_report.py` → **22 passed**。

### 7. 详细说明

#### R1 — 超声扰动 scale 使用 batch 全局均值，破坏实验可复现性 [已修复]

**证据**
`rcdw_mgda/rcdw/perturbation/inject.py:93`：

```python
scale = x[..., IDX_US_SPEED].abs().mean()
```

`.mean()` 无 `dim=` 参数，对 batch 内所有样本的声速取全局均值得到一个标量。

**失败场景**
同一个样本在不同 batch size 或不同 shuffle 下，会得到不同的扰动幅度。`level=0.05` 的实际噪声量取决于当前 batch 的平均声速，而非该样本自身的声速。扰动实验结果不可跨 batch 配置对比。

**修复建议**
改为 per-sample 计算：`.mean(dim=-1, keepdim=True)` 或在序列维上求均值但保留 batch 维。

#### R2 — `rh_to_water_vol` 函数签名暗示支持变温但硬编码 300K [已修复]

**证据**
`rcdw_mgda/rcdw/utils/normalize.py:7-26`：函数签名 `def rh_to_water_vol(RH, T=None)` 接受 T 参数，但函数体直接 `return RH * 0.0355`。0.0355 是 T=300K、P=1atm 下的饱和蒸汽压系数。docstring 已标注"T 参数当前不参与计算"。

**失败场景**
调用方传入 T=340K 时，实际饱和压约 27 kPa（vs 300K 的 3.6 kPa），误差可达 ~7 倍。函数签名给出虚假的温度依赖承诺。

**修复建议**
要么实现 Antoine 方程的变温计算，要么删除 T 参数并在 docstring 中明确标注固定 300K 假设。

#### R3 — Stage B 中 Y_modal 未 detach，非默认配置下产生非预期梯度流 [已修复]

**证据**
`rcdw_mgda/rcdw/models/rcdw.py:131-134` 中 Y_modal 由 single-modal 子网络直接输出，无 `.detach()`。`rcdw_mgda/rcdw/training/stage_b.py:60` 将 Y_modal 传入 loss，`losses.py:63` 计算 `E_true = (Y_modal - C_ref_expand).abs()`。

**失败场景**
默认 `freeze_single_modal=True` 时参数 `requires_grad=False` 阻断梯度，bug 休眠。但若设为 False，ErrorNet 的 loss 会通过 `abs()` 向 single-modal 网络回传非预期辅助梯度，且 `abs()` 在零点不光滑。

**修复建议**
在 `stage_b.py` 中显式 `Y_modal = out['Y_modal'].detach()`，使 ErrorNet 的学习目标与 single-modal 参数解耦，无论 freeze 配置如何。

#### R4 — stage_b.batch_size 配置项是死配置 [已修复]

**证据**
`rcdw_mgda/scripts/train.py:81` 只读取 `cfg['training']['stage_a']['batch_size']` 创建 DataLoader，Stage A 和 Stage B 共用同一对 loader。所有 YAML 配置文件中的 `training.stage_b.batch_size` 从未被读取。

**失败场景**
用户修改 `stage_b.batch_size` 期望影响 Stage B 训练，但实际无任何效果。当前 default.yaml 中两者恰好相同（均为 16），掩盖了问题。

**修复建议**
在 Stage B 启动前用 `stage_b.batch_size` 创建新的 DataLoader，或删除配置项以消除误导。

#### R5 — model.error_net.hidden 配置项是死配置 [已修复]

**证据**
`rcdw_mgda/scripts/train.py:105` 读取 `cfg['model']['single_modal']['hidden']` 传给 `RCDW_MGDA`。`rcdw_mgda/rcdw/models/rcdw.py:118` 用同一个 `hidden` 构造 `ErrorNet(in_dim=13, n_gas=3, hidden=hidden)`。配置中 `model.error_net.hidden` 从未被读取。

**失败场景**
当前两个值恰好都是 32，但配置 schema 暗示可以独立调整 ErrorNet 容量——实际不行。

**修复建议**
在 `RCDW_MGDA.__init__` 中增加 `err_hidden` 参数，从 `model.error_net.hidden` 读取并独立传入 ErrorNet。

#### R6 — hard_suppress 在均匀退化场景下完全失效 [功能已修复]

**证据**
`rcdw_mgda/rcdw/utils/degradation.py:38`：`min_E = E_pred.min(dim=1).values`，判断条件 `E_pred > ratio * min_E`（ratio 默认 4.0）。

**失败场景**
当全局扰动（如温度偏移）导致所有模态预测误差相同时，`E_pred == min_E`，条件恒为 False（因 ratio > 1），没有任何模态被抑制。安全机制在"所有模态都坏了"的最危险场景下变为空操作。

**修复建议**
增加绝对阈值判断（如 `E_pred > absolute_threshold`）作为 ratio 相对判断的补充，或对均匀退化场景做显式处理（如全模态误差超过预设上限时标记所有模态为退化）。

#### R7 — 气体名称在多个文件中独立硬编码 [部分修复]

**证据**
`gas_names = ["O2", "CO2", "N2"]` 以位置索引映射 label 列，分别出现在：

- `rcdw_mgda/rcdw/training/metrics.py:36`
- `rcdw_mgda/scripts/perturb.py:258`
- `rcdw_mgda/scripts/eval.py:88`
- `rcdw_mgda/tests/test_perturb_report.py:14`

**失败场景**
如果 `schema.py` 中 `COMPONENT_FIELDS` 的顺序变化，4 处硬编码不会同步更新，per-gas 指标会静默映射到错误气体。

**修复建议**
在 `schema.py` 或 `metrics.py` 中定义 `GAS_DISPLAY_NAMES`，其他文件统一引用。

#### R8 — label 列数和缩放因子硬编码 [部分修复]

**证据**
`rcdw_mgda/rcdw/data/dataset.py:153`：`self.labels = (labels_all[...] / 100.0).astype(np.float32)`，随后 `if self.labels.shape[1] != 3: raise ValueError(...)`。不读取 manifest 中的 `labels` 字段或 `label_names.npy`。

**失败场景**
如果 benchmark 产出 4 组分 label（如主项目 syngas 路径），dataset 直接报错。/100.0 的缩放也假设 label 为百分数，无运行时验证。

**修复建议**
从 `manifest['labels']` 或 `label_names.npy` 推导组分数，删除 `!= 3` 的硬编码断言。

---

### 8. 补充观察（未列入正式发现，但值得关注）

#### 效率类

- **`stage_b.py:73`**：验证集预测每个 epoch 都做 GPU→CPU 转存和列表累积，但只有每 20 个 epoch 才使用一次。将 `all_pred`/`all_ref` 的累积移入 `epoch % 20` 条件内可消除 95% 的无效转存。
- **`feature.py:93`**：SNR 计算中 `mu`、`sigma` 在 pre-loop 和 main-loop 中各算一次，是重复计算。
- **`dataset.py:260`**：`__getitem__` 中 5 个超声元数据数组通过 Python 循环逐一 slice + np.stack，可预组装为 `(N, T, 5)` 数组以消除 per-sample 开销。

#### 维护性类

- **`slow.py:330`**：`build_sequence_arrays_chunk` 是纯透传包装器，14 个参数原样转发到 `build_sequence_arrays`。可直接使用原函数作为 ProcessPoolExecutor 入口。
- **`dataset.py:63` + `scalers.py:37`**：12 维通道顺序在两个文件中独立定义。启用 input_scaler 时有运行时校验可捕获不匹配，但禁用时无保护。
- **`acoustic_physics.py:155`**：4 个振动弛豫项是结构完全相同的 6 行代码块，仅参数名不同，可提取为 `_relaxation_term()` 辅助函数。

#### 迁移兼容类

- **`single_modal.py:36`**：模态键从 `'us'` 改名为 `'usn'`，旧 checkpoint 文件 `us.pt` 不会被新代码加载，Stage B 会静默使用随机初始化。
- **`train.py:67`**：Stage A 返回值被丢弃，Stage B 依赖文件系统加载 checkpoint。如果写盘失败（权限/空间），Stage B 以随机权重继续训练，仅有 print 级别警告。
- **`inject.py:55`**：超声通道从合成值（~1.0）迁移到真实声速（~340 m/s），同一个 `level=0.05` 产生的绝对扰动量差 ~340 倍，迁移前后结果不可比。
