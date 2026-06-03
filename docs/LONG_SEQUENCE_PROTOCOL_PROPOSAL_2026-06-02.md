# 长时序实验协议改进方案（时间步与阶段划分重构）

- 日期：2026-06-02
- 范围：`src/sim`（数据生成时间轴与阶段调度）、`src/dl`（模型感受野与时序聚合）、评估与数据增强
- 状态：提案（待评审 / 分阶段落地）
- 关联文档：`docs/ARCHITECTURE.md`、`docs/IMPLEMENTATION_PLAN.md`

---

## 0. TL;DR

当前 benchmark 每条序列只有 **128 个时间步 × 0.5 s = 64 s**，按**固定四分位**切成 `baseline / exposure / steady / recovery` 四段（各 32 步），相变点恒定写死在 `t = 0 / 32 / 64 / 96`。这套设计存在三个互相叠加的问题，导致**深度学习在长时序上的优势无法体现**：

1. **序列太短**：64 s、128 步，远低于长序列模型（TCN/Transformer/SSM）发挥优势所需的尺度，长程依赖近乎平凡。
2. **阶段点固定**：相变永远在同样的索引，模型无需"定位事件"，时间结构退化为可被静态特征替代的信息。
3. **模型侧把时序抹平**：`CNN1DRegressor` 与 `TCNRegressor` 末端都用 `AdaptiveAvgPool1d(1)` 全局平均池化；且当前 TCN 感受野仅 **29 步（占序列 23%）**。两者叠加 → 时间顺序被平均掉 → 传统 ML ridge（140 维聚合特征）能打平甚至接近 DL（实测 `train R²=0.93`）。

本方案提出四组改进（**P1 时间轴可配置化 → P2 阶段调度抽象+随机化 → P3 更真实的瞬态动力学 → P4 模型与评估升级**），并给出文件级落点、接口草图与向后兼容策略，使数据集进入"**长序列 + 事件需定位 + 长程依赖非平凡**"的区间，从而让长时序模型的优势变得**可测量、可复现**。

---

## 1. 现状（代码事实核对）

| 维度 | 现状 | 代码位置 |
|---|---|---|
| 时间步数 | `timesteps = 128`（默认，可传参但默认值固定） | `src/sim/generation/benchmark.py:50` |
| 采样间隔 | `dt_s = 0.5`，总时长 64 s | `src/sim/generation/benchmark.py:51` |
| 阶段边界 | 写死四分位 `q1=T//4, q2=T//2, q3=3T//4` | `src/sim/generation/phases.py:4-10` |
| 阶段名 | `baseline / exposure / steady / recovery`（4 段等长） | `src/sim/core/schema.py:23`、`phases.py:13-21` |
| 浓度混合曲线 | 梯形 `blend`：0 → 线性升 → 1 平台 → 线性降 | `phases.py:24-33` |
| 传感器动力学 | 一阶：`tau_rise` 上升 / `exp(−Δt/tau_decay)` 衰减（**单时间常数**） | `src/sim/generation/slow.py`（`_channel_value`） |
| stage_profile | 字段存在但**只有 `standard_exposure` 一种** | `benchmark.py:54`、`sequence_index.csv` |
| 模型时序聚合 | `AdaptiveAvgPool1d(1)` 全局平均 | `cnn1d.py:42`、`tcn.py:83` |
| TCN 感受野 | `channels=[32,64,64] → dilations=(1,2,4), k=3 → RF=29` 步 | `tcn.py:71-74,91-93` |
| 注册模型 | 仅 `cnn1d`、`tcn`（`LSTM/Transformer` 未实现） | `dl/models/registry.py:11-12` |

> 实测验证：`data/wv4-smoke` 每条序列恰好 `baseline:32 / exposure:32 / steady:32 / recovery:32`，相变在 `timestamp_s = 0/16/32/48`。

---

## 2. 问题诊断（为什么"体现不出 DL 优势"）

### 2.1 序列太短，长程依赖平凡
长序列建模文献的共识是：**深度时序模型的优势只有在序列足够长、且依赖关系非平凡时才显现**。Bai et al. 的 TCN 评测显示，在需要长有效历史的任务（copy-memory `T>200`、permuted-MNIST 784 步）上，TCN 远胜 LSTM/GRU，而 RNN 的"无限记忆"在实践中基本不存在 [1]。长序列预测（LSTF）整条研究线（Informer/Autoformer/FEDformer/PatchTST/TimesNet）正是为 **`L` 量级数百~数千** 的输入而生 [2][3][4]。64 s / 128 步的序列处于"短序列"区间，本质上是一次准静态测量，无法逼出长时序模型的能力差。

### 2.2 阶段点固定 → 无需"定位事件"
相变永远在 `t=32/64/96`。模型可以直接"背索引"，无需从信号里**定位** baseline/exposure/steady/recovery 的边界。一旦相变时刻随机化，模型必须在时间轴上**搜索并对齐事件**——这正是注意力 / 长程卷积 / 状态空间模型相对于"全局统计特征 + ridge"的核心增量能力。

### 2.3 模型侧把时间顺序抹平（最关键）
`CNN1DRegressor` 和 `TCNRegressor` 的 forward 都是 `encoder → AdaptiveAvgPool1d(1) → head`。全局平均池化对时间维做**平均**，对"先升后降"还是"先降后升"几乎不敏感——也就是说当前 DL 模型在全局尺度上**近似时序顺序不变**。再叠加 TCN 感受野仅 29 步（< 一个完整瞬态 96 步），模型连"跨越整个 baseline→recovery 过程"的局部窗口都没有。结果：时序结构提供的额外信息很少被利用，于是 140 维聚合特征的 ridge 能与 DL 打平（实测 `ridge train R²=0.93`），**DL 的序列建模优势在当前任务设定下没有用武之地**。

### 2.4 单一 stage_profile + 单时间常数 → 缺乏多样性与真实感
真实气体传感的响应/恢复是**慢过程（分钟级）**、且通常是**多时间常数**而非单一指数，并常伴随**基线漂移 / 不完全恢复**和**记忆效应** [9][10][11]。当前仿真用单一梯形 + 单 `tau` 一阶动力学、且只有一种协议，既不够真实，也不足以制造"必须看长上下文才能解决"的难度。

---

## 3. 文献依据

### 3.1 长时序模型谱系（"优势在哪、需要多长"）
- **TCN（膨胀因果卷积）**：用指数增长的膨胀率得到指数级感受野，`RF = 1 + Σ 2·(k−1)·d_i`；通过加深/加大膨胀可灵活匹配所需历史长度，且"感受野应 ≥ 输入序列长度" [1][12]。
- **Transformer-LSTF**：Informer（ProbSparse，`O(L log L)`）、Autoformer（序列分解 + 自相关）、FEDformer（频域）、**PatchTST**（分块 + 通道独立，支持更长回看窗口）、**TimesNet**（按周期 1D→2D）等，专为长序列设计 [2][3][4]。
- **状态空间模型（SSM）**：S4 / **Mamba**（选择性 SSM，线性复杂度，可扩展到百万级长度、推理吞吐 5×）；S-Mamba、Mamba4Cast、Mambaformer 把它用于时序 [5][6]。
- **重要警示（DLinear）**：Zeng et al. 指出在若干 LTSF 基准上，一个简单线性模型可超过复杂 Transformer [2][3]。**启示**：改进必须配强 baseline（线性 / ridge / 全局池化 CNN），否则无法证明"长时序模型确有优势"。

### 3.2 气体传感时序的领域依据（如何把序列做"长且有信息"）
- **瞬态富含判别信息**：e-nose 的"全特征"算法显式利用**响应 + 平衡 + 恢复**全过程，并发现**解吸/恢复段**对区分贡献更大；采用 1D-CNN + RNN + 通道/时间注意力 [7]。
- **早期（未达平衡）识别**：CLSTM 在**达到平衡之前**的瞬态段即可高精度识别气体，说明瞬态段、而非仅稳态值，承载关键信息 [8]。
- **TCN 用于 E-nose 浓度预测**：TF-TCN 用膨胀因果卷积 + 时频结合做气体浓度回归，优于 LSTM/GRU/普通 TCN [13]。
- **慢响应 / 多时间常数 / 基线漂移**：MOx 传感器对阶跃浓度的响应时间可达数分钟，响应/恢复曲线常需**多个时间常数**拟合（扩散 + 吸附/解吸多机制），且室温下解吸困难会造成**基线漂移（不完全恢复）** [9][10][11]。物理信息 + GRU 的混合方法表明，瞬态过程导出的参数具有**规律性与稳定性**，有助于泛化 [9]。

### 3.3 可变协议与数据增强
- **时序数据增强综述**：窗口切片（window slicing）、窗口扭曲（window warping，随机段拉伸/压缩）、抖动、置换、cutout/cutmix/mixup 等；其中**窗口切片与窗口扭曲**在 CNN/RNN 上最稳健有效 [14][15][16]。
- **变长序列**：TCN/RNN/Transformer 天然支持变长输入（padding + mask）；窗口扭曲会改变序列长度，需与切片配合 [1][15]。
- **启示**：把"阶段时刻/时长随机化"既是**更真实的实验协议**，又等价于内建的强数据增强，能逼模型学"定位 + 对齐"而非"背索引"。

---

## 4. 改进方案

> 设计原则：**配置驱动、向后兼容、可证伪**。所有默认值保持现状可复现；新能力通过显式配置开启；改进效果以"长序列模型 vs 强 baseline 的可测量差距"来验收（§6）。

### P1. 时间轴可配置化并拉长

**目标**：让 `timesteps` / `dt_s` / 总时长成为一等配置项，并提供"长序列档位"。

- `BenchmarkGenerationSpec` 已有 `timesteps`、`dt_s` 字段（`benchmark.py:50-51`），**默认值过小**。新增长序列预设档位，例如：
  - `short`（兼容现状）：`T=128, dt=0.5`（64 s）
  - `standard`：`T=512, dt=0.5`（256 s）
  - `long`：`T=1024, dt=0.5`（512 s）
  - `xlong`：`T=2048`（用于压力测试长程模型）
- 这些档位作为命名 spec 工厂或 Hydra `configs/data/*.yaml`，不改协议 `schema`。

**依据**：§3.1 长序列模型需要数百~数千步；§3.2 真实瞬态分钟级，64 s 偏短。

---

### P2. 阶段调度抽象化 + 随机化 + 多 profile（核心）

**目标**：把"固定四分位"替换为**可配置、可随机、可多脉冲**的阶段调度器，强制模型从信号里定位事件。

**2.1 抽象 `PhaseSchedule`，替换写死的 `phase_boundaries`**

`src/sim/generation/phases.py` 现状是无参四分位。改造为"调度对象 → 逐时间步产出 `(phase_id, blend)`"：

```python
# src/sim/generation/phases.py（接口草图）
@dataclass(frozen=True)
class PhaseSegment:
    name: str            # baseline / exposure / steady / recovery / ...
    duration_frac: float # 占总时长比例（或绝对步数）
    blend_shape: str     # "hold0" | "ramp_up" | "hold1" | "ramp_down" | "decay" ...

@dataclass(frozen=True)
class PhaseSchedule:
    segments: tuple[PhaseSegment, ...]
    def boundaries(self, timesteps: int) -> list[int]: ...
    def phase_for_timestep(self, t: int, timesteps: int) -> str: ...
    def blend_for_timestep(self, t: int, timesteps: int) -> float: ...

# 兼容层：旧四分位 = 一个具名 schedule
STANDARD_EXPOSURE = PhaseSchedule((
    PhaseSegment("baseline", 0.25, "hold0"),
    PhaseSegment("exposure", 0.25, "ramp_up"),
    PhaseSegment("steady",   0.25, "hold1"),
    PhaseSegment("recovery", 0.25, "ramp_down"),
))
```

- 保留**模块级 `phase_for_timestep/blend_for_timestep` 包装**（委托给 `STANDARD_EXPOSURE`），使 `data/wv4-smoke` 在 `short` 档位下**逐位重现**当前产物（回归测试钉死）。

**2.2 per-sequence 随机化（事件定位）**

新增 `stage_jitter` 配置：对每条序列的相变点做有界随机抖动（onset 偏移、各段时长伸缩），由 `seed` 派生的逐序列 RNG 控制（保持可复现）。等价于内建 window-warp 式增强 [14][15]。

**2.3 多 stage_profile 库**（`stage_profile` 字段终于名副其实）

| profile | 形态 | 目的 |
|---|---|---|
| `standard_exposure` | 当前梯形四段 | 兼容 baseline |
| `variable_onset` | 注气时刻/时长随机 | 强制事件定位 |
| `fast_transient` | 短暴露 + 长基线/恢复 | 早期识别（§3.2 [8]） |
| `incomplete_recovery` | 恢复不回零 + 基线漂移 | 真实 MOx 行为（§3.2 [10][11]） |
| **`multi_pulse`** | **一条长序列含 K 次注气/恢复循环（随机间隔）** | **制造跨脉冲长程依赖——长时序模型的主战场** |

> `multi_pulse` 是体现长程依赖的关键：要在多脉冲长序列上稳健估计浓度 / 检测漂移，模型必须整合整段历史，而非局部窗口 + 平均。

**落点**：`phases.py`（新调度）、`benchmark.py`（spec 增加 `phase_schedule` / `stage_jitter`，写入 `manifest.json` 与 `metadata/waveform_spec.json` 做溯源）、`schema.py`（`PHASE_NAMES` 扩展为按 profile 动态，或保留超集）。

---

### P3. 更真实的瞬态动力学

**目标**：让"长"不仅是步数多，而是**信息密度高、必须建模历史**。

- **多时间常数响应/恢复**：`slow.py` 的 `_channel_value` 由单 `tau_rise/tau_decay` 升级为**双/多指数**（快慢两个时间常数叠加），贴合多机制吸附/解吸 [10][11]。
- **基线漂移 / 不完全恢复**：恢复段不回到精确基线，叠加缓慢漂移项 [11]。
- **记忆效应（可选）**：当前段的初值依赖上一段末值（尤其 `multi_pulse`），制造真正的跨时间依赖。

**落点**：`src/sim/generation/slow.py`、`src/sim/generation/acoustic_physics.py`（`PROCESSING_PARAMS` 增加多 tau / drift 参数）。保持"可校准代理模型"的边界声明不变（不冒充实测硬件）。

---

### P4. 模型与评估升级（让优势可测量）

**目标**：给长序列足够的模型容量与"会用顺序"的结构，并设计能暴露差距的评估。

**4.1 模型侧**
- **感受野随 `T` 缩放**：`TCNRegressor` 的 `channels`/`dilations` 需随目标 `T` 增长。给 `tcn.py` 加 `receptive_field >= timesteps` 的断言/告警（现 `RF=29`，远小于长序列）。参考覆盖配置：

  | 目标 T | 需要 dilations | 得到 RF |
  |---|---|---|
  | 512 | `[1,2,4,…,128]`（8 blocks） | 1021 |
  | 1024 | `…,256`（9 blocks） | 2045 |
  | 2048 | `…,512`（10 blocks） | 4093 |

- **补齐 / 新增序列感知模型**（`registry.py` 现仅 `cnn1d/tcn`）：
  - 落地 `LSTM`（`context_brief` 已宣称但缺失）；
  - 新增 **Transformer / PatchTST**（分块 + 通道独立，长回看）[4]；
  - 评估引入 **Mamba/SSM**（线性复杂度，长序列高效）[5][6]。
- **替换/补充全局平均池化**：除 `AdaptiveAvgPool1d(1)` 外，提供**保序聚合头**（last-step / attention pooling / 取 steady 段聚合），避免把时间顺序直接平均掉（§2.3）。

**4.2 数据增强**
- 训练管线加入 **window slicing + window warping + jitter**（§3.3 [14][15][16]）；与 P2 的协议随机化形成"数据生成期 + 训练期"双层增强。

**4.3 评估协议（关键：要能证明差距）**
- **强 baseline 对照**：ridge / 全局池化 CNN（顺序不敏感）作为下界——若长序列模型相对它们的优势随 `T` 增大而扩大，则"长时序优势"被实证（直接回应 DLinear 警示 [2]）。
- **分阶段指标**：按 `phase_id` 给 per-phase R²（baseline/exposure/steady/recovery 分别评）。
- **早期识别指标**：仅用前 `x%` 步预测的精度曲线（§3.2 [8]）。
- **外推到未见 profile**：训练 `standard_exposure`，测试 `variable_onset/multi_pulse`，考察泛化。

---

## 5. 分阶段路线图

| 阶段 | 内容 | 交付 | 依赖 |
|---|---|---|---|
| **S0** | `PhaseSchedule` 抽象 + 兼容层；钉死 `short` 档位逐位重现 wv4-smoke 的回归测试 | `phases.py` 重构 + 测试 | — |
| **S1** | 长序列档位（`standard/long`）+ `timesteps/dt_s` 配置化；TCN 感受野随 T 缩放 + 断言 | 新数据集 + TCN 配置 | S0 |
| **S2** | 多 profile（`variable_onset/multi_pulse/incomplete_recovery`）+ per-sequence jitter | `stage_profile` 库 | S0 |
| **S3** | 多时间常数 + 基线漂移动力学 | `slow.py`/`acoustic_physics.py` | S1 |
| **S4** | 模型补齐（LSTM/Transformer/PatchTST，Mamba 可选）+ 保序聚合头 + 增强 | `registry.py`/新模型/训练管线 | S1 |
| **S5** | 评估协议（强 baseline、per-phase、early、跨 profile 外推）+ 在 `long`/`multi_pulse` 上跑对比 | 基线对比报告 | S1–S4 |

> 建议先做 **S0+S1+S5 的最小闭环**（拉长 + 感受野修正 + 强 baseline 对比），用最小成本验证"加长之后，长序列模型相对 ridge 的差距是否真的拉开"。再决定是否投入 S2–S4 的协议与动力学复杂化。

---

## 6. 验收标准（如何证明改进达成目标）

1. **可配置性**：`timesteps/dt_s/phase_schedule/stage_jitter/stage_profile` 全部可经配置/工厂设定，`short` 档位逐位重现旧 `wv4-smoke`（回归测试通过）。
2. **感受野自洽**：所选 TCN/卷积模型满足 `receptive_field ≥ timesteps`（断言通过）。
3. **优势可测量（核心）**：在 `long` 或 `multi_pulse` 数据上，序列感知模型（TCN-大RF / Transformer / PatchTST / Mamba）相对**顺序不敏感强 baseline**（ridge、全局池化 CNN）的 R² 差距 **随 `T` 增大而单调扩大**；在 `short` 档位上差距应很小（对照组）。
4. **领域合理性**：per-phase 指标显示 exposure/recovery 段信息被有效利用；早期识别曲线随可见步数上升而改善。
5. **可复现与溯源**：`phase_schedule`、`stage_profile`、`dt_s`、`timesteps` 全部写入 `manifest.json` / `waveform_spec.json`。

---

## 7. 风险与权衡

- **算力/存储**：`T` 增大线性抬高波形数组体积与训练成本。缓解：分档位、memmap 懒加载（已具备）、必要时降 `dt`/下采样。
- **过度复杂化**：协议与动力学复杂化可能掩盖契约 bug。缓解：S0 钉死兼容回归；每阶段独立验收。
- **"加长 ≠ 更难"**：若仅拉长但相变仍固定、模型仍全局平均，则优势依然不显（§2.3）。**必须 P2（事件定位）+ P4（保序聚合/大感受野）配套**，加长才有意义。
- **强 baseline 反超的可能**：参考 DLinear，简单模型可能很强 [2]。这正是验收标准 §6.3 的意义——以差距而非绝对值判定，结果无论正负都有信息量。
- **边界声明**：动力学复杂化仍属"可校准仿真代理"，不得宣称等价于真实硬件（沿用既有边界）。

---

## 8. 参考文献

1. Bai, Kolter, Koltun. *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* (TCN), arXiv:1803.01271. https://arxiv.org/abs/1803.01271
2. *Long sequence time-series forecasting with deep learning: A survey*, Information Fusion (ScienceDirect). https://www.sciencedirect.com/science/article/abs/pii/S1566253523001355
3. *Deep Time Series Models: A Comprehensive Survey and Benchmark*, arXiv:2407.13278. https://arxiv.org/html/2407.13278v2
4. *A systematic review for transformer-based long-term series forecasting*, Artificial Intelligence Review (2025). https://link.springer.com/article/10.1007/s10462-024-11044-2
5. Gu, Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752. https://arxiv.org/abs/2312.00752
6. *Is Mamba Effective for Time Series Forecasting?* (S-Mamba), arXiv:2403.11144. https://arxiv.org/html/2403.11144
7. *All-feature olfactory algorithm (AFOA): an e-nose using response/equilibrium/recovery with 1D-CNN + RNN + attention*（预印本）。https://d197for5662m48.cloudfront.net/documents/publicationstatus/80524/preprint_pdf/2c85bfcd8a09760ab7068f538cc27595.pdf
8. *Early-Stage Gas Identification Using Convolutional Long Short-Term Neural Network with Sensor Array Time Series Data*, Sensors 2021, 21(14):4826. https://www.mdpi.com/1424-8220/21/14/4826
9. *A deep learning approach for gas sensor data regression: Incorporating surface state model and GRU-based model*, APL Machine Learning 2(1):016104. https://pubs.aip.org/aip/aml/article/2/1/016104/2933789
10. *Oxygen Adsorption and Desorption Kinetics in CuO Nanowire Bundle Networks*（多时间常数响应/恢复）, ACS Applied Nano Materials. https://pubs.acs.org/doi/10.1021/acsanm.2c01245
11. *A New Model and Its Application for the Dynamic Response of RGO Resistive Gas Sensor*（基线漂移/不完全恢复）, Sensors (PMC6412666). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6412666/
12. keras-tcn 文档（感受野公式与"RF≥序列长度"工程实践）。https://github.com/philipperemy/keras-tcn/blob/master/README.md
13. *TF-TCN: A time-frequency combined gas concentration prediction model for E-nose data*, Sensors and Actuators B (2024). https://www.sciencedirect.com/science/article/abs/pii/S0924424724006484
14. Wen et al. *Time Series Data Augmentation for Deep Learning: A Survey*, IJCAI 2021. https://www.ijcai.org/proceedings/2021/0631.pdf
15. Iwana, Uchida. *An empirical survey of data augmentation for time series classification with neural networks*, PLOS ONE 2021. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0254841
16. *Complementary MEMS gas sensor array and lightweight deep learning for DGA*（膨胀卷积处理基线漂移与慢解吸；CNN-LSTM-Attention/TCN）, Sensors and Actuators B (2025). https://www.sciencedirect.com/science/article/abs/pii/S0925400525012584

---

*本提案为代理仿真链路的实验设计改进，不改变"当前声学/光学实现为可校准代理模型"的既有边界声明；落地时所有新配置默认保持现状可复现。*
