# RCDW 数据集主线对齐改动方案 — 审查报告

## 0. 审查元信息

| 项 | 内容 |
|---|------|
| 审查对象 | `docs/学长算法/RCDW_数据集主线对齐改动方案.md` v1.0 |
| 审查日期 | 2026-06-30 |
| 审查范围 | 方案完整性、物理正确性、与主线一致性、可执行性、内部一致性 |
| 参照源码 | `src/sim/` 主线（schema / conditions / phases / slow / acoustic_physics / splits / scalers / validation）、`rcdw_mgda/` 现有代码（synth / single_modal / feature / inject / stage_a / stage_b / rcdw / configs） |
| 参照文档 | `CLAUDE.md`、`AGENTS.md`、`docs/学长算法/RCDW_独立复现方案.md`、`docs/学长算法/RCDW_实施完成情况.md` |

---

## 1. 总体评价

方案质量很高。从 schema 定义到物理仿真、打包、校验、训练侧适配、扰动评测全链路覆盖，Phase 拆分合理，验收命令具体可执行。物理决策（模态-气体敏感性矩阵、光学波段选择、声学建模）有明确依据，与 HG 主线的隔离原则贯穿始终。

以下按严重程度分四级组织审查发现：🔴 必须修正（会导致 bug 或契约冲突）、🟡 建议讨论（非 bug 但值得斟酌）、🟢 细节建议（实施时顺手修正）、✅ 优点确认。

---

## 2. ✅ 优点确认

### 2.1 隔离原则贯穿始终

§1.4 明确了四层隔离：不 import → 可借鉴 → 必须独立维护 → `RCDW-` 命名前缀。与 CLAUDE.md 的 "rcdw_mgda 与 src/ 完全隔离" 以及 AGENTS.md 的 "不 import src" 约束完全一致。

### 2.2 物理建模决策清晰

- §2.3 "O₂ 均核双原子无永久偶极矩 → 中红外完全透明 → NDIR 仅设 CO₂ 单通道" 推理严密。
- §2.2 模态-气体敏感性矩阵物理依据充分（Beer-Lambert / 热导率差异 / 声速 ∝ 1/√M_mix）。
- §5.5 声学模型新增 O₂ 弛豫项、删除 H₂ 扩散项，改动有理有据。

### 2.3 通道布局方案 A（12 维单一拼接张量）正确

保持 `(B, L, C)` 单输入假设，改动集中在索引常量而非模型 forward 签名。与旧 RCDW 的 `(B, L, 6)` 输入模式一致，升级路径清晰。

### 2.4 LEGACY 黑名单防御性声明

虽 RCDW 不存在历史包袱，但提前声明 `LEGACY_CONDITION_FIELDS` 与 HG validation 一致，是好的防御性设计。

### 2.5 Phase 分步落地 + 验收命令

5 个 Phase 每步有明确的涉及文件、产物描述、`pytest` 验收命令和风险标注，可直接作为 sprint backlog。

---

## 3. 🔴 必须修正的问题

### 3.1 §6.1 与 §8.2 的 tof_quality / tof_accepted 进训练矛盾

**位置**：§6.1 arrays.py 落盘清单 vs §8.2 通道布局

**问题描述**：

§6.1 落盘清单中：

| 文件 | 进训练 |
|------|--------|
| `ultrasonic_tof_quality.npy` | **否** |
| `ultrasonic_tof_accepted.npy` | **否** |

但 §8.2 通道布局将它们拼入 12 维输入张量：

```
IDX_US_QUALITY  = 10    # ultrasonic_tof_quality
IDX_US_ACCEPTED = 11    # ultrasonic_tof_accepted
```

两处结论互斥：如果不进训练，就不应拼入模型输入；如果拼入，就是进了训练。

**修正建议**：将 §6.1 中这两项的"进训练"列改为"**是**"。理由：`tof_quality` 和 `tof_accepted` 作为超声测量置信度信号，对 ErrorNet 判断超声模态可靠性有直接帮助，物理上合理。

### 3.2 `rcdw.py` 未出现在改动文件清单中

**位置**：§8.5 / §8.6、Phase 4 涉及文件列表

**问题描述**：

`rcdw_mgda/rcdw/models/rcdw.py` 第 93 行的类 docstring 写 "输入: (B, L=8, 6)"，`forward` 方法注释也写 `x: (B, L=8, 6)`。通道维从 6 改为 12 后，这些注释需要同步更新。

实际上 `rcdw.py` 的运行代码无需修改（`x[:, -1, :]` 不依赖硬编码维度，`extract_modal_input` 的改动在 `single_modal.py` 中），但方案未在任何涉及文件清单中提及 `rcdw.py`，容易导致实施遗漏。

**修正建议**：在 Phase 4 涉及文件列表中增加 `rcdw/models/rcdw.py`（注释/docstring 更新：`(B, L=8, 6)` → `(B, L=8, 12)`）。

### 3.3 baseline = 100% N₂ 的隐含假设未显式说明

**位置**：§5.4 `_blend_composition`

**问题描述**：

```python
def _blend_composition(condition, blend):
    return {
        "x_o2":  float(condition["x_O2"])  * blend,
        "x_co2": float(condition["x_CO2"]) * blend,
        "x_n2":  100.0 + (float(condition["x_N2"]) - 100.0) * blend,
    }
```

当 `blend=0` 时，`x_o2=0, x_co2=0, x_n2=100`——即 baseline 状态为纯 N₂。这在物理上是正确的（标定前通 100% N₂ 背景气），且与 HG 主线逻辑一致（HG 的 H₂/CH₄/CO₂ 同样在 blend=0 时为 0，N₂ = 100%）。

但方案文本中没有任何一处显式说明 "baseline = 100% N₂ 纯背景气"。实现者可能误以为 baseline 是空气（约 78% N₂ + 21% O₂）。

**修正建议**：在 §5.4 的 `_blend_composition` 代码前增加一段说明："RCDW 的 baseline 状态定义为 100% N₂ 纯背景气（与 HG 主线一致）。blend=0 时所有目标气体浓度归零，N₂ 充满腔室；blend=1 时恢复到采样目标浓度。"

### 3.4 缺少 blake2b 确定性 RNG 策略说明

**位置**：§5.4 slow.py、§5.9 benchmark.py

**问题描述**：

HG 主线 `slow.py` 使用 `_stable_uint32(seed, global_sequence_index, stream_name)` 通过 **blake2b 哈希**为每个 sequence 生成两个独立的确定性 RNG 流（`"condition"` 和 `"sequence"`）。这保证了在多进程并行（`ProcessPoolExecutor` + chunk）下，无论 chunk 划分如何变化，同一 `(seed, sequence_index)` 组合始终产生相同结果。

方案 §5.4 和 §5.9 只提到 `seed` 参数，没有说明 RNG 流隔离策略。如果实现时简单用 `np.random.default_rng(seed + i)` 代替，在改变 `workers` 或 `chunk_size` 时结果将不可复现。

**修正建议**：在 §5.4 增加一小节 "RNG seeding 策略"，说明复用 HG 的 blake2b 双流模式（在 rcdw_mgda 内独立重写，不 import 主线）。

### 3.5 split 比例差异需显式标注

**位置**：§6.4 splits.py

**问题描述**：

方案 §6.4 说 "按 70/15/15 比例分配 train、val、test"。但 HG 主线 `splits.py` 默认比例实际是 **70/15/10**（`train_ratio=0.70, val_ratio=0.15, test_ratio=0.10`），剩余 ~5% 归 extrapolation。

RCDW 删掉了 extrapolation，将其份额归入 test（从 10% → 15%）。这是合理的有意调整，但方案中没有指出这一差异的来源和理由。

**修正建议**：在 §6.4 或 §2.6 中增加一行："与 HG 主线的 70/15/10/5(extrapolation) 比例不同，RCDW 将 extrapolation 份额并入 test，最终比例为 70/15/15。原因：RCDW 三组分范围已覆盖全 simplex，外推 holdout 无物理意义（见 §2.6 决策）。"实现时需确保 `build_split_groups` 的 `test_ratio` 默认值为 0.15 而非照搬 HG 的 0.10。

---

## 4. 🟡 建议讨论的设计选择

### 4.1 12 维输入中 4 个通道实际未被读取

**位置**：§8.2 / §8.3

**现状**：12 维输入中，`SingleModal` 每个分支只取 1 个 sensor + 3 个 env（共 4 维），`FeatureExtractor` 取 3 个 sensor 信号做统计。实际被模型读取的通道：

| 索引 | 通道 | 被哪个模块读取 |
|------|------|---------------|
| 0 | V_NDIR_CO2 | NDIRNet + FeatureExtractor |
| 1 | V_TCS | TCDNet + FeatureExtractor |
| 2 | T_C | 所有分支（环境） |
| 3 | P_MPa | 所有分支（环境） |
| 4 | H_RH | 所有分支（环境） |
| 5 | L_m | **无** |
| 6 | piston_position_m | **无** |
| 7 | ultrasonic_tof_observed_s | **无** |
| 8 | ultrasonic_sound_speed_estimated | USNet + FeatureExtractor |
| 9 | ultrasonic_peak_index | **无** |
| 10 | ultrasonic_tof_quality | **无** |
| 11 | ultrasonic_tof_accepted | **无** |

共 6 个通道未被任何模块读取（idx 5, 6, 7, 9, 10, 11）。

**评估**：这不是错误。预留通道供未来扩展（如 ErrorNet 直接读取 tof_quality 判断可靠性）是合理的设计。通道索引一次到位避免后续再改。但如果关注 GPU 内存效率（尤其是 formal benchmark 6000+ 序列），可考虑先用 10 维甚至 9 维精简版。

**建议**：保留 12 维方案。在方案 §8.2 末尾增加一段说明哪些通道当前未被模型使用但预留供扩展，使读者不困惑。

### 4.2 5 种 PhaseSchedule 全部复制但只用 1 种

**位置**：§5.2 / §5.1

**现状**：方案完整复制了 HG 的 5 种 PhaseSchedule，但 §5.1 的 `generate_condition_rows` 设计中 "每个 mixture 仅一个 sequence"，且配置中 `stage_profile: "standard_exposure"`（固定）。这意味着其余 4 种 schedule（`variable_onset`、`fast_transient`、`incomplete_recovery`、`multi_pulse`）被定义但永远不会被调用。

HG 主线之所以有多 sequence/mixture，是因为同一组分可以搭配不同 stage_profile。RCDW 当前一对一映射使此机制退化。

**建议**：方案中应显式说明：
1. Phase 1–4 只使用 `standard_exposure`，其余 schedule 仅作为代码就绪（code-ready）。
2. 后续若需增加数据多样性（如 formal benchmark），可通过 "每个 mixture 生成多个 sequence（每个 sequence 使用不同 stage_profile）" 来激活其余 schedule。
3. 届时 mixture_id:sequence_id 关系从 1:1 变为 1:N，split 才真正发挥 "同 mixture 不跨 split" 的保护作用。

### 4.3 ENV_INDICES 顺序变更需显式标注

**位置**：§8.2 / §8.3 / §8.4

**现状**：

| 版本 | ENV_INDICES | 对应通道 |
|------|-------------|----------|
| 旧 | `[3, 4, 5]` | `[P, T, RH]` |
| 新 | `[2, 3, 4]` | `[T_C, P_MPa, H_RH]` |

变化有两个维度：
- **索引位置**变了（3,4,5 → 2,3,4）——方案已说明。
- **语义顺序**变了（P,T,RH → T,P,RH）——方案未提及。

由于 `SingleModal` 是无结构 MLP（纯从训练中学习映射），顺序改变不影响最终效果（重训即可）。`FeatureExtractor` 内 `delta_T/delta_P/delta_RH` 的计算按新顺序编写，示例代码的注释是正确的。

**建议**：在 §8.2 通道布局表后增加一句："注意：环境变量输入顺序从旧版 `[P, T, RH]` 变为新版 `[T_C, P_MPa, H_RH]`，这是有意调整以匹配 SLOW_CHANNELS 的自然顺序。所有历史 ckpt 不兼容，必须重训。"

### 4.4 声速线性混合模型 vs HG 的处理

**位置**：§5.5

**现状**：方案使用简化线性混合：`c_mix = Σ x_i * c_i + 0.6*(T-25)`。HG 主线也使用类似的线性混合（而非理想气体 `c = sqrt(γRT/M_mix)` + 弛豫修正），但 HG 的 c_i 常数有明确来源（H₂: 1306, CH₄: 446, CO₂: 268, N₂: 353 m/s）。

方案中 O₂ 声速标 "TBD（暂用 330.0）"。纯 O₂ 在 25°C 下的声速文献值约 330 m/s，这个暂定值是合理的。

**建议**：在 `acoustic_model_metadata` 中记录 `"model": "linear_mixing_v1"`。待 Phase 2 实施时，将 O₂ 声速定为 329.5 m/s（NIST Chemistry WebBook @ 25°C, 1 atm）并标注来源。

### 4.5 `composition_scheme` 命名风格

**位置**：§2.6

**现状**：方案用 `"rcdw_o2_co2_n2"`。HG 用 `"hydrogen_ng"`——没有把组分名写进 scheme 名。

**评估**：RCDW 只有一种组分体系，简化为 `"rcdw"` 更简洁；但冗长命名在 debug/日志中信息密度更高。非阻塞，口味问题。

### 4.6 是否移植 legacy empirical 动力学路径

**位置**：§5.4

**现状**：HG `slow.py` 内部有两条动力学路径：

| 路径 | 触发条件 | 模型 |
|------|----------|------|
| Legacy empirical | `standard_exposure` + 无 jitter + `empirical` 后端 | 单指数 RC，固定 phase 分界 |
| Multi-tau equilibrium | 所有其他情况 | 双指数 RC + recovery floor + random walk |

方案仅描述了 multi-tau 路径。由于 RCDW 使用 `hitran_hapi_v1` 后端，永远不会命中 legacy 路径——不移植是正确的。

**建议**：在 §5.4 显式声明 "RCDW 不移植 legacy empirical 路径，仅保留 multi-tau equilibrium。原因：RCDW 使用 HITRAN 后端，legacy 路径的触发条件（empirical 后端）不适用。"避免实现者看到 HG 源码后困惑两条路径该搬哪条。

---

## 5. 🟢 细节建议

### 5.1 condition dict 值的类型约定

HG 主线 `generate_condition_rows` 返回 `list[dict[str, str]]`——所有值都是字符串（通过 `_fmt(value, digits)` 格式化）。方案 §5.1 签名正确，§5.4 代码示例中也正确使用了 `float(condition["x_O2"])`。

**建议**：在 §5.1 的接口说明后增加一行备注："与 HG 一致，所有 condition row 值为字符串类型，组分值保留 6 位小数。下游使用时需显式 `float()` 转换。"

### 5.2 LHS d=2 的 N₂ 边界回退几乎不会触发

§5.1 的组分范围：`O2 ∈ [0, 25]`，`CO2 ∈ [0, 20]`，`N2 = 100 - O2 - CO2`。

当 O₂ 和 CO₂ 同时取最大值时：`x_N2 = 100 - 25 - 20 = 55`，恰好等于 N₂ 下限。因此理论上除浮点精度问题外，**不会有任何样本触发回退**。

§13.4 中 "若回退率超过 10% 可改 Dirichlet" 的备选基本不会触发。方案应在 §5.1 或 §13.4 中指出这一事实，以免读者误以为回退是常态。

等比例缩减算法也应显式给出，避免歧义：

```python
total_oc = x_O2 + x_CO2
if total_oc > 45.0:  # 100 - 55
    scale = 45.0 / total_oc
    x_O2 *= scale
    x_CO2 *= scale
    x_N2 = 55.0
```

### 5.3 `BenchmarkDataset.__init__` 的 window 参数化

§8.1 的 `BenchmarkDataset(window=8)` 与 `RCDW_MGDA.feat = FeatureExtractor(window=8)` 都硬编码了 `window=8`。建议两处统一从配置读取 `cfg["data"]["window"]`，避免修改一处忘记另一处。

### 5.4 旧 `data.n_train / n_val / n_test` 字段清理

§10.3 提到删除旧字段，但只是简略一笔。当前 `scripts/train.py` 通过 `cfg["data"]["n_train"]` 等调用 `synth.make_splits`。Phase 4 必须将 `scripts/train.py` 的数据加载全面改写。

**建议**：在 Phase 4 涉及文件列表中显式列出 `scripts/train.py`（数据加载改写：从 `synth.make_splits` 切换到 `BenchmarkDataset`）以及 `configs/*.yaml`（删除 `data.n_train / n_val / n_test`，新增 `data.dataset_root / train_modalities`）。

### 5.5 `test_synth.py` 处理方式

方案说归档到 `tests/_legacy/`。更简洁的做法是直接删除——git 保留历史，无需维护 `_legacy/` 子目录。主线 `src/` 也不维护 legacy 子目录。

### 5.6 扰动强度物理语义变化

§9.1 将 `ultrasonic` 扰动从 `x[..., 2]`（旧 S_us 为 toy 归一化值）改为 `x[..., IDX_US_SPEED]`（真实 m/s 量级，约 340–360）。`level * scale * randn` 中 `scale = x.abs().mean()` 会自适应量级，不会出错。但扰动强度的物理含义变了——同一 `level=0.05` 对应的绝对扰动幅度完全不同。

§13 Phase 5 风险已提到 "v1 vs v2 扰动语义不同"，建议在 Phase 5 产物中生成一份对照表记录每种扰动在新旧布局下的实际扰动幅度，作为文档归档。

### 5.7 smoke 配置的 `timesteps: 32` 安全性

方案注释说 "必须 ≥ 12（MULTI_PULSE 段数）"。smoke 用 `stage_profile: "standard_exposure"`（仅 4 段），32 对 MULTI_PULSE 的 12 段也充裕。无需修改，仅确认安全。

### 5.8 HITRAN 缓存预计算入口

§13.1 建议在 `scripts/generate_benchmark.py` 入口加 `--precompute-cache-only` 选项。这是好实践，HG 主线有独立的 `pipeline.precompute_hitran_benchmark_cache` CLI。RCDW 应在 Phase 2 或 Phase 3 落地时同步提供此入口。

---

## 6. 现有代码参照点（实施时快速定位）

供实施时参照的 HG 主线关键对应表：

| RCDW 目标模块 | HG 主线参照文件 | 重点关注 |
|--------------|----------------|---------|
| `rcdw/sim/core/schema.py` | `src/sim/core/schema.py` | 常量命名模式、字段 tuple 定义方式 |
| `rcdw/sim/generation/conditions.py` | `src/sim/generation/conditions.py` | `_fmt()` 格式化、LHS d=3 → d=2 降维、`_stable_uint32` RNG |
| `rcdw/sim/generation/phases.py` | `src/sim/generation/phases.py` | 5 种 schedule 定义、`resolve_timeline` 批量计算、`jittered` 随机化 |
| `rcdw/sim/generation/slow.py` | `src/sim/generation/slow.py` | `_blend_composition`、`_channel_dynamic_params`（9 参数/通道）、`_multi_tau_channel_step`、`build_sequence_arrays` 返回字典 13 个 key |
| `rcdw/sim/generation/acoustic_physics.py` | `src/sim/generation/acoustic_physics.py` | 声速线性混合、衰减 6 项加和（RCDW 改为 5 项）、`PROCESSING_PARAMS` |
| `rcdw/sim/packaging/splits.py` | `src/sim/packaging/splits.py` | `build_split_groups` 的比例参数（HG: 70/15/10 → RCDW: 70/15/15）、边界保护 |
| `rcdw/sim/packaging/scalers.py` | `src/sim/packaging/scalers.py` | train-only 拟合、`Z_SCORE_STD_EPSILON`、modal_groups 分组 |
| `rcdw/sim/validation/integrity.py` | `src/sim/validation/integrity.py` | 参数化 `component_fields`/`slow_channels`/`background_fields`、`SPLIT_NAMES` 引用（RCDW 无 extrapolation） |

---

## 7. 评分汇总

| 维度 | 评分 | 说明 |
|------|------|------|
| 完整性 | ⭐⭐⭐⭐⭐ | 从 schema 到 validation 到训练侧再到扰动评测，全链路覆盖；Phase 划分清晰 |
| 物理正确性 | ⭐⭐⭐⭐ | O₂/CO₂/N₂ 模态敏感性矩阵正确；光学 "仅 CO₂ 通道" 无可挑剔；O₂ 弛豫参数标 TBD 是诚实处理 |
| 与主线一致性 | ⭐⭐⭐⭐ | 忠实复制 HG 的 phase/packaging/validation/splits 设计；split 比例有意调整合理 |
| 可执行性 | ⭐⭐⭐⭐ | 验收命令具体，风险标注充分，Phase 可直接作为 sprint backlog |
| 内部一致性 | ⭐⭐⭐½ | §6.1 与 §8.2 的 tof_quality/tof_accepted 矛盾是唯一 hard bug；其余为表述遗漏 |

---

## 8. 修正优先级

### 执行前必须修正（🔴）

| # | 问题 | 位置 | 修正内容 |
|---|------|------|----------|
| 3.1 | tof_quality/accepted 进训练矛盾 | §6.1 + §8.2 | 统一 §6.1 表格为"是" |
| 3.2 | `rcdw.py` 漏列改动 | Phase 4 涉及文件 | 增加 `rcdw/models/rcdw.py`（docstring 更新） |
| 3.3 | baseline=100% N₂ 未说明 | §5.4 | 增加 baseline 状态物理说明 |
| 3.4 | RNG 策略缺失 | §5.4 | 增加 blake2b 双流 RNG 说明 |
| 3.5 | split 比例差异未标注 | §6.4 | 增加与 HG 的差异说明和理由 |

### 建议讨论后决定（🟡）

| # | 选择 | 位置 | 建议倾向 |
|---|------|------|----------|
| 4.1 | 6 个未读取通道是否精简 | §8.2 | 保留 12 维，增加说明 |
| 4.2 | 5 种 schedule 仅用 1 种 | §5.2 | 保留全部，增加规划说明 |
| 4.3 | ENV_INDICES 顺序变更 | §8.2 | 增加显式说明 |
| 4.4 | 声速模型记录 | §5.5 | 在 metadata 记录 `linear_mixing_v1` |
| 4.5 | composition_scheme 命名 | §2.6 | 口味问题，不阻塞 |
| 4.6 | legacy empirical 路径 | §5.4 | 显式声明不移植 |

### 实施时顺手修正（🟢）

| # | 细节 | 位置 |
|---|------|------|
| 5.1 | condition dict 类型约定备注 | §5.1 |
| 5.2 | N₂ 回退几乎不触发的说明 | §5.1 / §13.4 |
| 5.3 | window 参数化 | §8.1 |
| 5.4 | 旧字段清理 + scripts/train.py 改写 | Phase 4 文件列表 |
| 5.5 | test_synth.py 直接删除而非归档 | §11.3 |
| 5.6 | 扰动强度语义变化对照表 | Phase 5 |
| 5.7 | smoke timesteps 安全性确认 | §10.2 |
| 5.8 | HITRAN 缓存预计算入口 | Phase 2–3 |

---

## 9. 审查结论

方案可以执行。建议先修正 §3 中的 5 个必须修正项（预计 30 分钟内可完成文档更新），然后按 Phase 1 → 5 落地。§4 中的讨论项可在 Phase 实施过程中逐步决定。

---

## 10. 修正落实记录

以下 5 项 🔴 必须修正已全部落实到方案文档 v1.1（2026-06-30）：

| # | 审查问题 | 修正位置 | 修正内容 | 状态 |
|---|----------|----------|----------|------|
| 3.1 | tof_quality/accepted 进训练矛盾 | §6.1 落盘清单表 + 表后说明 | 两项"进训练"列从"否"改为"**是**"；"明确不进训练的数组"列表中删除这两项；增加审查修正 blockquote 说明理由 | ✅ 已修正 |
| 3.2 | `rcdw.py` 漏列改动 | Phase 4 涉及文件 | 增加 `rcdw/models/rcdw.py`（docstring 更新）、`scripts/train.py`（数据加载改写）、`configs/*.yaml`（字段增删） | ✅ 已修正 |
| 3.3 | baseline=100% N₂ 未说明 | §5.4 `_blend_composition` 前 | 增加 blockquote 说明 baseline 物理定义：100% N₂ 纯背景气，对应标定流程 | ✅ 已修正 |
| 3.4 | RNG 策略缺失 | §5.4 动力学描述后 | 增加 "RNG seeding 策略" 子节：blake2b 双流模式 + `_stable_uint32` 函数签名 + 禁止简单偏移方案 | ✅ 已修正 |
| 3.5 | split 比例差异未标注 | §6.4 逻辑列表后 | 增加 blockquote 说明 HG 70/15/10+5 vs RCDW 70/15/15 的差异、原因、实现注意事项 | ✅ 已修正 |

方案版本号已从 v1.0 升级为 **v1.1（审查修正版）**，修订摘要写在 §0 修订记录中。
