# RCDW 数据集向主线 HG 对齐改动方案

## 0. 修订记录与文档定位

- **版本**：v1.2（YAGNI 精简 + 归一化策略锁定版）
- **日期**：2026-06-30
- **v1.2 修订摘要**（对照 `RCDW_数据集主线对齐方案_审查报告.md` §4.1 与 §4.2 的讨论结论）：
  - §2.5 / §5.2 / §3.2 / §11.1：YAGNI 精简 PhaseSchedule，仅保留 `STANDARD_EXPOSURE`，删除其余 4 种（`VARIABLE_ONSET`、`FAST_TRANSIENT`、`INCOMPLETE_RECOVERY`、`MULTI_PULSE`）。`resolve_phase_schedule` 对未实现 profile 显式 `raise NotImplementedError`，并在 §5.2 末尾给出"未来激活其他 schedule 的三步路径"。理由：当前 `generate_condition_rows` 是 mixture:sequence = 1:1 映射，其余 4 种 schedule 在架构上不可达；复制 HG 全部 5 种会引入 4 段死代码，违反 AGENTS.md "禁止无依据防御性编程"。
  - §6.5 scalers.py：增加异质通道归一化策略表，显式锁定每个通道的处理方式（连续物理量走 train-only Z-score；`ultrasonic_tof_quality`、`ultrasonic_tof_accepted`、`ultrasonic_peak_index` 跳过归一化以保留 [0,1] / 0/1 / 离散下标语义）。理由：审查 §4.1 仅指出"保留 12 维加注释"不够，预留通道若被 Z-score 拟合会丢失物理可解释性，ErrorNet 未来读取 `tof_quality` 时拿到的将是奇怪的标准化值。
  - §8.2：在通道布局表后增加"未被模型直接读取的预留通道说明"，标注 6 个未读通道的当前用途与未来扩展方向。
- **v1.1 修订摘要**（对照 `RCDW_数据集主线对齐方案_审查报告.md` §3 的 5 项必须修正）：
  - §6.1：`ultrasonic_tof_quality` 与 `ultrasonic_tof_accepted` 的"进训练"列从"否"改为"是"，与 §8.2 通道布局统一
  - Phase 4 涉及文件:增加 `rcdw/models/rcdw.py`（docstring 更新）、`scripts/train.py`（数据加载改写）、`configs/*.yaml`（字段增删）
  - §5.4：增加 baseline = 100% N₂ 纯背景气的显式物理说明
  - §5.4：增加 blake2b 双流 RNG seeding 策略说明
  - §6.4：增加与 HG 主线 split 比例差异的显式标注（70/15/10+extrapolation → 70/15/15）
- **目的**：将 `src/sim` 主线 HG 的 H2/CH4/CO2/N2 四组分数据集构建思路（phase 时间结构、多时间常数慢通道动力学、HITRAN 光学栈、声学物理、packaging 目录树、scalers、splits、validation、ID 体系）完整搬到 `rcdw_mgda/` 子工程，使其拥有与主线同质量的数据生成管线，但组分固定为 O2/CO2/N2 三组分。
- **与 `docs/学长算法/RCDW_独立复现方案.md` 的关系**：互补，不替换。复现方案定义了 RCDW 算法侧的 W_base、FeatureExtractor、RCDWFusion、Stage A/B 训练、扰动评测等；本方案定义数据生成侧的 schema、物理建模、打包、校验、切分，以及训练侧如何从新数据集读取数据。两者共同构成完整工程。
- **适用场景**：rcdw_mgda 子工程内所有代码；不涉及 `src/sim` 主线。

---

## 1. 目标与边界

### 1.1 目标列表

| # | 目标 | 说明 |
|---|------|------|
| G1 | 建立 rcdw_mgda 独立的 schema / ID 体系 | 不与 `src/sim` 共享任何常量或 ID 类型 |
| G2 | 复现 HG 的 phase 时间结构（仅保留 `STANDARD_EXPOSURE`，v1.2 YAGNI） | baseline → exposure → steady → recovery，支持 jittered；其他 schedule 见 §5.2 未来激活路径 |
| G3 | 复现 HG 的慢通道动力学（多时间常数 RC + drift + random walk + noise） | 按通道指定 tau_rise / tau_decay / fast_tau_fraction 等参数 |
| G4 | 建立 O2/CO2/N2 适用的声学物理后端 | 声速依赖三组分摩尔质量加权 + 温度修正；衰减含经典 + 弛豫 + 扩散 |
| G5 | 建立 O2/CO2/N2 适用的 HITRAN 光学栈 | 仅 CO2 和 H2O 有中红外吸收；O2 与 N2 透明；决定 NDIR 通道数 |
| G6 | 完整保留光纤麦克风生成代码 | 与超声同步生成、落盘，但不进入训练管线 |
| G7 | 建立与 HG 一致的 packaging 目录树 | manifest.json、sequence_index.csv、splits/、scalers/、metadata/ 等 |
| G8 | mixture_id 分层切分 + train/val/test | 不引入 extrapolation（见 2.6 决策） |
| G9 | 训练侧 Dataset 适配新通道布局 | 从磁盘读取 slow + ultrasonic，按窗口切片 (B, L, C) |
| G10 | 保持 RCDW 算法契约不被破坏 | W_base (3,3)、单模态三分支、Stage A/B/扰动三阶段 |

### 1.2 范围之内

- `rcdw_mgda/rcdw/sim/` 下新建数据生成子包（core / generation / packaging / validation）
- `rcdw_mgda/rcdw/data/` 下重写 `dataset.py`（替换旧 `synth.py` 的 toy 生成）
- `rcdw_mgda/configs/` 扩增 generation / spectral / acoustic / phases / splits / scalers 配置节
- `rcdw_mgda/tests/` 新增数据管线测试
- `rcdw_mgda/scripts/` 新增 benchmark 生成入口脚本
- 训练侧 `single_modal.py`、`feature.py` 的通道索引常量重新定义

### 1.3 范围之外

- `src/sim` 主线任何文件的修改
- `src/dl`、`src/ml`、`src/pipeline` 主线模块的修改
- 旧 `rcdw_mgda/rcdw/data/synth.py` 的维护（将替换为新生成管线，旧文件归档或删除）
- 光纤麦克风进入 DataLoader 或训练 loss
- 引入 H2/CH4 组分或四组分体系
- 历史 `rcdw_mgda/runs/` ckpt 兼容（明确废弃）

### 1.4 与 `src/sim` 主线的隔离原则

- **不 import**：rcdw_mgda 任何模块不得 `import sim` 或 `from sim`。
- **可借鉴**：可以阅读 `src/sim` 源码理解设计，然后在 rcdw_mgda 内独立重写。
- **必须独立维护**：rcdw_mgda 的 schema 常量、ID 类型、物理参数全部独立定义，不与主线共享变量或配置文件。
- **命名空间前缀**：ID 使用 `RCDW-` 前缀以与主线 `wv4-`、`sg4-` 区分。
- **类比 syngas 模式**：参考 `docs/syngas/README.md` 的做法——创建独立的 schema 文件和 generation 子包，不修改 HG 代码。

---

## 2. 业务决策

### 2.1 组分：固定 O2/CO2/N2 三组分

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 组分集合 | `(x_O2, x_CO2, x_N2)`，sum = 100%，闭包 | 与 RCDW 框架及现有 W_base (3,3) 语义对齐；HG 的 `composition_scheme = "hydrogen_ng"` 也是闭包 |
| 采样策略 | **LHS（Latin Hypercube Sampling）**，d = 2（三组分 sum = 100%，自由度为 2） | 与 HG 主线一致；Dirichlet 适合玩具但不利于均匀覆盖 simplex 空间；LHS 在 [0,1]² 采样后映射到 simplex 可获得更好的空间填充性 |
| 采样映射 | 从 LHS 2D unit cube → simplex 三组分：`x_O2 = u1 * max_O2`，`x_CO2 = u2 * max_CO2`，`x_N2 = 100 - x_O2 - x_CO2`，若 `x_N2 < min_N2` 则缩回 O2 与 CO2 | 简单直接；HG 的四组分 LHS→3D 映射模式可参考，但 RCDW 仅需 2D |
| 组分范围 | O2 ∈ [0, 25]%，CO2 ∈ [0, 20]%，N2 ∈ [55, 100]% 作为 complement | 贴合实际燃烧尾气与空气场景：N2 主导，O2 和 CO2 为次要组分 |
| 环境参数 | `T_C_base` ∈ [15, 35]，`P_MPa_base` ∈ [0.10, 0.709]，`H_RH_base` ∈ [20, 80]，`L_m_base` ∈ [0.2, 1.8] | 与 HG 主线环境范围一致 |

**推荐**：LHS d = 2 + simplex 映射。**备选**：Dirichlet(α_O2, α_CO2, α_N2)，若希望浓度分布更自然偏斜；但 LHS 提供更好的实验设计均匀性。

### 2.2 模态-气体敏感性矩阵

物理映射（列 = 气体，行 = 模态）：

| 模态 | O2 | CO2 | N2 | 物理基础 |
|------|----|-----|-----|----------|
| **NDIR** | 极弱（中红外无吸收） | **强**（4.26 μm 强吸收带） | 无（中红外透明） | Beer-Lambert：只有 CO2 和 H2O 在中红外 NDIR 窗口有吸收 |
| **TCD** | 中（λ_O2 ≈ 0.026 W/m·K） | 弱（λ_CO2 ≈ 0.017 W/m·K） | 中（λ_N2 ≈ 0.026 W/m·K） | 热导率差异：O2 与 N2 热导接近，CO2 明显偏低 |
| **超声** | 中（M_O2 = 32） | 强（M_CO2 = 44，分子量最大） | 中（M_N2 = 28） | 声速 v ∝ 1/√M_mix；CO2 分子量最高，对声速影响最大 |

**新 W_base 建议**（列规约，非定值——具体数值在训练侧可调，但列语义固定）：

| 目标气体 | NDIR | TCD | US | 每列 sum |
|----------|------|-----|-----|----------|
| O2 | 0.05 | 0.50 | 0.45 | 1.0 |
| CO2 | **0.70** | 0.15 | 0.15 | 1.0 |
| N2 | 0.05 | 0.45 | 0.50 | 1.0 |

**与旧 RCDW W_base 一致**，无需修改。行 = 模态（NDIR/TCD/US），列 = 气体（O2/CO2/N2），每列 sum = 1.0。

### 2.3 光学波段选择

**核心物理事实**：
- CO2 在 4.26 μm（2347 cm⁻¹）有极强的 ν₃ 非对称伸缩振动吸收带。
- O2 是均核双原子分子，无永久偶极矩，中红外完全透明。O2 仅在近红外 760 nm（A-band）有极弱的电子跃迁吸收，但常规 NDIR 不覆盖该波段。
- N2 同样是均核双原子分子，中红外完全透明。
- H2O 在 1–8 μm 全波段有密集吸收线，是主要交叉干扰源。

**决策**：

| 决策项 | 选择 | 理由 |
|--------|------|------|
| NDIR 通道数 | **1 个 CO2 通道**（中心 2347 cm⁻¹，FWHM 93 cm⁻¹） | O2 与 N2 在中红外无吸收，增加 O2 或 N2 专用通道没有物理意义 |
| O2 近红外 A-band 通道 | **不引入** | ① 760 nm 吸收极弱（约 10⁻⁵ 量级），需要长光程或高灵敏度；② 引入会破坏现有 W_base 3×3 假设（需要第 4 模态或混合模态）；③ 与 RCDW 框架「NDIR 仅对 CO2 敏感」的物理前提一致 |
| H2O 交叉敏感性 | **NDIR CO2 通道含 H2O 交叉**：在 HITRAN 计算中同时加载 CO2 和 H2O 光谱，自然产生光谱重叠吸收 | 与 HG 主线 `DEFAULT_HITRAN_GAS_SPECS = (CH4, CO2, H2O)` 的设计一致；RCDW 的 `DEFAULT_HITRAN_GAS_SPECS` 应为 `(CO2, H2O)` |
| NDIR 通道清单 | **仅 `V_NDIR_CO2`** | 物理驱动的极简设计；与旧 RCDW `S_ndir` 通道（索引 0）语义对齐 |

**HITRAN 气体清单**：`(CO2, H2O)`，对应 molecule_id 2 和 1。不再需要 CH4（molecule_id 6）。

### 2.4 光纤麦克风策略

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 生成 | **完整保留** `simulate_fiber_mic_measurement` | 与 HG 同步生成，物理建模一致（干涉仪相位解调 + 声反射 + 噪声），保证数据完整性 |
| 落盘 | `fiber_mic_int16.npy`、`fiber_mic_scale.npy` | 与 HG 目录结构一致；manifest 中记录 `fiber_mic_spec` |
| DataLoader | **不读取** fiber_mic 数组 | Dataset 仅加载 `slow.npy` 与 `ultrasonic_*.npy`；`fiber_mic_*` 文件存在但 DataLoader 不触碰 |
| Scaler 拟合 | **不参与** | `fit_z_score_scalers` 仅对参与训练的 slow 通道计算 mean 与 std；fiber_mic 不在 SLOW_CHANNELS 中，自然不被 scaler 覆盖 |
| Validation 形状检查 | **检查存在性 + 形状** | `validate_benchmark_assets` 验证 `fiber_mic.shape[0] == sequence_count`，但不要求与 slow 第二维一致（waveform_samples 不同） |
| 训练 loss | **不参与** | 光纤麦克风信号不进入任何 loss 项 |

### 2.5 Phase 是否完全照搬 HG 的 5 种 Schedule

**决策**（v1.2 修订）：**仅保留 `STANDARD_EXPOSURE` 一种 schedule，不复制其余 4 种**。

**理由**：

1. **架构上不可达**：§5.1 `generate_condition_rows` 当前为 `mixture:sequence = 1:1` 映射，配置 `stage_profile: "standard_exposure"` 单值固定。其余 4 种 schedule（`VARIABLE_ONSET`、`FAST_TRANSIENT`、`INCOMPLETE_RECOVERY`、`MULTI_PULSE`）即使复制也永远不会被调用。
2. **违反 AGENTS.md 不变量**：项目根 `AGENTS.md` 明确"禁止无依据的防御性编程和隐式兜底逻辑"。复制 4 段不可达代码不属于"暂时不用"，是死代码。
3. **保留成本几乎为零**：HG `src/sim/generation/phases.py` 完整保留这 5 种 schedule 的实现；未来真要激活时 `git show src/sim/generation/phases.py` 复制相关 dataclass 即可，是分钟级动作。
4. **激活路径清晰**：在 §5.2 末尾给出显式三步激活路径，未来扩展时按图索骥即可，不会因为"代码就绪"而误以为已经可用。

**RCDW v1.x 实际保留**：

- `PhaseSegment(name, duration_frac, blend_shape, blend_floor)` dataclass 定义
- `PhaseSchedule(name, segments)` dataclass + 方法（`boundaries()`、`phase_for_timestep()`、`blend_for_timestep()`、`resolve_timeline()`、`jittered()`）
- 仅一个 schedule 实例：`STANDARD_EXPOSURE`（4 段：baseline / exposure / steady / recovery）
- `resolve_phase_schedule(stage_profile)` 对未实现 profile 显式 `raise NotImplementedError`，禁止隐式回退

**与 HG 主线的关系**：HG 多 schedule 是为了配合多 sequence/mixture 提升数据多样性。RCDW 当前 1:1 映射使该机制天然退化；未来若要激活，需要按 §5.2 的扩展路径同步改三处，单独复制 schedule 是无用的。

### 2.6 ID、切分、版本号

| 决策项 | 选择 | 理由 |
|--------|------|------|
| schema_version | `"rcdw-benchmark-1"` | 与 HG 的 `"v4-benchmark-1"` 区分，独立命名空间 |
| MixtureId 格式 | `RCDW-M{index:06d}` | 前缀 `RCDW-` 防止与主线 `M000001` 混淆 |
| SequenceId 格式 | `RCDW-Q{index:06d}` | 同上 |
| BenchmarkDatasetId | 用户指定 slug，如 `rcdw-formal` | 与 HG 的 `wv4-*` 与 syngas 的 `sg4-*` 命名风格一致 |
| composition_scheme | `"rcdw_o2_co2_n2"` | manifest 中记录，下游加载器据此判断组分语义 |
| split 列表 | **仅 train / val / test，不保留 extrapolation** | RCDW 当前是玩具与原理验证场景，不需要外推 holdout；HG 的 extrapolation 是为评估模型在未见组分范围的泛化能力而设计的，RCDW 三组分范围已覆盖全 simplex，外推无物理意义 |
| 切分方式 | mixture_id 分层（shuffle group IDs → 按 70/15/15 比例分配） | 与 HG 一致，保证同一 mixture 的所有 sequence 不跨 split |
| LEGACY 黑名单 | `("base_condition_id", "noise_seed_index", "noise_seed")` | RCDW 不存在历史包袱，但提前声明黑名单，防止未来误引入；validation 阶段检查这些字段不存在 |

---

## 3. 目录与文件结构改动

### 3.1 当前 rcdw_mgda 树

```
rcdw_mgda/
├── pyproject.toml
├── configs/
│   ├── default.yaml
│   └── smoke.yaml
├── rcdw/
│   ├── data/          # synth.py, preprocess.py, __init__.py
│   ├── models/        # single_modal.py, feature.py, error_net.py, rcdw.py
│   ├── training/      # losses.py, metrics.py, stage_a.py, stage_b.py
│   ├── perturbation/  # inject.py
│   └── utils/         # degradation.py, normalize.py
├── scripts/           # train.py, eval.py, perturb.py, numerical_check.py
├── tests/             # 7 个测试文件
└── runs/              # stage_a/, stage_b/, perturb/
```

### 3.2 目标 rcdw_mgda 树

```
rcdw_mgda/
├── pyproject.toml
├── configs/
│   ├── default.yaml              # 扩增 generation/spectral/acoustic/phases/splits/scalers 节
│   ├── smoke.yaml                # 对应缩小
│   └── spectral-defaults.json    # [新增] RCDW 专用光谱默认配置
├── rcdw/
│   ├── data/
│   │   ├── dataset.py            # [新增] 从磁盘目录读取 slow+ultrasonic 的 Dataset
│   │   ├── preprocess.py         # [保留] 滤波/校准（占位，真实数据时启用）
│   │   ├── synth.py              # [废弃] 归档到 rcdw/data/_legacy/ 或直接删除
│   │   └── __init__.py
│   ├── models/                   # [修改] single_modal.py 通道索引重新定义
│   │   ├── single_modal.py
│   │   ├── feature.py            # [修改] 通道索引常量
│   │   ├── error_net.py
│   │   └── rcdw.py
│   ├── training/                 # [不变] Stage A/B 流程不变，DataLoader 来源改为新 Dataset
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   ├── stage_a.py
│   │   └── stage_b.py
│   ├── perturbation/             # [修改] inject.py 通道索引重新映射
│   │   └── inject.py
│   ├── utils/
│   │   ├── degradation.py
│   │   └── normalize.py
│   ├── sim/                      # [新增] 独立数据生成子包
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── schema.py         # [新增] SCHEMA_VERSION / COMPONENT_FIELDS / SLOW_CHANNELS 等
│   │   │   └── ids.py            # [新增] BenchmarkDatasetId / MixtureId / SequenceId
│   │   ├── generation/
│   │   │   ├── __init__.py
│   │   │   ├── conditions.py     # [新增] LHS 采样 + 组分映射
│   │   │   ├── phases.py         # [新增] PhaseSegment / PhaseSchedule（仅保留 STANDARD_EXPOSURE，v1.2 YAGNI）
│   │   │   ├── gas_state.py      # [新增] Magnus 公式 H2O 摩尔分数
│   │   │   ├── slow.py           # [新增] 多时间常数 RC 动力学 + drift + noise
│   │   │   ├── acoustic_physics.py     # [新增] O2/CO2/N2 声速 + 衰减
│   │   │   ├── waveforms.py            # [新增] 超声 + 光纤麦克风波形仿真
│   │   │   ├── optical_backend.py      # [新增] HITRAN 光学吸收计算
│   │   │   ├── optical_crosstalk.py    # [新增] CO2 通道 H2O 交叉
│   │   │   ├── benchmark.py            # [新增] 全流程编排
│   │   │   └── spectral/
│   │   │       ├── __init__.py
│   │   │       ├── defaults.py         # [新增] 从 spectral-defaults.json 加载
│   │   │       ├── filters.py          # [新增] NDIRFilter + gaussian_filter
│   │   │       ├── hitran_backend.py   # [新增] HAPI 调用 + 缓存读写
│   │   │       ├── tabulated_backend.py # [新增] TabulatedSpectrum 吸收计算
│   │   │       ├── integration.py      # [新增] 通道吸光度积分
│   │   │       └── cache.py            # [新增] SpectralCacheKey + 缓存文件读写
│   │   ├── packaging/
│   │   │   ├── __init__.py
│   │   │   ├── arrays.py         # [新增] 落盘 npy + npz
│   │   │   ├── manifest.py       # [新增] build_manifest
│   │   │   ├── index.py          # [新增] sequence_index.csv
│   │   │   ├── scalers.py        # [新增] z-score scaler 拟合
│   │   │   ├── splits.py         # [新增] mixture_id 分层切分
│   │   │   ├── io.py             # [新增] write_csv / write_json
│   │   │   └── constants.py      # [新增] Z_SCORE_STD_EPSILON
│   │   └── validation/
│   │       ├── __init__.py
│   │       └── integrity.py      # [新增] 不变量校验
│   └── __init__.py
├── scripts/
│   ├── train.py                  # [修改] 从新 Dataset 加载数据
│   ├── eval.py                   # [修改] 从新 Dataset 加载数据
│   ├── perturb.py                # [修改] 从新 Dataset 加载数据
│   ├── numerical_check.py        # [保留]
│   └── generate_benchmark.py     # [新增] benchmark 生成入口
├── tests/
│   ├── test_synth.py             # [废弃或重写] 改为 test_dataset_loader.py
│   ├── test_single_modal.py      # [保留]
│   ├── test_feature.py           # [修改] 通道索引适配
│   ├── test_error_net.py         # [保留]
│   ├── test_rcdw_fusion.py       # [保留]
│   ├── test_degradation.py       # [保留]
│   ├── test_perturbation.py      # [修改] 通道索引适配
│   ├── test_conditions.py        # [新增]
│   ├── test_phases.py            # [新增]
│   ├── test_slow.py              # [新增]
│   ├── test_waveforms.py         # [新增]
│   ├── test_optical_backend.py   # [新增]
│   ├── test_packaging.py         # [新增]
│   ├── test_validation.py        # [新增]
│   ├── test_dataset_loader.py    # [新增]
│   └── test_w_base_alignment.py  # [新增]
├── data/
│   └── hitran_cache/             # [新增] RCDW 专用 HITRAN 缓存目录
└── runs/                         # [保留] stage_a/, stage_b/, perturb/
```

### 3.3 模块隔离边界

| 模块 | 可 import | 不可 import |
|------|-----------|-------------|
| `rcdw.sim.core` | `rcdw.sim.generation`、`rcdw.sim.packaging`、`rcdw.sim.validation` | `src.sim` 任何模块 |
| `rcdw.sim.generation` | `rcdw.sim.core` | `src.sim.generation` |
| `rcdw.sim.packaging` | `rcdw.sim.core`、`numpy` | `src.sim.packaging` |
| `rcdw.data.dataset` | `rcdw.sim.core.schema`（读 SLOW_CHANNELS 常量） | `src.sim` |
| `rcdw.models.*` | `rcdw.sim.core.schema`（读通道索引） | `src.sim` |
| `rcdw.training.*` | `rcdw.data.dataset` | `src.dl`、`src.ml` |
| `scripts.generate_benchmark` | `rcdw.sim.*` | `src.sim`、`src.pipeline` |

---

## 4. Schema 与 ID 契约

### 4.1 SCHEMA_VERSION

```
SCHEMA_VERSION = "rcdw-benchmark-1"
```

### 4.2 COMPONENT_FIELDS

```
COMPONENT_FIELDS = ("x_O2", "x_CO2", "x_N2")
```

自由度 d = 2（三组分 sum = 100%，仅两个独立变量）。

### 4.3 SLOW_CHANNELS（全集）

```
SLOW_CHANNELS = (
    "V_NDIR_CO2",         # NDIR CO2 通道电压 [V] — 进训练
    "V_TCS",              # 热导传感器电压 [V] — 进训练
    "T_C",                # 温度 [°C] — 进训练（环境上下文）
    "P_MPa",              # 压力 [MPa] — 进训练（环境上下文）
    "H_RH",               # 相对湿度 [%] — 进训练（环境上下文）
    "L_m",                # 声程 [m] — 进训练（环境上下文）
    "piston_position_m",  # 活塞位置 [m] — 进训练（冗余，用于多光程扫描）
)
```

与 HG 的差异：HG 有 8 个 slow 通道（`V_NDIR_CH4`、`V_NDIR_CO2`、`V_TCS`、`T_C`、`P_MPa`、`H_RH`、`L_m`、`piston_position_m`）；RCDW 删除了 `V_NDIR_CH4`（RCDW 无 CH4），NDIR 仅保留 `V_NDIR_CO2`。共 7 个 slow 通道。

### 4.4 SLOW_DYNAMIC_CHANNELS

```
SLOW_DYNAMIC_CHANNELS = ("V_NDIR_CO2", "V_TCS")
```

仅这两个通道参与 RC 动力学、drift、random walk、噪声建模。环境通道（`T_C`、`P_MPa`、`H_RH`、`L_m`、`piston_position_m`）直接取值，不经过动力学。

### 4.5 SLOW_MODAL_GROUPS

```
SLOW_MODAL_GROUPS = {
    "optical": ("V_NDIR_CO2",),
    "thermal": ("V_TCS",),
    "environment": ("T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"),
}
```

此分组用于 `fit_z_score_scalers` 的 modal scaler 输出——按组分别记录 mean 与 std。

### 4.6 PHASE_NAMES 与 MULTI_PATH_PHASES

```
PHASE_NAMES = ("baseline", "exposure", "steady", "recovery")
MULTI_PATH_PHASES = ("off", "baseline", "steady")
```

与 HG 完全一致。`multi_path_phase` 控制多光程扫描在哪个 phase 进行（默认 `"steady"`）。

### 4.7 CONDITION_GRID_FIELDS / SEQUENCE_INDEX_FIELDS / SEQUENCE_LABEL_FIELDS / SPLIT_FIELDS / SPLIT_NAMES

```
CONDITION_GRID_FIELDS = (
    "sequence_id",
    "mixture_id",
    *COMPONENT_FIELDS,           # x_O2, x_CO2, x_N2
    "T_C_base",
    "P_MPa_base",
    "H_RH_base",
    "L_m_base",
    "status",
)

SEQUENCE_INDEX_FIELDS = (
    "sequence_id",
    "mixture_id",
    "stage_profile",
    "status",
    "n_timesteps",
    "dt_s",
)

SEQUENCE_LABEL_FIELDS = ("sequence_id", *COMPONENT_FIELDS)

SLOW_SEQUENCE_FIELDS = ("sequence_id", "timestep", "timestamp_s", "phase_id", *SLOW_CHANNELS)

SPLIT_FIELDS = ("sequence_id", "mixture_id")

SPLIT_NAMES = ("train", "val", "test")   # 无 extrapolation
```

### 4.8 LEGACY 字段黑名单

```
LEGACY_CONDITION_FIELDS = ("base_condition_id", "noise_seed_index", "noise_seed")
```

与 HG 一致。虽然 RCDW 不存在历史包袱，但提前声明黑名单，validation 阶段检查这些字段不存在于 condition rows 中。

### 4.9 ID 类型

```
@dataclass(frozen=True, slots=True)
class BenchmarkDatasetId:
    value: str  # 例: "rcdw-formal", "rcdw-smoke"

@dataclass(frozen=True, slots=True)
class MixtureId:
    value: str  # 例: "RCDW-M000001"

@dataclass(frozen=True, slots=True)
class SequenceId:
    value: str  # 例: "RCDW-Q000001"

def make_mixture_id(index: int) -> MixtureId:
    return MixtureId(f"RCDW-M{index:06d}")

def make_sequence_id(index: int) -> SequenceId:
    return SequenceId(f"RCDW-Q{index:06d}")
```

---

## 5. 生成阶段（generation）

### 5.1 conditions.py

**文件**：`rcdw_mgda/rcdw/sim/generation/conditions.py`

**接口**：

```
def generate_condition_rows(
    sequence_count: int,
    *,
    seed: int,
    sampling_strategy: str = "lhs",
) -> list[dict[str, str]]:
```

**内部逻辑**：
1. 调用 `_generate_lhs_samples(sequence_count, seed=seed+1)` 生成 N 个 2D unit cube 样本 `(u_o2, u_co2)`。
2. 映射到组分：`x_O2 = u_o2 * 25.0`，`x_CO2 = u_co2 * 20.0`，`x_N2 = 100.0 - x_O2 - x_CO2`。
3. 若 `x_N2 < 55.0`，则等比例缩减 O2 与 CO2 使 `x_N2 = 55.0`（保证 N2 最小 55%）。
4. 环境参数独立随机：`T_C_base ∈ [15, 35]`、`P_MPa_base ∈ [0.10, 0.709]`、`H_RH_base ∈ [20, 80]`、`L_m_base ∈ [0.2, 1.8]`。
5. 每个 condition 同时产生 `mixture_id` 和 `sequence_id`（一对一映射：每个 mixture 仅一个 sequence；这与 HG 不同——HG 中同一 mixture 可有多个 sequence（不同 stage_profile）；RCDW 暂不需要 sequence 级多样性，可后续扩展）。
6. 返回 condition rows 列表。

**辅助函数**：

```
def build_label_rows(conditions: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    # 返回 [{"sequence_id": ..., "x_O2": ..., "x_CO2": ..., "x_N2": ...}]
```

**与 HG 的差异**：
- HG 的 `_sample_components_lhs` 有 3D LHS + H2 双峰分布 + CH4 complement + N2 缩减逻辑；RCDW 为 2D LHS + 简单 simplex 映射。
- HG 的 `_map_hydrogen_lhs` 双峰逻辑不适用于 RCDW（无 H2）。

### 5.2 phases.py

**文件**：`rcdw_mgda/rcdw/sim/generation/phases.py`

**复用 HG 的 dataclass 结构，但仅保留 `STANDARD_EXPOSURE` 一种实例**（v1.2 YAGNI 决策，见 §2.5）。核心类型：

```
@dataclass(frozen=True, slots=True)
class PhaseSegment:
    name: str           # "baseline" | "exposure" | "steady" | "recovery"
    duration_frac: float
    blend_shape: str    # "hold0" | "ramp_up" | "hold1" | "ramp_down"
    blend_floor: float = 0.0

@dataclass(frozen=True, slots=True)
class PhaseSchedule:
    name: str
    segments: tuple[PhaseSegment, ...]
    # 方法：boundaries(timesteps), phase_for_timestep(timestep, timesteps),
    #       blend_for_timestep(timestep, timesteps), resolve_timeline(timesteps),
    #       jittered(rng, jitter_frac), to_dict()
```

**唯一保留的 schedule 实例**：

```
STANDARD_EXPOSURE = PhaseSchedule(
    name="standard_exposure",
    segments=(
        PhaseSegment("baseline", 0.15, "hold0"),
        PhaseSegment("exposure", 0.25, "ramp_up"),
        PhaseSegment("steady",   0.35, "hold1"),
        PhaseSegment("recovery", 0.25, "ramp_down", blend_floor=0.05),
    ),
)
```

**`resolve_phase_schedule` 显式契约**：

```python
def resolve_phase_schedule(stage_profile: str | PhaseSchedule) -> PhaseSchedule:
    if isinstance(stage_profile, PhaseSchedule):
        return stage_profile
    if stage_profile == "standard_exposure":
        return STANDARD_EXPOSURE
    raise NotImplementedError(
        f"RCDW v1.x 仅支持 stage_profile='standard_exposure'，收到 {stage_profile!r}。"
        f"要新增其他 schedule（VARIABLE_ONSET / FAST_TRANSIENT / INCOMPLETE_RECOVERY / MULTI_PULSE 等），"
        f"请按本方案 §5.2 末尾的『未来激活路径』同步改三处后再添加。"
    )
```

**禁止隐式回退**：任何未实现的 `stage_profile` 必须显式 `raise NotImplementedError`，不允许"未识别就默认 standard_exposure"之类的兜底（违反 AGENTS.md）。

#### 未来激活其他 schedule 的路径（三步）

当未来需要数据多样性（如 formal benchmark 6000+ 序列）激活其他 schedule 时，按以下三步执行，不能只复制 schedule 实例：

1. **`generate_condition_rows` 改 1:N**：将 mixture:sequence 映射从 1:1 改为 1:N，每个 mixture 生成 N 个 sequence，每个 sequence 分配不同 `stage_profile`。`condition_grid` 字段中加入 `stage_profile` 列（HG 主线已是这种写法）。
2. **`phases.py` 增加 schedule 实例**：从 HG `src/sim/generation/phases.py` 拷回需要的 schedule（`VARIABLE_ONSET` 等），同时扩展 `resolve_phase_schedule` 的分支查找。
3. **`splits.py` 确认 mixture 分组保护**：v1.x 切分已经按 `mixture_id` 分组（见 §6.4），同 mixture 的多 sequence 自动不跨 split，无需额外改动；但要在 validation 中加 "每个 mixture 的 sequence 数 == 配置 N" 不变量。

**注意**：`MULTI_PULSE` 的 12 段结构（3 个脉冲周期）需要 `timesteps ≥ 12`。若未来激活该 schedule，smoke 配置的 `timesteps` 必须同步上调。v1.2 当前仅用 `STANDARD_EXPOSURE`（4 段），smoke 配置 `timesteps=32` 已充裕。

### 5.3 gas_state.py

**文件**：`rcdw_mgda/rcdw/sim/generation/gas_state.py`

**完全复用 HG 的 gas_state.py**（Magnus 公式计算 H2O 摩尔分数、以及温度、压力单位转换）。

```
MPA_PER_ATM = 0.101325

def h2o_mole_percent_from_rh(t_c: float, p_mpa: float, h_rh: float) -> float:
    # Magnus 公式：p_sat = 0.61121 * exp(17.502 * T / (240.97 + T)) [kPa]
    # H2O% = (RH/100) * (p_sat / p_amb) * 100
    # clamped to [0, 5]%

def hitran_temperature_k(t_c: float) -> float:
    return round(t_c + 273.15, 3)

def hitran_pressure_atm(p_mpa: float) -> float:
    return round(p_mpa / MPA_PER_ATM, 6)
```

**与旧 RCDW 的差异**：旧 `synth.py` 使用硬编码 `rh_to_water_vol = 0.0355`（H2O 体积分数 3.55%）；新代码使用 Magnus 公式，H2O 含量由 T、P、RH 物理推导。

### 5.4 slow.py

**文件**：`rcdw_mgda/rcdw/sim/generation/slow.py`

**核心接口**：

```
def build_sequence_arrays(
    conditions: list[dict[str, str]],
    *,
    timesteps: int,
    dt_s: float,
    seed: int,
    multi_path_phase: str,
    ultrasonic_spec: WaveformSpec,
    fiber_mic_spec: FiberMicSpec,
    path_lms: tuple[float, ...],
    phase_schedule: str | PhaseSchedule = "standard_exposure",
    stage_jitter: float = 0.0,
    optical_absorption_backend: str = "hitran_hapi_v1",
    hitran_cache_root: str = "data/hitran_cache",
    start_sequence_index: int = 0,
) -> dict[str, object]:
```

**返回字典**：与 HG 完全一致（见 HG `slow.py` 的返回），包含 slow、ultrasonic、fiber_mic 数组以及 slow_rows。

**通道动力学参数表**（RCDW 版）：

```
TAU_RISE_SYSTEM_S = {
    "V_NDIR_CO2": (6.0, 18.0),
    "V_TCS": (10.0, 35.0),
}
TAU_DECAY_SYSTEM_S = {
    "V_NDIR_CO2": (10.0, 28.0),
    "V_TCS": (20.0, 60.0),
}
NOISE_FRACTION = {"V_NDIR_CO2": 0.0025, "V_TCS": 0.003}
```

**与 HG 的差异**：
1. 删除 `V_NDIR_CH4` 相关参数。
2. `_main_feature_condition` 的组分字段改为 `x_O2 / x_CO2 / x_N2`。
3. `_blend_composition` 的 blend 映射改为三组分：

   > **Baseline 状态物理定义**（审查修正 v1.1 补充）：RCDW 的 baseline 状态定义为 **100% N₂ 纯背景气**（与 HG 主线一致）。blend=0 时所有目标气体浓度归零，N₂ 充满腔室；blend=1 时恢复到采样目标浓度。这对应真实标定流程中 "通纯 N₂ 建立基线 → 切换目标混合气 → 达到稳态 → 恢复纯 N₂" 的物理过程。

   ```
   def _blend_composition(condition, blend):
       return {
           "x_o2":  float(condition["x_O2"])  * blend,
           "x_co2": float(condition["x_CO2"]) * blend,
           "x_n2":  100.0 + (float(condition["x_N2"]) - 100.0) * blend,
       }
   ```
   baseline 全 0，exposure target = condition 中采样的组分。
4. `_hitran_ndir_equilibrium` 仅处理 `V_NDIR_CO2`（删除 CH4 通道），调用 `compute_hitran_optical_absorption` 时只传 CO2 与 H2O 浓度。
5. `_slow_row` 中字段名从 `V_NDIR_CH4` / `V_NDIR_CO2` 改为仅 `V_NDIR_CO2`。

**保留的多时间常数动力学**：
- `_multi_tau_channel_step`（fast_tau + slow_tau 双指数 + recovery floor）
- `_channel_dynamic_params`（每个通道独立采样的 tau_rise、tau_decay、fast_tau_fraction、slow_tau_multiplier、fast_response_weight、recovery_floor_fraction、noise_sigma、random_walk_sigma、drift_slope）
- `_dynamic_features_from_equilibrium`（逐时间步 RC + drift + random walk + 高斯噪声）

> **不移植 legacy empirical 路径**（审查修正 v1.1 补充）：HG `slow.py` 内部存在两条动力学路径：legacy empirical（单指数 RC，仅在 `standard_exposure` + 无 jitter + `empirical` 后端时触发）和 multi-tau equilibrium（双指数 RC + recovery floor + random walk，所有其他情况）。RCDW 使用 `hitran_hapi_v1` 后端，永远不会命中 legacy 路径的触发条件，因此**仅移植 multi-tau equilibrium 路径**，不移植 legacy empirical。

**RNG seeding 策略**（审查修正 v1.1 补充）：

与 HG 主线一致，RCDW 采用 **blake2b 哈希 + 双流 RNG** 策略保证多进程并行下的确定性复现。核心函数（在 rcdw_mgda 内独立重写，不 import 主线）：

```
def _stable_uint32(seed: int, sequence_index: int, stream_name: str) -> int:
    """基于 blake2b 哈希生成确定性 uint32 种子。

    保证同一 (seed, sequence_index, stream_name) 组合
    无论 chunk 划分或 workers 数量如何变化，都产生相同结果。
    """
    data = f"{seed}:{sequence_index}:{stream_name}".encode()
    h = hashlib.blake2b(data, digest_size=4)
    return int.from_bytes(h.digest(), "little")
```

每个 sequence 通过 `_stable_uint32` 获得两个独立 RNG 流：
- `"condition"` 流：用于环境参数（T、P、RH、L）的随机化
- `"sequence"` 流：用于动力学参数采样（tau、noise、drift）和波形噪声

**禁止使用 `np.random.default_rng(seed + i)` 等简单偏移方案**——改变 `workers` 或 `chunk_size` 时会导致不同 chunk 边界引起的 RNG 流漂移，使结果不可复现。

### 5.5 acoustic_physics.py

**文件**：`rcdw_mgda/rcdw/sim/generation/acoustic_physics.py`

**核心函数**：

```
def rcdw_sound_speed(x_o2: float, x_co2: float, x_n2: float, t_c: float) -> float:
```

**声速计算**（摩尔质量加权线性近似 + 温度修正）：

| 气体 | 摩尔质量 (g/mol) | 声速系数 (m/s) |
|------|------------------|----------------|
| O2 | 32.0 | `_SPEED_O2_MS` TBD（实测校核，暂用 330.0） |
| CO2 | 44.0 | `_SPEED_CO2_MS` TBD（实测校核，暂用 268.0） |
| N2 | 28.0 | `_SPEED_N2_MS` TBD（实测校核，暂用 353.0） |

公式：`c_mix = x_o2_frac * c_O2 + x_co2_frac * c_CO2 + x_n2_frac * c_N2 + 0.6 * (t_c - 25.0)`，clamp min = 200。

> **声学模型版本标注**（审查修正 v1.1 补充）：在 `acoustic_model_metadata` 中记录 `"model": "linear_mixing_v1"`，以便后续对比或替换为更复杂模型（如理想气体 `c = sqrt(γRT/M_mix)` + 弛豫修正）时可审计。O₂ 声速暂定 330.0 m/s，待 Phase 2 实施时确认为 329.5 m/s（NIST Chemistry WebBook @ 25°C, 1 atm）并标注来源。

```
def rcdw_attenuation(
    x_o2: float, x_co2: float, x_n2: float,
    t_c: float, p_mpa: float, h_rh: float,
    *, c_mix: float | None = None, f_hz: float | None = None,
) -> dict[str, float]:
```

**衰减计算**（经典吸收 + 弛豫吸收 + 扩散吸收）：

| 贡献项 | RCDW 来源 | 参数 |
|--------|-----------|------|
| α_classical | 经典 Stokes-Kirchhoff | `alpha_classical_K_ref = 1.84e-11`（与 HG 同） |
| α_CO2 | CO2 振动弛豫 | `alpha_lambda_max_co2 = 0.12`、`f_relax_co2_per_atm = 28000.0`、`k_h2o_to_f_relax_co2 = 0.015`（与 HG 同） |
| α_N2 | N2 振动弛豫 | `alpha_lambda_max_n2 = 0.004`、`f_relax_n2_per_atm = 65000.0`（与 HG 同） |
| α_O2 | O2 振动弛豫 | **新增**：`alpha_lambda_max_o2` TBD（暂用 0.002）、`f_relax_o2_per_atm` TBD（暂用 50000.0）。O2 弛豫频率在 40 kHz 附近有文献值，需实测校核 |
| α_H2O | H2O 振动弛豫 | `alpha_lambda_max_h2o = 0.01`、`f_relax_h2o_per_atm = 100000.0`（与 HG 同） |
| α_diffusion | 扩散吸收 | **删除**（HG 的 `alpha_diff_npm` 仅与 H2 相关，RCDW 无 H2，扩散贡献极小，忽略） |

**与 HG 的差异**：
1. 声速公式参数从 H2/CH4/CO2/N2 四组分改为 O2/CO2/N2 三组分。
2. 衰减公式新增 O2 弛豫项，删除 H2 扩散项和 CH4 弛豫项。
3. `main_sensor_features` 和 `thermal_conductivity_sensor_feature` 改为 RCDW 版（组分字段适配 + 删除 `V_NDIR_CH4` 相关计算）。

**热导传感器**（`thermal_conductivity_sensor_feature`）：

```
def rcdw_thermal_conductivity_sensor_feature(condition, rng) -> dict[str, float]:
```

热导率混合模型：`λ_mix = 0.026 * x_o2_frac + 0.017 * x_co2_frac + 0.026 * x_n2_frac`（W/m·K @ 300K），加上温度、压力漂移与噪声。TCS 电压通过 `PROCESSING_PARAMS` 中的 `tcs_response_slope`、`tcs_lambda_offset`、`tcs_temperature_response` 转换。

### 5.6 waveforms.py

**文件**：`rcdw_mgda/rcdw/sim/generation/waveforms.py`

**完全复用 HG 的 waveforms.py**，仅需修改以下内容：

1. `simulate_waveform_measurement` 和 `simulate_fiber_mic_measurement` 的签名改为 RCDW 组分：`x_o2, x_co2, x_n2`（删除 `x_h2, x_ch4`）。
2. `_compute_physics` 内部调用 `rcdw_sound_speed` 和 `rcdw_attenuation`。
3. 其余逻辑（burst pulse 生成、transducer 响应、数字化、TOF 估计、光纤麦克风干涉仪解调、声反射）完全不变。

**保留的常量**（与 HG 一致）：
- `CENTER_FREQUENCY_HZ = 40000.0`
- `BURST_CYCLES = 8`
- `SAMPLE_RATE_HZ = 200000`
- `ULTRASONIC_MEASUREMENT_WINDOW_S = 0.005`
- `FIBER_MIC_MEASUREMENT_WINDOW_S = 0.010`
- `ADC_MAX_INT16 = 32767`

**保留的数据类**：`WaveformSpec`、`FiberProbeSpec`、`FiberMicSpec`（完全不变）。

### 5.7 optical_backend.py 与 spectral 子包

**文件结构**：

```
rcdw_mgda/rcdw/sim/generation/
├── optical_backend.py      # 主入口：compute_hitran_optical_absorption
├── optical_crosstalk.py    # CO2 通道 H2O 交叉
└── spectral/
    ├── defaults.py             # 从 configs/spectral-defaults.json 加载
    ├── filters.py              # NDIRFilter + gaussian_filter
    ├── hitran_backend.py       # HAPI 调用 + HitranGasSpec/HitranGridSpec
    ├── tabulated_backend.py    # TabulatedSpectrum + 吸光度计算
    ├── integration.py          # 通道吸光度积分
    └── cache.py                # SpectralCacheKey + 缓存读写
```

**spectral 子包**：完全复用 HG 的 `src/sim/generation/spectral/` 代码，仅修改：
1. `defaults.py` 的配置文件路径指向 `rcdw_mgda/configs/spectral-defaults.json`。
2. `DEFAULT_HITRAN_GAS_SPECS` 改为 `(CO2, H2O)`（删除 CH4）。

**optical_backend.py**：
- `DEFAULT_HITRAN_GAS_SPECS`：`(CO2, H2O)`，对应 molecule_id = 2 和 1。
- `compute_hitran_optical_absorption` 的 `concentrations_pct` 仅包含 `{"CO2": ..., "H2O": ...}`。
- NDIR 通道：仅 `co2` 通道（`get_default_ndir_filter("co2")` 返回 2347 cm⁻¹ / 93 cm⁻¹ FWHM）。
- 删除 `ch4` 通道相关逻辑。
- `collect_hitran_cache_requirements` 的 `channels` 参数默认改为 `("co2",)`。

**HITRAN 缓存策略**：
- 缓存目录：`rcdw_mgda/data/hitran_cache/`（与主线 `data/hitran_cache/` 隔离）。
- 缓存文件命名规则与 HG 一致（`{backend}__{gas}__{version}__{wmin}_{wmax}_{step}__T{temp}__P{press}.npz`）。
- 预计算：`pipeline.precompute_hitran_benchmark_cache` 的 RCDW 版本需要新 CLI 入口。
- 首次运行需下载 HAPI 数据（`hapi.fetch`），约几 MB（仅 CO2 + H2O，比 HG 的 CH4 + CO2 + H2O 更小）。

**spectral-defaults.json**（RCDW 版）：

```json
{
  "optical_absorption_backend": "hitran_hapi_v1",
  "gas_specs": [
    {"gas": "CO2", "table_name": "CO2", "molecule_id": 2, "isotopologue_id": 1},
    {"gas": "H2O", "table_name": "H2O", "molecule_id": 1, "isotopologue_id": 1}
  ],
  "filters": {
    "co2": {"channel": "co2", "center_cm1": 2347.0, "fwhm_cm1": 93.0}
  },
  "hitran_grids": {
    "co2": {
      "wavenumber_min_cm1": 2250.0,
      "wavenumber_max_cm1": 2445.0,
      "wavenumber_step_cm1": 0.1,
      "temperature_k": 296.0,
      "pressure_atm": 1.0
    }
  }
}
```

### 5.8 optical_crosstalk.py

**文件**：`rcdw_mgda/rcdw/sim/generation/optical_crosstalk.py`

**简化版**：仅定义 CO2 通道对 H2O 的交叉敏感性（O2 和 N2 在中红外无吸收，交叉系数 = 0）。

```
@dataclass(frozen=True, slots=True)
class OpticalCrosstalkSpec:
    co2_channel_h2o_response: float = 0.012  # H2O → CO2 通道交叉系数

def apply_optical_crosstalk(
    *,
    absorption_co2: float,
    absorption_h2o: float,
    spec: OpticalCrosstalkSpec = DEFAULT_OPTICAL_CROSSTALK_SPEC,
) -> dict[str, float]:
    # absorption_co2_observed = absorption_co2 + spec.co2_channel_h2o_response * absorption_h2o
```

**与 HG 的差异**：HG 有 CH4↔CO2 双向交叉；RCDW 仅 CO2 通道，交叉仅来自 H2O。

### 5.9 benchmark.py

**文件**：`rcdw_mgda/rcdw/sim/generation/benchmark.py`

**完全复用 HG 的 benchmark.py 编排逻辑**，差异：
1. 导入来源全部改为 `rcdw.sim.*`。
2. `conditions` 调用 `rcdw.sim.generation.conditions.generate_condition_rows`。
3. `phase_schedule` 调用 `rcdw.sim.generation.phases.resolve_phase_schedule`。
4. `build_sequence_arrays` 调用 `rcdw.sim.generation.slow.build_sequence_arrays`。
5. `validate_hitran_benchmark_cache` 调用 `rcdw.sim.generation.optical_backend.validate_hitran_benchmark_cache`。
6. `SPLIT_NAMES` 仅含 `("train", "val", "test")`（无 extrapolation）。
7. `COMPONENT_FIELDS` 为 `("x_O2", "x_CO2", "x_N2")`。
8. 多进程并行逻辑（`ProcessPoolExecutor` + chunk）保持不变。

**核心入口**：

```
@dataclass(frozen=True, slots=True)
class BenchmarkGenerationSpec:
    dataset_slug: str
    sequence_count: int
    seed: int
    timesteps: int = 128
    dt_s: float = 0.5
    storage: str = "memmap"
    multi_path_phase: str = "steady"
    stage_profile: str = "standard_exposure"
    stage_jitter: float = 0.0
    sampling_strategy: str = "lhs"
    path_lms: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40)
    optical_absorption_backend: str = "hitran_hapi_v1"
    hitran_cache_root: str = "data/hitran_cache"
    workers: int = 1
    chunk_size: int | None = None
    temp_dir: str | None = None
    keep_chunks: bool = False

def generate_benchmark_dataset(output_root: Path | str, spec: BenchmarkGenerationSpec) -> dict[str, object]:
```

---

## 6. 打包阶段（packaging）

### 6.1 arrays.py

**文件**：`rcdw_mgda/rcdw/sim/packaging/arrays.py`

**接口**：

```
def write_arrays(
    output_dir: Path,
    arrays: dict[str, object],
    labels: np.ndarray,
    sequence_ids: list[str],
    slow_channel_names: tuple[str, ...],
    label_names: tuple[str, ...],
    storage: str,
) -> dict[str, list[int]]:
```

**落盘清单**：

| 文件 | 内容 | shape | dtype | 进训练 |
|------|------|-------|-------|--------|
| `sequences/slow.npy` | 慢通道时序 | (N_seq, T, 7) | float32 | **是** |
| `sequences/ultrasonic_int16.npy` | 超声波形 | (N_seq, T, W_us) | int16 | 否（但元数据进训练） |
| `sequences/ultrasonic_scale.npy` | 超声量化 scale | (N_seq, T) | float32 | 否 |
| `sequences/ultrasonic_tof_s.npy` | TOF 真值 | (N_seq, T) | float32 | **是** |
| `sequences/ultrasonic_tof_observed_s.npy` | TOF 观测值 | (N_seq, T) | float32 | **是** |
| `sequences/ultrasonic_peak_index.npy` | 峰值索引 | (N_seq, T) | int32 | **是** |
| `sequences/ultrasonic_sound_speed_m_per_s.npy` | 声速真值 | (N_seq, T) | float32 | **是** |
| `sequences/ultrasonic_sound_speed_estimated_m_per_s.npy` | 声速估计值 | (N_seq, T) | float32 | **是** |
| `sequences/ultrasonic_alpha_true_npm.npy` | 衰减系数 | (N_seq, T) | float32 | 否 |
| `sequences/ultrasonic_tof_quality.npy` | TOF 质量 | (N_seq, T) | float32 | **是** |
| `sequences/ultrasonic_tof_accepted.npy` | TOF 接受标志 | (N_seq, T) | int8 | **是** |
| `sequences/fiber_mic_int16.npy` | 光纤麦克风波形 | (N_seq, T, W_fm) | int16 | **否** |
| `sequences/fiber_mic_scale.npy` | 光纤麦克风 scale | (N_seq, T) | float32 | **否** |
| `labels/y.npy` | 标签 | (N_seq, 3) | float32 | **是** |
| `metadata/sequence_ids.npy` | 序列 ID | (N_seq,) | object | 否 |
| `metadata/slow_channel_names.npy` | 通道名 | (7,) | object | 否 |
| `metadata/label_names.npy` | 标签名 | (3,) | object | 否 |

**明确不进训练的数组**：`fiber_mic_int16.npy`、`fiber_mic_scale.npy`、`ultrasonic_int16.npy`（原始波形）、`ultrasonic_scale.npy`、`ultrasonic_alpha_true_npm.npy`。

> **审查修正（v1.1）**：`ultrasonic_tof_quality.npy` 与 `ultrasonic_tof_accepted.npy` 原标"不进训练"，但 §8.2 通道布局将其拼入 12 维输入张量（IDX_US_QUALITY=10, IDX_US_ACCEPTED=11）。两处结论互斥，现统一为"进训练"。理由：tof_quality 和 tof_accepted 作为超声测量置信度信号，对 ErrorNet 判断超声模态可靠性有直接帮助，物理上合理。

### 6.2 manifest.json

**文件**：`rcdw_mgda/rcdw/sim/packaging/manifest.py`

**接口**：

```
def build_manifest(
    *,
    dataset_slug: str,
    sequence_count: int,
    seed: int,
    timesteps: int,
    dt_s: float,
    storage: str,
    multi_path_phase: str,
    stage_profile: str,
    stage_jitter: float,
    phase_schedule: dict[str, object],
    sampling_strategy: str,
    path_lms: tuple[float, ...],
    optical_absorption_backend: str,
    shapes: dict[str, list[int]],
    slow_channels: tuple[str, ...],
    labels: tuple[str, ...],
    optical_absorption_metadata: dict[str, object] | None = None,
    acoustic_model_metadata: dict[str, object] | None = None,
    schema_version: str = "rcdw-benchmark-1",
    composition_scheme: str = "rcdw_o2_co2_n2",
    background_fields: tuple[str, ...] = (),
) -> dict[str, object]:
```

**manifest.json 关键字段**：

```json
{
  "schema_version": "rcdw-benchmark-1",
  "composition_scheme": "rcdw_o2_co2_n2",
  "dataset_slug": "rcdw-formal",
  "primary_key": "mixture_id",
  "instance_key": "sequence_id",
  "split_group_field": "mixture_id",
  "slow_channels": ["V_NDIR_CO2", "V_TCS", "T_C", "P_MPa", "H_RH", "L_m", "piston_position_m"],
  "labels": ["x_O2", "x_CO2", "x_N2"],
  "background_fields": [],
  "train_modalities": ["slow", "ultrasonic"],
  "fiber_mic_spec": { "...": "..." }
}
```

### 6.3 sequence_index.csv 与 sequence_labels.csv

与 HG 一致，仅字段名适配 RCDW schema。

### 6.4 splits.py

**文件**：`rcdw_mgda/rcdw/sim/packaging/splits.py`

**接口**：

```
def build_default_split_rows(
    conditions: list[Mapping[str, object]],
    *,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
```

**逻辑**：
1. 提取所有 `mixture_id`，shuffle。
2. 按 70/15/15 比例分配 train、val、test。
3. **不生成 extrapolation split**。
4. 返回 `{"train": [...], "val": [...], "test": [...]}`。

> **与 HG 主线的比例差异**（审查修正 v1.1 补充）：HG 主线 `build_split_groups` 的默认比例为 train=70% / val=15% / test=10%，剩余 ~5% 归 extrapolation split。RCDW 删除了 extrapolation split（见 §2.6 决策：三组分范围已覆盖全 simplex，外推 holdout 无物理意义），将其份额并入 test，最终比例为 **70/15/15**。实现时需确保 `build_split_groups` 的 `test_ratio` 默认值为 **0.15** 而非照搬 HG 的 0.10。

### 6.5 scalers.py

**文件**：`rcdw_mgda/rcdw/sim/packaging/scalers.py`

**接口**：

```
def fit_z_score_scalers(
    matrix: np.ndarray,                           # (N_seq, T, C) slow 数组
    train_indexes: list[int],                     # train split 的序列索引
    channel_names: tuple[str, ...],               # SLOW_CHANNELS
    modal_groups: dict[str, tuple[str, ...]],     # SLOW_MODAL_GROUPS
    transform_target: str = "slow",
) -> tuple[dict[str, object], dict[str, object]]:
    # 返回 (sequence_scaler, modal_scaler)
```

**关键约束**：scaler 仅对 `SLOW_CHANNELS`（7 个 slow 通道）拟合，不包含 fiber_mic、ultrasonic 原始波形。`train_indexes` 仅使用 train split 的序列索引。

#### 异质通道归一化策略（v1.2 新增）

§8.2 的 12 维输入张量是 slow（7）+ ultrasonic 元数据（5）的拼接，其中部分通道并非连续物理量（[0,1] 评分、0/1 二值、离散下标）。对这些通道做 Z-score 会丢失物理语义（例如 `tof_accepted` 接近恒定 1，Z-score 后变成高方差伪噪声；`tof_quality` ∈ [0,1] 的可解释性消失），损害 ErrorNet 未来读取置信度通道的下游用途。

为此 RCDW 明确锁定每个通道的归一化策略（实现时按此表分组拟合）：

| 索引 | 通道 | 类别 | 策略 | 当前消费者 |
|------|------|------|------|-----------|
| 0 | V_NDIR_CO2 | 连续物理量（光学读数） | train-only Z-score | NDIRNet + FeatureExtractor |
| 1 | V_TCS | 连续物理量（热导读数） | train-only Z-score | TCDNet + FeatureExtractor |
| 2 | T_C | 连续物理量（温度） | train-only Z-score | 所有分支（环境） |
| 3 | P_MPa | 连续物理量（压力） | train-only Z-score | 所有分支（环境） |
| 4 | H_RH | 连续物理量（湿度） | train-only Z-score | 所有分支（环境） |
| 5 | L_m | 连续物理量（光程） | train-only Z-score | 预留 |
| 6 | piston_position_m | 连续准周期量 | train-only Z-score | 预留 |
| 7 | ultrasonic_tof_observed_s | 连续物理量（飞行时间） | train-only Z-score | 预留 |
| 8 | ultrasonic_sound_speed_estimated | 连续物理量（声速） | train-only Z-score | USNet + FeatureExtractor |
| 9 | ultrasonic_peak_index | **离散整数下标** | **跳过归一化**（或除以 timesteps 改为 [0,1]） | 预留 |
| 10 | ultrasonic_tof_quality | **[0,1] 评分** | **跳过归一化**（保留物理可解释性） | 预留（ErrorNet 未来直读） |
| 11 | ultrasonic_tof_accepted | **0/1 二值** | **跳过归一化**（保留 0/1 语义） | 预留（ErrorNet 未来直读） |

**实现要点**：

1. `fit_z_score_scalers` 接收一个 `skip_channels: tuple[str, ...]` 参数（默认 `("ultrasonic_peak_index", "ultrasonic_tof_quality", "ultrasonic_tof_accepted")`），对这些通道不拟合 `(mean, std)`，应用阶段 `transform()` 对这些通道直接 passthrough。
2. `peak_index` 的处理方式（跳过 vs 除以 timesteps）记录在 manifest `scaler_metadata.peak_index_strategy` 字段中，默认 `"skip"`；若未来 ErrorNet 需要将其作为连续输入，可切换为 `"normalized_by_timesteps"`，但必须更新 manifest。
3. `sequence_scaler` 与 `modal_scaler` 输出字典中，跳过通道的条目仍要记录 `{"strategy": "passthrough"}`，便于 validation 检查"无遗漏字段"。
4. validation 阶段（§7.1）增加不变量："跳过归一化的通道在 scaler 输出字典中以 `strategy=passthrough` 显式标记"。

### 6.6 io.py 与 constants.py

完全复用 HG 的 `io.py`（`write_csv` / `write_json`）和 `constants.py`（`Z_SCORE_STD_EPSILON = 1e-12`），无需修改。

---

## 7. 校验阶段（validation）

### 7.1 integrity.py

**文件**：`rcdw_mgda/rcdw/sim/validation/integrity.py`

**接口**：

```
def validate_benchmark_assets(
    conditions: list[dict[str, str]],
    split_rows: dict[str, list[dict[str, str]]],
    arrays: dict[str, object] | None = None,
    labels: np.ndarray | None = None,
    *,
    component_fields: tuple[str, ...] = ("x_O2", "x_CO2", "x_N2"),
    slow_channels: tuple[str, ...] = SLOW_CHANNELS,
    background_fields: tuple[str, ...] = (),
    require_sum_100: bool = True,
) -> dict[str, object]:
```

**不变量列表**：

| # | 检查项 | 条件 |
|---|--------|------|
| 1 | 无 LEGACY 字段 | `LEGACY_CONDITION_FIELDS` 中任何字段不得出现在 condition rows 中 |
| 2 | sequence_id 唯一 | 所有 condition 的 `sequence_id` 无重复 |
| 3 | mixture_id 唯一 | 所有 condition 的 `mixture_id` 无重复 |
| 4 | 组分和 = 100% ± 1e-5 | `x_O2 + x_CO2 + x_N2` 与 100 的差 < 1e-5 |
| 5 | split 覆盖完整 | 所有 sequence_id 恰好出现在一个 split 中，无遗漏、无重复 |
| 6 | slow 数组形状 | `slow.shape[0] == N_seq`、`slow.shape[2] == len(SLOW_CHANNELS)` |
| 7 | labels 形状 | `labels.shape == (N_seq, 3)` |
| 8 | ultrasonic 数组序列轴 | `ultrasonic.shape[0] == N_seq` |
| 9 | fiber_mic 数组序列轴 | `fiber_mic.shape[0] == N_seq` |
| 10 | ultrasonic 元数组形状 | `ultrasonic_tof_s` 等形状 `== slow.shape[:2]` |

### 7.2 fiber_mic 的特殊校验

- 检查 `fiber_mic` 与 `fiber_mic_scale` 存在且序列轴一致（不要求与 slow timesteps 一致——waveform_samples 维度不同）。
- 不检查 fiber_mic 是否被 scaler 覆盖（明确不进 scaler）。
- 不检查 fiber_mic 是否被 DataLoader 读取（由 Dataset 实现保证）。

### 7.3 校验点插入顺序

在 `benchmark.py` 的 `generate_benchmark_dataset` 中：
1. 生成 conditions → 立即校验（无 LEGACY、ID 唯一、组分和）。
2. 生成 split_rows → 校验 split 覆盖。
3. 生成 arrays + labels → 校验形状。
4. 写入 manifest → 校验 manifest 字段完整性（可选，由 `build_manifest` 内部保证）。

---

## 8. 训练侧适配

### 8.1 新的 Dataset

**文件**：`rcdw_mgda/rcdw/data/dataset.py`

**接口**：

```
class BenchmarkDataset(Dataset):
    def __init__(
        self,
        data_root: Path | str,
        split: str,                        # "train" | "val" | "test"
        window: int = 8,
        modalities: tuple[str, ...] = ("slow", "ultrasonic"),
    ):
        # 1. 读取 manifest.json → 获取 shapes, slow_channels, labels
        # 2. 读取 splits/{split}.csv → 获取该 split 的 sequence_ids
        # 3. 按需 mmap 加载 slow.npy + ultrasonic_*.npy
        # 4. 读取 labels/y.npy
        # 5. 构建 sequence_id → index 映射
        # 6. 将整个时序切分为 (N_windows, L, C) 滑窗

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 返回 (x_window, y_label)
        # x_window: (L, C) 滑窗
        # y_label:  (3,) 组分标签（取窗口最后时刻）

    def __len__(self) -> int:
        # 返回窗口总数
```

**关键设计**：
- 不读取 `fiber_mic_*` 文件。
- `modalities` 参数控制加载哪些模态数据（默认 `("slow", "ultrasonic")`）。
- 超声模态不加载原始 `int16` 波形，而是加载 `ultrasonic_tof_s`、`ultrasonic_tof_observed_s`、`ultrasonic_sound_speed_m_per_s`、`ultrasonic_sound_speed_estimated_m_per_s`、`ultrasonic_peak_index` 等元数据数组。
- 窗口切分与旧 `make_windows` 逻辑一致：标签取窗口最后一个时刻的组分。

### 8.2 通道布局重新设计

旧 RCDW 通道布局（`synth.py`）：

```
(B, L, 6) = [S_ndir, S_tc, S_us, P, T, RH]
索引:         0        1      2     3  4  5
```

新 RCDW 通道布局（合并 slow + ultrasonic 元数据）：

**方案 A（推荐）**：将 slow 通道（7 维）与 ultrasonic 元数据通道合并为一个拼接张量，作为模型的统一输入。

新输入维度：

```
(B, L, C) 其中 C 由以下拼接组成：
  slow 通道 (7):       V_NDIR_CO2, V_TCS, T_C, P_MPa, H_RH, L_m, piston_position_m
  ultrasonic 元数据 (5): ultrasonic_tof_observed_s,
                          ultrasonic_sound_speed_estimated_m_per_s,
                          ultrasonic_peak_index,
                          ultrasonic_tof_quality,
                          ultrasonic_tof_accepted
  -----------------
  合计 C = 12
```

通道索引常量（`rcdw/models/single_modal.py` 中重新定义）：

```
# Slow 通道
IDX_NDIR_CO2 = 0        # V_NDIR_CO2
IDX_TCS      = 1        # V_TCS
IDX_T_C      = 2        # T_C
IDX_P_MPa    = 3        # P_MPa
IDX_H_RH     = 4        # H_RH
IDX_L_m      = 5        # L_m
IDX_PISTON   = 6        # piston_position_m

# Ultrasonic 元数据
IDX_US_TOF      = 7     # ultrasonic_tof_observed_s
IDX_US_SPEED    = 8     # ultrasonic_sound_speed_estimated_m_per_s
IDX_US_PEAK     = 9     # ultrasonic_peak_index
IDX_US_QUALITY  = 10    # ultrasonic_tof_quality
IDX_US_ACCEPTED = 11    # ultrasonic_tof_accepted

ENV_INDICES = [IDX_T_C, IDX_P_MPa, IDX_H_RH]  # 环境上下文 [2, 3, 4]
```

**方案 B（备选）**：保持 slow 通道 7 维不变，ultrasonic 元数据作为独立输入传给 USNet。但这会破坏 `(B, L, C)` 的单一输入假设，需要修改模型 forward 签名。

**推荐方案 A**：单一张量输入，模型内部按索引拆分。这与旧 RCDW 的 `(B, L, 6)` 输入模式一致，只需修改维度从 6 到 12。

> **环境变量顺序变更说明**（审查修正 v1.1 补充）：旧版 `ENV_INDICES = [3, 4, 5]` 对应 `[P, T, RH]`；新版 `ENV_INDICES = [2, 3, 4]` 对应 `[T_C, P_MPa, H_RH]`。变化包含两个维度：索引位置（3,4,5 → 2,3,4）和语义顺序（P,T,RH → T,P,RH）。这是有意调整以匹配 SLOW_CHANNELS 的自然顺序。由于 `SingleModal` 是无结构 MLP，顺序改变不影响学习能力，但**所有历史 ckpt 不兼容，必须删除并重训**。

#### 未被模型直接读取的预留通道（v1.2 新增说明）

12 维输入中 6 个通道当前未被任何模块（NDIRNet / TCDNet / USNet / FeatureExtractor / ErrorNet）读取，但仍纳入张量布局以避免下游索引平移：

| 索引 | 通道 | 物理意义 | 预留用途 |
|------|------|----------|----------|
| 5 | L_m | 光程长度（m） | 未来若 NDIR 支持多光程几何，可作为 NDIR 分支输入 |
| 6 | piston_position_m | 活塞位置（m） | 声学路径几何随时间变化时供 USNet 使用 |
| 7 | ultrasonic_tof_observed_s | 飞行时间观测值（s） | 与 `ultrasonic_sound_speed_estimated` 互为冗余，ErrorNet 可用于交叉一致性检查 |
| 9 | ultrasonic_peak_index | 峰值采样点下标（int） | 用于诊断超声波形质量，ErrorNet 可作为离散特征 |
| 10 | ultrasonic_tof_quality | TOF 估计质量分数（[0,1]） | **ErrorNet 直读判定超声分支可靠性**（重点扩展） |
| 11 | ultrasonic_tof_accepted | TOF 是否被接受（0/1） | **ErrorNet 直读做超声分支门控**（重点扩展） |

**与 §6.5 归一化策略的对应关系**：

- 索引 5、6、7 是连续物理量，参与 train-only Z-score 拟合；当前虽未被模型 forward 读取，但 scaler 已经为它们准备好统计量，未来激活时无需重新拟合 scaler。
- 索引 9、10、11 是离散/有界量，§6.5 已明确跳过归一化，保留原始物理语义；ErrorNet 未来读取时直接拿到 `peak_index ∈ [0, T)`、`quality ∈ [0, 1]`、`accepted ∈ {0, 1}`，可解释性完整。

**索引保留承诺**：v1.x 不再调整这 12 个通道的索引位置；后续若需要新增通道，统一从索引 12 往后追加。

### 8.3 单模态分支输入定义

| 模态 | 输入来源 | 输入维度 | 说明 |
|------|----------|----------|------|
| **NDIR** | `[V_NDIR_CO2, T_C, P_MPa, H_RH]` | 4 | 取窗口最后时刻的慢通道 |
| **TCD** | `[V_TCS, T_C, P_MPa, H_RH]` | 4 | 同上 |
| **US** | `[ultrasonic_sound_speed_estimated_m_per_s, T_C, P_MPa, H_RH]` | 4 | 取超声声速估计值作为主信号 |

`extract_modal_input` 函数重写：

```
def extract_modal_input(x_last: torch.Tensor, modality: str) -> torch.Tensor:
    """
    x_last:   (B, 12) 窗口最后时刻
    modality: "ndir" | "tcd" | "usn"
    Returns:  (B, 4)
    """
    if modality == "ndir":
        sensor_val = x_last[:, IDX_NDIR_CO2 : IDX_NDIR_CO2 + 1]
    elif modality == "tcd":
        sensor_val = x_last[:, IDX_TCS : IDX_TCS + 1]
    elif modality == "usn":
        sensor_val = x_last[:, IDX_US_SPEED : IDX_US_SPEED + 1]
    env = x_last[:, ENV_INDICES]  # T_C, P_MPa, H_RH
    return torch.cat([sensor_val, env], dim=-1)
```

### 8.4 FeatureExtractor 重新定义

旧 FeatureExtractor 假设输入 `(B, L, 6)` 中前三通道为 `S_ndir / S_tc / S_us`。新 FeatureExtractor 需要适配：

- 传感器信号部分：`S = x[:, :, [IDX_NDIR_CO2, IDX_TCS, IDX_US_SPEED]]` → `(B, L, 3)`
- 环境部分：`env = x[:, :, ENV_INDICES]` → `(B, L, 3)`

13 维特征的定义不变（CV、D、G、Q、B、delta_T、delta_P、delta_RH、dev_max、dev_mean、snr_proxy、drift、dt），仅索引来源改变。

**关键改动**：`delta_T`、`delta_P`、`delta_RH` 的计算：

```
delta_T  = (env[:, -1, 0] - env[:, -2, 0]).abs()   # env[:, 0] = T_C
delta_P  = (env[:, -1, 1] - env[:, -2, 1]).abs()   # env[:, 1] = P_MPa
delta_RH = (env[:, -1, 2] - env[:, -2, 2]).abs()   # env[:, 2] = H_RH
```

### 8.5 W_base 新值建议

**不变**，与旧 RCDW 完全一致：

```
W_base = torch.tensor([
    [0.05, 0.70, 0.05],   # NDIR: O2 弱, CO2 强, N2 弱
    [0.50, 0.15, 0.45],   # TCD:  O2 中, CO2 弱, N2 中
    [0.45, 0.15, 0.50],   # US:   O2 中, CO2 弱, N2 中
])
```

物理语义验证：
- NDIR 对 CO2 权重最高（0.70），因为 NDIR 是 CO2 唯一直接测量模态。
- O2 与 N2 无 NDIR 直接测量，权重分配在 TCD 与 US 之间（各约 0.45-0.50）。
- TCD 与 US 对 CO2 权重较低（各 0.15），因为 CO2 已由 NDIR 主导。

### 8.6 Stage A 与 Stage B 流程

**流程不变**：
1. Stage A：依次训练 NDIRNet、TCDNet、USNet（`train_single_modal`），保存 `runs/stage_a/{ndir,tcd,usn}.pt`。
2. Stage B：加载 Stage A ckpt，冻结单模态网络，训练 ErrorNet + RCDWFusion，保存 `runs/stage_b/rcdw.pt`。

**ckpt 命名**：依然使用 `usn`（不是 `us`）以保持与现有模型属性名一致（`model.usn`）。

**历史 ckpt 废弃声明**：

> **数据契约变更后必须重训**。新的 `slow.npy` 通道布局（7 维 → 12 维拼接）、新的标签语义（O2/CO2/N2 来自 HITRAN 物理建模而非 toy Dirichlet）、新的 scaler 统计量——所有旧 `runs/stage_a/*.pt` 和 `runs/stage_b/rcdw.pt` 与新数据不兼容，必须删除并重训。

---

## 9. 扰动评测适配

### 9.1 inject.py 通道索引重映射

旧 `inject.py` 硬编码通道索引 `[0, 1, 2, 4]` 对应 `[S_ndir, S_tc, S_us, T]`。新布局下重映射：

| 扰动类型 | 旧目标索引 | 新目标索引 | 说明 |
|----------|-----------|-----------|------|
| `optical_atten` | `x[..., 0]` (S_ndir) | `x[..., IDX_NDIR_CO2]` (V_NDIR_CO2) | NDIR 通道衰减 |
| `optical_scat` | `x[..., 0]` (S_ndir) | `x[..., IDX_NDIR_CO2]` | NDIR 加性噪声 |
| `thermal` | `x[..., 1]` (S_tc) | `x[..., IDX_TCS]` | TCS 通道扰动 |
| `ultrasonic` | `x[..., 2]` (S_us) | `x[..., IDX_US_SPEED]` | 超声声速估计值加噪 |
| `temperature` | `x[..., 4]` (T) | `x[..., IDX_T_C]` | 温度通道偏移 |

### 9.2 是否新增扰动类型

**推荐新增**（可选，Phase 5 实施）：
- `h2o_cross`：H2O 交叉敏感性扫描——改变 `H_RH` 通道值，观察 CO2 NDIR 通道是否受影响，以及 ErrorNet 是否检测到干扰。
- `pressure_drift`：压力漂移扫描——改变 `P_MPa` 通道值，观察声速估计是否受影响。

**优先保留现有 5 类**，新增扰动作为后续扩展。

### 9.3 perturb.py 输出

保持不变：5 类 × 2 张图（指标曲线 + CO2 权重曲线），共 10 张 PNG。

---

## 10. 配置文件改动

### 10.1 configs/default.yaml 新字段

```yaml
data:
  dataset_root: "data/rcdw-formal"        # [新增] benchmark 数据根目录
  window: 8
  seed: 42
  train_modalities: ["slow", "ultrasonic"] # [新增] 明确不进 fiber_mic

generation:                               # [新增] 生成参数节
  sequence_count: 2000
  seed: 42
  timesteps: 128
  dt_s: 0.5
  storage: "memmap"
  multi_path_phase: "steady"
  stage_profile: "standard_exposure"
  stage_jitter: 0.0
  sampling_strategy: "lhs"
  path_lms: [0.20, 0.25, 0.30, 0.35, 0.40]
  optical_absorption_backend: "hitran_hapi_v1"
  hitran_cache_root: "data/hitran_cache"
  workers: 1
  chunk_size: null
  keep_chunks: false

spectral:                                 # [新增] 光谱配置
  config_path: "configs/spectral-defaults.json"

acoustic:                                 # [新增] 声学参数（可覆盖默认值）
  center_frequency_hz: 40000.0
  burst_cycles: 8
  sample_rate_hz: 200000

phases:                                   # [新增] phase 配置
  default_schedule: "standard_exposure"

splits:                                   # [新增] 切分配置
  ratios: { train: 0.70, val: 0.15, test: 0.15 }
  group_field: "mixture_id"

scalers:                                  # [新增] 标准化配置
  transform_target: "slow"
  modal_groups_from_schema: true          # 直接读取 SLOW_MODAL_GROUPS

# ---- 以下为既有字段（保留） ----
model:
  single_modal:
    hidden: 32
  error_net:
    hidden: 32
  fusion:
    beta: 8.0
    alpha_min: 0.1
    alpha_max: 0.9
    tau_a: 0.05
    s_min: 0.05
    s_max: 0.40
    tau_s: 0.05
  W_base:
    - [0.05, 0.70, 0.05]
    - [0.50, 0.15, 0.45]
    - [0.45, 0.15, 0.50]

training:
  stage_a:
    epochs: 200
    batch_size: 16
    lr: 1.0e-3
    weight_decay: 1.0e-4
    patience: 30
    ndir_loss_weights: [0.1, 1.0, 0.1]
  stage_b:
    epochs: 200
    batch_size: 16
    lr: 1.0e-3
    weight_decay: 1.0e-4
    patience: 30
    lambda_error: 1.0
    lambda_sum: 0.1
    freeze_single_modal: true

perturbation:
  kinds: [optical_atten, optical_scat, thermal, ultrasonic, temperature]
  levels: [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.11]

degradation:
  ratio: 4.0
  cap: 0.04

eval:
  basis: dry
```

### 10.2 configs/smoke.yaml 对应缩小

```yaml
data:
  dataset_root: "data/rcdw-smoke"
  window: 8
  seed: 42
  train_modalities: ["slow", "ultrasonic"]

generation:
  sequence_count: 64            # 比 default 小 30 倍
  seed: 42
  timesteps: 32                 # v1.2 仅用 STANDARD_EXPOSURE（4 段），32 充裕；未来激活 MULTI_PULSE 需 >= 12
  dt_s: 0.5
  storage: "memmap"
  multi_path_phase: "steady"
  stage_profile: "standard_exposure"
  stage_jitter: 0.0
  sampling_strategy: "lhs"
  path_lms: [0.20, 0.30, 0.40]
  optical_absorption_backend: "hitran_hapi_v1"
  hitran_cache_root: "data/hitran_cache"
  workers: 1

spectral:
  config_path: "configs/spectral-defaults.json"

splits:
  ratios: { train: 0.70, val: 0.15, test: 0.15 }
  group_field: "mixture_id"

scalers:
  transform_target: "slow"
  modal_groups_from_schema: true

# 训练 / 模型 / 扰动节继承 default 的结构，仅修改 epoch
training:
  stage_a:
    epochs: 20
    patience: 10
  stage_b:
    epochs: 20
    patience: 10
```

### 10.3 与既有字段的兼容性

- `model.fusion.*`、`training.stage_a.*`、`training.stage_b.*`、`perturbation.*`、`degradation.*`、`eval.*`：完全保留，无破坏性变更。
- 新增 `generation.*`、`spectral.*`、`splits.*`、`scalers.*`：旧脚本未读取这些键，新代码必须主动读取；smoke 配置必须同步扩增，否则 `generate_benchmark.py` 无法运行。
- 删除字段：旧 `data.n_train / n_val / n_test`（合成数据规模）已无意义，新流程从 manifest 的 split 文件读取样本数，应从 `data` 节移除以防误用。

---

## 11. 测试覆盖

### 11.1 新增测试文件清单

| 文件 | 覆盖范围 |
|------|----------|
| `tests/test_conditions.py` | LHS 采样可复现性、组分和约束、N2 ≥ 55% 边界 |
| `tests/test_phases.py` | `STANDARD_EXPOSURE` 的 boundaries 与 blend 形状；`jittered` 随机性；duration_frac 之和 = 1；`resolve_phase_schedule("variable_onset")` 等未实现 profile 抛 `NotImplementedError`（v1.2 YAGNI 契约）
| `tests/test_slow.py` | `_blend_composition` 三组分插值正确、动力学 RC 步进单调性、噪声方差范围 |
| `tests/test_waveforms.py` | 超声与光纤麦克风波形 shape、量化精度、TOF 估计误差范围、`x_h2 / x_ch4` 等遗留字段不被接受 |
| `tests/test_optical_backend.py` | HITRAN 缓存读写、CO2 与 H2O 浓度对吸光度的单调性、`get_default_ndir_filter("ch4")` 抛 KeyError（仅保留 co2） |
| `tests/test_packaging.py` | 落盘目录结构完整、`fiber_mic_*` 存在但未进 scaler、`labels/y.npy` shape == (N, 3) |
| `tests/test_validation.py` | 9 类不变量分别触发与通过、LEGACY 字段被拒绝 |
| `tests/test_dataset_loader.py` | `BenchmarkDataset` 仅读取 slow + ultrasonic 元数据；fiber_mic 文件存在但不被 `__getitem__` 触碰；窗口 shape (L, 12) |
| `tests/test_w_base_alignment.py` | 新通道布局下 `extract_modal_input` 返回的 sensor 索引与 `W_base` 行顺序一致（NDIR/TCD/USN ↔ V_NDIR_CO2/V_TCS/US_SPEED） |

### 11.2 关键不变量（每个测试至少覆盖一个）

- **schema 不变量**：`SLOW_CHANNELS` 长度 = 7；`COMPONENT_FIELDS` 顺序 = (O2, CO2, N2)；`SPLIT_NAMES` 不含 extrapolation；`PhaseSchedule` 注册表仅含 `STANDARD_EXPOSURE` 一项（v1.2 YAGNI）。
- **生成不变量**：组分和 = 100 ± 1e-5；mixture_id 与 sequence_id 全局唯一；LEGACY 字段被 validation 拒收。
- **packaging 不变量**：manifest `schema_version == "rcdw-benchmark-1"`；`train_modalities == ["slow", "ultrasonic"]`；fiber_mic 出现在 shapes 但不出现在 scaler 输出；scaler 输出中 `ultrasonic_peak_index` / `ultrasonic_tof_quality` / `ultrasonic_tof_accepted` 显式标记 `strategy=passthrough`（v1.2 归一化策略）。
- **训练侧不变量**：DataLoader 返回 (B, 8, 12)；`extract_modal_input` 返回 (B, 4)；W_base shape (3, 3) 且每列 sum = 1.0。

### 11.3 现有测试如何迁移或废弃

| 旧测试文件 | 处理 |
|------------|------|
| `tests/test_synth.py` | **废弃**，由 `tests/test_dataset_loader.py` 替代；旧文件归档到 `tests/_legacy/`，不在 pytest collection 中执行 |
| `tests/test_single_modal.py` | **保留**，仅修改 `test_extract_modal_input` 中的索引断言（从旧 6 维改为新 12 维布局） |
| `tests/test_feature.py` | **修改**，CV/D/G/Q/B 等特征值的输入构造改用新通道布局 |
| `tests/test_error_net.py` | **保留**，与通道布局无关 |
| `tests/test_rcdw_fusion.py` | **保留** |
| `tests/test_degradation.py` | **保留** |
| `tests/test_perturbation.py` | **修改**，断言扰动目标列改为 `IDX_NDIR_CO2 / IDX_TCS / IDX_US_SPEED / IDX_T_C` |

---

## 12. 落地步骤（分 Phase）

### Phase 1：core + conditions + phases（schema 骨架）

| 项 | 内容 |
|---|------|
| 目标 | 建立 `rcdw/sim/core/` 与 `rcdw/sim/generation/conditions.py + phases.py + gas_state.py`，可独立生成 condition_grid_sequence.csv |
| 涉及文件 | `rcdw/sim/core/schema.py`、`rcdw/sim/core/ids.py`、`rcdw/sim/generation/conditions.py`、`rcdw/sim/generation/phases.py`、`rcdw/sim/generation/gas_state.py`；`tests/test_conditions.py`、`tests/test_phases.py` |
| 产物 | 一个调用 `generate_condition_rows(64, seed=42)` 的 demo 脚本可在 console 打印前 5 行 condition |
| 验收命令 | `cd rcdw_mgda && python -m pytest tests/test_conditions.py tests/test_phases.py -q` |
| 风险 | LHS 采样的 N2 缩减回退逻辑与 simplex 边界的相互作用，需要 monkey-patch 边界 case |

### Phase 2：声学 + 光学 + waveforms（物理栈）

| 项 | 内容 |
|---|------|
| 目标 | 完成 `acoustic_physics.py`、`waveforms.py`、`optical_backend.py`、`optical_crosstalk.py`、`spectral/*` |
| 涉及文件 | `rcdw/sim/generation/acoustic_physics.py`、`rcdw/sim/generation/waveforms.py`、`rcdw/sim/generation/optical_backend.py`、`rcdw/sim/generation/optical_crosstalk.py`、`rcdw/sim/generation/spectral/*`、`configs/spectral-defaults.json`；`tests/test_waveforms.py`、`tests/test_optical_backend.py` |
| 产物 | `rcdw/sim/generation/optical_backend.compute_hitran_optical_absorption({"CO2": 5.0, "H2O": 1.0}, ...)` 返回合法吸光度；HITRAN 缓存目录 `rcdw_mgda/data/hitran_cache/` 自动创建 |
| 验收命令 | `pytest tests/test_optical_backend.py tests/test_waveforms.py -q` |
| 风险 | HAPI 首次运行需联网下载光谱（CO2 与 H2O 各几 MB）；O2 弛豫参数（`alpha_lambda_max_o2`、`f_relax_o2_per_atm`）的 TBD 数值需在 RCDW 文档单独标注，并以后续调参回退 |

### Phase 3：slow + benchmark + packaging + validation（端到端落盘）

| 项 | 内容 |
|---|------|
| 目标 | 端到端生成一个 smoke benchmark：`data/rcdw-smoke/`，含 manifest、sequence_index、labels、splits、scalers、validation_summary |
| 涉及文件 | `rcdw/sim/generation/slow.py`、`rcdw/sim/generation/benchmark.py`、`rcdw/sim/packaging/*`、`rcdw/sim/validation/integrity.py`、`scripts/generate_benchmark.py`；`tests/test_slow.py`、`tests/test_packaging.py`、`tests/test_validation.py` |
| 产物 | `cd rcdw_mgda && python -m scripts.generate_benchmark --config configs/smoke.yaml` 成功落盘 64 条 sequence；manifest.json 内容符合 6.2 节字段；fiber_mic 文件存在但 scaler JSON 中没有 fiber_mic 的统计量 |
| 验收命令 | `pytest tests/test_slow.py tests/test_packaging.py tests/test_validation.py -q && ls data/rcdw-smoke/sequences/` |
| 风险 | memmap 写盘在 Windows 路径含中文时的兼容性；并行 chunk 合并时 mixture_id 顺序需稳定 |

### Phase 4：Dataset + 模型通道适配 + Stage A/B

| 项 | 内容 |
|---|------|
| 目标 | DataLoader 从 `data/rcdw-smoke/` 加载窗口张量，Stage A 三模态训练、Stage B 联合训练全部通过；老的 `synth.py` 归档 |
| 涉及文件 | `rcdw/data/dataset.py`、`rcdw/models/single_modal.py`（通道索引）、`rcdw/models/feature.py`、`rcdw/models/rcdw.py`（docstring 更新：`(B, L=8, 6)` → `(B, L=8, 12)`）、`rcdw/training/stage_a.py`、`rcdw/training/stage_b.py`、`scripts/train.py`（数据加载从 `synth.make_splits` 切换到 `BenchmarkDataset`）、`scripts/eval.py`；`configs/default.yaml`、`configs/smoke.yaml`（删除 `data.n_train/n_val/n_test`，新增 `data.dataset_root/train_modalities`）；`tests/test_dataset_loader.py`、`tests/test_w_base_alignment.py`；`tests/test_single_modal.py`、`tests/test_feature.py`、`tests/test_perturbation.py` 适配 |
| 产物 | 在 smoke benchmark 上 Stage A 三模态早停收敛、Stage B 早停收敛；`runs/stage_a/{ndir,tcd,usn}.pt` 与 `runs/stage_b/rcdw.pt` 重新生成 |
| 验收命令 | `pytest -q && python -m scripts.train --config configs/smoke.yaml && python -m scripts.eval --ckpt runs/stage_b/rcdw.pt --split test` |
| 风险 | 通道维从 6 到 12 改动会触发若干现有测试断言失败，必须同步适配；Stage A 历史 ckpt 必须删除 |

### Phase 5：扰动评测 + 配置 + 文档收口

| 项 | 内容 |
|---|------|
| 目标 | `scripts/perturb.py` 在新通道布局下生成 5 类 × 2 张图；configs/default.yaml 与 smoke.yaml 收口；文档与 recallloom 更新 |
| 涉及文件 | `rcdw/perturbation/inject.py`、`scripts/perturb.py`、`configs/default.yaml`、`configs/smoke.yaml`；`docs/学长算法/RCDW_*.md` 同步更新 |
| 产物 | `runs/perturb/{kind}_metrics.png` 与 `{kind}_weights_CO2.png` 共 10 张图；新增 `h2o_cross` 与 `pressure_drift`（可选）；本方案补「实施完成」附录或单独 `RCDW_数据集主线对齐_完成情况.md` |
| 验收命令 | `python -m scripts.perturb --ckpt runs/stage_b/rcdw.pt --config configs/smoke.yaml && ls runs/perturb/` |
| 风险 | inject 通道索引漂移后，旧扰动曲线不可直接横向比较，需要在 docs 中明确「v1 vs v2 扰动语义不同」 |

---

## 13. 已知风险与开放问题

### 13.1 HITRAN 缓存大小与首次运行耗时

- 首次运行需要联网调用 `hapi.fetch`，CO2 与 H2O 谱线表合计约 8-15 MB。
- 缓存写盘后单次冷启动约 30-60 秒，热缓存下小于 5 秒。
- **建议**：在 `scripts/generate_benchmark.py` 入口加 `--precompute-cache-only` 选项，与正式 benchmark 生成解耦。

### 13.2 O2 NIR A-band 是否引入

- 本方案选择**不引入**。引入 O2 通道会导致 NDIR 通道数大于 1、W_base 形状不再是 (3, 3)、单模态网络数量增加等连锁修改。
- **开放问题**：是否在 v2 方案中作为独立 NDIR-NIR 模态引入？届时需要重新设计 W_base 为 (4, 3) 或 (3, 3) + auxiliary head。
- **暂定策略**：v1 保持 3×3，将 O2 仅由 TCD 与 US 间接重建。

### 13.3 O2 与 N2 弛豫吸收经验值不确定

- HG 中 N2 弛豫参数（`f_relax_n2_per_atm = 65000.0`）有依据；O2 弛豫频率在文献中分布于 30-60 kHz，需在物理参数文档中给出**引用来源**或**实测校核流程**。
- **风险**：错误的弛豫参数会使 `α_attenuation` 在 40 kHz 工作点产生 10%-30% 的系统偏差，影响 TOF 与 amplitude 模型；但因 RCDW 关注算法层面权重，影响通过训练吸收，不会破坏整体管线。
- **缓解**：将 `alpha_lambda_max_o2`、`f_relax_o2_per_atm` 标 `# TBD: 实测校核`，并在 `acoustic_model_metadata` 写入 `"o2_relaxation_source": "placeholder_v1"`，便于后续替换时审计。

### 13.4 三组分 sum = 1 + Dirichlet vs LHS 选择对训练分布的影响

- LHS 在 simplex 上的均匀性更好，但 N2 ≥ 55% 的硬约束会让一部分 LHS 样本被「拉回边界」，造成边界点密度异常。
- Dirichlet 可以直接控制分布偏斜，但与 HG 主线不一致。
- **缓解**：在 `tests/test_conditions.py` 中记录边界回退率（被缩减的样本占比），并在 manifest 中写入 `"lhs_rescaled_count"`，便于审计。
- **后续可选**：若边界回退率过高（如超过 10%），可改用 Dirichlet(α_O2=2, α_CO2=1, α_N2=6)，与旧 toy 分布一致。

### 13.5 训练侧通道维变更

- 通道维从 6 到 12，模型结构 `SingleModal(in_dim=4)` 保持不变（取 sensor + 3 维环境），但 FeatureExtractor 和 inject 必须同步重写索引。
- 任何对 `(B, L, 6)` 的硬编码引用都必须 grep 排查，常见隐患在测试断言和注释中。

---

## 14. 文档与产物归档

### 14.1 本方案落点

- 本文件落在 `docs/学长算法/RCDW_数据集主线对齐改动方案.md`，与下列文档处于同一层级：
  - `docs/学长算法/学长算法框架.md`（不变）
  - `docs/学长算法/RCDW_独立复现方案.md`（不变；本方案与其互补）
  - `docs/学长算法/RCDW_实施指南.md`（不变；本方案落地后需在该文档追加「数据集已替换为 benchmark 形态」说明）
  - `docs/学长算法/RCDW_实施完成情况.md`（不变；Phase 落地后追加完成情况）

### 14.2 实施完成后需要更新的 docs

| 文档 | 更新内容 |
|------|----------|
| `docs/学长算法/RCDW_实施指南.md` | 在「数据生成」章节末尾追加：「rcdw_mgda 已于 v1.x 切换为 benchmark 形态，详见 `RCDW_数据集主线对齐改动方案.md`」 |
| `docs/学长算法/RCDW_实施完成情况.md` | 追加「数据集主线对齐」章节，列出 Phase 1-5 完成进度，并标注历史 ckpt 已废弃 |
| `docs/学长算法/RCDW_独立复现方案.md` | 在「12. 扰动实验」之后追加附录：「通道布局从 6 维升级到 12 维后，扰动通道索引重映射表」 |
| `AGENTS.md` | 第二段「禁止 mixture_id 回退」条目下追加：「`rcdw_mgda` 子工程同样遵循该不变量；其独立 schema_version 为 `rcdw-benchmark-1`」 |
| `docs/学长算法/RCDW_数据集主线对齐_完成情况.md` | **新建**（Phase 5 收口时）：记录 Phase 1-5 实际产出、与本方案的偏差、TBD 参数的最终取值 |

### 14.3 recallloom 写入条目（仅指明需要更新，不写入实际内容）

| section | 触发时机 |
|---------|----------|
| `rolling_summary` | Phase 1 完成时追加「rcdw_mgda 启动数据集对齐」；Phase 5 完成时追加「rcdw_mgda 完成数据集对齐」 |
| `daily_logs` | 每个 Phase 完成当日记录实际产出与遇到的问题 |
| `context_brief` | 当 RCDW 在主线讨论中重新出现时，在 brief 中更新「rcdw_mgda 已切换 benchmark」 |
| `update_protocol` | 无需变更，沿用现有协议 |

写入时严格遵循 AGENTS.md 中 RecallLoom 写入规约：`PYTHONUTF8=1` 前缀、中文与斜杠字母数字之间加空格或顿号。
