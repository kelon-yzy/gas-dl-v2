# tv3 组分区间拓宽实施规划（CO₂ 0.03–10%，O₂ 15–25%）

> 状态：**F5-S 代码已落地（2026-07-22）**：`bidir_spxy_observed_ab_v1`（50 维 AB-only）+ recompute 适配 + 协议 12 格判据 (d)。`stage_status.f5*_wide=f5s_code_ready_awaiting_formal_matrix`。**下一步：smoke-wide 端到端审计 → 再服务器 6000-wide**；incomplete 仍 exit 2。
> 
> 责任边界：本规划**不**覆盖任何已冻结产物（v1 单向审计、F4 窄域 verdict、F3 窄域保真、B1/B7/E1d-SB、S 线、tv3-formal-6000），**不**改写 F 线物理/流速契约，**不**声称推翻 `coarse_monitoring_only`。新域按独立注册域处理。

---

## 0. 结论先行

| 问题                                | 判定                                                                                                                |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 物理能否支持新区间？                        | **能**。声速/衰减/热导均为连续混合公式，无区间硬编码；CO₂=10% NDIR 吸光度≈0.45（饱和阈 4.0，余量充足），CO₂=10%+L=0.3m 声学幅度保留≈55%（量级估算，须 F3 重验）。        |
| 新区间是否有域动机？                        | **有**。O₂[15,25] 覆盖 OSHA 缺氧线（<19.5%）到富氧火险线（>23.5%）；CO₂[0.03,10] 覆盖新鲜空气到超 IDLH（4%）严重积聚。比窄域 O₂[18,21.2] 更贴近真实危害监测包络。 |
| 是否推翻 F4 `coarse_monitoring_only`？ | **不推翻**。窄窗 P90 门由温度(1K)/jitter 的**局部** vol% 误差主导，与区间宽度无关。拓宽只放大 R²/相对指标，不移动绝对精度墙。                                  |
| 推荐策略                              | **F 线独立新注册域**：不动共享默认值，新增 `WIDE_COMPOSITION_RANGES`，新 benchmark 名 + 新 F0 registry + 新 F4/F5 输出目录；旧窄域全部冻结留档。        |
| 主要风险                              | (a) CO₂=10%/L=0.3m 最差 SNR 下 F3 峰定位保真门；(b) F5 判据 (c) 跨域失效，须在域内重锚。                                                  |

**一句话**：这是一次"注册仿真域拓宽"，不是"两个数字替换"。按新域独立走 F0'→F5，旧域产物不动，可比性只在新域内部成立。

---

## Decision Gate 0 —— 已确认（2026-07-22）

> **用户确认：A1 仅 F 线 + B1 并行留档。** 改动面据此锁定，进入 F0'-wide 落代码。

改动路径在这两项上分叉，**必须先定，否则会误伤单向链路或破坏 write-once 纪律**。

### 决策 A：作用范围

| 选项               | 含义                                                       | 代价                                                        | 推荐  |
| ---------------- | -------------------------------------------------------- | --------------------------------------------------------- | --- |
| **A1 仅 F 线（推荐）** | 新增独立 `WIDE_COMPOSITION_RANGES`，只走双向生成路径；单向 S/E/B 线保持窄域冻结 | 需给 benchmark spec 加 `composition_ranges` 字段并线穿到 bidir 生成器 | ✅   |
| A2 项目级替换         | 直接改 `TUNNEL_VENTILATION_RANGES` 默认值，全场景新域                | 单向 S/E/B/tv3-formal-6000 任何再生都会静默移域；旧结论全部悬空               | ❌   |

推荐 A1：`conditions.py:51` 的 `TUNNEL_VENTILATION_RANGES` 同时被 `generate_tunnel_ventilation_condition_rows`（单向）和 `generate_tunnel_ventilation_bidir_condition_rows`（双向）默认引用。改默认值 = 改两条线。F 线拓宽不应牵动单向历史链路。

### 决策 B：旧窄域产物去向

| 选项              | 含义                                        | 推荐             |
| --------------- | ----------------------------------------- | -------------- |
| **B1 并行留档（推荐）** | 旧窄域 v1/F4/F3/B1 全部冻结不动，新域用 `-wide` 后缀命名并列 | ✅              |
| B2 归档替换         | 旧窄域标记 superseded，新域成为唯一操作域                | 仅当明确放弃窄域研究价值时选 |

推荐 B1：F 线计划 §Format.1 明文"每阶段产物一次写入不可覆盖"。窄域 F4 已产出 `coarse_monitoring_only` 冻结 verdict（2026-07-21），是正式历史结论。新域作 v2-wide 并列报告。

> **已确认（2026-07-22）**：用户选定 **A1 仅 F 线 + B1 并行留档**。下文步骤按 A1+B1 执行；A2/B2 分支作废。

---

## 1. 新区间定义与物理动机

### 1.1 目标区间

| 变量      | 旧（窄域）        | 新（危害监测域）         | N₂ 残差推导                                   |
| ------- | ------------ | ---------------- | ----------------------------------------- |
| `x_CO2` | 0.03–5.00%   | **0.03–10.00%**  | —                                         |
| `x_O2`  | 18.00–21.20% | **15.00–25.00%** | —                                         |
| `x_N2`  | 73.80–81.97% | **65.00–84.97%** | min=100−10−25=65.00；max=100−0.03−15=84.97 |

可行性：`max(x_CO2+x_O2)=10+25=35 < 100`，2D LHS 独立采样任意角点 N₂≥65>0，**无需联合约束修正**（与旧域同构，闭包自动满足）。

### 1.2 域动机（登记为 `literature_bound` / `engineering_scenario`）

- **O₂[15,25]**：OSHA 1910.146——O₂-deficient <19.5%、O₂-enriched >23.5%。15% 为显著缺氧下界、25% 为富氧火险上界，单域同时覆盖两类危害。`literature_bound`。
- **CO₂[0.03,10]**：新鲜空气 0.04% 到严重积聚。CO₂ IDLH=4%，10% 覆盖超 IDLH 昏迷级危害包络。`engineering_scenario`（受限空间严重积聚，非单一规程阈值）。

### 1.3 窄窗重铺（F4 用）

旧域 4 窗（中心 18.4/19.2/20.0/20.8，宽 0.8）铺满 O₂[18,21.2]。新域 O₂[15,25] 宽 10 vol%，推荐**危害锚定**窗（保持宽 0.8）：

```
15.5（严重缺氧）, 17.5, 19.5（OSHA 缺氧线）, 20.9（常压空气）, 23.0, 24.5（富氧火险）
```

理由：均匀铺满需 ~12 窗，冗余；危害锚定 6 窗覆盖决策相关 O₂ 水平，且保留 20.9 与旧域 20.0 窗近邻可作跨域松对照。

---

## 2. 物理可行性判定（须 F1/F3 正式重验，本节为立项量级）

| 检查项         | 旧域                     | 新域最差点                                         | 量级估算（未跑仿真）                                    | 结论                                 |
| ----------- | ---------------------- | --------------------------------------------- | --------------------------------------------- | ---------------------------------- |
| NDIR CO₂ 饱和 | 5% → 吸光度 0.225         | CO₂=10% → 0.45                                | 饱和阈 4.0，V_NDIR_CO2≈2.5·e^−0.45≈1.6V（>0.1 底噪）  | ✅ 动态范围反而更大                         |
| 声学衰减/SNR    | 5%,L=0.3 → 幅度≈0.75     | CO₂=10%,L=0.3 → α_co2≈1.9 Np/m，幅度≈e^−0.6≈0.55 | 峰定位方差 ∝1/SNR²，最差帧或从 P95 0.087 升至 ~0.16 sample | ⚠️ **须 F3 重验**（门=0.25 sample，预期仍过） |
| 声速跨度        | M_mix ~29.0–29.6       | M_mix ~29.0–30.2                              | TOF~735μs ≪5ms 窗，无溢出                          | ✅                                  |
| O₂ 信号跨度     | 3.2 vol% → δc≈0.76 m/s | 10 vol% → δc≈2.4 m/s，ΔTOF≈5.1μs@0.25m         | vs nominal jitter 0.5μs → SNR ~10×            | ✅ 信号跨度 ~3×                         |
| O₂ V-T 弛豫   | fr,O≈24Hz≪200kHz       | 同                                             | alpha_o2≈0 不变                                 | ✅ 无关区间                             |

**唯一实质物理风险**：CO₂=10%+L=0.3m 低 SNR 帧对 F3 峰定位保真门的冲击。非阻断（预期仍过），但**必须**在新域 F3 集上实测，不得沿用窄域 F3 数值。

---

## 3. 完整改动面（A1+B1 路径）

### 3.1 代码（生成新数据必须改）

| 文件                                                            | 动作                                                                                                                                                    | 约束                               |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `tv3/sim/generation/tunnel_ventilation/conditions.py:40-51`   | 新增 `WIDE_COMPOSITION_RANGES = TunnelVentilationRanges(co2=(0.03,10.0), o2=(15.0,25.0), n2_min=65.0, n2_max=84.97)`；**不改** `TUNNEL_VENTILATION_RANGES` | 更新 dataclass 上方 docstring：标注两域并存 |
| `tv3/sim/generation/tunnel_ventilation/conditions.py:103-130` | `ranges` 仍默认 narrow；wide **必须**由 spec 显式传入                                                                                                            | 单向默认不动；避免模块级默认静默移域               |
| `tv3/sim/generation/tunnel_ventilation/benchmark.py:118-146`  | `TunnelVentilationBenchmarkGenerationSpec` 加 `composition_domain: str = "narrow"` 字段                                                                  | frozen dataclass，加字段兼容旧调用        |
| `tv3/sim/generation/tunnel_ventilation/benchmark.py:168-173`  | bidir 分支按 `spec.composition_domain` 选 ranges 传入生成器                                                                                                    | 单向分支不变                           |
| `tv3/pipeline/generate_tunnel_ventilation_benchmark.py`       | 新增 `--composition-domain {narrow,wide}`；wide 时 dataset 自动加 `-wide` 后缀；旧 bidir preset 保持 narrow                                                        | 默认 narrow，保持旧命令行为                |

> 关键：`benchmark.py` 现调用 `generate_tunnel_ventilation_bidir_condition_rows` 未传 ranges（走模块默认）。A1 路径下 spec 显式携带域选择并线穿，避免依赖模块级全局状态——符合可复现纪律。

### 3.2 F0 registry 修订（登记新注册域）

现 `parameter_registry.json` 只有 `composition_anchor`（字段名），**未登记组分区间**。`claim_scope="registered_simulation_domain_only"` 要求组分域也可追溯。

| 文件                                                                     | 动作                                                                                                                                                                              |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configs/tv3_bidir/parameter_registry_wide.json`（新建，**不覆盖** dc61d9e7…） | 复制 F0 registry，`composition_anchor` 下新增 `composition_ranges` 块（co2/o2/n2 上下界 + 上表 §1.2 source_tag + ref）；`sim_revision_tag` 追加 `composition_domain: "wide_hazard_v1"`；重算 sha256 |

### 3.3 F4 审计配置（新域审计，写新目录）

| 文件                                                   | 动作                                                                                                                                                                                                                                                                                                                                       |
| ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configs/tv3_bidir_identifiability_v2_wide.json`（新建） | 基于旧 config 改：`parameter_bounds.co2=[0.03,10]`、`o2=[15,25]`；`global_grid.co2=[0.03,5.0,10.0]`、`o2=[15,20,25]`；`narrow_windows` 换 §1.3 六窗；`narrow_context_grid.co2` 同步；`output_dir` → `outputs/tv3_bidir/identifiability_v2_wide`；`f0_registry.expected_sha256` → 新 hash；`prior_crosscheck.reference_point` 保持 co2=1/o2=20（仍在域内，期望值为局部量不变） |
| `tv3/audit/identifiability_v2.py`                    | **无需改**。审计是网格驱动，`BidirAcousticPoint.validate()` 只查非负闭包，不硬编码上下界                                                                                                                                                                                                                                                                           |

### 3.4 F5 模型协议（新域正式对照，写新目录）

| 文件                                               | 动作                                                                                                                                                                                                                     |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `configs/tv3_bidir_model_protocol_wide.json`（新建） | `source_dataset_dir` → `data/tv3-bidir-6000-wide`；`splits_root`/`output_dir` → `-wide`；**判据 (c) 重锚**：`s_line_b1_reference_o2_mae` 改用**域内 A1 的 v_path=0 锚点**，不引用旧窄域 B1（跨分布无意义）；`f4_prerequisite.verdict_path` → 新 F4 目录 |
| `tv3/ml/bidir_s_flow.py`                         | **无需改**。S-Flow 划分是 `v_path` 驱动（`mixture_median_abs_v_path`），与组分区间正交                                                                                                                                                    |

> **判据 (c) 重锚是本规划最关键的科学决定**：F5 五臂中 A1（仅 AB 单向特征）本身就是**新域内的单向基线**。把判据 (c) 锚到 A1 的零流子集，全部比较落在新域内部，避免"新域 A3 vs 旧域 B1"跨分布伪比较。旧窄域 B1 至多作松散历史交叉核对，且须显式标注 `cross_distribution`。

### 3.5 测试

| 文件                                                                | 动作                                                                      |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `tests/test_tunnel_ventilation_bidir_identifiability_v2.py:21-23` | bounds fixtures 加新域档或参数化；核心测的是导数数学，旧值仍过，为一致性更新                          |
| `tests/test_tunnel_ventilation_schema.py` / conditions 相关测试       | **先跑**，定位是否有 N₂∈[73.80,81.97] 硬断言；有则加新域档，不删旧域断言                         |
| 新增 `tests/test_tunnel_ventilation_wide_composition.py`            | 新域可行性小测：co2=10/o2=15/o2=25 角点声速有限、闭包成立、CO₂=10%,L=0.3 幅度>底噪、bidir 零流退化一致 |

### 3.6 文档（记录一致性；旧结论加新行不覆盖）

| 文件                                                                | 动作                                                                        |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `docs/foundation/sampling_design.md`                              | §1.1 表/§2.2 伪代码/§1.2 N₂ 校验/§四 覆盖标准：标注 narrow/wide 双域，新增 wide 行            |
| `docs/foundation/adaptation_plan.md:84`                           | 组分范围行加 wide 域说明                                                           |
| `docs/foundation/CO2_O2_N2_气体检测场景规划.md:42-44`                     | 同步双域                                                                      |
| `docs/references/tunnel_ventilation_sensing_survey.md:32`         | 采样区间描述加 wide 域                                                            |
| `docs/references/tv3_acoustic_simulation_fidelity_review.md:36`   | 采样范围行加 wide 域                                                             |
| `docs/methods/tv3_名词与实验顺序导读.md:152`                               | "全域"定义补 wide 域 O₂[15,25]                                                  |
| `docs/active/tv3_bidirectional_ultrasound_implementation_plan.md` | §Task.3 可比性声明加"wide 域为独立注册域"；§Context.2 锚点说明标注 O₂=20 仍在域内；实施记录加 F*-wide 行 |
| `docs/掘进通风项目记忆库.md:85,159,160`                                    | 结论表**新增 wide 域行**，旧窄域 O₂[18,21.2] 行保持不动                                   |

### 3.7 不触碰（冻结留档）

```
outputs/tv3_identifiability/              # v1 单向审计
outputs/tv3_bidir/identifiability_v2/     # F4 窄域 coarse_monitoring_only
outputs/tv3_bidir/dsp_fidelity/           # F3 窄域保真
outputs/tv3_bidir/model_protocol/         # F5 窄域（若已跑）
configs/tv3_bidir/parameter_registry.json # F0 窄域 dc61d9e7…
data/tv3-bidir-6000, tv3-formal-6000      # 窄域正式集
B1/B7/E1d-SB/S 线全部冻结
```

---

## 4. 科学影响分析（诚实边界）

### 4.1 拓宽**改善**什么

- **R²(O₂) 上升**：目标方差 Var(O₂) 增大约 (10/3.2)²≈10×，同等绝对误差下 R² 显著变好——但这是分母效应，非精度提升。
- **O₂ 信号跨度 ~3×**：δc 跨度 0.76→2.4 m/s，模型可利用的声速梯度更大，A3−A1 双向增益更易越过预注册 0.5 vol% 幅度门。
- **域覆盖更真实**：贴合缺氧/富氧/CO₂ 积聚的实际危害包络。

### 4.2 拓宽**不改变**什么（关键诚实点）

- **窄窗 P90 绝对精度墙不动**：P90 门问"给定 nuisance，0.8 vol% 窗内 O₂ 估计 P90 展宽"，是**局部**量。温度(1K)→2.4 vol%、jitter(3μs)→5.8 vol% 的映射由局部导数决定，与全域宽度无关。拓宽不使 `coarse_monitoring_only` 变成 `continuous_regression_supported`。
- **T–M 混叠不解**：`c=√(γRT/M)` 仍是一方程两未知，T 仍须外部测量。
- **要达 0.4 vol% 连续回归门**：仍需 ~0.1K 温度感测 + jitter 控制，与本次拓宽正交。

> 若目标是"让 O₂ 看起来可回归"——拓宽会让 R² 好看但精度墙没动，属指标错觉，须避免误读。若目标是"更真实的危害监测包络下量化双向增益"——拓宽动机充分，正当立项。

### 4.3 可比性断裂清单

| 断裂                   | 处理                    |
| -------------------- | --------------------- |
| v1/F4 窄域审计           | 新域作 v2-wide 并列，旧目录不改写 |
| F5 判据 (c) 引用旧 B1     | 域内重锚到 A1 零流子集（§3.4）   |
| tv3-formal-6000 单向基线 | 不作新域基线；A1 承担域内单向对照    |
| 记忆库窄域结论              | 加 wide 行，不覆盖          |

---

## 5. 分阶段执行步骤（A1+B1）

沿用 F 线阶段门结构，全部写 `-wide` 独立目录。**F3 帧保真门未过前禁止任何模型训练**（同 D2b 停止条件）。

### F0'-wide —— registry 修订与域登记

1. 新建 `configs/tv3_bidir/parameter_registry_wide.json`：加 `composition_ranges` 块（§1.2 source_tag）+ `composition_domain: wide_hazard_v1`，重算 sha256。
2. 更新 §3.6 foundation 文档为双域。
3. 门：每项组分界有 source_tag；sha256 落定并写入下游 config `expected_sha256`。

### F1-wide —— 物理与单元测试

```bash
python -m pytest -q tests/test_tunnel_ventilation_bidir_physics.py tests/test_tunnel_ventilation_physics.py
python -m pytest -q tests/test_tunnel_ventilation_wide_composition.py   # 新增
```

- 门：新域角点（co2=10/o2=15/o2=25）声速有限、闭包成立、bidir 零流退化逐点一致、reciprocal-sum 精确性保持（与组分无关，形式性通过）。

### F2-wide —— smoke benchmark

```bash
python scripts/run_tv3_bidir_f0_registry.py --wide
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark \
  --output-root data --preset bidir-smoke --composition-domain wide
python scripts/check_slow_channels.py data/tv3-bidir-smoke-wide
python scripts/run_tv3_bidir_f2_benchmark_audit.py --composition-domain wide
```

- 门：validation 全过；manifest 记录 wide 域 + `f0_registry_sha256`；int16+scale 存储自洽；采样覆盖新 O₂/CO₂ 全域。
- **已通过（2026-07-22）**：verdict=`f2_wide_smoke_passed`；产物 `data/tv3-bidir-smoke-wide`、`outputs/tv3_bidir/benchmark_audit_wide/`；`stage_status.f2_wide` 并列，窄域不动。

### F3-wide —— DSP 保真（**物理风险重验点**）

```bash
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark \
  --output-root data --preset bidir-f3 --composition-domain wide
python scripts/run_tv3_bidir_dsp_fidelity.py --config configs/tv3_bidir_dsp_fidelity_wide.json
```

- 门：**逐方向峰定位 P95 ≤0.25 sample（含 CO₂≥8% × L≥0.28 m 最差帧；path_lms 上限 0.28 作 L=0.3 代理）**；τ̂ 误差 ≤0.10μs；全 flow 网格 ĉ/v̂ 偏置 ≤0.05 m/s；reciprocity P95 ≤0.10μs。
- **失败动作**：先修 DSP（模板/窗/加权），不训练模型。若最差帧确超门，考虑 CO₂ 高端降 L_m 上限或加权，登记为域约束。
- **已通过（2026-07-22）**：verdict=`f3_wide_dsp_passed`；stress 30 帧 max peak AB/BA ≈0.090/0.047；窄域 `outputs/tv3_bidir/dsp_fidelity/` 未动；`stage_status.f3_wide` 并列。

### F4-wide —— 可辨识性 v2-wide 审计

```bash
python scripts/run_tv3_identifiability_v2.py --config configs/tv3_bidir_identifiability_v2_wide.json
```

- 门：联合 Fisher 非秩亏；产出六窗 P90 与主导项排序；与 §Context.2 先验交叉核对（O₂=20 锚点不变）。
- 预期：业务 verdict 大概率仍 `coarse_monitoring_only`（精度墙未动）；阶段门通过即可进 F5。**F5 幅度门在 F4 产出后预注册**，不得事后调。
- **已通过（2026-07-22）**：业务=`coarse_monitoring_only`（阶段门 pass）；六窗危害锚定；拒绝率 0；窄窗 P90 max 名义≈4.50 / 保守≈9.74 vol% O₂（门 0.4 未达，与窄域同构）；先验交叉核对通过；F5-wide 预注册含 `criterion_c_anchor=in_domain_a1_v_path_zero`；窄域 `identifiability_v2/` 未改写。

### F5-wide —— 正式数据与五臂对照（服务器）

- **F5-S 已落地（2026-07-22）**：`bidir_spxy_observed_ab_v1`；`recompute_tv3_split --spxy-x-profile bidir_spxy_observed_ab_v1`；协议 `derive_secondary_selectors=true` 跑 S-Y/S-L×3 seeds 全矩阵并评 12 格判据 (d)。
- **仍建议先 smoke 再 6000**：本地用 `tv3-bidir-smoke-wide` 跑通 bootstrap→派生→重建 cache→门逻辑后，再服务器正式集。

工作目录 `tunnel_ventilation/`：

```bash
# 1) 正式 6000-wide（勿覆盖 data/tv3-bidir-6000）
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark \
  --output-root data --preset bidir-formal-6000 --composition-domain wide

python scripts/check_slow_channels.py data/tv3-bidir-6000-wide

# 2) 五臂 + F5-S（写 outputs/tv3_bidir/model_protocol_wide/）
# incomplete → exit 2；矩阵完整但 (d) 失败 → exit 1；全过 → exit 0
python scripts/run_tv3_bidir_model_protocol.py \
  --config configs/tv3_bidir_model_protocol_wide.json --stage all --device cuda

# 3) 门逻辑自检
python -m pytest -q tests/test_tunnel_ventilation_bidir_secondary_selectors.py \
  tests/test_tunnel_ventilation_bidir_model_protocol.py
```

- 判据（域内）：(a)–(e) 同前；(d) 为 **2×3×2=12 格** A3−A1 O₂ ΔR²，禁止均值掩盖。
- 验收：`f5_verdict.json` 且 `verdict=f5_model_protocol_passed`（exit 0）后进 F6-wide。

### F6-wide —— verdict 与回填

- 汇总 F4/F5-wide，记忆库结论表加 wide 行；窄域行保持不变。

---

## 6. 回滚与 write-once 纪律

- 全部 `-wide` 产物一次写入、独立目录、不覆盖窄域。
- 代码改动可回滚：`WIDE_COMPOSITION_RANGES` 与 spec `composition_domain` 字段独立，删除即回窄域默认；单向路径全程未动。
- registry/config/输出目录 `-wide` 后缀全局可 grep，误用即显形。
- 若 F3-wide 门失败且不可修 → 记 `estimator_failed`，不进 F4/F5，不产出误导性模型结论。

---

## 7. 待确认清单（执行前）

1. ~~Decision Gate 0-A：作用范围~~ → **已确认 A1 仅 F 线（2026-07-22）**
2. ~~Decision Gate 0-B：旧窄域去向~~ → **已确认 B1 并行留档（2026-07-22）**
3. ~~F5 判据 (c) 重锚到域内 A1 零流~~ → **已确认并落地（2026-07-22）**
4. ~~窄窗铺法~~ → **危害锚定 6 窗（F4-wide 已用）**
5. ~~F3-wide CO₂ 高端是否降 L_m~~ → **F3-wide 已过门，无需降 L_m**

> Decision Gate 全部关闭；F5-S 代码已落地；F5-wide 仅差 smoke 端到端 → 服务器正式跑数。
