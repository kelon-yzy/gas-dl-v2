# tv3 多频弛豫谱与湿度差分（MRS 线）实现指导

> 立项日期：2026-07-24。状态：**✅ 已收尾（fail 路径交付完毕）**；MRS-0/1 已通过；MRS-2=`mrs2_rank_upgraded_p90_fail`（升秩成立，P90/nuisance 未过门）；**MRS-3 未进入**；MRS-6 已于 2026-07-25 交付（`mrs6_hardware_requirements_delivered`），`allowed_next_stage=MRS_line_closed`。本文档登记设计与门禁；MRS-2 正式 verdict 已写入记忆库；MRS-3/5 设计预核数字在对应正式 verdict 产生前不进结论表。
> 优先级：相对 F 线与 COMSOL 隧道多物理场线，本线为当前唯一排期主线；后两者暂缓，不与本线抢正式算力与验收窗口。
> 来源综述：[`references/声速法_N2-O2辨识_深度学习突破路径_综述.md`](../../references/声速法_N2-O2辨识_深度学习突破路径_综述.md)（2026-07-24，deep-research，引用逐条经 Crossref/arXiv 验证）。
> 定位：对 v1 可辨识性 verdict `information_source_upgrade_required` 的第一条**直接升秩**回应线——把观测从"单频 200 kHz TOF 标量"升级为"多频声速色散 c(f) + 弛豫吸收谱 α(f) + 湿度差分"。
> claim scope：`registered_simulation_domain_only`。不做现场/硬件能力声明，不改写 v1 / F4 / F5-wide verdicts，不替换 B7，不重新生成 `tv3-formal-6000` 与 bidir 数据集。
> 关联：[`tv3_identifiability_implementation_plan.md`](tv3_identifiability_implementation_plan.md)（v1 审计与冻结门限）、[`tv3_static_air_feasibility_implementation_plan.md`](../../active/tv3_static_air_feasibility_implementation_plan.md)（S 线；flow=0 基线共享，并行不阻塞）、[`tv3_bidirectional_ultrasound_implementation_plan.md`](../../active/tv3_bidirectional_ultrasound_implementation_plan.md)（F 线暂缓；方向维度与本线正交）、[`tv3_comsol_multiphysics_dl_implementation_plan.md`](../../active/tv3_comsol_multiphysics_dl_implementation_plan.md)（COMSOL 线暂缓）、[`references/tv3_identifiability_business_threshold_evidence.md`](../../references/tv3_identifiability_business_threshold_evidence.md)（P90/nuisance/拒绝率三门的证据来源）。

---

## Context

### 1. 为什么是多频弛豫谱：综述判断与项目实测的对接

综述的核心操作判据（§5.1），登记为本线的第一性规则：

> 单频 TOF 观测下 (O₂,CO₂,T,L) 联合 Fisher 秩为 1。凡不升秩的 DL 改造，天花板就是 rank-1 的 CRB；凡升秩的路线，DL 是把新观测最优反演的工具。**评估任何新算法前，先问"它升秩了吗"。**

项目自身实测已从两个方向验证了这条判据：

- **不升秩的上限已被反复触到**：E1 learned encoder O₂ R² 退化到 −0.01；E1r/E1d/E1d-SB/attachment/LS 全链最多恢复 B1≈0.4 非劣；模块 C、R7 均未超越 B7。这些全部是"同一 rank-1 观测上的更深回归器"。
- **升 nuisance 维不等于升组分秩**：F 线双向把 pair 声速偏差从 1.81 降到 0.037 m/s（flow 解耦有效），但 F5-wide 判据 (a) 的 O₂ OOD MAE 改善只有 0.117 vol%（冻结门 0.5）。方向维度移除了流动混杂，却没有产生新的 O₂↔N₂ 分离信息。

综述五支与本项目的处置对照：

| 综述分支                  | 判断                        | 本项目处置                         | 落点         |
| --------------------- | ------------------------- | ----------------------------- | ---------- |
| A 频域扩展（弛豫吸收-色散谱）      | 攻击秩亏源头，物理正解               | **采纳为主线**                     | MRS-1/2/3  |
| B 波形扩展（全波形/传递函数）      | 不升秩，重新定位为 A 的前端           | 仅在 MRS-4 之后作宽带前端候选；单载波回归臂禁止新增 | MRS-4 / H3 |
| C 混杂即信号（T 解耦 + RH 催化） | 温度墙 2.5 %O₂/K；湿度是 O₂ 专属探针 | C1 温度解耦进反演器表示；C2 湿度差分进观测臂     | MRS-2/3/5  |
| D 主动协同（学习测什么）         | 直接回应 upgrade 要求，但受硬件可行域约束 | 硬件需求说明书为强制交付；可微激励设计为可选扩展      | MRS-6      |
| E 跨模态类比               | 选择性不可迁移，仅架构范式可借           | 只作 H2 反演器结构参考，不作可分性证据         | MRS-5      |

综述同时指出一个可占据的空白：多频弛豫谱至今只与解析反演和 wavelet+SVM 结合（Jia 2018 是唯一 ML 工作），未与现代 DL 结合。本线 MRS-5 的 H2（物理信息反演器）即针对该空白，但**必须先赢过解析反演基线 H1** 才能声明 DL 增益。

### 2. 当前前向模型的信息缺口（2026-07-24 代码核对）

对 `tv3/sim/generation/tunnel_ventilation/acoustic_physics.py` 的核对结论：

| 项           | 当前代码事实                                                                                             | 对 MRS 线的含义                                            |
| ----------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| 声速无频散       | `hidden_sound_speed_v2` = √(γ_mix·R·T/M_mix)，与频率无关（平衡态比热）                                          | 综述主推的 c(f) 色散信息**不存在**，需新增弛豫色散模型                      |
| α 只在单频求值    | `hidden_attenuation_v2` 仅在 200 kHz 载波处计算 Lorentzian 弛豫项                                            | α(f) 谱形信息未被表达                                         |
| O₂ 弛豫整体关闭   | `alpha_lambda_max_o2=0.0`；`f_relax_o2_per_atm=24.0` 无湿度催化项                                         | C2 湿度差分信息源**不存在**；Bass 湿度催化公式需实现                      |
| N₂ 弛豫频率已修正  | `f_relax_n2_per_atm=9.0`（2026-07-24 工作区修正；旧值 65000 错误约 4 个数量级，综述 §A3 疑点已解决）                        | 200 kHz 处影响可忽略，但 MRS 频段内 N₂ 项将真实参与                    |
| N₂ 弛豫强度存疑   | `alpha_lambda_max_n2=0.004` 为经验值；按单弛豫关系 α_λmax≈π·Δc/c 反推 Δc≈0.45 m/s，与 C_vib 推导的 ≈0.03 m/s 差约 15 倍 | 200 kHz 下无影响（f_r=9 Hz 深度冻结），但 MRS 频段内强度必须按 C_vib 重新推导 |
| H₂O 弛豫为经验占位 | `alpha_lambda_max_h2o=0.01`、`f_relax_h2o_per_atm=1e5`                                              | 需按 Bass 1995 文献边界核验后登记                                |
| 湿度只进 CO₂ 弛豫 | `k_h2o_to_f_relax_co2=0.015` 线性修正                                                                  | O₂/N₂ 的湿度催化（非线性、非对称）完全缺失                              |

含义：**综述推荐的信息源目前在数字孪生里不存在，任何模型层工作之前必须先做物理升级。** 因此本线顺序是：MRS-1 物理升级 → MRS-2 纯前向 Fisher 审计判生死 → 之后才允许生成数据与训练模型。这与综述 §5.3 完全一致："先在 forward model 里加入多频/多湿度采样，重新计算 Fisher 秩与窄窗 P90；若仿真中秩仍为 1，则问题在硬件带宽/控湿，而非算法。"

### 3. 频段与量级预核（手算设计输入；MRS-1 单元测试确认前不作为正式依据）

采样域（现行 conditions）：T 15–35 ℃、RH 20–80%、P 0.10–0.709 MPa（≈1–7 atm）、L_m 0.2–0.3 m、x_O2 18–21.2%、x_CO2 0.03–5%。

**弛豫频率覆盖（Bass 1990/ISO 9613-1 公式手算）**：

| 量             | 采样域内范围                 | 端点算例                                                          |
| ------------- | ---------------------- | ------------------------------------------------------------- |
| 水汽摩尔分数 h_w    | ≈0.05–4.5 mol%         | 低端 15 ℃/RH20%/7 atm；高端 35 ℃/RH80%/1 atm                       |
| f_r,O（O₂，湿催化） | ≈**2.3–166 kHz**       | 低端 7×(24+301)≈2.3 kHz；中点 25 ℃/RH50%/1 atm ≈51 kHz；高端 ≈166 kHz |
| f_r,N（N₂，湿催化） | ≈**0.16–1.3 kHz**      | 与 f_r,O 相差约两个数量级——峰位非对称是可分机制之一                                |
| f_r,CO₂       | ≈28–200 kHz（∝p_atm）    | 7 atm 时移近 200 kHz 载波（现有 L_m 上限 0.3 m 的原因）                     |
| f_r,H₂O       | ~100 kHz/atm（经验占位，待核验） | —                                                             |

**强度非对称（可分机制之二）**：C_vib(300 K)：O₂≈0.029R、N₂≈0.0016R，比值≈19。纯气全色散台阶 Δc：O₂≈0.5 m/s、N₂≈0.03 m/s。混合气内 O₂（18–21.2%）分量贡献 ≈0.09–0.11 m/s。

**与业务门的相对量级**：0.4 vol% O₂ 业务门等价单频声速差 ≈0.09 m/s（∂c/∂x_O₂≈−0.227 m/s/%）。O₂ 色散台阶与业务门同量级，原则上"够得着"，但温度经 p_sat(T)→h_w 强耦合进 f_r,O，**温度免疫性必须在联合 Fisher 中量化，禁止用单变量手算下结论**——这正是 MRS-2 存在的理由。

**几何约束（频率下限）**：L=0.2–0.3 m 下按"路径 ≥ 约 10 个波长"要求，可信频率下限 ≈15–20 kHz（10 kHz 处 L/λ≈6–9 属边界，衍射/近场误差需登记）。低 RH×高 P 角落 f_r,O≈2–10 kHz 落在带外，预期是信息死角——Fisher 热力图必须显式覆盖 (f 子集, RH, P) 三轴，把死角量化出来（综述开放问题 1 的项目内回答）。

**频点数**：Zhu 2017a 的 2N+1 规则，带内活跃弛豫过程 ≤3（O₂-湿、CO₂、H₂O）⇒ 声速色散重建需 ≥7 频点。基线设计 K=8 对数栅格 {10, 16, 25, 40, 63, 100, 160, 200} kHz（MRS-0 冻结；MRS-2 对 K=4/6/8 子集做敏感性）。

**逐频观测噪声（登记义务）**：低频载波的 burst 更长（10 kHz 8 周期 = 800 μs，接近 TOF≈725 μs），需减周期数并登记相位法 TOF 精度 ∝1/(f·SNR) 的逐频噪声模型；trigger jitter 3 μs 沿用 v1，逐频独立。α 观测还需逐频幅度标定误差情景。这些噪声全部进 MRS-2 的 CRB。

### 4. 与现有线的关系

- **S 线（静止空气 P0）**：MRS v1 基线同样 `flow=0`，扰动登记（jitter/固定延迟/T/RH/P/SNR）直接复用 S 线与 v1 registry 的既有条目；MRS 不重复登记。
- **F 线（双向）**：方向维度与频率维度正交。双向×多频合流（逐频 AB/BA）列为 MRS 通过后的可选扩展，不进本计划验收。
- **G 线（COMSOL）**：可选 P2 用 FEM 验证色散前向与低频衍射修正，非阻断项。
- **EC-MSW / B 支**：`e2_allowed=false` 维持不变。宽带波形端到端（H3）仅在 MRS-4 显示 DSP 提取存在瓶颈且 H2 已过门时立项；任何单载波新回归臂直接拒绝（第一性规则 + 记忆库停止条件既有）。
- **TDLAS 760 nm**：仍是突破 0.70 的物理后备，与本线并行不冲突；本线不承诺 0.70。

---

## Task

### 1. 目标与非目标

目标：

1. 在数字孪生内实现弛豫色散前向：c(f)、α(f)、湿度催化 f_r,O/f_r,N（Bass 全式）、C_vib 推导的弛豫强度（MRS-1）。
2. 以 v1 冻结门（P90≤0.4 vol%、单 nuisance≤50%、拒绝率≤5%）+ 升秩判据，判定"多频 + 湿度差分"在已登记域内是否成立（MRS-2，生死门）。
3. 若升秩成立：多频 benchmark → 多频 DSP builder → 解析反演基线 H1 → 物理信息 DL 反演器 H2，量化 DL 相对解析法的净增益（MRS-3/4/5）。
4. 无论成败：输出定量硬件需求说明书（频点/带宽/控湿/变压/标定精度），作为 D 支交付（MRS-6）。

非目标：不做现场声明；不复活 E2（FiLM/attention/MoE）；不承诺突破 0.70；不改动 `hidden_sound_speed_v2` / `hidden_attenuation_v2` 的现有行为；不触碰既有正式数据集与 B7 默认头。

### 2. 阶段与验收门（所有数值门在 MRS-0/MRS-2 冻结，正式运行后不得调门）

#### MRS-0 参数 registry 冻结 + 文档勘误复核

- 新建 `configs/tv3_mrs/parameter_registry.json`（模式对齐 bidir F0）：
  - Bass f_r,O(h,p) 与 f_r,N(h,p,T) 完整公式与系数（`literature_bound`，DOI:10.1121/1.400176 + 10.1121/1.412989）；
  - θ_vib(O₂)=2270 K、θ_vib(N₂)=3390 K、CO₂ 弯曲模 ≈960 K 与 C_vib→(Δc, α_λmax) 推导式（`literature_bound`）；
  - K=8 频点集与逐频 burst 周期数、逐频 TOF/幅度噪声模型（`engineering_scenario`）；
  - RH 调制臂定义（同序列双湿度点、稳定时间）与变压臂定义（域内两点）（`engineering_scenario`）；
  - 低频衍射/波束发散：v1 `not_represented` + 先验残差登记，升级路径指向 G 线；
  - 每参数 source tag 四选一，任一参数缺 source 即 fail。
- 勘误复核（本批 2026-07-24 已完成，MRS-0 验收时复查）：`co2_o2_n2_gas_properties.md`（N₂ 65 kHz→9 Hz/atm ×2 处、Bass DOI ×2 处）、`foundation/physics_references.md`（65 kHz ×1、DOI ×1）、`tv3_acoustic_simulation_fidelity_review.md`（DOI ×1）。
- 名词导读联更（§8.8）与 active/docs README 索引（本批已完成）。
- 产物：`outputs/tv3_mrs/mrs0_registry/`；verdict `mrs0_registry_frozen`（registry sha256 写入 `configs/tv3_mrs/stage_status.json`）。

#### MRS-1 弛豫色散前向 + 物理单元测试

- 新模块 `tv3/sim/generation/tunnel_ventilation/relaxation_spectrum.py`（纯函数，numpy/math，200–400 行内）：
  - `relaxation_spectrum(x_co2, x_o2, x_n2, t_c, p_mpa, h_rh, f_hz_array) -> {c_f, alpha_f, 各过程 f_r/强度}`；
  - 色散模型（单弛豫加和近似）：`c(f) = c_eq·(1 + Σ_i (Δc_i/c_eq)·f²/(f²+f_r,i²))`；α(f) = 经典 + Σ Lorentzian；
  - 强度由 C_vib(θ_vib, T) 推导；Kramers–Kronig 单弛豫一致性 α_λmax,i = π·Δc_i/c 作为内部约束；
  - f_r,O/f_r,N 用 Bass 全式（含湿度催化与 T 修正），h_w 复用 `h2o_mole_percent_from_rh`。
- **不改 `hidden_*_v2` 任何现有行为**；回归测试锁定 v2 输出逐位不变（既有 `test_tunnel_ventilation_physics.py` 全绿即证）。
- 单元测试 `tests/test_tunnel_ventilation_mrs_physics.py` 锚点：
  1. f_r,O(h=0)=24 Hz/atm；f_r,O(h=1%, 1 atm)≈29.6 kHz（±1%）；
  2. f_r,N(h=0, 20 ℃)=9 Hz/atm；f_r,N(h=1%, 20 ℃)≈289 Hz/atm（±2%）；
  3. C_vib(O₂,300 K)≈0.029R、C_vib(N₂,300 K)≈0.0016R（±5%）；
  4. 纯气全台阶 Δc(O₂)≈0.5 m/s、Δc(N₂)≈0.03 m/s（±30%，容差来源=综述 §A3(c) 手算链）；
  5. CO₂ 推导强度与现用经验 λ_max=0.12 同量级（因子 2 内；sanity，非阻断，偏差登记）；
  6. f→0 极限回归 `hidden_sound_speed_v2` 的 c_eq（相对差 <1e-9）；f→∞ 回归冻结声速 c_eq+ΣΔc_i；
  7. 200 kHz 处 α(f) 与现 v2 输出的一致性差异登记（新增 O₂/重推导 N₂ 强度会引入可解释偏差，逐项说明来源，不得静默吸收）。
- verdict `mrs1_physics_passed`。

#### MRS-2 前向可辨识性审计（生死门；无波形生成、无训练，成本最低）

- 新模块 `tv3/audit/identifiability_v3_mrs.py`，镜像 v2 结构：point dataclass → 有限差分 Jacobian → Fisher → 相对 SVD 秩（复用 v2 `_relative_svd_rank` 约定）→ CRB → 窄窗 P90。
- 观测臂（命名避开 F5 的 A1–A5）：

| 臂                 | 观测向量                 | 对应综述路线                   | 部署现实性               |
| ----------------- | -------------------- | ------------------------ | ------------------- |
| `obs-single-200k` | 单频 200 kHz TOF       | 负对照                      | 须复现 v1 秩 1，否则审计实现有错 |
| `obs-cfreq`       | K 频 c(f_k)（TOF 派生）   | A 支 Zhu-2017a 最低成本       | 最高（无需测 α）           |
| `obs-calpha`      | c(f_k)+α(f_k)        | A 支完整                    | 中（α 需逐频幅度标定）        |
| `obs-rh-diff`     | obs-calpha + 同点双湿度差分 | C2 湿度探针                  | 中低（需主动控湿两点）         |
| `obs-p-scan`      | obs-rh-diff + 域内变压两点 | D 支变压先例（Petculescu 2006） | 低（需控压）              |

- 参数向量 θ=(x_O₂, x_CO₂, T, L) + RH（联合估计）；nuisance 情景沿用 v1 冻结（1 K、3 μs jitter）+ 新登记：RH 测量误差（文献边界）、逐频幅度标定误差、逐频独立 jitter。
- 门（全部沿用 v1 预注册值，不得重新调整）：窄窗联合 P90 ≤ 0.4 vol% O₂；单 nuisance ≤50%；拒绝率 ≤5%；升秩判据：联合 Fisher 相对 SVD 秩 ≥2。
- 交付：Fisher/CRB 热力图 vs (f 子集, RH, P)；K=4/6/8 频点子集敏感性；死角（低 RH×高 P）定量。
- verdict 分流：
  - `mrs2_rank_upgraded_p90_pass` → 允许进 MRS-3；
  - `mrs2_rank_upgraded_p90_fail` → 不进 MRS-3，转 MRS-6（结论：信息存在但观测精度/频段预算不足，输出精度规格缺口）；
  - `mrs2_rank_still_deficient` → 停线 + MRS-6；结论按综述判据固定表述："问题在硬件带宽/控湿能力，不在算法"。禁止改门重新运行。

#### MRS-3 多频 benchmark（schema + 生成 + smoke 审计）[MRS-2 pass 后才创建代码]

- schema `tunnel-ventilation-mrs-1`（`tv3/sim/core/tunnel_ventilation_mrs_schema.py`）；`sim_revision` 标签 `v8-mrs-dispersion-v1`。
- 生成：每 timestep K 载波顺序 burst（复用 bidir 同 timestep 多发射模式）；逐频 observed TOF/amp/quality；oracle c(f)/α(f) 仅作审计数组。
- 存储决策（registry 冻结）：K=8 全波形约 8×3 GB，采用"波形只存 4 个代表载波 + 全 K observed 特征"方案；int16 + per-timestep scale + `--skip-fiber-mic` 默认沿用。
- RH 调制臂：condition schedule 扩展 `rh_modulation` 字段（baseline 无调制；调制臂=同序列两级 RH + 稳定时间）。
- 数据集 `tv3-mrs-smoke`（本地）→ `tv3-mrs-6000`（服务器）；F2 式审计（样本守恒 / schema / provenance / write-once）。
- verdict `mrs3_smoke_passed` → formal 生成完成。

#### MRS-4 多频 DSP fidelity + builder

- `raw_dsp_mrs_v1`（`tv3/ml/mrs_features.py`，新 builder 名）：逐频 (ĉ_k, α̂_k) + 色散差分 Δĉ(f_k, f_ref) + 复用 e1d_sb 校准栈（corrected TOF / SNR / quality）。
- 门：逐频 peak P95 ≤0.25 sample（对齐 D2b/F3 判定标准）；重建 ĉ_k 相对 oracle 的 MAE 上限由 MRS-2 CRB 预算折算后在 MRS-0 补登记。
- verdict `mrs4_dsp_passed`。

#### MRS-5 模型协议（冻结对照 + 预注册幅度门）

- 冻结参照：B1/B7 不重训，直接引用既有正式指标；split/selector 复用 B7 协议冻结子集（具体臂数在 MRS-0 决定并登记）。
- 头：
  - **H1 解析反演基线**：Zhu-2017a 式最小二乘弛豫重建 →（弛豫参数 + ĉ 特征）Ridge。**DL 必须先赢 H1** 才能声明净增益——这是综述"白地"声明的诚实前提。
  - **H2 物理信息反演器**：encoder（稀疏 (ĉ_k, α̂_k) + 慢通道 → 潜变量 = 组分 + 弛豫参数）+ decoder = MRS-1 前向的 torch 可微实现（`tv3/dl/models/mrs_forward_torch.py`），谱重建残差 + 标签联合 loss；温度解耦（C1）以显式 √T 归一化坐标进入表示。
  - **H3（可选）**：宽带波形端到端解逐频 H(f)；仅当 H2 过门且 MRS-4 暴露提取瓶颈时立项。
- 幅度门在 MRS-0/2 由 CRB 预算折算后预注册（数值先冻结再运行）；结构仿 F5 五判据：窄窗 O₂（P90/MAE）与 OOD O₂ 相对 B7 的改善幅度门 + 三组分非劣门 + 泄漏负对照。
- **泄漏负对照**：H2 限制为单频输入时不得超过 B7；若超过，判特征/数据泄漏，冻结该 run 先查泄漏再谈结论。
- verdict：`mrs5_protocol_pass` / `mrs5_dl_no_gain_over_analytic`（保留 H1，DL 空白声明在已登记域内不成立）/ `mrs5_failed`。

#### MRS-6 硬件需求说明书（D 支交付；任何分流路径都必须产出）

- 从 Fisher 热力图导出定量规格：最优频点集与最小频点数；换能器带宽方案（多对换能器——Ejakov 2003 四对先例 / 宽带接收候选，含重启 fiber_mic 通道作为宽带接收器的仿真评估——代码保留可恢复）；控湿两点规格（ΔRH、稳定时间、湿度计精度）；变压需求（是否必要、两点位置）；α 通道幅度标定精度需求。
- 可选扩展（仅 MRS-5 pass 后立项）：可微激励设计——学习 (f, P, RH) 采样计划以最小化窄窗 O₂ CRB（综述 D 支；Scarlett 2022 方法学）。

### 3. 数据契约增量（MRS-3 起生效）

- 新数组 stems（示例）：`ultrasonic_f{k}`（仅存储子集）、`ultrasonic_tof_observed_f{k}_s`、`ultrasonic_amp_observed_f{k}`、`ultrasonic_tof_quality_f{k}`；oracle-only：`ultrasonic_sound_speed_true_f{k}_m_per_s`、`ultrasonic_alpha_true_f{k}_np_per_m`。
- oracle 真值维持"仅审计/auxiliary"边界，禁止进部署输入；特征变更必须用新 builder 名（`raw_dsp_mrs_v1`）。
- manifest 新增：`mrs_frequency_set_hz`、`rh_modulation`、`sim_revision.tag=v8-mrs-dispersion-v1`。
- 正式模型输出契约不变：`raw3`、`out_dim=3`，闭包类 loss 与 `target_transform` 仍被拒绝。

### 4. 停止条件

1. MRS-2 三个多频臂（obs-cfreq/calpha/rh-diff）联合秩仍为 1 → 停线，仅交付 MRS-6；结论固定为"观测维度/硬件问题，不是算法问题"；禁止改门重新运行。
2. MRS-2 升秩但含 obs-p-scan 在内全部臂 P90 超门 → 不进 MRS-3；MRS-6 输出精度规格缺口。
3. MRS-5 H2 不超 H1 → 保留解析反演 H1；无新机制证据不得追加 DL 变体（对齐记忆库"停止同架构调参"惯例）。
4. 任何增益只出现在 val、不同步到 test/OOD → 判未通过（项目通则）。
5. 提案新增单载波回归臂或 E2 结构 → 直接拒绝（第一性规则 + `e2_allowed=false`）。
6. 泄漏负对照触发 → 冻结该 run，完成泄漏排查并登记前不得引用其任何指标。

### 5. 文件清单

| 类型               | 文件                                                                                                                                                                              | 创建时机      |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| 新建               | `configs/tv3_mrs/parameter_registry.json`、`configs/tv3_mrs/stage_status.json`                                                                                                   | MRS-0     |
| 新建               | `scripts/run_tv3_mrs_registry_freeze.py`                                                                                                                                        | MRS-0     |
| 新建               | `tv3/sim/generation/tunnel_ventilation/relaxation_spectrum.py`、`tests/test_tunnel_ventilation_mrs_physics.py`、`scripts/run_tv3_mrs1_physics_gate.py`                            | MRS-1     |
| 新建               | `tv3/audit/identifiability_v3_mrs.py`、`scripts/run_tv3_identifiability_v3_mrs.py`、`configs/tv3_mrs_identifiability.json`、`tests/test_tunnel_ventilation_mrs_identifiability.py` | MRS-2     |
| 新建（MRS-2 pass 后） | `tv3/sim/core/tunnel_ventilation_mrs_schema.py`、生成器扩展、`tv3/ml/mrs_features.py`、`tv3/dl/models/mrs_forward_torch.py`、协议脚本与配置                                                     | MRS-3/4/5 |
| 修改               | `docs/`（勘误 + 索引 + 导读，本批已完成）                                                                                                                                                     | 立项批       |
| 禁改               | `hidden_sound_speed_v2` / `hidden_attenuation_v2` 行为、`tv3-formal-6000` 与 bidir 数据、B7 头、v1/F4/F5 verdicts                                                                        | 全程        |

---

## Format

### 1. 执行顺序与前置条件

```text
MRS-0 registry 冻结 ─→ MRS-1 物理升级+单测 ─→ MRS-2 前向 Fisher 审计（生死门）
    │                                              │
    │                            pass ─→ MRS-3 benchmark ─→ MRS-4 DSP ─→ MRS-5 模型协议
    │                            fail ─┐
    └──────────────────────────────────┴─→ MRS-6 硬件需求说明书（必交付）
```

- MRS-0/1/2 全部本地可执行（无波形生成、无 GPU）；MRS-3 起 formal 需服务器。
- MRS-1 不触碰共用文件时只需 tunnel_ventilation 子工程 pytest；若改动 `waveforms.py` 等共享层，须按 CLAUDE.md 规则对三场景分别运行测试。

### 2. 最小验证

```bash
# MRS-0：registry 冻结与 sha256 登记
python scripts/run_tv3_mrs_registry_freeze.py --config configs/tv3_mrs/parameter_registry.json

# MRS-1：物理锚点单测 + v2 行为回归
python -m pytest tests/test_tunnel_ventilation_mrs_physics.py tests/test_tunnel_ventilation_physics.py -q
python scripts/run_tv3_mrs1_physics_gate.py

# MRS-2：前向可辨识性审计（五臂 + 热力图）
python scripts/run_tv3_identifiability_v3_mrs.py --config configs/tv3_mrs_identifiability.json

# 子工程回归
python -m pytest -q
```

### 3. 文档回填与联更义务

- 名词导读 §8.8（MRS 线）、`active/README.md`、`docs/README.md`：立项批已更新；每个 verdict 产生后同批次回填。
- 记忆库：仅在 MRS-2 / MRS-5 正式 verdict 产生或触发停止条件时更新（§2.1 状态表 + §六 执行路线）；smoke 与设计决策不进记忆库。
- 综述开放问题 ↔ 本线交付映射（回填责任）：开放问题 1（Fisher 热力图）→ MRS-2；问题 2（弛豫谱深度反演 vs 解析）→ MRS-5；问题 3（湿度差分现场稳健性）→ MRS-2 nuisance + MRS-6 规格；问题 4（激励设计的硬件可行域）→ MRS-6；问题 5（仿真-真实鸿沟）→ claim scope 常设边界，不在本线内解决。

---

## 备选设计与明确不做（记录取舍，防止重复讨论）

- **不做完整耦合弛豫矩阵**（Dain–Lueptow 全式）：v1 用单弛豫加和近似，误差边界在 registry 登记；若 MRS-2 显示近似误差影响判定，再立 MRS-1b 升级，不预先实现。
- **不做变温差分激励**：温度是最强混杂（2.5 %O₂/K），不作为主动激励维度；温度只做解耦表示。
- **不做宽带 chirp 端到端**（H3 之前）：先用离散 K 载波把信息维度问题与波形提取问题分开。
- **c-only 优先原则**：若 obs-cfreq 单独过门，α 观测臂降级为可选——多频声速色散复用现有 TOF 链路，部署改造成本最低（Zhu 2017a 的核心价值）。
- **不在本线内重新生成旧数据**：单频 200 kHz 数据集与全部既有结论不受 MRS-1 物理升级影响（新物理只进新 schema/新 revision 数据）。

---

## 实施记录

- 2026-07-24 立项。本批完成：本计划文档；`active/README.md`、`docs/README.md`、名词导读 §8.8 索引联更；三份文档勘误（N₂ 65 kHz→9 Hz/atm ×3 处、Bass 1990 DOI 10.1121/1.400476→10.1121/1.400176 ×4 处）。代码侧 N₂ 弛豫频率修正（四场景 `f_relax_n2_per_atm=9.0`）由同日更早批次完成，见工作区未提交改动。MRS-0 尚未启动。
- 2026-07-24 MRS-0 通过：`verdict=mrs0_registry_frozen`；`configs/tv3_mrs/parameter_registry.json` sha256=`cb5f697597dddb05b580f4d954a797ec9e2891cd1501530397901d32eb051422`；产物 `outputs/tv3_mrs/mrs0_registry/`；勘误三文档复查通过；`allowed_next_stage=MRS-1_relaxation_spectrum_physics`。未改 `hidden_*_v2`、未生成波形、未进记忆库。
- 2026-07-24 MRS-1 通过：`verdict=mrs1_physics_passed`；新模块 `tv3/sim/generation/tunnel_ventilation/relaxation_spectrum.py`（Bass f_r + C_vib 强度 + 单弛豫色散/吸收）；单测 `tests/test_tunnel_ventilation_mrs_physics.py` + v2 回归共 32 passed；CO₂ 弯曲模简并度 g=2（否则相对经验 λ_max=0.12 超出因子 2）；200 kHz α 相对 v2 偏差已显式登记（O₂ 启用 + N₂/CO₂ 强度重推导）；产物 `outputs/tv3_mrs/mrs1_physics/`；`allowed_next_stage=MRS-2_forward_identifiability_audit`。未改 `hidden_*_v2`、未进记忆库。
- 2026-07-24 MRS-2 正式审计：`verdict=mrs2_rank_upgraded_p90_fail`。负对照 `obs-single-200k` 秩=1；多频臂最小秩 cfreq/calpha/rh-diff=3/4/5（升秩成立，frac_rank≥2=100%）；全臂窄窗 P90 与 nuisance 均未过门（最佳 `obs-p-scan` max P90≈4.03、median≈2.40 vs 门 0.4）。噪声模型：逐频独立 3 μs jitter（v1 对齐）+ NDIR CO₂ 先验 0.05 vol%。**禁止进 MRS-3**；`allowed_next_stage=MRS-6_hardware_requirements`；草案见 `active/tv3_mrs6_hardware_requirements.md`。记忆库已更新。
- 2026-07-25 计划执行审查（文献核对 + 数值复核）：MRS-1 公式/常数/锚点与 Bass 1990/1995、ISO 9613-1 及标准单弛豫理论逐项一致（Bass 双 DOI Crossref 核实）；MRS-2 判定与独立手算量级吻合（带内 O₂ 色散信号 ≈0.0048 m/s/vol% → TOF 差分 ≈0.01 μs/vol%，被 3 μs jitter 淹没约 300×）。登记事项：负对照单行观测结构必然秩 1（检验力弱）；unstable_fd_count=567 定位为域边界单侧差分截断 artifact；c_eq 无湿度依赖未入 representation_audit；H₂O 弛豫 Bass 1995 核验未闭合；"Zhu 2017a" 出处待核（最接近 Zhu 2018 MST，DOI:10.1088/1361-6501/aa96da）。均不改变 MRS-2 verdict。
- 2026-07-25 MRS-6 交付：`verdict=mrs6_hardware_requirements_delivered`。新增 `tv3/audit/mrs6_noise_budget.py` + `scripts/run_tv3_mrs6_hardware_requirements.py` + `configs/tv3_mrs6_hardware.json` + `tests/test_tunnel_ventilation_mrs6_budget.py`（复用 MRS-2 Jacobian 机制做噪声预算复评，不重跑 MRS-2、不改门）。核心结论：(σ_TOF×T 先验) 全扫描无组合过 0.4 门（≈1.0–1.3 vol% 饱和，第一约束=L 先验 0.1 mm，第二=α 标定 2%）；full_stack（10 ns + 0.1 K + L 10 μm + α 0.1% + 双湿双压）max P90=0.287 过规格目标但属极端硬件栈；K=4 {25,63,100,200} kHz 与 K=8 差 <1%。产物 `outputs/tv3_mrs/mrs6_hardware/`；说明书升级为正式版；`allowed_next_stage=MRS_line_closed`。记忆库已更新。
- 2026-07-25 业务决策：**0.4 vol% 精度要求暂缓强制、降级为参考标注**（登记于 `active/tv3_mrs6_hardware_requirements.md` §0 与 `references/tv3_identifiability_business_threshold_evidence.md` 状态注记）。历史 verdict（含本线 MRS-2）不改写；不构成改门重跑。直接后果：MRS-6 §6 粗精度选项（obs-cfreq K4 + σ_TOF 0.5 μs + T 0.1 K → ≈1.51/1.23 vol%，约 0.4 参考线 3–4 倍）成为最低成本现实候选；原理解读要点已回填说明书 §9。MRS-3 数据生成/训练仍不重启，按参考精度立项粗监测方案需新计划文档。
