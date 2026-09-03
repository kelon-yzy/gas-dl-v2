# gas-dl-v2 — 多模态气体组分检测 v4

## 项目使命

基于 **NDIR 光学 + 超声波 + 光纤麦克风 + TCS 热导** 四模态传感器仿真信号，使用 DL/ML 模型预测混合气体各组分浓度。

已落地三个检测场景，可并存：

- **掺氢天然气（hydrogen_ng）**：H₂/CH₄/CO₂/N₂，sum=100% 闭包。benchmark `wv4-*`。
- **合成气 / 煤气化制气（syngas）**：H₂/CH₄/CO₂/CO，N₂ 为背景气，sum<100%。Stage Ⅰ 基线 + Stage Ⅱ ablation 完成，benchmark `sg4-smoke` / `sg4-formal`（empirical 后端，6000 序列）可用，HITRAN 后端待补。独立模块路径。
- **掘进通风（tunnel_ventilation）**：CO₂/O₂/N₂，sum=100% 严格闭包但模型层不使用闭包残差头。N₂ 升格为显式预测目标，O₂ 为同核双原子（无红外活性，仅声学+TCS 间接推断）。阶段 1–4 链路 + DL 适配已落地，benchmark `tv3-smoke` 与 `tv3-formal`（600 序列，旧 schema 含 V_NDIR_CH4，待重新生成）/ `tv3-formal-6000`（clean，7 通道无 V_NDIR_CH4）可用，Ridge/TCN 首轮基线 + R0/R1a/R1b + D0 六组特征拆分（clean 6000）已完成；D0 确认 oracle 膨胀 0.18（oracle val O₂ R²=0.6025 vs observed 0.4226）、o2_bins 物理极限，结论 D2 优先、D1 暂缓；多模态 fusion 必须用 `raw3` 直接三输出。2026-07-05 存储优化：tv3 默认 int16 + per-timestep scale + `--skip-fiber-mic`（数据集 17→3 GB，误差/噪声 ≈ 1%，光纤代码保留可恢复，见 `tunnel_ventilation/docs/operations/server_training_guide.md`）。独立模块路径。

## 主线阶段

项目主线自 2026-08-27 起改为「通用多模态融合算法」执行计划（现为 v10），定义在 [general_fusion/项目总体规划.md](general_fusion/项目总体规划.md)，长期方向见 [general_fusion/多模态气体检测通用融合算法_项目指导方向.md](general_fusion/多模态气体检测通用融合算法_项目指导方向.md)。

阶段链：A0 统一契约（2026-08-27）→ A1 Ar-He-CO₂ 仿真 benchmark（2026-08-28）→ A2 通用融合核心 → A2H 高难度压力实验 → A2M 综合收口 → A3 xylene-e-nose 外部验证 → A4 可变传感器集合 → A5 真实混合气验证；A6 论文整合并行。A2、A2H、A2M 首轮分别以负结果、负结果和 `MLP_RETAINED` 关闭；`GF-I14 TQIF-Net` 又在 A2 两档容量比较中全面退化，项目终态为 `SCIENTIFIC_FAILURE / ABANDONED`。TQIF 不得重开、改名或局部修补；当前在下一轮 A2 内先执行 `A2-DYN` 动态数据子工作包（A2-DYN-0 至 A2-DYN-4 已完成：机器协议、物理、pilot、开发难度审计与完整 6,300 观测 / 4,410 组数据包 `DATA_FROZEN`，2026-09-03，见 `general_fusion/docs/algorithm/13_Ar-He-CO2动态时间序列仿真与数据分布规划.md`），A2-DYN-5 时间增量信息门通过后才重新构思具有新算法 ID 的候选，当前候选尚未冻结。A3 因没有创新候选继续阻塞，`general_fusion/configs/experiment/a3_temporal_matrix.json` 只作历史草案。

旧 P0–P3 主线已全部关闭（P3 于 2026-08-26 在 G3-4 失败停止），不再作为前置执行链；tv3 不再排期新实验，hg / sg 保留为跨场景复现台。

## 代码结构

仓库按场景隔离（见根目录 `场景隔离重构计划.md`）。每个子工程是独立 Python 包，各自带 `pyproject.toml`、`configs/`、`tests/`，命令在子工程目录下执行。

```
<subproject>/            # hydrogen_ng | syngas | tunnel_ventilation
├── <pkg>/               #   包名对应 hg | sg | tv3
│   ├── sim/             #   物理仿真：声学、光学、慢通道、波形生成、打包
│   │   ├── core/        #     schema.py / syngas_schema.py / tunnel_ventilation_schema.py
│   │   ├── generation/  #     conditions / acoustic_physics / slow / optical_*
│   │   ├── packaging/   #     数据打包、manifest、scalers（schema 无关）
│   │   └── validation/  #     数据完整性校验（可注入 component_fields / background_fields）
│   ├── dl/              #   深度学习：data/ + models/ + training/ + cli.py
│   ├── ml/              #   传统 ML（Ridge / Mean baseline）
│   ├── pipeline/        #   CLI 编排、实验运行、benchmark 生成
│   ├── common/          #   composition.py / metrics.py / scalers.py
│   └── audit/           #   仅 tv3：Fisher/CRB、MRS/MEI 审计与 freeze 检查
├── configs/             # 实验配置 JSON（含 data/ 子目录）
├── data/                # benchmark 数据集（.gitignore 排除）
├── docs/                # 场景专属文档
├── outputs/             # 运行产物与 freeze（.gitignore 排除）
├── scripts/
└── tests/

general_fusion/          # 通用融合主线（当前活跃子工程，包名 gf，src 布局）
├── src/gf/              #   sim/（Ar-He-CO₂ 生成与审计）dl/（契约、适配器、融合核心、训练、主流架构）ml/（基线）pipeline/（各阶段 benchmark 与 smoke 编排）
├── configs/             #   data/ model/ train/ eval/ experiment/ 五类配置（正式事实源）
├── docs/algorithm/      #   A0–A2M 契约、TQIF 失败归档、评审记录与结果记录
├── docs/history/        #   旧项目历史算法与失败经验复盘（只读）
├── data/ outputs/       #   benchmark 与运行产物（.gitignore 排除）
└── tests/

rcdw_mgda/               # 学长算法 RCDW 复现，独立子工程（rcdw/ 包）
shared/                  # 跨场景 HITRAN 缓存
docs/                    # 项目级文档：阶段规划与阶段产出（docs/p1/ 等）
```

DL 模型族：CNN1D / TCN / CNN1DTCNFusion / PhaseWindowTCN / LSTM / Transformer / PatchTST；`<pkg>/dl/data/dataset.py` 由 manifest 驱动，自动兼容三场景；`<pkg>/dl/training/` 按 `composition_scheme` 切换分箱与 loss 校验。

关键入口（在对应子工程目录下执行）：
- `python -m hg.pipeline.generate_benchmark` — hg benchmark 生成
- `python -m sg.pipeline.generate_syngas_benchmark` — sg benchmark 生成（empirical 后端可用，HITRAN 后端未实现）
- `python -m tv3.pipeline.generate_tunnel_ventilation_benchmark` — tv3 benchmark 生成（仅 empirical 后端，HITRAN 待后续阶段）
- `python -m hg.pipeline.precompute_hitran_benchmark_cache` — hg HITRAN 谱线缓存预计算
- `python -m <pkg>.dl.cli --config <json>` — DL 训练（manifest 自动决定走 hg / sg / tv3 路径）
- `python -m hg.pipeline.run_experiment --config <json>` — hg 多 run 实验编排（sg / tv3 未接入）
- `python -m pytest` — 该子工程测试（pyproject 已设 `testpaths = ["tests"]`）

## 核心概念

### 组分字段约定

- **hydrogen_ng**：`hydrogen_ng/hg/sim/core/schema.py` 中 `COMPONENT_FIELDS = ("x_H2", "x_CH4", "x_CO2", "x_N2")`，4 列 sum=100%。
- **syngas**：`syngas/sg/sim/core/syngas_schema.py` 中 `COMPONENT_FIELDS = ("x_H2", "x_CH4", "x_CO2", "x_CO")` + `BACKGROUND_FIELDS = ("x_N2",)`，4 列 sum<100%；`x_N2 = 100 - sum(targets)`，写入 condition grid 但不入 labels。
- **tunnel_ventilation**：`tunnel_ventilation/tv3/sim/core/tunnel_ventilation_schema.py` 中 `COMPONENT_FIELDS = ("x_CO2", "x_O2", "x_N2")`，`BACKGROUND_FIELDS = ()`，3 列 sum=100% 严格闭包。N₂ 是显式预测目标（与 syngas 不同），数据层闭包但模型层不使用闭包残差头。

下游加载器优先读 `manifest.composition_scheme` 与 `metadata/label_names.npy`，不要直接 import 全局常量推断 schema。

### 慢通道

- **hydrogen_ng**：8 个 — V_NDIR_CH4, V_NDIR_CO2, V_TCS, T_C, P_MPa, H_RH, L_m, piston_position_m。
- **syngas**：9 个，在 hg 基础上新增 `V_NDIR_CO`。
- **tunnel_ventilation**：7 个 — V_NDIR_CO2, V_TCS, T_C, P_MPa, H_RH, L_m, piston_position_m（无 V_NDIR_CH4，场景无 CH₄）。

### Loss 体系

- 闭包类（sum=100%，仅 hg 可用）：`compositional_mse`, `ilr_mse`, `free_component_mse`, `weighted_free_component_mse`
- 开放类（三场景兼容）：`weighted_component_mse`, `mse`, `mae`, `smooth_l1`, `huber`

syngas / tunnel_ventilation 场景下闭包类 loss 由 `validate_loss_composition_scheme()` 自动拒绝；这两个场景也不允许 `target_transform`（ILR/ALR 依赖 sum=100% 闭包残差头）。tunnel_ventilation 数据层 sum=100% 闭包，`sum_abs_error` 可计算作监控项，但模型输出不强制归一化；`cnn1d_tcn_fusion` 在 tv3 下必须使用 `output_mode="raw3"`、`out_dim=3`，`gas_head` 被校验拒绝。

### Benchmark 命名

> **2026-07-02 仿真链路对齐**：超声载波 40kHz→200kHz（PSC200K）、采样率 200k→1MS/s、ADC 16-bit→20-bit（NI-6453）、L_m 范围 0.2~1.8m→0.2~0.3m（200kHz 下长声程信号被 CH₄/CO₂ 弛豫吸收淹没，见 `hydrogen_ng/docs/Phase0_物理可行性核对记录.md`）。旧 benchmark 已归档至 `data/_archived_pre_200khz/`，**不可用于新链路训练**。manifest 新增 `sim_revision` 字段标记链路版本（`v5-200khz-20bit-L03`）。`wv4-smoke` / `sg4-smoke` 已按新链路重生成；formal 集（`wv4-formal-hitran-standard-6000`、`sg4-formal`、`sg4-formal-crosstalk`、`sg4-hitran-smoke`）需用原参数重生成后才能使用。

> **2026-07-03 物理模型严格化**：声速改为理想气混合 `c=sqrt(γ_mix·R·T/M_mix)`、热导改为 Wassiljewa-Mason-Saxena 混合规则、波形用 Lagrange 5 阶分数延迟 FIR 实现亚样本 TOF 定位（1 MS/s 不变，精度 <0.002μs，见 `hydrogen_ng/docs/物理模型严格化实施计划.md`）。旧 v5 benchmark 归档至 `data/_archived_pre_phys_strict/`，manifest `sim_revision.tag` 升级为 `v6-phys-strict`，`physics_backend: "ideal_gas_wms_fracdelay"`。`wv4-smoke` / `sg4-smoke` 已按 v6 重生成；formal 集需重生成。

- `wv4-smoke` / `wv4-formal*` — 掺氢天然气；正式 6000 序列集 `wv4-formal-hitran-standard-6000` 可通过 `--experiment-preset formal-hitran-standard-6000` 一键固定。
- `sg4-smoke` / `sg4-formal` — 合成气 smoke / 正式集（empirical 后端，6000 序列）
- `tv3-smoke` / `tv3-formal` — 掘进通风 smoke / 正式集（仅 empirical 后端，HITRAN 待后续阶段）；`tv3-smoke` 已生成，`tv3-formal` 已按 600 序列规模生成。2026-07-05 起 tv3 默认 int16 + per-timestep scale + `--skip-fiber-mic`（数据集 17→3 GB，物理 ADC 仍 20-bit，存储 dtype=int16，光纤代码保留可恢复）

## 开发注意事项

### 物理建模关键约束

1. **CO 与 N₂ 声学近简并**：两者摩尔质量均为 28 g/mol，声速差 <1 m/s。声学和 TCS 通道几乎无法区分 CO 和 N₂，CO 的可观测性主要依赖 NDIR 光学通道。
2. **N₂ 双重角色**：syngas 场景中 N₂ 不在预测目标中，但声学/衰减/热导计算仍需要 N₂ 浓度（`x_N2 = 100 - sum(targets)`），由 syngas `conditions.py` 与 `slow.py._main_feature_condition` 自动透传。
3. **CO NDIR 串扰**：CO 基频 2143 cm⁻¹ 与 CO₂ ν₃ 2349 cm⁻¹ 间隔 ~200 cm⁻¹，宽带滤光片下存在串扰。已实现 3×3 矩阵 `syngas/sg/sim/generation/syngas/optical_crosstalk.py`；默认 `enable_co_crosstalk=False`（Step 1：CO 通道仅含自身吸收），切到 `True` 启用 CO₂↔CO 互扰（Step 2 ablation）。
4. **O₂/N₂ 声学辨识**（tunnel_ventilation）：O₂ 与 N₂ 摩尔质量差 14.3%（32 vs 28 g/mol），声速差约 6.4%（~22 m/s），是超声通道区分两者的主要物理基础；热导率差异仅约 2.3%，TCS 提供边际辨识力。O₂ 为同核双原子，无红外活性，不设 NDIR 通道。
5. **O₂ 弛豫在 200 kHz 下可忽略**（tunnel_ventilation）：dry air 下 O₂ V-T 弛豫频率 fr,O ≈ 24 Hz/atm（Bass 1990 JASA），远低于 200 kHz 载波，工程实现取 alpha_o2 ≈ 0。
6. **tunnel_ventilation 无光学串扰**：仅 CO₂ 一个红外活性组分（O₂/N₂ 同核双原子无红外活性），不需要串扰矩阵。

### 文件约定

- ID 规范：MixtureId = `M000001`，SequenceId = `Q000001`
- split 主键：`mixture_id`（不是 sequence_id）
- benchmark 产物：`condition_grid_sequence.csv`, `sequence_index.csv`, `manifest.json`, `labels/y.npy`, `sequences/slow.npy`, `metadata/label_names.npy` 等
- manifest 必须含 `composition_scheme` / `background_fields` / `slow_channels` / `labels`，下游以 manifest 为准而非全局常量
- 配置驱动：所有超参数通过 `<subproject>/configs/*.json`，不硬编码

### 测试

修改某个子工程 `<pkg>/` 下的代码后，在该子工程目录运行 `python -m pytest`，以实际通过数为准。syngas 功能在 `syngas/tests/test_syngas_*.py` 系列；tunnel_ventilation 功能在 `tunnel_ventilation/tests/test_tunnel_ventilation_*.py` 系列。

场景隔离后三个子工程各自持有一份仿真链路代码，没有跨场景共享包。改动 waveforms / manifest / validation / metrics / losses / trainer / cli 这类同名文件时，注意判断是否需要在其他场景做同样修改——它们不会自动同步，但也不会互相影响测试。

## 相关文档

| 文档 | 位置 | 内容 |
|------|------|------|
| **项目总体规划** | `general_fusion/项目总体规划.md` | **主线入口（v10）**：A0–A6 阶段定义、TQIF 失败归档、新算法重新构思与近期动作 |
| 项目指导方向 | `general_fusion/多模态气体检测通用融合算法_项目指导方向.md` | 长期方向、数据职责、论文结构论证边界（与总体规划配套） |
| A0–A2M 与 TQIF 记录 | `general_fusion/docs/algorithm/` | 统一契约、评价协议、历史评审、TQIF 失败归档与算法实验结果记录 |
| P1 阶段产物 | `docs/p1/` | 检索协议、七视角检索结果、创新点候选集、评审记录、正式关闭审查 |
| 项目级文档导航 | `docs/README.md` | 跨场景文档索引与各场景文档入口 |
| 场景隔离重构 | `场景隔离重构计划.md` | 子工程拆分的目录归属与迁移记录 |
| 当前实验导读（hg） | `hydrogen_ng/docs/AI_CONTEXT_GUIDE.md` | 比 ARCHITECTURE.md 更接近当前状态的动态主线导读 |
| 当前 DL 主线（hg） | `hydrogen_ng/docs/DL相位统计稳定提取与保留方案.md` | 当前活跃的 DL 方案；旧 PhaseWindowTCN 归档于 `hydrogen_ng/docs/整理归档/dl_iteration_plans/` |
| 架构说明（hg） | `hydrogen_ng/docs/ARCHITECTURE.md` | 已落地的 v4 架构契约 |
| 合成气文档导航 | `syngas/docs/README.md` | 新四组分文档索引、阅读顺序与迁移说明 |
| 合成气适配方案 | `syngas/docs/adaptation_plan.md` | 整合审查修正的完整实施方案 |
| 物理常数速查 | `syngas/docs/physics_references.md` | CO 声学/光学常数 + 编码片段 |
| LHS 采样设计 | `syngas/docs/lhs_sampling_design.md` | 方案 B + 条件顺序采样实现规范 |
| CO 串扰设计 | `syngas/docs/co_crosstalk_design.md` | CO 通道 3×3 光学串扰与消融设计 |
| 文献检索汇总 | `syngas/docs/references/README.md` | 4 份子报告索引 + 可编码常数 |
| CO 声学常数 | `syngas/docs/references/co_acoustic_constants.md` | 声速/弛豫/H₂O 耦合详细文献 |
| CO 光学参数 | `syngas/docs/references/co_optical_hitran.md` | HITRAN 谱线/NDIR 滤光片/串扰 |
| 组分分布 | `syngas/docs/references/syngas_composition_ranges.md` | LHS 采样区间文献支撑 |
| 传感器综述 | `syngas/docs/references/syngas_sensing_survey.md` | 商用系统对比 + 可行性评估 |
| 掘进通风代码契约事实源 | `tunnel_ventilation/docs/掘进通风代码契约事实源.md` | **tv3 硬约束入口**：字段与常量、冻结默认、禁令清单、报告口径、freeze 规则。改 tv3 代码前先读 |
| 掘进通风实验日志 | `tunnel_ventilation/docs/掘进通风实验日志.md` | **tv3 经验入口**：实验时间轴、跨实验规律、已封闭方向、开放问题 |
| 掘进通风文档导航 | `tunnel_ventilation/docs/README.md` | tv3 场景文档索引与阅读顺序 |
| 掘进通风适配方案 | `tunnel_ventilation/docs/foundation/adaptation_plan.md` | tv3 主实施方案：架构决策、分阶段任务、文件清单 |
| 掘进通风物性常数 | `tunnel_ventilation/docs/foundation/physics_references.md` | CO₂/O₂/N₂ 声学、热导、光学物性速查 |
| 掘进通风采样设计 | `tunnel_ventilation/docs/foundation/sampling_design.md` | 2D LHS 采样、联合约束、状态分层 |
| 掘进通风 DL 方案 | `tunnel_ventilation/docs/archive/legacy/dl_training_plan.md` | 通道可辨识性、模型选型、Loss、实验矩阵 |
| 掘进通风服务器训练手册 | `tunnel_ventilation/docs/operations/server_training_guide.md` | Linux + RTX 5880 48GB 服务器训练完整步骤（环境/生成/训练/回收）；§4.5 波形 `waveform_preprocess` gpu/cpu |
| 掘进通风小样本 DL 策略 | `tunnel_ventilation/docs/methods/small_sample_dl_strategies.md` | 9 类小样本策略（数据增强/正则化/轻量模型/集成/蒸馏/元学习/自监督/半监督/物理约束）+ 文献 + 优先级 |
| 掘进通风算法方法论文说明 | `tunnel_ventilation/docs/methods/tv3_算法方法论文说明.md` | **论文素材**：前向物理模型、RawDSP、反演算法族、不确定度量化、可辨识性审计的完整方法链 + 已验证数值 + 投稿须声明的边界 |
| 掘进通风论文结构与投稿方案 | `tunnel_ventilation/docs/methods/tv3_论文结构与投稿方案.md` | 论文定位、图表清单、候选期刊与审稿风险；P0 已产出素材，未做绘图与润色 |
| 学长 RCDW 复现 | `rcdw_mgda/docs/学长算法/RCDW_实施完成情况.md` | 独立子工程 `rcdw_mgda/` 的端到端落地状态（与三个场景子工程完全隔离，互不影响彼此 tests） |
| 工作原则 | `AGENTS.md` | AI 协作规则与边界 |

## 环境

- Python 3.10–3.13（排除 3.14）；当前已验证环境为 Python 3.12 虚拟环境
- 核心依赖：numpy, scipy, scikit-learn, torch, hitran-api, pytest
- 安装：各子工程独立安装，在子工程目录执行 `pip install -e .`；需要测试依赖时用 `pip install -e .[dev]`。根目录没有 `requirements.txt`，依赖声明在各自的 `pyproject.toml`
