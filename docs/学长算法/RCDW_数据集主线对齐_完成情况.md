# RCDW 数据集主线对齐 — 完成情况

## 0. 文档定位

- **版本**：v1.7（Phase 1-5 完成 + Phase 6A-6E 实际结果同步）
- **日期**：2026-07-01
- **方案**：`docs/学长算法/RCDW_数据集主线对齐改动方案.md` v1.2
- **审查**：`docs/学长算法/RCDW_数据集主线对齐方案_审查报告.md`
- **Phase 6 路线**：`docs/学长算法/RCDW_数据集主线对齐_Phase6路线.md`
- **总测试数**：222 项全部通过（2026-07-01 在 `rcdw_mgda` 下实测）
- **代码总量**：约 +8400 行（含测试）
- **commit 序列**：`7671e7f`（Phase 1+2）→ `da5cddc`（Phase 3）→ `5eb8001`（Phase 4+5）→ `099d8cc`（文档同步）→ `de4b08f`（P2 修复）

> 当前文档保留 Phase 1-5 的落地记录，同时同步 Phase 6 的实际代码进度。Phase 6A-6D 已完成实验闭环；Phase 6E 的 `pressure_drift` 已完成代码、配置、测试与 perturb 实验闭环，并在 `runs/phase6e_pressure/perturb/` 生成 12 张 PNG。

---

## 1. Phase 1：core schema + IDs + conditions + phases + gas_state

**commit**：`7671e7f`（与 Phase 2 合并提交）

**产出**：

| 文件 | 说明 |
|------|------|
| `rcdw/sim/core/schema.py` | `SCHEMA_VERSION="rcdw-benchmark-1"`、7 个 SLOW_CHANNELS（删 V_NDIR_CH4）、SPLIT_NAMES 仅 train/val/test、LEGACY 黑名单 |
| `rcdw/sim/core/ids.py` | `RCDW-M{index:06d}` / `RCDW-Q{index:06d}` 前缀 |
| `rcdw/sim/generation/gas_state.py` | Magnus 公式 H2O 摩尔分数（独立重写，不 import 主线） |
| `rcdw/sim/generation/conditions.py` | LHS d=2 + simplex 映射 + N2 ≥ 55% 边界保护 |
| `rcdw/sim/generation/phases.py` | v1.2 YAGNI 仅保留 `STANDARD_EXPOSURE`；`resolve_phase_schedule` 对未实现 profile 显式 `NotImplementedError` |

**测试**：`test_conditions.py`（11）+ `test_phases.py`（21）= 32 项

---

## 2. Phase 2：声学 + 光学 + waveforms 物理栈

**commit**：`7671e7f`

**产出**：

| 文件 | 说明 |
|------|------|
| `configs/spectral-defaults.json` | 仅 (CO2, H2O) + co2 单通道 |
| `rcdw/sim/generation/spectral/{cache,filters,integration,tabulated_backend,hitran_backend,defaults,__init__}.py` | 完整 7 文件子包，独立维护 |
| `rcdw/sim/generation/acoustic_physics.py` | `linear_mixing_v1` 三组分声速（O2=329.5/CO2=268/N2=353 m/s）、5 项衰减（经典+CO2/N2/O2/H2O 弛豫，新增 O2 弛豫，删 H2 扩散和 CH4 弛豫），`acoustic_model_metadata()` |
| `rcdw/sim/generation/waveforms.py` | 超声 + 光纤麦克风，接口签名改为 `x_o2/x_co2/x_n2` |
| `rcdw/sim/generation/optical_crosstalk.py` | 仅 CO2 ← H2O 单向交叉 |
| `rcdw/sim/generation/optical_backend.py` | HITRAN 缓存收集/校验/co2 单通道吸光度 |

**测试**：`test_acoustic_physics.py`（16）+ `test_waveforms.py`（12）+ `test_optical_backend.py`（19）= 47 项

**关键决策落实**：
- O₂ 声速取 NIST WebBook 25°C 1 atm 值 329.5 m/s（方案 §5.5 v1.1 修正）
- `acoustic_model_metadata.o2_relaxation_source = "placeholder_v1"` 显式标 TBD（方案 §13.3）
- `get_default_ndir_filter("ch4")` 抛 `ValueError`（方案 §11.1）
- waveforms 签名拒收 `x_h2 / x_ch4` 旧参数

---

## 3. Phase 3：slow + benchmark + packaging + validation 端到端落盘

**commit**：`da5cddc`

**产出**：

| 文件 | 说明 |
|------|------|
| `rcdw/sim/generation/slow.py` | multi-tau equilibrium 双指数 RC + recovery floor + drift + random walk + 高斯噪声；blake2b 双流 RNG（`_stable_uint32`）；baseline = 100% N2；不移植 legacy empirical（方案 v1.1 §5.4）；HITRAN + empirical 双后端 |
| `rcdw/sim/packaging/arrays.py` | 13 个 sequence npy + labels/y.npy + metadata |
| `rcdw/sim/packaging/manifest.py` | `composition_scheme="rcdw_o2_co2_n2"`、`train_modalities=[slow, ultrasonic]`、`scaler_metadata.peak_index_strategy="skip"` |
| `rcdw/sim/packaging/splits.py` | mixture_id 分层 70/15/15（`test_ratio=0.15`，剩余 group 全归 test 保证全覆盖，不保留 extrapolation） |
| `rcdw/sim/packaging/scalers.py` | train-only Z-score + 异质通道 `passthrough` 策略（方案 v1.2 §6.5）：`peak_index / tof_quality / tof_accepted` 跳过归一化，scaler 输出对这些通道显式标 `strategy=passthrough` |
| `rcdw/sim/packaging/{index,io,constants}.py` | 等价 HG，独立维护 |
| `rcdw/sim/validation/integrity.py` | 10 项不变量校验；新增 scaler passthrough 标记不变量（v1.2） |
| `rcdw/sim/generation/benchmark.py` | `BenchmarkGenerationSpec` + `generate_benchmark_dataset` 全流程编排（单进程版） |
| `scripts/generate_benchmark.py` | CLI 入口 |

**测试**：`test_slow.py`（20）+ `test_packaging.py`（15）+ `test_validation.py`（14）+ `test_benchmark_e2e.py`（10）= 59 项

**配置**：`configs/default.yaml` + `configs/smoke.yaml` 新增 `generation / spectral / phases / splits / scalers` 节、`data.dataset_root / data.train_modalities` 字段

**CLI smoke 验证（empirical 后端）**：
```
python -m scripts.generate_benchmark --config configs/smoke.yaml --dataset-slug rcdw-smoke
→ 64 sequence × 32 timestep, validation status=pass, split 44/9/11
```

---

## 4. Phase 4：Dataset + 模型通道适配 + Stage A/B

**commit**：`5eb8001`（与 Phase 5 合并提交）

**产出**：

| 文件 | 说明 |
|------|------|
| `rcdw/models/single_modal.py` | 12 维通道索引常量（`IDX_NDIR_CO2=0 / IDX_TCS=1 / IDX_T_C=2 / IDX_P_MPa=3 / IDX_H_RH=4 / IDX_L_m=5 / IDX_PISTON=6 / IDX_US_TOF=7 / IDX_US_SPEED=8 / IDX_US_PEAK=9 / IDX_US_QUALITY=10 / IDX_US_ACCEPTED=11`）、`SENSOR_INDICES={"ndir":0, "tcd":1, "usn":8}`、`ENV_INDICES=[2,3,4]`、`INPUT_CHANNELS=12` |
| `rcdw/models/feature.py` | 12 维输入校验；`delta_T/P/RH` 从 ENV_INDICES 读取；旧 6 维输入抛 ValueError |
| `rcdw/models/rcdw.py` | docstring 更新 `(B, L=8, 12)`；`RCDW_MGDA.__init__` 新增 `window` 参数 |
| `rcdw/data/dataset.py`（新增） | `BenchmarkDataset(data_root, split, window, modalities)`：从磁盘读 slow（7 维）+ ultrasonic 元数据（5 维）拼成 12 维 (B, L, 12) 张量；fiber_mic 不读；标签从百分比转 [0,1] 闭包 |
| `rcdw/perturbation/inject.py` | 通道索引重映射到 12 维新布局（方案 §9.1）；拒收旧 6 维输入；ultrasonic 目标改为 IDX_US_SPEED |
| `scripts/{train,eval,perturb}.py` | 从 `BenchmarkDataset` 加载（替换旧 `synth.make_splits`） |
| `rcdw/data/synth.py` | **删除** |

**测试**（更新 + 新增）：
- `test_single_modal.py`（11）— 重写，覆盖 12 维索引常量、`extract_modal_input` 三模态、未知 modality 抛错
- `test_feature.py`（8）— 12 维输入；拒收 6 维；ENV_INDICES 读取断言
- `test_perturbation.py`（Phase 4 时 9 项，当前 10 项）— 12 维输入；5 类基础扰动新索引；Phase 6E 追加 `pressure_drift` 压力通道测试；逐 kind 验证只改目标通道不改其他
- `test_dataset_loader.py`（10）— **新增**：module 级 fixture 生成 16-seq smoke benchmark，验证窗口 shape (8,12)、split 不重叠、标签 sum=1、前 7 维=slow 后 5 维=ultrasonic、fiber_mic 不读、torch DataLoader 兼容
- `test_w_base_alignment.py`（5）— **新增**：W_base 行序与 SENSOR_INDICES 一致；W_base 每列 sum=1；`RCDW_MGDA` forward 端到端 12 维通过；Y_modal 输出有效分布
- `test_synth.py`（旧）— **删除**（方案 §11.3）

**Phase 4 累计**：43 项更新测试 + 15 项新增测试

**Stage A/B 端到端**（empirical 后端，3 epoch smoke 验证）：
```
generate (64 sequence) → train (Stage A: NDIR/TCD/USN ckpt; Stage B: ErrorNet+Fusion)
  → eval test: MAE ≈ 0.07-0.08（合理玩具水平，标签已转 [0,1]）
```

**历史 ckpt 废弃**：旧 `runs/stage_a/*.pt` 与 `runs/stage_b/rcdw.pt`（基于旧 toy `synth.py` 与 6 维输入）与新数据契约不兼容，必须删除重训（方案 §8.6）。

---

## 5. Phase 5：扰动评测 + 配置 + 文档收口

**commit**：`5eb8001`（与 Phase 4 合并提交）

**产出**：

| 文件 | 说明 |
|------|------|
| `rcdw/perturbation/inject.py` | 已在 Phase 4 重写完成 |
| `scripts/perturb.py` | 从 `BenchmarkDataset.test` 加载，扰动注入 → 模型推理 → 按配置中扰动类型生成 PNG |
| `docs/学长算法/RCDW_数据集主线对齐_完成情况.md` | **本文档**（新增） |

**端到端扰动验证**（empirical 后端 + 3 epoch smoke 模型）：
- 5 类扰动 × 7 强度 × 2 张图 = **10 张 PNG 全部生成**
- thermal/ultrasonic 扰动随 level 提升 MAE 单调上升（符合预期）
- optical_atten/scat/temperature 在该模型规模下表现稳定
- 退化硬抑制 `hard_suppress` 在 level=0 即触发，与历史一致

当前补充：Phase 6E 已在代码中新增第 6 类 `pressure_drift`；使用 `configs/phase6e-hitran-smoke-128-pressure.yaml` 跑 perturb 已在 `runs/phase6e_pressure/perturb/` 生成 12 张 PNG。结果显示 `temperature` 仍是当前 scaler-on 128-seq smoke 的主导扰动源，`pressure_drift` 未造成可观测退化。

**扰动语义变化对照**（方案 §13 风险）：

| 扰动 | 旧目标 (idx) | 旧物理量 | 新目标 (idx) | 新物理量 | 语义差异 |
|------|------------|---------|-------------|---------|---------|
| optical_atten | 0 (S_ndir) | toy 归一化 | 0 (V_NDIR_CO2) | 真实 V | 物理意义更清晰 |
| optical_scat | 0 (S_ndir) | toy 归一化 | 0 (V_NDIR_CO2) | 真实 V | 同上 |
| thermal | 1 (S_tc) | toy 归一化 | 1 (V_TCS) | 真实 V | 同上 |
| ultrasonic | 2 (S_us) | toy 归一化 | 8 (US_SPEED) | m/s ~340-360 | **量级变化，level 含义不同** |
| temperature | 4 (T) | 旧 K 单位 | 2 (T_C) | °C 单位 | level=0.1 对应 +8°C |

---

## 6. 累计统计

| 阶段 | commit | 新增文件 | 修改文件 | 删除文件 | 新增测试 | 累计测试 |
|------|--------|---------|---------|---------|---------|---------|
| Phase 1+2 | `7671e7f` | 27 | 0 | 0 | 79 | 79 |
| Phase 3 | `da5cddc` | 17 | 2 | 0 | 59 | 138 |
| Phase 4+5 | `5eb8001` | 4 | 11 | 2 | 60 | 198 |
| 文档同步 | `099d8cc` | - | - | - | - | 198 |
| P2 修复 | `de4b08f` | 0 | 6 | 0 | 0 | 198 |
| Phase 6A-6D | 工作区增量 | `test_generate_benchmark_cli.py`、`test_input_scaler.py` 等 | 多个生成、训练、评估、扰动模块 | 0 | +23 | 221 |
| Phase 6E pressure_drift 接入与实验 | 工作区增量 | `configs/phase6e-hitran-smoke-128-pressure.yaml`、`runs/phase6e_pressure/perturb/*.png` | `inject.py`、`test_perturbation.py` | 0 | +1 | 222 |
| **当前合计** | 工作区实测 | - | - | **2** | **222** | **222** |

`de4b08f` 不新增测试文件，复用既有 validation / dataset / benchmark / feature / W_base 测试覆盖，重点修复：
- `validation_summary.json` 写出并校验 scaler 相关字段，避免 summary 与 manifest/scalers 失配。
- `BenchmarkDataset.window` 在初始化阶段要求 `>= 2`，避免窗口长度 0 或 1 进入后续训练路径。

Phase 6 关键增量：
- 6A：`scripts.generate_benchmark --precompute-cache-only` 已实现，HITRAN cache 可独立预热。
- 6B：64-seq 与 128-seq HITRAN smoke 已完成 generate -> train -> eval -> perturb。
- 6C：benchmark 生成支持 `num_workers/chunk_size` 可选并行，确定性 parity 测试通过。
- 6D：12 维 `input_scaler.json` 已由 train split 拟合并通过 `BenchmarkDataset` 默认应用；scaler-on 明显优于 scaler-off。
- 6E：`pressure_drift` 已接入 `PERTURBATION_KINDS`、Phase6E 配置与测试；6 类扰动图和趋势说明已完成，输出在 `runs/phase6e_pressure/perturb/`。

---

## 7. 与方案的偏差与说明

| 项 | 方案描述 | 实际实现 | 说明 |
|---|---------|---------|------|
| Phase 4 `data.n_train / n_val / n_test` 字段 | 方案 §10.3 要求移除 | **已清理** | Phase 6 代码质量修复中删除死配置节；新代码只读 `data.dataset_root`、`window`、`seed`、`train_modalities`、`apply_input_scaler` |
| `tests/_legacy/` 归档 | 方案 §3.2 提到归档目录 | **直接 git rm** | 审查 §5.5 建议：git 保留历史，无需 _legacy/ 子目录。本次实施按审查意见处理 |
| benchmark.py 并行 chunk | 方案 §5.9 描述 ProcessPoolExecutor + chunk | **已实现可选并行** | `num_workers=1` 默认保持单进程；`num_workers/chunk_size` 并行路径已通过 empirical 与 HITRAN 合成 cache parity 测试 |
| `precompute-cache-only` CLI 选项 | 方案 §13.1 建议 | **已实现** | HITRAN 后端可只预热 cache，不落盘半成品 benchmark；empirical 后端为 no-op |
| `h2o_cross` / `pressure_drift` 扰动 | 方案 §9.2 可选新增 | **`pressure_drift` 已工程与实验闭环，`h2o_cross` 未实现** | `pressure_drift` 已在 `inject.py`、Phase6E 配置、测试与 perturb 输出中落地；当前实现只移动输入 `P_MPa`，未重新生成压力派生物理观测，结果未显示可观测退化 |
| 标签语义 | 方案 §8.6 暗示 [0,1] | **`BenchmarkDataset` 把磁盘 0-100 转 [0,1]** | benchmark 落盘组分仍是百分比 0-100（方便审计），训练侧统一为 [0,1] 闭包以与 `SingleModal` 输出与 `normalize_composition` 对齐 |

---

## 8. 历史 ckpt 废弃声明

> **数据契约变更后必须重训**。

以下旧 ckpt 与新数据集**不兼容**：

- `rcdw_mgda/runs/stage_a/*.pt`（基于旧 toy `synth.py` 6 维输入训练）
- `rcdw_mgda/runs/stage_b/rcdw.pt`

新数据契约：
1. 通道布局 6 → 12 维
2. 标签语义 toy Dirichlet → HITRAN 物理建模
3. Scaler 统计量重新拟合（train-only + passthrough 通道）
4. baseline = 100% N₂ 物理化

**必须**：删除旧 ckpt 目录，按新数据生成 → Stage A 重训 → Stage B 重训。

---

## 9. 后续建议（Phase 6+）

完整路线已拆分到 [RCDW_数据集主线对齐_Phase6路线.md](RCDW_数据集主线对齐_Phase6路线.md)。本节只保留摘要，避免后续入口分散。

| 优先级 | 项 | 说明 |
|--------|-----|------|
| P0 | HITRAN cache 预热与默认 smoke 可用 | ✅ 已完成：`--precompute-cache-only` + HITRAN smoke generate 闭环 |
| P1 | HITRAN smoke 证据链 | ✅ 已完成：64 与 128 sequence 均完成 generate -> train -> eval -> perturb |
| P1 | benchmark.py 并行化 | ✅ 已完成确定性门槛：可选并行 + parity 测试 |
| P2 | 12 维 input scaler 应用 | ✅ 已完成：`BenchmarkDataset()` 默认跟随 manifest 应用 scaler，scaler-off ablation 可通过 `apply_input_scaler=False` |
| P2 | `pressure_drift` / `h2o_cross` 扰动 | `pressure_drift` 已完成代码/配置/测试/实验闭环；`h2o_cross` 尚需定义语义后再实现 |
| P3 | O₂ 弛豫参数实测校核 | 替换 `alpha_lambda_max_o2=0.002 / f_relax_o2_per_atm=50000.0` 占位值 |
| P3 | 多 `stage_profile` 激活 | 按方案 §5.2 三步路径：`generate_condition_rows` 改 1:N → `phases.py` 拷回其他 schedule → splits 同 mixture 分组验证 |

---

## 10. 验证清单

| 项 | 通过状态 |
|---|---------|
| 222 项测试全部通过 | ✅ |
| 端到端 generate → train → eval → perturb 跑通 | ✅ |
| Phase 1-5 基础 10 张扰动 PNG 全部生成 | ✅ |
| Phase 6E `pressure_drift` 配置 12 张扰动 PNG 全部生成 | ✅ |
| validation_summary `status=pass`（empirical 后端 smoke） | ✅ |
| `composition_scheme=rcdw_o2_co2_n2` / `schema_version=rcdw-benchmark-1` | ✅ |
| `train_modalities=[slow, ultrasonic]` | ✅ |
| `scaler_metadata.peak_index_strategy=skip` | ✅ |
| `peak_index/tof_quality/tof_accepted` 标 `strategy=passthrough` | ✅ |
| SPLIT_NAMES 不含 extrapolation | ✅ |
| `get_default_ndir_filter("ch4")` 抛 ValueError | ✅ |
| `resolve_phase_schedule("variable_onset")` 抛 NotImplementedError | ✅ |
| LEGACY 字段被 validation 拒收 | ✅ |
| RCDW-M/Q 前缀 ID | ✅ |
| baseline = 100% N₂ | ✅ |
| `_stable_uint32` blake2b 双流 RNG | ✅ |
| inject 旧 6 维输入抛 ValueError | ✅ |
| `BenchmarkDataset` 不读 fiber_mic | ✅ |
| W_base 行序与 SENSOR_INDICES 对齐 | ✅ |
| `--precompute-cache-only` 可预热 HITRAN cache | ✅ |
| 64/128 sequence HITRAN smoke 训练评估扰动闭环 | ✅ |
| `num_workers/chunk_size` 可选并行确定性等价 | ✅ |
| 12 维 input scaler 默认应用且支持 scaler-off ablation | ✅ |
| `pressure_drift` 只改 `P_MPa` 通道 | ✅ |
| Phase 6E `pressure_drift` 实验图与趋势表 | ✅ |
