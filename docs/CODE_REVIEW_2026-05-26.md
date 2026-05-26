# 代码质量与实现逻辑审查报告

- 评审日期：2026-05-26
- 评审范围：50 个源文件、约 3654 行；docs + tests 125 个；`configs/data/spectral-defaults.json`
- 测试基线：sidecar 记录的 `pytest -q` 125 passed，评审时未重跑（未改动代码）

## 总体评价

骨架清晰，目标契约（`mixture_id` 主键、HITRAN cache-only 预检查、empirical/HITRAN 双 backend、splits/manifest）落得稳，物理路径数学上是正确的。问题集中在三块：

- **dl 数据通路的 mmap 失效与 epsilon 不一致**
- **HITRAN 主线下的死代码与重复计算**
- **`configs/` 与代码之间的"配置仅作文档"漂移**

没有发现会让 benchmark 结果错的硬 bug，但有几处会让后面接 trainer / 跑大 benchmark 时踩坑。

---

## 🔴 Major（影响正确性 / 性能 / 契约）

### M1. `src/dl/data/dataset.py:99-104` — lazy + memmap 实际未生效，全量驻留内存

```python
self._slow = np.load(seq_dir / "slow.npy", mmap_mode="r").astype(np.float32)
self._ultrasonic = np.load(..., mmap_mode="r").astype(np.float32)
self._fiber_mic = np.load(..., mmap_mode="r").astype(np.float32)
```

`mmap_mode="r"` 打开后立刻 `.astype(np.float32)` 会**强制把整个 mmap 复制为 in-memory float32**。fiber_mic 一条 int16 波形 = `(seq, ts, ~2000)`，float32 翻 4 倍。对真实 benchmark（几千 seq × 128 ts × 2000 sample）直接吃光内存。文档和测试名（`test_lazy_loading_defers_array_load`）都误导。

**修复方向**（任选其一）：

- 完全去掉 `.astype(np.float32)`，保持原 dtype，cast 推迟到 `_build_input` 里只对当前 `src_idx` 的切片做（这是真正的 lazy）。
- 显式用 `np.memmap` 而不是 `np.load(mmap_mode='r').astype(...)`。

测试 `test_lazy_loading_defers_array_load` 只检查 `_slow is not None`，没断言"未加载整张"，得跟着改。

### M2. `src/dl/data/scalers.py:29` 与 `src/sim/packaging/scalers.py:12` 阈值不一致

- fit 端：`std = np.where(std > 1e-15, std, 1.0)`
- apply 端：`std = np.where(std < 1e-12, 1.0, std)`

绝大多数通道不会落在 `1e-15 ~ 1e-12` 之间，但一旦某通道在 train split 内方差极小，会出现"fit 时认为可缩放（std≈1e-13）写入，apply 时却被当成 0 强制改成 1"，导致同一份数据 fit 和 apply 算出来的归一化值不同。统一为一个常量（建议 `1e-12`，落入 `sim.core.schema` 或新建 `sim.packaging.constants`）。

### M3. `src/sim/packaging/scalers.py` — 未生效参数 + 无类型注解

```python
def fit_z_score_scalers(matrix, train_indexes, channel_names, modal_groups,
                        transform_target="slow", channel_axis=2):
    ...
    mean = train_x.mean(axis=(0, 1))   # 硬编码 channels-last
```

`transform_target`、`channel_axis` 只写进 JSON 元数据，**计算时被忽略**。任何人传 `channel_axis=1` 都会静默拿到错的 mean/std。违反 `coding-style.md` 的 type hints 必填，也违反 `AGENTS.md` 的"禁止无依据的防御性 / 兜底"。

**修复**：要么真按 `channel_axis` 派生 reduce 维度，要么删掉这两个参数。

### M4. HITRAN 主线下 `src/sim/generation/slow.py:67-78` 仍跑完整 empirical baseline / target

HITRAN backend 分支只用了 `main_sensor_features` 结果里的 `V_TCS`，但代码仍走完整条 `main_sensor_features`（含 `hidden_attenuation_v2`、empirical NDIR 计算、`apply_optical_crosstalk`）来算 `baseline_main` 和 `target_main`。

- 每条 sequence 多跑 2 次完整 empirical 路径，结果丢弃。
- 给"HITRAN 默认主线、empirical 只做对照"的语义打补丁感很强。

**修复**：在 HITRAN 分支只算 `V_TCS` 所需的最小输入；或抽一个 `tcs_baseline_target(condition)` 专用函数避免顺带计算 NDIR。

### M5. `src/sim/generation/spectral/tabulated_backend.py` — 每次调用都重算 filter 与重新校验栅格

```python
for spectrum in spectra:
    if spectrum.wavenumber_cm1.shape != wavenumber.shape \
            or not np.allclose(spectrum.wavenumber_cm1, wavenumber):
        raise ValueError(...)
```

`compute_tabulated_ndir_absorbance` 每 timestep × 每 channel 都重算 `gaussian_filter(wavenumber, ...)` 并对 3 条 spectra 跑 `np.allclose`。HITRAN 默认主线下，对一个 1000 seq × 128 ts benchmark = 256k 次 trapezoid + 256k 次 `allclose`。

**修复**：把 `(grid, filter)` 的派生量缓存进 `_cached_tabulated_spectra` 同一层；栅格一致性在缓存载入时校验一次即可。

### M6. `configs/data/spectral-defaults.json` 是文档级镜像，未被代码读取

全工程只有 `docs/` 和一个测试引用它，`src/sim/generation/spectral/defaults.py` 是真正生效的来源。当前测试只校"两边数值要一致"，意味着：

- 改 JSON 不会改行为。
- 改 Python 默认值如果忘改 JSON，CI 会因不一致挂掉，但运行行为已经偏了。

**修复方向**：要么把 JSON 当成 source-of-truth（在 `defaults.py` 里读 JSON 构造常量），要么把 JSON 改成"自动从代码导出的快照"。`AGENTS.md` 强调"禁止隐式兜底"，现在的镜像状态比有效配置驱动更易让人误判。

---

## 🟡 Notable（建模语义 / 工程边界）

### N1. `src/sim/generation/optical_backend.py:138-148` — `cross_from_*` 是近似分解，不是严格量

```python
"absorption_ch4_cross_from_co2": ch4["absorbance_observed"] - ch4["absorbance_by_gas"].get("CH4", 0.0)
```

由于 NDIR 积分对 transmittance 求加权平均再取 log，`observed - by_gas[CH4]` 在数学上**不等于** CO2/H2O 单独的吸光贡献。CH4 通道里 CO2/H2O 的 τ 很小（近线性近似）时近似可用，但写成 `cross_from_co2` 暗示了"只来自 CO2"的强语义。

**建议**：改名为 `absorption_ch4_residual_from_other_gases`，或在 docstring 写明"近似分解，仅作诊断"。

### N2. `src/sim/generation/slow.py:_path_l_m_for_timestep` 的边界问题

```python
if is_baseline_scan and timestep < q1:
    return path_lms[min(len(path_lms) - 1, timestep // max(1, q1 // len(path_lms)))]
```

当 `q1 < len(path_lms)`（小 `timesteps` + 多声程候选），`q1 // len = 0` → 步长退化为 1 → 实际只走得到 `path_lms[:q1]`，后面的候选根本扫不到。`steady` 分支同问题。

**建议**：用 `np.linspace` 式均匀映射，`idx = int(local * len / span)`，并 clamp。

### N3. `src/sim/generation/conditions.py:69-77` — CH4 < 40% 时静默改 N2，未声明

LHS 路径里 `x_h2 ≤ 30, x_co2 ≤ 15 → 总 ≤ 45 < 60`，所以这段分支永远走不到。但代码留着这段未注释的"修复 N2"逻辑，未来如果有人调宽 H2/CO2 上限，会进入隐式重分布。要么 `assert x_ch4 >= 40.0` 让真正异常的情况显式失败（更符合"禁止兜底"），要么把"理论上不可达"写进注释 + 加 assert。

### N4. `src/sim/generation/acoustic_physics.py:165-166` 计算的 `f_peak / A_fft_max / TOF / Amp` 在 slow path 全部被丢

`build_sequence_arrays` 只取 `V_NDIR_*` 和 `V_TCS`。这些频域 / 时域特征当前是死字段。可能为后续 ML feature 准备，但目前每条 sequence 做 2 次 `main_sensor_features` 都白算（叠加 M4）。

**建议**：要么明确标注"为 Phase 5 ml 特征预留"，要么从 slow 路径里裁剪。

### N5. `src/sim/generation/optical_backend.py:91-94` — 排序 lambda 命名误导

```python
return tuple(
    requirements[key]
    for key in sorted(
        requirements,
        key=lambda item: (item.temperature_k, item.pressure_atm, item.gas, ...),
    )
)
```

`requirements` 是 `dict[SpectralCacheKey, ...]`，`sorted(dict)` 遍历的是 `SpectralCacheKey` 键。lambda 参数命名 `item` 容易让人以为是 `HitranCacheRequirement`，实际是 cache key。功能正确（`SpectralCacheKey` 恰好有同名属性），但可读性差。重命名 `key=lambda cache_key: (cache_key.temperature_k, ...)` 或先 `for req in requirements.values()` 排序更清楚。

### N6. `src/sim/packaging/run_contract.py` 是孤儿

`REQUIRED_RUN_FILES = (...)` 与 `RunOutputContract` 只被自己的测试引用，trainer 还没接入。可以保留（Phase 4 trainer 会用），但应在 docstring 显式标记"待 trainer 消费"，否则容易被未来的人误删。

### N7. `src/dl/training/metrics.py` "整体 R²" 的口径

`regression_metrics(y_pred, y_true)` 在多组分输入下，`ss_tot` 使用 `mean(true, dim=0, keepdim=True)`（per-channel mean），等价于对总平方和的多维 R²。这和"逐组分 R² 平均"、`R²(macro_avg)` 都不同。论文里常见的 NDIR / 混合气 R² 写法是 per-component 后宏平均。

**建议**：在 docstring 里写清口径，避免审稿时被指。

---

## 🟢 Minor（风格 / 契约 / 可读性）

| #   | 位置                                                                                                                                                                                                          | 问题                                                                                | 建议                                                                                |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| m1  | `src/sim/__init__.py`、`sim/core/__init__.py`、`sim/generation/__init__.py`、`sim/packaging/__init__.py`、`sim/validation/__init__.py`、`pipeline/__init__.py`、`ml/__init__.py` | 全是 docstring 占位，没有 `__all__`，没 re-export                                          | `coding-style.md` 要求 `__all__`；至少把 `sim.core.ids/schema` re-export 一下，外部不必走长路径    |
| m2  | `src/dl/models/cnn1d.py:31`                                                                                                                                                                                 | `k = kernel_size if i < 2 else 3` 硬编码                                             | 改为 `kernel_sizes: list[int]` 或都用 `kernel_size`                                    |
| m3  | `src/dl/models/cnn1d.py:37-38`                                                                                                                                                                              | dropout 只挂前 2 层，无理由                                                               | 全挂或写明理由                                                                           |
| m4  | `tests/test_dl_data.py:128-159`                                                                                                                                                                             | `try/except ValueError: assert "..." in str(exc)`，没 `else: raise AssertionError`   | 用 `pytest.raises` 改测试                                                             |
| m5  | `src/dl/data/scalers.py:30-36`                                                                                                                                                                              | `apply_scaler` 对 ndim≠2/3 不报错，会按 1D 静默广播                                          | 显式校验 ndim ∈ {2,3}                                                                 |
| m6  | `src/pipeline/generate_benchmark.py:26`                                                                                                                                                                     | `--seed default 20260524` 硬编码日期                                                   | 用纯数值常量或从配置读取                                                                      |
| m7  | `src/sim/generation/waveforms.py:181`                                                                                                                                                                       | `if peak_abs_v <= 0.0: raise`                                                     | 当前是合理的"显式失败"；建议补一行注释解释这不是兜底                                                       |
| m8  | `src/sim/generation/spectral/defaults.py:43-54`                                                                                                                                                             | `KeyError → ValueError` 用 `from exc` 包装，但错误信息可加上 available channels 列表           | `f"Unknown NDIR channel: {channel!r}. Available: {list(DEFAULT_NDIR_FILTERS)}"`   |
| m9  | `src/sim/generation/optical_backend.py:_compute_hitran_channel_from_cache`                                                                                                                                  | `filter_spec: object` 应为 `NDIRFilter`                                              | 加正确类型                                                                             |
| m10 | `src/dl/training/metrics.py:32`                                                                                                                                                                             | `if ss_tot.item() == 0:` float 等于 0 比较                                            | 用 `< 1e-12`                                                                       |
| m11 | `src/dl/models/registry.py:27-29`                                                                                                                                                                           | `if isinstance(entry, type) and issubclass(entry, nn.Module): return entry(...)` 和下一行 `return entry(...)` 等价 | 合并为一行                                                                             |
| m12 | `src/sim/validation/integrity.py:48`                                                                                                                                                                        | `abs(total - 100.0) > 1e-5`，但 `conditions.py` 用 `round(x, 6)` 量化                  | 收紧到 `1e-4` 之类一个能解释的值，或注释为什么是 `1e-5`                                               |
| m13 | 多处                                                                                                                                                                                                          | `random.Random.randrange(0, 2**32)` 到处都是                                          | 抽 `def random_uint32(rng)` 复用                                                     |
| m14 | `src/sim/generation/slow.py:223`                                                                                                                                                                            | `dict[tuple[str, HitranGridSpec], ...]` 类型对，但 `HitranGridSpec` 浮点字段哈希需依赖前置量化      | 已经在 `gas_state.hitran_temperature_k/pressure_atm` 量化过，OK，加一行注释说明                  |
| m15 | `src/pipeline/__init__.py` 与 `src/ml/__init__.py` 空模块                                                                                                                                                       | 可见占位                                                                              | 不动；但 `ml` 在 `IMPLEMENTATION_PLAN.md` Phase 5，列入 TODO                               |

---

## ✅ 做得好的地方（保留方向）

- `src/sim/core/schema.py` 把所有契约字段都集中在常量里，下游打包 / 校验 / dataset 全部引用，主键语义不会回退到 `sequence_id`。
- HITRAN cache-only 预检查在写 dataset **之前** 失败（`benchmark.py:69 → validate_hitran_benchmark_cache`），不会留半成品。
- `MissingHitranBenchmarkCacheError` 把缺失 cache 的前 8 条 preview 和正确修复命令（`pipeline.precompute_hitran_benchmark_cache`）写进 message —— 失败可操作。
- 测试同时覆盖默认 HITRAN 主线、缺 cache 早失败、逐 timestep `L_m` 影响 NDIR、`T/P/RH` helper 可复现，这部分对照得很硬。
- `convert_hitran_coeff_to_per_percent_m` 的物理推导（HITRAN cm²/molec → 1% 浓度 1 m 路径的光学深度）数学上是对的，注释充分。
- frozen dataclass + slots 大面积使用，配置不可变。

---

## 建议的修复顺序

1. **先修 M1 + M2**：dl 通路的 lazy 失效和 epsilon 不一致 —— 阻塞下一阶段大 benchmark 训练。
2. **接着 M3 + M4 + M5**：清掉 HITRAN 主线下的死代码、重复计算和非功能参数，trainer 接入前把 sim 链路收敛干净。
3. **再处理 M6**：把 `configs/data/spectral-defaults.json` 要么改成 source-of-truth，要么明确为"代码导出快照"。
4. Minor 与 N 类按机会修。N7（R² 口径）建议在接 trainer 之前先定。
