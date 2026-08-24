# R5' TabPFN 观测特征回归 实施计划

> 状态：正式 6000 完成（2026-07-09），判据通过。产物 `outputs/tv3_r5/tabpfn_observed/metrics.json`
> 日期：2026-07-09
> 依据：[掘进通风项目记忆库.md](../legacy/掘进通风项目记忆库.md) §6.4 D0 结论、§5.4 Rocket 分支、§8.4 路线图；D2 证伪结论（raw waveform 端到端封闭）；TabPFN 官方文档（context7 `/priorlabs/tabpfn`）与 Hollmann et al. 2025 *Nature* 638:319。

## 结论

R5' 把 D0 已验证的 **D0-observed** 观测特征（排除 oracle 特征的 864 维 physics_stats）从线性 **RidgeCV** 回归头换成 **TabPFN**（Tabular Prior-data Fitted Network），作为"observed 标量空间非线性上限"的低成本预检。定位是**决定 D1 生死的判据实验，不是终极部署方案**：

- 若 TabPFN 相对 D0-observed（val O₂ R²=0.4226）无明显增益 → 坐实 observed 已到极限，D1（PatchTST/iTransformer 序列模型）正式关闭。
- 若有明显增益 → observed 空间存在非线性空间，D1/物理约束路线才有依据。**正式 6000 结果：判据通过（val O₂ +0.245）**。

真正把 O₂ 推向商用 0.2% 精度要靠新增 O₂ 光学通道（760nm A-band，仿真层改动），不是换回归头。R5' 的产物是判据，不是达标手段。

## 背景事实

- **D0-observed 是可部署真实基线**（记忆库 §6.4）：排除 true sound speed / true alpha / true TOF 等 oracle 特征，val O₂ R²=0.4226、test 0.4571、extrap 0.3708。
- **端到端 raw waveform 两条路已封闭**：R1b（MiniRocket raw 波形）与 D2（可微 TOF-PhaseNet）均证伪。
- **TabPFN 容量**（context7 `/priorlabs/tabpfn`，TabPFN-3）：支持 100k 行 × 2000 特征。tv3 规模 6000 行 × 864 特征完整落在包络内，**无需降维**；样本 6000 在"小样本"甜区（论文在 ≤10000 样本超越所有基线）。
- **TabPFN 最强场景**：回归 + OOD（Chen et al. 2026 *JCIM*），对应 tv3 的 extrapolation split 与 O₂ 弱信号。
- **物理墙不变**：O₂ 窄区间辨识是物理极限（D0 oracle o2_bins 全负）。TabPFN 换的是拟合能力，不是可观测性。

## 不变量

1. **特征口径与 D0-observed 逐位一致**：任何 oracle 特征混入都会重演 R0 的 oracle 膨胀，使判据失效。接入前核查无污染。
2. **禁止外部 StandardScaler**：TabPFN 官方明确建议避免外部缩放/one-hot，内部自带鲁棒预处理。这与现有 Ridge head 内嵌 scaler 的写法相反，是最易踩错的点。
3. **多输出必须 per-target 拆分**：TabPFN 原生单输出，CO₂/O₂/N₂ 各训一个回归器。三组分独立训练不违反"模型层不用闭包残差头"约束（数据层仍 sum=100%）。
4. **同 split 同分箱对比**：train/val/test/extrapolation 与 o2_bins/co2_bins 分箱口径与 D0-observed 完全一致，逐格对比。
5. **判据阈值沿用 §6.4**：O₂ 相对 D0-observed 提升是否超过 +0.05。

## 当前工程切入点

| 现有入口 | 说明 |
| --- | --- |
| `tv3/ml/rocket_training.py` `_build_head()` | 唯一 head 扩展点，当前支持 `ridgecv` / `ridge_closed_form` |
| `tv3/ml/rocket_training.py` `_model_diagnostics()` | 依赖 Ridge 的 `coef_`，TabPFN 无线性系数，需打补丁分支 |
| `tv3/ml/rocket_training.py` `train_tv3_rocket_regressor()` | 单一训练入口，特征已走缓存 + evaluate_regressor |
| `tv3/pipeline/run_tv3_rocket_baseline.py` | CLI，`--config` 驱动，复用即可 |
| `configs/tv3_d0_observed_ridge.json` | R5' 特征口径的对照母本 |

## 最小实现闭环

1. 新增 `_TabPFNMultiRegressor` 包装类（符合 `.fit(x, y, feature_names)` / `.predict(x)` 契约，内部 per-target 拆分，不套 scaler）。
2. `_build_head` 增加 `"tabpfn"` 分支，透传 `device`。
3. `_model_diagnostics` 增加 `head == "tabpfn"` 分支，返回 `None` 或 permutation importance，避免访问 `coef_`。
4. 新增 `configs/tv3_r5_tabpfn.json`，与 D0-observed 同特征口径，仅换 head。
5. 服务器 GPU 运行，逐格对比 D0-observed，回填记忆库。

## 分阶段实施步骤

### R5'-0：环境与版本核查（前置）

TabPFN 不是基础运行依赖。运行 `head="tabpfn"` 前必须安装项目的 `tabpfn` extra；Ridge 与 MLP head 不需要该依赖。

```bash
pip install -e ".[tabpfn]"
pip show tabpfn          # 确认版本:TabPFN-3(100k×2000)还是旧 v2(约 500 特征/10000 样本上限)
python -c "import torch; print(torch.cuda.is_available())"
```

- 若实装为旧 v2（特征上限低于 864）：按 D0 特征重要性选 top-200，或开 `ignore_pretraining_limits=True`。**先确认版本再决定，不要盲开该开关。**
- 离线服务器需提前预置 checkpoint 权重（首次 fit 触发下载）。

### R5'-1：实现 TabPFN 回归头

在 `tv3/ml/rocket_training.py` 新增（延迟导入，避免未装 tabpfn 时影响其他 head）：

```python
import numpy as np


class _TabPFNMultiRegressor:
    """TabPFN 多输出回归头。原生单输出,按标签列拆分 per-target 回归器。
    关键差异:不套 StandardScaler(TabPFN 内部处理)。"""

    def __init__(self, *, device: str = "cuda", ignore_pretraining_limits: bool = False):
        from tabpfn import TabPFNRegressor
        self._make = lambda: TabPFNRegressor(
            device=device, ignore_pretraining_limits=ignore_pretraining_limits,
        )
        self._models: list = []

    def fit(self, x: np.ndarray, y: np.ndarray, *, feature_names=None) -> "_TabPFNMultiRegressor":
        y = np.asarray(y)
        if y.ndim == 1:
            y = y[:, None]
        self._models = [self._make() for _ in range(y.shape[1])]
        for col, model in enumerate(self._models):
            model.fit(x, y[:, col])   # 直接喂原始 physics_stats,不缩放
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(x) for m in self._models])
```

### R5'-2：注册 head 与 diagnostics 补丁

```python
def _build_head(head, *, ridge_alphas, closed_form_alpha, device="cuda"):
    if head == "ridgecv":
        return _ScaledRidgeCVRegressor(alphas=ridge_alphas)
    if head == "ridge_closed_form":
        return _ScaledClosedFormRidgeRegressor(alpha=closed_form_alpha)
    if head == "tabpfn":
        return _TabPFNMultiRegressor(device=device)
    raise ValueError(f"unsupported rocket head {head!r}. available=('ridgecv', 'ridge_closed_form', 'tabpfn')")
```

`_model_diagnostics` 增加 `head == "tabpfn"` 分支：返回 `None`（或 permutation importance），不访问 `coef_`。

### R5'-3：配置文件

`configs/tv3_r5_tabpfn.json`，`physics_arrays` 与 `configs/tv3_d0_observed_ridge.json` 完全一致，`head="tabpfn"`、`device="cuda"`、`eval_splits=["val","test","extrapolation"]`、`output_dir="outputs/tv3_r5/tabpfn_observed"`。

### R5'-4：运行与对比

```bash
python -m tv3.pipeline.run_tv3_rocket_baseline --config configs/tv3_r5_tabpfn.json
```

对 `metrics.json` 逐格对比 D0-observed（val/test/extrap × CO₂/O₂/N₂ + o2_bins），按 §6.4 阈值出结论。

### R5'-5：回填

结果回填记忆库 §5.4 Rocket 表与 §8.4 路线图；判读写入 §6 实验结果新增小节。

## 影响文件清单

| 文件 | 改动 |
| --- | --- |
| `tv3/ml/rocket_training.py` | 新增 `_TabPFNMultiRegressor`；`_build_head` / `_model_diagnostics` 加 tabpfn 分支 |
| `configs/tv3_r5_tabpfn.json` | 新增，与 D0-observed 同特征口径 |
| `pyproject.toml` / 环境 | 新增 `tabpfn` 依赖（可选 extras，避免污染主线） |
| `outputs/tv3_r5/tabpfn_observed/` | 训练产物 |
| `tests/test_tunnel_ventilation_rocket_*.py` | 新增 tabpfn head smoke（可选，mock 或跳过 GPU） |

## 验收标准

| 项 | 标准 |
| --- | --- |
| 工程验收 | tabpfn head 跑通，metrics.json 产出完整；现有 ridge head 与 tv3 测试零回归 |
| 特征验收 | 864 维 observed 特征与 D0-observed 逐位对齐，无 oracle 污染 |
| 判据产出 | val/test/extrap 三 split O₂ R² 与 D0-observed 逐格对比，给出"有增益/无增益"结论 |

## 停止条件

1. 实装 TabPFN 版本特征上限低于 864 且开 `ignore_pretraining_limits=True` 后精度明显劣化：改为 D0 top-200 特征子集，不硬撑全维。
2. TabPFN 三 split O₂ R² 均 ≤ D0-observed + 0.05：判定 observed 到极限，关闭 D1，资源转 O₂ 光学通道仿真层改造。
3. 仅 val 提升、test/extrap 不提升：视为过拟合，不通过。

## 暂不做

- 不引入 TabPFN fine-tuning / 生成式用法（超出预检范围）。
- 不改特征口径去凑 O₂（那是 oracle 违规）。
- 不把 R5' 当部署模型；O₂ 达标靠新增光学氧通道，另行规划。
- 不动 D2/D1 端到端代码（已封闭/暂缓）。

## 预期结果（推断 vs 实测）

CO₂ 已 0.988 无大幅提升空间；N₂ 0.88 有小幅改善；O₂ 是唯一看点。

| 指标 | 推断（实施前） | 实测（正式 6000） |
| --- | --- | --- |
| val O₂ R² | 0.43–0.50 | **0.6673**（超预期） |
| vs D0 +0.05 判据 | 难破 0.70 验收线 | **+0.245，判据通过**；仍差验收线 0.033 |
| o2_bins | 仍全负 | 仍全负，但负值得分优于 D0 Ridge |

R5' 输出的是"observed 空间上限判据"，不是达标模型。

## 关键文献

- Hollmann et al. 2025, *Nature* 638:319 — TabPFN，≤10000 样本表格回归超越所有方法。
- Chen et al. 2026, *JCIM* — TabPFN 在小样本 + OOD 回归上优势最强、退化平缓。
- context7 `/priorlabs/tabpfn` — TabPFN-3 容量 100k×2000，禁止外部缩放，per-target 单输出。

## 实施记录（2026-07-09）

### R5'-0 环境核查

- TabPFN 8.0.8（TabPFN-3），远超旧 v2 500 特征上限，864 维特征完整落在 100k×2000 包络内，无需 `ignore_pretraining_limits`
- CUDA 可用：NVIDIA GeForce RTX 5060 Laptop GPU
- 需 PriorLabs 许可证认证，通过 `TABPFN_TOKEN` 环境变量设置 API Key

### R5'-1/2 代码落地

修改文件：

| 文件 | 改动 |
|------|------|
| `tv3/ml/rocket_training.py` | 新增 `_TabPFNMultiRegressor`（per-target 拆分，不套 StandardScaler）；`_build_head` 增加 `tabpfn` 分支与 `device` 参数；`_model_diagnostics` 增加 `tabpfn` 分支返回占位 note；`train_tv3_rocket_regressor` 透传 `device` |
| `tv3/pipeline/run_tv3_rocket_baseline.py` | `--head` choices 增加 `tabpfn`；新增 `--device` 参数（默认 `"auto"`）；DEFAULT_CONFIG 新增 `device` 字段 |

### R5'-3 配置文件

`configs/tv3_r5_tabpfn.json`：与 D0-observed 逐位一致（`physics_arrays` 仅 observed 五项，`feature_builder="d0_observed_physics_stats_v1"`），`head="tabpfn"`，`device="cuda"`。

### 回归测试

全量 154 passed，0 failed。Ridge 路径零回归。

### R5'-4 本地 Smoke Test

使用 `data/tv3-formal`（600 序列）验证端到端链路：

| Split | O₂ R² | CO₂ R² | N₂ R² |
|-------|-------|--------|-------|
| train | 0.9986 | 0.9999 | 0.9894 |
| val | 0.4544 | 0.9988 | 0.9208 |
| test | 0.3167 | 0.9983 | 0.8718 |
| extrapolation | 0.5708 | 0.9978 | 0.9126 |

**判读**：

1. 严重过拟合：train O₂ R²=0.9986 → val 0.4544，gap 0.54。600 序列对 TabPFN 太少（TabPFN 论文甜区 ≤10000，600 在甜区下沿以下）。
2. o2_bins 全部为负（val −18.96 ~ −1.13），与 D0 物理墙一致——TabPFN 也只捕获粗粒度 O₂ 区间差异。
3. CO₂ val 0.9988 > D0-observed 0.9878（6000 seq），N₂ 0.9208 > 0.8799，但小样本过拟合使 test 端不可靠。
4. **本地 600 序列结果不可直接对比 D0-observed（6000 序列）**。正式 6000 已验证：val O₂ 0.6673，过拟合 gap 0.314 但仍大幅领先 D0。

### R5'-4 正式 6000 运行（2026-07-09，tv3-formal-6000）

```bash
export TABPFN_TOKEN="<api-key>"
cd tunnel_ventilation && source .venv/bin/activate
python -m tv3.pipeline.run_tv3_rocket_baseline --config configs/tv3_r5_tabpfn.json
```

产物：`outputs/tv3_r5/tabpfn_observed/metrics.json`。

**三组分 R²（vs D0-observed Ridge）**：

| Split | CO₂ | O₂ | N₂ |
|-------|:---:|:--:|:--:|
| val | 0.9995 (+0.012) | **0.6673 (+0.245)** | 0.9242 (+0.044) |
| test | 0.9994 (+0.010) | **0.6594 (+0.202)** | 0.9228 (+0.034) |
| extrap | 0.9993 (+0.011) | **0.6279 (+0.257)** | 0.9194 (+0.040) |

**过拟合**：train O₂ 0.981 → val 0.667（gap 0.314），高于 Ridge gap 0.078，但三 eval split 均大幅领先。

**o2_bins（val）**：四 bin 仍全负（−6.19 ~ −2.42），物理墙未破；各 bin 优于 D0 Ridge（−14.00 ~ −3.71）。

**闭包**：val `sum_abs_error`≈0.157（TabPFN）vs ≈3.4×10⁻⁸（Ridge）。不可部署。

**判据结论**：

1. O₂ val 提升 +0.245 >> +0.05 → **判据通过**，observed 非线性有空间。
2. test/extrap 同步提升 → 非仅 val 过拟合。
3. TabPFN 是上限探针，非部署模型；可部署 baseline 仍为 D0-observed Ridge。
4. 下一步曾为 R5（小 MLP）；R5 正式 6000 已跑完且判据未通过（见 `r5_mlp_implementation_plan.md` / 记忆库 §6.9）。当前 P0 转为 O₂ 光学通道评估。

已回填记忆库 §5.4 / §6.8 / §8.4。
