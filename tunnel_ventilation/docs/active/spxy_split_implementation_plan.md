# SPXY 与 OOD 数据集划分实施规划

> 正文已按附录 A/B 的调研与审查结论修订：SPXY 保留为训练覆盖候选方法，但不再单独承担 extrapolation 构造或最终性能估计；extrapolation 必须由独立的 OOD 规则产生；X/Y 特征必须先标准化；所有 SPXY 实现必须向量化并通过划分诊断验证。

## 1. 背景与动机

### 1.1 当前划分机制的问题

当前 tv3 数据集划分 (`tv3/sim/packaging/splits.py`) 为纯随机 shuffle：

1. 提取所有 `mixture_id` → `random.Random(seed).shuffle()` → 按 70/15/10/5% 比例切片
2. tv3 中 `mixture_id` 与 `sequence_id` **一一对应**（`M000001` ≈ `Q000001`），分组无实际效果
3. LHS 采样实现的 space-filling 被 shuffle 破坏，各 split 内部不再均匀覆盖成分空间
4. extrapolation split（30 条）仅为 shuffle 末尾剩余，不基于成分极值选取

### 1.2 为什么保留 SPXY，但不单独依赖 SPXY

| 方法 | X 空间覆盖 | Y 空间覆盖 | OOD/extrapolation 构造 | DL 直接证据 | 正文定位 |
| ------------------ |:------:|:------:|:-------------------:|:---------:| -------- |
| 纯随机 | ❌ | ❌ | ❌ | ✅ | baseline，保留现有 `random_mixture_id_split_v4` |
| LHS/Y 分箱分层随机 | ⚠️ | ✅ | ⚠️ | 间接 | 必须加入的简单对照，避免只比较 random vs SPXY |
| **SPXY** | **✅** | **✅** | ❌ | 弱 | ID 训练集覆盖候选，不用于反向选择 extrapolation |
| K-means / Group-aware | ✅ | ⚠️ | ✅ | 较强 | extrapolation/OOD 候选，按边界簇或未见 LHS 子区域留出 |
| XYOnion | ✅ | ✅ | ✅ | 弱 | 机制上最干净的候选，但需自实现与交叉验证 |
| Repeated stratified K-fold | ⚠️ | ✅ | ❌ | 较强 | 方差量化工具，不替代 OOD 评估 |

SPXY 是 KS 的扩展，同时在 X 空间（慢通道 + 超声波形特征）和 Y 空间（CO₂/O₂/N₂ 浓度）上执行 max-min 距离选择。这对 tv3 的 ID 训练集覆盖仍有价值：O₂/N₂ 物理可辨识性弱（声速差 ~6.4%，热导差 ~2%），训练集需要覆盖目标工作域内的 Y 空间。

但附录 B 指出两个边界必须写入正文不变量：

1. **SPXY 不能反向用来挑 extrapolation 极值**。SPXY 的目标是把代表性和边界样本放进训练集；把这些点留给 extrapolation 会破坏训练覆盖，也与 SPXY 设计初衷相反。
2. **SPXY 不能单独作为 DL 性能估计依据**。Xu & Goodacre 2018 指出 KS/SPXY 类系统采样可能导致性能估计偏差；tv3 的 TCN 对 SPXY 的适用性应作为迁移验证，而不是已证实前提。
## 2. 算法原理与使用边界

### 2.1 数学公式

**参考**: Galvão et al., Talanta 67, 736–740 (2005).  被引 880+ 次.

给定 $N$ 个样本，原始特征矩阵 $X \in \mathbb{R}^{N \times P}$，标签矩阵 $Y \in \mathbb{R}^{N \times K}$。距离计算前必须先逐维标准化：

$$\hat{X}=\operatorname{scale}(X), \quad \hat{Y}=\operatorname{scale}(Y)$$

$$d_{ij}^X = \sqrt{\sum_{p=1}^{P} (\hat{x}_{ip} - \hat{x}_{jp})^2}$$

$$d_{ij}^Y = \sqrt{\sum_{k=1}^{K} (\hat{y}_{ik} - \hat{y}_{jk})^2}$$

$$d_{ij} = \alpha \frac{d_{ij}^X}{\max_{p,q} d_{pq}^X} + (1 - \alpha) \frac{d_{ij}^Y}{\max_{p,q} d_{pq}^Y}, \quad \alpha \in [0, 1]$$

这里的 `scale` 默认使用 `StandardScaler`。对 X 只做 `d_x / d_x_max` 不够，因为它只缩放整个距离矩阵，不能消除 `tof_s`、`sound_speed`、慢通道统计量之间跨数量级的维度支配问题。Y 默认使用 CO₂/O₂ 两个自由度；N₂ 由闭包决定，ILR/CLR 成分几何可作为后续优化项，但不进入第一版必做范围。

### 2.2 SPXY 选择流程

```
1. 计算标准化后的 X/Y pairwise 距离矩阵
2. 合成加权联合距离 D_xy
3. 选择 D_xy 中距离最大的两个样本 → 训练候选集
4. while 训练候选集 < 目标大小:
     a. 对每个剩余样本，维护其到已选训练样本的最小距离
     b. 选择最小距离最大的样本加入训练候选集
5. 剩余样本仅表示“未被 SPXY 优先选中”，不能直接解释为代表性 val/test 或 extrapolation
```

SPXY 是确定性算法，结果仅取决于输入特征、目标比例和权重参数。若 val/test 也用递归 SPXY 从剩余集合继续切分，剩余集代表性会逐层下降，因此正文不再采用递归 SPXY 四分类。

### 2.3 加权 SPXY 变体

**参考**: Tian et al., Infrared Physics & Technology 95, 88–92 (2018).  被引 81 次.

- $\alpha = 1.0$：退化为 KS（仅 X 空间）
- $\alpha = 0.5$：标准 SPXY（X/Y 等权重）
- $\alpha = 0.0$：仅 Y 空间（浓度驱动选择）

Tian et al. 报告在 NIR 组分分析中，$\alpha=0.3$–$0.5$ 时 PLS 模型预测效果最优。对 tv3，$\alpha=0.5$ 仅作为主实验默认值，$\alpha=0.3/0.7$ 用于敏感性验证；不能预设 SPXY 会提升 TCN 的 test R²。

### 2.4 正文采纳的不变量

1. `spxy_v1` 只在 ID pool 内选择 train，不参与 extrapolation 选择。
2. extrapolation 由 `y_margin_ood`、`lhs_boundary` 或 `kmeans_boundary` 等独立 OOD 规则产生，并在 `split_policy` 中记录。
3. ID test 用于性能估计；extrapolation 只用于外推压力测试，不参与模型选择或早停。
4. 每个 split 必须输出诊断：样本数、CO₂/O₂/N₂ 范围、Y 覆盖率、到 train 凸包或最近邻的距离分布、X/Y pairwise 距离摘要。
5. 若 OOD 选择规则退化（例如所有候选点到凸包距离为 0），必须显式失败并更换策略，不做静默回退。

## 3. 修订后的划分策略

### 3.1 策略分工

附录 A/B 的结论是：tv3 的两个需求不能交给单一 SPXY 同时解决。

| 需求 | 推荐方法 | 说明 |
| ---- | -------- | ---- |
| ID 训练覆盖 | SPXY 或 LHS/Y 分箱分层随机 | 保证 train 在目标工作域内覆盖 X/Y；SPXY 是候选，不是唯一方案 |
| 外推/OOD 构造 | Y 边界距离、LHS 边界组、K-means 边界簇、XYOnion 外壳 | 必须独立于 SPXY max-min；外推集不是“SPXY 剩余” |
| 性能估计稳定性 | repeated stratified K-fold | 用于报告 mean±std，回应单次 hold-out 方差问题 |
| 方法学对照 | random、LHS stratified、SPXY、K-means OOD、XYOnion smoke | 避免正文只证明“SPXY 优于 random” |

### 3.2 默认落地方案

第一版正式实现采用 `spxy_v1 + y_margin_ood`：

1. 从全量样本中按预声明的 Y/LHS 边界规则选出 5% extrapolation。该步骤不调用 SPXY。
2. 在剩余 ID pool 内构建标准化 X/Y，使用向量化 SPXY 选择 70% train。
3. 对 SPXY 剩余的 ID 样本，按 CO₂/O₂ 分箱做 stratified random，得到 15% val 和 10% test。
4. 同时实现 `lhs_stratified_split_v1` 作为简单对照；`kmeans_boundary` 和 `xyonion_v1` 先作为实验候选，不阻塞 `spxy_v1` 落地。

这个设计承认一个重要取舍：若 extrapolation 是真实 OOD，train 就不可能同时覆盖全量 Y 边界。因此正文只要求 train 覆盖 **ID 工作域**，并单独报告被留出的 OOD 区域范围。

### 3.3 不采用的旧方案

旧正文的“递归 SPXY 四分类 + 用低 α SPXY 反向选择 extrapolation”不再采用，原因有三点：

1. 反向 SPXY 不是极值选择器，概念上与训练覆盖目标冲突。
2. 递归 SPXY 会让 val/test 代表性逐层下降，正好触发 Xu & Goodacre 2018 的偏差警告。
3. N=6000 时旧 `_spxy_split` 的纯 Python 嵌套循环会变成主要耗时瓶颈。
## 4. tv3 实现方案

### 4.1 特征空间 X 的选取

tv3 的完整输入是高维时序数据，不适合直接用作 pairwise 距离的 X。需先做特征提取，再统一标准化。

**方案 A（推荐）：使用慢通道 + 超声波形统计特征**

对每条 sequence（512 timesteps），提取：

- 慢通道的时序统计：mean、std、min、max、trend（每通道 5 个统计量 × 7 通道 = 35 维）
- 超声波形的时序统计：tof_s 的 mean/std/trend、sound_speed 的 mean/std、alpha 的 mean/std（约 10 维）
- phase 分段统计：steady/baseline/transition 各段的慢通道均值

总维度约 50–100，远小于原始 5000 点波形。这种聚合特征的 pairwise 距离在物理上有意义（表示两个 sequence 的整体工况差异）。聚合后的 X 必须执行 `StandardScaler` 或 `RobustScaler`，否则声速、慢通道均值等大量级特征会支配距离。

**方案 B（备选）：对所有慢通道 timestep 做 PCA 降维**

对 (N, 512, 7) reshape 为 (N, 3584)，PCA 降维到约 50 维。但损失时序结构信息，且仍需标准化。

**方案 C（备选）：仅用 Y 空间分层或 SPXY α=0**

如果发现 X 空间距离与组分变化弱相关，可退化为纯 Y 空间选择。但这应作为显式实验策略记录在 `split_policy`，不能混入默认 SPXY。

### 4.2 X/Y 预处理

```python
from sklearn.preprocessing import StandardScaler

X_raw = _build_spxy_features(conditions, arrays)
X_scaled = StandardScaler().fit_transform(X_raw)

# tv3 的 CO2 + O2 + N2 = 100%，第一版用两个自由度计算 Y 距离。
y_basis = labels[:, [0, 1]]  # CO2, O2
y_scaled = StandardScaler().fit_transform(y_basis)
```

`N2` 仍保留在 split rows 与诊断摘要中，但不作为第一版 SPXY 欧氏距离的独立维度。若后续引入 ILR/CLR，需要在 split summary 中记录 `y_geometry="ilr"` 或 `"clr"`，并保持模型训练 target 不变。

### 4.3 四分类划分策略

```
全量 N 条
  │
  ├─ OOD selector: y_margin_ood / lhs_boundary / kmeans_boundary
  │     └── extrapolation (N×0.05)
  │
  └── ID pool (N×0.95)
        │
        ├─ 向量化 SPXY 选 train (N×0.70)
        │
        └─ 对剩余 ID 样本按 CO2/O2 分箱 stratified random
              ├── val  (N×0.15)
              └── test (N×0.10)
```

extrapolation 默认使用 `y_margin_ood`：按 CO₂/O₂ 分位数或 LHS 网格先定义 interior domain 与 boundary candidates，再计算候选点到 interior domain 凸包的距离，选择距离最大的 5%。若凸包距离全部为 0，说明该规则无法在当前采样上产生外推集，生成过程应报错并要求显式切换到 `lhs_boundary` 或 `kmeans_boundary`，不能静默回退为随机剩余。

val/test 不再由递归 SPXY 产生，而是从 ID 剩余样本做 Y 分箱分层随机。这样保留 seed 可重复实验，也避免把 SPXY 剩余集直接解释为代表性性能估计集。

### 4.4 代码结构

新增文件 `tv3/sim/packaging/spxy_split.py`，与现有 `splits.py` 并存：

```python
# tv3/sim/packaging/spxy_split.py

def build_spxy_split_rows(
    conditions: list[dict[str, str]],
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    extrapolation_ratio: float = 0.05,
    alpha: float = 0.5,
    extrapolation_strategy: str = "y_margin_ood",
) -> dict[str, list[dict[str, str]]]:
    """构建 tv3 的 SPXY+OOD 四分类 split rows。

    不变量：
    - SPXY 只在 ID pool 内选择 train。
    - extrapolation 由独立 OOD selector 产生。
    - val/test 从 ID 剩余样本按 Y 分箱随机划分。
    """
    X_scaled, y_scaled, y_basis = _build_scaled_split_features(conditions, arrays, labels)

    ext_idx = _select_extrapolation_indices(
        y_basis=y_basis,
        X_scaled=X_scaled,
        ratio=extrapolation_ratio,
        strategy=extrapolation_strategy,
        seed=seed,
    )

    id_idx = np.setdiff1d(np.arange(len(conditions)), ext_idx, assume_unique=False)
    train_size = int(len(conditions) * train_ratio)
    train_local_idx, id_remainder_local_idx = _spxy_select_train(
        X_scaled[id_idx], y_scaled[id_idx], train_size=train_size, alpha=alpha
    )

    train_idx = id_idx[train_local_idx]
    id_remainder_idx = id_idx[id_remainder_local_idx]

    val_idx, test_idx = _stratified_val_test_split(
        indices=id_remainder_idx,
        y_basis=y_basis[id_remainder_idx],
        val_size=int(len(conditions) * val_ratio),
        test_size=int(len(conditions) * test_ratio),
        seed=seed,
    )

    return _build_split_rows_from_indices(conditions, train_idx, val_idx, test_idx, ext_idx)
```

SPXY 核心选择必须用增量向量化写法，不能使用旧正文中的 Python 嵌套 `min(...)`：

```python
def _spxy_select_train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_size: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.distance import pdist, squareform

    d_x = squareform(pdist(X, metric="euclidean"))
    d_y = squareform(pdist(y, metric="euclidean"))
    d_x = _normalize_distance_matrix(d_x, name="X")
    d_y = _normalize_distance_matrix(d_y, name="Y")
    d_xy = alpha * d_x + (1.0 - alpha) * d_y

    i, j = np.unravel_index(np.argmax(d_xy), d_xy.shape)
    selected = [int(i), int(j)]
    selected_mask = np.zeros(len(X), dtype=bool)
    selected_mask[[i, j]] = True
    min_dist_to_selected = np.minimum(d_xy[:, i], d_xy[:, j])
    min_dist_to_selected[selected_mask] = -np.inf

    while len(selected) < train_size:
        sel = int(np.argmax(min_dist_to_selected))
        selected.append(sel)
        selected_mask[sel] = True
        min_dist_to_selected = np.minimum(min_dist_to_selected, d_xy[:, sel])
        min_dist_to_selected[selected_mask] = -np.inf

    train_idx = np.array(selected, dtype=int)
    remainder_idx = np.flatnonzero(~selected_mask)
    return train_idx, remainder_idx
```

### 4.5 与现有 benchmark 集成

当前 `benchmark.py` 在 arrays 生成前调用 `build_default_split_rows`。SPXY 需要 `arrays["slow"]` 和 `arrays["ultrasonic_*"]`，因此 split 调用必须后移到 `_build_sequence_arrays_for_spec` 之后。

实际改动点：

1. `TunnelVentilationBenchmarkGenerationSpec` 增加字段：`split_strategy: str = "random"`、`spxy_alpha: float = 0.5`、`extrapolation_strategy: str = "none"`。
2. CLI `tv3/pipeline/generate_tunnel_ventilation_benchmark.py` 增加 `--split-strategy`、`--spxy-alpha`、`--extrapolation-strategy`。
3. `benchmark.py` 的生成顺序调整为：conditions → arrays → labels → split rows → validation → scalers。
4. `_split_summary` 的 `split_policy` 必须区分 `random_mixture_id_split_v4`、`spxy_v1:y_margin_ood`、`lhs_stratified_split_v1` 等来源。
5. `validate_benchmark_assets` 与 DL 侧读取接口保持 split CSV 格式不变。

```python
arrays = _build_sequence_arrays_for_spec(...)
labels = _labels_from_conditions(conditions)

if spec.split_strategy == "spxy_v1":
    from sim.packaging.spxy_split import build_spxy_split_rows
    split_rows = build_spxy_split_rows(
        conditions,
        arrays,
        labels,
        seed=spec.seed,
        alpha=spec.spxy_alpha,
        extrapolation_strategy=spec.extrapolation_strategy,
    )
elif spec.split_strategy == "lhs_stratified_split_v1":
    split_rows = build_lhs_stratified_split_rows(conditions, labels, seed=spec.seed)
else:
    split_rows = build_default_split_rows(conditions, seed=spec.seed)
```
## 5. 实验设计

### 5.1 验证目标

| 指标 | 当前 random split | 修订方案目标 |
| ---------------------- |:---------------:| -------- |
| X/Y 标准化正确性 | 无 | X 各维均值/方差受控，距离不被单一大量级特征支配 |
| ID train 覆盖率 | 不一定 | 覆盖 ID 工作域内 CO₂/O₂/N₂ 范围，并报告被 OOD 留出的边界 |
| val/test 代表性 | 随机波动 | Y 分箱分布接近 ID pool，不使用 SPXY 剩余集直接充当性能估计 |
| extrapolation 外推性 | 无定义 | 与 train 在 Y 边界、LHS 子区域或聚类簇上有可量化距离 |
| split 可追溯性 | 单一 `random_mixture_id_split_v4` | `split_policy` 明确记录策略、α、OOD 规则、seed |
| DL 适用性 | 未验证 | TCN/Ridge 对比用于探索 SPXY 迁移有效性，不预设 SPXY 必然提升 |

### 5.2 对比实验矩阵

在 `tv3-formal`（600 序列）上运行：

| 实验 ID | Split 策略 | 模型 | 用途 |
| ----- | ----------------- | -------------- | ---------- |
| A1 | random (baseline) | TCN multimodal | 现有基线 |
| A2 | LHS/Y stratified random | TCN multimodal | 简单分层对照，检验是否无需 SPXY |
| A3 | `spxy_v1` α=0.5 + `y_margin_ood` | TCN multimodal | 主候选方案 |
| A4 | `spxy_v1` α=0.3 + `y_margin_ood` | TCN multimodal | Y 权重敏感性 |
| A5 | K-means boundary OOD | TCN multimodal | DL 文献支持更强的 OOD 对照 |
| A6 | XYOnion smoke | TCN multimodal | 新方法机制验证，非第一版阻塞项 |
| B1 | random | Ridge | 线性基线对照 |
| B2 | `spxy_v1` α=0.5 + `y_margin_ood` | Ridge | 线性基线 SPXY |
| C1 | repeated stratified K-fold | TCN multimodal | 估计划分方差，报告 mean±std |

预期调整：不再写“SPXY 下 CO₂/O₂ test R² 应至少不低于 random”。正确验收是先看 split 诊断是否达标，再比较 TCN 与 Ridge 的 ID test 和 extrapolation 指标。如果 SPXY 提升不明显，但 LHS stratified 或 K-means OOD 更稳定，应优先保留更简单、证据更强的方案。

### 5.3 回归测试

```bash
# 新增单元测试
pytest tests/test_spxy_split.py -v

# 确保 tv3 benchmark 生成仍可通过（random 与 spxy_v1）
python -m tv3.pipeline.generate_tunnel_ventilation_benchmark \
    --dataset tv3-spxy-test --sequences 32 --workers 1 \
    --split-strategy spxy_v1 \
    --extrapolation-strategy y_margin_ood

# 全量回归
pytest tests/ -x --ignore=tests/test_rcdw_mgda.py
```

新增测试至少覆盖：

1. X/Y scaler 在距离计算前生效。
2. `_spxy_select_train` 结果确定、无重复、比例正确。
3. 向量化实现与小 N 朴素实现结果一致。
4. extrapolation selector 不调用 SPXY，且能输出非零 OOD 诊断；退化时显式失败。
5. train/val/test/extrapolation 四集合互斥且总数守恒。
6. `split_policy`、`spxy_alpha`、`extrapolation_strategy` 写入 summary。
7. 旧 `random_mixture_id_split_v4` 行为不变。

## 6. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
| ---------------------------- |:---:| --- | --------------------------------------------------- |
| X 特征未标准化导致距离被大量级特征支配 | 中 | 高 | scaler 写入必测项；诊断输出每维贡献或消融距离摘要 |
| SPXY 用于性能估计产生偏差 | 中 | 高 | ID test 使用分层随机；SPXY 只负责 train 覆盖；报告 repeated K-fold 方差 |
| extrapolation 规则与 train 覆盖目标冲突 | 高 | 中 | 明确区分 ID 覆盖和 OOD 留出范围；summary 同时报告两者 |
| `y_margin_ood` 在 LHS 上退化为零凸包距离 | 中 | 中 | 显式失败并要求切换 `lhs_boundary` 或 `kmeans_boundary`，不静默随机化 |
| N=6000 下 SPXY 计算慢 | 中 | 中 | 使用向量化增量最小距离；只保留 O(N²) 距离矩阵，不写 O(N²·train_size) Python 循环 |
| astartes API 未验证 | 中 | 低 | 第一版不依赖；仅作为交叉验证候选 |
| XYOnion 文献新、DL 证据弱 | 中 | 中 | 先 smoke 验证，不作为默认正式 split |
| 与现有 DL 训练流程兼容 | 低 | 高 | split CSV 格式不变；只扩展生成端 spec/CLI/summary |

## 7. 实施阶段

| 阶段 | 内容 | 预估工时 |
|:---:| ----------------------------------------------- |:---------:|
| 0 | 固化不变量：SPXY 不选 extrapolation、X/Y 必须标准化、val/test 不递归 SPXY | 0.5h |
| Ⅰ | 实现 `spxy_split.py`：特征构建、scaler、向量化 SPXY、split rows | 2h |
| Ⅱ | 实现 `y_margin_ood` 与 `lhs_stratified_split_v1`，补充 OOD 诊断 | 2h |
| Ⅲ | 集成 `benchmark.py` 调用时序、spec 字段、CLI 参数、`split_policy` summary | 1.5h |
| Ⅳ | 单元测试与 32-sequence smoke benchmark | 1.5h |
| Ⅴ | 跑 A1–A5、B1–B2 对比矩阵；A6/XYOnion 仅做 smoke | 4–6h（含 GPU 训练） |
| Ⅵ | 用 split 诊断 + ID/OOD 指标决定是否推广到 hg/sg 主线 | 1h |
## 8. 参考文献

1. **Galvão, R.K.H., Araujo, M.C.U., José, G.E., Pontes, M.J.C., Silva, E.C., Saldanha, T.C.B.** (2005). A method for calibration and validation subset partitioning. *Talanta*, 67(4), 736–740. DOI: [10.1016/j.talanta.2005.03.025](https://doi.org/10.1016/j.talanta.2005.03.025) — **原始 SPXY 论文，被引 880+**

2. **Tian, H., Zhang, L., Li, M., Wang, Y., Sheng, D., Liu, J., Wang, C.** (2018). Weighted SPXY method for calibration set selection for composition analysis based on near-infrared spectroscopy. *Infrared Physics & Technology*, 95, 88–92. DOI: [10.1016/j.infrared.2018.10.030](https://doi.org/10.1016/j.infrared.2018.10.030) — **加权 SPXY，被引 81**

3. **Xu, Y. & Goodacre, R.** (2018). On splitting training and validation set: A comparative study of cross-validation, bootstrap and systematic sampling for estimating the generalization performance of supervised learning. *Journal of Analysis and Testing*. DOI: [10.1007/s41664-018-0068-2](https://doi.org/10.1007/s41664-018-0068-2) — **高引批判性文献，指出 KS/SPXY 性能估计偏差风险**

4. **Apinantanakon et al.** (2019). M-SPXY for ANN. — **针对神经网络场景的 SPXY 改进方向，提示原版 SPXY 迁移到 ANN/DL 需验证**

5. **Ezenarro, J.** (2025). XYOnion: a layer-based method for splitting datasets into calibration and validation subsets. *Analytica Chimica Acta*, 1364, 344229. DOI: [10.1016/j.aca.2025.344229](https://doi.org/10.1016/j.aca.2025.344229) — **最新分层方法，结合 SPXY+Onion**

6. **Fooladi et al.** (2025). K-means clustering based OOD splitting for molecular ML. *Journal of Chemical Information and Modeling*. DOI: [10.1021/acs.jcim.5c00475](https://doi.org/10.1021/acs.jcim.5c00475) — **K-means OOD 对 GNN/ML 的直接证据**

7. **Tahir et al.** (2024). Source/group-aware partitioning for DL generalization evaluation. *Expert Systems with Applications*. DOI: [10.1007/s10489-024-05848-6](https://doi.org/10.1007/s10489-024-05848-6) — **Group-aware/LOGO 防泄漏证据**

8. **Calle et al.** (2025). NACHOS: Nested cross-validation and automated hyperparameter optimization for deep learning. *Computer Methods and Programs in Biomedicine*. DOI: [10.1016/j.cmpb.2025.109063](https://doi.org/10.1016/j.cmpb.2025.109063) — **DL 方差量化证据**

9. **Chen, W., Chen, H., Feng, Q., Mo, L., Hong, S.** (2021). A hybrid optimization method for sample partitioning in near-infrared analysis. *Spectrochimica Acta Part A*, 248, 119182. DOI: [10.1016/j.saa.2020.119182](https://doi.org/10.1016/j.saa.2020.119182) — **AHCTS 对比 KS/SPXY；正文修正年份与卷号**

10. **Kennard, R.W., Stone, L.A.** (1969). Computer Aided Design of Experiments. *Technometrics*, 11(1), 137–148. — **KS 原始论文，SPXY 的前身**

11. **astartes**: Python library with SPXY/KS implementation. [github.com/JacksonBurns/astartes](https://github.com/JacksonBurns/astartes) — **API 未独立验证，仅作为后续交叉验证候选**

---
# 附录 A：深度学习数据集划分方法调研与评估

> 调研日期：2026-07-06
> 触发原因：第 1–8 章主体方案引用的 SPXY 证据全部来自 PLS/MLR 校准场景，对 tv3 的 DL 模型（TCN）适用性证据不足。本附录正面调研适用于深度学习的划分方法，评估后选取 top 5，作为主体 SPXY 方案的对照与组合候选。
> 检索范围：Semantic Scholar、CrossRef、arXiv、PubMed（Scopus 鉴权失败）。三组并行子代理各调研 5 个方法，每组 4–5 轮查询。

## A.1 评估方法

评估维度（针对 tv3 场景，各 2 分共 10 分）：

| 维度 | 含义 | tv3 为何重要 |
|---|---|---|
| DL 回归文献 | 有无 DL/神经网络回归场景的直接文献 | SPXY 类方法多在 PLS 上验证，迁移 DL 需证据 |
| 多输出 Y | 天然支持三输出回归 | CO₂/O₂/N₂ 同时预测 |
| 外推集构造 | 能否天然选出 OOD 极值点 | tv3 明确要 extrapolation 集测外推 |
| Y 空间覆盖 | 保证训练集覆盖 Y 全范围 | O₂/N₂ 弱辨识，训练集必须覆盖全范围 |
| 实现成熟度 | 现成库、sklearn 支持、复杂度 | 600~6000 样本下的可行性与可维护性 |

## A.2 15 个候选方法评分总表

| 方法 | DL文献 | 多输出Y | 外推集 | Y覆盖 | 成熟度 | 总分 | 组别 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **SPXY** | 1 | 2 | 1 | 2 | 2 | **8** | 距离 |
| **K-means 聚类划分** | 2 | 1 | 2 | 0 | 2 | **7** | 聚类 |
| **XYOnion** | 0 | 2 | 2 | 2 | 1 | **7** | 距离 |
| **Group-aware/LOGO** | 2 | 2 | 1 | 0 | 2 | **7** | 时序/基线 |
| **重复分层 K-fold** | 2 | 1 | 0 | 2 | 2 | **7** | 聚类 |
| 分层随机(按Y分箱) | 1 | 1 | 0 | 2 | 2 | 6 | 聚类 |
| 分块时序 CV | 2 | 2 | 0 | 0 | 2 | 6 | 时序/基线 |
| 纯随机 baseline | 2 | 2 | 0 | 0 | 2 | 6 | 时序/基线 |
| KS (Kennard-Stone) | 2 | 0 | 1 | 0 | 2 | 5 | 距离 |
| UMAP/HDBSCAN | 1 | 1 | 2 | 0 | 1 | 5 | 聚类 |
| 主动学习式选取 | 1 | 1 | 1 | 0 | 1 | 4 | 时序/基线 |
| D-optimal | 0 | 0 | 1 | 0 | 0 | 1 | 距离 |
| Mahalanobis 距离 | 0 | 0 | 1 | 0 | 1 | 2 | 距离 |
| Butina 聚类 | 0 | 1 | 1 | 0 | 1 | 3 | 聚类 |
| 模型感知/难度感知 | 0 | 1 | 2 | 0 | 0 | 3 | 时序/基线 |

## A.3 Top 5 方法详细评估

### A.3.1 SPXY（8 分）

**原理**：X+Y 联合 max-min 距离选择，归一化后 `d = α·d_x/d_x_max + (1-α)·d_y/d_y_max`，迭代选距已选集合最远的点，保证 X 与 Y 空间同时覆盖。

**DL 文献**：
- Galvão 2005, Talanta, [10.1016/j.talanta.2005.03.025](https://doi.org/10.1016/j.talanta.2005.03.025)，880 引——原始 PLS 场景
- Wang & Cui 2025, Applied Sciences, [10.3390/app152211902](https://doi.org/10.3390/app152211902)，3 引——**SPXY+ResNet 高光谱分类**，少有的 SPXY+DL 直接组合
- **Xu & Goodacre 2018**, J. Analysis and Testing, [10.1007/s41664-018-0068-2](https://doi.org/10.1007/s41664-018-0068-2)，900+ 引——批评 SPXY 性能估计偏乐观（代表性样本进训练集，验证集代表性差）

**tv3 适用性**：
- **优势**：Y 空间 max-min 直接命中"训练集覆盖 O₂/N₂ 全范围"痛点；Y 作为向量参与距离，天然支持三输出；实现简单，有参考库。
- **风险**：Xu & Goodacre 的批评针对"用 SPXY 验证集做性能估计"。**可绕开**——tv3 用独立 extrapolation set 测外推，不依赖 SPXY 的 val/test 做性能估计。
- **主体方案已指出的实现缺陷必须修**：X 特征需 StandardScaler（聚合特征量级跨 9 个数量级）；`_spxy_split` 的 Python 循环需向量化（N=6000 会卡）。

**实施建议**：α=0.5 主划分；X 用慢通道+超声统计特征并 StandardScaler；外推集单独用 Y 极值选取（不用 SPXY 反向逻辑）。

### A.3.2 K-means 聚类划分（7 分）

**原理**：对 X 特征空间 K-means 聚类，从每簇按比例抽 train/test，test 簇与 train 簇在特征空间分离，直接产生分布偏移。

**DL 文献**：
- **Fooladi et al. 2025**, JCIM, [10.1021/acs.jcim.5c00475](https://doi.org/10.1021/acs.jcim.5c00475)，约 20 引——对比 12 模型×8 数据集×7 种 OOD 划分，**K-means 聚类（ECFP4 指纹）对 GNN 和经典 ML 都构成最大挑战**，ID-OOD 性能相关性降到 r≈0.4（scaffold split 仍有 r≈0.9）。DL 场景直接证据。
- Du et al. 2025, Comp Biol Chem, [10.1016/j.compbiolchem.2025.108778](https://doi.org/10.1016/j.compbiolchem.2025.108778)——K-means 采样 + GCN 抗菌预测 ROC-AUC 0.97

**tv3 适用性**：
- **优势**：DL 直接文献最强（Fooladi 2025）；外推集构造能力强（test 簇天然 OOD）；sklearn 原生 `KMeans`；确定性可复现（给定 random_state）。
- **风险**：纯 X 聚类，不保证 Y 覆盖（需配合分层）；需先做时序特征聚合（512 timestep 不能直接聚类）；K 值选择影响大。

**实施建议**：用慢通道+超声统计特征做 K-means，K 取 8-12（按 silhouette 选）；留 1-2 个边界簇作 extrapolation，其余簇内按比例分 train/val/test。

### A.3.3 XYOnion（7 分）

**原理**：Ezenarro 2025 提出，SPXY 的 X-Y 联合距离 + Onion 分层 shell。先用联合距离排序，再以同心壳层分配样本，保证校准/验证集在 X 与 Y 空间均衡覆盖。用 DISTSLCT 算法避免全距离矩阵，可扩展性优于 KS/SPXY。

**DL 文献**：
- **Ezenarro 2025**, Anal. Chim. Acta, [10.1016/j.aca.2025.344229](https://doi.org/10.1016/j.aca.2025.344229)，5 引——原文，PLSR 场景，对比 random/KS/SPXY/Onion，声称产生"更现实稳定的性能指标"。**DL 场景零文献，独立验证极少**。

**tv3 适用性**：
- **优势**：Onion 分层 shell 是 15 个方法中**唯一天然适配"外推集测试"的设计**——外围 shell 直接作 extrapolation，核心 shell 作训练集；多输出 Y 支持（继承 SPXY）；DISTSLCT 避免全距离矩阵，N=6000 可承受。
- **风险**：2025 新方法，引用仅 5，DL 场景完全未验证；无成熟实现，需自行编码并验证正确性；作者未公开官方代码。
- **注意**：Ezenarro 原文目标是"防止验证集外推"，tv3 要的是"构造外推集"，方向相反但机制相同——取外围 shell 即可。

**实施建议**：若团队愿意承担新方法验证成本，机制最干净。建议先与 SPXY 交叉验证（两者都基于 X-Y 联合距离，结果应可对照），确认实现正确后再用于正式实验。

### A.3.4 Group-aware split / Leave-one-group-out（7 分）

**原理**：按组键（mixture_id、工况批次、成分子空间）划分，同一组不跨训练/测试，防止组级信息泄漏。

**DL 文献**：
- **Tahir et al. 2024**, Expert Systems with Applications, [10.1007/s10489-024-05848-6](https://doi.org/10.1007/s10489-024-05848-6)，6 引——增强子-启动子 DL 预测，证明随机划分导致基因组区域泄漏、性能高估；改用 Leave-one-chromosome-out 后 DL 性能急剧下降，揭示真实泛化能力远低于随机划分报告值
- Chen et al. 2026, Frontiers in Sustainable Food Systems, [10.3389/fsufs.2026.1798121](https://doi.org/10.3389/fsufs.2026.1798121)——多模态 DL 回归（GRA 含量预测）用 LOGO CV，R²=0.979
- Kilinc & Uysal 2015, IEEE ICMLA, [10.1109/icmla.2015.216](https://doi.org/10.1109/icmla.2015.216)——Source-Aware Partitioning

**tv3 适用性**：
- **优势**：与 tv3 的 `mixture_id` split 主键天然契合；sklearn 原生 `GroupKFold`/`GroupShuffleSplit`；DL 防泄漏文献充分；多输出 Y 直接支持。
- **风险**：tv3 中 mixture_id 与 sequence_id 一一对应，纯按 mixture_id 分组**退化为随机划分**，无防泄漏收益。必须改用"按工况子空间分组"（如按 LHS 网格区域或 CO₂ 浓度区间分组）才发挥价值。
- **定位**：不是独立的主划分策略，而是"防泄漏约束层"——叠加在其他方法上。

**实施建议**：定义组键为"LHS 子区域"（如 CO₂ 四分位数×O₂ 四分位数 = 16 组），同一子区域不跨 train/extrapolation，保证 extrapolation 集是未见工况子空间。

### A.3.5 重复分层 K-fold（7 分）

**原理**：重复执行 N 次分层 K-fold（每次不同 seed），得 N×K 个性能估计，用均值±标准差量化划分方差。

**DL 文献**：
- **Calle et al. 2025, NACHOS**, Comput Methods Programs Biomed, [10.1016/j.cmpb.2025.109063](https://doi.org/10.1016/j.cmpb.2025.109063)，15 引——DL 医学影像中 Nested CV + AHPO，强调单次 hold-out 无法量化估计方差
- Vu et al. 2022, J Environ Manage, [10.1016/j.jenvman.2022.114869](https://doi.org/10.1016/j.jenvman.2022.114869)，180 引——RNN-LSTM 废弃物预测，7-fold CV 使 MAPE 降 44.57%
- Moss et al. 2018, arXiv [1806.07139](https://arxiv.org/abs/1806.07139)，40 引——J-K-fold CV 用于 NLP 调参，证明单次 CV 估计不稳定

**tv3 适用性**：
- **优势**：DL 方差量化文献最充分；直接回应 Xu & Goodacre 警告的"小数据集单次划分方差大"问题；sklearn 原生 `RepeatedKFold`；多输出可改造（按 Y 分箱后 stratify）。
- **风险**：N×K 次训练计算开销大（6000 序列×TCN 需评估时长）；本质是 IID 评估的方差量化工具，**不产生外推集**；重复 CV 各折非完全独立，方差估计有偏（Moss 2018 讨论）。
- **定位**：不替代 OOD 评估，而是"性能稳定性补充"。

**实施建议**：作为 cluster split / SPXY 之外的补充评估——用 5×5 repeated K-fold 量化模型性能的划分方差，报告 mean±std。

## A.4 Top 5 之外值得关注的两个方法

- **UMAP/HDBSCAN（5 分）**：理论上外推评估最严，但子代理核实 Fooladi 2025 时发现 chemrxiv 预印本（[10.26434/chemrxiv-2025-g1vjf](https://doi.org/10.26434/chemrxiv-2025-g1vjf)）摘要明确说"K-means clustering using ECFP4 fingerprints poses the hardest challenge"，**未提 UMAP/HDBSCAN**。JCIM 正式版全文需订阅，"UMAP/HDBSCAN 最严"目前是未验证状态。建议先确认正式版全文（访问 [pubs.acs.org](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00475)）再决定是否纳入对比。
- **分层随机按 Y 分箱（6 分）**：解决"训练集覆盖 Y 全范围"痛点，但不产生外推集。适合做内层 train/val 划分，或与 K-means 组合。

## A.5 组合使用建议

tv3 的两个核心需求——**(a) 训练集覆盖 Y 全范围**（O₂/N₂ 弱辨识）和 **(b) 构造 extrapolation 集测外推**——由不同方法满足，单一方法都不够，建议组合：

```
阶段 1（主划分）：SPXY 或 XYOnion
  - 保证 train 覆盖 (CO₂, O₂) 全范围（需求 a）
  - SPXY 求稳，XYOnion 外推集机制更干净

阶段 2（外推集）：K-means 聚类
  - 在剩余样本上 K-means 聚类，留边界簇作 extrapolation（需求 b）
  - 或用 Group-aware 按 LHS 子区域留出未见工况

阶段 3（方差量化）：重复分层 K-fold
  - 在 train 内做 5×5 repeated K-fold，报告性能 mean±std
  - 回应 Xu & Goodacre 的小数据集方差警告
```

这个组合覆盖了 top 5 中的 4 个（SPXY/XYOnion + K-means + LOGO + Repeated K-fold），每个方法各司其职。

## A.6 关键文献清单

| 文献 | DOI | 引用 | 用途 |
|---|---|---|---|
| Galvão 2005, Talanta | [10.1016/j.talanta.2005.03.025](https://doi.org/10.1016/j.talanta.2005.03.025) | 880 | SPXY 原始 |
| Xu & Goodacre 2018, J Anal Testing | [10.1007/s41664-018-0068-2](https://doi.org/10.1007/s41664-018-0068-2) | 900+ | 划分方法对比，批评 SPXY |
| Fooladi 2025, JCIM | [10.1021/acs.jcim.5c00475](https://doi.org/10.1021/acs.jcim.5c00475) | ~20 | K-means 对 GNN 最严（DL 直接证据） |
| Ezenarro 2025, Anal Chim Acta | [10.1016/j.aca.2025.344229](https://doi.org/10.1016/j.aca.2025.344229) | 5 | XYOnion 原始 |
| Tahir 2024, ESWA | [10.1007/s10489-024-05848-6](https://doi.org/10.1007/s10489-024-05848-6) | 6 | LOGO 防泄漏 DL 证据 |
| Calle 2025, NACHOS | [10.1016/j.cmpb.2025.109063](https://doi.org/10.1016/j.cmpb.2025.109063) | 15 | DL Nested CV 方差量化 |
| Saptoro 2012, CPPM | [10.1515/1934-2659.1645](https://doi.org/10.1515/1934-2659.1645) | 151 | KS+ANN 改进 |
| Wang 2025, Applied Sciences | [10.3390/app152211902](https://doi.org/10.3390/app152211902) | 3 | SPXY+ResNet（分类） |

## A.7 未验证与待确认项

1. **Fooladi 2025 JCIM 正式版全文**：未确认是否包含 UMAP/HDBSCAN 划分。chemrxiv 预印本摘要说 K-means 最严。建议通过机构订阅访问正式版确认。
2. **XYOnion 官方代码**：子代理未检索到作者公开实现，需自行按原文编码并验证。
3. **astartes 库**：WebSearch 在本环境持续返回空，未能独立验证其 SPXY/KS 实现 API。
4. **Mahalanobis 距离划分的 DL 文献**：仅找到 PLSR 场景（Shenk & Westerhaus 1991, 550 引），DL 场景零文献，故未入选 top 5。

---

# 附录 B：SPXY 方案审查意见

> 审查日期：2026-07-06
> 审查对象：第 1–8 章主体 SPXY 实施方案
> 与附录 A 的关系：本附录是对主体方案的审查，正是审查中发现"SPXY 对 DL 适用性证据不足"（B.3.2），触发了附录 A 的对照方法调研。两附录合起来构成"评估主体方案 → 调研替代方法"的完整过程。
> 核实方式：通过 CrossRef/Semantic Scholar 逐篇验证 DOI、作者、年份、引用数；读取 `tv3/sim/packaging/splits.py`、`tv3/sim/generation/tunnel_ventilation/benchmark.py`、`tv3/pipeline/generate_tunnel_ventilation_benchmark.py` 核对实现假设。

## B.1 文献核实结论

| 文献 | 文档描述 | 实际核实 | 判定 |
|---|---|---|---|
| Galvão 2005, Talanta 67(4):736-740 | 被引 880+ | 被引 880，卷期页码全对 | ✅ 准确 |
| Tian 2018, Infrared Phys. 95:88-92 | 被引 81 | 被引 81，页码对 | ✅ 准确 |
| Ezenarro 2025, Anal. Chim. Acta 1364:344229 | 最新进展 | 被引 5（2025-08 发表），标题/作者对 | ✅ 准确 |
| Chen 2020, Spectrochim. Acta A 245:119182 | 被引 38 | **年份应为 2021**（CrossRef 返回 2021-01-01），卷号 248，被引 39 | ⚠️ 年份错 |
| Kennard-Stone 1969, Technometrics 11(1):137-148 | KS 前身 | 被引 3241，全对 | ✅ 准确 |
| astartes (JacksonBurns) | 参考实现 | WebSearch 三次返回空，学术库无直接命中 | ⚠️ 未独立验证 |
| NIRPY Research Blog | 教程实现 | 非同行评审博客 | ⚠️ 来源层级低 |

**主体方案遗漏的关键文献**（高影响）：

1. **Xu & Goodacre (2018)**, *J. Analysis and Testing*, DOI [10.1007/s41664-018-0068-2](https://doi.org/10.1007/s41664-018-0068-2)，被引 **900+**。系统对比 CV/bootstrap/KS/SPXY，结论：**"systematic sampling method such as K-S and SPXY generally had very poor estimation of the model performance"**——因为 SPXY 把代表性样本优先选入训练集，验证集代表性差。这篇直接质疑 SPXY 用于"性能评估"的合理性，主体方案完全没引用，是最大遗漏。

2. **Apinantanakon et al. (2019)**, M-SPXY for ANN —— 提出针对神经网络的改进版 SPXY，说明原版 SPXY 用于 ANN 有已知不足。tv3 用 TCN，这篇有直接参考价值。

3. **Wang et al. (2020)**, *Ecol. Indic.* 116:106467，被引 20 —— SPXY + 随机森林（非 PLS）的正向案例，是 SPXY 迁移到非线性模型的少量证据。

## B.2 算法与实现层面的问题

### B.2.1 X 特征未标准化（严重缺陷）

`_build_spxy_features` 直接拼接 `slow_mean`（量级 ~10²）、`tof_s` 的 std（量级 ~10⁻⁷）、`sound_speed`（~340）等不同量级特征。`_spxy_split` 里只做了 `d_x = d_x / d_x_max`（除以最大值），**没有对 X 各维标准化**。

主体方案 4.2 节对 Y 做了 `StandardScaler`，X 却没做，这是 KS/SPXY 实现的经典陷阱。除以 `d_x_max` 只是把距离矩阵整体缩放到 [0,1]，不改变各维对距离的相对贡献——大量级特征（如温度、声速）会主导 `d_x`，小量级特征（如 tof std）几乎被忽略。Galvão 原文里 X 是光谱吸光度，各维同量级，归一化即可；tv3 的聚合特征量级跨 9 个数量级，必须先 `StandardScaler` 或 `RobustScaler`。

**建议**：在 `_build_spxy_features` 末尾对 X 做 `StandardScaler().fit_transform(X)`，与 Y 处理对称。

### B.2.2 split 调用时序与现有代码冲突（集成风险）

当前 `benchmark.py:137` `build_default_split_rows` 在 `benchmark.py:148` `_build_sequence_arrays_for_spec` **之前**调用。SPXY 需要 `arrays["slow"]` 和 `arrays["ultrasonic_*"]` 构建 X 特征，split 必须推迟到 arrays 生成之后。

主体方案 4.5 节的集成示例只给了 split 调用片段，没有体现这个时序变化。实际改动链：
- split 从第 137 行后移到第 156 行（arrays 生成后）
- `validate_benchmark_assets`（159 行）输入顺序不变，但要确认 split_rows 已就绪
- `fit_z_score_scalers`（228 行）依赖 `train_sequence_ids`，时序仍 OK

### B.2.3 `_spxy_split` 的 O(N²·train_size) 纯 Python 循环（性能隐患）

```python
min_dists = [min(d_xy[r, t] for t in train_idx) for r in remaining]
```

每次迭代 O(|remaining| × |train_idx|)，整体 O(N²·train_size)。N=600, train_size≈420 → ~1.5×10⁸ 次，纯 Python 秒级可接受；**N=6000, train_size≈4200 → ~1.5×10¹¹ 次，纯 Python 小时级甚至更久**。主体方案第 6 节风险评估只提了距离矩阵内存（288 MB），漏了这块计算复杂度。

**建议**：向量化 `np.min(d_xy[remaining][:, train_idx], axis=1)`，或直接用 sklearn 的 `pairwise_distances` + 增量更新。astartes 库如果有验证过的实现，优先复用而非手写。

### B.2.4 spec / CLI 改动点未列全

主体方案 4.5 只给了 split 调用片段。实际要改：
- `TunnelVentilationBenchmarkGenerationSpec`（frozen dataclass）加 `split_strategy: str = "random"` 字段
- CLI `generate_tunnel_ventilation_benchmark.py` 加 `--split-strategy` 参数（当前确实没有）
- `_split_summary` 的 `split_policy` 字段需区分 `"random_mixture_id_split_v4"` / `"spxy_v1"`，否则下游分析无法追溯划分来源

## B.3 概念层面的问题

### B.3.1 extrapolation 选取逻辑与 SPXY 设计初衷矛盾（最值得商榷）

主体方案 4.3 节选项 2：在 test_ext 集上用 α=0.1 的 SPXY"优先选 Y 空间极值点为 extrapolation"。

问题在于 SPXY 的 max-min 第一步选的是**距离最远的两个点**（边界极值），后续 max-min 选**远离已选的点**（覆盖性选择，不是极值选择）。把"极值"留给 extrapolation，等于让训练集失去边界覆盖——这与 SPXY"训练集覆盖全空间"的设计目标相反，也违背了主体方案 1.2 节自己强调的"训练集必须覆盖 Y 空间全部范围"。

Xu & Goodacre 的研究正好印证：SPXY 把代表性样本选入训练集后，剩余集代表性差。反向使用（把极值留给 extrapolation）会让训练集代表性更差。

tv3 的 extrapolation 应该测的是"成分空间外推"。更直接的做法是：**按 Y 距训练集凸包的距离排序**选外推点，或按 CO₂/O₂ 的极值排序，而不是用 SPXY max-min。SPXY 不适合做"选极值"这件事。

### B.3.2 SPXY 对 DL 的适用性证据不足

主体方案引用的 SPXY 证据全部来自 PLS/MLR 校准（NIR 光谱）。tv3 用 TCN（深度学习）。SPXY 对 DL 的适用性缺乏文献支撑：
- Xu & Goodacre 在 PLS/SVM 上发现 SPXY 性能估计偏差，DL 对训练集分布通常更敏感，偏差可能更大
- Apinantanakon 2019 专门提出 M-SPXY 改进版用于 ANN，说明原版用于神经网络有已知问题
- 主体方案 5.2 的 A1–A4 TCN 对比实验本质上是**迁移验证**，应明确标注为探索性，而非"已验证方法的落地"

（此条是附录 A 调研的触发点。）

### B.3.3 与 LHS 采样的协同未论证

tv3 用 2D LHS 采样保证全局 space-filling。主体方案 1.1 说 shuffle 破坏了 LHS 的 space-filling，但 SPXY 是否是最佳补救？更简单的替代是**按 LHS 网格分层随机抽样**：直接按 CO₂/O₂ 的分箱做分层抽样，实现更简单、保持 LHS 结构、支持随机性（可多 seed）。主体方案 1.2 的对比表没列这个选项，直接跳到 SPXY。

### B.3.4 递归 SPXY 的统计性质未讨论

主体方案 4.3 用递归 SPXY 做四分类（train/val/test/extrapolation）。第一层在全量选 train_val，第二层在子集上选 train/val。val 集的覆盖性、test 的代表性都无保证。Xu & Goodacre 指出 SPXY 剩余集代表性差，递归会放大这个问题。主体方案应该说明：递归 SPXY 后，val/test 的代表性会逐层下降，并给出量化检查（如各 split 的 Y 范围覆盖率）。

### B.3.5 闭包 Y 的距离几何（优化项，非错误）

tv3 的 CO₂+O₂+N₂=100%，Y 只有 2 个自由度。`StandardScaler` 后欧氏距离在 3D 空间但实际约束在 2D 子平面。更严谨的做法是用 ILR/CLR 变换计算 Y 距离（成分数据几何）。主体方案提到 tv3 模型层不允许 `target_transform`，但 SPXY 的 Y 距离是**划分用**，与模型 loss 无关，可以独立用成分数据变换。这是优化项。

## B.4 总体评价与修订建议

**计划质量**：结构完整，动机清楚，公式正确，文献引用基本准确（仅 Chen 年份错），α 参数化设计灵活（可退化为 KS 或纯 Y）。作为方案文档达到可实施水平。

**主要风险**：
- **X 未标准化**会让 SPXY 实际退化为"被大量级特征主导的伪 KS"，必须修
- **extrapolation 选取逻辑**与 SPXY 设计初衷矛盾，概念上站不住，建议换方案
- **性能**在 N=6000 场景会卡在纯 Python 循环，需向量化或复用 astartes
- **文献遗漏 Xu & Goodacre 2018**——这篇高引批判性文献直接影响 5.1 节"extrapolation R² 更准确反映外推能力"的预期，该预期可能不成立

**修订方向**（按优先级）：
1. 修 X 标准化问题（必须）
2. 补 Xu & Goodacre 2018、Apinantanakon 2019 两篇文献，调整 5.1 的预期表述
3. extrapolation 选取改为按 Y 距训练凸包距离排序，弃用 SPXY max-min 反向逻辑
4. `_spxy_split` 向量化，或评估直接引入 astartes 作为依赖（需先验证其 API）
5. 对比实验矩阵加一个"分层随机（按 LHS 分箱）"对照组，避免只对比 random vs SPXY
6. 修正 Chen 文献年份为 2021

**未验证项**：astartes 库的 API、NIRPY 博客代码的正确性，建议实施前直接查 [github.com/JacksonBurns/astartes](https://github.com/JacksonBurns/astartes) 源码确认，与手写实现交叉验证。
