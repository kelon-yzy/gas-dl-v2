"""SPXY + OOD 数据集划分（tv3 场景）。

实施依据：docs/掘进通风/spxy_split_implementation_plan.md。

不变量（文档 §2.4）：
1. SPXY 只在 ID pool 内选择 train，不参与 extrapolation 选择。
2. extrapolation 由独立 OOD selector（y_margin_ood / lhs_boundary / kmeans_boundary）产生。
3. val/test 从 ID 剩余样本按 Y 分箱分层随机划分，不递归 SPXY。
4. X/Y 特征在距离计算前必须 StandardScaler 标准化。
5. OOD 选择退化时显式失败，不静默回退为随机。

与 splits.py 的 build_default_split_rows 并存；split CSV 格式（sequence_id, mixture_id）不变。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from hg.sim.core.schema import SPLIT_NAMES


class SpxySplitError(RuntimeError):
    """SPXY/OOD 划分退化错误。

    OOD selector 无法在当前采样上产生外推集时抛出，要求显式切换策略，
    不静默回退为随机剩余（文档 §2.4 不变量 5）。
    """


# ---------------------------------------------------------------------------
# 特征构建与标准化
# ---------------------------------------------------------------------------

# 超声统计特征配置：(arrays key, 是否计算 trend)
_ULTRASONIC_FEATURE_KEYS: tuple[tuple[str, bool], ...] = (
    ("ultrasonic_tof_s", True),
    ("ultrasonic_sound_speed_m_per_s", False),
    ("ultrasonic_alpha_true_npm", False),
)


def _slow_trend(slow: np.ndarray) -> np.ndarray:
    """逐通道线性斜率，向量化解析实现。slow: (N, T, C) -> (N, C)。"""
    n, t, c = slow.shape
    if t < 2:
        return np.zeros((n, c), dtype=np.float64)
    ts = np.arange(t, dtype=np.float64)
    ts_c = ts - ts.mean()
    denom = float((ts_c ** 2).sum())
    if denom <= 0.0:
        return np.zeros((n, c), dtype=np.float64)
    slow_c = slow - slow.mean(axis=1, keepdims=True)
    return (ts_c[None, :, None] * slow_c).sum(axis=1) / denom


def _series_trend(arr: np.ndarray) -> np.ndarray:
    """(N, T) -> (N,) 线性斜率。"""
    n, t = arr.shape
    if t < 2:
        return np.zeros(n, dtype=np.float64)
    ts = np.arange(t, dtype=np.float64)
    ts_c = ts - ts.mean()
    denom = float((ts_c ** 2).sum())
    if denom <= 0.0:
        return np.zeros(n, dtype=np.float64)
    arr_c = arr - arr.mean(axis=1, keepdims=True)
    return (ts_c[None, :] * arr_c).sum(axis=1) / denom


def _build_spxy_features(conditions: Sequence[Mapping[str, object]], arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """方案 A：慢通道时序统计 + 超声波形统计。输出 (N, F) 未标准化。

    慢通道：7 通道 × (mean/std/min/max/trend) = 35 维。
    超声：tof_s(mean/std/trend) + sound_speed(mean/std) + alpha(mean/std) = 7 维。
    合计 42 维。聚合特征量级跨数个数量级，必须经 StandardScaler。
    """
    slow = np.asarray(arrays["slow"], dtype=np.float64)  # (N, T, C)
    n = slow.shape[0]
    feats: list[np.ndarray] = [
        slow.mean(axis=1),
        slow.std(axis=1),
        slow.min(axis=1),
        slow.max(axis=1),
        _slow_trend(slow),
    ]
    for key, with_trend in _ULTRASONIC_FEATURE_KEYS:
        arr = np.asarray(arrays[key], dtype=np.float64)  # (N, T)
        if arr.ndim == 1:
            arr = arr.reshape(n, -1)
        feats.append(arr.mean(axis=1, keepdims=True))
        feats.append(arr.std(axis=1, keepdims=True))
        if with_trend:
            feats.append(_series_trend(arr).reshape(n, 1))
    return np.concatenate(feats, axis=1)


def _build_scaled_split_features(
    conditions: Sequence[Mapping[str, object]],
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建标准化 X / Y 与 Y 自由度基。

    tv3 的 CO2+O2+N2=100%，第一版用 CO2/O2 两个自由度计算 Y 距离（文档 §4.2）。
    返回 (X_scaled, y_scaled, y_basis)；y_basis 为 (N, 2) 未标准化的 CO2/O2。
    """
    X_raw = _build_spxy_features(conditions, arrays)
    X_scaled = StandardScaler().fit_transform(X_raw)
    y_basis = np.asarray(labels, dtype=np.float64)[:, :2]  # CO2, O2
    y_scaled = StandardScaler().fit_transform(y_basis)
    return X_scaled, y_scaled, y_basis


# ---------------------------------------------------------------------------
# SPXY 向量化选择
# ---------------------------------------------------------------------------


def _normalize_distance_matrix(d: np.ndarray, *, name: str) -> np.ndarray:
    """距离矩阵除以最大值归一化到 [0,1]。全零矩阵保持全零。"""
    max_val = float(d.max()) if d.size else 0.0
    if max_val <= 0.0:
        return np.zeros_like(d)
    return d / max_val


def _spxy_select_train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_size: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """向量化 SPXY 在 ID pool 内选 train。

    增量维护 min_dist_to_selected，O(N²) 距离矩阵 + O(N·train_size) 增量更新，
    无 Python 嵌套 min 循环（文档 §4.4、附录 B.2.3）。

    返回 (train_local_idx, remainder_local_idx)，均相对输入 X/y 的局部索引。
    """
    n = len(X)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    train_size = max(0, min(train_size, n))
    if train_size == 0:
        return np.array([], dtype=int), np.arange(n, dtype=int)
    if train_size == 1:
        return np.array([0], dtype=int), np.arange(1, n, dtype=int)

    d_x = _normalize_distance_matrix(squareform(pdist(X, metric="euclidean")), name="X")
    d_y = _normalize_distance_matrix(squareform(pdist(y, metric="euclidean")), name="Y")
    d_xy = alpha * d_x + (1.0 - alpha) * d_y

    # 初始对：距离最远的两个样本（屏蔽对角线）
    d_xy_offdiag = d_xy.copy()
    np.fill_diagonal(d_xy_offdiag, -np.inf)
    i, j = np.unravel_index(int(np.argmax(d_xy_offdiag)), d_xy_offdiag.shape)
    if i == j:  # 全零距离退化
        i, j = 0, 1

    selected = [int(i), int(j)]
    selected_mask = np.zeros(n, dtype=bool)
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


def _spxy_select_train_naive(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_size: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """朴素 O(N²·train_size) 实现，仅用于测试交叉验证向量化版本正确性。"""
    n = len(X)
    d_x = _normalize_distance_matrix(squareform(pdist(X, metric="euclidean")), name="X")
    d_y = _normalize_distance_matrix(squareform(pdist(y, metric="euclidean")), name="Y")
    d_xy = alpha * d_x + (1.0 - alpha) * d_y
    d_xy_offdiag = d_xy.copy()
    np.fill_diagonal(d_xy_offdiag, -np.inf)
    i, j = np.unravel_index(int(np.argmax(d_xy_offdiag)), d_xy_offdiag.shape)
    if i == j:
        i, j = 0, 1
    selected = [int(i), int(j)]
    while len(selected) < train_size:
        remaining = [r for r in range(n) if r not in selected]
        if not remaining:
            break
        best = max(remaining, key=lambda r: min(d_xy[r, t] for t in selected))
        selected.append(best)
    selected_arr = np.array(selected, dtype=int)
    mask = np.zeros(n, dtype=bool)
    mask[selected_arr] = True
    return selected_arr, np.flatnonzero(~mask)


# ---------------------------------------------------------------------------
# Y 分箱分层 val/test 划分
# ---------------------------------------------------------------------------


def _y_bin_labels(y_basis: np.ndarray, *, n_bins: int = 4) -> np.ndarray:
    """CO2/O2 各分 n_bins 箱，组合成 (n_bins*n_bins) 类标签。"""
    n = len(y_basis)
    if n == 0:
        return np.array([], dtype=int)
    co2 = y_basis[:, 0]
    o2 = y_basis[:, 1]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    co2_q = np.quantile(co2, edges)
    o2_q = np.quantile(o2, edges)
    # searchsorted 保证每个样本落入某箱；用内部边界切片避免端点重复
    co2_label = np.clip(np.searchsorted(co2_q[1:-1], co2, side="right"), 0, n_bins - 1)
    o2_label = np.clip(np.searchsorted(o2_q[1:-1], o2, side="right"), 0, n_bins - 1)
    return co2_label * n_bins + o2_label


def _stratified_val_test_split(
    indices: np.ndarray,
    y_basis: np.ndarray,
    *,
    val_size: int,
    test_size: int,
    seed: int,
    n_bins: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """对 ID 剩余样本按 CO2/O2 分箱分层随机划分 val/test。

    小样本下 StratifiedShuffleSplit 可能因每类不足而失败，此时回退为纯随机
    （回退仅在 N 极小时触发，不违反 OOD 不变量——此层只切 val/test，不涉及 extrapolation）。
    """
    n = len(indices)
    total = val_size + test_size
    if total >= n:
        # 不够分：val 优先
        val_size = min(val_size, n)
        test_size = max(0, n - val_size)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        return indices[perm[:val_size]], indices[perm[val_size:val_size + test_size]]

    bin_label = _y_bin_labels(y_basis, n_bins=n_bins)
    unique = np.unique(bin_label)
    if len(unique) < 2:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        return indices[perm[:val_size]], indices[perm[val_size:val_size + test_size]]

    try:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            train_size=val_size,
            random_state=seed,
        )
        val_local, test_local = next(sss.split(indices, bin_label))
        return indices[val_local], indices[test_local]
    except ValueError:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        return indices[perm[:val_size]], indices[perm[val_size:val_size + test_size]]


# ---------------------------------------------------------------------------
# OOD / extrapolation selectors
# ---------------------------------------------------------------------------


def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab)) + 1e-30
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _point_to_hull_distances(points: np.ndarray, hull_points: np.ndarray) -> np.ndarray:
    """points 中每个点到 hull_points 凸包的距离。凸包内为 0，凸包外为到边界最短距离。"""
    n_p = len(points)
    if n_p == 0:
        return np.array([], dtype=np.float64)
    # Delaunay 判断点是否在凸包内
    try:
        tri = Delaunay(hull_points)
        inside = tri.find_simplex(points) >= 0
    except Exception:
        inside = np.zeros(n_p, dtype=bool)
    try:
        hull = ConvexHull(hull_points)
    except Exception:
        # hull_points 退化（共线或点数<3）：距离退化为到最近顶点的欧氏距离
        dists = np.linalg.norm(points[:, None, :] - hull_points[None, :, :], axis=2).min(axis=1)
        return dists
    verts = hull_points[hull.vertices]
    n_v = len(verts)
    dists = np.empty(n_p, dtype=np.float64)
    for i in range(n_p):
        if inside[i]:
            dists[i] = 0.0
            continue
        p = points[i]
        d_min = float("inf")
        for j in range(n_v):
            a = verts[j]
            b = verts[(j + 1) % n_v]
            d = _point_to_segment_dist(p, a, b)
            if d < d_min:
                d_min = d
        dists[i] = d_min
    return dists


def _y_margin_ood_select(
    y_basis: np.ndarray,
    *,
    n_ext: int,
    interior_quantiles: tuple[float, float],
) -> np.ndarray:
    """按 CO2/O2 中心分位定义 interior domain，boundary candidates 到 interior 凸包距离最大的 n_ext 个。"""
    if n_ext <= 0:
        return np.array([], dtype=int)
    lo_q, hi_q = interior_quantiles
    co2 = y_basis[:, 0]
    o2 = y_basis[:, 1]
    co2_lo, co2_hi = np.quantile(co2, [lo_q, hi_q])
    o2_lo, o2_hi = np.quantile(o2, [lo_q, hi_q])
    interior_mask = (co2 >= co2_lo) & (co2 <= co2_hi) & (o2 >= o2_lo) & (o2 <= o2_hi)
    interior_idx = np.flatnonzero(interior_mask)
    boundary_idx = np.flatnonzero(~interior_mask)
    if len(interior_idx) < 4:
        raise SpxySplitError(
            f"y_margin_ood 退化：interior domain 样本不足（{len(interior_idx)} < 4），"
            f"无法构成 2D 凸包。请放宽 interior_quantiles 或改用 lhs_boundary/kmeans_boundary。"
        )
    if len(boundary_idx) < n_ext:
        raise SpxySplitError(
            f"y_margin_ood 退化：boundary candidates 不足（{len(boundary_idx)} < {n_ext}）。"
            f"请放宽 interior_quantiles 或改用 lhs_boundary/kmeans_boundary。"
        )
    dists = _point_to_hull_distances(y_basis[boundary_idx], y_basis[interior_idx])
    positive_mask = dists > 0.0
    n_positive = int(positive_mask.sum())
    if n_positive < n_ext:
        raise SpxySplitError(
            f"y_margin_ood 退化：到 interior 凸包距离 > 0 的 boundary candidates 不足"
            f"（{n_positive} < {n_ext}）。请改用 lhs_boundary 或 kmeans_boundary。"
        )
    positive_local = np.flatnonzero(positive_mask)
    order = np.argsort(-dists[positive_local])[:n_ext]
    return boundary_idx[positive_local[order]]


def _lhs_boundary_select(
    y_basis: np.ndarray,
    *,
    n_ext: int,
    n_bins: int,
    seed: int,
) -> np.ndarray:
    """按 CO2/O2 分箱网格，边缘格样本中随机抽 n_ext 个作 extrapolation。"""
    if n_ext <= 0:
        return np.array([], dtype=int)
    co2 = y_basis[:, 0]
    o2 = y_basis[:, 1]
    co2_label = np.clip(np.digitize(co2, np.quantile(co2, np.linspace(0, 1, n_bins + 1))[1:-1]), 0, n_bins - 1)
    o2_label = np.clip(np.digitize(o2, np.quantile(o2, np.linspace(0, 1, n_bins + 1))[1:-1]), 0, n_bins - 1)
    edge_mask = (
        (co2_label == 0) | (co2_label == n_bins - 1)
        | (o2_label == 0) | (o2_label == n_bins - 1)
    )
    edge_idx = np.flatnonzero(edge_mask)
    if len(edge_idx) < n_ext:
        raise SpxySplitError(
            f"lhs_boundary 退化：边缘格样本不足（{len(edge_idx)} < {n_ext}）。"
            f"请减少 n_bins 或改用 kmeans_boundary/y_margin_ood。"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(edge_idx, size=n_ext, replace=False)
    return np.sort(chosen)


def _kmeans_boundary_select(
    X_scaled: np.ndarray,
    *,
    n_ext: int,
    n_clusters: int,
    seed: int,
) -> np.ndarray:
    """对 X_scaled K-means 聚类，距全局中心最远的 1-2 个边界簇中随机抽 n_ext 个。"""
    if n_ext <= 0:
        return np.array([], dtype=int)
    n = len(X_scaled)
    n_clusters = max(2, min(n_clusters, n))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)
    centers = km.cluster_centers_
    global_center = X_scaled.mean(axis=0)
    center_dists = np.linalg.norm(centers - global_center, axis=1)
    n_ext_clusters = max(1, min(n_clusters // 4, 2))
    far_clusters = np.argsort(-center_dists)[:n_ext_clusters]
    pool = np.flatnonzero(np.isin(labels, far_clusters))
    if len(pool) < n_ext:
        raise SpxySplitError(
            f"kmeans_boundary 退化：边界簇样本不足（{len(pool)} < {n_ext}）。"
            f"请调整 n_clusters 或改用 y_margin_ood/lhs_boundary。"
        )
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, size=n_ext, replace=False)
    return np.sort(chosen)


def _select_extrapolation_indices(
    *,
    y_basis: np.ndarray,
    X_scaled: np.ndarray,
    n_ext: int,
    strategy: str,
    seed: int,
    interior_quantiles: tuple[float, float],
    n_bins: int,
    n_clusters: int,
) -> np.ndarray:
    """OOD selector 分派。SPXY 不参与此步（不变量 1）。"""
    if strategy == "none" or n_ext <= 0:
        return np.array([], dtype=int)
    if strategy == "y_margin_ood":
        return _y_margin_ood_select(y_basis, n_ext=n_ext, interior_quantiles=interior_quantiles)
    if strategy == "lhs_boundary":
        return _lhs_boundary_select(y_basis, n_ext=n_ext, n_bins=n_bins, seed=seed)
    if strategy == "kmeans_boundary":
        return _kmeans_boundary_select(X_scaled, n_ext=n_ext, n_clusters=n_clusters, seed=seed)
    raise ValueError(f"未知 extrapolation_strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# split rows 组装
# ---------------------------------------------------------------------------


def _build_split_rows_from_indices(
    conditions: Sequence[Mapping[str, object]],
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    ext_idx: np.ndarray,
) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_NAMES}
    buckets = (("train", train_idx), ("val", val_idx), ("test", test_idx), ("extrapolation", ext_idx))
    for name, idx_arr in buckets:
        for idx in idx_arr:
            cond = conditions[int(idx)]
            rows[name].append(
                {
                    "sequence_id": str(cond["sequence_id"]),
                    "mixture_id": str(cond["mixture_id"]),
                }
            )
    return rows


def _build_spxy_split(
    conditions: Sequence[Mapping[str, object]],
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    extrapolation_ratio: float = 0.05,
    alpha: float = 0.5,
    extrapolation_strategy: str = "y_margin_ood",
    interior_quantiles: tuple[float, float] = (0.10, 0.90),
    n_bins: int = 4,
    n_clusters: int = 8,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object], tuple[np.ndarray, np.ndarray]]:
    """SPXY+OOD 四分类核心。返回 (rows, summary_extra, (X_scaled, y_basis))。"""
    X_scaled, y_scaled, y_basis = _build_scaled_split_features(conditions, arrays, labels)
    n = len(conditions)
    n_ext = int(round(n * extrapolation_ratio))
    ext_idx = _select_extrapolation_indices(
        y_basis=y_basis,
        X_scaled=X_scaled,
        n_ext=n_ext,
        strategy=extrapolation_strategy,
        seed=seed,
        interior_quantiles=interior_quantiles,
        n_bins=n_bins,
        n_clusters=n_clusters,
    )
    id_idx = np.setdiff1d(np.arange(n, dtype=int), ext_idx, assume_unique=False)
    train_size = min(int(round(n * train_ratio)), len(id_idx))
    train_local, remainder_local = _spxy_select_train(
        X_scaled[id_idx], y_scaled[id_idx], train_size=train_size, alpha=alpha
    )
    train_idx = id_idx[train_local]
    remainder_idx = id_idx[remainder_local]
    val_size = int(round(n * val_ratio))
    test_size = int(round(n * test_ratio))
    val_idx, test_idx = _stratified_val_test_split(
        remainder_idx, y_basis[remainder_idx], val_size=val_size, test_size=test_size, seed=seed, n_bins=n_bins
    )
    rows = _build_split_rows_from_indices(
        conditions, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, ext_idx=ext_idx
    )
    policy = f"spxy_v1:{extrapolation_strategy}" if extrapolation_strategy != "none" else "spxy_v1"
    summary = {
        "split_policy": policy,
        "spxy_alpha": float(alpha),
        "extrapolation_strategy": extrapolation_strategy,
        "group_field": "mixture_id",
    }
    return rows, summary, (X_scaled, y_basis)


def build_spxy_split_rows(
    conditions: list[Mapping[str, object]],
    arrays: Mapping[str, np.ndarray],
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
    """构建 tv3 的 SPXY+OOD 四分类 split rows（文档 §4.4 接口）。"""
    rows, _, _ = _build_spxy_split(
        conditions,
        arrays,
        labels,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        extrapolation_ratio=extrapolation_ratio,
        alpha=alpha,
        extrapolation_strategy=extrapolation_strategy,
    )
    return {name: rows.get(name, []) for name in SPLIT_NAMES}


def build_spxy_split_with_summary(
    conditions: list[Mapping[str, object]],
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    extrapolation_ratio: float = 0.05,
    alpha: float = 0.5,
    extrapolation_strategy: str = "y_margin_ood",
    interior_quantiles: tuple[float, float] = (0.10, 0.90),
    n_bins: int = 4,
    n_clusters: int = 8,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    """SPXY+OOD 划分，附带 split_policy / alpha / strategy / diagnostics 的 summary。"""
    from hg.sim.packaging.spxy_diagnostics import compute_split_diagnostics

    rows, summary, (X_scaled, y_basis) = _build_spxy_split(
        conditions,
        arrays,
        labels,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        extrapolation_ratio=extrapolation_ratio,
        alpha=alpha,
        extrapolation_strategy=extrapolation_strategy,
        interior_quantiles=interior_quantiles,
        n_bins=n_bins,
        n_clusters=n_clusters,
    )
    summary["diagnostics"] = compute_split_diagnostics(conditions, labels, rows, X_scaled, y_basis)
    return {name: rows.get(name, []) for name in SPLIT_NAMES}, summary


# ---------------------------------------------------------------------------
# LHS / Y 分箱分层随机对照（不调用 SPXY，不调用 OOD selector）
# ---------------------------------------------------------------------------


def _build_lhs_stratified_split(
    conditions: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    extrapolation_ratio: float = 0.05,
    n_bins: int = 4,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    """全量按 CO2/O2 分箱分层随机四分类。

    extrapolation 是 Y 边界格随机样本（按 _lhs_boundary_select），但不调用 SPXY，
    作为 SPXY 的简单对照（文档 §3.2 第 3 点、附录 B.3.3）。
    """
    n = len(conditions)
    y_basis = np.asarray(labels, dtype=np.float64)[:, :2]
    n_ext = int(round(n * extrapolation_ratio))
    try:
        ext_idx = _lhs_boundary_select(y_basis, n_ext=n_ext, n_bins=n_bins, seed=seed)
    except SpxySplitError:
        # 边界格不足时退化为按 CO2+O2 到中心距离最远的 n_ext 个
        center = y_basis.mean(axis=0)
        dists = np.linalg.norm(y_basis - center, axis=1)
        ext_idx = np.argsort(-dists)[:n_ext]
    id_idx = np.setdiff1d(np.arange(n, dtype=int), ext_idx, assume_unique=False)
    train_size = min(int(round(n * train_ratio)), len(id_idx))
    val_size = int(round(n * val_ratio))
    test_size = int(round(n * test_ratio))
    # train 先从 id_idx 按 stratified 抽出
    y_id = y_basis[id_idx]
    bin_label = _y_bin_labels(y_id, n_bins=n_bins)
    rng = np.random.default_rng(seed)
    try:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_size + test_size, train_size=train_size, random_state=seed
        )
        train_local, vt_local = next(sss.split(id_idx, bin_label))
    except ValueError:
        perm = rng.permutation(len(id_idx))
        train_local = perm[:train_size]
        vt_local = perm[train_size:train_size + val_size + test_size]
    train_idx = id_idx[train_local]
    vt_idx = id_idx[vt_local]
    val_idx, test_idx = _stratified_val_test_split(
        vt_idx, y_basis[vt_idx], val_size=val_size, test_size=test_size, seed=seed, n_bins=n_bins
    )
    rows = _build_split_rows_from_indices(
        conditions, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, ext_idx=ext_idx
    )
    summary = {
        "split_policy": "lhs_stratified_split_v1",
        "group_field": "mixture_id",
        "n_bins": int(n_bins),
    }
    return rows, summary


def build_lhs_stratified_split_rows(
    conditions: list[Mapping[str, object]],
    labels: np.ndarray,
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    extrapolation_ratio: float = 0.05,
) -> dict[str, list[dict[str, str]]]:
    """LHS/Y 分箱分层随机四分类（SPXY 简单对照）。"""
    rows, _ = _build_lhs_stratified_split(
        conditions, labels, seed=seed, train_ratio=train_ratio,
        val_ratio=val_ratio, test_ratio=test_ratio, extrapolation_ratio=extrapolation_ratio,
    )
    return {name: rows.get(name, []) for name in SPLIT_NAMES}


def build_lhs_stratified_split_with_summary(
    conditions: list[Mapping[str, object]],
    labels: np.ndarray,
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    extrapolation_ratio: float = 0.05,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    """LHS 分层随机划分，附带 summary 与 diagnostics。"""
    from hg.sim.packaging.spxy_diagnostics import compute_split_diagnostics

    rows, summary = _build_lhs_stratified_split(
        conditions, labels, seed=seed, train_ratio=train_ratio,
        val_ratio=val_ratio, test_ratio=test_ratio, extrapolation_ratio=extrapolation_ratio,
    )
    # 对照组无需 X 特征，诊断只输出 Y 维度
    summary["diagnostics"] = compute_split_diagnostics(conditions, labels, rows, None, None)
    return {name: rows.get(name, []) for name in SPLIT_NAMES}, summary
