"""SPXY/OOD 划分诊断。

文档 §2.4 不变量 4：每个 split 必须输出诊断——样本数、CO2/O2/N2 范围、Y 覆盖率、
到 train 的最近邻距离分布、X/Y pairwise 距离摘要。

诊断在 build_*_split_with_summary 中计算并写入 split_summary.json，
不影响 split CSV 格式或下游训练流程。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

from sg.sim.core.schema import SPLIT_NAMES


def _indices_for_split(
    conditions: Sequence[Mapping[str, object]],
    rows: Sequence[Mapping[str, str]],
) -> np.ndarray:
    seq_to_idx = {str(c["sequence_id"]): i for i, c in enumerate(conditions)}
    return np.array([seq_to_idx[r["sequence_id"]] for r in rows], dtype=int)


def _range_coverage(split_vals: np.ndarray, full_range: float) -> float:
    if full_range <= 0.0 or len(split_vals) == 0:
        return 0.0
    return float((split_vals.max() - split_vals.min()) / full_range)


def compute_split_diagnostics(
    conditions: Sequence[Mapping[str, object]],
    labels: np.ndarray,
    split_rows: Mapping[str, Sequence[Mapping[str, str]]],
    X_scaled: np.ndarray | None,
    y_basis: np.ndarray | None,
) -> dict[str, object]:
    """计算四分类诊断。

    Args:
        conditions: condition 行列表（用于 sequence_id -> 序列索引映射）。
        labels: (N, K) 标签数组，K>=2（CO2, O2, [N2]）。
        split_rows: 各 split 的 sequence_id 行。
        X_scaled: 标准化后的 X 特征，None 时跳过 X pairwise 诊断。
        y_basis: (N, 2) CO2/O2 自由度基，None 时跳过 Y 距离诊断。

    Returns:
        每个 split 的诊断 dict；train 额外含 x/y pairwise 距离摘要，
        val/test/extrapolation 额外含到 train 的最近邻 Y 距离分布。
    """
    labels = np.asarray(labels, dtype=np.float64)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    n = len(conditions)
    split_idx = {name: _indices_for_split(conditions, split_rows.get(name, [])) for name in SPLIT_NAMES}
    full_co2_range = float(labels[:, 0].max() - labels[:, 0].min()) if n else 0.0
    full_o2_range = float(labels[:, 1].max() - labels[:, 1].min()) if (n and labels.shape[1] > 1) else 0.0

    diag: dict[str, object] = {}
    train_idx = split_idx["train"]
    for name in SPLIT_NAMES:
        idx = split_idx[name]
        if len(idx) == 0:
            diag[name] = {"sequence_count": 0}
            continue
        co2 = labels[idx, 0]
        entry: dict[str, object] = {
            "sequence_count": int(len(idx)),
            "co2_range": [float(co2.min()), float(co2.max())],
            "co2_coverage": _range_coverage(co2, full_co2_range),
        }
        if labels.shape[1] > 1:
            o2 = labels[idx, 1]
            entry["o2_range"] = [float(o2.min()), float(o2.max())]
            entry["o2_coverage"] = _range_coverage(o2, full_o2_range)
        if labels.shape[1] > 2:
            n2 = labels[idx, 2]
            entry["n2_range"] = [float(n2.min()), float(n2.max())]
        diag[name] = entry

    # val/test/extrapolation 到 train 的最近邻 Y 距离
    if y_basis is not None and len(train_idx) > 0:
        yb = np.asarray(y_basis, dtype=np.float64)
        train_y = yb[train_idx]
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(train_y)
        for name in ("val", "test", "extrapolation"):
            idx = split_idx[name]
            if len(idx) == 0:
                continue
            dists, _ = nn.kneighbors(yb[idx])
            dists = dists.ravel()
            diag[name]["nn_to_train_y_distance"] = {
                "mean": float(dists.mean()),
                "median": float(np.median(dists)),
                "max": float(dists.max()),
            }

    # train X/Y pairwise 距离摘要
    if len(train_idx) > 1:
        if X_scaled is not None:
            px = pdist(np.asarray(X_scaled, dtype=np.float64)[train_idx], metric="euclidean")
            diag["train"]["x_pairwise_distance"] = {
                "mean": float(px.mean()),
                "max": float(px.max()),
            }
        if y_basis is not None:
            py = pdist(np.asarray(y_basis, dtype=np.float64)[train_idx], metric="euclidean")
            diag["train"]["y_pairwise_distance"] = {
                "mean": float(py.mean()),
                "max": float(py.max()),
            }
    return diag
