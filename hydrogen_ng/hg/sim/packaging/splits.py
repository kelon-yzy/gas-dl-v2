from __future__ import annotations

import random
from collections.abc import Iterable, Mapping

from sim.core.schema import SPLIT_NAMES


def build_split_groups(
    conditions: Iterable[Mapping[str, object]],
    *,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
) -> dict[str, set[str]]:
    """按 mixture_id 分组后 shuffle 按比例切片。

    注意：此函数不执行分层采样（stratified sampling）。它只是将 mixture_id 打乱后按比例分配。
    当 mixture_id 与 sequence_id 一一对应时（tv3 / hg / sg 均为 1:1），等价于纯随机划分。

    设计意图：mixture_id 可用于分组（同一组分可对应不同噪声/路径的多个 sequence），
    按 mixture_id 分组可防止同组分不同噪声变体泄漏到测试集。但在当前实现中每条
    sequence 对应唯一条 mixture，因此分组无实际效果。

    局限：不保证各 split 内的成分空间均匀覆盖（破坏 LHS 的 space-filling 性质）。
    extrapolation split 是从 shuffle 末尾随机剩余，不基于成分极值或分布外距离选取。
    """
    group_ids = sorted({str(row["mixture_id"]) for row in conditions})
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = max(1, int(n_groups * train_ratio))
    n_val = max(1, int(n_groups * val_ratio)) if n_groups >= 3 else 0
    n_test = max(1, int(n_groups * test_ratio)) if n_groups >= 4 else max(0, n_groups - n_train - n_val)
    if n_train + n_val + n_test > n_groups:
        n_train = max(1, n_groups - 2)
        n_val = 1 if n_groups >= 3 else 0
        n_test = max(0, n_groups - n_train - n_val)

    train_end = n_train
    val_end = train_end + n_val
    test_end = val_end + n_test
    return {
        "train": set(group_ids[:train_end]),
        "val": set(group_ids[train_end:val_end]),
        "test": set(group_ids[val_end:test_end]),
        "extrapolation": set(group_ids[test_end:]),
    }


def build_split_rows_from_group_sets(
    conditions: Iterable[Mapping[str, object]],
    split_groups: Mapping[str, set[str]],
    *,
    group_field: str = "mixture_id",
) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {name: [] for name in split_groups}
    for condition in conditions:
        group_value = str(condition[group_field])
        split_name = _split_name_for_group(group_value, split_groups)
        rows[split_name].append(
            {
                "sequence_id": str(condition["sequence_id"]),
                "mixture_id": str(condition["mixture_id"]),
            }
        )
    return rows


def build_default_split_rows(
    conditions: list[Mapping[str, object]],
    *,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    split_groups = build_split_groups(conditions, seed=seed)
    rows = build_split_rows_from_group_sets(conditions, split_groups)
    return {name: rows.get(name, []) for name in SPLIT_NAMES}


def _split_name_for_group(group_value: str, split_groups: Mapping[str, set[str]]) -> str:
    for split_name, group_set in split_groups.items():
        if group_value in group_set:
            return split_name
    raise ValueError(f"group {group_value!r} was not assigned to a split")
