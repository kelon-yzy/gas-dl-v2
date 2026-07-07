"""v4 split CSV IO — shared implementation lives in ``common.splits``.

Kept as a thin re-export so the deep-learning data API stays at
``dl.data.splits`` while the single source of truth is shared with the ml path.
"""

from __future__ import annotations

from tv3.common.splits import (
    SPLIT_NAMES,
    load_splits,
    resolve_split_indices,
    split_sequence_ids,
)

__all__ = ["SPLIT_NAMES", "load_splits", "resolve_split_indices", "split_sequence_ids"]
