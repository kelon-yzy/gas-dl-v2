"""v4 z-score scaler IO — shared implementation lives in ``common.scalers``.

Kept as a thin re-export so the deep-learning data API stays at
``dl.data.scalers.{load_scaler,apply_scaler}`` while the single source of truth
(and the ``Z_SCORE_STD_EPSILON`` threshold) is shared with the ml path.
"""

from __future__ import annotations

from tv3.common.scalers import apply_scaler, load_scaler

__all__ = ["apply_scaler", "load_scaler"]
