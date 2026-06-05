"""Root-level launcher package for ``python -m ml.<tool>`` commands."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_SRC_ML_DIR = _SRC_DIR / "ml"

if _SRC_DIR.is_dir():
    src_path = str(_SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

if _SRC_ML_DIR.is_dir():
    src_ml_path = str(_SRC_ML_DIR)
    if src_ml_path not in __path__:
        __path__.append(src_ml_path)
