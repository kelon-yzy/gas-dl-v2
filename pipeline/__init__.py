"""Root-level launcher package for ``python -m pipeline.<tool>`` commands."""

from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
_SRC_PIPELINE_DIR = _SRC_DIR / "pipeline"

if _SRC_DIR.is_dir():
    src_path = str(_SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

if _SRC_PIPELINE_DIR.is_dir():
    src_pipeline_path = str(_SRC_PIPELINE_DIR)
    if src_pipeline_path not in __path__:
        __path__.append(src_pipeline_path)
