from __future__ import annotations

import os
from pathlib import Path


HITRAN_CACHE_ROOT_ENV = "HG_HITRAN_CACHE_ROOT"


def default_hitran_cache_root() -> str:
    configured = os.environ.get(HITRAN_CACHE_ROOT_ENV)
    if configured is not None:
        if not configured.strip():
            raise ValueError(f"{HITRAN_CACHE_ROOT_ENV} must not be empty when set")
        return configured

    workspace_root = Path(__file__).resolve().parents[4]
    return str(workspace_root / "shared" / "hitran_cache")
