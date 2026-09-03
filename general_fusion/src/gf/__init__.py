"""general_fusion (gf)：通用多模态气体融合主线包。"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


__all__ = ["dl", "ml", "pipeline", "sim"]


def __getattr__(name: str) -> ModuleType:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"gf.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
