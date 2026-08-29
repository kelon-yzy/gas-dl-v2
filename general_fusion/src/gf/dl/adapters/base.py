from __future__ import annotations

from typing import Protocol

from gf.dl.contracts import UnifiedSample


class AdapterError(ValueError):
    """Raised when raw data cannot be mapped to the frozen contract."""


class DatasetAdapter(Protocol):
    def load_samples(self) -> list[UnifiedSample]: ...
