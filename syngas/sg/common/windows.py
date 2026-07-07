from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WINDOW_KIND_PHASE = "phase"
WINDOW_KIND_EARLY = "early"
WINDOW_KIND_OPTIONS = (WINDOW_KIND_PHASE, WINDOW_KIND_EARLY)


@dataclass(frozen=True, slots=True)
class WindowConfig:
    kind: str
    value: str | float


def resolve_window_config(value: object) -> WindowConfig | None:
    if value is None:
        return None
    if isinstance(value, WindowConfig):
        return value
    if not isinstance(value, dict):
        raise ValueError("window must be null or a JSON object")
    if set(value) != {"kind", "value"}:
        raise ValueError("window must contain exactly ['kind', 'value']")
    kind = str(value["kind"])
    if kind == WINDOW_KIND_PHASE:
        phase = str(value["value"])
        if not phase:
            raise ValueError("phase window value must be a non-empty string")
        return WindowConfig(kind=kind, value=phase)
    if kind == WINDOW_KIND_EARLY:
        try:
            fraction = float(value["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError("early window value must be a number") from exc
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"early window value must be in (0, 1], got {fraction}")
        return WindowConfig(kind=kind, value=fraction)
    raise ValueError(f"Unknown window kind: {kind!r}. Available: {WINDOW_KIND_OPTIONS}")


def window_to_payload(window: WindowConfig | None) -> dict[str, Any] | None:
    if window is None:
        return None
    return {"kind": window.kind, "value": window.value}


def window_label(window: WindowConfig | None) -> str:
    if window is None:
        return "full"
    if window.kind == WINDOW_KIND_PHASE:
        return f"phase:{window.value}"
    if window.kind == WINDOW_KIND_EARLY:
        return f"early:{float(window.value):.2f}"
    raise ValueError(f"Unknown window kind: {window.kind!r}")
