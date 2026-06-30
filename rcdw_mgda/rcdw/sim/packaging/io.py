"""CSV / JSON 写盘工具（等价 HG 主线，独立维护）。"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def write_csv(
    path: Path, fieldnames: Iterable[str], rows: Iterable[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )
