from __future__ import annotations

from pathlib import Path


TOP_LEVEL_DIRS = ("hg", "configs", "data", "outputs", "docs", "scripts", "tests")
CONFIG_GROUPS = ("data",)
OUTPUT_GROUPS = ("runs", "summary", "reports", "archive")


def ensure_project_layout(root: Path) -> None:
    for dirname in TOP_LEVEL_DIRS:
        (root / dirname).mkdir(parents=True, exist_ok=True)
    for group in CONFIG_GROUPS:
        (root / "configs" / group).mkdir(parents=True, exist_ok=True)
    for group in OUTPUT_GROUPS:
        (root / "outputs" / group).mkdir(parents=True, exist_ok=True)
