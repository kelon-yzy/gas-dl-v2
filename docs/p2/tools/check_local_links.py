"""Validate local Markdown links in the P2 document tree."""

from __future__ import annotations

import re
from pathlib import Path


P2_ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:")


def main() -> int:
    missing: list[str] = []
    checked = 0
    for document in sorted(P2_ROOT.rglob("*.md")):
        text = document.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            target = match.group(1).strip().strip("<>")
            if not target or target.startswith("#") or target.startswith(EXTERNAL_PREFIXES):
                continue
            path_text = target.split("#", 1)[0]
            if not path_text:
                continue
            checked += 1
            if not (document.parent / path_text).resolve().exists():
                missing.append(f"{document.relative_to(P2_ROOT)} -> {target}")
    if missing:
        raise SystemExit("missing local Markdown links:\n" + "\n".join(missing))
    print(f"checked_local_links={checked}; missing=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
