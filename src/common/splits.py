from __future__ import annotations

import csv
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test", "extrapolation")


def load_splits(split_dir: Path | str) -> dict[str, list[dict[str, str]]]:
    """Load v4 split CSV files from a split directory.

    Each file ``{name}.csv`` must contain ``sequence_id`` and ``mixture_id``
    columns.  Returns a dict keyed by split name with lists of rows.

    Raises ``FileNotFoundError`` if any of the four standard split files
    is missing.
    """
    split_dir = Path(split_dir)
    splits: dict[str, list[dict[str, str]]] = {}
    for name in SPLIT_NAMES:
        path = split_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[name] = _read_csv(path)
        _validate_split_rows(splits[name], name)
    return splits


def split_sequence_ids(splits: dict[str, list[dict[str, str]]]) -> dict[str, list[str]]:
    """Extract ordered sequence_id lists from loaded splits."""
    return {name: [row["sequence_id"] for row in rows] for name, rows in splits.items()}


def resolve_split_indices(
    splits: dict[str, list[dict[str, str]]],
    sequence_ids: list[str],
) -> dict[str, list[int]]:
    """Map split sequence_ids to integer indices into a master id list.

    Returns a dict with the same keys as ``splits``, where each value is a
    list of integer indices.
    """
    lookup = {sid: idx for idx, sid in enumerate(sequence_ids)}
    indices: dict[str, list[int]] = {}
    for name, rows in splits.items():
        indices[name] = []
        for row in rows:
            sid = row["sequence_id"]
            if sid not in lookup:
                raise KeyError(f"sequence_id {sid} (split={name}) not found in master id list")
            indices[name].append(lookup[sid])
    return indices


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_split_rows(rows: list[dict[str, str]], split_name: str) -> None:
    required = {"sequence_id", "mixture_id"}
    if not rows:
        return
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Split {split_name} missing required columns: {sorted(missing)}")
