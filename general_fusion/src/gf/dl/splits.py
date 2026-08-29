from __future__ import annotations

from collections.abc import Iterable, Mapping


class SplitLeakageError(ValueError):
    """Raised when group identities leak across data splits."""


def validate_group_splits(
    splits: Mapping[str, Iterable[str]],
    *,
    known_group_ids: Iterable[str] | None = None,
) -> dict[str, frozenset[str]]:
    required = {"train", "val", "test"}
    if set(splits) != required:
        raise SplitLeakageError(f"split keys must be exactly {sorted(required)}, got {sorted(splits)}")

    normalized: dict[str, frozenset[str]] = {}
    for split_name in sorted(required):
        values = list(splits[split_name])
        if not values:
            raise SplitLeakageError(f"{split_name} split must contain at least one group")
        if any(not value for value in values):
            raise SplitLeakageError(f"{split_name} split contains an empty group_id")
        if len(set(values)) != len(values):
            raise SplitLeakageError(f"{split_name} split contains duplicate group_id values")
        normalized[split_name] = frozenset(values)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = normalized[left] & normalized[right]
        if overlap:
            raise SplitLeakageError(f"group leakage between {left} and {right}: {sorted(overlap)}")

    if known_group_ids is not None:
        known = frozenset(known_group_ids)
        assigned = frozenset().union(*normalized.values())
        if assigned != known:
            missing = sorted(known - assigned)
            unknown = sorted(assigned - known)
            raise SplitLeakageError(f"split manifest mismatch: missing={missing}, unknown={unknown}")

    return normalized
