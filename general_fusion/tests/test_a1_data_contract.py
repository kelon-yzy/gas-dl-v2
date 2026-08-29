from __future__ import annotations

import json

import numpy as np

from gf.sim.a1_dataset import (
    FORBIDDEN_MANIFEST_KEYS,
    TARGET_NAMES,
    assign_a1_splits,
    generate_a1_conditions,
    generate_dataset,
    load_dataset,
)


def test_a1_condition_counts_and_fixed_split() -> None:
    conditions = generate_a1_conditions(
        binary_per_pair=20,
        ternary_count=180,
        generation_seed=20260827,
    )
    assert len(conditions) == 240
    assert sum(condition.condition_family == "binary" for condition in conditions) == 60
    assert sum(condition.condition_family == "ternary" for condition in conditions) == 180
    assert all(np.isclose(sum(condition.composition), 100.0) for condition in conditions)
    assert all(sum(value > 0.0 for value in condition.composition) == 2 for condition in conditions[:60])
    assert all(min(condition.composition) > 0.0 for condition in conditions[60:])

    split_conditions = assign_a1_splits(conditions, split_seed=20260827)
    assert {
        split: sum(condition.split == split for condition in split_conditions)
        for split in ("train", "val", "test")
    } == {"train": 168, "val": 36, "test": 36}


def test_a1_dataset_is_reproducible_and_has_no_legacy_keys(tmp_path) -> None:
    first = generate_dataset(
        tmp_path / "first",
        binary_per_pair=2,
        ternary_count=6,
        generation_seed=20260827,
        split_seed=20260827,
        data_version="test-r1",
    )
    second = generate_dataset(
        tmp_path / "second",
        binary_per_pair=2,
        ternary_count=6,
        generation_seed=20260827,
        split_seed=20260827,
        data_version="test-r1",
    )
    assert first.manifest["content_sha256"] == second.manifest["content_sha256"]
    np.testing.assert_array_equal(first.signals, second.signals)
    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text(encoding="utf-8"))
    assert _all_mapping_keys(manifest).isdisjoint(FORBIDDEN_MANIFEST_KEYS)
    assert manifest["target_names"] == list(TARGET_NAMES)
    reloaded = load_dataset(tmp_path / "first")
    assert len(reloaded.samples()) == 12
    assert all(sample.group_id == sample.metadata["mixture_id"] for sample in reloaded.samples())
    assert all(sample.time[0].shape == (1,) for sample in reloaded.samples())


def _all_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {key for child in value.values() for key in _all_mapping_keys(child)}
    if isinstance(value, list):
        return {key for child in value for key in _all_mapping_keys(child)}
    return set()
