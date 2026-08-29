from __future__ import annotations

import json
from pathlib import Path

from gf.sim.a2h_dataset import (
    FORBIDDEN_KEYS,
    A2H_SPLITS,
    compute_split_family_hash,
    load_a2h_dataset,
    nominal_signal_parity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "a2h_v2"


def _assert_no_forbidden_keys(value: object) -> None:
    if isinstance(value, dict):
        assert not FORBIDDEN_KEYS.intersection(value)
        for child in value.values():
            _assert_no_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_keys(child)


def test_a2h_development_view_and_full_manifest_contract() -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    development = load_a2h_dataset(DATA_DIR)
    full = load_a2h_dataset(DATA_DIR, include_hard_test=True)

    assert development.hard_test_indices.size == 0
    assert full.hard_test_indices.size == 435
    assert full.signals.shape == (manifest["sample_count"], 3)
    assert compute_split_family_hash(manifest) == manifest["split_family_hash"]
    assert manifest["hard_test_locked_by_default"] is True
    _assert_no_forbidden_keys(manifest)

    for family in {observation.split_family for observation in full.observations}:
        groups_by_split = {
            split: {
                observation.mixture_id
                for observation in full.observations
                if observation.split_family == family and observation.split == split
            }
            for split in A2H_SPLITS
        }
        for left_index, left_split in enumerate(A2H_SPLITS):
            for right_split in A2H_SPLITS[left_index + 1 :]:
                assert groups_by_split[left_split].isdisjoint(groups_by_split[right_split])


def test_a2h_nominal_signal_parity_is_zero_with_frozen_a1_core() -> None:
    dataset = load_a2h_dataset(DATA_DIR)
    parity = nominal_signal_parity([condition.composition for condition in dataset.conditions[:16]])
    assert parity["sample_count"] == 16
    assert parity["max_absolute_difference"] < 1.0e-12
