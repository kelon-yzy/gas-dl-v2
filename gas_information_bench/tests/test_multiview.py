import json
from pathlib import Path

import numpy as np

from gib.pipeline.multiview import FrozenEncoder, _paired_bootstrap_upper, _view_indices


ROOT = Path(__file__).resolve().parents[1]


def test_views_are_disjoint_and_cover_all_deployment_features():
    groups = _view_indices()
    combined = np.concatenate(list(groups.values()))
    assert len(combined) == 177
    assert np.array_equal(np.sort(combined), np.arange(177))


def test_all_encoders_have_same_output_width_and_are_deterministic():
    config = json.loads((ROOT / "configs" / "p3_c4_multiview_plan.json").read_text(encoding="utf-8"))["encoder"]
    rng = np.random.default_rng(7)
    train = rng.normal(size=(20, 177))
    test = rng.normal(size=(5, 177))
    for method in ("early_fusion", "shared_private", "random_grouping"):
        first = FrozenEncoder(method, config, 101).fit(train).transform(test)
        second = FrozenEncoder(method, config, 101).fit(train).transform(test)
        assert first.shape == (5, 32)
        assert np.array_equal(first, second)


def test_paired_bootstrap_uses_mixture_aggregates():
    gate = {"bootstrap_seed": 20260824, "bootstrap_resamples": 10000, "confidence": 0.95}
    values = {"M1": [-0.2, -0.1], "M2": [-0.3], "M3": [-0.15, -0.25]}
    first = _paired_bootstrap_upper(values, gate)
    second = _paired_bootstrap_upper(values, gate)
    assert first == second
    assert first < 0.0
