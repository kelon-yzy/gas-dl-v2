from __future__ import annotations

import numpy as np

from gf.dl.evaluation import group_bootstrap_comparison


def test_group_bootstrap_uses_component_target_ranges_for_macro_rnmae() -> None:
    targets = np.zeros((1, 3), dtype=np.float64)
    method = np.full((1, 3), 10.0, dtype=np.float64)
    baseline = np.zeros((1, 3), dtype=np.float64)

    result = group_bootstrap_comparison(
        method,
        baseline,
        targets,
        ["mix-1"],
        seed=17,
        samples=25,
        target_ranges=[10.0, 20.0, 40.0],
    )

    assert result["mean"] == (1.0 + 0.5 + 0.25) / 3.0
    assert result["percentile_2_5"] == result["mean"]
    assert result["percentile_97_5"] == result["mean"]
