from __future__ import annotations

import numpy as np

from gf.dl.contracts import collate_samples
from gf.ml.baselines import run_baseline_suite
from gf.sim.a1_audit import run_information_audit
from gf.sim.a1_dataset import generate_dataset


def test_a1_audit_and_baseline_smoke(tmp_path) -> None:
    dataset = generate_dataset(
        tmp_path / "dataset",
        binary_per_pair=3,
        ternary_count=30,
        generation_seed=20260827,
        split_seed=20260827,
        data_version="smoke-r1",
    )
    audit = run_information_audit(dataset)
    assert audit["gate"]["status"] == "PASS"
    assert audit["jacobian"]["rank_min"] == 2

    samples = dataset.samples()
    batch = collate_samples(samples)
    assert tuple(batch.signals.shape) == (39, 3, 1, 1)
    result = run_baseline_suite(dataset)
    assert result.summary["gate"]["status"] == "PASS"
    assert result.summary["not_applicable"]["B8"]
    assert {row["baseline_id"] for row in result.summary["models"]} == {
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
        "B6",
        "B7",
    }
    assert all(np.isfinite(prediction).all() for prediction in result.predictions.values())
