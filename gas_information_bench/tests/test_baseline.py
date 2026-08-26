import json
from pathlib import Path

import numpy as np

from gib.pipeline.baseline import _metrics, _model_hashes, _oracle_predictions


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_baseline_plan_binds_models_and_both_gate_branches():
    plan = json.loads((ROOT / "configs" / "p3_baseline_plan.json").read_text(encoding="utf-8"))
    assert plan["plan_status"] == "frozen_before_fit"
    assert set(plan["models"]) == {
        "ridge",
        "gbdt",
        "xgboost_strong_table",
        "mlp_fixed",
        "tcn_fixed",
    }
    assert plan["g3_3"]["minimum_oracle_r2_gap"] == 0.05
    assert plan["g3_3"]["non_inferiority_bands"] == {
        "N2": 0.008,
        "CO2": 0.003,
        "O2": 0.01,
        "Ar": 0.005,
    }
    hashes = _model_hashes(plan)
    assert set(hashes) == set(plan["models"])
    assert all(len(value) == 64 for value in hashes.values())


def test_oracle_is_deterministic_and_metric_rows_are_per_component():
    truth = np.asarray([[0.10, 0.01], [0.20, 0.02], [0.30, 0.03]])
    crb = np.full_like(truth, 0.02)
    first = _oracle_predictions(truth, crb, ["S1", "S2", "S3"], 101, 0.05)
    second = _oracle_predictions(truth, crb, ["S1", "S2", "S3"], 101, 0.05)
    assert np.array_equal(first, second)
    rows = _metrics(truth, first, ["N2", "CO2"])
    assert [row["component"] for row in rows] == ["N2", "CO2"]
    assert all(row["sample_count"] == 3 for row in rows)
    assert all(set(row) == {"component", "p90", "rmse", "mae", "r2", "sample_count"} for row in rows)
