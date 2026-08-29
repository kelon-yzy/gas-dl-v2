from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from gf.dl.evaluation import evaluate_predictions
from gf.sim.a1_dataset import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_a1_formal_saved_predictions_reproduce_after_shared_evaluator_extraction() -> None:
    dataset = load_dataset(PROJECT_ROOT / "data" / "a1_formal")
    expected = json.loads(
        (PROJECT_ROOT / "outputs" / "summary" / "a1_formal" / "baseline_summary.json").read_text(
            encoding="utf-8"
        )
    )
    with (PROJECT_ROOT / "outputs" / "summary" / "a1_formal" / "predictions.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(dataset.conditions)
    targets = np.vstack([condition.composition for condition in dataset.conditions]).astype(np.float64)
    groups = np.asarray(dataset.group_ids, dtype=object)
    predictions = np.asarray(
        [
            [
                float(row[f"B5__mean__{target_name}"])
                for target_name in ("x_Ar_pct", "x_He_pct", "x_CO2_pct")
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    validation_indices = np.asarray(
        [index for index, condition in enumerate(dataset.conditions) if condition.split == "val"],
        dtype=np.int64,
    )
    test_indices = np.asarray(
        [index for index, condition in enumerate(dataset.conditions) if condition.split == "test"],
        dtype=np.int64,
    )
    validation = evaluate_predictions(targets, predictions, groups, validation_indices)
    test = evaluate_predictions(targets, predictions, groups, test_indices)

    assert expected["best_overall_full_input"]["key"] == "B5__mean"
    assert validation["macro_RNMAE"] == pytest.approx(
        expected["best_overall_full_input"]["validation_macro_RNMAE"], abs=1e-8
    )
    assert test["macro_RNMAE"] == pytest.approx(
        expected["best_overall_full_input"]["test_macro_RNMAE"], abs=1e-8
    )
