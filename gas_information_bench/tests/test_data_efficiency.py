import json
from pathlib import Path

import numpy as np
import pytest

from gib.contract import ContractError
from gib.pipeline.baseline import PilotMetadata
from gib.pipeline import data_efficiency as data_efficiency_module
from gib.pipeline.data_efficiency import (
    METHOD_IDS,
    NestedSplit,
    evaluate_candidate_verdict,
    make_model,
    paired_group_p90_ci,
    paired_reduction_ci,
    random_non_nested_groups,
    validate_data_efficiency_plan,
    validate_nested_group_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _plan():
    return json.loads((ROOT / "configs" / "p3_c5b_data_efficiency_plan.json").read_text(encoding="utf-8"))


def _metadata():
    records = [
        {"sequence_id": f"Q{index}", "mixture_id": mixture, "grade": {"grid_cell_id": "CELL"}}
        for index, mixture in enumerate(("M1", "M2", "M3", "M4", "M5", "M6", "M7"))
    ]
    split_rows = [
        {"sequence_id": record["sequence_id"], "mixture_id": record["mixture_id"], "split_id": "S1", "partition": partition}
        for record, partition in zip(records, ("train", "train", "train", "train", "train", "val", "test"))
    ]
    return PilotMetadata(
        root=Path("."), records=records, deployment=[], split_rows=split_rows,
        sequence_to_index={record["sequence_id"]: index for index, record in enumerate(records)},
        cell_ids=np.asarray(["CELL"] * len(records)),
        mixture_ids=np.asarray([record["mixture_id"] for record in records]), generation_summary={},
    )


def test_plan_freezes_methods_nested_groups_statistics_and_timing():
    plan = _plan()
    validate_data_efficiency_plan(plan)
    assert tuple(plan["models"]) == METHOD_IDS
    assert plan["feature_contract"]["source"] == "gib.pipeline.baseline.deployment_features"
    assert plan["feature_contract"]["oracle_features_allowed"] is False
    assert plan["selection"]["selection_space_locked"] is True
    assert plan["timing"]["formal_training_fraction"] == 100
    assert plan["timing"]["formal_method_scope"] == ["fo_mplselm", "validation_selected_reference"]
    assert plan["execution"]["checkpoint_unit"] == ["split_id", "seed", "training_fraction"]
    broken = json.loads(json.dumps(plan))
    broken["timing"]["independent_repeats"] = 1
    with pytest.raises(ContractError, match="timing profile"):
        validate_data_efficiency_plan(broken)


def test_nested_contract_rejects_reordered_and_cross_partition_groups():
    metadata = _metadata()
    valid = NestedSplit(
        train_group_order=("M1", "M2", "M3", "M4", "M5"),
        train_prefixes={10: ("M1",), 25: ("M1", "M2"), 50: ("M1", "M2", "M3"), 75: ("M1", "M2", "M3", "M4"), 100: ("M1", "M2", "M3", "M4", "M5")},
        val_mixture_ids=("M6",), test_mixture_ids=("M7",),
    )
    validate_nested_group_contract(metadata, {"S1": valid}, (10, 25, 50, 75, 100))
    broken = NestedSplit(
        train_group_order=valid.train_group_order,
        train_prefixes={**valid.train_prefixes, 50: ("M1", "M3", "M2")},
        val_mixture_ids=valid.val_mixture_ids, test_mixture_ids=valid.test_mixture_ids,
    )
    with pytest.raises(ContractError, match="not frozen nested prefixes"):
        validate_nested_group_contract(metadata, {"S1": broken}, (10, 25, 50, 75, 100))


def test_random_non_nested_control_is_deterministic_and_train_only():
    specification = NestedSplit(
        train_group_order=("M1", "M2", "M3", "M4", "M5"),
        train_prefixes={10: ("M1",), 25: ("M1", "M2"), 50: ("M1", "M2", "M3"), 75: ("M1", "M2", "M3", "M4"), 100: ("M1", "M2", "M3", "M4", "M5")},
        val_mixture_ids=("M6",), test_mixture_ids=("M7",),
    )
    first = random_non_nested_groups(specification, 50, 101)
    assert first == random_non_nested_groups(specification, 50, 101)
    assert len(first) == len(specification.train_prefixes[50])
    assert set(first) <= set(specification.train_group_order)
    assert not set(first) & set(specification.val_mixture_ids + specification.test_mixture_ids)


def test_fo_mplselm_is_fixed_deterministic_and_controls_are_distinct():
    plan = _plan()
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 8))
    y = rng.normal(size=(24, 4))
    fisher = np.asarray([0.5, 1.0, 2.0, 4.0])
    first = make_model("fo_mplselm", plan, 101, fisher_weights=fisher).fit(x, y).predict(x)
    second = make_model("fo_mplselm", plan, 101, fisher_weights=fisher).fit(x, y).predict(x)
    no_fisher = make_model("fo_mplselm", plan, 101, fisher_weights=fisher, control="without_fisher_weights").fit(x, y).predict(x)
    random_direction = make_model("fo_mplselm", plan, 101, fisher_weights=fisher, control="random_orthogonal_directions").fit(x, y).predict(x)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, no_fisher)
    assert not np.array_equal(first, random_direction)


def test_paired_bootstrap_uses_groups_and_timing_requires_aligned_repeats():
    truth = np.zeros(6)
    candidate = np.asarray([0.1, 0.1, 0.2, 0.2, 0.3, 0.3])
    reference = candidate + 0.1
    groups = ["M1", "M1", "M2", "M2", "M3", "M3"]
    first = paired_group_p90_ci(truth, candidate, reference, groups, resamples=200, seed=10)
    second = paired_group_p90_ci(truth, candidate, reference, groups, resamples=200, seed=10)
    assert first == second
    assert first[0] < 0.0
    reduction = paired_reduction_ci([8, 16, 24], [10, 20, 30], resamples=200, seed=10)
    assert reduction[0] == pytest.approx(0.2)
    with pytest.raises(ContractError, match="aligned positive"):
        paired_reduction_ci([1], [1, 2], resamples=10, seed=1)


def test_vectorized_group_bootstrap_matches_its_frozen_draws():
    truth = np.zeros(6)
    candidate = np.asarray([0.1, 0.2, 0.2, 0.4, 0.3, 0.6])
    reference = candidate + 0.1
    groups = ["M1", "M1", "M2", "M2", "M3", "M3"]
    point, lower, upper = paired_group_p90_ci(truth, candidate, reference, groups, resamples=50, seed=12)
    rng = np.random.default_rng(12)
    positions = np.asarray([[0, 1], [2, 3], [4, 5]])
    indices = positions[rng.integers(0, 3, size=(50, 3))].reshape(50, -1)
    expected = np.quantile(np.abs(candidate)[indices], 0.9, axis=1, method="higher") - np.quantile(
        np.abs(reference)[indices], 0.9, axis=1, method="higher"
    )
    assert point == pytest.approx(-0.1)
    assert lower == pytest.approx(np.quantile(expected, 0.025))
    assert upper == pytest.approx(np.quantile(expected, 0.975))


def test_variable_size_group_bootstrap_matches_scalar_frozen_draws():
    truth = np.zeros(6)
    candidate = np.asarray([0.1, 0.2, 0.2, 0.4, 0.3, 0.6])
    reference = candidate + 0.1
    groups = np.asarray(["M1", "M2", "M2", "M3", "M3", "M3"])
    point, lower, upper = paired_group_p90_ci(truth, candidate, reference, groups, resamples=40, seed=13)
    rng = np.random.default_rng(13)
    unique = ("M1", "M2", "M3")
    expected = []
    for draw in rng.integers(0, 3, size=(40, 3)):
        indices = np.concatenate([np.flatnonzero(groups == unique[index]) for index in draw])
        expected.append(
            np.quantile(np.abs(candidate)[indices], 0.9, method="higher")
            - np.quantile(np.abs(reference)[indices], 0.9, method="higher")
        )
    assert point == pytest.approx(-0.1)
    assert lower == pytest.approx(np.quantile(expected, 0.025))
    assert upper == pytest.approx(np.quantile(expected, 0.975))


def test_checkpoint_payload_hash_detects_tampering(tmp_path):
    payload = {"unit_id": "S1__seed-101__fraction-100", "rows": [{"value": 1.0}], "elapsed_ns": 1}
    wrapper = data_efficiency_module._checkpoint_wrapper(payload["unit_id"], payload)
    path = tmp_path / "unit.json"
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    assert data_efficiency_module._read_checkpoint(path, payload["unit_id"]) == payload
    wrapper["payload"]["rows"][0]["value"] = 2.0
    path.write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(ContractError, match="hash mismatch"):
        data_efficiency_module._read_checkpoint(path, payload["unit_id"])


def test_formal_unit_repeats_only_candidate_and_selected_reference(monkeypatch):
    plan = _plan()
    metadata = _metadata()
    split = NestedSplit(
        train_group_order=("M1", "M2", "M3", "M4", "M5"),
        train_prefixes={10: ("M1",), 25: ("M1", "M2"), 50: ("M1", "M2", "M3"), 75: ("M1", "M2", "M3", "M4"), 100: ("M1", "M2", "M3", "M4", "M5")},
        val_mixture_ids=("M6",),
        test_mixture_ids=("M7",),
    )
    fit_calls = []

    class FakeModel:
        def __init__(self, method_id):
            self.method_id = method_id

        def fit(self, x, y):
            fit_calls.append(self.method_id)
            return self

        def predict(self, x):
            return np.zeros((len(x), 4), dtype=np.float64)

    monkeypatch.setattr(data_efficiency_module, "_labels", lambda metadata, indices: np.zeros((len(indices), 4)))
    monkeypatch.setattr(data_efficiency_module, "fisher_target_weights", lambda *args, **kwargs: np.ones(4))
    monkeypatch.setattr(data_efficiency_module, "make_model", lambda method_id, *args, **kwargs: FakeModel(method_id))
    monkeypatch.setattr(
        data_efficiency_module,
        "_metrics",
        lambda truth, prediction, components: [
            {"component": component, "p90": 0.0, "rmse": 0.0, "mae": 0.0, "r2": 1.0}
            for component in components
        ],
    )
    monkeypatch.setattr(data_efficiency_module, "_inference_timings", lambda *args, **kwargs: [])

    payload = data_efficiency_module._run_data_efficiency_unit(
        metadata=metadata,
        split_id="S1",
        split=split,
        seed=101,
        fraction=100,
        features=np.ones((7, 3), dtype=np.float64),
        config=plan,
        execution_fingerprint_sha256="F" * 64,
        fisher_matrix_cache={},
        label_cache={},
    )
    assert len(fit_calls) == 7 + 29 * 2 + 3
    assert len(payload["timing_records"]) == 7 + 29 * 2


def test_verdict_has_enter_reject_and_inconclusive_paths():
    plan = _plan()
    ni = [{"component": component, "ci_upper": 0.0, "status": "complete"} for component in plan["components"]]
    timing = [
        {"metric": "training_wall_clock", "ci_lower": 0.21, "status": "complete"},
        {"metric": "batch_size_1_latency", "ci_lower": 0.0, "status": "complete"},
    ]
    controls = {
        "random_orthogonal_directions": {"passed": True},
        "without_fisher_weights": {"passed": True},
        "random_non_nested_subset": {"passed": True},
    }
    passed = evaluate_candidate_verdict(
        ni_rows=ni, first_target_fraction=75, timing_rows=timing, negative_controls=controls,
        config=plan, equivalent_to_plselm=False,
    )
    assert passed["candidate_verdict"] == "enter_P4"
    rejected = evaluate_candidate_verdict(
        ni_rows=ni, first_target_fraction=75, timing_rows=timing, negative_controls=controls,
        config=plan, equivalent_to_plselm=True,
    )
    assert rejected["candidate_verdict"] == "reject"
    inconclusive = evaluate_candidate_verdict(
        ni_rows=[], first_target_fraction=None, timing_rows=timing, negative_controls=controls,
        config=plan, equivalent_to_plselm=False,
    )
    assert inconclusive["candidate_verdict"] == "inconclusive"
