import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gib.pipeline.adaptive_sampling import (
    AdaptiveSamplingError,
    SamplingCheckpoint,
    SamplingObservation,
    freeze_strategy,
    run_c5a,
    run_stage_b,
    select_checkpoint,
    simulate_stage_a,
    validate_sampling_plan,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "configs" / "p3_c5a_sampling_plan.json"


def _plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _checkpoints():
    return (
        SamplingCheckpoint(
            sample_points=50,
            measurement_time_ms=5.0,
            active_modalities=("ndir",),
            estimated_crb_p90=(0.15, 0.15, 0.15, 0.15),
            uncertainty=(0.12, 0.06, 0.15, 0.08),
            delta_i=0.50,
            delta_i_per_cost=0.10,
            cumulative_information_fraction=0.50,
            flops=50.0,
        ),
        SamplingCheckpoint(
            sample_points=75,
            measurement_time_ms=7.5,
            active_modalities=("ndir", "acoustic"),
            estimated_crb_p90=(0.11, 0.11, 0.11, 0.11),
            uncertainty=(0.07, 0.02, 0.09, 0.04),
            delta_i=0.35,
            delta_i_per_cost=0.14,
            cumulative_information_fraction=0.85,
            flops=70.0,
        ),
        SamplingCheckpoint(
            sample_points=100,
            measurement_time_ms=10.0,
            active_modalities=("ndir", "acoustic", "thermal"),
            estimated_crb_p90=(0.10, 0.10, 0.10, 0.10),
            uncertainty=(0.05, 0.02, 0.08, 0.03),
            delta_i=0.15,
            delta_i_per_cost=0.06,
            cumulative_information_fraction=1.0,
            flops=100.0,
        ),
    )


def _observation(cell, split_id="GIB-SPLIT-01", seed=101, index=0):
    band = {"SUF": "sufficient", "CRI": "critical", "INS": "insufficient"}[cell.split("-")[2]]
    truth = (0.25, 0.25, 0.25, 0.25)
    reference = (0.251, 0.251, 0.251, 0.251)
    return SamplingObservation(
        mixture_id=f"GIB-M-{index:016X}",
        sequence_id=f"GIB-Q-{index:016X}",
        grid_cell_id=cell,
        information_band=band,
        split_id=split_id,
        seed=seed,
        truth=truth,
        reference_prediction=reference,
        raw_waveform=np.ones((2, 100), dtype=np.float64),
        checkpoints=_checkpoints(),
        policy_inputs={"measurement_features": [1.0], "native_measurement_cost": 10.0},
    )


def _stage_a_observations():
    return [_observation(cell, index=index) for index, cell in enumerate(_plan()["grid_cells"], start=1)]


def _stage_b_observations():
    observations = []
    index = 1
    for cell in _plan()["grid_cells"]:
        for split_id in _plan()["split_ids"]:
            for seed in _plan()["seeds"]:
                observations.append(_observation(cell, split_id, seed, index))
                index += 1
    return observations


def _stage_a_predictor(observation, checkpoint, method):
    del checkpoint
    if method in {"random_stop", "equal_length_fixed", "crb_rank_shuffle"}:
        return np.asarray(observation.truth) + 0.05
    return observation.reference_prediction


def test_plan_freezes_p2_statistics_gates_and_thresholds_before_results():
    plan = _plan()
    validate_sampling_plan(plan)
    assert plan["statistics"]["resamples"] == 10000
    assert plan["statistics"]["seed"] == 20260824
    assert plan["gates"]["ni_bands"] == {"N2": 0.008, "CO2": 0.003, "O2": 0.01, "Ar": 0.005}
    assert plan["threshold_selection"] == "pre_registered_no_result_tuning"

    drifted = copy.deepcopy(plan)
    drifted["policies"]["crb_dynamic_modality"]["maximum_relative_crb"] = 0.9
    with pytest.raises(AdaptiveSamplingError, match="relative CRB"):
        validate_sampling_plan(drifted)


def test_checkpoint_selection_uses_only_frozen_policy_inputs_and_is_deterministic():
    observation = _stage_a_observations()[0]
    plan = _plan()
    selected = select_checkpoint(observation, "crb_dynamic_modality", plan)
    assert selected.sample_points == 75
    assert select_checkpoint(observation, "crb_dynamic_modality", plan) == selected
    assert select_checkpoint(observation, "full_sampling", plan).sample_points == 100

    leaked = SamplingObservation(
        **{**observation.__dict__, "policy_inputs": {"oracle_features": [1.0]}}
    )
    with pytest.raises(AdaptiveSamplingError, match="oracle fields"):
        simulate_stage_a([leaked, *_stage_a_observations()[1:]], plan, _stage_a_predictor)


def test_stage_a_covers_all_cells_reports_cost_information_and_negative_controls():
    report = simulate_stage_a(_stage_a_observations(), _plan(), _stage_a_predictor)
    assert report["coverage"]["grid_cells"] == sorted(_plan()["grid_cells"])
    assert {"critical", "insufficient"}.issubset(report["coverage"]["information_bands"])
    assert report["candidate_gate_pass"] is True
    assert report["negative_controls_pass"] is True
    assert report["stage_a_gate_pass"] is True
    candidate_rows = report["rows"]["crb_dynamic_modality"]
    assert {row["stop_sample_points"] for row in candidate_rows} == {75}
    assert all(row["delta_i"] > 0 and row["delta_i_per_cost"] > 0 for row in candidate_rows)
    assert report["methods"]["crb_dynamic_modality"]["costs"]["measurement_time_ms"]["ci_lower_reduction"] == pytest.approx(0.25)
    assert all(not control["joint_gate_pass"] for control in report["negative_controls"].values())


def test_stage_a_failure_rejects_without_calling_stage_b():
    called = False

    def poor_predictor(observation, checkpoint, method):
        del checkpoint, method
        return np.asarray(observation.truth) + 0.1

    def forbidden_runner(observation, checkpoint, truncated):
        del observation, checkpoint, truncated
        nonlocal called
        called = True
        raise AssertionError("stage B must not run")

    result = run_c5a(_stage_b_observations(), _plan(), poor_predictor, forbidden_runner)
    assert result["candidate_verdict"] == "reject"
    assert result["stage_b"] == {"status": "not_run", "reason": "stage_a_gate_failed"}
    assert called is False


def test_stage_b_physically_truncates_with_unchanged_frozen_strategy_and_covers_9x5x3():
    plan = _plan()
    stage_a = simulate_stage_a(_stage_a_observations(), plan, _stage_a_predictor)
    strategy = freeze_strategy(stage_a, plan)
    seen_shapes = []

    def runner(observation, checkpoint, truncated):
        seen_shapes.append(truncated.shape)
        assert truncated.shape[-1] == checkpoint.sample_points == 75
        assert not np.shares_memory(truncated, observation.raw_waveform)
        return observation.reference_prediction

    report = run_stage_b(_stage_b_observations(), plan, strategy, runner)
    assert report["coverage"]["grid_split_seed_count"] == 135
    assert len(report["strata"]) == 135
    assert report["gate"]["joint_gate_pass"] is True
    assert len(seen_shapes) == 135

    changed = copy.deepcopy(strategy)
    changed["policy"]["maximum_relative_crb"] = 1.3
    with pytest.raises(AdaptiveSamplingError, match="differs"):
        run_stage_b(_stage_b_observations(), plan, changed, runner)


def test_terminal_verdict_requires_stage_b_to_clear_the_same_gate():
    result = run_c5a(
        _stage_b_observations(),
        _plan(),
        _stage_a_predictor,
        lambda observation, checkpoint, truncated: observation.reference_prediction,
    )
    assert result["candidate_verdict"] == "enter_P4"
    assert result["frozen_strategy"]["threshold_selection"] == "pre_registered_no_result_tuning"
    assert result["stage_a"]["stage_a_gate_pass"] is True
    assert result["stage_b"]["gate"]["joint_gate_pass"] is True
    json.dumps(result, allow_nan=False)
