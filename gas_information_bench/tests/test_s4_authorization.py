import json
from pathlib import Path


REGISTRY_PATH = Path(__file__).parents[1] / "configs" / "p2_s4_metric_registry.json"


def _registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def test_h1_authorization_is_complete_and_component_specific() -> None:
    registry = _registry()
    revision = registry["authorization"]["p2_proposed_revision"]

    assert registry["authorization_status"] == "authorized"
    assert revision["verdict"] == "authorized"
    assert revision["authorized_date"] == "2026-08-25"
    assert revision["non_inferiority_band"]["value"] == {
        "N2": 0.008,
        "CO2": 0.003,
        "O2": 0.01,
        "Ar": 0.005,
    }


def test_efficiency_and_timing_gates_are_directly_judgeable() -> None:
    registry = _registry()
    revision = registry["authorization"]["p2_proposed_revision"]
    gates = revision["efficiency_gate"]["value"]
    repeats = revision["timing_repetition_policy"]["value"]

    assert gates["minimum_relative_reduction"] == {
        "iterations": 0.3,
        "forward_calls": 0.3,
        "solver_wall_clock": 0.2,
        "single_sample_latency": 0.2,
    }
    assert gates["maximum_relative_regression_for_other_primary_metrics"] == 0.05
    assert repeats["independent_repeats"] == 30
    assert repeats["warmup_batches"] == 10
    assert repeats["timed_batches_per_repeat"] == 30
    assert registry["training_timing"]["exact_repeat_count"]["value"] == 30


def test_data_efficiency_target_is_absolute_not_non_inferiority_delta() -> None:
    registry = _registry()
    target = registry["data_efficiency"]["precision_target_point"]

    assert target["value"] == {
        "N2": 0.08,
        "CO2": 0.03,
        "O2": 0.1,
        "Ar": 0.05,
    }
    assert target["source"] == "configs/p2_s1_grid.json target_error_tau"
    assert target["unit"] == "mol_per_mol_absolute_p90"
