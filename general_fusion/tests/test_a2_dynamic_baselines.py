from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gf.dl.temporal_baselines import (
    CausalGRU,
    CausalTCN,
    causal_sequence_matrix,
    fit_causal_neural_model,
    fit_formal_classical_model,
    formal_feature_vector,
)
from gf.pipeline.a2_dynamic_baselines import (
    assert_a2_dynamic_test_unlocked,
    run_a2_dynamic_handoff,
)
import gf.pipeline.a2_dynamic_benchmark as a2_dynamic_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_formal_features_are_prefix_causal() -> None:
    prefix = np.arange(30 * 3, dtype=np.float64).reshape(30, 3)
    extended = np.vstack((prefix, np.full((8, 3), 9999.0)))
    for model_id in ("B-LAST", "B-DELTA", "B-EWMA", "B-STAT"):
        assert np.array_equal(
            formal_feature_vector(model_id, prefix),
            formal_feature_vector(model_id, extended[:30]),
        )


def test_causal_sequence_matrix_never_reads_after_endpoint() -> None:
    signals = np.zeros((1, 3, 20, 1), dtype=np.float32)
    signals[0, :, :10, 0] = np.arange(30, dtype=np.float32).reshape(3, 10)
    signals[0, :, 10:, 0] = 1000.0
    first = causal_sequence_matrix(signals, np.asarray([0]), np.asarray([9]), sequence_length=16)
    changed = signals.copy()
    changed[0, :, 10:, 0] = -1000.0
    second = causal_sequence_matrix(changed, np.asarray([0]), np.asarray([9]), sequence_length=16)
    assert np.array_equal(first, second)


def test_formal_classical_and_neural_models_produce_finite_outputs() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(16, 3))
    targets = rng.uniform(0.0, 100.0, size=(16, 3))
    predictions, diagnostics, _ = fit_formal_classical_model(
        "B-LAST", features, targets, [features[:4]], seed=17
    )
    assert diagnostics["status"] == "PASS"
    assert predictions[0].shape == (4, 3)
    sequences = rng.normal(size=(16, 12, 3)).astype(np.float32)
    for model_id, expected_type in (("B-TCN", CausalTCN), ("B-GRU", CausalGRU)):
        outputs, diagnostics, model = fit_causal_neural_model(
            model_id, sequences, targets, [sequences[:4]], seed=17, epochs=1
        )
        assert diagnostics["status"] == "PASS"
        assert outputs[0].shape == (4, 3)
        assert isinstance(model, expected_type)


def test_dynamic_test_gate_rejects_failed_freeze_status() -> None:
    with pytest.raises(ValueError, match="locked"):
        assert_a2_dynamic_test_unlocked({"status": "DATA_FREEZE_FAILED"})
    assert_a2_dynamic_test_unlocked({"status": "DATA_FROZEN"}) is None


def test_baseline_artifact_has_development_only_scope_when_available() -> None:
    path = PROJECT_ROOT / "outputs" / "summary" / "a2_dynamic_v1" / "a2_dyn_5_baselines.json"
    if not path.is_file():
        pytest.skip("A2-DYN-5 baseline stage has not been executed in this checkout")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["status"] == "DEVELOPMENT_BASELINES_COMPLETE"
    assert summary["test_access"]["test_rows_read"] == 0
    assert summary["data_manifest_status"] != "DATA_FROZEN"
    assert set(summary["prediction_scope"]) == {"train", "val", "stress_val"}
    assert summary["new_algorithm_handoff_allowed"] is False


def test_a2_dynamic_handoff_records_freeze_block_without_generating_handoff() -> None:
    closure_path = PROJECT_ROOT / "outputs" / "summary" / "a2_dynamic_v1" / "a2_dyn_6_closure.json"
    required = [
        PROJECT_ROOT / "outputs" / "summary" / "a2_dynamic_v1" / name
        for name in (
            "a2_dyn_5_baselines.json",
            "a2_dyn_5_replay_smoke.json",
            "a2_dyn_5_report.json",
            "a2_dyn_3r2_audit.json",
            "a2_dyn_4r2_freeze_audit.json",
        )
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("A2-DYN upstream artifacts are not available in this checkout")
    result = run_a2_dynamic_handoff(PROJECT_ROOT)
    assert result["stage"] == "A2-DYN-6"
    assert result["status"] == "A2_DYN_6_BLOCKED_DATA_FREEZE_FAILED"
    assert result["handoff_generated"] is False
    assert result["new_algorithm_handoff_allowed"] is False
    assert result["algorithm_search_allowed"] is False
    assert result["formal_gate_status"] == "BLOCKED_DATA_FREEZE_FAILED"
    assert closure_path.is_file()
    assert not (closure_path.parent / "dynamic_handoff.json").is_file()
    assert json.loads(closure_path.read_text(encoding="utf-8")) == result
    assert "handoff" in a2_dynamic_benchmark.PLANNED_STAGES
