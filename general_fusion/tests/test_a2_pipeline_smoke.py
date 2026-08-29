from __future__ import annotations

from pathlib import Path

from gf.pipeline.a2_benchmark import (
    run_a2_smoke,
    run_a2_oof_diagnostic,
    run_a2_torch_concat_validation,
    run_a2_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_a2_pipeline_smoke_trains_concat_and_deepsets_without_test_metrics() -> None:
    result = run_a2_smoke(project_root=PROJECT_ROOT, max_epochs=2)
    assert result["status"] == "PASS"
    assert set(result["models"]) == {"C1", "M1"}
    assert all(
        payload["validation"]["group_count"] == 180
        for payload in result["models"].values()
    )
    assert result["manifest"]["test_access"]["unlocked"] is False
    assert result["manifest"]["prediction_hash"] is not None


def test_a2_validation_smoke_preserves_five_seed_and_test_lock() -> None:
    result = run_a2_validation(
        project_root=PROJECT_ROOT,
        stage="deepsets",
        max_epochs_override=1,
    )
    assert result["status"] == "EXECUTED"
    assert result["execution_mode"] == "smoke_override"
    assert result["seed_order"] == [17, 29, 43, 71, 101]
    assert set(result["models"]) == {"C1", "M1"}
    assert result["parameter_parity"]["within_tolerance"] is True
    assert result["manifest"]["test_access"]["unlocked"] is False


def test_a2_torch_concat_smoke_has_five_seed_records_and_no_test_unlock() -> None:
    result = run_a2_torch_concat_validation(
        project_root=PROJECT_ROOT,
        max_epochs_override=1,
    )
    assert result["status"] == "EXECUTED"
    assert result["seed_order"] == [17, 29, 43, 71, 101]
    assert len(result["model"]["seed_records"]) == 5
    assert result["gate"]["reference_validation_macro_RNMAE"] == 0.00581589
    assert result["manifest"]["test_access"]["unlocked"] is False


def test_a2_oof_pipeline_smoke_records_grouped_provenance_without_test() -> None:
    result = run_a2_oof_diagnostic(project_root=PROJECT_ROOT, bootstrap_samples=20)
    assert result["status"] == "EXECUTED"
    assert set(result["models"]) == {"B4", "B5"}
    assert all(payload["provenance_rows"] == 1680 for payload in result["models"].values())
    assert result["manifest"]["test_access"]["unlocked"] is False
