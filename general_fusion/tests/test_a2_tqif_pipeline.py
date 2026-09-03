from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gf.pipeline.a2_tqif_benchmark import (
    _build_seeded_model,
    _comparison_gate,
    _select_capacity_recipe,
    run_tqif,
    run_tqif_protocol,
)
from gf.pipeline.tqif_common import TQIFArtifactError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_model_construction_seed_is_independent_of_prior_stage_rng() -> None:
    torch.manual_seed(901)
    torch.rand(37)
    first = _build_seeded_model(lambda: torch.nn.Linear(3, 2), seed=17)

    torch.manual_seed(1234)
    torch.rand(11)
    second = _build_seeded_model(lambda: torch.nn.Linear(3, 2), seed=17)

    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_baseline_before_protocol_is_blocked_by_prerequisite(tmp_path: Path) -> None:
    with pytest.raises(TQIFArtifactError) as error:
        run_tqif(project_root=tmp_path, stage="baseline")
    assert error.value.code == "PREREQUISITE_NOT_PASSED"


def test_protocol_rejects_old_test_path_before_reading_data(tmp_path: Path) -> None:
    config = json.loads(
        (PROJECT_ROOT / "configs" / "experiment" / "a2_tqif_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    config["data_manifest"] = "data/a1_formal_test/manifest.json"
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(TQIFArtifactError) as error:
        run_tqif_protocol(
            project_root=PROJECT_ROOT,
            protocol_config_path=protocol_path,
        )
    assert error.value.code == "PROTOCOL_ACCESS_VIOLATION"


def test_protocol_artifact_records_parameter_and_source_contracts() -> None:
    path = PROJECT_ROOT / "outputs" / "runs" / "a2" / "tqif" / "protocol_manifest.json"
    if not path.is_file():
        pytest.skip("protocol stage has not been executed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "tqif-protocol-1"
    assert manifest["status"] == "PASS"
    assert manifest["source"]["git_commit"]
    assert manifest["source"]["git_dirty"] is True
    assert manifest["source"]["source_diff_hash"]
    for profile in manifest["parameter_profiles"]["recipes"].values():
        assert all(
            comparison["relative_difference"] <= 0.10
            for comparison in profile["comparisons"].values()
        )


def test_successful_protocol_is_idempotent_without_overwrite() -> None:
    path = PROJECT_ROOT / "outputs" / "runs" / "a2" / "tqif" / "protocol_manifest.json"
    if not path.is_file():
        pytest.skip("protocol stage has not been executed")
    before = path.read_bytes()
    result = run_tqif_protocol(project_root=PROJECT_ROOT)
    assert result["status"] == "PASS"
    assert path.read_bytes() == before


def test_comparison_gate_uses_all_five_seeds_and_paired_group_bootstrap(
    tmp_path: Path,
) -> None:
    groups = tuple(f"mix-{index}" for index in range(8))
    targets = np.zeros((len(groups), 3), dtype=np.float64)
    candidate_records = []
    baseline_records = []
    for seed in (17, 29, 43, 71, 101):
        candidate_path = tmp_path / f"candidate-{seed}.csv"
        baseline_path = tmp_path / f"baseline-{seed}.csv"
        _write_predictions(candidate_path, groups, prediction=0.8)
        _write_predictions(baseline_path, groups, prediction=1.0)
        candidate_records.append(
            _comparison_record("TQIF-H0", seed, candidate_path.name, macro=0.008)
        )
        baseline_records.append(
            _comparison_record("C0", seed, baseline_path.name, macro=0.010)
        )
    result = _comparison_gate(
        candidate_records,
        baseline_records,
        context={"root": tmp_path, "targets": targets, "groups": groups},
        eval_config={
            "bootstrap": {"seed": 123, "repeats": 2000},
            "promotion": {
                "min_macro_relative_gain": 0.05,
                "min_seed_same_direction": 4,
                "max_component_absolute_rnmae_degradation": 0.005,
            },
        },
    )
    assert result["status"] == "PASS"
    assert result["same_direction_seed_count"] == 5
    assert result["paired_group_bootstrap"]["percentile_97_5"] < 0.0


def test_capacity_selection_prefers_token16_when_registered_evidence_ties() -> None:
    comparisons = {
        "TQIF-H0__vs__C0::tqif_token16_pair16": {
            "status": "PASS",
            "relative_improvement": 0.1,
            "median_seed_improvement": 0.001,
        },
        "TQIF-H0__vs__C0::tqif_token32_pair32": {
            "status": "PASS",
            "relative_improvement": 0.1,
            "median_seed_improvement": 0.001,
        },
    }
    models = {
        "TQIF-H0::tqif_token16_pair16": {
            "worst_seed_validation_macro_RNMAE": 0.01,
            "parameter_count": 6000,
        },
        "TQIF-H0::tqif_token32_pair32": {
            "worst_seed_validation_macro_RNMAE": 0.01,
            "parameter_count": 23000,
        },
    }
    assert _select_capacity_recipe(comparisons, models) == "tqif_token16_pair16"


def _comparison_record(
    model_id: str,
    seed: int,
    prediction_path: str,
    *,
    macro: float,
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "recipe_id": "recipe",
        "seed": seed,
        "prediction_path": prediction_path,
        "validation": {
            "macro_RNMAE": macro,
            "component_RNMAE": [macro, macro, macro],
        },
        "resources": {"parameter_count": 100},
    }


def _write_predictions(path: Path, groups: tuple[str, ...], *, prediction: float) -> None:
    fieldnames = [
        "split",
        "mixture_id",
        "pred_x_Ar_pct",
        "pred_x_He_pct",
        "pred_x_CO2_pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            writer.writerow(
                {
                    "split": "val",
                    "mixture_id": group,
                    "pred_x_Ar_pct": prediction,
                    "pred_x_He_pct": prediction,
                    "pred_x_CO2_pct": prediction,
                }
            )
