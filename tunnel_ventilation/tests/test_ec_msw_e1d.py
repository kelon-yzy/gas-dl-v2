from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.dl.evaluation.ec_msw_e1d_diagnosis import (
    E1dFeatureSpec,
    _build_summary,
    _build_verdict,
    _diagnostic_feature_count,
    _parity_split_gate,
    _positive_control_split_gate,
    _validate_config,
    build_e1d_feature_matrix,
    default_e1d_specs,
    run_ec_msw_e1d_diagnosis,
)
from tv3.pipeline.build_tv3_raw_dsp_features import (
    build_tv3_raw_dsp_feature_cache,
    preflight_tv3_raw_dsp_dataset,
)
from tv3.sim.generation.tunnel_ventilation import (
    TunnelVentilationBenchmarkGenerationSpec,
    generate_tunnel_ventilation_benchmark_dataset,
)


def _make_dataset(tmp_path: Path, *, sequences: int = 16) -> Path:
    generate_tunnel_ventilation_benchmark_dataset(
        tmp_path,
        TunnelVentilationBenchmarkGenerationSpec(
            dataset_slug="tv3-e1d-smoke",
            sequence_count=sequences,
            seed=20260714,
            timesteps=16,
            storage="npz",
            optical_absorption_backend="empirical_v1",
            workers=1,
        ),
    )
    dataset_dir = tmp_path / "tv3-e1d-smoke"
    preflight = preflight_tv3_raw_dsp_dataset(dataset_dir)
    build_tv3_raw_dsp_feature_cache(
        preflight,
        cache_dir=dataset_dir / "features" / "raw_dsp" / "raw_dsp_frame_v1",
        template_mode="train_baseline_median",
        workers=1,
    )
    return dataset_dir


@pytest.fixture(scope="module")
def e1d_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _make_dataset(tmp_path_factory.mktemp("e1d_dataset"))


class TestE1dFeatureSpecs:
    def test_default_ladder_covers_e1d_stages(self):
        specs = default_e1d_specs()
        stages = {spec.stage for spec in specs}
        assert stages == {"E1d-0", "E1d-1", "E1d-2", "E1d-3"}
        assert specs[0].is_full_b1
        assert any(spec.name == "peak_lmm" for spec in specs)
        assert any(spec.name == "peak_stats7_phase" for spec in specs)
        assert any(
            spec.name == "e1r_sequence_embedding"
            and spec.representation_source == "e1r_sequence"
            for spec in specs
        )
        assert any(
            spec.name == "e1r_peak_lmm" and spec.representation_source == "e1r_peak"
            for spec in specs
        )

    def test_peak_lmm_feature_matrix_shapes(self, e1d_dataset: Path):
        spec = next(item for item in default_e1d_specs() if item.name == "peak_lmm")
        matrix = build_e1d_feature_matrix(e1d_dataset, split="train", spec=spec)
        assert matrix.x.ndim == 2
        assert matrix.x.shape[0] == len(matrix.sequence_ids)
        assert matrix.x.shape[1] == len(matrix.feature_names)
        assert any(name.endswith("ultrasonic_peak_index_raw_dsp:last") for name in matrix.feature_names)
        assert any(name.startswith("full|slow:") for name in matrix.feature_names)
        assert np.isfinite(matrix.x).all()

    def test_sequence_scalar_features_append(self, e1d_dataset: Path):
        spec = E1dFeatureSpec(
            name="peak_plus_delay",
            stage="E1d-2",
            role="unit",
            frame_arrays=("ultrasonic_peak_index_raw_dsp",),
            sequence_scalars=("ultrasonic_delay_calibration_s",),
            sequence_statistics=("last", "mean", "max"),
        )
        matrix = build_e1d_feature_matrix(e1d_dataset, split="val", spec=spec)
        assert "seq|ultrasonic_delay_calibration_s" in matrix.feature_names
        assert matrix.x.shape[1] == len(matrix.feature_names)


class TestE1dGatesAndVerdict:
    def test_parity_gate_thresholds(self):
        candidate = {
            "x_CO2": {"r2": 0.99},
            "x_O2": {"r2": 0.40},
            "x_N2": {"r2": 0.86},
        }
        reference = {
            "x_CO2": {"r2": 0.99},
            "x_O2": {"r2": 0.43},
            "x_N2": {"r2": 0.88},
        }
        gate = _parity_split_gate(
            candidate,
            reference,
            {"o2_r2_drop_max": 0.05, "co2_n2_r2_drop_max": 0.03},
        )
        assert gate["passed"] is True
        assert gate["r2_delta_vs_control"]["x_O2"] == pytest.approx(-0.03)

        fail = _parity_split_gate(
            {
                "x_CO2": {"r2": 0.90},
                "x_O2": {"r2": 0.10},
                "x_N2": {"r2": 0.80},
            },
            reference,
            {"o2_r2_drop_max": 0.05, "co2_n2_r2_drop_max": 0.03},
        )
        assert fail["passed"] is False

    def test_positive_control_requires_exact_reproduction(self):
        reference = {
            "x_CO2": {"r2": 0.99},
            "x_O2": {"r2": 0.43},
            "x_N2": {"r2": 0.88},
        }
        same = _positive_control_split_gate(reference, reference, 1e-6)
        assert same["passed"] is True
        drifted = _positive_control_split_gate(
            {**reference, "x_O2": {"r2": 0.429}},
            reference,
            1e-6,
        )
        assert drifted["passed"] is False

    def test_verdict_prefers_compact_set(self):
        summary = {
            "compact_parity_passing_sets": ["peak_stats7_phase"],
            "parity_passing_sets": ["peak_stats7_phase", "cal_plus_quality_full"],
            "positive_control_passed": True,
        }
        verdict = _build_verdict(summary, {"o2_r2_drop_max": 0.05, "co2_n2_r2_drop_max": 0.03})
        assert verdict["status"] == "minimal_deployable_set_found"
        assert verdict["e2_allowed"] is False
        assert verdict["continue_structured_builder"] is True

    def test_verdict_stops_when_only_near_full_b1_passes(self):
        summary = {
            "compact_parity_passing_sets": [],
            "parity_passing_sets": ["cal_plus_quality_full"],
            "positive_control_passed": True,
        }
        verdict = _build_verdict(summary, {"o2_r2_drop_max": 0.05, "co2_n2_r2_drop_max": 0.03})
        assert verdict["status"] == "only_near_full_b1_passes"
        assert verdict["e2_allowed"] is False
        assert verdict["continue_structured_builder"] is False

    def test_verdict_blocks_when_positive_control_fails(self):
        verdict = _build_verdict(
            {
                "compact_parity_passing_sets": ["peak_lmm"],
                "parity_passing_sets": ["peak_lmm"],
                "positive_control_passed": False,
            },
            {"o2_r2_drop_max": 0.05, "co2_n2_r2_drop_max": 0.03},
        )
        assert verdict["status"] == "positive_control_failed"
        assert verdict["continue_structured_builder"] is False

    def test_summary_compactness_excludes_frozen_slow_features(self):
        full = _summary_entry(
            "full_b1",
            is_full_b1=True,
            feature_count=1008,
            diagnostic_feature_count=504,
        )
        compact = _summary_entry(
            "peak_lmm",
            is_full_b1=False,
            feature_count=507,
            diagnostic_feature_count=3,
        )
        summary = _build_summary((full, compact), DEFAULT_GATES, True)
        assert summary["positive_control_passed"] is True
        assert summary["compact_parity_passing_sets"] == ["peak_lmm"]
        assert _diagnostic_feature_count(
            ("full|slow:T_C:mean", "full|physics:peak:last")
        ) == 1

    def test_summary_rejects_missing_eval_split(self):
        entry = _summary_entry(
            "full_b1",
            is_full_b1=True,
            feature_count=1008,
            diagnostic_feature_count=504,
        )
        del entry["splits"]["extrapolation"]
        with pytest.raises(ValueError, match="missing required eval splits"):
            _build_summary((entry,), DEFAULT_GATES, True)

    def test_config_rejects_partial_eval_protocol(self):
        with pytest.raises(ValueError, match="eval_splits must be exactly"):
            _validate_config(
                {
                    "dataset_dir": "data/example",
                    "output_dir": "outputs/example",
                    "run_kind": "smoke",
                    "ridge_alphas": [0.1],
                    "eval_splits": ["val"],
                    "feature_sets": ["full_b1"],
                }
            )


def test_run_e1d_smoke_writes_artifacts(e1d_dataset: Path, tmp_path: Path):
    config_path = tmp_path / "e1d_config.json"
    output_dir = tmp_path / "e1d_out"
    config_path.write_text(
        json.dumps(
            {
                "dataset_dir": str(e1d_dataset),
                "output_dir": str(output_dir),
                "run_kind": "smoke",
                "ridge_alphas": [0.01, 0.1, 1.0, 10.0],
                "feature_sets": [
                    "full_b1",
                    "peak_lmm",
                    "peak_stats7",
                    "peak_stats7_phase",
                    "peak_phase_plus_delay",
                    "cal_plus_quality_full",
                ],
            }
        ),
        encoding="utf-8",
    )

    written = run_ec_msw_e1d_diagnosis(config_path, project_root=tmp_path)
    assert written == output_dir
    for name in (
        "manifest.json",
        "feature_sets.json",
        "summary.json",
        "verdict.json",
        "ablation_table.csv",
        "narrow_o2_windows.csv",
    ):
        assert (output_dir / name).is_file()

    verdict = json.loads((output_dir / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["e2_allowed"] is False
    assert verdict["status"] == "smoke_only"

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert "E1d-0" in summary["stages"]
    assert "E1d-1" in summary["stages"]
    assert summary["full_b1_feature_count"] is not None

    feature_sets = json.loads((output_dir / "feature_sets.json").read_text(encoding="utf-8"))
    assert len(feature_sets["feature_sets"]) == 6
    for entry in feature_sets["feature_sets"]:
        for split in ("val", "test", "extrapolation"):
            assert split in entry["splits"]
            for component in ("x_CO2", "x_O2", "x_N2"):
                metrics = entry["splits"][split]["component_metrics"][component]
                assert "r2" in metrics
                assert "mae" in metrics
                assert "bias" in metrics


DEFAULT_GATES = {"o2_r2_drop_max": 0.05, "co2_n2_r2_drop_max": 0.03}


def _summary_entry(
    name: str,
    *,
    is_full_b1: bool,
    feature_count: int,
    diagnostic_feature_count: int,
) -> dict[str, object]:
    splits = {}
    for split in ("val", "test", "extrapolation"):
        splits[split] = {
            "component_metrics": {
                "x_CO2": {"r2": 0.99},
                "x_O2": {"r2": 0.45},
                "x_N2": {"r2": 0.88},
            },
            "gate": {"passed": True, "r2_delta_vs_control": {}},
        }
    return {
        "name": name,
        "stage": "E1d-0" if is_full_b1 else "E1d-1",
        "role": "test",
        "is_full_b1": is_full_b1,
        "feature_count": feature_count,
        "diagnostic_feature_count": diagnostic_feature_count,
        "splits": splits,
    }
