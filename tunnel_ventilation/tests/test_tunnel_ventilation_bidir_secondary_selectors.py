"""F5-S unit tests: bidir AB SPXY profile, criterion-d 12-cell gate, recompute adapter."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tv3.ml.bidir_f5_secondary import (
    F5S_X_PROFILE,
    audit_derived_split_summary,
    evaluate_criterion_d,
    expected_x_feature_digest,
    f5s_split_relpath,
)
from tv3.sim.packaging.spxy_split import (
    BIDIR_SPXY_OBSERVED_AB_EXPECTED_DIM as DIM,
    SPXY_X_PROFILE_BIDIR_OBSERVED_AB_V1,
    _build_spxy_features,
    build_spxy_split_with_summary,
    resolve_spxy_x_profile,
    spxy_x_feature_names,
)


def _ab_arrays(n: int = 40, t: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "slow": rng.normal(size=(n, t, 7)),
        "ultrasonic_tof_corrected_ab_raw_dsp_s": rng.normal(size=(n, t)) * 1e-4 + 8e-4,
        "ultrasonic_peak_index_ab_raw_dsp": rng.normal(size=(n, t)) + 100.0,
        "ultrasonic_sound_speed_ab_raw_dsp_m_per_s": rng.normal(size=(n, t)) + 340.0,
        "ultrasonic_snr_db_ab": rng.normal(size=(n, t)) + 20.0,
        "ultrasonic_psr_ab": rng.normal(size=(n, t)) + 5.0,
        "ultrasonic_quality_ab_raw_dsp": rng.uniform(0.5, 1.0, size=(n, t)),
        "ultrasonic_accepted_ab_raw_dsp": (rng.random(size=(n, t)) > 0.1).astype(np.float64),
    }


def test_bidir_ab_profile_is_50_dim_and_named():
    meta = resolve_spxy_x_profile(SPXY_X_PROFILE_BIDIR_OBSERVED_AB_V1)
    assert meta["x_feature_profile"] == "bidir_spxy_observed_ab_stats_v1"
    assert meta["role"] == "f5s_bidir_secondary_selector"
    names = spxy_x_feature_names(SPXY_X_PROFILE_BIDIR_OBSERVED_AB_V1)
    assert len(names) == DIM == 50
    assert all("_ba_" not in n for n in names)
    assert any("tof_corrected_ab" in n for n in names)
    digest = expected_x_feature_digest()
    assert digest["x_feature_profile_cli"] == F5S_X_PROFILE
    assert len(digest["x_feature_names_digest"]) == 64


def test_bidir_ab_features_reject_nan_and_require_keys():
    arrays = _ab_arrays()
    conditions = [{"sequence_id": f"Q{i}", "mixture_id": f"M{i // 2}"} for i in range(40)]
    X, names = _build_spxy_features(conditions, arrays, x_profile=SPXY_X_PROFILE_BIDIR_OBSERVED_AB_V1)
    assert X.shape == (40, 50)
    assert len(names) == 50

    bad = dict(arrays)
    del bad["ultrasonic_psr_ab"]
    with pytest.raises(KeyError, match="ultrasonic_psr_ab"):
        _build_spxy_features(conditions, bad, x_profile=SPXY_X_PROFILE_BIDIR_OBSERVED_AB_V1)

    nan_arr = dict(arrays)
    nan_arr["ultrasonic_snr_db_ab"] = arrays["ultrasonic_snr_db_ab"].copy()
    nan_arr["ultrasonic_snr_db_ab"][0, 0] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        _build_spxy_features(conditions, nan_arr, x_profile=SPXY_X_PROFILE_BIDIR_OBSERVED_AB_V1)


def test_bidir_ab_split_summary_contract():
    n = 80
    arrays = _ab_arrays(n=n)
    conditions = [
        {"sequence_id": f"Q{i:04d}", "mixture_id": f"M{i // 4:04d}"} for i in range(n)
    ]
    labels = np.column_stack(
        [
            np.linspace(0.1, 4.0, n),
            np.linspace(18.0, 21.0, n),
            100.0 - np.linspace(0.1, 4.0, n) - np.linspace(18.0, 21.0, n),
        ]
    )
    rows, summary = build_spxy_split_with_summary(
        conditions,
        arrays,
        labels,
        seed=20260704,
        alpha=0.5,
        extrapolation_strategy="y_margin_ood",
        x_profile=SPXY_X_PROFILE_BIDIR_OBSERVED_AB_V1,
    )
    assert summary["x_feature_profile"] == "bidir_spxy_observed_ab_stats_v1"
    assert summary["x_feature_count"] == 50
    assert summary["spxy_x_profile_cli"] == SPXY_X_PROFILE_BIDIR_OBSERVED_AB_V1
    assert summary["ood_set_hash"]
    assert summary["x_feature_matrix_hash"]
    summary["raw_dsp_bootstrap"] = {"role": "split_selection_bootstrap_only"}
    assert audit_derived_split_summary(summary, require_split_hash=False) == []
    assert set(rows) >= {"train", "val", "test", "extrapolation"}


def test_criterion_d_twelve_cells_and_no_mean_masking():
    cells = {}
    for selector in ("s_y", "s_l"):
        for seed in (20260704, 20260712, 20260720):
            for arm, r2 in (("A1", 0.40), ("A3", 0.50)):
                cells[f"{selector}:{seed}:{arm}:b1_ridge"] = {
                    "evaluations": {
                        "test": {"component_metrics": {"x_O2": {"r2": r2}}},
                        "extrapolation": {"component_metrics": {"x_O2": {"r2": r2}}},
                    }
                }
    ok = evaluate_criterion_d(cells, threshold=-0.01)
    assert ok["expected_cells"] == 12
    assert ok["passed"] is True
    assert ok["status"] == "passed"

    cells["s_y:20260704:A3:b1_ridge"]["evaluations"]["extrapolation"]["component_metrics"]["x_O2"][
        "r2"
    ] = 0.30
    fail = evaluate_criterion_d(cells, threshold=-0.01)
    assert fail["passed"] is False
    assert fail["status"] == "failed"
    assert any(
        (not c["passed"]) and c["split"] == "extrapolation" and c["selector_id"] == "s_y"
        for c in fail["cells"]
    )


def test_criterion_d_incomplete_when_missing_cell():
    result = evaluate_criterion_d({}, threshold=-0.01)
    assert result["status"] == "incomplete"
    assert result["passed"] is False
    assert len(result["missing"]) == 12


def test_f5s_split_relpath_and_config_flags():
    assert f5s_split_relpath(selector_id="s_y", seed=20260704) == Path(
        "s_y/spxy_ab_a05_ymargin_s20260704"
    )
    assert f5s_split_relpath(selector_id="s_l", seed=20260712) == Path(
        "s_l/spxy_ab_a05_lhsboundary_s20260712"
    )
    root = Path(__file__).resolve().parents[1]
    for name in ("tv3_bidir_model_protocol.json", "tv3_bidir_model_protocol_wide.json"):
        cfg = json.loads((root / "configs" / name).read_text(encoding="utf-8"))
        assert cfg["derive_secondary_selectors"] is True
        assert cfg["f5s"]["x_profile"] == "bidir_spxy_observed_ab_v1"
        assert cfg["f5s"]["split_seeds"] == [20260704, 20260712, 20260720]
        assert cfg["f5_amplitude_gates"]["selector_r2_noninferior_delta"] == -0.01
