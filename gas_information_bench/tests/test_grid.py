import numpy as np

from gib.audit.grid import (
    ANGLE_TARGETS_DEG,
    ANGLE_TOLERANCE_DEG,
    INFORMATION_PROFILES,
    build_grid,
    g3_2_grid_audit,
    grid_summary,
)
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_grid_has_nine_recomputable_accessible_cells():
    cells = build_grid()
    assert len(cells) == 9
    assert all(cell["accessible"] for cell in cells)
    assert all(np.asarray(cell["effective_fisher"]).shape == (3, 3) for cell in cells)
    assert all(len(cell["crb_p90"]) == 4 for cell in cells)


def test_information_bands_are_separated_and_monotone():
    cells = build_grid()
    max_ratios = {
        band: max(cell["max_crb_p90_over_tau"] for cell in cells if cell["information_band"] == band)
        for band in INFORMATION_PROFILES
    }
    assert max_ratios["sufficient"] <= 0.5
    assert 0.8 <= max_ratios["critical"] <= 1.2
    assert max_ratios["insufficient"] >= 2.0


def test_angle_bands_hit_targets_without_using_noise_as_angle_axis():
    cells = build_grid()
    for angle_band, target in ANGLE_TARGETS_DEG.items():
        band_cells = [cell for cell in cells if cell["angle_band"] == angle_band]
        assert len(band_cells) == 3
        assert all(abs(cell["actual_angle_deg"] - target) <= ANGLE_TOLERANCE_DEG for cell in band_cells)
    for information_band in INFORMATION_PROFILES:
        band_cells = [cell for cell in cells if cell["information_band"] == information_band]
        assert len({cell["noise_scale"] for cell in band_cells}) == 1
        assert len({cell["coupling_strength"] for cell in band_cells}) == 3


def test_grid_summary_contains_source_and_configuration_contract():
    summary = grid_summary()
    assert summary["grid_id"] == "GIB-S1-3x3-v1"
    assert set(summary["target_error_tau"]) == {"N2", "CO2", "O2", "Ar"}
    assert len(summary["cells"]) == 9


def test_g3_2_grid_audit_recomputes_all_frozen_cells_and_passes_every_gate():
    frozen = json.loads((ROOT / "configs" / "p2_s1_grid.json").read_text(encoding="utf-8"))
    report = g3_2_grid_audit(frozen)
    assert report["gate_verdict"] == "pass", report["checks"]
    assert report["next_allowed_task"] == "P3-03"
    assert all(check["passed"] for check in report["checks"].values())
    assert report["checks"]["unique_reachable_cells"]["accessible_count"] == 9


def test_g3_2_grid_audit_rejects_frozen_value_drift():
    frozen = json.loads((ROOT / "configs" / "p2_s1_grid.json").read_text(encoding="utf-8"))
    frozen["cells"][0]["actual_angle_deg"] += 0.01
    report = g3_2_grid_audit(frozen)
    assert report["gate_verdict"] == "fail"
    assert not report["checks"]["frozen_json_value_match"]["passed"]
