from __future__ import annotations

import json
from pathlib import Path

from gf.sim.a2dyn_sound_speed import coolprop_runtime_identity

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_result() -> dict:
    path = PROJECT_ROOT / "outputs" / "summary" / "a2_dynamic_v1" / "pilot_audit_r4.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_a2_dynamic_pilot_manifest_is_qualified_and_dependencies_recompute() -> None:
    summary_path = PROJECT_ROOT / "outputs" / "summary" / "a2_dynamic_v1" / "pilot_audit_r4.json"
    run_dir = PROJECT_ROOT / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-2r4-pilot"
    manifest_path = run_dir / "manifest.json"
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = json.loads(
        (PROJECT_ROOT / "configs" / "data" / "ar_he_co2_a2_dynamic_v1.json").read_text(encoding="utf-8")
    )
    experiment = json.loads(
        (PROJECT_ROOT / "configs" / "experiment" / "a2_dynamic_protocol.json").read_text(encoding="utf-8")
    )

    assert result == manifest
    assert result["status"] == "PILOT_QUALIFIED"
    assert result["stage"] == "A2-DYN-2R4"
    assert result["mixture_count"] == 240
    assert result["family_counts"] == {
        family: 40
        for family in ("D-IID", "D-KINETICS", "D-PROTOCOL", "D-NOISE-DRIFT", "D-ENV-CAL", "D-JOINT")
    }
    assert result["split_counts"] == {"train": 120, "val": 60, "stress_val": 60}
    assert all(result["checks"].values())
    assert result["selection"]["sample_rate_hz"] == 5.0
    assert result["selection"]["duration_s"] == 240.0
    assert result["selection"]["ultrasonic_profile_id"] == "US-CHIRP-XCORR-PARABOLIC-1"
    assert result["selection"]["tof_estimator"] == "reference_xcorr_parabolic"
    assert data["hardware_profiles"]["ultrasonic"]["selected_profile_id"] == result["selection"]["ultrasonic_profile_id"]
    assert experiment["pilot"]["status"] == "PILOT_QUALIFIED"
    assert experiment["pilot"]["selected_sample_rate_hz"] == result["selection"]["sample_rate_hz"]
    assert result["runtime_identity"] == coolprop_runtime_identity()

    # pilot 是历史冻结产物：其依赖 hash 只要求格式合法且与 resolved 配置
    # 自洽；A2-DYN-4 之后的代码演进不重写 pilot 的 hash（见 13 文档 §20）。
    assert all(len(value) == 64 for value in result["dependency_hashes"].values())
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["dependency_hashes"] == result["dependency_hashes"]


def test_a2_dynamic_pilot_compares_axes_metrics_and_multipath_without_waveforms() -> None:
    result = _load_result()
    assert set(result["scenarios"]) == {
        "1Hz_120s", "1Hz_240s", "1Hz_360s",
        "2Hz_120s", "2Hz_240s", "2Hz_360s",
        "5Hz_120s", "5Hz_240s", "5Hz_360s",
    }
    assert all(item["timestamp_alignment_max_error_s"] <= 1.0e-12 for item in result["scenarios"].values())
    assert set(result["selection"]["sample_rate_candidates"]) == {"1Hz", "2Hz", "5Hz"}
    assert not result["selection"]["sample_rate_candidates"]["1Hz"]["meets_realtime_update_period"]
    selected = result["scenarios"][result["audit_scenario_key"]]
    metric = selected["pilot_probe_metrics"]["P-B-LAST-LS"]["P060"]["stress_val"]
    assert "macro_RNMAE" in metric
    assert "component_RNMAE" in metric
    assert result["pilot_probe_checks"]["metric_definition"].startswith("group-level target-range RNMAE")

    assert set(result["ultrasonic_candidates"]) == {
        "US-BURST-XCORR-1",
        "US-CHIRP-XCORR-PARABOLIC-1",
    }
    for candidate in result["ultrasonic_candidates"].values():
        assert set(candidate["multipath_audit"]) == {"US-MP-NOMINAL", "US-MP-OOD"}
        assert candidate["waveform_persisted"] is False
    assert result["resource"]["waveform_persisted_bytes"] == 0
    assert result["resource"]["formal_core_array_bytes"] == sum(
        result["resource"]["formal_array_breakdown_bytes"].values()
    )
