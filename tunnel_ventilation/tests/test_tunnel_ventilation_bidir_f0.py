"""F0 tests: bidir schema draft + parameter registry gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tv3.sim.core.tunnel_ventilation_bidir_schema import (
    BIDIR_ORACLE_ARRAYS,
    BIDIR_REQUIRED_ARRAYS,
    COMPONENT_FIELDS,
    COMPOSITION_SCHEME,
    CONDITION_GRID_FLOW_FIELDS,
    FORMAL_FEATURE_BUILDER,
    SCHEMA_VERSION,
    SIM_REVISION_TAG,
    SLOW_CHANNELS,
)
from tv3.sim.core.tunnel_ventilation_schema import (
    COMPONENT_FIELDS as BASE_COMPONENT_FIELDS,
    SCHEMA_VERSION as BASE_SCHEMA_VERSION,
    SLOW_CHANNELS as BASE_SLOW_CHANNELS,
)
from tv3.sim.generation.tunnel_ventilation.bidir_registry import (
    audit_f0_gate,
    default_config_dir,
    load_f0_registry,
)


def test_bidir_schema_isolated_and_reuses_base_constants():
    assert SCHEMA_VERSION == "tunnel-ventilation-bidir-1"
    assert SCHEMA_VERSION != BASE_SCHEMA_VERSION
    assert COMPOSITION_SCHEME == "tunnel_ventilation_bidir"
    assert COMPONENT_FIELDS == BASE_COMPONENT_FIELDS == ("x_CO2", "x_O2", "x_N2")
    assert SLOW_CHANNELS == BASE_SLOW_CHANNELS
    assert len(SLOW_CHANNELS) == 7
    assert "V_NDIR_CH4" not in SLOW_CHANNELS
    assert SIM_REVISION_TAG == "v7-bidir-flow-v1"
    assert FORMAL_FEATURE_BUILDER == "raw_dsp_bidirectional_v1"


def test_condition_and_array_contract():
    assert CONDITION_GRID_FLOW_FIELDS == (
        "v_path_m_per_s",
        "flow_scenario",
        "pair_interval_s",
        "delay_asymmetry_s",
        "jitter_correlation",
    )
    assert "ultrasonic_ab" in BIDIR_REQUIRED_ARRAYS
    assert "ultrasonic_ba" in BIDIR_REQUIRED_ARRAYS
    assert "ultrasonic_v_path_true_m_per_s" in BIDIR_ORACLE_ARRAYS
    assert "ultrasonic_alpha_true_npm" in BIDIR_ORACLE_ARRAYS
    # flow must not be a slow channel
    assert "v_path_m_per_s" not in SLOW_CHANNELS


def test_f0_registry_loads_and_passes_gate():
    bundle = load_f0_registry()
    assert Path(bundle["path"]).is_file()
    assert len(bundle["sha256"]) == 64
    registry = bundle["registry"]
    for forbidden in ("allowed_next_stage", "f1_status", "f2_status"):
        assert forbidden not in registry
    audit = audit_f0_gate()
    assert audit["passed"] is True
    assert audit["verdict"] == "f0_registry_frozen"
    assert audit["allowed_next_stage"] == "F1_physics_unit_tests"
    assert audit["issues"] == []
    assert audit["jitter_scenarios"]["has_nominal_literature_bound"] is True
    assert "conservative_v1" in audit["jitter_scenarios"]["scenario_ids"]
    assert "nominal_daq_half_sample" in audit["jitter_scenarios"]["scenario_ids"]


def test_f0_gate_rejects_mutable_stage_keys(tmp_path: Path):
    src = default_config_dir() / "parameter_registry.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    data["allowed_next_stage"] = "F3_dsp_estimator"
    data["f1_status"] = {"verdict": "pollution"}
    (tmp_path / "parameter_registry.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    audit = audit_f0_gate(tmp_path)
    assert audit["passed"] is False
    assert any("mutable stage key" in item for item in audit["issues"])


def test_f0_gate_fails_without_nominal_literature_bound(tmp_path: Path):
    src = default_config_dir() / "parameter_registry.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    for sc in data["trigger_jitter_scenarios"]["scenarios"]:
        if sc["id"] == "nominal_daq_half_sample":
            sc["source"] = "engineering_scenario"
            sc.pop("derivation", None)
    (tmp_path / "parameter_registry.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    audit = audit_f0_gate(tmp_path)
    assert audit["passed"] is False
    assert audit["verdict"] == "inconclusive_parameter_bounds"
    assert any("literature_bound" in item for item in audit["issues"])


def test_f0_gate_fails_if_flow_enters_slow(tmp_path: Path):
    src = default_config_dir() / "parameter_registry.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    data["composition_anchor"]["flow_in_slow_channels"] = True
    (tmp_path / "parameter_registry.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    audit = audit_f0_gate(tmp_path)
    assert audit["passed"] is False
    assert any("flow_in_slow_channels" in item for item in audit["issues"])
