"""G0 schema / ID / registry tests for tunnel-ventilation-comsol-1."""
from __future__ import annotations

import pytest

from tv3.sim.comsol.ids import (
    AcousticCaseId,
    FieldCaseId,
    SensorLayoutId,
    assert_ids_not_aliased,
    make_acoustic_case_id,
    make_field_case_id,
    make_sensor_layout_id,
)
from tv3.sim.comsol.registry import audit_g0_gate, load_g0_registries
from tv3.sim.comsol.schema import (
    CASE_REGISTRY_FIELDS,
    COMPONENT_FIELDS,
    COMPOSITION_SCHEME,
    LOCAL_STATE_FEATURES,
    PATH_STATE_FEATURES,
    SCHEMA_VERSION,
    SEQUENCE_INDEX_FIELDS,
    SLOW_CHANNELS,
)
from tv3.sim.core.ids import make_mixture_id, make_sequence_id
from tv3.sim.core.tunnel_ventilation_schema import (
    SCHEMA_VERSION as STATIC_SCHEMA_VERSION,
)


def test_comsol_schema_isolated_from_static_air():
    assert SCHEMA_VERSION == "tunnel-ventilation-comsol-1"
    assert SCHEMA_VERSION != STATIC_SCHEMA_VERSION
    assert COMPOSITION_SCHEME == "tunnel_ventilation_comsol"
    assert COMPONENT_FIELDS == ("x_CO2", "x_O2", "x_N2")
    assert len(SLOW_CHANNELS) == 7
    assert "V_NDIR_CH4" not in SLOW_CHANNELS


def test_core_table_fields_match_plan():
    assert "field_case_id" in CASE_REGISTRY_FIELDS
    assert "mixture_id" in CASE_REGISTRY_FIELDS
    assert CASE_REGISTRY_FIELDS[0] == "field_case_id"
    assert SEQUENCE_INDEX_FIELDS[:4] == (
        "sequence_id",
        "mixture_id",
        "field_case_id",
        "sensor_layout_id",
    )
    assert len(LOCAL_STATE_FEATURES) == 12
    assert len(PATH_STATE_FEATURES) == 8


def test_id_formats():
    assert str(make_field_case_id(1)) == "F000001"
    assert str(make_sensor_layout_id(1)) == "SL0001"
    assert str(make_acoustic_case_id(1)) == "A000001"
    assert str(make_mixture_id(1)) == "M000001"
    assert str(make_sequence_id(1)) == "Q000001"
    with pytest.raises(ValueError):
        make_field_case_id(0)
    with pytest.raises(ValueError):
        FieldCaseId("Q000001")
    with pytest.raises(ValueError):
        SensorLayoutId("SL1")
    with pytest.raises(ValueError):
        AcousticCaseId("A1")


def test_ids_not_aliased():
    assert_ids_not_aliased(
        mixture_id="M000001",
        sequence_id="Q000001",
        field_case_id="F000001",
    )
    with pytest.raises(ValueError):
        assert_ids_not_aliased(
            mixture_id="Q000001",
            sequence_id="Q000001",
            field_case_id="F000001",
        )
    with pytest.raises(ValueError):
        assert_ids_not_aliased(
            mixture_id="M000001",
            sequence_id="M000001",
            field_case_id="F000001",
        )


def test_g0_registries_load_and_block_formal():
    bundle = load_g0_registries()
    assert set(bundle["sha256"]) == {
        "parameter",
        "geometry",
        "sensor_layout",
        "validation",
    }
    for digest in bundle["sha256"].values():
        assert len(digest) == 64
    audit = audit_g0_gate()
    assert audit["schema_version"] == "tunnel-ventilation-comsol-1"
    assert audit["formal_ready"] is False
    assert audit["smoke_allowed"] is True
    assert audit["verdict"] == "g0_input_blocked"
    assert len(audit["blocking_items"]) >= 1
    assert "gas_chamber_simplified.step" in " ".join(audit["forbidden_tunnel_geometry"])
    assert bundle["registries"]["parameter"]["parameters"]["duct_volume_flow_m3_s"]["value"] is not None
    assert bundle["registries"]["validation"]["claim_without_validation"] == "numerical-only"


def test_geometry_forbids_chamber_as_tunnel():
    geom = load_g0_registries()["registries"]["geometry"]
    forbidden = " ".join(geom["forbidden_as_tunnel_geometry"])
    assert "gas_chamber_simplified" in forbidden
    assert geom["legacy_acoustic_p0"]["must_not_modify_for_flow"] is True
