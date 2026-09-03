from __future__ import annotations

import math

import CoolProp.CoolProp as coolprop
import pytest

from gf.sim.a2_dynamic_physics import audit_coolprop_sound_speed_grid
from gf.sim.a2dyn_sound_speed import (
    DIRECT_HEOS_SOUND_SPEED_MODEL_ID,
    a2dyn_sound_speed_for_model,
    coolprop_runtime_identity,
    direct_multifluid_heos_sound_speed,
)


@pytest.mark.parametrize(
    "composition",
    [
        {"Ar": 1.0, "He": 0.0, "CO2": 0.0},
        {"Ar": 0.0, "He": 1.0, "CO2": 0.0},
        {"Ar": 0.0, "He": 0.0, "CO2": 1.0},
        {"Ar": 0.25, "He": 0.75, "CO2": 0.0},
        {"Ar": 0.20, "He": 0.30, "CO2": 0.50},
    ],
)
def test_direct_heos_matches_the_pinned_coolprop_operator_exactly(
    composition: dict[str, float],
) -> None:
    names = ("Argon", "Helium", "CarbonDioxide")
    fractions = tuple(composition[name] for name in ("Ar", "He", "CO2"))
    nonzero = tuple(index for index, value in enumerate(fractions) if value > 0.0)
    if len(nonzero) == 1:
        state = coolprop.AbstractState("HEOS", names[nonzero[0]])
    else:
        state = coolprop.AbstractState("HEOS", "&".join(names))
        state.set_mole_fractions(fractions)
    state.specify_phase(coolprop.iphase_gas)
    state.update(coolprop.PT_INPUTS, 101325.0, 298.15)

    actual = direct_multifluid_heos_sound_speed(composition, 298.15, 101325.0)

    assert actual == float(state.speed_sound())


def test_direct_heos_rejects_inputs_outside_the_formal_domain() -> None:
    composition = {"Ar": 0.2, "He": 0.3, "CO2": 0.5}

    with pytest.raises(ValueError, match="pressure_pa is outside"):
        direct_multifluid_heos_sound_speed(composition, 298.15, 89999.0)
    with pytest.raises(ValueError, match="temperature_k is outside"):
        direct_multifluid_heos_sound_speed(composition, 313.16, 101325.0)
    with pytest.raises(ValueError, match="exactly the components"):
        direct_multifluid_heos_sound_speed({"Ar": 0.5, "He": 0.5}, 298.15, 101325.0)


def test_direct_heos_router_has_no_unknown_model_fallback() -> None:
    with pytest.raises(ValueError, match="unsupported sound speed model"):
        a2dyn_sound_speed_for_model(
            {"Ar": 0.2, "He": 0.3, "CO2": 0.5},
            298.15,
            101325.0,
            model_id="unregistered",
        )


def test_direct_heos_runtime_identity_is_frozen() -> None:
    identity = coolprop_runtime_identity()

    assert identity["version"] == "8.0.0"
    assert identity["source_revision"] == "61b616edfbb49f32633b21d1f901bdba1002340a"
    assert len(identity["binary_sha256"]) == 64


def test_direct_heos_grid_audit_is_generator_consistency_not_a_fit() -> None:
    report = audit_coolprop_sound_speed_grid(
        temperature_values_k=[278.15, 313.15],
        pressure_values_pa=[90000.0, 112000.0],
        simplex_step_pct=50.0,
        max_relative_error=0.0,
        sound_speed_model_id=DIRECT_HEOS_SOUND_SPEED_MODEL_ID,
        off_grid_count=16,
        off_grid_seed=20260831,
        check_pressure_direction=True,
    )

    assert report["grid_status"] == "PASS"
    assert report["off_grid"]["status"] == "PASS"
    assert report["pressure_direction"]["status"] == "PASS"
    assert math.isclose(report["max_relative_error"], 0.0, abs_tol=0.0)
