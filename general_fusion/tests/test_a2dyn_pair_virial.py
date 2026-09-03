from __future__ import annotations

import math

import numpy as np
import pytest

from gf.sim.a2dyn_pair_virial import (
    PAIR_IDS,
    PAIR_SOUND_SPEED_MODEL_ID,
    PAIR_TEMPERATURE_RANGE_K,
    canonical_pair_id,
    mixture_pair_virial,
    pair_virial_sound_speed,
    pair_virial_sound_speed_for_model,
    pair_virial_terms,
)
from gf.sim.ar_he_co2 import (
    GAS_MOLAR_MASS_KG_MOL,
    R_GAS_J_MOL_K,
    a2dyn_ideal_heat_capacity,
)


def test_pair_ids_are_canonical_and_symmetric() -> None:
    assert canonical_pair_id("He", "Ar") == "Ar-He"
    assert canonical_pair_id("CO2", "Ar") == "Ar-CO2"
    assert canonical_pair_id("CO2", "He") == "He-CO2"
    assert canonical_pair_id("CO2", "CO2") == "CO2-CO2"
    for pair_id in PAIR_IDS:
        gas_i, gas_j = pair_id.split("-")
        assert pair_virial_terms(gas_i, gas_j, 298.15) == pair_virial_terms(gas_j, gas_i, 298.15)


def test_pair_derivatives_are_from_the_same_analytic_representation() -> None:
    lower, upper = PAIR_TEMPERATURE_RANGE_K
    temperatures = (lower + 0.1, 293.15, 298.15, upper - 0.1)
    step = 1.0e-2
    for pair_id in PAIR_IDS:
        gas_i, gas_j = pair_id.split("-")
        for temperature in temperatures:
            value, first, second = pair_virial_terms(gas_i, gas_j, temperature)
            left = pair_virial_terms(gas_i, gas_j, temperature - step)[0]
            right = pair_virial_terms(gas_i, gas_j, temperature + step)[0]
            finite_first = (right - left) / (2.0 * step)
            finite_second = (right - 2.0 * value + left) / step**2
            assert math.isclose(first, finite_first, rel_tol=2.0e-6, abs_tol=2.0e-12)
            assert math.isclose(second, finite_second, rel_tol=2.0e-5, abs_tol=2.0e-11)


def test_pair_mixture_uses_each_cross_pair_once_with_symmetric_weight() -> None:
    fractions = {"Ar": 0.2, "He": 0.3, "CO2": 0.5}
    actual = mixture_pair_virial(fractions, 298.15)
    expected = np.zeros(3, dtype=np.float64)
    components = ("Ar", "He", "CO2")
    for index, gas_i in enumerate(components):
        for gas_j in components[index:]:
            weight = fractions[gas_i] * fractions[gas_j]
            if gas_i != gas_j:
                weight *= 2.0
            expected += weight * np.asarray(pair_virial_terms(gas_i, gas_j, 298.15))
    assert np.allclose(list(actual.values()), expected, rtol=0.0, atol=1.0e-18)


def test_pair_sound_speed_selection_is_explicit_and_has_no_fallback() -> None:
    fractions = {"Ar": 0.2, "He": 0.3, "CO2": 0.5}
    speed = pair_virial_sound_speed(fractions, 298.15, 101325.0)
    assert math.isfinite(speed) and speed > 0.0
    assert pair_virial_sound_speed_for_model(
        fractions,
        298.15,
        101325.0,
        model_id=PAIR_SOUND_SPEED_MODEL_ID,
    ) == speed
    heat_capacity = a2dyn_ideal_heat_capacity(fractions, 298.15)
    molar_mass = sum(fractions[gas] * GAS_MOLAR_MASS_KG_MOL[gas] for gas in fractions)
    ideal_limit = math.sqrt(
        heat_capacity["cp_molar_j_mol_k"]
        / heat_capacity["cv_molar_j_mol_k"]
        * R_GAS_J_MOL_K
        * 298.15
        / molar_mass
    )
    assert pair_virial_sound_speed(fractions, 298.15, 0.0) == ideal_limit
    with pytest.raises(ValueError, match="unsupported pair sound speed model"):
        pair_virial_sound_speed_for_model(fractions, 298.15, 101325.0, model_id="unknown")
