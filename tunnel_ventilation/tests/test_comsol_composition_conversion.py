"""Mole percent ↔ mass fraction conversion tests."""
from __future__ import annotations

import numpy as np
import pytest

from tv3.sim.comsol.composition import (
    MOLAR_MASS_KG_PER_MOL,
    dry_mole_percent_to_mass_fraction,
    mass_fraction_to_dry_mole_percent,
    mixture_molar_mass_kg_per_mol,
    validate_dry_mole_percent_closure,
    wet_to_dry_mole_percent,
)


def test_molar_mass_aligned_with_tv3_physics():
    assert MOLAR_MASS_KG_PER_MOL["CO2"] == 0.04401
    assert MOLAR_MASS_KG_PER_MOL["O2"] == 0.031998
    assert MOLAR_MASS_KG_PER_MOL["N2"] == 0.02801


def test_roundtrip_mole_mass_mole():
    x = {"x_CO2": 1.0, "x_O2": 20.5, "x_N2": 78.5}
    w = dry_mole_percent_to_mass_fraction(x)
    assert isinstance(w, dict)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    x2 = mass_fraction_to_dry_mole_percent(w)
    for k in x:
        assert abs(x2[k] - x[k]) < 1e-8


def test_roundtrip_array_batch():
    x = np.array(
        [
            [0.04, 20.9, 79.06],
            [5.0, 18.0, 77.0],
        ],
        dtype=np.float64,
    )
    w = dry_mole_percent_to_mass_fraction(x)
    assert w.shape == (2, 3)
    x2 = mass_fraction_to_dry_mole_percent(w)
    np.testing.assert_allclose(x2, x, atol=1e-8)


def test_pure_n2_mass_fraction():
    w = dry_mole_percent_to_mass_fraction({"x_CO2": 0.0, "x_O2": 0.0, "x_N2": 100.0})
    assert abs(w["w_N2"] - 1.0) < 1e-12
    assert abs(w["w_CO2"]) < 1e-12


def test_closure_failures():
    with pytest.raises(ValueError):
        validate_dry_mole_percent_closure({"x_CO2": 1.0, "x_O2": 20.0, "x_N2": 70.0})
    with pytest.raises(ValueError):
        dry_mole_percent_to_mass_fraction({"x_CO2": -1.0, "x_O2": 21.0, "x_N2": 80.0})


def test_wet_to_dry():
    dry = wet_to_dry_mole_percent(
        {"x_CO2": 0.99, "x_O2": 20.295, "x_N2": 76.715},
        x_h2o_vol_pct=2.0,
    )
    assert abs(sum(dry.values()) - 100.0) < 1e-8
    assert dry["x_CO2"] > 0.99  # renormalized upward after removing water


def test_mixture_molar_mass_between_pure_bounds():
    m = mixture_molar_mass_kg_per_mol({"x_CO2": 1.0, "x_O2": 20.5, "x_N2": 78.5})
    assert MOLAR_MASS_KG_PER_MOL["N2"] < m < MOLAR_MASS_KG_PER_MOL["CO2"]
