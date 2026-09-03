"""A2-DYN pair-specific second-virial source-parity evaluator.

The evaluator is deliberately narrow: it reads the registered draft asset,
rejects temperatures outside the asset domain, and obtains ``B``, ``dB/dT``
and ``d2B/dT2`` from one analytic Chebyshev representation.  It has no
CoolProp, TAPPS or implicit model fallback at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np


PAIR_COMPONENTS = ("Ar", "He", "CO2")
PAIR_IDS = ("Ar-Ar", "Ar-He", "Ar-CO2", "He-He", "He-CO2", "CO2-CO2")
PAIR_SOUND_SPEED_MODEL_ID = "a2dyn_cp_t_pair_virial_v2"
ASSET_PATH = Path(__file__).resolve().parents[3] / "configs" / "data" / "a2dyn_eos_coefficients_v2.json"


def _load_asset() -> Mapping[str, object]:
    asset = json.loads(ASSET_PATH.read_text(encoding="utf-8"))
    if not isinstance(asset, dict):
        raise ValueError(f"A2-DYN pair asset must be an object: {ASSET_PATH}")
    if asset.get("model_id") != "a2dyn_cp_t_pair_virial_v2":
        raise ValueError("A2-DYN pair asset model_id is not registered")
    if asset.get("schema_version") != "gf-a2dyn-eos-coefficients-2":
        raise ValueError("A2-DYN pair asset schema is unsupported")
    return MappingProxyType(asset)


PAIR_ASSET = _load_asset()
PAIR_TEMPERATURE_RANGE_K = tuple(float(value) for value in PAIR_ASSET["temperature_range_k"])
PAIR_PRESSURE_RANGE_PA = tuple(float(value) for value in PAIR_ASSET["pressure_range_pa"])
PAIR_COEFFICIENT_VERSION = str(PAIR_ASSET["coefficient_version"])


def canonical_pair_id(gas_i: str, gas_j: str) -> str:
    if gas_i not in PAIR_COMPONENTS or gas_j not in PAIR_COMPONENTS:
        raise ValueError(f"unsupported A2-DYN pair: {gas_i}-{gas_j}")
    if gas_i == gas_j:
        return f"{gas_i}-{gas_j}"
    first, second = sorted((gas_i, gas_j), key=PAIR_COMPONENTS.index)
    return f"{first}-{second}"


def pair_virial_terms(
    gas_i: str,
    gas_j: str,
    temperature_k: float,
) -> tuple[float, float, float]:
    """Return ``B``, ``dB/dT`` and ``d2B/dT2`` in SI units."""

    temperature = float(temperature_k)
    lower, upper = PAIR_TEMPERATURE_RANGE_K
    if not math.isfinite(temperature) or not lower <= temperature <= upper:
        raise ValueError(
            f"A2-DYN pair temperature must be within [{lower}, {upper}] K, got {temperature}"
        )
    pair_id = canonical_pair_id(gas_i, gas_j)
    virial = PAIR_ASSET["virial"]
    if not isinstance(virial, dict):
        raise ValueError("A2-DYN pair asset virial section is invalid")
    pairs = virial.get("pairs")
    if not isinstance(pairs, dict) or pair_id not in pairs:
        raise ValueError(f"A2-DYN pair asset is missing {pair_id}")
    pair = pairs[pair_id]
    if not isinstance(pair, dict):
        raise ValueError(f"A2-DYN pair asset entry is invalid: {pair_id}")
    domain = tuple(float(value) for value in pair["temperature_domain_k"])
    if domain != PAIR_TEMPERATURE_RANGE_K:
        raise ValueError(f"A2-DYN pair domain mismatch for {pair_id}")
    coefficients = np.asarray(pair["B_m3_mol_coefficients"], dtype=np.float64)
    if coefficients.ndim != 1 or coefficients.size < 2 or not np.isfinite(coefficients).all():
        raise ValueError(f"A2-DYN pair coefficients are invalid for {pair_id}")
    z = (2.0 * temperature - lower - upper) / (upper - lower)
    dz_dtemperature = 2.0 / (upper - lower)
    b = np.polynomial.chebyshev.chebval(z, coefficients)
    db_dz = np.polynomial.chebyshev.chebval(
        z, np.polynomial.chebyshev.chebder(coefficients)
    )
    d2b_dz2 = np.polynomial.chebyshev.chebval(
        z, np.polynomial.chebyshev.chebder(coefficients, 2)
    )
    terms = (float(b), float(db_dz * dz_dtemperature), float(d2b_dz2 * dz_dtemperature**2))
    if not all(math.isfinite(value) for value in terms):
        raise ValueError(f"A2-DYN pair evaluator produced invalid terms for {pair_id}")
    return terms


def mixture_pair_virial(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
) -> dict[str, float]:
    """Assemble the six canonical pair values with symmetric weights."""

    if set(mole_fractions) != set(PAIR_COMPONENTS):
        raise ValueError("A2-DYN pair mixture must contain exactly Ar, He and CO2")
    fractions = {gas: float(mole_fractions[gas]) for gas in PAIR_COMPONENTS}
    if any(not math.isfinite(value) or value < 0.0 for value in fractions.values()):
        raise ValueError("A2-DYN pair mixture fractions must be finite and non-negative")
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("A2-DYN pair mixture fractions must sum to one")
    total = np.zeros(3, dtype=np.float64)
    for index, gas_i in enumerate(PAIR_COMPONENTS):
        for gas_j in PAIR_COMPONENTS[index:]:
            weight = fractions[gas_i] * fractions[gas_j]
            if gas_i != gas_j:
                weight *= 2.0
            total += weight * np.asarray(pair_virial_terms(gas_i, gas_j, temperature_k))
    return {
        "B_m3_mol": float(total[0]),
        "dB_dT_m3_mol_k": float(total[1]),
        "d2B_dT2_m3_mol_k2": float(total[2]),
    }


def pair_virial_thermodynamic_state(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    pressure_pa: float,
) -> dict[str, float]:
    """Evaluate the draft v2 EOS without changing the frozen v1 module."""

    from gf.sim.ar_he_co2 import (
        GAS_MOLAR_MASS_KG_MOL,
        R_GAS_J_MOL_K,
        a2dyn_ideal_heat_capacity,
    )

    temperature = float(temperature_k)
    lower, upper = PAIR_TEMPERATURE_RANGE_K
    if not math.isfinite(temperature) or not lower <= temperature <= upper:
        raise ValueError(
            f"A2-DYN pair temperature must be within [{lower}, {upper}] K, got {temperature}"
        )
    pressure = float(pressure_pa)
    if not math.isfinite(pressure) or pressure < 0.0 or pressure > PAIR_PRESSURE_RANGE_PA[1]:
        raise ValueError(
            f"A2-DYN pair pressure must be within [0, {PAIR_PRESSURE_RANGE_PA[1]}] Pa, got {pressure}"
        )
    heat_capacity = a2dyn_ideal_heat_capacity(mole_fractions, temperature)
    virial = mixture_pair_virial(mole_fractions, temperature)
    molar_mass = sum(
        mole_fractions[gas] * GAS_MOLAR_MASS_KG_MOL[gas] for gas in PAIR_COMPONENTS
    )
    if pressure == 0.0:
        return {
            **heat_capacity,
            **virial,
            "molar_mass_kg_mol": float(molar_mass),
            "molar_density_mol_m3": 0.0,
            "pressure_derivative_density_pa_m3_mol": R_GAS_J_MOL_K * temperature,
            "pressure_derivative_temperature_pa_k_mol_m3": 0.0,
            "sound_speed_m_s": _ideal_sound_speed_from_heat_capacity(
                mole_fractions, temperature, heat_capacity, GAS_MOLAR_MASS_KG_MOL, R_GAS_J_MOL_K
            ),
        }

    b_value = virial["B_m3_mol"]
    reduced_pressure = pressure / (R_GAS_J_MOL_K * temperature)
    discriminant = 1.0 + 4.0 * b_value * reduced_pressure
    if not math.isfinite(discriminant) or discriminant <= 0.0:
        raise ValueError(f"pair virial EOS has no positive-density root: discriminant={discriminant}")
    molar_density = 2.0 * reduced_pressure / (1.0 + math.sqrt(discriminant))
    if not math.isfinite(molar_density) or molar_density <= 0.0:
        raise ValueError(f"pair virial EOS produced invalid molar density: {molar_density}")
    db_dtemperature = virial["dB_dT_m3_mol_k"]
    d2b_dtemperature2 = virial["d2B_dT2_m3_mol_k2"]
    pressure_density = R_GAS_J_MOL_K * temperature * (1.0 + 2.0 * b_value * molar_density)
    pressure_temperature = R_GAS_J_MOL_K * molar_density * (
        1.0 + b_value * molar_density + temperature * db_dtemperature * molar_density
    )
    cv = heat_capacity["cv_molar_j_mol_k"] - R_GAS_J_MOL_K * molar_density * (
        2.0 * temperature * db_dtemperature + temperature**2 * d2b_dtemperature2
    )
    if not math.isfinite(cv) or cv <= 0.0:
        raise ValueError(f"pair virial EOS produced invalid molar Cv: {cv}")
    cp_minus_cv = temperature * pressure_temperature**2 / (
        molar_density**2 * pressure_density
    )
    cp = cv + cp_minus_cv
    sound_speed_squared = (
        pressure_density
        + temperature * pressure_temperature**2 / (molar_density**2 * cv)
    ) / molar_mass
    if not math.isfinite(cp) or cp <= 0.0 or not math.isfinite(sound_speed_squared) or sound_speed_squared <= 0.0:
        raise ValueError("pair virial EOS produced invalid heat capacity or sound speed")
    return {
        **heat_capacity,
        **virial,
        "molar_mass_kg_mol": float(molar_mass),
        "molar_density_mol_m3": float(molar_density),
        "cv_molar_j_mol_k": float(cv),
        "cp_molar_j_mol_k": float(cp),
        "pressure_derivative_density_pa_m3_mol": float(pressure_density),
        "pressure_derivative_temperature_pa_k_mol_m3": float(pressure_temperature),
        "sound_speed_m_s": float(math.sqrt(sound_speed_squared)),
    }


def _ideal_sound_speed_from_heat_capacity(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    heat_capacity: Mapping[str, float],
    molar_masses: Mapping[str, float],
    gas_constant: float,
) -> float:
    molar_mass = sum(mole_fractions[gas] * molar_masses[gas] for gas in PAIR_COMPONENTS)
    speed_squared = (
        heat_capacity["cp_molar_j_mol_k"]
        / heat_capacity["cv_molar_j_mol_k"]
        * gas_constant
        * temperature_k
        / molar_mass
    )
    if not math.isfinite(speed_squared) or speed_squared <= 0.0:
        raise ValueError("pair virial EOS produced invalid ideal sound speed")
    return math.sqrt(speed_squared)


def pair_virial_sound_speed(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    pressure_pa: float,
) -> float:
    """Return sound speed from the explicit pair-virial v2 evaluator."""

    return pair_virial_thermodynamic_state(
        mole_fractions, temperature_k, pressure_pa
    )["sound_speed_m_s"]


def pair_virial_sound_speed_for_model(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    pressure_pa: float,
    *,
    model_id: str,
) -> float:
    """Resolve only the registered v2 model; unknown IDs fail explicitly."""

    if model_id != PAIR_SOUND_SPEED_MODEL_ID:
        raise ValueError(f"unsupported pair sound speed model: {model_id!r}")
    return pair_virial_sound_speed(mole_fractions, temperature_k, pressure_pa)


__all__ = [
    "ASSET_PATH",
    "PAIR_COEFFICIENT_VERSION",
    "PAIR_COMPONENTS",
    "PAIR_IDS",
    "PAIR_PRESSURE_RANGE_PA",
    "PAIR_SOUND_SPEED_MODEL_ID",
    "PAIR_TEMPERATURE_RANGE_K",
    "canonical_pair_id",
    "mixture_pair_virial",
    "pair_virial_sound_speed",
    "pair_virial_sound_speed_for_model",
    "pair_virial_thermodynamic_state",
    "pair_virial_terms",
]
