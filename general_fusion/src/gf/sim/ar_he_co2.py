from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np


R_GAS_J_MOL_K = 8.314462618
REFERENCE_TEMPERATURE_K = 298.15
REFERENCE_PRESSURE_PA = 101_325.0
SYSTEM_DELAY_S = 80e-6
NDIR_BASELINE_V = 2.5
NDIR_EFFECTIVE_ABSORBANCE_PER_CO2_PERCENT = 0.045
TCS_BASELINE_V = 1.1
TCS_RESPONSE_V_PER_W_M_K = 15.0
TCS_REFERENCE_CONDUCTIVITY_W_M_K = 0.026

GAS_MOLAR_MASS_KG_MOL: Mapping[str, float] = MappingProxyType(
    {"Ar": 0.039948, "He": 0.004002602, "CO2": 0.0440095}
)
GAS_CP_J_MOL_K: Mapping[str, float] = MappingProxyType(
    {"Ar": 20.786, "He": 20.786, "CO2": 37.135}
)
GAS_THERMAL_CONDUCTIVITY_W_M_K: Mapping[str, float] = MappingProxyType(
    {"Ar": 0.01772, "He": 0.1513, "CO2": 0.0166}
)
GAS_VISCOSITY_PA_S: Mapping[str, float] = MappingProxyType(
    {"Ar": 22.61e-6, "He": 19.60e-6, "CO2": 14.91e-6}
)

A1_SOUND_SPEED_MODEL_ID = "a1_constant_cp_ideal_v1"
A2DYN_SOUND_SPEED_MODEL_ID = "a2dyn_cp_t_virial_v1"
A2DYN_COEFFICIENT_VERSION = "a2dyn-eos-coefficients-20260901-r1"
A2DYN_TEMPERATURE_RANGE_K = (278.15, 313.15)
A2DYN_OPERATIONAL_PRESSURE_RANGE_PA = (90_000.0, 112_000.0)

# NASA7 coefficients from the NASA Glenn thermodynamic database.  The first
# five coefficients are the dimensionless Cp/R polynomial; the two integration
# constants are retained in the registry asset but are not needed for sound speed.
A2DYN_NASA7_COEFFICIENTS: Mapping[str, tuple[float, ...]] = MappingProxyType(
    {
        "Ar": (2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 4.37967491),
        "He": (2.5, 0.0, 0.0, 0.0, 0.0, -745.375, 0.928724724),
        "CO2": (
            2.35677352,
            8.98459677e-03,
            -7.12356269e-06,
            2.45919022e-09,
            -1.43699548e-13,
            -4.83719697e04,
            9.90105222,
        ),
    }
)

# Critical properties and acentric factors used by the published Tsonopoulos
# corresponding-states correlation.  Cross pairs use Tsonopoulos' registered
# critical-property rule in ``_virial_pair_parameters``; no pair is zeroed.
A2DYN_VIRIAL_CRITICAL_PROPERTIES: Mapping[str, tuple[float, float, float]] = MappingProxyType(
    {
        "Ar": (150.687, 4.898e6, -0.00219),
        "He": (5.1953, 2.2746e5, -0.385),
        "CO2": (304.1282, 7.3773e6, 0.22394),
    }
)
A2DYN_VIRIAL_PAIR_IDS = (
    "Ar-Ar",
    "Ar-He",
    "Ar-CO2",
    "He-He",
    "He-CO2",
    "CO2-CO2",
)

SENSOR_TYPES: Mapping[str, str] = MappingProxyType(
    {
        "ultrasonic_tof": "acoustic_tof",
        "thermal_conductivity_voltage": "thermal_conductivity",
        "ndir_co2_voltage": "ndir",
    }
)


@dataclass(frozen=True)
class PilotCondition:
    mixture_id: str
    x_ar_pct: float
    x_he_pct: float
    x_co2_pct: float
    split: str
    temperature_k: float = REFERENCE_TEMPERATURE_K
    pressure_pa: float = REFERENCE_PRESSURE_PA
    path_length_m: float = 0.2

    def __post_init__(self) -> None:
        if not self.mixture_id:
            raise ValueError("mixture_id must be non-empty")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train, val, or test, got {self.split!r}")
        fractions = (self.x_ar_pct, self.x_he_pct, self.x_co2_pct)
        if any(not math.isfinite(value) or value < 0.0 or value > 100.0 for value in fractions):
            raise ValueError(f"composition values must be finite and within [0,100], got {fractions}")
        if not math.isclose(sum(fractions), 100.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"composition must sum to 100 mol%, got {sum(fractions)}")
        if not math.isclose(self.temperature_k, REFERENCE_TEMPERATURE_K, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("A0 pilot is frozen at 298.15 K")
        if not math.isclose(self.pressure_pa, REFERENCE_PRESSURE_PA, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("A0 pilot is frozen at 101325 Pa")
        if not math.isfinite(self.path_length_m) or self.path_length_m <= 0.0:
            raise ValueError("path_length_m must be finite and positive")

    @property
    def mole_fractions(self) -> dict[str, float]:
        return {
            "Ar": self.x_ar_pct / 100.0,
            "He": self.x_he_pct / 100.0,
            "CO2": self.x_co2_pct / 100.0,
        }


@dataclass(frozen=True)
class PilotRecord:
    condition: PilotCondition
    time_s: np.ndarray
    signals: Mapping[str, np.ndarray]


def build_pilot_record(condition: PilotCondition, *, timesteps: int, dt_s: float) -> PilotRecord:
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    time_s = np.arange(timesteps, dtype=np.float64) * dt_s
    baseline_fractions = {"Ar": 1.0, "He": 0.0, "CO2": 0.0}
    target_fractions = condition.mole_fractions

    baseline_speed = ideal_gas_sound_speed(baseline_fractions, condition.temperature_k)
    target_speed = ideal_gas_sound_speed(target_fractions, condition.temperature_k)
    baseline_tof = condition.path_length_m / baseline_speed + SYSTEM_DELAY_S
    target_tof = condition.path_length_m / target_speed + SYSTEM_DELAY_S

    baseline_conductivity = wms_thermal_conductivity(baseline_fractions)
    target_conductivity = wms_thermal_conductivity(target_fractions)
    baseline_tcs = thermal_conductivity_voltage(baseline_conductivity)
    target_tcs = thermal_conductivity_voltage(target_conductivity)

    baseline_ndir = ndir_co2_voltage(0.0, condition.pressure_pa, condition.temperature_k)
    target_ndir = ndir_co2_voltage(condition.x_co2_pct, condition.pressure_pa, condition.temperature_k)

    signals = {
        "ultrasonic_tof": _first_order_response(time_s, baseline_tof, target_tof, tau_s=0.5),
        "thermal_conductivity_voltage": _first_order_response(time_s, baseline_tcs, target_tcs, tau_s=10.0),
        "ndir_co2_voltage": _first_order_response(time_s, baseline_ndir, target_ndir, tau_s=8.0),
    }
    return PilotRecord(condition=condition, time_s=time_s, signals=MappingProxyType(signals))


def ideal_gas_sound_speed(mole_fractions: Mapping[str, float], temperature_k: float) -> float:
    _validate_mole_fractions(mole_fractions)
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature_k must be finite and positive")
    molar_mass = sum(mole_fractions[gas] * GAS_MOLAR_MASS_KG_MOL[gas] for gas in GAS_MOLAR_MASS_KG_MOL)
    cp_mix = sum(mole_fractions[gas] * GAS_CP_J_MOL_K[gas] for gas in GAS_CP_J_MOL_K)
    cv_mix = cp_mix - R_GAS_J_MOL_K
    if molar_mass <= 0.0 or cv_mix <= 0.0:
        raise ValueError("mixture produces invalid molar mass or heat capacity")
    gamma = cp_mix / cv_mix
    return math.sqrt(gamma * R_GAS_J_MOL_K * temperature_k / molar_mass)


def sound_speed_for_model(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    pressure_pa: float,
    *,
    model_id: str = A1_SOUND_SPEED_MODEL_ID,
) -> float:
    """Resolve the registered sound-speed model without an implicit fallback."""

    if model_id == A1_SOUND_SPEED_MODEL_ID:
        return ideal_gas_sound_speed(mole_fractions, temperature_k)
    if model_id == A2DYN_SOUND_SPEED_MODEL_ID:
        return a2dyn_cp_t_virial_sound_speed(mole_fractions, temperature_k, pressure_pa)
    raise ValueError(f"unsupported sound speed model: {model_id!r}")


def a2dyn_species_heat_capacity(
    gas: str,
    temperature_k: float,
) -> tuple[float, float, float]:
    """Return NASA7 ``Cp``, first derivative and second derivative for one gas."""

    coefficients = A2DYN_NASA7_COEFFICIENTS.get(gas)
    if coefficients is None:
        raise ValueError(f"unsupported A2DYN species: {gas!r}")
    temperature = _validate_a2dyn_temperature(temperature_k)
    a1, a2, a3, a4, a5, _, _ = coefficients
    cp = R_GAS_J_MOL_K * (
        a1 + a2 * temperature + a3 * temperature**2 + a4 * temperature**3 + a5 * temperature**4
    )
    dcp = R_GAS_J_MOL_K * (
        a2 + 2.0 * a3 * temperature + 3.0 * a4 * temperature**2 + 4.0 * a5 * temperature**3
    )
    d2cp = R_GAS_J_MOL_K * (
        2.0 * a3 + 6.0 * a4 * temperature + 12.0 * a5 * temperature**2
    )
    if not math.isfinite(cp) or cp <= 0.0:
        raise ValueError(f"NASA7 produced invalid Cp for {gas}: {cp}")
    return cp, dcp, d2cp


def a2dyn_ideal_heat_capacity(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
) -> dict[str, float]:
    """Return mixture ideal ``Cp``/``Cv`` from the registered NASA7 set."""

    _validate_mole_fractions(mole_fractions)
    temperature = _validate_a2dyn_temperature(temperature_k)
    species_terms = [
        a2dyn_species_heat_capacity(gas, temperature) for gas in GAS_MOLAR_MASS_KG_MOL
    ]
    cp = sum(
        mole_fractions[gas] * species_terms[index][0]
        for index, gas in enumerate(GAS_MOLAR_MASS_KG_MOL)
    )
    dcp = sum(
        mole_fractions[gas] * species_terms[index][1]
        for index, gas in enumerate(GAS_MOLAR_MASS_KG_MOL)
    )
    d2cp = sum(
        mole_fractions[gas] * species_terms[index][2]
        for index, gas in enumerate(GAS_MOLAR_MASS_KG_MOL)
    )
    cv = cp - R_GAS_J_MOL_K
    if not math.isfinite(cv) or cv <= 0.0:
        raise ValueError(f"NASA7 produced invalid mixture Cv: {cv}")
    return {
        "cp_molar_j_mol_k": float(cp),
        "cv_molar_j_mol_k": float(cv),
        "dcp_dtemperature_j_mol_k2": float(dcp),
        "d2cp_dtemperature2_j_mol_k3": float(d2cp),
    }


def a2dyn_mixture_virial(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
) -> dict[str, float]:
    """Return ``B``, ``dB/dT`` and ``d2B/dT2`` for the complete binary mixture."""

    _validate_mole_fractions(mole_fractions)
    temperature = _validate_a2dyn_temperature(temperature_k)
    total_b = 0.0
    total_db = 0.0
    total_d2b = 0.0
    for gas_i in GAS_MOLAR_MASS_KG_MOL:
        for gas_j in GAS_MOLAR_MASS_KG_MOL:
            fraction = mole_fractions[gas_i] * mole_fractions[gas_j]
            pair_b, pair_db, pair_d2b = _tsonopoulos_pair_terms(gas_i, gas_j, temperature)
            total_b += fraction * pair_b
            total_db += fraction * pair_db
            total_d2b += fraction * pair_d2b
    return {
        "B_m3_mol": float(total_b),
        "dB_dT_m3_mol_k": float(total_db),
        "d2B_dT2_m3_mol_k2": float(total_d2b),
    }


def a2dyn_thermodynamic_state(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    pressure_pa: float,
) -> dict[str, float]:
    """Evaluate the truncated virial EOS and all derivatives used by sound speed."""

    _validate_mole_fractions(mole_fractions)
    temperature = _validate_a2dyn_temperature(temperature_k)
    pressure = _validate_a2dyn_pressure(pressure_pa)
    heat_capacity = a2dyn_ideal_heat_capacity(mole_fractions, temperature)
    virial = a2dyn_mixture_virial(mole_fractions, temperature)
    molar_mass = sum(
        mole_fractions[gas] * GAS_MOLAR_MASS_KG_MOL[gas] for gas in GAS_MOLAR_MASS_KG_MOL
    )
    if pressure == 0.0:
        return {
            **heat_capacity,
            **virial,
            "molar_mass_kg_mol": float(molar_mass),
            "molar_density_mol_m3": 0.0,
            "cv_molar_j_mol_k": heat_capacity["cv_molar_j_mol_k"],
            "cp_molar_j_mol_k": heat_capacity["cp_molar_j_mol_k"],
            "pressure_derivative_density_pa_m3_mol": R_GAS_J_MOL_K * temperature,
            "pressure_derivative_temperature_pa_k_mol_m3": 0.0,
            "sound_speed_m_s": _a2dyn_ideal_sound_speed_from_heat_capacity(
                mole_fractions, temperature, heat_capacity
            ),
        }

    b_value = virial["B_m3_mol"]
    reduced_pressure = pressure / (R_GAS_J_MOL_K * temperature)
    discriminant = 1.0 + 4.0 * b_value * reduced_pressure
    if not math.isfinite(discriminant) or discriminant <= 0.0:
        raise ValueError(f"virial EOS has no positive-density root: discriminant={discriminant}")
    molar_density = 2.0 * reduced_pressure / (1.0 + math.sqrt(discriminant))
    if not math.isfinite(molar_density) or molar_density <= 0.0:
        raise ValueError(f"virial EOS produced invalid molar density: {molar_density}")
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
        raise ValueError(f"virial EOS produced invalid molar Cv: {cv}")
    cp_minus_cv = temperature * pressure_temperature**2 / (
        molar_density**2 * pressure_density
    )
    cp = cv + cp_minus_cv
    sound_speed_squared = (
        pressure_density
        + temperature * pressure_temperature**2 / (molar_density**2 * cv)
    ) / molar_mass
    if not math.isfinite(cp) or cp <= 0.0 or not math.isfinite(sound_speed_squared) or sound_speed_squared <= 0.0:
        raise ValueError("virial EOS produced invalid heat capacity or sound speed")
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


def a2dyn_cp_t_virial_sound_speed(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    pressure_pa: float,
) -> float:
    """Sound speed from the registered temperature-Cp and second-virial EOS."""

    return a2dyn_thermodynamic_state(mole_fractions, temperature_k, pressure_pa)["sound_speed_m_s"]


def _a2dyn_ideal_sound_speed_from_heat_capacity(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    heat_capacity: Mapping[str, float],
) -> float:
    molar_mass = sum(
        mole_fractions[gas] * GAS_MOLAR_MASS_KG_MOL[gas] for gas in GAS_MOLAR_MASS_KG_MOL
    )
    cp = heat_capacity["cp_molar_j_mol_k"]
    cv = heat_capacity["cv_molar_j_mol_k"]
    speed_squared = cp / cv * R_GAS_J_MOL_K * temperature_k / molar_mass
    if not math.isfinite(speed_squared) or speed_squared <= 0.0:
        raise ValueError("temperature-dependent ideal EOS produced invalid sound speed")
    return math.sqrt(speed_squared)


def _validate_a2dyn_temperature(temperature_k: float) -> float:
    temperature = float(temperature_k)
    lower, upper = A2DYN_TEMPERATURE_RANGE_K
    if not math.isfinite(temperature) or not lower <= temperature <= upper:
        raise ValueError(
            f"A2DYN temperature must be within [{lower}, {upper}] K, got {temperature}"
        )
    return temperature


def _validate_a2dyn_pressure(pressure_pa: float) -> float:
    pressure = float(pressure_pa)
    _, upper = A2DYN_OPERATIONAL_PRESSURE_RANGE_PA
    if not math.isfinite(pressure) or pressure < 0.0 or pressure > upper:
        raise ValueError(f"A2DYN pressure must be within [0, {upper}] Pa, got {pressure}")
    return pressure


def _virial_pair_parameters(gas_i: str, gas_j: str) -> tuple[float, float, float]:
    if gas_i not in A2DYN_VIRIAL_CRITICAL_PROPERTIES or gas_j not in A2DYN_VIRIAL_CRITICAL_PROPERTIES:
        raise ValueError(f"unsupported virial pair: {gas_i}-{gas_j}")
    if gas_i == gas_j:
        return A2DYN_VIRIAL_CRITICAL_PROPERTIES[gas_i]
    critical_i = A2DYN_VIRIAL_CRITICAL_PROPERTIES[gas_i]
    critical_j = A2DYN_VIRIAL_CRITICAL_PROPERTIES[gas_j]
    critical_pressure = (
        0.5 * (critical_i[1] ** (2.0 / 3.0) + critical_j[1] ** (2.0 / 3.0))
    ) ** (3.0 / 2.0)
    return (
        math.sqrt(critical_i[0] * critical_j[0]),
        critical_pressure,
        0.5 * (critical_i[2] + critical_j[2]),
    )


def _tsonopoulos_pair_terms(
    gas_i: str,
    gas_j: str,
    temperature_k: float,
) -> tuple[float, float, float]:
    critical_temperature, critical_pressure, acentric_factor = _virial_pair_parameters(gas_i, gas_j)
    reduced_temperature = temperature_k / critical_temperature
    if not math.isfinite(reduced_temperature) or reduced_temperature <= 0.0:
        raise ValueError("virial reduced temperature must be finite and positive")
    coefficients = (
        (0, 0.1445 + acentric_factor * 0.0637),
        (1, -0.3300),
        (2, -0.1385 + acentric_factor * 0.3310),
        (3, -0.0121 - acentric_factor * 0.4230),
        (8, -0.000607 - acentric_factor * 0.0080),
    )
    dimensionless = sum(coefficient * reduced_temperature ** (-power) for power, coefficient in coefficients)
    first_derivative = sum(
        -power * coefficient * reduced_temperature ** (-power) / temperature_k
        for power, coefficient in coefficients
        if power > 0
    )
    second_derivative = sum(
        power * (power + 1.0) * coefficient * reduced_temperature ** (-power) / temperature_k**2
        for power, coefficient in coefficients
        if power > 0
    )
    scale = R_GAS_J_MOL_K * critical_temperature / critical_pressure
    return scale * dimensionless, scale * first_derivative, scale * second_derivative


def wms_thermal_conductivity(mole_fractions: Mapping[str, float]) -> float:
    _validate_mole_fractions(mole_fractions)
    conductivity = 0.0
    for gas_i, fraction_i in mole_fractions.items():
        denominator = sum(
            mole_fractions[gas_j] * _wilke_phi(gas_i, gas_j) for gas_j in GAS_MOLAR_MASS_KG_MOL
        )
        if denominator <= 0.0:
            raise ValueError(f"invalid WMS denominator for {gas_i}")
        conductivity += fraction_i * GAS_THERMAL_CONDUCTIVITY_W_M_K[gas_i] / denominator
    return conductivity


def thermal_conductivity_voltage(
    conductivity_w_m_k: float,
    *,
    baseline_v: float = TCS_BASELINE_V,
    response_v_per_w_m_k: float = TCS_RESPONSE_V_PER_W_M_K,
    reference_conductivity_w_m_k: float = TCS_REFERENCE_CONDUCTIVITY_W_M_K,
) -> float:
    if not math.isfinite(conductivity_w_m_k) or conductivity_w_m_k <= 0.0:
        raise ValueError("conductivity_w_m_k must be finite and positive")
    if not math.isfinite(baseline_v):
        raise ValueError("baseline_v must be finite")
    if not math.isfinite(response_v_per_w_m_k) or response_v_per_w_m_k <= 0.0:
        raise ValueError("response_v_per_w_m_k must be finite and positive")
    if not math.isfinite(reference_conductivity_w_m_k) or reference_conductivity_w_m_k <= 0.0:
        raise ValueError("reference_conductivity_w_m_k must be finite and positive")
    return baseline_v + response_v_per_w_m_k * (
        conductivity_w_m_k - reference_conductivity_w_m_k
    )


def ndir_co2_voltage(
    x_co2_pct: float,
    pressure_pa: float,
    temperature_k: float,
    *,
    effective_absorbance_per_co2_percent: float = NDIR_EFFECTIVE_ABSORBANCE_PER_CO2_PERCENT,
    baseline_v: float = NDIR_BASELINE_V,
) -> float:
    if not math.isfinite(x_co2_pct) or x_co2_pct < 0.0 or x_co2_pct > 100.0:
        raise ValueError("x_co2_pct must be finite and within [0,100]")
    if not math.isfinite(pressure_pa) or pressure_pa <= 0.0:
        raise ValueError("pressure_pa must be finite and positive")
    if not math.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("temperature_k must be finite and positive")
    if (
        not math.isfinite(effective_absorbance_per_co2_percent)
        or effective_absorbance_per_co2_percent <= 0.0
    ):
        raise ValueError("effective_absorbance_per_co2_percent must be finite and positive")
    if not math.isfinite(baseline_v) or baseline_v <= 0.0:
        raise ValueError("baseline_v must be finite and positive")
    absorbance = (
        effective_absorbance_per_co2_percent
        * x_co2_pct
        * pressure_pa
        / REFERENCE_PRESSURE_PA
        * REFERENCE_TEMPERATURE_K
        / temperature_k
    )
    return baseline_v * math.exp(-absorbance)


def _wilke_phi(gas_i: str, gas_j: str) -> float:
    viscosity_ratio = math.sqrt(GAS_VISCOSITY_PA_S[gas_i] / GAS_VISCOSITY_PA_S[gas_j])
    mass_ratio = (GAS_MOLAR_MASS_KG_MOL[gas_j] / GAS_MOLAR_MASS_KG_MOL[gas_i]) ** 0.25
    numerator = (1.0 + viscosity_ratio * mass_ratio) ** 2
    denominator = math.sqrt(8.0 * (1.0 + GAS_MOLAR_MASS_KG_MOL[gas_i] / GAS_MOLAR_MASS_KG_MOL[gas_j]))
    return numerator / denominator


def _first_order_response(time_s: np.ndarray, baseline: float, target: float, *, tau_s: float) -> np.ndarray:
    if tau_s <= 0.0:
        raise ValueError("tau_s must be positive")
    response = baseline + (target - baseline) * (1.0 - np.exp(-time_s / tau_s))
    return response.astype(np.float32).reshape(-1, 1)


def _validate_mole_fractions(mole_fractions: Mapping[str, float]) -> None:
    expected = set(GAS_MOLAR_MASS_KG_MOL)
    if set(mole_fractions) != expected:
        raise ValueError(f"mole fraction keys must be {sorted(expected)}, got {sorted(mole_fractions)}")
    values = tuple(mole_fractions[gas] for gas in GAS_MOLAR_MASS_KG_MOL)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ValueError(f"mole fractions must be finite and within [0,1], got {values}")
    if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"mole fractions must sum to 1, got {sum(values)}")
