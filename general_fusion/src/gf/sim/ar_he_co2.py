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
