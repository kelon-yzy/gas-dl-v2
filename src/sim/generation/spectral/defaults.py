from __future__ import annotations

from sim.generation.spectral.filters import NDIRFilter
from sim.generation.spectral.hitran_backend import HitranGasSpec, HitranGridSpec


DEFAULT_HITRAN_GAS_SPECS = (
    HitranGasSpec("CH4", "CH4", molecule_id=6, isotopologue_id=1),
    HitranGasSpec("CO2", "CO2", molecule_id=2, isotopologue_id=1),
    HitranGasSpec("H2O", "H2O", molecule_id=1, isotopologue_id=1),
)

DEFAULT_NDIR_FILTERS = {
    "ch4": NDIRFilter(channel="ch4", center_cm1=3030.0, fwhm_cm1=30.0),
    "co2": NDIRFilter(channel="co2", center_cm1=2347.0, fwhm_cm1=24.0),
}

DEFAULT_HITRAN_GRID_SPECS = {
    "ch4": HitranGridSpec(
        wavenumber_min_cm1=2960.0,
        wavenumber_max_cm1=3100.0,
        wavenumber_step_cm1=0.1,
        temperature_k=296.0,
        pressure_atm=1.0,
    ),
    "co2": HitranGridSpec(
        wavenumber_min_cm1=2280.0,
        wavenumber_max_cm1=2410.0,
        wavenumber_step_cm1=0.1,
        temperature_k=296.0,
        pressure_atm=1.0,
    ),
}


def get_default_ndir_filter(channel: str) -> NDIRFilter:
    try:
        return DEFAULT_NDIR_FILTERS[channel.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown NDIR channel: {channel!r}") from exc


def get_default_hitran_grid(channel: str) -> HitranGridSpec:
    try:
        return DEFAULT_HITRAN_GRID_SPECS[channel.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown HITRAN grid channel: {channel!r}") from exc
